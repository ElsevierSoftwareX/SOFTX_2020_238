/*
 * Copyright (C) 2009 Mireia Crispin Ortuzar <mcrispin@caltech.edu>,
 * Kipp Cannon <kipp.cannon@ligo.org>, Chad Hanna <chad.hanna@ligo.org>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */


/*
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


/*
 * stuff from the C library
 */


#include <complex.h>
#include <math.h>
#include <string.h>


/*
 *  stuff from gobject/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstadapter.h>
#include <gst/base/gstbasetransform.h>
#include <gstlal.h>
#include <gstlal_autochisq.h>


/*
 * stuff from GSL
 */


#include <gsl/gsl_complex.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>


/*
 * parameters
 */


#define CHI2_USES_REAL_ONLY FALSE


/*
 * ============================================================================
 *
 *                                 Utilities
 *
 * ============================================================================
 */


/*
 * return the number of autocorrleation vectors
 */


static unsigned autocorrelation_channels(const GSTLALAutoChiSq *element)
{
	return element->autocorrelation_matrix->size1;
}


/*
 * return the number of samples in the autocorrelation vectors
 */


static unsigned autocorrelation_length(const GSTLALAutoChiSq *element)
{
	return element->autocorrelation_matrix->size2;
}


/*
 * construct a buffer of zeros and push into adapter
 */


static int push_zeros(GSTLALAutoChiSq *element, unsigned samples)
{
	GstBuffer *zerobuf = gst_buffer_new_and_alloc(samples * autocorrelation_channels(element) * sizeof(complex double));
	if(!zerobuf) {
		GST_DEBUG_OBJECT(element, "failure allocating zero-pad buffer");
		return -1;
	}
	memset(GST_BUFFER_DATA(zerobuf), 0, GST_BUFFER_SIZE(zerobuf));
	gst_adapter_push(element->adapter, zerobuf);
	element->zeros_in_adapter += samples;
	return 0;
}


/*
 * set the metadata on an output buffer
 */


static void set_metadata(GSTLALAutoChiSq *element, GstBuffer *buf, guint64 outsamples, gboolean gap)
{
	GST_BUFFER_SIZE(buf) = outsamples * autocorrelation_channels(element) * sizeof(double);
	GST_BUFFER_OFFSET(buf) = element->next_out_offset;
	element->next_out_offset += outsamples;
	GST_BUFFER_OFFSET_END(buf) = element->next_out_offset;
	GST_BUFFER_TIMESTAMP(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET(buf) - element->offset0, GST_SECOND, element->rate);
	GST_BUFFER_DURATION(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET_END(buf) - element->offset0, GST_SECOND, element->rate) - GST_BUFFER_TIMESTAMP(buf);
	if(element->need_discont) {
		GST_BUFFER_FLAG_SET(buf, GST_BUFFER_FLAG_DISCONT);
		element->need_discont = FALSE;
	}
	if(gap)
		GST_BUFFER_FLAG_SET(buf, GST_BUFFER_FLAG_GAP);
	else
		GST_BUFFER_FLAG_UNSET(buf, GST_BUFFER_FLAG_GAP);
}


/*
 * the number of samples available in the adapter
 */


static guint64 get_available_samples(GSTLALAutoChiSq *element)
{
	return gst_adapter_available(element->adapter) / (autocorrelation_channels(element) * sizeof(complex double));
}


/*
 * compute autocorrelation norms --- the expectation value in noise
 */


static gsl_vector *compute_autocorrelation_norm(GSTLALAutoChiSq *element)
{
	gsl_vector *norm = gsl_vector_alloc(autocorrelation_channels(element));
	unsigned channel;

	for(channel = 0; channel < autocorrelation_channels(element); channel++) {
		gsl_vector_complex_view row = gsl_matrix_complex_row(element->autocorrelation_matrix, channel);
#if CHI2_USES_REAL_ONLY
		unsigned sample;
		double n = 0;
		for(sample = 0; sample < row.vector.size; sample++)
			n += 1 - pow(GSL_REAL(gsl_vector_complex_get(&row.vector, sample)), 2);
		gsl_vector_set(norm, channel, n);
#else
		gsl_vector_set(norm, channel, 2 * autocorrelation_length(element) - pow(gsl_blas_dznrm2(&row.vector), 2));
#endif
	}

	return norm;
}


/*
 * transform input samples to output samples using a time-domain algorithm
 */


static GstFlowReturn filter(GSTLALAutoChiSq *element, GstBuffer *outbuf)
{
	unsigned channels = autocorrelation_channels(element);
	unsigned available_length;
	unsigned output_length;
	const complex double *input;
	double *output;
	double *output_end;

	/*
	 * do we have enough data to do anything?
	 */

	available_length = get_available_samples(element);
	if(available_length < autocorrelation_length(element))
		return GST_BASE_TRANSFORM_FLOW_DROPPED;

	/*
	 * initialize pointers.  the +1 in output_length is because when
	 * there is 1 correlation-length of data in the adapter then we can
	 * produce 1 output sample, not 0.
	 */

	input = (complex double *) gst_adapter_peek(element->adapter, available_length * channels * sizeof(complex double));
	output = (double *) GST_BUFFER_DATA(outbuf);
	output_length = available_length - autocorrelation_length(element) + 1;
	output_end = output + output_length * channels;

	/*
	 * compute output samples.  note:  we assume that gsl_complex can
	 * be aliased to complex double.  I think it says somewhere in the
	 * documentation that this is true.
	 */

	/* check the autocorrelation matrix' packing */
	g_assert(element->autocorrelation_matrix->tda == autocorrelation_length(element));

	while(output < output_end) {
		const complex double *autocorrelation = (complex double *) gsl_matrix_complex_const_ptr(element->autocorrelation_matrix, 0, 0);
		unsigned channel;

		for(channel = 0; channel < channels; channel++) {
			/*
			 * start of input data block to be used for this
			 * output sample
			 */

			const complex double *indata = input;

			/*
			 * the input sample by which the autocorrelation
			 * funcion will be scaled
			 */

			complex double snr = input[((gint) autocorrelation_length(element) - 1 + element->latency) * channels];

			/*
			 * multiplying snr by this makes it real
			 */

			complex double invsnrphase = cexp(-I*carg(snr));

			/*
			 * end of this channel's row in the autocorrelation
			 * matrix
			 */

			const complex double *autocorrelation_end = autocorrelation + autocorrelation_length(element);

			/*
			 * \chi^{2} sum
			 */

			double chisq;

			/*
			 * compute \sum_{i} (A_{i} * \rho_{0} - \rho_{i})^{2}
			 */

			for(chisq = 0; autocorrelation < autocorrelation_end; autocorrelation++, indata += channels) {
				complex double z = (*autocorrelation * snr - *indata) * invsnrphase;
#if CHI2_USES_REAL_ONLY
				chisq += pow(creal(z), 2);
#else
				chisq += pow(creal(z), 2) + pow(cimag(z), 2);
#endif
			}

			/*
			 * record \chi^{2} sum, advance to next output sample
			 */

			*(output++) = chisq / gsl_vector_get(element->autocorrelation_norm, channel);

			/*
			 * advance to next input sample
			 */

			input++;
		}
	}

	/*
	 * flush the data from the adapter
	 */

	gst_adapter_flush(element->adapter, output_length * channels * sizeof(complex double));
	if(element->zeros_in_adapter > available_length - output_length)
		/*
		 * some trailing zeros have been flushed from the adapter
		 */

		element->zeros_in_adapter = available_length - output_length;

	/*
	 * set buffer metadata
	 */

	set_metadata(element, outbuf, output_length, FALSE);

	/*
	 * done
	 */

	return GST_FLOW_OK;
}


/*
 * ============================================================================
 *
 *                                  Signals
 *
 * ============================================================================
 */


enum gstlal_autochisq_signal {
	SIGNAL_RATE_CHANGED,
	NUM_SIGNALS
};


static guint signals[NUM_SIGNALS] = {0, };


static void rate_changed(GstElement *element, gint rate, void *data)
{
	/* FIXME;  send updated segment event downstream? */
}


/*
 * ============================================================================
 *
 *                           GStreamer Boiler Plate
 *
 * ============================================================================
 */


static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE(
	"sink",
	GST_PAD_SINK,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		"audio/x-raw-complex, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) [1, MAX], " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) 128"
	)
);


static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE(
	"src",
	GST_PAD_SRC,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		"audio/x-raw-float, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) [1, MAX], " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) 64"
	)
);


GST_BOILERPLATE(
	GSTLALAutoChiSq,
	gstlal_autochisq,
	GstBaseTransform,
	GST_TYPE_BASE_TRANSFORM
);


enum property {
	ARG_AUTOCORRELATION_MATRIX = 1,
	ARG_LATENCY
};


#define DEFAULT_LATENCY 0


/*
 * ============================================================================
 *
 *                     GstBaseTransform Method Overrides
 *
 * ============================================================================
 */


/*
 * get_unit_size()
 */


static gboolean get_unit_size(GstBaseTransform *trans, GstCaps *caps, guint *size)
{
	GstStructure *str;
	gint width;
	gint channels;

	str = gst_caps_get_structure(caps, 0);
	if(!gst_structure_get_int(str, "channels", &channels)) {
		GST_DEBUG_OBJECT(trans, "unable to parse channels from %" GST_PTR_FORMAT, caps);
		return FALSE;
	}
	if(!gst_structure_get_int(str, "width", &width)) {
		GST_DEBUG_OBJECT(trans, "unable to parse width from %" GST_PTR_FORMAT, caps);
		return FALSE;
	}

	*size = width / 8 * channels;

	return TRUE;
}


/*
 * transform_caps()
 */


static GstCaps *transform_caps(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps)
{
	guint n;

	caps = gst_caps_copy(caps);

	switch(direction) {
	case GST_PAD_SRC:
		/*
		 * sink pad's format is the same as the source pad's except
		 * the type is complex and each sample has twice the width
		 */

		for(n = 0; n < gst_caps_get_size(caps); n++) {
			gint width;
			GstStructure *s = gst_caps_get_structure(caps, n);
			gst_structure_set_name(s, "audio/x-raw-complex");
			gst_structure_get_int(s, "width", &width);
			gst_structure_set(s, "width", G_TYPE_INT, width * 2, NULL);
		}
		break;

	case GST_PAD_SINK:
		/*
		 * source pad's format is the same as the sink pad's except
		 * the type is float and each sample has half the width
		 */

		for(n = 0; n < gst_caps_get_size(caps); n++) {
			gint width;
			GstStructure *s = gst_caps_get_structure(caps, n);
			gst_structure_set_name(s, "audio/x-raw-float");
			gst_structure_get_int(s, "width", &width);
			gst_structure_set(s, "width", G_TYPE_INT, width / 2, NULL);
		}
		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		gst_caps_unref(caps);
		return GST_CAPS_NONE;
	}

	return caps;
}


/*
 * transform_size()
 */


static gboolean transform_size(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, guint size, GstCaps *othercaps, guint *othersize)
{
	GSTLALAutoChiSq *element = GSTLAL_AUTOCHISQ(trans);
	guint unit_size;
	guint other_unit_size;

	if(!get_unit_size(trans, caps, &unit_size))
		return FALSE;
	if(size % unit_size) {
		GST_DEBUG_OBJECT(element, "size not a multiple of %u", unit_size);
		return FALSE;
	}
	if(!get_unit_size(trans, othercaps, &other_unit_size))
		return FALSE;

	switch(direction) {
	case GST_PAD_SRC:
		/*
		 * just keep the sample count the same
		 */

		*othersize = (size / unit_size) * other_unit_size;
		break;

	case GST_PAD_SINK:
		/*
		 * upper bound of sample count on source pad is input
		 * sample count plus the number of samples in the adapter
		 * minus the length of the autocorrelation vectors (-1
		 * because if there's 1 autocorrelation-length of data then
		 * we can generate 1 sample, not 0)
		 */

		*othersize = size / unit_size + get_available_samples(element);
		if(*othersize >= autocorrelation_length(element))
			*othersize = (*othersize - autocorrelation_length(element) + 1) * other_unit_size;
		else
			*othersize = 0;
		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		return FALSE;
	}

	return TRUE;
}


/*
 * set_caps()
 */


static gboolean set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps)
{
	GSTLALAutoChiSq *element = GSTLAL_AUTOCHISQ(trans);
	GstStructure *s;
	gint rate;
	gint channels;
	gboolean success = TRUE;

	s = gst_caps_get_structure(incaps, 0);
	if(!gst_structure_get_int(s, "channels", &channels)) {
		GST_DEBUG_OBJECT(element, "unable to parse channels from %" GST_PTR_FORMAT, incaps);
		success = FALSE;
	}
	if(!gst_structure_get_int(s, "rate", &rate)) {
		GST_DEBUG_OBJECT(element, "unable to parse rate from %" GST_PTR_FORMAT, incaps);
		success = FALSE;
	}

	if(element->autocorrelation_matrix && (channels != (gint) autocorrelation_channels(element))) {
		GST_DEBUG_OBJECT(element, "channels != %d in %" GST_PTR_FORMAT, autocorrelation_channels(element), incaps);
		success = FALSE;
	}

	if(success) {
		if(rate != element->rate) {
			g_signal_emit(G_OBJECT(trans), signals[SIGNAL_RATE_CHANGED], 0, rate, NULL);
		}
		element->rate = rate;
	}

	return success;
}


/*
 * start()
 */


static gboolean start(GstBaseTransform *trans)
{
	GSTLALAutoChiSq *element = GSTLAL_AUTOCHISQ(trans);
	element->adapter = gst_adapter_new();
	element->zeros_in_adapter = 0;
	element->t0 = GST_CLOCK_TIME_NONE;
	element->offset0 = GST_BUFFER_OFFSET_NONE;
	element->next_in_offset = GST_BUFFER_OFFSET_NONE;
	element->next_out_offset = GST_BUFFER_OFFSET_NONE;
	element->need_discont = TRUE;
	return TRUE;
}


/*
 * stop()
 */


static gboolean stop(GstBaseTransform *trans)
{
	GSTLALAutoChiSq *element = GSTLAL_AUTOCHISQ(trans);
	g_object_unref(element->adapter);
	element->adapter = NULL;
	return TRUE;
}


/*
 * transform()
 */


static GstFlowReturn transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf)
{
	GSTLALAutoChiSq *element = GSTLAL_AUTOCHISQ(trans);
	guint64 length;
	GstFlowReturn result;

	/*
	 * wait for autocorrelation matrix
	 * FIXME:  add a way to get out of this loop
	 */

	g_mutex_lock(element->autocorrelation_lock);
	while(!element->autocorrelation_matrix)
		g_cond_wait(element->autocorrelation_available, element->autocorrelation_lock);

	/*
	 * check for discontinuity
	 *
	 * FIXME:  instead of reseting, flush/pad adapter as needed
	 */

	if(GST_BUFFER_OFFSET(inbuf) != element->next_in_offset || !GST_CLOCK_TIME_IS_VALID(element->t0)) {
		/*
		 * flush adapter
		 */

		gst_adapter_clear(element->adapter);
		element->zeros_in_adapter = 0;

		/*
		 * (re)sync timestamp and offset book-keeping
		 */

		element->t0 = GST_BUFFER_TIMESTAMP(inbuf);
		element->offset0 = GST_BUFFER_OFFSET(inbuf);
		element->next_out_offset = element->offset0 + autocorrelation_length(element) - 1 + element->latency;

		/*
		 * be sure to flag the next output buffer as a discontinuity
		 */

		element->need_discont = TRUE;
	}
	element->next_in_offset = GST_BUFFER_OFFSET_END(inbuf);

	/*
	 * gap logic
	 */

	length = GST_BUFFER_OFFSET_END(inbuf) - GST_BUFFER_OFFSET(inbuf);
	if(!GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP)) {
		/*
		 * input is not 0s.
		 */

		gst_buffer_ref(inbuf);	/* don't let the adapter free it */
		gst_adapter_push(element->adapter, inbuf);
		element->zeros_in_adapter = 0;
		result = filter(element, outbuf);
	} else if(element->zeros_in_adapter >= autocorrelation_length(element) - 1) {
		/*
		 * input is 0s and we are past the tail of the impulse
		 * response so output is all 0s.  output is a gap with the
		 * same number of samples as the input.
		 */

		memset(GST_BUFFER_DATA(outbuf), 0, GST_BUFFER_SIZE(outbuf));
		set_metadata(element, outbuf, GST_BUFFER_OFFSET_END(inbuf) - GST_BUFFER_OFFSET(inbuf), TRUE);
		result = GST_FLOW_OK;
	} else if(element->zeros_in_adapter + length < autocorrelation_length(element)) {
		/*
		 * input is 0s, we are not yet past the tail of the impulse
		 * response and the input is not long enough to change
		 * that.  push length 0s into the adapter and run normal
		 * filtering
		 */

		push_zeros(element, length);
		result = filter(element, outbuf);
	} else {
		/*
		 * input is 0s, we are not yet past the tail of the impulse
		 * response, but the input is long enough to push us past
		 * the end.  this code path also handles the case of the
		 * first buffer being a gap, in which case
		 * available_samples is 0
		 */

		guint64 non_zero_samples = get_available_samples(element) - element->zeros_in_adapter;

		/*
		 * push (correlation-length - zeros-in-adapter - 1) 0s into
		 * adapter to allow previous non-zero data to be finished
		 * off.  push_zeros() modifies zeros_in_adapter so update
		 * length first.
		 */

		length -= autocorrelation_length(element) - element->zeros_in_adapter - 1;
		push_zeros(element, autocorrelation_length(element) - element->zeros_in_adapter - 1);

		/*
		 * run normal filter code to finish off adapter's contents,
		 * and manually push buffer downstream.
		 */

		if(non_zero_samples) {
			GstPad *srcpad = GST_BASE_TRANSFORM_SRC_PAD(GST_BASE_TRANSFORM(element));
			GstBuffer *buf;

			result = gst_pad_alloc_buffer(srcpad, element->next_out_offset, non_zero_samples * autocorrelation_channels(element) * sizeof(double), GST_PAD_CAPS(srcpad), &buf);
			if(result != GST_FLOW_OK)
				goto done;
			result = filter(element, buf);
			g_assert(result == GST_FLOW_OK);
			result = gst_pad_push(srcpad, buf);
			if(result != GST_FLOW_OK)
				goto done;
		}

		/*
		 * remainder of input produces 0s in output.  make outbuf a
		 * gap whose size matches the remainder of the input gap
		 */

		GST_BUFFER_SIZE(outbuf) = length * autocorrelation_channels(element) * sizeof(double);
		memset(GST_BUFFER_DATA(outbuf), 0, GST_BUFFER_SIZE(outbuf));
		set_metadata(element, outbuf, length, TRUE);
	}

	/*
	 * done
	 */

done:
	g_mutex_unlock(element->autocorrelation_lock);
	return result;
}


/*
 * ============================================================================
 *
 *                          GObject Method Overrides
 *
 * ============================================================================
 */


/*
 * set_property()
 */


static void set_property(GObject *object, enum property prop_id, const GValue *value, GParamSpec *pspec)
{
	GSTLALAutoChiSq *element = GSTLAL_AUTOCHISQ(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_AUTOCORRELATION_MATRIX: {
		unsigned channels;
		g_mutex_lock(element->autocorrelation_lock);
		if(element->autocorrelation_matrix) {
			channels = autocorrelation_channels(element);
			gsl_matrix_complex_free(element->autocorrelation_matrix);
		} else
			channels = 0;
		element->autocorrelation_matrix = gstlal_gsl_matrix_complex_from_g_value_array(g_value_get_boxed(value));

		/*
		 * number of channels has changed, force a caps
		 * renegotiation
		 */

		if(autocorrelation_channels(element) != channels)
			gst_pad_set_caps(GST_BASE_TRANSFORM_SRC_PAD(GST_BASE_TRANSFORM(object)), NULL);

		/*
		 * compute norms
		 */

		if(element->autocorrelation_norm)
			gsl_vector_free(element->autocorrelation_norm);
		element->autocorrelation_norm = compute_autocorrelation_norm(element);

		/*
		 * check for invalid latency
		 */

		if(-element->latency >= (gint) autocorrelation_length(element)) {
			GST_ERROR_OBJECT(object, "invalid latency %ld, must be in (%u, 0]", element->latency, -(gint) autocorrelation_length(element));
			element->latency = -(autocorrelation_length(element) - 1);
		}

		/*
		 * signal availability of new autocorrelation vectors
		 */

		g_cond_broadcast(element->autocorrelation_available);
		g_mutex_unlock(element->autocorrelation_lock);
		break;
	}

	case ARG_LATENCY: {
		gint64 latency = g_value_get_int64(value);
		g_mutex_lock(element->autocorrelation_lock);
		if(element->autocorrelation_matrix && -latency >= (gint) autocorrelation_length(element))
			GST_ERROR_OBJECT(object, "invalid latency %ld, must be in (%u, 0]", latency, -(gint) autocorrelation_length(element));
		else
			element->latency = latency;
		g_mutex_unlock(element->autocorrelation_lock);
		break;
	}

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * get_property()
 */


static void get_property(GObject *object, enum property prop_id, GValue *value, GParamSpec *pspec)
{
	GSTLALAutoChiSq *element = GSTLAL_AUTOCHISQ(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_AUTOCORRELATION_MATRIX:
		g_mutex_lock(element->autocorrelation_lock);
		if(element->autocorrelation_matrix)
			g_value_take_boxed(value, gstlal_g_value_array_from_gsl_matrix_complex(element->autocorrelation_matrix));
		/* FIXME:  else? */
		g_mutex_unlock(element->autocorrelation_lock);
		break;

	case ARG_LATENCY:
		g_value_set_int64(value, element->latency);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * finalize()
 */


static void finalize(GObject *object)
{
	GSTLALAutoChiSq *element = GSTLAL_AUTOCHISQ(object);

	g_mutex_free(element->autocorrelation_lock);
	element->autocorrelation_lock = NULL;
	g_cond_free(element->autocorrelation_available);
	element->autocorrelation_available = NULL;
	if(element->autocorrelation_matrix) {
		gsl_matrix_complex_free(element->autocorrelation_matrix);
		element->autocorrelation_matrix = NULL;
	}
	if(element->autocorrelation_norm) {
		gsl_vector_free(element->autocorrelation_norm);
		element->autocorrelation_norm = NULL;
	}

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


/*
 * base_init()
 */


static void gstlal_autochisq_base_init(gpointer gclass)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(gclass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(gclass);

	gst_element_class_set_details_simple(
		element_class,
		"Autocorrelation \\chi^{2}",
		"Filter/Audio",
		"Computes the chisquared time series from a filter's autocorrelation",
		"Kipp Cannon <kipp.cannon@ligo.org>, Mireia Crispin Ortuzar <mcrispin@caltech.edu>, Chad Hanna <chad.hanna@ligo.org>"
	);

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));

	transform_class->get_unit_size = GST_DEBUG_FUNCPTR(get_unit_size);
	transform_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
	transform_class->transform = GST_DEBUG_FUNCPTR(transform);
	transform_class->transform_caps = GST_DEBUG_FUNCPTR(transform_caps);
	transform_class->transform_size = GST_DEBUG_FUNCPTR(transform_size);
	transform_class->start = GST_DEBUG_FUNCPTR(start);
	transform_class->stop = GST_DEBUG_FUNCPTR(stop);
}


/*
 * class_init()
 */


static void gstlal_autochisq_class_init(GSTLALAutoChiSqClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	klass->rate_changed = GST_DEBUG_FUNCPTR(rate_changed);

	g_object_class_install_property(
		gobject_class,
		ARG_AUTOCORRELATION_MATRIX,
		g_param_spec_value_array(
			"autocorrelation-matrix",
			"Autocorrelation Matrix",
			"Array of complex autocorrelation vectors.  Number of vectors (rows) in matrix sets number of channels.  All vectors must have the same length.",
			g_param_spec_value_array(
				"autocorrelation",
				"Autocorrelation",
				"Array of autocorrelation samples.",
				/* FIXME:  should be complex */
				g_param_spec_double(
					"sample",
					"Sample",
					"Autocorrelation sample",
					-G_MAXDOUBLE, G_MAXDOUBLE, 0.0,
					G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
				),
				G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
			),
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_LATENCY,
		g_param_spec_int64(
			"latency",
			"Latency",
			"Filter latency in samples.  Must be in (-autocorrelation length, 0].",
			G_MININT64, 0, DEFAULT_LATENCY,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);

	signals[SIGNAL_RATE_CHANGED] = g_signal_new(
		"rate-changed",
		G_TYPE_FROM_CLASS(klass),
		G_SIGNAL_RUN_FIRST,
		G_STRUCT_OFFSET(
			GSTLALAutoChiSqClass,
			rate_changed
		),
		NULL,
		NULL,
		g_cclosure_marshal_VOID__INT,
		G_TYPE_NONE,
		1,
		G_TYPE_INT
	);
}


/*
 * init()
 */


static void gstlal_autochisq_init(GSTLALAutoChiSq *filter, GSTLALAutoChiSqClass *kclass)
{
	filter->latency = DEFAULT_LATENCY;
	filter->adapter = NULL;
	filter->autocorrelation_lock = g_mutex_new();
	filter->autocorrelation_available = g_cond_new();
	filter->autocorrelation_matrix = NULL;
	filter->autocorrelation_norm = NULL;
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(filter), TRUE);
}

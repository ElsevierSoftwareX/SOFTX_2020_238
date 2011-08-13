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
#include <gst/base/gstbasetransform.h>
#include <gstlal.h>
#include <gstlal_autochisq.h>


/*
 * stuff from GSL
 */


#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>


/*
 * our own stuff
 */


#include <gstaudioadapter.h>


GST_DEBUG_CATEGORY(gstlal_autochisq_debug);


/*
 * ============================================================================
 *
 *                                 Parameters
 *
 * ============================================================================
 */


#define GST_CAT_DEFAULT gstlal_autochisq_debug
#define CHI2_USES_REAL_ONLY FALSE
#define DEFAULT_SNR_THRESH 0


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


static guint get_available_samples(GSTLALAutoChiSq *element)
{
	guint size;

	g_object_get(element->adapter, "size", &size, NULL);

	return size;
}


/*
 * compute autocorrelation norms --- the expectation value in noise.  must
 * be called with autocorrelation_matrix_lock held
 */


static gsl_vector *compute_autocorrelation_norm(GSTLALAutoChiSq *element)
{
	gsl_vector *norm;
	unsigned channel;

	if(element->autocorrelation_mask_matrix && (autocorrelation_channels(element) != element->autocorrelation_mask_matrix->size1 || autocorrelation_length(element) != element->autocorrelation_mask_matrix->size2)) {
		GST_ELEMENT_ERROR(element, STREAM, FAILED, ("array size mismatch"), ("autocorrelation matrix (%dx%d) and mask matrix (%dx%d) do not have the same size", autocorrelation_channels(element), autocorrelation_length(element), element->autocorrelation_mask_matrix->size1, element->autocorrelation_mask_matrix->size2));
		return NULL;
	}

	norm = gsl_vector_alloc(autocorrelation_channels(element));

	for(channel = 0; channel < autocorrelation_channels(element); channel++) {
		gsl_vector_complex_view row = gsl_matrix_complex_row(element->autocorrelation_matrix, channel);
		gsl_vector_int_view mask;
		unsigned sample;
		double n = 0;
		
		if(element->autocorrelation_mask_matrix)
			mask = gsl_matrix_int_row(element->autocorrelation_mask_matrix, channel);
		for(sample = 0; sample < row.vector.size; sample++) {
			if(element->autocorrelation_mask_matrix && !gsl_vector_int_get(&mask.vector, sample))
				continue;
#if CHI2_USES_REAL_ONLY
			n += 1 - pow(GSL_REAL(gsl_vector_complex_get(&row.vector, sample)), 2);
#else
			n += 2 - gsl_complex_abs2(gsl_vector_complex_get(&row.vector, sample));
#endif
		}
		gsl_vector_set(norm, channel, n);
	}

	return norm;
}


/*
 * transform input samples to output samples using a time-domain algorithm
 */


static unsigned filter(GSTLALAutoChiSq *element, GstBuffer *outbuf)
{
	unsigned channels = autocorrelation_channels(element);
	unsigned zeros_in_adapter;
	unsigned available_length;
	unsigned output_length = 0;
	complex double *inputbuf, *input;
	double *output;
	double *output_end;

	/*
	 * do we have enough data to do anything?
	 */

	available_length = get_available_samples(element);
	if(available_length < autocorrelation_length(element))
		goto done;
	zeros_in_adapter = gst_audioadapter_tail_gap_length(element->adapter);
	if(zeros_in_adapter >= autocorrelation_length(element))
		/* the +1 is because when there is 1 correlation-length of
		 * 0s at the tail of the adapter the output will contain 1
		 * sample equal to 0, not 0 samples. */
		available_length -= zeros_in_adapter - autocorrelation_length(element) + 1;

	/* the +1 is because when there is 1 correlation-length of data in
	 * the adapter then we can produce 1 output sample, not 0. */
	output_length = available_length - autocorrelation_length(element) + 1;
	if(!output_length)
		goto done;

	/*
	 * initialize pointers.
	 */

	input = inputbuf = g_malloc(available_length * channels * sizeof(*input));
	gst_audioadapter_copy(element->adapter, inputbuf, available_length, NULL, NULL);
	output = (double *) GST_BUFFER_DATA(outbuf);
	output_end = output + output_length * channels;

	/*
	 * safety checks
	 */

	g_assert(element->autocorrelation_matrix->tda == autocorrelation_length(element));
	if(element->autocorrelation_mask_matrix) {
		g_assert(autocorrelation_channels(element) == element->autocorrelation_mask_matrix->size1);
		g_assert(autocorrelation_length(element) == element->autocorrelation_mask_matrix->size2);
		g_assert(element->autocorrelation_mask_matrix->tda == autocorrelation_length(element));
	}

	/*
	 * compute output samples.  note:  we assume that gsl_complex can
	 * be aliased to complex double.  I think it says somewhere in the
	 * documentation that this is true.
	 */

	while(output < output_end) {
		const complex double *autocorrelation = (const complex double *) gsl_matrix_complex_const_ptr(element->autocorrelation_matrix, 0, 0);
		const int *autocorrelation_mask = element->autocorrelation_mask_matrix ? (const int *) gsl_matrix_int_const_ptr(element->autocorrelation_mask_matrix, 0, 0) : NULL;
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

			if(cabs(snr) >= element->snr_thresh) {
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

				if(autocorrelation_mask) {
					for(chisq = 0; autocorrelation < autocorrelation_end; autocorrelation++, autocorrelation_mask++, indata += channels) {
						complex double z;
						if(!*autocorrelation_mask)
							continue;
						z = (*autocorrelation * snr - *indata) * invsnrphase;
#if CHI2_USES_REAL_ONLY
						chisq += pow(creal(z), 2);
#else
						chisq += pow(creal(z), 2) + pow(cimag(z), 2);
#endif
					}
				} else {
					for(chisq = 0; autocorrelation < autocorrelation_end; autocorrelation++, indata += channels) {
						complex double z = (*autocorrelation * snr - *indata) * invsnrphase;
#if CHI2_USES_REAL_ONLY
						chisq += pow(creal(z), 2);
#else
						chisq += pow(creal(z), 2) + pow(cimag(z), 2);
#endif
					}
				}

				/*
				 * record \chi^{2} sum, advance to next output sample
				 */

				*output = chisq / gsl_vector_get(element->autocorrelation_norm, channel);
			} else {
				autocorrelation += autocorrelation_length(element);
				if(autocorrelation_mask)
					autocorrelation_mask += autocorrelation_length(element);
				*output = 0;
			}

			/*
			 * advance to next sample
			 */

			output++;
			input++;
		}
	}

	/*
	 * flush the data from the adapter
	 */

	g_free(inputbuf);
	gst_audioadapter_flush(element->adapter, output_length);

done:
	/*
	 * set buffer metadata
	 */

	set_metadata(element, outbuf, output_length, FALSE);

	/*
	 * done
	 */

	return output_length;
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
	GST_BASE_TRANSFORM_SINK_NAME,
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
	GST_BASE_TRANSFORM_SRC_NAME,
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
	ARG_AUTOCORRELATION_MASK_MATRIX,
	ARG_LATENCY,
	ARG_SNR_THRESH
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
	gboolean success = TRUE;

	if(!get_unit_size(trans, caps, &unit_size))
		return FALSE;
	if(size % unit_size) {
		GST_DEBUG_OBJECT(element, "size not a multiple of %u", unit_size);
		return FALSE;
	}
	if(!get_unit_size(trans, othercaps, &other_unit_size))
		return FALSE;

	/*
	 * wait for autocorrelation matrix
	 */

	g_mutex_lock(element->autocorrelation_lock);
	while(!element->autocorrelation_matrix) {
		GST_DEBUG_OBJECT(element, "autocorrelation matrix not available, waiting ...");
		g_cond_wait(element->autocorrelation_available, element->autocorrelation_lock);
		if(GST_STATE(GST_ELEMENT(trans)) == GST_STATE_NULL) {
			GST_DEBUG_OBJECT(element, "element now in null state, abandoning wait for autocorrelation matrix");
			success = FALSE;
			goto done;
		}
	}

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
		 * minus the length of the autocorrelation vectors +1
		 * (because if there's 1 autocorrelation-length of data
		 * then we can generate 1 sample, not 0)
		 */

		*othersize = size / unit_size + get_available_samples(element);
		if(*othersize >= autocorrelation_length(element))
			*othersize = (*othersize - autocorrelation_length(element) + 1) * other_unit_size;
		else
			*othersize = 0;
		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		success = FALSE;
		break;
	}

done:
	g_mutex_unlock(element->autocorrelation_lock);
	return success;
}


/*
 * set_caps()
 */


static gboolean set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps)
{
	GSTLALAutoChiSq *element = GSTLAL_AUTOCHISQ(trans);
	GstStructure *s;
	gint channels;
	gint width;
	gint rate;
	gboolean success = TRUE;

	s = gst_caps_get_structure(incaps, 0);
	if(!gst_structure_get_int(s, "channels", &channels)) {
		GST_DEBUG_OBJECT(element, "unable to parse channels from %" GST_PTR_FORMAT, incaps);
		success = FALSE;
	}
	if(!gst_structure_get_int(s, "width", &width)) {
		GST_DEBUG_OBJECT(element, "unable to parse width from %" GST_PTR_FORMAT, incaps);
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
		gint old_rate = element->rate;
		element->rate = rate;
		if(element->rate != old_rate)
			g_signal_emit(G_OBJECT(trans), signals[SIGNAL_RATE_CHANGED], 0, element->rate, NULL);
		g_object_set(element->adapter, "unit-size", width / 8 * channels, NULL);
	}

	return success;
}


/*
 * start()
 */


static gboolean start(GstBaseTransform *trans)
{
	GSTLALAutoChiSq *element = GSTLAL_AUTOCHISQ(trans);
	element->adapter = g_object_new(GST_TYPE_AUDIOADAPTER, NULL);
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
	gboolean history_is_gap, input_is_gap;
	guint zeros_in_adapter;
	guint output_length;
	GstFlowReturn result = GST_FLOW_OK;

	/*
	 * wait for autocorrelation matrix
	 */

	g_mutex_lock(element->autocorrelation_lock);
	while(!element->autocorrelation_matrix) {
		GST_DEBUG_OBJECT(element, "autocorrelation matrix not available, waiting ...");
		g_cond_wait(element->autocorrelation_available, element->autocorrelation_lock);
		if(GST_STATE(GST_ELEMENT(trans)) == GST_STATE_NULL) {
			GST_DEBUG_OBJECT(element, "element now in null state, abandoning wait for autocorrelation matrix");
			result = GST_FLOW_WRONG_STATE;
			goto done;
		}
	}

	/*
	 * recompute norms if needed
	 */

	if(!element->autocorrelation_norm) {
		element->autocorrelation_norm = compute_autocorrelation_norm(element);
		if(!element->autocorrelation_norm) {
			GST_DEBUG_OBJECT(element, "failed to compute autocorrelation norms");
			result = GST_FLOW_ERROR;
			goto done;
		}
	}

	/*
	 * check for discontinuity
	 */

	if(GST_BUFFER_IS_DISCONT(inbuf) || GST_BUFFER_OFFSET(inbuf) != element->next_in_offset || !GST_CLOCK_TIME_IS_VALID(element->t0)) {
		/*
		 * flush adapter
		 */

		gst_audioadapter_clear(element->adapter);

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
	} else
		g_assert(GST_BUFFER_TIMESTAMP(inbuf) == gst_audioadapter_expected_timestamp(element->adapter));
	element->next_in_offset = GST_BUFFER_OFFSET_END(inbuf);

	/*
	 * build output buffer(s)
	 */

	input_is_gap = GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP);
	history_is_gap = gst_audioadapter_is_gap(element->adapter);

	output_length = get_available_samples(element) + GST_BUFFER_OFFSET_END(inbuf) - GST_BUFFER_OFFSET(inbuf);
	if(output_length >= autocorrelation_length(element))
		output_length -= autocorrelation_length(element) - 1;
	else
		output_length = 0;
	g_assert(output_length == GST_BUFFER_SIZE(outbuf) / (autocorrelation_channels(element) * sizeof(double)));

	gst_buffer_ref(inbuf);	/* don't let calling code free buffer */
	gst_audioadapter_push(element->adapter, inbuf);

	zeros_in_adapter = gst_audioadapter_tail_gap_length(element->adapter);

	if(!input_is_gap) {
		/*
		 * because the history that remains in the adapter cannot
		 * be large enough to compute even 1 sample, the output is
		 * a single non-gap buffer whether or not the history is
		 * known to be all 0s
		 */

		guint samples = filter(element, outbuf);
		g_assert(output_length == samples);
	} else if(history_is_gap) {
		/*
		 * all data in hand is known to be 0s, the output is a
		 * single gap buffer
		 */

		gst_audioadapter_flush(element->adapter, output_length);
		memset(GST_BUFFER_DATA(outbuf), 0, GST_BUFFER_SIZE(outbuf));
		set_metadata(element, outbuf, output_length, TRUE);
	} else if(zeros_in_adapter < autocorrelation_length(element)) {
		/*
		 * the history contains some amount of non-zero data and
		 * whatever zeros are at the end of the history combined
		 * with the input, which is known to be all 0, together do
		 * not add up to enough to produce any 0-valued output
		 * samples, so the output is a single non-gap buffer
		 */

		guint samples = filter(element, outbuf);
		g_assert(output_length == samples);
	} else {
		/*
		 * the tailing zeros in the history combined with the input
		 * data are together large enough to yield 0s in the
		 * output. the output will be two buffers, a non-gap buffer
		 * to finish off the non-zero data in the history followed
		 * by a gap buffer.
		 */

		GstPad *srcpad = GST_BASE_TRANSFORM_SRC_PAD(GST_BASE_TRANSFORM(element));
		guint gap_length = zeros_in_adapter - autocorrelation_length(element) + 1;
		guint samples;
		GstBuffer *buf;

		g_mutex_unlock(element->autocorrelation_lock);
		result = gst_pad_alloc_buffer(srcpad, element->next_out_offset, (output_length - gap_length) * autocorrelation_channels(element) * sizeof(double), GST_PAD_CAPS(srcpad), &buf);
		g_mutex_lock(element->autocorrelation_lock);
		if(result != GST_FLOW_OK)
			goto done;
		samples = filter(element, buf);
		g_assert(output_length - gap_length == samples);

		g_mutex_unlock(element->autocorrelation_lock);
		result = gst_pad_push(srcpad, buf);
		g_mutex_lock(element->autocorrelation_lock);
		if(result != GST_FLOW_OK)
			goto done;

		GST_BUFFER_SIZE(outbuf) = gap_length * autocorrelation_channels(element) * sizeof(double);
		memset(GST_BUFFER_DATA(outbuf), 0, GST_BUFFER_SIZE(outbuf));
		set_metadata(element, outbuf, gap_length, TRUE);
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

		if(autocorrelation_channels(element) != channels) {
			/* FIXME:  is this right? */
			gst_pad_set_caps(GST_BASE_TRANSFORM_SRC_PAD(GST_BASE_TRANSFORM(object)), NULL);
			/*gst_base_transform_reconfigure(GST_BASE_TRANSFORM(object));*/
		}

		/*
		 * induce norms to be recomputed
		 */

		if(element->autocorrelation_norm) {
			gsl_vector_free(element->autocorrelation_norm);
			element->autocorrelation_norm = NULL;
		}

		/*
		 * check for invalid latency
		 */

		if(-element->latency >= (gint) autocorrelation_length(element)) {
			GST_ERROR_OBJECT(object, "invalid latency %" G_GINT64_FORMAT ", must be in (%u, 0]", element->latency, -(gint) autocorrelation_length(element));
			element->latency = -(autocorrelation_length(element) - 1);
		}

		/*
		 * signal availability of new autocorrelation vectors
		 */

		g_cond_broadcast(element->autocorrelation_available);
		g_mutex_unlock(element->autocorrelation_lock);
		break;
	}

	case ARG_AUTOCORRELATION_MASK_MATRIX:
		g_mutex_lock(element->autocorrelation_lock);

		if(element->autocorrelation_mask_matrix)
			gsl_matrix_int_free(element->autocorrelation_mask_matrix);
		element->autocorrelation_mask_matrix = gstlal_gsl_matrix_int_from_g_value_array(g_value_get_boxed(value));

		/*
		 * induce norms to be recomputed
		 */

		if(element->autocorrelation_norm) {
			gsl_vector_free(element->autocorrelation_norm);
			element->autocorrelation_norm = NULL;
		}

		g_mutex_unlock(element->autocorrelation_lock);
		break;

	case ARG_LATENCY: {
		gint64 latency = g_value_get_int64(value);
		g_mutex_lock(element->autocorrelation_lock);
		if(element->autocorrelation_matrix && -latency >= (gint) autocorrelation_length(element))
			GST_ERROR_OBJECT(object, "invalid latency %" G_GINT64_FORMAT ", must be in (%u, 0]", latency, -(gint) autocorrelation_length(element));
		else
			element->latency = latency;
		g_mutex_unlock(element->autocorrelation_lock);
		break;
	}

	case ARG_SNR_THRESH:
		element->snr_thresh = g_value_get_double(value);
		break;

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

	case ARG_AUTOCORRELATION_MASK_MATRIX:
		g_mutex_lock(element->autocorrelation_lock);
		if(element->autocorrelation_mask_matrix)
			g_value_take_boxed(value, gstlal_g_value_array_from_gsl_matrix_int(element->autocorrelation_mask_matrix));
		/* FIXME:  else? */
		g_mutex_unlock(element->autocorrelation_lock);
		break;

	case ARG_LATENCY:
		g_value_set_int64(value, element->latency);
		break;

	case ARG_SNR_THRESH:
		g_value_set_double(value, element->snr_thresh);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * dispose()
 */


static void dispose(GObject *object)
{
	GSTLALAutoChiSq *element = GSTLAL_AUTOCHISQ(object);

	/*
	 * wake up any threads that are waiting for the autocorrelation
	 * matrix to become available;  since we are being finalized the
	 * element state should be NULL causing those threads to bail out
	 */

	g_mutex_lock(element->autocorrelation_lock);
	g_cond_broadcast(element->autocorrelation_available);
	g_mutex_unlock(element->autocorrelation_lock);
}


/*
 * finalize()
 */


static void finalize(GObject *object)
{
	GSTLALAutoChiSq *element = GSTLAL_AUTOCHISQ(object);

	/*
	 * free resources
	 */

	g_mutex_free(element->autocorrelation_lock);
	element->autocorrelation_lock = NULL;
	g_cond_free(element->autocorrelation_available);
	element->autocorrelation_available = NULL;
	if(element->autocorrelation_matrix) {
		gsl_matrix_complex_free(element->autocorrelation_matrix);
		element->autocorrelation_matrix = NULL;
	}
	if(element->autocorrelation_mask_matrix) {
		gsl_matrix_int_free(element->autocorrelation_mask_matrix);
		element->autocorrelation_mask_matrix = NULL;
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
	gobject_class->dispose = GST_DEBUG_FUNCPTR(dispose);
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
		ARG_AUTOCORRELATION_MASK_MATRIX,
		g_param_spec_value_array(
			"autocorrelation-mask-matrix",
			"Autocorrelation Mask Matrix",
			"Array of integer mask vectors.  Matrix must be the same size as the autocorrelation matrix.  Only autocorrelation vector samples corresponding to non-zero samples in these vectors will be used to construct the \\chi^{2} statistic.  If this matrix is not supplied, all autocorrelation samples are used.",
			g_param_spec_value_array(
				"autocorrelation-mask",
				"Autocorrelation Mask",
				"Array of autocorrelation mask samples.",
				g_param_spec_int(
					"sample",
					"Sample",
					"Autocorrelation mask sample",
					G_MININT, G_MAXINT, 0.0,
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
	g_object_class_install_property(
		gobject_class,
		ARG_SNR_THRESH,
		g_param_spec_double(
			"snr-thresh",
			"SNR Threshold",
			"SNR Threshold that determines a trigger.",
			0, G_MAXDOUBLE, DEFAULT_SNR_THRESH,
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


static void gstlal_autochisq_init(GSTLALAutoChiSq *filter, GSTLALAutoChiSqClass *klass)
{
	filter->latency = DEFAULT_LATENCY;
	filter->adapter = NULL;
	filter->autocorrelation_lock = g_mutex_new();
	filter->autocorrelation_available = g_cond_new();
	filter->autocorrelation_matrix = NULL;
	filter->autocorrelation_mask_matrix = NULL;
	filter->autocorrelation_norm = NULL;
	filter->snr_thresh = DEFAULT_SNR_THRESH;
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(filter), TRUE);
}

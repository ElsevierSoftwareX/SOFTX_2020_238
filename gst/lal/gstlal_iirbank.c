/*
 * Copyright (C) 2010 Shaun Hooper <hoopes01@student.uwa.edu.au>
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
#include <string.h>


/*
 *  stuff from gobject/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstadapter.h>
#include <gst/base/gstbasetransform.h>
#include <gstlal.h>
#include <gstlal_iirbank.h>


/*
 * stuff from FFTW and GSL
 */


#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>


/*
 * ============================================================================
 *
 *                                 Utilities
 *
 * ============================================================================
 */

/*
 * return the number of IIR channels
 */

static unsigned iir_channels(const GSTLALIIRBank *element)
{
  return 2; /* FIXME 2 is the number of output channels */
}


/*
 * construct a buffer of zeros and push into adapter
 */


static int push_zeros(GSTLALIIRBank *element, unsigned samples)
{
	GstBuffer *zerobuf = gst_buffer_new_and_alloc(samples * sizeof(double));
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


static void set_metadata(GSTLALIIRBank *element, GstBuffer *buf, guint64 outsamples, gboolean gap)
{
        GST_BUFFER_SIZE(buf) = outsamples * iir_channels(element) * sizeof(double);
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


static guint64 get_available_samples(GSTLALIIRBank *element)
{
	return gst_adapter_available(element->adapter) / sizeof(double);
}


/*
 * transform input samples to output samples using a time-domain algorithm
 */


static GstFlowReturn filter(GSTLALIIRBank *element, GstBuffer *outbuf)
{
	unsigned available_length;
	unsigned output_length;
	double *input;
	complex double *output;
	int dmax, dmin;


	/*
	 * how much data is available?
	 */

	gsl_vector_int_minmax(element->delay, &dmin, &dmax); 
	available_length = get_available_samples(element);
	output_length = available_length - (dmax - dmin);

	/*
	 * wrap the adapter's contents in a GSL vector view.
	 */

	input = (double *) gst_adapter_peek(element->adapter, available_length * sizeof(double));

	/*
	 * wrap output buffer in a complex double array.
	 */

	output = (complex double *) GST_BUFFER_DATA(outbuf);

	
	/*
	 * output produced?
	 */

	if(!output_length)
		return GST_BASE_TRANSFORM_FLOW_DROPPED;

	/*
	 * flush the data from the adapter
	 */

	gst_adapter_flush(element->adapter, output_length * sizeof(double));
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


enum gstlal_iirbank_signal {
	SIGNAL_RATE_CHANGED,
	NUM_SIGNALS
};


static guint signals[NUM_SIGNALS] = {0, };


static void rate_changed(GstElement *element, gint rate, void *data)
{
	/* FIXME: send updated segment downstream?  because latency now
	 * means something different */
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
		"audio/x-raw-float, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) 1, " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) 64"
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
	GSTLALIIRBank,
	gstlal_iirbank,
	GstBaseTransform,
	GST_TYPE_BASE_TRANSFORM
);


enum property {
	ARG_IIR_A1 = 1,
	ARG_IIR_B0,
	ARG_IIR_DELAY
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
	gint channels;

	str = gst_caps_get_structure(caps, 0);
	if(!gst_structure_get_int(str, "channels", &channels)) {
		GST_DEBUG_OBJECT(trans, "unable to parse channels from %" GST_PTR_FORMAT, caps);
		return FALSE;
	}

	*size = sizeof(double) * channels;

	return TRUE;
}


/*
 * transform_caps()
 */


static GstCaps *transform_caps(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps)
{
	GSTLALIIRBank *element = GSTLAL_IIRBANK(trans);
	guint n;

	caps = gst_caps_copy(caps);

	switch(direction) {
	case GST_PAD_SRC:
		/*
		 * sink pad's format is the same as the source pad's except
		 * it must have only 1 channel
		 */

		for(n = 0; n < gst_caps_get_size(caps); n++)
			gst_structure_set(gst_caps_get_structure(caps, n), "channels", G_TYPE_INT, 1, NULL);
		break;

	case GST_PAD_SINK:
		/*
		 * source pad's format is the same as the sink pad's except
		 * it can have any number of channels or, if the size of
		 * the IIR matrix is known, the number of channels must
		 * equal to twice the number of template IIR outputs.
		 */


		g_mutex_lock(element->iir_matrix_lock);
		for(n = 0; n < gst_caps_get_size(caps); n++) {
			if(element->delay)
			        gst_structure_set(gst_caps_get_structure(caps, n), "channels", G_TYPE_INT, iir_channels(element), NULL);
			else
				gst_structure_set(gst_caps_get_structure(caps, n), "channels", GST_TYPE_INT_RANGE, 1, G_MAXINT, NULL);
		}
		g_mutex_unlock(element->iir_matrix_lock);
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
	GSTLALIIRBank *element = GSTLAL_IIRBANK(trans);
	guint unit_size;
	guint other_unit_size;
	int dmin, dmax;

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
		 * minus the impulse response length of the filters (-1
		 * because if there's 1 impulse response of data then we
		 * can generate 1 sample, not 0)
		 */
	  
        	g_assert(element->delay != NULL);
         	gsl_vector_int_minmax(element->delay, &dmin, &dmax); 
		
		*othersize = size / unit_size + get_available_samples(element);

		if((gint) *othersize > dmax - dmin)
		        *othersize = (*othersize - (dmax-dmin)) * other_unit_size;
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
	GSTLALIIRBank *element = GSTLAL_IIRBANK(trans);
	GstStructure *s;
	gint rate;
	gint channels;
	gboolean success = TRUE;

	s = gst_caps_get_structure(outcaps, 0);
	if(!gst_structure_get_int(s, "channels", &channels)) {
		GST_DEBUG_OBJECT(element, "unable to parse channels from %" GST_PTR_FORMAT, outcaps);
		success = FALSE;
	} else if(!gst_structure_get_int(s, "rate", &rate)) {
		GST_DEBUG_OBJECT(element, "unable to parse rate from %" GST_PTR_FORMAT, outcaps);
		success = FALSE;
	} else if(element->delay && (channels != (gint) iir_channels(element))) {
		GST_DEBUG_OBJECT(element, "channels != %d in %" GST_PTR_FORMAT, iir_channels(element), outcaps);
		success = FALSE;
	}

	if(success) {
		if(rate != element->rate)
			g_signal_emit(G_OBJECT(trans), signals[SIGNAL_RATE_CHANGED], 0, rate, NULL);
		element->rate = rate;
	}

	return success;
}


/*
 * start()
 */


static gboolean start(GstBaseTransform *trans)
{
	GSTLALIIRBank *element = GSTLAL_IIRBANK(trans);
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
	GSTLALIIRBank *element = GSTLAL_IIRBANK(trans);
	g_object_unref(element->adapter);
	element->adapter = NULL;
	return TRUE;
}


/*
 * transform()
 */


static GstFlowReturn transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf)
{
	guint64 length;
	GSTLALIIRBank *element = GSTLAL_IIRBANK(trans);
	GstFlowReturn result;

	/*
	 * wait for IIR matrix
	 * FIXME:  add a way to get out of this loop
	 */

	g_mutex_lock(element->iir_matrix_lock);
	while(!element->delay || !element->a1 || !element->b0)
		g_cond_wait(element->iir_matrix_available, element->iir_matrix_lock);

	g_assert(element->b0->size == element->delay->size);
	g_assert(element->a1->size == element->delay->size);

	if(!element->ylast)
	        element->ylast = gsl_vector_complex_calloc(element->a1->size);
	else
	        g_assert(element->ylast->size == element->delay->size);

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
		push_zeros(element, gsl_vector_int_max(element->delay));
		
                /*
		 * (re)sync timestamp and offset book-keeping
		 */

		element->t0 = GST_BUFFER_TIMESTAMP(inbuf);
		element->offset0 = GST_BUFFER_OFFSET(inbuf);
		element->next_out_offset = element->offset0;

		/*
		 * be sure to flag the next output buffer as a discontinuity
		 */

		element->need_discont = TRUE;
	}
	element->next_in_offset = GST_BUFFER_OFFSET_END(inbuf);

	/*
	 * gap logic
	 *
	 * FIXME:  gap handling is busted
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
	} else if(TRUE) {
		/*
		 * input is 0s, we are not yet past the tail of the impulse
		 * response and the input is not long enough to change
		 * that.  push length 0s into the adapter and run normal
		 * filtering
		 */

		push_zeros(element, length);
		result = filter(element, outbuf);
	}

	/*
	 * done
	 */

	g_mutex_unlock(element->iir_matrix_lock);
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
	GSTLALIIRBank *element = GSTLAL_IIRBANK(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_IIR_A1:
		g_mutex_lock(element->iir_matrix_lock);
		if(element->a1)
		        gsl_vector_complex_free(element->a1);

		element->a1 = gstlal_gsl_vector_complex_from_g_value_array(g_value_get_boxed(value));

		/*
		 * signal change of IIR coeffs
		 */

		g_cond_broadcast(element->iir_matrix_available);
		g_mutex_unlock(element->iir_matrix_lock);
		break;

	case ARG_IIR_B0:
		g_mutex_lock(element->iir_matrix_lock);
		if(element->b0)
		        gsl_vector_complex_free(element->b0);

		element->b0 = gstlal_gsl_vector_complex_from_g_value_array(g_value_get_boxed(value));

		/*
		 * signal change of IIR coeffs
		 */

		g_cond_broadcast(element->iir_matrix_available);
		g_mutex_unlock(element->iir_matrix_lock);
		break;

	case ARG_IIR_DELAY:
		g_mutex_lock(element->iir_matrix_lock);
		if(element->delay)
		        gsl_vector_int_free(element->delay);

		element->delay = gstlal_gsl_vector_int_from_g_value_array(g_value_get_boxed(value));

		/*
		 * signal change of IIR delays
		 */

		g_cond_broadcast(element->iir_matrix_available);
		g_mutex_unlock(element->iir_matrix_lock);
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
	GSTLALIIRBank *element = GSTLAL_IIRBANK(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_IIR_A1:
		g_mutex_lock(element->iir_matrix_lock);
		if(element->a1)
			g_value_take_boxed(value, gstlal_g_value_array_from_gsl_vector_complex(element->a1));
		/* FIXME:  else? */
		g_mutex_unlock(element->iir_matrix_lock);
		break;

	case ARG_IIR_B0:
		g_mutex_lock(element->iir_matrix_lock);
		if(element->b0)
			g_value_take_boxed(value, gstlal_g_value_array_from_gsl_vector_complex(element->b0));
		/* FIXME:  else? */
		g_mutex_unlock(element->iir_matrix_lock);
		break;

	case ARG_IIR_DELAY:
		g_mutex_lock(element->iir_matrix_lock);
		if(element->delay)
			g_value_take_boxed(value, gstlal_g_value_array_from_gsl_vector_int(element->delay));
		/* FIXME:  else? */
		g_mutex_unlock(element->iir_matrix_lock);
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
	GSTLALIIRBank *element = GSTLAL_IIRBANK(object);

	g_mutex_free(element->iir_matrix_lock);
	element->iir_matrix_lock = NULL;
	g_cond_free(element->iir_matrix_available);
	element->iir_matrix_available = NULL;
	if(element->a1) {
		gsl_vector_complex_free(element->a1);
		element->a1 = NULL;
	}
	if(element->b0) {
		gsl_vector_complex_free(element->b0);
		element->b0 = NULL;
	}
	if(element->delay) {
		gsl_vector_int_free(element->delay);
		element->delay = NULL;
	}
	if(element->ylast) {
		gsl_vector_complex_free(element->ylast);
		element->ylast = NULL;
	}

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


/*
 * base_init()
 */


static void gstlal_iirbank_base_init(gpointer gclass)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(gclass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(gclass);

	gst_element_class_set_details_simple(element_class, "IIR Filter Bank", "Filter/Audio", "Projects a single audio channel onto a bank of IIR filters to produce a multi-channel output", "Shaun Hooper <hoopes01@student.uwa.edu.au>");

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


static void gstlal_iirbank_class_init(GSTLALIIRBankClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	klass->rate_changed = GST_DEBUG_FUNCPTR(rate_changed);

	g_object_class_install_property(
		gobject_class,
		ARG_IIR_A1,
		g_param_spec_value_array(
			"a1",
			"IIR feedback coefficient",
			"Array of first order IIR filter feedback coefficients.  All filters must have the same length.",
			g_param_spec_double(
				"coefficient",
				"Coefficient",
				"Feedback coefficient",
				-G_MAXDOUBLE, G_MAXDOUBLE, 0.0,
				G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
			),
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);

	g_object_class_install_property(
		gobject_class,
		ARG_IIR_B0,
		g_param_spec_value_array(
			"b0",
			"IIR coefficient",
			"Array of first order IIR filter coefficients.  All filters must have the same length.",
			g_param_spec_double(
				"coefficient",
				"Coefficient",
				"Current input sample coefficient",
				-G_MAXDOUBLE, G_MAXDOUBLE, 0.0,
				G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
			),
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);

	g_object_class_install_property(
		gobject_class,
		ARG_IIR_DELAY,
		g_param_spec_value_array(
			"delay",
			"Delay for IIR filter",
			"Array of delays for first order IIR filters.  All filters must have the same length.",
			g_param_spec_int(
				"delay",
				"Delay",
				"Delay for IIR filter",
				0, G_MAXINT, 0,
				G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
			),
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);

	signals[SIGNAL_RATE_CHANGED] = g_signal_new(
		"rate-changed",
		G_TYPE_FROM_CLASS(klass),
		G_SIGNAL_RUN_FIRST,
		G_STRUCT_OFFSET(
			GSTLALIIRBankClass,
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


static void gstlal_iirbank_init(GSTLALIIRBank *filter, GSTLALIIRBankClass *kclass)
{
	filter->adapter = NULL;
	filter->iir_matrix_lock = g_mutex_new();
	filter->iir_matrix_available = g_cond_new();
	filter->a1 = NULL;
	filter->b0 = NULL;
	filter->delay = NULL;
	filter->ylast = NULL;
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(filter), TRUE);
}

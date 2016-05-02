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
#include <time.h>


/*
 *  stuff from gobject/gstreamer
*/


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstadapter.h>
#include <gst/base/gstbasetransform.h>
#include <gstlal/gstlal.h>
#include <gstlal_iirbank.h>


/*
 * stuff from FFTW and GSL
 */


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
	if(element->a1)
		return 2 * element->a1->size1;
	return 0;
}


/*
 * construct a buffer of zeros and push into adapter
 */


static int push_zeros(GSTLALIIRBank *element, unsigned samples)
{
	GstBuffer *zerobuf = gst_buffer_new_allocate(NULL, samples * (GST_AUDIO_INFO_WIDTH(&(element->audio_info)) / 8), NULL);
	if(!zerobuf) {
		GST_DEBUG_OBJECT(element, "failure allocating zero-pad buffer");
		return -1;
	}
	gst_adapter_push(element->adapter, zerobuf);
	element->zeros_in_adapter += samples;
	return 0;
}


/*
 * set the metadata on an output buffer. 
 */


static void set_metadata(GSTLALIIRBank *element, GstBuffer *buf, guint64 outsamples, gboolean isgap)
{
        GST_BUFFER_OFFSET(buf) = element->next_out_offset;
	element->next_out_offset += outsamples;
	GST_BUFFER_OFFSET_END(buf) = element->next_out_offset;
	GST_BUFFER_TIMESTAMP(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET(buf) - element->offset0, GST_SECOND, GST_AUDIO_INFO_RATE(&(element->audio_info)));
	GST_BUFFER_DURATION(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET_END(buf) - element->offset0, GST_SECOND, GST_AUDIO_INFO_RATE(&(element->audio_info))) - GST_BUFFER_TIMESTAMP(buf);
	if(G_UNLIKELY(element->need_discont)) {
		GST_BUFFER_FLAG_SET(buf, GST_BUFFER_FLAG_DISCONT);
		element->need_discont = FALSE;
	}
	if(isgap)
		GST_BUFFER_FLAG_SET(buf, GST_BUFFER_FLAG_GAP);
	else
		GST_BUFFER_FLAG_UNSET(buf, GST_BUFFER_FLAG_GAP);
}


/*
 * the number of samples available in the adapter
 */


static guint64 get_available_samples(GSTLALIIRBank *element)
{
	return gst_adapter_available(element->adapter) / (GST_AUDIO_INFO_WIDTH(&(element->audio_info)) / 8);
}


/*
 * transform input samples to output samples using a time-domain algorithm
 */


static GstFlowReturn filter_d(GSTLALIIRBank *element, GstMapInfo *map, gboolean isgap, unsigned *out_length)
{
	unsigned available_length;
	unsigned output_length;
	const double * restrict input;
	complex double * restrict output;
	int dmax, dmin;
	complex double * restrict y, * restrict a1, * restrict b0;
	int * restrict d;

	y = (complex double *) gsl_matrix_complex_ptr(element->y, 0, 0);
	a1 = (complex double *) gsl_matrix_complex_ptr(element->a1, 0, 0);
	b0 = (complex double *) gsl_matrix_complex_ptr(element->b0, 0, 0);
	d = gsl_matrix_int_ptr(element->delay, 0, 0);

	/*
	 * how much data is available?
	 */

	gsl_matrix_int_minmax(element->delay, &dmin, &dmax);
	dmin = 0;
	available_length = get_available_samples(element);
	output_length = *out_length = available_length - (dmax - dmin);

	if(!output_length)
		return GST_BASE_TRANSFORM_FLOW_DROPPED;

	/*
	 * wrap the adapter's contents in an array
	 */

	input = (const double *) gst_adapter_map(element->adapter, available_length * sizeof(double));

	/*
	 * wrap output buffer in a complex double array.
	 */

	output = (complex double *) map->data;
	g_assert(output_length * iir_channels(element) / 2 * sizeof(complex double) <= map->size);

	memset(output, 0, output_length * iir_channels(element) / 2 * sizeof(*output));

	uint num_templates, num_filters;
	num_templates = element->a1->size1;
	num_filters = element->a1->size2;

	uint iter_tplts, iter_flts, iter_length; 

	for (iter_tplts = 0; iter_tplts < num_templates; iter_tplts++) {
		for (iter_flts = 0; iter_flts < num_filters; iter_flts++) {
			complex double ytemp = *y;
			const double *in = &input[dmax -*d];
			complex double *out = output;

			for(iter_length = 0; iter_length < output_length; iter_length++) { /* sample # */
				ytemp = *a1 * ytemp + *b0 * ((double)(*(in + iter_length))) + 1e-20;
				*out += ytemp;
				out += num_templates;
			}
			*y = ytemp;
			y++;
			a1++;
			b0++;
			d++;
		}
		output++;
	}
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
	 * done
	 */

	return GST_FLOW_OK;
}


/*
 * transform input samples to output samples using a time-domain algorithm -- single precision
 */


static GstFlowReturn filter_s(GSTLALIIRBank *element, GstMapInfo *map, gboolean isgap, unsigned *out_length)
{
	unsigned available_length;
	unsigned output_length;
	const float * restrict input;
	complex float * restrict output;
	int dmax, dmin;
	complex double * restrict y, * restrict a1, * restrict b0;
	int * restrict d;
	y = (complex double *) gsl_matrix_complex_ptr(element->y, 0, 0);
	a1 = (complex double *) gsl_matrix_complex_ptr(element->a1, 0, 0);
	b0 = (complex double *) gsl_matrix_complex_ptr(element->b0, 0, 0);
	d = gsl_matrix_int_ptr(element->delay, 0, 0);

	/*
	 * how much data is available?
	 */

	gsl_matrix_int_minmax(element->delay, &dmin, &dmax);
	dmin = 0;
	available_length = get_available_samples(element);
	output_length = *out_length = available_length - (dmax - dmin);

	if(!output_length)
		return GST_BASE_TRANSFORM_FLOW_DROPPED;

	/*
	 * wrap the adapter's contents in an array
	 */

	input = (const float *) gst_adapter_map(element->adapter, available_length * sizeof(float));

	/*
	 * wrap output buffer in a complex double array.
	 */

	output = (complex float *) map->data;
	g_assert(output_length * iir_channels(element) / 2 * sizeof(complex float) <= map->size);

	memset(output, 0, output_length * iir_channels(element) / 2 * sizeof(*output));

	uint num_templates, num_filters;
	num_templates = element->a1->size1;
	num_filters = element->a1->size2;

	uint iter_tplts, iter_flts, iter_length; 

	for (iter_tplts = 0; iter_tplts < num_templates; iter_tplts++) {
		for (iter_flts = 0; iter_flts < num_filters; iter_flts++) {
			complex double ytemp = *y;
			const float *in = &input[dmax -*d];
			complex float *out = output;

			for(iter_length = 0; iter_length < output_length; iter_length++) { /* sample # */
				ytemp = *a1 * ytemp + *b0 * ((double)(*(in + iter_length))) + 1e-20;
				*out += ((float)creal(ytemp) + (float)cimag(ytemp) * _Complex_I);
				out += num_templates;
			}
			*y = ytemp;
			y++;
			a1++;
			b0++;
			d++;
		}
		output++;
	}

	/*
	 * flush the data from the adapter
	 */

	g_assert(gst_adapter_available(element->adapter) >= (output_length * GST_AUDIO_INFO_WIDTH(&(element->audio_info)) / 8));
	gst_adapter_flush(element->adapter, output_length * (GST_AUDIO_INFO_WIDTH(&(element->audio_info)) / 8));
	if(element->zeros_in_adapter > available_length - output_length)
		/*
		 * some trailing zeros have been flushed from the adapter
		 */

		element->zeros_in_adapter = available_length - output_length;

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
		"audio/x-raw, " \
		"rate = " GST_AUDIO_RATE_RANGE ", " \
		"channels = (int) 1, " \
		"format = (string) {" GST_AUDIO_NE(F32) ", " GST_AUDIO_NE(F64) "}, " \
		"layout = (string) interleaved"
	)
);


static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SRC_NAME,
	GST_PAD_SRC,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		"audio/x-raw, " \
		"rate = " GST_AUDIO_RATE_RANGE ", " \
		"channels = " GST_AUDIO_CHANNELS_RANGE ", " \
		"format = (string) {" GST_AUDIO_NE(F32) ", " GST_AUDIO_NE(F64) "}, " \
		"layout = (string) interleaved"
	)
);


G_DEFINE_TYPE(
	GSTLALIIRBank,
	gstlal_iirbank,
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


static gboolean get_unit_size(GstBaseTransform *trans, GstCaps *caps, gsize *size)
{
	GstAudioInfo info;
	gboolean success = TRUE;

	success &= gst_audio_info_from_caps(&info, caps);

	if(success)
		*size = GST_AUDIO_INFO_BPF(&info);
	else
		GST_WARNING_OBJECT(trans, "unable to parse channels from %" GST_PTR_FORMAT, caps);

	return success;
}



/*
 * transform_caps()
 */


static GstCaps *transform_caps(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, GstCaps *filter)
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


		g_mutex_lock(&element->iir_matrix_lock);
		for(n = 0; n < gst_caps_get_size(caps); n++) {
			if(element->delay)
			        gst_structure_set(gst_caps_get_structure(caps, n), "channels", G_TYPE_INT, iir_channels(element), NULL);
			else
				gst_structure_set(gst_caps_get_structure(caps, n), "channels", GST_TYPE_INT_RANGE, 1, G_MAXINT, NULL);
		}
		g_mutex_unlock(&element->iir_matrix_lock);
		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		gst_caps_unref(caps);
		return GST_CAPS_NONE;
	}

	if(filter) {
		GstCaps *result = gst_caps_intersect(caps, filter);
		gst_caps_unref(caps);
		caps = result;
	}

	return caps;
}


/*
 * transform_size()
 */


static gboolean transform_size(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, gsize size, GstCaps *othercaps, gsize *othersize)
{
	GSTLALIIRBank *element = GSTLAL_IIRBANK(trans);
	gsize unit_size;
	gsize other_unit_size;
	int dmin, dmax;

	if(!get_unit_size(trans, caps, &unit_size))
		return FALSE;
	if(size % unit_size) {
		GST_DEBUG_OBJECT(element, "size not a multiple of %zu", unit_size);
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

		g_mutex_lock(&element->iir_matrix_lock);
		while(!element->delay || !element->a1 || !element->b0)
			g_cond_wait(&element->iir_matrix_available, &element->iir_matrix_lock);

         	gsl_matrix_int_minmax(element->delay, &dmin, &dmax);
		dmin = 0;
		g_mutex_unlock(&element->iir_matrix_lock);

		*othersize = size / unit_size + get_available_samples(element);

		if((gint) *othersize > dmax - dmin && get_available_samples(element) > 0)
		        *othersize = (*othersize - (dmax-dmin)) * other_unit_size;
		else
			/* FIXME: Is this correct? */
			*othersize *= other_unit_size ;
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
	GstAudioInfo info;
	gboolean success = TRUE;

	success &= gst_audio_info_from_caps(&info, outcaps);

	if(success && element->delay && (GST_AUDIO_INFO_CHANNELS(&info) != (gint) iir_channels(element))) {
		GST_DEBUG_OBJECT(element, "channels != %d in %" GST_PTR_FORMAT, iir_channels(element), outcaps);
		success = FALSE;
	}

	if(success) {
		gint old_rate = GST_AUDIO_INFO_IS_VALID(&(element->audio_info)) ? GST_AUDIO_INFO_RATE(&(element->audio_info)) : 0;
		if(!gst_audio_info_is_equal(&info, &(element->audio_info))) {
			gst_adapter_clear(element->adapter);
			element->zeros_in_adapter = 0;
			element->t0 = GST_CLOCK_TIME_NONE;	/* force discont */
		}
		element->audio_info = info;
		if(GST_AUDIO_INFO_RATE(&(element->audio_info)) != old_rate)
			g_signal_emit(G_OBJECT(trans), signals[SIGNAL_RATE_CHANGED], 0, GST_AUDIO_INFO_RATE(&(element->audio_info)), NULL);

	} else
		GST_ERROR_OBJECT(element, "unable to parse and/or accept caps %" GST_PTR_FORMAT, outcaps);

	return success;
}


/*
 * start()
 */


static gboolean start(GstBaseTransform *trans)
{
	GSTLALIIRBank *element = GSTLAL_IIRBANK(trans);
	element->zeros_in_adapter = 0;
	element->t0 = GST_CLOCK_TIME_NONE;
	element->offset0 = GST_BUFFER_OFFSET_NONE;
	element->next_in_offset = GST_BUFFER_OFFSET_NONE;
	element->next_out_offset = GST_BUFFER_OFFSET_NONE;
	element->need_discont = TRUE;
	return TRUE;
}


/*
 * transform()
 */


static GstFlowReturn transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf)
{
	GSTLALIIRBank *element = GSTLAL_IIRBANK(trans);
	GstMapInfo mapinfo;
	unsigned output_length;
	GstFlowReturn result = GST_FLOW_ERROR;

	/*
	 * wait for IIR matrix
	 * FIXME:  add a way to get out of this loop
	 */

	g_mutex_lock(&element->iir_matrix_lock);
	while(!element->delay || !element->a1 || !element->b0)
		g_cond_wait(&element->iir_matrix_available, &element->iir_matrix_lock);

	g_assert(element->b0->size1 == element->delay->size1);
	g_assert(element->a1->size1 == element->delay->size1);
	g_assert(element->b0->size2 == element->delay->size2);
	g_assert(element->a1->size2 == element->delay->size2);

	if(!element->y)
	        element->y = gsl_matrix_complex_calloc(element->a1->size1, element->a1->size2);
	else if(element->y->size1 != element->delay->size1 || element->y->size2 != element->delay->size2 ) {
		gsl_matrix_complex_free(element->y);
	        element->y = gsl_matrix_complex_calloc(element->a1->size1, element->a1->size2);
	}

	/*
	 * check for discontinuity
	 *
	 * FIXME:  instead of reseting, flush/pad adapter as needed
	 */

	if(GST_BUFFER_IS_DISCONT(inbuf) || GST_BUFFER_OFFSET(inbuf) != element->next_in_offset || !GST_CLOCK_TIME_IS_VALID(element->t0)) {
		int dmin, dmax;

		/*
		 * flush adapter. Erase contents of adaptor. Push dmax - dmin zeros, so the IIR filter bank can start filtering from t0.
		 */
		gst_adapter_clear(element->adapter);
		element->zeros_in_adapter = 0;

		gsl_matrix_int_minmax(element->delay, &dmin, &dmax);
		dmin = 0;
		push_zeros(element, dmax-dmin);

                /*
		 * (re)sync timestamp and offset book-keeping. Set t0 and offset0 to be the timestamp and offset of the inbuf. Next_out_offset is to set the offset of the outbuf.
		 */

		element->t0 = GST_BUFFER_TIMESTAMP(inbuf);
		element->offset0 = GST_BUFFER_OFFSET(inbuf);
		element->next_out_offset = element->offset0 + dmin;

		/*
		 * be sure to flag the next output buffer as a discontinuity, because it is the first output buffer.
		 */

		element->need_discont = TRUE;
	}
	element->next_in_offset = GST_BUFFER_OFFSET_END(inbuf);

	/*
	 * gap logic
	 *
	 * FIXME:  gap handling is busted
	 */

	gst_buffer_map(outbuf, &mapinfo, GST_MAP_WRITE);
	if(!GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP)) {
		/*
		 * input is not 0s
		 */

		gst_buffer_ref(inbuf);	/* don't let the adapter free it */
		gst_adapter_push(element->adapter, inbuf);
		element->zeros_in_adapter = 0;
		if(GST_AUDIO_INFO_WIDTH(&(element->audio_info)) == 64)
			result = filter_d(element, &mapinfo, FALSE, &output_length);
		else if(GST_AUDIO_INFO_WIDTH(&(element->audio_info)) == 32)
			result = filter_s(element, &mapinfo, FALSE, &output_length);
	} else {
		/*
		 * input is 0s, FIXME here. Make dependent on decay rate.
		 */

		push_zeros(element, GST_BUFFER_OFFSET_END(inbuf) - GST_BUFFER_OFFSET(inbuf));
		if(GST_AUDIO_INFO_WIDTH(&(element->audio_info)) == 64)
			result = filter_d(element, &mapinfo, TRUE, &output_length);
		else if(GST_AUDIO_INFO_WIDTH(&(element->audio_info)) == 32)
			result = filter_s(element, &mapinfo, TRUE, &output_length);
	}
	gst_buffer_unmap(outbuf, &mapinfo);

	/*
	 * set buffer metadata
	 */

	set_metadata(element, outbuf, output_length, FALSE);

	/*
	 * done
	 */

	g_mutex_unlock(&element->iir_matrix_lock);
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
		g_mutex_lock(&element->iir_matrix_lock);
		if(element->a1)
		        gsl_matrix_complex_free(element->a1);

		element->a1 = gstlal_gsl_matrix_complex_from_g_value_array(g_value_get_boxed(value));

		/*
		 * signal change of IIR coeffs
		 */

		g_cond_broadcast(&element->iir_matrix_available);
		g_mutex_unlock(&element->iir_matrix_lock);
		break;

	case ARG_IIR_B0:
		g_mutex_lock(&element->iir_matrix_lock);
		if(element->b0)
		        gsl_matrix_complex_free(element->b0);

		element->b0 = gstlal_gsl_matrix_complex_from_g_value_array(g_value_get_boxed(value));

		/*
		 * signal change of IIR coeffs
		 */

		g_cond_broadcast(&element->iir_matrix_available);
		g_mutex_unlock(&element->iir_matrix_lock);
		break;

	case ARG_IIR_DELAY: {
		int dmin, dmax;
		int dmin_new, dmax_new;

		g_mutex_lock(&element->iir_matrix_lock);
		if(element->delay) {
			gsl_matrix_int_minmax(element->delay, &dmin, &dmax);
			dmin = 0;
			gsl_matrix_int_free(element->delay);
		} else
			dmin = dmax = 0;

		element->delay = gstlal_gsl_matrix_int_from_g_value_array(g_value_get_boxed(value));
		gsl_matrix_int_minmax(element->delay, &dmin_new, &dmax_new);
		dmin_new = 0;

		if(dmax_new-dmin_new > dmax-dmin)
			push_zeros(element, dmax_new-dmin_new-(dmax-dmin));

		/*
		 * signal change of IIR delays
		 */

		g_cond_broadcast(&element->iir_matrix_available);
		g_mutex_unlock(&element->iir_matrix_lock);
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
	GSTLALIIRBank *element = GSTLAL_IIRBANK(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_IIR_A1:
		g_mutex_lock(&element->iir_matrix_lock);
		if(element->a1)
			g_value_take_boxed(value, gstlal_g_value_array_from_gsl_matrix_complex(element->a1));
		/* FIXME:  else? */
		g_mutex_unlock(&element->iir_matrix_lock);
		break;

	case ARG_IIR_B0:
		g_mutex_lock(&element->iir_matrix_lock);
		if(element->b0)
			g_value_take_boxed(value, gstlal_g_value_array_from_gsl_matrix_complex(element->b0));
		/* FIXME:  else? */
		g_mutex_unlock(&element->iir_matrix_lock);
		break;

	case ARG_IIR_DELAY:
		g_mutex_lock(&element->iir_matrix_lock);
		if(element->delay)
			g_value_take_boxed(value, gstlal_g_value_array_from_gsl_matrix_int(element->delay));
		/* FIXME:  else? */
		g_mutex_unlock(&element->iir_matrix_lock);
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

	g_mutex_clear(&element->iir_matrix_lock);
	g_cond_clear(&element->iir_matrix_available);
	if(element->a1) {
		gsl_matrix_complex_free(element->a1);
		element->a1 = NULL;
	}
	if(element->b0) {
		gsl_matrix_complex_free(element->b0);
		element->b0 = NULL;
	}
	if(element->delay) {
		gsl_matrix_int_free(element->delay);
		element->delay = NULL;
	}
	if(element->y) {
		gsl_matrix_complex_free(element->y);
		element->y = NULL;
	}
	g_object_unref(element->adapter);
	element->adapter = NULL;

	G_OBJECT_CLASS(gstlal_iirbank_parent_class)->finalize(object);
}


/*
 * class_init()
 */


static void gstlal_iirbank_class_init(GSTLALIIRBankClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(klass);

	gst_element_class_set_details_simple(element_class, "IIR Filter Bank", "Filter/Audio", "Projects a single audio channel onto a bank of IIR filters to produce a multi-channel output", "Shaun Hooper <hoopes01@student.uwa.edu.au>");

	transform_class->get_unit_size = GST_DEBUG_FUNCPTR(get_unit_size);
	transform_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
	transform_class->transform = GST_DEBUG_FUNCPTR(transform);
	transform_class->transform_caps = GST_DEBUG_FUNCPTR(transform_caps);
	transform_class->transform_size = GST_DEBUG_FUNCPTR(transform_size);
	transform_class->start = GST_DEBUG_FUNCPTR(start);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));

	klass->rate_changed = GST_DEBUG_FUNCPTR(rate_changed);

	g_object_class_install_property(
		gobject_class,
		ARG_IIR_A1,
		g_param_spec_value_array(
			"a1-matrix",
			"Matric of IIR feedback coefficients",
			"A Matrix of first order IIR filter feedback coefficients. Each row represents a different IIR bank.",
			g_param_spec_value_array(
				"a1",
				"IIR bank feedback coefficients",
				"A parallel bank of first order IIR filter feedback coefficients",
				g_param_spec_double(
					"coefficient",
					"Coefficient",
					"Feedback coefficient",
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
		ARG_IIR_B0,
		g_param_spec_value_array(
			"b0-matrix",
			"Matrix of IIR bank feedforward coefficients",
			"Array of first order IIR filter coefficients. Each row represents a different IIR bank.",
			g_param_spec_value_array(
				"b0",
				"IIR bank of feedforward coefficients",
				"A parallel bank of first order IIR filter feedforward coefficents",
				g_param_spec_double(
					 "coefficient",
					 "Coefficient",
					 "Current input sample coefficient",
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
		ARG_IIR_DELAY,
		g_param_spec_value_array(
			"delay-matrix",
			"Matrix of delays for IIR filter bank",
			"Matrix of delays for first order IIR filters.  All filters must have the same length.",
			g_param_spec_value_array(
				"delay",
				"delays for IIR bank",
				"A parallel bank of first order IIR filter delays",
				g_param_spec_int(
					"delay",
					"Delay",
					"Delay for IIR filter",
					0, G_MAXINT, 0,
					G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
				),
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


static void gstlal_iirbank_init(GSTLALIIRBank *filter)
{
	filter->adapter = gst_adapter_new();
	g_mutex_init(&filter->iir_matrix_lock);
	g_cond_init(&filter->iir_matrix_available);
	filter->a1 = NULL;
	filter->b0 = NULL;
	filter->delay = NULL;
	filter->y = NULL;
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(filter), TRUE);
}

/*
 * An element to do averages and things like averages.
 *
 * Copyright (C) 2010  Kipp Cannon, Chad Hanna
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


#include <string.h>
#include <math.h>

/*
 *  stuff from gobject/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstadapter.h>
#include <gst/base/gstbasetransform.h>


/*
 * stuff from gstlal
 */


#include <gstlal_peak.h>


/*
 * ============================================================================
 *
 *                                 Utilities
 *
 * ============================================================================
 */


static int max_over_n_process(GSTLALPeak *element, guint64 available_length, guint64 output_length, double *in, double *out)
{
	guint64 i,j,k;
	double currentvalue = 0.;
	guint64 channels = (guint64) element->channels;

	if (!element->max) element->max = (double *) calloc(channels, sizeof(double));
	if (!element->lastmax) element->lastmax = (guint64 *) calloc(channels, sizeof(guint64));

	/* pre compute the max for the first sample */
	guint64 offset = available_length - output_length;
	for (j = 0; j < channels; j++) {
		element->max[j] = pow(in[(offset - element->n + 1) * channels + j], element->moment);
		element->lastmax[j] = offset - element->n;
		for(k = 0; k < element->n-1; k++) {
			if(k > offset) break;
			currentvalue = pow(in[(offset - k) * channels + j], element->moment);
			if (currentvalue >= element->max[j] ) {
				element->max[j] = currentvalue;
				element->lastmax[j] = offset-k;
			}
		}
	}
	
	for(i = 0; i < output_length; i++) {
		/* How far to look back in the input */
		offset = available_length - output_length + i;
		for (j = 0; j < channels; j++) {

			/* Check to see if the current value exceeds the previous maximum */
			currentvalue = pow(in[(offset) * channels + j], element->moment);
			if (currentvalue >= element->max[j]) {
				element->max[j] = currentvalue;
				element->lastmax[j] = offset;
			}

			/* Check to see if the last maximum was recent, and use it if so */
			if ( (offset - element->lastmax[j]) < element->n ) out[i*channels +j] = element->max[j];
			/* Otherwise recompute the maximum over the last n samples */
			else {
				element->max[j] = pow(in[(offset - element->n + 1) * channels + j], element->moment);
				element->lastmax[j] = offset - element->n;
				for(k = 0; k < element->n; k++) {
					if(k > offset) break;
					currentvalue = pow(in[(offset - k) * channels + j], element->moment);
					if (currentvalue >= element->max[j] ) {
						element->max[j] = currentvalue;
						element->lastmax[j] = offset - k;
					}
				}
			out[i*channels +j] = element->max[j];
			}
		}
	}
	return 0;
}


/*
 * the number of samples available in the adapter
 */


static guint64 get_available_samples(GSTLALPeak *element)
{
	return gst_adapter_available(element->adapter) / sizeof(double) / element->channels;
}


/*
 * construct a buffer of zeros and push into adapter
 */


static int push_zeros(GSTLALPeak *element, unsigned samples)
{
	GstBuffer *zerobuf = gst_buffer_new_and_alloc(samples * sizeof(double) * element->channels);
	if(!zerobuf) {
		GST_DEBUG_OBJECT(element, "failure allocating zero-pad buffer");
		return -1;
	}
	memset(GST_BUFFER_DATA(zerobuf), 0, GST_BUFFER_SIZE(zerobuf));
	gst_adapter_push(element->adapter, zerobuf);
	return 0;
}


/*
 * set the metadata on an output buffer
 */


static void set_metadata(GSTLALPeak *element, GstBuffer *buf, guint64 outsamples, gboolean gap)
{
	GST_BUFFER_SIZE(buf) = outsamples * 1 * sizeof(double) * element->channels;	/* 1 = channels */
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
 * flush adapter
 */


static void flush(GSTLALPeak *element, guint64 available_length)
{
	if(available_length > element->n - 1)
		gst_adapter_flush(element->adapter, (available_length - (element->n - 1)) * sizeof(double) * element->channels);
}


/*
 * averaging algorithm front-end
 */


static GstFlowReturn filter(GSTLALPeak *element, GstBuffer *outbuf, guint64 output_length)
{
	double *in, *out;
	guint64 available_length;

	/*
	 * how much data is available?
	 */

	available_length = get_available_samples(element);
	if (available_length < (output_length+element->n - 1)) {
		push_zeros(element, output_length - available_length + element->n - 1);
		available_length = get_available_samples(element);
	}

	g_assert(available_length >= (output_length+element->n - 1));
	/*
	 * compute output samples FIXME support other caps
	 */

	in = (double *) gst_adapter_peek(element->adapter, available_length * sizeof(double) * element->channels);
	out = (double *) GST_BUFFER_DATA(outbuf);

	element->process(element, available_length, output_length, in, out);

	/*
	 * output produced?
	 */

	if(!output_length)
		return GST_BASE_TRANSFORM_FLOW_DROPPED;

	/*
	 * flush data from the adapter.  we want n-1 samples to remain
	 */

	flush(element, available_length-element->n + 1);

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
		"channels = (int) [1, MAX], " \
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
	GSTLALPeak,
	gstlal_peak,
	GstBaseTransform,
	GST_TYPE_BASE_TRANSFORM
);


enum property {
	ARG_N = 1,
	ARG_TYPE
};

#define DEFAULT_N 1
#define DEFAULT_TYPE 1
#define TYPE_MAX_OVER_N 1

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
 * set_caps()
 */


static gboolean set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps)
{
	GSTLALPeak *element = GSTLAL_PEAK(trans);
	GstStructure *s;
	gint rate;
	gint channels;
	gboolean success = TRUE;

	s = gst_caps_get_structure(outcaps, 0);
	if(!gst_structure_get_int(s, "rate", &rate)) {
		GST_DEBUG_OBJECT(element, "unable to parse rate from %" GST_PTR_FORMAT, outcaps);
		success = FALSE;
	}

	if(success) {
		element->rate = rate;
	}

	if(!gst_structure_get_int(s, "channels", &channels)) {
		GST_DEBUG_OBJECT(element, "unable to parse channels from %" GST_PTR_FORMAT, outcaps);
		success = FALSE;
	}

	if(success) {
		element->channels = channels;
	}

	/* FIXME I don't know where is best to do this */
	if (element->type == TYPE_MAX_OVER_N) element->process = GST_DEBUG_FUNCPTR(max_over_n_process);
	return success;
}


/*
 * start()
 */


static gboolean start(GstBaseTransform *trans)
{
	GSTLALPeak *element = GSTLAL_PEAK(trans);
	element->adapter = gst_adapter_new();
	element->t0 = GST_CLOCK_TIME_NONE;
	element->offset0 = GST_BUFFER_OFFSET_NONE;
	element->next_out_offset = GST_BUFFER_OFFSET_NONE;
	element->need_discont = TRUE;
	element->invert_thresh=FALSE;

	return TRUE;
}


/*
 * stop()
 */


static gboolean stop(GstBaseTransform *trans)
{
	GSTLALPeak *element = GSTLAL_PEAK(trans);
	g_object_unref(element->adapter);
	element->adapter = NULL;
	if (element->sum1) free(element->sum1);
	if (element->sum2) free(element->sum2);
	return TRUE;
}


/*
 * transform()
 */


static GstFlowReturn transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf)
{
	GSTLALPeak *element = GSTLAL_PEAK(trans);
	guint64 in_length;
	GstFlowReturn result;

	/*
	 * check for discontinuity
	 */

	if(GST_BUFFER_IS_DISCONT(inbuf)) {
		/*
		 * flush adapter
		 */

		gst_adapter_clear(element->adapter);

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

	/*
	 * gap logic
	 */

	in_length = GST_BUFFER_OFFSET_END(inbuf) - GST_BUFFER_OFFSET(inbuf);
	if(!GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP)) {
		/*
		 * input is not 0s.
		 */

		gst_buffer_ref(inbuf);	/* don't let the adapter free it */
		gst_adapter_push(element->adapter, inbuf);
		result = filter(element, outbuf, in_length);
	} else {
		/*
		 * input is 0s.
		 */

		push_zeros(element, in_length);
		flush(element, get_available_samples(element));
		memset(GST_BUFFER_DATA(outbuf), 0, GST_BUFFER_SIZE(outbuf));
		set_metadata(element, outbuf, in_length, TRUE);
		result = GST_FLOW_OK;
	}

	/*
	 * done
	 */

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
	GSTLALPeak *element = GSTLAL_PEAK(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_N: {
		guint32 old_n = element->n;
		element->n = g_value_get_uint(value);
		if(element->n != old_n)
			g_object_notify(object, "n");
		break;
	}

	case ARG_TYPE: {
		guint32 old_type = element->type;
		element->type = g_value_get_uint(value);
		if(element->type != old_type)
			g_object_notify(object, "type");
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
	GSTLALPeak *element = GSTLAL_PEAK(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_N:
		g_value_set_uint(value, element->n);
		break;

	case ARG_TYPE:
		g_value_set_uint(value, element->type);
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
	GSTLALPeak *element = GSTLAL_PEAK(object);

	/*
	 * free resources
	 */

	if(element->adapter) {
		g_object_unref(element->adapter);
		element->adapter = NULL;
	}


	if (element->sum1) free(element->sum1);
	if (element->sum2) free(element->sum2);
	if (element->max) free(element->max);
	if (element->lastmax) free(element->lastmax);
	if (element->lastcross) free(element->lastcross);

	/*
	 * chain to parent class' finalize() method
	 */

	G_OBJECT_CLASS(parent_class)->finalize(object);

}


/*
 * base_init()
 */


static void gstlal_peak_base_init(gpointer gclass)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(gclass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(gclass);

	gst_element_class_set_details_simple(element_class, "Average, max or thresh of last N samples", "Filter/Audio", "Each output sample is some average-like quantity of the N most recent samples", "Kipp Cannon <kipp.cannon@ligo.org>, Chad Hanna <chad.hanna@ligo.org>");

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));

	transform_class->get_unit_size = GST_DEBUG_FUNCPTR(get_unit_size);
	transform_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
	transform_class->transform = GST_DEBUG_FUNCPTR(transform);
	transform_class->start = GST_DEBUG_FUNCPTR(start);
	transform_class->stop = GST_DEBUG_FUNCPTR(stop);
}


/*
 * class_init()
 */


static void gstlal_peak_class_init(GSTLALPeakClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	g_object_class_install_property(
		gobject_class,
		ARG_N,
		g_param_spec_uint(
			"n",
			"n",
			"Number of samples to peak find over (symmetric).",
			0, G_MAXUINT, DEFAULT_N,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);

	g_object_class_install_property(
		gobject_class,
		ARG_TYPE,
		g_param_spec_uint(
			"type",
			"type",
			"type of peak finding 1=max over n",
			0, G_MAXUINT, DEFAULT_TYPE,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);
}


/*
 * init()
 */


static void gstlal_peak_init(GSTLALPeak *filter, GSTLALPeakClass *klass)
{
	filter->rate = 0;
	filter-> moment = 1;
	filter->adapter = NULL;
	filter->sum1 = NULL;
	filter->sum2 = NULL;
	filter->max = NULL;
	filter->lastmax = NULL;
	filter->lastcross = NULL;
	filter->n = DEFAULT_N;
	filter->type = DEFAULT_TYPE;
	filter->process = GST_DEBUG_FUNCPTR(max_over_n_process);
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(filter), TRUE);
}

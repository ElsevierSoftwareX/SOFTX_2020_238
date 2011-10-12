/*
 * An element to find peaks
 *
 * Copyright (C) 2011  Chad Hanna
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
 * ========================================================================
 *
 *                                  Preamble
 *
 * ========================================================================
 */


/*
 * struff from the C library
 */


/*
 * stuff from glib/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstadapter.h>
#include <math.h>
#include <string.h>

/*
 * our own stuff
 */

#include <gstlal.h>
#include <gstlal_peak.h>


static guint64 output_num_samps(GSTLALPeak *element)
{
	return (guint64) element->n;
}


static guint64 output_num_bytes(GSTLALPeak *element)
{
	return (guint64) output_num_samps(element) * element->unit_size;
}


static GstClockTime output_duration(GSTLALPeak *element)
{
	return (GstClockTime) gst_util_uint64_scale_int_round(GST_SECOND, output_num_samps(element), element->rate);
}


static guint64 available_samples(GSTLALPeak *element)
{
	/*FIXME check that the number of bytes is factor of unit size */
	return (guint64) gst_adapter_available(element->adapter) / element->unit_size;
}


static int reset_time_and_offset(GSTLALPeak *element)
{
	element->next_output_offset = 0;
	element->next_output_timestamp = GST_CLOCK_TIME_NONE;
	return 0;
}


static guint peak_finder(GSTLALPeak *element, GstBuffer *srcbuf)
{
	guint64 length = output_num_samps(element);
	/*
	 * Pointer arithmetic to define three buffer regions, we'll identify triggers in the middle
	 * Peeks are okay, we want the data to remain in the adapters
	 */

	const double *data = (const double *) gst_adapter_peek(element->adapter, 1 * GST_BUFFER_SIZE(srcbuf));
	double *outputdata = (double *) GST_BUFFER_DATA(srcbuf);

	guint sample;
	guint channel;
	double *maxdata = NULL;
	guint *maxsample = NULL;
	guint index;
	
	/* FIXME make array to store the max part of element instance for performance */
	if (!element->maxdata) element->maxdata = (double *) calloc(element->channels, sizeof(double));
	else memset(element->maxdata, 0.0, element->channels * sizeof(double));
	if (!element->maxsample) element->maxsample = (guint *) calloc(element->channels, sizeof(guint));
	else memset(element->maxsample, 0.0, element->channels * sizeof(guint));
	maxdata = element->maxdata;
	maxsample = element->maxsample;
	
	/* Find maxima of the data */
	for(sample = 0; sample < length; sample++) {
		for(channel = 0; channel < element->channels; channel++) {
			if(fabs(*data) > fabs(maxdata[channel])) {
				maxdata[channel] = *data;
				maxsample[channel] = sample;
			}
		data++;
		}
	}
	
	/* Decide if there are any events to keep */
	for(channel = 0; channel < element->channels; channel++) {
		if ( maxdata[channel] ) {
			index = maxsample[channel] * element->channels + channel;
			outputdata[index] = maxdata[channel];
		}
	}
	return 0;
}


/*
 * ============================================================================
 *
 *                                 Properties
 *
 * ============================================================================
 */


enum property {
	ARG_N = 1
};


static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec)
{
	GSTLALPeak *element = GSTLAL_PEAK(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_N:
		element->n = g_value_get_uint(value);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *pspec)
{
	GSTLALPeak *element = GSTLAL_PEAK(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_N:
		g_value_set_uint(value, element->n);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * ============================================================================
 *
 *                                  Sink Pad
 *
 * ============================================================================
 */


/*
 * getcaps()
 */


static GstCaps *getcaps(GstPad * pad)
{
	GSTLALPeak *element = GSTLAL_PEAK(gst_pad_get_parent(pad));
	GstCaps *peercaps, *caps;

	/*
	 * get our own allowed caps.  use the fixed caps function to avoid
	 * recursing back into this function.
	 */

	caps = gst_pad_get_fixed_caps_func(pad);

	/*
	 * get the allowed caps from the downstream peer if the peer has
	 * caps, intersect without our own.
	 */

	peercaps = gst_pad_peer_get_caps(element->srcpad);
	if(peercaps) {
		GstCaps *result = gst_caps_intersect(peercaps, caps);
		gst_caps_unref(peercaps);
		gst_caps_unref(caps);
		caps = result;
	}

	/*
	 * done
	 */

	gst_object_unref(element);
	return caps;
}


/*
 * setcaps()
 */


static gboolean setcaps(GstPad *pad, GstCaps *caps)
{
	GSTLALPeak *element = GSTLAL_PEAK(gst_pad_get_parent(pad));
	GstStructure *structure;
	gint rate, width, channels;
	gboolean success = TRUE;

	/*
	 * parse caps
	 */

	structure = gst_caps_get_structure(caps, 0);
	if(!gst_structure_get_int(structure, "rate", &rate))
		success = FALSE;
	if(!gst_structure_get_int(structure, "width", &width))
		success = FALSE;
	if(!gst_structure_get_int(structure, "channels", &channels))
		success = FALSE;

	/*
	 * try setting caps on downstream element
	 */

	if(success)
		success = gst_pad_set_caps(element->srcpad, caps);

	/*
	 * update the element metadata
	 */

	if(success) {
		element->channels = channels;
		element->rate = rate;
		element->unit_size = width / 8 * channels;
	}

	/*
	 * done
	 */

	gst_object_unref(element);
	return success;
}


/*
 * chain()
 */


static GstFlowReturn prepare_output_buffer(GSTLALPeak *element, GstBuffer **srcbuf)
{
	GstFlowReturn result = gst_pad_alloc_buffer(element->srcpad, element->next_output_offset, output_num_bytes(element), GST_PAD_CAPS(element->srcpad), srcbuf);
	if(result != GST_FLOW_OK)
		return result;

	g_assert(GST_BUFFER_CAPS(*srcbuf) != NULL);
	g_assert(GST_PAD_CAPS(element->srcpad) == GST_BUFFER_CAPS(*srcbuf));
	
	/* set the offset */
	GST_BUFFER_OFFSET(*srcbuf) = element->next_output_offset;
	element->next_output_offset += output_num_samps(element);
	GST_BUFFER_OFFSET_END(*srcbuf) = element->next_output_offset;

	/* set the time stamps */
	GST_BUFFER_TIMESTAMP(*srcbuf) = element->next_output_timestamp;
	element->next_output_timestamp += output_duration(element);
	GST_BUFFER_DURATION(*srcbuf) = output_duration(element);
	
	/* memset to zero */
	memset(GST_BUFFER_DATA(*srcbuf), 0, GST_BUFFER_SIZE(*srcbuf));

	return result;

}

static int push_zeros(GSTLALPeak *element, GstBuffer *inbuf)
{

	/*
	 * Function to push zeros into the specified adapter
	 * useful when there are gaps
	 */

	guint bytes = GST_BUFFER_SIZE(inbuf);
	GstBuffer *zerobuf = gst_buffer_new_and_alloc(bytes);
	if(!zerobuf) {
		GST_DEBUG_OBJECT(element, "failure allocating zero-pad buffer");
		return -1;
	}
	gst_buffer_copy_metadata(zerobuf, inbuf, GST_BUFFER_COPY_TIMESTAMPS | GST_BUFFER_COPY_CAPS);
	memset(GST_BUFFER_DATA(zerobuf), 0, GST_BUFFER_SIZE(zerobuf));
	gst_adapter_push(element->adapter, zerobuf);
	return 0;
}

static GstFlowReturn push_buffer(GSTLALPeak *element, GstBuffer *srcbuf)
{
	GST_DEBUG_OBJECT(element, "pushing %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(srcbuf));
	return gst_pad_push(element->srcpad, srcbuf);
}

static GstFlowReturn chain(GstPad *pad, GstBuffer *sinkbuf)
{
	GSTLALPeak *element = GSTLAL_PEAK(gst_pad_get_parent(pad));
	GstFlowReturn result = GST_FLOW_OK;
	GstBuffer *srcbuf = NULL;
	guint64 maxsize = output_num_bytes(element);

	/*
	 * check validity of timestamp and offsets
	 */

	if(!GST_BUFFER_TIMESTAMP_IS_VALID(sinkbuf) || !GST_BUFFER_DURATION_IS_VALID(sinkbuf) || !GST_BUFFER_OFFSET_IS_VALID(sinkbuf) || !GST_BUFFER_OFFSET_END_IS_VALID(sinkbuf)) {
		gst_buffer_unref(sinkbuf);
		GST_ERROR_OBJECT(element, "error in input stream: buffer has invalid timestamp and/or offset");
		result = GST_FLOW_ERROR;
		goto done;
	}

	/* FIXME if we were more careful we wouldn't lose so much data around disconts */
	if (GST_BUFFER_FLAG_IS_SET(sinkbuf, GST_BUFFER_FLAG_DISCONT)) {
		reset_time_and_offset(element);
		gst_adapter_clear(element->adapter);
	}

	/* if we don't have a valid first timestamp yet take this one */
	if (element->next_output_timestamp == GST_CLOCK_TIME_NONE) {
		element->next_output_timestamp = GST_BUFFER_TIMESTAMP(sinkbuf);
	}

	/* put the incoming buffer into an adapter or push zeros if gap */

	if (GST_BUFFER_FLAG_IS_SET(sinkbuf, GST_BUFFER_FLAG_GAP))
		push_zeros(element, sinkbuf);
	else {
		gst_adapter_push(element->adapter, sinkbuf);
	}
	
	/* while we have enough data to do the calculation, do it and push out buffers n samples long */
	while(gst_adapter_available(element->adapter) >= 1 * maxsize && result == GST_FLOW_OK) {
		if (prepare_output_buffer(element, &srcbuf) == GST_FLOW_OK) {
			peak_finder(element, srcbuf);
			result = push_buffer(element, srcbuf);
			/* knock off the first buffers worth of bytes since we don't need them any more */
			gst_adapter_flush(element->adapter, output_num_bytes(element));
		}
	}

done:
	gst_object_unref(element);
	return result;
}


/*
 * ============================================================================
 *
 *                                Type Support
 *
 * ============================================================================
 */


/*
 * Parent class.
 */


static GstElementClass *parent_class = NULL;


/*
 * Instance finalize function.  See ???
 */


static void finalize(GObject *object)
{
	GSTLALPeak *element = GSTLAL_PEAK(object);

	gst_object_unref(element->sinkpad);
	element->sinkpad = NULL;
	gst_object_unref(element->srcpad);
	element->srcpad = NULL;

	/*FIXME handle adapter */

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


/*
 * Base init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GBaseInitFunc
 */


#define CAPS \
	"audio/x-raw-float, " \
	"rate = (int) [1, MAX], " \
	"channels = (int) [1, MAX], " \
	"endianness = (int) BYTE_ORDER, " \
	"width = (int) {64}; "


static void base_init(gpointer class)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(class);

	gst_element_class_set_details_simple(
		element_class,
		"Peak",
		"Filter",
		"Find peaks in a time series every n samples",
		"Chad Hanna <chad.hanna@ligo.org>"
	);

	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"sink",
			GST_PAD_SINK,
			GST_PAD_ALWAYS,
			gst_caps_from_string(CAPS)
		)
	);
	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"src",
			GST_PAD_SRC,
			GST_PAD_ALWAYS,
			gst_caps_from_string(CAPS)
		)
	);
}


/*
 * Class init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GClassInitFunc
 */


static void class_init(gpointer class, gpointer class_data)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(class);

	parent_class = g_type_class_ref(GST_TYPE_ELEMENT);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	g_object_class_install_property(
		gobject_class,
		ARG_N,
		g_param_spec_uint(
			"n",
			"n",
			"number of samples over which to identify peaks",
			0, G_MAXUINT, 0,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
}


/*
 * Instance init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GInstanceInitFunc
 */


static void instance_init(GTypeInstance *object, gpointer class)
{
	GSTLALPeak *element = GSTLAL_PEAK(object);
	GstPad *pad;

	gst_element_create_all_pads(GST_ELEMENT(element));

	/* configure (and ref) sink pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "sink");
	gst_pad_set_getcaps_function(pad, GST_DEBUG_FUNCPTR(getcaps));
	gst_pad_set_setcaps_function(pad, GST_DEBUG_FUNCPTR(setcaps));
	gst_pad_set_chain_function(pad, GST_DEBUG_FUNCPTR(chain));
	element->sinkpad = pad;

	/* retrieve (and ref) src pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "src");
	element->srcpad = pad;

	/* internal data */
	element->rate = 0;
	element->unit_size = 0;
	reset_time_and_offset(element);

	element->adapter = gst_adapter_new();
}


/*
 * gstlal_peak_get_type().
 */


GType gstlal_peak_get_type(void)
{
	static GType type = 0;

	if(!type) {
		static const GTypeInfo info = {
			.class_size = sizeof(GSTLALPeakClass),
			.class_init = class_init,
			.base_init = base_init,
			.instance_size = sizeof(GSTLALPeak),
			.instance_init = instance_init,
		};
		type = g_type_register_static(GST_TYPE_ELEMENT, "lal_peak", &info, 0);
	}

	return type;
}

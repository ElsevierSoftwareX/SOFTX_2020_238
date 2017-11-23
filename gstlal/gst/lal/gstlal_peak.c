/*
 * An peak finding element
 *
 * Copyright (C) 2011  Chad Hanna, Kipp Cannon
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


#include <math.h>
#include <string.h>


/*
 * stuff from glib/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/audio/audio.h>
#include <gst/base/gstadapter.h>


/*
 * our own stuff
 */


#include <gstlal/gstlal_debug.h>
#include <gstlal/gstlal_peakfinder.h>
#include <gstlal/gstaudioadapter.h>
#include <gstlal_peak.h>


static guint64 output_num_samps(GSTLALPeak *element)
{
	return element->n;
}


static guint64 output_num_bytes(GSTLALPeak *element)
{
	return output_num_samps(element) * element->adapter->unit_size;
}


static int reset_time_and_offset(GSTLALPeak *element)
{
	element->samples_since_last_discont = 0;
	element->timestamp_at_last_discont = GST_CLOCK_TIME_NONE;
	element->next_output_offset = 0;
	element->next_output_timestamp = GST_CLOCK_TIME_NONE;
	return 0;
}


static guint gst_audioadapter_available_samples(GstAudioAdapter *adapter)
{
	guint size;
	g_object_get(adapter, "size", &size, NULL);
	return size;
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
 *                                    Pads
 *
 * ============================================================================
 */


/*
 * getcaps()
 */


static GstCaps *getcaps(GSTLALPeak *peak, GstPad *pad, GstCaps *filter)
{
	GstCaps *result, *peercaps, *current_caps, *filter_caps;

	/* take filter */
	filter_caps = filter ? gst_caps_ref(filter) : NULL;

	/* 
	 * If the filter caps are empty (but not NULL), there is nothing we can
	 * do, there will be no intersection
	 */
	if (filter_caps && gst_caps_is_empty (filter_caps)) {
		GST_WARNING_OBJECT (pad, "Empty filter caps");
		return filter_caps;
	}

	/* get the downstream possible caps */
	peercaps = gst_pad_peer_query_caps(peak->srcpad, filter_caps);

	/* get the allowed caps on this sinkpad */
	current_caps = gst_pad_get_pad_template_caps(pad);
	if (!current_caps)
			current_caps = gst_caps_new_any();

	if (peercaps) {
		/* if the peer has caps, intersect */
		GST_DEBUG_OBJECT(peak, "intersecting peer and our caps");
		result = gst_caps_intersect_full(peercaps, current_caps, GST_CAPS_INTERSECT_FIRST);
		/* neither peercaps nor current_caps are needed any more */
		gst_caps_unref(peercaps);
		gst_caps_unref(current_caps);
	}
	else {
		/* the peer has no caps (or there is no peer), just use the allowed caps
		* of this sinkpad. */
		/* restrict with filter-caps if any */
		if (filter_caps) {
			GST_DEBUG_OBJECT(peak, "no peer caps, using filtered caps");
			result = gst_caps_intersect_full(filter_caps, current_caps, GST_CAPS_INTERSECT_FIRST);
			/* current_caps are not needed any more */
			gst_caps_unref(current_caps);
		}
		else {
			GST_DEBUG_OBJECT(peak, "no peer caps, using our caps");
			result = current_caps;
		}
	}

	result = gst_caps_make_writable (result);

	if (filter_caps)
		gst_caps_unref (filter_caps);

	GST_LOG_OBJECT (peak, "getting caps on pad %p,%s to %" GST_PTR_FORMAT, pad, GST_PAD_NAME(pad), result);

	return result;
}


/*
 * setcaps()
 */


static gboolean setcaps(GSTLALPeak *peak, GstPad *pad, GstCaps *caps)
{
	GstAudioInfo info;

	/*
	 * parse caps
	 */

	gboolean success = gst_audio_info_from_caps(&info, caps);

	/*
	 * try setting caps on downstream element
	 */

	if(success)
		success = gst_pad_set_caps(peak->srcpad, caps);

	/*
	 * update the element metadata
	 */

	if(success) {
		peak->channels = GST_AUDIO_INFO_CHANNELS(&info);
		peak->rate = GST_AUDIO_INFO_RATE(&info);
		g_object_set(peak->adapter, "unit-size", GST_AUDIO_INFO_BPF(&info), NULL);
		switch(GST_AUDIO_INFO_WIDTH(&info)) {
		case 64:
			peak->peak_type = GSTLAL_PEAK_DOUBLE;
			break;
		case 32:
			peak->peak_type = GSTLAL_PEAK_FLOAT;
			break;
		default:
			g_assert_not_reached();
			break;
		}
		peak->maxdata = gstlal_peak_state_new(peak->channels, peak->peak_type);
	}

	/*
	 * done
	 */

	return success;
}


/*
 * Events and queries
 */


static gboolean src_query(GstPad *pad, GstObject *parent, GstQuery *query)
{
	gboolean res = FALSE;

	switch (GST_QUERY_TYPE (query))
	{
		default:
			res = gst_pad_query_default (pad, parent, query);
			break;
	}
	return res;
}


static gboolean src_event(GstPad *pad, GstObject *parent, GstEvent *event)
{
	GSTLALPeak *peak = GSTLAL_PEAK(parent);
	gboolean result = TRUE;
	GST_DEBUG_OBJECT (pad, "Got %s event on src pad", GST_EVENT_TYPE_NAME(event));

	switch (GST_EVENT_TYPE (event))
	{	
		default:
			/* just forward the rest for now */
			GST_DEBUG_OBJECT(peak, "forward unhandled event: %s", GST_EVENT_TYPE_NAME (event));
			gst_pad_event_default(pad, parent, event);
			break;
	}

	return result;
}


static gboolean sink_query(GstPad *pad, GstObject *parent, GstQuery * query)
{
	GSTLALPeak *peak = GSTLAL_PEAK(parent);
	gboolean res = TRUE;
	GstCaps *filter, *caps;

	switch (GST_QUERY_TYPE (query)) 
	{
		case GST_QUERY_CAPS:
			gst_query_parse_caps (query, &filter);
			caps = getcaps(peak, pad, filter);
			gst_query_set_caps_result (query, caps);
			gst_caps_unref (caps);
			break;
		default:
			break;
	}

	if (G_LIKELY (query))
		return gst_pad_query_default (pad, parent, query);
	else
		return res;

  return res;
}


static gboolean sink_event(GstPad *pad, GstObject *parent, GstEvent *event)
{
	GSTLALPeak *peak = GSTLAL_PEAK(parent);
	gboolean res = TRUE;
	GstCaps *caps;

	GST_DEBUG_OBJECT(pad, "Got %s event on sink pad", GST_EVENT_TYPE_NAME (event));

	switch (GST_EVENT_TYPE (event))
	{
		case GST_EVENT_CAPS:
			gst_event_parse_caps(event, &caps);
			res = setcaps(peak, pad, caps);
			gst_event_unref(event);
			event = NULL;
			break;
		default:
			break;
	}

	if (G_LIKELY (event))
		return gst_pad_event_default(pad, parent, event);
	else
		return res;
}


/*
 * chain()
 */

static void update_state(GSTLALPeak *element, GstBuffer *srcbuf)
{
	element->next_output_offset = GST_BUFFER_OFFSET_END(srcbuf);
	gint samples = GST_BUFFER_OFFSET_END(srcbuf) - GST_BUFFER_OFFSET(srcbuf);
	GST_BUFFER_PTS(srcbuf) = gst_util_uint64_scale_int_round(element->samples_since_last_discont, GST_SECOND, element->rate) + element->timestamp_at_last_discont;
	GST_BUFFER_DURATION(srcbuf) = gst_util_uint64_scale_int_round(samples, GST_SECOND, element->rate);
	element->next_output_timestamp = gst_util_uint64_scale_int_round(element->samples_since_last_discont + samples, GST_SECOND, element->rate) + element->timestamp_at_last_discont;;
	element->samples_since_last_discont += samples;
}

static GstFlowReturn push_buffer(GSTLALPeak *element, GstBuffer *srcbuf)
{
	GST_DEBUG_OBJECT(element, "pushing %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(srcbuf));
	return gst_pad_push(element->srcpad, srcbuf);
}

static GstFlowReturn chain(GstPad *pad, GstObject *parent, GstBuffer *sinkbuf)
{
	GSTLALPeak *element = GSTLAL_PEAK(parent);
	GstFlowReturn result = GST_FLOW_OK;
	GstBuffer *srcbuf = NULL;
	guint64 maxsize = output_num_bytes(element);
	gboolean copied_gap, copied_nongap;
	guint outsamps, gapsamps, nongapsamps;

	/* if we haven't allocated storage do it now, we should never try to copy from an adapter with a larger buffer than this */
	if (!element->data)
		element->data = g_malloc(maxsize);

	/*
	 * check validity of timestamp and offsets
	 */

	if(!GST_BUFFER_PTS_IS_VALID(sinkbuf) || !GST_BUFFER_DURATION_IS_VALID(sinkbuf) || !GST_BUFFER_OFFSET_IS_VALID(sinkbuf) || !GST_BUFFER_OFFSET_END_IS_VALID(sinkbuf)) {
		gst_buffer_unref(sinkbuf);
		GST_ERROR_OBJECT(element, "error in input stream: buffer has invalid timestamp and/or offset");
		result = GST_FLOW_ERROR;
		goto done;
	}

	/* FIXME if we were more careful we wouldn't lose so much data around disconts */
	if (GST_BUFFER_FLAG_IS_SET(sinkbuf, GST_BUFFER_FLAG_DISCONT)) {
		reset_time_and_offset(element);
		gst_audioadapter_clear(element->adapter);
	}

	/* if we don't have a valid first timestamp yet take this one */
	if (element->next_output_timestamp == GST_CLOCK_TIME_NONE) {
		element->timestamp_at_last_discont = GST_BUFFER_PTS(sinkbuf);
		element->next_output_timestamp = GST_BUFFER_PTS(sinkbuf);
	}

	/* put the incoming buffer into an adapter, handles gaps */
	gst_audioadapter_push(element->adapter, sinkbuf);

	/* while we have enough data to do the calculation, do it and push out buffers n samples long */
	while(gst_audioadapter_available_samples(element->adapter) >= element->n) {

		/* See if the output is a gap or not */
		nongapsamps = gst_audioadapter_head_nongap_length(element->adapter);
		gapsamps = gst_audioadapter_head_gap_length(element->adapter);

		if (gapsamps > 0) {
			outsamps = gapsamps > element->n ? element->n : gapsamps;
			/* Clearing the max data structure causes the resulting buffer to be a GAP */
			gstlal_peak_state_clear(element->maxdata);
		}
		else {
			outsamps = nongapsamps > element->n ? element->n : nongapsamps;
			/* call the peak finding library on a buffer from the adapter if no events are found the result will be a GAP */
			gst_audioadapter_copy_samples(element->adapter, element->data, outsamps, &copied_gap, &copied_nongap);
			gstlal_peak_over_window(element->maxdata, (const void*) element->data, outsamps);
		}	
		
		srcbuf = gstlal_new_buffer_from_peak(element->maxdata, element->srcpad, element->next_output_offset, outsamps, element->next_output_timestamp, element->rate);

		/* set the time stamp and offset state */
		update_state(element, srcbuf);

		/* push the result */
		result = push_buffer(element, srcbuf);

		/* knock off the first buffers worth of bytes since we don't need them any more */
		gst_audioadapter_flush_samples(element->adapter, outsamps);
	}

done:
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


static GstElementClass *gstlal_peak_parent_class = NULL;


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
	gst_audioadapter_clear(element->adapter);
	g_object_unref(element->adapter);
	if (element->maxdata)
		gstlal_peak_state_free(element->maxdata);
	if (element->data)
		g_free(element->data);  
	G_OBJECT_CLASS(gstlal_peak_parent_class)->finalize(object);
}


/*
 * class_init()
 */


#define CAPS \
	GST_AUDIO_CAPS_MAKE("{" GST_AUDIO_NE(F32) ", " GST_AUDIO_NE(F64) "}") ", " \
	"layout = (string) interleaved, " \
	"channel-mask = (bitmask) 0"


static void class_init(gpointer class, gpointer class_data)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(class);
	GObjectClass *gobject_class = G_OBJECT_CLASS(class);

	gst_element_class_set_details_simple(
		element_class,
		"Peak",
		"Filter",
		"Find peaks in a time series every n samples",
		"Chad Hanna <chad.hanna@ligo.org>"
	);

	gstlal_peak_parent_class = g_type_class_ref(GST_TYPE_ELEMENT);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

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
	gst_pad_set_query_function(pad, GST_DEBUG_FUNCPTR(sink_query));
	gst_pad_set_event_function(pad, GST_DEBUG_FUNCPTR(sink_event));
	gst_pad_set_chain_function(pad, GST_DEBUG_FUNCPTR(chain));
	element->sinkpad = pad;

	/* retrieve (and ref) src pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "src");
	gst_pad_set_query_function(pad, GST_DEBUG_FUNCPTR (src_query));
	gst_pad_set_event_function(pad, GST_DEBUG_FUNCPTR (src_event));
	element->srcpad = pad;

	/* internal data */
	element->rate = 0;
	reset_time_and_offset(element);
	element->maxdata = NULL;
	element->adapter = g_object_new(GST_TYPE_AUDIOADAPTER, NULL);
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
			.instance_size = sizeof(GSTLALPeak),
			.instance_init = instance_init,
		};
		type = g_type_register_static(GST_TYPE_ELEMENT, "GSTLALPeak", &info, 0);
	}

	return type;
}

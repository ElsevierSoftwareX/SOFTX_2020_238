/*
 * Inpaints.
 *
 * Copyright (C) 2020 Cody Messick
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
 *
 */
		   

/*
 * ========================================================================
 *
 *				  Preamble
 *
 * ========================================================================
 */


/*
 * stuff from the C library
 */


#include <math.h>
#include <stdint.h>
#include <string.h>


/*
 * stuff from gstreamer
 */


#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
//#include <gst/audio/audio.h> 

/*
 * our own stuff
 */


#include <gstlal/gstlal.h>
#include <gstlal/gstlal_tags.h>
#include <gstlal_inpaint.h>

/*
 * stuff from LAL
 */

#include <lal/Date.h>


/*
 * ============================================================================
 *
 *                                Boilerplate
 *
 * ============================================================================
 */


#define GST_CAT_DEFAULT gstlal_inpaint_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


G_DEFINE_TYPE_WITH_CODE(
	GSTLALInpaint,
	gstlal_inpaint,
	GST_TYPE_BASE_TRANSFORM,
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "lal_inpaint", 0, "lal_inpaint element")
);


/*
 * ============================================================================
 *
 *                         GstBaseTransform Overrides
 *
 * ============================================================================
 */


static gboolean taglist_extract_string(GstObject *object, GstTagList *taglist, const char *tagname, gchar **dest)
{
	if(!gst_tag_list_get_string(taglist, tagname, dest)) {
		GST_WARNING_OBJECT(object, "unable to parse \"%s\" from %" GST_PTR_FORMAT, tagname, taglist);
		return FALSE;
	}
        return TRUE;
}

/*
 * sink_event()
 */


static gboolean gstlal_inpaint_sink_event(GstBaseTransform *trans, GstEvent *event)
{
	GSTLALInpaint *inpaint = GSTLAL_INPAINT(trans);
	gboolean result = TRUE;

	GST_DEBUG_OBJECT(trans, "Got %s event on sink pad", GST_EVENT_TYPE_NAME (event));

	switch (GST_EVENT_TYPE(event)) {
		case GST_EVENT_TAG:
		{
			GstTagList *taglist;
			gchar *instrument = NULL, *channel_name = NULL, *units = NULL;

			/*
			 * attempt to extract all 3 tags from the event's taglist
			 */

			gst_event_parse_tag(event, &taglist);
			result = taglist_extract_string(GST_OBJECT(trans), taglist, GSTLAL_TAG_INSTRUMENT, &instrument);
			result &= taglist_extract_string(GST_OBJECT(trans), taglist, GSTLAL_TAG_CHANNEL_NAME, &channel_name);
			result &= taglist_extract_string(GST_OBJECT(trans), taglist, GSTLAL_TAG_UNITS, &units);

			if(result) {
				GST_DEBUG_OBJECT(inpaint, "found tags \"%s\"=\"%s\", \"%s\"=\"%s\"", GSTLAL_TAG_INSTRUMENT, instrument, GSTLAL_TAG_CHANNEL_NAME, channel_name);
				free(inpaint->instrument);
				inpaint->instrument = instrument;
				free(inpaint->channel_name);
				inpaint->channel_name = channel_name;
				free(inpaint->units);
				inpaint->units = units;
			}
		}
                        break;
		default:
			break;
	}

	if(!result) 
		gst_event_unref(event);
	else 
		result = GST_BASE_TRANSFORM_CLASS(gstlal_inpaint_parent_class)->sink_event(trans, event);

	return result;
}


/*
 * chain()
 */


static GstFlowReturn gstlal_inpaint_transform_ip(GstBaseTransform *trans, GstBuffer *buf)
{
	GSTLALInpaint *inpaint = GSTLAL_INPAINT(trans);
	GstFlowReturn result = GST_FLOW_OK;
	GstMapInfo info;
	REAL8TimeSeries *hoft;
	LIGOTimeGPS gate_start, gate_end, hoft_start, hoft_end, t_idx;
	guint idx;

	// Prototype: Just set h(t) during GW170817 gate to zero in L1 (test to
	// make sure I understand how these pieces fit together since this is
	// my first transform element)
	// GW170817 gate: 1187008881.37875 to 1187008881.44125
	if(strcmp(inpaint->instrument, "L1") != 0)
		return result;

	hoft = gstlal_buffer_map_REAL8TimeSeries(buf, gst_pad_get_current_caps(GST_BASE_TRANSFORM_SINK_PAD(trans)), &info, inpaint->instrument, inpaint->channel_name, inpaint->units);
	if(!hoft) {
		GST_ERROR_OBJECT(GST_ELEMENT(trans), "failure wrapping buffer in REAL8TimeSeries");
		result = GST_FLOW_ERROR;
		return result;
	}

	XLALGPSSet(&gate_start, 1187008881, 378750000);
	XLALGPSSet(&gate_end, 1187008881, 441250000);
	hoft_start = hoft->epoch;
	hoft_end = hoft_start;
	// hoft_end is timestamp of first sample not included in buffer
	XLALGPSAdd(&hoft_end, hoft->data->length * hoft->deltaT); 

	// If the gate is completely disjoint with the buffer, free refs and move on
	// NOTE Assumes all segments are the half-inclusive interval [start, end)
	if(XLALGPSCmp(&gate_start, &hoft_end) >= 0 || XLALGPSCmp(&gate_end, &hoft_start) == -1) { 
		gstlal_buffer_unmap_REAL8TimeSeries(buf, &info, hoft);
		return result;
	}
	
	for(idx=0; idx < hoft->data->length; idx++) {
		if(idx == 0)
			t_idx = hoft_start;
		else
			XLALGPSAdd(&t_idx, hoft->deltaT);

		if(XLALGPSCmp(&t_idx, &gate_start) >= 0 && XLALGPSCmp(&t_idx, &gate_end) == -1)
			hoft->data->data[idx] = 0;
	}

	gstlal_buffer_unmap_REAL8TimeSeries(buf, &info, hoft);

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
 * Properties
 */



/*
 * finalize()
 */


static void gstlal_inpaint_finalize(GObject * object)
{
	GSTLALInpaint *inpaint = GSTLAL_INPAINT(object);

	free(inpaint->instrument);
	free(inpaint->channel_name);
	free(inpaint->units);

	G_OBJECT_CLASS(gstlal_inpaint_parent_class)->finalize(object);
}


/*
 * class_init()
 */


#define CAPS \
	"audio/x-raw, " \
	"format = (string) " GST_AUDIO_NE(F64) ", " \
	"rate = " GST_AUDIO_RATE_RANGE ", " \
	"channels = (int) 1, " \
	"layout = (string) interleaved, " \
	"channel-mask = (bitmask) 0"


static void gstlal_inpaint_class_init(GSTLALInpaintClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(klass);

	//gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	//gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(gstlal_inpaint_finalize);

	transform_class->sink_event = GST_DEBUG_FUNCPTR(gstlal_inpaint_sink_event);
	transform_class->transform_ip = GST_DEBUG_FUNCPTR(gstlal_inpaint_transform_ip);

	gst_element_class_set_metadata(
		element_class,
		"Inpaint",
		"Filter",
		"A routine that replaces replaces glitchy data with data based on the surrounding times.",
		"Cody Messick <cody.messick@ligo.org>"
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
 * instance init
 */


static void gstlal_inpaint_init(GSTLALInpaint *inpaint)
{
	inpaint->instrument = NULL;
	inpaint->channel_name = NULL;
	inpaint->units = NULL;
}

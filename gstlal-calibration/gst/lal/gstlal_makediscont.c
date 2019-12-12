/*
 * Copyright (C) 2019 Aaron Viets <aaron.viets@ligo.org>
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation; either version 2 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */


/*
 * Stuff from C
 */


#include <math.h>
#include <stdlib.h>


/*
 * Stuff from GStreamer
 */


#include <gst/gst.h>


/* 
 * Our own stuff
 */


#include <gstlal/gstlal.h>
#include <gstlal/gstlal_debug.h>
#include <gstlal_makediscont.h>


/*
 * ============================================================================
 *
 *			     GStreamer Boiler Plate
 *
 * ============================================================================
 */


static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SINK_NAME,	/* macro for padname */
	GST_PAD_SINK,			/* direction, GST_PAD_SINK==2 */
	GST_PAD_ALWAYS,			/* presence, GST_PAD_ALWAYS==0 */
	GST_STATIC_CAPS("ANY")		/* the GST_STATIC_CAPS("ANY") macro */
);


static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SRC_NAME,
	GST_PAD_SRC,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS("ANY")
);


#define GST_CAT_DEFAULT gstlal_makediscont_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


G_DEFINE_TYPE_WITH_CODE(
	GSTLALMakeDiscont,
	gstlal_makediscont,
	GST_TYPE_ELEMENT,
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "lal_makediscont", 0, "lal_makediscont element")
);


/* Element properties */
enum property {
	ARG_DROPOUT_TIME = 1,
	ARG_DATA_TIME,
	ARG_FAKE
};


static GParamSpec *properties[ARG_FAKE];


/*
 * ============================================================================
 *
 *			       GstElement Overrides
 *
 * ============================================================================
 */


/* 
 * sink_event()
 */


static gboolean sink_event(GstPad * pad, GstObject * parent, GstEvent * event) {

	gboolean ret;

	switch (GST_EVENT_TYPE (event)) {
	case GST_EVENT_CAPS: ;
		GstCaps *caps;
		gst_event_parse_caps(event, &caps);
		ret = gst_pad_event_default(pad, parent, event);
		break;
	default:
		ret = gst_pad_event_default (pad, parent, event);
		break;
	}
	return ret;
}


/*
 * chain()
 */


static GstFlowReturn chain(GstPad *pad, GstObject *parent, GstBuffer *buf) {

	GSTLALMakeDiscont *element = GSTLAL_MAKEDISCONT(parent);

	/*
	 * check for discontinuity
	 */

	if(G_UNLIKELY(GST_BUFFER_IS_DISCONT(buf) || GST_BUFFER_OFFSET(buf) != element->next_in_offset || !GST_CLOCK_TIME_IS_VALID(element->t0))) {
		element->t0 = GST_BUFFER_PTS(buf);
		element->offset0 = GST_BUFFER_OFFSET(buf);
		element->current_data_state = TRUE;
		element->next_dropout_tstart = element->t0 + element->data_time;
	}
	element->next_in_offset = GST_BUFFER_OFFSET_END(buf);

	if(element->current_data_state) {
		/* Check if we need to start a dropout */
		if(GST_BUFFER_PTS(buf) >= element->next_dropout_tstart) {
			element->current_data_state = FALSE;
			element->next_data_tstart = GST_BUFFER_PTS(buf) + element->dropout_time;
		}
	} else {
		/* Check if we need to start a new data segment */
		if(GST_BUFFER_PTS(buf) >= element->next_data_tstart) {
			element->current_data_state = TRUE;
			element->next_dropout_tstart = GST_BUFFER_PTS(buf) + element->data_time;
		}
	}

	if(element->current_data_state)
		/* Push out the incoming buffer without touching it */
		return gst_pad_push(element->srcpad, buf);
	else
		/* Don't push a buffer, and trick GStreamer into thinking everything is ok */
		return GST_FLOW_OK;
}


/*
 * ============================================================================
 *
 *				GObject Overrides
 *
 * ============================================================================
 */


/* 
 * set_property()
 */


static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec) {

	GSTLALMakeDiscont *element = GSTLAL_MAKEDISCONT(object);
	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_DROPOUT_TIME:
		element->dropout_time = g_value_get_uint64(value);
		break;
	case ARG_DATA_TIME:
		element->data_time = g_value_get_uint64(value);
		break;
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * get_property()
 */


static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *pspec) {

	GSTLALMakeDiscont *element = GSTLAL_MAKEDISCONT(object);
	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_DROPOUT_TIME:
		g_value_set_uint64(value, element->dropout_time);
		break;
	case ARG_DATA_TIME:
		g_value_set_uint64(value, element->data_time);
		break;
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


/* 
 * class_init()
 */


static void gstlal_makediscont_class_init(GSTLALMakeDiscontClass * klass) {

	GObjectClass *gobject_class = (GObjectClass *) klass;
	GstElementClass *element_class = (GstElementClass *) klass;

	gst_element_class_set_details_simple(element_class,
		"Make discontinuity",
		"Filter",
		"Drops buffers in a stream to create a discontinuous stream",
		"Aaron Viets <aaron.viets@ligo.org>");

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));

	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);


	properties[ARG_DROPOUT_TIME] = g_param_spec_uint64(
		"dropout-time",
		"Dropout time",
		"The minimum length in nanoseconds of each data dropout.  An integer number of\n\t\t\t"
		"buffers will be dropped, and since buffers have nonzero length, the actual length\n\t\t\t"
		"of a dropout may be longer.  Default is 1 second.",
		0, G_MAXUINT64, 1000000000,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_DATA_TIME] = g_param_spec_uint64(
		"data-time",
		"Data time",
		"The minimum length in nanoseconds of each data segment that is not dropped.  Each\n\t\t\t"
		"data segment is an integer number of buffers, so the actual length of a segment\n\t\t\t"
		"may be longer.  The start of stream is not dropped.  Default is 10 seconds.",
		0, G_MAXUINT64, 10000000000,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);


	g_object_class_install_property(
		gobject_class,
		ARG_DROPOUT_TIME,
		properties[ARG_DROPOUT_TIME]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_DATA_TIME,
		properties[ARG_DATA_TIME]
	);
}


/*
 * init()
 */


static void gstlal_makediscont_init(GSTLALMakeDiscont * element) {

	/* Sink Pad */
	element->sinkpad = gst_pad_new_from_static_template(&sink_factory, "sink");
	gst_pad_set_event_function(element->sinkpad, GST_DEBUG_FUNCPTR(sink_event));
	gst_pad_set_chain_function(element->sinkpad, GST_DEBUG_FUNCPTR(chain));
	GST_PAD_SET_PROXY_CAPS(element->sinkpad);
	GST_PAD_SET_PROXY_ALLOCATION(element->sinkpad);
	GST_PAD_SET_PROXY_SCHEDULING(element->sinkpad);
	gst_element_add_pad(GST_ELEMENT(element), element->sinkpad);

	/* Source Pad */
	element->srcpad = gst_pad_new_from_static_template(&src_factory, "src");
	GST_PAD_SET_PROXY_CAPS(element->srcpad);
	GST_PAD_SET_PROXY_ALLOCATION(element->srcpad);
	GST_PAD_SET_PROXY_SCHEDULING(element->srcpad);
	gst_element_add_pad(GST_ELEMENT(element), element->srcpad);

	element->current_data_state = TRUE;
	element->next_dropout_tstart = G_MAXUINT64;
	element->next_data_tstart = G_MAXUINT64;
	element->t0 = GST_CLOCK_TIME_NONE;
	element->offset0 = GST_BUFFER_OFFSET_NONE;
	element->next_in_offset = GST_BUFFER_OFFSET_NONE;
}



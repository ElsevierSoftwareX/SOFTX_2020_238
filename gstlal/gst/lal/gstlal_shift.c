/*
 * An element to shift buffer time stamps.
 *
 * Copyright (C) 2009,2011  Kipp Cannon
 * Copyright (C) 2014 Chad Hanna
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
 * stuff from glib/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>


/*
 * our own stuff
 */


#include <gstlal/gstlal.h>
#include <gstlal_shift.h>


/*
 * ============================================================================
 *
 *                                Boilerplate
 *
 * ============================================================================
 */


G_DEFINE_TYPE(
	GSTLALShift,
	gstlal_shift,
	GST_TYPE_ELEMENT
);


/*
 * ============================================================================
 *
 *                                    Pads
 *
 * ============================================================================
 */


/*
 * Events
 */


static gboolean sink_event(GstPad *pad, GstObject *parent, GstEvent *event)
{
	GSTLALShift *shift = GSTLAL_SHIFT(parent);

	/*
	 * adjust segment events by +shift
	 */

	switch(GST_EVENT_TYPE(event)) {
	case GST_EVENT_SEGMENT: {
		GstSegment segment;
		gst_event_copy_segment(event, &segment);
		if(segment.format == GST_FORMAT_TIME) {
			if(GST_CLOCK_TIME_IS_VALID(segment.start))
				segment.start += shift->shift;
			if(GST_CLOCK_TIME_IS_VALID(segment.stop))
				segment.stop += shift->shift;
		}
		gst_event_unref(event);
		event = gst_event_new_segment(&segment);
		break;
	}

	default:
		break;
	}

	return gst_pad_event_default(pad, parent, event);
}


/* FIXME:  upstream queries for segments and position and so-on need to be
 * adjusted to.  oh well, who cares */

static gboolean src_event(GstPad *pad, GstObject *parent, GstEvent *event)
{
	GSTLALShift *shift = GSTLAL_SHIFT(parent);

	/*
	 * adjust seek events by -shift
	 */

	switch(GST_EVENT_TYPE(event)) {
	case GST_EVENT_SEEK: {
		gdouble rate;
		GstFormat format;
		GstSeekFlags flags;
		GstSeekType start_type, stop_type;
		gint64 start, stop;
		gst_event_parse_seek(event, &rate, &format, &flags, &start_type, &start, &stop_type, &stop);
		gst_event_unref(event);

		if(format == GST_FORMAT_TIME) {
			if(GST_CLOCK_TIME_IS_VALID(start))
				start -= shift->shift;
			if(GST_CLOCK_TIME_IS_VALID(stop))
				stop -= shift->shift;
		}

		event = gst_event_new_seek(rate, format, flags, start_type, start, stop_type, stop);
		break;
	}

	default:
		break;
	}

	/*
	 * invoke default handler
	 */

	return gst_pad_event_default(pad, parent, event);
}


/*
 * chain()
 */


static GstFlowReturn chain(GstPad *pad, GstObject *parent, GstBuffer *sinkbuf)
{
	GSTLALShift *element = GSTLAL_SHIFT(parent);
	GstFlowReturn result = GST_FLOW_OK;

	/*
	 * check validity of timestamp and offsets
	 */

	if(!GST_BUFFER_TIMESTAMP_IS_VALID(sinkbuf) || !GST_BUFFER_DURATION_IS_VALID(sinkbuf) || !GST_BUFFER_OFFSET_IS_VALID(sinkbuf) || !GST_BUFFER_OFFSET_END_IS_VALID(sinkbuf)) {
		gst_buffer_unref(sinkbuf);
		GST_ERROR_OBJECT(element, "error in input stream: buffer has invalid timestamp and/or offset");
		result = GST_FLOW_ERROR;
		goto done;
	}

	/* Check for underflow */
	if (((gint64) GST_BUFFER_TIMESTAMP(sinkbuf) + element->shift) >= 0)
		GST_BUFFER_TIMESTAMP(sinkbuf) = (GstClockTime) ( (gint64) GST_BUFFER_TIMESTAMP(sinkbuf) + element->shift );
	else
		g_error("Cannot shift buffer with time stamp %" G_GUINT64_FORMAT " by %" G_GINT64_FORMAT, GST_BUFFER_TIMESTAMP(sinkbuf), element->shift);

	/* Finally apply the discont flag if a new shift was detected */
	if (element->have_discont) {
		GST_BUFFER_FLAG_SET(sinkbuf, GST_BUFFER_FLAG_DISCONT);
		element->have_discont = FALSE;
	}

	result = gst_pad_push(element->srcpad, sinkbuf);
	if(G_UNLIKELY(result != GST_FLOW_OK))
		GST_WARNING_OBJECT(element, "Failed to push drain: %s", gst_flow_get_name(result));

	/*
	 * done
	 */

done:
	return result;
}


/*
 * ============================================================================
 *
 *                             GObject Overrides
 *
 * ============================================================================
 */


/*
 * properties
 */


enum property {
	ARG_SHIFT = 1
};


static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec)
{
	GSTLALShift *element = GSTLAL_SHIFT(object);
	gint64 newshift = 0;

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_SHIFT:
		newshift = g_value_get_int64(value);
		if (newshift != element->shift) {
			element->shift = newshift;
			element->have_discont = TRUE;
		}
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *pspec)
{
	GSTLALShift *element = GSTLAL_SHIFT(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_SHIFT:
		g_value_set_int64(value, element->shift);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * finalize()
 */


static void finalize(GObject *object)
{
	GSTLALShift *element = GSTLAL_SHIFT(object);

	gst_object_unref(element->sinkpad);
	element->sinkpad = NULL;
	gst_object_unref(element->srcpad);
	element->srcpad = NULL;

	G_OBJECT_CLASS(gstlal_shift_parent_class)->finalize(object);
}


/*
 * gstlal_shift_class_init()
 */


#define CAPS \
	GST_AUDIO_CAPS_MAKE(GSTLAL_AUDIO_FORMATS_ALL) ", " \
	"layout = (string) interleaved, " \
	"channel-mask = (bitmask) 0"


static void gstlal_shift_class_init(GSTLALShiftClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	g_object_class_install_property(
		gobject_class,
		ARG_SHIFT,
		g_param_spec_int64(
			"shift",
			"Shift nanoseconds",
			"number of nanoseconds to shift from the beginning of a stream",
			G_MININT64, G_MAXINT64, 0,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);

	gst_element_class_set_details_simple(
		element_class,
		"Shift",
		"Filter",
		"Shift the time stamp of buffers",
		"Chad Hanna <chad.hanna@ligo.org>, Kipp Cannon <kipp.cannon@ligo.org>"
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
 * gstlal_shift_init()
 */


static void gstlal_shift_init(GSTLALShift *element)
{
	GstPad *pad;

	gst_element_create_all_pads(GST_ELEMENT(element));

	/* configure (and ref) sink pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "sink");
	gst_pad_set_chain_function(pad, GST_DEBUG_FUNCPTR(chain));
	gst_pad_set_event_function(pad, GST_DEBUG_FUNCPTR(sink_event));
	GST_PAD_SET_PROXY_CAPS(pad);
	GST_PAD_SET_PROXY_ALLOCATION(pad);
	GST_PAD_SET_PROXY_SCHEDULING(pad);
	element->sinkpad = pad;

	/* retrieve (and ref) src pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "src");
	gst_pad_set_event_function(pad, GST_DEBUG_FUNCPTR(src_event));
	GST_PAD_SET_PROXY_CAPS(pad);
	GST_PAD_SET_PROXY_ALLOCATION(pad);
	GST_PAD_SET_PROXY_SCHEDULING(pad);
	element->srcpad = pad;

	/* internal data */
	element->shift = 0;
	element->have_discont = FALSE;
}

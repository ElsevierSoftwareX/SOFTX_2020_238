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
 * struff from the C library
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
 *                                 Properties
 *
 * ============================================================================
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
 * ============================================================================
 *
 *                                    Pads
 *
 * ============================================================================
 */


/*
 * getcaps()
 */


static GstCaps *getcaps(GSTLALShift *shift, GstPad * pad, GstCaps * filter)
{
	GstCaps *result, *peercaps, *current_caps, *filter_caps;

	/*
	 * take filter
	 */

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
	peercaps = gst_pad_peer_query_caps(shift->srcpad, filter_caps);

	/* get the allowed caps on this sinkpad */
	current_caps = gst_pad_get_pad_template_caps(pad);
	if(!current_caps)
		current_caps = gst_caps_new_any();

	if(peercaps) {
		/* if the peer has caps, intersect */
		GST_DEBUG_OBJECT(shift, "intersecting peer and our caps");
		result = gst_caps_intersect_full(peercaps, current_caps, GST_CAPS_INTERSECT_FIRST);
		/* neither peercaps nor current_caps are needed any more */
		gst_caps_unref(peercaps);
		gst_caps_unref(current_caps);
	} else {
		/* the peer has no caps (or there is no peer), just use the allowed caps
		* of this sinkpad. */
		/* restrict with filter-caps if any */
		if (filter_caps) {
			GST_DEBUG_OBJECT(shift, "no peer caps, using filtered caps");
			result = gst_caps_intersect_full(filter_caps, current_caps, GST_CAPS_INTERSECT_FIRST);
			/* current_caps are not needed any more */
			gst_caps_unref(current_caps);
		} else {
			GST_DEBUG_OBJECT(shift, "no peer caps, using our caps");
			result = current_caps;
		}
	}

	result = gst_caps_make_writable(result);

	if(filter_caps)
		gst_caps_unref(filter_caps);

	GST_LOG_OBJECT(shift, "getting caps on pad %p,%s to %" GST_PTR_FORMAT, pad, GST_PAD_NAME(pad), result);

	return result;
}


/*
 * setcaps()
 */


static gboolean setcaps(GSTLALShift *shift, GstPad *pad, GstCaps *caps)
{
	gboolean success = TRUE;

	/*
	 * try setting caps on downstream element
	 */

	success = gst_pad_set_caps(shift->srcpad, caps);

	/*
	 * update the element metadata
	 */

	return success;
}


/*
 * Events
 */


static gboolean sink_event(GstPad *pad, GstObject *parent, GstEvent *event)
{
	GSTLALShift *shift = GSTLAL_SHIFT(parent);
	GstSegment segment;
	GstCaps *caps;
	GstFormat format;
	gint64 start;
	gint64 stop;

	switch(GST_EVENT_TYPE(event)) {
	case GST_EVENT_SEGMENT:
		GST_DEBUG_OBJECT(pad, "new segment;  adjusting boundary");
		gst_event_copy_segment(event, &segment);

		if (format == GST_FORMAT_TIME && GST_CLOCK_TIME_IS_VALID(start) && GST_CLOCK_TIME_IS_VALID(stop)) {
			start += shift->shift;
			stop += shift->shift;
			if (! GST_CLOCK_TIME_IS_VALID(start))
				start = GST_CLOCK_TIME_NONE;
			if (! GST_CLOCK_TIME_IS_VALID(stop))
				stop = GST_CLOCK_TIME_NONE;
		}
		return gst_pad_push_event(shift->srcpad, gst_event_new_segment(&segment));

	case GST_EVENT_CAPS:
		gst_event_parse_caps(event, &caps);
		gst_event_unref(event);
		return setcaps(shift, pad, caps);

	default:
		break;
	}

	return gst_pad_event_default(pad, parent, event);
}


static gboolean src_event(GstPad *pad, GstObject *parent, GstEvent *event)
{
	GSTLALShift *shift = GSTLAL_SHIFT(parent);
	GstEvent *newevent = NULL;
	GstSegment segment;
	GstFormat format;
	gint64 start;
	gint64 stop;

	switch(GST_EVENT_TYPE(event)) {
	case GST_EVENT_SEGMENT:
		GST_DEBUG_OBJECT(pad, "new segment;  adjusting boundary");
		gst_event_copy_segment(event, &segment);

		if (format == GST_FORMAT_TIME && GST_CLOCK_TIME_IS_VALID(start) && GST_CLOCK_TIME_IS_VALID(stop)) {
			start += shift->shift;
			stop += shift->shift;
			if (! GST_CLOCK_TIME_IS_VALID(start))
				start = GST_CLOCK_TIME_NONE;
			if (! GST_CLOCK_TIME_IS_VALID(stop))
				stop = GST_CLOCK_TIME_NONE;
		}

		event = gst_event_new_segment(&segment);
		break;

	default:
		break;
	}

	/*
	 * sink events are forwarded to src pad
	 */

	if(newevent)
		return gst_pad_push_event(shift->sinkpad, newevent);
	else
		return gst_pad_push_event(shift->sinkpad, event);
}


static gboolean src_query(GstPad *pad, GstObject *parent, GstQuery *query)
{
	gboolean res = FALSE;

	switch(GST_QUERY_TYPE (query)) {
	default:
		res = gst_pad_query_default (pad, parent, query);
		break;
	}
	return res;
}


static gboolean sink_query(GstPad *pad, GstObject *parent, GstQuery * query)
{
	GSTLALShift *shift = GSTLAL_SHIFT(parent);
	gboolean res = TRUE;
	GstCaps *filter, *caps;

	switch(GST_QUERY_TYPE(query)) {
	case GST_QUERY_CAPS:
		gst_query_parse_caps(query, &filter);
		caps = getcaps(shift, pad, filter);
		gst_query_set_caps_result(query, caps);
		gst_caps_unref(caps);
		break;
	default:
		break;
	}

	if(G_LIKELY (query))
		return gst_pad_query_default(pad, parent, query);
	else
		return res;
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
 *                                Type Support
 *
 * ============================================================================
 */


/*
 * Parent class.
 */


static GstElementClass *gstlal_shift_parent_class = NULL;


/*
 * Instance finalize function.  See ???
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
 * Class init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GClassInitFunc
 */


#define CAPS \
	GST_AUDIO_CAPS_MAKE(GSTLAL_AUDIO_FORMATS_ALL) ", " \
	"layout = (string) interleaved, " \
	"channel-mask = (bitmask) 0"


static void class_init(gpointer class, gpointer class_data)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(class);
	GstElementClass *element_class = GST_ELEMENT_CLASS(class);

	gstlal_shift_parent_class = g_type_class_ref(GST_TYPE_ELEMENT);

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
 * Instance init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GInstanceInitFunc
 */


static void instance_init(GTypeInstance *object, gpointer class)
{
	GSTLALShift *element = GSTLAL_SHIFT(object);
	GstPad *pad;

	gst_element_create_all_pads(GST_ELEMENT(element));

	/* configure (and ref) sink pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "sink");
	gst_pad_set_chain_function(pad, GST_DEBUG_FUNCPTR(chain));
	gst_pad_set_query_function(pad, GST_DEBUG_FUNCPTR(sink_query));
	gst_pad_set_event_function(pad, GST_DEBUG_FUNCPTR(sink_event));
	element->sinkpad = pad;

	/* retrieve (and ref) src pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "src");
	gst_pad_set_query_function(pad, GST_DEBUG_FUNCPTR (src_query));
	gst_pad_set_event_function(pad, GST_DEBUG_FUNCPTR(src_event));
	element->srcpad = pad;

	/* internal data */
	element->shift = 0;
	element->have_discont = FALSE;
}


/*
 * gstlal_shift_get_type().
 */


GType gstlal_shift_get_type(void)
{
	static GType type = 0;

	if(!type) {
		static const GTypeInfo info = {
			.class_size = sizeof(GSTLALShiftClass),
			.class_init = class_init,
			.instance_size = sizeof(GSTLALShift),
			.instance_init = instance_init,
		};
		type = g_type_register_static(GST_TYPE_ELEMENT, "GSTLALShift", &info, 0);
	}

	return type;
}

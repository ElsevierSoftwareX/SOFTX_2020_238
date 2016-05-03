/*
 * An element to chop up audio buffers into smaller pieces.
 *
 * Copyright (C) 2009,2011  Kipp Cannon
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


/**
 * SECTION:gstlal_drop
 * @short_description:  Drop samples from the start of a stream.
 *
 * Reviewed:  185fc2b55190824ac79df11d4165d0d704d68464 2014-08-12 K.
 * Cannon, J.  Creighton, B. Sathyaprakash.
 *
 * Action:
 *
 * - write unit test
 *
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
#include <gst/audio/audio.h>


/*
 * our own stuff
 */


#include <gstlal/gstlal.h>
#include <gstlal_drop.h>


/*
 * ============================================================================
 *
 *                                 Utility functions
 *
 * ============================================================================
 */


static gboolean setcaps (GSTLALDrop *drop, GstPad *pad, GstCaps *caps)
{
	GstAudioInfo info;
	gboolean success = TRUE;

	/*
	 * parse caps
	 */

	success &= gst_audio_info_from_caps(&info, caps);

	/*
	 * try setting caps on downstream element
	 */

	if(success)
		success = gst_pad_set_caps(drop->srcpad, caps);

	/*
	 * update the element metadata
	 */

	if(success) {
		drop->rate = GST_AUDIO_INFO_RATE(&info);
		drop->unit_size = GST_AUDIO_INFO_WIDTH(&info) / 8 * GST_AUDIO_INFO_CHANNELS(&info);
	} else
		GST_ERROR_OBJECT(drop, "unable to parse and/or accept caps %" GST_PTR_FORMAT, caps);

	/*
	 * done
	 */

	return success;

}


/*
 * getcaps()
 */


static GstCaps *getcaps (GstPad * pad, GstCaps * filter)
{
	GSTLALDrop *drop;
	GstCaps *result, *peercaps, *current_caps, *filter_caps;
	drop = GSTLAL_DROP(GST_PAD_PARENT (pad));

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
	peercaps = gst_pad_peer_query_caps(drop->srcpad, filter_caps);

	/* get the allowed caps on this sinkpad */
	current_caps = gst_pad_get_pad_template_caps(pad);
	if (!current_caps)
			current_caps = gst_caps_new_any();

	if (peercaps) {
		/* if the peer has caps, intersect */
		GST_DEBUG_OBJECT(drop, "intersecting peer and our caps");
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
			GST_DEBUG_OBJECT(drop, "no peer caps, using filtered caps");
			result = gst_caps_intersect_full(filter_caps, current_caps, GST_CAPS_INTERSECT_FIRST);
			/* current_caps are not needed any more */
			gst_caps_unref(current_caps);
		}
		else {
			GST_DEBUG_OBJECT(drop, "no peer caps, using our caps");
			result = current_caps;
		}
	}

	result = gst_caps_make_writable (result);

	if (filter_caps)
		gst_caps_unref (filter_caps);

	GST_LOG_OBJECT (drop, "getting caps on pad %p,%s to %" GST_PTR_FORMAT, pad, GST_PAD_NAME(pad), result);

	return result;
}




/*
 * ============================================================================
 *
 *                                 Properties
 *
 * ============================================================================
 */


enum property {
	ARG_DROP_SAMPLES = 1
};


static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec)
{
	GSTLALDrop *element = GSTLAL_DROP(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_DROP_SAMPLES:
		element->drop_samples = g_value_get_uint(value);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *pspec)
{
	GSTLALDrop *element = GSTLAL_DROP(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_DROP_SAMPLES:
		g_value_set_uint(value, element->drop_samples);
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


static gboolean src_query(GstPad *pad, GstObject *parent, GstQuery *query)
{
	gboolean res = FALSE;

	switch(GST_QUERY_TYPE(query)) {
	default:
		res = gst_pad_query_default (pad, parent, query);
		break;
	}
	return res;
}


static gboolean src_event(GstPad *pad, GstObject *parent, GstEvent *event)
{
	GSTLALDrop *drop;
	gboolean result = TRUE;
	drop = GSTLAL_DROP (parent);
	GST_DEBUG_OBJECT (pad, "Got %s event on src pad", GST_EVENT_TYPE_NAME(event));

	switch(GST_EVENT_TYPE(event)) {
	default:
		/* just forward the rest for now */
		GST_DEBUG_OBJECT(drop, "forward unhandled event: %s", GST_EVENT_TYPE_NAME (event));
		gst_pad_event_default(pad, parent, event);
		break;
	}

	return result;
}


static gboolean sink_query(GstPad *pad, GstObject *parent, GstQuery * query)
{
	gboolean res = TRUE;
	GstCaps *filter, *caps;

	switch(GST_QUERY_TYPE(query)) {
	case GST_QUERY_CAPS:
		gst_query_parse_caps (query, &filter);
		caps = getcaps (pad, filter);
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
	GSTLALDrop *drop = GSTLAL_DROP(parent);
	gboolean res = TRUE;
	GstCaps *caps;

	GST_DEBUG_OBJECT(pad, "Got %s event on sink pad", GST_EVENT_TYPE_NAME (event));

	switch(GST_EVENT_TYPE(event)) {
	case GST_EVENT_CAPS:
		gst_event_parse_caps(event, &caps);
		res = setcaps(drop, pad, caps);
		gst_event_unref(event);
		event = NULL;

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


static GstFlowReturn chain(GstPad *pad, GstObject *parent, GstBuffer *sinkbuf)
{
	GSTLALDrop *element = GSTLAL_DROP(parent);
	GstFlowReturn result = GST_FLOW_OK;
	guint dropsize = (guint) element->drop_samples*element->unit_size;

	/*
	 * check validity of timestamp and offsets
	 */

	if(!GST_BUFFER_TIMESTAMP_IS_VALID(sinkbuf) || !GST_BUFFER_DURATION_IS_VALID(sinkbuf) || !GST_BUFFER_OFFSET_IS_VALID(sinkbuf) || !GST_BUFFER_OFFSET_END_IS_VALID(sinkbuf)) {
		gst_buffer_unref(sinkbuf);
		GST_ERROR_OBJECT(element, "error in input stream: buffer has invalid timestamp and/or offset");
		result = GST_FLOW_ERROR;
		goto done;
	}

	/*
	 * process buffer
	 */

	if(dropsize <= 0) {
		/* pass entire buffer */
		if(element->need_discont && !GST_BUFFER_IS_DISCONT(sinkbuf)) {
			sinkbuf = gst_buffer_make_writable(sinkbuf);
			GST_BUFFER_FLAG_SET(sinkbuf, GST_BUFFER_FLAG_DISCONT);
		}
		result = gst_pad_push(element->srcpad, sinkbuf);
		if(G_UNLIKELY(result != GST_FLOW_OK))
			GST_WARNING_OBJECT(element, "gst_pad_push() failed: %s", gst_flow_get_name(result));
		element->need_discont = FALSE;
	} else if(gst_buffer_get_size(sinkbuf) <= dropsize) {
		/* drop entire buffer */
		gst_buffer_unref(sinkbuf);
		element->drop_samples -= (GST_BUFFER_OFFSET_END(sinkbuf) - GST_BUFFER_OFFSET(sinkbuf));
		element->need_discont = TRUE;
		result = GST_FLOW_OK;
	} else {
		/* drop part of buffer, pass the rest */
		GstBuffer *srcbuf = gst_buffer_copy_region(sinkbuf, GST_BUFFER_COPY_FLAGS | GST_BUFFER_COPY_MEMORY | GST_BUFFER_COPY_TIMESTAMPS, dropsize, gst_buffer_get_size(sinkbuf) - dropsize);
		GstClockTime toff = gst_util_uint64_scale_int_round(element->drop_samples, GST_SECOND, element->rate);
		GST_BUFFER_OFFSET(srcbuf) = GST_BUFFER_OFFSET(sinkbuf) + element->drop_samples;
		GST_BUFFER_OFFSET_END(srcbuf) = GST_BUFFER_OFFSET_END(sinkbuf);
		GST_BUFFER_TIMESTAMP(srcbuf) = GST_BUFFER_TIMESTAMP(sinkbuf) + toff;
		GST_BUFFER_DURATION(srcbuf) = GST_BUFFER_DURATION(sinkbuf) - toff;
		GST_BUFFER_FLAG_SET(srcbuf, GST_BUFFER_FLAG_DISCONT);

		result = gst_pad_push(element->srcpad, srcbuf);
		if(G_UNLIKELY(result != GST_FLOW_OK))
			GST_WARNING_OBJECT(element, "gst_pad_push() failed: %s", gst_flow_get_name(result));
		/* never come back */
		element->drop_samples = 0;
		element->need_discont = FALSE;
		gst_buffer_unref(sinkbuf);
	}

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


static GstElementClass *gstlal_drop_parent_class = NULL;


/*
 * Instance finalize function.  See ???
 */


static void finalize(GObject *object)
{
	GSTLALDrop *element = GSTLAL_DROP(object);

	gst_object_unref(element->sinkpad);
	element->sinkpad = NULL;
	gst_object_unref(element->srcpad);
	element->srcpad = NULL;

	G_OBJECT_CLASS(gstlal_drop_parent_class)->finalize(object);
}


/*
 * class_init()
 */


#define CAPS GST_AUDIO_CAPS_MAKE(GSTLAL_AUDIO_FORMATS_ALL)


static void class_init(gpointer class, gpointer class_data)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(class);
	GObjectClass *gobject_class = G_OBJECT_CLASS(class);

	gst_element_class_set_details_simple(
		element_class,
		"Drop",
		"Filter",
		"Drop samples from the start of a stream",
		"Kipp Cannon <kipp.cannon@ligo.org>"
	);

	gstlal_drop_parent_class = g_type_class_ref(GST_TYPE_ELEMENT);

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
		ARG_DROP_SAMPLES,
		g_param_spec_uint(
			"drop-samples",
			"Drop samples",
			"number of samples to drop from the beginning of a stream",
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
	GSTLALDrop *element = GSTLAL_DROP(object);
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
	element->unit_size = 0;
	element->need_discont = TRUE;
}


/*
 * gstlal_drop_get_type().
 */


GType gstlal_drop_get_type(void)
{
	static GType type = 0;

	if(!type) {
		static const GTypeInfo info = {
			.class_size = sizeof(GSTLALDropClass),
			.class_init = class_init,
			.instance_size = sizeof(GSTLALDrop),
			.instance_init = instance_init,
		};
		type = g_type_register_static(GST_TYPE_ELEMENT, "GSTLALDrop", &info, 0);
	}

	return type;
}

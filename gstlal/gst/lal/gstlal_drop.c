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
 *                                Boilerplate
 *
 * ============================================================================
 */


G_DEFINE_TYPE(
	GSTLALDrop,
	gstlal_drop,
	GST_TYPE_ELEMENT
);


/*
 * ============================================================================
 *
 *                                    Pads
 *
 * ============================================================================
 */


static gboolean sink_event(GstPad *pad, GstObject *parent, GstEvent *event)
{
	GSTLALDrop *drop = GSTLAL_DROP(parent);
	gboolean success = TRUE;

	GST_DEBUG_OBJECT(pad, "Got %s event on sink pad", GST_EVENT_TYPE_NAME (event));

	switch(GST_EVENT_TYPE(event)) {
	case GST_EVENT_CAPS: {
		GstCaps *caps;
		GstAudioInfo info;
		gst_event_parse_caps(event, &caps);
		success = gst_audio_info_from_caps(&info, caps);
		if(success) {
			drop->rate = GST_AUDIO_INFO_RATE(&info);
			drop->unit_size = GST_AUDIO_INFO_BPF(&info);
		}
		break;
	}

	default:
		break;
	}

	if(!success)
		gst_event_unref(event);
	else
		success = gst_pad_event_default(pad, parent, event);

	return success;
}


/*
 * chain()
 */


static GstFlowReturn chain(GstPad *pad, GstObject *parent, GstBuffer *sinkbuf)
{
	GSTLALDrop *element = GSTLAL_DROP(parent);
	GstFlowReturn result = GST_FLOW_OK;

	/*
	 * check validity of timestamp and offsets
	 */

	if(!(
		GST_BUFFER_PTS_IS_VALID(sinkbuf) &&
		GST_BUFFER_DURATION_IS_VALID(sinkbuf) &&
		GST_BUFFER_OFFSET_IS_VALID(sinkbuf) &&
		GST_BUFFER_OFFSET_END_IS_VALID(sinkbuf) &&
		(GST_BUFFER_OFFSET_END(sinkbuf) - GST_BUFFER_OFFSET(sinkbuf)) * element->unit_size == gst_buffer_get_size(sinkbuf)
	)) {
		gst_buffer_unref(sinkbuf);
		GST_ELEMENT_ERROR(element, STREAM, FORMAT, (NULL), ("buffer has invalid timestamp and/or offset, or has sample count/size mismatch"));
		result = GST_FLOW_ERROR;
		goto done;
	}

	/*
	 * process buffer
	 */

	if(element->drop_samples <= 0) {
		/* pass entire buffer */
		if(element->need_discont && !GST_BUFFER_IS_DISCONT(sinkbuf)) {
			sinkbuf = gst_buffer_make_writable(sinkbuf);
			GST_BUFFER_FLAG_SET(sinkbuf, GST_BUFFER_FLAG_DISCONT);
		}
		result = gst_pad_push(element->srcpad, sinkbuf);
		if(G_UNLIKELY(result != GST_FLOW_OK))
			GST_WARNING_OBJECT(element, "gst_pad_push() failed: %s", gst_flow_get_name(result));
		element->need_discont = FALSE;
		result = GST_FLOW_OK;
	} else if(gst_buffer_get_size(sinkbuf) <= element->drop_samples * element->unit_size) {
		/* drop entire buffer */
		element->drop_samples -= GST_BUFFER_OFFSET_END(sinkbuf) - GST_BUFFER_OFFSET(sinkbuf);
		gst_buffer_unref(sinkbuf);
		element->need_discont = TRUE;
	} else {
		/* drop part of buffer, pass the rest */
		GstClockTime toff = gst_util_uint64_scale_int_round(element->drop_samples, GST_SECOND, element->rate);
		sinkbuf = gst_buffer_make_writable(sinkbuf);
		gst_buffer_resize(sinkbuf, element->drop_samples * element->unit_size, -1);
		GST_BUFFER_OFFSET(sinkbuf) += element->drop_samples;
		GST_BUFFER_PTS(sinkbuf) += toff;
		GST_BUFFER_DURATION(sinkbuf) -= toff;
		GST_BUFFER_FLAG_SET(sinkbuf, GST_BUFFER_FLAG_DISCONT);

		result = gst_pad_push(element->srcpad, sinkbuf);
		if(G_UNLIKELY(result != GST_FLOW_OK))
			GST_WARNING_OBJECT(element, "gst_pad_push() failed: %s", gst_flow_get_name(result));
		/* never come back */
		element->drop_samples = 0;
		element->need_discont = FALSE;
		result = GST_FLOW_OK;
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
 *                             GObject Overrides
 *
 * ============================================================================
 */


/*
 * Properties
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
 * finalize()
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


#define CAPS \
	GST_AUDIO_CAPS_MAKE(GSTLAL_AUDIO_FORMATS_ALL) ", " \
	"layout = (string) interleaved, " \
	"channel-mask = (bitmask) 0"


static void gstlal_drop_class_init(GSTLALDropClass *klass)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

	gst_element_class_set_details_simple(
		element_class,
		"Drop",
		"Filter",
		"Drop samples from the start of a stream",
		"Kipp Cannon <kipp.cannon@ligo.org>"
	);

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
 * gstlal_drop_init()
 */


static void gstlal_drop_init(GSTLALDrop *element)
{
	GstPad *pad;

	gst_element_create_all_pads(GST_ELEMENT(element));

	/* configure (and ref) sink pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "sink");
	gst_pad_set_event_function(pad, GST_DEBUG_FUNCPTR(sink_event));
	gst_pad_set_chain_function(pad, GST_DEBUG_FUNCPTR(chain));
	GST_PAD_SET_PROXY_ALLOCATION(pad);
	GST_PAD_SET_PROXY_CAPS(pad);
	GST_PAD_SET_PROXY_SCHEDULING(pad);
	element->sinkpad = pad;

	/* retrieve (and ref) src pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "src");
	GST_PAD_SET_PROXY_ALLOCATION(pad);
	GST_PAD_SET_PROXY_CAPS(pad);
	GST_PAD_SET_PROXY_SCHEDULING(pad);
	element->srcpad = pad;

	/* internal data */
	element->rate = 0;
	element->unit_size = 0;
	element->need_discont = TRUE;
}

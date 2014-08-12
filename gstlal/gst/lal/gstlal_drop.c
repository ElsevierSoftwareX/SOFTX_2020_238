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


#include <gstlal_drop.h>


/*
 * ============================================================================
 *
 *                                 Parameters
 *
 * ============================================================================
 */




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


/*
 * getcaps()
 */


static GstCaps *getcaps(GstPad * pad)
{
	GSTLALDrop *element = GSTLAL_DROP(gst_pad_get_parent(pad));
	GstPad *otherpad = pad == element->srcpad ? element->sinkpad : element->srcpad;
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

	peercaps = gst_pad_peer_get_caps_reffed(otherpad);
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
 * acceptcaps()
 */


static gboolean acceptcaps(GstPad *pad, GstCaps *caps)
{
	GSTLALDrop *element = GSTLAL_DROP(gst_pad_get_parent(pad));
	GstPad *otherpad = pad == element->srcpad ? element->sinkpad : element->srcpad;
	gboolean success;

	/*
	 * ask downstream peer
	 */

	success = gst_pad_peer_accept_caps(otherpad, caps);

	/*
	 * done
	 */

	gst_object_unref(element);
	return success;
}


/*
 * setcaps()
 */


static gboolean setcaps(GstPad *pad, GstCaps *caps)
{
	GSTLALDrop *element = GSTLAL_DROP(gst_pad_get_parent(pad));
	GstStructure *structure;
	gint rate, width, channels;
	gboolean success = TRUE;

	/*
	 * parse caps
	 */

	structure = gst_caps_get_structure(caps, 0);
	success &= gst_structure_get_int(structure, "rate", &rate);
	success &= gst_structure_get_int(structure, "width", &width);
	success &= gst_structure_get_int(structure, "channels", &channels);

	/*
	 * try setting caps on downstream element
	 */

	if(success)
		success = gst_pad_set_caps(element->srcpad, caps);

	/*
	 * update the element metadata
	 */

	if(success) {
		element->rate = rate;
		element->unit_size = width / 8 * channels;
	} else
		GST_ERROR_OBJECT(element, "unable to parse and/or accept caps %" GST_PTR_FORMAT, caps);

	/*
	 * done
	 */

	gst_object_unref(element);
	return success;
}


/*
 * chain()
 */


static GstFlowReturn chain(GstPad *pad, GstBuffer *sinkbuf)
{
	GSTLALDrop *element = GSTLAL_DROP(gst_pad_get_parent(pad));
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

	if(!dropsize) {
		/* pass entire buffer */
		if(element->need_discont && !GST_BUFFER_IS_DISCONT(sinkbuf)) {
			sinkbuf = gst_buffer_make_metadata_writable(sinkbuf);
			GST_BUFFER_FLAG_SET(sinkbuf, GST_BUFFER_FLAG_DISCONT);
		}
		result = gst_pad_push(element->srcpad, sinkbuf);
		if(G_UNLIKELY(result != GST_FLOW_OK))
			GST_WARNING_OBJECT(element, "gst_pad_push() failed: %s", gst_flow_get_name(result));
		element->need_discont = FALSE;
	} else if(GST_BUFFER_SIZE(sinkbuf) <= dropsize) {
		/* drop entire buffer */
		gst_buffer_unref(sinkbuf);
		element->drop_samples -= GST_BUFFER_OFFSET_END(sinkbuf) - GST_BUFFER_OFFSET(sinkbuf);
		element->need_discont = TRUE;
		result = GST_FLOW_OK;
	} else {
		/* drop part of buffer, pass the rest */
		GstBuffer *srcbuf = gst_buffer_create_sub(sinkbuf, dropsize, GST_BUFFER_SIZE(sinkbuf) - dropsize);
		GstClockTime toff = gst_util_uint64_scale_int_round(element->drop_samples, GST_SECOND, element->rate);
		gst_buffer_unref(sinkbuf);
		gst_buffer_copy_metadata(srcbuf, sinkbuf, GST_BUFFER_COPY_FLAGS | GST_BUFFER_COPY_CAPS);
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
	}

	/*
	 * done
	 */

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
	GSTLALDrop *element = GSTLAL_DROP(object);

	gst_object_unref(element->sinkpad);
	element->sinkpad = NULL;
	gst_object_unref(element->srcpad);
	element->srcpad = NULL;

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


/*
 * Base init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GBaseInitFunc
 */


#define CAPS \
	"audio/x-raw-int, " \
	"rate = (int) [1, MAX], " \
	"channels = (int) [1, MAX], " \
	"endianness = (int) BYTE_ORDER, " \
	"width = (int) {8, 16, 32, 64}, " \
	"signed = (boolean) {true, false}; " \
	"audio/x-raw-float, " \
	"rate = (int) [1, MAX], " \
	"channels = (int) [1, MAX], " \
	"endianness = (int) BYTE_ORDER, " \
	"width = (int) {32, 64}; " \
	"audio/x-raw-complex, " \
	"rate = (int) [1, MAX], " \
	"channels = (int) [1, MAX], " \
	"endianness = (int) BYTE_ORDER, " \
	"width = (int) {64, 128}"


/*
 * base_init()
 */


static void base_init(gpointer class)
{
}


/*
 * class_init()
 */


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

	parent_class = g_type_class_ref(GST_TYPE_ELEMENT);

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
	gst_pad_set_getcaps_function(pad, GST_DEBUG_FUNCPTR(getcaps));
	gst_pad_set_acceptcaps_function(pad, GST_DEBUG_FUNCPTR(acceptcaps));
	gst_pad_set_setcaps_function(pad, GST_DEBUG_FUNCPTR(setcaps));
	gst_pad_set_chain_function(pad, GST_DEBUG_FUNCPTR(chain));
	element->sinkpad = pad;

	/* retrieve (and ref) src pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "src");
	gst_pad_set_getcaps_function(pad, GST_DEBUG_FUNCPTR(getcaps));
	gst_pad_set_acceptcaps_function(pad, GST_DEBUG_FUNCPTR(acceptcaps));
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
			.base_init = base_init,
			.instance_size = sizeof(GSTLALDrop),
			.instance_init = instance_init,
		};
		type = g_type_register_static(GST_TYPE_ELEMENT, "GSTLALDrop", &info, 0);
	}

	return type;
}

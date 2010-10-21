/*
 * An element to chop up audio buffers into smaller pieces.
 *
 * Copyright (C) 2009  Kipp Cannon
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


#include <gstlal_reblock.h>


/*
 * ============================================================================
 *
 *                                 Parameters
 *
 * ============================================================================
 */


#define DEFAULT_BLOCK_DURATION GST_SECOND


/*
 * ============================================================================
 *
 *                                 Properties
 *
 * ============================================================================
 */


enum property {
	ARG_BLOCK_DURATION = 1
};


static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec)
{
	GSTLALReblock *element = GSTLAL_REBLOCK(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_BLOCK_DURATION:
		element->block_duration = g_value_get_uint64(value);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *pspec)
{
	GSTLALReblock *element = GSTLAL_REBLOCK(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_BLOCK_DURATION:
		g_value_set_uint64(value, element->block_duration);
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
	GSTLALReblock *element = GSTLAL_REBLOCK(gst_pad_get_parent(pad));
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
	GSTLALReblock *element = GSTLAL_REBLOCK(gst_pad_get_parent(pad));
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


static GstFlowReturn chain(GstPad *pad, GstBuffer *sinkbuf)
{
	GSTLALReblock *element = GSTLAL_REBLOCK(gst_pad_get_parent(pad));
	guint64 offset;
	guint64 length, block_length;
	GstFlowReturn result = GST_FLOW_OK;

	/*
	 * check validity of timestamp and offsets
	 */

	if(!GST_BUFFER_TIMESTAMP_IS_VALID(sinkbuf) || !GST_BUFFER_DURATION_IS_VALID(sinkbuf) || !GST_BUFFER_OFFSET_IS_VALID(sinkbuf) || !GST_BUFFER_OFFSET_END_IS_VALID(sinkbuf)) {
		GST_ERROR_OBJECT(element, "error in input stream: buffer has invalid timestamp and/or offset");
		result = GST_FLOW_ERROR;
		goto done;
	}

	length = GST_BUFFER_OFFSET_END(sinkbuf) - GST_BUFFER_OFFSET(sinkbuf);

	/*
	 * compute the block length
	 */

	if(GST_BUFFER_DURATION(sinkbuf) <= element->block_duration) {
		/*
		 * buffer is already small enough, push down stream
		 */

		gst_buffer_ref(sinkbuf);	/* we'll unref it again later */
		result = gst_pad_push(element->srcpad, sinkbuf);
		if(G_UNLIKELY(result != GST_FLOW_OK))
			GST_WARNING_OBJECT(element, "Failed to push drain: %s", gst_flow_get_name (result));
		goto done;
	} else {
		guint64 n = (GST_BUFFER_DURATION(sinkbuf) + element->block_duration - 1) / element->block_duration;	/* ciel */
		block_length = (length + n - 1) / n;	/* ciel */
	}

	/*
	 * loop over the contents of the input buffer
	 */

	for(offset = 0; offset < length; offset += block_length) {
		GstBuffer *srcbuf;

		if(length - offset < block_length)
			block_length = length - offset;

		srcbuf = gst_buffer_create_sub(sinkbuf, offset * element->unit_size, block_length * element->unit_size);
		if(!srcbuf) {
			GST_ERROR_OBJECT(element, "failure creating sub-buffer");
			result = GST_FLOW_ERROR;
			goto done;
		}

		/*
		 * set flags, caps, offset, and timestamps.
		 */

		gst_buffer_copy_metadata(srcbuf, sinkbuf, GST_BUFFER_COPY_FLAGS | GST_BUFFER_COPY_CAPS);
		GST_BUFFER_OFFSET(srcbuf) = GST_BUFFER_OFFSET(sinkbuf) + offset;
		GST_BUFFER_OFFSET_END(srcbuf) = GST_BUFFER_OFFSET(srcbuf) + block_length;
		GST_BUFFER_TIMESTAMP(srcbuf) = GST_BUFFER_TIMESTAMP(sinkbuf) + gst_util_uint64_scale_int_round(GST_BUFFER_DURATION(sinkbuf), offset, length);
		GST_BUFFER_DURATION(srcbuf) = GST_BUFFER_TIMESTAMP(sinkbuf) + gst_util_uint64_scale_int_round(GST_BUFFER_DURATION(sinkbuf), offset + block_length, length) - GST_BUFFER_TIMESTAMP(srcbuf);

		/*
		 * only the first subbuffer of a buffer flagged as a
		 * discontinuity is a discontinuity.
		 */

		if(offset)
			GST_BUFFER_FLAG_UNSET(srcbuf, GST_BUFFER_FLAG_DISCONT);

		/*
		 * push buffer down stream
		 */

		result = gst_pad_push(element->srcpad, srcbuf);
		if(G_UNLIKELY(result != GST_FLOW_OK)) {
			GST_WARNING_OBJECT(element, "Failed to push drain: %s", gst_flow_get_name (result));
			goto done;
		}
	}

	/*
	 * done
	 */

done:
	gst_buffer_unref(sinkbuf);
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
	GSTLALReblock *element = GSTLAL_REBLOCK(object);

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


static void base_init(gpointer class)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(class);

	gst_element_class_set_details_simple(
		element_class,
		"Reblock",
		"Filter",
		"Chop audio buffers into smaller pieces",
		"Kipp Cannon <kipp.cannon@ligo.org>"
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
		ARG_BLOCK_DURATION,
		g_param_spec_uint64(
			"block-duration",
			"Block duration",
			"Desired output buffer duration in nanoseconds.",
			0, G_MAXUINT64, DEFAULT_BLOCK_DURATION,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
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
	GSTLALReblock *element = GSTLAL_REBLOCK(object);
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
	element->block_duration = DEFAULT_BLOCK_DURATION;
	element->rate = 0;
	element->unit_size = 0;
}


/*
 * gstlal_reblock_get_type().
 */


GType gstlal_reblock_get_type(void)
{
	static GType type = 0;

	if(!type) {
		static const GTypeInfo info = {
			.class_size = sizeof(GSTLALReblockClass),
			.class_init = class_init,
			.base_init = base_init,
			.instance_size = sizeof(GSTLALReblock),
			.instance_init = instance_init,
		};
		type = g_type_register_static(GST_TYPE_ELEMENT, "lal_reblock", &info, 0);
	}

	return type;
}

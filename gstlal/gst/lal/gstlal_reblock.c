/*
 * An element to chop up audio buffers into smaller pieces.
 *
 * Copyright (C) 2009,2011,2013  Kipp Cannon
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
 *                           Gstreamer Boilerplate
 *
 * ============================================================================
 */


GST_BOILERPLATE(
	GSTLALReblock,
	gstlal_reblock,
	GstElement,
	GST_TYPE_ELEMENT
);


/*
 * ============================================================================
 *
 *                                 Parameters
 *
 * ============================================================================
 */


#define DEFAULT_BLOCK_DURATION (1 * GST_SECOND)


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

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
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
	GSTLALReblock *element = GSTLAL_REBLOCK(gst_pad_get_parent(pad));
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
	GSTLALReblock *element = GSTLAL_REBLOCK(gst_pad_get_parent(pad));
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
	GSTLALReblock *element = GSTLAL_REBLOCK(gst_pad_get_parent(pad));
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
	GSTLALReblock *element = GSTLAL_REBLOCK(gst_pad_get_parent(pad));
	guint64 offset, length;
	guint64 blocks, block_length;
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

	/*
	 * if buffer is already small enough, push down stream
	 */


	if(GST_BUFFER_DURATION(sinkbuf) <= element->block_duration) {
		/* consumes reference */
		result = gst_pad_push(element->srcpad, sinkbuf);
		if(G_UNLIKELY(result != GST_FLOW_OK))
			GST_WARNING_OBJECT(element, "Failed to push drain: %s", gst_flow_get_name(result));
		goto done;
	}

	/*
	 * compute the block length
	 */

	blocks = (GST_BUFFER_DURATION(sinkbuf) + element->block_duration - 1) / element->block_duration;	/* ciel */
	g_assert_cmpuint(blocks, >, 0);	/* guaranteed by check for short-buffers above */
	length = GST_BUFFER_OFFSET_END(sinkbuf) - GST_BUFFER_OFFSET(sinkbuf);
	block_length = (length + blocks - 1) / blocks;	/* ciel */
	g_assert_cmpuint(block_length, >, 0);	/* barf to avoid infinite loop */

	/*
	 * loop over the contents of the input buffer
	 */

	for(offset = 0; offset < length; offset += block_length) {
		GstBuffer *srcbuf;

		/*
		 * extract sub-buffer
		 */

		if(length - offset < block_length)
			block_length = length - offset;

		srcbuf = gst_buffer_create_sub(sinkbuf, offset * element->unit_size, block_length * element->unit_size);
		if(G_UNLIKELY(!srcbuf)) {
			GST_ERROR_OBJECT(element, "failure creating sub-buffer");
			result = GST_FLOW_ERROR;
			break;
		}

		/*
		 * tweak for buffers that don't have data in them
		 *
		 * FIXME:  gst_buffer_create_sub() needs to check for this
		 * itself if gstreamer is going to allow such buffers.
		 * submit a patch?
		 */

		if(!GST_BUFFER_DATA(sinkbuf))
			GST_BUFFER_DATA(srcbuf) = NULL;

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
			GST_WARNING_OBJECT(element, "Failed to push drain: %s", gst_flow_get_name(result));
			break;
		}
	}
	gst_buffer_unref(sinkbuf);

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
 * Instance finalize function.
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
 * Base init function.
 */


static void gstlal_reblock_base_init(gpointer klass)
{
}


/*
 * Class init function.
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


static void gstlal_reblock_class_init(GSTLALReblockClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	gst_element_class_set_details_simple(
		element_class,
		"Reblock",
		"Filter",
		"Chop audio buffers into smaller pieces to enforce a maximum allowed buffer duration",
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

	g_object_class_install_property(
		gobject_class,
		ARG_BLOCK_DURATION,
		g_param_spec_uint64(
			"block-duration",
			"Block duration",
			"Maximum output buffer duration in nanoseconds.  Buffers may be smaller than this.",
			0, G_MAXUINT64, DEFAULT_BLOCK_DURATION,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
}


/*
 * Instance init function.
 */


static void gstlal_reblock_init(GSTLALReblock *element, GSTLALReblockClass *Klass)
{
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
}

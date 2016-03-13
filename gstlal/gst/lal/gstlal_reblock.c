/*
 * An element to chop up audio buffers into smaller pieces.
 *
 * Copyright (C) 2009,2011,2013,2015  Kipp Cannon
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
 * SECTION:gstlal_reblock
 * @short_description:  Chop audio buffers into smaller pieces to enforce a maximum allowed buffer duration.
 *
 * Input buffers whose duration is longer than #block-duration are split
 * into two or more buffers so that no output buffer is longer than
 * #block-duration.  This element is a no-op for buffers not longer than
 * #block-duration.
 *
 * If the configured #block-duration does not correspond to an integer
 * number of samples of the stream, the number of samples placed into each
 * output buffer is the ceiling of the count of samples corresponding to
 * #block-duration.  In particular this means that when a buffer is split,
 * each output buffer contains at least 1 sample regardless of the sample
 * rate or #block-duration.
 *
 * Reviewed:  2affb49291b24e189afd23d1fd56690e223845b6 2014-08-12 K.
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
#include <gst/audio/audio.h>


/*
 * our own stuff
 */


#include <gstlal/gstlal.h>
#include <gstlal/gstlal_debug.h>
#include <gstlal_reblock.h>


/*
 * ============================================================================
 *
 *                           Gstreamer Boilerplate
 *
 * ============================================================================
 */


#define GST_CAT_DEFAULT gstlal_reblock_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


G_DEFINE_TYPE_WITH_CODE(
	GSTLALReblock,
	gstlal_reblock,
	GST_TYPE_ELEMENT,
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "lal_reblock", 0, "lal_reblock element")
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
 *                                    Pads
 *
 * ============================================================================
 */


/*
 * setcaps()
 */


static gboolean setcaps(GSTLALReblock *reblock, GstPad *pad, GstCaps *caps)
{
	GstAudioInfo info;
	gboolean success = TRUE;

	/*
	 * parse caps
	 * NOTE: rate, width and channels must be present
	 */

	success &= gst_audio_info_from_caps(&info, caps);

	/*
	 * try setting caps on downstream element
	 */

	if(success)
		success = gst_pad_set_caps(reblock->srcpad, caps);

	/*
	 * update the element metadata
	 */

	if(success) {
		reblock->rate = GST_AUDIO_INFO_RATE(&info);
		reblock->unit_size = GST_AUDIO_INFO_BPF(&info);
	}

	/*
	 * done
	 */

	return success;
}


/*
 * ============================================================================
 *
 *                                Event and Query
 *
 * ============================================================================
 */


static gboolean src_query(GstPad *pad, GstObject * parent, GstQuery * query)
{
	gboolean ret;
	GSTLALReblock *reblock = GSTLAL_REBLOCK(parent);

	switch (GST_QUERY_TYPE(query)) {
	case GST_QUERY_CAPS:
	{
		GstCaps *temp, *caps, *filt, *tcaps = NULL;
		/* Get the other pads caps */
		caps = gst_pad_get_current_caps(reblock->sinkpad);
		if (!caps)
			caps = gst_pad_get_pad_template_caps(reblock->sinkpad);
		/* Get the filter caps */
		gst_query_parse_caps(query, &filt);

		/* make sure we only return results that intersect our padtemplate */
		tcaps = gst_pad_get_pad_template_caps(pad);
		temp = gst_caps_intersect(caps, tcaps);
		gst_caps_unref(caps);
		gst_caps_unref(tcaps);
		caps = temp;
		/* filter against the query filter when needed */
		if (filt) {
			temp = gst_caps_intersect(caps, filt);
			gst_caps_unref(caps);
			caps = temp;
		}
		gst_query_set_caps_result(query, caps);
		gst_caps_unref(caps);
		ret = TRUE;
		break;
	}
	default:
		ret = gst_pad_query_default(pad, parent, query);
		break;
	}
	return ret;
}


static gboolean sink_query(GstPad *pad, GstObject * parent, GstQuery * query)
{
	gboolean ret;
	GSTLALReblock *reblock = GSTLAL_REBLOCK(parent);

	switch (GST_QUERY_TYPE(query)) {
	case GST_QUERY_CAPS:
	{
		GstCaps *temp, *caps, *filt, *tcaps;

		caps = gst_pad_get_allowed_caps(reblock->srcpad);
		/* If the caps are NULL, there is probably not a peer yet */
		if (!caps) {
			caps = gst_pad_get_pad_template_caps(reblock->srcpad);
		}
		gst_query_parse_caps(query, &filt);

		/* make sure we only return results that intersect our padtemplate */
		tcaps = gst_pad_get_pad_template_caps(pad);
		if (tcaps) {
			temp = gst_caps_intersect(caps, tcaps);
			gst_caps_unref(caps);
			gst_caps_unref(tcaps);
			caps = temp;
		}
		/* filter against the query filter when needed */
		if (filt) {
			temp = gst_caps_intersect(caps, filt);
			gst_caps_unref(caps);
			caps = temp;
		}
		gst_query_set_caps_result(query, caps);
		gst_caps_unref(caps);
		ret = TRUE;
		break;
	}
	default:
		ret = gst_pad_query_default(pad, parent, query);
		break;
	}
	return ret;
}


static gboolean src_event(GstPad *pad, GstObject *parent, GstEvent *event)
{
	GSTLALReblock *reblock = GSTLAL_REBLOCK(parent);
	gboolean result = TRUE;
	GST_DEBUG_OBJECT (pad, "Got %s event on src pad", GST_EVENT_TYPE_NAME(event));

	switch(GST_EVENT_TYPE(event)) {
	default:
		/* just forward the rest for now */
		GST_DEBUG_OBJECT(reblock, "forward unhandled event: %s", GST_EVENT_TYPE_NAME (event));
		gst_pad_event_default(pad, parent, event);
		break;
	}

	return result;
}


static gboolean sink_event(GstPad *pad, GstObject *parent, GstEvent *event)
{
	GSTLALReblock *reblock = GSTLAL_REBLOCK(parent);
	gboolean res = TRUE;
	GstCaps *caps;

	GST_DEBUG_OBJECT(pad, "Got %s event on sink pad", GST_EVENT_TYPE_NAME (event));

	switch(GST_EVENT_TYPE(event)) {
	case GST_EVENT_CAPS:
		gst_event_parse_caps(event, &caps);
		res = setcaps(reblock, pad, caps);
		gst_event_unref(event);
		event = NULL;
	default:
		break;
	}

	if(G_LIKELY(event))
		return gst_pad_event_default(pad, parent, event);
	else
		return res;
}


/*
 * chain()
 */


static GstFlowReturn chain(GstPad *pad, GstObject *parent, GstBuffer *sinkbuf)
{
	GSTLALReblock *element = GSTLAL_REBLOCK(parent);
	guint64 offset, length;
	guint64 blocks, block_length;
	GstFlowReturn result = GST_FLOW_OK;

	GST_DEBUG_OBJECT(element, "received %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(sinkbuf));

	/*
	 * if buffer is already small enough or if it doesn't possess
	 * valid metadata, push down stream
	 */

	if(!(GST_BUFFER_TIMESTAMP_IS_VALID(sinkbuf) && GST_BUFFER_DURATION_IS_VALID(sinkbuf) && GST_BUFFER_OFFSET_IS_VALID(sinkbuf) && GST_BUFFER_OFFSET_END_IS_VALID(sinkbuf)) || GST_BUFFER_DURATION(sinkbuf) <= element->block_duration) {
		GST_DEBUG_OBJECT(element, "pushing verbatim");
		/* consumes reference */
		result = gst_pad_push(element->srcpad, sinkbuf);
		if(G_UNLIKELY(result != GST_FLOW_OK))
			GST_WARNING_OBJECT(element, "push failed: %s", gst_flow_get_name(result));
		goto done;
	}

	/*
	 * compute the block length
	 */

	blocks = (GST_BUFFER_DURATION(sinkbuf) + element->block_duration - 1) / element->block_duration;	/* ceil */
	g_assert_cmpuint(blocks, >, 0);	/* guaranteed by check for short-buffers above */
	length = GST_BUFFER_OFFSET_END(sinkbuf) - GST_BUFFER_OFFSET(sinkbuf);
	block_length = (length + blocks - 1) / blocks;	/* ceil */
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

		srcbuf = gst_buffer_copy_region(sinkbuf, GST_BUFFER_COPY_META | GST_BUFFER_COPY_TIMESTAMPS | GST_BUFFER_COPY_FLAGS | GST_BUFFER_COPY_MEMORY, offset * element->unit_size, block_length * element->unit_size);
		if(G_UNLIKELY(!srcbuf)) {
			GST_ERROR_OBJECT(element, "failure creating sub-buffer");
			result = GST_FLOW_ERROR;
			break;
		}

		/*
		 * set flags, caps, offset, and timestamps.
		 */

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

		GST_DEBUG_OBJECT(element, "pushing sub-buffer %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(srcbuf));
		result = gst_pad_push(element->srcpad, srcbuf);
		if(G_UNLIKELY(result != GST_FLOW_OK)) {
			GST_WARNING_OBJECT(element, "push failed: %s", gst_flow_get_name(result));
			break;
		}
	}
	gst_buffer_unref(sinkbuf);

	/*
	 * done
	 */

done:
	return result;
}


/*
 * ============================================================================
 *
 *                              GObject Methods
 *
 * ============================================================================
 */


/*
 * properties
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
 * finalize()
 */


static void finalize(GObject *object)
{
	GSTLALReblock *element = GSTLAL_REBLOCK(object);

	gst_object_unref(element->sinkpad);
	element->sinkpad = NULL;
	gst_object_unref(element->srcpad);
	element->srcpad = NULL;

	G_OBJECT_CLASS(gstlal_reblock_parent_class)->finalize(object);
}


/*
 * class_init()
 */


#define CAPS \
	"audio/x-raw, " \
	"rate = " GST_AUDIO_RATE_RANGE ", " \
	"channels = " GST_AUDIO_CHANNELS_RANGE ", " \
	"format = (string) " GSTLAL_AUDIO_FORMATS_ALL ", " \
	"layout = (string) interleaved, " \
	"channel-mask = (bitmask) 0"


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
 * instance init
 */


static void gstlal_reblock_init(GSTLALReblock *element)
{
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
}

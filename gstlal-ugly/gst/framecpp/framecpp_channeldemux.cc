/*
 * framecpp channel demultiplexor
 *
 * Copyright (C) 2011--2014  Kipp Cannon, Ed Maros
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
 *				  Preamble
 *
 * ========================================================================
 */


/*
 * stuff from standard library
 */


#include <iostream>
#include <stdint.h>
#include <stdexcept>
#include <string.h>


/*
 * stuff from glib/gobject/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/audio/audio.h>


/*
 * from framecpp
 */


#include <framecpp/Common/MemoryBuffer.hh>
#include <framecpp/Common/Verify.hh>
#include <framecpp/FrameH.hh>
#include <framecpp/FrAdcData.hh>
#include <framecpp/FrProcData.hh>
#include <framecpp/FrRawData.hh>
#include <framecpp/FrSimData.hh>
#include <framecpp/FrVect.hh>
#include <framecpp/IFrameStream.hh>


/*
 * our own stuff
 */


#include <gstlal/gstlal.h>
#include <gstlal/gstlal_debug.h>
#include <gstlal/gstlal_tags.h>
#include <framecpp_channeldemux.h>
#include <gstfrpad.h>


/*
 * ============================================================================
 *
 *                                Boilerplate
 *
 * ============================================================================
 */


#define GST_CAT_DEFAULT framecpp_channeldemux_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


static void additional_initializations(GType type)
{
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "framecpp_channeldemux", 0, "framecpp_channeldemux element");
}


GST_BOILERPLATE_FULL(GstFrameCPPChannelDemux, framecpp_channeldemux, GstElement, GST_TYPE_ELEMENT, additional_initializations);


/*
 * ============================================================================
 *
 *                                 Parameters
 *
 * ============================================================================
 */


#define DEFAULT_DO_FILE_CHECKSUM FALSE
#define DEFAULT_SKIP_BAD_FILES FALSE
#define DEFAULT_FRAME_FORMAT_VERSION 0
#define DEFAULT_FRAME_LIBRARY_VERSION 255
#define DEFAULT_FRAME_LIBRARY_NAME ""
#define DEFAULT_FRAME_NAME ""
#define DEFAULT_FRAME_RUN -1
#define DEFAULT_FRAME_NUMBER 0


#define MAX_TIMESTAMP_JITTER 1	/* allow this many ns before calling it a discont */


/*
 * ============================================================================
 *
 *                                 Utilities
 *
 * ============================================================================
 */


/*
 * split a string of the form "INSTRUMENT:CHANNEL" into two strings
 * containing the instrument and channel parts separately.  the pointers
 * placed in the instrument and channel pointers should be free()ed when no
 * longer needed.  returns 0 on success, < 0 on failure.
 */


static int split_name(const char *name, gchar **instrument, gchar **channel)
{
	const char *colon = strchr(name, ':');

	if(!colon) {
		*instrument = *channel = NULL;
		return -1;
	}

	*instrument = g_strndup(name, colon - name);
	*channel = g_strdup(colon + 1);

	return 0;
}


/*
 * check if a channel name is in the requested-channel list.  returns TRUE
 * if the channel list is empty (demultiplex all channels)
 */


static gboolean is_requested_channel(GstFrameCPPChannelDemux *element, const char *name)
{
	return !g_hash_table_size(element->channel_list) || g_hash_table_lookup(element->channel_list, name);
}


/*
 * build the GstCaps describing the format of the contents of an FrVect
 */


static GstCaps *FrVect_get_caps(LDASTools::AL::SharedPtr<FrameCPP::FrVect> vect, gint *rate, guint *unit_size)
{
	GstCaps *caps;
	gint width = vect->GetTypeSize() * 8;
	*rate = round(1.0 / vect->GetDim(0).GetDx());

	g_assert(1.0 / *rate == vect->GetDim(0).GetDx());

	switch(vect->GetType()) {
	case FrameCPP::FrVect::FR_VECT_4R:
	case FrameCPP::FrVect::FR_VECT_8R:
		caps = gst_caps_new_simple("audio/x-raw-float",
			"rate", G_TYPE_INT, *rate,
			"channels", G_TYPE_INT, 1,
			"endianness", G_TYPE_INT, G_BYTE_ORDER,
			"width", G_TYPE_INT, width,
			NULL);
		break;

	case FrameCPP::FrVect::FR_VECT_8C:
	case FrameCPP::FrVect::FR_VECT_16C:
		caps = gst_caps_new_simple("audio/x-raw-complex",
			"rate", G_TYPE_INT, *rate,
			"channels", G_TYPE_INT, 1,
			"endianness", G_TYPE_INT, G_BYTE_ORDER,
			"width", G_TYPE_INT, width,
			NULL);
		break;

	case FrameCPP::FrVect::FR_VECT_C:
	case FrameCPP::FrVect::FR_VECT_2S:
	case FrameCPP::FrVect::FR_VECT_4S:
	case FrameCPP::FrVect::FR_VECT_8S:
		caps = gst_caps_new_simple("audio/x-raw-int",
			"rate", G_TYPE_INT, *rate,
			"channels", G_TYPE_INT, 1,
			"endianness", G_TYPE_INT, G_BYTE_ORDER,
			"width", G_TYPE_INT, width,
			"depth", G_TYPE_INT, width,	/* FIXME:  for Adc use nBits */
			"signed", G_TYPE_BOOLEAN, TRUE,
			NULL);
		break;

	case FrameCPP::FrVect::FR_VECT_1U:
	case FrameCPP::FrVect::FR_VECT_2U:
	case FrameCPP::FrVect::FR_VECT_4U:
	case FrameCPP::FrVect::FR_VECT_8U:
		caps = gst_caps_new_simple("audio/x-raw-int",
			"rate", G_TYPE_INT, *rate,
			"channels", G_TYPE_INT, 1,
			"endianness", G_TYPE_INT, G_BYTE_ORDER,
			"width", G_TYPE_INT, width,
			"depth", G_TYPE_INT, width,	/* FIXME;  for Adc use nBits */
			"signed", G_TYPE_BOOLEAN, FALSE,
			NULL);
		break;

	default:
		g_assert_not_reached();
	}

	*unit_size = 1 * width / 8;	/* 1 channel */

	return caps;
}


/*
 * transfer the contents of an FrVect into a newly-created GstBuffer.
 * caller must unref buffer when no longer needed.
 */


static GstBuffer *FrVect_to_GstBuffer(LDASTools::AL::SharedPtr<FrameCPP::FrVect> vect, GstClockTime timestamp, guint64 offset, gint *rate, guint *unit_size)
{
	GstBuffer *buffer;

	/*
	 * trigger data decompression before calling GetNBytes()
	 */

	vect->GetDataUncompressed();

	/*
	 * allocate buffer
	 */

	buffer = gst_buffer_new_and_alloc(vect->GetNBytes());
	if(!buffer) {
		/* silence possibly-uninitialized warnings */
		*rate = *unit_size = 0;
		return NULL;
	}

	/*
	 * copy data into buffer
	 * FIXME:  it would be nice to remove the memcpy() by hooking the
	 * GstBuffer's clean-up into framecpp's reference counting
	 * machinery
	 */

	g_assert_cmpuint(vect->GetNDim(), ==, 1);
	memcpy(GST_BUFFER_DATA(buffer), vect->GetData().get(), GST_BUFFER_SIZE(buffer));

	/*
	 * set buffer format
	 */

	GST_BUFFER_CAPS(buffer) = FrVect_get_caps(vect, rate, unit_size);
	g_assert(GST_BUFFER_CAPS(buffer) != NULL);

	/*
	 * set timestamp and duration
	 */

	GST_BUFFER_TIMESTAMP(buffer) = timestamp + (GstClockTime) round(vect->GetDim(0).GetStartX() * GST_SECOND);
	GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale_int(vect->GetNData(), GST_SECOND, *rate);
	GST_BUFFER_OFFSET(buffer) = offset;
	GST_BUFFER_OFFSET_END(buffer) = offset + vect->GetNData();

	/*
	 * done
	 */

	return buffer;
}


/*
 * local version of gst_audio_buffer_clip()
 * FIXME:  remove this when the stock version doesn't segfault
 */


static GstBuffer *my_gst_audio_buffer_clip(GstBuffer *buffer, GstSegment *segment, gint rate, gint unit_size)
{
	guint offset, offset_end;

	g_assert_cmpuint((GST_BUFFER_OFFSET_END(buffer) - GST_BUFFER_OFFSET(buffer)) * unit_size, ==, GST_BUFFER_SIZE(buffer));

	if(GST_CLOCK_TIME_IS_VALID(segment->start)) {
		if(GST_BUFFER_TIMESTAMP(buffer) + GST_BUFFER_DURATION(buffer) <= (guint64) segment->start) {
			gst_buffer_unref(buffer);
			return NULL;
		}
		offset = gst_util_uint64_scale_int_round(MAX(GST_BUFFER_TIMESTAMP(buffer), (guint64) segment->start) - GST_BUFFER_TIMESTAMP(buffer), rate, GST_SECOND);
		g_assert_cmpuint(offset, <=, GST_BUFFER_OFFSET_END(buffer) - GST_BUFFER_OFFSET(buffer));
	} else
		offset = 0;

	if(GST_CLOCK_TIME_IS_VALID(segment->stop)) {
		if(GST_BUFFER_TIMESTAMP(buffer) >= (guint64) segment->stop) {
			gst_buffer_unref(buffer);
			return NULL;
		}
		offset_end = gst_util_uint64_scale_int_round(MIN(GST_BUFFER_TIMESTAMP(buffer) + GST_BUFFER_DURATION(buffer), (guint64) segment->stop) - GST_BUFFER_TIMESTAMP(buffer), rate, GST_SECOND);
		g_assert_cmpuint(offset_end, <=, GST_BUFFER_OFFSET_END(buffer) - GST_BUFFER_OFFSET(buffer));
	} else
		offset_end = GST_BUFFER_OFFSET_END(buffer) - GST_BUFFER_OFFSET(buffer);

	g_assert_cmpuint(offset, <=, offset_end);

	if(offset_end - offset != GST_BUFFER_OFFSET_END(buffer) - GST_BUFFER_OFFSET(buffer)) {
		/* buffer lies partially outside requested segment */
		GstBuffer *newbuf = gst_buffer_create_sub(buffer, offset * unit_size, (offset_end - offset) * unit_size);
		gst_buffer_copy_metadata(newbuf, buffer, (GstBufferCopyFlags) (GST_BUFFER_COPY_FLAGS | GST_BUFFER_COPY_CAPS));
		GST_BUFFER_TIMESTAMP(newbuf) = GST_BUFFER_TIMESTAMP(buffer) + gst_util_uint64_scale_int_round(offset, GST_SECOND, rate);
		GST_BUFFER_DURATION(newbuf) = GST_BUFFER_TIMESTAMP(buffer) + gst_util_uint64_scale_int_round(offset_end, GST_SECOND, rate) - GST_BUFFER_TIMESTAMP(newbuf);
		GST_BUFFER_OFFSET(newbuf) = GST_BUFFER_OFFSET(buffer) + offset;
		GST_BUFFER_OFFSET_END(newbuf) = GST_BUFFER_OFFSET(buffer) + offset_end;
		GST_DEBUG("clipped buffer spanning %" GST_BUFFER_BOUNDARIES_FORMAT " to %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(buffer), GST_BUFFER_BOUNDARIES_ARGS(newbuf));
		gst_buffer_unref(buffer);
		buffer = newbuf;
	}

	return buffer;
}


/*
 * ============================================================================
 *
 *                                Source Pads
 *
 * ============================================================================
 */


/*
 * pad state
 */


struct pad_state {
	gboolean need_discont;
	gboolean need_new_segment;
	gboolean need_tags;
	GstClockTime next_timestamp;
	guint64 next_out_offset;
};


/*
 * linked event handler.  create & initialize pad's state
 */


static void src_pad_linked_handler(GstPad *pad, GstPad *peer, gpointer data)
{
	struct pad_state *pad_state = g_new0(struct pad_state, 1);

	pad_state->need_discont = TRUE;
	pad_state->need_new_segment = TRUE;
	pad_state->need_tags = TRUE;
	pad_state->next_timestamp = GST_CLOCK_TIME_NONE;
	pad_state->next_out_offset = 0;

	g_assert(gst_pad_get_element_private(pad) == NULL);
	gst_pad_set_element_private(pad, pad_state);
}


/*
 * unlinked event handler.  free pad's state
 */


static void src_pad_unlinked_handler(GstPad *pad, GstPad *peer, gpointer data)
{
	struct pad_state *pad_state = (struct pad_state *) gst_pad_get_element_private(pad);

	gst_pad_set_element_private(pad, NULL);
	g_free(pad_state);
}


/*
 * pad tags notify handler
 */


static void src_pad_new_tags_handler(GObject *object, GParamSpec *pspec, gpointer user_data)
{
	struct pad_state *pad_state = (struct pad_state *) gst_pad_get_element_private(GST_PAD(object));

	if(pad_state)
		pad_state->need_tags = TRUE;
}


/*
 * create a new source pad and add to element.  does not check if name is
 * already in use
 */


static GstPad *add_src_pad(GstFrameCPPChannelDemux *element, const char *name)
{
	GstFrPad *srcpad = NULL;
	gchar *instrument, *channel;

	/*
	 * construct the pad
	 */

	srcpad = gst_frpad_new_from_template(gst_element_class_get_pad_template(GST_ELEMENT_CLASS(G_OBJECT_GET_CLASS(element)), "%s"), name);
	g_assert(srcpad != NULL);

	/*
	 * connect signal handlers
	 */

	g_signal_connect(srcpad, "linked", (GCallback) src_pad_linked_handler, NULL);
	g_signal_connect(srcpad, "unlinked", (GCallback) src_pad_unlinked_handler, NULL);
	g_signal_connect(srcpad, "notify::tags", (GCallback) src_pad_new_tags_handler, NULL);

	/*
	 * set instrument and channel-name
	 */

	split_name(name, &instrument, &channel);
	g_object_set(srcpad, "instrument", instrument, "channel-name", channel, NULL);
	g_free(instrument);
	g_free(channel);

	/*
	 * add pad to element.  must ref it because _add_pad()
	 * consumes a reference
	 */

	gst_pad_set_active(GST_PAD(srcpad), TRUE);
	gst_object_ref(srcpad);
	gst_element_add_pad(GST_ELEMENT(element), GST_PAD(srcpad));

	/*
	 * done
	 */

	return GST_PAD(srcpad);
}


/*
 * remove a source pad from element
 *
 * FIXME:  what about EOS, state changes, etc.?  do we have to do that?
 */

/* NOT USED
static gboolean remove_pad(GstFrameCPPChannelDemux *element, const char *name)
{
	GstPad *srcpad = gst_element_get_static_pad(GST_ELEMENT(element), name);
	g_assert(srcpad != NULL);
	gst_pad_set_active(srcpad, FALSE);
	return gst_element_remove_pad(GST_ELEMENT(element), srcpad);
}
*/


/*
 * get pad by name, creating it if it doesn't exist
 */


static GstPad *get_src_pad(GstFrameCPPChannelDemux *element, const char *name, enum gst_frpad_type_t pad_type, gboolean *pad_added)
{
	GstPad *srcpad;

	/*
	 * retrieve the pad.  if element does not already have a pad by
	 * this name, create it
	 */

	srcpad = gst_element_get_static_pad(GST_ELEMENT(element), name);
	if(!srcpad) {
		srcpad = add_src_pad(element, name);
		if(srcpad) {
			g_object_set(srcpad, "pad-type", pad_type, NULL);
			*pad_added = TRUE;
		}
	} else {
		/* FIXME:  bother checking if pad_type is correct? */
	}

	return srcpad;
}


/*
 * push pending events
 */


static gboolean src_pad_do_pending_events(GstFrameCPPChannelDemux *element, GstPad *pad)
{
	struct pad_state *pad_state = (struct pad_state *) gst_pad_get_element_private(pad);
	gboolean success = TRUE;

	g_assert(pad_state != NULL);

	/*
	 * forward most recent new segment event
	 */

	if(pad_state->need_new_segment && element->last_new_segment_event) {
		GST_LOG_OBJECT(pad, "push %" GST_PTR_FORMAT, element->last_new_segment_event);
		gst_event_ref(element->last_new_segment_event);
		success = gst_pad_push_event(pad, element->last_new_segment_event);
		if(!success)
			GST_ERROR_OBJECT(pad, "failed to push new segment");
		else
			pad_state->need_new_segment = FALSE;
	}

	/*
	 * send tags
	 */

	if(success && pad_state->need_tags) {
		GstTagList *tag_list;
		g_object_get(pad, "tags", &tag_list, NULL);
		gst_tag_list_insert(tag_list, element->tag_list, GST_TAG_MERGE_KEEP);
		GST_LOG_OBJECT(pad, "push new %" GST_PTR_FORMAT, tag_list);
		gst_element_found_tags_for_pad(GST_ELEMENT(element), pad, tag_list);
		pad_state->need_tags = FALSE;
	}

	return success;
}


/*
 * convert an FrVect to a GstBuffer, and push out a source pad.
 */


static GstFlowReturn frvect_to_buffer_and_push(GstFrameCPPChannelDemux *element, GstPad *pad, LDASTools::AL::SharedPtr<FrameCPP::FrVect> vect, GstClockTime timestamp)
{
	struct pad_state *pad_state = (struct pad_state *) gst_pad_get_element_private(pad);
	GstBuffer *buffer;
	gint rate;
	guint unit_size;
	GstFlowReturn result = GST_FLOW_OK;

	g_assert(pad_state != NULL);

	/*
	 * convert FrVect to GstBuffer
	 */

	buffer = FrVect_to_GstBuffer(vect, timestamp, pad_state->next_out_offset, &rate, &unit_size);
	g_assert(buffer != NULL);

	/*
	 * if the format matches the pad's replace the buffer's caps with
	 * the pad's to reduce the number of objects in memory and simplify
	 * subsequent comparisons.  NOTE:  the caps on the source pad get
	 * set explicitly, here, to trigger any pipeline graph adjustments
	 * that might happen as a result of the format discovery and to
	 * trigger an update of the tags before pending segments and tags
	 * are pushed downstream.
	 */

	if(gst_caps_is_equal(GST_BUFFER_CAPS(buffer), GST_PAD_CAPS(pad)))
		gst_buffer_set_caps(buffer, GST_PAD_CAPS(pad));
	else {
		GST_LOG_OBJECT(pad, "new caps: %P", GST_BUFFER_CAPS(buffer));
		gst_pad_set_caps(pad, GST_BUFFER_CAPS(buffer));
	}

	/*
	 * do pending events.  FIXME:  check for errors?
	 */

	src_pad_do_pending_events(element, pad);

	/*
	 * clip buffer to configured segment
	 */

	if(element->last_new_segment_event && element->segment.format == GST_FORMAT_TIME) {
#if 0
		/* FIXME:  this function segfaults sometimes.  my guess is bad rounding because it happens more often for very low sample rates */
		buffer = gst_audio_buffer_clip(buffer, &element->segment, rate, unit_size);
#else
		buffer = my_gst_audio_buffer_clip(buffer, &element->segment, rate, unit_size);
#endif
		if(!buffer) {
			GST_WARNING_OBJECT(pad, "buffer outside of configured segment: dropped");
			goto done;
		}
	}

	/*
	 * check for disconts
	 */

	if(GST_CLOCK_TIME_IS_VALID(pad_state->next_timestamp) && GST_BUFFER_TIMESTAMP(buffer) < pad_state->next_timestamp)
		GST_WARNING_OBJECT(pad, "time reversal detected:  expected %" GST_TIME_SECONDS_FORMAT ", got %" GST_TIME_SECONDS_FORMAT, GST_TIME_SECONDS_ARGS(pad_state->next_timestamp), GST_TIME_SECONDS_ARGS(GST_BUFFER_TIMESTAMP(buffer)));
	if(pad_state->need_discont || (GST_CLOCK_TIME_IS_VALID(pad_state->next_timestamp) && llabs(GST_BUFFER_TIMESTAMP(buffer) - pad_state->next_timestamp) > MAX_TIMESTAMP_JITTER)) {
		GST_BUFFER_FLAG_SET(buffer, GST_BUFFER_FLAG_DISCONT);
		pad_state->need_discont = FALSE;
	}

	/*
	 * record state for next time
	 */

	pad_state->next_timestamp = GST_BUFFER_TIMESTAMP(buffer) + GST_BUFFER_DURATION(buffer);
	pad_state->next_out_offset = GST_BUFFER_OFFSET_END(buffer);

	/*
	 * push buffer downstream
	 */

	GST_LOG_OBJECT(pad, "pushing buffer spanning %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(buffer));
	result = gst_pad_push(pad, buffer);

	/*
	 * done
	 */

done:
	return result;
}


/*
 * forward_heart_beat()
 */


static GstFlowReturn push_heart_beat(GstFrameCPPChannelDemux *element, GstPad *pad, GstClockTime timestamp)
{
	struct pad_state *pad_state = (struct pad_state *) gst_pad_get_element_private(pad);
	GstBuffer *buffer;

	g_assert(pad_state != NULL);

	/*
	 * do pending events.  FIXME:  check for errors?
	 */

	src_pad_do_pending_events(element, pad);

	/*
	 * don't push a heart beat if the caps aren't set yet
	 */

	if(!GST_PAD_CAPS(pad)) {
		GST_WARNING_OBJECT(pad, "caps not set, not pushing heart beat;  if this is a problem, consider setting caps on this pad manually");
		return GST_FLOW_OK;
	}

	/*
	 * create heartbeat buffer for this pad
	 */

	buffer = gst_buffer_new();
	GST_BUFFER_TIMESTAMP(buffer) = timestamp;
	GST_BUFFER_DURATION(buffer) = 0;
	GST_BUFFER_OFFSET(buffer) = GST_BUFFER_OFFSET_END(buffer) = pad_state->next_out_offset;
	gst_buffer_set_caps(buffer, GST_PAD_CAPS(pad));

	/*
	 * check for disconts
	 */

	if(pad_state->need_discont || (GST_CLOCK_TIME_IS_VALID(pad_state->next_timestamp) && llabs(GST_BUFFER_TIMESTAMP(buffer) - pad_state->next_timestamp) > MAX_TIMESTAMP_JITTER)) {
		GST_BUFFER_FLAG_SET(buffer, GST_BUFFER_FLAG_DISCONT);
		pad_state->need_discont = FALSE;
	}

	/*
	 * record state for next time
	 */

	pad_state->next_timestamp = GST_BUFFER_TIMESTAMP(buffer) + GST_BUFFER_DURATION(buffer);
	pad_state->next_out_offset = GST_BUFFER_OFFSET_END(buffer);

	/*
	 * push buffer
	 */

	GST_LOG_OBJECT(pad, "pushing 0-length heart-beat buffer");
	return gst_pad_push(pad, buffer);
}


struct push_heart_beat_data {
	GstFrameCPPChannelDemux *element;
	GstClockTime timestamp;
};


static void push_heart_beat_iter_wrapper(gpointer object, gpointer anon_data)
{
	GstPad *pad = GST_PAD(object);
	struct push_heart_beat_data *data = (struct push_heart_beat_data *) anon_data;

	if(gst_pad_is_linked(pad))
		/*
		 * push buffer.  ignore failures
		 * FIXME:  should failures be ignored?
		 */

		push_heart_beat(data->element, pad, data->timestamp);

	gst_object_unref(pad);
}


static GstFlowReturn forward_heart_beat(GstFrameCPPChannelDemux *element, GstClockTime t)
{
	GstIterator *iter = gst_element_iterate_src_pads(GST_ELEMENT(element));
	struct push_heart_beat_data data = {element, t};
	GstFlowReturn result = GST_FLOW_OK;

	gst_iterator_foreach(iter, push_heart_beat_iter_wrapper, &data);
	gst_iterator_free(iter);

	return result;
}


/*
 * ============================================================================
 *
 *                                  Sink Pad
 *
 * ============================================================================
 */


/*
 * event()
 */


static void gst_event_parse_new_segment_segment(GstEvent *event, GstSegment *segment)
{
	gboolean update;
	gdouble rate, applied_rate;
	GstFormat format;
	gint64 start, stop, position;

	gst_event_parse_new_segment_full(event, &update, &rate, &applied_rate, &format, &start, &stop, &position);
	gst_segment_set_newsegment_full(segment, update, rate, applied_rate, format, start, stop, position);
}


static void forward_sink_event(gpointer object, gpointer data)
{
	GstPad *pad = GST_PAD(object);
	GstEvent *event = GST_EVENT(data);
	if(gst_pad_is_linked(pad)) {
		/*
		 * forward event out this pad.  ignore failures
		 * FIXME:  should failures be ignored?
		 */
		gst_event_ref(event);
		gst_pad_push_event(pad, event);
	}
	gst_object_unref(pad);
}


static gboolean sink_event(GstPad *pad, GstEvent *event)
{
	GstFrameCPPChannelDemux *element = FRAMECPP_CHANNELDEMUX(gst_pad_get_parent(pad));
	GstIterator *iter;
	gboolean success = TRUE;

	switch(GST_EVENT_TYPE(event)) {
	case GST_EVENT_NEWSEGMENT:
		if(element->last_new_segment_event)
			gst_event_unref(element->last_new_segment_event);
		gst_event_ref(event);
		element->last_new_segment_event = event;
		gst_event_parse_new_segment_segment(event, &element->segment);
		break;

	case GST_EVENT_EOS:
		if(element->last_new_segment_event)
			gst_event_unref(element->last_new_segment_event);
		element->last_new_segment_event = NULL;
		/*
		 * if there are no source pads, the EOS event will not be
		 * recieved by any sink elements, it will not get posted to
		 * the pipeline message bus, and the application's pipeline
		 * handler will never find out.  this will lead to a hung
		 * application.  FIXME:  asserts can be compiled out:  is
		 * this a genuine error condition that should be detected
		 * and reported, or is checking for it debugging-like?
		 */
		g_assert(GST_ELEMENT(element)->numsrcpads > 0);
		break;

	default:
		break;
	}

	/* FIXME:  what does gst_pad_event_default(pad, event) do?  can I just use that? */
	iter = gst_element_iterate_src_pads(GST_ELEMENT(element));
	gst_iterator_foreach(iter, forward_sink_event, event);
	gst_iterator_free(iter);

	gst_event_unref(event);
	gst_object_unref(element);
	return success;
}


/*
 * chain()
 */


static GstFlowReturn chain(GstPad *pad, GstBuffer *inbuf)
{
	GstFrameCPPChannelDemux *element = FRAMECPP_CHANNELDEMUX(gst_pad_get_parent(pad));
	gboolean pads_added = FALSE;
	GstPad *srcpad = NULL;
	GstFlowReturn result = GST_FLOW_OK;

	/*
	 * special case:  0-length input buffers are treated as heart
	 * beats, we forward a heart beat out each source pad
	 */

	if(!GST_BUFFER_SIZE(inbuf)) {
		result = forward_heart_beat(element, GST_BUFFER_TIMESTAMP(inbuf));
		goto done;
	}

	/*
	 * decode frame file
	 */

	GST_LOG_OBJECT(element, "begin IGWD file decode");

	try {
		if(element->do_file_checksum) {
			/*
			 * File Checksum verification
			 *
			 * This additional scope allows for cleanup of variables used
			 * only for the file checksum validation.
			 * Resources are returned to the system as the variables go
			 * out of scope.
			 */

			FrameCPP::Common::MemoryBuffer *ibuf(new FrameCPP::Common::MemoryBuffer(std::ios::in));
			ibuf->pubsetbuf((char *) GST_BUFFER_DATA(inbuf), GST_BUFFER_SIZE(inbuf));
			FrameCPP::IFrameStream ifs(ibuf);

			FrameCPP::Common::Verify verifier;

			verifier.BufferSize(GST_BUFFER_SIZE(inbuf));
			verifier.UseMemoryMappedIO(false);
			verifier.CheckDataValid(false);	/* FIXME:  what's this? */
			verifier.Expandability(false);
			verifier.MustHaveEOFChecksum(true);
			verifier.Strict(false);
			verifier.ValidateMetadata(false);
			verifier.CheckFileChecksumOnly(true);

			if(verifier(ifs) != 0)
				throw std::runtime_error(verifier.ErrorInfo());
		}

		FrameCPP::Common::MemoryBuffer *ibuf(new FrameCPP::Common::MemoryBuffer(std::ios::in));
		ibuf->pubsetbuf((char *) GST_BUFFER_DATA(inbuf), GST_BUFFER_SIZE(inbuf));
		FrameCPP::IFrameStream ifs(ibuf);

		/*
		 * update element properties
		 */

		if(ifs.Version() != element->frame_format_version) {
			element->frame_format_version = ifs.Version();
			g_object_notify(G_OBJECT(element), "frame-format-version");
		}
		if(ifs.LibraryRevision() != element->frame_library_version) {
			element->frame_library_version = ifs.LibraryRevision();
			g_object_notify(G_OBJECT(element), "frame-library-version");
		}
		if(g_strcmp0(ifs.FrameLibraryName().c_str(), element->frame_library_name)) {
			g_free(element->frame_library_name);
			element->frame_library_name = g_strdup(ifs.FrameLibraryName().c_str());
			g_object_notify(G_OBJECT(element), "frame-library-name");
		}

		GST_LOG_OBJECT(element, "found version %u frame file generated by %s library revision %d", element->frame_format_version, element->frame_library_name, element->frame_library_version);

		while(1) {
			FrameCPP::IFrameStream::frame_h_type frame;
			try {
				frame = ifs.ReadNextFrame();
			} catch(const std::out_of_range& Error) {
				/* no more frames */
				break;
			}

			GstClockTime frame_timestamp = 1000000000L * frame->GetGTime().GetSeconds() + frame->GetGTime().GetNanoseconds();

			GST_LOG_OBJECT(element, "found frame %d at %" GST_TIME_SECONDS_FORMAT, ifs.GetCurrentFrameOffset(), GST_TIME_SECONDS_ARGS(frame_timestamp));

			/*
			 * update element properties
			 */

			if(g_strcmp0(frame->GetName().c_str(), element->frame_name)) {
				g_free(element->frame_name);
				element->frame_name = g_strdup(frame->GetName().c_str());
				g_object_notify(G_OBJECT(element), "frame-name");
			}
			if(frame->GetRun() != element->frame_run) {
				element->frame_run = frame->GetRun();
				g_object_notify(G_OBJECT(element), "frame-run");
			}
			/* assume this changes */
			element->frame_number = frame->GetFrame();
			g_object_notify(G_OBJECT(element), "frame-number");

			/*
			 * populate tags from frame metadata.  the tags
			 * pushed out a source pad are taken from the pad's
			 * own metadata and this list populated from the
			 * frame metadata.  this list is updated for each
			 * new frame, but a new tag list will only be
			 * pushed out a source pad if that pad's own tags
			 * are changed.
			 */

			{
			GstDateTime *date_time = gstlal_datetime_new_from_gps(frame_timestamp);
			gchar *container_format = g_strdup_printf("IGWD frame file v%d", element->frame_format_version);
			gst_tag_list_add(element->tag_list, GST_TAG_MERGE_KEEP, GST_TAG_DATE_TIME, date_time, GST_TAG_CONTAINER_FORMAT, container_format, GST_TAG_ENCODER, element->frame_library_name, GST_TAG_ENCODER_VERSION, element->frame_library_version, GST_TAG_ORGANIZATION, element->frame_name, NULL);
			gst_date_time_unref(date_time);
			g_free(container_format);
			}

			/*
			 * process ADC data
			 */

			FrameCPP::FrameH::rawData_type rd = frame->GetRawData();
			if(rd) {
				for(FrameCPP::FrRawData::firstAdc_iterator current = rd->RefFirstAdc().begin(), last = rd->RefFirstAdc().end(); current != last; current++) {
					FrameCPP::FrAdcData::data_type vects = (*current)->RefData();
					GstClockTime timestamp = frame_timestamp + (GstClockTimeDiff) round((*current)->GetTimeOffset() * 1e9);
					const char *name = (*current)->GetName().c_str();

					GST_LOG_OBJECT(element, "found FrAdcData %s at %" GST_TIME_SECONDS_FORMAT, name, GST_TIME_SECONDS_ARGS(timestamp));

					/*
					 * retrieve the source pad.  create
					 * it if it doesn't exist.  update
					 * properties to reflect stream
					 * metadata.  if the pad has no
					 * peer or it not in the requested
					 * channel list, skip this channel.
					 */

					if(!is_requested_channel(element, name)) {
						GST_LOG_OBJECT(element, "skipping: channel not requested");
						continue;
					}
					srcpad = get_src_pad(element, name, GST_FRPAD_TYPE_FRADCDATA, &pads_added);
					/* FIXME:  units */
					g_object_set(srcpad,
						"comment", (*current)->GetComment().c_str(),
						"channel-group", (*current)->GetChannelGroup(),
						"channel-number", (*current)->GetChannelNumber(),
						"nbits", (*current)->GetNBits(),	/* FIXME:  set depth in caps */
						"bias", (*current)->GetBias(),
						"slope", (*current)->GetSlope(),
						NULL
					);
					if(!gst_pad_is_linked(srcpad)) {
						GST_LOG_OBJECT(srcpad, "skipping: not linked");
						gst_object_unref(srcpad);
						srcpad = NULL;
						continue;
					}

					/*
					 * convert FrVects to GstBuffers
					 * and push out source pad,
					 * checking for disconts and
					 * recording state for next time.
					 */

					/* FIXME:  what about checking "dataValid" vect in the aux list? */
					if((*current)->GetDataValid() == 0 && vects.size()) {
						for(FrameCPP::FrAdcData::data_type::iterator vect = vects.begin(), last_vect = vects.end(); vect != last_vect; vect++) {
							/* FIXME:  do something like this? */
							/*g_object_set(srcpad, "compression-scheme", vect->GetCompress(), NULL);*/
							result = frvect_to_buffer_and_push(element, srcpad, *vect, timestamp);
							if(result != GST_FLOW_OK) {
								GST_ERROR_OBJECT(srcpad, "failure: %s", gst_flow_get_name(result));
								gst_object_unref(srcpad);
								srcpad = NULL;
								goto done;
							}
						}
					} else {
						if((*current)->GetDataValid() != 0)
							GST_DEBUG_OBJECT(srcpad, "FrAdcData invalid (dataValid=0x%04x)", (*current)->GetDataValid());
						if(!vects.size())
							GST_DEBUG_OBJECT(srcpad, "no FrVects");
						result = push_heart_beat(element, srcpad, timestamp);
						if(result != GST_FLOW_OK) {
							GST_ERROR_OBJECT(srcpad, "failure: %s", gst_flow_get_name(result));
							gst_object_unref(srcpad);
							srcpad = NULL;
							goto done;
						}
					}
					gst_object_unref(srcpad);
					srcpad = NULL;
				}
			}

			/*
			 * process proc data
			 */

			for(FrameCPP::FrameH::procData_iterator current = frame->RefProcData().begin(), last = frame->RefProcData().end(); current != last; current++) {
				FrameCPP::FrProcData::data_type vects = (*current)->RefData();
				GstClockTime timestamp = frame_timestamp + (GstClockTimeDiff) round((*current)->GetTimeOffset() * 1e9);
				const char *name = (*current)->GetName().c_str();

				/*
				 * FIXME:  check the FrProcData "type"
				 * field, must be time series.  might also
				 * be able to support frequency series and
				 * time-frequency types in the future
				 */

				GST_LOG_OBJECT(element, "found FrProcData %s at %" GST_TIME_SECONDS_FORMAT, name, GST_TIME_SECONDS_ARGS(timestamp));

				/*
				 * retrieve the source pad.  create it if
				 * it doesn't exist.  update properties to
				 * reflect stream metadata.  if the pad has
				 * no peer or it not in the requested
				 * channel list, skip this channel.
				 */

				if(!is_requested_channel(element, name)) {
					GST_LOG_OBJECT(element, "skipping: channel not requested");
					continue;
				}
				srcpad = get_src_pad(element, name, GST_FRPAD_TYPE_FRPROCDATA, &pads_added);
				/* FIXME: units, history */
				g_object_set(srcpad,
					"comment", (*current)->GetComment().c_str(),
					NULL
				);
				if(!gst_pad_is_linked(srcpad)) {
					GST_LOG_OBJECT(srcpad, "skipping: not linked");
					gst_object_unref(srcpad);
					srcpad = NULL;
					continue;
				}

				/*
				 * convert FrVects to GstBuffers and push
				 * out source pad, checking for disconts
				 * and recording state for next time.
				 */

				/* FIXME:  what about checking "dataValid" vect in the aux list? */
				if(vects.size()) {
					for(FrameCPP::FrProcData::data_type::iterator vect = vects.begin(), last_vect = vects.end(); vect != last_vect; vect++) {
						/* FIXME:  do something like this? */
						/*g_object_set(srcpad, "compression-scheme", vect->GetCompress(), NULL);*/
						result = frvect_to_buffer_and_push(element, srcpad, *vect, timestamp);
						if(result != GST_FLOW_OK) {
							GST_ERROR_OBJECT(srcpad, "failure: %s", gst_flow_get_name(result));
							gst_object_unref(srcpad);
							srcpad = NULL;
							goto done;
						}
					}
				} else {
					GST_LOG_OBJECT(srcpad, "no FrVects");
					result = push_heart_beat(element, srcpad, timestamp);
					if(result != GST_FLOW_OK) {
						GST_ERROR_OBJECT(srcpad, "failure: %s", gst_flow_get_name(result));
						gst_object_unref(srcpad);
						srcpad = NULL;
						goto done;
					}
				}
				gst_object_unref(srcpad);
				srcpad = NULL;
			}

			/*
			 * process simulated data
			 */

			for(FrameCPP::FrameH::simData_iterator current = frame->RefSimData().begin(), last = frame->RefSimData().end(); current != last; current++) {
				FrameCPP::FrSimData::data_type vects = (*current)->RefData();
				GstClockTime timestamp = frame_timestamp + (GstClockTimeDiff) round((*current)->GetTimeOffset() * 1e9);
				const char *name = (*current)->GetName().c_str();

				GST_LOG_OBJECT(element, "found FrSimData %s at %" GST_TIME_SECONDS_FORMAT, name, GST_TIME_SECONDS_ARGS(timestamp));

				/*
				 * retrieve the source pad.  create it if
				 * it doesn't exist.  update properties to
				 * reflect stream metadata.  if the pad has
				 * no peer or it not in the requested
				 * channel list, skip this channel.
				 */

				if(!is_requested_channel(element, name)) {
					GST_LOG_OBJECT(element, "skipping: channel not requested");
					continue;
				}
				srcpad = get_src_pad(element, name, GST_FRPAD_TYPE_FRSIMDATA, &pads_added);
				/* FIXME: units */
				g_object_set(srcpad,
					"comment", (*current)->GetComment().c_str(),
					NULL
				);
				if(!gst_pad_is_linked(srcpad)) {
					GST_LOG_OBJECT(srcpad, "skipping: not linked");
					gst_object_unref(srcpad);
					srcpad = NULL;
					continue;
				}

				/*
				 * convert FrVect to a GstBuffer and push
				 * out source pad, checking for disconts
				 * and recording state for next time.
				 */

				/* FIXME:  what about checking "dataValid" vect in the aux list? */
				if(vects.size()) {
					for(FrameCPP::FrSimData::data_type::iterator vect = vects.begin(), last_vect = vects.end(); vect != last_vect; vect++) {
						/* FIXME:  do something like this? */
						/*g_object_set(srcpad, "compression-scheme", vect->GetCompress(), NULL);*/
						result = frvect_to_buffer_and_push(element, srcpad, *vect, timestamp);
						if(result != GST_FLOW_OK) {
							GST_ERROR_OBJECT(srcpad, "failure: %s", gst_flow_get_name(result));
							gst_object_unref(srcpad);
							srcpad = NULL;
							goto done;
						}
					}
				} else {
					GST_LOG_OBJECT(srcpad, "no FrVects");
					result = push_heart_beat(element, srcpad, timestamp);
					if(result != GST_FLOW_OK) {
						GST_ERROR_OBJECT(srcpad, "failure: %s", gst_flow_get_name(result));
						gst_object_unref(srcpad);
						srcpad = NULL;
						goto done;
					}
				}
				gst_object_unref(srcpad);
				srcpad = NULL;
			}
		}
	} catch(const std::exception& Exception) {
		if(srcpad)
			gst_object_unref(srcpad);
		if(element->skip_bad_files)
			GST_ELEMENT_WARNING(element, STREAM, DECODE, (NULL), ("libframecpp raised exception: %s", Exception.what()));
		else {
			GST_ELEMENT_ERROR(element, STREAM, DECODE, (NULL), ("libframecpp raised exception: %s", Exception.what()));
			result = GST_FLOW_ERROR;
			goto done;
		}
	} catch(...) {
		if(srcpad)
			gst_object_unref(srcpad);
		if(element->skip_bad_files)
			GST_ELEMENT_WARNING(element, STREAM, DECODE, (NULL), ("libframecpp raised unknown exception"));
		else {
			GST_ELEMENT_ERROR(element, STREAM, DECODE, (NULL), ("libframecpp raised unknown exception"));
			result = GST_FLOW_ERROR;
			goto done;
		}
	}

	/*
	 * FIXME:  should src pads whose names weren't found in this frame
	 * file be removed?  this can also handle the case of the requested
	 * channel list having names removed from it
	 */

	/*
	 * Done
	 */

	GST_LOG_OBJECT(element, "finished IGWD file decode");

done:
	if(pads_added)
		gst_element_no_more_pads(GST_ELEMENT(element));
	gst_buffer_unref(inbuf);
	gst_object_unref(element);
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
 * properites
 */


enum property {
	ARG_DO_FILE_CHECKSUM = 1,
	ARG_SKIP_BAD_FILES,
	ARG_CHANNEL_LIST,
	ARG_FRAME_FORMAT_VERSION,
	ARG_FRAME_LIBRARY_VERSION,
	ARG_FRAME_LIBRARY_NAME,
	ARG_FRAME_NAME,
	ARG_FRAME_RUN,
	ARG_FRAME_NUMBER
};


static void set_property(GObject *object, guint id, const GValue *value, GParamSpec *pspec)
{
	GstFrameCPPChannelDemux *element = FRAMECPP_CHANNELDEMUX(object);

	GST_OBJECT_LOCK(element);

	switch((enum property) id) {
	case ARG_DO_FILE_CHECKSUM:
		element->do_file_checksum = g_value_get_boolean(value);
		break;

	case ARG_SKIP_BAD_FILES:
		element->skip_bad_files = g_value_get_boolean(value);
		break;

	case ARG_CHANNEL_LIST: {
		GValueArray *channel_list = (GValueArray *) g_value_get_boxed(value);
		guint i;

		g_hash_table_remove_all(element->channel_list);

		for(i = 0; i < channel_list->n_values; i++) {
			gchar *channel_name = g_value_dup_string(g_value_array_get_nth(channel_list, i));
			g_hash_table_replace(element->channel_list, channel_name, channel_name);
		}

		break;
	}

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, guint id, GValue *value, GParamSpec *pspec)
{
	GstFrameCPPChannelDemux *element = FRAMECPP_CHANNELDEMUX(object);

	GST_OBJECT_LOCK(element);

	switch((enum property) id) {
	case ARG_DO_FILE_CHECKSUM:
		g_value_set_boolean(value, element->do_file_checksum);
		break;

	case ARG_SKIP_BAD_FILES:
		g_value_set_boolean(value, element->skip_bad_files);
		break;

	case ARG_CHANNEL_LIST: {
		GValueArray *channel_list = g_value_array_new(0);
		GValue channel_name = {0};
		GHashTableIter iter;
		gchar *key, *ignored;

		g_value_init(&channel_name, G_TYPE_STRING);
		g_hash_table_iter_init(&iter, element->channel_list);

		while(g_hash_table_iter_next(&iter, (void **) &key, (void **) &ignored)) {
			g_value_set_string(&channel_name, key);
			g_value_array_append(channel_list, &channel_name);
			g_value_reset(&channel_name);
		}

		g_value_take_boxed(value, channel_list);
		break;
	}

	case ARG_FRAME_FORMAT_VERSION:
		g_value_set_uint(value, element->frame_format_version);
		break;

	case ARG_FRAME_LIBRARY_VERSION:
		g_value_set_uint(value, element->frame_library_version);
		break;

	case ARG_FRAME_LIBRARY_NAME:
		g_value_set_string(value, element->frame_library_name);
		break;

	case ARG_FRAME_NAME:
		g_value_set_string(value, element->frame_name);
		break;

	case ARG_FRAME_RUN:
		g_value_set_int(value, element->frame_run);
		break;

	case ARG_FRAME_NUMBER:
		g_value_set_uint(value, element->frame_number);
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


static void finalize(GObject * object)
{
	GstFrameCPPChannelDemux *element = FRAMECPP_CHANNELDEMUX(object);

	if(element->last_new_segment_event)
		gst_event_unref(element->last_new_segment_event);
	element->last_new_segment_event = NULL;
	g_hash_table_unref(element->channel_list);
	element->channel_list = NULL;
	gst_tag_list_free(element->tag_list);
	element->tag_list = NULL;
	g_free(element->frame_library_name);
	element->frame_library_name = NULL;
	g_free(element->frame_name);
	element->frame_name = NULL;

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


/*
 * Pad templates.
 */


static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE(
	"%s",
	GST_PAD_SRC,
	GST_PAD_SOMETIMES,
	GST_STATIC_CAPS(
		"audio/x-raw-float, " \
			"rate = (int) [1, MAX], " \
			"channels = (int) 1, " \
			"endianness = (int) BYTE_ORDER, " \
			"width = (int) {32, 64}; "\
		"audio/x-raw-int, " \
			"rate = (int) [1, MAX], " \
			"channels = (int) 1, " \
			"endianness = (int) BYTE_ORDER, " \
			"width = (int) 8, " \
			"depth = (int) [1, 8], " \
			"signed = (boolean) {true, false};" \
		"audio/x-raw-int, " \
			"rate = (int) [1, MAX], " \
			"channels = (int) 1, " \
			"endianness = (int) BYTE_ORDER, " \
			"width = (int) 16, " \
			"depth = (int) [1, 16], " \
			"signed = (boolean) {true, false};" \
		"audio/x-raw-int, " \
			"rate = (int) [1, MAX], " \
			"channels = (int) 1, " \
			"endianness = (int) BYTE_ORDER, " \
			"width = (int) 32, " \
			"depth = (int) [1, 32], " \
			"signed = (boolean) {true, false};" \
		"audio/x-raw-int, " \
			"rate = (int) [1, MAX], " \
			"channels = (int) 1, " \
			"endianness = (int) BYTE_ORDER, " \
			"width = (int) 64, " \
			"depth = (int) [1, 64], " \
			"signed = (boolean) {true, false};" \
	)
);


/*
 * base_init()
 */


static void framecpp_channeldemux_base_init(gpointer klass)
{
}


/*
 * class_init()
 */


static void framecpp_channeldemux_class_init(GstFrameCPPChannelDemuxClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	gst_element_class_set_details_simple(
		element_class,
		"IGWD frame file channel demuxer",
		"Codec/Demuxer",
		"demux streams from IGWD frame files (https://dcc.ligo.org/cgi-bin/DocDB/ShowDocument?docid=329)",
		"Kipp Cannon <kipp.cannon@ligo.org>, Ed Maros <ed.maros@ligo.org>"
	);

	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"sink",
			GST_PAD_SINK,
			GST_PAD_ALWAYS,
			gst_caps_new_simple(
				"application/x-igwd-frame",
				"framed", G_TYPE_BOOLEAN, TRUE,
				NULL
			)
		)
	);

	gst_element_class_add_pad_template(
		element_class,
		gst_static_pad_template_get(&src_factory)
	);

	g_object_class_install_property(
		gobject_class,
		ARG_DO_FILE_CHECKSUM,
		g_param_spec_boolean(
			"do-file-checksum",
			"Do file checksum",
			"Checks the file-level checksum of each input file (individual structure checksums are always checked).  This is costly for large (e.g., level 0) frame files, so it is disabled by default.",
			DEFAULT_DO_FILE_CHECKSUM,
			(GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_SKIP_BAD_FILES,
		g_param_spec_boolean(
			"skip-bad-files",
			"Ignore bad files",
			"Treat files that fail validation checks as though they are missing instead of raising an error.  The next buffers to be demultiplexed will be marked as discontinuities.",
			DEFAULT_SKIP_BAD_FILES,
			(GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_CHANNEL_LIST,
		g_param_spec_value_array(
			"channel-list",
			"Channel list",
			"Restrict demultiplexed channels to those in this list.  An empty list (default) causes all channels to be demultiplexed.  The use of this feature can improve performance when demultiplexing files with large numbers of channels;  it can be ignored for small files.  It is not an error for names in this list to not appear in the frame files.",
			g_param_spec_string(
				"channel",
				"Channel name",
				"Name of channel to demultiplex.",
				NULL,
				(GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)
			),
			(GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_FRAME_FORMAT_VERSION,
		g_param_spec_uint(
			"frame-format-version",
			"Frame format version",
			"Format version number from file header.",
			0, 255, DEFAULT_FRAME_FORMAT_VERSION,
			(GParamFlags) (G_PARAM_READABLE | G_PARAM_STATIC_STRINGS)
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_FRAME_LIBRARY_VERSION,
		g_param_spec_uint(
			"frame-library-version",
			"Frame library version",
			"Frame library version from file header.",
			0, 255, DEFAULT_FRAME_LIBRARY_VERSION,
			(GParamFlags) (G_PARAM_READABLE | G_PARAM_STATIC_STRINGS)
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_FRAME_LIBRARY_NAME,
		g_param_spec_string(
			"frame-library-name",
			"Frame library name",
			"Frame library name from file header.",
			DEFAULT_FRAME_LIBRARY_NAME,
			(GParamFlags) (G_PARAM_READABLE | G_PARAM_STATIC_STRINGS)
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_FRAME_NAME,
		g_param_spec_string(
			"frame-name",
			"Frame name",
			"Name appearing in frame header.",
			DEFAULT_FRAME_NAME,
			(GParamFlags) (G_PARAM_READABLE | G_PARAM_STATIC_STRINGS)
		)
	);
	G_PARAM_SPEC_STRING(g_object_class_find_property(gobject_class, "frame-name"))->ensure_non_null = TRUE;
	g_object_class_install_property(
		gobject_class,
		ARG_FRAME_RUN,
		g_param_spec_int(
			"frame-run",
			"Run number",
			"Run number appearing in frame header.",
			G_MININT, G_MAXINT, DEFAULT_FRAME_RUN,
			(GParamFlags) (G_PARAM_READABLE | G_PARAM_STATIC_STRINGS)
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_FRAME_NUMBER,
		g_param_spec_uint(
			"frame-number",
			"Frame number",
			"Current frame number.",
			0, G_MAXUINT, DEFAULT_FRAME_NUMBER,
			(GParamFlags) (G_PARAM_READABLE | G_PARAM_STATIC_STRINGS)
		)
	);
}


/*
 * instance_init()
 */


static void framecpp_channeldemux_init(GstFrameCPPChannelDemux *element, GstFrameCPPChannelDemuxClass *klass)
{
	GstPad *pad;

	gst_element_create_all_pads(GST_ELEMENT(element));

	/* configure sink pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "sink");
	gst_pad_set_chain_function(pad, GST_DEBUG_FUNCPTR(chain));
	gst_pad_set_event_function(pad, GST_DEBUG_FUNCPTR(sink_event));
	gst_object_unref(pad);

	/* internal data */
	element->last_new_segment_event = NULL;
	element->channel_list = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, NULL);
	element->tag_list = gst_tag_list_new();
	element->frame_format_version = DEFAULT_FRAME_FORMAT_VERSION;
	element->frame_library_version = DEFAULT_FRAME_LIBRARY_VERSION;
	element->frame_library_name = g_strdup(DEFAULT_FRAME_LIBRARY_NAME);
	element->frame_name = g_strdup(DEFAULT_FRAME_NAME);
	element->frame_run = DEFAULT_FRAME_RUN;
	element->frame_number = DEFAULT_FRAME_NUMBER;
}

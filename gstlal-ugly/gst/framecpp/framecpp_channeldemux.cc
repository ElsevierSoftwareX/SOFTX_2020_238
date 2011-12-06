/*
 * framecpp channel demultiplexor
 *
 * Copyright (C) 2011  Kipp Cannon, Ed Maros
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


/*
 * stuff from gstreamer
 */


#include <gst/gst.h>


/*
 * from framecpp
 */


#include <framecpp/Common/MemoryBuffer.hh>
#include <framecpp/IFrameStream.hh>
#include <framecpp/FrameH.hh>
#include <framecpp/FrAdcData.hh>
#include <framecpp/FrProcData.hh>
#include <framecpp/FrRawData.hh>
#include <framecpp/FrSimData.hh>
#include <framecpp/FrVect.hh>


/*
 * our own stuff
 */


#include <gstlal/gstlal_debug.h>
#include <gstlal/gstlal_tags.h>
#include <framecpp_channeldemux.h>


GST_DEBUG_CATEGORY(framecpp_channeldemux_debug);


using namespace FrameCPP;


/*
 * ============================================================================
 *
 *                                 Parameters
 *
 * ============================================================================
 */



#define GST_CAT_DEFAULT framecpp_channeldemux_debug


/*
 * ============================================================================
 *
 *                                 Utilities
 *
 * ============================================================================
 */


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
	GstClockTime next_timestamp;
	guint64 next_out_offset;
};


static guint pad_state_hash_func(gconstpointer key)
{
	return (uintptr_t) key;
}


static gboolean pad_state_key_equal_func(gconstpointer a, gconstpointer b)
{
	return a == b;
}


static struct pad_state *get_src_pad_state(GSTFrameCPPChannelDemux *element, const GstPad *pad)
{
	return (struct pad_state *) g_hash_table_lookup(element->pad_state, pad);
}


/*
 * split a string of the form "INSTRUMENT:CHANNEL" into two strings
 * containing the instrument and channel parts separately.  the pointers
 * placed in the instrument and channel pointers should be free()ed when no
 * longer needed.  returns 0 on success, < 0 on failure.
 */


static int split_name(const char *name, char **instrument, char **channel)
{
	const char *colon = strchr(name, ':');

	if(!colon) {
		*instrument = *channel = NULL;
		return -1;
	}

	*instrument = strndup(name, colon - name);
	*channel = strdup(colon + 1);

	return 0;
}


/*
 * tags
 */


static gboolean send_tags(GstPad *pad)
{
	char *instrument, *channel;
	GstTagList *taglist;
	GstEvent *event;

	split_name(GST_PAD_NAME(pad), &instrument, &channel);

	taglist = gst_tag_list_new_full(
		GSTLAL_TAG_INSTRUMENT, instrument,
		GSTLAL_TAG_CHANNEL_NAME, channel,
		GSTLAL_TAG_UNITS, "strain",	/* FIXME */
		NULL
	);
	g_assert(taglist);

	event = gst_event_new_tag(taglist);
	g_assert(event);

	return gst_pad_push_event(pad, event);
}


/*
 * linked event handler
 */


static void src_pad_linked(GstPad *pad, GstPad *peer, gpointer data)
{
	GSTFrameCPPChannelDemux *element = FRAMECPP_CHANNELDEMUX(gst_pad_get_parent(pad));
	struct pad_state *pad_state = get_src_pad_state(element, pad);

	/*
	 * reset pad's state
	 */

	pad_state->need_discont = TRUE;
	pad_state->next_timestamp = GST_CLOCK_TIME_NONE;
	pad_state->next_out_offset = 0;

	/*
	 * forward most recent new segment event
	 */

	if(element->last_new_segment) {
		gst_event_ref(element->last_new_segment);
		if(!gst_pad_push_event(pad, element->last_new_segment))
			GST_ERROR_OBJECT(element, "failed to push newsegment");
	}

	/*
	 * send tags
	 */

	if(!send_tags(pad))
		GST_ERROR_OBJECT(element, "failed to push tags");

	/*
	 * done
	 */

	gst_object_unref(element);
}


/*
 * get pad by name, creating it if it doesn't exist
 */


static GstPad *get_src_pad(GSTFrameCPPChannelDemux *element, const char *name)
{
	GstPad *srcpad = gst_element_get_static_pad(GST_ELEMENT(element), name);

	/*
	 * if element does not already have a pad by this name, create it
	 */

	if(!srcpad) {
		struct pad_state *pad_state;

		/*
		 * construct the pad
		 */

		srcpad = gst_pad_new_from_template(gst_element_class_get_pad_template(GST_ELEMENT_CLASS(G_OBJECT_GET_CLASS(element)), "src"), name);
		g_assert(srcpad != NULL);
		g_signal_connect(srcpad, "linked", (GCallback) src_pad_linked, NULL);

		/*
		 * create & initialize pad state
		 */

		pad_state = g_new(struct pad_state, 1);
		g_assert(pad_state != NULL);
		pad_state->need_discont = TRUE;
		pad_state->next_timestamp = GST_CLOCK_TIME_NONE;
		pad_state->next_out_offset = 0;
		g_hash_table_insert(element->pad_state, srcpad, pad_state);

		/*
		 * add pad to element.  must ref it because _add_pad()
		 * consumes a reference
		 */

		gst_pad_set_active(srcpad, TRUE);
		gst_object_ref(srcpad);
		gst_element_add_pad(GST_ELEMENT(element), srcpad);
	}

	return srcpad;
}


/*
 * transfer the contents of an FrVect into a newly-created GstBuffer.
 * caller must unref buffer when no longer needed.
 */


static GstCaps *FrVect_get_caps(General::SharedPtr < FrVect > vect)
{
	GstCaps *caps;
	gint rate = round(1.0 / vect->GetDim(0).GetDx());
	gint width = vect->GetTypeSize() * 8;

	switch(vect->GetType()) {
	case FrVect::FR_VECT_4R:
	case FrVect::FR_VECT_8R:
		caps = gst_caps_new_simple("audio/x-raw-float",
			"rate", G_TYPE_INT, rate,
			"channels", G_TYPE_INT, 1,
			"endianness", G_TYPE_INT, G_BYTE_ORDER,
			"width", G_TYPE_INT, width,
			NULL);
		break;

	case FrVect::FR_VECT_8C:
	case FrVect::FR_VECT_16C:
		caps = gst_caps_new_simple("audio/x-raw-complex",
			"rate", G_TYPE_INT, rate,
			"channels", G_TYPE_INT, 1,
			"endianness", G_TYPE_INT, G_BYTE_ORDER,
			"width", G_TYPE_INT, width,
			NULL);
		break;

	case FrVect::FR_VECT_C:
	case FrVect::FR_VECT_2S:
	case FrVect::FR_VECT_4S:
	case FrVect::FR_VECT_8S:
		caps = gst_caps_new_simple("audio/x-raw-int",
			"rate", G_TYPE_INT, rate,
			"channels", G_TYPE_INT, 1,
			"endianness", G_TYPE_INT, G_BYTE_ORDER,
			"width", G_TYPE_INT, width,
			"depth", G_TYPE_INT, width,
			"signed", G_TYPE_BOOLEAN, TRUE,
			NULL);
		break;

	case FrVect::FR_VECT_1U:
	case FrVect::FR_VECT_2U:
	case FrVect::FR_VECT_4U:
	case FrVect::FR_VECT_8U:
		caps = gst_caps_new_simple("audio/x-raw-int",
			"rate", G_TYPE_INT, rate,
			"channels", G_TYPE_INT, 1,
			"endianness", G_TYPE_INT, G_BYTE_ORDER,
			"width", G_TYPE_INT, width,
			"depth", G_TYPE_INT, width,
			"signed", G_TYPE_BOOLEAN, FALSE,
			NULL);
		break;

	default:
		g_assert_not_reached();
	}

	return caps;
}


static GstBuffer *FrVect_to_GstBuffer(General::SharedPtr < FrVect > vect, GstClockTime timestamp, guint64 offset)
{
	gint rate = round(1.0 / vect->GetDim(0).GetDx());
	GstBuffer *buffer;

	/*
	 * trigger data decompression before calling GetNBytes()
	 */

	vect->GetDataUncompressed();

	/*
	 * allocate buffer
	 */

	buffer = gst_buffer_new_and_alloc(vect->GetNBytes());
	if(!buffer)
		return NULL;

	/*
	 * copy data into buffer
	 */

	g_assert(vect->GetNDim() == 1);
	memcpy(GST_BUFFER_DATA(buffer), vect->GetData().get(), GST_BUFFER_SIZE(buffer));

	/*
	 * set timestamp and duration
	 */

	GST_BUFFER_TIMESTAMP(buffer) = timestamp;
	GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale_int(vect->GetNData(), GST_SECOND, rate);
	GST_BUFFER_OFFSET(buffer) = offset;
	GST_BUFFER_OFFSET_END(buffer) = offset + vect->GetNData();

	/*
	 * set buffer format
	 */

	GST_BUFFER_CAPS(buffer) = FrVect_get_caps(vect);
	g_assert(GST_BUFFER_CAPS(buffer) != NULL);

	/*
	 * done
	 */

	return buffer;
}


/*
 * convert an FrVect to a GstBuffer, and push out a source pad.
 */


static GstFlowReturn frvect_to_buffer_and_push(GSTFrameCPPChannelDemux *element, GstPad *pad, const char *name, General::SharedPtr < FrVect > vect, GstClockTime timestamp)
{
	struct pad_state *pad_state;
	GstBuffer *buffer;
	GstFlowReturn result = GST_FLOW_OK;

	/*
	 * retrieve source pad state
	 */

	pad_state = get_src_pad_state(element, pad);
	g_assert(pad_state != NULL);

	/*
	 * convert FrVect to GstBuffer
	 */

	buffer = FrVect_to_GstBuffer(vect, timestamp, pad_state->next_out_offset);
	g_assert(buffer != NULL);

	/*
	 * if the format matches the pad's replace the buffer's caps with
	 * the pad's to reduce the number of objects in memory and simplify
	 * subsequent comparisons
	 */

	if(gst_caps_is_equal(GST_BUFFER_CAPS(buffer), GST_PAD_CAPS(pad)))
		gst_buffer_set_caps(buffer, GST_PAD_CAPS(pad));

	/*
	 * check for disconts
	 */

	if(pad_state->need_discont || (GST_CLOCK_TIME_IS_VALID(pad_state->next_timestamp) && llabs(GST_BUFFER_TIMESTAMP(buffer) - pad_state->next_timestamp) > 1)) {
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

	GST_LOG_OBJECT(element, "pushing buffer on %s spanning %" GST_BUFFER_BOUNDARIES_FORMAT, name, GST_BUFFER_BOUNDARIES_ARGS(buffer));
	result = gst_pad_push(pad, buffer);

	/*
	 * done
	 */

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


static gboolean sink_event(GstPad *pad, GstEvent *event)
{
	GSTFrameCPPChannelDemux *element = FRAMECPP_CHANNELDEMUX(gst_pad_get_parent(pad));
	gboolean result = TRUE;

	switch(GST_EVENT_TYPE(event)) {
	case GST_EVENT_NEWSEGMENT:
		if(element->last_new_segment)
			gst_event_unref(element->last_new_segment);
		gst_event_ref(event);
		element->last_new_segment = event;
		break;

	default:
		break;
	}

	gst_object_unref(element);
	return result;
}


/*
 * chain()
 */


static GstFlowReturn chain(GstPad *pad, GstBuffer *inbuf)
{
	using FrameCPP::Common::MemoryBuffer;
	GSTFrameCPPChannelDemux *element = FRAMECPP_CHANNELDEMUX(gst_pad_get_parent(pad));
	IFrameStream::frame_h_type frame;
	GstPad *srcpad = NULL;
	GstFlowReturn result = GST_FLOW_OK;

	try {
		MemoryBuffer *ibuf(new MemoryBuffer(std::ios::in));

		ibuf->pubsetbuf((char *) GST_BUFFER_DATA(inbuf), GST_BUFFER_SIZE(inbuf));

		IFrameStream ifs(ibuf);

		GST_LOG_OBJECT(element, "found version %u frame file generated by library %d revision %d", ifs.Version(), ifs.FrameLibrary(), ifs.LibraryRevision());

		while((frame = ifs.ReadNextFrame())) {
			GstClockTime frame_timestamp = 1000000000L * frame->GetGTime().GetSeconds() + frame->GetGTime().GetNanoseconds();

			GST_LOG_OBJECT(element, "found frame %d at %" GST_TIME_SECONDS_FORMAT, ifs.GetCurrentFrameOffset(), GST_TIME_SECONDS_ARGS(frame_timestamp));

			/*
			 * process ADC data
			 */
	
			FrameH::rawData_type rd = frame->GetRawData();
			if(rd) {
				for(FrRawData::firstAdc_iterator current = rd->RefFirstAdc().begin(), last = rd->RefFirstAdc().end(); current != last; current++) {
					FrAdcData::data_type vects = (*current)->RefData();
					GstClockTime timestamp = frame_timestamp + (int) round((*current)->GetTimeOffset() * 1e9);
					const char *name = (*current)->GetName().c_str();

					GST_LOG_OBJECT(element, "found FrAdcData %s at %" GST_TIME_SECONDS_FORMAT, name, GST_TIME_SECONDS_ARGS(timestamp));

					/*
					 * retrieve the source pad.  create it if
					 * it doesn't exist.  if the pad has no
					 * peer, skip this channel.
					 */

					srcpad = get_src_pad(element, name);
					if(!gst_pad_is_linked(srcpad)) {
						GST_LOG_OBJECT(element, "skipping: not linked");
						gst_object_unref(srcpad);
						srcpad = NULL;
						continue;
					}

					/*
					 * convert FrVect to a GstBuffer
					 * and push out source pad,
					 * checking for disconts and
					 * recording state for next time.
					 */

					result = frvect_to_buffer_and_push(element, srcpad, name, vects[0], timestamp);
					gst_object_unref(srcpad);
					srcpad = NULL;
					if(result != GST_FLOW_OK) {
						GST_ERROR_OBJECT(element, "failure: %s", gst_flow_get_name(result));
						goto done;
					}
				}
			}

			/*
			 * process proc data
			 */

			for(FrameH::procData_iterator current = frame->RefProcData().begin(), last = frame->RefProcData().end(); current != last; current++) {
				FrProcData::data_type vects = (*current)->RefData();
				GstClockTime timestamp = frame_timestamp + (int) round((*current)->GetTimeOffset() * 1e9);
				const char *name = (*current)->GetName().c_str();

				GST_LOG_OBJECT(element, "found FrProcData %s at %" GST_TIME_SECONDS_FORMAT, name, GST_TIME_SECONDS_ARGS(timestamp));

				/*
				 * retrieve the source pad.  create it if
				 * it doesn't exist.  if the pad has no
				 * peer, skip this channel.
				 */

				srcpad = get_src_pad(element, name);
				if(!gst_pad_is_linked(srcpad)) {
					GST_LOG_OBJECT(element, "skipping: not linked");
					gst_object_unref(srcpad);
					srcpad = NULL;
					continue;
				}

				/*
				 * convert FrVect to a GstBuffer and push
				 * out source pad, checking for disconts
				 * and recording state for next time.
				 */

				result = frvect_to_buffer_and_push(element, srcpad, name, vects[0], timestamp);
				gst_object_unref(srcpad);
				srcpad = NULL;
				if(result != GST_FLOW_OK) {
					GST_ERROR_OBJECT(element, "failure: %s", gst_flow_get_name(result));
					goto done;
				}
			}

			/*
			 * process simulated data
			 */

			for(FrameH::simData_iterator current = frame->RefSimData().begin(), last = frame->RefSimData().end(); current != last; current++) {
				FrSimData::data_type vects = (*current)->RefData();
				GstClockTime timestamp = frame_timestamp + (int) round((*current)->GetTimeOffset() * 1e9);
				const char *name = (*current)->GetName().c_str();

				GST_LOG_OBJECT(element, "found FrSimData %s at %" GST_TIME_SECONDS_FORMAT, name, GST_TIME_SECONDS_ARGS(timestamp));

				/*
				 * retrieve the source pad.  create it if
				 * it doesn't exist.  if the pad has no
				 * peer, skip this channel.
				 */

				srcpad = get_src_pad(element, name);
				if(!gst_pad_is_linked(srcpad)) {
					GST_LOG_OBJECT(element, "skipping: not linked");
					gst_object_unref(srcpad);
					srcpad = NULL;
					continue;
				}

				/*
				 * convert FrVect to a GstBuffer and push
				 * out source pad, checking for disconts
				 * and recording state for next time.
				 */

				result = frvect_to_buffer_and_push(element, srcpad, name, vects[0], timestamp);
				gst_object_unref(srcpad);
				srcpad = NULL;
				if(result != GST_FLOW_OK) {
					GST_ERROR_OBJECT(element, "failure: %s", gst_flow_get_name(result));
					goto done;
				}
			}
		}
	} catch(...) {
		if(srcpad)
			gst_object_unref(srcpad);
		result = GST_FLOW_ERROR;
	}

	/*
	 * Done
	 */

done:
	gst_buffer_unref(inbuf);
	gst_object_unref(element);
	return result;
}


/*
 * ============================================================================
 *
 *                           GStreamer Element
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


static void finalize(GObject * object)
{
	GSTFrameCPPChannelDemux *element = FRAMECPP_CHANNELDEMUX(object);

	if(element->last_new_segment)
		gst_event_unref(element->last_new_segment);
	element->last_new_segment = NULL;
	g_hash_table_unref(element->pad_state);
	element->pad_state = NULL;

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


/*
 * Pad templates.
 */


static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE(
	"src",
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
			"depth = (int) 8, " \
			"signed = (boolean) {true, false};" \
		"audio/x-raw-int, " \
			"rate = (int) [1, MAX], " \
			"channels = (int) 1, " \
			"endianness = (int) BYTE_ORDER, " \
			"width = (int) 16, " \
			"depth = (int) 16, " \
			"signed = (boolean) {true, false};" \
		"audio/x-raw-int, " \
			"rate = (int) [1, MAX], " \
			"channels = (int) 1, " \
			"endianness = (int) BYTE_ORDER, " \
			"width = (int) 32, " \
			"depth = (int) 32, " \
			"signed = (boolean) {true, false};" \
		"audio/x-raw-int, " \
			"rate = (int) [1, MAX], " \
			"channels = (int) 1, " \
			"endianness = (int) BYTE_ORDER, " \
			"width = (int) 64, " \
			"depth = (int) 64, " \
			"signed = (boolean) {true, false};" \
	)
);


/*
 * base_init()
 */


static void base_init(gpointer klass)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);

	gst_element_class_set_details_simple(
		element_class,
		"IGWD frame channel demuxer",
		"Codec/Demuxer",
		"IGWD frame channel demuxer",
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
				NULL
			)
		)
	);

	gst_element_class_add_pad_template(
		element_class,
		gst_static_pad_template_get(&src_factory)
	);
}


/*
 * class_init()
 */


static void class_init(gpointer klass, gpointer klass_data)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

	parent_class = (GstElementClass *) g_type_class_ref(GST_TYPE_ELEMENT);

	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);
}


/*
 * instance_init()
 */


static void instance_init(GTypeInstance *object, gpointer klass)
{
	GSTFrameCPPChannelDemux *element = FRAMECPP_CHANNELDEMUX(object);
	GstPad *pad;

	gst_element_create_all_pads(GST_ELEMENT(element));

	/* configure sink pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "sink");
	gst_pad_set_chain_function(pad, GST_DEBUG_FUNCPTR(chain));
	gst_pad_set_event_function(pad, GST_DEBUG_FUNCPTR(sink_event));
	gst_object_unref(pad);

	/* internal data */
	element->last_new_segment = NULL;
	element->pad_state = g_hash_table_new_full(pad_state_hash_func, pad_state_key_equal_func, NULL, g_free);
}


/*
 * framecpp_channeldemux_get_type().
 */


GType framecpp_channeldemux_get_type(void)
{
	static GType type = 0;

	if(!type) {
		static const GTypeInfo info = {
			sizeof(GSTFrameCPPChannelDemuxClass), /* class_size */
			base_init, /* base_init */
			NULL, /* base_finalize */
			class_init, /* class_init */
			NULL, /* class_finalize */
			NULL, /* class_data */
			sizeof(GSTFrameCPPChannelDemux), /* instance_size */
			0, /* n_preallocs */
			instance_init, /* instance_init */
			NULL /* value_table */
		};
		type = g_type_register_static(GST_TYPE_ELEMENT, "framecpp_channeldemux", &info, (GTypeFlags) 0);
	}

	return type;
}

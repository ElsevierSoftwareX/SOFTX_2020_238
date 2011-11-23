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
 * transfer the contents of an FrVect into a newly-created GstBuffer.
 * caller must unref buffer when no longer needed.
 */


static GstBuffer *FrVect_to_GstBuffer(General::SharedPtr < FrVect > vect, GstClockTime epoch)
{
	gint rate = round(1.0 / vect->GetDim(0).GetDx());
	gint width = vect->GetTypeSize() * 8;
	GstBuffer *buffer = gst_buffer_new_and_alloc(vect->GetNBytes());

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

	GST_BUFFER_TIMESTAMP(buffer) = epoch;
	GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale_int(vect->GetNData(), GST_SECOND, rate);
	GST_BUFFER_OFFSET(buffer) = 0;
	GST_BUFFER_OFFSET_END(buffer) = vect->GetNData();

	/*
	 * set buffer format
	 */

	switch(vect->GetType()) {
	case FrVect::FR_VECT_4R:
	case FrVect::FR_VECT_8R:
		GST_BUFFER_CAPS(buffer) = gst_caps_new_simple("audio/x-raw-float",
			"rate", G_TYPE_INT, rate,
			"channels", G_TYPE_INT, 1,
			"endianness", G_TYPE_INT, G_BYTE_ORDER,
			"width", G_TYPE_INT, width,
			NULL);
		break;

	case FrVect::FR_VECT_8C:
	case FrVect::FR_VECT_16C:
		GST_BUFFER_CAPS(buffer) = gst_caps_new_simple("audio/x-raw-complex",
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
		GST_BUFFER_CAPS(buffer) = gst_caps_new_simple("audio/x-raw-int",
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
		GST_BUFFER_CAPS(buffer) = gst_caps_new_simple("audio/x-raw-int",
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

	return buffer;
}


/*
 * ============================================================================
 *
 *                           GStreamer Element
 *
 * ============================================================================
 */


/*
 * chain()
 */


static GstFlowReturn chain(GstPad *pad, GstBuffer *inbuf)
{
	using FrameCPP::Common::MemoryBuffer;
	GSTFrameCPPChannelDemux *element = FRAMECPP_CHANNELDEMUX(gst_pad_get_parent(pad));
	IFrameStream::frame_h_type frame;
	GstPadTemplate *src_pad_template = gst_element_class_get_pad_template(GST_ELEMENT_CLASS(FRAMECPP_CHANNELDEMUX_GET_CLASS(element)), "src");
	GstBuffer *outbuf = NULL;
	GstPad *srcpad = NULL;
	GstFlowReturn result = GST_FLOW_OK;

	g_assert(src_pad_template != NULL);

	try {
		MemoryBuffer *ibuf(new MemoryBuffer(std::ios::in));

		ibuf->pubsetbuf((char *) GST_BUFFER_DATA(inbuf), GST_BUFFER_SIZE(inbuf));

		IFrameStream ifs(ibuf);
		frame = ifs.ReadNextFrame();
		GstClockTime frame_timestamp = 1000000000L * frame->GetGTime().GetSeconds() + frame->GetGTime().GetNanoseconds();

		GST_LOG_OBJECT(element, "found version %d frame file", ifs.Version());
		GST_LOG_OBJECT(element, "found frame at %lu.%09lu s", frame->GetGTime().GetSeconds(), frame->GetGTime().GetNanoseconds());

		/*
		 * process ADCs
		 */
	
		FrameH::rawData_type rd = frame->GetRawData();
		if(rd) {
			FrRawData::firstAdc_iterator current = rd->RefFirstAdc().begin(), last = rd->RefFirstAdc().end();
			for(; current != last; ++current) {
				FrAdcData::data_type vects = (*current)->RefData();
				double timeOffset = (*current)->GetTimeOffset();
				const char *name = (*current)->GetName().c_str();
				char *instrument, *channel;

				GST_LOG_OBJECT(element, "found FrAdc %s", name);

				/*
				 * retrieve the source pad.  this will
				 * induce it to be created if it doesn't
				 * exist.
				 */

				srcpad = gst_element_request_pad(GST_ELEMENT(element), src_pad_template, name, NULL);
				g_assert(srcpad != NULL);

				/* FIXME:  keep track of these, and send
				 * tags events when they change */
				split_name(name, &instrument, &channel);
				free(instrument);
				free(channel);

				/*
				 * construct output buffer.  if they're
				 * equal, replace the buffer's caps with
				 * the pad's to reduce the number of caps
				 * objects in play and simplify subsequent
				 * comparisons
				 */

				outbuf = FrVect_to_GstBuffer(vects[0], frame_timestamp + (int) round(timeOffset * 1e9));
				g_assert(outbuf != NULL);
				if(gst_caps_is_equal(GST_BUFFER_CAPS(outbuf), GST_PAD_CAPS(srcpad)))
					gst_buffer_set_caps(outbuf, GST_PAD_CAPS(srcpad));

				/*
				 * push buffer downstream
				 */

				result = gst_pad_push(srcpad, outbuf);
				outbuf = NULL;
				if(result != GST_FLOW_OK) {
					GST_ERROR_OBJECT(element, "failed to push buffer");
					goto done;
				}
				gst_object_unref(srcpad);
				srcpad = NULL;
			}
		}

	} catch(...) {
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
 * Parent class.
 */


static GstElementClass *parent_class = NULL;


/*
 * Instance finalize function.  See ???
 */


static void finalize(GObject * object)
{
	GSTFrameCPPChannelDemux *element = FRAMECPP_CHANNELDEMUX(object);

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
	gst_object_unref(pad);

	/* internal data */
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

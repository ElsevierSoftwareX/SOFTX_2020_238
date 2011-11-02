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


#include <gstlal_tags.h>
#include <framecpp_channeldemux.h>


GST_DEBUG_CATEGORY(framecpp_channeldemux_debug);


/*
 * ============================================================================
 *
 *			     GStreamer Element
 *
 * ============================================================================
 */


/*
 * chain()
 */


static GstFlowReturn chain(GstPad *pad, GstBuffer *buf)
{
	using namespace FrameCPP;
	using FrameCPP::Common::FrameBuffer;
	GSTFrameCPPChannelDemux *element = FRAMECPP_CHANNELDEMUX(gst_pad_get_parent(pad));
	GstFlowReturn result = GST_FLOW_OK;

	/*
	 * to be removed and replaced with the proper buffer when framecpp
	 * api is updated
	 */

	void *frame_image = GST_BUFFER_DATA(buf);
	FrameH *frame = NULL;

	try {
		FrameBuffer< std::filebuf >*
			ibuf(new FrameBuffer< std::filebuf >(std::ios::in));
		IFrameStream ifs(ibuf);
		frame = ifs.ReadNextFrame();

		/*
		 * process ADCs
		 */
	
		FrameH::rawData_type *rd = frame->GetRawData();
		if(rd) {
			FrRawData::firstAdc_iterator
				current = rd->RefFirstAdc().begin(),
				last = rd->RefFirstAdc().end();
			for(; current != last; ++current) {
				FrAdcData::data_type vects = (*current)->RefData();
				FrVect *vect = vects[0];
				gint rate = round(1.0 / vect->GetDim(0).GetDx());
				gint width = vect->GetTypeSize() * 8;
				GstCaps *caps;

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
			}
		}

	} catch(...) {
		result = GST_FLOW_ERROR;
	}

	/*
	 * Done
	 */

done:
	delete frame;
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

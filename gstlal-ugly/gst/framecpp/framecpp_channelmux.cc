/*
 * framecpp channel multiplexor
 *
 * Copyright (C) 2012  Kipp Cannon, Ed Maros
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


/*
 * from framecpp
 */


#include <framecpp/Common/MemoryBuffer.hh>
#include <framecpp/Common/Verify.hh>
#include <framecpp/OFrameStream.hh>
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
#include <framecpp_channelmux.h>
#include <muxcollectpads.h>
#include <muxqueue.h>


/*
 * ============================================================================
 *
 *                                Boilerplate
 *
 * ============================================================================
 */


#define GST_CAT_DEFAULT framecpp_channelmux_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


static void additional_initializations(GType type)
{
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "framecpp_channelmux", 0, "framecpp_channelmux element");
}


GST_BOILERPLATE_FULL(GstFrameCPPChannelMux, framecpp_channelmux, GstElement, GST_TYPE_ELEMENT, additional_initializations);


/*
 * ============================================================================
 *
 *                                 Parameters
 *
 * ============================================================================
 */


#define DEFAULT_FRAME_DURATION 1
#define DEFAULT_FRAMES_PER_FILE 128


#define FRAME_FILE_DURATION(mux) ((mux)->frames_per_file * (mux)->frame_duration)


/*
 * ============================================================================
 *
 *                                  Src Pad
 *
 * ============================================================================
 */


/*
 * forwards the event to all sinkpads, takes ownership of the event
 */


typedef struct {
	GstEvent *event;
	gboolean flush;
} EventData;


static gboolean forward_src_event_func(GstPad *pad, GValue *ret, EventData *data)
{
	gst_event_ref(data->event);
	if(!gst_pad_push_event(pad, data->event)) {
		/* quick hack to unflush the pads. ideally we need  a way
		 * to just unflush this single collect pad */
		if(data->flush)
			gst_pad_send_event(pad, gst_event_new_flush_stop());
	} else {
		g_value_set_boolean(ret, TRUE);
	}
	gst_object_unref(GST_OBJECT(pad));
	return TRUE;
}


static gboolean forward_src_event(GstFrameCPPChannelMux *mux, GstEvent *event, gboolean flush)
{
	GstIterator *it;
	GValue vret = {0};
	EventData data = {
		event,
		flush
	};
	gboolean success;

	g_value_init(&vret, G_TYPE_BOOLEAN);
	g_value_set_boolean(&vret, FALSE);

	it = gst_element_iterate_sink_pads(GST_ELEMENT(mux));
	while(TRUE) {
		switch(gst_iterator_fold(it, (GstIteratorFoldFunction) forward_src_event_func, &vret, &data)) {
		case GST_ITERATOR_RESYNC:
			gst_iterator_resync(it);
			g_value_set_boolean(&vret, TRUE);
			break;

		case GST_ITERATOR_OK:
		case GST_ITERATOR_DONE:
			success = g_value_get_boolean(&vret);
			goto done;

		default:
			success = FALSE;
			goto done;
		}
	}
done:
	gst_iterator_free(it);
	gst_event_unref(event);

	return success;
}


/*
 * handle events received on the source pad
 */


static gboolean src_event(GstPad *pad, GstEvent *event)
{
	GstFrameCPPChannelMux *mux = FRAMECPP_CHANNELMUX(gst_pad_get_parent(pad));
	gboolean success;

	switch(GST_EVENT_TYPE(event)) {
	case GST_EVENT_SEEK: {
		gdouble rate;
		GstSeekFlags flags;
		GstSeekType curtype, endtype;
		gint64 cur, end;
		gboolean flush;

		gst_event_parse_seek(event, &rate, NULL, &flags, &curtype, &cur, &endtype, &end);
		flush = flags & GST_SEEK_FLAG_FLUSH;

		/* FIXME:  copy the adder's logic re flushing */

		success = forward_src_event(mux, event, flush);
		break;
	}

	/* events that can't be handled */
	case GST_EVENT_QOS:
	case GST_EVENT_NAVIGATION:
		gst_event_unref(event);
		success = FALSE;
		break;

	/* forward the rest out all sink pads */
	default:
		success = forward_src_event(mux, event, FALSE);
		break;
	}

	gst_object_unref(GST_OBJECT(mux));
	return success;
}


/*
 * ============================================================================
 *
 *                                 Sink Pads
 *
 * ============================================================================
 */


/*
 * get informed of a sink pad's caps
 */


static gboolean sink_setcaps(GstPad *pad, GstCaps *caps)
{
	GstFrameCPPChannelMux *mux = FRAMECPP_CHANNELMUX(gst_pad_get_parent(pad));
	GstStructure *structure;
	int width, channels, rate;
	const gchar *media_type;
	gboolean success = TRUE;

	/*
	 * parse caps, set unit size on collect pads
	 */

	structure = gst_caps_get_structure(caps, 0);
	media_type = gst_structure_get_name(structure);
	gst_structure_get_int(structure, "width", &width);
	gst_structure_get_int(structure, "channels", &channels);
	gst_structure_get_int(structure, "rate", &rate);
	if(!strcmp(media_type, "audio/x-raw-int")) {
	} else if(!strcmp(media_type, "audio/x-raw-float")) {
	} else if(!strcmp(media_type, "audio/x-raw-complex")) {
	} else
		success = FALSE;

	if(success) {
		GObject *queue;
		GST_OBJECT_LOCK(mux->collect);
		queue = G_OBJECT(((FrameCPPMuxCollectPadsData *) gst_pad_get_element_private(pad))->queue);
		g_object_set(queue, "rate", (gint) rate, NULL);
		g_object_set(queue, "unit-size", (guint) (width / 8 * channels), NULL);
		GST_OBJECT_UNLOCK(mux->collect);
	}

	gst_object_unref(GST_OBJECT(mux));
	return success;
}


/*
 * handle sink pad events
 */


static gboolean sink_event(GstPad *pad, GstEvent *event)
{
	GstFrameCPPChannelMux *mux = FRAMECPP_CHANNELMUX(gst_pad_get_parent(pad));
	gboolean success = TRUE;

	switch(GST_EVENT_TYPE(event)) {
	case GST_EVENT_EOS:
		/* FIXME:  flush final frame file? */
		break;

	default:
		break;
	}

	gst_pad_push_event(mux->srcpad, event);

	gst_object_unref(GST_OBJECT(mux));
	return success;
}


/*
 * create a new sink pad and add to element.  does not check if name is
 * already in use
 */


static GstPad *request_new_pad(GstElement *element, GstPadTemplate *templ, const gchar *name)
{
	GstFrameCPPChannelMux *mux = FRAMECPP_CHANNELMUX(element);
	FrameCPPMuxCollectPadsData *data;
	GstPad *pad;

	/*
	 * construct the pad
	 */

	pad = gst_pad_new_from_template(templ, name);
	if(!pad)
		goto no_pad;
	gst_pad_set_setcaps_function(pad, GST_DEBUG_FUNCPTR(sink_setcaps));

	/*
	 * add pad to element.
	 */

	GST_OBJECT_LOCK(GST_OBJECT(mux->collect));
	data = framecpp_muxcollectpads_add_pad(mux->collect, pad);
	if(!data)
		goto could_not_add_to_collectpads;
	framecpp_muxcollectpads_set_event_function(data, GST_DEBUG_FUNCPTR(sink_event));
	if(!gst_element_add_pad(element, pad))
		goto could_not_add_to_element;
	GST_OBJECT_UNLOCK(GST_OBJECT(mux->collect));

	/*
	 * done
	 */

	return pad;

	/*
	 * errors
	 */

could_not_add_to_element:
	framecpp_muxcollectpads_remove_pad(mux->collect, pad);
could_not_add_to_collectpads:
	gst_object_unref(pad);
	GST_OBJECT_UNLOCK(GST_OBJECT(mux->collect));
no_pad:
	return NULL;
}


/*
 * remove a sink pad from element
 */


static void release_pad(GstElement *element, GstPad *pad)
{
	GstFrameCPPChannelMux *mux = FRAMECPP_CHANNELMUX(element);

	GST_OBJECT_LOCK(GST_OBJECT(mux->collect));
	framecpp_muxcollectpads_remove_pad(mux->collect, pad);
	gst_element_remove_pad(element, pad);
	GST_OBJECT_UNLOCK(GST_OBJECT(mux->collect));
}


/*
 * collected signal handler
 */


static GstFlowReturn collected_handler(FrameCPPMuxCollectPads *collectpads, GstClockTime collected_t_start, GstClockTime collected_t_end, GstFrameCPPChannelMux *mux)
{
	GstClockTime gwf_t_start, gwf_t_end;
	GstFlowReturn result = GST_FLOW_OK;

	/*
	 * make sure we don't have our wires crossed
	 */

	g_assert(GST_IS_FRAMECPP_MUXCOLLECTPADS(mux));
	g_assert(mux->collect == collectpads);
	g_assert(GST_CLOCK_TIME_IS_VALID(collected_t_start));
	g_assert(GST_CLOCK_TIME_IS_VALID(collected_t_end));

	/*
	 * loop over available data
	 */

	for(gwf_t_start = collected_t_start, gwf_t_end = collected_t_start - collected_t_start % FRAME_FILE_DURATION(mux) + FRAME_FILE_DURATION(mux); gwf_t_end <= collected_t_end; gwf_t_start = gwf_t_end, gwf_t_end += FRAME_FILE_DURATION(mux)) {
		GstBuffer *outbuf = NULL;

		/*
		 * build frame file
		 */

		try {
			FrameCPP::Common::MemoryBuffer *obuf(new FrameCPP::Common::MemoryBuffer(std::ios::out));
			FrameCPP::OFrameStream ofs(obuf);
			GstClockTime frame_t_start, frame_t_end;

			/* FIXME: initialize frame file */

			/*
			 * loop over frames
			 */

			for(frame_t_start = gwf_t_start, frame_t_end = gwf_t_start + mux->frame_duration; frame_t_end <= gwf_t_end; frame_t_start = frame_t_end, frame_t_end += mux->frame_duration) {
				GSList *collectdatalist;

				/* FIXME:  initialize frame */

				/*
				 * loop over pads
				 */

				for(collectdatalist = collectpads->pad_list; collectdatalist; collectdatalist = g_slist_next(collectdatalist)) {
					FrameCPPMuxCollectPadsData *data = (FrameCPPMuxCollectPadsData *) collectdatalist->data;
					/* we own this list */
					GList *buffer_list = framecpp_muxcollectpads_take_list(data, frame_t_end);

					/* FIXME:  initialize FrProc */

					for(; buffer_list; buffer_list = g_list_delete_link(buffer_list, buffer_list)) {
						GstBuffer *buffer = GST_BUFFER(buffer_list->data);

						/* FIXME:  append/write/memcpy/whatever buffer contents into FrProc */

						gst_buffer_unref(buffer);
					}

					/* FIXME:  add FrProc to frame */
				}

				/* FIXME:  add frame to frame file */
			}
			g_assert(frame_t_start == gwf_t_end);	/* safety check */

			/* FIXME:  finish/close/whatever frame file */

			/* FIXME:  extract/copy/whatever bytes into GstBuffer */
			outbuf = gst_buffer_new_and_alloc(0);
			GST_BUFFER_TIMESTAMP(outbuf) = gwf_t_start;
			GST_BUFFER_DURATION(outbuf) = gwf_t_end - gwf_t_start;

		} catch(const std::exception& Exception) {
			GST_ELEMENT_ERROR(mux, STREAM, ENCODE, (NULL), ("libframecpp raised exception: %s", Exception.what()));
			result = GST_FLOW_ERROR;
			goto done;
		} catch(...) {
			GST_ELEMENT_ERROR(mux, STREAM, ENCODE, (NULL), ("libframecpp raised unknown exception"));
			result = GST_FLOW_ERROR;
			goto done;
		}

		/*
		 * push downstream
		 */

		result = gst_pad_push(mux->srcpad, outbuf);
		if(result != GST_FLOW_OK) {
			GST_ELEMENT_ERROR(mux, CORE, PAD, (NULL), ("gst_pad_push() failed (%s)", gst_flow_get_name(result)));
			goto done;
		}
	}

	g_assert(gwf_t_start != collected_t_start);	/* it's a deadlock if we haven't consumed any data */

	/*
	 * done
	 */

done:
	return result;
}


/*
 * ============================================================================
 *
 *                            GstElement Overrides
 *
 * ============================================================================
 */


GstStateChangeReturn change_state(GstElement *element, GstStateChange transition)
{
	GstFrameCPPChannelMux *mux = FRAMECPP_CHANNELMUX(element);
	GstStateChangeReturn result;

	switch(transition) {
	case GST_STATE_CHANGE_READY_TO_PAUSED:
		framecpp_muxcollectpads_start(mux->collect);
		break;

	case GST_STATE_CHANGE_PAUSED_TO_READY:
		framecpp_muxcollectpads_stop(mux->collect);
		break;

	default:
		break;
	}

	result = GST_ELEMENT_CLASS(parent_class)->change_state(element, transition);

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
 * properties
 */


enum property {
	ARG_FRAME_DURATION = 1,
	ARG_FRAMES_PER_FILE
};


static void set_property(GObject *object, guint id, const GValue *value, GParamSpec *pspec)
{
	GstFrameCPPChannelMux *element = FRAMECPP_CHANNELMUX(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_FRAME_DURATION:
		element->frame_duration = g_value_get_int(value) * GST_SECOND;
		g_object_set(G_OBJECT(element->collect), "max-size-time", (guint64) (element->frames_per_file * element->frame_duration), NULL);
		break;

	case ARG_FRAMES_PER_FILE:
		element->frames_per_file = g_value_get_int(value);
		g_object_set(G_OBJECT(element->collect), "max-size-time", (guint64) (element->frames_per_file * element->frame_duration), NULL);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, guint id, GValue *value, GParamSpec *pspec)
{
	GstFrameCPPChannelMux *element = FRAMECPP_CHANNELMUX(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_FRAME_DURATION:
		g_value_set_int(value, element->frame_duration / GST_SECOND);
		break;

	case ARG_FRAMES_PER_FILE:
		g_value_set_int(value, element->frames_per_file);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * Instance finalize function.  See ???
 */


static void finalize(GObject * object)
{
	GstFrameCPPChannelMux *element = FRAMECPP_CHANNELMUX(object);

	if(element->collect)
		gst_object_unref(GST_OBJECT(element->collect));
	element->collect = NULL;
	if(element->srcpad)
		gst_object_unref(element->srcpad);
	element->srcpad = NULL;

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


/*
 * Pad templates.
 */


static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE(
	"sink_%d",
	GST_PAD_SINK,
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


static void framecpp_channelmux_base_init(gpointer klass)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);

	gst_element_class_set_details_simple(
		element_class,
		"IGWD frame file channel muxer",
		"Codec/Muxer",
		"mux streams into IGWD frame files (https://dcc.ligo.org/cgi-bin/DocDB/ShowDocument?docid=329)",
		"Kipp Cannon <kipp.cannon@ligo.org>, Ed Maros <ed.maros@ligo.org>"
	);

	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"src",
			GST_PAD_SRC,
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
		gst_static_pad_template_get(&sink_factory)
	);
}


/*
 * class_init()
 */


static void framecpp_channelmux_class_init(GstFrameCPPChannelMuxClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *gstelement_class = GST_ELEMENT_CLASS(klass);

	parent_class = (GstElementClass *) g_type_class_ref(GST_TYPE_ELEMENT);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	g_object_class_install_property(
		gobject_class,
		ARG_FRAME_DURATION,
		g_param_spec_int(
			"frame-duration",
			"Frame duration",
			"Duration of each frame in seconds.",
			1, G_MAXINT, DEFAULT_FRAME_DURATION,
			(GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_FRAMES_PER_FILE,
		g_param_spec_int(
			"frames-per-file",
			"Frames per file",
			"Number of frames in each frame file.",
			1, G_MAXINT, DEFAULT_FRAMES_PER_FILE,
			(GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)
		)
	);

	gstelement_class->request_new_pad = GST_DEBUG_FUNCPTR(request_new_pad);
	gstelement_class->release_pad = GST_DEBUG_FUNCPTR(release_pad);
	gstelement_class->change_state = GST_DEBUG_FUNCPTR(change_state);
}


/*
 * instance_init()
 */


static void framecpp_channelmux_init(GstFrameCPPChannelMux *mux, GstFrameCPPChannelMuxClass *kclass)
{
	gst_element_create_all_pads(GST_ELEMENT(mux));

	/* configure src pad */
	mux->srcpad = gst_element_get_static_pad(GST_ELEMENT(mux), "src");
	/*gst_pad_set_query_function(pad, GST_DEBUG_FUNCPTR(src_query));*/ /* FIXME:  implement */
	gst_pad_set_event_function(mux->srcpad, GST_DEBUG_FUNCPTR(src_event));

	/* configure collect pads.  max-size-time will get set when our
	 * properties are initialized */
	mux->collect = FRAMECPP_MUXCOLLECTPADS(g_object_new(FRAMECPP_MUXCOLLECTPADS_TYPE, NULL));
	g_signal_connect(G_OBJECT(mux->collect), "collected", G_CALLBACK(collected_handler), mux);
}

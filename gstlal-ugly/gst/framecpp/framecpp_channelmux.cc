/*
 * framecpp channel multiplexor
 *
 * Copyright (C) 2012,2013  Kipp Cannon, Ed Maros
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
#include <framecpp/Dimension.hh>
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
#include <gstfrpad.h>
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
#define DEFAULT_FRAME_NAME ""
#define DEFAULT_FRAME_RUN -1
#define DEFAULT_FRAME_NUMBER 0


#define FRAME_FILE_DURATION(mux) ((mux)->frames_per_file * (mux)->frame_duration)


/*
 * ============================================================================
 *
 *                                 Utilities
 *
 * ============================================================================
 */


static void update_instruments(GstFrameCPPChannelMux *mux)
{
	GstIterator *it = gst_element_iterate_sink_pads(GST_ELEMENT(mux));

	g_hash_table_remove_all(mux->instruments);

	while(TRUE) {
		GstPad *pad;
		gchar *instrument = NULL;
		switch(gst_iterator_next(it, (void **) &pad)) {
		case GST_ITERATOR_OK:
			g_object_get(pad, "instrument", &instrument, NULL);
			if(instrument)
				g_hash_table_replace(mux->instruments, instrument, instrument);
			gst_object_unref(pad);
			break;

		case GST_ITERATOR_RESYNC:
			g_hash_table_remove_all(mux->instruments);
			gst_iterator_resync(it);
			break;

		case GST_ITERATOR_DONE:
		case GST_ITERATOR_ERROR:
			goto done;
		}
	}
done:
	gst_iterator_free(it);
}


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
 * tags
 */


static GstTagList *get_srcpad_tag_list(GstFrameCPPChannelMux *mux)
{
	GstTagList *tag_list = gst_tag_list_new();
	GHashTableIter it;
	gchar *instrument;

	g_hash_table_iter_init(&it, mux->instruments);
	while(g_hash_table_iter_next(&it, (void **) &instrument, NULL))
		gst_tag_list_add(tag_list, GST_TAG_MERGE_APPEND, GSTLAL_TAG_INSTRUMENT, instrument, NULL);

	GST_DEBUG_OBJECT(mux, "tag list: %" GST_PTR_FORMAT, tag_list);

	return tag_list;
}


/*
 * ============================================================================
 *
 *                                 Sink Pads
 *
 * ============================================================================
 */


/*
 * private data for pre-computed frame object format parameters
 */


typedef struct _framecpp_channelmux_appdata {
	FrameCPP::FrVect::data_types_type type;
	gint nDims;
	FrameCPP::Dimension *dims;
	gint rate;
	guint unit_size;
	gchar *unitY;
} framecpp_channelmux_appdata;


static void framecpp_channelmux_appdata_free(framecpp_channelmux_appdata *appdata)
{
	if(appdata) {
		delete[] appdata->dims;
		g_free(appdata->unitY);
	}
	g_free(appdata);
}


static framecpp_channelmux_appdata *get_appdata(FrameCPPMuxCollectPadsData *data)
{
	return (framecpp_channelmux_appdata *) data->appdata;
}


/*
 * get informed of a sink pad's caps
 */


static gboolean sink_setcaps(GstPad *pad, GstCaps *caps)
{
	GstFrameCPPChannelMux *mux = FRAMECPP_CHANNELMUX(gst_pad_get_parent(pad));
	FrameCPPMuxCollectPadsData *data = framecpp_muxcollectpads_get_data(pad);
	framecpp_channelmux_appdata *appdata = get_appdata(data);
	GstStructure *structure;
	FrameCPP::FrVect::data_types_type type;
	int width, channels, rate;
	const gchar *media_type;
	gboolean success = TRUE;

	/*
	 * parse caps
	 */

	structure = gst_caps_get_structure(caps, 0);
	media_type = gst_structure_get_name(structure);
	success &= gst_structure_get_int(structure, "width", &width);
	success &= gst_structure_get_int(structure, "channels", &channels);
	success &= gst_structure_get_int(structure, "rate", &rate);
	if(!success)
		GST_ERROR_OBJECT(pad, "cannot parse width, channels, and/or rate from %" GST_PTR_FORMAT, caps);
	if(!strcmp(media_type, "audio/x-raw-int")) {
		gboolean is_signed;
		success &= gst_structure_get_boolean(structure, "signed", &is_signed);
		switch(width) {
		case 8:
			type = is_signed ? FrameCPP::FrVect::FR_VECT_C : FrameCPP::FrVect::FR_VECT_1U;
			break;
		case 16:
			type = is_signed ? FrameCPP::FrVect::FR_VECT_2S : FrameCPP::FrVect::FR_VECT_2U;
			break;
		case 32:
			type = is_signed ? FrameCPP::FrVect::FR_VECT_4S : FrameCPP::FrVect::FR_VECT_4U;
			break;
		case 64:
			type = is_signed ? FrameCPP::FrVect::FR_VECT_8S : FrameCPP::FrVect::FR_VECT_8U;
			break;
		default:
			GST_ERROR_OBJECT(pad, "unsupported width %d", width);
			success = FALSE;
			break;
		}
	} else if(!strcmp(media_type, "audio/x-raw-float")) {
		switch(width) {
		case 32:
			type = FrameCPP::FrVect::FR_VECT_4R;
			break;
		case 64:
			type = FrameCPP::FrVect::FR_VECT_8R;
			break;
		default:
			GST_ERROR_OBJECT(pad, "unsupported width %d", width);
			success = FALSE;
			break;
		}
	} else if(!strcmp(media_type, "audio/x-raw-complex")) {
		switch(width) {
		case 64:
			type = FrameCPP::FrVect::FR_VECT_8C;
			break;
		case 128:
			type = FrameCPP::FrVect::FR_VECT_16C;
			break;
		default:
			GST_ERROR_OBJECT(pad, "unsupported width %d", width);
			success = FALSE;
			break;
		}
	} else {
		GST_ERROR_OBJECT(pad, "unsupported media type %s", media_type);
		success = FALSE;
	}

	if(success) {
		GObject *queue;
		GST_OBJECT_LOCK(mux->collect);
		queue = G_OBJECT(data->queue);
		FRAMECPP_MUXQUEUE_LOCK(data->queue);
		/* FIXME:  flush queue on format change */
		appdata->type = type;
		appdata->dims[0].SetDx(1.0 / (double) rate);
		appdata->rate = rate;
		appdata->unit_size = width / 8 * channels;
		g_object_set(queue, "rate", appdata->rate, "unit-size", appdata->unit_size, NULL);
		FRAMECPP_MUXQUEUE_UNLOCK(data->queue);
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
	FrameCPPMuxCollectPadsData *data = framecpp_muxcollectpads_get_data(pad);
	framecpp_channelmux_appdata *appdata = get_appdata(data);
	gboolean success = TRUE;

	switch(GST_EVENT_TYPE(event)) {
	case GST_EVENT_NEWSEGMENT:
		mux->need_discont = TRUE;
		break;

	case GST_EVENT_TAG: {
		GstTagList *tag_list;
		gchar *value = NULL;

		gst_event_parse_tag(event, &tag_list);
		if(gst_tag_list_get_string(tag_list, GSTLAL_TAG_UNITS, &value)) {
			g_strstrip(value);
			g_object_set(pad, "units", value, NULL);
			g_free(appdata->unitY);
			appdata->unitY = value;
		} else
			g_free(value);
		value = NULL;
		if(gst_tag_list_get_string(tag_list, GSTLAL_TAG_INSTRUMENT, &value))
			g_strstrip(value);
			g_object_set(pad, "instrument", value, NULL);
		g_free(value);
		value = NULL;
		if(gst_tag_list_get_string(tag_list, GSTLAL_TAG_CHANNEL_NAME, &value))
			g_strstrip(value);
			g_object_set(pad, "channel-name", value, NULL);
		g_free(value);
		value = NULL;

		gst_event_unref(event);
		goto done;
	}

	case GST_EVENT_EOS:
		/* FIXME:  flush final (short) frame file? */
		break;

	default:
		break;
	}

	GST_DEBUG_OBJECT(mux, "forwarding downstream event %" GST_PTR_FORMAT, event);
	gst_pad_push_event(mux->srcpad, event);

done:
	gst_object_unref(GST_OBJECT(mux));
	return success;
}


/*
 * sink pad notify::instrument handler
 */


static void notify_instrument_handler(GObject *object, GParamSpec *pspec, gpointer user_data)
{
	GstFrameCPPChannelMux *mux = FRAMECPP_CHANNELMUX(gst_pad_get_parent(GST_PAD(object)));

	mux->need_tag_list = TRUE;

	gst_object_unref(GST_OBJECT(mux));
}


/*
 * create a new sink pad and add to element.  does not check if name is
 * already in use
 */


static GstPad *request_new_pad(GstElement *element, GstPadTemplate *templ, const gchar *name)
{
	GstFrameCPPChannelMux *mux = FRAMECPP_CHANNELMUX(element);
	FrameCPPMuxCollectPadsData *data;
	GstFrPad *pad;

	/*
	 * construct the pad
	 */

	pad = gst_frpad_new_from_template(templ, name);
	if(!pad)
		goto no_pad;
	gst_pad_set_setcaps_function(GST_PAD(pad), GST_DEBUG_FUNCPTR(sink_setcaps));
	g_signal_connect(G_OBJECT(pad), "notify::instrument", G_CALLBACK(notify_instrument_handler), NULL);

	/*
	 * add pad to element.  just like FrameCPPMuxCollectPadsData, the
	 * appdata structure is allocated inline.  there is no stand-alone
	 * constructor.
	 */

	GST_OBJECT_LOCK(mux->collect);
	data = framecpp_muxcollectpads_add_pad(mux->collect, GST_PAD(pad), (FrameCPPMuxCollectPadsDataDestroyNotify) GST_DEBUG_FUNCPTR(framecpp_channelmux_appdata_free));
	if(!data)
		goto could_not_add_to_collectpads;
	framecpp_muxcollectpads_set_event_function(data, GST_DEBUG_FUNCPTR(sink_event));
	data->appdata = g_new0(framecpp_channelmux_appdata, 1);
	if(!data->appdata)
		goto could_not_create_appdata;
	get_appdata(data)->nDims = 1;
	get_appdata(data)->dims = new FrameCPP::Dimension[get_appdata(data)->nDims];
	get_appdata(data)->unitY = strdup("");
	if(!gst_element_add_pad(element, GST_PAD(pad)))
		goto could_not_add_to_element;
	GST_OBJECT_UNLOCK(mux->collect);

	/*
	 * done
	 */

	return GST_PAD(pad);

	/*
	 * errors
	 */

could_not_add_to_element:
could_not_create_appdata:
	framecpp_muxcollectpads_remove_pad(mux->collect, GST_PAD(pad));
could_not_add_to_collectpads:
	gst_object_unref(pad);
	GST_OBJECT_UNLOCK(mux->collect);
no_pad:
	return NULL;
}


/*
 * remove a sink pad from element
 */


static void release_pad(GstElement *element, GstPad *pad)
{
	GstFrameCPPChannelMux *mux = FRAMECPP_CHANNELMUX(element);

	GST_OBJECT_LOCK(mux->collect);
	framecpp_muxcollectpads_remove_pad(mux->collect, pad);
	gst_element_remove_pad(element, pad);
	GST_OBJECT_UNLOCK(mux->collect);

	mux->need_tag_list = TRUE;
}


/*
 * collected signal handler
 */


static void collected_handler(FrameCPPMuxCollectPads *collectpads, GstClockTime collected_t_start, GstClockTime collected_t_end, GstFrameCPPChannelMux *mux)
{
	GstClockTime gwf_t_start, gwf_t_end;

	/*
	 * make sure we don't have our wires crossed
	 */

	g_assert(GST_IS_FRAMECPP_CHANNELMUX(mux));
	g_assert(mux->collect == collectpads);
	g_assert(GST_CLOCK_TIME_IS_VALID(collected_t_start));
	g_assert(GST_CLOCK_TIME_IS_VALID(collected_t_end));

	/*
	 * do tag list if needed
	 */

	if(mux->need_tag_list) {
		update_instruments(mux);
		if(g_hash_table_size(mux->instruments))
			gst_element_found_tags(GST_ELEMENT(mux), get_srcpad_tag_list(mux));
		else
			GST_DEBUG_OBJECT(mux, "not pushing tags:  no instruments");
		mux->need_tag_list = FALSE;
	}

	/*
	 * loop over available data
	 */

	for(gwf_t_start = collected_t_start, gwf_t_end = collected_t_start - collected_t_start % FRAME_FILE_DURATION(mux) + FRAME_FILE_DURATION(mux); gwf_t_end <= collected_t_end; gwf_t_start = gwf_t_end, gwf_t_end += FRAME_FILE_DURATION(mux)) {
		GstFlowReturn result;
		GstBuffer *outbuf;

		/*
		 * build frame file
		 *
		 * FIXME:  do we need to add FrDetector structures to each
		 * frame file based on the instrument list?
		 */

		try {
			FrameCPP::Common::MemoryBuffer *obuf(new FrameCPP::Common::MemoryBuffer(std::ios::out));
			FrameCPP::OFrameStream ofs(obuf);
			GstClockTime frame_t_start, frame_t_end;
			GST_DEBUG_OBJECT(mux, "building frame file [%" GST_TIME_SECONDS_FORMAT ", %" GST_TIME_SECONDS_FORMAT ")", GST_TIME_SECONDS_ARGS(gwf_t_start), GST_TIME_SECONDS_ARGS(gwf_t_end));

			/*
			 * loop over frames
			 */

			for(frame_t_start = gwf_t_start, frame_t_end = gwf_t_start - gwf_t_start % mux->frame_duration + mux->frame_duration; frame_t_end <= gwf_t_end; frame_t_start = frame_t_end, frame_t_end += mux->frame_duration) {
				GSList *collectdatalist;
				General::GPSTime gpstime(frame_t_start / GST_SECOND, frame_t_start % GST_SECOND);
				General::SharedPtr<FrameCPP::FrameH> frame(new FrameCPP::FrameH(mux->frame_name, mux->frame_run, mux->frame_number, gpstime, gpstime.GetLeapSeconds(), (double) mux->frame_duration / GST_SECOND));
				GST_DEBUG_OBJECT(mux, "building frame %d [%" GST_TIME_SECONDS_FORMAT ", %" GST_TIME_SECONDS_FORMAT ")", mux->frame_number, GST_TIME_SECONDS_ARGS(frame_t_start), GST_TIME_SECONDS_ARGS(frame_t_end));

				/*
				 * loop over pads
				 */

				for(collectdatalist = collectpads->pad_list; collectdatalist; collectdatalist = g_slist_next(collectdatalist)) {
					FrameCPPMuxCollectPadsData *data = (FrameCPPMuxCollectPadsData *) collectdatalist->data;
					framecpp_channelmux_appdata *appdata = get_appdata(data);
					GstFrPad *frpad = GST_FRPAD(data->pad);
					/* we own this list and its
					 * contents */
					GList *buffer_list = framecpp_muxcollectpads_take_list(data, frame_t_end);
					if(buffer_list) {
						GstClockTime buffer_list_t_start;
						GstClockTime buffer_list_t_end;
						framecpp_muxcollectpads_buffer_list_boundaries(buffer_list, &buffer_list_t_start, &buffer_list_t_end);
						guint buffer_list_length = gst_util_uint64_scale_int_round(buffer_list_t_end - buffer_list_t_start, appdata->rate, GST_SECOND);
						char *dest = (char *) g_malloc0(buffer_list_length * appdata->unit_size);

						g_assert_cmpuint(frame_t_start, <=, buffer_list_t_start);
						g_assert_cmpuint(buffer_list_t_end, <=, frame_t_end);

						/*
						 * copy buffer list
						 * contents into contiguous
						 * array
						 */

						for(; buffer_list; buffer_list = g_list_delete_link(buffer_list, buffer_list)) {
							GstBuffer *buffer = GST_BUFFER(buffer_list->data);
							/* FIXME:  how to indicate gaps in frame file? */
							if(!GST_BUFFER_FLAG_IS_SET(buffer, GST_BUFFER_FLAG_GAP))
								memcpy(dest + gst_util_uint64_scale_int_round(GST_BUFFER_TIMESTAMP(buffer) - buffer_list_t_start, appdata->rate, GST_SECOND) * appdata->unit_size, GST_BUFFER_DATA(buffer), GST_BUFFER_SIZE(buffer));
							gst_buffer_unref(buffer);
						}

						/*
						 * build FrVect from data,
						 * then Fr{Adc,Proc,Sim}Data
						 * from FrVect and append
						 * to frame
						 */

						appdata->dims[0].SetNx(buffer_list_length);
						FrameCPP::FrVect vect(GST_PAD_NAME(data->pad), appdata->type, appdata->nDims, appdata->dims, FrameCPP::BYTE_ORDER_HOST, dest, appdata->unitY);
						switch(frpad->pad_type) {
						case GST_FRPAD_TYPE_FRADCDATA: {
							FrameCPP::FrAdcData adc_data(GST_PAD_NAME(data->pad), frpad->channel_group, frpad->channel_number, frpad->nbits, appdata->rate);
							adc_data.AppendComment(frpad->comment);
							GST_DEBUG_OBJECT(mux, "FrAdcData \"%s\" [%" GST_TIME_SECONDS_FORMAT ", %" GST_TIME_SECONDS_FORMAT ")", GST_PAD_NAME(data->pad), GST_TIME_SECONDS_ARGS(buffer_list_t_start), GST_TIME_SECONDS_ARGS(buffer_list_t_end));
							adc_data.RefData().append(vect);
							if(!frame->GetRawData()) {
								FrameCPP::FrameH::rawData_type rawData(new FrameCPP::FrameH::rawData_type::element_type);
								frame->SetRawData(rawData);
							}
							frame->GetRawData()->RefFirstAdc().append(adc_data);
							break;
						}

						case GST_FRPAD_TYPE_FRPROCDATA: {
							FrameCPP::FrProcData proc_data(GST_PAD_NAME(data->pad), frpad->comment, 1, 0, (double) (buffer_list_t_start - frame_t_start) / GST_SECOND, (double) (buffer_list_t_end - buffer_list_t_start) / GST_SECOND, 0.0, 0.0, 0.0, 0.0);
							GST_DEBUG_OBJECT(mux, "FrProcData \"%s\" [%" GST_TIME_SECONDS_FORMAT ", %" GST_TIME_SECONDS_FORMAT ")", GST_PAD_NAME(data->pad), GST_TIME_SECONDS_ARGS(buffer_list_t_start), GST_TIME_SECONDS_ARGS(buffer_list_t_end));
							proc_data.RefData().append(vect);
							frame->RefProcData().append(proc_data);
							break;
						}

						case GST_FRPAD_TYPE_FRSIMDATA: {
							FrameCPP::FrSimData sim_data(GST_PAD_NAME(data->pad), frpad->comment, appdata->rate, 0.0, 0.0, (double) (buffer_list_t_start - frame_t_start) / GST_SECOND);
							GST_DEBUG_OBJECT(mux, "FrSimData \"%s\" [%" GST_TIME_SECONDS_FORMAT ", %" GST_TIME_SECONDS_FORMAT ")", GST_PAD_NAME(data->pad), GST_TIME_SECONDS_ARGS(buffer_list_t_start), GST_TIME_SECONDS_ARGS(buffer_list_t_end));
							sim_data.RefData().append(vect);
							frame->RefSimData().append(sim_data);
							break;
						}

						default:
							g_assert_not_reached();
							break;
						}
						g_free(dest);
					}
				}

				/*
				 * add frame to file
				 */

				ofs.WriteFrame(frame, FrameCPP::Common::CheckSum::CRC);
				mux->frame_number++;
				g_object_notify(G_OBJECT(mux), "frame-number");
			}
			g_assert_cmpuint(frame_t_start, ==, gwf_t_end);	/* safety check */

			/*
			 * close frame file, extract bytes into GstBuffer
			 */

			ofs.Close();

			/* FIXME:  can this be done without a memcpy()? */
			outbuf = gst_buffer_new_and_alloc(obuf->str().length());
			memcpy(GST_BUFFER_DATA(outbuf), &(obuf->str()[0]), GST_BUFFER_SIZE(outbuf));

			gst_buffer_set_caps(outbuf, GST_PAD_CAPS(mux->srcpad));
			if(mux->need_discont) {
				GST_BUFFER_FLAG_SET(outbuf, GST_BUFFER_FLAG_DISCONT);
				mux->need_discont = FALSE;
			}
			GST_BUFFER_TIMESTAMP(outbuf) = gwf_t_start;
			GST_BUFFER_DURATION(outbuf) = gwf_t_end - gwf_t_start;
			GST_BUFFER_OFFSET(outbuf) = mux->next_out_offset;
			GST_BUFFER_OFFSET_END(outbuf) = GST_BUFFER_OFFSET(outbuf) + GST_BUFFER_SIZE(outbuf);
			mux->next_out_offset = GST_BUFFER_OFFSET_END(outbuf);
		} catch(const std::exception& Exception) {
			GST_ELEMENT_ERROR(mux, STREAM, ENCODE, (NULL), ("libframecpp raised exception: %s", Exception.what()));
			goto done;
		} catch(...) {
			GST_ELEMENT_ERROR(mux, STREAM, ENCODE, (NULL), ("libframecpp raised unknown exception"));
			goto done;
		}

		/*
		 * push downstream
		 */

		GST_DEBUG_OBJECT(mux, "pushing frame file spanning %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(outbuf));
		result = gst_pad_push(mux->srcpad, outbuf);
		if(result != GST_FLOW_OK) {
			GST_ELEMENT_ERROR(mux, CORE, PAD, (NULL), ("gst_pad_push() failed (%s)", gst_flow_get_name(result)));
			goto done;
		}
	}

	/*
	 * done
	 */

done:
	return;
}


/*
 * ============================================================================
 *
 *                            GstElement Overrides
 *
 * ============================================================================
 */


static GstStateChangeReturn change_state(GstElement *element, GstStateChange transition)
{
	GstFrameCPPChannelMux *mux = FRAMECPP_CHANNELMUX(element);
	GstStateChangeReturn result;

	switch(transition) {
	case GST_STATE_CHANGE_READY_TO_PAUSED:
		mux->need_discont = TRUE;
		mux->next_out_offset = 0;
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
	ARG_FRAMES_PER_FILE,
	ARG_FRAME_NAME,
	ARG_FRAME_RUN,
	ARG_FRAME_NUMBER
};


static void set_property(GObject *object, guint id, const GValue *value, GParamSpec *pspec)
{
	GstFrameCPPChannelMux *element = FRAMECPP_CHANNELMUX(object);

	GST_OBJECT_LOCK(element);

	switch((enum property) id) {
	case ARG_FRAME_DURATION:
		element->frame_duration = g_value_get_uint(value) * GST_SECOND;
		g_object_set(G_OBJECT(element->collect), "max-size-time", (guint64) FRAME_FILE_DURATION(element), NULL);
		break;

	case ARG_FRAMES_PER_FILE:
		element->frames_per_file = g_value_get_uint(value);
		g_object_set(G_OBJECT(element->collect), "max-size-time", (guint64) FRAME_FILE_DURATION(element), NULL);
		break;

	case ARG_FRAME_NAME:
		g_free(element->frame_name);
		element->frame_name = g_value_dup_string(value);
		break;

	case ARG_FRAME_RUN:
		element->frame_run = g_value_get_int(value);
		break;

	case ARG_FRAME_NUMBER:
		element->frame_number = g_value_get_uint(value);
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

	switch((enum property) id) {
	case ARG_FRAME_DURATION:
		g_value_set_uint(value, element->frame_duration / GST_SECOND);
		break;

	case ARG_FRAMES_PER_FILE:
		g_value_set_uint(value, element->frames_per_file);
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
	g_hash_table_unref(element->instruments);
	element->instruments = NULL;

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


/*
 * Pad templates.
 */


static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE(
	"%s",
	GST_PAD_SINK,
	GST_PAD_REQUEST,
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
		g_param_spec_uint(
			"frame-duration",
			"Frame duration",
			"Duration of each frame in seconds.",
			1, G_MAXUINT, DEFAULT_FRAME_DURATION,
			(GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_FRAMES_PER_FILE,
		g_param_spec_uint(
			"frames-per-file",
			"Frames per file",
			"Number of frames in each frame file.",
			1, G_MAXUINT, DEFAULT_FRAMES_PER_FILE,
			(GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_FRAME_NAME,
		g_param_spec_string(
			"frame-name",
			"Frame name",
			"Name appearing in each frame header.",
			DEFAULT_FRAME_NAME,
			(GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)
		)
	);
	G_PARAM_SPEC_STRING(g_object_class_find_property(gobject_class, "frame-name"))->ensure_non_null = TRUE;
	g_object_class_install_property(
		gobject_class,
		ARG_FRAME_RUN,
		g_param_spec_int(
			"frame-run",
			"Run number",
			"Run number appearing in each frame header.",
			G_MININT, G_MAXINT, DEFAULT_FRAME_RUN,
			(GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_FRAME_NUMBER,
		g_param_spec_uint(
			"frame-number",
			"Frame number",
			"Current frame number.  Automatically incremented for each new frame.",
			0, G_MAXUINT, DEFAULT_FRAME_NUMBER,
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

	/* element's code assumes gstreamer is using integer nanoseconds.
	 * it's readily fixed if this assumption ever fails to hold, but
	 * for now I'm leaving it */
	g_assert_cmpuint(GST_SECOND, ==, 1000000000);

	/* configure src pad */
	mux->srcpad = gst_element_get_static_pad(GST_ELEMENT(mux), "src");
	/*gst_pad_set_query_function(pad, GST_DEBUG_FUNCPTR(src_query));*/ /* FIXME:  implement */
	gst_pad_set_event_function(mux->srcpad, GST_DEBUG_FUNCPTR(src_event));

	/* configure collect pads.  max-size-time will get set when our
	 * properties are initialized */
	mux->collect = FRAMECPP_MUXCOLLECTPADS(g_object_new(FRAMECPP_MUXCOLLECTPADS_TYPE, NULL));
	g_signal_connect(G_OBJECT(mux->collect), "collected", G_CALLBACK(collected_handler), mux);

	/* initialize other internal data */
	mux->instruments = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, NULL);
	mux->need_tag_list = FALSE;
}

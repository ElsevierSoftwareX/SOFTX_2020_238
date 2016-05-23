/*
 * framecpp channel multiplexor
 *
 * Copyright (C) 2012--2014  Kipp Cannon, Ed Maros
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
#include <framecpp/Detectors.hh>
#include <framecpp/Dimension.hh>
#include <framecpp/FrameH.hh>
#include <framecpp/FrAdcData.hh>
#include <framecpp/FrHistory.hh>
#include <framecpp/FrProcData.hh>
#include <framecpp/FrRawData.hh>
#include <framecpp/FrSimData.hh>
#include <framecpp/FrVect.hh>
#include <framecpp/GPSTime.hh>
#include <framecpp/OFrameStream.hh>


/*
 * our own stuff
 */


#include <gstlal/gstlal_debug.h>
#include <gstlal/gstlal_frhistory.h>
#include <gstlal/gstlal_tags.h>
#include <framecpp_channelmux.h>
#include <gstfrpad.h>
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


G_DEFINE_TYPE_WITH_CODE(
	GstFrameCPPChannelMux,
	framecpp_channelmux,
	GST_TYPE_ELEMENT,
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "framecpp_channelmux", 0, "framecpp_channelmux element")
);


/*
 * Compression scheme enum type
 */


GType framecpp_channelmux_compression_scheme_get_type(void)
{
	static GType type = 0;
	if(!type) {
		static GEnumValue values[] = {
			{FrameCPP::FrVect::RAW, "RAW", "No compression."},
			{FrameCPP::FrVect::GZIP, "GZIP", "Use gzip compression."},
			{FrameCPP::FrVect::DIFF_GZIP, "DIFF_GZIP", "Use gzip compression, differential values."},
			{FrameCPP::FrVect::ZERO_SUPPRESS_WORD_2, "ZERO_SUPPRESS_WORD_2", "Use zero suppression."},
			{FrameCPP::FrVect::ZERO_SUPPRESS_OTHERWISE_GZIP, "ZERO_SUPPRESS_OTHERWISE_GZIP", "Use zero suppression for integer values, gzip on floats."},
			{FrameCPP::FrVect::ZERO_SUPPRESS_WORD_4, "ZERO_SUPPRESS_WORD_4", "Use zero suppression for 4 byte words, gzip on on all others."},
			{FrameCPP::FrVect::BEST_COMPRESSION, "BEST_COMPRESSION", "Use best available compression."},
			{FrameCPP::FrVect::ZERO_SUPPRESS_WORD_8, "ZERO_SUPPRESS_WORD_8", "Use zero supression for 8 byte words."},
			{0, NULL, NULL}
		};

		type = g_enum_register_static("framecpp_channelmux_compression_scheme", values);
	}

	return type;
}

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
#define DEFAULT_COMPRESSION_SCHEME FrameCPP::FrVect::RAW
#define DEFAULT_COMPRESSION_LEVEL 0


#define FRAME_FILE_DURATION(mux) ((mux)->frames_per_file * (mux)->frame_duration)


/*
 * ============================================================================
 *
 *                                 Utilities
 *
 * ============================================================================
 */


/*
 * iterate over sink pads and update instrument set
 */


static void update_instruments(GstFrameCPPChannelMux *mux)
{
	GstIterator *it = gst_element_iterate_sink_pads(GST_ELEMENT(mux));

	g_hash_table_remove_all(mux->instruments);

	while(TRUE) {
		GValue item = G_VALUE_INIT;
		gchar *instrument = NULL;
		switch(gst_iterator_next(it, &item)) {
		case GST_ITERATOR_OK:
			g_object_get(g_value_get_object(&item), "instrument", &instrument, NULL);
			if(instrument)
				g_hash_table_replace(mux->instruments, instrument, instrument);
			g_value_reset(&item);
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
 * private collectpads data for pre-computed frame object format parameters
 */


typedef struct _framecpp_channelmux_appdata {
	FrameCPP::FrVect::data_types_type type;
	gint nDims;
	FrameCPP::Dimension *dims;
	gint rate;
	guint unit_size;
} framecpp_channelmux_appdata;


static void framecpp_channelmux_appdata_free(gpointer appdata)
{
	if(appdata)
		delete[] ((framecpp_channelmux_appdata *) appdata)->dims;
	g_free(appdata);
}


static framecpp_channelmux_appdata *get_appdata(FrameCPPMuxCollectPadsData *data)
{
	return (framecpp_channelmux_appdata *) data->appdata;
}


/*
 * tags
 */


static GstTagList *get_srcpad_tag_list(GstFrameCPPChannelMux *mux)
{
	GstTagList *tag_list = gst_tag_list_new_empty();
	GHashTableIter it;
	gchar *instrument;

	g_hash_table_iter_init(&it, mux->instruments);
	while(g_hash_table_iter_next(&it, (void **) &instrument, NULL))
		gst_tag_list_add(tag_list, GST_TAG_MERGE_APPEND, GSTLAL_TAG_INSTRUMENT, instrument, NULL);

	GST_DEBUG_OBJECT(mux, "tag list: %" GST_PTR_FORMAT, tag_list);

	return tag_list;
}


/*
 * build frame file from queue contents and push downstream
 */


static GstFlowReturn build_and_push_frame_file(GstFrameCPPChannelMux *mux, GstClockTime gwf_t_start, GstClockTime gwf_t_end)
{
	GstBuffer *outbuf = NULL;
	GstMapInfo mapinfo;	/* used for input and output buffers */
	GstFlowReturn result = GST_FLOW_OK;

	g_assert_cmpuint(gwf_t_start, <=, gwf_t_end);

	GST_LOG_OBJECT(mux, "building frame file [%" GST_TIME_SECONDS_FORMAT ", %" GST_TIME_SECONDS_FORMAT ")", GST_TIME_SECONDS_ARGS(gwf_t_start), GST_TIME_SECONDS_ARGS(gwf_t_end));

	try {
		FrameCPP::Common::MemoryBuffer *obuf(new FrameCPP::Common::MemoryBuffer(std::ios::out));
		FrameCPP::OFrameStream ofs(obuf);
		GstClockTime frame_t_start, frame_t_end;

		/*
		 * loop over frames
		 */

		for(frame_t_start = gwf_t_start, frame_t_end = MIN(gwf_t_start - gwf_t_start % mux->frame_duration + mux->frame_duration, gwf_t_end); frame_t_start < gwf_t_end; frame_t_start = frame_t_end, frame_t_end = MIN(frame_t_end + mux->frame_duration, gwf_t_end)) {
			GHashTableIter it;
			gchar *instrument;
			guint i;
			GSList *collectdatalist;
			FrameCPP::GPSTime gpstime(frame_t_start / GST_SECOND, frame_t_start % GST_SECOND);
			LDASTools::AL::SharedPtr<FrameCPP::FrameH> frame(new FrameCPP::FrameH(mux->frame_name, mux->frame_run, mux->frame_number, gpstime, gpstime.GetLeapSeconds(), (double) (frame_t_end - frame_t_start) / GST_SECOND));

			GST_LOG_OBJECT(mux, "building frame %d [%" GST_TIME_SECONDS_FORMAT ", %" GST_TIME_SECONDS_FORMAT ")", mux->frame_number, GST_TIME_SECONDS_ARGS(frame_t_start), GST_TIME_SECONDS_ARGS(frame_t_end));

			/*
			 * add FrDetector objects
			 */

			g_hash_table_iter_init(&it, mux->instruments);
			while(g_hash_table_iter_next(&it, (void **) &instrument, NULL)) {
				if(!strcmp(instrument, "H1"))
					frame->RefDetectProc().append(FrameCPP::GetDetector(FrameCPP::DETECTOR_LOCATION_H1, gpstime));
				if(!strcmp(instrument, "H2"))
					frame->RefDetectProc().append(FrameCPP::GetDetector(FrameCPP::DETECTOR_LOCATION_H2, gpstime));
				if(!strcmp(instrument, "L1"))
					frame->RefDetectProc().append(FrameCPP::GetDetector(FrameCPP::DETECTOR_LOCATION_L1, gpstime));
				if(!strcmp(instrument, "V1"))
					frame->RefDetectProc().append(FrameCPP::GetDetector(FrameCPP::DETECTOR_LOCATION_V1, gpstime));
				else
					GST_WARNING_OBJECT(mux, "not adding FrDetector for unknown instrument '%s'", instrument);
			}

			/*
			 * add frame-level FrHistory objects
			 */

			for(i = 0; i < mux->frame_history->n_values; i++) {
				GstLALFrHistory *history = GSTLAL_FRHISTORY(g_value_get_boxed(g_value_array_get_nth(mux->frame_history, i)));
				gchar *str = gstlal_frhistory_to_string(history);
				GST_LOG_OBJECT(mux, "FrHistory: %s", str);
				g_free(str);
				frame->RefHistory().append(FrameCPP::FrHistory(gstlal_frhistory_get_name(history), gstlal_frhistory_get_timestamp(history) / GST_SECOND, gstlal_frhistory_get_comment(history)));
			}

			/*
			 * loop over pads
			 */

			for(collectdatalist = mux->collect->pad_list; collectdatalist; collectdatalist = g_slist_next(collectdatalist)) {
				FrameCPP::Common::Container<FrameCPP::FrVect> *container;
				FrameCPPMuxCollectPadsData *data = (FrameCPPMuxCollectPadsData *) collectdatalist->data;
				framecpp_channelmux_appdata *appdata = get_appdata(data);
				GstFrPad *frpad = GST_FRPAD(data->pad);
				gdouble timeOffset = 0.0;
				/* we own this list and its contents */
				GList *buffer_list = framecpp_muxcollectpads_take_list(data, frame_t_end);

				/*
				 * merge contiguous buffers, ignoring gap
				 * state
				 *
				 * FIXME:  the difference between the first
				 * buffer's start time and the start of the
				 * frame could be absorbed into timeOffset
				 * instead of going into startX of the
				 * FrVect
				 */

				buffer_list = framecpp_muxcollectpads_buffer_list_join(buffer_list, FALSE);
				/* FIXME:  the next two tests should be
				 * removed.  the muxer no longer requires
				 * the buffer list to contain exactly 1
				 * buffer.  these checks are here
				 * temporarily to reproduce old behaviour
				 * that the code has been generalized to no
				 * longer require */
				if(!buffer_list)
					continue;
				g_assert_cmpuint(g_list_length(buffer_list), ==, 1);

				/*
				 * build Fr{Adc,Proc,Sim}Data, append to
				 * frame
				 */

				g_assert_cmpuint(frame_t_start + (GstClockTime) round(timeOffset * GST_SECOND), <=, frame_t_end);

				switch(frpad->pad_type) {
				case GST_FRPAD_TYPE_FRADCDATA: {
					FrameCPP::FrAdcData adc_data(GST_PAD_NAME(frpad), frpad->channel_group, frpad->channel_number, frpad->nbits, appdata->rate, frpad->bias, frpad->slope, frpad->units, 0.0, timeOffset, frpad->datavalid, frpad->phase);
					adc_data.AppendComment(frpad->comment);
					GST_LOG_OBJECT(frpad, "appending FrAdcData starting at %" GST_TIME_SECONDS_FORMAT, GST_TIME_SECONDS_ARGS(frame_t_start + (GstClockTime) round(timeOffset * GST_SECOND)));
					if(!frame->GetRawData()) {
						FrameCPP::FrameH::rawData_type rawData(new FrameCPP::FrameH::rawData_type::element_type);
						frame->SetRawData(rawData);
					}
					frame->GetRawData()->RefFirstAdc().append(adc_data);
					container = &(frame->GetRawData()->RefFirstAdc().back()->RefData());
					break;
				}

				case GST_FRPAD_TYPE_FRPROCDATA: {
					/* FIXME:  history */
					FrameCPP::FrProcData proc_data(GST_PAD_NAME(frpad), frpad->comment, 1, 0, timeOffset, (double) GST_CLOCK_DIFF(frame_t_start, frame_t_end) / GST_SECOND - timeOffset, 0.0, 0.0, appdata->rate / 2.0, 0.0);
					GST_LOG_OBJECT(frpad, "appending FrProcData spanning [%" GST_TIME_SECONDS_FORMAT ", %" GST_TIME_SECONDS_FORMAT ")", GST_TIME_SECONDS_ARGS(frame_t_start + (GstClockTime) round(timeOffset * GST_SECOND)), GST_TIME_SECONDS_ARGS(frame_t_end));
					frame->RefProcData().append(proc_data);
					container = &(frame->RefProcData().back()->RefData());
					break;
				}

				case GST_FRPAD_TYPE_FRSIMDATA: {
					FrameCPP::FrSimData sim_data(GST_PAD_NAME(frpad), frpad->comment, appdata->rate, 0.0, 0.0, timeOffset);
					GST_LOG_OBJECT(frpad, "appending FrSimData starting at %" GST_TIME_SECONDS_FORMAT, GST_TIME_SECONDS_ARGS(frame_t_start + (GstClockTime) round(timeOffset * GST_SECOND)));
					frame->RefSimData().append(sim_data);
					container = &(frame->RefSimData().back()->RefData());
					break;
				}

				default:
					g_assert_not_reached();
					break;
				}

				/*
				 * build FrVects from buffers, append to
				 * channel
				 */

				for(; buffer_list; buffer_list = g_list_delete_link(buffer_list, buffer_list)) {
					GstBuffer *buffer = GST_BUFFER(buffer_list->data);
					GstClockTime buffer_t_start = GST_BUFFER_TIMESTAMP(buffer);
					GstClockTime buffer_t_end = buffer_t_start + GST_BUFFER_DURATION(buffer);

					/*
					 * safety checks.  1 ns of rounding
					 * noise in the buffer's start and
					 * end times is permitted because
					 * the process of slicing the input
					 * stream into chunks for frames
					 * can lead to off-by-one rounding
					 * errors since it is performed
					 * without global state
					 * information.
					 */

					g_assert(GST_BUFFER_TIMESTAMP_IS_VALID(buffer));
					g_assert(GST_BUFFER_DURATION_IS_VALID(buffer));
					g_assert_cmpuint(GST_BUFFER_OFFSET_END(buffer) - GST_BUFFER_OFFSET(buffer), ==, gst_util_uint64_scale_int_round(GST_BUFFER_DURATION(buffer), appdata->rate, GST_SECOND));
					if(llabs(frame_t_start - buffer_t_start) <= 1)
						buffer_t_start = frame_t_start;
					if(llabs(frame_t_end - buffer_t_end) <= 1)
						buffer_t_end = frame_t_end;
					g_assert_cmpuint(frame_t_start, <=, buffer_t_start);
					g_assert_cmpuint(buffer_t_start, <=, buffer_t_end);
					g_assert_cmpuint(buffer_t_end, <=, frame_t_end);

					/*
					 * build FrVect, append to channel
					 */

					GST_LOG_OBJECT(frpad, "appending FrVect [%" GST_TIME_SECONDS_FORMAT ", %" GST_TIME_SECONDS_FORMAT ")", GST_TIME_SECONDS_ARGS(buffer_t_start), GST_TIME_SECONDS_ARGS(buffer_t_end));
					appdata->dims[0].SetNx(GST_BUFFER_OFFSET_END(buffer) - GST_BUFFER_OFFSET(buffer));
					appdata->dims[0].SetStartX((double) GST_CLOCK_DIFF(frame_t_start, buffer_t_start) / GST_SECOND - timeOffset);
					gst_buffer_map(buffer, &mapinfo, GST_MAP_READ);
					container->append(FrameCPP::FrVect(GST_PAD_NAME(frpad), appdata->type, appdata->nDims, appdata->dims, FrameCPP::BYTE_ORDER_HOST, mapinfo.data, frpad->units));
					gst_buffer_unmap(buffer, &mapinfo);
					gst_buffer_unref(buffer);
				}
			}

			/*
			 * add frame to file
			 */

			ofs.WriteFrame(frame, (gushort) mux->compression_scheme, (gushort) mux->compression_level, FrameCPP::Common::CheckSum::CRC);
			mux->frame_number++;
			g_object_notify(G_OBJECT(mux), "frame-number");
		}
		g_assert_cmpuint(frame_t_start, ==, gwf_t_end);	/* safety check */

		/*
		 * close frame file, extract bytes into GstBuffer
		 */

		ofs.Close();

		/* FIXME:  can this be done without a memcpy()? */
		outbuf = gst_buffer_new_allocate(NULL, obuf->str().length(), NULL);
		if(!outbuf) {
			GST_ELEMENT_ERROR(mux, CORE, PAD, (NULL), ("gst_pad_alloc_buffer() failed (%s)", gst_flow_get_name(result)));
			result = GST_FLOW_ERROR;
			goto done;
		}
		if(!gst_buffer_map(outbuf, &mapinfo, GST_MAP_WRITE)) {
			GST_ELEMENT_ERROR(mux, RESOURCE, WRITE, (NULL), ("buffer cannot be mapped for write"));
			gst_buffer_unref(outbuf);
			result = GST_FLOW_ERROR;
			goto done;
		}
		memcpy(mapinfo.data, &(obuf->str()[0]), mapinfo.size);
		if(mux->need_discont) {
			GST_BUFFER_FLAG_SET(outbuf, GST_BUFFER_FLAG_DISCONT);
			mux->need_discont = FALSE;
		}
		GST_BUFFER_TIMESTAMP(outbuf) = gwf_t_start;
		GST_BUFFER_DURATION(outbuf) = gwf_t_end - gwf_t_start;
		GST_BUFFER_OFFSET(outbuf) = mux->next_out_offset;
		GST_BUFFER_OFFSET_END(outbuf) = GST_BUFFER_OFFSET(outbuf) + mapinfo.size;
		mux->next_out_offset = GST_BUFFER_OFFSET_END(outbuf);
		gst_buffer_unmap(outbuf, &mapinfo);
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

	GST_LOG_OBJECT(mux, "pushing frame file spanning %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(outbuf));
	result = gst_pad_push(mux->srcpad, outbuf);
	outbuf = NULL;	/* gst_pad_push() took ownership */
	if(result != GST_FLOW_OK) {
		GST_ELEMENT_ERROR(mux, CORE, PAD, (NULL), ("gst_pad_push() failed (%s)", gst_flow_get_name(result)));
		goto done;
	}

	/*
	 * done
	 */

done:
	if(outbuf)
		gst_buffer_unref(outbuf);
	return result;
}


/*
 * flush remaining queued data
 */


static GstFlowReturn flush(GstFrameCPPChannelMux *mux)
{
	GstClockTime collected_t_start, collected_t_end;
	GstClockTime gwf_t_start, gwf_t_end;
	GstFlowReturn result = GST_FLOW_OK;

	GST_LOG_OBJECT(mux, "flushing enqueued data");

	/*
	 * get span
	 */

	if(!framecpp_muxcollectpads_get_span(mux->collect, &collected_t_start, &collected_t_end)) {
		GST_ERROR_OBJECT(mux, "unable to determine span of queues");
		result = GST_FLOW_ERROR;
		goto done;
	}

	/*
	 * do tag list if needed
	 */

	if(mux->need_tag_list) {
		update_instruments(mux);
		if(g_hash_table_size(mux->instruments))
			gst_pad_push_event(mux->srcpad, gst_event_new_tag(get_srcpad_tag_list(mux)));
		else
			GST_DEBUG_OBJECT(mux, "not pushing tags:  no instruments");
		mux->need_tag_list = FALSE;
	}

	/*
	 * loop over available data
	 */

	for(gwf_t_start = collected_t_start, gwf_t_end = MIN(collected_t_start - collected_t_start % FRAME_FILE_DURATION(mux) + FRAME_FILE_DURATION(mux), collected_t_end); gwf_t_start < collected_t_end; gwf_t_start = gwf_t_end, gwf_t_end = MIN(gwf_t_end + FRAME_FILE_DURATION(mux), collected_t_end)) {
		result = build_and_push_frame_file(mux, gwf_t_start, gwf_t_end);
		if(result != GST_FLOW_OK)
			goto done;
	}
	g_assert_cmpuint(gwf_t_start, ==, collected_t_end);	/* safety check */

	/*
	 * done
	 */

done:
	return result;
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


static gboolean forward_src_event_func(const GValue *item, GValue *ret, gpointer user_data)
{
	GstPad *pad = GST_PAD_CAST(g_value_get_object(item));
	EventData *data = (EventData *) user_data;

	gst_event_ref(data->event);
	if(!gst_pad_push_event(pad, data->event)) {
		/* quick hack to unflush the pads. ideally we need  a way
		 * to just unflush this single collect pad */
		if(data->flush)
			gst_pad_send_event(pad, gst_event_new_flush_stop(FALSE));
	} else {
		g_value_set_boolean(ret, TRUE);
	}
	gst_object_unref(GST_OBJECT(pad));
	return TRUE;
}


static gboolean forward_src_event(GstFrameCPPChannelMux *mux, GstEvent *event, gboolean flush)
{
	GstIterator *it;
	GValue vret = G_VALUE_INIT;
	EventData data = {
		event,
		flush
	};
	gboolean success;

	g_value_init(&vret, G_TYPE_BOOLEAN);
	g_value_set_boolean(&vret, FALSE);

	it = gst_element_iterate_sink_pads(GST_ELEMENT(mux));
	while(TRUE) {
		switch(gst_iterator_fold(it, forward_src_event_func, &vret, &data)) {
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


static gboolean src_event(GstPad *pad, GstObject *parent, GstEvent *event)
{
	GstFrameCPPChannelMux *mux = FRAMECPP_CHANNELMUX(parent);
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
 * get informed of a sink pad's caps.  this is called with the
 * FrameCPPMuxCollectPads lock held.
 */


static gboolean sink_setcaps(GstPad *pad, GstObject *parent, GstCaps *caps)
{
	FrameCPPMuxCollectPadsData *data = framecpp_muxcollectpads_get_data(pad);
	framecpp_channelmux_appdata *appdata = get_appdata(data);
	GstAudioInfo info;
	FrameCPP::FrVect::data_types_type type;
	gboolean success = TRUE;

	/*
	 * parse caps
	 */

	success &= gst_audio_info_from_caps(&info, caps);
	if(success) {
		const gchar *name = GST_AUDIO_INFO_NAME(&info);
		switch(GST_AUDIO_INFO_FORMAT(&info)) {
		case GST_AUDIO_FORMAT_S8:
			type = FrameCPP::FrVect::FR_VECT_C;
			break;
		case GST_AUDIO_FORMAT_U8:
			type = FrameCPP::FrVect::FR_VECT_1U;
			break;
		case GST_AUDIO_FORMAT_S16:
			type = FrameCPP::FrVect::FR_VECT_2S;
			break;
		case GST_AUDIO_FORMAT_U16:
			type = FrameCPP::FrVect::FR_VECT_2U;
			break;
		case GST_AUDIO_FORMAT_S32:
			type = FrameCPP::FrVect::FR_VECT_4S;
			break;
		case GST_AUDIO_FORMAT_U32:
			type = FrameCPP::FrVect::FR_VECT_4U;
			break;
		/* FIXME no longer natively supported by gstreamer */
		/*case GST_AUDIO_FORMAT_S64:
			type = FrameCPP::FrVect::FR_VECT_8S;
			break;
		case GST_AUDIO_FORMAT_U64:
			type = FrameCPP::FrVect::FR_VECT_8U;
			break;*/
		case GST_AUDIO_FORMAT_F32:
			type = FrameCPP::FrVect::FR_VECT_4R;
			break;
		case GST_AUDIO_FORMAT_F64:
			type = FrameCPP::FrVect::FR_VECT_8R;
			break;
		default:
			/*
			 * Handle the complex types which are "special" formats
			 */
			if(!strncmp(name, "Z64", 3))
				type = FrameCPP::FrVect::FR_VECT_8C;
			else if(!strncmp(name, "Z128", 4))
				type = FrameCPP::FrVect::FR_VECT_16C;
			else {
				GST_ERROR_OBJECT(pad, "unsupported format %" GST_PTR_FORMAT, caps);
				success = FALSE;
			}
			break;
		}
	}

	if(success) {
		GObject *queue;
		queue = G_OBJECT(data->queue);
		FRAMECPP_MUXQUEUE_LOCK(data->queue);
		/* FIXME:  flush queue on format change */
		appdata->type = type;
		appdata->dims[0].SetDx(1.0 / (double) GST_AUDIO_INFO_RATE(&info));
		appdata->rate = GST_AUDIO_INFO_RATE(&info);
		appdata->unit_size = GST_AUDIO_INFO_BPF(&info);
		g_object_set(queue, "rate", appdata->rate, "unit-size", appdata->unit_size, NULL);
		FRAMECPP_MUXQUEUE_UNLOCK(data->queue);
	}

	return success;
}


/*
 * handle sink pad events
 */


static gboolean sink_event(GstPad *pad, GstObject *parent, GstEvent *event)
{
	GstFrameCPPChannelMux *mux = FRAMECPP_CHANNELMUX(parent);
	gboolean success = TRUE;

	switch(GST_EVENT_TYPE(event)) {
	case GST_EVENT_SEGMENT:
		mux->need_discont = TRUE;
		break;

	case GST_EVENT_CAPS:
		GstCaps *caps;
		gst_event_parse_caps(event, &caps);
		sink_setcaps(pad, parent, caps);
		/* do not formward */
		gst_event_unref(event);
		goto done;
		break;

	case GST_EVENT_TAG: {
		GstTagList *tag_list;
		gchar *value_s = NULL;
		gfloat value_f;
		guint value_u;

		/* FIXME:  should the GstFrPad's tags property be writable,
		 * instead? */
		gst_event_parse_tag(event, &tag_list);

		if(gst_tag_list_get_string(tag_list, GSTLAL_TAG_UNITS, &value_s)) {
			g_strstrip(value_s);
			g_object_set(pad, "units", value_s, NULL);
		} else
			g_free(value_s);
		value_s = NULL;

		if(gst_tag_list_get_string(tag_list, GSTLAL_TAG_INSTRUMENT, &value_s)) {
			g_strstrip(value_s);
			g_object_set(pad, "instrument", value_s, NULL);
		}
		g_free(value_s);
		value_s = NULL;

		if(gst_tag_list_get_string(tag_list, GSTLAL_TAG_CHANNEL_NAME, &value_s)) {
			g_strstrip(value_s);
			g_object_set(pad, "channel-name", value_s, NULL);
		}
		g_free(value_s);
		value_s = NULL;

		if(gst_tag_list_get_float(tag_list, GSTLAL_TAG_BIAS, &value_f))
			g_object_set(pad, "bias", value_f, NULL);

		if(gst_tag_list_get_float(tag_list, GSTLAL_TAG_SLOPE, &value_f))
			g_object_set(pad, "slope", value_f, NULL);

		if(gst_tag_list_get_float(tag_list, GSTLAL_TAG_PHASE, &value_f))
			g_object_set(pad, "phase", value_f, NULL);

		if(gst_tag_list_get_uint(tag_list, GSTLAL_TAG_DATAVALID, &value_u))
			g_object_set(pad, "datavalid", value_u, NULL);

		gst_event_unref(event);
		goto done;
	}

	case GST_EVENT_EOS:
		GST_LOG_OBJECT(mux, "EOS");
	case GST_EVENT_FLUSH_START:
		flush(mux);
		break;

	default:
		break;
	}

	GST_LOG_OBJECT(mux, "forwarding downstream event %" GST_PTR_FORMAT, event);
	gst_pad_push_event(mux->srcpad, event);

done:
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

/* FIXME:  should the caps be used somehow? */

static GstPad *request_new_pad(GstElement *element, GstPadTemplate *templ, const gchar *name, const GstCaps *caps)
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
	g_signal_connect(G_OBJECT(pad), "notify::instrument", G_CALLBACK(notify_instrument_handler), NULL);

	/*
	 * add pad to element.  just like FrameCPPMuxCollectPadsData, the
	 * appdata structure is allocated inline.  there is no stand-alone
	 * constructor.
	 */

	GST_OBJECT_LOCK(mux->collect);
	data = framecpp_muxcollectpads_add_pad(mux->collect, GST_PAD(pad), GST_DEBUG_FUNCPTR(framecpp_channelmux_appdata_free));
	if(!data)
		goto could_not_add_to_collectpads;
	framecpp_muxcollectpads_set_event_function(data, GST_DEBUG_FUNCPTR(sink_event));
	data->appdata = g_new0(framecpp_channelmux_appdata, 1);
	if(!data->appdata)
		goto could_not_create_appdata;
	get_appdata(data)->nDims = 1;
	get_appdata(data)->dims = new FrameCPP::Dimension[get_appdata(data)->nDims];
	gst_object_ref(pad);	/* we need to return a reference */
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
			gst_pad_push_event(mux->srcpad, gst_event_new_tag(get_srcpad_tag_list(mux)));
		else
			GST_DEBUG_OBJECT(mux, "not pushing tags:  no instruments");
		mux->need_tag_list = FALSE;
	}

	/*
	 * loop over available data
	 */

	for(gwf_t_start = collected_t_start, gwf_t_end = collected_t_start - collected_t_start % FRAME_FILE_DURATION(mux) + FRAME_FILE_DURATION(mux); gwf_t_end <= collected_t_end; gwf_t_start = gwf_t_end, gwf_t_end += FRAME_FILE_DURATION(mux))
		if(build_and_push_frame_file(mux, gwf_t_start, gwf_t_end) != GST_FLOW_OK)
			break;

	/*
	 * done
	 */
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

	result = GST_ELEMENT_CLASS(framecpp_channelmux_parent_class)->change_state(element, transition);

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
	ARG_FRAME_NUMBER,
	ARG_FRAME_HISTORY,
	ARG_COMPRESSION_SCHEME,
	ARG_COMPRESSION_LEVEL
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

	case ARG_COMPRESSION_SCHEME:
		element->compression_scheme = (enum FrameCPP::FrVect::compression_scheme_type) g_value_get_enum(value);
		break;

	case ARG_COMPRESSION_LEVEL:
		element->compression_level = g_value_get_uint(value);
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
	case ARG_COMPRESSION_SCHEME:
		g_value_set_enum(value, element->compression_scheme);
		break;

	case ARG_COMPRESSION_LEVEL:
		g_value_set_uint(value, element->compression_level);
		break;

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

	case ARG_FRAME_HISTORY:
		g_value_set_boxed(value, element->frame_history);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * Instance finalize function.
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
	g_value_array_free(element->frame_history);
	element->frame_history = NULL;

	G_OBJECT_CLASS(framecpp_channelmux_parent_class)->finalize(object);
}


/*
 * Pad templates.
 */


static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE(
	"%s",
	GST_PAD_SINK,
	GST_PAD_REQUEST,
	GST_STATIC_CAPS(
		"audio/x-raw, " \
		"rate = " GST_AUDIO_RATE_RANGE ", " \
		"channels = (int) 1, " \
		"format = (string) { U8, " GST_AUDIO_NE(U16) ", " GST_AUDIO_NE(U32) ", " GST_AUDIO_NE(U64) ", S8, " GST_AUDIO_NE(S16) ", " GST_AUDIO_NE(S32) ", " GST_AUDIO_NE(S64) ", " GST_AUDIO_NE(F32) ", " GST_AUDIO_NE(F64) ", " GST_AUDIO_NE(Z64) ", " GST_AUDIO_NE(Z128) "}, " \
		"layout = (string) interleaved, " \
		"channel-mask = (bitmask) 0"
	)
);


/*
 * class_init()
 */


static void framecpp_channelmux_class_init(GstFrameCPPChannelMuxClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GstElementClass *gstelement_class = GST_ELEMENT_CLASS(klass);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	gstelement_class->request_new_pad = GST_DEBUG_FUNCPTR(request_new_pad);
	gstelement_class->release_pad = GST_DEBUG_FUNCPTR(release_pad);
	gstelement_class->change_state = GST_DEBUG_FUNCPTR(change_state);

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
	g_object_class_install_property(
		gobject_class,
		ARG_FRAME_HISTORY,
		g_param_spec_value_array(
			"frame-history",
			"Frame-level history list",
			"List of GstFrHistory objects.",
			g_param_spec_boxed(
				"history",
				"History entry",
				"GstFrHistory object.",
				GSTLAL_FRHISTORY_TYPE,
				(GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)
			),
			(GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_COMPRESSION_SCHEME,
		g_param_spec_enum(
			"compression-scheme",
			"Compression scheme",
			"Scheme to use in compression of data.",
			FRAMECPP_CHANNELMUX_COMPRESSION_SCHEME_TYPE,
			DEFAULT_COMPRESSION_SCHEME,
			(GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_COMPRESSION_LEVEL,
		g_param_spec_uint(
			"compression-level",
			"Compression level",
			"Compression level to use where applicable.",
			0, G_MAXUINT, DEFAULT_COMPRESSION_LEVEL,
			(GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)
		)
	);
}


/*
 * instance_init()
 */


static void framecpp_channelmux_init(GstFrameCPPChannelMux *mux)
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
	gst_pad_use_fixed_caps(mux->srcpad);

	/* configure collect pads.  max-size-time will get set when our
	 * properties are initialized */
	mux->collect = FRAMECPP_MUXCOLLECTPADS(g_object_new(FRAMECPP_MUXCOLLECTPADS_TYPE, NULL));
	g_signal_connect(G_OBJECT(mux->collect), "collected", G_CALLBACK(collected_handler), mux);

	/* initialize other internal data */
	mux->instruments = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, NULL);
	mux->need_tag_list = FALSE;
	mux->frame_history = g_value_array_new(0);
}

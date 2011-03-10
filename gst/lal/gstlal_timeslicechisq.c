/*
 * A time-slice-based \chi^{2} element for the inspiral pipeline.
 *
 * Copyright (C) 2011  Kipp Cannon, Chad Hanna, Drew Keppel
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
 * stuff from the C library
 */


#include <complex.h>
#include <math.h>
#include <string.h>


/*
 * stuff from glib/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstcollectpads.h>


/*
 * stuff from GSL
 */


#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>


/*
 * our own stuff
 */

#include <gstlal.h>
#include <gstlalcollectpads.h>
#include <gstlal_timeslicechisq.h>


/*
 * ========================================================================
 *
 *                             Utility Functions
 *
 * ========================================================================
 */


static int num_channels(const GSTLALTimeSliceChiSquare *element)
{
	return element->chifacs->size;
}


/*
 * ============================================================================
 *
 *                           GStreamer Boiler Plate
 *
 * ============================================================================
 */


enum property {
	ARG_0,
	ARG_CHIFACS
};


/*
 * ============================================================================
 *
 *                              Caps --- SNR Pad
 *
 * ============================================================================
 */


/*
 * we can only accept caps that both ourselves and the downstream peer can
 * handle, and the number of channels must match the size of the mixing
 * matrix
 */


static GstCaps *getcaps_snr(GstPad *pad)
{
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(gst_pad_get_parent(pad));
	GstCaps *peercaps;
	GstCaps *caps;

	/*
	 * start by retrieving our own caps.  use get_fixed_caps_func() to
	 * avoid recursing back into this function.
	 */

	GST_OBJECT_LOCK(element);
	caps = gst_pad_get_fixed_caps_func(pad);

	/*
	 * intersect with the downstream peer's caps if known.
	 */

	peercaps = gst_pad_peer_get_caps(element->srcpad);
	if(peercaps) {
		GstCaps *result = gst_caps_intersect(peercaps, caps);
		GST_DEBUG_OBJECT(element, "intersecting %" GST_PTR_FORMAT " and %" GST_PTR_FORMAT, caps, peercaps);
		gst_caps_unref(caps);
		gst_caps_unref(peercaps);
		caps = result;
		GST_DEBUG_OBJECT(element, "intersection %" GST_PTR_FORMAT, caps);
	}
	GST_OBJECT_UNLOCK(element);

	/*
	 * done
	 */

	return caps;
}


/*
 * when setting new caps, extract the sample rate and bytes/sample from the
 * caps
 */


static gboolean setcaps_snr(GstPad *pad, GstCaps *caps)
{
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(gst_pad_get_parent(pad));
	GstStructure *structure;
	gint rate, width, channels;
	gboolean success = TRUE;

	GST_LOG_OBJECT(element, "setting caps on pad %p,%s to %" GST_PTR_FORMAT, pad, GST_PAD_NAME (pad), caps);

	/*
	 * parse the caps
	 */

	/* FIXME, see if the timeslicessnrpad can accept the format. Also lock the
	 * format on the other pads to this new format. */
//	GST_OBJECT_LOCK(element);
//	gst_caps_replace(&GST_PAD_CAPS(element->timeslicesnrpad), caps);
//	GST_OBJECT_UNLOCK(element);

	GST_DEBUG_OBJECT(element, "(%s) trying %" GST_PTR_FORMAT "\n", GST_PAD_NAME(pad), caps);
	structure = gst_caps_get_structure(caps, 0);
	if(!gst_structure_get_int(structure, "rate", &rate))
		success = FALSE;
	if(!gst_structure_get_int(structure, "width", &width))
		success = FALSE;
	if(!gst_structure_get_int(structure, "channels", &channels))
		success = FALSE;

	/*
	 * if we have a chifacs, the number of channels must match the length
	 * of chifacs.
	 */

	g_mutex_lock(element->coefficients_lock);
	if(element->chifacs && (channels != (gint) num_channels(element))) {
		GST_DEBUG_OBJECT(element, "channels != %d in %" GST_PTR_FORMAT, num_channels(element), caps);
		success = FALSE;
	}
	g_mutex_unlock(element->coefficients_lock);

	/*
	 * will the downstream peer will accept the caps?  (the output
	 * stream has the same caps as the SNR input stream)
	 */

	if(success) {
		GST_DEBUG_OBJECT(element, "(%s) trying to set caps %" GST_PTR_FORMAT " on downstream peer\n", GST_PAD_NAME(pad), caps);
		success = gst_pad_set_caps(element->srcpad, caps);
		GST_DEBUG_OBJECT(element, "(%s) %s\n", GST_PAD_NAME(pad), success ? "accepted" : "rejected");
	}

	/*
	 * if that was successful, update our metadata
	 */

	if(success) {
		GST_OBJECT_LOCK(element);
		gstlal_collect_pads_set_unit_size(pad, (width / 8) * channels);
		element->rate = rate;
		GST_OBJECT_UNLOCK(element);
	}

	/*
	 * done
	 */

	gst_object_unref(element);
	return success;
}


/*
 * ============================================================================
 *
 *                        Caps --- Time-Slice SNR Pad
 *
 * ============================================================================
 */


/*
 * we can only accept caps that both ourselves and the downstream peer can
 * handle, and the number of channels must match the size of the chifacs
 */


static GstCaps *getcaps_timeslicesnr(GstPad *pad)
{
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(gst_pad_get_parent(pad));
	GstCaps *peercaps;
	GstCaps *caps;

	/*
	 * start by retrieving our own caps.  use get_fixed_caps_func() to
	 * avoid recursing back into this function.
	 */

	GST_OBJECT_LOCK(element);
	caps = gst_pad_get_fixed_caps_func(pad);

	/*
	 * intersect with the downstream peer's caps if known.
	 */

	peercaps = gst_pad_peer_get_caps(element->srcpad);
	if(peercaps) {
		GstCaps *result = gst_caps_intersect(peercaps, caps);
		GST_DEBUG_OBJECT(element, "intersecting %" GST_PTR_FORMAT " and %" GST_PTR_FORMAT, caps, peercaps);
		gst_caps_unref(caps);
		gst_caps_unref(peercaps);
		caps = result;
		GST_DEBUG_OBJECT(element, "intersection %" GST_PTR_FORMAT, caps);

	}
	GST_OBJECT_UNLOCK(element);

	/*
	 * done
	 */

	gst_object_unref(element);
	return caps;
}


/*
 * when setting new caps, extract the sample rate and bytes/sample from the
 * caps
 */


static gboolean setcaps_timeslicesnr(GstPad *pad, GstCaps *caps)
{
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(gst_pad_get_parent(pad));
	GstStructure *structure;
	gint rate, width, channels;
	gboolean success = TRUE;

	/*
	 * parse the caps
	 */

	/* FIXME, see if the timeslicessnrpad can accept the format. Also lock the
	 * format on the other pads to this new format. */
//	GST_OBJECT_LOCK(element);
//	gst_caps_replace(&GST_PAD_CAPS(element->snrpad), caps);
//	GST_OBJECT_UNLOCK(element);

	GST_DEBUG_OBJECT(element, "(%s) trying %" GST_PTR_FORMAT "\n", GST_PAD_NAME(pad), caps);
	structure = gst_caps_get_structure(caps, 0);
	if(!gst_structure_get_int(structure, "rate", &rate))
		success = FALSE;
	if(!gst_structure_get_int(structure, "width", &width))
		success = FALSE;
	if(!gst_structure_get_int(structure, "channels", &channels))
		success = FALSE;

	/*
	 * if we have a chifacs, the number of channels must match the length
	 * of chifacs.
	 */

	g_mutex_lock(element->coefficients_lock);
	if(element->chifacs && (channels != (gint) num_channels(element))) {
		GST_DEBUG_OBJECT(element, "channels != %d in %" GST_PTR_FORMAT, num_channels(element), caps);
		success = FALSE;
	}
	g_mutex_unlock(element->coefficients_lock);

	/*
	 * if everything OK, update our metadata
	 */

	if(success) {
		GST_OBJECT_LOCK(element);
		gstlal_collect_pads_set_unit_size(pad, (width / 8) * channels);
		GST_OBJECT_UNLOCK(element);
	}

	/*
	 * done
	 */

	gst_object_unref(element);
	return success;
}


/*
 * ============================================================================
 *
 *                                  Queries
 *
 * ============================================================================
 */


/*
 * convert an output offset to an output timestamp.
 */


static GstClockTime output_timestamp_from_offset(const GSTLALTimeSliceChiSquare *element, guint64 offset)
{
	return element->segment.start + gst_util_uint64_scale_int_round(offset, GST_SECOND, element->rate);
}


/*
 * The duration query should reflect how long you will produce data, that
 * is the amount of stream time until you will emit EOS.  This is the max
 * of all the durations of upstream since we emit EOS when all of them
 * finished.
 *
 * FIXME: This is true for asynchronous mixing.  For synchronous mixing, an
 * input stream might be delayed so that although it reports X seconds
 * remaining until EOS, that data might not have started being mixed into
 * the output yet and won't start for another Y seconds, so our output has
 * X + Y seconds reminaing in it.  We know that delay so we should be able
 * to incorporate it in our answer.  However, even if all streams are being
 * mixed, i.e., the output timestamp has advanced into all input segments
 * so that none are being held back, then taking the maximum of the
 * upstream durations is mostly correct but there is still the possibility
 * that discontinuities will occur in one or more input streams which
 * become gaps that get filled by other input streams so that the total
 * duration of our output is larger than the durations of any of the
 * upstream peers.  In general, there is no way to compute the duration of
 * our output without advance knowledge of the intervals of time for which
 * each input will provide data, which we don't have.  In the synchronous
 * case, the duration will always have to be an approximation that becomes
 * more accurate the closer we get to the true EOS.
 */


static gboolean query_duration(GSTLALTimeSliceChiSquare *element, GstQuery *query)
{
	GstIterator *it = NULL;
	gint64 max = -1;
	GstFormat format;
	gboolean success = TRUE;
	gboolean done = FALSE;

	/*
	 * parse duration query format
	 */

	gst_query_parse_duration(query, &format, NULL);

	/*
	 * iterate over sink pads
	 */

	it = gst_element_iterate_sink_pads(GST_ELEMENT_CAST(element));
	while(!done && success) {
		gpointer item;

		switch(gst_iterator_next(it, &item)) {
		case GST_ITERATOR_DONE:
			done = TRUE;
			break;

		case GST_ITERATOR_OK: {
			GstPad *pad = GST_PAD_CAST(item);
			gint64 duration;

			/*
			 * query upstream peer for duration
			 */

			if(gst_pad_query_peer_duration(pad, &format, &duration)) {
				/*
				 * query succeeded
				 */

				if(duration == -1) {
					/*
					 * unknown duration --> the
					 * duration of our output is
					 * unknown
					 */

					max = duration;
					done = TRUE;
				} else if(duration > max) {
					/*
					 * take largest duration
					 */

					max = duration;
				}
			} else {
				/*
				 * query failed
				 */

				success = FALSE;
			}
			gst_object_unref(pad);
			break;
		}

		case GST_ITERATOR_RESYNC:
			max = -1;
			success = TRUE;
			gst_iterator_resync(it);
			break;

		default:
			success = FALSE;
			done = TRUE;
			break;
		}
	}
	gst_iterator_free(it);

	if(success) {
		/*
		 * store the max
		 */

		GST_DEBUG_OBJECT(element, "Total duration in format %s: %" GST_TIME_FORMAT, gst_format_get_name(format), GST_TIME_ARGS(max));
		gst_query_set_duration(query, format, max);
	}

	return success;
}


static gboolean query_latency(GSTLALTimeSliceChiSquare *element, GstQuery *query)
{
	GstIterator *it = NULL;
	GstClockTime min = 0;
	GstClockTime max = GST_CLOCK_TIME_NONE;
	gboolean live = FALSE;
	gboolean success = TRUE;
	gboolean done = FALSE;

	/*
	 * iterate over sink pads
	 */

	it = gst_element_iterate_sink_pads(GST_ELEMENT_CAST(element));
	while(!done && success) {
		gpointer item;

		switch(gst_iterator_next(it, &item)) {
		case GST_ITERATOR_DONE:
			done = TRUE;
			break;

		case GST_ITERATOR_OK: {
			GstPad *pad = GST_PAD_CAST(item);
			GstQuery *peerquery = gst_query_new_latency();

			/* 
			 * query upstream peer for latency
			 */

			if(gst_pad_peer_query(pad, peerquery)) {
				/*
				 * query succeeded
				 */

				GstClockTime min_cur;
				GstClockTime max_cur;
				gboolean live_cur;

				gst_query_parse_latency(peerquery, &live_cur, &min_cur, &max_cur);

				/*
				 * take the largest of the latencies
				 */

				if(min_cur > min)
					min = min_cur;

				if(max_cur != GST_CLOCK_TIME_NONE && ((max != GST_CLOCK_TIME_NONE && max_cur > max) || (max == GST_CLOCK_TIME_NONE)))
					max = max_cur;

				/*
				 * we're live if any upstream peer is live
				 */

				live |= live_cur;
			} else {
				/*
				 * query failed
				 */

				success = FALSE;
			}

			gst_query_unref(peerquery);
			gst_object_unref(pad);
			break;
		}

		case GST_ITERATOR_RESYNC:
			live = FALSE;
			min = 0;
			max = GST_CLOCK_TIME_NONE;
			success = TRUE;
			gst_iterator_resync(it);
			break;

		default:
			success = FALSE;
			done = TRUE;
			break;
		}
	}
	gst_iterator_free(it);

	if(success) {
		/* store the results */
		GST_DEBUG_OBJECT(element, "Calculated total latency: live %s, min %" GST_TIME_FORMAT ", max %" GST_TIME_FORMAT, (live ? "yes" : "no"), GST_TIME_ARGS(min), GST_TIME_ARGS(max));
		gst_query_set_latency(query, live, min, max);
	}

	return success;
}


static gboolean query_function(GstPad *pad, GstQuery *query)
{
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(gst_pad_get_parent(pad));
	gboolean success = TRUE;

	switch(GST_QUERY_TYPE(query)) {
	case GST_QUERY_POSITION: {
		GstFormat format;

		gst_query_parse_position(query, &format, NULL);

		switch(format) {
		case GST_FORMAT_TIME:
			/* FIXME, bring to stream time, might be tricky */
			gst_query_set_position(query, format, output_timestamp_from_offset(element, element->offset));
			break;

		case GST_FORMAT_DEFAULT:
			/* default format for audio is sample count */
			gst_query_set_position(query, format, element->offset);
			break;

		default:
			success = FALSE;
			break;
		}
		break;
	}

	case GST_QUERY_DURATION:
		success = query_duration(element, query);
		break;

	case GST_QUERY_LATENCY:
		success = query_latency(element, query);
		break;

	default:
		/* FIXME, needs a custom query handler because we have multiple
		 * sinkpads */
		success = gst_pad_query_default(pad, query);
		break;
	}

	gst_object_unref(element);
	return success;
}


/*
 * ============================================================================
 *
 *                                   Events
 *
 * ============================================================================
 */


/*
 * helper function used by forward_event() (see below)
 */


static gboolean forward_event_func(GstPad *pad, GValue *ret, GstEvent *event)
{
	gst_event_ref(event);
	GST_LOG_OBJECT(pad, "pushing event %p (%s) on pad %s:%s", event, GST_EVENT_TYPE_NAME(event), GST_DEBUG_PAD_NAME(pad));
	if(!gst_pad_push_event(pad, event)) {
		g_value_set_boolean(ret, FALSE);
		GST_WARNING_OBJECT(pad, "event %p (%s) push failed.", event, GST_EVENT_TYPE_NAME(event));
	} else
		GST_LOG_OBJECT(pad, "event %p (%s) pushed.", event, GST_EVENT_TYPE_NAME(event));
	gst_object_unref(pad);
	return TRUE;
}


/*
 * forwards the event to all sinkpads.  takes ownership of the event.
 * returns TRUE if the event could be forwarded on all sinkpads, returns
 * FALSE if not.
 */


static gboolean forward_event(GSTLALTimeSliceChiSquare *element, GstEvent *event)
{
	GstIterator *it = NULL;
	GValue vret = {0};

	GST_LOG_OBJECT(element, "forwarding event %p (%s)", event, GST_EVENT_TYPE_NAME(event));

	g_value_init(&vret, G_TYPE_BOOLEAN);
	g_value_set_boolean(&vret, TRUE);
	it = gst_element_iterate_sink_pads(GST_ELEMENT_CAST(element));
	gst_iterator_fold(it, (GstIteratorFoldFunction) forward_event_func, &vret, event);
	gst_iterator_free(it);
	gst_event_unref(event);

	return g_value_get_boolean(&vret);
}


/*
 * src pad event handler
 */


static gboolean src_event_function(GstPad *pad, GstEvent *event)
{
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(gst_pad_get_parent(pad));
	gboolean result;

	switch (GST_EVENT_TYPE(event)) {
	case GST_EVENT_QOS:
	case GST_EVENT_NAVIGATION:
		/*
		 * not handled. QoS might be tricky, navigation is
		 * pointless.
		 */

		result = FALSE;
		break;

	case GST_EVENT_SEEK: {
		GstSeekFlags flags;
		GstSeekType curtype;
		gint64 cur;
		gboolean flush;

		/*
		 * parse the seek parameters
		 */

		gst_event_parse_seek(event, &element->segment.rate, NULL, &flags, &curtype, &cur, NULL, NULL);
		flush = !!(flags & GST_SEEK_FLAG_FLUSH);

		/*
		 * is it a flushing seek?
		 */

		if(flush) {
			/*
			 * make sure we accept nothing more and return
			 * WRONG_STATE
			 */

			gst_collect_pads_set_flushing(element->collect, TRUE);

			/*
			 * start flush downstream.  the flush will be done
			 * when all pads received a FLUSH_STOP.
			 */

			gst_pad_push_event(element->srcpad, gst_event_new_flush_start());
		}

		/*
		 * wait for the collected to be finished and mark a new
		 * segment
		 */

		GST_OBJECT_LOCK(element->collect);
		element->segment_pending = TRUE;
		if(flush) {
			/* Yes, we need to call _set_flushing again *WHEN* the streaming threads
			 * have stopped so that the cookie gets properly updated. */
			gst_collect_pads_set_flushing(element->collect, TRUE);
		}
		element->flush_stop_pending = flush;
		GST_OBJECT_UNLOCK(element->collect);

		result = forward_event(element, event);
		break;
	}

	default:
		/*
		 * forward the rest.
		 */

		result = forward_event(element, event);
		break;
	}

	/*
	 * done
	 */

	gst_object_unref(element);
	return result;
}


/*
 * sink pad event handler.  this is hacked in as an override of the collect
 * pads object's own event handler so that we can detect new segments and
 * flush stop events arriving on sink pads.  the real event handling is
 * accomplished by chaining to the original event handler installed by the
 * collect pads object.
 */


static gboolean sink_event_function(GstPad *pad, GstEvent *event)
{
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(gst_pad_get_parent(pad));

	GST_DEBUG("got event %p (%s) on pad %s:%s", event, GST_EVENT_TYPE_NAME(event), GST_DEBUG_PAD_NAME(pad));

	/*
	 * handle events
	 */

	switch (GST_EVENT_TYPE(event)) {
	case GST_EVENT_FLUSH_STOP:
		/*
		 * mark a pending new segment. This event is synchronized
		 * with the streaming thread so we can safely update the
		 * variable without races. It's somewhat weird because we
		 * assume the collectpads forwarded the FLUSH_STOP past us
		 * and downstream (using our source pad, the bastard!).
		 */

		GST_OBJECT_LOCK(element->collect);
		element->segment_pending = TRUE;
		element->flush_stop_pending = FALSE;
		GST_OBJECT_UNLOCK(element->collect);
		break;

	default:
		break;
	}

	/*
	 * now chain to GstCollectPads handler to take care of the rest.
	 */

	gst_object_unref(element);
	return element->collect_event(pad, event);
}


/*
 * ============================================================================
 *
 *                            \chi^{2} Computation
 *
 * ============================================================================
 */


static GstFlowReturn collected(GstCollectPads *pads, gpointer user_data)
{
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(user_data);
	GstClockTime t_start, t_end;
	guint64 earliest_input_offset, earliest_input_offset_end;
	guint64 sample, length;
	int unit_size = 0;
	GstBuffer *outbuf = NULL;
	gpointer outbytes = NULL;
	GstFlowReturn result;

	GstBuffer *snrbuf = NULL;
	GstBuffer *timeslicesnrbuf = NULL;
	gint channel, numchannels;

	/*
	 * forward flush-stop event
	 */

	if(element->flush_stop_pending) {
		gst_pad_push_event(element->srcpad, gst_event_new_flush_stop());
		element->flush_stop_pending = FALSE;
	}

	/*
	 * check for new segment
	 */

	if(element->segment_pending) {
		GstEvent *event;
		GstSegment *segment = gstlal_collect_pads_get_segment(element->collect);
		if(!segment) {
			/* FIXME:  failure getting bounding segment, do
			 * something about it */
		}
		element->segment = *segment;
		element->offset = 0;
		gst_segment_free(segment);

		event = gst_event_new_new_segment_full(FALSE, element->segment.rate, 1.0, GST_FORMAT_TIME, element->segment.start, element->segment.stop, element->segment.start);
		if(!event) {
			/* FIXME:  failure getting event, do something
			 * about it */
		}
		gst_pad_push_event(element->srcpad, event);

		element->segment_pending = FALSE;
	}

	/*
	 * get the range of offsets (in the output stream) spanned by the
	 * available input buffers.
	 */

	if(!gstlal_collect_pads_get_earliest_times(element->collect, &t_start, &t_end, element->rate)) {
		GST_ERROR_OBJECT(element, "cannot deduce input timestamp offset information");
		goto error;
	}
	fprintf(stderr,"tstart=%i, tend=%i\n", (int) t_start, (int) t_end);

	/*
	 * check for EOS
	 */

	if(!GST_CLOCK_TIME_IS_VALID(t_start))
		goto eos;

	/*
	 * don't let time go backwards.  in principle we could be smart and
	 * handle this, but the audiorate element can be used to correct
	 * screwed up time series so there is no point in re-inventing its
	 * capabilities here.
	 */

	earliest_input_offset = gst_util_uint64_scale_int_round(t_start - element->segment.start, element->rate, GST_SECOND);
	earliest_input_offset_end = gst_util_uint64_scale_int_round(t_end - element->segment.start, element->rate, GST_SECOND);
	if(earliest_input_offset < element->offset) {
		GST_ERROR_OBJECT(element, "detected time reversal in at least one input stream:  expected nothing earlier than offset %" G_GUINT64_FORMAT ", found sample at offset %" G_GUINT64_FORMAT, element->offset, earliest_input_offset);
		result = GST_FLOW_ERROR;
		goto error;
	}

	/*
	 * compute the number of samples in each channel
	 */

	length = earliest_input_offset_end - earliest_input_offset;

	/*
	 * get buffers upto the desired end offset.
	 */

	snrbuf = gstlal_collect_pads_take_buffer_sync(pads, element->snrcollectdata, t_end, element->rate);
	timeslicesnrbuf = gstlal_collect_pads_take_buffer_sync(pads, element->timeslicesnrcollectdata, t_end, element->rate);

	/*
	 * NULL means EOS.
	 */

	if(!snrbuf || !timeslicesnrbuf) {
		if(snrbuf)
			gst_buffer_unref(snrbuf);
		if(timeslicesnrbuf)
			gst_buffer_unref(timeslicesnrbuf);
		goto eos;
	}

	/* determine the output unit_size */

	GSList *collectdatalist = NULL;
	for(collectdatalist = pads->data; collectdatalist; collectdatalist = g_slist_next(collectdatalist)) {
		GstLALCollectData *data = collectdatalist->data;

		unit_size = data->unit_size;
	}
		

	/*
	 * alloc a new output buffer of length samples, and
	 * set its offset.
	 */

	GST_LOG_OBJECT(element, "requesting output buffer of %" G_GUINT64_FORMAT " samples", length);

	result = gst_pad_alloc_buffer(element->srcpad, earliest_input_offset, length * unit_size, GST_PAD_CAPS(element->srcpad), &outbuf);
	if(result != GST_FLOW_OK) {
		/* FIXME: handle failure */
		outbuf = NULL;
	}
	outbytes = GST_BUFFER_DATA(outbuf);

	/*
	 * Check for mis-aligned input buffers.  This can happen, but we
	 * can't handle it.
	 */

	if(GST_BUFFER_OFFSET(snrbuf) != GST_BUFFER_OFFSET(timeslicesnrbuf) || GST_BUFFER_OFFSET_END(snrbuf) != GST_BUFFER_OFFSET_END(timeslicesnrbuf)) {
		GST_ERROR_OBJECT(element, "misaligned buffer boundaries:  got snr offsets [%" G_GUINT64_FORMAT ", %" G_GUINT64_FORMAT ") and time-slice snr offsets [%" G_GUINT64_FORMAT ", %" G_GUINT64_FORMAT ")", GST_BUFFER_OFFSET(snrbuf), GST_BUFFER_OFFSET_END(snrbuf), GST_BUFFER_OFFSET(timeslicesnrbuf), GST_BUFFER_OFFSET_END(timeslicesnrbuf));
		goto error;
	}

	/*
	 * check for discontinuity
	 */

	if(element->offset != GST_BUFFER_OFFSET(outbuf))
		GST_BUFFER_FLAG_SET(outbuf, GST_BUFFER_FLAG_DISCONT);

	/*
	 * Gap --> pass-through
	 */

	if(GST_BUFFER_FLAG_IS_SET(snrbuf, GST_BUFFER_FLAG_GAP) || GST_BUFFER_FLAG_IS_SET(timeslicesnrbuf, GST_BUFFER_FLAG_GAP)) {
		memset(GST_BUFFER_DATA(outbuf), 0, GST_BUFFER_SIZE(outbuf));
		GST_BUFFER_FLAG_SET(outbuf, GST_BUFFER_FLAG_GAP);
		goto done;
	}

	/*
	 * make sure the chifacs vectors is available, wait until it is
	 */

	g_mutex_lock(element->coefficients_lock);
	while(!element->chifacs) {
		g_cond_wait(element->coefficients_available, element->coefficients_lock);
		/* FIXME:  we need some way of getting out of this loop.
		 * maybe check for a flag set in an event handler */
	}

	numchannels = (guint) num_channels(element);

	for(sample = 0; sample < length; sample++) {
		double *data = &((double *) GST_BUFFER_DATA(outbuf))[numchannels * sample];
		const double *snrdata = &((const double *) GST_BUFFER_DATA(snrbuf))[numchannels * sample];
		const double *timeslicesnrdata = &((const double *) GST_BUFFER_DATA(timeslicesnrbuf))[numchannels * sample];
		for(channel = 0; channel < numchannels; channel++) {
			double snr = snrdata[channel];
			double timeslicesnr = timeslicesnrdata[channel];
			double chifacs_coefficient = gsl_vector_get(element->chifacs, channel);
			double chifacs_coefficient2 = chifacs_coefficient*chifacs_coefficient;
			double chifacs_coefficient3 = chifacs_coefficient2*chifacs_coefficient;

			data[channel] = pow(snr * chifacs_coefficient - timeslicesnr, 2.0)/2.0/(chifacs_coefficient2 - chifacs_coefficient3);
		}
	}
	g_mutex_unlock(element->coefficients_lock);

	/*
	 * push the buffer downstream
	 */

done:
	gst_buffer_unref(timeslicesnrbuf);
	gst_buffer_unref(snrbuf);
	element->offset = GST_BUFFER_OFFSET_END(outbuf);
	return gst_pad_push(element->srcpad, outbuf);

eos:
	GST_DEBUG_OBJECT(element, "no data available (EOS)");
	gst_pad_push_event(element->srcpad, gst_event_new_eos());
	return GST_FLOW_UNEXPECTED;

error:
	if(outbuf)
		gst_buffer_unref(outbuf);
	if(snrbuf)
		gst_buffer_unref(snrbuf);
	if(timeslicesnrbuf)
		gst_buffer_unref(timeslicesnrbuf);
	return GST_FLOW_ERROR;
}


/*
 * ============================================================================
 *
 *                          GObject Method Overrides
 *
 * ============================================================================
 */


/*
 * set_property()
 */


static void set_property(GObject *object, enum property prop_id, const GValue *value, GParamSpec *pspec)
{
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_CHIFACS: {
		int channels;
		g_mutex_lock(element->coefficients_lock);
		if(element->chifacs) {
			channels = num_channels(element);
			gsl_vector_free(element->chifacs);
		} else
			channels = 0;
		element->chifacs = gstlal_gsl_vector_from_g_value_array(g_value_get_boxed(value));

		/*
		 * number of channels has changed, force a caps
		 * renegotiation
		 */

		if(num_channels(element) != channels) {
			/* FIXME:  what do we do here? */
			fprintf(stderr,"CHANNELS DON'T MATCH, channels=%i, num_channels=%i\n", channels, num_channels(element));
			gst_pad_set_caps(element->srcpad, NULL);
		}

		/*
		 * signal availability of new chifacs vector
		 */

		g_cond_broadcast(element->coefficients_available);
		g_mutex_unlock(element->coefficients_lock);
		break;
	}

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * get_property()
 */


static void get_property(GObject *object, enum property prop_id, GValue *value, GParamSpec *pspec)
{
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_CHIFACS:
		g_mutex_lock(element->coefficients_lock);
		if(element->chifacs)
			g_value_take_boxed(value, gstlal_g_value_array_from_gsl_vector(element->chifacs));
		/* FIXME:  else? */
		g_mutex_unlock(element->coefficients_lock);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
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
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(object);

	gst_object_unref(element->timeslicesnrpad);
	element->timeslicesnrpad = NULL;
	gst_object_unref(element->snrpad);
	element->snrpad = NULL;
	gst_object_unref(element->srcpad);
	element->srcpad = NULL;

	gst_object_unref(element->collect);
	element->timeslicesnrcollectdata = NULL;
	element->snrcollectdata = NULL;
	element->collect = NULL;

	g_mutex_free(element->coefficients_lock);
	element->coefficients_lock = NULL;
	g_cond_free(element->coefficients_available);
	element->coefficients_available = NULL;
	if(element->chifacs) {
		gsl_vector_free(element->chifacs);
		element->chifacs = NULL;
	}

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


/*
 * change state.  reset element's internal state and start the collect pads
 * on READY --> PAUSED state change.  stop the collect pads on PAUSED -->
 * READY state change.
 */


static GstStateChangeReturn change_state(GstElement *element, GstStateChange transition)
{
	GSTLALTimeSliceChiSquare *timeslicechisquare = GSTLAL_TIMESLICECHISQUARE(element);

	switch(transition) {
	case GST_STATE_CHANGE_NULL_TO_READY:
		break;

	case GST_STATE_CHANGE_READY_TO_PAUSED:
		timeslicechisquare->segment_pending = TRUE;
		gst_segment_init(&timeslicechisquare->segment, GST_FORMAT_UNDEFINED);
		timeslicechisquare->offset = 0;
		gst_collect_pads_start(timeslicechisquare->collect);
		break;

	case GST_STATE_CHANGE_PAUSED_TO_PLAYING:
		break;

	case GST_STATE_CHANGE_PAUSED_TO_READY:
		/* need to unblock the collectpads before calling the
		 * parent change_state so that streaming can finish */
		gst_collect_pads_stop(timeslicechisquare->collect);
		break;

	default:
		break;
	}

	return parent_class->change_state(element, transition);
}


/*
 * Base init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GBaseInitFunc
 */



static void base_init(gpointer class)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(class);

	gst_element_class_set_details_simple(
		element_class,
		"Inspiral time-slice-based \\chi^{2}",
		"Filter",
		"A time-slice-based \\chi^{2} statistic for the inspiral pipeline",
		"Kipp Cannon <kipp.cannon@ligo.org>, Chad Hanna <channa@ligo.caltech.edu>, Drew Keppel <drew.keppel@ligo.org>"
	);
	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"timeslicesnr",
			GST_PAD_SINK,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
				"audio/x-raw-float, " \
				"rate = (int) [ 1, MAX ], " \
				"channels = (int) [ 1, MAX ], " \
				"endianness = (int) BYTE_ORDER, " \
				"width = (int) 64"
			)
		)
	);
	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"snr",
			GST_PAD_SINK,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
				"audio/x-raw-float, " \
				"rate = (int) [ 1, MAX ], " \
				"channels = (int) [ 1, MAX ], " \
				"endianness = (int) BYTE_ORDER, " \
				"width = (int) 64"
			)
		)
	);
	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"src",
			GST_PAD_SRC,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
				"audio/x-raw-float, " \
				"rate = (int) [ 1, MAX ], " \
				"channels = (int) [ 1, MAX ], " \
				"endianness = (int) BYTE_ORDER, " \
				"width = (int) 64"
			)
		)
	);
}


/*
 * Class init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GClassInitFunc
 */


static void class_init(gpointer klass, gpointer class_data)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *gstelement_class = GST_ELEMENT_CLASS(klass);

	parent_class = g_type_class_ref(GST_TYPE_ELEMENT);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	gstelement_class->change_state = GST_DEBUG_FUNCPTR(change_state);

	g_object_class_install_property(
		gobject_class,
		ARG_CHIFACS,
		g_param_spec_value_array(
			"chifacs",
			"Chisquared Factors",
			"Vector of chisquared factors. Number of rows sets number of channels.",
			g_param_spec_double(
				"sample",
				"Sample",
				"Chifacs sample",
				-G_MAXDOUBLE, G_MAXDOUBLE, 0.0,
				G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
			),
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
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(object);
	GstPad *pad;

	gst_element_create_all_pads(GST_ELEMENT(element));
	element->collect = gst_collect_pads_new();
	gst_collect_pads_set_function(element->collect, GST_DEBUG_FUNCPTR(collected), element);

	/* configure (and ref) timeslice SNR sink pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "timeslicesnr");
	element->collect_event = (GstPadEventFunction) GST_PAD_EVENTFUNC(pad);
	gst_pad_set_event_function(pad, GST_DEBUG_FUNCPTR(sink_event_function));
	gst_pad_set_getcaps_function(pad, GST_DEBUG_FUNCPTR(getcaps_timeslicesnr));
	gst_pad_set_setcaps_function(pad, GST_DEBUG_FUNCPTR(setcaps_timeslicesnr));
	element->timeslicesnrcollectdata = gstlal_collect_pads_add_pad(element->collect, pad, sizeof(*element->timeslicesnrcollectdata));
	element->timeslicesnrpad = pad;

	/* configure (and ref) SNR sink pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "snr");
	element->collect_event = (GstPadEventFunction) GST_PAD_EVENTFUNC(pad);
	gst_pad_set_event_function(pad, GST_DEBUG_FUNCPTR(sink_event_function));
	gst_pad_set_getcaps_function(pad, GST_DEBUG_FUNCPTR(getcaps_snr));
	gst_pad_set_setcaps_function(pad, GST_DEBUG_FUNCPTR(setcaps_snr));
	element->snrcollectdata = gstlal_collect_pads_add_pad(element->collect, pad, sizeof(*element->snrcollectdata));
	element->snrpad = pad;

	/* retrieve (and ref) src pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "src");
	gst_pad_set_event_function(pad, GST_DEBUG_FUNCPTR(src_event_function));
	gst_pad_set_query_function(pad, GST_DEBUG_FUNCPTR(query_function));
	element->srcpad = pad;

	/* internal data */
	element->rate = 0;
	element->coefficients_lock = g_mutex_new();
	element->coefficients_available = g_cond_new();
	element->chifacs = NULL;
}


/*
 * gstlal_timeslicechisquare_get_type().
 */


GType gstlal_timeslicechisquare_get_type(void)
{
	static GType type = 0;

	if(!type) {
		static const GTypeInfo info = {
			.class_size = sizeof(GSTLALTimeSliceChiSquareClass),
			.class_init = class_init,
			.base_init = base_init,
			.instance_size = sizeof(GSTLALTimeSliceChiSquare),
			.instance_init = instance_init,
		};
		type = g_type_register_static(GST_TYPE_ELEMENT, "lal_timeslicechisquare", &info, 0);
	}

	return type;
}

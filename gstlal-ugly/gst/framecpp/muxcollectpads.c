/*
 * FrameCPPMuxCollectPads
 *
 * Copyright (C) 2012,2013  Kipp Cannon
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
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


#include <glib.h>
#include <gst/gst.h>


#include <gstlal/gstlal_debug.h>
#include <muxcollectpads.h>
#include <muxqueue.h>
#include <marshal.h>


#ifndef GST_BUFFER_LIST_BOUNDARIES_FORMAT
#define GST_BUFFER_LIST_BOUNDARIES_FORMAT ".d[%" GST_TIME_SECONDS_FORMAT ", %" GST_TIME_SECONDS_FORMAT ") = offsets [%" G_GUINT64_FORMAT ", %" G_GUINT64_FORMAT ")"
#define GST_BUFFER_LIST_BOUNDARIES_ARGS(list) 0, GST_TIME_SECONDS_ARGS(GST_BUFFER_TIMESTAMP(GST_BUFFER(g_list_first(list)->data))), GST_TIME_SECONDS_ARGS(GST_BUFFER_TIMESTAMP(GST_BUFFER(g_list_last(list)->data)) + GST_BUFFER_DURATION(GST_BUFFER(g_list_last(list)->data))), GST_BUFFER_OFFSET(GST_BUFFER(g_list_first(list)->data)), GST_BUFFER_OFFSET_END(GST_BUFFER(g_list_last(list)->data))
#endif /* GST_BUFFER_SLIST_BOUNDARIES_FORMAT */


/*
 * ============================================================================
 *
 *                                Boilerplate
 *
 * ============================================================================
 */


GST_BOILERPLATE(FrameCPPMuxCollectPads, framecpp_muxcollectpads, GstObject, GST_TYPE_OBJECT);


/*
 * ============================================================================
 *
 *                                 Parameters
 *
 * ============================================================================
 */


#define DEFAULT_MAX_SIZE_TIME GST_SECOND


/*
 * ============================================================================
 *
 *                                  Signals
 *
 * ============================================================================
 */


enum framecpp_muxcollectpads_signal {
	SIGNAL_COLLECTED,
	NUM_SIGNALS
};


static guint signals[NUM_SIGNALS] = {0, };


/*
 * ============================================================================
 *
 *                             Internal Functions
 *
 * ============================================================================
 */


static GstFlowReturn chain(GstPad *pad, GstBuffer *buffer)
{
	FrameCPPMuxCollectPadsData *data = gst_pad_get_element_private(pad);
	FrameCPPMuxCollectPads *collectpads = data->collect;
	GstFlowReturn result;

	g_assert(GST_IS_FRAMECPP_MUXCOLLECTPADS(collectpads));

	if(data->eos || data->segment.format == GST_FORMAT_UNDEFINED) {
		gst_buffer_unref(buffer);
		result = GST_FLOW_UNEXPECTED;
	} else
		result = framecpp_muxqueue_push(data->queue, buffer) ? GST_FLOW_OK : GST_FLOW_ERROR;

	return result;
}


static gboolean all_pads_are_at_eos(FrameCPPMuxCollectPads *collectpads)
{
	GSList *collectdatalist;

	FRAMECPP_MUXCOLLECTPADS_PADS_LOCK(collectpads);
	for(collectdatalist = collectpads->pad_list; collectdatalist; collectdatalist = g_slist_next(collectdatalist))
		if(!((FrameCPPMuxCollectPadsData *) collectdatalist->data)->eos) {
			FRAMECPP_MUXCOLLECTPADS_PADS_UNLOCK(collectpads);
			return FALSE;
		}
	FRAMECPP_MUXCOLLECTPADS_PADS_UNLOCK(collectpads);
	return TRUE;
}


static gboolean update_segment(FrameCPPMuxCollectPads *collectpads)
{
	GSList *collectdatalist;
	gboolean success = TRUE;

	/*
	 * clear the segment boundaries
	 */

	gst_segment_init(&collectpads->segment, GST_FORMAT_UNDEFINED);

	/*
	 * loop over pads
	 */

	for(collectdatalist = collectpads->pad_list; collectdatalist; collectdatalist = g_slist_next(collectdatalist)) {
		FrameCPPMuxCollectPadsData *data = collectdatalist->data;

		/*
		 * all pads must have segments
		 */

		if(data->segment.format == GST_FORMAT_UNDEFINED || data->segment.start == -1) {
			GST_DEBUG_OBJECT(collectpads, "%" GST_PTR_FORMAT ": segment not known", data->pad);
			success = FALSE;
			goto done;
		}

		/*
		 * if this is the first segment we've found, initialize from it
		 */

		if(collectpads->segment.format == GST_FORMAT_UNDEFINED) {
			collectpads->segment = data->segment;	/* FIXME:  is this OK? */
			continue;
		}

		/*
		 * check for format/rate mismatch
		 */

		if(collectpads->segment.format != data->segment.format || collectpads->segment.applied_rate != data->segment.applied_rate) {
			GST_ERROR_OBJECT(collectpads, "%" GST_PTR_FORMAT ": mismatch in segment format and/or applied rate", data->pad);
			success = FALSE;
			goto done;
		}

		/*
		 * expand start and stop
		 */

		GST_DEBUG_OBJECT(collectpads, "%" GST_PTR_FORMAT ": have segment [%" G_GUINT64_FORMAT ", %" G_GUINT64_FORMAT ")", data->pad, data->segment.start, data->segment.stop);
		if(collectpads->segment.start > data->segment.start)
			collectpads->segment.start = data->segment.start;
		if(data->segment.stop == -1 || (collectpads->segment.stop != -1 && collectpads->segment.stop < data->segment.stop))
			collectpads->segment.stop = data->segment.stop;
	}

	/*
	 * success?
	 */

	if(collectpads->segment.format == GST_FORMAT_UNDEFINED) {
		GST_ERROR_OBJECT(collectpads, "failed to compute union of input segments");
		success = FALSE;
		goto done;
	}
	GST_DEBUG_OBJECT(collectpads, "union of segments = [%" G_GUINT64_FORMAT ", %" G_GUINT64_FORMAT ")", collectpads->segment.start, collectpads->segment.stop);
done:
	return success;
}


static gboolean event(GstPad *pad, GstEvent *event)
{
	FrameCPPMuxCollectPadsData *data = gst_pad_get_element_private(pad);
	FrameCPPMuxCollectPads *collectpads = data->collect;
	GstPadEventFunction event_func = data->event_func;
	gboolean success = TRUE;

	g_assert(GST_IS_FRAMECPP_MUXCOLLECTPADS(collectpads));

	GST_OBJECT_LOCK(collectpads);

	switch(GST_EVENT_TYPE(event)) {
	case GST_EVENT_NEWSEGMENT: {
		gboolean update;
		gdouble rate, applied_rate;
		GstFormat format;
		gint64 start, stop, position;
		gst_event_parse_new_segment_full(event, &update, &rate, &applied_rate, &format, &start, &stop, &position);
		gst_event_unref(event);
		GST_DEBUG_OBJECT(pad, "new segment [%" G_GINT64_FORMAT ", %" G_GINT64_FORMAT ")", start, stop);

		FRAMECPP_MUXCOLLECTPADS_PADS_LOCK(collectpads);
		gst_segment_set_newsegment_full(&data->segment, update, rate, applied_rate, format, start, stop, position);
		data->eos = FALSE;
		success &= update_segment(collectpads);
		FRAMECPP_MUXCOLLECTPADS_PADS_UNLOCK(collectpads);
		if(success && event_func)
			success &= event_func(pad, gst_event_new_new_segment_full(update, collectpads->segment.rate, collectpads->segment.applied_rate, collectpads->segment.format, collectpads->segment.start, collectpads->segment.stop, collectpads->segment.time));
		break;
	}

	case GST_EVENT_FLUSH_START:
	case GST_EVENT_FLUSH_STOP:
		framecpp_muxqueue_set_flushing(data->queue, GST_EVENT_TYPE(event) == GST_EVENT_FLUSH_START);
		if(event_func)
			success &= event_func(pad, event);
		else
			gst_event_unref(event);
		break;

	case GST_EVENT_EOS:
		GST_DEBUG_OBJECT(pad, "received EOS");
		gst_segment_init(&data->segment, GST_FORMAT_UNDEFINED);
		data->eos = TRUE;
		if(all_pads_are_at_eos(collectpads)) {
			GST_DEBUG_OBJECT(collectpads, "all sink pads are at EOS");
			if(event_func)
				success &= event_func(pad, event);
		} else
			gst_event_unref(event);
		break;

	default:
		if(event_func)
			success &= event_func(pad, event);
		else
			gst_event_unref(event);
		break;
	}

	GST_OBJECT_UNLOCK(collectpads);

	return success;
}


static gboolean get_common_span(FrameCPPMuxCollectPads *collectpads, GstClockTime *min_t_start, GstClockTime *min_t_end)
{
	GSList *collectdatalist;

	*min_t_start = *min_t_end = GST_CLOCK_TIME_NONE;

	FRAMECPP_MUXCOLLECTPADS_PADS_LOCK(collectpads);
	for(collectdatalist = collectpads->pad_list; collectdatalist; collectdatalist = g_slist_next(collectdatalist)) {
		FrameCPPMuxCollectPadsData *data = collectdatalist->data;
		GstClockTime t_start = framecpp_muxqueue_timestamp(data->queue);
		GstClockTime t_end = t_start + framecpp_muxqueue_duration(data->queue);;

		if(!GST_CLOCK_TIME_IS_VALID(t_start)) {
			if(data->eos)
				continue;
			if(data->segment.format == GST_FORMAT_UNDEFINED) {
				GST_DEBUG_OBJECT(collectpads, "%" GST_PTR_FORMAT " has no data, no segment", data->pad);
				FRAMECPP_MUXCOLLECTPADS_PADS_UNLOCK(collectpads);
				return FALSE;
			}
			g_assert(data->segment.format == GST_FORMAT_TIME);
			t_start = t_end = data->segment.start;
		}

		GST_DEBUG_OBJECT(collectpads, "%" GST_PTR_FORMAT ": queue spans [%" GST_TIME_SECONDS_FORMAT ", %" GST_TIME_SECONDS_FORMAT ")", data->pad, GST_TIME_SECONDS_ARGS(t_start), GST_TIME_SECONDS_ARGS(t_end));

		g_assert(GST_CLOCK_TIME_IS_VALID(t_start));
		g_assert(GST_CLOCK_TIME_IS_VALID(t_end));
		g_assert_cmpuint(t_start, <=, t_end);
		*min_t_start = GST_CLOCK_TIME_IS_VALID(*min_t_start) ? MIN(*min_t_start, t_start) : t_start;
		*min_t_end = GST_CLOCK_TIME_IS_VALID(*min_t_end) ? MIN(*min_t_end, t_end) : t_end;
	}
	FRAMECPP_MUXCOLLECTPADS_PADS_UNLOCK(collectpads);

	g_assert(GST_CLOCK_TIME_IS_VALID(*min_t_start));
	g_assert(GST_CLOCK_TIME_IS_VALID(*min_t_end));

	return TRUE;
}


static gboolean get_span(FrameCPPMuxCollectPads *collectpads, GstClockTime *min_t_start, GstClockTime *max_t_end)
{
	GSList *collectdatalist;

	*min_t_start = *max_t_end = GST_CLOCK_TIME_NONE;

	FRAMECPP_MUXCOLLECTPADS_PADS_LOCK(collectpads);
	for(collectdatalist = collectpads->pad_list; collectdatalist; collectdatalist = g_slist_next(collectdatalist)) {
		FrameCPPMuxCollectPadsData *data = collectdatalist->data;
		GstClockTime t_start = framecpp_muxqueue_timestamp(data->queue);
		GstClockTime t_end = t_start + framecpp_muxqueue_duration(data->queue);;

		if(!GST_CLOCK_TIME_IS_VALID(t_start)) {
			if(data->eos)
				continue;
			if(data->segment.format == GST_FORMAT_UNDEFINED) {
				GST_DEBUG_OBJECT(collectpads, "%" GST_PTR_FORMAT " has no data, no segment", data->pad);
				FRAMECPP_MUXCOLLECTPADS_PADS_UNLOCK(collectpads);
				return FALSE;
			}
			g_assert(data->segment.format == GST_FORMAT_TIME);
			t_start = t_end = data->segment.start;
		}

		GST_DEBUG_OBJECT(collectpads, "%" GST_PTR_FORMAT ": queue spans [%" GST_TIME_SECONDS_FORMAT ", %" GST_TIME_SECONDS_FORMAT ")", data->pad, GST_TIME_SECONDS_ARGS(t_start), GST_TIME_SECONDS_ARGS(t_end));

		g_assert(GST_CLOCK_TIME_IS_VALID(t_start));
		g_assert(GST_CLOCK_TIME_IS_VALID(t_end));
		g_assert_cmpuint(t_start, <=, t_end);
		*min_t_start = GST_CLOCK_TIME_IS_VALID(*min_t_start) ? MIN(*min_t_start, t_start) : t_start;
		*max_t_end = GST_CLOCK_TIME_IS_VALID(*max_t_end) ? MAX(*max_t_end, t_end) : t_end;
	}
	FRAMECPP_MUXCOLLECTPADS_PADS_UNLOCK(collectpads);

	g_assert(GST_CLOCK_TIME_IS_VALID(*min_t_start));
	g_assert(GST_CLOCK_TIME_IS_VALID(*max_t_end));

	return TRUE;
}


static void waiting_handler(FrameCPPMuxQueue *queue, FrameCPPMuxCollectPadsData *activedata)
{
	FrameCPPMuxCollectPads *collectpads = activedata->collect;
	GstClockTime min_t_start;
	GstClockTime min_t_end;

	g_assert(GST_IS_FRAMECPP_MUXCOLLECTPADS(collectpads));
	GST_DEBUG_OBJECT(collectpads, "woken by %" GST_PTR_FORMAT, activedata->pad);

	GST_OBJECT_LOCK(collectpads);

	/*
	 * if the common interval of data has changed, wake up the
	 * streaming task
	 */

	if(!get_common_span(collectpads, &min_t_start, &min_t_end)) {
		GST_DEBUG_OBJECT(collectpads, "going back to sleep");
		goto done;
	}
	GST_DEBUG_OBJECT(collectpads, "[%" GST_TIME_SECONDS_FORMAT ", %" GST_TIME_SECONDS_FORMAT ") available on all pads", GST_TIME_SECONDS_ARGS(min_t_start), GST_TIME_SECONDS_ARGS(min_t_end));
	if(min_t_start != collectpads->min_t_start || min_t_end != collectpads->min_t_end) {
		collectpads->min_t_start = min_t_start;
		collectpads->min_t_end = min_t_end;
		g_signal_emit(collectpads, signals[SIGNAL_COLLECTED], 0, collectpads->min_t_start, collectpads->min_t_end);
	}

	/*
	 * done
	 */

done:
	GST_OBJECT_UNLOCK(collectpads);
}


/*
 * ============================================================================
 *
 *                                Exported API
 *
 * ============================================================================
 */


/**
 * Add a pad to the collect pads.  This is the only way to allocate a new
 * FrameCPPMuxCollectPadsData structure.  The calling code does not own the
 * FrameCPPMuxCollectPadsData structure, it is owned by the
 * FrameCPPMuxCollectPads object for which it has been allocated.
 *
 * This function should be called with the collectpads' object lock held.
 */


FrameCPPMuxCollectPadsData *framecpp_muxcollectpads_add_pad(FrameCPPMuxCollectPads *collectpads, GstPad *pad, FrameCPPMuxCollectPadsDataDestroyNotify destroy_notify)
{
	FrameCPPMuxCollectPadsData *data = g_malloc0(sizeof(*data));

	FRAMECPP_MUXCOLLECTPADS_PADS_LOCK(collectpads);

	data->collect = collectpads;
	data->pad = gst_object_ref(pad);
	data->queue = FRAMECPP_MUXQUEUE(g_object_new(FRAMECPP_MUXQUEUE_TYPE, "max-size-time", collectpads->max_size_time, NULL));
	gst_segment_init(&data->segment, GST_FORMAT_UNDEFINED);
	data->appdata = NULL;
	data->destroy_notify = destroy_notify;
	data->eos = FALSE;

	GST_OBJECT_LOCK(pad);
	gst_pad_set_element_private(pad, data);
	GST_OBJECT_UNLOCK(pad);
	gst_pad_set_chain_function(pad, GST_DEBUG_FUNCPTR(chain));
	gst_pad_set_event_function(pad, GST_DEBUG_FUNCPTR(event));
	if(collectpads->started)
		gst_pad_set_active(pad, TRUE);

	collectpads->pad_list = g_slist_append(collectpads->pad_list, data);
	data->waiting_handler_id = g_signal_connect(data->queue, "waiting", G_CALLBACK(waiting_handler), data);

	FRAMECPP_MUXCOLLECTPADS_PADS_UNLOCK(collectpads);

	return data;
}


/**
 * Remove a pad from the collect pads.  This is the only way to free a
 * FrameCPPMuxCollectPadsData structure.
 *
 * This function should be called with the collectpads' object lock held.
 */


gboolean framecpp_muxcollectpads_remove_pad(FrameCPPMuxCollectPads *collectpads, GstPad *pad)
{
	FrameCPPMuxCollectPadsData *data;
	gboolean success = TRUE;

	FRAMECPP_MUXCOLLECTPADS_PADS_LOCK(collectpads);

	GST_OBJECT_LOCK(pad);
	data = gst_pad_get_element_private(pad);
	if(data->destroy_notify)
		data->destroy_notify(data);
	gst_pad_set_element_private(pad, NULL);
	GST_OBJECT_UNLOCK(pad);
	if(!collectpads->started)
		gst_pad_set_active(pad, FALSE);

	collectpads->pad_list = g_slist_remove(collectpads->pad_list, data);

	gst_object_unref(data->pad);
	data->pad = NULL;

	/*
	 * FIXME:  there is a race condition:  if the waiting handler is
	 * invoked while the pad removal process is occuring, a
	 * use-after-free error can occur.  the chances of this happening
	 * are small, and it's tricky to prevent, so I'm just leaving it
	 * for now.
	 */

	g_signal_handler_disconnect(data->queue, data->waiting_handler_id);

	gst_object_unref(data->queue);
	data->queue = NULL;

	g_free(data);

	FRAMECPP_MUXCOLLECTPADS_PADS_UNLOCK(collectpads);

	return success;
}


/**
 * Retrieve the FrameCPPMuxCollectPadsData associated with the GstPad.
 */


FrameCPPMuxCollectPadsData *framecpp_muxcollectpads_get_data(GstPad *pad)
{
	return pad->element_private;
}


/**
 * Set an event function for a pad.  An internal event handler will be
 * installed on the pad to do work required by the FrameCPPMuxCollectPads
 * object, and that event handler will chain to the event handler set using
 * this function.
 *
 * Newsegment and EOS events are intercepted, and only chained to the
 * handler set using this function when all pads on the
 * FrameCPPMuxCollectPads have received newsegments or EOS events,
 * respectively.  That is, when the user-supplied event function will only
 * see one EOS event regardless of how many sink pads it has, and when it
 * sees an EOS event all sink pads are at EOS and the element should
 * respond appropriately.
 *
 * The event handler is invoked with the collectpads' object lock held.
 */


void framecpp_muxcollectpads_set_event_function(FrameCPPMuxCollectPadsData *data, GstPadEventFunction func)
{
	data->event_func = func;
}


/**
 * Set all pads to flushing or not flushing
 */


void framecpp_muxcollectpads_set_flushing(FrameCPPMuxCollectPads *collectpads, gboolean flushing)
{
	GSList *pad_list;

	FRAMECPP_MUXCOLLECTPADS_PADS_LOCK(collectpads);
	for(pad_list = collectpads->pad_list; pad_list; pad_list = g_slist_next(pad_list)) {
		FrameCPPMuxCollectPadsData *data = (FrameCPPMuxCollectPadsData *) pad_list->data;
		framecpp_muxqueue_set_flushing(data->queue, flushing);
	}
	FRAMECPP_MUXCOLLECTPADS_PADS_UNLOCK(collectpads);
}


/**
 * Start the collect pads
 */


void framecpp_muxcollectpads_start(FrameCPPMuxCollectPads *collectpads)
{
	GST_OBJECT_LOCK(collectpads);
	collectpads->min_t_start = GST_CLOCK_TIME_NONE;
	collectpads->min_t_end = GST_CLOCK_TIME_NONE;
	collectpads->started = TRUE;
	framecpp_muxcollectpads_set_flushing(collectpads, FALSE);
	GST_OBJECT_UNLOCK(collectpads);
}


/**
 * Stop the collect pads.
 */


void framecpp_muxcollectpads_stop(FrameCPPMuxCollectPads *collectpads)
{
	GST_OBJECT_LOCK(collectpads);
	collectpads->started = FALSE;
	framecpp_muxcollectpads_set_flushing(collectpads, TRUE);
	GST_OBJECT_UNLOCK(collectpads);
}


/**
 * Determine the interval of time up to the end of which all pads have
 * data.  Returns TRUE on success, FALSE if one or more pads do not yet
 * have data or segment information.  On success, min_t_start will be
 * populated with the earliest time for which data is available;  min_t_end
 * will be populated with the earliest of the last times for which data is
 * available.  Not all pads will have data for all times in between.  On
 * failure min_t_start and min_t_end are undefined.  Should be called with
 * the colledpads' object lock held.
 */


gboolean framecpp_muxcollectpads_get_common_span(FrameCPPMuxCollectPads *collectpads, GstClockTime *min_t_start, GstClockTime *min_t_end)
{
	return get_common_span(collectpads, min_t_start, min_t_end);
}


/**
 * Determine the interval of time spanned by the data on all pads.  Returns
 * TRUE on success, FALSE if one or more pads do not yet have data or
 * segment information.  On success, min_t_start will be populated with the
 * earliest time for which data is available;  max_t_end will be populated
 * with the last time for which data is available.  Not all pads will have
 * data for all times in between.  On failure, min_t_start and max_t_end
 * are undefined.  Should be called with the colledpads' object lock held.
 */


gboolean framecpp_muxcollectpads_get_span(FrameCPPMuxCollectPads *collectpads, GstClockTime *min_t_start, GstClockTime *max_t_end)
{
	return get_span(collectpads, min_t_start, max_t_end);
}


/**
 * Wrapper for framecpp_muxqueue_get_list().  Returns a list of buffers (in
 * the order in which they were received) containing samples taken from the
 * start of the queue upto (not including) the timestamp t_end.  The list
 * returned might be empty if the queue does not have data prior to the
 * requested time.  The data (if any) is flushed from the queue.
 *
 * It is an error to request a timestamp t_end beyond the end of the
 * currently en-queued data.
 *
 * The calling code owns the list returned by this function.  It should be
 * freed and the buffers unref()ed when no longer needed.
 *
 * This function should be called with the pads list lock held, e.g. from
 * within the "collected" signal handler.
 */


GList *framecpp_muxcollectpads_take_list(FrameCPPMuxCollectPadsData *data, GstClockTime t_end)
{
	GstClockTime queue_timestamp;
	GList *result = NULL;

	/*
	 * checks
	 */

	g_return_val_if_fail(data != NULL, NULL);

	/*
	 * retrieve the requested buffer list, flush the queue
	 */

	queue_timestamp = framecpp_muxqueue_timestamp(data->queue);
	if(GST_CLOCK_TIME_IS_VALID(queue_timestamp) && t_end > queue_timestamp) {
		result = framecpp_muxqueue_get_list(data->queue, t_end - queue_timestamp);
		framecpp_muxqueue_flush(data->queue, t_end - queue_timestamp);
	}

	/*
	 * done
	 */

	if(result)
		GST_DEBUG_OBJECT(GST_PAD_PARENT(data->pad), "(%s): taking %" GST_BUFFER_LIST_BOUNDARIES_FORMAT, GST_PAD_NAME(data->pad), GST_BUFFER_LIST_BOUNDARIES_ARGS(result));
	else
		GST_DEBUG_OBJECT(GST_PAD_PARENT(data->pad), "(%s): nothing available prior to %" GST_TIME_SECONDS_FORMAT, GST_PAD_NAME(data->pad), GST_TIME_SECONDS_ARGS(t_end));
	return result;
}


/**
 * Return the extent of the buffers in the buffer list, for example as
 * returned by framecpp_muxcollectpads_take_list().  The buffer list cannot
 * be empty.
 */


void framecpp_muxcollectpads_buffer_list_boundaries(GList *list, GstClockTime *t_start, GstClockTime *t_end)
{
	GstBuffer *last;

	g_assert(list != NULL);

	*t_start = GST_BUFFER_TIMESTAMP(GST_BUFFER(g_list_first(list)->data));
	last = GST_BUFFER(g_list_last(list)->data);
	*t_end = GST_BUFFER_TIMESTAMP(last) + GST_BUFFER_DURATION(last);

	g_assert_cmpuint(*t_start, <=, *t_end);
}


/**
 * Join contiguous buffers in the buffer list into single buffers.  Returns
 * the new list.
 */

/*
 * FIXME:  this is an easy implementation, but calling gst_buffer_merge()
 * for every pair of adjacent buffers results in many more malloc()s and
 * memcpy()s than required.  for example, if the entire list can be merged
 * into a single buffer then there should be a single malloc and each byte
 * of every buffer should be copied into the new region, and that's it.
 * but this implementation does a new malloc for every buffer, and then
 * recopies all previously copied bytes into the new buffer.
 */

GList *framecpp_muxcollectpads_buffer_list_join(GList *list, gboolean distinct_gaps)
{
	GList *this;

	for(this = list; this; this = g_list_next(this)) {
		GstBuffer *this_buf = GST_BUFFER(this->data);
		GList *next;

		while((next = g_list_next(this))) {
			GstBuffer *next_buf = GST_BUFFER(next->data);

			/* allow 1 ns of timestamp mismatch */
			if(llabs(GST_CLOCK_DIFF(GST_BUFFER_TIMESTAMP(this_buf) + GST_BUFFER_DURATION(this_buf), GST_BUFFER_TIMESTAMP(next_buf))) > 1)
				break;

			/* if distinct_gaps == TRUE, can't merge gaps with
			 * non-gaps */
			if(distinct_gaps && GST_BUFFER_FLAG_IS_SET(this_buf, GST_BUFFER_FLAG_GAP) != GST_BUFFER_FLAG_IS_SET(next_buf, GST_BUFFER_FLAG_GAP))
				break;

			list = g_list_delete_link(list, next);

			/* _merge() and _join() do not copy caps and flags,
			 * so we have to do it ourselves, so we have to use
			 * _merge() to keep the source buffers around */
			this->data = gst_buffer_make_metadata_writable(gst_buffer_merge(this_buf, next_buf));
			gst_buffer_copy_metadata(this->data, this_buf, GST_BUFFER_COPY_FLAGS | GST_BUFFER_COPY_CAPS);
			if(!GST_BUFFER_FLAG_IS_SET(this_buf, GST_BUFFER_FLAG_GAP) || !GST_BUFFER_FLAG_IS_SET(next_buf, GST_BUFFER_FLAG_GAP))
				GST_BUFFER_FLAG_UNSET(this->data, GST_BUFFER_FLAG_GAP);
			gst_buffer_unref(this_buf);
			gst_buffer_unref(next_buf);
			this_buf = this->data;
		}
	}

	return list;
}


/*
 * ============================================================================
 *
 *                              GObject Methods
 *
 * ============================================================================
 */


enum property {
	ARG_MAX_SIZE_TIME = 1
};


static void set_property(GObject *object, guint id, const GValue *value, GParamSpec *pspec)
{
	FrameCPPMuxCollectPads *collectpads = FRAMECPP_MUXCOLLECTPADS(object);

	GST_OBJECT_LOCK(collectpads);

	switch(id) {
	case ARG_MAX_SIZE_TIME: {
		GSList *pad_list;
		collectpads->max_size_time = g_value_get_uint64(value);
		FRAMECPP_MUXCOLLECTPADS_PADS_LOCK(collectpads);
		for(pad_list = collectpads->pad_list; pad_list; pad_list = g_slist_next(pad_list)) {
			FrameCPPMuxCollectPadsData *data = (FrameCPPMuxCollectPadsData *) pad_list->data;
			g_object_set(data->queue, "max-size-time", collectpads->max_size_time, NULL);
		}
		FRAMECPP_MUXCOLLECTPADS_PADS_UNLOCK(collectpads);
		break;
	}

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(collectpads);
}


static void get_property(GObject *object, guint id, GValue *value, GParamSpec *pspec)
{
	FrameCPPMuxCollectPads *collectpads = FRAMECPP_MUXCOLLECTPADS(object);

	GST_OBJECT_LOCK(collectpads);

	switch(id) {
	case ARG_MAX_SIZE_TIME:
		g_value_set_uint64(value, collectpads->max_size_time);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(collectpads);
}


static void dispose(GObject *object)
{
	FrameCPPMuxCollectPads *collectpads = FRAMECPP_MUXCOLLECTPADS(object);

/* FIXME:  this segfaults.  why?
	while(collectpads->pad_list)
		framecpp_muxcollectpads_remove_pad(collectpads, ((FrameCPPMuxCollectPadsData *) collectpads->pad_list->data)->pad);
*/
	collectpads->pad_list = NULL;

	G_OBJECT_CLASS(parent_class)->dispose(object);
}


static void finalize(GObject *object)
{
	FrameCPPMuxCollectPads *collectpads = FRAMECPP_MUXCOLLECTPADS(object);

	g_mutex_free(collectpads->pad_list_lock);
	collectpads->pad_list_lock = NULL;

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


static void framecpp_muxcollectpads_base_init(gpointer klass)
{
	/* no-op */
}


static void framecpp_muxcollectpads_class_init(FrameCPPMuxCollectPadsClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->dispose = GST_DEBUG_FUNCPTR(dispose);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	g_object_class_install_property(
		gobject_class,
		ARG_MAX_SIZE_TIME,
		g_param_spec_uint64(
			"max-size-time",
			"Maximum enqueued time",
			"Maximum time in nanoseconds to be buffered on each input queue.",
			0, G_MAXUINT64, DEFAULT_MAX_SIZE_TIME,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);

	signals[SIGNAL_COLLECTED] = g_signal_new(
		"collected",
		G_TYPE_FROM_CLASS(klass),
		G_SIGNAL_RUN_LAST,	/* CLEANUP instead? */
		G_STRUCT_OFFSET(
			FrameCPPMuxCollectPadsClass,
			collected
		),
		NULL,
		NULL,
		framecpp_marshal_VOID__CLOCK_TIME__CLOCK_TIME,
		G_TYPE_NONE,
		2,
		G_TYPE_UINT64,
		G_TYPE_UINT64
	);
}


static void framecpp_muxcollectpads_init(FrameCPPMuxCollectPads *collectpads, FrameCPPMuxCollectPadsClass *klass)
{
	collectpads->pad_list_lock = g_mutex_new();
	collectpads->pad_list = NULL;
	gst_segment_init(&collectpads->segment, GST_FORMAT_UNDEFINED);
	collectpads->started = FALSE;
	collectpads->min_t_start = GST_CLOCK_TIME_NONE;
	collectpads->min_t_end = GST_CLOCK_TIME_NONE;
}

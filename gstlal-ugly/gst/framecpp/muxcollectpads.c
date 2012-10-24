/*
 * FrameCPPMuxCollectPads
 *
 * Copyright (C) 2012  Kipp Cannon
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
 *                             Internal Functions
 *
 * ============================================================================
 */


static gboolean get_queue_segment(FrameCPPMuxCollectPadsData *data, GstClockTime *t_start, GstClockTime *t_end)
{
	*t_start = framecpp_muxqueue_timestamp(data->queue);
	*t_end = *t_start + framecpp_muxqueue_duration(data->queue);

	/*
	 * require a valid start offset and timestamp
	 */

	if(!GST_CLOCK_TIME_IS_VALID(*t_start) || !GST_CLOCK_TIME_IS_VALID(*t_end) || *t_start > *t_end) {
		GST_ERROR_OBJECT(data->collect, "%" GST_PTR_FORMAT ": %" GST_PTR_FORMAT " does not have a valid timestamp and/or duration", data->pad, data->queue);
		*t_start = *t_end = GST_CLOCK_TIME_NONE;
		return FALSE;
	}

	return TRUE;
}


static GstFlowReturn chain(GstPad *pad, GstBuffer *buffer)
{
	FrameCPPMuxCollectPadsData *data = gst_pad_get_element_private(pad);

	return !data->eos && framecpp_muxqueue_push(data->queue, buffer) ? GST_FLOW_OK : GST_FLOW_UNEXPECTED;
}



static gboolean all_pads_have_new_segments(FrameCPPMuxCollectPads *collectpads)
{
	GSList *collectdatalist;

	for(collectdatalist = collectpads->pad_list; collectdatalist; collectdatalist = g_slist_next(collectdatalist))
		if(!((FrameCPPMuxCollectPadsData *) collectdatalist->data)->new_segment)
			return FALSE;
	return TRUE;
}


static gboolean all_pads_are_at_eos(FrameCPPMuxCollectPads *collectpads)
{
	GSList *collectdatalist;

	for(collectdatalist = collectpads->pad_list; collectdatalist; collectdatalist = g_slist_next(collectdatalist))
		if(!((FrameCPPMuxCollectPadsData *) collectdatalist->data)->eos)
			return FALSE;
	return TRUE;
}


static gboolean update_segment(FrameCPPMuxCollectPads *collectpads)
{
	GSList *collectdatalist = collectpads->pad_list;
	FrameCPPMuxCollectPadsData *data;

	/* assume there's at least one pad */
	g_assert(collectdatalist != NULL);

	/*
	 * start by copying the segment from the first collect pad
	 */

	data = collectdatalist->data;
	collectpads->segment = data->segment;	/* !?  how is this supposed to be done? */

	for(collectdatalist = g_slist_next(collectdatalist); collectdatalist; collectdatalist = g_slist_next(collectdatalist)) {
		data = collectdatalist->data;

		/*
		 * check for format/rate mismatch
		 */

		if(collectpads->segment.format != data->segment.format || collectpads->segment.applied_rate != data->segment.applied_rate) {
			GST_ERROR_OBJECT(collectpads, "%" GST_PTR_FORMAT ": mismatch in segment format and/or applied rate", data->pad);
			return FALSE;
		}

		/*
		 * expand start and stop
		 */

		GST_DEBUG_OBJECT(collectpads, "%" GST_PTR_FORMAT ": have segment [%" G_GUINT64_FORMAT ", %" G_GUINT64_FORMAT ")", data->pad, data->segment.start, data->segment.stop);
		if(collectpads->segment.start == -1 || collectpads->segment.start > data->segment.start)
			collectpads->segment.start = data->segment.start;
		if(collectpads->segment.stop == -1 || collectpads->segment.stop < data->segment.stop)
			collectpads->segment.stop = data->segment.stop;
	}

	/*
	 * success --> clear new_segment flags
	 */

	for(collectdatalist = collectpads->pad_list; collectdatalist; collectdatalist = g_slist_next(collectdatalist))
		((FrameCPPMuxCollectPadsData *) collectdatalist->data)->new_segment = FALSE;

	return TRUE;
}


static gboolean event(GstPad *pad, GstEvent *event)
{
	FrameCPPMuxCollectPadsData *data = gst_pad_get_element_private(pad);
	FrameCPPMuxCollectPads *collectpads = data->collect;
	GstPadEventFunction event_func = data->event_func;
	gboolean success = TRUE;

	g_assert(GST_IS_FRAMECPP_MUXCOLLECTPADS(collectpads));

	switch(GST_EVENT_TYPE(event)) {
	case GST_EVENT_NEWSEGMENT: {
		gboolean update;
		gdouble rate, applied_rate;
		GstFormat format;
		gint64 start, stop, position;
		gst_event_parse_new_segment_full(event, &update, &rate, &applied_rate, &format, &start, &stop, &position);
		gst_event_unref(event);

		FRAMECPP_MUXCOLLECTPADS_PADS_LOCK(collectpads);
		gst_segment_set_newsegment_full(&data->segment, update, rate, applied_rate, format, start, stop, position);
		data->new_segment = TRUE;
		data->eos = FALSE;
		if(all_pads_have_new_segments(collectpads)) {
			success &= update_segment(collectpads);
			if(success && event_func)
				success &= event_func(pad, gst_event_new_new_segment_full(FALSE, collectpads->segment.rate, collectpads->segment.applied_rate, collectpads->segment.format, collectpads->segment.start, collectpads->segment.stop, collectpads->segment.time));
		}
		FRAMECPP_MUXCOLLECTPADS_PADS_UNLOCK(collectpads);
		break;
	}

	case GST_EVENT_FLUSH_START:
		framecpp_muxqueue_set_flushing(data->queue, TRUE);
		if(event_func)
			success &= event_func(pad, event);
		else
			gst_event_unref(event);
		break;

	case GST_EVENT_FLUSH_STOP:
		framecpp_muxqueue_set_flushing(data->queue, FALSE);
		if(event_func)
			success &= event_func(pad, event);
		else
			gst_event_unref(event);
		break;

	case GST_EVENT_EOS:
		data->eos = TRUE;
		if(all_pads_are_at_eos(collectpads) && event_func)
			success &= event_func(pad, event);
		else
			gst_event_unref(event);
		break;

	default:
		if(event_func)
			success &= event_func(pad, event);
		else
			gst_event_unref(event);
		break;
	}

	return success;
}


static void waiting_handler(FrameCPPMuxQueue *queue, gpointer user_data)
{
	FrameCPPMuxCollectPads *collectpads = ((FrameCPPMuxCollectPadsData *) user_data)->collect;
	GSList *collectdatalist;
	GstClockTime available_time = GST_CLOCK_TIME_NONE;

	g_assert(GST_IS_FRAMECPP_MUXCOLLECTPADS(collectpads));

	/* FIXME:  if this fails, there needs to be soem way of erroring out */

	FRAMECPP_MUXCOLLECTPADS_PADS_LOCK(collectpads);
	for(collectdatalist = collectpads->pad_list; collectdatalist; collectdatalist = g_slist_next(collectdatalist)) {
		FrameCPPMuxCollectPadsData *data = collectdatalist->data;
		GstClockTime queue_t_start, queue_t_end;

		if(g_queue_is_empty(GST_AUDIOADAPTER(data->queue)->queue) || !get_queue_segment(data, &queue_t_start, &queue_t_end)) {
			FRAMECPP_MUXCOLLECTPADS_PADS_UNLOCK(collectpads);
			goto done;
		}

		GST_DEBUG_OBJECT(collectpads, "%" GST_PTR_FORMAT ": queue spans [%" GST_TIME_SECONDS_FORMAT ", %" GST_TIME_SECONDS_FORMAT ")", data->pad, GST_TIME_SECONDS_ARGS(queue_t_start), GST_TIME_SECONDS_ARGS(queue_t_end));

		available_time = MIN(available_time, queue_t_end - queue_t_start);
	}
	FRAMECPP_MUXCOLLECTPADS_PADS_UNLOCK(collectpads);

	g_assert(GST_CLOCK_TIME_IS_VALID(available_time));
	g_assert_cmpuint(available_time, >=, collectpads->max_size_time);
	GST_DEBUG_OBJECT(collectpads, "%" GST_TIME_SECONDS_FORMAT " available", GST_TIME_SECONDS_ARGS(available_time));

	collectpads->available_time = available_time;
	g_object_notify(G_OBJECT(collectpads), "available-time");

done:
	return;
}


/*
 * ============================================================================
 *
 *                                Exported API
 *
 * ============================================================================
 */


/**
 * Add a pad to the collect pads
 */


FrameCPPMuxCollectPadsData *framecpp_muxcollectpads_add_pad(FrameCPPMuxCollectPads *collectpads, GstPad *pad)
{
	FrameCPPMuxCollectPadsData *data = g_malloc0(sizeof(*data));

	FRAMECPP_MUXCOLLECTPADS_PADS_LOCK(collectpads);

	data->collect = collectpads;
	data->pad = gst_object_ref(pad);
	data->queue = FRAMECPP_MUXQUEUE(g_object_new(FRAMECPP_MUXQUEUE_TYPE, NULL));
	g_object_set(data->queue, "max-size-time", collectpads->max_size_time, NULL);
	gst_segment_init(&data->segment, GST_FORMAT_UNDEFINED);
	data->new_segment = FALSE;
	data->eos = FALSE;

	GST_OBJECT_LOCK(pad);
	gst_pad_set_element_private(pad, data);
	gst_pad_set_chain_function(pad, chain);
	gst_pad_set_event_function(pad, event);
	GST_OBJECT_UNLOCK(pad);

	collectpads->pad_list = g_slist_append(collectpads->pad_list, data);
	data->waiting_handler_id = g_signal_connect(data->queue, "waiting", G_CALLBACK(waiting_handler), data);

	FRAMECPP_MUXCOLLECTPADS_PADS_UNLOCK(collectpads);

	return data;
}


/**
 * Remove a pad from the collect pads
 */


gboolean framecpp_muxcollectpads_remove_pad(FrameCPPMuxCollectPads *collectpads, GstPad *pad)
{
	FrameCPPMuxCollectPadsData *data;
	gboolean success = TRUE;

	FRAMECPP_MUXCOLLECTPADS_PADS_LOCK(collectpads);

	GST_OBJECT_LOCK(pad);
	data = gst_pad_get_element_private(pad);
	gst_pad_set_element_private(pad, NULL);
	GST_OBJECT_UNLOCK(pad);

	collectpads->pad_list = g_slist_remove(collectpads->pad_list, data);

	gst_object_unref(data->pad);
	data->pad = NULL;

	g_signal_handler_disconnect(data->queue, data->waiting_handler_id);
	gst_object_unref(data->queue);
	data->queue = NULL;

	g_free(data);

	FRAMECPP_MUXCOLLECTPADS_PADS_UNLOCK(collectpads);

	return success;
}


/**
 * Set an event call-back for a pad
 */


void framecpp_muxcollectpads_set_event_function(FrameCPPMuxCollectPadsData *data, GstPadEventFunction func)
{
	data->event_func = func;
}


/**
 * Start the collect pads
 */


void framecpp_muxcollectpads_start(FrameCPPMuxCollectPads *collectpads)
{
	/* FIXME */
}


/**
 * Stop the collect pads.
 */


void framecpp_muxcollectpads_stop(FrameCPPMuxCollectPads *collectpads)
{
	/* FIXME */
}


gboolean framecpp_muxcollectpads_all_pads_are_at_eos(FrameCPPMuxCollectPads *collectpads)
{
	gboolean answer;
	FRAMECPP_MUXCOLLECTPADS_PADS_LOCK(collectpads);
	answer = all_pads_are_at_eos(collectpads);
	FRAMECPP_MUXCOLLECTPADS_PADS_UNLOCK(collectpads);
	return answer;
}


/**
 * Computes the earliest of the start and of the end times of the
 * FrameCPPMuxCollectPads's input queues.
 *
 * The return value indicates the successful execution of this function.
 * TRUE indicates the function was able to procede to a successful
 * conclusion, FALSE indicates that one or more errors occured.
 *
 * Upon the successful completion of this function, both time parameters
 * will be set to GST_CLOCK_TIME_NONE if all input streams are at EOS.
 * Otherwise, if at least one stream is not at EOS, the times are set to
 * the earliest interval spanned by all the buffers that are available.
 *
 * Note that if no input pads have data available, this condition is
 * interpreted as EOS.  EOS is, therefore, indistinguishable from the
 * initial state, wherein no data has yet arrived.  It is assumed this
 * function will only be invoked from within the collected() method, and
 * therefore only after at least one pad has received a buffer, and
 * therefore the "no data available" condition is only seen at EOS.
 *
 * Summary:
 *
 * condition   return value   times
 * ----------------------------------
 * bad input   FALSE          ?
 * EOS         TRUE           GST_CLOCK_TIME_NONE
 * success     TRUE           >= 0
 *
 * Should be called with the FrameCPPMuxCollectPads' lock held (i.e., from
 * the collected() method).
 */


gboolean framecpp_muxcollectpads_get_earliest_times(FrameCPPMuxCollectPads *collectpads, GstClockTime *t_start, GstClockTime *t_end)
{
	gboolean all_eos = TRUE;
	GSList *collectdatalist;

	/*
	 * initilize
	 */

	g_return_val_if_fail(t_start != NULL, FALSE);
	g_return_val_if_fail(t_end != NULL, FALSE);

	*t_start = *t_end = G_MAXUINT64;

	g_return_val_if_fail(collectpads != NULL, FALSE);
	g_return_val_if_fail(GST_IS_FRAMECPP_MUXCOLLECTPADS(collectpads), FALSE);

	/*
	 * loop over sink pads
	 */

	for(collectdatalist = collectpads->pad_list; collectdatalist; collectdatalist = g_slist_next(collectdatalist)) {
		FrameCPPMuxCollectPadsData *data = collectdatalist->data;
		GstClockTime queue_t_start, queue_t_end;

		/*
		 * check for EOS
		 */

		if(data->eos) {
			GST_DEBUG_OBJECT(collectpads, "%" GST_PTR_FORMAT ": EOS", data->pad);
			continue;
		}
		if(g_queue_is_empty(GST_AUDIOADAPTER(data->queue)->queue)) {
			GST_DEBUG_OBJECT(collectpads, "%" GST_PTR_FORMAT ": no data yet", data->pad);
			continue;
		}

		/*
		 * compute this queue's start and end times
		 */

		if(!get_queue_segment(data, &queue_t_start, &queue_t_end))
			return FALSE;

		GST_DEBUG_OBJECT(collectpads, "%" GST_PTR_FORMAT ": queue spans [%" GST_TIME_SECONDS_FORMAT ", %" GST_TIME_SECONDS_FORMAT ")", data->pad, GST_TIME_SECONDS_ARGS(queue_t_start), GST_TIME_SECONDS_ARGS(queue_t_end));

		/*
		 * update the minima
		 */

		if(queue_t_start < *t_start)
			*t_start = queue_t_start;
		if(queue_t_end < *t_end)
			*t_end = queue_t_end;

		/*
		 * with at least one valid pair of times, we can return
		 * meaningful numbers.
		 */

		all_eos = FALSE;
	}

	/*
	 * found at least one buffer?
	 */

	if(all_eos)
		*t_start = *t_end = GST_CLOCK_TIME_NONE;
	GST_DEBUG_OBJECT(collectpads, "earliest common spanned interval [%" GST_TIME_SECONDS_FORMAT ", %" GST_TIME_SECONDS_FORMAT ")", GST_TIME_SECONDS_ARGS(*t_start), GST_TIME_SECONDS_ARGS(*t_end));

	return TRUE;
}


/**
 * Wrapper for framecpp_muxqueue_get_list().  Returns a list of buffers
 * containing the samples taken from the start of the pad's queue upto (not
 * including) the offset corresponding to t_end.  The list returned might
 * be shorter if the pad does not have data upto the requested time.  The
 * list returned by this function has its offset and offset_end set to
 * indicate its location in the input stream.  Calling this function has
 * the effect of flushing the queue upto the offset corresponding to t_end
 * or the upper bound of the available data, whichever comes first.
 *
 * If the pad has no data available then NULL is returned, this indicates
 * EOS.  If the pad has data available but it is subsequent to the
 * requested interval then a list containing a zero-length buffer is
 * returned.
 *
 * Should be called with the FrameCPPMuxCollectPads' lock held (i.e., from
 * the collected() method).
 */


GList *framecpp_muxcollectpads_take_list(FrameCPPMuxCollectPadsData *data, GstClockTime t_end)
{
	FrameCPPMuxCollectPads *collectpads;
	GList *result;

	/*
	 * checks
	 */

	g_return_val_if_fail(data != NULL, NULL);
	collectpads = data->collect;
	g_return_val_if_fail(collectpads != NULL, NULL);
	g_return_val_if_fail(GST_IS_FRAMECPP_MUXCOLLECTPADS(collectpads), FALSE);

	/*
	 * retrieve the requested buffer list, flush the queue
	 */

	result = framecpp_muxqueue_get_list(data->queue, t_end);
	framecpp_muxqueue_flush(data->queue, t_end);

	/*
	 * done
	 */

	{
	GstBuffer *first = GST_BUFFER(g_list_first(result)->data);
	GstBuffer *last = GST_BUFFER(g_list_last(result)->data);
	GstBuffer span;
	GST_BUFFER_TIMESTAMP(&span) = GST_BUFFER_TIMESTAMP(first);
	GST_BUFFER_DURATION(&span) = GST_BUFFER_TIMESTAMP(last) + GST_BUFFER_DURATION(last) - GST_BUFFER_TIMESTAMP(&span);
	GST_BUFFER_OFFSET(&span) = GST_BUFFER_OFFSET(first);
	GST_BUFFER_OFFSET_END(&span) = GST_BUFFER_OFFSET_END(last);
	GST_DEBUG_OBJECT(GST_PAD_PARENT(data->pad), "(%s): returning %" GST_BUFFER_BOUNDARIES_FORMAT, GST_PAD_NAME(data->pad), GST_BUFFER_BOUNDARIES_ARGS(&span));
	}

	return result;
}


/*
 * ============================================================================
 *
 *                              GObject Methods
 *
 * ============================================================================
 */


enum property {
	ARG_MAX_SIZE_TIME = 1,
	ARG_AVAILABLE_TIME
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
		for(pad_list = collectpads->pad_list; pad_list; pad_list = g_slist_next(pad_list))
			g_object_set(FRAMECPP_MUXQUEUE(pad_list->data), "max-size-time", collectpads->max_size_time, NULL);
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

	case ARG_AVAILABLE_TIME:
		g_value_set_uint64(value, collectpads->available_time);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(collectpads);
}


static void dispose(GObject *object)
{
	G_OBJECT_CLASS(parent_class)->dispose(object);
}


static void finalize(GObject *object)
{
	FrameCPPMuxCollectPads *collectpads = FRAMECPP_MUXCOLLECTPADS(object);

	while(collectpads->pad_list)
		framecpp_muxcollectpads_remove_pad(collectpads, ((FrameCPPMuxCollectPadsData *) collectpads->pad_list->data)->pad);
	collectpads->pad_list = NULL;
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

	gobject_class->set_property = set_property;
	gobject_class->get_property = get_property;
	gobject_class->dispose = dispose;
	gobject_class->finalize = finalize;

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
	g_object_class_install_property(
		gobject_class,
		ARG_AVAILABLE_TIME,
		g_param_spec_uint64(
			"available-time",
			"Available time",
			"Duration in nanoseconds of intersection of segments currently enqueued on all input streams.",
			0, G_MAXUINT64, 0,
			G_PARAM_READABLE | G_PARAM_STATIC_STRINGS
		)
	);
}


static void framecpp_muxcollectpads_init(FrameCPPMuxCollectPads *collectpads, FrameCPPMuxCollectPadsClass *klass)
{
	collectpads->pad_list_lock = g_mutex_new();
	collectpads->pad_list = NULL;
	gst_segment_init(&collectpads->segment, GST_FORMAT_UNDEFINED);
	collectpads->available_time = 0;
}

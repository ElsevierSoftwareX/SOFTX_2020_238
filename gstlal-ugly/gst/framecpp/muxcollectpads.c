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

	if(data->eos || data->segment.format == GST_FORMAT_UNDEFINED)
		return GST_FLOW_UNEXPECTED;
	return framecpp_muxqueue_push(data->queue, buffer) ? GST_FLOW_OK : GST_FLOW_ERROR;
}



static gboolean all_pads_have_segments(FrameCPPMuxCollectPads *collectpads)
{
	GSList *collectdatalist;

	for(collectdatalist = collectpads->pad_list; collectdatalist; collectdatalist = g_slist_next(collectdatalist))
		if(((FrameCPPMuxCollectPadsData *) collectdatalist->data)->segment.format == GST_FORMAT_UNDEFINED)
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
	GSList *collectdatalist;

	/*
	 * clear the segment boundaries
	 */

	gst_segment_set_newsegment_full(&collectpads->segment, FALSE, 1.0, 1.0, GST_FORMAT_UNDEFINED, -1, -1, -1);

	/*
	 * loop over pads
	 */

	for(collectdatalist = collectpads->pad_list; collectdatalist; collectdatalist = g_slist_next(collectdatalist)) {
		FrameCPPMuxCollectPadsData *data = collectdatalist->data;

		/*
		 * ignore pads whose segments aren't known
		 */

		if(data->segment.format == GST_FORMAT_UNDEFINED || data->segment.start == -1 || data->segment.stop == -1) {
			GST_DEBUG_OBJECT(collectpads, "%" GST_PTR_FORMAT ": segment not known", data->pad);
			continue;
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
			return FALSE;
		}

		/*
		 * expand start and stop
		 */

		GST_DEBUG_OBJECT(collectpads, "%" GST_PTR_FORMAT ": have segment [%" G_GUINT64_FORMAT ", %" G_GUINT64_FORMAT ")", data->pad, data->segment.start, data->segment.stop);
		if(collectpads->segment.start > data->segment.start)
			collectpads->segment.start = data->segment.start;
		if(collectpads->segment.stop < data->segment.stop)
			collectpads->segment.stop = data->segment.stop;
	}

	/*
	 * success?
	 */

	if(collectpads->segment.format == GST_FORMAT_UNDEFINED) {
		GST_ERROR_OBJECT(collectpads, "failed to compute union of input segments");
		return FALSE;
	}
	return TRUE;
}


static void clear_segments(FrameCPPMuxCollectPads *collectpads, FrameCPPMuxCollectPadsData *except)
{
	GSList *collectdatalist;

	for(collectdatalist = collectpads->pad_list; collectdatalist; collectdatalist = g_slist_next(collectdatalist)) {
		FrameCPPMuxCollectPadsData *data = collectdatalist->data;
		if(data != except)
			data->segment.format = GST_FORMAT_UNDEFINED;
	}
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
		data->eos = FALSE;
		success &= update_segment(collectpads);
		if(!update)
			clear_segments(collectpads, data);
		if(success && all_pads_have_segments(collectpads)) {
			if(event_func)
				success &= event_func(pad, gst_event_new_new_segment_full(update, collectpads->segment.rate, collectpads->segment.applied_rate, collectpads->segment.format, collectpads->segment.start, collectpads->segment.stop, collectpads->segment.time));
		}
		FRAMECPP_MUXCOLLECTPADS_PADS_UNLOCK(collectpads);
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
		data->segment.format = GST_FORMAT_UNDEFINED;
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


static void waiting_handler(FrameCPPMuxQueue *queue, GstClockTime t_start, GstClockTime t_end, gpointer user_data)
{
	FrameCPPMuxCollectPads *collectpads = ((FrameCPPMuxCollectPadsData *) user_data)->collect;
	GSList *collectdatalist;
	GstClockTime min_t_start = GST_CLOCK_TIME_NONE;
	GstClockTime min_t_end = GST_CLOCK_TIME_NONE;
	GstFlowReturn result = GST_FLOW_OK;

	g_assert(GST_IS_FRAMECPP_MUXCOLLECTPADS(collectpads));

	FRAMECPP_MUXCOLLECTPADS_PADS_LOCK(collectpads);
	for(collectdatalist = collectpads->pad_list; collectdatalist; collectdatalist = g_slist_next(collectdatalist)) {
		FrameCPPMuxCollectPadsData *data = collectdatalist->data;
		GstClockTime t_start, t_end;

		if(gst_audioadapter_is_empty(GST_AUDIOADAPTER(data->queue)) || !get_queue_segment(data, &t_start, &t_end)) {
			FRAMECPP_MUXCOLLECTPADS_PADS_UNLOCK(collectpads);
			goto done;
		}

		GST_DEBUG_OBJECT(collectpads, "%" GST_PTR_FORMAT ": queue spans [%" GST_TIME_SECONDS_FORMAT ", %" GST_TIME_SECONDS_FORMAT ")", data->pad, GST_TIME_SECONDS_ARGS(t_start), GST_TIME_SECONDS_ARGS(t_end));

		min_t_start = MIN(min_t_start, t_start);
		min_t_end = MIN(min_t_end, t_end);
	}
	FRAMECPP_MUXCOLLECTPADS_PADS_UNLOCK(collectpads);

	g_assert(GST_CLOCK_TIME_IS_VALID(min_t_start));
	g_assert(GST_CLOCK_TIME_IS_VALID(min_t_end));
	if(min_t_end - min_t_start >= collectpads->max_size_time) {
		GST_DEBUG_OBJECT(collectpads, "[%" GST_TIME_SECONDS_FORMAT ", %" GST_TIME_SECONDS_FORMAT ") available", GST_TIME_SECONDS_ARGS(min_t_start), GST_TIME_SECONDS_ARGS(min_t_end));

		g_signal_emit_by_name(collectpads, "collected", min_t_start, min_t_end, &result);
		/* FIXME:  do something with the result */
	}

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
 * Add a pad to the collect pads.  This is the only way to allocate a new
 * FrameCPPMuxCollectPadsData structure.  The calling code does not own the
 * FrameCPPMuxCollectPadsData structure, it is owned by the
 * FrameCPPMuxCollectPads object for which it has been allocated.
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
 * Remove a pad from the collect pads.  This is the only way to free a
 * FrameCPPMuxCollectPadsData structure.
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
 * Set an event function for a pad.  An event handler will be installed on
 * the pad to do work required by the FrameCPPMuxCollectPads object, and
 * that event handler will chain to the event handler set using this
 * function.
 *
 * Newsegment and EOS events are intercepted, and only chained to the
 * handler set using this function when all pads on the
 * FrameCPPMuxCollectPads have received newsegments or EOS events,
 * respectively.  That is, when the user-supplied event function sees an
 * EOS event then all sink pads are at EOS and the element should respond
 * appropriately.  Until then, at least one pad is not yet at EOS.
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
	if(t_end > queue_timestamp) {
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


static gboolean collected_accumulator(GSignalInvocationHint *ihint, GValue *accu_return, const GValue *handler_return, gpointer data)
{
	g_value_copy(handler_return, accu_return);

	return (GstFlowReturn) g_value_get_enum(handler_return) == GST_FLOW_OK;
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

	signals[SIGNAL_COLLECTED] = g_signal_new(
		"collected",
		G_TYPE_FROM_CLASS(klass),
		G_SIGNAL_RUN_LAST,	/* CLEANUP instead? */
		G_STRUCT_OFFSET(
			FrameCPPMuxCollectPadsClass,
			collected
		),
		collected_accumulator,
		NULL,
		framecpp_marshal_FLOW_RETURN__CLOCK_TIME__CLOCK_TIME,
		GST_TYPE_FLOW_RETURN,
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
}

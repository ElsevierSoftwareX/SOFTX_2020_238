/*
 * FrameCPPMuxQueue
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


#include <gstlal/gstaudioadapter.h>
#include <muxqueue.h>


/*
 * ============================================================================
 *
 *                                Boilerplate
 *
 * ============================================================================
 */


GST_BOILERPLATE(FrameCPPMuxQueue, framecpp_muxqueue, GstAudioAdapter, GST_TYPE_AUDIOADAPTER);


/*
 * ============================================================================
 *
 *                             Internal Functions
 *
 * ============================================================================
 */


static GstClockTime _framecpp_muxqueue_timestamp(FrameCPPMuxQueue *queue)
{
	GstAudioAdapter *adapter = GST_AUDIOADAPTER(queue);
	GstBuffer *buf = GST_BUFFER(g_queue_peek_head(adapter->queue));

	g_assert(GST_BUFFER_TIMESTAMP_IS_VALID(buf));

	return GST_BUFFER_TIMESTAMP(buf) + gst_util_uint64_scale_int_round(adapter->skip, GST_SECOND, queue->rate);
}


static GstClockTime _framecpp_muxqueue_duration(FrameCPPMuxQueue *queue)
{
	GstAudioAdapter *adapter = GST_AUDIOADAPTER(queue);
	GstBuffer *head, *tail;
	GstClockTimeDiff duration;

	if(g_queue_is_empty(adapter->queue))
		return 0;

	/* the first buffer is the head, the last is the tail */
	head = GST_BUFFER(g_queue_peek_head(adapter->queue));
	tail = GST_BUFFER(g_queue_peek_tail(adapter->queue));

	duration = GST_CLOCK_DIFF(GST_BUFFER_TIMESTAMP(head), GST_BUFFER_TIMESTAMP(tail) + GST_BUFFER_DURATION(tail)) - gst_util_uint64_scale_int_round(adapter->skip, GST_SECOND, queue->rate);
	g_assert(duration >= 0);

	return duration;
}


/*
 * ============================================================================
 *
 *                                Exported API
 *
 * ============================================================================
 */


GstClockTime framecpp_muxqueue_timestamp(FrameCPPMuxQueue *queue)
{
	GstAudioAdapter *adapter = GST_AUDIOADAPTER(queue);
	GstClockTime timestamp;

	FRAMECPP_MUXQUEUE_LOCK(queue);
	timestamp = g_queue_is_empty(adapter->queue) ? GST_CLOCK_TIME_NONE : _framecpp_muxqueue_timestamp(queue);
	FRAMECPP_MUXQUEUE_UNLOCK(queue);

	return timestamp;
}


GstClockTime framecpp_muxqueue_duration(FrameCPPMuxQueue *queue)
{
	GstClockTime duration;

	FRAMECPP_MUXQUEUE_LOCK(queue);
	duration = _framecpp_muxqueue_duration(queue);
	FRAMECPP_MUXQUEUE_UNLOCK(queue);

	return duration;
}


gboolean framecpp_muxqueue_push(FrameCPPMuxQueue *queue, GstBuffer *buf)
{
	GstAudioAdapter *adapter = GST_AUDIOADAPTER(queue);
	gboolean success = TRUE;

	g_assert(GST_BUFFER_TIMESTAMP_IS_VALID(buf));
	g_assert(GST_BUFFER_DURATION_IS_VALID(buf));
	g_assert_cmpuint(gst_util_uint64_scale_int_round(GST_BUFFER_DURATION(buf), queue->rate, GST_SECOND), ==, GST_BUFFER_OFFSET_END(buf) - GST_BUFFER_OFFSET(buf));

	FRAMECPP_MUXQUEUE_LOCK(queue);
	while(queue->max_size_time && !queue->flushing && _framecpp_muxqueue_duration(queue) >= queue->max_size_time)
		g_cond_wait(&queue->activity, &queue->lock);
	if(!queue->flushing) {
		gst_audioadapter_push(adapter, buf);
		g_cond_broadcast(&queue->activity);
	} else
		success = FALSE;
	FRAMECPP_MUXQUEUE_UNLOCK(queue);

	return success;
}


void framecpp_muxqueue_flush(FrameCPPMuxQueue *queue, GstClockTime time)
{
	GstAudioAdapter *adapter = GST_AUDIOADAPTER(queue);

	FRAMECPP_MUXQUEUE_LOCK(queue);
	gst_audioadapter_flush(adapter, gst_util_uint64_scale_int_round(time, queue->rate, GST_SECOND));
	g_cond_broadcast(&queue->activity);
	FRAMECPP_MUXQUEUE_UNLOCK(queue);
}


void framecpp_muxqueue_set_flushing(FrameCPPMuxQueue *queue, gboolean flushing)
{
	FRAMECPP_MUXQUEUE_LOCK(queue);
	queue->flushing = flushing;
	g_cond_broadcast(&queue->activity);
	FRAMECPP_MUXQUEUE_UNLOCK(queue);
}


gboolean framecpp_muxqueue_get_flushing(FrameCPPMuxQueue *queue)
{
	return queue->flushing;
}


GList *framecpp_muxqueue_get_list(FrameCPPMuxQueue *queue, GstClockTime time)
{
	GstAudioAdapter *adapter = GST_AUDIOADAPTER(queue);
	GstClockTime timestamp;
	GList *head;
	GList *result;

	FRAMECPP_MUXQUEUE_LOCK(queue);
	timestamp = g_queue_is_empty(adapter->queue) ? GST_CLOCK_TIME_NONE : _framecpp_muxqueue_timestamp(queue) + gst_util_uint64_scale_int_round(adapter->skip, GST_SECOND, queue->rate);
	result = gst_audioadapter_get_list(adapter, gst_util_uint64_scale_int_round(time, queue->rate, GST_SECOND));
	FRAMECPP_MUXQUEUE_UNLOCK(queue);

	if(result)
		/* adjust timestamp of first buffer */
		GST_BUFFER_TIMESTAMP(GST_BUFFER(result->data)) = timestamp;
	/* require all buffers in list to be contiguous */
	for(head = result; head && g_list_next(head); head = g_list_next(head)) {
		GstBuffer *this = GST_BUFFER(head->data);
		GstBuffer *next = GST_BUFFER(g_list_next(head)->data);
		g_assert_cmpuint(GST_BUFFER_TIMESTAMP(this) + GST_BUFFER_DURATION(this), ==, GST_BUFFER_TIMESTAMP(next));
	}
	if(head) {
		/* adjust duration of last buffer */
		GstBuffer *buf = GST_BUFFER(head->data);
		GST_BUFFER_DURATION(buf) = gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET_END(buf) - GST_BUFFER_OFFSET(buf), GST_SECOND, queue->rate);
	}

	return result;
}


/*
 * ============================================================================
 *
 *                                  Signals
 *
 * ============================================================================
 */


enum framecpp_muxqueue_signals {
	SIGNAL_WAITING,
	NUM_SIGNALS
};


static guint signals[NUM_SIGNALS] = {0, };


/*
 * ============================================================================
 *
 *                              GObject Methods
 *
 * ============================================================================
 */


enum property {
	PROP_RATE = 1,
	PROP_MAX_SIZE_TIME
};


static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec)
{
	FrameCPPMuxQueue *queue = FRAMECPP_MUXQUEUE(object);

	switch(id) {
	case PROP_RATE:
		queue->rate = g_value_get_int(value);
		break;

	case PROP_MAX_SIZE_TIME:
		queue->max_size_time = g_value_get_uint64(value);
		FRAMECPP_MUXQUEUE_LOCK(queue);
		g_cond_broadcast(&queue->activity);
		FRAMECPP_MUXQUEUE_UNLOCK(queue);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}
}


static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *pspec)
{
	FrameCPPMuxQueue *queue = FRAMECPP_MUXQUEUE(object);

	switch(id) {
	case PROP_RATE:
		g_value_set_int(value, queue->rate);
		break;

	case PROP_MAX_SIZE_TIME:
		g_value_set_uint64(value, queue->max_size_time);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}
}


static void dispose(GObject *object)
{
	framecpp_muxqueue_set_flushing(FRAMECPP_MUXQUEUE(object), TRUE);

	G_OBJECT_CLASS(parent_class)->dispose(object);
}


static void finalize(GObject *object)
{
	FrameCPPMuxQueue *queue = FRAMECPP_MUXQUEUE(object);

	g_mutex_clear(&queue->lock);
	g_cond_clear(&queue->activity);

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


static void framecpp_muxqueue_base_init(gpointer klass)
{
	/* no-op */
}


static void framecpp_muxqueue_class_init(FrameCPPMuxQueueClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

	gobject_class->set_property = set_property;
	gobject_class->get_property = get_property;
	gobject_class->dispose = dispose;
	gobject_class->finalize = finalize;

	g_object_class_install_property(
		gobject_class,
		PROP_RATE,
		g_param_spec_int(
			"rate",
			"Sample rate",
			"The sample rate in Hz.",
			0, G_MAXINT, 0,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		PROP_MAX_SIZE_TIME,
		g_param_spec_uint64(
			"max-size-time",
			"Max size time",
			"Max. amount of data in the queue in ns (0 = disable).",
			0, G_MAXUINT64, 0,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);

	signals[SIGNAL_WAITING] = g_signal_new(
		"waiting",
		G_TYPE_FROM_CLASS(klass),
		G_SIGNAL_RUN_FIRST,
		G_STRUCT_OFFSET(
			FrameCPPMuxQueueClass,
			waiting
		),
		NULL,
		NULL,
		g_cclosure_marshal_VOID__VOID,
		G_TYPE_NONE,	/* returns void */
		0	/* 0 parameters */
	);
}


static void framecpp_muxqueue_init(FrameCPPMuxQueue *queue, FrameCPPMuxQueueClass *klass)
{
	g_mutex_init(&queue->lock);
	g_cond_init(&queue->activity);
	queue->flushing = FALSE;
}

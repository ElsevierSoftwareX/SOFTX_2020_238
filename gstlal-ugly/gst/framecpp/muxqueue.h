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


#ifndef __FRAMECPP_MUXQUEUE_H__
#define __FRAMECPP_MUXQUEUE_H__


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


G_BEGIN_DECLS


/*
 * ============================================================================
 *
 *                                    Type
 *
 * ============================================================================
 */


#define FRAMECPP_MUXQUEUE_TYPE \
	(framecpp_muxqueue_get_type())
#define FRAMECPP_MUXQUEUE(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), FRAMECPP_MUXQUEUE_TYPE, FrameCPPMuxQueue))
#define FRAMECPP_MUXQUEUE_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), FRAMECPP_MUXQUEUE_TYPE, FrameCPPMuxQueueClass))
#define FRAMECPP_MUXQUEUE_GET_CLASS(obj) \
	(G_TYPE_INSTANCE_GET_CLASS((obj), FRAMECPP_MUXQUEUE_TYPE, FrameCPPMuxQueueClass))
#define GST_IS_FRAMECPP_MUXQUEUE(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), FRAMECPP_MUXQUEUE_TYPE))
#define GST_IS_FRAMECPP_MUXQUEUE_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), FRAMECPP_MUXQUEUE_TYPE))


typedef struct _FrameCPPMuxQueueClass FrameCPPMuxQueueClass;
typedef struct _FrameCPPMuxQueue FrameCPPMuxQueue;


struct _FrameCPPMuxQueueClass {
	GstAudioAdapterClass parent_class;

	void (*waiting)(FrameCPPMuxQueue *, gpointer);
};


/**
 * FrameCPPMuxQueue
 *
 * The opaque #FrameCPPMuxQueue data structure.
 */


struct _FrameCPPMuxQueue {
	GstAudioAdapter object;

	/*< private >*/
	GMutex *lock;
	GCond *activity;
	gboolean flushing;
	gint rate;
	GstClockTime max_size_time;
};


/*
 * ============================================================================
 *
 *                                Exported API
 *
 * ============================================================================
 */


#define FRAMECPP_MUXQUEUE_GETLOCK(queue) (queue->lock)
#define FRAMECPP_MUXQUEUE_LOCK(queue) g_mutex_lock(FRAMECPP_MUXQUEUE_GETLOCK(queue))
#define FRAMECPP_MUXQUEUE_UNLOCK(queue) g_mutex_unlock(FRAMECPP_MUXQUEUE_GETLOCK(queue))


GstClockTime framecpp_muxqueue_timestamp(FrameCPPMuxQueue *);
GstClockTime framecpp_muxqueue_duration(FrameCPPMuxQueue *);
GstFlowReturn framecpp_muxqueue_push(FrameCPPMuxQueue *, GstBuffer *);
void framecpp_muxqueue_flush(FrameCPPMuxQueue *, GstClockTime);
void framecpp_muxqueue_clear(FrameCPPMuxQueue *);
void framecpp_muxqueue_set_flushing(FrameCPPMuxQueue *, gboolean);
gboolean framecpp_muxqueue_get_flushing(FrameCPPMuxQueue *);
GList *framecpp_muxqueue_get_list(FrameCPPMuxQueue *, GstClockTime);


GType framecpp_muxqueue_get_type(void);


G_END_DECLS


#endif	/* __FRAMECPP_MUXQUEUE_H__ */

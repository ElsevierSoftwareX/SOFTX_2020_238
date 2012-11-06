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


#ifndef __FRAMECPP_MUXCOLLECTPADS_H__
#define __FRAMECPP_MUXCOLLECTPADS_H__


/*
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


#include <glib.h>
#include <gst/gst.h>


#include <muxqueue.h>


G_BEGIN_DECLS


/*
 * ============================================================================
 *
 *                                    Type
 *
 * ============================================================================
 */


#define FRAMECPP_MUXCOLLECTPADS_TYPE \
	(framecpp_muxcollectpads_get_type())
#define FRAMECPP_MUXCOLLECTPADS(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), FRAMECPP_MUXCOLLECTPADS_TYPE, FrameCPPMuxCollectPads))
#define FRAMECPP_MUXCOLLECTPADS_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), FRAMECPP_MUXCOLLECTPADS_TYPE, FrameCPPMuxCollectPadsClass))
#define FRAMECPP_MUXCOLLECTPADS_GET_CLASS(obj) \
	(G_TYPE_INSTANCE_GET_CLASS((obj), FRAMECPP_MUXCOLLECTPADS_TYPE, FrameCPPMuxCollectPadsClass))
#define GST_IS_FRAMECPP_MUXCOLLECTPADS(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), FRAMECPP_MUXCOLLECTPADS_TYPE))
#define GST_IS_FRAMECPP_MUXCOLLECTPADS_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), FRAMECPP_MUXCOLLECTPADS_TYPE))


typedef struct _FrameCPPMuxCollectPadsClass FrameCPPMuxCollectPadsClass;
typedef struct _FrameCPPMuxCollectPads FrameCPPMuxCollectPads;
typedef struct _FrameCPPMuxCollectPadsData FrameCPPMuxCollectPadsData;
typedef void (*FrameCPPMuxCollectPadsDataDestroyNotify) (FrameCPPMuxCollectPadsData *);


struct _FrameCPPMuxCollectPadsClass {
	GstObjectClass parent_class;

	void (*collected)(FrameCPPMuxCollectPads *, GstClockTime, GstClockTime, gpointer);
};


/**
 * FrameCPPMuxCollectPads
 *
 * The #FrameCPPMuxCollectPads data structure.
 */


struct _FrameCPPMuxCollectPads {
	GstObject object;

	GMutex *pad_list_lock;
	GSList *pad_list;

	GstSegment segment;

	/*< private >*/
	GstClockTime max_size_time;

	gboolean started;
	GstClockTime min_t_start;
	GstClockTime min_t_end;
};


/**
 * FrameCPPMuxCollectPadsData
 *
 * The #FrameCPPMuxCollectPadsData data structure.
 */


struct _FrameCPPMuxCollectPadsData {
	FrameCPPMuxCollectPads *collect;
	GstPad *pad;
	FrameCPPMuxQueue *queue;
	GstSegment segment;

	gpointer appdata;
	FrameCPPMuxCollectPadsDataDestroyNotify destroy_notify;

	/*< private >*/
	GstPadEventFunction event_func;
	gulong waiting_handler_id;
	gboolean eos;
};


/*
 * ============================================================================
 *
 *                                Exported API
 *
 * ============================================================================
 */


#define FRAMECPP_MUXCOLLECTPADS_PADS_GETLOCK(pads) (pads->pad_list_lock)
#define FRAMECPP_MUXCOLLECTPADS_PADS_LOCK(pads) g_mutex_lock(FRAMECPP_MUXCOLLECTPADS_PADS_GETLOCK(pads))
#define FRAMECPP_MUXCOLLECTPADS_PADS_UNLOCK(pads) g_mutex_unlock(FRAMECPP_MUXCOLLECTPADS_PADS_GETLOCK(pads))


FrameCPPMuxCollectPadsData *framecpp_muxcollectpads_add_pad(FrameCPPMuxCollectPads *, GstPad *, FrameCPPMuxCollectPadsDataDestroyNotify);
gboolean framecpp_muxcollectpads_remove_pad(FrameCPPMuxCollectPads *, GstPad *);
void framecpp_muxcollectpads_set_event_function(FrameCPPMuxCollectPadsData *, GstPadEventFunction);
void framecpp_muxcollectpads_set_flushing(FrameCPPMuxCollectPads *, gboolean);
void framecpp_muxcollectpads_start(FrameCPPMuxCollectPads *);
void framecpp_muxcollectpads_stop(FrameCPPMuxCollectPads *);
GList *framecpp_muxcollectpads_take_list(FrameCPPMuxCollectPadsData *, GstClockTime);
void framecpp_muxcollectpads_buffer_list_boundaries(GList *, GstClockTime *, GstClockTime *);


GType framecpp_muxcollectpads_get_type(void);


G_END_DECLS


#endif	/* __FRAMECPP_MUXCOLLECTPADS_H__ */

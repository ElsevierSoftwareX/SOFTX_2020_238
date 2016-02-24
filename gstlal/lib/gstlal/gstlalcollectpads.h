/*
 *
 * Copyright (C) 2008 Kipp Cannon <kipp.cannon@ligo.org>
 *
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Library General Public License as published
 * by the Free Software Foundation; either version 2 of the License, or (at
 * your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307,
 * USA.
 */


#ifndef __GSTLAL_COLLECTPADS_H__
#define __GSTLAL_COLLECTPADS_H__


#include <gst/gst.h>
#include <gst/base/gstcollectpads.h>


G_BEGIN_DECLS


/**
 * GstLALCollectData:
 * @as_gstcollectdata:  the parent structure
 * @unit_size:  size of one "unit", e.g. (multi-channel) audio sample,
 * video frame, etc.  For audio, = (sample width) / 8 * (channels).
 * @rate:  number of units per second
 */


typedef struct _GstLALCollectData {
	GstCollectData as_gstcollectdata;

	guint unit_size;
	gint rate;
} GstLALCollectData;


/*
 * Function prototypes.
 */


GstLALCollectData *gstlal_collect_pads_add_pad(GstCollectPads *pads, GstPad *pad, guint size, GstCollectDataDestroyNotify destroy_notify, gboolean lock);
gboolean gstlal_collect_pads_remove_pad(GstCollectPads *pads, GstPad *pad);
void gstlal_collect_pads_set_unit_size(GstPad *pad, guint unit_size);
guint gstlal_collect_pads_get_unit_size(GstPad *pad);
void gstlal_collect_pads_set_rate(GstPad *pad, gint rate);
gint gstlal_collect_pads_get_rate(GstPad *pad);
GstSegment *gstlal_collect_pads_get_segment(GstCollectPads *pads);
gboolean gstlal_collect_pads_get_earliest_times(GstCollectPads *pads, GstClockTime *t_start, GstClockTime *t_end);
GstBuffer *gstlal_collect_pads_take_buffer_sync(GstCollectPads *pads, GstLALCollectData *data, GstClockTime t_end);


G_END_DECLS


#endif	/* __GSTLAL_COLLECTPADS_H__ */

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
 * Custom GstCollectData structure with extra metadata required for
 * synchronous mixing of input streams.
 */


typedef struct _GstLALCollectData {
	/*
	 * parent structure first so we can be cast to it
	 */

	GstCollectData as_gstcollectdata;

	/*
	 * size of one "unit", e.g. (multi-channel) audio sample, video
	 * frame, etc.  For audio, = (sample width) / 8 * (channels).
	 */

	guint unit_size;

	/*
	 * number of units per second.
	 */

	gint rate;
} GstLALCollectData;


/*
 * Function prototypes.
 */


GstLALCollectData *gstlal_collect_pads_add_pad(GstCollectPads *, GstPad *, guint);
GstLALCollectData *gstlal_collect_pads_add_pad_full(GstCollectPads *, GstPad *, guint, GstCollectDataDestroyNotify);
gboolean gstlal_collect_pads_remove_pad(GstCollectPads *, GstPad *);
void gstlal_collect_pads_set_unit_size(GstPad *, guint);
guint gstlal_collect_pads_get_unit_size(GstPad *);
void gstlal_collect_pads_set_rate(GstPad *, gint);
gint gstlal_collect_pads_get_rate(GstPad *);
GstSegment *gstlal_collect_pads_get_segment(GstCollectPads *pads);
gboolean gstlal_collect_pads_get_earliest_times(GstCollectPads *, GstClockTime *, GstClockTime *);
GstBuffer *gstlal_collect_pads_take_buffer_sync(GstCollectPads *, GstLALCollectData *, GstClockTime);


G_END_DECLS


#endif	/* __GSTLAL_COLLECTPADS_H__ */

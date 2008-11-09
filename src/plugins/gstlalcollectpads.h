/*
 *
 * Copyright (C) 2008 Kipp Cannon <kcannon@ligo.caltech.edu>
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

	GstCollectData collectdata;

	/*
	 * event handler (chains to the original one)
	 */

	GstPadEventFunction collect_event_func;

	/*
	 * offset_offset is the difference between this input stream's
	 * offset counter and the adder's output stream's offset counter
	 * for the same timestamp in both streams: offset_offset =
	 * intput_offset - output_offset @ a common timestamp
	 */

	gboolean offset_offset_valid;
	gint64 offset_offset;
} GstLALCollectData;


/*
 * Function prototypes.
 */


GstLALCollectData *gstlal_collect_pads_add_pad(GstCollectPads *, GstPad *, guint);
gboolean gstlal_collect_pads_remove_pad(GstCollectPads *, GstPad *);
gboolean gstlal_collect_pads_get_earliest_offsets(GstCollectPads *, guint64 *, guint64 *, gint, gint, GstClockTime);
GstBuffer *gstlal_collect_pads_take_buffer(GstCollectPads *, GstLALCollectData *, guint64, gint64, size_t);


G_END_DECLS


#endif	/* __GSTLAL_COLLECTPADS_H__ */

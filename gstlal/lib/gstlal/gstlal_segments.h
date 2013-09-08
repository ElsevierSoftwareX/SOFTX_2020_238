/*
 * A simple segment list for gstlal
 *
 * Copyright (C) 2011,2013  Kipp Cannon, Chad Hanna
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */


#ifndef __GSTLAL_SEGMENTS_H__
#define __GSTLAL_SEGMENTS_H__


#include <glib.h>
#include <glib-object.h>

G_BEGIN_DECLS


/**
 * struct gstlal_segment:
 * @start: the start of the segment
 * @stop: the stop of the segment
 *
 * An interval (e.g., of time).  The segment spans [stop, stop).
 */


struct gstlal_segment {
	/*< public >*/
	guint64 start;
	guint64 stop;
};


/**
 * struct gstlal_segment_list:
 *
 * The opaque gstlal_segment_list structure.
 */


struct gstlal_segment_list {
	/*< private >*/
	struct gstlal_segment *segments;
	gint length;
};


struct gstlal_segment *gstlal_segment_new(guint64 start, guint64 stop);
void gstlal_segment_free(struct gstlal_segment *segment);
struct gstlal_segment_list *gstlal_segment_list_new(void);
void gstlal_segment_list_free(struct gstlal_segment_list *segmentlist);
gint gstlal_segment_list_length(const struct gstlal_segment_list *segmentlist);
struct gstlal_segment_list *gstlal_segment_list_append(struct gstlal_segment_list *segmentlist, struct gstlal_segment *segment);
gint gstlal_segment_list_index(const struct gstlal_segment_list *segmentlist, guint64 t);
struct gstlal_segment *gstlal_segment_list_get(struct gstlal_segment_list *segmentlist, gint index);
struct gstlal_segment_list *gstlal_segment_list_get_range(const struct gstlal_segment_list *segmentlist, guint64 start, guint64 stop);

struct gstlal_segment_list *gstlal_segment_list_from_g_value_array(GValueArray *va);
struct gstlal_segment *gstlal_segment_from_g_value_array(GValueArray *va);
GValueArray * g_value_array_from_gstlal_segment(struct gstlal_segment seg);
GValueArray * g_value_array_from_gstlal_segment_list(struct gstlal_segment_list *seglist);

G_END_DECLS


#endif	/* __GSTLAL_SEGMENTS_H__ */

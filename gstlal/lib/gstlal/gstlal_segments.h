/*
 * A simple segment list for gstlal
 *
 * Copyright (C) 2011  Kipp Cannon, Chad Hanna
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


struct gstlal_segment_list {
	struct gstlal_segment {
		guint64 start;
		guint64 stop;
	} *segments;
	gint length;
};


struct gstlal_segment *gstlal_segment_new(guint64, guint64);
void gstlal_segment_free(struct gstlal_segment *);
struct gstlal_segment_list *gstlal_segment_list_new(void);
void gstlal_segment_list_free(struct gstlal_segment_list *);
gint gstlal_segment_list_length(const struct gstlal_segment_list *);
struct gstlal_segment_list *gstlal_segment_list_append(struct gstlal_segment_list *, struct gstlal_segment *);
gint gstlal_segment_list_index(const struct gstlal_segment_list *, guint64);
struct gstlal_segment *gstlal_segment_list_get(struct gstlal_segment_list *, gint);
struct gstlal_segment_list *gstlal_segment_list_get_range(const struct gstlal_segment_list *, guint64, guint64);

struct gstlal_segment_list *gstlal_segment_list_from_g_value_array(GValueArray *);
struct gstlal_segment *gstlal_segment_from_g_value_array(GValueArray *);
GValueArray * g_value_array_from_gstlal_segment(struct gstlal_segment);
GValueArray * g_value_array_from_gstlal_segment_list(struct gstlal_segment_list *);

G_END_DECLS


#endif	/* __GSTLAL_SEGMENTS_H__ */

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


#include <glib.h>
#include <gstlal_segments.h>


struct gstlal_segment *gstlal_segment_new(guint64 start, guint64 stop)
{
	struct gstlal_segment *new = g_new(struct gstlal_segment, 1);

	if(!new)
		return NULL;

	new->start = start;
	new->stop = stop;

	return new;
}


void gstlal_segment_free(struct gstlal_segment *segment)
{
	g_free(segment);
}


struct gstlal_segment_list *gstlal_segment_list_new(void)
{
	struct gstlal_segment_list *new = g_new(struct gstlal_segment_list, 1);

	if(!new)
		return NULL;

	new->segments = NULL;
	new->length = 0;

	return  new;
}


void gstlal_segment_list_free(struct gstlal_segment_list *segmentlist)
{
	if(segmentlist) {
		g_free(segmentlist->segments);
		segmentlist->segments = NULL;
	}
	g_free(segmentlist);
}


gint gstlal_segment_list_length(const struct gstlal_segment_list *segmentlist)
{
	return segmentlist->length;
}


struct gstlal_segment_list *gstlal_segment_list_append(struct gstlal_segment_list *segmentlist, struct gstlal_segment *segment)
{
	struct gstlal_segment *new_segments = g_try_realloc(segmentlist->segments, (segmentlist->length + 1) * sizeof(*segment));

	if(!new_segments)
		return NULL;

	segmentlist->segments = new_segments;
	segmentlist->segments[segmentlist->length] = *segment;
	gstlal_segment_free(segment);
	segmentlist->length += 1;

	return segmentlist;
}


gint gstlal_segment_list_index(const struct gstlal_segment_list *segmentlist, guint64 t)
{
	gint i;

	for(i = segmentlist->length - 1; i >= 0; i--)
		if(segmentlist->segments[i].start <= t)
			break;
	return i;
}


struct gstlal_segment *gstlal_segment_list_get(struct gstlal_segment_list *segmentlist, gint index)
{
	return &segmentlist->segments[index];
}


struct gstlal_segment_list *gstlal_segment_list_get_range(const struct gstlal_segment_list *segmentlist, guint64 start, guint64 stop)
{
	struct gstlal_segment_list *new = gstlal_segment_list_new();
	gint lo, hi;
	gint i;

	if(!new)
		return NULL;

	lo = gstlal_segment_list_index(segmentlist, start);
	hi = gstlal_segment_list_index(segmentlist, stop);

	for(i = lo; i <= hi; i++) {
		if(i < 0 || segmentlist->segments[i].stop < start)
			continue;
		if(!gstlal_segment_list_append(new, gstlal_segment_new(segmentlist->segments[i].start, segmentlist->segments[i].stop))) {
			gstlal_segment_list_free(new);
			return NULL;
		}
	}

	if(new->length) {
		if(new->segments[0].start < start)
			new->segments[0].start = start;
		if(new->segments[new->length - 1].stop > stop)
			new->segments[new->length - 1].stop = stop;
	}

	return new;
}


/*
 * Functions to support segment lists to and from GValueArrays
 */


struct gstlal_segment *gstlal_segment_from_g_value_array(GValueArray *va)
{
	return gstlal_segment_new(g_value_get_uint64(g_value_array_get_nth(va, 0)), g_value_get_uint64(g_value_array_get_nth(va, 1)));
}


struct gstlal_segment_list *gstlal_segment_list_from_g_value_array(GValueArray *va)
{
	guint i;
	struct gstlal_segment_list *seglist = gstlal_segment_list_new();

	for(i = 0; i < va->n_values; i++)
		gstlal_segment_list_append(seglist, gstlal_segment_from_g_value_array(g_value_get_boxed(g_value_array_get_nth(va, i))));

	return seglist;
}


GValueArray * g_value_array_from_gstlal_segment(struct gstlal_segment seg)
{
	GValueArray *va = g_value_array_new(2);
	GValue v = {0,};
	g_value_set_uint64(&v, seg.start);
	g_value_array_append(va, &v);
	g_value_set_uint64(&v, seg.stop);
	g_value_array_append(va, &v);
	return va;
}


GValueArray * g_value_array_from_gstlal_segment_list(struct gstlal_segment_list *seglist)
{
	gint i;
	GValueArray *va = g_value_array_new(seglist->length);
	GValue v = {0,};
	g_value_init(&v, G_TYPE_VALUE_ARRAY);

	for(i = 0; i < seglist->length; i++) {
		g_value_take_boxed(&v, g_value_array_from_gstlal_segment(seglist->segments[i]));
		g_value_array_append(va, &v);
	}

	return va;
}

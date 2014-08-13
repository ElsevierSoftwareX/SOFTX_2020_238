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


/**
 * SECTION:gstlal_segments
 * @title: Segments
 * @include: gstlal/gstlal_segments.h
 * @short_description: Support for passing segment lists through GObject properties.
 *
 * Here is defined a structure for storing a start/stop pair (a "segment")
 * and a structure for storing a list of such structures (a "segment
 * list").  Unlike other libraries, like the segments library provided by
 * the Python glue package, this library does not implement high-level
 * segment arithmetic or other logical operations, only the most basic
 * storage and retrieval functions are provided here.  The purpose of this
 * code is to support passing segment lists through #GObject properties as
 * #GValueArrays, not to implement a segment arithmetic library.
 */


/*
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


#include <glib.h>
#include <gstlal_segments.h>


/*
 * ============================================================================
 *
 *                                Exported API
 *
 * ============================================================================
 */


/**
 * gstlal_segment_new:
 * @start:  the segment start
 * @stop:  the segment stop
 *
 * Allocate and initialize a struct gstlal_segment.  The segment represents
 * an interval (e.g., of time) and spans [start, stop).
 *
 * See also:  gstlal_segment_free()
 *
 * Returns:  a newly-allocated segment or %NULL on failure.
 */


struct gstlal_segment *gstlal_segment_new(guint64 start, guint64 stop)
{
	struct gstlal_segment *new = g_new(struct gstlal_segment, 1);

	if(!new)
		return NULL;

	new->start = start;
	new->stop = stop;

	return new;
}


/**
 * gstlal_segment_free:
 * @segment:  the struct gstlal_segment to free
 *
 * Frees all memory associated with a struct gstlal_segment.
 *
 * See also:  gstlal_segment_new()
 */


void gstlal_segment_free(struct gstlal_segment *segment)
{
	g_free(segment);
}


/**
 * gstlal_segment_list_new:
 *
 * Allocate a new struct gstlal_segment_list.
 *
 * See also:  gstlal_segment_list_free()
 *
 * Returns:  the newly-allocated struct gstlal_segment_list or %NULL on
 * failure.
 */


struct gstlal_segment_list *gstlal_segment_list_new(void)
{
	struct gstlal_segment_list *new = g_new(struct gstlal_segment_list, 1);

	if(!new)
		return NULL;

	new->segments = NULL;
	new->length = 0;

	return  new;
}


/**
 * gstlal_segment_list_free:
 * @segmentlist:  the struct gstlal_segment_list to free
 *
 * Frees all memory associated with a struct gstlal_segment_list including
 * any segments within it.
 *
 * See also:  gstlal_segment_list_new()
 */


void gstlal_segment_list_free(struct gstlal_segment_list *segmentlist)
{
	if(segmentlist) {
		g_free(segmentlist->segments);
		segmentlist->segments = NULL;
	}
	g_free(segmentlist);
}


/**
 * gstlal_segment_list_length:
 * @segmentlist:  the struct gstlal_segment_list whose length is to be
 * reported
 *
 * Returns:  the length of the struct gstlal_segment_list.
 */


gint gstlal_segment_list_length(const struct gstlal_segment_list *segmentlist)
{
	return segmentlist->length;
}


/**
 * gstlal_segment_list_append:
 * @segmentlist:  the struct gstlal_segment_list to which to append the
 * struct gstlal_segment
 * @segment:  the struct gstlal_segment to append
 *
 * Append a struct gstlal_segment to a struct gstlal_segment_list.  This
 * function takes ownership of the struct gstlal_segment, and the calling
 * code must not access it after invoking this function (even if the append
 * operation fails).
 *
 * Note that no check is made to ensure the segments in the list are in
 * order and disjoint.  Any conditions such as those must be enforced by
 * the application.
 *
 * Returns:  the struct gstlal_segment_list or NULL on failure.
 */


struct gstlal_segment_list *gstlal_segment_list_append(struct gstlal_segment_list *segmentlist, struct gstlal_segment *segment)
{
	struct gstlal_segment *new_segments = g_try_realloc(segmentlist->segments, (segmentlist->length + 1) * sizeof(*segment));

	if(!new_segments) {
		gstlal_segment_free(segment);
		return NULL;
	}

	segmentlist->segments = new_segments;
	segmentlist->segments[segmentlist->length] = *segment;
	gstlal_segment_free(segment);
	segmentlist->length += 1;

	return segmentlist;
}


/**
 * gstlal_segment_list_index:
 * @segmentlist:  the struct gstlal_segment_list to search
 * @t:  the value to search for
 *
 * Search for the first struct gstlal_segment in @segmentlist for which t <
 * stop.
 *
 * Returns:  the index of the first matching struct gstlal_segment or the
 * length of the list if no struct gstlal_segments match.
 */


gint gstlal_segment_list_index(const struct gstlal_segment_list *segmentlist, guint64 t)
{
	gint i;

	for(i = 0; i < segmentlist->length; i++)
		if(t < segmentlist->segments[i].stop)
			break;

	return i;
}


/**
 * gstlal_segment_list_get:
 * @segmentlist:  a struct gstlal_segment_list
 * @index:  the index of the segment to retrieve
 *
 * Returns:  the struct gstlal_segment at the requested position in the
 * list.  The address is a pointer into the list, it cannot be free'ed.
 */


struct gstlal_segment *gstlal_segment_list_get(struct gstlal_segment_list *segmentlist, gint index)
{
	return &segmentlist->segments[index];
}


/**
 * gstlal_segment_list_get_range:
 * @segmentlist:  a segment list
 * @start:  the start of the range to retrieve
 * @stop:  the end of the range to retrieve
 *
 * Constructs a new struct gstlal_segment_list containing the intersection
 * of @segmentlist and the segment [start, stop).
 *
 * Returns:  a newly-allocated struct gstlal_segment_list.
 */


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


/**
 * gstlal_segment_from_g_value_array:
 * @va:  a two-element #GValueArray
 *
 * Creates a new struct gstlal_segment from a two-element #GValueArray.
 * The start and stop of the segment are set to the 0th and 1st elements of
 * the array, respectively.  This function borrows a reference to the
 * GValueArray.  NOTE:  very little error checking is done!
 *
 * See also:  g_value_array_from_gstlal_segment()
 *
 * Returns:  the newly-allocated struct gstlal_segment.
 */


struct gstlal_segment *gstlal_segment_from_g_value_array(GValueArray *va)
{
	return gstlal_segment_new(g_value_get_uint64(g_value_array_get_nth(va, 0)), g_value_get_uint64(g_value_array_get_nth(va, 1)));
}


/**
 * gstlal_segment_list_from_g_value_array:
 * @va:  a #GValueArray of two-element #GValueArrays
 *
 * Creates a new struct gstlal_segment_list from a #GValueArray of
 * two-element #GValueArrays.  Each two-element #GValueArray is converted
 * to a struct gstlal_segment using gstlal_segment_from_g_value_array(),
 * and the struct gstlal_segment_list populated with the results in order.
 * This function borrows a reference to the #GValueArray.
 *
 * See also:  g_value_array_from_gstlal_segment_list()
 *
 * Returns:  the newly-allocated struct gstlal_segment_list.
 */


struct gstlal_segment_list *gstlal_segment_list_from_g_value_array(GValueArray *va)
{
	guint i;
	struct gstlal_segment_list *seglist = gstlal_segment_list_new();

	for(i = 0; i < va->n_values; i++)
		gstlal_segment_list_append(seglist, gstlal_segment_from_g_value_array(g_value_get_boxed(g_value_array_get_nth(va, i))));

	return seglist;
}


/**
 * g_value_array_from_gstlal_segment:
 * @seg:  the struct gstlal_segment to convert to a #GValueArray
 *
 * Create a two-element #GValueArray containing the start and stop of a
 * struct gstlal_segment.
 *
 * See also:  gstlal_segment_from_g_value_array()
 *
 * Returns:  the newly-allocated two-element #GValueArray
 */


GValueArray *g_value_array_from_gstlal_segment(struct gstlal_segment seg)
{
	GValueArray *va = g_value_array_new(2);
	GValue v = {0,};
	g_value_set_uint64(&v, seg.start);
	g_value_array_append(va, &v);
	g_value_set_uint64(&v, seg.stop);
	g_value_array_append(va, &v);
	return va;
}


/**
 * g_value_array_from_gstlal_segment_list:
 * @seglist:  the struct gstlal_segment_list to convert to a #GValueArray
 * of two-element #GValueArrays
 *
 * Create a #GValueArray of two-element #GValueArrays containing the
 * contents of a struct gstlal_segment_list.
 *
 * See also:  gstlal_segment_list_from_g_value_array()
 *
 * Returns:  the newly-allocated #GValueArray
 */


GValueArray *g_value_array_from_gstlal_segment_list(struct gstlal_segment_list *seglist)
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

/*
 * GstFrHistory
 *
 * Copyright (C) 2013  Kipp Cannon
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


/*
 * stuff from GObject/GStreamer
 */


#include <glib.h>
#include <gst/gst.h>


/*
 * our own stuff
 */


#include <gstfrhistory.h>


/*
 * ============================================================================
 *
 *                                Exported API
 *
 * ============================================================================
 */


/**
 * gst_frhistory_new:
 * @name:  the name to give the new #GstFrHistory
 *
 * Creates a new GstFrHistory object.  The object returned should be freed
 * with #gst_frhistory_free().
 *
 * Returns: a new #GstFrHistory
 */


GstFrHistory *gst_frhistory_new(const gchar *name)
{
	GstFrHistory *new = g_slice_new(GstFrHistory);

	new->type = GST_FRHISTORY_TYPE;
	new->name = g_strdup(name);
	new->time = -1;
	new->comment = NULL;

	return new;
}


/**
 * gst_frhistory_copy:
 * @frhistory: object to copy
 *
 * Creates a deep copy of the GstFrHistory object @frhistory.
 *
 * Free function:  gst_frhistory_free
 *
 * Returns: a new #GstFrHistory
 */


GstFrHistory *gst_frhistory_copy(const GstFrHistory *frhistory)
{
	GstFrHistory *new = gst_frhistory_new(frhistory->name);

	new->time = frhistory->time;
	new->comment = g_strdup(frhistory->comment);

	return new;
}


/**
 * gst_frhistory_to_string:
 * @frhistory: #GstFrHistory object to represent as string
 *
 * Creates a human-readable string representation of the contents of a
 * GstFrHistory object.  The returned string should be freed with #g_free()
 * when no longer needed.
 *
 * Returns: a newly-allocated string holding the result
 */


gchar *gst_frhistory_to_string(const GstFrHistory *frhistory)
{
	g_return_val_if_fail(GST_IS_FRHISTORY(frhistory), NULL);

	if(frhistory->time == (guint32) -1)
		return g_strdup_printf("%s @ (unknown) s: %s", frhistory->name, frhistory->comment);
	return g_strdup_printf("%s @ %u s: %s", frhistory->name, frhistory->time, frhistory->comment);
}


/**
 * gst_frhistory_set_timestamp:
 * @frhistory: #GstFrHistory whose timestamp is to be set
 * @time: @GstClockTime value to set timestamp to
 *
 * FrHistory objects can only store 32-bit integer second timestamps.  This
 * function adapts GStreamer's native 64-bit integer nanosecond timestamps
 * to a value suitable for an FrHistory by truncating to the largest
 * integer second not greater than the timestamp.
 */


void gst_frhistory_set_timestamp(GstFrHistory *frhistory, GstClockTime time)
{
	g_return_if_fail(GST_IS_FRHISTORY(frhistory));

	frhistory->time = GST_CLOCK_TIME_IS_VALID(time) ? time / GST_SECOND : (guint32) -1;
}


/**
 * gst_frhistory_get_timestamp:
 * @frhistory: #GstFrHistory whose timestamp is to be retrieved
 *
 * Returns:  the GstClockTime timestamp
 */


GstClockTime gst_frhistory_get_timestamp(const GstFrHistory *frhistory)
{
	g_return_val_if_fail(GST_IS_FRHISTORY(frhistory), GST_CLOCK_TIME_NONE);

	return frhistory->time == (guint32) -1 ? GST_CLOCK_TIME_NONE : frhistory->time * GST_SECOND;
}


/**
 * gst_frhistory_set_comment:
 * @frhistory: #GstFrHistory whose comment is to be set
 * @comment: comment string
 *
 * Sets the comment string of the @frhistory to a copy of @comment.  Any
 * previous value will be #g_free()ed.
 */


void gst_frhistory_set_comment(GstFrHistory *frhistory, const gchar *comment)
{
	g_return_if_fail(GST_IS_FRHISTORY(frhistory));

	g_free(frhistory->comment);
	frhistory->comment = g_strdup(comment);
}


/**
 * gst_frhistory_get_comment:
 * @frhistory: #GstFrHistory whose comment is to be retrieved
 *
 * Returns a borrowed reference to the comment string stored in the
 * #GstFrHistory.  The calling code does not own the string and should not
 * free it.
 *
 * Returns:  a borrowed reference to the comment string
 */


const gchar *gst_frhistory_get_comment(const GstFrHistory *frhistory)
{
	g_return_val_if_fail(GST_IS_FRHISTORY(frhistory), NULL);

	return frhistory->comment;
}


/**
 * gst_frhistory_get_name:
 * @frhistory: #GstFrHistory whose name is to be retrieved
 *
 * Returns a borrowed reference to the name string stored in the
 * #GstFrHistory.  The calling code does not own the string and should not
 * free it.
 *
 * Returns:  a borrowed reference to the name string
 */


const gchar *gst_frhistory_get_name(const GstFrHistory *frhistory)
{
	g_return_val_if_fail(GST_IS_FRHISTORY(frhistory), NULL);

	return frhistory->name;
}


/**
 * gst_frhistory_free
 * @frhistory: object to free
 *
 * Frees all memory associated with @frhistory.
 */


void gst_frhistory_free(GstFrHistory *frhistory)
{
	if(frhistory) {
		g_free(frhistory->name);
		g_free(frhistory->comment);
	}
	g_slice_free(GstFrHistory, frhistory);
}


/**
 * gst_frhistory_compare_by_time
 * @a: address of GstFrHistory a
 * @b: address of GstFrHistory b
 *
 * Return <0, 0, >0 if @a's timestamp is less than, equal to, or greater
 * than, respectively, @b's timestamp.  Use with g_list_sort() to put a
 * GValueArray of GstFrHistory objects into time order.  Uninitialized
 * timestamps are treated as being less than all other timestamps.  If
 * either or both of @a and @b is NULL the return value is undefined.
 */


gint gst_frhistory_compare_by_time(gconstpointer a, gconstpointer b)
{
	guint32 t_a, t_b;

	g_return_val_if_fail(GST_IS_FRHISTORY(a) && GST_IS_FRHISTORY(b), -1);

	t_a = ((const GstFrHistory *) a)->time;
	t_b = ((const GstFrHistory *) b)->time;

	if(t_a == (guint32) -1)
		t_a = 0;	/* smallest allowed value */
	if(t_b == (guint32) -1)
		t_b = 0;	/* smallest allowed value */
	return t_a < t_b ? -1 : t_a > t_b ? +1 : 0;
}


/*
 * ============================================================================
 *
 *                               GType Methods
 *
 * ============================================================================
 */


static GstFrHistory *copy_conditional(GstFrHistory *src)
{
	return src ? gst_frhistory_copy(src) : NULL;
}


static void to_string(const GValue *src, GValue *dst)
{
	g_return_if_fail(src != NULL);
	g_return_if_fail(dst != NULL);

	dst->data[0].v_pointer = gst_frhistory_to_string(src->data[0].v_pointer);
}


GType gst_frhistory_get_type(void)
{
	static GType type = 0;

	if(G_UNLIKELY(type == 0)) {
		type = g_boxed_type_register_static(
			"GstFrHistory",
			(GBoxedCopyFunc) copy_conditional,
			(GBoxedFreeFunc) gst_frhistory_free
		);

		g_value_register_transform_func(type, G_TYPE_STRING, to_string);
	}

	return type;
}

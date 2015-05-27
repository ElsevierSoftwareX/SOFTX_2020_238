/*
 * GstLALFrHistory
 *
 * Copyright (C) 2013,2015  Kipp Cannon
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


/**
 * SECTION:gstlal_frhistory
 * @include:  gstlal/gstlal_frhistory.h
 * @short_description:  #GValue type for holding FrHistory information.
 *
 * #GstLALFrHistory is a #GValue type that carries a name, a timestamp and
 * a comment.
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


#include <gstlal_frhistory.h>


/*
 * ============================================================================
 *
 *                                Exported API
 *
 * ============================================================================
 */


/**
 * gstlal_frhistory_new:
 * @name:  (transfer none):  The name to give the new #GstLALFrHistory.
 * The calling code retains ownership of the string.
 *
 * Creates a new #GstLALFrHistory object.  The object returned should be
 * freed with #gstlal_frhistory_free().
 *
 * Returns:  (transfer full):  A new #GstLALFrHistory.  Free with
 * #gstlal_frhistory_free().
 */


GstLALFrHistory *gstlal_frhistory_new(const gchar *name)
{
	GstLALFrHistory *new = g_slice_new(GstLALFrHistory);

	g_return_val_if_fail(name, NULL);

	new->name = g_strdup(name);
	new->time = -1;
	new->comment = NULL;

	return new;
}


/**
 * gstlal_frhistory_copy:  (method)
 * @self:  (transfer none):  #GstLALFrHistory object to copy.
 *
 * Creates a deep copy of the #GstLALFrHistory object @self.
 *
 * Returns:  (transfer full):  A new #GstLALFrHistory.  Free with
 * #gstlal_frhistory_free().
 */


GstLALFrHistory *gstlal_frhistory_copy(const GstLALFrHistory *self)
{
	GstLALFrHistory *new = gstlal_frhistory_new(self->name);

	new->time = self->time;
	new->comment = g_strdup(self->comment);

	return new;
}


/**
 * gstlal_frhistory_to_string:  (method)
 * @self:  (transfer none):  #GstLALFrHistory object to represent as a string.
 *
 * Creates a human-readable string representation of the contents of a
 * #GstLALFrHistory object.
 *
 * Returns:  (transfer full):  A newly-allocated string holding the result.
 * Free with #g_free().
 */


gchar *gstlal_frhistory_to_string(const GstLALFrHistory *self)
{
	if(self->time == (guint32) -1)
		return g_strdup_printf("%s @ (unknown) s: %s", self->name, self->comment);
	return g_strdup_printf("%s @ %u s: %s", self->name, self->time, self->comment); }


/**
 * gstlal_frhistory_set_timestamp:  (method)
 * @self:  (transfer none):  #GstLALFrHistory object whose timestamp is to
 * be set.
 * @time:  #GstClockTime value to which to set timestamp.
 *
 * FrHistory objects can only store 32-bit integer second timestamps.  This
 * function adapts GStreamer's native 64-bit integer nanosecond timestamps
 * to a value suitable for an FrHistory by truncating to the largest
 * integer second not greater than the timestamp.
 */


void gstlal_frhistory_set_timestamp(GstLALFrHistory *self, GstClockTime time)
{
	self->time = GST_CLOCK_TIME_IS_VALID(time) ? time / GST_SECOND : (guint32) -1;
}


/**
 * gstlal_frhistory_get_timestamp:  (method)
 * @self:  (transfer none):  #GstLALFrHistory object whose timestamp is to
 * be retrieved.
 *
 * Returns:  #GstClockTime timestamp.
 */


GstClockTime gstlal_frhistory_get_timestamp(const GstLALFrHistory *self)
{
	return self->time == (guint32) -1 ? GST_CLOCK_TIME_NONE : self->time * GST_SECOND;
}


/**
 * gstlal_frhistory_set_comment:  (method)
 * @self:  (transfer none):  #GstLALFrHistory object whose comment is to be
 * set.
 * @comment:  (transfer none) (nullable):  Comment string or NULL.  Calling
 * code retains ownership.
 *
 * Sets the comment string of the @self to a copy of @comment.  Any
 * previous value will be #g_free()ed.
 */


void gstlal_frhistory_set_comment(GstLALFrHistory *self, const gchar *comment)
{
	g_free(self->comment);
	self->comment = g_strdup(comment);
}


/**
 * gstlal_frhistory_get_comment:  (method)
 * @self:  (transfer none):  #GstLALFrHistory object whose comment is to be
 * retrieved.
 *
 * Returns a borrowed reference to the comment string stored in the
 * #GstLALFrHistory.  The calling code does not own the string and should
 * not free it.
 *
 * Returns:  (transfer none):  A borrowed reference to the comment string.
 * Do not free.
 */


const gchar *gstlal_frhistory_get_comment(const GstLALFrHistory *self)
{
	return self->comment;
}


/**
 * gstlal_frhistory_get_name:  (method)
 * @self:  (transfer none):  #GstLALFrHistory object whose name is to be
 * retrieved.
 *
 * Returns a borrowed reference to the name string stored in the
 * #GstLALFrHistory.  The calling code does not own the string and should
 * not free it.
 *
 * Returns:  (transfer none):  A borrowed reference to the name string.  Do
 * not free.
 */


const gchar *gstlal_frhistory_get_name(const GstLALFrHistory *self)
{
	return self->name;
}


/**
 * gstlal_frhistory_free:  (method)
 * @self:  (transfer none):  #GstLALFrHistory object to free.
 *
 * Frees all memory associated with @self.
 */


void gstlal_frhistory_free(GstLALFrHistory *self)
{
	if(self) {
		g_free(self->name);
		g_free(self->comment);
	}
	g_slice_free(GstLALFrHistory, self);
}


/**
 * gstlal_frhistory_compare_by_time:
 * @a: (transfer none):  Address of #GstLALFrHistory a.
 * @b: (transfer none):  Address of #GstLALFrHistory b.
 *
 * Returns:  <0, 0, >0 if @a's timestamp is less than, equal to, or greater
 * than, respectively, @b's timestamp.  Use with #g_list_sort() to put a
 * #GValueArray of #GstLALFrHistory objects into time order.  Uninitialized
 * timestamps are treated as being less than all other timestamps.  If
 * either or both of @a and @b is NULL the return value is undefined.
 */


gint gstlal_frhistory_compare_by_time(gconstpointer a, gconstpointer b)
{
	guint32 t_a = ((const GstLALFrHistory *) a)->time;
	guint32 t_b = ((const GstLALFrHistory *) b)->time;

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


static GstLALFrHistory *copy_conditional(GstLALFrHistory *src)
{
	return src ? gstlal_frhistory_copy(src) : NULL;
}


static void to_string(const GValue *src, GValue *dst)
{
	g_return_if_fail(GSTLAL_VALUE_HOLDS_FRHISTORY(src));

	dst->data[0].v_pointer = gstlal_frhistory_to_string(src->data[0].v_pointer);
}


GType gstlal_frhistory_get_type(void)
{
	static GType type = 0;

	if(G_UNLIKELY(type == 0)) {
		type = g_boxed_type_register_static(
			"GstLALFrHistory",
			(GBoxedCopyFunc) copy_conditional,
			(GBoxedFreeFunc) gstlal_frhistory_free
		);

		g_value_register_transform_func(type, G_TYPE_STRING, to_string);
	}

	return type;
}

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


#ifndef __GSTLAL_FRHISTORY_H__
#define __GSTLAL_FRHISTORY_H__


/*
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


#include <glib.h>
#include <gst/gst.h>


G_BEGIN_DECLS


/*
 * ============================================================================
 *
 *                                    Type
 *
 * ============================================================================
 */


#define GSTLAL_FRHISTORY_TYPE \
	(gstlal_frhistory_get_type())
#define GSTLAL_FRHISTORY(obj) \
	((GstLALFrHistory *) (obj))
#define GSTLAL_VALUE_HOLDS_FRHISTORY(x) \
	((x) != NULL && G_VALUE_TYPE(x) == GSTLAL_FRHISTORY_TYPE)


typedef struct _GstLALFrHistory GstLALFrHistory;


/**
 * GstLALFrHistory:
 * @time:  Raw FrHistory time entry.  Use #gstlal_frhistory_set_timestamp() and
 * #gstlal_frhistory_get_timestamp() to access as a #GstClockTime.
 * @comment:  Comment string.
 */


struct _GstLALFrHistory {
	guint32 time;
	gchar *comment;

	/*< private >*/
	gchar *name;	/* immutable FrHistory name */
};


/*
 * ============================================================================
 *
 *                                Exported API
 *
 * ============================================================================
 */


GType gstlal_frhistory_get_type(void);


GstLALFrHistory *gstlal_frhistory_new(const gchar *name);
GstLALFrHistory *gstlal_frhistory_copy(const GstLALFrHistory *self);
gchar *gstlal_frhistory_to_string(const GstLALFrHistory *self);
void gstlal_frhistory_set_timestamp(GstLALFrHistory *self, GstClockTime time);
GstClockTime gstlal_frhistory_get_timestamp(const GstLALFrHistory *self);
void gstlal_frhistory_set_comment(GstLALFrHistory *self, const gchar *comment);
const gchar *gstlal_frhistory_get_comment(const GstLALFrHistory *self);
const gchar *gstlal_frhistory_get_name(const GstLALFrHistory *self);
void gstlal_frhistory_free(GstLALFrHistory *self);
gint gstlal_frhistory_compare_by_time(gconstpointer a, gconstpointer b);


G_END_DECLS


#endif	/* __GSTLAL_FRHISTORY_H__ */

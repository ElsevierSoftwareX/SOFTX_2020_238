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


#ifndef __GST_FRHISTORY_H__
#define __GST_FRHISTORY_H__


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


typedef struct _GstFrHistoryClass GstFrHistoryClass;
typedef struct _GstFrHistory GstFrHistory;


#define GST_FRHISTORY_TYPE \
	(gst_frhistory_get_type())
#define GST_FRHISTORY(obj) \
	((GstFrHistory *) (obj))
#define GST_IS_FRHISTORY(obj) \
	((obj) && (GST_FRHISTORY(obj)->type == GST_FRHISTORY_TYPE)


/**
 * GstFrHistory
 */


struct _GstFrHistory {
	GType type;

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


GType gst_frhistory_get_type(void);


GstFrHistory *gst_frhistory_new(const gchar *);
GstFrHistory *gst_frhistory_copy(const GstFrHistory *);
gchar *gst_frhistory_to_string(const GstFrHistory *);
void gst_frhistory_set_timestamp(GstFrHistory *, GstClockTime);
GstClockTime gst_frhistory_get_timestamp(const GstFrHistory *);
void gst_frhistory_set_comment(GstFrHistory *, const gchar *);
const gchar *gst_frhistory_get_comment(const GstFrHistory *);
const gchar *gst_frhistory_get_name(const GstFrHistory *);
void gst_frhistory_free(GstFrHistory *);
gint gst_frhistory_compare_by_time(gconstpointer, gconstpointer);


G_END_DECLS


#endif	/* __GST_FRHISTORY_H__ */

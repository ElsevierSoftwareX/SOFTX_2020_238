/*
 * GstFrPad
 *
 * Copyright (C) 2012  Kipp Cannon
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


#ifndef __GST_FRPAD_H__
#define __GST_FRPAD_H__


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
 *                               Pad Type Enum
 *
 * ============================================================================
 */


enum gst_frpad_type_t {
	GST_FRPAD_TYPE_FRADCDATA,
	GST_FRPAD_TYPE_FRPROCDATA,
	GST_FRPAD_TYPE_FRSIMDATA
};


#define GST_FRPAD_TYPE_TYPE \
	(gst_frpad_type_get_type())


GType gst_frpad_type_get_type(void);


/*
 * ============================================================================
 *
 *                                    Type
 *
 * ============================================================================
 */


#define GST_FRPAD_TYPE \
	(gst_frpad_get_type())
#define GST_FRPAD(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GST_FRPAD_TYPE, GstFrPad))
#define GST_FRPAD_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GST_FRPAD_TYPE, GstFrPadClass))
#define GST_FRPAD_GET_CLASS(obj) \
	(G_TYPE_INSTANCE_GET_CLASS((obj), GST_FRPAD_TYPE, GstFrPadClass))
#define GST_IS_FRPAD(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_FRPAD_TYPE))
#define GST_IS_FRPAD_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GST_FRPAD_TYPE))


typedef struct _GstFrPadClass GstFrPadClass;
typedef struct _GstFrPad GstFrPad;


struct _GstFrPadClass {
	GstPadClass parent_class;
};


/**
 * GstFrPad
 *
 * The opaque #GstFrPad data structure.
 */


struct _GstFrPad {
	GstPad pad;

	enum gst_frpad_type_t pad_type;
	gchar *comment;
};


/*
 * ============================================================================
 *
 *                                Exported API
 *
 * ============================================================================
 */


GType gst_frpad_get_type(void);


GstFrPad *gst_frpad_new_from_template(GstPadTemplate *, const gchar *);


G_END_DECLS


#endif	/* __GST_FRPAD_H__ */

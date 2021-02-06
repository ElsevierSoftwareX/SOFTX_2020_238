/*
 * Copyright (C) 2021  Aaron Viets <aaron.viets@ligo.org>
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


#ifndef __GSTLAL_MINMAX_H__
#define __GSTLAL_MINMAX_H__

#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>

G_BEGIN_DECLS
#define GSTLAL_MINMAX_TYPE \
	(gstlal_minmax_get_type())
#define GSTLAL_MINMAX(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_MINMAX_TYPE, GSTLALMinMax))
#define GSTLAL_MINMAX_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_MINMAX_TYPE, GSTLALMinMaxClass))
#define GST_IS_GSTLAL_MINMAX(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_MINMAX_TYPE))
#define GST_IS_GSTLAL_MINMAX_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_MINMAX_TYPE))


typedef struct _GSTLALMinMax GSTLALMinMax;
typedef struct _GSTLALMinMaxClass GSTLALMinMaxClass;


/**
 * GSTLALMinMax:
 */


struct _GSTLALMinMax {
	GstBaseTransform element;

	/* stream info */
	gint rate;
	gint channels_in;
	gint unit_size_in;
	gint unit_size_out;
	enum gstlal_minmax_data_type {
		GSTLAL_MINMAX_F32 = 0,
		GSTLAL_MINMAX_F64,
		GSTLAL_MINMAX_Z64,
		GSTLAL_MINMAX_Z128
	} data_type;
	double max_double;
	float max_float;

	/* timestamp book-keeping */
	GstClockTime t0;
	guint64 offset0;
	guint64 next_in_offset;
	guint64 next_out_offset;
	gboolean need_discont;

	/* properties */
	gint mode;
};


/**
 * GSTLALMinMaxClass:
 * @parent_class:  the parent class
 */


struct _GSTLALMinMaxClass {
	GstBaseTransformClass parent_class;
};


GType gstlal_minmax_get_type(void);


G_END_DECLS


#endif	/* __GSTLAL_MINMAX_H__ */

/*
 * An element to chop up audio buffers into smaller pieces.
 *
 * Copyright (C) 2008  Kipp Cannon, Chad Hanna
 *
 *  This program is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU General Public License as published by the
 *  Free Software Foundation; either version 2 of the License, or (at your
 *  option) any later version.
 *
 *  This program is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */


#ifndef __GSTLAL_SHIFT_H__
#define __GSTLAL_SHIFT_H__


#include <glib.h>
#include <gst/gst.h>


G_BEGIN_DECLS


#define GSTLAL_SHIFT_TYPE \
	(gstlal_shift_get_type())
#define GSTLAL_SHIFT(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_SHIFT_TYPE, GSTLALShift))
#define GSTLAL_SHIFT_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_SHIFT_TYPE, GSTLALShiftClass))
#define GST_IS_GSTLAL_SHIFT(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_SHIFT_TYPE))
#define GST_IS_GSTLAL_SHIFT_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_SHIFT_TYPE))


typedef struct {
	GstElementClass parent_class;
} GSTLALShiftClass;


typedef struct _GSTLALShift {
	GstElement element;

	GstPad *sinkpad;
	GstPad *srcpad;

	gint rate;
	gint unit_size;
	gint64 shift;
} GSTLALShift;


GType gstlal_shift_get_type(void);


G_END_DECLS


#endif	/* __GSTLAL_SHIFT_H__ */

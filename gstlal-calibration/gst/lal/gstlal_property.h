/*
 * Copyright (C) 2018  Aaron Viets <aaron.viets@ligo.org>
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


#ifndef __GSTLAL_PROPERTY_H__
#define __GSTLAL_PROPERTY_H__


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasesink.h>


G_BEGIN_DECLS


#define GSTLAL_PROPERTY_TYPE \
	(gstlal_property_get_type())
#define GSTLAL_PROPERTY(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_PROPERTY_TYPE, GSTLALProperty))
#define GSTLAL_PROPERTY_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_PROPERTY_TYPE, GSTLALPropertyClass))
#define GST_IS_GSTLAL_PROPERTY(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_PROPERTY_TYPE))
#define GST_IS_GSTLAL_PROPERTY_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_PROPERTY_TYPE))


typedef struct _GSTLALProperty GSTLALProperty;
typedef struct _GSTLALPropertyClass GSTLALPropertyClass;


/**
 * GSTLALProperty:
 */


struct _GSTLALProperty {
	GstBaseSink basesink;

	/* stream info */
	gint rate;
	gint unit_size;
	enum gstlal_property_data_type {
		GSTLAL_PROPERTY_SIGNED = 0,
		GSTLAL_PROPERTY_UNSIGNED,
		GSTLAL_PROPERTY_FLOAT,
	} data_type;

	/* timestamp bookkeeping */
	GstClockTime t0;
	guint64 offset0;
	guint64 next_in_offset;

	/* filter memory */
	gint64 num_in_avg;

	/* properties */
	gint64 update_samples;
	gint64 shift_samples;
	gint64 average_samples;
	gboolean update_when_change;
	double current_average;
	guint64 timestamp;
};


/**
 * GSTLALPropertyClass:
 * @parent_class:  the parent class
 */


struct _GSTLALPropertyClass {
	GstBaseSinkClass parent_class;
};


GType gstlal_property_get_type(void);


G_END_DECLS


#endif	/* __GSTLAL_PROPERTY_H__ */

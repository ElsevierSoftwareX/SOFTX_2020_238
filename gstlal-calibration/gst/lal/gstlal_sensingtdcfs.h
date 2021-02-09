/*
 * Copyright (C) 2019  Aaron Viets <aaron.viets@ligo.org>
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


#ifndef __GSTLAL_SENSINGTDCFS_H__
#define __GSTLAL_SENSINGTDCFS_H__

#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>

G_BEGIN_DECLS
#define GSTLAL_SENSINGTDCFS_TYPE \
	(gstlal_sensingtdcfs_get_type())
#define GSTLAL_SENSINGTDCFS(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_SENSINGTDCFS_TYPE, GSTLALSensingTDCFs))
#define GSTLAL_SENSINGTDCFS_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_SENSINGTDCFS_TYPE, GSTLALSensingTDCFsClass))
#define GST_IS_GSTLAL_SENSINGTDCFS(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_SENSINGTDCFS_TYPE))
#define GST_IS_GSTLAL_SENSINGTDCFS_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_SENSINGTDCFS_TYPE))


typedef struct _GSTLALSensingTDCFs GSTLALSensingTDCFs;
typedef struct _GSTLALSensingTDCFsClass GSTLALSensingTDCFsClass;


/**
 * GSTLALSensingTDCFs:
 */


struct _GSTLALSensingTDCFs {
	GstBaseTransform element;

	/* stream info */
	gint rate;
	gint channels_in;
	gint channels_out;
	gint unit_size_in;
	gint unit_size_out;
	enum gstlal_sensingtdcfs_data_type {
		GSTLAL_SENSINGTDCFS_FLOAT = 0,
		GSTLAL_SENSINGTDCFS_DOUBLE
	} data_type;

	/* timestamp book-keeping */
	GstClockTime t0;
	guint64 offset0;
	guint64 next_in_offset;
	guint64 next_out_offset;
	gboolean need_discont;

	/* properties */
	gint sensing_model;
	double freq1;
	double freq2;
	double freq4;
	double current_kc;
	double current_fcc;
	double current_fs_squared;
	double current_fs_over_Q;
};


/**
 * GSTLALSensingTDCFsClass:
 * @parent_class:  the parent class
 */


struct _GSTLALSensingTDCFsClass {
	GstBaseTransformClass parent_class;
};


GType gstlal_sensingtdcfs_get_type(void);


G_END_DECLS


#endif	/* __GSTLAL_SENSINGTDCFS_H__ */

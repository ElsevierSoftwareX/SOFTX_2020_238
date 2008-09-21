/*
 * A many-to-many mixer.
 *
 * Copyright (C) 2008  Kipp Cannon, Chad Hanna
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


#ifndef __GSTLAL_MATRIXMIXER_H__
#define __GSTLAL_MATRIXMIXER_H__


#include <gst/gst.h>


#include <gsl/gsl_matrix.h>


G_BEGIN_DECLS


#define GSTLAL_MATRIXMIXER_TYPE \
	(gstlal_matrixmixer_get_type())
#define GSTLAL_MATRIXMIXER(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_MATRIXMIXER_TYPE, GSTLALMatrixMixer))
#define GSTLAL_MATRIXMIXER_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_MATRIXMIXER_TYPE, GSTLALMatrixMixerClass))
#define GST_IS_GSTLAL_MATRIXMIXER(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_MATRIXMIXER_TYPE))
#define GST_IS_GST_MATRIXMIXER_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_MATRIXMIXER_TYPE))


typedef struct {
	GstElementClass parent_class;
} GSTLALMatrixMixerClass;


typedef struct {
	GstElement element;

	GstPad *sinkpad;
	GstPad *srcpad;

	GMutex *mixmatrix_lock;
	GCond *mixmatrix_available;
	GstBuffer *mixmatrix_buf;
	gsl_matrix_view mixmatrix;
} GSTLALMatrixMixer;


GType gstlal_matrixmixer_get_type(void);


G_END_DECLS


#endif	/* __GSTLAL_MATRIXMIXER_H__ */

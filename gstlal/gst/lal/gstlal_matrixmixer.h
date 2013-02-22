/*
 * A many-to-many mixer.
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


#ifndef __GSTLAL_MATRIXMIXER_H__
#define __GSTLAL_MATRIXMIXER_H__


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>


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
#define GST_IS_GSTLAL_MATRIXMIXER_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_MATRIXMIXER_TYPE))


typedef struct {
	GstBaseTransformClass parent_class;
} GSTLALMatrixMixerClass;


typedef struct {
	GstBaseTransform element;

	/*
	 * The mixer is controled using a matrix whose elements provide the
	 * mixing coefficients and whose size sets the number of input and
	 * output channels.  The meaning of the coefficients is shown in
	 * the following diagram.
	 *
	 *                1 -->  a11  a12  a13  a14  a15
	 *
	 * input channel  2 -->  a21  a22  a23  a24  a25
	 *
	 *                3 -->  a31  a32  a33  a34  a35
	 *
	 *                        |    |    |    |    |
	 *                        V    V    V    V    V
	 *
	 *                        1    2    3    4    5
	 *
	 *                           output channel
	 *
	 * The matrix is provided to the element via the "matrix" property
	 * as a GValueArray of rows, each row is a GValueArray containing
	 * double-precision floats (the rows must all be the same size).
	 *
	 * The coefficient ordering and the manner in which they map input
	 * channels to output channels is chosen so that the transformation
	 * of an input buffer into an output buffer can be performed as a
	 * single matrix-matrix multiplication.
	 */

	GMutex *mixmatrix_lock;
	GCond *mixmatrix_available;
	enum {
		GSTLAL_MATRIXMIXER_FLOAT,
		GSTLAL_MATRIXMIXER_DOUBLE,
		GSTLAL_MATRIXMIXER_COMPLEX_FLOAT,
		GSTLAL_MATRIXMIXER_COMPLEX_DOUBLE
	} data_type;
	union {
		void *as_void;
		gsl_matrix_float *as_float;
		gsl_matrix *as_double;
		gsl_matrix_complex_float *as_complex_float;
		gsl_matrix_complex *as_complex_double;
	} mixmatrix;
} GSTLALMatrixMixer;


GType gstlal_matrixmixer_get_type(void);


G_END_DECLS


#endif	/* __GSTLAL_MATRIXMIXER_H__ */

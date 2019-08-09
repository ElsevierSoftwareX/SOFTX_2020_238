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


#ifndef __GSTLAL_MATRIXSOLVER_H__
#define __GSTLAL_MATRIXSOLVER_H__

#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_permutation.h>

G_BEGIN_DECLS
#define GSTLAL_MATRIXSOLVER_TYPE \
	(gstlal_matrixsolver_get_type())
#define GSTLAL_MATRIXSOLVER(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_MATRIXSOLVER_TYPE, GSTLALMatrixSolver))
#define GSTLAL_MATRIXSOLVER_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_MATRIXSOLVER_TYPE, GSTLALMatrixSolverClass))
#define GST_IS_GSTLAL_MATRIXSOLVER(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_MATRIXSOLVER_TYPE))
#define GST_IS_GSTLAL_MATRIXSOLVER_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_MATRIXSOLVER_TYPE))


typedef struct _GSTLALMatrixSolver GSTLALMatrixSolver;
typedef struct _GSTLALMatrixSolverClass GSTLALMatrixSolverClass;


/**
 * GSTLALMatrixSolver:
 */


struct _GSTLALMatrixSolver {
	GstBaseTransform element;

	/* stream info */
	gint rate;
	gint channels_in;
	gint channels_out;
	gint unit_size_out;
	enum gstlal_matrixsolver_data_type {
		GSTLAL_MATRIXSOLVER_F32 = 0,
		GSTLAL_MATRIXSOLVER_F64,
		GSTLAL_MATRIXSOLVER_Z64,
		GSTLAL_MATRIXSOLVER_Z128
	} data_type;

	/* timestamp book-keeping */
	GstClockTime t0;
	guint64 offset0;
	guint64 next_in_offset;
	guint64 next_out_offset;
	gboolean need_discont;

	/* gsl stuff for solving systens of linear equations */
	union {
		struct {
			gsl_vector *invec;
			gsl_vector *outvec;
			gsl_matrix *matrix;
		} real;
		struct {
			gsl_vector_complex *invec;
			gsl_vector_complex *outvec;
			gsl_matrix_complex *matrix;
		} cplx;
	} workspace;
	gsl_permutation *permutation;
};


/**
 * GSTLALMatrixSolverClass:
 * @parent_class:  the parent class
 */


struct _GSTLALMatrixSolverClass {
	GstBaseTransformClass parent_class;
};


GType gstlal_matrixsolver_get_type(void);


G_END_DECLS


#endif	/* __GSTLAL_MATRIXSOLVER_H__ */

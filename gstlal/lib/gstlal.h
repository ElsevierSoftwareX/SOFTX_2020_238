/*
 * Various bits of LAL wrapped in gstreamer elements.
 *
 * Copyright (C) 2000,2001,2008  Kipp C. Cannon
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


#ifndef __GSTLAL_H__
#define __GSTLAL_H__


#include <glib.h>
#include <gst/gst.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>


#include <lal/LALDatatypes.h>
#include <lal/Units.h>


G_BEGIN_DECLS


/*
 * Hack to work on ancient CentOS
 */


#ifndef G_PARAM_STATIC_STRINGS
#define G_PARAM_STATIC_STRINGS (G_PARAM_STATIC_NAME | G_PARAM_STATIC_NICK | G_PARAM_STATIC_BLURB)
#endif


/*
 * Function prototypes
 */


void gstlal_fftw_lock(void);
void gstlal_fftw_unlock(void);
void gstlal_load_fftw_wisdom(void);


/* int type */
GValueArray *gstlal_g_value_array_from_ints(const gint *, gint);
gint *gstlal_ints_from_g_value_array(GValueArray *, gint *, gint *);
gsl_vector_int *gstlal_gsl_vector_int_from_g_value_array(GValueArray *);
GValueArray *gstlal_g_value_array_from_gsl_vector_int(const gsl_vector_int *);
gsl_matrix_int *gstlal_gsl_matrix_int_from_g_value_array(GValueArray *);
GValueArray *gstlal_g_value_array_from_gsl_matrix_int(const gsl_matrix_int *);

/* long unsigned int type FIXME add vector support */
guint64 *gstlal_uint64s_from_g_value_array(GValueArray *, guint64 *, gint *);
gsl_matrix_ulong *gstlal_gsl_matrix_ulong_from_g_value_array(GValueArray *);
GValueArray *gstlal_g_value_array_from_gsl_matrix_ulong(const gsl_matrix_ulong *);
GValueArray *gstlal_g_value_array_from_uint64s(const guint64 *, gint);


/* double type */
GValueArray *gstlal_g_value_array_from_doubles(const gdouble *, gint);
gdouble *gstlal_doubles_from_g_value_array(GValueArray *, gdouble *, gint *);
gsl_vector *gstlal_gsl_vector_from_g_value_array(GValueArray *);
GValueArray *gstlal_g_value_array_from_gsl_vector(const gsl_vector *);
gsl_matrix *gstlal_gsl_matrix_from_g_value_array(GValueArray *);
GValueArray *gstlal_g_value_array_from_gsl_matrix(const gsl_matrix *);

/* complex type */
gsl_vector_complex *gstlal_gsl_vector_complex_from_g_value_array(GValueArray *);
GValueArray *gstlal_g_value_array_from_gsl_vector_complex(const gsl_vector_complex *);
gsl_matrix_complex *gstlal_gsl_matrix_complex_from_g_value_array(GValueArray *);
GValueArray *gstlal_g_value_array_from_gsl_matrix_complex(const gsl_matrix_complex *);


char *gstlal_build_full_channel_name(const char *, const char *);
REAL8TimeSeries *gstlal_REAL8TimeSeries_from_buffer(GstBuffer *, const char *, const char *, const char *);
LALUnit gstlal_lalStrainPerADCCount(void);
LALUnit gstlal_lalUnitSquaredPerHertz(LALUnit);


G_END_DECLS


#endif	/* __GSTLAL_H__ */

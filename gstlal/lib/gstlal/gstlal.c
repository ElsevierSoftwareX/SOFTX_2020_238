/*
 * Copyright (C) 2008--2013  Kipp Cannon, Chad Hanna
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


/**
 * SECTION:gstlal
 * @title: Misc
 * @include: gstlal/gstlal.h
 * @short_description: Collection of miscellaneous utility functions.
 */


/*
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


/*
 * Stuff from the C library
 */


#include <math.h>
#include <stdio.h>
#include <time.h>


/*
 * Stuff from glib/GStreamer
 */


#include <glib.h>
#include <gst/gst.h>


/*
 * stuff from GSL
 */


#include <gsl/gsl_matrix.h>


/*
 * Stuff from LAL
 */


#include <lal/Date.h>
#include <lal/FrequencySeries.h>
#include <lal/LALDatatypes.h>
#include <lal/Sequence.h>
#include <lal/TimeSeries.h>
#include <lal/Units.h>
#include <lal/XLALError.h>


/*
 * Our own stuff
 */


#include <gstlal.h>


/*
 * ============================================================================
 *
 *                 GSL Matrixes and Vectors and GValueArrays
 *
 * ============================================================================
 */


/**
 * gstlal_ints_from_g_value_array:
 * @va: the #GValueArray from which to copy the elements
 * @dest:  address of memory large enough to hold elements, or %NULL
 * @n:  address of integer that will be set to the number of elements, or
 * %NULL
 *
 * Convert a #GValueArray of ints to an array of ints.  If @dest is %NULL
 * then new memory will be allocated otherwise the ints are copied into the
 * memory pointed to by @dest, which must be large enough to hold them.  If
 * memory is allocated by this function, free with g_free().  If @n is not
 * %NULL it is set to the number of elements in the array.
 *
 * Returns: @dest or the address of the newly-allocated memory on success,
 * or %NULL on failure.
 */


gint *gstlal_ints_from_g_value_array(GValueArray *va, gint *dest, gint *n)
{
	guint i;

	if(!va)
		return NULL;
	if(!dest)
		dest = g_new(gint, va->n_values);
	if(!dest)
		return NULL;
	if(n)
		*n = va->n_values;
	for(i = 0; i < va->n_values; i++)
		dest[i] = g_value_get_int(g_value_array_get_nth(va, i));
	return dest;
}

/**
 * gstlal_g_value_array_from_ints:
 * @src:  start of array
 * @n: number of elements in array
 *
 * Build a #GValueArray from an array of ints.
 *
 * Returns: newly-allocated #GValueArray object or %NULL on failure.
 */


GValueArray *gstlal_g_value_array_from_ints(const gint *src, gint n)
{
	GValueArray *va;
	GValue v = G_VALUE_INIT;
	gint i;
	g_value_init(&v, G_TYPE_INT);

	if(!src)
		return NULL;
	va = g_value_array_new(n);
	if(!va)
		return NULL;
	for(i = 0; i < n; i++) {
		g_value_set_int(&v, src[i]);
		g_value_array_append(va, &v);
	}
	return va;
}


/**
 * gstlal_uint64s_from_g_value_array:
 * @va: the #GValueArray from which to copy the elements
 * @dest:  address of memory large enough to hold elements, or %NULL
 * @n:  address of integer that will be set to the number of elements, or
 * %NULL
 *
 * Convert a #GValueArray of #guint64 to an array of #guint64.  If @dest is
 * %NULL then new memory will be allocated otherwise the doubles are copied
 * into the memory pointed to by @dest, which must be large enough to hold
 * them.  If memory is allocated by this function, free with g_free().  If
 * @n is not %NULL it is set to the number of elements in the array.
 *
 * Returns: @dest or the address of the newly-allocated memory on success,
 * or %NULL on failure.
 */


guint64 *gstlal_uint64s_from_g_value_array(GValueArray *va, guint64 *dest, gint *n)
{
	guint i;

	if(!va)
		return NULL;
	if(!dest)
		dest = g_new(guint64, va->n_values);
	if(!dest)
		return NULL;
	if(n)
		*n = va->n_values;
	for(i = 0; i < va->n_values; i++)
		dest[i] = g_value_get_uint64(g_value_array_get_nth(va, i));
	return dest;
}


/**
 * gstlal_g_value_array_from_uint64s:
 * @src:  start of array
 * @n: number of elements in array
 *
 * Build a #GValueArray from an array of #guint64.
 *
 * Returns: newly-allocated #GValueArray object or %NULL on failure.
 */


GValueArray *gstlal_g_value_array_from_uint64s(const guint64 *src, gint n)
{
	GValueArray *va;
	GValue v = G_VALUE_INIT;
	gint i;
	g_value_init(&v, G_TYPE_UINT64);

	if(!src)
		return NULL;
	va = g_value_array_new(n);
	if(!va)
		return NULL;
	for(i = 0; i < n; i++) {
		g_value_set_uint64(&v, src[i]);
		g_value_array_append(va, &v);
	}
	return va;
}


/**
 * gstlal_doubles_from_g_value_array:
 * @va: the #GValueArray from which to copy the elements
 * @dest:  address of memory large enough to hold elements, or %NULL
 * @n:  address of integer that will be set to the number of elements, or
 * %NULL
 *
 * Convert a #GValueArray of doubles to an array of doubles.  If @dest is
 * %NULL then new memory will be allocated otherwise the doubles are copied
 * into the memory pointed to by @dest, which must be large enough to hold
 * them.  If memory is allocated by this function, free with g_free().  If
 * @n is not %NULL it is set to the number of elements in the array.
 *
 * Returns: @dest or the address of the newly-allocated memory on success,
 * or %NULL on failure.
 */


gdouble *gstlal_doubles_from_g_value_array(GValueArray *va, gdouble *dest, gint *n)
{
	guint i;

	if(!va)
		return NULL;
	if(!dest)
		dest = g_new(gdouble, va->n_values);
	if(!dest)
		return NULL;
	if(n)
		*n = va->n_values;
	for(i = 0; i < va->n_values; i++)
		dest[i] = g_value_get_double(g_value_array_get_nth(va, i));
	return dest;
}


/**
 * gstlal_g_value_array_from_doubles:
 * @src:  start of array
 * @n: number of elements in array
 *
 * Build a #GValueArray from an array of doubles.
 *
 * Returns: newly-allocated #GValueArray object or %NULL on failure.
 */


GValueArray *gstlal_g_value_array_from_doubles(const gdouble *src, gint n)
{
	GValueArray *va;
	GValue v = G_VALUE_INIT;
	gint i;
	g_value_init(&v, G_TYPE_DOUBLE);

	if(!src)
		return NULL;
	va = g_value_array_new(n);
	if(!va)
		return NULL;
	for(i = 0; i < n; i++) {
		g_value_set_double(&v, src[i]);
		g_value_array_append(va, &v);
	}
	return va;
}


/**
 * gstlal_gsl_vector_int_from_g_value_array:
 * @va:  #GValueArray of ints
 *
 * Build a #gsl_vector_int from a #GValueArray of ints.
 *
 * Returns:  the newly-allocated #gsl_vector_int or %NULL on failure.
 */


gsl_vector_int *gstlal_gsl_vector_int_from_g_value_array(GValueArray *va)
{
	gsl_vector_int *vector = gsl_vector_int_alloc(va->n_values);
	if(!vector)
		return NULL;
	if(!gstlal_ints_from_g_value_array(va, gsl_vector_int_ptr(vector, 0), NULL)) {
		gsl_vector_int_free(vector);
		return NULL;
	}
	return vector;
}


/**
 * gstlal_g_value_array_from_gsl_vector_int:
 * @vector:  #gsl_vector_int
 *
 * Build a #GValueArray of ints from a #gsl_vector_int.
 *
 * Returns:  the newly-allocated #GValueArray of ints or %NULL on failure.
 */


GValueArray *gstlal_g_value_array_from_gsl_vector_int(const gsl_vector_int *vector)
{
	return gstlal_g_value_array_from_ints(gsl_vector_int_const_ptr(vector, 0), vector->size);
}


/**
 * gstlal_gsl_vector_from_g_value_array:
 * @va:  #GValueArray of doubles
 *
 * Build a #gsl_vector from a #GValueArray of doubles.
 *
 * Returns:  the newly-allocated #gsl_vector or %NULL on failure.
 */


gsl_vector *gstlal_gsl_vector_from_g_value_array(GValueArray *va)
{
	gsl_vector *vector = gsl_vector_alloc(va->n_values);
	if(!vector)
		return NULL;
	if(!gstlal_doubles_from_g_value_array(va, gsl_vector_ptr(vector, 0), NULL)) {
		gsl_vector_free(vector);
		return NULL;
	}
	return vector;
}


/**
 * gstlal_g_value_array_from_gsl_vector:
 * @vector:  #gsl_vector
 *
 * Build a #GValueArray of doubles from a #gsl_vector.
 *
 * Returns:  the newly-allocated #GValueArray of doubles or %NULL on
 * failure.
 */


GValueArray *gstlal_g_value_array_from_gsl_vector(const gsl_vector *vector)
{
	return gstlal_g_value_array_from_doubles(gsl_vector_const_ptr(vector, 0), vector->size);
}


/**
 * gstlal_gsl_vector_complex_from_g_value_array:
 * @va:  #GValueArray of doubles (two elements per complex number)
 *
 * Build a #gsl_vector_complex from a #GValueArray of doubles packed as
 * real,imag,real,imag,...
 *
 * Returns:  the newly-allocated #gsl_vector_complex or %NULL on failure.
 */


/* FIXME:  update when glib has a complex type */
gsl_vector_complex *gstlal_gsl_vector_complex_from_g_value_array(GValueArray *va)
{
	gsl_vector_complex *vector = gsl_vector_complex_alloc(va->n_values / 2);
	if(!vector)
		return NULL;
	if(!gstlal_doubles_from_g_value_array(va, (double *) gsl_vector_complex_ptr(vector, 0), NULL)) {
		gsl_vector_complex_free(vector);
		return NULL;
	}
	return vector;
}


/**
 * gstlal_g_value_array_from_gsl_vector_complex:
 * @vector:  #gsl_vector_complex
 *
 * Build a #GValueArray of doubles from a #gsl_vector_complex.  The
 * complex numbers are packed into the result as real,imag,real,imag,...
 *
 * Returns:  the newly-allocated #GValueArray of doubles or %NULL on
 * failure.
 */


/* FIXME:  update when glib has a complex type */
GValueArray *gstlal_g_value_array_from_gsl_vector_complex(const gsl_vector_complex *vector)
{
	return gstlal_g_value_array_from_doubles((const double *) gsl_vector_complex_const_ptr(vector, 0), vector->size * 2);
}


/**
 * gstlal_gsl_matrix_int_from_g_value_array:
 * @va:  #GValueArray of #GValueArrays of ints
 *
 * Build a #gsl_matrix_int from a #GValueArray of #GValueArrays of ints.
 *
 * Returns:  the newly-allocated #gsl_matrix_int or %NULL on failure.
 */


gsl_matrix_int *gstlal_gsl_matrix_int_from_g_value_array(GValueArray *va)
{
	gsl_matrix_int *matrix;
	GValueArray *row;
	guint rows, cols;
	guint i;

	if(!va)
		return NULL;
	rows = va->n_values;
	if(!rows)
		/* 0x0 matrix */
		return gsl_matrix_int_alloc(0, 0);

	row = g_value_get_boxed(g_value_array_get_nth(va, 0));
	cols = row->n_values;
	matrix = gsl_matrix_int_alloc(rows, cols);
	if(!matrix)
		/* allocation failure */
		return NULL;
	if(!gstlal_ints_from_g_value_array(row, gsl_matrix_int_ptr(matrix, 0, 0), NULL)) {
		/* row conversion failure */
		gsl_matrix_int_free(matrix);
		return NULL;
	}
	for(i = 1; i < rows; i++) {
		row = g_value_get_boxed(g_value_array_get_nth(va, i));
		if(row->n_values != cols) {
			/* one of the rows has the wrong number of columns */
			gsl_matrix_int_free(matrix);
			return NULL;
		}
		if(!gstlal_ints_from_g_value_array(row, gsl_matrix_int_ptr(matrix, i, 0), NULL)) {
			/* row conversion failure */
			gsl_matrix_int_free(matrix);
			return NULL;
		}
	}

	return matrix;
}


/**
 * gstlal_g_value_array_from_gsl_matrix_int:
 * @matrix:  a #gsl_matrix_int
 *
 * Build a #GValueArray of #GValueArrays of ints from a #gsl_matrix_int.
 *
 * Returns:  the newl-allocated #GValueArray of newly-allocated
 * #GValueArrays of ints or %NULL on failure.
 */


GValueArray *gstlal_g_value_array_from_gsl_matrix_int(const gsl_matrix_int *matrix)
{
	GValueArray *va;
	GValue v = G_VALUE_INIT;
	guint i;
	g_value_init(&v, G_TYPE_VALUE_ARRAY);

	va = g_value_array_new(matrix->size1);
	if(!va)
		return NULL;
	for(i = 0; i < matrix->size1; i++) {
		g_value_take_boxed(&v, gstlal_g_value_array_from_ints(gsl_matrix_int_const_ptr(matrix, i, 0), matrix->size2));
		g_value_array_append(va, &v);
	}
	return va;
}


/**
 * gstlal_gsl_matrix_ulong_from_g_value_array:
 * @va:  #GValueArray of #GValueArrays of #guint64
 *
 * Build a #gsl_matrix_ulong from a #GValueArray of #GValueArrays of
 * #guint64.
 *
 * Returns:  the newly-allocated #gsl_matrix_ulong or %NULL on failure.
 */


gsl_matrix_ulong *gstlal_gsl_matrix_ulong_from_g_value_array(GValueArray *va)
{
	gsl_matrix_ulong *matrix;
	GValueArray *row;
	guint rows, cols;
	guint i;

	if(!va)
		return NULL;
	rows = va->n_values;
	if(!rows)
		/* 0x0 matrix */
		return gsl_matrix_ulong_alloc(0, 0);

	row = g_value_get_boxed(g_value_array_get_nth(va, 0));
	cols = row->n_values;
	matrix = gsl_matrix_ulong_alloc(rows, cols);
	if(!matrix)
		/* allocation failure */
		return NULL;
	if(!gstlal_uint64s_from_g_value_array(row, (guint64 *) gsl_matrix_ulong_ptr(matrix, 0, 0), NULL)) {
		/* row conversion failure */
		gsl_matrix_ulong_free(matrix);
		return NULL;
	}
	for(i = 1; i < rows; i++) {
		row = g_value_get_boxed(g_value_array_get_nth(va, i));
		if(row->n_values != cols) {
			/* one of the rows has the wrong number of columns */
			gsl_matrix_ulong_free(matrix);
			return NULL;
		}
		if(!gstlal_uint64s_from_g_value_array(row, (guint64 *) gsl_matrix_ulong_ptr(matrix, i, 0), NULL)) {
			/* row conversion failure */
			gsl_matrix_ulong_free(matrix);
			return NULL;
		}
	}

	return matrix;
}


/**
 * gstlal_g_value_array_from_gsl_matrix_ulong:
 * @matrix: a #gsl_matrix_ulong
 *
 * Build a #GValueArray of #GValueArrays of #guin64 from a
 * #gsl_matrix_ulong.
 *
 * Returns:  the newly-allocated #GValueArray of newly-allocated
 * #GValueArrays of #guint64s or %NULL on failure.
 */


GValueArray *gstlal_g_value_array_from_gsl_matrix_ulong(const gsl_matrix_ulong *matrix)
{
	GValueArray *va;
	GValue v = G_VALUE_INIT;
	guint i;
	g_value_init(&v, G_TYPE_VALUE_ARRAY);

	va = g_value_array_new(matrix->size1);
	if(!va)
		return NULL;
	for(i = 0; i < matrix->size1; i++) {
		g_value_take_boxed(&v, gstlal_g_value_array_from_uint64s((guint64*) gsl_matrix_ulong_const_ptr(matrix, i, 0), matrix->size2));
		g_value_array_append(va, &v);
	}
	return va;
}


/**
 * gstlal_gsl_matrix_from_g_value_array:
 * @va:  #GValueArray of #GValueArrays of double
 *
 * Build a #gsl_matrix from a #GValueArray of #GValueArrays of doubles.
 *
 * Returns:  the newly-allocated #gsl_matrix or %NULL on failure.
 */


gsl_matrix *gstlal_gsl_matrix_from_g_value_array(GValueArray *va)
{
	gsl_matrix *matrix;
	GValueArray *row;
	guint rows, cols;
	guint i;

	if(!va)
		return NULL;
	rows = va->n_values;
	if(!rows)
		/* 0x0 matrix */
		return gsl_matrix_alloc(0, 0);

	row = g_value_get_boxed(g_value_array_get_nth(va, 0));
	cols = row->n_values;
	matrix = gsl_matrix_alloc(rows, cols);
	if(!matrix)
		/* allocation failure */
		return NULL;
	if(!gstlal_doubles_from_g_value_array(row, gsl_matrix_ptr(matrix, 0, 0), NULL)) {
		/* row conversion failure */
		gsl_matrix_free(matrix);
		return NULL;
	}
	for(i = 1; i < rows; i++) {
		row = g_value_get_boxed(g_value_array_get_nth(va, i));
		if(row->n_values != cols) {
			/* one of the rows has the wrong number of columns */
			gsl_matrix_free(matrix);
			return NULL;
		}
		if(!gstlal_doubles_from_g_value_array(row, gsl_matrix_ptr(matrix, i, 0), NULL)) {
			/* row conversion failure */
			gsl_matrix_free(matrix);
			return NULL;
		}
	}

	return matrix;
}


/**
 * gstlal_g_value_array_from_gsl_matrix:
 * @matrix: a #gsl_matrix
 *
 * Build a #GValueArray of #GValueArrays of doubles from a #gsl_matrix.
 *
 * Returns:  the newly-allocated #GValueArray of newly-allocated
 * #GValueArrays of doubles or %NULL on failure.
 */


GValueArray *gstlal_g_value_array_from_gsl_matrix(const gsl_matrix *matrix)
{
	GValueArray *va;
	GValue v = G_VALUE_INIT;
	guint i;
	g_value_init(&v, G_TYPE_VALUE_ARRAY);

	va = g_value_array_new(matrix->size1);
	if(!va)
		return NULL;
	for(i = 0; i < matrix->size1; i++) {
		g_value_take_boxed(&v, gstlal_g_value_array_from_doubles(gsl_matrix_const_ptr(matrix, i, 0), matrix->size2));
		g_value_array_append(va, &v);
	}
	return va;
}


/**
 * gstlal_gsl_matrix_complex_from_g_value_array:
 * @va:  #GValueArray of #GValueArrays of doubles
 *
 * Build a #gsl_matrix_complex from a #GValueArray of #GValueArrays of
 * doubles.  The complex numbers are unpacked from the doubles as
 * real,imag,real,imag,...
 *
 * Returns:  the newly-allocated #gsl_matrix_complex or %NULL on failure.
 */


/* FIXME:  update when glib has a complex type */
gsl_matrix_complex *gstlal_gsl_matrix_complex_from_g_value_array(GValueArray *va)
{
	gsl_matrix_complex *matrix;
	GValueArray *row;
	guint rows, cols;
	guint i;

	if(!va)
		return NULL;
	rows = va->n_values;
	if(!rows)
		/* 0x0 matrix */
		return gsl_matrix_complex_alloc(0, 0);

	row = g_value_get_boxed(g_value_array_get_nth(va, 0));
	cols = row->n_values;
	matrix = gsl_matrix_complex_alloc(rows, cols / 2);
	if(!matrix)
		/* allocation failure */
		return NULL;
	if(!gstlal_doubles_from_g_value_array(row, (double *) gsl_matrix_complex_ptr(matrix, 0, 0), NULL)) {
		/* row conversion failure */
		gsl_matrix_complex_free(matrix);
		return NULL;
	}
	for(i = 1; i < rows; i++) {
		row = g_value_get_boxed(g_value_array_get_nth(va, i));
		if(row->n_values != cols) {
			/* one of the rows has the wrong number of columns */
			gsl_matrix_complex_free(matrix);
			return NULL;
		}
		if(!gstlal_doubles_from_g_value_array(row, (double *) gsl_matrix_complex_ptr(matrix, i, 0), NULL)) {
			/* row conversion failure */
			gsl_matrix_complex_free(matrix);
			return NULL;
		}
	}

	return matrix;
}


/**
 * gstlal_g_value_array_from_gsl_matrix_complex:
 * @matrix: a #gsl_matrix_complex
 *
 * Build a #GValueArray of #GValueArrays of doubles from a
 * #gsl_matrix_complex.  The complex numbers are packed into the doubles as
 * real,imag,real,imag,...
 *
 * Returns:  the newly-allocated #GValueArray of newly-allocated
 * #GValueArrays of doubles or %NULL on failure.
 */


/* FIXME:  update when glib has a complex type */
GValueArray *gstlal_g_value_array_from_gsl_matrix_complex(const gsl_matrix_complex *matrix)
{
	GValueArray *va;
	GValue v = G_VALUE_INIT;
	guint i;
	g_value_init(&v, G_TYPE_VALUE_ARRAY);

	va = g_value_array_new(matrix->size1);
	if(!va)
		return NULL;
	for(i = 0; i < matrix->size1; i++) {
		g_value_take_boxed(&v, gstlal_g_value_array_from_doubles((const double *) gsl_matrix_complex_const_ptr(matrix, i, 0), matrix->size2 * 2));
		g_value_array_append(va, &v);
	}
	return va;
}


/*
 * ============================================================================
 *
 *                             Utility Functions
 *
 * ============================================================================
 */


/**
 * gstlal_build_full_channel_name:
 * @instrument:  name of instrument (e.g., "H1") or %NULL
 * @channel_name:  name of channel (e.g., "LSC-STRAIN") or %NULL
 *
 * Prefix a channel name with the instrument name.  I.e., turn "H1" and
 * "LSC-STRAIN" into "H1:LSC-STRAIN".  If either instrument or channel_name
 * is %NULL, then the corresponding part of the result is left blank and
 * the colon is omited.  If both are %NULL an empty string is returned.
 *
 * Returns:  newly-allocated string on succes, %NULL on failure.  Free with
 * g_free().
 */


char *gstlal_build_full_channel_name(const char *instrument, const char *channel_name)
{
	char *full_channel_name;
	int len = 2;	/* for ":" and null terminator */

	if(instrument)
		len += strlen(instrument);
	if(channel_name)
		len += strlen(channel_name);

	full_channel_name = g_malloc(len * sizeof(*full_channel_name));
	if(!full_channel_name)
		return NULL;

	snprintf(full_channel_name, len, instrument && channel_name ? "%s:%s" : "%s%s", instrument ? instrument : "", channel_name ? channel_name : "");

	return full_channel_name;
}


/**
 * gstlal_buffer_map_REAL8TimeSeries:
 * @buf:  the #GstBuffer to wrap
 * @caps:  the #GstCaps for the buffer's contents
 * @info:  a #GstMapInfo object to populated with the mapping metadata.  Pass to gstlal_buffer_unmap_REAL8TimeSeries to unmap
 * @instrument:  name of the instrument, e.g., "H1", or %NULL
 * @channel_name:  name of the channel, e.g., "LSC-STRAIN", or %NULL
 * @units:  string describing the units, parsable by XLALParseUnitString()
 *
 * Maps a #GstBuffer read/write and wraps the memory in a #REAL8TimeSeries.
 * The time series's data->data pointer points to the #GstBuffer's own
 * data, so the series cannot be freed using the normal
 * XLALDestroyREAL8TimeSeries() function.  Instead, ues
 * gstlal_buffer_unmap_REAL8TimeSeries() to safely unmap the #GstBuffer and
 * free the #REAL8TimeSeries.  Only #GstBuffers containing single-channel
 * time series data are supported.  The @instrument and @channel_name are
 * used to build the #REAL8TimeSeries' name using
 * gstlal_build_full_channel_name().
 *
 * <example>
 *   <title>Create a REAL8TimeSeries and free it</title>
 *   <programlisting>
 * REAL8TimeSeries *series;
 *
 * series = gstlal_buffer_map_REAL8TimeSeries(buf, &info, "H1", "LSC-STRAIN", "strain");
 * if(!series)
 * 	handle_error();
 * else {
 *	blah_blah_blah();
 *
 * 	gstlal_buffer_unmap_REAL8TimeSeries(buf, &info, series);
 * }
 *   </programlisting>
 * </example>
 *
 * Returns:  newly-allocated REAL8TimeSeries.  Free with
 * gstlal_buffer_unmap_REAL8TimeSeries().
 */


REAL8TimeSeries *gstlal_buffer_map_REAL8TimeSeries(GstBuffer *buf, GstCaps *caps, GstMapInfo *info, const char *instrument, const char *channel_name, const char *units)
{
	GstStructure *structure;
	char *name = NULL;
	gint rate;
	gint channels;
	LALUnit lalunits;
	LIGOTimeGPS epoch;
	REAL8TimeSeries *series = NULL;

	/*
	 * Build the full channel name, parse the units, and retrieve the
	 * sample rate and number of channels from the caps.
	 */

	name = gstlal_build_full_channel_name(instrument, channel_name);
	if(!XLALParseUnitString(&lalunits, units)) {
		GST_ERROR("failure parsing units");
		goto done;
	}
	structure = gst_caps_get_structure(caps, 0);
	if(!structure || !gst_structure_get_int(structure, "rate", &rate) || !gst_structure_get_int(structure, "channels", &channels)) {
		GST_ERROR("cannot extract rate and/or channels from caps");
		goto done;
	}
	if(channels != 1) {
		/* FIXME:  might do something like return an array of time
		 * series? */
		GST_ERROR("cannot wrap multi-channel buffers in LAL time series");
		goto done;
	}

	/*
	 * Retrieve the epoch from the time stamp and the length from the
	 * size.
	 */

	XLALINT8NSToGPS(&epoch, GST_BUFFER_PTS(buf));

	/*
	 * Build a zero-length time series with the correct metadata
	 */

	series = XLALCreateREAL8TimeSeries(name, &epoch, 0.0, 1.0 / rate, &lalunits, 0);
	if(!series) {
		GST_ERROR("XLALCreateREAL8TimeSeries() failed");
		goto done;
	}

	/*
	 * Replace the time series' data pointer with the GstBuffer's, and
	 * manually set the time series' length.
	 */

	XLALFree(series->data->data);
	if(!gst_buffer_map(buf, info, GST_MAP_READWRITE)) {
		GST_ERROR("buffer map failed");
		XLALDestroyREAL8TimeSeries(series);
		series = NULL;
		goto done;
	}
	series->data->data = (double *) (info->data);
	series->data->length = info->size / sizeof(*series->data->data);
	if(info->size % sizeof(*series->data->data)) {
		GST_ERROR("buffer size not an integer multiple of the sample size");
		series->data->data = NULL;
		XLALDestroyREAL8TimeSeries(series);
		series = NULL;
		gst_buffer_unmap(buf, info);
		goto done;
	}

	/*
	 * Done.
	 */

done:
	gst_caps_unref(caps);
	g_free(name);
	return series;
}


/**
 * gstlal_buffer_unmap_REAL8TimeSeries:
 * @buf:  the #GstBuffer to unwrap
 * @info:  a #GstMapInfo object populated with the mapping metadata
 * @series:  the #REAL8TimeSeries wrapping the data to free
 *
 * Safely frees the #REAL8TimeSeries and unmaps the #GstBuffer.
 */


void gstlal_buffer_unmap_REAL8TimeSeries(GstBuffer *buf, GstMapInfo *info, REAL8TimeSeries *series)
{
	if(series) {
		series->data->data = NULL;
		XLALDestroyREAL8TimeSeries(series);
	}
	gst_buffer_unmap(buf, info);
}


/**
 * gstlal_lalUnitSquaredPerHertz:
 *
 * Returns:  #LALUnit equal to "units^2 / Hz".
 */


LALUnit gstlal_lalUnitSquaredPerHertz(LALUnit unit)
{
	return *XLALUnitMultiply(&unit, XLALUnitSquare(&unit, &unit), &lalSecondUnit);
}


/**
 * gstlal_datetime_new_from_gps:
 * @gps:  GPS time
 *
 * Converts a #GstClockTime containing a GPS time (in integer nanoseconds)
 * to a #GstDateTime object representing the same time.
 *
 * Returns:  newly-allocated #GstDateTime.  Use gst_date_time_unref() to
 * free.
 */


GstDateTime *gstlal_datetime_new_from_gps(GstClockTime gps)
{
	struct tm utc;
	XLALGPSToUTC(&utc, GST_TIME_AS_SECONDS(gps));
	return gst_date_time_new(
		0.0,	/* time zone offset */
		1900 + utc.tm_year,
		1 + utc.tm_mon,
		utc.tm_mday,
		utc.tm_hour,
		utc.tm_min,
		utc.tm_sec + (double) (gps % GST_SECOND) / GST_SECOND
	);
}

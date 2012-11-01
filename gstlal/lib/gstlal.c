/*
 * Copyright (C) 2008  Kipp Cannon, Chad Hanna
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
#include <lal/FFTWMutex.h>
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
 *                                Global Data
 *
 * ============================================================================
 */


/*
 * ============================================================================
 *
 *                            FFTW Plan Protection
 *
 * ============================================================================
 */


#ifndef LAL_PTHREAD_LOCK
static GStaticMutex gstlal_fftw_lock_mutex = G_STATIC_MUTEX_INIT;
#endif


void gstlal_fftw_lock(void)
{
#ifdef LAL_PTHREAD_LOCK
	LAL_FFTW_PTHREAD_MUTEX_LOCK;
#else
	g_static_mutex_lock(&gstlal_fftw_lock_mutex);
#endif
}


void gstlal_fftw_unlock(void)
{
#ifdef LAL_PTHREAD_LOCK
	LAL_FFTW_PTHREAD_MUTEX_UNLOCK;
#else
	g_static_mutex_unlock(&gstlal_fftw_lock_mutex);
#endif
}


/*
 * ============================================================================
 *
 *                 GSL Matrixes and Vectors and GValueArrays
 *
 * ============================================================================
 */


/**
 * convert a GValueArray of ints to an array of ints.  if dest is NULL then
 * new memory will be allocated otherwise the ints are copied into the
 * memory pointed to by dest, which must be large enough to hold them.  the
 * return value is dest or the newly allocated address on success or NULL
 * on failure.
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
 * convert an array of ints to a GValueArray.  the return value is the
 * newly allocated GValueArray object.
 */


GValueArray *gstlal_g_value_array_from_ints(const gint *src, gint n)
{
	GValueArray *va;
	GValue v = {0,};
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
 * convert a GValueArray of guint64 to an array of guint64.  if dest is
 * NULL then new memory will be allocated otherwise the doubles are copied
 * into the memory pointed to by dest, which must be large enough to hold
 * them.  the return value is dest or the newly allocated address on
 * success or NULL on failure.
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
 * convert an array of guint64 to a GValueArray.  the return value is the
 * newly allocated GValueArray object.
 */


GValueArray *gstlal_g_value_array_from_uint64s(const guint64 *src, gint n)
{
	GValueArray *va;
	GValue v = {0,};
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
 * convert a GValueArray of doubles to an array of doubles.  if dest is
 * NULL then new memory will be allocated otherwise the doubles are copied
 * into the memory pointed to by dest, which must be large enough to hold
 * them.  the return value is dest or the newly allocated address on
 * success or NULL on failure.
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
 * convert an array of doubles to a GValueArray.  the return value is the
 * newly allocated GValueArray object.
 */


GValueArray *gstlal_g_value_array_from_doubles(const gdouble *src, gint n)
{
	GValueArray *va;
	GValue v = {0,};
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
 * convert a GValueArray of ints to a GSL vector.  the return value is the
 * newly allocated vector on success or NULL on failure.
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
 * convert a GSL vector of ints to a GValueArray.  the return value is the
 * newly allocated GValueArray object.
 */


GValueArray *gstlal_g_value_array_from_gsl_vector_int(const gsl_vector_int *vector)
{
	return gstlal_g_value_array_from_ints(gsl_vector_int_const_ptr(vector, 0), vector->size);
}


/**
 * convert a GValueArray of doubles to a GSL vector.  the return value is
 * the newly allocated vector on success or NULL on failure.
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
 * convert a GSL vector of doubles to a GValueArray.  the return value is
 * the newly allocated GValueArray object.
 */


GValueArray *gstlal_g_value_array_from_gsl_vector(const gsl_vector *vector)
{
	return gstlal_g_value_array_from_doubles(gsl_vector_const_ptr(vector, 0), vector->size);
}


/**
 * convert a GValueArray of complex doubles to a GSL vector.  the return
 * value is the newly allocated vector on success or NULL on failure.
 * Note: glib/gobject don't support complex floats, so the data are assumed
 * to be stored as a GValueArray of doubles packed as
 * real,imag,real,imag,...
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
 * convert a GSL vector of complex doubles to a GValueArray.  the return
 * value is the newly allocated GValueArray object.  Note:  glib/gobject
 * don't support complex floats, so the data are assumed to be stored as a
 * GValueArray of doubles packed as real,imag,real,imag,...
 */


/* FIXME:  update when glib has a complex type */
GValueArray *gstlal_g_value_array_from_gsl_vector_complex(const gsl_vector_complex *vector)
{
	return gstlal_g_value_array_from_doubles((const double *) gsl_vector_complex_const_ptr(vector, 0), vector->size * 2);
}


/**
 * convert a GValueArray of GValueArrays of ints to a GSL matrix.  the
 * return value is the newly allocated matrix on success or NULL on
 * failure.
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
 * convert a GSL matrix of ints to a GValueArray of GValueArrays.  the
 * return value is the newly allocated GValueArray object.
 */


GValueArray *gstlal_g_value_array_from_gsl_matrix_int(const gsl_matrix_int *matrix)
{
	GValueArray *va;
	GValue v = {0,};
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
 * convert a GValueArray of GValueArrays of guint64 to a GSL matrix.  the
 * return value is the newly allocated matrix on success or NULL on
 * failure.
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
 * convert a GValueArray of GValueArrays of doubles to a GSL matrix.  the
 * return value is the newly allocated matrix on success or NULL on
 * failure.
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
 * convert a GSL matrix of ulong to a GValueArray of GValueArrays.  the
 * return value is the newly allocated GValueArray object.
 */


GValueArray *gstlal_g_value_array_from_gsl_matrix_ulong(const gsl_matrix_ulong *matrix)
{
	GValueArray *va;
	GValue v = {0,};
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
 * convert a GSL matrix of doubles to a GValueArray of GValueArrays.  the
 * return value is the newly allocated GValueArray object.
 */


GValueArray *gstlal_g_value_array_from_gsl_matrix(const gsl_matrix *matrix)
{
	GValueArray *va;
	GValue v = {0,};
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
 * convert a GValueArray of GValueArrays of doubles to a GSL complex
 * matrix.  the return value is the newly allocated matrix on success or
 * NULL on failure.  Note:  glib/gobject don't support complex floats, so
 * the data are assumed to be stored as rows (GValueArrays) packed as
 * real,imag,real,imag,...
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
 * convert a GSL matrix of complex doubles to a GValueArray of
 * GValueArrays.  the return value is the newly allocated GValueArray
 * object.  Note: glib/gobject don't support complex floats, so the data
 * are assumed to be stored as rows (GValueArrays) packed as
 * real,imag,real,imag,...
 */


/* FIXME:  update when glib has a complex type */
GValueArray *gstlal_g_value_array_from_gsl_matrix_complex(const gsl_matrix_complex *matrix)
{
	GValueArray *va;
	GValue v = {0,};
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
 * Prefix a channel name with the instrument name.  I.e., turn "H1" and
 * "LSC-STRAIN" into "H1:LSC-STRAIN".  If either instrument or channel_name
 * is NULL, then the corresponding part of the result is left blank and the
 * colon is omited.  The return value is NULL on failure or a
 * newly-allocated string.   The calling code should g_free() the string
 * when finished with it.
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
 * Wrap a GstBuffer in a REAL8TimeSeries.  The time series's data->data
 * pointer points to the GstBuffer's own data, so this pointer must not be
 * free()ed.  That means it must be set to NULL before passing the time
 * series to the XLAL destroy function.
 *
 * Example:
 *
 * REAL8TimeSeries *series;
 *
 * series = gstlal_REAL8TimeSeries_from_buffer(buf);
 * if(!series)
 * 	handle_error();
 *
 * blah_blah_blah();
 *
 * series->data->data = NULL;
 * XLALDestroyREAL8TimeSeries(series);
 */


REAL8TimeSeries *gstlal_REAL8TimeSeries_from_buffer(GstBuffer *buf, const char *instrument, const char *channel_name, const char *units)
{
	GstCaps *caps = gst_buffer_get_caps(buf);
	GstStructure *structure;
	char *name = NULL;
	gint rate;
	gint channels;
	double deltaT;
	LALUnit lalunits;
	LIGOTimeGPS epoch;
	size_t length;
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
	if(!gst_structure_get_int(structure, "rate", &rate) || !gst_structure_get_int(structure, "channels", &channels)) {
		GST_ERROR("cannot extract rate and/or channels from caps");
		goto done;
	}
	if(channels != 1) {
		/* FIXME:  might do something like return an array of time
		 * series? */
		GST_ERROR("cannot wrap multi-channel buffers in LAL time series");
		goto done;
	}
	deltaT = 1.0 / rate;

	/*
	 * Retrieve the epoch from the time stamp and the length from the
	 * size.
	 */

	XLALINT8NSToGPS(&epoch, GST_BUFFER_TIMESTAMP(buf));
	length = GST_BUFFER_SIZE(buf) / sizeof(*series->data->data) / channels;
	if(channels * length * sizeof(*series->data->data) != GST_BUFFER_SIZE(buf)) {
		GST_ERROR("buffer size not an integer multiple of the sample size");
		goto done;
	}

	/*
	 * Build a zero-length time series with the correct metadata
	 */

	series = XLALCreateREAL8TimeSeries(name, &epoch, 0.0, deltaT, &lalunits, 0);
	if(!series) {
		GST_ERROR("XLALCreateREAL8TimeSeries() failed");
		goto done;
	}

	/*
	 * Replace the time series' data pointer with the GstBuffer's, and
	 * manually set the time series' length.
	 */

	XLALFree(series->data->data);
	series->data->data = (double *) GST_BUFFER_DATA(buf);
	series->data->length = length;

	/*
	 * Done.
	 */

done:
	gst_caps_unref(caps);
	g_free(name);
	return series;
}


/**
 * Return a LALUnit structure equal to "strain / ADC count".
 */


LALUnit gstlal_lalStrainPerADCCount(void)
{
	LALUnit unit;

	return *XLALUnitDivide(&unit, &lalStrainUnit, &lalADCCountUnit);
}


/**
 * Return a LALUnit structure equal to "units^2 / Hz".
 */


LALUnit gstlal_lalUnitSquaredPerHertz(LALUnit unit)
{
	return *XLALUnitMultiply(&unit, XLALUnitSquare(&unit, &unit), &lalSecondUnit);
}

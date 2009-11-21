/*
 * Various bits of LAL wrapped in gstreamer elements.
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
 * Stuff from GStreamer
 */


#include <gst/gst.h>


/*
 * stuff from GSL
 */


#include <gsl/gsl_matrix.h>
#include <gsl/gsl_spline.h>


/*
 * Stuff from LAL
 */


#include <lal/Date.h>
#include <lal/LALDatatypes.h>
#include <lal/Sequence.h>
#include <lal/TimeSeries.h>
#include <lal/FrequencySeries.h>
#include <lal/Units.h>
#include <lal/XLALError.h>


/*
 * Our own stuff
 */


#include <gstlal.h>
#include <gstlal_framesrc.h>
#include <gstlal_templatebank.h>
#include <gstlal_matrixmixer.h>
#include <gstlal_simulation.h>
#include <gstlal_whiten.h>
#include <gstlal_nxydump.h>
#include <gstadder.h>
#include <gstlal_triggergen.h>
#include <gstlal_gate.h>
#include <gstlal_chisquare.h>
#include <gstlal_autochisq.h>
#include <gstlal_firbank.h>
#include <gstlal_sumsquares.h>
#include <gstlal_togglecomplex.h>


/*
 * ============================================================================
 *
 *                                    Data
 *
 * ============================================================================
 */


GMutex *gstlal_fftw_lock;


/*
 * ============================================================================
 *
 *                          Messages and Properties
 *
 * ============================================================================
 */


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
 * convert a GValueArray to a GSL vector.  the return value is the newly
 * allocated vector on success or NULL on failure.
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
 * convert a gsl_vector to a GValueArray of doubles.  the return value is
 * the newly allocated GValueArray object.
 */


GValueArray *gstlal_g_value_array_from_gsl_vector(const gsl_vector *vector)
{
	return gstlal_g_value_array_from_doubles(gsl_vector_const_ptr(vector, 0), vector->size);
}


/**
 * convert a GValueArray to a GSL complex vector.  the return value is the
 * newly allocated vector on success or NULL on failure.  Note:
 * glib/gobject don't support complex floats, so the data is assumed to be
 * stored as a GValueArray of doubles packed as real,imag,real,imag,...
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
 * convert a gsl_vector_complex to a GValueArray of doubles.  the return
 * value is the newly allocated GValueArray object.  Note:  glib/gobject
 * don't support complex floats, so the data is assumed to be stored as a
 * GValueArray of doubles packed as real,imag,real,imag,...
 */


/* FIXME:  update when glib has a complex type */
GValueArray *gstlal_g_value_array_from_gsl_vector_complex(const gsl_vector_complex *vector)
{
	return gstlal_g_value_array_from_doubles((const double *) gsl_vector_complex_const_ptr(vector, 0), vector->size * 2);
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
 * convert a gsl_matrix to a GValueArray of GValueArray of doubles.  the
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
 * the data is assumed to be stored as rows (GValueArrays) packed as
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
 * convert a gsl_matrix_complex to a GValueArray of GValueArray of doubles.
 * the return value is the newly allocated GValueArray object.  Note:
 * glib/gobject don't support complex floats, so the data is assumed to be
 * stored as rows (GValueArrays) packed as real,imag,real,imag,...
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
 * newly-allocated string.   The calling code should free() the string when
 * finished with it.
 */


char *gstlal_build_full_channel_name(const char *instrument, const char *channel_name)
{
	char *full_channel_name;
	int len = 2;	/* for ":" and null terminator */

	if(instrument)
		len += strlen(instrument);
	if(channel_name)
		len += strlen(channel_name);

	full_channel_name = malloc(len * sizeof(*full_channel_name));
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
	free(name);
	return series;
}


/**
 * Return the LALUnit structure equal to "strain / ADC count".
 */


LALUnit gstlal_lalStrainPerADCCount(void)
{
	LALUnit unit;

	return *XLALUnitDivide(&unit, &lalStrainUnit, &lalADCCountUnit);
}


/**
 * Return the LALUnit structure equal to "units^2 / Hz".
 */


LALUnit gstlal_lalUnitSquaredPerHertz(LALUnit unit)
{
	return *XLALUnitMultiply(&unit, XLALUnitSquare(&unit, &unit), &lalSecondUnit);
}


/**
 * Read a PSD from a spectrum file.  Ugly file format, but if everything
 * uses *this* function to read it, we can switch to something more cool
 * later.
 */


REAL8FrequencySeries *gstlal_read_reference_psd(const char *filename)
{
	LIGOTimeGPS gps_zero = {0, 0};
	LALUnit strain_squared_per_hertz = gstlal_lalUnitSquaredPerHertz(lalStrainUnit);
	REAL8FrequencySeries *psd;
	FILE *file;
	unsigned i;

	/*
	 * open the psd file
	 */

	file = fopen(filename, "r");
	if(!file) {
		perror("gstlal_read_reference_psd()");
		GST_ERROR("fopen(\"%s\") failed", filename);
		return NULL;
	}

	/*
	 * allocate a frequency series
	 */

	psd = XLALCreateREAL8FrequencySeries("PSD", &gps_zero, 0.0, 0.0, &strain_squared_per_hertz, 0);
	if(!psd) {
		GST_ERROR("XLALCreateREAL8FrequencySeries() failed");
		fclose(file);
		return NULL;
	}

	/*
	 * read the psd into the frequency series one sample at a time
	 */

	for(i = 0; 1; i++) {
		int result;
		double f, amp;

		/*
		 * parse f and one psd sample from one line of input text
		 */

		result = fscanf(file, " %lg %lg", &f, &amp);
		if(result == EOF || result < 2) {
			if(feof(file))
				/*
				 * eof == done w/ success
				 */
				break;
			if(result < 0)
				/*
				 * I/O error of some kind
				 */
				perror("gstlal_read_reference_psd()");
			else
				/*
				 * no errors, but couldn't parse file
				 */
				GST_ERROR("unable to parse \"%s\"", filename);
			fclose(file);
			XLALDestroyREAL8FrequencySeries(psd);
			return NULL;
		}

		/*
		 * store in frequency series, replacing any infs with 0
		 */

		if(!XLALResizeREAL8Sequence(psd->data, 0, i + 1)) {
			GST_ERROR("XLALResizeREAL8Sequence() failed");
			fclose(file);
			XLALDestroyREAL8FrequencySeries(psd);
			return NULL;
		}

		psd->data->data[i] = isinf(amp) ? 0 : amp;

		/*
		 * update the metadata
		 */

		if(i == 0)
			psd->f0 = f;
		else
			psd->deltaF = (f - psd->f0) / i;
	}

	/*
	 * done
	 */

	fclose(file);

	return psd;
}


/**
 * Retrieve a PSD from a spectrum file, and re-interpolate to the desired
 * frequency band and resolution.
 */


REAL8FrequencySeries *gstlal_get_reference_psd(const char *filename, double f0, double deltaF, size_t length)
{
	REAL8FrequencySeries *psd;
	double *f;
	gsl_spline *spline;
	gsl_interp_accel *accel;
	unsigned i;

	/*
	 * load the reference PSD
	 */

	psd = gstlal_read_reference_psd(filename);
	if(!psd)
		return NULL;

	/*
	 * feelin' lucky?
	 */

	if(psd->f0 == f0 && psd->deltaF == deltaF && psd->data->length == length)
		return psd;

	/*
	 * construct an interpolator
	 */

	f = malloc(psd->data->length * sizeof(*f));
	spline = gsl_spline_alloc(gsl_interp_linear, psd->data->length);
	accel = gsl_interp_accel_alloc();
	if(!f || !spline || !accel) {
		GST_ERROR("gsl_spline_alloc() or gsl_interp_accel_alloc() failed");
		XLALDestroyREAL8FrequencySeries(psd);
		free(f);
		if(spline)
			gsl_spline_free(spline);
		if(accel)
			gsl_interp_accel_free(accel);
		return NULL;
	}
	for(i = 0; i < psd->data->length; i++)
		f[i] = psd->f0 + i * psd->deltaF;
	if(gsl_spline_init(spline, f, psd->data->data, psd->data->length)) {
		XLALDestroyREAL8FrequencySeries(psd);
		free(f);
		gsl_spline_free(spline);
		gsl_interp_accel_free(accel);
		return NULL;
	}

	/*
	 * repopulate reference PSD from interpolator to match desired
	 * resolution and size
	 *
	 * FIXME:  what if part of the desired frequency band lies outside
	 * the reference spectrum loaded from the file?
	 */

	if(!XLALResizeREAL8Sequence(psd->data, 0, length)) {
		XLALDestroyREAL8FrequencySeries(psd);
		free(f);
		gsl_spline_free(spline);
		gsl_interp_accel_free(accel);
		return NULL;
	}
	for(i = 0; i < psd->data->length; i++)
		psd->data->data[i] = gsl_spline_eval(spline, f0 + i * deltaF, accel);

	/*
	 * adjust normalization for the new bin size, then update the
	 * metadata
	 */

	for(i = 0; i < psd->data->length; i++)
		psd->data->data[i] *= deltaF / psd->deltaF;
	psd->f0 = f0;
	psd->deltaF = deltaF;

	/*
	 * done
	 */

	free(f);
	gsl_spline_free(spline);
	gsl_interp_accel_free(accel);
	return psd;
}


/*
 * ============================================================================
 *
 *                             Plugin Entry Point
 *
 * ============================================================================
 */


static gboolean plugin_init(GstPlugin *plugin)
{
	struct {
		const gchar *name;
		GType type;
	} *element, elements[] = {
		{"lal_framesrc", GSTLAL_FRAMESRC_TYPE},
		{"lal_templatebank", GSTLAL_TEMPLATEBANK_TYPE},
		{"lal_matrixmixer", GSTLAL_MATRIXMIXER_TYPE},
		{"lal_simulation", GSTLAL_SIMULATION_TYPE},
		{"lal_whiten", GSTLAL_WHITEN_TYPE},
		{"lal_nxydump", GSTLAL_NXYDUMP_TYPE},
		{"lal_adder", GST_TYPE_ADDER},
		{"lal_triggergen", GSTLAL_TRIGGERGEN_TYPE},
		{"lal_triggerxmlwriter", GSTLAL_TRIGGERXMLWRITER_TYPE},
		{"lal_gate", GSTLAL_GATE_TYPE},
		{"lal_chisquare", GSTLAL_CHISQUARE_TYPE},
		{"lal_autochisq", GSTLAL_AUTOCHISQ_TYPE},
		{"lal_firbank", GSTLAL_FIRBANK_TYPE},
		{"lal_sumsquares", GSTLAL_SUMSQUARES_TYPE},
		{"lal_togglecomplex", GSTLAL_TOGGLECOMPLEX_TYPE},
		{NULL, 0},
	};
	struct {
		const gchar *name;
		GstTagFlag flag;
		GType type;
		const gchar *nick;
		const gchar *blurb;
		GstTagMergeFunc func;
	} *tagarg, tagargs[] = {
		{GSTLAL_TAG_INSTRUMENT, GST_TAG_FLAG_META, G_TYPE_STRING, "instrument", "The short name of the instrument or observatory where this data was recorded, e.g., \"H1\"", gst_tag_merge_strings_with_comma},
		{GSTLAL_TAG_CHANNEL_NAME, GST_TAG_FLAG_META, G_TYPE_STRING, "channel name", "The name of this channel, e.g., \"LSC-STRAIN\"", gst_tag_merge_strings_with_comma},
		{GSTLAL_TAG_UNITS, GST_TAG_FLAG_META, G_TYPE_STRING, "units", "The units for this channel (as encoded by LAL), e.g., \"strain\".", NULL},
		{NULL,},
	};

	/*
	 * Set the LAL debug level.
	 */

	lalDebugLevel = LALINFO | LALWARNING | LALERROR | LALNMEMDBG | LALNMEMPAD | LALNMEMTRK;
	XLALSetSilentErrorHandler();

	/*
	 * Initialize the mutices.
	 */

	gstlal_fftw_lock = g_mutex_new();

	/*
	 * Tell GStreamer about the elements.
	 */

	for(element = elements; element->name; element++)
		if(!gst_element_register(plugin, element->name, GST_RANK_NONE, element->type))
			return FALSE;

	/*
	 * Tell GStreamer about the custom tags.
	 */

	for(tagarg = tagargs; tagarg->name; tagarg++)
		gst_tag_register(tagarg->name, tagarg->flag, tagarg->type, tagarg->nick, tagarg->blurb, tagarg->func);

	/*
	 * Done.
	 */

	return TRUE;
}


/*
 * This is the structure that gst-register looks for.
 */


GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, "gstlal", "Various bits of the LIGO Algorithm Library wrapped in gstreamer elements", plugin_init, PACKAGE_VERSION, "GPL", PACKAGE_NAME, "http://www.lsc-group.phys.uwm.edu/daswg")

/*
 * Copyright (C) 2018  Aaron Viets <aaron.viets@ligo.org>
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
 * SECTION:gstlal_transferfunction
 * @short_description:  Compute transfer functions between two or more
 * time series.
 */


/*
 * ========================================================================
 *
 *				  Preamble
 *
 * ========================================================================
 */


/*
 * stuff from the C library
 */


#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>


/*
 * stuff from glib/gstreamer
 */


#include <glib.h>
#include <glib/gprintf.h>
#include <gst/gst.h>
#include <gst/base/gstbasesink.h>
#include <gst/audio/audio.h>
#include <gst/audio/audio-format.h>


/*
 * stuff from FFTW and GSL
 */


#include <fftw3.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_permutation.h>


/*
 * our own stuff
 */


#include <gstlal/gstlal.h>
#include <gstlal/gstlal_audio_info.h>
#include <gstlal/gstlal_debug.h>
#include <gstlal-calibration/gstlal_firtools.h>
#include <gstlal_transferfunction.h>


#define SINC_LENGTH 25


/*
 * ============================================================================
 *
 *			      Custom Types
 *
 * ============================================================================
 */


/*
 * window type enum
 */


GType gstlal_transferfunction_window_get_type(void) {

	static GType type = 0;

	if(!type) {
		static GEnumValue values[] = {
			{GSTLAL_TRANSFERFUNCTION_DPSS, "GSTLAL_TRANSFERFUNCTION_DPSS", "Maximize energy concentration in main lobe"},
			{GSTLAL_TRANSFERFUNCTION_KAISER, "GSTLAL_TRANSFERFUNCTION_KAISER", "Simple approximtion to DPSS window"},
			{GSTLAL_TRANSFERFUNCTION_DOLPH_CHEBYSHEV, "GSTLAL_TRANSFERFUNCTION_DOLPH_CHEBYSHEV", "Attenuate all side lobes equally"},
			{0, NULL, NULL}
		};

		type = g_enum_register_static("GSTLAL_TRANSFERFUNCTION_WINDOW", values);
	}

	return type;
}


/*
 * ============================================================================
 *
 *			   GStreamer Boilerplate
 *
 * ============================================================================
 */


#define GST_CAT_DEFAULT gstlal_transferfunction_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


G_DEFINE_TYPE_WITH_CODE(
	GSTLALTransferFunction,
	gstlal_transferfunction,
	GST_TYPE_BASE_SINK,
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "lal_transferfunction", 0, "lal_transferfunction element")
);


enum property {
	ARG_FFT_LENGTH = 1,
	ARG_FFT_OVERLAP,
	ARG_NUM_FFTS,
	ARG_MIN_FFTS,
	ARG_USE_MEDIAN,
	ARG_UPDATE_SAMPLES,
	ARG_UPDATE_AFTER_GAP,
	ARG_USE_FIRST_AFTER_GAP,
	ARG_UPDATE_DELAY_SAMPLES,
	ARG_PARALLEL_MODE,
	ARG_WRITE_TO_SCREEN,
	ARG_FILENAME,
	ARG_MAKE_FIR_FILTERS,
	ARG_FIR_LENGTH,
	ARG_FREQUENCY_RESOLUTION,
	ARG_HIGH_PASS,
	ARG_LOW_PASS,
	ARG_NOTCH_FREQUENCIES,
	ARG_FIR_TIMESHIFT,
	ARG_TRANSFER_FUNCTIONS,
	ARG_FIR_FILTERS,
	ARG_FIR_ENDTIME,
	ARG_WINDOW,
	ARG_FAKE
};


static GParamSpec *properties[ARG_FAKE];


/*
 * ============================================================================
 *
 *				  Utilities
 *
 * ============================================================================
 */


static void rebuild_workspace_and_reset(GObject *object)
{
	return;
}


double minimum(double value1, double value2) {
	return value1 < value2 ? value1 : value2;
}


double maximum(double value1, double value2) {
	return value1 > value2 ? value1 : value2; \
}


#define DEFINE_MINIMUM(size) \
gint ## size minimum ## size(gint ## size value1, gint ## size value2) { \
	return value1 < value2 ? value1 : value2; \
}


DEFINE_MINIMUM(8);
DEFINE_MINIMUM(16);
DEFINE_MINIMUM(32);
DEFINE_MINIMUM(64);


#define DEFINE_MAXIMUM(size) \
gint ## size maximum ## size(gint ## size value1, gint ## size value2) { \
	return value1 > value2 ? value1 : value2; \
}


DEFINE_MAXIMUM(8);
DEFINE_MAXIMUM(16);
DEFINE_MAXIMUM(32);
DEFINE_MAXIMUM(64);


static void write_transfer_functions(complex double *tfs, char *element_name, double df, gint64 rows, int columns, double t_start, double t_finish, gboolean write_to_screen, char *filename, gboolean free_name) {
	gint64 i;
	int j, j_stop;
	if(write_to_screen) {
		g_print("\n\n==================== Transfer functions computed by %s from %f until %f ====================\nfrequency\t\t  ", element_name, t_start, t_finish);
		for(j = 1; j < columns; j++)
			g_print("ch%d -> ch0\t\t\t\t  ", j);
		g_print("ch%d -> ch0\n\n", columns);

		j_stop = columns - 1;
		for(i = 0; i < rows; i++) {
			g_print("%10f\t", i * df);
			for(j = 0; j < j_stop; j++) {
				if(cimag(tfs[i + j * rows]) < 0.0)
					g_print("%10e - %10ei\t\t", creal(tfs[i + j * rows]), -cimag(tfs[i + j * rows]));
				else
					g_print("%10e + %10ei\t\t", creal(tfs[i + j * rows]), cimag(tfs[i + j * rows]));
			}
			if(cimag(tfs[i + j_stop * rows]) < 0.0)
				g_print("%10e - %10ei\n", creal(tfs[i + j_stop * rows]), -cimag(tfs[i + j_stop * rows]));
			else
				g_print("%10e + %10ei\n", creal(tfs[i + j_stop * rows]), cimag(tfs[i + j_stop * rows]));
		}
		g_print("\n\n");
	}

	if(filename) {
		FILE *fp;
		fp = fopen(filename, "a");
		g_fprintf(fp, "==================== Transfer functions computed by %s from %f until %f ====================\nfrequency\t\t  ", element_name, t_start, t_finish);
		for(j = 1; j < columns; j++)
			g_fprintf(fp, "ch%d -> ch0\t\t\t\t  ", j);
		g_fprintf(fp, "ch%d -> ch0\n\n", columns);

		j_stop = columns - 1;
		for(i = 0; i < rows; i++) {
			g_fprintf(fp, "%10f\t", i * df);
			for(j = 0; j < j_stop; j++) {
				if(cimag(tfs[i + j * rows]) < 0.0)
					g_fprintf(fp, "%10e - %10ei\t\t", creal(tfs[i + j * rows]), -cimag(tfs[i + j * rows]));
				else
					g_fprintf(fp, "%10e + %10ei\t\t", creal(tfs[i + j * rows]), cimag(tfs[i + j * rows]));
			}
			if(cimag(tfs[i + j_stop * rows]) < 0.0)
				g_fprintf(fp, "%10e - %10ei\n", creal(tfs[i + j_stop * rows]), -cimag(tfs[i + j_stop * rows]));
			else
				g_fprintf(fp, "%10e + %10ei\n", creal(tfs[i + j_stop * rows]), cimag(tfs[i + j_stop * rows]));
		}
		g_fprintf(fp, "\n\n");
		fclose(fp);
	}
	if(free_name)
		g_free(element_name);
}


static void write_fir_filters(double *filters, char *element_name, gint64 rows, int columns, double t_start, double t_finish, gboolean write_to_screen, char *filename, gboolean free_name) {
	gint64 i;
	int j, j_stop;
	if(write_to_screen) {
		g_print("================== FIR filters computed by %s from %f until %f ==================\n", element_name, t_start, t_finish);
		for(j = 1; j < columns; j++)
			g_print("ch%d -> ch0\t", j);
		g_print("ch%d -> ch0\n\n", columns);

		j_stop = columns - 1;
		for(i = 0; i < rows; i++) {
			for(j = 0; j < j_stop; j++)
				g_print("%10e\t", filters[i + j * rows]);
			g_print("%10e\n", filters[i + j_stop * rows]);
		}
		g_print("\n\n");
	}

	if(filename) {
		FILE *fp;
		fp = fopen(filename, "a");
		g_fprintf(fp, "================== FIR filters computed by %s from %f until %f ==================\n", element_name, t_start, t_finish);
		for(j = 1; j < columns; j++)
			g_fprintf(fp, "ch%d -> ch0\t", j);
		g_fprintf(fp, "ch%d -> ch0\n\n", columns);

		j_stop = columns - 1;
		for(i = 0; i < rows; i++) {
			for(j = 0; j < j_stop; j++)
				g_fprintf(fp, "%10e\t", filters[i + j * rows]);
			g_fprintf(fp, "%10e\n", filters[i + j_stop * rows]);
		}
		g_fprintf(fp, "\n\n");
		fclose(fp);
	}
	if(free_name)
		g_free(element_name);
}


#define DEFINE_UPDATE_MEDIAN(DTYPE) \
static void update_median_ ## DTYPE(DTYPE *median_array, DTYPE new_element, gint64 array_size, gint64 num_in_median, gint64 *index_median) { \
	gint64 i; \
	if(!num_in_median) { \
		/* This will be the first element in the array */ \
		*median_array = new_element; \
		*index_median = 0; \
 \
	} else if(num_in_median < array_size) { \
		/* If the array is not full, start at the last value and work down */ \
		i = num_in_median; \
		while(i && new_element < median_array[i > 1 ? i - 1 : 0]) { \
			median_array[i] = median_array[i - 1]; \
			i--; \
		} \
		median_array[i] = new_element; \
		/* Reset the location of the median in the array. Note that if the array size is even, we choose the later/larger value. */ \
		*index_median = (num_in_median + 1) / 2; \
 \
	} else if(new_element > median_array[array_size / 2]) { \
		/* If the array is full and the new value is larger than the central value, start at the end and work down */ \
		if(new_element < median_array[array_size - 1]) { \
			i = array_size - 2; \
			while(new_element < median_array[i]) { \
				median_array[i + 1] = median_array[i]; \
				i--; \
			} \
			median_array[i + 1] = new_element; \
		} \
		/* We want to shift the location of the median up one slot if the new number of samples is even */ \
		if(num_in_median % 2) \
			(*index_median)++; \
	} else { \
		/* The array is full and the new value is smaller than the central value, so we start at the beginning */ \
		if(new_element > *median_array) { \
			i = 1; \
			while(new_element > median_array[i]) { \
				median_array[i - 1] = median_array[i]; \
				i++; \
			} \
			median_array[i - 1] = new_element; \
		} \
		/* We want to shift the location of the median down one slot if the new number of samples is odd */ \
		if(!(num_in_median % 2)) \
			(*index_median)--; \
	} \
 \
	return; \
}


DEFINE_UPDATE_MEDIAN(float);
DEFINE_UPDATE_MEDIAN(double);


#define DEFINE_UPDATE_TRANSFER_FUNCTIONS(DTYPE, F_OR_BLANK) \
static gboolean update_transfer_functions_ ## DTYPE(complex DTYPE *autocorrelation_matrix, int num_tfs, gint64 fd_fft_length, gint64 fd_tf_length, DTYPE *sinc_table, gint64 sinc_length, gint64 sinc_taps_per_df, gint64 num_avg, gsl_vector_complex *transfer_functions_at_f, gsl_vector_complex *transfer_functions_solved_at_f, gsl_matrix_complex *autocorrelation_matrix_at_f, gsl_permutation *permutation, complex double *transfer_functions) { \
 \
	gboolean success = TRUE; \
	gint64 i, sinc_taps_per_input, sinc_taps_per_output, k, k_stop, input0, sinc0; \
	int j, j_stop, elements_per_freq, signum; \
	complex DTYPE z; \
	gsl_complex gslz; \
	/*
	 * We may need to resample and/or low-pass filter the transfer functions so they have
	 * the right length and frequency resolution. First, calculate the number of taps of the
	 * sinc filter that corresponds to one increment in the input and output. 
	 */ \
	sinc_taps_per_input = fd_tf_length > fd_fft_length ? sinc_taps_per_df * (fd_tf_length - 1) / (fd_fft_length - 1) : sinc_taps_per_df; \
	sinc_taps_per_output = fd_tf_length > fd_fft_length ? sinc_taps_per_df : sinc_taps_per_df * (fd_fft_length - 1) / (fd_tf_length - 1); \
	elements_per_freq = num_tfs * (1 + num_tfs); \
	for(i = 0; i < fd_tf_length; i++) { \
		/* First, copy samples at a specific frequency from the big autocorrelation matrix to the gsl vector transfer_functions_at_f, applying any required resampling and smoothing */ \
		for(j = 0; j < num_tfs; j++) { \
			z = 0.0; \
			/*
			 * First, apply the sinc filter to higher frequencies. We could hit the edge
			 * of the sinc table or the Nyquist frequency of the transfer function.
			 */ \
			sinc0 = (sinc_taps_per_input - i * sinc_taps_per_output % sinc_taps_per_input) % sinc_taps_per_input; \
			input0 = j + (i * (fd_fft_length - 1) + fd_tf_length - 2) / (fd_tf_length - 1) * elements_per_freq; \
			k_stop = minimum64((sinc_taps_per_input + sinc_length / 2 - sinc0) / sinc_taps_per_input, fd_fft_length - (i * (fd_fft_length - 1) + fd_tf_length - 2) / (fd_tf_length - 1)); \
			for(k = 0; k < k_stop; k++) \
				z += sinc_table[sinc0 + k * sinc_taps_per_input] * autocorrelation_matrix[input0 + k * elements_per_freq]; \
			/*
			 * If we hit the Nyquist frequency of the transfer function but not the edge of
			 * the sinc table, turn around and keep going until we hit the edge of the sinc table.
			 */ \
			sinc0 += k_stop * sinc_taps_per_input; \
			input0 += (k_stop - 2) * elements_per_freq; \
			k_stop = (sinc_taps_per_input + sinc_length / 2 - sinc0) / sinc_taps_per_input; \
			for(k = 0; k < k_stop; k++) \
				z += sinc_table[sinc0 + k * sinc_taps_per_input] * conj ## F_OR_BLANK(autocorrelation_matrix[input0 - k * elements_per_freq]); \
			/*
			 * Now, go back and apply the sinc filter to the lower frequencies. We could hit the edge
			 * of the sinc table or the DC component of the transfer function.
			 */ \
			sinc0 = 1 + (sinc_taps_per_input + i * sinc_taps_per_output - 1) % sinc_taps_per_input; \
			input0 = j + (i * (fd_fft_length - 1) - 1) / (fd_tf_length - 1) * elements_per_freq; \
			k_stop = minimum64((sinc_taps_per_input + sinc_length / 2 - sinc0) / sinc_taps_per_input, (i * (fd_fft_length - 1) - 1) / (fd_tf_length - 1)); \
			for(k = 0; k < k_stop; k++) \
				z += sinc_table[sinc0 + k * sinc_taps_per_input] * autocorrelation_matrix[input0 - k * elements_per_freq]; \
			/*
			 * If we hit the DC component of the transfer function but not the edge of the 
			 * sinc table, turn around and keep going until we hit the edge of the sinc table.
			 */ \
			sinc0 += k_stop * sinc_taps_per_input; \
			input0 = j; \
			k_stop = (sinc_taps_per_input + sinc_length / 2 - sinc0) / sinc_taps_per_input; \
			for(k = 0; k < k_stop; k++) \
				z += sinc_table[sinc0 + k * sinc_taps_per_input] * conj ## F_OR_BLANK(autocorrelation_matrix[input0 + k * elements_per_freq]); \
 \
			/* Set an element of the GSL vector transfer_functions_at_f */ \
			gsl_vector_complex_set(transfer_functions_at_f, j, gsl_complex_rect(creal((complex double) z) / num_avg, cimag((complex double) z) / num_avg)); \
		} \
 \
		/* Next, copy samples at a specific frequency from the big autocorrelation matrix to the gsl matrix autocorrelation_matrix_at_f, applying any required resampling and smoothing */ \
		j_stop = num_tfs * num_tfs; \
		for(j = 0; j < j_stop; j++) { \
			z = 0.0; \
			/*
			 * First, apply the sinc filter to higher frequencies. We could hit the edge
			 * of the sinc table or the Nyquist frequency of the transfer function.
			 */ \
			sinc0 = (sinc_taps_per_input - i * sinc_taps_per_output % sinc_taps_per_input) % sinc_taps_per_input; \
			input0 = j + num_tfs + (i * (fd_fft_length - 1) + fd_tf_length - 2) / (fd_tf_length - 1) * elements_per_freq; \
			k_stop = minimum64((sinc_taps_per_input + sinc_length / 2 - sinc0) / sinc_taps_per_input, fd_fft_length - (i * (fd_fft_length - 1) + fd_tf_length - 2) / (fd_tf_length - 1)); \
			for(k = 0; k < k_stop; k++) \
				z += sinc_table[sinc0 + k * sinc_taps_per_input] * autocorrelation_matrix[input0 + k * elements_per_freq]; \
			/*
			 * If we hit the Nyquist frequency of the transfer function but not the edge of
			 * the sinc table, turn around and keep going until we hit the edge of the sinc table.
			 */ \
			sinc0 += k_stop * sinc_taps_per_input; \
			input0 += (k_stop - 2) * elements_per_freq; \
			k_stop = (sinc_taps_per_input + sinc_length / 2 - sinc0) / sinc_taps_per_input; \
			for(k = 0; k < k_stop; k++) \
				z += sinc_table[sinc0 + k * sinc_taps_per_input] * conj ## F_OR_BLANK(autocorrelation_matrix[input0 - k * elements_per_freq]); \
			/*
			 * Now, go back and apply the sinc filter to the lower frequencies. We could hit the edge
			 * of the sinc table or the DC component of the transfer function.
			 */ \
			sinc0 = 1 + (sinc_taps_per_input + i * sinc_taps_per_output - 1) % sinc_taps_per_input; \
			input0 = j + num_tfs + (i * (fd_fft_length - 1) - 1) / (fd_tf_length - 1) * elements_per_freq; \
			k_stop = minimum64((sinc_taps_per_input + sinc_length / 2 - sinc0) / sinc_taps_per_input, (i * (fd_fft_length - 1) - 1) / (fd_tf_length - 1)); \
			for(k = 0; k < k_stop; k++) \
				z += sinc_table[sinc0 + k * sinc_taps_per_input] * autocorrelation_matrix[input0 - k * elements_per_freq]; \
			/*
			 * If we hit the DC component of the transfer function but not the edge of the 
			 * sinc table, turn around and keep going until we hit the edge of the sinc table.
			 */ \
			sinc0 += k_stop * sinc_taps_per_input; \
			input0 = j + num_tfs; \
			k_stop = (sinc_taps_per_input + sinc_length / 2 - sinc0) / sinc_taps_per_input; \
			for(k = 0; k < k_stop; k++) \
				z += sinc_table[sinc0 + k * sinc_taps_per_input] * conj ## F_OR_BLANK(autocorrelation_matrix[input0 + k * elements_per_freq]); \
 \
			/* Set an element of the GSL matrix autocorrelation_matrix_at_f */ \
			gsl_matrix_complex_set(autocorrelation_matrix_at_f, j / num_tfs, j % num_tfs, gsl_complex_rect(creal((complex double) z) / num_avg, cimag((complex double) z) / num_avg)); \
		} \
 \
		/* Now solve [autocorrelation_matrix_at_f] [transfer_functions(f)] = [transfer_functions_at_f] for [transfer_functions(f)] using gsl */ \
		gsl_linalg_complex_LU_decomp(autocorrelation_matrix_at_f, permutation, &signum); \
		gsl_linalg_complex_LU_solve(autocorrelation_matrix_at_f, permutation, transfer_functions_at_f, transfer_functions_solved_at_f); \
 \
		/* Now copy the result into transfer_functions */ \
		for(j = 0; j < num_tfs; j++) { \
			gslz = gsl_vector_complex_get(transfer_functions_solved_at_f, j); \
			if(isnormal((GSL_REAL(gslz) + GSL_IMAG(gslz))) || GSL_REAL(gslz) + GSL_IMAG(gslz) == 0.0) \
				transfer_functions[j * fd_tf_length + i] = GSL_REAL(gslz) + I * GSL_IMAG(gslz); \
			else { \
				success = FALSE; \
				transfer_functions[j * fd_tf_length + i] = 0.0; \
			} \
		} \
	} \
 \
	return success; \
}


DEFINE_UPDATE_TRANSFER_FUNCTIONS(float, f);
DEFINE_UPDATE_TRANSFER_FUNCTIONS(double, );


#define DEFINE_UPDATE_FIR_FILTERS(DTYPE, F_OR_BLANK) \
static gboolean update_fir_filters_ ## DTYPE(complex double *transfer_functions, int num_tfs, gint64 fir_length, int sample_rate, complex DTYPE *fir_filter, fftw ## F_OR_BLANK ## _plan fir_plan, DTYPE *fd_window, double *fir_window, double *fir_filters) { \
 \
	gboolean success = TRUE; \
	int i; \
	gint64 j, fd_fir_length; \
	fd_fir_length = fir_length / 2 + 1; \
 \
	for(i = 0; i < num_tfs; i++) { \
		for(j = 0; j < fd_fir_length; j++) { \
			/*
			 * Copy each transfer function to fir_filter, which will be ifft'ed. Apply a frequency-domain
			 * window in case we are adding a high- or low-pass filter. Add a delay of half the filter
			 * length, to center it in time.
			 */ \
			fir_filter[j] = (1 - 2 * (j % 2)) * fd_window[j] * (complex DTYPE) transfer_functions[i * fd_fir_length + j]; \
		} \
 \
		/* Make sure the DC and Nyquist components are purely real */ \
		fir_filter[0] = (complex DTYPE) creal ## F_OR_BLANK(fir_filter[0]); \
		fir_filter[fd_fir_length - 1] = (complex DTYPE) creal ## F_OR_BLANK(fir_filter[fd_fir_length - 1]); \
 \
		/* Take the inverse Fourier transform */ \
		fftw ## F_OR_BLANK ## _execute(fir_plan); \
 \
		/* Apply the Tukey window and copy to fir_filters */ \
		DTYPE *real_filter = (DTYPE *) fir_filter; \
		for(j = 0; j < fir_length; j++) { \
			fir_filters[i * fir_length + j] = fir_window[j] * real_filter[j]; \
			success &= isnormal(fir_filters[i * fir_length + j]) || fir_filters[i * fir_length + j] == 0.0; \
		} \
	} \
 \
	return success; \
}


DEFINE_UPDATE_FIR_FILTERS(float, f);
DEFINE_UPDATE_FIR_FILTERS(double, );


#define DEFINE_FIND_TRANSFER_FUNCTION(DTYPE, S_OR_D, F_OR_BLANK) \
static gboolean find_transfer_functions_ ## DTYPE(GSTLALTransferFunction *element, DTYPE *src, guint64 src_size, guint64 pts, gboolean gap) { \
 \
	gboolean success = TRUE; \
 \
	/* Convert src_size from bytes to samples */ \
	g_assert(!(src_size % element->unit_size)); \
	src_size /= element->unit_size; \
 \
	gint64 i, j, k, m, num_ffts, num_ffts_in_avg_if_nogap, k_start, k_stop, first_index, first_index2, fd_fft_length, fd_tf_length, stride, num_tfs; \
	fd_fft_length = element->fft_length / 2 + 1; \
	fd_tf_length = element->fir_length / 2 + 1; \
	stride = element->fft_length - element->fft_overlap; \
	num_tfs = element->channels - 1; \
	DTYPE *real_fft = (DTYPE *) element->workspace.w ## S_OR_D ## pf.fft; \
 \
	/* How many FFTs would there be in the average if there had been no gaps in the data used for transfer functions? Useful for parallel mode. */ \
	num_ffts_in_avg_if_nogap = element->parallel_mode ? (element->sample_count - (gint64) src_size - element->update_samples - element->fft_overlap) / stride : element->workspace.w ## S_OR_D ## pf.num_ffts_in_avg; \
	num_ffts_in_avg_if_nogap = maximum64(num_ffts_in_avg_if_nogap, 0); \
 \
	/* Determine how many FFTs we will calculate from combined leftover and new input data */ \
	num_ffts = minimum64((element->workspace.w ## S_OR_D ## pf.num_leftover + stride - 1) / stride, element->num_ffts - num_ffts_in_avg_if_nogap); \
	num_ffts = minimum64(num_ffts, (element->workspace.w ## S_OR_D ## pf.num_leftover + (gint64) src_size - element->fft_overlap) / stride); \
	if(num_ffts < 0) \
		num_ffts = 0; \
 \
	/* Loop through the input data and compute transfer functions */ \
	for(i = 0; i < num_ffts; i++) { \
		for(j = 0; j < element->channels; j++) { \
			/* First, copy the inputs from leftover data */ \
			k_stop = element->workspace.w ## S_OR_D ## pf.num_leftover - i * stride; \
			first_index = i * stride * element->channels + j; \
			for(k = 0; k < k_stop; k++) \
				real_fft[k] = element->workspace.w ## S_OR_D ## pf.fft_window[k] * element->workspace.w ## S_OR_D ## pf.leftover_data[first_index + k * element->channels]; \
 \
			/* Now copy the inputs from new input data */ \
			k_start = k_stop; \
			k_stop = element->fft_length; \
			for(k = k_start; k < k_stop; k++) \
				real_fft[k] = element->workspace.w ## S_OR_D ## pf.fft_window[k] * src[j + element->channels * (k - k_start)]; \
 \
			/* Take an FFT */ \
			fftw ## F_OR_BLANK ## _execute(element->workspace.w ## S_OR_D ## pf.plan); \
 \
			/* Copy FFT to the proper location */ \
			first_index = j * fd_fft_length; \
			for(k = 0; k < fd_fft_length; k++) \
				element->workspace.w ## S_OR_D ## pf.ffts[first_index + k] = element->workspace.w ## S_OR_D ## pf.fft[k]; \
 \
			/* Fill in any requested "notches" with straight lines */ \
			int n; \
			for(n = 0; n < element->num_notches; n++) { \
				gint64 notch_start = element->notch_indices[2 * n]; \
				gint64 notch_end = element->notch_indices[2 * n + 1]; \
				complex DTYPE fft_start = element->workspace.w ## S_OR_D ## pf.fft[element->notch_indices[2 * n]]; \
				complex DTYPE fft_end = element->workspace.w ## S_OR_D ## pf.fft[element->notch_indices[2 * n + 1]]; \
				for(k = notch_start + 1; k < notch_end; k++) \
					element->workspace.w ## S_OR_D ## pf.ffts[first_index + k] = fft_start * (k - notch_start) / (notch_end - notch_start) + fft_end * (notch_end - k) / (notch_end - notch_start); \
			} \
		} \
 \
		/* Check the FFTs to see if their values will produce usable transfer functions */ \
		for(j = 0; j < fd_fft_length * element->channels; j++) \
			success &= isnormal(cabs ## F_OR_BLANK(element->workspace.w ## S_OR_D ## pf.ffts[j])); \
 \
		if(success && element->use_median) { \
			/*
			 * Put all the transfer functions of the autocorrelation matrix in a median
			 * array. We will track the location of the median as the array gets filled.
			 * Note that the data in the median array is stored in "frequency-major"
			 * order: transfer functions at a particular frequency are stored
			 * contiguously in memory before incrementing to the next frequency.
			 */ \
			DTYPE complex cplx; \
			DTYPE real; \
			DTYPE imag; \
			gint64 max_in_median = element->num_ffts / 2 + 1; \
			gint64 num_in_median = element->workspace.w ## S_OR_D ## pf.num_ffts_in_avg - element->workspace.w ## S_OR_D ## pf.num_ffts_dropped + i; \
			gint64 median_array_num; \
			for(j = 0; j < fd_fft_length; j++) { \
				first_index = j * element->channels * num_tfs - 1; \
				for(k = 1; k <= num_tfs; k++) { \
					/* First, divide FFTs of first channel by others to get those transfer functions */ \
					cplx = element->workspace.w ## S_OR_D ## pf.ffts[j] / element->workspace.w ## S_OR_D ## pf.ffts[j + k * fd_fft_length]; \
					median_array_num = j * num_tfs * num_tfs + k - 1; \
					real = creal ## F_OR_BLANK(cplx); \
					update_median_ ## DTYPE(element->workspace.w ## S_OR_D ## pf.autocorrelation_median_real + median_array_num * max_in_median, real, max_in_median, num_in_median, element->workspace.w ## S_OR_D ## pf.index_median_real + median_array_num); \
					imag = cimag ## F_OR_BLANK(cplx); \
					update_median_ ## DTYPE(element->workspace.w ## S_OR_D ## pf.autocorrelation_median_imag + median_array_num * max_in_median, imag, max_in_median, num_in_median, element->workspace.w ## S_OR_D ## pf.index_median_imag + median_array_num); \
 \
					/* Now find off-diagonal elements of the autocorrelation matrix */ \
					for(m = 1; m < k; m++) { \
						/* Elements above diagonal */ \
						cplx = element->workspace.w ## S_OR_D ## pf.ffts[j + k * fd_fft_length] / element->workspace.w ## S_OR_D ## pf.ffts[j + m * fd_fft_length]; \
						median_array_num = j * num_tfs * num_tfs + num_tfs + (m - 1) * (num_tfs - 1) + k - 2; \
						real = creal ## F_OR_BLANK(cplx); \
						update_median_ ## DTYPE(element->workspace.w ## S_OR_D ## pf.autocorrelation_median_real + median_array_num * max_in_median, real, max_in_median, num_in_median, element->workspace.w ## S_OR_D ## pf.index_median_real + median_array_num); \
						imag = cimag ## F_OR_BLANK(cplx); \
						update_median_ ## DTYPE(element->workspace.w ## S_OR_D ## pf.autocorrelation_median_imag + median_array_num * max_in_median, imag, max_in_median, num_in_median, element->workspace.w ## S_OR_D ## pf.index_median_imag + median_array_num); \
						/* Elements below diagonal */ \
						cplx = element->workspace.w ## S_OR_D ## pf.ffts[j + m * fd_fft_length] / element->workspace.w ## S_OR_D ## pf.ffts[j + k * fd_fft_length]; \
						median_array_num = j * num_tfs * num_tfs + num_tfs + (k - 1) * (num_tfs - 1) + m - 1; \
						real = creal ## F_OR_BLANK(cplx); \
						update_median_ ## DTYPE(element->workspace.w ## S_OR_D ## pf.autocorrelation_median_real + median_array_num * max_in_median, real, max_in_median, num_in_median, element->workspace.w ## S_OR_D ## pf.index_median_real + median_array_num); \
						imag = cimag ## F_OR_BLANK(cplx); \
						update_median_ ## DTYPE(element->workspace.w ## S_OR_D ## pf.autocorrelation_median_imag + median_array_num * max_in_median, imag, max_in_median, num_in_median, element->workspace.w ## S_OR_D ## pf.index_median_imag + median_array_num); \
					} \
				} \
			} \
		} else if(success) { \
			/* 
			 * Add into the autocorrelation matrix to be averaged. The autocorrelation
			 * matrix includes all transfer functions. Note that the data is stored in
			 * "frequency-major" order: transfer functions at a particular frequency are
			 * stored contiguously in memory before incrementing to the next frequency.
			 */ \
			for(j = 0; j < fd_fft_length; j++) { \
				first_index = j * element->channels * num_tfs - 1; \
				for(k = 1; k <= num_tfs; k++) { \
					/* First, divide FFTs of first channel by others to get those transfer functions */ \
					element->workspace.w ## S_OR_D ## pf.autocorrelation_matrix[first_index + k] += element->workspace.w ## S_OR_D ## pf.ffts[j] / element->workspace.w ## S_OR_D ## pf.ffts[j + k * fd_fft_length]; \
 \
					/* Now set elements of the autocorrelation matrix along the diagonal equal to one */ \
					first_index2 = first_index + k * element->channels; \
					element->workspace.w ## S_OR_D ## pf.autocorrelation_matrix[first_index2] += 1.0; \
 \
					/* Now find all other elements of the autocorrelation matrix */ \
					for(m = 1; m <= num_tfs - k; m++) { \
						element->workspace.w ## S_OR_D ## pf.autocorrelation_matrix[first_index2 + m] += element->workspace.w ## S_OR_D ## pf.ffts[j + (k + m) * fd_fft_length] / element->workspace.w ## S_OR_D ## pf.ffts[j + k * fd_fft_length]; \
						element->workspace.w ## S_OR_D ## pf.autocorrelation_matrix[first_index2 + m * num_tfs] += element->workspace.w ## S_OR_D ## pf.ffts[j + k * fd_fft_length] / element->workspace.w ## S_OR_D ## pf.ffts[j + (k + m) * fd_fft_length]; \
					} \
				} \
			} \
		} else { \
			GST_WARNING_OBJECT(element, "Computed FFT is not usable. Dropping..."); \
			element->workspace.w ## S_OR_D ## pf.num_ffts_dropped++; \
			success = TRUE; \
		} \
	} \
 \
	element->workspace.w ## S_OR_D ## pf.num_ffts_in_avg += num_ffts; \
	num_ffts_in_avg_if_nogap += num_ffts; \
 \
	/* Determine how many FFTs we will calculate from only new input samples, computed differently in parallel mode */ \
	num_ffts = (element->sample_count - element->update_samples - element->fft_overlap) / stride - num_ffts_in_avg_if_nogap; /* how many more we could compute */ \
	num_ffts = minimum64(num_ffts, element->num_ffts - num_ffts_in_avg_if_nogap); /* how many more we need to update transfer functions */ \
	if(num_ffts < 0 || (element->parallel_mode && gap)) \
		num_ffts = 0; \
 \
	/* Find the location of the first sample in src that will be used */ \
	DTYPE *ptr; \
	if(element->update_samples >= element->sample_count - (gint64) src_size) \
		ptr = src + (element->update_samples - element->sample_count + (gint64) src_size) * element->channels; \
	else \
		ptr = src + ((stride - (element->sample_count - (gint64) src_size - element->update_samples) % stride) % stride) * element->channels; \
 \
	/* Loop through the input data and compute transfer functions */ \
	for(i = 0; i < num_ffts; i++) { \
		for(j = 0; j < element->channels; j++) { \
			/* Copy inputs to take an FFT */ \
			k_stop = element->fft_length; \
			first_index = i * stride * element->channels + j; \
			for(k = 0; k < k_stop; k++) \
				real_fft[k] = element->workspace.w ## S_OR_D ## pf.fft_window[k] * ptr[first_index + k * element->channels]; \
 \
			/* Take an FFT */ \
			fftw ## F_OR_BLANK ## _execute(element->workspace.w ## S_OR_D ## pf.plan); \
 \
			/* Copy FFT to the proper location */ \
			first_index = j * fd_fft_length; \
			for(k = 0; k < fd_fft_length; k++) \
				element->workspace.w ## S_OR_D ## pf.ffts[first_index + k] = element->workspace.w ## S_OR_D ## pf.fft[k]; \
 \
			/* Fill in any requested "notches" with straight lines */ \
			int n; \
			for(n = 0; n < element->num_notches; n++) { \
				gint64 notch_start = element->notch_indices[2 * n]; \
				gint64 notch_end = element->notch_indices[2 * n + 1]; \
				complex DTYPE fft_start = element->workspace.w ## S_OR_D ## pf.fft[element->notch_indices[2 * n]]; \
				complex DTYPE fft_end = element->workspace.w ## S_OR_D ## pf.fft[element->notch_indices[2 * n + 1]]; \
				for(k = notch_start + 1; k < notch_end; k++) \
					element->workspace.w ## S_OR_D ## pf.ffts[first_index + k] = fft_start * (k - notch_start) / (notch_end - notch_start) + fft_end * (notch_end - k) / (notch_end - notch_start); \
			} \
		} \
 \
		/* Check the FFTs to see if their values will produce usable transfer functions */ \
		for(j = 0; j < fd_fft_length * element->channels; j++) \
			success &= isnormal(cabs ## F_OR_BLANK(element->workspace.w ## S_OR_D ## pf.ffts[j])); \
 \
		if(success && element->use_median) { \
			/*
			 * Put all the transfer functions of the autocorrelation matrix in a median
			 * array. We will track the location of the median as the array gets filled.
			 * Note that the data in the median array is stored in "frequency-major"
			 * order: transfer functions at a particular frequency are stored
			 * contiguously in memory before incrementing to the next frequency.
			 */ \
			DTYPE complex cplx; \
			DTYPE real; \
			DTYPE imag; \
			gint64 max_in_median = element->num_ffts / 2 + 1; \
			gint64 num_in_median = element->workspace.w ## S_OR_D ## pf.num_ffts_in_avg - element->workspace.w ## S_OR_D ## pf.num_ffts_dropped + i; \
			gint64 median_array_num; \
			for(j = 0; j < fd_fft_length; j++) { \
				first_index = j * element->channels * num_tfs - 1; \
				for(k = 1; k <= num_tfs; k++) { \
					/* First, divide FFTs of first channel by others to get those transfer functions */ \
					cplx = element->workspace.w ## S_OR_D ## pf.ffts[j] / element->workspace.w ## S_OR_D ## pf.ffts[j + k * fd_fft_length]; \
					median_array_num = j * num_tfs * num_tfs + k - 1; \
					real = creal ## F_OR_BLANK(cplx); \
					update_median_ ## DTYPE(element->workspace.w ## S_OR_D ## pf.autocorrelation_median_real + median_array_num * max_in_median, real, max_in_median, num_in_median, element->workspace.w ## S_OR_D ## pf.index_median_real + median_array_num); \
					imag = cimag ## F_OR_BLANK(cplx); \
					update_median_ ## DTYPE(element->workspace.w ## S_OR_D ## pf.autocorrelation_median_imag + median_array_num * max_in_median, imag, max_in_median, num_in_median, element->workspace.w ## S_OR_D ## pf.index_median_imag + median_array_num); \
 \
					/* Now find off-diagonal elements of the autocorrelation matrix */ \
					for(m = 1; m < k; m++) { \
						/* Elements above diagonal */ \
						cplx = element->workspace.w ## S_OR_D ## pf.ffts[j + k * fd_fft_length] / element->workspace.w ## S_OR_D ## pf.ffts[j + m * fd_fft_length]; \
						median_array_num = j * num_tfs * num_tfs + num_tfs + (m - 1) * (num_tfs - 1) + k - 2; \
						real = creal ## F_OR_BLANK(cplx); \
						update_median_ ## DTYPE(element->workspace.w ## S_OR_D ## pf.autocorrelation_median_real + median_array_num * max_in_median, real, max_in_median, num_in_median, element->workspace.w ## S_OR_D ## pf.index_median_real + median_array_num); \
						imag = cimag ## F_OR_BLANK(cplx); \
						update_median_ ## DTYPE(element->workspace.w ## S_OR_D ## pf.autocorrelation_median_imag + median_array_num * max_in_median, imag, max_in_median, num_in_median, element->workspace.w ## S_OR_D ## pf.index_median_imag + median_array_num); \
						/* Elements below diagonal */ \
						cplx = element->workspace.w ## S_OR_D ## pf.ffts[j + m * fd_fft_length] / element->workspace.w ## S_OR_D ## pf.ffts[j + k * fd_fft_length]; \
						median_array_num = j * num_tfs * num_tfs + num_tfs + (k - 1) * (num_tfs - 1) + m - 1; \
						real = creal ## F_OR_BLANK(cplx); \
						update_median_ ## DTYPE(element->workspace.w ## S_OR_D ## pf.autocorrelation_median_real + median_array_num * max_in_median, real, max_in_median, num_in_median, element->workspace.w ## S_OR_D ## pf.index_median_real + median_array_num); \
						imag = cimag ## F_OR_BLANK(cplx); \
						update_median_ ## DTYPE(element->workspace.w ## S_OR_D ## pf.autocorrelation_median_imag + median_array_num * max_in_median, imag, max_in_median, num_in_median, element->workspace.w ## S_OR_D ## pf.index_median_imag + median_array_num); \
					} \
				} \
			} \
		} else if(success) { \
			/* 
			 * Add into the autocorrelation matrix to be averaged. The autocorrelation
			 * matrix includes all transfer functions. Note that the data is stored in
			 * "frequency-major" order: transfer functions at a particular frequency are
			 * stored contiguously in memory before incrementing to the next frequency.
			 */ \
			for(j = 0; j < fd_fft_length; j++) { \
				first_index = j * element->channels * num_tfs - 1; \
				for(k = 1; k <= num_tfs; k++) { \
					/* First, divide FFTs of first channel by others to get those transfer functions */ \
					element->workspace.w ## S_OR_D ## pf.autocorrelation_matrix[first_index + k] += element->workspace.w ## S_OR_D ## pf.ffts[j] / element->workspace.w ## S_OR_D ## pf.ffts[j + k * fd_fft_length]; \
 \
					/* Now set elements of the autocorrelation matrix along the diagonal equal to one */ \
					first_index2 = first_index + k * element->channels; \
					element->workspace.w ## S_OR_D ## pf.autocorrelation_matrix[first_index2] += 1.0; \
 \
					/* Now find all other elements of the autocorrelation matrix */ \
					for(m = 1; m <= num_tfs - k; m++) { \
						element->workspace.w ## S_OR_D ## pf.autocorrelation_matrix[first_index2 + m] += element->workspace.w ## S_OR_D ## pf.ffts[j + (k + m) * fd_fft_length] / element->workspace.w ## S_OR_D ## pf.ffts[j + k * fd_fft_length]; \
						element->workspace.w ## S_OR_D ## pf.autocorrelation_matrix[first_index2 + m * num_tfs] += element->workspace.w ## S_OR_D ## pf.ffts[j + k * fd_fft_length] / element->workspace.w ## S_OR_D ## pf.ffts[j + (k + m) * fd_fft_length]; \
					} \
				} \
			} \
		} else { \
			GST_WARNING_OBJECT(element, "Computed FFT is not usable. Dropping..."); \
			element->workspace.w ## S_OR_D ## pf.num_ffts_dropped++; \
			success = TRUE; \
		} \
	} \
 \
	element->workspace.w ## S_OR_D ## pf.num_ffts_in_avg += num_ffts; \
	num_ffts_in_avg_if_nogap += num_ffts; \
	g_assert_cmpint(element->workspace.w ## S_OR_D ## pf.num_ffts_in_avg, <=, element->num_ffts); \
	g_assert_cmpint(num_ffts_in_avg_if_nogap, <=, element->num_ffts); \
 \
	/* Now store samples for the next buffer. First, find the sample count of the start of the next fft */ \
	if(!gap) { \
		gint64 sample_count_next_fft; \
		if(num_ffts_in_avg_if_nogap == element->num_ffts) \
			sample_count_next_fft = 2 * element->update_samples + element->num_ffts * stride + element->fft_overlap + 1; /* If we finished updating the transfer functions */ \
		else \
			sample_count_next_fft = element->update_samples + 1 + num_ffts_in_avg_if_nogap * stride; \
 \
		/* Deal with any leftover samples that will remain leftover */ \
		first_index = (sample_count_next_fft - 1 - (element->sample_count - (gint64) src_size - element->workspace.w ## S_OR_D ## pf.num_leftover)) * element->channels; \
		k_stop = (element->sample_count - (gint64) src_size + 1 - sample_count_next_fft) * element->channels; \
		for(k = 0; k < k_stop; k++) \
			element->workspace.w ## S_OR_D ## pf.leftover_data[k] = element->workspace.w ## S_OR_D ## pf.leftover_data[first_index + k]; \
 \
		/* Deal with new samples that will be leftover */ \
		k_start = maximum64(k_stop, 0); \
		k_stop = (element->sample_count + 1 - sample_count_next_fft) * element->channels; \
		first_index = (sample_count_next_fft - (element->sample_count - (gint64) src_size + 1)) * element->channels; \
		first_index = maximum64(first_index, 0) - k_start; /* since we are adding k below */ \
		for(k = k_start; k < k_stop; k++) \
			element->workspace.w ## S_OR_D ## pf.leftover_data[k] = src[first_index + k]; \
 \
		/* Record the total number of leftover samples */ \
		element->workspace.w ## S_OR_D ## pf.num_leftover = maximum64(0, element->sample_count + 1 - sample_count_next_fft); \
	} \
	/* Finally, update transfer functions if ready */ \
	if(num_ffts_in_avg_if_nogap == element->num_ffts || (!element->parallel_mode && num_ffts_in_avg_if_nogap - element->workspace.w ## S_OR_D ## pf.num_ffts_dropped >= element->min_ffts && *element->transfer_functions == 0.0)) { \
		if(element->use_median) { \
			/* Then we still need to fill the autocorrelation matrix with median values */ \
			gint64 median_length = element->num_ffts / 2 + 1; \
			int elements_per_freq = element->channels * num_tfs; \
			int medians_per_freq = num_tfs * num_tfs; \
			gint64 median_array_num; \
			if((element->workspace.w ## S_OR_D ## pf.num_ffts_in_avg - element->workspace.w ## S_OR_D ## pf.num_ffts_dropped) % 2) { \
				/* Length of median array is odd, so the median is just the middle value */ \
				for(i = 0; i < fd_fft_length; i++) { \
					for(j = 0; j < num_tfs; j++) { \
						/* First, the ratios fft(first channel) / fft(other channels) */ \
						median_array_num = i * medians_per_freq + j; \
						element->workspace.w ## S_OR_D ## pf.autocorrelation_matrix[i * elements_per_freq + j] = element->workspace.w ## S_OR_D ## pf.autocorrelation_median_real[median_array_num * median_length + element->workspace.w ## S_OR_D ## pf.index_median_real[median_array_num]] + I * element->workspace.w ## S_OR_D ## pf.autocorrelation_median_imag[median_array_num * median_length + element->workspace.w ## S_OR_D ## pf.index_median_imag[median_array_num]]; \
						/* Elements along the diagonal are one */ \
						element->workspace.w ## S_OR_D ## pf.autocorrelation_matrix[i * elements_per_freq + num_tfs + j * (num_tfs + 1)] = 1.0; \
						if(j < num_tfs - 1) { \
							for(k = 0; k < num_tfs; k++) { \
								/* Off-diagonal elements come from the median array */ \
								median_array_num = i * medians_per_freq + (j + 1) * num_tfs + k; \
								element->workspace.w ## S_OR_D ## pf.autocorrelation_matrix[i * elements_per_freq + num_tfs + 1 + j * (num_tfs + 1) + k] = element->workspace.w ## S_OR_D ## pf.autocorrelation_median_real[median_array_num * median_length + element->workspace.w ## S_OR_D ## pf.index_median_real[median_array_num]] + I * element->workspace.w ## S_OR_D ## pf.autocorrelation_median_imag[median_array_num * median_length + element->workspace.w ## S_OR_D ## pf.index_median_imag[median_array_num]]; \
							} \
						} \
					} \
				} \
			} else { \
				/* Length of median array is even, so the median is the average of the two middle values */ \
				for(i = 0; i < fd_fft_length; i++) { \
					for(j = 0; j < num_tfs; j++) { \
						/* First, the ratios fft(first channel) / fft(other channels) */ \
						median_array_num = i * medians_per_freq + j; \
						element->workspace.w ## S_OR_D ## pf.autocorrelation_matrix[i * elements_per_freq + j] = (element->workspace.w ## S_OR_D ## pf.autocorrelation_median_real[median_array_num * median_length + element->workspace.w ## S_OR_D ## pf.index_median_real[median_array_num] - 1] + element->workspace.w ## S_OR_D ## pf.autocorrelation_median_real[median_array_num * median_length + element->workspace.w ## S_OR_D ## pf.index_median_real[median_array_num]] + I * (element->workspace.w ## S_OR_D ## pf.autocorrelation_median_imag[median_array_num * median_length + element->workspace.w ## S_OR_D ## pf.index_median_imag[median_array_num] - 1] + element->workspace.w ## S_OR_D ## pf.autocorrelation_median_imag[median_array_num * median_length + element->workspace.w ## S_OR_D ## pf.index_median_imag[median_array_num]])) / 2.0; \
						/* Elements along the diagonal are one */ \
						element->workspace.w ## S_OR_D ## pf.autocorrelation_matrix[i * elements_per_freq + num_tfs + j * (num_tfs + 1)] = 1.0; \
						if(j < num_tfs - 1) { \
							for(k = 0; k < num_tfs; k++) { \
								/* Off-diagonal elements come from the median array */ \
								median_array_num = i * medians_per_freq + (j + 1) * num_tfs + k; \
								element->workspace.w ## S_OR_D ## pf.autocorrelation_matrix[i * elements_per_freq + num_tfs + 1 + j * (num_tfs + 1) + k] = (element->workspace.w ## S_OR_D ## pf.autocorrelation_median_real[median_array_num * median_length + element->workspace.w ## S_OR_D ## pf.index_median_real[median_array_num] - 1] + element->workspace.w ## S_OR_D ## pf.autocorrelation_median_real[median_array_num * median_length + element->workspace.w ## S_OR_D ## pf.index_median_real[median_array_num]] + I * (element->workspace.w ## S_OR_D ## pf.autocorrelation_median_imag[median_array_num * median_length + element->workspace.w ## S_OR_D ## pf.index_median_imag[median_array_num] - 1] + element->workspace.w ## S_OR_D ## pf.autocorrelation_median_imag[median_array_num * median_length + element->workspace.w ## S_OR_D ## pf.index_median_imag[median_array_num]])) / 2.0; \
							} \
						} \
					} \
				} \
			} \
		} \
		success &= update_transfer_functions_ ## DTYPE(element->workspace.w ## S_OR_D ## pf.autocorrelation_matrix, num_tfs, fd_fft_length, fd_tf_length, element->workspace.w ## S_OR_D ## pf.sinc_table, element->workspace.w ## S_OR_D ## pf.sinc_length, element->workspace.w ## S_OR_D ## pf.sinc_taps_per_df, element->use_median ? 1 : element->workspace.w ## S_OR_D ## pf.num_ffts_in_avg - element->workspace.w ## S_OR_D ## pf.num_ffts_dropped, element->workspace.w ## S_OR_D ## pf.transfer_functions_at_f, element->workspace.w ## S_OR_D ## pf.transfer_functions_solved_at_f, element->workspace.w ## S_OR_D ## pf.autocorrelation_matrix_at_f, element->workspace.w ## S_OR_D ## pf.permutation, element->transfer_functions); \
		if(success) { \
			GST_LOG_OBJECT(element, "Just computed new transfer functions"); \
			/* Let other elements know about the update */ \
			g_object_notify_by_pspec(G_OBJECT(element), properties[ARG_TRANSFER_FUNCTIONS]); \
			/* Write transfer functions to the screen or a file if we want */ \
			if(element->write_to_screen || element->filename) \
				write_transfer_functions(element->transfer_functions, gst_element_get_name(element), element->rate / 2.0 / (fd_tf_length - 1.0), fd_tf_length, num_tfs, element->t_start_tf, element->t_start_tf + (double) (num_ffts_in_avg_if_nogap * stride + element->fft_overlap) / element->rate, element->write_to_screen, element->filename, TRUE); \
			/* If this is this first transfer function after a gap, we may wish to store it */ \
			if(element->use_first_after_gap && !element->num_tfs_since_gap) { \
				for(i = 0; i < num_tfs * fd_tf_length; i++) \
					element->post_gap_transfer_functions[i] = element->transfer_functions[i]; \
			} \
			element->num_tfs_since_gap++; \
		} else if (element->parallel_mode) { \
			GST_LOG_OBJECT(element, "Transfer function(s) computation failed. Using zeros."); \
			memset(element->transfer_functions, 0, (element->channels - 1) * fd_tf_length * sizeof(*element->transfer_functions)); \
			success = TRUE; \
		} else \
			GST_WARNING_OBJECT(element, "Transfer function(s) computation failed. Trying again."); \
 \
		if(num_ffts_in_avg_if_nogap == element->num_ffts) { \
			element->sample_count = (gint64) (gst_util_uint64_scale_int_round(pts, element->rate, GST_SECOND) + src_size + element->update_samples - element->update_delay_samples) % (element->update_samples + element->num_ffts * stride + element->fft_overlap); \
			if(element->sample_count > element->update_samples) \
				element->sample_count -= element->update_samples + element->num_ffts * stride + element->fft_overlap; \
			element->workspace.w ## S_OR_D ## pf.num_ffts_in_avg = 0; \
			element->workspace.w ## S_OR_D ## pf.num_ffts_dropped = 0; \
			element->computed_full_tfs = TRUE; \
		} \
		/* Update FIR filters if we want */ \
		if(success && element->make_fir_filters) { \
			success &= update_fir_filters_ ## DTYPE(element->transfer_functions, num_tfs, element->fir_length, element->rate, element->workspace.w ## S_OR_D ## pf.fir_filter, element->workspace.w ## S_OR_D ## pf.fir_plan, element->workspace.w ## S_OR_D ## pf.fd_fir_window, element->workspace.w ## S_OR_D ## pf.fir_window, element->fir_filters); \
			if(success) { \
				GST_LOG_OBJECT(element, "Just computed new FIR filters"); \
				/* Let other elements know about the update */ \
				g_object_notify_by_pspec(G_OBJECT(element), properties[ARG_FIR_FILTERS]); \
				/* Provide a timestamp indicating when the filter becomes invalid if requested */ \
				if(element->fir_timeshift < G_MAXINT64) { \
					if(element->fir_timeshift < 0 && (guint64) (-element->fir_timeshift) > pts) \
						element->fir_endtime = 0; \
					else \
						element->fir_endtime = pts + element->fir_timeshift; \
					g_object_notify_by_pspec(G_OBJECT(element), properties[ARG_FIR_ENDTIME]); \
				} \
				/* Write FIR filters to the screen or a file if we want */ \
				if(element->write_to_screen || element->filename) \
					write_fir_filters(element->fir_filters, gst_element_get_name(element), element->fir_length, num_tfs, element->t_start_tf, element->t_start_tf + (double) (num_ffts_in_avg_if_nogap * stride + element->fft_overlap) / element->rate, element->write_to_screen, element->filename, TRUE); \
				/* If this is this first FIR filter after a gap, we may wish to store it */ \
				if(element->use_first_after_gap && element->num_tfs_since_gap == 1) { \
					for(i = 0; i < num_tfs * element->fir_length; i++) \
						element->post_gap_fir_filters[i] = element->fir_filters[i]; \
				} \
			} else if (element->parallel_mode) \
				GST_WARNING_OBJECT(element, "FIR filter(s) computation failed. Waiting for the next cycle."); \
			else \
				GST_WARNING_OBJECT(element, "FIR filter(s) computation failed. Trying again."); \
		} \
	} \
 \
	return success; \
}


DEFINE_FIND_TRANSFER_FUNCTION(float, s, f);
DEFINE_FIND_TRANSFER_FUNCTION(double, d, );


/*
 * ============================================================================
 *
 *			    GstBaseSink Overrides
 *
 * ============================================================================
 */


/*
 * start()
 */


static gboolean start(GstBaseSink *sink) {

	GSTLALTransferFunction *element = GSTLAL_TRANSFERFUNCTION(sink);

	/* Timestamp bookkeeping */
	element->t0 = GST_CLOCK_TIME_NONE;
	element->offset0 = GST_BUFFER_OFFSET_NONE;
	element->next_in_offset = GST_BUFFER_OFFSET_NONE;

	/* At start of stream, we want the element to compute a transfer function as soon as possible, unless in parallel mode */
	element->min_ffts = minimum64(element->min_ffts, element->num_ffts);
	if(!element->parallel_mode) {
		gint64 long_samples = element->num_ffts * (element->fft_length - element->fft_overlap) + element->fft_overlap;
		gint64 short_samples = element->min_ffts * (element->fft_length - element->fft_overlap) + element->fft_overlap;
		element->sample_count = element->update_samples - (short_samples * element->update_delay_samples + long_samples - 1) / long_samples;
	}
	element->computed_full_tfs = FALSE;

	/* If we are writing output to file, and a file already exists with the same name, remove it */
	if(element->filename)
		remove(element->filename);

	/* Sanity checks */
	/* FIXME: find a better place to put these. Putting them in set_property seems problematic. */
	if(element->num_ffts > 1 && element->fft_overlap >= element->fft_length) {
		GST_WARNING_OBJECT(element, "fft_overlap must be less than fft_length! Resetting fft_overlap to fft_length - 1.");
		element->fft_overlap = element->fft_length - 1;
	}
	if(element->fir_length % 2) {
		GST_WARNING_OBJECT(element, "The chosen fir-length must be even. Adding 1 to make it even.");
		element->fir_length += 1;
	}
	if(element->make_fir_filters && element->frequency_resolution < (double) element->rate / element->fir_length)
		GST_WARNING_OBJECT(element, "The specified frequency resolution is finer than 1/fir_length, which cannot be achieved. The actual frequency resolution will be reset to 1/MIN(fft_length, fir_length).");
	if(element->make_fir_filters && element->frequency_resolution < (double) element->rate / element->fft_length)
		GST_WARNING_OBJECT(element, "The specified frequency resolution is finer than 1/fft_length, which cannot be achieved. The actual frequency resolution will be reset to 1/MIN(fft_length, fir_length).");
	if(!element->make_fir_filters && element->high_pass != 0)
		GST_WARNING_OBJECT(element, "A FIR filter high-pass cutoff frequency is set, but no FIR filter is being produced. Set the property make-fir-filters to a nonzero value to make FIR filters.");
	if(!element->make_fir_filters && element->low_pass != 0)
		GST_WARNING_OBJECT(element, "A FIR filter low-pass cutoff frequency is set, but no FIR filter is being produced. Set the property make-fir-filters to a nonzero value to make FIR filters.");
	if(element->high_pass != 0 && element->low_pass != 0 && element->high_pass > element->low_pass)
		GST_WARNING_OBJECT(element, "The high-pass cutoff frequency of the FIR filters is above the low-pass cutoff frequency. Reset high-pass and/or low-pass to change this.");

	return TRUE;
}


/*
 * event()
 */


static gboolean event(GstBaseSink *sink, GstEvent *event) {
	GSTLALTransferFunction *element = GSTLAL_TRANSFERFUNCTION(sink);
	gboolean success = TRUE;
	GST_DEBUG_OBJECT(element, "Got %s event on sink pad", GST_EVENT_TYPE_NAME(event));

	if(GST_EVENT_TYPE(event) == GST_EVENT_EOS && !element->parallel_mode) {
		/* If End Of Stream is here and we have not yet computed transfer functions from a full data set, use whatever data we have to compute them now. */
		if(element->data_type == GSTLAL_TRANSFERFUNCTION_F32) {
			if(element->workspace.wspf.num_ffts_in_avg - element->workspace.wspf.num_ffts_dropped > element->min_ffts && !element->computed_full_tfs) {
				/* Then we should use the transfer functions we have now, since they won't get any better */
				gint64 fd_fft_length = element->fft_length / 2 + 1;
				gint64 fd_tf_length = element->fir_length / 2 + 1;
				gint64 num_tfs = element->channels - 1;
				if(element->use_median) {
					/* Then we still need to fill the autocorrelation matrix with median values */
					gint64 median_length = element->num_ffts / 2 + 1;
					int elements_per_freq = element->channels * num_tfs;
					int medians_per_freq = num_tfs * num_tfs;
					gint64 median_array_num;
					gint64 i, j, k;
					if((element->workspace.wspf.num_ffts_in_avg - element->workspace.wspf.num_ffts_dropped) % 2) {
						/* Length of median array is odd, so the median is just the middle value */
						for(i = 0; i < fd_fft_length; i++) {
							for(j = 0; j < num_tfs; j++) {
								/* First, the ratios fft(first channel) / fft(other channels) */
								median_array_num = i * medians_per_freq + j;
								element->workspace.wspf.autocorrelation_matrix[i * elements_per_freq + j] = element->workspace.wspf.autocorrelation_median_real[median_array_num * median_length + element->workspace.wspf.index_median_real[median_array_num]] + I * element->workspace.wspf.autocorrelation_median_imag[median_array_num * median_length + element->workspace.wspf.index_median_imag[median_array_num]];
								/* Elements along the diagonal are one */
								element->workspace.wspf.autocorrelation_matrix[i * elements_per_freq + num_tfs + j * (num_tfs + 1)] = 1.0;
								if(j < num_tfs - 1) {
									for(k = 0; k < num_tfs; k++) {
										/* Off-diagonal elements come from the median array */
										median_array_num = i * medians_per_freq + (j + 1) * num_tfs + k;
										element->workspace.wspf.autocorrelation_matrix[i * elements_per_freq + num_tfs + 1 + j * (num_tfs + 1) + k] = element->workspace.wspf.autocorrelation_median_real[median_array_num * median_length + element->workspace.wspf.index_median_real[median_array_num]] + I * element->workspace.wspf.autocorrelation_median_imag[median_array_num * median_length + element->workspace.wspf.index_median_imag[median_array_num]];
									}
								}
							}
						}
					} else {
						/* Length of median array is even, so the median is the average of the two middle values */
						for(i = 0; i < fd_fft_length; i++) {
							for(j = 0; j < num_tfs; j++) {
								/* First, the ratios fft(first channel) / fft(other channels) */
								median_array_num = i * medians_per_freq + j;
								element->workspace.wspf.autocorrelation_matrix[i * elements_per_freq + j] = (element->workspace.wspf.autocorrelation_median_real[median_array_num * median_length + element->workspace.wspf.index_median_real[median_array_num] - 1] + element->workspace.wspf.autocorrelation_median_real[median_array_num * median_length + element->workspace.wspf.index_median_real[median_array_num]] + I * (element->workspace.wspf.autocorrelation_median_imag[median_array_num * median_length + element->workspace.wspf.index_median_imag[median_array_num] - 1] + element->workspace.wspf.autocorrelation_median_imag[median_array_num * median_length + element->workspace.wspf.index_median_imag[median_array_num]])) / 2.0;
								/* Elements along the diagonal are one */
								element->workspace.wspf.autocorrelation_matrix[i * elements_per_freq + num_tfs + j * (num_tfs + 1)] = 1.0;
								if(j < num_tfs - 1) {
									for(k = 0; k < num_tfs; k++) {
										/* Off-diagonal elements come from the median array */
										median_array_num = i * medians_per_freq + (j + 1) * num_tfs + k;
										element->workspace.wspf.autocorrelation_matrix[i * elements_per_freq + num_tfs + 1 + j * (num_tfs + 1) + k] = (element->workspace.wspf.autocorrelation_median_real[median_array_num * median_length + element->workspace.wspf.index_median_real[median_array_num] - 1] + element->workspace.wspf.autocorrelation_median_real[median_array_num * median_length + element->workspace.wspf.index_median_real[median_array_num]] + I * (element->workspace.wspf.autocorrelation_median_imag[median_array_num * median_length + element->workspace.wspf.index_median_imag[median_array_num] - 1] + element->workspace.wspf.autocorrelation_median_imag[median_array_num * median_length + element->workspace.wspf.index_median_imag[median_array_num]])) / 2.0;
									}
								}
							}
						}
					}
				}
				success &= update_transfer_functions_float(element->workspace.wspf.autocorrelation_matrix, num_tfs, fd_fft_length, fd_tf_length, element->workspace.wspf.sinc_table, element->workspace.wspf.sinc_length, element->workspace.wspf.sinc_taps_per_df, element->use_median ? 1 : element->num_ffts - element->workspace.wspf.num_ffts_dropped, element->workspace.wspf.transfer_functions_at_f, element->workspace.wspf.transfer_functions_solved_at_f, element->workspace.wspf.autocorrelation_matrix_at_f, element->workspace.wspf.permutation, element->transfer_functions);
				if(success) {
					GST_LOG_OBJECT(element, "Just computed new transfer functions");
					/* Let other elements know about the update */
					g_object_notify_by_pspec(G_OBJECT(element), properties[ARG_TRANSFER_FUNCTIONS]);
					/* Write transfer functions to the screen or a file if we want */
					if(element->write_to_screen || element->filename)
						write_transfer_functions(element->transfer_functions, gst_element_get_name(element), element->rate / 2.0 / (fd_tf_length - 1.0), fd_tf_length, num_tfs, element->t_start_tf, element->t_start_tf + (double) (element->workspace.wspf.num_ffts_in_avg * (element->fft_length - element->fft_overlap) + element->fft_overlap) / element->rate, element->write_to_screen, element->filename, TRUE);
				} else
					GST_WARNING_OBJECT(element, "Transfer function(s) computation failed. No transfer functions will be produced.");
				/* Update FIR filters if we want */
				if(success && element->make_fir_filters) {
					success &= update_fir_filters_float(element->transfer_functions, num_tfs, element->fir_length, element->rate, element->workspace.wspf.fir_filter, element->workspace.wspf.fir_plan, element->workspace.wspf.fd_fir_window, element->workspace.wspf.fir_window, element->fir_filters);
					if(success) {
						GST_LOG_OBJECT(element, "Just computed new FIR filters");
						/* Let other elements know about the update */
						g_object_notify_by_pspec(G_OBJECT(element), properties[ARG_FIR_FILTERS]);
						/* Write FIR filters to the screen or a file if we want */
						if(element->write_to_screen || element->filename)
							write_fir_filters(element->fir_filters, gst_element_get_name(element), element->fir_length, num_tfs, element->t_start_tf, element->t_start_tf + (double) (element->workspace.wspf.num_ffts_in_avg * (element->fft_length - element->fft_overlap) + element->fft_overlap) / element->rate, element->write_to_screen, element->filename, TRUE);
					} else
						GST_WARNING_OBJECT(element, "FIR filter(s) computation failed. No FIR filters will be produced.");
				}
			}
		} else {
			if(element->workspace.wdpf.num_ffts_in_avg - element->workspace.wdpf.num_ffts_dropped > element->min_ffts && !element->computed_full_tfs) {
				/* Then we should use the transfer functions we have now, since they won't get any better */
				gint64 fd_fft_length = element->fft_length / 2 + 1;
				gint64 fd_tf_length = element->fir_length / 2 + 1;
				gint64 num_tfs = element->channels - 1;
				if(element->use_median) {
					/* Then we still need to fill the autocorrelation matrix with median values */
					gint64 median_length = element->num_ffts / 2 + 1;
					int elements_per_freq = element->channels * num_tfs;
					int medians_per_freq = num_tfs * num_tfs;
					gint64 median_array_num;
					gint64 i, j, k;
					if((element->workspace.wdpf.num_ffts_in_avg - element->workspace.wdpf.num_ffts_dropped) % 2) {
						/* Length of median array is odd, so the median is just the middle value */
						for(i = 0; i < fd_fft_length; i++) {
							for(j = 0; j < num_tfs; j++) {
								/* First, the ratios fft(first channel) / fft(other channels) */
								median_array_num = i * medians_per_freq + j;
								element->workspace.wdpf.autocorrelation_matrix[i * elements_per_freq + j] = element->workspace.wdpf.autocorrelation_median_real[median_array_num * median_length + element->workspace.wdpf.index_median_real[median_array_num]] + I * element->workspace.wdpf.autocorrelation_median_imag[median_array_num * median_length + element->workspace.wdpf.index_median_imag[median_array_num]];
								/* Elements along the diagonal are one */
								element->workspace.wdpf.autocorrelation_matrix[i * elements_per_freq + num_tfs + j * (num_tfs + 1)] = 1.0;
								if(j < num_tfs - 1) {
									for(k = 0; k < num_tfs; k++) {
										/* Off-diagonal elements come from the median array */
										median_array_num = i * medians_per_freq + (j + 1) * num_tfs + k;
										element->workspace.wdpf.autocorrelation_matrix[i * elements_per_freq + num_tfs + 1 + j * (num_tfs + 1) + k] = element->workspace.wdpf.autocorrelation_median_real[median_array_num * median_length + element->workspace.wdpf.index_median_real[median_array_num]] + I * element->workspace.wdpf.autocorrelation_median_imag[median_array_num * median_length + element->workspace.wdpf.index_median_imag[median_array_num]];
									}
								}
							}
						}
					} else {
						/* Length of median array is even, so the median is the average of the two middle values */
						for(i = 0; i < fd_fft_length; i++) {
							for(j = 0; j < num_tfs; j++) {
								/* First, the ratios fft(first channel) / fft(other channels) */
								median_array_num = i * medians_per_freq + j;
								element->workspace.wdpf.autocorrelation_matrix[i * elements_per_freq + j] = (element->workspace.wdpf.autocorrelation_median_real[median_array_num * median_length + element->workspace.wdpf.index_median_real[median_array_num] - 1] + element->workspace.wdpf.autocorrelation_median_real[median_array_num * median_length + element->workspace.wdpf.index_median_real[median_array_num]] + I * (element->workspace.wdpf.autocorrelation_median_imag[median_array_num * median_length + element->workspace.wdpf.index_median_imag[median_array_num] - 1] + element->workspace.wdpf.autocorrelation_median_imag[median_array_num * median_length + element->workspace.wdpf.index_median_imag[median_array_num]])) / 2.0;
								/* Elements along the diagonal are one */
								element->workspace.wdpf.autocorrelation_matrix[i * elements_per_freq + num_tfs + j * (num_tfs + 1)] = 1.0;
								if(j < num_tfs - 1) {
									for(k = 0; k < num_tfs; k++) {
										/* Off-diagonal elements come from the median array */
										median_array_num = i * medians_per_freq + (j + 1) * num_tfs + k;
										element->workspace.wdpf.autocorrelation_matrix[i * elements_per_freq + num_tfs + 1 + j * (num_tfs + 1) + k] = (element->workspace.wdpf.autocorrelation_median_real[median_array_num * median_length + element->workspace.wdpf.index_median_real[median_array_num] - 1] + element->workspace.wdpf.autocorrelation_median_real[median_array_num * median_length + element->workspace.wdpf.index_median_real[median_array_num]] + I * (element->workspace.wdpf.autocorrelation_median_imag[median_array_num * median_length + element->workspace.wdpf.index_median_imag[median_array_num] - 1] + element->workspace.wdpf.autocorrelation_median_imag[median_array_num * median_length + element->workspace.wdpf.index_median_imag[median_array_num]])) / 2.0;
									}
								}
							}
						}
					}
				}
				success &= update_transfer_functions_double(element->workspace.wdpf.autocorrelation_matrix, num_tfs, fd_fft_length, fd_tf_length, element->workspace.wdpf.sinc_table, element->workspace.wdpf.sinc_length, element->workspace.wdpf.sinc_taps_per_df, element->use_median ? 1 : element->num_ffts - element->workspace.wdpf.num_ffts_dropped, element->workspace.wdpf.transfer_functions_at_f, element->workspace.wdpf.transfer_functions_solved_at_f, element->workspace.wdpf.autocorrelation_matrix_at_f, element->workspace.wdpf.permutation, element->transfer_functions);
				if(success) {
					GST_LOG_OBJECT(element, "Just computed new transfer functions");
					/* Let other elements know about the update */
					g_object_notify_by_pspec(G_OBJECT(element), properties[ARG_TRANSFER_FUNCTIONS]);
					/* Write transfer functions to the screen or a file if we want */
					if(element->write_to_screen || element->filename)
						write_transfer_functions(element->transfer_functions, gst_element_get_name(element), element->rate / 2.0 / (fd_tf_length - 1.0), fd_tf_length, num_tfs, element->t_start_tf, element->t_start_tf + (double) (element->workspace.wdpf.num_ffts_in_avg * (element->fft_length - element->fft_overlap) + element->fft_overlap) / element->rate, element->write_to_screen, element->filename, TRUE);
				} else
					GST_WARNING_OBJECT(element, "Transfer function(s) computation failed. No transfer functions will be produced.");
				/* Update FIR filters if we want */
				if(success && element->make_fir_filters) {
					success &= update_fir_filters_double(element->transfer_functions, num_tfs, element->fir_length, element->rate, element->workspace.wdpf.fir_filter, element->workspace.wdpf.fir_plan, element->workspace.wdpf.fd_fir_window, element->workspace.wdpf.fir_window, element->fir_filters);
					if(success) {
						GST_LOG_OBJECT(element, "Just computed new FIR filters");
						/* Let other elements know about the update */
						g_object_notify_by_pspec(G_OBJECT(element), properties[ARG_FIR_FILTERS]);
						/* Write FIR filters to the screen or a file if we want */
						if(element->write_to_screen || element->filename)
							write_fir_filters(element->fir_filters, gst_element_get_name(element), element->fir_length, num_tfs, element->t_start_tf, element->t_start_tf + (double) (element->workspace.wdpf.num_ffts_in_avg * (element->fft_length - element->fft_overlap) + element->fft_overlap) / element->rate, element->write_to_screen, element->filename, TRUE);
					} else
						GST_WARNING_OBJECT(element, "FIR filter(s) computation failed. No FIR filters will be produced.");
				}
			}
		}
	}

	if(GST_EVENT_TYPE(event) == GST_EVENT_EOS && element->fir_timeshift < G_MAXINT64) {
		/* These filters should remain usable as long as possible */
		element->fir_endtime = G_MAXUINT64 - 1;
		g_object_notify_by_pspec(G_OBJECT(element), properties[ARG_FIR_ENDTIME]);
	}

	success = GST_BASE_SINK_CLASS(gstlal_transferfunction_parent_class)->event(sink, event);

	return success;
}


/*
 * set_caps()
 */


static gboolean set_caps(GstBaseSink *sink, GstCaps *caps) {

	GSTLALTransferFunction *element = GSTLAL_TRANSFERFUNCTION(sink);

	gboolean success = TRUE;

	/* Parse the caps to find the format, sample rate, and number of channels */
	GstStructure *str = gst_caps_get_structure(caps, 0);
	const gchar *name = gst_structure_get_string(str, "format");
	success &= (name != NULL);
	success &= gst_structure_get_int(str, "rate", &element->rate);
	success &= gst_structure_get_int(str, "channels", &element->channels);
	g_assert_cmpint(element->channels, >, 1);

	/* Record the data type and unit size */
	if(success) {
		if(!strcmp(name, GST_AUDIO_NE(F32))) {
			element->data_type = GSTLAL_TRANSFERFUNCTION_F32;
			element->unit_size = 4 * element->channels;
		} else if(!strcmp(name, GST_AUDIO_NE(F64))) {
			element->data_type = GSTLAL_TRANSFERFUNCTION_F64;
			element->unit_size = 8 * element->channels;
		} else
			g_assert_not_reached();
	}

	/* Sanity check */
	if(element->update_samples + element->fft_length < element->rate)
		GST_WARNING_OBJECT(element, "The chosen fft_length and update_samples are very short. Errors may result.");

	/*
	 * Free any memory that depends on stream parameters
	 */

	if(element->post_gap_transfer_functions) {
		g_free(element->post_gap_transfer_functions);
		element->post_gap_transfer_functions = NULL;
	}
	if(element->post_gap_fir_filters) {
		g_free(element->post_gap_fir_filters);
		element->post_gap_fir_filters = NULL;
	}
	if(element->transfer_functions) {
		g_free(element->transfer_functions);
		element->transfer_functions = NULL;
	}
	if(element->fir_filters) {
		g_free(element->fir_filters);
		element->fir_filters = NULL;
	}
	if(element->workspace.wspf.fd_fir_window) {
		g_free(element->workspace.wspf.fd_fir_window);
		element->workspace.wspf.fd_fir_window = NULL;
	}
	if(element->workspace.wspf.sinc_table) {
		g_free(element->workspace.wspf.sinc_table);
		element->workspace.wspf.sinc_table = NULL;
	}
	if(element->workspace.wspf.leftover_data) {
		g_free(element->workspace.wspf.leftover_data);
		element->workspace.wspf.leftover_data = NULL;
	}
	element->workspace.wspf.num_leftover = 0;
	if(element->workspace.wspf.ffts) {
		g_free(element->workspace.wspf.ffts);
		element->workspace.wspf.ffts = NULL;
	}
	element->workspace.wspf.num_ffts_in_avg = 0;
	element->workspace.wspf.num_ffts_dropped = 0;
	if(element->workspace.wspf.autocorrelation_matrix) {
		g_free(element->workspace.wspf.autocorrelation_matrix);
		element->workspace.wspf.autocorrelation_matrix = NULL;
	}
	if(element->workspace.wspf.autocorrelation_median_real) {
		g_free(element->workspace.wspf.autocorrelation_median_real);
		element->workspace.wspf.autocorrelation_median_real = NULL;
	}
	if(element->workspace.wspf.autocorrelation_median_imag) {
		g_free(element->workspace.wspf.autocorrelation_median_imag);
		element->workspace.wspf.autocorrelation_median_imag = NULL;
	}
	if(element->workspace.wspf.transfer_functions_at_f) {
		gsl_vector_complex_free(element->workspace.wspf.transfer_functions_at_f);
		element->workspace.wspf.transfer_functions_at_f = NULL;
	}
	if(element->workspace.wspf.transfer_functions_solved_at_f) {
		gsl_vector_complex_free(element->workspace.wspf.transfer_functions_solved_at_f);
		element->workspace.wspf.transfer_functions_solved_at_f = NULL;
	}
	if(element->workspace.wspf.autocorrelation_matrix_at_f) {
		gsl_matrix_complex_free(element->workspace.wspf.autocorrelation_matrix_at_f);
		element->workspace.wspf.autocorrelation_matrix_at_f = NULL;
	}
	if(element->workspace.wspf.permutation) {
		gsl_permutation_free(element->workspace.wspf.permutation);
		element->workspace.wspf.permutation = NULL;
	}
	if(element->workspace.wspf.fft) {
		gstlal_fftw_lock();
		fftwf_free(element->workspace.wspf.fft);
		element->workspace.wspf.fft = NULL;
		fftwf_destroy_plan(element->workspace.wspf.plan);
		gstlal_fftw_unlock();
	}
	if(element->workspace.wspf.fir_filter) {
		gstlal_fftw_lock();
		fftwf_free(element->workspace.wspf.fir_filter);
		element->workspace.wspf.fir_filter = NULL;
		fftwf_destroy_plan(element->workspace.wspf.fir_plan);
		gstlal_fftw_unlock();
	}
	if(element->workspace.wspf.fd_fir_window) {
		g_free(element->workspace.wspf.fd_fir_window);
		element->workspace.wspf.fd_fir_window = NULL;
	}
	if(element->workspace.wdpf.sinc_table) {
		g_free(element->workspace.wdpf.sinc_table);
		element->workspace.wdpf.sinc_table = NULL;
	}
	if(element->workspace.wdpf.leftover_data) {
		g_free(element->workspace.wdpf.leftover_data);
		element->workspace.wdpf.leftover_data = NULL;
	}
	element->workspace.wdpf.num_leftover = 0;
	if(element->workspace.wdpf.ffts) {
		g_free(element->workspace.wdpf.ffts);
		element->workspace.wdpf.ffts = NULL;
	}
	element->workspace.wdpf.num_ffts_in_avg = 0;
	element->workspace.wspf.num_ffts_dropped = 0;
	if(element->workspace.wdpf.autocorrelation_matrix) {
		g_free(element->workspace.wdpf.autocorrelation_matrix);
		element->workspace.wdpf.autocorrelation_matrix = NULL;
	}
	if(element->workspace.wdpf.autocorrelation_median_real) {
		g_free(element->workspace.wdpf.autocorrelation_median_real);
		element->workspace.wdpf.autocorrelation_median_real = NULL;
	}
	if(element->workspace.wdpf.autocorrelation_median_imag) {
		g_free(element->workspace.wdpf.autocorrelation_median_imag);
		element->workspace.wdpf.autocorrelation_median_imag = NULL;
	}
	if(element->workspace.wdpf.transfer_functions_at_f) {
		gsl_vector_complex_free(element->workspace.wdpf.transfer_functions_at_f);
		element->workspace.wdpf.transfer_functions_at_f = NULL;
	}
	if(element->workspace.wdpf.transfer_functions_solved_at_f) {
		gsl_vector_complex_free(element->workspace.wdpf.transfer_functions_solved_at_f);
		element->workspace.wdpf.transfer_functions_solved_at_f = NULL;
	}
	if(element->workspace.wdpf.autocorrelation_matrix_at_f) {
		gsl_matrix_complex_free(element->workspace.wdpf.autocorrelation_matrix_at_f);
		element->workspace.wdpf.autocorrelation_matrix_at_f = NULL;
	}
	if(element->workspace.wdpf.permutation) {
		gsl_permutation_free(element->workspace.wdpf.permutation);
		element->workspace.wdpf.permutation = NULL;
	}
	if(element->workspace.wdpf.fft) {
		gstlal_fftw_lock();
		fftw_free(element->workspace.wdpf.fft);
		element->workspace.wdpf.fft = NULL;
		fftw_destroy_plan(element->workspace.wdpf.plan);
		gstlal_fftw_unlock();
	}
	if(element->workspace.wdpf.fir_filter) {
		gstlal_fftw_lock();
		fftw_free(element->workspace.wdpf.fir_filter);
		element->workspace.wdpf.fir_filter = NULL;
		fftw_destroy_plan(element->workspace.wdpf.fir_plan);
		gstlal_fftw_unlock();
	}

	/*
	 * Allocate any memory that depends on stream parameters
	 */

	gint64 fd_fft_length = element->fft_length / 2 + 1;
	if(!element->fir_length)
		element->fir_length = element->fft_length;
	gint64 fd_fir_length = element->fir_length / 2 + 1;
	element->transfer_functions = g_malloc((element->channels - 1) * fd_fir_length * sizeof(*element->transfer_functions));
	memset(element->transfer_functions, 0, (element->channels - 1) * fd_fir_length * sizeof(*element->transfer_functions));
	if(element->use_first_after_gap) {
		element->post_gap_transfer_functions = g_malloc((element->channels - 1) * fd_fir_length * sizeof(*element->post_gap_transfer_functions));
		memset(element->post_gap_transfer_functions, 0, (element->channels - 1) * fd_fir_length * sizeof(*element->post_gap_transfer_functions));
	}
	if(element->make_fir_filters) {
		element->fir_filters = g_malloc((element->channels - 1) * element->fir_length * sizeof(*element->fir_filters));
		memset(element->fir_filters, 0, (element->channels - 1) * element->fir_length * sizeof(*element->fir_filters));
		if(element->use_first_after_gap) {
			element->post_gap_fir_filters = g_malloc((element->channels - 1) * element->fir_length * sizeof(*element->post_gap_fir_filters));
			memset(element->post_gap_fir_filters, 0, (element->channels - 1) * element->fir_length * sizeof(*element->post_gap_fir_filters));
		}
	}

	/* Prepare to deal with any notches */
	if(element->num_notches && !element->notch_indices) {
		int k, k_stop = element->num_notches;
		double df = (double) element->rate / element->fft_length;
		element->notch_indices = g_malloc(2 * element->num_notches * sizeof(*element->notch_indices));
		for(k = 0; k < k_stop; k++) {
			if(element->notch_frequencies[2 * k + 1] >= element->rate / 2.0) {
				GST_WARNING_OBJECT(element, "Cannot include notch with upper bound at %f Hz, since that is above the Nyquist rate of %f Hz", element->notch_frequencies[2 * k + 1], element->rate / 2.0);
				element->num_notches--;
			} else {
				element->notch_indices[2 * k] = (gint64) (element->notch_frequencies[2 * k] / df - 1.0);
				element->notch_indices[2 * k + 1] = (gint64) (element->notch_frequencies[2 * k + 1] / df + 2.0);
			}
		}
	}

	/* Prepare workspace for finding transfer functions and FIR filters */
	/* Frequency resolution in units of frequency bins of fft data */
	double fft_alpha = maximum(maximum(element->frequency_resolution / element->rate, 1.0 / element->fir_length), 1.0 / element->fft_length) * element->fft_length;
	/* Frequency resolution in units of frequency bins of the FIR filters */
	double fir_alpha = maximum(maximum(element->frequency_resolution / element->rate, 1.0 / element->fir_length), 1.0 / element->fft_length) * element->fir_length;
	if(element->data_type == GSTLAL_TRANSFERFUNCTION_F32) {

		/*
		 * window functions
		 */

		gint64 i, i_stop, i_start;
		if(!element->workspace.wspf.fft_window) {
			switch(element->window) {
			case GSTLAL_TRANSFERFUNCTION_DPSS:
				element->workspace.wspf.fft_window = dpss_float(element->fft_length, fft_alpha, 5.0, NULL, FALSE);
				break;

			case GSTLAL_TRANSFERFUNCTION_KAISER:
				element->workspace.wspf.fft_window = kaiser_float(element->fft_length, M_PI * fft_alpha, NULL, FALSE);
				break;

			case GSTLAL_TRANSFERFUNCTION_DOLPH_CHEBYSHEV:
				element->workspace.wspf.fft_window = DolphChebyshev_float(element->fft_length, fft_alpha, NULL, FALSE);
				break;

			default:
				GST_ERROR_OBJECT(element, "Invalid window type.  See properties for appropriate window types.");
				g_assert_not_reached();
				break;
			}
		}

		if(!element->workspace.wspf.sinc_table) {

			/*
			 * Make a sinc table to resample and/or low-pass filter the transfer functions when we make FIR filters
			 */

			if(element->fir_length == element->fft_length && element->frequency_resolution <= (double) element->rate / element->fir_length) {
				element->workspace.wspf.sinc_length = 1;
				element->workspace.wspf.sinc_table = g_malloc(sizeof(*element->workspace.wspf.sinc_table));
				*element->workspace.wspf.sinc_table = 1.0;
				element->workspace.wspf.sinc_taps_per_df = 1;
			} else {
				/* 
				 * element->workspace.wspf.sinc_taps_per_df is the number of taps per frequency bin (at the finer
				 * frequency resolution). If fir_length is an integer multiple or divisor of fft_length, taps_per_df is 1.
				 */
				gint64 common_denominator, short_length, long_length;
				short_length = minimum64(fd_fft_length - 1, fd_fir_length - 1);
				long_length = maximum64(fd_fft_length - 1, fd_fir_length - 1);
				common_denominator = long_length;
				while(common_denominator % short_length)
					common_denominator += long_length;
				element->workspace.wspf.sinc_taps_per_df = common_denominator / long_length;
				/* taps_per_osc is the number of taps per half-oscillation in the sinc table */
				gint64 taps_per_osc = element->workspace.wspf.sinc_taps_per_df * (gint64) (maximum(maximum(element->frequency_resolution * element->fir_length / element->rate, element->frequency_resolution * element->fft_length / element->rate), maximum((double) element->fft_length / element->fir_length, (double) element->fir_length / element->fft_length)) + 0.5);
				element->workspace.wspf.sinc_length = minimum64(element->workspace.wspf.sinc_taps_per_df * maximum64(fd_fir_length / 2, fd_fft_length / 2) - 1, 1 + SINC_LENGTH * taps_per_osc);

				/* To save memory, we use symmetry and record only half of the sinc table */
				element->workspace.wspf.sinc_table = g_malloc((1 + element->workspace.wspf.sinc_length / 2) * sizeof(*element->workspace.wspf.sinc_table));
				*element->workspace.wspf.sinc_table = 1.0;
				gint64 j;
				float sin_arg, normalization;
				for(i = 1; i <= element->workspace.wspf.sinc_length / 2; i++) {
					sin_arg = M_PI * i / taps_per_osc;
					element->workspace.wspf.sinc_table[i] = sinf(sin_arg) / sin_arg;
				}

				/* Window the sinc table */
				kaiser_float(element->workspace.wspf.sinc_length, 10.0, element->workspace.wspf.sinc_table, TRUE);

				/* 
				 * Normalize the sinc table to make the DC gain exactly 1. We need to account for the fact 
				 * that the density of taps in the filter could be higher than the density of input samples.
				 */
				gint64 taps_per_input = fd_fir_length > fd_fft_length ? common_denominator / short_length : element->workspace.wspf.sinc_taps_per_df;
				for(i = 0; i < (taps_per_input + 1) / 2; i++) {
					normalization = 0.0;
					for(j = i; j <= element->workspace.wspf.sinc_length / 2; j += taps_per_input)
						normalization += element->workspace.wspf.sinc_table[j];
					for(j = taps_per_input - i; j <= element->workspace.wspf.sinc_length / 2; j += taps_per_input)
						normalization += element->workspace.wspf.sinc_table[j];
					for(j = i; j <= element->workspace.wspf.sinc_length / 2; j += taps_per_input)
						element->workspace.wspf.sinc_table[j] /= normalization;
					if(i) {
						for(j = taps_per_input - i; j <= element->workspace.wspf.sinc_length / 2; j += taps_per_input)
							element->workspace.wspf.sinc_table[j] /= normalization;
					}
				}
				/* If taps_per_input is even, we need to account for one more normalization without "over-normalizing." */
				if(!((taps_per_input) % 2)) {
					normalization = 0.0;
					for(j = taps_per_input / 2; j <= element->workspace.wspf.sinc_length / 2; j += taps_per_input)
						normalization += 2 * element->workspace.wspf.sinc_table[j];
					for(j = taps_per_input / 2; j <= element->workspace.wspf.sinc_length / 2; j += taps_per_input)
						element->workspace.wspf.sinc_table[j] /= normalization;
				}
			}
		}

		if(element->make_fir_filters && (!element->workspace.wspf.fd_fir_window)) {

			/*
			 * Make a frequency-domain window to roll off low and high frequencies
			 */

			element->workspace.wspf.fd_fir_window = g_malloc(fd_fir_length * sizeof(*element->workspace.wspf.fd_fir_window));

			/* Initialize to ones */
			for(i = 0; i < fd_fir_length; i++)
				element->workspace.wspf.fd_fir_window[i] = 1.0;

			int f_nyquist = element->rate / 2;
			float df_per_hz = (fd_fir_length - 1.0) / f_nyquist;
			int freq_res_samples = (int) (fir_alpha + 0.5);

			/* high-pass filter */
			/* Remove low frequencies */
			i_stop = (gint64) (element->high_pass * df_per_hz + 0.5) - freq_res_samples;
			for(i = 0; i < i_stop; i++)
				element->workspace.wspf.fd_fir_window[i] = 0.0;

			/* Apply half of a Hann window */
			i_start = i_stop;
			i_stop += freq_res_samples;
			for(i = i_start; i < i_stop; i++)
				element->workspace.wspf.fd_fir_window[i] *= (float) pow(sin((M_PI / 2.0) * (i - i_start) / (i_stop - i_start)), 2.0);

			/* low-pass filter */
			if(element->low_pass > 0) {
				/* Apply half of a Hann window */
				i_start = (gint64) (element->low_pass * df_per_hz + 0.5);
				i_stop = minimum64(fd_fir_length, 1.4 * i_start);
				for(i = i_start; i < i_stop; i++)
					element->workspace.wspf.fd_fir_window[i] *= (float) pow(cos((M_PI / 2.0) * (i - i_start) / (i_stop - i_start)), 2.0);

				/* Remove high frequencies */
				i_start = i_stop;
				i_stop = fd_fir_length;
				for(i = i_start; i < i_stop; i++)
					element->workspace.wspf.fd_fir_window[i] = 0.0;
			}

			/*
			 * Make a time-domain window to apply to the FIR filters.
			 */

			if(!element->workspace.wspf.fir_window) {

				switch(element->window) {
				case GSTLAL_TRANSFERFUNCTION_DPSS:
					element->workspace.wspf.fir_window = dpss_double(element->fir_length, fir_alpha, 5.0, NULL, FALSE);
					break;

				case GSTLAL_TRANSFERFUNCTION_KAISER:
					element->workspace.wspf.fir_window = kaiser_double(element->fir_length, M_PI * fir_alpha, NULL, FALSE);
					break;

				case GSTLAL_TRANSFERFUNCTION_DOLPH_CHEBYSHEV:
					element->workspace.wspf.fir_window = DolphChebyshev_double(element->fir_length, fir_alpha, NULL, FALSE);
					break;

				default:
					GST_ERROR_OBJECT(element, "Invalid window type.  See properties for appropriate window types.");
					g_assert_not_reached();
					break;
				}
			}
		}

		/* intermediate data storage */
		element->workspace.wspf.leftover_data = g_malloc(element->channels * (element->fft_length - 1) * sizeof(*element->workspace.wspf.leftover_data));
		element->workspace.wspf.num_leftover = 0;
		element->workspace.wspf.ffts = g_malloc(element->channels * fd_fft_length * sizeof(*element->workspace.wspf.ffts));
		element->workspace.wspf.num_ffts_in_avg = 0;
		element->workspace.wspf.autocorrelation_matrix = g_malloc(element->channels * (element->channels - 1) * fd_fft_length * sizeof(*element->workspace.wspf.autocorrelation_matrix));
		if(element->use_median) {
			element->workspace.wspf.autocorrelation_median_real = g_malloc((element->num_ffts / 2 + 1) * (element->channels - 1) * (element->channels - 1) * fd_fft_length * sizeof(*element->workspace.wspf.autocorrelation_median_real));
			element->workspace.wspf.index_median_real = g_malloc((element->channels - 1) * (element->channels - 1) * fd_fft_length * sizeof(*element->workspace.wspf.index_median_real));
			element->workspace.wspf.autocorrelation_median_imag = g_malloc((element->num_ffts / 2 + 1) * (element->channels - 1) * (element->channels - 1) * fd_fft_length * sizeof(*element->workspace.wspf.autocorrelation_median_imag));
			element->workspace.wspf.index_median_imag = g_malloc((element->channels - 1) * (element->channels - 1) * fd_fft_length * sizeof(*element->workspace.wspf.index_median_imag));
		}

		/* Allocate memory for gsl matrix manipulations. The same memory locations will be used repeatedly */
		element->workspace.wspf.transfer_functions_at_f = gsl_vector_complex_alloc(element->channels - 1);
		element->workspace.wspf.transfer_functions_solved_at_f = gsl_vector_complex_alloc(element->channels - 1);
		element->workspace.wspf.autocorrelation_matrix_at_f = gsl_matrix_complex_alloc(element->channels - 1, element->channels - 1);
		element->workspace.wspf.permutation = gsl_permutation_alloc(element->channels - 1);

		/* Allocate memory for fftwf to do Fourier transforms of data. The same memory locations will be used repeatedly */
		gstlal_fftw_lock();

		GST_LOG_OBJECT(element, "starting FFTWF planning");

		/* data that will be Fourier transformed into frequency domain */
		element->workspace.wspf.fft = (complex float *) fftwf_malloc(fd_fft_length * sizeof(*element->workspace.wspf.fft));
		element->workspace.wspf.plan = fftwf_plan_dft_r2c_1d(element->fft_length, (float *) element->workspace.wspf.fft, element->workspace.wspf.fft, FFTW_ESTIMATE);

		if(element->make_fir_filters && !element->workspace.wspf.fir_filter) {

			/* data that will be inverse Fourier transformed back into the time domain */
			element->workspace.wspf.fir_filter = (complex float *) fftwf_malloc(fd_fir_length * sizeof(*element->workspace.wspf.fir_filter));
			element->workspace.wspf.fir_plan = fftwf_plan_dft_c2r_1d(element->fir_length, element->workspace.wspf.fir_filter, (float *) element->workspace.wspf.fir_filter, FFTW_ESTIMATE);
		}
		GST_LOG_OBJECT(element, "FFTWF planning complete");

		gstlal_fftw_unlock();

	} else if(element->data_type == GSTLAL_TRANSFERFUNCTION_F64) {

		/* 
		 * window functions
		 */

		gint64 i, i_stop, i_start;
		if(!element->workspace.wdpf.fft_window) {
			switch(element->window) {
			case GSTLAL_TRANSFERFUNCTION_DPSS:
				element->workspace.wdpf.fft_window = dpss_double(element->fft_length, fft_alpha, 5.0, NULL, FALSE);
				break;

			case GSTLAL_TRANSFERFUNCTION_KAISER:
				element->workspace.wdpf.fft_window = kaiser_double(element->fft_length, M_PI * fft_alpha, NULL, FALSE);
				break;

			case GSTLAL_TRANSFERFUNCTION_DOLPH_CHEBYSHEV:
				element->workspace.wdpf.fft_window = DolphChebyshev_double(element->fft_length, fft_alpha, NULL, FALSE);
				break;

			default:
				GST_ERROR_OBJECT(element, "Invalid window type.  See properties for appropriate window types.");
				g_assert_not_reached();
				break;
			}
		}

		if(!element->workspace.wdpf.sinc_table) {

			/*
			 * Make a sinc table to resample and/or low-pass filter the transfer functions when we make FIR filters
			 */

			if(element->fir_length == element->fft_length && element->frequency_resolution < (double) element->rate / element->fir_length) {
				element->workspace.wdpf.sinc_length = 1;
				element->workspace.wdpf.sinc_table = g_malloc(sizeof(*element->workspace.wdpf.sinc_table));
				*element->workspace.wdpf.sinc_table = 1.0;
				element->workspace.wspf.sinc_taps_per_df = 1;
			} else {
				/* 
				 * element->workspace.wspf.sinc_taps_per_df is the number of taps per frequency bin (at the finer
				 * frequency resolution). If fir_length is an integer multiple of divisor of fft_length, taps_per_df is 1.
				 */
				gint64 common_denominator, short_length, long_length;
				short_length = minimum64(fd_fft_length - 1, fd_fir_length - 1);
				long_length = maximum64(fd_fft_length - 1, fd_fir_length - 1);
				common_denominator = long_length;
				while(common_denominator % short_length)
					common_denominator += long_length;
				element->workspace.wspf.sinc_taps_per_df = common_denominator / long_length;
				/* taps_per_osc is the number of taps per half-oscillation in the sinc table */
				gint64 taps_per_osc = element->workspace.wspf.sinc_taps_per_df * (gint64) (maximum(maximum(element->frequency_resolution * element->fir_length / element->rate, element->frequency_resolution * element->fft_length / element->rate), maximum((double) element->fft_length / element->fir_length, (double) element->fir_length / element->fft_length)) + 0.5);
				element->workspace.wdpf.sinc_length = minimum64(element->workspace.wspf.sinc_taps_per_df * maximum64(fd_fir_length / 2, fd_fft_length / 2) - 1, 1 + SINC_LENGTH * taps_per_osc);

				/* To save memory, we use symmetry and record only half of the sinc table */
				element->workspace.wdpf.sinc_table = g_malloc((1 + element->workspace.wdpf.sinc_length / 2) * sizeof(*element->workspace.wdpf.sinc_table));
				*element->workspace.wdpf.sinc_table = 1.0;
				gint64 j;
				double sin_arg, normalization;
				for(i = 1; i <= element->workspace.wdpf.sinc_length / 2; i++) {
					sin_arg = M_PI * i / taps_per_osc;
					element->workspace.wdpf.sinc_table[i] = sin(sin_arg) / sin_arg;
				}

				/* Window the sinc table */
				kaiser_double(element->workspace.wdpf.sinc_length, 10.0, element->workspace.wdpf.sinc_table, TRUE);

				/* 
				 * Normalize the sinc table to make the DC gain exactly 1. We need to account for the fact 
				 * that the density of taps in the filter could be higher than the density of input samples.
				 */
				gint64 taps_per_input = fd_fir_length > fd_fft_length ? common_denominator / short_length : element->workspace.wspf.sinc_taps_per_df;
				for(i = 0; i < (taps_per_input + 1) / 2; i++) {
					normalization = 0.0;
					for(j = i; j <= element->workspace.wdpf.sinc_length / 2; j += taps_per_input)
						normalization += element->workspace.wdpf.sinc_table[j];
					for(j = taps_per_input - i; j <= element->workspace.wdpf.sinc_length / 2; j += taps_per_input)
						normalization += element->workspace.wdpf.sinc_table[j];
					for(j = i; j <= element->workspace.wdpf.sinc_length / 2; j += taps_per_input)
						element->workspace.wdpf.sinc_table[j] /= normalization;
					if(i) {
						for(j = taps_per_input - i; j <= element->workspace.wdpf.sinc_length / 2; j += taps_per_input)
							element->workspace.wdpf.sinc_table[j] /= normalization;
					}
				}
				/* If taps_per_input is even, we need to account for one more normalization without "over-normalizing." */
				if(!((taps_per_input) % 2)) {
					normalization = 0.0;
					for(j = taps_per_input / 2; j <= element->workspace.wdpf.sinc_length / 2; j += taps_per_input)
						normalization += 2 * element->workspace.wdpf.sinc_table[j];
					for(j = taps_per_input / 2; j <= element->workspace.wdpf.sinc_length / 2; j += taps_per_input)
						element->workspace.wdpf.sinc_table[j] /= normalization;
				}
			}
		}

		if(element->make_fir_filters && (!element->workspace.wdpf.fd_fir_window)) {

			/*
			 * Make a frequency-donain window to roll off low and high frequencies
			 */

			element->workspace.wdpf.fd_fir_window = g_malloc(fd_fir_length * sizeof(*element->workspace.wdpf.fd_fir_window));

			/* Initialize to ones */
			for(i = 0; i < fd_fir_length; i++)
				element->workspace.wdpf.fd_fir_window[i] = 1.0;

			int f_nyquist = element->rate / 2;
			double df_per_hz = (fd_fir_length - 1.0) / f_nyquist;
			int freq_res_samples = (int) (fir_alpha + 0.5);

			/* high-pass filter */
			/* Remove low frequencies */
			i_stop = (gint64) (element->high_pass * df_per_hz + 0.5) - freq_res_samples;
			for(i = 0; i < i_stop; i++)
				element->workspace.wdpf.fd_fir_window[i] = 0.0;

			/* Apply half of a Hann window */
			i_start = i_stop;
			i_stop += freq_res_samples;
			for(i = i_start; i < i_stop; i++)
				element->workspace.wdpf.fd_fir_window[i] *= pow(sin((M_PI / 2.0) * (i - i_start) / (i_stop - i_start)), 2.0);

			/* low-pass filter */
			if(element->low_pass > 0) {
				/* Apply half of a Hann window */
				i_start = (gint64) (element->low_pass * df_per_hz + 0.5);
				i_stop = minimum64(fd_fir_length, 1.4 * i_start);
				for(i = i_start; i < i_stop; i++)
					element->workspace.wdpf.fd_fir_window[i] *= pow(cos((M_PI / 2.0) * (i - i_start) / (i_stop - i_start)), 2.0);

				/* Remove high frequencies */
				i_start = i_stop;
				i_stop = fd_fir_length;
				for(i = i_start; i < i_stop; i++)
					element->workspace.wdpf.fd_fir_window[i] = 0.0;
			}

			/*
			 * Make a time-domain window to apply to the FIR filters.
			 */

			if(!element->workspace.wdpf.fir_window) {

				switch(element->window) {
				case GSTLAL_TRANSFERFUNCTION_DPSS:
					element->workspace.wdpf.fir_window = dpss_double(element->fir_length, fir_alpha, 5.0, NULL, FALSE);
					break;

				case GSTLAL_TRANSFERFUNCTION_KAISER:
					element->workspace.wdpf.fir_window = kaiser_double(element->fir_length, M_PI * fir_alpha, NULL, FALSE);
					break;

				case GSTLAL_TRANSFERFUNCTION_DOLPH_CHEBYSHEV:
					element->workspace.wdpf.fir_window = DolphChebyshev_double(element->fir_length, fir_alpha, NULL, FALSE);
					break;

				default:
					GST_ERROR_OBJECT(element, "Invalid window type.  See properties for appropriate window types.");
					g_assert_not_reached();
					break;
				}
			}
		}

		/* intermediate data storage */
		element->workspace.wdpf.leftover_data = g_malloc(element->channels * (element->fft_length - 1) * sizeof(*element->workspace.wdpf.leftover_data));
		element->workspace.wdpf.num_leftover = 0;
		element->workspace.wdpf.ffts = g_malloc(element->channels * fd_fft_length * sizeof(*element->workspace.wdpf.ffts));
		element->workspace.wdpf.num_ffts_in_avg = 0;
		element->workspace.wdpf.autocorrelation_matrix = g_malloc(element->channels * (element->channels - 1) * fd_fft_length * sizeof(*element->workspace.wdpf.autocorrelation_matrix));
		if(element->use_median) {
			element->workspace.wdpf.autocorrelation_median_real = g_malloc((element->num_ffts / 2 + 1) * (element->channels - 1) * (element->channels - 1) * fd_fft_length * sizeof(*element->workspace.wdpf.autocorrelation_median_real));
			element->workspace.wdpf.index_median_real = g_malloc((element->channels - 1) * (element->channels - 1) * fd_fft_length * sizeof(*element->workspace.wdpf.index_median_real));
			element->workspace.wdpf.autocorrelation_median_imag = g_malloc((element->num_ffts / 2 + 1) * (element->channels - 1) * (element->channels - 1) * fd_fft_length * sizeof(*element->workspace.wdpf.autocorrelation_median_imag));
			element->workspace.wdpf.index_median_imag = g_malloc((element->channels - 1) * (element->channels - 1) * fd_fft_length * sizeof(*element->workspace.wdpf.index_median_imag));
		}

		/* Allocate memory for gsl matrix manipulations. The same memory locations will be used repeatedly */
		element->workspace.wdpf.transfer_functions_at_f = gsl_vector_complex_alloc(element->channels - 1);
		element->workspace.wdpf.transfer_functions_solved_at_f = gsl_vector_complex_alloc(element->channels - 1);
		element->workspace.wdpf.autocorrelation_matrix_at_f = gsl_matrix_complex_alloc(element->channels - 1, element->channels - 1);
		element->workspace.wdpf.permutation = gsl_permutation_alloc(element->channels - 1);

		/* Allocate memory for fftw to do Fourier transforms of data. The same memory locations will be used repeatedly */
		gstlal_fftw_lock();

		GST_LOG_OBJECT(element, "starting FFTW planning");

		/* data that will be Fourier transformed into frequency domain */
		element->workspace.wdpf.fft = (complex double *) fftw_malloc(fd_fft_length * sizeof(*element->workspace.wdpf.fft));
		element->workspace.wdpf.plan = fftw_plan_dft_r2c_1d(element->fft_length, (double *) element->workspace.wdpf.fft, element->workspace.wdpf.fft, FFTW_ESTIMATE);

		if(element->make_fir_filters && !element->workspace.wdpf.fir_filter) {

			/* data that will be inverse Fourier transformed back into the time domain */
			element->workspace.wdpf.fir_filter = (complex double *) fftw_malloc(fd_fir_length * sizeof(*element->workspace.wdpf.fir_filter));
			element->workspace.wdpf.fir_plan = fftw_plan_dft_c2r_1d(element->fir_length, element->workspace.wdpf.fir_filter, (double *) element->workspace.wdpf.fir_filter, FFTW_ESTIMATE);
		}
		GST_LOG_OBJECT(element, "FFTW planning complete");

		gstlal_fftw_unlock();

	} else
		success = FALSE;

	return success;
}


/*
 * render()
 */


static GstFlowReturn render(GstBaseSink *sink, GstBuffer *buffer) {

	GSTLALTransferFunction *element = GSTLAL_TRANSFERFUNCTION(sink);
	GstMapInfo mapinfo;
	GstFlowReturn result = GST_FLOW_OK;

	/*
	 * check for discontinuity
	 */

	if(G_UNLIKELY(GST_BUFFER_IS_DISCONT(buffer) || GST_BUFFER_OFFSET(buffer) != element->next_in_offset || !GST_CLOCK_TIME_IS_VALID(element->t0))) {
		element->t0 = GST_BUFFER_PTS(buffer);
		element->offset0 = GST_BUFFER_OFFSET(buffer);
		if(element->parallel_mode) {
			/* In this case, we want to compute the transfer functions on a schedule, not asap */
			element->sample_count = (gint64) (gst_util_uint64_scale_int_round(element->t0, element->rate, GST_SECOND) + element->update_samples - element->update_delay_samples) % (element->update_samples + element->num_ffts * (element->fft_length - element->fft_overlap) + element->fft_overlap);
			if(element->sample_count > element->update_samples)
				element->sample_count -= element->update_samples + element->num_ffts * (element->fft_length - element->fft_overlap) + element->fft_overlap;
		} else if(element->sample_count > element->update_samples) {
			if(*element->transfer_functions == 0.0) {
				/* Transfer functions have not been computed, so scale the delay samples down appropriately to compute them asap */
				gint64 long_samples = element->num_ffts * (element->fft_length - element->fft_overlap) + element->fft_overlap;
				gint64 short_samples = element->min_ffts * (element->fft_length - element->fft_overlap) + element->fft_overlap;
				element->sample_count = element->update_samples - (short_samples * element->update_delay_samples + long_samples - 1) / long_samples;
			} else
				/* Transfer functions have been computed, so apply the usual number of delay samples */
				element->sample_count = element->update_samples - element->update_delay_samples;
		}
		if(element->data_type == GSTLAL_TRANSFERFUNCTION_F32) {
			element->workspace.wspf.num_ffts_in_avg = 0;
			element->workspace.wspf.num_ffts_dropped = 0;
			element->workspace.wspf.num_leftover = 0;
		} else {
			element->workspace.wdpf.num_ffts_in_avg = 0;
			element->workspace.wdpf.num_ffts_dropped = 0;
			element->workspace.wdpf.num_leftover = 0;
		}
	}
	element->next_in_offset = GST_BUFFER_OFFSET_END(buffer);
	GST_DEBUG_OBJECT(element, "have buffer spanning %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(buffer));

	/* Get the data from the buffer */
	gst_buffer_map(buffer, &mapinfo, GST_MAP_READ);

	/* Deal with gaps */
	gboolean gap = FALSE;
	if(GST_BUFFER_FLAG_IS_SET(buffer, GST_BUFFER_FLAG_GAP) && mapinfo.size != 0) {
		gap = TRUE;
		/* Check if we should switch over to post-gap transfer functions */
		if(element->gap_samples < element->use_first_after_gap && element->gap_samples + (gint64) (mapinfo.size / element->unit_size) > element->use_first_after_gap && *element->post_gap_transfer_functions) {
			/* Reset the transfer function count */
			element->num_tfs_since_gap = 0;
			/* Copy samples from post-gap transfer functions and fir filters */
			gint64 i;
			for(i = 0; i < (element->channels - 1) * (element->fir_length / 2 + 1); i++)
				element->transfer_functions[i] = element->post_gap_transfer_functions[i];
			GST_LOG_OBJECT(element, "Just reverted to post-gap transfer functions");
			/* Let other elements know about the update */
			g_object_notify_by_pspec(G_OBJECT(element), properties[ARG_TRANSFER_FUNCTIONS]);
			/* Write transfer functions to the screen or a file if we want */
			if(element->write_to_screen || element->filename)
				write_transfer_functions(element->transfer_functions, gst_element_get_name(element), (double) element->rate / element->fir_length, element->fir_length / 2 + 1, element->channels - 1, 0, 0, element->write_to_screen, element->filename, TRUE);
			if(element->make_fir_filters) {
				for(i = 0; i < (element->channels - 1) * element->fir_length; i++)
					element->fir_filters[i] = element->post_gap_fir_filters[i];
				GST_LOG_OBJECT(element, "Just reverted to post-gap FIR filters");
				/* Let other elements know about the update */
				g_object_notify_by_pspec(G_OBJECT(element), properties[ARG_FIR_FILTERS]);
				/* Write FIR filters to the screen or a file if we want */
				if(element->write_to_screen || element->filename)
					write_fir_filters(element->fir_filters, gst_element_get_name(element), element->fir_length, element->channels - 1, 0, 0, element->write_to_screen, element->filename, TRUE);
			}
			
		}
		/* Track the number of samples since the last non-gap data */
		element->gap_samples += (mapinfo.size / element->unit_size);

		if(element->parallel_mode) {
			/* Update the sample count no matter what */
			element->sample_count += (mapinfo.size / element->unit_size);
			/* Throw away stored data */
			if(element->data_type == GSTLAL_TRANSFERFUNCTION_F32)
				element->workspace.wspf.num_leftover = 0;
			else
				element->workspace.wdpf.num_leftover = 0;
		} else if(element->update_after_gap) {
			/* Trick it into updating things after the gap ends */
			if(*element->transfer_functions == 0.0) {
				/* Transfer functions have not been computed, so scale the delay samples down appropriately to compute them asap */
				gint64 long_samples = element->num_ffts * (element->fft_length - element->fft_overlap) + element->fft_overlap;
				gint64 short_samples = element->min_ffts * (element->fft_length - element->fft_overlap) + element->fft_overlap;
				element->sample_count = element->update_samples - (short_samples * element->update_delay_samples + long_samples - 1) / long_samples;
			} else {
				/* Transfer functions have been computed, so apply the usual number of delay samples */
				element->sample_count = element->update_samples - element->update_delay_samples;
			}
			if(element->data_type == GSTLAL_TRANSFERFUNCTION_F32) {
				element->workspace.wspf.num_ffts_in_avg = 0;
				element->workspace.wspf.num_ffts_dropped = 0;
				element->workspace.wspf.num_leftover = 0;
			} else {
				element->workspace.wdpf.num_ffts_in_avg = 0;
				element->workspace.wdpf.num_ffts_dropped = 0;
				element->workspace.wdpf.num_leftover = 0;
			}
		} else {
			element->sample_count = minimum64(element->sample_count + mapinfo.size / element->unit_size, element->update_samples);
			if(element->data_type == GSTLAL_TRANSFERFUNCTION_F32) {
				element->workspace.wspf.num_ffts_in_avg = 0;
				element->workspace.wspf.num_ffts_dropped = 0;
				element->workspace.wspf.num_leftover = 0;
			} else {
				element->workspace.wdpf.num_ffts_in_avg = 0;
				element->workspace.wdpf.num_ffts_dropped = 0;
				element->workspace.wdpf.num_leftover = 0;
			}
		}
	} else {
		/* Increment the sample count */
		element->sample_count += (mapinfo.size / element->unit_size);
		/* Reset the count of gap samples */
		element->gap_samples = 0;
	}

	/* Check whether we need to do anything with this data */
	if(element->sample_count > element->update_samples && mapinfo.size) {
		gboolean success;
		if(element->data_type == GSTLAL_TRANSFERFUNCTION_F32) {
			/* If we are just beginning to compute new transfer functions with this data, initialize memory that we will fill to zero */
			if(!element->workspace.wspf.num_ffts_in_avg) {
				memset(element->workspace.wspf.autocorrelation_matrix, 0, element->channels * (element->channels - 1) * (element->fft_length / 2 + 1) * sizeof(*element->workspace.wspf.autocorrelation_matrix));
				element->t_start_tf = (double) (gst_util_uint64_scale_int_round(GST_BUFFER_PTS(buffer) + GST_BUFFER_DURATION(buffer), element->rate, GST_SECOND) - element->sample_count + element->update_samples) / element->rate;
			}
			/* Send the data to a function to compute fft's and transfer functions */
			success = find_transfer_functions_float(element, (float *) mapinfo.data, mapinfo.size, GST_BUFFER_PTS(buffer), gap);
		} else {
			/* If we are just beginning to compute new transfer functions with this data, initialize memory that we will fill to zero */
			if(!element->workspace.wdpf.num_ffts_in_avg) {
				memset(element->workspace.wdpf.autocorrelation_matrix, 0, element->channels * (element->channels - 1) * (element->fft_length / 2 + 1) * sizeof(*element->workspace.wdpf.autocorrelation_matrix));
				element->t_start_tf = (double) (gst_util_uint64_scale_int_round(GST_BUFFER_PTS(buffer) + GST_BUFFER_DURATION(buffer), element->rate, GST_SECOND) - element->sample_count + element->update_samples) / element->rate;
			}
			/* Send the data to a function to compute fft's and transfer functions */
			success = find_transfer_functions_double(element, (double *) mapinfo.data, mapinfo.size, GST_BUFFER_PTS(buffer), gap);
		}

		if(!success && !element->parallel_mode) {
			/* Try again */
			element->sample_count = element->update_samples;
			if(element->data_type == GSTLAL_TRANSFERFUNCTION_F32) {
				element->workspace.wspf.num_ffts_in_avg = 0;
				element->workspace.wspf.num_ffts_dropped = 0;
				element->workspace.wspf.num_leftover = 0;
			} else {
				element->workspace.wdpf.num_ffts_in_avg = 0;
				element->workspace.wdpf.num_ffts_dropped = 0;
				element->workspace.wdpf.num_leftover = 0;
			}
		}
	}
	gst_buffer_unmap(buffer, &mapinfo);

	return result;
}


/*
 * ============================================================================
 *
 *			      GObject Methods
 *
 * ============================================================================
 */


/*
 * properties
 */


static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec)
{
	GSTLALTransferFunction *element = GSTLAL_TRANSFERFUNCTION(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_FFT_LENGTH:
		element->fft_length = g_value_get_int64(value);
		break;

	case ARG_FFT_OVERLAP:
		element->fft_overlap = g_value_get_int64(value);
		break;

	case ARG_NUM_FFTS:
		element->num_ffts = g_value_get_int64(value);
		break;

	case ARG_MIN_FFTS:
		element->min_ffts = g_value_get_int64(value);
		break;

	case ARG_USE_MEDIAN:
		element->use_median = g_value_get_boolean(value);
		break;

	case ARG_UPDATE_SAMPLES:
		element->update_samples = g_value_get_int64(value);
		break;

	case ARG_UPDATE_AFTER_GAP:
		element->update_after_gap = g_value_get_boolean(value);
		break;

	case ARG_USE_FIRST_AFTER_GAP:
		element->use_first_after_gap = g_value_get_int64(value);
		break;

	case ARG_UPDATE_DELAY_SAMPLES:
		element->update_delay_samples = g_value_get_int64(value);
		break;

	case ARG_PARALLEL_MODE:
		element->parallel_mode = g_value_get_boolean(value);
		break;

	case ARG_WRITE_TO_SCREEN:
		element->write_to_screen = g_value_get_boolean(value);
		break;

	case ARG_FILENAME:
		element->filename = g_value_dup_string(value);
		break;

	case ARG_MAKE_FIR_FILTERS:
		element->make_fir_filters = g_value_get_double(value);
		break;

	case ARG_FIR_LENGTH:
		element->fir_length = g_value_get_int64(value);
		break;

	case ARG_FREQUENCY_RESOLUTION:
		element->frequency_resolution = g_value_get_double(value);
		break;

	case ARG_HIGH_PASS:
		element->high_pass = g_value_get_double(value);
		break;

	case ARG_LOW_PASS:
		element->low_pass = g_value_get_double(value);
		break;

	case ARG_NOTCH_FREQUENCIES:
		if(element->notch_frequencies) {
			g_free(element->notch_frequencies);
			element->notch_frequencies = NULL;
		}
		element->num_notches = gst_value_array_get_size(value);
		if(element->num_notches % 2)
			GST_ERROR_OBJECT(element, "Array length for property notch_frequencies must be even");
		element->notch_frequencies = g_malloc(element->num_notches * sizeof(double));
		int k;
		for(k = 0; k < element->num_notches; k++)
			element->notch_frequencies[k] = g_value_get_double(gst_value_array_get_value(value, k));
		element->num_notches /= 2;

		break;

	case ARG_FIR_TIMESHIFT:
		element->fir_timeshift = g_value_get_int64(value);
		break;

	case ARG_WINDOW:
		element->window = g_value_get_enum(value);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *pspec)
{
	GSTLALTransferFunction *element = GSTLAL_TRANSFERFUNCTION(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_FFT_LENGTH:
		g_value_set_int64(value, element->fft_length);
		break;

	case ARG_FFT_OVERLAP:
		g_value_set_int64(value, element->fft_overlap);
		break;

	case ARG_NUM_FFTS:
		g_value_set_int64(value, element->num_ffts);
		break;

	case ARG_MIN_FFTS:
		g_value_set_int64(value, element->min_ffts);
		break;

	case ARG_USE_MEDIAN:
		g_value_set_boolean(value, element->use_median);
		break;

	case ARG_UPDATE_SAMPLES:
		g_value_set_int64(value, element->update_samples);
		break;

	case ARG_UPDATE_AFTER_GAP:
		g_value_set_boolean(value, element->update_after_gap);
		break;

	case ARG_USE_FIRST_AFTER_GAP:
		g_value_set_int64(value, element->use_first_after_gap);
		break;

	case ARG_UPDATE_DELAY_SAMPLES:
		g_value_set_int64(value, element->update_delay_samples);
		break;

	case ARG_PARALLEL_MODE:
		g_value_set_boolean(value, element->parallel_mode);
		break;

	case ARG_WRITE_TO_SCREEN:
		g_value_set_boolean(value, element->write_to_screen);
		break;

	case ARG_FILENAME:
		g_value_set_string(value, element->filename);
		break;

	case ARG_MAKE_FIR_FILTERS:
		g_value_set_double(value, element->make_fir_filters);
		break;

	case ARG_FIR_LENGTH:
		g_value_set_int64(value, element->fir_length);
		break;

	case ARG_FREQUENCY_RESOLUTION:
		g_value_set_double(value, element->frequency_resolution);
		break;

	case ARG_HIGH_PASS:
		g_value_set_double(value, element->high_pass);
		break;

	case ARG_LOW_PASS:
		g_value_set_double(value, element->low_pass);
		break;

	case ARG_NOTCH_FREQUENCIES: ;
		GValue valuearray = G_VALUE_INIT;
		g_value_init(&valuearray, GST_TYPE_ARRAY);
		int k;
		for(k = 0; k < 2 * element->num_notches; k++) {
			GValue notch = G_VALUE_INIT;
			g_value_init(&notch, G_TYPE_DOUBLE);
			g_value_set_double(&notch, element->notch_frequencies[k]);
			gst_value_array_append_value(&valuearray, &notch);
			g_value_unset(&notch);
		}
		g_value_copy(&valuearray, value);
		g_value_unset(&valuearray);

		break;

	case ARG_FIR_TIMESHIFT:
		g_value_set_int64(value, element->fir_timeshift);
		break;

	case ARG_TRANSFER_FUNCTIONS:
		if(element->transfer_functions) {
			double *double_tfs = (double *) element->transfer_functions;
			GValue va = G_VALUE_INIT;
			g_value_init(&va, GST_TYPE_ARRAY);
			int i, j;
			for(i = 0; i < element->channels - 1; i++) {
				GValue va_row = G_VALUE_INIT;
				g_value_init(&va_row, GST_TYPE_ARRAY);
				for(j = 0; j < element->fir_length + 2; j++) {
					GValue v = G_VALUE_INIT;
					g_value_init(&v, G_TYPE_DOUBLE);
					g_value_set_double(&v, double_tfs[i * (element->fir_length + 2) + j]);
					gst_value_array_append_value(&va_row, &v);
					g_value_unset(&v);
				}
				gst_value_array_append_value(&va, &va_row);
				g_value_unset(&va_row);
			}
			g_value_copy(&va, value);
			g_value_unset(&va);
		}
		break;

	case ARG_FIR_FILTERS:
		if(element->fir_filters) {
			GValue varray = G_VALUE_INIT;
			g_value_init(&varray, GST_TYPE_ARRAY);
			int m, n;
			for(m = 0; m < element->channels - 1; m++) {
				GValue varray_row = G_VALUE_INIT;
				g_value_init(&varray_row, GST_TYPE_ARRAY);
				for(n = 0; n < element->fir_length; n++) {
					GValue val = G_VALUE_INIT;
					g_value_init(&val, G_TYPE_DOUBLE);
					g_value_set_double(&val, element->fir_filters[m * element->fir_length + n]);
					gst_value_array_append_value(&varray_row, &val);
					g_value_unset(&val);
				}
				gst_value_array_append_value(&varray, &varray_row);
				g_value_unset(&varray_row);
			}
			g_value_copy(&varray, value);
			g_value_unset(&varray);
		}
		break;

	case ARG_FIR_ENDTIME:
		g_value_set_uint64(value, element->fir_endtime);
		break;

	case ARG_WINDOW:
		g_value_set_enum(value, element->window);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * finalize()
 */


static void finalize(GObject *object) {

	GSTLALTransferFunction *element = GSTLAL_TRANSFERFUNCTION(object);

	if(element->post_gap_transfer_functions) {
		g_free(element->post_gap_transfer_functions);
		element->post_gap_transfer_functions = NULL;
	}
	if(element->post_gap_fir_filters) {
		g_free(element->post_gap_fir_filters);
		element->post_gap_fir_filters = NULL;
	}
	if(element->transfer_functions) {
		g_free(element->transfer_functions);
		element->transfer_functions = NULL;
	}
	if(element->fir_filters) {
		g_free(element->fir_filters);
		element->fir_filters = NULL;
	}
	if(element->notch_frequencies) {
		g_free(element->notch_frequencies);
		element->notch_frequencies = NULL;
	}
	if(element->notch_indices) {
		g_free(element->notch_indices);
		element->notch_indices = NULL;
	}
	if(element->data_type == GSTLAL_TRANSFERFUNCTION_F32) {

		/* free allocated memory in workspace */
		g_free(element->workspace.wspf.fft_window);
		element->workspace.wspf.fft_window = NULL;
		g_free(element->workspace.wspf.leftover_data);
		element->workspace.wspf.leftover_data = NULL;
		g_free(element->workspace.wspf.ffts);
		element->workspace.wspf.ffts = NULL;
		g_free(element->workspace.wspf.autocorrelation_matrix);
		element->workspace.wspf.autocorrelation_matrix = NULL;
		if(element->make_fir_filters) {
			g_free(element->workspace.wspf.fd_fir_window);
			element->workspace.wspf.fd_fir_window = NULL;
			g_free(element->workspace.wspf.sinc_table);
			element->workspace.wspf.sinc_table = NULL;
			g_free(element->workspace.wspf.fir_window);
			element->workspace.wspf.fir_window = NULL;
		}

		if(element->use_median) {
			g_free(element->workspace.wspf.autocorrelation_median_real);
			element->workspace.wspf.autocorrelation_median_real = NULL;
			g_free(element->workspace.wspf.index_median_real);
			element->workspace.wspf.index_median_real = NULL;
			g_free(element->workspace.wspf.autocorrelation_median_imag);
			element->workspace.wspf.autocorrelation_median_imag = NULL;
			g_free(element->workspace.wspf.index_median_imag);
			element->workspace.wspf.index_median_imag = NULL;
		}

		/* free gsl stuff in workspace */
		gsl_vector_complex_free(element->workspace.wspf.transfer_functions_at_f);
		element->workspace.wspf.transfer_functions_at_f = NULL;
		gsl_vector_complex_free(element->workspace.wspf.transfer_functions_solved_at_f);
		element->workspace.wspf.transfer_functions_solved_at_f = NULL;
		gsl_matrix_complex_free(element->workspace.wspf.autocorrelation_matrix_at_f);
		element->workspace.wspf.autocorrelation_matrix_at_f = NULL;
		gsl_permutation_free(element->workspace.wspf.permutation);
		element->workspace.wspf.permutation = NULL;

		/* free fftwf stuff in workspace */
		gstlal_fftw_lock();
		fftwf_free(element->workspace.wspf.fft);
		element->workspace.wspf.fft = NULL;
		fftwf_destroy_plan(element->workspace.wspf.plan);
		if(element->make_fir_filters) {
			fftwf_free(element->workspace.wspf.fir_filter);
			element->workspace.wspf.fir_filter = NULL;
			fftwf_destroy_plan(element->workspace.wspf.fir_plan);
		}
		gstlal_fftw_unlock();

	} else {

		/* free allocated memory in workspace */
		g_free(element->workspace.wdpf.fft_window);
		element->workspace.wdpf.fft_window = NULL;
		g_free(element->workspace.wdpf.leftover_data);
		element->workspace.wdpf.leftover_data = NULL;
		g_free(element->workspace.wdpf.ffts);
		element->workspace.wdpf.ffts = NULL;
		g_free(element->workspace.wdpf.autocorrelation_matrix);
		element->workspace.wdpf.autocorrelation_matrix = NULL;
		if(element->make_fir_filters) {
			g_free(element->workspace.wdpf.fd_fir_window);
			element->workspace.wdpf.fd_fir_window = NULL;
			g_free(element->workspace.wdpf.sinc_table);
			element->workspace.wdpf.sinc_table = NULL;
			g_free(element->workspace.wdpf.fir_window);
			element->workspace.wdpf.fir_window = NULL;
		}

		if(element->use_median) {
			g_free(element->workspace.wdpf.autocorrelation_median_real);
			element->workspace.wdpf.autocorrelation_median_real = NULL;
			g_free(element->workspace.wdpf.index_median_real);
			element->workspace.wdpf.index_median_real = NULL;
			g_free(element->workspace.wdpf.autocorrelation_median_imag);
			element->workspace.wdpf.autocorrelation_median_imag = NULL;
			g_free(element->workspace.wdpf.index_median_imag);
			element->workspace.wdpf.index_median_imag = NULL;
		}

		/* free gsl stuff in workspace */
		gsl_vector_complex_free(element->workspace.wdpf.transfer_functions_at_f);
		element->workspace.wdpf.transfer_functions_at_f = NULL;
		gsl_vector_complex_free(element->workspace.wdpf.transfer_functions_solved_at_f);
		element->workspace.wdpf.transfer_functions_solved_at_f = NULL;
		gsl_matrix_complex_free(element->workspace.wdpf.autocorrelation_matrix_at_f);
		element->workspace.wdpf.autocorrelation_matrix_at_f = NULL;
		gsl_permutation_free(element->workspace.wdpf.permutation);
		element->workspace.wdpf.permutation = NULL;

		/* free fftw stuff in workspace */
		gstlal_fftw_lock();
		fftw_free(element->workspace.wdpf.fft);
		element->workspace.wdpf.fft = NULL;
		fftw_destroy_plan(element->workspace.wdpf.plan);
		if(element->make_fir_filters) {
			fftw_free(element->workspace.wdpf.fir_filter);
			element->workspace.wdpf.fir_filter = NULL;
			fftw_destroy_plan(element->workspace.wdpf.fir_plan);
		}
		gstlal_fftw_unlock();
	}

	G_OBJECT_CLASS(gstlal_transferfunction_parent_class)->finalize(object);
}


/*
 * class_init()
 */


#define CAPS \
	"audio/x-raw, " \
	"rate = (int) [1, MAX], " \
	"channels = (int) [2, MAX], " \
	"format = (string) {"GST_AUDIO_NE(F32)", "GST_AUDIO_NE(F64)"}, " \
	"layout = (string) interleaved, " \
	"channel-mask = (bitmask) 0"


static void gstlal_transferfunction_class_init(GSTLALTransferFunctionClass *klass) {

	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GstBaseSinkClass *gstbasesink_class = GST_BASE_SINK_CLASS(klass);

	gstbasesink_class->start = GST_DEBUG_FUNCPTR(start);
	gstbasesink_class->event = GST_DEBUG_FUNCPTR(event);
	gstbasesink_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
	gstbasesink_class->render = GST_DEBUG_FUNCPTR(render);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	gst_element_class_set_details_simple(
		element_class,
		"Compute transfer functions",
		"Sink",
		"Compute the transfer function(s) between an output signal and one or more input signals.\n\t\t\t   "
		"This sink element only has one sink pad, so it requires interleaving all input data. The\n\t\t\t   "
		"first channel is treated as the output of the transfer function, and the rest are\n\t\t\t   "
		"treated as inputs. If there are multiple inputs, the transfer functions are optimized\n\t\t\t   "
		"to minimize the RMS difference between the output and the approximated output.",
		"Aaron Viets <aaron.viets@ligo.org>"
	);

	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"sink",
			GST_PAD_SINK,
			GST_PAD_ALWAYS,
			gst_caps_from_string(CAPS)
		)
	);


	properties[ARG_FFT_LENGTH] = g_param_spec_int64(
		"fft-length",
		"FFT Length",
		"Length in samples of the FFTs used to compute the transfer function(s)",
		1, G_MAXINT64, 16384,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_FFT_OVERLAP] = g_param_spec_int64(
		"fft-overlap",
		"FFT Overlap",
		"The overlap in samples of the FFTs used to compute the transfer function(s)",
		-G_MAXINT64, G_MAXINT64, 8192,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_NUM_FFTS] = g_param_spec_int64(
		"num-ffts",
		"Number of FFTs",
		"Number of FFTs that will be averaged to compute the transfer function(s)",
		1, G_MAXINT64, 16,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_MIN_FFTS] = g_param_spec_int64(
		"min-ffts",
		"Minimum number of FFTs",
		"Number of FFTs necessary to compute the first set of transfer functions.",
		1, G_MAXINT64, G_MAXINT64,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_USE_MEDIAN] = g_param_spec_boolean(
		"use-median",
		"Use Median",
		"Set to True in order to take a median of transfer functions instead of a\n\t\t\t"
		"mean. This is more expensive in terms of memory and CPU, but it makes the\n\t\t\t"
		"result less sensitive to occasional outliers such as glitches.",
		FALSE,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_UPDATE_SAMPLES] = g_param_spec_int64(
		"update-samples",
		"Update Samples",
		"Number of input samples after which to update the transfer function(s)",
		0, G_MAXINT64, 58982400,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_UPDATE_AFTER_GAP] = g_param_spec_boolean(
		"update-after-gap",
		"Update After Gap",
		"Set to True in order to update the transfer function(s) after a gap in the\n\t\t\t"
		"input data.",
		FALSE,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_USE_FIRST_AFTER_GAP] = g_param_spec_int64(
		"use-first-after-gap",
		"Use first computed transfer function after a gap",
		"If set to a positive value, the element will revert to the most recent\n\t\t\t"
		"transfer function computed just after a gap, instead of using the most\n\t\t\t"
		"recently-computed transfer function. The integer value of the property\n\t\t\t"
		"(if positive) is the number of gap samples necessary for this to take\n\t\t\t"
		"effect.",
		0, G_MAXINT64, 0,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_UPDATE_DELAY_SAMPLES] = g_param_spec_int64(
		"update-delay-samples",
		"Update Delay Samples",
		"How many extra samples to wait for after a would-be update of transfer\n\t\t\t"
		"functions.",
		0, G_MAXINT64, 0,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_PARALLEL_MODE] = g_param_spec_boolean(
		"parallel-mode",
		"Parallel Mode",
		"When set to true, output produced will be independent of start time.\n\t\t\t"
		"Transfer function calculations are started and finished on a predetermined\n\t\t\t"
		"schedule. This is useful when running jobs in parallel on contiguous sets\n\t\t\t"
		"of data.",
		FALSE,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_WRITE_TO_SCREEN] = g_param_spec_boolean(
		"write-to-screen",
		"Write to Screen",
		"Set to True in order to write transfer functions and/or FIR filters to\n\t\t\t"
		"the screen.",
		FALSE,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_FILENAME] = g_param_spec_string(
		"filename",
		"Filename",
		"Name of file to write transfer functions and/or FIR filters to. If not\n\t\t\t"
		"given, no file is produced.",
		NULL,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_MAKE_FIR_FILTERS] = g_param_spec_double(
		"make-fir-filters",
		"Make FIR Filters",
		"If set to to a non-zero value, FIR filters will be produced each time the\n\t\t\t"
		"transfer functions are computed with this gain factor relative to the\n\t\t\t"
		"transfer functions. If unset (or set to 0), no FIR filters are produced.",
		-G_MAXDOUBLE, G_MAXDOUBLE, 0,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_FIR_LENGTH] = g_param_spec_int64(
		"fir-length",
		"FIR filter length",
		"Length in samples of FIR filters produced. The length of the transfer\n\t\t\t"
		"functions produced is also compute from this, as fir-length / 2 + 1. If\n\t\t\t"
		"unset, the length of the transfer functions and FIR filters will be based on\n\t\t\t"
		"fft-length. Must be an even number.",
		0, G_MAXINT64, 0,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_FREQUENCY_RESOLUTION] = g_param_spec_double(
		"frequency-resolution",
		"Frequency resolution",
		"Frequency resolution of the transfer functions and FIR filters in Hz.\n\t\t\t"
		"This must be greater than or equal to sample rate/fir-length and sample\n\t\t\t"
		"rate/fft-length in order to be effective.",
		0, G_MAXDOUBLE, 0,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_HIGH_PASS] = g_param_spec_double(
		"high-pass",
		"High Pass",
		"The high-pass cutoff frequency (in Hz) of the FIR filters. If zero, no\n\t\t\t"
		"high-pass cutoff is added. The property frequency-resolution takes\n\t\t\t"
		"precedence over this.",
		0, G_MAXDOUBLE, 0,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_LOW_PASS] = g_param_spec_double(
		"low-pass",
		"Low Pass",
		"The low-pass cutoff frequency (in Hz) of the FIR filters. If zero, no\n\t\t\t"
		"low-pass cutoff is added. The property frequency-resolution takes\n\t\t\t"
		"precedence over this.",
		0, G_MAXDOUBLE, 0,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_NOTCH_FREQUENCIES] = gst_param_spec_array(
		"notch-frequencies",
		"Notch Frequencies",
		"Array of minima and maxima of frequency ranges where the Fourier transform\n\t\t\t"
		"of the signal of interest will be replaced by a straight line. This can be\n\t\t\t"
		"useful if there are loud lines in the signal that are not present in the\n\t\t\t"
		"witness channels.",
		g_param_spec_double(
			"frequency",
			"Frequency",
			"A frequency in the array, either a minimum or maximum of a notch.",
			-G_MAXDOUBLE, G_MAXDOUBLE, 0.0,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		),
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_FIR_TIMESHIFT] = g_param_spec_int64(
		"fir-timeshift",
		"FIR time-shift",
		"The number of nanoseconds after the completion of a FIR filter calculation\n\t\t\t"
		"that the FIR filter remains valid for use on the filtered data.  This is\n\t\t\t"
		"added to the presentation timestamp when the filter is completed to compute\n\t\t\t"
		"the fir-endtime property.  Default is to disable.",
		G_MININT64, G_MAXINT64, G_MAXINT64,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_TRANSFER_FUNCTIONS] = gst_param_spec_array(
		"transfer-functions",
		"Transfer Functions",
		"Array of the computed transfer functions",
		gst_param_spec_array(
			"transfer-function",
			"Transfer Function",
			"A single transfer function",
			g_param_spec_double(
				"value",
				"Value",
				"Value of the transfer function at a particular frequency",
				-G_MAXDOUBLE, G_MAXDOUBLE, 0.0,
				G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
			),
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		),
		G_PARAM_READABLE | G_PARAM_STATIC_STRINGS
	);
	properties[ARG_FIR_FILTERS] = gst_param_spec_array(
		"fir-filters",
		"FIR Filters",
		"Array of the computed FIR filters",
		gst_param_spec_array(
			"fir-filter",
			"FIR Filter",
			"A single FIR filter",
			g_param_spec_double(
				"sample",
				"Sample",
				"A sample from the FIR filter",
				-G_MAXDOUBLE, G_MAXDOUBLE, 0.0,
				G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
			),
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		),
		G_PARAM_READABLE | G_PARAM_STATIC_STRINGS
	);
	properties[ARG_FIR_ENDTIME] = g_param_spec_uint64(
		"fir-endtime",
		"FIR end time",
		"The time when a computed FIR filter ceases to be valid for use on\n\t\t\t"
		"filtered data.  This can be compared to the presentation timestamps of the\n\t\t\t"
		"filtered data to determine whether the filter is still valid.  Default is\n\t\t\t"
		"to disable.",
		0, G_MAXUINT64, G_MAXUINT64,
		G_PARAM_READABLE | G_PARAM_STATIC_STRINGS
	);
	properties[ARG_WINDOW] = g_param_spec_enum(
		"window",
		"Window Function",
		"What window function to apply to incoming data and to the FIR filters",
		GSTLAL_TRANSFERFUNCTION_WINDOW_TYPE,
		GSTLAL_TRANSFERFUNCTION_DPSS,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);


	g_object_class_install_property(
		gobject_class,
		ARG_FFT_LENGTH,
		properties[ARG_FFT_LENGTH]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_FFT_OVERLAP,
		properties[ARG_FFT_OVERLAP]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_NUM_FFTS,
		properties[ARG_NUM_FFTS]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_MIN_FFTS,
		properties[ARG_MIN_FFTS]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_USE_MEDIAN,
		properties[ARG_USE_MEDIAN]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_UPDATE_SAMPLES,
		properties[ARG_UPDATE_SAMPLES]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_UPDATE_AFTER_GAP,
		properties[ARG_UPDATE_AFTER_GAP]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_USE_FIRST_AFTER_GAP,
		properties[ARG_USE_FIRST_AFTER_GAP]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_UPDATE_DELAY_SAMPLES,
		properties[ARG_UPDATE_DELAY_SAMPLES]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_PARALLEL_MODE,
		properties[ARG_PARALLEL_MODE]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_WRITE_TO_SCREEN,
		properties[ARG_WRITE_TO_SCREEN]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_FILENAME,
		properties[ARG_FILENAME]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_MAKE_FIR_FILTERS,
		properties[ARG_MAKE_FIR_FILTERS]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_FIR_LENGTH,
		properties[ARG_FIR_LENGTH]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_FREQUENCY_RESOLUTION,
		properties[ARG_FREQUENCY_RESOLUTION]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_HIGH_PASS,
		properties[ARG_HIGH_PASS]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_LOW_PASS,
		properties[ARG_LOW_PASS]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_NOTCH_FREQUENCIES,
		properties[ARG_NOTCH_FREQUENCIES]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_FIR_TIMESHIFT,
		properties[ARG_FIR_TIMESHIFT]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_TRANSFER_FUNCTIONS,
		properties[ARG_TRANSFER_FUNCTIONS]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_FIR_FILTERS,
		properties[ARG_FIR_FILTERS]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_FIR_ENDTIME,
		properties[ARG_FIR_ENDTIME]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_WINDOW,
		properties[ARG_WINDOW]
	);
}


/*
 * init()
 */


static void gstlal_transferfunction_init(GSTLALTransferFunction *element) {

	g_signal_connect(G_OBJECT(element), "notify::transfer-functions", G_CALLBACK(rebuild_workspace_and_reset), NULL);
	g_signal_connect(G_OBJECT(element), "notify::fir-filters", G_CALLBACK(rebuild_workspace_and_reset), NULL);
	element->post_gap_transfer_functions = NULL;
	element->post_gap_fir_filters = NULL;
	element->transfer_functions = NULL;
	element->fir_filters = NULL;
	element->notch_frequencies = NULL;
	element->notch_indices = NULL;
	element->num_notches = 0;
	element->rate = 0;
	element->unit_size = 0;
	element->channels = 0;
	element->gap_samples = 0;
	element->num_tfs_since_gap = 0;
	element->computed_full_tfs = FALSE;

	gst_base_sink_set_sync(GST_BASE_SINK(element), FALSE);
	gst_base_sink_set_async_enabled(GST_BASE_SINK(element), FALSE);
}


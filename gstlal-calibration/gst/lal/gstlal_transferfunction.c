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
 * @short_description:  Compute transfer function between two or more
 * time series (up to 11 total).
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
#include <gstlal_transferfunction.h>


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
	ARG_UPDATE_SAMPLES,
	ARG_UPDATE_AFTER_GAP,
	ARG_WRITE_TO_SCREEN,
	ARG_FILENAME,
	ARG_MAKE_FIR_FILTERS,
	ARG_HIGH_PASS,
	ARG_LOW_PASS,
	ARG_TRANSFER_FUNCTIONS,
	ARG_FIR_FILTERS,
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


static void write_transfer_functions(complex double *tfs, char *element_name, gint64 rows, int columns, gboolean write_to_screen, char *filename) {
	gint64 i;
	int j, j_stop;
	if(write_to_screen) {
		g_print("\n\n========= Transfer functions computed by %s =========\n", element_name);
		for(j = 1; j < columns; j++)
			g_print("ch%d -> ch0\t", j);
		g_print("ch%d -> ch0\n\n", columns);

		j_stop = columns - 1;
		for(i = 0; i < rows; i++) {
			for(j = 0; j < j_stop; j++)
				g_print("%10e + %10e i\t", creal(tfs[i + j * rows]), cimag(tfs[i + j * rows]));
			g_print("%10e + %10e i\n", creal(tfs[i + j_stop * rows]), cimag(tfs[i + j_stop * rows]));
		}
		g_print("\n\n");
	}

	if(filename) {
		FILE *fp;
		fp = fopen(filename, "w");
		g_fprintf(fp, "========= Transfer functions computed by %s =========\n", element_name);
		for(j = 1; j < columns; j++)
			g_fprintf(fp, "ch%d -> ch0\t", j);
		g_fprintf(fp, "ch%d -> ch0\n\n", columns);

		j_stop = columns - 1;
		for(i = 0; i < rows; i++) {
			for(j = 0; j < j_stop; j++)
				g_fprintf(fp, "%10e + %10e i\t", creal(tfs[i + j * rows]), cimag(tfs[i + j * rows]));
			g_fprintf(fp, "%10e + %10e i\n", creal(tfs[i + j_stop * rows]), cimag(tfs[i + j_stop * rows]));
		}
		fclose(fp);
	}
	g_free(element_name);
}


static void write_fir_filters(double *filters, char *element_name, gint64 rows, int columns, gboolean write_to_screen, char *filename) {
	gint64 i;
	int j, j_stop;
	if(write_to_screen) {
		g_print("============ FIR filters computed by %s ============\n", element_name);
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
		fp = fopen(filename, "w");
		g_fprintf(fp, "\n\n============ FIR filters computed by %s ============\n", element_name);
		for(j = 1; j < columns; j++)
			g_fprintf(fp, "ch%d -> ch0\t", j);
		g_fprintf(fp, "ch%d -> ch0\n\n", columns);

		j_stop = columns - 1;
		for(i = 0; i < rows; i++) {
			for(j = 0; j < j_stop; j++)
				g_fprintf(fp, "%10e\t", filters[i + j * rows]);
			g_fprintf(fp, "%10e\n", filters[i + j_stop * rows]);
		}
		fclose(fp);
	}
	g_free(element_name);
}


#define DEFINE_UPDATE_TRANSFER_FUNCTIONS(DTYPE) \
static void update_transfer_functions_ ## DTYPE(complex DTYPE *autocorrelation_matrix, int num_tfs, gint64 length_tfs, gint64 num_avg, gsl_vector_complex *transfer_functions_at_f, gsl_vector_complex *transfer_functions_solved_at_f, gsl_matrix_complex *autocorrelation_matrix_at_f, gsl_permutation *permutation, complex double *transfer_functions) { \
 \
	gint64 i, first_index; \
	int j, j_stop, signum; \
	complex double z; \
	gsl_complex gslz; \
	for(i = 0; i < length_tfs; i++) { \
		/* First, copy samples at a specific frequency from the big autocorrelation matrix to the gsl vector transfer_functions_at_f */ \
		first_index = i * num_tfs; \
		for(j = 0; j < num_tfs; j++) { \
			z = (complex double) autocorrelation_matrix[first_index + j] / num_avg; \
			gsl_vector_complex_set(transfer_functions_at_f, j, gsl_complex_rect(creal(z), cimag(z))); \
		} \
 \
		/* Next, copy samples at a specific frequency from the big autocorrelation matrix to the gsl matrix autocorrelation_matrix_at_f */ \
		j_stop = num_tfs * num_tfs; \
		first_index += num_tfs; \
		for(j = 0; j < j_stop; j++) { \
			z = (complex double) autocorrelation_matrix[first_index + j] / num_avg; \
			gsl_matrix_complex_set(autocorrelation_matrix_at_f, j / num_tfs, j % num_tfs, gsl_complex_rect(creal(z), cimag(z))); \
			/* autocorrelation_matrix_at_f->data[j] = gsl_complex_rect(creal(z), cimag(z)); */ \
		} \
 \
		/* Now solve [autocorrelation_matrix_at_f] [transfer_functions(f)] = [transfer_functions_at_f] for [transfer_functions(f)] using gsl */ \
		gsl_linalg_complex_LU_decomp(autocorrelation_matrix_at_f, permutation, &signum); \
		gsl_linalg_complex_LU_solve (autocorrelation_matrix_at_f, permutation, transfer_functions_at_f, transfer_functions_solved_at_f); \
 \
		/* Now copy the result into transfer_functions */ \
		for(j = 0; j < num_tfs; j++) { \
			gslz = gsl_vector_complex_get(transfer_functions_solved_at_f, j); \
			transfer_functions[j * length_tfs + i] = GSL_REAL(gslz) + I * GSL_IMAG(gslz); \
		} \
	} \
}


DEFINE_UPDATE_TRANSFER_FUNCTIONS(float);
DEFINE_UPDATE_TRANSFER_FUNCTIONS(double);


#define DEFINE_UPDATE_FIR_FILTERS(DTYPE, F_OR_BLANK) \
static void update_fir_filters_ ## DTYPE(complex double *transfer_functions, int num_tfs, gint64 length_tfs, int sample_rate, complex DTYPE *fir_filter, fftw ## F_OR_BLANK ## _plan fir_plan, DTYPE *fd_window, double *tukey, double *fir_filters) { \
 \
	int i; \
	gint64 j, zero_index, fir_length; \
	DTYPE df, delay, exp_arg; \
	fir_length = 2 * (length_tfs - 1); \
	df = sample_rate / 2.0 / (length_tfs - 1); /* frequency spacing is Nyquist frequency / number of frequency increments */ \
	delay = (length_tfs - 1.0) / sample_rate; /* number of samples of delay is length of transfer functions - 1 */ \
	exp_arg = -2.0 * M_PI * I * df * delay; \
	for(i = 0; i < num_tfs; i++) { \
		/*
		 * First, copy samples from transfer_functions to fir_filter for fftw(f) to take an inverse fft.
		 * The frequency domain window is applied here to roll off low and high freqneucies.
		 * A delay is also added in order to center the filter in time.
		 */ \
		for(j = 0; j < length_tfs; j++) \
			fir_filter[j] = fd_window[j] * cexp ## F_OR_BLANK(exp_arg * j) * transfer_functions[i * length_tfs + j]; \
 \
		/* Now make fir_filter conjugate symmetric by filling the remaining memory with complex conjugates */ \
		zero_index = length_tfs - 1; \
		for(j = 1; j <= length_tfs - 2; j++) \
			fir_filter[zero_index + j] = conj ## F_OR_BLANK(fir_filter[zero_index - j]); \
 \
		/* Take the inverse Fourier transform */ \
		fftw ## F_OR_BLANK ## _execute(fir_plan); \
 \
		/* Apply the Tukey window and copy to fir_filters */ \
		for(j = 0; j < fir_length; j++) \
			fir_filters[i * fir_length + j] = tukey[j] * fir_filter[j]; \
	} \
}


DEFINE_UPDATE_FIR_FILTERS(float, f);
DEFINE_UPDATE_FIR_FILTERS(double, );


#define DEFINE_FIND_TRANSFER_FUNCTION(DTYPE, S_OR_D, F_OR_BLANK) \
static void find_transfer_functions_ ## DTYPE(GSTLALTransferFunction *element, DTYPE *src, guint64 src_size) { \
 \
	/* Convert src_size from bytes to samples */ \
	g_assert(!(src_size % element->unit_size)); \
	src_size /= element->unit_size; \
 \
	gint64 i, j, k, m, num_ffts, k_start, k_stop, first_index, first_index2, stride, num_tfs; \
	stride = element->fft_length - element->fft_overlap; \
	num_tfs = element->channels - 1; \
 \
	/* Determine how many fft's we will calculate from combined leftover and new input data */ \
	num_ffts = minimum64((element->workspace.w ## S_OR_D ## pf.num_leftover + stride - 1) / stride, element->num_ffts); \
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
				element->workspace.w ## S_OR_D ## pf.fft[k] = element->workspace.w ## S_OR_D ## pf.fft_window[k] * element->workspace.w ## S_OR_D ## pf.leftover_data[first_index + k * element->channels]; \
 \
			/* Now copy the inputs from new input data */ \
			k_start = k_stop; \
			k_stop = element->fft_length; \
			for(k = k_start; k < k_stop; k++) \
				element->workspace.w ## S_OR_D ## pf.fft[k] = element->workspace.w ## S_OR_D ## pf.fft_window[k] * src[j + element->channels * (k - k_start)]; \
 \
			/* Take an FFT */ \
			fftw ## F_OR_BLANK ## _execute(element->workspace.w ## S_OR_D ## pf.plan); \
 \
			/* Copy FFT to the proper location */ \
			first_index = j * element->fft_length; \
			for(k = 0; k < element->fft_length; k++) \
				element->workspace.w ## S_OR_D ## pf.ffts[first_index + k] = element->workspace.w ## S_OR_D ## pf.fft[k]; \
		} \
 \
		/* 
		 * Add into the autocorrelation matrix to be averaged. The autocorrelation
		 * matrix includes all transfer functions. Note that the data is stored in
		 * "frequency-major" order: transfer functions at a particular frequency are
		 * stored contiguously in memory before incrementing to the next frequency.
		 */ \
		for(j = 0; j < element->fft_length; j++) { \
			first_index = j * element->channels * num_tfs - 1; \
			for(k = 1; k <= num_tfs; k++) { \
				/* First, divide FFT's of first channel by others to get those transfer functions */ \
				element->workspace.w ## S_OR_D ## pf.autocorrelation_matrix[first_index + k] += element->workspace.w ## S_OR_D ## pf.ffts[j] / element->workspace.w ## S_OR_D ## pf.ffts[j + k * element->fft_length]; \
 \
				/* Now set elements of the autocorrelation matrix along the diagonal equal to one */ \
				element->workspace.w ## S_OR_D ## pf.autocorrelation_matrix[first_index + k * element->channels] += 1.0; \
 \
				/* Now find all other elements of the autocorrelation matrix */ \
				first_index2 = first_index + k * element->channels; \
				for(m = 1; m <= num_tfs - k; m++) { \
					element->workspace.w ## S_OR_D ## pf.autocorrelation_matrix[first_index2 + m] += element->workspace.w ## S_OR_D ## pf.ffts[j + (k + m) * element->fft_length] / element->workspace.w ## S_OR_D ## pf.ffts[j + k * element->fft_length]; \
					element->workspace.w ## S_OR_D ## pf.autocorrelation_matrix[first_index2 + m * num_tfs] += 1.0 / element->workspace.w ## S_OR_D ## pf.autocorrelation_matrix[first_index2 + m]; \
				} \
			} \
		} \
	} \
 \
	element->workspace.w ## S_OR_D ## pf.num_ffts_in_avg += num_ffts; \
 \
	/* Determine how many fft's we will calculate from only new input samples */ \
	num_ffts = (element->sample_count - element->update_samples - element->fft_overlap) / stride - element->workspace.w ## S_OR_D ## pf.num_ffts_in_avg; /* how many more we could compute */ \
	num_ffts = minimum64(num_ffts, element->num_ffts - element->workspace.w ## S_OR_D ## pf.num_ffts_in_avg); /* how many more we need to update the transfer functions */ \
	if(num_ffts < 0) \
		num_ffts = 0; \
 \
	/* Find the location of the first sample in src that will be used */ \
	DTYPE *ptr; \
	if(element->update_samples >= element->sample_count - (gint64) src_size) \
		ptr = src + (element->update_samples - element->sample_count + (gint64) src_size) * element->channels; \
	else \
		ptr = src + (stride - (element->sample_count - (gint64) src_size - element->update_samples) % stride) % stride; \
 \
	/* Loop through the input data and compute transfer functions */ \
	for(i = 0; i < num_ffts; i++) { \
		for(j = 0; j < element->channels; j++) { \
			/* Copy inputs to take an FFT */ \
			k_stop = element->fft_length; \
			first_index = i * stride * element->channels + j; \
			for(k = 0; k < k_stop; k++) \
				element->workspace.w ## S_OR_D ## pf.fft[k] = element->workspace.w ## S_OR_D ## pf.fft_window[k] * ptr[first_index + k * element->channels]; \
 \
			/* Take an FFT */ \
			fftw ## F_OR_BLANK ## _execute(element->workspace.w ## S_OR_D ## pf.plan); \
 \
			/* Copy FFT to the proper location */ \
			first_index = j * element->fft_length; \
			for(k = 0; k < element->fft_length; k++) \
				element->workspace.w ## S_OR_D ## pf.ffts[first_index + k] = element->workspace.w ## S_OR_D ## pf.fft[k]; \
		} \
 \
		/* 
		 * Add into the autocorrelation matrix to be averaged. The autocorrelation
		 * matrix includes all transfer functions. Note that the data is stored in
		 * "frequency-major" order: transfer functions at a particular frequency are
		 * stored contiguously in memory before incrementing to the next frequency.
		 */ \
		for(j = 0; j < element->fft_length; j++) { \
			first_index = j * element->channels * num_tfs - 1; \
			for(k = 1; k <= num_tfs; k++) { \
				/* First, divide FFT's of first channel by others to get those transfer functions */ \
				element->workspace.w ## S_OR_D ## pf.autocorrelation_matrix[first_index + k] += element->workspace.w ## S_OR_D ## pf.ffts[j] / element->workspace.w ## S_OR_D ## pf.ffts[j + k * element->fft_length]; \
 \
				/* Now set elements of the autocorrelation matrix along the diagonal equal to one */ \
				element->workspace.w ## S_OR_D ## pf.autocorrelation_matrix[first_index + k * element->channels] += 1.0; \
 \
				/* Now find all other elements of the autocorrelation matrix */ \
				first_index2 = first_index + k * element->channels; \
				for(m = 1; m <= num_tfs - k; m++) { \
					element->workspace.w ## S_OR_D ## pf.autocorrelation_matrix[first_index2 + m] += element->workspace.w ## S_OR_D ## pf.ffts[j + (k + m) * element->fft_length] / element->workspace.w ## S_OR_D ## pf.ffts[j + k * element->fft_length]; \
					element->workspace.w ## S_OR_D ## pf.autocorrelation_matrix[first_index2 + m * num_tfs] += 1.0 / element->workspace.w ## S_OR_D ## pf.autocorrelation_matrix[first_index2 + m]; \
				} \
			} \
		} \
	} \
 \
	element->workspace.w ## S_OR_D ## pf.num_ffts_in_avg += num_ffts; \
	g_assert_cmpint(element->workspace.w ## S_OR_D ## pf.num_ffts_in_avg, <=, element->num_ffts); \
 \
	/* Now store samples for the next buffer. First, find the sample count of the start of the next fft */ \
	gint64 sample_count_next_fft; \
	if(element->workspace.w ## S_OR_D ## pf.num_ffts_in_avg == element->num_ffts) \
		sample_count_next_fft = 2 * element->update_samples + element->num_ffts * stride + element->fft_overlap + 1; /* If we finished updating the transfer functions */ \
	else \
		sample_count_next_fft = element->update_samples + 1 + element->workspace.w ## S_OR_D ## pf.num_ffts_in_avg * stride; \
 \
	/* Deal with any leftover samples that will remain leftover */ \
	first_index = sample_count_next_fft - (element->sample_count - (gint64) src_size - element->workspace.w ## S_OR_D ## pf.num_leftover); \
	k_stop = (element->sample_count - (gint64) src_size + 1 - sample_count_next_fft) * element->channels; \
	for(k = 0; k < k_stop; k++) \
		element->workspace.w ## S_OR_D ## pf.leftover_data[k] = element->workspace.w ## S_OR_D ## pf.leftover_data[first_index + k]; \
 \
	/* Deal with new samples that will be leftover */ \
	first_index = sample_count_next_fft - (element->sample_count - (gint64) src_size + 1); \
	k_start = maximum64(k_stop, 0); \
	k_stop = (element->sample_count + 1 - sample_count_next_fft) * element->channels; \
	for(k = k_start; k < k_stop; k++) \
		element->workspace.w ## S_OR_D ## pf.leftover_data[k] = src[first_index + k]; \
 \
	/* k_stop is the total number of leftover samples */ \
	element->workspace.w ## S_OR_D ## pf.num_leftover = maximum64(0, k_stop); \
 \
	/* Finally, update transfer functions if ready */ \
	if(element->workspace.w ## S_OR_D ## pf.num_ffts_in_avg == element->num_ffts) { \
		update_transfer_functions_ ## DTYPE(element->workspace.w ## S_OR_D ## pf.autocorrelation_matrix, num_tfs, element->fft_length, element->num_ffts, element->workspace.w ## S_OR_D ## pf.transfer_functions_at_f, element->workspace.w ## S_OR_D ## pf.transfer_functions_solved_at_f, element->workspace.w ## S_OR_D ## pf.autocorrelation_matrix_at_f, element->workspace.w ## S_OR_D ## pf.permutation, element->transfer_functions); \
		GST_INFO_OBJECT(element, "Just computed new transfer functions"); \
		g_object_notify_by_pspec(G_OBJECT(element), properties[ARG_TRANSFER_FUNCTIONS]); \
		element->sample_count -= element->update_samples + element->num_ffts * stride + element->fft_overlap; \
		element->workspace.w ## S_OR_D ## pf.num_ffts_in_avg = 0; \
 \
		/* Update FIR filters if we want */ \
		if(element->make_fir_filters) { \
			update_fir_filters_ ## DTYPE(element->transfer_functions, num_tfs, element->fft_length, element->rate, element->workspace.w ## S_OR_D ## pf.fir_filter, element->workspace.w ## S_OR_D ## pf.fir_plan, element->workspace.w ## S_OR_D ## pf.fir_window, element->workspace.w ## S_OR_D ## pf.tukey, element->fir_filters); \
			GST_INFO_OBJECT(element, "Just computed new FIR filters"); \
			g_object_notify_by_pspec(G_OBJECT(element), properties[ARG_FIR_FILTERS]); \
		} \
 \
		/* Write output to the screen or a file if we want */ \
		if(element->write_to_screen || element->filename) { \
			write_transfer_functions(element->transfer_functions, gst_element_get_name(element), element->fft_length, num_tfs, element->write_to_screen, element->filename); \
			if(element->make_fir_filters) \
				write_fir_filters(element->fir_filters, gst_element_get_name(element), 2 * (element->fft_length - 1), num_tfs, element->write_to_screen, element->filename); \
		} \
	} \
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
	return success;
}


/*
 * start()
 */


static gboolean start(GstBaseSink *sink) {

	GSTLALTransferFunction *element = GSTLAL_TRANSFERFUNCTION(sink);

	/* Sanity checks */
	if(element->num_ffts > 1 && element->fft_overlap >= element->fft_length) {
		GST_ERROR_OBJECT(element, "fft_overlap must not be greater than fft_length! Reset fft_length and/or fft_overlap properties.");
		g_assert_not_reached();
	}
	if((!element->make_fir_filters) && (element->high_pass != 9 || element->low_pass != 0))
		GST_WARNING_OBJECT(element, "A FIR filter cutoff frequency is set, but no FIR filter is being produced. Set the property make_fir_filters = True to make FIR filters.");
	if(element->high_pass != 0 && element->low_pass != 0 && element->high_pass > element->low_pass)
		GST_WARNING_OBJECT(element, "The high-pass cutoff frequency of the FIR filters is above the low-pass cutoff frequency. Reset high_pass and/or low_pass to change this.");
	if(element->update_samples + element->fft_length < element->rate)
		GST_WARNING_OBJECT(element, "The chosen fft_length and update_samples are very short. Errors may result.");

	/* Timestamp bookkeeping */
	element->t0 = GST_CLOCK_TIME_NONE;
	element->offset0 = GST_BUFFER_OFFSET_NONE;
	element->next_in_offset = GST_BUFFER_OFFSET_NONE;

	/* Memory allocation */
	if(!element->transfer_functions)
		element->transfer_functions = g_malloc((element->channels - 1) * element->fft_length * sizeof(*element->transfer_functions));
	if(element->make_fir_filters && !element->fir_filters)
		element->fir_filters = g_malloc((element->channels - 1) * 2 * (element->fft_length - 1) * sizeof(*element->fir_filters));

	element->sample_count = element->update_samples;

	/* Prepare workspace for finding transfer functions and FIR filters */
	if(element->data_type == GSTLAL_TRANSFERFUNCTION_F32) {
		/* window functions */
		element->workspace.wspf.fft_window = g_malloc(element->fft_length * sizeof(*element->workspace.wspf.fft_window));
		gint64 i, i_stop, i_start;
		for(i = 0; i < element->fft_length; i++)
			element->workspace.wspf.fft_window[i] = (float) pow(sin(M_PI * i / (element->fft_length - 1)), 2.0);
		if(element->make_fir_filters) {

			/*
			 * Make a frequency-donain window to roll off low and high frequencies
			 */

			element->workspace.wspf.fir_window = g_malloc(element->fft_length * sizeof(*element->workspace.wspf.fir_window));

			/* Initialize to ones */
			for(i = 0; i < element->fft_length; i++)
				element->workspace.wspf.fir_window[i] = 1.0;

			int f_nyquist = element->rate / 2;
			float df_per_hz = (element->fft_length - 1.0) / f_nyquist;

			/* high-pass filter */
			/* Remove low frequencies */
			i_stop = (gint64) (element->high_pass * df_per_hz / 2.0 + 0.49);
			for(i = 0; i < i_stop; i++)
				element->workspace.wspf.fir_window[i] = 0.0;

			/* Apply half of a Hann window raised to the fourth power */
			i_start = i_stop;
			i_stop = (gint64) (element->high_pass * df_per_hz + 0.49);
			for(i = i_start; i < i_stop; i++)
				element->workspace.wspf.fir_window[i] *= (float) pow(sin((M_PI / 2.0) * (i - i_start) / (i_stop - i_start)), 8.0);

			/* low-pass filter */
			if(element->low_pass > 0) {
				/* Apply half of a Hann window */
				i_start = (gint64) (element->low_pass * df_per_hz + 0.49);
				i_stop = minimum64(element->fft_length, 1.4 * i_start);
				for(i = i_start; i < i_stop; i++)
					element->workspace.wspf.fir_window[i] *= (float) pow(cos((M_PI / 2.0) * (i - i_start) / (i_stop - i_start)), 2.0);

				/* Remove high frequencies */
				i_start = i_stop;
				i_stop = element->fft_length;
				for(i = i_start; i < i_stop; i++)
					element->workspace.wspf.fir_window[i] = 0.0;
			}

			/*
			 * Make a time-domain Tukey window so that the filter falls off smoothly at the edges
			 */

			gint64 fir_length, edge_to_corner;
			fir_length = 2 * (element->fft_length - 1);
			edge_to_corner = (gint64) (0.45 * fir_length);

			/* first curve of window */
			for(i = 0; i < edge_to_corner; i++)
				element->workspace.wspf.tukey[i] = pow(sin((M_PI / 2.0) * i / edge_to_corner), 2.0);

			/* flat top of window */
			i_stop = fir_length - edge_to_corner;
			for(i = edge_to_corner; i < i_stop; i++)
				element->workspace.wspf.tukey[i] = 1.0;

			/* last curve of window */
			i_start = i_stop;
			for(i = i_start; i < fir_length; i++)
				element->workspace.wspf.tukey[i] = pow(cos((M_PI / 2.0) * (i + 1 - i_start) / (fir_length - i_start)), 2.0);
		}

		/* intermediate data storage */
		element->workspace.wspf.leftover_data = g_malloc(element->channels * (element->fft_length - 1) * sizeof(*element->workspace.wspf.leftover_data));
		element->workspace.wspf.num_leftover = 0;
		element->workspace.wspf.ffts = g_malloc(element->channels * element->fft_length * sizeof(*element->workspace.wspf.ffts));
		element->workspace.wspf.num_ffts_in_avg = 0;
		element->workspace.wspf.autocorrelation_matrix = g_malloc(element->channels * (element->channels - 1) * element->fft_length * sizeof(*element->workspace.wspf.autocorrelation_matrix));

		/* Allocate memory for gsl matrix manipulations. The same memory locations will be used repeatedly */
		element->workspace.wspf.transfer_functions_at_f = gsl_vector_complex_alloc(element->channels - 1);
		element->workspace.wspf.transfer_functions_solved_at_f = gsl_vector_complex_alloc(element->channels - 1);
		element->workspace.wspf.autocorrelation_matrix_at_f = gsl_matrix_complex_alloc(element->channels - 1, element->channels - 1);
		element->workspace.wspf.permutation = gsl_permutation_alloc(element->channels - 1);

		/* Allocate memory for fftwf to do Fourier transforms of data. The same memory locations will be used repeatedly */
		gstlal_fftw_lock();

		GST_LOG_OBJECT(element, "starting FFTWF planning");

		/* data that will be Fourier transformed into frequency domain */
		element->workspace.wspf.fft = (complex float *) fftwf_malloc(element->fft_length * sizeof(*element->workspace.wspf.fft));
		element->workspace.wspf.plan = fftwf_plan_dft_r2c_1d(element->fft_length, (float *) element->workspace.wspf.fft, element->workspace.wspf.fft, FFTW_MEASURE);

		if(element->make_fir_filters && !element->fir_filters) {

			/* data that will be inverse Fourier transformed back into the time domain */
			element->workspace.wspf.fir_filter = (complex float *) fftwf_malloc(2 * (element->fft_length - 1) * sizeof(*element->workspace.wspf.fir_filter));
			element->workspace.wspf.fir_plan = fftwf_plan_dft_c2r_1d(2 * (element->fft_length - 1), element->workspace.wspf.fir_filter, (float *) element->workspace.wspf.fir_filter, FFTW_MEASURE);
		}
		GST_LOG_OBJECT(element, "FFTWF planning complete");

		gstlal_fftw_unlock();

	} else if(element->data_type == GSTLAL_TRANSFERFUNCTION_F64) {
		/* window functions */
		element->workspace.wdpf.fft_window = g_malloc(element->fft_length * sizeof(*element->workspace.wdpf.fft_window));
		gint64 i, i_stop, i_start;
		for(i = 0; i < element->fft_length; i++)
			element->workspace.wdpf.fft_window[i] = pow(sin(M_PI * i / (element->fft_length - 1)), 2.0);
		if(element->make_fir_filters) {

			/*
			 * Make a frequency-donain window to roll off low and high frequencies
			 */

			element->workspace.wdpf.fir_window = g_malloc(element->fft_length * sizeof(*element->workspace.wdpf.fir_window));

			/* Initialize to ones */
			for(i = 0; i < element->fft_length; i++)
				element->workspace.wdpf.fir_window[i] = 1.0;

			int f_nyquist = element->rate / 2;
			double df_per_hz = (element->fft_length - 1.0) / f_nyquist;

			/* high-pass filter */
			/* Remove low frequencies */
			i_stop = (gint64) (element->high_pass * df_per_hz / 2.0 + 0.49);
			for(i = 0; i < i_stop; i++)
				element->workspace.wdpf.fir_window[i] = 0.0;

			/* Apply half of a Hann window raised to the fourth power */
			i_start = i_stop;
			i_stop = (gint64) (element->high_pass * df_per_hz + 0.49);
			for(i = i_start; i < i_stop; i++)
				element->workspace.wdpf.fir_window[i] *= pow(sin((M_PI / 2.0) * (i - i_start) / (i_stop - i_start)), 8.0);

			/* low-pass filter */
			if(element->low_pass > 0) {
				/* Apply half of a Hann window */
				i_start = (gint64) (element->low_pass * df_per_hz + 0.49);
				i_stop = minimum64(element->fft_length, 1.4 * i_start);
				for(i = i_start; i < i_stop; i++)
					element->workspace.wdpf.fir_window[i] *= pow(cos((M_PI / 2.0) * (i - i_start) / (i_stop - i_start)), 2.0);

				/* Remove high frequencies */
				i_start = i_stop;
				i_stop = element->fft_length;
				for(i = i_start; i < i_stop; i++)
					element->workspace.wdpf.fir_window[i] = 0.0;
			}

			/*
			 * Make a time-domain Tukey window so that the filter falls off smoothly at the edges
			 */

			gint64 fir_length, edge_to_corner;
			fir_length = 2 * (element->fft_length - 1);
			edge_to_corner = (gint64) (0.45 * fir_length);

			/* first curve of window */
			for(i = 0; i < edge_to_corner; i++)
				element->workspace.wdpf.tukey[i] = pow(sin((M_PI / 2.0) * i / edge_to_corner), 2.0);

			/* flat top of window */
			i_stop = fir_length - edge_to_corner;
			for(i = edge_to_corner; i < i_stop; i++)
				element->workspace.wdpf.tukey[i] = 1.0;

			/* last curve of window */
			i_start = i_stop;
			for(i = i_start; i < fir_length; i++)
				element->workspace.wdpf.tukey[i] = pow(cos((M_PI / 2.0) * (i + 1 - i_start) / (fir_length - i_start)), 2.0);
		}

		/* intermediate data storage */
		element->workspace.wdpf.leftover_data = g_malloc(element->channels * (element->fft_length - 1) * sizeof(*element->workspace.wdpf.leftover_data));
		element->workspace.wdpf.num_leftover = 0;
		element->workspace.wdpf.ffts = g_malloc(element->channels * element->fft_length * sizeof(*element->workspace.wdpf.ffts));
		element->workspace.wdpf.num_ffts_in_avg = 0;
		element->workspace.wdpf.autocorrelation_matrix = g_malloc(element->channels * (element->channels - 1) * element->fft_length * sizeof(*element->workspace.wdpf.autocorrelation_matrix));

		/* Allocate memory for gsl matrix manipulations. The same memory locations will be used repeatedly */
		element->workspace.wdpf.transfer_functions_at_f = gsl_vector_complex_alloc(element->channels - 1);
		element->workspace.wdpf.transfer_functions_solved_at_f = gsl_vector_complex_alloc(element->channels - 1);
		element->workspace.wdpf.autocorrelation_matrix_at_f = gsl_matrix_complex_alloc(element->channels - 1, element->channels - 1);
		element->workspace.wdpf.permutation = gsl_permutation_alloc(element->channels - 1);

		/* Allocate memory for fftw to do Fourier transforms of data. The same memory locations will be used repeatedly */
		gstlal_fftw_lock();

		GST_LOG_OBJECT(element, "starting FFTW planning");

		/* data that will be Fourier transformed into frequency domain */
		element->workspace.wdpf.fft = (complex double *) fftw_malloc(element->fft_length * sizeof(*element->workspace.wdpf.fft));
		element->workspace.wdpf.plan = fftw_plan_dft_r2c_1d(element->fft_length, (double *) element->workspace.wdpf.fft, element->workspace.wdpf.fft, FFTW_MEASURE);

		if(element->make_fir_filters && !element->fir_filters) {

			/* data that will be inverse Fourier transformed back into the time domain */
			element->workspace.wdpf.fir_filter = (complex double *) fftw_malloc(2 * (element->fft_length  - 1) * sizeof(*element->workspace.wdpf.fir_filter));
			element->workspace.wdpf.fir_plan = fftw_plan_dft_c2r_1d(2 * (element->fft_length - 1), element->workspace.wdpf.fir_filter, (double *) element->workspace.wdpf.fir_filter, FFTW_MEASURE);
		}
		GST_LOG_OBJECT(element, "FFTW planning complete");

		gstlal_fftw_unlock();

	} else
		g_assert_not_reached();

	return TRUE;
}


/*
 * stop()
 */


static gboolean stop(GstBaseSink *sink) {

	GSTLALTransferFunction *element = GSTLAL_TRANSFERFUNCTION(sink);

	g_free(element->transfer_functions);
	element->transfer_functions = NULL;
	if(element->make_fir_filters) {
		g_free(element->fir_filters);
		element->fir_filters = NULL;
	}
	if(element->data_type == GSTLAL_TRANSFERFUNCTION_F32) {

		/* free allocated memory in workspace */
		g_free(element->workspace.wspf.leftover_data);
		element->workspace.wspf.leftover_data = NULL;
		g_free(element->workspace.wspf.ffts);
		element->workspace.wspf.ffts = NULL;
		g_free(element->workspace.wspf.autocorrelation_matrix);
		element->workspace.wspf.autocorrelation_matrix = NULL;

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
		g_free(element->workspace.wdpf.leftover_data);
		element->workspace.wdpf.leftover_data = NULL;
		g_free(element->workspace.wdpf.ffts);
		element->workspace.wdpf.ffts = NULL;
		g_free(element->workspace.wdpf.autocorrelation_matrix);
		element->workspace.wdpf.autocorrelation_matrix = NULL;

		/* free gsl stuff in workspace */
		gsl_vector_complex_free(element->workspace.wdpf.transfer_functions_at_f);
		element->workspace.wdpf.transfer_functions_at_f = NULL;
		gsl_vector_complex_free(element->workspace.wdpf.transfer_functions_solved_at_f);
		element->workspace.wdpf.transfer_functions_solved_at_f = NULL;
		gsl_matrix_complex_free(element->workspace.wdpf.autocorrelation_matrix_at_f);
		element->workspace.wdpf.autocorrelation_matrix_at_f = NULL;
		gsl_permutation_free(element->workspace.wdpf.permutation);
		element->workspace.wdpf.permutation = NULL;

		/* free fftw stuff in workspace*/
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
	return TRUE;
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
		if(element->sample_count > element->update_samples) 
			element->sample_count = element->update_samples;
		if(element->data_type == GSTLAL_TRANSFERFUNCTION_F32) {
			element->workspace.wspf.num_ffts_in_avg = 0;
			element->workspace.wspf.num_leftover = 0;
		} else {
			element->workspace.wdpf.num_ffts_in_avg = 0;
			element->workspace.wdpf.num_leftover = 0;
		}
	}
	element->next_in_offset = GST_BUFFER_OFFSET_END(buffer);
	GST_DEBUG_OBJECT(element, "have buffer spanning %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(buffer));

	/* Get the data from the buffer */
	gst_buffer_map(buffer, &mapinfo, GST_MAP_READ);

	/* Increment the sample count */
	element->sample_count += (mapinfo.size / element->unit_size);

	/* Deal with gaps */
	if(GST_BUFFER_FLAG_IS_SET(buffer, GST_BUFFER_FLAG_GAP) && mapinfo.size != 0) {
		if(element->update_after_gap || element->sample_count > element->update_samples)
			element->sample_count = element->update_samples;
		if(element->data_type == GSTLAL_TRANSFERFUNCTION_F32) {
			element->workspace.wspf.num_ffts_in_avg = 0;
			element->workspace.wspf.num_leftover = 0;
		} else {
			element->workspace.wdpf.num_ffts_in_avg = 0;
			element->workspace.wdpf.num_leftover = 0;
		}
	}

	/* Check whether we need to do anything with this data */
	if(element->sample_count > element->update_samples) {
		if(element->data_type == GSTLAL_TRANSFERFUNCTION_F32) {
			/* If we are just beginning to compute new transfer functions with this data, initialize memory that we will fill to zero */
			if(!element->workspace.wspf.num_ffts_in_avg)
				memset(element->workspace.wspf.autocorrelation_matrix, 0, element->channels * (element->channels - 1) * element->fft_length * sizeof(*element->workspace.wspf.autocorrelation_matrix));

			/* Send the data to a function to compute fft's and transfer functions */
			find_transfer_functions_float(element, (float *) mapinfo.data, mapinfo.size);
		} else {
			/* If we are just beginning to compute new transfer functions with this data, initialize memory that we will fill to zero */
			if(!element->workspace.wdpf.num_ffts_in_avg)
				memset(element->workspace.wdpf.autocorrelation_matrix, 0, element->channels * (element->channels - 1) * element->fft_length * sizeof(*element->workspace.wdpf.autocorrelation_matrix));

			/* Send the data to a function to compute fft's and transfer functions */
			find_transfer_functions_double(element, (double *) mapinfo.data, mapinfo.size);
		}
	}

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

	case ARG_UPDATE_SAMPLES:
		element->update_samples = g_value_get_int64(value);
		break;

	case ARG_UPDATE_AFTER_GAP:
		element->update_after_gap = g_value_get_boolean(value);
		break;

	case ARG_WRITE_TO_SCREEN:
		element->write_to_screen = g_value_get_boolean(value);
		break;

	case ARG_FILENAME:
		element->filename = g_value_dup_string(value);
		break;

	case ARG_MAKE_FIR_FILTERS:
		element->make_fir_filters = g_value_get_boolean(value);
		break;

	case ARG_HIGH_PASS:
		element->high_pass = g_value_get_int(value);
		break;

	case ARG_LOW_PASS:
		element->low_pass = g_value_get_int(value);
		break;

	case ARG_TRANSFER_FUNCTIONS:
		if(element->transfer_functions)
			g_free(element->transfer_functions);
		int n;
		gstlal_doubles_from_g_value_array(g_value_get_boxed(value), (double *) element->transfer_functions, &n);
		break;

	case ARG_FIR_FILTERS:
		if(element->fir_filters)
			g_free(element->fir_filters);
		int m;
		gstlal_doubles_from_g_value_array(g_value_get_boxed(value), element->fir_filters, &m);
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

	case ARG_UPDATE_SAMPLES:
		g_value_set_int64(value, element->update_samples);
		break;

	case ARG_UPDATE_AFTER_GAP:
		g_value_set_boolean(value, element->update_after_gap);
		break;

	case ARG_WRITE_TO_SCREEN:
		g_value_set_boolean(value, element->write_to_screen);
		break;

	case ARG_FILENAME:
		g_value_set_string(value, element->filename);
		break;

	case ARG_MAKE_FIR_FILTERS:
		g_value_set_boolean(value, element->make_fir_filters);
		break;

	case ARG_HIGH_PASS:
		g_value_set_int(value, element->high_pass);
		break;

	case ARG_LOW_PASS:
		g_value_set_int(value, element->low_pass);
		break;

	case ARG_TRANSFER_FUNCTIONS:
		if(element->transfer_functions) {
			GValueArray *va;
			va = g_value_array_new(element->channels - 1);
			GValue v = G_VALUE_INIT;
			g_value_init(&v, G_TYPE_VALUE_ARRAY);
			int i;
			for(i = 0; i < element->channels - 1; i++) {
				g_value_take_boxed(&v, gstlal_g_value_array_from_doubles((double *) element->transfer_functions, 2 * element->fft_length));
				g_value_array_append(va, &v);
			}
			g_value_take_boxed(value, va);
		}
		break;

	case ARG_FIR_FILTERS:
		if(element->fir_filters) {
			GValueArray *val_array;
			val_array = g_value_array_new(element->channels - 1);
			GValue val = G_VALUE_INIT;
			g_value_init(&val, G_TYPE_VALUE_ARRAY);
			int j;
			for(j = 0; j < element->channels - 1; j++) {
				g_value_take_boxed(&val, gstlal_g_value_array_from_doubles(element->fir_filters, 2 * (element->fft_length - 1)));
				g_value_array_append(val_array, &val);
			}
			g_value_take_boxed(value, val_array);
		}
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

	if(element->transfer_functions) {
		g_free(element->transfer_functions);
		element->transfer_functions = NULL;
	}
	if(element->fir_filters) {
		g_free(element->fir_filters);
		element->fir_filters = NULL;
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

	gstbasesink_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
	gstbasesink_class->start = GST_DEBUG_FUNCPTR(start);
	gstbasesink_class->render = GST_DEBUG_FUNCPTR(render);
	gstbasesink_class->stop = GST_DEBUG_FUNCPTR(stop);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	gst_element_class_set_details_simple(
		element_class,
		"TransferFunction",
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
		0, G_MAXINT64, 16384,
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
	properties[ARG_WRITE_TO_SCREEN] = g_param_spec_boolean(
		"write-to-screen",
		"Write to Screen",
		"Set to True in order to write transfer functions and/or FIR filters to the screen.",
		FALSE,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_FILENAME] = g_param_spec_string(
		"filename",
		"Filename",
		"Name of file to write transfer functions and/or FIR filters to. If not given,\n\t\t\t"
		"no file is produced.",
		NULL,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_MAKE_FIR_FILTERS] = g_param_spec_boolean(
		"make-fir-filters",
		"Make FIR Filters",
		"If True, FIR filters will be produced each time the transfer functions are computed.",
		FALSE,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_HIGH_PASS] = g_param_spec_int(
		"high-pass",
		"High Pass",
		"The high-pass cutoff frequency (in Hz) of the FIR filters.\n\t\t\t"
		"If zero, no high-pass cutoff is added.",
		0, G_MAXINT, 9,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_LOW_PASS] = g_param_spec_int(
		"low-pass",
		"Low Pass",
		"The low-pass cutoff frequency (in Hz) of the FIR filters.\n\t\t\t"
		"If zero, no low-pass cutoff is added.",
		0, G_MAXINT, 0,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_TRANSFER_FUNCTIONS] = g_param_spec_value_array(
		"transfer-functions",
		"Transfer Functions",
		"Array of the computed transfer functions",
		g_param_spec_value_array(
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
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_CONTROLLABLE
	);
	properties[ARG_FIR_FILTERS] = g_param_spec_value_array(
		"fir-filters",
		"FIR Filters",
		"Array of the computed FIR filters",
		g_param_spec_value_array(
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
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_CONTROLLABLE
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
		ARG_TRANSFER_FUNCTIONS,
		properties[ARG_TRANSFER_FUNCTIONS]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_FIR_FILTERS,
		properties[ARG_FIR_FILTERS]
	);
}


/*
 * init()
 */


static void gstlal_transferfunction_init(GSTLALTransferFunction *element) {

	g_signal_connect(G_OBJECT(element), "notify::transfer-function", G_CALLBACK(rebuild_workspace_and_reset), NULL);
	g_signal_connect(G_OBJECT(element), "notify::fir-filter", G_CALLBACK(rebuild_workspace_and_reset), NULL);
	element->rate = 0;
	element->unit_size = 0;
	element->channels = 0;
}


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
 * SECTION:gstlal_adaptivefirfilt
 * @short_description:  Compute an adaptive FIR filter given interleaved
 * complex channels as input.
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
 * our own stuff
 */


#include <gstlal/gstlal.h>
#include <gstlal/gstlal_audio_info.h>
#include <gstlal/gstlal_debug.h>
#include <gstlal_firtools.h>
#include <gstlal_adaptivefirfilt.h>


/*
 * ============================================================================
 *
 *			    Custom Types
 *
 * ============================================================================
 */


/*
 * window type enum
 */


GType gstlal_adaptivefirfilt_window_get_type(void) {

	static GType type = 0;

	if(!type) {
		static GEnumValue values[] = {
			{GSTLAL_ADAPTIVEFIRFILT_DPSS, "GSTLAL_ADAPTIVEFIRFILT_DPSS", "Maximize energy concentration in main lobe"},
			{GSTLAL_ADAPTIVEFIRFILT_KAISER, "GSTLAL_ADAPTIVEFIRFILT_KAISER", "Simple approximtion to DPSS window"},
			{GSTLAL_ADAPTIVEFIRFILT_DOLPH_CHEBYSHEV, "GSTLAL_ADAPTIVEFIRFILT_DOLPH_CHEBYSHEV", "Attenuate all side lobes equally"},
			{GSTLAL_ADAPTIVEFIRFILT_BLACKMAN, "GSTLAL_ADAPTIVEFIRFILT_BLACKMAN", "Strongly attenuate distant side lobes"},
			{GSTLAL_ADAPTIVEFIRFILT_HANN, "GSTLAL_ADAPTIVEFIRFILT_HANN", "Cosine squared window"},
			{GSTLAL_ADAPTIVEFIRFILT_NONE, "GSTLAL_ADAPTIVEFIRFILT_NONE", "Do not apply a window function"},
			{0, NULL, NULL}
		};

		type = g_enum_register_static("GSTLAL_ADAPTIVEFIRFILT_WINDOW", values);
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


#define GST_CAT_DEFAULT gstlal_adaptivefirfilt_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


G_DEFINE_TYPE_WITH_CODE(
	GSTLALAdaptiveFIRFilt,
	gstlal_adaptivefirfilt,
	GST_TYPE_BASE_SINK,
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "lal_adaptivefirfilt", 0, "lal_adaptivefirfilt element")
);


enum property {
	ARG_UPDATE_SAMPLES = 1,
	ARG_AVERAGE_SAMPLES,
	ARG_NUM_ZEROS,
	ARG_NUM_POLES,
	ARG_STATIC_ZEROS,
	ARG_STATIC_POLES,
	ARG_PHASE_MEASUREMENT_FREQUENCY,
	ARG_STATIC_FILTER,
	ARG_STATIC_MODEL,
	ARG_MINIMIZE_FILTER_LENGTH,
	ARG_ADAPTIVE_FILTER,
	ARG_ADAPTIVE_FILTER_LENGTH,
	ARG_FREQUENCY_RESOLUTION,
	ARG_FILTER_SAMPLE_RATE,
	ARG_FILTER_TIMESHIFT,
	ARG_FILTER_ENDTIME,
	ARG_WRITE_TO_SCREEN,
	ARG_FILENAME,
	ARG_WINDOW_TYPE,
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


static void rebuild_workspace_and_reset(GObject *object) {
	return;
}


double min(double value1, double value2) {
	return value1 < value2 ? value1 : value2;
}


double max(double value1, double value2) {
	return value1 > value2 ? value1 : value2; \
}


#define DEFINE_MINIMUM(size) \
gint ## size min ## size(gint ## size value1, gint ## size value2) { \
	return value1 < value2 ? value1 : value2; \
}


DEFINE_MINIMUM(8);
DEFINE_MINIMUM(16);
DEFINE_MINIMUM(32);
DEFINE_MINIMUM(64);


#define DEFINE_MAXIMUM(size) \
gint ## size max ## size(gint ## size value1, gint ## size value2) { \
	return value1 > value2 ? value1 : value2; \
}


DEFINE_MAXIMUM(8);
DEFINE_MAXIMUM(16);
DEFINE_MAXIMUM(32);
DEFINE_MAXIMUM(64);


static void write_filter(double *filter, char *element_name, char *filter_type, gint64 rows, gboolean write_to_screen, char *filename, gboolean free_name) {
	gint64 i;
	if(write_to_screen) {
		g_print("================== %s FIR filter computed by %s ==================\n", filter_type, element_name);

		for(i = 0; i < rows; i++)
			g_print("%1.15e\n", filter[i]);
		g_print("\n\n");
	}

	if(filename) {
		FILE *fp;
		fp = fopen(filename, "a");
		g_fprintf(fp, "================== %s FIR filter computed by %s ==================\n", filter_type, element_name);

		for(i = 0; i < rows; i++)
			g_fprintf(fp, "%1.15e\n", filter[i]);
		g_fprintf(fp, "\n\n");
		fclose(fp);
	}
	if(free_name)
		g_free(element_name);
}


double *convolve(double *filter1, int length1, double *filter2, int length2, double *convolved_filter) {
	int i, j;
	if(!convolved_filter)
		convolved_filter = g_malloc((length1 + length2 - 1) * sizeof(*convolved_filter));
	if(convolved_filter == filter1) {
		/* We are putting the result into the memory location of filter1, so we need to make a copy of filter1 */
		double *filter1_copy = g_malloc(length1 * sizeof(*filter1));
		memcpy(filter1_copy, filter1, length1 * sizeof(*filter1));
		memset(convolved_filter, 0, (length1 + length2 - 1) * sizeof(*convolved_filter));
		for(i = 0; i < length1; i++) {
			for(j = 0; j < length2; j++)
				convolved_filter[i + j] += filter1_copy[i] * filter2[j];
		}
		g_free(filter1_copy);
	} else if(convolved_filter == filter2) {
		/* We are putting the result into the memory location of filter2, so we need to make a copy of filter2 */
		double *filter2_copy = g_malloc(length2 * sizeof(*filter2));
		memcpy(filter2_copy, filter2, length2 * sizeof(*filter2));
		memset(convolved_filter, 0, (length1 + length2 - 1) * sizeof(*convolved_filter));
		for(i = 0; i < length1; i++) {
			for(j = 0; j < length2; j++)
				convolved_filter[i + j] += filter1[i] * filter2_copy[j];
		}
		g_free(filter2_copy);
	} else {
		/* All three memory locations are different */
		memset(convolved_filter, 0, (length1 + length2 - 1) * sizeof(*convolved_filter));
		for(i = 0; i < length1; i++) {
			for(j = 0; j < length2; j++)
				convolved_filter[i + j] += filter1[i] * filter2[j];
		}
	}
	return convolved_filter;
}


static gboolean update_padded_filter(complex double *padded_model, gint64 padded_model_length, double *padded_filter, gint64 padded_filter_length, int filter_sample_rate, complex double *input_average, int num_zeros, int num_poles, gboolean filter_has_gain, complex double *static_zeros, int num_static_zeros, complex double *static_poles, int num_static_poles, double phase_measurement_frequency, complex double *static_model, gint64 static_model_length) {

	/*
	 * Compute the filter in the frequency domain
	 */

	gint64 n;
	int m;
	complex double gain, two_pi_i_df_t, df = (complex double) filter_sample_rate / 2.0 / (padded_model_length - 1.0);

	if(filter_has_gain && phase_measurement_frequency) {
		/*
		 * In this case, we have a phase correction to include in the filter. At each point
		 * in frequency space, the filter is gain * exp(2 pi i f t). The value of f (= n * df)
		 * ranges from DC (0) to the Nyquist frequency (fir_sample_rate / 2) in evenly spaced
		 * increments. The rest of the parameters in the exp() are constant.
		 */

		gain = (complex double) cabs(input_average[num_zeros + num_poles]) / padded_filter_length;
		two_pi_i_df_t = clog(input_average[num_zeros + num_poles] / gain / padded_filter_length) / phase_measurement_frequency * df;
	} else if(filter_has_gain) {
		gain = input_average[num_zeros + num_poles] / padded_filter_length;
		two_pi_i_df_t = 0.0;
	} else {
		gain = (complex double) 1.0 / padded_filter_length;
		two_pi_i_df_t = 0.0;
	}

	/*
	 * At each point in frequency space, the gain/phase portion of the filter is 
	 * gain * exp(2 pi i f t). The value of f (= n * df) ranges from DC (0) to the
	 * Nyquist frequency (fir_sample_rate / 2) in evenly spaced increments. The
	 * rest of the parameters in the exp() are constant. 
	 */

	for(n = 0; n < padded_model_length; n += 2)
		padded_model[n] = gain * cexp(two_pi_i_df_t * n);

	/* Negating every other value adds a delay of half the length of the filter, centering it in time. */
	for(n = 1; n < padded_model_length; n += 2)
		padded_model[n] = -gain * cexp(two_pi_i_df_t * n);

	/* Now add zeros and poles */
	for(n = 0; n < padded_model_length; n++) {
		/* variable zeros */
		for(m = 0; m < num_zeros; m++)
			padded_model[n] *= 1.0 + I * n * df / input_average[m];
		/* variable poles */
		for(m = num_zeros; m < num_zeros + num_poles; m++)
			padded_model[n] /= 1.0 + I * n * df / input_average[m];
		/* static zeros */
		for(m = 0; m < num_static_zeros; m++)
			padded_model[n] *= 1.0 + I * n * df / static_zeros[m];
		/* static poles */
		for(m = 0; m < num_static_poles; m++)
			padded_model[n] /= 1.0 + I * n * df / static_poles[m];
	}

	if(static_model_length > 0) {
		/* Multiply by the static model */
		for(n = 0; n < padded_model_length; n++) {
			padded_model[n] *= static_model[n];
		}
	}

	/* Make sure the DC component is real */
	padded_model[0] = creal(padded_model[0]);

	if(!(padded_filter_length % 2))
		/* Make sure the Nyquist component is real */
		padded_model[padded_model_length - 1] = creal(padded_model[padded_model_length - 1]);

	/* Take the inverse Fourier transform */
	double *padded_filter_temp = gstlal_irfft_double(padded_model, padded_model_length, (guint *) &padded_filter_length, NULL, 0, NULL, FALSE, 0, NULL, 0, NULL, NULL, 0, NULL, 0, NULL, NULL, NULL, FALSE);

	/* Check if the filter has sain values in it */
	gboolean success = TRUE;
	for(n = 0; n < padded_filter_length; n++) {
		success &= isnormal(padded_filter_temp[n]) || padded_filter_temp[n] == 0.0;
		padded_filter[n] = padded_filter_temp[n];
	}

	g_free(padded_filter_temp);

	return success;
}


static gboolean update_adaptive_filter_minimized(double *adaptive_filter, int adaptive_filter_length, int filter_sample_rate, complex double *input_average, int num_zeros, gboolean filter_has_gain, complex double *static_zeros, int num_static_zeros, double *static_filter, gint64 static_filter_length) {

	int i;

	if(num_zeros) {
		/* There is at least one zero computed from inputs. We compute a two-tap filter based on the first of these */
		*adaptive_filter = 0.5 + filter_sample_rate / (2.0 * M_PI * creal(*input_average));
		adaptive_filter[1] = 0.5 - filter_sample_rate / (2.0 * M_PI * creal(*input_average));
	} else if(num_static_zeros) {
		/* There are no inputs that are zeros, but we wish to compute a two-tap filter from a static zero instead */
		*adaptive_filter = 0.5 + filter_sample_rate / (2.0 * M_PI * creal(*input_average));
		adaptive_filter[1] = 0.5 - filter_sample_rate / (2.0 * M_PI * creal(*input_average));
	} else {
		/* There are no zeros, only a gain */
		*adaptive_filter = 1.0;
	}

	if(num_zeros + num_static_zeros > 1) {
		/* We will convolve the filter with more two-tap filters to complete it */
		double *tempfilt = g_malloc(2 * sizeof(*adaptive_filter));
		for(i = 1; i < num_zeros; i++) {
			*tempfilt = 0.5 + filter_sample_rate / (2.0 * M_PI * creal(input_average[i]));
			tempfilt[1] = 0.5 - filter_sample_rate / (2.0 * M_PI * creal(input_average[i]));
			convolve(adaptive_filter, i + 1, tempfilt, 2, adaptive_filter);
		}
		for(i = num_zeros ? num_zeros : 1; i < num_zeros + num_static_zeros; i++) {
			*tempfilt = 0.5 + filter_sample_rate / (2.0 * M_PI * creal(static_zeros[i - num_zeros]));
			tempfilt[1] = 0.5 - filter_sample_rate / (2.0 * M_PI * creal(static_zeros[i - num_zeros]));
			convolve(adaptive_filter, i + 1, tempfilt, 2, adaptive_filter);
		}
		g_free(tempfilt);
	}

	/* Convolve with a static filter if there is one */
	if(static_filter_length > 0)
		convolve(adaptive_filter, num_zeros + num_static_zeros + 1, static_filter, static_filter_length, adaptive_filter);

	/* Include a gain factor if there is one */
	if(filter_has_gain) {
		for(i = 0; i < adaptive_filter_length; i++)
			adaptive_filter[i] *= creal(input_average[num_zeros]);
	}

	/* Check if the filter has sain values in it */
	gboolean success = TRUE;
	for(i = 0; i < adaptive_filter_length; i++)
		success &= isnormal((adaptive_filter)[i]);

	return success;
}


#define DEFINE_AVERAGE_INPUT_DATA(DTYPE) \
static void average_input_data_ ## DTYPE(GSTLALAdaptiveFIRFilt *element, complex DTYPE *src, guint64 src_size, guint64 pts) { \
 \
	gint64 start_sample, initial_samples, samples_to_add, i; \
	int j; \
	/* Find the location in src of the first sample that will go into the average */ \
	if(element->num_in_avg) \
		start_sample = 0; \
	else \
		start_sample = (gint64) (element->update_samples - gst_util_uint64_scale_int_round(pts, element->rate, GST_SECOND) % element->update_samples) % element->update_samples; \
 \
	/* How many samples from this buffer will we need to add into this average? */ \
	samples_to_add = min64(element->average_samples - element->num_in_avg, src_size - start_sample); \
	while(samples_to_add > 0) { \
		initial_samples = element->num_in_avg; \
		for(i = start_sample; i < start_sample + samples_to_add; i++) { \
			for(j = 0; j < element->channels; j++) \
				element->input_average[j] += src[i * element->channels + j]; \
		} \
		element->num_in_avg += samples_to_add; \
		if(element->num_in_avg >= element->average_samples) { \
 \
			/* Number of samples in average should not become greater than specified by the user */ \
			g_assert_cmpint(element->num_in_avg, ==, element->average_samples); \
 \
			/* We still need to divide by n to get the average */ \
			for(j = 0; j < element->channels; j++) \
				element->input_average[j] /= element->num_in_avg; \
 \
			/* Update the adaptive FIR filter */ \
			gboolean success; \
			if(element->minimize_filter_length) \
				success = update_adaptive_filter_minimized(element->adaptive_filter, (int) element->adaptive_filter_length, element->filter_sample_rate, element->input_average, element->num_zeros, element->filter_has_gain, element->static_zeros, element->num_static_zeros, element->static_filter, element->static_filter_length); \
			else \
				success = update_padded_filter(element->padded_model, element->padded_model_length, element->padded_filter, element->padded_filter_length, element->filter_sample_rate, element->input_average, element->num_zeros, element->num_poles, element->filter_has_gain, element->static_zeros, element->num_static_zeros, element->static_poles, element->num_static_poles, element->phase_measurement_frequency, element->static_model, element->static_model_length); \
			if(success) { \
 \
				/* Apply the window function so that the filter falls off smoothly at the edges */ \
				if(element->window) { \
					for(i = 0; i < element->adaptive_filter_length; i++) \
						element->adaptive_filter[i] *= element->window[i]; \
				} \
				GST_LOG_OBJECT(element, "Just computed new FIR filter"); \
 \
				/* Let other elements know about the update */ \
				g_object_notify_by_pspec(G_OBJECT(element), properties[ARG_ADAPTIVE_FILTER]); \
				/* Provide a timestamp indicating when the filter becomes invalid if requested */ \
				if(element->filter_timeshift < G_MAXINT64) { \
					if(element->filter_timeshift < 0 && (guint64) (-element->filter_timeshift) > pts) \
						element->filter_endtime = 0; \
					else \
						element->filter_endtime = pts + element->filter_timeshift; \
					g_object_notify_by_pspec(G_OBJECT(element), properties[ARG_FILTER_ENDTIME]); \
				} \
 \
				/* Write FIR filter to the screen or a file if we want */ \
				if(element->write_to_screen || element->filename) \
					write_filter((double *) element->adaptive_filter, gst_element_get_name(element), "Adaptive", element->adaptive_filter_length, element->write_to_screen, element->filename, TRUE); \
 \
			} \
			element->num_in_avg = 0; \
			for(j = 0; j < element->channels; j++) \
				element->input_average[j] = 0.0; \
		} \
		start_sample += element->update_samples - initial_samples; \
		samples_to_add = min64(element->average_samples - element->num_in_avg, src_size - start_sample); \
	} \
 \
	return; \
}


DEFINE_AVERAGE_INPUT_DATA(float);
DEFINE_AVERAGE_INPUT_DATA(double);


/*
 * ============================================================================
 *
 *			    GstBaseSink Overrides
 *
 * ============================================================================
 */


/*
 * get_unit_size()
 */


static gboolean get_unit_size(GstBaseSink *sink, GstCaps *caps, gsize *size) {

	GstAudioInfo info;
	gboolean success = gstlal_audio_info_from_caps(&info, caps);
	if(success)
		*size = GST_AUDIO_INFO_BPF(&info);
	else
		GST_WARNING_OBJECT(sink, "unable to parse caps %" GST_PTR_FORMAT, caps);
	return success;
}


/*
 * start()
 */


static gboolean start(GstBaseSink *sink) {

	GSTLALAdaptiveFIRFilt *element = GSTLAL_ADAPTIVEFIRFILT(sink);
	double alpha;
	guint i;

	/* Timestamp bookkeeping */
	element->t0 = GST_CLOCK_TIME_NONE;
	element->offset0 = GST_BUFFER_OFFSET_NONE;
	element->next_in_offset = GST_BUFFER_OFFSET_NONE;

	/* If we are writing output to file, and a file already exists with the same name, remove it */
	if(element->filename)
		remove(element->filename);

	/* Sanity checks */
	if(element->average_samples > element->update_samples)
		GST_ERROR_OBJECT(element, "average_samples cannot be greater than update_samples");
	if(element->minimize_filter_length) {
		if(element->num_poles || element->num_static_poles) {
			GST_WARNING_OBJECT(element, "Cannot set option minimize_filter_length = True if there are poles. Keeping adaptive_filter_length as is.");
			element->minimize_filter_length = FALSE;
		} else {
			element->adaptive_filter_length = element->num_zeros + element->num_static_zeros + 1;
			element->adaptive_filter_length += (element->static_filter_length > 0 ? element->static_filter_length - 1 : 0);
		}
	}
	if(element->adaptive_filter_length * element->frequency_resolution < element->filter_sample_rate) {
		if(element->frequency_resolution > 0.0) {
			/*
			 * The user set an invalid frequency resolution, so give a warning and
			 * set the finest resolution that is within reason.
			 */
			GST_WARNING_OBJECT(element, "frequency-resolution is too fine.  Resetting frequency-resolution to be equal to the inverse of the filter length in seconds.");
			element->frequency_resolution = (double) element->filter_sample_rate / element->adaptive_filter_length;
		} else
			/* This is the default value, so set the resolution to a typical value. */
			element->frequency_resolution = 3.0 * element->filter_sample_rate / element->adaptive_filter_length;
	}

	/*
	 * Memory allocation
	 */

	/* Make a window function as specified by element properties */
	/* Frequency resolution in units of frequency bins of fft data */
	alpha = element->frequency_resolution * element->adaptive_filter_length / element->filter_sample_rate;
	switch(element->window_type) {
	case GSTLAL_ADAPTIVEFIRFILT_DPSS:
		element->window = dpss_double(element->adaptive_filter_length, alpha, 5.0, NULL, FALSE, FALSE);
		break;

	case GSTLAL_ADAPTIVEFIRFILT_KAISER:
		element->window = kaiser_double(element->adaptive_filter_length, M_PI * alpha, NULL, FALSE);
		break;

	case GSTLAL_ADAPTIVEFIRFILT_DOLPH_CHEBYSHEV:
		element->window = DolphChebyshev_double(element->adaptive_filter_length, alpha, NULL, FALSE);
		break;

	case GSTLAL_ADAPTIVEFIRFILT_BLACKMAN:
		element->window = blackman_double(element->adaptive_filter_length, NULL, FALSE);
		break;

	case GSTLAL_ADAPTIVEFIRFILT_HANN:
		element->window = hann_double(element->adaptive_filter_length, NULL, FALSE);
		break;

	case GSTLAL_ADAPTIVEFIRFILT_NONE:
		element->window = NULL;
		break;

	default:
		GST_ERROR_OBJECT(element, "Invalid window type.  See properties for appropriate window types.");
		g_assert_not_reached();
		break;
	}

	if(!element->adaptive_filter && !element->minimize_filter_length) {
		/* We are making a frequency-domain model and using gstlal_irfft to produce the adaptive filter. */
		if(element->static_model_length == 0 && element->static_filter_length != 0) {
			/* Then compute the static frequency-domain model from the static filter */
			element->static_model = gstlal_rfft_double(element->static_filter, (guint) element->static_filter_length, NULL, 0, NULL, FALSE, 0, NULL, 0, NULL, NULL, NULL, FALSE);
			element->static_model_length = element->static_filter_length / 2 + 1;

			/*
			 * Advance the static model, since the filter was (probably) centered in time.
			 * FIXME: we should handle general filters that may not be centered in time.
			 */
			for(i = 0; i < element->static_model_length; i++)
				element->static_model[i] *= cexp(2 * M_PI * I * i * (element->static_filter_length / 2) / element->static_filter_length);
		}

		/* We may need to upsample the static model or the adaptive model */
		element->padded_model_length = element->static_model_length > element->adaptive_filter_length / 2 + 1 ? element->static_model_length : element->adaptive_filter_length / 2 + 1;

		element->padded_filter_length = 2 * (element->padded_model_length - 1) + element->adaptive_filter_length % 2;

		element->padded_model = g_malloc(element->padded_model_length * sizeof(*element->padded_model));
		element->padded_filter = g_malloc(element->padded_filter_length * sizeof(*element->padded_filter));

		/* How many pad samples are there at the beginning of the padded filter? */
		element->pad_samples = (element->padded_filter_length - element->adaptive_filter_length) / 2;

		/* Set the pointer to the adaptive filter to the right location */
		element->adaptive_filter = element->padded_filter + element->pad_samples;

		/*  Upsample if we need to */
		if(element->static_model_length != 0 && element->static_model_length < element->padded_model_length) {
			element->static_model = fir_resample_complexdouble(element->static_model, (guint) element->static_model_length, (guint) element->padded_model_length);
			element->static_model_length = element->padded_model_length;
		}

	} else if (!element->adaptive_filter) {
		/* Allocate memory for the adaptive filter, but in this case, no fft's are necessary */
		element->padded_filter = g_malloc(element->adaptive_filter_length * sizeof(*element->adaptive_filter));
		element->adaptive_filter = element->padded_filter;
		element->pad_samples = 0;
	}

	return TRUE;
}


/*
 * event()
 */


static gboolean event(GstBaseSink *sink, GstEvent *event) {

	GSTLALAdaptiveFIRFilt *element = GSTLAL_ADAPTIVEFIRFILT(sink);
	gboolean success;
	GST_DEBUG_OBJECT(element, "Got %s event on sink pad", GST_EVENT_TYPE_NAME(event));

	if(GST_EVENT_TYPE(event) == GST_EVENT_EOS && element->filter_timeshift < G_MAXINT64) {
		/* These filters should remain usable as long as possible */
		element->filter_endtime = G_MAXUINT64 - 1;
		g_object_notify_by_pspec(G_OBJECT(element), properties[ARG_FILTER_ENDTIME]);
	}

	success = GST_BASE_SINK_CLASS(gstlal_adaptivefirfilt_parent_class)->event(sink, event);

	return success;
}


/*
 * set_caps()
 */


static gboolean set_caps(GstBaseSink *sink, GstCaps *caps) {

	GSTLALAdaptiveFIRFilt *element = GSTLAL_ADAPTIVEFIRFILT(sink);

	gboolean success = TRUE;

	gsize unit_size;

	/* Parse the caps to find the format, sample rate, and number of channels */
	GstStructure *str = gst_caps_get_structure(caps, 0);
	const gchar *name = gst_structure_get_string(str, "format");
	success &= (name != NULL);
	success &= gst_structure_get_int(str, "rate", &element->rate);
	success &= gst_structure_get_int(str, "channels", &element->channels);

	/* Find unit size */
	success &= get_unit_size(sink, caps, &unit_size);
	element->unit_size = unit_size;

	/* Record the data type */
	if(success) {
		if(!strcmp(name, GST_AUDIO_NE(Z64)))
			element->data_type = GSTLAL_ADAPTIVEFIRFILT_Z64;
		else if(!strcmp(name, GST_AUDIO_NE(Z128)))
			element->data_type = GSTLAL_ADAPTIVEFIRFILT_Z128;
		else
			g_assert_not_reached();
	}

	/* channels are zeros, poles, and an optional gain/phase factor */
	if(element->channels == element->num_zeros + element->num_poles)
		element->filter_has_gain = FALSE;
	else if(element->channels == element->num_zeros + element->num_poles + 1)
		element->filter_has_gain = TRUE;
	else
		GST_ERROR_OBJECT(element, "Number of channels must equal number of zeros plus number of poles, or one more than this. channels = %d, zeros + poles = %d", element->channels, element->num_zeros + element->num_poles);

	if(element->input_average) {
		g_free(element->input_average);
		element->input_average = NULL;
	}
	element->input_average = g_malloc(element->channels * sizeof(*element->input_average));
	memset(element->input_average, 0, element->channels * sizeof(*element->input_average));

	return success;
}


/*
 * render()
 */


static GstFlowReturn render(GstBaseSink *sink, GstBuffer *buffer) {

	GSTLALAdaptiveFIRFilt *element = GSTLAL_ADAPTIVEFIRFILT(sink);
	GstMapInfo mapinfo;
	GstFlowReturn result = GST_FLOW_OK;

	/*
	 * check for discontinuity
	 */

	if(G_UNLIKELY(GST_BUFFER_IS_DISCONT(buffer) || GST_BUFFER_OFFSET(buffer) != element->next_in_offset || !GST_CLOCK_TIME_IS_VALID(element->t0))) {
		element->t0 = GST_BUFFER_PTS(buffer);
		element->offset0 = GST_BUFFER_OFFSET(buffer);
	}
	element->next_in_offset = GST_BUFFER_OFFSET_END(buffer);
	GST_DEBUG_OBJECT(element, "have buffer spanning %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(buffer));

	/* Check if the data on this buffer is usable and if we plan to use it */
	gint64 next_start_sample = (element->update_samples - gst_util_uint64_scale_int_round(GST_BUFFER_PTS(buffer), element->rate, GST_SECOND) % element->update_samples) % element->update_samples;
	if(!GST_BUFFER_FLAG_IS_SET(buffer, GST_BUFFER_FLAG_GAP) && mapinfo.size && (element->num_in_avg || next_start_sample < (gint64) gst_util_uint64_scale_int_round(GST_BUFFER_DURATION(buffer), element->rate, GST_SECOND))) {
		/* Get the data from the buffer */
		gst_buffer_map(buffer, &mapinfo, GST_MAP_READ);

		if(element->data_type == GSTLAL_ADAPTIVEFIRFILT_Z64)
			average_input_data_float(element, (complex float *) mapinfo.data, mapinfo.size / element->unit_size, GST_BUFFER_PTS(buffer));
		else if(element->data_type == GSTLAL_ADAPTIVEFIRFILT_Z128)
			average_input_data_double(element, (complex double *) mapinfo.data, mapinfo.size / element->unit_size, GST_BUFFER_PTS(buffer));
		else
			g_assert_not_reached();

		gst_buffer_unmap(buffer, &mapinfo);
	}

	return result;
}


/*
 * stop()
 */


static gboolean stop(GstBaseSink *sink) {

	GSTLALAdaptiveFIRFilt *element = GSTLAL_ADAPTIVEFIRFILT(sink);

	dpss_double(0, 0.0, 0.0, NULL, FALSE, TRUE);

	if(element->padded_filter) {
		g_free(element->padded_filter);
		element->padded_filter = NULL;
	}
	if(element->padded_model) {
		g_free(element->padded_model);
		element->padded_model = NULL;
	}
	if(element->input_average) {
		g_free(element->input_average);
		element->input_average = NULL;
	}
	if(element->static_zeros) {
		g_free(element->static_zeros);
		element->static_zeros = NULL;
	}
	if(element->static_poles) {
		g_free(element->static_poles);
		element->static_poles = NULL;
	}
	if(element->static_model) {
		g_free(element->static_model);
		element->static_model = NULL;
	}
	if(element->static_filter) {
		g_free(element->static_filter);
		element->static_filter = NULL;
	}
	if(element->filename) {
		g_free(element->filename);
		element->filename = NULL;
	}
	if(element->window) {
		g_free(element->window);
		element->window = NULL;
	}

	return TRUE;
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
	GSTLALAdaptiveFIRFilt *element = GSTLAL_ADAPTIVEFIRFILT(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_UPDATE_SAMPLES:
		element->update_samples = g_value_get_int64(value);
		break;

	case ARG_AVERAGE_SAMPLES:
		element->average_samples = g_value_get_int64(value);
		break;

	case ARG_NUM_ZEROS:
		element->num_zeros = g_value_get_int(value);
		break;

	case ARG_NUM_POLES:
		element->num_poles = g_value_get_int(value);
		break;

	case ARG_STATIC_ZEROS:
		if(element->static_zeros) {
			g_free(element->static_zeros);
			element->static_zeros = NULL;
		}
		element->num_static_zeros = gst_value_array_get_size(value);
		if(element->num_static_zeros % 2)
			GST_ERROR_OBJECT(element, "Array length for property static-zeros must be even");
		double *double_static_zeros = g_malloc(element->num_static_zeros * sizeof(double));
		int i;
		for(i = 0; i < element->num_static_zeros; i++)
			double_static_zeros[i] = g_value_get_double(gst_value_array_get_value(value, i));
		element->static_zeros = (complex double *) double_static_zeros;

		/* Since we passed a complex array as though it were real, there are only half as many zeros */
		element->num_static_zeros /= 2;

		break;

	case ARG_STATIC_POLES:
		if(element->static_poles) {
			g_free(element->static_poles);
			element->static_poles = NULL;
		}
		element->num_static_poles = gst_value_array_get_size(value);
		if(element->num_static_poles % 2)
			GST_ERROR_OBJECT(element, "Array length for property static-poles must be even");
		double *double_static_poles = g_malloc(element->num_static_poles * sizeof(double));
		int j;
		for(j = 0; j < element->num_static_poles; j++)
			double_static_poles[j] = g_value_get_double(gst_value_array_get_value(value, j));
		element->static_poles = (complex double *) double_static_poles;

		/* Since we passed a complex array as though it were real, there are only half as many poles */
		element->num_static_poles /= 2;

		break;

	case ARG_PHASE_MEASUREMENT_FREQUENCY:
		element->phase_measurement_frequency = g_value_get_double(value);
		break;

	case ARG_STATIC_MODEL:
		if(element->static_model) {
			g_free(element->static_model);
			element->static_model = NULL;
		}
		element->static_model_length = gst_value_array_get_size(value);
		if(element->static_model_length % 2)
			GST_ERROR_OBJECT(element, "Array length for property static-model must be even");
		double *double_static_model = g_malloc(element->static_model_length * sizeof(double));
		int n;
		for(n = 0; n < element->static_model_length; n++)
			double_static_model[n] = g_value_get_double(gst_value_array_get_value(value, n));
		element->static_model = (complex double *) double_static_model;

		/* Since we passed a complex array as though it were real, it is actually only half as long. */
		element->static_model_length /= 2;

		break;

	case ARG_STATIC_FILTER:
		if(element->static_filter) {
			g_free(element->static_filter);
			element->static_filter = NULL;
		}
		element->static_filter_length = gst_value_array_get_size(value);
		element->static_filter = g_malloc(element->static_filter_length * sizeof(double));
		int k;
		for(k = 0; k < element->static_filter_length; k++)
			element->static_filter[k] = g_value_get_double(gst_value_array_get_value(value, k));

		/* If no static filter is provided, the filter is just 1 */
		if(element->static_filter_length == 0) {
			element->static_filter_length = 1;
			element->static_filter = g_malloc(sizeof(*element->static_filter));
			*element->static_filter = 1.0;
		}
		break;

	case ARG_MINIMIZE_FILTER_LENGTH:
		element->minimize_filter_length = g_value_get_boolean(value);
		break;

	case ARG_ADAPTIVE_FILTER_LENGTH:
		element->adaptive_filter_length = g_value_get_int64(value);
		break;

	case ARG_FREQUENCY_RESOLUTION:
		element->frequency_resolution = g_value_get_double(value);
		break;

	case ARG_FILTER_SAMPLE_RATE:
		element->filter_sample_rate = g_value_get_int(value);
		break;

	case ARG_FILTER_TIMESHIFT:
		element->filter_timeshift = g_value_get_int64(value);
		break;

	case ARG_WRITE_TO_SCREEN:
		element->write_to_screen = g_value_get_boolean(value);
		break;

	case ARG_FILENAME:
		element->filename = g_value_dup_string(value);
		break;

	case ARG_WINDOW_TYPE:
		element->window_type = g_value_get_enum(value);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *pspec)
{
	GSTLALAdaptiveFIRFilt *element = GSTLAL_ADAPTIVEFIRFILT(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_UPDATE_SAMPLES:
		g_value_set_int64(value, element->update_samples);
		break;

	case ARG_AVERAGE_SAMPLES:
		g_value_set_int64(value, element->average_samples);
		break;

	case ARG_NUM_ZEROS:
		g_value_set_int(value, element->num_zeros);
		break;

	case ARG_NUM_POLES:
		g_value_set_int(value, element->num_poles);
		break;

	case ARG_STATIC_ZEROS: ;
		GValue staticzeros = G_VALUE_INIT;
		g_value_init(&staticzeros, GST_TYPE_ARRAY);
		if(element->static_zeros) {
			double *double_static_zeros = (double *) element->static_zeros;
			int i;
			for(i = 0; i < 2 * element->num_static_zeros; i++) {
				GValue staticzero = G_VALUE_INIT;
				g_value_init(&staticzero, G_TYPE_DOUBLE);
				g_value_set_double(&staticzero, double_static_zeros[i]);
				gst_value_array_append_value(&staticzeros, &staticzero);
				g_value_unset(&staticzero);
			}
		}
		g_value_copy(&staticzeros, value);
		g_value_unset(&staticzeros);

		break;

	case ARG_STATIC_POLES: ;
		GValue staticpoles = G_VALUE_INIT;
		g_value_init(&staticpoles, GST_TYPE_ARRAY);
		if(element->static_poles) {
			double *double_static_poles = (double *) element->static_poles;
			int j;
			for(j = 0; j < 2 * element->num_static_poles; j++) {
				GValue staticpole = G_VALUE_INIT;
				g_value_init(&staticpole, G_TYPE_DOUBLE);
				g_value_set_double(&staticpole, double_static_poles[j]);
				gst_value_array_append_value(&staticpoles, &staticpole);
				g_value_unset(&staticpole);
			}
		}
		g_value_copy(&staticpoles, value); 
		g_value_unset(&staticpoles);

		break;

	case ARG_PHASE_MEASUREMENT_FREQUENCY:
		g_value_set_double(value, element->phase_measurement_frequency);
		break;

	case ARG_STATIC_MODEL: ;
		GValue staticmodel = G_VALUE_INIT;
		g_value_init(&staticmodel, GST_TYPE_ARRAY);
		if(element->static_model) {
			double *double_static_model = (double *) element->static_model;
			int n;
			for(n = 0; n < 2 * element->static_model_length; n++) {
				GValue staticmodelval = G_VALUE_INIT;
				g_value_init(&staticmodelval, G_TYPE_DOUBLE);
				g_value_set_double(&staticmodelval, double_static_model[n]);
				gst_value_array_append_value(&staticmodel, &staticmodelval);
				g_value_unset(&staticmodelval);
			}
		}
		g_value_copy(&staticmodel, value);
		g_value_unset(&staticmodel);

		break;

	case ARG_STATIC_FILTER: ;
		GValue staticfilter = G_VALUE_INIT;
		g_value_init(&staticfilter, GST_TYPE_ARRAY);
		if(element->static_filter) {
			int k;
			for(k = 0; k < element->static_filter_length; k++) {
				GValue static_filter_tap = G_VALUE_INIT;
				g_value_init(&static_filter_tap, G_TYPE_DOUBLE);
				g_value_set_double(&static_filter_tap, element->static_filter[k]);
				gst_value_array_append_value(&staticfilter, &static_filter_tap);
				g_value_unset(&static_filter_tap);
			}
		}
		g_value_copy(&staticfilter, value);
		g_value_unset(&staticfilter);

		break;

	case ARG_MINIMIZE_FILTER_LENGTH:
		g_value_set_boolean(value, element->minimize_filter_length);
		break;

	case ARG_ADAPTIVE_FILTER: ;
		GValue adaptivefilter = G_VALUE_INIT;
		g_value_init(&adaptivefilter, GST_TYPE_ARRAY);
		if(element->adaptive_filter) {
			int m;
			for(m = 0; m < element->adaptive_filter_length; m++) {
				GValue adaptive_filter_tap = G_VALUE_INIT;
				g_value_init(&adaptive_filter_tap, G_TYPE_DOUBLE);
				g_value_set_double(&adaptive_filter_tap, element->adaptive_filter[m]);
				gst_value_array_append_value(&adaptivefilter, &adaptive_filter_tap);
				g_value_unset(&adaptive_filter_tap);
			}
		}
		g_value_copy(&adaptivefilter, value);
		g_value_unset(&adaptivefilter);

		break;

	case ARG_ADAPTIVE_FILTER_LENGTH:
		g_value_set_int64(value, element->adaptive_filter_length);
		break;

	case ARG_FREQUENCY_RESOLUTION:
		g_value_set_double(value, element->frequency_resolution);
		break;

	case ARG_FILTER_SAMPLE_RATE:
		g_value_set_int(value, element->filter_sample_rate);
		break;

	case ARG_FILTER_TIMESHIFT:
		g_value_set_int64(value, element->filter_timeshift);
		break;

	case ARG_FILTER_ENDTIME:
		g_value_set_uint64(value, element->filter_endtime);
		break;

	case ARG_WRITE_TO_SCREEN:
		g_value_set_boolean(value, element->write_to_screen);
		break;

	case ARG_FILENAME:
		g_value_set_string(value, element->filename);
		break;

	case ARG_WINDOW_TYPE:
		g_value_set_enum(value, element->window_type);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * class_init()
 */


#define CAPS \
	"audio/x-raw, " \
	"rate = (int) [1, MAX], " \
	"channels = (int) [1, MAX], " \
	"format = (string) {"GST_AUDIO_NE(Z64)", "GST_AUDIO_NE(Z128)"}, " \
	"layout = (string) interleaved, " \
	"channel-mask = (bitmask) 0"


static void gstlal_adaptivefirfilt_class_init(GSTLALAdaptiveFIRFiltClass *klass) {

	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GstBaseSinkClass *gstbasesink_class = GST_BASE_SINK_CLASS(klass);

	gstbasesink_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
	gstbasesink_class->start = GST_DEBUG_FUNCPTR(start);
	gstbasesink_class->event = GST_DEBUG_FUNCPTR(event);
	gstbasesink_class->render = GST_DEBUG_FUNCPTR(render);
	gstbasesink_class->stop = GST_DEBUG_FUNCPTR(stop);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);

	gst_element_class_set_details_simple(
		element_class,
		"Compute and update a FIR filter",
		"Sink",
		"Compute and update a FIR filter using interleaved input streams of complex\n\t\t\t   "
		"numbers. The first input channel(s) represent(s) the zero(s) of the filter.\n\t\t\t   "
		"The second set of channels represents the pole(s) of the filter. If there is\n\t\t\t   "
		"another channel, it represents amplitude and phase at a chosen measurement\n\t\t\t   "
		"frequency, used to compute a gain and a frequency-independent time delay or\n\t\t\t   "
		"advance. The filter applied is therefore:\n\t\t\t   "
		"\n\t\t\t   "
		"\t     product_m[1 + i * f / f0_m]\n\t\t\t   "
		"   F(f) = --------------------------- * K * exp(2 * pi * i * f * t_adv)\n\t\t\t   "
		"\t     product_n[1 + i * f / fp_n]\n\t\t\t   "
		"\n\t\t\t   "
		"where the zeros f0_m, poles fp_n, gain K and time advance t_adv are computed\n\t\t\t   "
		"internally from the input channels. If static zeros and/or poles are\n\t\t\t   "
		"provided to this element, they will be included in the filter. Additionally,\n\t\t\t   "
		"the element can convolve a provided static filter with the computed adaptive\n\t\t\t   "
		"filter. This may be useful if a portion of a filter varies little with time\n\t\t\t   "
		"and/or is not well-modeled by zeros, poles, gain, and a time-shift.",
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


	properties[ARG_UPDATE_SAMPLES] = g_param_spec_int64(
		"update-samples",
		"Update Samples",
		"Number of input samples after which to update the adaptive FIR filter",
		0, G_MAXINT64, 320,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_AVERAGE_SAMPLES] = g_param_spec_int64(
		"average-samples",
		"Average Samples",
		"Number of input samples to average before producing the adaptive FIR filter",
		0, G_MAXINT64, 320,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_NUM_ZEROS] = g_param_spec_int(
		"num-zeros",
		"Number of Zeros",
		"Number of variable zeros in the adaptive filter. This will set the number of \n\t\t\t"
		"channels assumed to represent zeros in the filter. The element assumes that\n\t\t\t"
		"these channels come first.",
		0, G_MAXINT, 0,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_NUM_POLES] = g_param_spec_int(
		"num-poles",
		"Number of Poles",
		"Number of variable poles in the adaptive filter. This will set the number of\n\t\t\t"
		"channels assumed to represent poles in the filter. The element assumes that\n\t\t\t"
		"these channels come second after the zeros.",
		0, G_MAXINT, 0,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_STATIC_ZEROS] = gst_param_spec_array(
		"static-zeros",
		"Static Zeros",
		"Array of static zeros, which will be included in each computed filter. Since\n\t\t\t"
		"zeros can be complex, this array must contain real and imaginary parts,\n\t\t\t"
		"e.g., [z0_real, z0_imag, z1_real, z1_imag, ...]",
		g_param_spec_double(
			"static-zero",
			"Static Zero",
			"A zero (real or imaginary part) from the array of static zeros",
			-G_MAXDOUBLE, G_MAXDOUBLE, 0.0,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		),
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_STATIC_POLES] = gst_param_spec_array(
		"static-poles",
		"Static Poles",
		"Array of static poles, which will be included in each computed filter. Since\n\t\t\t"
		"poles can be complex, this array must contain real and imaginary parts,\n\t\t\t"
		"e.g., [p0_real, p0_imag, p1_real, p1_imag, ...]",
		g_param_spec_double(
			"static-pole",
			"Static Pole",
			"A pole (real or imaginary part) from the array of static poles",
			-G_MAXDOUBLE, G_MAXDOUBLE, 0.0,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		),
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_PHASE_MEASUREMENT_FREQUENCY] = g_param_spec_double(
		"phase-measurement-frequency",
		"Phase Measurement Frequency",
		"Frequency at which gain factor and timing offset are computed. If\n\t\t\t"
		"unset (or set to zero), any gain/phase channel is assumed to represent a\n\t\t\t"
		"frequency-independent gain/phase factor.",
		-G_MAXDOUBLE, G_MAXDOUBLE, 0.0,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_STATIC_MODEL] = gst_param_spec_array(
		"static-model",
		"Static Model",
		"A static frequency-domain model that is multiplied by the computed adaptive\n\t\t\t"
		"filter in the frequency domain",
		g_param_spec_double(
			"static-model-sample",
			"Static Model Sample",
			"A sample from the static model",
			-G_MAXDOUBLE, G_MAXDOUBLE, 0.0,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		),
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_STATIC_FILTER] = gst_param_spec_array(
		"static-filter",
		"Static Filter",
		"A static filter that is convolved with the computed adaptive filter",
		g_param_spec_double(
			"static-sample",
			"Static Sample",
			"A sample from the static filter",
			-G_MAXDOUBLE, G_MAXDOUBLE, 0.0,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		),
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_MINIMIZE_FILTER_LENGTH] = g_param_spec_boolean(
		"minimize-filter-length",
		"Minimize Filter Length",
		"If true, the adaptive FIR filter will have the minimum number of taps needed\n\t\t\t"
		"to model the given zeros (one more than the number of zeros). This cannot be\n\t\t\t"
		"done if there are poles.",
		FALSE,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_ADAPTIVE_FILTER] = gst_param_spec_array(
		"adaptive-filter",
		"Adaptive Filter",
		"The computed adaptive filter. This includes both time-varying parameters like\n\t\t\t"
		"the zeros, poles, and gain, as well as a static filter if provided. It is\n\t\t\t"
		"updated and handed off to any filtering elements as often as is set by the\n\t\t\t"
		"property update-samples.",
		g_param_spec_double(
			"sample",
			"Sample",
			"A sample from the adaptive filter",
			-G_MAXDOUBLE, G_MAXDOUBLE, 0.0,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		),
		G_PARAM_READABLE | G_PARAM_STATIC_STRINGS
	);
	properties[ARG_ADAPTIVE_FILTER_LENGTH] = g_param_spec_int64(
		"adaptive-filter-length",
		"Adaptive Filter Length",
		"Length in samples of adaptive filter produced.  This will be overridden if\n\t\t\t"
		"minimize-filter-length is set to TRUE.",
		1, G_MAXINT64, 16384,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_FREQUENCY_RESOLUTION] = g_param_spec_double(
		"frequency-resolution",
		"Frequency Resolution of FIR filter",
		"This parameter sets the frequency resolution (in Hz) of the window function\n\t\t\t"
		"applied to the adaptive filter.  It must be greater than the inverse of the\n\t\t\t"
		"length of the filter in seconds; otherwise, it will be overridden.  If unset\n\t\t\t"
		"or set to zero (default), the frequency resolution will be reset to 3\n\t\t\t"
		"frequency bins, or 3 times the inverse of the temporal duratation of the\n\t\t\t"
		"adaptive filter.",
		0.0, G_MAXDOUBLE, 0.0,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_FILTER_SAMPLE_RATE] = g_param_spec_int(
		"filter-sample-rate",
		"Filter Sample Rate",
		"The sample rate of the data being filtered",
		1, G_MAXINT, 16384,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_FILTER_TIMESHIFT] = g_param_spec_int64(
		"filter-timeshift",
		"Filter time-shift",
		"The number of nanoseconds after the completion of a FIR filter calculation\n\t\t\t"
		"that the FIR filter remains valid for use on the filtered data.  This is\n\t\t\t"
		"added to the presentation timestamp when the filter is completed to compute\n\t\t\t"
		"the filter-endtime property.  Default is to disable.",
		G_MININT64, G_MAXINT64, G_MAXINT64,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_FILTER_ENDTIME] = g_param_spec_uint64(
		"filter-endtime",
		"Filter end time",
		"The time when a computed FIR filter ceases to be valid for use on\n\t\t\t"
		"filtered data.  This can be compared to the presentation timestamps of the\n\t\t\t"
		"filtered data to determine whether the filter is still valid.  Default is\n\t\t\t"
		"to disable.",
		0, G_MAXUINT64, G_MAXUINT64,
		G_PARAM_READABLE | G_PARAM_STATIC_STRINGS
	);
	properties[ARG_WRITE_TO_SCREEN] = g_param_spec_boolean(
		"write-to-screen",
		"Write to Screen",
		"Set to true in order to write the computed FIR filters to the screen.",
		FALSE,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_FILENAME] = g_param_spec_string(
		"filename",
		"Filename",
		"Name of file to write computed FIR filters to. If not given, no file\n\t\t\t"
		"is produced.",
		NULL,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_WINDOW_TYPE] = g_param_spec_enum(
		"window-type",
		"Window Function Type",
		"What window function to apply to the FIR filters",
		GSTLAL_ADAPTIVEFIRFILT_WINDOW_TYPE,
		GSTLAL_ADAPTIVEFIRFILT_DPSS,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);


	g_object_class_install_property(
		gobject_class,
		ARG_UPDATE_SAMPLES,
		properties[ARG_UPDATE_SAMPLES]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_AVERAGE_SAMPLES,
		properties[ARG_AVERAGE_SAMPLES]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_NUM_ZEROS,
		properties[ARG_NUM_ZEROS]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_NUM_POLES,
		properties[ARG_NUM_POLES]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_STATIC_ZEROS,
		properties[ARG_STATIC_ZEROS]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_STATIC_POLES,
		properties[ARG_STATIC_POLES]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_PHASE_MEASUREMENT_FREQUENCY,
		properties[ARG_PHASE_MEASUREMENT_FREQUENCY]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_STATIC_MODEL,
		properties[ARG_STATIC_MODEL]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_STATIC_FILTER,
		properties[ARG_STATIC_FILTER]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_MINIMIZE_FILTER_LENGTH,
		properties[ARG_MINIMIZE_FILTER_LENGTH]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_ADAPTIVE_FILTER,
		properties[ARG_ADAPTIVE_FILTER]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_ADAPTIVE_FILTER_LENGTH,
		properties[ARG_ADAPTIVE_FILTER_LENGTH]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_FREQUENCY_RESOLUTION,
		properties[ARG_FREQUENCY_RESOLUTION]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_FILTER_SAMPLE_RATE,
		properties[ARG_FILTER_SAMPLE_RATE]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_FILTER_TIMESHIFT,
		properties[ARG_FILTER_TIMESHIFT]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_FILTER_ENDTIME,
		properties[ARG_FILTER_ENDTIME]
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
		ARG_WINDOW_TYPE,
		properties[ARG_WINDOW_TYPE]
	);
}


/*
 * init()
 */


static void gstlal_adaptivefirfilt_init(GSTLALAdaptiveFIRFilt *element) {

	g_signal_connect(G_OBJECT(element), "notify::adaptive-filter", G_CALLBACK(rebuild_workspace_and_reset), NULL);
	element->rate = 0;
	element->unit_size = 0;
	element->channels = 0;

	element->input_average = NULL;
	element->num_in_avg = 0;
	element->static_zeros = NULL;
	element->num_static_zeros = 0;
	element->num_zeros = 0;
	element->static_poles = NULL;
	element->num_static_poles = 0;
	element->num_poles = 0;
	element->static_model = NULL;
	element->static_model_length = 0;
	element->static_filter = NULL;
	element->static_filter_length = 0;
	element->adaptive_filter = NULL;
	element->adaptive_filter_length = 0;
	element->padded_filter = NULL;
	element->padded_filter_length = 0;
	element->pad_samples = 0;
	element->padded_model = NULL;
	element->padded_model_length = 0;
	element->frequency_resolution = 0.0;
	element->window = NULL;
	element->filename = NULL;
	element->window_type = 0;

	gst_base_sink_set_sync(GST_BASE_SINK(element), FALSE);
	gst_base_sink_set_async_enabled(GST_BASE_SINK(element), FALSE);
}


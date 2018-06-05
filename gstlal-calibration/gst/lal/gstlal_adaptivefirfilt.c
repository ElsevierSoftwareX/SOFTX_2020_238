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
 * @short_description:  Compute linear-phase FIR filter given gain and
 * phase in the form of a complex input.
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
 * stuff from FFTW
 */


#include <fftw3.h>


/*
 * our own stuff
 */


#include <gstlal/gstlal.h>
#include <gstlal/gstlal_audio_info.h>
#include <gstlal/gstlal_debug.h>
#include <gstlal_adaptivefirfilt.h>


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
	ARG_VARIABLE_FILTER_LENGTH,
	ARG_ADAPTIVE_FILTER,
	ARG_FILTER_SAMPLE_RATE,
	ARG_WRITE_TO_SCREEN,
	ARG_FILENAME,
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


static void write_adaptive_filter(double *filter, char *element_name, gint64 rows, gboolean write_to_screen, char *filename, gboolean free_name) {
	gint64 i;
	if(write_to_screen) {
		g_print("================== FIR filter computed by %s ==================\n", element_name);

		for(i = 0; i < rows; i++)
			g_print("%10e\n", filter[i]);
		g_print("\n\n");
	}

	if(filename) {
		FILE *fp;
		fp = fopen(filename, "a");
		g_fprintf(fp, "================== FIR filter computed by %s ==================\n", element_name);

		for(i = 0; i < rows; i++)
			g_fprintf(fp, "%10e\n", filter[i]);
		g_fprintf(fp, "\n\n");
		fclose(fp);
	}
	if(free_name)
		g_free(element_name);
}


static void convolve(double *filter1, gint64 length1, double *filter2, gint64 length2, double *convolved_filter) {
	gint64 i, j;
	memset(convolved_filter, 0, (length1 + length2 - 1) * sizeof(*convolved_filter));
	for(i = 0; i < length1; i++) {
		for(j = 0; j < length2; j++)
			convolved_filter[i + j] += filter1[i] * filter2[j];
	}
}


static void update_variable_filter(complex double *variable_filter, gint64 variable_filter_length, int filter_sample_rate, fftw_plan variable_filter_plan, complex double *input_average, int num_zeros, int num_poles, gboolean filter_has_gain, complex double *static_zeros, int num_static_zeros, complex double *static_poles, int num_static_poles, double phase_measurement_frequency) {

	/*
	 * Compute the filter in the frequency domain
	 */

	gint64 n, fd_variable_filter_length = variable_filter_length / 2 + 1;
	int m;
	complex double gain, two_pi_i_df_t, df = (complex double) filter_sample_rate / 2.0 / (fd_variable_filter_length - 1.0);

	if(filter_has_gain && phase_measurement_frequency) {
		/*
		 * In this case, we have a phase correction to include in the filter. At each point
		 * in frequency space, the filter is gain * exp(2 pi i f t). The value of f (= n * df)
		 * ranges from DC (0) to the Nyquist frequency (fir_sample_rate / 2) in evenly spaced
		 * increments. The rest of the parameters in the exp() are constant.
		 */

		gain = (complex double) cabs(input_average[num_zeros + num_poles]) / variable_filter_length;
		two_pi_i_df_t = clog(input_average[num_zeros + num_poles] / gain / variable_filter_length) / phase_measurement_frequency * df;
	} else if(filter_has_gain) {
		gain = input_average[num_zeros + num_poles] / variable_filter_length;
		two_pi_i_df_t = 0.0;
	} else {
		gain = (complex double) 1.0 / variable_filter_length;
		two_pi_i_df_t = 0.0;
	}

	/*
	 * At each point in frequency space, the gain/phase portion of the filter is 
	 * gain * exp(2 pi i f t). The value of f (= n * df) ranges from DC (0) to the
	 * Nyquist frequency (fir_sample_rate / 2) in evenly spaced increments. The
	 * rest of the parameters in the exp() are constant. 
	 */

	for(n = 0; n < fd_variable_filter_length; n += 2)
		variable_filter[n] = gain * cexp(two_pi_i_df_t * n);

	/* Negating every other value adds a delay of half the length of the filter, centering it in time. */
	for(n = 1; n < fd_variable_filter_length; n += 2)
		variable_filter[n] = -gain * cexp(two_pi_i_df_t * n);

	/* Now add zeros and poles */
	for(n = 0; n < fd_variable_filter_length; n++) {
		/* variable zeros */
		for(m = 0; m < num_zeros; m++)
			variable_filter[n] *= n * df - input_average[m];
		/* variable poles */
		for(m = num_zeros; m < num_zeros + num_poles; m++)
			variable_filter[n] /= n * df - input_average[m];
		/* static zeros */
		for(m = 0; m < num_static_zeros; m++)
			variable_filter[n] *= n * df - static_zeros[m];
		/* static poles */
		for(m = 0; m < num_static_poles; m++)
			variable_filter[n] /= n * df - static_poles[m];
	}

	/* Make sure the DC and Nyquist components are purely real */
	variable_filter[0] = creal(variable_filter[0]);
	variable_filter[fd_variable_filter_length - 1] = creal(variable_filter[fd_variable_filter_length - 1]);

	/* Take the inverse Fourier transform */
	fftw_execute(variable_filter_plan);

	return;
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
			/* Update variable portion of adaptive FIR filter */ \
			update_variable_filter(element->variable_filter, element->variable_filter_length, element->filter_sample_rate, element->variable_filter_plan, element->input_average, element->num_zeros, element->num_poles, element->filter_has_gain, element->static_zeros, element->num_static_zeros, element->static_poles, element->num_static_poles, element->phase_measurement_frequency); \
 \
			/* Convolve the variable filter with the static filter */ \
			convolve((double *) element->variable_filter, element->variable_filter_length, element->static_filter, element->static_filter_length, element->adaptive_filter); \
 \
			GST_LOG_OBJECT(element, "Just computed new FIR filter"); \
 \
			/* Let other elements know about the update */ \
			g_object_notify_by_pspec(G_OBJECT(element), properties[ARG_ADAPTIVE_FILTER]); \
 \
			/* Write FIR filter to the screen or a file if we want */ \
			if(element->write_to_screen || element->filename) \
				write_adaptive_filter((double *) element->adaptive_filter, gst_element_get_name(element), element->variable_filter_length + element->static_filter_length - 1, element->write_to_screen, element->filename, TRUE); \
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
 * start()
 */


static gboolean start(GstBaseSink *sink) {

	GSTLALAdaptiveFIRFilt *element = GSTLAL_ADAPTIVEFIRFILT(sink);

	/* Timestamp bookkeeping */
	element->t0 = GST_CLOCK_TIME_NONE;
	element->offset0 = GST_BUFFER_OFFSET_NONE;
	element->next_in_offset = GST_BUFFER_OFFSET_NONE;

	/* If we are writing output to file, and a file already exists with the same name, remove it */
	if(element->filename)
		remove(element->filename);

	/* Sanity check */
	if(element->average_samples > element->update_samples)
		GST_ERROR_OBJECT(element, "average_samples cannot be greater than update_samples");

	return TRUE;
}


/*
 * set_caps()
 */


static gboolean set_caps(GstBaseSink *sink, GstCaps *caps) {

	GSTLALAdaptiveFIRFilt *element = GSTLAL_ADAPTIVEFIRFILT(sink);

	gboolean success = TRUE;

	/* Parse the caps to find the format, sample rate, and number of channels */
	GstStructure *str = gst_caps_get_structure(caps, 0);
	const gchar *name = gst_structure_get_string(str, "format");
	success &= (name != NULL);
	success &= gst_structure_get_int(str, "rate", &element->rate);
	success &= gst_structure_get_int(str, "channels", &element->channels);

	/* Record the data type and unit size */
	if(success) {
		if(!strcmp(name, GST_AUDIO_NE(Z64))) {
			element->data_type = GSTLAL_ADAPTIVEFIRFILT_Z64;
			element->unit_size = 8;
		} else if(!strcmp(name, GST_AUDIO_NE(Z128))) {
			element->data_type = GSTLAL_ADAPTIVEFIRFILT_Z128;
			element->unit_size = 16;
		} else
			g_assert_not_reached();
	}

	/* channels are zeros, poles, and an optional gain/phase factor */
	if(element->channels == element->num_zeros + element->num_poles)
		element->filter_has_gain = FALSE;
	else if(element->channels == element->num_zeros + element->num_poles + 1)
		element->filter_has_gain = TRUE;
	else
		GST_ERROR_OBJECT(element, "Number of channels must equal number of zeros plus number of poles, or one more than this.");

	/* Memory allocation */
	if(element->input_average) {
		g_free(element->input_average);
		element->input_average = NULL;
	}
	element->input_average = g_malloc(element->channels * sizeof(*element->input_average));

	if(!element->adaptive_filter)
		element->adaptive_filter = g_malloc((element->variable_filter_length + element->static_filter_length - 1) * sizeof(*element->adaptive_filter));

	if(!element->variable_filter) {
		/* Allocate memory for fftw to do an inverse Fourier transform of the filter. */
		gstlal_fftw_lock();

		GST_LOG_OBJECT(element, "starting FFTW planning");

		gint64 fd_variable_filter_length = element->variable_filter_length / 2 + 1;
		element->variable_filter = (complex double *) fftw_malloc(fd_variable_filter_length * sizeof(*element->variable_filter));
		element->variable_filter_plan = fftw_plan_dft_c2r_1d(element->variable_filter_length, element->variable_filter, (double *) element->variable_filter, FFTW_MEASURE);

		GST_LOG_OBJECT(element, "FFTW planning complete");

		gstlal_fftw_unlock();

	}

	/* Initialize averages */
	int i;
	for(i = 0; i < element->channels; i++)
		element->input_average[i] = 0.0;
	element->num_in_avg = 0;

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
	}

	return result;
}


/*
 * stop()
 */


static gboolean stop(GstBaseSink *sink) {

	GSTLALAdaptiveFIRFilt *element = GSTLAL_ADAPTIVEFIRFILT(sink);

	gstlal_fftw_lock();
	fftw_free(element->variable_filter);
	element->variable_filter = NULL;
	fftw_destroy_plan(element->variable_filter_plan);
	gstlal_fftw_unlock();

	g_free(element->input_average);
	element->input_average = NULL;
	g_free(element->static_zeros);
	element->static_zeros = NULL;
	g_free(element->static_poles);
	element->static_poles = NULL;
	g_free(element->static_filter);
	element->static_filter = NULL;
	g_free(element->adaptive_filter);
	element->adaptive_filter = NULL;
	g_free(element->filename);
	element->filename = NULL;

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
		if(element->static_zeros)
			g_free(element->static_zeros);
		gstlal_doubles_from_g_value_array(g_value_get_boxed(value), (double *) element->static_zeros, &element->num_static_zeros);
		/* Since we passed a complex array as though it were real, there are only half as many zeros */
		element->num_static_zeros /= 2;
		break;

	case ARG_STATIC_POLES:
		if(element->static_poles)
			g_free(element->static_poles);
		gstlal_doubles_from_g_value_array(g_value_get_boxed(value), (double *) element->static_poles, &element->num_static_poles);
		/* Since we passed a complex array as though it were real, there are only half as many poles */
		element->num_static_poles /= 2;
		break;

	case ARG_PHASE_MEASUREMENT_FREQUENCY:
		element->phase_measurement_frequency = g_value_get_double(value);
		break;

	case ARG_STATIC_FILTER:
		if(element->static_filter)
			g_free(element->static_filter);
		gstlal_doubles_from_g_value_array(g_value_get_boxed(value), element->static_filter, (int *) &element->static_filter_length);

		/* If no static filter is provided, the filter is just 1 */
		if(element->static_filter_length == 0) {
			element->static_filter_length = 1;
			*element->static_filter = 1.0;
		}
		break;

	case ARG_VARIABLE_FILTER_LENGTH:
		element->variable_filter_length = g_value_get_int64(value);
		break;

	case ARG_ADAPTIVE_FILTER:
		if(element->adaptive_filter)
			g_free(element->adaptive_filter);
		int m;
		gstlal_doubles_from_g_value_array(g_value_get_boxed(value), element->adaptive_filter, &m);
		break;

	case ARG_FILTER_SAMPLE_RATE:
		element->filter_sample_rate = g_value_get_int(value);
		break;

	case ARG_WRITE_TO_SCREEN:
		element->write_to_screen = g_value_get_boolean(value);
		break;

	case ARG_FILENAME:
		element->filename = g_value_dup_string(value);
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

	case ARG_STATIC_ZEROS:
		if(element->static_zeros)
			g_value_take_boxed(value, gstlal_g_value_array_from_doubles((double *) element->static_zeros, 2 * element->num_static_zeros));
		else
			g_value_take_boxed(value, g_value_array_new(0));
		break;

	case ARG_STATIC_POLES:
		if(element->static_poles)
			g_value_take_boxed(value, gstlal_g_value_array_from_doubles((double *) element->static_poles, 2 * element->num_static_poles));
		else
			g_value_take_boxed(value, g_value_array_new(0));
		break;

	case ARG_PHASE_MEASUREMENT_FREQUENCY:
		g_value_set_double(value, element->phase_measurement_frequency);
		break;

	case ARG_STATIC_FILTER:
		if(element->static_filter)
			g_value_take_boxed(value, gstlal_g_value_array_from_doubles(element->static_filter, element->static_filter_length));
		else
			g_value_take_boxed(value, g_value_array_new(0));
		break;

	case ARG_VARIABLE_FILTER_LENGTH:
		g_value_set_int64(value, element->variable_filter_length);
		break;

	case ARG_ADAPTIVE_FILTER:
		if(element->adaptive_filter)
			g_value_take_boxed(value, gstlal_g_value_array_from_doubles(element->adaptive_filter, element->variable_filter_length + element->static_filter_length - 1));
		else
			g_value_take_boxed(value, g_value_array_new(0));
		break;

	case ARG_FILTER_SAMPLE_RATE:
		g_value_set_int(value, element->filter_sample_rate);
		break;

	case ARG_WRITE_TO_SCREEN:
		g_value_set_boolean(value, element->write_to_screen);
		break;

	case ARG_FILENAME:
		g_value_set_string(value, element->filename);
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

	GSTLALAdaptiveFIRFilt *element = GSTLAL_ADAPTIVEFIRFILT(object);

	if(element->variable_filter) {
		g_free(element->variable_filter);
		element->variable_filter = NULL;
	}
	G_OBJECT_CLASS(gstlal_adaptivefirfilt_parent_class)->finalize(object);
}


/*
 * class_init()
 */


#define CAPS \
	"audio/x-raw, " \
	"rate = (int) [1, MAX], " \
	"channels = (int) 1, " \
	"format = (string) {"GST_AUDIO_NE(Z64)", "GST_AUDIO_NE(Z128)"}, " \
	"layout = (string) interleaved, " \
	"channel-mask = (bitmask) 0"


static void gstlal_adaptivefirfilt_class_init(GSTLALAdaptiveFIRFiltClass *klass) {

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
		"Compute and update a FIR filter",
		"Sink",
		"Compute and update a FIR filter using interleaved input streams of complex\n\t\t\t   "
		"numbers. The first input channel(s) represent(s) the zero(s) of the filter.\n\t\t\t   "
		"The second set of channels represents the pole(s) of the filter. If there is\n\t\t\t   "
		"another channel, it represents amplitude and phase at a chosen measurement\n\t\t\t   "
		"frequency, used to compute a gain and a frequency-independent time delay or\n\t\t\t   "
		"advance. The filter applied is therefore:\n\t\t\t   "
		"\n\t\t\t   "
		"\t     product_i[f - f_i]\n\t\t\t   "
		"   F(f) = ------------------ * K * exp(2 * pi * i * f * t_adv)\n\t\t\t   "
		"\t     product_j[f - f_j]\n\t\t\t   "
		"\n\t\t\t   "
		"where the zeros f_i, poles f_j, gain K and time advance t_adv are computed\n\t\t\t   "
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
		"Number of zeros in the variable filter. This will set the number of channels\n\t\t\t"
		"assumed to represent zeros in the filter. The element assumes that these\n\t\t\t"
		"channels come first.",
		0, G_MAXINT, 0,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_NUM_POLES] = g_param_spec_int(
		"num-poles",
		"Number of Poles",
		"Number of poles in the variable filter. This will set the number of channels\n\t\t\t"
		"assumed to represent poles in the filter. The element assumes that these\n\t\t\t"
		"channels come second after the zeros.",
		0, G_MAXINT, 0,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_STATIC_ZEROS] = g_param_spec_value_array(
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
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_CONTROLLABLE
	);
	properties[ARG_STATIC_POLES] = g_param_spec_value_array(
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
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_CONTROLLABLE
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
	properties[ARG_STATIC_FILTER] = g_param_spec_value_array(
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
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_CONTROLLABLE
	);
	properties[ARG_VARIABLE_FILTER_LENGTH] = g_param_spec_int64(
		"variable-filter-length",
		"Variable Filter Length",
		"Length in samples of variable filter produced. This length does not include\n\t\t\t"
		"the length of the static filter. The total filter length will be one less\n\t\t\t"
		"than the sum of the variable filter length and the static filter length.",
		1, G_MAXINT64, 16384,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_ADAPTIVE_FILTER] = g_param_spec_value_array(
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
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_CONTROLLABLE
	);
	properties[ARG_FILTER_SAMPLE_RATE] = g_param_spec_int(
		"filter-sample-rate",
		"Filter Sample Rate",
		"The sample rate of the data being filtered",
		1, G_MAXINT, 16384,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
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
		ARG_STATIC_FILTER,
		properties[ARG_STATIC_FILTER]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_VARIABLE_FILTER_LENGTH,
		properties[ARG_VARIABLE_FILTER_LENGTH]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_ADAPTIVE_FILTER,
		properties[ARG_ADAPTIVE_FILTER]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_FILTER_SAMPLE_RATE,
		properties[ARG_FILTER_SAMPLE_RATE]
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
}


/*
 * init()
 */


static void gstlal_adaptivefirfilt_init(GSTLALAdaptiveFIRFilt *element) {

	g_signal_connect(G_OBJECT(element), "notify::fir-filter", G_CALLBACK(rebuild_workspace_and_reset), NULL);
	element->rate = 0;
	element->unit_size = 0;
	element->channels = 0;

	element->input_average = NULL;
	element->variable_filter = NULL;
	element->static_zeros = NULL;
	element->static_poles = NULL;
	element->static_filter = NULL;
	element->adaptive_filter = NULL;
	element->filename = NULL;

	gst_base_sink_set_sync(GST_BASE_SINK(element), FALSE);
	gst_base_sink_set_async_enabled(GST_BASE_SINK(element), FALSE);
}


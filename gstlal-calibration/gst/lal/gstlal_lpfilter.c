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
 * SECTION:gstlal_lpfilter
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
#include <gstlal_lpfilter.h>


#define SINC_LENGTH 25


/*
 * ============================================================================
 *
 *			   GStreamer Boilerplate
 *
 * ============================================================================
 */


#define GST_CAT_DEFAULT gstlal_lpfilter_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


G_DEFINE_TYPE_WITH_CODE(
	GSTLALLPFilter,
	gstlal_lpfilter,
	GST_TYPE_BASE_SINK,
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "lal_lpfilter", 0, "lal_lpfilter element")
);


enum property {
	ARG_MEASUREMENT_FREQUENCY = 1,
	ARG_UPDATE_SAMPLES,
	ARG_AVERAGE_SAMPLES,
	ARG_WRITE_TO_SCREEN,
	ARG_FILENAME,
	ARG_FIR_LENGTH,
	ARG_FIR_SAMPLE_RATE,
	ARG_FIR_FILTER,
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


static void write_fir_filter(double *filter, char *element_name, gint64 rows, gboolean write_to_screen, char *filename, gboolean free_name) {
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


static void update_fir_filter(complex double *fir_filter, fftw_plan fir_plan, complex double input_average, double measurement_frequency, gint64 fir_length, int fir_sample_rate) {

	gint64 n, fd_fir_length = fir_length / 2 + 1;

	/*
	 * Compute the filter in the frequency domain 
	 */

	double gain = cabs(input_average) / fir_length;

	/*
	 * At each point in frequency space, the filter is gain * exp(2 pi i f t). The value
	 * of f (= n * df) ranges from DC (0) to the Nyquist frequency (fir_sample_rate / 2)
	 * in evenly spaced increments. The rest of the parameters in the exp() are constant. 
	 */

	complex double two_pi_i_df_t = clog(input_average / gain / fir_length) / measurement_frequency * fir_sample_rate / 2 / (fd_fir_length - 1);
	for(n = 0; n < fd_fir_length; n += 2)
		fir_filter[n] = gain * cexp(two_pi_i_df_t * n);

	/* Negating every other value adds a delay of half the length of the filter, centering it in time. */
	for(n = 1; n < fd_fir_length; n += 2)
		fir_filter[n] = -gain * cexp(two_pi_i_df_t * n);

	/* Make sure the DC and Nyquist components are purely real */
	fir_filter[0] = creal(fir_filter[0]);
	fir_filter[fd_fir_length - 1] = creal(fir_filter[fd_fir_length - 1]);

	/* Take the inverse Fourier transform */
	fftw_execute(fir_plan);

	return;
}


#define DEFINE_AVERAGE_INPUT_DATA(DTYPE) \
static void average_input_data_ ## DTYPE(GSTLALLPFilter *element, complex DTYPE *src, guint64 src_size, guint64 pts) { \
 \
	gint64 start_sample, initial_samples, samples_to_add, i; \
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
		for(i = start_sample; i < start_sample + samples_to_add; i++) \
			element->input_average += src[i]; \
		element->num_in_avg += samples_to_add; \
		if(element->num_in_avg >= element->average_samples) { \
 \
			/* Number of samples in average should not become greater than specified by the user */ \
			g_assert_cmpint(element->num_in_avg, ==, element->average_samples); \
 \
			/* We still need to divide by n to get the average */ \
			element->input_average /= element->num_in_avg; \
 \
			/* Update FIR filter */ \
			update_fir_filter(element->fir_filter, element->fir_plan, element->input_average, element->measurement_frequency, element->fir_length, element->fir_sample_rate); \
			GST_LOG_OBJECT(element, "Just computed new linear-phase FIR filter"); \
 \
			/* Let other elements know about the update */ \
			g_object_notify_by_pspec(G_OBJECT(element), properties[ARG_FIR_FILTER]); \
 \
			/* Write FIR filter to the screen or a file if we want */ \
			if(element->write_to_screen || element->filename) \
				write_fir_filter((double *) element->fir_filter, gst_element_get_name(element), element->fir_length, element->write_to_screen, element->filename, TRUE); \
			element->num_in_avg = 0; \
			element->input_average = 0.0; \
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

	GSTLALLPFilter *element = GSTLAL_LPFILTER(sink);

	/* Timestamp bookkeeping */
	element->t0 = GST_CLOCK_TIME_NONE;
	element->offset0 = GST_BUFFER_OFFSET_NONE;
	element->next_in_offset = GST_BUFFER_OFFSET_NONE;

	/* Filter memory */
	element->input_average = 0.0;
	element->num_in_avg = 0;

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

	GSTLALLPFilter *element = GSTLAL_LPFILTER(sink);

	gboolean success = TRUE;

	/* Parse the caps to find the format, sample rate, and number of channels */
	GstStructure *str = gst_caps_get_structure(caps, 0);
	const gchar *name = gst_structure_get_string(str, "format");
	success &= (name != NULL);
	success &= gst_structure_get_int(str, "rate", &element->rate);

	/* Record the data type and unit size */
	if(success) {
		if(!strcmp(name, GST_AUDIO_NE(Z64))) {
			element->data_type = GSTLAL_LPFILTER_Z64;
			element->unit_size = 8;
		} else if(!strcmp(name, GST_AUDIO_NE(Z128))) {
			element->data_type = GSTLAL_LPFILTER_Z128;
			element->unit_size = 16;
		} else
			g_assert_not_reached();
	}

	if(!element->fir_filter) {
		/* Allocate memory for fftw to do an inverse Fourier transform of the filter. */
		gstlal_fftw_lock();

		GST_LOG_OBJECT(element, "starting FFTW planning");

		gint64 fd_fir_length = element->fir_length / 2 + 1;
		element->fir_filter = (complex double *) fftw_malloc(fd_fir_length * sizeof(*element->fir_filter));
		element->fir_plan = fftw_plan_dft_c2r_1d(element->fir_length, element->fir_filter, (double *) element->fir_filter, FFTW_MEASURE);

		GST_LOG_OBJECT(element, "FFTW planning complete");

		gstlal_fftw_unlock();

	}

	return success;
}


/*
 * render()
 */


static GstFlowReturn render(GstBaseSink *sink, GstBuffer *buffer) {

	GSTLALLPFilter *element = GSTLAL_LPFILTER(sink);
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

		if(element->data_type == GSTLAL_LPFILTER_Z64)
			average_input_data_float(element, (complex float *) mapinfo.data, mapinfo.size / element->unit_size, GST_BUFFER_PTS(buffer));
		else if(element->data_type == GSTLAL_LPFILTER_Z128)
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

	GSTLALLPFilter *element = GSTLAL_LPFILTER(sink);

	gstlal_fftw_lock();
	fftw_free(element->fir_filter);
	element->fir_filter = NULL;
	fftw_destroy_plan(element->fir_plan);
	gstlal_fftw_unlock();

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
	GSTLALLPFilter *element = GSTLAL_LPFILTER(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_MEASUREMENT_FREQUENCY:
		element->measurement_frequency = g_value_get_double(value);
		break;

	case ARG_UPDATE_SAMPLES:
		element->update_samples = g_value_get_int64(value);
		break;

	case ARG_AVERAGE_SAMPLES:
		element->average_samples = g_value_get_int64(value);
		break;

	case ARG_WRITE_TO_SCREEN:
		element->write_to_screen = g_value_get_boolean(value);
		break;

	case ARG_FILENAME:
		element->filename = g_value_dup_string(value);
		break;

	case ARG_FIR_LENGTH:
		element->fir_length = g_value_get_int64(value);
		break;

	case ARG_FIR_SAMPLE_RATE:
		element->fir_sample_rate = g_value_get_int(value);
		break;

	case ARG_FIR_FILTER:
		if(element->fir_filter)
			g_free(element->fir_filter);
		int m;
		gstlal_doubles_from_g_value_array(g_value_get_boxed(value), (double *) element->fir_filter, &m);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *pspec)
{
	GSTLALLPFilter *element = GSTLAL_LPFILTER(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_MEASUREMENT_FREQUENCY:
		g_value_set_double(value, element->measurement_frequency);
		break;

	case ARG_UPDATE_SAMPLES:
		g_value_set_int64(value, element->update_samples);
		break;

	case ARG_AVERAGE_SAMPLES:
		g_value_set_int64(value, element->average_samples);
		break;

	case ARG_WRITE_TO_SCREEN:
		g_value_set_boolean(value, element->write_to_screen);
		break;

	case ARG_FILENAME:
		g_value_set_string(value, element->filename);
		break;

	case ARG_FIR_LENGTH:
		g_value_set_int64(value, element->fir_length);
		break;

	case ARG_FIR_SAMPLE_RATE:
		g_value_set_int(value, element->fir_sample_rate);
		break;

	case ARG_FIR_FILTER:
		if(element->fir_filter)
			g_value_take_boxed(value, gstlal_g_value_array_from_doubles((double *) element->fir_filter, element->fir_length));
		else
			g_value_take_boxed(value, g_value_array_new(0));
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

	GSTLALLPFilter *element = GSTLAL_LPFILTER(object);

	if(element->fir_filter) {
		g_free(element->fir_filter);
		element->fir_filter = NULL;
	}
	G_OBJECT_CLASS(gstlal_lpfilter_parent_class)->finalize(object);
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


static void gstlal_lpfilter_class_init(GSTLALLPFilterClass *klass) {

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
		"Compute Linear-phase FIR filter",
		"Sink",
		"Compute and update a linear-phase FIR filter using an input stream of complex\n\t\t\t   "
		"numbers representing the amplitude and phase at a chosen measurement frequency.\n\t\t\t   "
		"This filter applies only a gain and a frequency-independent time delay/advance.",
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


	properties[ARG_MEASUREMENT_FREQUENCY] = g_param_spec_double(
		"measurement-frequency",
		"Measurement Frequency",
		"Frequency at which gain factor and timing offset are computed",
		-G_MAXDOUBLE, G_MAXDOUBLE, 0.0,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_UPDATE_SAMPLES] = g_param_spec_int64(
		"update-samples",
		"Update Samples",
		"Number of input samples after which to update the linear-phase FIR filter",
		0, G_MAXINT64, 320,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_AVERAGE_SAMPLES] = g_param_spec_int64(
		"average-samples",
		"Average Samples",
		"Number of input samples to average before producing FIR filter",
		0, G_MAXINT64, 320,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_WRITE_TO_SCREEN] = g_param_spec_boolean(
		"write-to-screen",
		"Write to Screen",
		"Set to True in order to write linear-phase FIR filters to\n\t\t\t"
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
	properties[ARG_FIR_LENGTH] = g_param_spec_int64(
		"fir-length",
		"FIR filter length",
		"Length in samples of FIR filters produced. Must be an even number.",
		1, G_MAXINT64, 16384,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_FIR_SAMPLE_RATE] = g_param_spec_int(
		"fir-sample-rate",
		"FIR sample rate",
		"The sample rate of the data being filtered",
		1, G_MAXINT, 16384,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_FIR_FILTER] = g_param_spec_value_array(
		"fir-filter",
		"FIR Filter",
		"The computed FIR filter",
		g_param_spec_double(
			"sample",
			"Sample",
			"A sample from the FIR filter",
			-G_MAXDOUBLE, G_MAXDOUBLE, 0.0,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		),
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_CONTROLLABLE
	);


	g_object_class_install_property(
		gobject_class,
		ARG_MEASUREMENT_FREQUENCY,
		properties[ARG_MEASUREMENT_FREQUENCY]
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
		ARG_FIR_LENGTH,
		properties[ARG_FIR_LENGTH]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_FIR_SAMPLE_RATE,
		properties[ARG_FIR_SAMPLE_RATE]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_FIR_FILTER,
		properties[ARG_FIR_FILTER]
	);
}


/*
 * init()
 */


static void gstlal_lpfilter_init(GSTLALLPFilter *element) {

	g_signal_connect(G_OBJECT(element), "notify::fir-filter", G_CALLBACK(rebuild_workspace_and_reset), NULL);
	element->rate = 0;
	element->unit_size = 0;

	gst_base_sink_set_sync(GST_BASE_SINK(element), FALSE);
	gst_base_sink_set_async_enabled(GST_BASE_SINK(element), FALSE);
}


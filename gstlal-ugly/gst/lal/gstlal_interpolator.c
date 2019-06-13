/*
 * An interpolator element
 *
 * Copyright (C) 2015  Chad Hanna, Kipp Cannon
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
 * ========================================================================
 *
 *                                  Preamble
 *
 * ========================================================================
 */


#include <glib.h>
#include <glib/gprintf.h>
#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <gst/base/gstadapter.h>
#include <math.h>
#include <string.h>
#include <fftw3.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

/*
 * our own stuff
 */

#include <gstlal/gstlal.h>
#include <gstlal/gstlal_debug.h>
#include <gstlal/gstaudioadapter.h>
#include <gstlal_interpolator.h>

#define PI 3.141592653589793

#define GST_CAT_DEFAULT gstlal_interpolator_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);

G_DEFINE_TYPE_WITH_CODE(
	GSTLALInterpolator,
	gstlal_interpolator,
	GST_TYPE_BASE_TRANSFORM,
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "lal_interpolator", 0, "lal_interpolator element")
);




static gsl_vector_float** upkernel32(int half_length_at_original_rate, int f) {

	/*
	 * This is a sinc windowed sinc function kernel
	 * The baseline kernel is defined as
	 *
	 * g[k] = sin(pi / f * (k-c)) / (pi / f * (k-c)) * (1 - (k-c)^2 / c / c)	k != c
	 * g[k] = 1									k = c
	 *
	 * Where:
	 *
	 * 	f: interpolation factor, must be power of 2, e.g., 2, 4, 8, ...
	 * 	c: defined as half the full kernel length
	 *
	 * You specify the half filter length at the original rate in samples,
	 * the kernel length is then given by:
	 *
	 *	kernel_length = half_length_at_original_rate * 2 * f + 1
	 *
	 * Interpolation is then defined as a two step process.  First the
	 * input data is zero filled to bring it up to the new sample rate,
	 * i.e., the input data, x, is transformed to x' such that:
	 *
	 * x'[i] = x[i/f]	if (i%f) == 0
	 *	 = 0		if (i%f) > 0
	 *
	 * y[i] = sum_{k=0}^{2c+1} x'[i-k] g[k]
	 *
	 * Since more than half the terms in this series would be zero, the
	 * convolution is implemented by breaking up the kernel into f separate
	 * kernels each 1/f as large as the originalcalled z, i.e.,:
	 *
	 * z[0][k/f] = g[k*f]
	 * z[1][k/f] = g[k*f+1]
	 * ...
	 * z[f-1][k/f] = g[k*f + f-1]
	 *
	 * Now the convolution can be written as:
	 *
	 * y[i] = sum_{k=0}^{2c/f+1} x[i/f] z[i%f][k]
	 *
	 * which avoids multiplying zeros.  Note also that by construction the
	 * sinc function has its zeros arranged such that z[0][:] had only one
	 * nonzero sample at its center. Therefore the actual convolution is:
	 *
	 * y[i] = x[i/f]					if i%f == 0
	 * y[i] = sum_{k=0}^{2c/f+1} x[i/f] z[i%f][k]		otherwise
	 *
	 */

	int kernel_length = 2 * half_length_at_original_rate * f + 1;
	int sub_kernel_length = 2 * half_length_at_original_rate + 1;

	/* the domain should be the kernel_length divided by two */
	int c = kernel_length / 2;

	gsl_vector_float **vecs = malloc(sizeof(*vecs) * f);
	for (int i = 0; i < f; i++)
		vecs[i] = gsl_vector_float_calloc(sub_kernel_length);

	float *out = fftwf_malloc(sizeof(*out) * kernel_length);
	memset(out, 0, kernel_length * sizeof(*out));


	for (int i = 0; i < kernel_length; i++) {
		int x = i - c;
		if (x == 0)
			out[i] = 1.;
		else
			out[i] = sin(PI * x / f) / (PI * x / f) * sin(PI * x / c) / (PI * x / c);
	}

	for (int j = 0; j < f; j++) {
		for (int i = 0; i < sub_kernel_length; i++) {
			int index = i * f + j;
			if (index < kernel_length)
				gsl_vector_float_set(vecs[j], sub_kernel_length - i - 1, out[index]);
		}
	}

	free(out);
	return vecs;
}


static gsl_vector** upkernel64(int half_length_at_original_rate, int f) {

	int kernel_length = 2 * half_length_at_original_rate * f + 1;
	int sub_kernel_length = 2 * half_length_at_original_rate + 1;

	/* the domain should be the kernel_length divided by two */
	int c = kernel_length / 2;

	gsl_vector **vecs = malloc(sizeof(*vecs) * f);
	for (int i = 0; i < f; i++)
		vecs[i] = gsl_vector_calloc(sub_kernel_length);

	double *out = fftw_malloc(sizeof(*out) * kernel_length);
	memset(out, 0, kernel_length * sizeof(*out));


	for (int i = 0; i < kernel_length; i++) {
		int x = i - c;
		if (x == 0)
			out[i] = 1.;
		else
			out[i] = sin(PI * x / f) / (PI * x / f) * sin(PI * x / c) / (PI * x / c);
	}

	for (int j = 0; j < f; j++) {
		for (int i = 0; i < sub_kernel_length; i++) {
			int index = i * f + j;
			if (index < kernel_length)
				gsl_vector_set(vecs[j], sub_kernel_length - i - 1, out[index]);
		}
	}

	free(out);
	return vecs;
}


static gsl_vector_float** downkernel32(int half_length_at_target_rate, int f) {

	/*
	 * This is a sinc windowed sinc function kernel
	 * The baseline kernel is defined as
	 *
	 * g[k] = sin(pi / f * (k-c)) / (pi / f * (k-c)) * (1 - (k-c)^2 / c / c)	k != c
	 * g[k] = 1									k = c
	 *
	 * Where:
	 *
	 * 	f: downsample factor, must be power of 2, e.g., 2, 4, 8, ...
	 * 	c: defined as half the full kernel length
	 *
	 * You specify the half filter length at the target rate in samples,
	 * the kernel length is then given by:
	 *
	 *	kernel_length = half_length_at_original_rate * 2 * f + 1
	 */

	int kernel_length = 2 * half_length_at_target_rate * f + 1;

	/* the domain should be the kernel_length divided by two */
	int c = kernel_length / 2;

	// There is only one kernel for downsampling, it is not interleaved,
	// instead the convolution is on a stride.
	gsl_vector_float **vecs = malloc(sizeof(*vecs));
	vecs[0] = gsl_vector_float_calloc(kernel_length);
	double val = 0.;
	double norm = 0.;
	for (int i = 0; i < kernel_length; i++) {
		int x = i - c;
		if (x == 0) {
			gsl_vector_float_set(vecs[0], i, 1.0);
			norm += 1.;
		}
		else {
			val = sin(PI * x / f) / (PI * x / f) * sin(PI * x / c) / (PI * x / c);
			norm += val * val;
			gsl_vector_float_set(vecs[0], i, val);
		}
	}

	for (int i = 0; i < kernel_length; i++)
		gsl_vector_float_set(vecs[0], i, gsl_vector_float_get(vecs[0], i) / sqrt(norm * f));
	return vecs;
}


static gsl_vector **downkernel64(int half_length_at_target_rate, int f) {

	int kernel_length = 2 * half_length_at_target_rate * f + 1;

	/* the domain should be the kernel_length divided by two */
	int c = kernel_length / 2;

	// There is only one kernel for downsampling, it is not interleaved,
	// instead the convolution is on a stride.
	gsl_vector **vecs = malloc(sizeof(*vecs));
	vecs[0] = gsl_vector_calloc(kernel_length);
	double val = 0.;
	double norm = 0.;
	for (int i = 0; i < kernel_length; i++) {
		int x = i - c;
		if (x == 0) {
			gsl_vector_set(vecs[0], i, 1.0);
			norm += 1.;
		}
		else {
			val = sin(PI * x / f) / (PI * x / f) * sin(PI * x / c) / (PI * x / c);
			norm += val * val;
			gsl_vector_set(vecs[0], i, val);
		}
	}

	for (int i = 0; i < kernel_length; i++)
		gsl_vector_set(vecs[0], i, gsl_vector_get(vecs[0], i) / sqrt(norm * f));
	return vecs;
}


static void convolve32(float *output, gsl_vector_float *thiskernel, float *input, guint kernel_length, guint channels) {

	/*
	 * This function will multiply a matrix of input values by vector to
	 * produce a vector of output values.  It is a single sample output of a
	 * convolution with "channels" number of channels
	 */

	gsl_vector_float_view output_vector = gsl_vector_float_view_array(output, channels);
	gsl_matrix_float_view input_matrix = gsl_matrix_float_view_array(input, kernel_length, channels);

	gsl_blas_sgemv (CblasTrans, 1.0, &(input_matrix.matrix), thiskernel, 0, &(output_vector.vector));
	return;
}

static void convolve64(double *output, gsl_vector *thiskernel, double *input, guint kernel_length, guint channels) {

	gsl_vector_view output_vector = gsl_vector_view_array(output, channels);
	gsl_matrix_view input_matrix = gsl_matrix_view_array(input, kernel_length, channels);

	gsl_blas_dgemv (CblasTrans, 1.0, &(input_matrix.matrix), thiskernel, 0, &(output_vector.vector));
	return;
}

static void upsample32(float *output, gsl_vector_float **thiskernel, float *input, guint kernel_length, guint factor, guint channels, guint blockstrideout, gboolean nongap) {

	/*
	 * This function is responsible for the resampling of the input time
	 * series.  It accomplishes the convolution by matrix multiplications
	 * on the input data sample-by-sample.  Gaps are skipped and the output
	 * is set to zero.  NOTE only gaps that are entirely zero in the input
	 * matrix will map to an output of zero.  Gaps smaller than that will
	 * still be convolved even though it is silly to do so.  The input
	 * stride is 32 samples though, so most gaps will be bigger than that.
	 *
	 */

	if (!nongap) {
		memset(output, 0, sizeof(*output) * blockstrideout * channels);
		return;
	}
	guint kernel_offset, output_offset, input_offset;
	for (guint samp = 0; samp < blockstrideout; samp++) {
		kernel_offset = samp % factor;
		output_offset = samp * channels;
		input_offset = samp / factor;
		input_offset *= channels;

		convolve32(output + output_offset, thiskernel[kernel_offset], input + input_offset, kernel_length, channels);
	}
	return;
}


static void upsample64(double *output, gsl_vector **thiskernel, double *input, guint kernel_length, guint factor, guint channels, guint blockstrideout, gboolean nongap) {

	if (!nongap) {
		memset(output, 0, sizeof(*output) * blockstrideout * channels);
		return;
	}
	guint kernel_offset, output_offset, input_offset;
	for (guint samp = 0; samp < blockstrideout; samp++) {
		kernel_offset = samp % factor;
		output_offset = samp * channels;
		input_offset = samp / factor;
		input_offset *= channels;

		convolve64(output + output_offset, thiskernel[kernel_offset], input + input_offset, kernel_length, channels);
	}
	return;
}


static void downsample32(float *output, gsl_vector_float **thiskernel, float *input, guint kernel_length, guint factor, guint channels, guint blockstrideout, gboolean nongap) {

	/*
	 * This function is responsible for the resampling of the input time
	 * series.  It accomplishes the convolution by matrix multiplications
	 * on the input data sample-by-sample.  Gaps are skipped and the output
	 * is set to zero.  NOTE only gaps that are entirely zero in the input
	 * matrix will map to an output of zero.  Gaps smaller than that will
	 * still be convolved even though it is silly to do so.  The input
	 * stride is 32 samples though, so most gaps will be bigger than that.
	 *
	 */

	if (!nongap) {
		memset(output, 0, sizeof(*output) * blockstrideout * channels);
		return;
	}
	guint output_offset, input_offset;
	for (guint samp = 0; samp < blockstrideout; samp++) {
		output_offset = samp * channels;
		/*
		 * NOTE that only every "factor" if input samples is convolved
		 * since the convolution of inbetween samples would be dropped
		 * in the output anyway.
		 */
		input_offset = samp * channels * factor;
		convolve32(output + output_offset, thiskernel[0], input + input_offset, kernel_length, channels);
	}
	return;
}


static void downsample64(double *output, gsl_vector **thiskernel, double *input, guint kernel_length, guint factor, guint channels, guint blockstrideout, gboolean nongap) {

	if (!nongap) {
		memset(output, 0, sizeof(*output) * blockstrideout * channels);
		return;
	}
	guint output_offset, input_offset;
	for (guint samp = 0; samp < blockstrideout; samp++) {
		output_offset = samp * channels;
		/*
		 * NOTE that only every "factor" if input samples is convolved
		 * since the convolution of inbetween samples would be dropped
		 * in the output anyway.
		 */
		input_offset = samp * channels * factor;
		convolve64(output + output_offset, thiskernel[0], input + input_offset, kernel_length, channels);
	}
	return;
}


/*
 * gstreamer boiler plate
 */


/*
 * Pads
 */


static GstStaticPadTemplate sink_template =
	GST_STATIC_PAD_TEMPLATE ("sink",
	GST_PAD_SINK,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS ("audio/x-raw, " \
		"format = (string) {" GST_AUDIO_NE(F32) ", " GST_AUDIO_NE(F64) "}, " \
		"rate =  (int) {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768}, " \
		"channels = " GST_AUDIO_CHANNELS_RANGE ", " \
		"layout = (string) interleaved, " \
		"channel-mask = (bitmask) 0")
	);


static GstStaticPadTemplate src_template =
	GST_STATIC_PAD_TEMPLATE ("src",
	GST_PAD_SRC,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS ("audio/x-raw, " \
		"format = (string) {" GST_AUDIO_NE(F32) ", " GST_AUDIO_NE(F64) "}, " \
		"rate =  (int) {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768}, " \
		"channels = " GST_AUDIO_CHANNELS_RANGE ", " \
		"layout = (string) interleaved, " \
		"channel-mask = (bitmask) 0")

	);


/*
 * Virtual method protototypes
 */


static void finalize(GObject *object);
static gboolean get_unit_size(GstBaseTransform *trans, GstCaps *caps, gsize *size);
static gboolean set_caps (GstBaseTransform * base, GstCaps * incaps, GstCaps * outcaps);
static GstFlowReturn transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf);
static GstCaps* transform_caps (GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, GstCaps *filter);
static gboolean transform_size(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, gsize size, GstCaps *othercaps, gsize *othersize);
static gboolean start(GstBaseTransform *trans);
static gboolean stop(GstBaseTransform *trans);


/*
 * class_init()
 */


static void gstlal_interpolator_class_init(GSTLALInterpolatorClass *klass)
{

	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(klass);

	gst_element_class_set_details_simple(element_class, "Interpolator", "Filter/Audio", "Interpolates multichannel audio data using BLAS", "Chad Hanna <chad.hanna@ligo.org>, Kipp Cannon <kipp.cannon@ligo.org>, Patrick Brockill <brockill@uwm.edu>, Alex Pace <alexander.pace@ligo.org>");

	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);
	transform_class->get_unit_size = GST_DEBUG_FUNCPTR(get_unit_size);
	transform_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
	transform_class->transform = GST_DEBUG_FUNCPTR(transform);
	transform_class->transform_caps = GST_DEBUG_FUNCPTR(transform_caps);
	transform_class->transform_size = GST_DEBUG_FUNCPTR(transform_size);
	transform_class->start = GST_DEBUG_FUNCPTR(start);
	transform_class->stop = GST_DEBUG_FUNCPTR(stop);

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_template));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_template));
}

static float kernel_length(GSTLALInterpolator *element) {
	/*
	 * The kernel length is specified relative to the input rate
	 */
	if (element->outrate > element->inrate)
		return element->half_length * 2 + 1;
	else
		return element->half_length * 2 * element->inrate / element->outrate + 1;
}

static void gstlal_interpolator_init(GSTLALInterpolator *element)
{
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(element), TRUE);

	/* internal data */
	element->inrate = 0;
	element->outrate = 0;

	element->kernel32 = NULL;
	element->kernel64 = NULL;
	element->workspace32 = NULL;
	element->workspace64 = NULL;

	/* hardcoded kernel size */
	element->half_length = 0;

	/* Always initialize with a discont */
	element->need_discont = TRUE;
	element->need_pretend = TRUE;
	element->last_gap_state = FALSE;

	element->adapter = g_object_new(GST_TYPE_AUDIOADAPTER, NULL);
}


static GstCaps* transform_caps (GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, GstCaps *filter) {

	/*
	 * All powers of two rates from 4 to 32768 Hz are allowed.  This
	 * element is designed to be efficient when using power of two rate
	 * ratios.
	 */

	gint channels;
	GstAudioInfo info;
	char capsstr[256] = {0};

	if (direction == GST_PAD_SINK && gst_caps_is_fixed(caps)) {
		gst_audio_info_from_caps(&info, caps);
		channels = GST_AUDIO_INFO_CHANNELS(&info);
		sprintf(capsstr, "audio/x-raw, format = (string) %s, rate = (int) {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768}, channels = (int) %d, layout=(string)interleaved, channel-mask=(bitmask)0", GST_AUDIO_INFO_NAME(&info), channels);
		
		return gst_caps_from_string(capsstr);
	}

	return gst_caps_from_string("audio/x-raw, format= (string) {" GST_AUDIO_NE(F32) ", " GST_AUDIO_NE(F64) "}, rate = (int) {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768}, channels = (int) [1, MAX]");
}


static gboolean set_caps (GstBaseTransform * base, GstCaps * incaps, GstCaps * outcaps) {
	GSTLALInterpolator *element = GSTLAL_INTERPOLATOR (base);
	GstStructure *instruct, *outstruct;
	gint inchannels, inrate, outchannels, outrate;
	gboolean success = gst_audio_info_from_caps(&element->audio_info, outcaps);

	instruct = gst_caps_get_structure (incaps, 0);
	outstruct = gst_caps_get_structure (outcaps, 0);


	g_return_val_if_fail(gst_structure_get_int (instruct, "channels", &inchannels), FALSE);
	g_return_val_if_fail(gst_structure_get_int (instruct, "rate", &inrate), FALSE);
	g_return_val_if_fail(gst_structure_get_int (outstruct, "channels", &outchannels), FALSE);
	g_return_val_if_fail(gst_structure_get_int (outstruct, "rate", &outrate), FALSE);

	GST_INFO_OBJECT(element, "in channels %d in rate %d out channels %d out rate %d", inchannels, inrate, outchannels, outrate);

	g_return_val_if_fail(inchannels == outchannels, FALSE);

	element->inrate = inrate;
	element->outrate = outrate;
	element->channels = inchannels;
	element->width = GST_AUDIO_INFO_WIDTH(&element->audio_info);

	get_unit_size(base, outcaps, &(element->unitsize));

	/* Timestamp and offset bookeeping */
	element->t0 = GST_CLOCK_TIME_NONE;
	element->offset0 = GST_BUFFER_OFFSET_NONE;
	element->next_input_offset = GST_BUFFER_OFFSET_NONE;
	element->next_output_offset = GST_BUFFER_OFFSET_NONE;
	element->need_discont = TRUE;
	element->need_pretend = TRUE;

	if (element->kernel32) {
		gsl_vector_float_free(element->kernel32[0]);
		for (int i = 1; i < element->outrate / element->inrate; i++)
			gsl_vector_float_free(element->kernel32[i]);
	}
	if (element->kernel64) {
		gsl_vector_free(element->kernel64[0]);
		for (int i = 1; i < element->outrate / element->inrate; i++)
			gsl_vector_free(element->kernel64[i]);
	}
	// Upsampling
	if (element->outrate > element->inrate) {
		/* hardcoded kernel size */
		element->half_length = 8;
		element->kernel32 = upkernel32(element->half_length, element->outrate / element->inrate);
		element->kernel64 = upkernel64(element->half_length, element->outrate / element->inrate);
	}
	// Downsampling
	else {
		/* hardcoded kernel size */
		element->half_length = 32;
		element->kernel32 = downkernel32(element->half_length, element->inrate / element->outrate);
		element->kernel64 = downkernel64(element->half_length, element->inrate / element->outrate);
	}
	/*
	 * Keep blockstride small to prevent GAPS from growing to be large
	 * FIXME probably this should be decoupled
	 */

	// Upsampling
	if (element->outrate > element->inrate) {
		element->blockstridein = 32;
		// You can produce blockstridein samples of output by having an
		// extra kernel length (minus 1) of input samples
		element->blocksampsin = element->blockstridein + kernel_length(element) - 1;
		// Upsampling produces an output that is the ratio of rates larger.
		element->blockstrideout = element->blockstridein * element->outrate / element->inrate;
		element->blocksampsout = element->blocksampsin * element->outrate / element->inrate;
	}
	// Downsampling
	else {
		element->blockstrideout = 32;
		// Downsampling requires the ratio of rates (in/out) of input to produce a given output
		element->blockstridein = element->blockstrideout * element->inrate / element->outrate;
		// We need to have kernel length -1 extra samples going in to hit a target input stride.
		element->blocksampsin = element->blockstridein + kernel_length(element) - 1;
		element->blocksampsout = element->blocksampsin * element->outrate / element->inrate;
	}
	GST_INFO_OBJECT(element, "blocksampsin %d, blocksampsout %d, blockstridein %d, blockstrideout %d unit size %d width %d", element->blocksampsin, element->blocksampsout, element->blockstridein, element->blockstrideout, (int) element->unitsize, element->width);

	if (element->workspace32)
		gsl_matrix_float_free(element->workspace32);
	element->workspace32 = gsl_matrix_float_calloc (element->blocksampsin, element->channels);
	if (element->workspace64)
		gsl_matrix_free(element->workspace64);
	element->workspace64 = gsl_matrix_calloc (element->blocksampsin, element->channels);

	g_object_set(element->adapter, "unit-size", element->unitsize, NULL);

	return success;
}


/*
 * ============================================================================
 *
 *                     GstBaseTransform Method Overrides
 *
 * ============================================================================
 */


/*
 * get_unit_size()
 */


static gboolean get_unit_size(GstBaseTransform *trans, GstCaps *caps, gsize *size)
{
	GstAudioInfo info;
	gboolean success = gst_audio_info_from_caps(&info, caps);


	if(success) {
		*size = GST_AUDIO_INFO_BPF(&info);
	}
	else
		GST_WARNING_OBJECT(trans, "unable to parse channels from %" GST_PTR_FORMAT, caps);

	return success;
}


static guint64 get_available_samples(GSTLALInterpolator *element)
{
	guint size;
	g_object_get(element->adapter, "size", &size, NULL);
	return size;
}


static guint minimum_input_length(GSTLALInterpolator *element, guint samps) {
	// Upsampling
	if (element->outrate >  element->inrate)
		return (guint) ceil( (float) samps * element->inrate / element->outrate) + kernel_length(element) - 1;
	// Downsampling
	else
		return (guint) ceil( (float) samps * element->outrate / element->inrate) + kernel_length(element) - 1;
}


static guint minimum_input_size(GSTLALInterpolator *element, guint size) {
	return minimum_input_length(element, size / element->unitsize) * element->unitsize;
}


static guint get_output_length(GSTLALInterpolator *element, guint samps) {

	/*
	 * The output length is either a multiple of the blockstride or 0 if
	 * there is not enough data.
	 *
	*/

	/* Pretend that we have a half_length set of samples if we are at a discont */
	guint pretend_samps = 0;
	if (element->need_pretend) {
		if (element->outrate > element->inrate)
			pretend_samps = element->half_length;
		else
			pretend_samps = element->half_length * element->inrate / element->outrate;
	}
	guint numinsamps = get_available_samples(element) + samps + pretend_samps;
	if (numinsamps < element->blocksampsin)
		return 0;
	// Note this could be zero
	guint numoutsamps = (numinsamps - kernel_length(element) - 1) * element->outrate / element->inrate;
	guint numblocks = numoutsamps / element->blockstrideout;

	/* NOTE could be zero */
	return numblocks * element->blockstrideout;
}


static guint get_output_size(GSTLALInterpolator *element, guint size) {
	return get_output_length(element, size / element->unitsize) * element->unitsize;
}


static gboolean transform_size(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, gsize size, GstCaps *othercaps, gsize *othersize)
{
	GSTLALInterpolator *element = GSTLAL_INTERPOLATOR(trans);
	gsize unit_size;
	gsize other_unit_size;
	gboolean success = TRUE;

	if(!get_unit_size(trans, caps, &unit_size))
		return FALSE;
	if(size % unit_size) {
		GST_ERROR_OBJECT(element, "size not a multiple of %" G_GSIZE_FORMAT, unit_size);
		return FALSE;
	}
	if(!get_unit_size(trans, othercaps, &other_unit_size))
		return FALSE;

	switch(direction) {
	case GST_PAD_SRC:

		/*
		 * number of input bytes required to produce an output
		 * buffer of (at least) the requested size
		 */

		*othersize = minimum_input_size(element, size);
		GST_INFO_OBJECT(element, "producing %" G_GSIZE_FORMAT " (bytes) buffer for request on SRC pad", *othersize);
		break;

	case GST_PAD_SINK:

		/*
		 * number of output bytes to be generated by the receipt of
		 * an input buffer of the given size.
		 */

		*othersize = get_output_size(element, size);
		GST_INFO_OBJECT(element, "SINK pad buffer of size %" G_GSIZE_FORMAT " (bytes) %" G_GSIZE_FORMAT " (samples) provided. Transforming to size %" G_GSIZE_FORMAT " (bytes) %" G_GSIZE_FORMAT " (samples).", size, size / element->unitsize,  *othersize, *othersize / element->unitsize);
		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		success = FALSE;
		break;
	}

	return success;
}


static gboolean start(GstBaseTransform *trans)
{
	GSTLALInterpolator *element = GSTLAL_INTERPOLATOR(trans);
	element->t0 = GST_CLOCK_TIME_NONE;
	element->offset0 = GST_BUFFER_OFFSET_NONE;
	element->next_input_offset = GST_BUFFER_OFFSET_NONE;
	element->next_output_offset = GST_BUFFER_OFFSET_NONE;
	element->need_discont = TRUE;
	element->need_pretend = TRUE;
	/* FIXME properly handle segments */
	return TRUE;
}


static gboolean stop(GstBaseTransform *trans)
{
	GSTLALInterpolator *element = GSTLAL_INTERPOLATOR(trans);
	g_object_unref(element->adapter);
	element->adapter = NULL;
	return TRUE;
}


static void flush_history(GSTLALInterpolator *element) {
	GST_INFO_OBJECT(element, "flushing adapter contents");
	gst_audioadapter_clear(element->adapter);
}


static void set_metadata(GSTLALInterpolator *element, GstBuffer *buf, guint64 outsamples, gboolean gap)
{
	GST_BUFFER_OFFSET(buf) = element->next_output_offset;
	element->next_output_offset += outsamples;
	GST_BUFFER_OFFSET_END(buf) = element->next_output_offset;
	GST_BUFFER_PTS(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET(buf) - element->offset0, GST_SECOND, element->outrate);
	GST_BUFFER_DURATION(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET_END(buf) - element->offset0, GST_SECOND, GST_AUDIO_INFO_RATE(&(element->audio_info))) - GST_BUFFER_PTS(buf);
	if(G_UNLIKELY(element->need_discont)) {
		GST_BUFFER_FLAG_SET(buf, GST_BUFFER_FLAG_DISCONT);
		element->need_discont = FALSE;
	}
	if(gap) {
		GST_BUFFER_FLAG_SET(buf, GST_BUFFER_FLAG_GAP);
		element->last_gap_state = TRUE;
	}
	else {
		GST_BUFFER_FLAG_UNSET(buf, GST_BUFFER_FLAG_GAP);
		element->last_gap_state = FALSE;
	}
	GST_INFO_OBJECT(element, "%s%s output_buffer %p spans %" GST_BUFFER_BOUNDARIES_FORMAT, gap ? "gap" : "nongap", GST_BUFFER_FLAG_IS_SET(buf, GST_BUFFER_FLAG_DISCONT) ? "+discont" : "", buf, GST_BUFFER_BOUNDARIES_ARGS(buf));
}


static GstFlowReturn transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf)
{
	GSTLALInterpolator *element = GSTLAL_INTERPOLATOR(trans);
	guint output_length;
	GstFlowReturn result = GST_FLOW_OK;
	gboolean copied_nongap;
	GstMapInfo mapinfo;

	g_assert(GST_BUFFER_PTS_IS_VALID(inbuf));
	g_assert(GST_BUFFER_DURATION_IS_VALID(inbuf));
	g_assert(GST_BUFFER_OFFSET_IS_VALID(inbuf));
	g_assert(GST_BUFFER_OFFSET_END_IS_VALID(inbuf));

	if(G_UNLIKELY(GST_BUFFER_IS_DISCONT(inbuf) || GST_BUFFER_OFFSET(inbuf) != element->next_input_offset || !GST_CLOCK_TIME_IS_VALID(element->t0))) {

		/*
		 * flush any previous history and clear the adapter
		 */

		flush_history(element);

		/*
		 * (re)sync timestamp and offset book-keeping
		 */

		element->t0 = GST_BUFFER_PTS(inbuf);
		element->offset0 = GST_BUFFER_OFFSET(inbuf);
		element->next_output_offset = element->offset0;

		/*
		 * be sure to flag the next output buffer as a discontinuity
		 */

		element->need_discont = TRUE;
		element->need_pretend = TRUE;
		/* FIXME-- clean up this debug statement */
		/* GST_INFO_OBJECT(element, "A discontinuity was detected. The offset has been reset to %" G_GUINT64_FORMAT " and the timestamp has been reset to %" GST_TIME_SECONDS_FORMAT, element->offset0, element->t0); */

	}
	else {
		g_assert_cmpuint(GST_BUFFER_PTS(inbuf), ==, gst_audioadapter_expected_timestamp(element->adapter));
	}


	element->next_input_offset = GST_BUFFER_OFFSET_END(inbuf);

	GST_INFO_OBJECT(element, "%s input_buffer %p spans %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_DISCONT) ? "+discont" : "", inbuf, GST_BUFFER_BOUNDARIES_ARGS(inbuf));

	/* FIXME- clean up this debug statement */
	GST_INFO_OBJECT(element, "pushing %d (bytes) %d (samples) sample buffer into adapter with size %d (samples) offset %d offset end %d: unitsize %d adapter->unit-size %d", (int) gst_buffer_get_size(inbuf), (int) gst_buffer_get_size(inbuf) / (int) element->unitsize, (int) get_available_samples(element), (int) GST_BUFFER_OFFSET(inbuf), (int) GST_BUFFER_OFFSET_END(inbuf), (int) element->unitsize, (int) element->adapter->unit_size);

	gst_buffer_ref(inbuf);  /* don't let calling code free buffer */
	gst_audioadapter_push(element->adapter, inbuf);
	GST_INFO_OBJECT(element, "adapter_size %u (samples)", (guint) get_available_samples(element));

	/*
	 * Handle the different possiblilities
	 */

	output_length = gst_buffer_get_size(outbuf) / element->unitsize;

	if (gst_buffer_get_size(outbuf) == 0)
		set_metadata(element, outbuf, 0, element->last_gap_state);
	else {

		if (element->width == 32) {
			guint processed = 0;
			gst_buffer_map(outbuf, &mapinfo, GST_MAP_WRITE);
			float *output = (float *) mapinfo.data;
			memset(mapinfo.data, 0, mapinfo.size);
			/* FIXME- clean up this print statement (format) */
			/* GST_INFO_OBJECT(element, "Processing a %d sample output buffer from %d input", output_length); */

			while (processed < output_length) {


				/* Special case to handle discontinuities: effectively
				 * zero pad. FIXME make this more elegant
				 */

				if (element->need_pretend) {
					memset(element->workspace32->data, 0, sizeof(*element->workspace32->data) * element->workspace32->size1 * element->workspace32->size2);
					if (element->outrate > element->inrate)
						gst_audioadapter_copy_samples(element->adapter, element->workspace32->data + (element->half_length) * element->channels, element->blocksampsin - element->half_length, NULL, &copied_nongap);
					else
						gst_audioadapter_copy_samples(element->adapter, element->workspace32->data + (element->half_length * element->inrate / element->outrate) * element->channels, element->blocksampsin - element->half_length * element->inrate / element->outrate, NULL, &copied_nongap);
				}
				else
					gst_audioadapter_copy_samples(element->adapter, element->workspace32->data, element->blocksampsin, NULL, &copied_nongap);

				if (element->outrate > element->inrate)
					upsample32(output, element->kernel32, element->workspace32->data, kernel_length(element), element->outrate / element->inrate, element->channels, element->blockstrideout, copied_nongap);
				else
					downsample32(output, element->kernel32, element->workspace32->data, kernel_length(element), element->inrate / element->outrate, element->channels, element->blockstrideout, copied_nongap);
				if (element->need_pretend) {
					element->need_pretend = FALSE;
					if (element->outrate > element->inrate)
						gst_audioadapter_flush_samples(element->adapter, element->blockstridein - element->half_length);
					else
						gst_audioadapter_flush_samples(element->adapter, element->blockstridein - element->half_length * element->inrate / element->outrate);
				}
				else {
					GST_INFO_OBJECT(element, "Flushing %d samples : processed %d samples : output length %d samples", element->blockstridein, processed, output_length);
					gst_audioadapter_flush_samples(element->adapter, element->blockstridein);
				}
				output += element->blockstrideout * element->channels;
				processed += element->blockstrideout;
			}
			GST_INFO_OBJECT(element, "Processed a %d samples", processed);
			set_metadata(element, outbuf, output_length, !copied_nongap);
			gst_buffer_unmap(outbuf, &mapinfo);
		}
		if (element->width == 64) {
			guint processed = 0;
			gst_buffer_map(outbuf, &mapinfo, GST_MAP_WRITE);
			double *output = (double *) mapinfo.data;
			memset(mapinfo.data, 0, mapinfo.size);
			/* FIXME- clean up this print statement (format) */
			/* GST_INFO_OBJECT(element, "Processing a %d sample output buffer from %d input", output_length); */

			while (processed < output_length) {


				/* Special case to handle discontinuities: effectively
				 * zero pad. FIXME make this more elegant
				 */

				if (element->need_pretend) {
					memset(element->workspace64->data, 0, sizeof(*element->workspace64->data) * element->workspace64->size1 * element->workspace64->size2);
					if (element->outrate > element->inrate)
						gst_audioadapter_copy_samples(element->adapter, element->workspace64->data + (element->half_length) * element->channels, element->blocksampsin - element->half_length, NULL, &copied_nongap);
					else
						gst_audioadapter_copy_samples(element->adapter, element->workspace64->data + (element->half_length * element->inrate / element->outrate) * element->channels, element->blocksampsin - element->half_length * element->inrate / element->outrate, NULL, &copied_nongap);
				}
				else
					gst_audioadapter_copy_samples(element->adapter, element->workspace64->data, element->blocksampsin, NULL, &copied_nongap);

				if (element->outrate > element->inrate)
					upsample64(output, element->kernel64, element->workspace64->data, kernel_length(element), element->outrate / element->inrate, element->channels, element->blockstrideout, copied_nongap);
				else
					downsample64(output, element->kernel64, element->workspace64->data, kernel_length(element), element->inrate / element->outrate, element->channels, element->blockstrideout, copied_nongap);
				if (element->need_pretend) {
					element->need_pretend = FALSE;
					if (element->outrate > element->inrate)
						gst_audioadapter_flush_samples(element->adapter, element->blockstridein - element->half_length);
					else
						gst_audioadapter_flush_samples(element->adapter, element->blockstridein - element->half_length * element->inrate / element->outrate);
				}
				else {
					GST_INFO_OBJECT(element, "Flushing %d samples : processed %d samples : output length %d samples", element->blockstridein, processed, output_length);
					gst_audioadapter_flush_samples(element->adapter, element->blockstridein);
				}
				output += element->blockstrideout * element->channels;
				processed += element->blockstrideout;
			}
			GST_INFO_OBJECT(element, "Processed a %d samples", processed);
			set_metadata(element, outbuf, output_length, !copied_nongap);
			gst_buffer_unmap(outbuf, &mapinfo);
		}
	}
	return result;
}


static void finalize(GObject *object)
{
	GSTLALInterpolator *element = GSTLAL_INTERPOLATOR(object);

	/*
	 * free resources
	 */

	gsl_vector_float_free(element->kernel32[0]);
	for (int i = 1; i < element->outrate / element->inrate; i++)
		gsl_vector_float_free(element->kernel32[i]);
	gsl_vector_free(element->kernel64[0]);
	for (int i = 1; i < element->outrate / element->inrate; i++)
		gsl_vector_free(element->kernel64[i]);
	gsl_matrix_float_free(element->workspace32);
	gsl_matrix_free(element->workspace64);

	/*
	 * chain to parent class' finalize() method
	 */

	G_OBJECT_CLASS(gstlal_interpolator_parent_class)->finalize(object);
}

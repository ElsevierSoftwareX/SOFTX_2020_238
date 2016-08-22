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


/*
 * struff from the C library
 */


/*
 * stuff from glib/gstreamer
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


gsl_vector_float** kernel(int half_length_at_original_rate, int f) {

	/*
	 * This is a parabolic windowed sinc function kernel
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

	gsl_vector_float **vecs = malloc(sizeof(gsl_vector_float *) * f);
	for (int i = 0; i < f; i++)
		vecs[i] = gsl_vector_float_calloc(sub_kernel_length);

	float *out = fftwf_malloc(sizeof(float) * kernel_length);
	memset(out, 0, kernel_length * sizeof(float));


	for (int i = 0; i < kernel_length; i++) {
		int x = i - c;
		if (x == 0)
			out[i] = 1.;
		else
			out[i] = sin(PI * x / f) / (PI * x / f) * (1. - (float) x*x / c / c);
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


void convolve(float *output, gsl_vector_float *thiskernel, float *input, guint kernel_length, guint channels) {

	/* 
	 * This function will multiply a matrix of input values by vector to
 	 * produce a vector of output values.  It is a single sample output of a
	 * convolution with channels number of channels
	 */

	gsl_vector_float_view output_vector = gsl_vector_float_view_array(output, channels);
	gsl_matrix_float_view input_matrix = gsl_matrix_float_view_array(input, kernel_length, channels);
	gsl_blas_sgemv (CblasTrans, 1.0, &(input_matrix.matrix), thiskernel, 0, &(output_vector.vector));
	return;
}

void copy_input(float *output, gsl_vector_float *thiskernel, float *input, guint kernel_length, guint channels) {

	/*
	 * For a special set of input samples a convolution is not necessary.
	 * These are input samples that are exactly divisble by the upsample
	 * factor. For these we can save computation and simply copy the input
	 * samples to the output.
	 *
	*/
	gsl_vector_float_view output_vector = gsl_vector_float_view_array(output, channels);
	gsl_vector_float_view input_vector = gsl_vector_float_view_array(input + kernel_length / 2, channels);
	gsl_blas_scopy(&(input_vector.vector), &(output_vector.vector));
	return;
}

void resample(float *output, gsl_vector_float **thiskernel, float *input, guint kernel_length, guint factor, guint channels, guint blockstrideout, gboolean nongap) {
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
		memset(output, 0, sizeof(float) * blockstrideout * channels);
		return;
	}
	guint kernel_offset, output_offset, input_offset;
	for (guint samp = 0; samp < blockstrideout; samp++) {
		kernel_offset = samp % factor;
		output_offset = samp * channels;
		input_offset = samp / factor * channels;
		/*
		 * The first kernel is a delta function by definition, so just
 		 * copy the input 
		 */
		// AEP- disable the copy conditional, always perform the convolution.
		/*if (kernel_offset == 0)
			copy_input(output + output_offset, thiskernel[kernel_offset], input + input_offset, kernel_length, channels);
		else*/
			convolve(output + output_offset, thiskernel[kernel_offset], input + input_offset, kernel_length, channels);
	}
	return;
}

/*
 * gstreamer boiler plate
 */

#define GST_CAT_DEFAULT gstlal_interpolator_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);

G_DEFINE_TYPE_WITH_CODE(
        GSTLALInterpolator,
        gstlal_interpolator,
        GST_TYPE_BASE_TRANSFORM,
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "lal_interpolator", 0, "lal_interpolator element")
);


// FIXME- This is defined but never used? Commented out to squash warnings. 
//static void gstlal_interpolator_base_init(gpointer klass){}

/* Pads */

static GstStaticPadTemplate sink_template =
	GST_STATIC_PAD_TEMPLATE ("sink",
	GST_PAD_SINK,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS ("audio/x-raw-float, "
		"endianness = (int) BYTE_ORDER, "
		"width = (int) 32, "
		"rate = (int) {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768}, "
		"channels = (int) [1, MAX]")
	);

static GstStaticPadTemplate src_template =
	GST_STATIC_PAD_TEMPLATE ("src",
	GST_PAD_SRC,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS ("audio/x-raw-float, "
		"endianness = (int) BYTE_ORDER, "
		"width = (int) 32, "
		"rate = (int) {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768}, "
		"channels = (int) [1, MAX]")
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

        gst_element_class_set_details_simple(element_class, "Interpolator", "Filter/Audio", "Interpolates multichannel audio data using FFTs", "Chad Hanna <chad.hanna@ligo.org>, Patrick Brockill <brockill@uwm.edu>");

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


static void gstlal_interpolator_init(GSTLALInterpolator *element)
{
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(element), TRUE);

	/* internal data */
	element->inrate = 0;
	element->outrate = 0;

	element->factor = 0; // size of complex output to FFT
	element->kernel = NULL;
	element->workspace = NULL;

	// hardcoded kernel size
	element->half_length = 16;
	element->kernel_length = element->half_length * 2 + 1;

	// Always initialize with a discont
	element->need_discont = TRUE;
	element->need_pretend = TRUE;

	element->adapter = g_object_new(GST_TYPE_AUDIOADAPTER, NULL);
}

static GstCaps* transform_caps (GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, GstCaps *filter) {

	/* 
         * FIXME actually pull out the allowed rates so that we can prevent
	 *  downsampling at the negotiation stage
	 */
	GstStructure *capsstruct;
	gint channels;
	capsstruct = gst_caps_get_structure (caps, 0);
	char capsstr[256] = {0};

	if (direction == GST_PAD_SINK && gst_structure_get_int (capsstruct, "channels", &channels)) {
		sprintf(capsstr, "audio/x-raw-float, endianness = (int) BYTE_ORDER, width = (int) 32, rate = (int) {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768}, channels = (int) %d", channels);
		return gst_caps_from_string(capsstr);
	}

	return gst_caps_from_string("audio/x-raw-float, endianness = (int) BYTE_ORDER, width = (int) 32, rate = (int) {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768}, channels = (int) [1, MAX]");

}

static gboolean set_caps (GstBaseTransform * base, GstCaps * incaps, GstCaps * outcaps) {
	GSTLALInterpolator *element = GSTLAL_INTERPOLATOR (base);
	GstStructure *instruct, *outstruct;
	gint inchannels, inrate, outchannels, outrate;

	instruct = gst_caps_get_structure (incaps, 0);
	outstruct = gst_caps_get_structure (outcaps, 0);
	g_return_val_if_fail(gst_structure_get_int (instruct, "channels", &inchannels), FALSE);
	g_return_val_if_fail(gst_structure_get_int (instruct, "rate", &inrate), FALSE);
	g_return_val_if_fail(gst_structure_get_int (outstruct, "channels", &outchannels), FALSE);
	g_return_val_if_fail(gst_structure_get_int (outstruct, "rate", &outrate), FALSE);

	g_return_val_if_fail(inchannels == outchannels, FALSE);
	g_return_val_if_fail(outrate >= inrate, FALSE);
	g_return_val_if_fail(outrate % inrate == 0, FALSE);

	element->inrate = inrate;
	element->outrate = outrate;
	element->channels = inchannels;
	element->factor = outrate / inrate;

	/* Timestamp and offset bookeeping */

	element->t0 = GST_CLOCK_TIME_NONE;
        element->offset0 = GST_BUFFER_OFFSET_NONE;
        element->next_input_offset = GST_BUFFER_OFFSET_NONE;
        element->next_output_offset = GST_BUFFER_OFFSET_NONE;
	element->need_discont = TRUE;
	element->need_pretend = TRUE;

	if (element->kernel)
		free(element->kernel);
	element->kernel = kernel(element->half_length, element->factor);

	// Keep blockstride small to prevent GAPS from growing to be large
	// FIXME probably this should be decoupled 
	element->blockstridein = 32;//element->inrate;
	element->blocksampsin = element->blockstridein + element->kernel_length;
	element->blockstrideout = element->blockstridein * element->factor;//element->outrate;
	element->blocksampsout = element->blockstrideout + (element->kernel_length) * element->factor;

	GST_INFO_OBJECT(element, "blocksampsin %d, blocksampsout %d, blockstridein %d, blockstrideout %d", element->blocksampsin, element->blocksampsout, element->blockstridein, element->blockstrideout);

	if (element->workspace)
		gsl_matrix_float_free(element->workspace);
	element->workspace = gsl_matrix_float_calloc (element->blocksampsin, element->channels);

	return TRUE;
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
	GSTLALInterpolator *element = GSTLAL_INTERPOLATOR(trans);
	GstStructure *str;
	gint width, channels;
	gboolean success = TRUE;

	str = gst_caps_get_structure(caps, 0);
	success &= gst_structure_get_int(str, "channels", &channels);
	success &= gst_structure_get_int(str, "width", &width);
		

	if(success) {
		*size = width / 8 * channels;
		element->unitsize = *size;
		g_object_set(element->adapter, "unit-size", *size, NULL);
		GST_INFO_OBJECT(element, "channels %d, width %d", channels, width);
	}
	else
		GST_WARNING_OBJECT(trans, "unable to parse channels from %" GST_PTR_FORMAT, caps);

	return success;
}

static guint64 get_available_samples(GSTLALInterpolator *element)
{
	//FIXME is size here really samples, I guess so??
	guint size;
	g_object_get(element->adapter, "size", &size, NULL);
	return size;
}


static guint minimum_input_length(GSTLALInterpolator *element, guint samps) {
	return samps / element->factor + element->kernel_length; // FIXME check this
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

	// Pretend that we have a half_length set of samples if we are at a discont
	guint pretend_samps = element->need_pretend ? element->half_length : 0;
	guint numinsamps = get_available_samples(element) + samps + pretend_samps;
	if (numinsamps <= element->kernel_length)
		return 0;
	guint numoutsamps = (numinsamps - element->kernel_length) * element->factor; // FIXME check this
	guint numblocks = numoutsamps / element->blockstrideout; //truncation

	return numblocks * element->blockstrideout; // Could be zero
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
		//GST_INFO_OBJECT(element, "producing %d (bytes) buffer for request on SRC pad", *othersize);
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
	// FIXME properly handle segments
	// element->need_new_segment = TRUE;
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
	GST_BUFFER_DURATION(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET_END(buf) - element->offset0, GST_SECOND, element->outrate) - GST_BUFFER_PTS(buf);
	if(G_UNLIKELY(element->need_discont)) {
		GST_BUFFER_FLAG_SET(buf, GST_BUFFER_FLAG_DISCONT);
		element->need_discont = FALSE;
	}
	if(gap)
		GST_BUFFER_FLAG_SET(buf, GST_BUFFER_FLAG_GAP);
	else
		GST_BUFFER_FLAG_UNSET(buf, GST_BUFFER_FLAG_GAP);
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
		//FIXME-- clean up this print statement
		//GST_INFO_OBJECT(element, "A discontinuity was detected. The offset has been reset to %" G_GUINT64_FORMAT " and the timestamp has been reset to %" GST_TIME_SECONDS_FORMAT, element->offset0, element->t0);

	}
	else {
		g_assert_cmpuint(GST_BUFFER_PTS(inbuf), ==, gst_audioadapter_expected_timestamp(element->adapter));
	}


	element->next_input_offset = GST_BUFFER_OFFSET_END(inbuf);

	GST_INFO_OBJECT(element, "%s input_buffer %p spans %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_DISCONT) ? "+discont" : "", inbuf, GST_BUFFER_BOUNDARIES_ARGS(inbuf));

	//FIXME- clean up this print statement
	//GST_INFO_OBJECT(element, "pushing %d (bytes) %d (samples) sample buffer into adapter with size %d (samples)", GST_BUFFER_SIZE(inbuf), GST_BUFFER_SIZE(inbuf) / element->unitsize, get_available_samples(element));

	gst_buffer_ref(inbuf);  /* don't let calling code free buffer */
	gst_audioadapter_push(element->adapter, inbuf);
	GST_INFO_OBJECT(element, "adapter_size %u (samples)", (guint) get_available_samples(element));
	
	// FIXME check the sanity of the output buffer

	/*
	 * Handle the different possiblilities
	 */

	output_length = gst_buffer_get_size(outbuf) / element->unitsize;

	if (gst_buffer_get_size(outbuf) == 0)
		set_metadata(element, outbuf, 0, FALSE);
	else {


		guint processed = 0;
		//float *output = (float *) GST_BUFFER_DATA(outbuf);
		gst_buffer_map(outbuf, &mapinfo, GST_MAP_WRITE);
		float *output = (float *) outbuf;
		//memset(GST_BUFFER_DATA(outbuf), 0, GST_BUFFER_SIZE(outbuf));  // FIXME necesary?
		// FIXME- clean up this print statement (format)
		//GST_INFO_OBJECT(element, "Processing a %d sample output buffer from %d input", output_length);

		while (processed < output_length) {


			// Special case to handle discontinuities: effectively zero pad. FIXME make this more elegant
			if (element->need_pretend) {
				memset(element->workspace->data, 0, sizeof(float) * element->workspace->size1 * element->workspace->size2); // FIXME necessary?
				gst_audioadapter_copy_samples(element->adapter, element->workspace->data + (element->half_length) * element->channels, element->blocksampsin - element->half_length, NULL, &copied_nongap);
			}
			else
				gst_audioadapter_copy_samples(element->adapter, element->workspace->data, element->blocksampsin, NULL, &copied_nongap);

			resample(output, element->kernel, element->workspace->data, element->kernel_length, element->factor, element->channels, element->blockstrideout, copied_nongap);

			if (element->need_pretend) {
				element->need_pretend = FALSE;
				gst_audioadapter_flush_samples(element->adapter, element->blockstridein - element->half_length);
			}
			else
				gst_audioadapter_flush_samples(element->adapter, element->blockstridein);
			output += element->blockstrideout * element->channels;
			processed += element->blockstrideout;
		}
		GST_INFO_OBJECT(element, "Processed a %d samples", processed);
		set_metadata(element, outbuf, output_length, !copied_nongap);

	}
	return result;
}


static void finalize(GObject *object)
{
        GSTLALInterpolator *element = GSTLAL_INTERPOLATOR(object);

        /*
         * free resources
         */

	for (guint i = 0; i < element->factor; i++)	
		gsl_vector_float_free(element->kernel[i]);
	gsl_matrix_float_free(element->workspace);

        /*
         * chain to parent class' finalize() method
         */

        G_OBJECT_CLASS(gstlal_interpolator_parent_class)->finalize(object);
}


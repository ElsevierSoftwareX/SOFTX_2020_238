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

/*
 * our own stuff
 */

#include <gstlal/gstlal.h>
#include <gstlal/gstlal_debug.h>
#include <gstlal/gstaudioadapter.h>
#include <gstlal_interpolator.h>

/*
 * Utility functions for tapering the FFT buffers
 */

#define PI 3.141592653589793

static float* kernel(int half_length, int factor) {
	int kernel_length = (2 * half_length + 1) * factor;
	int domain = kernel_length / 2; // kernel length is gauranteed to be even
	float *out = fftwf_alloc_real(kernel_length + 1);
	float norm = 0.;

	for (int j = 0; j < kernel_length + 1; j++) {
		float x = j - domain;
		if (j == domain)
			out[j] = 1.;
		else
			out[j] = sin(PI * x / factor) / (PI * x / factor) * (1. - x*x / domain / domain);
	}

	for (int i = 0; i < kernel_length + 1; i++)
		norm += out[i] * out[i];

	for (int i = 0; i < kernel_length + 1; i++)
		out[i] /= sqrt(norm / factor);

	return out;
}

void convolve(float *output, float *thiskernel, float *input, guint kernel_length, guint factor, guint channels) {
	for (guint i = 1; i < kernel_length; i++) {
		*output += (*thiskernel) * (*input);
		input += channels;
		thiskernel += factor;
	}
	return;
}

void resample(float *output, float *thiskernel, float *input, guint kernel_length, guint factor, guint channels, guint blockstrideout) {
	guint kernel_offset, output_offset, input_offset;
	for (gint samp = 0; samp < blockstrideout; samp++) {
		kernel_offset = factor - samp % factor + factor / 2;
		output_offset = samp * channels;
		input_offset = (samp / factor) * channels; // first input sample
		for (gint chan = 0; chan < channels; chan++)
			convolve(output + output_offset + chan, thiskernel + kernel_offset, input + input_offset + chan + channels, kernel_length, factor, channels);
	}
	return;
}

/*
 * gstreamer boiler plate
 */

#define GST_CAT_DEFAULT gstlal_interpolator_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);

static void additional_initializations(GType type)
{
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "lal_interpolator", 0, "lal_interpolator element");
}

GST_BOILERPLATE_FULL(
        GSTLALInterpolator,
        gstlal_interpolator,
        GstBaseTransform,
        GST_TYPE_BASE_TRANSFORM,
        additional_initializations
);

static void gstlal_interpolator_base_init(gpointer klass){}

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
static gboolean get_unit_size(GstBaseTransform *trans, GstCaps *caps, guint *size);
static gboolean set_caps (GstBaseTransform * base, GstCaps * incaps, GstCaps * outcaps);
static GstFlowReturn transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf);
static GstCaps* transform_caps (GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps);
static gboolean transform_size(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, guint size, GstCaps *othercaps, guint *othersize);
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


static void gstlal_interpolator_init(GSTLALInterpolator *element, GSTLALInterpolatorClass *klass)
{
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(element), TRUE);

	/* internal data */
	element->inrate = 0;
	element->outrate = 0;

	element->factor = 0; // size of complex output to FFT
	element->kernel = NULL;
	element->workspace = NULL;

	// hardcoded kernel size
	element->half_length = 8;
	element->kernel_length = element->half_length * 2 + 1;

	// Always initialize with a discont
	element->need_discont = TRUE;
	element->need_pretend = TRUE;

	element->adapter = g_object_new(GST_TYPE_AUDIOADAPTER, NULL);
}

static GstCaps* transform_caps (GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps) {

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

	// Assume that we process inrate worth of samples at a time (e.g. 1s stride)
	element->blockstridein = 128;//element->inrate;
	element->blocksampsin = element->blockstridein + element->kernel_length;
	element->blockstrideout = element->blockstridein * element->factor;//element->outrate;
	element->blocksampsout = element->blockstrideout + (element->kernel_length) * element->factor;

	GST_INFO_OBJECT(element, "blocksampsin %d, blocksampsout %d, blockstridein %d, blockstrideout %d", element->blocksampsin, element->blocksampsout, element->blockstridein, element->blockstrideout);

	if (element->workspace)
		free(element->workspace);
	element->workspace = (float *) fftw_alloc_real(element->blocksampsin * element->channels);
	memset(element->workspace, 0, sizeof(float) * element->blocksampsin * element->channels);

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


static gboolean get_unit_size(GstBaseTransform *trans, GstCaps *caps, guint *size)
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

static gboolean transform_size(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, guint size, GstCaps *othercaps, guint *othersize)
{
	GSTLALInterpolator *element = GSTLAL_INTERPOLATOR(trans);
	guint unit_size;
	guint other_unit_size;
	gboolean success = TRUE;

	if(!get_unit_size(trans, caps, &unit_size))
		return FALSE;
	if(size % unit_size) {
		GST_ERROR_OBJECT(element, "size not a multiple of %u", unit_size);
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
		GST_INFO_OBJECT(element, "producing %d (bytes) buffer for request on SRC pad", *othersize);
		break;

	case GST_PAD_SINK:
		/*
		 * number of output bytes to be generated by the receipt of
		 * an input buffer of the given size.
		 */

		*othersize = get_output_size(element, size);
		GST_INFO_OBJECT(element, "SINK pad buffer of size %d (bytes) %d (samples) provided. Transforming to size %d (bytes) %d (samples).", size, size / element->unitsize,  *othersize, *othersize / element->unitsize);
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
	GST_BUFFER_TIMESTAMP(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET(buf) - element->offset0, GST_SECOND, element->outrate);
	GST_BUFFER_DURATION(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET_END(buf) - element->offset0, GST_SECOND, element->outrate) - GST_BUFFER_TIMESTAMP(buf);
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
	guint input_length, output_length, expected_output_size;
	GstFlowReturn result = GST_FLOW_OK;

	g_assert(GST_BUFFER_TIMESTAMP_IS_VALID(inbuf));
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

		element->t0 = GST_BUFFER_TIMESTAMP(inbuf);
		element->offset0 = GST_BUFFER_OFFSET(inbuf);
		element->next_output_offset = element->offset0;

		/*
		 * be sure to flag the next output buffer as a discontinuity
		 */

		element->need_discont = TRUE;
		element->need_pretend = TRUE;
		GST_INFO_OBJECT(element, "A discontinuity was detected. The offset has been reset to %" G_GUINT64_FORMAT " and the timestamp has been reset to %" GST_TIME_SECONDS_FORMAT, element->offset0, element->t0);

	}
	else {
		g_assert_cmpuint(GST_BUFFER_TIMESTAMP(inbuf), ==, gst_audioadapter_expected_timestamp(element->adapter));
	}


	element->next_input_offset = GST_BUFFER_OFFSET_END(inbuf);

	GST_INFO_OBJECT(element, "%s input_buffer %p spans %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_DISCONT) ? "+discont" : "", inbuf, GST_BUFFER_BOUNDARIES_ARGS(inbuf));

	GST_INFO_OBJECT(element, "pushing %d (bytes) %d (samples) sample buffer into adapter with size %d (samples)", GST_BUFFER_SIZE(inbuf), GST_BUFFER_SIZE(inbuf) / element->unitsize, get_available_samples(element));

	gst_buffer_ref(inbuf);  /* don't let calling code free buffer */
	gst_audioadapter_push(element->adapter, inbuf);
	GST_INFO_OBJECT(element, "adapter_size %d (samples)", get_available_samples(element));
	
	// FIXME check the sanity of the output buffer

	/*
	 * Handle the different possiblilities
	 */

	output_length = GST_BUFFER_SIZE(outbuf) / element->unitsize;

	if (GST_BUFFER_SIZE(outbuf) == 0)
		set_metadata(element, outbuf, 0, FALSE);
	else {


		guint flushed = 0;
		guint processed = 0;
		gint input_offset;
		gint kernel_offset;
		gint output_offset;
		gint f;
		gint i;
		float *output = (float *) GST_BUFFER_DATA(outbuf);
		memset(GST_BUFFER_DATA(outbuf), 0, GST_BUFFER_SIZE(outbuf));  // FIXME necesary?

		GST_INFO_OBJECT(element, "Processing a %d sample output buffer from %d input", output_length);

		while (processed < output_length) {

			memset(element->workspace, 0, sizeof(float) * element->blocksampsin * element->channels); // FIXME necessary?

			// Special case to handle discontinuities: effectively zero pad. FIXME make this more elegant
			if (element->need_pretend)
				gst_audioadapter_copy_samples(element->adapter, element->workspace + (element->half_length) * element->channels, element->blocksampsin - element->half_length, NULL, NULL);
			else
				gst_audioadapter_copy_samples(element->adapter, element->workspace, element->blocksampsin, NULL, NULL);

			resample(output, element->kernel, element->workspace, element->kernel_length, element->factor, element->channels, element->blockstrideout);

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
		set_metadata(element, outbuf, output_length, FALSE);

	}
	return result;
}


static void finalize(GObject *object)
{
        GSTLALInterpolator *element = GSTLAL_INTERPOLATOR(object);

        /*
         * free resources
         */
	
	free(element->kernel);
	free(element->workspace);

        /*
         * chain to parent class' finalize() method
         */

        G_OBJECT_CLASS(parent_class)->finalize(object);
}


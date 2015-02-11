/*
 * An interpolator element
 *
 * Copyright (C) 2011  Chad Hanna, Kipp Cannon
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

static float* taperup(int samps) {
	float x;
	float *out = (float *) calloc(samps, sizeof(float));
	for (int i = 0; i < samps; i++) {
		x = cos(PI / 2. * (float) i / samps);
		out[i] = 1. - x * x;
	}
	return out;
}

static float* taperdown(int samps) {
	float x;
	float *out = (float *) calloc(samps, sizeof(float));
	for (int i = 0; i < samps; i++) {
		x = cos(PI / 2. * (float) i / samps);
		out[i] = x * x;
	}
	return out;
}

static int applytaper(float *in, int end, float *taper) {
	for (int i = 0; i < end; i++) {
		in[i] *= taper[i];
	}
	return 0;
}

static int blend(float *in1, float *in2, int start, int end) {
	for (int i = start; i < end; i++)
		in1[i] += in2[i];
	return 0;
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

        gst_element_class_set_details_simple(element_class, "Interpolator", "Filter/Audio", "Interpolates multichannel audio data using FFTs", "Chad Hanna <chad.hanna@ligo.org>");

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

	element->nrin = 0; // size of real input to FFT
	element->ncin = 0; // size of complex input to FFFT
	element->nrout = 0; // size of real output to FFT
	element->ncout = 0; // size of complex output to FFT
	element->tapersampsin = 0;
	element->tapersampsout = 0;
	element->up = NULL;
	element->down = NULL;
	element->last = NULL;
	element->rin = NULL;
	element->cin = NULL;
	element->rout = NULL;
	element->cout = NULL;
	element->data = NULL;

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

	/* Timestamp and offset bookeeping */

	element->t0 = GST_CLOCK_TIME_NONE;
        element->offset0 = GST_BUFFER_OFFSET_NONE;
        element->next_input_offset = GST_BUFFER_OFFSET_NONE;
        element->next_output_offset = GST_BUFFER_OFFSET_NONE;
	element->need_discont = TRUE;

	/* 
	 * NOTE: This element hardcodes transform sizes to be 1 second so we
 	 * can set up a lot of stuff right now. Might as well.
	 */

	element->nrin = inrate;
	element->ncin = element->nrin / 2 + 1; // FFTW documentation for complex size
	element->nrout = outrate;
	element->ncout = element->nrout / 2 + 1; // FFTW documentation for complex size

	element->tapersampsin = element->nrin / 4;
	element->tapersampsout = element->nrout / 4;
	element->blocksampsin = element->nrin - element->tapersampsin;
	element->blocksampsout = element->nrout - element->tapersampsout;

	if (element->up)
		free(element->up);
	element->up = taperup(element->tapersampsin);

	if (element->down)
		free(element->down);
	element->down = taperdown(element->tapersampsin);

	if (element->last)
		free(element->last);
	element->last = (float *) fftw_alloc_real(element->tapersampsout * element->channels);
	memset(element->last, 0, sizeof(float) * element->tapersampsout * element->channels);
		
	if (!element->data)
		free(element->data);
	element->data = (float*) fftwf_alloc_real(element->nrin * element->channels);

	if (element->rin)
		fftwf_free(element->rin);
	element->rin = (float*) fftwf_alloc_real(element->nrin);

	if (element->cin)
		fftwf_free(element->cin);
	element->cin = (fftwf_complex*) fftwf_alloc_complex(element->ncin);

	if (element->rout)
		fftwf_free(element->rout);
	element->rout = (float*) fftwf_alloc_real(element->nrout);

	if (element->cout)
		fftwf_free(element->cout);
	element->cout = (fftwf_complex*) fftwf_alloc_complex(element->ncout);

	gstlal_fftw_lock();
	element->fwdplan_in = fftwf_plan_dft_r2c_1d(element->nrin, element->rin, element->cin, FFTW_PATIENT);
	element->revplan_out = fftwf_plan_dft_c2r_1d(element->nrout, element->cout, element->rout, FFTW_PATIENT);
	gstlal_fftw_unlock();

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
		GST_INFO_OBJECT(element, "get_unit_size(): channels %d, width %d", channels, width);
	}
	else
		GST_WARNING_OBJECT(trans, "unable to parse channels from %" GST_PTR_FORMAT, caps);

	return success;
}

static guint minimum_input_length(GSTLALInterpolator *element, guint samps) {
	return ceil(samps / element->blocksampsin) * element->blocksampsin + element->tapersampsin;
}

static guint minimum_input_size(GSTLALInterpolator *element, guint size) {
	return minimum_input_length(element, size / element->unitsize) * element->unitsize;
}

static guint64 get_available_samples(GSTLALInterpolator *element)
{
	//FIXME is size here really samples, I guess so??
	guint size;
	g_object_get(element->adapter, "size", &size, NULL);
	return size;
}

static guint get_output_length(GSTLALInterpolator *element, guint samps) {
	guint remainder;
	guint numinsamps = get_available_samples(element) + samps;
	if (numinsamps == 0)
		return 0;
	guint numoutsamps = numinsamps * element->outrate / element->inrate;
	guint numblocks = numoutsamps / element->blocksampsout; //truncation
	if (numblocks != 0)
		remainder = numoutsamps % numblocks; // what doesn't fit into a multiple of block sizes
	else
		remainder = numoutsamps;
	if ((remainder < element->tapersampsout) && (numblocks > 0)) // we can't actually produce output for the last tapersampsout worth of samples since those will be blended
		numblocks -= 1;
	return numblocks * element->blocksampsout; // Could be zero
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
		GST_INFO_OBJECT(element, "transform_size() producing %d (bytes) buffer for request on src pad", *othersize);
		break;

	case GST_PAD_SINK:
		/*
		 * number of output bytes to be generated by the receipt of
		 * an input buffer of the given size.
		 */

		*othersize = get_output_size(element, size);
		GST_INFO_OBJECT(element, "transform_size() SINK pad buffer of size %d (bytes) %d (samples) provided. Transforming to size %d (bytes) %d (samples).", size, size / element->unitsize,  *othersize, *othersize / element->unitsize);
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
	//element->need_new_segment = TRUE;
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
	GST_INFO_OBJECT(element, "set_metadata() %s%s output buffer %p spans %" GST_BUFFER_BOUNDARIES_FORMAT, gap ? "gap" : "nongap", GST_BUFFER_FLAG_IS_SET(buf, GST_BUFFER_FLAG_DISCONT) ? "+discont" : "", buf, GST_BUFFER_BOUNDARIES_ARGS(buf));
}


static GstFlowReturn transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf)
{
// FIXME, finish this. It is just (partially) copied from firbank
	GSTLALInterpolator *element = GSTLAL_INTERPOLATOR(trans);
	guint expected_output_length, expected_output_size;
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
		element->next_output_offset = element->offset0 + get_output_length(element, GST_BUFFER_SIZE(inbuf) / element->unitsize);

		/*
		 * be sure to flag the next output buffer as a discontinuity
		 */

		element->need_discont = TRUE;
		GST_INFO_OBJECT(element, "transform() A discontinuity was detected. The offset has been reset to %" G_GUINT64_FORMAT " and the timestamp has been reset to %" GST_TIME_SECONDS_FORMAT, element->offset0, element->t0);

	} else
		g_assert_cmpuint(GST_BUFFER_TIMESTAMP(inbuf), ==, gst_audioadapter_expected_timestamp(element->adapter));

	element->next_input_offset = GST_BUFFER_OFFSET_END(inbuf);

	/*
	 * Put the input buffer into the adapter first
	 * Then check the output buffer size that was expected.
	 * Note that the transform size function tells you what you can produce
	 * *after* receiving the next input buffer so this order is important
	 */

	GST_INFO_OBJECT(element, "transform() pushing %d (bytes) %d (samples) sample buffer into adapter with size %d (samples)", GST_BUFFER_SIZE(inbuf), GST_BUFFER_SIZE(inbuf) / element->unitsize, get_available_samples(element));

	gst_buffer_ref(inbuf);  /* don't let calling code free buffer */
	gst_audioadapter_push(element->adapter, inbuf);
	GST_INFO_OBJECT(element, "transform() adapter size is now %d (samples)", get_available_samples(element));

	expected_output_length = get_output_length(element, 0); // just check the output length based on what we have in the adapter already
	expected_output_size = get_output_size(element, 0); // Ditto here
	GST_INFO_OBJECT(element, "transform() expect an output buffer with size %d (%d samples): got one with size %d", expected_output_length, expected_output_size, GST_BUFFER_SIZE(outbuf));
	g_assert_cmpuint(expected_output_size, ==, GST_BUFFER_SIZE(outbuf));


	/*
	 * Handle the different possiblilities
	 */

	if (GST_BUFFER_SIZE(outbuf) == 0)
		set_metadata(element, outbuf, 0, FALSE);
	else {
		guint flushed = 0;
		guint processed = 0;
		float *last = NULL;
		float *output = (float *) GST_BUFFER_DATA(outbuf);

		// FIXME actually handle gaps properly
		while ((get_available_samples(element) >= element->nrin) && (processed < expected_output_length)) {

			/* First copy the data we need */
			gst_audioadapter_copy_samples(element->adapter, element->data, element->nrin, NULL, NULL);

			/* Resample each channel */
			for (guint i = 0; i < element->channels; i++) {

				last = element->last + i * element->tapersampsout; // get a pointer to the history for this channel

				/* Adapt output for FFTW FIXME FIXME FIXME we
				 * need to avoid this copying around by making use of the advanced FFTW
				 * interface
				 */
				for (guint j = 0; j < element->nrin; j++) {
					element->rin[j] = element->data[i+j*element->channels];
				}
				/* Clear the output */
				memset(element->cout, 0, sizeof(fftwf_complex) * element->ncout);

				/* taper */
				applytaper(element->rin, element->tapersampsin, element->up);
				applytaper(element->rin + element->nrin - element->tapersampsin, element->tapersampsin, element->down);

				/* resample */
				fftwf_execute(element->fwdplan_in);
				memcpy(element->cout, element->cin, sizeof(fftwf_complex) * element->ncin);
				fftwf_execute(element->revplan_out);

				/* Blend the outputs */
				blend(element->rout, last, 0, element->tapersampsout);
				memcpy(last, element->rout + (element->nrout - element->tapersampsout), sizeof(*last) * element->tapersampsout);

				/* Adapt output for FFTW FIXME FIXME FIXME we
				 * need to avoid this copying around by making use of the advanced FFTW
				 * interface
				 */
				for (guint j = 0; j < element->nrout-element->tapersampsout; j++) {
					output[i+j*element->channels] = element->rout[j] / element->nrin;
				}
			}

			/* Then flush the data we will never need again */
			// FIXME add a check that we have processed the correct number of samples
			gst_audioadapter_flush_samples(element->adapter, element->nrin - element->tapersampsin);
			flushed += element->nrin - element->tapersampsin;
			processed += element->nrout - element->tapersampsout;
			// Dont forget to increase the output pointer
			output = (float *) GST_BUFFER_DATA(outbuf) + processed * element->channels;
		}
		GST_INFO_OBJECT(element, "flushed %d processed %d expected output length %d", flushed, processed, expected_output_length);
		set_metadata(element, outbuf, expected_output_length, FALSE);

	}
	return result;
}


static void finalize(GObject *object)
{
        GSTLALInterpolator *element = GSTLAL_INTERPOLATOR(object);

        /*
         * free resources
         */
	
	free(element->up);
	free(element->down);
	free(element->last);
	fftwf_free(element->rin);
	fftwf_free(element->cin);
	fftwf_free(element->rout);
	fftwf_free(element->cout);
	fftwf_free(element->data);

        /*
         * chain to parent class' finalize() method
         */

        G_OBJECT_CLASS(parent_class)->finalize(object);
}


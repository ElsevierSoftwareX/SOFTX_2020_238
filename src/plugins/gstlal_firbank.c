/*
 * Copyright (C) 2009 Kipp Cannon <kipp.cannon@ligo.org>
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * Alternatively, the contents of this file may be used under the GNU
 * Lesser General Public License Version 2.1 (the "LGPL"), in which case
 * the following provisions apply instead of the ones mentioned above:
 *
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Library General Public License as published
 * by the Free Software Foundation; either version 2 of the License, or (at
 * your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307,
 * USA.
 */


/*
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


/*
 * stuff from the C library
 */


#include <complex.h>
#include <string.h>


/*
 *  stuff from gobject/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstadapter.h>
#include <gst/base/gstbasetransform.h>
#include "gstlal.h"
#include "gstlal_firbank.h"


/*
 * stuff from fftw
 */


#include <fftw3.h>


/*
 * stuff from GSL
 */


#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>


/*
 * ============================================================================
 *
 *                                 Utilities
 *
 * ============================================================================
 */


/*
 * return the number of FIR vectors
 */


static int fir_channels(const GSTLALFIRBank *element)
{
	return element->fir_matrix->size1;
}


/*
 * return the number of samples in the FIR vectors
 */


static int fir_samples(const GSTLALFIRBank *element)
{
	return element->fir_matrix->size2;
}


/*
 * construct a buffer of zeros and push into adapter
 */


static int push_zeros(GSTLALFIRBank *element, int samples)
{
	GstBuffer *zerobuf = gst_buffer_new_and_alloc(samples * fir_channels(element) * sizeof(double));
	if(!zerobuf) {
		GST_DEBUG_OBJECT(element, "failure allocating zero-pad buffer");
		return -1;
	}
	memset(GST_BUFFER_DATA(zerobuf), 0, GST_BUFFER_SIZE(zerobuf));
	gst_adapter_push(element->adapter, zerobuf);
	element->zeros_in_adapter += samples;
	return 0;
}


/*
 * set the metadata on an output buffer
 */


static void set_metadata(GSTLALFIRBank *element, GstBuffer *buf, guint64 outsamples)
{
	GST_BUFFER_SIZE(buf) = outsamples * fir_channels(element) * sizeof(double);
	GST_BUFFER_OFFSET(buf) = element->next_out_offset;
	element->next_out_offset += outsamples;
	GST_BUFFER_OFFSET_END(buf) = element->next_out_offset;
	GST_BUFFER_TIMESTAMP(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET(buf) - element->offset0, GST_SECOND, element->rate);
	GST_BUFFER_DURATION(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET_END(buf) - element->offset0, GST_SECOND, element->rate) - GST_BUFFER_TIMESTAMP(buf);
	if(element->need_discont) {
		GST_BUFFER_FLAG_SET(buf, GST_BUFFER_FLAG_DISCONT);
		element->need_discont = FALSE;
	}
}


/*
 * construct frequency-domain filters from time-domain
 */


static gsl_matrix_complex *make_frequency_domain_filters()
{
	return NULL;
}


/*
 * ============================================================================
 *
 *                           GStreamer Boiler Plate
 *
 * ============================================================================
 */


static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE(
	"sink",
	GST_PAD_SINK,
	GST_PAD_ALWAYS,
/* FIXME:  BYTEORDER */
	GST_STATIC_CAPS(
		"audio/x-raw-float, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) 1, " \
		"endianness = (int) 1234, " \
		"width = (int) 64"
	)
);


static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE(
	"src",
	GST_PAD_SRC,
	GST_PAD_ALWAYS,
/* FIXME:  BYTEORDER */
	GST_STATIC_CAPS(
		"audio/x-raw-float, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) [1, MAX], " \
		"endianness = (int) 1234, " \
		"width = (int) 64"
	)
);


GST_BOILERPLATE(
	GSTLALFIRBank,
	gstlal_firbank,
	GstBaseTransform,
	GST_TYPE_BASE_TRANSFORM
);


enum property {
	ARG_BLOCK_LENGTH_FACTOR = 1,
	ARG_FIR_MATRIX
};


#define DEFAULT_BLOCK_LENGTH_FACTOR 4


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
	GstStructure *str;
	gint channels;

	str = gst_caps_get_structure(caps, 0);
	if(!gst_structure_get_int(str, "channels", &channels)) {
		GST_DEBUG_OBJECT(trans, "unable to parse channels from %" GST_PTR_FORMAT, caps);
		return FALSE;
	}

	*size = sizeof(double) * channels;

	return TRUE;
}


/*
 * transform_caps()
 */


static GstCaps *transform_caps(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps)
{
	guint channels = fir_channels(GSTLAL_FIRBANK(trans));
	guint n;

	caps = gst_caps_make_writable(caps);

	switch(direction) {
	case GST_PAD_SRC:
		for(n = 0; n < gst_caps_get_size(caps); n++)
			gst_structure_set(gst_caps_get_structure(caps, n), "channels", G_TYPE_INT, 1, NULL);
		break;

	case GST_PAD_SINK:
		for(n = 0; n < gst_caps_get_size(caps); n++) {
			/* non-zero == fir matrix size is known */
			if(channels)
				gst_structure_set(gst_caps_get_structure(caps, n), "channels", G_TYPE_INT, channels, NULL);
			else
				gst_structure_set(gst_caps_get_structure(caps, n), "channels", GST_TYPE_INT_RANGE, 0, G_MAXINT, NULL);
		}
		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		gst_caps_unref(caps);
		return NULL;
	}

	return caps;
}


/*
 * set_caps()
 */


static gboolean set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps)
{
	GSTLALFIRBank *element = GSTLAL_FIRBANK(trans);
	GstStructure *s;
	gint rate;
	gint channels;

	s = gst_caps_get_structure(outcaps, 0);
	if(!gst_structure_get_int(s, "channels", &channels)) {
		GST_DEBUG_OBJECT(element, "unable to parse channels from %" GST_PTR_FORMAT, incaps);
		return FALSE;
	}
	if(!gst_structure_get_int(s, "rate", &rate)) {
		GST_DEBUG_OBJECT(element, "unable to parse rate from %" GST_PTR_FORMAT, incaps);
		return FALSE;
	}

	if(element->fir_matrix && (channels != fir_channels(element))) {
		GST_DEBUG_OBJECT(element, "channel != %d in %" GST_PTR_FORMAT, fir_channels(element), outcaps);
		return FALSE;
	}

	element->rate = rate;

	return TRUE;
}


/*
 * start()
 */


static gboolean start(GstBaseTransform *trans)
{
	GSTLALFIRBank *element = GSTLAL_FIRBANK(trans);
	element->adapter = gst_adapter_new();
	element->zeros_in_adapter = 0;
	element->t0 = GST_CLOCK_TIME_NONE;
	element->offset0 = GST_BUFFER_OFFSET_NONE;
	element->next_in_offset = GST_BUFFER_OFFSET_NONE;
	element->next_out_offset = GST_BUFFER_OFFSET_NONE;
	element->need_discont = TRUE;
	return TRUE;
}


/*
 * stop()
 */


static gboolean stop(GstBaseTransform *trans)
{
	GSTLALFIRBank *element = GSTLAL_FIRBANK(trans);
	g_object_unref(element->adapter);
	element->adapter = NULL;
	return TRUE;
}


/*
 * transform()
 */

static guint64 get_available_samples(GSTLALFIRBank *element)
{
	return gst_adapter_available(element->adapter) / (fir_channels(element) * sizeof(double));
}


static GstFlowReturn transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf)
{
	gint length;
	GSTLALFIRBank *element = GSTLAL_FIRBANK(trans);
	GstFlowReturn result;

	/*
	 * wait for FIR matrix
	 */

	g_mutex_lock(element->fir_matrix_lock);
	while(!element->fir_matrix) {
		g_cond_wait(element->fir_matrix_available, element->fir_matrix_lock);
		/* FIXME:  add a way to get out of this loop */
	}

	/*
	 * check for discontinuity
	 *
	 * FIXME:  flush/pad adapter as needed
	 */

	if(GST_BUFFER_OFFSET(inbuf) != element->next_in_offset || !GST_CLOCK_TIME_IS_VALID(element->t0)) {
		/*
		 * flush adapter, pad with zeros
		 */

		gst_adapter_clear(element->adapter);
		element->zeros_in_adapter = 0;
		push_zeros(element, (fir_samples(element) - 1) / 2);

		/*
		 * (re)sync timestamp and offset book-keeping
		 */

		element->t0 = GST_BUFFER_TIMESTAMP(inbuf);
		element->offset0 = GST_BUFFER_OFFSET(inbuf);
		element->next_out_offset = element->offset0;

		/*
		 * be sure to flag the next output buffer as a discontinuity
		 */

		element->need_discont = TRUE;
	}
	element->next_in_offset = GST_BUFFER_OFFSET_END(inbuf);

	/*
	 * gap logic
	 */

	length = GST_BUFFER_OFFSET_END(inbuf) - GST_BUFFER_OFFSET(inbuf);
	if(!GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP)) {
		gst_buffer_ref(inbuf);	/* don't let the adapter free it */
		gst_adapter_push(element->adapter, inbuf);
		element->zeros_in_adapter = 0;
		/*result = chisquared(outbuf, element);*/
	} else if(element->zeros_in_adapter >= fir_samples(element) - 1) {
		/* base transform has given us an output buffer that is the same size
		 * as the input buffer, which is the size we need now.  all we have to
		 * do is make it a gap */
		GST_BUFFER_FLAG_SET(outbuf, GST_BUFFER_FLAG_GAP);
		memset(GST_BUFFER_DATA(outbuf), 0, GST_BUFFER_SIZE(outbuf));
		set_metadata(element, outbuf, GST_BUFFER_OFFSET_END(inbuf) - GST_BUFFER_OFFSET(inbuf));
		result = GST_FLOW_OK;
	} else if(length <= fir_samples(element) - 1) {
		/* push length zeroes in the adapter and run normal \chi^{2} code */
		push_zeros(element, length);
		/*result = chisquared(outbuf, element);*/
	} else {
		GstPad *srcpad = GST_BASE_TRANSFORM_SRC_PAD(GST_BASE_TRANSFORM(element));
		guint64 available_samples = get_available_samples(element);
		GstBuffer *buf;

		/* Push length_A-1 zeroes into adapter  */
		push_zeros(element, fir_samples(element) - 1);
		length -= fir_samples(element) - 1;

		/* run normal \chi^{2} code to finish off adapter's contents, and
		 * manually push buffer downstream.  note:  we only get here if
		 * element->adpater_empty is FALSE, so we know there is at least one
		 * sample in the adapter before the zeroes we just pushed in */
		result = gst_pad_alloc_buffer(srcpad, element->next_out_offset, available_samples * fir_channels(element) * sizeof(complex double), GST_PAD_CAPS(srcpad), &buf);
		if(result != GST_FLOW_OK)
			return result;
		/*result = chisquared(buf, element);*/
		g_assert(result == GST_FLOW_OK);
		result = gst_pad_push(srcpad, buf);
		if(result != GST_FLOW_OK)
			return result;

		/* make outbuf a gap of whose size matches the remainder of the input
		 * gap */
		GST_BUFFER_FLAG_SET(outbuf, GST_BUFFER_FLAG_GAP);
		GST_BUFFER_SIZE(outbuf) = length * fir_channels(element) * sizeof(complex double);
		memset(GST_BUFFER_DATA(outbuf), 0, GST_BUFFER_SIZE(outbuf));
		set_metadata(element, outbuf, length);
	}

	g_mutex_unlock(element->fir_matrix_lock);
	return result;
}


/*
 * ============================================================================
 *
 *                          GObject Method Overrides
 *
 * ============================================================================
 */


/*
 * set_property()
 */


static void set_property(GObject *object, enum property prop_id, const GValue *value, GParamSpec *pspec)
{
	GSTLALFIRBank *element = GSTLAL_FIRBANK(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_BLOCK_LENGTH_FACTOR:
		element->block_length_factor = g_value_get_int(value);
		break;

	case ARG_FIR_MATRIX: {
		GValueArray *va = g_value_get_boxed(value);
		g_mutex_lock(element->fir_matrix_lock);
		if(element->fir_matrix) {
			gsl_matrix_free(element->fir_matrix);
			element->fir_matrix = NULL;
		}
		element->fir_matrix = gstlal_gsl_matrix_from_g_value_array(va);
		g_cond_signal(element->fir_matrix_available);
		g_mutex_unlock(element->fir_matrix_lock);
		break;
	}

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * get_property()
 */


static void get_property(GObject *object, enum property prop_id, GValue *value, GParamSpec *pspec)
{
	GSTLALFIRBank *element = GSTLAL_FIRBANK(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_BLOCK_LENGTH_FACTOR:
		g_value_set_int(value, element->block_length_factor);
		break;

	case ARG_FIR_MATRIX:
		g_mutex_lock(element->fir_matrix_lock);
		g_value_take_boxed(value, gstlal_g_value_array_from_gsl_matrix(element->fir_matrix));
		g_mutex_unlock(element->fir_matrix_lock);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * finalize()
 */


static void finalize(GObject *object)
{
	GSTLALFIRBank *element = GSTLAL_FIRBANK(object);

	g_mutex_free(element->fir_matrix_lock);
	element->fir_matrix_lock = NULL;
	g_cond_free(element->fir_matrix_available);
	element->fir_matrix_available = NULL;
	if(element->fir_matrix) {
		gsl_matrix_free(element->fir_matrix);
		element->fir_matrix = NULL;
	}
	if(element->fir_matrix_fd) {
		gsl_matrix_complex_free(element->fir_matrix_fd);
		element->fir_matrix_fd = NULL;
	}

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


/*
 * base_init()
 */


static void gstlal_firbank_base_init(gpointer gclass)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(gclass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(gclass);

	gst_element_class_set_details_simple(element_class, "FIR Filter Bank", "Filter/Audio", "Projects a single audio channel onto a bank of FIR filters to produce a multi-channel output", "Kipp Cannon <kipp.cannon@ligo.org>");

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));

	transform_class->get_unit_size = get_unit_size;
	transform_class->set_caps = set_caps;
	transform_class->transform = transform;
	transform_class->transform_caps = transform_caps;
	transform_class->start = start;
	transform_class->stop = stop;
}


/*
 * class_init()
 */


static void gstlal_firbank_class_init(GSTLALFIRBankClass *klass)
{
	GObjectClass *gobject_class;
	GstBaseTransformClass *base_transform_class;

	gobject_class = (GObjectClass *) klass;
	base_transform_class = (GstBaseTransformClass *) klass;

	gobject_class->set_property = set_property;
	gobject_class->get_property = get_property;
	gobject_class->finalize = finalize;

	g_object_class_install_property(
		gobject_class,
		ARG_BLOCK_LENGTH_FACTOR,
		g_param_spec_int(
			"block-length-factor",
			"Convolution block size in multiples of the FIR size",
			"When using FFT convolutions, use this many times the number of samples in each FIR vector for the convolution block size.",
			1, G_MAXINT, DEFAULT_BLOCK_LENGTH_FACTOR,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)
		);
	g_object_class_install_property(
		gobject_class,
		ARG_FIR_MATRIX,
		g_param_spec_value_array(
			"fir-matrix",
			"FIR Matrix",
			"Array of impulse response vectors.  Number of filters (rows) in matrix sets number of output channels.  All filters must have the same length.",
			g_param_spec_value_array(
				"response",
				"Impulse Response",
				"Array of coefficients.",
				g_param_spec_double(
					"coefficient",
					"Coefficient",
					"Impulse response coefficient",
					-G_MAXDOUBLE, G_MAXDOUBLE, 1.0,
					G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
				),
				G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
			),
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);
}


/*
 * init()
 */


static void gstlal_firbank_init(GSTLALFIRBank *filter, GSTLALFIRBankClass *kclass)
{
	filter->block_length_factor = DEFAULT_BLOCK_LENGTH_FACTOR;
	filter->adapter = NULL;
	filter->fir_matrix_lock = g_mutex_new();
	filter->fir_matrix_available = g_cond_new();
	filter->fir_matrix = NULL;
	filter->fir_matrix_fd = NULL;
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(filter), TRUE);
}

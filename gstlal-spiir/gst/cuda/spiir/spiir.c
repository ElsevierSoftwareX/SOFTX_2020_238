/*
 * Copyright (C) 2012-2013 Qi Chu <chicsheep@gmail.com>
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


#include <gstlal/gstlal.h>
#include <spiir/spiir.h>

/*
 * stuff from FFTW and GSL
 */


#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

#include <time.h>

/*
 * functions from cuda file
 */

#include <spiir/spiir_kernel.h>

/*
 * =================================================================
 *
 *			GStreamer Boiler Plate
 *
 * =================================================================
 */

#define GST_CAT_DEFAULT cuda_iirbank_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);

G_DEFINE_TYPE_WITH_CODE(
    GSTLALIIRBankCuda,
    cuda_iirbank,
    GST_TYPE_BASE_TRANSFORM,
    GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "cuda_iirbank", 0, "cuda_iirbank element")
    );

/*
 * ============================================================================
 *
 *                                 Utilities
 *
 * ============================================================================
 */


/*
 * construct a buffer of zeros and push into adapter
 */


static int push_zeros(GSTLALIIRBankCuda *element, unsigned samples)
{
	GstBuffer *zerobuf = gst_buffer_new_allocate(NULL, samples * (element->width / 8), NULL); 
	if(!zerobuf) {
		GST_DEBUG_OBJECT(element, "failure allocating zero-pad buffer");
		return -1;
	}
	gst_buffer_memset(zerobuf, 0, 0, samples * (element->width / 8));
	
	gst_audioadapter_push(element->adapter, zerobuf);
	element->zeros_in_adapter += samples;
	return 0;
}


/*
 * ============================================================================
 *
 *                                  Signals
 *
 * ============================================================================
 */


enum cuda_iirbank_signal {
	SIGNAL_RATE_CHANGED,
	NUM_SIGNALS
};


static guint signals[NUM_SIGNALS] = {0, };


static void rate_changed(GstElement *element, gint rate, void *data)
{
	/* FIXME: send updated segment downstream?  because latency now
	 * means something different */
}


/*
 * ============================================================================
 *
 *                           GStreamer Boiler Plate
 *
 * ============================================================================
 */


static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE(
        GST_BASE_TRANSFORM_SINK_NAME,
	GST_PAD_SINK,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		"audio/x-raw, " \
		"rate = " GST_AUDIO_RATE_RANGE ", " \
		"channels = (int) 1, " \
		"format = (string) {" GST_AUDIO_NE(F32) ", " GST_AUDIO_NE(F64) "}, " \
		"layout = (string) interleaved, " \
		"channel-mask = (bitmask) 0"
	)
);


static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SRC_NAME,
	GST_PAD_SRC,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
	  GST_AUDIO_CAPS_MAKE("{" GST_AUDIO_NE(F32) ", " GST_AUDIO_NE(F64) "}") ", " \
		"layout = (string) interleaved, " \
		"channel-mask = (bitmask) 0"
	)
);


enum property {
	ARG_IIR_A1 = 1,
	ARG_IIR_B0,
	ARG_IIR_DELAY
};


#define DEFAULT_LATENCY 0


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
	gint width, channels;
	gboolean success = TRUE;

	str = gst_caps_get_structure(caps, 0);
	success &= gst_structure_get_int(str, "channels", &channels);
	success &= gst_structure_get_int(str, "width", &width);

	if(success)
		*size = width / 8 * channels;
	else
		GST_WARNING_OBJECT(trans, "unable to parse channels from %" GST_PTR_FORMAT, caps);

	return success;
}


/*
 * transform_caps()
 */


static GstCaps *transform_caps(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps)
{
	GSTLALIIRBankCuda *element = CUDA_IIRBANK(trans);
	guint n;

	caps = gst_caps_copy(caps);

	switch(direction) {
	case GST_PAD_SRC:
		/*
		 * sink pad's format is the same as the source pad's except
		 * it must have only 1 channel
		 */

		for(n = 0; n < gst_caps_get_size(caps); n++)
			gst_structure_set(gst_caps_get_structure(caps, n), "channels", G_TYPE_INT, 1, NULL);
		break;

	case GST_PAD_SINK:
		/*
		 * source pad's format is the same as the sink pad's except
		 * it can have any number of channels or, if the size of
		 * the IIR matrix is known, the number of channels must
		 * equal to twice the number of template IIR outputs.
		 */


		g_mutex_lock(&element->iir_matrix_lock);
		for(n = 0; n < gst_caps_get_size(caps); n++) {
			if(element->delay)
			        gst_structure_set(gst_caps_get_structure(caps, n), "channels", G_TYPE_INT, iir_channels(element), NULL);
			else
				gst_structure_set(gst_caps_get_structure(caps, n), "channels", GST_TYPE_INT_RANGE, 1, G_MAXINT, NULL);
		}
		g_mutex_unlock(&element->iir_matrix_lock);
		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		gst_caps_unref(caps);
		return GST_CAPS_NONE;
	}

	return caps;
}


/*
 * transform_size()
 */


static gboolean transform_size(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, guint size, GstCaps *othercaps, guint *othersize)
{
	GSTLALIIRBankCuda *element = CUDA_IIRBANK(trans);
	guint unit_size;
	guint other_unit_size;
	int dmin, dmax;

	if(!get_unit_size(trans, caps, &unit_size))
		return FALSE;
	if(size % unit_size) {
		GST_DEBUG_OBJECT(element, "size not a multiple of %u", unit_size);
		return FALSE;
	}
	if(!get_unit_size(trans, othercaps, &other_unit_size))
		return FALSE;

	switch(direction) {
	case GST_PAD_SRC:
		/*
		 * just keep the sample count the same
		 */

		*othersize = (size / unit_size) * other_unit_size;
		break;

	case GST_PAD_SINK:
		/*
		 * upper bound of sample count on source pad is input
		 * sample count plus the number of samples in the adapter
		 * minus the impulse response length of the filters (-1
		 * because if there's 1 impulse response of data then we
		 * can generate 1 sample, not 0)
		 */

		g_mutex_lock(&element->iir_matrix_lock);
		while(!element->delay || !element->a1 || !element->b0)
			g_cond_wait(&element->iir_matrix_available, &element->iir_matrix_lock);

         	gsl_matrix_int_minmax(element->delay, &dmin, &dmax);
		dmin = 0;
		g_mutex_unlock(&element->iir_matrix_lock);

		*othersize = size / unit_size + get_available_samples(element);

		if((gint) *othersize > dmax - dmin)
		        *othersize = (*othersize - (dmax-dmin)) * other_unit_size;
		else
			*othersize = 0;
		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		return FALSE;
	}

	return TRUE;
}


/*
 * set_caps()
 */


static gboolean set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps)
{
	GSTLALIIRBankCuda *element = CUDA_IIRBANK(trans);
	GstStructure *s;
	gint rate;
	gint channels;
	gint width;
	gboolean success = TRUE;

	s = gst_caps_get_structure(outcaps, 0);
	success &= gst_structure_get_int(s, "channels", &channels);
	success &= gst_structure_get_int(s, "width", &width);
	success &= gst_structure_get_int(s, "rate", &rate);

	if(success && element->delay && (channels != (gint) iir_channels(element))) {
		GST_DEBUG_OBJECT(element, "channels != %d in %" GST_PTR_FORMAT, iir_channels(element), outcaps);
		success = FALSE;
	}

	if(success) {
		gboolean format_changed = element->rate != rate || element->width != width;
		gint old_rate = element->rate;
		element->rate = rate;
		element->width = width;
		if(format_changed){
			gst_audioadapter_clear(element->adapter);
			element->zeros_in_adapter = 0;
			element->t0 = GST_CLOCK_TIME_NONE;	/* force discont */
		}
		if(element->rate != old_rate)
			g_signal_emit(G_OBJECT(trans), signals[SIGNAL_RATE_CHANGED], 0, element->rate, NULL);

	} else
		GST_ERROR_OBJECT(element, "unable to parse and/or accept caps %" GST_PTR_FORMAT, outcaps);

	return success;
}

/*
 * start()
 */


static gboolean start(GstBaseTransform *trans)
{
	GSTLALIIRBankCuda *element = CUDA_IIRBANK(trans);
	element->zeros_in_adapter = 0;
	element->t0 = GST_CLOCK_TIME_NONE;
	element->offset0 = GST_BUFFER_OFFSET_NONE;
	element->next_in_offset = GST_BUFFER_OFFSET_NONE;
	element->next_out_offset = GST_BUFFER_OFFSET_NONE;
	element->need_discont = TRUE;
	return TRUE;
}


/*
 * transform()
 */


static GstFlowReturn transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf)
{
	guint64 length;
	GSTLALIIRBankCuda *element = CUDA_IIRBANK(trans);
	GstFlowReturn result = GST_FLOW_ERROR;

	/*
	 * wait for IIR matrix
	 * FIXME:  add a way to get out of this loop
	 */

	g_mutex_lock(&element->iir_matrix_lock);
	while(!element->delay || !element->a1 || !element->b0)
		g_cond_wait(&element->iir_matrix_available, &element->iir_matrix_lock);

	g_assert(element->b0->size1 == element->delay->size1);
	g_assert(element->a1->size1 == element->delay->size1);
	g_assert(element->b0->size2 == element->delay->size2);
	g_assert(element->a1->size2 == element->delay->size2);

	if(!element->y)
	        element->y = gsl_matrix_complex_calloc(element->a1->size1, element->a1->size2);
	else if(element->y->size1 != element->delay->size1 || element->y->size2 != element->delay->size2 ) {
		gsl_matrix_complex_free(element->y);
	        element->y = gsl_matrix_complex_calloc(element->a1->size1, element->a1->size2);
	}

	/*
	 * check for discontinuity
	 *
	 * FIXME:  instead of reseting, flush/pad adapter as needed
	 */

	if(GST_BUFFER_IS_DISCONT(inbuf) || GST_BUFFER_OFFSET(inbuf) != element->next_in_offset || !GST_CLOCK_TIME_IS_VALID(element->t0)) {
		int dmin, dmax;

		/*
		 * flush adapter. Erase contents of adaptor. Push dmax - dmin zeros, so the IIR filter bank can start filtering from t0.
		 */

		gst_audioadapter_clear(element->adapter);
		element->zeros_in_adapter = 0;

		gsl_matrix_int_minmax(element->delay, &dmin, &dmax);
		dmin = 0;
		push_zeros(element, dmax-dmin);

                /*
		 * (re)sync timestamp and offset book-keeping. Set t0 and offset0 to be the timestamp and offset of the inbuf. Next_out_offset is to set the offset of the outbuf.
		 */

		element->t0 = GST_BUFFER_TIMESTAMP(inbuf);
		element->offset0 = GST_BUFFER_OFFSET(inbuf);
		element->next_out_offset = element->offset0 + dmin;

		/*
		 * be sure to flag the next output buffer as a discontinuity, because it is the first output buffer.
		 */

		element->need_discont = TRUE;
	}
	element->next_in_offset = GST_BUFFER_OFFSET_END(inbuf);

	/*
	 * gap logic
	 *
	 * FIXME:  gap handling is busted
	 */

	length = GST_BUFFER_OFFSET_END(inbuf) - GST_BUFFER_OFFSET(inbuf);
	if(!GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP)) {
		/*
		 * input is not 0s
		 */
		gst_buffer_ref(inbuf);	/* don't let the adapter free it */
		gst_audioadapter_push(element->adapter, inbuf);
		element->zeros_in_adapter = 0;
		if (element->width == 64)
			result = filter_d(element, outbuf);
		else if (element->width == 32)
			result = filter_s(element, outbuf);

	} else if(TRUE) {
		/*
		 * input is 0s, FIXME here. Make dependent on decay rate.
		 */

		push_zeros(element, length);
		if (element->width == 64)
			result = filter_d(element, outbuf);
		else if (element->width == 32)
			result = filter_s(element, outbuf);

	}
	//fprintf(stderr, "TIMESTAMP %f, BUFFERSIZE %d\n", (double) 1e-9 * GST_BUFFER_TIMESTAMP(inbuf), GST_BUFFER_SIZE(inbuf) / sizeof(double));

	int dmin, dmax;
	gsl_matrix_int_minmax(element->delay, &dmin, &dmax);
	dmin = 0;
	/*fprintf(stderr, "IIR elem %17s : input timestamp %llu, output timestamp %llu, input offset %llu, output offset %llu, %d, %d\n",
		GST_ELEMENT_NAME(element),
		GST_BUFFER_TIMESTAMP(inbuf),
		GST_BUFFER_TIMESTAMP(outbuf),
		GST_BUFFER_OFFSET(inbuf),
		GST_BUFFER_OFFSET(outbuf),
		dmin,
		dmax);*/
	//fprintf(stderr, "IIR elem %17s : input offset %llu, output offset %llu, %d, %d\n", GST_ELEMENT_NAME(element), GST_BUFFER_OFFSET(inbuf), GST_BUFFER_OFFSET(outbuf), dmin, dmax);
	/*
	 * done
	 */

	g_mutex_unlock(&element->iir_matrix_lock);
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
	GSTLALIIRBankCuda *element = CUDA_IIRBANK(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_IIR_A1:
		g_mutex_lock(&element->iir_matrix_lock);
		if(element->a1)
		        gsl_matrix_complex_free(element->a1);

		element->a1 = gstlal_gsl_matrix_complex_from_g_value_array(g_value_get_boxed(value));

		/*
		 * signal change of IIR coeffs
		 */

		g_cond_broadcast(&element->iir_matrix_available);
		g_mutex_unlock(&element->iir_matrix_lock);
		break;

	case ARG_IIR_B0:
		g_mutex_lock(&element->iir_matrix_lock);
		if(element->b0)
		        gsl_matrix_complex_free(element->b0);

		element->b0 = gstlal_gsl_matrix_complex_from_g_value_array(g_value_get_boxed(value));

		/*
		 * signal change of IIR coeffs
		 */

		g_cond_broadcast(&element->iir_matrix_available);
		g_mutex_unlock(&element->iir_matrix_lock);
		break;

	case ARG_IIR_DELAY: {
		int dmin, dmax;
		int dmin_new, dmax_new;

		g_mutex_lock(&element->iir_matrix_lock);
		if(element->delay) {
			gsl_matrix_int_minmax(element->delay, &dmin, &dmax);
			dmin = 0;
			gsl_matrix_int_free(element->delay);
		} else
			dmin = dmax = 0;

		element->delay = gstlal_gsl_matrix_int_from_g_value_array(g_value_get_boxed(value));
		gsl_matrix_int_minmax(element->delay, &dmin_new, &dmax_new);
		dmin_new = 0;

		if(dmax_new-dmin_new > dmax-dmin)
			push_zeros(element, dmax_new-dmin_new-(dmax-dmin));

		/*
		 * signal change of IIR delays
		 */

		g_cond_broadcast(&element->iir_matrix_available);
		g_mutex_unlock(&element->iir_matrix_lock);
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
	GSTLALIIRBankCuda *element = CUDA_IIRBANK(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_IIR_A1:
		g_mutex_lock(&element->iir_matrix_lock);
		if(element->a1)
			g_value_take_boxed(value, gstlal_g_value_array_from_gsl_matrix_complex(element->a1));
		/* FIXME:  else? */
		g_mutex_unlock(&element->iir_matrix_lock);
		break;

	case ARG_IIR_B0:
		g_mutex_lock(&element->iir_matrix_lock);
		if(element->b0)
			g_value_take_boxed(value, gstlal_g_value_array_from_gsl_matrix_complex(element->b0));
		/* FIXME:  else? */
		g_mutex_unlock(&element->iir_matrix_lock);
		break;

	case ARG_IIR_DELAY:
		g_mutex_lock(&element->iir_matrix_lock);
		if(element->delay)
			g_value_take_boxed(value, gstlal_g_value_array_from_gsl_matrix_int(element->delay));
		/* FIXME:  else? */
		g_mutex_unlock(&element->iir_matrix_lock);
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
	GSTLALIIRBankCuda *element = CUDA_IIRBANK(object);

	g_mutex_clear(&element->iir_matrix_lock);
	//element->iir_matrix_lock = NULL;
	g_cond_clear(&element->iir_matrix_available);
	//element->iir_matrix_available = NULL;
	if(element->a1) {
		gsl_matrix_complex_free(element->a1);
		element->a1 = NULL;
	}
	if(element->b0) {
		gsl_matrix_complex_free(element->b0);
		element->b0 = NULL;
	}
	if(element->delay) {
		gsl_matrix_int_free(element->delay);
		element->delay = NULL;
	}
	if(element->y) {
		gsl_matrix_complex_free(element->y);
		element->y = NULL;
	}
	if(element->bank) {
		cuda_bank_free(element);
		element->bank = NULL;
	}
	g_object_unref(element->adapter);
	element->adapter = NULL;

	G_OBJECT_CLASS(cuda_iirbank_parent_class)->finalize(object);
}


/*
 * base_init()
 * Obsolete, moved to class init
 */

#if 0
static void cuda_iirbank_base_init(gpointer gclass)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(gclass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(gclass);

	gst_element_class_set_details_simple(element_class, "Cuda Mode IIR Filter Bank", "Filter/Audio", "Projects a single audio channel onto a bank of IIR filters to produce a multi-channel output", "Qi Chu <chicsheep@gmail.com>");

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));

	transform_class->get_unit_size = GST_DEBUG_FUNCPTR(get_unit_size);
	transform_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
	transform_class->transform = GST_DEBUG_FUNCPTR(transform);
	transform_class->transform_caps = GST_DEBUG_FUNCPTR(transform_caps);
	transform_class->transform_size = GST_DEBUG_FUNCPTR(transform_size);
	transform_class->start = GST_DEBUG_FUNCPTR(start);
}
#endif

/*
 * class_init()
 */


static void cuda_iirbank_class_init(GSTLALIIRBankCudaClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(klass);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	klass->rate_changed = GST_DEBUG_FUNCPTR(rate_changed);

	transform_class->get_unit_size = GST_DEBUG_FUNCPTR(get_unit_size);
	transform_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
	transform_class->transform = GST_DEBUG_FUNCPTR(transform);
	transform_class->transform_caps = GST_DEBUG_FUNCPTR(transform_caps);
	transform_class->transform_size = GST_DEBUG_FUNCPTR(transform_size);
	transform_class->start = GST_DEBUG_FUNCPTR(start);

	gst_element_class_set_details_simple(element_class, "Cuda Mode IIR Filter Bank", "Filter/Audio", "Projects a single audio channel onto a bank of IIR filters to produce a multi-channel output", "Qi Chu <chicsheep@gmail.com>");

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));


	g_object_class_install_property(
		gobject_class,
		ARG_IIR_A1,
		g_param_spec_value_array(
			"a1-matrix",
			"Matric of IIR feedback coefficients",
			"A Matrix of first order IIR filter feedback coefficients. Each row represents a different IIR bank.",
			g_param_spec_value_array(
				"a1",
				"IIR bank feedback coefficients",
				"A parallel bank of first order IIR filter feedback coefficients",
				g_param_spec_double(
					"coefficient",
					"Coefficient",
					"Feedback coefficient",
					-G_MAXDOUBLE, G_MAXDOUBLE, 0.0,
					G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
				),
				G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
			),
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
	        )
	);

	g_object_class_install_property(
		gobject_class,
		ARG_IIR_B0,
		g_param_spec_value_array(
			"b0-matrix",
			"Matrix of IIR bank feedforward coefficients",
			"Array of first order IIR filter coefficients. Each row represents a different IIR bank.",
			g_param_spec_value_array(
				"b0",
				"IIR bank of feedforward coefficients",
				"A parallel bank of first order IIR filter feedforward coefficents",
				g_param_spec_double(
					 "coefficient",
					 "Coefficient",
					 "Current input sample coefficient",
					 -G_MAXDOUBLE, G_MAXDOUBLE, 0.0,
					 G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
			        ),
				G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
			),
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);

	g_object_class_install_property(
		gobject_class,
		ARG_IIR_DELAY,
		g_param_spec_value_array(
			"delay-matrix",
			"Matrix of delays for IIR filter bank",
			"Matrix of delays for first order IIR filters.  All filters must have the same length.",
			g_param_spec_value_array(
				"delay",
				"delays for IIR bank",
				"A parallel bank of first order IIR filter delays",
				g_param_spec_int(
					"delay",
					"Delay",
					"Delay for IIR filter",
					0, G_MAXINT, 0,
					G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
				),
				G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
			),
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);

	signals[SIGNAL_RATE_CHANGED] = g_signal_new(
		"rate-changed",
		G_TYPE_FROM_CLASS(klass),
		G_SIGNAL_RUN_FIRST,
		G_STRUCT_OFFSET(
			GSTLALIIRBankCudaClass,
			rate_changed
		),
		NULL,
		NULL,
		g_cclosure_marshal_VOID__INT,
		G_TYPE_NONE,
		1,
		G_TYPE_INT
	);
}


/*
 * init()
 */


static void cuda_iirbank_init(GSTLALIIRBankCuda *filter)
{
	filter->adapter = NULL;
	g_mutex_init(&filter->iir_matrix_lock);
	g_cond_init(&filter->iir_matrix_available);
	filter->a1 = NULL;
	filter->b0 = NULL;
	filter->delay = NULL;
	filter->y = NULL;
	filter->bank = NULL;
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(filter), TRUE);
}


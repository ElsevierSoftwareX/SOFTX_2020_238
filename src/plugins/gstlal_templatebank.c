/*
 * A template bank.
 *
 * Copyright (C) 2008  Kipp Cannon, Chad Hanna
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */


/*
 * ========================================================================
 *
 *                                  Preamble
 *
 * ========================================================================
 */


/*
 * stuff from the C library
 */


#include <stdint.h>
#include <string.h>


/*
 * stuff from gstreamer
 */


#include <gst/gst.h>
#include <gst/base/gstadapter.h>


/*
 * stuff from LAL
 */


/*
 * stuff from GSL
 */


#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>


/*
 * our own stuff
 */


#include <gstlal.h>
#include <gstlal_templatebank.h>
#include <low_latency_inspiral_functions.h>


/*
 * ========================================================================
 *
 *                                 Parameters
 *
 * ========================================================================
 */


#define DEFAULT_T_START 0
#define DEFAULT_T_END G_MAXUINT
#define DEFAULT_SNR_LENGTH 2048	/* samples */
#define TEMPLATE_DURATION 64	/* seconds */
#define CHIRPMASS_START 1.15	/* M_sun */
#define TEMPLATE_SAMPLE_RATE 2048	/* Hertz */
#define NUM_TEMPLATES 200
#define TOLERANCE 0.97


/*
 * ========================================================================
 *
 *                             Utility Functions
 *
 * ========================================================================
 */


static void svd_destroy(GSTLALTemplateBank *element)
{
	if(element->U) {
		gsl_matrix_free(element->U);
		element->U = NULL;
	}
	if(element->S) {
		gsl_vector_free(element->S);
		element->S = NULL;
	}
	if(element->V) {
		gsl_matrix_free(element->V);
		element->V = NULL;
	}
	if(element->chifacs) {
		gsl_vector_free(element->chifacs);
		element->chifacs = NULL;
	}
}


static int svd_create(GSTLALTemplateBank *element, int sample_rate)
{
	int verbose = 1;

	/* be sure we don't leak memory */

	svd_destroy(element);

	/* clip t_start and t_end so that 0 <= t_start <= t_end <=
	 * TEMPLATE_DURATION (both are unsigned so can't be negative) */

	if(element->t_start > TEMPLATE_DURATION)
		element->t_start = TEMPLATE_DURATION;
	if(element->t_end < element->t_start)
		element->t_end = element->t_start;
	else if(element->t_end > TEMPLATE_DURATION)
		element->t_end = TEMPLATE_DURATION;

	/* generate orthogonal template bank */

	generate_bank_svd(&element->U, &element->S, &element->V, &element->chifacs, CHIRPMASS_START, TEMPLATE_SAMPLE_RATE, TEMPLATE_SAMPLE_RATE / sample_rate, NUM_TEMPLATES, element->t_start, element->t_end, TEMPLATE_DURATION, TOLERANCE, verbose);

	/* done */

	return 0;
}


/*
 * ============================================================================
 *
 *                          GStreamer Source Element
 *
 * ============================================================================
 */


/*
 * Properties
 */


enum property {
	ARG_T_START = 1,
	ARG_T_END,
	ARG_SNR_LENGTH
};


static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec)
{
	GSTLALTemplateBank *element = GSTLAL_TEMPLATEBANK(object);

	switch(id) {
	case ARG_T_START:
		element->t_start = g_value_get_uint(value);
		break;

	case ARG_T_END:
		element->t_end = g_value_get_uint(value);
		break;

	case ARG_SNR_LENGTH:
		element->snr_length = g_value_get_uint(value);
		break;
	}
}


static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *pspec)
{
	GSTLALTemplateBank *element = GSTLAL_TEMPLATEBANK(object);

	switch(id) {
	case ARG_T_START:
		g_value_set_uint(value, element->t_start);
		break;

	case ARG_T_END:
		g_value_set_uint(value, element->t_end);
		break;

	case ARG_SNR_LENGTH:
		g_value_set_uint(value, element->snr_length);
	}
}


/*
 * setcaps()
 */


static gboolean setcaps_sink(GstPad *pad, GstCaps *caps)
{
	GSTLALTemplateBank *element = GSTLAL_TEMPLATEBANK(gst_pad_get_parent(pad));
	gboolean result = TRUE;

	element->sample_rate = g_value_get_int(gst_structure_get_value(gst_caps_get_structure(caps, 0), "rate"));

	gst_object_unref(element);
	return result;
}


static gboolean setcaps_orthogonal_snr_sink(GstPad *pad, GstCaps *caps)
{
	GSTLALTemplateBank *element = GSTLAL_TEMPLATEBANK(gst_pad_get_parent(pad));
	gboolean result = TRUE;

	/*
	 * get a modifiable copy of the caps
	 */

	caps = gst_caps_make_writable(caps);

	/*
	 * has the reconstruction matrix been built yet?
	 */

	if(!element->V)
		goto done;

	/*
	 * check that the number of input channels matches the size of the
	 * reconstruction matrix
	 */

	if((unsigned) g_value_get_int(gst_structure_get_value(gst_caps_get_structure(caps, 0), "channels")) != element->V->size1) {
		result = FALSE;
		goto done;
	}

	/*
	 * set the number of output channels and forward caps to next
	 * element
	 */

	gst_caps_set_simple(caps, "channels", G_TYPE_INT, element->V->size2, NULL);
	result = gst_pad_set_caps(element->snr_pad, caps);

done:
	gst_caps_unref(caps);
	gst_object_unref(element);
	return result;
}


/*
 * chain()
 */


static GstFlowReturn chain_sink(GstPad *pad, GstBuffer *sinkbuf)
{
	GSTLALTemplateBank *element = GSTLAL_TEMPLATEBANK(gst_pad_get_parent(pad));
	GstCaps *caps = gst_buffer_get_caps(sinkbuf);
	gboolean is_discontinuity = FALSE;
	GstFlowReturn result = GST_FLOW_OK;
	int output_length;
	int i;

	/*
	 * Now that we know the sample rate, construct orthogonal basis for
	 * the template bank if not already done.
	 */

	if(!element->U) {
		GstCaps *srccaps;
		gboolean success;

		/*
		 * Create the orthogonal basis.
		 */

		svd_create(element, element->sample_rate);

		/*
		 * Now that we know how many channels we'll produce, set
		 * the srcpad's caps properly.  gst_caps_make_writable()
		 * unref()s its argument so we have to ref() it first to
		 * keep it valid.
		 */

		success = gst_pad_set_caps(element->orthogonal_snr_sum_squares_pad, caps);
		if(success != TRUE) {
			result = GST_FLOW_NOT_NEGOTIATED;
			goto done;
		}
		gst_caps_ref(caps);
		srccaps = gst_caps_make_writable(caps);
		gst_caps_set_simple(srccaps, "channels", G_TYPE_INT, element->U->size1, NULL);
		success = gst_pad_set_caps(element->orthogonal_snr_pad, srccaps);
		gst_caps_unref(srccaps);
		if(success != TRUE) {
			result = GST_FLOW_NOT_NEGOTIATED;
			goto done;
		}
	}

	/*
	 * Check for a discontinuity
	 */

	if(GST_BUFFER_IS_DISCONT(sinkbuf)) {
		is_discontinuity = TRUE;
		gst_adapter_clear(element->adapter);
		element->adapter_head_timestamp = GST_BUFFER_TIMESTAMP(sinkbuf);
	}

	/*
	 * Put buffer into adapter.
	 */

	gst_adapter_push(element->adapter, sinkbuf);

	/*
	 * While there is enough data in the adapter to produce a buffer of
	 * SNR.
	 */

	while(1) {
		gsl_vector time_series = {
			.size = element->U->size2,
			.stride = 1,
			.data = NULL,
			.block = NULL,
			.owner = 0
		};
		gsl_vector orthogonal_snr = {
			.size = element->U->size1,
			.stride = 1,
			.data = NULL,
			.block = NULL,
			.owner = 0
		};
		gsl_vector orthogonal_snr_sum_squares = {
			.size = 0,
			.stride = 1,
			.data = NULL,
			.block = NULL,
			.owner = 0
		};
		GstBuffer *orthogonal_snr_buf;
		GstBuffer *orthogonal_snr_sum_squares_buf;

		/*
		 * Check for available data, clip to the required output
		 * length.
		 */

		output_length = (gst_adapter_available(element->adapter) / sizeof(*time_series.data)) - element->U->size2;
		if(element->snr_length) {
			if(output_length < (int) element->snr_length)
				break;
			output_length = element->snr_length;
		} else if(output_length <= 0)
			break;

		/*
		 * Get a buffer from the downstream peer
		 */

		result = gst_pad_alloc_buffer(element->orthogonal_snr_pad, element->next_sample, element->U->size1 * output_length * sizeof(*orthogonal_snr.data), GST_PAD_CAPS(element->orthogonal_snr_pad), &orthogonal_snr_buf);
		if(result != GST_FLOW_OK)
			goto done;

		result = gst_pad_alloc_buffer(element->orthogonal_snr_sum_squares_pad, element->next_sample, output_length * sizeof(*orthogonal_snr_sum_squares.data), GST_PAD_CAPS(element->orthogonal_snr_sum_squares_pad), &orthogonal_snr_sum_squares_buf);
		if(result != GST_FLOW_OK)
			/* FIXME: unref other buffers */
			goto done;

		/*
		 * Set the metadata.  The time of the start of the h(t)
		 * buffer from which the orthogonal SNR buffer has been
		 * constructed is element->adapter_head_timestamp.
		 * Relative to the time-of-coalescence --- the "time" of
		 * the template --- the first sample of the orthogonal
		 * template vector is at -t_end + 1 * deltaT.  The "time"
		 * of an SNR sample is, therefore, the start of the h(t)
		 * buffer + t_end - 1*deltaT.
		 */

		if(is_discontinuity) {
			GST_BUFFER_FLAG_SET(orthogonal_snr_sum_squares_buf, GST_BUFFER_FLAG_DISCONT);
			GST_BUFFER_FLAG_SET(orthogonal_snr_buf, GST_BUFFER_FLAG_DISCONT);
			is_discontinuity = FALSE;
		}
		GST_BUFFER_OFFSET_END(orthogonal_snr_sum_squares_buf) = GST_BUFFER_OFFSET_END(orthogonal_snr_buf) = GST_BUFFER_OFFSET(orthogonal_snr_buf) + output_length - 1;
		GST_BUFFER_TIMESTAMP(orthogonal_snr_sum_squares_buf) = GST_BUFFER_TIMESTAMP(orthogonal_snr_buf) = element->adapter_head_timestamp + element->t_end * GST_SECOND - GST_SECOND / element->sample_rate;
		GST_BUFFER_DURATION(orthogonal_snr_sum_squares_buf) = GST_BUFFER_DURATION(orthogonal_snr_buf) = output_length * GST_SECOND / element->sample_rate;

		/*
		 * Assemble the orthogonal SNR time series as the columns
		 * of a matrix.  Instead of shuffling the bytes around in
		 * memory, we play games with the adapter and pointers.  We
		 * start by asking for all the data we will need for the
		 * convolution, but only use part of it and then shift the
		 * pointer and iterate.  Later we'll use a single flush to
		 * discard all the samples we no longer need.  The adapter
		 * will keep the data valid until another method is called,
		 * so we have to be careful not to call any methods on the
		 * adapter until the loop is finished.
		 */

		time_series.data = (double *) gst_adapter_peek(element->adapter, (time_series.size + output_length - 1) * sizeof(*time_series.data));
		orthogonal_snr.data = (double *) GST_BUFFER_DATA(orthogonal_snr_buf);
		orthogonal_snr_sum_squares.data = (double *) GST_BUFFER_DATA(orthogonal_snr_sum_squares_buf);
		orthogonal_snr_sum_squares.size = output_length;

		for(i = 0; i < output_length; i++) {
			/*
			 * Compute one vector of orthogonal SNR samples ---
			 * the projection of h(t) onto the template bank's
			 * orthonormal basis.
			 */

			gsl_blas_dgemv(CblasNoTrans, 1.0, element->U, &time_series, 0.0, &orthogonal_snr);

			/*
			 * From the projection of h(t) onto the bank's
			 * orthonormal basis, compute the magnitude of the
			 * component of h(t) in the bank
			 */

			gsl_vector_set(&orthogonal_snr_sum_squares, i, pow(gsl_blas_dnrm2(&orthogonal_snr), 2));

			/*
			 * Advance the pointers.
			 */

			time_series.data++;
			orthogonal_snr.data += orthogonal_snr.size;
		}

		/*
		 * Push the buffers downstream.
		 */

		result = gst_pad_push(element->orthogonal_snr_pad, orthogonal_snr_buf);
		if(result != GST_FLOW_OK)
			goto done;

		fprintf(stderr, "largest orthogonal SNR sum-of-squares = %.16g (at %.16g s)\n", gsl_vector_max(&orthogonal_snr_sum_squares), GST_BUFFER_TIMESTAMP(orthogonal_snr_sum_squares_buf) / (double) GST_SECOND + gsl_vector_max_index(&orthogonal_snr_sum_squares) / (double) element->sample_rate);
		result = gst_pad_push(element->orthogonal_snr_sum_squares_pad, orthogonal_snr_sum_squares_buf);
		if(result != GST_FLOW_OK)
			goto done;

		/*
		 * Flush the data from the adapter that is no longer
		 * required, and advance the sample count.
		 */

		gst_adapter_flush(element->adapter, output_length * sizeof(*time_series.data));
		element->adapter_head_timestamp += (GstClockTime) output_length * GST_SECOND / element->sample_rate;
		element->next_sample += output_length;
	}

	/*
	 * Done
	 */

done:
	gst_caps_unref(caps);
	gst_object_unref(element);
	return result;
}


static GstFlowReturn chain_orthogonal_snr_sink(GstPad *pad, GstBuffer *sinkbuf)
{
	GSTLALTemplateBank *element = GSTLAL_TEMPLATEBANK(gst_pad_get_parent(pad));
	gsl_matrix orthogonal_snr = {
		/* number of samples in each SNR channel */
		.size1 = GST_BUFFER_SIZE(sinkbuf) / sizeof(*orthogonal_snr.data) / element->U->size1,
		/* number of orthogonal SNR channels coming in */
		.size2 = element->V->size1,
		.tda = element->V->size1,
		.data = (double *) GST_BUFFER_DATA(sinkbuf),
		.block = NULL,
		.owner = 0
	};
	GstBuffer *srcbuf;
	gsl_matrix snr = {
		/* number of samples in each SNR channel */
		.size1 = orthogonal_snr.size1,
		/* number of SNR channels going out */
		.size2 = element->V->size2,
		.tda = element->V->size2,
		.data = NULL,
		.block = NULL,
		.owner = 0
	};
	GstFlowReturn result = GST_FLOW_OK;

	/*
	 * Get a buffer from the downstream peer
	 */

	result = gst_pad_alloc_buffer(element->snr_pad, GST_BUFFER_OFFSET(sinkbuf), snr.size1 * snr.size2 * sizeof(*snr.data), GST_PAD_CAPS(element->snr_pad), &srcbuf);
	if(result != GST_FLOW_OK)
		goto done;
	snr.data = (double *) GST_BUFFER_DATA(srcbuf);

	/*
	 * Copy metadata
	 */

	GST_BUFFER_OFFSET_END(srcbuf) = GST_BUFFER_OFFSET_END(sinkbuf);
	GST_BUFFER_TIMESTAMP(srcbuf) = GST_BUFFER_TIMESTAMP(sinkbuf);
	GST_BUFFER_DURATION(srcbuf) = GST_BUFFER_DURATION(sinkbuf);

	/*
	 * Reconstruct SNRs
	 */

	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, &orthogonal_snr, element->V, 0, &snr);

	/*
	 * Push the buffer downstream
	 */

	result = gst_pad_push(element->snr_pad, srcbuf);
	if(result != GST_FLOW_OK)
		goto done;

	/*
	 * Done
	 */

done:
	gst_object_unref(element);
	return result;
}


/*
 * Parent class.
 */


static GstElementClass *parent_class = NULL;


/*
 * Instance dispose function.  See ???
 */


static void dispose(GObject *object)
{
	GSTLALTemplateBank *element = GSTLAL_TEMPLATEBANK(object);

	gst_object_unref(element->orthogonal_snr_pad);
	element->orthogonal_snr_pad = NULL;
	gst_object_unref(element->orthogonal_snr_sum_squares_pad);
	element->orthogonal_snr_sum_squares_pad = NULL;
	gst_object_unref(element->snr_pad);
	element->snr_pad = NULL;
	g_object_unref(element->adapter);
	element->adapter = NULL;

	svd_destroy(element);

	G_OBJECT_CLASS(parent_class)->dispose(object);
}


/*
 * Base init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GBaseInitFunc
 */


static void base_init(gpointer class)
{
	static GstElementDetails plugin_details = {
		"Template Bank",
		"Filter",
		"A time-domain filter bank",
		"Kipp Cannon <kcannon@ligo.caltech.edu>, Chan Hanna <channa@ligo.caltech.edu>"
	};
	GstElementClass *element_class = GST_ELEMENT_CLASS(class);

	gst_element_class_set_details(element_class, &plugin_details);

	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"sink",
			GST_PAD_SINK,
			GST_PAD_ALWAYS,
			gst_caps_new_simple(
				"audio/x-raw-float",
				"rate", GST_TYPE_INT_RANGE, 1, TEMPLATE_SAMPLE_RATE,
				"channels", G_TYPE_INT, 1,
				"endianness", G_TYPE_INT, G_BYTE_ORDER,
				"width", G_TYPE_INT, 64,
				NULL
			)
		)
	);
	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"orthogonal_snr",
			GST_PAD_SRC,
			GST_PAD_ALWAYS,
			gst_caps_new_simple(
				"audio/x-raw-float",
				"rate", GST_TYPE_INT_RANGE, 1, TEMPLATE_SAMPLE_RATE,
				"channels", GST_TYPE_INT_RANGE, 1, G_MAXINT,
				"endianness", G_TYPE_INT, G_BYTE_ORDER,
				"width", G_TYPE_INT, 64,
				NULL
			)
		)
	);
	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"orthogonal_snr_sum_squares",
			GST_PAD_SRC,
			GST_PAD_ALWAYS,
			gst_caps_new_simple(
				"audio/x-raw-float",
				"rate", GST_TYPE_INT_RANGE, 1, TEMPLATE_SAMPLE_RATE,
				"channels", G_TYPE_INT, 1,
				"endianness", G_TYPE_INT, G_BYTE_ORDER,
				"width", G_TYPE_INT, 64,
				NULL
			)
		)
	);
	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"orthogonal_snr_sink",
			GST_PAD_SINK,
			GST_PAD_ALWAYS,
			gst_caps_new_simple(
				"audio/x-raw-float",
				"rate", GST_TYPE_INT_RANGE, 1, G_MAXINT,
				"channels", GST_TYPE_INT_RANGE, 1, G_MAXINT,
				"endianness", G_TYPE_INT, G_BYTE_ORDER,
				"width", G_TYPE_INT, 64,
				NULL
			)
		)
	);
	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"snr",
			GST_PAD_SRC,
			GST_PAD_ALWAYS,
			gst_caps_new_simple(
				"audio/x-raw-float",
				"rate", GST_TYPE_INT_RANGE, 1, G_MAXINT,
				"channels", GST_TYPE_INT_RANGE, 1, G_MAXINT,
				"endianness", G_TYPE_INT, G_BYTE_ORDER,
				"width", G_TYPE_INT, 64,
				NULL
			)
		)
	);
}


/*
 * Class init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GClassInitFunc
 */


static void class_init(gpointer class, gpointer class_data)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(class);

	parent_class = g_type_class_ref(GST_TYPE_ELEMENT);

	gobject_class->set_property = set_property;
	gobject_class->get_property = get_property;
	gobject_class->dispose = dispose;

	g_object_class_install_property(gobject_class, ARG_T_START, g_param_spec_uint("t-start", "Start time", "Start time of subtemplate in seconds measure backwards from end of bank", 0, G_MAXUINT, DEFAULT_T_START, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	g_object_class_install_property(gobject_class, ARG_T_END, g_param_spec_uint("t-end", "End time", "End time of subtemplate in seconds measure backwards from end of bank", 0, G_MAXUINT, DEFAULT_T_END, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	g_object_class_install_property(gobject_class, ARG_SNR_LENGTH, g_param_spec_uint("snr-length", "SNR length", "Length, in samples, of the output SNR time series (0 = no limit)", 0, G_MAXUINT, DEFAULT_SNR_LENGTH, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
}


/*
 * Instance init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GInstanceInitFunc
 */


static void instance_init(GTypeInstance *object, gpointer class)
{
	GSTLALTemplateBank *element = GSTLAL_TEMPLATEBANK(object);
	GstPad *pad;

	gst_element_create_all_pads(GST_ELEMENT(element));

	/* configure sink pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "sink");
	gst_pad_set_setcaps_function(pad, setcaps_sink);
	gst_pad_set_chain_function(pad, chain_sink);
	gst_object_unref(pad);

	/* retrieve (and ref) orthogonal_snr pad */
	element->orthogonal_snr_pad = gst_element_get_static_pad(GST_ELEMENT(element), "orthogonal_snr");

	/* retrieve (and ref) orthogonal_snr_sum_squares pad */
	element->orthogonal_snr_sum_squares_pad = gst_element_get_static_pad(GST_ELEMENT(element), "orthogonal_snr_sum_squares");

	/* configure orthogonal_snr_sink pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "orthogonal_snr_sink");
	gst_pad_set_setcaps_function(pad, setcaps_orthogonal_snr_sink);
	gst_pad_set_chain_function(pad, chain_orthogonal_snr_sink);
	gst_object_unref(pad);

	/* retrieve (and ref) snr pad */
	element->snr_pad = gst_element_get_static_pad(GST_ELEMENT(element), "snr");

	/* internal data */
	element->adapter = gst_adapter_new();
	element->t_start = DEFAULT_T_START;
	element->t_end = DEFAULT_T_END;
	element->snr_length = DEFAULT_SNR_LENGTH;

	element->next_sample = 0;
	element->adapter_head_timestamp = 0;

	element->U = NULL;
	element->S = NULL;
	element->V = NULL;
	element->chifacs = NULL;
}


/*
 * gstlal_templatebank_get_type().
 */


GType gstlal_templatebank_get_type(void)
{
	static GType type = 0;

	if(!type) {
		static const GTypeInfo info = {
			.class_size = sizeof(GSTLALTemplateBankClass),
			.class_init = class_init,
			.base_init = base_init,
			.instance_size = sizeof(GSTLALTemplateBank),
			.instance_init = instance_init,
		};
		type = g_type_register_static(GST_TYPE_ELEMENT, "lal_templatebank", &info, 0);
	}

	return type;
}

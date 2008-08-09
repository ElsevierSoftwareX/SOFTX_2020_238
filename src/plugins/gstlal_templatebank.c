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
#define CHIRPMASS_START 1.0	/* M_sun */
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

	/* clip t_start and t_end to [0, TEMPLATE_DURATION] (both are
	 * unsigned so can't be negative) */

	if(element->t_start > TEMPLATE_DURATION)
		element->t_start = TEMPLATE_DURATION;
	if(element->t_end > TEMPLATE_DURATION)
		element->t_end = TEMPLATE_DURATION;

	/* generate orthogonal template bank */

	generate_bank_svd(&element->U, &element->S, &element->V, &element->chifacs, CHIRPMASS_START, TEMPLATE_SAMPLE_RATE, TEMPLATE_SAMPLE_RATE / sample_rate, NUM_TEMPLATES, element->t_start, element->t_end, TEMPLATE_DURATION, TOLERANCE, verbose);

	/* done */

	return 0;
}


static void srcpads_destroy(GSTLALTemplateBank *element)
{
	GList *padlist;

	for(padlist = element->srcpads; padlist; padlist = g_list_next(padlist))
		gst_object_unref(GST_PAD(padlist->data));

	g_list_free(element->srcpads);
	element->srcpads = NULL;
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


static gboolean setcaps(GstPad *pad, GstCaps *caps)
{
	GSTLALTemplateBank *element = GSTLAL_TEMPLATEBANK(gst_pad_get_parent(pad));
	gboolean result = TRUE;
	GList *padlist;

	for(padlist = element->srcpads; padlist; padlist = g_list_next(padlist)) {
		result = gst_pad_set_caps(GST_PAD(padlist->data), caps);
		if(result != TRUE)
			goto done;
	}

done:
	gst_object_unref(element);
	return result;
}


/*
 * chain()
 */


static GstFlowReturn chain(GstPad *pad, GstBuffer *sinkbuf)
{
	GSTLALTemplateBank *element = GSTLAL_TEMPLATEBANK(gst_pad_get_parent(pad));
	GstCaps *caps = gst_buffer_get_caps(sinkbuf);
	GstFlowReturn result = GST_FLOW_OK;
	gsl_vector time_series = {
		.size = 0,
		.stride = 1,
		.data = NULL,
		.block = NULL,
		.owner = 0
	};
	gsl_vector *orthogonal_snr_samples;
	gsl_matrix *orthogonal_snr;
	int sample_rate;
	int output_length;
	GList *padlist;
	int i;

	/*
	 * Extract the sample rate from the input buffer's caps
	 */

	sample_rate = g_value_get_int(gst_structure_get_value(gst_caps_get_structure(caps, 0), "rate"));

	/*
	 * Construct orthogonal template bank if required
	 */

	if(!element->U)
		svd_create(element, sample_rate);

	/*
	 * Put buffer into adapter, and measure the length of the SNR time
	 * series we can generate (we're done if this is <= 0).
	 */

	gst_adapter_push(element->adapter, sinkbuf);

	/*
	 * While there is enough data in the adapter to produce a buffer of
	 * SNR.
	 */

	while(1) {
		/*
		 * Check for available data, clip the output length.
		 */

		output_length = (gst_adapter_available(element->adapter) / sizeof(*time_series.data)) - element->U->size2;
		if(element->snr_length) {
			if(output_length < (int) element->snr_length)
				break;
			output_length = element->snr_length;
		} else if(output_length <= 0)
			break;

		/*
		 * Assemble the orthogonal SNR time series as the rows of a
		 * matrix.
		 */

		time_series.size = element->U->size2;
		orthogonal_snr = gsl_matrix_alloc(element->U->size1, output_length);
		orthogonal_snr_samples = gsl_vector_alloc(element->U->size1);

		for(i = 0; i < output_length; i++) {
			/*
			 * Instead of shuffling the bytes around in memory,
			 * we play games with the adapter and pointers.  We
			 * start by asking for input_size samples.  In the
			 * next iteration, we want to shift the samples by
			 * 1, but instead of flushing 1 sample from the
			 * adapter we ask for input_size+1 samples, and add
			 * 1 to the address we are given.  Later we'll use
			 * a single flush to discard all the samples we no
			 * longer need.
			 */

			time_series.data = ((double *) gst_adapter_peek(element->adapter, (time_series.size + i) * sizeof(*time_series.data))) + i;

			/*
			 * Compute one vector of orthogonal SNR samples
			 */

			gsl_blas_dgemv(CblasNoTrans, 1.0, element->U, &time_series, 0.0, orthogonal_snr_samples);

			/*
			 * Store in matrix of orthogonal SNR series.
			 */

			gsl_matrix_set_col(orthogonal_snr, i, orthogonal_snr_samples);
		}

		gsl_vector_free(orthogonal_snr_samples);

		/*
		 * Flush the data from the adapter that is no longer
		 * required.
		 */

		gst_adapter_flush(element->adapter, output_length);

		/*
		 * Push the orthogonal SNR time series out their pads
		 */

		time_series.size = output_length;

		for(padlist = element->srcpads, i = 0; padlist && (i < orthogonal_snr->size1); padlist = g_list_next(padlist), i++) {
			GstPad *srcpad = GST_PAD(padlist->data);
			GstBuffer *srcbuf;

			result = gst_pad_alloc_buffer(srcpad, element->next_sample, output_length * sizeof(*time_series.data), GST_PAD_CAPS(srcpad), &srcbuf);
			if(result != GST_FLOW_OK)
				goto done;
			GST_BUFFER_OFFSET_END(srcbuf) = element->next_sample + output_length - 1;
			GST_BUFFER_TIMESTAMP(srcbuf) = (GstClockTime) element->next_sample * 1000000000 / sample_rate + element->t_start * 1000000000;
			GST_BUFFER_DURATION(srcbuf) = (GstClockTime) output_length * 1000000000 / sample_rate;

			time_series.data = (double *) GST_BUFFER_DATA(srcbuf);

			gsl_matrix_get_row(&time_series, orthogonal_snr, i);

			result = gst_pad_push(srcpad, srcbuf);
			if(result != GST_FLOW_OK)
				goto done;
		}

		gsl_matrix_free(orthogonal_snr);

		/*
		 * Advance the sample count.
		 */

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

	g_object_unref(element->adapter);
	element->adapter = NULL;
	srcpads_destroy(element);

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
		"Kipp Cannon <kcannon@ligo.caltech.edu>, Chan Hanna <chann@ligo.caltech.edu>"
	};
	GstElementClass *element_class = GST_ELEMENT_CLASS(class);
	GstPadTemplate *sinkpad_template = gst_pad_template_new(
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
	);
	GstPadTemplate *srcpad_template = gst_pad_template_new(
		"orthosnr%04d",
		GST_PAD_SRC,
		GST_PAD_SOMETIMES,
		gst_caps_new_simple(
			"audio/x-raw-float",
			"rate", GST_TYPE_INT_RANGE, 1, TEMPLATE_SAMPLE_RATE,
			"channels", G_TYPE_INT, 1,
			"endianness", G_TYPE_INT, G_BYTE_ORDER,
			"width", G_TYPE_INT, 64,
			NULL
		)
	);

	gst_element_class_set_details(element_class, &plugin_details);

	gst_element_class_add_pad_template(element_class, sinkpad_template);
	gst_element_class_add_pad_template(element_class, srcpad_template);
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
	gst_pad_set_setcaps_function(pad, setcaps);
	gst_pad_set_chain_function(pad, chain);
	gst_object_unref(pad);

	/* src pads */
	element->srcpads = NULL;
	{
	/* FIXME:  make the pads dynamic, based on the result of the SVD
	 * decomposition. */
	int i;
	GstPadTemplate *template = gst_element_class_get_pad_template(class, "orthosnr%04d");
	for(i = 0; i < 10; i++) {
		gchar *padname = g_strdup_printf(template->name_template, i);
		pad = gst_pad_new_from_template(template, padname);
		g_free(padname);
		gst_object_ref(pad);	/* for our linked list */
		gst_element_add_pad(GST_ELEMENT(element), pad);
		element->srcpads = g_list_append(element->srcpads, pad);
	}
	}

	/* internal data */
	element->adapter = gst_adapter_new();
	element->t_start = DEFAULT_T_START;
	element->t_end = DEFAULT_T_END;
	element->snr_length = DEFAULT_SNR_LENGTH;

	element->next_sample = 0;

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

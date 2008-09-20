/*
 * A many-to-many mixer.
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


#include <string.h>


/*
 * stuff from gstreamer
 */


#include <gst/gst.h>


/*
 * stuff from LAL
 */


/*
 * stuff from GSL
 */


#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>


/*
 * our own stuff
 */


#include <gstlal.h>
#include <gstlal_matrixmixer.h>


/*
 * ========================================================================
 *
 *                                 Parameters
 *
 * ========================================================================
 */


/*
 * ========================================================================
 *
 *                             Utility Functions
 *
 * ========================================================================
 */


/*
 * ============================================================================
 *
 *                             GStreamer Element
 *
 * ============================================================================
 */


/*
 * setcaps()
 */


static gboolean setcaps(GstPad *pad, GstCaps *caps)
{
	GSTLALMatrixMixer *element = GSTLAL_MATRIXMIXER(gst_pad_get_parent(pad));
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
	result = gst_pad_set_caps(element->srcpad, caps);
	if(!result)
		goto done;

done:
	gst_caps_unref(caps);
	gst_object_unref(element);
	return result;
}


/*
 * chain()
 */


static GstFlowReturn chain(GstPad *pad, GstBuffer *sinkbuf)
{
	GSTLALMatrixMixer *element = GSTLAL_MATRIXMIXER(gst_pad_get_parent(pad));
	GstCaps *caps = gst_buffer_get_caps(sinkbuf);
	gsl_matrix orthogonal_snr = {
		/* number of samples in each SNR channel */
		.size1 = GST_BUFFER_SIZE(sinkbuf) / sizeof(*orthogonal_snr.data) / element->V->size1,
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
	 * Check the number of channels coming in
	 */

	if(!element->V) {
		/* mixing matrix hasn't been set.  this should be
		 * impossible because this pad should be blocked until
		 * there's a valid mixing matrix, but it can't hurt to
		 * check. */
		result = GST_FLOW_NOT_NEGOTIATED;
		goto done;
	}
	if(g_value_get_int(gst_structure_get_value(gst_caps_get_structure(caps, 0), "channels")) != (int) element->V->size1) {
		result = GST_FLOW_NOT_NEGOTIATED;
		goto done;
	}

	/*
	 * Get a buffer from the downstream peer
	 */

	result = gst_pad_alloc_buffer(element->srcpad, GST_BUFFER_OFFSET(sinkbuf), snr.size1 * snr.size2 * sizeof(*snr.data), GST_PAD_CAPS(element->srcpad), &srcbuf);
	if(result != GST_FLOW_OK)
		goto done;
	snr.data = (double *) GST_BUFFER_DATA(srcbuf);

	/*
	 * Copy metadata
	 */

	gst_buffer_copy_metadata(srcbuf, sinkbuf, GST_BUFFER_COPY_FLAGS | GST_BUFFER_COPY_TIMESTAMPS);

	/*
	 * Reconstruct SNRs
	 */

	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, &orthogonal_snr, element->V, 0, &snr);

	/*
	 * Push the buffer downstream
	 */

	result = gst_pad_push(element->srcpad, srcbuf);
	if(result != GST_FLOW_OK)
		goto done;

	/*
	 * Done
	 */

done:
	gst_buffer_unref(sinkbuf);
	gst_caps_unref(caps);
	gst_object_unref(element);
	return result;
}


static GstFlowReturn chain_matrix(GstPad *pad, GstBuffer *sinkbuf)
{
	GSTLALMatrixMixer *element = GSTLAL_MATRIXMIXER(gst_pad_get_parent(pad));
	GstCaps *caps = gst_buffer_get_caps(sinkbuf);
	GstFlowReturn result = GST_FLOW_OK;
	int rows;
	int cols;

	/*
	 * Get the matrix size.
	 */

	cols = g_value_get_int(gst_structure_get_value(gst_caps_get_structure(caps, 0), "channels"));
	rows = GST_BUFFER_SIZE(sinkbuf) / sizeof(*element->V->data) / cols;

	/*
	 * Free the current matrix.
	 */

	if(element->V) {
		gsl_matrix_free(element->V);
		element->V = NULL;
	}

	/*
	 * Allocate a new matrix.
	 */

	element->V = gsl_matrix_alloc(rows, cols);
	if(!element->V) {
		GST_ERROR("gst_matrix_alloc() failed");
		result = GST_FLOW_ERROR;
		goto done;
	}

	/*
	 * Copy the data into it.
	 */

	memcpy(element->V->data, GST_BUFFER_DATA(sinkbuf), GST_BUFFER_SIZE(sinkbuf));

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
	GSTLALMatrixMixer *element = GSTLAL_MATRIXMIXER(object);

	gst_object_unref(element->sinkpad);
	element->sinkpad = NULL;
	gst_object_unref(element->srcpad);
	element->srcpad = NULL;
	if(element->V) {
		gsl_matrix_free(element->V);
		element->V = NULL;
	}

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
		"Matrix Mixer",
		"Filter",
		"A many-to-many mixer",
		"Kipp Cannon <kcannon@ligo.caltech.edu>, Chan Hanna <channa@ligo.caltech.edu>"
	};
	GstElementClass *element_class = GST_ELEMENT_CLASS(class);

	gst_element_class_set_details(element_class, &plugin_details);

	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"matrix",
			GST_PAD_SINK,
			GST_PAD_ALWAYS,
			gst_caps_new_simple(
				"audio/x-raw-float",
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
			"sink",
			GST_PAD_SINK,
			GST_PAD_ALWAYS,
			gst_caps_new_simple(
				"audio/x-raw-float",
				"rate", GST_TYPE_INT_RANGE, 1, G_MAXINT,
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
			"src",
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

	gobject_class->dispose = dispose;
}


/*
 * Instance init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GInstanceInitFunc
 */


static void instance_init(GTypeInstance *object, gpointer class)
{
	GSTLALMatrixMixer *element = GSTLAL_MATRIXMIXER(object);
	GstPad *pad;

	gst_element_create_all_pads(GST_ELEMENT(element));

	/* configure (and ref) sink pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "sink");
	gst_pad_set_setcaps_function(pad, setcaps);
	gst_pad_set_chain_function(pad, chain);
	element->sinkpad = pad;

	/* configure matrix pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "matrix");
	gst_pad_set_chain_function(pad, chain_matrix);
	gst_object_unref(pad);

	/* retrieve (and ref) src pad */
	element->srcpad = gst_element_get_static_pad(GST_ELEMENT(element), "src");

	/* internal data */
	element->V = NULL;
}


/*
 * gstlal_matrixmixer_get_type().
 */


GType gstlal_matrixmixer_get_type(void)
{
	static GType type = 0;

	if(!type) {
		static const GTypeInfo info = {
			.class_size = sizeof(GSTLALMatrixMixerClass),
			.class_init = class_init,
			.base_init = base_init,
			.instance_size = sizeof(GSTLALMatrixMixer),
			.instance_init = instance_init,
		};
		type = g_type_register_static(GST_TYPE_ELEMENT, "lal_matrixmixer", &info, 0);
	}

	return type;
}

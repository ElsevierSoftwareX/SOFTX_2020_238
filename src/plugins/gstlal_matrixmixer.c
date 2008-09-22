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
 * stuff from glib/gstreamer
 */


#include <glib.h>
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
	 * do we have a mixing matrix yet?
	 */

	g_mutex_lock(element->mixmatrix_lock);
	if(!element->mixmatrix_buf) {
		g_mutex_unlock(element->mixmatrix_lock);
		goto done;
	}

	/*
	 * check that the number of input channels matches the number of
	 * rows in the mixing matrix
	 */

	if(g_value_get_int(gst_structure_get_value(gst_caps_get_structure(caps, 0), "channels")) != (int) element->mixmatrix.matrix.size1) {
		g_mutex_unlock(element->mixmatrix_lock);
		result = FALSE;
		goto done;
	}

	/*
	 * set the number of output channels to the number of columns in
	 * the mixing matrix, and try forwarding the caps to next element
	 */

	gst_caps_set_simple(caps, "channels", G_TYPE_INT, element->mixmatrix.matrix.size2, NULL);
	g_mutex_unlock(element->mixmatrix_lock);
	result = gst_pad_set_caps(element->srcpad, caps);

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
	gsl_matrix_view input_channels;
	GstBuffer *srcbuf;
	GstFlowReturn result = GST_FLOW_OK;

	/*
	 * Make sure we have a mixing matrix, wait until we do.
	 */

	g_mutex_lock(element->mixmatrix_lock);
	if(!element->mixmatrix_buf) {
		g_cond_wait(element->mixmatrix_available, element->mixmatrix_lock);
		if(!element->mixmatrix_buf) {
			/* mixing matrix didn't get set.  probably means
			 * we're being disposed(). */
			g_mutex_unlock(element->mixmatrix_lock);
			GST_ERROR("no mixing matrix available");
			result = GST_FLOW_NOT_NEGOTIATED;
			goto done;
		}
	}

	/*
	 * Check the number of channels coming in, must be the same as the
	 * number of rows in the mixing matrix.
	 */

	if(g_value_get_int(gst_structure_get_value(gst_caps_get_structure(caps, 0), "channels")) != (int) element->mixmatrix.matrix.size1) {
		GST_ERROR("channel count mismatch:  mixing matrix requires %u channels, received buffer with %d", element->mixmatrix.matrix.size1, g_value_get_int(gst_structure_get_value(gst_caps_get_structure(caps, 0), "channels")));
		g_mutex_unlock(element->mixmatrix_lock);
		result = GST_FLOW_NOT_NEGOTIATED;
		goto done;
	}

	/*
	 * Wrap the incoming buffer in a GSL matrix view.
	 */

	input_channels = gsl_matrix_view_array((double *) GST_BUFFER_DATA(sinkbuf), GST_BUFFER_SIZE(sinkbuf) / sizeof(*element->mixmatrix.matrix.data) / element->mixmatrix.matrix.size1, element->mixmatrix.matrix.size1);

	if(input_channels.matrix.size1 * input_channels.matrix.size2 * sizeof(*element->mixmatrix.matrix.data) != GST_BUFFER_SIZE(sinkbuf)) {
		GST_ERROR("buffer size mismatch:  input buffer size not divisible by the channel count");
		g_mutex_unlock(element->mixmatrix_lock);
		result = GST_FLOW_NOT_NEGOTIATED;
		goto done;
	}

	/*
	 * Get a buffer from the downstream peer
	 */

	result = gst_pad_alloc_buffer(element->srcpad, GST_BUFFER_OFFSET(sinkbuf), input_channels.matrix.size1 * element->mixmatrix.matrix.size2 * sizeof(*element->mixmatrix.matrix.data), GST_PAD_CAPS(element->srcpad), &srcbuf);
	if(result != GST_FLOW_OK) {
		g_mutex_unlock(element->mixmatrix_lock);
		goto done;
	}

	/*
	 * Copy metadata
	 */

	gst_buffer_copy_metadata(srcbuf, sinkbuf, GST_BUFFER_COPY_FLAGS | GST_BUFFER_COPY_TIMESTAMPS);

	/*
	 * Only do the real work if the buffer isn't a gap.
	 */

	if(!GST_BUFFER_FLAG_IS_SET(sinkbuf, GST_BUFFER_FLAG_GAP)) {
		/*
		 * Wrap the outgoing buffer in a GSL matrix view.
		 */

		gsl_matrix_view output_channels = gsl_matrix_view_array((double *) GST_BUFFER_DATA(srcbuf), input_channels.matrix.size1, element->mixmatrix.matrix.size2);

		/*
		 * Mix input channels into output channels.
		 */

		gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, &input_channels.matrix, &element->mixmatrix.matrix, 0, &output_channels.matrix);
	} else
		memset(GST_BUFFER_DATA(srcbuf), 0, GST_BUFFER_SIZE(srcbuf));

	g_mutex_unlock(element->mixmatrix_lock);

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
	gst_caps_unref(caps);
	gst_buffer_unref(sinkbuf);
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
	rows = GST_BUFFER_SIZE(sinkbuf) / sizeof(*element->mixmatrix.matrix.data) / cols;
	if(rows * cols * sizeof(*element->mixmatrix.matrix.data) != GST_BUFFER_SIZE(sinkbuf)) {
		GST_ERROR("buffer size mismatch:  input buffer size not divisible by the channel count");
		gst_buffer_unref(sinkbuf);
		result = GST_FLOW_NOT_NEGOTIATED;
		goto done;
	}

	/*
	 * Replace the current matrix with the new one.
	 */

	g_mutex_lock(element->mixmatrix_lock);
	if(element->mixmatrix_buf)
		gst_buffer_unref(element->mixmatrix_buf);
	element->mixmatrix_buf = sinkbuf;
	element->mixmatrix = gsl_matrix_view_array((double *) GST_BUFFER_DATA(sinkbuf), rows, cols);
	g_cond_signal(element->mixmatrix_available);
	g_mutex_unlock(element->mixmatrix_lock);

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

	gst_object_unref(element->srcpad);
	element->srcpad = NULL;
	g_mutex_free(element->mixmatrix_lock);
	element->mixmatrix_lock = NULL;
	g_cond_free(element->mixmatrix_available);
	element->mixmatrix_available = NULL;
	if(element->mixmatrix_buf) {
		gst_buffer_unref(element->mixmatrix_buf);
		element->mixmatrix_buf = NULL;
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

	/* configure sink pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "sink");
	gst_pad_set_setcaps_function(pad, setcaps);
	gst_pad_set_chain_function(pad, chain);
	gst_object_unref(pad);

	/* configure matrix pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "matrix");
	gst_pad_set_chain_function(pad, chain_matrix);
	gst_object_unref(pad);

	/* retrieve (and ref) src pad */
	element->srcpad = gst_element_get_static_pad(GST_ELEMENT(element), "src");

	/* internal data */
	element->mixmatrix_lock = g_mutex_new();
	element->mixmatrix_available = g_cond_new();
	element->mixmatrix_buf = NULL;
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

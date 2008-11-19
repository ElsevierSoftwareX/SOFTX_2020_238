/*
 * A \Chi^{2} element for the inspiral pipeline.
 *
 * Copyright (C) 2008  Kipp Cannon, Chad Hanna
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation; either version 2 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
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


#include <complex.h>
#include <string.h>


/*
 * stuff from glib/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstcollectpads.h>


/*
 * stuff from GSL
 */


#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>


/*
 * our own stuff
 */


#include <gstlal.h>
#include <gstlalcollectpads.h>
#include <gstlal_chisquare.h>


/*
 * ========================================================================
 *
 *                             Utility Functions
 *
 * ========================================================================
 */


static int num_input_channels(const GSTLALChiSquare *element)
{
	return element->mixmatrix.matrix.size1;
}


static int num_output_channels(const GSTLALChiSquare *element)
{
	return element->mixmatrix.matrix.size2;
}


static size_t mixmatrix_element_size(const GSTLALChiSquare *element)
{
	return sizeof(*element->mixmatrix.matrix.data);
}


/*
 * ============================================================================
 *
 *                                    Caps
 *
 * ============================================================================
 */


/*
 * we can only accept caps that both ourselves and the downstream peer can
 * handle
 */


static GstCaps *getcaps_snr(GstPad *pad)
{
	GSTLALChiSquare *element = GSTLAL_CHISQUARE(GST_PAD_PARENT(pad));
	GstCaps *result, *peercaps, *sinkcaps;

	GST_OBJECT_LOCK(element);

	/*
	 * get the allowed caps from the downstream peer
	 */

	peercaps = gst_pad_peer_get_caps(element->srcpad);

	/*
	 * get our own allowed caps.  use the fixed caps function to avoid
	 * recursing back into this function.
	 */

	sinkcaps = gst_pad_get_fixed_caps_func(pad);

	/*
	 * if the peer has caps, intersect.  if the peer has no caps (or
	 * there is no peer), use the allowed caps of this sinkpad.
	 */

	if(peercaps) {
		GST_DEBUG_OBJECT(element, "intersecting peer and template caps");
		result = gst_caps_intersect(peercaps, sinkcaps);
		gst_caps_unref(peercaps);
		gst_caps_unref(sinkcaps);
	} else {
		GST_DEBUG_OBJECT(element, "no peer caps, using sinkcaps");
		result = sinkcaps;
	}

	/*
	 * done
	 */

	GST_OBJECT_UNLOCK(element);
	return result;
}


/*
 * when getting new caps, extract the sample rate and bytes/sample from the
 * caps
 */


static gboolean setcaps(GstPad *pad, GstCaps *caps)
{
	GSTLALChiSquare *element = GSTLAL_CHISQUARE(GST_PAD_PARENT(pad));
	GstStructure *structure;
	const char *media_type;
	gint width, channels;

	/*
	 * parse caps
	 */

	structure = gst_caps_get_structure(caps, 0);
	media_type = gst_structure_get_name(structure);
	gst_structure_get_int(structure, "rate", &element->rate);
	gst_structure_get_int(structure, "width", &width);
	gst_structure_get_int(structure, "channels", &channels);

	/*
	 * pre-calculate bytes / sample
	 */

	gstlal_collect_pads_set_bytes_per_sample(pad, (width / 8) * channels);

	/*
	 * done
	 */

	return TRUE;
}


/*
 * ============================================================================
 *
 *                            \Chi^{2} Computation
 *
 * ============================================================================
 */


static GstFlowReturn collected(GstCollectPads *pads, gpointer user_data)
{
	GSTLALChiSquare *element = GSTLAL_CHISQUARE(user_data);
	guint64 earliest_input_offset, earliest_input_offset_end;
	guint length;
	GstBuffer *buf;
	GstBuffer *orthosnrbuf;

	/*
	 * get the range of offsets (in the output stream) spanned by the
	 * available input buffers.
	 */

	if(!gstlal_collect_pads_get_earliest_offsets(element->collect, &earliest_input_offset, &earliest_input_offset_end, element->rate, element->output_timestamp_at_zero)) {
		GST_ERROR_OBJECT(element, "cannot deduce input timestamp offset information");
		return GST_FLOW_ERROR;
	}

	/*
	 * check for EOS
	 */

	if(earliest_input_offset == GST_BUFFER_OFFSET_NONE)
		goto eos;

	/*
	 * don't let time go backwards.  in principle we could be smart and
	 * handle this, but the audiorate element can be used to correct
	 * screwed up time series so there is no point in re-inventing its
	 * capabilities here.
	 */

	if(earliest_input_offset < element->output_offset) {
		GST_ERROR_OBJECT(element, "detected time reversal in at least one input stream:  expected nothing earlier than offset %llu, found sample at offset %llu", (unsigned long long) element->output_offset, (unsigned long long) earliest_input_offset);
		return GST_FLOW_ERROR;
	}

	/*
	 * compute the number of samples for which all sink pads can
	 * contribute information.  0 does not necessarily mean EOS.
	 */

	length = earliest_input_offset_end - earliest_input_offset;

	/*
	 * get buffers upto the desired end offset.
	 */

	buf = gstlal_collect_pads_take_buffer(pads, element->snrcollectdata, earliest_input_offset_end);
	orthosnrbuf = gstlal_collect_pads_take_buffer(pads, element->orthosnrcollectdata, earliest_input_offset_end);

	/*
	 * NULL means EOS.
	 */

	if(!buf && !orthosnrbuf) {
		/* FIXME:  handle EOS */
	}

	/*
	 * compute the \Chi^{2} values in-place in the input buffer
	 */

	/* FIXME:  do this */

	/*
	 * push the buffer downstream
	 */

	return gst_pad_push(element->srcpad, buf);

eos:
	GST_DEBUG_OBJECT(element, "no data available (EOS)");
	gst_pad_push_event(element->srcpad, gst_event_new_eos());
	return GST_FLOW_UNEXPECTED;
}


/*
 * ============================================================================
 *
 *                                Type Support
 *
 * ============================================================================
 */


/*
 * Parent class.
 */


static GstElementClass *parent_class = NULL;


/*
 * Instance finalize function.  See ???
 */


static void finalize(GObject *object)
{
	GSTLALChiSquare *element = GSTLAL_CHISQUARE(object);

	gst_object_unref(element->orthosnrpad);
	element->orthosnrpad = NULL;
	gst_object_unref(element->snrpad);
	element->snrpad = NULL;
	gst_object_unref(element->matrixpad);
	element->matrixpad = NULL;
	gst_object_unref(element->srcpad);
	element->srcpad = NULL;

	gst_object_unref(element->collect);
	element->orthosnrcollectdata = NULL;
	element->snrcollectdata = NULL;
	element->collect = NULL;

	g_mutex_free(element->mixmatrix_lock);
	element->mixmatrix_lock = NULL;
	g_cond_free(element->mixmatrix_available);
	element->mixmatrix_available = NULL;
	if(element->mixmatrix_buf) {
		gst_buffer_unref(element->mixmatrix_buf);
		element->mixmatrix_buf = NULL;
	}

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


/*
 * change state
 */


static GstStateChangeReturn change_state(GstElement * element, GstStateChange transition)
{
	GSTLALChiSquare *chisquare = GSTLAL_CHISQUARE(element);

	switch(transition) {
	case GST_STATE_CHANGE_READY_TO_PAUSED:
		chisquare->output_offset = 0;
		chisquare->output_timestamp_at_zero = 0;
		break;

	default:
		break;
	}

	return parent_class->change_state(element, transition);
}


/*
 * Base init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GBaseInitFunc
 */


static void base_init(gpointer class)
{
	static GstElementDetails plugin_details = {
		"Inspiral \\Chi^{2}",
		"Filter",
		"A \\Chi^{2} statistic for the inspiral pipeline",
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
			gst_caps_from_string(
				"audio/x-raw-float, " \
				"channels = (int) [ 1, MAX ], " \
				"endianness = (int) BYTE_ORDER, " \
				"width = (int) {32, 64} ; " \
				"audio/x-raw-complex, " \
				"channels = (int) [ 1, MAX ], " \
				"endianness = (int) BYTE_ORDER, " \
				"width = (int) {64, 128}"
			)
		)
	);
	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"orthosnr",
			GST_PAD_SINK,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
				"audio/x-raw-float, " \
				"rate = (int) [ 1, MAX ], " \
				"channels = (int) [ 1, MAX ], " \
				"endianness = (int) BYTE_ORDER, " \
				"width = (int) {32, 64} ; " \
				"audio/x-raw-complex, " \
				"rate = (int) [ 1, MAX ], " \
				"channels = (int) [ 1, MAX ], " \
				"endianness = (int) BYTE_ORDER, " \
				"width = (int) {64, 128}"
			)
		)
	);
	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"snr",
			GST_PAD_SINK,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
				"audio/x-raw-float, " \
				"rate = (int) [ 1, MAX ], " \
				"channels = (int) [ 1, MAX ], " \
				"endianness = (int) BYTE_ORDER, " \
				"width = (int) {32, 64} ; " \
				"audio/x-raw-complex, " \
				"rate = (int) [ 1, MAX ], " \
				"channels = (int) [ 1, MAX ], " \
				"endianness = (int) BYTE_ORDER, " \
				"width = (int) {64, 128}"
			)
		)
	);
	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"src",
			GST_PAD_SRC,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
				"audio/x-raw-float, " \
				"rate = (int) [ 1, MAX ], " \
				"channels = (int) [ 1, MAX ], " \
				"endianness = (int) BYTE_ORDER, " \
				"width = (int) {32, 64} ; " \
				"audio/x-raw-complex, " \
				"rate = (int) [ 1, MAX ], " \
				"channels = (int) [ 1, MAX ], " \
				"endianness = (int) BYTE_ORDER, " \
				"width = (int) {64, 128}"
			)
		)
	);
}


/*
 * Class init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GClassInitFunc
 */


static void class_init(gpointer klass, gpointer class_data)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *gstelement_class = GST_ELEMENT_CLASS(klass);

	parent_class = g_type_class_ref(GST_TYPE_ELEMENT);

	gobject_class->finalize = finalize;

	gstelement_class->change_state = change_state;
}


/*
 * Instance init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GInstanceInitFunc
 */


static void instance_init(GTypeInstance *object, gpointer class)
{
	GSTLALChiSquare *element = GSTLAL_CHISQUARE(object);
	GstPad *pad;

	gst_element_create_all_pads(GST_ELEMENT(element));
	element->collect = gst_collect_pads_new();
	gst_collect_pads_set_function(element->collect, collected, element);

	/* configure (and ref) matrix pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "matrix");
	/*gst_pad_set_setcaps_function(pad, setcaps_matrix);
	gst_pad_set_chain_function(pad, chain_matrix);*/
	element->matrixpad = pad;

	/* configure (and ref) orthogonal SNR sink pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "orthosnr");
	/*gst_pad_set_getcaps_function(pad, getcaps);*/
	gst_pad_set_setcaps_function(pad, setcaps);
	element->orthosnrcollectdata = gstlal_collect_pads_add_pad(element->collect, pad, sizeof(*element->orthosnrcollectdata));
	element->orthosnrpad = pad;

	/* configure (and ref) SNR sink pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "snr");
	gst_pad_set_getcaps_function(pad, getcaps_snr);
	gst_pad_set_setcaps_function(pad, setcaps);
	element->snrcollectdata = gstlal_collect_pads_add_pad(element->collect, pad, sizeof(*element->snrcollectdata));
	element->snrpad = pad;

	/* retrieve (and ref) src pad */
	element->srcpad = gst_element_get_static_pad(GST_ELEMENT(element), "src");

	/* internal data */
	element->rate = 0;
	element->mixmatrix_lock = g_mutex_new();
	element->mixmatrix_available = g_cond_new();
	element->mixmatrix_buf = NULL;
}


/*
 * gstlal_chisquare_get_type().
 */


GType gstlal_chisquare_get_type(void)
{
	static GType type = 0;

	if(!type) {
		static const GTypeInfo info = {
			.class_size = sizeof(GSTLALChiSquareClass),
			.class_init = class_init,
			.base_init = base_init,
			.instance_size = sizeof(GSTLALChiSquare),
			.instance_init = instance_init,
		};
		type = g_type_register_static(GST_TYPE_ELEMENT, "lal_chisquare", &info, 0);
	}

	return type;
}

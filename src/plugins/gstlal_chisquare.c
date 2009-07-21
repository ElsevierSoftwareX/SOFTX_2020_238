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
#include <math.h>
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
 *                                 Parameters
 *
 * ========================================================================
 */


/* FIXME: Hard coded (for now) max degrees of freedom.  Why a max of 10 you
 * might ask?  Well For inspiral analysis we usually have about 5 different
 * pieces of the waveform (give or take a few).  So computing a 10 degree
 * chisq test on each gives 50 degrees of freedom total.  The std dev of
 * that chisq distribution is sqrt(50) and can be compared to the SNR^2
 * that we are trying to distinguish from a glitch.  That means we can
 * begin to have discriminatory power at SNR = 50^(1/4) = 2.66 which is in
 * the bulk of the SNR distribution expected from Gaussian noise - exactly
 * where we want to be*/

#define DEFAULT_MAX_DOF 5


/*
 * ========================================================================
 *
 *                             Utility Functions
 *
 * ========================================================================
 */


static int num_orthosnr_channels(const GSTLALChiSquare *element)
{
	return element->mixmatrix.matrix.size1;
}


static int num_snr_channels(const GSTLALChiSquare *element)
{
	return element->mixmatrix.matrix.size2;
}


static size_t mixmatrix_element_size(const GSTLALChiSquare *element)
{
	/* evaluating this does not require the lock to be held */
	return sizeof(*element->mixmatrix.matrix.data);
}


static size_t chifacs_element_size(const GSTLALChiSquare *element)
{
	/* evaluating this does not require the lock to be held */
	return sizeof(*element->chifacs.vector.data);
}


/*
 * parse caps, and set bytes per sample on collect pads object
 */


static void set_bytes_per_sample(GstPad *pad, GstCaps *caps)
{
	GstStructure *structure;
	gint width, channels;

	structure = gst_caps_get_structure(caps, 0);
	gst_structure_get_int(structure, "width", &width);
	gst_structure_get_int(structure, "channels", &channels);

	gstlal_collect_pads_set_bytes_per_sample(pad, (width / 8) * channels);
}


/*
 * ============================================================================
 *
 *                              Caps --- SNR Pad
 *
 * ============================================================================
 */


/*
 * we can only accept caps that both ourselves and the downstream peer can
 * handle, and the number of channels must match the size of the mixing
 * matrix
 */


static GstCaps *getcaps_snr(GstPad *pad)
{
	GSTLALChiSquare *element = GSTLAL_CHISQUARE(gst_pad_get_parent(pad));
	GstCaps *peercaps, *caps;

	GST_OBJECT_LOCK(element);

	/*
	 * start by retrieving our own caps.  use get_fixed_caps_func() to
	 * avoid recursing back into this function.
	 */

	caps = gst_pad_get_fixed_caps_func(pad);

	/*
	 * if we have a mixing matrix the sink pad's media type and sample
	 * width must be the same as the mixing matrix's, and the number of
	 * channels must match the number of columns in the mixing matrix.
	 */

	if(element->mixmatrix_buf) {
		GstCaps *matrixcaps = gst_caps_make_writable(gst_buffer_get_caps(element->mixmatrix_buf));
		GstCaps *result;
		guint n;

		for(n = 0; n < gst_caps_get_size(matrixcaps); n++)
			gst_structure_set(gst_caps_get_structure(matrixcaps, n), "channels", G_TYPE_INT, num_snr_channels(element), NULL);
		result = gst_caps_intersect(matrixcaps, caps);
		gst_caps_unref(caps);
		gst_caps_unref(matrixcaps);
		caps = result;
	}

	/*
	 * now compute the intersection of the caps with the downstream
	 * peer's caps if known.
	 */

	peercaps = gst_pad_peer_get_caps(element->srcpad);
	if(peercaps) {
		GstCaps *result = gst_caps_intersect(peercaps, caps);
		gst_caps_unref(caps);
		gst_caps_unref(peercaps);
		caps = result;
	}

	/*
	 * done
	 */

	GST_OBJECT_UNLOCK(element);
	gst_object_unref(element);
	return caps;
}


/*
 * when setting new caps, extract the sample rate and bytes/sample from the
 * caps
 */


static gboolean setcaps_snr(GstPad *pad, GstCaps *caps)
{
	GSTLALChiSquare *element = GSTLAL_CHISQUARE(gst_pad_get_parent(pad));
	GstStructure *structure;
	const char *media_type;
	gboolean result = TRUE;

	GST_OBJECT_LOCK(element);

	/*
	 * set bytes per sample on collect pads object
	 */

	set_bytes_per_sample(pad, caps);

	/*
	 * extract rate
	 */

	structure = gst_caps_get_structure(caps, 0);
	media_type = gst_structure_get_name(structure);
	gst_structure_get_int(structure, "rate", &element->rate);

	/*
	 * done
	 */

	GST_OBJECT_UNLOCK(element);
	gst_object_unref(element);
	return result;
}


/*
 * ============================================================================
 *
 *                        Caps --- Orthogonal SNR Pad
 *
 * ============================================================================
 */


/*
 * we can only accept caps that both ourselves and the downstream peer can
 * handle, and the number of channels must match the size of the mixing
 * matrix
 */


static GstCaps *getcaps_orthosnr(GstPad *pad)
{
	GSTLALChiSquare *element = GSTLAL_CHISQUARE(gst_pad_get_parent(pad));
	GstCaps *result, *snrcaps, *caps;

	GST_OBJECT_LOCK(element);

	/*
	 * start by retrieving our own caps.  use get_fixed_caps_func() to
	 * avoid recursing back into this function.
	 */

	caps = gst_pad_get_fixed_caps_func(pad);

	/*
	 * now intersect with the SNR pad's caps, replacing the number of
	 * channels with the number of rows in the mixing matrix if known.
	 * Retrieving the SNR pad's caps in-turn performs an intersection
	 * with the down-stream peer so we don't have to do that here
	 */

	GST_OBJECT_UNLOCK(element);
	snrcaps = gst_pad_get_caps(element->snrpad);
	GST_OBJECT_LOCK(element);

	if(element->mixmatrix_buf) {
		guint n;
		for(n = 0; n < gst_caps_get_size(snrcaps); n++)
			gst_structure_set(gst_caps_get_structure(snrcaps, n), "channels", G_TYPE_INT, num_orthosnr_channels(element), NULL);
	}

	result = gst_caps_intersect(snrcaps, caps);
	gst_caps_unref(caps);
	gst_caps_unref(snrcaps);

	/*
	 * done
	 */

	GST_OBJECT_UNLOCK(element);
	gst_object_unref(element);
	return result;
}


/*
 * when setting new caps, extract the sample rate and bytes/sample from the
 * caps
 */


static gboolean setcaps_orthosnr(GstPad *pad, GstCaps *caps)
{
	GSTLALChiSquare *element = GSTLAL_CHISQUARE(gst_pad_get_parent(pad));

	GST_OBJECT_LOCK(element);

	/*
	 * set bytes per sample on collect pads object
	 */

	set_bytes_per_sample(pad, caps);

	/*
	 * extract rate
	 */

	/* FIXME:  check that it is allowed */

	/*
	 * done
	 */

	GST_OBJECT_UNLOCK(element);
	gst_object_unref(element);
	return TRUE;
}


/*
 * ============================================================================
 *
 *                                 Matrix Pad
 *
 * ============================================================================
 */


/*
 * chain()
 */


static GstFlowReturn chain_matrix(GstPad *pad, GstBuffer *sinkbuf)
{
	GSTLALChiSquare *element = GSTLAL_CHISQUARE(gst_pad_get_parent(pad));
	GstCaps *caps = gst_buffer_get_caps(sinkbuf);
	GstStructure *structure = gst_caps_get_structure(caps, 0);
	GstFlowReturn result = GST_FLOW_OK;
	gint rows, cols;

	GST_OBJECT_LOCK(element);

	/*
	 * get the matrix size
	 */

	gst_structure_get_int(structure, "channels", &cols);
	rows = GST_BUFFER_SIZE(sinkbuf) / mixmatrix_element_size(element) / cols;
	if(rows * cols * mixmatrix_element_size(element) != GST_BUFFER_SIZE(sinkbuf)) {
		GST_ERROR_OBJECT(pad, "buffer size mismatch:  input buffer size not divisible by the channel count");
		gst_buffer_unref(sinkbuf);
		result = GST_FLOW_NOT_NEGOTIATED;
		goto done;
	}

	/*
	 * replace the current matrix with the new one.  this consumes the
	 * reference.
	 */

	if(element->mixmatrix_buf)
		gst_buffer_unref(element->mixmatrix_buf);
	element->mixmatrix_buf = sinkbuf;
	element->mixmatrix = gsl_matrix_view_array((double *) GST_BUFFER_DATA(sinkbuf), rows, cols);
	g_cond_signal(element->coefficients_available);

	/* FIXME:  clear the caps on the snr and orthosnr pads to induce a
	 * caps negotiation sequence when the next buffers arrive in order
	 * to check the number of channels */

	/*
	 * done
	 */

done:
	gst_caps_unref(caps);
	GST_OBJECT_UNLOCK(element);
	gst_object_unref(element);
	return result;
}


/*
 * ============================================================================
 *
 *                                Chifacs Pad
 *
 * ============================================================================
 */


/*
 * chain()
 */


static GstFlowReturn chain_chifacs(GstPad *pad, GstBuffer *sinkbuf)
{
	GSTLALChiSquare *element = GSTLAL_CHISQUARE(gst_pad_get_parent(pad));
	GstCaps *caps = gst_buffer_get_caps(sinkbuf);
	GstStructure *structure = gst_caps_get_structure(caps, 0);
	GstFlowReturn result = GST_FLOW_OK;
	gint cols;

	GST_OBJECT_LOCK(element);

	/*
	 * get the vector size
	 */

	gst_structure_get_int(structure, "channels", &cols);
	if(cols * chifacs_element_size(element) != GST_BUFFER_SIZE(sinkbuf)) {
		GST_ERROR_OBJECT(pad, "buffer size mismatch:  input buffer size not divisible by the channel count");
		gst_buffer_unref(sinkbuf);
		result = GST_FLOW_NOT_NEGOTIATED;
		goto done;
	}

	/*
	 * replace the current chifacs vector with the new one.  this
	 * consumes the reference.
	 */

	if(element->chifacs_buf)
		gst_buffer_unref(element->chifacs_buf);
	element->chifacs_buf = sinkbuf;
	element->chifacs = gsl_vector_view_array((double *) GST_BUFFER_DATA(sinkbuf), cols);
	g_cond_signal(element->coefficients_available);

	/* FIXME:  clear the caps on the snr and orthosnr pads to induce a
	 * caps negotiation sequence when the next buffers arrive in order
	 * to check the number of channels;  for now we blindly assume
	 * everything's OK */

	/*
	 * done
	 */

done:
	gst_caps_unref(caps);
	GST_OBJECT_UNLOCK(element);
	gst_object_unref(element);
	return result;
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
	guint sample, length;
	GstBuffer *buf;
	GstBuffer *orthosnrbuf;
	gint dof;
	gint ortho_channel, numorthochannels;
	gint channel, numchannels;
	gint chisq_start, chisq_end, chisq_stride;

	GST_OBJECT_LOCK(element);

	/*
	 * check for new segment
	 */

	if(element->segment_pending) {
		GstEvent *event;
		GstSegment *segment = gstlal_collect_pads_get_segment(element->collect);
		if(!segment) {
			/* FIXME:  failure getting bounding segment, do
			 * something about it */
		}
		element->segment = *segment;
		gst_segment_free(segment);
		element->offset = GST_BUFFER_OFFSET_NONE;

		event = gst_event_new_new_segment_full(FALSE, element->segment.rate, 1.0, GST_FORMAT_TIME, element->segment.start, element->segment.stop, element->segment.start);
		if(!event) {
			/* FIXME:  failure getting event, do something
			 * about it */
		}
		GST_OBJECT_UNLOCK(element);
		gst_pad_push_event(element->srcpad, event);
		GST_OBJECT_LOCK(element);

		element->segment_pending = FALSE;
	}

	/*
	 * get the range of offsets (in the output stream) spanned by the
	 * available input buffers.
	 */

	if(!gstlal_collect_pads_get_earliest_offsets(element->collect, &earliest_input_offset, &earliest_input_offset_end, element->rate, element->segment.start)) {
		GST_ERROR_OBJECT(element, "cannot deduce input timestamp offset information");
		goto error;
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

	if((element->offset != GST_BUFFER_OFFSET_NONE) && (earliest_input_offset < element->offset)) {
		GST_ERROR_OBJECT(element, "detected time reversal in at least one input stream:  expected nothing earlier than offset %lu, found sample at offset %lu", element->offset, earliest_input_offset);
		goto error;
	}

	/*
	 * get buffers upto the desired end offset.
	 */

	buf = gstlal_collect_pads_take_buffer(pads, element->snrcollectdata, earliest_input_offset_end, element->rate, element->segment.start);
	orthosnrbuf = gstlal_collect_pads_take_buffer(pads, element->orthosnrcollectdata, earliest_input_offset_end, element->rate, element->segment.start);

	/*
	 * NULL means EOS.
	 */
 
	if(!buf || !orthosnrbuf) {
		/* FIXME:  handle EOS */
	}

	/*
	 * FIXME:  rethink the collect pads system so that this doesn't
	 * happen  (I think the second part already cannot happen because
	 * we get the collect pads system to tell us the upper bound of
	 * available offsets before asking for the buffers, but need to
	 * check this)
	 */

	if(GST_BUFFER_OFFSET(buf) != GST_BUFFER_OFFSET(orthosnrbuf) || GST_BUFFER_OFFSET_END(buf) != GST_BUFFER_OFFSET_END(orthosnrbuf)) {
		gst_buffer_unref(buf);
		gst_buffer_unref(orthosnrbuf);
		GST_ERROR_OBJECT(element, "misaligned buffer boundaries:  requested offsets upto %lu, got snr offsets %lu--%lu and ortho snr offsets %lu--%lu", earliest_input_offset_end, GST_BUFFER_OFFSET(buf), GST_BUFFER_OFFSET_END(buf), GST_BUFFER_OFFSET(orthosnrbuf), GST_BUFFER_OFFSET_END(orthosnrbuf));
		goto error;
	}

	/*
	 * in-place transform, buf must be writable
	 */

	buf = gst_buffer_make_writable(buf);

	/*
	 * check for discontinuity
	 */

	if((element->offset == GST_BUFFER_OFFSET_NONE) || (element->offset != GST_BUFFER_OFFSET(buf)))
		GST_BUFFER_FLAG_SET(buf, GST_BUFFER_FLAG_DISCONT);

	/*
	 * Gap --> pass-through
	 */

	if(GST_BUFFER_FLAG_IS_SET(buf, GST_BUFFER_FLAG_GAP) || GST_BUFFER_FLAG_IS_SET(orthosnrbuf, GST_BUFFER_FLAG_GAP)) {
		memset(GST_BUFFER_DATA(buf), 0, GST_BUFFER_SIZE(buf));
		GST_BUFFER_FLAG_SET(buf, GST_BUFFER_FLAG_GAP);
		goto done;
	}

	/*
	 * compute the number of samples in each channel
	 */

	length = GST_BUFFER_OFFSET_END(buf) - GST_BUFFER_OFFSET(buf);

	/*
	 * make sure the mix matrix and chifacs vectors are available, wait
	 * until they are
	 */

	while(!element->mixmatrix_buf || !element->chifacs_buf) {
		g_cond_wait(element->coefficients_available, GST_OBJECT_GET_LOCK(element));
		/* FIXME:  we need some way of getting out of this loop.
		 * maybe check for a flag set in an event handler */
	}

	/*
	 * compute the \Chi^{2} values in-place in the input buffer
	 */
	/* FIXME:  Assumes that the most important basis vectors are at the
	 * beginning;  this is a sensible assumption */
	/* FIXME: do with gsl functions?? */

	numorthochannels = (guint) num_orthosnr_channels(element);
	numchannels = (guint) num_snr_channels(element);
	dof = (numorthochannels < element->max_dof) ? numorthochannels : element->max_dof;

	/* FIXME: using this as the start in the for loop tests the "least"
	 * important basis vectors, this is maybe the right thing?
	 */ 	
	/* chisq_start = (numorthochannels - dof < 0 ) ? 0 : numorthochannels - dof; */
	/* chisq_end = numorthochannels */

	/* This assumes you want the top dof basis vectors in the test */
	chisq_start = 0;
	chisq_end = numorthochannels;
	/* okay because of conditional on setting dof to be no more 
	 * than the number of orthonormal channels
	 */
	chisq_stride = numorthochannels / dof;

	for(sample = 0; sample < length; sample++) {
		double *data = &((double *) GST_BUFFER_DATA(buf))[numchannels * sample];
		const double *orthodata = &((const double *) GST_BUFFER_DATA(orthosnrbuf))[numorthochannels * sample];
		for(channel = 0; channel < numchannels; channel += 2) {
			complex double csnr = data[channel] + I * data[channel + 1];
			double snr = cabs(csnr);
			double arg = carg(csnr);
			double cos_arg = cos(arg);
			double sin_arg = sin(arg);

			data[channel] = 0;
			for(ortho_channel = chisq_start; ortho_channel < chisq_end; ortho_channel+=chisq_stride) {
				double mixing_coefficient_re = gsl_matrix_get(&element->mixmatrix.matrix, ortho_channel, channel);
				double mixing_coefficient_im = gsl_matrix_get(&element->mixmatrix.matrix, ortho_channel, channel+1);
				double mc = mixing_coefficient_re * cos_arg + mixing_coefficient_im * sin_arg;
				data[channel] += pow((snr * mc - orthodata[ortho_channel]), 2.0);
			}
			data[channel+1] = data[channel];
		}
	}

	/*
	 * push the buffer downstream
	 */

done:
	gst_buffer_unref(orthosnrbuf);
	element->offset = GST_BUFFER_OFFSET_END(buf);
	GST_OBJECT_UNLOCK(element);
	return gst_pad_push(element->srcpad, buf);

eos:
	GST_DEBUG_OBJECT(element, "no data available (EOS)");
	GST_OBJECT_UNLOCK(element);
	gst_pad_push_event(element->srcpad, gst_event_new_eos());
	return GST_FLOW_UNEXPECTED;

error:
	GST_OBJECT_UNLOCK(element);
	return GST_FLOW_ERROR;
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
	gst_object_unref(element->chifacspad);
	element->chifacspad = NULL;
	gst_object_unref(element->srcpad);
	element->srcpad = NULL;

	gst_object_unref(element->collect);
	element->orthosnrcollectdata = NULL;
	element->snrcollectdata = NULL;
	element->collect = NULL;

	g_cond_free(element->coefficients_available);
	element->coefficients_available = NULL;
	if(element->mixmatrix_buf) {
		gst_buffer_unref(element->mixmatrix_buf);
		element->mixmatrix_buf = NULL;
	}
	if(element->chifacs_buf) {
		gst_buffer_unref(element->chifacs_buf);
		element->chifacs_buf = NULL;
	}

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


/*
 * change state.  reset element's internal state and start the collect pads
 * on READY --> PAUSED state change.  stop the collect pads on PAUSED -->
 * READY state change.
 */


static GstStateChangeReturn change_state(GstElement *element, GstStateChange transition)
{
	GSTLALChiSquare *chisquare = GSTLAL_CHISQUARE(element);

	switch(transition) {
	case GST_STATE_CHANGE_NULL_TO_READY:
		break;

	case GST_STATE_CHANGE_READY_TO_PAUSED:
		chisquare->segment_pending = TRUE;
		gst_segment_init(&chisquare->segment, GST_FORMAT_UNDEFINED);
		chisquare->offset = GST_BUFFER_OFFSET_NONE;
		gst_collect_pads_start(chisquare->collect);
		break;

	case GST_STATE_CHANGE_PAUSED_TO_PLAYING:
		break;

	case GST_STATE_CHANGE_PAUSED_TO_READY:
		/* need to unblock the collectpads before calling the
		 * parent change_state so that streaming can finish */
		gst_collect_pads_stop(chisquare->collect);
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
		"Kipp Cannon <kcannon@ligo.caltech.edu>, Chad Hanna <channa@ligo.caltech.edu>"
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
				"width = (int) 64"
			)
		)
	);
	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"chifacs",
			GST_PAD_SINK,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
				"audio/x-raw-float, " \
				"channels = (int) [ 1, MAX ], " \
				"endianness = (int) BYTE_ORDER, " \
				"width = (int) 64"
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
				"width = (int) 64"
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
				"width = (int) 64"
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
				"width = (int) 64"
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
	gst_pad_set_chain_function(pad, chain_matrix);
	element->matrixpad = pad;

	/* configure (and ref) chifacs pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "chifacs");
	gst_pad_set_chain_function(pad, chain_chifacs);
	element->chifacspad = pad;

	/* configure (and ref) orthogonal SNR sink pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "orthosnr");
	gst_pad_set_getcaps_function(pad, getcaps_orthosnr);
	gst_pad_set_setcaps_function(pad, setcaps_orthosnr);
	element->orthosnrcollectdata = gstlal_collect_pads_add_pad(element->collect, pad, sizeof(*element->orthosnrcollectdata));
	element->orthosnrpad = pad;

	/* configure (and ref) SNR sink pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "snr");
	gst_pad_set_getcaps_function(pad, getcaps_snr);
	gst_pad_set_setcaps_function(pad, setcaps_snr);
	element->snrcollectdata = gstlal_collect_pads_add_pad(element->collect, pad, sizeof(*element->snrcollectdata));
	element->snrpad = pad;

	/* retrieve (and ref) src pad */
	element->srcpad = gst_element_get_static_pad(GST_ELEMENT(element), "src");

	/* internal data */
	element->rate = 0;
	element->max_dof = DEFAULT_MAX_DOF;
	element->coefficients_available = g_cond_new();
	element->mixmatrix_buf = NULL;
	element->chifacs_buf = NULL;
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

/*
 * A time-slice-based \chi^{2} element for the inspiral pipeline.
 *
 * Copyright (C) 2011  Kipp Cannon, Chad Hanna, Drew Keppel
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
#include <gstlal_timeslicechisq.h>


/*
 * ========================================================================
 *
 *                             Utility Functions
 *
 * ========================================================================
 */


static int num_channels(const GSTLALTimeSliceChiSquare *element)
{
	return element->chifacs->size;
}


/*
 * ============================================================================
 *
 *                           GStreamer Boiler Plate
 *
 * ============================================================================
 */


enum property {
	ARG_0,
	ARG_CHIFACS
};


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
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(gst_pad_get_parent(pad));
	GstCaps *peercaps, *caps;

	/*
	 * start by retrieving our own caps.  use get_fixed_caps_func() to
	 * avoid recursing back into this function.
	 */

	GST_OBJECT_LOCK(element);
	caps = gst_pad_get_fixed_caps_func(pad);

	/*
	 * intersect with the downstream peer's caps if known.
	 */

	peercaps = gst_pad_peer_get_caps(element->srcpad);
	if(peercaps) {
		GstCaps *result = gst_caps_intersect(peercaps, caps);
		gst_caps_unref(caps);
		gst_caps_unref(peercaps);
		caps = result;
	}
	GST_OBJECT_UNLOCK(element);

	/*
	 * done
	 */

	gst_object_unref(element);
	return caps;
}


/*
 * when setting new caps, extract the sample rate and bytes/sample from the
 * caps
 */


static gboolean setcaps_snr(GstPad *pad, GstCaps *caps)
{
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(gst_pad_get_parent(pad));
	GstStructure *structure;
	gint rate, width, channels;
	gboolean success = TRUE;

	GST_LOG_OBJECT(element, "setting caps on pad %p,%s to %" GST_PTR_FORMAT, pad, GST_PAD_NAME (pad), caps);

	/*
	 * parse the caps
	 */

	/* FIXME, see if the timeslicessnrpad can accept the format. Also lock the
	 * format on the other pads to this new format. */
//	GST_OBJECT_LOCK(element);
//	gst_caps_replace(&GST_PAD_CAPS(element->timeslicesnrpad), caps);
//	GST_OBJECT_UNLOCK(element);

	GST_DEBUG_OBJECT(element, "(%s) trying %" GST_PTR_FORMAT "\n", GST_PAD_NAME(pad), caps);
	structure = gst_caps_get_structure(caps, 0);
	if(!gst_structure_get_int(structure, "rate", &rate))
		success = FALSE;
	if(!gst_structure_get_int(structure, "width", &width))
		success = FALSE;
	if(!gst_structure_get_int(structure, "channels", &channels))
		success = FALSE;


	/*
	 * if we have a chifacs, the number of channels must match the length
	 * of chifacs.
	 */

	g_mutex_lock(element->coefficients_lock);
	if(element->chifacs && (channels != (gint) num_channels(element))) {
		GST_DEBUG_OBJECT(element, "channels != %d in %" GST_PTR_FORMAT, num_channels(element), caps);
		success = FALSE;
	}
	g_mutex_unlock(element->coefficients_lock);

	/*
	 * will the downstream peer will accept the caps?  (the output
	 * stream has the same caps as the SNR input stream)
	 */

	if(success) {
		GST_DEBUG_OBJECT(element, "(%s) trying to set caps %" GST_PTR_FORMAT " on downstream peer\n", GST_PAD_NAME(pad), caps);
		success = gst_pad_set_caps(element->srcpad, caps);
		GST_DEBUG_OBJECT(element, "(%s) %s\n", GST_PAD_NAME(pad), success ? "accepted" : "rejected");
	}

	/*
	 * if that was successful, update our metadata
	 */

	if(success) {
		GST_OBJECT_LOCK(element);
		gstlal_collect_pads_set_unit_size(pad, (width / 8) * channels);
		element->rate = rate;
		GST_OBJECT_UNLOCK(element);
	}

	/*
	 * done
	 */

	gst_object_unref(element);
	return success;
}


/*
 * ============================================================================
 *
 *                        Caps --- Time-Slice SNR Pad
 *
 * ============================================================================
 */


/*
 * we can only accept caps that both ourselves and the downstream peer can
 * handle, and the number of channels must match the size of the chifacs
 */


static GstCaps *getcaps_timeslicesnr(GstPad *pad)
{
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(gst_pad_get_parent(pad));
	GstCaps *peercaps, *caps;

	/*
	 * start by retrieving our own caps.  use get_fixed_caps_func() to
	 * avoid recursing back into this function.
	 */

	GST_OBJECT_LOCK(element);
	caps = gst_pad_get_fixed_caps_func(pad);

	/*
	 * intersect with the downstream peer's caps if known.
	 */

	peercaps = gst_pad_peer_get_caps(element->srcpad);
	if(peercaps) {
		GstCaps *result;
		guint n;

		for(n = 0; n < gst_caps_get_size(peercaps); n++)
			gst_structure_remove_field(gst_caps_get_structure(peercaps, n), "channels");
		result = gst_caps_intersect(peercaps, caps);
		gst_caps_unref(caps);
		gst_caps_unref(peercaps);
		caps = result;
	}
	GST_OBJECT_UNLOCK(element);

	/*
	 * done
	 */

	gst_object_unref(element);
	return caps;
}


/*
 * when setting new caps, extract the sample rate and bytes/sample from the
 * caps
 */


static gboolean setcaps_timeslicesnr(GstPad *pad, GstCaps *caps)
{
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(gst_pad_get_parent(pad));
	GstStructure *structure;
	gint rate, width, channels;
	gboolean success = TRUE;

	/*
	 * parse the caps
	 */

	/* FIXME, see if the timeslicessnrpad can accept the format. Also lock the
	 * format on the other pads to this new format. */
//	GST_OBJECT_LOCK(element);
//	gst_caps_replace(&GST_PAD_CAPS(element->snrpad), caps);
//	GST_OBJECT_UNLOCK(element);

	GST_DEBUG_OBJECT(element, "(%s) trying %" GST_PTR_FORMAT "\n", GST_PAD_NAME(pad), caps);
	structure = gst_caps_get_structure(caps, 0);
	if(!gst_structure_get_int(structure, "rate", &rate))
		success = FALSE;
	if(!gst_structure_get_int(structure, "width", &width))
		success = FALSE;
	if(!gst_structure_get_int(structure, "channels", &channels))
		success = FALSE;

	/*
	 * if we have a chifacs, the number of channels must match the length
	 * of chifacs.
	 */

	g_mutex_lock(element->coefficients_lock);
	if(element->chifacs && (channels != (gint) num_channels(element))) {
		GST_DEBUG_OBJECT(element, "channels != %d in %" GST_PTR_FORMAT, num_channels(element), caps);
		success = FALSE;
	}
	g_mutex_unlock(element->coefficients_lock);

	/*
	 * if everything OK, update our metadata
	 */

	if(success) {
		GST_OBJECT_LOCK(element);
		gstlal_collect_pads_set_unit_size(pad, (width / 8) * channels);
		GST_OBJECT_UNLOCK(element);
	}

	/*
	 * done
	 */

	gst_object_unref(element);
	return success;
}


/*
 * ============================================================================
 *
 *                            \chi^{2} Computation
 *
 * ============================================================================
 */


static GstFlowReturn collected(GstCollectPads *pads, gpointer user_data)
{
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(user_data);
	GstClockTime earliest_input_t_start, earliest_input_t_end;
	guint sample, length;
	GstBuffer *buf = NULL;
	GstBuffer *timeslicesnrbuf = NULL;
	gint timeslice_channel;
	gint channel, numchannels;

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
		element->offset = 0;
		gst_segment_free(segment);

		event = gst_event_new_new_segment_full(FALSE, element->segment.rate, 1.0, GST_FORMAT_TIME, element->segment.start, element->segment.stop, element->segment.start);
		if(!event) {
			/* FIXME:  failure getting event, do something
			 * about it */
		}
		gst_pad_push_event(element->srcpad, event);

		element->segment_pending = FALSE;
	}

	/*
	 * get the range of offsets (in the output stream) spanned by the
	 * available input buffers.
	 */

	if(!gstlal_collect_pads_get_earliest_times(element->collect, &earliest_input_t_start, &earliest_input_t_end, element->rate)) {
		GST_ERROR_OBJECT(element, "cannot deduce input timestamp offset information");
		goto error;
	}

	/*
	 * check for EOS
	 */

	if(!GST_CLOCK_TIME_IS_VALID(earliest_input_t_start))
		goto eos;

	/*
	 * get buffers upto the desired end offset.
	 */

	buf = gstlal_collect_pads_take_buffer_sync(pads, element->snrcollectdata, earliest_input_t_end, element->rate);
	timeslicesnrbuf = gstlal_collect_pads_take_buffer_sync(pads, element->timeslicesnrcollectdata, earliest_input_t_end, element->rate);

	if(!buf || !timeslicesnrbuf) {
		/*
		 * NULL means EOS.
		 */
		if(buf)
			gst_buffer_unref(buf);
		if(timeslicesnrbuf)
			gst_buffer_unref(timeslicesnrbuf);
		goto eos;
	}

	buf = gst_buffer_make_metadata_writable(buf);
	GST_BUFFER_OFFSET(buf) = gst_util_uint64_scale_int_round(GST_BUFFER_TIMESTAMP(buf) - element->segment.start, element->rate, GST_SECOND);
	GST_BUFFER_OFFSET_END(buf) = gst_util_uint64_scale_int_round(GST_BUFFER_TIMESTAMP(buf) + GST_BUFFER_DURATION(buf) - element->segment.start, element->rate, GST_SECOND);
	timeslicesnrbuf = gst_buffer_make_metadata_writable(timeslicesnrbuf);
	GST_BUFFER_OFFSET(timeslicesnrbuf) = gst_util_uint64_scale_int_round(GST_BUFFER_TIMESTAMP(timeslicesnrbuf) - element->segment.start, element->rate, GST_SECOND);
	GST_BUFFER_OFFSET_END(timeslicesnrbuf) = gst_util_uint64_scale_int_round(GST_BUFFER_TIMESTAMP(timeslicesnrbuf) + GST_BUFFER_DURATION(timeslicesnrbuf) - element->segment.start, element->rate, GST_SECOND);

	/*
	 * Check for mis-aligned input buffers.  This can happen, but we
	 * can't handle it.
	 */

	if(GST_BUFFER_OFFSET(buf) != GST_BUFFER_OFFSET(timeslicesnrbuf) || GST_BUFFER_OFFSET_END(buf) != GST_BUFFER_OFFSET_END(timeslicesnrbuf)) {
		GST_ERROR_OBJECT(element, "misaligned buffer boundaries:  got snr offsets [%" G_GUINT64_FORMAT ", %" G_GUINT64_FORMAT ") and time-slice snr offsets [%" G_GUINT64_FORMAT ", %" G_GUINT64_FORMAT ")", GST_BUFFER_OFFSET(buf), GST_BUFFER_OFFSET_END(buf), GST_BUFFER_OFFSET(timeslicesnrbuf), GST_BUFFER_OFFSET_END(timeslicesnrbuf));
		goto error;
	}

	/*
	 * don't let time go backwards.  in principle we could be smart and
	 * handle this, but the audiorate element can be used to correct
	 * screwed up time series so there is no point in re-inventing its
	 * capabilities here.
	 */

	if(GST_BUFFER_OFFSET(buf) < element->offset) {
		GST_ERROR_OBJECT(element, "detected time reversal in at least one input stream:  expected nothing earlier than offset %" G_GUINT64_FORMAT ", found sample at offset %" G_GUINT64_FORMAT, element->offset, GST_BUFFER_OFFSET(buf));
		goto error;
	}

	/*
	 * in-place transform, buf must be writable
	 */

	buf = gst_buffer_make_writable(buf);

	/*
	 * check for discontinuity
	 */

	if(element->offset != GST_BUFFER_OFFSET(buf))
		GST_BUFFER_FLAG_SET(buf, GST_BUFFER_FLAG_DISCONT);

	/*
	 * Gap --> pass-through
	 */

	if(GST_BUFFER_FLAG_IS_SET(buf, GST_BUFFER_FLAG_GAP) || GST_BUFFER_FLAG_IS_SET(timeslicesnrbuf, GST_BUFFER_FLAG_GAP)) {
		memset(GST_BUFFER_DATA(buf), 0, GST_BUFFER_SIZE(buf));
		GST_BUFFER_FLAG_SET(buf, GST_BUFFER_FLAG_GAP);
		goto done;
	}

	/*
	 * compute the number of samples in each channel
	 */

	length = GST_BUFFER_OFFSET_END(buf) - GST_BUFFER_OFFSET(buf);

	/*
	 * make sure the chifacs vectors is available, wait until it is
	 */

	g_mutex_lock(element->coefficients_lock);
	while(!element->chifacs) {
		g_cond_wait(element->coefficients_available, element->coefficients_lock);
		/* FIXME:  we need some way of getting out of this loop.
		 * maybe check for a flag set in an event handler */
	}

	numchannels = (guint) num_channels(element);

	for(sample = 0; sample < length; sample++) {
		double *data = &((double *) GST_BUFFER_DATA(buf))[numchannels * sample];
		const double *timeslicedata = &((const double *) GST_BUFFER_DATA(timeslicesnrbuf))[numchannels * sample];
		for(channel = 0; channel < numchannels; channel += 1) {
			double snr = data[channel];

			data[channel] = 0;
			for(timeslice_channel = 0; timeslice_channel < numchannels; timeslice_channel+=1) {
				double chifacs_coefficient = gsl_vector_get(element->chifacs, timeslice_channel);
				double chifacs_coefficient2 = chifacs_coefficient*chifacs_coefficient;
				double chifacs_coefficient3 = chifacs_coefficient2*chifacs_coefficient;

				data[channel] += pow(snr * chifacs_coefficient - timeslicedata[timeslice_channel], 2.0)/2.0/(chifacs_coefficient2 - chifacs_coefficient3);
			}
			data[channel+1] = data[channel];
		}
	}
	g_mutex_unlock(element->coefficients_lock);

	/*
	 * push the buffer downstream
	 */

done:
	gst_buffer_unref(timeslicesnrbuf);
	element->offset = GST_BUFFER_OFFSET_END(buf);
	return gst_pad_push(element->srcpad, buf);

eos:
	GST_DEBUG_OBJECT(element, "no data available (EOS)");
	gst_pad_push_event(element->srcpad, gst_event_new_eos());
	return GST_FLOW_UNEXPECTED;

error:
	if(buf)
		gst_buffer_unref(buf);
	if(timeslicesnrbuf)
		gst_buffer_unref(timeslicesnrbuf);
	return GST_FLOW_ERROR;
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
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_CHIFACS: {
		int channels;
		g_mutex_lock(element->coefficients_lock);
		if(element->chifacs) {
			channels = num_channels(element);
			gsl_vector_free(element->chifacs);
		} else
			channels = 0;
		element->chifacs = gstlal_gsl_vector_from_g_value_array(g_value_get_boxed(value));

		/*
		 * number of channels has changed, force a caps
		 * renegotiation
		 */

		if(num_channels(element) != channels) {
			/* FIXME:  what do we do here? */
		}

		/*
		 * signal availability of new chifacs vector
		 */

		g_cond_broadcast(element->coefficients_available);
		g_mutex_unlock(element->coefficients_lock);
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
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_CHIFACS:
		g_mutex_lock(element->coefficients_lock);
		if(element->chifacs)
			g_value_take_boxed(value, gstlal_g_value_array_from_gsl_vector(element->chifacs));
		/* FIXME:  else? */
		g_mutex_unlock(element->coefficients_lock);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
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
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(object);

	gst_object_unref(element->timeslicesnrpad);
	element->timeslicesnrpad = NULL;
	gst_object_unref(element->snrpad);
	element->snrpad = NULL;
	gst_object_unref(element->srcpad);
	element->srcpad = NULL;

	gst_object_unref(element->collect);
	element->timeslicesnrcollectdata = NULL;
	element->snrcollectdata = NULL;
	element->collect = NULL;

	g_mutex_free(element->coefficients_lock);
	element->coefficients_lock = NULL;
	g_cond_free(element->coefficients_available);
	element->coefficients_available = NULL;
	if(element->chifacs) {
		gsl_vector_free(element->chifacs);
		element->chifacs = NULL;
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
	GSTLALTimeSliceChiSquare *timeslicechisquare = GSTLAL_TIMESLICECHISQUARE(element);

	switch(transition) {
	case GST_STATE_CHANGE_NULL_TO_READY:
		break;

	case GST_STATE_CHANGE_READY_TO_PAUSED:
		timeslicechisquare->segment_pending = TRUE;
		gst_segment_init(&timeslicechisquare->segment, GST_FORMAT_UNDEFINED);
		timeslicechisquare->offset = 0;
		gst_collect_pads_start(timeslicechisquare->collect);
		break;

	case GST_STATE_CHANGE_PAUSED_TO_PLAYING:
		break;

	case GST_STATE_CHANGE_PAUSED_TO_READY:
		/* need to unblock the collectpads before calling the
		 * parent change_state so that streaming can finish */
		gst_collect_pads_stop(timeslicechisquare->collect);
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
	GstElementClass *element_class = GST_ELEMENT_CLASS(class);

	gst_element_class_set_details_simple(
		element_class,
		"Inspiral time-slice-based \\chi^{2}",
		"Filter",
		"A time-slice-based \\chi^{2} statistic for the inspiral pipeline",
		"Kipp Cannon <kipp.cannon@ligo.org>, Chad Hanna <channa@ligo.caltech.edu>, Drew Keppel <drew.keppel@ligo.org>"
	);
	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"timeslicesnr",
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

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	gstelement_class->change_state = GST_DEBUG_FUNCPTR(change_state);

	g_object_class_install_property(
		gobject_class,
		ARG_CHIFACS,
		g_param_spec_value_array(
			"chifacs",
			"Chisquared Factors",
			"Vector of chisquared factors. Number of rows sets number of channels.",
			g_param_spec_double(
				"sample",
				"Sample",
				"Chifacs sample",
				-G_MAXDOUBLE, G_MAXDOUBLE, 0.0,
				G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
			),
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);

}


/*
 * Instance init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GInstanceInitFunc
 */


static void instance_init(GTypeInstance *object, gpointer class)
{
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(object);
	GstPad *pad;

	gst_element_create_all_pads(GST_ELEMENT(element));
	element->collect = gst_collect_pads_new();
	gst_collect_pads_set_function(element->collect, GST_DEBUG_FUNCPTR(collected), element);

	/* configure (and ref) timeslice SNR sink pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "timeslicesnr");
	gst_pad_set_getcaps_function(pad, GST_DEBUG_FUNCPTR(getcaps_timeslicesnr));
	gst_pad_set_setcaps_function(pad, GST_DEBUG_FUNCPTR(setcaps_timeslicesnr));
	element->timeslicesnrcollectdata = gstlal_collect_pads_add_pad(element->collect, pad, sizeof(*element->timeslicesnrcollectdata));
	element->timeslicesnrpad = pad;

	/* configure (and ref) SNR sink pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "snr");
	gst_pad_set_getcaps_function(pad, GST_DEBUG_FUNCPTR(getcaps_snr));
	gst_pad_set_setcaps_function(pad, GST_DEBUG_FUNCPTR(setcaps_snr));
	element->snrcollectdata = gstlal_collect_pads_add_pad(element->collect, pad, sizeof(*element->snrcollectdata));
	element->snrpad = pad;

	/* retrieve (and ref) src pad */
	element->srcpad = gst_element_get_static_pad(GST_ELEMENT(element), "src");

	/* internal data */
	element->rate = 0;
	element->coefficients_lock = g_mutex_new();
	element->coefficients_available = g_cond_new();
	element->chifacs = NULL;
}


/*
 * gstlal_timeslicechisquare_get_type().
 */


GType gstlal_timeslicechisquare_get_type(void)
{
	static GType type = 0;

	if(!type) {
		static const GTypeInfo info = {
			.class_size = sizeof(GSTLALTimeSliceChiSquareClass),
			.class_init = class_init,
			.base_init = base_init,
			.instance_size = sizeof(GSTLALTimeSliceChiSquare),
			.instance_init = instance_init,
		};
		type = g_type_register_static(GST_TYPE_ELEMENT, "lal_timeslicechisquare", &info, 0);
	}

	return type;
}

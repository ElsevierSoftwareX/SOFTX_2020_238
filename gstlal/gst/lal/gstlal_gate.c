/*
 * An element to flag buffers in a stream as silence or not based on the
 * value of a control input.
 *
 * Copyright (C) 2008-2012  Kipp Cannon, Chad Hanna
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
#include <stdlib.h>
#include <string.h>


/*
 * stuff from glib/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>


/*
 * our own stuff
 */


#include <gstlal_debug.h>
#include <gstlal_gate.h>


GST_DEBUG_CATEGORY(gstlal_gate_debug);


/*
 * ============================================================================
 *
 *                                 Parameters
 *
 * ============================================================================
 */


#define GST_CAT_DEFAULT gstlal_gate_debug
#define DEFAULT_EMIT_SIGNALS FALSE
#define DEFAULT_DEFAULT_STATE FALSE
#define DEFAULT_THRESHOLD 0
#define DEFAULT_ATTACK_LENGTH 0
#define DEFAULT_HOLD_LENGTH 0
#define DEFAULT_LEAKY FALSE
#define DEFAULT_INVERT FALSE


/*
 * ============================================================================
 *
 *                             Utility Functions
 *
 * ============================================================================
 */


/*
 * add sample counts to a timestamp
 */


static GstClockTime timestamp_add_offset(GstClockTime t, gint64 offset, gint rate)
{
	if(offset < 0) {
		GstClockTime dt = gst_util_uint64_scale_int_round(-offset, GST_SECOND, rate);
		/* don't allow wrap-around */
		return t < dt ? 0 : t - dt;
	}
	return t + gst_util_uint64_scale_int_round(offset, GST_SECOND, rate);
}


/*
 * array type cast macros:  interpret contents of an array as various C
 * types, retrieve value at given offset, compute magnitude, and cast to
 * double-precision float
 */


static gdouble control_sample_int8(const gpointer data, guint64 offset)
{
	return abs(((const gint8 *) data)[offset]);
}


static gdouble control_sample_uint8(const gpointer data, guint64 offset)
{
	return ((const guint8 *) data)[offset];
}


static gdouble control_sample_int16(const gpointer data, guint64 offset)
{
	return abs(((const gint16 *) data)[offset]);
}


static gdouble control_sample_uint16(const gpointer data, guint64 offset)
{
	return ((const guint16 *) data)[offset];
}


static gdouble control_sample_int32(const gpointer data, guint64 offset)
{
	return abs(((const gint32 *) data)[offset]);
}


static gdouble control_sample_uint32(const gpointer data, guint64 offset)
{
	return ((const guint32 *) data)[offset];
}


static gdouble control_sample_float32(const gpointer data, guint64 offset)
{
	return fabsf(((const float *) data)[offset]);
}


static gdouble control_sample_float64(const gpointer data, guint64 offset)
{
	return fabs(((const double *) data)[offset]);
}


static gdouble control_sample_complex64(const gpointer data, guint64 offset)
{
	return cabsf(((const float complex *) data)[offset]);
}


static gdouble control_sample_complex128(const gpointer data, guint64 offset)
{
	return cabs(((const double complex *) data)[offset]);
}


/*
 * ============================================================================
 *
 *                            Control Buffer Queue
 *
 * ============================================================================
 */


/*
 * add a segment to the control segment array.  note they are appended, and
 * the code assumes they are in order and do not overlap
 */


struct control_segment {
	GstClockTime start, stop;
	gboolean state;
};


static void control_add_segment(GSTLALGate *element, GstClockTime start, GstClockTime stop, gboolean state)
{
	struct control_segment new_segment = {
		.start = start,
		.stop = stop,
		.state = element->invert_control ? !state : state
	};

	g_assert_cmpuint(start, <=, stop);
	GST_DEBUG_OBJECT(element, "found control segment [%" GST_TIME_SECONDS_FORMAT ", %" GST_TIME_SECONDS_FORMAT ") in state %d", GST_TIME_SECONDS_ARGS(new_segment.start), GST_TIME_SECONDS_ARGS(new_segment.stop), new_segment.state);

	/* try coalescing the new segment with the most recent one */
	/* FIXME:  it might be more efficient to apply the attack and hold
	 * properties here, adjusting the segment list as we construct it,
	 * than scanning for intersections with a segment later.  would
	 * require smarter coalescing logic */
	if(element->control_segments->len) {
		struct control_segment *final_segment = &((struct control_segment *) element->control_segments->data)[element->control_segments->len - 1];
		/* if the most recent segment and the new segment have the
		 * same state and they touch, merge them */
		if(final_segment->state == new_segment.state && final_segment->stop >= new_segment.start) {
			g_assert_cmpuint(new_segment.stop, >=, final_segment->stop);
			final_segment->stop = new_segment.stop;
			return;
		}
		/* otherwise, if the most recent segment had 0 length,
		 * replace it entirely with the new one.  note that the
		 * state carried by a zero-length segment is meaningless,
		 * zero-length segments are merely interpreted as a
		 * heart-beat indicating how far the control stream has
		 * advanced */
		if(final_segment->stop == final_segment->start) {
			*final_segment = new_segment;
			return;
		}
	}
	/* otherwise append a new segment */
	g_array_append_val(element->control_segments, new_segment);
}


/*
 * return the most recently recorded control segment
 */


static struct control_segment control_get_final_segment(GSTLALGate *element)
{
	g_assert(element->control_segments->len > 0);	/* redundant? */
	return g_array_index(element->control_segments, struct control_segment, element->control_segments->len - 1);
}


/*
 * flush the control segments.  must be called with the control lock held
 */


static void control_flush(GSTLALGate *element)
{
	if(element->control_segments->len)
		g_array_remove_range(element->control_segments, 0, element->control_segments->len);
}


/*
 * wait for the control segments to span the requested interval.  must be
 * called with the control lock released
 */


static void control_get_interval(GSTLALGate *element, GstClockTime tmin, GstClockTime tmax)
{
	g_mutex_lock(element->control_lock);
	element->t_sink_head = tmax;
	g_cond_broadcast(element->control_queue_head_changed);
	while(1) {
		guint i;

		/*
		 * flush old segments.
		 */

		for(i = 0; i < element->control_segments->len && g_array_index(element->control_segments, struct control_segment, i).stop <= tmin; i++);
		if(i) {
			GST_DEBUG_OBJECT(element, "flushing %u obsolete control segments", i);
			g_array_remove_range(element->control_segments, 0, i);
		}

		/*
		 * has head advanced far enough, or are we at EOS?
		 */

		if(element->control_segments->len) {
			GST_DEBUG_OBJECT(element, "have %u control segments upto %" GST_TIME_SECONDS_FORMAT, element->control_segments->len, GST_TIME_SECONDS_ARGS(control_get_final_segment(element).stop));
			if(control_get_final_segment(element).stop >= tmax)
				break;
		} else
			GST_DEBUG_OBJECT(element, "have 0 control segments");
		if(element->control_eos)
			break;

		/*
		 * no, wait for buffer to arrive
		 */

		GST_DEBUG_OBJECT(element, "waiting for control to advance to %" GST_TIME_SECONDS_FORMAT, GST_TIME_SECONDS_ARGS(tmax));
		g_cond_wait(element->control_queue_head_changed, element->control_lock);
	}
	if(element->control_eos)
		GST_DEBUG_OBJECT(element, "control is at EOS");
	g_mutex_unlock(element->control_lock);
}


/*
 * return the state of the control input over a range of timestamps.  must
 * be called with the control lock held.
 *
 * the return value is FALSE if control data is available for (at least
 * part of) the times requested and is less than the threshold everywhere
 * therein, or TRUE if control data is available for (at least part of) the
 * times requested and is greather than or equal to the threshold for any
 * interval of time therein.  if no control data is available for the
 * requested interval the return value is the element's default state (TRUE
 * or FALSE).
 */


static gboolean control_get_state(GSTLALGate *element, GstClockTime tmin, GstClockTime tmax)
{
	gboolean state = element->default_state;
	guint i;

	g_assert_cmpuint(tmin, <=, tmax);

	/*
	 * handle 0-length scan intervals.  this only works because segment
	 * boundaries and tmin and tmax are all restricted to being
	 * integers
	 *
	 * -+--+--+--+--+--+--+--+-
	 *  [        ) A
	 *           [           ) B
	 *        ^  ^
	 *        1  2
	 *
	 * if tmin=tmax=1, then incrementing tmax by 1 results in the state
	 * from segment A being the result, which is correct.  if
	 * tmin=tmax=2, then incrementing tmax by 1 results in the state
	 * from segment B being the result, which is correct.  if a segment
	 * has 0 length it is ignored so incrementing tmax will not cause
	 * segments to be skipped that wouldn't have been anyway.  if
	 * tmax!=tmin then neither is adjusted.
	 */

	if(tmax == tmin)
		tmax++;

	/*
	 * loop assumes control segments are in order and do not overlap.
	 * won't crash if this isn't true, but the result is undefined.
	 */

	for(i = 0; i < element->control_segments->len; i++) {
		struct control_segment segment = g_array_index(element->control_segments, struct control_segment, i);
		if(segment.stop <= tmin)
			continue;
		if(tmax <= segment.start)
			break;
		/* zero-length segments are heart beats, they do not
		 * indicate true control state */
		if(segment.start == segment.stop)
			continue;
		/* if we get here, segment intersects the requested
		 * interval */
		if(segment.state)
			return TRUE;
		state = FALSE;
	}

	return state;
}


/*
 * ============================================================================
 *
 *                                  Signals
 *
 * ============================================================================
 */


enum gstlal_gate_signal {
	SIGNAL_RATE_CHANGED,
	SIGNAL_START,
	SIGNAL_STOP,
	NUM_SIGNALS
};


static guint signals[NUM_SIGNALS] = {0, };


static void rate_changed(GstElement *element, gint rate, void *data)
{
	/* FIXME:  do something? */
}


static void start(GstElement *element, guint64 timestamp, void *data)
{
	/* FIXME:  do something? */
}


static void stop(GstElement *element, guint64 timestamp, void *data)
{
	/* FIXME:  do something? */
}


/*
 * ============================================================================
 *
 *                                 Properties
 *
 * ============================================================================
 */


enum property {
	ARG_EMIT_SIGNALS = 1,
	ARG_DEFAULT_STATE,
	ARG_THRESHOLD,
	ARG_ATTACK_LENGTH,
	ARG_HOLD_LENGTH,
	ARG_LEAKY,
	ARG_INVERT
};


static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec)
{
	GSTLALGate *element = GSTLAL_GATE(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_EMIT_SIGNALS:
		element->emit_signals = g_value_get_boolean(value);
		break;

	case ARG_DEFAULT_STATE:
		element->default_state = g_value_get_boolean(value);
		break;

	case ARG_THRESHOLD:
		element->threshold = g_value_get_double(value);
		break;

	case ARG_ATTACK_LENGTH:
		element->attack_length = g_value_get_int64(value);
		break;

	case ARG_HOLD_LENGTH:
		element->hold_length = g_value_get_int64(value);
		break;

	case ARG_LEAKY:
		element->leaky = g_value_get_boolean(value);
		break;

	case ARG_INVERT:
		element->invert_control = g_value_get_boolean(value);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *pspec)
{
	GSTLALGate *element = GSTLAL_GATE(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_EMIT_SIGNALS:
		g_value_set_boolean(value, element->emit_signals);
		break;

	case ARG_DEFAULT_STATE:
		g_value_set_boolean(value, element->default_state);
		break;

	case ARG_THRESHOLD:
		g_value_set_double(value, element->threshold);
		break;

	case ARG_ATTACK_LENGTH:
		g_value_set_int64(value, element->attack_length);
		break;

	case ARG_HOLD_LENGTH:
		g_value_set_int64(value, element->hold_length);
		break;

	case ARG_LEAKY:
		g_value_set_boolean(value, element->leaky);
		break;
	
	case ARG_INVERT:
		g_value_set_boolean(value, element->invert_control);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * ============================================================================
 *
 *                                Control Pad
 *
 * ============================================================================
 */


/*
 * setcaps()
 */


static gboolean control_setcaps(GstPad *pad, GstCaps *caps)
{
	GSTLALGate *element = GSTLAL_GATE(gst_pad_get_parent(pad));
	GstStructure *structure;
	const gchar *media_type;
	gdouble (*control_sample_func)(const gpointer, guint64) = NULL;
	gint rate, width;
	gboolean success = TRUE;

	/*
	 * parse caps
	 */

	structure = gst_caps_get_structure(caps, 0);
	media_type = gst_structure_get_name(structure);
	if(!gst_structure_get_int(structure, "rate", &rate))
		success = FALSE;
	if(!gst_structure_get_int(structure, "width", &width))
		success = FALSE;
	if(!strcmp(media_type, "audio/x-raw-float")) {
		switch(width) {
		case 32:
			control_sample_func = control_sample_float32;
			break;
		case 64:
			control_sample_func = control_sample_float64;
			break;
		default:
			success = FALSE;
			break;
		}
	} else if(!strcmp(media_type, "audio/x-raw-complex")) {
		switch(width) {
		case 64:
			control_sample_func = control_sample_complex64;
			break;
		case 128:
			control_sample_func = control_sample_complex128;
			break;
		default:
			success = FALSE;
			break;
		}
	} else if(!strcmp(media_type, "audio/x-raw-int")) {
		gboolean is_signed;
		if(!gst_structure_get_boolean(structure, "signed", &is_signed))
			success = FALSE;
		switch(width) {
		case 8:
			control_sample_func = is_signed ? control_sample_int8 : control_sample_uint8;
			break;
		case 16:
			control_sample_func = is_signed ? control_sample_int16 : control_sample_uint16;
			break;
		case 32:
			control_sample_func = is_signed ? control_sample_int32 : control_sample_uint32;
			break;
		default:
			success = FALSE;
			break;
		}
	} else
		success = FALSE;

	/*
	 * update element
	 */

	if(success) {
		g_mutex_lock(element->control_lock);
		element->control_sample_func = control_sample_func;
		element->control_rate = rate;
		g_mutex_unlock(element->control_lock);
	}

	/*
	 * done.
	 */

	gst_object_unref(element);
	return success;
}


/*
 * chain()
 */


static GstFlowReturn control_chain(GstPad *pad, GstBuffer *sinkbuf)
{
	GSTLALGate *element = GSTLAL_GATE(gst_pad_get_parent(pad));
	GstFlowReturn result = GST_FLOW_OK;

	/*
	 * check validity of timestamp and offsets
	 */

	if(!(GST_BUFFER_TIMESTAMP_IS_VALID(sinkbuf) && GST_BUFFER_DURATION_IS_VALID(sinkbuf) && GST_BUFFER_OFFSET_IS_VALID(sinkbuf) && GST_BUFFER_OFFSET_END_IS_VALID(sinkbuf))) {
		GST_ELEMENT_ERROR(pad, STREAM, FAILED, ("invalid timestamp and/or offset"), ("%" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(sinkbuf)));
		gst_buffer_unref(sinkbuf);
		result = GST_FLOW_ERROR;
		goto done;
	}
	GST_DEBUG_OBJECT(pad, "have buffer %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(sinkbuf));

	/*
	 * wait until this buffer is needed
	 */

	g_mutex_lock(element->control_lock);
	while(!(element->sink_eos || (GST_CLOCK_TIME_IS_VALID(element->t_sink_head) && GST_BUFFER_TIMESTAMP(sinkbuf) < element->t_sink_head) || !element->control_segments->len)) {
		GST_DEBUG_OBJECT(pad, "waiting for space in queue: sink_eos = %d, t_sink_head is valid = %d, timestamp (%" GST_TIME_SECONDS_FORMAT ") >= t_sink_head (%" GST_TIME_SECONDS_FORMAT ") = %d", element->sink_eos, GST_CLOCK_TIME_IS_VALID(element->t_sink_head), GST_TIME_SECONDS_ARGS(GST_BUFFER_TIMESTAMP(sinkbuf)), GST_TIME_SECONDS_ARGS(element->t_sink_head), GST_BUFFER_TIMESTAMP(sinkbuf) >= element->t_sink_head);
		g_cond_wait(element->control_queue_head_changed, element->control_lock);
	}

	/*
	 * if we're at eos on sink pad, discard
	 */

	if(element->sink_eos) {
		GST_DEBUG_OBJECT(pad, "sink is at end-of-stream, discarding buffer");
		gst_buffer_unref(sinkbuf);
		g_mutex_unlock(element->control_lock);
		goto done;
	}

	/*
	 * digest buffer into segments of contiguous state:  TRUE = at or
	 * above threshold, FALSE = below threshold.
	 */

	if(GST_BUFFER_FLAG_IS_SET(sinkbuf, GST_BUFFER_FLAG_GAP) || !GST_BUFFER_DURATION(sinkbuf)) {
		control_add_segment(element, GST_BUFFER_TIMESTAMP(sinkbuf), GST_BUFFER_TIMESTAMP(sinkbuf) + GST_BUFFER_DURATION(sinkbuf), FALSE);
	} else {
		guint buffer_length = GST_BUFFER_OFFSET_END(sinkbuf) - GST_BUFFER_OFFSET(sinkbuf);
		guint segment_start;
		guint segment_length;

		for(segment_start = 0; segment_start < buffer_length; segment_start += segment_length) {
			/* state for this segment */
			gboolean state = element->control_sample_func(GST_BUFFER_DATA(sinkbuf), segment_start) >= element->threshold;
			for(segment_length = 1; segment_start + segment_length < buffer_length; segment_length++)
				if(state != (element->control_sample_func(GST_BUFFER_DATA(sinkbuf), segment_start + segment_length) >= element->threshold))
					/* state has changed */
					break;
			control_add_segment(element, GST_BUFFER_TIMESTAMP(sinkbuf) + gst_util_uint64_scale_int_round(GST_BUFFER_DURATION(sinkbuf), segment_start, buffer_length), GST_BUFFER_TIMESTAMP(sinkbuf) + gst_util_uint64_scale_int_round(GST_BUFFER_DURATION(sinkbuf), segment_start + segment_length, buffer_length), state);
		}
	}
	gst_buffer_unref(sinkbuf);
	GST_DEBUG_OBJECT(pad, "buffer %" GST_BUFFER_BOUNDARIES_FORMAT " digested", GST_BUFFER_BOUNDARIES_ARGS(sinkbuf));
	g_cond_broadcast(element->control_queue_head_changed);
	g_mutex_unlock(element->control_lock);

	/*
	 * done
	 */

done:
	gst_object_unref(element);
	return result;
}


/*
 * event()
 */


static gboolean control_event(GstPad *pad, GstEvent *event)
{
	GSTLALGate *element = GSTLAL_GATE(GST_PAD_PARENT(pad));

	switch(GST_EVENT_TYPE(event)) {
	case GST_EVENT_NEWSEGMENT:
		GST_DEBUG_OBJECT(pad, "new segment;  clearing end-of-stream flag and flushing control queue");
		g_mutex_lock(element->control_lock);
		element->control_eos = FALSE;
		control_flush(element);
		g_mutex_unlock(element->control_lock);
		break;

	case GST_EVENT_EOS:
		GST_DEBUG_OBJECT(pad, "end-of-stream;  setting end-of-stream flag");
		g_mutex_lock(element->control_lock);
		element->control_eos = TRUE;
		g_cond_broadcast(element->control_queue_head_changed);
		g_mutex_unlock(element->control_lock);
		break;

	default:
		break;
	}

	/*
	 * events on arriving on control pad are not forwarded
	 */

	gst_event_unref(event);
	return TRUE;
}


/*
 * ============================================================================
 *
 *                                  Sink Pad
 *
 * ============================================================================
 */


/*
 * getcaps()
 */


static GstCaps *getcaps(GstPad *pad)
{
	GSTLALGate *element = GSTLAL_GATE(gst_pad_get_parent(pad));
	GstPad *otherpad = pad == element->srcpad ? element->sinkpad : element->srcpad;
	GstCaps *peercaps, *caps;

	/*
	 * get our own allowed caps.  use the fixed caps function to avoid
	 * recursing back into this function.
	 */

	caps = gst_pad_get_fixed_caps_func(pad);

	/*
	 * get the allowed caps from the downstream peer if the peer has
	 * caps, intersect without our own.
	 */

	peercaps = gst_pad_peer_get_caps_reffed(otherpad);
	if(peercaps) {
		GstCaps *result = gst_caps_intersect(peercaps, caps);
		gst_caps_unref(peercaps);
		gst_caps_unref(caps);
		caps = result;
	}

	/*
	 * done
	 */

	gst_object_unref(element);
	return caps;
}


/*
 * acceptcaps()
 */


static gboolean acceptcaps(GstPad *pad, GstCaps *caps)
{
	GSTLALGate *element = GSTLAL_GATE(gst_pad_get_parent(pad));
	GstPad *otherpad = pad == element->srcpad ? element->sinkpad : element->srcpad;
	gboolean success;

	/*
	 * ask downstream peer
	 */

	success = gst_pad_peer_accept_caps(otherpad, caps);

	/*
	 * done
	 */

	gst_object_unref(element);
	return success;
}


/*
 * setcaps()
 */


static gboolean sink_setcaps(GstPad *pad, GstCaps *caps)
{
	GSTLALGate *element = GSTLAL_GATE(gst_pad_get_parent(pad));
	GstStructure *structure;
	gint rate, width, channels;
	gboolean success = TRUE;

	/*
	 * parse caps
	 */

	structure = gst_caps_get_structure(caps, 0);
	if(!gst_structure_get_int(structure, "rate", &rate))
		success = FALSE;
	if(!gst_structure_get_int(structure, "width", &width))
		success = FALSE;
	if(!gst_structure_get_int(structure, "channels", &channels))
		success = FALSE;

	/*
	 * try setting caps on downstream element
	 */

	if(success)
		success = gst_pad_set_caps(element->srcpad, caps);

	/*
	 * update the element metadata
	 */

	if(success) {
		gint old_rate = element->rate;
		element->rate = rate;
		element->unit_size = width / 8 * channels;
		if(element->rate != old_rate)
			g_signal_emit(G_OBJECT(element), signals[SIGNAL_RATE_CHANGED], 0, element->rate, NULL);
	}

	/*
	 * done
	 */

	gst_object_unref(element);
	return success;
}


/*
 * chain()
 */


static GstFlowReturn sink_chain(GstPad *pad, GstBuffer *sinkbuf)
{
	GSTLALGate *element = GSTLAL_GATE(gst_pad_get_parent(pad));
	guint64 sinkbuf_length;
	guint64 start, length;
	GstFlowReturn result = GST_FLOW_OK;

	/*
	 * check validity of timestamp and offsets
	 */

	if(!(GST_BUFFER_TIMESTAMP_IS_VALID(sinkbuf) && GST_BUFFER_DURATION_IS_VALID(sinkbuf) && GST_BUFFER_OFFSET_IS_VALID(sinkbuf) && GST_BUFFER_OFFSET_END_IS_VALID(sinkbuf))) {
		GST_ELEMENT_ERROR(pad, STREAM, FAILED, ("invalid timestamp and/or offset"), ("%" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(sinkbuf)));
		result = GST_FLOW_ERROR;
		goto done;
	}
	if(element->attack_length + element->hold_length < 0) {
		GST_ERROR_OBJECT(element, "attack-length + hold-length < 0");
		result = GST_FLOW_ERROR;
		goto done;
	}

	sinkbuf_length = GST_BUFFER_OFFSET_END(sinkbuf) - GST_BUFFER_OFFSET(sinkbuf);

	if(GST_BUFFER_IS_DISCONT(sinkbuf))
		element->need_discont = TRUE;

	/*
	 * wait for control queue to span the necessary interval
	 */

	GST_DEBUG_OBJECT(element->sinkpad, "got buffer %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(sinkbuf));
	control_get_interval(element, timestamp_add_offset(GST_BUFFER_TIMESTAMP(sinkbuf), -element->hold_length, element->rate), timestamp_add_offset(GST_BUFFER_TIMESTAMP(sinkbuf), (gint64) sinkbuf_length + element->attack_length, element->rate));

	/*
	 * is input zero size or already a gap?  then push it as is
	 */

	if(!GST_BUFFER_SIZE(sinkbuf)) {
		/*
		 * is a discontinuity pending?
		 */

		if(element->need_discont && !GST_BUFFER_IS_DISCONT(sinkbuf)) {
			sinkbuf = gst_buffer_make_metadata_writable(sinkbuf);
			GST_BUFFER_FLAG_SET(sinkbuf, GST_BUFFER_FLAG_DISCONT);
		}
		element->need_discont = FALSE;

		/*
		 * push buffer
		 */

		GST_DEBUG_OBJECT(element->srcpad, "pushing reused zero-length buffer %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(sinkbuf));
		result = gst_pad_push(element->srcpad, sinkbuf);
		sinkbuf = NULL;
		goto done;
	} else if(GST_BUFFER_FLAG_IS_SET(sinkbuf, GST_BUFFER_FLAG_GAP)) {
		/*
		 * tell the world about state changes
		 */

		if(element->emit_signals && FALSE != element->last_state) {
			g_signal_emit(G_OBJECT(element), signals[SIGNAL_STOP], 0, GST_BUFFER_TIMESTAMP(sinkbuf), NULL);
			element->last_state = FALSE;
		}

		if(element->leaky) {
			/*
			 * discard buffer
			 */

			if(sinkbuf_length)
				/*
				 * skipping an interval of non-zero length,
				 * next buffer must be a discont
				 */

				element->need_discont = TRUE;
			GST_DEBUG_OBJECT(element->srcpad, "discarding gap buffer %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(sinkbuf));
			goto done;
		}

		/*
		 * is a discontinuity pending?
		 */

		if(element->need_discont && !GST_BUFFER_IS_DISCONT(sinkbuf)) {
			sinkbuf = gst_buffer_make_metadata_writable(sinkbuf);
			GST_BUFFER_FLAG_SET(sinkbuf, GST_BUFFER_FLAG_DISCONT);
		}
		element->need_discont = FALSE;

		/*
		 * push buffer
		 */

		GST_DEBUG_OBJECT(element->srcpad, "pushing reused gap buffer %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(sinkbuf));
		result = gst_pad_push(element->srcpad, sinkbuf);
		sinkbuf = NULL;
		goto done;
	}

	/*
	 * loop over the contents of the input buffer.
	 */

	for(start = 0; start < sinkbuf_length; start += length) {
		GstBuffer *srcbuf;
		GstClockTime timestamp;
		gboolean state;

		/*
		 * find the next interval of continuous control state
		 */

		g_mutex_lock(element->control_lock);
		state = control_get_state(element, timestamp_add_offset(GST_BUFFER_TIMESTAMP(sinkbuf), (gint64) start - element->hold_length, element->rate), timestamp_add_offset(GST_BUFFER_TIMESTAMP(sinkbuf), (gint64) start + element->attack_length, element->rate));
		for(length = 1; start + length < sinkbuf_length; length++) {
			if(state != control_get_state(element, timestamp_add_offset(GST_BUFFER_TIMESTAMP(sinkbuf), (gint64) (start + length) - element->hold_length, element->rate), timestamp_add_offset(GST_BUFFER_TIMESTAMP(sinkbuf), (gint64) (start + length) + element->attack_length, element->rate)))
				break;
		}
		g_mutex_unlock(element->control_lock);

		/*
		 * if the output state has changed, tell the world about it
		 */

		timestamp = GST_BUFFER_TIMESTAMP(sinkbuf) + gst_util_uint64_scale_int_round(GST_BUFFER_DURATION(sinkbuf), start, sinkbuf_length);
		if(element->emit_signals && state != element->last_state) {
			g_signal_emit(G_OBJECT(element), signals[state ? SIGNAL_START : SIGNAL_STOP], 0, timestamp, NULL);
			element->last_state = state;
		}

		/*
		 * if the output is a gap and we're in leaky mode, discard
		 * it.  next buffer must be a discont because we know the
		 * gap has non-zero length
		 */

		if(!state && element->leaky) {
			element->need_discont = TRUE;
			continue;
		}

		/*
		 * output is a non-gap buffer of non-zero length.  if it's
		 * the entire input buffer then re-use it otherwise create
		 * a subbuffer from it
		 */

		if(length == sinkbuf_length) {
			srcbuf = sinkbuf;
			sinkbuf = NULL;
		} else {
			srcbuf = gst_buffer_create_sub(sinkbuf, start * element->unit_size, length * element->unit_size);
			if(!srcbuf) {
				GST_ERROR_OBJECT(element, "failure creating sub-buffer");
				result = GST_FLOW_ERROR;
				goto done;
			}

			/*
			 * set flags, caps, offset, and timestamps.
			 */

			gst_buffer_copy_metadata(srcbuf, sinkbuf, GST_BUFFER_COPY_FLAGS | GST_BUFFER_COPY_CAPS);
			GST_BUFFER_OFFSET(srcbuf) = GST_BUFFER_OFFSET(sinkbuf) + start;
			GST_BUFFER_OFFSET_END(srcbuf) = GST_BUFFER_OFFSET(srcbuf) + length;
			GST_BUFFER_TIMESTAMP(srcbuf) = timestamp;
			GST_BUFFER_DURATION(srcbuf) = GST_BUFFER_TIMESTAMP(sinkbuf) + gst_util_uint64_scale_int_round(GST_BUFFER_DURATION(sinkbuf), start + length, sinkbuf_length) - GST_BUFFER_TIMESTAMP(srcbuf);
		}

		/*
		 * is a discontinuity pending?
		 */

		if(!!element->need_discont != !!GST_BUFFER_IS_DISCONT(srcbuf)) {
			srcbuf = gst_buffer_make_metadata_writable(srcbuf);
			if(element->need_discont)
				GST_BUFFER_FLAG_SET(srcbuf, GST_BUFFER_FLAG_DISCONT);
			else
				GST_BUFFER_FLAG_UNSET(srcbuf, GST_BUFFER_FLAG_DISCONT);
		}
		element->need_discont = FALSE;

		/*
		 * if control input was below threshold then flag buffer as
		 * silence.
		 */

		if(!state) {
			srcbuf = gst_buffer_make_metadata_writable(srcbuf);
			GST_BUFFER_FLAG_SET(srcbuf, GST_BUFFER_FLAG_GAP);
		}

		/*
		 * push buffer down stream
		 */

		GST_DEBUG_OBJECT(element->srcpad, "pushing buffer %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(srcbuf));
		result = gst_pad_push(element->srcpad, srcbuf);
		if(G_UNLIKELY(result != GST_FLOW_OK)) {
			GST_WARNING_OBJECT(element->srcpad, "gst_pad_push() failed (%s)", gst_flow_get_name(result));
			goto done;
		}
	}

	/*
	 * done
	 */

done:
	/* need to check that we haven't discarded it or re-used sinkbuf as
	 * srcbuf */
	if(sinkbuf)
		gst_buffer_unref(sinkbuf);
	gst_object_unref(element);
	return result;
}


/*
 * event()
 */


static gboolean sink_event(GstPad *pad, GstEvent *event)
{
	GSTLALGate *element = GSTLAL_GATE(GST_PAD_PARENT(pad));

	switch(GST_EVENT_TYPE(event)) {
	case GST_EVENT_NEWSEGMENT:
		GST_DEBUG_OBJECT(pad, "new segment;  clearing end-of-stream flag");
		g_mutex_lock(element->control_lock);
		element->t_sink_head = GST_CLOCK_TIME_NONE;
		element->sink_eos = FALSE;
		element->last_state = -1;	/* force signal on initial state */
		element->need_discont = TRUE;
		g_mutex_unlock(element->control_lock);
		break;

	case GST_EVENT_EOS:
		GST_DEBUG_OBJECT(pad, "end-of-stream;  setting end-of-stream flag and flushing control queue");
		g_mutex_lock(element->control_lock);
		element->sink_eos = TRUE;
		control_flush(element);
		g_cond_broadcast(element->control_queue_head_changed);
		g_mutex_unlock(element->control_lock);
		break;

	default:
		break;
	}

	/*
	 * sink events are forwarded to src pad
	 */

	return gst_pad_push_event(element->srcpad, event);
}


/*
 * ============================================================================
 *
 *                                 Source Pad
 *
 * ============================================================================
 */


/*
 * event()
 *
 * push event on control and sink pads.  the default event handler just
 * picks one of the two at random, but we should send it to both.
 */


static gboolean src_event(GstPad *pad, GstEvent *event)
{
	GSTLALGate *element = GSTLAL_GATE(GST_PAD_PARENT(pad));
	gboolean success = TRUE;

	gst_event_ref(event);
	success &= gst_pad_push_event(element->sinkpad, event);
	success &= gst_pad_push_event(element->controlpad, event);

	return success;
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
	GSTLALGate *element = GSTLAL_GATE(object);

	gst_object_unref(element->controlpad);
	element->controlpad = NULL;
	gst_object_unref(element->sinkpad);
	element->sinkpad = NULL;
	gst_object_unref(element->srcpad);
	element->srcpad = NULL;
	g_mutex_free(element->control_lock);
	element->control_lock = NULL;
	g_array_unref(element->control_segments);
	element->control_segments = NULL;
	g_cond_free(element->control_queue_head_changed);
	element->control_queue_head_changed = NULL;

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


/*
 * Base init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GBaseInitFunc
 */


#define CAPS \
	"audio/x-raw-int, " \
	"rate = (int) [1, MAX], " \
	"channels = (int) [1, MAX], " \
	"endianness = (int) BYTE_ORDER, " \
	"width = (int) {8, 16, 32, 64}, " \
	"signed = (boolean) {true, false}; " \
	"audio/x-raw-float, " \
	"rate = (int) [1, MAX], " \
	"channels = (int) [1, MAX], " \
	"endianness = (int) BYTE_ORDER, " \
	"width = (int) {32, 64}; " \
	"audio/x-raw-complex, " \
	"rate = (int) [1, MAX], " \
	"channels = (int) [1, MAX], " \
	"endianness = (int) BYTE_ORDER, " \
	"width = (int) {64, 128}"


static void base_init(gpointer class)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(class);

	gst_element_class_set_details_simple(
		element_class,
		"Gate",
		"Filter",
		"Flag buffers as gaps based on the value of a control input",
		"Kipp Cannon <kipp.cannon@ligo.org>, Chad Hanna <channa@ligo.caltech.edu>"
	);

	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"control",
			GST_PAD_SINK,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
				"audio/x-raw-int, " \
				"rate = (int) [1, MAX], " \
				"channels = (int) 1, " \
				"endianness = (int) BYTE_ORDER, " \
				"width = (int) {8, 16, 32, 64}, " \
				"signed = (boolean) {true, false} ; " \
				"audio/x-raw-float, " \
				"rate = (int) [1, MAX], " \
				"channels = (int) 1, " \
				"endianness = (int) BYTE_ORDER, " \
				"width = (int) {32, 64};" \
				"audio/x-raw-complex, " \
				"rate = (int) [1, MAX], " \
				"channels = (int) 1, " \
				"endianness = (int) BYTE_ORDER, " \
				"width = (int) {64, 128}"
			)
		)
	);
	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"sink",
			GST_PAD_SINK,
			GST_PAD_ALWAYS,
			gst_caps_from_string(CAPS)
		)
	);
	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"src",
			GST_PAD_SRC,
			GST_PAD_ALWAYS,
			gst_caps_from_string(CAPS)
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
	GSTLALGateClass *gstlal_gate_class = GSTLAL_GATE_CLASS(klass);

	parent_class = g_type_class_ref(GST_TYPE_ELEMENT);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	gstlal_gate_class->rate_changed = GST_DEBUG_FUNCPTR(rate_changed);
	gstlal_gate_class->start = GST_DEBUG_FUNCPTR(start);
	gstlal_gate_class->stop = GST_DEBUG_FUNCPTR(stop);

	g_object_class_install_property(
		gobject_class,
		ARG_EMIT_SIGNALS,
		g_param_spec_boolean(
			"emit-signals",
			"Emit signals",
			"Emit start and stop signals (rate-changed is always emited).  The start and stop signals are emited on gap-to-non-gap and non-gap-to-gap transitions in the output stream respectively.",
			DEFAULT_EMIT_SIGNALS,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_DEFAULT_STATE,
		g_param_spec_boolean(
			"default-state",
			"Default State",
			"Control state to assume when control input is not available",
			DEFAULT_DEFAULT_STATE,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_THRESHOLD,
		g_param_spec_double(
			"threshold",
			"Threshold",
			"Output will be flagged as non-gap when magnitude of control input is >= this value.  See also invert-control.",
			0, G_MAXDOUBLE, DEFAULT_THRESHOLD,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_ATTACK_LENGTH,
		g_param_spec_int64(
			"attack-length",
			"Attack",
			"Number of samples of the input stream ahead of negative-to-positive threshold crossing to include in non-gap output.",
			G_MININT64, G_MAXINT64, DEFAULT_ATTACK_LENGTH,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_HOLD_LENGTH,
		g_param_spec_int64(
			"hold-length",
			"Hold",
			"Number of samples of the input stream following positive-to-negative threshold crossing to include in non-gap output.",
			G_MININT64, G_MAXINT64, DEFAULT_HOLD_LENGTH,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_LEAKY,
		g_param_spec_boolean(
			"leaky",
			"Leaky",
			"Drop buffers instead of forwarding gaps.",
			DEFAULT_LEAKY,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_INVERT,
		g_param_spec_boolean(
			"invert-control",
			"Invert",
			"Logically invert the control input.  If false (default) then the output is a gap if and only if the control is < threshold;  if true then the output is a gap if and only if the control is >= threshold.",
			DEFAULT_INVERT,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);

	signals[SIGNAL_RATE_CHANGED] = g_signal_new(
		"rate-changed",
		G_TYPE_FROM_CLASS(klass),
		G_SIGNAL_RUN_FIRST,
		G_STRUCT_OFFSET(
			GSTLALGateClass,
			rate_changed
		),
		NULL,
		NULL,
		g_cclosure_marshal_VOID__INT,
		G_TYPE_NONE,
		1,
		G_TYPE_INT
	);
	signals[SIGNAL_START] = g_signal_new(
		"start",
		G_TYPE_FROM_CLASS(klass),
		G_SIGNAL_RUN_FIRST,
		G_STRUCT_OFFSET(
			GSTLALGateClass,
			start
		),
		NULL,
		NULL,
		gst_marshal_VOID__INT64,
		G_TYPE_NONE,
		1,
		G_TYPE_UINT64
	);
	signals[SIGNAL_STOP] = g_signal_new(
		"stop",
		G_TYPE_FROM_CLASS(klass),
		G_SIGNAL_RUN_FIRST,
		G_STRUCT_OFFSET(
			GSTLALGateClass,
			stop
		),
		NULL,
		NULL,
		gst_marshal_VOID__INT64,
		G_TYPE_NONE,
		1,
		G_TYPE_UINT64
	);
}


/*
 * Instance init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GInstanceInitFunc
 */


static void instance_init(GTypeInstance *object, gpointer klass)
{
	GSTLALGate *element = GSTLAL_GATE(object);
	GstPad *pad;

	gst_element_create_all_pads(GST_ELEMENT(element));

	/* configure (and ref) control pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "control");
	gst_pad_set_setcaps_function(pad, GST_DEBUG_FUNCPTR(control_setcaps));
	gst_pad_set_chain_function(pad, GST_DEBUG_FUNCPTR(control_chain));
	gst_pad_set_event_function(pad, GST_DEBUG_FUNCPTR(control_event));
	element->controlpad = pad;

	/* configure (and ref) sink pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "sink");
	gst_pad_set_getcaps_function(pad, GST_DEBUG_FUNCPTR(getcaps));
	gst_pad_set_acceptcaps_function(pad, GST_DEBUG_FUNCPTR(acceptcaps));
	gst_pad_set_setcaps_function(pad, GST_DEBUG_FUNCPTR(sink_setcaps));
	gst_pad_set_chain_function(pad, GST_DEBUG_FUNCPTR(sink_chain));
	gst_pad_set_event_function(pad, GST_DEBUG_FUNCPTR(sink_event));
	element->sinkpad = pad;

	/* retrieve (and ref) src pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "src");
	gst_pad_set_getcaps_function(pad, GST_DEBUG_FUNCPTR(getcaps));
	gst_pad_set_acceptcaps_function(pad, GST_DEBUG_FUNCPTR(acceptcaps));
	gst_pad_set_event_function(pad, GST_DEBUG_FUNCPTR(src_event));
	element->srcpad = pad;

	/* internal data */
	element->control_lock = g_mutex_new();
	element->control_eos = FALSE;
	element->sink_eos = FALSE;
	element->t_sink_head = GST_CLOCK_TIME_NONE;
	element->control_segments = g_array_new(FALSE, FALSE, sizeof(struct control_segment));
	element->control_queue_head_changed = g_cond_new();
	element->control_sample_func = NULL;
	element->last_state = -1;	/* force signal on initial state */
	element->rate = 0;
	element->unit_size = 0;
	element->control_rate = 0;
	element->need_discont = FALSE;
}


/*
 * gstlal_gate_get_type().
 */


GType gstlal_gate_get_type(void)
{
	static GType type = 0;

	if(!type) {
		static const GTypeInfo info = {
			.class_size = sizeof(GSTLALGateClass),
			.class_init = class_init,
			.base_init = base_init,
			.instance_size = sizeof(GSTLALGate),
			.instance_init = instance_init,
		};
		type = g_type_register_static(GST_TYPE_ELEMENT, "lal_gate", &info, 0);
	}

	return type;
}

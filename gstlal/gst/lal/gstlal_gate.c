/*
 * An element to flag buffers in a stream as silence or not based on the
 * value of a control input.
 *
 * Copyright (C) 2008-2013  Kipp Cannon, Chad Hanna
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


/**
 * SECTION:gstlal_gate
 * @short_description:  Flag buffers as gaps based on the value of a control input.
 *
 * Reviewed:  8466e17ed01185bd3182603207d2ac322f502967 2014-08-14 K.
 * Cannon, J.  Creighton, B. Sathyaprakash.
 *
 * Completed Actions:
 * - removed 64-bit support for control stream:  not possibleto specify
 * threshold to that precision
 * - Why not signal control_queue_head_changed on receipt of NEW_SEGMENT? not needed.
 *
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
#include <gst/audio/audio.h>
#include <gst/audio/audio-format.h>


/*
 * our own stuff
 */


#include <gstlal/gstlal_audio_info.h>
#include <gstlal/gstlal_debug.h>
#include <gstlal_gate.h>


/*
 * ============================================================================
 *
 *                           GStreamer Boilerplate
 *
 * ============================================================================
 */


#define GST_CAT_DEFAULT gstlal_gate_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


G_DEFINE_TYPE_WITH_CODE(
	GSTLALGate,
	gstlal_gate,
	GST_TYPE_ELEMENT,
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "lal_gate", 0, "lal_gate element")
);


/*
 * ============================================================================
 *
 *                                 Parameters
 *
 * ============================================================================
 */


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
 * return the stop time of the i-th segment in the control queue
 */


static GstClockTime control_get_tstop(GSTLALGate *element, guint i)
{
	return g_array_index(element->control_segments, struct control_segment, i).stop;
}


/*
 * flush the control segments.  must be called with the control lock held
 */


static void control_flush(GSTLALGate *element)
{
	if(element->control_segments->len)
		g_array_remove_range(element->control_segments, 0, element->control_segments->len);
}


static void control_flush_upto(GSTLALGate *element, GstClockTime t)
{
	guint i;

	for(i = 0; i < element->control_segments->len && control_get_tstop(element, i) <= t; i++);
	if(i) {
		GST_DEBUG_OBJECT(element, "flushing %u obsolete control segments", i);
		g_array_remove_range(element->control_segments, 0, i);
	}
}


/*
 * wait for the control segments to span the interval needed to decide the
 * state of [timestamp,timestamp+duration).  must be called with the
 * control lock released
 */


static void control_get_interval(GSTLALGate *element, GstClockTime timestamp, GstClockTime duration)
{
	GstClockTime tmin, tmax;

	/*
	 * compute the interval the control data must span
	 */

	tmin = timestamp_add_offset(timestamp, MIN(-element->hold_length, +element->attack_length), element->rate);
	tmax = timestamp_add_offset(timestamp + duration, MAX(-element->hold_length, +element->attack_length), element->rate);
	if(tmin > tmax) {
		GstClockTime t = tmin;
		tmin = tmax;
		tmax = t;
	}

	/*
	 * wait loop
	 */

	g_mutex_lock(&element->control_lock);
	element->t_sink_head = tmax;
	g_cond_broadcast(&element->control_queue_head_changed);
	while(1) {
		/*
		 * flush old segments.  do this in the loop so that we can
		 * clear out newly received yet useless buffers as they
		 * arrive
		 */

		control_flush_upto(element, tmin);

		/*
		 * has head advanced far enough, or are we at EOS?
		 */

		if(element->control_segments->len) {
			GstClockTime control_tmax = control_get_tstop(element, element->control_segments->len - 1);
			GST_DEBUG_OBJECT(element, "have %u control segments upto %" GST_TIME_SECONDS_FORMAT, element->control_segments->len, GST_TIME_SECONDS_ARGS(control_tmax));
			if(control_tmax >= tmax)
				break;
		} else
			GST_DEBUG_OBJECT(element, "have 0 control segments");
		if(element->control_eos) {
			GST_DEBUG_OBJECT(element, "control is at EOS");
			break;
		}

		/*
		 * no, wait for buffer to arrive
		 */

		GST_DEBUG_OBJECT(element, "waiting for control to advance to %" GST_TIME_SECONDS_FORMAT, GST_TIME_SECONDS_ARGS(tmax));
		g_cond_wait(&element->control_queue_head_changed, &element->control_lock);
	}
	g_mutex_unlock(&element->control_lock);
}


/*
 * return the state of the control input over a range of timestamps.  must
 * be called with the control lock held.
 *
 * if no control data is available for the requested interval the return
 * value is the element's default state (TRUE or FALSE).  otherwise,
 *
 * when tmin <= tmax:
 *
 * the return value is TRUE if control data is available for (at least part
 * of) the times requested and is above threshold for any interval of time
 * therein, and FALSE otherwise
 *
 * when tmax < tmin:
 *
 * the return value is FALSE if control data is available for (at least
 * part of) the times requested and is below threshold for any interval of
 * time therein, and TRUE otherwise
 *
 * the case of tmax and tmin being out of order occurs when at least one of
 * attack and hold is negative and the other is not sufficiently positive
 * for their sum to be non-negative.  you'd have to draw a bunch of
 * pictures to see why that condition should be handled the way it is here
 */


static gboolean control_get_state(GSTLALGate *element, GstClockTime tmin, GstClockTime tmax)
{
	gboolean state = element->default_state;
	gboolean inverted_interval = tmax < tmin;
	guint i;

	/*
	 * get tmin and tmax the right way around
	 */

	if(inverted_interval) {
		GstClockTime t = tmin;
		tmin = tmax;
		tmax = t;
	}

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
		if(inverted_interval) {
			if(!segment.state)
				return FALSE;
			state = TRUE;
		} else {
			if(segment.state)
				return TRUE;
			state = FALSE;
		}
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


static void rate_changed(GSTLALGate *element, gint rate, void *data)
{
	/* FIXME:  do something? */
}


static void start(GSTLALGate *element, guint64 timestamp, void *data)
{
	/* FIXME:  do something? */
}


static void stop(GSTLALGate *element, guint64 timestamp, void *data)
{
	/* FIXME:  do something? */
}


/*
 * ============================================================================
 *
 *                                Control Pad
 *
 * ============================================================================
 */


/*
 * control_setcaps()
 */


static gboolean control_setcaps(GSTLALGate *gate, GstPad *pad, GstCaps *caps)
{
	gdouble (*control_sample_func)(const gpointer, guint64) = NULL;
	GstAudioInfo info;
	gboolean success = gstlal_audio_info_from_caps(&info, caps);

	/*
	 * parse the format
	 */

	if(success) {
		switch(GST_AUDIO_INFO_FORMAT(&info)) {
		case GST_AUDIO_FORMAT_U8:
			control_sample_func = control_sample_uint8;
			break;
		case GST_AUDIO_FORMAT_U16:
			control_sample_func = control_sample_uint16;
			break;
		case GST_AUDIO_FORMAT_U32:
			control_sample_func = control_sample_uint32;
			break;
		case GST_AUDIO_FORMAT_S8:
			control_sample_func = control_sample_int8;
			break;
		case GST_AUDIO_FORMAT_S16:
			control_sample_func = control_sample_int16;
			break;
		case GST_AUDIO_FORMAT_S32:
			control_sample_func = control_sample_int32;
			break;
		case GST_AUDIO_FORMAT_F32:
			control_sample_func = control_sample_float32;
			break;
		case GST_AUDIO_FORMAT_F64:
			control_sample_func = control_sample_float64;
			break;
		case GST_AUDIO_FORMAT_Z64:
			control_sample_func = control_sample_complex64;
			break;
		case GST_AUDIO_FORMAT_Z128:
			control_sample_func = control_sample_complex128;
			break;
		default:
			success = FALSE;
			break;
		}
	}

	/*
	 * update element
	 */

	if(success) {
		g_mutex_lock(&gate->control_lock);
		gate->control_sample_func = control_sample_func;
		gate->control_rate = GST_AUDIO_INFO_RATE(&info);
		g_mutex_unlock(&gate->control_lock);
	} else
		GST_ERROR_OBJECT(gate, "unable to parse and/or accept caps %" GST_PTR_FORMAT, caps);

	/*
	 * done.
	 */

	return success;
}


/*
 * chain()
 */


static GstFlowReturn control_chain(GstPad *pad, GstObject *parent, GstBuffer *controlbuf)
{
	GSTLALGate *element = GSTLAL_GATE(parent);
	GstFlowReturn result = GST_FLOW_OK;

	/*
	 * check validity of timestamp and offsets
	 */

	if(!(GST_BUFFER_TIMESTAMP_IS_VALID(controlbuf) && GST_BUFFER_DURATION_IS_VALID(controlbuf) && GST_BUFFER_OFFSET_IS_VALID(controlbuf) && GST_BUFFER_OFFSET_END_IS_VALID(controlbuf))) {
		GST_ELEMENT_ERROR(pad, STREAM, FAILED, ("invalid timestamp and/or offset"), ("%" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(controlbuf)));
		result = GST_FLOW_ERROR;
		goto done;
	}
	GST_DEBUG_OBJECT(pad, "have buffer %p %" GST_BUFFER_BOUNDARIES_FORMAT, controlbuf, GST_BUFFER_BOUNDARIES_ARGS(controlbuf));

	/*
	 * wait until this buffer is needed
	 */

	g_mutex_lock(&element->control_lock);
	while(!(element->sink_eos || (GST_CLOCK_TIME_IS_VALID(element->t_sink_head) && GST_BUFFER_TIMESTAMP(controlbuf) < element->t_sink_head) || !element->control_segments->len)) {
		GST_DEBUG_OBJECT(pad, "waiting for space in queue: sink_eos = %d, t_sink_head is valid = %d, timestamp (%" GST_TIME_SECONDS_FORMAT ") >= t_sink_head (%" GST_TIME_SECONDS_FORMAT ") = %d", element->sink_eos, GST_CLOCK_TIME_IS_VALID(element->t_sink_head), GST_TIME_SECONDS_ARGS(GST_BUFFER_TIMESTAMP(controlbuf)), GST_TIME_SECONDS_ARGS(element->t_sink_head), GST_BUFFER_TIMESTAMP(controlbuf) >= element->t_sink_head);
		g_cond_wait(&element->control_queue_head_changed, &element->control_lock);
	}

	/*
	 * if we're at eos on sink pad, discard
	 */

	if(element->sink_eos) {
		GST_DEBUG_OBJECT(pad, "sink is at end-of-stream, discarding buffer");
		g_mutex_unlock(&element->control_lock);
		goto done;
	}

	/*
	 * digest buffer into segments of contiguous state:  TRUE = at or
	 * above threshold, FALSE = below threshold.
	 */


	if(GST_BUFFER_FLAG_IS_SET(controlbuf, GST_BUFFER_FLAG_GAP) || !GST_BUFFER_DURATION(controlbuf)) {
		control_add_segment(element, GST_BUFFER_TIMESTAMP(controlbuf), GST_BUFFER_TIMESTAMP(controlbuf) + GST_BUFFER_DURATION(controlbuf), FALSE);
	} else {
		GstMapInfo info;
		guint buffer_length = GST_BUFFER_OFFSET_END(controlbuf) - GST_BUFFER_OFFSET(controlbuf);
		guint segment_start;
		guint segment_length;
		g_assert_cmpuint(GST_BUFFER_OFFSET_END(controlbuf), >, GST_BUFFER_OFFSET(controlbuf));

		gst_buffer_map(controlbuf, &info, GST_MAP_READ);
		for(segment_start = 0; segment_start < buffer_length; segment_start += segment_length) {
			/* state for this segment */
			gboolean state = element->control_sample_func(info.data, segment_start) >= element->threshold;
			for(segment_length = 1; segment_start + segment_length < buffer_length; segment_length++)
				if(state != (element->control_sample_func(info.data, segment_start + segment_length) >= element->threshold))
					/* state has changed */
					break;
			control_add_segment(element, GST_BUFFER_TIMESTAMP(controlbuf) + gst_util_uint64_scale_int_round(GST_BUFFER_DURATION(controlbuf), segment_start, buffer_length), GST_BUFFER_TIMESTAMP(controlbuf) + gst_util_uint64_scale_int_round(GST_BUFFER_DURATION(controlbuf), segment_start + segment_length, buffer_length), state);
		}
		gst_buffer_unmap(controlbuf, &info);
	}
	GST_DEBUG_OBJECT(pad, "buffer %" GST_BUFFER_BOUNDARIES_FORMAT " digested", GST_BUFFER_BOUNDARIES_ARGS(controlbuf));
	g_cond_broadcast(&element->control_queue_head_changed);
	g_mutex_unlock(&element->control_lock);

	/*
	 * done
	 */

done:
	gst_buffer_unref(controlbuf);
	return result;
}


/*
 * event()
 */


static gboolean control_event(GstPad *pad, GstObject *parent, GstEvent *event)
{
	GSTLALGate *element = GSTLAL_GATE(parent);
	gboolean res = TRUE;

	switch(GST_EVENT_TYPE(event)) {
	case GST_EVENT_STREAM_START:
		GST_DEBUG_OBJECT(pad, "new segment;  clearing end-of-stream flag and flushing control queue");
		g_mutex_lock(&element->control_lock);
		element->control_eos = FALSE;
		control_flush(element);
		g_mutex_unlock(&element->control_lock);
		break;

	case GST_EVENT_EOS:
		GST_DEBUG_OBJECT(pad, "end-of-stream;  setting end-of-stream flag");
		g_mutex_lock(&element->control_lock);
		element->control_eos = TRUE;
		g_cond_broadcast(&element->control_queue_head_changed);
		g_mutex_unlock(&element->control_lock);
		break;

	case GST_EVENT_CAPS: {
		GstCaps *caps;
		gst_event_parse_caps(event, &caps);
		res = control_setcaps(element, pad, caps);
		break;
	}

	default:
		break;
	}

	/*
	 * events on arriving on control pad are not forwarded
	 */

	gst_event_unref(event);
	return res;
}


/*
 * ============================================================================
 *
 *                                  Sink Pad
 *
 * ============================================================================
 */


/*
 * chain()
 */


static GstFlowReturn sink_chain(GstPad *pad, GstObject *parent, GstBuffer *sinkbuf)
{
	GSTLALGate *element = GSTLAL_GATE(parent);
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

	sinkbuf_length = GST_BUFFER_OFFSET_END(sinkbuf) - GST_BUFFER_OFFSET(sinkbuf);

	if(GST_BUFFER_IS_DISCONT(sinkbuf))
		element->need_discont = TRUE;

	/*
	 * wait for control queue to span the necessary interval
	 */

	GST_DEBUG_OBJECT(element->sinkpad, "got buffer %p %" GST_BUFFER_BOUNDARIES_FORMAT, sinkbuf, GST_BUFFER_BOUNDARIES_ARGS(sinkbuf));
	g_assert_cmpuint(sinkbuf_length, ==, gst_util_uint64_scale_int_round(GST_BUFFER_DURATION(sinkbuf), element->rate, GST_SECOND));
	control_get_interval(element, GST_BUFFER_TIMESTAMP(sinkbuf), GST_BUFFER_DURATION(sinkbuf));

	/*
	 * is input zero size or already a gap?  then push it as is
	 */

	if(G_UNLIKELY(!sinkbuf_length)) {
		/*
		 * is a discontinuity pending?
		 */

		if(element->need_discont && !GST_BUFFER_IS_DISCONT(sinkbuf)) {
			sinkbuf = gst_buffer_make_writable(sinkbuf);
			GST_BUFFER_FLAG_SET(sinkbuf, GST_BUFFER_FLAG_DISCONT);
		}
		element->need_discont = FALSE;

		/*
		 * push buffer
		 */

		GST_DEBUG_OBJECT(element->srcpad, "pushing reused zero-length buffer %p %" GST_BUFFER_BOUNDARIES_FORMAT, sinkbuf, GST_BUFFER_BOUNDARIES_ARGS(sinkbuf));
		result = gst_pad_push(element->srcpad, sinkbuf);
		if(G_UNLIKELY(result != GST_FLOW_OK))
			GST_WARNING_OBJECT(element->srcpad, "gst_pad_push() failed (%s)", gst_flow_get_name(result));
		sinkbuf = NULL;
		goto done;
	} else if(GST_BUFFER_FLAG_IS_SET(sinkbuf, GST_BUFFER_FLAG_GAP)) {
		/*
		 * tell the world about state changes
		 */

		if(element->emit_signals && FALSE != element->last_state)
			g_signal_emit(G_OBJECT(element), signals[SIGNAL_STOP], 0, GST_BUFFER_TIMESTAMP(sinkbuf), NULL);
		element->last_state = FALSE;

		if(element->leaky) {
			/*
			 * discard buffer.  skipping an interval of
			 * non-zero length so next buffer must be a
			 * discont
			 */

			element->need_discont = TRUE;
			GST_DEBUG_OBJECT(element->srcpad, "discarding gap buffer %p %" GST_BUFFER_BOUNDARIES_FORMAT, sinkbuf, GST_BUFFER_BOUNDARIES_ARGS(sinkbuf));
			goto done;
		}

		/*
		 * is a discontinuity pending?
		 */

		if(element->need_discont && !GST_BUFFER_IS_DISCONT(sinkbuf)) {
			sinkbuf = gst_buffer_make_writable(sinkbuf);
			GST_BUFFER_FLAG_SET(sinkbuf, GST_BUFFER_FLAG_DISCONT);
		}
		element->need_discont = FALSE;

		/*
		 * push buffer
		 */

		GST_DEBUG_OBJECT(element->srcpad, "pushing reused gap buffer %p %" GST_BUFFER_BOUNDARIES_FORMAT, sinkbuf, GST_BUFFER_BOUNDARIES_ARGS(sinkbuf));
		result = gst_pad_push(element->srcpad, sinkbuf);
		if(G_UNLIKELY(result != GST_FLOW_OK))
			GST_WARNING_OBJECT(element->srcpad, "gst_pad_push() failed (%s)", gst_flow_get_name(result));
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

		g_mutex_lock(&element->control_lock);
		state = control_get_state(element, timestamp_add_offset(GST_BUFFER_TIMESTAMP(sinkbuf), (gint64) start - element->hold_length, element->rate), timestamp_add_offset(GST_BUFFER_TIMESTAMP(sinkbuf), (gint64) start + element->attack_length, element->rate));
		for(length = 1; start + length < sinkbuf_length; length++) {
			if(state != control_get_state(element, timestamp_add_offset(GST_BUFFER_TIMESTAMP(sinkbuf), (gint64) (start + length) - element->hold_length, element->rate), timestamp_add_offset(GST_BUFFER_TIMESTAMP(sinkbuf), (gint64) (start + length) + element->attack_length, element->rate)))
				break;
		}
		g_mutex_unlock(&element->control_lock);

		/*
		 * if the output state has changed, tell the world about it
		 */

		timestamp = GST_BUFFER_TIMESTAMP(sinkbuf) + gst_util_uint64_scale_int_round(GST_BUFFER_DURATION(sinkbuf), start, sinkbuf_length);
		if(element->emit_signals && state != element->last_state)
			g_signal_emit(G_OBJECT(element), signals[state ? SIGNAL_START : SIGNAL_STOP], 0, timestamp, NULL);
		element->last_state = state;

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
		 * output is a buffer of non-zero length.  if it's the
		 * entire input buffer then re-use it otherwise create a
		 * subbuffer from it
		 */

		if(length == sinkbuf_length) {
			GST_DEBUG_OBJECT(element, "reusing input buffer %p", sinkbuf);
			srcbuf = sinkbuf;
			sinkbuf = NULL;
		} else {
			GST_DEBUG_OBJECT(element, "creating sub-buffer from samples [%" G_GUINT64_FORMAT ", %" G_GUINT64_FORMAT ")", start, start + length);
			srcbuf = gst_buffer_copy_region(sinkbuf, GST_BUFFER_COPY_FLAGS | GST_BUFFER_COPY_MEMORY | GST_BUFFER_COPY_TIMESTAMPS, start * element->unit_size, length * element->unit_size);
			if(!srcbuf) {
				GST_ERROR_OBJECT(element, "failure creating sub-buffer");
				result = GST_FLOW_ERROR;
				goto done;
			}

			/*
			 * set offset, and timestamps
			 */

			GST_BUFFER_OFFSET(srcbuf) = GST_BUFFER_OFFSET(sinkbuf) + start;
			GST_BUFFER_OFFSET_END(srcbuf) = GST_BUFFER_OFFSET(srcbuf) + length;
			GST_BUFFER_TIMESTAMP(srcbuf) = timestamp;
			GST_BUFFER_DURATION(srcbuf) = GST_BUFFER_TIMESTAMP(sinkbuf) + gst_util_uint64_scale_int_round(GST_BUFFER_DURATION(sinkbuf), start + length, sinkbuf_length) - GST_BUFFER_TIMESTAMP(srcbuf);
		}

		/*
		 * is a discontinuity pending?
		 */

		if(!!element->need_discont != !!GST_BUFFER_IS_DISCONT(srcbuf)) {
			srcbuf = gst_buffer_make_writable(srcbuf);
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
			srcbuf = gst_buffer_make_writable(srcbuf);
			GST_BUFFER_FLAG_SET(srcbuf, GST_BUFFER_FLAG_GAP);
		}

		/*
		 * push buffer down stream
		 */

		GST_DEBUG_OBJECT(element->srcpad, "pushing buffer %p %" GST_BUFFER_BOUNDARIES_FORMAT, srcbuf, GST_BUFFER_BOUNDARIES_ARGS(srcbuf));
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

	/* only one of two outcomes:  OK or ERROR */
	return result == GST_FLOW_OK ? result : GST_FLOW_ERROR;
}


/*
 * event()
 */


static gboolean sink_event(GstPad *pad, GstObject *parent, GstEvent *event)
{
	GSTLALGate *element = GSTLAL_GATE(parent);
	gboolean success = TRUE;

	switch(GST_EVENT_TYPE(event)) {
	case GST_EVENT_STREAM_START:
		GST_DEBUG_OBJECT(pad, "new segment;  clearing end-of-stream flag");
		g_mutex_lock(&element->control_lock);
		element->t_sink_head = GST_CLOCK_TIME_NONE;
		element->sink_eos = FALSE;
		element->last_state = -1;	/* force signal on initial state */
		element->need_discont = TRUE;
		g_mutex_unlock(&element->control_lock);
		break;

	case GST_EVENT_EOS:
		GST_DEBUG_OBJECT(pad, "end-of-stream;  setting end-of-stream flag and flushing control queue");
		g_mutex_lock(&element->control_lock);
		element->sink_eos = TRUE;
		control_flush(element);
		g_cond_broadcast(&element->control_queue_head_changed);
		g_mutex_unlock(&element->control_lock);
		break;

	case GST_EVENT_CAPS: {
		GstCaps *caps;
		GstAudioInfo info;
		gst_event_parse_caps(event, &caps);
		success = gstlal_audio_info_from_caps(&info, caps);
		if(success) {
			gint old_rate = element->rate;
			element->rate = GST_AUDIO_INFO_RATE(&info);
			element->unit_size = GST_AUDIO_INFO_BPF(&info);
			if(element->rate != old_rate)
				g_signal_emit(parent, signals[SIGNAL_RATE_CHANGED], 0, element->rate, NULL);
		}
		break;
	}

	default:
		break;
	}

	/*
	 * sink events are forwarded to src pad
	 */

	if(!success)
		gst_event_unref(event);
	else
		success = gst_pad_event_default(pad, parent, event);

	return success;
}


/*
 * ============================================================================
 *
 *                                 Source Pad
 *
 * ============================================================================
 */


/*
 * src_event()
 *
 * upstream events should be forwarded through both the sink and control
 * pads.
 */


static gboolean src_event(GstPad *pad, GstObject *parent, GstEvent *event)
{
	gboolean success = TRUE;
	/* each push_event() consumes one reference, so we need an extra */
	gst_event_ref(event);
	success &= gst_pad_push_event(GSTLAL_GATE(parent)->controlpad, event);
	success &= gst_pad_push_event(GSTLAL_GATE(parent)->sinkpad, event);
	return success;
}


/*
 * src_query()
 *
 * queries are referred to the sink pad's peer for the answer
 */


static gboolean src_query(GstPad *pad, GstObject *parent, GstQuery *query)
{
	return gst_pad_peer_query(GSTLAL_GATE(parent)->sinkpad, query);
}


/*
 * ============================================================================
 *
 *                              GObject Methods
 *
 * ============================================================================
 */


/*
 * properties
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
 * finalize()
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
	g_mutex_clear(&element->control_lock);
	g_array_unref(element->control_segments);
	element->control_segments = NULL;
	g_cond_clear(&element->control_queue_head_changed);

	G_OBJECT_CLASS(gstlal_gate_parent_class)->finalize(object);
}


/*
 * class_init()
 */


#define CAPS \
	GST_AUDIO_CAPS_MAKE("{ S8, " GST_AUDIO_NE(S16) ", " GST_AUDIO_NE(S32) ", " GST_AUDIO_NE(S64) ", U8, " GST_AUDIO_NE(U16) ", " GST_AUDIO_NE(U32) ", " GST_AUDIO_NE(U64) ", " GST_AUDIO_NE(F32) ", " GST_AUDIO_NE(F64) ", " GST_AUDIO_NE(Z64) ", " GST_AUDIO_NE(Z128) "}") ", " \
	"layout = (string) interleaved, " \
	"channel-mask = (bitmask) 0"


static void gstlal_gate_class_init(GSTLALGateClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	klass->rate_changed = GST_DEBUG_FUNCPTR(rate_changed);
	klass->start = GST_DEBUG_FUNCPTR(start);
	klass->stop = GST_DEBUG_FUNCPTR(stop);

	gst_element_class_set_details_simple(
		element_class,
		"Gate",
		"Filter",
		"Flag buffers as gaps based on the value of a control input",
		"Kipp Cannon <kipp.cannon@ligo.org>, Chad Hanna <channa@ligo.caltech.edu>"
	);

	/* no 64-bit int support for control because cannot specify
	 * threshold to that precision */
	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"control",
			GST_PAD_SINK,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
				"audio/x-raw, " \
				"rate = " GST_AUDIO_RATE_RANGE ", " \
				"channels = (int) 1, " \
				"format = (string) { S8, " GST_AUDIO_NE(S16) ", " GST_AUDIO_NE(S32) ", " GST_AUDIO_NE(S64) ", U8, " GST_AUDIO_NE(U16) ", " GST_AUDIO_NE(U32) ", " GST_AUDIO_NE(U64) ", " GST_AUDIO_NE(F32) ", " GST_AUDIO_NE(F64) ", " GST_AUDIO_NE(Z64) ", " GST_AUDIO_NE(Z128) "}, " \
				"layout = (string) interleaved, " \
				"channel-mask = (bitmask) 0"
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
			0, INFINITY, DEFAULT_THRESHOLD,
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
		g_cclosure_marshal_VOID__LONG,
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
		g_cclosure_marshal_VOID__LONG,
		G_TYPE_NONE,
		1,
		G_TYPE_UINT64
	);
}


/*
 * init()
 */


static void gstlal_gate_init(GSTLALGate *element)
{
	GstPad *pad;

	gst_element_create_all_pads(GST_ELEMENT(element));

	/* control pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "control");
	gst_pad_set_chain_function(pad, GST_DEBUG_FUNCPTR(control_chain));
	gst_pad_set_event_function(pad, GST_DEBUG_FUNCPTR(control_event));
	element->controlpad = pad;

	/* sink pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "sink");
	gst_pad_set_chain_function(pad, GST_DEBUG_FUNCPTR(sink_chain));
	gst_pad_set_event_function(pad, GST_DEBUG_FUNCPTR(sink_event));
	GST_PAD_SET_PROXY_ALLOCATION(pad);
	GST_PAD_SET_PROXY_CAPS(pad);
	GST_PAD_SET_PROXY_SCHEDULING(pad);
	element->sinkpad = pad;

	/* src pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "src");
	gst_pad_set_event_function(pad, GST_DEBUG_FUNCPTR(src_event));
	gst_pad_set_query_function(pad, GST_DEBUG_FUNCPTR(src_query));
	element->srcpad = pad;

	/* internal data */
	g_mutex_init(&element->control_lock);
	element->control_eos = FALSE;
	element->sink_eos = FALSE;
	element->t_sink_head = GST_CLOCK_TIME_NONE;
	element->control_segments = g_array_new(FALSE, FALSE, sizeof(struct control_segment));
	g_cond_init(&element->control_queue_head_changed);
	element->control_sample_func = NULL;
	element->last_state = -1;	/* force signal on initial state */
	element->rate = 0;
	element->unit_size = 0;
	element->control_rate = 0;
	element->need_discont = FALSE;
}

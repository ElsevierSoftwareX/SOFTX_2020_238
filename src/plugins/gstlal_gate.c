/*
 * An element to flag buffers in a stream as silence or not based on the
 * value of a control input.
 *
 * Copyright (C) 2008  Kipp Cannon, Chad Hanna
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


/*
 * stuff from glib/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>


/*
 * our own stuff
 */


#include <gstlal.h>
#include <gstlal_gate.h>


#define GST_CAT_DEFAULT gstlal_gate_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


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
	if(offset >= 0)
		return t + gst_util_uint64_scale_int_round(offset, GST_SECOND, rate);
	return t - gst_util_uint64_scale_int_round(-offset, GST_SECOND, rate);
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
 * gst_buffer_unref() wrapper for use with g_list_foreach()
 */


static void g_list_foreach_gst_buffer_unref(gpointer data, gpointer not_used)
{
	gst_buffer_unref(GST_BUFFER(data));
}


/*
 * unref the control buffer, signal it being flushed.  must be called with
 * the control lock held
 */


static void control_flush(GSTLALGate *element)
{
	g_queue_foreach(element->control_queue, g_list_foreach_gst_buffer_unref, NULL);
	g_queue_clear(element->control_queue);
	g_cond_broadcast(element->control_availability);
}


/*
 * wait for the control queue to span the requested interval.  must be
 * called with the control lock held
 */


static void control_get_interval(GSTLALGate *element, GstClockTime tmin, GstClockTime tmax)
{
	while(1) {
		GstBuffer *buf;
		gboolean flushed;

		/*
		 * flush old data from tail
		 */

		buf = g_queue_peek_tail(element->control_queue);
		flushed = FALSE;
		while(buf && (GST_BUFFER_TIMESTAMP(buf) + GST_BUFFER_DURATION(buf) <= tmin)) {
			GST_DEBUG_OBJECT(element, "flushing control queue to %" GST_TIME_SECONDS_FORMAT, GST_TIME_SECONDS_ARGS(GST_BUFFER_TIMESTAMP(buf) + GST_BUFFER_DURATION(buf)));
			g_queue_pop_tail(element->control_queue);
			gst_buffer_unref(buf);
			flushed = TRUE;
			buf = g_queue_peek_tail(element->control_queue);
		}
		if(flushed)
			g_cond_broadcast(element->control_availability);

		/*
		 * has head advanced far enough, or are we at EOS?
		 */

		buf = g_queue_peek_head(element->control_queue);
		if((buf && (GST_BUFFER_TIMESTAMP(buf) + GST_BUFFER_DURATION(buf) >= tmax)) || element->control_eos)
			return;

		/*
		 * no, wait for buffer to arrive
		 */

		GST_DEBUG_OBJECT(element, "waiting for control to advance to %" GST_TIME_SECONDS_FORMAT, GST_TIME_SECONDS_ARGS(tmax));
		g_cond_wait(element->control_availability, element->control_lock);
	}
}


/*
 * return the state of the control input over a range of timestamps.  must
 * be called with the control lock held.
 *
 * the return value is < 0 if there is no control data for the times
 * requested, or 0 if control data is available for (at least part of) the
 * times requested and is less than the threshold everywhere therein, or >
 * 0 if control data is available for (at least part of) the times
 * requested and is greather than or equal to the threshold for at least 1
 * sample therein.
 */


struct g_list_for_each_gst_buffer_peak_data {
	GSTLALGate *element;
	GstClockTime tmin, tmax;
	gdouble peak;
};


static void g_list_for_each_gst_buffer_peak(gpointer data, gpointer user_data)
{
	GstBuffer *buf = GST_BUFFER(data);
	GSTLALGate *element = ((struct g_list_for_each_gst_buffer_peak_data *) user_data)->element;
	GstClockTime tmin = ((struct g_list_for_each_gst_buffer_peak_data *) user_data)->tmin;
	GstClockTime tmax = ((struct g_list_for_each_gst_buffer_peak_data *) user_data)->tmax;
	guint64 offset, last;
	guint64 length = GST_BUFFER_OFFSET_END(buf) - GST_BUFFER_OFFSET(buf);
	gdouble peak = -1;

	if(tmin <= GST_BUFFER_TIMESTAMP(buf))
		offset = 0;
	else {
		offset = gst_util_uint64_scale_int_round(tmin - GST_BUFFER_TIMESTAMP(buf), element->control_rate, GST_SECOND);
		if(offset >= length)
			return;
	}

	if(tmax < GST_BUFFER_TIMESTAMP(buf))
		return;
	last = gst_util_uint64_scale_int_round(tmax - GST_BUFFER_TIMESTAMP(buf), element->control_rate, GST_SECOND);
	if(last <= offset)
		/* always test at least one sample */
		last = offset + 1;
	else if(last > length)
		last = length;

	if(!GST_BUFFER_FLAG_IS_SET(buf, GST_BUFFER_FLAG_GAP)) {
		for(; offset < last; offset++) {
			gdouble val = element->control_sample_func(GST_BUFFER_DATA(buf), offset);
			if(val > peak)
				peak = val;
		}
	} else {
		/* treat gaps as buffers of 0 */
		peak = 0.0;
	}

	if(peak > ((struct g_list_for_each_gst_buffer_peak_data *) user_data)->peak)
		((struct g_list_for_each_gst_buffer_peak_data *) user_data)->peak = peak;
}


static gint control_get_state(GSTLALGate *element, GstClockTime tmin, GstClockTime tmax)
{
	struct g_list_for_each_gst_buffer_peak_data data = {
		.element = element,
		.tmin = tmin,
		.tmax = tmax,
		.peak = -1
	};

	g_queue_foreach(element->control_queue, g_list_for_each_gst_buffer_peak, &data);

	return data.peak >= element->threshold ? +1 : data.peak >= 0 ? 0 : -1;
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
	ARG_LEAKY
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
		control_flush(element);
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

	if(!GST_BUFFER_TIMESTAMP_IS_VALID(sinkbuf) || !GST_BUFFER_OFFSET_IS_VALID(sinkbuf) || !GST_BUFFER_OFFSET_END_IS_VALID(sinkbuf)) {
		GST_ERROR_OBJECT(element, "error in control stream: buffer does not have valid timestamp and/or offset");
		gst_buffer_unref(sinkbuf);
		result = GST_FLOW_ERROR;
		goto done;
	}

	/*
	 * if there's already enough control data wait for it to be flushed
	 */

	g_mutex_lock(element->control_lock);
	/* FIXME:  check that this is right */
	/*while(!g_queue_is_empty(element->control_queue)) {
		GstBuffer *head = g_queue_peek_head(element->control_queue);
		GstBuffer *tail = g_queue_peek_tail(element->control_queue);
		if((gint64) gst_util_uint64_scale_int(GST_BUFFER_OFFSET_END(head) - GST_BUFFER_OFFSET_END(tail), element->rate, element->control_rate) < element->attack_length + element->hold_length + 1)
			break;
		GST_DEBUG_OBJECT(element, "waiting for space in control queue");
		g_cond_wait(element->control_availability, element->control_lock);
	}*/

	/*
	 * if we're at eos on sink pad, discard
	 */

	if(element->sink_eos) {
		GST_DEBUG_OBJECT(element, "sink is at end-of-stream, discarding control buffer");
		gst_buffer_unref(sinkbuf);
		result = GST_FLOW_OK;
		g_mutex_unlock(element->control_lock);
		goto done;
	}

	/*
	 * store this buffer
	 */

	g_queue_push_head(element->control_queue, sinkbuf);

	/*
	 * signal the buffer's availability
	 */

	GST_DEBUG_OBJECT(element, "new control buffer available");
	g_cond_broadcast(element->control_availability);
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
		g_mutex_lock(element->control_lock);
		GST_DEBUG_OBJECT(pad, "new segment;  clearing end-of-stream flag and flushing queue");
		element->control_eos = FALSE;
		control_flush(element);
		g_mutex_unlock(element->control_lock);
		break;

	case GST_EVENT_EOS:
		g_mutex_lock(element->control_lock);
		if(element->sink_eos) {
			GST_DEBUG_OBJECT(pad, "end-of-stream;  sink is at end-of-stream, flushing queue");
			control_flush(element);
		}
		GST_DEBUG_OBJECT(pad, "end-of-stream;  setting end-of-stream flag");
		element->control_eos = TRUE;
		g_cond_broadcast(element->control_availability);
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


static GstCaps *sink_getcaps(GstPad * pad)
{
	GSTLALGate *element = GSTLAL_GATE(gst_pad_get_parent(pad));
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

	peercaps = gst_pad_peer_get_caps(element->srcpad);
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
		if(rate != element->rate)
			g_signal_emit(G_OBJECT(element), signals[SIGNAL_RATE_CHANGED], 0, rate, NULL);
		element->rate = rate;
		element->unit_size = width / 8 * channels;
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

	g_mutex_lock(element->control_lock);

	/*
	 * check validity of timestamp and offsets
	 */

	if(!GST_BUFFER_TIMESTAMP_IS_VALID(sinkbuf) || !GST_BUFFER_DURATION_IS_VALID(sinkbuf) || !GST_BUFFER_OFFSET_IS_VALID(sinkbuf) || !GST_BUFFER_OFFSET_END_IS_VALID(sinkbuf)) {
		GST_ERROR_OBJECT(element, "error in input stream: buffer has invalid timestamp and/or offset");
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

	control_get_interval(element, timestamp_add_offset(GST_BUFFER_TIMESTAMP(sinkbuf), -element->hold_length, element->rate), timestamp_add_offset(GST_BUFFER_TIMESTAMP(sinkbuf), sinkbuf_length + element->attack_length, element->rate));

	/*
	 * loop over the contents of the input buffer.
	 */

	for(start = 0; start < sinkbuf_length; start += length) {
		GstClockTime timestamp;

		/*
		 * -1 = unknown, 0 = off, 1 = on
		 */

		gint state = -1;

		/*
		 * find the next interval of continuous control state
		 */

		for(length = 0; start + length < sinkbuf_length; length++) {
			gint state_now = control_get_state(element, timestamp_add_offset(GST_BUFFER_TIMESTAMP(sinkbuf), (gint64) (start + length) - element->hold_length, element->rate), timestamp_add_offset(GST_BUFFER_TIMESTAMP(sinkbuf), (gint64) (start + length) + element->attack_length, element->rate));
			if(length == 0)
				/*
				 * state for this interval
				 */

				state = state_now;
			else if(state != state_now)
				/*
				 * control state has changed
				 */

				break;
		}

		/*
		 * apply default state if needed
		 */

		if(state < 0)
			state = element->default_state;

		/*
		 * if the interval has non-zero length and should not be
		 * leaked, build a buffer out of it and push down stream.
		 */

		if(!length)
			continue;
		timestamp = GST_BUFFER_TIMESTAMP(sinkbuf) + gst_util_uint64_scale_int_round(GST_BUFFER_DURATION(sinkbuf), start, sinkbuf_length);
		if(element->emit_signals && state != element->last_state) {
			g_signal_emit(G_OBJECT(element), signals[state > 0 ? SIGNAL_START : SIGNAL_STOP], 0, timestamp, NULL);
			element->last_state = state;
		}
		if(state > 0 || !element->leaky) {
			GstBuffer *srcbuf = gst_buffer_create_sub(sinkbuf, start * element->unit_size, length * element->unit_size);
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

			/*
			 * is a discontinuity pending?
			 */

			if(element->need_discont) {
				GST_BUFFER_FLAG_SET(srcbuf, GST_BUFFER_FLAG_DISCONT);
				element->need_discont = FALSE;
			} else
				GST_BUFFER_FLAG_UNSET(srcbuf, GST_BUFFER_FLAG_DISCONT);

			/*
			 * if control input was below threshold or
			 * unavailable then flag buffer as silence.
			 */

			if(state <= 0)
				GST_BUFFER_FLAG_SET(srcbuf, GST_BUFFER_FLAG_GAP);

			/*
			 * push buffer down stream
			 */

			result = gst_pad_push(element->srcpad, srcbuf);
			if(G_UNLIKELY(result != GST_FLOW_OK)) {
				GST_WARNING_OBJECT(element, "%s: gst_pad_push() failed (%s)", GST_PAD_NAME(element->srcpad), gst_flow_get_name(result));
				goto done;
			}
		} else {
			/*
			 * skipping an interval with non-zero length, next
			 * buffer must be a discont
			 */

			element->need_discont = TRUE;
		}
	}

	/*
	 * done
	 */

done:
	g_mutex_unlock(element->control_lock);
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
		g_mutex_lock(element->control_lock);
		GST_DEBUG_OBJECT(pad, "new segment;  clearing internal end-of-stream flag");
		element->sink_eos = FALSE;
		element->last_state = FALSE;
		element->need_discont = TRUE;
		g_mutex_unlock(element->control_lock);
		break;

	case GST_EVENT_EOS:
		g_mutex_lock(element->control_lock);
		GST_DEBUG_OBJECT(pad, "end-of-stream;  setting internal end-of-stream flag and flushing control buffer");
		element->sink_eos = TRUE;
		control_flush(element);
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
	g_cond_free(element->control_availability);
	element->control_availability = NULL;
	if(element->control_queue) {
		g_queue_foreach(element->control_queue, g_list_foreach_gst_buffer_unref, NULL);
		g_queue_clear(element->control_queue);
		g_queue_free(element->control_queue);
		element->control_queue = NULL;
	}

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
			"Emit start and stop signals (rate-changed is always emited).",
			DEFAULT_EMIT_SIGNALS,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
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
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_THRESHOLD,
		g_param_spec_double(
			"threshold",
			"Threshold",
			"Control threshold",
			0, G_MAXDOUBLE, DEFAULT_THRESHOLD,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_ATTACK_LENGTH,
		g_param_spec_int64(
			"attack-length",
			"Attack",
			"Number of samples ahead of negative-to-positive threshold crossing to include in output.",
			G_MININT64, G_MAXINT64, DEFAULT_ATTACK_LENGTH,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_HOLD_LENGTH,
		g_param_spec_int64(
			"hold-length",
			"Hold",
			"Number of samples following positive-to-negative threshold crossing to include in output.",
			G_MININT64, G_MAXINT64, DEFAULT_HOLD_LENGTH,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
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
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
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
	gst_pad_set_getcaps_function(pad, GST_DEBUG_FUNCPTR(sink_getcaps));
	gst_pad_set_setcaps_function(pad, GST_DEBUG_FUNCPTR(sink_setcaps));
	gst_pad_set_chain_function(pad, GST_DEBUG_FUNCPTR(sink_chain));
	gst_pad_set_event_function(pad, GST_DEBUG_FUNCPTR(sink_event));
	element->sinkpad = pad;

	/* retrieve (and ref) src pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "src");
	gst_pad_set_event_function(pad, GST_DEBUG_FUNCPTR(src_event));
	element->srcpad = pad;

	/* internal data */
	element->control_lock = g_mutex_new();
	element->control_availability = g_cond_new();
	element->control_eos = FALSE;
	element->sink_eos = FALSE;
	element->control_queue = g_queue_new();
	element->control_sample_func = NULL;
	element->emit_signals = DEFAULT_EMIT_SIGNALS;
	element->default_state = DEFAULT_DEFAULT_STATE;
	element->last_state = FALSE;
	element->threshold = DEFAULT_THRESHOLD;
	element->attack_length = DEFAULT_ATTACK_LENGTH;
	element->hold_length = DEFAULT_HOLD_LENGTH;
	element->leaky = DEFAULT_LEAKY;
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
		GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "gate", 0, "gate element");
	}

	return type;
}

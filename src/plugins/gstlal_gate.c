/*
 * An element to flag buffers in a stream as silence or not based on the
 * value of a control input.
 *
 * Copyright (C) 2008  Kipp Cannon, Chad Hanna
 *
 *  This program is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU General Public License as published by the
 *  Free Software Foundation; either version 2 of the License, or (at your
 *  option) any later version.
 *
 *  This program is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */


/*
 * ========================================================================
 *
 *                                  Preamble
 *
 * ========================================================================
 */


/*
 * struff from the C library
 */


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


/*
 * ============================================================================
 *
 *                                 Parameters
 *
 * ============================================================================
 */


#define DEFAULT_THRESHOLD 0


/*
 * ============================================================================
 *
 *                             Utility Functions
 *
 * ============================================================================
 */


/*
 * unref the control buffer, signal it being flushed.  must be called with
 * the control lock held.
 */


static void control_flush(GSTLALGate *element)
{
	if(element->control_buf) {
		gst_buffer_unref(element->control_buf);
		element->control_buf = NULL;
	}
	element->control_length = GST_BUFFER_OFFSET_NONE;
	g_cond_signal(element->control_flushed);
}


/*
 * return the state of the control input at the given timestamp.  the
 * return value is < 0 for times outside the interval spanned by the
 * currently-queued control buffer, 0 if the control buffer is < the
 * threshold at the given time, and > 0 if the control buffer is >= the
 * threshold at the given time.
 */


static double control_sample_int8(const GSTLALGate *element, guint64 sample)
{
	return ((const gint8 *) GST_BUFFER_DATA(element->control_buf))[sample];
}


static double control_sample_uint8(const GSTLALGate *element, guint64 sample)
{
	return ((const guint8 *) GST_BUFFER_DATA(element->control_buf))[sample];
}


static double control_sample_int16(const GSTLALGate *element, guint64 sample)
{
	return ((const gint16 *) GST_BUFFER_DATA(element->control_buf))[sample];
}


static double control_sample_uint16(const GSTLALGate *element, guint64 sample)
{
	return ((const guint16 *) GST_BUFFER_DATA(element->control_buf))[sample];
}


static double control_sample_int32(const GSTLALGate *element, guint64 sample)
{
	return ((const gint32 *) GST_BUFFER_DATA(element->control_buf))[sample];
}


static double control_sample_uint32(const GSTLALGate *element, guint64 sample)
{
	return ((const guint32 *) GST_BUFFER_DATA(element->control_buf))[sample];
}


static double control_sample_float32(const GSTLALGate *element, guint64 sample)
{
	return ((const float *) GST_BUFFER_DATA(element->control_buf))[sample];
}


static double control_sample_float64(const GSTLALGate *element, guint64 sample)
{
	return ((const double *) GST_BUFFER_DATA(element->control_buf))[sample];
}


static gint control_get_state(GSTLALGate *element, guint64 offset)
{
	return fabs(element->control_sample_func(element, offset)) >= element->threshold;
}


static guint64 control_t_to_offset(GSTLALGate *element, GstClockTime t)
{
	guint64 offset;

	if(G_UNLIKELY(t < GST_BUFFER_TIMESTAMP(element->control_buf))) {
		/* t precedes start of buffer, but it might round to sample
		 * offset 0 so check (but anything else is out-of-bounds) */
		offset = gst_util_uint64_scale_int_round(GST_BUFFER_TIMESTAMP(element->control_buf) - t, element->control_rate, GST_SECOND);
		if(offset != 0)
			return GST_BUFFER_OFFSET_NONE;
	} else {
		offset = gst_util_uint64_scale_int_round(t - GST_BUFFER_TIMESTAMP(element->control_buf), element->control_rate, GST_SECOND);
		if(offset >= element->control_length)
			return GST_BUFFER_OFFSET_NONE;
	}

	return offset;
}


/*
 * ============================================================================
 *
 *                                 Properties
 *
 * ============================================================================
 */


enum property {
	ARG_THRESHOLD = 1
};


static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec)
{
	GSTLALGate *element = GSTLAL_GATE(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_THRESHOLD:
		element->threshold = g_value_get_double(value);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *pspec)
{
	GSTLALGate *element = GSTLAL_GATE(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_THRESHOLD:
		g_value_set_double(value, element->threshold);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * ============================================================================
 *
 *                                    Caps
 *
 * ============================================================================
 */


/*
 * control pad
 */


static gboolean control_setcaps(GstPad *pad, GstCaps *caps)
{
	GSTLALGate *element = GSTLAL_GATE(gst_pad_get_parent(pad));
	GstStructure *structure;
	const gchar *media_type;
	double (*control_sample_func)(const struct _GSTLALGate *, size_t) = NULL;
	gint rate;
	gint width;
	gboolean success = TRUE;

	/*
	 * parse caps
	 */

	structure = gst_caps_get_structure(caps, 0);
	media_type = gst_structure_get_name(structure);
	if(!gst_structure_get_int(structure, "width", &width))
		success = FALSE;
	if(!gst_structure_get_int(structure, "rate", &rate))
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
	} else if(!strcmp(media_type, "audio/x-raw-int")) {
		gboolean is_signed = TRUE;
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
		element->control_sample_func = control_sample_func;
		element->control_rate = rate;
	}

	/*
	 * done.
	 */

	gst_object_unref(element);
	return success;
}


/*
 * sink pad
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
		GstCaps *intersection = gst_caps_intersect(peercaps, caps);
		gst_caps_unref(peercaps);
		gst_caps_unref(caps);
		caps = intersection;
	}

	/*
	 * done
	 */

	gst_object_unref(element);
	return caps;
}


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
	 * update the element metadata
	 */

	if(success) {
		element->rate = rate;
		element->bytes_per_sample = width / 8 * channels;
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
 *                                   Chain
 *
 * ============================================================================
 */


/*
 * control pad
 */


static GstFlowReturn control_chain(GstPad *pad, GstBuffer *sinkbuf)
{
	GSTLALGate *element = GSTLAL_GATE(gst_pad_get_parent(pad));
	GstFlowReturn result = GST_FLOW_OK;

	/*
	 * check validity of timestamp and offsets
	 */

	if(!GST_BUFFER_TIMESTAMP_IS_VALID(sinkbuf) || !GST_BUFFER_OFFSET_IS_VALID(sinkbuf) || !GST_BUFFER_OFFSET_END_IS_VALID(sinkbuf)) {
		GST_ERROR_OBJECT(element, "error in control stream: buffer has invalid timestamp and/or offset");
		gst_buffer_unref(sinkbuf);
		result = GST_FLOW_ERROR;
		goto done;
	}

	/*
	 * if there's already a buffer stored, wait for it to be flushed
	 */

	g_mutex_lock(element->control_lock);
	while(element->control_buf)
		g_cond_wait(element->control_flushed, element->control_lock);

	/*
	 * store this buffer, extract some metadata
	 */

	element->control_buf = sinkbuf;
	element->control_length = GST_BUFFER_OFFSET_END(sinkbuf) - GST_BUFFER_OFFSET(sinkbuf);

	/*
	 * signal the buffer's availability
	 */

	g_cond_signal(element->control_available);
	g_mutex_unlock(element->control_lock);

	/*
	 * done
	 */

done:
	gst_object_unref(element);
	return result;
}


/*
 * sink pad
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

	if(!GST_BUFFER_TIMESTAMP_IS_VALID(sinkbuf) || !GST_BUFFER_OFFSET_IS_VALID(sinkbuf) || !GST_BUFFER_OFFSET_END_IS_VALID(sinkbuf)) {
		GST_ERROR_OBJECT(element, "error in input stream: buffer has invalid timestamp and/or offset");
		result = GST_FLOW_ERROR;
		goto done;
	}

	sinkbuf_length = GST_BUFFER_OFFSET_END(sinkbuf) - GST_BUFFER_OFFSET(sinkbuf);

	/*
	 * loop over the contents of the input buffer.
	 */

	for(start = 0; start < sinkbuf_length; start += length) {
		gint state = -1;	/* initialize to silence warnings */

		/*
		 * find the next interval of continuous control state
		 */

		g_mutex_lock(element->control_lock);
		for(length = 0; start + length < sinkbuf_length; length++) {
			GstClockTime t = GST_BUFFER_TIMESTAMP(sinkbuf) + gst_util_uint64_scale_int_round(start + length, GST_SECOND, element->rate);
			guint64 control_offset;

			/*
			 * if there is no control buffer available or the input has
			 * advanced beyond its end, flush the control buffer and
			 * wait for one that overlaps the input data
			 */

			while(1) {
				while(!element->control_buf)
					g_cond_wait(element->control_available, element->control_lock);
				control_offset = control_t_to_offset(element, t);
				if(control_offset != GST_BUFFER_OFFSET_NONE)
					break;
				control_flush(element);
			}

			/*
			 * check the state of the control input
			 */

			if(length == 0) {
				/*
				 * the control state for this interval
				 */

				state = control_get_state(element, control_offset);
			} else if(control_get_state(element, control_offset) != state)
				/*
				 * control state has changed
				 */

				break;
		}
		g_mutex_unlock(element->control_lock);

		/*
		 * if the interval has non-zero length, build a buffer out
		 * of it and push down stream.
		 */

		if(length) {
			GstBuffer *srcbuf = gst_buffer_create_sub(sinkbuf, start * element->bytes_per_sample, length * element->bytes_per_sample);
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
			GST_BUFFER_TIMESTAMP(srcbuf) = GST_BUFFER_TIMESTAMP(sinkbuf) + gst_util_uint64_scale_int_round(start, GST_SECOND, element->rate);
			GST_BUFFER_DURATION(srcbuf) = GST_BUFFER_TIMESTAMP(sinkbuf) + gst_util_uint64_scale_int_round(start + length, GST_SECOND, element->rate) - GST_BUFFER_TIMESTAMP(srcbuf);

			/*
			 * only the first subbuffer of a buffer flagged as
			 * a discontinuity is a discontinuity.
			 */

			if(start)
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
			if(result != GST_FLOW_OK)
				goto done;
		}
	}

	/*
	 * done
	 */

done:
	gst_buffer_unref(sinkbuf);
	gst_object_unref(element);
	return result;
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
	g_cond_free(element->control_available);
	element->control_available = NULL;
	g_cond_free(element->control_flushed);
	element->control_flushed = NULL;
	if(element->control_buf) {
		gst_buffer_unref(element->control_buf);
		element->control_buf = NULL;
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
	"rate = (int) [ 1, MAX ], " \
	"channels = (int) [ 1, MAX ], " \
	"endianness = (int) BYTE_ORDER, " \
	"width = (int) {8, 16, 32, 64}, " \
	"signed = (boolean) {true, false} ; " \
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


static void base_init(gpointer class)
{
	static GstElementDetails plugin_details = {
		"Gate",
		"Filter",
		"Flag buffers as gaps based on the value of a control input",
		"Kipp Cannon <kcannon@ligo.caltech.edu>, Chad Hanna <channa@ligo.caltech.edu>"
	};
	GstElementClass *element_class = GST_ELEMENT_CLASS(class);

	gst_element_class_set_details(element_class, &plugin_details);

	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"control",
			GST_PAD_SINK,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
				"audio/x-raw-int, " \
				"rate = (int) [ 1, MAX ], " \
				"channels = (int) 1, " \
				"endianness = (int) BYTE_ORDER, " \
				"width = (int) {8, 16, 32, 64}, " \
				"signed = (boolean) {true, false} ; " \
				"audio/x-raw-float, " \
				"rate = (int) [ 1, MAX ], " \
				"channels = (int) 1, " \
				"endianness = (int) BYTE_ORDER, " \
				"width = (int) {32, 64}"
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


static void class_init(gpointer class, gpointer class_data)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(class);

	parent_class = g_type_class_ref(GST_TYPE_ELEMENT);

	gobject_class->set_property = set_property;
	gobject_class->get_property = get_property;
	gobject_class->finalize = finalize;

	g_object_class_install_property(gobject_class, ARG_THRESHOLD, g_param_spec_double("threshold", "Threshold", "Control input threshold", 0, G_MAXDOUBLE, DEFAULT_THRESHOLD, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
}


/*
 * Instance init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GInstanceInitFunc
 */


static void instance_init(GTypeInstance *object, gpointer class)
{
	GSTLALGate *element = GSTLAL_GATE(object);
	GstPad *pad;

	gst_element_create_all_pads(GST_ELEMENT(element));

	/* configure (and ref) control pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "control");
	gst_pad_set_setcaps_function(pad, control_setcaps);
	gst_pad_set_chain_function(pad, control_chain);
	element->controlpad = pad;

	/* configure (and ref) sink pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "sink");
	gst_pad_set_getcaps_function(pad, sink_getcaps);
	gst_pad_set_setcaps_function(pad, sink_setcaps);
	gst_pad_set_chain_function(pad, sink_chain);
	element->sinkpad = pad;

	/* retrieve (and ref) src pad */
	element->srcpad = gst_element_get_static_pad(GST_ELEMENT(element), "src");

	/* internal data */
	element->control_lock = g_mutex_new();
	element->control_available = g_cond_new();
	element->control_flushed = g_cond_new();
	element->control_buf = NULL;
	element->control_length = GST_BUFFER_OFFSET_NONE;
	element->control_sample_func = NULL;
	element->threshold = DEFAULT_THRESHOLD;
	element->rate = 0;
	element->bytes_per_sample = 0;
	element->control_rate = 0;
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

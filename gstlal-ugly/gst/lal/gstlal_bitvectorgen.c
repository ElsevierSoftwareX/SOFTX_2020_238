/*
 * Copyright (C) 2012 Kipp Cannon <kipp.cannon@ligo.org>, Chad Hanna <chad.hanna@ligo.org>
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
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


/*
 * stuff from the C library
 */


#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>


/*
 *  stuff from gobject/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/audio/audio.h>
#include <gst/base/gstbasetransform.h>


/*
 * our own stuff
 */


#include <gstlal_bitvectorgen.h>


/*
 * ============================================================================
 *
 *                                 Parameters
 *
 * ============================================================================
 */


#define DEFAULT_EMIT_SIGNALS FALSE
#define DEFAULT_THRESHOLD 0
#define DEFAULT_ATTACK_LENGTH 0
#define DEFAULT_HOLD_LENGTH 0
#define DEFAULT_INVERT FALSE
#define DEFAULT_NONGAP_IS_CONTROL FALSE
#define DEFAULT_BIT_VECTOR 0xffffffff


/*
 * ============================================================================
 *
 *                           GStreamer Boiler Plate
 *
 * ============================================================================
 */


#define GST_CAT_DEFAULT gstlal_bitvectorgen_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


G_DEFINE_TYPE_WITH_CODE(
	GSTLALBitVectorGen,
	gstlal_bitvectorgen,
	GST_TYPE_BASE_TRANSFORM,
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "lal_bitvectorgen", 0, "lal_bitvectorgen element")
);


/*
 * ============================================================================
 *
 *                                  Signals
 *
 * ============================================================================
 */


enum gstlal_bitvectorgen_signal {
	SIGNAL_RATE_CHANGED,
	SIGNAL_START,
	SIGNAL_STOP,
	NUM_SIGNALS
};


static guint signals[NUM_SIGNALS] = {0, };


static void rate_changed_handler(GSTLALBitVectorGen *element, gint rate, void *data)
{
	/* FIXME:  do something? */
}


static void start_handler(GSTLALBitVectorGen *element, guint64 timestamp, void *data)
{
	/* FIXME:  do something? */
}


static void stop_handler(GSTLALBitVectorGen *element, guint64 timestamp, void *data)
{
	/* FIXME:  do something? */
}


/*
 * ============================================================================
 *
 *                             I/O Type Handling
 *
 * ============================================================================
 */


/*
 * array type cast macros:  interpret contents of an array as various C
 * types, retrieve value at given offset, compute magnitude, and cast to
 * double-precision float
 */


static gdouble get_input_int8(void **in)
{
	return abs(*(*(gint8 **) in)++);
}


static gdouble get_input_uint8(void **in)
{
	return *(*(guint8 **) in)++;
}


static gdouble get_input_int16(void **in)
{
	return abs(*(*(gint16 **) in)++);
}


static gdouble get_input_uint16(void **in)
{
	return *(*(guint16 **) in)++;
}


static gdouble get_input_int32(void **in)
{
	return abs(*(*(gint32 **) in)++);
}


static gdouble get_input_uint32(void **in)
{
	return *(*(guint32 **) in)++;
}


static gdouble get_input_float32(void **in)
{
	return fabsf(*(*(float **) in)++);
}


static gdouble get_input_float64(void **in)
{
	return fabs(*(*(double **) in)++);
}


static gdouble get_input_complex64(void **in)
{
	return cabsf(*(*(float complex **) in)++);
}


static gdouble get_input_complex128(void **in)
{
	return cabs(*(*(double complex **) in)++);
}


/*
 * array type cast macros:  interpret contents of an array as various C
 * types, set value at given offset
 */


static void set_output_uint8(void **out, guint32 bit_vector)
{
	*(*(guint8 **) out)++ = bit_vector;
}


static void set_output_uint16(void **out, guint32 bit_vector)
{
	*(*(guint16 **) out)++ = bit_vector;
}


static void set_output_uint32(void **out, guint32 bit_vector)
{
	*(*(guint32 **) out)++ = bit_vector;
}


/*
 * ============================================================================
 *
 *                     GstBaseTransform Method Overrides
 *
 * ============================================================================
 */


/*
 * get_unit_size()
 */


static gboolean get_unit_size(GstBaseTransform *trans, GstCaps *caps, gsize *size)
{
	GstAudioInfo info;
	gboolean success = TRUE;

	success &= gst_audio_info_from_caps(&info, caps);

	if(success)
		*size = GST_AUDIO_INFO_BPF(&info);
	else
		GST_WARNING_OBJECT(trans, "unable to parse caps %" GST_PTR_FORMAT, caps);

	return success;
}


/*
 * transform_caps()
 */


static GstCaps *transform_caps(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, GstCaps *filter)
{
	GstCaps *othercaps = NULL;
	guint i;

	switch(direction) {
	case GST_PAD_SRC:
		/*
		 * sink pad must have same sample rate as source pad.
		 * FIXME:  this doesn't work out all the allowed
		 * permutations, it just takes the rate from the first
		 * structure on the source pad and copies it into all the
		 * structures on the sink pad
		 */

		othercaps = gst_caps_copy(gst_pad_get_pad_template_caps(GST_BASE_TRANSFORM_SINK_PAD(trans)));
		for(i = 0; i < gst_caps_get_size(othercaps); i++)
			gst_structure_set_value(gst_caps_get_structure(othercaps, i), "rate", gst_structure_get_value(gst_caps_get_structure(caps, 0), "rate"));
		break;

	case GST_PAD_SINK:
		/*
		 * source pad must have same sample rate as sink pad.
		 * FIXME:  this doesn't work out all the allowed
		 * permutations, it just takes the rate from the first
		 * structure on the sink pad and copies it into all the
		 * structures on the source pad
		 */

		othercaps = gst_caps_copy(gst_pad_get_pad_template_caps(GST_BASE_TRANSFORM_SRC_PAD(trans)));
		for(i = 0; i < gst_caps_get_size(caps); i++)
			gst_structure_set_value(gst_caps_get_structure(othercaps, i), "rate", gst_structure_get_value(gst_caps_get_structure(caps, 0), "rate"));
		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		gst_caps_unref(caps);
		return GST_CAPS_NONE;
	}

	if(filter) {
		GstCaps *intersection = gst_caps_intersect(othercaps, filter);
		gst_caps_unref(othercaps);
		othercaps = intersection;
	}

	return othercaps;
}


/*
 * set_caps()
 */


static gboolean set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps)
{
	GSTLALBitVectorGen *element = GSTLAL_BITVECTORGEN(trans);
	GstAudioInfo info;
	gint rate;
	gdouble (*get_input_func)(void **) = NULL;
	void (*set_output_func)(void **, guint32) = NULL;
	guint64 mask;
	gboolean success = TRUE;

	/*
	 * parse the input caps
	 */

	success &= gst_audio_info_from_caps(&info, incaps);
	if(success) {
		const gchar *name = GST_AUDIO_INFO_NAME(&info);
		rate = GST_AUDIO_INFO_RATE(&info);
		switch(GST_AUDIO_INFO_FORMAT(&info)) {
		case GST_AUDIO_FORMAT_S8:
			get_input_func = get_input_int8;
			break;
		case GST_AUDIO_FORMAT_U8:
			get_input_func = get_input_uint8;
			break;
		case GST_AUDIO_FORMAT_S16:
			get_input_func = get_input_int16;
			break;
		case GST_AUDIO_FORMAT_U16:
			get_input_func = get_input_uint16;
			break;
		case GST_AUDIO_FORMAT_S32:
			get_input_func = get_input_int32;
			break;
		case GST_AUDIO_FORMAT_U32:
			get_input_func = get_input_uint32;
			break;
		case GST_AUDIO_FORMAT_F32:
			get_input_func = get_input_float32;
			break;
		case GST_AUDIO_FORMAT_F64:
			get_input_func = get_input_float64;
			break;
		default:
			/*
			 * Handle the complex types which are "special" formats
			 */
			if(!strncmp(name, "Z64", 3))
				get_input_func = get_input_complex64;
			else if(!strncmp(name, "Z128", 4))
				get_input_func = get_input_complex128;
			else
				success = FALSE;
			break;
		}
	}

	/*
	 * parse the output caps
	 */

	success &= gst_audio_info_from_caps(&info, outcaps);
	if(success) {
		switch(GST_AUDIO_INFO_FORMAT(&info)) {
		case GST_AUDIO_FORMAT_U8:
			set_output_func = set_output_uint8;
			mask = 0xff;
			break;

		case GST_AUDIO_FORMAT_U16:
			set_output_func = set_output_uint16;
			mask = 0xffff;
			break;

		case GST_AUDIO_FORMAT_U32:
			set_output_func = set_output_uint32;
			mask = 0xffffffff;
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
		gint oldrate = element->rate;
		element->get_input_func = get_input_func;
		element->set_output_func = set_output_func;
		element->rate = rate;
		element->mask = mask;
		if(element->rate != oldrate)
			g_signal_emit(G_OBJECT(element), signals[SIGNAL_RATE_CHANGED], 0, element->rate, NULL);
	} else
		GST_ERROR_OBJECT(element, "unable to parse and/or accept input caps %" GST_PTR_FORMAT ", output caps %" GST_PTR_FORMAT, incaps, outcaps);

	return success;
}


/*
 * start()
 */


static gboolean start(GstBaseTransform *trans)
{	
	GSTLALBitVectorGen *element = GSTLAL_BITVECTORGEN(trans);
	element->last_state = -1;	/* force start/stop signal on first output */
	return TRUE;
}


/*
 * transform()
 */


static GstFlowReturn transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf)
{
	GSTLALBitVectorGen *element = GSTLAL_BITVECTORGEN(trans);
	guint64 bit_vector = element->bit_vector & element->mask;
	GstMapInfo outmap;
	GstFlowReturn result = GST_FLOW_OK;

	g_assert(element->get_input_func != NULL);
	g_assert(element->set_output_func != NULL);

	gst_buffer_map(outbuf, &outmap, GST_MAP_WRITE);

	/* FIXME:  implement attack and hold */
	/* FIXME:  add support for disconts in nongap-is-control mode */

	if(GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP) || element->nongap_is_control) {
		gboolean state = (element->nongap_is_control && !GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP)) ^ element->invert_control;

		if(state) {
			guint8 *out = outmap.data;
			guint8 *end = outmap.data + outmap.size;
			while(out < end)
				element->set_output_func((void **) &out, bit_vector);
		} else
			memset(outmap.data, 0, outmap.size);

		if(!element->nongap_is_control && GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP))
			GST_BUFFER_FLAG_SET(outbuf, GST_BUFFER_FLAG_GAP);

		if(element->emit_signals && (state != element->last_state))
			g_signal_emit(G_OBJECT(element), signals[state ? SIGNAL_START : SIGNAL_STOP], 0, GST_BUFFER_TIMESTAMP(inbuf), NULL);
		element->last_state = state;
	} else {
		GstMapInfo inmap;
		gst_buffer_map(inbuf, &inmap, GST_MAP_READ);

		guint8 *in = inmap.data;
		guint8 *out = outmap.data;
		guint8 *end = inmap.data + inmap.size;

		while(in < end) {
			guint8 *in_saved = in;	/* for computing timestamps */
			gboolean state = (element->get_input_func((void **) &in) >= element->threshold) ^ element->invert_control;
			element->set_output_func((void **) &out, state ? bit_vector : 0);
			if(element->emit_signals && state != element->last_state) {
				/* need to use in_saved because in has been
				 * incremented */
				GstClockTime timestamp = GST_BUFFER_TIMESTAMP(inbuf) + gst_util_uint64_scale_int_round(in_saved - inmap.data, GST_BUFFER_DURATION(inbuf), inmap.size);
				g_signal_emit(G_OBJECT(element), signals[state ? SIGNAL_START : SIGNAL_STOP], 0, timestamp, NULL);
			}
			element->last_state = state;
		}

		gst_buffer_unmap(inbuf, &inmap);
	}

	gst_buffer_unmap(outbuf, &outmap);

	/*
	 * done
	 */

	return result;
}


/*
 * ============================================================================
 *
 *                          GObject Method Overrides
 *
 * ============================================================================
 */


/*
 * properties
 */


enum property {
	ARG_EMIT_SIGNALS = 1,
	ARG_THRESHOLD,
	ARG_INVERT,
	ARG_NONGAP_IS_CONTROL,
	ARG_BIT_VECTOR
};


static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec)
{
	GSTLALBitVectorGen *element = GSTLAL_BITVECTORGEN(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_EMIT_SIGNALS:
		element->emit_signals = g_value_get_boolean(value);
		break;

	case ARG_THRESHOLD:
		element->threshold = g_value_get_double(value);
		break;

	case ARG_INVERT:
		element->invert_control = g_value_get_boolean(value);
		break;

	case ARG_NONGAP_IS_CONTROL:
		element->nongap_is_control = g_value_get_boolean(value);
		break;

	case ARG_BIT_VECTOR:
		element->bit_vector = g_value_get_uint(value);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *pspec)
{
	GSTLALBitVectorGen *element = GSTLAL_BITVECTORGEN(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_EMIT_SIGNALS:
		g_value_set_boolean(value, element->emit_signals);
		break;

	case ARG_THRESHOLD:
		g_value_set_double(value, element->threshold);
		break;

	case ARG_INVERT:
		g_value_set_boolean(value, element->invert_control);
		break;

	case ARG_NONGAP_IS_CONTROL:
		g_value_set_boolean(value, element->nongap_is_control);
		break;

	case ARG_BIT_VECTOR:
		g_value_set_uint(value, element->bit_vector);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * class_init()
 */


static void gstlal_bitvectorgen_class_init(GSTLALBitVectorGenClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(klass);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);

	transform_class->get_unit_size = GST_DEBUG_FUNCPTR(get_unit_size);
	transform_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
	transform_class->start = GST_DEBUG_FUNCPTR(start);
	transform_class->transform = GST_DEBUG_FUNCPTR(transform);
	transform_class->transform_caps = GST_DEBUG_FUNCPTR(transform_caps);

	klass->rate_changed = GST_DEBUG_FUNCPTR(rate_changed_handler);
	klass->start = GST_DEBUG_FUNCPTR(start_handler);
	klass->stop = GST_DEBUG_FUNCPTR(stop_handler);

	gst_element_class_set_details_simple(
		element_class,
		"Bit Vector Generator",
		"Filter",
		"Generate a bit vector stream based on the value of a control input",
		"Kipp Cannon <kipp.cannon@ligo.org>, Chad Hanna <channa@ligo.caltech.edu>"
	);

	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			GST_BASE_TRANSFORM_SINK_NAME,
			GST_PAD_SINK,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
				"audio/x-raw, " \
				"rate = " GST_AUDIO_RATE_RANGE ", " \
				"channels = 1, " \
				"format = (string) {" GST_AUDIO_NE(U8) ", " GST_AUDIO_NE(U16) ", " GST_AUDIO_NE(U32) ", " GST_AUDIO_NE(S8) ", " GST_AUDIO_NE(S16) ", " GST_AUDIO_NE(S32) ", " GST_AUDIO_NE(F32) ", " GST_AUDIO_NE(F64) ", " GST_AUDIO_NE(Z64) ", " GST_AUDIO_NE(Z128) "}, " \
				"layout = (string) interleaved"
			)
		)
	);
	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			GST_BASE_TRANSFORM_SRC_NAME,
			GST_PAD_SRC,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
				"audio/x-raw, " \
				"rate = " GST_AUDIO_RATE_RANGE ", " \
				"channels = 1, " \
				"format = (string) {" GST_AUDIO_NE(U8) ", " GST_AUDIO_NE(U16) ", " GST_AUDIO_NE(U32) "}, " \
				"layout = (string) interleaved"
			)
		)
	);

	g_object_class_install_property(
		gobject_class,
		ARG_EMIT_SIGNALS,
		g_param_spec_boolean(
			"emit-signals",
			"Emit signals",
			"Emit start and stop signals (rate-changed is always emited).  The start and stop signals are emited for on-to-off and off-to-on transitions in the output stream respectively.",
			DEFAULT_EMIT_SIGNALS,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_THRESHOLD,
		g_param_spec_double(
			"threshold",
			"Threshold",
			"Output will be \"on\" when magnitude of control input is >= this value.  See also invert-control.",
			0, G_MAXDOUBLE, DEFAULT_THRESHOLD,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_INVERT,
		g_param_spec_boolean(
			"invert-control",
			"Invert",
			"Logically invert the control input.  If false (default) then the output is \"off\" if and only if the control is < threshold, \"on\" if >= threshold;  if true then the output is \"off\" if and only if the control is >= threshold, \"on\" if < threshold.",
			DEFAULT_INVERT,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_NONGAP_IS_CONTROL,
		g_param_spec_boolean(
			"nongap-is-control",
			"Use gap flag for control",
			"Instead of applying a threshold to the input data, all non-gap input buffers are \"on\" and gap buffers and missing buffers are \"off\".  The attack, hold, and invert-control properties still apply.",
			DEFAULT_NONGAP_IS_CONTROL,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_BIT_VECTOR,
		g_param_spec_uint(
			"bit-vector",
			"Bit Vector",
			"Value to generate when output is \"on\" (output is 0 otherwise).  Only as many low-order bits as are needed by the output word size will be used.",
			0, G_MAXUINT32, DEFAULT_BIT_VECTOR,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);

	signals[SIGNAL_RATE_CHANGED] = g_signal_new(
		"rate-changed",
		G_TYPE_FROM_CLASS(klass),
		G_SIGNAL_RUN_FIRST,
		G_STRUCT_OFFSET(
			GSTLALBitVectorGenClass,
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
			GSTLALBitVectorGenClass,
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
			GSTLALBitVectorGenClass,
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


static void gstlal_bitvectorgen_init(GSTLALBitVectorGen *element)
{
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(element), TRUE);

	element->rate = -1;	/* force rate-changed signal on first caps */
}

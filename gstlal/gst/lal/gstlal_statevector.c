/*
 * Copyright (C) 2011--2012,2014,2015 Kipp Cannon <kipp.cannon@ligo.org>
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
 * SECTION:gstlal_statevector
 * @short_description:  Converts a state vector stream into booleans, for example to drive a lal_gate element.
 *
 * Each sample of the input stream is interpreted as a bit vector, and
 * mapped one-to-one to boolean-valued output samples.  Each bit of the
 * input vectors can be required to be on, required to be off, or ignored.
 * The bits that must be on are set with the #required-on property;  the
 * bits that must be off are set with the #required-off property.  For each
 * input sample that satisfies the on/off requirements the output is a
 * non-zero sample, all other output samples are 0.  Note that if the
 * bitwise intersection of the #required-on and #required-off properties is
 * non-zero it will be impossible for the input stream to satisfy the
 * conditions and the output will be identically 0.
 *
 * Typically this element is used to transform a bit vector-valued stream
 * into a boolean stream suitable for controling a gate element.
 *
 * Reviewed:  f989b34f43aec056f021f10e5e01866846a3c58d 2014-08-10 K.
 * Cannon, J.  Creighton, B. Sathyaprakash.
 *
 * Completed Actions:
 * - added warning messages if required-on/required-off have too many bits for width of input stream
 * - generalized transform_caps() so that sink-->src conversions are complete
 * - added notifications for sample count properties
 * - wrote unit test
 * - why the mask?  remove?  maybe safer to remove.  removed
 *
 */


/*
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


/*
 *  stuff from gobject/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <gst/audio/audio.h>


/*
 * our own stuff
 */


#include <gstlal/gstlal.h>
#include <gstlal/gstlal_debug.h>
#include <gstlal_statevector.h>


#define GST_CAT_DEFAULT gstlal_statevector_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


/*
 * ============================================================================
 *
 *                                 Parameters
 *
 * ============================================================================
 */


#define DEFAULT_REQUIRED_ON 0
#define DEFAULT_REQUIRED_OFF 0


/*
 * ============================================================================
 *
 *                            Input Type Handling
 *
 * ============================================================================
 */


static guint get_input_uint8(guint8 *in)
{
	return *(*(guint8 **) in)++;
}


static guint get_input_uint16(guint8 *in)
{
	return *(*(guint16 **) in)++;
}


static guint get_input_uint32(guint8 *in)
{
	return *(*(guint32 **) in)++;
}


/*
 * ============================================================================
 *
 *                           GStreamer Boiler Plate
 *
 * ============================================================================
 */


static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SINK_NAME,
	GST_PAD_SINK,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
        "audio/x-raw, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) 1, " \
		"depth = (int) 32, " \
		"signed = {true, false}; " \
		"audio/x-raw-int, " \
		"depth = (int) 16, " \
		"signed = {true, false}; " \
		"rate = (int) [1, MAX], " \
		"channels = (int) 1, " \
        "format = (string) {" GST_AUDIO_NE(F32) ", " GST_AUDIO_NE(F64) ", " GST_AUDIO_NE(S8) ", " GST_AUDIO_NE(S16) ", "  GST_AUDIO_NE(S32) ", " GST_AUDIO_NE(U8) ", " GST_AUDIO_NE(U16) ", "  GST_AUDIO_NE(U32) "}, " \
		"layout = (string) interleaved")
);


static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SRC_NAME,
	GST_PAD_SRC,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		"audio/x-raw, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) 1, " \
		"depth = (int) 1, " \
         "format = (string) {" GST_AUDIO_NE(F32) ", " GST_AUDIO_NE(F64) ", " GST_AUDIO_NE(S8) ", " GST_AUDIO_NE(S16) ", "  GST_AUDIO_NE(S32) ", " GST_AUDIO_NE(U8) ", " GST_AUDIO_NE(U16) ", "  GST_AUDIO_NE(U32) "}, " \
		"signed = false"
	)
);


//static void additional_initializations(GType type)
//{
//	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "lal_statevector", 0, "lal_statevector element");
//}


G_DEFINE_TYPE_WITH_CODE(
	GSTLALStateVector,
	gstlal_statevector,
	GST_TYPE_BASE_TRANSFORM,
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "lal_statevector", 0, "lal_statevector element")
);


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

	if(gst_caps_get_size(caps) > 1)
		GST_WARNING_OBJECT(trans, "not yet smart enough to transform complex formats");

	switch(direction) {
	case GST_PAD_SRC:
		/*
		 * sink pad must have same sample rate as source pad
		 * FIXME:  this doesn't work out all the allowed
		 * permutations, it just takes the rate from the
		 * first structure on the source pad and copies it into all
		 * the structures on the sink pad
		 */

		othercaps = gst_caps_copy(gst_pad_get_pad_template_caps(GST_BASE_TRANSFORM_SINK_PAD(trans)));
		for(i = 0; i < gst_caps_get_size(othercaps); i++)
			gst_structure_set_value(gst_caps_get_structure(othercaps, i), "rate", gst_structure_get_value(gst_caps_get_structure(caps, 0), "rate"));
		break;

	case GST_PAD_SINK:
		/*
		 * source pad's format must be 8-bit boolean
		 */

		othercaps = gst_caps_copy(caps);
		for(i = 0; i < gst_caps_get_size(othercaps); i++)
			gst_structure_set(gst_caps_get_structure(othercaps, i), "width", G_TYPE_INT, 8, "depth", G_TYPE_INT, 1, "signed", G_TYPE_BOOLEAN, FALSE, NULL);
		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		gst_caps_unref(caps);
		return GST_CAPS_NONE;
	}

	return othercaps;
}


/*
 * set_caps()
 */


static gboolean set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps)
{
	GSTLALStateVector *element = GSTLAL_STATEVECTOR(trans);
	GstAudioInfo info;

	/*
	 * parse the caps
	 */

	if(!gst_audio_info_from_caps(&info, incaps))
		return FALSE;

	/*
	 * set the sample value function
	 */

	switch(GST_AUDIO_INFO_WIDTH(&info)) {
	case 8:
		element->get_input = get_input_uint8;
		break;

	case 16:
		element->get_input = get_input_uint16;
		break;

	case 32:
		element->get_input = get_input_uint32;
		break;

	default:
		GST_ERROR_OBJECT(element, "unable to parse and/or accept caps %" GST_PTR_FORMAT, incaps);
		return FALSE;
	}

	return TRUE;
}


/*
 * start()
 */


static gboolean start(GstBaseTransform *trans)
{	
	GSTLALStateVector *filter = GSTLAL_STATEVECTOR(trans);
	filter->on_samples = 0;
	filter->off_samples = 0;
	filter->gap_samples = 0;
	return TRUE;
}


/*
 * transform()
 */


static GstFlowReturn transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf)
{
	GstMapInfo in_info;
	GstMapInfo out_info;
	GSTLALStateVector *element = GSTLAL_STATEVECTOR(trans);
	GstFlowReturn result = GST_FLOW_OK;
	guint64 on_samples = element->on_samples;
	guint64 off_samples = element->off_samples;
	guint64 gap_samples = element->gap_samples;

	g_assert(element->get_input != NULL);

	GST_LOG_OBJECT(element, "processing %s%s buffer %p spanning %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP) ? "gap" : "nongap", GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_DISCONT) ? "+discont" : "", inbuf, GST_BUFFER_BOUNDARIES_ARGS(inbuf));

	if(!GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP)) {
		/*
		 * input is not GAP.
		 */

		gst_buffer_map(inbuf, &in_info, GST_MAP_READ);
		gst_buffer_map(outbuf, &out_info, GST_MAP_WRITE);
		guint8 *out = out_info.data;
		guint8 *end = in_info.data + in_info.size;
		guint required_on = element->required_on;
		guint required_off = element->required_off;

		for(; in_info.data < end; out_info.data++) {
			guint input = element->get_input(in_info.data);
			if(((input & required_on) == required_on) && ((~input & required_off) == required_off)) {
				*out_info.data = 0x80;
				element->on_samples++;
			} else {
				*out = 0x00;
				element->off_samples++;
			}
		}
	} else {
		/*
		 * input is GAP.
		 */

		GST_BUFFER_FLAG_SET(outbuf, GST_BUFFER_FLAG_GAP);
		gst_buffer_map(outbuf, &out_info, GST_MAP_WRITE);
		memset(out_info.data, 0, out_info.size);
		element->gap_samples += GST_BUFFER_OFFSET_END(inbuf) - GST_BUFFER_OFFSET(inbuf);
	}

	/*
	 * notify sample count changes
	 */

	if(element->on_samples != on_samples)
		g_object_notify(G_OBJECT(trans), "on-samples");
	if(element->off_samples != off_samples)
		g_object_notify(G_OBJECT(trans), "off-samples");
	if(element->gap_samples != gap_samples)
		g_object_notify(G_OBJECT(trans), "gap-samples");

	/*
	 * done
	 */

	gst_buffer_unmap(inbuf, &in_info);
	gst_buffer_unmap(outbuf, &out_info);

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
	ARG_REQUIRED_ON = 1,
	ARG_REQUIRED_OFF,
	ARG_ON_SAMPLES,
	ARG_OFF_SAMPLES,
	ARG_GAP_SAMPLES
};


static void set_property(GObject *object, enum property prop_id, const GValue *value, GParamSpec *pspec)
{
	GSTLALStateVector *element = GSTLAL_STATEVECTOR(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_REQUIRED_ON:
		element->required_on = g_value_get_uint(value);
		break;

	case ARG_REQUIRED_OFF:
		element->required_off = g_value_get_uint(value);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, enum property prop_id, GValue *value, GParamSpec *pspec)
{
	GSTLALStateVector *element = GSTLAL_STATEVECTOR(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_REQUIRED_ON:
		g_value_set_uint(value, element->required_on);
		break;

	case ARG_REQUIRED_OFF:
		g_value_set_uint(value, element->required_off);
		break;

	case ARG_ON_SAMPLES:
		g_value_set_uint64(value, element->on_samples);
		break;

	case ARG_OFF_SAMPLES:
		g_value_set_uint64(value, element->off_samples);
		break;

	case ARG_GAP_SAMPLES:
		g_value_set_uint64(value, element->gap_samples);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * class_init()
 */


static void gstlal_statevector_class_init(GSTLALStateVectorClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(klass);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);

	transform_class->get_unit_size = GST_DEBUG_FUNCPTR(get_unit_size);
	transform_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
	transform_class->transform = GST_DEBUG_FUNCPTR(transform);
	transform_class->transform_caps = GST_DEBUG_FUNCPTR(transform_caps);
	transform_class->start = GST_DEBUG_FUNCPTR(start);

	gst_element_class_set_details_simple(element_class, "LIGO State Vector Parser", "Filter/Audio", "Converts a state vector stream into booleans, for example to drive a lal_gate element.", "Kipp Cannon <kipp.cannon@ligo.org>");

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));

	g_object_class_install_property(
		gobject_class,
		ARG_REQUIRED_ON,
		g_param_spec_uint(
			"required-on",
			"On bits",
			"Bit mask setting the bits that must be on in the state vector.  Note:  if the mask is wider than the input stream, the high-order bits should be 0 or the on condition will never be met.",
			0, G_MAXUINT, DEFAULT_REQUIRED_ON,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_REQUIRED_OFF,
		g_param_spec_uint(
			"required-off",
			"Off bits",
			"Bit mask setting the bits that must be off in the state vector.",
			0, G_MAXUINT, DEFAULT_REQUIRED_OFF,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_ON_SAMPLES,
		g_param_spec_uint64(
			"on-samples",
			"On samples",
			"Number of samples seen thus far marked as on",
			0, G_MAXUINT64, 0,
			G_PARAM_READABLE | G_PARAM_STATIC_STRINGS
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_OFF_SAMPLES,
		g_param_spec_uint64(
			"off-samples",
			"Off samples",
			"Number of samples seen thus far marked as off",
			0, G_MAXUINT64, 0,
			G_PARAM_READABLE | G_PARAM_STATIC_STRINGS
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_GAP_SAMPLES,
		g_param_spec_uint64(
			"gap-samples",
			"Gap samples",
			"number of samples seen thus far marked as gap",
			0, G_MAXUINT64, 0,
			G_PARAM_READABLE | G_PARAM_STATIC_STRINGS
		)
	);
}


/*
 * init()
 */


static void gstlal_statevector_init(GSTLALStateVector *filter)
{
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(filter), TRUE);

	filter->get_input = NULL;
}

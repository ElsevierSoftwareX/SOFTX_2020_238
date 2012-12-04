/*
 * Copyright (C) 2012 Kipp Cannon <kipp.cannon@ligo.org>, Chris Pankow <chris.pankow@ligo.org>
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
 *  stuff from gobject/gstreamer
 */


#include <glib.h>
#include <string.h>
#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>


/*
 * our own stuff
 */


#include <gstlal_odc_to_dqv.h>
#include <gstlal/gstlal_debug.h>


#define GST_CAT_DEFAULT gstlal_odc_to_dqv_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


/*
 * ============================================================================
 *
 *                                 Parameters
 *
 * ============================================================================
 */

#define DEFAULT_REQUIRED_ON 0x1
#define DEFAULT_STATUS_OUT 0x7

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
		"audio/x-raw-int, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) 1, " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) 32, " \
		"signed = false" 
	)
);


static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SRC_NAME,
	GST_PAD_SRC,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		"audio/x-raw-int, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) 1, " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) 32, " \
		"signed = false"
	)
);


static void additional_initializations(GType type)
{
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "lal_odc_to_dqv", 0, "lal_odc_to_dqv element");
}


GST_BOILERPLATE_FULL(
	GSTLALODCtoDQV,
	gstlal_odc_to_dqv,
	GstBaseTransform,
	GST_TYPE_BASE_TRANSFORM,
	additional_initializations
);

enum property {
	ARG_REQUIRED_ON = 1,
	ARG_STATUS_OUT,
	ARG_GAP_SAMPLES
};


/*
 * ============================================================================
 *
 *                     GstBaseTransform Method Overrides
 *
 * ============================================================================
 */


/*
 * transform_caps()
 */


static GstCaps *transform_caps(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps)
{
	GstCaps *othercaps = NULL;
	guint i;

	switch(direction) {
	case GST_PAD_SRC:
		/*
		 * sink pad must have same sample rate as source pad
		 * FIXME:  this doesn't work out all the allowed
		 * permutations, it just takes the channel count from the
		 * first structure on the source pad and copies it into all
		 * the structures on the sink pad
		 */

		othercaps = gst_caps_copy(gst_pad_get_pad_template_caps(GST_BASE_TRANSFORM_SINK_PAD(trans)));
		for(i = 0; i < gst_caps_get_size(othercaps); i++)
			gst_structure_set_value(gst_caps_get_structure(othercaps, i), "channels", gst_structure_get_value(gst_caps_get_structure(caps, 0), "channels"));
		break;

	case GST_PAD_SINK:
		/*
		 * source pad must have same sample rate as sink pad
		 * FIXME:  this doesn't work out all the allowed
		 * permutations, it just takes the channel count from the
		 * first structure on the sink pad and copies it into all
		 * the structures on the source pad
		 */

		othercaps = gst_caps_copy(gst_pad_get_pad_template_caps(GST_BASE_TRANSFORM_SRC_PAD(trans)));
		for(i = 0; i < gst_caps_get_size(caps); i++)
			gst_structure_set_value(gst_caps_get_structure(othercaps, i), "channels", gst_structure_get_value(gst_caps_get_structure(caps, 0), "channels"));
		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		gst_caps_unref(caps);
		return GST_CAPS_NONE;
	}

	return othercaps;
}


/*
 * start()
 */

static gboolean start(GstBaseTransform *trans)
{	
	GSTLALODCtoDQV *filter = GSTLAL_ODC_TO_DQV(trans);
	filter->gap_samples = 0;
	return TRUE;
}


/*
 * transform()
 */


static GstFlowReturn transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf)
{
	GSTLALODCtoDQV *element = GSTLAL_ODC_TO_DQV(trans);
	GstFlowReturn result = GST_FLOW_OK;

	GST_INFO_OBJECT(element, "processing %s%s buffer %p spanning %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP) ? "gap" : "nongap", GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_DISCONT) ? "+discont" : "", inbuf, GST_BUFFER_BOUNDARIES_ARGS(inbuf));

	if(!GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP)) {
		/*
		 * input is not GAP.
		 */

		guint *in = GST_BUFFER_DATA(inbuf);
		guint *end = GST_BUFFER_DATA(inbuf) + GST_BUFFER_SIZE(inbuf);
		guint *out = (guint*)GST_BUFFER_DATA(outbuf);

		for(; in < end; in++, out++) {
			*out = *in & element->required_on ? element->status_out : 0x0;
		}
	} else {
		/*
		 * input is GAP.
		 */

		GST_BUFFER_FLAG_SET(outbuf, GST_BUFFER_FLAG_GAP);
		memset(GST_BUFFER_DATA(outbuf), 0, GST_BUFFER_SIZE(outbuf));
		element->gap_samples += GST_BUFFER_OFFSET_END(inbuf) - GST_BUFFER_OFFSET(inbuf);
	}

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
 * set_property()
 */

static void set_property(GObject *object, enum property prop_id, const GValue *value, GParamSpec *pspec)
{
	GSTLALODCtoDQV *element = GSTLAL_ODC_TO_DQV(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
    case ARG_REQUIRED_ON:
        element->required_on = g_value_get_uint(value);
        break;

    case ARG_STATUS_OUT:
        element->status_out = g_value_get_uint(value);
        break;
	case ARG_GAP_SAMPLES:
		element->gap_samples = g_value_get_uint64(value);
		break;

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
	GSTLALODCtoDQV *element = GSTLAL_ODC_TO_DQV(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
   case ARG_REQUIRED_ON:
        g_value_set_uint(value, element->required_on);
        break;
    case ARG_STATUS_OUT:
        g_value_set_uint(value, element->status_out);
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
 * base_init()
 */


static void gstlal_odc_to_dqv_base_init(gpointer gclass)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(gclass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(gclass);

	gst_element_class_set_details_simple(element_class, "ODC to DQ", "Filter/Audio", "Processes the Online Detector Characterization channel with the supplied bitmask to produce a data quality output status integer.", "Chris Pankow <chris.pankow@ligo.org>");

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));

	transform_class->transform = GST_DEBUG_FUNCPTR(transform);
	transform_class->transform_caps = GST_DEBUG_FUNCPTR(transform_caps);
	transform_class->start = GST_DEBUG_FUNCPTR(start);
}


/*
 * class_init()
 */


static void gstlal_odc_to_dqv_class_init(GSTLALODCtoDQVClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);

    g_object_class_install_property(
        gobject_class,
        ARG_REQUIRED_ON,
        g_param_spec_uint(
            "required-on",
            "On bits",
            "Bit mask setting the bits that must be on in the state vector.  Only as many of the low bits as the input stream is wide will be considered.",
            0, G_MAXUINT, DEFAULT_REQUIRED_ON,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
        )
    );
    g_object_class_install_property(
        gobject_class,
        ARG_STATUS_OUT,
        g_param_spec_uint(
            "status-out",
            "Output bits",
            "Value to output if required-on mask is true.",
            0, G_MAXUINT, DEFAULT_STATUS_OUT,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
        )
    );
    g_object_class_install_property(
        gobject_class,
        ARG_GAP_SAMPLES,
        g_param_spec_uint64(
            "gap-samples",
            "Gap samples",
            "number of samples seen thus far marked as gap",
            0, G_MAXUINT, 0,
            G_PARAM_READABLE | G_PARAM_STATIC_STRINGS
        )
    );

}


/*
 * init()
 */


static void gstlal_odc_to_dqv_init(GSTLALODCtoDQV *filter, GSTLALODCtoDQVClass *klass)
{
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(filter), TRUE);
}

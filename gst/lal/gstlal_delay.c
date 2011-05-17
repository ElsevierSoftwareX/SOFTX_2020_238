/*
 * Copyright (C) 2009 Stephen Privitera <sprivite@ligo.caltech.edu>,
 * Kipp Cannon <kipp.cannon@ligo.org>, Chad Hanna <chad.hanna@ligo.org>
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
#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <gstlal.h>
#include <gstlal_delay.h>

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
		"audio/x-raw-float, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) [1, MAX], "\
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) {32,64}; " \
		"audio/x-raw-int, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) [1, MAX], " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) {8,16,32,64}, " \
		"signed = (boolean) {true,false}; " \
		"audio/x-raw-complex, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) [1, MAX], " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) {64,128}" \
	)
);


static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SRC_NAME,
	GST_PAD_SRC,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		"audio/x-raw-float, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) [1, MAX], " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) {32,64};"
		"audio/x-raw-int, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) [1, MAX], " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) {8,16,32,64}, " \
		"signed = (boolean) {true,false}; " \
		"audio/x-raw-complex, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) [1, MAX], " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) {64,128}"
	)
);


GST_BOILERPLATE(
	GSTLALDelay,
	gstlal_delay,
	GstBaseTransform,
	GST_TYPE_BASE_TRANSFORM
);


/*
 * ============================================================================
 *
 *                     GstBaseTransform Method Overrides
 *
 * ============================================================================
 */


enum property {
	ARG_DELAY = 1
};

#define DEFAULT_DELAY 0

/*
 * get_unit_size() stores the size (in bytes) of a single sample
 * from a single channel in the buffer.
 * The "width" of a buffer is equal to the total number of channels
 * times the number of bits per channel.
 */
static gboolean get_unit_size(GstBaseTransform *trans, GstCaps *caps, guint *size)
{
	GstStructure *str;
	gint width;
	gint channels;

	str = gst_caps_get_structure(caps, 0);
	if(!gst_structure_get_int(str, "channels", &channels)) {
		GST_DEBUG_OBJECT(trans, "unable to parse channels from %" GST_PTR_FORMAT, caps);
		return FALSE;
	}
	if(!gst_structure_get_int(str, "width", &width)) {
		GST_DEBUG_OBJECT(trans, "unable to parse width from %" GST_PTR_FORMAT, caps);
		return FALSE;
	}

	*size = width / 8 * channels;
	return TRUE;
}


/*
 * When the caps on an element's pads a finally set, this function is called.
 * We use this opportunity to record the chosen sampling rate and the unit
 * size.
 */
static gboolean set_caps(GstBaseTransform *trans,
			 GstCaps *incaps,
			 GstCaps *outcaps)
{
	GSTLALDelay *element = GSTLAL_DELAY(trans);

	/* sampling rate of this channel */
	gst_structure_get_int(gst_caps_get_structure(incaps, 0), "rate", &element->rate);

	/* size of unit sample */
	get_unit_size(trans,incaps,&element->unit_size);

	return TRUE;
}


/*
 * The transform function actually does the heavy lifting on buffers.
 * Given an input buffer and an output buffer (the latter of which is
 * set in prepare_output_buffer), determine what data actually gets put
 * into the output buffer.
 */
static GstFlowReturn transform_ip( GstBaseTransform *trans, GstBuffer *inbuf)
{
	GSTLALDelay *element = GSTLAL_DELAY(trans);
	GstFlowReturn result;
	guint delaysize = (guint) element->delay*element->unit_size;
	
	if ( GST_BUFFER_SIZE(inbuf) <= delaysize )
	/* drop entire buffer */
	{
		element->delay -= GST_BUFFER_OFFSET_END(inbuf) - GST_BUFFER_OFFSET(inbuf);
		result = GST_BASE_TRANSFORM_FLOW_DROPPED;
	}
	else if ( 0 < element->delay )
	/* drop part of buffer, pass the rest */
	{
		/* never come back */
		element->delay = 0;

		result = GST_FLOW_OK;
	}
	else
	/* pass entire buffer */
	{
		result = GST_FLOW_OK;
	}

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
	GSTLALDelay *element = GSTLAL_DELAY(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id)
	{
	case ARG_DELAY:
		element->delay = g_value_get_uint64(value);
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
	GSTLALDelay *element = GSTLAL_DELAY(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id)
	{
	case ARG_DELAY:
	  g_value_set_uint64(value, element->delay);
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
static void
gstlal_delay_base_init(gpointer gclass)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(gclass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(gclass);

	gst_element_class_set_details_simple(
		element_class,
		"Drops beginning of a stream",
		"Filter/Audio",
		"Drops beginning of a stream",
		"Stephen Privitera <sprivite@ligo.caltech.edu>"
	);

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));

	transform_class->transform_ip = GST_DEBUG_FUNCPTR(transform_ip);
	transform_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
}


/*
 * class_init()
 */
static void gstlal_delay_class_init(GSTLALDelayClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);

	g_object_class_install_property(
		gobject_class,
		ARG_DELAY,
		g_param_spec_uint64(
			"delay",
			"Time delay",
			"Amount of data (in samples) to ignore at front of stream.",
			0, G_MAXUINT64, DEFAULT_DELAY,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);

}


/*
 * init() -- equivalent to python's __init__()
 */
static void gstlal_delay_init(GSTLALDelay *filter, GSTLALDelayClass *kclass)
{
	filter->delay = DEFAULT_DELAY;
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(filter), TRUE);
	gst_base_transform_set_in_place(GST_BASE_TRANSFORM(filter), TRUE);
	gst_base_transform_set_passthrough(GST_BASE_TRANSFORM(filter), TRUE);
}

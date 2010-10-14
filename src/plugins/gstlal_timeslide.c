/*
 * Copyright (C) Leo Singer <leo.singer@ligo.org>
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
 *  stuff from the C library
 */


#include <string.h>


/*
 *  stuff from gobject/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <gstlal_timeslide.h>


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
	GST_STATIC_CAPS_ANY
);


static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SRC_NAME,
	GST_PAD_SRC,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS_ANY
);


GST_BOILERPLATE(
	GSTLALTimeSlide,
	gstlal_timeslide,
	GstBaseTransform,
	GST_TYPE_BASE_TRANSFORM
);


/*
 * ============================================================================
 *
 *                                 Properties
 *
 * ============================================================================
 */


enum property {
	ARG_SLIDE = 1
};


static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec)
{
	GSTLALTimeSlide *element = GSTLAL_TIMESLIDE(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_SLIDE:
		element->slide = g_value_get_int64(value);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *pspec)
{
	GSTLALTimeSlide *element = GSTLAL_TIMESLIDE(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_SLIDE:
		g_value_set_int64(value, element->slide);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * ============================================================================
 *
 *                     GstBaseTransform Method Overrides
 *
 * ============================================================================
 */


/*
 * prepare_output_buffer()
 */


static GstFlowReturn prepare_output_buffer(GstBaseTransform *trans, GstBuffer *input, gint size, GstCaps *caps, GstBuffer **buf)
{
	GSTLALTimeSlide *element = GSTLAL_TIMESLIDE(trans);

	/*
	 * create sub-buffer from input with writeable metadata
	 */

	gst_buffer_ref(input);
	*buf = gst_buffer_make_metadata_writable(input);
	if(!*buf) {
		GST_DEBUG_OBJECT(trans, "failure creating sub-buffer from input");
		return GST_FLOW_ERROR;
	}

	/*
	 * modify timestamp
	 */

	GST_BUFFER_TIMESTAMP(*buf) += element->slide;
	GST_ERROR_OBJECT(trans, "%lld", GST_BUFFER_TIMESTAMP(*buf));

	/*
	 * done
	 */

	return GST_FLOW_OK;
}


/*
 * ============================================================================
 *
 *                          GObject Method Overrides
 *
 * ============================================================================
 */


/*
 * base_init()
 */


static void gstlal_timeslide_base_init(gpointer gclass)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(gclass);

	gst_element_class_set_details_simple(
		element_class,
		"Time Slide",
		"Filter/Audio",
		"Time slide any data stream by adding a programmable offset to all timestamps.",
		"Leo Singer <leo.singer@ligo.org>"
	);

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));
}


/*
 * class_init()
 */


static void gstlal_timeslide_class_init(GSTLALTimeSlideClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(klass);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);

	g_object_class_install_property(
		gobject_class,
		ARG_SLIDE,
		g_param_spec_int64(
			"slide",
			"Time slide offset",
			"Offset to add to all timestamps (nanoseconds)",
			G_MININT64, G_MAXINT64, 0,
			G_PARAM_READWRITE | G_PARAM_CONSTRUCT | G_PARAM_STATIC_STRINGS
		)
	);

	transform_class->prepare_output_buffer = GST_DEBUG_FUNCPTR(prepare_output_buffer);
}


/*
 * init()
 */


static void gstlal_timeslide_init(GSTLALTimeSlide *element, GSTLALTimeSlideClass *kclass)
{
	gst_base_transform_set_in_place(GST_BASE_TRANSFORM(element), TRUE);
	gst_base_transform_set_qos_enabled(GST_BASE_TRANSFORM(element), TRUE);
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(element), TRUE);
}

/*
 * Copyright (C) 2017 Patrick Godwin  <patrick.godwin@ligo.org>
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
 *				  Preamble
 *
 * ============================================================================
 */


/*
 * stuff from C
 */

#define _XOPEN_SOURCE
#include <time.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/*
 *  stuff from gobject/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/audio/audio.h>
#include <gst/base/gstbasetransform.h>


/*
 * stuff from LAL/gstlal
 */


#include <lal/Date.h>
#include <lal/LALAtomicDatatypes.h>
#include <gstlal/gstlal_audio_info.h>
#include <gstlal_latency.h>


/*
 * ============================================================================
 *
 *			   GStreamer Boiler Plate
 *
 * ============================================================================
 */


#define GST_CAT_DEFAULT gstlal_latency_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


G_DEFINE_TYPE_WITH_CODE(
	GSTLALLatency,
	gstlal_latency,
	GST_TYPE_BASE_TRANSFORM,
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "lal_latency", 0, "lal_latency element")
);


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



/*
 * ============================================================================
 *
 *		     GstBaseTransform Method Overrides
 *
 * ============================================================================
 */


/*
 * transform_ip()
 */


static GstFlowReturn transform_ip(GstBaseTransform *trans, GstBuffer *buf)
{
	GSTLALLatency *element = GSTLAL_LATENCY(trans);
	gboolean silent = element->silent;

	GstDateTime *current_gst_time = gst_date_time_new_now_utc();
	gchar *current_utc_time = gst_date_time_to_iso8601_string(current_gst_time);

	// parse DateTime to gps time
	struct tm tm;
	strptime(current_utc_time, "%Y-%m-%dT%H:%M:%SZ", &tm);

	gdouble current_s = (double) XLALUTCToGPS(&tm);
	gdouble current_us = (double) gst_date_time_get_microsecond(current_gst_time) * pow(10,-6);

	gdouble current_time = current_s + current_us;
	gdouble buffer_time = (double) GST_TIME_AS_SECONDS(GST_BUFFER_PTS(buf));
	
	gdouble latency = current_time - buffer_time;
	 
	if (!silent) {
		FILE *out_file;
		out_file = fopen("latency_output.txt", "a");

		fprintf(out_file, "current time = %9.3f, buffer time = %9d, latency = %6.3f, %s\n",
			current_time, (int) buffer_time, latency, GST_OBJECT_NAME(element));

		fclose(out_file);
	}
	
	gst_date_time_unref(current_gst_time);
	g_free(current_utc_time);

	return GST_FLOW_OK;
}


/*
 * ============================================================================
 *
 *			  GObject Method Overrides
 *
 * ============================================================================
 */


enum property {
	ARG_SILENT = 1
};

#define DEFAULT_SILENT FALSE


/*
 * set_property()
 */


static void set_property(GObject *object, enum property prop_id, const GValue *value, GParamSpec *pspec)
{
	GSTLALLatency *element = GSTLAL_LATENCY(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {

	case ARG_SILENT:
		element->silent = g_value_get_boolean(value);
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
	GSTLALLatency *element = GSTLAL_LATENCY(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {

	case ARG_SILENT:
		g_value_set_boolean(value, element->silent);
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


static void gstlal_latency_class_init(GSTLALLatencyClass *klass)
{
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

	gst_element_class_set_details_simple(
		element_class,
		"Latency",
		"Testing",
		"Outputs the current GPS time at time of data flow",
		"Patrick Godwin <patrick.godwin@ligo.org>"
	);
	
	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);

	g_object_class_install_property(
		gobject_class,
		ARG_SILENT,
		g_param_spec_boolean(
			"silent",
			"Silent",
			"Do not print output to stdout.",
			DEFAULT_SILENT,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));

	transform_class->transform_ip = GST_DEBUG_FUNCPTR(transform_ip);
}


/*
 * init()
 */


static void gstlal_latency_init(GSTLALLatency *element)
{
	gst_base_transform_set_passthrough(GST_BASE_TRANSFORM(element), TRUE);
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(element), TRUE);

	element->silent = DEFAULT_SILENT;
}

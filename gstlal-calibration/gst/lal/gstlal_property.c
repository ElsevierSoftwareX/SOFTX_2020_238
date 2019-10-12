/*
 * Copyright (C) 2018  Aaron Viets <aaron.viets@ligo.org>
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
 *				  Preamble
 *
 * ========================================================================
 */


/*
 * stuff from the C library
 */


#include <complex.h>
#include <math.h>
#include <string.h>


/*
 * stuff from glib/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasesink.h>
#include <gst/audio/audio.h>
#include <gst/audio/audio-format.h>


/*
 * our own stuff
 */


#include <gstlal/gstlal.h>
#include <gstlal/gstlal_audio_info.h>
#include <gstlal/gstlal_debug.h>
#include <gstlal_property.h>


/*
 * ============================================================================
 *
 *			   GStreamer Boilerplate
 *
 * ============================================================================
 */


#define GST_CAT_DEFAULT gstlal_property_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


G_DEFINE_TYPE_WITH_CODE(
	GSTLALProperty,
	gstlal_property,
	GST_TYPE_BASE_SINK,
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "lal_property", 0, "lal_property element")
);


enum property {
	ARG_UPDATE_SAMPLES = 1,
	ARG_AVERAGE_SAMPLES,
	ARG_SHIFT_SAMPLES,
	ARG_UPDATE_WHEN_CHANGE,
	ARG_CURRENT_AVERAGE,
	ARG_TIMESTAMPED_AVERAGE,
	ARG_FAKE
};


static GParamSpec *properties[ARG_FAKE];


/*
 * ============================================================================
 *
 *				  Utilities
 *
 * ============================================================================
 */


static void rebuild_workspace_and_reset(GObject *object) {
	return;
}


#define DEFINE_AVERAGE_INPUT_DATA(DTYPE) \
static void average_input_data_ ## DTYPE(GSTLALProperty *element, DTYPE *src, guint64 src_size) { \
 \
	gint64 i; \
	if(element->update_when_change) { \
		for(i = 0; i < (gint64) src_size; i++) { \
			/* Check if the input value has changed */ \
			if((double) src[i] != element->current_average) { \
				element->current_average = (double) src[i]; \
				GST_LOG_OBJECT(element, "Just computed new property"); \
				/* When exactly did this change occur? */ \
				element->timestamp += gst_util_uint64_scale_int_round((guint64) i, GST_SECOND, element->rate); \
				/* Let other elements know when the change occurred */ \
				g_object_notify_by_pspec(G_OBJECT(element), properties[ARG_TIMESTAMPED_AVERAGE]); \
				/* Let other elements know about the update */ \
				g_object_notify_by_pspec(G_OBJECT(element), properties[ARG_CURRENT_AVERAGE]); \
			} \
		} \
	} else { \
		gint64 start_sample, initial_samples, samples_to_add; \
		/* Find the location in src of the first sample that will go into the average */ \
		if(element->num_in_avg) \
			start_sample = 0; \
		else \
			start_sample = (gint64) (element->update_samples - (gst_util_uint64_scale_int_round(element->timestamp, element->rate, GST_SECOND) + element->average_samples - element->shift_samples) % element->update_samples) % element->update_samples; \
 \
		/* How many samples from this buffer will we need to add into this average? */ \
		samples_to_add = element->average_samples - element->num_in_avg < (gint64) src_size - start_sample ? element->average_samples - element->num_in_avg : (gint64) src_size - start_sample; \
		while(samples_to_add > 0) { \
			initial_samples = element->num_in_avg; \
			for(i = start_sample; i < start_sample + samples_to_add; i++) { \
				element->current_average += (double) src[i]; \
			} \
			element->num_in_avg += samples_to_add; \
			if(element->num_in_avg >= element->average_samples) { \
 \
				/* Number of samples in average should not become greater than specified by the user */ \
				g_assert_cmpint(element->num_in_avg, ==, element->average_samples); \
 \
				/* We still need to divide by n to get the average */ \
				element->current_average /= element->num_in_avg; \
 \
				/* When exactly did this change occur? */ \
				element->timestamp += gst_util_uint64_scale_int_round((guint64) samples_to_add, GST_SECOND, element->rate); \
 \
				GST_LOG_OBJECT(element, "Just computed new property"); \
 \
				/* Let other elements know when the change occurred */ \
				g_object_notify_by_pspec(G_OBJECT(element), properties[ARG_TIMESTAMPED_AVERAGE]); \
				/* Let other elements know about the update */ \
				g_object_notify_by_pspec(G_OBJECT(element), properties[ARG_CURRENT_AVERAGE]); \
 \
				element->num_in_avg = 0; \
				element->current_average = 0.0; \
			} \
			start_sample += element->update_samples - initial_samples; \
			samples_to_add = element->average_samples - element->num_in_avg < (gint64) src_size - start_sample ? element->average_samples - element->num_in_avg : (gint64) src_size - start_sample; \
		} \
	} \
 \
	return; \
}


DEFINE_AVERAGE_INPUT_DATA(gint8);
DEFINE_AVERAGE_INPUT_DATA(gint16);
DEFINE_AVERAGE_INPUT_DATA(gint32);
DEFINE_AVERAGE_INPUT_DATA(guint8);
DEFINE_AVERAGE_INPUT_DATA(guint16);
DEFINE_AVERAGE_INPUT_DATA(guint32); 
DEFINE_AVERAGE_INPUT_DATA(float);
DEFINE_AVERAGE_INPUT_DATA(double);


/*
 * ============================================================================
 *
 *			    GstBaseSink Overrides
 *
 * ============================================================================
 */


/*
 * get_unit_size()
 */


static gboolean get_unit_size(GstBaseSink *sink, GstCaps *caps, gsize *size) {

	GstAudioInfo info;
	gboolean success = gstlal_audio_info_from_caps(&info, caps);
	if(success)
		*size = GST_AUDIO_INFO_BPF(&info);
	else
		GST_WARNING_OBJECT(sink, "unable to parse caps %" GST_PTR_FORMAT, caps);
	return success;
}


/*
 * set_caps()
 */


static gboolean set_caps(GstBaseSink *sink, GstCaps *caps) {

	GSTLALProperty *element = GSTLAL_PROPERTY(sink);

	gboolean success = TRUE;

	gsize unit_size;

	/* Parse the caps to find the format, sample rate, and number of channels */
	GstStructure *str = gst_caps_get_structure(caps, 0);
	const gchar *name = gst_structure_get_string(str, "format");
	success &= (name != NULL);
	success &= gst_structure_get_int(str, "rate", &element->rate);

	/* Find unit size */
	success &= get_unit_size(sink, caps, &unit_size);
	element->unit_size = unit_size;

	/* Record the data type */
	if(success) {
		if(strchr(name, 'S'))
			element->data_type = GSTLAL_PROPERTY_SIGNED;
		else if(strchr(name, 'U'))
			element->data_type = GSTLAL_PROPERTY_UNSIGNED;
		else if(strchr(name, 'F'))
			element->data_type = GSTLAL_PROPERTY_FLOAT;
		else
			g_assert_not_reached();
	}

	return success;
}


/*
 * render()
 */


static GstFlowReturn render(GstBaseSink *sink, GstBuffer *buffer) {

	GSTLALProperty *element = GSTLAL_PROPERTY(sink);
	GstMapInfo mapinfo;
	GstFlowReturn result = GST_FLOW_OK;

	/*
	 * check for discontinuity
	 */

	if(G_UNLIKELY(GST_BUFFER_IS_DISCONT(buffer) || GST_BUFFER_OFFSET(buffer) != element->next_in_offset || !GST_CLOCK_TIME_IS_VALID(element->t0))) {
		element->t0 = GST_BUFFER_PTS(buffer);
		element->offset0 = GST_BUFFER_OFFSET(buffer);
		if(!element->update_when_change)
			element->current_average = 0.0;
	}
	element->timestamp = GST_BUFFER_PTS(buffer);
	element->next_in_offset = GST_BUFFER_OFFSET_END(buffer);
	GST_DEBUG_OBJECT(element, "have buffer spanning %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(buffer));

	/* Check if the data on this buffer is usable and if we plan to use it */
	gint64 next_start_sample = (element->update_samples - (gst_util_uint64_scale_int_round(GST_BUFFER_PTS(buffer), element->rate, GST_SECOND) + element->average_samples - element->shift_samples) % element->update_samples) % element->update_samples;
	if(!GST_BUFFER_FLAG_IS_SET(buffer, GST_BUFFER_FLAG_GAP) && mapinfo.size && (element->num_in_avg || next_start_sample < (gint64) gst_util_uint64_scale_int_round(GST_BUFFER_DURATION(buffer), element->rate, GST_SECOND) || element->update_when_change)) {
		/* Get the data from the buffer */
		gst_buffer_map(buffer, &mapinfo, GST_MAP_READ);

		switch(element->data_type) {
		case GSTLAL_PROPERTY_SIGNED:
			switch(element->unit_size) {
			case 1:
				average_input_data_gint8(element, (gint8 *) mapinfo.data, mapinfo.size / element->unit_size);
				break;
			case 2:
				average_input_data_gint16(element, (gint16 *) mapinfo.data, mapinfo.size / element->unit_size);
				break;
			case 4:
				average_input_data_gint32(element, (gint32 *) mapinfo.data, mapinfo.size / element->unit_size);
				break;
			default:
				g_assert_not_reached();
				break;
			}
			break;
		case GSTLAL_PROPERTY_UNSIGNED:
			switch(element->unit_size) {
			case 1:
				average_input_data_guint8(element, (guint8 *) mapinfo.data, mapinfo.size / element->unit_size);
				break;
			case 2:
				average_input_data_guint16(element, (guint16 *) mapinfo.data, mapinfo.size / element->unit_size);
				break;
			case 4:
				average_input_data_guint32(element, (guint32 *) mapinfo.data, mapinfo.size / element->unit_size);
				break;
			default:
				g_assert_not_reached();
				break;
			}
			break;
		case GSTLAL_PROPERTY_FLOAT:
			switch(element->unit_size) {
			case 4:
				average_input_data_float(element, (float *) mapinfo.data, mapinfo.size / element->unit_size);
				break;
			case 8:
				average_input_data_double(element, (double *) mapinfo.data, mapinfo.size / element->unit_size);
				break;
			default:
				g_assert_not_reached();
				break;
			}
			break;
		default:
			g_assert_not_reached();
			break;

		}
		gst_buffer_unmap(buffer, &mapinfo);
	}

	return result;
}


/*
 * ============================================================================
 *
 *			      GObject Methods
 *
 * ============================================================================
 */


/*
 * properties
 */


static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec)
{
	GSTLALProperty *element = GSTLAL_PROPERTY(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_UPDATE_SAMPLES:
		element->update_samples = g_value_get_int64(value);
		break;

	case ARG_AVERAGE_SAMPLES:
		element->average_samples = g_value_get_int64(value);
		break;

	case ARG_SHIFT_SAMPLES:
		element->shift_samples = g_value_get_int64(value);
		break;

	case ARG_UPDATE_WHEN_CHANGE:
		element->update_when_change = g_value_get_boolean(value);
		break;

	case ARG_CURRENT_AVERAGE:
		element->current_average = g_value_get_double(value);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *pspec)
{
	GSTLALProperty *element = GSTLAL_PROPERTY(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_UPDATE_SAMPLES:
		g_value_set_int64(value, element->update_samples);
		break;

	case ARG_AVERAGE_SAMPLES:
		g_value_set_int64(value, element->average_samples);
		break;

	case ARG_SHIFT_SAMPLES:
		g_value_set_int64(value, element->shift_samples);
		break;

	case ARG_UPDATE_WHEN_CHANGE:
		g_value_set_boolean(value, element->update_when_change);
		break;

	case ARG_CURRENT_AVERAGE:
		g_value_set_double(value, element->current_average);
		break;

	case ARG_TIMESTAMPED_AVERAGE: ;
		GValue varray = G_VALUE_INIT;
		g_value_init(&varray, GST_TYPE_ARRAY);
		GValue t = G_VALUE_INIT;
		GValue avg = G_VALUE_INIT;
		g_value_init(&t, G_TYPE_DOUBLE);
		g_value_init(&avg, G_TYPE_DOUBLE);
		g_value_set_double(&t, (double) element->timestamp / GST_SECOND);
		g_value_set_double(&avg, element->current_average);
		gst_value_array_append_value(&varray, &t);
		gst_value_array_append_value(&varray, &avg);
		g_value_copy(&varray, value);
		g_value_unset(&t);
		g_value_unset(&avg);
		g_value_unset(&varray);
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


#define CAPS \
	"audio/x-raw, " \
	"rate = (int) [1, MAX], " \
	"channels = (int) 1, " \
	"format = (string) {"GST_AUDIO_NE(F32)", "GST_AUDIO_NE(F64)", "GST_AUDIO_NE(U8)", "GST_AUDIO_NE(U16)", "GST_AUDIO_NE(U32)", "GST_AUDIO_NE(S8)", "GST_AUDIO_NE(S16)", "GST_AUDIO_NE(S32)"}, " \
	"layout = (string) interleaved, " \
	"channel-mask = (bitmask) 0"


static void gstlal_property_class_init(GSTLALPropertyClass *klass) {

	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GstBaseSinkClass *gstbasesink_class = GST_BASE_SINK_CLASS(klass);

	gstbasesink_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
	gstbasesink_class->render = GST_DEBUG_FUNCPTR(render);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);

	gst_element_class_set_details_simple(
		element_class,
		"Convert input data to a GObject property",
		"Sink",
		"Convert single-channel input data into a GObject property that can\n\t\t\t   "
		"be passed to other elements.  The timing and frequency of updates\n\t\t\t   "
		"can be controlled by the user, or updates can be made to happen\n\t\t\t   "
		"anytime the input values change.",
		"Aaron Viets <aaron.viets@ligo.org>"
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


	properties[ARG_UPDATE_SAMPLES] = g_param_spec_int64(
		"update-samples",
		"Update Samples",
		"Number of input samples after which to update the property",
		0, G_MAXINT64, 320,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_AVERAGE_SAMPLES] = g_param_spec_int64(
		"average-samples",
		"Average Samples",
		"Number of input samples to average before updating the property",
		0, G_MAXINT64, 1,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_SHIFT_SAMPLES] = g_param_spec_int64(
		"shift-samples",
		"Shift Samples",
		"Number of input samples to shift the time of an update from a multiple of\n\t\t\t"
		"update-samples",
		G_MININT64, G_MAXINT64, 0,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_UPDATE_WHEN_CHANGE] = g_param_spec_boolean(
		"update-when-change",
		"Update When Change",
		"If true, updates will happen anytime there is a change in the input values",
		FALSE,
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
	);
	properties[ARG_CURRENT_AVERAGE] = g_param_spec_double(
		"current-average",
		"Current Average",
		"The current value of the property, averaged over average-samples samples",
		-G_MAXDOUBLE, G_MAXDOUBLE, -G_MAXDOUBLE,
		G_PARAM_READABLE | G_PARAM_STATIC_STRINGS
	);
	properties[ARG_TIMESTAMPED_AVERAGE] = gst_param_spec_array(
		"timestamped-average",
		"Timestamped Average",
		"A GstArray containing the timestamp in seconds and the current average.  The\n\t\t\t"
		"timestamp is first, then the average.  Both are double-precision floats.",
		g_param_spec_double(
			"sample",
			"Sample",
			"Either the timestamp or the average value",
			-G_MAXDOUBLE, G_MAXDOUBLE, 0.0,
			G_PARAM_READABLE | G_PARAM_STATIC_STRINGS
		),
		G_PARAM_READABLE | G_PARAM_STATIC_STRINGS
	);


	g_object_class_install_property(
		gobject_class,
		ARG_UPDATE_SAMPLES,
		properties[ARG_UPDATE_SAMPLES]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_AVERAGE_SAMPLES,
		properties[ARG_AVERAGE_SAMPLES]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_SHIFT_SAMPLES,
		properties[ARG_SHIFT_SAMPLES]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_UPDATE_WHEN_CHANGE,
		properties[ARG_UPDATE_WHEN_CHANGE]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_CURRENT_AVERAGE,
		properties[ARG_CURRENT_AVERAGE]
	);
	g_object_class_install_property(
		gobject_class,
		ARG_TIMESTAMPED_AVERAGE,
		properties[ARG_TIMESTAMPED_AVERAGE]
	);
}


/*
 * init()
 */


static void gstlal_property_init(GSTLALProperty *element) {

	g_signal_connect(G_OBJECT(element), "notify::current-average", G_CALLBACK(rebuild_workspace_and_reset), NULL);
	element->rate = 0;
	element->unit_size = 0;
	element->current_average = -G_MAXDOUBLE;
	element->num_in_avg = 0;
	element->update_samples = 0;
	element->average_samples = 0;
	element->shift_samples = 0;
	element->update_when_change = FALSE;

	gst_base_sink_set_sync(GST_BASE_SINK(element), FALSE);
	gst_base_sink_set_async_enabled(GST_BASE_SINK(element), FALSE);
}


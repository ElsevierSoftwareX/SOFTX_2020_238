/*
 * Copyright (C) 2020 Aaron Viets <aaron.viets@ligo.org>
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


#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>


/*
 *  stuff from gobject/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/audio/audio.h>
#include <gst/base/gstbasetransform.h>
#include <glib/gprintf.h>


/*
 * our own stuff
 */


#include <gstlal/gstlal_audio_info.h>
#include <gstlal_detectchange.h>


/*
 * ============================================================================
 *
 *			   GStreamer Boiler Plate
 *
 * ============================================================================
 */


#define GST_CAT_DEFAULT gstlal_detectchange_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


G_DEFINE_TYPE_WITH_CODE(
	GSTLALDetectChange,
	gstlal_detectchange,
	GST_TYPE_BASE_TRANSFORM,
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "lal_detectchange", 0, "lal_detectchange element")
);


static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SINK_NAME,
	GST_PAD_SINK,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		GST_AUDIO_CAPS_MAKE("{" GST_AUDIO_NE(F32) ", " GST_AUDIO_NE(F64) ", " GST_AUDIO_NE(Z64) ", " GST_AUDIO_NE(Z128) "}") ", " \
		"layout = (string) interleaved, " \
		"channel-mask = (bitmask) 0"
	)
);


static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SRC_NAME,
	GST_PAD_SRC,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		GST_AUDIO_CAPS_MAKE("{" GST_AUDIO_NE(F32) ", " GST_AUDIO_NE(F64) ", " GST_AUDIO_NE(Z64) ", " GST_AUDIO_NE(Z128) "}") ", " \
		"layout = (string) interleaved, " \
		"channel-mask = (bitmask) 0"
	)
);


/*
 * ============================================================================
 *
 *		     GstBaseTransform Method Overrides
 *
 * ============================================================================
 */


/*
 * get_unit_size()
 */


static gboolean get_unit_size(GstBaseTransform *trans, GstCaps *caps, gsize *size) {

	GstAudioInfo info;
	gboolean success = TRUE;

	success &= gstlal_audio_info_from_caps(&info, caps);

	if(success) {
		*size = GST_AUDIO_INFO_BPF(&info);
	} else
		GST_WARNING_OBJECT(trans, "unable to parse caps %" GST_PTR_FORMAT, caps);

	return success;
}


/*
 * set_caps()
 */


static gboolean set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps) {

	GSTLALDetectChange *element = GSTLAL_DETECTCHANGE(trans);
	gint rate_in, rate_out, channels;
	gsize unit_size;

	/*
 	 * parse the caps
 	 */

	GstStructure *str = gst_caps_get_structure(incaps, 0);
	const gchar *name = gst_structure_get_string(str, "format");
	if(!name) {
		GST_DEBUG_OBJECT(element, "unable to parse format from %" GST_PTR_FORMAT, incaps);
		return FALSE;
	}
	if(!get_unit_size(trans, incaps, &unit_size)) {
		GST_DEBUG_OBJECT(element, "function 'get_unit_size' failed");
		return FALSE;
	}
	if(!gst_structure_get_int(str, "rate", &rate_in)) {
		GST_DEBUG_OBJECT(element, "unable to parse rate from %" GST_PTR_FORMAT, incaps);
		return FALSE;
	}
	if(!gst_structure_get_int(gst_caps_get_structure(outcaps, 0), "rate", &rate_out)) {
		GST_DEBUG_OBJECT(element, "unable to parse rate from %" GST_PTR_FORMAT, outcaps);
		return FALSE;
	}
	if(!gst_structure_get_int(str, "channels", &channels)) {
		GST_DEBUG_OBJECT(element, "unable to parse channels from %" GST_PTR_FORMAT, incaps);
		return FALSE;
	}

	/*
 	 * require the output rate to be equal to the input rate
 	 */

	if(rate_out != rate_in) {
		GST_ERROR_OBJECT(element, "output rate is not equal to input rate.  input caps = %" GST_PTR_FORMAT " output caps = %" GST_PTR_FORMAT, incaps, outcaps);
		return FALSE;
	}

	/*
 	 * record stream parameters
 	 */

	if(!strcmp(name, GST_AUDIO_NE(F32))) {
			element->data_type = GSTLAL_DETECTCHANGE_F32;
			g_assert_cmpuint(unit_size, ==, 4 * (guint) channels);
		} else if(!strcmp(name, GST_AUDIO_NE(F64))) {
			element->data_type = GSTLAL_DETECTCHANGE_F64;
			g_assert_cmpuint(unit_size, ==, 8 * (guint) channels);
		} else if(!strcmp(name, GST_AUDIO_NE(Z64))) {
			element->data_type = GSTLAL_DETECTCHANGE_Z64;
			g_assert_cmpuint(unit_size, ==, 8 * (guint) channels);
		} else if(!strcmp(name, GST_AUDIO_NE(Z128))) {
			element->data_type = GSTLAL_DETECTCHANGE_Z128;
			g_assert_cmpuint(unit_size, ==, 16 * (guint) channels);
		} else
			g_assert_not_reached();

	element->rate = rate_in;
	element->unit_size = unit_size;

	return TRUE;
}


/*
 * transform_ip()
 */


static GstFlowReturn transform_ip(GstBaseTransform *trans, GstBuffer *buf) {

	GSTLALDetectChange *element = GSTLAL_DETECTCHANGE(trans);
	GstMapInfo mapinfo;
	GstFlowReturn result = GST_FLOW_OK;

	if(!GST_BUFFER_FLAG_IS_SET(buf, GST_BUFFER_FLAG_GAP)) {

		gst_buffer_map(buf, &mapinfo, GST_MAP_READWRITE);

		if(element->data_type == GSTLAL_DETECTCHANGE_F32) {

			gfloat *addr, *end;
			g_assert(mapinfo.size % sizeof(gfloat) == 0);
			addr = (gfloat *) mapinfo.data;
			end = (gfloat *) (mapinfo.data + mapinfo.size);
			while(addr < end) {
				element->current_average += *addr++;
				element->num_in_average++;
				if(element->num_in_average >= element->average_samples) {

					element->current_average /= element->average_samples;
					complex double diff = element->current_average - element->last_average;

					/* Check for detection of a change. */
					if(element->have_data && cabs(diff) >= element->detection_threshold) {
						if(element->filename) {
							FILE *fp;
							fp = fopen(element->filename, "a");
							g_fprintf(fp, "%s detected change at %10.2f: %e\n", GST_OBJECT(element)->name, GST_BUFFER_PTS(buf) / 1.0e9, creal(diff));
							fclose(fp);
						} else
							g_print("%s detected change at %10.2f: %e\n", GST_OBJECT(element)->name, GST_BUFFER_PTS(buf) / 1.0e9, creal(diff));

					}

					/* Reset element for the next average. */
					element->num_in_average = 0;
					element->last_average = element->current_average;
					element->current_average = 0.0;
					element->have_data = TRUE;
				}
			}

		} else if(element->data_type == GSTLAL_DETECTCHANGE_F64) {

			gdouble *addr, *end;
			g_assert(mapinfo.size % sizeof(gdouble) == 0);
			addr = (gdouble *) mapinfo.data;
			end = (gdouble *) (mapinfo.data + mapinfo.size);
			while(addr < end) {
				element->current_average += *addr++;
				element->num_in_average++;
				if(element->num_in_average >= element->average_samples) {

					element->current_average /= element->average_samples;
					complex double diff = element->current_average - element->last_average;

					/* Check for detection of a change. */
					if(element->have_data && cabs(diff) >= element->detection_threshold) {
						if(element->filename) {
							FILE *fp;
							fp = fopen(element->filename, "a");
							g_fprintf(fp, "%s detected change at %10.2f: %e\n", GST_OBJECT(element)->name, GST_BUFFER_PTS(buf) / 1.0e9, creal(diff));
							fclose(fp);
						} else
							g_print("%s detected change at %10.2f: %e\n", GST_OBJECT(element)->name, GST_BUFFER_PTS(buf) / 1.0e9, creal(diff));
					}

					/* Reset element for the next average. */
					element->num_in_average = 0;
					element->last_average = element->current_average;
					element->current_average = 0.0;
					element->have_data = TRUE;
				}
			}

		} else if(element->data_type == GSTLAL_DETECTCHANGE_Z64) {

			complex float *addr, *end;
			g_assert(mapinfo.size % sizeof(complex float) == 0);
			addr = (complex float *) mapinfo.data;
			end = (complex float *) (mapinfo.data + mapinfo.size);
			while(addr < end) {
				element->current_average += *addr++;
				element->num_in_average++;
				if(element->num_in_average >= element->average_samples) {

					element->current_average /= element->average_samples;
					complex double diff = element->current_average - element->last_average;

					/* Check for detection of a change. */
					if(element->have_data && cabs(diff) >= element->detection_threshold) {
						if(element->filename) {
							FILE *fp;
							fp = fopen(element->filename, "a");
							g_fprintf(fp, "%s detected change at %10.2f: %e + %ei\n", GST_OBJECT(element)->name, GST_BUFFER_PTS(buf) / 1.0e9, creal(diff), cimag(diff));
							fclose(fp);
						} else
							g_print("%s detected change at %10.2f: %e + %ei\n", GST_OBJECT(element)->name, GST_BUFFER_PTS(buf) / 1.0e9, creal(diff), cimag(diff));
					}
					/* Reset element for the next average. */
					element->num_in_average = 0;
					element->last_average = element->current_average;
					element->current_average = 0.0;
					element->have_data = TRUE;
				}
			}

		} else if(element->data_type == GSTLAL_DETECTCHANGE_Z128) {

			complex double *addr, *end;
			g_assert(mapinfo.size % sizeof(complex double) == 0);
			addr = (complex double *) mapinfo.data;
			end = (complex double *) (mapinfo.data + mapinfo.size);
			while(addr < end) {
				element->current_average += *addr++;
				element->num_in_average++;
				if(element->num_in_average >= element->average_samples) {

					element->current_average /= element->average_samples;
					complex double diff = element->current_average - element->last_average;

					/* Check for detection of a change. */
					if(element->have_data && cabs(diff) >= element->detection_threshold) {
						if(element->filename) {
							FILE *fp;
							fp = fopen(element->filename, "a");
							g_fprintf(fp, "%s detected change at %10.2f: %e + %ei\n", GST_OBJECT(element)->name, GST_BUFFER_PTS(buf) / 1.0e9, creal(diff), cimag(diff));
							fclose(fp);
						} else
							g_print("%s detected change at %10.2f: %e + %ei\n", GST_OBJECT(element)->name, GST_BUFFER_PTS(buf) / 1.0e9, creal(diff), cimag(diff));
					}
					/* Reset element for the next average. */
					element->num_in_average = 0;
					element->last_average = element->current_average;
					element->current_average = 0.0;
					element->have_data = TRUE;
				}
			}

		} else
			g_assert_not_reached();

		gst_buffer_unmap(buf, &mapinfo);
	}

	return result;
}


/*
 * ============================================================================
 *
 *			  GObject Method Overrides
 *
 * ============================================================================
 */


enum property {
	ARG_AVERAGE_SAMPLES = 1,
	ARG_DETECTION_THRESHOLD,
	ARG_FILENAME
};


/*
 * set_property()
 */


static void set_property(GObject *object, enum property prop_id, const GValue *value, GParamSpec *pspec) {

	GSTLALDetectChange *element = GSTLAL_DETECTCHANGE(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_AVERAGE_SAMPLES:
		element->average_samples = g_value_get_uint64(value);
		break;

	case ARG_DETECTION_THRESHOLD:
		element->detection_threshold = g_value_get_double(value);
		break;

	case ARG_FILENAME:
		element->filename = g_value_dup_string(value);
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


static void get_property(GObject *object, enum property prop_id, GValue *value, GParamSpec *pspec) {

	GSTLALDetectChange *element = GSTLAL_DETECTCHANGE(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_AVERAGE_SAMPLES:
		g_value_set_uint64(value, element->average_samples);
		break;

	case ARG_DETECTION_THRESHOLD:
		g_value_set_double(value, element->detection_threshold);
		break;

	case ARG_FILENAME:
		g_value_set_string(value, element->filename);
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


static void gstlal_detectchange_class_init(GSTLALDetectChangeClass *klass) {

	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

	gst_element_class_set_details_simple(
		element_class,
		"Detect change",
		"Filter/Audio",
		"Detects a change in a data stream.",
		"Aaron Viets <aaron.viets@ligo.org>"
	);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);

	g_object_class_install_property(
		gobject_class,
		ARG_AVERAGE_SAMPLES,
		g_param_spec_uint64(
			"average-samples",
			"Average Samples",
			"Number of samples to include in the average.",
			1, G_MAXUINT64, 16384,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_DETECTION_THRESHOLD,
		g_param_spec_double(
			"detection-threshold",
			"Detection Threshold",
			"Minimum magnitude of difference between averages for detection.",
			0, G_MAXDOUBLE, 0.01,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_FILENAME,
		g_param_spec_string(
			"filename",
			"Filename",
			"Name of file in which to write output. If not given, no file is produced.",
			NULL,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));

	transform_class->get_unit_size = GST_DEBUG_FUNCPTR(get_unit_size);
	transform_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
	transform_class->transform_ip = GST_DEBUG_FUNCPTR(transform_ip);
}


/*
 * init()
 */


static void gstlal_detectchange_init(GSTLALDetectChange *element) {

	element->last_average = 0.0;
	element->current_average = 0.0;
	element->num_in_average = 0;
	element->have_data = FALSE;
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(element), TRUE);
}


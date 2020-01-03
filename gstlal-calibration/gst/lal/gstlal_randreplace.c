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


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
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


/*
 * our own stuff
 */


#include <gstlal/gstlal_audio_info.h>
#include <gstlal_randreplace.h>


/*
 * ============================================================================
 *
 *			   GStreamer Boiler Plate
 *
 * ============================================================================
 */


#define GST_CAT_DEFAULT gstlal_randreplace_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


G_DEFINE_TYPE_WITH_CODE(
	GSTLALRandReplace,
	gstlal_randreplace,
	GST_TYPE_BASE_TRANSFORM,
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "lal_randreplace", 0, "lal_randreplace element")
);


static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SINK_NAME,
	GST_PAD_SINK,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		GST_AUDIO_CAPS_MAKE("{" GST_AUDIO_NE(U32) ", " GST_AUDIO_NE(F32) ", " GST_AUDIO_NE(F64) ", " GST_AUDIO_NE(Z64) ", " GST_AUDIO_NE(Z128) "}") ", " \
		"layout = (string) interleaved, " \
		"channel-mask = (bitmask) 0"
	)
);


static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SRC_NAME,
	GST_PAD_SRC,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		GST_AUDIO_CAPS_MAKE("{" GST_AUDIO_NE(U32) ", " GST_AUDIO_NE(F32) ", " GST_AUDIO_NE(F64) ", " GST_AUDIO_NE(Z64) ", " GST_AUDIO_NE(Z128) "}") ", " \
		"layout = (string) interleaved, " \
		"channel-mask = (bitmask) 0"
	)
);


/*
 * ============================================================================
 *
 *				    Utilities
 *
 * ============================================================================
 */


#define DEFINE_RANDOM_REPLACE(DTYPE, COMPLEX) \
static void random_replace_ ## DTYPE ## COMPLEX(COMPLEX DTYPE *data, guint64 total_samples, double min, double max, guint64 max_replace_samples, enum gstlal_randreplace_data_type data_type) { \
 \
	srand(time(0)); \
 \
	guint64 replace_samples, i = 0; \
	COMPLEX DTYPE *end, replacement; \
	int sign; \
	double log10max = log10(max); \
	double log10min = log10(min); \
	double log10replacement; \
 \
	while(i < total_samples) { \
		/* First, determine how many samples in a row we will replace. */ \
		replace_samples = 1 + (guint64) ((double) rand() / RAND_MAX * (max_replace_samples - 1)); \
		/* Don't go beyond the end of the buffer */ \
		if(replace_samples > total_samples - i) \
			replace_samples = total_samples - i; \
 \
		/* Now compute the replace value */ \
		if(data_type == GSTLAL_RANDREPLACE_U32) \
			replacement = (DTYPE) (min + (max - min) * rand() / RAND_MAX); \
		else { \
			sign = rand() > RAND_MAX / 2 ? 1 : -1; \
			log10replacement = log10min + (log10max - log10min) * rand() / RAND_MAX; \
			replacement = (COMPLEX DTYPE) (sign * pow(10.0, log10replacement)); \
		} \
 \
		/* Now do the replacing */ \
		end = data + replace_samples; \
		while(data < end) { \
			*data = replacement; \
			data++; \
		} \
		i += replace_samples; \
	} \
 \
	return; \
}


DEFINE_RANDOM_REPLACE(guint32, );
DEFINE_RANDOM_REPLACE(float, );
DEFINE_RANDOM_REPLACE(double, );
DEFINE_RANDOM_REPLACE(float, complex);
DEFINE_RANDOM_REPLACE(double, complex);


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

	GSTLALRandReplace *element = GSTLAL_RANDREPLACE(trans);
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

	if(!strcmp(name, GST_AUDIO_NE(U32))) {
		element->data_type = GSTLAL_RANDREPLACE_U32;
		g_assert_cmpuint(unit_size, ==, 4 * (guint) channels);
		if(element->replace_max > G_MAXUINT32) {
			GST_INFO_OBJECT(element, "Stream is unsigned integers, so replace-max must not be greater than %u.  Setting replace-max to %u.", G_MAXUINT32, G_MAXUINT32);
			element->replace_max = G_MAXUINT32;
		}
		if(element->replace_min_magnitude > G_MAXUINT32) {
			GST_INFO_OBJECT(element, "Stream is unsigned integers, so replace-min-magnitude must not be greater than %u.  Setting replace-min-magnitude to %u.", G_MAXUINT32, G_MAXUINT32);
			element->replace_min_magnitude = G_MAXUINT32;
		}
		/* Truncate it so that the default value is zero. */
		element->replace_min_magnitude = (double) ((guint64) element->replace_min_magnitude);
	} else if(!strcmp(name, GST_AUDIO_NE(F32))) {
		element->data_type = GSTLAL_RANDREPLACE_F32;
		g_assert_cmpuint(unit_size, ==, 4 * (guint) channels);
		if(element->replace_max > G_MAXFLOAT) {
			GST_INFO_OBJECT(element, "Single-precision floating point stream cannot take values greater than %e.  Setting replace-max to %e", G_MAXFLOAT, G_MAXFLOAT);
			element->replace_max = G_MAXFLOAT;
		}
		if(element->replace_min_magnitude < G_MINFLOAT) {
			GST_INFO_OBJECT(element, "Single-precision floating point stream cannot take values smaller in magnitude than %e.  Setting replace-min-magnitude to %e", G_MINFLOAT, G_MINFLOAT);
			element->replace_min_magnitude = G_MINFLOAT;
		}
	} else if(!strcmp(name, GST_AUDIO_NE(F64))) {
		element->data_type = GSTLAL_RANDREPLACE_F64;
		g_assert_cmpuint(unit_size, ==, 8 * (guint) channels);
		if(element->replace_min_magnitude < G_MINDOUBLE) {
			GST_INFO_OBJECT(element, "Double-precision floating point stream cannot take values smaller in magnitude than %e.  Setting replace-min-magnitude to %e", G_MINDOUBLE, G_MINDOUBLE);
			element->replace_min_magnitude = G_MINFLOAT;
		}
	} else if(!strcmp(name, GST_AUDIO_NE(Z64))) {
		element->data_type = GSTLAL_RANDREPLACE_Z64;
		g_assert_cmpuint(unit_size, ==, 8 * (guint) channels);
		if(element->replace_max > G_MAXFLOAT) {
			GST_INFO_OBJECT(element, "Single-precision floating point stream cannot take values greater than %e.  Setting replace-max to %e", G_MAXFLOAT, G_MAXFLOAT);
			element->replace_max = G_MAXFLOAT;
		}
		if(element->replace_min_magnitude < G_MINFLOAT) {
			GST_INFO_OBJECT(element, "Single-precision floating point stream cannot take values smaller in magnitude than %e.  Setting replace-min-magnitude to %e", G_MINFLOAT, G_MINFLOAT);
			element->replace_min_magnitude = G_MINFLOAT;
		}
	} else if(!strcmp(name, GST_AUDIO_NE(Z128))) {
		element->data_type = GSTLAL_RANDREPLACE_Z128;
		g_assert_cmpuint(unit_size, ==, 16 * (guint) channels);
		if(element->replace_min_magnitude < G_MINDOUBLE) {
			GST_INFO_OBJECT(element, "Double-precision floating point stream cannot take values smaller in magnitude than %e.  Setting replace-min-magnitude to %e", G_MINDOUBLE, G_MINDOUBLE);
			element->replace_min_magnitude = G_MINFLOAT;
		}
	} else
		g_assert_not_reached();

	element->rate = rate_in;
	element->unit_size = unit_size;

	/* Some checks */
	if(element->replace_max < element->replace_min_magnitude) {
		GST_WARNING_OBJECT(element, "replace-max should be greater than replace-min-magnitude (replace-max=%e, replace-min=%e).  Switching them.", element->replace_max, element->replace_min_magnitude);
		double max = element->replace_min_magnitude;
		element->replace_min_magnitude = element->replace_max;
		element->replace_max = max;
	}

	return TRUE;
}


/*
 * transform_ip()
 */


static GstFlowReturn transform_ip(GstBaseTransform *trans, GstBuffer *buf) {

	GSTLALRandReplace *element = GSTLAL_RANDREPLACE(trans);
	GstFlowReturn result = GST_FLOW_OK;

	srand(time(0));
	if((double) rand() / RAND_MAX < element->replace_probability) {

		GstMapInfo mapinfo;
		gst_buffer_map(buf, &mapinfo, GST_MAP_READWRITE);

		switch(element->data_type) {

		case GSTLAL_RANDREPLACE_U32:
			random_replace_guint32((void *) mapinfo.data, mapinfo.size / element->unit_size, element->replace_min_magnitude, element->replace_max, element->max_replace_samples, element->data_type);
			break;

		case GSTLAL_RANDREPLACE_F32:
			random_replace_float((void *) mapinfo.data, mapinfo.size / element->unit_size, element->replace_min_magnitude, element->replace_max, element->max_replace_samples, element->data_type);
			break;

		case GSTLAL_RANDREPLACE_F64:
			random_replace_double((void *) mapinfo.data, mapinfo.size / element->unit_size, element->replace_min_magnitude, element->replace_max, element->max_replace_samples, element->data_type);
			break;

		case GSTLAL_RANDREPLACE_Z64:
			random_replace_floatcomplex((void *) mapinfo.data, mapinfo.size / element->unit_size, element->replace_min_magnitude, element->replace_max, element->max_replace_samples, element->data_type);
			break;

		case GSTLAL_RANDREPLACE_Z128:
			random_replace_doublecomplex((void *) mapinfo.data, mapinfo.size / element->unit_size, element->replace_min_magnitude, element->replace_max, element->max_replace_samples, element->data_type);
			break;

		default:
			g_assert_not_reached();
		}

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
	ARG_REPLACE_PROBABILITY = 1,
	ARG_REPLACE_MAX,
	ARG_REPLACE_MIN_MAGNITUDE,
	ARG_MAX_REPLACE_SAMPLES
};


/*
 * set_property()
 */


static void set_property(GObject *object, enum property prop_id, const GValue *value, GParamSpec *pspec) {

	GSTLALRandReplace *element = GSTLAL_RANDREPLACE(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_REPLACE_PROBABILITY:
		element->replace_probability = g_value_get_double(value);
		break;

	case ARG_REPLACE_MAX:
		element->replace_max = g_value_get_double(value);
		break;

	case ARG_REPLACE_MIN_MAGNITUDE:
		element->replace_min_magnitude = g_value_get_double(value);
		break;

	case ARG_MAX_REPLACE_SAMPLES:
		element->max_replace_samples = g_value_get_uint64(value);
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

	GSTLALRandReplace *element = GSTLAL_RANDREPLACE(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_REPLACE_PROBABILITY:
		g_value_set_double(value, element->replace_probability);
		break;

	case ARG_REPLACE_MAX:
		g_value_set_double(value, element->replace_max);
		break;

	case ARG_REPLACE_MIN_MAGNITUDE:
		g_value_set_double(value, element->replace_min_magnitude);
		break;

	case ARG_MAX_REPLACE_SAMPLES:
		g_value_set_uint64(value, element->max_replace_samples);
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


static void gstlal_randreplace_class_init(GSTLALRandReplaceClass *klass) {

	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

	gst_element_class_set_details_simple(
		element_class,
		"Random Replace",
		"Filter/Audio",
		"Replaces data in a stream with random numbers",
		"Aaron Viets <aaron.viets@ligo.org>"
	);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);

	g_object_class_install_property(
		gobject_class,
		ARG_REPLACE_PROBABILITY,
		g_param_spec_double(
			"replace-probability",
			"Replace probability",
			"Probability that a given buffer gets replaced",
			0, G_MAXDOUBLE, 0.5,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_REPLACE_MAX,
		g_param_spec_double(
			"replace-max",
			"Replace maximum",
			"Maximum value that can data can be replaced with.  For signed streams, the\n\t\t\t"
			"minumum is -replace-max.",
			0, G_MAXDOUBLE, G_MAXDOUBLE,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_REPLACE_MIN_MAGNITUDE,
		g_param_spec_double(
			"replace-min-magnitude",
			"Replace minimum magnitude",
			"Minimum magnitude of value that can data can be replaced with.  For integer\n\t\t\t"
			"streams, this will be truncated to an integer.",
			0, G_MAXDOUBLE, G_MINDOUBLE,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_MAX_REPLACE_SAMPLES,
		g_param_spec_uint64(
			"max-replace-samples",
			"Maximum replace samples",
			"Maximum number of samples in a row that can be replaced by the same value.\n\t\t\t"
			"The actual number in a row is random.",
			1, G_MAXUINT64, 1,
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


static void gstlal_randreplace_init(GSTLALRandReplace *element) {

	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(element), TRUE);
}



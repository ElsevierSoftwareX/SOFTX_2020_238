/*
 * Copyright (C) 2021  Aaron Viets <aaron.viets@ligo.org>
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
#include <gstlal_minmax.h>


/*
 * ============================================================================
 *
 *			   GStreamer Boiler Plate
 *
 * ============================================================================
 */


#define INCAPS \
	"audio/x-raw, " \
	"format = (string) {"GST_AUDIO_NE(F32)", "GST_AUDIO_NE(F64)", "GST_AUDIO_NE(Z64)", "GST_AUDIO_NE(Z128)"}, " \
	"rate = " GST_AUDIO_RATE_RANGE ", " \
	"channels = (int) [1, MAX], " \
	"layout = (string) interleaved, " \
	"channel-mask = (bitmask) 0"

#define OUTCAPS \
	"audio/x-raw, " \
	"format = (string) {"GST_AUDIO_NE(F32)", "GST_AUDIO_NE(F64)", "GST_AUDIO_NE(Z64)", "GST_AUDIO_NE(Z128)"}, " \
	"rate = " GST_AUDIO_RATE_RANGE ", " \
	"channels = (int) 1, " \
	"layout = (string) interleaved, " \
	"channel-mask = (bitmask) 0"


static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SINK_NAME,
	GST_PAD_SINK,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(INCAPS)
);


static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SRC_NAME,
	GST_PAD_SRC,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(OUTCAPS)
);


G_DEFINE_TYPE(
	GSTLALMinMax,
	gstlal_minmax,
	GST_TYPE_BASE_TRANSFORM
);


/*
 * ============================================================================
 *
 *				Utilities
 *
 * ============================================================================
 */


/*
 * set the metadata on an output buffer
 */


static void set_metadata(GSTLALMinMax *element, GstBuffer *buf, guint64 outsamples, gboolean gap) {

	GST_BUFFER_OFFSET(buf) = element->next_out_offset;
	element->next_out_offset += outsamples;
	GST_BUFFER_OFFSET_END(buf) = element->next_out_offset;
	GST_BUFFER_TIMESTAMP(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET(buf) - element->offset0, GST_SECOND, element->rate);
	GST_BUFFER_DURATION(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET_END(buf) - element->offset0, GST_SECOND, element->rate) - GST_BUFFER_TIMESTAMP(buf);
	GST_BUFFER_FLAG_UNSET(buf, GST_BUFFER_FLAG_GAP);
	if(G_UNLIKELY(element->need_discont)) {
		GST_BUFFER_FLAG_SET(buf, GST_BUFFER_FLAG_DISCONT);
		element->need_discont = FALSE;
	}
	if(gap)
		GST_BUFFER_FLAG_SET(buf, GST_BUFFER_FLAG_GAP);
	else
		GST_BUFFER_FLAG_UNSET(buf, GST_BUFFER_FLAG_GAP);
}


/*
 * Find the minimum or maximum, or the minimum or maximum of an absolute value
 */


#define DEFINE_MINMAX(COMPLEX, DTYPE, C_OR_F, F_OR_NOT) \
void minmax_ ## COMPLEX ## DTYPE(const COMPLEX DTYPE *src, COMPLEX DTYPE *dst, guint64 dst_size, GSTLALMinMax *element) { \
	guint64 i; \
	int j, j_max; \
	COMPLEX DTYPE out; \
	switch(element->mode) { \
	case 0: \
		/* We are finding the minimum. */ \
		for(i = 0; i < dst_size; i++) { \
			out = element->max_ ## DTYPE; \
			j_max = (i + 1) * element->channels_in; \
			for(j = i * element->channels_in; j < j_max; j++) \
				out = creal ## F_OR_NOT(out) < creal ## F_OR_NOT(src[j]) ? out : src[j]; \
			dst[i] = out; \
		} \
		break; \
	case 1: \
		/* We are finding the minimum of the absolute value. */ \
		for(i = 0; i < dst_size; i++) { \
			out = element->max_ ## DTYPE; \
			j_max = (i + 1) * element->channels_in; \
			for(j = i * element->channels_in; j < j_max; j++) \
				out = C_OR_F ## abs ## F_OR_NOT(out) < C_OR_F ## abs ## F_OR_NOT(src[j]) ? out : src[j]; \
			dst[i] = out; \
		} \
		break; \
	case 2: \
		/* We are finding the maximum. */ \
		for(i = 0; i < dst_size; i++) { \
			out = -element->max_ ## DTYPE; \
			j_max = (i + 1) * element->channels_in; \
			for(j = i * element->channels_in; j < j_max; j++) \
				out = creal ## F_OR_NOT(out) > creal ## F_OR_NOT(src[j]) ? out : src[j]; \
			dst[i] = out; \
		} \
		break; \
	case 3: \
		/* We are finding the maximum of the absolute value. */ \
		for(i = 0; i < dst_size; i++) { \
			out = 0.0; \
			j_max = (i + 1) * element->channels_in; \
			for(j = i * element->channels_in; j < j_max; j++) \
				out = C_OR_F ## abs ## F_OR_NOT(out) > C_OR_F ## abs ## F_OR_NOT(src[j]) ? out : src[j]; \
			dst[i] = out; \
		} \
		break; \
	} \
 \
	return; \
}


DEFINE_MINMAX( , float, f, f);
DEFINE_MINMAX( , double, f, );
DEFINE_MINMAX(complex, float, c, f);
DEFINE_MINMAX(complex, double, c, );


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


static gboolean get_unit_size(GstBaseTransform *trans, GstCaps *caps, gsize *size)
{
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
 * transform_caps()
 */


static GstCaps *transform_caps(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, GstCaps *filter) {

	guint n;
	caps = gst_caps_normalize(gst_caps_copy(caps));

	switch(direction) {
	case GST_PAD_SRC:
		/* 
		 * We know the caps on the source pad, and we want to put constraints on
		 * the sink pad caps.  The sink pad caps can have any number of channels,
		 * but are otherwise the same.
		 */
		for(n = 0; n < gst_caps_get_size(caps); n++) {
			GstStructure *str = gst_caps_get_structure(caps, n);
			gst_structure_set(str, "channels", GST_TYPE_INT_RANGE, 1, G_MAXINT, NULL);
		}
		break;

	case GST_PAD_SINK:
		/*
		 * We know the caps on the sink pad, and we want to put constraints on
		 * the source pad caps.  The source pad caps must have exactly one
		 * channel, but are otherwise the same.
		 */
		for(n = 0; n < gst_caps_get_size(caps); n++) {
			GstStructure *str = gst_caps_get_structure(caps, n);
			gst_structure_set(str, "channels", G_TYPE_INT, 1, NULL);
		}
		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		goto error;
	default:
		g_assert_not_reached();
	}

	if(filter) {
		GstCaps *intersection = gst_caps_intersect(caps, filter);
		gst_caps_unref(caps);
		caps = intersection;
	}
	return gst_caps_simplify(caps);

error:
	gst_caps_unref(caps);
	return GST_CAPS_NONE;
}


/*
 * set_caps()
 */


static gboolean set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps)
{
	GSTLALMinMax *element = GSTLAL_MINMAX(trans);
	gint rate, channels_in;
	gsize unit_size_in, unit_size_out;

	/*
 	 * parse the caps
 	 */

	GstStructure *outstr = gst_caps_get_structure(outcaps, 0);
	GstStructure *instr = gst_caps_get_structure(incaps, 0);
	const gchar *name = gst_structure_get_string(outstr, "format");
	if(!name) {
		GST_DEBUG_OBJECT(element, "unable to parse format from %" GST_PTR_FORMAT, outcaps);
		return FALSE;
	}
	if(!get_unit_size(trans, outcaps, &unit_size_out)) {
		GST_DEBUG_OBJECT(element, "function 'get_unit_size' failed");
		return FALSE;
	}
	if(!get_unit_size(trans, incaps, &unit_size_in)) {
		GST_DEBUG_OBJECT(element, "function 'get_unit_size' failed");
		return FALSE;
	}
	if(!gst_structure_get_int(outstr, "rate", &rate)) {
		GST_DEBUG_OBJECT(element, "unable to parse rate from %" GST_PTR_FORMAT, outcaps);
		return FALSE;
	}
	if(!gst_structure_get_int(instr, "channels", &channels_in)) {
		GST_DEBUG_OBJECT(element, "unable to parse channels from %" GST_PTR_FORMAT, incaps);
		return FALSE;
	}

	/*
 	 * record stream parameters
 	 */

	if(!strcmp(name, GST_AUDIO_NE(F32))) {
		element->data_type = GSTLAL_MINMAX_F32;
		g_assert_cmpuint(unit_size_in, ==, 4 * (guint) channels_in);
	} else if(!strcmp(name, GST_AUDIO_NE(F64))) {
		element->data_type = GSTLAL_MINMAX_F64;
		g_assert_cmpuint(unit_size_in, ==, 8 * (guint) channels_in);
	} else if(!strcmp(name, GST_AUDIO_NE(Z64))) {
		element->data_type = GSTLAL_MINMAX_Z64;
		g_assert_cmpuint(unit_size_in, ==, 8 * (guint) channels_in);
	} else if(!strcmp(name, GST_AUDIO_NE(Z128))) {
		element->data_type = GSTLAL_MINMAX_Z128;
		g_assert_cmpuint(unit_size_in, ==, 16 * (guint) channels_in);
	} else
		g_assert_not_reached();

	element->rate = rate;
	element->channels_in = channels_in;
	element->unit_size_out = unit_size_out;
	element->unit_size_in = unit_size_in;

	return TRUE;
}


/*
 * transform_size{}
 */


static gboolean transform_size(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, gsize size, GstCaps *othercaps, gsize *othersize) {

	GSTLALMinMax *element = GSTLAL_MINMAX(trans);

	/*
	 * The data types of inputs and outputs are the same, but the number of channels differs.
	 */

	switch(direction) {
	case GST_PAD_SRC:
		/*
		 * We know the size of the output buffer and want to compute the size of the input buffer.
		 * The size of the output buffer should be a multiple of unit_size_out.
		 */

		if(G_UNLIKELY(size % element->unit_size_out)) {
			GST_DEBUG_OBJECT(element, "buffer size %" G_GSIZE_FORMAT " is not a multiple of %" G_GSIZE_FORMAT, size, (gsize) element->unit_size_out);
			return FALSE;
		}

		*othersize = size * element->unit_size_in / element->unit_size_out;

		break;

	case GST_PAD_SINK:
		/*
		 * We know the size of the input buffer and want to compute the size of the output buffer.
		 * The size of the input buffer should be a multiple of unit_size_in.
		 */

		if(G_UNLIKELY(size % element->unit_size_in)) {
			GST_ERROR_OBJECT(element, "buffer size %" G_GSIZE_FORMAT " is not a multiple of %" G_GSIZE_FORMAT, size, (gsize) element->unit_size_in);
			return FALSE;
		}

		*othersize = size * element->unit_size_out / element->unit_size_in;

		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		return FALSE;
	}

	return TRUE;
}


/*
 * transform()
 */


static GstFlowReturn transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf) {

	GSTLALMinMax *element = GSTLAL_MINMAX(trans);
	GstMapInfo inmap, outmap;
	GstFlowReturn result = GST_FLOW_OK;

	/*
	 * check for discontinuity
	 */

	if(G_UNLIKELY(GST_BUFFER_IS_DISCONT(inbuf) || GST_BUFFER_OFFSET(inbuf) != element->next_in_offset || !GST_CLOCK_TIME_IS_VALID(element->t0))) {
		GST_DEBUG_OBJECT(element, "pushing discontinuous buffer at input timestamp %lu", (long unsigned) GST_TIME_AS_SECONDS(GST_BUFFER_PTS(inbuf)));
		element->t0 = GST_BUFFER_PTS(inbuf);
		element->offset0 = element->next_out_offset = GST_BUFFER_OFFSET(inbuf);
		element->need_discont = TRUE;
	}
	element->next_in_offset = GST_BUFFER_OFFSET_END(inbuf);

	/*
	 * process buffer
	 */

	gst_buffer_map(inbuf, &inmap, GST_MAP_READ);
	gst_buffer_map(outbuf, &outmap, GST_MAP_WRITE);

	switch(element->data_type) {
	case GSTLAL_MINMAX_F32:
		minmax_float((const float *) inmap.data, (float *) outmap.data, outmap.size / element->unit_size_out, element);
		break;
	case GSTLAL_MINMAX_F64:
		minmax_double((const double *) inmap.data, (double *) outmap.data, outmap.size / element->unit_size_out, element);
		break;
	case GSTLAL_MINMAX_Z64:
		minmax_complexfloat((const complex float *) inmap.data, (complex float *) outmap.data, outmap.size / element->unit_size_out, element);
		break;
	case GSTLAL_MINMAX_Z128:
		minmax_complexdouble((const complex double *) inmap.data, (complex double *) outmap.data, outmap.size / element->unit_size_out, element);
		break;
	default:
		g_assert_not_reached();
	}

	set_metadata(element, outbuf, outmap.size / element->unit_size_out, FALSE);
	gst_buffer_unmap(outbuf, &outmap);
	gst_buffer_unmap(inbuf, &inmap);

	/*
	 * done
	 */

	return result;
}


/*
 * ============================================================================
 *
 *			  GObject Method Overrides
 *
 * ============================================================================
 */


/*
 * properties
 */


enum property {
	ARG_MODE = 1,
};


static void set_property(GObject *object, enum property prop_id, const GValue *value, GParamSpec *pspec) {

	GSTLALMinMax *element = GSTLAL_MINMAX(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_MODE:
		element->mode = g_value_get_int(value);
		break;
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);	
}


static void get_property(GObject *object, enum property prop_id, GValue *value, GParamSpec *pspec) {

	GSTLALMinMax *element = GSTLAL_MINMAX(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_MODE:
		g_value_set_int(value, element->mode);
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


static void gstlal_minmax_class_init(GSTLALMinMaxClass *klass)
{
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

	gst_element_class_set_details_simple(
		element_class,
		"Min/Max",
		"Filter/Audio",
		"Reads in n channels and outputs one channel containing the minimum or maximum\n\t\t\t"
		"of each input value.",
		"Aaron Viets <aaron.viets@ligo.org>"
	);

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));

	transform_class->get_unit_size = GST_DEBUG_FUNCPTR(get_unit_size);
	transform_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
	transform_class->transform_caps = GST_DEBUG_FUNCPTR(transform_caps);
	transform_class->transform_size = GST_DEBUG_FUNCPTR(transform_size);
	transform_class->transform = GST_DEBUG_FUNCPTR(transform);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);

	g_object_class_install_property(
		gobject_class,
		ARG_MODE,
		g_param_spec_int(
			"mode",
			"Mode",
			"Which extreme value you want the element to find.\n\t\t\t"
			"mode=0: compute minimum of input\n\t\t\t"
			"mode=1: compute absolute value of minimum of input\n\t\t\t"
			"mode=2: compute maximum of input\n\t\t\t"
			"mode=3: compute absolute value of maximum of input\n\t\t\t"
			"Note that, for complex streams, if mode=0 or mode=2, only the real part is\n\t\t\t"
			"considered.",
			0, 3, 0,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
}


/*
 * init()
 */


static void gstlal_minmax_init(GSTLALMinMax *element) {

	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(element), TRUE);
	element->rate = 0;
	element->channels_in = 0;
	element->unit_size_in = 0;
	element->unit_size_out = 0;
	element->t0 = GST_CLOCK_TIME_NONE;
	element->offset0 = GST_BUFFER_OFFSET_NONE;
	element->next_in_offset = GST_BUFFER_OFFSET_NONE;
	element->next_out_offset = GST_BUFFER_OFFSET_NONE;
	element->need_discont = TRUE;
	element->max_double = G_MAXDOUBLE;
	element->max_float = G_MAXFLOAT;
}


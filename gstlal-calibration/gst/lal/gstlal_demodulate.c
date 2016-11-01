/*
 * Copyright (C) 2016 Aaron Viets <aaron.viets@ligo.org>
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
 *  stuff from the C library
 */


#include <string.h>
#include <complex.h>
#include <math.h>


/*
 *  stuff from gobject/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/audio/audio.h>
#include <gst/base/gstbasetransform.h>
#include <gst/audio/audio.h>


/*
 * our own stuff
 */


#include <gstlal/gstlal_audio_info.h>
#include <gstlal_demodulate.h>


/*
 * ============================================================================
 *
 *				 Utilities
 *
 * ============================================================================
 */


static void demodulate_float(const float *src, gsize src_size, float complex *dst, guint64 t_reduced, gint rate, const double frequency)
{
	const float *src_end;
	guint64 i = 0;
	guint64 t;
	for(src_end = src + src_size; src < src_end; src++, dst++) {
		t = t_reduced + gst_util_uint64_scale_int_round(i, 1000000000, rate);
		*dst = *src * cexpf(-2. * M_PI * I * frequency * t / 1000000000.0);
	}
}


static void demodulate_double(const double *src, gsize src_size, double complex *dst, guint64 t_reduced, gint rate, const double frequency)
{
	const double *src_end;
	guint64 i = 0;
	guint64 t;
	for(src_end = src + src_size; src < src_end; src++, dst++, i++) {
		t = t_reduced + gst_util_uint64_scale_int_round(i, 1000000000, rate);
		*dst = *src * cexp(-2. * M_PI * I * frequency * t / 1000000000.0);
	}
}


static void demodulate(const void *src, gsize src_size, void *dst, gsize dst_size, guint64 t_reduced, gint rate, gint unit_size, const double frequency)
{
	g_assert_cmpuint(src_size % unit_size, ==, 0);
	g_assert_cmpuint(dst_size % unit_size, ==, 0);
	g_assert_cmpuint(2 * src_size, ==, dst_size);

	dst_size /= unit_size;
	src_size /= unit_size;

	switch(unit_size) {
	case 4:
		demodulate_float(src, src_size, dst, t_reduced, rate, frequency);
		break;
	case 8:
		demodulate_double(src, src_size, dst, t_reduced, rate, frequency);
		break;
	default:
		g_assert_not_reached();
	}
}


/*
 * set the metadata on an output buffer
 */


static void set_metadata(GSTLALDemodulate *element, GstBuffer *buf, guint64 outsamples, gboolean gap)
{
	GST_BUFFER_OFFSET(buf) = element->next_out_offset;
	element->next_out_offset += outsamples;
	GST_BUFFER_OFFSET_END(buf) = element->next_out_offset;
	GST_BUFFER_TIMESTAMP(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET(buf) - element->offset0, GST_SECOND, element->rate);
	GST_BUFFER_DURATION(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET_END(buf) - element->offset0, GST_SECOND, element->rate) - GST_BUFFER_TIMESTAMP(buf);
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
 * ============================================================================
 *
 *			   GStreamer Boiler Plate
 *
 * ============================================================================
 */


#define INCAPS \
	"audio/x-raw, " \
	"format = (string) {"GST_AUDIO_NE(F32)", "GST_AUDIO_NE(F64)"}, " \
	"rate = " GST_AUDIO_RATE_RANGE ", " \
	"channels = (int) [1, MAX], " \
	"layout = (string) interleaved, " \
	"channel-mask = (bitmask) 0"

#define OUTCAPS \
	"audio/x-raw, " \
	"format = (string) {"GST_AUDIO_NE(Z64)", "GST_AUDIO_NE(Z128)"}, " \
	"rate = " GST_AUDIO_RATE_RANGE ", " \
	"channels = (int) [1, MAX], " \
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
	GSTLALDemodulate,
	gstlal_demodulate,
	GST_TYPE_BASE_TRANSFORM
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


static gboolean get_unit_size(GstBaseTransform *trans, GstCaps *caps, gsize *size)
{
	GstAudioInfo info;
	gboolean success = gstlal_audio_info_from_caps(&info, caps);
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
	guint n;

	caps = gst_caps_normalize(gst_caps_copy(caps));

	switch(direction) {
	case GST_PAD_SRC:
		for(n = 0; n < gst_caps_get_size(caps); n++) {
			GstStructure *str = gst_caps_get_structure(caps, n);
			const gchar *format = gst_structure_get_string(str, "format");

			if(!format) {
				GST_DEBUG_OBJECT(trans, "unrecognized caps %" GST_PTR_FORMAT, caps);
				goto error;
			} else if(!strcmp(format, GST_AUDIO_NE(Z64)))
				gst_structure_set(str, "format", G_TYPE_STRING, GST_AUDIO_NE(F32), NULL);
			else if(!strcmp(format, GST_AUDIO_NE(Z128)))
				gst_structure_set(str, "format", G_TYPE_STRING, GST_AUDIO_NE(F64), NULL);
			else {
				GST_DEBUG_OBJECT(trans, "unrecognized format %s in %" GST_PTR_FORMAT, format, caps);
				goto error;
			}
		}
		break;

	case GST_PAD_SINK:
		for(n = 0; n < gst_caps_get_size(caps); n++) {
			GstStructure *str = gst_caps_get_structure(caps, n);
			const gchar *format = gst_structure_get_string(str, "format");

			if(!format) {
				GST_DEBUG_OBJECT(trans, "unrecognized caps %" GST_PTR_FORMAT, caps);
				goto error;
			} else if(!strcmp(format, GST_AUDIO_NE(F32)))
				gst_structure_set(str, "format", G_TYPE_STRING, GST_AUDIO_NE(Z64), NULL);
			else if(!strcmp(format, GST_AUDIO_NE(F64)))
				gst_structure_set(str, "format", G_TYPE_STRING, GST_AUDIO_NE(Z128), NULL);
			else {
				GST_DEBUG_OBJECT(trans, "unrecognized format %s in %" GST_PTR_FORMAT, format, caps);
				goto error;
			}
		}
		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		goto error;
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
	GSTLALDemodulate *element = GSTLAL_DEMODULATE(trans);
	gint rate_in, rate_out;
	gsize unit_size;

	/*
	 * parse the caps
	 */

	if(!get_unit_size(trans, incaps, &unit_size)) {
		GST_DEBUG_OBJECT(element, "function 'get_unit_size' failed");
		return FALSE;
	}
	if(!gst_structure_get_int(gst_caps_get_structure(incaps, 0), "rate", &rate_in)) {
		GST_DEBUG_OBJECT(element, "unable to parse rate from %" GST_PTR_FORMAT, incaps);
		return FALSE;
	}
	if(!gst_structure_get_int(gst_caps_get_structure(outcaps, 0), "rate", &rate_out)) {
		GST_DEBUG_OBJECT(element, "unable to parse rate from %" GST_PTR_FORMAT, outcaps);
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

	element->rate = rate_in;
	element->unit_size = unit_size;

	return TRUE;
}


/*
 * transform_size()
 */


static gboolean transform_size(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, gsize size, GstCaps *othercaps, gsize *othersize)
{
	GSTLALDemodulate *element = GSTLAL_DEMODULATE(trans);

	gsize unit_size;

	switch(direction) {
	case GST_PAD_SRC:
		/*
		 * We have the size of the output buffer, and we set the size of the input buffer,
		 * which is half as large in bytes.
		 */

		if(!get_unit_size(trans, caps, &unit_size)) {
			GST_DEBUG_OBJECT(element, "function 'get_unit_size' failed");
			return FALSE;
		}

		/* buffer size in bytes should be a multiple of unit_size in bytes */
		if(G_UNLIKELY(size % unit_size)) {
			GST_DEBUG_OBJECT(element, "buffer size %" G_GSIZE_FORMAT " is not a multiple of %" G_GSIZE_FORMAT, size, unit_size);
			return FALSE;
		}

		*othersize = size / 2;
		break;

	case GST_PAD_SINK:
		/*
		 * We have the size of the input buffer, and we set the size of the output buffer,
		 * which is twice as large in bytes.
		 */

		if(!get_unit_size(trans, caps, &unit_size)) {
			GST_DEBUG_OBJECT(element, "function 'get_unit_size' failed");
			return FALSE;
		}

		/* buffer size in bytes should be a multiple of unit_size in bytes */
		if(G_UNLIKELY(size % unit_size)) {
			GST_DEBUG_OBJECT(element, "buffer size %" G_GSIZE_FORMAT " is not a multiple of %" G_GSIZE_FORMAT, size, unit_size);
			return FALSE;
		}

		*othersize = 2 * size;
		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		return FALSE;
	}

	return TRUE;
}


/*
 * start()
 */


static gboolean start(GstBaseTransform *trans)
{
	GSTLALDemodulate *element = GSTLAL_DEMODULATE(trans);

	element->t0 = GST_CLOCK_TIME_NONE;
	element->offset0 = GST_BUFFER_OFFSET_NONE;
	element->next_in_offset = GST_BUFFER_OFFSET_NONE;
	element->next_out_offset = GST_BUFFER_OFFSET_NONE;
	element->need_discont = TRUE;

	return TRUE;
}


/*
 * transform()
 */


static GstFlowReturn transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf)
{
	GSTLALDemodulate *element = GSTLAL_DEMODULATE(trans);
	GstMapInfo inmap, outmap;
	GstFlowReturn result = GST_FLOW_OK;

	/*
	 * check for discontinuity
	 */

	if(G_UNLIKELY(GST_BUFFER_IS_DISCONT(inbuf) || GST_BUFFER_OFFSET(inbuf) != element->next_in_offset || !GST_CLOCK_TIME_IS_VALID(element->t0))) {
		element->t0 = GST_BUFFER_TIMESTAMP(inbuf);
		element->offset0 = element->next_out_offset = GST_BUFFER_OFFSET(inbuf);
		element->need_discont = TRUE;
	}
	element->next_in_offset = GST_BUFFER_OFFSET_END(inbuf);

	/*
	 * process buffer
	 */

	if(!GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP)) {

		/*
		 * input is not gap.
		 */

		gst_buffer_map(inbuf, &inmap, GST_MAP_READ);
		gst_buffer_map(outbuf, &outmap, GST_MAP_WRITE);
		guint64 t_reduced = GST_BUFFER_TIMESTAMP(inbuf) - element->t0;
		demodulate(inmap.data, inmap.size, outmap.data, outmap.size, t_reduced, element->rate, element->unit_size, element->line_frequency);
		set_metadata(element, outbuf, outmap.size / (2 * element->unit_size), FALSE);
		gst_buffer_unmap(outbuf, &outmap);
		gst_buffer_unmap(inbuf, &inmap);
	} else {

		/*
		 * input is gap.
		 */

		GST_BUFFER_FLAG_SET(outbuf, GST_BUFFER_FLAG_GAP);
		gst_buffer_map(outbuf, &outmap, GST_MAP_WRITE);
		memset(outmap.data, 0, outmap.size);
		set_metadata(element, outbuf, outmap.size / (2 * element->unit_size), TRUE);
		gst_buffer_unmap(outbuf, &outmap);
	}

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
	ARG_LINE_FREQUENCY = 1,
};


static void set_property(GObject *object, enum property prop_id, const GValue *value, GParamSpec *pspec)
{
	GSTLALDemodulate *element = GSTLAL_DEMODULATE(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_LINE_FREQUENCY:
		element->line_frequency = g_value_get_double(value);
		break;
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, enum property prop_id, GValue *value, GParamSpec *pspec)
{
	GSTLALDemodulate *element = GSTLAL_DEMODULATE(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_LINE_FREQUENCY:
		g_value_set_double(value, element->line_frequency);
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


static void gstlal_demodulate_class_init(GSTLALDemodulateClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(klass);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);

	transform_class->transform_caps = GST_DEBUG_FUNCPTR(transform_caps);
	transform_class->get_unit_size = GST_DEBUG_FUNCPTR(get_unit_size);
	transform_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
	transform_class->transform_size = GST_DEBUG_FUNCPTR(transform_size);
	transform_class->start = GST_DEBUG_FUNCPTR(start);
	transform_class->transform = GST_DEBUG_FUNCPTR(transform);

	gst_element_class_set_details_simple(element_class,
		"Demodulate",
		"Filter/Audio",
		"Multiplies incoming float stream by exp(-i * 2 * pi * line_frequency * t)",
		"Aaron Viets <aaron.viets@ligo.org>"
	);

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));

	g_object_class_install_property(
		gobject_class,
		ARG_LINE_FREQUENCY,
		g_param_spec_double(
			"line-frequency",
			"Calibration line frequency",
			"The frequency of the calibration line corresponding to the calibration\n\t\t\t"
			"factor 'kappa' we wish to extract from incoming stream",
			0, G_MAXDOUBLE, 300.,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
}


/*
 * init()
 */


static void gstlal_demodulate_init(GSTLALDemodulate *element)
{
	element->rate = 0;
	element->unit_size = 0;
	gst_base_transform_set_qos_enabled(GST_BASE_TRANSFORM(element), TRUE);
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(element), TRUE);
}

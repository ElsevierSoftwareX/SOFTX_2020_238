/*
 * Copyright (C) 2015, 2016  Madeline Wade <madeline.wade@ligo.org>, Aaron Viets <aaron.viets@ligo.org>
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation; either version 2 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */


/*
 * =============================================================================
 *
 *				 Preamble
 *
 * =============================================================================
 */


/*
 * stuff from C
 */


#include <string.h>
#include <complex.h>


/*
 * stuff from gobject/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/audio/audio.h>
#include <gst/base/gstbasetransform.h>
#include <gstlal/gstlal.h>
#include <gstlal_resample.h>


/*
 * ============================================================================
 *
 *				 Utilities
 *
 * ============================================================================
 */


#define DEFINE_CONST_UPSAMPLE_FUNC(size) \
static void const_upsample_ ## size(const gint ## size *src, gint ## size *dst, guint64 src_size, guint cadence) \
{ \
	const gint ## size *src_end; \
	guint i; \
 \
	for(src_end = src + src_size; src < src_end; src++) { \
		for(i = 0; i < cadence; i++, dst++) \
			*dst = *src; \
	} \
}

DEFINE_CONST_UPSAMPLE_FUNC(8)
DEFINE_CONST_UPSAMPLE_FUNC(16)
DEFINE_CONST_UPSAMPLE_FUNC(32)
DEFINE_CONST_UPSAMPLE_FUNC(64)


static void const_upsample_other(const gint8 *src, gint8 *dst, guint64 src_size, gint unit_size, guint cadence)
{
	const gint8 *src_end;
	guint i;

	for(src_end = src + src_size * unit_size; src < src_end; src += unit_size) {
		for(i = 0; i < cadence; i++, dst += unit_size)
			memcpy(dst, src, unit_size);
	}
}


static void resample(const void *src, guint64 src_size, void *dst, guint64 dst_size, gstlal_resample_data_type data_type, gint unit_size, guint cadence, guint inv_cadence, guint polynomial_order, double *remaining_samples, double *dxdt0)
{
	g_assert_cmpuint(src_size % unit_size, ==, 0);
	g_assert_cmpuint(dst_size % unit_size, ==, 0);
	g_assert(src_size * cadence == dst_size || dst_size * inv_cadence == src_size);
	g_assert(cadence > 1 || inv_cadence > 1)

	src_size /= unit_size;
	dst_size /= unit_size;

	if(cadence > 1 && polynomial_order == 0) {
		switch(unit_size) {
		case 1:
			const_upsample_8(src, dst, src_size, cadence);
			break;
		case 2:
			const_upsample_16(src, dst, src_size, cadence);
			break;
		case 4:
			const_upsample_32(src, dst, src_size, cadence);
			break;
		case 8:
			const_upsample_64(src, dst, src_size, cadence);
			break;
		default:
			const_upsample_other(src, dst, src_size, unit_size, cadence);
			break;
		}

	} else if(cadence > 1 && polynomial_order == 1) {
		switch(data_type) {
		case GSTLAL_RESAMPLE_F32:
			linear_upsample_float(src, dst, src_size, unit_size, cadence, (float *) remaining_samples);
			break;
		case GSTLAL_RESAMPLE_F64:
			linear_upsample_double(src, dst, src_size, unit_size, cadence, remaining_samples);
			break;
		case GSTLAL_RESAMPLE_Z64:
			linear_upsample_floatcomplex(src, dst, src_size, unit_size, cadence, (float complex *) remaining_samples);
			break;
		case GSTLAL_RESAMPLE_Z128:
			linear_upsample_doublecomplex(src, dst, src_size, unit_size, cadence, (double complex *) remaining_samples);
			break;
		default:
			g_assert_not_reached();
			break;
		}

	} else if(cadence > 1 && polynomial_order == 2) {
		switch(data_type) {
		case GSTLAL_RESAMPLE_F32:
			quadratic_upsample_float(src, dst, src_size, unit_size, cadence, (float *) remaining_samples);
			break;
		case GSTLAL_RESAMPLE_F64:
			quadratic_upsample_double(src, dst, src_size, unit_size, cadence, remaining_samples);
			break;
		case GSTLAL_RESAMPLE_Z64:
			quadratic_upsample_floatcomplex(src, dst, src_size, unit_size, cadence, (float complex *) remaining_samples);
			break;
		case GSTLAL_RESAMPLE_Z128:
			quadratic_upsample_doublecomplex(src, dst, src_size, unit_size, cadence, (double complex *) remaining_samples);
			break;
		default:
			g_assert_not_reached();
			break;
		}

	} else if(cadence > 1 && polynomial_order == 3) {
		switch(data_type) {
		case GSTLAL_RESAMPLE_F32:
			cubic_upsample_float(src, dst, src_size, unit_size, cadence, (float *) remaining_samples, (float *) dxdt0);
			break;
		case GSTLAL_RESAMPLE_F64:
			cubic_upsample_double(src, dst, src_size, unit_size, cadence, remaining_samples, dxdt0);
			break;
		case GSTLAL_RESAMPLE_Z64:
			cubic_upsample_floatcomplex(src, dst, src_size, unit_size, cadence, (float complex *) remaining_samples, (float complex *) dxdt0);
			break;
		case GSTLAL_RESAMPLE_Z128:
			cubic_upsample_doublecomplex(src, dst, src_size, unit_size, cadence, (double complex *) remaining_samples, (double complex *) dxdt0);
			break;
		default:
			g_assert_not_reached();
			break;
		}

	} else if(inv_cadence > 1 && polynomial_order == 0) {

	} else if(inv_cadence > 1 && polynomial_order > 0) {

	} else
		g_assert_not_reached();
}


/*
 * set the metadata on an output buffer
 */


static void set_metadata(GSTLALResample *element, GstBuffer *buf, guint64 outsamples, gboolean gap)
{
	GST_BUFFER_OFFSET(buf) = element->next_out_offset;
	element->next_out_offset += outsamples;
	GST_BUFFER_OFFSET_END(buf) = element->next_out_offset;
	GST_BUFFER_PTS(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET(buf) - element->offset0, GST_SECOND, element->rate_out);
	GST_BUFFER_DURATION(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET_END(buf) - element->offset0, GST_SECOND, element->rate_out) - GST_BUFFER_PTS(buf);
	if(G_UNLIKELY(element->need_discont)) {
		GST_BUFFER_FLAG_SET(buf, GST_BUFFER_FLAG_DISCONT);
		element->need_discont = FALSE;
	}
	if(gap || element->need_gap) {
		GST_BUFFER_FLAG_SET(buf, GST_BUFFER_FLAG_GAP);
		if(outsamples > 0)
			element->need_gap = FALSE;
	}
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


#define CAPS \
	"audio/x-raw, " \
	"rate = (int) [1, MAX], " \
	"channels = (int) 1, " \
	"format = (string) {"GST_AUDIO_NE(F32)", "GST_AUDIO_NE(F64)", "GST_AUDIO_NE(Z64)", "GST_AUDIO_NE(Z128)"}, " \
	"layout = (string) interleaved, " \
	"channel-mask = (bitmask) 0"


static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SINK_NAME,
	GST_PAD_SINK,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(CAPS)
);


static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SRC_NAME,
	GST_PAD_SRC,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(CAPS)
);


G_DEFINE_TYPE(
	GSTLALResample,
	gstlal_resample,
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
	/*
	 * It seems that the function gst_audio_info_from_caps() does not work for gstlal's complex formats.
	 * Therefore, a different method is used below to parse the caps.
	 */
	const gchar *format;
	static char *formats[] = {"F32LE", "F32BE", "F64LE", "F64BE", "Z64LE", "Z64BE", "Z128LE", "Z128BE"};
	gint sizes[] = {4, 4, 8, 8, 8, 8, 16, 16};

	GstStructure *str = gst_caps_get_structure(caps, 0);
	g_assert(str);

	if(gst_structure_has_field(str, "format")) {
		format = gst_structure_get_string(str, "format");
	} else {
		GST_ERROR_OBJECT(trans, "No format! Cannot infer unit size.\n");
		return FALSE;
	}
	int test = 0;
	for(unsigned int i = 0; i < sizeof(formats) / sizeof(*formats); i++) {
		if(!strcmp(format, formats[i])) {
			*size = sizes[i];
			test++;
		}
	}
	if(test != 1)
		GST_WARNING_OBJECT(trans, "unit size not properly set");

	return TRUE;
}


/*
 * transform_caps()
 */


static GstCaps *transform_caps(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, GstCaps *filter)
{
	guint n;

	caps = gst_caps_copy(caps);

	switch(direction) {
	case GST_PAD_SRC:
		/*
		 * Source pad's format is the same as sink pad's 
		 * except it can have any sample rate equal to or 
		 * greater than the sink pad's. (Really needs to be 
		 * an integer multiple, actually, but that requirement 
		 * is not enforced here).
		 */

		for(n = 0; n < gst_caps_get_size(caps); n++) {
			GstStructure *s = gst_caps_get_structure(caps, n);
			const GValue *v = gst_structure_get_value(s, "rate");

			if(GST_VALUE_HOLDS_INT_RANGE(v)) {
				if(gst_value_get_int_range_max(v) == 1)
					gst_structure_set(s, "rate", G_TYPE_INT, 1, NULL);
				else
					gst_structure_set(s, "rate", GST_TYPE_INT_RANGE, 1, gst_value_get_int_range_max(v), NULL);
			} else if(G_VALUE_HOLDS_INT(v)) {
				if(g_value_get_int(v) == 1)
					gst_structure_set(s, "rate", G_TYPE_INT, 1, NULL);
				else
					gst_structure_set(s, "rate", GST_TYPE_INT_RANGE, 1, g_value_get_int(v), NULL);
			} else
				GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid type for rate in caps"));
		}
		break;

	case GST_PAD_SINK:
		/*
		 * Source pad's format is the same as sink pad's 
		 * except it can have any sample rate equal to or 
		 * greater than the sink pad's. (Really needs to be 
		 * an integer multiple, actually, but that requirement 
		 * is not enforced here).
		 */

		for(n = 0; n < gst_caps_get_size(caps); n++) {
			GstStructure *s = gst_caps_get_structure(caps, n);
			const GValue *v = gst_structure_get_value(s, "rate");

			if(GST_VALUE_HOLDS_INT_RANGE(v))
				gst_structure_set(s, "rate", GST_TYPE_INT_RANGE, gst_value_get_int_range_min(v), G_MAXINT, NULL);
			else if(G_VALUE_HOLDS_INT(v))
				gst_structure_set(s, "rate", GST_TYPE_INT_RANGE, g_value_get_int(v), G_MAXINT, NULL);
			else
				GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid type for rate in caps"));
		}
		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		gst_caps_unref(caps);
		return GST_CAPS_NONE;

	default:
		g_assert_not_reached();
	}

	return caps;
}


/*
 * set_caps()
 */


static gboolean set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps)
{
	GSTLALResample *element = GSTLAL_RESAMPLE(trans);
	gboolean success = TRUE;
	gint rate_in, rate_out;
	gsize unit_size;

	/*
	 * parse the caps
	 */

	success &= get_unit_size(trans, incaps, &unit_size);
	GstStructure *str = gst_caps_get_structure(incaps, 0);
	const gchar *name = gst_structure_get_string(str, "format");
	success &= (name != NULL);
	success &= gst_structure_get_int(str, "rate", &rate_in);
	success &= gst_structure_get_int(gst_caps_get_structure(outcaps, 0), "rate", &rate_out);
	if(!success)
		GST_ERROR_OBJECT(element, "unable to parse caps.  input caps = %" GST_PTR_FORMAT " output caps = %" GST_PTR_FORMAT, incaps, outcaps);

	/* require the output rate to be an integer multiple or divisor of the input rate */
	success &= (rate_out % rate_in) || (rate_in % rate_out)
	if(!((rate_out % rate_in) || (rate_in % rate_out)))
		GST_ERROR_OBJECT(element, "output rate is not an integer multiple or divisor of input rate.  input caps = %" GST_PTR_FORMAT " output caps = %" GST_PTR_FORMAT, incaps, outcaps);

	/*
	 * record stream parameters
	 */

	if(success) {
		if(!strcmp(name, GST_AUDIO_NE(F32))) {
			element->data_type = GSTLAL_RESAMPLE_F32;
			g_assert_cmpuint(unit_size, ==, 4);
		} else if(!strcmp(name, GST_AUDIO_NE(F64))) {
			element->data_type = GSTLAL_RESAMPLE_F64;
			g_assert_cmpuint(unit_size, ==, 8);
		} else if(!strcmp(name, GST_AUDIO_NE(Z64))) {
			element->data_type = GSTLAL_RESAMPLE_Z64;
			g_assert_cmpuint(unit_size, ==, 8);
		} else if(!strcmp(name, GST_AUDIO_NE(Z128))) {
			element->data_type = GSTLAL_RESAMPLE_Z128;
			g_assert_cmpuint(unit_size, ==, 16);
		} else
			g_assert_not_reached();

		element->rate_in = rate_in;
		element->rate_out = rate_out;
		element->unit_size = unit_size;
	}

	return success;
}


/*
 * transform_size()
 */


static gboolean transform_size(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, gsize size, GstCaps *othercaps, gsize *othersize)
{
	GSTLALResample *element = GSTLAL_RESAMPLE(trans);
	guint cadence = element->rate_out / element->rate_in;
	guint inv_cadence = element->rate_in / element->rate_out;
	g_assert(inv_cadence > 1 || cadence > 1)
	/* input and output unit sizes are the same */
	gsize unit_size;

	if(!get_unit_size(trans, caps, &unit_size))
		return FALSE;

	/*
	 * convert byte count to samples
	 */

	if(G_UNLIKELY(size % unit_size)) {
		GST_DEBUG_OBJECT(element, "buffer size %" G_GSIZE_FORMAT " is not a multiple of %" G_GSIZE_FORMAT, size, unit_size);
		return FALSE;
	}
	size /= unit_size;

	switch(direction) {
	case GST_PAD_SRC:
		/*
		 * compute samples needed on sink pad from sample count on source pad.
		 * size = # of samples needed on source pad
		 * cadence = # of output samples per input sample
		 * inv_cadence = # of input samples per output sample
		 */

		if(inv_cadence > 1)
			*othersize = size * inv_cadence;
		else
			*othersize = size / cadence;
		break;

	case GST_PAD_SINK:
		/*
		 * compute samples to be produced on source pad from sample
		 * count available on sink pad.
		 * size = # of samples available on sink pad
		 * cadence = # of output samples per input sample
		 * inv_cadence = # of input samples per output sample
		 */

		if(cadence > 1)
			*othersize = size * cadence;
		else {
			*othersize = size / inv_cadence;
			if(size % inv_cadence)
				element->need_buffer_resize = TRUE
		}
		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		return FALSE;

	default:
		g_assert_not_reached();
	}

	/*
	 * convert sample count to byte count
	 */

	*othersize *= unit_size;

	return TRUE;
}


/*
 * start()
 */


static gboolean start(GstBaseTransform *trans)
{
	GSTLALResample *element = GSTLAL_RESAMPLE(trans);

	element->t0 = GST_CLOCK_TIME_NONE;
	element->offset0 = GST_BUFFER_OFFSET_NONE;
	element->next_in_offset = GST_BUFFER_OFFSET_NONE;
	element->next_out_offset = GST_BUFFER_OFFSET_NONE;
	element->need_discont = TRUE;
	element->need_gap = FALSE;
	element->dxdt0 = 0.0;
	element->clipped_samples = (element->rate_out > element->rate_in) * (element->polynomial_order + 1) / 2;
	element->latency = (guint) ((1000000000.0 * element->clipped_samples) / element->rate_in + 0.5);

	return TRUE;
}


/*
 * transform()
 */


static GstFlowReturn transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf)
{
	GSTLALResample *element = GSTLAL_RESAMPLE(trans);
	GstMapInfo inmap, outmap;
	GstFlowReturn result = GST_FLOW_OK;

	/*
	 * check for discontinuity
	 */

	if(G_UNLIKELY(GST_BUFFER_IS_DISCONT(inbuf) || GST_BUFFER_OFFSET(inbuf) != element->next_in_offset || !GST_CLOCK_TIME_IS_VALID(element->t0))) {
		element->t0 = GST_BUFFER_PTS(inbuf);
		element->offset0 = element->next_out_offset = gst_util_uint64_scale_ceil(GST_BUFFER_OFFSET(inbuf), element->rate_out, element->rate_in);
		element->need_discont = TRUE;
		if(element->polynomial_order > 0)
			element->need_buffer_resize = TRUE;
	}
	element->next_in_offset = GST_BUFFER_OFFSET_END(inbuf);

	/*
	 * adjust output buffer size if necessary
	 */ 

	if(element->need_buffer_resize) {
		gssize outbuf_size;
		if(element->rate_out > element->rate_in)
			outbuf_size = gst_buffer_get_size(outbuf) - element->unit_size * ((element->polynomial_order + 1) / 2);
		else if(element->rate_in > element->rate_out) {
			guint inv_cadence = element->rate_in / element->rate_out;
			guint inbuf_samples = gst_buffer_get_size(inbuf) / element->unit_size;
			/* leading_samples is the numner of input samples that come before the first timestamp that is a multiple of the output sampling period */
			guint leading_samples = gst_util_uint64_scale_int_round(GST_BUFFER_PTS(inbuf), element->rate_in, 1000000000) % inv_cadence;
			if(leading_samples != 0)
				leading_samples = inv_cadence - leading_samples;
			outbuf_size = ((inbuf_samples - leading_samples + inv_cadence - 1) / inv_cadence) * element->unit_size;
			if(element->polynomial_order > 0) {
				/* trailing samples are input samples that come after the last timestamp that is a multiple of the output sampling period */
				guint trailing_samples = (inbuf_samples - leading_samples - 1) % inv_cadence;
				outbuf_size -= element->unit_size * (1 - (trailing_samples + 1) / ((inv_cadence + 1) / 2));
				if(!element->need_discont)
					outbuf_size += element->unit_size * (leading_samples / ((inv_cadence + 2) / 2));
			} else
				g_assert_not_reached();
		}
		gst_buffer_set_size(outbuf, outbuf_size);
		element->need_buffer_resize = FALSE;
	}

	/*
	 * process buffer
	 */

	if(!GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP)) {

		/*
		 * input is not 0s.
		 */

		gst_buffer_map(inbuf, &inmap, GST_MAP_READ);
		gst_buffer_map(outbuf, &outmap, GST_MAP_WRITE);
		resample(inmap.data, inmap.size, outmap.data, outmap.size, element->unit_size, element->data_type, element->rate_out / element->rate_in, element->rate_in / element->rate_out, element->polynomial_order, element->remaining_samples, &element->dxdt0);
		set_metadata(element, outbuf, outmap.size / element->unit_size, FALSE);
		gst_buffer_unmap(outbuf, &outmap);
		gst_buffer_unmap(inbuf, &inmap);
	} else {
		/*
		 * input is 0s.
		 */

		GST_BUFFER_FLAG_SET(outbuf, GST_BUFFER_FLAG_GAP);
		gst_buffer_map(outbuf, &outmap, GST_MAP_WRITE);
		memset(outmap.data, 0, outmap.size);
		set_metadata(element, outbuf, outmap.size / element->unit_size, TRUE);
		if(outmap.size / element->unit_size == 0)
			element->need_gap = TRUE;
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


static void set_property(GObject *object, enum property prop_id, const GValue *value, GParamSpec *pspec)
{
	GSTLALResample *element = GSTLAL_RESAMPLE(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_POLYNOMIAL_ORDER:
		element->polynomial_order = g_value_get_uint(value);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, enum property prop_id, GValue *value, GParamSpec *pspec)
{
	GSTLALResample *element = GSTLAL_RESAMPLE(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_POLYNOMIAL_ORDER:
		g_value_set_uint(value, element->polynomial_order);
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


static void gstlal_resample_class_init(GSTLALResampleClass *klass)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(klass);

	transform_class->transform_caps = GST_DEBUG_FUNCPTR(transform_caps);
	transform_class->transform_size = GST_DEBUG_FUNCPTR(transform_size);
	transform_class->get_unit_size = GST_DEBUG_FUNCPTR(get_unit_size);
	transform_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
	transform_class->start = GST_DEBUG_FUNCPTR(start);
	transform_class->transform = GST_DEBUG_FUNCPTR(transform);
	transform_class->passthrough_on_same_caps = TRUE;

	gst_element_class_set_details_simple(element_class,
		"Resamples a data stream",
		"Filter/Audio",
		"Resamples a stream with adjustable (or no) interpolation.",
		"Madeline Wade <madeline.wade@ligo.org>, Aaron Viets <aaron.viets@ligo.org>"
	);

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));

	g_object_class_install_property(
		gobject_class,
		ARG_REQUIRED_ON,
		g_param_spec_uint(
			"polynomial-order",
			"Interpolating Polynomial Order",
			"When upsampling, this is the order of the polynomial used to interpolate between\n\t\t\t"
			"input samples. 0 yields a constant upsampling, 1 is linear interpolation, 3 is a\n\t\t\t"
			"cubic spline. When downsampling, this determines whether we just pick every nth\n\t\t\t"
			"sample, or take an average of n input samples surrounding output sample timestamps.",
			0, 3, 0,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
}


/*
 * init()
 */


static void gstlal_resample_init(GSTLALResample *element)
{
	element->rate_in = 0;
	element->rate_out = 0;
	element->unit_size = 0;
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(element), TRUE);
}

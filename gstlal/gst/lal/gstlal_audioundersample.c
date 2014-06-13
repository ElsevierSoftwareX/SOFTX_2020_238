/*
 * Copyright (C) 2011--2013 Kipp Cannon <kipp.cannon@ligo.org>
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


/**
 * SECTION:gstlal_audioundersample
 * @short_description:  Undersample an audio stream.
 *
 * This element implements an undersampling transform on time series data.
 * For more information about undersampling, see <ulink
 * url="https://en.wikipedia.org/wiki/Undersampling">https://en.wikipedia.org/wiki/Undersampling</ulink>.
 * Note that this element only performs the undersampling itself, not the
 * band-pass pre-filtering that is usually performed as part of the
 * transform.  This element can only generate output streams whose sample
 * rates are integer divisors of the input stream.  To achieve other sample
 * rates, precede this element with an audioresample element to resample
 * the time series to a rate that is an integer multiple of the final
 * desired sample rate.
 *
 * Example (assumes 44.1 kHz source material):
 *
 * $ gst-launch filesrc location="song.mp3" ! decodebin ! lal_audioundersample ! audio/x-raw-int, rate=7350 ! audioresample ! autoaudiosink
 */


/*
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


/*
 * stuff from C
 */


#include <string.h>


/*
 *  stuff from gobject/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <gstlal/gstlal.h>
#include <gstlal_audioundersample.h>


/*
 * ============================================================================
 *
 *                                 Utilities
 *
 * ============================================================================
 */


#define DEFINE_UNDERSAMPLE_FUNC(size) \
static guint64 undersample_ ## size(const gint ## size *src, gint ## size *dst, guint64 dst_size, guint cadence) \
{ \
	const gint ## size *dst_end; \
 \
	for(dst_end = dst + dst_size; dst < dst_end; src += cadence, dst++) \
		*dst = *src; \
 \
	return dst_size; \
}

DEFINE_UNDERSAMPLE_FUNC(8)
DEFINE_UNDERSAMPLE_FUNC(16)
DEFINE_UNDERSAMPLE_FUNC(32)
DEFINE_UNDERSAMPLE_FUNC(64)


static guint64 undersample_other(const gint8 *src, gint8 *dst, guint64 dst_size, gint unit_size, guint cadence)
{
	const gint8 *dst_end;

	cadence *= unit_size;
	for(dst_end = dst + dst_size * unit_size; dst < dst_end; src += cadence, dst += unit_size)
		memcpy(dst, src, unit_size);

	return dst_size;
}


static guint64 undersample(const void *src, guint64 src_size, void *dst, guint64 dst_size, gint unit_size, guint cadence, guint64 *remainder)
{
	g_assert_cmpuint(src_size % unit_size, ==, 0);
	g_assert_cmpuint(dst_size % unit_size, ==, 0);

	src_size /= unit_size;
	dst_size /= unit_size;

	if(src_size <= *remainder) {
		*remainder -= src_size;
		return 0;
	}

	src += *remainder * unit_size;
	src_size -= *remainder;
	*remainder = src_size % cadence ? cadence - src_size % cadence : 0;

	g_assert_cmpuint(dst_size * cadence, ==, src_size + *remainder);

	switch(unit_size) {
	case 1:
		return undersample_8(src, dst, dst_size, cadence);

	case 2:
		return undersample_16(src, dst, dst_size, cadence);

	case 4:
		return undersample_32(src, dst, dst_size, cadence);

	case 8:
		return undersample_64(src, dst, dst_size, cadence);

	default:
		return undersample_other(src, dst, dst_size, unit_size, cadence);
	}
}


/*
 * set the metadata on an output buffer
 */


static void set_metadata(GSTLALAudioUnderSample *element, GstBuffer *buf, guint64 outsamples, gboolean gap)
{
	GST_BUFFER_SIZE(buf) = outsamples * element->unit_size;
	GST_BUFFER_OFFSET(buf) = element->next_out_offset;
	element->next_out_offset += outsamples;
	GST_BUFFER_OFFSET_END(buf) = element->next_out_offset;
	GST_BUFFER_TIMESTAMP(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET(buf) - element->offset0, GST_SECOND, element->rate_out);
	GST_BUFFER_DURATION(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET_END(buf) - element->offset0, GST_SECOND, element->rate_out) - GST_BUFFER_TIMESTAMP(buf);
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
		"channels = (int) [1, MAX], " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) {8, 16, 32, 64}, " \
		"signed = (boolean) {true, false}; " \
		"audio/x-raw-float, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) [1, MAX], " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) {32, 64};" \
		"audio/x-raw-complex, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) [1, MAX], " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) {64, 128}"
	)
);


static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SRC_NAME,
	GST_PAD_SRC,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		"audio/x-raw-int, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) [1, MAX], " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) {8, 16, 32, 64}, " \
		"signed = (boolean) {true, false}; " \
		"audio/x-raw-float, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) [1, MAX], " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) {32, 64};" \
		"audio/x-raw-complex, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) [1, MAX], " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) {64, 128}"
	)
);


GST_BOILERPLATE(
	GSTLALAudioUnderSample,
	gstlal_audioundersample,
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


/*
 * get_unit_size()
 */


static gboolean get_unit_size(GstBaseTransform *trans, GstCaps *caps, guint *size)
{
	GstStructure *str;
	gint channels;
	gint width;
	gboolean success = TRUE;

	str = gst_caps_get_structure(caps, 0);
	success &= gst_structure_get_int(str, "channels", &channels);
	success &= gst_structure_get_int(str, "width", &width);

	if(success)
		*size = width / 8 * channels;
	else
		GST_WARNING_OBJECT(trans, "unable to parse caps %" GST_PTR_FORMAT, caps);

	return success;
}


/*
 * transform_caps()
 */


static GstCaps *transform_caps(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps)
{
	guint n;

	caps = gst_caps_copy(caps);

	switch(direction) {
	case GST_PAD_SRC:
		/*
		 * sink pad's format is the same as the source pad's except
		 * it can have any sample rate equal to or greater than the
		 * source pad's.  actually it must be an integer multiple
		 * of the source pad's rate, but there are potentially many
		 * thousands of allowed values and it is impractical to
		 * exhaustively list them as a set.
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

	case GST_PAD_SINK:
		/*
		 * source pad's format is the same as the sink pad's except
		 * it can have any sample rate equal to or less than the
		 * sink pad's.  actually it must be an integer divisor of
		 * the sink pad's rate, but there are potentially many
		 * thousands of allowed values and it is impractical to
		 * exhaustively list them as a set.
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

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		gst_caps_unref(caps);
		return GST_CAPS_NONE;
	}

	return caps;
}


/*
 * set_caps()
 */


static gboolean set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps)
{
	GSTLALAudioUnderSample *element = GSTLAL_AUDIOUNDERSAMPLE(trans);
	gint rate_in, rate_out;
	guint unit_size;

	/*
	 * parse the caps
	 */

	if(!get_unit_size(trans, incaps, &unit_size))
		return FALSE;
	if(!gst_structure_get_int(gst_caps_get_structure(incaps, 0), "rate", &rate_in)) {
		GST_DEBUG_OBJECT(element, "unable to parse rate from %" GST_PTR_FORMAT, incaps);
		return FALSE;
	}
	if(!gst_structure_get_int(gst_caps_get_structure(outcaps, 0), "rate", &rate_out)) {
		GST_DEBUG_OBJECT(element, "unable to parse rate from %" GST_PTR_FORMAT, outcaps);
		return FALSE;
	}

	/*
	 * require the input rate to be an integer multiple of the output
	 * rate
	 */

	if(rate_in % rate_out) {
		GST_ERROR_OBJECT(element, "input rate is not an integer multiple of output rate.  input caps = %" GST_PTR_FORMAT " output caps = %" GST_PTR_FORMAT, incaps, outcaps);
		return FALSE;
	}

	/*
	 * record stream parameters
	 */

	element->rate_in = rate_in;
	element->rate_out = rate_out;
	element->unit_size = unit_size;

	return TRUE;
}


/*
 * transform_size()
 */


static gboolean transform_size(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, guint size, GstCaps *othercaps, guint *othersize)
{
	GSTLALAudioUnderSample *element = GSTLAL_AUDIOUNDERSAMPLE(trans);
	gint cadence = element->rate_in / element->rate_out;
	/* input and output unit sizes are the same */
	guint unit_size;

	if(!get_unit_size(trans, caps, &unit_size))
		return FALSE;

	/*
	 * convert byte count to samples
	 */

	if(G_UNLIKELY(size % unit_size)) {
		GST_DEBUG_OBJECT(element, "buffer size %u is not a multiple of %u", size, unit_size);
		return FALSE;
	}
	size /= unit_size;

	switch(direction) {
	case GST_PAD_SRC:
		/*
		 * compute samples required on sink pad to produce
		 * requested sample count on source pad
		 *
		 * size = # of samples requested on source pad
		 *
		 * cadence = # of input samples per output sample
		 *
		 * remainder = how many extra samples of input are needed
		 * before producing an output sample (because the most
		 * recent input buffer ended before a complete cycle)
		 */

		*othersize = size * cadence + element->remainder;
		break;

	case GST_PAD_SINK:
		/*
		 * compute samples to be produced on source pad from sample
		 * count available on sink pad
		 *
		 * size = # of samples available on sink pad
		 *
		 * reminder = how many have already been spoken for because
		 * the most recent input buffer ended before a complete
		 * cycle
		 *
		 * cadence = # of input samples per output sample
		 *
		 * adding cadence-1 implements the ceiling function:  when
		 * remainder is 0, the first input sample can produce 1
		 * output sample, not 0
		 */

		if(size >= element->remainder)
			*othersize = (size - element->remainder + cadence - 1) / cadence;
		else
			*othersize = 0;
		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		return FALSE;
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
	GSTLALAudioUnderSample *element = GSTLAL_AUDIOUNDERSAMPLE(trans);

	element->t0 = GST_CLOCK_TIME_NONE;
	element->offset0 = GST_BUFFER_OFFSET_NONE;
	element->next_in_offset = GST_BUFFER_OFFSET_NONE;
	element->next_out_offset = GST_BUFFER_OFFSET_NONE;
	element->need_discont = TRUE;

	element->remainder = 0;

	return TRUE;
}


/*
 * transform()
 */


static GstFlowReturn transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf)
{
	GSTLALAudioUnderSample *element = GSTLAL_AUDIOUNDERSAMPLE(trans);
	GstFlowReturn result = GST_FLOW_OK;

	/*
	 * check for discontinuity
	 */

	if(G_UNLIKELY(GST_BUFFER_IS_DISCONT(inbuf) || GST_BUFFER_OFFSET(inbuf) != element->next_in_offset || !GST_CLOCK_TIME_IS_VALID(element->t0))) {
		element->t0 = GST_BUFFER_TIMESTAMP(inbuf);
		element->offset0 = element->next_out_offset = gst_util_uint64_scale_ceil(GST_BUFFER_OFFSET(inbuf), element->rate_out, element->rate_in);
		element->need_discont = TRUE;
		element->remainder = 0;
	}
	element->next_in_offset = GST_BUFFER_OFFSET_END(inbuf);

	/*
	 * process buffer
	 */

	if(!GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP)) {
		guint64 output_length;

		/*
		 * input is not 0s.
		 */

		output_length = undersample((void *) GST_BUFFER_DATA(inbuf), GST_BUFFER_SIZE(inbuf), (void *) GST_BUFFER_DATA(outbuf), GST_BUFFER_SIZE(outbuf), element->unit_size, element->rate_in / element->rate_out, &element->remainder);
		set_metadata(element, outbuf, output_length, FALSE);
	} else {
		/*
		 * input is 0s.
		 */

		GST_BUFFER_FLAG_SET(outbuf, GST_BUFFER_FLAG_GAP);
		memset(GST_BUFFER_DATA(outbuf), 0, GST_BUFFER_SIZE(outbuf));
		set_metadata(element, outbuf, GST_BUFFER_SIZE(outbuf) / element->unit_size, TRUE);
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
 * base_init()
 */


static void gstlal_audioundersample_base_init(gpointer gclass)
{
}


/*
 * class_init()
 */


static void gstlal_audioundersample_class_init(GSTLALAudioUnderSampleClass *klass)
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

	gst_element_class_set_details_simple(element_class, "Undersample", "Filter/Audio", "Undersamples an audio stream.  Undersampling downsamples by taking every n-th sample, with no antialiasing or low-pass filter.  For data confined to a narrow frequency band, this transformation simultaneously downconverts and downsamples the data (otherwise it does weird things).  This element's output sample rate must be an integer divisor of its input sample rate.", "Kipp Cannon <kipp.cannon@ligo.org>");

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));
}


/*
 * init()
 */


static void gstlal_audioundersample_init(GSTLALAudioUnderSample *element, GSTLALAudioUnderSampleClass *klass)
{
	element->rate_in = 0;
	element->rate_out = 0;
	element->unit_size = 0;
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(element), TRUE);
}

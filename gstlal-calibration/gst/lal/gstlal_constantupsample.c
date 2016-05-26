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


/*
 * stuff from gobject/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/audio/audio.h>
#include <gst/base/gstbasetransform.h>
#include <gstlal/gstlal.h>
#include <gstlal_constantupsample.h>


/*
 * ============================================================================
 *
 *				 Utilities
 *
 * ============================================================================
 */


#define DEFINE_UPSAMPLE_FUNC(size) \
static void upsample_ ## size(const gint ## size *src, gint ## size *dst, guint64 src_size, guint cadence) \
{ \
	const gint ## size *src_end; \
	guint i; \
 \
	for(src_end = src + src_size; src < src_end; src++) { \
		for(i = 0; i < cadence; i++, dst++) \
			*dst = *src; \
	} \
}

DEFINE_UPSAMPLE_FUNC(8)
DEFINE_UPSAMPLE_FUNC(16)
DEFINE_UPSAMPLE_FUNC(32)
DEFINE_UPSAMPLE_FUNC(64)


static void upsample_other(const gint8 *src, gint8 *dst, guint64 src_size, gint unit_size, guint cadence)
{
	const gint8 *src_end;
	guint i;

	for(src_end = src + src_size * unit_size; src < src_end; src += unit_size) {
		for(i = 0; i < cadence; i++, dst += unit_size)
			memcpy(dst, src, unit_size);
	}
}


static void upsample(const void *src, guint64 src_size, void *dst, guint64 dst_size, gint unit_size, guint cadence)
{
	g_assert_cmpuint(src_size % unit_size, ==, 0);
	g_assert_cmpuint(dst_size % unit_size, ==, 0);
	g_assert_cmpuint(src_size * cadence, ==, dst_size);

	src_size /= unit_size;
	dst_size /= unit_size;

	switch(unit_size) {
	case 1:
		upsample_8(src, dst, src_size, cadence);
		break;

	case 2:
		upsample_16(src, dst, src_size, cadence);
		break;

	case 4:
		upsample_32(src, dst, src_size, cadence);
		break;

	case 8:
		upsample_64(src, dst, src_size, cadence);
		break;

	default:
		upsample_other(src, dst, src_size, unit_size, cadence);
		break;
	}
}


/*
 * set the metadata on an output buffer
 */


static void set_metadata(GSTLALConstantUpSample *element, GstBuffer *buf, guint64 outsamples, gboolean gap)
{
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
 *			   GStreamer Boiler Plate
 *
 * ============================================================================
 */


#define CAPS \
	"audio/x-raw, " \
	"rate = (int) [1, MAX], " \
	"channels = (int) 1, " \
	"format = (string) " GST_AUDIO_NE(F64) " , " \
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
	GSTLALConstantUpSample,
	gstlal_constantupsample,
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
	gboolean success = TRUE;

	success &= gst_audio_info_from_caps(&info, caps);

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
	GSTLALConstantUpSample *element = GSTLAL_CONSTANTUPSAMPLE(trans);
	gint rate_in, rate_out;
	gsize unit_size;

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
	 * require the output rate to be an integer multiple of the input
	 * rate
	 */

	if(rate_out % rate_in) {
		GST_ERROR_OBJECT(element, "output rate is not an integer multiple of input rate.  input caps = %" GST_PTR_FORMAT " output caps = %" GST_PTR_FORMAT, incaps, outcaps);
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


static gboolean transform_size(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, gsize size, GstCaps *othercaps, gsize *othersize)
{
	GSTLALConstantUpSample *element = GSTLAL_CONSTANTUPSAMPLE(trans);
	element->cadence = element->rate_out / element->rate_in;
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
		 * compute samples to be produced on source pad from sample
		 * count available on sink pad
		 *
		 * size = # of samples available on sink pad
		 *
		 * cadence = # of output samples per input sample
		 */

		if(size >= element->cadence)
			*othersize = size / element->cadence;
		else
			*othersize = 0;
		break;

	case GST_PAD_SINK:
		/*
		 * compute samples to be produced on source pad from sample
		 * count available on sink pad
		 *
		 * size = # of samples available on sink pad
		 *
		 * cadence = # of output samples per input sample
		 */
		*othersize = size * element->cadence;
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
	GSTLALConstantUpSample *element = GSTLAL_CONSTANTUPSAMPLE(trans);

	element->t0 = GST_CLOCK_TIME_NONE;
	element->offset0 = GST_BUFFER_OFFSET_NONE;
	element->next_in_offset = GST_BUFFER_OFFSET_NONE;
	element->next_out_offset = GST_BUFFER_OFFSET_NONE;
	element->need_discont = TRUE;
	element->need_gap = FALSE;

	return TRUE;
}


/*
 * transform()
 */


static GstFlowReturn transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf)
{
	GSTLALConstantUpSample *element = GSTLAL_CONSTANTUPSAMPLE(trans);
	GstMapInfo inmap, outmap;
	GstFlowReturn result = GST_FLOW_OK;

	/*
	 * check for discontinuity
	 */

	if(G_UNLIKELY(GST_BUFFER_IS_DISCONT(inbuf) || GST_BUFFER_OFFSET(inbuf) != element->next_in_offset || !GST_CLOCK_TIME_IS_VALID(element->t0))) {
		element->t0 = GST_BUFFER_TIMESTAMP(inbuf);
		element->offset0 = element->next_out_offset = gst_util_uint64_scale_ceil(GST_BUFFER_OFFSET(inbuf), element->rate_out, element->rate_in);
		element->need_discont = TRUE;
	}
	element->next_in_offset = GST_BUFFER_OFFSET_END(inbuf);

	/*
	 * process buffer
	 */

	if(!GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP)) {

		/*
		 * input is not 0s.
		 */

		gst_buffer_map(inbuf, &inmap, GST_MAP_READ);
		gst_buffer_map(outbuf, &outmap, GST_MAP_WRITE);
		upsample(inmap.data, inmap.size, outmap.data, outmap.size, element->unit_size, element->rate_out / element->rate_in);
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
 * class_init()
 */


static void gstlal_constantupsample_class_init(GSTLALConstantUpSampleClass *klass)
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
		"Upsample stream of constant values",
		"Filter/Audio",
		"Upsamples a stream filling the upsampled samples with the same constant value as the input",
		"Madeline Wade <madeline.wade@ligo.org>, Aaron Viets <aaron.viets@ligo.org>"
	);

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));
}


/*
 * init()
 */


static void gstlal_constantupsample_init(GSTLALConstantUpSample *element)
{
	element->rate_in = 0;
	element->rate_out = 0;
	element->unit_size = 0;
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(element), TRUE);
}

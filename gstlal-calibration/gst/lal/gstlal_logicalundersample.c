/*
 * Copyright (C) 2014, 2016  Madeline Wade, Chris Pankow, Aaron Viets
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
 *			       Preamble
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
#include <gstlal/gstlal_audio_info.h>
#include <gstlal_logicalundersample.h>


/*
 * ============================================================================
 *
 *			       Utilities
 *
 * ============================================================================
 */


#define DEFINE_UNDERSAMPLE_FUNC(sign,width) \
static void logical_op_g ## sign ## width(const g ## sign ## width *src, guint64 src_size, guint ## width *dst, guint64 dst_size, guint cadence, guint64 *remainder, g ## sign ## width *leftover, g ## sign ## width *required_on, guint ## width status_out) \
{ \
	g ## sign ## width inbits[*remainder + src_size]; \
 \
	for(unsigned int i = 0; i < *remainder; i++) \
		inbits[i] = *leftover; \
 \
	for(unsigned int j = *remainder; j < *remainder + src_size; j++, src++) \
		inbits[j] = *src; \
 \
	for(unsigned int k_start = 0; k_start < dst_size; k_start++, dst++) { \
		g ## sign ## width cadence_samples = *required_on; \
		for(unsigned int k = cadence * k_start; k < cadence * (k_start + 1); k++) \
			cadence_samples = cadence_samples & inbits[k]; \
		if(cadence_samples == *required_on) \
			*dst = status_out; \
		else \
			*dst = 0x00; \
	} \
 \
	unsigned int first_leftover_index = src_size + *remainder - ((src_size + *remainder) % cadence); \
	*remainder = (src_size + *remainder) % cadence; \
	if(*remainder != 0) { \
		*leftover = *required_on; \
		for(unsigned int m = first_leftover_index; m < first_leftover_index + *remainder; m++) \
			*leftover = *leftover & inbits[m]; \
	} else \
		*leftover = 0; \
}


DEFINE_UNDERSAMPLE_FUNC(int,32)
DEFINE_UNDERSAMPLE_FUNC(uint,32)


static void undersample(const void *src, guint64 src_size, void *dst, guint64 dst_size, guint unit_size, gboolean sign, guint cadence, guint64 *remainder, void *leftover, void *required_on, guint32 status_out)
{
	g_assert_cmpuint(src_size % unit_size, ==, 0);
	g_assert_cmpuint(dst_size % unit_size, ==, 0);

	dst_size /= unit_size;
	src_size /= unit_size;

	if(sign) {

		switch(unit_size) {
		case 4:
			logical_op_gint32(src, src_size, dst, dst_size, cadence, remainder, leftover, required_on, status_out);
			break;
		default:
			g_assert_not_reached();
		}
	} else {

		switch(unit_size) {
		case 4:
			logical_op_guint32(src, src_size, dst, dst_size, cadence, remainder, leftover, required_on, status_out);
			break;
		default:
			g_assert_not_reached();
		}
	}
}


/*
 * set the metadata on an output buffer
 */


static void set_metadata(GSTLALLogicalUnderSample *element, GstBuffer *buf, guint64 outsamples, gboolean gap)
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


static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SINK_NAME,
	GST_PAD_SINK,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		"audio/x-raw, " \
		"rate = " GST_AUDIO_RATE_RANGE ", " \
		"channels = (int) 1, " \
		"format = (string) {" GST_AUDIO_NE(S32) ", " GST_AUDIO_NE(U32) "}, " \
		"layout = (string) interleaved, " \
		"channel-mask = (bitmask) 0"
	)
);


static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SRC_NAME,
	GST_PAD_SRC,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		"audio/x-raw, " \
		"rate = " GST_AUDIO_RATE_RANGE ", " \
		"channels = (int) 1, " \
		"format = (string) " GST_AUDIO_NE(U32) ", " \
		"layout = (string) interleaved, " \
		"channel-mask = (bitmask) 0"
	)
);


G_DEFINE_TYPE(
	GSTLALLogicalUnderSample,
	gstlal_logicalundersample,
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


static GstCaps *transform_caps(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, GstCaps *filter)
{
        GstCaps *othercaps = NULL;
        guint i;

        if(gst_caps_get_size(caps) > 1)
                GST_WARNING_OBJECT(trans, "not yet smart enough to transform complex formats");

        switch(direction) {
        case GST_PAD_SRC:
                /*
		 * Sink pad's format is the same as the source pad's except
		 * it can have any sample rate equal to or greater than the
		 * source pad's. It actually must be an integer multiple
		 * of the source pad's rate, but that requirement is not
		 * enforced here.
                 * FIXME:  this doesn't work out all the allowed
                 * permutations, it just takes the rate from the
                 * first structure on the source pad and copies it into all
                 * the structures on the sink pad
                 */

                othercaps = gst_caps_normalize(gst_pad_get_pad_template_caps(GST_BASE_TRANSFORM_SINK_PAD(trans)));
                for(i = 0; i < gst_caps_get_size(othercaps); i++) {

                        const GValue *v = gst_structure_get_value(gst_caps_get_structure(caps, 0), "rate");
                        g_assert(v);

			GstStructure *otherstr = gst_caps_get_structure(othercaps, i);

                        if(GST_VALUE_HOLDS_INT_RANGE(v))
                                gst_structure_set(otherstr, "rate", GST_TYPE_INT_RANGE, gst_value_get_int_range_min(v), G_MAXINT, NULL);
                        else if(G_VALUE_HOLDS_INT(v))
                                gst_structure_set(otherstr, "rate", GST_TYPE_INT_RANGE, g_value_get_int(v), G_MAXINT, NULL);
                        else
                                GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid type for rate in caps"));
		}
                break;

        case GST_PAD_SINK:
                /*
		 * Sink pad's format is the same as the source pad's except
		 * it can have any sample rate equal to or greater than the
		 * source pad's. It actually must be an integer multiple
		 * of the source pad's rate, but that requirement is not
		 * enforced here.
                 */

                othercaps = gst_caps_normalize(gst_caps_copy(caps));
                for(i = 0; i < gst_caps_get_size(othercaps); i++) {

                        const GValue *v = gst_structure_get_value(gst_caps_get_structure(caps, 0), "rate");
                        g_assert(v);

			GstStructure *otherstr = gst_caps_get_structure(othercaps, i);

                        gst_structure_set(otherstr, "channels", G_TYPE_INT, 1, NULL);
                        gst_structure_set(otherstr, "format", G_TYPE_STRING, GST_AUDIO_NE(U32), NULL);

                        if(GST_VALUE_HOLDS_INT_RANGE(v)) {
                                if(gst_value_get_int_range_max(v) == 1)
                                        gst_structure_set(otherstr, "rate", G_TYPE_INT, 1, NULL);
                                else
                                        gst_structure_set(otherstr, "rate", GST_TYPE_INT_RANGE, 1, gst_value_get_int_range_max(v), NULL);
                        } else if(G_VALUE_HOLDS_INT(v)) {
                                if(g_value_get_int(v) == 1)
                                        gst_structure_set(otherstr, "rate", G_TYPE_INT, 1, NULL);
                                else
                                        gst_structure_set(otherstr, "rate", GST_TYPE_INT_RANGE, 1, g_value_get_int(v), NULL);
                        } else {
                                GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid type for rate in caps"));
                        }

                }
                break;

        case GST_PAD_UNKNOWN:
                GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
                gst_caps_unref(caps);
                return GST_CAPS_NONE;
        }

        othercaps = gst_caps_simplify(othercaps);

        if(filter) {
                GstCaps *intersection = gst_caps_intersect(othercaps, filter);
                gst_caps_unref(othercaps);
                othercaps = intersection;
        }

        return othercaps;
}


/*
 * set_caps()
 */


static gboolean set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps)
{
	GSTLALLogicalUnderSample *element = GSTLAL_LOGICALUNDERSAMPLE(trans);
	gint rate_in, rate_out;
	gsize unit_size;
	const gchar *format;
	static char *formats[] = {"S32LE", "S32BE", "U32LE", "U32BE"};
	gboolean sign[] = {TRUE, TRUE, FALSE, FALSE};

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

	/* Check the incaps to see if it contains S32, etc. */
	GstStructure *str = gst_caps_get_structure(incaps, 0);
	g_assert(str);

	if(gst_structure_has_field(str, "format")) {
		format = gst_structure_get_string(str, "format");
	} else {
		GST_ERROR_OBJECT(element, "No format! Cannot set element caps.\n");
		return FALSE;
	}
	int test = 0;
	for(unsigned int i = 0; i < sizeof(formats) / sizeof(*formats); i++) {
		if(!strcmp(format, formats[i])) {
			element->sign = sign[i];
			test++;
		}			
	}
	if(test != 1)
		GST_WARNING_OBJECT(element, "element->sign not properly set");

	return TRUE;
}


/*
 * transform_size()
 */


static gboolean transform_size(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, gsize size, GstCaps *othercaps, gsize *othersize)
{
	GSTLALLogicalUnderSample *element = GSTLAL_LOGICALUNDERSAMPLE(trans);

	element->cadence = element->rate_in / element->rate_out;

	/* input and output unit sizes are the same */
	gsize unit_size;

	if(!get_unit_size(trans, caps, &unit_size)) {
		GST_DEBUG_OBJECT(element, "function 'get_unit_size' failed");
		return FALSE;
	}

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
		 * compute othersize = # of samples required on sink
		 * pad to produce requested sample count on source pad
		 *
		 * size = # of samples requested on source pad
		 *
		 * cadence = # of input samples per output sample
		 *
		 * remainder = how many extra samples of input are
		 * present on the sink pad from the previous input
		 * buffer, which ended before a complete cycle
		 */

		*othersize = size * element->cadence - element->remainder;
		break;

	case GST_PAD_SINK:
		/*
		 * compute othersize = # of samples to be produced on
		 * source pad from sample count available on sink pad
		 *
		 * size = # of samples available on sink pad
		 *
		 * cadence = # of input samples per output sample
		 *
		 * remainder = how many extra input samples have been
		 * stored because the most recent input buffer
		 * ended before a complete cycle
		 */

		if(size >= element->cadence - element->remainder)
			*othersize = (size + element->remainder) / element->cadence;
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
	GSTLALLogicalUnderSample *element = GSTLAL_LOGICALUNDERSAMPLE(trans);

	element->t0 = GST_CLOCK_TIME_NONE;
	element->offset0 = GST_BUFFER_OFFSET_NONE;
	element->next_in_offset = GST_BUFFER_OFFSET_NONE;
	element->next_out_offset = GST_BUFFER_OFFSET_NONE;
	element->need_discont = TRUE;

	element->remainder = 0;
	element->leftover = 0;

	return TRUE;
}


/*
 * transform()
 */


static GstFlowReturn transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf)
{
	GSTLALLogicalUnderSample *element = GSTLAL_LOGICALUNDERSAMPLE(trans);
	GstMapInfo inmap, outmap;
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

		/*
		 * input is not gap.
		 */

		gst_buffer_map(inbuf, &inmap, GST_MAP_READ);
		gst_buffer_map(outbuf, &outmap, GST_MAP_WRITE);
		undersample(inmap.data, inmap.size, outmap.data, outmap.size, element->unit_size, element->sign, element->rate_in / element->rate_out, &element->remainder, &element->leftover, &element->required_on, element->status_out);
		set_metadata(element, outbuf, outmap.size / element->unit_size, FALSE);
		gst_buffer_unmap(outbuf, &outmap);
		gst_buffer_unmap(inbuf, &inmap);
	} else {
		/*
		 * input is gap.
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
 * properties
 */


enum property {
	ARG_REQUIRED_ON = 1,
	ARG_STATUS_OUT
};


static void set_property(GObject *object, enum property prop_id, const GValue *value, GParamSpec *pspec)
{
	GSTLALLogicalUnderSample *element = GSTLAL_LOGICALUNDERSAMPLE(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_REQUIRED_ON:
		element->required_on = g_value_get_uint(value);
		break;

	case ARG_STATUS_OUT:
		element->status_out = g_value_get_uint(value);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, enum property prop_id, GValue *value, GParamSpec *pspec)
{
	GSTLALLogicalUnderSample *element = GSTLAL_LOGICALUNDERSAMPLE(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_REQUIRED_ON:
		g_value_set_uint(value, element->required_on);
		break;

	case ARG_STATUS_OUT:
		g_value_set_uint(value, element->status_out);
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


static void gstlal_logicalundersample_class_init(GSTLALLogicalUnderSampleClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(klass);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);

	transform_class->transform_caps = GST_DEBUG_FUNCPTR(transform_caps);
	transform_class->transform_size = GST_DEBUG_FUNCPTR(transform_size);
	transform_class->get_unit_size = GST_DEBUG_FUNCPTR(get_unit_size);
	transform_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
	transform_class->start = GST_DEBUG_FUNCPTR(start);
	transform_class->transform = GST_DEBUG_FUNCPTR(transform);

	gst_element_class_set_details_simple(element_class,
		"Undersample",
		"Filter/Audio",
		"Undersamples an integer stream. The undersampling applies a bit mask across all cadence samples.  (Cadence samples are the input samples that are combined via bitwise & to make one output sample.) The undersampled stream is therefore a summary of the cadence samples.  This element's output sample rate must be an integer divisor of its input sample rate.",
		"Madeline Wade <madeline.wade@ligo.org>, Aaron Viets <aaron.viets@ligo.org>"
	);

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));

	g_object_class_install_property(
		gobject_class,
		ARG_REQUIRED_ON,
		g_param_spec_uint(
			"required-on",
			"On bits",
			"Bit mask setting the bits that must be on in the incoming stream.  Note:  if the mask is wider than the input stream, the high-order bits should be 0 or the on condition will never be met.",
			0, G_MAXUINT, 0x1,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_STATUS_OUT,
		g_param_spec_uint(
			"status-out",
			"Out bits",
			"Value of output if required-on mask is true.",
			0, G_MAXUINT, 0x1,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
}


/*
 * init()
 */


static void gstlal_logicalundersample_init(GSTLALLogicalUnderSample *element)
{
	element->rate_in = 0;
	element->rate_out = 0;
	element->unit_size = 0;
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(element), TRUE);
}


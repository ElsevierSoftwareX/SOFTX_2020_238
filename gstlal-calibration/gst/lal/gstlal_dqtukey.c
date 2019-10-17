/*
 * Copyright (C) 2018  Aaron Viets <aaron.viets@ligo.org>
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
#include <math.h>


/*
 * stuff from gobject/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/audio/audio.h>
#include <gst/base/gstbasetransform.h>
#include <gstlal/gstlal.h>
#include <gstlal/gstlal_audio_info.h>
#include <gstlal_dqtukey.h>


/*
 * ============================================================================
 *
 *			       Utilities
 *
 * ============================================================================
 */


#define DEFINE_DQ_TO_TUKEY(INTYPE, WIDTH, OUTTYPE) \
static void dq_g ## INTYPE ## WIDTH ## _to_tukey_ ## OUTTYPE(const g ## INTYPE ## WIDTH *src, gint64 src_size, OUTTYPE *dst, enum gstlal_dqtukey_state *state, guint32 required_on, guint32 required_on_xor_off, OUTTYPE *ramp, gint64 transition_samples, gint64 *ramp_up_index, gint64 *ramp_down_index, int num_cycle_in, int num_cycle_out, gint64 *num_leftover, int *remainder, gint64 *num_since_bad, gboolean invert_window, gboolean invert_control) { \
 \
	gint64 i, i_stop; \
	i = 0; \
	int j, j_stop; \
 \
	switch(*state) { \
	case START: \
		goto start; \
	case ONES: \
		goto ones; \
	case ZEROS: \
		goto zeros; \
	case RAMP_UP: \
		goto ramp_up; \
	case RAMP_DOWN: \
		goto ramp_down; \
	case DOUBLE_RAMP: \
		goto double_ramp; \
	default: \
		g_assert_not_reached(); \
	} \
 \
start: \
	i_stop = src_size * num_cycle_out / num_cycle_in + *num_leftover < transition_samples ? src_size : ((transition_samples - *num_leftover) * num_cycle_in + num_cycle_out - 1) / num_cycle_out; \
	if(invert_control) { \
		for(i = 0; i < i_stop; i++) { \
			if((src[i] ^ required_on) & required_on_xor_off) { \
				/* Since invert_control is TRUE, this means the conditions were met */ \
				if(!(i % num_cycle_in)) \
					*num_since_bad += num_cycle_out; \
			} else \
				*num_since_bad = 0; \
		} \
	} else { \
		for(i = 0; i < i_stop; i++) { \
			if((src[i] ^ required_on) & required_on_xor_off) { \
				/* The conditions were not all met */ \
				*num_since_bad = 0; \
			} \
			else if(!(i % num_cycle_in)) \
				*num_since_bad += num_cycle_out; \
		} \
	} \
	/* Track the number of output samples we could produce if there were no latency */ \
	*num_leftover += i_stop * num_cycle_out / num_cycle_in; \
	/* Check if any output samples should be produced */ \
	if(*num_leftover > transition_samples) { \
		j_stop = *num_leftover - transition_samples; \
		for(j = 0; j < j_stop; j++, dst++) { \
			if(*num_since_bad == *num_leftover) \
				*dst = invert_window ? 0.0 : 1.0; \
			else if(j_stop - j <= *num_since_bad - transition_samples) { \
				*dst = invert_window ? 1.0 - ramp[j - j_stop + *num_since_bad - transition_samples] : ramp[j - j_stop + *num_since_bad - transition_samples]; \
				(*ramp_up_index)++; \
			} else \
				*dst = invert_window ? 1.0 : 0.0; \
		} \
	} \
	/* Check if we are at the end of the input buffer and still need more input to make output */ \
	if(*num_leftover < transition_samples) { \
		return; \
		/* Check if we have enough leftover samples to start producing output, but no more */ \
	} else if(i_stop == src_size) { \
		/* Decide what we should do with the next data */ \
		if(*num_since_bad == *num_leftover) \
			*state = ONES; \
		else if(*num_since_bad <= transition_samples) \
			*state = ZEROS; \
		else \
			*state = RAMP_UP; \
		*num_leftover = transition_samples; \
		return; \
	} else { \
		/* Decide what we should do with the next data on this buffer */ \
		if(*num_since_bad == *num_leftover) { \
			*num_leftover = transition_samples; \
			*state = ONES; \
			goto ones; \
		} \
		*num_leftover = transition_samples; \
		if(*num_since_bad <= transition_samples) { \
			*state = ZEROS; \
			goto zeros; \
		} else { \
			*state = RAMP_UP; \
			goto ramp_up; \
		} \
	} \
 \
ones: \
	/* Deal with any output samples that still need to be produced from the last input */ \
	if(invert_window) { \
		for(j = 0; j < *remainder; j++, dst++) \
			*dst = 0.0; \
	} else { \
		for(j = 0; j < *remainder; j++, dst++) \
			*dst = 1.0; \
	} \
	*remainder = 0; \
	while(i < src_size) { \
		/*
		 * (a ? b : !b) is the equivalent of (a == b), that is, checking if a and b have the same
		 * truth value. Using (a == b) does not work, even if a is typecasted to a boolean. The
		 * case where a is nonzero and not 1 and b is TRUE fails.
		 */ \
		if((src[i] ^ required_on) & required_on_xor_off ? invert_control : !invert_control) { \
			/* Conditions were met */ \
			if(!((i + 1) % num_cycle_in)) { \
				/* In case rate in > rate out */ \
				if(invert_window) { \
					for(j = 0; j < num_cycle_out; j++, dst++) \
						*dst = 0.0; \
				} else { \
					for(j = 0; j < num_cycle_out; j++, dst++) \
						*dst = 1.0; \
				} \
			} \
		} else { \
			/* Failed to meet conditions */ \
			*num_since_bad = 0; \
			*state = RAMP_DOWN; \
			goto ramp_down; \
		} \
		i++; \
	} \
	return; \
 \
zeros: \
	/* Deal with any output samples that still need to be produced from the last input */ \
	if(invert_window) { \
		for(j = 0; j < *remainder; j++, dst++) \
			*dst = 1.0; \
	} else { \
		for(j = 0; j < *remainder; j++, dst++) \
			*dst = 0.0; \
	} \
	*remainder = 0; \
	while(i < src_size) { \
		if((src[i] ^ required_on) & required_on_xor_off ? invert_control : !invert_control) { \
			/* Conditions were met */ \
			if(!(i % num_cycle_in)) { \
				/* In case rate in > rate out */ \
				*num_since_bad += num_cycle_out; \
			} \
		} else { \
			/* Failed to meet conditions */ \
			*num_since_bad = 0; \
		} \
		i++; \
		if(!(i % num_cycle_in)) { \
			if(*num_since_bad <= transition_samples) { \
				if(invert_window) { \
					for(j = 0; j < num_cycle_out; j++, dst++) \
						*dst = 1.0; \
				} else { \
					for(j = 0; j < num_cycle_out; j++, dst++) \
						*dst = 0.0; \
				} \
			} else { \
				if(invert_window) { \
					for(j = 0; j < transition_samples + num_cycle_out - *num_since_bad; j++, dst++) \
						*dst = 1.0; \
				} else { \
					for(j = 0; j < transition_samples + num_cycle_out - *num_since_bad; j++, dst++) \
						*dst = 0.0; \
				} \
				*remainder = *num_since_bad - transition_samples; \
				*state = RAMP_UP; \
				goto ramp_up; \
			} \
		} \
	} \
	return; \
 \
ramp_up: \
	/* Deal with any output samples that still need to be produced from the last input */ \
	for(j = 0; j < *remainder; j++, dst++, (*ramp_up_index)++) { \
		if(invert_window) \
			*dst = 1.0 - ramp[*ramp_up_index]; \
		else \
			*dst = ramp[*ramp_up_index]; \
		if(*ramp_up_index == transition_samples - 1) { \
			/* The transition is over */ \
			*ramp_up_index = 0; \
			*state = ONES; \
			*remainder -= j + 1; \
			dst++; \
			goto ones; \
		} \
	} \
	*remainder = 0; \
	while(i < src_size) { \
		if((src[i] ^ required_on) & required_on_xor_off ? invert_control : !invert_control) { \
			/* Conditions were met */ \
			if(!((i + 1) % num_cycle_in)) { \
				/* In case rate in > rate out */ \
				for(j = 0; j < num_cycle_out; j++, dst++, (*ramp_up_index)++) { \
					if(invert_window) \
						*dst = 1.0 - ramp[*ramp_up_index]; \
					else \
						*dst = ramp[*ramp_up_index]; \
 \
					if(*ramp_up_index == transition_samples - 1) { \
						/* The transition is over */ \
						*ramp_up_index = 0; \
						*state = ONES; \
						*remainder = (num_cycle_out - j - 1) % num_cycle_out; \
						i++; \
						dst++; \
						goto ones; \
					} \
				} \
			} \
		} else { \
			/* Failed to meet conditions */ \
			*num_since_bad = 0; \
			*state = DOUBLE_RAMP; \
			goto double_ramp; \
		} \
		i++; \
	} \
	return; \
 \
ramp_down: \
	while(i < src_size) { \
		if((src[i] ^ required_on) & required_on_xor_off ? invert_control : !invert_control) { \
			/* Conditions were met */ \
			if(!(i % num_cycle_in)) { \
				/* In case rate in > rate out */ \
				*num_since_bad += num_cycle_out; \
			} \
		} else { \
			/* Failed to meet conditions */ \
			*num_since_bad = 0; \
		} \
		if(!(i % num_cycle_in)) { \
			for(j = 0; j < num_cycle_out; j++, dst++, (*ramp_down_index)++) { \
				if(invert_window) \
					*dst = ramp[*ramp_down_index]; \
				else \
					*dst = 1.0 - ramp[*ramp_down_index]; \
 \
				if(*ramp_down_index == transition_samples - 1) { \
					/* The transition is over */ \
					*ramp_down_index = 0; \
					*state = ZEROS; \
					*remainder = (num_cycle_out - j - 1) % num_cycle_out; \
					i++; \
					dst++; \
					goto zeros; \
				} \
			} \
		} \
		i++; \
	} \
	return; \
 \
double_ramp: \
	while(i < src_size) { \
		if((src[i] ^ required_on) & required_on_xor_off ? invert_control : !invert_control) { \
			/* Conditions were met */ \
			if(!(i % num_cycle_in)) { \
				/* In case rate in > rate out */ \
				*num_since_bad += num_cycle_out; \
			} \
		} else { \
			/* Failed to meet conditions */ \
			*num_since_bad = 0; \
		} \
		if(!(i % num_cycle_in)) { \
			for(j = 0; j < num_cycle_out; j++, dst++, (*ramp_down_index)++, (*ramp_up_index)++) { \
				if(invert_window) \
					*dst = 1.0 - (1.0 - ramp[*ramp_down_index]) * (*ramp_up_index < transition_samples ? ramp[*ramp_up_index] : 1.0); \
				else \
					*dst = (1.0 - ramp[*ramp_down_index]) * (*ramp_up_index < transition_samples ? ramp[*ramp_up_index] : 1.0); \
 \
				if(*ramp_down_index == transition_samples - 1) { \
					/* The transition is over */ \
					*ramp_up_index = 0; \
					*ramp_down_index = 0; \
					*state = ZEROS; \
					*remainder = (num_cycle_out - j - 1) % num_cycle_out; \
					i++; \
					dst++; \
					goto zeros; \
				} \
			} \
		} \
		i++; \
	} \
	return; \
}


DEFINE_DQ_TO_TUKEY(int, 8, float)
DEFINE_DQ_TO_TUKEY(int, 8, double)
DEFINE_DQ_TO_TUKEY(uint, 8, float)
DEFINE_DQ_TO_TUKEY(uint, 8, double)
DEFINE_DQ_TO_TUKEY(int, 16, float)
DEFINE_DQ_TO_TUKEY(int, 16, double)
DEFINE_DQ_TO_TUKEY(uint, 16, float)
DEFINE_DQ_TO_TUKEY(uint, 16, double)
DEFINE_DQ_TO_TUKEY(int, 32, float)
DEFINE_DQ_TO_TUKEY(int, 32, double)
DEFINE_DQ_TO_TUKEY(uint, 32, float)
DEFINE_DQ_TO_TUKEY(uint, 32, double)


/*
 * set the metadata on an output buffer
 */


static void set_metadata(GSTLALDQTukey *element, GstBuffer *buf, guint64 outsamples) {

	GST_BUFFER_OFFSET(buf) = element->next_out_offset;
	element->next_out_offset += outsamples;
	GST_BUFFER_OFFSET_END(buf) = element->next_out_offset;
	GST_BUFFER_PTS(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET(buf) - element->offset0, GST_SECOND, element->rate_out);
	GST_BUFFER_DURATION(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET_END(buf) - element->offset0, GST_SECOND, element->rate_out) - GST_BUFFER_PTS(buf);
	if(G_UNLIKELY(element->need_discont)) {
		GST_BUFFER_FLAG_SET(buf, GST_BUFFER_FLAG_DISCONT);
		element->need_discont = FALSE;
	}
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
		"format = (string) {" GST_AUDIO_NE(S8) ", " GST_AUDIO_NE(S16) ", " GST_AUDIO_NE(S32) ", " GST_AUDIO_NE(U8) ", " GST_AUDIO_NE(U16) ", " GST_AUDIO_NE(U32) "}, " \
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
		"format = (string) {" GST_AUDIO_NE(F32) ", " GST_AUDIO_NE(F64) "}, " \
		"layout = (string) interleaved, " \
		"channel-mask = (bitmask) 0"
	)
);


G_DEFINE_TYPE(
	GSTLALDQTukey,
	gstlal_dqtukey,
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
 * transform_caps()
 */


static GstCaps *transform_caps(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, GstCaps *filter) {

	GstCaps *othercaps = NULL;

	if(gst_caps_get_size(caps) > 1)
		GST_WARNING_OBJECT(trans, "not yet smart enough to transform complex formats");

	switch(direction) {
	case GST_PAD_SRC:
		/*
		 * The sink pad caps are always the same, regardless of the caps
		 * on the source pad.
		 */
		othercaps = gst_caps_normalize(gst_pad_get_pad_template_caps(GST_BASE_TRANSFORM_SINK_PAD(trans)));
		break;

	case GST_PAD_SINK:
		/*
		 * The source pad caps are always the same, regardless of the caps
		 * on the sink pad.
		 */
		othercaps = gst_caps_normalize(gst_pad_get_pad_template_caps(GST_BASE_TRANSFORM_SRC_PAD(trans)));
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


static gboolean set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps) {

	GSTLALDQTukey *element = GSTLAL_DQTUKEY(trans);
	gint rate_in, rate_out;
	gsize unit_size_in, unit_size_out;
	const gchar *format;
	static char *formats[] = {GST_AUDIO_NE(S8), GST_AUDIO_NE(S16), GST_AUDIO_NE(S32), GST_AUDIO_NE(U8), GST_AUDIO_NE(U16), GST_AUDIO_NE(U32)};
	gboolean sign[] = {TRUE, TRUE, TRUE, FALSE, FALSE, FALSE};

	/*
	 * parse the caps
	 */

	if(!get_unit_size(trans, incaps, &unit_size_in)) {
		GST_DEBUG_OBJECT(element, "failed to get unit size from input caps %" GST_PTR_FORMAT, incaps);
		return FALSE;
	}
	if(!get_unit_size(trans, outcaps, &unit_size_out)) {
		GST_DEBUG_OBJECT(element, "failed to get unit size from output caps %" GST_PTR_FORMAT, outcaps);
		return FALSE;
	}
	if(!gst_structure_get_int(gst_caps_get_structure(incaps, 0), "rate", &rate_in)) {
		GST_DEBUG_OBJECT(element, "unable to parse rate from input  caps %" GST_PTR_FORMAT, incaps);
		return FALSE;
	}
	if(!gst_structure_get_int(gst_caps_get_structure(outcaps, 0), "rate", &rate_out)) {
		GST_DEBUG_OBJECT(element, "unable to parse rate from output caps %" GST_PTR_FORMAT, outcaps);
		return FALSE;
	}

	/*
	 * require the input rate to be an integer multiple or divisor 
	 * of the output rate
	 */

	if(rate_in % rate_out && rate_out % rate_in) {
		GST_ERROR_OBJECT(element, "input rate is not an integer multiple or divisor of output rate.  input caps = %" GST_PTR_FORMAT " output caps = %" GST_PTR_FORMAT, incaps, outcaps);
		return FALSE;
	}

	/*
	 * record stream parameters
	 */

	element->rate_in = rate_in;
	element->rate_out = rate_out;
	element->num_cycle_in = rate_in > rate_out ? rate_in / rate_out : 1;
	element->num_cycle_out = rate_out > rate_in ? rate_out / rate_in : 1;
	element->unit_size_in = unit_size_in;
	element->unit_size_out = unit_size_out;

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
	if(test != 1) {
		GST_ERROR_OBJECT(element, "element->sign not properly set");
		return FALSE;
	}

	return TRUE;
}


/*
 * transform_size()
 */


static gboolean transform_size(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, gsize size, GstCaps *othercaps, gsize *othersize) {

	GSTLALDQTukey *element = GSTLAL_DQTUKEY(trans);
	gint64 temp_othersize;

	switch(direction) {
	case GST_PAD_SRC:

		/*
		 * convert byte count to samples
		 */

		if(G_UNLIKELY(size % element->unit_size_out)) {
			GST_DEBUG_OBJECT(element, "buffer size %" G_GSIZE_FORMAT " is not a multiple of unit_size %" G_GSIZE_FORMAT, size, (gsize) element->unit_size_out);
			return FALSE;
		}
		size /= element->unit_size_out;

		/*
		 * compute othersize = # of samples required on sink
		 * pad to produce requested sample count on source pad
		 *
		 * size = # of samples requested on source pad
		 * transition_samples = # of latency samples of this element
		 * num_leftover = # of samples saved from previous buffer
		 */

		temp_othersize = ((gint64) size + element->transition_samples - element->num_leftover) * element->rate_in / element->rate_out;

		/*
		 * convert sample count to byte count
		 */

		*othersize = (gsize) (temp_othersize * element->unit_size_in);

		break;

	case GST_PAD_SINK:

		/*
		 * convert byte count to samples
		 */

		if(G_UNLIKELY(size % element->unit_size_in)) {
			GST_DEBUG_OBJECT(element, "buffer size %" G_GSIZE_FORMAT " is not a multiple of unit_size %" G_GSIZE_FORMAT, size, (gsize) element->unit_size_in);
			return FALSE;
		}
		size /= element->unit_size_in;

		/*
		 * compute othersize = # of samples to be produced on
		 * source pad from sample count available on sink pad
		 *
		 * size = # of samples available on sink pad
		 * transition_samples = # of latency samples of this element
		 * num_leftover = # of samples saved from previous buffer
		 */

		temp_othersize = (gint64) size * element->rate_out / element->rate_in - element->transition_samples + element->num_leftover;

		/*
		 * convert sample count to byte count, and don't allow for negative sizes
		 */

		*othersize = temp_othersize > 0 ? (gsize) (temp_othersize * element->unit_size_out) : 0;

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
	GSTLALDQTukey *element = GSTLAL_DQTUKEY(trans);

	element->t0 = GST_CLOCK_TIME_NONE;
	element->offset0 = GST_BUFFER_OFFSET_NONE;
	element->next_in_offset = GST_BUFFER_OFFSET_NONE;
	element->next_out_offset = GST_BUFFER_OFFSET_NONE;
	element->need_discont = TRUE;

	element->state = START;
	element->ramp_up_index = 0;
	element->ramp_down_index = 0;
	element->num_leftover = 0;
	element->remainder = 0;
	element->num_since_bad = 0;

	if(element->required_on & element->required_off)
		GST_WARNING_OBJECT(element, "One or more bits are requested to be required both on and off. These bits will be ignored.");
	element->required_on_xor_off = element->required_on ^ element->required_off;

	return TRUE;
}


/*
 * transform()
 */


static GstFlowReturn transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf) {

	GSTLALDQTukey *element = GSTLAL_DQTUKEY(trans);
	GstMapInfo inmap, outmap;
	GstFlowReturn result = GST_FLOW_OK;

	/*
	 * check for discontinuity
	 */

	if(G_UNLIKELY(GST_BUFFER_IS_DISCONT(inbuf) || GST_BUFFER_OFFSET(inbuf) != element->next_in_offset || !GST_CLOCK_TIME_IS_VALID(element->t0))) {
		GST_DEBUG_OBJECT(element, "pushing discontinuous buffer at input timestamp %lu", (long unsigned) GST_TIME_AS_SECONDS(GST_BUFFER_PTS(inbuf)));
		element->t0 = GST_BUFFER_PTS(inbuf);
		element->offset0 = element->next_out_offset = gst_util_uint64_scale_ceil(GST_BUFFER_OFFSET(inbuf), element->rate_out, element->rate_in);
		element->need_discont = TRUE;
		element->state = START;
		element->ramp_up_index = 0;
		element->ramp_down_index = 0;
		element->num_leftover = 0;
		element->remainder = 0;
		element->num_since_bad = 0;

		/* If we haven't made a window for transitions between zeros and ones yet, do it now */
		if(!element->ramp && element->planck_taper) {
			/* Make a Planck-taper window */
			if(element->unit_size_out == 4) {
				element->ramp = g_malloc(element->transition_samples * sizeof(float));
				float *ramp = (float *) element->ramp;
				gint64 i;
				float i_frac, Z;
				for(i = 0; i < element->transition_samples; i++) {
					i_frac = (i + 1.0) / (element->transition_samples + 1.0);
					Z = 1.0 / i_frac + 1.0 / (i_frac - 1);
					ramp[i] = (float) (1.0 / (1.0 + expf(Z)));
				}
			} else {
				element->ramp = g_malloc(element->transition_samples * sizeof(double));
				double *ramp = (double *) element->ramp;
				gint64 i;
				double i_frac, Z;
				for(i = 0; i < element->transition_samples; i++) {
					i_frac = (i + 1.0) / (element->transition_samples + 1.0);
					Z = 1.0 / i_frac + 1.0 / (i_frac - 1);
					ramp[i] = 1.0 / (1.0 + exp(Z));
				}
			}
		} else if(!element->ramp) {
			/* Make a half-Hann window */
			if(element->unit_size_out == 4) {
				element->ramp = g_malloc(element->transition_samples * sizeof(float));
				float *ramp = (float *) element->ramp;
				gint64 i;
				for(i = 0; i < element->transition_samples; i++)
					ramp[i] = (float) sin((i + 1) * M_PI / (2 * (element->transition_samples + 1))) * sin((i + 1) * M_PI / (2 * (element->transition_samples + 1)));
			} else {
				element->ramp = g_malloc(element->transition_samples * sizeof(double));
				double *ramp = (double *) element->ramp;
				gint64 i;
				for(i = 0; i < element->transition_samples; i++)
					ramp[i] = sin((i + 1) * M_PI / (2 * (element->transition_samples + 1))) * sin((i + 1) * M_PI / (2 * (element->transition_samples + 1)));
			}
		}
	}

	element->next_in_offset = GST_BUFFER_OFFSET_END(inbuf);

	/*
	 * process buffer
	 */

	gst_buffer_map(inbuf, &inmap, GST_MAP_READ);
	gst_buffer_map(outbuf, &outmap, GST_MAP_WRITE);

	g_assert_cmpuint(inmap.size % element->unit_size_in, ==, 0);
	g_assert_cmpuint(outmap.size % element->unit_size_out, ==, 0);

	gint64 src_size = (gint64) (inmap.size / element->unit_size_in);

	switch(element->unit_size_in) {
	case 1:
		if(element->sign) {
			if(element->unit_size_out == 4)
				dq_gint8_to_tukey_float((const gint8 *) inmap.data, src_size, (float *) outmap.data, &element->state, element->required_on, element->required_on_xor_off, element->ramp, element->transition_samples, &element->ramp_up_index, &element->ramp_down_index, element->num_cycle_in, element->num_cycle_out, &element->num_leftover, &element->remainder, &element->num_since_bad, element->invert_window, element->invert_control);
			else
				dq_gint8_to_tukey_double((const gint8 *) inmap.data, src_size, (double *) outmap.data, &element->state, element->required_on, element->required_on_xor_off, element->ramp, element->transition_samples, &element->ramp_up_index, &element->ramp_down_index, element->num_cycle_in, element->num_cycle_out, &element->num_leftover, &element->remainder, &element->num_since_bad, element->invert_window, element->invert_control);
		} else {
			if(element->unit_size_out == 4)
				dq_guint8_to_tukey_float((const guint8 *) inmap.data, src_size, (float *) outmap.data, &element->state, element->required_on, element->required_on_xor_off, element->ramp, element->transition_samples, &element->ramp_up_index, &element->ramp_down_index, element->num_cycle_in, element->num_cycle_out, &element->num_leftover, &element->remainder, &element->num_since_bad, element->invert_window, element->invert_control);
			else
				dq_guint8_to_tukey_double((const guint8 *) inmap.data, src_size, (double *) outmap.data, &element->state, element->required_on, element->required_on_xor_off, element->ramp, element->transition_samples, &element->ramp_up_index, &element->ramp_down_index, element->num_cycle_in, element->num_cycle_out, &element->num_leftover, &element->remainder, &element->num_since_bad, element->invert_window, element->invert_control);
		}
		break;
	case 2:
		if(element->sign) {
			if(element->unit_size_out == 4)
				dq_gint16_to_tukey_float((const gint16 *) inmap.data, src_size, (float *) outmap.data, &element->state, element->required_on, element->required_on_xor_off, element->ramp, element->transition_samples, &element->ramp_up_index, &element->ramp_down_index, element->num_cycle_in, element->num_cycle_out, &element->num_leftover, &element->remainder, &element->num_since_bad, element->invert_window, element->invert_control);
			else
				dq_gint16_to_tukey_double((const gint16 *) inmap.data, src_size, (double *) outmap.data, &element->state, element->required_on, element->required_on_xor_off, element->ramp, element->transition_samples, &element->ramp_up_index, &element->ramp_down_index, element->num_cycle_in, element->num_cycle_out, &element->num_leftover, &element->remainder, &element->num_since_bad, element->invert_window, element->invert_control);
		} else {
			if(element->unit_size_out == 4)
				dq_guint16_to_tukey_float((const guint16 *) inmap.data, src_size, (float *) outmap.data, &element->state, element->required_on, element->required_on_xor_off, element->ramp, element->transition_samples, &element->ramp_up_index, &element->ramp_down_index, element->num_cycle_in, element->num_cycle_out, &element->num_leftover, &element->remainder, &element->num_since_bad, element->invert_window, element->invert_control);
			else
				dq_guint16_to_tukey_double((const guint16 *) inmap.data, src_size, (double *) outmap.data, &element->state, element->required_on, element->required_on_xor_off, element->ramp, element->transition_samples, &element->ramp_up_index, &element->ramp_down_index, element->num_cycle_in, element->num_cycle_out, &element->num_leftover, &element->remainder, &element->num_since_bad, element->invert_window, element->invert_control);
		}
		break;
	case 4:
		if(element->sign) {
			if(element->unit_size_out == 4)
				dq_gint32_to_tukey_float((const gint32 *) inmap.data, src_size, (float *) outmap.data, &element->state, element->required_on, element->required_on_xor_off, element->ramp, element->transition_samples, &element->ramp_up_index, &element->ramp_down_index, element->num_cycle_in, element->num_cycle_out, &element->num_leftover, &element->remainder, &element->num_since_bad, element->invert_window, element->invert_control);
			else
				dq_gint32_to_tukey_double((const gint32 *) inmap.data, src_size, (double *) outmap.data, &element->state, element->required_on, element->required_on_xor_off, element->ramp, element->transition_samples, &element->ramp_up_index, &element->ramp_down_index, element->num_cycle_in, element->num_cycle_out, &element->num_leftover, &element->remainder, &element->num_since_bad, element->invert_window, element->invert_control);
		} else {
			if(element->unit_size_out == 4)
				dq_guint32_to_tukey_float((const guint32 *) inmap.data, src_size, (float *) outmap.data, &element->state, element->required_on, element->required_on_xor_off, element->ramp, element->transition_samples, &element->ramp_up_index, &element->ramp_down_index, element->num_cycle_in, element->num_cycle_out, &element->num_leftover, &element->remainder, &element->num_since_bad, element->invert_window, element->invert_control);
			else
				dq_guint32_to_tukey_double((const guint32 *) inmap.data, src_size, (double *) outmap.data, &element->state, element->required_on, element->required_on_xor_off, element->ramp, element->transition_samples, &element->ramp_up_index, &element->ramp_down_index, element->num_cycle_in, element->num_cycle_out, &element->num_leftover, &element->remainder, &element->num_since_bad, element->invert_window, element->invert_control);
		}
		break;
	default:
		g_assert_not_reached();
	}

	set_metadata(element, outbuf, outmap.size / element->unit_size_out);
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
	ARG_REQUIRED_ON = 1,
	ARG_REQUIRED_OFF,
	ARG_TRANSITION_SAMPLES,
	ARG_INVERT_WINDOW,
	ARG_INVERT_CONTROL,
	ARG_PLANCK_TAPER
};


static void set_property(GObject *object, enum property prop_id, const GValue *value, GParamSpec *pspec) {

	GSTLALDQTukey *element = GSTLAL_DQTUKEY(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_REQUIRED_ON:
		element->required_on = (guint32) g_value_get_uint64(value);
		break;

	case ARG_REQUIRED_OFF:
		element->required_off = (guint32) g_value_get_uint64(value);
		break;

	case ARG_TRANSITION_SAMPLES:
		element->transition_samples = g_value_get_int64(value);
		break;

	case ARG_INVERT_WINDOW:
		element->invert_window = g_value_get_boolean(value);
		break;

	case ARG_INVERT_CONTROL:
		element->invert_control = g_value_get_boolean(value);
		break;

	case ARG_PLANCK_TAPER:
		element->planck_taper = g_value_get_boolean(value);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, enum property prop_id, GValue *value, GParamSpec *pspec) {

	GSTLALDQTukey *element = GSTLAL_DQTUKEY(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_REQUIRED_ON:
		g_value_set_uint64(value, (guint64) element->required_on);
		break;

	case ARG_REQUIRED_OFF:
		g_value_set_uint64(value, (guint64) element->required_off);
		break;

	case ARG_TRANSITION_SAMPLES:
		g_value_set_int64(value, element->transition_samples);
		break;

	case ARG_INVERT_WINDOW:
		g_value_set_boolean(value, element->invert_window);
		break;

	case ARG_INVERT_CONTROL:
		g_value_set_boolean(value, element->invert_control);
		break;

	case ARG_PLANCK_TAPER:
		g_value_set_boolean(value, element->planck_taper);
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


static void gstlal_dqtukey_class_init(GSTLALDQTukeyClass *klass) {

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
		"DQTukey",
		"Filter/Audio",
		"Reads in a DQ bit vector and writes out a Tukey window. If the required bits in\n\t\t\t   "
		"the DQ vector are on, the output will be ones. If the required bits are off, the\n\t\t\t   "
		"output will be zeros. The transition between zeros and ones is made smooth with\n\t\t\t   "
		"half of a Hann window, which occupies the time during which the required bits\n\t\t\t   "
		"are on, just before or after a transition.",
		"Aaron Viets <aaron.viets@ligo.org>"
	);

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));

	g_object_class_install_property(
		gobject_class,
		ARG_REQUIRED_ON,
		g_param_spec_uint64(
			"required-on",
			"On bits",
			"Bit mask setting the bits that must be on in the incoming stream for the output\n\t\t\t"
			"stream to be 1.0. Note: if the mask is wider than the input stream, the\n\t\t\t"
			"high-order bits should be 0 or the on condition will never be met.",
			0, G_MAXUINT32, 0x1,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_REQUIRED_OFF,
		g_param_spec_uint64(
			"required-off",
			"Off bits",
			"Bit mask setting the bits that must be off in the incoming stream for the\n\t\t\t"
			"output stream to be 1.0.",
			0, G_MAXUINT32, 0x0,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_TRANSITION_SAMPLES,
		g_param_spec_int64(
			"transition-samples",
			"Transition Samples",
			"Number of output samples used for smooth transitions between 0.0 and 1.0. Half\n\t\t\t"
			"of a Hann window is used to make transitions.",
			0, G_MAXINT64, 4096,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_INVERT_WINDOW,
		g_param_spec_boolean(
			"invert-window",
			"Invert Window",
			"If set to True, output is replaced by 1 - output. This inverts the Tukey window.",
			FALSE,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_INVERT_CONTROL,
		g_param_spec_boolean(
			"invert-control",
			"Invert Control",
			"If set to True, the conditions required by the bitmasks must not all be met in\n\t\t\t"
			"order for the output stream to be 1.0",
			FALSE,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_PLANCK_TAPER,
		g_param_spec_boolean(
			"planck-taper",
			"Planck Taper",
			"Set to True to use a Planck-taper window instead of a half-Hann window.",
			FALSE,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
}


/*
 * init()
 */


static void gstlal_dqtukey_init(GSTLALDQTukey *element) {

	element->rate_in = 0;
	element->rate_out = 0;
	element->unit_size_in = 0;
	element->unit_size_out = 0;
	element->ramp = NULL;
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(element), TRUE);
}


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


/**
 * SECTION:gstlal_insertgap
 * @short_description: This element replaces undesired values, specified
 * by the property #bad_data_intervals, with gaps.
 *
 * The primary purpose of this element is to reject certain values
 * considered by the user to be "bad data." This can be done by flagging
 * that data as a gap, replacing it with a specified value, or both. The
 * criteria for bad data is specified by the property #bad_data_intervals,
 * which is an array of at most 16 elements. Array indices 0, 2, 4, etc.,
 * represent maxima, and array indices 1, 3, 5, etc., represent the 
 * corresponding minima. For example, if 
 * #bad_data_intervals = [0, 1, 2, 3],
 * then any values that fall in one of the closed intervals [-inf, 0],
 * [1, 2], or [3, inf] will be rejected and gapped and/or replaced
 * as specified by the user. To reject a single value, say zero,
 * #bad_data_intervals should be [-max_double, 0, 0, max_double].
 * If the data stream is complex, the real and and imaginary parts of
 * each input is tested, and if either is bad, the value is rejected.
 * The #bad_data_intervals and #replace_value properties are applied to
 * both real and imaginary parts.
 * This element also has the ability to fill in discontinuities if the
 * property #fill-discont is set to true. Presentation timestamps and
 * buffer offsets are adjusted as needed.
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


#include <math.h>
#include <string.h>
#include <complex.h>


/*
 * stuff from gobject/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/audio/audio.h>


/*
 * our own stuff
 */


#include <gstlal/gstlal.h>
#include <gstlal/gstlal_debug.h>
#include <gstlal/gstlal_audio_info.h>
#include <gstlal_insertgap.h>


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
		"rate = (int) [1, MAX], " \
		"channels = (int) 1, " \
		"format = (string) { "GST_AUDIO_NE(U32)", " GST_AUDIO_NE(F32) ", " GST_AUDIO_NE(F64) ", " GST_AUDIO_NE(Z64) ", " GST_AUDIO_NE(Z128) " }, " \
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
		"rate = (int) [1, MAX], " \
		"channels = (int) 1, " \
		"format = (string) { "GST_AUDIO_NE(U32)", " GST_AUDIO_NE(F32) ", " GST_AUDIO_NE(F64) ", " GST_AUDIO_NE(Z64) ", " GST_AUDIO_NE(Z128) " }, " \
		"layout = (string) interleaved, " \
		"channel-mask = (bitmask) 0"
	)
);


#define GST_CAT_DEFAULT gstlal_insertgap_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


G_DEFINE_TYPE_WITH_CODE(
	GSTLALInsertGap,
	gstlal_insertgap,
	GST_TYPE_ELEMENT,
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "lal_insertgap", 0, "lal_insertgap element")
);


/*
 * ============================================================================
 *
 *			       Utilities
 *
 * ============================================================================
 */


static gboolean check_data(double complex data, double *bad_data_intervals, gint array_length, gboolean remove_nan, gboolean remove_inf, gboolean complex_data) {
	double data_re = creal(data);
	gint i;
	if(complex_data){
		double data_im = cimag(data);
		if(bad_data_intervals) {
			if(data_re <= bad_data_intervals[0] || data_re >= bad_data_intervals[array_length - 1] || data_im <= bad_data_intervals[0] || data_im >= bad_data_intervals[array_length - 1])
				return TRUE;
			for(i = 1; i < array_length - 1; i += 2) {
				if((data_re >= bad_data_intervals[i] && data_re <= bad_data_intervals[i + 1]) || (data_im >= bad_data_intervals[i] && data_im <= bad_data_intervals[i + 1]))
					return TRUE;
			}
		}
		if(remove_nan && (isnan(data_re) || isnan(data_im)))
			return TRUE;
		if(remove_inf && (isinf(data_re) || isinf(data_im)))
			return TRUE;
	} else {
		if(bad_data_intervals) {
			if(data_re <= bad_data_intervals[0] || data_re >= bad_data_intervals[array_length - 1])
				return TRUE;
			for(i = 1; i < array_length - 1; i += 2) {
				if(data_re >= bad_data_intervals[i] && data_re <= bad_data_intervals[i + 1])
					return TRUE;
			}
		}
		if(remove_nan && (isnan(data_re)))
			return TRUE;
		if(remove_inf && (isinf(data_re)))
			return TRUE;
	}

	return FALSE;
}


#define DEFINE_PROCESS_INBUF(DTYPE,COMPLEX) \
static GstFlowReturn process_inbuf_ ## DTYPE ## COMPLEX(const DTYPE COMPLEX *indata, DTYPE COMPLEX *outdata, GSTLALInsertGap *element, gboolean sinkbuf_gap, gboolean sinkbuf_discont, guint64 sinkbuf_offset, guint64 sinkbuf_offset_end, GstClockTime sinkbuf_dur, GstClockTime sinkbuf_pts, gboolean complex_data) \
{ \
	GstFlowReturn result = GST_FLOW_OK; \
	guint64 blocks, max_block_length, missing_samples; \
	missing_samples = 0; \
 \
	/*
	 * First, deal with discontinuity if necessary
	 */ \
	if(element->fill_discont && (element->last_sinkbuf_offset_end != 0) && (sinkbuf_pts != element->last_sinkbuf_ets)) { \
 \
		/* Track discont length and number of zero-length buffers */ \
		element->discont_time += (sinkbuf_pts - element->last_sinkbuf_ets); \
		element->empty_bufs += (sinkbuf_dur ? 0 : 1); \
 \
		/* Find number of missing samples and max block length in samples */ \
		missing_samples = gst_util_uint64_scale_int_round(sinkbuf_pts - element->last_sinkbuf_ets, element->rate, 1000000000); \
		blocks = (sinkbuf_pts - element->last_sinkbuf_ets + element->block_duration - 1) / element->block_duration; /* ceil */ \
		g_assert_cmpuint(blocks, >, 0); /* make sure that the discont is not zero length */ \
		max_block_length = (missing_samples + blocks - 1) / blocks; /* ceil */ \
		g_assert_cmpuint(max_block_length, >, 0); \
 \
		/* Message for debugging */ \
		if(sinkbuf_dur && sinkbuf_offset != sinkbuf_offset_end) \
			GST_WARNING_OBJECT(element, "filling discontinuity lasting %f seconds (%lu samples) including %lu zero-length buffers and starting at %f seconds (offset %lu)", (((double) element->discont_time) / 1000000000.0), gst_util_uint64_scale_int_round(element->discont_time, element->rate, 1000000000), element->empty_bufs, (double) sinkbuf_pts / 1000000000.0 - (double) element->discont_time / 1000000000.0, sinkbuf_offset); \
 \
		guint standard_blocks = (guint) (missing_samples / max_block_length); \
		guint64 last_block_length = missing_samples % max_block_length; \
		DTYPE COMPLEX sample_value; \
		if(complex_data) \
			sample_value = (element->replace_value < G_MAXDOUBLE) ? ((DTYPE) (element->replace_value)) * (1 + I) : 0; \
		else \
			sample_value = (element->replace_value < G_MAXDOUBLE) ? ((DTYPE) (element->replace_value)) : 0; \
 \
		/* first make and push any buffers of size max_buffer_size */ \
		if(standard_blocks != 0) { \
			guint buffer_num; \
			for(buffer_num = 0; buffer_num < standard_blocks; buffer_num++) { \
				GstBuffer *discont_buf; \
				DTYPE COMPLEX *discont_buf_data; \
				discont_buf_data = g_malloc(max_block_length * sizeof(DTYPE COMPLEX)); \
				guint sample_num; \
				for(sample_num = 0; sample_num < max_block_length; sample_num++) { \
					*discont_buf_data = sample_value; \
					discont_buf_data++; \
				} \
				discont_buf = gst_buffer_new_wrapped((discont_buf_data - max_block_length), max_block_length * sizeof(DTYPE COMPLEX)); \
				if(G_UNLIKELY(!discont_buf)) { \
					GST_ERROR_OBJECT(element, "failure creating sub-buffer"); \
					result = GST_FLOW_ERROR; \
					goto done; \
				} \
 \
				/* set flags, caps, offset, and timestamps. */ \
				GST_BUFFER_OFFSET(discont_buf) = element->last_sinkbuf_offset_end + element->discont_offset + buffer_num * max_block_length; \
				GST_BUFFER_OFFSET_END(discont_buf) = GST_BUFFER_OFFSET(discont_buf) + max_block_length; \
				GST_BUFFER_PTS(discont_buf) = element->last_sinkbuf_ets + gst_util_uint64_scale_round(sinkbuf_pts - element->last_sinkbuf_ets, (guint64) buffer_num * max_block_length, missing_samples); \
				GST_BUFFER_DURATION(discont_buf) = element->last_sinkbuf_ets + gst_util_uint64_scale_round(sinkbuf_pts - element->last_sinkbuf_ets, ((guint64) buffer_num + 1) * max_block_length, missing_samples) - GST_BUFFER_PTS(discont_buf); \
				GST_BUFFER_FLAG_UNSET(discont_buf, GST_BUFFER_FLAG_DISCONT); \
				if(element->insert_gap) \
					GST_BUFFER_FLAG_SET(discont_buf, GST_BUFFER_FLAG_GAP); \
				else \
					GST_BUFFER_FLAG_UNSET(discont_buf, GST_BUFFER_FLAG_GAP); \
 \
				/* push buffer downstream */ \
				GST_DEBUG_OBJECT(element, "pushing sub-buffer %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(discont_buf)); \
				result = gst_pad_push(element->srcpad, discont_buf); \
				if(G_UNLIKELY(result != GST_FLOW_OK)) { \
					GST_WARNING_OBJECT(element, "push failed: %s", gst_flow_get_name(result)); \
					goto done; \
				} \
			} \
		} \
 \
		/* then make and push the remainder buffer */ \
		if(last_block_length != 0) { \
			GstBuffer *last_discont_buf; \
			DTYPE COMPLEX *last_discont_buf_data; \
			last_discont_buf_data = g_malloc(last_block_length * sizeof(DTYPE COMPLEX)); \
			guint sample_num; \
			for(sample_num = 0; sample_num < last_block_length; sample_num++) { \
				*last_discont_buf_data = sample_value; \
				last_discont_buf_data++; \
			} \
			last_discont_buf = gst_buffer_new_wrapped((last_discont_buf_data - last_block_length), last_block_length * sizeof(DTYPE COMPLEX)); \
			if(G_UNLIKELY(!last_discont_buf)) { \
				GST_ERROR_OBJECT(element, "failure creating sub-buffer"); \
				result = GST_FLOW_ERROR; \
				goto done; \
			} \
 \
			/* set flags, caps, offset, and timestamps. */ \
			GST_BUFFER_OFFSET(last_discont_buf) = element->last_sinkbuf_offset_end + element->discont_offset + missing_samples - last_block_length; \
			GST_BUFFER_OFFSET_END(last_discont_buf) = GST_BUFFER_OFFSET(last_discont_buf) + last_block_length; \
			GST_BUFFER_PTS(last_discont_buf) = element->last_sinkbuf_ets + gst_util_uint64_scale_round(sinkbuf_pts - element->last_sinkbuf_ets, missing_samples - last_block_length, missing_samples); \
			GST_BUFFER_DURATION(last_discont_buf) = sinkbuf_pts - GST_BUFFER_PTS(last_discont_buf); \
			GST_BUFFER_FLAG_UNSET(last_discont_buf, GST_BUFFER_FLAG_DISCONT); \
			if(element->insert_gap) \
				GST_BUFFER_FLAG_SET(last_discont_buf, GST_BUFFER_FLAG_GAP); \
			else \
				GST_BUFFER_FLAG_UNSET(last_discont_buf, GST_BUFFER_FLAG_GAP); \
 \
			/* push buffer downstream */ \
			GST_DEBUG_OBJECT(element, "pushing sub-buffer %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(last_discont_buf)); \
			result = gst_pad_push(element->srcpad, last_discont_buf); \
			if(G_UNLIKELY(result != GST_FLOW_OK)) { \
				GST_WARNING_OBJECT(element, "push failed: %s", gst_flow_get_name(result)); \
				goto done; \
			} \
		} \
		element->discont_offset += missing_samples; \
	} \
	if(!sinkbuf_dur) \
		goto done; \
 \
	element->discont_time = 0; \
	element->empty_bufs = 0; \
 \
	/*
	 * Now, use data on input buffer to make next output buffer(s)
	 */ \
	gboolean data_is_bad, srcbuf_gap, srcbuf_gap_next; \
	guint64 offset, current_srcbuf_length; \
	current_srcbuf_length = 0; \
 \
	/* compute length of incoming buffer and maximum block length in samples */ \
	blocks = (sinkbuf_dur + element->block_duration - 1) / element->block_duration; /* ceil */ \
	g_assert_cmpuint(blocks, >, 0); /* make sure that the sinkbuf is not zero length */ \
	guint64 length = sinkbuf_offset_end - sinkbuf_offset; \
	max_block_length = (length + blocks - 1) / blocks; /* ceil */ \
	g_assert_cmpuint(max_block_length, >, 0); \
 \
	/* Check first sample */ \
	data_is_bad = check_data((double complex) *indata, element->bad_data_intervals, element->array_length, element->remove_nan, element->remove_inf, complex_data); \
	srcbuf_gap = (sinkbuf_gap && (!(element->remove_gap))) || ((element->insert_gap) && data_is_bad); \
	for(offset = 0; offset < length; offset++) { \
		data_is_bad = check_data((double complex) *indata, element->bad_data_intervals, element->array_length, element->remove_nan, element->remove_inf, complex_data); \
		srcbuf_gap_next = (sinkbuf_gap && (!(element->remove_gap))) || ((element->insert_gap) && data_is_bad); \
		if(complex_data) \
			*outdata = (((element->replace_value) < G_MAXDOUBLE) && data_is_bad) ? ((DTYPE) (element->replace_value)) * (1 + I) : *indata; \
		else \
			*outdata = (((element->replace_value) < G_MAXDOUBLE) && data_is_bad) ? (DTYPE) (element->replace_value) : *indata; \
		current_srcbuf_length++; \
		indata++; \
		outdata++; \
 \
		/* 
		 * We need to push an output buffer if:
		 * 1) The number of samples to be output equals the maximum output buffer size
		 * 2) We have reached the end of the input buffer 
		 * 3) The output data changes from non-gap to gap or vice-versa
		 */ \
		if((current_srcbuf_length >= max_block_length) || (offset >= length - 1) || (srcbuf_gap_next != srcbuf_gap)) { \
			if((current_srcbuf_length > max_block_length) || (offset > length - 1)) \
				g_assert_not_reached(); \
			if(srcbuf_gap_next != srcbuf_gap) { \
				/*
				 * In this case, we don't want the most recent sample since its
				 * gap state is different. Put it on the next output buffer.
				 */ \
				offset--; \
				current_srcbuf_length--; \
				indata--; \
				outdata--; \
			} \
			GstBuffer *srcbuf; \
			DTYPE COMPLEX *srcbuf_data; \
			srcbuf_data = g_malloc(current_srcbuf_length * sizeof(DTYPE COMPLEX)); \
			memcpy(srcbuf_data, (outdata - current_srcbuf_length), current_srcbuf_length * sizeof(DTYPE COMPLEX)); \
			srcbuf = gst_buffer_new_wrapped(srcbuf_data, current_srcbuf_length * sizeof(DTYPE COMPLEX)); \
 \
			if(G_UNLIKELY(!srcbuf)) { \
				GST_ERROR_OBJECT(element, "failure creating sub-buffer"); \
				result = GST_FLOW_ERROR; \
				goto done; \
			} \
 \
			/* set flags, caps, offset, and timestamps. */ \
			GST_BUFFER_OFFSET(srcbuf) = sinkbuf_offset + element->discont_offset + offset + 1 - current_srcbuf_length; \
			GST_BUFFER_OFFSET_END(srcbuf) = GST_BUFFER_OFFSET(srcbuf) + current_srcbuf_length; \
			GST_BUFFER_PTS(srcbuf) = sinkbuf_pts + gst_util_uint64_scale_int_round(sinkbuf_dur, offset + 1 - current_srcbuf_length, length); \
			GST_BUFFER_DURATION(srcbuf) = sinkbuf_pts + gst_util_uint64_scale_int_round(sinkbuf_dur, offset + 1, length) - GST_BUFFER_PTS(srcbuf); \
			if(srcbuf_gap) \
				GST_BUFFER_FLAG_SET(srcbuf, GST_BUFFER_FLAG_GAP); \
			else \
				GST_BUFFER_FLAG_UNSET(srcbuf, GST_BUFFER_FLAG_GAP); \
 \
			/*
			 * only the first subbuffer of a buffer flagged as a
			 * discontinuity is a discontinuity.
			 */ \
			if(sinkbuf_discont && (offset + 1 - current_srcbuf_length == 0) && ((!(element->fill_discont)) || (element->last_sinkbuf_ets == 0))) \
				GST_BUFFER_FLAG_SET(srcbuf, GST_BUFFER_FLAG_DISCONT); \
			else \
				GST_BUFFER_FLAG_UNSET(srcbuf, GST_BUFFER_FLAG_DISCONT); \
			if(srcbuf_gap_next != srcbuf_gap) { \
				/* We need to reset our place in the input buffer */ \
				offset++; \
				indata++; \
				outdata++; \
				current_srcbuf_length = 1; \
				srcbuf_gap = srcbuf_gap_next; \
			} else \
				current_srcbuf_length = 0; \
 \
			/* push buffer downstream */ \
			GST_DEBUG_OBJECT(element, "pushing sub-buffer %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(srcbuf)); \
			result = gst_pad_push(element->srcpad, srcbuf); \
			if(G_UNLIKELY(result != GST_FLOW_OK)) { \
				GST_WARNING_OBJECT(element, "push failed: %s", gst_flow_get_name(result)); \
				goto done; \
			} \
		} \
	} \
done: \
	element->last_sinkbuf_ets = sinkbuf_pts + sinkbuf_dur; \
	element->last_sinkbuf_offset_end = sinkbuf_offset_end ? sinkbuf_offset_end : element->last_sinkbuf_offset_end; \
	return result; \
}


DEFINE_PROCESS_INBUF(guint32, )
DEFINE_PROCESS_INBUF(float, )
DEFINE_PROCESS_INBUF(double, )
DEFINE_PROCESS_INBUF(float,complex)
DEFINE_PROCESS_INBUF(double,complex)


/*
 * ============================================================================
 *
 *				     Sink Pad
 *
 * ============================================================================
 */


/*
 * sink_event()
 */


static gboolean sink_event(GstPad *pad, GstObject *parent, GstEvent *event)
{
	GSTLALInsertGap *element = GSTLAL_INSERTGAP(parent);
	gboolean success = TRUE;
	GST_DEBUG_OBJECT(pad, "Got %s event on sink pad", GST_EVENT_TYPE_NAME(event));

	switch(GST_EVENT_TYPE(event)) {
	case GST_EVENT_CAPS: {
		GstCaps *caps;
		GstAudioInfo info;
		gst_event_parse_caps(event, &caps);
		success &= gstlal_audio_info_from_caps(&info, caps);
		GstStructure *str = gst_caps_get_structure(caps, 0);
		const gchar *name = gst_structure_get_string(str, "format");
		success &= (name != NULL);
		if(success) {
			/* record stream parameters */
			element->rate = GST_AUDIO_INFO_RATE(&info);
			element->unit_size = GST_AUDIO_INFO_BPF(&info);
			if(!strcmp(name, GST_AUDIO_NE(U32))) {
				element->data_type = GSTLAL_INSERTGAP_U32;
				g_assert_cmpuint(element->unit_size, ==, 4);
			} else if(!strcmp(name, GST_AUDIO_NE(F32))) {
				element->data_type = GSTLAL_INSERTGAP_F32;
				g_assert_cmpuint(element->unit_size, ==, 4);
			} else if(!strcmp(name, GST_AUDIO_NE(F64))) {
				element->data_type = GSTLAL_INSERTGAP_F64;
				g_assert_cmpuint(element->unit_size, ==, 8);
			} else if(!strcmp(name, GST_AUDIO_NE(Z64))) {
				element->data_type = GSTLAL_INSERTGAP_Z64;
				g_assert_cmpuint(element->unit_size, ==, 8);
			} else if(!strcmp(name, GST_AUDIO_NE(Z128))) {
				element->data_type = GSTLAL_INSERTGAP_Z128;
				g_assert_cmpuint(element->unit_size, ==, 16);
			} else
				g_assert_not_reached();
		}
		break;
	}

	default:
		break;
	}

	if(!success) {
		gst_event_unref(event);
	} else {
		success = gst_pad_event_default(pad, parent, event);
	}
	return success;
}


/*
 * chain()
 */


static GstFlowReturn chain(GstPad *pad, GstObject *parent, GstBuffer *sinkbuf)
{
	GSTLALInsertGap *element = GSTLAL_INSERTGAP(parent);
	GstFlowReturn result = GST_FLOW_OK;
	GST_DEBUG_OBJECT(element, "received %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(sinkbuf));

	if(GST_BUFFER_PTS_IS_VALID(sinkbuf)) {
		/* Set the timestamp of the first output sample) */
		if(element->t0 == GST_CLOCK_TIME_NONE)
			element->t0 = GST_BUFFER_PTS(sinkbuf) + element->chop_length;

		/* If we are throwing away any initial data, do it now, and send a zero-length buffer downstream to let other elements know when to expect the first buffer */
		if(element->chop_length && GST_BUFFER_PTS(sinkbuf) + GST_BUFFER_DURATION(sinkbuf) <= element->t0) {
			GstBuffer *srcbuf = gst_buffer_new();
			GST_BUFFER_OFFSET(srcbuf) = GST_BUFFER_OFFSET(sinkbuf) + gst_util_uint64_scale_round(element->t0 - GST_BUFFER_PTS(sinkbuf), (guint64) element->rate, 1000000000);
			GST_BUFFER_OFFSET_END(srcbuf) = GST_BUFFER_OFFSET(srcbuf);
			GST_BUFFER_PTS(srcbuf) = element->t0;
			GST_BUFFER_DURATION(srcbuf) = 0;
			result = gst_pad_push(element->srcpad, srcbuf);
			gst_buffer_unref(sinkbuf);
			goto done;
		} else if(GST_BUFFER_PTS(sinkbuf) < element->t0) {
			guint64 size_removed = element->unit_size * gst_util_uint64_scale_round(element->t0 - GST_BUFFER_PTS(sinkbuf), (guint64) element->rate, 1000000000);
			guint64 time_removed = gst_util_uint64_scale_round(size_removed / element->unit_size, 1000000000, (guint64) element->rate);
			guint64 newsize = element->unit_size * (GST_BUFFER_OFFSET_END(sinkbuf) - GST_BUFFER_OFFSET(sinkbuf)) - size_removed;
			gst_buffer_resize(sinkbuf, size_removed, newsize);
			GST_BUFFER_OFFSET(sinkbuf) = GST_BUFFER_OFFSET(sinkbuf) + size_removed / element->unit_size;
			GST_BUFFER_PTS(sinkbuf) = GST_BUFFER_PTS(sinkbuf) + time_removed;
			GST_BUFFER_DURATION(sinkbuf) = GST_BUFFER_DURATION(sinkbuf) - time_removed;
		}
	}

	/* if buffer does not possess valid metadata or is zero length and we are not filling in discontinuities, push gap downstream */
	if(!(GST_BUFFER_PTS_IS_VALID(sinkbuf) && GST_BUFFER_DURATION_IS_VALID(sinkbuf) && GST_BUFFER_OFFSET_IS_VALID(sinkbuf) && GST_BUFFER_OFFSET_END_IS_VALID(sinkbuf)) || (!element->fill_discont && (GST_BUFFER_DURATION(sinkbuf) == 0 || GST_BUFFER_OFFSET(sinkbuf) == GST_BUFFER_OFFSET_END(sinkbuf)))) {
		GST_DEBUG_OBJECT(element, "pushing gap buffer at timestamp %lu seconds", (long unsigned) GST_TIME_AS_SECONDS(GST_BUFFER_PTS(sinkbuf)));
		GstBuffer *srcbuf;
		srcbuf = gst_buffer_copy(sinkbuf);
		GST_BUFFER_FLAG_SET(srcbuf, GST_BUFFER_FLAG_GAP);
		result = gst_pad_push(element->srcpad, srcbuf);
		if(G_UNLIKELY(result != GST_FLOW_OK))
			GST_WARNING_OBJECT(element, "push failed: %s", gst_flow_get_name(result));
		goto done;
	}

	/* if buffer is zero length and we are filling in discontinuities, fill it in, unless it has no valid timestamp. */
	if(element->fill_discont && (GST_BUFFER_DURATION(sinkbuf) == 0 || GST_BUFFER_OFFSET(sinkbuf) == GST_BUFFER_OFFSET_END(sinkbuf))) {
		if(GST_BUFFER_PTS_IS_VALID(sinkbuf) && GST_BUFFER_PTS(sinkbuf) > element->last_sinkbuf_ets) {
			switch(element->data_type) {
			case GSTLAL_INSERTGAP_U32:
				result = process_inbuf_guint32(NULL, NULL, element, TRUE, TRUE, 0, 0, 0, GST_BUFFER_PTS(sinkbuf), FALSE);
				break;
			case GSTLAL_INSERTGAP_F32:
				result = process_inbuf_float(NULL, NULL, element, TRUE, TRUE, 0, 0, 0, GST_BUFFER_PTS(sinkbuf), FALSE);
				break;
			case GSTLAL_INSERTGAP_F64:
				result = process_inbuf_double(NULL, NULL, element, TRUE, TRUE, 0, 0, 0, GST_BUFFER_PTS(sinkbuf), FALSE);
				break;
			case GSTLAL_INSERTGAP_Z64:
				result = process_inbuf_floatcomplex(NULL, NULL, element, TRUE, TRUE, 0, 0, 0, GST_BUFFER_PTS(sinkbuf), TRUE);
				break;
			case GSTLAL_INSERTGAP_Z128:
				result = process_inbuf_doublecomplex(NULL, NULL, element, TRUE, TRUE, 0, 0, 0, GST_BUFFER_PTS(sinkbuf), TRUE);
				break;
			default:
				g_assert_not_reached();
			}
		} else {
			GST_DEBUG_OBJECT(element, "dropping zero length buffer at timestamp %lu seconds", (long unsigned) GST_TIME_AS_SECONDS(GST_BUFFER_PTS(sinkbuf)));
			gst_buffer_unref(sinkbuf);
		}
		goto done;
	}

	GstMapInfo inmap;
	gst_buffer_map(sinkbuf, &inmap, GST_MAP_READ);

	/* We'll need these to decide gaps, offsets, and timestamps on the outgoing buffer(s) */
	gboolean sinkbuf_gap = GST_BUFFER_FLAG_IS_SET(sinkbuf, GST_BUFFER_FLAG_GAP);
	gboolean sinkbuf_discont = GST_BUFFER_FLAG_IS_SET(sinkbuf, GST_BUFFER_FLAG_DISCONT);
	guint64 sinkbuf_offset = GST_BUFFER_OFFSET(sinkbuf);
	guint64 sinkbuf_offset_end = GST_BUFFER_OFFSET_END(sinkbuf);
	GstClockTime sinkbuf_dur = GST_BUFFER_DURATION(sinkbuf);
	GstClockTime sinkbuf_pts = GST_BUFFER_PTS(sinkbuf);

	g_assert_cmpuint(inmap.size % element->unit_size, ==, 0);
	if(!element->chop_length || sinkbuf_pts > element->t0)
		g_assert_cmpuint((sinkbuf_offset_end - sinkbuf_offset), ==, inmap.size / element->unit_size); /* sanity checks */

	/* outdata will be filled with the data that goes on the outgoing buffer(s) */
	void *outdata;
	outdata = g_malloc((sinkbuf_offset_end - sinkbuf_offset) * element->unit_size);

	switch(element->data_type) {
	case GSTLAL_INSERTGAP_U32:
		result = process_inbuf_guint32((guint32 *) inmap.data, outdata, element, sinkbuf_gap, sinkbuf_discont, sinkbuf_offset, sinkbuf_offset_end, sinkbuf_dur, sinkbuf_pts, FALSE);
		break;
	case GSTLAL_INSERTGAP_F32:
		result = process_inbuf_float((float *) inmap.data, outdata, element, sinkbuf_gap, sinkbuf_discont, sinkbuf_offset, sinkbuf_offset_end, sinkbuf_dur, sinkbuf_pts, FALSE);
		break;
	case GSTLAL_INSERTGAP_F64:
		result = process_inbuf_double((double *) inmap.data, outdata, element, sinkbuf_gap, sinkbuf_discont, sinkbuf_offset, sinkbuf_offset_end, sinkbuf_dur, sinkbuf_pts, FALSE);
		break;
	case GSTLAL_INSERTGAP_Z64:
		result = process_inbuf_floatcomplex((float complex *) inmap.data, outdata, element, sinkbuf_gap, sinkbuf_discont, sinkbuf_offset, sinkbuf_offset_end, sinkbuf_dur, sinkbuf_pts, TRUE);
		break;
	case GSTLAL_INSERTGAP_Z128:
		result = process_inbuf_doublecomplex((double complex *) inmap.data, outdata, element, sinkbuf_gap, sinkbuf_discont, sinkbuf_offset, sinkbuf_offset_end, sinkbuf_dur, sinkbuf_pts, TRUE);
		break;

	default:
		g_assert_not_reached();
	}

	g_free(outdata);
	outdata = NULL;
	gst_buffer_unmap(sinkbuf, &inmap);
	gst_buffer_unref(sinkbuf);
	/*
	 * done
	 */

done:
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
	ARG_INSERT_GAP = 1,
	ARG_REMOVE_GAP,
	ARG_REMOVE_NAN,
	ARG_REMOVE_INF,
	ARG_FILL_DISCONT,
	ARG_REPLACE_VALUE,
	ARG_BAD_DATA_INTERVALS,
	ARG_BLOCK_DURATION,
	ARG_CHOP_LENGTH
};


static void set_property(GObject *object, enum property prop_id, const GValue *value, GParamSpec *pspec)
{
	GSTLALInsertGap *element = GSTLAL_INSERTGAP(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_INSERT_GAP:
		element->insert_gap = g_value_get_boolean(value);
		break;
	case ARG_REMOVE_GAP:
		element->remove_gap = g_value_get_boolean(value);
		break;
	case ARG_REMOVE_NAN:
		element->remove_nan = g_value_get_boolean(value);
		break;
	case ARG_REMOVE_INF:
		element->remove_inf = g_value_get_boolean(value);
		break;
	case ARG_FILL_DISCONT:
		element->fill_discont = g_value_get_boolean(value);
		break;
	case ARG_REPLACE_VALUE:
		element->replace_value = g_value_get_double(value);
		break;
	case ARG_BAD_DATA_INTERVALS: {
		GValueArray *va = g_value_get_boxed(value);
		element->bad_data_intervals = g_malloc(16 * sizeof(double));
		element->array_length = 1;
		gstlal_doubles_from_g_value_array(va, element->bad_data_intervals, &element->array_length);
		if(element->array_length % 2)
			GST_ERROR_OBJECT(element, "Array length for property bad_data_intervals must be even");
		break;
	}
	case ARG_BLOCK_DURATION:
		element->block_duration = g_value_get_uint64(value);
		break;
	case ARG_CHOP_LENGTH:
		element->chop_length = g_value_get_uint64(value);
		break;
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, enum property prop_id, GValue *value, GParamSpec *pspec)
{
	GSTLALInsertGap *element = GSTLAL_INSERTGAP(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_INSERT_GAP:
		g_value_set_boolean(value, element->insert_gap);
		break;
	case ARG_REMOVE_GAP:
		g_value_set_boolean(value, element->remove_gap);
		break;
	case ARG_REMOVE_NAN:
		g_value_set_boolean(value, element->remove_nan);
		break;
	case ARG_REMOVE_INF:
		g_value_set_boolean(value, element->remove_inf);
		break;
	case ARG_FILL_DISCONT:
		g_value_set_boolean(value, element->fill_discont);
		break;
	case ARG_REPLACE_VALUE:
		g_value_set_double(value, element->replace_value);
		break;
	case ARG_BAD_DATA_INTERVALS:
		g_value_take_boxed(value, gstlal_g_value_array_from_doubles(element->bad_data_intervals, element->array_length));
		break;
	case ARG_BLOCK_DURATION:
		g_value_set_uint64(value, element->block_duration);
		break;
	case ARG_CHOP_LENGTH:
		g_value_set_uint64(value, element->chop_length);
		break;
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * finalize()
 */


static void finalize(GObject *object)
{
	GSTLALInsertGap *element = GSTLAL_INSERTGAP(object);

	gst_object_unref(element->sinkpad);
	element->sinkpad = NULL;
	gst_object_unref(element->srcpad);
	element->srcpad = NULL;
	g_free(element->bad_data_intervals);
	element->bad_data_intervals = NULL;

	G_OBJECT_CLASS(gstlal_insertgap_parent_class)->finalize(object);
}


/*
 * class_init()
 */


static void gstlal_insertgap_class_init(GSTLALInsertGapClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);

	gst_element_class_set_details_simple(
		element_class,
		"Replace unwanted data with gaps",
		"Filter",
		"Replace unwanted data, specified with the property bad-data-intervals, with gaps.\n\t\t\t   "
		"Also can replace with another value, given by replace-value. Can also remove gaps\n\t\t\t   "
		"where data is acceptable, and fill in discontinuities if desired.",
		"Aaron Viets <aaron.viets@ligo.org>"
	);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));

	g_object_class_install_property(
		gobject_class,
		ARG_INSERT_GAP,
		g_param_spec_boolean(
			"insert-gap",
			"Insert gap",
			"If set to true (default), any data fitting the criteria specified by the property\n\t\t\t"
			"bad-data-intervals is replaced with gaps. Also, NaN's and inf's are replaced with\n\t\t\t"
			"gaps if the properties remove-nan and remove-inf are set to true, respectively.",
			TRUE,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);

	g_object_class_install_property(
		gobject_class,
		ARG_REMOVE_GAP,
		g_param_spec_boolean(
			"remove-gap",
			"Remove gap",
			"If set to true, any data in an input gap buffer that does not fit the criteria\n\t\t\t"
			"specified by the property bad-data-intervals will be marked as non-gap. If the\n\t\t\t"
			"property insert-gap is false and remove-gap is true, gaps with unacceptable\n\t\t\t"
			"data will be replaced by the value specified by the property replace-value.",
			FALSE,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);

	g_object_class_install_property(
		gobject_class,
		ARG_REMOVE_NAN,
		g_param_spec_boolean(
			"remove-nan",
			"Remove NaN",
			"If set to true (default), NaN's in the data stream will be replaced with gaps\n\t\t\t"
			"and/or the replace-value, as specified by user.",
			TRUE,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);

	g_object_class_install_property(
		gobject_class,
		ARG_REMOVE_INF,
		g_param_spec_boolean(
			"remove-inf",
			"Remove inf",
			"If set to true (default), infinities in the data stream will be replaced with\n\t\t\t"
			"gaps and/or the replace-value, as specified by user.",
			TRUE,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);

	g_object_class_install_property(
		gobject_class,
		ARG_FILL_DISCONT,
		g_param_spec_boolean(
			"fill-discont",
			"Fill discontinuity",
			"If set to true, discontinuities in the data stream will be filled with the\n\t\t\t"
			"replace-value (if set, otherwise 0), and gapped if insert-gap is true.",
			FALSE,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);

	g_object_class_install_property(
		gobject_class,
		ARG_REPLACE_VALUE,
		g_param_spec_double(
			"replace-value",
			"Replace value",
			"If set, this value is used to replace any data that fits the criteria\n\t\t\t"
			"specified by the property bad-data-intervals. If unset, values are not replaced.", 
			-G_MAXDOUBLE, G_MAXDOUBLE, G_MAXDOUBLE,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);

	g_object_class_install_property(
		gobject_class,
		ARG_BAD_DATA_INTERVALS,
		g_param_spec_value_array(
			"bad-data-intervals",
			"Bad data intervals",
			"Array of at most 16 elements containing minima and maxima of closed intervals\n\t\t\t"
			"in which data is considered unacceptable and will be replaced with gaps and/or\n\t\t\t"
			"the replace-value. Array indices 0, 2, 4, etc., represent maxima, and\n\t\t\t"
			"array indices 1, 3, 5, etc., represent the corresponding minima.",
			g_param_spec_double(
				"coefficient",
				"Coefficient",
				"Coefficient",
				-G_MAXDOUBLE, G_MAXDOUBLE, 0.0,
				G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
			),
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);

	g_object_class_install_property(
		gobject_class,
		ARG_BLOCK_DURATION,
		g_param_spec_uint64(
			"block-duration",
			"Block duration",
			"Maximum output buffer duration in nanoseconds. Buffers may be smaller than this.\n\t\t\t"
			"Default is to not change buffer length except as required by added/removed gaps.",
			0, G_MAXUINT64, G_MAXUINT64 / 2,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);

	g_object_class_install_property(
		gobject_class,
		ARG_CHOP_LENGTH,
		g_param_spec_uint64(
			"chop-length",
			"Chop length",
			"Amount of initial data to throw away before producing output data, in nanoseconds.",
			0, G_MAXUINT64, 0,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
}


/*
 * instance init
 */


static void gstlal_insertgap_init(GSTLALInsertGap *element)
{
	GstPad *pad;

	gst_element_create_all_pads(GST_ELEMENT(element));

	/* configure (and ref) sink pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "sink");
	gst_pad_set_event_function(pad, GST_DEBUG_FUNCPTR(sink_event));
	gst_pad_set_chain_function(pad, GST_DEBUG_FUNCPTR(chain));
	GST_PAD_SET_PROXY_CAPS(pad);
	GST_PAD_SET_PROXY_ALLOCATION(pad);
	GST_PAD_SET_PROXY_SCHEDULING(pad);
	element->sinkpad = pad;

	/* retrieve (and ref) src pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "src");
	GST_PAD_SET_PROXY_CAPS(pad);
	GST_PAD_SET_PROXY_ALLOCATION(pad);
	GST_PAD_SET_PROXY_SCHEDULING(pad);
	element->srcpad = pad;

	/* internal data */
	element->t0 = GST_CLOCK_TIME_NONE;
	element->bad_data_intervals = NULL;
	element->array_length = 0;
	element->rate = 0;
	element->unit_size = 0;
	element->last_sinkbuf_ets = 0;
	element->last_sinkbuf_offset_end = 0;
	element->discont_offset = 0;
	element->discont_time = 0;
	element->empty_bufs = 0;
}

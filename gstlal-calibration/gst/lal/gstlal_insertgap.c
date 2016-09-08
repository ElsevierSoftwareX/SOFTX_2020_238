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
 * represent minima, and array indices 1, 3, 5, etc., represent the 
 * corresponding maxima. For example, if 
 * #bad_data_intervals = [0, 1, 9, 10],
 * then any values that fall either in the closed interval [0, 1] or in 
 * the closed interval [9, 10] will be rejected and gapped and/or replaced
 * as specified by the user. To reject a single value, say zero,
 * #bad_data_intervals should be chosen to be [0, 0].
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
		"format = (string) { " GST_AUDIO_NE(F32) ", " GST_AUDIO_NE(F64) " }, " \
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
		"format = (string) { " GST_AUDIO_NE(F32) ", " GST_AUDIO_NE(F64) " }, " \
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


static gboolean check_data(double data, double *bad_data_intervals, gint array_length, gboolean remove_nan, gboolean remove_inf) {
	gboolean data_is_bad = FALSE;
	gint i;
	for(i = 0; i < array_length; i += 2) {
		if(data >= bad_data_intervals[i] && data <= bad_data_intervals[i + 1])
			data_is_bad = TRUE;
	}
	if(remove_nan && isnan(data))
		data_is_bad = TRUE;
	if(remove_inf && isinf(data))
		data_is_bad = TRUE;

	return data_is_bad;
}


#define DEFINE_PROCESS_INBUF(DTYPE) \
static GstFlowReturn process_inbuf_ ## DTYPE(const DTYPE *indata, DTYPE *outdata, guint64 length, guint64 max_block_length, double *bad_data_intervals, gint array_length, gboolean remove_nan, gboolean remove_inf, gboolean insert_gap, gboolean remove_gap, double replace_value, GstPad *srcpad, gboolean sinkbuf_gap, gboolean sinkbuf_discont, guint64 sinkbuf_offset, GstClockTime sinkbuf_dur, GstClockTime sinkbuf_pts) \
{ \
	GstFlowReturn result = GST_FLOW_OK; \
	gboolean data_is_bad, srcbuf_gap, srcbuf_gap_next; \
	guint64 offset, current_srcbuf_length; \
	current_srcbuf_length = 0; \
	data_is_bad = check_data((double) *indata, bad_data_intervals, array_length, remove_nan, remove_inf); \
	srcbuf_gap = (sinkbuf_gap && (!remove_gap)) || (insert_gap && data_is_bad); \
	for(offset = 0; offset < length; offset++) { \
		data_is_bad = check_data((double) *indata, bad_data_intervals, array_length, remove_nan, remove_inf); \
		srcbuf_gap_next = (sinkbuf_gap && (!remove_gap)) || (insert_gap && data_is_bad); \
		*outdata = ((replace_value < G_MAXDOUBLE) && data_is_bad) ? (DTYPE) replace_value : *indata; \
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
			DTYPE *srcbuf_data; \
			srcbuf_data = g_malloc(current_srcbuf_length * sizeof(DTYPE)); \
			memcpy(srcbuf_data, (outdata - current_srcbuf_length), current_srcbuf_length * sizeof(DTYPE)); \
			srcbuf = gst_buffer_new_wrapped(srcbuf_data, current_srcbuf_length * sizeof(DTYPE)); \
 \
			if(G_UNLIKELY(!srcbuf)) { \
				GST_ERROR("failure creating sub-buffer"); \
				result = GST_FLOW_ERROR; \
				break; \
			} \
 \
			/* set flags, caps, offset, and timestamps. */ \
			GST_BUFFER_OFFSET(srcbuf) = sinkbuf_offset + offset - current_srcbuf_length + 1; \
			GST_BUFFER_OFFSET_END(srcbuf) = GST_BUFFER_OFFSET(srcbuf) + current_srcbuf_length; \
			GST_BUFFER_PTS(srcbuf) = sinkbuf_pts + gst_util_uint64_scale_int_round(sinkbuf_dur, offset - current_srcbuf_length + 1, length); \
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
			if((offset + 1 - current_srcbuf_length == 0) && sinkbuf_discont) \
				GST_BUFFER_FLAG_SET(srcbuf, GST_BUFFER_FLAG_DISCONT); \
 \
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
			GST_DEBUG("pushing sub-buffer %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(srcbuf)); \
			result = gst_pad_push(srcpad, srcbuf); \
			if(G_UNLIKELY(result != GST_FLOW_OK)) { \
				GST_WARNING("push failed: %s", gst_flow_get_name(result)); \
				break; \
			} \
		} \
	} \
	return result; \
}


DEFINE_PROCESS_INBUF(float)
DEFINE_PROCESS_INBUF(double)


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
		success = gst_audio_info_from_caps(&info, caps);
		if(success) {
			element->rate = GST_AUDIO_INFO_RATE(&info);
			element->unit_size = GST_AUDIO_INFO_BPF(&info);
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
	guint64 length;
	guint64 blocks, max_block_length;
	GstFlowReturn result = GST_FLOW_OK;
	GST_DEBUG_OBJECT(element, "received %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(sinkbuf));

	/* if buffer does not possess valid metadata, push gap downstream */
	if(!(GST_BUFFER_PTS_IS_VALID(sinkbuf) && GST_BUFFER_DURATION_IS_VALID(sinkbuf) && GST_BUFFER_OFFSET_IS_VALID(sinkbuf) && GST_BUFFER_OFFSET_END_IS_VALID(sinkbuf))) {
		/* FIXME: What is the best course of action in this case? */
		GST_DEBUG_OBJECT(element, "pushing gap buffer");
		GstBuffer *srcbuf;
		srcbuf = gst_buffer_copy(sinkbuf);
		GST_BUFFER_FLAG_SET(srcbuf, GST_BUFFER_FLAG_GAP);
		result = gst_pad_push(element->srcpad, srcbuf);
		if(G_UNLIKELY(result != GST_FLOW_OK))
			GST_WARNING_OBJECT(element, "push failed: %s", gst_flow_get_name(result));
		goto done;
	}

	/* compute length of incoming buffer and maximum block length in samples */
	blocks = (GST_BUFFER_DURATION(sinkbuf) + element->block_duration - 1) / element->block_duration; /* ceil */
	g_assert_cmpuint(blocks, >, 0); /* make sure that the sinkbuf is not zero length */
	length = GST_BUFFER_OFFSET_END(sinkbuf) - GST_BUFFER_OFFSET(sinkbuf);
	max_block_length = (length + blocks - 1) / blocks; /* ceil */
	g_assert_cmpuint(max_block_length, >, 0);

	GstMapInfo inmap;
	gst_buffer_map(sinkbuf, &inmap, GST_MAP_READ);

	g_assert_cmpuint(inmap.size % element->unit_size, ==, 0);
	g_assert_cmpuint(length, ==, inmap.size / element->unit_size); /*sanity checks */

	/* We'll need these to decide gaps, offsets, and timestamps on the outgoing buffers */
	gboolean sinkbuf_gap = GST_BUFFER_FLAG_IS_SET(sinkbuf, GST_BUFFER_FLAG_GAP);
	gboolean sinkbuf_discont = GST_BUFFER_FLAG_IS_SET(sinkbuf, GST_BUFFER_FLAG_DISCONT);
	guint64 sinkbuf_offset = GST_BUFFER_OFFSET(sinkbuf);
	GstClockTime sinkbuf_dur = GST_BUFFER_DURATION(sinkbuf);
	GstClockTime sinkbuf_pts = GST_BUFFER_PTS(sinkbuf);;

	/* outdata will be filled with the data that goes on the outgoing buffer(s) */
	void *outdata;
	outdata = g_malloc(length * element->unit_size);

	switch(element->unit_size) {
	case 4:
		result = process_inbuf_float((float *) inmap.data, outdata, length, max_block_length, element->bad_data_intervals, element->array_length, element->remove_nan, element->remove_inf, element->insert_gap, element->remove_gap, element->replace_value, element->srcpad, sinkbuf_gap, sinkbuf_discont, sinkbuf_offset, sinkbuf_dur, sinkbuf_pts);
		break;
	case 8:
		result = process_inbuf_double((double *) inmap.data, outdata, length, max_block_length, element->bad_data_intervals, element->array_length, element->remove_nan, element->remove_inf, element->insert_gap, element->remove_gap, element->replace_value, element->srcpad, sinkbuf_gap, sinkbuf_discont, sinkbuf_offset, sinkbuf_dur, sinkbuf_pts);
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
	ARG_REPLACE_VALUE,
	ARG_BAD_DATA_INTERVALS,
	ARG_BLOCK_DURATION
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
	case ARG_REPLACE_VALUE:
		element->replace_value = g_value_get_double(value);
		break;
	case ARG_BAD_DATA_INTERVALS: {
		GValueArray *va = g_value_get_boxed(value);
		element->bad_data_intervals = g_malloc(16 * sizeof(double));
		element->array_length = 1;
		gstlal_doubles_from_g_value_array(va, element->bad_data_intervals, &element->array_length);
		break;
	}
	case ARG_BLOCK_DURATION:
		element->block_duration = g_value_get_uint64(value);
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
	case ARG_REPLACE_VALUE:
		g_value_set_double(value, element->replace_value);
		break;
	case ARG_BAD_DATA_INTERVALS:
		g_value_take_boxed(value, gstlal_g_value_array_from_doubles(element->bad_data_intervals, element->array_length));
		break;
	case ARG_BLOCK_DURATION:
		g_value_set_uint64(value, element->block_duration);
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
		"Replace unwanted data, specified with the property bad-data-intervals, with gaps. Also can replace with another value, given by replace-value. Can also remove gaps where data is acceptable.",
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
			"If set to true (default), any data fitting the criteria specified by the property bad-data-intervals is replaced with gaps. Also, NaN's and inf's are replaced with gaps if the properties remove-nan and remove-inf are set to true, respectively.",
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
			"If set to true, any data in an input gap buffer that does not fit the criteria specified by the property bad-data-intervals will be marked as non-gap. If the property insert-gap is false and remove-gap is true, gaps with unacceptable data will be replaced by the value specified by the property replace-value.",
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
			"If set to true (default), NaN's in the data stream will be replaced with gaps and/or the replace-value, as specified by user.",
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
			"If set to true (default), infinities in the data stream will be replaced with gaps and/or the replace-value, as specified by user.",
			TRUE,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);

	g_object_class_install_property(
		gobject_class,
		ARG_REPLACE_VALUE,
		g_param_spec_double(
			"replace-value",
			"Replace value",
			"If set, this value is used to replace any data that fits the criteria specified by the property bad-data-intervals. If unset, values are not replaced.", 
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
			"Array of at most 16 elements containing minima and maxima of closed intervals in which data is considered unacceptable and will be replaced with gaps and/or the replace-value. Array indices 0, 2, 4, etc., represent minima, and array indices 1, 3, 5, etc., represent the corresponding maxima.",
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
			"Maximum output buffer duration in nanoseconds. Buffers may be smaller than this. Default is to not change buffer length except as required by added/removed gaps.",
			0, G_MAXUINT64, G_MAXUINT64 / 2,
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
	element->rate = 0;
	element->unit_size = 0;
}

/*
 * GstAudioRateFaker
 *
 * Copyright (C) 2013,2014  Kipp Cannon
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
 *                                  Preamble
 *
 * ============================================================================
 */


#include <math.h>


#include <glib.h>
#include <gst/gst.h>
#include <gst/audio/audio.h>
#include <gst/base/gstbasetransform.h>


#include <audioratefaker.h>


/*
 * ============================================================================
 *
 *                                Boilerplate
 *
 * ============================================================================
 */


#define GST_CAT_DEFAULT gst_audio_rate_faker_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


static void additional_initializations(void)
{
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "audioratefaker", 0, "audioratefaker element");
}


G_DEFINE_TYPE_WITH_CODE(GstAudioRateFaker, gst_audio_rate_faker, GST_TYPE_BASE_TRANSFORM, additional_initializations(););


/*
 * ============================================================================
 *
 *                             Internal Functions
 *
 * ============================================================================
 */


static gboolean do_new_segment(GstAudioRateFaker *element)
{
	gboolean success = TRUE;

	if(element->last_segment) {
		GstSegment segment;

		gst_event_copy_segment(element->last_segment, &segment);

		if(segment.format == GST_FORMAT_TIME) {
			if(GST_CLOCK_TIME_IS_VALID(segment.start))
				segment.start = gst_util_uint64_scale_int_round(segment.start, element->inrate_over_outrate_num, element->inrate_over_outrate_den);
			if(GST_CLOCK_TIME_IS_VALID(segment.stop))
				segment.stop = gst_util_uint64_scale_int_round(segment.stop, element->inrate_over_outrate_num, element->inrate_over_outrate_den);
			if(GST_CLOCK_TIME_IS_VALID(segment.position))
				segment.position = gst_util_uint64_scale_int_round(segment.position, element->inrate_over_outrate_num, element->inrate_over_outrate_den);
			success = gst_pad_push_event(GST_BASE_TRANSFORM_SRC_PAD(element), gst_event_new_segment(&segment));
		} else {
			gst_event_ref(element->last_segment);
			success = gst_pad_push_event(GST_BASE_TRANSFORM_SRC_PAD(element), element->last_segment);
		}

		element->need_new_segment = FALSE;
	}

	return success;
}


/*
 * ============================================================================
 *
 *                          GstBaseTransform Methods
 *
 * ============================================================================
 */


static GstCaps *transform_caps(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, GstCaps *filter)
{
	GstCaps *othercaps = NULL;
	guint i;

	GST_DEBUG_OBJECT(trans, "transforming %" GST_PTR_FORMAT " on %s pad with filter %" GST_PTR_FORMAT, caps, direction == GST_PAD_SRC ? GST_BASE_TRANSFORM_SRC_NAME : GST_BASE_TRANSFORM_SINK_NAME, filter);

	/*
	 * sink and source pads must have same channel count and sample
	 * format.  different rates are permitted, but we should try to
	 * operate in pass-through mode if possible.
	 */

	/* make a copy of caps with all rate elements removed */
	othercaps = gst_caps_copy(caps);
	for(i = 0; i < gst_caps_get_size(othercaps); i++)
		gst_structure_remove_field(gst_caps_get_structure(othercaps, i), "rate");
	/* append the result to a copy of caps, and free.  having the
	 * original caps appear first informs peers of our desire to
	 * operate in pass-through mode */
	caps = gst_caps_copy(caps);
	gst_caps_append(caps, othercaps);

	/* intersect that result with the caps allowed by the pad template.
	 * this repopulates the rate elements with the allowed ranges */
	switch(direction) {
	case GST_PAD_SRC: {
		GstCaps *tmpltcaps = gst_pad_get_pad_template_caps(GST_BASE_TRANSFORM_SINK_PAD(trans));
		othercaps = gst_caps_intersect_full(caps, tmpltcaps, GST_CAPS_INTERSECT_FIRST);
		gst_caps_unref(tmpltcaps);
		break;
	}

	case GST_PAD_SINK: {
		GstCaps *tmpltcaps = gst_pad_get_pad_template_caps(GST_BASE_TRANSFORM_SRC_PAD(trans));
		othercaps = gst_caps_intersect_full(caps, tmpltcaps, GST_CAPS_INTERSECT_FIRST);
		gst_caps_unref(tmpltcaps);
		break;
	}

	default:
		g_assert_not_reached();
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		gst_caps_ref(GST_CAPS_NONE);
		othercaps = GST_CAPS_NONE;
		break;
	}
	gst_caps_unref(caps);

	if(filter) {
		caps = othercaps;
		othercaps = gst_caps_intersect_full(filter, othercaps, GST_CAPS_INTERSECT_FIRST);
		gst_caps_unref(caps);
	}

	GST_DEBUG_OBJECT(trans, "transformed to %" GST_PTR_FORMAT, othercaps);

	return othercaps;
}


static GstCaps *fixate_caps(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, GstCaps *othercaps)
{
	GstStructure *s;
	gint rate_num, rate_den = 1;

	GST_DEBUG_OBJECT(trans, "fixating %s caps %" GST_PTR_FORMAT " (other pad is %" GST_PTR_FORMAT ")", direction == GST_PAD_SRC ? GST_BASE_TRANSFORM_SRC_NAME : GST_BASE_TRANSFORM_SINK_NAME, othercaps, caps);

	s = gst_caps_get_structure(caps, 0);
	if(!gst_structure_get_int(s, "rate", &rate_num))
		if(!gst_structure_get_fraction(s, "rate", &rate_num, &rate_den)) {
			GST_ERROR_OBJECT(trans, "could not deduce rate from %" GST_PTR_FORMAT, caps);
			goto done;
		}

	othercaps = gst_caps_truncate(othercaps);	/* does this leak memory? */
	s = gst_caps_get_structure(othercaps, 0);
	if(gst_structure_has_field_typed(s, "rate", G_TYPE_INT))
		gst_structure_fixate_field_nearest_int(s, "rate", rate_den == 1 ? rate_num : (int) round((double) rate_num / rate_den));
	else if(gst_structure_has_field_typed(s, "rate", GST_TYPE_FRACTION))
		gst_structure_fixate_field_nearest_fraction(s, "rate", rate_num, rate_den);

done:
	return othercaps;
}


static gboolean set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps)
{
	GstAudioRateFaker *element = GST_AUDIO_RATE_FAKER(trans);
	GstStructure *s;
	gint inrate_num, inrate_den = 1;
	gint outrate_num, outrate_den = 1;
	gboolean success = TRUE;

	s = gst_caps_get_structure(incaps, 0);
	if(!gst_structure_get_int(s, "rate", &inrate_num))
		success &= gst_structure_get_fraction(s, "rate", &inrate_num, &inrate_den);

	s = gst_caps_get_structure(outcaps, 0);
	if(!gst_structure_get_int(s, "rate", &outrate_num))
		success &= gst_structure_get_fraction(s, "rate", &outrate_num, &outrate_den);

	if(success) {
		gst_util_fraction_multiply(inrate_num, inrate_den, outrate_den, outrate_num, &element->inrate_over_outrate_num, &element->inrate_over_outrate_den);
		GST_DEBUG_OBJECT(element, "in rate / out rate = %d/%d", element->inrate_over_outrate_num, element->inrate_over_outrate_den);
		do_new_segment(element);
	} else
		GST_ERROR_OBJECT(element, "failed to parse rates from incaps = %" GST_PTR_FORMAT ", outcaps = %" GST_PTR_FORMAT, incaps, outcaps);

	return success;
}


static gboolean sink_event(GstBaseTransform *trans, GstEvent *event)
{
	GstAudioRateFaker *element = GST_AUDIO_RATE_FAKER(trans);
	gboolean success = TRUE;

	switch(GST_EVENT_TYPE(event)) {
	case GST_EVENT_SEGMENT:
		if(element->last_segment)
			gst_event_unref(element->last_segment);
		element->last_segment = event;
		element->need_new_segment = TRUE;
		break;

	default:
		success = GST_BASE_TRANSFORM_CLASS(gst_audio_rate_faker_parent_class)->sink_event(trans, event);
		break;
	}

	return success;
}


static GstFlowReturn transform_ip(GstBaseTransform *trans, GstBuffer *buf)
{
	GstAudioRateFaker *element = GST_AUDIO_RATE_FAKER(trans);
	GstFlowReturn result = GST_FLOW_OK;

	if(element->need_new_segment)
		do_new_segment(element);

	if(GST_BUFFER_PTS_IS_VALID(buf) && GST_BUFFER_DURATION_IS_VALID(buf)) {
		GstClockTime timestamp = GST_BUFFER_PTS(buf);
		GstClockTime duration = GST_BUFFER_DURATION(buf);

		GST_BUFFER_PTS(buf) = gst_util_uint64_scale_int_round(timestamp, element->inrate_over_outrate_num, element->inrate_over_outrate_den);
		GST_BUFFER_DURATION(buf) = gst_util_uint64_scale_int_round(timestamp + duration, element->inrate_over_outrate_num, element->inrate_over_outrate_den) - GST_BUFFER_PTS(buf);
	} else if(GST_BUFFER_PTS_IS_VALID(buf))
		GST_BUFFER_PTS(buf) = gst_util_uint64_scale_int_round(GST_BUFFER_PTS(buf), element->inrate_over_outrate_num, element->inrate_over_outrate_den);
	else if(GST_BUFFER_DURATION_IS_VALID(buf))
		GST_BUFFER_DURATION(buf) = gst_util_uint64_scale_int_round(GST_BUFFER_DURATION(buf), element->inrate_over_outrate_num, element->inrate_over_outrate_den);
	if(GST_BUFFER_DTS_IS_VALID(buf))
		GST_BUFFER_DTS(buf) = gst_util_uint64_scale_int_round(GST_BUFFER_DTS(buf), element->inrate_over_outrate_num, element->inrate_over_outrate_den);

	return result;
}


/*
 * ============================================================================
 *
 *                              GObject Methods
 *
 * ============================================================================
 */


static void finalize(GObject *object)
{
	GstAudioRateFaker *element = GST_AUDIO_RATE_FAKER(object);

	if(element->last_segment)
		gst_event_unref(element->last_segment);
	element->last_segment = NULL;

	/*
	 * chain to parent class' finalize() method
	 */

	G_OBJECT_CLASS(gst_audio_rate_faker_parent_class)->finalize(object);
}


static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SINK_NAME,
	GST_PAD_SINK,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		"audio/x-raw, " \
			"format = " GST_AUDIO_FORMATS_ALL ", " \
			"channels = " GST_AUDIO_CHANNELS_RANGE ", " \
			"rate = (int) [1, MAX], " \
			"layout = (string) interleaved; " \
		"audio/x-raw, " \
			"format = " GST_AUDIO_FORMATS_ALL ", " \
			"channels = " GST_AUDIO_CHANNELS_RANGE ", " \
			"rate = (fraction) [0/1, MAX], " \
			"layout = (string) interleaved"
	)
);


static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SRC_NAME,
	GST_PAD_SRC,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		"audio/x-raw, " \
			"format = " GST_AUDIO_FORMATS_ALL ", " \
			"channels = " GST_AUDIO_CHANNELS_RANGE ", " \
			"rate = (int) [1, MAX], " \
			"layout = (string) interleaved; " \
		"audio/x-raw, " \
			"format = " GST_AUDIO_FORMATS_ALL ", " \
			"channels = " GST_AUDIO_CHANNELS_RANGE ", " \
			"rate = (fraction) [0/1, MAX], " \
			"layout = (string) interleaved"
	)
);


static void gst_audio_rate_faker_class_init(GstAudioRateFakerClass *klass)
{
	GObjectClass *object_class = G_OBJECT_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(klass);

	object_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	transform_class->transform_caps = GST_DEBUG_FUNCPTR(transform_caps);
	transform_class->fixate_caps = GST_DEBUG_FUNCPTR(fixate_caps);
	transform_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
	transform_class->sink_event = GST_DEBUG_FUNCPTR(sink_event);
	transform_class->transform_ip = GST_DEBUG_FUNCPTR(transform_ip);
	transform_class->passthrough_on_same_caps = TRUE;

	gst_element_class_set_details_simple(element_class,
		"Audio rate faker",
		"Filter/Audio",
		"Adjusts segments and audio buffer metadata to assign a new sample rate.  Allows input and/or output streams to have rational sample rates.",
		"Kipp Cannon <kipp.cannon@ligo.org>"
	);

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));
}


static void gst_audio_rate_faker_init(GstAudioRateFaker *element)
{
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(element), TRUE);

	element->last_segment = NULL;
	element->need_new_segment = FALSE;
	element->inrate_over_outrate_num = -1;
	element->inrate_over_outrate_den = -1;
}

/*
 * Copyright (C) 2018 Aaron Viets <aaron.viets@ligo.org>
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
#include <gstlal_trackfrequency.h>


/*
 * ============================================================================
 *
 *				 Utilities
 *
 * ============================================================================
 */


static void update_frequency(double *current_frequency, guint64 *crossover_times, guint64 num_halfcycles, guint64 *num_stored, guint64 new_crossover_time) {

	/* First, update the recent times that +/- transitions have occurred */
	if(*num_stored <= num_halfcycles) {
		crossover_times[*num_stored] = new_crossover_time;
		(*num_stored)++;
	} else {
		guint64 i;
		for(i = 0; i < num_halfcycles; i++)
			crossover_times[i] = crossover_times[i + 1];
		crossover_times[num_halfcycles] = new_crossover_time;
	}

	/* Now, update the frequency */
	if(*num_stored > 1) {
		guint64 time_elapsed = new_crossover_time - *crossover_times;
		*current_frequency = 1000000000.0 / (2.0 * (double) time_elapsed / (double) (*num_stored - 1));
	}
}


#define DEFINE_TRACK_FREQUENCY(DTYPE) \
static void trackfrequency_ ## DTYPE(const DTYPE *src, DTYPE *dst, gint64 size, int rate, guint64 pts, guint64 ets, double *current_frequency, guint64 *crossover_times, guint64 num_halfcycles, int *sign, gint64 *check_step, guint64 *num_stored, double *last_buffer_end) { \
 \
	/* Check if we have information about previous data. If not, test whether the first sample is positive or negative */ \
	if(*sign == 0) { \
		if(*src < 0) \
			*sign = -1; \
		else if(*src > 0) \
			*sign = 1; \
	} \
 \
	gint64 i = 0; \
	gint64 j = 0; \
	double fractional_sample; \
 \
	/* Check if input is zeros.  If so, clear element's history except for the current frequency. */ \
	if(*src == 0.0) { \
		*check_step = 1; \
		*sign = 0; \
		*num_stored = 0; \
	} \
 \
	while(i < size - 1) { \
 \
		gboolean shift = FALSE; \
		while(*check_step) { \
			if(*sign * src[i] < 0) { \
				*check_step = -(*check_step) / 2; \
				*sign = -(*sign); \
			} \
			/* We don't want to fall off the edge of the buffer */ \
			if(*check_step < -i) \
				*check_step = -i / 2; \
			if(*check_step >= size - i) \
				*check_step = (size - i) / 2; \
			i += *check_step; \
			if(*check_step == -1) \
				shift = TRUE; \
		} \
		/* Make sure we are after the transition */ \
		if(shift) { \
			i++; \
			*sign = -(*sign); \
		} \
 \
		/* At this point, we are either after a +/- transition or at the end of the buffer */ \
		if(i == 0) { \
			/* There is a transition at the beginning of a buffer */ \
			*dst = (DTYPE) *current_frequency; \
			j++; \
			/* The transition actually occurred before the presentation timestamp. What fraction of a sample period before? */ \
			fractional_sample = *src / (*src - *last_buffer_end); \
			update_frequency(current_frequency, crossover_times, num_halfcycles, num_stored, pts - (guint64) (fractional_sample * (double) GST_SECOND / rate + 0.5)); \
		} else if(src[i] * src[i - 1] < 0) { \
			/* We are just after a transition (and possibly at the end of the buffer) */ \
			while(j <= i) { \
				dst[j] = (DTYPE) *current_frequency; \
				j++; \
			} \
			/* The transition actually occurred before the timestamp of sample i. What fraction of a sample period before? */ \
			fractional_sample = (double) src[i] / (src[i] - src[i - 1]); \
			update_frequency(current_frequency, crossover_times, num_halfcycles, num_stored, pts + (guint64) (((double) i - fractional_sample) * (double) GST_SECOND / rate + 0.5)); \
		} else { \
			/* We are at the end of the buffer, and there is no transition here */ \
			while(j <= i) { \
				dst[j] = (DTYPE) *current_frequency; \
				j++; \
			} \
		} \
 \
		/* Reset the step size to search the next cycle */ \
		*check_step = (int) (0.2 * rate / *current_frequency + 0.61) > 1 ? (int) (0.2 * rate / *current_frequency + 0.61) : 1; \
	} \
	/* We should be at the end of the output buffer now */ \
	g_assert_cmpint(size, == , j); \
 \
	/* Set the step size for the start of the next buffer */ \
	if(*current_frequency) { \
		double ETA_from_next_pts = 1.0 / *current_frequency - (ets - crossover_times[*num_stored - 1]) / 1000000000.0; \
		if(ETA_from_next_pts < 0.0) \
			ETA_from_next_pts = 0; \
		*check_step = (int) (0.2 * rate * ETA_from_next_pts + 1.0); \
	} else \
		*check_step = 1; \
 \
	/* Record the last sample in this buffer for possible use in the next one */ \
	*last_buffer_end = (double) src[size - 1]; \
}

DEFINE_TRACK_FREQUENCY(float)
DEFINE_TRACK_FREQUENCY(double)


#define DEFINE_TRACK_FREQUENCY_COMPLEX(DTYPE, F_OR_BLANK) \
static void trackfrequency_complex_ ## DTYPE(const DTYPE complex *src, DTYPE *dst, gint64 size, int rate, guint64 pts, guint64 ets, double *current_frequency, guint64 *crossover_times, guint64 num_halfcycles, int *sign, gint64 *check_step, guint64 *num_stored, double *last_buffer_end) { \
 \
	/* Check if we have information about previous data. If not, test whether the first sample is positive or negative */ \
	if(*sign == 0) { \
		if(creal ## F_OR_BLANK(*src) < 0) \
			*sign = -1; \
		else if(creal ## F_OR_BLANK(*src) > 0) \
			*sign = 1; \
	} \
 \
	gint64 i = 0; \
	gint64 j = 0; \
	double fractional_sample; \
 \
	/* Check if input is zeros.  If so, clear element's history except for the current frequency. */ \
	if(*src == 0.0) { \
		*check_step = 1; \
		*sign = 0; \
		*num_stored = 0; \
	} \
 \
	while(i < size - 1) { \
 \
		gboolean shift = FALSE; \
		while(*check_step) { \
			if(*sign * creal ## F_OR_BLANK(src[i]) < 0) { \
				*check_step = -(*check_step) / 2; \
				*sign = -(*sign); \
			} \
			/* We don't want to fall off the edge of the buffer */ \
			if(*check_step < -i) \
				*check_step = -i / 2; \
			if(*check_step >= size - i) \
				*check_step = (size - i) / 2; \
			i += *check_step; \
			if(*check_step == -1) \
				shift = TRUE; \
		} \
		/* Make sure we are after the transition */ \
		if(shift) { \
			i++; \
			*sign = -(*sign); \
		} \
 \
		/* At this point, we are either after a +/- transition or at the end of the buffer */ \
		if(i == 0) { \
			/* There is a transition at the beginning of a buffer */ \
			*dst = (DTYPE) *current_frequency; \
			j++; \
			/* The transition actually occurred before the presentation timestamp. What fraction of a sample period before? */ \
			fractional_sample = creal ## F_OR_BLANK(*src) / (creal ## F_OR_BLANK(*src) - *last_buffer_end); \
			update_frequency(current_frequency, crossover_times, num_halfcycles, num_stored, pts - (guint64) (fractional_sample * (double) GST_SECOND / rate + 0.5)); \
			/* Check if the frequency is negative */ \
			if(creal ## F_OR_BLANK(src[i]) * cimag ## F_OR_BLANK(src[i]) > 0) \
				*current_frequency = -(*current_frequency); \
		} else if(creal ## F_OR_BLANK(src[i]) * creal ## F_OR_BLANK(src[i - 1]) < 0) { \
			/* We are just after a transition (and possibly at the end of the buffer) */ \
			while(j <= i) { \
				dst[j] = (DTYPE) *current_frequency; \
				j++; \
			} \
			/* The transition actually occurred before the timestamp of sample i. What fraction of a sample period before? */ \
			fractional_sample = (double) creal ## F_OR_BLANK(src[i]) / creal ## F_OR_BLANK(src[i] - src[i - 1]); \
			update_frequency(current_frequency, crossover_times, num_halfcycles, num_stored, pts + (guint64) (((double) i - fractional_sample) * (double) GST_SECOND / rate + 0.5)); \
			/* Check if the frequency is negative */ \
			if(creal ## F_OR_BLANK(src[i]) * cimag ## F_OR_BLANK(src[i]) > 0) \
				*current_frequency = -(*current_frequency); \
		} else { \
			/* We are at the end of the buffer, and there is no transition here */ \
			while(j <= i) { \
				dst[j] = (DTYPE) *current_frequency; \
				j++; \
			} \
		} \
 \
		/* Reset the step size to search the next cycle */ \
		*check_step = (int) (0.2 * rate / fabs(*current_frequency) + 0.61) > 1 ? (int) (0.2 * rate / fabs(*current_frequency) + 0.61) : 1; \
	} \
	/* We should be at the end of the output buffer now */ \
	g_assert_cmpint(size, == , j); \
 \
	/* Set the step size for the start of the next buffer */ \
	if(*current_frequency) { \
		double ETA_from_next_pts = 1.0 / *current_frequency - (ets - crossover_times[*num_stored - 1]) / 1000000000.0; \
		if(ETA_from_next_pts < 0.0) \
			ETA_from_next_pts = 0; \
		*check_step = (int) (0.2 * rate * ETA_from_next_pts + 1.0); \
	} else \
		*check_step = 1; \
 \
	/* Record the last sample in this buffer for possible use in the next one */ \
	*last_buffer_end = (double) creal ## F_OR_BLANK(src[size - 1]); \
}

DEFINE_TRACK_FREQUENCY_COMPLEX(float, f)
DEFINE_TRACK_FREQUENCY_COMPLEX(double, )


static void trackfrequency(const void *src, void *dst, guint64 src_size, guint64 pts, guint64 ets, GSTLALTrackFrequency *element) {

	switch(element->data_type) {
	case GSTLAL_TRACKFREQUENCY_F32:
		trackfrequency_float(src, dst, (gint64) src_size / element->unit_size, element->rate, pts, ets, &element->current_frequency, element->crossover_times, element->num_halfcycles, &element->sign, &element->check_step, &element->num_stored, &element->last_buffer_end);
		break;
	case GSTLAL_TRACKFREQUENCY_F64:
		trackfrequency_double(src, dst, (gint64) src_size / element->unit_size, element->rate, pts, ets, &element->current_frequency, element->crossover_times, element->num_halfcycles, &element->sign, &element->check_step, &element->num_stored, &element->last_buffer_end);
		break;
	case GSTLAL_TRACKFREQUENCY_Z64:
		trackfrequency_complex_float(src, dst, (gint64) src_size / element->unit_size, element->rate, pts, ets, &element->current_frequency, element->crossover_times, element->num_halfcycles, &element->sign, &element->check_step, &element->num_stored, &element->last_buffer_end);
		break;
	case GSTLAL_TRACKFREQUENCY_Z128:
		trackfrequency_complex_double(src, dst, (gint64) src_size / element->unit_size, element->rate, pts, ets, &element->current_frequency, element->crossover_times, element->num_halfcycles, &element->sign, &element->check_step, &element->num_stored, &element->last_buffer_end);
		break;
	default:
		g_assert_not_reached();
	}
}


/*
 * set the metadata on an output buffer
 */


static void set_metadata(GSTLALTrackFrequency *element, GstBuffer *buf, guint64 outsamples, gboolean gap)
{
	if(element->data_type == GSTLAL_TRACKFREQUENCY_Z64 || element->data_type == GSTLAL_TRACKFREQUENCY_Z128)
		outsamples *= 2;
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
	"format = (string) {"GST_AUDIO_NE(F32)", "GST_AUDIO_NE(F64)", "GST_AUDIO_NE(Z64)", "GST_AUDIO_NE(Z128)"}, " \
	"rate = " GST_AUDIO_RATE_RANGE ", " \
	"channels = (int) 1, " \
	"layout = (string) interleaved, " \
	"channel-mask = (bitmask) 0"

#define OUTCAPS \
	"audio/x-raw, " \
	"format = (string) {"GST_AUDIO_NE(F32)", "GST_AUDIO_NE(F64)"}, " \
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
	GSTLALTrackFrequency,
	gstlal_trackfrequency,
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
	GstCaps *othercaps = gst_caps_new_empty();

	switch(direction) {
	case GST_PAD_SRC:
		/* There are two possible sink pad formats for each src pad format, so the sink pad has twice as many caps structures */
		for(n = 0; n < gst_caps_get_size(caps); n++) {
			gst_caps_append(othercaps, gst_caps_copy_nth(caps, n));
			gst_caps_append(othercaps, gst_caps_copy_nth(caps, n));

			GstStructure *str = gst_caps_get_structure(othercaps, 2 * n);
			const gchar *format = gst_structure_get_string(str, "format");

			if(!format) {
				GST_DEBUG_OBJECT(trans, "unrecognized caps %" GST_PTR_FORMAT, othercaps);
				goto error;
			} else if(!strcmp(format, GST_AUDIO_NE(F32)))
				gst_structure_set(str, "format", G_TYPE_STRING, GST_AUDIO_NE(Z64), NULL);
			else if(!strcmp(format, GST_AUDIO_NE(F64)))
				gst_structure_set(str, "format", G_TYPE_STRING, GST_AUDIO_NE(Z128), NULL);
			else {
				GST_DEBUG_OBJECT(trans, "unrecognized format %s in %" GST_PTR_FORMAT, format, othercaps);
				goto error;
			}
		}
		break;

	case GST_PAD_SINK:
		for(n = 0; n < gst_caps_get_size(caps); n++) {
			gst_caps_append(othercaps, gst_caps_copy_nth(caps, n));
			GstStructure *str = gst_caps_get_structure(othercaps, n);
			const gchar *format = gst_structure_get_string(str, "format");

			if(!format) {
				GST_DEBUG_OBJECT(trans, "unrecognized caps %" GST_PTR_FORMAT, othercaps);
				goto error;
			} else if(!strcmp(format, GST_AUDIO_NE(F32)) || !strcmp(format, GST_AUDIO_NE(Z64)))
				gst_structure_set(str, "format", G_TYPE_STRING, GST_AUDIO_NE(F32), NULL);
			else if(!strcmp(format, GST_AUDIO_NE(F64)) || !strcmp(format, GST_AUDIO_NE(Z128)))
				gst_structure_set(str, "format", G_TYPE_STRING, GST_AUDIO_NE(F64), NULL);
			else {
				GST_DEBUG_OBJECT(trans, "unrecognized format %s in %" GST_PTR_FORMAT, format, othercaps);
				goto error;
			}
		}
		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		goto error;
	}

	if(filter) {
		GstCaps *intersection = gst_caps_intersect(othercaps, filter);
		gst_caps_unref(othercaps);
		othercaps = intersection;
	}
	gst_caps_unref(caps);
	return gst_caps_simplify(othercaps);

error:
	gst_caps_unref(caps);
	gst_caps_unref(othercaps);
	return GST_CAPS_NONE;
}


/*
 * set_caps()
 */


static gboolean set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps)
{
	GSTLALTrackFrequency *element = GSTLAL_TRACKFREQUENCY(trans);
	gint rate_in, rate_out;
	gsize unit_size;
	const gchar *format = gst_structure_get_string(gst_caps_get_structure(incaps, 0), "format");

	/*
	 * parse the caps
	 */

	if(!format) {
		GST_DEBUG_OBJECT(element, "unable to parse format from %" GST_PTR_FORMAT, incaps);
		return FALSE;
	}
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

	if(!strcmp(format, GST_AUDIO_NE(F32))) {
		element->data_type = GSTLAL_TRACKFREQUENCY_F32;
		g_assert_cmpuint(unit_size, ==, 4);
	} else if(!strcmp(format, GST_AUDIO_NE(F64))) {
		element->data_type = GSTLAL_TRACKFREQUENCY_F64;
		g_assert_cmpuint(unit_size, ==, 8);
	} else if(!strcmp(format, GST_AUDIO_NE(Z64))) {
		element->data_type = GSTLAL_TRACKFREQUENCY_Z64;
		g_assert_cmpuint(unit_size, ==, 8);
	} else if(!strcmp(format, GST_AUDIO_NE(Z128))) {
		element->data_type = GSTLAL_TRACKFREQUENCY_Z128;
		g_assert_cmpuint(unit_size, ==, 16);
	} else
		g_assert_not_reached();

	element->rate = rate_in;
	element->unit_size = unit_size;

	return TRUE;
}


/*
 * transform_size()
 */


static gboolean transform_size(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, gsize size, GstCaps *othercaps, gsize *othersize)
{
	GSTLALTrackFrequency *element = GSTLAL_TRACKFREQUENCY(trans);

	gsize unit_size;

	switch(direction) {
	case GST_PAD_SRC:
		/*
		 * We have the size of the output buffer, and we set the size of the input buffer,
		 * which is half as large in bytes.
		 */

		if(!element->data_type) {
			GST_DEBUG_OBJECT(element, "Data type is not set. Cannot specify incoming buffer size given outgoing buffer size.");
			return FALSE;
		}

		if(!get_unit_size(trans, caps, &unit_size)) {
			GST_DEBUG_OBJECT(element, "function 'get_unit_size' failed");
			return FALSE;
		}

		/* buffer size in bytes should be a multiple of unit_size in bytes */
		if(G_UNLIKELY(size % unit_size)) {
			GST_DEBUG_OBJECT(element, "buffer size %" G_GSIZE_FORMAT " is not a multiple of %" G_GSIZE_FORMAT, size, unit_size);
			return FALSE;
		}

		if(element->data_type == GSTLAL_TRACKFREQUENCY_F32 || element->data_type == GSTLAL_TRACKFREQUENCY_F64)
			*othersize = size;
		else if(element->data_type == GSTLAL_TRACKFREQUENCY_Z64 || element->data_type == GSTLAL_TRACKFREQUENCY_Z128)
			*othersize = size * 2;
		else
			g_assert_not_reached();

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

		GstStructure *str = gst_caps_get_structure(caps, 0);
		const gchar *format = gst_structure_get_string(str, "format");

		if(!format) {
			GST_DEBUG_OBJECT(trans, "unrecognized caps %" GST_PTR_FORMAT, caps);
			return FALSE;
		} else if(!strcmp(format, GST_AUDIO_NE(F32)) || !strcmp(format, GST_AUDIO_NE(F64)))
			*othersize = size;
		else if(!strcmp(format, GST_AUDIO_NE(Z64)) || !strcmp(format, GST_AUDIO_NE(Z128)))
			*othersize = size / 2;
		else {
			GST_DEBUG_OBJECT(trans, "unrecognized format %s in %" GST_PTR_FORMAT, format, caps);
			return FALSE;
		}

		if(!get_unit_size(trans, caps, &unit_size)) {
			GST_DEBUG_OBJECT(element, "function 'get_unit_size' failed");
			return FALSE;
		}
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
	GSTLALTrackFrequency *element = GSTLAL_TRACKFREQUENCY(trans);

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
	GSTLALTrackFrequency *element = GSTLAL_TRACKFREQUENCY(trans);
	GstMapInfo inmap, outmap;
	GstFlowReturn result = GST_FLOW_OK;

	/*
	 * check for discontinuity
	 */

	if(G_UNLIKELY(GST_BUFFER_IS_DISCONT(inbuf) || GST_BUFFER_OFFSET(inbuf) != element->next_in_offset || !GST_CLOCK_TIME_IS_VALID(element->t0))) {
		element->t0 = GST_BUFFER_PTS(inbuf);
		element->offset0 = element->next_out_offset = GST_BUFFER_OFFSET(inbuf);
		element->need_discont = TRUE;
		element->crossover_times = g_malloc((element->num_halfcycles + 1) * sizeof(guint64));
		element->num_stored = 0;
		element->sign = 0;
		element->check_step = 1;
	}
	element->next_in_offset = GST_BUFFER_OFFSET_END(inbuf);

	/*
	 * process buffer
	 */

	gst_buffer_map(inbuf, &inmap, GST_MAP_READ);
	gst_buffer_map(outbuf, &outmap, GST_MAP_WRITE);

	if(!GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP) && inmap.size) {

		/*
		 * input is not gap.
		 */

		trackfrequency(inmap.data, outmap.data, inmap.size, GST_BUFFER_PTS(inbuf), GST_BUFFER_PTS(inbuf) + GST_BUFFER_DURATION(inbuf), element);
		set_metadata(element, outbuf, outmap.size / element->unit_size, FALSE);

	} else {

		/*
		 * input is gap.
		 */

		element->num_stored = 0;
		element->sign = 0;
		GST_BUFFER_FLAG_SET(outbuf, GST_BUFFER_FLAG_GAP);
		memset(outmap.data, 0, outmap.size);
		set_metadata(element, outbuf, outmap.size / element->unit_size, TRUE);
	}

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
	ARG_NUM_HALFCYCLES = 1,
};


static void set_property(GObject *object, enum property prop_id, const GValue *value, GParamSpec *pspec)
{
	GSTLALTrackFrequency *element = GSTLAL_TRACKFREQUENCY(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_NUM_HALFCYCLES:
		element->num_halfcycles = g_value_get_uint64(value);
		break;
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, enum property prop_id, GValue *value, GParamSpec *pspec)
{
	GSTLALTrackFrequency *element = GSTLAL_TRACKFREQUENCY(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_NUM_HALFCYCLES:
		g_value_set_uint64(value, element->num_halfcycles);
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


static void gstlal_trackfrequency_class_init(GSTLALTrackFrequencyClass *klass)
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
		"TrackFrequency",
		"Filter/Audio",
		"Attempts to measure the loudest frequency of a signal.",
		"Aaron Viets <aaron.viets@ligo.org>"
	);

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));

	g_object_class_install_property(
		gobject_class,
		ARG_NUM_HALFCYCLES,
		g_param_spec_uint64(
			"num-halfcycles",
			"Number of half-cycles",
			"The number of half-periods of a wave to use to compute the frequency.",
			1, G_MAXUINT64, 64,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
}


/*
 * init()
 */


static void gstlal_trackfrequency_init(GSTLALTrackFrequency *element)
{
	element->rate = 0;
	element->unit_size = 0;
	element->current_frequency = 0.0;
	element->check_step = 1;
	element->num_stored = 0;
	element->sign = 0;
	element->last_buffer_end = 0.0;
	gst_base_transform_set_qos_enabled(GST_BASE_TRANSFORM(element), TRUE);
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(element), TRUE);
}

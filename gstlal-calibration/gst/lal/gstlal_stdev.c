/*
 * Copyright (C) 2021 Aaron Viets <aaron.viets@ligo.org>
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
 * stuff from C
 */


#include <math.h>
#include <string.h>
#include <complex.h>


/*
 *  stuff from gobject/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/audio/audio.h>
#include <gst/base/gstbasetransform.h>


/*
 * our own stuff
 */


#include <gstlal/gstlal.h>
#include <gstlal/gstlal_debug.h>
#include <gstlal/gstlal_audio_info.h>
#include <gstlal_stdev.h>


/*
 * ============================================================================
 *
 *			      Custom Types
 *
 * ============================================================================
 */


/*
 * mode enum
 */


GType gstlal_stdev_mode_get_type(void) {

	static GType type = 0;

	if(!type) {
		static GEnumValue values[] = {
			{GSTLAL_STDEV_ABSOLUTE, "GSTLAL_STDEV_ABSOLUTE", "Compute the absolute uncertainty"},
			{GSTLAL_STDEV_RELATIVE, "GSTLAL_STDEV_RELATIVE", "Compute the relative uncertainty"},
			{0, NULL, NULL}
		};

		type = g_enum_register_static("GSTLAL_STDEV_MODE", values);
	}

	return type;
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
		"rate = (int) [1, MAX], " \
		"channels = (int) 1, " \
		"format = (string) { " GST_AUDIO_NE(F32) ", " GST_AUDIO_NE(F64) ", " GST_AUDIO_NE(Z64) ", " GST_AUDIO_NE(Z128) " }, " \
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


#define GST_CAT_DEFAULT gstlal_stdev_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


G_DEFINE_TYPE_WITH_CODE(
	GSTLALStDev,
	gstlal_stdev,
	GST_TYPE_BASE_TRANSFORM,
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "lal_stdev", 0, "lal_stdev element")
);


/*
 * ============================================================================
 *
 *				 Utilities
 *
 * ============================================================================
 */


/*
 * set the metadata on an output buffer
 */


static void set_metadata(GSTLALStDev *element, GstBuffer *buf, guint64 outsamples) {

	GST_BUFFER_OFFSET(buf) = element->next_out_offset;
	element->next_out_offset += outsamples;
	element->total_insamples += outsamples;
	GST_BUFFER_OFFSET_END(buf) = element->next_out_offset;
	GST_BUFFER_TIMESTAMP(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET(buf) - element->offset0, GST_SECOND, element->rate);
	GST_BUFFER_DURATION(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET_END(buf) - element->offset0, GST_SECOND, element->rate) - GST_BUFFER_TIMESTAMP(buf);
	GST_BUFFER_FLAG_UNSET(buf, GST_BUFFER_FLAG_GAP);
	if(G_UNLIKELY(element->need_discont)) {
		GST_BUFFER_FLAG_SET(buf, GST_BUFFER_FLAG_DISCONT);
		element->need_discont = FALSE;
	}
}


static void create_workspace(GSTLALStDev *element) {

	switch(element->data_type) {
	case GSTLAL_STDEV_F32:
		if(element->workspace.typefloat.array)
			g_free(element->workspace.typefloat.array);
		element->workspace.typefloat.array = g_malloc(element->array_size * sizeof(float));
		element->workspace.typefloat.current_stdev = G_MAXFLOAT;
		break;
	case GSTLAL_STDEV_F64:
		if(element->workspace.typedouble.array)
			g_free(element->workspace.typedouble.array);
		element->workspace.typedouble.array = g_malloc(element->array_size * sizeof(double));
		element->workspace.typedouble.current_stdev = G_MAXDOUBLE;
		break;
	case GSTLAL_STDEV_Z64:
		if(element->workspace.typecomplexfloat.array)
			g_free(element->workspace.typecomplexfloat.array);
		element->workspace.typecomplexfloat.array = g_malloc(element->array_size * sizeof(complex float));
		element->workspace.typecomplexfloat.current_stdev = G_MAXFLOAT;
		break;
	case GSTLAL_STDEV_Z128:
		if(element->workspace.typecomplexdouble.array)
			g_free(element->workspace.typecomplexdouble.array);
		element->workspace.typecomplexdouble.array = g_malloc(element->array_size * sizeof(complex double));
		element->workspace.typecomplexdouble.current_stdev = G_MAXDOUBLE;
		break;
	default:
		g_assert_not_reached();
	}
	return;
}


static void free_workspace(GSTLALStDev *element) {

	switch(element->data_type) {
	case GSTLAL_STDEV_F32:
		if(element->workspace.typefloat.array)
			g_free(element->workspace.typefloat.array);
		element->workspace.typefloat.array = NULL;
		break;
	case GSTLAL_STDEV_F64:
		if(element->workspace.typedouble.array)
			g_free(element->workspace.typedouble.array);
		element->workspace.typedouble.array = NULL;
		break;
	case GSTLAL_STDEV_Z64:
		if(element->workspace.typecomplexfloat.array)
			g_free(element->workspace.typecomplexfloat.array);
		element->workspace.typecomplexfloat.array = NULL;
		break;
	case GSTLAL_STDEV_Z128:
		if(element->workspace.typecomplexdouble.array)
			g_free(element->workspace.typecomplexdouble.array);
		element->workspace.typecomplexdouble.array = NULL;
		break;
	default:
		g_assert_not_reached();
	}

	return;
}


#define DEFINE_COMPUTE_STDEV(COMPLEX, DTYPE, C_OR_F, F_OR_NOT) \
static DTYPE compute_stdev ## COMPLEX ## DTYPE(COMPLEX DTYPE *array, guint64 array_size, guint64 start_index, guint64 samples_in_array, enum gstlal_stdev_mode mode) { \
 \
	guint64 i, end; \
	DTYPE next_term, var, std; \
	COMPLEX DTYPE sum, avg; \
 \
	/* Start by finding the average. */ \
	sum = 0; \
	end = start_index + samples_in_array; \
	for(i = start_index; i < end; i++) \
		sum += array[i % array_size]; \
 \
	avg = sum / samples_in_array; \
 \
	/* Now find the standard deviation */ \
	var = 0; \
	for(i = start_index; i < end; i++) { \
		next_term = C_OR_F ## abs ## F_OR_NOT(array[i % array_size] - avg); \
		var += next_term * next_term; \
	} \
	var /= samples_in_array - 1; \
 \
	/* If input is complex, divide variance by 2 */ \
	if((COMPLEX DTYPE) I == I) \
		var /= 2; \
 \
	std = sqrt ## F_OR_NOT(var); \
 \
	/* If we are computing relative uncertainty, divide by magnitude of average */ \
	if(mode == GSTLAL_STDEV_RELATIVE) \
		std /= C_OR_F ## abs ## F_OR_NOT(avg); \
 \
	return std; \
}


DEFINE_COMPUTE_STDEV( , float, f, f);
DEFINE_COMPUTE_STDEV( , double, f, );
DEFINE_COMPUTE_STDEV(complex, float, c, f);
DEFINE_COMPUTE_STDEV(complex, double, c, );


#define DEFINE_PROCESS_INDATA(COMPLEX, DTYPE) \
static GstFlowReturn process_indata_ ## COMPLEX ## DTYPE(const COMPLEX DTYPE *src, guint64 src_size, DTYPE *dst, guint64 dst_size, GSTLALStDev *element) { \
 \
	guint64 i, i_start, i_stop = 0; \
 \
	/* If array is not full, fill it as much as possible before computing any uncertainty */ \
	if(element->samples_in_array < element->array_size && src_size > 0) { \
		while(element->samples_in_array < element->array_size && element->buffer_index < src_size) { \
			element->workspace.type ## COMPLEX ## DTYPE.array[element->array_index] = src[element->buffer_index]; \
			element->buffer_index += element->coherence_length; \
			element->samples_in_array++; \
			element->array_index++; \
			element->array_index %= element->array_size; \
		} \
		/* Compute first output */ \
		if(element->samples_in_array > 1) \
			element->workspace.type ## COMPLEX ## DTYPE.current_stdev = compute_stdev ## COMPLEX ## DTYPE(element->workspace.type ## COMPLEX ## DTYPE.array, element->array_size, element->start_index, element->samples_in_array, element->mode); \
		/* How many samples in the output should be equal to this? */ \
		i_stop = dst_size + element->buffer_index > src_size ? dst_size + element->buffer_index - src_size : 0; \
		i_stop = i_stop < dst_size ? i_stop : dst_size; \
		for(i = 0; i < i_stop; i++) \
			dst[i] = element->workspace.type ## COMPLEX ## DTYPE.current_stdev; \
	} else { \
		/* Fill the output with the current uncertainty up to the buffer_index, where the next value becomes valid. */ \
		i_stop = element->buffer_index < dst_size ? element->buffer_index : dst_size; \
		for(i = 0; i < i_stop; i++) \
			dst[i] = element->workspace.type ## COMPLEX ## DTYPE.current_stdev; \
	} \
 \
	/* Now finish off the inputs */ \
	while(element->buffer_index < src_size) { \
		element->workspace.type ## COMPLEX ## DTYPE.array[element->array_index] = src[element->buffer_index]; \
		element->workspace.type ## COMPLEX ## DTYPE.current_stdev = compute_stdev ## COMPLEX ## DTYPE(element->workspace.type ## COMPLEX ## DTYPE.array, element->array_size, 0, element->samples_in_array, element->mode); \
		i_start = i_stop; \
		i_stop += element->coherence_length; \
		i_stop = i_stop < dst_size ? i_stop : dst_size; \
		for(i = i_start; i < i_stop; i++) \
			dst[i] = element->workspace.type ## COMPLEX ## DTYPE.current_stdev; \
		element->buffer_index += element->coherence_length; \
		element->array_index++; \
		element->array_index %= element->array_size; \
	} \
 \
	/* Now fill the rest of the outputs with the current uncertainty */ \
	for(i = i_stop; i < dst_size; i++) \
		dst[i] = element->workspace.type ## COMPLEX ## DTYPE.current_stdev; \
 \
	/* Calculate the buffer_index of the next buffer */ \
	element->buffer_index -= src_size; \
 \
	return GST_FLOW_OK; \
}


DEFINE_PROCESS_INDATA( , float);
DEFINE_PROCESS_INDATA( , double);
DEFINE_PROCESS_INDATA(complex, float);
DEFINE_PROCESS_INDATA(complex, double);


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
	gboolean success = gstlal_audio_info_from_caps(&info, caps);

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
	guint n;

	caps = gst_caps_normalize(gst_caps_copy(caps));
	GstCaps *othercaps = gst_caps_new_empty();

	switch(direction) {
	case GST_PAD_SRC:
		/*
		 * We know the caps on the source (outgoing data) pad, and we want to find the
		 * caps of the sink pad.  There are two possible sink pad formats for each src
		 * pad format, so the sink pad caps have twice as many structures.
		 */
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
		/*
		 * We know the caps on the sink (incoming data) pad, and we want to find the
		 * caps of the source pad.  There are two possible sink pad formats for each
		 * src pad format, so the sink pad caps have twice as many structures.
		 */
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


static gboolean set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps) {

	GSTLALStDev *element = GSTLAL_STDEV(trans);
	gboolean success = TRUE;
	gsize unit_size;
	gint rate_in, rate_out;

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

	/* require the input and output rates to be equal */
	success &= (rate_in == rate_out);
	if(rate_in != rate_out)
		GST_ERROR_OBJECT(element, "output rate is not equal to input rate.  input caps = %" GST_PTR_FORMAT " output caps = %" GST_PTR_FORMAT, incaps, outcaps);

	/*
	 * record stream parameters
	 */

	if(success) {
		if(!strcmp(name, GST_AUDIO_NE(F32))) {
			element->data_type = GSTLAL_STDEV_F32;
			g_assert_cmpuint(unit_size, ==, 4);
		} else if(!strcmp(name, GST_AUDIO_NE(F64))) {
			element->data_type = GSTLAL_STDEV_F64;
			g_assert_cmpuint(unit_size, ==, 8);
		} else if(!strcmp(name, GST_AUDIO_NE(Z64))) {
			element->data_type = GSTLAL_STDEV_Z64;
			g_assert_cmpuint(unit_size, ==, 8);
		} else if(!strcmp(name, GST_AUDIO_NE(Z128))) {
			element->data_type = GSTLAL_STDEV_Z128;
			g_assert_cmpuint(unit_size, ==, 16);
		} else
			g_assert_not_reached();

		element->unit_size = unit_size;
		element->rate = rate_in;
	}

	create_workspace(element);

	return success;
}


/*
 * sink_event()
 */


static gboolean sink_event(GstBaseTransform *trans, GstEvent *event) {
	GSTLALStDev *element = GSTLAL_STDEV(trans);
	gboolean success = TRUE;
	GST_DEBUG_OBJECT(element, "Got %s event on sink pad", GST_EVENT_TYPE_NAME(event));

	guint64 waste_samples = (guint64) (element->filter_latency * (element->array_size * element->coherence_length - 1));
	if(GST_EVENT_TYPE(event) == GST_EVENT_EOS && waste_samples > 0) {
		void *data;
		GstFlowReturn result;
		switch(element->data_type) {
		case GSTLAL_STDEV_F32:
			data = g_malloc(waste_samples * sizeof(float));
			result = process_indata_float(NULL, 0, (float *) data, waste_samples, element);
			break;
		case GSTLAL_STDEV_F64:
			data = g_malloc(waste_samples * sizeof(double));
			result = process_indata_double(NULL, 0, (double *) data, waste_samples, element);
			break;
		case GSTLAL_STDEV_Z64:
			data = g_malloc(waste_samples * sizeof(float));
			result = process_indata_complexfloat(NULL, 0, (float *) data, waste_samples, element);
			break;
		case GSTLAL_STDEV_Z128:
			data = g_malloc(waste_samples * sizeof(double));
			result = process_indata_complexdouble(NULL, 0, (double *) data, waste_samples, element);
			break;
		default:
			result = GST_FLOW_ERROR;
			success = FALSE;
			break;
		}

		if(result == GST_FLOW_OK) {
			GstBuffer *buf;
			buf = gst_buffer_new_wrapped(data, waste_samples * element->unit_size);

			set_metadata(element, buf, waste_samples);

			/* push buffer downstream */
			GST_DEBUG_OBJECT(element, "pushing final buffer %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(buf));
			result = gst_pad_push(element->srcpad, buf);
		}
		if(G_UNLIKELY(result != GST_FLOW_OK)) {
			GST_WARNING_OBJECT(element, "push failed: %s", gst_flow_get_name(result));
			success = FALSE;
		}
	}

	success &= GST_BASE_TRANSFORM_CLASS(gstlal_stdev_parent_class)->sink_event(trans, event);

	return success;
}


/*
 * transform_size()
 */


static gboolean transform_size(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, gsize size, GstCaps *othercaps, gsize *othersize) {

	GSTLALStDev *element = GSTLAL_STDEV(trans);

	gsize unit_size, other_unit_size;
	if(!get_unit_size(trans, caps, &unit_size)) {
		GST_DEBUG_OBJECT(element, "function 'get_unit_size' failed");
		return FALSE;
	}
	if(!get_unit_size(trans, othercaps, &other_unit_size)) {
		GST_DEBUG_OBJECT(element, "function 'get_unit_size' failed");
		return FALSE;
	}

	/* buffer size in bytes should be a multiple of unit_size in bytes */
	if(G_UNLIKELY(size % unit_size)) {
		GST_DEBUG_OBJECT(element, "buffer size %" G_GSIZE_FORMAT " is not a multiple of %" G_GSIZE_FORMAT, size, unit_size);
		return FALSE;
	}

	size /= unit_size;

	/* How many samples do we need to throw away based on the filter latency? */
	guint64 waste_samples = (guint64) (element->filter_latency * (element->array_size * element->coherence_length - 1));

	switch(direction) {
	case GST_PAD_SRC:
		/* We have the size of the output buffer, and we set the size of the input buffer. */
		/* Check if we need to clip the output buffer */
		if(element->total_insamples >= waste_samples)
			*othersize = size;
		else
			*othersize = size + waste_samples - element->total_insamples;
		break;

	case GST_PAD_SINK:
		/* We have the size of the input buffer, and we set the size of the output buffer. */
		/* Check if we need to clip the output buffer */
		if(element->total_insamples >= waste_samples)
			*othersize = size;
		else if(size > (guint) (waste_samples - element->total_insamples))
			*othersize = size - waste_samples + element->total_insamples;
		else
			*othersize = 0;
		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		return FALSE;
	}

	*othersize *= other_unit_size;

	return TRUE;
}


/*
 * transform()
 */


static GstFlowReturn transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf) {

	GSTLALStDev *element = GSTLAL_STDEV(trans);
	GstMapInfo inmap, outmap;
	GstFlowReturn result;

	/*
	 * Check for discontinuity
	 */

	if(G_UNLIKELY(GST_BUFFER_IS_DISCONT(inbuf) || GST_BUFFER_OFFSET(inbuf) != element->next_in_offset || !GST_CLOCK_TIME_IS_VALID(element->t0))) {
		/* How many samples do we need to throw away based on the filter latency? */
		guint64 waste_samples = (guint64) (element->filter_latency * (element->array_size * element->coherence_length - 1));
		guint64 shift_samples = waste_samples < element->total_insamples ? waste_samples : element->total_insamples;
		element->offset0 = element->next_out_offset = GST_BUFFER_OFFSET(inbuf) - shift_samples;
		element->t0 = GST_BUFFER_PTS(inbuf) - gst_util_uint64_scale_int_round(shift_samples, GST_SECOND, element->rate);
		element->need_discont = TRUE;
		guint64 sample_number = gst_util_uint64_scale_int_round(GST_BUFFER_PTS(inbuf), element->rate, GST_SECOND);
		element->buffer_index = sample_number % element->coherence_length;
		element->buffer_index = (element->coherence_length - element->buffer_index) % element->coherence_length;
		element->array_index = element->start_index = ((sample_number + element->buffer_index) / element->coherence_length) % element->array_size;
	}
	element->next_in_offset = GST_BUFFER_OFFSET_END(inbuf);

	GST_INFO_OBJECT(element, "processing %s%s buffer %p spanning %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP) ? "gap" : "nongap", GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_DISCONT) ? "+discont" : "", inbuf, GST_BUFFER_BOUNDARIES_ARGS(inbuf));

	gst_buffer_map(outbuf, &outmap, GST_MAP_WRITE);
	gst_buffer_map(inbuf, &inmap, GST_MAP_READ);

	/* sanity checks */
	g_assert_cmpuint(inmap.size % element->unit_size, ==, 0);
	g_assert_cmpuint(outmap.size % element->unit_size, ==, 0);

	/* Process data in buffer */
	switch(element->data_type) {
	case GSTLAL_STDEV_F32:
		result = process_indata_float((const float *) inmap.data, inmap.size / element->unit_size, (float *) outmap.data, outmap.size / element->unit_size, element);
		break;
	case GSTLAL_STDEV_F64:
		result = process_indata_double((const double *) inmap.data, inmap.size / element->unit_size, (double *) outmap.data, outmap.size / element->unit_size, element);
		break;
	case GSTLAL_STDEV_Z64:
		result = process_indata_complexfloat((const float complex *) inmap.data, inmap.size / element->unit_size, (float *) outmap.data, outmap.size / element->unit_size, element);
		break;
	case GSTLAL_STDEV_Z128:
		result = process_indata_complexdouble((const double complex *) inmap.data, inmap.size / element->unit_size, (double *) outmap.data, outmap.size / element->unit_size, element);
		break;
	default:
		g_assert_not_reached();
	}

	set_metadata(element, outbuf, outmap.size / element->unit_size);

	gst_buffer_unmap(inbuf, &inmap);

	gst_buffer_unmap(outbuf, &outmap);

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
	ARG_ARRAY_SIZE = 1,
	ARG_COHERENCE_LENGTH,
	ARG_MODE,
	ARG_FILTER_LATENCY
};


static void set_property(GObject *object, enum property prop_id, const GValue *value, GParamSpec *pspec) {

	GSTLALStDev *element = GSTLAL_STDEV(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_ARRAY_SIZE:
		element->array_size = g_value_get_uint64(value);
		break;
	case ARG_COHERENCE_LENGTH:
		element->coherence_length = g_value_get_uint64(value);
		break;
	case ARG_MODE:
		element->mode = g_value_get_enum(value);
		break;
	case ARG_FILTER_LATENCY:
		element->filter_latency = g_value_get_double(value);
		break;
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, enum property prop_id, GValue *value, GParamSpec *pspec) {

	GSTLALStDev *element = GSTLAL_STDEV(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_ARRAY_SIZE:
		g_value_set_uint64(value, element->array_size);
		break;
	case ARG_COHERENCE_LENGTH:
		g_value_set_uint64(value, element->coherence_length);
		break;
	case ARG_MODE:
		g_value_set_enum(value, element->mode);
		break;
	case ARG_FILTER_LATENCY:
		g_value_set_double(value, element->filter_latency);
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


static void finalize(GObject *object) {

	GSTLALStDev *element = GSTLAL_STDEV(object);
	free_workspace(element);
	gst_object_unref(element->srcpad);
	element->srcpad = NULL;
	G_OBJECT_CLASS(gstlal_stdev_parent_class)->finalize(object);
}


/*
 * class_init()
 */


static void gstlal_stdev_class_init(GSTLALStDevClass *klass) {

	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(klass);

	gst_element_class_set_details_simple(
		element_class,
		"Compute Standard Deviation",
		"Filter/Audio",
		"Computes the standard deviation of a data stream.  For complex-valued streams, a real-valued\n\t\t\t   "
		"standard deviation is computed, and the calculation is sensitive to any type of fluctuation,\n\t\t\t   "
		"regardless of the shape of the distribution in the complex plane.  If the distribution is\n\t\t\t   "
		"circularly symmetric, the result is the standard deviation of the magnitude, real part, and\n\t\t\t   "
		"imaginary part, all of which are equal.",
		"Aaron Viets <aaron.viets@ligo.org>"
	);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	transform_class->get_unit_size = GST_DEBUG_FUNCPTR(get_unit_size);
	transform_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
	transform_class->sink_event = GST_DEBUG_FUNCPTR(sink_event);
	transform_class->transform_caps = GST_DEBUG_FUNCPTR(transform_caps);
	transform_class->transform_size = GST_DEBUG_FUNCPTR(transform_size);
	transform_class->transform = GST_DEBUG_FUNCPTR(transform);

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));

	g_object_class_install_property(
		gobject_class,
		ARG_ARRAY_SIZE,
		g_param_spec_uint64(
			"array-size",
			"Array Size",
			"Number of samples needed to compute the standard deviation.",
			2, G_MAXUINT64, 16,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_COHERENCE_LENGTH,
		g_param_spec_uint64(
			"coherence-length",
			"Coherence Length",
			"Number of samples to advance in the input data each time a new sample is\n\t\t\t"
			"added to the standard deviation array",
			1, G_MAXUINT64, 128,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_MODE,
		g_param_spec_enum(
			"mode",
			"Standard Deviation Mode",
			"Whether to compute the absolute or relative uncertainty",
			GSTLAL_STDEV_MODE,
			GSTLAL_STDEV_ABSOLUTE,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_FILTER_LATENCY,
		g_param_spec_double(
			"filter-latency",
			"Filter Latency",
			"The latency added by the element, as a fraction of the length of input\n\t\t\t"
			"data required to compute the standard deviation. If 0, there is no\n\t\t\t"
			"latency.  If 1, the latency is the amount of data needed to compute the\n\t\t\t"
			"standard deviation.",
			0.0, 1.0, 0.0,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
}


/*
 * init()
 */


static void gstlal_stdev_init(GSTLALStDev *element) {

	/* retrieve (and ref) src pad */
	GstPad *pad = gst_element_get_static_pad(GST_ELEMENT(element), "src");
	GST_PAD_SET_PROXY_CAPS(pad);
	GST_PAD_SET_PROXY_ALLOCATION(pad);
	GST_PAD_SET_PROXY_SCHEDULING(pad);
	element->srcpad = pad;

	element->unit_size = 0;
	element->rate = 0;
	element->array_size = 0;
	memset(&element->workspace, 0, sizeof(element->workspace));
	element->array_index = 0;
	element->samples_in_array = 0;
	element->total_insamples = 0;
	element->need_discont = TRUE;
	gst_base_transform_set_qos_enabled(GST_BASE_TRANSFORM(element), TRUE);
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(element), TRUE);
}

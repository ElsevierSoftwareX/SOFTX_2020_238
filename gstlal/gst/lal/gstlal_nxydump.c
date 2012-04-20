/*
 * An "nxy" dumper to produce files that Grace can read
 *
 * Copyright (C) 2008  Kipp Cannon, Chad Hanna
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
 * ========================================================================
 *
 *                                  Preamble
 *
 * ========================================================================
 */


/*
 * stuff from the C library
 */


#include <math.h>
#include <stdio.h>
#include <string.h>


/*
 * stuff from glib/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>


/*
 * our own stuff
 */


#include <gstlal_nxydump.h>


GST_DEBUG_CATEGORY(gstlal_nxydump_debug);


/*
 * ========================================================================
 *
 *                                 Parameters
 *
 * ========================================================================
 */


#define GST_CAT_DEFAULT gstlal_nxydump_debug
#define DEFAULT_START_TIME 0
#define DEFAULT_STOP_TIME G_MAXUINT64


/*
 * the maximum number of characters it takes to print a timestamp.
 * G_MAXUINT64 / GST_SECOND = 11 digits left of the decimal place, plus 1
 * decimal point, plus 9 digits right of the decimal place.
 */

#define MAX_CHARS_PER_TIMESTAMP 21

/*
 * the maximum number of characters it takes to print the value for one
 * channel including white space, sign characters, etc.;  double-precision
 * floats in "%.16g" format can be upto 23 characters, plus 1 space between
 * columns
 */

#define MAX_CHARS_PER_COLUMN (23 + 1)

/*
 * a newline is sometimes two characters.
 */

#define MAX_EXTRA_BYTES_PER_LINE 2


/*
 * ========================================================================
 *
 *                             Utility Functions
 *
 * ========================================================================
 */


/**
 * compute the number of output bytes to allocate per sample
 */


static size_t src_bytes_per_sample(gint channels)
{
	return MAX_CHARS_PER_TIMESTAMP + channels * MAX_CHARS_PER_COLUMN + MAX_EXTRA_BYTES_PER_LINE;
}


/**
 * Convert a timestamp to a sample offset relative to the timestamp of the
 * start of a buffer, clipped to the buffer boundaries.
 */


static guint64 timestamp_to_sample_clipped(GstClockTime start, guint64 length, gint rate, GstClockTime t)
{
	return t <= start ? 0 : MIN(gst_util_uint64_scale_int_round(t - start, rate, GST_SECOND), length);
}


/**
 * Print the samples from a buffer of double precision floats into a buffer
 * of text.
 */


static GstFlowReturn print_samples(GstBuffer *out, GstClockTime timestamp, const double *samples, int channels, int rate, guint64 length)
{
	char *location = (char *) GST_BUFFER_DATA(out);
	guint64 i;
	int j;

	for(i = 0; i < length; i++) {
		/*
		 * The current timestamp
		 */

		GstClockTime t = timestamp + gst_util_uint64_scale_int_round(i, GST_SECOND, rate);

		/*
		 * Saftey check.
		 */

		g_assert_cmpuint(((guint8 *) location - GST_BUFFER_DATA(out)) + src_bytes_per_sample(channels), <=, GST_BUFFER_SIZE(out));

		/*
		 * Print the time.
		 */

		location += sprintf(location, "%lu.%09u", (unsigned long) (t / GST_SECOND), (unsigned) (t % GST_SECOND));

		/*
		 * Print the channel samples.
		 */

		for(j = 0; j < channels; j++)
			location += sprintf(location, " %.16g", *samples++);

		/*
		 * Finish with a new line
		 */

		location += sprintf(location, "\n");
	}

	/*
	 * Record the actual size of the buffer, but don't bother
	 * realloc()ing.  Note that the final size excludes the \0
	 * terminator.  That's appropriate for strings intended to be
	 * written to a file.
	 */

	GST_BUFFER_SIZE(out) = (guint8 *) location - GST_BUFFER_DATA(out);

	/*
	 * Done
	 */

	return GST_FLOW_OK;
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
		"audio/x-raw-float, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) [1, MAX], " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) 64"
	)
);


static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SRC_NAME,
	GST_PAD_SRC,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		"text/plain"
	)
);


GST_BOILERPLATE(
	GSTLALNXYDump,
	gstlal_nxydump,
	GstBaseTransform,
	GST_TYPE_BASE_TRANSFORM
);


enum property {
	ARG_START_TIME = 1,
	ARG_STOP_TIME
};


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

	str = gst_caps_get_structure(caps, 0);
	if(gst_structure_has_name(str, "text/plain")) {
		*size = 1;
	} else {
		gint channels, width;
		if(!gst_structure_get_int(str, "channels", &channels)) {
			GST_ERROR_OBJECT(trans, "unable to parse channels from %" GST_PTR_FORMAT, caps);
			return FALSE;
		}
		if(!gst_structure_get_int(str, "width", &width)) {
			GST_ERROR_OBJECT(trans, "unable to parse width from %" GST_PTR_FORMAT, caps);
			return FALSE;
		}

		*size = width / 8 * channels;
	}

	return TRUE;
}


/*
 * transform_caps()
 */


static GstCaps *transform_caps(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps)
{
	/*
	 * always return the template caps of the other pad
	 */

	switch(direction) {
	case GST_PAD_SRC:
		caps = gst_caps_copy(gst_pad_get_pad_template_caps(GST_BASE_TRANSFORM_SINK_PAD(trans)));
		break;

	case GST_PAD_SINK:
		caps = gst_caps_copy(gst_pad_get_pad_template_caps(GST_BASE_TRANSFORM_SRC_PAD(trans)));
		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		caps = GST_CAPS_NONE;
		break;
	}

	return caps;
}


/*
 * transform_size()
 */


static gboolean transform_size(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, guint size, GstCaps *othercaps, guint *othersize)
{
	guint unit_size;
	guint other_unit_size;
	gboolean success = TRUE;

	if(gst_structure_has_name(gst_caps_get_structure(caps, 0), "text/plain")) {
		gint channels;
		if(!gst_structure_get_int(gst_caps_get_structure(othercaps, 0), "channels", &channels)) {
			GST_ERROR_OBJECT(trans, "unable to parse channels from %" GST_PTR_FORMAT, othercaps);
			return FALSE;
		}
		unit_size = src_bytes_per_sample(channels);
		if(!get_unit_size(trans, othercaps, &other_unit_size))
			return FALSE;
	} else {
		gint channels;
		if(!get_unit_size(trans, caps, &unit_size))
			return FALSE;
		if(!gst_structure_get_int(gst_caps_get_structure(caps, 0), "channels", &channels)) {
			GST_ERROR_OBJECT(trans, "unable to parse channels from %" GST_PTR_FORMAT, caps);
			return FALSE;
		}
		other_unit_size = src_bytes_per_sample(channels);
	}

	/* do in two steps to prevent optimizer-induced arithmetic bugs */
	*othersize = size / unit_size;
	*othersize *= other_unit_size;

	return success;
}


/*
 * set_caps()
 */


static gboolean set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps)
{
	GSTLALNXYDump *element = GSTLAL_NXYDUMP(trans);
	GstStructure *str = gst_caps_get_structure(incaps, 0);
	gint rate, channels;
	gboolean success = TRUE;

	success &= gst_structure_get_int(str, "rate", &rate);
	success &= gst_structure_get_int(str, "channels", &channels);

	if(success) {
		element->rate = rate;
		element->channels = channels;
	} else
		GST_DEBUG_OBJECT(element, "unable to extract rate and/or channels from caps %" GST_PTR_FORMAT, incaps);

	return success;
}


/*
 * transform()
 */


static GstFlowReturn transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf)
{
	GSTLALNXYDump *element = GSTLAL_NXYDUMP(trans);
	guint64 length;
	guint64 start, stop;
	GstFlowReturn result = GST_FLOW_OK;

	/*
	 * Measure the number of samples.
	 */

	if(!(GST_BUFFER_OFFSET_IS_VALID(inbuf) && GST_BUFFER_OFFSET_END_IS_VALID(inbuf))) {
		GST_ERROR_OBJECT(element, "cannot compute number of input samples:  invalid offset and/or end offset");
		result = GST_FLOW_ERROR;
		goto done;
	}
	length = GST_BUFFER_OFFSET_END(inbuf) - GST_BUFFER_OFFSET(inbuf);

	/*
	 * Compute the desired start and stop samples relative to the start
	 * of this buffer, clipped to the buffer edges.
	 */

	if(GST_BUFFER_TIMESTAMP_IS_VALID(inbuf)) {
		start = timestamp_to_sample_clipped(GST_BUFFER_TIMESTAMP(inbuf), length, element->rate, element->start_time);
		stop = timestamp_to_sample_clipped(GST_BUFFER_TIMESTAMP(inbuf), length, element->rate, element->stop_time);
	} else {
		/* don't know the buffer's start time, go ahead and process
		 * the whole thing */
		start = 0;
		stop = length;
	}

	/*
	 * Set metadata.
	 */

	GST_BUFFER_OFFSET_END(outbuf) = GST_BUFFER_OFFSET_NONE;

	/*
	 * Construct output buffer.
	 */

	if(GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP) || (stop == start)) {
		/*
		 * The input is a gap or we're not going to print any of
		 * the samples --> the output is a gap.
		 */

		GST_BUFFER_FLAG_SET(outbuf, GST_BUFFER_FLAG_GAP);
		GST_BUFFER_SIZE(outbuf) = 0;
	} else {
		/*
		 * Print samples into output buffer.
		 */

		result = print_samples(outbuf, GST_BUFFER_TIMESTAMP(inbuf) + gst_util_uint64_scale_int_round(start, GST_SECOND, element->rate), (const double *) GST_BUFFER_DATA(inbuf) + start * element->channels, element->channels, element->rate, stop - start);
	}

	/*
	 * Done
	 */

done:
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
 * set_property()
 */


static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec)
{
	GSTLALNXYDump *element = GSTLAL_NXYDUMP(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_START_TIME:
		element->start_time = g_value_get_uint64(value);
		break;

	case ARG_STOP_TIME:
		element->stop_time = g_value_get_uint64(value);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * get_property()
 */


static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *pspec)
{
	GSTLALNXYDump *element = GSTLAL_NXYDUMP(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_START_TIME:
		g_value_set_uint64(value, element->start_time);
		break;

	case ARG_STOP_TIME:
		g_value_set_uint64(value, element->stop_time);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * base_init()
 */


static void gstlal_nxydump_base_init(gpointer klass)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(klass);

	gst_element_class_set_details_simple(
		element_class,
		"NXY Dump",
		"Filter",
		"A time-series dumper compatible with Grace's \"nxy\" input format",
		"Kipp Cannon <kipp.cannon@ligo.org>, Chad Hanna <channa@ligo.caltech.edu>"
	);

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));

	transform_class->get_unit_size = GST_DEBUG_FUNCPTR(get_unit_size);
	transform_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
	transform_class->transform = GST_DEBUG_FUNCPTR(transform);
	transform_class->transform_caps = GST_DEBUG_FUNCPTR(transform_caps);
	transform_class->transform_size = GST_DEBUG_FUNCPTR(transform_size);
}


/*
 * class_init()
 */


static void gstlal_nxydump_class_init(GSTLALNXYDumpClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);

	g_object_class_install_property(
		gobject_class,
		ARG_START_TIME,
		g_param_spec_uint64(
			"start-time",
			"Start time",
			"Start time in nanoseconds.",
			0, G_MAXUINT64, DEFAULT_START_TIME,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_STOP_TIME,
		g_param_spec_uint64(
			"stop-time",
			"Stop time",
			"Stop time in nanoseconds.",
			0, G_MAXUINT64, DEFAULT_STOP_TIME,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
}


/*
 * init()
 */


static void gstlal_nxydump_init(GSTLALNXYDump *element, GSTLALNXYDumpClass *klass)
{
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(element), TRUE);
}

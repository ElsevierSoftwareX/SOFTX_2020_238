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


/*
 * our own stuff
 */


#include <gstlal.h>
#include <gstlal_nxydump.h>


/*
 * ========================================================================
 *
 *                                 Parameters
 *
 * ========================================================================
 */


#define DEFAULT_START_TIME 0
#define DEFAULT_STOP_TIME G_MAXUINT64

/*
 * the maximum number of characters it takes to print a timestamp.
 * G_MAXUINT64 / GST_SECOND = 11 digits left of the decimal place, plus 1
 * decimal point, plus 9 digits right of the decimal place.
 */

#define MAX_BYTES_PER_TIMESTAMP 21

/*
 * the maximum number of characters it takes to print the value for one
 * channel including white space, sign characters, etc.;  double-precision
 * floats in "%.16g" format can be upto 23 characters, plus 1 space between
 * columns
 */

#define MAX_BYTES_PER_COLUMN 23 + 1

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
 * Convert a time stamp and a sample rate into a sample offset relative to
 * the time stamp of the start of a buffer.
 */


static guint64 timestamp_to_sample_clipped(GstClockTime start, guint64 length, gint rate, GstClockTime t)
{
	guint64 offset;

	if(t < start)
		return 0;

	offset = gst_util_uint64_scale_int_round(t - start, rate, GST_SECOND);
	return MIN(offset, length);
}


/**
 * Construct an empty buffer flagged as a gap and push it on the pad.  The
 * buffer's time stamp, etc., will be derived from the template, and will
 * be constructed so as to span the interval in the stream corresponding to
 * the samples [start, stop) relative to the template's time stamp given
 * the sample rate.
 */


static GstFlowReturn push_gap(GstPad *pad, const GstBuffer *template, gint rate, guint64 start, guint64 stop)
{
	GstFlowReturn result = GST_FLOW_OK;
	GstBuffer *buf;

	result = gst_pad_alloc_buffer(pad, GST_BUFFER_OFFSET_NONE, 0, GST_PAD_CAPS(pad), &buf);
	if(result != GST_FLOW_OK) {
		GST_ERROR("gst_pad_alloc_buffer() failed allocating gap buffer");
		return result;
	}

	gst_buffer_copy_metadata(buf, template, GST_BUFFER_COPY_FLAGS);
	GST_BUFFER_FLAG_SET(buf, GST_BUFFER_FLAG_GAP);
	GST_BUFFER_OFFSET_END(buf) = GST_BUFFER_OFFSET_NONE;
	if(GST_BUFFER_TIMESTAMP_IS_VALID(template)) {
		GST_BUFFER_TIMESTAMP(buf) = GST_BUFFER_TIMESTAMP(template) + gst_util_uint64_scale_int_round(start, GST_SECOND, rate);
		GST_BUFFER_DURATION(buf) = GST_BUFFER_TIMESTAMP(template) + gst_util_uint64_scale_int_round(stop, GST_SECOND, rate) - GST_BUFFER_TIMESTAMP(buf);
	} else
		GST_BUFFER_TIMESTAMP(buf) = GST_BUFFER_DURATION(buf) = GST_CLOCK_TIME_NONE;

	result = gst_pad_push(pad, buf);
	if(result != GST_FLOW_OK) {
		GST_ERROR("gst_pad_push() failed pushing gap buffer");
		return result;
	}

	return result;
}


/**
 * Print the samples from a buffer of double precision floats into a buffer
 * of text.
 */


static GstFlowReturn print_samples(GstBuffer *out, const double *samples, int channels, int rate, guint64 start, guint64 stop)
{
	char *location = (char *) GST_BUFFER_DATA(out);
	guint64 i;
	int j;

	samples += channels * start;

	for(i = start; i < stop; i++) {
		/*
		 * The current timestamp
		 */

		GstClockTime t = GST_BUFFER_TIMESTAMP(out) + gst_util_uint64_scale_int_round(i - start, GST_SECOND, rate);

		/*
		 * Saftey check.
		 */

		g_assert((guint8 *) location - GST_BUFFER_DATA(out) + MAX_BYTES_PER_TIMESTAMP + channels * MAX_BYTES_PER_COLUMN + MAX_EXTRA_BYTES_PER_LINE <= GST_BUFFER_SIZE(out));

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
 *                             GStreamer Element
 *
 * ============================================================================
 */


/*
 * Properties
 */


enum property {
	ARG_START_TIME = 1,
	ARG_STOP_TIME
};


static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec)
{
	GSTLALNXYDump *element = GSTLAL_NXYDUMP(object);

	switch(id) {
	case ARG_START_TIME:
		element->start_time = g_value_get_uint64(value);
		break;

	case ARG_STOP_TIME:
		element->stop_time = g_value_get_uint64(value);
		break;
	}
}


static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *pspec)
{
	GSTLALNXYDump *element = GSTLAL_NXYDUMP(object);

	switch(id) {
	case ARG_START_TIME:
		g_value_set_uint64(value, element->start_time);
		break;

	case ARG_STOP_TIME:
		g_value_set_uint64(value, element->stop_time);
		break;
	}
}


/*
 * setcaps()
 */


static gboolean setcaps(GstPad *pad, GstCaps *caps)
{
	GSTLALNXYDump *element = GSTLAL_NXYDUMP(gst_pad_get_parent(pad));
	GstStructure *structure = gst_caps_get_structure(caps, 0);
	gint rate, channels;
	gboolean success = TRUE;

	success &= gst_structure_get_int(structure, "rate", &rate);
	success &= gst_structure_get_int(structure, "channels", &channels);

	if(success) {
		element->rate = rate;
		element->channels = channels;
	} else
		GST_DEBUG_OBJECT(element, "unable to extract rate and/or channels from caps %" GST_PTR_FORMAT, caps);

	gst_object_unref(element);
	return success;
}


/*
 * chain()
 */


static GstFlowReturn chain(GstPad *pad, GstBuffer *sinkbuf)
{
	GSTLALNXYDump *element = GSTLAL_NXYDUMP(gst_pad_get_parent(pad));
	GstCaps *caps = gst_buffer_get_caps(sinkbuf);
	GstBuffer *srcbuf;
	GstFlowReturn result = GST_FLOW_OK;
	guint64 length;
	guint64 start, stop;

	/*
	 * Measure the number of samples.
	 */

	if(!(GST_BUFFER_OFFSET_IS_VALID(sinkbuf) && GST_BUFFER_OFFSET_END_IS_VALID(sinkbuf))) {
		GST_ERROR_OBJECT(element, "cannot compute number of input samples:  invalid offset and/or end offset");
		result = GST_FLOW_ERROR;
		goto done;
	}
	length = GST_BUFFER_OFFSET_END(sinkbuf) - GST_BUFFER_OFFSET(sinkbuf);

	/*
	 * Compute the desired start and stop samples relative to the start
	 * of this buffer, clipped to the buffer edges.
	 */

	if(GST_BUFFER_TIMESTAMP_IS_VALID(sinkbuf)) {
		start = timestamp_to_sample_clipped(GST_BUFFER_TIMESTAMP(sinkbuf), length, element->rate, element->start_time);
		stop = timestamp_to_sample_clipped(GST_BUFFER_TIMESTAMP(sinkbuf), length, element->rate, element->stop_time);
	} else {
		/* don't know the buffer's start time, go ahead and process
		 * the whole thing */
		start = 0;
		stop = length;
	}

	/*
	 * If we don't need any of the samples from this buffer or it's a
	 * gap then we're done --> push gap buffer downstream.
	 */

	if(GST_BUFFER_FLAG_IS_SET(sinkbuf, GST_BUFFER_FLAG_GAP) || (stop == start)) {
		result = push_gap(element->srcpad, sinkbuf, element->rate, 0, length);
		goto done;
	}

	/*
	 * If start != 0, push a gap buffer to precede the data.
	 */

	if(start) {
		result = push_gap(element->srcpad, sinkbuf, element->rate, 0,  start);
		if(result != GST_FLOW_OK)
			goto done;
	}

	/*
	 * Start an output buffer.  Assume an additional channel for the
	 * time stamps.  If the buffer isn't big enough, it will be resized
	 * later.
	 */

	result = gst_pad_alloc_buffer(element->srcpad, GST_BUFFER_OFFSET_NONE, (stop - start) * (MAX_BYTES_PER_TIMESTAMP + element->channels * MAX_BYTES_PER_COLUMN + MAX_EXTRA_BYTES_PER_LINE), GST_PAD_CAPS(element->srcpad), &srcbuf);
	if(result != GST_FLOW_OK) {
		GST_ERROR_OBJECT(element, "failure allocating output buffer");
		goto done;
	}

	/*
	 * Set metadata.
	 */

	gst_buffer_copy_metadata(srcbuf, sinkbuf, GST_BUFFER_COPY_FLAGS);
	GST_BUFFER_OFFSET_END(srcbuf) = GST_BUFFER_OFFSET_NONE;
	GST_BUFFER_TIMESTAMP(srcbuf) = GST_BUFFER_TIMESTAMP(sinkbuf) + gst_util_uint64_scale_int_round(start, GST_SECOND, element->rate);
	GST_BUFFER_DURATION(srcbuf) = GST_BUFFER_TIMESTAMP(sinkbuf) + gst_util_uint64_scale_int_round(stop, GST_SECOND, element->rate) - GST_BUFFER_TIMESTAMP(srcbuf);

	/*
	 * Print samples into output buffer.
	 */

	result = print_samples(srcbuf, (const double *) GST_BUFFER_DATA(sinkbuf), element->channels, element->rate, start, stop);
	if(result != GST_FLOW_OK) {
		gst_buffer_unref(srcbuf);
		goto done;
	}

	/*
	 * Push the buffer downstream.
	 */

	result = gst_pad_push(element->srcpad, srcbuf);
	if(result != GST_FLOW_OK)
		goto done;

	/*
	 * If stop != samples, push a gap buffer to pad the output stream
	 * up to the end of the input buffer.
	 */

	if(stop < length) {
		result = push_gap(element->srcpad, sinkbuf, element->rate, stop,  length);
		if(result != GST_FLOW_OK)
			goto done;
	}

	/*
	 * Done
	 */

done:
	gst_caps_unref(caps);
	gst_buffer_unref(sinkbuf);
	gst_object_unref(element);
	return result;
}


/*
 * Parent class.
 */


static GstElementClass *parent_class = NULL;


/*
 * Instance finalize function.  See ???
 */


static void finalize(GObject *object)
{
	GSTLALNXYDump *element = GSTLAL_NXYDUMP(object);

	gst_object_unref(element->srcpad);

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


/*
 * Base init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GBaseInitFunc
 */


static void base_init(gpointer class)
{
	static GstElementDetails plugin_details = {
		"NXY Dump",
		"Filter",
		"A time-series dumper compatible with Grace's \"nxy\" input format",
		"Kipp Cannon <kipp.cannon@ligo.org>, Chad Hanna <channa@ligo.caltech.edu>"
	};
	GstElementClass *element_class = GST_ELEMENT_CLASS(class);

	gst_element_class_set_details(element_class, &plugin_details);

	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"sink",
			GST_PAD_SINK,
			GST_PAD_ALWAYS,
			gst_caps_new_simple(
				"audio/x-raw-float",
				"rate", GST_TYPE_INT_RANGE, 1, G_MAXINT,
				"channels", GST_TYPE_INT_RANGE, 1, G_MAXINT,
				"endianness", G_TYPE_INT, G_BYTE_ORDER,
				"width", G_TYPE_INT, 64,
				NULL
			)
		)
	);
	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"src",
			GST_PAD_SRC,
			GST_PAD_ALWAYS,
			gst_caps_new_simple(
				"text/plain",
				NULL
			)
		)
	);
}


/*
 * Class init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GClassInitFunc
 */


static void class_init(gpointer class, gpointer class_data)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(class);

	parent_class = g_type_class_ref(GST_TYPE_ELEMENT);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	g_object_class_install_property(
		gobject_class,
		ARG_START_TIME,
		g_param_spec_uint64(
			"start-time",
			"Start time",
			"Start time in nanoseconds.",
			0, G_MAXUINT64, DEFAULT_START_TIME,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
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
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);
}


/*
 * Instance init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GInstanceInitFunc
 */


static void instance_init(GTypeInstance *object, gpointer class)
{
	GSTLALNXYDump *element = GSTLAL_NXYDUMP(object);
	GstPad *pad;

	gst_element_create_all_pads(GST_ELEMENT(element));

	/* configure sink pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "sink");
	gst_pad_set_setcaps_function(pad, GST_DEBUG_FUNCPTR(setcaps));
	gst_pad_set_chain_function(pad, GST_DEBUG_FUNCPTR(chain));
	gst_object_unref(pad);

	/* retrieve (and ref) src pad */
	element->srcpad = gst_element_get_static_pad(GST_ELEMENT(element), "src");

	/* internal data */
	element->rate = 0;
	element->channels = 0;
	element->start_time = DEFAULT_START_TIME;
	element->stop_time = DEFAULT_STOP_TIME;
}


/*
 * gstlal_nxydump_get_type().
 */


GType gstlal_nxydump_get_type(void)
{
	static GType type = 0;

	if(!type) {
		static const GTypeInfo info = {
			.class_size = sizeof(GSTLALNXYDumpClass),
			.class_init = class_init,
			.base_init = base_init,
			.instance_size = sizeof(GSTLALNXYDump),
			.instance_init = instance_init,
		};
		type = g_type_register_static(GST_TYPE_ELEMENT, "lal_nxydump", &info, 0);
	}

	return type;
}

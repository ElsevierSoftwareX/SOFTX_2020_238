/*
 * An "nxy" dumper to produce files that Grace can read
 *
 * Copyright (C) 2008  Kipp Cannon, Chad Hanna
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
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
#include <string.h>


/*
 * stuff from glib/gstreamer
 */


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


/*
 * ========================================================================
 *
 *                             Utility Functions
 *
 * ========================================================================
 */


static int timestamp_to_sample_clipped(GstClockTime start, int samples, int sample_rate, GstClockTime t)
{
	if(t <= start)
		return 0;
	t -= start;
	if((long) t > samples * GST_SECOND / sample_rate + 1)
		return samples;
	return t * sample_rate / GST_SECOND;
}


static GstFlowReturn push_gap(GstPad *pad, const GstBuffer *template, int sample_rate, int start, int stop)
{
	GstFlowReturn result = GST_FLOW_OK;
	GstBuffer *buf;

	result = gst_pad_alloc_buffer(pad, GST_BUFFER_OFFSET(template) + start, 0, GST_PAD_CAPS(pad), &buf);
	if(result != GST_FLOW_OK) {
		GST_ERROR("gst_pad_alloc_buffer() failed allocating gap buffer");
		return result;
	}

	gst_buffer_copy_metadata(buf, template, GST_BUFFER_COPY_FLAGS | GST_BUFFER_COPY_TIMESTAMPS);
	GST_BUFFER_FLAG_SET(buf, GST_BUFFER_FLAG_GAP);
	GST_BUFFER_OFFSET_END(buf) = GST_BUFFER_OFFSET(buf) + (stop - start) - 1;
	GST_BUFFER_TIMESTAMP(buf) = GST_BUFFER_TIMESTAMP(template) + start * GST_SECOND / sample_rate;
	GST_BUFFER_DURATION(buf) = (stop - start) * GST_SECOND / sample_rate;

	result = gst_pad_push(pad, buf);
	if(result != GST_FLOW_OK) {
		GST_ERROR("gst_pad_push() failed pushing gap buffer");
		return result;
	}

	return result;
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
		element->start_time = g_value_get_int64(value);
		break;

	case ARG_STOP_TIME:
		element->stop_time = g_value_get_int64(value);
		break;
	}
}


static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *pspec)
{
	GSTLALNXYDump *element = GSTLAL_NXYDUMP(object);

	switch(id) {
	case ARG_START_TIME:
		g_value_set_int64(value, element->start_time);
		break;

	case ARG_STOP_TIME:
		g_value_set_int64(value, element->stop_time);
		break;
	}
}


/*
 * setcaps()
 */


static gboolean setcaps(GstPad *pad, GstCaps *caps)
{
	GSTLALNXYDump *element = GSTLAL_NXYDUMP(gst_pad_get_parent(pad));
	gboolean result = TRUE;

	element->sample_rate = g_value_get_int(gst_structure_get_value(gst_caps_get_structure(caps, 0), "rate"));

	gst_object_unref(element);
	return result;
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
	int channels;
	int samples;
	int start, stop;
	char *location;
	int i, j;

	/*
	 * Retrieve the number of channels, and measure the number of
	 * samples.
	 */

	channels = g_value_get_int(gst_structure_get_value(gst_caps_get_structure(caps, 0), "channels"));
	samples = GST_BUFFER_SIZE(sinkbuf) / sizeof(double) / channels;

	/*
	 * Compute the desired start and stop samples relative to the start
	 * of this buffer, clipped to the buffer edges.
	 */

	if(GST_BUFFER_TIMESTAMP(sinkbuf) != GST_CLOCK_TIME_NONE) {
		start = timestamp_to_sample_clipped(GST_BUFFER_TIMESTAMP(sinkbuf), samples, element->sample_rate, element->start_time);
		stop = timestamp_to_sample_clipped(GST_BUFFER_TIMESTAMP(sinkbuf), samples, element->sample_rate, element->stop_time);
	} else {
		/* don't know the buffer's start time, go ahead and process
		 * the whole thing */
		start = 0;
		stop = samples;
	}

	/*
	 * If we don't need any of the samples from this buffer, we're
	 * done --> push gap buffer downstream.
	 */

	if(stop == start) {
		result = push_gap(element->srcpad, sinkbuf, element->sample_rate, 0, samples);
		goto done;
	}

	/*
	 * If start != 0, need to push a gap buffer.
	 */

	if(start) {
		result = push_gap(element->srcpad, sinkbuf, element->sample_rate, 0,  start);
		if(result != GST_FLOW_OK)
			goto done;
	}

	/*
	 * Start an output buffer.  Assume 24 bytes per channel per sample,
	 * with an additional channel for the time stamps.  It will be
	 * resized if it's not big enough.
	 */

	srcbuf = gst_buffer_new_and_alloc((channels + 1) * (stop - start) * 24);
	if(!srcbuf) {
		GST_ERROR_OBJECT(element, "failure allocating output buffer");
		result = GST_FLOW_ERROR;
		goto done;
	}

	/*
	 * Set metadata.
	 */

	gst_buffer_copy_metadata(srcbuf, sinkbuf, GST_BUFFER_COPY_FLAGS);
	GST_BUFFER_OFFSET(srcbuf) = GST_BUFFER_OFFSET_NONE;
	GST_BUFFER_OFFSET_END(srcbuf) = GST_BUFFER_OFFSET_NONE;
	GST_BUFFER_TIMESTAMP(srcbuf) = GST_BUFFER_TIMESTAMP(sinkbuf) + start * GST_SECOND / element->sample_rate;
	GST_BUFFER_DURATION(srcbuf) = (stop - start) * GST_SECOND / element->sample_rate;
	gst_buffer_set_caps(srcbuf, GST_PAD_CAPS(element->srcpad));

	/*
	 * Print samples into output buffer.  Note that the final size
	 * excludes the \0 terminator.  That's appropriate for strings
	 * intended to be written to a file.
	 */

	location = (char *) GST_BUFFER_DATA(srcbuf);
	for(i = start; i < stop; i++) {
		GstClockTime t = GST_BUFFER_TIMESTAMP(sinkbuf) + GST_SECOND * i / element->sample_rate;

		if((guint8 *) location - GST_BUFFER_DATA(srcbuf) + 1000 >= GST_BUFFER_SIZE(srcbuf)) {
			guint8 *new = realloc(GST_BUFFER_DATA(srcbuf), GST_BUFFER_SIZE(srcbuf) + 1024);
			if(!new)
				break;
			GST_BUFFER_DATA(srcbuf) = GST_BUFFER_MALLOCDATA(srcbuf) = new;
			GST_BUFFER_SIZE(srcbuf) += 1024;
		}

		location += sprintf(location, "%d.%09u", (int) (t / GST_SECOND), (unsigned) (t % GST_SECOND));
		for(j = 0; j < channels; j++)
			location += sprintf(location, " %.16g", *((double *) GST_BUFFER_DATA(sinkbuf) + i * channels + j));
		location += sprintf(location, "\n");
	}
	GST_BUFFER_SIZE(srcbuf) = (guint8 *) location - GST_BUFFER_DATA(srcbuf);

	/*
	 * Push the buffer downstream
	 */

	result = gst_pad_push(element->srcpad, srcbuf);
	if(result != GST_FLOW_OK)
		goto done;

	/*
	 * If stop != samples, finish with a gap buffer.
	 */

	if(stop != samples) {
		result = push_gap(element->srcpad, sinkbuf, element->sample_rate, stop,  samples);
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
 * Instance dispose function.  See ???
 */


static void dispose(GObject *object)
{
	GSTLALNXYDump *element = GSTLAL_NXYDUMP(object);

	gst_object_unref(element->srcpad);

	G_OBJECT_CLASS(parent_class)->dispose(object);
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
		"Kipp Cannon <kcannon@ligo.caltech.edu>, Chan Hanna <channa@ligo.caltech.edu>"
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

	gobject_class->set_property = set_property;
	gobject_class->get_property = get_property;
	gobject_class->dispose = dispose;

	/* FIXME:  "string" is not the best type for these ... */
	g_object_class_install_property(gobject_class, ARG_START_TIME, g_param_spec_int64("start-time", "Start time", "Start time in nanoseconds.", 0, G_MAXINT64, 0, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	g_object_class_install_property(gobject_class, ARG_STOP_TIME, g_param_spec_int64("stop-time", "Stop time", "Stop time in seconds.", 0, G_MAXINT64, G_MAXINT64, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
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
	gst_pad_set_setcaps_function(pad, setcaps);
	gst_pad_set_chain_function(pad, chain);
	gst_object_unref(pad);

	/* retrieve (and ref) src pad */
	element->srcpad = gst_element_get_static_pad(GST_ELEMENT(element), "src");

	/* internal data */
	element->sample_rate = 0;
	element->start_time = 0;
	element->stop_time = 0;
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

/*
 * A multi-channel scope.
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
#include <stdint.h>


/*
 * stuff from gstreamer
 */


#include <gst/gst.h>
#include <gst/base/gstadapter.h>


/*
 * stuff from GSL
 */


#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>


/*
 * our own stuff
 */


#include <gstlal.h>
#include <gstlal_multiscope.h>


/*
 * ========================================================================
 *
 *                                 Parameters
 *
 * ========================================================================
 */


#define SCOPE_WIDTH 320
#define SCOPE_HEIGHT 200
#define PIXELS_PER_SIGMA 20
#define DEFAULT_TRACE_DURATION 1.0
#define DEFAULT_AVERAGE_LENGTH 1000


/*
 * ========================================================================
 *
 *                             Utility Functions
 *
 * ========================================================================
 */


static uint32_t pixel_colour(int n_channels, int channel)
{
	uint32_t colour = 0x7f * (n_channels - channel) / n_channels;

	return 0x001010ff | (colour << 8) | (colour << 16);
}


/*
 * ============================================================================
 *
 *                          GStreamer Source Element
 *
 * ============================================================================
 */


/*
 * Properties
 */


enum property {
	ARG_WIDTH = 1,
	ARG_HEIGHT,
	ARG_TRACE_DURATION
};


static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec)
{
	GSTLALMultiScope *element = GSTLAL_MULTISCOPE(object);

	switch(id) {
	}
}


static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *pspec)
{
	GSTLALMultiScope *element = GSTLAL_MULTISCOPE(object);

	switch(id) {
	}
}


/*
 * setcaps()
 */


static gboolean setcaps(GstPad *pad, GstCaps *caps)
{
	GSTLALMultiScope *element = GSTLAL_MULTISCOPE(gst_pad_get_parent(pad));
	gboolean result = TRUE;

	/*
	 * Extract the sample rate and channel count from the input
	 * buffer's caps
	 */

	element->rate = g_value_get_int(gst_structure_get_value(gst_caps_get_structure(caps, 0), "rate"));
	element->channels = g_value_get_int(gst_structure_get_value(gst_caps_get_structure(caps, 0), "channels"));

	/*
	 * Try a setcaps on the src pad
	 */

	caps = gst_pad_get_allowed_caps(element->srcpad);
	gst_caps_do_simplify(caps);
	result = gst_pad_set_caps(element->srcpad, caps);
	gst_caps_unref(caps);

	gst_object_unref(element);
	return result;
}


/*
 * chain()
 */


static GstFlowReturn chain(GstPad *pad, GstBuffer *sinkbuf)
{
	GSTLALMultiScope *element = GSTLAL_MULTISCOPE(gst_pad_get_parent(pad));
	GstCaps *caps = gst_buffer_get_caps(sinkbuf);
	GstFlowReturn result = GST_FLOW_OK;
	int samples = element->trace_duration * element->rate;
	uint32_t *pixels;

	/*
	 * Put buffer into adapter, and measure the length of the SNR time
	 * series we can generate (we're done if this is <= 0).
	 */

	gst_adapter_push(element->adapter, sinkbuf);

	/*
	 * Loop while data's available.
	 */

	while(gst_adapter_available(element->adapter) >= samples * element->channels * sizeof(double)) {
		double *data = (double *) gst_adapter_peek(element->adapter, samples * element->channels * sizeof(*data));
		double *d;
		GstBuffer *srcbuf;
		int i, j;

		/*
		 * Get a buffer from the downstream peer
		 */

		result = gst_pad_alloc_buffer(element->srcpad, element->next_sample, SCOPE_WIDTH * SCOPE_HEIGHT * sizeof(*pixels), GST_PAD_CAPS(element->srcpad), &srcbuf);
		if(result != GST_FLOW_OK)
			goto done;
		pixels = (uint32_t *) GST_BUFFER_DATA(srcbuf);

		GST_BUFFER_OFFSET_END(srcbuf) = GST_BUFFER_OFFSET(srcbuf) + samples - 1;
		/*GST_BUFFER_TIMESTAMP(srcbuf) = (GstClockTime) GST_BUFFER_OFFSET(srcbuf) * 1000000000 / element->rate;
		GST_BUFFER_DURATION(srcbuf) = (GstClockTime) samples * 1000000000 / element->rate;*/
		GST_BUFFER_TIMESTAMP(srcbuf) = GST_CLOCK_TIME_NONE;
		GST_BUFFER_DURATION(srcbuf) = GST_CLOCK_TIME_NONE;

		/*
		 * Set the buffer to all white
		 */

		for(i = 0; i < SCOPE_WIDTH * SCOPE_HEIGHT; i++)
			/* white = red | green | blue */
			pixels[i] = 0x00ff0000 | 0x0000ff00 | 0x000000ff;

		/*
		 * Update the trace mean and variance
		 */

		d = data;
		for(i = 0; i < samples * element->channels; i++) {
			element->variance = (element->variance * (element->average_length - 1) + pow(*d - element->mean, 2)) / element->average_length;
			element->mean = (element->mean * (element->average_length - 1) + *d) / element->average_length;
			d++;
		}

		/*
		 * Draw the traces
		 */

		d = data;
		for(i = 0; i < samples; i++) {
			int x = i * SCOPE_WIDTH / samples;
			for(j = 0; j < element->channels; j++) {
				int y = PIXELS_PER_SIGMA * (*(d++) - element->mean) / sqrt(element->variance) + 0.5;
				y = SCOPE_HEIGHT / 2 - y;
				if(0 <= y && y < SCOPE_HEIGHT)
					pixels[y * SCOPE_WIDTH + x] = pixel_colour(element->channels, j);
			}
		}

		/*
		 * Push the buffer downstream.
		 */

		result = gst_pad_push(element->srcpad, srcbuf);
		if(result != GST_FLOW_OK)
			goto done;

		/*
		 * Flush the data from the adapter.
		 */

		gst_adapter_flush(element->adapter, samples * element->channels * sizeof(*data));
		element->next_sample += samples;
	}

	/*
	 * Done
	 */

done:
	gst_caps_unref(caps);
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
	GSTLALMultiScope *element = GSTLAL_MULTISCOPE(object);

	g_object_unref(element->adapter);
	element->adapter = NULL;

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
		"Multi-scope",
		"Filter",
		"A multi-channel scope",
		"Kipp Cannon <kcannon@ligo.caltech.edu>, Chan Hanna <chann@ligo.caltech.edu>"
	};
	GstElementClass *element_class = GST_ELEMENT_CLASS(class);
	GstPadTemplate *sinkpad_template = gst_pad_template_new(
		"sink",
		GST_PAD_SINK,
		GST_PAD_ALWAYS,
		gst_caps_new_simple(
			"audio/x-raw-float",
			"rate", GST_TYPE_INT_RANGE, 1, 32768,
			"channels", GST_TYPE_INT_RANGE, 1, 16384,
			"endianness", G_TYPE_INT, G_BYTE_ORDER,
			"width", G_TYPE_INT, 64,
			NULL
		)
	);
	GstPadTemplate *srcpad_template = gst_pad_template_new(
		"src",
		GST_PAD_SRC,
		GST_PAD_ALWAYS,
		gst_caps_new_simple(
			"video/x-raw-rgb",
			"width", G_TYPE_INT, SCOPE_WIDTH,
			"height", G_TYPE_INT, SCOPE_HEIGHT,
			"framerate", GST_TYPE_FRACTION, 60, 1,
			"bpp", G_TYPE_INT, 32,
			"depth", G_TYPE_INT, 24,
			"red_mask", G_TYPE_INT, 0x0000ff00,
			"green_mask", G_TYPE_INT, 0x00ff0000,
			"blue_mask", G_TYPE_INT, 0xff000000,
			"endianness", G_TYPE_INT, 4321,
			NULL
		)
	);

	gst_element_class_set_details(element_class, &plugin_details);

	gst_element_class_add_pad_template(element_class, sinkpad_template);
	gst_element_class_add_pad_template(element_class, srcpad_template);
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
}


/*
 * Instance init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GInstanceInitFunc
 */


static void instance_init(GTypeInstance *object, gpointer class)
{
	GSTLALMultiScope *element = GSTLAL_MULTISCOPE(object);
	GstPad *pad;

	gst_element_create_all_pads(GST_ELEMENT(element));

	/* configure sink pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "sink");
	gst_pad_set_setcaps_function(pad, setcaps);
	gst_pad_set_chain_function(pad, chain);
	gst_object_unref(pad);

	/* src pads */
	pad = gst_element_get_static_pad(GST_ELEMENT(object), "src");

	/* consider this to consume the refernece */
	element->srcpad = pad;

	/* internal data */
	element->adapter = gst_adapter_new();
	element->channels = 0;
	element->rate = 0;
	element->trace_duration = DEFAULT_TRACE_DURATION;
	element->next_sample = 0;
	element->mean = 0.0;
	element->variance = 0.0;
	element->average_length = DEFAULT_AVERAGE_LENGTH;
}


/*
 * gstlal_templatebank_get_type().
 */


GType gstlal_multiscope_get_type(void)
{
	static GType type = 0;

	if(!type) {
		static const GTypeInfo info = {
			.class_size = sizeof(GSTLALMultiScopeClass),
			.class_init = class_init,
			.base_init = base_init,
			.instance_size = sizeof(GSTLALMultiScope),
			.instance_init = instance_init,
		};
		type = g_type_register_static(GST_TYPE_ELEMENT, "lal_multiscope", &info, 0);
	}

	return type;
}

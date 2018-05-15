/*
 * GStreamer
 * Copyright (C) 2005 Thomas Vander Stichele <thomas@apestaart.org>
 * Copyright (C) 2005 Ronald S. Bultje <rbultje@ronald.bitfreak.net>
 * Copyright (C) 2018 Chad Hanna <<chad.hanna@ligo.org>>
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Alternatively, the contents of this file may be used under the
 * GNU Lesser General Public License Version 2.1 (the "LGPL"), in
 * which case the following provisions apply instead of the ones
 * mentioned above:
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */


#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif
#include <stdio.h>
#include <math.h>
#include <gst/gst.h>
#include <gst/audio/audio.h>

#include "gstlaldeglitchfilter.h"
#include <gstlal/gstlal_segments.h>
#include <lal/Window.h>

GST_DEBUG_CATEGORY_STATIC (gst_laldeglitch_filter_debug);
#define GST_CAT_DEFAULT gst_laldeglitch_filter_debug

/* Filter signals and args */
enum
{
	/* FILL ME */
	LAST_SIGNAL
};

enum
{
	PROP_0,
	ARG_SEGMENT_LIST
};

/* the capabilities of the inputs and outputs.
 *
 * describe the real formats here.
 */
static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE ("sink",
	GST_PAD_SINK,
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

static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE ("src",
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

#define gst_laldeglitch_filter_parent_class parent_class
G_DEFINE_TYPE (GstLALDeglitchFilter, gst_laldeglitch_filter, GST_TYPE_ELEMENT);

static void gst_laldeglitch_filter_set_property (GObject * object, guint prop_id, const GValue * value, GParamSpec * pspec);

static void gst_laldeglitch_filter_get_property (GObject * object, guint prop_id, GValue * value, GParamSpec * pspec);

static gboolean gst_laldeglitch_filter_sink_event (GstPad * pad, GstObject * parent, GstEvent * event);

static GstFlowReturn gst_laldeglitch_filter_chain (GstPad * pad, GstObject * parent, GstBuffer * buf);

/* GObject vmethod implementations */

static void finalize(GObject *object)
{
	GstLALDeglitchFilter *filter = GST_LALDEGLITCHFILTER(object);

	/*
	* free resources
	*/

	gstlal_segment_list_free(filter->seglist);
	filter->seglist = NULL;
	g_mutex_clear(&filter->segment_matrix_lock);

	/*
	* chain to parent class' finalize() method
	*/

	G_OBJECT_CLASS(gst_laldeglitch_filter_parent_class)->finalize(object);
}


/* initialize the laldeglitchfilter's class */
static void
gst_laldeglitch_filter_class_init (GstLALDeglitchFilterClass * klass)
{
	GObjectClass *gobject_class;
	GstElementClass *gstelement_class;

	gobject_class = (GObjectClass *) klass;
	gstelement_class = (GstElementClass *) klass;

	gobject_class->set_property = gst_laldeglitch_filter_set_property;
	gobject_class->get_property = gst_laldeglitch_filter_get_property;

	g_object_class_install_property(
		gobject_class,
		ARG_SEGMENT_LIST,
		g_param_spec_value_array(
			"segment-list",
			"Segment List",
			"List of Segments.  This is an Nx2 array where N (the rows) is the number of segments. The columns are the start and stop times of each segment.",
				g_param_spec_value_array(
				"segment",
				"[start, stop)",
				"Start and stop time of segment.",
					g_param_spec_uint64(
					"time",
					"Time",
					"Time (in nanoseconds)",
					0, G_MAXUINT64, 0,
					G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
					),
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
			),
		G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);



	gst_element_class_set_details_simple(gstelement_class,
		"LALDeglitchFilter",
		"Removes glitches",
		"Removes glitches",
		"Chad Hanna <<chad.hanna@ligo.org>>");

	gst_element_class_add_pad_template (gstelement_class,
	gst_static_pad_template_get (&src_factory));
	gst_element_class_add_pad_template (gstelement_class,
	gst_static_pad_template_get (&sink_factory));

	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);
}

/* initialize the new element
 * instantiate pads and add them to element
 * set pad calback functions
 * initialize instance structure
 */
static void
gst_laldeglitch_filter_init (GstLALDeglitchFilter * filter)
{
	filter->sinkpad = gst_pad_new_from_static_template (&sink_factory, "sink");
	gst_pad_set_event_function (filter->sinkpad, GST_DEBUG_FUNCPTR(gst_laldeglitch_filter_sink_event));
	gst_pad_set_chain_function (filter->sinkpad, GST_DEBUG_FUNCPTR(gst_laldeglitch_filter_chain));
	GST_PAD_SET_PROXY_CAPS (filter->sinkpad);
	gst_element_add_pad (GST_ELEMENT (filter), filter->sinkpad);

	filter->srcpad = gst_pad_new_from_static_template (&src_factory, "src");
	GST_PAD_SET_PROXY_CAPS (filter->srcpad);
	gst_element_add_pad (GST_ELEMENT (filter), filter->srcpad);

	filter->seglist = NULL;
	filter->rate = 0;
	filter->width = 0;
	g_mutex_init(&filter->segment_matrix_lock);
}

static void
gst_laldeglitch_filter_set_property (GObject * object, guint prop_id, const GValue * value, GParamSpec * pspec)
{
	GstLALDeglitchFilter *filter = GST_LALDEGLITCHFILTER (object);

	switch (prop_id) {

		case ARG_SEGMENT_LIST:
			g_mutex_lock(&filter->segment_matrix_lock);
			gstlal_segment_list_free(filter->seglist);
			filter->seglist = gstlal_segment_list_from_g_value_array(g_value_get_boxed(value));
			g_mutex_unlock(&filter->segment_matrix_lock);
			break;

		default:
			G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
			break;
	}
}

static void
gst_laldeglitch_filter_get_property (GObject * object, guint prop_id, GValue * value, GParamSpec * pspec)
{
	GstLALDeglitchFilter *filter = GST_LALDEGLITCHFILTER (object);

	switch (prop_id) {

		case ARG_SEGMENT_LIST:
			g_mutex_lock(&filter->segment_matrix_lock);
			if(filter->seglist)
				g_value_take_boxed(value, g_value_array_from_gstlal_segment_list(filter->seglist));
			/* FIXME:  else? */
			g_mutex_unlock(&filter->segment_matrix_lock);
			break;

		default:
			G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
			break;
	}
}

/* GstElement vmethod implementations */

/* this function handles sink events */
static gboolean gst_laldeglitch_filter_sink_event (GstPad * pad, GstObject * parent, GstEvent * event)
{
	GstLALDeglitchFilter *filter;
	gboolean ret;

	filter = GST_LALDEGLITCHFILTER (parent);

	GST_LOG_OBJECT (filter, "Received %s event: %" GST_PTR_FORMAT,
	GST_EVENT_TYPE_NAME (event), event);

	switch (GST_EVENT_TYPE (event)) {
		case GST_EVENT_CAPS:
			{
			GstCaps * caps;

			gst_event_parse_caps (event, &caps);
			GstAudioInfo info;
			ret = gst_audio_info_from_caps(&info, caps);
			if (ret) {
				filter->rate = GST_AUDIO_INFO_RATE(&info);
				filter->width = GST_AUDIO_FORMAT_INFO_WIDTH(info.finfo);
			}
			/* and forward */
			ret = gst_pad_event_default (pad, parent, event);
			break;
			}
		default:
			ret = gst_pad_event_default (pad, parent, event);
			break;
	}
	return ret;
}

static GstFlowReturn process_buffer(GstLALDeglitchFilter *filter, GstBuffer * inbuf)
{
	GstBuffer *buf = gst_buffer_make_writable(inbuf);
	gfloat *data32;
	gdouble *data64;
	GstClockTime start = GST_BUFFER_PTS(buf);
	GstClockTime stop = GST_BUFFER_PTS(buf) + GST_BUFFER_DURATION(buf);

	GstMapInfo info;
	gst_buffer_map(buf, &info, GST_MAP_WRITE);
	guint buffer_length = GST_BUFFER_OFFSET_END(buf) - GST_BUFFER_OFFSET(buf);

	gint i, j, k;

	data32 = (gfloat *) info.data;
	data64 = (gdouble *) info.data;

	/* 
	 * This is ridiculous, but doesn't require sorted segments. NOTE: THEY
	 * MUST BE COALESCED THOUGH  Could some fancy data structure help?
	 */
	for (i = 0; i < filter->seglist->length; i++) {
		/* clip segment to buffer */
		GstClockTime segstart = filter->seglist->segments[i].start;
		GstClockTime segstop  = filter->seglist->segments[i].stop;

		/* 
		 * Check that the buffer and segment look like one of these
		 * |-----------|        Buffer
		 *      |----------|    Segment
		 *
		 * |-----------|        Segment
		 *      |----------|    Buffer
		 *
		 * |-----------------|  Segment
		 *      |----------|    Buffer
		 *
		 * |-----------------|  Buffer
		 *      |----------|    Segment
		 */

		if ( (segstart > start && segstart < stop)
			|| (segstop > start && segstop < stop)
			|| (segstart < start && segstop > stop)
			|| (start < segstart && stop > segstop) 
		) {

			/* convert to samples */
			/* NOTE: yes these can be negative that is intentional */
			gint64 startix = (gint64) round(((gint64) segstart - (gint64) start) * filter->rate / GST_SECOND);
			gint64 stopix = (gint64) round(((gint64) segstop - (gint64) start) * filter->rate / GST_SECOND);
			
			/* 
			 * Figure out the window parameters hardcoded 0.25s on
			 * each side
			 */

			guint64 duration = segstop - segstart;
			g_assert(duration >= 1.0 * GST_SECOND); /* must be greater than or equal to 1.0s */
			g_assert(filter->rate >= 128); /* This probably doesnt make sense for low sample rates, so this is just here as a safety check. FIXME */
			gdouble beta = 1.0 * GST_SECOND / duration;
			guint length = stopix - startix;

			REAL8Window *window = XLALCreateTukeyREAL8Window(length, beta);

			/* set samples */
			k = -1;
			for(j = startix; j < stopix; j++) {
				k++;
				if (j < 0)
					continue;
				/* 
				 * NOTE this cast is safe and intended.  j can
				 * be negative, but if it becomes positive and
				 * greater than the buffer length
				 * we would get a seg fault 
				 */
				if (j >= (gint64) buffer_length)
					break;
				if (filter->width == 64)
					data64[j] *= 1.0 - window->data->data[k];
				if (filter->width == 32)
					data32[j] *= 1.0 - window->data->data[k];
			}

			XLALDestroyREAL8Window(window);
		}
	}
	gst_buffer_unmap(buf, &info);
	return gst_pad_push (filter->srcpad, buf);
}

/* chain function
 * this function does the actual processing
 */
static GstFlowReturn
gst_laldeglitch_filter_chain (GstPad * pad, GstObject * parent, GstBuffer * buf)
{
	GstLALDeglitchFilter *filter;
	filter = GST_LALDEGLITCHFILTER (parent);

	return process_buffer(filter, buf);
}


/* entry point to initialize the plug-in
 * initialize the plug-in itself
 * register the element factories and other features
 */
static gboolean
laldeglitchfilter_init (GstPlugin * laldeglitchfilter)
{
	/* debug category for fltering log messages
	*
	* exchange the string 'Template laldeglitchfilter' with your description
	*/
	GST_DEBUG_CATEGORY_INIT (gst_laldeglitch_filter_debug, "laldeglitchfilter", 0, "Template laldeglitchfilter");

	return gst_element_register (laldeglitchfilter, "laldeglitchfilter", GST_RANK_NONE, GST_TYPE_LALDEGLITCHFILTER);
}


/*
 * Copyright (C) 2016  DASWG <daswg@ligo.org>
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation; either version 2 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */


#include <gst/gst.h>
#include <glib/gprintf.h>

/* 
 * include files for the time, see
 * http://stackoverflow.com/questions/3756323/getting-the-current-time-in-milliseconds
 */
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#include "splitcounter.h"

/* the capabilities of the inputs and outputs.
 *
 * describe the real formats here.
 */
static GstStaticPadTemplate sink_factory =
  {
    "sink",	       // padname
    GST_PAD_SINK,	 // direction, GST_PAD_SINK==2
    GST_PAD_ALWAYS,       // presence, GST_PAD_ALWAYS==0
    {		     // the GST_STATIC_CAPS("ANY") macro
      NULL,
      "ANY",	      // caps
      { NULL }	    // GST_PADDING_INIT
    }
  };

static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("ANY")
    );


// G_DEFINE_TYPE is a macro which when passed Splitcounter and
// split_counter defines/creates/references:
//     split_counter_init (Splitcounter *self)
//	 We have to create this
//     void split_counter_class_init(SplitcounterClass *klass)
//	 We have to create this
//     gpointer split_counter_parent_class (initialized to NULL)
//     gint split_counter_private_offset
//     void split_counter_class_intern_init(gpointer klass)
//     split_counter_get_instance_private(const Splitcounter *self)
//     GType split_counter_get_type(void)
//	 We have to create this
G_DEFINE_TYPE (Splitcounter, split_counter, GST_TYPE_ELEMENT);

static gboolean split_counter_sink_event (GstPad * pad, GstObject * parent, GstEvent * event);
static GstFlowReturn split_counter_chain (GstPad * pad, GstObject * parent, GstBuffer * buf);

/* GObject vmethod implementations */

enum property {
	ARG_FILENAME = 1,
	ARG_FAKE
};

static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec) {
  Splitcounter *element = SPLIT_COUNTER(object);
  GST_OBJECT_LOCK(element);
  switch(id) {
  case ARG_FILENAME:
    element->filename = g_value_dup_string(value);
    if(element->filename)
      remove(element->filename);
    break;
  default:
    G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
    break;
  }
  GST_OBJECT_UNLOCK(element);
}

static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *pspec) {
  Splitcounter *element = SPLIT_COUNTER(object);
  GST_OBJECT_LOCK(element);
  switch(id) {
  case ARG_FILENAME:
    g_value_set_string(value, element->filename);
    break;
  default:
    G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
    break;
  }
  GST_OBJECT_UNLOCK(element);
}

/* initialize the plugin's class */
static void
split_counter_class_init (SplitcounterClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;

  gst_element_class_set_details_simple(gstelement_class,
    "Splitcounter",
    "Filter",
    "Prints buffer timestamps and offsets to the screen",
    "DASWG <daswg@ligo.org>");

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&src_factory));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sink_factory));

  gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
  gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);

  g_object_class_install_property(
    gobject_class,
    ARG_FILENAME,
    g_param_spec_string(
      "filename",
      "Filename",
      "If set, a file with this name will be written with the presentation timestamp\n\t\t\t"
      "in seconds as the left column and unix time as the right column.",
      NULL,
      G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
    )
  );
}

/* initialize the new element
 * instantiate pads and add them to element
 * set pad callback functions
 * initialize instance structure
 */
static void
split_counter_init (Splitcounter * filter)
{
  filter->sinkpad = gst_pad_new_from_static_template (&sink_factory, "sink");
  gst_pad_set_event_function (filter->sinkpad,
			      GST_DEBUG_FUNCPTR(split_counter_sink_event));
  gst_pad_set_chain_function (filter->sinkpad,
			      GST_DEBUG_FUNCPTR(split_counter_chain));
  GST_PAD_SET_PROXY_CAPS (filter->sinkpad);
  gst_element_add_pad (GST_ELEMENT (filter), filter->sinkpad);

  filter->srcpad = gst_pad_new_from_static_template (&src_factory, "src");
  GST_PAD_SET_PROXY_CAPS (filter->srcpad);
  gst_element_add_pad (GST_ELEMENT (filter), filter->srcpad);
}

/* GstElement vmethod implementations */

/* this function handles sink events */
static gboolean
split_counter_sink_event (GstPad * pad, GstObject * parent, GstEvent * event)
{
  gboolean ret;

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_CAPS:
    {
      GstCaps * caps;

      gst_event_parse_caps (event, &caps);
      /* do something with the caps */

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

/* chain function
 * this function does the actual processing
 */
static GstFlowReturn
split_counter_chain (GstPad * pad, GstObject * parent, GstBuffer * buf)
{
  Splitcounter *filter;
  struct timespec spec;
  clock_gettime(CLOCK_REALTIME, &spec);
  double seconds_since_epoch = (intmax_t) spec.tv_sec + ((double) spec.tv_nsec)/1e9;

  filter = SPLIT_COUNTER (parent);

  g_print ("[%f] (%s): just passed a buffer from (%s)->(%s) pts=%lu dts=%lu duration=%lu, offset=%lu offset_end=%lu\n", seconds_since_epoch, ((GstObject*) parent)->name, ((GstObject*) (((GstObject*) (pad->peer))->parent))->name, ((GstObject*) ((GstObject*) ((filter->srcpad)->peer))->parent)->name, buf->pts, buf->dts, buf->duration, GST_BUFFER_OFFSET(buf), GST_BUFFER_OFFSET_END(buf));

  if(filter->filename) {
    FILE *fp;
    fp = fopen(filter->filename, "a");
    g_fprintf(fp, "%f\t%f\n", (double) buf->pts / 1e9, seconds_since_epoch);
    fclose(fp);
}

  /* just push out the incoming buffer without touching it */
  return gst_pad_push (filter->srcpad, buf);
}

/* entry point to initialize the plug-in
 * initialize the plug-in itself
 * register the element factories and other features
 */
static gboolean
plugin_init (GstPlugin * plugin)
{
  /* debug category for fltering log messages
   *
   * exchange the string 'Template plugin' with your description
   */
  return gst_element_register (plugin, "splitcounter", GST_RANK_NONE,
      TYPE_SPLIT_COUNTER);
}

/* PACKAGE: this is usually set by autotools depending on some _INIT macro
 * in configure.ac and then written into and defined in config.h, but we can
 * just set it ourselves here in case someone doesn't use autotools to
 * compile this code. GST_PLUGIN_DEFINE needs PACKAGE to be defined.
 */
// #ifndef PACKAGE
// #define PACKAGE "splitcounter"
// #endif

/* gstreamer looks for this structure to register plugins
 *
 * exchange the string 'Template plugin' with your plugin description
 */
// GST_PLUGIN_DEFINE (
//     GST_VERSION_MAJOR,
//     GST_VERSION_MINOR,
//     plugin,
//     "Template plugin",
//     plugin_init,
//     "1.0.0", // was: VERSION
//     "LGPL",
//     "GStreamer",
//     "http://gstreamer.net/"
// )

GstPluginDesc gst_plugin_desc = {
  1, // GST_VERSION_MAJOR,
  4, // GST_VERSION_MINOR,
  "splitcounter",
  (gchar *) "Debugging Element",
  plugin_init,
  "1.0.0", // was: VERSION
  "GPL",  // the license
  "gstlal-calibration", // was: PACKAGE
  "gstlal-calibration", // package
  "http://www.lsc-group.phys.uwm.edu/daswg", // origin
  __GST_PACKAGE_RELEASE_DATETIME,
  GST_PADDING_INIT
};


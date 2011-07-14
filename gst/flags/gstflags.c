/* 
 * GStreamer
 * Copyright (C) 2011 Leo Singer <leo.singer@ligo.org>
 * Copyright (C) 2007 Sebastian Dröge <slomo@circular-chaos.org>
 * Copyright (C) 2006 Stefan Kost <ensonic@users.sf.net>
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

/**
 * SECTION:element-flags
 *
 * Set and clear buffer flags on all buffers in a stream.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch audiotestsrc wave=silence ! flags clear=gap ! autoaudiosink
 * ]|
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>

#include "gstflags.h"

#define GST_CAT_DEFAULT gst_flags_debug
GST_DEBUG_CATEGORY_STATIC (GST_CAT_DEFAULT);

/* Filter signals and args */
enum
{
  /* FILL ME */
  LAST_SIGNAL
};

enum
{
  PROP_0,
  PROP_SET,
  PROP_CLEAR
};

#define ALLOWED_CAPS \
    GST_AUDIO_INT_PAD_TEMPLATE_CAPS ";" GST_AUDIO_FLOAT_PAD_TEMPLATE_CAPS

#define DEBUG_INIT(bla) \
  GST_DEBUG_CATEGORY_INIT (gst_flags_debug, "flags", 0, "flags element");

GST_BOILERPLATE_FULL (GstFlags, gst_flags, GstFlags,
    GST_TYPE_BASE_TRANSFORM, DEBUG_INIT);

static void gst_flags_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_flags_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static GstFlowReturn gst_flags_prepare_output_buffer (GstBaseTransform * base,
    GstBuffer * input, gint size, GstCaps * caps, GstBuffer ** buf);
static GstFlowReturn gst_flags_transform_ip (GstBaseTransform * base,
    GstBuffer * buf);

/* GObject vmethod implementations */

static void
gst_flags_base_init (gpointer klass)
{
  GstElementClass *element_class = GST_ELEMENT_CLASS (klass);

  gst_element_class_set_details_simple (element_class,
      "Set and clear buffer flags", "Filter",
      "Set and clear buffer flags on all buffers in a stream",
      "Leo Singer <leo.singer@ligo.org>");

  gst_element_class_add_pad_template (element_class,
      gst_pad_template_new (GST_BASE_TRANSFORM_SINK_NAME, GST_PAD_SINK,
          GST_PAD_ALWAYS, GST_CAPS_ANY));
  gst_element_class_add_pad_template (element_class,
      gst_pad_template_new (GST_BASE_TRANSFORM_SRC_NAME, GST_PAD_SRC,
          GST_PAD_ALWAYS, GST_CAPS_ANY));
}

static void
gst_flags_class_init (GstFlagsClass * klass)
{
  GObjectClass *gobject_class;

  gobject_class = (GObjectClass *) klass;
  gobject_class->set_property = gst_flags_set_property;
  gobject_class->get_property = gst_flags_get_property;

  g_object_class_install_property (gobject_class, PROP_SET,
      g_param_spec_flags ("set", "Set",
          "Flags to set on all buffers", GST_TYPE_BUFFER_FLAG,
          0, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_CLEAR,
      g_param_spec_flags ("clear", "Clear",
          "Flags to clear on all buffers", GST_TYPE_BUFFER_FLAG,
          0, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  GST_BASE_TRANSFORM_CLASS (klass)->prepare_output_buffer =
      GST_DEBUG_FUNCPTR (gst_flags_prepare_output_buffer);
  GST_BASE_TRANSFORM_CLASS (klass)->transform_ip =
      GST_DEBUG_FUNCPTR (gst_flags_transform_ip);
}

static void
gst_flags_init (GstFlags * filter, GstFlagsClass * klass)
{
  gst_base_transform_set_gap_aware (GST_BASE_TRANSFORM (filter), TRUE);
}

static void
gst_flags_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstFlags *filter = GST_FLAGS (object);

  switch (prop_id) {
    case PROP_SET:
      filter->set = g_value_get_flags (value);
      break;
    case PROP_CLEAR:
      filter->clear = g_value_get_flags (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_flags_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstFlags *filter = GST_FLAGS (object);

  switch (prop_id) {
    case PROP_SET:
      g_value_set_flags (value, filter->set);
      break;
    case PROP_CLEAR:
      g_value_set_flags (value, filter->clear);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static GstFlowReturn
gst_flags_prepare_output_buffer (GstBaseTransform * base,
    GstBuffer * input, gint size, GstCaps * caps, GstBuffer ** buffer)
{
  *buffer = gst_buffer_create_sub (input, 0, GST_BUFFER_SIZE (input));
  *buffer = gst_buffer_make_metadata_writable (*buffer);

  return GST_FLOW_OK;
}

static GstFlowReturn
gst_flags_transform_ip (GstBaseTransform * base, GstBuffer * buf)
{
  GstFlags *filter = GST_FLAGS (base);

  GST_BUFFER_FLAG_SET (buf, filter->set);
  GST_BUFFER_FLAG_UNSET (buf, filter->clear);

  return GST_FLOW_OK;
}

/* entry point to initialize the plug-in
 * initialize the plug-in itself
 * register the element factories and pad templates
 * register the features
 */
static gboolean
plugin_init (GstPlugin * plugin)
{
  return gst_element_register (plugin, "flags", GST_RANK_NONE, GST_TYPE_FLAGS);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    "flags",
    "Buffer flag manipulation plugin",
    plugin_init, PACKAGE_VERSION, "LGPL", PACKAGE_NAME,
    "http://www.lsc-group.phys.uwm.edu/daswg")

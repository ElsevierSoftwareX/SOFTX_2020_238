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
 * SECTION:element-audiodelay
 *
 * Delays an audio stream by a given number of samples by emitting a gap buffer.
 * Use to shift timestamps of a stream.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch audiotestsrc ! audiodelay samples=441000 ! audio/x-raw-float,rate=44100 ! autoaudiosink
 * ]|
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <gst/audio/audio.h>
#include <gst/audio/gstaudiofilter.h>
#include <string.h>

#include "audiodelay.h"

#define GST_CAT_DEFAULT gst_audio_delay_debug
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
  PROP_SAMPLES
};

#define ALLOWED_CAPS \
    GST_AUDIO_INT_PAD_TEMPLATE_CAPS ";" GST_AUDIO_FLOAT_PAD_TEMPLATE_CAPS

#define DEBUG_INIT(bla) \
  GST_DEBUG_CATEGORY_INIT (gst_audio_delay_debug, "audiodelay", 0, "audiodelay element");

GST_BOILERPLATE_FULL (GstAudioDelay, gst_audio_delay, GstAudioFilter,
    GST_TYPE_AUDIO_FILTER, DEBUG_INIT);

static void gst_audio_delay_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_audio_delay_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static gboolean gst_audio_delay_start (GstBaseTransform * base);
static GstFlowReturn gst_audio_delay_prepare_output_buffer (GstBaseTransform *
    base, GstBuffer * input, gint size, GstCaps * caps, GstBuffer ** buf);
static GstFlowReturn gst_audio_delay_transform_ip (GstBaseTransform * base,
    GstBuffer * buf);

/* GObject vmethod implementations */

static void
gst_audio_delay_base_init (gpointer klass)
{
  GstElementClass *element_class = GST_ELEMENT_CLASS (klass);
  GstCaps *caps;

  gst_element_class_set_details_simple (element_class, "Audio delay line",
      "Filter/Audio",
      "Delays an audio stream by a given number of samples",
      "Leo Singer <leo.singer@ligo.org>");

  caps = gst_caps_from_string (ALLOWED_CAPS);
  gst_audio_filter_class_add_pad_templates (GST_AUDIO_FILTER_CLASS (klass),
      caps);
  gst_caps_unref (caps);
}

static void
gst_audio_delay_class_init (GstAudioDelayClass * klass)
{
  GObjectClass *gobject_class;

  gobject_class = (GObjectClass *) klass;
  gobject_class->set_property = gst_audio_delay_set_property;
  gobject_class->get_property = gst_audio_delay_get_property;

  g_object_class_install_property (gobject_class, PROP_SAMPLES,
      g_param_spec_uint64 ("samples", "Samples",
          "Delay stream by this number of samples", 0, G_MAXUINT64,
          0,
          G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY |
          G_PARAM_STATIC_STRINGS));

  GST_BASE_TRANSFORM_CLASS (klass)->start =
      GST_DEBUG_FUNCPTR (gst_audio_delay_start);
  GST_BASE_TRANSFORM_CLASS (klass)->prepare_output_buffer =
      GST_DEBUG_FUNCPTR (gst_audio_delay_prepare_output_buffer);
  GST_BASE_TRANSFORM_CLASS (klass)->transform_ip =
      GST_DEBUG_FUNCPTR (gst_audio_delay_transform_ip);
}

static void
gst_audio_delay_init (GstAudioDelay * filter, GstAudioDelayClass * klass)
{
  filter->samples = 0;
  gst_base_transform_set_gap_aware (GST_BASE_TRANSFORM (filter), TRUE);
}

static void
gst_audio_delay_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstAudioDelay *filter = GST_AUDIO_DELAY (object);

  switch (prop_id) {
    case PROP_SAMPLES:
      filter->samples = g_value_get_uint64 (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_audio_delay_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstAudioDelay *filter = GST_AUDIO_DELAY (object);

  switch (prop_id) {
    case PROP_SAMPLES:
      g_value_set_uint64 (value, filter->samples);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/* GstBaseTransform vmethod implementations */
static gboolean
gst_audio_delay_start (GstBaseTransform * base)
{
  GstAudioDelay *filter = GST_AUDIO_DELAY (base);

  filter->t0 = GST_CLOCK_TIME_NONE;
  filter->offset0 = GST_BUFFER_OFFSET_NONE;
  filter->offset_samples = 0;

  return TRUE;
}

static GstFlowReturn
gst_audio_delay_prepare_output_buffer (GstBaseTransform * base,
    GstBuffer * input, gint size, GstCaps * caps, GstBuffer ** buffer)
{
  GstAudioDelay *filter = GST_AUDIO_DELAY (base);

  if (filter->samples != 0) {
    *buffer = gst_buffer_create_sub (input, 0, GST_BUFFER_SIZE (input));
    *buffer = gst_buffer_make_metadata_writable (*buffer);
  } else
    *buffer = gst_buffer_ref (input);

  return GST_FLOW_OK;
}

static GstFlowReturn
gst_audio_delay_transform_ip (GstBaseTransform * base, GstBuffer * buf)
{
  GstAudioDelay *filter = GST_AUDIO_DELAY (base);
  GstFlowReturn ret = GST_FLOW_OK;

  if (filter->samples != 0) {
    GstAudioFilter *audiofilter = GST_AUDIO_FILTER (base);
    gint rate = audiofilter->format.rate;
    gint frame_size = audiofilter->format.bytes_per_sample;
    guint64 len;

    if (GST_BUFFER_OFFSET (buf) == GST_BUFFER_OFFSET_NONE
        || GST_BUFFER_OFFSET_END (buf) == GST_BUFFER_OFFSET_NONE)
      len = gst_audio_frame_length (GST_BASE_TRANSFORM_SINK_PAD (base), buf);
    else
      len = GST_BUFFER_OFFSET_END (buf) - GST_BUFFER_OFFSET (buf);

    if (filter->offset_samples == 0) {
      GstBuffer *firstbuf;

      filter->t0 = GST_BUFFER_TIMESTAMP (buf);
      filter->offset0 = GST_BUFFER_OFFSET (buf);
      filter->offset_samples = filter->samples;

      firstbuf = gst_buffer_new_and_alloc (frame_size * filter->samples);
      gst_buffer_copy_metadata (firstbuf, buf,
          GST_BUFFER_COPY_FLAGS | GST_BUFFER_COPY_CAPS);
      memset (GST_BUFFER_DATA (firstbuf), 0, GST_BUFFER_SIZE (firstbuf));
      GST_BUFFER_FLAG_SET (firstbuf, GST_BUFFER_FLAG_DISCONT);
      GST_BUFFER_FLAG_SET (firstbuf, GST_BUFFER_FLAG_GAP);
      GST_BUFFER_OFFSET (firstbuf) = filter->offset0;
      if (filter->offset0 != GST_BUFFER_OFFSET_NONE)
        GST_BUFFER_OFFSET_END (firstbuf) =
            filter->offset0 + filter->offset_samples;
      GST_BUFFER_TIMESTAMP (firstbuf) = GST_BUFFER_TIMESTAMP (buf);
      GST_BUFFER_DURATION (firstbuf) =
          gst_util_uint64_scale_int_round (filter->offset_samples, GST_SECOND,
          rate);
      ret = gst_pad_push (GST_BASE_TRANSFORM_SRC_PAD (base), firstbuf);
      if (ret != GST_FLOW_OK)
        GST_ERROR_OBJECT (filter, "Failed to push initial buffer");

      GST_BUFFER_FLAG_UNSET (buf, GST_BUFFER_FLAG_DISCONT);
    }

    if (filter->offset0 != GST_BUFFER_OFFSET_NONE)
      GST_BUFFER_OFFSET (buf) = filter->offset0 + filter->offset_samples;

    if (GST_CLOCK_TIME_IS_VALID (filter->t0))
      GST_BUFFER_TIMESTAMP (buf) =
          filter->t0 + gst_util_uint64_scale_int_round (filter->offset_samples,
          GST_SECOND, rate);

    filter->offset_samples += len;

    if (filter->offset0 != GST_BUFFER_OFFSET_NONE)
      GST_BUFFER_OFFSET_END (buf) = filter->offset0 + filter->offset_samples;

    GST_BUFFER_DURATION (buf) =
        gst_util_uint64_scale_int_round (len, GST_SECOND, rate);
  }

  return ret;
}

static gboolean
plugin_init (GstPlugin * plugin)
{
  return gst_element_register (plugin, "audiodelay", GST_RANK_NONE,
      GST_TYPE_AUDIO_DELAY);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    "audiodelay",
    "Audio delay line",
    plugin_init, VERSION, GST_LICENSE, GST_PACKAGE_NAME, GST_PACKAGE_ORIGIN)

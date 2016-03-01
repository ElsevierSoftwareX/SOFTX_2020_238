/*
 * A tab-separated values dumper to produce files for plotting
 *
 * Copyright (C) 2008--2015  Kipp Cannon, Chad Hanna
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


/**
 * SECTION:gstlal_nxydump
 * @short_description:  Converts audio time-series to tab-separated ascii text, a format compatible with most plotting utilities.
 *
 * The output is multi-column tab-separated ASCII text.  The first column
 * is the time, the remaining columns are the values of the channels in
 * order.
 *
 * Example:
 *
 * $ gst-launch audiotestsrc ! audio/x-raw, format=F32LE, rate=64 !
 * lal_nxydump ! fdsink fd=1
 *
 * Reviewed:  434cd4387c6349e68e764b78ed44e2867839c06d 2014-08-12 K.
 * Cannon, J.  Creighton, B. Sathyaprakash.
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
#include <gst/audio/audio.h>

/*
 * our own stuff
 */


#include <gstlal_nxydump.h>


#define GST_CAT_DEFAULT gst_tsvenc_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


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

#define MAX_CHARS_PER_TIMESTAMP 21

/*
 * the maximum number of characters it takes to print the value for one
 * channel including delimeter, sign characters, etc.;  double-precision
 * floats in "%.16g" format can be upto 23 characters, plus 1 tab character
 * between columns.  all other data types require fewer characters
 */

#define MAX_CHARS_PER_COLUMN (23 + 1)

/*
 * newline is "CRLF"
 */

#define TSVEOL "\r\n"
#define MAX_EXTRA_BYTES_PER_LINE strlen(TSVEOL)


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
  return MAX_CHARS_PER_TIMESTAMP + channels * MAX_CHARS_PER_COLUMN +
      MAX_EXTRA_BYTES_PER_LINE;
}


/**
 * Convert a timestamp to a sample offset relative to the timestamp of the
 * start of a buffer, clipped to the buffer boundaries.
 */


static guint64 timestamp_to_sample_clipped(GstClockTime start, guint64 length,
    gint rate, GstClockTime t)
{
  return t <= start ? 0 : MIN(gst_util_uint64_scale_int_round(t - start, rate,
          GST_SECOND), length);
}


/**
 * Print the samples from a buffer of channel data into a buffer of text.
 */


static int printsample_double(char *location, const void **sample)
{
  return sprintf(location, "\t%.16g", *(*(const double **) sample)++);
}


static int printsample_float(char *location, const void **sample)
{
  return sprintf(location, "\t%.8g", (double) *(*(const float **) sample)++);
}


static int printsample_int32(char *location, const void **sample)
{
  return sprintf(location, "\t%d", (int) *(*(const gint32 **) sample)++);
}


static int printsample_uint32(char *location, const void **sample)
{
  return sprintf(location, "\t%u", (unsigned) *(*(const guint32 **) sample)++);
}


static int printsample_int16(char *location, const void **sample)
{
  return sprintf(location, "\t%d", (int) *(*(const gint16 **) sample)++);
}


static int printsample_uint16(char *location, const void **sample)
{
  return sprintf(location, "\t%u", (unsigned) *(*(const guint16 **) sample)++);
}


static int printsample_int8(char *location, const void **sample)
{
  return sprintf(location, "\t%d", (int) *(*(const gint8 **) sample)++);
}


static int printsample_uint8(char *location, const void **sample)
{
  return sprintf(location, "\t%u", (unsigned) *(*(const guint8 **) sample)++);
}

/*
 * Given a function printsample which moves through the samples,
 * print a text version to "out". Update "out"s length accordingly.
 */
static GstFlowReturn print_samples(GstBuffer * out, GstClockTime timestamp,
    const void *samples, int (*printsample) (char *, const void **),
    int channels, int rate, guint64 length)
{
  /*
   * location shows the head where we are currently writing into
   * the output buffer
   */
  char *location;
  GstMapInfo mapinfo;
  
  guint64 offset;
  int channel;

  g_assert(printsample != NULL);

  gst_buffer_map(out, &mapinfo, GST_MAP_WRITE);
  location = (char*) mapinfo.data;

  for(offset = 0; offset < length; offset++) {
    /*
     * The current timestamp
     */

    GstClockTime t =
        timestamp + gst_util_uint64_scale_int_round(offset, GST_SECOND, rate);

    /*
     * Safety check.
     */

    g_assert_cmpuint(((guint8 *) location - (guint8 *) mapinfo.data) +
        src_bytes_per_sample(channels), <=, mapinfo.size);

    /*
     * Print the time.
     */

    location +=
        sprintf(location, "%lu.%09u", (unsigned long) (t / GST_SECOND),
        (unsigned) (t % GST_SECOND));

    /*
     * Print the channel samples.
     */

    for(channel = 0; channel < channels; channel++)
      location += printsample(location, &samples);

    /*
     * Finish with an end-of-line.
     */

    location = stpcpy(location, TSVEOL);
  }

  /*
   * Record the actual size of the buffer, but don't bother
   * realloc()ing.  Note that the final size excludes the \0
   * terminator.
   * FIXME, might do something else for performance
   */

  gst_buffer_set_size(out, (guint8 *) location - mapinfo.data);
  gst_buffer_unmap(out, &mapinfo);
  
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


static GstStaticPadTemplate sink_factory =
    GST_STATIC_PAD_TEMPLATE(GST_BASE_TRANSFORM_SINK_NAME,
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS("audio/x-raw, "
        "rate = " GST_AUDIO_RATE_RANGE ", " \
        "channels = " GST_AUDIO_CHANNELS_RANGE ", " \
	"format = (string) {" GST_AUDIO_NE(F32) ", " GST_AUDIO_NE(F64) ", " GST_AUDIO_NE(S8) ", " GST_AUDIO_NE(S16) ", "  GST_AUDIO_NE(S32) ", " GST_AUDIO_NE(U8) ", " GST_AUDIO_NE(U16) ", "  GST_AUDIO_NE(U32) "}, "
	"layout = (string) interleaved")
    );


static GstStaticPadTemplate src_factory =
GST_STATIC_PAD_TEMPLATE(GST_BASE_TRANSFORM_SRC_NAME,
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS("text/tab-separated-values")
    );


G_DEFINE_TYPE_WITH_CODE(GstTSVEnc,
    gst_tsvenc,
    GST_TYPE_BASE_TRANSFORM,
    GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "lal_nxydump", 0,
        "lal_nxydump element")
);
 

enum property
{
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


static gboolean get_unit_size(GstBaseTransform * trans, GstCaps * caps,
    gsize * size)
{
  GstStructure *str;
  gboolean success = TRUE;

  str = gst_caps_get_structure(caps, 0);
  if(gst_structure_has_name(str, "text/tab-separated-values")) {
    *size = 1;
  } else {
    GstAudioInfo info;
    success = gst_audio_info_from_caps(&info, caps);

    if(success)
      *size = GST_AUDIO_INFO_WIDTH(&info) / 8 * GST_AUDIO_INFO_CHANNELS(&info);
    else
      GST_WARNING_OBJECT(trans, "unable to parse caps %" GST_PTR_FORMAT, caps);
  }

  return success;
}


/*
 * transform_caps()
 */


static GstCaps *transform_caps(GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps *filter)
{
  /* FIXME: GstCaps *filter new in 1.0 but not used here yet */ 

  /*
   * always return the template caps of the other pad
   */

  switch (direction) {
    case GST_PAD_SRC:
      caps =
          gst_caps_copy(gst_pad_get_pad_template_caps
          (GST_BASE_TRANSFORM_SINK_PAD(trans)));
      break;

    case GST_PAD_SINK:
      caps =
          gst_caps_copy(gst_pad_get_pad_template_caps(GST_BASE_TRANSFORM_SRC_PAD
              (trans)));
      break;

    case GST_PAD_UNKNOWN:
      GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL),
          ("invalid direction GST_PAD_UNKNOWN"));
      caps = GST_CAPS_NONE;
      break;
  }

  return caps;
}


/*
 * transform_size()
 */


static gboolean transform_size(GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, gsize size, GstCaps * othercaps,
    gsize * othersize)
{
  gsize unit_size;
  gsize other_unit_size;
  gboolean success = TRUE;

  if(gst_structure_has_name(gst_caps_get_structure(caps, 0), "text/tab-separated-values")) {
    gint channels;
    if(!gst_structure_get_int(gst_caps_get_structure(othercaps, 0), "channels",
            &channels)) {
      GST_ERROR_OBJECT(trans, "unable to parse channels from %" GST_PTR_FORMAT,
          othercaps);
      return FALSE;
    }
    unit_size = src_bytes_per_sample(channels);
    if(!get_unit_size(trans, othercaps, &other_unit_size))
      return FALSE;
  } else {
    gint channels;
    if(!get_unit_size(trans, caps, &unit_size))
      return FALSE;
    if(!gst_structure_get_int(gst_caps_get_structure(caps, 0), "channels",
            &channels)) {
      GST_ERROR_OBJECT(trans, "unable to parse channels from %" GST_PTR_FORMAT,
          caps);
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


static gboolean set_caps(GstBaseTransform * trans, GstCaps * incaps,
    GstCaps * outcaps)
{
  GstTSVEnc *element = GST_TSVENC(trans);
  int (*printsample) (char *, const void **);
  gboolean success = gst_audio_info_from_caps(&(element->audio_info), incaps);

  element->printsample = NULL;  /* in case it doesn't get set */

  /*
   * Parse the format
   */

  switch (GST_AUDIO_INFO_FORMAT(&(element->audio_info))) {
    case GST_AUDIO_FORMAT_U8 :
       printsample = printsample_uint8;
       break;
    case GST_AUDIO_FORMAT_U16 :
       printsample = printsample_uint16;
       break;
    case GST_AUDIO_FORMAT_U32 :
       printsample = printsample_uint32;
       break;
    case GST_AUDIO_FORMAT_S8 :
       printsample = printsample_int8;
       break;
    case GST_AUDIO_FORMAT_S16 :
       printsample = printsample_int16;
       break;
    case GST_AUDIO_FORMAT_S32 :
       printsample = printsample_int32;
       break;
    case GST_AUDIO_FORMAT_F32 :
       printsample = printsample_float;
       break;
    case GST_AUDIO_FORMAT_F64 :
       printsample = printsample_double;
       break;
    default:
       success = FALSE;
       break;
  }

  if(success) {
    element->unit_size = GST_AUDIO_INFO_WIDTH(&(element->audio_info))
        / 8 * GST_AUDIO_INFO_CHANNELS(&(element->audio_info));
    element->printsample = printsample;
  } else
    GST_ERROR_OBJECT(element,
        "unable to parse and/or accept caps %" GST_PTR_FORMAT, incaps);

  return success;
}


/*
 * transform()
 */


static GstFlowReturn transform(GstBaseTransform * trans, GstBuffer * inbuf,
    GstBuffer * outbuf)
{
  GstTSVEnc *element = GST_TSVENC(trans);
  guint64 length;
  guint64 start, stop;
  GstFlowReturn result = GST_FLOW_OK;

  /*
   * Measure the number of samples.
   */

  if(!(GST_BUFFER_OFFSET_IS_VALID(inbuf)
          && GST_BUFFER_OFFSET_END_IS_VALID(inbuf))) {
    GST_ERROR_OBJECT(element,
        "cannot compute number of input samples:  invalid offset and/or end offset");
    result = GST_FLOW_ERROR;
    goto done;
  }
  length = GST_BUFFER_OFFSET_END(inbuf) - GST_BUFFER_OFFSET(inbuf);

  /*
   * Compute the desired start and stop samples relative to the start
   * of this buffer, clipped to the buffer edges.
   */

  if(GST_BUFFER_TIMESTAMP_IS_VALID(inbuf)) {
    start =
        timestamp_to_sample_clipped(GST_BUFFER_TIMESTAMP(inbuf), length,
        GST_AUDIO_INFO_RATE(&(element->audio_info)), element->start_time);
    stop =
        timestamp_to_sample_clipped(GST_BUFFER_TIMESTAMP(inbuf), length,
        GST_AUDIO_INFO_RATE(&(element->audio_info)), element->stop_time);
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
    /*
     * This was previously:
     * GST_BUFFER_SIZE(outbuf) = 0;
     */
    gst_buffer_set_size(outbuf, 0);
    
  } else {
      GstMapInfo mapinfo;
      gst_buffer_map(inbuf, &mapinfo, GST_MAP_READ);

      result =
          print_samples(outbuf,
		GST_BUFFER_TIMESTAMP(inbuf) + gst_util_uint64_scale_int_round(start,
		      GST_SECOND, GST_AUDIO_INFO_RATE(&(element->audio_info))),
		mapinfo.data + start * element->unit_size, /* previously was: GST_BUFFER_DATA(inbuf) + start * element->unit_size, */
		element->printsample, GST_AUDIO_INFO_CHANNELS(&(element->audio_info)), GST_AUDIO_INFO_RATE(&(element->audio_info)), stop - start);

      gst_buffer_unmap(inbuf, &mapinfo);
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


static void set_property(GObject * object, enum property id,
    const GValue * value, GParamSpec * pspec)
{
  GstTSVEnc *element = GST_TSVENC(object);

  GST_OBJECT_LOCK(element);

  switch (id) {
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


static void get_property(GObject * object, enum property id, GValue * value,
    GParamSpec * pspec)
{
  GstTSVEnc *element = GST_TSVENC(object);

  GST_OBJECT_LOCK(element);

  switch (id) {
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
 * class_init()
 */


static void gst_tsvenc_class_init(GstTSVEncClass * klass)
{
  GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(klass);
  GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
  GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

  gst_element_class_set_details_simple(element_class,
      "tab-separated values encoder",
      "Codec/Encoder/Audio",
      "Converts audio time-series to tab-separated ascii text, a format compatible with most plotting utilities.",
      "Kipp Cannon <kipp.cannon@ligo.org>, Chad Hanna <channa@ligo.caltech.edu>");

  gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
  gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);

  gst_element_class_add_pad_template(element_class,
      gst_static_pad_template_get(&src_factory));
  gst_element_class_add_pad_template(element_class,
      gst_static_pad_template_get(&sink_factory));

  g_object_class_install_property(gobject_class,
      ARG_START_TIME,
      g_param_spec_uint64("start-time",
          "Start time",
          "Start dumping data at this time in nanoseconds.",
          0, G_MAXUINT64, DEFAULT_START_TIME,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)
      );
  g_object_class_install_property(gobject_class,
      ARG_STOP_TIME,
      g_param_spec_uint64("stop-time",
          "Stop time",
          "Stop dumping data at this time in nanoseconds.",
          0, G_MAXUINT64, DEFAULT_STOP_TIME,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)
      );

  transform_class->get_unit_size = GST_DEBUG_FUNCPTR(get_unit_size);
  transform_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
  transform_class->transform = GST_DEBUG_FUNCPTR(transform);
  transform_class->transform_caps = GST_DEBUG_FUNCPTR(transform_caps);
  transform_class->transform_size = GST_DEBUG_FUNCPTR(transform_size);
}


/*
 * init()
 */


static void gst_tsvenc_init(GstTSVEnc * element)
{
  gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(element), TRUE);

  element->printsample = NULL;
}

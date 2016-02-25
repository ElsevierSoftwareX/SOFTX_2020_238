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
 * SECTION:element-multiratefirinterp
 *
 * Apply an FIR interpolation filter to a stream using a direct form polyphase
 * implementation.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <string.h>

#include "multiratefirinterp.h"

#define GST_CAT_DEFAULT gst_multirate_fir_interp_debug
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
  PROP_KERNEL,
  PROP_LAG
};

static GstStaticPadTemplate gst_multirate_fir_interp_sink_template =
    GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("audio/x-raw-float, "
        "endianness = (int) BYTE_ORDER, "
        "width = (int) 64, "
        "rate = (int) [1, MAX], "
        "channels = (int) [1, MAX]")
    );

static GstStaticPadTemplate gst_multirate_fir_interp_src_template =
    GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("audio/x-raw-float, "
        "endianness = (int) BYTE_ORDER, "
        "width = (int) 64, "
        "rate = (int) [1, MAX], "
        "channels = (int) [1, MAX]")
    );

#define DEBUG_INIT(bla) \
  GST_DEBUG_CATEGORY_INIT (gst_multirate_fir_interp_debug, "multiratefirinterp", 0, "multiratefirinterp element");

GST_BOILERPLATE_FULL (GstMultirateFirInterp, gst_multirate_fir_interp, GstBaseTransform,
    GST_TYPE_BASE_TRANSFORM, DEBUG_INIT);

static void gst_multirate_fir_interp_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_multirate_fir_interp_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static void gst_multirate_fir_interp_finalize (GObject * object);

static gboolean gst_multirate_fir_interp_set_caps (GstBaseTransform * base,
    GstCaps * incaps, GstCaps * outcaps);
static GstCaps *
gst_multirate_fir_interp_transform_caps (GstBaseTransform * base,
    GstPadDirection direction, GstCaps * caps);
static gboolean
gst_multirate_fir_interp_transform_size (GstBaseTransform * base,
    GstPadDirection direction, GstCaps * caps, guint size, GstCaps * othercaps,
    guint * othersize);
static gboolean gst_multirate_fir_interp_start (GstBaseTransform * base);
static gboolean gst_multirate_fir_interp_event (GstBaseTransform * base,
    GstEvent * event);
static GstFlowReturn gst_multirate_fir_interp_transform (GstBaseTransform * base,
    GstBuffer * inbuf, GstBuffer * outbuf);

/* GObject vmethod implementations */

static void
gst_multirate_fir_interp_base_init (gpointer klass)
{
}

static void
gst_multirate_fir_interp_class_init (GstMultirateFirInterpClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *element_class = GST_ELEMENT_CLASS (klass);

  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&gst_multirate_fir_interp_sink_template));
  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&gst_multirate_fir_interp_src_template));
  gst_element_class_set_details_simple (element_class, "Multirate FIR interpolator",
      "Filter/Audio",
      "Interpolate an audio stream using a direct form polyphase FIR interpolator",
      "Leo Singer <leo.singer@ligo.org>");

  gobject_class = (GObjectClass *) klass;
  gobject_class->set_property = gst_multirate_fir_interp_set_property;
  gobject_class->get_property = gst_multirate_fir_interp_get_property;
  gobject_class->finalize = gst_multirate_fir_interp_finalize;

  g_object_class_install_property (gobject_class, PROP_KERNEL,
      g_param_spec_value_array ("kernel", "Filter Kernel",
          "Kernel for the FIR filter",
          g_param_spec_double ("Element", "Filter Kernel Element",
              "Element of the filter kernel", -G_MAXDOUBLE, G_MAXDOUBLE, 0.0,
              G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS),
          G_PARAM_WRITABLE | GST_PARAM_MUTABLE_READY |
          G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_LAG,
      g_param_spec_uint64 ("lag", "Lag",
          "Cause the output to lag the input by this many samples", 0, G_MAXUINT64,
          0,
          G_PARAM_WRITABLE | GST_PARAM_MUTABLE_READY |
          G_PARAM_STATIC_STRINGS));

  GST_BASE_TRANSFORM_CLASS (klass)->set_caps =
      GST_DEBUG_FUNCPTR (gst_multirate_fir_interp_set_caps);
  GST_BASE_TRANSFORM_CLASS (klass)->transform_caps =
      GST_DEBUG_FUNCPTR (gst_multirate_fir_interp_transform_caps);
  GST_BASE_TRANSFORM_CLASS (klass)->transform_size =
      GST_DEBUG_FUNCPTR (gst_multirate_fir_interp_transform_size);
  GST_BASE_TRANSFORM_CLASS (klass)->start =
      GST_DEBUG_FUNCPTR (gst_multirate_fir_interp_start);
  GST_BASE_TRANSFORM_CLASS (klass)->event =
      GST_DEBUG_FUNCPTR (gst_multirate_fir_interp_event);
  GST_BASE_TRANSFORM_CLASS (klass)->transform =
      GST_DEBUG_FUNCPTR (gst_multirate_fir_interp_transform);
}

static void
gst_multirate_fir_interp_init (GstMultirateFirInterp * filter, GstMultirateFirInterpClass * klass)
{
  filter->adapter = gst_adapter_new ();
}

static void gst_multirate_fir_interp_finalize (GObject * object)
{
  GstMultirateFirInterp * self = GST_MULTIRATE_FIR_INTERP (object);

  g_free (self->kernel);
  self->kernel = NULL;
  g_free (self->reordered_kernel);
  self->reordered_kernel = NULL;
  self->kernel_ptr = NULL;
  if (self->adapter) {
    g_object_unref (self->adapter);
    self->adapter = NULL;
  }

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

static void
gst_multirate_fir_interp_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstMultirateFirInterp *filter = GST_MULTIRATE_FIR_INTERP (object);

  GST_OBJECT_LOCK (filter);

  switch (prop_id) {
    case PROP_KERNEL:
      {
        guint i;
        GValueArray *va = g_value_get_boxed (value);
        filter->kernel_length = va->n_values;
        g_free (filter->kernel);
        g_free (filter->reordered_kernel);
        filter->reordered_kernel = NULL;
        filter->kernel = g_new (double, filter->kernel_length);
        for (i = 0; i < filter->kernel_length; i ++)
          filter->kernel[i] = g_value_get_double (g_value_array_get_nth (va, i));
      }
      break;
    case PROP_LAG:
      filter->lag = g_value_get_uint64 (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }

  GST_OBJECT_UNLOCK (filter);
}

static void
gst_multirate_fir_interp_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstMultirateFirInterp *filter = GST_MULTIRATE_FIR_INTERP (object);

  GST_OBJECT_LOCK (filter);

  switch (prop_id) {
    /* FIXME: make properties readwritable and add getters*/
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }

  GST_OBJECT_UNLOCK (filter);
}

/* GstBaseTransform vmethod implementations */
static gboolean
gst_multirate_fir_interp_set_caps (GstBaseTransform * base,
    GstCaps * incaps, GstCaps * outcaps)
{
  GstMultirateFirInterp *filter = GST_MULTIRATE_FIR_INTERP (base);
  GstStructure *instruct, *outstruct;
  gint inchannels, inrate, outchannels, outrate;

  instruct = gst_caps_get_structure (incaps, 0);
  outstruct = gst_caps_get_structure (outcaps, 0);
  g_return_val_if_fail (
      gst_structure_get_int (instruct, "channels", &inchannels), FALSE);
  g_return_val_if_fail (
      gst_structure_get_int (instruct, "rate", &inrate), FALSE);
  g_return_val_if_fail (
      gst_structure_get_int (outstruct, "channels", &outchannels), FALSE);
  g_return_val_if_fail (
      gst_structure_get_int (outstruct, "rate", &outrate), FALSE);

  g_return_val_if_fail (inchannels == outchannels, FALSE);
  g_return_val_if_fail (outrate >= inrate, FALSE);
  g_return_val_if_fail (outrate % inrate == 0, FALSE);

  filter->inrate = inrate;
  filter->outrate = outrate;
  filter->channels = inchannels;
  filter->upsample_factor = outrate / inrate;
  g_free (filter->reordered_kernel);
  filter->reordered_kernel = NULL;

  return TRUE;
}

static gboolean
gst_multirate_fir_interp_start (GstBaseTransform * base)
{
  GstMultirateFirInterp *filter = GST_MULTIRATE_FIR_INTERP (base);

  filter->samples = 0;
  filter->needs_timestamp = TRUE;
  filter->kernel_ptr = filter->reordered_kernel;
  gst_adapter_clear (filter->adapter);

  return TRUE;
}

static gboolean
gst_multirate_fir_interp_transform_size (GstBaseTransform * base,
    GstPadDirection direction, GstCaps * caps, guint size, GstCaps * othercaps,
    guint * othersize)
{
  GstMultirateFirInterp *filter = GST_MULTIRATE_FIR_INTERP (base);

  if (direction == GST_PAD_SINK)
    *othersize = size * filter->upsample_factor;
  else if (direction == GST_PAD_SRC)
    *othersize = size / filter->upsample_factor;
  else
    return FALSE;

  return TRUE;
}

static GstCaps *
gst_multirate_fir_interp_transform_caps (GstBaseTransform * base,
    GstPadDirection direction, GstCaps * caps)
{
  GstCaps *othercaps = gst_caps_copy (caps);
  GValue v = {0};
  g_value_init (&v, GST_TYPE_INT_RANGE);
  gst_value_set_int_range (&v, 1, G_MAXINT);
  gst_caps_set_value (othercaps, "rate", &v);
  return othercaps;
}

static GstFlowReturn
gst_multirate_fir_interp_push_residue (GstMultirateFirInterp * filter)
{
  GstBaseTransform *base = GST_BASE_TRANSFORM (filter);
  GstBuffer *inbuf, *outbuf;
  double *indata, *outdata, *outend, *kernelend;
  gint phase;
  guint i, j;
  guint availsize;
  gint channel;
  guint minsize;

  /* Reorder kernel so that polyphase components are grouped together. */
  if (G_UNLIKELY (!(filter->reordered_kernel))) {
    if (G_UNLIKELY (!(filter->kernel))) {
      GST_ERROR_OBJECT (filter, "kernel not set");
      return GST_FLOW_ERROR;
    }

    filter->reordered_kernel_length = gst_util_uint64_scale_int_ceil (
        filter->kernel_length, 1, filter->upsample_factor) *
        filter->upsample_factor;

    /* FIXME: don't force the kernel length to be a multiple of the upsampling factor. */
    filter->reordered_kernel = g_new (double, filter->reordered_kernel_length);
    for (phase = filter->upsample_factor - 1, i = 0; phase >= 0; phase --) {
      for (j = phase; j < filter->reordered_kernel_length; j += filter->upsample_factor, i++) {
        if (G_LIKELY (j < filter->kernel_length))
          filter->reordered_kernel[i] = filter->kernel[filter->kernel_length - j - 1];
      }
    }
    filter->kernel_ptr = filter->reordered_kernel;
  }
  minsize = filter->reordered_kernel_length * filter->channels * sizeof(double) / filter->upsample_factor;

  /* Put inbuf into adapter. We have to ref inbuf because gst_adapter_push takes
   ownership of it, but transform() is not responsible for unreffing it. */
  inbuf = gst_buffer_new_and_alloc (filter->lag * filter->channels * sizeof(double));
  memset (GST_BUFFER_DATA (inbuf), 0, GST_BUFFER_SIZE (inbuf));
  gst_buffer_set_caps (inbuf, GST_PAD_CAPS (GST_BASE_TRANSFORM_SINK_PAD (base)));
  GST_BUFFER_FLAG_SET (inbuf, GST_BUFFER_FLAG_GAP);
  gst_adapter_push (filter->adapter, inbuf);

  /* Extract input data from the buffer. */
  availsize = gst_adapter_available (filter->adapter);
  if (G_UNLIKELY (availsize < minsize)) {
    GST_WARNING_OBJECT (filter, "not enough data in adapter to produce output");
    return GST_BASE_TRANSFORM_FLOW_DROPPED;
  }
  outbuf = gst_buffer_new_and_alloc ((availsize - minsize) * filter->upsample_factor);
  gst_buffer_set_caps (outbuf, GST_PAD_CAPS (GST_BASE_TRANSFORM_SRC_PAD (base)));
  indata = (double *) gst_adapter_peek (filter->adapter, availsize);
  kernelend = filter->reordered_kernel + filter->reordered_kernel_length;
  outdata = (double *) GST_BUFFER_DATA (outbuf);
  outend = (double *) ((guint8 *) outdata + GST_BUFFER_SIZE (outbuf));
  memset (outdata, 0, GST_BUFFER_SIZE (outbuf));

  /* Evaluate filter. */
  for (; outdata < outend; outdata += filter->channels) {
    for (i = 0; i < filter->reordered_kernel_length / filter->upsample_factor; i ++) {
      for (channel = 0; channel < filter->channels; channel++) {
        *(outdata + channel) += *(indata + i * filter->channels + channel) * (*filter->kernel_ptr);
      }
      if (G_UNLIKELY (++filter->kernel_ptr >= kernelend)) {
        filter->kernel_ptr = filter->reordered_kernel;
        indata += filter->channels;
      }
    }
  }

  /* Throw away the contents of the adapter that we have consumed. */
  gst_adapter_flush (filter->adapter, availsize - minsize);

  /* Set buffer metadata. */
  if (G_LIKELY (filter->offset0 != GST_BUFFER_OFFSET_NONE))
    GST_BUFFER_OFFSET (outbuf) = filter->offset0 + filter->samples * filter->upsample_factor;
  if (G_LIKELY (GST_CLOCK_TIME_IS_VALID (filter->t0)))
    GST_BUFFER_TIMESTAMP (outbuf) = filter->t0 + gst_util_uint64_scale_int_round(filter->samples * filter->upsample_factor, GST_SECOND, filter->outrate);
  filter->samples += (availsize - minsize) / (sizeof(double) * filter->channels);
  if (G_LIKELY (filter->offset0 != GST_BUFFER_OFFSET_NONE))
    GST_BUFFER_OFFSET_END (outbuf) = filter->offset0 + filter->samples * filter->upsample_factor;
  GST_BUFFER_DURATION (outbuf) = gst_util_uint64_scale_int_round((availsize - minsize) / (sizeof(double) * filter->channels) * filter->upsample_factor, GST_SECOND, filter->outrate);

  /* Done. */
  return gst_pad_push (GST_BASE_TRANSFORM_SRC_PAD (base), outbuf);
}

static gboolean
gst_multirate_fir_interp_event (GstBaseTransform * base, GstEvent * event)
{
  GstMultirateFirInterp *filter = GST_MULTIRATE_FIR_INTERP (base);

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_EOS:
      GST_INFO_OBJECT (filter, "pushing residue");
      if (G_UNLIKELY (gst_multirate_fir_interp_push_residue (filter) != GST_FLOW_OK))
        GST_ERROR_OBJECT (filter, "failed to push residue");
      filter->samples = 0;
      filter->needs_timestamp = TRUE;
      filter->kernel_ptr = filter->reordered_kernel;
      gst_adapter_clear (filter->adapter);
      break;
    case GST_EVENT_NEWSEGMENT:
      filter->samples = 0;
      filter->needs_timestamp = TRUE;
      filter->kernel_ptr = filter->reordered_kernel;
      gst_adapter_clear (filter->adapter);
      break;
    default:
      break;
  }

  return GST_BASE_TRANSFORM_CLASS (parent_class)->event (base, event);
}

static GstFlowReturn
gst_multirate_fir_interp_transform (GstBaseTransform * base, GstBuffer * inbuf,
    GstBuffer * outbuf)
{
  GstMultirateFirInterp *filter = GST_MULTIRATE_FIR_INTERP (base);
  double *indata, *outdata, *outend, *kernelend;
  gint phase;
  guint i, j;
  guint availsize;
  gint channel;
  guint minsize;

  /* Reorder kernel so that polyphase components are grouped together. */
  if (G_UNLIKELY (!(filter->reordered_kernel))) {
    if (G_UNLIKELY (!(filter->kernel))) {
      GST_ERROR_OBJECT (filter, "kernel not set");
      return GST_FLOW_ERROR;
    }

    filter->reordered_kernel_length = gst_util_uint64_scale_int_ceil (
        filter->kernel_length, 1, filter->upsample_factor) *
        filter->upsample_factor;

    /* FIXME: don't force the kernel length to be a multiple of the upsampling factor. */
    filter->reordered_kernel = g_new (double, filter->reordered_kernel_length);
    for (phase = filter->upsample_factor - 1, i = 0; phase >= 0; phase --) {
      for (j = phase; j < filter->reordered_kernel_length; j += filter->upsample_factor, i++) {
        if (G_LIKELY (j < filter->kernel_length))
          filter->reordered_kernel[i] = filter->kernel[filter->kernel_length - j - 1];
      }
    }
    filter->kernel_ptr = filter->reordered_kernel;
  }
  minsize = filter->reordered_kernel_length * filter->channels * sizeof(double) / filter->upsample_factor;

  /* Store initial timestamp and offset if this is the first buffer. */
  if (G_UNLIKELY (filter->needs_timestamp)) {
    filter->t0 = GST_BUFFER_TIMESTAMP (inbuf);
    filter->offset0 = GST_BUFFER_OFFSET (inbuf);
    filter->needs_timestamp = FALSE;

    GstBuffer *gapbuf = gst_buffer_new_and_alloc (minsize - filter->lag * filter->channels * sizeof(double));
    memset (GST_BUFFER_DATA (gapbuf), 0, GST_BUFFER_SIZE (gapbuf));
    gst_buffer_copy_metadata (gapbuf, inbuf, GST_BUFFER_COPY_CAPS);
    GST_BUFFER_FLAG_SET (gapbuf, GST_BUFFER_FLAG_GAP);
    gst_adapter_push (filter->adapter, gapbuf);
  }

  /* Put inbuf into adapter. We have to ref inbuf because gst_adapter_push takes
   ownership of it, but transform() is not responsible for unreffing it. */
  gst_buffer_ref (inbuf);
  gst_adapter_push (filter->adapter, inbuf);

  /* Extract input data from the buffer. */
  availsize = gst_adapter_available (filter->adapter);
  if (G_UNLIKELY (availsize < minsize)) {
    GST_WARNING_OBJECT (filter, "not enough data in adapter to produce output");
    return GST_BASE_TRANSFORM_FLOW_DROPPED;
  }
  GST_BUFFER_SIZE (outbuf) = (availsize - minsize) * filter->upsample_factor;
  indata = (double *) gst_adapter_peek (filter->adapter, availsize);
  kernelend = filter->reordered_kernel + filter->reordered_kernel_length;
  outdata = (double *) GST_BUFFER_DATA (outbuf);
  outend = (double *) ((guint8 *) outdata + GST_BUFFER_SIZE (outbuf));
  memset (outdata, 0, GST_BUFFER_SIZE (outbuf));

  /* Evaluate filter. */
  for (; outdata < outend; outdata += filter->channels) {
    for (i = 0; i < filter->reordered_kernel_length / filter->upsample_factor; i ++) {
      for (channel = 0; channel < filter->channels; channel++) {
        *(outdata + channel) += *(indata + i * filter->channels + channel) * (*filter->kernel_ptr);
      }
      if (G_UNLIKELY (++filter->kernel_ptr >= kernelend)) {
        filter->kernel_ptr = filter->reordered_kernel;
        indata += filter->channels;
      }
    }
  }

  /* Throw away the contents of the adapter that we have consumed. */
  gst_adapter_flush (filter->adapter, availsize - minsize);

  /* Set buffer metadata. */
  if (G_LIKELY (filter->offset0 != GST_BUFFER_OFFSET_NONE))
    GST_BUFFER_OFFSET (outbuf) = filter->offset0 + filter->samples * filter->upsample_factor;
  if (G_LIKELY (GST_CLOCK_TIME_IS_VALID (filter->t0)))
    GST_BUFFER_TIMESTAMP (outbuf) = filter->t0 + gst_util_uint64_scale_int_round(filter->samples * filter->upsample_factor, GST_SECOND, filter->outrate);
  filter->samples += (availsize - minsize) / (sizeof(double) * filter->channels);
  if (G_LIKELY (filter->offset0 != GST_BUFFER_OFFSET_NONE))
    GST_BUFFER_OFFSET_END (outbuf) = filter->offset0 + filter->samples * filter->upsample_factor;
  GST_BUFFER_DURATION (outbuf) = gst_util_uint64_scale_int_round((availsize - minsize) / (sizeof(double) * filter->channels) * filter->upsample_factor, GST_SECOND, filter->outrate);

  /* Done. */
  return GST_FLOW_OK;
}

/* GStreamer
 * Copyright (C) 1999,2000 Erik Walthinsen <omega@cse.ogi.edu>
 *                    2001 Thomas <thomas@apestaart.org>
 *               2005,2006 Wim Taymans <wim@fluendo.com>
 *               2011 Kipp Cannon <kipp.cannon@ligo.org>
 *
 * adder.c: Adder element, N in, one out, samples are added
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
 * SECTION:element-adder
 *
 * The adder allows to mix several streams into one by adding the data.
 * Mixed data is clamped to the min/max values of the data format.
 *
 * If the element's sync property is TRUE the streams are mixed with the
 * timestamps synchronized.  If the sync property is FALSE (the default, to
 * be compatible with older versions), then the first samples from each
 * stream are added to produce the first sample of the output, the second
 * samples are added to produce the second sample of the output, and so on.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch audiotestsrc freq=100 ! adder name=mix ! audioconvert ! alsasink audiotestsrc freq=500 ! mix.
 * ]| This pipeline produces two sine waves mixed together.
 * </refsect2>
 *
 * Last reviewed on 2006-05-09 (0.10.7)
 */
/* Element-Checklist-Version: 5 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <complex.h>
#include "gstadder.h"
#include <gst/audio/audio.h>
#include <string.h>             /* strcmp */
#include "gstadderorc.h"
#include <gstlalcollectpads.h>
#include <gstlal_debug.h>

/* highest positive/lowest negative x-bit value we can use for clamping */
#define MAX_INT_32  ((gint32) (0x7fffffff))
#define MAX_INT_16  ((gint16) (0x7fff))
#define MAX_INT_8   ((gint8)  (0x7f))
#define MAX_UINT_32 ((guint32)(0xffffffff))
#define MAX_UINT_16 ((guint16)(0xffff))
#define MAX_UINT_8  ((guint8) (0xff))

#define MIN_INT_32  ((gint32) (0x80000000))
#define MIN_INT_16  ((gint16) (0x8000))
#define MIN_INT_8   ((gint8)  (0x80))
#define MIN_UINT_32 ((guint32)(0x00000000))
#define MIN_UINT_16 ((guint16)(0x0000))
#define MIN_UINT_8  ((guint8) (0x00))

enum
{
  PROP_0,
  PROP_FILTER_CAPS,
  PROP_SYNCHRONOUS
};

#define GST_CAT_DEFAULT gst_adder_debug
GST_DEBUG_CATEGORY_STATIC (GST_CAT_DEFAULT);

/* elementfactory information */

#define CAPS \
  "audio/x-raw-int, " \
  "rate = (int) [ 1, MAX ], " \
  "channels = (int) [ 1, MAX ], " \
  "endianness = (int) BYTE_ORDER, " \
  "width = (int) 32, " \
  "depth = (int) 32, " \
  "signed = (boolean) { true, false } ;" \
  "audio/x-raw-int, " \
  "rate = (int) [ 1, MAX ], " \
  "channels = (int) [ 1, MAX ], " \
  "endianness = (int) BYTE_ORDER, " \
  "width = (int) 16, " \
  "depth = (int) 16, " \
  "signed = (boolean) { true, false } ;" \
  "audio/x-raw-int, " \
  "rate = (int) [ 1, MAX ], " \
  "channels = (int) [ 1, MAX ], " \
  "endianness = (int) BYTE_ORDER, " \
  "width = (int) 8, " \
  "depth = (int) 8, " \
  "signed = (boolean) { true, false } ;" \
  "audio/x-raw-float, " \
  "rate = (int) [ 1, MAX ], " \
  "channels = (int) [ 1, MAX ], " \
  "endianness = (int) BYTE_ORDER, " \
  "width = (int) { 32, 64 } ;" \
  "audio/x-raw-complex, " \
  "rate = (int) [ 1, MAX ], " \
  "channels = (int) [ 1, MAX ], " \
  "endianness = (int) BYTE_ORDER, " \
  "width = (int) { 64, 128 }"

static GstStaticPadTemplate gst_adder_src_template =
GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (CAPS)
    );

static GstStaticPadTemplate gst_adder_sink_template =
GST_STATIC_PAD_TEMPLATE ("sink%d",
    GST_PAD_SINK,
    GST_PAD_REQUEST,
    GST_STATIC_CAPS (CAPS)
    );

GST_BOILERPLATE (GstLALAdder, gstlal_adder, GstElement, GST_TYPE_ELEMENT);

static void gst_adder_dispose (GObject * object);
static void gst_adder_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_adder_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static gboolean gst_adder_setcaps (GstPad * pad, GstCaps * caps);
static gboolean gst_adder_query (GstPad * pad, GstQuery * query);
static gboolean gst_adder_src_event (GstPad * pad, GstEvent * event);
static gboolean gst_adder_sink_event (GstPad * pad, GstEvent * event);

static GstPad *gst_adder_request_new_pad (GstElement * element,
    GstPadTemplate * temp, const gchar * unused);
static void gst_adder_release_pad (GstElement * element, GstPad * pad);

static GstStateChangeReturn gst_adder_change_state (GstElement * element,
    GstStateChange transition);

static GstBuffer *gst_adder_do_clip (GstCollectPads * pads,
    GstCollectData * data, GstBuffer * buffer, gpointer user_data);
static GstFlowReturn gst_adder_collected (GstCollectPads * pads,
    gpointer user_data);

/* non-clipping versions (for float) */
#define MAKE_FUNC_NC(name,type)                                 \
static void name (type *out, type *in, gint samples) {          \
  gint i;                                                       \
  for (i = 0; i < samples; i++)                                 \
    out[i] += in[i];                                            \
}

/* *INDENT-OFF* */
MAKE_FUNC_NC (add_float64, gdouble)
MAKE_FUNC_NC (add_complex64, complex float)
MAKE_FUNC_NC (add_complex128, complex double)
/* *INDENT-ON* */

/* we can only accept caps that we and downstream can handle.
 * if we have filtercaps set, use those to constrain the target caps.
 */
static GstCaps *
gst_adder_sink_getcaps (GstPad * pad)
{
  GstLALAdder *adder;
  GstCaps *result, *peercaps, *sinkcaps, *filter_caps;

  adder = GST_ADDER (GST_PAD_PARENT (pad));

  GST_OBJECT_LOCK (adder);
  /* take filter */
  if ((filter_caps = adder->filter_caps))
    gst_caps_ref (filter_caps);
  GST_OBJECT_UNLOCK (adder);

  /* get the downstream possible caps */
  peercaps = gst_pad_peer_get_caps (adder->srcpad);

  /* get the allowed caps on this sinkpad, we use the fixed caps function so
   * that it does not call recursively in this function. */
  sinkcaps = gst_pad_get_fixed_caps_func (pad);
  if (peercaps) {
    /* restrict with filter-caps if any */
    if (filter_caps) {
      GST_DEBUG_OBJECT (adder, "filtering peer caps");
      result = gst_caps_intersect (peercaps, filter_caps);
      gst_caps_unref (peercaps);
      peercaps = result;
    }
    /* if the peer has caps, intersect */
    GST_DEBUG_OBJECT (adder, "intersecting peer and template caps");
    result = gst_caps_intersect (peercaps, sinkcaps);
    gst_caps_unref (peercaps);
    gst_caps_unref (sinkcaps);
  } else {
    /* the peer has no caps (or there is no peer), just use the allowed caps
     * of this sinkpad. */
    /* restrict with filter-caps if any */
    if (filter_caps) {
      GST_DEBUG_OBJECT (adder, "no peer caps, using filtered sinkcaps");
      result = gst_caps_intersect (sinkcaps, filter_caps);
      gst_caps_unref (sinkcaps);
    } else {
      GST_DEBUG_OBJECT (adder, "no peer caps, using sinkcaps");
      result = sinkcaps;
    }
  }

  if (filter_caps)
    gst_caps_unref (filter_caps);

  GST_LOG_OBJECT (adder, "getting caps on pad %p,%s to %" GST_PTR_FORMAT, pad,
      GST_PAD_NAME (pad), result);

  return result;
}

/* the first caps we receive on any of the sinkpads will define the caps for all
 * the other sinkpads because we can only mix streams with the same caps.
 */
static gboolean
gst_adder_setcaps (GstPad * pad, GstCaps * caps)
{
  GstLALAdder *adder;
  GList *pads;
  GstStructure *structure;
  const char *media_type;

  adder = GST_ADDER (GST_PAD_PARENT (pad));

  GST_LOG_OBJECT (adder, "setting caps on pad %p,%s to %" GST_PTR_FORMAT, pad,
      GST_PAD_NAME (pad), caps);

  /* FIXME, see if the other pads can accept the format. Also lock the
   * format on the other pads to this new format. */
  GST_OBJECT_LOCK (adder);
  pads = GST_ELEMENT (adder)->pads;
  while (pads) {
    GstPad *otherpad = GST_PAD (pads->data);

    if (otherpad != pad) {
      gst_caps_replace (&GST_PAD_CAPS (otherpad), caps);
    }
    pads = g_list_next (pads);
  }
  GST_OBJECT_UNLOCK (adder);

  /* parse caps now */
  structure = gst_caps_get_structure (caps, 0);
  media_type = gst_structure_get_name (structure);
  if (strcmp (media_type, "audio/x-raw-int") == 0) {
    adder->format = GST_ADDER_FORMAT_INT;
    gst_structure_get_int (structure, "width", &adder->width);
    gst_structure_get_int (structure, "depth", &adder->depth);
    gst_structure_get_int (structure, "endianness", &adder->endianness);
    gst_structure_get_boolean (structure, "signed", &adder->is_signed);

    GST_INFO_OBJECT (pad, "parse_caps sets adder to format int, %d bit",
        adder->width);

    if (adder->endianness != G_BYTE_ORDER)
      goto not_supported;

    switch (adder->width) {
      case 8:
        adder->func = (adder->is_signed ?
            (GstAdderFunction) add_int8 : (GstAdderFunction) add_uint8);
        break;
      case 16:
        adder->func = (adder->is_signed ?
            (GstAdderFunction) add_int16 : (GstAdderFunction) add_uint16);
        break;
      case 32:
        adder->func = (adder->is_signed ?
            (GstAdderFunction) add_int32 : (GstAdderFunction) add_uint32);
        break;
      default:
        goto not_supported;
    }
  } else if (strcmp (media_type, "audio/x-raw-float") == 0) {
    adder->format = GST_ADDER_FORMAT_FLOAT;
    gst_structure_get_int (structure, "width", &adder->width);
    gst_structure_get_int (structure, "endianness", &adder->endianness);

    GST_INFO_OBJECT (pad, "parse_caps sets adder to format float, %d bit",
        adder->width);

    if (adder->endianness != G_BYTE_ORDER)
      goto not_supported;

    switch (adder->width) {
      case 32:
        adder->func = (GstAdderFunction) add_float32;
        break;
      case 64:
        adder->func = (GstAdderFunction) add_float64;
        break;
      default:
        goto not_supported;
    }
  } else if (strcmp (media_type, "audio/x-raw-complex") == 0) {
    adder->format = GST_ADDER_FORMAT_COMPLEX;
    gst_structure_get_int (structure, "width", &adder->width);
    gst_structure_get_int (structure, "endianness", &adder->endianness);

    GST_INFO_OBJECT (pad, "parse_caps sets adder to format complex, %d bit",
        adder->width);

    if (adder->endianness != G_BYTE_ORDER)
      goto not_supported;

    switch (adder->width) {
      case 64:
        adder->func = (GstAdderFunction) add_complex64;
        break;
      case 128:
        adder->func = (GstAdderFunction) add_complex128;
        break;
      default:
        goto not_supported;
    }
  } else {
    goto not_supported;
  }

  gst_structure_get_int (structure, "channels", &adder->channels);
  gst_structure_get_int (structure, "rate", &adder->rate);
  /* precalc bps */
  adder->sample_size = adder->width / 8;
  adder->bps = adder->sample_size * adder->channels;

  /* set unit size on collect pads */
  GST_OBJECT_LOCK (adder->collect);
  for (pads = GST_ELEMENT (adder)->pads; pads; pads = g_list_next (pads)) {
    GstPad *pad = GST_PAD (pads->data);
    if (gst_pad_get_direction (pad) == GST_PAD_SINK) {
      gstlal_collect_pads_set_unit_size (pad, adder->bps);
      gstlal_collect_pads_set_rate (pad, adder->rate);
    }
  }
  GST_OBJECT_UNLOCK (adder->collect);

  return TRUE;

  /* ERRORS */
not_supported:
  {
    GST_DEBUG_OBJECT (adder, "unsupported format set as caps");
    return FALSE;
  }
}

/* FIXME, the duration query should reflect how long you will produce
 * data, that is the amount of stream time until you will emit EOS.
 *
 * For synchronized mixing this is always the max of all the durations
 * of upstream since we emit EOS when all of them finished.
 *
 * We don't do synchronized mixing so this really depends on where the
 * streams where punched in and what their relative offsets are against
 * eachother which we can get from the first timestamps we see.
 *
 * When we add a new stream (or remove a stream) the duration might
 * also become invalid again and we need to post a new DURATION
 * message to notify this fact to the parent.
 * For now we take the max of all the upstream elements so the simple
 * cases work at least somewhat.
 */
static gboolean
gst_adder_query_duration (GstLALAdder * adder, GstQuery * query)
{
  gint64 max;
  gboolean res;
  GstFormat format;
  GstIterator *it;
  gboolean done;

  /* parse format */
  gst_query_parse_duration (query, &format, NULL);

  max = -1;
  res = TRUE;
  done = FALSE;

  it = gst_element_iterate_sink_pads (GST_ELEMENT_CAST (adder));
  while (!done) {
    GstIteratorResult ires;

    gpointer item;

    ires = gst_iterator_next (it, &item);
    switch (ires) {
      case GST_ITERATOR_DONE:
        done = TRUE;
        break;
      case GST_ITERATOR_OK:
      {
        GstPad *pad = GST_PAD_CAST (item);

        gint64 duration;

        /* ask sink peer for duration */
        res &= gst_pad_query_peer_duration (pad, &format, &duration);
        /* take max from all valid return values */
        if (res) {
          /* valid unknown length, stop searching */
          if (duration == -1) {
            max = duration;
            done = TRUE;
          }
          /* else see if bigger than current max */
          else if (duration > max)
            max = duration;
        }
        gst_object_unref (pad);
        break;
      }
      case GST_ITERATOR_RESYNC:
        max = -1;
        res = TRUE;
        gst_iterator_resync (it);
        break;
      default:
        res = FALSE;
        done = TRUE;
        break;
    }
  }
  gst_iterator_free (it);

  if (res) {
    /* and store the max */
    GST_DEBUG_OBJECT (adder, "Total duration in format %s: %"
        GST_TIME_FORMAT, gst_format_get_name (format), GST_TIME_ARGS (max));
    gst_query_set_duration (query, format, max);
  }

  return res;
}

static gboolean
gst_adder_query_latency (GstLALAdder * adder, GstQuery * query)
{
  GstClockTime min, max;
  gboolean live;
  gboolean res;
  GstIterator *it;
  gboolean done;

  res = TRUE;
  done = FALSE;

  live = FALSE;
  min = 0;
  max = GST_CLOCK_TIME_NONE;

  /* Take maximum of all latency values */
  it = gst_element_iterate_sink_pads (GST_ELEMENT_CAST (adder));
  while (!done) {
    GstIteratorResult ires;

    gpointer item;

    ires = gst_iterator_next (it, &item);
    switch (ires) {
      case GST_ITERATOR_DONE:
        done = TRUE;
        break;
      case GST_ITERATOR_OK:
      {
        GstPad *pad = GST_PAD_CAST (item);
        GstQuery *peerquery;
        GstClockTime min_cur, max_cur;
        gboolean live_cur;

        peerquery = gst_query_new_latency ();

        /* Ask peer for latency */
        res &= gst_pad_peer_query (pad, peerquery);

        /* take max from all valid return values */
        if (res) {
          gst_query_parse_latency (peerquery, &live_cur, &min_cur, &max_cur);

          if (min_cur > min)
            min = min_cur;

          if (max_cur != GST_CLOCK_TIME_NONE &&
              ((max != GST_CLOCK_TIME_NONE && max_cur > max) ||
                  (max == GST_CLOCK_TIME_NONE)))
            max = max_cur;

          live = live || live_cur;
        }

        gst_query_unref (peerquery);
        gst_object_unref (pad);
        break;
      }
      case GST_ITERATOR_RESYNC:
        live = FALSE;
        min = 0;
        max = GST_CLOCK_TIME_NONE;
        res = TRUE;
        gst_iterator_resync (it);
        break;
      default:
        res = FALSE;
        done = TRUE;
        break;
    }
  }
  gst_iterator_free (it);

  if (res) {
    /* store the results */
    GST_DEBUG_OBJECT (adder, "Calculated total latency: live %s, min %"
        GST_TIME_FORMAT ", max %" GST_TIME_FORMAT,
        (live ? "yes" : "no"), GST_TIME_ARGS (min), GST_TIME_ARGS (max));
    gst_query_set_latency (query, live, min, max);
  }

  return res;
}

static gboolean
gst_adder_query (GstPad * pad, GstQuery * query)
{
  GstLALAdder *adder = GST_ADDER (gst_pad_get_parent (pad));
  gboolean res = FALSE;

  switch (GST_QUERY_TYPE (query)) {
    case GST_QUERY_POSITION:
    {
      GstFormat format;

      gst_query_parse_position (query, &format, NULL);

      switch (format) {
        case GST_FORMAT_TIME:
          /* FIXME, bring to stream time, might be tricky */
          gst_query_set_position (query, format, adder->timestamp);
          res = TRUE;
          break;
        case GST_FORMAT_DEFAULT:
          gst_query_set_position (query, format, adder->offset);
          res = TRUE;
          break;
        default:
          break;
      }
      break;
    }
    case GST_QUERY_DURATION:
      res = gst_adder_query_duration (adder, query);
      break;
    case GST_QUERY_LATENCY:
      res = gst_adder_query_latency (adder, query);
      break;
    default:
      /* FIXME, needs a custom query handler because we have multiple
       * sinkpads */
      res = gst_pad_query_default (pad, query);
      break;
  }

  gst_object_unref (adder);
  return res;
}

typedef struct
{
  GstEvent *event;
  gboolean flush;
} EventData;

static gboolean
forward_event_func (GstPad * pad, GValue * ret, EventData * data)
{
  GstEvent *event = data->event;

  gst_event_ref (event);
  GST_LOG_OBJECT (pad, "About to send event %s", GST_EVENT_TYPE_NAME (event));
  if (!gst_pad_push_event (pad, event)) {
    GST_WARNING_OBJECT (pad, "Sending event  %p (%s) failed.",
        event, GST_EVENT_TYPE_NAME (event));
    /* quick hack to unflush the pads, ideally we need a way to just unflush
     * this single collect pad */
    if (data->flush)
      gst_pad_send_event (pad, gst_event_new_flush_stop ());
  } else {
    g_value_set_boolean (ret, TRUE);
    GST_LOG_OBJECT (pad, "Sent event  %p (%s).",
        event, GST_EVENT_TYPE_NAME (event));
  }
  gst_object_unref (pad);

  /* continue on other pads, even if one failed */
  return TRUE;
}

/* forwards the event to all sinkpads, takes ownership of the
 * event
 *
 * Returns: TRUE if the event could be forwarded on all
 * sinkpads.
 */
static gboolean
forward_event (GstLALAdder * adder, GstEvent * event, gboolean flush)
{
  gboolean ret;
  GstIterator *it;
  GstIteratorResult ires;
  GValue vret = { 0 };
  EventData data;

  GST_LOG_OBJECT (adder, "Forwarding event %p (%s)", event,
      GST_EVENT_TYPE_NAME (event));

  data.event = event;
  data.flush = flush;

  g_value_init (&vret, G_TYPE_BOOLEAN);
  g_value_set_boolean (&vret, FALSE);
  it = gst_element_iterate_sink_pads (GST_ELEMENT_CAST (adder));
  while (TRUE) {
    ires = gst_iterator_fold (it, (GstIteratorFoldFunction) forward_event_func,
        &vret, &data);
    switch (ires) {
      case GST_ITERATOR_RESYNC:
        GST_WARNING ("resync");
        gst_iterator_resync (it);
        g_value_set_boolean (&vret, TRUE);
        break;
      case GST_ITERATOR_OK:
      case GST_ITERATOR_DONE:
        ret = g_value_get_boolean (&vret);
        goto done;
      default:
        ret = FALSE;
        goto done;
    }
  }
done:
  gst_iterator_free (it);
  GST_LOG_OBJECT (adder, "Forwarded event %p (%s), ret=%d", event,
      GST_EVENT_TYPE_NAME (event), ret);
  gst_event_unref (event);

  return ret;
}

static gboolean
gst_adder_src_event (GstPad * pad, GstEvent * event)
{
  GstLALAdder *adder;
  gboolean result;

  adder = GST_ADDER (gst_pad_get_parent (pad));

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_SEEK:
    {
      gdouble rate;
      GstSeekFlags flags;
      GstSeekType curtype, endtype;
      gint64 cur, end;
      gboolean flush;

      /* parse the seek parameters */
      gst_event_parse_seek (event, &rate, NULL, &flags, &curtype,
          &cur, &endtype, &end);

      if ((curtype != GST_SEEK_TYPE_NONE) && (curtype != GST_SEEK_TYPE_SET)) {
        result = FALSE;
        GST_DEBUG_OBJECT (adder,
            "seeking failed, unhandled seek type for start: %d", curtype);
        goto done;
      }
      if ((endtype != GST_SEEK_TYPE_NONE) && (endtype != GST_SEEK_TYPE_SET)) {
        result = FALSE;
        GST_DEBUG_OBJECT (adder,
            "seeking failed, unhandled seek type for end: %d", endtype);
        goto done;
      }

      flush = (flags & GST_SEEK_FLAG_FLUSH) == GST_SEEK_FLAG_FLUSH;

      /* check if we are flushing */
      if (flush) {
        /* make sure we accept nothing anymore and return WRONG_STATE */
        gst_collect_pads_set_flushing (adder->collect, TRUE);

        /* flushing seek, start flush downstream, the flush will be done
         * when all pads received a FLUSH_STOP. */
        gst_pad_push_event (adder->srcpad, gst_event_new_flush_start ());

        /* We can't send FLUSH_STOP here since upstream could start pushing data
         * after we unlock adder->collect.
         * We set flush_stop_pending to TRUE instead and send FLUSH_STOP after
         * forwarding the seek upstream or from gst_adder_collected,
         * whichever happens first.
         */
        adder->flush_stop_pending = TRUE;
      }
      GST_DEBUG_OBJECT (adder, "handling seek event: %" GST_PTR_FORMAT, event);

      /* now wait for the collected to be finished and mark a new
       * segment. After we have the lock, no collect function is running and no
       * new collect function will be called for as long as we're flushing. */
      GST_OBJECT_LOCK (adder->collect);
      /* make sure we push a new segment, to inform about new basetime
       * see FIXME in gst_adder_collected() */
      adder->segment_pending = TRUE;
      if (flush) {
        /* Yes, we need to call _set_flushing again *WHEN* the streaming threads
         * have stopped so that the cookie gets properly updated. */
        gst_collect_pads_set_flushing (adder->collect, TRUE);
      }
      GST_OBJECT_UNLOCK (adder->collect);

      GST_DEBUG_OBJECT (adder, "forwarding seek event: %" GST_PTR_FORMAT,
          event);
      result = forward_event (adder, event, flush);
      if (!result) {
        /* seek failed. maybe source is a live source. */
        GST_DEBUG_OBJECT (adder, "seeking failed");
      }
      if (g_atomic_int_compare_and_exchange (&adder->flush_stop_pending,
              TRUE, FALSE)) {
        GST_DEBUG_OBJECT (adder, "pending flush stop");
        gst_pad_push_event (adder->srcpad, gst_event_new_flush_stop ());
      }
      break;
    }
    case GST_EVENT_QOS:
      /* QoS might be tricky */
      result = FALSE;
      break;
    case GST_EVENT_NAVIGATION:
      /* navigation is rather pointless. */
      result = FALSE;
      break;
    default:
      /* just forward the rest for now */
      GST_DEBUG_OBJECT (adder, "forward unhandled event: %s",
          GST_EVENT_TYPE_NAME (event));
      result = forward_event (adder, event, FALSE);
      break;
  }

done:
  gst_object_unref (adder);

  return result;
}

static gboolean
gst_adder_sink_event (GstPad * pad, GstEvent * event)
{
  GstLALAdder *adder;
  gboolean ret = TRUE;

  adder = GST_ADDER (gst_pad_get_parent (pad));

  GST_DEBUG ("Got %s event on pad %s:%s", GST_EVENT_TYPE_NAME (event),
      GST_DEBUG_PAD_NAME (pad));

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_FLUSH_STOP:
      /* we received a flush-stop. The collect_event function will push the
       * event past our element. We simply forward all flush-stop events, even
       * when no flush-stop was pending, this is required because collectpads
       * does not provide an API to handle-but-not-forward the flush-stop.
       * We unset the pending flush-stop flag so that we don't send anymore
       * flush-stop from the collect function later.
       */
      GST_OBJECT_LOCK (adder->collect);
      adder->segment_pending = TRUE;
      adder->flush_stop_pending = FALSE;
      /* Clear pending tags */
      /* FIXME:  switch to
        g_list_free_full (adder->pending_events, (GDestroyNotify) gst_event_unref);
        adder->pending_events = NULL;
      */
      while (adder->pending_events) {
        GstEvent *ev = GST_EVENT (adder->pending_events->data);
        gst_event_unref (ev);
        adder->pending_events = g_list_remove (adder->pending_events, ev);
      }
      GST_OBJECT_UNLOCK (adder->collect);
      break;
    case GST_EVENT_TAG:
      GST_OBJECT_LOCK (adder->collect);
      /* collect tags here so we can push them out when we collect data */
      adder->pending_events = g_list_append (adder->pending_events, event);
      GST_OBJECT_UNLOCK (adder->collect);
      goto beach;
    default:
      break;
  }

  /* now GstCollectPads can take care of the rest, e.g. EOS */
  ret = adder->collect_event (pad, event);

beach:
  gst_object_unref (adder);
  return ret;
}

static void
gstlal_adder_base_init (gpointer g_class)
{
  GstElementClass *gstelement_class = GST_ELEMENT_CLASS (g_class);

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_adder_src_template));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_adder_sink_template));
  gst_element_class_set_details_simple (gstelement_class, "Adder",
      "Generic/Audio",
      "Add N audio channels together",
      "Thomas Vander Stichele <thomas at apestaart dot org>");
}

static void
gstlal_adder_class_init (GstLALAdderClass * klass)
{
  GObjectClass *gobject_class = (GObjectClass *) klass;
  GstElementClass *gstelement_class = (GstElementClass *) klass;

  gobject_class->set_property = gst_adder_set_property;
  gobject_class->get_property = gst_adder_get_property;
  gobject_class->dispose = gst_adder_dispose;

  /**
   * GstAdder:caps:
   *
   * Since: 0.10.24
   */
  g_object_class_install_property (gobject_class, PROP_FILTER_CAPS,
      g_param_spec_boxed ("caps", "Target caps",
          "Set target format for mixing (NULL means ANY). "
          "Setting this property takes a reference to the supplied GstCaps "
          "object.", GST_TYPE_CAPS,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_SYNCHRONOUS,
      g_param_spec_boolean ("sync", "Synchronous",
          "Align the time stamps of input streams. ",
          FALSE,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  gstelement_class->request_new_pad =
      GST_DEBUG_FUNCPTR (gst_adder_request_new_pad);
  gstelement_class->release_pad = GST_DEBUG_FUNCPTR (gst_adder_release_pad);
  gstelement_class->change_state = GST_DEBUG_FUNCPTR (gst_adder_change_state);
}

static void
gstlal_adder_init (GstLALAdder * adder, GstLALAdderClass * klass)
{
  GstPadTemplate *template;

  template = gst_static_pad_template_get (&gst_adder_src_template);
  adder->srcpad = gst_pad_new_from_template (template, "src");
  gst_object_unref (template);

  gst_pad_set_getcaps_function (adder->srcpad,
      GST_DEBUG_FUNCPTR (gst_pad_proxy_getcaps));
  gst_pad_set_setcaps_function (adder->srcpad,
      GST_DEBUG_FUNCPTR (gst_adder_setcaps));
  gst_pad_set_query_function (adder->srcpad,
      GST_DEBUG_FUNCPTR (gst_adder_query));
  gst_pad_set_event_function (adder->srcpad,
      GST_DEBUG_FUNCPTR (gst_adder_src_event));
  gst_element_add_pad (GST_ELEMENT (adder), adder->srcpad);

  adder->format = GST_ADDER_FORMAT_UNSET;
  adder->padcount = 0;
  adder->func = NULL;

  adder->filter_caps = NULL;

  /* keep track of the sinkpads requested */
  adder->collect = gst_collect_pads_new ();
  gst_collect_pads_set_function (adder->collect,
      GST_DEBUG_FUNCPTR (gst_adder_collected), adder);
  /* gst_collect_pads_set_clip_function (adder->collect,
      GST_DEBUG_FUNCPTR (gst_adder_do_clip), adder); */
}

static void
gst_adder_dispose (GObject * object)
{
  GstLALAdder *adder = GST_ADDER (object);

  if (adder->collect) {
    gst_object_unref (adder->collect);
    adder->collect = NULL;
  }
  gst_caps_replace (&adder->filter_caps, NULL);
  /* FIXME:  switch to
    g_list_free_full (adder->pending_events, (GDestroyNotify) gst_event_unref);
    adder->pending_events = NULL;
  */
  while (adder->pending_events) {
    GstEvent *ev = GST_EVENT (adder->pending_events->data);
    gst_event_unref (ev);
    adder->pending_events = g_list_remove (adder->pending_events, ev);
  }

  G_OBJECT_CLASS (parent_class)->dispose (object);
}

static void
gst_adder_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstLALAdder *adder = GST_ADDER (object);

  switch (prop_id) {
    case PROP_FILTER_CAPS:{
      GstCaps *new_caps = NULL;
      GstCaps *old_caps;
      const GstCaps *new_caps_val = gst_value_get_caps (value);

      if (new_caps_val != NULL) {
        new_caps = (GstCaps *) new_caps_val;
        gst_caps_ref (new_caps);
      }

      GST_OBJECT_LOCK (adder);
      old_caps = adder->filter_caps;
      adder->filter_caps = new_caps;
      GST_OBJECT_UNLOCK (adder);

      if (old_caps)
        gst_caps_unref (old_caps);

      GST_DEBUG_OBJECT (adder, "set new caps %" GST_PTR_FORMAT, new_caps);
      break;
    }
    case PROP_SYNCHRONOUS:
      /* FIXME:  on asynchronous --> synchronous transition, mark all
       * collect pad's offset_offsets as invalid to force a resync */
      adder->synchronous = g_value_get_boolean (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_adder_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec)
{
  GstLALAdder *adder = GST_ADDER (object);

  switch (prop_id) {
    case PROP_FILTER_CAPS:
      GST_OBJECT_LOCK (adder);
      gst_value_set_caps (value, adder->filter_caps);
      GST_OBJECT_UNLOCK (adder);
      break;
    case PROP_SYNCHRONOUS:
      g_value_set_boolean (value, adder->synchronous);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}


static GstPad *
gst_adder_request_new_pad (GstElement * element, GstPadTemplate * templ,
    const gchar * unused)
{
  gchar *name;
  GstLALAdder *adder;
  GstPad *newpad;
  GstLALCollectData *data = NULL;
  gint padcount;

  if (templ->direction != GST_PAD_SINK)
    goto not_sink;

  adder = GST_ADDER (element);

  /* increment pad counter */
  padcount = g_atomic_int_exchange_and_add (&adder->padcount, 1);

  name = g_strdup_printf ("sink%d", padcount);
  newpad = gst_pad_new_from_template (templ, name);
  GST_DEBUG_OBJECT (adder, "request new pad %s", name);
  g_free (name);

  gst_pad_set_getcaps_function (newpad,
      GST_DEBUG_FUNCPTR (gst_adder_sink_getcaps));
  gst_pad_set_setcaps_function (newpad, GST_DEBUG_FUNCPTR (gst_adder_setcaps));
  GST_OBJECT_LOCK (adder->collect);
  data = gstlal_collect_pads_add_pad (adder->collect, newpad, sizeof (*data));
  gstlal_collect_pads_set_unit_size (newpad, adder->bps);
  gstlal_collect_pads_set_rate (newpad, adder->rate);
  GST_OBJECT_UNLOCK (adder->collect);

  /* FIXME: hacked way to override/extend the event function of
   * GstCollectPads; because it sets its own event function giving the
   * element no access to events */
  adder->collect_event = (GstPadEventFunction) GST_PAD_EVENTFUNC (newpad);
  gst_pad_set_event_function (newpad, GST_DEBUG_FUNCPTR (gst_adder_sink_event));

  /* takes ownership of the pad */
  if (!gst_element_add_pad (GST_ELEMENT (adder), newpad))
    goto could_not_add;

  return newpad;

  /* errors */
not_sink:
  {
    g_warning ("gstadder: request new pad that is not a SINK pad\n");
    return NULL;
  }
could_not_add:
  {
    GST_DEBUG_OBJECT (adder, "could not add pad");
    gstlal_collect_pads_remove_pad (adder->collect, newpad);
    gst_object_unref (newpad);
    return NULL;
  }
}

static void
gst_adder_release_pad (GstElement * element, GstPad * pad)
{
  GstLALAdder *adder = GST_ADDER (element);

  GST_DEBUG_OBJECT (adder, "release pad %s:%s", GST_DEBUG_PAD_NAME (pad));

  gstlal_collect_pads_remove_pad (adder->collect, pad);
  gst_element_remove_pad (element, pad);
}

static GstBuffer *
gst_adder_do_clip (GstCollectPads * pads, GstCollectData * data,
    GstBuffer * buffer, gpointer user_data)
{
  GstLALAdder *adder = GST_ADDER (user_data);

  buffer = gst_audio_buffer_clip (buffer, &data->segment, adder->rate,
      adder->bps);

  return buffer;
}

static GstClockTime
output_timestamp_from_offset (const GstLALAdder *adder, guint64 offset)
{
  return adder->segment.start + gst_util_uint64_scale_int_round (offset,
      GST_SECOND, adder->rate);
}

static GstFlowReturn
gst_adder_collected (GstCollectPads * pads, gpointer user_data)
{
  /*
   * combine streams by adding data values
   * basic algorithm :
   * - this function is called when all pads have a buffer
   * - get available bytes on all pads.
   * - repeat for each input pad :
   *   - read available bytes, copy or add to target buffer
   *   - if there's an EOS event, remove the input channel
   * - push out the output buffer
   *
   * todo:
   * - would be nice to have a mixing mode, where instead of adding we mix
   *   - for float we could downscale after collect loop
   *   - for int we need to downscale each input to avoid clipping or
   *     mix into a temp (float) buffer and scale afterwards as well
   */
  GstLALAdder *adder;
  GSList *collected;
  GstBuffer *outbuf = NULL;
  GSList *partial_nongap_buffers = NULL;
  GstBuffer *full_gap_buffer = NULL;
  gboolean have_gap_buffers = FALSE;
  GstFlowReturn ret;
  guint64 outlength;
  GstClockTime t_start, t_end;
  guint64 earliest_output_offset, earliest_output_offset_end;

  adder = GST_ADDER (user_data);

  /* this is fatal */
  if (G_UNLIKELY (adder->func == NULL))
    goto not_negotiated;

  /* flush stop event if needed */
  if (g_atomic_int_compare_and_exchange (&adder->flush_stop_pending,
          TRUE, FALSE)) {
    GST_DEBUG_OBJECT (adder, "pending flush stop");
    gst_pad_push_event (adder->srcpad, gst_event_new_flush_stop ());
  }

  /* do new segment event if needed */
  if (G_UNLIKELY (adder->segment_pending)) {
    GstSegment *segment;
    GstEvent *event;

    segment = gstlal_collect_pads_get_segment (adder->collect);
    if (segment) {
      /* FIXME:  are other formats OK? */
      g_assert (segment->format == GST_FORMAT_TIME);
      adder->segment = *segment;
      gst_segment_free (segment);
    } else
      GST_ELEMENT_ERROR (adder, STREAM, FORMAT, (NULL), ("failed to deduce output segment, falling back to undefined default"));

    /* FIXME, use rate/applied_rate as set on all sinkpads.
     * - currently we just set rate as received from last seek-event
     *
     * When seeking we set the start and stop positions as given in the seek
     * event. We also adjust offset & timestamp acordingly.
     * This basically ignores all newsegments sent by upstream.
     */
    event = gst_event_new_new_segment_full (FALSE, adder->segment.rate,
        1.0, GST_FORMAT_TIME, adder->segment.start, adder->segment.stop,
        adder->segment.start);
    if (adder->segment.rate > 0.0) {
      adder->timestamp = adder->segment.start;
      adder->offset = 0;
    } else {
      adder->timestamp = adder->segment.stop;
      adder->offset = gst_util_uint64_scale_round (adder->segment.stop - adder->segment.start, adder->rate, GST_SECOND);
    }
    GST_INFO_OBJECT (adder, "seg_start %" G_GUINT64_FORMAT ", seg_end %"
        G_GUINT64_FORMAT, adder->segment.start, adder->segment.stop);
    GST_INFO_OBJECT (adder, "timestamp %" G_GINT64_FORMAT ", new offset %"
        G_GINT64_FORMAT, adder->timestamp, adder->offset);

    if (event) {
      if (!gst_pad_push_event (adder->srcpad, event)) {
        GST_WARNING_OBJECT (adder->srcpad, "Sending event  %p (%s) failed.",
            event, GST_EVENT_TYPE_NAME (event));
      }
      adder->segment_pending = FALSE;
    } else {
      GST_WARNING_OBJECT (adder->srcpad, "Creating new segment event for "
          "start:%" G_GINT64_FORMAT "  end:%" G_GINT64_FORMAT " failed",
          adder->segment.start, adder->segment.stop);
    }
  }

  /* do other pending events, e.g., tags */
  if (G_UNLIKELY (adder->pending_events)) {
    while (adder->pending_events) {
      GstEvent *ev = GST_EVENT (adder->pending_events->data);
      gst_pad_push_event (adder->srcpad, ev);
      adder->pending_events = g_list_remove (adder->pending_events, ev);
    }
  }

  /* get the range of offsets in the output stream spanned by the available
   * input buffers */
  if (adder->synchronous) {
    /* when doing synchronous adding, determine the offsetes for real */
    if (!gstlal_collect_pads_get_earliest_times (adder->collect, &t_start, &t_end)) {
      GST_ELEMENT_ERROR (adder, STREAM, FORMAT, (NULL), ("cannot deduce input timestamp offset information"));
      goto bad_timestamps;
    }
    /* check for EOS */
    if (!GST_CLOCK_TIME_IS_VALID (t_start))
      goto eos;
    /* don't let time go backwards */
    earliest_output_offset = gst_util_uint64_scale_int_round (t_start - adder->segment.start, adder->rate, GST_SECOND);
    earliest_output_offset_end = gst_util_uint64_scale_int_round (t_end - adder->segment.start, adder->rate, GST_SECOND);

    if (earliest_output_offset < adder->offset) {
      GST_ELEMENT_ERROR (adder, STREAM, FORMAT, (NULL), ("detected time reversal in at least one input stream:  expected nothing earlier than offset %" G_GUINT64_FORMAT ", found sample at offset %" G_GUINT64_FORMAT, adder->offset, earliest_output_offset));
      goto bad_timestamps;
    }
  } else {
    /* when not doing synchronous adding, use the element's output offset
     * counter and the number of bytes available.
     * gst_collect_pads_available() returns 0 on EOS, which we'll figure
     * out later when no pads produce buffers. */
    earliest_output_offset = adder->offset;
    earliest_output_offset_end = earliest_output_offset + gst_collect_pads_available (pads) / adder->bps;
  }

  /* compute the number of samples for which all sink pads can contribute
   * information.  0 does not necessarily mean EOS. */
  outlength = earliest_output_offset_end - earliest_output_offset;

  /* collect input buffers */
  GST_LOG_OBJECT(adder, "cycling through channels, offsets [%"
      G_GUINT64_FORMAT ", %" G_GUINT64_FORMAT ") (@ %d Hz) relative to %"
      GST_TIME_SECONDS_FORMAT " available", earliest_output_offset,
      earliest_output_offset_end, adder->rate,
      GST_TIME_SECONDS_ARGS(adder->segment.start));
  for (collected = pads->data; collected; collected = g_slist_next (collected)) {
    GstLALCollectData *collect_data = (GstLALCollectData *) collected->data;
    GstBuffer *inbuf;
    guint offset;
    guint64 inlength;

    /* (try to) get a buffer upto the desired end offset.
     * NULL means EOS or an empty buffer so we still need to flush in
     * case of an empty buffer.
     * determine the buffer's location relative to the desired range of
     * offsets.  we've checked above that time hasn't gone backwards on any
     * input buffer so offset can't be negative.  if not doing synchronous
     * adding the buffer starts "now". */
    if (adder->synchronous) {
      inbuf = gstlal_collect_pads_take_buffer_sync (pads, collect_data, t_end);
      if (inbuf == NULL) {
        GST_LOG_OBJECT (adder, "channel %p: no bytes available", collect_data);
        continue;
      }
      offset = gst_util_uint64_scale_int_round (GST_BUFFER_TIMESTAMP (inbuf) - adder->segment.start, adder->rate, GST_SECOND) - earliest_output_offset;
      inlength = GST_BUFFER_OFFSET_END (inbuf) - GST_BUFFER_OFFSET (inbuf);
      g_assert (inlength == GST_BUFFER_SIZE (inbuf) / adder->bps || GST_BUFFER_FLAG_IS_SET (inbuf, GST_BUFFER_FLAG_GAP));
    } else {
      inbuf = gst_collect_pads_take_buffer (pads, (GstCollectData *) collect_data, outlength * adder->bps);
      if (inbuf == NULL) {
        GST_LOG_OBJECT (adder, "channel %p: no bytes available", collect_data);
        continue;
      }
      offset = 0;
      inlength = GST_BUFFER_SIZE (inbuf) / adder->bps;
    }
    g_assert (offset + inlength <= outlength || inlength == 0);
    GST_LOG_OBJECT (adder, "channel %p: retrieved %d sample buffer at %" GST_TIME_FORMAT, collect_data, inlength, GST_TIME_ARGS (GST_BUFFER_TIMESTAMP (inbuf)));

    /* this buffer is for the future, we don't need it yet. */
    if (offset > outlength || (offset == outlength && outlength != 0)) {
      /* it must be empty or there's a bug in the collect pads class */
      g_assert (inlength == 0);
      GST_LOG_OBJECT (adder, "channel %p: discarding 0 sample buffer from the future", collect_data);
      gst_buffer_unref (inbuf);
      continue;
    }

    /* keep one of the full gap buffers to reuse as output incase we don't
     * get anything else, record whether or not we saw any gap buffers at
     * all, add all full non-gap buffers together, and collect a list of
     * the partial non-gap buffers to add into the result later. */
    if (GST_BUFFER_FLAG_IS_SET (inbuf, GST_BUFFER_FLAG_GAP)) {	/* is it a gap? */
      have_gap_buffers = TRUE;
      if (offset == 0 && inlength == outlength && !full_gap_buffer)	/* does it span the full output interval?  and we haven't yet seen one that does? */
        full_gap_buffer = inbuf;
      else	/* we don't need this buffer */
        gst_buffer_unref (inbuf);
    } else if (offset == 0 && inlength == outlength) {	/* not a gap, does it span the full output interval? */
      if (!outbuf)	/* if we don't have a buffer to hold the output yet, this one's it */
        outbuf = inbuf;
      else {	/* add this buffer to the output buffer */
        outbuf = gst_buffer_make_writable (outbuf);
        adder->func (GST_BUFFER_DATA (outbuf), GST_BUFFER_DATA (inbuf), GST_BUFFER_SIZE (outbuf) / adder->sample_size);
        gst_buffer_unref (inbuf);
      }
    } else	/* not a gap, doesn't span the full output interval, process it later */
      partial_nongap_buffers = g_slist_prepend (partial_nongap_buffers, inbuf);
  }

  /* now add partial non-gap buffers */
  if (partial_nongap_buffers) {
    if (!outbuf) {
      /* this code path should only be possible if the input included a gap
       * buffer spanning the full input interval */
      g_assert(full_gap_buffer != NULL);
      /* get a buffer of zeros */
      ret = gst_pad_alloc_buffer (adder->srcpad, earliest_output_offset, outlength * adder->bps, GST_BUFFER_CAPS (full_gap_buffer), &outbuf);
      if (ret != GST_FLOW_OK) {
        /* FIXME:  replace with
        g_slist_free_full (partial_nongap_buffers, (GDestroyNotify) gst_buffer_unref);
        */
        while (partial_nongap_buffers) {
          GstBuffer *inbuf = GST_BUFFER (partial_nongap_buffers->data);
          gst_buffer_unref (inbuf);
          partial_nongap_buffers = g_slist_remove (partial_nongap_buffers, inbuf);
	}
	goto no_buffer;
      }
      g_assert (GST_BUFFER_CAPS (outbuf) != NULL);
      memset (GST_BUFFER_DATA (outbuf), 0, GST_BUFFER_SIZE (outbuf));
    } else
      outbuf = gst_buffer_make_writable (outbuf);
    while (partial_nongap_buffers) {
      GstBuffer *inbuf = GST_BUFFER (partial_nongap_buffers->data);
      guint offset = adder->synchronous ?  gst_util_uint64_scale_int_round (GST_BUFFER_TIMESTAMP (inbuf) - adder->segment.start, adder->rate, GST_SECOND) - earliest_output_offset : 0;
      g_assert (offset * adder->bps + GST_BUFFER_SIZE (inbuf) <= GST_BUFFER_SIZE (outbuf) || GST_BUFFER_SIZE (inbuf) == 0);
      adder->func (GST_BUFFER_DATA (outbuf) + offset * adder->bps, GST_BUFFER_DATA (inbuf), GST_BUFFER_SIZE (inbuf) / adder->sample_size);
      partial_nongap_buffers = g_slist_remove (partial_nongap_buffers, inbuf);
      gst_buffer_unref (inbuf);
    }
  }

  /* if we don't have an output buffer yet, then if there's a full gap
   * buffer it becomes our output, otherwise we're at EOS */
  if (outbuf) {
    if (full_gap_buffer)
      gst_buffer_unref (full_gap_buffer);
  } else if (full_gap_buffer)
    outbuf = full_gap_buffer;
  else if (have_gap_buffers) {
    /* the condition of having only partial gap buffers and nothing else is
     * not possible.  getting here implies a bug in the code that
     * determines the times spanned by the available input buffers */
    g_assert_not_reached();
  } else
    goto eos;

  /* FIXME:  this logic can't run backwards */
  /* set timestamps on the output buffer */
  outbuf = gst_buffer_make_metadata_writable (outbuf);
  GST_BUFFER_OFFSET (outbuf) = earliest_output_offset;
  GST_BUFFER_TIMESTAMP (outbuf) = output_timestamp_from_offset (adder, GST_BUFFER_OFFSET (outbuf));
  if (GST_BUFFER_OFFSET (outbuf) == 0 || GST_BUFFER_TIMESTAMP (outbuf) != adder->timestamp)
    GST_BUFFER_FLAG_SET (outbuf, GST_BUFFER_FLAG_DISCONT);
  else
    GST_BUFFER_FLAG_UNSET (outbuf, GST_BUFFER_FLAG_DISCONT);
  GST_BUFFER_OFFSET_END (outbuf) = GST_BUFFER_OFFSET (outbuf) + outlength;
  adder->timestamp = output_timestamp_from_offset (adder, GST_BUFFER_OFFSET_END (outbuf));
  adder->offset = GST_BUFFER_OFFSET_END (outbuf);
  GST_BUFFER_DURATION (outbuf) = adder->timestamp - GST_BUFFER_TIMESTAMP (outbuf);

  /* send it out */
  g_assert (GST_BUFFER_CAPS (outbuf) != NULL);
  GST_LOG_OBJECT (adder, "pushing outbuf %p spanning %" GST_BUFFER_BOUNDARIES_FORMAT, outbuf, GST_BUFFER_BOUNDARIES_ARGS (outbuf));
  ret = gst_pad_push (adder->srcpad, outbuf);
  GST_LOG_OBJECT (adder, "pushed outbuf, result = %s", gst_flow_get_name (ret));

  return ret;

  /* ERRORS */
no_buffer:
bad_timestamps:
  {
    return GST_FLOW_ERROR;
  }
not_negotiated:
  {
    GST_ELEMENT_ERROR (adder, STREAM, FORMAT, (NULL),
        ("Unknown data received, not negotiated"));
    return GST_FLOW_NOT_NEGOTIATED;
  }
eos:
  {
    GST_DEBUG_OBJECT (adder, "no data available, must be EOS");
    gst_pad_push_event (adder->srcpad, gst_event_new_eos ());
    return GST_FLOW_UNEXPECTED;
  }
}

static GstStateChangeReturn
gst_adder_change_state (GstElement * element, GstStateChange transition)
{
  GstLALAdder *adder;
  GstStateChangeReturn ret;

  adder = GST_ADDER (element);

  switch (transition) {
    case GST_STATE_CHANGE_NULL_TO_READY:
      break;
    case GST_STATE_CHANGE_READY_TO_PAUSED:
      adder->timestamp = 0;
      adder->offset = 0;
      adder->flush_stop_pending = FALSE;
      adder->segment_pending = TRUE;
      gst_segment_init (&adder->segment, GST_FORMAT_UNDEFINED);
      gst_collect_pads_start (adder->collect);
      break;
    case GST_STATE_CHANGE_PAUSED_TO_PLAYING:
      break;
    case GST_STATE_CHANGE_PAUSED_TO_READY:
      /* need to unblock the collectpads before calling the
       * parent change_state so that streaming can finish */
      gst_collect_pads_stop (adder->collect);
      break;
    default:
      break;
  }

  ret = GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);

  switch (transition) {
    default:
      break;
  }

  return ret;
}


static gboolean
plugin_init (GstPlugin * plugin)
{
  GST_DEBUG_CATEGORY_INIT (GST_CAT_DEFAULT, "lal_adder", 0,
      "audio channel mixing element");

  gst_adder_orc_init ();

  if (!gst_element_register (plugin, "lal_adder", GST_RANK_NONE, GST_TYPE_ADDER)) {
    return FALSE;
  }

  return TRUE;
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    "lal_adder",
    "Adds multiple streams",
    plugin_init, VERSION, "LGPL", GST_PACKAGE_NAME, GST_PACKAGE_ORIGIN)

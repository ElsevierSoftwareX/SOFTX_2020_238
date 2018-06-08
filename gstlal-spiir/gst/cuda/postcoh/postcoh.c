/* 
 * Copyright (C) 2014 Qi Chu <qi.chu@ligo.org>
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

/* This element will synchronize the snr sequencies from all detectors, find 
 * peaks from all detectors and for each peak, do null stream analysis.
 */


#include <gst/gst.h>
#include <lal/Date.h>
#include <lal/LIGOMetadataTables.h>
#include <chealpix.h>

#include <string.h>
#include <math.h>
#include "postcoh.h"
#include "postcoh_utils.h"
#include "postcohinspiral_table_utils.h"
#include <cuda_debug.h>

#define GST_CAT_DEFAULT gstlal_postcoh_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);

#define DEFAULT_DETRSP_FNAME "L1H1V1_detrsp.xml"
#define EPSILON 5
#define PEAKFINDER_CLUSTER_WINDOW 5
#define RAD2DEG 57.2957795
#define L_MAPPING 0
#define H_MAPPING 1
#define V_MAPPING 2
#define ACCELERATE_POSTCOH_MEMORY_COPY
static gboolean cuda_postcoh_sink_setcaps(CudaPostcoh *postcoh, GstPad *pad, GstCaps *caps);
static guint64 adapter_available_samples(GstAudioAdapter *adapter);
const guint8* gst_adapter_peek_cuda(GstAudioAdapter *adapter, guint size);
static void cuda_postcoh_device_set_init(CudaPostcoh *element);
static void cuda_postcoh_set_property(GObject *object, guint id, const GValue *value, GParamSpec *pspec);
static gboolean sink_event(GstCollectPads *pads, GstCollectData *pad, GstEvent *event, gpointer user_data);


G_DEFINE_TYPE_WITH_CODE(
	CudaPostcoh,
	cuda_postcoh,
	GST_TYPE_ELEMENT,
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "cuda_postcoh", 0, "cuda_postcoh element")
);

//G_DEFINE_TYPE(CudaPostcohPad, cuda_postcoh_pad, GST_TYPE_PAD);

static guint64 adapter_available_samples(GstAudioAdapter *adapter)
{
  guint size;
  g_object_get(adapter, "size", &size, NULL);
  return size;
}

#ifdef ACCELERATE_POSTCOH_MEMORY_COPY
// This function is just copy from gst_adapter_peek and change malloc to cudaMallocHost
// FIXME: temporarily used to speed up cudaMemcpy, will be deleted when transfered to gstreamer1.x
const guint8* gst_adapter_peek_cuda(GstAudioAdapter *adapter, guint size)
{
  //int DEFAULT_SIZE = 16;
  //GstBuffer *cur;
  //GSList *cur_list;
  //guint copied;
  
  guint8 *output;

#if 0
  g_return_val_if_fail (GST_IS_ADAPTER (adapter), NULL);
  g_return_val_if_fail (size > 0, NULL);

  /* we don't have enough data, return NULL. This is unlikely
 *    * as one usually does an _available() first instead of peeking a
 *       * random size. */
  if (G_UNLIKELY (size > adapter->size))
    return NULL;

  /* we have enough assembled data, return it */
  if (adapter->assembled_len >= size)
    return adapter->assembled_data;

  /* our head buffer has enough data left, return it */
  cur = adapter->buflist->data;
  if (GST_BUFFER_SIZE (cur) >= size + adapter->skip)
    return GST_BUFFER_DATA (cur) + adapter->skip;

#endif
  /*if (adapter->assembled_size < size) {
    adapter->assembled_size = (size / DEFAULT_SIZE + 1) * DEFAULT_SIZE;
    GST_DEBUG_OBJECT (adapter, "setting size of internal buffer to %u",
        adapter->assembled_size);
    CUDA_CHECK(cudaMallocHost((void**)&(adapter->assembled_data), adapter->assembled_size));
  }*/

  /* FIXME: Currently using a very simplified code, can add those conditions check later  */
  CUDA_CHECK(cudaMallocHost((void **)&output, size));
  gst_audioadapter_copy_samples(adapter, output, size, NULL, NULL);

#if 0
  adapter->assembled_len = size;
  copied = GST_BUFFER_SIZE (cur) - adapter->skip;
  memcpy (adapter->assembled_data, GST_BUFFER_DATA (cur) + adapter->skip,
      copied);

  cur_list = g_slist_next (adapter->buflist);
  while (copied < size) {
    g_assert (cur_list);
    cur = cur_list->data;
    cur_list = g_slist_next (cur_list);
    memcpy (adapter->assembled_data + copied, GST_BUFFER_DATA (cur),
        MIN (GST_BUFFER_SIZE (cur), size - copied));
    copied = MIN (size, copied + GST_BUFFER_SIZE (cur));
  }
#endif
  //return adapter->assembled_data;
  return output;
}
#endif





//FIXME: not support width=64 yet
static GstStaticPadTemplate cuda_postcoh_sink_template =
GST_STATIC_PAD_TEMPLATE (
		"%s",
		GST_PAD_SINK, 
		GST_PAD_REQUEST, 
		GST_STATIC_CAPS(
		  GST_AUDIO_CAPS_MAKE("{" GST_AUDIO_NE(F32) "}") ", " \
		"layout = (string) interleaved, " \
		"channel-mask = (bitmask) 0"
		));
/* the following is for a src template that's the same with
 * the sink template 
static GstStaticPadTemplate cuda_postcoh_src_template =
GST_STATIC_PAD_TEMPLATE (
		"src",
		GST_PAD_SRC, 
		GST_PAD_ALWAYS, 
		GST_STATIC_CAPS(
		"audio/x-raw-float, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) [1, MAX], " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) 32"
		));
*/


enum 
{
	PROP_0,
	PROP_DETRSP_FNAME,
	PROP_SPIIR_BANK_FNAME,
	PROP_SNGL_TMPLT_FNAME,
	PROP_HIST_TRIALS,
	PROP_TRIAL_INTERVAL,
	PROP_OUTPUT_SKYMAP,
	PROP_COHSNR_THRESH,
	PROP_SNGLSNR_THRESH,
	PROP_STREAM_ID
};

static void cuda_postcoh_device_set_init(CudaPostcoh *element)
{
	if (element->device_id == NOT_INIT) {
		int deviceCount;
		CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
		element->device_id = element->stream_id % deviceCount;
		GST_LOG("device for postcoh %d\n", element->device_id);
		CUDA_CHECK(cudaSetDevice(element->device_id));
		CUDA_CHECK(cudaStreamCreateWithFlags(&element->stream, cudaStreamNonBlocking));
	}

}

static void cuda_postcoh_set_property(GObject *object, guint id, const GValue *value, GParamSpec *pspec)
{
	CudaPostcoh *element = CUDA_POSTCOH(object);

	GST_OBJECT_LOCK(element);
	switch(id) {
		/* read in detector response map */
		case PROP_DETRSP_FNAME:
       			/* must make sure stream_id has already loaded */
			g_assert(element->stream_id != NOT_INIT);
			g_mutex_lock(&element->prop_lock);
			element->detrsp_fname = g_value_dup_string(value);
			cuda_postcoh_device_set_init(element);
			CUDA_CHECK(cudaSetDevice(element->device_id));
			cuda_postcoh_map_from_xml(element->detrsp_fname, element->state, element->stream);
			GST_DEBUG("detrsp map has been read in, broad cast the lock avail");
			g_cond_broadcast(&element->prop_avail);
			g_mutex_unlock(&element->prop_lock);
			GST_DEBUG("detrsp map lock broad cast done");
			break;

		/* read in autocorrelation and sigmasq */
		case PROP_SPIIR_BANK_FNAME: 
       			/* must make sure stream_id has already loaded */
			g_assert(element->stream_id != NOT_INIT);
			GST_DEBUG("autocorrelation and sigma acquiring the lock");
			g_mutex_lock(&element->prop_lock);
			GST_DEBUG("autocorrelation and sigma have acquired the lock");
			cuda_postcoh_device_set_init(element);
			CUDA_CHECK(cudaSetDevice(element->device_id));
			element->spiir_bank_fname = g_value_dup_string(value);
			cuda_postcoh_autocorr_from_xml(element->spiir_bank_fname, element->state, element->stream);
			cuda_postcoh_sigmasq_from_xml(element->spiir_bank_fname, element->state);
			GST_DEBUG("autocorrelation and sigma have been read in, broad cast the lock avail");
			g_cond_broadcast(&element->prop_avail);
			g_mutex_unlock(&element->prop_lock);
			GST_DEBUG("autocorrelation and sigma lock broad cast done");
			break;

		/* read in source information masses and spins */
		case PROP_SNGL_TMPLT_FNAME: 
       			/* must make sure stream_id has already loaded */
			g_assert(element->stream_id != NOT_INIT);
			GST_DEBUG("sngl table acquiring the lock");
			g_mutex_lock(&element->prop_lock);
			GST_DEBUG("sngl table has acquired the lock");
			element->sngl_tmplt_fname = g_value_dup_string(value);
			cuda_postcoh_sngl_tmplt_from_xml(element->sngl_tmplt_fname, &(element->sngl_table));
			GST_DEBUG("sngl tables has been read in, broad cast the lock avail");
			g_cond_broadcast(&element->prop_avail);
			g_mutex_unlock(&element->prop_lock);
			GST_DEBUG("sngl tables lock broad cast done");
			break;


		case PROP_HIST_TRIALS:
			g_mutex_lock(&element->prop_lock);
			element->hist_trials = g_value_get_int(value);
			g_cond_broadcast(&element->prop_avail);
			g_mutex_unlock(&element->prop_lock);
			break;

		case PROP_TRIAL_INTERVAL:
			g_mutex_lock(&element->prop_lock);
			element->trial_interval = g_value_get_float(value);
			g_cond_broadcast(&element->prop_avail);
			g_mutex_unlock(&element->prop_lock);
			break;

		case PROP_OUTPUT_SKYMAP:
			element->output_skymap = g_value_get_int(value);
			break;

		case PROP_COHSNR_THRESH:
			element->cohsnr_thresh = g_value_get_float(value);
			break;

		case PROP_SNGLSNR_THRESH:
			element->snglsnr_thresh = g_value_get_float(value);
			element->state->snglsnr_thresh = element->snglsnr_thresh;
			break;

		case PROP_STREAM_ID:
			element->stream_id = g_value_get_int(value);
			break;

		default:
			G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
			break;
	}
	GST_OBJECT_UNLOCK(element);
}


static void cuda_postcoh_get_property(GObject * object, guint id, GValue * value, GParamSpec * pspec)
{
	CudaPostcoh *element = CUDA_POSTCOH(object);

	GST_OBJECT_LOCK(element);
	switch(id) {
		case PROP_DETRSP_FNAME:
			g_value_set_string(value, element->detrsp_fname);
			break;

		case PROP_SPIIR_BANK_FNAME:
			g_value_set_string(value, element->spiir_bank_fname);
			break;

		case PROP_SNGL_TMPLT_FNAME:
			g_value_set_string(value, element->sngl_tmplt_fname);
			break;


		case PROP_HIST_TRIALS:
			g_value_set_int(value, element->hist_trials);
			break;

		case PROP_TRIAL_INTERVAL:
			g_value_set_float(value, element->trial_interval);
			break;

		case PROP_OUTPUT_SKYMAP:
			g_value_set_int(value, element->output_skymap);
			break;

		case PROP_COHSNR_THRESH:
			g_value_set_float(value, element->cohsnr_thresh);
			break;


		case PROP_SNGLSNR_THRESH:
			g_value_set_float(value, element->snglsnr_thresh);
			break;

		case PROP_STREAM_ID:
			g_value_set_int (value, element->stream_id);
			break;

		default:
			G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
			break;
	}
	GST_OBJECT_UNLOCK(element);
}

static void set_offset_per_nanosecond(GstPostcohCollectData *data, double offset_per_nanosecond)
{
	data->offset_per_nanosecond = offset_per_nanosecond;

}

static void set_channels(GstPostcohCollectData *data, gint channels)
{
	data->channels = channels;

}

static void set_aligned_offset0(GstPostcohCollectData *data, guint64 offset)
{
	data->aligned_offset0 = offset;

}

/*
 * Copied from gstadder.c on 14 Nov 2071, shinkee
 */
#if 0
static GstCaps *
cuda_postcoh_sink_getcaps (GstPad * pad, GstCaps * filter)
{
  CudaPostcoh *postcoh;
  GstCaps *result, *peercaps, *current_caps, *filter_caps;

  postcoh = CUDA_POSTCOH(GST_PAD_PARENT (pad));

  GST_LOG_OBJECT(pad, "Debug pad %" GST_PTR_FORMAT, pad);
  GST_OBJECT_LOCK (postcoh);
  /* take filter */
  if ((filter_caps = postcoh->filter_caps)) {
    if (filter)
      filter_caps =
          gst_caps_intersect_full (filter, filter_caps,
          GST_CAPS_INTERSECT_FIRST);
    else
      gst_caps_ref (filter_caps);
  } else {
    filter_caps = filter ? gst_caps_ref (filter) : NULL;
  }
  GST_OBJECT_UNLOCK (postcoh);

  if (filter_caps && gst_caps_is_empty (filter_caps)) {
    GST_WARNING_OBJECT (pad, "Empty filter caps");
    return filter_caps;
  }

  /* get the downstream possible caps */
  peercaps = gst_pad_peer_query_caps (postcoh->srcpad, filter_caps);
  GST_LOG_OBJECT(peercaps, "Debug caps %" GST_PTR_FORMAT, peercaps);

  /* get the allowed caps on this sinkpad */
  GST_OBJECT_LOCK (postcoh);
  current_caps =
      postcoh->current_caps ? gst_caps_ref (postcoh->current_caps) : NULL;
  if (current_caps == NULL) {
    current_caps = gst_pad_get_pad_template_caps (pad);
    if (!current_caps)
      current_caps = gst_caps_new_any ();
  }
  GST_OBJECT_UNLOCK (postcoh);

  if (peercaps) {
    /* if the peer has caps, intersect */
    GST_DEBUG_OBJECT (postcoh, "intersecting peer and our caps");
    result =
        gst_caps_intersect_full (peercaps, current_caps,
        GST_CAPS_INTERSECT_FIRST);
    gst_caps_unref (peercaps);
    gst_caps_unref (current_caps);
  } else {
    /* the peer has no caps (or there is no peer), just use the allowed caps
     * of this sinkpad. */
    /* restrict with filter-caps if any */
    if (filter_caps) {
      GST_DEBUG_OBJECT (postcoh, "no peer caps, using filtered caps");
      result =
          gst_caps_intersect_full (filter_caps, current_caps,
          GST_CAPS_INTERSECT_FIRST);
      gst_caps_unref (current_caps);
    } else {
      GST_DEBUG_OBJECT (postcoh, "no peer caps, using our caps");
      result = current_caps;
    }
  }

  if (filter_caps)
    gst_caps_unref (filter_caps);

  GST_LOG_OBJECT (postcoh, 
      "getting caps on pad %p, %s to %" GST_PTR_FORMAT, 
      pad, GST_PAD_NAME (pad), result);

  return result;
}
#endif

/*
 * Copied from gstadder.c on 14 Nov 2071, shinkee
 */
static gboolean
cuda_postcoh_query_duration (CudaPostcoh * postcoh, GstQuery * query)
{
  gint64 max;
  gboolean res;
  GstFormat format;
  GstIterator *it;
  gboolean done;
  GValue item = { 0, };

  /* parse format */
  gst_query_parse_duration (query, &format, NULL);

  max = -1;
  res = TRUE;
  done = FALSE;

  it = gst_element_iterate_sink_pads (GST_ELEMENT_CAST (postcoh));
  while (!done) {
    GstIteratorResult ires;

    ires = gst_iterator_next (it, &item);
    switch (ires) {
      case GST_ITERATOR_DONE:
        done = TRUE;
        break;
      case GST_ITERATOR_OK:
      {
        GstPad *pad = g_value_get_object (&item);
        gint64 duration;

        /* ask sink peer for duration */
        res &= gst_pad_peer_query_duration (pad, format, &duration);
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
        g_value_reset (&item);
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
  g_value_unset (&item);
  gst_iterator_free (it);

  if (res) {
    /* and store the max */
    GST_DEBUG_OBJECT (postcoh, "Total duration in format %s: %"
        GST_TIME_FORMAT, gst_format_get_name (format), GST_TIME_ARGS (max));
    gst_query_set_duration (query, format, max);
  }

  return res;
}

/*
 * Copied from gstadder.c on 14 Nov 2071, shinkee
 */

static gboolean
cuda_postcoh_query_latency (CudaPostcoh *postcoh, GstQuery * query)
{
  GstClockTime min, max;
  gboolean live;
  gboolean res;
  GstIterator *it;
  gboolean done;
  GValue item = { 0, };

  res = TRUE;
  done = FALSE;

  live = FALSE;
  min = 0;
  max = GST_CLOCK_TIME_NONE;

  /* Take maximum of all latency values */
  it = gst_element_iterate_sink_pads (GST_ELEMENT_CAST (postcoh));
  while (!done) {
    GstIteratorResult ires;

    ires = gst_iterator_next (it, &item);
    switch (ires) {
      case GST_ITERATOR_DONE:
        done = TRUE;
        break;
      case GST_ITERATOR_OK:
      {
        GstPad *pad = g_value_get_object (&item);
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
        g_value_reset (&item);
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
  g_value_unset (&item);
  gst_iterator_free (it);

  if (res) {
    /* store the results */
    GST_DEBUG_OBJECT (postcoh, "Calculated total latency: live %s, min %"
        GST_TIME_FORMAT ", max %" GST_TIME_FORMAT,
        (live ? "yes" : "no"), GST_TIME_ARGS (min), GST_TIME_ARGS (max));
    gst_query_set_latency (query, live, min, max);
  }

  return res;
}

/*
 * Copied from gstadder.c on 14 Nov 2071, shinkee
 */
static gboolean 
src_query(GstPad * pad, GstObject * parent, GstQuery * query)
{
  CudaPostcoh *postcoh = CUDA_POSTCOH(parent);
  gboolean res = FALSE;

  GST_DEBUG_OBJECT(pad, "Debug postcoh src_query got query %s", GST_QUERY_TYPE_NAME(query));

  switch (GST_QUERY_TYPE (query)) {
    case GST_QUERY_POSITION:
    {
      GstFormat format;

      gst_query_parse_position (query, &format, NULL);

      switch (format) {
        case GST_FORMAT_TIME:
          /* FIXME, bring to stream time, might be tricky */
          gst_query_set_position (query, format, postcoh->segment.position);
          res = TRUE;
          break;
        case GST_FORMAT_DEFAULT:
          gst_query_set_position (query, format, postcoh->offset);
          res = TRUE;
          break;
        default:
          break;
      }
      break;
    }
    case GST_QUERY_DURATION:
      res = cuda_postcoh_query_duration (postcoh, query);
      break;
    case GST_QUERY_LATENCY:
      res = cuda_postcoh_query_latency (postcoh, query);
      break;
    default:
      /* FIXME, needs a custom query handler because we have multiple
       * sinkpads */
      res = gst_pad_query_default (pad, parent, query);
      break;
  }

  return res;
}

/*
 * Copied from gstadder.c on 14 Nov 2071, shinkee
 */
#if 0
static gboolean
sink_query (GstCollectPads * pads, GstCollectData * pad,
    GstQuery *query, gpointer user_data)
{
  gboolean res = FALSE;

  switch (GST_QUERY_TYPE (query)) {
    case GST_QUERY_CAPS:
    {
      GstCaps *filter, *caps;

      gst_query_parse_caps (query, &filter);
      caps = cuda_postcoh_sink_getcaps (pad->pad, filter);
      gst_query_set_caps_result (query, caps);
      gst_caps_unref (caps);
      res = TRUE;
      break;
    }
    default:
      res = gst_collect_pads_query_default (pads, pad, query, FALSE);
      break;
  }

  return res;
}
#endif

/*
 * Updated copy from gstadder.c on 14 Nov 2017, shinkee
 */
static gboolean
sink_event(GstCollectPads *pads, GstCollectData *pad, GstEvent *event, gpointer user_data)
{
  CudaPostcoh *postcoh = CUDA_POSTCOH(user_data);
  gboolean res = TRUE, discard = FALSE;

  switch(GST_EVENT_TYPE(event)) {
    case GST_EVENT_CAPS:
    {
      GstCaps *caps;
      gst_event_parse_caps(event, &caps);
      res = cuda_postcoh_sink_setcaps(postcoh, pad->pad, caps);
      gst_event_unref(event);
      event = NULL;
      break;
    }
    case GST_EVENT_FLUSH_START:
      /* ensure that we will send a flush stop */
      res = gst_collect_pads_event_default (pads, pad, event, discard);
      event = NULL;
      GST_COLLECT_PADS_STREAM_LOCK (postcoh->collect);
      postcoh->flush_stop_pending = TRUE;
      GST_COLLECT_PADS_STREAM_UNLOCK (postcoh->collect);
      break;
    case GST_EVENT_FLUSH_STOP:
      /* we received a flush-stop. We will only forward it when
       * flush_stop_pending is set, and we will unset it then.
       */
      g_atomic_int_set (&postcoh->new_segment_pending, TRUE);
      GST_COLLECT_PADS_STREAM_LOCK (postcoh->collect);
      if (postcoh->flush_stop_pending) {
        GST_DEBUG_OBJECT (pad->pad, "forwarding flush stop");
        res = gst_collect_pads_event_default (pads, pad, event, discard);
        postcoh->flush_stop_pending = FALSE;
        event = NULL;
      } else {
        discard = TRUE;
        GST_DEBUG_OBJECT (pad->pad, "eating flush stop");
      }
      GST_COLLECT_PADS_STREAM_UNLOCK (postcoh->collect);
      /* Clear pending tags */
      if (postcoh->pending_events) {
        g_list_foreach (postcoh->pending_events, (GFunc) gst_event_unref, NULL);
        g_list_free (postcoh->pending_events);
        postcoh->pending_events = NULL;
      }
      break;
    case GST_EVENT_TAG:
      /* collect tags here so we can push them out when we collect data */
      postcoh->pending_events = g_list_append (postcoh->pending_events, event);
      event = NULL;
      break;
    case GST_EVENT_SEGMENT:
      GST_DEBUG_OBJECT(pad, "new segment");
      const GstSegment *segment;
      gst_event_parse_segment(event, &segment);
      if( segment->rate != postcoh->segment.rate )
      {
	GST_ERROR_OBJECT(pad->pad, "Got segment event with wrong rate %lf, expected %lf",
	    segment->rate, postcoh->segment.rate);
	res = FALSE;
	gst_event_unref(event);
	event = NULL;
      }
      discard = TRUE;
      break;
    // do not process tag.
    default:
      break;
  }

  if(G_LIKELY(event))
    return gst_collect_pads_event_default(pads, pad, event, discard);
  else
    return res;
}

/*
 * forwards the event to all sinkpads, takes ownership of the event
 */


typedef struct {
	GstEvent *event;
	gboolean flush;
} EventData;

/*
 * Updated copy from gstadder.c on 14 Nov 2017, shinkee
 */
static gboolean forward_src_event_func(GValue *val, GValue *ret, EventData *data)
{
  GstPad *pad = g_value_get_object(val);
  GstEvent *event = data->event;
  GstPad *peer;

  gst_event_ref(event);

  GST_DEBUG_OBJECT(pad, "Debug postcoh forward an event %s", GST_EVENT_TYPE_NAME(event));
  peer = gst_pad_get_peer(pad);

  if(!peer || !gst_pad_send_event(peer, event))
  {
    if(!peer)
      gst_event_unref(event);
    GST_WARNING_OBJECT(pad, "Sending event %p (%s) failed.",
	event, GST_EVENT_TYPE_NAME(event));

    if(data->flush)
      gst_pad_send_event(pad, gst_event_new_flush_stop(TRUE));
  } 
  else
  {
    g_value_set_boolean(ret, TRUE);
    GST_LOG_OBJECT(pad, "Sent event %p (%s).",
	event, GST_EVENT_TYPE_NAME(event));
  }
  if(peer)
    gst_object_unref(peer);

  return TRUE;
}

/*
 * Updated copy from gstadder.c on 14 Nov 2017, shinkee
 */
static gboolean forward_src_event(CudaPostcoh *postcoh, GstEvent *event, gboolean flush)
{
  gboolean ret;
  GstIterator *it;
  GstIteratorResult ires;
  GValue vret = {0};
  EventData data = {event, flush};
  
  g_value_init(&vret, G_TYPE_BOOLEAN);
  g_value_set_boolean(&vret, FALSE);

  it = gst_element_iterate_sink_pads(GST_ELEMENT(postcoh));
  while(TRUE) {
    ires = gst_iterator_fold(it, (GstIteratorFoldFunction) forward_src_event_func, &vret, &data);
    switch (ires)
    {
      case GST_ITERATOR_RESYNC:
	GST_WARNING("resync");
	gst_iterator_resync(it);
	g_value_set_boolean(&vret, TRUE);
	break;

      case GST_ITERATOR_OK:
      case GST_ITERATOR_DONE:
	ret = g_value_get_boolean(&vret);
	goto done;

      default:
	ret = FALSE;
	goto done;
    }
  }
done:
  gst_iterator_free(it);
  GST_LOG_OBJECT(postcoh, "Debug postcoh forwarded event %p (%s), ret=%d", event, GST_EVENT_TYPE_NAME(event), ret);
  gst_event_unref(event);

  return ret;
}


/*
 * handle events received on the source pad
 * Updated copy from gstadder.c on 14 Nov 2017, shinkee
 */
static gboolean src_event(GstPad *pad, GstObject *parent, GstEvent *event)
{
  CudaPostcoh *postcoh;
  gboolean result;
  
  postcoh = CUDA_POSTCOH(parent);

  GST_DEBUG_OBJECT(pad, "Debug postcoh got %s event on src pad",
      GST_EVENT_TYPE_NAME(event));
  switch(GST_EVENT_TYPE(event)) {
    case GST_EVENT_SEEK: 
    {
      GstSeekFlags flags;
      gdouble rate;
      GstSeekType start_type, stop_type;
      gint64 start, stop;
      GstFormat seek_format, dest_format;
      gboolean flush;

      gst_event_parse_seek(event, &rate, &seek_format, &flags, &start_type, &start, &stop_type, &stop);
	
      if( (start_type != GST_SEEK_TYPE_NONE)
	&& (start_type != GST_SEEK_TYPE_SET) )
      {
	result = FALSE;
	GST_DEBUG_OBJECT( postcoh,
	    "seeking failed, unhandled seek type for start: %d", start_type);
	goto done;
	/* FIXME:  copy the adder's logic re flushing */
      }
      if( (stop_type != GST_SEEK_TYPE_NONE) 
	  && (stop_type != GST_SEEK_TYPE_SET) )
      {
	result = FALSE;
	GST_DEBUG_OBJECT( postcoh,
	    "seeking failed, unhandled seek type for end: %d", stop_type);
	goto done;
      }

      dest_format = postcoh->segment.format;
      if( seek_format != dest_format )
      {
	result = FALSE;
	GST_DEBUG_OBJECT( postcoh,
	    "seeking failed, unhandled seek format: %d", seek_format );
      }
      
      flush = (flags & GST_SEEK_FLAG_FLUSH) == GST_SEEK_FLAG_FLUSH;

      if( flush )
      {
	gst_pad_push_event( postcoh->srcpad, gst_event_new_flush_start() );
	gst_collect_pads_set_flushing( postcoh->collect, TRUE );

	GST_COLLECT_PADS_STREAM_LOCK(postcoh->collect);
	postcoh->flush_stop_pending = TRUE;
	GST_COLLECT_PADS_STREAM_UNLOCK(postcoh->collect);
	GST_DEBUG_OBJECT(postcoh, "mark pending flush stop event");
      }
      GST_DEBUG_OBJECT(postcoh, "handling seek event : %" GST_PTR_FORMAT, event);
      GST_COLLECT_PADS_STREAM_LOCK(postcoh->collect);

      if( postcoh->segment.stop != (guint64) -1 )
      {
	postcoh->segment.position = postcoh->segment.stop;
      }
      gst_segment_do_seek(&postcoh->segment, rate, seek_format, flags,
	  start_type, start, stop_type, stop, NULL);

      if( flush )
      {
	gst_collect_pads_set_flushing(postcoh->collect, TRUE);
      }
      GST_COLLECT_PADS_STREAM_UNLOCK(postcoh->collect);
      GST_DEBUG_OBJECT(postcoh, "forwarding seek event: %" GST_PTR_FORMAT, event);
      GST_DEBUG_OBJECT(postcoh, "updated segment: %" GST_SEGMENT_FORMAT, &postcoh->segment);

      g_atomic_int_set( &postcoh->new_segment_pending, TRUE);
      result = forward_src_event(postcoh, event, flush);

      if(!result)
      {
	GST_DEBUG_OBJECT(postcoh, "seeking failed");
      }
      if(g_atomic_int_compare_and_exchange(&postcoh->flush_stop_pending, TRUE, FALSE))
      {
	GST_DEBUG_OBJECT(postcoh, "pending flush stop");
      }
      if(!gst_pad_push_event(postcoh->srcpad, 
	    gst_event_new_flush_stop(TRUE)))
      {
	GST_WARNING_OBJECT(postcoh, "Sending flush stop event failed");
      }
    }
    /* events that can't be handled */
    case GST_EVENT_QOS:
      result = FALSE;
      gst_event_unref(event);
      break;
    case GST_EVENT_NAVIGATION:
      gst_event_unref(event);
      result = FALSE;
      break;

    /* forward the rest out all sink pads */
    default:
      GST_DEBUG_OBJECT(postcoh, "forward unhandled event %s", GST_EVENT_TYPE_NAME(event));
      result = forward_src_event(postcoh, event, FALSE);
      break;
  }
  
done:
  return result;
}


/* Updated copy from gstadder.c on 14 Nov 2017 by shinkee */
static gboolean
cuda_postcoh_sink_setcaps(CudaPostcoh *postcoh, GstPad *pad, GstCaps *caps)
{
  GstAudioInfo info;

  if( !gstlal_audio_info_from_caps(&info, caps) )
  {
    GST_DEBUG_OBJECT(caps, "Invalid format");
    return FALSE;
  }

  size_t freemem;
  size_t totalmem;

  GList *sinkpads;
  GstPostcohCollectData *data;
  
  PostcohState *state = postcoh->state;
  
  g_mutex_lock(&postcoh->prop_lock);
  while (state->npix == NOT_INIT || state->autochisq_len == NOT_INIT || postcoh->hist_trials ==NOT_INIT) 
  {
    g_cond_wait(&postcoh->prop_avail, &postcoh->prop_lock);
    GST_LOG_OBJECT(postcoh, "setcaps have to wait");
  }
  g_mutex_unlock(&postcoh->prop_lock);

  CUDA_CHECK(cudaSetDevice(postcoh->device_id));
  

  /* FIXME: this is copied from gstadder.c. Replace with new version of that file
   * if any. */
  GST_OBJECT_LOCK(postcoh);

  if( postcoh->current_caps != NULL )
  {
    if( gst_audio_info_is_equal(&info, &postcoh->info) )
    {
      gstlal_collect_pads_set_unit_size( pad, GST_AUDIO_INFO_BPF(&postcoh->info) );
      gstlal_collect_pads_set_rate( pad, GST_AUDIO_INFO_RATE(&postcoh->info) );
      GST_OBJECT_UNLOCK(postcoh);
    } else
    {
      GST_DEBUG_OBJECT( pad, "got input caps %" GST_PTR_FORMAT ", but "
	  "current caps are %" GST_PTR_FORMAT, caps, postcoh->current_caps );
      GST_OBJECT_UNLOCK(postcoh);
      gst_pad_push_event( pad, gst_event_new_reconfigure() );
    }
  } else
  {
    GST_INFO_OBJECT( pad, "setting caps to %" GST_PTR_FORMAT, caps );
    postcoh->current_caps = gst_caps_ref(caps);

    memcpy( &postcoh->info, &info, sizeof(info) );
    gstlal_collect_pads_set_unit_size(pad, GST_AUDIO_INFO_BPF(&postcoh->info));
    gstlal_collect_pads_set_rate(pad, GST_AUDIO_INFO_RATE(&postcoh->info));
    GST_OBJECT_UNLOCK(postcoh);

    GST_INFO_OBJECT(pad, "handle caps change to %" GST_PTR_FORMAT, caps);
  }

  //postcoh->bps = GST_AUDIO_INFO_BPS(&postcoh->info);
  postcoh->channels = GST_AUDIO_INFO_CHANNELS(&postcoh->info);
  postcoh->width = GST_AUDIO_INFO_WIDTH(&postcoh->info);
  postcoh->rate = GST_AUDIO_INFO_RATE(&postcoh->info);

  /* postcoh and state initialization */
  postcoh->bps = (postcoh->width/8) * postcoh->channels;	
  postcoh->offset_per_nanosecond = postcoh->bps / 1e9 * (postcoh->rate);

  GST_DEBUG("Debug bps %d, width %d, channels %d\n", postcoh->bps, postcoh->width, postcoh->channels);
  GST_DEBUG_OBJECT(postcoh, "setting GstPostcohCollectData offset_per_nanosecond %f and channels", postcoh->offset_per_nanosecond);

  state->nifo = GST_ELEMENT(postcoh)->numsinkpads;
  state->ifo_mapping = (gint8 *)malloc(sizeof(gint8) * state->nifo);
  state->peak_list = (PeakList **)malloc(sizeof(PeakList*) * state->nifo);
  state->dt = (float) 1/postcoh->rate;
  GST_DEBUG("Debug nifo %d peaklist size %d\n", state->nifo, sizeof(PeakList*));

  /* need to cover head and tail */
  postcoh->preserved_len = state->autochisq_len + 160; 
  postcoh->exe_len = postcoh->rate;

  state->exe_len = postcoh->rate;
  state->tmp_maxsnr = (float *) malloc(sizeof(float) * state->exe_len);
  state->tmp_tmpltidx = (int *) malloc(sizeof(int) * state->exe_len);
  state->max_npeak = postcoh->rate > postcoh->channels/2 ? postcoh->channels/2 : postcoh->rate;
  state->trial_sample_inv = round(postcoh->trial_interval * postcoh->rate);
  state->snglsnr_len = postcoh->preserved_len + postcoh->exe_len + postcoh->hist_trials * state->trial_sample_inv;
  state->hist_trials = postcoh->hist_trials;
  state->snglsnr_start_load = postcoh->hist_trials * state->trial_sample_inv;
  state->snglsnr_start_exe = state->snglsnr_start_load;

  GST_DEBUG_OBJECT(postcoh, "Debug hist_trials %d, autochisq_len %d, preserved_len %d, sngl_len %d, start_load %d, start_exe %d, max_npeak %d", 
    state->hist_trials, state->autochisq_len, postcoh->preserved_len, state->snglsnr_len, state->snglsnr_start_load, state->snglsnr_start_exe, state->max_npeak);

  state->ntmplt = postcoh->channels/2;
  cudaMemGetInfo(&freemem, &totalmem);
  GST_LOG( "Free memory: %d MB\nTotal memory: %d MB\n", (int)(freemem / 1024 / 1024), (int)(totalmem / 1024 / 1024) );
  GST_LOG( "Allocating %d B\n", (int) sizeof(COMPLEX_F *) * state->nifo );

  CUDA_CHECK(cudaMalloc((void **)&(state->dd_snglsnr), sizeof(COMPLEX_F *) * state->nifo));
  state->d_snglsnr = (COMPLEX_F **)malloc(sizeof(COMPLEX_F *) * state->nifo);

  gint8 i = 0, j = 0, cur_ifo = 0;

  GST_OBJECT_LOCK(postcoh->collect);

  /* initialize ifo_mapping, snglsnr matrix, and peak_list */
  for (sinkpads = GST_ELEMENT(postcoh)->sinkpads; sinkpads; sinkpads = g_list_next(sinkpads), i++) {
    GstPad *pad = GST_PAD(sinkpads->data);
    data = (GstPostcohCollectData *) gst_pad_get_element_private(pad);
    set_offset_per_nanosecond(data, postcoh->offset_per_nanosecond);
    set_channels(data, postcoh->channels);
  GST_DEBUG("Debug line no: %d nifo %d", __LINE__, state->nifo);
    for (j=0; j<state->nifo; j++) {
      GST_DEBUG("Debug line no: %d ifo %d", __LINE__, j);
      GST_DEBUG("Debug line no: %d IFO_MAP %c %c", __LINE__, IFO_MAP[0][0], IFO_MAP[0][1] );
      GST_DEBUG("Debug line no: %d ifo_name %u channels %d", __LINE__, &(data->ifo_name), postcoh->channels);
      GST_DEBUG("Debug line no: %d ifo_name full %s", __LINE__, data->ifo_name);
      if (strncmp(data->ifo_name, IFO_MAP[j], 2) == 0 ) {
	GST_DEBUG("Debug line no: %d", __LINE__);
	state->ifo_mapping[i] = j;
	cur_ifo = j;
	break;
      }
    }
  GST_DEBUG("Debug line no: %d", __LINE__);

    guint mem_alloc_size = state->snglsnr_len * postcoh->bps;
    //printf("device id %d, stream addr %p, alloc for snglsnr %d\n", postcoh->device_id, postcoh->stream, mem_alloc_size);
		
    cudaMemGetInfo(&freemem, &totalmem);
    GST_DEBUG( "Debug Free memory: %d MB\nTotal memory: %d MB\n", (int)(freemem / 1024 / 1024), (int)(totalmem / 1024 / 1024) );
    GST_DEBUG( "Debug Allocating %d MB\n", (int) (mem_alloc_size / 1024 / 1024) );

    CUDA_CHECK(cudaMalloc((void**) &(state->d_snglsnr[cur_ifo]), mem_alloc_size));
    CUDA_CHECK(cudaMemsetAsync(state->d_snglsnr[cur_ifo], 0, mem_alloc_size, postcoh->stream));
    CUDA_CHECK(cudaMemcpyAsync(&(state->dd_snglsnr[cur_ifo]), &(state->d_snglsnr[cur_ifo]), sizeof(COMPLEX_F *), cudaMemcpyHostToDevice, postcoh->stream));

    state->peak_list[cur_ifo] = create_peak_list(postcoh->state, postcoh->stream);
  }

  state->is_member_init = INIT;
  GST_OBJECT_UNLOCK(postcoh->collect);

  return TRUE;
}

static void destroy_notify(GstPostcohCollectData *data)
{
  GST_DEBUG("Debug line no: %d Destroy ifo_name", __LINE__);
	if (data) {
		free(data->ifo_name);
		if (data->adapter) {
			gst_audioadapter_clear(data->adapter);
			g_object_unref(data->adapter);
			data->adapter = NULL;
		}
	}

}

/*  */
static GstPad *cuda_postcoh_request_new_pad(GstElement *element, GstPadTemplate *templ, const gchar *name)
{
  CudaPostcoh* postcoh = CUDA_POSTCOH(element);
  
  GstPad* newpad;
  gint padcount;

  padcount = g_atomic_int_add(&postcoh->padcount, 1);

  //newpad = g_object_new(GST_TYPE_CUDA_POSTCOH_PAD, "name", newname, 
  //    "direction", templ->direction, "template", templ, NULL);
  newpad = gst_pad_new_from_template(templ, name);

  GST_DEBUG_OBJECT(postcoh, "Debug creating newpad %s", name);

  /* Obsolete function in 1.0 */
  //gst_pad_set_setcaps_function(GST_PAD(newpad), GST_DEBUG_FUNCPTR(cuda_postcoh_sink_setcaps));

  GstPostcohCollectData* data;
  data = (GstPostcohCollectData*) gst_collect_pads_add_pad(postcoh->collect, newpad, sizeof(GstPostcohCollectData), (GstCollectDataDestroyNotify) GST_DEBUG_FUNCPTR(destroy_notify), 0);
  //postcoh->collect_event = (GstPadEventFunction) GST_PAD_EVENTFUNC(newpad);
  gstlal_collect_pads_set_unit_size(newpad, GST_AUDIO_INFO_BPF(&postcoh->info));
  gstlal_collect_pads_set_rate(newpad, GST_AUDIO_INFO_RATE(&postcoh->info));
  
  //gst_pad_set_event_function(newpad, sink_event);

  if (!data || !gst_element_add_pad(element, newpad)) 
  {
    GST_DEBUG_OBJECT(postcoh, "Debug error in adding pad");
    gst_collect_pads_remove_pad(postcoh->collect, newpad);
    gst_object_unref(newpad);
    return NULL;
  }

  data->ifo_name = (gchar *)malloc(3*sizeof(gchar));
  data->ifo_name[2] = '\0';
  strncpy(data->ifo_name, name, 2*sizeof(gchar));
  //data->adapter = gst_adapter_new();
  data->adapter = g_object_new(GST_TYPE_AUDIOADAPTER, NULL);
  data->is_aligned = FALSE;
  data->aligned_offset0 = 0;
  data->next_offset = 0;
  GST_DEBUG_OBJECT(element, "Debug line no: %d new pad for %s is added and initialised ifo_name %u", __LINE__, data->ifo_name, &(data->ifo_name));

  return newpad; 
}



static void cuda_postcoh_release_pad(GstElement *element, GstPad *pad)
{
	CudaPostcoh* postcoh = CUDA_POSTCOH(element);

	gst_collect_pads_remove_pad(postcoh->collect, pad);
	gst_element_remove_pad(element, pad);
}


static GstStateChangeReturn cuda_postcoh_change_state(GstElement *element, GstStateChange transition)
{
	CudaPostcoh *postcoh = CUDA_POSTCOH(element);

	switch(transition) {
	case GST_STATE_CHANGE_NULL_TO_READY:
		break;

	case GST_STATE_CHANGE_READY_TO_PAUSED:
		gst_collect_pads_start(postcoh->collect);
		break;

	case GST_STATE_CHANGE_PAUSED_TO_PLAYING:
		break;

	case GST_STATE_CHANGE_PAUSED_TO_READY:
		/* need to unblock the collectpads before calling the
		 * parent change_state so that streaming can finish */
		gst_collect_pads_stop(postcoh->collect);
		break;

	default:
		break;
	}

	return GST_ELEMENT_CLASS(cuda_postcoh_parent_class)->change_state(element, transition);
}

static gboolean cuda_postcoh_get_latest_start_time(GstCollectPads *pads, GstClockTime *t_latest_start, guint64 *offset_latest_start)
{
	GSList *collectlist;
	GstPostcohCollectData *data;
	GstClockTime t_start_cur = GST_CLOCK_TIME_NONE;
	GstBuffer *buf;

	*t_latest_start = GST_CLOCK_TIME_NONE;

	/* invalid pads */
	g_return_val_if_fail(pads != NULL, FALSE);
	g_return_val_if_fail(GST_IS_COLLECT_PADS(pads), FALSE);

	for (collectlist = pads->data; collectlist; collectlist = g_slist_next(collectlist)) {
		data = collectlist->data;
		buf = gst_collect_pads_peek(pads, (GstCollectData *)data);
		/* eos */
		if(!buf) {
			GST_ERROR_OBJECT(pads, "%s pad:EOS", data->ifo_name);
			gst_buffer_unref(buf);
			return FALSE;
		}
		/* invalid offset */
		if(!GST_BUFFER_OFFSET_IS_VALID(buf) || !GST_BUFFER_OFFSET_END_IS_VALID(buf)) {
			GST_ERROR_OBJECT(pads, "%" GST_PTR_FORMAT ": %" GST_PTR_FORMAT " does not have valid offsets", ((GstCollectData *) data)->pad, buf);
			gst_buffer_unref(buf);
			return FALSE;
		}
		/* invalid timestamp */
		if(!GST_BUFFER_PTS_IS_VALID(buf) || !GST_BUFFER_DURATION_IS_VALID(buf)) {
			GST_ERROR_OBJECT(pads, "%" GST_PTR_FORMAT ": %" GST_PTR_FORMAT " does not have a valid timestamp and/or duration", ((GstCollectData *) data)->pad, buf);
			gst_buffer_unref(buf);
			return FALSE;
		}

		t_start_cur = GST_BUFFER_PTS(buf);

		if (*t_latest_start == GST_CLOCK_TIME_NONE) {
			*t_latest_start = t_start_cur;
			*offset_latest_start = GST_BUFFER_OFFSET(buf);
		} else {
			if (*t_latest_start < t_start_cur) {
				*t_latest_start = t_start_cur;
				*offset_latest_start = GST_BUFFER_OFFSET(buf);
			}
		}
		gst_buffer_unref(buf);

	}
	return TRUE;
}


static gboolean
cuda_postcoh_fillin_discont(GstCollectPads *pads, CudaPostcoh *postcoh)
{
	GSList *collectlist;
	GstPostcohCollectData *data;
	GstBuffer *buf = NULL;

	GstMapInfo mapinfo;

	/* invalid pads */
	g_return_val_if_fail(pads != NULL, FALSE);
	g_return_val_if_fail(GST_IS_COLLECT_PADS(pads), FALSE);


	for (collectlist = pads->data; collectlist; collectlist = g_slist_next(collectlist)) {
		data = collectlist->data;
		buf = gst_collect_pads_peek(pads, (GstCollectData *)data);

		if (buf != NULL) {
		/* if the buffer in the pad is behind what we expected,
		 * we span the gap using zero buffer.
		 */ 
		if (GST_BUFFER_OFFSET(buf) > data->next_offset) {
			printf("catch a gap\n");
			GST_DEBUG_OBJECT(data, "gap :data offset %" G_GUINT64_FORMAT "current next offset %" G_GUINT64_FORMAT, GST_BUFFER_OFFSET(buf), data->next_offset);
			GstBuffer *zerobuf = gst_buffer_new_allocate(NULL, (GST_BUFFER_OFFSET(buf)- data->next_offset) * postcoh->bps, NULL); 
			if(!zerobuf) {
				GST_DEBUG_OBJECT(data, "failure allocating zero-pad buffer");
			}

			gst_buffer_map(buf, &mapinfo, GST_MAP_WRITE);
			memset(mapinfo.data, 0, mapinfo.size);
			gst_audioadapter_push(data->adapter, zerobuf);
			gst_buffer_unmap(buf, &mapinfo);

		}
		((GstPostcohCollectData *)data)->next_offset = GST_BUFFER_OFFSET_END(buf);

		gst_buffer_unref(buf);
		}
	}
	return TRUE;
}

static gint cuda_postcoh_push_and_get_common_size(GstCollectPads *pads, CudaPostcoh *postcoh)
{

	/* first fill in any discontinuity */
	cuda_postcoh_fillin_discont(pads, postcoh);

	PostcohState *state = postcoh->state;
	GSList *collectlist;
	GstPostcohCollectData *data;
	GstBuffer *buf = NULL;

	gint i = 0, min_size = 0, size_cur;
	gboolean min_size_init = FALSE;
	state->cur_nifo = 0;

	/* The logic to find common size:
	 * if one detector has no data, we obtain the data size in the adapter
	 * and find the common size of this detector with other detectors who have
	 * data. if there is no data in this adapter, the common size is determined
	 * by other detectors.
	 */
	for (i=0, collectlist = pads->data; collectlist; collectlist = g_slist_next(collectlist), i++) {
			data = collectlist->data;
			buf = gst_collect_pads_pop(pads, (GstCollectData *)data);
			if (buf == NULL) {
				size_cur = adapter_available_samples(data->adapter);
				if(!min_size_init) {
					min_size = size_cur;
					min_size_init = size_cur > 0 ? TRUE : FALSE;
				} else {
					min_size = min_size > size_cur ? size_cur : min_size;
				}

				continue;
			}

			gst_audioadapter_push(data->adapter, buf);
			size_cur = adapter_available_samples(data->adapter);
			if(!min_size_init) {
				min_size = size_cur;
				min_size_init = TRUE;
			} else {
				min_size = min_size > size_cur ? size_cur : min_size;
			}
			strncpy(state->cur_ifos + 2*state->cur_nifo, data->ifo_name, 2*sizeof(char));
			state->cur_nifo++;

	}
	/* If all pads returns NULL buffers, this means all pads at EOS,
	 * we flag min_size as -1 to indicate we need to send an EOS event */ 
	if (state->cur_nifo == 0)
		min_size = -1;

	GST_LOG_OBJECT(postcoh, "get common size %d", min_size);

	return min_size;
}

static gboolean cuda_postcoh_align_collected(GstCollectPads *pads, CudaPostcoh *postcoh)
{

	GSList *collectlist;
	GstPostcohCollectData *data;
	GstBuffer *buf, *subbuf;
	GstClockTime t_start_cur, t_end_cur;
	gboolean all_aligned = TRUE;
	guint64 offset_cur, offset_end_cur, buf_aligned_offset0;
	GstClockTime t0 = postcoh->t0;

	GST_DEBUG_OBJECT(pads, "begin to align offset0");

	for (collectlist = pads->data; collectlist; collectlist = g_slist_next(collectlist)) {
		data = collectlist->data;
		GST_DEBUG_OBJECT(pads, "Debug line no %d now at %s is aligned %d", __LINE__, data->ifo_name, data->is_aligned);
		if (data->is_aligned) {
			/* do not collect the buffer in this pad. wait for other pads to be aligned */
			//buf = gst_collect_pads_pop(pads, (GstCollectData *)data);
			//gst_adapter_push(data->adapter, buf);
			continue;
		}
		buf = gst_collect_pads_pop(pads, (GstCollectData *)data);
		
		t_start_cur = GST_BUFFER_PTS(buf);
		t_end_cur = t_start_cur + GST_BUFFER_DURATION(buf);
		offset_cur = GST_BUFFER_OFFSET(buf);
		offset_end_cur = GST_BUFFER_OFFSET_END(buf);
  GST_DEBUG("Debug line no: %d buffer duration %d", __LINE__, GST_BUFFER_DURATION(buf) );
  GST_DEBUG("Debug line no: %d buffer size %d", __LINE__, gst_buffer_get_size(buf) );
		if (t_end_cur > t0) {
		  if( GST_BUFFER_PTS_IS_VALID(buf) && GST_BUFFER_DURATION_IS_VALID(buf) && GST_BUFFER_OFFSET_IS_VALID(buf) && GST_BUFFER_OFFSET_END_IS_VALID(buf) )
		  {
		    GST_DEBUG("Debug line no: %d all valid", __LINE__);
		    GST_DEBUG("Duration %" GST_TIME_FORMAT, GST_TIME_ARGS(GST_BUFFER_DURATION(buf)) );
		  }
		  buf_aligned_offset0 = (gint) (postcoh->offset0 - offset_cur);
		  GST_DEBUG_OBJECT(data, "buffer aligned offset0 %u offsetcur %u offset end %u", (gsize)buf_aligned_offset0, offset_cur, offset_end_cur);
		  GST_DEBUG("Debug line no: %d subbuf size %d channel %d", __LINE__, (gsize) (offset_end_cur - offset_cur - buf_aligned_offset0), data->channels );
		  GST_DEBUG("Data channels %d", data->channels);
		  subbuf = gst_buffer_copy_region(buf, GST_BUFFER_COPY_FLAGS | GST_BUFFER_COPY_MEMORY | GST_BUFFER_COPY_TIMESTAMPS, (gsize) buf_aligned_offset0, (gsize) (offset_end_cur - offset_cur - buf_aligned_offset0) * data->channels * sizeof(float));
		  if( GST_BUFFER_PTS_IS_VALID(subbuf) && GST_BUFFER_DURATION_IS_VALID(subbuf) && GST_BUFFER_OFFSET_IS_VALID(subbuf) && GST_BUFFER_OFFSET_END_IS_VALID(subbuf) )
		  {
		    GST_DEBUG("Debug line no: %d all valid", __LINE__);
		    GST_DEBUG("Duration %" GST_TIME_FORMAT, GST_TIME_ARGS(GST_BUFFER_DURATION(subbuf)));
		  }
  GST_DEBUG("Debug line no: %d", __LINE__);
			GST_DEBUG_OBJECT (pads,
				"Created sub buffer of (%ld bytes) with timestamp %" GST_TIME_FORMAT ", duration %"
				GST_TIME_FORMAT ", offset %" G_GUINT64_FORMAT ", offset_end %"
				G_GUINT64_FORMAT,  gst_buffer_get_size(subbuf),
				GST_TIME_ARGS (GST_BUFFER_PTS(subbuf)),
				GST_TIME_ARGS (GST_BUFFER_DURATION (subbuf)),
				GST_BUFFER_OFFSET (subbuf), GST_BUFFER_OFFSET_END (subbuf));
  GST_DEBUG("Debug line no: %d", __LINE__);
			gst_audioadapter_push(data->adapter, subbuf);
  GST_DEBUG("Debug line no: %d", __LINE__);
			data->is_aligned = TRUE;
			set_aligned_offset0(data, buf_aligned_offset0 + offset_cur);
			/* from the first buffer in the adapter, we initiate the next offset */
			data->next_offset = GST_BUFFER_OFFSET_END(buf);
			/* discard this buffer in collectpads so it can collect new one */
			gst_buffer_unref(buf);
		} else {
			all_aligned = FALSE;
			/* discard this buffer in collectpads so it can collect new one */
			gst_buffer_unref(buf);

		}
	}

	return all_aligned;
		
	

}

#if 0
/* Unused */
static void cuda_postcoh_flush(GstCollectPads *pads, guint64 common_size)
{
	GSList *collectlist;
	GstPostcohCollectData *data;

	for (collectlist = pads->data; collectlist; collectlist = g_slist_next(collectlist)) {
		data = collectlist->data;
		gst_audioadapter_flush_samples(data->adapter, common_size);
	}

}
#endif

static gboolean is_cur_ifo_has_data(PostcohState *state, gint cur_ifo)
{
	for (int i=0; i<state->cur_nifo; i++) {
		if (strncmp(state->cur_ifos+2*i, IFO_MAP[cur_ifo], 2) == 0)
			return TRUE;
	}
	return FALSE;

}

static int cuda_postcoh_pklist_mark_invalid_background(PeakList *pklist,int iifo, int hist_trials, int max_npeak, float cohsnr_thresh)
{
	int ipeak, npeak, peak_cur, itrial, background_cur, left_backgrounds = 0;
	npeak = pklist->npeak[0];
	for (ipeak=0; ipeak<npeak; ipeak++) {
		peak_cur = pklist->peak_pos[ipeak];
		for (itrial=1; itrial<=hist_trials; itrial++) {
			background_cur = (itrial - 1)*max_npeak + peak_cur;
//			if (sqrt(pklist->cohsnr_bg[background_cur]) > cohsnr_thresh * pklist->snglsnr_L[iifo*max_npeak + peak_cur])
			if (sqrt(pklist->cohsnr_bg[background_cur]) > 1.414 + pklist->snglsnr_L[iifo*max_npeak + peak_cur]) {
				left_backgrounds++;
				GST_LOG("mark back,%d ipeak, %d itrial", ipeak, itrial);
			}
			else
				pklist->cohsnr_bg[background_cur] = -1;
		}
	}
	return left_backgrounds;
}

static int cuda_postcoh_rm_invalid_peak(PostcohState *state, float cohsnr_thresh)
{
	int iifo, ipeak, npeak, nifo = state->nifo,  cluster_peak_pos[state->max_npeak], bubbled_peak_pos[state->max_npeak], peak_cur;
	int final_peaks = 0, bubbled_peaks = 0;
	PeakList *pklist; 
	int *peak_pos;
	int left_entries = 0;

	for(iifo=0; iifo<nifo && is_cur_ifo_has_data(state, iifo); iifo++) {
		final_peaks = 0;
		bubbled_peaks = 0;
		pklist= state->peak_list[iifo];
		npeak = pklist->npeak[0];
		peak_pos = pklist->peak_pos;

		/*
		 * select background that satisfy the criteria: cohsnr > triggersnr + coh_thresh
		 */
		if (npeak > 0 && state->cur_nifo == state->nifo)
			left_entries += cuda_postcoh_pklist_mark_invalid_background(pklist, iifo, state->hist_trials, state->max_npeak, cohsnr_thresh);
	
		/*
		 * mark the rest of peak positions to be -1 to identify invalid background
		 */

		for(ipeak=0; ipeak<state->max_npeak; ipeak++)
		   cluster_peak_pos[ipeak] = -1;
		memcpy(cluster_peak_pos, peak_pos, sizeof(int) * npeak);

		/*
		 * select zerolag that satisfy the criteria: cohsnr > triggersnr + coh_thresh
		 */
		for(ipeak=0; ipeak<npeak; ipeak++) {
			/* if the difference of maximum single snr and coherent snr is ignorable,
			 * it means that only one detector is in action,
			 * we abandon this peak
			 * */
			peak_cur = peak_pos[ipeak];
			if (sqrt(pklist->cohsnr[peak_cur]) > 1.414 + pklist->snglsnr_L[iifo*(state->max_npeak) + peak_cur]) {
				cluster_peak_pos[final_peaks++] = peak_cur;
			} else 
				bubbled_peak_pos[bubbled_peaks++] = peak_cur;

		}

		/*
		 * bubble out the rest peaks
		 */
		for(ipeak=final_peaks; ipeak<npeak; ipeak++)
			cluster_peak_pos[ipeak] = bubbled_peak_pos[ipeak - final_peaks];


		npeak = final_peaks;
		memcpy(peak_pos, cluster_peak_pos, sizeof(int) * state->max_npeak);
		pklist->npeak[0] = npeak;

		GST_DEBUG("ifo %d, back entries %d, npeak %d", iifo, left_entries, npeak);
		/* mark background triggers which do not pass the test */
		left_entries += npeak;
	}
	return left_entries;

}

static void cuda_postcoh_write_table_to_buf(CudaPostcoh *postcoh, GstBuffer *outbuf)
{
	  
	PostcohState *state = postcoh->state;

	GstMapInfo mapinfo;
	PostcohInspiralTable *output;
	
	int iifo = 0, nifo = state->nifo;
	int ifos_size = sizeof(char) * 2 * state->cur_nifo, one_ifo_size = sizeof(char) * 2 ;
	int ipeak, npeak = 0, itrial = 0, exe_len = state->exe_len, max_npeak = state->max_npeak;
	int hist_trials = postcoh->hist_trials;

	int tmplt_idx;
	int write_entries = 0;
	
	GstClockTime ts = GST_BUFFER_PTS(outbuf);

	int livetime = (int) ((ts - postcoh->t0)/GST_SECOND), cur_tmplt_idx;

	gst_buffer_map(outbuf, &mapinfo, GST_MAP_WRITE);
	output = (PostcohInspiralTable *)mapinfo.data;
	SnglInspiralTable *sngl_table = postcoh->sngl_table;

	for(iifo=0; iifo<nifo && is_cur_ifo_has_data(state, iifo); iifo++)
	{
		PeakList *pklist = state->peak_list[iifo];
		npeak = pklist->npeak[0];
		LIGOTimeGPS end_time;

		int peak_cur, len_cur, peak_cur_bg;
		for(ipeak=0; ipeak<npeak; ipeak++) {
			XLALINT8NSToGPS(&end_time, ts);
			int *peak_pos = pklist->peak_pos;
			peak_cur = peak_pos[ipeak];
			len_cur = pklist->len_idx[peak_cur];
			XLALGPSAdd(&(end_time), (double) len_cur/exe_len);
			output->end_time = end_time;
			XLALGPSAdd(&(end_time), (double) pklist->ntoff_L[peak_cur]/exe_len);
			output->end_time_L = end_time;
			XLALGPSAdd(&(end_time), (double) (pklist->ntoff_H[peak_cur] - pklist->ntoff_L[peak_cur])/exe_len);
			output->end_time_H = end_time;
			XLALGPSAdd(&(end_time), (double) (pklist->ntoff_V[peak_cur] - pklist->ntoff_H[peak_cur])/exe_len);
			output->end_time_V = end_time;
			output->snglsnr_L = pklist->snglsnr_L[peak_cur];
			output->snglsnr_H = pklist->snglsnr_H[peak_cur];
			output->snglsnr_V = pklist->snglsnr_V[peak_cur];
			output->coaphase_L = pklist->coaphase_L[peak_cur];
			output->coaphase_H = pklist->coaphase_H[peak_cur];
			output->coaphase_V = pklist->coaphase_V[peak_cur];
			output->chisq_L = pklist->chisq_L[peak_cur];
			output->chisq_H = pklist->chisq_H[peak_cur];
			output->chisq_V = pklist->chisq_V[peak_cur];
			cur_tmplt_idx = pklist->tmplt_idx[peak_cur];

			output->deff_L = sqrt(state->sigmasq[L_MAPPING][cur_tmplt_idx])/ pklist->snglsnr_L[peak_cur]; 
			output->deff_H = sqrt(state->sigmasq[H_MAPPING][cur_tmplt_idx])/ pklist->snglsnr_H[peak_cur]; 
			if (pklist->snglsnr_V[peak_cur] > 0)
				output->deff_V = sqrt(state->sigmasq[V_MAPPING][cur_tmplt_idx])/ pklist->snglsnr_V[peak_cur]; 
			output->is_background = 0;
			output->livetime = livetime;
			strncpy(output->ifos, state->cur_ifos, ifos_size);
			output->ifos[2*state->cur_nifo] = '\0';
		       	strncpy(output->pivotal_ifo, IFO_MAP[iifo], one_ifo_size);
			output->pivotal_ifo[2] = '\0';
			output->tmplt_idx = cur_tmplt_idx;
			output->pix_idx = pklist->pix_idx[peak_cur];
			output->cohsnr = sqrt(pklist->cohsnr[peak_cur]); /* the returned snr from cuda kernel is snr^2 */
			output->nullsnr = pklist->nullsnr[peak_cur];
			output->cmbchisq = pklist->cmbchisq[peak_cur];
			output->spearman_pval = 0;
			output->fap = 0;
			output->far = 0;
			/* covert template index to mass values */
			tmplt_idx = output->tmplt_idx;
			output->template_duration = sngl_table[tmplt_idx].template_duration;
			output->mchirp = sngl_table[tmplt_idx].mchirp;
			output->mtotal = sngl_table[tmplt_idx].mtotal;
			output->mass1 = sngl_table[tmplt_idx].mass1;
			output->mass2 = sngl_table[tmplt_idx].mass2;
			output->spin1x = sngl_table[tmplt_idx].spin1x;
			output->spin1y = sngl_table[tmplt_idx].spin1y;
			output->spin1z = sngl_table[tmplt_idx].spin1z;
			output->spin2x = sngl_table[tmplt_idx].spin2x;
			output->spin2y = sngl_table[tmplt_idx].spin2y;
			output->spin2z = sngl_table[tmplt_idx].spin2z;
			output->eta = sngl_table[tmplt_idx].eta;
			/* convert pixel index to ra and dec */
			double theta, phi;
			/* ra = phi, dec = pi/2 - theta */	
			pix2ang_nest(postcoh->state->nside, output->pix_idx, &theta, &phi);
	
			output->ra = phi*RAD2DEG;
			output->dec = (M_PI_2 - theta)*RAD2DEG;
			//output->event_id = postcoh->cur_event_id++;
			if (postcoh->output_skymap && state->snglsnr_max > MIN_OUTPUT_SKYMAP_SNR) {
				GString *filename = NULL;
				FILE *file = NULL;
				filename = g_string_new(output->ifos);
				g_string_append_printf(filename, "_skymap/%s_%d_%d", output->pivotal_ifo, output->end_time_L.gpsSeconds, output->end_time_L.gpsNanoSeconds);
				g_string_append_printf(filename, "_%d", output->tmplt_idx);
				strcpy(output->skymap_fname, filename->str);
				printf("file %s is written, skymap addr %p\n", output->skymap_fname, &(pklist->cohsnr_skymap[ipeak * state->npix]));
				file = fopen(output->skymap_fname, "w");
				fwrite(&(pklist->cohsnr_skymap[ipeak * state->npix]), sizeof(float), state->npix, file);
				fclose(file);
				file = NULL;
				g_string_free(filename, TRUE);
			} else
				output->skymap_fname[0] = '\0';

			GST_LOG_OBJECT(postcoh, "end_time_L %d, ipeak %d, peak_cur %d, len_cur %d, tmplt_idx %d, pix_idx %d \t,"
				"snglsnr_L %f, snglsnr_H %f, snglsnr_V %f,"
			     "coaphase_L %f, coaphase_H %f, coa_phase_V %f,"
			     "chisq_L %f, chisq_H %f, chisq_V %f,"
			     "cohsnr %f, nullsnr %f, cmbchisq %f\n",
			     output->end_time_L.gpsSeconds, ipeak, peak_cur, len_cur, output->tmplt_idx, output->pix_idx,
			     output->snglsnr_L, output->snglsnr_H, output->snglsnr_V,
			     output->coaphase_L, output->coaphase_H, output->coaphase_V,
			     output->chisq_L, output->chisq_H, output->chisq_V,
			     output->cohsnr, output->nullsnr, output->cmbchisq
			      );
			output++;
			write_entries++;
		}

		if (pklist->d_cohsnr_skymap) {
			cudaFree(pklist->d_cohsnr_skymap);
			pklist->d_cohsnr_skymap = NULL;
		}

		if (pklist->cohsnr_skymap) {
			cudaFreeHost(pklist->cohsnr_skymap);
			pklist->cohsnr_skymap = NULL;
		}

		if (state->cur_nifo == state->nifo) {
			int *peak_pos = pklist->peak_pos;
			for(ipeak=0; ipeak<state->max_npeak; ipeak++) {
				for(itrial=1; itrial<=hist_trials; itrial++) {
					peak_cur = peak_pos[ipeak];
					len_cur = pklist->len_idx[peak_cur];
					/* check if cohsnr pass the valid test */
					peak_cur_bg = (itrial - 1)*max_npeak + peak_cur;
					if (peak_cur >= 0 && pklist->cohsnr_bg[peak_cur_bg] > 0) {
					//output->end_time = end_time[ipeak];
					output->is_background = 1;
					output->livetime = livetime;
					strncpy(output->ifos, state->cur_ifos, ifos_size);
					output->ifos[2*state->cur_nifo] = '\0';
			       		strncpy(output->pivotal_ifo, IFO_MAP[iifo], one_ifo_size);
					output->pivotal_ifo[2] = '\0';
					output->tmplt_idx = pklist->tmplt_idx[peak_cur];
					output->snglsnr_L = pklist->snglsnr_bg_L[peak_cur_bg];
					output->snglsnr_H = pklist->snglsnr_bg_H[peak_cur_bg];
					output->snglsnr_V = pklist->snglsnr_bg_V[peak_cur_bg];
					output->coaphase_L = pklist->coaphase_bg_L[peak_cur_bg];
					output->coaphase_H = pklist->coaphase_bg_H[peak_cur_bg];
					output->coaphase_V = pklist->coaphase_bg_V[peak_cur_bg];
					output->chisq_L = pklist->chisq_bg_L[peak_cur_bg];
					output->chisq_H = pklist->chisq_bg_H[peak_cur_bg];
					output->chisq_V = pklist->chisq_bg_V[peak_cur_bg];
	
					//output->pix_idx = pklist->pix_idx[itrial*max_npeak + peak_cur];
					output->cohsnr = sqrt(pklist->cohsnr_bg[peak_cur_bg]);
					output->nullsnr = pklist->nullsnr_bg[peak_cur_bg];
					output->cmbchisq = pklist->cmbchisq_bg[peak_cur_bg];
					output->spearman_pval = 0;
					output->fap = 0;
					output->far = 0;
					output->skymap_fname[0] ='\0';
					GST_LOG_OBJECT(postcoh, "ipeak %d, itrial %d, len_cur %d, tmplt_idx %d, pix_idx %d,"
					"snglsnr_L %f, snglsnr_H %f, snglsnr_V %f,"
				     "coaphase_L %f, coaphase_H %f, coa_phase_V %f,"
				     "chisq_L %f, chisq_H %f, chisq_V %f,"
				     "cohsnr %f, nullsnr %f, cmbchisq %f\n",
				     ipeak, itrial, len_cur, output->tmplt_idx, output->pix_idx,
				     output->snglsnr_L, output->snglsnr_H, output->snglsnr_V,
				     output->coaphase_L, output->coaphase_H, output->coaphase_V,
				     output->chisq_L, output->chisq_H, output->chisq_V,
				     output->cohsnr, output->nullsnr, output->cmbchisq
				      );
	
					output++;
					write_entries++;
					}
				}
			}
		}
		GST_LOG_OBJECT(postcoh, "write to output, ifo %d, npeak %d, %d total entries", iifo, npeak, write_entries);
	}
	gst_buffer_unmap(outbuf, &mapinfo);
}

static GstBuffer* cuda_postcoh_new_buffer(CudaPostcoh *postcoh, gint out_len)
{
	GstMapInfo mapinfo;
	GstBuffer *outbuf = NULL;
	GstPad *srcpad = postcoh->srcpad;
	//GstCaps *caps = gst_pad_get_current_caps(srcpad);
	GstFlowReturn ret;
	PostcohState *state = postcoh->state;
	int left_entries = 0;

	left_entries = cuda_postcoh_rm_invalid_peak(state, postcoh->cohsnr_thresh);
	int out_size = sizeof(PostcohInspiralTable) * left_entries ;

	outbuf = gst_buffer_new_allocate(NULL, out_size, NULL);

	ret = gst_pad_push(srcpad, outbuf);

	if (ret != GST_FLOW_OK) {
		GST_ERROR_OBJECT(srcpad, "Could not allocate postcoh-inspiral buffer %d", ret);
		return NULL;
	}

	gst_buffer_map(outbuf, &mapinfo, GST_MAP_WRITE);
	memset(mapinfo.data, 0, out_size);
	gst_buffer_unmap(outbuf, &mapinfo);

        /* set the time stamps */
	GstClockTime ts = postcoh->t0 + gst_util_uint64_scale_int_round(postcoh->samples_out, GST_SECOND,
		       	postcoh->rate);

        GST_BUFFER_PTS(outbuf) = ts;
	GST_BUFFER_DURATION(outbuf) = (GstClockTime) gst_util_uint64_scale_int_round(GST_SECOND, out_len, postcoh->rate);

	/* set the offset */
        GST_BUFFER_OFFSET(outbuf) = postcoh->offset0 + postcoh->samples_out;
        GST_BUFFER_OFFSET_END(outbuf) = GST_BUFFER_OFFSET(outbuf) + out_len;

	gst_buffer_set_size(outbuf, out_size);

	cuda_postcoh_write_table_to_buf(postcoh, outbuf);

	GST_LOG_OBJECT (srcpad,
		"Processed of (%d entries) with timestamp %" GST_TIME_FORMAT ", duration %"
		GST_TIME_FORMAT ", offset %" G_GUINT64_FORMAT ", offset_end %"
		G_GUINT64_FORMAT,  left_entries,
		GST_TIME_ARGS (GST_BUFFER_PTS(outbuf)),
		GST_TIME_ARGS (GST_BUFFER_DURATION (outbuf)),
		GST_BUFFER_OFFSET (outbuf), GST_BUFFER_OFFSET_END (outbuf));

	return outbuf;
}

int timestamp_to_gps_idx(long gps_start, int gps_step, GstClockTime t)
{
	int seconds_in_one_day = 24 * 3600;
	int gps_len = seconds_in_one_day / gps_step;

	/* DEPRECATED: not using utc0 as the gps_start time any more */
	//unsigned long days_from_gps_start = (t / GST_SECOND) / seconds_in_one_day;
	//double time_in_one_day = (double) (t/GST_SECOND) - days_from_gps_start * seconds_in_one_day;
	//int gps_idx = (int) (round( time_in_one_day / gps_step)) % gps_len;

	int days_from_gps_start = floor((t / GST_SECOND - (double)gps_start) / seconds_in_one_day);
	double time_in_one_day = (double) (t/GST_SECOND) -  ((double)gps_start + days_from_gps_start * seconds_in_one_day);
	int gps_idx = (int) (round( time_in_one_day / gps_step)) % gps_len;

	GST_DEBUG("current days from gps_start %d, current time in one day %f, length of gps array %d, gps_idx %d,\n", days_from_gps_start, time_in_one_day, gps_len, gps_idx);
	return gps_idx;
}

static int peaks_over_thresh(COMPLEX_F *snglsnr, PostcohState *state, int cur_ifo, cudaStream_t stream)
{
        state->snglsnr_max = 0;
	int exe_len = state->exe_len, ntmplt = state->ntmplt, itmplt, ilen, jlen, npeak = 0;
	COMPLEX_F *isnr = snglsnr;
	float tmp_abssnr, tmp_tmplt, snglsnr_thresh = state->snglsnr_thresh;
	PeakList *pklist = state->peak_list[cur_ifo];
	float *tmp_maxsnr = state->tmp_maxsnr;
	int *tmp_tmpltidx = state->tmp_tmpltidx;
	int *peak_pos = pklist->peak_pos;
	int *tmplt_idx = pklist->tmplt_idx;
	int *len_idx = pklist->len_idx;
	/* find maxsnr for each sampling point, keep the record of the tmplt_idx */
	for (ilen=0; ilen<exe_len; ilen++) {
		tmp_maxsnr[ilen] = 0.0;
		tmp_tmpltidx[ilen] = -1;
		for (itmplt=0; itmplt<ntmplt; itmplt++) {
			tmp_abssnr = sqrt((*isnr).re * (*isnr).re + (*isnr).im * (*isnr).im);
			if (tmp_abssnr > tmp_maxsnr[ilen]) {
				tmp_maxsnr[ilen] = tmp_abssnr;
				tmp_tmpltidx[ilen] = itmplt;
			}
			isnr++;
		}
	}
	/* find the maxsnr acrros each tmplt */
	for (ilen=0; ilen<exe_len; ilen++) {
		if (tmp_tmpltidx[ilen] > -1) {
			/* find if the subsequential snr has larger snr, 
			 * yes: continue to next sample point, 
			 * no: save this snr and delete snr on other times of the same tmplt*/
			for (jlen=ilen+1; jlen<exe_len; jlen++) {
				if (tmp_tmpltidx[jlen] == tmp_tmpltidx[ilen] && tmp_maxsnr[jlen] > tmp_maxsnr[ilen])
					break;
				if (tmp_tmpltidx[jlen] == tmp_tmpltidx[ilen] && tmp_maxsnr[jlen] < tmp_maxsnr[ilen]) 
					tmp_tmpltidx[jlen] = -1;
			}

			if(jlen == exe_len && tmp_maxsnr[ilen] > snglsnr_thresh) {
				len_idx[npeak] = ilen;
				tmplt_idx[npeak] = tmp_tmpltidx[ilen];
				peak_pos[npeak] = npeak;
				npeak++;
			}
		}
	}

        /* keep track of the maximum single snr in this snr chunk */
	for (ilen=0; ilen<exe_len; ilen++) {
	  if (tmp_maxsnr[ilen] > state->snglsnr_max)
	    state->snglsnr_max = tmp_maxsnr[ilen];
	}

	/* do clustering every PEAKFINDER_CLUSTER_WINDOW samples, FIXME: if set to 0, the size of output will be ten times */
	int cluster_peak_pos[state->max_npeak], len_cluster_peak, len_next_peak, final_peaks=0, ipeak;
	cluster_peak_pos[0] = peak_pos[0];
	for(ipeak=0; ipeak<npeak-1; ipeak++) {
		if (peak_pos[ipeak+1] - cluster_peak_pos[final_peaks] > PEAKFINDER_CLUSTER_WINDOW) {
			final_peaks++;
			cluster_peak_pos[final_peaks] = peak_pos[ipeak+1];
		}
		else { // update the cluster_peak_pos if next peak pos has larger SNR
			len_cluster_peak = len_idx[cluster_peak_pos[final_peaks]];
			len_next_peak = len_idx[peak_pos[ipeak+1]];
			if (tmp_maxsnr[len_cluster_peak] < tmp_maxsnr[len_next_peak])
				cluster_peak_pos[final_peaks] = peak_pos[ipeak+1];
		}
	}

	npeak = npeak == 0 ? 0 : final_peaks + 1;
	memcpy(peak_pos, cluster_peak_pos, sizeof(int) * npeak);
	pklist->npeak[0] = npeak;

	//printf("peaks_over_thresh , ifo %d, npeak %d\n", cur_ifo, npeak);
	CUDA_CHECK(cudaMemcpyAsync(	pklist->d_npeak, 
			pklist->npeak, 
			sizeof(int) * (pklist->peak_intlen), 
			cudaMemcpyHostToDevice,
			stream));

#if 0
	CUDA_CHECK(cudaMemcpyAsync(	pklist->d_maxsnglsnr, 
			pklist->maxsnglsnr, 
			sizeof(float) * (pklist->peak_floatlen), 
			cudaMemcpyHostToDevice,
			stream));
#endif
	return npeak;
}

static void cuda_postcoh_process(GstCollectPads *pads, gint common_size, gint one_take_size, gint exe_size, CudaPostcoh *postcoh)
{
	GSList *collectlist;
	GstPostcohCollectData *data;
	COMPLEX_F *snglsnr;
	//*pos_dd_snglsnr, *pos_in_snglsnr;
	gint one_take_len = one_take_size / postcoh->bps;

	int i = 0, cur_ifo = 0;
	PostcohState *state = postcoh->state;

	GstFlowReturn ret;

	int c_npeak;
	GstClockTime ts = postcoh->t0 + gst_util_uint64_scale_int_round(postcoh->samples_out, GST_SECOND,
		       	postcoh->rate);

	LIGOTimeGPS ligo_time;
	XLALINT8NSToGPS(&ligo_time, ts);
	while (common_size >= one_take_size) {
		int gps_idx = timestamp_to_gps_idx(state->gps_start, state->gps_step, postcoh->next_exe_t);
		/* copy the snr data to the right location for all detectors */ 
		for (i=0, collectlist = pads->data; collectlist; collectlist = g_slist_next(collectlist), i++) {
			data = collectlist->data;
			cur_ifo = state->ifo_mapping[i];
			PeakList *pklist = state->peak_list[cur_ifo];

			if (is_cur_ifo_has_data(state, cur_ifo)) {
#ifdef ACCELERATE_POSTCOH_MEMORY_COPY
			// temporal solution for low memory copy speed
			snglsnr = (COMPLEX_F *) gst_adapter_peek_cuda(data->adapter, one_take_size);
#else 
			GST_DEBUG("Debug postcoh wrong compilation");
			//snglsnr = (COMPLEX_F *) gst_adapter_peek(data->adapter, one_take_size);
#endif
//			printf("auto_len %d, npix %d\n", state->autochisq_len, state->npix);
			c_npeak = peaks_over_thresh(snglsnr, state, cur_ifo, postcoh->stream);

			GST_LOG_OBJECT(postcoh, "gps %d, ifo %d, c_npeak %d, max_snglsnr %f\n", ligo_time.gpsSeconds, cur_ifo, c_npeak, state->snglsnr_max);

			/* 
			// because you use new postcoh kernel optimized by Xiaoyang Guo now, you cannot use 
			// the following memory copy code any more
			// new postcoh kernel need transposed state->d_snglsnr[cur_ifo] matrix and there is no transpose in the following code 
			pos_dd_snglsnr = state->d_snglsnr[cur_ifo] + state->snglsnr_start_load * state->ntmplt;
			// copy the snglsnr to the right cuda memory
			if(state->snglsnr_start_load + one_take_len <= state->snglsnr_len){
				// when the snglsnr can be put in as one chunk 
				CUDA_CHECK(cudaMemcpyAsync(pos_dd_snglsnr, snglsnr, one_take_size, cudaMemcpyHostToDevice, postcoh->stream));
				GST_LOG("load snr to gpu as a chunk");
			} else {

				int tail_cpy_size = (state->snglsnr_len - state->snglsnr_start_load) * postcoh->bps;
				CUDA_CHECK(cudaMemcpyAsync(pos_dd_snglsnr, snglsnr, tail_cpy_size, cudaMemcpyHostToDevice, postcoh->stream));
				int head_cpy_size = one_take_size - tail_cpy_size;
				pos_dd_snglsnr = state->d_snglsnr[cur_ifo];
				pos_in_snglsnr = snglsnr + (state->snglsnr_len - state->snglsnr_start_load) * state->ntmplt;
				CUDA_CHECK(cudaMemcpyAsync(pos_dd_snglsnr, pos_in_snglsnr, head_cpy_size, cudaMemcpyHostToDevice, postcoh->stream));
				GST_LOG("load snr to gpu as as two chunks");
			}
			*/			

			// this is necessory for new postcoh kernel
			// 1. expand temporal memory space if necessary
			g_assert(pklist->len_snglsnr_buffer >= 0);
			if (one_take_size > pklist->len_snglsnr_buffer) {
				// re-malloc pklist->d_snglsnr_buffer
				if (pklist->d_snglsnr_buffer != NULL) {
					cudaFree(pklist->d_snglsnr_buffer);
				}
				cudaMalloc((void**)&pklist->d_snglsnr_buffer, one_take_size);
				pklist->len_snglsnr_buffer = one_take_size;
			}
			// 2. copy snglsnr to temporal gpu memory d_snglsnr_buffer
			CUDA_CHECK(cudaMemcpyAsync(pklist->d_snglsnr_buffer, snglsnr, one_take_size, cudaMemcpyHostToDevice, postcoh->stream));
			// 3. do transpose, at the same time, snr data will be moved to proper positions in state->d_snglsnr[cur_ifo]
			transpose_snglsnr(pklist->d_snglsnr_buffer, state->d_snglsnr[cur_ifo], state->snglsnr_start_load, one_take_len, state->snglsnr_len, state->ntmplt, postcoh->stream);
			}

		}
		cudaStreamSynchronize(postcoh->stream);
		for (i=0, collectlist = pads->data; collectlist; collectlist = g_slist_next(collectlist), i++) {

			data = collectlist->data;
			cur_ifo = state->ifo_mapping[i];

			if (is_cur_ifo_has_data(state, cur_ifo)) {
				// FIXME: GPU peakfinder produces much less peaks than peaks_over_thresh function
#if 0
				GST_LOG("peak finder for ifo %d", cur_ifo);
				peakfinder(state, cur_ifo, postcoh->stream);
				CUDA_CHECK(cudaMemcpyAsync(	state->peak_list[cur_ifo]->npeak, 
						state->peak_list[cur_ifo]->d_npeak, 
						sizeof(int), 
						cudaMemcpyDeviceToHost,
						postcoh->stream));

				cudaStreamSynchronize(postcoh->stream);

				printf("gps %d, ifo %d, gpu peak %d\n", ligo_time.gpsSeconds, cur_ifo, state->peak_list[cur_ifo]->npeak[0]);
#endif
				if (state->peak_list[cur_ifo]->npeak[0] > 0 && state->cur_nifo == state->nifo) {
					cohsnr_and_chisq(state, cur_ifo, gps_idx, postcoh->output_skymap, postcoh->stream);
					GST_LOG("after coherent analysis for ifo %d, npeak %d", cur_ifo, state->peak_list[cur_ifo]->npeak[0]);
				}

				/* move along */
				gst_audioadapter_flush_samples(data->adapter, exe_size);
			}
		}
		common_size -= exe_size;
		int exe_len = state->exe_len;
		state->snglsnr_start_load = (state->snglsnr_start_load + exe_len) % state->snglsnr_len;
		state->snglsnr_start_exe = (state->snglsnr_start_exe + exe_len) % state->snglsnr_len;
		postcoh->next_exe_t += exe_len / postcoh->rate * GST_SECOND;

		/* make a buffer and send it out */

		GstBuffer *outbuf;
		outbuf = cuda_postcoh_new_buffer(postcoh, exe_len);

		// g_assert(GST_BUFFER_CAPS(outbuf) != NULL);
		ret = gst_pad_push(postcoh->srcpad, outbuf);
		GST_LOG_OBJECT(postcoh, "pushed buffer, result = %s", gst_flow_get_name(ret));
		/* move along */
		postcoh->samples_out += exe_len;
	}


}

static GstFlowReturn collected(GstCollectPads *pads, gpointer user_data)
{
	CudaPostcoh* postcoh = CUDA_POSTCOH(user_data);
	PostcohState *state = postcoh->state;
	g_mutex_lock(&postcoh->prop_lock);
	while (state->npix == NOT_INIT || state->autochisq_len == NOT_INIT || postcoh->hist_trials == NOT_INIT) {
		g_cond_wait(&postcoh->prop_avail, &postcoh->prop_lock);
		GST_LOG_OBJECT(postcoh, "collected have to wait");
	}
	g_mutex_unlock(&postcoh->prop_lock);

	CUDA_CHECK(cudaSetDevice(postcoh->device_id));
	GstElement* element = GST_ELEMENT(postcoh);
	GstClockTime t_latest_start;
	GstFlowReturn res;
	guint64 offset_latest_start = 0;
	gint common_size; 

	GST_DEBUG_OBJECT(postcoh, "collected");
	/* Assure that we have enough sink pads. */
	if (element->numsinkpads < 2)
	{
		GST_ERROR_OBJECT(postcoh, "not enough sink pads, 2 required but only %d are present", element->numsinkpads < 2);
		return GST_FLOW_ERROR;
	}

	if (!postcoh->set_starttime) {
		/* get the latest timestamp */
		if (!cuda_postcoh_get_latest_start_time(pads, &t_latest_start, &offset_latest_start)) {
			/* bad buffer : one of the buffers is at EOS or invalid timestamp/ offset */
			GST_ERROR_OBJECT(postcoh, "cannot deduce start timestamp/ offset information");
			return GST_FLOW_ERROR;
		}
		postcoh->t0 = t_latest_start;
		postcoh->next_exe_t = postcoh->t0;
		postcoh->offset0 = offset_latest_start;
		GST_DEBUG("Debug all aligned line no %d", __LINE__);
		GST_DEBUG_OBJECT(postcoh, "set the aligned time to %" GST_TIME_FORMAT 
				", start offset to %" G_GUINT64_FORMAT,
			       	GST_TIME_ARGS(postcoh->t0),
				postcoh->offset0);
		postcoh->is_all_aligned = cuda_postcoh_align_collected(pads, postcoh);
		GST_DEBUG("Debug all aligned line no %d", __LINE__);
		postcoh->set_starttime = TRUE;
		return GST_FLOW_OK;
	}

		
	if (postcoh->is_all_aligned) {
		common_size = cuda_postcoh_push_and_get_common_size(pads, postcoh);
		GST_DEBUG_OBJECT(postcoh, "get spanned size %d, get spanned samples %d", common_size, common_size/ postcoh->bps);

		if (common_size == -1) {
			res = gst_pad_push_event(postcoh->srcpad, gst_event_new_eos());
			return res;
		}

		gint exe_size = postcoh->exe_len * postcoh->bps;
		gint one_take_size = postcoh->preserved_len/2 * postcoh->bps + exe_size;
		cuda_postcoh_process(pads, common_size, one_take_size, exe_size, postcoh);

	} else {
		GST_DEBUG("Debug all aligned line no %d", __LINE__);
		postcoh->is_all_aligned = cuda_postcoh_align_collected(pads, postcoh);
		GST_DEBUG("Debug all aligned line no %d", __LINE__);
	}
#if 0
	if (!GST_CLOCK_TIME_IS_VALID(t_start)) {
		/* eos */
		GST_DEBUG_OBJECT(postcoh, "no data available, must be EOS");
		res = gst_pad_push_event(postcoh->srcpad, gst_event_new_eos());
		return res;
	}
	GST_LOG_OBJECT(postcoh, "t end %", GST_TIME_FORMAT, t_end);
#endif
	return GST_FLOW_OK;
}


static void cuda_postcoh_dispose(GObject *object)
{
  CudaPostcoh *element = CUDA_POSTCOH(object);
  GST_DEBUG_OBJECT(element, "disposing CudaPostcoh element");
  
  if(element->collect)
  {
    gst_object_unref(GST_OBJECT(element->collect));
    element->collect = NULL;
  }

  //gst_caps_replace(element->filter_caps, NULL);
  //gst_caps_replace(element->current_caps, NULL);

  if(element->state){
    state_destroy(element->state);
    free(element->state);
    element->state = NULL;
  }

  //if(element->srcpad)
  //  gst_object_unref(element->srcpad);
  //element->srcpad = NULL;

  g_mutex_clear(&element->prop_lock);
  g_cond_clear(&element->prop_avail);

  gst_caps_replace(&element->filter_caps, NULL);
  gst_caps_replace(&element->current_caps, NULL);

  if( element->pending_events) 
  {
    g_list_foreach( element->pending_events, (GFunc) gst_event_unref, NULL );
    g_list_free(element->pending_events);
    element->pending_events = NULL;
  }

  /* destroy hashtable and its contents */
  G_OBJECT_CLASS(cuda_postcoh_parent_class)->dispose(object);
}

static void cuda_postcoh_class_init(CudaPostcohClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *gstelement_class = GST_ELEMENT_CLASS(klass);

	cuda_postcoh_parent_class = g_type_class_ref(GST_TYPE_ELEMENT);

	gobject_class->get_property = GST_DEBUG_FUNCPTR(cuda_postcoh_get_property);
	gobject_class->set_property = GST_DEBUG_FUNCPTR(cuda_postcoh_set_property);
	gobject_class->dispose = GST_DEBUG_FUNCPTR(cuda_postcoh_dispose);
	gstelement_class->request_new_pad = GST_DEBUG_FUNCPTR(cuda_postcoh_request_new_pad);
	gstelement_class->release_pad = GST_DEBUG_FUNCPTR(cuda_postcoh_release_pad);
	gstelement_class->change_state = GST_DEBUG_FUNCPTR(cuda_postcoh_change_state);

	gst_element_class_set_details_simple(
		gstelement_class,
		"Post Coherent SNR and Nullstream Generator",
		"Filter",
		"Coherent trigger generation.\n",
		"Qi Chu <qi.chu at ligo dot org>"
	);
	gst_element_class_add_pad_template(
		gstelement_class,
		gst_static_pad_template_get(&cuda_postcoh_sink_template)
	);

	gst_element_class_add_pad_template(
		gstelement_class,
//		gst_static_pad_template_get(&cuda_postcoh_src_template)
		gst_pad_template_new(
			"src",
			GST_PAD_SRC,
			GST_PAD_ALWAYS,
			gst_caps_from_string("application/x-lal-postcoh")
		)
	);

	g_object_class_install_property(
		gobject_class,
		PROP_DETRSP_FNAME,
		g_param_spec_string(
			"detrsp-fname",
			"Detector response filename",
			"Should include U map and time_diff map",
			DEFAULT_DETRSP_FNAME,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);

	g_object_class_install_property(
		gobject_class,
		PROP_SPIIR_BANK_FNAME,
		g_param_spec_string(
			"autocorrelation-fname",
			"Autocorrelation matrix filename",
			"Autocorrelation matrix",
			NULL,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);

	g_object_class_install_property(
		gobject_class,
		PROP_SNGL_TMPLT_FNAME,
		g_param_spec_string(
			"sngl-tmplt-fname",
			"File that has SnglInspiralTable",
			"single template filename",
			NULL,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);


	g_object_class_install_property(
		gobject_class,
		PROP_HIST_TRIALS,
		g_param_spec_int(
			"hist-trials",
			"history trials",
			"history that should be kept in times",
			0, G_MAXINT, 1,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);

	g_object_class_install_property(
		gobject_class,
		PROP_TRIAL_INTERVAL,
		g_param_spec_float(
			"trial-interval",
			"trial interval in seconds",
			"trial interval in seconds",
			0, G_MAXFLOAT, 0.1,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);


	g_object_class_install_property(
		gobject_class,
		PROP_OUTPUT_SKYMAP,
		g_param_spec_int(
			"output-skymap",
			"if output skymap",
			"if output skymap",
			0, 1, 0,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);

	g_object_class_install_property(
		gobject_class,
		PROP_COHSNR_THRESH,
		g_param_spec_float(
			"cohsnr-thresh",
			"coherent snr threshold",
			"coherent snr threshold",
			0.0, G_MAXFLOAT, 1.05,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);


	g_object_class_install_property(
		gobject_class,
		PROP_SNGLSNR_THRESH,
		g_param_spec_float(
			"snglsnr-thresh",
			"single snr threshold",
			"single snr threshold",
			0.0, G_MAXFLOAT, 4.0,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);

	g_object_class_install_property(
		gobject_class, 
		PROP_STREAM_ID,
		g_param_spec_int(
			"stream-id",
			"id for cuda stream",
			"id for cuda stream",
			0, G_MAXINT, 0,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);


}


static void cuda_postcoh_init(CudaPostcoh *postcoh)
{
  GstElement *element = GST_ELEMENT(postcoh);

  gst_element_create_all_pads(element);

  postcoh->srcpad = gst_element_get_static_pad(element, "src");
  GST_DEBUG_OBJECT(postcoh, "Initialized postcoh srcpad %s caps %" GST_PTR_FORMAT, GST_PAD_NAME(postcoh->srcpad), gst_pad_query_caps(postcoh->srcpad, NULL));

  gst_pad_set_query_function(postcoh->srcpad,
      GST_DEBUG_FUNCPTR(src_query));
  gst_pad_set_event_function(postcoh->srcpad, 
      GST_DEBUG_FUNCPTR(src_event));
  //GST_PAD_SET_PROXY_CAPS(postcoh->srcpad);
  gst_pad_use_fixed_caps(postcoh->srcpad);
  
  //gst_element_add_pad(GST_ELEMENT(postcoh), postcoh->srcpad);

  postcoh->collect = gst_collect_pads_new();
  gst_collect_pads_set_function(postcoh->collect, 
      GST_DEBUG_FUNCPTR(collected), postcoh);
  gst_collect_pads_set_event_function( postcoh->collect, 
      GST_DEBUG_FUNCPTR(sink_event), postcoh );
 /* gst_collect_pads_set_query_function( postcoh->collect,
      GST_DEBUG_FUNCPTR(sink_query), postcoh );
*/
  postcoh->t0 = GST_CLOCK_TIME_NONE;
  postcoh->next_exe_t = GST_CLOCK_TIME_NONE;
  postcoh->offset0 = GST_BUFFER_OFFSET_NONE;
  //postcoh->next_in_offset = GST_BUFFER_OFFSET_NONE;
  postcoh->set_starttime = FALSE;
  postcoh->is_all_aligned = FALSE;
  postcoh->samples_in = 0;
  postcoh->samples_out = 0;
  postcoh->state = (PostcohState *) malloc (sizeof(PostcohState));
  postcoh->state->autochisq_len = NOT_INIT;
  postcoh->state->npix = NOT_INIT;
  postcoh->state->is_member_init = NOT_INIT;
  postcoh->hist_trials = NOT_INIT;
  g_mutex_init(&postcoh->prop_lock);
  g_cond_init(&postcoh->prop_avail);
  postcoh->stream_id = NOT_INIT;
  postcoh->device_id = NOT_INIT;
  postcoh->process_id = 0;
  postcoh->cur_event_id = 0;
  postcoh->padcount = 0;

  postcoh->current_caps = NULL;
  gst_audio_info_init(&postcoh->info);
  postcoh->filter_caps = NULL;
}



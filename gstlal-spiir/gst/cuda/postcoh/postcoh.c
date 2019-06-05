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
#include <postcoh/postcoh.h>
#include <pipe_macro.h> // for get_icombo, IFOComboMap
#include <postcoh/postcoh_utils.h>
#include <postcoh/postcohtable_utils.h>
#include <cuda_debug.h>

#define DEFAULT_DETRSP_FNAME "H1L1V1_detrsp.xml"
#define EPSILON 5
#define PEAKFINDER_CLUSTER_WINDOW 5
#define RAD2DEG 57.2957795
#define ACCELERATE_POSTCOH_MEMORY_COPY

#define GST_CAT_DEFAULT cuda_postcoh_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);

static void additional_initializations(GType type)
{
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "cuda_postcoh", 0, "cuda_postcoh element");
}


/*
 * add a segment to the control segment array.  note they are appended, and
 * the code assumes they are in order and do not overlap, see gstlal_gate.c control_segment
 */


typedef struct flag_segment {
	GstClockTime start, stop;
	gboolean is_gap; // GAP true, non-GAP, false
}FlagSegment;

static gboolean need_flag_gap(GstPostcohCollectData *data, GstClockTime start, GstClockTime stop)
{
	GstClockTime dur_gap = 0, dur_buf = stop - start;
	gboolean need_flush = FALSE;
	GArray *flag_segments = data->flag_segments;
	int i, flush_len = 0;
	FlagSegment *this_segment = &((FlagSegment *)flag_segments->data)[flag_segments->len - 1];
	/* make sure the last segment always later than the outbuf */
	g_assert(this_segment-> stop >= stop);

	for (i=0; i<flag_segments->len; i++) {
		this_segment = &((FlagSegment *)flag_segments->data)[i];
		/*		| start				| stop
		 *									| this_start (1)
		 * | s | e (2)
		 * | s							| e
		 * | s		| e
		 *            |s | e
		 *            |s				| e
		 */
		if (this_segment-> start >= stop)
			break;
		if (this_segment->stop <= start) {
			need_flush = TRUE;
			flush_len = i-1;
			continue;
		}
		if (this_segment->start <= start && this_segment->stop >= stop) {
			dur_gap += this_segment->is_gap ? dur_buf: 0;
			continue;
		}

		if (this_segment->start <= start && this_segment->stop < stop) {
			dur_gap += this_segment->is_gap ? this_segment->stop - start: 0;
			continue;
		}
		if (this_segment->start > start && this_segment->stop <= stop) {
			dur_gap += this_segment->is_gap ? this_segment->stop - this_segment->start: 0;
			continue;
		}
		if (this_segment->start > start && this_segment->stop > stop) {
			dur_gap += this_segment->is_gap ? stop - this_segment->start: 0;
			continue;
		}
	}
	if (need_flush && flush_len > 0)
		g_array_remove_range(flag_segments, 0, flush_len);

	if (dur_gap > dur_buf/2 - 1e-6)
		return TRUE;
	else
		return FALSE;


}

static void add_flag_segment(GstPostcohCollectData *data, GstClockTime start, GstClockTime stop, gboolean is_gap)
{
	FlagSegment new_segment = {
		.start = start,
		.stop = stop,
		.is_gap = is_gap
	};

	g_assert_cmpuint(start, <=, stop);
	GST_DEBUG_OBJECT(data, "found control segment [%" GST_TIME_FORMAT ", %" GST_TIME_FORMAT ") in state %d", GST_TIME_ARGS(new_segment.start), GST_TIME_ARGS(new_segment.stop), new_segment.is_gap);



	/* try coalescing the new segment with the most recent one */
	if(data->flag_segments->len) {
		FlagSegment *final_segment = &((FlagSegment *) data->flag_segments->data)[data->flag_segments->len - 1];
		/* if the most recent segment and the new segment have the
		 * same state and they touch, merge them */
		if(final_segment->is_gap == new_segment.is_gap && final_segment->stop >= new_segment.start) {
			g_assert_cmpuint(new_segment.stop, >=, final_segment->stop);
			final_segment->stop = new_segment.stop;
			return;
		}
		/* otherwise, if the most recent segment had 0 length,
		 * replace it entirely with the new one.  note that the
		 * state carried by a zero-length segment is meaningless,
		 * zero-length segments are merely interpreted as a
		 * heart-beat indicating how far the control stream has
		 * advanced */
		if(final_segment->stop == final_segment->start) {
			*final_segment = new_segment;
			return;
		}
	}
	/* otherwise append a new segment */
	g_array_append_val(data->flag_segments, new_segment);
}



#ifdef ACCELERATE_POSTCOH_MEMORY_COPY
// This function is just copy from gst_adapter_peek and change malloc to cudaMallocHost
// FIXME: temporarily used to speed up cudaMemcpy, will be deleted when transfered to gstreamer1.x
const guint8* gst_adapter_peek_cuda	(	GstAdapter * 	adapter, guint 	size )	
{
int DEFAULT_SIZE = 16;
  GstBuffer *cur;
  GSList *cur_list;
  guint copied;

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

  if (adapter->assembled_size < size) {
    adapter->assembled_size = (size / DEFAULT_SIZE + 1) * DEFAULT_SIZE;
    GST_DEBUG_OBJECT (adapter, "setting size of internal buffer to %u",
        adapter->assembled_size);
    //CUDA_CHECK(cudaFreeHost((void*)adapter->assembled_data));
    //assert(adapter->assembled_data == NULL);
    CUDA_CHECK(cudaMallocHost((void**)&(adapter->assembled_data), adapter->assembled_size));
        //g_realloc (adapter->assembled_data, adapter->assembled_size);
  }
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

  return adapter->assembled_data;
}
#endif




GST_BOILERPLATE_FULL(
	CudaPostcoh,
	cuda_postcoh,
	GstElement,
	GST_TYPE_ELEMENT,
	additional_initializations
);

//FIXME: not support width=64 yet
static GstStaticPadTemplate cuda_postcoh_sink_template =
GST_STATIC_PAD_TEMPLATE (
		"%s",
		GST_PAD_SINK, 
		GST_PAD_REQUEST, 
		GST_STATIC_CAPS(
		"audio/x-raw-float, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) [1, MAX], " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) 32"
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
	PROP_STREAM_ID,
	PROP_REFRESH_INTERVAL
};

static void cuda_postcoh_device_set_init(CudaPostcoh *element)
{
	if (element->device_id == POSTCOH_PARAMS_NOT_INIT) {
		int deviceCount;
		CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
		/* FIXME: only print device info like runtime version in debug mode */
		// cuda_device_print(deviceCount);
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
			g_assert(element->stream_id != POSTCOH_PARAMS_NOT_INIT);
			g_mutex_lock(element->prop_lock);
			element->detrsp_fname = g_value_dup_string(value);
			cuda_postcoh_device_set_init(element);
			CUDA_CHECK(cudaSetDevice(element->device_id));
			cuda_postcoh_map_from_xml(element->detrsp_fname, element->state, element->stream);
			GST_DEBUG("detrsp map has been read in, broad cast the lock avail");
			g_cond_broadcast(element->prop_avail);
			g_mutex_unlock(element->prop_lock);
			GST_DEBUG("detrsp map lock broad cast done");
			break;

		/* read in autocorrelation and sigmasq */
		case PROP_SPIIR_BANK_FNAME: 
       			/* must make sure stream_id has already loaded */
			g_assert(element->stream_id != POSTCOH_PARAMS_NOT_INIT);
			GST_DEBUG("autocorrelation and sigma acquiring the lock");
			g_mutex_lock(element->prop_lock);
			GST_DEBUG("autocorrelation and sigma have acquired the lock");
			cuda_postcoh_device_set_init(element);
			CUDA_CHECK(cudaSetDevice(element->device_id));
			element->spiir_bank_fname = g_value_dup_string(value);
			cuda_postcoh_autocorr_from_xml(element->spiir_bank_fname, element->state, element->stream);
			cuda_postcoh_sigmasq_from_xml(element->spiir_bank_fname, element->state);
			GST_DEBUG("autocorrelation and sigma have been read in, broad cast the lock avail");
			g_cond_broadcast(element->prop_avail);
			g_mutex_unlock(element->prop_lock);
			GST_DEBUG("autocorrelation and sigma lock broad cast done");
			break;

		/* read in source information masses and spins */
		case PROP_SNGL_TMPLT_FNAME: 
       			/* must make sure stream_id has already loaded */
			g_assert(element->stream_id != POSTCOH_PARAMS_NOT_INIT);
			GST_DEBUG("sngl table acquiring the lock");
			g_mutex_lock(element->prop_lock);
			GST_DEBUG("sngl table has acquired the lock");
			element->sngl_tmplt_fname = g_value_dup_string(value);
			cuda_postcoh_sngl_tmplt_from_xml(element->sngl_tmplt_fname, &(element->sngl_table));
			GST_DEBUG("sngl tables has been read in, broad cast the lock avail");
			g_cond_broadcast(element->prop_avail);
			g_mutex_unlock(element->prop_lock);
			GST_DEBUG("sngl tables lock broad cast done");
			break;


		case PROP_HIST_TRIALS:
			g_mutex_lock(element->prop_lock);
			element->hist_trials = g_value_get_int(value);
			g_cond_broadcast(element->prop_avail);
			g_mutex_unlock(element->prop_lock);
			break;

		case PROP_TRIAL_INTERVAL:
			g_mutex_lock(element->prop_lock);
			element->trial_interval = g_value_get_float(value);
			g_cond_broadcast(element->prop_avail);
			g_mutex_unlock(element->prop_lock);
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

		case PROP_REFRESH_INTERVAL:
			element->refresh_interval = g_value_get_int(value);
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

		case PROP_REFRESH_INTERVAL:
			g_value_set_int(value, element->refresh_interval);
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

static gboolean
sink_event(GstPad *pad, GstEvent *event)
{
	CudaPostcoh *postcoh = CUDA_POSTCOH(GST_PAD_PARENT(pad));
	GstPostcohCollectData *data = gst_pad_get_element_private(pad);
	gboolean ret = TRUE;

	switch(GST_EVENT_TYPE(event)) {
		case GST_EVENT_NEWSEGMENT:
			GST_DEBUG_OBJECT(pad, "new segment");
			break;
		// do not process tag.
		case GST_EVENT_TAG:
			gst_object_unref(event);
			return TRUE;
		default:
			break;
	}
	ret = postcoh->collect_event(pad, event);
	return TRUE;
}

/*
 * forwards the event to all sinkpads, takes ownership of the event
 */


typedef struct {
	GstEvent *event;
	gboolean flush;
} EventData;


static gboolean forward_src_event_func(GstPad *pad, GValue *ret, EventData *data)
{
	GST_DEBUG_OBJECT(pad, "forward an event");
	gst_event_ref(data->event);
	if(!gst_pad_push_event(pad, data->event)) {
		/* quick hack to unflush the pads. ideally we need  a way
		 * to just unflush this single collect pad */
		if(data->flush)
			gst_pad_send_event(pad, gst_event_new_flush_stop());
	} else {
		g_value_set_boolean(ret, TRUE);
	}
	gst_object_unref(GST_OBJECT(pad));
	return TRUE;
}


static gboolean forward_src_event(CudaPostcoh *postcoh, GstEvent *event, gboolean flush)
{
	GstIterator *it;
	GValue vret = {0};
	EventData data = {
		event,
		flush
	};
	gboolean success;

	g_value_init(&vret, G_TYPE_BOOLEAN);
	g_value_set_boolean(&vret, FALSE);

	it = gst_element_iterate_sink_pads(GST_ELEMENT(postcoh));
	while(TRUE) {
		switch(gst_iterator_fold(it, (GstIteratorFoldFunction) forward_src_event_func, &vret, &data)) {
		case GST_ITERATOR_RESYNC:
			gst_iterator_resync(it);
			g_value_set_boolean(&vret, TRUE);
			break;

		case GST_ITERATOR_OK:
		case GST_ITERATOR_DONE:
			success = g_value_get_boolean(&vret);
			goto done;

		default:
			success = FALSE;
			goto done;
		}
	}
done:
	gst_iterator_free(it);
	gst_event_unref(event);

	return success;
}


/*
 * handle events received on the source pad
 */


static gboolean src_event(GstPad *pad, GstEvent *event)
{
	CudaPostcoh *postcoh = CUDA_POSTCOH(gst_pad_get_parent(pad));
	gboolean success;

	switch(GST_EVENT_TYPE(event)) {
	case GST_EVENT_SEEK: {
		gdouble rate;
		GstSeekFlags flags;
		GstSeekType curtype, endtype;
		gint64 cur, end;
		gboolean flush;

		gst_event_parse_seek(event, &rate, NULL, &flags, &curtype, &cur, &endtype, &end);
		flush = flags & GST_SEEK_FLAG_FLUSH;

		/* FIXME:  copy the adder's logic re flushing */

		success = forward_src_event(postcoh, event, flush);
		break;
	}

	/* events that can't be handled */
	case GST_EVENT_QOS:
	case GST_EVENT_NAVIGATION:
		gst_event_unref(event);
		success = FALSE;
		break;

	/* forward the rest out all sink pads */
	default:
		GST_DEBUG_OBJECT(postcoh, "forward src event");
		success = forward_src_event(postcoh, event, FALSE);
		break;
	}

	return success;
}


/* This is modified from gstadder.c 0.10.32 */
static gboolean
cuda_postcoh_sink_setcaps(GstPad *pad, GstCaps *caps)
{
	size_t freemem;
	size_t totalmem;

	CudaPostcoh *postcoh = CUDA_POSTCOH(GST_PAD_PARENT(pad));
	PostcohState *state = postcoh->state;
	g_mutex_lock(postcoh->prop_lock);
	while (state->npix == POSTCOH_PARAMS_NOT_INIT || state->autochisq_len == POSTCOH_PARAMS_NOT_INIT || postcoh->hist_trials ==POSTCOH_PARAMS_NOT_INIT) {
		g_cond_wait(postcoh->prop_avail, postcoh->prop_lock);
		GST_LOG_OBJECT(postcoh, "setcaps have to wait");
	}
	g_mutex_unlock(postcoh->prop_lock);

	CUDA_CHECK(cudaSetDevice(postcoh->device_id));
	GList *sinkpads;
	GstStructure *s;
	GstPostcohCollectData *data;

	/* FIXME: this is copied from gstadder.c. Replace with new version of that file
	 * if any. */
	GST_OBJECT_LOCK(postcoh);
	sinkpads = GST_ELEMENT(postcoh)->sinkpads;

	GST_LOG_OBJECT(postcoh, "setting caps on pad %p,%s to %" GST_PTR_FORMAT, pad,
			GST_PAD_NAME(pad), caps);

	while (sinkpads) {
		GstPad *otherpad = GST_PAD(sinkpads->data);

		if (otherpad != pad) {
			gst_caps_replace(&GST_PAD_CAPS(otherpad), caps);
		}
		sinkpads = g_list_next(sinkpads);
	}
	GST_OBJECT_UNLOCK(postcoh);

	s = gst_caps_get_structure(caps, 0);
	gst_structure_get_int(s, "width", &postcoh->width);
	gst_structure_get_int(s, "rate", &postcoh->rate);
	gst_structure_get_int(s, "channels", &postcoh->channels);


	/* postcoh and state initialization */
	postcoh->bps = (postcoh->width/8) * postcoh->channels;	
	postcoh->offset_per_nanosecond = postcoh->bps / 1e9 * (postcoh->rate);	

	GST_DEBUG_OBJECT(postcoh, "setting GstPostcohCollectData width %d, rate %d, offset_per_nanosecond %f, bps %d, channels %d", postcoh->width, postcoh->rate, postcoh->offset_per_nanosecond, postcoh->bps, postcoh->channels);

	state->nifo = GST_ELEMENT(postcoh)->numsinkpads;
	state->input_ifo_mapping = (gint *)malloc(sizeof(gint) * state->nifo);
	state->ifo_combo_idx = 0;
	state->all_ifos = (gchar *)malloc(sizeof(gchar) * state->nifo * IFO_LEN+1);
	state->peak_list = (PeakList **)malloc(sizeof(PeakList*) * state->nifo);
	state->dt = (float) 1/postcoh->rate;

	/* need to cover head and tail */
	postcoh->preserved_len = state->autochisq_len + 160; 
	postcoh->exe_len = postcoh->rate;
	postcoh->exe_size = postcoh->exe_len * postcoh->bps;

	postcoh->one_take_len = postcoh->preserved_len/2  + postcoh->exe_len;
	postcoh->one_take_size = postcoh->one_take_len * postcoh->bps;

	state->exe_len = postcoh->rate;
	state->tmp_maxsnr = (float *) malloc(sizeof(float) * state->exe_len);
	state->tmp_tmpltidx = (int *) malloc(sizeof(int) * state->exe_len);
	state->max_npeak = postcoh->rate > postcoh->channels/2 ? postcoh->channels/2 : postcoh->rate;
	state->trial_sample_inv = round(postcoh->trial_interval * postcoh->rate);
	state->snglsnr_len = postcoh->preserved_len + postcoh->exe_len + postcoh->hist_trials * state->trial_sample_inv;
	state->hist_trials = postcoh->hist_trials;
	state->snglsnr_start_load = postcoh->hist_trials * state->trial_sample_inv;
	state->snglsnr_start_exe = state->snglsnr_start_load;

	GST_DEBUG_OBJECT(postcoh, "hist_trials %d, autochisq_len %d, preserved_len %d, sngl_len %d, bps %d, exe_len %d, one_take_len %d, one_take_size %d, start_load %d, start_exe %d, max_npeak %d", 
			state->hist_trials, state->autochisq_len, postcoh->preserved_len, state->snglsnr_len, postcoh->bps, postcoh->exe_len, postcoh->one_take_len, postcoh->one_take_size, state->snglsnr_start_load, state->snglsnr_start_exe, state->max_npeak);

	state->ntmplt = postcoh->channels/2;
	CUDA_CHECK(cudaMemGetInfo(&freemem, &totalmem));
	printf( "Free memory: %d MB\nTotal memory: %d MB\n", (int)(freemem / 1024 / 1024), (int)(totalmem / 1024 / 1024) );
	printf( "Allocating %d B for dd_snglsnr\n", (int) sizeof(COMPLEX_F *) * state->nifo );

	CUDA_CHECK(cudaMalloc((void **)&(state->dd_snglsnr), sizeof(COMPLEX_F *) * state->nifo));
	state->d_snglsnr = (COMPLEX_F **)malloc(sizeof(COMPLEX_F *) * state->nifo);
	/* when dumping the sngl outputs, need to follow the order in this structure, see ker_coh_max_and_chisq */
	state->write_ifo_mapping = (int *)malloc(sizeof(int) * state->nifo);

	gint8 i = 0, j = 0, cur_ifo = 0, nifo = state->nifo;
	GST_OBJECT_LOCK(postcoh->collect);
	/* find the ifo_combo_mapping and input_ifo_mapping:
	 * first find the ifo_combo index in the IFOComboMap
	 * then, map original sinkpad's ifo to the position in this combo */
	for (i=0, sinkpads = GST_ELEMENT(postcoh)->sinkpads; sinkpads; sinkpads = g_list_next(sinkpads), i++) {
		GstPad *pad = GST_PAD(sinkpads->data);
		data = gst_pad_get_element_private(pad);
		set_offset_per_nanosecond(data, postcoh->offset_per_nanosecond);
		set_channels(data, postcoh->channels);
		// FIXME: need to consider non-standard ifo indexing, like HV, need testing
		strncpy(state->all_ifos + IFO_LEN*i, data->ifo_name, sizeof(char)*IFO_LEN);
	}
	state->all_ifos[IFO_LEN*nifo] = '\0';
	state->ifo_combo_idx = get_icombo(state->all_ifos);
	/* overwrite all_ifos to be the same with the combo in the IFOComboMap */
	strncpy(state->all_ifos, IFOComboMap[state->ifo_combo_idx].name, sizeof(IFOComboMap[state->ifo_combo_idx].name));
	state->all_ifos[IFO_LEN*nifo] = '\0';

	/* initialize input_ifo_mapping, snglsnr matrix, and peak_list */
	for (i=0, sinkpads = GST_ELEMENT(postcoh)->sinkpads; sinkpads; sinkpads = g_list_next(sinkpads), i++) {
		GstPad *pad = GST_PAD(sinkpads->data);
		data = gst_pad_get_element_private(pad);
		for (j = 0; j < nifo; j++ )
			if (strncmp(data->ifo_name, state->all_ifos + IFO_LEN*j, IFO_LEN) == 0 ) {
				state->input_ifo_mapping[i] = j;
				cur_ifo = j;
				break;
		}
		guint mem_alloc_size = state->snglsnr_len * postcoh->bps;
		//printf("device id %d, stream addr %p, alloc for snglsnr %d\n", postcoh->device_id, postcoh->stream, mem_alloc_size);
		
		CUDA_CHECK(cudaMemGetInfo(&freemem, &totalmem));
		printf( "Free memory: %d MB  Total memory: %d MB\n", (int)(freemem / 1024 / 1024), (int)(totalmem / 1024 / 1024) );
		printf( "Allocating SNR series %u B, i.e. %d MB for ifo %d\n", mem_alloc_size, (int) (mem_alloc_size / 1024 / 1024), cur_ifo);

		CUDA_CHECK(cudaMalloc((void**) &(state->d_snglsnr[cur_ifo]), mem_alloc_size));
		CUDA_CHECK(cudaMemsetAsync(state->d_snglsnr[cur_ifo], 0, mem_alloc_size, postcoh->stream));
		CUDA_CHECK(cudaMemcpyAsync(&(state->dd_snglsnr[cur_ifo]), &(state->d_snglsnr[cur_ifo]), sizeof(COMPLEX_F *), cudaMemcpyHostToDevice, postcoh->stream));
		CUDA_CHECK(cudaStreamSynchronize(postcoh->stream));
		CUDA_CHECK(cudaPeekAtLastError());

		state->peak_list[cur_ifo] = create_peak_list(postcoh->state, postcoh->stream);
	}
	get_write_ifo_mapping(state->all_ifos, nifo, state->write_ifo_mapping);

    CUDA_CHECK(cudaMalloc((void**) &state->d_write_ifo_mapping, sizeof(int) * state->nifo));
	CUDA_CHECK(cudaMemsetAsync(state->d_write_ifo_mapping, 0, sizeof(int) * state->nifo, postcoh->stream));
	CUDA_CHECK(cudaMemcpyAsync(state->d_write_ifo_mapping, state->write_ifo_mapping, sizeof(int)* state->nifo, cudaMemcpyHostToDevice, postcoh->stream));
	CUDA_CHECK(cudaStreamSynchronize(postcoh->stream));

	state->is_member_init = POSTCOH_PARAMS_INIT;
	GST_OBJECT_UNLOCK(postcoh->collect);
	return TRUE;
}

static void destroy_notify(GstPostcohCollectData *data)
{
	if (data) {
		free(data->ifo_name);
		if (data->adapter) {
			gst_adapter_clear(data->adapter);
			g_object_unref(data->adapter);
			data->adapter = NULL;
		}
		if (data->flag_segments) {
			g_array_unref(data->flag_segments);
			data->flag_segments = NULL;
		}

	}

}

static GstPad *cuda_postcoh_request_new_pad(GstElement *element, GstPadTemplate *templ, const gchar *name)
{
	CudaPostcoh* postcoh = CUDA_POSTCOH(element);

	GstPad* newpad;
       	newpad = gst_pad_new_from_template(templ, name);

	gst_pad_set_setcaps_function(GST_PAD(newpad), GST_DEBUG_FUNCPTR(cuda_postcoh_sink_setcaps));

	if (!gst_element_add_pad(element, newpad)) {
		gst_object_unref(newpad);
		return NULL;
	}

	GstPostcohCollectData* data;
    data = (GstPostcohCollectData*) gst_collect_pads_add_pad_full(postcoh->collect, newpad, sizeof(GstPostcohCollectData), (GstCollectDataDestroyNotify) GST_DEBUG_FUNCPTR(destroy_notify));
	postcoh->collect_event = (GstPadEventFunction) GST_PAD_EVENTFUNC(newpad);
	gst_pad_set_event_function(newpad, sink_event);

	if (!data) {
		gst_element_remove_pad(element, newpad);
		gst_object_unref(newpad);
		return NULL;
	}

	data->ifo_name = (gchar *)malloc(IFO_LEN*sizeof(gchar));
	g_strlcpy(data->ifo_name, name, sizeof(data->ifo_name));
	data->adapter = gst_adapter_new();
	data->flag_segments = g_array_new(FALSE, FALSE, sizeof(FlagSegment));
	data->is_aligned = FALSE;
	data->aligned_offset0 = 0;
	data->next_offset = 0;
	GST_DEBUG_OBJECT(element, "new pad for %s is added and initialised", data->ifo_name);

	return GST_PAD(newpad);
}



static void cuda_postcoh_release_pad(GstElement *element, GstPad *pad)
{
	CudaPostcoh* postcoh = CUDA_POSTCOH(element);
	/* FIXME: free adapter and flag_segments */
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

	return GST_ELEMENT_CLASS(parent_class)->change_state(element, transition);
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
		if(!GST_BUFFER_TIMESTAMP_IS_VALID(buf) || !GST_BUFFER_DURATION_IS_VALID(buf)) {
			GST_ERROR_OBJECT(pads, "%" GST_PTR_FORMAT ": %" GST_PTR_FORMAT " does not have a valid timestamp and/or duration", ((GstCollectData *) data)->pad, buf);
			gst_buffer_unref(buf);
			return FALSE;
		}

		t_start_cur = GST_BUFFER_TIMESTAMP(buf);

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

	/* invalid pads */
	g_return_val_if_fail(pads != NULL, FALSE);
	g_return_val_if_fail(GST_IS_COLLECT_PADS(pads), FALSE);


	for (collectlist = pads->data; collectlist; collectlist = g_slist_next(collectlist)) {
		data = collectlist->data;
		buf = gst_collect_pads_peek(pads, (GstCollectData *)data);

		if (buf != NULL) { // != if(buf)
		/* if the buffer in the pad is behind what we expected,
		 * we span the gap using zero buffer.
		 */ 
		if (GST_BUFFER_OFFSET(buf) > data->next_offset) {
			GST_DEBUG_OBJECT(data, "gap :data offset %" G_GUINT64_FORMAT "current next offset %" G_GUINT64_FORMAT, GST_BUFFER_OFFSET(buf), data->next_offset);
			GstBuffer *zerobuf = gst_buffer_new_and_alloc((GST_BUFFER_OFFSET(buf)- data->next_offset) * postcoh->bps); 
			if(!zerobuf) {
				GST_DEBUG_OBJECT(data, "failure allocating zero-pad buffer");
			}
			memset(GST_BUFFER_DATA(zerobuf), 0, GST_BUFFER_SIZE(zerobuf));
			gst_adapter_push(data->adapter, zerobuf);

		}
		((GstPostcohCollectData *)data)->next_offset = GST_BUFFER_OFFSET_END(buf);
		gst_buffer_unref(buf);
		}
	}
	return TRUE;
}

static gint cuda_postcoh_push_and_get_common_size(GstCollectPads *pads, CudaPostcoh *postcoh)
{


	PostcohState *state = postcoh->state;
	GSList *collectlist;
	GstPostcohCollectData *data;
	GstBuffer *buf = NULL;

	gint i = 0, min_size = 0, size_cur, cur_ifo_has_data = 0;
	gboolean min_size_init = FALSE, is_gap;
	GstClockTime buf_end;

	/* The logic to find common size:
	 * if one detector has no data, we obtain the data size in the adapter
	 * and find the common size of this detector with other detectors who have
	 * data. if there is no data in this adapter, the common size is determined
	 * by other detectors.
	 */
	for (i=0, collectlist = pads->data; collectlist; collectlist = g_slist_next(collectlist), i++) {
			data = collectlist->data;
			buf = gst_collect_pads_pop(pads, (GstCollectData *)data);
			if (!buf) { //buf == NULL
				size_cur = gst_adapter_available(data->adapter);
				if(!min_size_init) {
					min_size = size_cur;
					min_size_init = size_cur > 0 ? TRUE : FALSE;
				} else {
					min_size = min_size > size_cur ? size_cur : min_size;
				}

				continue;
			}

			buf_end = GST_BUFFER_TIMESTAMP(buf) + GST_BUFFER_DURATION(buf); 
			is_gap = GST_BUFFER_FLAG_IS_SET(buf, GST_BUFFER_FLAG_GAP) ? TRUE : FALSE;
			add_flag_segment(data, GST_BUFFER_TIMESTAMP(buf), buf_end, is_gap);
			gst_adapter_push(data->adapter, buf);

			size_cur = gst_adapter_available(data->adapter);
			if(!min_size_init) {
				min_size = size_cur;
				min_size_init = TRUE;
			} else {
				min_size = min_size > size_cur ? size_cur : min_size;
			}
			cur_ifo_has_data++;
			/* should not unref the buf insce adapter is using it */
	}
	/* If all pads returns NULL buffers, this means all pads at EOS,
	 * we flag min_size as -1 to indicate we need to send an EOS event */ 
	if (cur_ifo_has_data == 0)
		min_size = -1;

	GST_LOG_OBJECT(postcoh, "get common size %d", min_size);
	return min_size;
}

static gboolean cuda_postcoh_need_recollect(GstCollectPads *pads, CudaPostcoh *postcoh)
{

	GSList *collectlist;
	GstPostcohCollectData *data;
	GstBuffer *buf = NULL;
	gboolean need_recollect = FALSE, is_gap;

    /* expected end time for the run */
	GstClockTime ts_expect = postcoh->t0 + gst_util_uint64_scale_int_round(postcoh->samples_out + postcoh->one_take_len, GST_SECOND,
		       	postcoh->rate), buf_end;


	for (collectlist = pads->data; collectlist; collectlist = g_slist_next(collectlist)) {
		data = collectlist->data;
		buf = gst_collect_pads_peek(pads, (GstCollectData *)data);
		if (buf != NULL) {  // != if(buf)
			/* zerobuf remove it */
			if (GST_BUFFER_SIZE(buf) == 0) {
				GST_DEBUG_OBJECT(postcoh, "bufsize is zero");
				gst_buffer_unref(buf);
				/* discard this buffer in collectpads so it can collect new one */
				buf = gst_collect_pads_pop(pads, (GstCollectData *)data);
				gst_buffer_unref(buf);
				need_recollect = TRUE;
				continue;
			}
			/* accumulate not enough data */
			buf_end = GST_BUFFER_TIMESTAMP(buf) + GST_BUFFER_DURATION(buf); 

			if (buf_end < ts_expect) {
				gst_buffer_unref(buf);
				/* dump this buffer in collectpads adaptor so it can collect new one */
				buf = gst_collect_pads_pop(pads, (GstCollectData *)data);
				is_gap = GST_BUFFER_FLAG_IS_SET(buf, GST_BUFFER_FLAG_GAP) ? TRUE : FALSE;
				add_flag_segment(data, GST_BUFFER_TIMESTAMP(buf), buf_end, is_gap);
				gst_adapter_push(data->adapter, buf);
				need_recollect = TRUE;
				continue;
			}

			// normal buffer and ready to be postcoh processed
			gst_buffer_unref(buf);
	
			// GST_DEBUG_OBJECT(postcoh, "bufsize is not zero  %" GST_TIME_FORMAT ",end %" GST_TIME_FORMAT, GST_TIME_ARGS(GST_BUFFER_TIMESTAMP(buf)), GST_TIME_ARGS(buf_end));
		}
		/* do nothing if it is null buffer */
	}
	return need_recollect;
}

static gboolean cuda_postcoh_align_collected(GstCollectPads *pads, CudaPostcoh *postcoh)
{

	GSList *collectlist;
	GstPostcohCollectData *data;
	GstBuffer *buf, *subbuf;
	GstClockTime t_start_cur, t_end_cur;
	gboolean all_aligned = TRUE, is_gap;
	guint64 offset_cur, offset_end_cur, buf_aligned_offset0;
	GstClockTime t0 = postcoh->t0;

	GST_DEBUG_OBJECT(pads, "begin to align offset0");

	for (collectlist = pads->data; collectlist; collectlist = g_slist_next(collectlist)) {
		data = collectlist->data;
		GST_DEBUG_OBJECT(pads, "now at %s is aligned %d", data->ifo_name, data->is_aligned);
		if (data->is_aligned) {
			/* do not collect the buffer in this pad. wait for other pads to be aligned */
			//buf = gst_collect_pads_pop(pads, (GstCollectData *)data);
			//gst_adapter_push(data->adapter, buf);
			continue;
		}
		buf = gst_collect_pads_pop(pads, (GstCollectData *)data);
		t_start_cur = GST_BUFFER_TIMESTAMP(buf);
		t_end_cur = t_start_cur + GST_BUFFER_DURATION(buf);
		offset_cur = GST_BUFFER_OFFSET(buf);
		offset_end_cur = GST_BUFFER_OFFSET_END(buf);
		if (t_end_cur > t0) {
			is_gap = GST_BUFFER_FLAG_IS_SET(buf, GST_BUFFER_FLAG_GAP) ? TRUE : FALSE;
			add_flag_segment(data, t_start_cur, t_end_cur, is_gap);

			buf_aligned_offset0 = (gint) (postcoh->offset0 - offset_cur);
			GST_DEBUG_OBJECT(data, "buffer aligned offset0 %u, size %u offset0 %" G_GUINT64_FORMAT " offset_cur %" G_GUINT64_FORMAT, buf_aligned_offset0, (offset_end_cur - offset_cur - buf_aligned_offset0) * data->channels * sizeof(float), postcoh->offset0, offset_cur);
			g_assert(buf_aligned_offset0 >= 0);
			subbuf = gst_buffer_create_sub(buf, buf_aligned_offset0, (offset_end_cur - offset_cur - buf_aligned_offset0) * data->channels * sizeof(float));
			g_assert(subbuf);
			GST_LOG_OBJECT (pads,
				"Created sub buffer of (%u bytes) with timestamp %" GST_TIME_FORMAT ", duration %"
				GST_TIME_FORMAT ", offset %" G_GUINT64_FORMAT ", offset_end %"
				G_GUINT64_FORMAT,  GST_BUFFER_SIZE (subbuf),
				GST_TIME_ARGS (GST_BUFFER_TIMESTAMP (subbuf)),
				GST_TIME_ARGS (GST_BUFFER_DURATION (subbuf)),
				GST_BUFFER_OFFSET (subbuf), GST_BUFFER_OFFSET_END (subbuf));

			gst_adapter_push(data->adapter, subbuf);

			data->is_aligned = TRUE;
			/* from the first buffer in the adapter, we initiate the next offset */
			data->next_offset = GST_BUFFER_OFFSET_END(buf);
			gst_buffer_unref(buf);
		} else {
			all_aligned = FALSE;
			gst_buffer_unref(buf);
		}
	}

	return all_aligned;

}
static void cuda_postcoh_flush(GstCollectPads *pads, guint64 common_size)
{
	GSList *collectlist;
	GstPostcohCollectData *data;

	for (collectlist = pads->data; collectlist; collectlist = g_slist_next(collectlist)) {
		data = collectlist->data;
		gst_adapter_flush(data->adapter, common_size);
	}

}

static int cuda_postcoh_select_background(PeakList *pklist,int write_ifo, int hist_trials, int max_npeak, float cohsnr_thresh)
{
	int ipeak, npeak, peak_cur, itrial, background_cur, left_backgrounds = 0;
	npeak = pklist->npeak[0];
	for (ipeak=0; ipeak<npeak; ipeak++) {
		peak_cur = pklist->peak_pos[ipeak];
		for (itrial=1; itrial<=hist_trials; itrial++) {
			background_cur = (itrial - 1)*max_npeak + peak_cur;
			// FIXME: consider a different threshold for 3-detector 
//			if (sqrt(pklist->cohsnr_bg[background_cur]) > cohsnr_thresh * pklist->snglsnr_H[iifo*max_npeak + peak_cur])
			if (sqrt(pklist->cohsnr_bg[background_cur]) > 0.1 + pklist->snglsnr_H[write_ifo*max_npeak + peak_cur]) {
				left_backgrounds++;
				GST_LOG("mark back,%d ipeak, %d itrial", ipeak, itrial);
			}
			else
				pklist->cohsnr_bg[background_cur] = -1;
		}
	}
	return left_backgrounds;
}

static int cuda_postcoh_select_foreground(PostcohState *state, float cohsnr_thresh)
{
	int iifo, ipeak, npeak, nifo = state->nifo,  cluster_peak_pos[state->max_npeak], bubbled_peak_pos[state->max_npeak], peak_cur;
	int final_peaks = 0, bubbled_peaks = 0;
	PeakList *pklist; 
	int *peak_pos;
	int left_entries = 0;

	for(iifo=0; iifo<nifo; iifo++) {
		if (state->cur_ifo_is_gap[iifo])
			continue;
		final_peaks = 0;
		bubbled_peaks = 0;
		pklist= state->peak_list[iifo];
		npeak = pklist->npeak[0];
		peak_pos = pklist->peak_pos;
		state->skymap_peakcur[iifo] = peak_pos[0];

		/*
		 * select background that satisfy the criteria: cohsnr > triggersnr + coh_thresh
		 */
		if (npeak > 0)
			left_entries += cuda_postcoh_select_background(pklist, state->write_ifo_mapping[iifo], state->hist_trials, state->max_npeak, cohsnr_thresh);
	
		/*
		 * mark the rest of peak positions to be -1 to identify invalid background
		 */

		for(ipeak=0; ipeak<state->max_npeak; ipeak++)
		   cluster_peak_pos[ipeak] = -1;
		memcpy(cluster_peak_pos, peak_pos, sizeof(int) * npeak);

		/*
		 * select zerolag that satisfy the criteria: cohsnr > triggersnr + coh_thresh
		 */
		int write_ifo = state->write_ifo_mapping[iifo];
		for(ipeak=0; ipeak<npeak; ipeak++) {
			/* if the difference of maximum single snr and coherent snr is ignorable,
			 * it means that only one detector is in action,
			 * we abandon this peak
			 * */
			peak_cur = peak_pos[ipeak];
			// FIXME: consider a different threshold for 3-detector 
			if (sqrt(pklist->cohsnr[peak_cur]) > 0.1 + pklist->snglsnr_H[write_ifo*(state->max_npeak) + peak_cur]) {
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

static int cuda_postcoh_write_table_to_buf(CudaPostcoh *postcoh, GstBuffer *outbuf)
{
	PostcohState *state = postcoh->state;

	PostcohInspiralTable *output = (PostcohInspiralTable *) GST_BUFFER_DATA(outbuf);
	int iifo = 0, jifo = 0, nifo = state->nifo;
	int ifos_size = sizeof(char) * IFO_LEN * state->cur_nifo, one_ifo_size = sizeof(char) * IFO_LEN ;
	int ipeak, npeak = 0, itrial = 0, exe_len = state->exe_len, max_npeak = state->max_npeak;
	int hist_trials = postcoh->hist_trials;

	int tmplt_idx;
	int write_entries = 0;
	
	GstClockTime ts = GST_BUFFER_TIMESTAMP(outbuf);
	LIGOTimeGPS end_time;

	int livetime = (int) ((ts - postcoh->t0)/GST_SECOND), cur_tmplt_idx;

	SnglInspiralTable *sngl_table = postcoh->sngl_table;
	/* the first entry is reserved to be used to indicate participating IFOs */
	g_assert(state->cur_nifo >= 1);
	XLALINT8NSToGPS(&end_time, ts);
	output->end_time = end_time;
	output->is_background = FLAG_EMPTY;
	strncpy(output->ifos, state->cur_ifos, ifos_size);
	output->ifos[IFO_LEN*state->cur_nifo] = '\0';
	output++;
	write_entries++;
	/* end of the first entry */

	/* FIXME: can output single-detector events, consider cohsnr = single snr, and cmbchisq = single chisq */
	/* only output multi-detector events, cohsnr, cmbchisq only make sense when cur_nifo >=2 */
	for(iifo=0; (iifo<nifo) && (state->cur_nifo >=2); iifo++) {
		if (state->cur_ifo_is_gap[iifo])
			continue;
		PeakList *pklist = state->peak_list[iifo];
		npeak = pklist->npeak[0];

		int peak_cur, len_cur, peak_cur_bg;
		int *peak_pos = pklist->peak_pos;
		for(ipeak=0; ipeak<npeak; ipeak++) {
			output->next = NULL;
			peak_cur = peak_pos[ipeak];
			cur_tmplt_idx = pklist->tmplt_idx[peak_cur];
			XLALINT8NSToGPS(&end_time, ts);
			// NOTE: adjust for the merger/epoch time of the trigger
			XLALGPSAddGPS(&end_time, &(sngl_table[cur_tmplt_idx].end));
			len_cur = pklist->len_idx[peak_cur];
			XLALGPSAdd(&(end_time), (double) len_cur/exe_len);
			output->end_time = end_time;
			XLALGPSAdd(&(end_time), (double) pklist->ntoff_H[peak_cur]/exe_len);
			output->end_time_H = end_time;
			end_time = output->end_time;
			XLALGPSAdd(&(end_time), (double) pklist->ntoff_L[peak_cur]/exe_len);
			output->end_time_L = end_time;
			end_time = output->end_time;
			XLALGPSAdd(&(end_time), (double) pklist->ntoff_V[peak_cur]/exe_len);
			output->end_time_V = end_time;
			output->snglsnr_H = pklist->snglsnr_H[peak_cur];
			output->snglsnr_L = pklist->snglsnr_L[peak_cur];
			output->snglsnr_V = pklist->snglsnr_V[peak_cur];
			output->coaphase_H = pklist->coaphase_H[peak_cur];
			output->coaphase_L = pklist->coaphase_L[peak_cur];
			output->coaphase_V = pklist->coaphase_V[peak_cur];
			output->chisq_H = pklist->chisq_H[peak_cur];
			output->chisq_L = pklist->chisq_L[peak_cur];
			output->chisq_V = pklist->chisq_V[peak_cur];

			for(jifo=0; jifo<nifo; jifo++)
			{
				int write_ifo = state->write_ifo_mapping[jifo];
				*(&output->deff_H+write_ifo) = sqrt(state->sigmasq[jifo][cur_tmplt_idx])/ *(pklist->snglsnr_H + write_ifo * state->max_npeak + peak_cur);  // in MPC
			}
			output->is_background = FLAG_FOREGROUND;
			output->livetime = livetime;
			strncpy(output->ifos, state->cur_ifos, ifos_size);
			output->ifos[IFO_LEN*state->cur_nifo] = '\0';
		    strncpy(output->pivotal_ifo, state->all_ifos + IFO_LEN*iifo, one_ifo_size);
			output->pivotal_ifo[IFO_LEN] = '\0';
			output->tmplt_idx = cur_tmplt_idx;
			output->bankid = postcoh->stream_id;
			output->pix_idx = pklist->pix_idx[peak_cur];
			output->cohsnr = sqrt(pklist->cohsnr[peak_cur]); /* the returned snr from cuda kernel is snr^2 */
			output->nullsnr = sqrt(pklist->nullsnr[peak_cur]);
			output->cmbchisq = pklist->cmbchisq[peak_cur]/state->cur_nifo;
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
			output->event_id = postcoh->cur_event_id++;
			if (postcoh->output_skymap && state->snglsnr_max[iifo] > postcoh->output_skymap && state->skymap_peakcur[iifo] == peak_cur) {
				GString *filename = NULL;
				FILE *file = NULL;
				filename = g_string_new(IFOComboMap[get_icombo(output->ifos)].name);
				g_string_append_printf(filename, "_skymap/%s_%d_%d_%d_%d", output->pivotal_ifo, output->end_time.gpsSeconds, output->end_time.gpsNanoSeconds, output->bankid, output->tmplt_idx);
				strcpy(output->skymap_fname, filename->str);
				GST_LOG("file %s is written, skymap addr %p\n", output->skymap_fname, pklist->cohsnr_skymap);
				file = fopen(output->skymap_fname, "w");
				fwrite(pklist->cohsnr_skymap, sizeof(float), state->npix, file);
				fwrite(pklist->nullsnr_skymap, sizeof(float), state->npix, file);
				fclose(file);
				file = NULL;
				g_string_free(filename, TRUE);
			} else
				output->skymap_fname[0] = '\0';
            output->rank = 0;

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
			
			XLALINT8NSToGPS(&output->epoch, ts);
			output->deltaT = 1./postcoh->rate;
			/* set the snr length to 0, FIXME: dump snr series of single ifo */
			output->snr_length = 0;
			output++;
			write_entries++;
		}

		/* NOTE: here needs to be max_npeak for bg, npeak for zerolag. */
		for(ipeak=0; ipeak<state->max_npeak; ipeak++) {
			for(itrial=1; itrial<=hist_trials; itrial++) {
				peak_cur = peak_pos[ipeak];
				len_cur = pklist->len_idx[peak_cur];
				/* check if cohsnr pass the valid test */
				peak_cur_bg = (itrial - 1)*max_npeak + peak_cur;
				if (peak_cur >= 0 && pklist->cohsnr_bg[peak_cur_bg] > 0) {
				//output->end_time = end_time[ipeak];
				output->is_background = FLAG_BACKGROUND;
				output->livetime = livetime;
				strncpy(output->ifos, state->cur_ifos, ifos_size);
				output->ifos[IFO_LEN*state->cur_nifo] = '\0';
				strncpy(output->pivotal_ifo, state->all_ifos + IFO_LEN*iifo, one_ifo_size);
				output->pivotal_ifo[IFO_LEN] = '\0';
				output->tmplt_idx = pklist->tmplt_idx[peak_cur];
				output->snglsnr_H = pklist->snglsnr_bg_H[peak_cur_bg];
				output->snglsnr_L = pklist->snglsnr_bg_L[peak_cur_bg];
				output->snglsnr_V = pklist->snglsnr_bg_V[peak_cur_bg];
				output->coaphase_H = pklist->coaphase_bg_H[peak_cur_bg];
				output->coaphase_L = pklist->coaphase_bg_L[peak_cur_bg];
				output->coaphase_V = pklist->coaphase_bg_V[peak_cur_bg];
				output->chisq_H = pklist->chisq_bg_H[peak_cur_bg];
				output->chisq_L = pklist->chisq_bg_L[peak_cur_bg];
				output->chisq_V = pklist->chisq_bg_V[peak_cur_bg];

				//output->pix_idx = pklist->pix_idx[itrial*max_npeak + peak_cur];
				output->cohsnr = sqrt(pklist->cohsnr_bg[peak_cur_bg]);
				output->nullsnr = sqrt(pklist->nullsnr_bg[peak_cur_bg]);
				output->cmbchisq = pklist->cmbchisq_bg[peak_cur_bg]/state->cur_nifo;
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
				
				/* do not dump snr for background */
				XLALINT8NSToGPS(&output->epoch, ts);
				output->deltaT = 0;
				output->snr_length = 0;
				output++;
				write_entries++;
				}
			}
		}

		GST_LOG_OBJECT(postcoh, "write to output, ifo %d, npeak %d, %d total entries", iifo, npeak, write_entries);
	}
	return write_entries;

}

static GstFlowReturn cuda_postcoh_new_buffer_and_push(CudaPostcoh *postcoh, gint out_len)
{
	GstBuffer *outbuf = NULL;
	GstPad *srcpad = postcoh->srcpad;
	GstCaps *caps = GST_PAD_CAPS(srcpad);
	GstFlowReturn ret;
	PostcohState *state = postcoh->state;
	int left_entries = 0;

	/* NOTE: explicitly add one more entry to indicate the participating IFOs */
	if (state->cur_nifo >= 2)
		left_entries = cuda_postcoh_select_foreground(state, postcoh->cohsnr_thresh) + 1;
	else if (state->cur_nifo == 1)
		left_entries = 1;

	int out_size = sizeof(PostcohInspiralTable) * left_entries ;

	ret = gst_pad_alloc_buffer(srcpad, 0, out_size, caps, &outbuf);
	if (ret != GST_FLOW_OK) {
		GST_ERROR_OBJECT(srcpad, "Could not allocate postcoh-inspiral buffer %d", ret);
		return ret;
	}
	memset(GST_BUFFER_DATA(outbuf), 0, out_size);
	
	        /* set the time stamps */
	GstClockTime ts = postcoh->t0 + gst_util_uint64_scale_int_round(postcoh->samples_out, GST_SECOND,
		       	postcoh->rate);
	
    GST_BUFFER_TIMESTAMP(outbuf) = ts;
	GST_BUFFER_DURATION(outbuf) = (GstClockTime) gst_util_uint64_scale_int_round(GST_SECOND, out_len, postcoh->rate);
	
	/* set the offset */
    GST_BUFFER_OFFSET(outbuf) = postcoh->offset0 + postcoh->samples_out;
    GST_BUFFER_OFFSET_END(outbuf) = GST_BUFFER_OFFSET(outbuf) + out_len;
	
	GST_BUFFER_SIZE(outbuf) = out_size;

	int write_entries = 0;
	if (left_entries >= 1)
		write_entries = cuda_postcoh_write_table_to_buf(postcoh, outbuf);

	/* make sure output entries equals estimation */
	g_assert(write_entries == left_entries);

	GST_LOG_OBJECT (srcpad,
		"Processed of (%d entries) with timestamp %" GST_TIME_FORMAT ", duration %"
		GST_TIME_FORMAT ", offset %" G_GUINT64_FORMAT ", offset_end %"
		G_GUINT64_FORMAT,  left_entries,
		GST_TIME_ARGS (GST_BUFFER_TIMESTAMP (outbuf)),
		GST_TIME_ARGS (GST_BUFFER_DURATION (outbuf)),
		GST_BUFFER_OFFSET (outbuf), GST_BUFFER_OFFSET_END (outbuf));

	if (left_entries == 0) 
		GST_BUFFER_FLAG_SET(outbuf, GST_BUFFER_FLAG_GAP);

	ret = gst_pad_push(postcoh->srcpad, outbuf);
	if (ret != GST_FLOW_OK) {
		fprintf(stderr, "failed to push buffer to next element, cohfar_accumbackground");
		exit(0);
	}
	GST_LOG_OBJECT(postcoh, "pushed buffer, result = %s", gst_flow_get_name(ret));
	return ret;

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
	int exe_len = state->exe_len, ntmplt = state->ntmplt, itmplt, ilen, jlen, npeak = 0, max_npeak = state->max_npeak;
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
		peak_pos[MIN(ilen, max_npeak)] = -1;
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
	  if (tmp_maxsnr[ilen] > state->snglsnr_max[cur_ifo])
	    state->snglsnr_max[cur_ifo] = tmp_maxsnr[ilen];
	}

	/* do clustering every PEAKFINDER_CLUSTER_WINDOW samples, FIXME: if set to 0, the size of output will be ten times */
	int cluster_peak_pos[max_npeak], len_cluster_peak, len_next_peak, final_peaks=0, ipeak;
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

static void cuda_postcoh_process(GstCollectPads *pads, gint common_size, CudaPostcoh *postcoh)
{
	GSList *collectlist;
	GstPostcohCollectData *data;
	COMPLEX_F *snglsnr, *pos_dd_snglsnr, *pos_in_snglsnr;

	int i = 0, cur_ifo = 0;
	PostcohState *state = postcoh->state;
	gint one_take_size = postcoh->one_take_size, exe_size = postcoh->exe_size;

	GstFlowReturn ret;

	int c_npeak = 0;
	GstClockTime ts = postcoh->t0 + gst_util_uint64_scale_int_round(postcoh->samples_out, GST_SECOND,
		       	postcoh->rate);


	/* Refresh the detector response U and Dt matrices if reached the refresh interval */
	if (postcoh->refresh_interval > 0 && (ts - postcoh->t_roll_start)/GST_SECOND > (unsigned) postcoh->refresh_interval) {
		postcoh->t_roll_start = ts;
		/* re-read matrices and send them to GPU */
		CUDA_CHECK(cudaSetDevice(postcoh->device_id));
		cuda_postcoh_map_from_xml(postcoh->detrsp_fname, postcoh->state, postcoh->stream);
		GST_DEBUG("detrsp map has been updated");
	}

	LIGOTimeGPS ligo_time;
	XLALINT8NSToGPS(&ligo_time, ts);
	GstClockTime ts_exe_end ;
	while (common_size >= one_take_size) {
		GST_DEBUG_OBJECT(postcoh, "cur time %" GST_TIME_FORMAT ", gps %d, common_size %d, one_take_size %d, exe_size %d\n", GST_TIME_ARGS(ts), ligo_time.gpsSeconds, common_size, one_take_size, exe_size);
		/* expected end time for the run */
		ts_exe_end = postcoh->t0 + gst_util_uint64_scale_int_round(postcoh->samples_out + postcoh->exe_len, GST_SECOND,
		       	postcoh->rate);

		int gps_idx = timestamp_to_gps_idx(state->gps_start, state->gps_step, postcoh->next_exe_t);
		/* copy the snr data to the right location for all detectors */ 
		state->cur_ifo_bits = 0;
		state->cur_nifo = 0;
		for (i=0, collectlist = pads->data; collectlist; collectlist = g_slist_next(collectlist), i++) {
			data = collectlist->data;
			cur_ifo = state->input_ifo_mapping[i];
			state->cur_ifo_is_gap[cur_ifo] = need_flag_gap(data, postcoh->next_exe_t, ts_exe_end);
			state->snglsnr_max[cur_ifo] = 0;
			PeakList *pklist = state->peak_list[cur_ifo];

			if (!state->cur_ifo_is_gap[cur_ifo]) {
				state->cur_ifo_bits += 1 << cur_ifo;
				state->cur_nifo += 1;
			}

#ifdef ACCELERATE_POSTCOH_MEMORY_COPY
			// temporal solution for low memory copy speed
			snglsnr = (COMPLEX_F *) gst_adapter_peek_cuda(data->adapter, one_take_size);
#else 
			snglsnr = (COMPLEX_F *) gst_adapter_peek(data->adapter, one_take_size);
#endif
			c_npeak = peaks_over_thresh(snglsnr, state, cur_ifo, postcoh->stream);

			GST_DEBUG_OBJECT(postcoh, "gps %d, ifo %d, c_npeak %d, max_snglsnr %f\n", ligo_time.gpsSeconds, cur_ifo, c_npeak, state->snglsnr_max[cur_ifo]);

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
			transpose_snglsnr(pklist->d_snglsnr_buffer, state->d_snglsnr[cur_ifo], state->snglsnr_start_load, postcoh->one_take_len, state->snglsnr_len, state->ntmplt, postcoh->stream);

		cudaStreamSynchronize(postcoh->stream);
		}
		cur_ifo = 0;
		for (int iifo = 0; iifo < state->nifo; ++iifo)
			if (!state->cur_ifo_is_gap[iifo]) {
				strncpy(state->cur_ifos + IFO_LEN* cur_ifo, IFOMap[iifo].name, sizeof(char) *IFO_LEN);
				cur_ifo ++;
			}

		g_assert(cur_ifo == state->cur_nifo);

		for (i=0, collectlist = pads->data; collectlist; collectlist = g_slist_next(collectlist), i++) {
			data = collectlist->data;
			cur_ifo = state->input_ifo_mapping[i];

			if ( state->cur_nifo >= 2 && (!state->cur_ifo_is_gap[cur_ifo])) {
				if (state->peak_list[cur_ifo]->npeak[0] > 0) {
					cohsnr_and_chisq(state, cur_ifo, gps_idx, postcoh->output_skymap && state->snglsnr_max[cur_ifo] > postcoh->output_skymap, postcoh->stream);
					GST_LOG("after coherent analysis for ifo %d, npeak %d", cur_ifo, state->peak_list[cur_ifo]->npeak[0]);
				}

			}
			/* move along */
			GST_DEBUG_OBJECT(postcoh, "flush adapter %d, size %d\n", cur_ifo, exe_size);
			gst_adapter_flush(data->adapter, exe_size);

		}
		common_size -= exe_size;
		int exe_len = state->exe_len;
		state->snglsnr_start_load = (state->snglsnr_start_load + exe_len) % state->snglsnr_len;
		state->snglsnr_start_exe = (state->snglsnr_start_exe + exe_len) % state->snglsnr_len;
		postcoh->next_exe_t += exe_len / postcoh->rate * GST_SECOND;
	
		/* make a buffer and send it out */
		cuda_postcoh_new_buffer_and_push(postcoh, exe_len);

		/* move along */
		postcoh->samples_out += exe_len;
	}


}

static GstFlowReturn collected(GstCollectPads *pads, gpointer user_data)
{
	CudaPostcoh* postcoh = CUDA_POSTCOH(user_data);
	PostcohState *state = postcoh->state;
	g_mutex_lock(postcoh->prop_lock);
	while (state->npix == POSTCOH_PARAMS_NOT_INIT || state->autochisq_len == POSTCOH_PARAMS_NOT_INIT || postcoh->hist_trials == POSTCOH_PARAMS_NOT_INIT) {
		g_cond_wait(postcoh->prop_avail, postcoh->prop_lock);
		GST_LOG_OBJECT(postcoh, "collected have to wait for detrsp_map, autocorrelation, and hist_trials to be read");
	}
	g_mutex_unlock(postcoh->prop_lock);

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
		postcoh->t_roll_start = t_latest_start;
		postcoh->next_exe_t = postcoh->t0;
		postcoh->offset0 = offset_latest_start;
		GST_DEBUG_OBJECT(postcoh, "set the aligned time t0 to %" GST_TIME_FORMAT 
				", start offset0 to %" G_GUINT64_FORMAT,
			       	GST_TIME_ARGS(postcoh->t0),
				postcoh->offset0);
		postcoh->set_starttime = TRUE;
	}
	
	if (postcoh->is_all_aligned) {
		/* first fill in any discontinuity */
		cuda_postcoh_fillin_discont(pads, postcoh);

		/* if buf in any of pads is 0 size, discard this buf.
		 * push the buf in adapter if it is too small
		 * this means this element starts to work only when
		 * there are non-zero buffers in all pads */
		if (cuda_postcoh_need_recollect(pads, postcoh))
			return GST_FLOW_OK;

		common_size = cuda_postcoh_push_and_get_common_size(pads, postcoh);
		GST_DEBUG_OBJECT(postcoh, "get spanned size %d, get spanned samples %f", common_size, (float)common_size/ postcoh->bps);

		if (common_size == -1) {
			res = gst_pad_push_event(postcoh->srcpad, gst_event_new_eos());
			return res;
		}

		cuda_postcoh_process(pads, common_size, postcoh);

	} else {
		postcoh->is_all_aligned = cuda_postcoh_align_collected(pads, postcoh);
	}
	return GST_FLOW_OK;
}


static void cuda_postcoh_dispose(GObject *object)
{
	CudaPostcoh *element = CUDA_POSTCOH(object);
	if(element->collect)
		gst_object_unref(GST_OBJECT(element->collect));
	element->collect = NULL;

	if(element->state){
		state_destroy(element->state);
		free(element->state);
		element->state = NULL;
	}

	if(element->srcpad)
		gst_object_unref(element->srcpad);
	element->srcpad = NULL;

	g_mutex_free(element->prop_lock);
	g_cond_free(element->prop_avail);

	/* destroy hashtable and its contents */
	G_OBJECT_CLASS(parent_class)->dispose(object);
}


static void cuda_postcoh_base_init(gpointer g_class)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(g_class);

	gst_element_class_set_details_simple(
		element_class,
		"Post Coherent SNR and Nullstream Generator",
		"Filter",
		"Coherent trigger generation.\n",
		"Qi Chu <qi.chu at ligo dot org>"
	);
	gst_element_class_add_pad_template(
		element_class,
		gst_static_pad_template_get(&cuda_postcoh_sink_template)
	);

	gst_element_class_add_pad_template(
		element_class,
//		gst_static_pad_template_get(&cuda_postcoh_src_template)
		gst_pad_template_new(
			"src",
			GST_PAD_SRC,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
				"application/x-lal-postcoh" 
			)
		)
	);
}


static void cuda_postcoh_class_init(CudaPostcohClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *gstelement_class = GST_ELEMENT_CLASS(klass);

	parent_class = g_type_class_ref(GST_TYPE_ELEMENT);

	gobject_class->get_property = GST_DEBUG_FUNCPTR(cuda_postcoh_get_property);
	gobject_class->set_property = GST_DEBUG_FUNCPTR(cuda_postcoh_set_property);
	gobject_class->dispose = GST_DEBUG_FUNCPTR(cuda_postcoh_dispose);
	gstelement_class->request_new_pad = GST_DEBUG_FUNCPTR(cuda_postcoh_request_new_pad);
	gstelement_class->release_pad = GST_DEBUG_FUNCPTR(cuda_postcoh_release_pad);
	gstelement_class->change_state = GST_DEBUG_FUNCPTR(cuda_postcoh_change_state);

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
			0, G_MAXINT, 0,
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

	g_object_class_install_property(
		gobject_class,
		PROP_REFRESH_INTERVAL,
		g_param_spec_int(
			"detrsp-refresh-interval",
			"detector response refresh interval",
			"(0) never refresh stats; (N) refresh stats every N seconds. ",
			0, G_MAXINT, 600,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);


}


static void cuda_postcoh_init(CudaPostcoh *postcoh, CudaPostcohClass *klass)
{
	GstElement *element = GST_ELEMENT(postcoh);

	gst_element_create_all_pads(element);
	postcoh->srcpad = gst_element_get_static_pad(element, "src");
	GST_DEBUG_OBJECT(postcoh, "%s caps %" GST_PTR_FORMAT, GST_PAD_NAME(postcoh->srcpad), gst_pad_get_caps(postcoh->srcpad));

	gst_pad_set_event_function(postcoh->srcpad, GST_DEBUG_FUNCPTR(src_event));
	postcoh->collect = gst_collect_pads_new();
	gst_collect_pads_set_function(postcoh->collect, GST_DEBUG_FUNCPTR(collected), postcoh);
	
 

	postcoh->t0 = GST_CLOCK_TIME_NONE;
	postcoh->next_exe_t = GST_CLOCK_TIME_NONE;
	postcoh->offset0 = GST_BUFFER_OFFSET_NONE;
	//postcoh->next_in_offset = GST_BUFFER_OFFSET_NONE;
	postcoh->set_starttime = FALSE;
	postcoh->is_all_aligned = FALSE;
	postcoh->samples_in = 0;
	postcoh->samples_out = 0;
	postcoh->state = (PostcohState *) malloc (sizeof(PostcohState));
	postcoh->state->autochisq_len = POSTCOH_PARAMS_NOT_INIT;
	postcoh->state->npix = POSTCOH_PARAMS_NOT_INIT;
	postcoh->state->is_member_init = POSTCOH_PARAMS_NOT_INIT;
	postcoh->hist_trials = POSTCOH_PARAMS_NOT_INIT;
	postcoh->prop_lock = g_mutex_new();
	postcoh->prop_avail = g_cond_new();
	postcoh->stream_id = POSTCOH_PARAMS_NOT_INIT;
	postcoh->device_id = POSTCOH_PARAMS_NOT_INIT;
	postcoh->process_id = 0;
	postcoh->cur_event_id = 0;
	postcoh->t_roll_start = GST_CLOCK_TIME_NONE;
	postcoh->refresh_interval = 0;
}



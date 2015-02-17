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

#include <string.h>
#include <math.h>
#include "postcoh.h"
#include "postcoh_utils.h"
#include "postcoh_table_utils.h"

#define GST_CAT_DEFAULT gstlal_postcoh_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);

#define DEFAULT_DETRSP_FNAME "L1H1V1_detrsp.xml"
#define EPSILON 1  

static void additional_initializations(GType type)
{
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "cuda_postcoh", 0, "cuda_postcoh element");
}


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
	PROP_AUTOCORRELATION_FNAME,
	PROP_HIST_TRIALS,
	PROP_OUTPUT_SKYMAP,
	PROP_SNGLSNR_THRESH
};


static void cuda_postcoh_set_property(GObject *object, guint id, const GValue *value, GParamSpec *pspec)
{
	CudaPostcoh *element = CUDA_POSTCOH(object);

	GST_OBJECT_LOCK(element);
	switch(id) {
		case PROP_DETRSP_FNAME:
			g_mutex_lock(element->prop_lock);
			element->detrsp_fname = g_value_dup_string(value);
			cuda_postcoh_map_from_xml(element->detrsp_fname, element->state);
			g_cond_broadcast(element->prop_avail);
			g_mutex_unlock(element->prop_lock);
			break;

		case PROP_AUTOCORRELATION_FNAME: 

			g_mutex_lock(element->prop_lock);
			element->autocorr_fname = g_value_dup_string(value);
			cuda_postcoh_autocorr_from_xml(element->autocorr_fname, element->state);
			g_cond_broadcast(element->prop_avail);
			g_mutex_unlock(element->prop_lock);
			break;

		case PROP_HIST_TRIALS:
			g_mutex_lock(element->prop_lock);
			element->hist_trials = g_value_get_int(value);
			g_cond_broadcast(element->prop_avail);
			g_mutex_unlock(element->prop_lock);
			break;

		case PROP_OUTPUT_SKYMAP:
			element->output_skymap = g_value_get_int(value);
			break;


		case PROP_SNGLSNR_THRESH:
			element->snglsnr_thresh = g_value_get_float(value);
			element->state->snglsnr_thresh = element->snglsnr_thresh;
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

		case PROP_AUTOCORRELATION_FNAME:
			g_value_set_string(value, element->autocorr_fname);
			break;

		case PROP_HIST_TRIALS:
			g_value_set_int(value, element->hist_trials);
			break;

		case PROP_OUTPUT_SKYMAP:
			g_value_set_int(value, element->output_skymap);
			break;

		case PROP_SNGLSNR_THRESH:
			g_value_set_float(value, element->snglsnr_thresh);
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
	CudaPostcoh *postcoh = CUDA_POSTCOH(GST_PAD_PARENT(pad));
	PostcohState *state = postcoh->state;
	g_mutex_lock(postcoh->prop_lock);
	while (!state->npix || !state->autochisq_len || postcoh->hist_trials == -1) {
		g_cond_wait(postcoh->prop_avail, postcoh->prop_lock);
		GST_LOG_OBJECT(postcoh, "setcaps have to wait");
	}
	g_mutex_unlock(postcoh->prop_lock);

	GList *sinkpads;
	GstStructure *s;
	GstPostcohCollectData *data;

	/* FIXME: this is copied from gstadder.c. Replace with new version of that file
	 * if any. */
	GST_OBJECT_LOCK(postcoh);
	sinkpads = GST_ELEMENT(postcoh)->sinkpads;
//	sinkpads = GST_ELEMENT(postcoh)->pads;

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

	GST_DEBUG_OBJECT(postcoh, "setting GstPostcohCollectData offset_per_nanosecond %f and channels", postcoh->offset_per_nanosecond);

	state->nifo = GST_ELEMENT(postcoh)->numsinkpads;
	state->ifo_mapping = (gint8 *)malloc(sizeof(gint8) * state->nifo);
	state->peak_list = (PeakList **)malloc(sizeof(PeakList*) * state->nifo);
	state->dt = (float) 1/postcoh->rate;

	/* need to cover head and tail */
	postcoh->preserved_len = state->autochisq_len + 160; 
	postcoh->exe_len = postcoh->rate;

	state->head_len = postcoh->preserved_len / 2;
	state->snglsnr_len = postcoh->preserved_len + postcoh->exe_len + postcoh->hist_trials * postcoh->exe_len;
	state->hist_trials = postcoh->hist_trials;
	state->snglsnr_start_load = postcoh->hist_trials * postcoh->exe_len;
	state->snglsnr_start_exe = state->snglsnr_start_load + state->head_len;

	GST_DEBUG_OBJECT(postcoh, "hist_trials %d, autochisq_len %d, preserved_len %d, sngl_len %d, start_load %d, start_exe %d", state->hist_trials, state->autochisq_len, postcoh->preserved_len, state->snglsnr_len, state->snglsnr_start_load, state->snglsnr_start_exe);

	state->ntmplt = postcoh->channels/2;
	state->exe_len = postcoh->rate;
	cudaMalloc((void **)&(state->dd_snglsnr), sizeof(COMPLEX_F *) * state->nifo);
	state->d_snglsnr = (COMPLEX_F **)malloc(sizeof(COMPLEX_F *) * state->nifo);

	gint8 i = 0, j = 0, cur_ifo = 0;
	GST_OBJECT_LOCK(postcoh->collect);

	/* initialize ifo_mapping, snglsnr matrix, and peak_list */
	for (sinkpads = GST_ELEMENT(postcoh)->sinkpads; sinkpads; sinkpads = g_list_next(sinkpads), i++) {
		GstPad *pad = GST_PAD(sinkpads->data);
		data = gst_pad_get_element_private(pad);
		set_offset_per_nanosecond(data, postcoh->offset_per_nanosecond);
		set_channels(data, postcoh->channels);
		for (j=0; j<state->nifo; j++) {
			if (strncmp(data->ifo_name, IFO_MAP[j], 2) == 0 ) {
				state->ifo_mapping[i] = j;
				cur_ifo = j;
				break;
			}
		}
		
		guint mem_alloc_size = state->snglsnr_len * postcoh->bps;
	       	cudaMalloc((void**) &(state->d_snglsnr[cur_ifo]), mem_alloc_size);
		cudaMemset(state->d_snglsnr[cur_ifo], 0, mem_alloc_size);
		cudaMemcpy(&(state->dd_snglsnr[cur_ifo]), &(state->d_snglsnr[cur_ifo]), sizeof(COMPLEX_F *), cudaMemcpyHostToDevice);

		state->peak_list[cur_ifo] = create_peak_list(postcoh->state);
	}

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

	data->ifo_name = (gchar *)malloc(2*sizeof(gchar));
	strncpy(data->ifo_name, name, 2*sizeof(gchar));
	data->adapter = gst_adapter_new();
	data->is_aligned = FALSE;
	data->aligned_offset0 = 0;
	GST_DEBUG_OBJECT(element, "new pad for %s is added and initialised", data->ifo_name);

	return GST_PAD(newpad);
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

	}
	return TRUE;
}

static gint cuda_postcoh_push_and_get_common_size(GstCollectPads *pads)
{
	GSList *collectlist;
	GstPostcohCollectData *data;
	GstBuffer *buf = NULL;

	gint min_size = 0, size_cur, null_bufs = 0;
	gboolean min_size_init = FALSE;

	for (collectlist = pads->data; collectlist; collectlist = g_slist_next(collectlist)) {
			data = collectlist->data;
			buf = gst_collect_pads_pop(pads, (GstCollectData *)data);
			if (buf == NULL) {
				null_bufs++;
				continue;
			}
			GST_LOG_OBJECT (data,
				"Push buffer to adapter of (%u bytes) with timestamp %" GST_TIME_FORMAT ", duration %"
				GST_TIME_FORMAT ", offset %" G_GUINT64_FORMAT ", offset_end %"
				G_GUINT64_FORMAT,  GST_BUFFER_SIZE (buf),
				GST_TIME_ARGS (GST_BUFFER_TIMESTAMP (buf)),
				GST_TIME_ARGS (GST_BUFFER_DURATION (buf)),
				GST_BUFFER_OFFSET (buf), GST_BUFFER_OFFSET_END (buf));

			gst_adapter_push(data->adapter, buf);
			size_cur = gst_adapter_available(data->adapter);
			if(!min_size_init) {
				min_size = size_cur;
				min_size_init = TRUE;
			} else {
				if (min_size > size_cur) {
					min_size = size_cur;
				}
			}

			
	}
	/* If two or more pads returns NULL buffers, this means two or more pads at EOS,
	 * we flag min_size as -1 to indicate we need to send an EOS event */ 
	if (null_bufs >= 2)
		min_size = -1;
	return min_size;
}

static gboolean cuda_postcoh_align_collected(GstCollectPads *pads, GstClockTime t0)
{
	GSList *collectlist;
	GstPostcohCollectData *data;
	GstBuffer *buf, *subbuf;
	GstClockTime t_start_cur, t_end_cur;
	gboolean all_aligned = TRUE;
	guint64 offset_cur, offset_end_cur, buf_aligned_offset0;

	GST_DEBUG_OBJECT(pads, "begin to align offset0");

	for (collectlist = pads->data; collectlist; collectlist = g_slist_next(collectlist)) {
		data = collectlist->data;
		GST_DEBUG_OBJECT(pads, "now at %s is aligned %d", data->ifo_name, data->is_aligned);
		if (data->is_aligned) {
			buf = gst_collect_pads_pop(pads, (GstCollectData *)data);
			gst_adapter_push(data->adapter, buf);
			continue;
		}
		buf = gst_collect_pads_pop(pads, (GstCollectData *)data);
		t_start_cur = GST_BUFFER_TIMESTAMP(buf);
		t_end_cur = t_start_cur + GST_BUFFER_DURATION(buf);
		offset_cur = GST_BUFFER_OFFSET(buf);
		offset_end_cur = GST_BUFFER_OFFSET_END(buf);
		if (t_end_cur > t0) {
			buf_aligned_offset0 = gst_util_uint64_scale_int(GST_CLOCK_DIFF(t0, t_start_cur), data->offset_per_nanosecond, 1);
			GST_DEBUG_OBJECT(data, "buffer aligned offset0 %u", buf_aligned_offset0);
			subbuf = gst_buffer_create_sub(buf, buf_aligned_offset0, (offset_end_cur - offset_cur - buf_aligned_offset0) * data->channels * sizeof(float));
			GST_LOG_OBJECT (pads,
				"Created sub buffer of (%u bytes) with timestamp %" GST_TIME_FORMAT ", duration %"
				GST_TIME_FORMAT ", offset %" G_GUINT64_FORMAT ", offset_end %"
				G_GUINT64_FORMAT,  GST_BUFFER_SIZE (subbuf),
				GST_TIME_ARGS (GST_BUFFER_TIMESTAMP (subbuf)),
				GST_TIME_ARGS (GST_BUFFER_DURATION (subbuf)),
				GST_BUFFER_OFFSET (subbuf), GST_BUFFER_OFFSET_END (subbuf));
			gst_adapter_push(data->adapter, subbuf);
			data->is_aligned = TRUE;
			set_aligned_offset0(data, buf_aligned_offset0 + offset_cur);
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
static void cuda_postcoh_flush(GstCollectPads *pads, guint64 common_size)
{
	GSList *collectlist;
	GstPostcohCollectData *data;

	for (collectlist = pads->data; collectlist; collectlist = g_slist_next(collectlist)) {
		data = collectlist->data;
		gst_adapter_flush(data->adapter, common_size);
	}

}

static	void cuda_postcoh_rm_invalid_peak(PostcohState *state)
{
	int iifo, ipeak, npeak, nifo = state->nifo, final_peaks = 0, tmp_peak_pos[state->exe_len];
	for(iifo=0; iifo<nifo; iifo++) {
		final_peaks = 0;
		PeakList *pklist = state->peak_list[iifo];
		npeak = pklist->npeak[0];
		int *peak_pos = pklist->peak_pos;
		for(ipeak=0; ipeak<npeak; ipeak++) {
			/* if the difference of maximum single snr and coherent snr is ignorable,
			 * it means that only one detector is in action,
			 * we abandon this peak
			 * */
			if (abs(pklist->maxsnglsnr[peak_pos[ipeak]]- pklist->cohsnr[ipeak]) > EPSILON) {
				tmp_peak_pos[final_peaks++] = peak_pos[ipeak];
			}

		}

		npeak = final_peaks;
		memcpy(peak_pos, tmp_peak_pos, sizeof(int) * npeak);
		pklist->npeak[0] = npeak;
	}

}

static GstBuffer* cuda_postcoh_new_buffer(CudaPostcoh *postcoh, gint out_len)
{
	GstBuffer *outbuf = NULL;
	GstPad *srcpad = postcoh->srcpad;
	GstCaps *caps = GST_PAD_CAPS(srcpad);
	GstFlowReturn ret;
	PostcohState *state = postcoh->state;

	cuda_postcoh_rm_invalid_peak(state);
	int allnpeak = 0, iifo, ipeak, nifo = state->nifo;
	for(iifo=0; iifo<nifo; iifo++)
		allnpeak += state->peak_list[iifo]->npeak[0];

	int hist_trials = postcoh->hist_trials;
	int out_size = sizeof(PostcohTable) * allnpeak * (1 + hist_trials);

	ret = gst_pad_alloc_buffer(srcpad, 0, out_size, caps, &outbuf);
	if (ret != GST_FLOW_OK) {
		GST_ERROR_OBJECT(srcpad, "Could not allocate postcoh-inspiral buffer %d", ret);
		return NULL;
	}

        /* set the time stamps */
	GstClockTime ts = postcoh->out_t0 + gst_util_uint64_scale_int_round(postcoh->samples_out, GST_SECOND,
		       	postcoh->rate);

        GST_BUFFER_TIMESTAMP(outbuf) = ts;
	GST_BUFFER_DURATION(outbuf) = (GstClockTime) gst_util_uint64_scale_int_round(GST_SECOND, out_len, postcoh->rate);

	/* set the offset */
        GST_BUFFER_OFFSET(outbuf) = postcoh->out_offset0 + postcoh->samples_out;
        GST_BUFFER_OFFSET_END(outbuf) = GST_BUFFER_OFFSET(outbuf) + out_len;


	PostcohTable *output = (PostcohTable *) GST_BUFFER_DATA(outbuf);
	int ifos_size = sizeof(char) * 2 * nifo, one_ifo_size = sizeof(char) * 2 ;
	char *ifos = (char *) malloc(ifos_size);

	for(iifo=0; iifo<nifo; iifo++) 
		strncpy(ifos+2*iifo, IFO_MAP[iifo], one_ifo_size);

	int npeak = 0, itrial = 0, exe_len = state->exe_len;
	for(iifo=0; iifo<nifo; iifo++) {
		PeakList *pklist = state->peak_list[iifo];
		npeak = pklist->npeak[0];
		LIGOTimeGPS end_time[npeak];

		GST_LOG_OBJECT(postcoh, "write to output, ifo %d, npeak %d", iifo, npeak);
		int peak_cur;
		for(ipeak=0; ipeak<npeak; ipeak++) {
			XLALINT8NSToGPS(&(end_time[ipeak]), ts);
			int *peak_pos = pklist->peak_pos;
			peak_cur = peak_pos[ipeak];
			XLALGPSAdd(&(end_time[ipeak]), (double) peak_cur /exe_len);
			output->end_time = end_time[ipeak];
			output->is_background = 0;
			strncpy(output->ifos, ifos, ifos_size);
			output->ifos[2*nifo] = '\0';
		       	strncpy(output->pivotal_ifo, IFO_MAP[iifo], one_ifo_size);
			output->pivotal_ifo[2] = '\0';
			output->tmplt_idx = pklist->tmplt_idx[peak_cur];
			output->pix_idx = pklist->pix_idx[peak_cur];
			output->maxsnglsnr = pklist->maxsnglsnr[peak_cur];
			output->cohsnr = pklist->cohsnr[peak_cur];
			output->nullsnr = pklist->nullsnr[peak_cur];
			output->chisq = pklist->chisq[peak_cur];
			if (postcoh->output_skymap) {
			GString *filename = NULL;
			FILE *file = NULL;
			filename = g_string_new(output->ifos);
			g_string_append_printf(filename, "_%s_%d_%d", output->pivotal_ifo, output->end_time.gpsSeconds, output->end_time.gpsNanoSeconds);
			g_string_append_printf(filename, "_%d_skymap.txt", output->tmplt_idx);
			strcpy(output->skymap_fname, filename->str);
//			printf("file %s is written, skymap addr %p\n", output->skymap_fname, &(pklist->cohsnr_skymap[ipeak * state->npix]));
			file = fopen(output->skymap_fname, "w");
			fwrite(&(pklist->cohsnr_skymap[ipeak * state->npix]), sizeof(float), state->npix, file);
			fclose(file);
			file = NULL;
			g_string_free(filename, TRUE);
			} else
				output->skymap_fname[0] = '\0';
			output++;
		}

		if (pklist->d_cohsnr_skymap) {
			cudaFree(pklist->d_cohsnr_skymap);
			pklist->d_cohsnr_skymap = NULL;
		}

		if (pklist->cohsnr_skymap) {
			free(pklist->cohsnr_skymap);
			pklist->cohsnr_skymap = NULL;
		}

		for(itrial=1; itrial<=hist_trials; itrial++) {
			for(ipeak=0; ipeak<npeak; ipeak++) {
				int *peak_pos = pklist->peak_pos;
				peak_cur = peak_pos[ipeak];
				output->end_time = end_time[ipeak];
				output->is_background = 1;
				strncpy(output->ifos, ifos, ifos_size);
				output->ifos[2*nifo] = '\0';
		       		strncpy(output->pivotal_ifo, IFO_MAP[iifo], one_ifo_size);
				output->pivotal_ifo[2] = '\0';
				output->tmplt_idx = pklist->tmplt_idx[peak_cur];
				output->pix_idx = pklist->pix_idx[peak_cur];
				output->maxsnglsnr = pklist->maxsnglsnr[peak_cur];
				output->cohsnr = pklist->cohsnr[itrial*exe_len + peak_cur];
				output->nullsnr = pklist->nullsnr[itrial*exe_len + peak_cur];
				output->chisq = pklist->chisq[itrial*exe_len + peak_cur];
				output->skymap_fname[0] ='\0';

				output++;
			}
		}
	}


	GST_LOG_OBJECT (srcpad,
		"Processed of (%u bytes) with timestamp %" GST_TIME_FORMAT ", duration %"
		GST_TIME_FORMAT ", offset %" G_GUINT64_FORMAT ", offset_end %"
		G_GUINT64_FORMAT,  GST_BUFFER_SIZE (outbuf),
		GST_TIME_ARGS (GST_BUFFER_TIMESTAMP (outbuf)),
		GST_TIME_ARGS (GST_BUFFER_DURATION (outbuf)),
		GST_BUFFER_OFFSET (outbuf), GST_BUFFER_OFFSET_END (outbuf));

	return outbuf;
}

int timestamp_to_gps_idx(int gps_step, GstClockTime t)
{
	int seconds_in_one_day = 24 * 3600;
	unsigned long days_from_utc0 = (t / GST_SECOND) / seconds_in_one_day;
	int gps_len = seconds_in_one_day / gps_step;
	double time_in_one_day = (double) (t/GST_SECOND) - days_from_utc0 * seconds_in_one_day;
	int gps_idx = (int) (round( time_in_one_day / gps_step)) % gps_len;

//	printf("days_from_utc0 %lu, time_in_one_day %f, gps_len %d, gps_idx %d,\n", days_from_utc0, time_in_one_day, gps_len, gps_idx);
	return gps_idx;
}

static void cuda_postcoh_process(GstCollectPads *pads, gint common_size, gint one_take_size, gint exe_size, CudaPostcoh *postcoh)
{
	GSList *collectlist;
	GstPostcohCollectData *data;
	COMPLEX_F *snglsnr, *pos_dd_snglsnr, *pos_in_snglsnr;
	gint one_take_len = one_take_size / postcoh->bps;

	int i = 0, cur_ifo = 0;
	PostcohState *state = postcoh->state;

	GstFlowReturn ret;

	while (common_size >= one_take_size) {
		int gps_idx = timestamp_to_gps_idx(state->gps_step, postcoh->next_t);
		/* copy the snr data to the right location for all detectors */ 
		for (i=0, collectlist = pads->data; collectlist; collectlist = g_slist_next(collectlist), i++) {
			data = collectlist->data;
			cur_ifo = state->ifo_mapping[i];
			snglsnr = (COMPLEX_F *) gst_adapter_peek(data->adapter, one_take_size);
//			printf("auto_len %d, npix %d\n", state->autochisq_len, state->npix);
			pos_dd_snglsnr = state->d_snglsnr[cur_ifo] + state->snglsnr_start_load * state->ntmplt;
			/* copy the snglsnr to the right cuda memory */
			if(state->snglsnr_start_load + one_take_len <= state->snglsnr_len){
				/* when the snglsnr can be put in as one chunk */
				cudaMemcpy(pos_dd_snglsnr, snglsnr, one_take_size, cudaMemcpyHostToDevice);
			} else {

				int tail_cpy_size = (state->snglsnr_len - state->snglsnr_start_load) * postcoh->bps;
				cudaMemcpy(pos_dd_snglsnr, snglsnr, tail_cpy_size, cudaMemcpyHostToDevice);
				int head_cpy_size = one_take_size - tail_cpy_size;
				pos_dd_snglsnr = state->d_snglsnr[cur_ifo];
				pos_in_snglsnr = snglsnr + (state->snglsnr_len - state->snglsnr_start_load) * state->ntmplt;
				cudaMemcpy(pos_dd_snglsnr, pos_in_snglsnr, head_cpy_size, cudaMemcpyHostToDevice);
			}

		}
		for (i=0, collectlist = pads->data; collectlist; collectlist = g_slist_next(collectlist), i++) {

			data = collectlist->data;
			cur_ifo = state->ifo_mapping[i];

			peakfinder(state, cur_ifo);
			cudaMemcpy(	state->peak_list[cur_ifo]->npeak, 
					state->peak_list[cur_ifo]->d_npeak, 
					sizeof(int), 
					cudaMemcpyDeviceToHost);

			if (state->peak_list[cur_ifo]->npeak[0] > 0) {
				cohsnr_and_chisq(state, cur_ifo, gps_idx, postcoh->output_skymap);
			}

			/* move along */
			gst_adapter_flush(data->adapter, exe_size);
		}
		common_size -= exe_size;
		int exe_len = state->exe_len;
		state->snglsnr_start_load = (state->snglsnr_start_load + exe_len) % state->snglsnr_len;
		state->snglsnr_start_exe = (state->snglsnr_start_exe + exe_len) % state->snglsnr_len;
		postcoh->next_t += exe_len / postcoh->rate * GST_SECOND;

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
	g_mutex_lock(postcoh->prop_lock);
	while (!state->npix || !state->autochisq_len || postcoh->hist_trials == -1) {
		g_cond_wait(postcoh->prop_avail, postcoh->prop_lock);
		GST_LOG_OBJECT(postcoh, "collected have to wait");
	}
	g_mutex_unlock(postcoh->prop_lock);

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
		postcoh->in_t0 = t_latest_start;
		postcoh->out_t0 = t_latest_start + gst_util_uint64_scale_int_round(
				postcoh->preserved_len/2, GST_SECOND, postcoh->rate);
		postcoh->next_t = postcoh->out_t0;
		postcoh->out_offset0 = offset_latest_start + postcoh->preserved_len/2 ;
		GST_DEBUG_OBJECT(postcoh, "set the aligned time to %" GST_TIME_FORMAT 
				", out t0 to %" GST_TIME_FORMAT ", start offset to %" G_GUINT64_FORMAT,
			       	GST_TIME_ARGS(postcoh->in_t0),
			       	GST_TIME_ARGS(postcoh->out_t0),
				postcoh->out_offset0);
		postcoh->is_all_aligned = cuda_postcoh_align_collected(pads, postcoh->in_t0);
		postcoh->set_starttime = TRUE;
		return GST_FLOW_OK;
	}

	gint exe_size = postcoh->exe_len * postcoh->bps;
		
	if (postcoh->is_all_aligned) {
		common_size = cuda_postcoh_push_and_get_common_size(pads);
		GST_DEBUG_OBJECT(postcoh, "get spanned size %d, get spanned samples %f", common_size, common_size/ postcoh->bps);

		if (common_size == -1) {
			res = gst_pad_push_event(postcoh->srcpad, gst_event_new_eos());
			return res;
		}

		gint one_take_size = postcoh->preserved_len * postcoh->bps + exe_size;
		cuda_postcoh_process(pads, common_size, one_take_size, exe_size, postcoh);

	} else {
		postcoh->is_all_aligned = cuda_postcoh_align_collected(pads, postcoh->in_t0);
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
	if(element->collect)
		gst_object_unref(GST_OBJECT(element->collect));
	element->collect = NULL;

	if(element->state){
		state_destroy(element->state);
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
#if 1
		gst_pad_template_new(
			"src",
			GST_PAD_SRC,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
				"application/x-lal-postcoh" 
			)
		)
#endif
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
		PROP_AUTOCORRELATION_FNAME,
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
		PROP_HIST_TRIALS,
		g_param_spec_int(
			"hist-trials",
			"history trials",
			"history that should be kept in seconds",
			0, G_MAXINT, 1,
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
		PROP_SNGLSNR_THRESH,
		g_param_spec_float(
			"snglsnr-thresh",
			"single snr threshold",
			"single snr threshold",
			0.0, G_MAXFLOAT, 4.0,
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

	postcoh->in_t0 = GST_CLOCK_TIME_NONE;
	postcoh->out_t0 = GST_CLOCK_TIME_NONE;
	postcoh->next_t = GST_CLOCK_TIME_NONE;
	postcoh->out_offset0 = GST_BUFFER_OFFSET_NONE;
	//postcoh->next_in_offset = GST_BUFFER_OFFSET_NONE;
	postcoh->set_starttime = FALSE;
	postcoh->is_all_aligned = FALSE;
	postcoh->samples_in = 0;
	postcoh->samples_out = 0;
	postcoh->state = (PostcohState *) malloc (sizeof(PostcohState));
	postcoh->state->autochisq_len = 0;
	postcoh->state->npix = 0;
	postcoh->hist_trials = -1;
	postcoh->prop_lock = g_mutex_new();
	postcoh->prop_avail = g_cond_new();
}



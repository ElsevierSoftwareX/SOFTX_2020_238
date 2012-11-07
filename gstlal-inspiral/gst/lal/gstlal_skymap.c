/*
 * Copyright (C) 2009 Leo Singer <leo.singer@ligo.org>, Antony Searle <antony.searle@ligo.caltech.edu>
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



#include <gst/gst.h>
#include <gst/base/gstadapter.h>
#include <gstlal/gstlal.h>
#include <gstlal/gstlal_tags.h>
#include <gstlal_skymap.h>
#include <lal/Date.h>
#include <lal/LIGOMetadataTables.h>
#include <lal/LIGOLwXMLInspiralRead.h>
#include <math.h>


#define GST_CAT_DEFAULT gstlal_skymap_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


static GstElementClass *parent_class = NULL;



/*
 * ============================================================================
 *
 *                              Property Support
 *
 * ============================================================================
 */


enum gen_property {
	ARG_DT = 1,
	ARG_TRIGGER_PRESENT_PADDING,
	ARG_TRIGGER_ABSENT_PADDING,
	ARG_BANK_FILENAME
};


static void free_bankfile(GSTLALSkymap *element)
{
	g_free(element->bank_filename);
	element->bank_filename = NULL;

	SnglInspiralTable* head = element->bank;
	while (head != NULL)
	{
		SnglInspiralTable* next = head->next;
		head->next = NULL;
		free(head);
		head = next;
	}
	element->bank = NULL;
}


static int setup_bankfile_input(GSTLALSkymap *element, char *bank_filename)
{
	free_bankfile(element);

	element->bank_filename = bank_filename;

	/* Why the **heck** is the return value for this function undocumented in the header file? How do I tell if this call failed? */
	element->ntemplates = LALSnglInspiralTableFromLIGOLw(&element->bank, element->bank_filename, -1, -1);

	return element->ntemplates;
}


static void set_property(GObject *object, enum gen_property id, const GValue *value, GParamSpec *pspec)
{
	GSTLALSkymap *element = GSTLAL_SKYMAP(object);

	GST_OBJECT_LOCK(element);
	switch(id) {
	case ARG_DT:
		element->dt = g_value_get_uint64(value);
		break;

	case ARG_TRIGGER_PRESENT_PADDING:
		element->trigger_present_padding = g_value_get_uint64(value);
		break;

	case ARG_TRIGGER_ABSENT_PADDING:
		element->trigger_absent_padding = g_value_get_uint64(value);
		break;

	case ARG_BANK_FILENAME:
		g_mutex_lock(element->bank_lock);
		setup_bankfile_input(element, g_value_dup_string(value));
		g_mutex_unlock(element->bank_lock);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}
	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject * object, enum gen_property id, GValue * value, GParamSpec * pspec)
{
	GSTLALSkymap *element = GSTLAL_SKYMAP(object);

	GST_OBJECT_LOCK(element);
	switch(id) {
	case ARG_DT:
		g_value_set_uint64(value, element->dt);
		break;

	case ARG_TRIGGER_PRESENT_PADDING:
		g_value_set_uint64(value, element->trigger_present_padding);
		break;

	case ARG_TRIGGER_ABSENT_PADDING:
		g_value_set_uint64(value, element->trigger_absent_padding);
		break;

	case ARG_BANK_FILENAME:
		g_mutex_lock(element->bank_lock);
		g_value_set_string(value, element->bank_filename);
		g_mutex_unlock(element->bank_lock);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}
	GST_OBJECT_UNLOCK(element);
}


/*
 * ============================================================================
 *
 *                              Pad Management
 *
 * ============================================================================
 */


typedef struct {
	GstCollectData gstcollectdata;
	GstClockTime last_end_time;
} GstSkymapCollectData;


typedef struct {
	GstSkymapCollectData skymapcollectdata;
	GstAdapter* adapter;
	gchar* instrument;
} GstSkymapSnrCollectData;


typedef struct {
	GstSkymapCollectData skymapcollectdata;
	GstBuffer* last_buffer;
} GstSkymapCoincCollectData;


void gst_skymap_snr_collectdata_destroy(GstSkymapSnrCollectData* collectdata)
{
	g_free(collectdata->instrument);
	collectdata->instrument = NULL;
	g_object_unref(collectdata->adapter);
	collectdata->adapter = NULL;
}


void gst_skymap_coinc_collectdata_destroy(GstSkymapCoincCollectData* collectdata)
{
	if (collectdata->last_buffer)
	{
		gst_buffer_unref(collectdata->last_buffer);
		collectdata->last_buffer = NULL;
	}
}


gint snr_collectdata_is_instrument(const GstSkymapSnrCollectData* collectdata, const gchar* instrument)
{
	if (collectdata->instrument == NULL)
		return -1;
	else
		return strcmp(collectdata->instrument, instrument);
}


gint snr_collectdata_is_pad(const GstCollectData* collectdata, const GstPad* pad)
{
	if (collectdata->pad == pad)
		return 0;
	else
		return -1;
}


static gboolean snr_event(GstPad *pad, GstEvent *event)
{
	GSTLALSkymap *element = GSTLAL_SKYMAP(GST_PAD_PARENT(pad));

	if (GST_EVENT_TYPE(event) == GST_EVENT_TAG)
	{
		GstTagList* taglist;
		gchar* instrument;
		gst_event_parse_tag(event, &taglist);

		if (gst_tag_list_get_string(taglist, GSTLAL_TAG_INSTRUMENT, &instrument))
		{
			GSList* found_same_named_pad = g_slist_find_custom(element->snr_collectdatas, instrument, (GCompareFunc)snr_collectdata_is_instrument);
			if (found_same_named_pad && ((GstCollectData*)(found_same_named_pad->data))->pad != pad)
			{
				GST_ELEMENT_ERROR(element, CORE, TAG, ("two pads provided tags designating the instrument \"%s\"", instrument), (NULL));
				g_free(instrument);
			} else {
				GSList* found_item = g_slist_find_custom(element->snr_collectdatas, pad, (GCompareFunc)snr_collectdata_is_pad);
				((GstSkymapSnrCollectData*)(found_item->data))->instrument = instrument;
			}
		}
	}

	/* Now let collectpads handle the event. */
	return element->collect_event(pad, event);
}


static void update_sink_caps(GSTLALSkymap* skymap)
{
	GstCaps* caps = gst_caps_new_simple("application/x-lal-snglinspiral", "channels", G_TYPE_INT, GST_ELEMENT(skymap)->numsinkpads - 1, NULL);
	g_assert(caps);

	gst_pad_set_caps(skymap->coinc_collectdata->pad, caps);
}


static GstPad *request_new_pad(GstElement *element, GstPadTemplate *templ, const gchar *name)
{
	GSTLALSkymap* skymap = GSTLAL_SKYMAP(element);

	GstPad* pad = gst_pad_new_from_template(templ, g_strdup_printf("sink%d", skymap->padcounter++));
	if (!pad) return pad;

	if (!gst_element_add_pad(element, pad)) goto bad_pad;

	gst_pad_use_fixed_caps(pad);

	GstSkymapSnrCollectData* data = (GstSkymapSnrCollectData*) gst_collect_pads_add_pad_full(skymap->collect, pad, sizeof(GstSkymapSnrCollectData), (GstCollectDataDestroyNotify)gst_skymap_snr_collectdata_destroy);
	if (!data) goto bad_collect;

	data->instrument = NULL;
	data->adapter = gst_adapter_new();
	data->skymapcollectdata.last_end_time = 0;

	/* FIXME: check to make sure caps match all the other snr pads. */

	/* Add this collectdata to the list. */
	skymap->snr_collectdatas = g_slist_prepend(skymap->snr_collectdatas, data);

	/* Hack to override the event function used by the collectpads. */
	skymap->collect_event = (GstPadEventFunction) GST_PAD_EVENTFUNC(pad);
	gst_pad_set_event_function(pad, GST_DEBUG_FUNCPTR(snr_event));

	update_sink_caps(skymap);
	return pad;

bad_collect:
	gst_element_remove_pad(element, pad);
bad_pad:
	gst_object_unref(pad);
	return NULL;
}


static void release_pad(GstElement *element, GstPad *pad)
{
	GSTLALSkymap* skymap = GSTLAL_SKYMAP(element);

	/* Remove pad from snr pad list. */
	skymap->snr_collectdatas = g_slist_remove(
		skymap->snr_collectdatas,
		g_slist_find_custom(skymap->snr_collectdatas, pad, (GCompareFunc)snr_collectdata_is_pad)
   );

	/* Remove pad from collectpads. */
	gst_collect_pads_remove_pad(skymap->collect, pad);

	/* Remove pad from element. */
	gst_element_remove_pad(element, pad);

	/* Update channel count on coinc sink pad. */
	update_sink_caps(skymap);
}


static GstStateChangeReturn change_state(GstElement *element, GstStateChange transition)
{
	GSTLALSkymap *skymap = GSTLAL_SKYMAP(element);

	switch(transition) {
	case GST_STATE_CHANGE_NULL_TO_READY:
		break;

	case GST_STATE_CHANGE_READY_TO_PAUSED:
		gst_collect_pads_start(skymap->collect);
		break;

	case GST_STATE_CHANGE_PAUSED_TO_PLAYING:
		break;

	case GST_STATE_CHANGE_PAUSED_TO_READY:
		/* need to unblock the collectpads before calling the
		 * parent change_state so that streaming can finish */
		gst_collect_pads_stop(skymap->collect);
		break;

	default:
		break;
	}

	return parent_class->change_state(element, transition);
}


/* Equality function for storing SnglInspiralTable pointers in a GHashTable */
/* FIXME: This is also used in lal_skymap; refactor into another library? */
static gboolean sngl_inspiral_equal(gconstpointer v1, gconstpointer v2)
{
	const SnglInspiralTable * const sngl1 = v1;
	const SnglInspiralTable * const sngl2 = v2;

	/* FIXME Check equality on other relevant parameters here. */
	return sngl1->mass1 == sngl2->mass1 && sngl1->mass2 == sngl2->mass2;
}


/* Return TRUE if a SnglInspiralTable structure is zero'd, FALSE if there are any nonzero bytes */
static gboolean sngl_inspiral_is_nil(const SnglInspiralTable* const sngl)
{
	const unsigned char* ptr = (const unsigned char*) sngl;
	const unsigned char* const end = (const unsigned char* const) &sngl[1];
	for (; ptr < end ; ptr++)
		if (*ptr)
			return FALSE;
	return TRUE;
}


static GstSkymapCollectData* find_earliest_collectdata(GSTLALSkymap* skymap)
{
	GstClockTime min_last_end_time = GST_CLOCK_TIME_NONE;
	GstSkymapCollectData* data = NULL;
	GSList* slist;

	for (slist = skymap->collect->data; slist; slist = g_slist_next(slist))
	{
		GstSkymapCollectData* this_data = slist->data;
		if (this_data->last_end_time < min_last_end_time)
		{
			min_last_end_time = this_data->last_end_time;
			data = this_data;
		}
	}

	return data;
}


static GstFlowReturn collected(GstCollectPads *pads, gpointer user_data)
{
	GSTLALSkymap* skymap = GSTLAL_SKYMAP(user_data);

	/* Assure that we have enough sink pads. */
	/* FIXME: This check may be unnecessary. I should just make sure that the element gracefully handles 1 or 0 detectors. */
	if (GST_ELEMENT(skymap)->numsinkpads < 3)
	{
		GST_ERROR_OBJECT(skymap, "not enough snr sink pads, 2 required but only %d are present", GST_ELEMENT(skymap)->numsinkpads - 1);
		return GST_FLOW_ERROR;
	}

	/*
	 * Pick the pad from which to pop the next buffer.  Choose the pad for
	 * which the last received end time is the earliest, and only accept
	 * data from that pad.
	 */
	{
		GstSkymapCollectData* data = find_earliest_collectdata(skymap);
		g_assert(data);

		GstBuffer* buf = gst_collect_pads_pop(pads, (GstCollectData*)data);
		if (buf)
		{
			data->last_end_time = GST_BUFFER_TIMESTAMP(buf) + GST_BUFFER_DURATION(buf);

			if (data == (GstSkymapCollectData*)(skymap->coinc_collectdata))
			{
				GST_DEBUG_OBJECT(skymap, "popping coinc buffer");
				g_assert(((GstSkymapCoincCollectData*)(skymap->coinc_collectdata))->last_buffer == NULL);
				if (GST_BUFFER_SIZE(buf) > 0)
					((GstSkymapCoincCollectData*)(skymap->coinc_collectdata))->last_buffer = buf;
				else
					gst_buffer_unref(buf);
			} else {
				GST_DEBUG_OBJECT(skymap, "popping SNR buffer");
				gst_adapter_push( ((GstSkymapSnrCollectData*)data)->adapter, buf );
			}
		} else {
			GST_DEBUG_OBJECT(skymap, "no buffer to pop");
		}
	}

	/* Determine sample rate. */
	int rate;
	{
		GstCaps* caps = GST_PAD_CAPS(((GstCollectData*)(skymap->snr_collectdatas->data))->pad);
		g_assert(caps);
		GstStructure* structure = gst_caps_get_structure(caps, 0);
		g_assert(structure);
		if(!gst_structure_get_int(structure, "rate", &rate))
		{
			/* FIXME: is there any way to recover from this? */
			GST_ELEMENT_ERROR(skymap, STREAM, DECODE, ("Pad caps did not provide sample rate"), (NULL));
			return GST_FLOW_ERROR;
		}
	}

	guint32 adapter_stride = 2 * skymap->ntemplates;
	guint32 adapter_stride_bytes = sizeof(double) * adapter_stride;

	{
		/* Retrieve last buffer that we got from the coinc pad. */
		GstBuffer* coincbuf = ((GstSkymapCoincCollectData*)(skymap->coinc_collectdata))->last_buffer;
		int nchannels = GST_ELEMENT(skymap)->numsinkpads - 1;

		/* Gobble up this last buffer, starting from the head. */
		gboolean processed = TRUE;
		while (coincbuf && GST_BUFFER_SIZE(coincbuf) > 0 && processed)
		{
			const SnglInspiralTable* const head = (const SnglInspiralTable*) GST_BUFFER_DATA(coincbuf);

			/* Determine the start and end time of the segment to analyze for detectors that did not participate in the coincidence. */
			GstClockTime min_start_time = GST_CLOCK_TIME_NONE;
			GstClockTime max_stop_time = 0;
			const SnglInspiralTable* ptr = head;
			const SnglInspiralTable* end = &head[nchannels];
			for (; ptr < end && !sngl_inspiral_is_nil(ptr); ptr++)
			{
				GstClockTime trigger_time = XLALGPSToINT8NS(&ptr->end_time);
				/* FIXME: badness if trigger_time < skymap->trigger_present_padding */
				GstClockTime start_time = trigger_time - skymap->trigger_present_padding;
				GstClockTime stop_time = trigger_time + skymap->trigger_present_padding;
				if (start_time < min_start_time)
					min_start_time = start_time;
				if (stop_time > max_stop_time)
					max_stop_time = stop_time;
			}
			end = ptr; /* Save ptr to end so that we don't have to repeat the sngl_inspiral_is_nil check */
			min_start_time -= skymap->trigger_absent_padding;
			max_stop_time += skymap->trigger_absent_padding;

			/* Make sure that all pads have enough data present to analyze this coinc. */
			GSList* slist;
			for (slist = skymap->snr_collectdatas; slist; slist = g_slist_next(slist))
			{
				GstSkymapSnrCollectData* collectdata = (GstSkymapSnrCollectData*)(slist->data);

				if (((GstSkymapCollectData*)collectdata)->last_end_time < max_stop_time)
				{
					GST_DEBUG_OBJECT(skymap, "nothing to do right now");
					processed = FALSE;
					break;
				} /* else {
					guint64 adapter_distance;
					GstClockTime adapter_start_time = gst_adapter_prev_timestamp(collectdata->adapter, &adapter_distance);
					GstClockTime adapter_pos_time = adapter_start_time + gst_util_uint64_scale(adapter_distance, GST_SECOND, skymap->ntemplates * 16 * rate);
					if (adapter_pos_time > start_time)
					{
						// This can only happen if the start time of the data that we have on hand is too *late* to process the coinc.  There is no way to recover from this if the input is time-ordered.
						g_assert_not_reached();
						processed = FALSE;
						break;
					}
				} */
			}

			if (processed)
			{
				GST_INFO_OBJECT(skymap, "starting sky localization");
				/* Build skymap. */
				{
					/* Make sure we don't have more than the supported number of channels. */
					g_assert(nchannels < XLALSKYMAP_N);

					/* Find template index for template. */
					int bank_index = 0;
					g_mutex_lock(skymap->bank_lock);
					const SnglInspiralTable* bank = skymap->bank;
					for (; bank != NULL && !sngl_inspiral_equal(head, bank); bank = bank->next, bank_index++)
						; /* loop body intentionally empty */
					g_mutex_unlock(skymap->bank_lock);
					if (bank == NULL)
					{
						/* FIXME: Is there any way to recover from this error? */
						GST_ELEMENT_ERROR(skymap, STREAM, DECODE, ("Did not find channel with matching template"), (NULL));
						return GST_FLOW_ERROR;
					}

					/* Set number of detectors. */
					skymap->wanalysis.n_detectors = nchannels;

					/* Set rate. */
					skymap->wanalysis.rate = rate;

					/* Set min and max time. */
					skymap->wanalysis.min_t = 0;
					skymap->wanalysis.max_t = 1.0e-9 * (max_stop_time - min_start_time);

					size_t xSw_nsamples = gst_util_uint64_scale_ceil(max_stop_time - min_start_time, rate, GST_SECOND);
					double* xSw_reals = g_malloc(sizeof(double) * xSw_nsamples * nchannels);
					double* xSw_imags = g_malloc(sizeof(double) * xSw_nsamples * nchannels);

					GSList* slist;
					int ichannel;
					for (ichannel = 0, slist = skymap->snr_collectdatas; slist; slist = g_slist_next(slist), ichannel++)
					{
						GstSkymapSnrCollectData* collectdata = (GstSkymapSnrCollectData*)(slist->data);
						const SnglInspiralTable* found_sngl = NULL;
						for (ptr = head; ptr < end; ptr++)
						{
							if (G_UNLIKELY(!(collectdata->instrument)))
							{
								/* FIXME should clean up some state here, but we can't recover from this error. */
								GST_ELEMENT_ERROR(skymap, CORE, TAG, ("one or mor SNR pads never recieved an 'instrument' tag"), (NULL));
								return GST_FLOW_ERROR;
							}
							if (strcmp(ptr->ifo, collectdata->instrument) == 0)
							{
								found_sngl = ptr;
								break;
							}
						}
						GstClockTime start_time;
						GstClockTime stop_time;
						if (found_sngl)
						{
							GstClockTime trigger_time = XLALGPSToINT8NS(&found_sngl->end_time);
							/* FIXME: badness if trigger_time < skymap->trigger_present_padding */
							start_time = trigger_time - skymap->trigger_present_padding;
							stop_time = trigger_time + skymap->trigger_present_padding;
						} else {
							start_time = min_start_time;
							stop_time = max_stop_time;
						}

						/* FIXME: where do we get effective distance from if some of the detectors did not have coincidences? */
						if (!found_sngl)
						{
							/* FIXME: leaking memory by bailing out here */
							GST_ELEMENT_ERROR(skymap, STREAM, TOO_LAZY, ("lal_skymap does not yet support coincidences in which some detectors do not participate, and \"%s\" was absent.", collectdata->instrument), (NULL));
							return GST_FLOW_ERROR;
						}

						double* xSw_real = &xSw_reals[xSw_nsamples * ichannel];
						double* xSw_imag = &xSw_imags[xSw_nsamples * ichannel];

						/* Calculate indices to retrieve from adapter. */
						guint64 adapter_distance_bytes;
						GstClockTime adapter_start_time = gst_adapter_prev_timestamp(collectdata->adapter, &adapter_distance_bytes);
						guint64 adapter_distance = adapter_distance_bytes / adapter_stride_bytes;
						g_assert(adapter_distance * adapter_stride_bytes == adapter_distance_bytes);

						guint64 adapter_offset = gst_util_uint64_scale_round(min_start_time - adapter_start_time, rate, GST_SECOND) - adapter_distance;

						/* Set per-channel analysis parameters. */
						skymap->wanalysis.detectors[ichannel] = analysis_identify_detector(collectdata->instrument);
						skymap->wanalysis.wSw[ichannel] = found_sngl->sigmasq;
						skymap->wanalysis.min_ts[ichannel] = 1.0e-9 * (start_time - min_start_time);
						skymap->wanalysis.max_ts[ichannel] = 1.0e-9 * (stop_time - min_start_time);
						skymap->wanalysis.xSw_real[ichannel] = xSw_real;
						skymap->wanalysis.xSw_imag[ichannel] = xSw_imag;

						/* Copy data from adapter. */
						double* adapter_bytes = g_malloc(adapter_stride_bytes * xSw_nsamples);
						gst_adapter_copy(collectdata->adapter, (void*)adapter_bytes, adapter_offset * adapter_stride_bytes, xSw_nsamples * adapter_stride_bytes);

						/* De-interleave real and imaginary parts, and convert from SNR to match. */
						unsigned int i;
						for (i = 0; i < xSw_nsamples; i ++)
						{
							xSw_real[i] = adapter_bytes[adapter_stride*i + bank_index*2] * sqrt(found_sngl->sigmasq);
							xSw_imag[i] = adapter_bytes[adapter_stride*i + bank_index*2 + 1] * sqrt(found_sngl->sigmasq);
						}

						/* Free data that was copied from adapter. */
						g_free(adapter_bytes);
					}

					/* Analyze SNR using skymap library. */
					skymap->wanalysis.log_skymap = g_malloc(sizeof(double) * skymap->wanalysis.n_directions);
					analyze(&skymap->wanalysis);

					/* Free stuff that was used for analysis. */
					g_free(xSw_reals);
					g_free(xSw_imags);

					/* Create output buffer. */
					guint64 outbuf_size = sizeof(double) * 4 * skymap->wanalysis.n_directions;
					GstBuffer* outbuf;
					GstFlowReturn result = gst_pad_alloc_buffer(skymap->srcpad, GST_BUFFER_OFFSET_NONE, outbuf_size, gst_pad_get_caps_reffed(skymap->srcpad), &outbuf); /* FIXME: why doesn't GST_PAD_CAPS(skymap->srcpad) work here? */
					if (result != GST_FLOW_OK)
					{
						g_free(skymap->wanalysis.log_skymap);
						return result;
					}

					/* Pack directions, spans, and logp's into output buffer. */
					{
						unsigned int i;
						for (i = 0; i < skymap->wanalysis.n_directions; i ++)
						{
							double* entry = &((double*)GST_BUFFER_DATA(outbuf))[4 * i];
							entry[0] = skymap->wanalysis.directions[2*i];
							entry[1] = skymap->wanalysis.directions[2*i+1];
							entry[2] = M_PI * 0.4 / 180.0; /* FIXME: How do I get the span out? */
							entry[3] = skymap->wanalysis.log_skymap[i];
						}
					}

					/* Free log_skymap. */
					g_free(skymap->wanalysis.log_skymap);

					/* FIXME: set buffer metadata here.  We probably only want to set the timestamp, and leave all the other fields blank. */
					GST_BUFFER_TIMESTAMP(outbuf) = min_start_time;
					GST_BUFFER_DURATION(outbuf) = max_stop_time - min_start_time;
					GST_BUFFER_OFFSET(outbuf) = GST_BUFFER_OFFSET_NONE;
					GST_BUFFER_OFFSET_END(outbuf) = GST_BUFFER_OFFSET_NONE;

					/* TODO: emit coinc event right before emitting buffer. */

					/* Push buffer. */
					GST_INFO_OBJECT(skymap, "completed sky localization");
					result = gst_pad_push(skymap->srcpad, outbuf);
					if (result != GST_FLOW_OK)
						return result;
				}

				/* Squash down coinc buffer. */
				size_t recordsize = nchannels * sizeof(*head);
				GstBuffer* newbuf = gst_buffer_create_sub(coincbuf, recordsize, GST_BUFFER_SIZE(coincbuf) - recordsize);
				g_assert(newbuf);
				gst_buffer_unref(coincbuf);
				if (GST_BUFFER_SIZE(newbuf) == 0)
				{
					gst_buffer_unref(newbuf);
					newbuf = NULL;
				}
				coincbuf = newbuf;
			}
		}
		((GstSkymapCoincCollectData*)(skymap->coinc_collectdata))->last_buffer = coincbuf;
	}

	/* If, at the end, this subbuffer becomes empty, flush all of the adapters
	 up to the timestamp provided (plus coincidence window), then unref the
	 buffer, and set it to NULL.
	 */
	if (((GstSkymapCoincCollectData*)(skymap->coinc_collectdata))->last_buffer == NULL)
	{
		GstClockTime last_end_time = ((GstSkymapCollectData*)(skymap->coinc_collectdata))->last_end_time;
		GstClockTime last_untouchable_time = last_end_time + skymap->dt + skymap->trigger_present_padding + skymap->trigger_absent_padding;
		GSList* slist;
		for (slist = skymap->snr_collectdatas; slist; slist = g_slist_next(slist))
		{
			GstSkymapSnrCollectData* collectdata = (GstSkymapSnrCollectData*)(slist->data);
			guint64 adapter_distance_bytes;
			GstClockTime adapter_start_time = gst_adapter_prev_timestamp(collectdata->adapter, &adapter_distance_bytes);
			guint64 adapter_distance = adapter_distance_bytes / adapter_stride_bytes;
			guint64 available_samples = gst_adapter_available(collectdata->adapter) / adapter_stride_bytes;
			GstClockTime adapter_earliest_time = adapter_start_time + gst_util_uint64_scale(adapter_distance, GST_SECOND, rate);
			GstClockTime adapter_latest_time = adapter_start_time + gst_util_uint64_scale(adapter_distance + available_samples, GST_SECOND, rate);
			if (last_untouchable_time >= adapter_latest_time)
			{
				guint64 flush_to_time;
				if (last_untouchable_time > adapter_latest_time)
					flush_to_time = adapter_latest_time;
				else
					flush_to_time = last_untouchable_time;
				guint64 flushable_samples = gst_util_uint64_scale(flush_to_time - adapter_earliest_time, rate, GST_SECOND);
				gst_adapter_flush(collectdata->adapter, flushable_samples * adapter_stride_bytes);
			}
		}
	}


	/* TODO: think about how to tell when we have reached EOS and what we do when we get there. */

	return GST_FLOW_OK;
}


/*
 * ============================================================================
 *
 *                              Element Support
 *
 * ============================================================================
 */


static void finalize(GObject *object)
{
	GSTLALSkymap *element = GSTLAL_SKYMAP(object);

	g_mutex_free(element->bank_lock);
	element->bank_lock = NULL;
	free_bankfile(element);

	gst_object_unref(element->collect);
	element->collect = NULL;

	gst_object_unref(element->srcpad);
	element->srcpad = NULL;

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


static void base_init(gpointer g_class)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(g_class);

	gst_element_class_set_details_simple(
		element_class,
		"Skymap",
		"Analyzer/Audio",
		"Assign [log] probabilities to directions on the sky",
		"Leo Singer <leo.singer@ligo.org>, Antony Searle <antony.searle@ligo.caltech.edu>"
	);
	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"sink",
			GST_PAD_SINK,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
				"application/x-lal-snglinspiral," \
				"channels = (int) [0, MAX]"
			)
		)
	);
	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"sink%d",
			GST_PAD_SINK,
			GST_PAD_REQUEST,
			gst_caps_from_string(
				"audio/x-raw-complex, " \
				"channels = (int) [1, MAX], " \
				"endianness = (int) BYTE_ORDER, " \
				"width = (int) 128"
			)
		)
	);
	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"src",
			GST_PAD_SRC,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
				"audio/x-raw-float, " \
				"channels = (int) 4, " \
				"endianness = (int) BYTE_ORDER, " \
				"width = (int) 64"
			)
		)
	);
}


static void class_init(gpointer klass, gpointer class_data)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *gstelement_class = GST_ELEMENT_CLASS(klass);

	parent_class = g_type_class_ref(GST_TYPE_ELEMENT);

	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);
	gstelement_class->request_new_pad = GST_DEBUG_FUNCPTR(request_new_pad);
	gstelement_class->release_pad = GST_DEBUG_FUNCPTR(release_pad);
	gstelement_class->change_state = GST_DEBUG_FUNCPTR(change_state);

	g_object_class_install_property(
		gobject_class,
		ARG_DT,
		g_param_spec_uint64(
			"dt",
			"Coincidence Window",
			"Maximum time delay between any number of triggers for a coincidence to occur.",
			1, G_MAXUINT64, 50 * GST_MSECOND,
			G_PARAM_READWRITE | G_PARAM_CONSTRUCT | G_PARAM_STATIC_STRINGS
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_TRIGGER_PRESENT_PADDING,
		g_param_spec_uint64(
			"trigger-present-padding",
			"Trigger Present Padding",
			"Number of nanoseconds before and after a trigger to include in an analysis.",
			1, G_MAXUINT64, 20 * GST_MSECOND, /* FIXME: what is a good default value for this? */
			G_PARAM_READWRITE | G_PARAM_CONSTRUCT | G_PARAM_STATIC_STRINGS
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_TRIGGER_ABSENT_PADDING,
		g_param_spec_uint64(
			"trigger-absent-padding",
			"Trigger Absent Padding",
			"Amount by which to extend analysis when a detector does not participate in a coincidence (perhaps the maximum distance between any pair of detectors).",
			1, G_MAXUINT64, 50 * GST_MSECOND,
			G_PARAM_READWRITE | G_PARAM_CONSTRUCT | G_PARAM_STATIC_STRINGS
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_BANK_FILENAME,
		g_param_spec_string(
			"bank-filename",
			"Bank file name",
			"Path to XML file used to generate the template bank.  Setting this property resets sigmasq to a vector of 0s.",
			NULL,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);
}


static void instance_init(GTypeInstance *object, gpointer klass)
{
	GSTLALSkymap *element = GSTLAL_SKYMAP(object);

	gst_element_create_all_pads(GST_ELEMENT(element));
	element->srcpad = gst_element_get_static_pad(GST_ELEMENT(element), "src");
	gst_pad_use_fixed_caps(element->srcpad);

	/* initialize collectpads */
	element->collect = gst_collect_pads_new();
	element->collect_event = NULL;
	gst_collect_pads_set_function(element->collect, GST_DEBUG_FUNCPTR(collected), element);

	element->coinc_collectdata = gst_collect_pads_add_pad_full(element->collect, gst_element_get_static_pad(GST_ELEMENT(element), "sink"), sizeof(GstSkymapCoincCollectData), (GstCollectDataDestroyNotify)gst_skymap_coinc_collectdata_destroy);
	((GstSkymapCoincCollectData*)(element->coinc_collectdata))->last_buffer = NULL;
	((GstSkymapCollectData*)(element->coinc_collectdata))->last_end_time = 0;
	element->snr_collectdatas = NULL;

	/* internal data */
	element->bank_lock = g_mutex_new();
	element->bank_filename = NULL;
	element->ntemplates = 0;
	element->bank = NULL;
	element->padcounter = 0;

	/* Initialize wanalysis. */
	analysis_default_construct(&element->wanalysis);
	element->wanalysis.p_realloc = g_realloc;
	element->wanalysis.p_free = g_free;
	analysis_default_directions(&element->wanalysis);
}


GType gstlal_skymap_get_type(void)
{
	static GType type = 0;

	if(!type) {
		static const GTypeInfo info = {
			.class_size = sizeof(GSTLALSkymapClass),
			.class_init = class_init,
			.base_init = base_init,
			.instance_size = sizeof(GSTLALSkymap),
			.instance_init = instance_init,
		};
		type = g_type_register_static(GST_TYPE_ELEMENT, "GSTLALSkymap", &info, 0);
		GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "lal_skymap", 0, "skymap element");
	}

	return type;
}

/*
 * Copyright (C) 2009 Leo Singer <leo.singer@ligo.org>
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
#include <gstlal_coinc.h>
#include <lal/Date.h>
#include <lal/LIGOMetadataTables.h>


#define GST_CAT_DEFAULT gstlal_coinc_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


static GstElementClass *parent_class = NULL;



enum gen_property {
	ARG_DT = 1
};


static void set_property(GObject *object, enum gen_property id, const GValue *value, GParamSpec *pspec)
{
	GSTLALCoinc *element = GSTLAL_COINC(object);

	GST_OBJECT_LOCK(element);
	switch(id) {
		case ARG_DT:
			element->dt = g_value_get_uint64(value);
			break;

		default:
			G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
			break;
	}
	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject * object, enum gen_property id, GValue * value, GParamSpec * pspec)
{
	GSTLALCoinc *element = GSTLAL_COINC(object);

	GST_OBJECT_LOCK(element);
	switch(id) {
		case ARG_DT:
			g_value_set_uint64(value, element->dt);
			break;

		default:
			G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
			break;
	}
	GST_OBJECT_UNLOCK(element);
}


typedef struct {
	GstCollectData gstcollectdata;
	GstClockTime last_end_time;
} GstCoincCollectData;


static void update_src_caps(GSTLALCoinc* coinc)
{
	GstCaps* caps = gst_caps_new_simple("application/x-lal-snglinspiral", "channels", G_TYPE_INT, GST_ELEMENT(coinc)->numsinkpads, NULL);
	g_assert(caps);

	gst_pad_set_caps(coinc->srcpad, caps);
}


static GstPad *request_new_pad(GstElement *element, GstPadTemplate *templ, const gchar *name)
{
	GSTLALCoinc* coinc = GSTLAL_COINC(element);

	GstPad* pad = gst_pad_new_from_template(templ, g_strdup_printf("sink%d", coinc->padcounter++));
	if (!gst_element_add_pad(element, pad)) goto bad_pad;

	gst_pad_use_fixed_caps(pad);

	GstCoincCollectData* data = (GstCoincCollectData*) gst_collect_pads_add_pad(coinc->collect, pad, sizeof(GstCoincCollectData));
	if (!data) goto bad_collect;

	data->last_end_time = 0;
	update_src_caps(coinc);
	return pad;

bad_collect:
	gst_element_remove_pad(element, pad);
bad_pad:
	gst_object_unref(pad);
	return NULL;
}



static void release_pad(GstElement *element, GstPad *pad)
{
	GSTLALCoinc* coinc = GSTLAL_COINC(element);

	gst_collect_pads_remove_pad(coinc->collect, pad);
	gst_element_remove_pad(element, pad);

	update_src_caps(coinc);
}


static GstStateChangeReturn change_state(GstElement *element, GstStateChange transition)
{
	GSTLALCoinc *coinc = GSTLAL_COINC(element);

	switch(transition) {
	case GST_STATE_CHANGE_NULL_TO_READY:
		break;

	case GST_STATE_CHANGE_READY_TO_PAUSED:
		gst_collect_pads_start(coinc->collect);
		break;

	case GST_STATE_CHANGE_PAUSED_TO_PLAYING:
		break;

	case GST_STATE_CHANGE_PAUSED_TO_READY:
		/* need to unblock the collectpads before calling the
		 * parent change_state so that streaming can finish */
		gst_collect_pads_stop(coinc->collect);
		break;

	default:
		break;
	}

	return parent_class->change_state(element, transition);
}


static gboolean sngl_inspiral_le(SnglInspiralTable* lhs, SnglInspiralTable* rhs)
{
	return XLALGPSCmp(&lhs->end_time, &rhs->end_time) <= 0;
}


/* Merge two sngl_inspiral tables into one list, and return the head. */
static SnglInspiralTable* sngl_inspiral_merge(SnglInspiralTable* lhs, SnglInspiralTable* rhs)
{
	if (lhs == NULL) return rhs;
	if (rhs == NULL) return lhs;

	if (!sngl_inspiral_le(lhs, rhs))
	{
		void* tmp = lhs;
		lhs = rhs;
		rhs = tmp;
	}
	SnglInspiralTable* top = lhs;

	do {
		SnglInspiralTable* head;
		do {
			head = lhs;
			lhs = lhs->next;
		} while (lhs && sngl_inspiral_le(lhs, rhs));
		head->next = rhs;

		/* swap lhs and rhs */
		void* tmp = lhs;
		lhs = rhs;
		rhs = tmp;
	} while (rhs);

	return top;
}


// Hash function for storing SnglInspiralTable pointers in a GHashTable
static guint sngl_inspiral_hash(gconstpointer v)
{
	const SnglInspiralTable * const sngl = v;

	/* Because the template parameters are sampled numerically, *any* mass
	 * parameter is almost guaranteed to uniquely specify a template.  The
	 * choice of mass1 as the hash key is completely arbitrary.
	 */
	return sngl->mass1 * 0.01 * G_MAXUINT;
}


/* Equality function for storing SnglInspiralTable pointers in a GHashTable */
static gboolean sngl_inspiral_equal(gconstpointer v1, gconstpointer v2)
{
	const SnglInspiralTable * const sngl1 = v1;
	const SnglInspiralTable * const sngl2 = v2;

	/* FIXME Check equality on other relevant parameters here. */
	return sngl1->mass1 == sngl2->mass1 && sngl1->mass2 == sngl2->mass2;
}


static gboolean sngl_inspiral_handle_empty(gpointer key, gpointer value, gpointer user_data)
{
	return *((SnglInspiralTable**)value) == NULL;
}


/* Resize a GArray to a multiple of n */
static void g_array_size_to_next_multiple(GArray* array, gint n)
{
	g_array_set_size(array, gst_util_uint64_scale_int_ceil(array->len, 1, n) * n);
}


static GstCoincCollectData* find_earliest_collectdata(GSTLALCoinc* coinc)
{
	GstClockTime min_last_end_time = GST_CLOCK_TIME_NONE;
	GstCoincCollectData* data = NULL;
	GSList* slist;

	for (slist = coinc->collect->data; slist; slist = g_slist_next(slist))
	{
		GstCoincCollectData* this_data = slist->data;
		if (this_data->last_end_time < min_last_end_time)
		{
			min_last_end_time = this_data->last_end_time;
			data = this_data;
		}
	}

	return data;
}


static void pop_coincs(GstCollectData *collectdata, GSTLALCoinc *coinc)
{
	GstCoincCollectData *data = (GstCoincCollectData*)collectdata;
	GstBuffer* buf = gst_collect_pads_pop(collectdata->collect, collectdata);
	if (buf)
	{
		data->last_end_time = GST_BUFFER_TIMESTAMP(buf) + GST_BUFFER_DURATION(buf);

		/* Collate sngl_inspiral records from that buffer according to template. */
		GHashTable* hash = g_hash_table_new_full(sngl_inspiral_hash, sngl_inspiral_equal, NULL, g_free);
		const SnglInspiralTable* sngl = (const SnglInspiralTable*) GST_BUFFER_DATA(buf);
		const SnglInspiralTable* const sngl_end = (const SnglInspiralTable* const) (GST_BUFFER_DATA(buf) + GST_BUFFER_SIZE(buf));

		for (; sngl < sngl_end; sngl++)
		{
			SnglInspiralTable* sngl_copy = g_memdup(sngl, sizeof(*sngl));
			sngl_copy->next = NULL;

			SnglInspiralTable** val_handle = g_hash_table_lookup(hash, sngl_copy);
			if (val_handle)
			{
				val_handle[1]->next = sngl_copy;
				val_handle[1] = sngl_copy;
			} else {
				val_handle = g_malloc(sizeof(SnglInspiralTable*) * 2);
				val_handle[0] = val_handle[1] = sngl_copy;
				g_hash_table_insert(hash, (gpointer)sngl_copy, (gpointer)val_handle);
			}
		}

		/* Unref input buffer. */
		gst_buffer_unref(buf);

		/* Merge input records with currently buffered records. */
		GHashTableIter iter;
		SnglInspiralTable* key;
		SnglInspiralTable** val_handle;
		GHashTable* dest_hash = coinc->trigger_sequence_hash;
		g_hash_table_iter_init(&iter, hash);
		while (g_hash_table_iter_next(&iter, (gpointer*)&key, (gpointer*)&val_handle))
		{
			SnglInspiralTable** dest_sngl_handle = g_hash_table_lookup(dest_hash, key);
			if (dest_sngl_handle)
			{
				*dest_sngl_handle = sngl_inspiral_merge(*val_handle, *dest_sngl_handle);
			} else {
				dest_sngl_handle = g_malloc(sizeof(SnglInspiralTable*));
				*dest_sngl_handle = *val_handle;
				g_hash_table_insert(dest_hash, g_memdup(*val_handle, sizeof(**val_handle)), dest_sngl_handle);
			}
		}

		/* Unref input hashtable. */
		g_hash_table_unref(hash);
	}
}


static GstFlowReturn collected(GstCollectPads *pads, gpointer user_data)
{
	GSTLALCoinc* coinc = GSTLAL_COINC(user_data);
	GstElement* element = GST_ELEMENT(coinc);

	/* Assure that we have enough sink pads. */
	if (element->numsinkpads < 2)
	{
		GST_ERROR_OBJECT(coinc, "not enough sink pads, 2 required but only %d are present", element->numsinkpads < 2);
		return GST_FLOW_ERROR;
	}

	/*
	 * Pick the pad from which to pop the next buffer.  Choose the pad for
	 * which the last received end time is the earliest, and only accept
	 * data from that pad.
	 */
	GstCoincCollectData* data = find_earliest_collectdata(coinc);
	g_assert(data);
	GstClockTime timestamp = data->last_end_time;


	/* Take buffers from all pads for which data is available. */
	g_slist_foreach(pads->data, (GFunc)pop_coincs, coinc);

	GstClockTime last_seen_time;
	gboolean eos;
	if (pads->eospads == element->numsinkpads)
	{
		/* No data to be read, must be EOS */
		eos = TRUE;
		last_seen_time = GST_CLOCK_TIME_NONE;
	} else {
		eos = FALSE;
		data = find_earliest_collectdata(coinc);
		last_seen_time = data->last_end_time;
	}

	/* Group each template's trigger stream into coincident groups.

	 Here's the strategy for each template:

	 Maintain a running list of consecutive triggers from all detectors.
	 We assume that the coincidence window is larger than the minimum time
	 between tirggers from a given detector.

	 At the beginning, this list is empty.

	 At the end of every subsequent iteration, the list must satisfy the
	 following two conditions:

	  1. Let t1 = the minimum of all the trigger times.
	     Let t2 = the maximum of all the trigger times, or the latest analyzed time, whichever is smaller.
	     (t2 - t1) must be less than dt.

	  2. If there are N detectors, there must be fewer than N elements in the list.

	 */

	GHashTableIter iter;
	SnglInspiralTable* key;
	SnglInspiralTable** val_handle;
	GHashTable* dest_hash = coinc->trigger_sequence_hash;
	GArray* outarray = g_array_new(FALSE, TRUE, sizeof(SnglInspiralTable));

	g_hash_table_iter_init(&iter, dest_hash);
	while (g_hash_table_iter_next(&iter, (gpointer*)&key, (gpointer*)&val_handle))
	{
		SnglInspiralTable* latest_sngl = *val_handle;
		SnglInspiralTable* earliest_sngl = NULL;
		GstClockTime earliest_time = 0; /* FIXME: this is needed to avoid uninitialized use warning, but you can check that this variable is always initialized on the first pass thru this loop! */
		GstClockTime latest_time;
		guint numtriggers = 0;

		do {
			/* Update late end of running list. */
			++numtriggers;
			latest_time = XLALGPSToINT8NS(&latest_sngl->end_time);

			/* Update early end of running list. */
			if (earliest_sngl == NULL)
			{
				earliest_sngl = latest_sngl;
				earliest_time = latest_time;
			}

			SnglInspiralTable* next_latest_sngl;
			if (latest_time >= last_seen_time)
			{
				/* we don't have enough data to process this trigger yet */
				next_latest_sngl = NULL;
			} else {
				next_latest_sngl = latest_sngl->next;

				/* If necessary, enforce both of the running list's conditions. */
				if (latest_time - earliest_time > coinc->dt) {
					/* create coincidence records */
					if (numtriggers > 2)
					{
						SnglInspiralTable* head_sngl;
						for (head_sngl = earliest_sngl; head_sngl != latest_sngl; head_sngl = head_sngl->next)
							g_array_append_vals(outarray, head_sngl, 1);

						/* resize output array to a multiple of numsinkpads */
						g_array_size_to_next_multiple(outarray, element->numsinkpads);
					}

					/* forget triggers that are no longer relevant */
					while (latest_time - earliest_time > coinc->dt)
					{
						SnglInspiralTable* next_sngl = earliest_sngl->next;
						earliest_sngl->next = NULL;
						g_free(earliest_sngl);
						earliest_sngl = next_sngl;
						earliest_time = XLALGPSToINT8NS(&earliest_sngl->end_time);
						g_assert(numtriggers > 0);
						--numtriggers;
					}
				} else if (numtriggers >= element->numsinkpads) {
					/* all detectors are present, form a coincidence. */
					SnglInspiralTable* head_sngl;
					if (numtriggers > 1)
					{
						for (head_sngl = earliest_sngl; head_sngl != next_latest_sngl; head_sngl = head_sngl->next)
							g_array_append_vals(outarray, head_sngl, 1);
					}

					head_sngl = earliest_sngl;
					while (head_sngl != next_latest_sngl)
					{
						SnglInspiralTable* next_sngl = head_sngl->next;
						head_sngl->next = NULL;
						g_free(head_sngl);
						head_sngl = next_sngl;
						g_assert(numtriggers > 0);
						--numtriggers;
					}
					earliest_sngl = NULL;
				}

				if (earliest_sngl && (next_latest_sngl == NULL || (GstClockTime)XLALGPSToINT8NS(&next_latest_sngl->end_time) >= last_seen_time) && latest_time + coinc->dt < last_seen_time)
				{
					SnglInspiralTable* head_sngl;
					/* create coincidence records */
					if (numtriggers > 1)
					{
						for (head_sngl = earliest_sngl; head_sngl != next_latest_sngl; head_sngl = head_sngl->next)
							g_array_append_vals(outarray, head_sngl, 1);

						/* resize output array to a multiple of numsinkpads */
						g_array_size_to_next_multiple(outarray, element->numsinkpads);
					}

					/* forget triggers that are no longer relevant */
					head_sngl = earliest_sngl;
					while (head_sngl != next_latest_sngl)
					{
						SnglInspiralTable* next_sngl = head_sngl->next;
						head_sngl->next = NULL;
						g_free(head_sngl);
						head_sngl = next_sngl;
						g_assert(numtriggers > 0);
						--numtriggers;
					}
					earliest_sngl = NULL;
				}
			}

			/* Take next value in list. */
			latest_sngl = next_latest_sngl;
		} while (latest_sngl != NULL);

		/* put back linked list after we have eaten up as many triggers as we can. */
		*val_handle = earliest_sngl;
	}

	/* wipe out null entries in hashtable */
	g_hash_table_foreach_remove(dest_hash, sngl_inspiral_handle_empty, NULL);

	GST_INFO_OBJECT(coinc, "found %d coincident triggers", outarray->len);

	GstFlowReturn retval;
	if (eos && outarray->len == 0)
	{
		g_array_free(outarray, TRUE);
		retval = GST_FLOW_UNEXPECTED;
	} else {
		guint64 siz = outarray->len * sizeof(SnglInspiralTable);

		/* wipe out next fields in case someone tries to dereference them */
		{
			SnglInspiralTable* ptr = (SnglInspiralTable*)outarray->data;
			SnglInspiralTable* end = &ptr[outarray->len];
			for (; ptr < end; ptr++)
				ptr->next = NULL;
		}

		/* Generate outgoing buffer. */
		GstBuffer *buf;
		retval = gst_pad_alloc_buffer(coinc->srcpad, GST_BUFFER_OFFSET_NONE, siz, GST_PAD_CAPS(coinc->srcpad), &buf);
		if (retval == GST_FLOW_OK)
		{
			memcpy(GST_BUFFER_DATA(buf), outarray->data, siz);
			g_array_free(outarray, TRUE);
			GST_BUFFER_TIMESTAMP(buf) = timestamp;
			GST_BUFFER_DURATION(buf) = last_seen_time - timestamp;
			GST_BUFFER_OFFSET(buf) = GST_BUFFER_OFFSET_NONE;
			GST_BUFFER_OFFSET_END(buf) = GST_BUFFER_OFFSET_NONE;
			retval = gst_pad_push(coinc->srcpad, buf);
		} else {
			GST_ERROR_OBJECT(coinc, "Failed to push buffer");
		}
	}

	if (eos)
		retval = gst_pad_push_event(coinc->srcpad, gst_event_new_eos());

	return retval;
}


static void finalize(GObject *object)
{
	GSTLALCoinc *coinc = GSTLAL_COINC(object);

	/* destroy hashtable and its contents */
	GHashTableIter iter;
	SnglInspiralTable* key;
	SnglInspiralTable** val;
	g_hash_table_iter_init(&iter, coinc->trigger_sequence_hash);
	while (g_hash_table_iter_next(&iter, (gpointer*)&key, (gpointer*)&val))
	{
		g_hash_table_iter_remove(&iter);
		while (*val != NULL)
		{
			SnglInspiralTable* next = (*val)->next;
			(*val)->next = NULL;
			g_free(*val);
			*val = next;
		}
	}
	g_hash_table_unref(coinc->trigger_sequence_hash);
	coinc->trigger_sequence_hash = NULL;

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


static void base_init(gpointer g_class)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(g_class);

	gst_element_class_set_details_simple(
		element_class,
		"Coincidence Generator",
		"Filter",
		"Assemble coincident triggers from single detector triggers.\n" \
		"\n" \
		"For the purposes of this element, a coincidence occurs whenever two or more\n" \
		"triggers occur at multiple detectors for the same template bank, with end times\n" \
		"that differ by no more than dt.\n" \
		"\n" \
		"At present, this element assumes that dt is less than the minimum time between\n" \
		"single-detector, single-template triggers, times the number of detectors.  This\n" \
		"restriction may be lifted in the future.\n" \
		"\n" \
		"Each detector is expected to provide buffers of SnglInspiralTable structures to\n" \
		"one of the sink pads.  The SnglInspiralTable do not have to be in chronological\n" \
		"order, although all of the SnglInspiralTable entries for a given template *do*\n" \
		"have to be in chronological order.\n" \
		"\n" \
		"The src pad provides buffers of multiple-channel SnglInspiralTable structures.\n" \
		"If a coincidence does not involve all of the detectors, then some of the\n" \
		"channels may be unused for that coincidence.  When this occurs, the unused\n" \
		"channels will be zeroed.  Channels will be used in order of ascending memory\n" \
		"address.\n" \
		"\n" \
		"Also, outgoing coincidences may not be chronologically ordered, but they *will*\n" \
		"be chronologically ordered on a per-template basis.\n" \
		"\n" \
		"The order in which triggers occur within a coincidence is undefined.\n" \
		"In particular, there is no reason that H1 triggers will ever appear in a\n" \
		"particular channel in the output buffer.\n" \
		"\n" \
		"The process by which triggers are formed is a greedy recursive algorithm.\n",
		"Leo Singer <leo.singer@ligo.org>"
	);
	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"sink%d",
			GST_PAD_SINK,
			GST_PAD_REQUEST,
			gst_caps_from_string(
				"application/x-lal-snglinspiral"
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
				"application/x-lal-snglinspiral," \
				"channels = (int) [0, MAX]"
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
}


static void instance_init(GTypeInstance *object, gpointer klass)
{
	GSTLALCoinc *coinc = GSTLAL_COINC(object);
	GstElement *element = GST_ELEMENT(coinc);

	gst_element_create_all_pads(element);
	coinc->srcpad = gst_element_get_static_pad(element, "src");

	coinc->collect = gst_collect_pads_new();
	gst_collect_pads_set_function(coinc->collect, GST_DEBUG_FUNCPTR(collected), coinc);
	coinc->padcounter = 0;

	coinc->trigger_sequence_hash = g_hash_table_new_full(sngl_inspiral_hash, sngl_inspiral_equal, g_free, g_free);
}


GType gstlal_coinc_get_type(void)
{
	static GType type = 0;

	if(!type) {
		static const GTypeInfo info = {
			.class_size = sizeof(GSTLALCoincClass),
			.class_init = class_init,
			.base_init = base_init,
			.instance_size = sizeof(GSTLALCoinc),
			.instance_init = instance_init,
		};
		type = g_type_register_static(GST_TYPE_ELEMENT, "GSTLALCoinc", &info, 0);
		GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "lal_coinc", 0, "coinc element");
	}

	return type;
}

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



#include <gstlal_gap_trigger_veto.h>

#include <stdlib.h>
#include <lal/Date.h>
#include <lal/LIGOMetadataTables.h>


static GstElementClass *parent_class = NULL;



/*
 * ============================================================================
 *
 *                              Pad Management
 *
 * ============================================================================
 */


static void collectdata_destroy(GstCollectData *collectdata)
{
	GSTLALGapTriggerVetoCollectData *cd = (GSTLALGapTriggerVetoCollectData*)collectdata;

	if (cd->last_buffer)
	{
		gst_buffer_unref(cd->last_buffer);
		cd->last_buffer = NULL;
	}
}


static void collectdata_init(GstCollectData *collectdata)
{
	GSTLALGapTriggerVetoCollectData *cd = (GSTLALGapTriggerVetoCollectData*)collectdata;

	cd->last_buffer = 0;
	cd->last_end_time = 0;
}


static GstStateChangeReturn change_state(GstElement *element, GstStateChange transition)
{
	GSTLALGapTriggerVeto *skymap = GSTLAL_GAP_TRIGGER_VETO(element);

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


static GSTLALGapTriggerVetoCollectData *find_earliest_collectdata(GstCollectPads *collectpads)
{
	GstClockTime min_last_end_time = GST_CLOCK_TIME_NONE;
	GSTLALGapTriggerVetoCollectData *data = NULL;
	GSList *slist;

	for (slist = collectpads->data; slist; slist = g_slist_next(slist))
	{
		GSTLALGapTriggerVetoCollectData *this_data = slist->data;
		if (this_data->last_end_time < min_last_end_time)
		{
			min_last_end_time = this_data->last_end_time;
			data = this_data;
		}
	}

	return data;
}


static int sngl_inspiral_cmp_end_time(const void *lhs, const void *rhs)
{
	return XLALGPSCmp(
		&(((const SnglInspiralTable*)lhs)->end_time),
		&(((const SnglInspiralTable*)rhs)->end_time)
	);
}


static int sngl_inspiral_cmp_key_end_time(const void *lhs, const void *rhs)
{
	return XLALGPSCmp(
		(const LIGOTimeGPS*)lhs,
		&(((const SnglInspiralTable*)rhs)->end_time)
	);
}


/**
 * bsearch_bisect
 *
 * Copied verbatim from libiberty's bsearch.c, excpet for the return statement.
 * A binary search that is identical in every way to bsearch, except that if no
 * match is found, a pointer to the first element that is less than the key is
 * returned instead.  If the key is greater than all of the elements, then a
 * pointer just past the end of the array is returned.
 */
static /* static added by us */ void *
bsearch_bisect (register const void *key, const void *base0,
         size_t nmemb, register size_t size,
         register int (*compar)(const void *, const void *))
{
	register const char *base = (const char *) base0;
	register int lim, cmp;
	register const void *p;

	for (lim = nmemb; lim != 0; lim >>= 1) {
		p = base + (lim >> 1) * size;
		cmp = (*compar)(key, p);
		if (cmp == 0)
			return (void *)p;
		if (cmp > 0) {	/* key > p: move right */
			base = (const char *)p + size;
			lim--;
		} /* else move left */
	}
	return (void *)(base); /* used to be return (NULL); */
}


static GstFlowReturn collected(GstCollectPads *pads, gpointer user_data)
{
	GSTLALGapTriggerVeto* element = GSTLAL_GAP_TRIGGER_VETO(user_data);

	/*
	 * Pick the pad from which to pop the next buffer.  Choose the pad for
	 * which the last received end time is the earliest, and only accept
	 * data from that pad.
	 */
	GSTLALGapTriggerVetoCollectData *data = find_earliest_collectdata(pads);
	g_assert(data);

	if (data->last_buffer) /* can't accept data from this pad yet; we haven't finished the last buffer! */
		return GST_FLOW_OK;

	data->last_buffer = gst_collect_pads_pop(pads, (GstCollectData*)data);
	if (!(data->last_buffer)) /* can't accept data from this pad yet; there isn't any data there! */
		return GST_FLOW_OK;

	data->last_end_time = GST_BUFFER_TIMESTAMP(data->last_buffer) + GST_BUFFER_DURATION(data->last_buffer);

	if (GST_BUFFER_DURATION(data->last_buffer) == 0)
	{
		/* You idiot!  This buffer is empty! */
		gst_buffer_unref(element->sinkdata->last_buffer);
		element->sinkdata->last_buffer = NULL;
		return GST_FLOW_OK;
	}
	
	/* If this is a buffer of triggers, (quick)sort them in-place by end_time. */
	if (data == element->sinkdata)
	{
		data->last_buffer = gst_buffer_make_writable(data->last_buffer);
		qsort(GST_BUFFER_DATA(data->last_buffer), GST_BUFFER_SIZE(data->last_buffer) / sizeof(SnglInspiralTable), sizeof(SnglInspiralTable),  sngl_inspiral_cmp_end_time);
	}

	/* If one or both pads do not have buffers, we need to wait longer */
	if (!(element->sinkdata->last_buffer) || !(element->controldata->last_buffer))
		return GST_FLOW_OK;

	if (element->controldata->last_end_time < GST_BUFFER_TIMESTAMP(element->sinkdata->last_buffer))
	{
		/* You idiot!  This SNR buffer can't have anything to do with this trigger buffer! */
		gst_buffer_unref(element->sinkdata->last_buffer);
		element->sinkdata->last_buffer = NULL;
		return GST_FLOW_OK;
	}

	if (element->sinkdata->last_end_time < GST_BUFFER_TIMESTAMP(element->controldata->last_buffer))
	{
		/* Oh, crap!  This trigger buffer is too old to every be processed! */
		GST_ELEMENT_ERROR(element, STREAM, FAILED, ("trigger buffer too old to process"), (NULL));
		gst_buffer_unref(element->sinkdata->last_buffer);
		element->sinkdata->last_buffer = NULL;
		return GST_FLOW_ERROR; /* FIXME: Should we try to recover from this, even though we would be dropping triggers? */
	}

	/* Find location from which to create sub-buffer. */
	LIGOTimeGPS controldata_last_gps_end_time;
	XLALINT8NSToGPS(&controldata_last_gps_end_time, element->controldata->last_end_time);
	SnglInspiralTable *found_sngl = bsearch_bisect(&controldata_last_gps_end_time, GST_BUFFER_DATA(element->sinkdata->last_buffer), GST_BUFFER_SIZE(element->sinkdata->last_buffer) / sizeof(SnglInspiralTable), sizeof(SnglInspiralTable), sngl_inspiral_cmp_key_end_time);

	/* Create sub-buffer to push. */
	guint64 found_sngl_offset = found_sngl - (SnglInspiralTable*)GST_BUFFER_DATA(element->sinkdata->last_buffer);
	GstBuffer *lhsbuf = gst_buffer_create_sub(element->sinkdata->last_buffer, 0, found_sngl_offset);

	/* Set duration of buffer. */
	GST_BUFFER_DURATION(lhsbuf) = element->controldata->last_end_time - GST_BUFFER_TIMESTAMP(lhsbuf);

	/* Copy value of gap flag from control buffer. */
	if (GST_BUFFER_FLAG_IS_SET(element->controldata->last_buffer, GST_BUFFER_FLAG_GAP))
		GST_BUFFER_FLAG_SET(lhsbuf, GST_BUFFER_FLAG_GAP);
	else
		GST_BUFFER_FLAG_UNSET(lhsbuf, GST_BUFFER_FLAG_GAP);

	if (element->controldata->last_end_time > element->sinkdata->last_end_time)
	{
		/* The trigger buffer should be empty by now!  If it isn't, something is really wrong with the stream! */
		if (found_sngl_offset != GST_BUFFER_SIZE(element->sinkdata->last_buffer))
			GST_ELEMENT_ERROR(element, STREAM, FAILED, ("trigger buffer not empty!"), (NULL));
		gst_buffer_unref(element->sinkdata->last_buffer);
		element->sinkdata->last_buffer = NULL;
	} else {
		/* Create a sub-buffer to keep the rest of the triggers. */
		GstBuffer *rhsbuf = gst_buffer_create_sub(element->sinkdata->last_buffer, found_sngl_offset, GST_BUFFER_SIZE(element->sinkdata->last_buffer) - found_sngl_offset);
		GST_BUFFER_TIMESTAMP(rhsbuf) = element->controldata->last_end_time;
		GST_BUFFER_DURATION(rhsbuf) = element->sinkdata->last_end_time - GST_BUFFER_TIMESTAMP(rhsbuf);
		gst_buffer_unref(element->sinkdata->last_buffer);
		element->sinkdata->last_buffer = rhsbuf;

		/* Toss away the rest of the SNR buffer. */
		gst_buffer_unref(element->controldata->last_buffer);
		element->controldata->last_buffer = NULL;
	}

	return gst_pad_push(element->srcpad, lhsbuf);
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
	GSTLALGapTriggerVeto *element = GSTLAL_GAP_TRIGGER_VETO(object);

	gst_object_unref(element->collect);
	element->collect = NULL;
	gst_object_unref(element->sinkpad);
	element->sinkpad = NULL;
	gst_object_unref(element->controlpad);
	element->controlpad = NULL;
	gst_object_unref(element->srcpad);
	element->srcpad = NULL;

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


static void base_init(gpointer g_class)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(g_class);

	gst_element_class_set_details_simple(
		element_class,
		"Gap Trigger Veto",
		"Filter",
		"Copy gap flags from time series to SnglInspiralTable buffers",
		"Leo Singer <leo.singer@ligo.org>"
	);
	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"sink",
			GST_PAD_SINK,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
				"application/x-lal-snglinspiral"
			)
		)
	);
	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"control",
			GST_PAD_SINK,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
				"audio/x-raw-float"
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
				"application/x-lal-snglinspiral"
			)
		)
	);
}


static void class_init(gpointer klass, gpointer class_data)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *gstelement_class = GST_ELEMENT_CLASS(klass);

	parent_class = g_type_class_ref(GST_TYPE_ELEMENT);

	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);
	gstelement_class->change_state = GST_DEBUG_FUNCPTR(change_state);
}


static void instance_init(GTypeInstance *object, gpointer klass)
{
	GSTLALGapTriggerVeto *element = GSTLAL_GAP_TRIGGER_VETO(object);

	gst_element_create_all_pads(GST_ELEMENT(element));

	element->controlpad = gst_element_get_static_pad(GST_ELEMENT(element), "control");
	gst_pad_use_fixed_caps(element->controlpad);

	element->sinkpad = gst_element_get_static_pad(GST_ELEMENT(element), "sink");
	gst_pad_use_fixed_caps(element->sinkpad);

	element->srcpad = gst_element_get_static_pad(GST_ELEMENT(element), "src");
	gst_pad_use_fixed_caps(element->srcpad);

	/* initialize collectpads */
	element->collect = gst_collect_pads_new();
	element->sinkdata = (GSTLALGapTriggerVetoCollectData*) gst_collect_pads_add_pad_full(element->collect, element->sinkpad, sizeof(GSTLALGapTriggerVetoCollectData), collectdata_destroy);
	element->controldata = (GSTLALGapTriggerVetoCollectData*) gst_collect_pads_add_pad_full(element->collect, element->controlpad, sizeof(GSTLALGapTriggerVetoCollectData), collectdata_destroy);
	collectdata_init(element->sinkdata);
	collectdata_init(element->controldata);
	gst_collect_pads_set_function(element->collect, collected, element);
}


GType gstlal_gap_trigger_veto_get_type(void)
{
	static GType type = 0;

	if(!type) {
		static const GTypeInfo info = {
			.class_size = sizeof(GSTLALGapTriggerVetoClass),
			.class_init = class_init,
			.base_init = base_init,
			.instance_size = sizeof(GSTLALGapTriggerVeto),
			.instance_init = instance_init,
		};
		type = g_type_register_static(GST_TYPE_ELEMENT, "lal_gap_trigger_veto", &info, 0);
	}

	return type;
}

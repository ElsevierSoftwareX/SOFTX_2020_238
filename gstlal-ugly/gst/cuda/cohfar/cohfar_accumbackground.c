/*
 * Copyright (C) 2015	Qi Chu	<qi.chu@uwa.edu.au>
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


/*
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */
#include <math.h>
#include <string.h>
/*
 *  stuff from gobject/gstreamer
*/


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <gstlal/gstlal.h>


/*
 * stuff from here
 */

#include <postcoh/postcohinspiral_table.h>
#include <cohfar/background_stats_utils.h>
#include <cohfar/cohfar_accumbackground.h>

#include <time.h>
#define NOT_INIT -1
#define DEFAULT_STATS_FNAME "stats.xml.gz"
/*
 * ============================================================================
 *
 *                                 Utilities
 *
 * ============================================================================
 */


/*
 * ============================================================================
 *
 *                           GStreamer Boiler Plate
 *
 * ============================================================================
 */


#define GST_CAT_DEFAULT cohfar_accumbackground_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


static void additional_initializations(GType type)
{
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "cohfar_accumbackground", 0, "cohfar_accumbackground element");
}

GST_BOILERPLATE_FULL(
	CohfarAccumbackground,
	cohfar_accumbackground,
	GstElement,
	GST_TYPE_ELEMENT,
	additional_initializations
);

enum property {
	PROP_0,
	PROP_IFOS,
	PROP_HIST_TRIALS,
	PROP_SNAPSHOT_INTERVAL,
	PROP_HISTORY_FNAME,
	PROP_OUTPUT_PREFIX,
	PROP_OUTPUT_FNAME
};

static void cohfar_accumbackground_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * pspec);
static void cohfar_accumbackground_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec);

/* vmethods */

static GstFlowReturn cohfar_accumbackground_chain (GstPad * pad, GstBuffer * inbuf);
static gboolean cohfar_accumbackground_sink_event (GstPad * pad, GstEvent * event);
static void cohfar_accumbackground_dispose (GObject *object);

/*
 * ============================================================================
 *
 *                     GstElement Method Overrides
 *
 * ============================================================================
 */


/*
 * chain()
 */


static GstFlowReturn cohfar_accumbackground_chain(GstPad *pad, GstBuffer *inbuf)
{
	CohfarAccumbackground *element = COHFAR_ACCUMBACKGROUND(GST_OBJECT_PARENT(pad));
	GstFlowReturn result = GST_FLOW_OK;

	GST_LOG_OBJECT (element, "receiving accum %s+%s buffer of %" G_GSIZE_FORMAT 
	" bytes, ts %" GST_TIME_FORMAT 
      ", duration %" GST_TIME_FORMAT ", offset %"
      G_GINT64_FORMAT ", offset_end %" G_GINT64_FORMAT,
      GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP) ? "GAP" : "NONGAP",
      GST_BUFFER_IS_DISCONT(inbuf) ? "DISCONT" : "CONT",
      GST_BUFFER_SIZE(inbuf), GST_TIME_ARGS (GST_BUFFER_TIMESTAMP (inbuf)),
      GST_TIME_ARGS (GST_BUFFER_DURATION (inbuf)),
      GST_BUFFER_OFFSET (inbuf), GST_BUFFER_OFFSET_END (inbuf));



	if (!GST_CLOCK_TIME_IS_VALID(element->t_roll_start))
		element->t_roll_start = GST_BUFFER_TIMESTAMP(inbuf);

	/* 
	 * initialize stats files 
	 */
	BackgroundStats **stats_snapshot = element->stats_snapshot;
	// BackgroundStats **stats_prompt = element->stats_prompt;
	// BackgroundStatsPointerList *stats_list = element->stats_list;
	// /* reset stats_prompt */
	// background_stats_reset(stats_prompt, element->ncombo);

	
	/*
	 * reset stats in the stats_list in order to input new background points
	 */
	// int pos = stats_list->pos;
	// BackgroundStats **cur_stats_in_list = stats_list->plist[pos];
	// background_stats_reset(cur_stats_in_list, element->ncombo);


	/*
	 * calculate number of output postcoh entries
	 */
	int outentries = 0;

	PostcohInspiralTable *intable = (PostcohInspiralTable *) GST_BUFFER_DATA(inbuf);
	PostcohInspiralTable *intable_end = (PostcohInspiralTable *) (GST_BUFFER_DATA(inbuf) + GST_BUFFER_SIZE(inbuf));
	for (; intable<intable_end; intable++) 
		if (intable->is_background == 0) 
			outentries++;

	/*
	 * allocate output buffer
	 */
	GstBuffer *outbuf = NULL;
	GstPad *srcpad = element->srcpad;
	GstCaps *caps = GST_PAD_CAPS(srcpad);

	/* allocate extra space for prompt stats */	
	//int out_size = sizeof(PostcohInspiralTable) * outentries + sizeof(BackgroundStats) * ncombo;
	int out_size = sizeof(PostcohInspiralTable) * outentries;
	result = gst_pad_alloc_buffer(srcpad, 0, out_size, caps, &outbuf);
	if (result != GST_FLOW_OK) {
		GST_ERROR_OBJECT(srcpad, "Could not allocate postcoh-inspiral buffer %d", result);
		return result;
	}

	/*
	 * update background rates
	 */

	intable = (PostcohInspiralTable *) GST_BUFFER_DATA(inbuf);
	PostcohInspiralTable *outtable = (PostcohInspiralTable *) GST_BUFFER_DATA(outbuf);
	int isingle, icombo, nifo;
	for (; intable<intable_end; intable++) {
		//printf("is_back %d\n", intable->is_background);
		if (intable->is_background == 1) {
			//printf("cohsnr %f, maxsnr %f\n", intable->cohsnr, intable->maxsnglsnr);
			icombo = get_icombo(intable->ifos);
			if (icombo > -1)
				background_stats_feature_rates_update((double)intable->cohsnr, (double)intable->cmbchisq, stats_snapshot[icombo]->feature, stats_snapshot[icombo]);

			nifo = strlen(intable->ifos)/IFO_LEN;
			/* add single detector stats */
			for (isingle=0; isingle< nifo; isingle++)
				background_stats_feature_rates_update((double)(*(&(intable->snglsnr_L) + isingle)), (double)(*(&(intable->chisq_L) + isingle)), stats_snapshot[isingle]->feature, stats_snapshot[isingle]);
			/* add stats to stats_list for prompt FAP estimation */
			// if (icombo > -1)
			// 	background_stats_feature_rates_update((double)intable->cohsnr, (double)intable->cmbchisq, cur_stats_in_list[icombo]->rates, cur_stats_in_list[icombo]);

		} else { /* coherent trigger entry */
			memcpy(outtable, intable, sizeof(PostcohInspiralTable));
			outtable++;
		} 
	}
	/*
	 * calculate immediate PDF using stats_prompt from stats_list
	 */
	// int ilist = 0, ncombo = element->ncombo;
	// if (outentries > 0) {
	// 	/* sum all stats in stats_list to stats_prompt */
	// 	for (ilist=0; ilist<stats_list->size; ilist++) {
	// 		cur_stats_in_list = stats_list->plist[(pos+ilist)%NSTATS_TO_PROMPT];
	// 		for (icombo=0; icombo<ncombo; icombo++)
	// 			background_stats_rates_add(stats_prompt[icombo]->rates, cur_stats_in_list[icombo]->rates, stats_prompt[icombo]);
	// 			memcpy(outtable, stats_prompt[icombo], sizeof(BackgroundStats))
	// 	}
	// }
	/* 
	 * deprecated: assign prompt FAP to 'fap' field of our trigger
	 */
	//outtable = (PostcohInspiralTable *) GST_BUFFER_DATA(outbuf);

	//int ientry = 0;
	//for (ientry=0; ientry<outentries; ientry++) {
	//	icombo = get_icombo(outtable->ifos);
	//	/* substitude fap with pdf */
	//	outtable->fap = background_stats_bins2D_get_val((double)outtable->cohsnr, (double)outtable->cmbchisq, stats_prompt[icombo]->pdf);
	//	outtable++;
	//}
	
	/*
	 * shuffle one step down in stats_list 
	 */
	// stats_list->pos = (stats_list->pos + 1) % NSTATS_TO_PROMPT;

	/* snapshot background xml file when reaching the snapshot point*/
	GstClockTime t_cur = GST_BUFFER_TIMESTAMP(inbuf);
	element->t_end = t_cur;
	gint duration = (int) ((element->t_end - element->t_roll_start) / GST_SECOND);
	if (element->snapshot_interval > 0 && duration >= element->snapshot_interval) {
		gint gps_time = (int) (element->t_roll_start / GST_SECOND);
		GString *tmp_fname = g_string_new(element->output_prefix);
		g_string_append_printf(tmp_fname, "_%d_%d.xml.gz", gps_time, duration);
		background_stats_to_xml(stats_snapshot, element->ncombo, element->hist_trials, tmp_fname->str);
		g_string_free(tmp_fname, TRUE);
		background_stats_reset(stats_snapshot, element->ncombo);
		element->t_roll_start = t_cur;
	}

	/*
	 * set the outbuf meta data
	 */
	GST_BUFFER_TIMESTAMP(outbuf) = GST_BUFFER_TIMESTAMP(inbuf);
	GST_BUFFER_DURATION(outbuf) = GST_BUFFER_DURATION(inbuf);
	GST_BUFFER_OFFSET(outbuf) = GST_BUFFER_OFFSET(inbuf);
	GST_BUFFER_OFFSET_END(outbuf) = GST_BUFFER_OFFSET_END(inbuf);
	GST_BUFFER_SIZE(outbuf) = sizeof(PostcohInspiralTable) * outentries;

	gst_buffer_unref(inbuf);
	result = gst_pad_push(srcpad, outbuf);

  GST_LOG_OBJECT (element, "pushed %s+%s buffer of %ld bytes, ts %"
      GST_TIME_FORMAT ", duration %" GST_TIME_FORMAT ", offset %"
      G_GINT64_FORMAT ", offset_end %" G_GINT64_FORMAT,
      GST_BUFFER_FLAG_IS_SET(outbuf, GST_BUFFER_FLAG_GAP) ? "GAP" : "NONGAP",
      GST_BUFFER_IS_DISCONT(outbuf) ? "DISCONT" : "CONT",
      GST_BUFFER_SIZE(outbuf), GST_TIME_ARGS (GST_BUFFER_TIMESTAMP (outbuf)),
      GST_TIME_ARGS (GST_BUFFER_DURATION (outbuf)),
      GST_BUFFER_OFFSET (outbuf), GST_BUFFER_OFFSET_END (outbuf));

	return result;
}


/*
 * ============================================================================
 *
 *                          GObject Method Overrides
 *
 * ============================================================================
 */

/* handle events (search) */
static gboolean
cohfar_accumbackground_sink_event (GstPad * pad, GstEvent * event)
{
  CohfarAccumbackground *element = COHFAR_ACCUMBACKGROUND(GST_OBJECT_PARENT(pad));

  switch (GST_EVENT_TYPE(event)) {
    case GST_EVENT_EOS:
//      if (fflush (sink->file))
//        goto flush_failed;

    GST_LOG_OBJECT(element, "EVENT EOS. ");
    if (element->snapshot_interval >= 0) {
	gint gps_time = (int) (element->t_roll_start / GST_SECOND);
	gint duration = (int) ((element->t_end - element->t_roll_start) / GST_SECOND);
	GString *tmp_fname = g_string_new(element->output_prefix);
	g_string_append_printf(tmp_fname, "_%d_%d.xml.gz", gps_time, duration);
	background_stats_to_xml(element->stats_snapshot, element->ncombo, element->hist_trials, tmp_fname->str);
	g_string_free(tmp_fname, TRUE);
    } else {
	GString *tmp_fname = g_string_new(element->output_fname);
	background_stats_to_xml(element->stats_snapshot, element->ncombo, element->hist_trials, tmp_fname->str);
	g_string_free(tmp_fname, TRUE);
    }

      break;
    default:
      break;
  }

  return gst_pad_event_default(pad, event);
}



/*
 * set_property()
 */


static void cohfar_accumbackground_set_property(GObject *object, enum property prop_id, const GValue *value, GParamSpec *pspec)
{
	CohfarAccumbackground *element = COHFAR_ACCUMBACKGROUND(object);

	GST_OBJECT_LOCK(element);
	switch(prop_id) {
		case PROP_IFOS:
			element->ifos = g_value_dup_string(value);
			element->nifo = strlen(element->ifos) / IFO_LEN;
			element->ncombo = get_ncombo(element->nifo);
			element->stats_snapshot = background_stats_create(element->ifos);
			// element->stats_prompt = background_stats_create(element->ifos);
			// element->stats_list = background_stats_list_create(element->ifos);
			break;

		case PROP_HISTORY_FNAME:

			/* must make sure ifos have been loaded */
			g_assert(element->ifos != NULL);
			element->history_fname = g_value_dup_string(value);
			background_stats_from_xml(element->stats_snapshot, element->ncombo, &(element->hist_trials), element->history_fname);
			break;

		case PROP_OUTPUT_FNAME:
			element->output_fname = g_value_dup_string(value);
			break;

		case PROP_OUTPUT_PREFIX:
			element->output_prefix = g_value_dup_string(value);
			break;

		case PROP_HIST_TRIALS:
			element->hist_trials = g_value_get_int(value);
			break;


		case PROP_SNAPSHOT_INTERVAL:
			element->snapshot_interval = g_value_get_int(value);
			break;

		default:
			G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
			break;
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * get_property()
 */


static void cohfar_accumbackground_get_property(GObject *object, enum property prop_id, GValue *value, GParamSpec *pspec)
{
	CohfarAccumbackground *element = COHFAR_ACCUMBACKGROUND(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
		case PROP_IFOS:
			g_value_set_string(value, element->ifos);
			break;

		case PROP_HISTORY_FNAME:
			g_value_set_string(value, element->history_fname);
			break;

		case PROP_OUTPUT_FNAME:
			g_value_set_string(value, element->output_fname);
			break;

		case PROP_OUTPUT_PREFIX:
			g_value_set_string(value, element->output_prefix);
			break;

		case PROP_HIST_TRIALS:
			g_value_set_int(value, element->hist_trials);
			break;
	
		case PROP_SNAPSHOT_INTERVAL:
			g_value_set_int(value, element->snapshot_interval);
			break;
		default:
			G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
			break;
	}
	GST_OBJECT_UNLOCK(element);
}


/*
 * dispose()
 */


static void cohfar_accumbackground_dispose(GObject *object)
{
	CohfarAccumbackground *element = COHFAR_ACCUMBACKGROUND(object);

	if(element->stats_snapshot) {
		// FIXME: free stats
	}
	G_OBJECT_CLASS(parent_class)->dispose(object);
}


/*
 * base_init()
 */


static void cohfar_accumbackground_base_init(gpointer gclass)
{
#if 1
	GstElementClass *element_class = GST_ELEMENT_CLASS(gclass);

	gst_element_class_set_details_simple(
		element_class,
		"Update background xml file given background entries",
		"Background xml updater",
		"Background xml updater.\n",
		"Qi Chu <qi.chu at ligo dot org>"
	);
	gst_element_class_add_pad_template(
		element_class,
//		gst_static_pad_template_get(&cohfar_background_src_template)
		gst_pad_template_new(
			"sink",
			GST_PAD_SINK,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
				"application/x-lal-postcoh" 
			)
		)
	
	);

	gst_element_class_add_pad_template(
		element_class,
//		gst_static_pad_template_get(&cohfar_background_src_template)
		gst_pad_template_new(
			"src",
			GST_PAD_SRC,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
				"application/x-lal-postcoh" 
			)
		)
	);
#endif
}


/*
 * class_init()
 */


static void cohfar_accumbackground_class_init(CohfarAccumbackgroundClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
;
	gobject_class->set_property = GST_DEBUG_FUNCPTR(cohfar_accumbackground_set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(cohfar_accumbackground_get_property);
	gobject_class->dispose = GST_DEBUG_FUNCPTR(cohfar_accumbackground_dispose);

	g_object_class_install_property(
		gobject_class,
		PROP_IFOS,
		g_param_spec_string(
			"ifos",
			"ifo names",
			"ifos that participate in the pipeline",
			NULL,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);

	g_object_class_install_property(
		gobject_class,
		PROP_HISTORY_FNAME,
		g_param_spec_string(
			"history-fname",
			"Input history filename",
			"Reference history background statstics filename",
			DEFAULT_STATS_FNAME,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);


	g_object_class_install_property(
		gobject_class,
		PROP_OUTPUT_FNAME,
		g_param_spec_string(
			"output-fname",
			"Output filename",
			"Output background statistics filename",
			DEFAULT_STATS_FNAME,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);


	g_object_class_install_property(
		gobject_class,
		PROP_OUTPUT_PREFIX,
		g_param_spec_string(
			"output-prefix",
			"Output filename prefix",
			"Output background statistics filename",
			DEFAULT_STATS_FNAME,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);


	g_object_class_install_property(
		gobject_class,
		PROP_HIST_TRIALS,
		g_param_spec_int(
			"hist-trials",
			"number of shifted slides",
			"Number of shifted slides.",
			0, G_MAXINT, 1,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);


	g_object_class_install_property(
		gobject_class,
		PROP_SNAPSHOT_INTERVAL,
		g_param_spec_int(
			"snapshot-interval",
			"snapshot interval",
			"(-1) never update; (0) snapshot at the end; (N) snapshot background statistics xml file every N seconds.",
			-1, G_MAXINT, 86400,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);

}
/*
 * init()
 */


static void cohfar_accumbackground_init(CohfarAccumbackground *element, CohfarAccumbackgroundClass *element_klass)
{
	GstElementClass *klass = GST_ELEMENT_CLASS(element_klass);
	element->sinkpad = gst_pad_new_from_template(
			gst_element_class_get_pad_template(klass, "sink"), "sink");
	gst_element_add_pad(GST_ELEMENT(element), element->sinkpad);

	element->srcpad = gst_pad_new_from_template(
			gst_element_class_get_pad_template(klass, "src"), "src");
	gst_element_add_pad(GST_ELEMENT(element), element->srcpad);

	gst_pad_set_event_function(element->sinkpad,
					GST_DEBUG_FUNCPTR(cohfar_accumbackground_sink_event));

	gst_pad_set_chain_function(element->sinkpad,
					GST_DEBUG_FUNCPTR(cohfar_accumbackground_chain));

	element->stats_snapshot = NULL;
	element->stats_prompt = NULL;
	element->t_roll_start = GST_CLOCK_TIME_NONE;
	element->snapshot_interval = NOT_INIT;
}

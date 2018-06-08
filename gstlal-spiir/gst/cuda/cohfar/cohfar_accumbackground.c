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

#if 0
static GstStaticPadTemplate cohfar_accumbackground_src_template = 
		GST_STATIC_PAD_TEMPLATE("src",
			GST_PAD_SRC,
			GST_PAD_ALWAYS,
			gst_caps_from_string("application/x-lal-postcoh")
		);

static GstStaticPadTemplate cohfar_accumbackground_sink_template = 
		GST_STATIC_PAD_TEMPLATE("sink",
			GST_PAD_SINK,
			GST_PAD_ALWAYS,
			gst_caps_from_string("application/x-lal-postcoh")
		);
#endif

#define GST_CAT_DEFAULT cohfar_accumbackground_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);

G_DEFINE_TYPE_WITH_CODE(
	CohfarAccumbackground,
	cohfar_accumbackground,
	GST_TYPE_ELEMENT,
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "cohfar_accumbackground", 0, "cohfar_accumbackground element")	
);

enum property {
	PROP_0,
	PROP_IFOS,
	PROP_HIST_TRIALS,
	PROP_SNAPSHOT_INTERVAL,
	PROP_HISTORY_FNAME,
	PROP_OUTPUT_FNAME_PREFIX
};

static void cohfar_accumbackground_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * pspec);
static void cohfar_accumbackground_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec);

/* vmethods */

static GstFlowReturn cohfar_accumbackground_chain (GstPad * pad, GstBuffer * inbuf);
static gboolean cohfar_accumbackground_sink_event (GstPad * pad, GstObject *parent, GstEvent * event);
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
	GstMapInfo inmap;
	GstMapInfo outmap;
	
	CohfarAccumbackground *element = COHFAR_ACCUMBACKGROUND(GST_OBJECT_PARENT(pad));
	GstFlowReturn result = GST_FLOW_OK;

	GST_LOG_OBJECT (element, "receiving accum %s+%s buffer of %" G_GSIZE_FORMAT 
	" bytes, ts %" GST_TIME_FORMAT 
      ", duration %" GST_TIME_FORMAT ", offset %"
      G_GINT64_FORMAT ", offset_end %" G_GINT64_FORMAT,
      GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP) ? "GAP" : "NONGAP",
      GST_BUFFER_IS_DISCONT(inbuf) ? "DISCONT" : "CONT",
      gst_buffer_get_size(inbuf), GST_TIME_ARGS (GST_BUFFER_PTS(inbuf)),
      GST_TIME_ARGS (GST_BUFFER_DURATION (inbuf)),
      GST_BUFFER_OFFSET (inbuf), GST_BUFFER_OFFSET_END (inbuf));



	if (!GST_CLOCK_TIME_IS_VALID(element->t_roll_start))
		element->t_roll_start = GST_BUFFER_PTS(inbuf);

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

	gst_buffer_map(inbuf, &inmap, GST_MAP_READ);
	PostcohInspiralTable *intable = (PostcohInspiralTable *) inmap.data;
	PostcohInspiralTable *intable_end = (PostcohInspiralTable *) (&inmap.data + inmap.size);
	for (; intable<intable_end; intable++) 
		if (intable->is_background == 0) 
			outentries++;

	gst_buffer_unmap(inbuf, &inmap);
	/*
	 * allocate output buffer
	 */
	GstBuffer *outbuf = NULL;
	GstPad *srcpad = element->srcpad;
	GstCaps *caps = gst_pad_get_current_caps(srcpad);

	GST_LOG("Debug accumbackground srcpad %" GST_PTR_FORMAT, srcpad);
	GST_LOG("Debug accumbackground caps %" GST_PTR_FORMAT, caps);
	/* allocate extra space for prompt stats */	
	//int out_size = sizeof(PostcohInspiralTable) * outentries + sizeof(BackgroundStats) * ncombo;
	int out_size = sizeof(PostcohInspiralTable) * outentries;
	//result = gst_pad_alloc_buffer(srcpad, 0, out_size, caps, &outbuf);
	//if (result != GST_FLOW_OK) {
	//	GST_ERROR_OBJECT(srcpad, "Could not allocate postcoh-inspiral buffer %d", result);
	
	outbuf = gst_buffer_new_allocate(NULL, out_size, NULL);

	if(!outbuf)
	  return GST_FLOW_ERROR;
	
	gst_pad_push(srcpad, outbuf);

	/*
	 * update background rates
	 */

	gst_buffer_map(inbuf, &inmap, GST_MAP_READ);
	gst_buffer_map(outbuf, &outmap, GST_MAP_READWRITE);
	intable = (PostcohInspiralTable *) inmap.data;
	PostcohInspiralTable *outtable = (PostcohInspiralTable *) outmap.data;
	int isingle, icombo, nifo;
	for (; intable<intable_end; intable++) {
		//printf("is_back %d\n", intable->is_background);
		if (intable->is_background == 1) {
			//printf("cohsnr %f, maxsnr %f\n", intable->cohsnr, intable->maxsnglsnr);
			icombo = get_icombo(intable->ifos);
			if (icombo > -1)
				background_stats_rates_update((double)intable->cohsnr, (double)intable->cmbchisq, stats_snapshot[icombo]->rates, stats_snapshot[icombo]);

			nifo = strlen(intable->ifos)/IFO_LEN;
			/* add single detector stats */
			for (isingle=0; isingle< nifo; isingle++)
				background_stats_rates_update((double)(*(&(intable->snglsnr_L) + isingle)), (double)(*(&(intable->chisq_L) + isingle)), stats_snapshot[isingle]->rates, stats_snapshot[isingle]);
			/* add stats to stats_list for prompt FAP estimation */
			// if (icombo > -1)
			// 	background_stats_rates_update((double)intable->cohsnr, (double)intable->cmbchisq, cur_stats_in_list[icombo]->rates, cur_stats_in_list[icombo]);

		} else { /* coherent trigger entry */
			memcpy(outtable, intable, sizeof(PostcohInspiralTable));
			outtable++;
		} 
	}
	gst_buffer_unmap(inbuf, &inmap);
	gst_buffer_unmap(outbuf, &outmap);
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
	GstClockTime t_cur = GST_BUFFER_PTS(inbuf);
	element->t_end = t_cur;
	gint duration = (int) ((element->t_end - element->t_roll_start) / GST_SECOND);
	if (element->snapshot_interval > 0 && duration >= element->snapshot_interval) {
		gint gps_time = (int) (element->t_roll_start / GST_SECOND);
		GString *tmp_fname = g_string_new(element->output_fname_prefix);
		g_string_append_printf(tmp_fname, "_%d_%d.xml.gz", gps_time, duration);
		background_stats_to_xml(stats_snapshot, element->ncombo, element->hist_trials, tmp_fname->str);
		g_string_free(tmp_fname, TRUE);
		background_stats_reset(stats_snapshot, element->ncombo);
		element->t_roll_start = t_cur;
	}

	/*
	 * set the outbuf meta data
	 */
	GST_BUFFER_PTS(outbuf) = GST_BUFFER_PTS(inbuf);
	GST_BUFFER_DURATION(outbuf) = GST_BUFFER_DURATION(inbuf);
	GST_BUFFER_OFFSET(outbuf) = GST_BUFFER_OFFSET(inbuf);
	GST_BUFFER_OFFSET_END(outbuf) = GST_BUFFER_OFFSET_END(inbuf);
	gst_buffer_set_size(outbuf, sizeof(PostcohInspiralTable) * outentries);

	gst_buffer_unref(inbuf);
	result = gst_pad_push(srcpad, outbuf);

  GST_LOG_OBJECT (element, "pushed %s+%s buffer of %ld bytes, ts %"
      GST_TIME_FORMAT ", duration %" GST_TIME_FORMAT ", offset %"
      G_GINT64_FORMAT ", offset_end %" G_GINT64_FORMAT,
      GST_BUFFER_FLAG_IS_SET(outbuf, GST_BUFFER_FLAG_GAP) ? "GAP" : "NONGAP",
      GST_BUFFER_IS_DISCONT(outbuf) ? "DISCONT" : "CONT",
      gst_buffer_get_size(outbuf), GST_TIME_ARGS (GST_BUFFER_PTS(outbuf)),
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
cohfar_accumbackground_sink_event (GstPad * pad, GstObject *parent, GstEvent * event)
{
  CohfarAccumbackground *element = COHFAR_ACCUMBACKGROUND(parent);

  GST_DEBUG_OBJECT(element, "Debug accumbackground event %s", GST_EVENT_TYPE_NAME(event));
  GST_DEBUG_OBJECT(element, "Debug accumbackground element %" GST_PTR_FORMAT, element);
  GST_DEBUG_OBJECT(element, "Debug accumbackground pad %" GST_PTR_FORMAT, pad);

  switch (GST_EVENT_TYPE(event)) {
    case GST_EVENT_EOS:
//      if (fflush (sink->file))
//        goto flush_failed;

    GST_LOG_OBJECT(element, "EVENT EOS. ");
    if (element->snapshot_interval >= 0) {
	gint gps_time = (int) (element->t_roll_start / GST_SECOND);
	gint duration = (int) ((element->t_end - element->t_roll_start) / GST_SECOND);
	GString *tmp_fname = g_string_new(element->output_fname_prefix);
	g_string_append_printf(tmp_fname, "_%d_%d.xml.gz", gps_time, duration);
	background_stats_to_xml(element->stats_snapshot, element->ncombo, element->hist_trials, tmp_fname->str);
	g_string_free(tmp_fname, TRUE);
    }
      break;
    default:
      GST_DEBUG_OBJECT(event, "Debug accumbackground event default %" GST_PTR_FORMAT, event);
      break;
  }



  return gst_pad_event_default(pad, NULL, event);
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

		case PROP_OUTPUT_FNAME_PREFIX:
			element->output_fname_prefix = g_value_dup_string(value);
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

		case PROP_OUTPUT_FNAME_PREFIX:
			g_value_set_string(value, element->output_fname_prefix);
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
	G_OBJECT_CLASS(cohfar_accumbackground_parent_class)->dispose(object);
}


/*
 * base_init()
 */

#if 0
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
#endif

/*
 * class_init()
 */


static void cohfar_accumbackground_class_init(CohfarAccumbackgroundClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	gobject_class->set_property = GST_DEBUG_FUNCPTR(cohfar_accumbackground_set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(cohfar_accumbackground_get_property);
	gobject_class->dispose = GST_DEBUG_FUNCPTR(cohfar_accumbackground_dispose);

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
		PROP_OUTPUT_FNAME_PREFIX,
		g_param_spec_string(
			"output-fname-prefix",
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


static void cohfar_accumbackground_init(CohfarAccumbackground *element)
{
  GstPad *sinkpad;
  GstPad *srcpad;

  gst_element_create_all_pads(GST_ELEMENT(element));
  /*
  GstPadTemplate *src_template;
  GstPadTemplate *sink_template;

  src_template = gst_pad_template_new(
	"src",
	GST_PAD_SRC,
	GST_PAD_ALWAYS,
	gst_caps_from_string("application/x-lal-postcoh")
  );

  sink_template = gst_pad_template_new("sink",
      GST_PAD_SINK,
      GST_PAD_ALWAYS,
      gst_caps_from_string("application/x-lal-postcoh")
      );

	element->sinkpad = gst_pad_new_from_template(
		sink_template, "sink");
	gst_element_add_pad(GST_ELEMENT(element), element->sinkpad);

	element->srcpad = gst_pad_new_from_template(
		src_template, "src");
	gst_element_add_pad(GST_ELEMENT(element), element->srcpad);
*/
  
  sinkpad = gst_element_get_static_pad(GST_ELEMENT(element), "sink");
  element->sinkpad = sinkpad;
  srcpad = gst_element_get_static_pad(GST_ELEMENT(element), "src");
  element->srcpad = srcpad;
  gst_pad_set_event_function(sinkpad,
    GST_DEBUG_FUNCPTR(cohfar_accumbackground_sink_event));
  gst_pad_set_chain_function(sinkpad,
    GST_DEBUG_FUNCPTR(cohfar_accumbackground_chain));

  gst_pad_use_fixed_caps(srcpad);
  gst_pad_use_fixed_caps(sinkpad);
  //gst_object_unref(srcpad);
  //gst_object_unref(sinkpad);

  element->stats_snapshot = NULL;
  element->stats_prompt = NULL;
  element->t_roll_start = GST_CLOCK_TIME_NONE;
  element->snapshot_interval = NOT_INIT;
}

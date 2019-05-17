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

#include <postcohtable.h>
#include <cohfar/background_stats_utils.h>
#include <cohfar/cohfar_assignfar.h>

#include <time.h>
#define DEFAULT_STATS_NAME "stats.xml.gz"
/* make sure the far value to be 0 to indicate no background event yet, or > FLT_MIN */
#define BOUND(a,b) (((b)>0)?((b)<(a)?(a):(b)):0)

/* required minimal background events */
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
#define STATS_FNAME_1W_IDX 0
#define STATS_FNAME_1D_IDX 1
#define STATS_FNAME_2H_IDX 2

#define GST_CAT_DEFAULT cohfar_assignfar_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


static void additional_initializations(GType type)
{
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "cohfar_assignfar", 0, "cohfar_assignfar element");
}

GST_BOILERPLATE_FULL(
	CohfarAssignfar,
	cohfar_assignfar,
	GstBaseTransform,
	GST_TYPE_BASE_TRANSFORM,
	additional_initializations
);

enum property {
	PROP_0,
	PROP_IFO_SENSE,
	PROP_REFRESH_INTERVAL,
	PROP_SILENT_TIME,
	PROP_INPUT_FNAME
};

static void cohfar_assignfar_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * pspec);
static void cohfar_assignfar_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec);

/* vmethods */
static GstFlowReturn cohfar_assignfar_transform_ip (GstBaseTransform * base,
    GstBuffer * buf);
static void cohfar_assignfar_dispose (GObject *object);

static void update_trigger_fars_lr(PostcohInspiralTable *table, int icombo, CohfarAssignfar *element)
{
	TriggerStats *cur_stats = element->bgstats_1w->multistats[icombo];
	int hist_trials = element->hist_trials;
	double n_s = cur_stats->rank_nevent/ (cur_stats->rank_livetime * hist_trials);
	int rank_idx = get_rank_idx(table, element->bgstats_1w, icombo, element->sense_ratio);
	table->far_1w = BOUND(FLT_MIN, gsl_vector_get((gsl_vector*)cur_stats->rank->rank_fap->data, rank_idx) *n_s);
    table->rank = calc_lr(table, element->bgstats_1w, element->sense_ratio);
	cur_stats = element->bgstats_1d->multistats[icombo];
	n_s = cur_stats->rank_nevent/ (cur_stats->rank_livetime * hist_trials);
	table->far_1d = BOUND(FLT_MIN, gsl_vector_get((gsl_vector*)cur_stats->rank->rank_fap->data, rank_idx) *n_s);
	cur_stats = element->bgstats_2h->multistats[icombo];
	if (cur_stats->rank_livetime > 0) {
		n_s = cur_stats->rank_nevent/ (cur_stats->rank_livetime * hist_trials);
		table->far_2h = BOUND(FLT_MIN, gsl_vector_get((gsl_vector*)cur_stats->rank->rank_fap->data, rank_idx) *n_s);
	}
	/* FIXME: currently hardcoded for single detectors FAR */
	cur_stats = element->bgstats_1w->multistats[0];
	table->far_h_1w = BOUND(FLT_MIN, get_cdf_from_feature((double)table->snglsnr_H, (double)table->chisq_H, cur_stats)*cur_stats->feature_nevent/ (cur_stats->feature_livetime * hist_trials));
	cur_stats = element->bgstats_1w->multistats[1];
	table->far_l_1w = BOUND(FLT_MIN, get_cdf_from_feature((double)table->snglsnr_L, (double)table->chisq_L, cur_stats)*cur_stats->feature_nevent/ (cur_stats->feature_livetime * hist_trials));
	cur_stats = element->bgstats_1w->multistats[2];
	table->far_v_1w = BOUND(FLT_MIN, get_cdf_from_feature((double)table->snglsnr_V, (double)table->chisq_V, cur_stats)*cur_stats->feature_nevent/ (cur_stats->feature_livetime * hist_trials));
	cur_stats = element->bgstats_1d->multistats[0];
	table->far_h_1d = BOUND(FLT_MIN, get_cdf_from_feature((double)table->snglsnr_H, (double)table->chisq_H, cur_stats)*cur_stats->feature_nevent/ (cur_stats->feature_livetime * hist_trials));
	cur_stats = element->bgstats_1d->multistats[1];
	table->far_l_1d = BOUND(FLT_MIN, get_cdf_from_feature((double)table->snglsnr_L, (double)table->chisq_L, cur_stats)*cur_stats->feature_nevent/ (cur_stats->feature_livetime * hist_trials));
	cur_stats = element->bgstats_1d->multistats[2];
	table->far_v_1d = BOUND(FLT_MIN, get_cdf_from_feature((double)table->snglsnr_V, (double)table->chisq_V, cur_stats)*cur_stats->feature_nevent/ (cur_stats->feature_livetime * hist_trials));
	cur_stats = element->bgstats_2h->multistats[0];
	if (cur_stats->feature_livetime > 0) {
		table->far_h_2h = BOUND(FLT_MIN, get_cdf_from_feature((double)table->snglsnr_H, (double)table->chisq_H, cur_stats)*cur_stats->feature_nevent/ (cur_stats->feature_livetime * hist_trials));
		cur_stats = element->bgstats_2h->multistats[1];
		table->far_l_2h = BOUND(FLT_MIN, get_cdf_from_feature((double)table->snglsnr_L, (double)table->chisq_L, cur_stats)*cur_stats->feature_nevent/ (cur_stats->feature_livetime * hist_trials));
		cur_stats = element->bgstats_2h->multistats[2];
		table->far_v_2h = BOUND(FLT_MIN, get_cdf_from_feature((double)table->snglsnr_V, (double)table->chisq_V, cur_stats)*cur_stats->feature_nevent/ (cur_stats->feature_livetime * hist_trials));
	}
	icombo = get_icombo(table->ifos);
	GST_DEBUG_OBJECT(element, "gps %d, %s, trigger ifo %s, lr 1w %e, 1w p_noise %e, p_signal %e, p_null %e\n", table->end_time.gpsSeconds, table->ifos, table->pivotal_ifo, table->rank, get_prob_noise_sngl(icombo, table, element->bgstats_1w), get_prob_snrs(icombo, table, element->sense_ratio), get_prob_null(table));
	GST_DEBUG_OBJECT(element, "gps %d, cohsnr %f, cmbchisq %f, nullsnr %f, snr_l %f, snr_h %f, snr_v %f, chisq_l %f, chisq_h %f chisq_v %f far_1w %e far_1d %e far_2h %e idx 1w %d\n", table->end_time.gpsSeconds,table->cohsnr, table->cmbchisq, table->nullsnr, table->snglsnr_L, table->snglsnr_H, table->snglsnr_V, table->chisq_L, table->chisq_H, table->chisq_V, table->far_1w, table->far_1d, table->far_2h, rank_idx);
}

/*
 * ============================================================================
 *
 *                     GstBaseTransform Method Overrides
 *
 * ============================================================================
 */




/*
 * transform_ip()
 */


static GstFlowReturn cohfar_assignfar_transform_ip(GstBaseTransform *trans, GstBuffer *buf)
{
	CohfarAssignfar *element = COHFAR_ASSIGNFAR(trans);
	GstFlowReturn result = GST_FLOW_OK;


	GstClockTime t_cur = GST_BUFFER_TIMESTAMP(buf);
	if (!GST_CLOCK_TIME_IS_VALID(element->t_start))
		element->t_start = t_cur;

	/* Check that we have collected enough backgrounds */
	if (!GST_CLOCK_TIME_IS_VALID(element->t_roll_start)&& (t_cur - element->t_start)/GST_SECOND >= (unsigned) element->silent_time) {
		/* FIXME: the order of input fnames must match the stats order */
		element->pass_silent_time = TRUE;
		element->t_roll_start = t_cur;
		if (!trigger_stats_xml_from_xml(element->bgstats_1w, &(element->hist_trials), element->input_fnames[STATS_FNAME_1W_IDX])) {
			element->pass_silent_time = FALSE;
			element->t_roll_start = GST_CLOCK_TIME_NONE;
		}
		if (!trigger_stats_xml_from_xml(element->bgstats_1d, &(element->hist_trials), element->input_fnames[STATS_FNAME_1D_IDX])) {
			element->pass_silent_time = FALSE;
			element->t_roll_start = GST_CLOCK_TIME_NONE;
		}
		if (!trigger_stats_xml_from_xml(element->bgstats_2h, &(element->hist_trials), element->input_fnames[STATS_FNAME_2H_IDX])) {
			element->pass_silent_time = FALSE;
			element->t_roll_start = GST_CLOCK_TIME_NONE;
		}
		GST_DEBUG_OBJECT(element, "read input stats to assign far %s, %s, %s, read %s\n", element->input_fnames[STATS_FNAME_1W_IDX], element->input_fnames[STATS_FNAME_1D_IDX], element->input_fnames[STATS_FNAME_2H_IDX], element->pass_silent_time ? "True": "False");
	}

	/* Check if it is time to refresh the background stats */
	if (element->pass_silent_time && element->refresh_interval > 0 && (t_cur - element->t_roll_start)/GST_SECOND > (unsigned) element->refresh_interval) {
		element->t_roll_start = t_cur;
		/* FIXME: the order of input fnames must match the stats order */
		//printf("read refreshed stats to assign far.");
		if (!trigger_stats_xml_from_xml(element->bgstats_1w, &(element->hist_trials), element->input_fnames[STATS_FNAME_1W_IDX])) {
			printf("1w data no longer available\n");
		}

		if (!trigger_stats_xml_from_xml(element->bgstats_1d, &(element->hist_trials), element->input_fnames[STATS_FNAME_1D_IDX])) {
			printf("1d data no longer available\n");
		}
		if (!trigger_stats_xml_from_xml(element->bgstats_2h, &(element->hist_trials), element->input_fnames[STATS_FNAME_2H_IDX])) {
			printf("2h data no longer available\n");
		}
	}

	TriggerStats *cur_stats;
	if (element->pass_silent_time) {
		int icombo;
		PostcohInspiralTable *table = (PostcohInspiralTable *) GST_BUFFER_DATA(buf);
		PostcohInspiralTable *table_end = (PostcohInspiralTable *) (GST_BUFFER_DATA(buf) + GST_BUFFER_SIZE(buf));
		for (; table<table_end; table++) {
			if (table->is_background == FLAG_EMPTY)
				continue;
			icombo = get_icombo(table->ifos);
			icombo = scan_trigger_ifos(icombo, table);
			if (icombo < 0) {
				fprintf(stderr, "icombo not found, cohfar_assignfar\n");
				exit(0);
			}
			cur_stats = element->bgstats_1w->multistats[element->ncombo-1]; 
			if (icombo > -1 && cur_stats->rank_nevent > MIN_BACKGROUND_NEVENT)
			{
				update_trigger_fars_lr(table, element->ncombo-1, element); // always use the last bin ncombo-1
			}
		}
	}

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
cohfar_assignfar_event (GstBaseTransform * base, GstEvent * event)
{
  CohfarAssignfar *element = COHFAR_ASSIGNFAR(base);

  switch (GST_EVENT_TYPE(event)) {
    case GST_EVENT_EOS:
//      if (fflush (sink->file))
//        goto flush_failed;

    GST_LOG_OBJECT(element, "EVENT EOS. Finish assign far");
      break;
    default:
      break;
  }

  return TRUE;
}



/*
 * set_property()
 */


static void cohfar_assignfar_set_property(GObject *object, enum property prop_id, const GValue *value, GParamSpec *pspec)
{
	CohfarAssignfar *element = COHFAR_ASSIGNFAR(object);

	GST_OBJECT_LOCK(element);
	switch(prop_id) {
		case PROP_IFO_SENSE:
			element->ifo_sense = g_value_dup_string(value);
			set_sense_ratio(element->ifo_sense, &(element->nifo), &(element->ifos), element->sense_ratio);
			element->ncombo = get_ncombo(element->nifo);
			element->bgstats_1w = trigger_stats_xml_create(element->ifos, STATS_XML_TYPE_BACKGROUND);
			element->bgstats_1d = trigger_stats_xml_create(element->ifos, STATS_XML_TYPE_BACKGROUND);
			element->bgstats_2h = trigger_stats_xml_create(element->ifos, STATS_XML_TYPE_BACKGROUND);
			break;

		case PROP_INPUT_FNAME:
			/* must make sure ifos have been loaded */
			g_assert(element->ifos != NULL);
			element->input_fnames = g_strsplit(g_value_dup_string(value), ",", -1);
			gchar *ifname;
			for (ifname = element->input_fnames; *ifname; ifname++, element->ninput++);
			if (element->ninput != 3)
				printf("ninput %d\n", element->ninput);
			/* FIXME: only allow three inputs to set fars */
			// g_assert(element->ninput == 3);
			break;

		case PROP_SILENT_TIME:
			element->silent_time = g_value_get_int(value);
			break;


		case PROP_REFRESH_INTERVAL:
			element->refresh_interval = g_value_get_int(value);
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


static void cohfar_assignfar_get_property(GObject *object, enum property prop_id, GValue *value, GParamSpec *pspec)
{
	CohfarAssignfar *element = COHFAR_ASSIGNFAR(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
		case PROP_IFO_SENSE:
			g_value_set_string(value, element->ifo_sense);
			break;

		case PROP_INPUT_FNAME:
			g_value_set_string(value, element->input_fnames);
			break;

		case PROP_SILENT_TIME:
			g_value_set_int(value, element->silent_time);
			break;

		case PROP_REFRESH_INTERVAL:
			g_value_set_int(value, element->refresh_interval);
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


static void cohfar_assignfar_dispose(GObject *object)
{
	CohfarAssignfar *element = COHFAR_ASSIGNFAR(object);

	if(element->bgstats_1w) {
        trigger_stats_xml_destroy(element->bgstats_1w);
        trigger_stats_xml_destroy(element->bgstats_1d);
        trigger_stats_xml_destroy(element->bgstats_2h);
	}
	G_OBJECT_CLASS(parent_class)->dispose(object);
	g_strfreev(element->input_fnames);
}


/*
 * base_init()
 */


static void cohfar_assignfar_base_init(gpointer gclass)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(gclass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(gclass);

	gst_element_class_set_details_simple(
		element_class,
		"assign far to postcoh triggers",
		"assign far",
		"assign far to postcoh triggers according to a given stats file.\n",
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

	transform_class->transform_ip = GST_DEBUG_FUNCPTR(cohfar_assignfar_transform_ip);
	transform_class->event = GST_DEBUG_FUNCPTR(cohfar_assignfar_event);

}


/*
 * class_init()
 */


static void cohfar_assignfar_class_init(CohfarAssignfarClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
;
	gobject_class->set_property = GST_DEBUG_FUNCPTR(cohfar_assignfar_set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(cohfar_assignfar_get_property);
	gobject_class->dispose = GST_DEBUG_FUNCPTR(cohfar_assignfar_dispose);

	g_object_class_install_property(
		gobject_class,
		PROP_IFO_SENSE,
		g_param_spec_string(
			"ifo-sense",
			"ifo:horizon_distance",
			"ifos and horizon distances",
			NULL,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);

	g_object_class_install_property(
		gobject_class,
		PROP_INPUT_FNAME,
		g_param_spec_string(
			"input-fname",
			"input filename",
			"Input background statistics filename",
			NULL,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);

	g_object_class_install_property(
		gobject_class,
		PROP_REFRESH_INTERVAL,
		g_param_spec_int(
			"refresh-interval",
			"refresh interval",
			"(0) never refresh stats; (N) refresh stats every N seconds. ",
			0, G_MAXINT, 600,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);

	g_object_class_install_property(
		gobject_class,
		PROP_SILENT_TIME,
		g_param_spec_int(
			"silent-time",
			"background silent time",
			"(0) do not need background silent time; (N) allow N seconds to accumulate background.",
			0, G_MAXINT, G_MAXINT,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);

}
/*
 * init()
 */


static void cohfar_assignfar_init(CohfarAssignfar *element, CohfarAssignfarClass *kclass)
{
	element->ifos = NULL;
	element->ifo_sense = NULL;
	element->bgstats_2h = NULL;
	element->bgstats_1d = NULL;
	element->bgstats_1w = NULL;
	element->input_fnames = NULL;
	element->t_start = GST_CLOCK_TIME_NONE;
	element->t_roll_start = GST_CLOCK_TIME_NONE;
	element->pass_silent_time = FALSE;
	element->ninput = -1;
	int i;
	for (i=0; i< MAX_NBICOMBO; i++)
		element->sense_ratio[i] = 0;

}

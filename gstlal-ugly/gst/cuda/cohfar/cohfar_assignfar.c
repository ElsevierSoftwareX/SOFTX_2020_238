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
#include <cohfar/cohfar_assignfar.h>

#include <time.h>
#define DEFAULT_STATS_NAME "stats.xml.gz"
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
	PROP_IFOS,
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
		element->t_roll_start = t_cur;
		/* FIXME: the order of input fnames must match the stats order */
		GST_DEBUG_OBJECT(element, "read input stats to assign far %s, %s, %s", element->input_fnames[STATS_FNAME_1W_IDX], element->input_fnames[STATS_FNAME_1W_IDX], element->input_fnames[STATS_FNAME_1W_IDX]);
		background_stats_from_xml(element->stats_1w, element->ncombo, element->input_fnames[STATS_FNAME_1W_IDX]);
		background_stats_from_xml(element->stats_1d, element->ncombo, element->input_fnames[STATS_FNAME_1D_IDX]);
		background_stats_from_xml(element->stats_2h, element->ncombo, element->input_fnames[STATS_FNAME_2H_IDX]);
		element->pass_silent_time = TRUE;
	}

	/* Check if it is time to refresh the background stats */
	if (element->pass_silent_time && element->refresh_interval > 0 && (t_cur - element->t_roll_start)/GST_SECOND > (unsigned) element->refresh_interval) {
		element->t_roll_start = t_cur;
		/* FIXME: the order of input fnames must match the stats order */
		GST_DEBUG_OBJECT(element, "read refreshed stats to assign far.");
		background_stats_from_xml(element->stats_1w, element->ncombo, element->input_fnames[STATS_FNAME_1W_IDX]);
		background_stats_from_xml(element->stats_1d, element->ncombo, element->input_fnames[STATS_FNAME_1D_IDX]);
		background_stats_from_xml(element->stats_2h, element->ncombo, element->input_fnames[STATS_FNAME_2H_IDX]);
	}


	if (element->pass_silent_time) {
		int icombo;
		BackgroundStats **stats_1w = element->stats_1w;
		BackgroundStats **stats_1d = element->stats_1d;
		BackgroundStats **stats_2h = element->stats_2h;
		PostcohInspiralTable *table = (PostcohInspiralTable *) GST_BUFFER_DATA(buf);
		PostcohInspiralTable *table_end = (PostcohInspiralTable *) (GST_BUFFER_DATA(buf) + GST_BUFFER_SIZE(buf));
		for (; table<table_end; table++) {
			icombo = get_icombo(table->ifos);
			if (icombo > -1)
			{
				table->far_1w = background_stats_bins2D_get_val((double)table->cohsnr, (double)table->cmbchisq, stats_1w[icombo]->cdf)*stats_1w[icombo]->nevent/ stats_1w[icombo]->duration;
				table->far_1d = background_stats_bins2D_get_val((double)table->cohsnr, (double)table->cmbchisq, stats_1d[icombo]->cdf)*stats_1d[icombo]->nevent/ stats_1d[icombo]->duration;
				table->far_2h = background_stats_bins2D_get_val((double)table->cohsnr, (double)table->cmbchisq, stats_2h[icombo]->cdf)*stats_2h[icombo]->nevent/ stats_2h[icombo]->duration;
				/* FIXME: currently hardcoded for single detectors FAR */
				table->far_h = background_stats_bins2D_get_val((double)table->snglsnr_H, (double)table->chisq_H, stats_1w[1]->cdf)*stats_1w[1]->nevent/ stats_1w[1]->duration;
				table->far_l = background_stats_bins2D_get_val((double)table->snglsnr_L, (double)table->chisq_L, stats_1w[0]->cdf)*stats_1w[0]->nevent/ stats_1w[0]->duration;
				table->far_v = background_stats_bins2D_get_val((double)table->snglsnr_V, (double)table->chisq_V, stats_1w[2]->cdf)*stats_1w[2]->nevent/ stats_1w[2]->duration;
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
		case PROP_IFOS:
			element->ifos = g_value_dup_string(value);
			int nifo = strlen(element->ifos) / IFO_LEN;
			element->ncombo = get_ncombo(nifo);
			element->stats_1w = background_stats_create(element->ifos);
			element->stats_1d = background_stats_create(element->ifos);
			element->stats_2h = background_stats_create(element->ifos);
			break;

		case PROP_INPUT_FNAME:
			/* must make sure ifos have been loaded */
			g_assert(element->ifos != NULL);
			element->input_fnames = g_strsplit(g_value_dup_string(value), ",", -1);
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
		case PROP_IFOS:
			g_value_set_string(value, element->ifos);
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

	if(element->stats_1w) {
		// FIXME: free stats
	}
	G_OBJECT_CLASS(parent_class)->dispose(object);
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
			0, G_MAXINT, 36000,
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
	element->stats_2h = NULL;
	element->stats_1d = NULL;
	element->stats_1w = NULL;
	element->input_fnames = NULL;
	element->t_start = GST_CLOCK_TIME_NONE;
	element->t_roll_start = GST_CLOCK_TIME_NONE;
	element->pass_silent_time = FALSE;
}

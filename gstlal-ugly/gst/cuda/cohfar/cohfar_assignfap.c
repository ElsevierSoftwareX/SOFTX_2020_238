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
#include <cohfar/cohfar_assignfap.h>

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


#define GST_CAT_DEFAULT cohfar_assignfap_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


static void additional_initializations(GType type)
{
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "cohfar_assignfap", 0, "cohfar_assignfap element");
}

GST_BOILERPLATE_FULL(
	CohfarAssignfap,
	cohfar_assignfap,
	GstBaseTransform,
	GST_TYPE_BASE_TRANSFORM,
	additional_initializations
);

enum property {
	PROP_0,
	PROP_IFOS,
	PROP_REFRESH_INTERVAL,
	PROP_COLLECTION_TIME,
	PROP_INPUT_FNAME
};

static void cohfar_assignfap_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * pspec);
static void cohfar_assignfap_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec);

/* vmethods */
static GstFlowReturn cohfar_assignfap_transform_ip (GstBaseTransform * base,
    GstBuffer * buf);
static void cohfar_assignfap_dispose (GObject *object);

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


static GstFlowReturn cohfar_assignfap_transform_ip(GstBaseTransform *trans, GstBuffer *buf)
{
	CohfarAssignfap *element = COHFAR_ASSIGNFAP(trans);
	GstFlowReturn result = GST_FLOW_OK;


	GstClockTime t_cur = GST_BUFFER_TIMESTAMP(buf);
	if (!GST_CLOCK_TIME_IS_VALID(element->t_start))
		element->t_start = t_cur;

	if (!GST_CLOCK_TIME_IS_VALID(element->t_roll_start)&& (t_cur - element->t_start)/GST_SECOND >= (unsigned) element->collection_time) {
		element->t_roll_start = t_cur;
		background_stats_from_xml(element->stats, element->ncombo, element->input_fname);
		element->pass_collection_time = TRUE;
	}

	if (element->refresh_interval > 0 && (t_cur - element->t_roll_start)/GST_SECOND > (unsigned) element->refresh_interval) {
		element->t_roll_start = t_cur;
		background_stats_from_xml(element->stats, element->ncombo, element->input_fname);
	}


	if (element->pass_collection_time) {
		int icombo;
		BackgroundStats **stats = element->stats;
		PostcohInspiralTable *table = (PostcohInspiralTable *) GST_BUFFER_DATA(buf);
		PostcohInspiralTable *table_end = (PostcohInspiralTable *) (GST_BUFFER_DATA(buf) + GST_BUFFER_SIZE(buf));
		for (; table<table_end; table++) {
			icombo = get_icombo(table->ifos);
			table->fap = background_stats_bins2D_get_val(table->cohsnr, table->chisq, stats[icombo]->cdf);
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
cohfar_assignfap_event (GstBaseTransform * base, GstEvent * event)
{
  CohfarAssignfap *element = COHFAR_ASSIGNFAP(base);

  switch (GST_EVENT_TYPE(event)) {
    case GST_EVENT_EOS:
//      if (fflush (sink->file))
//        goto flush_failed;

    GST_LOG_OBJECT(element, "EVENT EOS. Finish assign fap");
      break;
    default:
      break;
  }

  return TRUE;
}



/*
 * set_property()
 */


static void cohfar_assignfap_set_property(GObject *object, enum property prop_id, const GValue *value, GParamSpec *pspec)
{
	CohfarAssignfap *element = COHFAR_ASSIGNFAP(object);

	GST_OBJECT_LOCK(element);
	switch(prop_id) {
		case PROP_IFOS:
			element->ifos = g_value_dup_string(value);
			int nifo = strlen(element->ifos) / IFO_LEN;
			element->ncombo = pow(2, nifo) - 1 - nifo;
			element->stats = background_stats_create(element->ifos);
			break;

		case PROP_INPUT_FNAME:
			/* must make sure ifos have been loaded */
			g_assert(element->ifos != NULL);
			element->input_fname = g_value_dup_string(value);
			background_stats_from_xml(element->stats, element->ncombo, element->input_fname);
			break;

		case PROP_COLLECTION_TIME:
			element->collection_time = g_value_get_int(value);
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


static void cohfar_assignfap_get_property(GObject *object, enum property prop_id, GValue *value, GParamSpec *pspec)
{
	CohfarAssignfap *element = COHFAR_ASSIGNFAP(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
		case PROP_IFOS:
			g_value_set_string(value, element->ifos);
			break;

		case PROP_INPUT_FNAME:
			g_value_set_string(value, element->input_fname);
			break;

		case PROP_COLLECTION_TIME:
			g_value_set_int(value, element->collection_time);
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


static void cohfar_assignfap_dispose(GObject *object)
{
	CohfarAssignfap *element = COHFAR_ASSIGNFAP(object);

	if(element->stats) {
		// FIXME: free stats
	}
	G_OBJECT_CLASS(parent_class)->dispose(object);
}


/*
 * base_init()
 */


static void cohfar_assignfap_base_init(gpointer gclass)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(gclass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(gclass);

	gst_element_class_set_details_simple(
		element_class,
		"assign fap to postcoh triggers",
		"assign fap",
		"assign fap to postcoh triggers according to a given stats file.\n",
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

	transform_class->transform_ip = GST_DEBUG_FUNCPTR(cohfar_assignfap_transform_ip);
	transform_class->event = GST_DEBUG_FUNCPTR(cohfar_assignfap_event);

}


/*
 * class_init()
 */


static void cohfar_assignfap_class_init(CohfarAssignfapClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
;
	gobject_class->set_property = GST_DEBUG_FUNCPTR(cohfar_assignfap_set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(cohfar_assignfap_get_property);
	gobject_class->dispose = GST_DEBUG_FUNCPTR(cohfar_assignfap_dispose);

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
		PROP_COLLECTION_TIME,
		g_param_spec_int(
			"collection-time",
			"background collection time",
			"(0) do not need background collection time; (N) allow N seconds to accumulate background.",
			0, G_MAXINT, 36000,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);

}
/*
 * init()
 */


static void cohfar_assignfap_init(CohfarAssignfap *element, CohfarAssignfapClass *kclass)
{
	element->ifos = NULL;
	element->stats = NULL;
	element->t_start = GST_CLOCK_TIME_NONE;
	element->t_roll_start = GST_CLOCK_TIME_NONE;
	element->pass_collection_time = FALSE;
}

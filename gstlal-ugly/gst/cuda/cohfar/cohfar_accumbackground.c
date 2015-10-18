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

#include <postcoh/postcoh_table.h>
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
	GstBaseTransform,
	GST_TYPE_BASE_TRANSFORM,
	additional_initializations
);

enum property {
	PROP_0,
	PROP_IFOS,
	PROP_UPDATE_INTERVAL,
	PROP_HISTORY_FNAME,
	PROP_OUTPUT_FNAME
};

static void cohfar_accumbackground_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * pspec);
static void cohfar_accumbackground_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec);

/* vmethods */

static gboolean
cohfar_accumbackground_get_unit_size (GstBaseTransform * base, GstCaps * caps,
    guint * size);
static GstFlowReturn cohfar_accumbackground_transform (GstBaseTransform * base,
    GstBuffer * inbuf, GstBuffer * outbuf);
static gboolean cohfar_accumbackground_transform_size (GstBaseTransform * base,
   GstPadDirection direction, GstCaps * caps, guint size, GstCaps * othercaps,
    guint * othersize);
static gboolean cohfar_accumbackground_event (GstBaseTransform * base,
    GstEvent * event);
static void cohfar_accumbackground_dispose (GObject *object);

/*
 * ============================================================================
 *
 *                     GstBaseTransform Method Overrides
 *
 * ============================================================================
 */

static gboolean
cohfar_accumbackground_get_unit_size (GstBaseTransform * base, GstCaps * caps,
    guint * size)
{
	*size = sizeof(PostcohTable);
  return TRUE;
}

/*
 * transform_size()
 */


static gboolean cohfar_accumbackground_transform_size(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, guint size, GstCaps *othercaps, guint *othersize)
{
	CohfarAccumbackground *element = COHFAR_ACCUMBACKGROUND(trans);
  GST_LOG_OBJECT (trans, "asked to transform size %d in direction %s",
      size, direction == GST_PAD_SINK ? "SINK" : "SRC");


	switch(direction) {
	case GST_PAD_SRC:
	case GST_PAD_SINK:
		/*
		 * background entries will be eliminated from the table
		 */

		*othersize = size;

		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		return FALSE;
	}

  GST_LOG_OBJECT (trans, "transformed size %d to %d", size,
      *othersize);


	return TRUE;
}




/*
 * transform()
 */


static GstFlowReturn cohfar_accumbackground_transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf)
{
	CohfarAccumbackground *element = COHFAR_ACCUMBACKGROUND(trans);
	GstFlowReturn result = GST_FLOW_OK;

  GST_LOG_OBJECT (element, "transforming accum %s+%s buffer of %ld bytes, ts %"
      GST_TIME_FORMAT ", duration %" GST_TIME_FORMAT ", offset %"
      G_GINT64_FORMAT ", offset_end %" G_GINT64_FORMAT,
      GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP) ? "GAP" : "NONGAP",
      GST_BUFFER_IS_DISCONT(inbuf) ? "DISCONT" : "CONT",
      GST_BUFFER_SIZE(inbuf), GST_TIME_ARGS (GST_BUFFER_TIMESTAMP (inbuf)),
      GST_TIME_ARGS (GST_BUFFER_DURATION (inbuf)),
      GST_BUFFER_OFFSET (inbuf), GST_BUFFER_OFFSET_END (inbuf));



	if (!GST_CLOCK_TIME_IS_VALID(element->t_roll_start))
		element->t_roll_start = GST_BUFFER_TIMESTAMP(inbuf);

	/*
	 * update background rates
	 */

	int icombo, outentries = 0;
	BackgroundStats **stats = element->stats;
	PostcohTable *intable = (PostcohTable *) GST_BUFFER_DATA(inbuf);
	PostcohTable *intable_end = (PostcohTable *) (GST_BUFFER_DATA(inbuf) + GST_BUFFER_SIZE(inbuf));
	PostcohTable *outtable = (PostcohTable *) GST_BUFFER_DATA(outbuf);
	for (; intable<intable_end; intable++) {
		printf("is_back %d\n", intable->is_background);
		if (intable->is_background == 1) {
			printf("cohsnr %f, maxsnr %f\n", intable->cohsnr, intable->maxsnglsnr);
			icombo = get_icombo(intable->ifos);
			background_stats_rates_update(intable->cohsnr, intable->chisq, stats[icombo]->rates);
		} else { /* coherent trigger entry */
			memcpy(outtable, intable, sizeof(PostcohTable));
			outtable++;
			outentries++;
		} 
	}

	GstClockTime t_cur = GST_BUFFER_TIMESTAMP(inbuf);
	if (element->update_interval > 0 && (t_cur - element->t_roll_start)/GST_SECOND > (unsigned) element->update_interval) {
		/* update background xml file */
		background_stats_to_xml(stats, element->ncombo, element->output_fname);
		element->t_roll_start = t_cur;
	}
	/*
	 * set the outbuf meta data
	 */
	GST_BUFFER_TIMESTAMP(outbuf) = GST_BUFFER_TIMESTAMP(inbuf);
	GST_BUFFER_DURATION(outbuf) = GST_BUFFER_DURATION(inbuf);
	GST_BUFFER_OFFSET(outbuf) = GST_BUFFER_OFFSET(inbuf);
	GST_BUFFER_OFFSET_END(outbuf) = GST_BUFFER_OFFSET_END(inbuf);
	GST_BUFFER_SIZE(outbuf) = sizeof(PostcohTable) * outentries;


  GST_LOG_OBJECT (element, "transformed %s+%s buffer of %ld bytes, ts %"
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
cohfar_accumbackground_event (GstBaseTransform * base, GstEvent * event)
{
  CohfarAccumbackground *element = COHFAR_ACCUMBACKGROUND(base);

  switch (GST_EVENT_TYPE(event)) {
    case GST_EVENT_EOS:
//      if (fflush (sink->file))
//        goto flush_failed;

    GST_LOG_OBJECT(element, "EVENT EOS. ");
    if (element->update_interval >= 0)
      background_stats_to_xml(element->stats, element->ncombo, element->output_fname);
      break;
    default:
      break;
  }

  return TRUE;
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
			int nifo = strlen(element->ifos) / IFO_LEN;
			element->ncombo = pow(2, nifo) - 1 - nifo;
			element->stats = background_stats_create(element->ifos);
			break;

		case PROP_HISTORY_FNAME:

			/* must make sure ifos have been loaded */
			g_assert(element->ifos != NULL);
			element->history_fname = g_value_dup_string(value);
			background_stats_from_xml(element->stats, element->ncombo, element->history_fname);
			break;

		case PROP_OUTPUT_FNAME:
			element->output_fname = g_value_dup_string(value);
			break;


		case PROP_UPDATE_INTERVAL:
			element->update_interval = g_value_get_int(value);
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

		case PROP_UPDATE_INTERVAL:
			g_value_set_int(value, element->update_interval);
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

	g_mutex_free(element->prop_lock);
	element->prop_lock = NULL;
	g_cond_free(element->prop_avail);
	element->prop_avail = NULL;
	if(element->stats) {
		// FIXME: free stats
	}
	G_OBJECT_CLASS(parent_class)->dispose(object);
}


/*
 * base_init()
 */


static void cohfar_accumbackground_base_init(gpointer gclass)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(gclass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(gclass);

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

	transform_class->transform = GST_DEBUG_FUNCPTR(cohfar_accumbackground_transform);
	transform_class->transform_size = GST_DEBUG_FUNCPTR(cohfar_accumbackground_transform_size);
	transform_class->get_unit_size = GST_DEBUG_FUNCPTR(cohfar_accumbackground_get_unit_size);
	transform_class->event = GST_DEBUG_FUNCPTR(cohfar_accumbackground_event);

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
		PROP_UPDATE_INTERVAL,
		g_param_spec_int(
			"update-interval",
			"update interval",
			"(-1) never update; (0) update at the end; (N) update background statistics xml file every N seconds.",
			-1, G_MAXINT, 600,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);

}
/*
 * init()
 */


static void cohfar_accumbackground_init(CohfarAccumbackground *element, CohfarAccumbackgroundClass *kclass)
{
	element->stats = NULL;
	element->update_interval = NOT_INIT;
	element->prop_lock = g_mutex_new();
	element->prop_avail = g_cond_new();
}

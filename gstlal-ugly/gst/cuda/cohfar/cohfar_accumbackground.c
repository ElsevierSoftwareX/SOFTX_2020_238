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

#include <string.h>
/*
 *  stuff from gobject/gstreamer
*/


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <gstlal/gstlal.h>


/*
 * stuff from FFTW and GSL
 */


#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

#include <time.h>
#define DEFAULT_STATS_NAME "stats.xml.gz"
/*
 * ============================================================================
 *
 *                                 Utilities
 *
 * ============================================================================
 */
static int get_icombo(char *ifos) {
	int icombo = 0;
	for (icombo=0; icombo<MAX_COMBOS; icombo++) {
		if (strcmp(ifos, IFO_COMBO_MAP[icombo]) == 0)
			return icombo;
	}
	return -1;
}

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
	PROP_IFOS,
	PROP_HIST_TRIALS,
	PROP_UPDATE_INTERVAL,
	PROP_INPUT_FNAME,
	PROP_OUTPUT_FILENAME
};

static void cohfar_accumbackground_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * pspec);
static void cohfar_accumbackground_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec);

/* vmethods */
static GstFlowReturn cohfar_accumbackground_transform (GstBaseTransform * base,
    GstBuffer * inbuf, GstBuffer * outbuf);
static gboolean cohfar_accumbackground_transform_size (GstBaseTransform * base,
   GstPadDirection direction, GstCaps * caps, guint size, GstCaps * othercaps,
    guint * othersize);
static gboolean cohfar_accumbackground_event (GstBaseTransform * base,
    GstEvent * event);
static gboolean cohfar_accumbackground_dispose (GObject *object);
/
/*
 * ============================================================================
 *
 *                     GstBaseTransform Method Overrides
 *
 * ============================================================================
 */

/*
 * transform_size()
 */


static gboolean cohfar_accumbackground_transform_size(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, guint size, GstCaps *othercaps, guint *othersize)
{
	CohfarAccumbackground *element = COHFAR_ACCUMBACKGROUND(trans);
  GST_LOG_OBJECT (base, "asked to transform size %d in direction %s",
      size, direction == GST_PAD_SINK ? "SINK" : "SRC");


	switch(direction) {
	case GST_PAD_SRC:
		/*
		 * background entries will be eliminated from the table
		 */

		g_mutex_lock(element->prop_lock);
		while(element->hist_trials == NOT_INIT)
			g_cond_wait(element->prop_avail, element->prop_lock);

		*othersize = size * (1 + element->hist_trials);

		break;

	case GST_PAD_SINK:
		/*
		 * background entries will be eliminated from the table
		 */

		g_mutex_lock(element->prop_lock);
		while(element->hist_trials == NOT_INIT)
			g_cond_wait(element->prop_avail, element->prop_lock);

		*othersize = size / (1 + element->hist_trials);


	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		return FALSE;
	}

  GST_LOG_OBJECT (base, "transformed size %d to %d", size,
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


	if (!GST_CLOCK_TIME_IS_VALID(element->t_roll_start))
		element->t_roll_start = GST_BUFFER_TIMESTAMP(inbuf);
	/*
	 * set the outbuf meta data
	 */
	GST_BUFFER_TIMESTAMP(outbuf) = GST_BUFFER_TIMESTAMP(inbuf);
	GST_BUFFER_DURATION(outbuf) = GST_BUFFER_DURATION(inbuf);
	GST_BUFFER_OFFSET(outbuf) = GST_BUFFER_OFFSET(inbuf);
	GST_BUFFER_OFFSET_END(outbuf) = GST_BUFFER_OFFSET_END(inbuf);


	/*
	 * update background rates
	 */

	int icombo;
	BackgroundStats **stats = element->stats;
	PostcohTable *intable = (PostcohTable *) GST_BUFFER_DATA(inbuf);
	PostcohTable *intable_end = (PostcohTable *) (GST_BUFFER_DATA(inbuf) + GST_BUFFER_SIZE(inbuf));
	PostcohTable *outtable = (PostcohTable *) GST_BUFFER_DATA(outbuf);
	for (; intable<intable_end; intable++) {
		if (intable->is_background == 1) {
			icombo = get_icombo(intable->ifos);
			add_background_val_to_rates(intable->snr, stats[icombo]->rates->logsnr_bins);
			add_background_val_to_rates(intable->chisq, stats[icombo]->rates->logchisq_bins);
		} else { /* coherent trigger entry */
			memcpy(outtable, intable, sizeof(PostcohTable));
			outtable++;
		} 
	}

	GstClockTime t_cur = GST_BUFFER_TIMESTAMP(buf);
	if (element->update_interval > 0 && (t_cur - element->t_roll_start)/GST_SECOND > (unsigned) element->update_interval) {
		/* update background xml file */
		background_stats_to_xml(stats, element->ncombo, element->output_fname);
		element->t_roll_start = t_cur;
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
cohfar_accumbackground_event (GstBaseTransform * base, GstEvent * event)
{
  CohfarAccumbackground *element = COHFAR_ACCUMBACKGROUND(base);

  switch (GST_EVENT_TYPE(event)) {
    case GST_EVENT_EOS:
//      if (fflush (sink->file))
//        goto flush_failed;

    GST_LOG_OBJECT(sink, "EVENT EOS. Finish writing document");
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
			int nifo = strlen(ifos) / IFO_LEN;
			element->ncombo = power(2, nifo) - 1 - nifo;
			element->stats = background_stats_create(element->ifos);
			break;

		case PROP_INPUT_FNAME:

			/* must make sure ifos have been loaded */
			g_assert(element->ifos != NULL);
			element->input_fname = g_value_dup_string(value);
			background_stats_from_xml(element->stats, element->ncombo, element->input_fname);
			break;

		case PROP_OUTPUT_FNAME:
			element->output_fname = g_value_dup_string(value);
			break;

		case PROP_HIST_TRIALS:
			g_mutex_lock(element->prop_lock);
			element->hist_trials = g_value_get_int(value);
			g_cond_broadcast(element->prop_avail);
			g_mutex_unlock(element->prop_lock);
			break;


		case PROP_UPDATE_INTERVAL:
			element->update_interval = g_value_get_int(value);
			break;
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

		case PROP_INPUT_FNAME:
			g_value_set_string(value, element->input_fname);
			break;

		case PROP_OUTPUT_FNAME:
			g_value_set_string(value, element->output_fname);
			break;

		case PROP_HIST_TRIALS:
			g_value_set_int(value, element->hist_trials);
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

	transform_class->transform = GST_DEBUG_FUNCPTR(cohfar_upbackgrond_transform);
	transform_class->transform_size = GST_DEBUG_FUNCPTR(cohfar_accumbackground_transform_size);
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
		PROP_INPUT_FNAME,
		g_param_spec_string(
			"input-fname",
			"Input filename",
			"Reference input background statstics filename",
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
		PROP_HIST_TRIALS,
		g_param_spec_int(
			"hist-trials",
			"history trials",
			"history triggers that should be kept in times",
			0, G_MAXINT, 1,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);

	g_object_class_install_property(
		gobject_class,
		PROP_UPDATE_INTERVAL,
		g_param_spec_int(
			"update-interval",
			"update interval",
			"update background statistics xml file every update time",
			1, G_MAXINT, 600,
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
	element->hist_trials = NOT_INIT;
	element->update_interval = 0;
	element->prop_lock = g_mutex_new();
	element->prop_avail = g_cond_new();
}

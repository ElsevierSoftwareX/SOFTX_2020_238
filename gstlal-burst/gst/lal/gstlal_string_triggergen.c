/*
 *======================================================
 *
 *			Preamble
 *
 *======================================================
 */


/*
 * stuff from the C library
 */


#include <math.h>
#include <string.h>
#include <stdio.h>

/*
 * stuff from gobject/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/audio/audio.h>
#include <gst/base/gstbasetransform.h>


/*
 * stuff from LAL
 */


#include <lal/Date.h>
#include <lal/LIGOMetadataTables.h>
#include <lal/LIGOLwXMLBurstRead.h>
#include <lal/SnglBurstUtils.h>


/*
 * stuff from gstlal
 */


#include <gstlal_string_triggergen.h>
#include <gstlal/gstaudioadapter.h>
#include <gstlal/gstlal_debug.h>

/*
 *======================================================
 *
 *                GStreamer Boiler Plate
 *
 *======================================================
 */

#define GST_CAT_DEFAULT gstlal_string_triggergen_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);

G_DEFINE_TYPE_WITH_CODE(
	GSTLALStringTriggergen,
	gstlal_string_triggergen,
	GST_TYPE_BASE_TRANSFORM,
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "lal_string_triggergen", 0, "lal_string_triggergen element")
);


/*
 *======================================================
 *
 *               	Parameters
 *
 *======================================================
 */


#define DEFAULT_THRES 5.5
#define DEFAULT_CLUSTER 0.1


/*
 *======================================================
 *
 *          		Utilities
 *
 *======================================================
 */


static void free_bankfile(GSTLALStringTriggergen *element)
{
	g_free(element->bank_filename);
	element->bank_filename = NULL;
	free(element->bank);
	element->bank = NULL;
	free(element->last_time);
	element->last_time = NULL;
	element->num_templates = 0;
}


static int setup_bankfile_input(GSTLALStringTriggergen *element, char *bank_filename)
{
	SnglBurst *bank = NULL;
	int i;

	free_bankfile(element);

	element->bank_filename = bank_filename;
	bank = XLALSnglBurstTableFromLIGOLw(element->bank_filename);
	element->num_templates = XLALSnglBurstTableLength(bank);
	element->bank = calloc(element->num_templates, sizeof(*element->bank));
	element->last_time = calloc(element->num_templates, sizeof(*element->last_time));
	if(!bank || !element->bank || !element->last_time) {
		XLALDestroySnglBurstTable(bank);
		free(element->bank);
		element->bank = NULL;
		free(element->last_time);
		element->last_time = NULL;
		return -1;
	}


	for(i=0; bank; i++) {
		SnglBurst *next = bank->next;
		g_assert(i < element->num_templates);
		element->bank[i] = *bank;
		element->bank[i].next = NULL;
		free(bank);
		bank = next;
	}
	return element->num_templates;

}


/*
 *======================================================
 *
 *               Trigger Generator
 *
 *======================================================
 */


static GstFlowReturn trigger_generator(GSTLALStringTriggergen *element, GstBuffer *inbuf, GstBuffer *outbuf)
{
	/*
	 * find events. we do not use chisq.
	 */

	GstMapInfo inmap;
	float *snrdata;
	SnglBurst *triggers = NULL;
	guint ntriggers = 0;
	guint64 t0;
	guint64 length;
	guint sample;
	gint channel;

	g_mutex_lock(&element->bank_lock);
	gst_buffer_map(inbuf, &inmap, GST_MAP_WRITE);

	snrdata = (float *) inmap.data;

	t0 = GST_BUFFER_PTS(inbuf);
	length = GST_BUFFER_OFFSET_END(inbuf) - GST_BUFFER_OFFSET(inbuf);

	GST_DEBUG_OBJECT(element, "searching %" G_GUINT64_FORMAT " samples at %" GST_TIME_SECONDS_FORMAT " for events with SNR greater than %f", length, GST_TIME_SECONDS_ARGS(t0),element->threshold);
	for(sample = 0; sample < length; sample++){
		LIGOTimeGPS t;
		XLALINT8NSToGPS(&t, t0);
		XLALGPSAdd(&t, (float) sample / GST_AUDIO_INFO_RATE(&element->audio_info));

		for(channel = 0; channel < element->num_templates; channel++) {
			float snr = fabsf(*snrdata++);
			if(snr >= element->threshold) {
				element->last_time[channel] = t;
				if(snr > element->bank[channel].snr) {
					/*
					 * Higher SNR than the "current winner". Update.
					 */
					element->bank[channel].snr = snr;
					element->bank[channel].peak_time = t;
				}
			} else if(element->bank[channel].snr != 0. && XLALGPSDiff(&t, &element->last_time[channel]) > element->cluster) {
				/*
				 * Trigger is ready to be passed.
				 * Push trigger to buffer, and reset it.
				 */
				triggers = g_renew(SnglBurst, triggers, ntriggers + 1);
				triggers[ntriggers++] = element->bank[channel];
				element->bank[channel].snr = 0.0;
			}
		}
	}
	g_mutex_unlock(&element->bank_lock);

	gst_buffer_unmap(inbuf,&inmap);

	if(ntriggers)
		gst_buffer_replace_all_memory(outbuf, gst_memory_new_wrapped(GST_MEMORY_FLAG_PHYSICALLY_CONTIGUOUS, triggers, ntriggers * sizeof(*triggers), 0, ntriggers * sizeof(*triggers), triggers, g_free));
	else
		gst_buffer_remove_all_memory(outbuf);

	GST_BUFFER_OFFSET_END(outbuf) = GST_BUFFER_OFFSET(outbuf) + ntriggers;

	return GST_FLOW_OK;
}


/*
 *======================================================
 *
 *          GstBaseTransform Method Overrides
 *
 *======================================================
 */


/*
 * get_unit_size()
 */


static gboolean get_unit_size(GstBaseTransform *trans, GstCaps *caps, gsize *size)
{
	GstAudioInfo info;
	gboolean success = gst_audio_info_from_caps(&info, caps);

	if(success)
		*size = GST_AUDIO_INFO_BPF(&info);
	else {
		GstCaps * src_caps=gst_pad_get_pad_template_caps(GST_BASE_TRANSFORM_SRC_PAD(trans));
		if(gst_caps_is_strictly_equal (caps, src_caps)){
			*size = sizeof(SnglBurst);
			success = TRUE;
		}
		gst_caps_unref(src_caps);
	}

	return success;
}


/*
 * transform caps()
 */

static GstCaps *transform_caps(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, GstCaps *filter)
{
  /*
   * always return the template caps of the other pad
   * FIXME the number of templates is fixed, so need to constrain the sink pad
   */

  switch (direction) {
    case GST_PAD_SRC:
      caps = gst_pad_get_pad_template_caps(GST_BASE_TRANSFORM_SINK_PAD(trans));
      break;

    case GST_PAD_SINK:
      caps = gst_pad_get_pad_template_caps(GST_BASE_TRANSFORM_SRC_PAD(trans));
      break;

    default:
      GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction"));
      caps = GST_CAPS_NONE;
      gst_caps_ref(caps);
      break;
  }

  if(filter) {
    GstCaps *intersection = gst_caps_intersect(caps, filter);
    gst_caps_unref(caps);
    caps = intersection;
  }

  return caps;
}


/*
 * set caps()
 */


static gboolean set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps)
{
	GSTLALStringTriggergen *element = GSTLAL_STRING_TRIGGERGEN(trans);
	gboolean success = gst_audio_info_from_caps(&element->audio_info, incaps);

	g_object_set(element->adapter, "unit-size", GST_AUDIO_INFO_WIDTH(&element->audio_info) / 8, NULL);

	return success;
}


/*
 * start()
 */

static gboolean start(GstBaseTransform *trans)
{
	GSTLALStringTriggergen *element = GSTLAL_STRING_TRIGGERGEN(trans);
	gint i;

	for(i=0; i < element->num_templates; i++) {
		/*
		 * initialize data in template. the snr is 0'ed so that
		 * when the templates are used to initialize the last event
		 * info that field is set properly.
		 */

		XLALINT8NSToGPS(&element->bank[i].peak_time, 0);
		element->bank[i].snr = 0;

		/* initialize the last time array, too */

		XLALINT8NSToGPS(&element->last_time[i], 0);
	}

	return TRUE;
}


/*
 * prepare_output_buffer()
 */


static GstFlowReturn prepare_output_buffer(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer **outbuf)
{
	*outbuf = gst_buffer_new();

	return GST_FLOW_OK;
}


/*
 * transform()
 */


static GstFlowReturn transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf)
{
	GSTLALStringTriggergen *element = GSTLAL_STRING_TRIGGERGEN(trans);
	GstFlowReturn result;

	result = trigger_generator(element,inbuf,outbuf);

	/*
	 * done
	 */

	return result;
}


/*
 *======================================================
 *
 *               GObject Method Overrides
 *
 *======================================================
 */

/*
 * properties
 */

enum property {
	ARG_THRES = 1,
	ARG_CLUSTER,
	ARG_BANK_FILENAME
};


static void set_property(GObject *object, enum property prop_id, const GValue *value, GParamSpec *pspec)
{
	GSTLALStringTriggergen *element = GSTLAL_STRING_TRIGGERGEN(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_THRES:
		element->threshold = g_value_get_float(value);
		break;

	case ARG_CLUSTER:
		element->cluster = g_value_get_float(value);
		break;

	case ARG_BANK_FILENAME:
		g_mutex_lock(&element->bank_lock);
		setup_bankfile_input(element, g_value_dup_string(value));
		g_mutex_unlock(&element->bank_lock);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, enum property prop_id, GValue *value, GParamSpec *pspec)
{
	GSTLALStringTriggergen *element = GSTLAL_STRING_TRIGGERGEN(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_THRES:
		g_value_set_float(value, element->threshold);
		break;

	case ARG_CLUSTER:
		g_value_set_float(value, element->cluster);
		break;

	case ARG_BANK_FILENAME:
		g_mutex_lock(&element->bank_lock);
		g_value_set_string(value, element->bank_filename);
		g_mutex_unlock(&element->bank_lock);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


static void finalize(GObject *object)
{
	GSTLALStringTriggergen *element = GSTLAL_STRING_TRIGGERGEN(object);
	g_mutex_clear(&element->bank_lock);
	free_bankfile(element);
	g_free(element->instrument);
	element->instrument = NULL;
	g_free(element->channel_name);
	element->channel_name = NULL;
	gst_audioadapter_clear(element->adapter);
	g_object_unref(element->adapter);
	G_OBJECT_CLASS(gstlal_string_triggergen_parent_class)->finalize(object);
}


/*
 * class_init()
 */


static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SINK_NAME,
	GST_PAD_SINK,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		GST_AUDIO_CAPS_MAKE(GST_AUDIO_NE(F32)) ", " \
		"layout = (string) interleaved, " \
		"channel-mask = (bitmask) 0"
	)
);


static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SRC_NAME,
	GST_PAD_SRC,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS("application/x-lal-snglburst")
);


static void gstlal_string_triggergen_class_init(GSTLALStringTriggergenClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(klass);

	gst_element_class_set_details_simple(
		element_class,
		"Trigger generator for cosmic string search",
		"Filter/Audio",
		"Find trigger from snr time series",
		"Daichi Tsuna <daichi.tsuna@ligo.org>"
	);

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	transform_class->get_unit_size = GST_DEBUG_FUNCPTR(get_unit_size);
	transform_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
	transform_class->transform = GST_DEBUG_FUNCPTR(transform);
	transform_class->transform_caps = GST_DEBUG_FUNCPTR(transform_caps);
	transform_class->start = GST_DEBUG_FUNCPTR(start);
	transform_class->prepare_output_buffer = GST_DEBUG_FUNCPTR(prepare_output_buffer);

	g_object_class_install_property(
		gobject_class,
		ARG_THRES,
		g_param_spec_float(
			"threshold",
			"threshold",
			"SNR threshold.",
			0, G_MAXFLOAT, DEFAULT_THRES,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_BANK_FILENAME,
		g_param_spec_string(
			"bank-filename",
			"Bank file name",
			"Path to XML file used to generate the template bank.",
			NULL,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
			)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_CLUSTER,
		g_param_spec_float(
			"cluster",
			"cluster",
			"Time window in seconds of which events are clustered. Two events with interval smaller than this value will be integrated into a single event.",
			0, G_MAXFLOAT, DEFAULT_CLUSTER,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
}

/*
 * init()
 */


static void gstlal_string_triggergen_init(GSTLALStringTriggergen *element)
{
	g_mutex_init(&element->bank_lock);
	element->adapter = g_object_new(GST_TYPE_AUDIOADAPTER, NULL);
	element->bank_filename = NULL;
	element->bank = NULL;
	element->instrument = NULL;
	element->channel_name = NULL;
	element->num_templates = 0;
	element->last_time = NULL;
	element->audio_info.bpf = 0;	/* impossible value */
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(element), TRUE);
}

/*
 * Cosmic string trigger generator and autocorrelation chisq plugin element
 */


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
#include <gst/base/gstadapter.h>
#include <gst/base/gstbasetransform.h>
#include <gsl/gsl_errno.h>


/*
 * stuff from LAL
 */


#include <lal/Date.h>
#include <lal/LIGOMetadataTables.h>


/*
 * stuff from gstlal
 */


#include <gstlal/gstlal.h>
#include <gstlal_string_triggergen.h>
#include <gstlal/gstaudioadapter.h>
#include <gstlal/gstlal_autocorrelation_chi2.h>
#include <gstlal/gstlal_debug.h>
#include <gstlal/gstlal_peakfinder.h>
#include <gstlal-burst/gstlal_snglburst.h>



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


#define DEFAULT_THRES 4.0
#define DEFAULT_CLUSTER 0.1


/*
 *======================================================
 *
 *          		Utilities
 *
 *======================================================
 */


static unsigned autocorrelation_channels(const gsl_matrix_float *autocorrelation_matrix)
{
	return autocorrelation_matrix->size1;
}


static unsigned autocorrelation_length(const gsl_matrix_float *autocorrelation_matrix)
{
	return autocorrelation_matrix->size2;
}


static guint64 get_available_samples(GSTLALStringTriggergen *element)
{
	guint size;

	g_object_get(element->adapter, "size", &size, NULL);

	return size;
}


static void free_bankfile(GSTLALStringTriggergen *element)
{
	g_free(element->bank_filename);
	element->bank_filename = NULL;
	gstlal_snglburst_array_free(element->bank);
	element->bank = NULL;
	element->num_templates = 0;
	free(element->last_time);
	element->last_time = NULL;
}


static int setup_bankfile_input(GSTLALStringTriggergen *element, char *bank_filename)
{
	free_bankfile(element);

	element->bank_filename = bank_filename;
	element->num_templates = gstlal_snglburst_array_from_file(element->bank_filename, &element->bank);
	if(element->num_templates < 0) {
		free_bankfile(element);
		return -1;
	}
	element->last_time = calloc(element->num_templates, sizeof(*element->last_time));
	if(!element->last_time) {
		free_bankfile(element);
		return -1;
	}

	return element->num_templates;
}


/*
 * compute autocorrelation norms --- the expectation value in noise.
 */


static gsl_vector_float *gstlal_autocorrelation_chi2_compute_norms_string(const gsl_matrix_float *autocorrelation_matrix, const gsl_matrix_int *autocorrelation_mask_matrix)
{
	gsl_vector_float *norm;
	unsigned channel;

	if(autocorrelation_mask_matrix && (autocorrelation_channels(autocorrelation_matrix) != autocorrelation_mask_matrix->size1 || autocorrelation_length(autocorrelation_matrix) != autocorrelation_mask_matrix->size2)) {
		/* FIXME:  report errors how? */
		/*GST_ELEMENT_ERROR(element, STREAM, FAILED, ("array size mismatch"), ("autocorrelation matrix (%dx%d) and mask matrix (%dx%d) do not have the same size", autocorrelation_channels(autocorrelation_matrix), autocorrelation_length(autocorrelation_matrix), autocorrelation_mask_matrix->size1, autocorrelation_mask_matrix->size2));*/
		return NULL;
	}

	norm = gsl_vector_float_alloc(autocorrelation_channels(autocorrelation_matrix));

	for(channel = 0; channel < autocorrelation_channels(autocorrelation_matrix); channel++) {
		gsl_vector_float_const_view row = gsl_matrix_float_const_row(autocorrelation_matrix, channel);
		gsl_vector_int_const_view mask = autocorrelation_mask_matrix ? gsl_matrix_int_const_row(autocorrelation_mask_matrix, channel) : (gsl_vector_int_const_view) {{0}};
		unsigned sample;
		float n = 0.0;
		
		for(sample = 0; sample < row.vector.size; sample++) {
			if(autocorrelation_mask_matrix && !gsl_vector_int_get(&mask.vector, sample))
				continue;
			n += 1.0 - pow(gsl_vector_float_get(&row.vector, sample), 2.0);
		}
		gsl_vector_float_set(norm, channel, n);
	}

	return norm;
}


/*
 *======================================================
 *
 *               Trigger Generator
 *
 *======================================================
 */


static GstFlowReturn trigger_generator(GSTLALStringTriggergen *element, GstBuffer *outbuf)
{
	float *snrdata;
	float *snrsample;
	SnglBurst *triggers = NULL;
	guint ntriggers = 0;
	guint64 offset;
	guint64 length;
	guint sample;
	gint channel;

	length = get_available_samples(element);
	if(length < autocorrelation_length(element->autocorrelation_matrix)) {
		/* FIXME:  PTS and duration are not necessarily correct.
		 * they're correct for now because we know how this element
		 * is used in the current pipeline, but in general this
		 * behaviour is not correct.  right now, the adapter can
		 * only not have enough data at the start of a stream, but
		 * for general streams the adapter could get flushed in mid
		 * stream and then we might need to worry about what the
		 * last reported buffer's end time was.  maybe.  maybe not
		 */
		GST_BUFFER_PTS(outbuf) = element->t0;
		GST_BUFFER_DURATION(outbuf) = 0;
		GST_BUFFER_OFFSET_END(outbuf) = GST_BUFFER_OFFSET(outbuf) + ntriggers;
		return GST_FLOW_OK;
	}

	g_mutex_lock(&element->bank_lock);
	snrsample = snrdata = g_malloc(length * element->num_templates * sizeof(*snrdata));

	/* copy samples */
	offset = gst_audioadapter_offset(element->adapter);
	gst_audioadapter_copy_samples(element->adapter, snrdata, length, NULL, NULL);

	/* compute the chisq norm if it doesn't exist */
	if(!element->autocorrelation_norm)
		element->autocorrelation_norm = gstlal_autocorrelation_chi2_compute_norms_string(element->autocorrelation_matrix, NULL);

	/* check that autocorrelation vector has odd number of samples */
	g_assert(autocorrelation_length(element->autocorrelation_matrix) & 1);

	/* find events.  earliest sample that can be a new trigger starts a
	 * little bit in from the start of the adapter because we are
	 * re-using data from the last iteration for \chi^2 calculation.
	 * the last sample that can be a new trigger is not at the end of
	 * the adapter's contents for the same reason */

	snrsample += (autocorrelation_length(element->autocorrelation_matrix) - 1) / 2 * element->num_templates;
	for(sample = (autocorrelation_length(element->autocorrelation_matrix) - 1) / 2; sample < length - (autocorrelation_length(element->autocorrelation_matrix) - 1) / 2; sample++){
		LIGOTimeGPS t;
		XLALINT8NSToGPS(&t, element->t0);
		XLALGPSAdd(&t, (double) (offset - element->offset0 + sample) / GST_AUDIO_INFO_RATE(&element->audio_info));

		for(channel = 0; channel < element->num_templates; channel++, snrsample++) {
			float snr = fabsf(*snrsample);
			if(snr >= element->threshold) {
				/*
				 * If there was a discontinuity (e.g. gap) that made this sample and the last sample above threshold larger than
				 * the clustering time window, pass the previous trigger and reset the SNR so as to start with a new trigger.
				 */
				if(element->bank[channel].snr > element->threshold && XLALGPSDiff(&t, &element->last_time[channel]) > element->cluster) {
					triggers = g_renew(SnglBurst, triggers, ntriggers + 1);
					triggers[ntriggers++] = element->bank[channel];
					element->bank[channel].snr = 0.0;
					element->bank[channel].chisq = 0.0;
					element->bank[channel].chisq_dof = 0.0;
				}

				/*
				 * If this is the first sample above threshold (i.e. snr of trigger is (re)set to 0), record the start time.
				 */
				if(element->bank[channel].snr < element->threshold)
					element->bank[channel].start_time = t;
				/*
				 * Keep track of last time above threshold and the duration.
				 * For duration add a sample of fuzz on both sides (like in lalapps_StringSearch).
				 */
				element->last_time[channel] = t;
				element->bank[channel].duration = XLALGPSDiff(&element->last_time[channel], &element->bank[channel].start_time) + (float) 2.0 / GST_AUDIO_INFO_RATE(&element->audio_info);
				if(snr > element->bank[channel].snr) {
					/*
					 * Higher SNR than the "current winner". Update.
					 */
					const float *autocorrelation = (const float *) gsl_matrix_float_const_ptr(element->autocorrelation_matrix, channel, 0);
					const float *autocorrelation_end = autocorrelation + autocorrelation_length(element->autocorrelation_matrix);
					float *snrseries = snrsample - (autocorrelation_length(element->autocorrelation_matrix) - 1) / 2 * element->num_templates;
					float chisq;

					element->bank[channel].snr = snr;
					element->bank[channel].peak_time = t;

					/* calculate chisq */
					for(chisq = 0.0; autocorrelation < autocorrelation_end; autocorrelation++, snrseries += element->num_templates)
						chisq += pow(*autocorrelation * *snrsample - *snrseries, 2.0);
					element->bank[channel].chisq = chisq / gsl_vector_float_get(element->autocorrelation_norm, channel);
					element->bank[channel].chisq_dof = 1.0;
				}
			} else if(element->bank[channel].snr != 0. && XLALGPSDiff(&t, &element->last_time[channel]) > element->cluster) {
				/*
				 * Trigger is ready to be passed.
				 * Push trigger to buffer, and reset it.
				 */
				triggers = g_renew(SnglBurst, triggers, ntriggers + 1);
				triggers[ntriggers++] = element->bank[channel];
				element->bank[channel].snr = 0.0;
				element->bank[channel].chisq = 0.0;
				element->bank[channel].chisq_dof = 0.0;
			}
		}
	}
	g_mutex_unlock(&element->bank_lock);

	g_free(snrdata);
	gst_audioadapter_flush_samples(element->adapter, length - (autocorrelation_length(element->autocorrelation_matrix) - 1));

	if(ntriggers)
		gst_buffer_replace_all_memory(outbuf, gst_memory_new_wrapped(GST_MEMORY_FLAG_PHYSICALLY_CONTIGUOUS, triggers, ntriggers * sizeof(*triggers), 0, ntriggers * sizeof(*triggers), triggers, g_free));
	else
		gst_buffer_remove_all_memory(outbuf);

	/*
	 * obtain PTS and DURATION of output buffer.
	 */

	GST_BUFFER_PTS(outbuf) = element->t0 + gst_util_uint64_scale_int_round(offset + (autocorrelation_length(element->autocorrelation_matrix) - 1) / 2 - element->offset0, GST_SECOND, GST_AUDIO_INFO_RATE(&element->audio_info));
	for(channel = 0; channel < element->num_templates; channel++)
		if(element->bank[channel].snr > 0.0 && (GstClockTime) XLALGPSToINT8NS(&element->bank[channel].peak_time) < GST_BUFFER_PTS(outbuf))
			GST_BUFFER_PTS(outbuf) = XLALGPSToINT8NS(&element->bank[channel].peak_time);
	GST_BUFFER_DURATION(outbuf) = element->t0 + gst_util_uint64_scale_int_round(offset + length - (autocorrelation_length(element->autocorrelation_matrix) - 1) / 2 - element->offset0, GST_SECOND, GST_AUDIO_INFO_RATE(&element->audio_info)) - GST_BUFFER_PTS(outbuf);

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
		GstCaps *src_caps = gst_pad_get_pad_template_caps(GST_BASE_TRANSFORM_SRC_PAD(trans));
		if(gst_caps_is_strictly_equal(caps, src_caps)) {
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

	if(success) {
		if(GST_AUDIO_INFO_CHANNELS(&element->audio_info) != element->num_templates) {
			GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("number of channels %d is not equal to number of templates %d", GST_AUDIO_INFO_CHANNELS(&element->audio_info), element->num_templates));
			success = FALSE;
		}
		g_object_set(element->adapter, "unit-size", GST_AUDIO_INFO_BPF(&element->audio_info), NULL);
	}


	/*
	 * done
	 */

	return success;
}


/*
 * start()
 */

static gboolean start(GstBaseTransform *trans)
{
	GSTLALStringTriggergen *element = GSTLAL_STRING_TRIGGERGEN(trans);
	gint i;
	gboolean success = TRUE;

	if(!element->bank) {
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("no template bank"));
		success = FALSE;
	} else {
		for(i=0; i < element->num_templates; i++) {
			/*
			 * initialize data in template. the snr is 0'ed so that
			 * when the templates are used to initialize the last event
			 * info that field is set properly.
			 */

			XLALINT8NSToGPS(&element->bank[i].peak_time, 0);
			XLALINT8NSToGPS(&element->bank[i].start_time, 0);
			element->bank[i].duration = 0;
			element->bank[i].snr = 0;

			/*
			 * Initialize the chisq and chisq_dof, too.
			 * We follow the definition of the previous string search pipeline,
			 * The actual chi^2 is then chisq/chisq_dof. We can come
			 * back to the definition later if we have to.
			 */
			element->bank[i].chisq = 0;
			element->bank[i].chisq_dof = 0;

			/* initialize the last time array, too */
			XLALINT8NSToGPS(&element->last_time[i], 0);
		}
		element->t0 = GST_CLOCK_TIME_NONE;
		element->offset0 = GST_BUFFER_OFFSET_NONE;
		element->next_in_offset = GST_BUFFER_OFFSET_NONE;
		element->next_out_offset = GST_BUFFER_OFFSET_NONE;
		element->need_discont = TRUE;
	}

	return success;
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
	guint64 length;
	GstFlowReturn result;

	g_assert(GST_BUFFER_PTS_IS_VALID(inbuf));
	g_assert(GST_BUFFER_DURATION_IS_VALID(inbuf));
	g_assert(GST_BUFFER_OFFSET_IS_VALID(inbuf));
	g_assert(GST_BUFFER_OFFSET_END_IS_VALID(inbuf));
	
	if(GST_BUFFER_IS_DISCONT(inbuf) || GST_BUFFER_OFFSET(inbuf) != element->next_in_offset || !GST_CLOCK_TIME_IS_VALID(element->t0)) {
		gst_audioadapter_clear(element->adapter);
		element->t0 = GST_BUFFER_PTS(inbuf);
		element->offset0 = GST_BUFFER_OFFSET(inbuf);
		element->next_out_offset = GST_BUFFER_OFFSET(inbuf);
	} else if(!gst_audioadapter_is_empty(element->adapter))
		g_assert_cmpuint(GST_BUFFER_PTS(inbuf), ==, gst_audioadapter_expected_timestamp(element->adapter));
	element->next_in_offset = GST_BUFFER_OFFSET_END(inbuf);

	/*
	 * gap logic
	 */

	gst_buffer_ref(inbuf);
	gst_audioadapter_push(element->adapter, inbuf);
	if (!GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP)) {
		/* not gaps */
		result = trigger_generator(element, outbuf);
	} else {
		/* gaps */
		length = get_available_samples(element);
		element->next_out_offset += length;
		gst_audioadapter_flush_samples(element->adapter, length);
		GST_BUFFER_PTS(outbuf) = element->t0 + gst_util_uint64_scale_int_round(element->next_out_offset - element->offset0, GST_SECOND, GST_AUDIO_INFO_RATE(&element->audio_info));
		GST_BUFFER_DURATION(outbuf) = gst_util_uint64_scale_int_round(length, GST_SECOND, GST_AUDIO_INFO_RATE(&element->audio_info));
		/* we get no triggers, so outbuf offset is unchanged */
		GST_BUFFER_OFFSET_END(outbuf) = GST_BUFFER_OFFSET(outbuf);
		GST_BUFFER_FLAG_SET(outbuf, GST_BUFFER_FLAG_GAP);
		result = GST_FLOW_OK;
	}

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
	ARG_BANK_FILENAME,
	ARG_AUTOCORRELATION_MATRIX
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

	case ARG_AUTOCORRELATION_MATRIX:
		g_mutex_lock(&element->bank_lock);
		if(element->autocorrelation_matrix)
			gsl_matrix_float_free(element->autocorrelation_matrix);
		element->autocorrelation_matrix = gstlal_gsl_matrix_float_from_g_value_array(g_value_get_boxed(value));

		/*
		 * induce norms to be recomputed
		 */

		if(element->autocorrelation_norm) {
			gsl_vector_float_free(element->autocorrelation_norm);
			element->autocorrelation_norm = NULL;
		}

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
	
	case ARG_AUTOCORRELATION_MATRIX:
		g_mutex_lock(&element->bank_lock);
		if(element->autocorrelation_matrix)
			g_value_take_boxed(value, gstlal_g_value_array_from_gsl_matrix_float(element->autocorrelation_matrix));
		else {
			GST_WARNING_OBJECT(element, "no autocorrelation matrix");
			/* FIXME deprecated.. */
			g_value_take_boxed(value, g_value_array_new(0)); 
		}
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
	gst_audioadapter_clear(element->adapter);
	g_object_unref(element->adapter);

	g_mutex_clear(&element->bank_lock);
	if(element->autocorrelation_matrix) {
		gsl_matrix_float_free(element->autocorrelation_matrix);
		element->autocorrelation_matrix = NULL;
	}
	if(element->autocorrelation_norm) {
		gsl_vector_float_free(element->autocorrelation_norm);
		element->autocorrelation_norm = NULL;
	}
	free_bankfile(element);

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
	g_object_class_install_property(
		gobject_class,
		ARG_AUTOCORRELATION_MATRIX,
		g_param_spec_value_array(
			"autocorrelation-matrix",
			"Autocorrelation Matrix",
			"Array of autocorrelation vectors.  Number of vectors (rows) in matrix sets number of channels.  All vectors must have the same length.",
			g_param_spec_value_array(
				"autocorrelation",
				"Autocorrelation",
				"Array of autocorrelation samples.",
				/* FIXME:  should be complex */
				g_param_spec_float(
					"sample",
					"Sample",
					"Autocorrelation sample",
					-G_MAXFLOAT, G_MAXFLOAT, 0.0,
					G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
				),
				G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
			),
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);
}

/*
 * init()
 */


static void gstlal_string_triggergen_init(GSTLALStringTriggergen *element)
{
	element->bank_filename = NULL;
	element->autocorrelation_matrix = NULL;
	element->autocorrelation_norm = NULL;
	element->audio_info.bpf = 0;	/* impossible value */
	element->adapter = g_object_new(GST_TYPE_AUDIOADAPTER, NULL);
	g_mutex_init(&element->bank_lock);
	element->bank = NULL;
	element->num_templates = 0;
	element->last_time = NULL;
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(element), TRUE);
}

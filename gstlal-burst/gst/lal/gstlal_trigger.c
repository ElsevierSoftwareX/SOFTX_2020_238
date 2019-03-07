/*
 * An trigger trigger and autocorrelation chisq (itac) element
 *
 * Copyright (C) 2011  Duncan Meacher
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
 * ========================================================================
 *
 *                                  Preamble
 *
 * ========================================================================
 */


/*
 * stuff from C library, glib/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstadapter.h>
#include <gst/controller/controller.h>
#include <math.h>
#include <string.h>

/*
 * our own stuff
 */

#include <gstlal/gstlal.h>
#include <gstlal/gstlal_debug.h>
#include <gstlal_trigger.h>
#include <gstlal/gstlal_peakfinder.h>
#include <gstlal/gstaudioadapter.h>
#include <gstlal/gstlal_tags.h>
#include <gstlal/gstlal_autocorrelation_chi2.h>
#include <gstlal-burst/gstlal_sngltrigger.h>


/*
 * ============================================================================
 *
 *                           GStreamer Boiler Plate
 *
 * ============================================================================
 */


#define GST_CAT_DEFAULT gstlal_trigger_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


G_DEFINE_TYPE_WITH_CODE(
	GSTLALTrigger,
	gstlal_trigger,
	GST_TYPE_ELEMENT,
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "lal_trigger", 0, "lal_trigger debug category")
);


/*
 * ============================================================================
 *
 *                                 Parameters
 *
 * ============================================================================
 */


#define DEFAULT_SNR_THRESH 5.5
#define DEFAULT_MAX_SNR FALSE


/* FIXME the function placements should be moved around to avoid putting this static prototype here */
static GstFlowReturn process(GSTLALTrigger *element);


/*
 * ========================================================================
 *
 *                             Utility Functions
 *
 * ========================================================================
 */


static unsigned autocorrelation_channels(const GSTLALTrigger *element)
{
	return gstlal_autocorrelation_chi2_autocorrelation_channels(element->autocorrelation_matrix);
}


static unsigned autocorrelation_length(const GSTLALTrigger *element)
{
	return gstlal_autocorrelation_chi2_autocorrelation_length(element->autocorrelation_matrix);
}


static guint64 output_num_samps(GSTLALTrigger *element)
{
	return (guint64) element->n;
}


static guint64 output_num_bytes(GSTLALTrigger *element)
{
	return (guint64) output_num_samps(element) * element->adapter->unit_size;
}


static int reset_time_and_offset(GSTLALTrigger *element)
{
	element->next_output_offset = 0;
	element->next_output_timestamp = GST_CLOCK_TIME_NONE;
	return 0;
}


static guint gst_audioadapter_available_samples(GstAudioAdapter *adapter)
{
	guint size;
	g_object_get(adapter, "size", &size, NULL);
	return size;
}


static void update_peak_info_from_autocorrelation_properties(GSTLALTrigger *element)
{
	if (element->maxdata && element->autocorrelation_matrix) {
		element->maxdata->pad = autocorrelation_length(element) / 2;
		if (element->snr_mat)
			free(element->snr_mat);
		element->snr_mat = calloc(autocorrelation_channels(element) * autocorrelation_length(element), element->maxdata->unit);
		/*
		 * Each row is one sample point of the snr time series with N
		 * columns for N channels. Assumes proper packing to go from real to complex.
		 * FIXME assumes single precision
		 */
		element->snr_matrix_view = gsl_matrix_complex_float_view_array((float *) element->snr_mat, autocorrelation_length(element), autocorrelation_channels(element));
	}
}


/*
 * ============================================================================
 *
 *                                 Events
 *
 * ============================================================================
 */


static gboolean taglist_extract_string(GstObject *object, GstTagList *taglist, const char *tagname, gchar **dest)
{
	if(!gst_tag_list_get_string(taglist, tagname, dest)) {
		GST_WARNING_OBJECT(object, "unable to parse \"%s\" from %" GST_PTR_FORMAT, tagname, taglist);
		return FALSE;
	}
	return TRUE;
}


static gboolean setcaps(GSTLALTrigger *element, GstPad *pad, GstCaps *caps)
{
	guint width = 0;
	int tmp_channels = 0;
	GstStructure *str = gst_caps_get_structure(caps, 0);
	const gchar *format = gst_structure_get_string(str, "format");
	gboolean success = TRUE;
	gst_structure_get_int(str, "rate", &(element->rate));
	/* Using temp to type cast from int to unsigned int (guint) */
	gst_structure_get_int(str, "channels", &(tmp_channels));
	element->channels = (guint)tmp_channels;

	/*
	 * update the element metadata
	 */


	if(!strcmp(format, GST_AUDIO_NE(Z64)))
		width = 64;
	else if(!strcmp(format, GST_AUDIO_NE(Z128)))
		width = 128;
	else
		GST_ERROR_OBJECT(element, "unsupported format %s", format);

	g_object_set(element->adapter, "unit-size", width / 8 * element->channels, NULL);
	if (width == 128) {
		element->peak_type = GSTLAL_PEAK_DOUBLE_COMPLEX;
		element->chi2 = calloc(element->channels, sizeof(double));
		}
	if (width == 64) {
		element->peak_type = GSTLAL_PEAK_COMPLEX;
		element->chi2 = calloc(element->channels, sizeof(float));
		}
	if (element->maxdata)
		gstlal_peak_state_free(element->maxdata);
	element->maxdata = gstlal_peak_state_new(element->channels, element->peak_type);
	/* This should be called any time the autocorrelation property is updated */
	update_peak_info_from_autocorrelation_properties(element);

	/*
	 * done
	 */


	return success;
}


static gboolean sink_event(GstPad *pad, GstObject *parent, GstEvent *event)
{
	GSTLALTrigger *trigger = GSTLAL_TRIGGER(parent);
	gboolean res = TRUE;

	GST_DEBUG_OBJECT(pad, "Got %s event on sink pad", GST_EVENT_TYPE_NAME (event));

	switch (GST_EVENT_TYPE (event))
	{
		case GST_EVENT_CAPS:
		{
			GstCaps *caps;
			gst_event_parse_caps(event, &caps);
			res = setcaps(trigger, pad, caps);
			break;
		}
		case GST_EVENT_TAG:
		{
			GstTagList *taglist;
			gchar *instrument, *channel_name;
			gst_event_parse_tag(event, &taglist);
			res = taglist_extract_string(GST_OBJECT(pad), taglist, GSTLAL_TAG_INSTRUMENT, &instrument);
			res &= taglist_extract_string(GST_OBJECT(pad), taglist, GSTLAL_TAG_CHANNEL_NAME, &channel_name);
			if(res)
			{
				GST_DEBUG_OBJECT(pad, "found tags \"%s\"=\"%s\", \"%s\"=\"%s\"", GSTLAL_TAG_INSTRUMENT, instrument, GSTLAL_TAG_CHANNEL_NAME, channel_name);
				g_free(trigger->instrument);
				trigger->instrument = instrument;
				g_free(trigger->channel_name);
				trigger->channel_name = channel_name;
				g_mutex_lock(&trigger->bank_lock);
				g_mutex_unlock(&trigger->bank_lock);
			}
			break;
		}
		case GST_EVENT_EOS:
		{
			trigger->EOS = TRUE;
			/* ignore failures */
			process(trigger);
			break;
		}
		default:
			break;
	}

	if(!res)
		gst_event_unref(event);
	else
		res = gst_pad_event_default(pad, parent, event);

	return res;
}


/*
 * ============================================================================
 *
 *                                 Properties
 *
 * ============================================================================
 */


enum property {
	ARG_N = 1,
	ARG_SNR_THRESH,
	ARG_AUTOCORRELATION_MATRIX,
	ARG_AUTOCORRELATION_MASK,
	ARG_MAX_SNR
};


static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec)
{
	GSTLALTrigger *element = GSTLAL_TRIGGER(object);

	GST_OBJECT_LOCK(element);


	switch(id) {
	case ARG_N:
		element->n = g_value_get_uint(value);
		break;

	case ARG_SNR_THRESH:
		element->snr_thresh = g_value_get_double(value);
		break;

	case ARG_AUTOCORRELATION_MATRIX: {
		g_mutex_lock(&element->bank_lock);

		if(element->autocorrelation_matrix)
			gsl_matrix_complex_free(element->autocorrelation_matrix);

		element->autocorrelation_matrix = gstlal_gsl_matrix_complex_from_g_value_array(g_value_get_boxed(value));

		/* This should be called any time caps change too */
		update_peak_info_from_autocorrelation_properties(element);

		/*
		 * induce norms to be recomputed
		 */

		if(element->autocorrelation_norm) {
			gsl_vector_free(element->autocorrelation_norm);
			element->autocorrelation_norm = NULL;
		}

		g_mutex_unlock(&element->bank_lock);
		break;
	}

	case ARG_AUTOCORRELATION_MASK: {
		g_mutex_lock(&element->bank_lock);

		if(element->autocorrelation_mask)
			gsl_matrix_int_free(element->autocorrelation_mask);

		element->autocorrelation_mask = gstlal_gsl_matrix_int_from_g_value_array(g_value_get_boxed(value));

		g_mutex_unlock(&element->bank_lock);
		break;
	}

	case ARG_MAX_SNR:
		element->max_snr = g_value_get_boolean(value);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}


	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *pspec)
{
	GSTLALTrigger *element = GSTLAL_TRIGGER(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_N:
		g_value_set_uint(value, element->n);
		break;

	case ARG_SNR_THRESH:
		g_value_set_double(value, element->snr_thresh);
		break;

	case ARG_AUTOCORRELATION_MATRIX:
		g_mutex_lock(&element->bank_lock);
		if(element->autocorrelation_matrix)
			g_value_take_boxed(value, gstlal_g_value_array_from_gsl_matrix_complex(element->autocorrelation_matrix));
		else {
			GST_WARNING_OBJECT(element, "no autocorrelation matrix");
			g_value_take_boxed(value, g_value_array_new(0));
			}
		g_mutex_unlock(&element->bank_lock);
		break;

	case ARG_AUTOCORRELATION_MASK:
		g_mutex_lock(&element->bank_lock);
		if(element->autocorrelation_mask)
			g_value_take_boxed(value, gstlal_g_value_array_from_gsl_matrix_int(element->autocorrelation_mask));
		else {
			GST_WARNING_OBJECT(element, "no autocorrelation mask");
			g_value_take_boxed(value, g_value_array_new(0));
			}
		g_mutex_unlock(&element->bank_lock);
		break;

	case ARG_MAX_SNR:
		g_value_set_boolean(value, element->max_snr);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * ============================================================================
 *
 *                                  Sink Pad
 *
 * ============================================================================
 */


/*
 * chain()
 */

static void update_state(GSTLALTrigger *element, GstBuffer *srcbuf)
{
	element->next_output_offset = GST_BUFFER_OFFSET_END(srcbuf);
	element->next_output_timestamp = GST_BUFFER_PTS(srcbuf) - element->difftime;
	element->next_output_timestamp += GST_BUFFER_DURATION(srcbuf);
}

static GstFlowReturn push_buffer(GSTLALTrigger *element, GstBuffer *srcbuf)
{
	GstFlowReturn result = GST_FLOW_OK;
	GST_DEBUG_OBJECT(element, "pushing %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(srcbuf));
	GST_BUFFER_DTS(srcbuf) = GST_BUFFER_PTS(srcbuf);
	result =  gst_pad_push(element->srcpad, srcbuf);
	return result;
}

static GstFlowReturn push_gap(GSTLALTrigger *element, guint samps)
{
	GstBuffer *srcbuf = NULL;
	GstFlowReturn result = GST_FLOW_OK;
	/* Clearing the max data structure causes the resulting buffer to be a GAP */
	gstlal_peak_state_clear(element->maxdata);
	/* create the output buffer */
	srcbuf = gstlal_sngltrigger_new_buffer_from_peak(element->maxdata, element->channel_name, element->srcpad, element->next_output_offset, samps, element->next_output_timestamp, element->rate, NULL, NULL, element->difftime, element->max_snr);
	/* set the time stamp and offset state */
	update_state(element, srcbuf);
	/* push the result */
	result = push_buffer(element, srcbuf);
	return result;
}

static GstFlowReturn push_nongap(GSTLALTrigger *element, guint copysamps, guint outsamps)
{
	GstBuffer *srcbuf = NULL;
	GstFlowReturn result = GST_FLOW_OK;
	gsl_error_handler_t *old_gsl_error_handler;
	union {
		float complex * as_complex;
		double complex * as_double_complex;
		void * as_void;
		} dataptr;

	/* make sure the snr threshold is up-to-date */
	element->maxdata->thresh = element->snr_thresh;
	/* call the peak finding library on a buffer from the adapter if no events are found the result will be a GAP */
	gst_audioadapter_copy_samples(element->adapter, element->data, copysamps, NULL, NULL);

	/* turn off error handling for interpolation */
	old_gsl_error_handler=gsl_set_error_handler_off();

	/* put the data pointer one pad length in */
	if (element->peak_type == GSTLAL_PEAK_COMPLEX) {
		dataptr.as_complex = ((float complex *) element->data) + element->maxdata->pad * element->maxdata->channels;
		/* Find the peak */
		gstlal_float_complex_peak_over_window(element->maxdata, dataptr.as_complex, outsamps);
		}
	else if (element->peak_type == GSTLAL_PEAK_DOUBLE_COMPLEX) {
		dataptr.as_double_complex = ((double complex *) element->data) + element->maxdata->pad * element->maxdata->channels;
		/* Find the peak */
		gstlal_double_complex_peak_over_window(element->maxdata, dataptr.as_double_complex, outsamps);
		}
	else
		g_assert_not_reached();

	/* turn error handling back on */
	gsl_set_error_handler(old_gsl_error_handler);

	/* compute \chi^2 values if we can */
	if (element->autocorrelation_matrix) {
		/* compute the chisq norm if it doesn't exist */
		if (!element->autocorrelation_norm)
			element->autocorrelation_norm = gstlal_autocorrelation_chi2_compute_norms(element->autocorrelation_matrix, NULL);

		g_assert(autocorrelation_length(element) & 1);	/* must be odd */

		if (element->peak_type == GSTLAL_PEAK_DOUBLE_COMPLEX) {
			/* extract data around peak for chisq calculation */
			gstlal_double_complex_series_around_peak(element->maxdata, dataptr.as_double_complex, (double complex *) element->snr_mat, element->maxdata->pad);
			gstlal_autocorrelation_chi2((double *) element->chi2, (double complex *) element->snr_mat, autocorrelation_length(element), -((int) autocorrelation_length(element)) / 2, 0.0, element->autocorrelation_matrix, element->autocorrelation_mask, element->autocorrelation_norm);
			/* create the output buffer */
			srcbuf = gstlal_sngltrigger_new_buffer_from_peak(element->maxdata, element->channel_name, element->srcpad, element->next_output_offset, outsamps, element->next_output_timestamp, element->rate, element->chi2, NULL, element->difftime, element->max_snr);
			}
		else if (element->peak_type == GSTLAL_PEAK_COMPLEX) {
			/* extract data around peak for chisq calculation */
			gstlal_float_complex_series_around_peak(element->maxdata, dataptr.as_complex, (float complex *) element->snr_mat, element->maxdata->pad);
			gstlal_autocorrelation_chi2_float((float *) element->chi2, (float complex *) element->snr_mat, autocorrelation_length(element), -((int) autocorrelation_length(element)) / 2, 0.0, element->autocorrelation_matrix, element->autocorrelation_mask, element->autocorrelation_norm);
			/* create the output buffer */
			/* FIXME snr snippets not supported for double precision yet */
			srcbuf = gstlal_sngltrigger_new_buffer_from_peak(element->maxdata, element->channel_name, element->srcpad, element->next_output_offset, outsamps, element->next_output_timestamp, element->rate, element->chi2, &(element->snr_matrix_view), element->difftime, element->max_snr);
			}
		else
			g_assert_not_reached();
		}
	else
		srcbuf = gstlal_sngltrigger_new_buffer_from_peak(element->maxdata, element->channel_name, element->srcpad, element->next_output_offset, outsamps, element->next_output_timestamp, element->rate, NULL, NULL, element->difftime, element->max_snr);
		
	/* set the time stamp and offset state */
	update_state(element, srcbuf);
	/* push the result */
	result = push_buffer(element, srcbuf);
	return result;
}

static GstFlowReturn process(GSTLALTrigger *element)
{
	guint outsamps, gapsamps, nongapsamps, copysamps;
	GstFlowReturn result = GST_FLOW_OK;

	while( (element->EOS && gst_audioadapter_available_samples(element->adapter)) || gst_audioadapter_available_samples(element->adapter) > (element->n + 2 * element->maxdata->pad)) {

		/* See if the output is a gap or not */
		nongapsamps = gst_audioadapter_head_nongap_length(element->adapter);
		gapsamps = gst_audioadapter_head_gap_length(element->adapter);

		/* First check if the samples are gap */
		if (gapsamps > 0) {
			element->last_gap = TRUE;
			outsamps = gapsamps > element->n ? element->n : gapsamps;
			result = push_gap(element, outsamps);
			/* knock off the first buffers worth of bytes since we don't need them any more */
			gst_audioadapter_flush_samples(element->adapter, outsamps);
			}
		/* The check to see if we have enough nongap samples to compute an output, else it is a gap too */
		else if (nongapsamps <= 2 * element->maxdata->pad) {
			element->last_gap = TRUE;
			outsamps = nongapsamps;
			result = push_gap(element, outsamps);
			/* knock off the first buffers worth of bytes since we don't need them any more */
			gst_audioadapter_flush_samples(element->adapter, outsamps);
			}
		/* Else we have enough nongap samples to actually compute an output, but the first and last buffer might still be a gap */
		else {
			/* Check to see if we just came off a gap, if so then we need to push a gap for the startup transient if padding is requested */
			if (element->last_gap) {
				element->last_gap = FALSE;
				if (element->maxdata->pad > 0)
					result = push_gap(element, element->maxdata->pad);
				}
			/* if we have enough nongap samples then our output is length n, otherwise we have to knock the padding off of what is available */
			copysamps = (nongapsamps > (element->n + 2 * element->maxdata->pad)) ? (element->n + 2 * element->maxdata->pad) : nongapsamps;
			outsamps = (copysamps == nongapsamps) ? (copysamps - 2 * element->maxdata->pad) : element->n;
			result = push_nongap(element, copysamps, outsamps);
			/* knock off the first buffers worth of bytes since we don't need them any more */
			gst_audioadapter_flush_samples(element->adapter, outsamps);

			/* We are on another gap boundary so push the end transient as a gap */
			if (copysamps == nongapsamps) {
				element->last_gap = FALSE;
				if (element->maxdata->pad > 0) {
					result = push_gap(element, element->maxdata->pad);
					gst_audioadapter_flush_samples(element->adapter, 2 * element->maxdata->pad);
					}
				}
			}
		}

	return result;
}

static GstFlowReturn chain(GstPad *pad, GstObject *parent, GstBuffer *sinkbuf)
{
	GSTLALTrigger *element = GSTLAL_TRIGGER(parent);
	GstFlowReturn result = GST_FLOW_OK;
	guint64 maxsize;

	/* do this before accessing any element properties */
	gst_object_sync_values(parent, GST_BUFFER_PTS(sinkbuf));

	/* The max size to copy from an adapter is the typical output size plus the padding */
	maxsize = output_num_bytes(element) + element->adapter->unit_size * element->maxdata->pad * 2;

	/* if we haven't allocated storage do it now, we should never try to copy from an adapter with a larger buffer than this */
	if (!element->data)
		element->data = malloc(maxsize);

	/*
	 * check validity of timestamp, offsets, tags, bank array
	 */

	if(!GST_BUFFER_PTS_IS_VALID(sinkbuf) || !GST_BUFFER_DURATION_IS_VALID(sinkbuf) || !GST_BUFFER_OFFSET_IS_VALID(sinkbuf) || !GST_BUFFER_OFFSET_END_IS_VALID(sinkbuf)) {
		gst_buffer_unref(sinkbuf);
		GST_ERROR_OBJECT(element, "error in input stream: buffer has invalid timestamp and/or offset");
		result = GST_FLOW_ERROR;
		goto done;
	}

	if(!element->instrument || !element->channel_name) {
		GST_ELEMENT_ERROR(element, STREAM, FAILED, ("missing or invalid tags"), ("instrument and/or channel name not known (stream's tags must provide this information)"));
		result = GST_FLOW_ERROR;
		goto done;
	}

	//if (!element->bankarray) {
	//	GST_ELEMENT_ERROR(element, STREAM, FAILED, ("missing bank file"), ("must have a valid template bank to create events"));
	//	result = GST_FLOW_ERROR;
	//	goto done;
	//}

	/* FIXME if we were more careful we wouldn't lose so much data around disconts */
	if (GST_BUFFER_FLAG_IS_SET(sinkbuf, GST_BUFFER_FLAG_DISCONT)) {
		reset_time_and_offset(element);
		gst_audioadapter_clear(element->adapter);
	}


	/* if we don't have a valid first timestamp yet take this one */
	if (element->next_output_timestamp == GST_CLOCK_TIME_NONE) {
		element->next_output_timestamp = GST_BUFFER_PTS(sinkbuf);// + output_duration(element);
	}

	/* put the incoming buffer into an adapter, handles gaps */
	gst_audioadapter_push(element->adapter, sinkbuf);

	/* process the data we have */
	process(element);

done:
	return result;
}


/*
 * ============================================================================
 *
 *                                Type Support
 *
 * ============================================================================
 */


/*
 * Instance finalize function.  See ???
 */


static void finalize(GObject *object)
{

	GSTLALTrigger *element = GSTLAL_TRIGGER(object);
	g_mutex_clear(&element->bank_lock);
	gst_object_unref(element->sinkpad);
	element->sinkpad = NULL;
	gst_object_unref(element->srcpad);
	element->srcpad = NULL;
	gst_audioadapter_clear(element->adapter);
	g_object_unref(element->adapter);
	if (element->instrument) {
		free(element->instrument);
		element->instrument = NULL;
		}
	if (element->channel_name) {
		free(element->channel_name);
		element->channel_name = NULL;
		}
	if (element->maxdata) {
		gstlal_peak_state_free(element->maxdata);
		element->maxdata = NULL;
		}
	if (element->data) {
		free(element->data);
		element->data = NULL;
		}
	if(element->snr_mat) {
		free(element->snr_mat);
	}
	if(element->autocorrelation_matrix) {
		gsl_matrix_complex_free(element->autocorrelation_matrix);
		element->autocorrelation_matrix = NULL;
	}
	if(element->autocorrelation_mask) {
		gsl_matrix_int_free(element->autocorrelation_mask);
		element->autocorrelation_mask = NULL;
	}
	if(element->autocorrelation_norm) {
		gsl_vector_free(element->autocorrelation_norm);
		element->autocorrelation_norm = NULL;
	}
	if(element->chi2) {
		free(element->chi2);
	}
	G_OBJECT_CLASS(gstlal_trigger_parent_class)->finalize(object);
}


/*
 * Class init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GClassInitFunc
 */


#define CAPS \
	"audio/x-raw, " \
	"format = (string) { " GST_AUDIO_NE(Z64) ", " GST_AUDIO_NE(Z128) " }, " \
	"rate = " GST_AUDIO_RATE_RANGE ", " \
	"channels = " GST_AUDIO_CHANNELS_RANGE ", " \
	"layout = (string) interleaved, " \
	"channel-mask = (bitmask) 0"


static void gstlal_trigger_class_init(GSTLALTriggerClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);

	gst_element_class_set_details_simple(
		element_class,
		"Generic trigger generator",
		"Filter",
		"Find triggers in snr streams",
		"Duncan Meacher <duncan.meacher@ligo.org>"
	);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"sink",
			GST_PAD_SINK,
			GST_PAD_ALWAYS,
			gst_caps_from_string(CAPS)
		)
	);
	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"src",
			GST_PAD_SRC,
			GST_PAD_ALWAYS,
			gst_caps_from_string("application/gstlal-sngltrigger")
		)
	);

	g_object_class_install_property(
		gobject_class,
		ARG_N,
		g_param_spec_uint(
			"n",
			"n",
			"number of samples over which to identify triggers",
			0, G_MAXUINT, 0,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);

	g_object_class_install_property(
		gobject_class,
		ARG_SNR_THRESH,
		g_param_spec_double(
			"snr-thresh",
			"SNR Threshold",
			"SNR Threshold that determines a trigger.",
			0, G_MAXDOUBLE, DEFAULT_SNR_THRESH,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);
	
	g_object_class_install_property(
		gobject_class,
		ARG_AUTOCORRELATION_MATRIX,
		g_param_spec_value_array(
			"autocorrelation-matrix",
			"Autocorrelation Matrix",
			"Array of complex autocorrelation vectors.  Number of vectors (rows) in matrix sets number of channels.  All vectors must have the same length.",
			g_param_spec_value_array(
				"autocorrelation",
				"Autocorrelation",
				"Array of autocorrelation samples.",
				/* FIXME:  should be complex */
				g_param_spec_double(
					"sample",
					"Sample",
					"Autocorrelation sample",
					-G_MAXDOUBLE, G_MAXDOUBLE, 0.0,
					G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
				),
				G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
			),
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);

	g_object_class_install_property(
		gobject_class,
		ARG_AUTOCORRELATION_MASK,
		g_param_spec_value_array(
			"autocorrelation-mask",
			"Autocorrelation Mask Matrix",
			"Array of integer autocorrelation mask vectors.  Number of vectors (rows) in mask sets number of channels.  All vectors must have the same length. The mask values are either 0 or 1 and indicate whether to use the corresponding matrix entry in computing the autocorrelation chi-sq statistic.",
			g_param_spec_value_array(
				"autocorrelation-mask",
				"Autocorrelation-mask",
				"Array of autocorrelation mask values.",
				g_param_spec_int(
					"sample",
					"Sample",
					"Autocorrelation mask value",
					0, 1, 0,
					G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
				),
				G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
			),
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);

	g_object_class_install_property(
		gobject_class,
		ARG_MAX_SNR,
		g_param_spec_boolean(
			"max-snr",
			"Max SNR trigger in buffer",
			"Set flag to return single loudest trigger from buffer",
			DEFAULT_MAX_SNR,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);

}


/*
 * Instance init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GInstanceInitFunc
 */

static void gstlal_trigger_init(GSTLALTrigger *element)
{
	GstPad *pad;
	gst_element_create_all_pads(GST_ELEMENT(element));

	/* configure (and ref) sink pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "sink");
	gst_pad_set_event_function(pad, GST_DEBUG_FUNCPTR(sink_event));
	gst_pad_set_chain_function(pad, GST_DEBUG_FUNCPTR(chain));
	/* FIXME maybe these are necessary but they prevent buffers from leaving the element */
	//GST_PAD_SET_PROXY_CAPS(pad);
	//GST_PAD_SET_PROXY_ALLOCATION(pad);
	GST_PAD_SET_PROXY_SCHEDULING(pad);
	element->sinkpad = pad;
	gst_pad_use_fixed_caps(pad);

	/* retrieve (and ref) src pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "src");
	/* FIXME maybe these are necessary but they prevent buffers from leaving the element */
	//GST_PAD_SET_PROXY_CAPS(pad);
	//GST_PAD_SET_PROXY_ALLOCATION(pad);
	GST_PAD_SET_PROXY_SCHEDULING(pad);
	element->srcpad = pad;

	{
	GstCaps *caps = gst_caps_copy(gst_pad_get_pad_template_caps(pad));
	gst_pad_set_caps(pad, caps);
	gst_caps_unref(caps);
	}
	gst_pad_use_fixed_caps(pad);
	
	/* internal data */
	element->rate = 0;
	element->difftime = 0;
	element->snr_thresh = 0;
	reset_time_and_offset(element);
	element->adapter = g_object_new(GST_TYPE_AUDIOADAPTER, NULL);
	element->instrument = NULL;
	element->channel_name = NULL;
	element->data = NULL;
	element->maxdata = NULL;
	g_mutex_init(&element->bank_lock);
	element->last_gap = TRUE;
	element->EOS = FALSE;
	element->snr_mat = NULL;
	element->max_snr = FALSE;
	element->autocorrelation_matrix = NULL;
	element->autocorrelation_mask = NULL;
	element->autocorrelation_norm = NULL;
}

/*
 * An inspiral trigger, autocorrelation chisq, and coincidence (itacac) element
 *
 * Copyright (C) 2011 Chad Hanna, Kipp Cannon, 2018 Cody Messick, Alex Pace
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
#include <gmodule.h>
#include <gst/gst.h>
#include <gst/base/gstadapter.h>
#include <gst/base/gstaggregator.h>
#include <gst/controller/controller.h>
#include <math.h>
#include <string.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_blas.h>
#include <complex.h>

/*
 * our own stuff
 */

#include <gstlal/gstlal.h>
#include <gstlal/gstlal_debug.h>
#include <gstlal_itacac.h>
#include <gstlal/gstlal_peakfinder.h>
#include <gstlal/gstaudioadapter.h>
#include <gstlal/gstlal_tags.h>
#include <gstlal/gstlal_autocorrelation_chi2.h>
#include <gstlal_snglinspiral.h>
//#include <lal/LIGOMetadataTables.h>
#include <lal/LIGOMetadataUtils.h>
//#include <lal/LALDatatypes.h> 
#include <lal/TimeDelay.h>

/*
 * ============================================================================
 *
 *                           GStreamer Boiler Plate
 *
 * ============================================================================
 */


#define GST_CAT_DEFAULT gstlal_itacac_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);

G_DEFINE_TYPE(
	GSTLALItacacPad,
	gstlal_itacac_pad,
	GST_TYPE_AGGREGATOR_PAD
);

G_DEFINE_TYPE_WITH_CODE(
	GSTLALItacac,
	gstlal_itacac,
	GST_TYPE_AGGREGATOR,
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "lal_itacac", 0, "lal_itacac debug category")
);

/* 
 * Static pad templates, needed to make instances of GstAggregatorPad
 */

#define CAPS \
	"audio/x-raw, " \
	"format = (string) { " GST_AUDIO_NE(Z64) ", " GST_AUDIO_NE(Z128) ", " GST_AUDIO_NE(F64) "}, "\
	"rate = " GST_AUDIO_RATE_RANGE ", " \
	"channels = " GST_AUDIO_CHANNELS_RANGE ", " \
	"layout = (string) interleaved, " \
	"channel-mask = (bitmask) 0"

static GstStaticPadTemplate src_templ = GST_STATIC_PAD_TEMPLATE(
	"src",
	GST_PAD_SRC,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS("application/x-lal-snglinspiral")
);

static GstStaticPadTemplate sink_templ = GST_STATIC_PAD_TEMPLATE(
	"sink%d",
	GST_PAD_SINK,
	GST_PAD_REQUEST,
	GST_STATIC_CAPS(CAPS)
);

#define DEFAULT_SNR_THRESH 5.5
#define DEFAULT_COINC_THRESH 5

/*
 * ========================================================================
 *
 *                             Utility Functions
 *
 * ========================================================================
 */

static unsigned autocorrelation_channels(const GSTLALItacacPad *itacacpad) {
	return gstlal_autocorrelation_chi2_autocorrelation_channels(itacacpad->autocorrelation_matrix);
}

static unsigned autocorrelation_length(GSTLALItacacPad *itacacpad) {
	// The autocorrelation length should be the same for all of the detectors, so we can just use the first
	return gstlal_autocorrelation_chi2_autocorrelation_length(itacacpad->autocorrelation_matrix);
}

static guint64 output_num_samps(GSTLALItacacPad *itacacpad) {
	return (guint64) itacacpad->n;
}

static guint64 output_num_bytes(GSTLALItacacPad *itacacpad) {
	return (guint64) output_num_samps(itacacpad) * itacacpad->adapter->unit_size;
}

static int reset_time_and_offset(GSTLALItacac *itacac) {
	itacac->next_output_offset = 0;
	itacac->next_output_timestamp = GST_CLOCK_TIME_NONE;
        return 0;
}


static guint gst_audioadapter_available_samples(GstAudioAdapter *adapter) {
        guint size;
        g_object_get(adapter, "size", &size, NULL);
        return size;
}

static void free_bank(GSTLALItacacPad *itacacpad) {
	g_free(itacacpad->bank_filename);
	itacacpad->bank_filename = NULL;
	free(itacacpad->bankarray);
	itacacpad->bankarray = NULL;
}

static void update_peak_info_from_autocorrelation_properties(GSTLALItacacPad *itacacpad) {
	// FIXME Need to make sure that itacac can run without autocorrelation matrix
	if (itacacpad->maxdata && itacacpad->tmp_maxdata && itacacpad->autocorrelation_matrix) {
		itacacpad->maxdata->pad = itacacpad->tmp_maxdata->pad = autocorrelation_length(itacacpad) / 2;
		if (itacacpad->snr_mat)
			free(itacacpad->snr_mat);
		if (itacacpad->tmp_snr_mat)
			free(itacacpad->tmp_snr_mat);

		itacacpad->snr_mat = calloc(autocorrelation_channels(itacacpad) * autocorrelation_length(itacacpad), itacacpad->maxdata->unit);
		itacacpad->tmp_snr_mat = calloc(autocorrelation_channels(itacacpad) * autocorrelation_length(itacacpad), itacacpad->tmp_maxdata->unit);

		//
		// Each row is one sample point of the snr time series with N
		// columns for N channels. Assumes proper packing to go from real to complex.
		// FIXME assumes single precision
		//
		itacacpad->snr_matrix_view = gsl_matrix_complex_float_view_array((float *) itacacpad->snr_mat, autocorrelation_length(itacacpad), autocorrelation_channels(itacacpad));
		itacacpad->tmp_snr_matrix_view = gsl_matrix_complex_float_view_array((float *) itacacpad->tmp_snr_mat, autocorrelation_length(itacacpad), autocorrelation_channels(itacacpad));
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



static gboolean setcaps(GstAggregator *agg, GstAggregatorPad *aggpad, GstEvent *event) {
	GSTLALItacac *itacac = GSTLAL_ITACAC(agg);
	GSTLALItacacPad *itacacpad = GSTLAL_ITACAC_PAD(aggpad);
	GstCaps *caps;
	guint width = 0; 
	guint max_number_disjoint_sets_in_trigger_window;

	//
	// Update element metadata
	//
	gst_event_parse_caps(event, &caps);
	GstStructure *str = gst_caps_get_structure(caps, 0);
	const gchar *format = gst_structure_get_string(str, "format");
	gst_structure_get_int(str, "rate", &(itacacpad->rate));

	if(!strcmp(format, GST_AUDIO_NE(Z64))) {
		width = sizeof(float complex);
		itacacpad->peak_type = GSTLAL_PEAK_COMPLEX;
	} else if(!strcmp(format, GST_AUDIO_NE(Z128))) {
		width = sizeof(double complex);
		itacacpad->peak_type = GSTLAL_PEAK_DOUBLE_COMPLEX;
	} else
		GST_ERROR_OBJECT(itacac, "unsupported format %s", format);

	g_object_set(itacacpad->adapter, "unit-size", itacacpad->channels * width, NULL); 
	itacacpad->chi2 = calloc(itacacpad->channels, width);
	itacacpad->tmp_chi2 = calloc(itacacpad->channels, width);

	if (itacacpad->maxdata)
		gstlal_peak_state_free(itacacpad->maxdata);
	
	if (itacacpad->tmp_maxdata)
		gstlal_peak_state_free(itacacpad->tmp_maxdata);

	itacacpad->maxdata = gstlal_peak_state_new(itacacpad->channels, itacacpad->peak_type);
	itacacpad->tmp_maxdata = gstlal_peak_state_new(itacacpad->channels, itacacpad->peak_type);

	// This should be called any time the autocorrelation property is updated 
	update_peak_info_from_autocorrelation_properties(itacacpad);

	// The largest number of disjoint sets of non-gap-samples (large enough
	// to produce a trigger) that we could have in a given trigger window
	max_number_disjoint_sets_in_trigger_window = (guint) itacacpad->rate / (2 * itacacpad->maxdata->pad) + 1;
	itacacpad->saved_data->duration_dataoffset_trigwindowoffset_peakfindinglength_matrix = gsl_matrix_calloc(max_number_disjoint_sets_in_trigger_window, 3);

	// The max size that we may want to save is less than the typical
	// output size plus padding, plus another half of padding since we want
	// to save triggers towards the end of a trigger window in case
	// something in the next window is coincident

	return GST_AGGREGATOR_CLASS(gstlal_itacac_parent_class)->sink_event(agg, aggpad, event);


}

static gboolean sink_event(GstAggregator *agg, GstAggregatorPad *aggpad, GstEvent *event)
{
	GSTLALItacac *itacac = GSTLAL_ITACAC(agg);
	GSTLALItacacPad *itacacpad = GSTLAL_ITACAC_PAD(aggpad);
	gboolean result = TRUE;

	GST_DEBUG_OBJECT(aggpad, "Got %s event on sink pad", GST_EVENT_TYPE_NAME (event));

	switch (GST_EVENT_TYPE(event)) {
		case GST_EVENT_CAPS:
		{
			return setcaps(agg, aggpad, event);
		}
		case GST_EVENT_TAG:
		{
			GstTagList *taglist;
			gchar *instrument, *channel_name;
			gst_event_parse_tag(event, &taglist);
			result = taglist_extract_string(GST_OBJECT(aggpad), taglist, GSTLAL_TAG_INSTRUMENT, &instrument);
			result &= taglist_extract_string(GST_OBJECT(aggpad), taglist, GSTLAL_TAG_CHANNEL_NAME, &channel_name);
			if(result) {
				GST_DEBUG_OBJECT(aggpad, "found tags \"%s\"=\"%s\", \"%s\"=\"%s\"", GSTLAL_TAG_INSTRUMENT, instrument, GSTLAL_TAG_CHANNEL_NAME, channel_name);
				g_free(itacacpad->instrument);
				itacacpad->instrument = instrument;
				g_free(itacacpad->channel_name);
				itacacpad->channel_name = channel_name;
				g_mutex_lock(&itacacpad->bank_lock);
				gstlal_set_channel_in_snglinspiral_array(itacacpad->bankarray, itacacpad->channels, itacacpad->channel_name);
				gstlal_set_instrument_in_snglinspiral_array(itacacpad->bankarray, itacacpad->channels, itacacpad->instrument);
				g_mutex_unlock(&itacacpad->bank_lock);
			}
                        break;

		}
		case GST_EVENT_EOS:
		{
			itacac->EOS = TRUE;
			break;
		}
		default:
			break;
	}
	if(!result) {
		gst_event_unref(event);
	} else {
		result = GST_AGGREGATOR_CLASS(gstlal_itacac_parent_class)->sink_event(agg, aggpad, event);
	}
	return result;
}

/*
 * ============================================================================
 *
 *                                 Properties
 *
 * ============================================================================
 */


enum padproperty {
	ARG_N = 1,
	ARG_SNR_THRESH,
	ARG_BANK_FILENAME,
	ARG_SIGMASQ,
	ARG_AUTOCORRELATION_MATRIX,
	ARG_AUTOCORRELATION_MASK
};

enum itacacproperty {
	ARG_COINC_THRESH = 1
};

static void gstlal_itacac_pad_set_property(GObject *object, enum padproperty id, const GValue *value, GParamSpec *pspec)
{
	GSTLALItacacPad *itacacpad = GSTLAL_ITACAC_PAD(object);

	GST_OBJECT_LOCK(itacacpad);

	switch(id) {
	case ARG_N:
		itacacpad->n = g_value_get_uint(value);
		break;

	case ARG_SNR_THRESH:
		itacacpad->snr_thresh = g_value_get_double(value);
		break;

	case ARG_BANK_FILENAME:
		g_mutex_lock(&itacacpad->bank_lock);
		itacacpad->bank_filename = g_value_dup_string(value);
		itacacpad->channels = gstlal_snglinspiral_array_from_file(itacacpad->bank_filename, &(itacacpad->bankarray));
		gstlal_set_min_offset_in_snglinspiral_array(itacacpad->bankarray, itacacpad->channels, &(itacacpad->difftime));
		if(itacacpad->instrument && itacacpad->channel_name) {
			gstlal_set_instrument_in_snglinspiral_array(itacacpad->bankarray, itacacpad->channels, itacacpad->instrument);
			gstlal_set_channel_in_snglinspiral_array(itacacpad->bankarray, itacacpad->channels, itacacpad->channel_name);
		}
		g_mutex_unlock(&itacacpad->bank_lock);
		break;

	case ARG_SIGMASQ:
		g_mutex_lock(&itacacpad->bank_lock);
		if (itacacpad->bankarray) {
			gint length;
			double *sigmasq = gstlal_doubles_from_g_value_array(g_value_get_boxed(value), NULL, &length);
			if((gint) itacacpad->channels != length)
				GST_ERROR_OBJECT(itacacpad, "vector length (%d) does not match number of templates (%d)", length, itacacpad->channels);
			else
				gstlal_set_sigmasq_in_snglinspiral_array(itacacpad->bankarray, length, sigmasq);
			g_free(sigmasq);
		} else
			GST_WARNING_OBJECT(itacacpad, "must set template bank before setting sigmasq");
		g_mutex_unlock(&itacacpad->bank_lock);
		break;


	case ARG_AUTOCORRELATION_MATRIX:
		g_mutex_lock(&itacacpad->bank_lock);

		if(itacacpad->autocorrelation_matrix)
			gsl_matrix_complex_free(itacacpad->autocorrelation_matrix);

		itacacpad->autocorrelation_matrix = gstlal_gsl_matrix_complex_from_g_value_array(g_value_get_boxed(value)); 

		// This should be called any time caps change too
		update_peak_info_from_autocorrelation_properties(itacacpad);

		//
		// induce norms to be recomputed
		//

		if(itacacpad->autocorrelation_norm) {
			gsl_vector_free(itacacpad->autocorrelation_norm);
			itacacpad->autocorrelation_norm = NULL;
		}

		g_mutex_unlock(&itacacpad->bank_lock);
		break;

        case ARG_AUTOCORRELATION_MASK: 
		g_mutex_lock(&itacacpad->bank_lock);

		if(itacacpad->autocorrelation_mask)
			gsl_matrix_int_free(itacacpad->autocorrelation_mask);

		itacacpad->autocorrelation_mask = gstlal_gsl_matrix_int_from_g_value_array(g_value_get_boxed(value));

		g_mutex_unlock(&itacacpad->bank_lock);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(itacacpad);
}


static void gstlal_itacac_pad_get_property(GObject *object, enum padproperty id, GValue *value, GParamSpec *pspec)
{
	GSTLALItacacPad *itacacpad = GSTLAL_ITACAC_PAD(object);

	GST_OBJECT_LOCK(itacacpad);

	switch(id) {
	case ARG_N:
		g_value_set_uint(value, itacacpad->n);
		break;

	case ARG_SNR_THRESH:
		g_value_set_double(value, itacacpad->snr_thresh);
		break;

	case ARG_BANK_FILENAME:
		g_mutex_lock(&itacacpad->bank_lock);
		g_value_set_string(value, itacacpad->bank_filename);
		g_mutex_unlock(&itacacpad->bank_lock);
		break;

        case ARG_SIGMASQ:
		g_mutex_lock(&itacacpad->bank_lock);
		if(itacacpad->bankarray) {
			double sigmasq[itacacpad->channels];
			gint i;
			for(i = 0; i < (gint) itacacpad->channels; i++)
				sigmasq[i] = itacacpad->bankarray[i].sigmasq;
			g_value_take_boxed(value, gstlal_g_value_array_from_doubles(sigmasq, itacacpad->channels));
		} else {
			GST_WARNING_OBJECT(itacacpad, "no template bank");
			//g_value_take_boxed(value, g_value_array_new(0));
			// FIXME Is this right?
			g_value_take_boxed(value, g_array_sized_new(TRUE, TRUE, sizeof(double), 0));
		}
		g_mutex_unlock(&itacacpad->bank_lock);
		break;

	case ARG_AUTOCORRELATION_MATRIX:
		g_mutex_lock(&itacacpad->bank_lock);
		if(itacacpad->autocorrelation_matrix)
                        g_value_take_boxed(value, gstlal_g_value_array_from_gsl_matrix_complex(itacacpad->autocorrelation_matrix));
                else {
                        GST_WARNING_OBJECT(itacacpad, "no autocorrelation matrix");
			// FIXME g_value_array_new() is deprecated
                        //g_value_take_boxed(value, g_value_array_new(0)); 
			// FIXME Is this right?
			g_value_take_boxed(value, g_array_sized_new(TRUE, TRUE, sizeof(double), 0));
                        }
                g_mutex_unlock(&itacacpad->bank_lock);
                break;

	case ARG_AUTOCORRELATION_MASK:
		g_mutex_lock(&itacacpad->bank_lock);
		if(itacacpad->autocorrelation_mask)
			g_value_take_boxed(value, gstlal_g_value_array_from_gsl_matrix_int(itacacpad->autocorrelation_mask));
		else {
			GST_WARNING_OBJECT(itacacpad, "no autocorrelation mask");
			//g_value_take_boxed(value, g_value_array_new(0));
			// FIXME Is this right?
			g_value_take_boxed(value, g_array_sized_new(TRUE, TRUE, sizeof(double), 0));
		}
		g_mutex_unlock(&itacacpad->bank_lock);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(itacacpad);
}

static void gstlal_itacac_set_property(GObject *object, enum itacacproperty id, const GValue *value, GParamSpec *pspec)
{
	GSTLALItacac *itacac = GSTLAL_ITACAC(object);

	GST_OBJECT_LOCK(itacac);

	switch(id) {
	case ARG_COINC_THRESH:
		itacac->coinc_thresh = g_value_get_double(value);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(itacac);
}


static void gstlal_itacac_get_property(GObject *object, enum itacacproperty id, GValue *value, GParamSpec *pspec)
{
	GSTLALItacac *itacac = GSTLAL_ITACAC_PAD(object);

	GST_OBJECT_LOCK(itacac);

	switch(id) {
	case ARG_COINC_THRESH:
		g_value_set_double(value, itacac->coinc_thresh);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(itacac);
}

/*
 * aggregate()
*/ 

static void update_state(GSTLALItacac *itacac, GstBuffer *srcbuf) {
	itacac->next_output_offset = GST_BUFFER_OFFSET_END(srcbuf);
	itacac->next_output_timestamp = GST_BUFFER_PTS(srcbuf) - itacac->difftime;
	itacac->next_output_timestamp += GST_BUFFER_DURATION(srcbuf);
}

//static void update_sink_state(GSTLALItacac *itacac


static GstFlowReturn push_buffer(GSTLALItacac *itacac, GstBuffer *srcbuf) {
	GstFlowReturn result = GST_FLOW_OK;
	GSTLALItacacPad *itacacpad = GSTLAL_ITACAC_PAD(GST_ELEMENT(itacac)->sinkpads->data);

	update_state(itacac, srcbuf);

	GST_DEBUG_OBJECT(itacac, "pushing %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(srcbuf));

	result = gst_pad_push(GST_PAD((itacac->aggregator).srcpad), srcbuf);
	return result;
}

static GstFlowReturn push_gap(GSTLALItacac *itacac, guint samps) {
	GstFlowReturn result = GST_FLOW_OK;
	GSTLALItacacPad *itacacpad = GSTLAL_ITACAC_PAD(GST_ELEMENT(itacac)->sinkpads->data);
	GstBuffer *srcbuf = gst_buffer_new();
	GST_BUFFER_FLAG_SET(srcbuf, GST_BUFFER_FLAG_GAP);

	GST_BUFFER_OFFSET(srcbuf) = itacac->next_output_offset;
	GST_BUFFER_OFFSET_END(srcbuf) = itacac->next_output_offset + samps;
	GST_BUFFER_PTS(srcbuf) = itacac->next_output_timestamp + itacac->difftime;
	GST_BUFFER_DURATION(srcbuf) = (GstClockTime) gst_util_uint64_scale_int_round(GST_SECOND, samps, itacac->rate);
	GST_BUFFER_DTS(srcbuf) = GST_BUFFER_PTS(srcbuf);

	result = push_buffer(itacac, srcbuf);

	return result;

}

// FIXME Is there a better way to do this? 
// NOTE Will need to add new IFOs as we start using them
InterferometerNumber get_interferometer_number(GSTLALItacac *itacac, char *ifo) {
	InterferometerNumber result;
	if(strcmp(ifo, "H1") == 0) 
		result = LAL_IFO_H1;
	else if(strcmp(ifo, "L1") == 0)
		result = LAL_IFO_L1;
	else if(strcmp(ifo, "V1") == 0)
		result = LAL_IFO_V1;
	else
		GST_ERROR_OBJECT(GST_ELEMENT(itacac), "no support for ifo %s", ifo);

	return result;
}

static GstFlowReturn final_setup(GSTLALItacac *itacac) {
	// FIXME Need to add logic to finish initializing GLists. Make sure to ensure there always at least two elements in the GList, even in the case of only having one sinkpad
	GstElement *element = GST_ELEMENT(itacac);
	GSTLALItacacPad *itacacpad, *itacacpad2;
	GList *padlist, *padlist2;
	guint16 n_ifos = element->numsinkpads;
	guint16 i;
	guint16 num_pairwise_combinations = 1;
	LALDetector *ifo1, *ifo2;
	gdouble *coincidence_window = malloc(sizeof(gdouble));
	guint coinc_window_samps = 0;
	GstFlowReturn result = GST_FLOW_OK;

	// Ensure all of the pads have the same channels and rate, and set them on itacac for easy access
	for(padlist = element->sinkpads; padlist !=NULL; padlist = padlist->next) {
		itacacpad = GSTLAL_ITACAC_PAD(padlist->data);
		// FIXME Should gst_object_sync_values be called here too?
		if(padlist == element->sinkpads){
			itacac->channels = itacacpad->channels;
			itacac->rate = itacacpad->rate;
			itacac->difftime = itacacpad->difftime;
			itacac->peak_type = itacacpad->peak_type;
		} else {
			g_assert(itacac->channels == itacacpad->channels);
			g_assert(itacac->rate == itacacpad->rate);
			g_assert(itacac->difftime == itacacpad->difftime);
			g_assert(itacac->peak_type == itacacpad->peak_type);
		}

	}

	// Compute number of cominbations, n_ifos choose 2
	for(i=n_ifos; i > n_ifos - 2; i--) 
		num_pairwise_combinations *= i;

	num_pairwise_combinations /= 2;

	// Set-up hash table with g_hash_table_insert
	for(padlist = element->sinkpads; padlist !=NULL; padlist = padlist->next) {
		itacacpad = GSTLAL_ITACAC_PAD(padlist->data);
		// Get a LALDetector struct describing the ifo we care about, we need this to compute the light travel time
		XLALReturnDetector(ifo1, get_interferometer_number(itacac, itacacpad->instrument));
		// Set up IFO1IFO2 string, copy IFO1 into it
		//memcpy(itacac->ifo_pair, itacacpad->instrument, 2*sizeof(gchar));
		itacac->ifo_pair[0] = itacacpad->instrument[0];

		for(padlist2 = padlist->next; padlist2 != NULL; padlist2 = padlist2->next) {
			itacacpad2 = GSTLAL_ITACAC_PAD(padlist2->data);
			// Get a LALDetector struct describing the ifo we care about, we need this to compute the light travel time
			XLALReturnDetector(ifo2, get_interferometer_number(itacac, itacacpad2->instrument));
			// Set up IFO1IFO2 string, copy IFO2 into it
			//memcpy(itacac->ifo_pair + 2, itacacpad2->instrument, 2*sizeof(gchar));
			itacac->ifo_pair[1] = itacacpad2->instrument[0];

			// Get light travel time and add coincidence threshold to it. Coincidence threshold is in milliseconds, light travel time is in nanoseconds
			*coincidence_window = itacac->coinc_thresh*1000000 + (gdouble) XLALLightTravelTime(ifo1, ifo2);

			// Get largest coincidence window measured in sample points (rounded up)
			coinc_window_samps = (guint) ( (*coincidence_window / 1000000000) / (gdouble) itacac->rate ) + 1;
			itacac->max_coinc_window_samps = coinc_window_samps > itacac->max_coinc_window_samps ? coinc_window_samps : itacac->max_coinc_window_samps;

			g_hash_table_insert(itacac->coinc_window_hashtable, (gpointer) itacac->ifo_pair, (gpointer) coincidence_window);
		}
		// FIXME does coincidence_window need to be freed?
		free(coincidence_window);
	}

	// FIXME Currently the padding used to compute chisq much be larger than the largest coincidence window, meaning itacac cannot be run without computing chisq
	// This assumption should be undone in case we ever want to just compute SNRs for triggers
	g_assert(element->sinkpads->data->maxdata->pad > itacac->max_coinc_window_samps);

	// Set up the order that we want to check the pads for coincidence
	// FIXME Fow now this assumes <= 3 IFOs and the order will be L1H1V1
	// (so if you get a trigger in H1, first check L1 then V1; if you get a
	// trigger in V1, first check L1 then H1), this should either be
	// dynamically set or we should look for all coincidences, idk quit
	// asking me questions
	if(GST_ELEMENT(itacac)->numsinkpads > 1) {
		for(padlist = GST_ELEMENT(itacac)->sinkpads; padlist != NULL; padlist = padlist->next) {
			itacacpad = GSTLAL_ITACAC_PAD(padlist->data);
			for(padlist2=GST_ELEMENT(itacac)->sinkpads; padlist2 != NULL; padlist2 = padlist2->next) {
				itacacpad2 = GSTLAL_ITACAC_PAD(padlist2->data);
				if(strcmp(itacacpad->instrument, itacacpad2->instrument) == 0)
					continue;
				else if(strcmp(itacacpad->instrument, "H1") == 0) {
					if(strcmp(itacacpad2->instrument, "L1") == 0) {
						if(itacacpad->next_in_coinc_order == NULL)
							// list hasnt been set yet
							itacacpad->next_in_coinc_order = g_list_append(itacacpad->next_in_coinc_order, itacacpad2);
						else
							// V1 was already added to list
							itacacpad->next_in_coinc_order = g_list_prepend(itacacpad->next_in_coinc_order, itacacpad2);
					} else 
						// we've got V1, which should go at the end of the list
						itacacpad->next_in_coinc_order = g_list_append(itacacpad->next_in_coinc_order, itacacpad2);
					
				} else if(strcmp(itacacpad->instrument, "L1") == 0) {
					if(strcmp(itacacpad2->instrument, "H1") == 0) {
						if(itacacpad->next_in_coinc_order == NULL)
							// list hasnt been set yet
							itacacpad->next_in_coinc_order = g_list_append(itacacpad->next_in_coinc_order, itacacpad2);
						else
							// V1 was already added to list
							itacacpad->next_in_coinc_order = g_list_prepend(itacacpad->next_in_coinc_order, itacacpad2);
					} else 
						// we've got V1, which should go at the end of the list
						itacacpad->next_in_coinc_order = g_list_append(itacacpad->next_in_coinc_order, itacacpad2);

				} else if(strcmp(itacacpad->instrument, "V1") == 0) {
					if(strcmp(itacacpad2->instrument, "L1") == 0) {
						if(itacacpad->next_in_coinc_order == NULL)
							// list hasnt been set yet
							itacacpad->next_in_coinc_order = g_list_append(itacacpad->next_in_coinc_order, itacacpad2);
						else
							// H1 was already added to list
							itacacpad->next_in_coinc_order = g_list_prepend(itacacpad->next_in_coinc_order, itacacpad2);
					} else 
						// we've got H1, which should go at the end of the list
						itacacpad->next_in_coinc_order = g_list_append(itacacpad->next_in_coinc_order, itacacpad2);

				} else {
					GST_ERROR_OBJECT(GST_ELEMENT(aggregator), "instrument %s not supported", itacacpad->instrument);
					result = GST_FLOW_ERROR;
					return result;
				}
			}
		}
	}
	// The max size to copy from an adapter is the typical output size plus
	// the padding plus the largest coincidence window. we should never try
	// to copy from an adapter with a larger buffer than this. 
	itacacpad->data->data = malloc(output_num_bytes(itacacpad) + itacacpad->adapter->unit_size * (2 * itacacpad->maxdata->pad + itacac->max_coinc_window_samps));

	itacac->trigger_end = malloc(element->numsinkpads*sizeof(LIGOTimeGPS));

	return result;
}

static void copy_nongapsamps(GSTLALItacac *itacac, GSTLALItacacPad *itacacpad, guint copysamps, guint peak_finding_length, guint previously_searched_samples, gint offset_from_trigwindow) {
	guint data_container_index = 0;
	guint offset_from_copied_data = 0;
	guint duration = gsl_matrix_get(itacacpad->data->duration_dataoffset_trigwindowoffset_peakfindinglength_matrix, 0, 0);

	//
	// Description of duration_dataoffset_trigwindowoffset_peakfindinglength_matrix
	// First column is duration, or number of samples that we've copied
	// Second column is data offset, so the number of samples that come before the data that was just copied
	// Third column is trigger window offset, the distance in samples from the beginning of the data to the beginning of the trigger window. This number can be negative
	// Fourth column is the peak finding length used by the peakfinder
	//

	// set the columns of the matrix described above
	while(duration != 0) {
		offset_from_copied_data += duration;
		duration = gsl_matrix_get(itacacpad->data->duration_dataoffset_trigwindowoffset_peakfindinglength_matrix, ++data_container_index, 0);
	}
	gsl_matrix_set(itacacpad->data->duration_dataoffset_trigwindowoffset_peakfindinglength_matrix, data_container_index, 0, (double) copysamps);
	gsl_matrix_set(itacacpad->data->duration_dataoffset_trigwindowoffset_peakfindinglength_matrix, data_container_index, 1, (double) offset_from_copied_data);
	gsl_matrix_set(itacacpad->data->duration_dataoffset_trigwindowoffset_peakfindinglength_matrix, data_container_index, 2, (double) offset_from_trigwindow);
	gsl_matrix_set(itacacpad->data->duration_dataoffset_trigwindowoffset_peakfindinglength_matrix, data_container_index, 3, (double) peak_finding_length);

	// copy the samples that we will call the peak finding library on (if no events are found the result will be a GAP)
	gst_audioadapter_copy_samples(itacacpad->adapter, itacacpad->data->data + offset_from_copied_data * itacacpad->maxdata->channels, copysamps, NULL, NULL);

}

static void generate_triggers(GSTLALItacac *itacac, GSTLALItacacPad *itacacpad, void *data, guint peak_finding_start, guint peak_finding_length, guint samples_previously_searched, gboolean numerous_peaks_in_window) {
	gsl_error_handler_t *old_gsl_error_handler;

	struct gstlal_peak_state *this_maxdata;
	void *this_snr_mat;
	void *this_chi2;
	guint channel;

	// Need to use our tmp_chi2 and tmp_maxdata struct and its corresponding snr_mat if we've already found a peak in this window
	if(numerous_peaks_in_window) {
		this_maxdata = itacacpad->tmp_maxdata;
		this_snr_mat = itacacpad->tmp_snr_mat;
		this_chi2 = itacacpad->tmp_chi2;
		// FIXME At the moment, empty triggers are added to inform the
		// "how many instruments were on test", the correct thing to do
		// is probably to add metadata to the buffer containing
		// information about which instruments were on
		this_maxdata->no_peaks_past_threshold = itacacpad->maxdata->no_peaks_past_threshold;
	} else {
		this_maxdata = itacacpad->maxdata;
		this_snr_mat = itacacpad->snr_mat;
		this_chi2 = itacacpad->chi2;
		// This boolean will be set to false if we find any peaks above threshold
		// FIXME At the moment, empty triggers are added to inform the
		// "how many instruments were on test", the correct thing to do
		// is probably to add metadata to the buffer containing
		// information about which instruments were on
		this_maxdata->no_peaks_past_threshold = TRUE;
	}

	// Update the snr threshold
	this_maxdata->thresh = itacacpad->snr_thresh;

	// AEP- 180417 Turning XLAL Errors off
	old_gsl_error_handler=gsl_set_error_handler_off();

        if (itacac->peak_type == GSTLAL_PEAK_COMPLEX) {
		// put the data pointer at the start of the interval we care about
                itacacpad->data->dataptr.as_complex = ((float complex *) itacacpad->data->data) + peak_finding_start * this_maxdata->channels;
                // Find the peak 
                gstlal_float_complex_peak_over_window_interp(this_maxdata, itacacpad->data->dataptr.as_complex, peak_finding_length);
		//FIXME At the moment, empty triggers are added to inform the
		//"how many instruments were on test", the correct thing to do
		//is probably to add metadata to the buffer containing
		//information about which instruments were on
		if(this_maxdata->no_peaks_past_threshold) {
			for(channel = 0; channel < this_maxdata->channels; channel++) {
				if((this_maxdata->interpvalues).as_float_complex[channel] != (float complex) 0) {
					this_maxdata->no_peaks_past_threshold = FALSE;
					break;
				}
			}
		}
	}
        else if (itacac->peak_type == GSTLAL_PEAK_DOUBLE_COMPLEX) {
		// put the data pointer at the start of the interval we care about
                itacacpad->data->dataptr.as_double_complex = ((double complex *) itacacpad->data->data) + peak_finding_start * this_maxdata->channels;
                // Find the peak 
                gstlal_double_complex_peak_over_window_interp(this_maxdata, itacacpad->data->dataptr.as_double_complex, peak_finding_length);
		//FIXME At the moment, empty triggers are added to inform the
		//"how many instruments were on test", the correct thing to do
		//is probably to add metadata to the buffer containing
		//information about which instruments were on
		if(this_maxdata->no_peaks_past_threshold) {
			for(channel = 0; channel < this_maxdata->channels; channel++) {
				if((this_maxdata->interpvalues).as_double_complex[channel] != (double complex) 0) {
					this_maxdata->no_peaks_past_threshold = FALSE;
					break;
				}
			}
		}
	}
        else
                g_assert_not_reached();

        // AEP- 180417 Turning XLAL Errors back on
        gsl_set_error_handler(old_gsl_error_handler);

	// Compute \chi^2 values if we can
	if(itacacpad->autocorrelation_matrix && !this_maxdata->no_peaks_past_threshold) {
		// Compute the chisq norm if it doesn't exist
		if(!itacacpad->autocorrelation_norm)
			itacacpad->autocorrelation_norm = gstlal_autocorrelation_chi2_compute_norms(itacacpad->autocorrelation_matrix, NULL);

		g_assert(autocorrelation_length(itacacpad) & 1);  // must be odd 

		if(itacac->peak_type == GSTLAL_PEAK_DOUBLE_COMPLEX) {
			/* extract data around peak for chisq calculation */
			gstlal_double_complex_series_around_peak(this_maxdata, itacacpad->data->dataptr.as_double_complex, (double complex *) this_snr_mat, this_maxdata->pad);
			gstlal_autocorrelation_chi2((double *) this_chi2, (double complex *) this_snr_mat, autocorrelation_length(itacacpad), -((int) autocorrelation_length(itacacpad)) / 2, 0.0, itacacpad->autocorrelation_matrix, itacacpad->autocorrelation_mask, itacacpad->autocorrelation_norm);

		} else if (itacac->peak_type == GSTLAL_PEAK_COMPLEX) {
			/* extract data around peak for chisq calculation */
			gstlal_float_complex_series_around_peak(this_maxdata, itacacpad->data->dataptr.as_complex, (float complex *) this_snr_mat, this_maxdata->pad);
			gstlal_autocorrelation_chi2_float((float *) this_chi2, (float complex *) this_snr_mat, autocorrelation_length(itacacpad), -((int) autocorrelation_length(itacacpad)) / 2, 0.0, itacacpad->autocorrelation_matrix, itacacpad->autocorrelation_mask, itacacpad->autocorrelation_norm);
		} else
			g_assert_not_reached();
	} 

	// Adjust the location of the peak by the number of samples processed in this window before this function call
	if(samples_previously_searched > 0 && !this_maxdata->no_peaks_past_threshold) {
		if(itacac->peak_type == GSTLAL_PEAK_DOUBLE_COMPLEX) {
			for(channel=0; channel < this_maxdata->channels; channel++) {
				if(cabs( (double complex) (this_maxdata->values).as_double_complex[channel]) > 0) {
					this_maxdata->samples[channel] += samples_previously_searched;
					this_maxdata->interpsamples[channel] += (double) samples_previously_searched;
				}
			}
		} else {
			for(channel=0; channel < this_maxdata->channels; channel++) {
				if(cabs( (double complex) (this_maxdata->values).as_float_complex[channel]) > 0) {
					this_maxdata->samples[channel] += samples_previously_searched;
					this_maxdata->interpsamples[channel] += (double) samples_previously_searched;
				}
			}
		}
	}

	// Combine with previous peaks found if any
	if(numerous_peaks_in_window && !this_maxdata->no_peaks_past_threshold) {
		// replace an original peak with a second peak, we need to...
		// Replace maxdata->interpvalues.as_float_complex, maxdata->interpvalues.as_double_complex etc with whichever of the two is larger
		// // Do same as above, but with maxdata->values instead of maxdata->interpvalues
		// // Make sure to replace samples and interpsamples too
		gsl_vector_complex_float_view old_snr_vector_view, new_snr_vector_view;
		double old_snr, new_snr;
		if(itacac->peak_type == GSTLAL_PEAK_DOUBLE_COMPLEX) {
			double *old_chi2 = (double *) itacacpad->chi2;
			double *new_chi2 = (double *) itacacpad->tmp_chi2;
			for(channel=0; channel < this_maxdata->channels; channel++) {
				// Possible cases
				// itacacpad->maxdata has a peak but itacacpad->tmp_maxdata does not <--No change required
				// itacacpad->tmp_maxdata has a peak but itacacpad->maxdata does not <--Swap out peaks
				// Both have peaks and itacacpad->maxdata's is higher <--No change required
				// Both have peaks and itacacpad->tmp_maxdata's is higher <--Swap out peaks
				old_snr = cabs( (double complex) (itacacpad->maxdata->interpvalues).as_double_complex[channel]);
				new_snr = cabs( (double complex) (itacacpad->maxdata->interpvalues).as_double_complex[channel]);

				if(new_snr > old_snr) {
					// The previous peak found was larger than the current peak. If there was a peak before but not now, increment itacacpad->maxdata's num_events
					if(old_snr == 0)
						// FIXME confirm that this isnt affected by floating point error
						itacacpad->maxdata->num_events++;

					(itacacpad->maxdata->values).as_double_complex[channel] = (itacacpad->maxdata->values).as_double_complex[channel];
					(itacacpad->maxdata->interpvalues).as_double_complex[channel] = (itacacpad->maxdata->interpvalues).as_double_complex[channel];
					itacacpad->maxdata->samples[channel] = itacacpad->maxdata->samples[channel];
					itacacpad->maxdata->interpsamples[channel] = itacacpad->maxdata->interpsamples[channel];
					old_chi2[channel] = new_chi2[channel];

					if(itacacpad->autocorrelation_matrix) {
						// Replace the snr time series around the peak with the new one
						old_snr_vector_view = gsl_matrix_complex_float_column(&(itacacpad->snr_matrix_view.matrix), channel);
						new_snr_vector_view = gsl_matrix_complex_float_column(&(itacacpad->tmp_snr_matrix_view.matrix), channel);
						old_snr_vector_view.vector = new_snr_vector_view.vector;
					}
				}
			}
		} else {
			float *old_chi2 = (float *) itacacpad->chi2;
			float *new_chi2 = (float *) itacacpad->tmp_chi2;
			for(channel=0; channel < itacacpad->maxdata->channels; channel++) {
				// Possible cases
				// itacacpad->maxdata has a peak but itacacpad->tmp_maxdata does not <--No change required
				// itacacpad->tmp_maxdata has a peak but itacacpad->maxdata does not <--Swap out peaks
				// Both have peaks and itacacpad->maxdata's is higher <--No change required 
				// Both have peaks and itacacpad->tmp_maxdata's is higher <--Swap out peaks
				old_snr = cabs( (double complex) (itacacpad->maxdata->interpvalues).as_float_complex[channel]);
				new_snr = cabs( (double complex) (itacacpad->maxdata->interpvalues).as_float_complex[channel]);
				if(new_snr > old_snr) {
					// The previous peak found was larger than the current peak. If there was a peak before but not now, increment itacacpad->maxdata's num_events
					if(old_snr == 0)
						// FIXME confirm that this isnt affected by floating point error
						itacacpad->maxdata->num_events++;

					(itacacpad->maxdata->values).as_float_complex[channel] = (itacacpad->maxdata->values).as_float_complex[channel];
					(itacacpad->maxdata->interpvalues).as_float_complex[channel] = (itacacpad->maxdata->interpvalues).as_float_complex[channel];
					itacacpad->maxdata->samples[channel] = itacacpad->maxdata->samples[channel];
					itacacpad->maxdata->interpsamples[channel] = itacacpad->maxdata->interpsamples[channel];
					old_chi2[channel] = new_chi2[channel];

					if(itacacpad->autocorrelation_matrix) {
						// Replace the snr time series around the peak with the new one
						old_snr_vector_view = gsl_matrix_complex_float_column(&(itacacpad->snr_matrix_view.matrix), channel);
						new_snr_vector_view = gsl_matrix_complex_float_column(&(itacacpad->tmp_snr_matrix_view.matrix), channel);
						old_snr_vector_view.vector = new_snr_vector_view.vector;
					}
				}
			}
		}
	}

}

static void find_coincident_triggers(GSTLALItacac *itacac, GSTLALItacacPad *itacacpad) {
	// Iterate through triggers already recovered, checking for
	// coincidence. If a trigger exists in one detector and does not have a
	// coincident trigger in another detector, we will find the highest
	// peak the second detector that is in coincidence with the original
	// trigger, and output that along with the surrounding snr time series

	glist *padlist;
	GSTLALItacacPad *other_itacacpad;
	// FIXME Allocate in advance
	guint16 i;
	gdouble *coincidence_window, dt;
	guint channel;

	// NOTE this assume all detectors follow A1 convention (e.g. H1, L1, V1)
	itacac->ifo_pair[0] = itacacpad->instrument[0];

	
	// steps
	// 1) get trigger time for first pad
	// 2) Check if coincident trigger in other pads (should look at python coincidence code for how to do coincidence for 3+ ifos)
	// 3a) If coincident triggers in all pads, we're done here
	// 3b) If have (n-1) coincident triggers (where n is number of ifos), look for subthreshold coincident trigger in last that is coincident with all (n-1) triggers
	// 3c) If have no coincident triggers, first check most significant detector, then look for trig from least sensitive that is coinsistent with first two FIXME This currently assumes 3 ifos

	//
	// Ordering of maxdata structs in maxdata list
	//
	// After first entry in each list, which is the maxdata struct containing the original trigger from that ifo, order is same as order of itacacpads
	// e.g. if element->sinkpads is H1, element->sinkpads->next is L1, and element->sinkpads->next->next is V1, then the order of maxdata structs for 
	// the H1 pad is H1,L1,V1, the order of maxdata structs for the L1 pad is L1,H1,V1 and the order for the V1 pad is V1,H1,L1
	//

	if(itacac->peak_type == GSTLAL_PEAK_COMPLEX) {
		for(channel = 0; channel < itacacpad->maxdata->channels; channel++) {
			// FIXME Should make sure that values is used elsewhere instead of interpvalues as well for this check (specifically for no_peaks_past_threshold)
			if((itacacpad->maxdata->values).as_float_complex[channel] != (float complex) 0) {
				XLALINT8NSToGPS(itacac->trigger_end, itacac->next_output_timestamp + itacac->difftime);
				XLALGPSAdd(itacac->trigger_end, (double) itacacpad->maxdata->interpsamples[channel] / itacac->rate);

				// FIXME designed for 3 or less detectors right now
				padlist = itacacpad->next_in_coinc_order;
				itacacpad = GSTLAL_ITACAC_PAD(padlist->data);
				itacac->ifo_pair[1] = itacacpad->instrument[0];
				coincidence_window = g_hash_table_lookup(itacac->coinc_window_hashtable, itacac->ifo_pair);
				// If trigger was found and its coincident, we dont need to worry
				// If trigger was found and its not coincident, we need to find a coincident subthreshold trigger
				// If not trigger was found, we need to find a coincident subthreshold trigger
				/*
				if((itacacpad->maxdata->values).as_float_complex[channel] != (float complex) 0) {
					// No trigger above threshold was found
					XLALINT8NSToGPS(trigger_end + 1, itacac->next_output_timestamp + itacac->difftime);
					XLALGPSAdd(trigger_end + 1, (double) itacacpad->maxdata->interpsamples[channel] / itacac->rate);
				} else
					XLALINT8NSToGPS(trigger_end + 1, 0);
				*/
				
				dt = fabs(1000000000 * XLALGPSDiff(trigger_end[0], trigger_end[1]));
				if(dt > *coincidence_window) {
					// Need to find a subthreshold trigger
					// First figure out the range of samples that we're going to search
					// Then use that and duration_dataoffset_trigwindowoffset_peakfindinglength_matrix to figure out where the data we need is
					// Grab largest peak in window we care about
					// Compute chisq

				}
				}
			}
		}
	}
}

static GstFlowReturn process(GSTLALItacac *itacac) {
	// Iterate through audioadapters and generate triggers
	// FIXME Needs to push a gap for the padding of the first buffer with nongapsamps

	
	GstElement *element = GST_ELEMENT(itacac);
	guint outsamps, nongapsamps, copysamps, samples_left_in_window, previous_samples_left_in_window;
	guint gapsamps = 0;
	GstFlowReturn result = GST_FLOW_OK;
	GList *padlist;
	GSTLALItacacPad *itacacpad;
	GstBuffer *srcbuf = NULL;
	guint availablesamps;

	// Make sure we have enough samples to produce a trigger
	// All of the sinkpads should have the same number of samples, so we'll just check the first one FIXME this is probably not true
	// FIXME Currently assumes every pad has the same n
	itacacpad = GSTLAL_ITACAC_PAD(element->sinkpads->data);
	if(gst_audioadapter_available_samples( itacacpad->adapter ) <= itacacpad->n + 2*itacacpad->maxdata->pad + itacac->max_coinc_window_samps && !itacac->EOS)
		return result;


	for(padlist = element->sinkpads; padlist != NULL; padlist = padlist->next) {
		itacacpad = GSTLAL_ITACAC_PAD(padlist->data);

		samples_left_in_window = itacacpad->n;

		// FIXME Currently assumes n is the same for all detectors
		while( samples_left_in_window > 0 && ( !itacac->EOS || (itacac->EOS && gst_audioadapter_available_samples(itacacpad->adapter)) ) ) {

			// Check how many gap samples there are until a nongap
			// or vice versa, depending on which comes first
			previous_samples_left_in_window = samples_left_in_window;
			gapsamps = gst_audioadapter_head_gap_length(itacacpad->adapter);
			nongapsamps = gst_audioadapter_head_nongap_length(itacacpad->adapter);
			availablesamps = gst_audioadapter_available_samples(itacacpad->adapter);

			// NOTE Remember that if you didnt just come off a gap, you should always have a pad worth of nongap samples that have been processed already

			// Check if the samples are gap, and flush up to samples_left_in_window of them if so
			if(gapsamps > 0) {
				itacacpad->last_gap = TRUE; // FIXME This will not work for case where each itacac has multiple sinkpads
				outsamps = gapsamps > samples_left_in_window ? samples_left_in_window : gapsamps;
				gst_audioadapter_flush_samples(itacacpad->adapter, outsamps);
				samples_left_in_window -= outsamps;
			}
			// Check if we have enough nongap samples to compute chisq for a potential trigger, and if not, check if we should flush the samples or 
			// if theres a possibility we could get more nongapsamples in the future
			else if(nongapsamps <= 2 * itacacpad->maxdata->pad) {
				if(samples_left_in_window >= itacacpad->maxdata->pad) {
					// We are guaranteed to have at least one sample more than a pad worth of samples past the end of the 
					// trigger window, thus we know there must be a gap sample after these, and can ditch them, though we 
					// need to make sure we aren't flushing any samples from the next trigger window
					// The only time adjust_window != 0 is if you're at the beginning of the window
					g_assert(availablesamps > nongapsamps || itacac->EOS);
					outsamps = nongapsamps > samples_left_in_window ? samples_left_in_window : nongapsamps;
					gst_audioadapter_flush_samples(itacacpad->adapter, outsamps);

					if(!itacacpad->last_gap && itacacpad->adjust_window == 0) {
						// We are at the beginning of the window, and did not just come off a gap, thus the first 
						// pad + max_coinc_window_samps worth of samples we flushed came from the previous window
						g_assert(samples_left_in_window == itacacpad->n);
						samples_left_in_window -= outsamps - itacacpad->maxdata->pad - itacac->max_coinc_window_samps;
					} else 
						samples_left_in_window -= outsamps - itacacpad->adjust_window;

					if(itacacpad->adjust_window > 0)
						itacacpad->adjust_window = 0;

				} else if(nongapsamps == availablesamps && !itacac->EOS) {
					// We have reached the end of available samples, thus there could still be enough nongaps in the next 
					// window for a trigger, so we want to leave a pad worth of samples at the end of this window
					// FIXME this next line is assuming you have enough nongaps to fit into the next window, but it might just be a couple
					g_assert(nongapsamps > samples_left_in_window);
					itacacpad->adjust_window = samples_left_in_window;

					samples_left_in_window = 0;
					itacacpad->last_gap = FALSE;
				} else {
					// We know there is a gap after these, so we can flush these up to the edge of the window
					g_assert(nongapsamps < availablesamps || itacac->EOS);
					outsamps = nongapsamps >= samples_left_in_window ? samples_left_in_window : nongapsamps;
					gst_audioadapter_flush_samples(itacacpad->adapter, outsamps);
					samples_left_in_window -= outsamps;
				}
			}
			// Not enough samples left in the window to produce a trigger or possibly even fill up a pad for a trigger in the next window
			else if(samples_left_in_window <= itacacpad->maxdata->pad) {
				// Need to just zero out samples_left_in_window and set itacacpad->adjust_window for next iteration
				if(samples_left_in_window < itacacpad->maxdata->pad)
					itacacpad->adjust_window = samples_left_in_window;

				samples_left_in_window = 0;
				itacacpad->last_gap = FALSE;
			}
			// Previous window had fewer than maxdata->pad samples + max_coinc_window_samps, which needs to be accounted for when generating a 
			// trigger and flushing samples. This conditional will only ever return TRUE at the beginning of a window since itacacpad->adjust_window is only 
			// set to nonzero values at the end of a window
			else if(itacacpad->adjust_window > itacacpad->maxdata->pad) {
				// This should only ever happen at the beginning of a window, so we use itacacpad->n instead of samples_left_in_window for conditionals
				g_assert(samples_left_in_window == itacacpad->n);
				g_assert(!itacacpad->last_gap);
				copysamps = nongapsamps >= itacacpad->n + itacacpad->adjust_window + itacacpad->maxdata->pad ? itacacpad->n + itacacpad->adjust_window + itacacpad->maxdata->pad : nongapsamps;
				if(nongapsamps >= itacacpad->n + itacacpad->adjust_window + itacacpad->maxdata->pad) {
					// We have enough nongaps to cover this entire trigger window and a pad worth of samples in the next trigger window
					// We want to copy all of the samples up to a pad past the end of the window, and we want to flush 
					// all of the samples up until a (pad+max_coinc_window_samps) worth of samples before the end of the window (leaving samples for a pad in the next window)
					// We want the peak finding length to be the length from the first sample in the window to the last sample in the window.
					// copysamps = itacacpad->n + itacacpad->adjust_window + itacacpad->maxdata->pad
					// outsamps = itacacpad->n + itacacpad->adjust_window - itacacpad->maxdata->pad - max_coinc_window_samps
					// peak_finding_length = itacacpad->n
					outsamps = itacacpad->n + itacacpad->adjust_window - itacacpad->maxdata->pad - itacac->max_coinc_window_samps;
					copy_nongapsamps(itacac, itacacpad, copysamps, itacacpad->n, 0, -1 * (gint) itacacpad->adjust_window);
				} else if(nongapsamps >= itacacpad->n + itacacpad->adjust_window + itacacpad->maxdata->pad - itacac->max_coinc_window_samps) {
					// We have enough nongaps to cover this entire trigger window, but not cover a full pad worth of samples in the next window
					// Because we are guaranteed to have at least a pad worth of samples after this window, we know these samples preceed a gap,
					// but we also know that we have enough of them to provide padding for at least some samples that could produce a trigger
					// coincident with something at the start of the next window
					// We want to copy all of the nongap samples, and we want to flush all of the samples up until a pad before the first sample 
					// that can form a trigger coincident with something in the next window
					// we want the peak finding length to be from the beginning of the trigger window to the last sample that preceeds a pad worth of samples
					// copysamps = nongapsamps
					// outsamps = n + adjust_window - pad - max_coinc_window_samps
					// peak_finding_length = nongapsamps - adjust_window - pad
					g_assert(availablesamps > nongapsamps);
					outsamps = itacacpad->n + itacacpad->adjust_window - itacacpad->maxdata->pad - itacac->max_coinc_window_samps;
					copy_nongapsamps(itacac, itacacpad, copysamps, nongapsamps - itacacpad->adjust_window - itacacpad->maxdata->pad, 0, -1 * (gint) itacacpad->adjust_window);
					// FIXME FIXME FIXME SAVEMORE going to leave this next line until we've confirmed it's not needed
					//itacacpad->last_gap = TRUE;
				} else if(nongapsamps >= itacacpad->n + itacacpad->adjust_window) {
					// We have enough nongaps to cover this entire trigger window, but not cover a full pad worth of samples in the next window
					// Because we are guaranteed to have at least a pad worth of samples after this window, we know these samples preceed a gap,
					// and we know because of the previous conditional that we cant form any triggers which could be coincident with something in the next window
					// We want to copy all of the nongap samples, and we want to flush all of the samples up until the end of the current window
					// we want the peak finding length to be from the beginning of the window to the last sample that preceeds a pad worth of samples
					// copysamps = nongapsamps
					// outsamps = itacacpad->n + itacacpad->adjust_window
					// peak_finding_length = nongapsamps - adjust_window - pad
					g_assert(availablesamps > nongapsamps);
					outsamps = itacacpad->n + itacacpad->adjust_window;
					copy_nongapsamps(itacac, itacacpad, copysamps, nongapsamps - itacacpad->adjust_window - itacacpad->maxdata->pad, 0, -1 * (gint) itacacpad->adjust_window);
					itacacpad->last_gap = TRUE;
				} else {
					// There's a gap in the middle of this window or we've hit EOS
					// We want to copy and flush all of the samples up to the gap
					// We want the peak finding length to be the length from the first sample
					// in the window to the last sample that preceeds a pad worth of samples
					// copysamps = outsamps = nongapsamps
					// peak_finding_length = nongapsamps - adjust_window - pad
					g_assert(availablesamps > nongapsamps || itacac->EOS);
					outsamps = copysamps = nongapsamps;
					copy_nongapsamps(itacac, itacacpad, copysamps, outsamps - itacacpad->adjust_window - itacacpad->maxdata->pad, 0, -1 * (gint) itacacpad->adjust_window);
				}
				gst_audioadapter_flush_samples(itacacpad->adapter, outsamps);
				// FIXME This can put in the conditionals with outsamps and copy_nongapsamps once everything is working
				if(nongapsamps >= itacacpad->n + itacacpad->adjust_window) {
					samples_left_in_window = 0;
					//itacacpad->last_gap = FALSE;
				} else {
					samples_left_in_window -= (outsamps - itacacpad->adjust_window);
				}
				itacacpad->adjust_window = 0;
			}
			// Previous window had pad or fewer samples, meaning we cannot find generate any triggers with samples before the window begins and we 
			// may not have enough samples before the window begins to pad the beginning of the window, which needs to be accounted for when generating 
			// a trigger and flushing samples. This conditional will only ever return TRUE at the beginning of a window since itacacpad->adjust_window 
			// is only set to nonzero values at the end of a window
			else if(itacacpad->adjust_window > 0) {
				// This should only ever happen at the beginning of a window, so we use itacacpad->n instead of samples_left_in_window for conditionals
				g_assert(samples_left_in_window == itacacpad->n);
				g_assert(itacacpad->last_gap == FALSE);
				copysamps = nongapsamps >= itacacpad->n + itacacpad->adjust_window + itacacpad->maxdata->pad ? itacacpad->n + itacacpad->adjust_window + itacacpad->maxdata->pad : nongapsamps;
				if(nongapsamps >= itacacpad->n + itacacpad->adjust_window + itacacpad->maxdata->pad) {
					// We have enough nongaps to cover this entire trigger window and a pad worth of samples in the next trigger window
					// We want to copy all of the samples up to a pad past the end of the window, and we want to flush 
					// all of the samples up until a (pad+max_coinc_window_samps) worth of samples before the end of the window (leaving samples for a pad in the next window)
					// We want the peak finding length to be the length from the first sample after a pad worth of samples to the last sample in the window.
					// copysamps = n + adjust_window + pad
					// outsamps = n + adjust_window - pad - max_coinc_window_samps
					// peak_finding_length = n + adjust_window - itacacpad->maxdata->pad
					outsamps = itacacpad->n + itacacpad->adjust_window - itacacpad->maxdata->pad - itacac->max_coinc_window_samps;
					copy_nongapsamps(itacac, itacacpad, copysamps, itacacpad->n + itacacpad->adjust_window - itacacpad->maxdata->pad, 0, -1 * (gint) itacacpad->adjust_window);
					itacacpad->last_gap = FALSE;
				} else if(nongapsamps >= itacacpad->n + itacacpad->adjust_window + itacacpad->maxdata->pad - itacac->max_coinc_window_samps) {
					// We have enough nongaps to cover this entire trigger window, but not cover a full pad worth of samples in the next window
					// Because we are guaranteed to have at least a pad worth of samples after this window, we know these samples preceed a gap
					// We also know that there's a chance of forming a trigger at the end of this window would could be coincident with one in the next
					// We want to copy all of the nongap samples, and we want to flush up to a pad before the last sample that could form a coincidence
					// we want the peak finding length to be from the first sample after a pad worth of samples to the last sample that preceeds a pad worth of samples
					// copysamps = nongapsamps
					// outsamps = n + adjust_window - pad - max_coinc_window_samps
					// peak_finding_length = n + adjust_window - 2 * pad
					g_assert(availablesamps > nongapsamps);
					outsamps = itacacpad->n + itacacpad->adjust_window - itacacpad->maxdata->pad - itacac->max_coinc_window_samps;
					copy_nongapsamps(itacac, itacacpad, copysamps, itacacpad->n + itacacpad->adjust_window - 2*itacacpad->maxdata->pad, 0, -1 * (gint) itacacpad->adjust_window);
					// FIXME FIXME FIXME SAVEMORE going to leave this next line until we've confirmed it's not needed
					//itacacpad->last_gap = TRUE;
				} else if(nongapsamps >= itacacpad->n + itacacpad->adjust_window) {
					// We have enough nongaps to cover this entire trigger window, but not cover a full pad worth of samples in the next window
					// Because we are guaranteed to have at least a pad worth of samples after this window, we know these samples preceed a gap,
					// and we know there's no chance of forming a coincidence with something at the end of this window 
					// We want to copy all of the nongap samples, and we want to flush all of the samples up until the end of the current window
					// we want the peak finding length to be from the first sample after a pad worth of samples to the last sample that preceeds a pad worth of samples
					// copysamps = nongapsamps
					// outsamps = itacacpad->n + itacacpad->adjust_window
					// peak_finding_length = itacacpad->n + itacacpad->adjust_window - 2 * itacacpad->maxdata->pad = outsamps - 2 * itacacpad->maxdata->pad
					g_assert(availablesamps > nongapsamps);
					outsamps = itacacpad->n + itacacpad->adjust_window;
					copy_nongapsamps(itacac, itacacpad, copysamps, outsamps - 2 * itacacpad->maxdata->pad, 0, -1 * (gint) itacacpad->adjust_window);
					itacacpad->last_gap = TRUE;
				} else {
					// There's a gap in the middle of this window or we've hit EOS
					// We want to copy and flush all of the samples up to the gap
					// We want the peak finding length to be the length from the first sample
					// after a pad worth of samples to the last sample that preceeds a pad worth of samples
					// copysamps = outsamps = nongapsamps
					// peak_finding_length = nongapsamps - 2*itacacpad->maxdata->pad
					g_assert(availablesamps > nongapsamps || itacac->EOS);
					outsamps = copysamps = nongapsamps;
					copy_nongapsamps(itacac, itacacpad, copysamps, outsamps - 2*itacacpad->maxdata->pad, 0, -1 * (gint) itacacpad->adjust_window);
					//itacacpad->last_gap = TRUE;
				}
				gst_audioadapter_flush_samples(itacacpad->adapter, outsamps);
				// FIXME This can put in the conditionals with outsamps and copy_nongapsamps once everything is working
				if(nongapsamps >= itacacpad->n + itacacpad->adjust_window) {
					samples_left_in_window = 0;
					//itacacpad->last_gap = FALSE;
				} else {
					samples_left_in_window -= (outsamps - itacacpad->adjust_window);
				}
				itacacpad->adjust_window = 0;
			}
			// If we've made it this far, we have enough nongap samples to generate a trigger, now we need to check if we're close 
			// enough to the end of the trigger window that we'll be able to save the normal number of samples (which is enough for 
			// padding and the maximum number of coincident window samples)
			else if(samples_left_in_window < itacacpad->maxdata->pad + itacac->max_coinc_window_samps) {
				// Sanity check
				g_assert(itacacpad->last_gap);
				// we have enough samples in the next window to use for padding
				// we already know (from earlier in the if else if chain) that samples_left_in_window > 2pad
				// want to copy all samples up to a pad past the window boundary
				// we dont want to flush any samples because we're saving them for padding and coincident triggers for next trig window
				// want peak finding length to go from a pad into the nongapsamps to the end of the window, so samples_left_in_window - pad
				// samples_left_in_window will be zero after this
				if(nongapsamps >= samples_left_in_window + itacacpad->maxdata->pad) {
					copysamps = samples_left_in_window + itacacpad->maxdata->pad;
					copy_nongapsamps(itacac, itacacpad, copysamps, samples_left_in_window - itacacpad->maxdata->pad, itacacpad->n - samples_left_in_window, (gint) (itacacpad->n - samples_left_in_window));
					itacacpad->adjust_window = samples_left_in_window;
					samples_left_in_window = 0;
					itacacpad->last_gap = FALSE;
				}
				// We dont have enough samples in the next window for padding the final sample in this window
				// We are guaranteed to have samples out to at least a pad past the window boundary (assuming we havent hit EOS), 
				// thus we know a gap is after these nongaps. So we want want to copy all of the nongaps, but we dont want to flush 
				// them yet in case we need them to find coincident triggers in the next trigger window 
				// want peak finding length to go from a pad into the nongapsamps to a pad before the end of its, so nongapsamps - 2*pad
				// samples_left_in_window will be zero after this
				// FIXME NOTE this currently assumes the pad will always be greater than max_coinc_window_samps, if this is not true, 
				// we need at least one more conditional (possibly more?). This else would become else if(nongapsamps >= samples_left_in_window),
				// and we would also need to address the case where samples_left_in_window - nongapsamps < max_coinc_window_samps (i.e. there's a gap 
				// between the last nongap and the end of the trigger window, but at least some of the nongaps are close enough to the trigger window 
				// boundary that they could be coincident with a trigger in the next window)
				else {
					g_assert(availablesamps > nongapsamps || itacac->EOS);
					copysamps = nongapsamps;
					copy_nongapsamps(itacac, itacacpad, copysamps, copysamps - 2*itacacpad->maxdata->pad, itacacpad->n - samples_left_in_window, (gint) (itacacpad->n - samples_left_in_window));
					itacacpad->adjust_window = samples_left_in_window;
					samples_left_in_window = 0;
					// FIXME FIXME FIXME SAVEMORE going to leave this next line until we've confirmed it's not needed
					//itacacpad->last_gap = TRUE;
					itacacpad->last_gap = FALSE;
				}
			}
			// We now know we have enough nongap samples to generate triggers, and we dont need to worry about any corner cases
			else {

				// Possible scenarios
				//
				// REMEMBER that last_gap == FALSE means you're definitely at the beginning of the window (thus will use n instead of samples_left_in_window), 
				// and we have a pad worth of samples from before this window starts (though the negation is not true)
				//
				if(!itacacpad->last_gap) {
					// last_gap == FALSE and nongaps >= samples_left_in_window + 2*pad + max_coinc_window_samps
					// Have a pad worth of samples before this window and after this window
					// want to copy samples_left_in_window + 2* pad
					// Want to flush up to a pad worth of samples before the next window
					// Want peak finding length of samples_left_in_window
					// samples_left_in_window will be zero after this
					if(nongapsamps >= itacacpad->n + 2*itacacpad->maxdata->pad + itacac->max_coinc_window_samps) {
						copysamps = itacacpad->n + 2*itacacpad->maxdata->pad + itacac->max_coinc_window_samps;
						outsamps = itacacpad->n;
						copy_nongapsamps(itacac, itacacpad, copysamps, outsamps, 0, -1 * (gint) (itacacpad->maxdata->pad + itacac->max_coinc_window_samps));
						samples_left_in_window = 0;
					}
					// last_gap == FALSE and nongaps < samples_left_in_window + 2*pad + max_coinc_window_samps but 
					// nongaps > samples_left_in_window + 2pad
					// this means you do not have a full pad worth of samples in the next window before a gap, but we have enough 
					// that a trigger at the end of this window may be coincident with a trigger at the beginning of the next window
					// In this case we want to copy all the nongaps we have
					// We want to flush up to a pad before the last sample that could form a coincidence, which means we just want to 
					// flush a trigger window worth of samples (since we're starting at a point that is a pad+max_coinc_window_samps before the start of the window)
					// The peak finding length will be nongaps - 2*pad - itacac->max_coinc_window_samps
					// samples_left_in_window will be zero after this
					else if(nongapsamps >= itacacpad->n + 2*itacacpad->maxdata->pad) {
						g_assert(availablesamps > nongapsamps || itacac->EOS);
						copysamps = nongapsamps;
						outsamps = itacacpad->n;
						copy_nongapsamps(itacac, itacacpad, copysamps, copysamps - 2*itacacpad->maxdata->pad - itacac->max_coinc_window_samps, 0, -1 * (gint) (itacacpad->maxdata->pad + itacac->max_coinc_window_samps));
						samples_left_in_window = 0;
						// FIXME FIXME FIXME SAVEMORE going to leave this next line until we've confirmed it's not needed
						// itacacpad->last_gap = TRUE;
					}
					// last_gap == FALSE and nongaps < samples_left_in_window + 2*pad but nongaps >= samples_left_in_window + pad + max_coinc_window_samps
					// this means you do not have a full pad worth of samples in the next window, and since we always guaranteed to get at least 
					// a pad full of samples after the window boundary, we know there's a gap there, and because of the previous else if we know 
					// we dont have enough samples after the window to be able to make a trigger at the end of this window that could be coincident 
					// with something in the next window, so we can flush samples up to the window boundary.
					// In this case we want to copy all the nongaps we have
					// We want outsamps to go to the window boundary
					// The peak finding length will be nongaps - 2*pad - itacac->max_coinc_window_samps
					// samples_left_in_window will be zero after this
					else if(nongapsamps >= itacacpad->n + itacacpad->maxdata->pad + itacac->max_coinc_window_samps) {
						g_assert(availablesamps > nongapsamps || itacac->EOS);
						copysamps = nongapsamps;
						outsamps = itacacpad->n + itacacpad->maxdata->pad + itacac->max_coinc_window_samps;
						copy_nongapsamps(itacac, itacacpad, copysamps, copysamps - 2*itacacpad->maxdata->pad - itacac->max_coinc_window_samps, 0, -1 * (gint) (itacacpad->maxdata->pad + itacac->max_coinc_window_samps));
						samples_left_in_window = 0;
						itacacpad->last_gap = TRUE;
					}
					// last_gap == FALSE and nongaps < samples_left_in_window + pad + max_coinc_window_samps
					// This means there is a gap somewhere in this trigger window, so we want to copy and flush up to that gap
					// Peak finding length in this case will be nongaps - 2*pad - max_coinc_window_samps
					// samples_left_in_window -= (nongaps - pad - max_coinc_window_samps)
					// Note that nothing changes if nongaps < n
					// FIXME Note that this assumes the pad is larger than the largest coincidence window, havent thought through 
					// what would happen if this assumption wasnt true
					else {
						copysamps = outsamps = nongapsamps;
						copy_nongapsamps(itacac, itacacpad, copysamps, outsamps - 2*itacacpad->maxdata->pad - itacac->max_coinc_window_samps, 0, -1 * (gint) (itacacpad->maxdata->pad + itacac->max_coinc_window_samps));
						samples_left_in_window -= nongapsamps - itacacpad->maxdata->pad - itacac->max_coinc_window_samps;
					}
				} else {
					// last_gap == TRUE and nongaps >= samples_left_in_window + pad
					// this means we have enough samples in the next window to use for padding
					// we already know (from earlier in the if else if chain) that samples_left_in_window > 2pad
					// want to copy all samples up to a pad past the window boundary
					// want to flush all samples up to (pad+max_coinc_window_samps) before the window boundary
					// want peak finding length to go from a pad into the nongapsamps to the end of the window, so samples_left_in_window - pad
					// samples_left_in_window will be zero after this
					if(nongapsamps >= samples_left_in_window + itacacpad->maxdata->pad) {
						copysamps = samples_left_in_window + itacacpad->maxdata->pad;
						outsamps = samples_left_in_window - itacacpad->maxdata->pad - itacac->max_coinc_window_samps;
						copy_nongapsamps(itacac, itacacpad, copysamps, samples_left_in_window - itacacpad->maxdata->pad, itacacpad->n - samples_left_in_window, (gint) (itacacpad->n - samples_left_in_window));
						samples_left_in_window = 0;
						itacacpad->last_gap = FALSE;
					}
					// last_gap == TRUE and nongaps < samples_left_in_window + pad but nongaps > samples_left_in_window + pad - max_coinc_window_samps
					// We dont have enough samples in the next window for padding the final sample in this window
					// We are guaranteed to have samples out to at least a pad past the window boundary (assuming we havent hit EOS), 
					// thus we know a gap is after these nongaps, but we also know there's enough nongaps to provide padding for a trigger at the end 
					// of this window which could be coincident with a trigger at the start of the next window. 
					// So we want want to copy all of the nongaps, and flush up to a pad before the last sample that could form a trigger which 
					// could be coincident with one in the next window
					// want peak finding length to go from a pad into the nongapsamps to a pad before the end of its, so nongapsamps - 2*pad
					// samples_left_in_window will be zero after this
					else if(nongapsamps >= samples_left_in_window + itacacpad->maxdata->pad - itacac->max_coinc_window_samps) {
						g_assert(availablesamps > nongapsamps || itacac->EOS);
						copysamps = nongapsamps;
						outsamps = samples_left_in_window - itacac->max_coinc_window_samps - itacacpad->maxdata->pad;
						copy_nongapsamps(itacac, itacacpad, copysamps, copysamps - 2*itacacpad->maxdata->pad, itacacpad->n - samples_left_in_window, (gint) (itacacpad->n - samples_left_in_window));
						samples_left_in_window = 0;
						// FIXME FIXME FIXME SAVEMORE going to leave this next line until we've confirmed it's not needed
						// itacacpad->last_gap = TRUE;
					}
					// last_gap == TRUE and nongaps < samples_left_in_window + pad but nongaps >= samples_left_in_window
					// We dont have enough samples in the next window for padding the final sample in this window that could produce 
					// a trigger coincident with something in the next window
					// We are guaranteed to have samples out to at least a pad past the window boundary (assuming we havent hit EOS), 
					// thus we know a gap is after these nongaps. So we want want to copy all of the nongaps, and flush them up to the window boundary
					// want peak finding length to go from a pad into the nongapsamps to a pad before the end of its, so nongapsamps - 2*pad
					// samples_left_in_window will be zero after this
					else if(nongapsamps >= samples_left_in_window) {
						g_assert(availablesamps > nongapsamps || itacac->EOS);
						copysamps = nongapsamps;
						outsamps = samples_left_in_window;
						copy_nongapsamps(itacac, itacacpad, copysamps, copysamps - 2*itacacpad->maxdata->pad, itacacpad->n - samples_left_in_window, (gint) (itacacpad->n - samples_left_in_window));
						samples_left_in_window = 0;
						itacacpad->last_gap = TRUE;
					}
					// last_gap == TRUE and nongaps < samples_left_in_window
					// These nongaps are sandwiched between two gaps
					// want to copy and flush all the nongaps
					// peak finding length will nongaps - 2*pad
					// samples_left_in_window -= nongaps
					// FIXME NOTE this currently assumes the pad will always be greater than max_coinc_window_samps
					else {
						copysamps = outsamps = nongapsamps;
						copy_nongapsamps(itacac, itacacpad, copysamps, outsamps - 2*itacacpad->maxdata->pad, itacacpad->n - samples_left_in_window, (gint) (itacacpad->n - samples_left_in_window));
						samples_left_in_window -= nongapsamps;
						itacacpad->last_gap = FALSE;
					}
				}

				gst_audioadapter_flush_samples(itacacpad->adapter, outsamps);
			}
		}

	// FIXME put some logic here to return GST_FLOW_OK if there were only gaps in the window we just looked at

	guint data_container_index;
	guint peak_finding_start;
	guint duration;
	guint samples_searched_in_window;
	gint trig_offset;
	gboolean triggers_generated;

	for(padlist = element->sinkpads; padlist != NULL; padlist = padlist->next) {
		itacacpad = GSTLAL_ITACAC_PAD(padlist->data);
		data_container_index = 0;
		samples_searched_in_window = 0;
		duration = (guint) gsl_matrix_get(itacacpad->data->duration_dataoffset_trigwindowoffset_peakfindinglength_matrix, 0, 2);

		triggers_generated = FALSE;
		while(duration != 0) {
			// Sanity check
			g_assert(samples_searched_in_window < itacacpad->n - itacacpad->maxdata->pad);

			peak_finding_start = (guint) gsl_matrix_get(itacacpad->data->duration_dataoffset_trigwindowoffset_peakfindinglength_matrix, data_container_index, 0);
			trig_offset = (gint) gsl_matrix_get(itacacpad->data->duration_dataoffset_trigwindowoffset_peakfindinglength_matrix, data_container_index, 1);

			if(trig_offset < -1 * (gint) itacacpad->maxdata->pad) 
				// We have a pad worth of samples before the trigger window and some samples which we may check for a coincident trigger later
				peak_finding_start += (guint) abs(trig_offset);
			else
				// We need to use the first pad worth of samples to compute chisq for potential triggers
				peak_finding_start += itacacpad->maxdata->pad;

			generate_triggers(
				itacac, 
				itacacpad, 
				itacacpad->data->data, 
				peak_finding_start,
				(guint) gsl_matrix_get(itacacpad->data->duration_dataoffset_trigwindowoffset_peakfindinglength_matrix, data_container_index, 3),
				samples_searched_in_window,
				triggers_generated
			);

			if(trig_offset < 0)
				// First trig_offset worth of samples were from previous trigger window
				samples_searched_in_window = (guint) ( (gint) duration + trig_offset );
			else
				samples_searched_in_window = duration + (guint) trig_offset;

			triggers_generated = TRUE;
			duration = (guint) gsl_matrix_get(itacacpad->data->duration_dataoffset_trigwindowoffset_peakfindinglength_matrix, ++data_container_index, 2);
		}

		if(triggers_generated && element->numsinkpads > 1) {
			// FIXME save identifying information about coincident triggers so that we can avoid sending duplicates to python
			find_coincident_triggers
		}

		if(triggers_generated && itacacpad->autocorrelation_matrix) {
			if(srcbuf == NULL) {
				srcbuf = gstlal_snglinspiral_new_buffer_from_peak(itacacpad->maxdata, itacacpad->bankarray, GST_PAD((itacac->aggregator).srcpad), itacac->next_output_offset, itacacpad->n, itacac->next_output_timestamp, itacac->rate, itacacpad->chi2, &(itacacpad->snr_matrix_view), itacac->difftime);
			} else {
				gstlal_snglinspiral_append_peak_to_buffer(srcbuf, itacacpad->maxdata, itacacpad->bankarray, GST_PAD((itacac->aggregator).srcpad), itacac->next_output_offset, itacacpad->n, itacac->next_output_timestamp, itacac->rate, itacacpad->chi2, &(itacacpad->snr_matrix_view));
			}
		} else if(triggers_generated) {
			if(srcbuf == NULL)
				srcbuf = gstlal_snglinspiral_new_buffer_from_peak(itacacpad->maxdata, itacacpad->bankarray, GST_PAD((itacac->aggregator).srcpad), itacac->next_output_offset, itacacpad->n, itacac->next_output_timestamp, itacac->rate, itacacpad->chi2, NULL, itacac->difftime);
			else
				gstlal_snglinspiral_append_peak_to_buffer(srcbuf, itacacpad->maxdata, itacacpad->bankarray, GST_PAD((itacac->aggregator).srcpad), itacac->next_output_offset, itacacpad->n, itacac->next_output_timestamp, itacac->rate, itacacpad->chi2, NULL);
		}

	}


	if(!itacac->EOS) {
		if(srcbuf)
			result = push_buffer(itacac, srcbuf);
		else
			result = push_gap(itacac, itacacpad->n);
	} else {
		guint max_num_samps_left_in_any_pad = 0;
		guint available_samps;
		for(padlist=element->sinkpads; padlist != NULL; padlist = padlist->next) {
			itacacpad = GSTLAL_ITACAC_PAD(padlist->data);
			available_samps = gst_audioadapter_available_samples(itacacpad->adapter);
			max_num_samps_left_in_any_pad = available_samps > max_num_samps_left_in_any_pad ? available_samps : max_num_samps_left_in_any_pad;
		}

		// If there aren't any samples left to process, then we're ready to return GST_FLOW_EOS
		if(max_num_samps_left_in_any_pad > 0)
			result = process(itacac);
		else 
			result = GST_FLOW_EOS;
	}

	return result;
}

static GstFlowReturn aggregate(GstAggregator *aggregator, gboolean timeout)
{
	GSTLALItacac *itacac = GSTLAL_ITACAC(aggregator);
	GSTLALItacacPad *itacacpad;
	GList *padlist;
	GstBuffer *sinkbuf;
	GstFlowReturn result;


	// Calculate the coincidence windows and make sure the pads caps are compatible with each other if we're just starting
	if(itacac->rate == 0) {
		result = final_setup(itacac);
		return result;
	}

	if(itacac->EOS) {
		result = process(itacac);
		return result;
	}
		

	// FIXME need to confirm the aggregator does enough checks that the
	// checks itac does are unncessary
	for(padlist = GST_ELEMENT(aggregator)->sinkpads; padlist != NULL; padlist = padlist->next) {
		// Get the buffer from the pad we're looking at and assert it
		// has a valid timestamp
		itacacpad = GSTLAL_ITACAC_PAD(padlist->data);
		sinkbuf = gst_aggregator_pad_pop_buffer(GST_AGGREGATOR_PAD(itacacpad));
		g_assert(GST_BUFFER_PTS_IS_VALID(sinkbuf));

		// FIXME Is this necessary/did I understand what this does correctly?
		// Sync up the properties that may have changed, do this before
		// accessing any of the pad's properties
		gst_object_sync_values(GST_OBJECT(itacacpad), GST_BUFFER_PTS(sinkbuf));

		if(!GST_BUFFER_PTS_IS_VALID(sinkbuf) || !GST_BUFFER_DURATION_IS_VALID(sinkbuf) || !GST_BUFFER_OFFSET_IS_VALID(sinkbuf) || !GST_BUFFER_OFFSET_END_IS_VALID(sinkbuf)) {
			gst_buffer_unref(sinkbuf);
			GST_ERROR_OBJECT(GST_ELEMENT(aggregator), "error in input stream: buffer has invalid timestamp and/or offset");
			result = GST_FLOW_ERROR;
			return result;
		}

		// Check for instrument and channel name tags
		if(!itacacpad->instrument || !itacacpad->channel_name) {
			GST_ELEMENT_ERROR(itacacpad, STREAM, FAILED, ("missing or invalid tags"), ("instrument and/or channel name not known (stream's tags must provide this information)"));
			result = GST_FLOW_ERROR;
			return result;
		}

		if(!itacacpad->bankarray) {
			GST_ELEMENT_ERROR(itacacpad, STREAM, FAILED, ("missing bank file"), ("must have a valid template bank to create events"));
			result = GST_FLOW_ERROR;
			return result;
		}

		// FIXME if we were more careful we wouldn't lose so much data around disconts
		// FIXME I don't think this logic works for itacac, it came from itac, need to think carefully about what to do around disconts
		if (GST_BUFFER_FLAG_IS_SET(sinkbuf, GST_BUFFER_FLAG_DISCONT)) {
			reset_time_and_offset(itacac);
			gst_audioadapter_clear(itacacpad->adapter);
		}

		// If we dont have a valid first timestamp yet take this one
		// The aggregator keeps everything in sync, so it should be
		// fine to just take this one
		// FIXME This probably doesnt work
		if(itacac->next_output_timestamp == GST_CLOCK_TIME_NONE) {
			itacac->next_output_timestamp = GST_BUFFER_PTS(sinkbuf);
		}

		// put the incoming buffer into an adapter, handles gaps 
		// FIXME the aggregator does have some logic to deal with gaps,
		// should see if we can use some built-in freatures of the
		// aggregator instead of the audioadapter
		gst_audioadapter_push(itacacpad->adapter, sinkbuf);
	}

	result = process(itacac);

	return result;
}


/*
 * ============================================================================
 *
 *                                Type Support FIXME Is this appropriately named?
 *
 * ============================================================================
 */


/*
 * Instance finalize function.  See ???
 */

static void gstlal_itacac_pad_dispose(GObject *object)
{
	GSTLALItacacPad *itacacpad = GSTLAL_ITACAC_PAD(object);
	GList *glist;

	gst_audioadapter_clear(itacacpad->adapter);
	g_object_unref(itacacpad->adapter);

	if (itacacpad->bankarray)
		free_bank(itacacpad);

	if (itacacpad->instrument){
		free(itacacpad->instrument);
		itacacpad->instrument = NULL;
	}

	if(itacacpad->channel_name){
		free(itacacpad->channel_name);
		itacacpad->channel_name = NULL;
	}

	if(itacacpad->maxdata) {
		gstlal_peak_state_free(itacacpad->maxdata);
		itacacpad->maxdata = NULL;
	}

	if(itacacpad->tmp_maxdata) {
		gstlal_peak_state_free(itacacpad->tmp_maxdata);
		itacacpad->tmp_maxdata = NULL;
	}

	if(itacacpad->data->data) {
		free(itacacpad->data->data);
		itacacpad->data->data = NULL;
	}

	if(itacacpad->snr_mat) {
		free(itacacpad->snr_mat);
		itacacpad->snr_mat = NULL;
	}

	if(itacacpad->tmp_snr_mat) {
		free(itacacpad->tmp_snr_mat);
		itacacpad->tmp_snr_mat = NULL;
	}

	if(itacacpad->autocorrelation_matrix) {
		free(itacacpad->autocorrelation_matrix);
		itacacpad->autocorrelation_matrix = NULL;
	}

	if(itacacpad->autocorrelation_mask) {
		free(itacacpad->autocorrelation_mask);
		itacacpad->autocorrelation_mask = NULL;
	}

	if(itacacpad->autocorrelation_norm) {
		free(itacacpad->autocorrelation_norm);
		itacacpad->autocorrelation_norm = NULL;
	}

	if(itacacpad->chi2) {
		free(itacacpad->chi2);
		itacacpad->chi2 = NULL;
	}

	if(itacacpad->tmp_chi2) {
		free(itacacpad->tmp_chi2);
		itacacpad->tmp_chi2 = NULL;
	}

	G_OBJECT_CLASS(gstlal_itacac_pad_parent_class)->dispose(object);
}

static void gstlal_itacac_finalize(GObject *object)
{
	GSTLALItacac *itacac = GSTLAL_ITACAC(object);

	g_hash_table_destroy(itacac->coinc_window_hashtable);

	G_OBJECT_CLASS(gstlal_itacac_parent_class)->finalize(object);
}

/*
 * Class init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GClassInitFunc
 */


static void gstlal_itacac_pad_class_init(GSTLALItacacPadClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(gstlal_itacac_pad_set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(gstlal_itacac_pad_get_property);
	// Right now no memory is allocated in the class instance structure for GSTLALItacacPads, so we dont need a custom finalize function
	// If anything is added to the class structure, we will need a custom finalize function that chains up to the AggregatorPad's finalize function
	gobject_class->dispose = GST_DEBUG_FUNCPTR(gstlal_itacac_pad_dispose);

	//
	// Properties
	//

        g_object_class_install_property(
		gobject_class,
		ARG_N,
		g_param_spec_uint(
			"n",
			"n",
			"number of samples over which to identify itacs",
			0, G_MAXUINT, 0,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);

	g_object_class_install_property(
		gobject_class,
		ARG_BANK_FILENAME,
		g_param_spec_string(
			"bank-filename",
			"Bank file name",
			"Path to XML file used to generate the template bank.  Setting this property resets sigmasq to a vector of 0s.",
			NULL,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
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
		ARG_SIGMASQ,
		g_param_spec_value_array(
			"sigmasq",
			"\\sigma^{2} factors",
			"Vector of \\sigma^{2} factors.  The effective distance of a trigger is \\sqrt{sigma^{2}} / SNR.",
			g_param_spec_double(
				"sigmasq",
				"\\sigma^{2}",
				"\\sigma^{2} factor",
				-G_MAXDOUBLE, G_MAXDOUBLE, 0.0,
				G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
			),
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_CONTROLLABLE
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

}

static void gstlal_itacac_class_init(GSTLALItacacClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GstAggregatorClass *aggregator_class = GST_AGGREGATOR_CLASS(klass); 

	gst_element_class_set_metadata(
		element_class,
		"Itacac",
		"Filter",
		"Find coincident inspiral triggers in snr streams from multiple detectors",
		"Cody Messick <cody.messick@ligo.org>"
	);

	//
	// Our custom functions
	//

	aggregator_class->aggregate = GST_DEBUG_FUNCPTR(aggregate);
	aggregator_class->sink_event = GST_DEBUG_FUNCPTR(sink_event);
	gobject_class->set_property = GST_DEBUG_FUNCPTR(gstlal_itacac_set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(gstlal_itacac_get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(gstlal_itacac_finalize);


	//
	// static pad templates
	//

	gst_element_class_add_static_pad_template_with_gtype(
		element_class,
		&sink_templ,
		GSTLAL_ITACAC_PAD_TYPE
	);

	gst_element_class_add_static_pad_template_with_gtype(
		element_class,
		&src_templ,
		GST_TYPE_AGGREGATOR_PAD
	);

	//
	// Properties
	//

        g_object_class_install_property(
		gobject_class,
		ARG_COINC_THRESH,
		g_param_spec_double(
			"coinc-thresh",
			"Coincidence Threshold",
			"Time, in milliseconds, added to light-travel time between detectors to define the coincidence window.",
			0, G_MAXDOUBLE, DEFAULT_COINC_THRESH,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);

}


/*
 * Instance init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GInstanceInitFunc
 */

static void gstlal_itacac_pad_init(GSTLALItacacPad *itacacpad)
{
	itacacpad->adapter = g_object_new(GST_TYPE_AUDIOADAPTER, NULL);
	itacacpad->rate = 0;
	itacacpad->channels = 0;
	itacacpad->data->data = NULL;
	itacacpad->chi2 = NULL;
	itacacpad->tmp_chi2 = NULL;
	itacacpad->bank_filename = NULL;
	itacacpad->instrument = NULL;
	itacacpad->channel_name = NULL;
	itacacpad->difftime = 0;
	itacacpad->snr_thresh = 0;
	g_mutex_init(&itacacpad->bank_lock);

	itacacpad->autocorrelation_matrix = NULL;
	itacacpad->autocorrelation_mask = NULL;
	itacacpad->autocorrelation_norm = NULL;
	itacacpad->snr_mat = NULL;
	itacacpad->tmp_snr_mat = NULL;
	itacacpad->bankarray = NULL;
	itacacpad->last_gap = TRUE;

	itacacpad->adjust_window = 0;

	// Coincidence stuff
	itacacpad->next_in_coinc_order = NULL;
	itacacpad->saved_data->data = NULL;
	itacacpad->saved_data->samples_before_data_begin = NULL;
	itacacpad->saved_data->saved_samples_per_channel = NULL;
	itacacpad->saved_data->next_start = 0;

	gst_pad_use_fixed_caps(GST_PAD(itacacpad));

}

static void gstlal_itacac_init(GSTLALItacac *itacac)
{
	itacac->rate = 0;
	itacac->channels = 0;

	itacac->difftime = 0;
	itacac->coinc_window_hashtable = g_hash_table_new(g_str_hash, g_str_equal);
	itacac->max_coinc_window_samps = 0;
	itacac->trigger_end = NULL;
	
	reset_time_and_offset(itacac);

	itacac->EOS = FALSE;

}

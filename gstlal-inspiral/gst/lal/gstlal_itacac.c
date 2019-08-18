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


#include <complex.h>
#include <glib.h>
#include <gmodule.h>
#include <gst/gst.h>
#include <gst/base/gstaggregator.h>
#include <gst/controller/controller.h>
#include <math.h>
#include <string.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_blas.h>

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
	return gstlal_autocorrelation_chi2_autocorrelation_length(itacacpad->autocorrelation_matrix);
}

static guint64 output_num_samps(GSTLALItacacPad *itacacpad) {
	return (guint64) itacacpad->n;
}

static guint64 output_num_bytes(GSTLALItacacPad *itacacpad) {
	return (guint64) output_num_samps(itacacpad) * itacacpad->adapter->unit_size;
}

static int reset_time_and_offset(GSTLALItacac *itacac) {
	// NOTE This should only get called when itacac is first starting up,
	// there is an assert that guarantees this
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
	gstlal_snglinspiral_array_free(itacacpad->bankarray);
	itacacpad->bankarray = NULL;
}

static void update_peak_info_from_autocorrelation_properties(GSTLALItacacPad *itacacpad) {
	// FIXME Need to make sure that itacac can run without autocorrelation matrix
	if(itacacpad->maxdata && itacacpad->tmp_maxdata && itacacpad->autocorrelation_matrix) {
		itacacpad->maxdata->pad = itacacpad->tmp_maxdata->pad = autocorrelation_length(itacacpad) / 2;
		free(itacacpad->snr_mat);
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

static gboolean get_instrument_pad_pointer(GSTLALItacac *itacac, GSTLALItacacPad *itacacpad)
{
	gboolean result = TRUE;
	if(strcmp(itacacpad->instrument, "H1") == 0)
		itacac->H1_itacacpad = itacacpad;
	else if(strcmp(itacacpad->instrument, "K1") == 0)
		itacac->K1_itacacpad = itacacpad;
	else if(strcmp(itacacpad->instrument, "L1") == 0)
		itacac->L1_itacacpad = itacacpad;
	else if(strcmp(itacacpad->instrument, "V1") == 0)
		itacac->V1_itacacpad = itacacpad;
	else {
		GST_ERROR_OBJECT(itacacpad, "pad has unknown detector, %s", itacacpad->instrument);
		result = FALSE;
	}
	return result;
}

static gboolean setcaps(GstAggregator *agg, GstAggregatorPad *aggpad, GstEvent *event) {
	GSTLALItacac *itacac = GSTLAL_ITACAC(agg);
	GSTLALItacacPad *itacacpad = GSTLAL_ITACAC_PAD(aggpad);
	GstCaps *caps;
	guint width = 0;
	gstlal_peak_type_specifier peak_type = 0;
	gint rate;

	//
	// Update element metadata
	//
	gst_event_parse_caps(event, &caps);
	GstStructure *str = gst_caps_get_structure(caps, 0);
	const gchar *format = gst_structure_get_string(str, "format");
	gst_structure_get_int(str, "rate", &rate);

	if(!strcmp(format, GST_AUDIO_NE(Z64))) {
		width = sizeof(float complex);
		peak_type = GSTLAL_PEAK_COMPLEX;
	} else if(!strcmp(format, GST_AUDIO_NE(Z128))) {
		width = sizeof(double complex);
		peak_type = GSTLAL_PEAK_DOUBLE_COMPLEX;
	} else
		GST_ERROR_OBJECT(itacac, "unsupported format %s", format);

	g_mutex_lock(&itacac->caps_lock);
	if(itacac->rate == 0) {
		itacac->rate = (guint) rate;
		itacac->peak_type = peak_type;
	} else {
		g_assert(itacac->rate == (guint) rate);
		g_assert(itacac->peak_type == peak_type);
	}
	g_mutex_unlock(&itacac->caps_lock);

	g_object_set(itacacpad->adapter, "unit-size", itacac->channels * width, NULL);
	itacacpad->chi2 = calloc(itacac->channels, width);
	itacacpad->tmp_chi2 = calloc(itacac->channels, width);

	if(itacacpad->maxdata)
		gstlal_peak_state_free(itacacpad->maxdata);
	
	if(itacacpad->tmp_maxdata)
		gstlal_peak_state_free(itacacpad->tmp_maxdata);

	itacacpad->maxdata = gstlal_peak_state_new(itacac->channels, itacac->peak_type);
	itacacpad->tmp_maxdata = gstlal_peak_state_new(itacac->channels, itacac->peak_type);

	// This should be called any time the autocorrelation property is updated 
	update_peak_info_from_autocorrelation_properties(itacacpad);

	// Set up data_container struct
	// FIXME Can simplify this process by reworking audioadapter to provide
	// the information currently contained duration_..._matrix
	itacacpad->data->data = g_malloc(output_num_bytes(itacacpad) + itacacpad->adapter->unit_size * (2 * itacacpad->maxdata->pad));
	// The largest number of disjoint sets of non-gap-samples (large enough
	// to produce a trigger) that we could have in a given trigger window
	guint max_number_disjoint_sets_in_trigger_window = itacac->rate / (2 * itacacpad->maxdata->pad) + 1;
	itacacpad->data->duration_dataoffset_trigwindowoffset_peakfindinglength_matrix = gsl_matrix_calloc(max_number_disjoint_sets_in_trigger_window, 4);

	return GST_AGGREGATOR_CLASS(gstlal_itacac_parent_class)->sink_event(agg, aggpad, event);
}

static gboolean sink_event(GstAggregator *agg, GstAggregatorPad *aggpad, GstEvent *event)
{
	GSTLALItacac *itacac = GSTLAL_ITACAC(agg);
	GSTLALItacacPad *itacacpad = GSTLAL_ITACAC_PAD(aggpad);
	gboolean result = TRUE;
	GList *padlist;

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
				gstlal_set_channel_in_snglinspiral_array(itacacpad->bankarray, (int) itacac->channels, itacacpad->channel_name);
				gstlal_set_instrument_in_snglinspiral_array(itacacpad->bankarray, (int) itacac->channels, itacacpad->instrument);
				g_mutex_unlock(&itacacpad->bank_lock);
				result &= get_instrument_pad_pointer(itacac, itacacpad);
			}
                        break;

		}
		case GST_EVENT_EOS:
		{
			itacacpad->EOS = TRUE;
			itacac->EOS = TRUE;
			for(padlist = GST_ELEMENT(agg)->sinkpads; padlist != NULL; padlist = padlist->next)
				itacac->EOS = GSTLAL_ITACAC_PAD(padlist->data)->EOS && itacac->EOS;
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
 *                           Properties and Meta
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

static void gstlal_itacac_pad_set_property(GObject *object, enum padproperty id, const GValue *value, GParamSpec *pspec)
{
	GSTLALItacac *itacac = GSTLAL_ITACAC(gst_pad_get_parent(GST_PAD(object)));
	GSTLALItacacPad *itacacpad = GSTLAL_ITACAC_PAD(object);
	guint channels;
	GstClockTimeDiff difftime;

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
		free_bank(itacacpad);
		itacacpad->bank_filename = g_value_dup_string(value);
		channels = gstlal_snglinspiral_array_from_file(itacacpad->bank_filename, &(itacacpad->bankarray));
		gstlal_set_min_offset_in_snglinspiral_array(itacacpad->bankarray, (gint) channels, &difftime);
		if(itacacpad->instrument && itacacpad->channel_name) {
			gstlal_set_instrument_in_snglinspiral_array(itacacpad->bankarray, (gint) channels, itacacpad->instrument);
			gstlal_set_channel_in_snglinspiral_array(itacacpad->bankarray, (gint) channels, itacacpad->channel_name);
		}
		g_mutex_unlock(&itacacpad->bank_lock);

		g_assert(itacac != NULL);
		g_mutex_lock(&itacac->caps_lock);
		if(itacac->channels == 0) {
			itacac->channels = channels;
			itacac->difftime = difftime;
		} else {
			g_assert(itacac->channels == channels);
			g_assert(itacac->difftime == difftime);
		}
		g_mutex_unlock(&itacac->caps_lock);
		break;

	case ARG_SIGMASQ:
		g_mutex_lock(&itacacpad->bank_lock);
		if(itacacpad->bankarray) {
			g_assert(itacac != NULL);
			gint length;
			double *sigmasq = gstlal_doubles_from_g_value_array(g_value_get_boxed(value), NULL, &length);
			if((gint) itacac->channels != length)
				GST_ERROR_OBJECT(itacacpad, "vector length (%d) does not match number of templates (%u)", length, itacac->channels);
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
	if(itacac != NULL)
		gst_object_unref(GST_OBJECT(itacac));
}

static void gstlal_itacac_pad_get_property(GObject *object, enum padproperty id, GValue *value, GParamSpec *pspec)
{
	GSTLALItacac *itacac = GSTLAL_ITACAC(gst_pad_get_parent(GST_PAD(object)));
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
			g_assert(itacac != NULL);
			double sigmasq[itacac->channels];
			gint i;
			for(i = 0; i < (gint) itacac->channels; i++)
				sigmasq[i] = itacacpad->bankarray[i].sigmasq;
			g_value_take_boxed(value, gstlal_g_value_array_from_doubles(sigmasq, (gint) itacac->channels));
		} else {
			GST_WARNING_OBJECT(itacacpad, "no template bank");
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
			g_value_take_boxed(value, g_array_sized_new(TRUE, TRUE, sizeof(double), 0));
		}
		g_mutex_unlock(&itacacpad->bank_lock);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(itacacpad);
	if(itacac != NULL)
		gst_object_unref(GST_OBJECT(itacac));
}

/*
 * aggregate()
*/ 

static void update_state(GSTLALItacac *itacac, GstBuffer *srcbuf) {
	itacac->next_output_offset = GST_BUFFER_OFFSET_END(srcbuf);
	itacac->next_output_timestamp = GST_BUFFER_PTS(srcbuf) - itacac->difftime;
	itacac->next_output_timestamp += GST_BUFFER_DURATION(srcbuf);
}

static GstFlowReturn push_buffer(GstAggregator *agg, GstBuffer *srcbuf) {
	GSTLALItacac *itacac = GSTLAL_ITACAC(agg);
	update_state(itacac, srcbuf);

	GST_DEBUG_OBJECT(itacac, "pushing %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(srcbuf));
	return GST_AGGREGATOR_CLASS(gstlal_itacac_parent_class)->finish_buffer(agg, srcbuf);
}

static GstFlowReturn push_gap(GSTLALItacac *itacac, guint samps) {
	GstBuffer *srcbuf = gst_buffer_new();
	GST_BUFFER_FLAG_SET(srcbuf, GST_BUFFER_FLAG_GAP);

	GST_BUFFER_OFFSET(srcbuf) = itacac->next_output_offset;
	GST_BUFFER_OFFSET_END(srcbuf) = itacac->next_output_offset + samps;
	GST_BUFFER_PTS(srcbuf) = itacac->next_output_timestamp + itacac->difftime;
	GST_BUFFER_DURATION(srcbuf) = (GstClockTime) gst_util_uint64_scale_int_round(GST_SECOND, samps, (gint) itacac->rate);

	return gst_aggregator_finish_buffer(GST_AGGREGATOR(itacac), srcbuf);

}

static void copy_nongapsamps(GSTLALItacac *itacac, GSTLALItacacPad *itacacpad, guint copysamps, guint peak_finding_length, gint offset_from_trigwindow) {
	guint data_container_index = 0;
	guint offset_from_copied_data = 0;
	guint duration = (guint) gsl_matrix_get(itacacpad->data->duration_dataoffset_trigwindowoffset_peakfindinglength_matrix, 0, 0);

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
        if(itacac->peak_type == GSTLAL_PEAK_COMPLEX)
		gst_audioadapter_copy_samples(itacacpad->adapter, (float complex *) itacacpad->data->data + offset_from_copied_data * itacacpad->maxdata->channels, copysamps, NULL, NULL);
        else if(itacac->peak_type == GSTLAL_PEAK_DOUBLE_COMPLEX)
		gst_audioadapter_copy_samples(itacacpad->adapter, (double complex *) itacacpad->data->data + offset_from_copied_data * itacacpad->maxdata->channels, copysamps, NULL, NULL);
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

        if(itacac->peak_type == GSTLAL_PEAK_COMPLEX) {
                // Find the peak, making sure to put the data pointer at the start of the interval we care about
                gstlal_float_complex_peak_over_window_interp(this_maxdata, (float complex *) itacacpad->data->data + peak_finding_start * this_maxdata->channels, peak_finding_length);
		//FIXME At the moment, empty triggers are added to inform the
		//"how many instruments were on test", the correct thing to do
		//is probably to add metadata to the buffer containing
		//information about which instruments were on
		if(this_maxdata->no_peaks_past_threshold) {
			for(channel = 0; channel < this_maxdata->channels; channel++) {
				if(cabs((double complex) (this_maxdata->interpvalues).as_float_complex[channel]) > 0) {
					this_maxdata->no_peaks_past_threshold = FALSE;
					break;
				}
			}
		}
	}
        else if(itacac->peak_type == GSTLAL_PEAK_DOUBLE_COMPLEX) {
                // Find the peak, making sure to put the data pointer at the start of the interval we care about
                gstlal_double_complex_peak_over_window_interp(this_maxdata, (double complex *) itacacpad->data->data + peak_finding_start * this_maxdata->channels, peak_finding_length);
		//FIXME At the moment, empty triggers are added to inform the
		//"how many instruments were on test", the correct thing to do
		//is probably to add metadata to the buffer containing
		//information about which instruments were on
		if(this_maxdata->no_peaks_past_threshold) {
			for(channel = 0; channel < this_maxdata->channels; channel++) {
				if(cabs((this_maxdata->interpvalues).as_double_complex[channel]) > 0) {
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
			gstlal_double_complex_series_around_peak(this_maxdata, (double complex *) itacacpad->data->data + peak_finding_start * this_maxdata->channels, (double complex *) this_snr_mat, this_maxdata->pad);
			gstlal_autocorrelation_chi2((double *) this_chi2, (double complex *) this_snr_mat, autocorrelation_length(itacacpad), -((int) autocorrelation_length(itacacpad)) / 2, itacacpad->snr_thresh, itacacpad->autocorrelation_matrix, itacacpad->autocorrelation_mask, itacacpad->autocorrelation_norm);

		} else if(itacac->peak_type == GSTLAL_PEAK_COMPLEX) {
			/* extract data around peak for chisq calculation */
			gstlal_float_complex_series_around_peak(this_maxdata, (float complex *) itacacpad->data->data + peak_finding_start * this_maxdata->channels, (float complex *) this_snr_mat, this_maxdata->pad);
			gstlal_autocorrelation_chi2_float((float *) this_chi2, (float complex *) this_snr_mat, autocorrelation_length(itacacpad), -((int) autocorrelation_length(itacacpad)) / 2, itacacpad->snr_thresh, itacacpad->autocorrelation_matrix, itacacpad->autocorrelation_mask, itacacpad->autocorrelation_norm);
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
				new_snr = cabs( (double complex) (itacacpad->tmp_maxdata->interpvalues).as_double_complex[channel]);

				if(new_snr > old_snr) {
					// The previous peak found was larger than the current peak. If there was a peak before but not now, increment itacacpad->maxdata's num_events
					if(old_snr == 0)
						// FIXME confirm that this isnt affected by floating point error
						itacacpad->maxdata->num_events++;

					(itacacpad->maxdata->values).as_double_complex[channel] = (itacacpad->tmp_maxdata->values).as_double_complex[channel];
					(itacacpad->maxdata->interpvalues).as_double_complex[channel] = (itacacpad->tmp_maxdata->interpvalues).as_double_complex[channel];
					itacacpad->maxdata->samples[channel] = itacacpad->tmp_maxdata->samples[channel];
					itacacpad->maxdata->interpsamples[channel] = itacacpad->tmp_maxdata->interpsamples[channel];
					old_chi2[channel] = new_chi2[channel];

					if(itacacpad->autocorrelation_matrix) {
						// Replace the snr time series around the peak with the new one
						gsl_vector_complex_float_view old_snr_vector_view = gsl_matrix_complex_float_column(&(itacacpad->snr_matrix_view.matrix), channel);
						gsl_vector_complex_float_view new_snr_vector_view = gsl_matrix_complex_float_column(&(itacacpad->tmp_snr_matrix_view.matrix), channel);
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
				new_snr = cabs( (double complex) (itacacpad->tmp_maxdata->interpvalues).as_float_complex[channel]);
				if(new_snr > old_snr) {
					// The new peak found is larger than the previous peak. If there was a peak before but not now, increment itacacpad->maxdata's num_events
					if(old_snr == 0)
						// FIXME confirm that this isnt affected by floating point error
						itacacpad->maxdata->num_events++;

					(itacacpad->maxdata->values).as_float_complex[channel] = (itacacpad->tmp_maxdata->values).as_float_complex[channel];
					(itacacpad->maxdata->interpvalues).as_float_complex[channel] = (itacacpad->tmp_maxdata->interpvalues).as_float_complex[channel];
					itacacpad->maxdata->samples[channel] = itacacpad->tmp_maxdata->samples[channel];
					itacacpad->maxdata->interpsamples[channel] = itacacpad->tmp_maxdata->interpsamples[channel];
					old_chi2[channel] = new_chi2[channel];

					if(itacacpad->autocorrelation_matrix) {
						// Replace the snr time series around the peak with the new one
						gsl_vector_complex_float_view old_snr_vector_view = gsl_matrix_complex_float_column(&(itacacpad->snr_matrix_view.matrix), channel);
						gsl_vector_complex_float_view new_snr_vector_view = gsl_matrix_complex_float_column(&(itacacpad->tmp_snr_matrix_view.matrix), channel);
						old_snr_vector_view.vector = new_snr_vector_view.vector;
					}
				}
			}
		}

		itacacpad->maxdata->no_peaks_past_threshold = itacacpad->tmp_maxdata->no_peaks_past_threshold;
		gstlal_peak_state_clear(itacacpad->tmp_maxdata);
		itacacpad->tmp_maxdata->no_peaks_past_threshold = TRUE;
	}
}

static void get_snr_series(GSTLALItacac *itacac, GSTLALItacacPad *itacacpad, guint channel, guint peak_sample) {
	guint data_container_index, nongapsamps_duration, data_start_in_trigwindow, data_index, snr_index, series_start;
	gint trig_window_offset;
	double complex *tmp_snr_mat_doubleptr;
	float complex *tmp_snr_mat_floatptr;

	for(data_container_index = 0; data_container_index < itacacpad->data->duration_dataoffset_trigwindowoffset_peakfindinglength_matrix->size1; data_container_index++) {
		nongapsamps_duration = (guint) gsl_matrix_get(itacacpad->data->duration_dataoffset_trigwindowoffset_peakfindinglength_matrix, data_container_index, 0);

		// Check if we've looked at all of the available nongaps
		if(nongapsamps_duration == 0)
			break;

		// Get the start and stop sample indices for this set of nongaps
		// data_index describes where to start in the block of nongaps we have
		// data_start_in_trigwindow is how many samples came before the beginning of the data we care about in the current trigger window
		// notice that data_index and data_start_in_trigwindow give locations of the *same* sample point, just using different metrics
		// Thus if you subtract data_start_in_trigwindow from peak_sample, then add that to data_index, you'll find the location of your peak
		data_index = (guint) gsl_matrix_get(itacacpad->data->duration_dataoffset_trigwindowoffset_peakfindinglength_matrix, data_container_index, 1);
		trig_window_offset = (gint) gsl_matrix_get(itacacpad->data->duration_dataoffset_trigwindowoffset_peakfindinglength_matrix, data_container_index, 2);

		data_index += itacacpad->maxdata->pad;

		if(trig_window_offset < 0)
			data_start_in_trigwindow = itacacpad->maxdata->pad - (guint) abs(trig_window_offset);
		 else
			data_start_in_trigwindow = (guint) trig_window_offset + itacacpad->maxdata->pad;

		// Check if the samples we care about are in this set of nongaps
		// FIXME Can get time series out more often if we zero pad whatever existing time series when its not long enough
		if(peak_sample < data_start_in_trigwindow)
			break;
		if(peak_sample >= data_start_in_trigwindow + (guint) gsl_matrix_get(itacacpad->data->duration_dataoffset_trigwindowoffset_peakfindinglength_matrix, data_container_index, 3))
			continue;

		// notice that data_index and data_start_in_trigwindow give locations of the *same* sample point, just using different metrics
		// Thus if you subtract data_start_in_trigwindow from peak_sample, then add that to data_index, you'll find the location of your peak
		series_start = data_index + (peak_sample - data_start_in_trigwindow) - itacacpad->maxdata->pad;

		// find snr time series
		// FIXME Should the gstlal*series_around_peak functions be generalized so that they can do this?
		if(itacac->peak_type == GSTLAL_PEAK_DOUBLE_COMPLEX) {
			tmp_snr_mat_doubleptr = (double complex *) itacacpad->tmp_snr_mat;
		} else if(itacac->peak_type == GSTLAL_PEAK_COMPLEX) {
			tmp_snr_mat_floatptr = (float complex *) itacacpad->tmp_snr_mat;
		}

		for(snr_index = 0; snr_index < 2*itacacpad->maxdata->pad + 1; snr_index++) {

			if(itacac->peak_type == GSTLAL_PEAK_DOUBLE_COMPLEX)
				tmp_snr_mat_doubleptr[snr_index * itacacpad->maxdata->channels + channel] = *(((double complex *) itacacpad->data->data) + (series_start + snr_index) * itacacpad->maxdata->channels + channel);

			else if(itacac->peak_type == GSTLAL_PEAK_COMPLEX)
				tmp_snr_mat_floatptr[snr_index * itacacpad->maxdata->channels + channel] = *(((float complex *) itacacpad->data->data) + (series_start + snr_index) * itacacpad->maxdata->channels + channel);
		}

		// If we reached this point, we have what we need
		break;
	}
}

static void populate_snr_in_other_detectors(GSTLALItacac *itacac, GSTLALItacacPad *itacacpad) {
	guint channel, peak_sample;

	// First zero the tmp_snr_mat objects in the other pads
	if(strcmp(itacacpad->instrument, "H1") != 0 && itacac->H1_itacacpad != NULL && !itacac->H1_itacacpad->waiting)
		memset(itacac->H1_itacacpad->tmp_snr_mat, 0, autocorrelation_channels(itacacpad) * autocorrelation_length(itacacpad) * itacacpad->maxdata->unit);
	if(strcmp(itacacpad->instrument, "K1") != 0 && itacac->K1_itacacpad != NULL && !itacac->K1_itacacpad->waiting)
		memset(itacac->K1_itacacpad->tmp_snr_mat, 0, autocorrelation_channels(itacacpad) * autocorrelation_length(itacacpad) * itacacpad->maxdata->unit);
	if(strcmp(itacacpad->instrument, "L1") != 0 && itacac->L1_itacacpad != NULL && !itacac->L1_itacacpad->waiting)
		memset(itacac->L1_itacacpad->tmp_snr_mat, 0, autocorrelation_channels(itacacpad) * autocorrelation_length(itacacpad) * itacacpad->maxdata->unit);
	if(strcmp(itacacpad->instrument, "V1") != 0 && itacac->V1_itacacpad != NULL && !itacac->V1_itacacpad->waiting)
		memset(itacac->V1_itacacpad->tmp_snr_mat, 0, autocorrelation_channels(itacacpad) * autocorrelation_length(itacacpad) * itacacpad->maxdata->unit);

	for(channel = 0; channel < itacacpad->maxdata->channels; channel++) {
		// Identify which sample was the peak
		// See if we that have that sample in the other ifos
		// zeropad other ifos if we need to
		if(itacac->peak_type == GSTLAL_PEAK_DOUBLE_COMPLEX) {
			// First check if there's a trigger
			if(!itacacpad->maxdata->values.as_double_complex[channel])
				continue;
		} else if(itacac->peak_type == GSTLAL_PEAK_COMPLEX) {
			// First check if there's a trigger
			if(!itacacpad->maxdata->values.as_float_complex[channel])
				continue;
		}

		peak_sample = itacacpad->maxdata->samples[channel];
		if(strcmp(itacacpad->instrument, "H1") != 0 && itacac->H1_itacacpad != NULL && !itacac->H1_itacacpad->waiting)
			get_snr_series(itacac, itacac->H1_itacacpad, channel, peak_sample);
		if(strcmp(itacacpad->instrument, "K1") != 0 && itacac->K1_itacacpad != NULL && !itacac->K1_itacacpad->waiting)
			get_snr_series(itacac, itacac->K1_itacacpad, channel, peak_sample);
		if(strcmp(itacacpad->instrument, "L1") != 0 && itacac->L1_itacacpad != NULL && !itacac->L1_itacacpad->waiting)
			get_snr_series(itacac, itacac->L1_itacacpad, channel, peak_sample);
		if(strcmp(itacacpad->instrument, "V1") != 0 && itacac->V1_itacacpad != NULL && !itacac->V1_itacacpad->waiting)
			get_snr_series(itacac, itacac->V1_itacacpad, channel, peak_sample);

	}
}

static GstBuffer* assemble_srcbuf(GSTLALItacac *itacac, GSTLALItacacPad *itacacpad, GstBuffer *srcbuf) {
	if(strcmp(itacacpad->instrument, "H1") == 0) {
		if(srcbuf == NULL)
			srcbuf = gstlal_snglinspiral_new_buffer_from_peak(itacacpad->maxdata, itacacpad->bankarray, GST_PAD((itacac->aggregator).srcpad), itacac->next_output_offset, itacacpad->n, itacac->next_output_timestamp, itacac->rate, itacacpad->chi2, &(itacac->H1_itacacpad->snr_matrix_view), itacac->K1_itacacpad != NULL ? &(itacac->K1_itacacpad->tmp_snr_matrix_view) : NULL, itacac->L1_itacacpad != NULL ? &(itacac->L1_itacacpad->tmp_snr_matrix_view) : NULL, itacac->V1_itacacpad != NULL ? &(itacac->V1_itacacpad->tmp_snr_matrix_view) : NULL, itacac->difftime);
		else
			gstlal_snglinspiral_append_peak_to_buffer(srcbuf, itacacpad->maxdata, itacacpad->bankarray, GST_PAD((itacac->aggregator).srcpad), itacac->next_output_offset, itacacpad->n, itacac->next_output_timestamp, itacac->rate, itacacpad->chi2, &(itacac->H1_itacacpad->snr_matrix_view), itacac->K1_itacacpad != NULL ? &(itacac->K1_itacacpad->tmp_snr_matrix_view) : NULL, itacac->L1_itacacpad != NULL ? &(itacac->L1_itacacpad->tmp_snr_matrix_view) : NULL, itacac->V1_itacacpad != NULL ? &(itacac->V1_itacacpad->tmp_snr_matrix_view) : NULL);
	} else if(strcmp(itacacpad->instrument, "K1") == 0) {
		if(srcbuf == NULL)
			srcbuf = gstlal_snglinspiral_new_buffer_from_peak(itacacpad->maxdata, itacacpad->bankarray, GST_PAD((itacac->aggregator).srcpad), itacac->next_output_offset, itacacpad->n, itacac->next_output_timestamp, itacac->rate, itacacpad->chi2, itacac->H1_itacacpad != NULL ? &(itacac->H1_itacacpad->tmp_snr_matrix_view) : NULL, &(itacac->K1_itacacpad->snr_matrix_view), itacac->L1_itacacpad != NULL ? &(itacac->L1_itacacpad->tmp_snr_matrix_view) : NULL, itacac->V1_itacacpad != NULL ? &(itacac->V1_itacacpad->tmp_snr_matrix_view) : NULL, itacac->difftime);
		else
			gstlal_snglinspiral_append_peak_to_buffer(srcbuf, itacacpad->maxdata, itacacpad->bankarray, GST_PAD((itacac->aggregator).srcpad), itacac->next_output_offset, itacacpad->n, itacac->next_output_timestamp, itacac->rate, itacacpad->chi2, itacac->H1_itacacpad != NULL ? &(itacac->H1_itacacpad->tmp_snr_matrix_view) : NULL, &(itacac->K1_itacacpad->snr_matrix_view), itacac->L1_itacacpad != NULL ? &(itacac->L1_itacacpad->tmp_snr_matrix_view) : NULL, itacac->V1_itacacpad != NULL ? &(itacac->V1_itacacpad->tmp_snr_matrix_view) : NULL);
	} else if(strcmp(itacacpad->instrument, "L1") == 0) {
		if(srcbuf == NULL)
			srcbuf = gstlal_snglinspiral_new_buffer_from_peak(itacacpad->maxdata, itacacpad->bankarray, GST_PAD((itacac->aggregator).srcpad), itacac->next_output_offset, itacacpad->n, itacac->next_output_timestamp, itacac->rate, itacacpad->chi2, itacac->H1_itacacpad != NULL ? &(itacac->H1_itacacpad->tmp_snr_matrix_view) : NULL, itacac->K1_itacacpad != NULL ? &(itacac->K1_itacacpad->tmp_snr_matrix_view) : NULL, &(itacac->L1_itacacpad->snr_matrix_view), itacac->V1_itacacpad != NULL ? &(itacac->V1_itacacpad->tmp_snr_matrix_view) : NULL, itacac->difftime);
		else
			gstlal_snglinspiral_append_peak_to_buffer(srcbuf, itacacpad->maxdata, itacacpad->bankarray, GST_PAD((itacac->aggregator).srcpad), itacac->next_output_offset, itacacpad->n, itacac->next_output_timestamp, itacac->rate, itacacpad->chi2, itacac->H1_itacacpad != NULL ? &(itacac->H1_itacacpad->tmp_snr_matrix_view) : NULL, itacac->K1_itacacpad != NULL ? &(itacac->K1_itacacpad->tmp_snr_matrix_view) : NULL, &(itacac->L1_itacacpad->snr_matrix_view), itacac->V1_itacacpad != NULL ? &(itacac->V1_itacacpad->tmp_snr_matrix_view) : NULL);
	} else if(strcmp(itacacpad->instrument, "V1") == 0) {
		if(srcbuf == NULL)
			srcbuf = gstlal_snglinspiral_new_buffer_from_peak(itacacpad->maxdata, itacacpad->bankarray, GST_PAD((itacac->aggregator).srcpad), itacac->next_output_offset, itacacpad->n, itacac->next_output_timestamp, itacac->rate, itacacpad->chi2, itacac->H1_itacacpad != NULL ? &(itacac->H1_itacacpad->tmp_snr_matrix_view) : NULL, itacac->K1_itacacpad != NULL ? &(itacac->K1_itacacpad->tmp_snr_matrix_view) : NULL, itacac->L1_itacacpad != NULL ? &(itacac->L1_itacacpad->tmp_snr_matrix_view) : NULL, &(itacac->V1_itacacpad->snr_matrix_view), itacac->difftime);
		else
			gstlal_snglinspiral_append_peak_to_buffer(srcbuf, itacacpad->maxdata, itacacpad->bankarray, GST_PAD((itacac->aggregator).srcpad), itacac->next_output_offset, itacacpad->n, itacac->next_output_timestamp, itacac->rate, itacacpad->chi2, itacac->H1_itacacpad != NULL ? &(itacac->H1_itacacpad->tmp_snr_matrix_view) : NULL, itacac->K1_itacacpad != NULL ? &(itacac->K1_itacacpad->tmp_snr_matrix_view) : NULL, itacac->L1_itacacpad != NULL ? &(itacac->L1_itacacpad->tmp_snr_matrix_view) : NULL, &(itacac->V1_itacacpad->snr_matrix_view) );
	}
	return srcbuf;
}

static GstFlowReturn process(GSTLALItacac *itacac) {
	// Iterate through audioadapters and generate triggers
	GstElement *element = GST_ELEMENT(itacac);
	guint outsamps, nongapsamps, copysamps, samples_left_in_window;
	guint gapsamps = 0;
	GstFlowReturn result = GST_FLOW_OK;
	GList *padlist;
	GstBuffer *srcbuf = NULL;
	guint availablesamps;

        // Make sure we have enough samples to produce a trigger
        // FIXME Currently assumes every pad has the same n
	for(padlist = element->sinkpads; padlist != NULL; padlist = padlist->next) {
		GSTLALItacacPad *itacacpad = GSTLAL_ITACAC_PAD(padlist->data);
		if(gst_audioadapter_available_samples( itacacpad->adapter ) <= itacacpad->n + 2*itacacpad->maxdata->pad && !itacacpad->EOS && !itacacpad->waiting) {
			return result;
		}
	}


	for(padlist = element->sinkpads; padlist != NULL; padlist = padlist->next) {
		GSTLALItacacPad *itacacpad = GSTLAL_ITACAC_PAD(padlist->data);
		if(itacacpad->waiting)
			continue;

		samples_left_in_window = itacacpad->n;


		// FIXME Currently assumes n is the same for all detectors
		while( samples_left_in_window > 0 && gst_audioadapter_available_samples(itacacpad->adapter) ) {

			// Check how many gap samples there are until a nongap
			// or vice versa, depending on which comes first
			gapsamps = gst_audioadapter_head_gap_length(itacacpad->adapter);
			nongapsamps = gst_audioadapter_head_nongap_length(itacacpad->adapter);
			availablesamps = gst_audioadapter_available_samples(itacacpad->adapter);

			// NOTE Remember that if you didnt just come off a gap, you should always have a pad worth of nongap samples that have been processed already

			// Check if the samples are gap, and flush up to samples_left_in_window of them if so
			if(gapsamps > 0) {
				outsamps = gapsamps > samples_left_in_window ? samples_left_in_window : gapsamps;
				gst_audioadapter_flush_samples(itacacpad->adapter, outsamps);
				samples_left_in_window -= outsamps;
			}
			// Check if we have enough nongap samples to compute chisq for a potential trigger, and if not, check if we should flush the samples or 
			// if theres a possibility we could get more nongapsamples in the future
			else if(nongapsamps <= 2 * itacacpad->maxdata->pad) {
				if(nongapsamps == availablesamps && !itacacpad->EOS) {
					// We have reached the end of available samples, thus there could still be enough nongaps in the next window for triggers
					// Assertion logic: we're guarenteed at least one more sample than a trigger window and 2 pad's worth of samples, thus if
					// all of the available samples are nongapsamps and we have 2 pad worth of non gap samples samples or less, our nongapsamps
					// must go into the next window and we must have just come off a gap, meaning we dont have any samples already available for
					// padding
					g_assert(nongapsamps > samples_left_in_window);
					g_assert(itacacpad->samples_available_for_padding == 0);
					itacacpad->samples_available_for_padding = samples_left_in_window;
					samples_left_in_window = 0;
				} else {
					// We are guaranteed to have at least one sample more than a pad worth of samples past the end of the 
					// trigger window, thus we know there must be a gap sample after these, and can ditch them, though we 
					// need to make sure we aren't flushing any samples from the next trigger window
					g_assert(availablesamps > nongapsamps || itacacpad->EOS);
					g_assert(itacacpad->samples_available_for_padding == 0);
					outsamps = nongapsamps > samples_left_in_window ? samples_left_in_window : nongapsamps;
					gst_audioadapter_flush_samples(itacacpad->adapter, outsamps);
					samples_left_in_window -= outsamps;

				}
			}
			// Not enough samples left in the window to produce a trigger or possibly even fill up a pad for a trigger in the next window
			else if(samples_left_in_window <= itacacpad->maxdata->pad) {
				itacacpad->samples_available_for_padding = samples_left_in_window;
				samples_left_in_window = 0;
			}
			// Have a pad worth of samples after this window and possibly some (up to a pad) before
			// want to copy samples_left_in_window + pad + samples from previous window if any
			// Want to flush up to a pad worth of samples before the next window
			// Want peak finding length to go from first sample after pad to last sample of window
			// samples_left_in_window will be zero after this
			else if(nongapsamps >= samples_left_in_window + itacacpad->maxdata->pad + itacacpad->samples_available_for_padding) {
				copysamps = samples_left_in_window + itacacpad->maxdata->pad + itacacpad->samples_available_for_padding;
				outsamps = samples_left_in_window + itacacpad->samples_available_for_padding - itacacpad->maxdata->pad;
				copy_nongapsamps(itacac, itacacpad, copysamps, outsamps, (gint) itacacpad->n - 1 * (gint) samples_left_in_window - 1 * (gint) itacacpad->samples_available_for_padding);
				gst_audioadapter_flush_samples(itacacpad->adapter, outsamps);
				samples_left_in_window = 0;
				itacacpad->samples_available_for_padding = itacacpad->maxdata->pad;
			}
			// There is a gap coming up in the next window
			else {
				g_assert(availablesamps > nongapsamps || itacacpad->EOS);
				copysamps = nongapsamps;
				outsamps = nongapsamps >= samples_left_in_window + itacacpad->samples_available_for_padding ? samples_left_in_window + itacacpad->samples_available_for_padding : nongapsamps;
				copy_nongapsamps(itacac, itacacpad, copysamps, copysamps - 2*itacacpad->maxdata->pad, (gint) itacacpad->n - 1*(gint) samples_left_in_window - 1 * (gint) itacacpad->samples_available_for_padding);
				gst_audioadapter_flush_samples(itacacpad->adapter, outsamps);
				samples_left_in_window = nongapsamps >= samples_left_in_window + itacacpad->samples_available_for_padding ? 0 : samples_left_in_window + itacacpad->samples_available_for_padding - nongapsamps;
				itacacpad->samples_available_for_padding = 0;
				//samples_left_in_window + itacacpad->maxdata->pad + itacacpad->samples_available_for_padding;
			}

		}
	}

	guint data_container_index;
	guint peak_finding_start;
	guint duration;
	guint samples_searched_in_window;
	gint trig_offset;

	for(padlist = element->sinkpads; padlist != NULL; padlist = padlist->next) {
		GSTLALItacacPad *itacacpad = GSTLAL_ITACAC_PAD(padlist->data);
		if(itacacpad->waiting)
			continue;
		data_container_index = 0;
		duration = (guint) gsl_matrix_get(itacacpad->data->duration_dataoffset_trigwindowoffset_peakfindinglength_matrix, 0, 0);

		gboolean triggers_generated = FALSE;
		while(duration != 0) {
			peak_finding_start = (guint) gsl_matrix_get(itacacpad->data->duration_dataoffset_trigwindowoffset_peakfindinglength_matrix, data_container_index, 1);
			trig_offset = (gint) gsl_matrix_get(itacacpad->data->duration_dataoffset_trigwindowoffset_peakfindinglength_matrix, data_container_index, 2);

			peak_finding_start += itacacpad->maxdata->pad;
			if(trig_offset < 0) {
				// We need to use the first pad worth of samples to compute chisq for potential triggers
				samples_searched_in_window = itacacpad->maxdata->pad - (guint) abs(trig_offset);
			} else 
				samples_searched_in_window = (guint) trig_offset + itacacpad->maxdata->pad;

			// Sanity check

			g_assert(samples_searched_in_window < itacacpad->n);


			generate_triggers(
				itacac, 
				itacacpad, 
				itacacpad->data->data, 
				peak_finding_start,
				(guint) gsl_matrix_get(itacacpad->data->duration_dataoffset_trigwindowoffset_peakfindinglength_matrix, data_container_index, 3),
				samples_searched_in_window,
				triggers_generated
			);

			triggers_generated = TRUE;
			duration = (guint) gsl_matrix_get(itacacpad->data->duration_dataoffset_trigwindowoffset_peakfindinglength_matrix, ++data_container_index, 0);
		}

		// FIXME Could check for coincidence and only add extra snr
		// time series for a detector if it doesnt have a coincident
		// trigger. This could also be where an early version of the
		// network snr cut is imposed
		if(triggers_generated && !itacacpad->maxdata->no_peaks_past_threshold) {
			populate_snr_in_other_detectors(itacac, itacacpad);
		}

		if(triggers_generated && itacacpad->autocorrelation_matrix) {
			srcbuf = assemble_srcbuf(itacac, itacacpad, srcbuf);
		} else if(triggers_generated) {
			if(srcbuf == NULL) {
				srcbuf = gstlal_snglinspiral_new_buffer_from_peak(itacacpad->maxdata, itacacpad->bankarray, GST_PAD((itacac->aggregator).srcpad), itacac->next_output_offset, itacacpad->n, itacac->next_output_timestamp, itacac->rate, itacacpad->chi2, NULL, NULL, NULL, NULL, itacac->difftime);
			} else {
				gstlal_snglinspiral_append_peak_to_buffer(srcbuf, itacacpad->maxdata, itacacpad->bankarray, GST_PAD((itacac->aggregator).srcpad), itacac->next_output_offset, itacacpad->n, itacac->next_output_timestamp, itacac->rate, itacacpad->chi2, NULL, NULL, NULL, NULL);
			}
		}

	}

	// clear the matrix that tracks information about our saved data
	for(padlist = element->sinkpads; padlist != NULL; padlist = padlist->next) {
		GSTLALItacacPad *itacacpad = GSTLAL_ITACAC_PAD(padlist->data);
		if(itacacpad->waiting)
			continue;
		gsl_matrix_scale(itacacpad->data->duration_dataoffset_trigwindowoffset_peakfindinglength_matrix, 0);
	}

	if(!itacac->EOS) {
		if(srcbuf != NULL)
			result = gst_aggregator_finish_buffer(GST_AGGREGATOR(itacac), srcbuf);
		else 
			// FIXME Assumes n is same for all ifos
			result = push_gap(itacac, GSTLAL_ITACAC_PAD(element->sinkpads->data)->n);
	} else {
		guint max_num_samps_left_in_any_pad = 0;
		guint available_samps;
		for(padlist=element->sinkpads; padlist != NULL; padlist = padlist->next) {
			GSTLALItacacPad *itacacpad = GSTLAL_ITACAC_PAD(padlist->data);
			available_samps = gst_audioadapter_available_samples(itacacpad->adapter);
			max_num_samps_left_in_any_pad = available_samps > max_num_samps_left_in_any_pad ? available_samps : max_num_samps_left_in_any_pad;
		}

		// If there aren't any samples left to process, then we're ready to return GST_FLOW_EOS
		if(max_num_samps_left_in_any_pad > 0) {
			if(srcbuf != NULL)
				gst_aggregator_finish_buffer(GST_AGGREGATOR(itacac), srcbuf);
			else
				push_gap(itacac, GSTLAL_ITACAC_PAD(element->sinkpads->data)->n);

			result = process(itacac);
		} else {
			if(srcbuf != NULL) {
				gst_aggregator_finish_buffer(GST_AGGREGATOR(itacac), srcbuf);
			}
			result = GST_FLOW_EOS;
		}
	}

	return result;
}

static GstFlowReturn aggregate(GstAggregator *aggregator, gboolean timeout)
{
	GSTLALItacac *itacac = GSTLAL_ITACAC(aggregator);
	GList *padlist;
	GstFlowReturn result = GST_FLOW_OK;

	if(itacac->EOS) {
		result = process(itacac);
		return result;
	}

	for(padlist = GST_ELEMENT(aggregator)->sinkpads; padlist != NULL; padlist = padlist->next) {
		// Get the buffer from the pad we're looking at and assert it
		// has a valid timestamp
		GSTLALItacacPad *itacacpad = GSTLAL_ITACAC_PAD(padlist->data);
		// We don't need to worry about this if this pad is waiting
		if(itacacpad->waiting && gst_audioadapter_available_samples(itacacpad->adapter) != 0)
			continue;

		GstBuffer *sinkbuf = gst_aggregator_pad_peek_buffer(GST_AGGREGATOR_PAD(itacacpad));
		if(sinkbuf == NULL) {
			GST_DEBUG_OBJECT(itacac, "%s sinkbuf is NULL", itacacpad->instrument);
			continue;
		}
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
			gst_buffer_unref(sinkbuf);
			GST_ELEMENT_ERROR(itacacpad, STREAM, FAILED, ("missing or invalid tags"), ("instrument and/or channel name not known (stream's tags must provide this information)"));
			result = GST_FLOW_ERROR;
			return result;
		}

		if(!itacacpad->bankarray) {
			gst_buffer_unref(sinkbuf);
			GST_ELEMENT_ERROR(itacacpad, STREAM, FAILED, ("missing bank file"), ("must have a valid template bank to create events"));
			result = GST_FLOW_ERROR;
			return result;
		}


		// FIXME if we were more careful we wouldn't lose so much data around disconts
		if(GST_BUFFER_FLAG_IS_SET(sinkbuf, GST_BUFFER_FLAG_DISCONT)) {
			// FIXME For now, this should ensure we only see disconts at start up
			g_assert(gst_audioadapter_available_samples(itacacpad->adapter) == 0);
			itacacpad->initial_timestamp = GST_CLOCK_TIME_NONE;
			gst_audioadapter_clear(itacacpad->adapter);
			if(!itacacpad->waiting) { 
				reset_time_and_offset(itacac);
			}
		}

		// If the adapter has too many samples stored in memory, just
		// ignore this buffer for now. We need n + 2*maxdata->pad for a
		// trigger, so adapter should never need to hold more than 2*(2
		// + 2*maxdata->pad).
		if(gst_audioadapter_available_samples(itacacpad->adapter) > 2*(itacacpad->n + 2*itacacpad->maxdata->pad)) {
			gst_buffer_unref(sinkbuf);
			continue;
		}

		// Grab timestamp for this pad if we dont have it already
		if(itacacpad->initial_timestamp == GST_CLOCK_TIME_NONE) {
			g_assert(gst_audioadapter_available_samples(itacacpad->adapter) == 0);
			itacacpad->initial_timestamp = GST_BUFFER_PTS(sinkbuf);
		}

		// Push buf to gstaudioadapter
		gst_audioadapter_push(itacacpad->adapter, sinkbuf);
		gst_aggregator_pad_drop_buffer(GST_AGGREGATOR_PAD(itacacpad));


	}

	// Find earliest timestamp between all of the pads if we don't have the
	// first output timestamp, which means itacac hasn't pushed any buffers yet
	if(itacac->next_output_timestamp == GST_CLOCK_TIME_NONE) {
		for(padlist = GST_ELEMENT(aggregator)->sinkpads; padlist != NULL; padlist = padlist->next) {
			GSTLALItacacPad *itacacpad = GSTLAL_ITACAC_PAD(padlist->data);
			if(gst_audioadapter_available_samples(itacacpad->adapter) == 0)
				continue;
			itacac->next_output_timestamp = itacac->next_output_timestamp == GST_CLOCK_TIME_NONE ? itacacpad->initial_timestamp : itacac->next_output_timestamp <= itacacpad->initial_timestamp ? itacac->next_output_timestamp : itacacpad->initial_timestamp;
		}
	}

	// Determine if we can start taking data from each pad that is still waiting (if any)
	for(padlist = GST_ELEMENT(aggregator)->sinkpads; padlist != NULL; padlist = padlist->next) {
		if(itacac->next_output_timestamp == GST_CLOCK_TIME_NONE)
			// We're at the beginning and don't have any data yet (for reaons I don't understand, itacac receives empty buffers at start up)
			return result;

		GSTLALItacacPad *itacacpad = GSTLAL_ITACAC_PAD(padlist->data);
		if(!itacacpad->waiting || itacacpad->initial_timestamp > itacac->next_output_timestamp)
			continue;

		// FIXME Assumes n is the same for all detectors
		guint num_samples_behind = (guint) ((itacac->next_output_timestamp - itacacpad->initial_timestamp) / (1000000000 / itacac->rate));
		if(num_samples_behind > itacacpad->maxdata->pad)
			gst_audioadapter_flush_samples(itacacpad->adapter, MIN(num_samples_behind - itacacpad->maxdata->pad, gst_audioadapter_available_samples(itacacpad->adapter)));
		itacacpad->samples_available_for_padding = num_samples_behind > itacacpad->maxdata->pad ? itacacpad->maxdata->pad : num_samples_behind;

		itacacpad->waiting = FALSE;
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
	GST_DEBUG_OBJECT(GST_AGGREGATOR_PAD(object), "in pad_dispose");
	GSTLALItacacPad *itacacpad = GSTLAL_ITACAC_PAD(object);

	gst_audioadapter_clear(itacacpad->adapter);
	g_object_unref(itacacpad->adapter);
	itacacpad->adapter = NULL;

	free_bank(itacacpad);

	free(itacacpad->instrument);
	itacacpad->instrument = NULL;

	free(itacacpad->channel_name);
	itacacpad->channel_name = NULL;

	if(itacacpad->maxdata) {
		gstlal_peak_state_free(itacacpad->maxdata);
		itacacpad->maxdata = NULL;
	}

	if(itacacpad->tmp_maxdata) {
		gstlal_peak_state_free(itacacpad->tmp_maxdata);
		itacacpad->tmp_maxdata = NULL;
	}

	if(itacacpad->data) {
		free(itacacpad->data->data);
		itacacpad->data->data = NULL;

		if(itacacpad->data->duration_dataoffset_trigwindowoffset_peakfindinglength_matrix) {
			gsl_matrix_free(itacacpad->data->duration_dataoffset_trigwindowoffset_peakfindinglength_matrix);
			itacacpad->data->duration_dataoffset_trigwindowoffset_peakfindinglength_matrix = NULL;
		}

		free(itacacpad->data);
		itacacpad->data = NULL;
	}

	free(itacacpad->snr_mat);
	free(itacacpad->tmp_snr_mat);
	free(itacacpad->autocorrelation_matrix);
	free(itacacpad->autocorrelation_mask);
	free(itacacpad->autocorrelation_norm);
	free(itacacpad->chi2);
	free(itacacpad->tmp_chi2);
	itacacpad->snr_mat = NULL;
	itacacpad->tmp_snr_mat = NULL;
	itacacpad->autocorrelation_matrix = NULL;
	itacacpad->autocorrelation_mask = NULL;
	itacacpad->autocorrelation_norm = NULL;
	itacacpad->chi2 = NULL;
	itacacpad->tmp_chi2 = NULL;

	G_OBJECT_CLASS(gstlal_itacac_pad_parent_class)->dispose(object);
}

static void gstlal_itacac_finalize(GObject *object)
{
	//GSTLALItacac *itacac = GSTLAL_ITACAC(object);
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
			(GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)
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
			(GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)
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
			(GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)
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
				(GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)
			),
			(GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_CONTROLLABLE)
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
					(GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)
				),
				(GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)
			),
		(GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)
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
					(GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)
				),
				(GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)
			),
			(GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)
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
	aggregator_class->finish_buffer = GST_DEBUG_FUNCPTR(push_buffer);
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
}

/*
 * Instance init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GInstanceInitFunc
 */

static void gstlal_itacac_pad_init(GSTLALItacacPad *itacacpad)
{
	itacacpad->adapter = g_object_new(GST_TYPE_AUDIOADAPTER, NULL);
	itacacpad->data = g_malloc(sizeof(*itacacpad->data));
	itacacpad->data->duration_dataoffset_trigwindowoffset_peakfindinglength_matrix = NULL;
	itacacpad->data->data = NULL;
	itacacpad->chi2 = NULL;
	itacacpad->tmp_chi2 = NULL;
	itacacpad->bank_filename = NULL;
	itacacpad->instrument = NULL;
	itacacpad->channel_name = NULL;
	itacacpad->snr_thresh = 0;
	g_mutex_init(&itacacpad->bank_lock);

	itacacpad->autocorrelation_matrix = NULL;
	itacacpad->autocorrelation_mask = NULL;
	itacacpad->autocorrelation_norm = NULL;
	itacacpad->snr_mat = NULL;
	itacacpad->tmp_snr_mat = NULL;
	itacacpad->bankarray = NULL;
	itacacpad->EOS = FALSE;
	itacacpad->waiting = TRUE;

	itacacpad->samples_available_for_padding = 0;
	itacacpad->initial_timestamp = GST_CLOCK_TIME_NONE;


	gst_pad_use_fixed_caps(GST_PAD(itacacpad));
}

static void gstlal_itacac_init(GSTLALItacac *itacac)
{
	itacac->rate = 0;
	itacac->channels = 0;

	itacac->difftime = 0;
	
	reset_time_and_offset(itacac);

	itacac->EOS = FALSE;
	itacac->H1_itacacpad = NULL;
	itacac->K1_itacacpad = NULL;
	itacac->L1_itacacpad = NULL;
	itacac->V1_itacacpad = NULL;
	g_mutex_init(&itacac->caps_lock);
}

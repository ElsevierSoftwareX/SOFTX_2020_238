/*
 * An inspiral trigger and autocorrelation chisq (burst_triggergen) element
 *
 * Copyright (C) 2011  Chad Hanna, Kipp Cannon
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
#include <math.h>
#include <string.h>

/*
 * our own stuff
 */

#include <gstlal/gstlal.h>
#include <gstlal/gstlal_debug.h>
#include <gstlal_burst_triggergen.h>
#include <gstlal/gstlal_peakfinder.h>
#include <gstlal/gstaudioadapter.h>
#include <gstlal/gstlal_tags.h>
#include <gstlal_snglburst.h>

#define DEFAULT_SNR_THRESH 5.5

static guint64 output_num_samps(GSTLALBurst_Triggergen *element)
{
	return (guint64) element->n;
}


static guint64 output_num_bytes(GSTLALBurst_Triggergen *element)
{
	return (guint64) output_num_samps(element) * element->adapter->unit_size;
}


static int reset_time_and_offset(GSTLALBurst_Triggergen *element)
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

static void free_bank(GSTLALBurst_Triggergen *element)
{
	g_free(element->bank_filename);
	element->bank_filename = NULL;
	free(element->bankarray);
	element->bankarray = NULL;
}


/*
 * ========================================================================
 *
 *                                  Triggers
 *
 * ========================================================================
 */


static int gstlal_set_channel_in_snglburst_array(SnglBurst *bankarray, int length, char *channel)
{
	int i;
	for (i = 0; i < length; i++) {
		if (channel) {
			strncpy(bankarray[i].channel, (const char*) channel, LIGOMETA_CHANNEL_MAX);
			bankarray[i].channel[LIGOMETA_CHANNEL_MAX - 1] = 0;
		}
	}
	return 0;
}


static int gstlal_set_instrument_in_snglburst_array(SnglBurst *bankarray, int length, char *instrument)
{
	int i;
	for (i = 0; i < length; i++) {
		if (instrument) {
	        	strncpy(bankarray[i].ifo, (const char*) instrument, LIGOMETA_IFO_MAX);
			bankarray[i].ifo[LIGOMETA_IFO_MAX - 1] = 0;
		}
	}
	return 0;
}


static SnglBurst *gstlal_snglburst_new_list_from_peak(struct gstlal_peak_state *input, SnglBurst *bankarray, GstClockTime etime, guint rate, SnglBurst* output)
{
	/* advance the pointer if we have one */
	guint channel;
	double complex *maxdata = input->values.as_double_complex;
	guint *maxsample = input->samples;

	/* FIXME do error checking */
	for(channel = 0; channel < input->channels; channel++) {
		if ( maxdata[channel] ) {
			SnglBurst *new_event = XLALCreateSnglBurst();
			memcpy(new_event, &(bankarray[channel]), sizeof(*new_event));
			LIGOTimeGPS peak_time;
			XLALINT8NSToGPS(&peak_time, etime);
			XLALGPSAdd(&peak_time, -new_event->duration/2);
			XLALGPSAdd(&peak_time, (double) maxsample[channel] / rate);
			LIGOTimeGPS start_time = peak_time;
			XLALGPSAdd(&start_time, -new_event->duration/2);
			new_event->snr = cabs(maxdata[channel]);
			new_event->start_time = start_time;
			new_event->peak_time = peak_time;
			new_event->next = output;
			output = new_event;
		}
	}

	return output;
}


static SnglBurst *gstlal_snglburst_new_list_from_double_peak(struct gstlal_peak_state *input, SnglBurst *bankarray, GstClockTime etime, guint rate, SnglBurst* output)
{
	/* advance the pointer if we have one */
	guint channel;
	double *maxdata = input->values.as_double;
	guint *maxsample = input->samples;

	/* FIXME do error checking */
	for(channel = 0; channel < input->channels; channel++) {
		if ( maxdata[channel] ) {
			SnglBurst *new_event = XLALCreateSnglBurst();
			memcpy(new_event, &(bankarray[channel]), sizeof(*new_event));
			LIGOTimeGPS peak_time;
			XLALINT8NSToGPS(&peak_time, etime);
			XLALGPSAdd(&peak_time, (double) maxsample[channel] / rate);
			XLALGPSAdd(&peak_time, -new_event->duration/2);
			// Center the tile
			XLALGPSAdd(&peak_time, 1.0/(2.0*rate));
			LIGOTimeGPS start_time = peak_time;
			XLALGPSAdd(&start_time, -new_event->duration/2);
			new_event->snr = fabs(maxdata[channel]);
			new_event->start_time = start_time;
			new_event->peak_time = peak_time;
			new_event->next = output;
			output = new_event;
		}
	}

	return output;
}


static GstBuffer *gstlal_snglburst_new_buffer_from_list(SnglBurst *input, GstPad *pad, guint64 offset, guint64 length, GstClockTime etime, guint rate, guint64 *count)
{
	/* FIXME check errors */

	/* size is length in samples times number of channels times number of bytes per sample */
	gint size = XLALSnglBurstTableLength(input);
	size *= sizeof(*input);
	GstBuffer *srcbuf = NULL;
	GstCaps *caps = GST_PAD_CAPS(pad);
	GstFlowReturn result = gst_pad_alloc_buffer(pad, offset, size, caps, &srcbuf);
	if (result != GST_FLOW_OK)
		return srcbuf;

	if (input) {
		//FIXME: Process ID
		(*count) = XLALSnglBurstAssignIDs(input, 0, *count);
	}

	/* Copy the events into the buffer */
	SnglBurst *output = (SnglBurst *) GST_BUFFER_DATA(srcbuf);
    SnglBurst *head = input;
	while (input) {
		*output = *input;
		/* Make the array look like a linked list */
		output->next = input->next ? output+1 : NULL;
		output++;
		input = input->next;
	}
	/* Forget about this set of events, it's the buffer's now. */
	XLALDestroySnglBurstTable(head);

	if (size == 0)
		GST_BUFFER_FLAG_SET(srcbuf, GST_BUFFER_FLAG_GAP);

	/* set the offset */
	GST_BUFFER_OFFSET(srcbuf) = offset;
	GST_BUFFER_OFFSET_END(srcbuf) = offset + length;

	/* set the time stamps */
	GST_BUFFER_TIMESTAMP(srcbuf) = etime;
	GST_BUFFER_DURATION(srcbuf) = (GstClockTime) gst_util_uint64_scale_int_round(GST_SECOND, length, rate);

	return srcbuf;
}

static SnglBurst *gstlal_snglburst_new_list_from_double_buffer(double *input, SnglBurst *bankarray, GstClockTime etime, guint channels, guint samples, guint rate, gdouble threshold, SnglBurst* output)
{
	/* advance the pointer if we have one */
	guint channel, sample;

	/* FIXME do error checking */
	for (channel = 0; channel < channels; channel++) {
	    for (sample = 0; sample < samples; sample++) {
		    if (input[channels*sample+channel] > threshold) {
			    SnglBurst *new_event = XLALCreateSnglBurst();
			    memcpy(new_event, &(bankarray[channel]), sizeof(*new_event));
			    LIGOTimeGPS peak_time;
			    XLALINT8NSToGPS(&peak_time, etime);
			    XLALGPSAdd(&peak_time, (double) sample / rate);
			    XLALGPSAdd(&peak_time, -new_event->duration/2);
			    // Center the tile
			    XLALGPSAdd(&peak_time, 1.0/(2.0*rate));
			    LIGOTimeGPS start_time = peak_time;
			    XLALGPSAdd(&start_time, -new_event->duration/2);
			    new_event->snr = fabs(input[channels*sample+channel]);
			    new_event->start_time = start_time;
			    new_event->peak_time = peak_time;
			    new_event->next = output;
			    output = new_event;
		    }
        }
	}

	return output;
}


static SnglBurst *gstlal_snglburst_new_list_from_complex_double_buffer(complex double *input, SnglBurst *bankarray, GstClockTime etime, guint channels, guint samples, guint rate, gdouble threshold, SnglBurst* output)
{
	/* advance the pointer if we have one */
	guint channel, sample;

	/* FIXME do error checking */
	for (channel = 0; channel < channels; channel++) {
	    for (sample = 0; sample < samples; sample++) {
	        /* FIXME Which are we thresholding on. the EP version uses the 
             * squared value, so we make this consistent */
		    if (cabs(input[channels*sample+channel]) > sqrt(threshold)) {
			    SnglBurst *new_event = XLALCreateSnglBurst();
			    memcpy(new_event, &(bankarray[channel]), sizeof(*new_event));
			    LIGOTimeGPS peak_time;
			    XLALINT8NSToGPS(&peak_time, etime);
			    XLALGPSAdd(&peak_time, (double) sample / rate);
			    XLALGPSAdd(&peak_time, -new_event->duration/2);
			    // Center the tile
			    XLALGPSAdd(&peak_time, 1.0/(2.0*rate));
			    LIGOTimeGPS start_time = peak_time;
			    XLALGPSAdd(&start_time, -new_event->duration/2);
			    new_event->snr = cabs(input[channels*sample+channel]);
			    new_event->start_time = start_time;
			    new_event->peak_time = peak_time;
			    new_event->next = output;
			    output = new_event;
		    }
        }
	}

	return output;
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
/* FIXME the function placements should be moved around to avoid putting this static prototype here */

static GstFlowReturn process(GSTLALBurst_Triggergen *element);

static gboolean sink_event(GstPad *pad, GstEvent *event)
{
	GSTLALBurst_Triggergen *element = GSTLAL_BURST_TRIGGERGEN(GST_PAD_PARENT(pad));
	gboolean success;
	GstFlowReturn result;

	switch(GST_EVENT_TYPE(event)) {

	case GST_EVENT_TAG: {
		GstTagList *taglist;
		gchar *instrument, *channel_name;
		gst_event_parse_tag(event, &taglist);
		success = taglist_extract_string(GST_OBJECT(pad), taglist, GSTLAL_TAG_INSTRUMENT, &instrument);
		success &= taglist_extract_string(GST_OBJECT(pad), taglist, GSTLAL_TAG_CHANNEL_NAME, &channel_name);
		if(success) {
			GST_DEBUG_OBJECT(pad, "found tags \"%s\"=\"%s\", \"%s\"=\"%s\"", GSTLAL_TAG_INSTRUMENT, instrument, GSTLAL_TAG_CHANNEL_NAME, channel_name);
			g_free(element->instrument);
			element->instrument = instrument;
			g_free(element->channel_name);
			element->channel_name = channel_name;
			g_mutex_lock(element->bank_lock);
			gstlal_set_channel_in_snglburst_array(element->bankarray, element->channels, element->channel_name);
			gstlal_set_instrument_in_snglburst_array(element->bankarray, element->channels, element->instrument);
			g_mutex_unlock(element->bank_lock);
			}
		success = gst_pad_event_default(pad, event);
		break;
		}
	/* FIXME, will this always occur before last chain function is called?? */
	case GST_EVENT_EOS: {
		element->EOS = TRUE;
		/* FIXME check this output */
		result = process(element);
		success = gst_pad_event_default(pad, event);
		break;
		}
	default: {
		success = gst_pad_event_default(pad, event);
		break;
		}
	}

	return success;
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
	ARG_BANK_FILENAME,
	ARG_SIGMASQ
};


static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec)
{
	GSTLALBurst_Triggergen *element = GSTLAL_BURST_TRIGGERGEN(object);

	GST_OBJECT_LOCK(element);


	switch(id) {
	case ARG_N:
		element->n = g_value_get_uint(value);
		break;

	case ARG_SNR_THRESH:
		element->snr_thresh = g_value_get_double(value);
		break;

	case ARG_BANK_FILENAME:
		g_mutex_lock(element->bank_lock);
		element->bank_filename = g_value_dup_string(value);
		element->channels = gstlal_snglburst_array_from_file(element->bank_filename, &(element->bankarray));
		if (element->instrument && element->channel_name) {
			gstlal_set_instrument_in_snglburst_array(element->bankarray, element->channels, element->instrument);
			gstlal_set_channel_in_snglburst_array(element->bankarray, element->channels, element->channel_name);
		}
		g_mutex_unlock(element->bank_lock);
		break;

	case ARG_SIGMASQ: {
		g_mutex_lock(element->bank_lock);
		if(element->bankarray) {
			gint length;
			double *sigmasq = gstlal_doubles_from_g_value_array(g_value_get_boxed(value), NULL, &length);
			if((gint) element->channels != length)
				GST_ERROR_OBJECT(element, "vector length (%d) does not match number of templates (%d)", length, element->channels);
			else
				/* FIXME support sigma sq, maybe convert to hrss?? */
				GST_WARNING_OBJECT(element, "sigmasq not supported yet");
			g_free(sigmasq);
		} else
			GST_WARNING_OBJECT(element, "must set template bank before setting sigmasq");
		g_mutex_unlock(element->bank_lock);
		break;
	}


	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}


	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *pspec)
{
	GSTLALBurst_Triggergen *element = GSTLAL_BURST_TRIGGERGEN(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_N:
		g_value_set_uint(value, element->n);
		break;

	case ARG_SNR_THRESH:
		g_value_set_double(value, element->snr_thresh);
		break;

	case ARG_BANK_FILENAME:
		g_mutex_lock(element->bank_lock);
		g_value_set_string(value, element->bank_filename);
		g_mutex_unlock(element->bank_lock);
		break;

	case ARG_SIGMASQ: {
		g_mutex_lock(element->bank_lock);
		GST_WARNING_OBJECT(element, "sigma sq not supported");
		g_mutex_unlock(element->bank_lock);
		break;
	}

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
 * getcaps()
 */


static GstCaps *getcaps(GstPad * pad)
{
	GSTLALBurst_Triggergen *element = GSTLAL_BURST_TRIGGERGEN(gst_pad_get_parent(pad));
	GstCaps *peercaps, *caps;

	/*
	 * get our own allowed caps.  use the fixed caps function to avoid
	 * recursing back into this function.
	 */

	caps = gst_pad_get_fixed_caps_func(pad);

	/*
	 * get the allowed caps from the downstream peer if the peer has
	 * caps, intersect without our own.
	 */

	peercaps = gst_pad_peer_get_caps(element->srcpad);
	if(peercaps) {
		GstCaps *result = gst_caps_intersect(peercaps, caps);
		gst_caps_unref(peercaps);
		gst_caps_unref(caps);
		caps = result;
	}

	/*
	 * done
	 */

	gst_object_unref(element);
	return caps;
}


/*
 * setcaps()
 */


static gboolean setcaps(GstPad *pad, GstCaps *caps)
{
	GSTLALBurst_Triggergen *element = GSTLAL_BURST_TRIGGERGEN(gst_pad_get_parent(pad));
	GstStructure *structure;
	gint rate, width, channels;
	gboolean success = TRUE;
	const char* media_type;

	/*
	 * parse caps
	 */

	structure = gst_caps_get_structure(caps, 0);
	if(!gst_structure_get_int(structure, "rate", &rate))
		success = FALSE;
	if(!gst_structure_get_int(structure, "width", &width))
		success = FALSE;
	if(!gst_structure_get_int(structure, "channels", &channels))
		success = FALSE;

	media_type = gst_structure_get_name(structure);
	if(!strcmp(media_type, "audio/x-raw-float") && width == 64){
        if(element->n == 0){
		    element->data_type = GSTLAL_BURSTTRIGGEN_DOUBLE_NO_PEAK;
        } else {
		    element->data_type = GSTLAL_BURSTTRIGGEN_DOUBLE;
        }
	} else if(!strcmp(media_type, "audio/x-raw-complex") && width == 128 ){
        if(element->n == 0){
		    element->data_type = GSTLAL_BURSTTRIGGEN_COMPLEX_DOUBLE_NO_PEAK;
        } else {
		    element->data_type = GSTLAL_BURSTTRIGGEN_COMPLEX_DOUBLE;
        }
	} else return FALSE;

	/*
	 * try setting caps on downstream element
	 */

	if(success)
		success = gst_pad_set_caps(element->sinkpad, caps);

	/*
	 * update the element metadata
	 */

	if(success) {
		gstlal_peak_type_specifier type;
		element->channels = channels;
		element->rate = rate;
		g_object_set(element->adapter, "unit-size", width / 8 * channels, NULL);
		type = GSTLAL_PEAK_DOUBLE_COMPLEX;
		element->maxdata = gstlal_peak_state_new(channels, type);
		element->maxdata->pad = 0;
		type = GSTLAL_PEAK_DOUBLE;
		element->maxdatad = gstlal_peak_state_new(channels, type);
		element->maxdatad->pad = 0;
	}

	/*
	 * done
	 */

	gst_object_unref(element);
	return success;
}


/*
 * chain()
 */

static void update_state(GSTLALBurst_Triggergen *element, GstBuffer *srcbuf)
{
	element->next_output_offset = GST_BUFFER_OFFSET_END(srcbuf);
	element->next_output_timestamp = GST_BUFFER_TIMESTAMP(srcbuf) + GST_BUFFER_DURATION(srcbuf);
}

static GstFlowReturn push_buffer(GSTLALBurst_Triggergen *element, GstBuffer *srcbuf)
{
	GST_DEBUG_OBJECT(element, "pushing %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(srcbuf));
	GstFlowReturn result = gst_pad_push(element->srcpad, srcbuf);
	element->total_offset = 0;
	element->event_buffer = NULL;
	return result;
}

static GstFlowReturn push_gap(GSTLALBurst_Triggergen *element, guint samps)
{
	GstBuffer *srcbuf = NULL;
	GstFlowReturn result = GST_FLOW_OK;
	/* Clearing the max data structure and advance the offset */
	gstlal_peak_state_clear(element->maxdata);
	gstlal_peak_state_clear(element->maxdatad);

	element->total_offset += samps;

	/* potentially push the result */
	GstClockTime total_duration = (GstClockTime) gst_util_uint64_scale_int_round(GST_SECOND, element->total_offset, element->rate);
	if (total_duration >= 1e10 || element->EOS || element->n == 0) {
		srcbuf = gstlal_snglburst_new_buffer_from_list(element->event_buffer, element->srcpad, element->next_output_offset, element->total_offset, element->next_output_timestamp, element->rate, &(element->count));

		if (srcbuf == NULL) { 
			return GST_FLOW_ERROR;
		}
		update_state(element, srcbuf);
		result = push_buffer(element, srcbuf);
	}
	return result;
}

static GstFlowReturn push_nongap(GSTLALBurst_Triggergen *element, guint copysamps, guint outsamps)
{
	GstBuffer *srcbuf = NULL;
	GstFlowReturn result = GST_FLOW_OK;
	gint copied_gap, copied_nongap;
	double complex *dataptr = NULL;
	double *dataptrd = NULL;

	/* advance the offset */
	GstClockTime total_duration = (GstClockTime) gst_util_uint64_scale_int_round(GST_SECOND, element->total_offset, element->rate);
	element->total_offset += outsamps;

	switch(element->data_type){
		case GSTLAL_BURSTTRIGGEN_COMPLEX_DOUBLE:
			/* call the peak finding library on a buffer from the adapter if no events are found the result will be a GAP */
			gst_audioadapter_copy_samples(element->adapter, (void *) element->data, copysamps, &copied_gap, &copied_nongap);
			/* put the data pointer one pad length in */
			dataptr = element->data + element->maxdata->pad * element->maxdata->channels;
			/* Find the peak */
			gstlal_double_complex_peak_over_window(element->maxdata, (const double complex*) dataptr, outsamps);
			/* Either update the current buffer or create the new output buffer */
			if (element->maxdata->num_events != 0) {
				element->event_buffer = gstlal_snglburst_new_list_from_peak(element->maxdata, element->bankarray, element->next_output_timestamp + total_duration, element->rate, element->event_buffer);
			}
			break;
		case GSTLAL_BURSTTRIGGEN_DOUBLE:
			/* call the peak finding library on a buffer from the adapter if no events are found the result will be a GAP */
			gst_audioadapter_copy_samples(element->adapter, (void *) element->datad, copysamps, &copied_gap, &copied_nongap);
			/* put the data pointer one pad length in */
			dataptrd = element->datad + element->maxdatad->pad * element->maxdatad->channels;
			/* Find the peak */
			gstlal_double_peak_over_window(element->maxdatad, (const double*) dataptrd, outsamps);
			/* Either update the current buffer or create the new output buffer */
			if (element->maxdatad->num_events != 0) {
				element->event_buffer = gstlal_snglburst_new_list_from_double_peak(element->maxdatad, element->bankarray, element->next_output_timestamp + total_duration, element->rate, element->event_buffer);
		    }
			break;
		case GSTLAL_BURSTTRIGGEN_COMPLEX_DOUBLE_NO_PEAK:
			/* Either update the current buffer or create the new output buffer */
            dataptr = malloc(sizeof(complex double)*copysamps*element->channels);
			gst_audioadapter_copy_samples(element->adapter, dataptrd, copysamps, &copied_gap, &copied_nongap);
            element->event_buffer = gstlal_snglburst_new_list_from_complex_double_buffer(dataptr, element->bankarray, element->next_output_timestamp, element->channels, copysamps, element->rate, element->snr_thresh, element->event_buffer);
            free(dataptrd);
			break;
		case GSTLAL_BURSTTRIGGEN_DOUBLE_NO_PEAK:
			/* Either update the current buffer or create the new output buffer */
            dataptrd = malloc(sizeof(double)*copysamps*element->channels);
			gst_audioadapter_copy_samples(element->adapter, dataptrd, copysamps, &copied_gap, &copied_nongap);
            element->event_buffer = gstlal_snglburst_new_list_from_double_buffer(dataptrd, element->bankarray, element->next_output_timestamp, element->channels, copysamps, element->rate, element->snr_thresh, element->event_buffer);
            free(dataptrd);
			break;
	}

	/* potentially push the result */
	if (total_duration >= 1e10 || element->EOS || element->n == 0) {
		srcbuf = gstlal_snglburst_new_buffer_from_list(element->event_buffer, element->srcpad, element->next_output_offset, element->total_offset, element->next_output_timestamp, element->rate, &(element->count));

		if (srcbuf == NULL) { 
			return GST_FLOW_ERROR;
		}
		update_state(element, srcbuf);
		result = push_buffer(element, srcbuf);
	}
	return result;
}

static GstFlowReturn process(GSTLALBurst_Triggergen *element)
{
	guint outsamps, gapsamps, nongapsamps, copysamps;
	GstFlowReturn result = GST_FLOW_OK;
	guint padbuf = 0;
	switch(element->data_type){
		case GSTLAL_BURSTTRIGGEN_COMPLEX_DOUBLE:
			padbuf = element->maxdata->pad;
			break;
		case GSTLAL_BURSTTRIGGEN_DOUBLE:
			padbuf = element->maxdatad->pad;
			break;
		case GSTLAL_BURSTTRIGGEN_COMPLEX_DOUBLE_NO_PEAK:
			break;
		case GSTLAL_BURSTTRIGGEN_DOUBLE_NO_PEAK:
			break;
	}

	while( (element->EOS && gst_audioadapter_available_samples(element->adapter)) || gst_audioadapter_available_samples(element->adapter) > (element->n + 2 * padbuf)) {

		/* See if the output is a gap or not */
		nongapsamps = gst_audioadapter_head_nongap_length(element->adapter);
		gapsamps = gst_audioadapter_head_gap_length(element->adapter);


        if (element->n == 0 && nongapsamps == 0) {
            outsamps = gapsamps + nongapsamps;
			result = push_gap(element, gapsamps);
			gst_audioadapter_flush_samples(element->adapter, gapsamps);
        }
        else if (element->n == 0) {
            outsamps = gapsamps + nongapsamps;
			result = push_nongap(element, outsamps, outsamps);
			gst_audioadapter_flush_samples(element->adapter, outsamps);
        }
		/* First check if the samples are gap */
		else if (gapsamps > 0) {
			element->last_gap = TRUE;
			outsamps = gapsamps > element->n ? element->n : gapsamps;
			result = push_gap(element, outsamps);
			/* knock off the first buffers worth of bytes since we don't need them any more */
			gst_audioadapter_flush_samples(element->adapter, outsamps);
		}
		/* The check to see if we have enough nongap samples to compute an output, else it is a gap too */
		else if (nongapsamps <= 2 * padbuf) {
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
				if (padbuf > 0)
					result = push_gap(element, padbuf);
				}
			/* if we have enough nongap samples then our output is length n, otherwise we have to knock the padding off of what is available */
			copysamps = (nongapsamps > (element->n + 2 * padbuf)) ? (element->n + 2 * padbuf) : nongapsamps;
			outsamps = (copysamps == nongapsamps) ? (copysamps - 2 * padbuf) : element->n;
			result = push_nongap(element, copysamps, outsamps);
			/* knock off the first buffers worth of bytes since we don't need them any more */
			gst_audioadapter_flush_samples(element->adapter, outsamps);

			/* We are on another gap boundary so push the end transient as a gap */
			if (copysamps == nongapsamps) {
				element->last_gap = FALSE;
				if (padbuf > 0) {
					result = push_gap(element, padbuf);
					gst_audioadapter_flush_samples(element->adapter, 2 * padbuf);
					}
				}
			}
		}

	return result;
}

static GstFlowReturn chain(GstPad *pad, GstBuffer *sinkbuf)
{
	GSTLALBurst_Triggergen *element = GSTLAL_BURST_TRIGGERGEN(gst_pad_get_parent(pad));
	GstFlowReturn result = GST_FLOW_OK;
	guint padbuf = 0;
	switch(element->data_type){
		case GSTLAL_BURSTTRIGGEN_COMPLEX_DOUBLE:
			padbuf = element->maxdata->pad;
			break;
		case GSTLAL_BURSTTRIGGEN_DOUBLE:
			padbuf = element->maxdatad->pad;
			break;
		case GSTLAL_BURSTTRIGGEN_COMPLEX_DOUBLE_NO_PEAK:
			break;
		case GSTLAL_BURSTTRIGGEN_DOUBLE_NO_PEAK:
			break;
	}
	/* The max size to copy from an adapter is the typical output size plus the padding */
	guint64 maxsize = output_num_bytes(element) + element->adapter->unit_size * padbuf * 2;

	/* if we haven't allocated storage do it now, we should never try to copy from an adapter with a larger buffer than this */
	if (!element->data)
		element->data = (double complex *) malloc(maxsize);
	if (!element->datad)
		element->datad = (double *) malloc(maxsize);

	/* see if the snr thresh on the element agrees with the maxdata, or else update */
	if (element->snr_thresh != element->maxdata->thresh)
		element->maxdata->thresh = element->snr_thresh;
	if (element->snr_thresh != element->maxdatad->thresh)
		element->maxdatad->thresh = element->snr_thresh;

	/*
	 * check validity of timestamp, offsets, tags, bank array
	 */

	if(!GST_BUFFER_TIMESTAMP_IS_VALID(sinkbuf) || !GST_BUFFER_DURATION_IS_VALID(sinkbuf) || !GST_BUFFER_OFFSET_IS_VALID(sinkbuf) || !GST_BUFFER_OFFSET_END_IS_VALID(sinkbuf)) {
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

	if (!element->bankarray) {
		GST_ELEMENT_ERROR(element, STREAM, FAILED, ("missing bank file"), ("must have a valid template bank to create events"));
		result = GST_FLOW_ERROR;
		goto done;
	}

	/* FIXME if we were more careful we wouldn't lose so much data around disconts */
	if (GST_BUFFER_FLAG_IS_SET(sinkbuf, GST_BUFFER_FLAG_DISCONT)) {
		reset_time_and_offset(element);
		gst_audioadapter_clear(element->adapter);
	}


	/* if we don't have a valid first timestamp yet take this one */
	if (element->next_output_timestamp == GST_CLOCK_TIME_NONE) {
		element->next_output_timestamp = GST_BUFFER_TIMESTAMP(sinkbuf);// + output_duration(element);
	}

	/* put the incoming buffer into an adapter, handles gaps */
	gst_audioadapter_push(element->adapter, sinkbuf);

	/* process the data we have */
	process(element);

done:
	gst_object_unref(element);
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
 * Parent class.
 */


static GstElementClass *parent_class = NULL;


/*
 * Instance finalize function.  See ???
 */


static void finalize(GObject *object)
{

	GSTLALBurst_Triggergen *element = GSTLAL_BURST_TRIGGERGEN(object);
	//FIXME make sure everything is freed
	g_mutex_free(element->bank_lock);
	gst_object_unref(element->sinkpad);
	element->sinkpad = NULL;
	gst_object_unref(element->srcpad);
	element->srcpad = NULL;
	g_object_unref(element->adapter);
	if (element->bankarray)
		free_bank(element);
	G_OBJECT_CLASS(parent_class)->finalize(object);
}


/*
 * Base init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GBaseInitFunc
 */


#define CAPS \
	"audio/x-raw-complex, " \
	"rate = (int) [1, MAX], " \
	"channels = (int) [1, MAX], " \
	"endianness = (int) BYTE_ORDER, " \
	"width = (int) {128}; " \
	"audio/x-raw-float, " \
	"rate = (int) [1, MAX], " \
	"channels = (int) [1, MAX], " \
	"endianness = (int) BYTE_ORDER, " \
	"width = (int) {64}; "


static void base_init(gpointer class)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(class);

	gst_element_class_set_details_simple(
		element_class,
		"Burst_Triggergen",
		"Filter",
		"Find burst triggers in snr streams",
		"Chad Hanna <chad.hanna@ligo.org>"
	);

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
			gst_caps_from_string("application/x-lal-snglburst")
		)
	);
}


/*
 * Class init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GClassInitFunc
 */


static void class_init(gpointer class, gpointer class_data)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(class);

	parent_class = g_type_class_ref(GST_TYPE_ELEMENT);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	g_object_class_install_property(
		gobject_class,
		ARG_N,
		g_param_spec_uint(
			"n",
			"n",
			"number of samples over which to identify burst_triggergens",
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
			"Vector of \\sigma^{2} factors (NOT IMPLEMENTED YET).",
			g_param_spec_double(
				"sigmasq",
				"\\sigma^{2}",
				"\\sigma^{2} factor",
				-G_MAXDOUBLE, G_MAXDOUBLE, 0.0,
				G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
			),
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);
}


/*
 * Instance init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GInstanceInitFunc
 */

static void instance_init(GTypeInstance *object, gpointer class)
{
	GSTLALBurst_Triggergen *element = GSTLAL_BURST_TRIGGERGEN(object);
	GstPad *pad;
	gst_element_create_all_pads(GST_ELEMENT(element));

	/* configure (and ref) sink pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "sink");
	gst_pad_set_getcaps_function(pad, GST_DEBUG_FUNCPTR(getcaps));
	gst_pad_set_setcaps_function(pad, GST_DEBUG_FUNCPTR(setcaps));
	gst_pad_set_chain_function(pad, GST_DEBUG_FUNCPTR(chain));
	gst_pad_set_event_function(pad, GST_DEBUG_FUNCPTR(sink_event));
	element->sinkpad = pad;

	/* retrieve (and ref) src pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "src");
	element->srcpad = pad;

	gst_pad_use_fixed_caps(pad);
	{
	GstCaps *caps = gst_caps_copy(gst_pad_get_pad_template_caps(pad));
	gst_pad_set_caps(pad, caps);
	gst_caps_unref(caps);
	}

	/* internal data */
	element->rate = 0;
	element->count = 0;
	reset_time_and_offset(element);
	element->adapter = g_object_new(GST_TYPE_AUDIOADAPTER, NULL);
	element->instrument = NULL;
	element->channel_name = NULL;
	element->bankarray = NULL;
	element->bank_filename = NULL;
	element->event_buffer = NULL;
	element->data = NULL;
	element->maxdata = NULL;
	element->bank_lock = g_mutex_new();
	element->last_gap = TRUE;
	element->EOS = FALSE;
}


/*
 * gstlal_burst_triggergen_get_type().
 */


GType gstlal_burst_triggergen_get_type(void)
{
	static GType type = 0;

	if(!type) {
		static const GTypeInfo info = {
			.class_size = sizeof(GSTLALBurst_TriggergenClass),
			.class_init = class_init,
			.base_init = base_init,
			.instance_size = sizeof(GSTLALBurst_Triggergen),
			.instance_init = instance_init,
		};
		type = g_type_register_static(GST_TYPE_ELEMENT, "GSTLALBurst_Triggergen", &info, 0);
	}

	return type;
}

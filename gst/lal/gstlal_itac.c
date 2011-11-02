/*
 * An inspiral trigger and autocorrelation chisq (itac) element
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

#include <gstlal.h>
#include <gstlal_itac.h>
#include <gstlal_peakfinder.h>
#include <gstaudioadapter.h>
#include <gstlal_tags.h>
#include <gstlal_snglinspiral.h>

#define DEFAULT_SNR_THRESH 5.5

static guint64 output_num_samps(GSTLALItac *element)
{
	return (guint64) element->n;
}


static guint64 output_num_bytes(GSTLALItac *element)
{
	return (guint64) output_num_samps(element) * element->adapter->unit_size;
}


static int reset_time_and_offset(GSTLALItac *element)
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

static void free_bank(GSTLALItac *element)
{
	g_free(element->bank_filename);
	element->bank_filename = NULL;
	free(element->bankarray);
	element->bankarray = NULL;
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

static GstFlowReturn process(GSTLALItac *element);

static gboolean sink_event(GstPad *pad, GstEvent *event)
{
	GSTLALItac *element = GSTLAL_ITAC(GST_PAD_PARENT(pad));
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
			gstlal_set_channel_in_snglinspiral_array(element->bankarray, element->channels, element->channel_name);
			gstlal_set_instrument_in_snglinspiral_array(element->bankarray, element->channels, element->instrument);
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
	ARG_SIGMASQ,
	ARG_AUTOCORRELATION_MATRIX
};


static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec)
{
	GSTLALItac *element = GSTLAL_ITAC(object);

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
		element->channels = gstlal_snglinspiral_array_from_file(element->bank_filename, &(element->bankarray));
		if (element->instrument && element->channel_name) {
			gstlal_set_instrument_in_snglinspiral_array(element->bankarray, element->channels, element->instrument);
			gstlal_set_channel_in_snglinspiral_array(element->bankarray, element->channels, element->channel_name);
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
				gstlal_set_sigmasq_in_snglinspiral_array(element->bankarray, length, sigmasq);
			g_free(sigmasq);
		} else
			GST_WARNING_OBJECT(element, "must set template bank before setting sigmasq");
		g_mutex_unlock(element->bank_lock);
		break;
	}

	case ARG_AUTOCORRELATION_MATRIX: {
		g_mutex_lock(element->bank_lock);

		if(element->autocorrelation_matrix)
			gsl_matrix_complex_free(element->autocorrelation_matrix);

		element->autocorrelation_matrix = gstlal_gsl_matrix_complex_from_g_value_array(g_value_get_boxed(value));

		/*
		 * induce norms to be recomputed
		 */

		if(element->autocorrelation_norm) {
			gsl_vector_free(element->autocorrelation_norm);
			element->autocorrelation_norm = NULL;
		}

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
	GSTLALItac *element = GSTLAL_ITAC(object);

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
		if(element->bankarray) {
			double sigmasq[element->channels];
			gint i;
			for(i = 0; i < (gint) element->channels; i++)
				sigmasq[i] = element->bankarray[i].sigmasq;
			g_value_take_boxed(value, gstlal_g_value_array_from_doubles(sigmasq, element->channels));
		} else {
			GST_WARNING_OBJECT(element, "no template bank");
			g_value_take_boxed(value, g_value_array_new(0));
		}
		g_mutex_unlock(element->bank_lock);
		break;
	}

	case ARG_AUTOCORRELATION_MATRIX:
		g_mutex_lock(element->bank_lock);
		if(element->autocorrelation_matrix)
			g_value_take_boxed(value, gstlal_g_value_array_from_gsl_matrix_complex(element->autocorrelation_matrix));
		/* FIXME:  else? */
		g_mutex_unlock(element->bank_lock);
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
 * getcaps()
 */


static GstCaps *getcaps(GstPad * pad)
{
	GSTLALItac *element = GSTLAL_ITAC(gst_pad_get_parent(pad));
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
	GSTLALItac *element = GSTLAL_ITAC(gst_pad_get_parent(pad));
	GstStructure *structure;
	gint rate, width, channels;
	gboolean success = TRUE;

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

	/*
	 * try setting caps on downstream element
	 */

	if(success)
		success = gst_pad_set_caps(element->srcpad, caps);

	/*
	 * update the element metadata
	 */

	if(success) {
		element->channels = channels;
		element->rate = rate;
		g_object_set(element->adapter, "unit-size", width / 8 * channels, NULL);
		element->maxdata = gstlal_double_complex_peak_samples_and_values_new(channels);
		//FIXME get this number from the autocorrelation matrix size!!!
		element->maxdata->pad = 5;
		//FIXME set this only once we have the autocorrelation matrix!!
		element->snr_mat = gsl_matrix_complex_calloc(element->channels, element->maxdata->pad);
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

static void update_state(GSTLALItac *element, GstBuffer *srcbuf)
{
	element->next_output_offset = GST_BUFFER_OFFSET_END(srcbuf);
	element->next_output_timestamp = GST_BUFFER_TIMESTAMP(srcbuf) + GST_BUFFER_DURATION(srcbuf);
}

static GstFlowReturn push_buffer(GSTLALItac *element, GstBuffer *srcbuf)
{
	GST_DEBUG_OBJECT(element, "pushing %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(srcbuf));
	return gst_pad_push(element->srcpad, srcbuf);
}

static GstFlowReturn push_gap(GSTLALItac *element, guint samps)
{
	GstBuffer *srcbuf = NULL;
	GstFlowReturn result = GST_FLOW_OK;
	/* Clearing the max data structure causes the resulting buffer to be a GAP */
	gstlal_double_complex_peak_samples_and_values_clear(element->maxdata);
	/* create the output buffer */
	srcbuf = gstlal_snglinspiral_new_buffer_from_peak(element->maxdata, element->bankarray, element->srcpad, element->next_output_offset, samps, element->next_output_timestamp, element->rate);
	/* set the time stamp and offset state */
	update_state(element, srcbuf);
	/* push the result */
	result = push_buffer(element, srcbuf);
	return result;
}

static GstFlowReturn push_nongap(GSTLALItac *element, guint copysamps, guint outsamps)
{
	GstBuffer *srcbuf = NULL;
	GstFlowReturn result = GST_FLOW_OK;
	gint copied_gap, copied_nongap;
	double complex *dataptr = NULL;

	/* call the peak finding library on a buffer from the adapter if no events are found the result will be a GAP */
	gst_audioadapter_copy(element->adapter, (void *) element->data, copysamps, &copied_gap, &copied_nongap);
	/* put the data pointer one pad length in */
	dataptr = element->data + element->maxdata->pad * element->maxdata->channels;
	/* Find the peak */
	gstlal_double_complex_peak_over_window(element->maxdata, (const double complex*) dataptr, outsamps);
	/* extract data around peak for chisq calculation */
	gstlal_double_complex_series_around_peak(element->maxdata, element->data, element->snr_mat, copysamps);

	/* create the output buffer */
	srcbuf = gstlal_snglinspiral_new_buffer_from_peak(element->maxdata, element->bankarray, element->srcpad, element->next_output_offset, outsamps, element->next_output_timestamp, element->rate);
	/* set the time stamp and offset state */
	update_state(element, srcbuf);
	/* push the result */
	result = push_buffer(element, srcbuf);
	return result;
}

static GstFlowReturn process(GSTLALItac *element)
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
			gst_audioadapter_flush(element->adapter, outsamps);
			}
		/* The check to see if we have enough nongap samples to compute an output, else it is a gap too */
		else if (nongapsamps <= 2 * element->maxdata->pad) {
			element->last_gap = TRUE;
			outsamps = nongapsamps;
			result = push_gap(element, outsamps);
			/* knock off the first buffers worth of bytes since we don't need them any more */
			gst_audioadapter_flush(element->adapter, outsamps);
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
			gst_audioadapter_flush(element->adapter, outsamps);

			/* We are on another gap boundary so push the end transient as a gap */
			if (copysamps == nongapsamps) {
				element->last_gap = FALSE;
				if (element->maxdata->pad > 0) {
					result = push_gap(element, element->maxdata->pad);
					gst_audioadapter_flush(element->adapter, 2 * element->maxdata->pad);
					}
				}
			}
		}

	return result;
}

static GstFlowReturn chain(GstPad *pad, GstBuffer *sinkbuf)
{
	GSTLALItac *element = GSTLAL_ITAC(gst_pad_get_parent(pad));
	GstFlowReturn result = GST_FLOW_OK;
	/* The max size to copy from an adapter is the typical output size plus the padding */
	guint64 maxsize = output_num_bytes(element) + element->adapter->unit_size * element->maxdata->pad * 2;

	/* if we haven't allocated storage do it now, we should never try to copy from an adapter with a larger buffer than this */
	if (!element->data)
		element->data = (double complex *) malloc(maxsize);

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

	GSTLALItac *element = GSTLAL_ITAC(object);
	//FIXME make sure everything is freed
	g_mutex_free(element->bank_lock);
	gst_object_unref(element->sinkpad);
	element->sinkpad = NULL;
	gst_object_unref(element->srcpad);
	element->srcpad = NULL;
	g_object_unref(element->adapter);
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
	"width = (int) {128}; "


static void base_init(gpointer class)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(class);

	gst_element_class_set_details_simple(
		element_class,
		"Itac",
		"Filter",
		"Find inspiral triggers in snr streams",
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
			gst_caps_from_string("application/x-lal-snglinspiral")
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
			"Vector of \\sigma^{2} factors.",
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
	GSTLALItac *element = GSTLAL_ITAC(object);
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
	reset_time_and_offset(element);
	element->adapter = g_object_new(GST_TYPE_AUDIOADAPTER, NULL);
	element->instrument = NULL;
	element->channel_name = NULL;
	element->bankarray = NULL;
	element->bank_filename = NULL;
	element->data = NULL;
	element->maxdata = NULL;
	element->bank_lock = g_mutex_new();
	element->last_gap = TRUE;
	element->EOS = FALSE;
	element->snr_mat = NULL;
}


/*
 * gstlal_itac_get_type().
 */


GType gstlal_itac_get_type(void)
{
	static GType type = 0;

	if(!type) {
		static const GTypeInfo info = {
			.class_size = sizeof(GSTLALItacClass),
			.class_init = class_init,
			.base_init = base_init,
			.instance_size = sizeof(GSTLALItac),
			.instance_init = instance_init,
		};
		type = g_type_register_static(GST_TYPE_ELEMENT, "lal_itac", &info, 0);
	}

	return type;
}

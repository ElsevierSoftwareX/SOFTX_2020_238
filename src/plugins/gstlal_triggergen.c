/*
 * Copyright (C) 2009 Kipp Cannon <kipp.cannon@ligo.org>, Chad Hanna
 * <chad.hanna@ligo.caltech.edu>
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


#include <complex.h>
#include <math.h>
#include <glib.h>
#include <gst/gst.h>
#include <gsl/gsl_matrix.h>
#include <gst/base/gstbasesink.h>
#include <gstlal.h>
#include <gstlal_triggergen.h>
#include <lal/Date.h>
#include <lal/LIGOMetadataTables.h>
#include <lal/LIGOLwXML.h>
#include <lal/LIGOLwXMLRead.h>
#include <lal/LALStdlib.h>


/*
 * ============================================================================
 *
 *                             Trigger Generator
 *
 * ============================================================================
 */


/*
 * ============================================================================
 *
 *                                 Parameters
 *
 * ============================================================================
 */


#define DEFAULT_SNR_THRESH 5.5
#define DEFAULT_MAX_GAP 0.01


/*
 * ============================================================================
 *
 *                                 Utilities
 *
 * ============================================================================
 */


static double eta(double m1, double m2)
{
	return m1 * m2 / pow(m1 + m2, 2);
}


static double mchirp(double m1, double m2)
{
	return pow(m1 * m2, 0.6) / pow(m1 + m2, 0.2);
}


static double effective_distance(double snr, double sigmasq)
{
	return sqrt(sigmasq) / snr;
}


static int setup_bankfile_input(GSTLALTriggerGen *element, char *bank_filename)
{
	SnglInspiralTable *bank = NULL;
	int i;

	element->bank_filename = bank_filename;
	element->num_templates = LALSnglInspiralTableFromLIGOLw(&bank, element->bank_filename, -1, -1);
	element->bank = calloc(element->num_templates, sizeof(*element->bank));
	element->last_event = calloc(element->num_templates, sizeof(*element->last_event));
	element->last_time = calloc(element->num_templates, sizeof(*element->last_time));
	if(!bank || !element->bank || !element->last_event || !element->last_time) {
		while(bank) {
			SnglInspiralTable *next = bank->next;
			free(bank);
			bank = next;
		}
		free(element->bank);
		element->bank = NULL;
		free(element->last_event);
		element->last_event = NULL;
		free(element->last_time);
		element->last_time = NULL;
		return -1;
	}

	/*
	 * copy the linked list of templates constructed by
	 * LALSnglInspiralTableFromLIGOLw() into the template array.  the
	 * end_time and snr are 0'ed so that when the templates are used to
	 * initialize the last_event info those fields are set properly.
	 * also sigmasq is 0'ed to "disable" effective distance calculation
	 * unless explicit values are provided.  while we're at it,
	 * initialize the last_time array.
	 */

	for(i = 0; bank; i++) {
		SnglInspiralTable *prev = bank;
		element->bank[i] = *bank;
		element->bank[i].next = NULL;
		element->bank[i].end_time = (LIGOTimeGPS) {0, 0};
		element->bank[i].snr = 0;
		element->bank[i].sigmasq = 0;
		bank = bank->next;
		free(prev);

		/* fix some buggered columns.  sigh. */
		element->bank[i].mtotal = element->bank[i].mass1 + element->bank[i].mass2;
		element->bank[i].mchirp = mchirp(element->bank[i].mass1, element->bank[i].mass2);
		element->bank[i].eta = eta(element->bank[i].mass1, element->bank[i].mass2);

		element->last_time[i] = (LIGOTimeGPS) {0, 0};
	}

	return element->num_templates;
}


static void free_bankfile(GSTLALTriggerGen *element)
{
	free(element->bank_filename);
	element->bank_filename = NULL;
	free(element->bank);
	element->bank = NULL;
	free(element->last_event);
	element->last_event = NULL;
	free(element->last_time);
	element->last_time = NULL;
	element->num_templates = 0;
}


static SnglInspiralTable *new_event(SnglInspiralTable *dest, LIGOTimeGPS end_time, double complex z, double chisq, int channel, GSTLALTriggerGen *element)
{
	double xi;

	*dest = element->bank[channel];

	dest->snr = cabs(z);
	dest->coa_phase = carg(z);
	dest->chisq = chisq;
	dest->chisq_dof = 1;
	dest->end_time = end_time;
	dest->end_time_gmst = XLALGreenwichMeanSiderealTime(&end_time);
	dest->eff_distance = effective_distance(dest->snr, dest->sigmasq);

	xi = dest->chisq / (1 + 0.1 * pow(dest->snr, 2));

	return dest;
}


/*
 * ============================================================================
 *
 *                                    Caps
 *
 * ============================================================================
 */


static gboolean gen_setcaps(GstPad *pad, GstCaps *caps)
{
	GSTLALTriggerGen *element = GSTLAL_TRIGGERGEN(gst_pad_get_parent(pad));
	GstStructure *structure;
	gint rate, width, channels;
	gboolean success = TRUE;

	GST_OBJECT_LOCK(element);

	structure = gst_caps_get_structure(caps, 0);
	if(!gst_structure_get_int(structure, "rate", &rate))
		success = FALSE;
	if(!gst_structure_get_int(structure, "width", &width))
		success = FALSE;
	if(!gst_structure_get_int(structure, "channels", &channels))
		success = FALSE;

	if(success) {
		element->rate = rate;
		gstlal_collect_pads_set_unit_size(pad, (width / 8) * channels);
	}

	GST_OBJECT_UNLOCK(element);
	gst_object_unref(element);
	return success;
}


/*
 * ============================================================================
 *
 *                              Event Generation
 *
 * ============================================================================
 */


static GstFlowReturn gen_collected(GstCollectPads *pads, gpointer user_data)
{
	GSTLALTriggerGen *element = GSTLAL_TRIGGERGEN(user_data);
	guint64 earliest_input_offset, earliest_input_offset_end;
	GstBuffer *snrbuf = NULL;
	GstBuffer *chisqbuf = NULL;
	GstBuffer *srcbuf = NULL;
	GstFlowReturn result;

	/*
	 * check for new segment
	 */

	if(element->segment_pending) {
		GstEvent *event;

		/*
		 * update the segment boundary and timestamp book-keeping
		 * information
		 */

		GstSegment *segment = gstlal_collect_pads_get_segment(element->collect);
		if(!segment) {
			/* FIXME:  failure getting bounding segment, do
			 * something about it */
		}
		element->segment = *segment;
		gst_segment_free(segment);
		element->next_output_offset = 0;
		element->next_output_timestamp = GST_CLOCK_TIME_NONE;

		/*
		 * transmit the new-segment event downstream
		 */

		event = gst_event_new_new_segment_full(FALSE, element->segment.rate, 1.0, GST_FORMAT_TIME, element->segment.start, element->segment.stop, element->segment.start);
		if(!event) {
			/* FIXME:  failure building event, do something
			 * about it */
		}
		gst_pad_push_event(element->srcpad, event);

		/*
		 * reset the last inspiral event information
		 */

		g_mutex_lock(element->bank_lock);
		memcpy(element->last_event, element->bank, element->num_templates * sizeof(*element->last_event));
		g_mutex_unlock(element->bank_lock);

		element->segment_pending = FALSE;
	}

	/*
	 * get the range of offsets (in the output stream) spanned by the
	 * available input buffers.
	 */

	if(!gstlal_collect_pads_get_earliest_offsets(element->collect, &earliest_input_offset, &earliest_input_offset_end, element->segment.start, 0, element->rate)) {
		GST_ERROR_OBJECT(element, "cannot deduce input timestamp offset information");
		result = GST_FLOW_ERROR;
		goto error;
	}

	/*
	 * check for EOS
	 */

	if(earliest_input_offset == GST_BUFFER_OFFSET_NONE)
		goto eos;

	/*
	 * get buffers upto the desired end offset.
	 */

	snrbuf = gstlal_collect_pads_take_buffer(pads, element->snrcollectdata, earliest_input_offset_end, element->segment.start, 0, element->rate);
	chisqbuf = gstlal_collect_pads_take_buffer(pads, element->chisqcollectdata, earliest_input_offset_end, element->segment.start, 0, element->rate);

	/*
	 * NULL means EOS.  EOS on one means our EOS.
	 */

	if(!snrbuf || !chisqbuf) {
		if(snrbuf) {
			gst_buffer_unref(snrbuf);
			snrbuf = NULL;
		}
		if(chisqbuf) {
			gst_buffer_unref(chisqbuf);
			chisqbuf = NULL;
		}
		goto eos;
	}

	/*
	 * Construct output buffer.  timestamp is earliest of the two input
	 * timestamps, and end time is last of the two input end times.
	 */

	if(GST_BUFFER_FLAG_IS_SET(snrbuf, GST_BUFFER_FLAG_GAP) || GST_BUFFER_FLAG_IS_SET(chisqbuf, GST_BUFFER_FLAG_GAP)) {
		/*
		 * GAP --> no-op
		 */

		result = gst_pad_alloc_buffer(element->srcpad, element->next_output_offset, 0, GST_PAD_CAPS(element->srcpad), &srcbuf);
		if(result != GST_FLOW_OK) {
			/* FIXME: handle failure */
		}
		GST_BUFFER_OFFSET(srcbuf) = GST_BUFFER_OFFSET_END(srcbuf) = element->next_output_offset;
		GST_BUFFER_FLAG_SET(srcbuf, GST_BUFFER_FLAG_GAP);
	} else {
		/*
		 * !GAP --> find events
		 */

		const double complex *snrdata = (const double complex *) GST_BUFFER_DATA(snrbuf);
		const double *chisqdata = (const double *) GST_BUFFER_DATA(chisqbuf);
		guint64 t0;
		guint64 length;
		guint sample;
		gint channel;
		SnglInspiralTable *head = NULL;
		guint nevents = 0;

		length = MIN(GST_BUFFER_OFFSET_END(snrbuf), GST_BUFFER_OFFSET_END(chisqbuf));
		if(GST_BUFFER_OFFSET(snrbuf) > GST_BUFFER_OFFSET(chisqbuf)) {
			t0 = GST_BUFFER_TIMESTAMP(snrbuf);
			chisqdata += (GST_BUFFER_OFFSET(snrbuf) - GST_BUFFER_OFFSET(chisqbuf)) * element->num_templates;
			length -= GST_BUFFER_OFFSET(snrbuf);
		} else {
			t0 = GST_BUFFER_TIMESTAMP(chisqbuf);
			snrdata += (GST_BUFFER_OFFSET(chisqbuf) - GST_BUFFER_OFFSET(snrbuf)) * element->num_templates;
			length -= GST_BUFFER_OFFSET(chisqbuf);
		}

		g_mutex_lock(element->bank_lock);
		for(sample = 0; sample < length; sample++) {
			LIGOTimeGPS t;
			XLALINT8NSToGPS(&t, t0);
			XLALGPSAdd(&t, (double) sample / element->rate);

			for(channel = 0; channel < element->num_templates; channel++) {
				if(cabs(*snrdata) >= element->snr_thresh) {
					if(XLALGPSDiff(&t, &element->last_time[channel]) > element->max_gap) {
						/*
						 * New event.  prepend last
						 * event to event list and
						 * start a new one.
						 */
						if(element->last_event[channel].snr != 0) {
							SnglInspiralTable *new = calloc(1, sizeof(*new));
							*new = element->last_event[channel];
							new->next = head;
							head = new;
							nevents++;
						}
						new_event(&element->last_event[channel], t, *snrdata, *chisqdata, channel, element);
					} else if(cabs(*snrdata) > element->last_event[channel].snr) {
						/*
						 * Same event, higher SNR,
						 * update
						 */
						new_event(&element->last_event[channel], t, *snrdata, *chisqdata, channel, element);
					} else {
						/*
						 * Same event, not higher
						 * SNR, do nothing
						 */
					}
					element->last_time[channel] = t;
				}

				snrdata++;
				chisqdata++;
			}
		}
		g_mutex_unlock(element->bank_lock);

		if(nevents) {
			SnglInspiralTable *dest;

			result = gst_pad_alloc_buffer(element->srcpad, element->next_output_offset, nevents * sizeof(*head), GST_PAD_CAPS(element->srcpad), &srcbuf);
			if(result != GST_FLOW_OK) {
				/* FIXME: handle failure */
			}
			GST_BUFFER_OFFSET(srcbuf) = element->next_output_offset;
			element->next_output_offset += nevents;
			GST_BUFFER_OFFSET_END(srcbuf) = element->next_output_offset;

			/*
			 * this loop puts them in the buffer in time order.
			 * when loop terminates, dest points to first event
			 * slot preceding the buffer, so the last event in
			 * the buffer is at offset nevents not nevents-1
			 */

			for(dest = ((SnglInspiralTable *) GST_BUFFER_DATA(srcbuf)) + nevents - 1; head; dest--) {
				SnglInspiralTable *next = head->next;
				memcpy(dest, head, sizeof(*head));
				dest->next = dest + 1;
				free(head);
				head = next;
			}
			dest[nevents].next = NULL;
		} else {
			result = gst_pad_alloc_buffer(element->srcpad, element->next_output_offset, 0, GST_PAD_CAPS(element->srcpad), &srcbuf);
			if(result != GST_FLOW_OK) {
				/* FIXME: handle failure */
			}
			GST_BUFFER_OFFSET(srcbuf) = GST_BUFFER_OFFSET_END(srcbuf) = element->next_output_offset;
			GST_BUFFER_FLAG_SET(srcbuf, GST_BUFFER_FLAG_GAP);
		}
	}

	/*
	 * Push buffer downstream
	 */

	GST_BUFFER_TIMESTAMP(srcbuf) = MAX(GST_BUFFER_TIMESTAMP(snrbuf), GST_BUFFER_TIMESTAMP(chisqbuf));
	GST_BUFFER_DURATION(srcbuf) = MIN(GST_BUFFER_TIMESTAMP(snrbuf) + GST_BUFFER_DURATION(snrbuf), GST_BUFFER_TIMESTAMP(chisqbuf) + GST_BUFFER_DURATION(chisqbuf)) - GST_BUFFER_TIMESTAMP(srcbuf);

	if(element->next_output_timestamp != GST_BUFFER_TIMESTAMP(srcbuf))
		GST_BUFFER_FLAG_SET(srcbuf, GST_BUFFER_FLAG_DISCONT);
	element->next_output_timestamp = GST_BUFFER_TIMESTAMP(srcbuf) + GST_BUFFER_DURATION(srcbuf);

	gst_buffer_unref(snrbuf);
	gst_buffer_unref(chisqbuf);

	return gst_pad_push(element->srcpad, srcbuf);

	/*
	 * Errors
	 */


error:
	if(snrbuf)
		gst_buffer_unref(snrbuf);
	if(chisqbuf)
		gst_buffer_unref(chisqbuf);
	if(srcbuf)
		gst_buffer_unref(srcbuf);
	return result;

eos:
	GST_DEBUG_OBJECT(element, "no data available (EOS)");
	gst_pad_push_event(element->srcpad, gst_event_new_eos());
	return GST_FLOW_UNEXPECTED;
}


/*
 * ============================================================================
 *
 *                                 Properties
 *
 * ============================================================================
 */


enum gen_property {
	ARG_SNR_THRESH = 1, 
	ARG_BANK_FILENAME,
	ARG_MAX_GAP,
	ARG_SIGMASQ
};


static void gen_set_property(GObject *object, enum gen_property id, const GValue *value, GParamSpec *pspec)
{
	GSTLALTriggerGen *element = GSTLAL_TRIGGERGEN(object);

	GST_OBJECT_LOCK(element);
	switch(id) {
	case ARG_SNR_THRESH:
		element->snr_thresh = g_value_get_double(value);
		break;

	case ARG_BANK_FILENAME:
		g_mutex_lock(element->bank_lock);
		free_bankfile(element);
		setup_bankfile_input(element, g_value_dup_string(value));
		g_mutex_unlock(element->bank_lock);
		break;

	case ARG_MAX_GAP:
		element->max_gap = g_value_get_double(value);
		break;

	case ARG_SIGMASQ: {
		g_mutex_lock(element->bank_lock);
		if(element->bank) {
			gint length;
			double *sigmasq = gstlal_doubles_from_g_value_array(g_value_get_boxed(value), NULL, &length);
			if(element->num_templates != length)
				GST_ERROR_OBJECT(element, "vector length (%d) does not match number of templates (%d)", length, element->num_templates);
			else
				while(length--) {
					element->bank[length].sigmasq = sigmasq[length];
					if(element->last_event)
						element->last_event[length].sigmasq = sigmasq[length];
				}
			g_free(sigmasq);
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


static void gen_get_property(GObject * object, enum gen_property id, GValue * value, GParamSpec * pspec)
{
	GSTLALTriggerGen *element = GSTLAL_TRIGGERGEN(object);

	GST_OBJECT_LOCK(element);
	switch(id) {
	case ARG_SNR_THRESH:
		g_value_set_double(value, element->snr_thresh);
		break;

	case ARG_BANK_FILENAME:
		g_mutex_lock(element->bank_lock);
		g_value_set_string(value, element->bank_filename);
		g_mutex_unlock(element->bank_lock);
		break;

	case ARG_MAX_GAP:
		g_value_set_double(value, element->max_gap);
		break;

	case ARG_SIGMASQ: {
		g_mutex_lock(element->bank_lock);
		if(element->bank) {
			double sigmasq[element->num_templates];
			gint i;
			for(i = 0; i < element->num_templates; i++)
				sigmasq[i] = element->bank[i].sigmasq;
			g_value_take_boxed(value, gstlal_g_value_array_from_doubles(sigmasq, element->num_templates));
		} else
			g_value_take_boxed(value, g_value_array_new(0));
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
 *                              Element Support
 *
 * ============================================================================
 */


static GstElementClass *gen_parent_class = NULL;


static void gen_finalize(GObject *object)
{
	GSTLALTriggerGen *element = GSTLAL_TRIGGERGEN(object);
	g_mutex_free(element->bank_lock);
	element->bank_lock = NULL;
	free_bankfile(element);
	G_OBJECT_CLASS(gen_parent_class)->finalize(object);
}


static GstStateChangeReturn gen_change_state(GstElement *element, GstStateChange transition)
{
	GSTLALTriggerGen *triggergen = GSTLAL_TRIGGERGEN(element);

	switch(transition) {
	case GST_STATE_CHANGE_NULL_TO_READY:
		break;

	case GST_STATE_CHANGE_READY_TO_PAUSED:
		triggergen->segment_pending = TRUE;
		gst_segment_init(&triggergen->segment, GST_FORMAT_UNDEFINED);
		triggergen->next_output_offset = GST_BUFFER_OFFSET_NONE;
		triggergen->next_output_timestamp = GST_CLOCK_TIME_NONE;
		gst_collect_pads_start(triggergen->collect);
		break;

	case GST_STATE_CHANGE_PAUSED_TO_PLAYING:
		break;

	case GST_STATE_CHANGE_PAUSED_TO_READY:
		/* need to unblock the collectpads before calling the
		 * parent change_state so that streaming can finish */
		gst_collect_pads_stop(triggergen->collect);
		break;

	default:
		break;
	}

	return gen_parent_class->change_state(element, transition);
}


static void gen_base_init(gpointer g_class)
{
	static GstElementDetails plugin_details = {
		"Trigger Generator",
		"Filter",
		"SNR and \\chi^{2} in, Triggers out",
		"Kipp Cannon <kipp.cannon@ligo.org>, Chad Hanna <channa@ligo.caltech.edu>"
	};

	GstElementClass *element_class = GST_ELEMENT_CLASS(g_class);
	gst_element_class_set_details (element_class, &plugin_details);

	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"snr",
			GST_PAD_SINK,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
				"audio/x-raw-complex, " \
				"rate = (int) [1, MAX], " \
				"channels = (int) [1, MAX], " \
				"endianness = (int) BYTE_ORDER, " \
				"width = (int) 128"
			)
		)
	);

	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"chisquare",
			GST_PAD_SINK,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
				"audio/x-raw-float, " \
				"rate = (int) [1, MAX], " \
				"channels = (int) [1, MAX], " \
				"endianness = (int) BYTE_ORDER, " \
				"width = (int) 64"
			)
		)
	);

	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"src",
			GST_PAD_SRC,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
				"application/x-lal-snglinspiral"
			)
		)
	);
}


static void gen_class_init(gpointer klass, gpointer class_data)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *gstelement_class = GST_ELEMENT_CLASS(klass);

	gen_parent_class = g_type_class_ref(GST_TYPE_ELEMENT);
	gobject_class->set_property = GST_DEBUG_FUNCPTR(gen_set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(gen_get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(gen_finalize);
	gstelement_class->change_state = GST_DEBUG_FUNCPTR(gen_change_state);

	g_object_class_install_property(
		gobject_class,
		ARG_BANK_FILENAME,
		g_param_spec_string(
			"bank-filename",
			"Bank file name",
			"Path to XML file used to generate the template bank.  Setting this property resets sigma to a vector of 0s.",
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
		ARG_MAX_GAP,
		g_param_spec_double(
			"max-gap",
			"Maximum below-threshold gap (seconds)",
			"If the SNR drops below threshold for less than this interval (in seconds) then it is not the start of a new event.",
			0, G_MAXDOUBLE, DEFAULT_MAX_GAP,
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


static void gen_instance_init(GTypeInstance *object, gpointer klass)
{
	GSTLALTriggerGen *element = GSTLAL_TRIGGERGEN(object);
	GstPad *pad;

	gst_element_create_all_pads(GST_ELEMENT(element));
	element->collect = gst_collect_pads_new();
	gst_collect_pads_set_function(element->collect, GST_DEBUG_FUNCPTR(gen_collected), element);

	/* configure snr pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "snr");
	gst_pad_set_setcaps_function(pad, GST_DEBUG_FUNCPTR(gen_setcaps));
	gst_pad_use_fixed_caps(pad);
	element->snrcollectdata = gstlal_collect_pads_add_pad(element->collect, pad, sizeof(*element->snrcollectdata));
	element->snrpad = pad;

	/* configure chisquare pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "chisquare");
	gst_pad_set_setcaps_function(pad, GST_DEBUG_FUNCPTR(gen_setcaps));
	gst_pad_use_fixed_caps(pad);
	element->chisqcollectdata = gstlal_collect_pads_add_pad(element->collect, pad, sizeof(*element->chisqcollectdata));
	element->chisqpad = pad;

	/* retrieve (and ref) src pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "src");
	element->srcpad = pad;

	/* internal data */
	element->bank_lock = g_mutex_new();
	element->rate = 0;
	element->bank_filename = NULL;
	element->bank = NULL;
	element->num_templates = 0;
	element->snr_thresh = DEFAULT_SNR_THRESH;
	element->max_gap = DEFAULT_MAX_GAP;
	element->last_event = NULL;
	element->last_time = NULL;
}


GType gstlal_triggergen_get_type(void)
{
	static GType type = 0;

	if(!type) {
		static const GTypeInfo info = {
			.class_size = sizeof(GSTLALTriggerGenClass),
			.class_init = gen_class_init,
			.base_init = gen_base_init,
			.instance_size = sizeof(GSTLALTriggerGen),
			.instance_init = gen_instance_init,
		};
		type = g_type_register_static(GST_TYPE_ELEMENT, "lal_triggergen", &info, 0);
	}

	return type;
}


/*
 * ============================================================================
 *
 *                             Trigger XML Writer
 *
 * ============================================================================
 */


static gboolean xmlwriter_start(GstBaseSink *sink)
{
	GSTLALTriggerXMLWriter *element = GSTLAL_TRIGGERXMLWRITER(sink);
	LALStatus status;
	memset(&status, 0, sizeof(status));

	g_assert(element->location != NULL);

	element->xml = XLALOpenLIGOLwXMLFile(element->location);
	LALBeginLIGOLwXMLTable(&status, element->xml, sngl_inspiral_table);

	return TRUE;
}


static gboolean xmlwriter_stop(GstBaseSink *sink)
{
	GSTLALTriggerXMLWriter *element = GSTLAL_TRIGGERXMLWRITER(sink);
	LALStatus status;
	memset(&status, 0, sizeof(status));

	if(element->xml) {
		LALEndLIGOLwXMLTable(&status, element->xml);
		XLALCloseLIGOLwXMLFile(element->xml);
		element->xml = NULL;
	}

	return TRUE;
}


static GstFlowReturn xmlwriter_render(GstBaseSink *sink, GstBuffer *buffer)
{
	GSTLALTriggerXMLWriter *element = GSTLAL_TRIGGERXMLWRITER(sink);
	LALStatus status;
	MetadataTable table = {
		.snglInspiralTable = (SnglInspiralTable *) GST_BUFFER_DATA(buffer)
	};
	memset(&status, 0, sizeof(status));

	if(!GST_BUFFER_FLAG_IS_SET(buffer, GST_BUFFER_FLAG_GAP))
		LALWriteLIGOLwXMLTable(&status, element->xml, table, sngl_inspiral_table);

	return GST_FLOW_OK;
}


enum xmlwriter_property {
	ARG_LOCATION = 1
};


static void xmlwriter_set_property(GObject * object, enum xmlwriter_property id, const GValue * value, GParamSpec * pspec)
{
	GSTLALTriggerXMLWriter *element = GSTLAL_TRIGGERXMLWRITER(object);

	GST_OBJECT_LOCK(element);
	switch(id) {
	case ARG_LOCATION:
		free(element->location);
		element->location = g_value_dup_string(value);
		break;
	}
	GST_OBJECT_UNLOCK(element);
}


static void xmlwriter_get_property(GObject * object, enum xmlwriter_property id, GValue * value, GParamSpec * pspec)
{
	GSTLALTriggerXMLWriter *element = GSTLAL_TRIGGERXMLWRITER(object);

	GST_OBJECT_LOCK(element);
	switch(id) {
	case ARG_LOCATION:
		g_value_set_string(value,element->location);
		break;
	}
	GST_OBJECT_UNLOCK(element);
}


static GstBaseSink *xmlwriter_parent_class = NULL;


static void xmlwriter_base_init(gpointer g_class)
{
	static GstElementDetails plugin_details = {
		"Trigger XML Writer",
		"Sink/File",
		"Writes LAL's SnglInspiralTable C structures to an XML file",
		"Kipp Cannon <kipp.cannon@ligo.org>, Chad Hanna <channa@ligo.caltech.edu>"
	};

	GstElementClass *element_class = GST_ELEMENT_CLASS(g_class);
	gst_element_class_set_details (element_class, &plugin_details);

	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"sink",
			GST_PAD_SINK,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
				"application/x-lal-snglinspiral"
			)
		)
	);
}


static void xmlwriter_class_init(gpointer klass, gpointer class_data)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstBaseSinkClass *gstbasesink_class = GST_BASE_SINK_CLASS(klass);

	xmlwriter_parent_class = g_type_class_ref(GST_TYPE_BASE_SINK);
	gobject_class->set_property = GST_DEBUG_FUNCPTR(xmlwriter_set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(xmlwriter_get_property);
	gstbasesink_class->start = GST_DEBUG_FUNCPTR(xmlwriter_start);
	gstbasesink_class->stop = GST_DEBUG_FUNCPTR(xmlwriter_stop);
	gstbasesink_class->render = GST_DEBUG_FUNCPTR(xmlwriter_render);

	g_object_class_install_property(
		gobject_class,
		ARG_LOCATION,
		g_param_spec_string(
			"location",
			"filename",
			"Path to output file",
			NULL,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);
}


static void xmlwriter_instance_init(GTypeInstance *object, gpointer klass)
{
	GSTLALTriggerXMLWriter *element = GSTLAL_TRIGGERXMLWRITER(object);

	/*
	 * Internal data
	 */

	element->location = NULL;
	element->xml = NULL;
}


GType gstlal_triggerxmlwriter_get_type(void)
{
	static GType type = 0;

	if(!type) {
		static const GTypeInfo info = {
			.class_size = sizeof(GSTLALTriggerXMLWriterClass),
			.class_init = xmlwriter_class_init,
			.base_init = xmlwriter_base_init,
			.instance_size = sizeof(GSTLALTriggerXMLWriter),
			.instance_init = xmlwriter_instance_init,
		};
		type = g_type_register_static(GST_TYPE_BASE_SINK, "lal_triggerxmlwriter", &info, 0);
	}

	return type;
}

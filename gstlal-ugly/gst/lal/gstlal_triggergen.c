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
#include <gstlal/gstlal.h>
#include <gstlal/gstlal_debug.h>
#include <gstlal/gstlal_tags.h>
#include <gstlal_triggergen.h>
#include <lal/Date.h>
#include <lal/LIGOMetadataTables.h>
#include <lal/LIGOLwXMLInspiralRead.h>
#include <lal/LALStdlib.h>


#define GST_CAT_DEFAULT gstlal_triggergen_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


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


static void free_bankfile(GSTLALTriggerGen *element)
{
	g_free(element->bank_filename);
	element->bank_filename = NULL;
	free(element->bank);
	element->bank = NULL;
	free(element->last_event);
	element->last_event = NULL;
	free(element->last_time);
	element->last_time = NULL;
	element->num_templates = 0;
}


static int setup_bankfile_input(GSTLALTriggerGen *element, char *bank_filename)
{
	SnglInspiralTable *bank = NULL;
	int i;

	free_bankfile(element);

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
	 * LALSnglInspiralTableFromLIGOLw() into the template array.
	 */

	for(i = 0; bank; i++) {
		SnglInspiralTable *next = bank->next;
		g_assert(i < element->num_templates);
		element->bank[i] = *bank;
		element->bank[i].next = NULL;
		free(bank);
		bank = next;

		/*
		 * initialize data in template.  the snr is 0'ed so that
		 * when the templates are used to initialize the last_event
		 * info that field is set properly.  sigmasq is 0'ed to
		 * "disable" effective distance calculation unless a vector
		 * of values are provided via the sigmasq element property.
		 */

		element->bank[i].snr = 0;
		element->bank[i].sigmasq = 0;

		/*
		 * fix some buggered columns.  sigh.
		 */

		element->bank[i].mtotal = element->bank[i].mass1 + element->bank[i].mass2;
		element->bank[i].mchirp = mchirp(element->bank[i].mass1, element->bank[i].mass2);
		element->bank[i].eta = eta(element->bank[i].mass1, element->bank[i].mass2);

		/*
		 * initialize the last_time array, too
		 */

		element->last_time[i] = (LIGOTimeGPS) {0, 0};
	}

	return element->num_templates;
}


static SnglInspiralTable *record_inspiral_event(SnglInspiralTable *dest, LIGOTimeGPS end_time, double complex z, double chisq, int channel, GSTLALTriggerGen *element)
{
	/*
	 * copy the template whole
	 */

	*dest = element->bank[channel];

	/*
	 * replace the instrument and channel name.  note:  in the
	 * traditional inspiral pipeline the instrument and channel name
	 * are encoded in the templates, but we want to accomodate a
	 * scenario in which the same template bank is used across multiple
	 * instruments, so we need to replace the information stored in the
	 * templates with what we know about the data stream
	 */

	strncpy(dest->ifo, element->instrument, LIGOMETA_IFO_MAX * sizeof(*dest->ifo));
	dest->ifo[LIGOMETA_IFO_MAX - 1] = 0;
	strncpy(dest->channel, element->channel_name, LIGOMETA_CHANNEL_MAX * sizeof(*dest->channel));
	dest->channel[LIGOMETA_CHANNEL_MAX - 1] = 0;

	/*
	 * fill in the rest of the information.  the template end_time is
	 * treated as an offset that is added to the end time recorded from
	 * the filter output.
	 */

	dest->snr = cabs(z);
	dest->coa_phase = carg(z);
	dest->chisq = chisq;
	dest->chisq_dof = 1;
	XLALGPSAddGPS(&dest->end_time, &end_time);
	dest->end_time_gmst = XLALGreenwichMeanSiderealTime(&end_time);
	dest->eff_distance = effective_distance(dest->snr, dest->sigmasq);

	/*
	 * done
	 */

	return dest;
}


/*
 * ============================================================================
 *
 *                                    Caps
 *
 * ============================================================================
 */


static gboolean setcaps(GstPad *pad, GstCaps *caps)
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
 *                          Gstreamer Event Handler
 *
 * ============================================================================
 */


/*
 * src event.  inspired by the adder element's src pad event handler
 */


static gboolean src_event(GstPad *pad, GstEvent *event)
{
	GSTLALTriggerGen *element = GSTLAL_TRIGGERGEN(GST_PAD_PARENT(pad));
	gboolean success;

	switch(GST_EVENT_TYPE(event)) {
	case GST_EVENT_QOS:
	case GST_EVENT_NAVIGATION:
		/*
		 * not handled
		 */

		success = FALSE;
		break;

	case GST_EVENT_SEEK: {
		GstSeekFlags flags;
		GstSeekType curtype;
		gint64 cur;
		gboolean flush;

		/*
		 * parse the seek parameters
		 */

		gst_event_parse_seek(event, &element->segment.rate, NULL, &flags, &curtype, &cur, NULL, NULL);
		flush = !!(flags & GST_SEEK_FLAG_FLUSH);

		/*
		 * is it a flushing seek?
		 */

		if(flush) {
			/*
			 * make sure we accept nothing more and return
			 * WRONG_STATE
			 */

			gst_collect_pads_set_flushing(element->collect, TRUE);

			/*
			 * start flush downstream.
			 */

			gst_pad_push_event(element->srcpad, gst_event_new_flush_start());
		}

		/*
		 * wait for the collected to be finished and mark a new segment
		 */

		GST_OBJECT_LOCK(element->collect);
		element->segment_pending = TRUE;
		if(flush) {
			/*
			 * we need to do this again when the streaming
			 * threads have stoppped so that the cookie gets
			 * properly updated
			 */
			gst_collect_pads_set_flushing(element->collect, TRUE);
		}
		element->flush_stop_pending = flush;
		GST_OBJECT_UNLOCK(element->collect);

		gst_event_ref(event);
		success = gst_pad_push_event(element->snrpad, event);
		gst_event_ref(event);
		success &= gst_pad_push_event(element->chisqpad, event);
	}

	default:
		/*
		 * forward all other events
		 */

		gst_event_ref(event);
		success = gst_pad_push_event(element->snrpad, event);
		gst_event_ref(event);
		success &= gst_pad_push_event(element->chisqpad, event);
		break;
	}

	return success;
}


/*
 * snr event()
 */


static gboolean taglist_extract_string(GstObject *object, GstTagList *taglist, const char *tagname, gchar **dest)
{
	if(!gst_tag_list_get_string(taglist, tagname, dest)) {
		GST_WARNING_OBJECT(object, "unable to parse \"%s\" from %" GST_PTR_FORMAT, tagname, taglist);
		return FALSE;
	}
	return TRUE;
}


static gboolean snr_event(GstPad *pad, GstEvent *event)
{
	GSTLALTriggerGen *element = GSTLAL_TRIGGERGEN(GST_PAD_PARENT(pad));
	gboolean success;

	switch(GST_EVENT_TYPE(event)) {
	case GST_EVENT_FLUSH_STOP:
		/*
		 * mark a pending new segment.  this event is synchronized
		 * with the streaming thread so we can safely update the
		 * variable without races. it's somewhat weird because we
		 * assume the collectpads forwarded the FLUSH_STOP past us
		 * and downstream (using our source pad, the bastard!).
		 */

		GST_OBJECT_LOCK(element->collect);
		element->segment_pending = TRUE;
		element->flush_stop_pending = FALSE;
		GST_OBJECT_UNLOCK(element->collect);
		break;

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
		}
		break;
	}

	default:
		break;
	}

	return element->collect_event(pad, event);
}


/*
 * ============================================================================
 *
 *                         Inspiral Event Generation
 *
 * ============================================================================
 */


static GstFlowReturn collected(GstCollectPads *pads, gpointer user_data)
{
	GSTLALTriggerGen *element = GSTLAL_TRIGGERGEN(user_data);
	GstClockTime earliest_input_t_start, earliest_input_t_end;
	GstBuffer *snrbuf = NULL;
	GstBuffer *chisqbuf = NULL;
	GstBuffer *srcbuf = NULL;
	GstFlowReturn result;

	/*
	 * check for uninitialized stream
	 */

	if(!element->instrument || !element->channel_name) {
		GST_ELEMENT_ERROR(element, STREAM, FAILED, ("missing or invalid tags"), ("instrument and/or channel name not known (stream's tags must provide this information)"));
		result = GST_FLOW_ERROR;
		goto error;
	}


	/*
	 * forward flush-stop event
	 */

	if(element->flush_stop_pending) {
		gst_pad_push_event(element->srcpad, gst_event_new_flush_stop());
		element->flush_stop_pending = FALSE;
	}

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
			GST_ELEMENT_ERROR(element, STREAM, FAILED, ("missing or invalid segment"), ("unable to retrieve bounding segment from gstlal_collect_pads_get_segment()"));
			result = GST_FLOW_ERROR;
			goto error;
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
			GST_ERROR_OBJECT(element, "unable to create new segment event");
			result = GST_FLOW_ERROR;
			goto error;
		}
		GST_DEBUG_OBJECT(element->srcpad, "pushing newsegment event [%" GST_TIME_SECONDS_FORMAT ", %" GST_TIME_SECONDS_FORMAT ")", GST_TIME_SECONDS_ARGS(element->segment.start), GST_TIME_SECONDS_ARGS(element->segment.stop));
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

	if(!gstlal_collect_pads_get_earliest_times(element->collect, &earliest_input_t_start, &earliest_input_t_end, element->rate)) {
		GST_ERROR_OBJECT(element, "cannot deduce input timestamp offset information");
		result = GST_FLOW_ERROR;
		goto error;
	}

	/*
	 * check for EOS
	 */

	if(!GST_CLOCK_TIME_IS_VALID(earliest_input_t_start)) {
		GST_DEBUG_OBJECT(element, "gstlal_collect_pads_get_earliest_times() says we are at EOS");
		goto eos;
	}
	GST_DEBUG_OBJECT(element, "data available for [%" GST_TIME_SECONDS_FORMAT " s, %" GST_TIME_SECONDS_FORMAT " s)", GST_TIME_SECONDS_ARGS(earliest_input_t_start), GST_TIME_SECONDS_ARGS(earliest_input_t_end));

	/*
	 * get buffers upto the desired end time.
	 */

	snrbuf = gstlal_collect_pads_take_buffer_sync(pads, element->snrcollectdata, earliest_input_t_end, element->rate);
	chisqbuf = gstlal_collect_pads_take_buffer_sync(pads, element->chisqcollectdata, earliest_input_t_end, element->rate);

	/*
	 * NULL means EOS.  EOS on one means our EOS.
	 */

	if(!snrbuf || !chisqbuf) {
		if(snrbuf) {
			gst_buffer_unref(snrbuf);
			snrbuf = NULL;
		} else
			GST_DEBUG_OBJECT(element->snrpad, "at EOS");
		if(chisqbuf) {
			gst_buffer_unref(chisqbuf);
			chisqbuf = NULL;
		} else
			GST_DEBUG_OBJECT(element->chisqpad, "at EOS");
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

		/* FIXME:  if more than max_gap samples elapse, we should flush the triggers */

		GST_DEBUG_OBJECT(element, "input is gap, output is gap");
		result = gst_pad_alloc_buffer(element->srcpad, element->next_output_offset, 0, GST_PAD_CAPS(element->srcpad), &srcbuf);
		if(result != GST_FLOW_OK)
			goto error;
		g_assert(GST_BUFFER_CAPS(srcbuf) != NULL);
		g_assert(GST_PAD_CAPS(element->srcpad) == GST_BUFFER_CAPS(srcbuf));
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

		g_mutex_lock(element->bank_lock);
		length = MIN(GST_BUFFER_TIMESTAMP(snrbuf) + GST_BUFFER_DURATION(snrbuf), GST_BUFFER_TIMESTAMP(chisqbuf) + GST_BUFFER_DURATION(chisqbuf));
		if(GST_BUFFER_TIMESTAMP(snrbuf) > GST_BUFFER_TIMESTAMP(chisqbuf)) {
			t0 = GST_BUFFER_TIMESTAMP(snrbuf);
			chisqdata += gst_util_uint64_scale_int_round(GST_BUFFER_TIMESTAMP(snrbuf) - GST_BUFFER_TIMESTAMP(chisqbuf), element->rate, GST_SECOND) * element->num_templates;
		} else {
			t0 = GST_BUFFER_TIMESTAMP(chisqbuf);
			snrdata += gst_util_uint64_scale_int_round(GST_BUFFER_TIMESTAMP(chisqbuf) - GST_BUFFER_TIMESTAMP(snrbuf), element->rate, GST_SECOND) * element->num_templates;
		}
		length = gst_util_uint64_scale_int_round(length > t0 ? length - t0 : 0, element->rate, GST_SECOND);

		GST_DEBUG_OBJECT(element, "searching %" G_GUINT64_FORMAT " samples at %" GST_TIME_SECONDS_FORMAT " for events", length, GST_TIME_SECONDS_ARGS(t0));
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
						record_inspiral_event(&element->last_event[channel], t, *snrdata, *chisqdata, channel, element);
					} else if(cabs(*snrdata) > element->last_event[channel].snr) {
						/*
						 * Same event, higher SNR,
						 * update
						 */
						record_inspiral_event(&element->last_event[channel], t, *snrdata, *chisqdata, channel, element);
					} else {
						/*
						 * Same event, not higher
						 * SNR, do nothing
						 */
					}
					element->last_time[channel] = t;
				} else {
					/* FIXME:  if more than max_gap samples elapse, we should flush the triggers */
				}

				snrdata++;
				chisqdata++;
			}
		}
		g_mutex_unlock(element->bank_lock);

		if(nevents) {
			SnglInspiralTable *dest;

			GST_DEBUG_OBJECT(element, "found %d events", nevents);
			result = gst_pad_alloc_buffer(element->srcpad, element->next_output_offset, nevents * sizeof(*dest), GST_PAD_CAPS(element->srcpad), &srcbuf);
			if(result != GST_FLOW_OK) {
				while(head) {
					SnglInspiralTable *next = head->next;
					free(head);
					head = next;
				}
				goto error;
			}
			g_assert(GST_BUFFER_CAPS(srcbuf) != NULL);
			g_assert(GST_PAD_CAPS(element->srcpad) == GST_BUFFER_CAPS(srcbuf));
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
				*dest = *head;
				dest->next = dest + 1;
				free(head);
				head = next;
			}
			dest[nevents].next = NULL;
		} else {
			GST_DEBUG_OBJECT(element, "found 0 events, sending gap downstream");
			result = gst_pad_alloc_buffer(element->srcpad, element->next_output_offset, 0, GST_PAD_CAPS(element->srcpad), &srcbuf);
			if(result != GST_FLOW_OK)
				goto error;
			g_assert(GST_BUFFER_CAPS(srcbuf) != NULL);
			g_assert(GST_PAD_CAPS(element->srcpad) == GST_BUFFER_CAPS(srcbuf));
			GST_BUFFER_OFFSET(srcbuf) = GST_BUFFER_OFFSET_END(srcbuf) = element->next_output_offset;
			GST_BUFFER_FLAG_SET(srcbuf, GST_BUFFER_FLAG_GAP);
		}
	}

	/*
	 * Push buffer downstream
	 */

	GST_BUFFER_TIMESTAMP(srcbuf) = MAX(GST_BUFFER_TIMESTAMP(snrbuf), GST_BUFFER_TIMESTAMP(chisqbuf));
	GST_BUFFER_DURATION(srcbuf) = MIN(GST_BUFFER_TIMESTAMP(snrbuf) + GST_BUFFER_DURATION(snrbuf), GST_BUFFER_TIMESTAMP(chisqbuf) + GST_BUFFER_DURATION(chisqbuf));
	GST_BUFFER_DURATION(srcbuf) = GST_BUFFER_DURATION(srcbuf) < GST_BUFFER_TIMESTAMP(srcbuf) ? 0 : GST_BUFFER_DURATION(srcbuf) - GST_BUFFER_TIMESTAMP(srcbuf);

	if(element->next_output_timestamp != GST_BUFFER_TIMESTAMP(srcbuf))
		GST_BUFFER_FLAG_SET(srcbuf, GST_BUFFER_FLAG_DISCONT);
	element->next_output_timestamp = GST_BUFFER_TIMESTAMP(srcbuf) + GST_BUFFER_DURATION(srcbuf);

	gst_buffer_unref(snrbuf);
	gst_buffer_unref(chisqbuf);

	GST_DEBUG_OBJECT(element->srcpad, "pushing %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(srcbuf));
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
	GST_DEBUG_OBJECT(element->srcpad, "pushing EOS event");
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


enum property {
	ARG_SNR_THRESH = 1, 
	ARG_BANK_FILENAME,
	ARG_MAX_GAP,
	ARG_SIGMASQ
};


static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec)
{
	GSTLALTriggerGen *element = GSTLAL_TRIGGERGEN(object);

	GST_OBJECT_LOCK(element);
	switch(id) {
	case ARG_SNR_THRESH:
		element->snr_thresh = g_value_get_double(value);
		break;

	case ARG_BANK_FILENAME:
		g_mutex_lock(element->bank_lock);
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


static void get_property(GObject * object, enum property id, GValue * value, GParamSpec * pspec)
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
		} else {
			GST_WARNING_OBJECT(element, "no template bank");
			g_value_take_boxed(value, g_value_array_new(0));
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


/*
 * ============================================================================
 *
 *                              Element Support
 *
 * ============================================================================
 */


static GstElementClass *parent_class = NULL;


static void finalize(GObject *object)
{
	GSTLALTriggerGen *element = GSTLAL_TRIGGERGEN(object);
	g_mutex_free(element->bank_lock);
	element->bank_lock = NULL;
	free_bankfile(element);
	g_free(element->instrument);
	element->instrument = NULL;
	g_free(element->channel_name);
	element->channel_name = NULL;
	G_OBJECT_CLASS(parent_class)->finalize(object);
}


static GstStateChangeReturn change_state(GstElement *element, GstStateChange transition)
{
	GSTLALTriggerGen *triggergen = GSTLAL_TRIGGERGEN(element);

	switch(transition) {
	case GST_STATE_CHANGE_NULL_TO_READY:
		break;

	case GST_STATE_CHANGE_READY_TO_PAUSED:
		triggergen->segment_pending = TRUE;
		triggergen->flush_stop_pending = FALSE;
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

	return parent_class->change_state(element, transition);
}


static void base_init(gpointer g_class)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(g_class);

	gst_element_class_set_details_simple(
		element_class,
		"Trigger Generator",
		"Filter",
		"Produce sngl_inspiral records from SNR and chi squared.\n"
		"A trigger is recorded for every instant at which the absolute value of the SNR\n" \
		"is greater than snr-thresh, and also greater than at all of the max_gap seconds\n" \
		"of data that come before and after.  snr-thresh and max_gap are properties of\n" \
		"this element.\n" \
		"\n" \
		"The maximum possible trigger rate is (1/max_gap) Hz per template.\n", \
		"Kipp Cannon <kipp.cannon@ligo.org>, Chad Hanna <channa@ligo.caltech.edu>"
	);

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


static void class_init(gpointer klass, gpointer class_data)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *gstelement_class = GST_ELEMENT_CLASS(klass);

	parent_class = g_type_class_ref(GST_TYPE_ELEMENT);
	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);
	gstelement_class->change_state = GST_DEBUG_FUNCPTR(change_state);

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


static void instance_init(GTypeInstance *object, gpointer klass)
{
	GSTLALTriggerGen *element = GSTLAL_TRIGGERGEN(object);
	GstPad *pad;

	gst_element_create_all_pads(GST_ELEMENT(element));
	element->collect = gst_collect_pads_new();
	gst_collect_pads_set_function(element->collect, GST_DEBUG_FUNCPTR(collected), element);

	/* configure snr pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "snr");
	gst_pad_set_setcaps_function(pad, GST_DEBUG_FUNCPTR(setcaps));
	gst_pad_use_fixed_caps(pad);
	element->snrcollectdata = gstlal_collect_pads_add_pad(element->collect, pad, sizeof(*element->snrcollectdata));
	element->snrpad = pad;

	/* configure chisquare pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "chisquare");
	gst_pad_set_setcaps_function(pad, GST_DEBUG_FUNCPTR(setcaps));
	gst_pad_use_fixed_caps(pad);
	element->chisqcollectdata = gstlal_collect_pads_add_pad(element->collect, pad, sizeof(*element->chisqcollectdata));
	element->chisqpad = pad;

	/* FIXME: hacked way to override/extend the event function of
	 * GstCollectpads */
	element->collect_event = (GstPadEventFunction) GST_PAD_EVENTFUNC(element->snrpad);
	gst_pad_set_event_function(element->snrpad, GST_DEBUG_FUNCPTR(snr_event));

	/* retrieve (and ref) src pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "src");
	gst_pad_set_event_function(pad, GST_DEBUG_FUNCPTR(src_event));
	gst_pad_use_fixed_caps(pad);
	{
	GstCaps *caps = gst_caps_copy(gst_pad_get_pad_template_caps(pad));
	gst_pad_set_caps(pad, caps);
	gst_caps_unref(caps);
	}
	element->srcpad = pad;

	/* internal data */
	element->bank_lock = g_mutex_new();
	element->rate = 0;
	element->bank_filename = NULL;
	element->instrument = NULL;
	element->channel_name = NULL;
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
			.class_init = class_init,
			.base_init = base_init,
			.instance_size = sizeof(GSTLALTriggerGen),
			.instance_init = instance_init,
		};
		GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "lal_triggergen", 0, "lal_triggergen element");
		type = g_type_register_static(GST_TYPE_ELEMENT, "lal_triggergen", &info, 0);
	}

	return type;
}

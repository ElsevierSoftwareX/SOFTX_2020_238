/* GStreamer
 * Copyright (C) 1999,2000 Erik Walthinsen <omega@cse.ogi.edu>
 *                    2001 Thomas <thomas@apestaart.org>
 *               2005,2006 Wim Taymans <wim@fluendo.com>
 *               2011 Kipp Cannon <kipp.cannon@ligo.org>
 *               2011 Drew Keppel <drew.keppel@ligo.org>
 *
 * gstlal_timeslicechisq.c: Time-slice based chisq element
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */


/* Element-Checklist-Version: 5 */


/*
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


#ifdef HAVE_CONFIG_H
#include "config.h"
#endif


/*
 * stuff from the C library
 */


#include <complex.h>
#include <math.h>
#include <string.h>


/*
 * stuff from glib/gstreamer
 */


#include <gst/gst.h>
#include <gst/base/gstcollectpads.h>
#include <gst/audio/audio.h>


/*
 * stuff from GSL
 */


#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>


/*
 * our own stuff
 */


#include <gstlal/gstlal.h>
#include <gstlal/gstlal_debug.h>
#include <gstlal/gstlalcollectpads.h>
#include "gstlal_timeslicechisq.h"


#define GST_CAT_DEFAULT gstlal_timeslicechisq_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);

enum
property {
	ARG_0,
	ARG_CHIFACS
};

//GST_BOILERPLATE(GSTLALTimeSliceChiSquare, gstlal_timeslicechisq, GstElement, GST_TYPE_ELEMENT);


/*
 * ========================================================================
 *
 *                             Utility Functions
 *
 * ========================================================================
 */


/*
 * return the number of channels
 */


static int
num_channels(const GSTLALTimeSliceChiSquare *element)
{
	return element->chifacs->size2;
}


/*
 * return the number of timeslices
 */


static int
num_timeslices(const GSTLALTimeSliceChiSquare *element)
{
	return element->chifacs->size1;
}


/*
 * function to add doubles
 */


static void
add_complex128(gpointer out, const gpointer in, size_t bytes)
{
	double complex *_out = out;
	const double complex *_in = in;
	for(bytes /= sizeof(double complex); bytes--; _in++, _out++)
		*_out = (double complex) *_out + (double complex) *_in;
}


/*
 * ============================================================================
 *
 *                                 Properties
 *
 * ============================================================================
 */


static void
set_property(GObject *object, enum property id, const GValue *value, GParamSpec *psspec)
{
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(object);

	GST_OBJECT_LOCK(element);

	switch (id) {
	case ARG_CHIFACS: {
		int channels;
		g_mutex_lock(element->coefficients_lock);
		if(element->chifacs) {
			channels = num_channels(element);
			gsl_matrix_free(element->chifacs);
		} else
			channels = 0;
		element->chifacs = gstlal_gsl_matrix_from_g_value_array(g_value_get_boxed(value));

		/* number of channels has changed, force a caps renegotiation */
		if(num_channels(element) != channels) {
			/* FIXME: is this correct? */
			gst_pad_set_caps(element->srcpad, NULL);
		}

		/* signal availability of new chifacs vector */
		g_cond_broadcast(element->coefficients_available);
		g_mutex_unlock(element->coefficients_lock);
		break;
	}

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, psspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


static void
get_property(GObject *object, enum property id, GValue *value, GParamSpec *psspec)
{
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(object);

	GST_OBJECT_LOCK(element);

	switch (id) {
	case ARG_CHIFACS:
		g_mutex_lock(element->coefficients_lock);
		if(element->chifacs)
			g_value_take_boxed(value, gstlal_g_value_array_from_gsl_matrix(element->chifacs));
		/* FIXME:  else? */
		g_mutex_unlock(element->coefficients_lock);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, psspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * ============================================================================
 *
 *                                    Caps
 *
 * ============================================================================
 */


/*
 * we can only accept caps that both ourselves and the downstream peer can
 * handle
 */


static GstCaps *
sink_getcaps(GstPad *pad)
{
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(GST_PAD_PARENT(pad));
	GstCaps *peercaps;
	GstCaps *caps;

	/* get our own allowed caps.  use the fixed caps function to avoid
	 * recursing back into this function. */
	GST_OBJECT_LOCK(element);
	caps = gst_pad_get_fixed_caps_func(pad);

	/* get the allowed caps from the downstream peer.
	 * if the peer has caps, intersect. */
	peercaps = gst_pad_peer_get_caps(element->srcpad);
	if(peercaps) {
		GstCaps *result;
		GValue value = {0,};
		g_value_init(&value, G_TYPE_INT);
		GstStructure *peercaps_struct = gst_caps_steal_structure(peercaps, 0);
		gst_structure_set_name(peercaps_struct, (const gchar *) "audio/x-raw-complex");
		gst_caps_append_structure(peercaps, peercaps_struct);
		g_value_set_int(&value, 128);
		gst_caps_set_value(peercaps, "width", &value);

		GST_DEBUG_OBJECT(element, "intersecting %" GST_PTR_FORMAT " and %" GST_PTR_FORMAT, caps, peercaps);
		result = gst_caps_intersect(peercaps, caps);
		gst_caps_unref(peercaps);
		gst_caps_unref(caps);
		caps = result;
		GST_DEBUG_OBJECT(element, "intersection %" GST_PTR_FORMAT, caps);
	}
	else {
		GST_DEBUG_OBJECT (element, "no peer caps, using sink pad's caps");
	}
	GST_OBJECT_UNLOCK(element);

	/* done */
	return caps;
}


static GstCaps *
src_getcaps(GstPad *pad)
{
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(GST_PAD_PARENT(pad));
	GList *padlist = NULL;
	GstCaps *peercaps;
	GstCaps *caps;

	/* get our own allowed caps.  use the fixed caps function to avoid
	 * recursing back into this function. */

	GST_OBJECT_LOCK(element);
	caps = gst_pad_get_fixed_caps_func(pad);

	/* get the allowed caps from the upstream peer.
	 * if the peer has caps, intersect. */

	for(padlist = GST_ELEMENT(element)->pads; padlist; padlist = g_list_next(padlist)) {
		GstPad *otherpad = GST_PAD(padlist->data);
		if(otherpad != pad) {
			peercaps = gst_pad_peer_get_caps(otherpad);
			if(peercaps) {
				GstCaps *result;
				GValue value = {0,};
				g_value_init(&value, G_TYPE_INT);
				GstStructure *peercaps_struct = gst_caps_steal_structure(peercaps, 0);
				gst_structure_set_name(peercaps_struct, (const gchar *) "audio/x-raw-float");
				gst_caps_append_structure(peercaps, peercaps_struct);
				g_value_set_int(&value, 64);
				gst_caps_set_value(peercaps, "width", &value);

				GST_DEBUG_OBJECT(element, "intersecting %" GST_PTR_FORMAT " and %" GST_PTR_FORMAT, caps, peercaps);
				result = gst_caps_intersect(peercaps, caps);
				gst_caps_unref(peercaps);
				gst_caps_unref(caps);
				caps = result;
				GST_DEBUG_OBJECT(element, "intersection %" GST_PTR_FORMAT, caps);
			}
			break;
		}
	}
	GST_OBJECT_UNLOCK(element);

	/* done */
	return caps;
}


/*
 * the first caps we receive on any of the sinkpads will define the caps
 * for all the other sinkpads because we can only mix streams with the same
 * caps.
 */


static gboolean
setcaps(GstPad *pad, GstCaps *caps)
{
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(GST_PAD_PARENT(pad));
	GList *padlist = NULL;
	GstStructure *structure = NULL;
	GstCaps *sinkcaps = gst_caps_copy(caps);
	GstCaps *srccaps = gst_caps_copy(caps);

	GST_LOG_OBJECT(element, "setting caps on pad %s:%s to %" GST_PTR_FORMAT, GST_DEBUG_PAD_NAME(pad), caps);

	/* parse caps */
	structure = gst_caps_get_structure(caps, 0);
	gst_structure_get_int(structure, "rate", &element->rate);
	gst_structure_get_int(structure, "channels", &element->channels);

	/* pre-calculate bytes / sample */
	element->float_unit_size = 8 * element->channels;
	element->complex_unit_size = 16 * element->channels;

	/* create the caps appropriate for src and sink pads */
	if(gst_pad_get_direction(pad) == GST_PAD_SINK) {
		gint width;
		GValue value = {0,};
		g_value_init(&value, G_TYPE_INT);
		GstStructure *srccaps_struct = gst_caps_steal_structure(srccaps, 0);
		gst_structure_get_int(srccaps_struct, "width", &width);
		gst_structure_set_name(srccaps_struct, (const gchar *) "audio/x-raw-float");
		gst_caps_append_structure(srccaps, srccaps_struct);
		g_value_set_int(&value, width / 2);
		gst_caps_set_value(srccaps, "width", &value);
	}
	else {
		gint width;
		GValue value = {0,};
		g_value_init(&value, G_TYPE_INT);
		GstStructure *sinkcaps_struct = gst_caps_steal_structure(sinkcaps, 0);
		gst_structure_get_int(sinkcaps_struct, "width", &width);
		gst_structure_set_name(sinkcaps_struct, (const gchar *) "audio/x-raw-complex");
		gst_caps_append_structure(sinkcaps, sinkcaps_struct);
		g_value_set_int(&value, width * 2);
		gst_caps_set_value(sinkcaps, "width", &value);
	}

	/* loop over all of the element's pads (source and sink), and set
	 * them all to the appropriate format.
	 *
	 * FIXME, see if the other pads can accept the format. Also lock
	 * the format on the other pads to this new format. */

	GST_OBJECT_LOCK(element);
	for(padlist = GST_ELEMENT(element)->pads; padlist; padlist = g_list_next(padlist)) {
		GstPad *otherpad = GST_PAD(padlist->data);
		if(otherpad != pad) {
			/* don't use gst_pad_set_caps() because that would
			 * recurse into this function */
			if(gst_pad_get_direction(otherpad) == GST_PAD_SINK) {
				gst_caps_replace(&GST_PAD_CAPS(otherpad), sinkcaps);
			}
			else {
				gst_caps_replace(&GST_PAD_CAPS(otherpad), srccaps);
			}
		}
	}
	GST_OBJECT_UNLOCK(element);

	for(padlist = GST_ELEMENT(element)->pads; padlist; padlist = g_list_next(padlist)) {
		GstPad *pad = GST_PAD(padlist->data);
		if(gst_pad_get_direction(pad) == GST_PAD_SINK) {
			gstlal_collect_pads_set_unit_size(pad, element->complex_unit_size);
			gstlal_collect_pads_set_rate(pad, element->rate);
		}
	}

	/* done */
	gst_caps_unref(sinkcaps);
	gst_caps_unref(srccaps);	

	return TRUE;
}


/*
 * ============================================================================
 *
 *                                  Queries
 *
 * ============================================================================
 */


/*
 * convert an output offset to an output timestamp.
 */


static GstClockTime
output_timestamp_from_offset(const GSTLALTimeSliceChiSquare *element, guint64 offset)
{
	return element->segment.start + gst_util_uint64_scale_int_round(offset, GST_SECOND, element->rate);
}


/*
 * FIXME, the duration query should reflect how long you will produce
 * data, that is the amount of stream time until you will emit EOS.
 *
 * For synchronized mixing this is always the max of all the durations
 * of upstream since we emit EOS when all of them finished.
 *
 * We don't do synchronized mixing so this really depends on where the
 * streams where punched in and what their relative offsets are against
 * eachother which we can get from the first timestamps we see.
 *
 * When we add a new stream (or remove a stream) the duration might
 * also become invalid again and we need to post a new DURATION
 * message to notify this fact to the parent.
 * For now we take the max of all the upstream elements so the simple
 * cases work at least somewhat.
 */

static gboolean
query_duration(GSTLALTimeSliceChiSquare *element, GstQuery *query)
{
	GstIterator *it = NULL;
	gint64 max = -1;
	GstFormat format;
	gboolean success = TRUE;
	gboolean done = FALSE;

	/* parse duration query format */
	gst_query_parse_duration(query, &format, NULL);

	/* iterate over sink pads */
	it = gst_element_iterate_sink_pads(GST_ELEMENT_CAST(element));
	while(!done && success) {
		gpointer item;

		switch(gst_iterator_next(it, &item)) {
		case GST_ITERATOR_DONE:
			done = TRUE;
			break;

		case GST_ITERATOR_OK: {
			GstPad *pad = GST_PAD_CAST(item);
			gint64 duration;

			/* query upstream peer for duration */
			if(gst_pad_query_peer_duration(pad, &format, &duration)) {
				/* query succeeded */
				if(duration == -1) {
					/* unknown duration --> the duration of our output is unknown */
					max = duration;
					done = TRUE;
				} else if(duration > max) {
					/* take largest duration */
					max = duration;
				}
			} else {
				/* query failed */
				success = FALSE;
			}
			gst_object_unref(pad);
			break;
		}

		case GST_ITERATOR_RESYNC:
			max = -1;
			success = TRUE;
			gst_iterator_resync(it);
			break;

		default:
			success = FALSE;
			done = TRUE;
			break;
		}
	}
	gst_iterator_free(it);

	if(success) {
		/* store the max */
		GST_DEBUG_OBJECT(element, "Total duration in format %s: %" GST_TIME_FORMAT, gst_format_get_name(format), GST_TIME_ARGS(max));
		gst_query_set_duration(query, format, max);
	}

	return success;
}


static gboolean
query_latency(GSTLALTimeSliceChiSquare *element, GstQuery *query)
{
	GstIterator *it = NULL;
	GstClockTime min = 0;
	GstClockTime max = GST_CLOCK_TIME_NONE;
	gboolean live = FALSE;
	gboolean success = TRUE;
	gboolean done = FALSE;

	/* iterate over sink pads */
	it = gst_element_iterate_sink_pads(GST_ELEMENT_CAST(element));
	while(!done && success) {
		gpointer item;

		switch(gst_iterator_next(it, &item)) {
		case GST_ITERATOR_DONE:
			done = TRUE;
			break;

		case GST_ITERATOR_OK: {
			GstPad *pad = GST_PAD_CAST(item);
			GstQuery *peerquery = gst_query_new_latency();

			/* query upstream peer for latency */
			success &= gst_pad_peer_query(pad, peerquery);

			if(success) {
				/* query succeeded */
				GstClockTime min_cur;
				GstClockTime max_cur;
				gboolean live_cur;

				gst_query_parse_latency(peerquery, &live_cur, &min_cur, &max_cur);

				/* take the largest of the latencies */
				if(min_cur > min)
					min = min_cur;
				if(max_cur != GST_CLOCK_TIME_NONE && ((max != GST_CLOCK_TIME_NONE && max_cur > max) || (max == GST_CLOCK_TIME_NONE)))
					max = max_cur;
				/* we're live if any upstream peer is live */
				live |= live_cur;
			}

			gst_query_unref(peerquery);
			gst_object_unref(pad);
			break;
		}

		case GST_ITERATOR_RESYNC:
			live = FALSE;
			min = 0;
			max = GST_CLOCK_TIME_NONE;
			success = TRUE;
			gst_iterator_resync(it);
			break;

		default:
			success = FALSE;
			done = TRUE;
			break;
		}
	}
	gst_iterator_free(it);

	if(success) {
		/* store the results */
		GST_DEBUG_OBJECT(element, "Calculated total latency: live %s, min %" GST_TIME_FORMAT ", max %" GST_TIME_FORMAT, (live ? "yes" : "no"), GST_TIME_ARGS(min), GST_TIME_ARGS(max));
		gst_query_set_latency(query, live, min, max);
	}

	return success;
}


static gboolean
query_function(GstPad *pad, GstQuery *query)
{
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(gst_pad_get_parent(pad));
	gboolean success = TRUE;

	switch(GST_QUERY_TYPE(query)) {
	case GST_QUERY_POSITION: {
		GstFormat format;

		gst_query_parse_position(query, &format, NULL);

		switch(format) {
		case GST_FORMAT_TIME:
			/* FIXME, bring to stream time, might be tricky */
			gst_query_set_position(query, format, element->timestamp);
			break;

		case GST_FORMAT_DEFAULT:
			/* default format for audio is sample count */
			gst_query_set_position(query, format, element->offset);
			break;

		default:
			success = FALSE;
			break;
		}
		break;
	}

	case GST_QUERY_DURATION:
		success = query_duration(element, query);
		break;

	case GST_QUERY_LATENCY:
		success = query_latency(element, query);
		break;

	default:
		/* FIXME, needs a custom query handler because we have multiple
		 * sinkpads */
		success = gst_pad_query_default(pad, query);
		break;
	}

	gst_object_unref(element);
	return success;
}


/*
 * ============================================================================
 *
 *                                   Events
 *
 * ============================================================================
 */


typedef struct
{
  GstEvent *event;
  gboolean flush;
} EventData;


/*
 * helper function used by forward_event() (see below)
 */


static gboolean
forward_event_func(GstPad *pad, GValue *ret, EventData *data)
{
	GstEvent *event = data->event;

	gst_event_ref(event);
	GST_LOG_OBJECT(pad, "About to send event %s on pad %s:%s", GST_EVENT_TYPE_NAME(event), GST_DEBUG_PAD_NAME(pad));
	if(!gst_pad_push_event(pad, event)) {
		GST_WARNING_OBJECT(pad, "Sending event %p (%s) failed.", event, GST_EVENT_TYPE_NAME(event));
		/* quick hack to unflush the pads, ideally we need a way to just unflush
		 * this single collect pad */
		if (data->flush)
			gst_pad_send_event(pad, gst_event_new_flush_stop ());
	} else {
		g_value_set_boolean(ret, TRUE);
		GST_LOG_OBJECT(pad, "Sent event %p (%s).", event, GST_EVENT_TYPE_NAME(event));
	}
	gst_object_unref(pad);

	/* continue on other pads, even if one failed */
	return TRUE;
}


/* forwards the event to all sinkpads, takes ownership of the
 * event
 *
 * Returns: TRUE if the event could be forwarded on all
 * sinkpads.
 */


static gboolean
forward_event(GSTLALTimeSliceChiSquare *element, GstEvent *event, gboolean flush)
{
	gboolean ret;
	GstIterator *it = NULL;
	GstIteratorResult ires;
	GValue vret = {0};
	EventData data;

	GST_LOG_OBJECT(element, "Forwarding event %p (%s)", event, GST_EVENT_TYPE_NAME(event));

	data.event = event;
	data.flush = flush;

	g_value_init(&vret, G_TYPE_BOOLEAN);
	g_value_set_boolean(&vret, FALSE);
	it = gst_element_iterate_sink_pads(GST_ELEMENT_CAST(element));
	while (TRUE) {
		ires = gst_iterator_fold(it, (GstIteratorFoldFunction) forward_event_func, &vret, &data);
		switch (ires) {
		case GST_ITERATOR_RESYNC:
			GST_WARNING("resync");
			gst_iterator_resync(it);
			g_value_set_boolean(&vret, TRUE);
			break;
		case GST_ITERATOR_OK:
		case GST_ITERATOR_DONE:
			ret = g_value_get_boolean(&vret);
			goto done;
		default:
			ret = FALSE;
			goto done;
		}
	}
done:
	gst_iterator_free(it);
	GST_LOG_OBJECT(element, "Forwarded event %p (%s), ret=%d", event, GST_EVENT_TYPE_NAME(event), ret);
	gst_event_unref(event);

	return g_value_get_boolean(&vret);
}


/*
 * src pad event handler
 */


static gboolean
src_event_function(GstPad *pad, GstEvent *event)
{
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(gst_pad_get_parent(pad));
	gboolean result;

	switch (GST_EVENT_TYPE(event)) {
	case GST_EVENT_SEEK: {
		gdouble rate;
		GstSeekFlags flags;
		GstSeekType curtype, endtype;
		gint64 cur, end;
		gboolean flush;

		/* parse the seek parameters */
		gst_event_parse_seek(event, &rate, NULL, &flags, &curtype, &cur, &endtype, &end);

		if ((curtype != GST_SEEK_TYPE_NONE) && (curtype != GST_SEEK_TYPE_SET)) {
			result = FALSE;
			GST_DEBUG_OBJECT(element, "seeking failed, unhandled seek type for start: %d", curtype);
			goto done;
		}
		if ((endtype != GST_SEEK_TYPE_NONE) && (endtype != GST_SEEK_TYPE_SET)) {
			result = FALSE;
			GST_DEBUG_OBJECT(element, "seeking failed, unhandled seek type for end: %d", endtype);
			goto done;
		}

		flush = (flags & GST_SEEK_FLAG_FLUSH) == GST_SEEK_FLAG_FLUSH;

		/* check if we are flushing */
		if(flush) {
			/* make sure we accept nothing anymore and return WRONG_STATE */
			gst_collect_pads_set_flushing(element->collect, TRUE);

			/* flushing seek, start flush downstream, the flush will be done
			 * when all pads received a FLUSH_STOP. */
			gst_pad_push_event(element->srcpad, gst_event_new_flush_start());

			/* We can't send FLUSH_STOP here since upstream could start pushing data
			 * after we unlock adder->collect.
			 * We set flush_stop_pending to TRUE instead and send FLUSH_STOP after
			 * forwarding the seek upstream or from gst_adder_collected,
			 * whichever happens first. */
			element->flush_stop_pending = TRUE;
		}
		GST_DEBUG_OBJECT(element, "handling seek event: %" GST_PTR_FORMAT, event);

		/* now wait for the collected to be finished and mark a new
		 * segment. After we have the lock, no collect function is running and no
		 * new collect function will be called for as long as we're flushing. */
		GST_OBJECT_LOCK(element->collect);
		/* make sure we push a new segment, to inform about new basetime
		 * see FIXME in gst_adder_collected() */
		element->segment_pending = TRUE;
		if(flush) {
			/* Yes, we need to call _set_flushing again *WHEN* the streaming threads
			 * have stopped so that the cookie gets properly updated. */
			gst_collect_pads_set_flushing(element->collect, TRUE);
		}
		GST_OBJECT_UNLOCK(element->collect);

		GST_DEBUG_OBJECT(element, "forwarding seek event: %" GST_PTR_FORMAT, event);
		result = forward_event(element, event, flush);
		if (!result) {
			/* seek failed. maybe source is a live source. */
			GST_DEBUG_OBJECT(element, "seeking failed");
		}
		if (g_atomic_int_compare_and_exchange(&element->flush_stop_pending, TRUE, FALSE)) {
			GST_DEBUG_OBJECT(element, "pending flush stop");
			gst_pad_push_event(element->srcpad, gst_event_new_flush_stop());
		}
		break;
	}
	case GST_EVENT_QOS:
		/* QoS might be tricky */
		result = FALSE;
		break;
	case GST_EVENT_NAVIGATION:
		/* navigation is rather pointless. */
		result = FALSE;
		break;
	default:
		/* just forward the rest for now */
		GST_DEBUG_OBJECT(element, "forward unhandled event: %s", GST_EVENT_TYPE_NAME(event));
		result = forward_event(element, event, FALSE);
		break;
	}

done:
	gst_object_unref(element);
	return result;
}


/*
 * sink pad event handler.  this is hacked in as an override of the collect
 * pads object's own event handler so that we can detect new segments and
 * flush stop events arriving on sink pads.  the real event handling is
 * accomplished by chaining to the original event handler installed by the
 * collect pads object.
 */


static gboolean
sink_event_function(GstPad *pad, GstEvent *event)
{
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(gst_pad_get_parent(pad));
	gboolean ret = TRUE;

	GST_DEBUG("Got event %s on pad %s:%s", GST_EVENT_TYPE_NAME(event), GST_DEBUG_PAD_NAME(pad));

	switch (GST_EVENT_TYPE(event)) {
	case GST_EVENT_FLUSH_STOP:
		/* we received a flush-stop. The collect_event function will push the
		 * event past our element. We simply forward all flush-stop events, even
		 * when no flush-stop was pending, this is required because collectpads
		 * does not provide an API to handle-but-not-forward the flush-stop.
		 * We unset the pending flush-stop flag so that we don't send anymore
		 * flush-stop from the collect function later. */
		GST_OBJECT_LOCK(element->collect);
		element->segment_pending = TRUE;
		element->flush_stop_pending = FALSE;
		/* Clear pending tags */
		/* FIXME:  switch to
		  g_list_free_full (adder->pending_events, (GDestroyNotify) gst_event_unref);
		  adder->pending_events = NULL;
		 */
		while (element->pending_events) {
			GstEvent *ev = GST_EVENT(element->pending_events->data);
			gst_event_unref(ev);
			element->pending_events = g_list_remove(element->pending_events, ev);
		}
		GST_OBJECT_UNLOCK(element->collect);
		break;
	case GST_EVENT_TAG:
		GST_OBJECT_LOCK(element->collect);
		/* collect tags here so we can push them out when we collect data */
		element->pending_events = g_list_append(element->pending_events, event);
		GST_OBJECT_UNLOCK(element->collect);
		goto beach;
	default:
		break;
	}

	/* now GstCollectPads can take care of the rest, e.g. EOS */
	ret = element->collect_event(pad, event);

beach:
	gst_object_unref(element);
	return ret;
}


/*
 * ============================================================================
 *
 *                            Add/Remove Sink Pad
 *
 * ============================================================================
 */


static GstPad *
request_new_pad(GstElement *gstelement, GstPadTemplate *templ, const gchar *unused)
{
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(gstelement);
	gchar *name = NULL;
	GstPad *newpad = NULL;
	GstLALCollectData *data = NULL;
	gint padcount;

	/* new pads can only be sink pads */
	if(templ->direction != GST_PAD_SINK) {
		g_warning("gstlal_timeslicechisq: request new pad that is not a SINK pad\n");
		goto not_sink;
	}

	/* create a new pad */
	padcount = g_atomic_int_exchange_and_add(&element->padcount, 1);
	name = g_strdup_printf(GST_PAD_TEMPLATE_NAME_TEMPLATE(templ), padcount);
	newpad = gst_pad_new_from_template(templ, name);
	GST_DEBUG_OBJECT(element, "request new pad %s", name);
	g_free(name);

	/* configure new pad */
	gst_pad_set_getcaps_function(newpad, GST_DEBUG_FUNCPTR(sink_getcaps));
	gst_pad_set_setcaps_function(newpad, GST_DEBUG_FUNCPTR(setcaps));

	/* add pad to collect pads object */
	GST_OBJECT_LOCK(element->collect);
	data = gstlal_collect_pads_add_pad(element->collect, newpad, sizeof(*data));
	if(!data) {
		GST_DEBUG_OBJECT(element, "could not add pad to collectpads object");
		goto could_not_add_to_collectpads;
	}
	GST_OBJECT_UNLOCK(element->collect);

	/* FIXME: hacked way to override/extend the event function of
	 * GstCollectPads;  because it sets its own event function giving
	 * the element (us) no access to events */
	element->collect_event = (GstPadEventFunction) GST_PAD_EVENTFUNC(newpad);
	gst_pad_set_event_function(newpad, GST_DEBUG_FUNCPTR(sink_event_function));

	/* takes ownership of the pad */
	if(!gst_element_add_pad(GST_ELEMENT(element), newpad)) {
		GST_DEBUG_OBJECT(element, "could not add pad to element");
		goto could_not_add_to_element;
	}

	/* done */
	return newpad;

	/* ERRORS */
could_not_add_to_element:
	gstlal_collect_pads_remove_pad(element->collect, newpad);
could_not_add_to_collectpads:
	gst_object_unref(newpad);
not_sink:
	return NULL;
}


static void
release_pad(GstElement *gstelement, GstPad *pad)
{
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(gstelement);

	GST_DEBUG_OBJECT(element, "release pad %s:%s", GST_DEBUG_PAD_NAME(pad));

	gstlal_collect_pads_remove_pad(element->collect, pad);
	gst_element_remove_pad(gstelement, pad);
}


/*
 * ============================================================================
 *
 *                               Stream Mixing
 *
 * ============================================================================
 */


/*
 * GstCollectPads callback.  called when data is available on all input
 * pads
 */


static GstFlowReturn collected(GstCollectPads *pads, gpointer user_data)
{
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(user_data);
	GSList *collected = NULL;
	GstBuffer *outbuf = NULL;
	GSList *input_buffers = NULL;
	gboolean have_buffers = FALSE;
	GstFlowReturn ret;
	guint64 outlength;
	GstClockTime t_start, t_end;
	guint64 earliest_input_offset, earliest_input_offset_end;
	unsigned int timeslice;
	void *snrbytes = NULL;
	gpointer outbytes = NULL;
	unsigned int numchannels, numtimeslices, channel, sample;

	/* forward flush-stop event */
	if(g_atomic_int_compare_and_exchange (&element->flush_stop_pending, TRUE,FALSE)) {
		GST_DEBUG_OBJECT (element, "pending flush stop");
		gst_pad_push_event(element->srcpad, gst_event_new_flush_stop());
	}

	/* do new segment event if needed */
	if(G_UNLIKELY(element->segment_pending)) {
		GstSegment *segment = gstlal_collect_pads_get_segment(element->collect);
		GstEvent *event;

		/* FIXME:  are other formats OK? */
		g_assert (segment->format == GST_FORMAT_TIME);
		element->segment = *segment;
		gst_segment_free(segment);

		/* FIXME, use rate/applied_rate as set on all sinkpads.
		 * - currently we just set rate as received from last seek-event
		 *
		 * When seeking we set the start and stop positions as given in the seek
		 * event. We also adjust offset & timestamp acordingly.
		 * This basically ignores all newsegments sent by upstream. */
		event = gst_event_new_new_segment_full(FALSE, element->segment.rate, 1.0, GST_FORMAT_TIME, element->segment.start, element->segment.stop, element->segment.start);
		if (element->segment.rate > 0.0) {
			element->timestamp = element->segment.start;
			element->offset = 0;
		} else {
			element->timestamp = element->segment.stop;
			element->offset = gst_util_uint64_scale_round(element->segment.stop - element->segment.start, element->rate, GST_SECOND);
		}
		GST_INFO_OBJECT (element, "seg_start %" G_GUINT64_FORMAT ", seg_end %" G_GUINT64_FORMAT, element->segment.start, element->segment.stop);
		GST_INFO_OBJECT (element, "timestamp %" G_GINT64_FORMAT ",new offset %" G_GINT64_FORMAT, element->timestamp, element->offset);

		if (event) {
			if (!gst_pad_push_event(element->srcpad, event)) {
				GST_WARNING_OBJECT(element->srcpad, "Sending event %p (%s) failed.", event, GST_EVENT_TYPE_NAME (event));
			}
			element->segment_pending = FALSE;
		} else {
			GST_WARNING_OBJECT(element->srcpad, "Creating new segment event for start:%" G_GINT64_FORMAT " end:%" G_GINT64_FORMAT " failed", element->segment.start, element->segment.stop);
		}
	}

	/* do other pending events, e.g., tags */
	if (G_UNLIKELY(element->pending_events)) {
		while (element->pending_events) {
			GstEvent *ev = GST_EVENT(element->pending_events->data);
			gst_pad_push_event(element->srcpad, ev);
			element->pending_events = g_list_remove(element->pending_events, ev);
		}
	}

	/* get the range of offsets (in the output stream) spanned by the
	 * available input buffers.
	 *
	 * determine the offsets for real. */
	if(!gstlal_collect_pads_get_earliest_times(element->collect, &t_start, &t_end)) {
		GST_ELEMENT_ERROR(element, STREAM, FORMAT, (NULL), ("cannot deduce input timestamp offset information"));
		goto bad_timestamps;
	}

	/* check for EOS */
	if(!GST_CLOCK_TIME_IS_VALID(t_start))
		goto eos;

	/* don't let time go backwards.  in principle we could be
	 * smart and handle this, but the audiorate element can be
	 * used to correct screwed up time series so there is no
	 * point in re-inventing its capabilities here. */
	earliest_input_offset = gst_util_uint64_scale_int_round(t_start - element->segment.start, element->rate, GST_SECOND);
	earliest_input_offset_end = gst_util_uint64_scale_int_round(t_end - element->segment.start, element->rate, GST_SECOND);

	if(earliest_input_offset < element->offset) {
		GST_ELEMENT_ERROR(element, STREAM, FORMAT, (NULL), ("detected time reversal in at least one input stream: expected nothing earlier than offset %" G_GUINT64_FORMAT ", found sample at offset %" G_GUINT64_FORMAT, element->offset, earliest_input_offset));
		goto bad_timestamps;
	}

	/* compute the number of samples for which all sink pads can
	 * contribute information.  0 does not necessarily mean EOS. */
	outlength = earliest_input_offset_end - earliest_input_offset;

	/* alloc a new output buffer of length samples, and
	 * set its offset. */

	GST_LOG_OBJECT(element, "requesting output buffer of %" G_GUINT64_FORMAT " samples", outlength);
	ret = gst_pad_alloc_buffer(element->srcpad, earliest_input_offset, outlength * element->float_unit_size,
		GST_PAD_CAPS(element->srcpad), &outbuf);
	if(ret != GST_FLOW_OK) {
		GST_ERROR_OBJECT(element, "could not create buffer of requested size %zd with offset %" G_GUINT64_FORMAT, outlength * element->float_unit_size, earliest_input_offset);
		goto no_buffer;
	}
	outbytes = GST_BUFFER_DATA(outbuf);
	memset(outbytes, 0, GST_BUFFER_SIZE(outbuf));
	GST_BUFFER_FLAG_SET(outbuf, GST_BUFFER_FLAG_GAP);

	snrbytes = malloc(outlength * element->complex_unit_size);
	if(snrbytes == NULL) {
		GST_ERROR_OBJECT(element, "could not malloc memory to store snr of requested size %zd", outlength * element->complex_unit_size);
		goto no_snr;
	}
	memset(snrbytes, 0, outlength * element->complex_unit_size);

	/*
	 * loop over input pads, getting chunks of data from each in turn.
	 */

	g_mutex_lock(element->coefficients_lock);
	while(!element->chifacs)
		g_cond_wait(element->coefficients_available, element->coefficients_lock);

	GST_LOG_OBJECT(element, "cycling through channels, offsets [%" G_GUINT64_FORMAT ", %" G_GUINT64_FORMAT ") (@ %d Hz) relative to %" GST_TIME_SECONDS_FORMAT " available", earliest_input_offset, earliest_input_offset_end, element->rate, GST_TIME_SECONDS_ARGS(element->segment.start));
	for(collected = pads->data; collected; collected = g_slist_next(collected)) {
		GstLALCollectData *collect_data = (GstLALCollectData *) collected->data;
		GstBuffer *inbuf;
		guint offset;
		guint64 inlength, numbytes, offsetbytes;

		/* (try to) get a buffer upto the desired end offset. */
		inbuf = gstlal_collect_pads_take_buffer_sync(pads, collect_data, t_end);

		/* add inbuf to list of input_buffers. this is prepended since
		 * the last pad linked returns the first buffer  and done here
		 * since we need to keep track of buffers that were NULL */
		input_buffers = g_slist_prepend(input_buffers, inbuf);

		/* NULL means EOS or an empty buffer so we still need to flush
		 * in case of an empty buffer. */
		if(inbuf == NULL) {
			GST_LOG_OBJECT(element, "channel %p: no bytes available", collect_data);
			continue;
		} else
			have_buffers = TRUE;

		/* determine the buffer's location relative to the desired
		 * range of offsets.  we've checked above that time hasn't
		 * gone backwards on any input buffer so gap can't be
		 * negative. */
		offset = gst_util_uint64_scale_int_round(GST_BUFFER_TIMESTAMP(inbuf) - element->segment.start, element->rate, GST_SECOND) - earliest_input_offset;
		offsetbytes = (guint64) offset * element->complex_unit_size;
		inlength = GST_BUFFER_OFFSET_END(inbuf) - GST_BUFFER_OFFSET(inbuf);
		numbytes = inlength * element->complex_unit_size;
		GST_LOG_OBJECT(element, "channel %p: retrieved %ld sample buffer at %" GST_TIME_FORMAT, collect_data, inlength, GST_TIME_ARGS(GST_BUFFER_TIMESTAMP(inbuf)));

		/* add to snr
		 *
		 * if buffer is not a gap and has non-zero length
		 * add to previous data and mark output as
		 * non-empty, otherwise do nothing. */
		if(!GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP) && inlength) {
			GST_LOG_OBJECT(element, "channel %p: mixing %zd bytes from data %p", collect_data, numbytes, GST_BUFFER_DATA(inbuf));
			add_complex128(snrbytes + offsetbytes, GST_BUFFER_DATA(inbuf), numbytes);
			GST_BUFFER_FLAG_UNSET(outbuf, GST_BUFFER_FLAG_GAP);
		} else
			GST_LOG_OBJECT(element, "channel %p: skipping %zd bytes from data %p", collect_data, numbytes, GST_BUFFER_DATA(inbuf));
	}

	/* can only happen when no pads to collect or all EOS */
	if(!have_buffers) {
		g_mutex_unlock(element->coefficients_lock);
		goto eos;
	}

	/* get the number of channels */
	numchannels = (guint) num_channels(element);

	/* get the number of timeslices */
	numtimeslices = (guint) num_timeslices(element);

	/* check that the number of channels and timeslices match the dimension of chifacs */
	if (num_channels(element) != element->channels) {
		g_mutex_unlock(element->coefficients_lock);
		GST_ERROR_OBJECT(element, "number of channels from caps negotiation X does not match second dimension of chifacs matrix Y: X = %i, Y = %i\n", element->channels, num_channels(element));
		goto bad_numchannels;
	}
	if (num_timeslices(element) != element->padcount) {
		g_mutex_unlock(element->coefficients_lock);
		GST_ERROR_OBJECT(element, "number of sink pads X does not match first dimension of chifacs matrix Y: X = %i, Y = %i\n", element->padcount, num_timeslices(element));
		goto bad_numtimeslices;
	}

	/* loop over buffers from input pads computing the time-slice chisq */
	for(timeslice = 0; timeslice < numtimeslices; timeslice++) {
		GstBuffer *inbuf = GST_BUFFER(input_buffers->data);
		size_t offset;
		size_t inlength;

		/* NULL means EOS or an empty buffer so we still need to flush in
		 * case of an empty buffer. */
		if(!inbuf) {
			input_buffers = g_slist_remove(input_buffers, inbuf);
			continue;
		}

		/*
		 * add chisq to outbuf
		 */

		/* determine the buffer's location relative to the desired
		 * range of offsets.  we've checked above that time hasn't
		 * gone backwards on any input buffer so gap can't be
		 * negative. */

		offset = inbuf ? gst_util_uint64_scale_int_round(GST_BUFFER_TIMESTAMP(inbuf) - element->segment.start, element->rate, GST_SECOND) - earliest_input_offset : 0;
		inlength = inbuf ? GST_BUFFER_OFFSET_END(inbuf) - GST_BUFFER_OFFSET(inbuf) : 0;

		/* FIXME: this double loop could be optimized such that there
		 * is only one gst_matrix_get per channel */
		for (sample = 0; sample < outlength; sample++) {
			double *outdata = (double *) (outbytes + element->float_unit_size * (offset + sample));
			const double complex *snrdata = (double complex *) (snrbytes + element->complex_unit_size * (offset + sample));
			const double complex *indata = (sample >= offset && sample < offset + inlength) ? (double complex *) (GST_BUFFER_DATA(inbuf) + element->complex_unit_size * (offset + sample)) : NULL;
			for (channel = 0; channel < numchannels; channel++) {
				double chifacs = gsl_matrix_get(element->chifacs, (size_t) timeslice, (size_t) channel);
				double complex chisq_num = (indata ? indata[channel] : 0.0) - chifacs * snrdata[channel];
				outdata[channel] += chisq_num * conj(chisq_num) / chifacs;
			}
		}

		/* unreference the inbuf as we go */
		if (inbuf)
			gst_buffer_unref(inbuf);
		input_buffers = g_slist_remove(input_buffers, inbuf);
	}

	g_mutex_unlock(element->coefficients_lock);

	/* FIXME:  this logic can't run backwards */
	/* set timestamps on the output buffer */
	element->offset = GST_BUFFER_OFFSET_END(outbuf) = GST_BUFFER_OFFSET(outbuf) + outlength;
	GST_BUFFER_OFFSET(outbuf) = earliest_input_offset;
	GST_BUFFER_TIMESTAMP(outbuf) = output_timestamp_from_offset(element, GST_BUFFER_OFFSET (outbuf));
	if (GST_BUFFER_OFFSET(outbuf) == 0 || GST_BUFFER_TIMESTAMP(outbuf) != element->timestamp)
		GST_BUFFER_FLAG_SET(outbuf, GST_BUFFER_FLAG_DISCONT);
	else
		GST_BUFFER_FLAG_UNSET(outbuf, GST_BUFFER_FLAG_DISCONT);
	GST_BUFFER_OFFSET_END(outbuf) = GST_BUFFER_OFFSET(outbuf) + outlength;
	element->timestamp = output_timestamp_from_offset(element, GST_BUFFER_OFFSET_END(outbuf));
	element->offset = GST_BUFFER_OFFSET_END(outbuf);
	GST_BUFFER_DURATION(outbuf) = element->timestamp - GST_BUFFER_TIMESTAMP(outbuf);

	/* push the buffer downstream. */
	GST_LOG_OBJECT(element, "pushing outbuf %p spanning %" GST_BUFFER_BOUNDARIES_FORMAT, outbuf, GST_BUFFER_BOUNDARIES_ARGS(outbuf));
	ret = gst_pad_push(element->srcpad, outbuf);
	GST_LOG_OBJECT (element, "pushed outbuf, result = %s", gst_flow_get_name(ret));

	/* free the snr memory */
	free(snrbytes);
	snrbytes = NULL;

	return ret;

	/* ERRORS */
bad_numtimeslices:
bad_numchannels:
	if (outbuf)
		gst_buffer_unref(outbuf);
	while(input_buffers) {
		GstBuffer *inbuf = GST_BUFFER(input_buffers->data);
		/* unreference the inbuf as we go */
		if (inbuf)
			gst_buffer_unref(inbuf);
		input_buffers = g_slist_remove(input_buffers, inbuf);
	}
no_snr:
	free(snrbytes);
	snrbytes = NULL;
no_buffer:
bad_timestamps:
	return GST_FLOW_ERROR;
eos:
	GST_DEBUG_OBJECT(element, "no data available, must be EOS");
	gst_pad_push_event(element->srcpad, gst_event_new_eos());

	if (outbuf)
		gst_buffer_unref(outbuf);
	while(input_buffers) {
		GstBuffer *inbuf = GST_BUFFER(input_buffers->data);
		/* unreference the inbuf as we go */
		if (inbuf)
			gst_buffer_unref(inbuf);
		input_buffers = g_slist_remove(input_buffers, inbuf);
	}
	free(snrbytes);
	snrbytes = NULL;

	return GST_FLOW_UNEXPECTED;
}


/*
 * ============================================================================
 *
 *                                Type Support
 *
 * ============================================================================
 */


/*
 * parent class handle
 */


static GstElementClass *parent_class = NULL;


/*
 * elementfactory information
 */


#define SRC_CAPS \
	"audio/x-raw-float, " \
	"rate = (int) [ 1, MAX ], " \
	"channels = (int) [ 1, MAX ], " \
	"endianness = (int) BYTE_ORDER, " \
	"width = (int) 64;"

#define SINK_CAPS \
	"audio/x-raw-complex, " \
	"rate = (int) [ 1, MAX ], " \
	"channels = (int) [ 1, MAX ], " \
	"endianness = (int) BYTE_ORDER, " \
	"width = (int) 128;"


static GstStaticPadTemplate
src_template =
	GST_STATIC_PAD_TEMPLATE(
		"src",
		GST_PAD_SRC,
		GST_PAD_ALWAYS,
		GST_STATIC_CAPS(SRC_CAPS)
	);


static GstStaticPadTemplate
sink_template =
	GST_STATIC_PAD_TEMPLATE(
		"sink%d",
		GST_PAD_SINK,
		GST_PAD_REQUEST,
		GST_STATIC_CAPS(SINK_CAPS)
	);


/*
 * reset element's internal state and start the collect pads on READY -->
 * PAUSED state change.  stop the collect pads on PAUSED --> READY state
 * change.
 */


static GstStateChangeReturn
change_state(GstElement *gstelement, GstStateChange transition)
{
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(gstelement);

	switch(transition) {
	case GST_STATE_CHANGE_NULL_TO_READY:
		break;
	case GST_STATE_CHANGE_READY_TO_PAUSED:
		element->timestamp = 0;
		element->offset = 0;
		element->flush_stop_pending = FALSE;
		element->segment_pending = TRUE;
		gst_segment_init(&element->segment, GST_FORMAT_UNDEFINED);
		gst_collect_pads_start(element->collect);
		break;
	case GST_STATE_CHANGE_PAUSED_TO_PLAYING:
		break;
	case GST_STATE_CHANGE_PAUSED_TO_READY:
		/* need to unblock the collectpads before calling the
		 * parent change_state so that streaming can finish */
		gst_collect_pads_stop(element->collect);
		break;
	default:
		break;
	}

	return GST_ELEMENT_CLASS(parent_class)->change_state(gstelement, transition);
}


/*
 * free internal memory
 */


static void
dispose(GObject *object)
{
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(object);

	if (element->collect) {
		gst_object_unref(element->collect);
		element->collect = NULL;
	}

	while (element->pending_events) {
		GstEvent *ev = GST_EVENT(element->pending_events->data);
		gst_event_unref(ev);
		element->pending_events = g_list_remove(element->pending_events, ev);
	}

	g_mutex_free(element->coefficients_lock);
	element->coefficients_lock = NULL;
	g_cond_free(element->coefficients_available);
	element->coefficients_available = NULL;
	if(element->chifacs) {
		gsl_matrix_free(element->chifacs);
		element->chifacs = NULL;
	}

	G_OBJECT_CLASS(parent_class)->dispose(object);
}


/*
 * class init
 */


static void
class_init(gpointer klass, gpointer class_data)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *gstelement_class = GST_ELEMENT_CLASS(klass);
	GSTLALTimeSliceChiSquareClass *gstlal_timeslicechisq_class = GSTLAL_TIMESLICECHISQUARE_CLASS(klass);

	gobject_class->set_property = set_property;
	gobject_class->get_property = get_property;
	gobject_class->dispose = dispose;

	g_object_class_install_property(
		gobject_class,
		ARG_CHIFACS,
		g_param_spec_value_array(
			"chifacs-matrix",
			"Chisquared Factors Matrix",
			"Array of complex chisquared factor vectors.  Number of vectors (rows) in matrix sets number of sink pads.  All vectors must have the same length.",
			g_param_spec_value_array(
				"chifacs",
				"Chisquared Factors",
				"Vector of chisquared factors. Number of elements must equal number of channels.",
				g_param_spec_double(
					"sample",
					"Sample",
					"Chifacs sample",
					-G_MAXDOUBLE, G_MAXDOUBLE, 0.0,
					G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
				),
				G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
			),
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);

	gst_element_class_add_pad_template(gstelement_class, gst_static_pad_template_get(&src_template));
	gst_element_class_add_pad_template(gstelement_class, gst_static_pad_template_get(&sink_template));
	gst_element_class_set_details_simple(
		gstelement_class,
		"Time-slice-based \\chi^{2}",
		"Filter",
		"A time-slice-based \\chi^{2} statistic",
		"Drew Keppel <drew.keppel@ligo.org>"
	);

	parent_class = g_type_class_peek_parent(gstlal_timeslicechisq_class);	

	gstelement_class->request_new_pad = GST_DEBUG_FUNCPTR(request_new_pad);
	gstelement_class->release_pad = GST_DEBUG_FUNCPTR(release_pad);
	gstelement_class->change_state = GST_DEBUG_FUNCPTR(change_state);
}


/*
 * instance init
 */


static void
instance_init(GTypeInstance *object, gpointer class)
{
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(object);
	GstPadTemplate *template = NULL;

	template = gst_static_pad_template_get(&src_template);
	element->srcpad = gst_pad_new_from_template(template, "src");
	gst_object_unref(template);

	gst_pad_set_getcaps_function(element->srcpad, GST_DEBUG_FUNCPTR(src_getcaps));
	gst_pad_set_setcaps_function(element->srcpad, GST_DEBUG_FUNCPTR(setcaps));
	gst_pad_set_query_function(element->srcpad, GST_DEBUG_FUNCPTR(query_function));
	gst_pad_set_event_function(element->srcpad, GST_DEBUG_FUNCPTR(src_event_function));
	gst_element_add_pad(GST_ELEMENT(element), element->srcpad);

	element->padcount = 0;

	element->collect = gst_collect_pads_new();
	gst_collect_pads_set_function(element->collect, GST_DEBUG_FUNCPTR(collected), element);

	element->rate = 0;
	element->float_unit_size = 0;
	element->complex_unit_size = 0;
	element->channels = 0;
	element->coefficients_lock = g_mutex_new();
	element->coefficients_available = g_cond_new();
	element->chifacs = NULL;
}


GType gstlal_timeslicechisquare_get_type(void)
{
	static GType element_type = 0;

	if(G_UNLIKELY(element_type == 0)) {
		static const GTypeInfo element_info = {
			.class_size = sizeof(GSTLALTimeSliceChiSquareClass),
			.class_init = class_init,
			.instance_size = sizeof(GSTLALTimeSliceChiSquare),
			.instance_init = instance_init,
		};

		element_type = g_type_register_static(GST_TYPE_ELEMENT, "GSTLALTimeSliceChiSquare", &element_info, 0);
		GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "gstlal_timeslicechisq", 0, "");
	}

	return element_type;
}

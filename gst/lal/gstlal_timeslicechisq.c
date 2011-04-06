/* GStreamer
 * Copyright (C) 1999,2000 Erik Walthinsen <omega@cse.ogi.edu>
 *                    2001 Thomas <thomas@apestaart.org>
 *               2005,2006 Wim Taymans <wim@fluendo.com>
 *               2008 Kipp Cannon <kipp.cannon@ligo.org>
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


#include <gstlal.h>
#include <gstlalcollectpads.h>
#include "gstlal_timeslicechisq.h"


#define GST_CAT_DEFAULT gstlal_timeslicechisq_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


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


static int num_channels(const GSTLALTimeSliceChiSquare *element)
{
	return element->chifacs->size2;
}


/*
 * return the number of timeslices
 */


static int num_timeslices(const GSTLALTimeSliceChiSquare *element)
{
	return element->chifacs->size1;
}


/*
 * function to add doubles
 */


static void add_complex128(gpointer out, const gpointer in, size_t bytes)
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


enum property {
	ARG_0,
	ARG_CHIFACS
};


static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *psspec)
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

		/*
		 * number of channels has changed, force a caps renegotiation
		 */

		if(num_channels(element) != channels) {
			/* FIXME: is this correct? */
			gst_pad_set_caps(element->srcpad, NULL);
		}

		/*
		 * signal availability of new chifacs vector
		 */

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


static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *psspec)
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


static GstCaps *sink_getcaps(GstPad *pad)
{
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(GST_PAD_PARENT(pad));
	GstCaps *peercaps;
	GstCaps *caps;

	/*
	 * get our own allowed caps.  use the fixed caps function to avoid
	 * recursing back into this function.
	 */

	GST_OBJECT_LOCK(element);
	caps = gst_pad_get_fixed_caps_func(pad);

	/*
	 * get the allowed caps from the downstream peer.
	 * if the peer has caps, intersect.
	 */

	peercaps = gst_pad_peer_get_caps(element->srcpad);
	if(peercaps) {
		GstCaps *result;
		gint width;
		GValue value = {0,};
		g_value_init(&value, G_TYPE_INT);
		GstStructure *peercaps_struct = gst_caps_steal_structure(peercaps, 0);
		gst_structure_get_int(peercaps_struct, "width", &width);
		gst_structure_set_name(peercaps_struct, (const gchar *) "audio/x-raw-complex");
		gst_caps_append_structure(peercaps, peercaps_struct);
		g_value_set_int(&value, width * 2);
		gst_caps_set_value(peercaps, "width", &value);

		GST_DEBUG_OBJECT(element, "intersecting %" GST_PTR_FORMAT " and %" GST_PTR_FORMAT, caps, peercaps);
		result = gst_caps_intersect(peercaps, caps);
		gst_caps_unref(peercaps);
		gst_caps_unref(caps);
		caps = result;
		GST_DEBUG_OBJECT(element, "intersection %" GST_PTR_FORMAT, caps);
	}
	GST_OBJECT_UNLOCK(element);

	/*
	 * done
	 */

	return caps;
}


static GstCaps *src_getcaps(GstPad *pad)
{
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(GST_PAD_PARENT(pad));
	GList *padlist = NULL;
	GstCaps *peercaps;
	GstCaps *caps;

	/*
	 * get our own allowed caps.  use the fixed caps function to avoid
	 * recursing back into this function.
	 */

	GST_OBJECT_LOCK(element);
	caps = gst_pad_get_fixed_caps_func(pad);

	/*
	 * get the allowed caps from the upstream peer.
	 * if the peer has caps, intersect.
	 */

	for(padlist = GST_ELEMENT(element)->pads; padlist; padlist = g_list_next(padlist)) {
		GstPad *otherpad = GST_PAD(padlist->data);
		if(otherpad != pad) {
			peercaps = gst_pad_peer_get_caps(otherpad);
			if(peercaps) {
				GstCaps *result;
				gint width;
				GValue value = {0,};
				g_value_init(&value, G_TYPE_INT);
				GstStructure *peercaps_struct = gst_caps_steal_structure(peercaps, 0);
				gst_structure_get_int(peercaps_struct, "width", &width);
				gst_structure_set_name(peercaps_struct, (const gchar *) "audio/x-raw-float");
				gst_caps_append_structure(peercaps, peercaps_struct);
				g_value_set_int(&value, width / 2);
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

	/*
	 * done
	 */

	return caps;
}


/*
 * the first caps we receive on any of the sinkpads will define the caps
 * for all the other sinkpads because we can only mix streams with the same
 * caps.
 */


static gboolean setcaps(GstPad *pad, GstCaps *caps)
{
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(GST_PAD_PARENT(pad));
	GList *padlist = NULL;
	GstStructure *structure = NULL;
	GstCaps *sinkcaps = gst_caps_copy(caps);
	GstCaps *srccaps = gst_caps_copy(caps);

	GST_LOG_OBJECT(element, "setting caps on pad %s:%s to %" GST_PTR_FORMAT, GST_DEBUG_PAD_NAME(pad), caps);

	/*
	 * parse caps
	 */

	structure = gst_caps_get_structure(caps, 0);
	gst_structure_get_int(structure, "rate", &element->rate);
	gst_structure_get_int(structure, "channels", &element->channels);

	/*
	 * pre-calculate bytes / sample
	 */

	element->float_unit_size = 8 * element->channels;
	element->complex_unit_size = 16 * element->channels;

	/*
	 * create the caps appropriate for src and sink pads
	 */

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

	/*
	 * loop over all of the element's pads (source and sink), and set
	 * them all to the appropriate format.
	 */

	/* FIXME, see if the other pads can accept the format. Also lock
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
		if(gst_pad_get_direction(pad) == GST_PAD_SINK)
			gstlal_collect_pads_set_unit_size(pad, element->complex_unit_size);
	}

	/*
	 * done
	 */

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


static GstClockTime output_timestamp_from_offset(const GSTLALTimeSliceChiSquare *element, guint64 offset)
{
	return element->segment.start + gst_util_uint64_scale_int_round(offset, GST_SECOND, element->rate);
}


/*
 * The duration query should reflect how long you will produce data, that
 * is the amount of stream time until you will emit EOS.  This is the max
 * of all the durations of upstream since we emit EOS when all of them
 * finished.
 *
 * FIXME: This is true for asynchronous mixing.  For synchronous mixing, an
 * input stream might be delayed so that although it reports X seconds
 * remaining until EOS, that data might not have started being mixed into
 * the output yet and won't start for another Y seconds, so our output has
 * X + Y seconds reminaing in it.  We know that delay so we should be able
 * to incorporate it in our answer.  However, even if all streams are being
 * mixed, i.e., the output timestamp has advanced into all input segments
 * so that none are being held back, then taking the maximum of the
 * upstream durations is mostly correct but there is still the possibility
 * that discontinuities will occur in one or more input streams which
 * become gaps that get filled by other input streams so that the total
 * duration of our output is larger than the durations of any of the
 * upstream peers.  In general, there is no way to compute the duration of
 * our output without advance knowledge of the intervals of time for which
 * each input will provide data, which we don't have.  In the synchronous
 * case, the duration will always have to be an approximation that becomes
 * more accurate the closer we get to the true EOS.
 */


static gboolean query_duration(GSTLALTimeSliceChiSquare *element, GstQuery *query)
{
	GstIterator *it = NULL;
	gint64 max = -1;
	GstFormat format;
	gboolean success = TRUE;
	gboolean done = FALSE;

	/*
	 * parse duration query format
	 */

	gst_query_parse_duration(query, &format, NULL);

	/*
	 * iterate over sink pads
	 */

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

			/*
			 * query upstream peer for duration
			 */

			if(gst_pad_query_peer_duration(pad, &format, &duration)) {
				/*
				 * query succeeded
				 */

				if(duration == -1) {
					/*
					 * unknown duration --> the
					 * duration of our output is
					 * unknown
					 */

					max = duration;
					done = TRUE;
				} else if(duration > max) {
					/*
					 * take largest duration
					 */

					max = duration;
				}
			} else {
				/*
				 * query failed
				 */

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
		/*
		 * store the max
		 */

		GST_DEBUG_OBJECT(element, "Total duration in format %s: %" GST_TIME_FORMAT, gst_format_get_name(format), GST_TIME_ARGS(max));
		gst_query_set_duration(query, format, max);
	}

	return success;
}


static gboolean query_latency(GSTLALTimeSliceChiSquare *element, GstQuery *query)
{
	GstIterator *it = NULL;
	GstClockTime min = 0;
	GstClockTime max = GST_CLOCK_TIME_NONE;
	gboolean live = FALSE;
	gboolean success = TRUE;
	gboolean done = FALSE;

	/*
	 * iterate over sink pads
	 */

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

			/* 
			 * query upstream peer for latency
			 */

			if(gst_pad_peer_query(pad, peerquery)) {
				/*
				 * query succeeded
				 */

				GstClockTime min_cur;
				GstClockTime max_cur;
				gboolean live_cur;

				gst_query_parse_latency(peerquery, &live_cur, &min_cur, &max_cur);

				/*
				 * take the largest of the latencies
				 */

				if(min_cur > min)
					min = min_cur;

				if(max_cur != GST_CLOCK_TIME_NONE && ((max != GST_CLOCK_TIME_NONE && max_cur > max) || (max == GST_CLOCK_TIME_NONE)))
					max = max_cur;

				/*
				 * we're live if any upstream peer is live
				 */

				live |= live_cur;
			} else {
				/*
				 * query failed
				 */

				success = FALSE;
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


static gboolean query_function(GstPad *pad, GstQuery *query)
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
			gst_query_set_position(query, format, output_timestamp_from_offset(element, element->offset));
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


/*
 * helper function used by forward_event() (see below)
 */


static gboolean forward_event_func(GstPad *pad, GValue *ret, GstEvent *event)
{
	gst_event_ref(event);
	GST_LOG_OBJECT(pad, "pushing event %p (%s) on pad %s:%s", event, GST_EVENT_TYPE_NAME(event), GST_DEBUG_PAD_NAME(pad));
	if(!gst_pad_push_event(pad, event)) {
		g_value_set_boolean(ret, FALSE);
		GST_WARNING_OBJECT(pad, "event %p (%s) push failed.", event, GST_EVENT_TYPE_NAME(event));
	} else
		GST_LOG_OBJECT(pad, "event %p (%s) pushed.", event, GST_EVENT_TYPE_NAME(event));
	gst_object_unref(pad);
	return TRUE;
}


/*
 * forwards the event to all sinkpads.  takes ownership of the event.
 * returns TRUE if the event could be forwarded on all sinkpads, returns
 * FALSE if not.
 */


static gboolean forward_event(GSTLALTimeSliceChiSquare *element, GstEvent *event)
{
	GstIterator *it = NULL;
	GValue vret = {0};

	GST_LOG_OBJECT(element, "forwarding event %p (%s)", event, GST_EVENT_TYPE_NAME(event));

	g_value_init(&vret, G_TYPE_BOOLEAN);
	g_value_set_boolean(&vret, TRUE);
	it = gst_element_iterate_sink_pads(GST_ELEMENT_CAST(element));
	gst_iterator_fold(it, (GstIteratorFoldFunction) forward_event_func, &vret, event);
	gst_iterator_free(it);
	gst_event_unref(event);

	return g_value_get_boolean(&vret);
}


/*
 * src pad event handler
 */


static gboolean src_event_function(GstPad *pad, GstEvent *event)
{
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(gst_pad_get_parent(pad));
	gboolean result;

	switch (GST_EVENT_TYPE(event)) {
	case GST_EVENT_QOS:
	case GST_EVENT_NAVIGATION:
		/*
		 * not handled. QoS might be tricky, navigation is
		 * pointless.
		 */

		result = FALSE;
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
			 * start flush downstream.  the flush will be done
			 * when all pads received a FLUSH_STOP.
			 */

			gst_pad_push_event(element->srcpad, gst_event_new_flush_start());
		}

		/*
		 * wait for the collected to be finished and mark a new
		 * segment
		 */

		GST_OBJECT_LOCK(element->collect);
		element->segment_pending = TRUE;
		if(flush) {
			/* Yes, we need to call _set_flushing again *WHEN* the streaming threads
			 * have stopped so that the cookie gets properly updated. */
			gst_collect_pads_set_flushing(element->collect, TRUE);
		}
		element->flush_stop_pending = flush;
		GST_OBJECT_UNLOCK(element->collect);

		result = forward_event(element, event);
		break;
	}

	default:
		/*
		 * forward the rest.
		 */

		result = forward_event(element, event);
		break;
	}

	/*
	 * done
	 */

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


static gboolean sink_event_function(GstPad *pad, GstEvent *event)
{
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(gst_pad_get_parent(pad));

	GST_DEBUG("got event %p (%s) on pad %s:%s", event, GST_EVENT_TYPE_NAME(event), GST_DEBUG_PAD_NAME(pad));

	/*
	 * handle events
	 */

	switch (GST_EVENT_TYPE(event)) {
	case GST_EVENT_FLUSH_STOP:
		/*
		 * mark a pending new segment. This event is synchronized
		 * with the streaming thread so we can safely update the
		 * variable without races. It's somewhat weird because we
		 * assume the collectpads forwarded the FLUSH_STOP past us
		 * and downstream (using our source pad, the bastard!).
		 */

		GST_OBJECT_LOCK(element->collect);
		element->segment_pending = TRUE;
		element->flush_stop_pending = FALSE;
		GST_OBJECT_UNLOCK(element->collect);
		break;

	default:
		break;
	}

	/*
	 * now chain to GstCollectPads handler to take care of the rest.
	 */

	gst_object_unref(element);
	return element->collect_event(pad, event);
}


/*
 * ============================================================================
 *
 *                            Add/Remove Sink Pad
 *
 * ============================================================================
 */


static GstPad *request_new_pad(GstElement *gstelement, GstPadTemplate *templ, const gchar *unused)
{
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(gstelement);
	gchar *name = NULL;
	GstPad *newpad = NULL;
	GstLALCollectData *data = NULL;
	gint padcount;

	/*
	 * new pads can only be sink pads
	 */

	if(templ->direction != GST_PAD_SINK) {
		g_warning("gstlal_timeslicechisq: request new pad that is not a SINK pad\n");
		goto not_sink;
	}


	/*
	 * create a new pad
	 */

	padcount = g_atomic_int_exchange_and_add(&element->padcount, 1);
	name = g_strdup_printf(GST_PAD_TEMPLATE_NAME_TEMPLATE(templ), padcount);
	newpad = gst_pad_new_from_template(templ, name);
	GST_DEBUG_OBJECT(element, "request new pad %p (%s)", newpad, name);
	g_free(name);

	/*
	 * configure new pad
	 */

	gst_pad_set_getcaps_function(newpad, GST_DEBUG_FUNCPTR(sink_getcaps));
	gst_pad_set_setcaps_function(newpad, GST_DEBUG_FUNCPTR(setcaps));

	/*
	 * add pad to collect pads object
	 */

	data = gstlal_collect_pads_add_pad(element->collect, newpad, sizeof(*data));
	if(!data) {
		GST_DEBUG_OBJECT(element, "could not add pad to collectpads object");
		goto could_not_add_to_collectpads;
	}

	/*
	 * FIXME: hacked way to override/extend the event function of
	 * GstCollectPads;  because it sets its own event function giving
	 * the element (us) no access to events
	 */

	element->collect_event = (GstPadEventFunction) GST_PAD_EVENTFUNC(newpad);
	gst_pad_set_event_function(newpad, GST_DEBUG_FUNCPTR(sink_event_function));

	/*
	 * add pad to element (takes ownership of the pad).
	 */

	if(!gst_element_add_pad(GST_ELEMENT(element), newpad)) {
		GST_DEBUG_OBJECT(element, "could not add pad to element");
		goto could_not_add_to_element;
	}

	/*
	 * done
	 */

	return newpad;

	/*
	 * ERRORS
	 */

could_not_add_to_element:
	gstlal_collect_pads_remove_pad(element->collect, newpad);
could_not_add_to_collectpads:
	gst_object_unref(newpad);
not_sink:
	return NULL;
}


static void release_pad(GstElement *gstelement, GstPad *pad)
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
	GstClockTime t_start, t_end;
	guint64 earliest_input_offset, earliest_input_offset_end;
	guint64 length;
	unsigned int timeslice;
	void *snrbytes = NULL;
	GstBuffer *outbuf = NULL;
	GstBuffer *inbufs[num_timeslices(element)];
	gpointer outbytes = NULL;
	GstFlowReturn result;
	unsigned int numchannels, numtimeslices, channel, sample;

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
		GstSegment *segment = gstlal_collect_pads_get_segment(element->collect);
		if(!segment) {
			/* FIXME:  failure getting bounding segment, do
			 * something about it */
		}
		element->segment = *segment;
		element->offset = 0;
		gst_segment_free(segment);
	}

	/*
	 * get the range of offsets (in the output stream) spanned by the
	 * available input buffers.
	 */

	/*
	 * determine the offsets for real.
	 */

	if(!gstlal_collect_pads_get_earliest_times(element->collect, &t_start, &t_end, element->rate)) {
		GST_ERROR_OBJECT(element, "cannot deduce input timestamp offset information");
		result = GST_FLOW_ERROR;
		goto error;
	}

	/*
	 * check for EOS
	 */

	if(!GST_CLOCK_TIME_IS_VALID(t_start))
		goto eos;

	/*
	 * don't let time go backwards.  in principle we could be
	 * smart and handle this, but the audiorate element can be
	 * used to correct screwed up time series so there is no
	 * point in re-inventing its capabilities here.
	 */

	earliest_input_offset = gst_util_uint64_scale_int_round(t_start - element->segment.start, element->rate, GST_SECOND);
	earliest_input_offset_end = gst_util_uint64_scale_int_round(t_end - element->segment.start, element->rate, GST_SECOND);

	if(earliest_input_offset < element->offset) {
		GST_ERROR_OBJECT(element, "detected time reversal in at least one input stream:  expected nothing earlier than offset %" G_GUINT64_FORMAT ", found sample at offset %" G_GUINT64_FORMAT, element->offset, earliest_input_offset);
		result = GST_FLOW_ERROR;
		goto error;
	}

	/*
	 * compute the number of samples for which all sink pads can
	 * contribute information.  0 does not necessarily mean EOS.
	 */

	length = earliest_input_offset_end - earliest_input_offset;

	/*
	 * alloc a new output buffer of length samples, and
	 * set its offset.
	 */
	/* FIXME:  if this returns a short buffer we're
	 * sort of screwed.  a code re-organization could
	 * fix it:  request buffer before entering the loop
	 * and figure out a different way to check for EOS
	 */

	GST_LOG_OBJECT(element, "requesting output buffer of %" G_GUINT64_FORMAT " samples", length);
	result = gst_pad_alloc_buffer(element->srcpad, earliest_input_offset, length * element->float_unit_size,
		GST_PAD_CAPS(element->srcpad), &outbuf);
	if(result != GST_FLOW_OK) {
		/* FIXME: handle failure */
		outbuf = NULL;
	}
	outbytes = GST_BUFFER_DATA(outbuf);
	memset(outbytes, 0, GST_BUFFER_SIZE(outbuf));
	GST_BUFFER_FLAG_SET(outbuf, GST_BUFFER_FLAG_GAP);

	snrbytes = malloc(length * element->complex_unit_size);
	if(snrbytes == NULL) {
		/* FIXME: handle failure */
	}
	memset(snrbytes, 0, length * element->complex_unit_size);

	/*
	 * loop over input pads, getting chunks of data from each in turn.
	 */

	numtimeslices = element->padcount;
	timeslice = 0;

	GST_LOG_OBJECT(element, "cycling through channels, offsets [%" G_GUINT64_FORMAT ", %" G_GUINT64_FORMAT ") (@ %d Hz) relative to %" GST_TIME_SECONDS_FORMAT " available", earliest_input_offset, earliest_input_offset_end, element->rate, GST_TIME_SECONDS_ARGS(element->segment.start));
	for(collected = pads->data, timeslice = 0; collected; collected = g_slist_next(collected), timeslice++) {
		GstLALCollectData *data = collected->data;
		GstBuffer *inbuf;
		size_t gap;
		size_t len;

		/*
		 * (try to) get a buffer upto the desired end offset.
		 */

		inbuf = gstlal_collect_pads_take_buffer_sync(pads, data, t_end, element->rate);

		/*
		 * NULL means EOS.
		 */

		if(!inbuf) {
			GST_LOG_OBJECT(element, "channel %p: no bytes available (EOS)", data);
			continue;
		}

		/*
		 * determine the buffer's location relative to the desired
		 * range of offsets.  we've checked above that time hasn't
		 * gone backwards on any input buffer so gap can't be
		 * negative.  if not doing synchronous mixing, the buffer
		 * starts now.
		 */

		gap = (gst_util_uint64_scale_int_round(GST_BUFFER_TIMESTAMP(inbuf) - element->segment.start, element->rate, GST_SECOND) - earliest_input_offset) * element->complex_unit_size;
		len = GST_BUFFER_SIZE(inbuf);

		/*
		 * add to snr
		 */

		/*
		 * if buffer is not a gap and has non-zero length
		 * add to previous data and mark output as
		 * non-empty, otherwise do nothing.
		 */

		if(!GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP) && len) {
			GST_LOG_OBJECT(element, "channel %p: mixing %zd bytes from data %p", data, len, GST_BUFFER_DATA(inbuf));
			add_complex128(snrbytes + gap, GST_BUFFER_DATA(inbuf), len);
			GST_BUFFER_FLAG_UNSET(outbuf, GST_BUFFER_FLAG_GAP);
		} else
			GST_LOG_OBJECT(element, "channel %p: skipping %zd bytes from data %p", data, len, GST_BUFFER_DATA(inbuf));

		/*
		 * add inbuf to list of inbufs
		 * this is done in reverse order here since the last pad link
		 * returns the first buffer
		 */

		inbufs[numtimeslices - timeslice - 1] = inbuf;
	}

	/*
	 * get the number of channels
	 */

	numchannels = (guint) num_channels(element);

	/*
	 * loop over buffers from input pads computing the time-slice chisq
	 */

	g_mutex_lock(element->coefficients_lock);
	while(!element->chifacs)
		g_cond_wait(element->coefficients_available, element->coefficients_lock);

	/*
	 * check that the number of channels and timeslices match the dimension of chifacs
	 */

	if (num_channels(element) != element->channels) {
		g_mutex_unlock(element->coefficients_lock);
		GST_ERROR_OBJECT(element, "number of channels from caps negotiation X does not match second dimension of chifacs matrix Y: X = %i, Y = %i\n", element->channels, num_channels(element));
		result = GST_FLOW_ERROR;
		goto error;
	}

	if (num_timeslices(element) != element->padcount) {
		g_mutex_unlock(element->coefficients_lock);
		GST_ERROR_OBJECT(element, "number of sink pads X does not match first dimension of chifacs matrix Y: X = %i, Y = %i\n", element->padcount, num_timeslices(element));
		result = GST_FLOW_ERROR;
		goto error;
	}

	for(timeslice = 0; timeslice < numtimeslices; timeslice++) {
		GstBuffer *inbuf = inbufs[timeslice];
		size_t gap;
		size_t len;

		/*
		 * NULL means EOS.
		 */

		if(!inbuf) {
			continue;
		}

		/*
		 * add chisq to outbuf
		 */

		/*
		 * determine the buffer's location relative to the desired
		 * range of offsets.  we've checked above that time hasn't
		 * gone backwards on any input buffer so gap can't be
		 * negative.  if not doing synchronous mixing, the buffer
		 * starts now.
		 */

		gap = (gst_util_uint64_scale_int_round(GST_BUFFER_TIMESTAMP(inbuf) - element->segment.start, element->rate, GST_SECOND) - earliest_input_offset);
		len = GST_BUFFER_SIZE(inbuf) / element->complex_unit_size;

		if(!GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP) && len) {
			for (sample = gap; sample < len; sample++) {
				double *outdata = (double *) (outbytes + sample * element->float_unit_size);
				const double complex *snrdata = (double complex *) (snrbytes + sample * element->complex_unit_size);
				const double complex *indata = (double complex *) (GST_BUFFER_DATA(inbuf) + sample * element->complex_unit_size);
				for (channel = 0; channel < numchannels; channel++) {
					double chifacs = gsl_matrix_get(element->chifacs, (size_t) timeslice, (size_t) channel);
					double complex chisq_num = indata[channel] - chifacs * snrdata[channel];

					outdata[channel] += chisq_num * conj(chisq_num) / chifacs;
				}
			}
		}

		/*
		 * unreference the inbuf as we go
		 */

		gst_buffer_unref(inbuf);
		inbufs[timeslice] = NULL;
	}

	g_mutex_unlock(element->coefficients_lock);

	/*
	 * free the memomry associated with the snr
	 */

	free(snrbytes);
	snrbytes = NULL;

	/*
	 * can only happen when no pads to collect or all EOS
	 */

	if(!outbuf)
		goto eos;

	/*
	 * check for discontinuity.
	 */

	if(element->offset != GST_BUFFER_OFFSET(outbuf))
		GST_BUFFER_FLAG_SET(outbuf, GST_BUFFER_FLAG_DISCONT);

	/*
	 * set the timestamp, end offset, and duration.  computing the
	 * duration the way we do here ensures that if some downstream
	 * element adds all the buffer durations together they'll stay in
	 * sync with the timestamp.  the end offset is saved for comparison
	 * against the next start offset to watch for discontinuities.
	 */

	element->offset = GST_BUFFER_OFFSET_END(outbuf) = GST_BUFFER_OFFSET(outbuf) + length;
	GST_BUFFER_TIMESTAMP(outbuf) = output_timestamp_from_offset(element, GST_BUFFER_OFFSET(outbuf));
	GST_BUFFER_DURATION(outbuf) = output_timestamp_from_offset(element, GST_BUFFER_OFFSET_END(outbuf)) - GST_BUFFER_TIMESTAMP(outbuf);

	/*
	 * precede the buffer with a new_segment event if one is pending
	 */
	/* FIXME, use rate/applied_rate as set on all sinkpads.  currently
	 * we just set rate as received from last seek-event We could
	 * potentially figure out the duration as well using the current
	 * segment positions and the stated stop positions. */

	if(element->segment_pending) {
		/* FIXME:  the segment start time is almost certainly
		 * incorrect */
		GstEvent *event = gst_event_new_new_segment_full(FALSE, element->segment.rate, 1.0, GST_FORMAT_TIME, element->segment.start, element->segment.stop, GST_BUFFER_TIMESTAMP(outbuf));

		if(!event) {
			/* FIXME:  failure getting event, do something
			 * about it */
		}

		/*
		 * gst_pad_push_event() returns a boolean indicating
		 * whether or not the event was handled.  we ignore this.
		 * whether or not the new segment event can be handled is
		 * not our problem to worry about, our responsibility is
		 * just to send it.
		 */

		gst_pad_push_event(element->srcpad, event);
		element->segment_pending = FALSE;
	}

	/*
	 * push the buffer downstream.
	 */

	GST_LOG_OBJECT(element, "pushing outbuf, timestamp %" GST_TIME_SECONDS_FORMAT ", offset %" G_GUINT64_FORMAT, GST_TIME_SECONDS_ARGS(GST_BUFFER_TIMESTAMP(outbuf)), GST_BUFFER_OFFSET(outbuf));
	return gst_pad_push(element->srcpad, outbuf);

	/*
	 * ERRORS
	 */

eos:
	GST_DEBUG_OBJECT(element, "no data available (EOS)");
	gst_pad_push_event(element->srcpad, gst_event_new_eos());
	return GST_FLOW_UNEXPECTED;

error:
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


static GstStaticPadTemplate src_template =
	GST_STATIC_PAD_TEMPLATE(
		"src",
		GST_PAD_SRC,
		GST_PAD_ALWAYS,
		GST_STATIC_CAPS(SRC_CAPS)
	);


static GstStaticPadTemplate sink_template =
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


static GstStateChangeReturn change_state(GstElement *gstelement, GstStateChange transition)
{
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(gstelement);

	switch(transition) {
	case GST_STATE_CHANGE_NULL_TO_READY:
		break;

	case GST_STATE_CHANGE_READY_TO_PAUSED:
		element->segment_pending = TRUE;
		gst_segment_init(&element->segment, GST_FORMAT_UNDEFINED);
		element->offset = 0;
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


static void finalize(GObject *object)
{
	GSTLALTimeSliceChiSquare *element = GSTLAL_TIMESLICECHISQUARE(object);

	gst_object_unref(element->collect);
	element->collect = NULL;

	g_mutex_free(element->coefficients_lock);
	element->coefficients_lock = NULL;
	g_cond_free(element->coefficients_available);
	element->coefficients_available = NULL;
	if(element->chifacs) {
		gsl_matrix_free(element->chifacs);
		element->chifacs = NULL;
	}

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


/*
 * class init
 */


static void class_init(gpointer klass, gpointer class_data)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *gstelement_class = GST_ELEMENT_CLASS(klass);
	GSTLALTimeSliceChiSquareClass *gstlal_timeslicechisq_class = GSTLAL_TIMESLICECHISQUARE_CLASS(klass);

	gobject_class->set_property = set_property;
	gobject_class->get_property = get_property;
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

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


static void instance_init(GTypeInstance *object, gpointer class)
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
	gst_element_add_pad(GST_ELEMENT(object), element->srcpad);

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


/*
 * create/get type ID
 */


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

		element_type = g_type_register_static(GST_TYPE_ELEMENT, "GSTLALTimeSliceChisq", &element_info, 0);
		GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "gstlal_timeslicechisq", 0, "");
	}

	return element_type;
}


/*
 * ============================================================================
 *
 *                              Plug-in Support
 *
 * ============================================================================
 */


#if 0
static gboolean plugin_init(GstPlugin * plugin)
{
	return gst_element_register(plugin, "gstlal_timeslicechisq", GST_RANK_NONE, GSTLAL_TIMESLICECHISQUARE_TYPE);
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, "gstlal_timeslicechisq", "Compute time-slice chisq from multiple streams", plugin_init, VERSION, "LGPL", GST_PACKAGE_NAME, GST_PACKAGE_ORIGIN)
#endif

/* GStreamer
 * Copyright (C) 1999,2000 Erik Walthinsen <omega@cse.ogi.edu>
 *                    2001 Thomas <thomas@apestaart.org>
 *               2005,2006 Wim Taymans <wim@fluendo.com>
 *               2008 Kipp Cannon <kipp.cannon@ligo.org>
 *               2010 Drew Keppel <drew.keppel@ligo.ord>
 *
 * gstlal_multiplier.c: Multiplyer element, N in, one out, samples are multiplied
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


/**
 * SECTION:element-multipler
 *
 * The multiplier allows to mix several streams into one by multiplying the
 * data.  Mixed data is clamped to the min/max values of the data format.
 *
 * If the element's sync property is TRUE the streams are mixed with the
 * timestamps synchronized.  If the sync property is FALSE (the default, to be
 * compatible with older versions), then the first samples from each stream are
 * multiplied to produce the first sample of the output, the second samples are
 * multiplied to produce the second sample of the output, and so on.
 * 
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch audiotestsrc freq=100 ! lal_multiplier name=mix ! audioconvert ! alsasink audiotestsrc freq=500 ! mix.
 * ]| This pipeline produces two sine waves mixed together.
 * </refsect2>
 *
 * Last reviewed on 2006-05-09 (0.10.7)
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
#include <complex.h>
#include <math.h>
#include <string.h>
#include <gst/gst.h>
#include <gst/base/gstcollectpads.h>
#include <gst/audio/audio.h>
#include "gstlal_multiplier.h"
#include <gstlal/gstlalcollectpads.h>
#include <gstlal/gstlal_debug.h>


#define GST_CAT_DEFAULT gstlal_multiplier_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


/*
 * ============================================================================
 *
 *                             Multiplier Loops
 *
 * ============================================================================
 */


/*
 * highest positive/lowest negative x-bit value we can use for clamping
 */


#define MAX_INT_32  ((gint32) (0x7fffffff))
#define MAX_INT_16  ((gint16) (0x7fff))
#define MAX_INT_8   ((gint8)  (0x7f))
#define MAX_UINT_32 ((guint32)(0xffffffff))
#define MAX_UINT_16 ((guint16)(0xffff))
#define MAX_UINT_8  ((guint8) (0xff))


#define MIN_INT_32  ((gint32) (0x80000000))
#define MIN_INT_16  ((gint16) (0x8000))
#define MIN_INT_8   ((gint8)  (0x80))
#define MIN_UINT_32 ((guint32)(0x00000000))
#define MIN_UINT_16 ((guint16)(0x0000))
#define MIN_UINT_8  ((guint8) (0x00))


/*
 * clipping versions
 */


#define MAKE_FUNC(name, type, ttype, min, max)                               \
static void name(gpointer out, const gpointer in, size_t bytes)              \
{                                                                            \
        type *_out = out;                                                    \
	const type *_in = in;                                                \
	for(bytes /= sizeof(type); bytes--; _in++, _out++)                   \
		*_out = CLAMP((ttype) *_out * (ttype) *_in, min, max);       \
}


/*
 * non-clipping versions
 */


#define MAKE_FUNC_NC(name, type, ttype)                         \
static void name(gpointer out, const gpointer in, size_t bytes) \
{                                                               \
        type *_out = out;                                       \
	const type *_in = in;                                   \
	for(bytes /= sizeof(type); bytes--; _in++, _out++)      \
		*_out = (ttype) *_out * (ttype) *_in;           \
}


/* *INDENT-OFF* */
MAKE_FUNC(multiply_int32, gint32, gint64, MIN_INT_32, MAX_INT_32)
MAKE_FUNC(multiply_int16, gint16, gint32, MIN_INT_16, MAX_INT_16)
MAKE_FUNC(multiply_int8, gint8, gint16, MIN_INT_8, MAX_INT_8)
MAKE_FUNC(multiply_uint32, guint32, guint64, MIN_UINT_32, MAX_UINT_32)
MAKE_FUNC(multiply_uint16, guint16, guint32, MIN_UINT_16, MAX_UINT_16)
MAKE_FUNC(multiply_uint8, guint8, guint16, MIN_UINT_8, MAX_UINT_8)
MAKE_FUNC_NC(multiply_complex128, double complex, double complex)
MAKE_FUNC_NC(multiply_complex64, float complex, float complex)
MAKE_FUNC_NC(multiply_float64, gdouble, gdouble)
MAKE_FUNC_NC(multiply_float32, gfloat, gfloat)
/* *INDENT-ON* */


/*
 * ============================================================================
 *
 *                                 Properties
 *
 * ============================================================================
 */


enum property {
	ARG_SYNCHRONOUS = 1
};


static void set_property(GObject * object, enum property id, const GValue * value, GParamSpec * pspec)
{
	GSTLALMultiplier *multiplier = GSTLAL_MULTIPLIER(object);

	GST_OBJECT_LOCK(multiplier);

	switch (id) {
	case ARG_SYNCHRONOUS:
		multiplier->synchronous = g_value_get_boolean(value);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(multiplier);
}


static void get_property(GObject * object, enum property id, GValue * value, GParamSpec * pspec)
{
	GSTLALMultiplier *multiplier = GSTLAL_MULTIPLIER(object);

	GST_OBJECT_LOCK(multiplier);

	switch (id) {
	case ARG_SYNCHRONOUS:
		/* FIXME:  on asynchronous --> synchronous transition, mark
		 * all collect pad's offset_offsets as invalid to force a
		 * resync */
		g_value_set_boolean(value, multiplier->synchronous);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(multiplier);
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


static GstCaps *gstlal_multiplier_sink_getcaps(GstPad * pad)
{
	GSTLALMultiplier *multiplier = GSTLAL_MULTIPLIER(GST_PAD_PARENT(pad));
	GstCaps *peercaps;
	GstCaps *caps;

	/*
	 * get our own allowed caps.  use the fixed caps function to avoid
	 * recursing back into this function.
	 */

	GST_OBJECT_LOCK(multiplier);
	caps = gst_pad_get_fixed_caps_func(pad);

	/*
	 * get the allowed caps from the downstream peer.
	 * if the peer has caps, intersect.
	 */

	peercaps = gst_pad_peer_get_caps(multiplier->srcpad);
	if(peercaps) {
		GstCaps *result;
		GST_DEBUG_OBJECT(multiplier, "intersecting %" GST_PTR_FORMAT " and %" GST_PTR_FORMAT, caps, peercaps);
		result = gst_caps_intersect(peercaps, caps);
		gst_caps_unref(peercaps);
		gst_caps_unref(caps);
		caps = result;
		GST_DEBUG_OBJECT(multiplier, "intersection %" GST_PTR_FORMAT, caps);
	}
	GST_OBJECT_UNLOCK(multiplier);

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


static gboolean gstlal_multiplier_setcaps(GstPad * pad, GstCaps * caps)
{
	GSTLALMultiplier *multiplier = GSTLAL_MULTIPLIER(GST_PAD_PARENT(pad));
	GList *padlist = NULL;
	GstStructure *structure = NULL;
	const char *media_type;
	gint width;
	gint channels;

	GST_LOG_OBJECT(multiplier, "setting caps on pad %s:%s to %" GST_PTR_FORMAT, GST_DEBUG_PAD_NAME(pad), caps);

	/*
	 * loop over all of the element's pads (source and sink), and set
	 * them all to the same format.
	 */

	/* FIXME, see if the other pads can accept the format. Also lock
	 * the format on the other pads to this new format. */

	GST_OBJECT_LOCK(multiplier);
	for(padlist = GST_ELEMENT(multiplier)->pads; padlist; padlist = g_list_next(padlist)) {
		GstPad *otherpad = GST_PAD(padlist->data);
		if(otherpad != pad)
			/* don't use gst_pad_set_caps() because that would
			 * recurse into this function */
			gst_caps_replace(&GST_PAD_CAPS(otherpad), caps);
	}
	GST_OBJECT_UNLOCK(multiplier);

	/*
	 * parse caps
	 */

	structure = gst_caps_get_structure(caps, 0);
	media_type = gst_structure_get_name(structure);
	gst_structure_get_int(structure, "rate", &multiplier->rate);
	gst_structure_get_int(structure, "channels", &channels);
	gst_structure_get_int(structure, "width", &width);
	if(!strcmp(media_type, "audio/x-raw-int")) {
		gboolean is_signed;
		GST_DEBUG_OBJECT(multiplier, "gstlal_multiplier_setcaps() sets multiplier to format int");
		gst_structure_get_boolean(structure, "signed", &is_signed);
		switch (width) {
		case 8:
			multiplier->func = is_signed ? multiply_int8 : multiply_uint8;
			break;
		case 16:
			multiplier->func = is_signed ? multiply_int16 : multiply_uint16;
			break;
		case 32:
			multiplier->func = is_signed ? multiply_int32 : multiply_uint32;
			break;
		default:
			goto not_supported;
		}
	} else if(!strcmp(media_type, "audio/x-raw-float")) {
		GST_DEBUG_OBJECT(multiplier, "gstlal_multiplier_setcaps() sets multiplier to format float");
		switch (width) {
		case 32:
			multiplier->func = multiply_float32;
			break;
		case 64:
			multiplier->func = multiply_float64;
			break;
		default:
			goto not_supported;
		}
	} else if(!strcmp(media_type, "audio/x-raw-complex")) {
		GST_DEBUG_OBJECT(multiplier, "gstlal_multiplier_setcaps() sets multiplier to format complex");
		switch (width) {
		case 64:
			multiplier->func = multiply_complex64;
			break;
		case 128:
			multiplier->func = multiply_complex128;
			break;
		default:
			goto not_supported;
		}
	} else
		goto not_supported;

	/*
	 * pre-calculate bytes / sample
	 */

	multiplier->unit_size = (width / 8) * channels;

	for(padlist = GST_ELEMENT(multiplier)->pads; padlist; padlist = g_list_next(padlist)) {
		GstPad *pad = GST_PAD(padlist->data);
		if(gst_pad_get_direction(pad) == GST_PAD_SINK) {
			gstlal_collect_pads_set_unit_size(pad, multiplier->unit_size);
			gstlal_collect_pads_set_rate(pad, multiplier->rate);
		}
	}

	/*
	 * done
	 */

	return TRUE;

	/*
	 * ERRORS
	 */

not_supported:
	GST_DEBUG_OBJECT(multiplier, "unsupported format");
	return FALSE;
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


static GstClockTime output_timestamp_from_offset(const GSTLALMultiplier *multiplier, guint64 offset)
{
	return multiplier->segment.start + gst_util_uint64_scale_int_round(offset, GST_SECOND, multiplier->rate);
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
 *
 * FIXME:  when we add a new stream (or remove a stream) the duration might
 * become invalid and we need to post a new DURATION message to notify this
 * fact to the parent.
 */


static gboolean gstlal_multiplier_query_duration(GSTLALMultiplier * multiplier, GstQuery * query)
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

	it = gst_element_iterate_sink_pads(GST_ELEMENT_CAST(multiplier));
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

		GST_DEBUG_OBJECT(multiplier, "Total duration in format %s: %" GST_TIME_FORMAT, gst_format_get_name(format), GST_TIME_ARGS(max));
		gst_query_set_duration(query, format, max);
	}

	return success;
}


static gboolean gstlal_multiplier_query_latency(GSTLALMultiplier * multiplier, GstQuery * query)
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

	it = gst_element_iterate_sink_pads(GST_ELEMENT_CAST(multiplier));
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
		GST_DEBUG_OBJECT(multiplier, "Calculated total latency: live %s, min %" GST_TIME_FORMAT ", max %" GST_TIME_FORMAT, (live ? "yes" : "no"), GST_TIME_ARGS(min), GST_TIME_ARGS(max));
		gst_query_set_latency(query, live, min, max);
	}

	return success;
}


static gboolean gstlal_multiplier_query(GstPad * pad, GstQuery * query)
{
	GSTLALMultiplier *multiplier = GSTLAL_MULTIPLIER(gst_pad_get_parent(pad));
	gboolean success = TRUE;

	switch(GST_QUERY_TYPE(query)) {
	case GST_QUERY_POSITION: {
		GstFormat format;

		gst_query_parse_position(query, &format, NULL);

		switch(format) {
		case GST_FORMAT_TIME:
			/* FIXME, bring to stream time, might be tricky */
			gst_query_set_position(query, format, output_timestamp_from_offset(multiplier, multiplier->offset));
			break;

		case GST_FORMAT_DEFAULT:
			/* default format for audio is sample count */
			gst_query_set_position(query, format, multiplier->offset);
			break;

		default:
			success = FALSE;
			break;
		}
		break;
	}

	case GST_QUERY_DURATION:
		success = gstlal_multiplier_query_duration(multiplier, query);
		break;

	case GST_QUERY_LATENCY:
		success = gstlal_multiplier_query_latency(multiplier, query);
		break;

	default:
		/* FIXME, needs a custom query handler because we have multiple
		 * sinkpads */
		success = gst_pad_query_default(pad, query);
		break;
	}

	gst_object_unref(multiplier);
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


static gboolean forward_event_func(GstPad * pad, GValue * ret, GstEvent * event)
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


static gboolean forward_event(GSTLALMultiplier * multiplier, GstEvent * event)
{
	GstIterator *it = NULL;
	GValue vret = {0};

	GST_LOG_OBJECT(multiplier, "forwarding event %p (%s)", event, GST_EVENT_TYPE_NAME(event));

	g_value_init(&vret, G_TYPE_BOOLEAN);
	g_value_set_boolean(&vret, TRUE);
	it = gst_element_iterate_sink_pads(GST_ELEMENT_CAST(multiplier));
	gst_iterator_fold(it, (GstIteratorFoldFunction) forward_event_func, &vret, event);
	gst_iterator_free(it);
	gst_event_unref(event);

	return g_value_get_boolean(&vret);
}


/*
 * src pad event handler
 */


static gboolean gstlal_multiplier_src_event(GstPad * pad, GstEvent * event)
{
	GSTLALMultiplier *multiplier = GSTLAL_MULTIPLIER(gst_pad_get_parent(pad));
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

		gst_event_parse_seek(event, &multiplier->segment.rate, NULL, &flags, &curtype, &cur, NULL, NULL);
		flush = !!(flags & GST_SEEK_FLAG_FLUSH);

		/*
		 * is it a flushing seek?
		 */

		if(flush) {
			/*
			 * make sure we accept nothing more and return
			 * WRONG_STATE
			 */

			gst_collect_pads_set_flushing(multiplier->collect, TRUE);

			/*
			 * start flush downstream.  the flush will be done
			 * when all pads received a FLUSH_STOP.
			 */

			gst_pad_push_event(multiplier->srcpad, gst_event_new_flush_start());
		}

		/*
		 * wait for the collected to be finished and mark a new
		 * segment
		 */

		GST_OBJECT_LOCK(multiplier->collect);
		multiplier->segment_pending = TRUE;
		if(flush) {
			/* Yes, we need to call _set_flushing again *WHEN* the streaming threads
			 * have stopped so that the cookie gets properly updated. */
			gst_collect_pads_set_flushing(multiplier->collect, TRUE);
		}
		multiplier->flush_stop_pending = flush;
		GST_OBJECT_UNLOCK(multiplier->collect);

		result = forward_event(multiplier, event);
		break;
	}

	default:
		/*
		 * forward the rest.
		 */

		result = forward_event(multiplier, event);
		break;
	}

	/*
	 * done
	 */

	gst_object_unref(multiplier);
	return result;
}


/*
 * sink pad event handler.  this is hacked in as an override of the collect
 * pads object's own event handler so that we can detect new segments and
 * flush stop events arriving on sink pads.  the real event handling is
 * accomplished by chaining to the original event handler installed by the
 * collect pads object.
 */


static gboolean gstlal_multiplier_sink_event(GstPad * pad, GstEvent * event)
{
	GSTLALMultiplier *multiplier = GSTLAL_MULTIPLIER(gst_pad_get_parent(pad));

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

		GST_OBJECT_LOCK(multiplier->collect);
		multiplier->segment_pending = TRUE;
		multiplier->flush_stop_pending = FALSE;
		GST_OBJECT_UNLOCK(multiplier->collect);
		break;

	default:
		break;
	}

	/*
	 * now chain to GstCollectPads handler to take care of the rest.
	 */

	gst_object_unref(multiplier);
	return multiplier->collect_event(pad, event);
}


/*
 * ============================================================================
 *
 *                            Add/Remove Sink Pad
 *
 * ============================================================================
 */


static GstPad *gstlal_multiplier_request_new_pad(GstElement * element, GstPadTemplate * templ, const gchar * unused)
{
	GSTLALMultiplier *multiplier = GSTLAL_MULTIPLIER(element);
	gchar *name = NULL;
	GstPad *newpad = NULL;
	GstLALCollectData *data = NULL;
	gint padcount;

	/*
	 * new pads can only be sink pads
	 */

	if(templ->direction != GST_PAD_SINK) {
		g_warning("gstmultiplier: request new pad that is not a SINK pad\n");
		goto not_sink;
	}


	/*
	 * create a new pad
	 */

	padcount = g_atomic_int_exchange_and_add(&multiplier->padcount, 1);
	name = g_strdup_printf(GST_PAD_TEMPLATE_NAME_TEMPLATE(templ), padcount);
	newpad = gst_pad_new_from_template(templ, name);
	GST_DEBUG_OBJECT(multiplier, "request new pad %p (%s)", newpad, name);
	g_free(name);

	/*
	 * configure new pad
	 */

	gst_pad_set_getcaps_function(newpad, GST_DEBUG_FUNCPTR(gstlal_multiplier_sink_getcaps));
	gst_pad_set_setcaps_function(newpad, GST_DEBUG_FUNCPTR(gstlal_multiplier_setcaps));

	/*
	 * add pad to collect pads object
	 */

	data = gstlal_collect_pads_add_pad(multiplier->collect, newpad, sizeof(*data));
	if(!data) {
		GST_DEBUG_OBJECT(multiplier, "could not add pad to collectpads object");
		goto could_not_add_to_collectpads;
	}

	/*
	 * FIXME: hacked way to override/extend the event function of
	 * GstCollectPads;  because it sets its own event function giving
	 * the element (us) no access to events
	 */

	multiplier->collect_event = (GstPadEventFunction) GST_PAD_EVENTFUNC(newpad);
	gst_pad_set_event_function(newpad, GST_DEBUG_FUNCPTR(gstlal_multiplier_sink_event));

	/*
	 * add pad to element (takes ownership of the pad).
	 */

	if(!gst_element_add_pad(GST_ELEMENT(multiplier), newpad)) {
		GST_DEBUG_OBJECT(multiplier, "could not add pad to element");
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
	gstlal_collect_pads_remove_pad(multiplier->collect, newpad);
could_not_add_to_collectpads:
	gst_object_unref(newpad);
not_sink:
	return NULL;
}


static void gstlal_multiplier_release_pad(GstElement * element, GstPad * pad)
{
	GSTLALMultiplier *multiplier = GSTLAL_MULTIPLIER(element);

	GST_DEBUG_OBJECT(multiplier, "release pad %s:%s", GST_DEBUG_PAD_NAME(pad));

	gstlal_collect_pads_remove_pad(multiplier->collect, pad);
	gst_element_remove_pad(element, pad);
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


static GstFlowReturn gstlal_multiplier_collected(GstCollectPads * pads, gpointer user_data)
{
	GSTLALMultiplier *multiplier = GSTLAL_MULTIPLIER(user_data);
	GSList *collected = NULL;
	GstClockTime t_start, t_end;
	guint64 earliest_input_offset, earliest_input_offset_end;
	guint64 length;
	GstBuffer *outbuf = NULL;
	gpointer outbytes = NULL;
	GstFlowReturn result;

	/*
	 * this is fatal
	 */

	if(G_UNLIKELY(!multiplier->func)) {
		GST_ELEMENT_ERROR(multiplier, STREAM, FORMAT, (NULL), ("Unknown data received, not negotiated"));
		result = GST_FLOW_NOT_NEGOTIATED;
		goto error;
	}

	/*
	 * forward flush-stop event
	 */

	if(multiplier->flush_stop_pending) {
		gst_pad_push_event(multiplier->srcpad, gst_event_new_flush_stop());
		multiplier->flush_stop_pending = FALSE;
	}

	/*
	 * check for new segment
	 */

	if(multiplier->segment_pending) {
		GstSegment *segment = gstlal_collect_pads_get_segment(multiplier->collect);
		if(!segment) {
			/* FIXME:  failure getting bounding segment, do
			 * something about it */
		}
		multiplier->segment = *segment;
		multiplier->offset = 0;
		gst_segment_free(segment);
	}

	/*
	 * get the range of offsets (in the output stream) spanned by the
	 * available input buffers.
	 */

	if(multiplier->synchronous) {
		/*
		 * when doing synchronous multiplying, determine the offsets for
		 * real.
		 */
		
		if(!gstlal_collect_pads_get_earliest_times(multiplier->collect, &t_start, &t_end)) {
			GST_ERROR_OBJECT(multiplier, "cannot deduce input timestamp offset information");
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

		earliest_input_offset = gst_util_uint64_scale_int_round(t_start - multiplier->segment.start, multiplier->rate, GST_SECOND);
		earliest_input_offset_end = gst_util_uint64_scale_int_round(t_end - multiplier->segment.start, multiplier->rate, GST_SECOND);

		if(earliest_input_offset < multiplier->offset) {
			GST_ERROR_OBJECT(multiplier, "detected time reversal in at least one input stream:  expected nothing earlier than offset %" G_GUINT64_FORMAT ", found sample at offset %" G_GUINT64_FORMAT, multiplier->offset, earliest_input_offset);
			result = GST_FLOW_ERROR;
			goto error;
		}
	} else {
		/*
		 * when not doing synchronous multiplying use the element's
		 * output offset counter and the number of bytes available.
		 * gst_collect_pads_available() returns 0 on EOS, which
		 * we'll figure out later when no pads produce buffers.
		 */

		earliest_input_offset = multiplier->offset;
		earliest_input_offset_end = earliest_input_offset + gst_collect_pads_available(pads) / multiplier->unit_size;
	}

	/*
	 * compute the number of samples for which all sink pads can
	 * contribute information.  0 does not necessarily mean EOS.
	 */

	length = earliest_input_offset_end - earliest_input_offset;

	/*
	 * loop over input pads, getting chunks of data from each in turn.
	 */

	GST_LOG_OBJECT(multiplier, "cycling through channels, offsets [%" G_GUINT64_FORMAT ", %" G_GUINT64_FORMAT ") (@ %d Hz) relative to %" GST_TIME_SECONDS_FORMAT " available", earliest_input_offset, earliest_input_offset_end, multiplier->rate, GST_TIME_SECONDS_ARGS(multiplier->segment.start));
	for(collected = pads->data; collected; collected = g_slist_next(collected)) {
		GstLALCollectData *data = collected->data;
		GstBuffer *inbuf;
		size_t gap;
		size_t len;

		/*
		 * (try to) get a buffer upto the desired end offset.
		 */

		if(multiplier->synchronous)
			inbuf = gstlal_collect_pads_take_buffer_sync(pads, data, t_end);
		else
			inbuf = gst_collect_pads_take_buffer(pads, (GstCollectData *) data, length * multiplier->unit_size);

		/*
		 * NULL means EOS.
		 */

		if(!inbuf) {
			GST_LOG_OBJECT(multiplier, "channel %p: no bytes available (EOS)", data);
			continue;
		}

		/*
		 * determine the buffer's location relative to the desired
		 * range of offsets.  we've checked above that time hasn't
		 * gone backwards on any input buffer so gap can't be
		 * negative.  if not doing synchronous mixing, the buffer
		 * starts now.
		 */

		gap = multiplier->synchronous ? (gst_util_uint64_scale_int_round(GST_BUFFER_TIMESTAMP(inbuf) - multiplier->segment.start, multiplier->rate, GST_SECOND) - earliest_input_offset) * multiplier->unit_size : 0;
		len = GST_BUFFER_SIZE(inbuf);

		/*
		 * mix with output
		 */

		if(!outbuf) {
			/*
			 * alloc a new output buffer of length samples, and
			 * set its offset.
			 */
			/* FIXME:  if this returns a short buffer we're
			 * sort of screwed.  a code re-organization could
			 * fix it:  request buffer before entering the loop
			 * and figure out a different way to check for EOS
			 */

			GST_LOG_OBJECT(multiplier, "requesting output buffer of %" G_GUINT64_FORMAT " samples", length);
			result = gst_pad_alloc_buffer(multiplier->srcpad, earliest_input_offset, length * multiplier->unit_size, GST_PAD_CAPS(multiplier->srcpad), &outbuf);
			if(result != GST_FLOW_OK) {
				/* FIXME: handle failure */
				outbuf = NULL;
			}
			outbytes = GST_BUFFER_DATA(outbuf);

			/*
			 * if the input buffer isn't a gap and has non-zero
			 * length copy it into the output buffer and mark
			 * as non-empty, otherwise memset the new output
			 * buffer to 0 and flag it as a gap
			 */

			if(!GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP) && len) {
				GST_LOG_OBJECT(multiplier, "channel %p: copying %zd bytes from data %p", data, len, GST_BUFFER_DATA(inbuf));
				memset(outbytes, 0, gap);
				memcpy(outbytes + gap, GST_BUFFER_DATA(inbuf), len);
				memset(outbytes + gap + len, 0, GST_BUFFER_SIZE(outbuf) - len - gap);
				GST_BUFFER_FLAG_UNSET(outbuf, GST_BUFFER_FLAG_GAP);
			} else {
				GST_LOG_OBJECT(multiplier, "channel %p: zeroing %d bytes from data %p", data, GST_BUFFER_SIZE(outbuf), GST_BUFFER_DATA(inbuf));
				memset(outbytes, 0, GST_BUFFER_SIZE(outbuf));
				GST_BUFFER_FLAG_SET(outbuf, GST_BUFFER_FLAG_GAP);
			}
		} else {
			/*
			 * if buffer is not a gap and has non-zero length
			 * multiply by previous data and mark output as
			 * non-empty, otherwise do nothing.
			 */

			if(!GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP) && len) {
				GST_LOG_OBJECT(multiplier, "channel %p: mixing %zd bytes from data %p", data, len, GST_BUFFER_DATA(inbuf));
				multiplier->func(outbytes + gap, GST_BUFFER_DATA(inbuf), len);
				GST_BUFFER_FLAG_UNSET(outbuf, GST_BUFFER_FLAG_GAP);
			} else
				GST_LOG_OBJECT(multiplier, "channel %p: skipping %zd bytes from data %p", data, len, GST_BUFFER_DATA(inbuf));
		}

		gst_buffer_unref(inbuf);
	}

	/*
	 * can only happen when no pads to collect or all EOS
	 */

	if(!outbuf)
		goto eos;

	/*
	 * check for discontinuity.
	 */

	if(multiplier->offset != GST_BUFFER_OFFSET(outbuf))
		GST_BUFFER_FLAG_SET(outbuf, GST_BUFFER_FLAG_DISCONT);

	/*
	 * set the timestamp, end offset, and duration.  computing the
	 * duration the way we do here ensures that if some downstream
	 * element adds all the buffer durations together they'll stay in
	 * sync with the timestamp.  the end offset is saved for comparison
	 * against the next start offset to watch for discontinuities.
	 */

	multiplier->offset = GST_BUFFER_OFFSET_END(outbuf) = GST_BUFFER_OFFSET(outbuf) + length;
	GST_BUFFER_TIMESTAMP(outbuf) = output_timestamp_from_offset(multiplier, GST_BUFFER_OFFSET(outbuf));
	GST_BUFFER_DURATION(outbuf) = output_timestamp_from_offset(multiplier, GST_BUFFER_OFFSET_END(outbuf)) - GST_BUFFER_TIMESTAMP(outbuf);

	/*
	 * precede the buffer with a new_segment event if one is pending
	 */
	/* FIXME, use rate/applied_rate as set on all sinkpads.  currently
	 * we just set rate as received from last seek-event We could
	 * potentially figure out the duration as well using the current
	 * segment positions and the stated stop positions. */

	if(multiplier->segment_pending) {
		/* FIXME:  the segment start time is almost certainly
		 * incorrect */
		GstEvent *event = gst_event_new_new_segment_full(FALSE, multiplier->segment.rate, 1.0, GST_FORMAT_TIME, multiplier->segment.start, multiplier->segment.stop, GST_BUFFER_TIMESTAMP(outbuf));

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

		gst_pad_push_event(multiplier->srcpad, event);
		multiplier->segment_pending = FALSE;
	}

	/*
	 * push the buffer downstream.
	 */

	GST_LOG_OBJECT(multiplier, "pushing outbuf, timestamp %" GST_TIME_SECONDS_FORMAT ", offset %" G_GUINT64_FORMAT, GST_TIME_SECONDS_ARGS(GST_BUFFER_TIMESTAMP(outbuf)), GST_BUFFER_OFFSET(outbuf));
	return gst_pad_push(multiplier->srcpad, outbuf);

	/*
	 * ERRORS
	 */

eos:
	GST_DEBUG_OBJECT(multiplier, "no data available (EOS)");
	gst_pad_push_event(multiplier->srcpad, gst_event_new_eos());
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


#define CAPS \
	"audio/x-raw-int, " \
	"rate = (int) [ 1, MAX ], " \
	"channels = (int) [ 1, MAX ], " \
	"endianness = (int) BYTE_ORDER, " \
	"width = (int) 32, " \
	"depth = (int) 32, " \
	"signed = (boolean) { true, false } ;" \
	"audio/x-raw-int, " \
	"rate = (int) [ 1, MAX ], " \
	"channels = (int) [ 1, MAX ], " \
	"endianness = (int) BYTE_ORDER, " \
	"width = (int) 16, " \
	"depth = (int) 16, " \
	"signed = (boolean) { true, false } ;" \
	"audio/x-raw-int, " \
	"rate = (int) [ 1, MAX ], " \
	"channels = (int) [ 1, MAX ], " \
	"endianness = (int) BYTE_ORDER, " \
	"width = (int) 8, " \
	"depth = (int) 8, " \
	"signed = (boolean) { true, false } ;" \
	"audio/x-raw-float, " \
	"rate = (int) [ 1, MAX ], " \
	"channels = (int) [ 1, MAX ], " \
	"endianness = (int) BYTE_ORDER, " \
	"width = (int) { 32, 64 } ;" \
	"audio/x-raw-complex, " \
	"rate = (int) [ 1, MAX ], " \
	"channels = (int) [ 1, MAX ], " \
	"endianness = (int) BYTE_ORDER, " \
	"width = (int) { 64, 128 }"


static GstStaticPadTemplate gstlal_multiplier_src_template =
	GST_STATIC_PAD_TEMPLATE(
		"src",
		GST_PAD_SRC,
		GST_PAD_ALWAYS,
		GST_STATIC_CAPS(CAPS)
	);


static GstStaticPadTemplate gstlal_multiplier_sink_template =
	GST_STATIC_PAD_TEMPLATE(
		"sink%d",
		GST_PAD_SINK,
		GST_PAD_REQUEST,
		GST_STATIC_CAPS(CAPS)
	);


/*
 * reset element's internal state and start the collect pads on READY -->
 * PAUSED state change.  stop the collect pads on PAUSED --> READY state
 * change.
 */


static GstStateChangeReturn gstlal_multiplier_change_state(GstElement * element, GstStateChange transition)
{
	GSTLALMultiplier *multiplier = GSTLAL_MULTIPLIER(element);

	switch(transition) {
	case GST_STATE_CHANGE_NULL_TO_READY:
		break;

	case GST_STATE_CHANGE_READY_TO_PAUSED:
		multiplier->segment_pending = TRUE;
		gst_segment_init(&multiplier->segment, GST_FORMAT_UNDEFINED);
		multiplier->offset = 0;
		gst_collect_pads_start(multiplier->collect);
		break;

	case GST_STATE_CHANGE_PAUSED_TO_PLAYING:
		break;

	case GST_STATE_CHANGE_PAUSED_TO_READY:
		/* need to unblock the collectpads before calling the
		 * parent change_state so that streaming can finish */
		gst_collect_pads_stop(multiplier->collect);
		break;

	default:
		break;
	}

	return GST_ELEMENT_CLASS(parent_class)->change_state(element, transition);
}


/*
 * free internal memory
 */


static void gstlal_multiplier_finalize(GObject * object)
{
	GSTLALMultiplier *multiplier = GSTLAL_MULTIPLIER(object);

	gst_object_unref(multiplier->collect);
	multiplier->collect = NULL;

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


/*
 * class init
 */


static void gstlal_multiplier_class_init(gpointer klass, gpointer class_data)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *gstelement_class = GST_ELEMENT_CLASS(klass);
	GSTLALMultiplierClass *gstlal_multiplier_class = GSTLAL_MULTIPLIER_CLASS(klass);

	gobject_class->set_property = set_property;
	gobject_class->get_property = get_property;
	gobject_class->finalize = GST_DEBUG_FUNCPTR(gstlal_multiplier_finalize);

	g_object_class_install_property(gobject_class, ARG_SYNCHRONOUS, g_param_spec_boolean("sync", "Synchronous", "Align the time stamps of input streams", FALSE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

	gst_element_class_add_pad_template(gstelement_class, gst_static_pad_template_get(&gstlal_multiplier_src_template));
	gst_element_class_add_pad_template(gstelement_class, gst_static_pad_template_get(&gstlal_multiplier_sink_template));
	gst_element_class_set_details_simple(gstelement_class, "Multiplier", "Generic/Audio", "Multiply N audio channels together", "Thomas <thomas@apestaart.org>");

	parent_class = g_type_class_peek_parent(gstlal_multiplier_class);

	gstelement_class->request_new_pad = GST_DEBUG_FUNCPTR(gstlal_multiplier_request_new_pad);
	gstelement_class->release_pad = GST_DEBUG_FUNCPTR(gstlal_multiplier_release_pad);
	gstelement_class->change_state = GST_DEBUG_FUNCPTR(gstlal_multiplier_change_state);
}


/*
 * instance init
 */


static void gstlal_multiplier_init(GTypeInstance * object, gpointer class)
{
	GSTLALMultiplier *multiplier = GSTLAL_MULTIPLIER(object);
	GstPadTemplate *template = NULL;

	template = gst_static_pad_template_get(&gstlal_multiplier_src_template);
	multiplier->srcpad = gst_pad_new_from_template(template, "src");
	gst_object_unref(template);

	gst_pad_set_getcaps_function(multiplier->srcpad, GST_DEBUG_FUNCPTR(gst_pad_proxy_getcaps));
	gst_pad_set_setcaps_function(multiplier->srcpad, GST_DEBUG_FUNCPTR(gstlal_multiplier_setcaps));
	gst_pad_set_query_function(multiplier->srcpad, GST_DEBUG_FUNCPTR(gstlal_multiplier_query));
	gst_pad_set_event_function(multiplier->srcpad, GST_DEBUG_FUNCPTR(gstlal_multiplier_src_event));
	gst_element_add_pad(GST_ELEMENT(object), multiplier->srcpad);

	multiplier->padcount = 0;

	multiplier->collect = gst_collect_pads_new();
	gst_collect_pads_set_function(multiplier->collect, GST_DEBUG_FUNCPTR(gstlal_multiplier_collected), multiplier);

	multiplier->rate = 0;
	multiplier->unit_size = 0;
	multiplier->func = NULL;
}


/*
 * create/get type ID
 */


GType gstlal_multiplier_get_type(void)
{
	static GType multiplier_type = 0;

	if(G_UNLIKELY(multiplier_type == 0)) {
		static const GTypeInfo multiplier_info = {
			.class_size = sizeof(GSTLALMultiplierClass),
			.class_init = gstlal_multiplier_class_init,
			.instance_size = sizeof(GSTLALMultiplier),
			.instance_init = gstlal_multiplier_init,
		};

		multiplier_type = g_type_register_static(GST_TYPE_ELEMENT, "GSTLALMultiplier", &multiplier_info, 0);
		GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "multiplier", 0, "audio channel mixing element");
	}

	return multiplier_type;
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
	return gst_element_register(plugin, "multiplier", GST_RANK_NONE, GSTLAL_TYPE_MULTIPLIER);
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, "multiplier", "Adds multiple streams", plugin_init, VERSION, "LGPL", GST_PACKAGE_NAME, GST_PACKAGE_ORIGIN)
#endif

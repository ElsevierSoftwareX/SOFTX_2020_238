/* GStreamer
 * Copyright (C) 1999,2000 Erik Walthinsen <omega@cse.ogi.edu>
 *                    2001 Thomas <thomas@apestaart.org>
 *               2005,2006 Wim Taymans <wim@fluendo.com>
 *               2008 Kipp Cannon <kcannon@ligo.caltech.edu>
 *
 * adder.c: Adder element, N in, one out, samples are added
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
 * SECTION:element-adder
 *
 * The adder allows to mix several streams into one by adding the data.
 * Mixed data is clamped to the min/max values of the data format.
 *
 * If the element's sync property is TRUE the streams are mixed with the
 * timestamps synchronized.  If the sync property is FALSE (the default, to
 * be compatible with older versions), then the first samples from each
 * stream are added to produce the first sample of the output, the second
 * samples are added to produce the second sample of the output, and so on.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch audiotestsrc freq=100 ! adder name=mix ! audioconvert ! alsasink audiotestsrc freq=500 ! mix.
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
#include "gstadder.h"
#include <gstlalcollectpads.h>


#define GST_CAT_DEFAULT gst_adder_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


/*
 * ============================================================================
 *
 *                                Adder Loops
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
		*_out = CLAMP((ttype) *_out + (ttype) *_in, min, max);       \
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
		*_out = (ttype) *_out + (ttype) *_in;           \
}


/* *INDENT-OFF* */
MAKE_FUNC(add_int32, gint32, gint64, MIN_INT_32, MAX_INT_32)
MAKE_FUNC(add_int16, gint16, gint32, MIN_INT_16, MAX_INT_16)
MAKE_FUNC(add_int8, gint8, gint16, MIN_INT_8, MAX_INT_8)
MAKE_FUNC(add_uint32, guint32, guint64, MIN_UINT_32, MAX_UINT_32)
MAKE_FUNC(add_uint16, guint16, guint32, MIN_UINT_16, MAX_UINT_16)
MAKE_FUNC(add_uint8, guint8, guint16, MIN_UINT_8, MAX_UINT_8)
MAKE_FUNC_NC(add_complex128, double complex, double complex)
MAKE_FUNC_NC(add_complex64, float complex, float complex)
MAKE_FUNC_NC(add_float64, gdouble, gdouble)
MAKE_FUNC_NC(add_float32, gfloat, gfloat)
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


static void set_property(GObject * object, enum property id, const GValue * value, GParamSpec * psspec)
{
	GstAdder *adder = GST_ADDER(object);

	GST_OBJECT_LOCK(adder->collect);

	switch (id) {
	case ARG_SYNCHRONOUS:
		adder->synchronous = g_value_get_boolean(value);
		break;
	}

	GST_OBJECT_UNLOCK(adder->collect);
}


static void get_property(GObject * object, enum property id, GValue * value, GParamSpec * psspec)
{
	GstAdder *adder = GST_ADDER(object);

	GST_OBJECT_LOCK(adder->collect);

	switch (id) {
	case ARG_SYNCHRONOUS:
		/* FIXME:  on asynchronous --> synchronous transition, mark
		 * all collect pad's offset_offsets as invalid to force a
		 * resync */
		g_value_set_boolean(value, adder->synchronous);
		break;
	}

	GST_OBJECT_UNLOCK(adder->collect);
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


static GstCaps *gst_adder_sink_getcaps(GstPad * pad)
{
	GstAdder *adder = GST_ADDER(gst_pad_get_parent(pad));
	GstCaps *peercaps;
	GstCaps *caps;

	GST_OBJECT_LOCK(adder->collect);

	/*
	 * get our own allowed caps.  use the fixed caps function to avoid
	 * recursing back into this function.
	 */

	caps = gst_pad_get_fixed_caps_func(pad);

	/*
	 * get the allowed caps from the downstream peer.
	 * if the peer has caps, intersect.
	 */

	peercaps = gst_pad_peer_get_caps(adder->srcpad);
	if(peercaps) {
		GstCaps *result;
		GST_DEBUG_OBJECT(adder, "intersecting " GST_PTR_FORMAT " and " GST_PTR_FORMAT, caps, peercaps);
		result = gst_caps_intersect(peercaps, caps);
		gst_caps_unref(peercaps);
		gst_caps_unref(caps);
		caps = result;
		GST_DEBUG_OBJECT(adder, "intersection " GST_PTR_FORMAT, caps);
	}

	/*
	 * done
	 */

	GST_OBJECT_UNLOCK(adder->collect);
	gst_object_unref(adder);
	return caps;
}


/*
 * the first caps we receive on any of the sinkpads will define the caps
 * for all the other sinkpads because we can only mix streams with the same
 * caps.
 */


static gboolean gst_adder_setcaps(GstPad * pad, GstCaps * caps)
{
	GstAdder *adder = GST_ADDER(gst_pad_get_parent(pad));
	GList *padlist = NULL;
	GstStructure *structure = NULL;
	const char *media_type;
	gint width;
	gint channels;
	gboolean is_signed;

	GST_LOG_OBJECT(adder, "setting caps on pad %s:%s to %" GST_PTR_FORMAT, GST_DEBUG_PAD_NAME(pad), caps);

	GST_OBJECT_LOCK(adder->collect);

	/*
	 * loop over all of the element's pads (source and sink), and set
	 * them all to the same format.
	 */

	/* FIXME, see if the other pads can accept the format. Also lock
	 * the format on the other pads to this new format. */

	for(padlist = GST_ELEMENT(adder)->pads; padlist; padlist = g_list_next(padlist)) {
		GstPad *otherpad = GST_PAD(padlist->data);
		if(otherpad != pad)
			/* don't use gst_pad_set_caps() because that would
			 * recurse into this function */
			gst_caps_replace(&GST_PAD_CAPS(otherpad), caps);
	}

	/*
	 * parse caps
	 */

	structure = gst_caps_get_structure(caps, 0);
	media_type = gst_structure_get_name(structure);
	gst_structure_get_int(structure, "rate", &adder->rate);
	gst_structure_get_int(structure, "channels", &channels);
	if(!strcmp(media_type, "audio/x-raw-int")) {
		GST_DEBUG_OBJECT(adder, "gst_adder_setcaps() sets adder to format int");
		gst_structure_get_int(structure, "width", &width);
		gst_structure_get_boolean(structure, "signed", &is_signed);

		switch (width) {
		case 8:
			adder->func = is_signed ? add_int8 : add_uint8;
			break;
		case 16:
			adder->func = is_signed ? add_int16 : add_uint16;
			break;
		case 32:
			adder->func = is_signed ? add_int32 : add_uint32;
			break;
		default:
			goto not_supported;
		}
	} else if(!strcmp(media_type, "audio/x-raw-float")) {
		GST_DEBUG_OBJECT(adder, "gst_adder_setcaps() sets adder to format float");
		gst_structure_get_int(structure, "width", &width);

		switch (width) {
		case 32:
			adder->func = add_float32;
			break;
		case 64:
			adder->func = add_float64;
			break;
		default:
			goto not_supported;
		}
	} else if(!strcmp(media_type, "audio/x-raw-complex")) {
		GST_DEBUG_OBJECT(adder, "gst_adder_setcaps() sets adder to format complex");
		gst_structure_get_int(structure, "width", &width);

		switch (width) {
		case 64:
			adder->func = add_complex64;
			break;
		case 128:
			adder->func = add_complex128;
			break;
		default:
			goto not_supported;
		}
	} else
		goto not_supported;

	/*
	 * pre-calculate bytes / sample
	 */

	adder->unit_size = (width / 8) * channels;

	for(padlist = GST_ELEMENT(adder)->pads; padlist; padlist = g_list_next(padlist)) {
		GstPad *pad = GST_PAD(padlist->data);
		if(gst_pad_get_direction(pad) == GST_PAD_SINK)
			gstlal_collect_pads_set_unit_size(pad, adder->unit_size);
	}

	/*
	 * done
	 */

	GST_OBJECT_UNLOCK(adder->collect);
	gst_object_unref(adder);
	return TRUE;

	/*
	 * ERRORS
	 */

not_supported:
	GST_DEBUG_OBJECT(adder, "unsupported format");
	GST_OBJECT_UNLOCK(adder->collect);
	gst_object_unref(adder);
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


static GstClockTime output_timestamp_from_offset(const GstAdder *adder, guint64 offset)
{
	return adder->segment.start + gst_util_uint64_scale_int(offset, GST_SECOND, adder->rate);
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


static gboolean gst_adder_query_duration(GstAdder * adder, GstQuery * query)
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

	it = gst_element_iterate_sink_pads(GST_ELEMENT_CAST(adder));
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

		GST_DEBUG_OBJECT(adder, "Total duration in format %s: %" GST_TIME_FORMAT, gst_format_get_name(format), GST_TIME_ARGS(max));
		gst_query_set_duration(query, format, max);
	}

	return success;
}


static gboolean gst_adder_query_latency(GstAdder * adder, GstQuery * query)
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

	it = gst_element_iterate_sink_pads(GST_ELEMENT_CAST(adder));
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
		GST_DEBUG_OBJECT(adder, "Calculated total latency: live %s, min %" GST_TIME_FORMAT ", max %" GST_TIME_FORMAT, (live ? "yes" : "no"), GST_TIME_ARGS(min), GST_TIME_ARGS(max));
		gst_query_set_latency(query, live, min, max);
	}

	return success;
}


static gboolean gst_adder_query(GstPad * pad, GstQuery * query)
{
	GstAdder *adder = GST_ADDER(gst_pad_get_parent(pad));
	gboolean success = TRUE;

	switch(GST_QUERY_TYPE(query)) {
	case GST_QUERY_POSITION: {
		GstFormat format;

		gst_query_parse_position(query, &format, NULL);

		switch(format) {
		case GST_FORMAT_TIME:
			/* FIXME, bring to stream time, might be tricky */
			gst_query_set_position(query, format, output_timestamp_from_offset(adder, adder->offset));
			break;

		case GST_FORMAT_DEFAULT:
			/* default format for audio is sample count */
			gst_query_set_position(query, format, adder->offset);
			break;

		default:
			success = FALSE;
			break;
		}
		break;
	}

	case GST_QUERY_DURATION:
		success = gst_adder_query_duration(adder, query);
		break;

	case GST_QUERY_LATENCY:
		success = gst_adder_query_latency(adder, query);
		break;

	default:
		/* FIXME, needs a custom query handler because we have multiple
		 * sinkpads */
		success = gst_pad_query_default(pad, query);
		break;
	}

	gst_object_unref(adder);
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


static gboolean forward_event(GstAdder * adder, GstEvent * event)
{
	GstIterator *it = NULL;
	GValue vret = {0};

	GST_LOG_OBJECT(adder, "forwarding event %p (%s)", event, GST_EVENT_TYPE_NAME(event));

	g_value_init(&vret, G_TYPE_BOOLEAN);
	g_value_set_boolean(&vret, TRUE);
	it = gst_element_iterate_sink_pads(GST_ELEMENT_CAST(adder));
	gst_iterator_fold(it, (GstIteratorFoldFunction) forward_event_func, &vret, event);
	gst_iterator_free(it);
	gst_event_unref(event);

	return g_value_get_boolean(&vret);
}


/*
 * src pad event handler
 */


static gboolean gst_adder_src_event(GstPad * pad, GstEvent * event)
{
	GstAdder *adder = GST_ADDER(gst_pad_get_parent(pad));
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

		gst_event_parse_seek(event, &adder->segment.rate, NULL, &flags, &curtype, &cur, NULL, NULL);
		flush = !!(flags & GST_SEEK_FLAG_FLUSH);

		/*
		 * is it a flushing seek?
		 */

		if(flush) {
			/*
			 * make sure we accept nothing more and return
			 * WRONG_STATE
			 */

			gst_collect_pads_set_flushing(adder->collect, TRUE);

			/*
			 * start flush downstream.  the flush will be done
			 * when all pads received a FLUSH_STOP.
			 */

			gst_pad_push_event(adder->srcpad, gst_event_new_flush_start());
		}

		/*
		 * wait for the collected to be finished and mark a new
		 * segment
		 */

		GST_OBJECT_LOCK(adder->collect);
		adder->segment_pending = TRUE;
		if(flush) {
			/* Yes, we need to call _set_flushing again *WHEN* the streaming threads
			 * have stopped so that the cookie gets properly updated. */
			gst_collect_pads_set_flushing(adder->collect, TRUE);
		}
		adder->flush_stop_pending = flush;
		GST_OBJECT_UNLOCK(adder->collect);

		result = forward_event(adder, event);
		break;
	}

	default:
		/*
		 * forward the rest.
		 */

		result = forward_event(adder, event);
		break;
	}

	/*
	 * done
	 */

	gst_object_unref(adder);
	return result;
}


/*
 * sink pad event handler.  this is hacked in as an override of the collect
 * pads object's own event handler so that we can detect new segments and
 * flush stop events arriving on sink pads.  the real event handling is
 * accomplished by chaining to the original event handler installed by the
 * collect pads object.
 */


static gboolean gst_adder_sink_event(GstPad * pad, GstEvent * event)
{
	GstAdder *adder = GST_ADDER(gst_pad_get_parent(pad));

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

		GST_OBJECT_LOCK(adder->collect);
		adder->segment_pending = TRUE;
		adder->flush_stop_pending = FALSE;
		GST_OBJECT_UNLOCK(adder->collect);
		break;

	default:
		break;
	}

	/*
	 * now chain to GstCollectPads handler to take care of the rest.
	 */

	gst_object_unref(adder);
	return adder->collect_event(pad, event);
}


/*
 * ============================================================================
 *
 *                            Add/Remove Sink Pad
 *
 * ============================================================================
 */


static GstPad *gst_adder_request_new_pad(GstElement * element, GstPadTemplate * templ, const gchar * unused)
{
	GstAdder *adder = GST_ADDER(element);
	gchar *name = NULL;
	GstPad *newpad = NULL;
	GstLALCollectData *data = NULL;
	gint padcount;

	/*
	 * new pads can only be sink pads
	 */

	if(templ->direction != GST_PAD_SINK) {
		g_warning("gstadder: request new pad that is not a SINK pad\n");
		goto not_sink;
	}


	/*
	 * create a new pad
	 */

	padcount = g_atomic_int_exchange_and_add(&adder->padcount, 1);
	name = g_strdup_printf(GST_PAD_TEMPLATE_NAME_TEMPLATE(templ), padcount);
	newpad = gst_pad_new_from_template(templ, name);
	GST_DEBUG_OBJECT(adder, "request new pad %p (%s)", newpad, name);
	g_free(name);

	/*
	 * configure new pad
	 */

	gst_pad_set_getcaps_function(newpad, GST_DEBUG_FUNCPTR(gst_adder_sink_getcaps));
	gst_pad_set_setcaps_function(newpad, GST_DEBUG_FUNCPTR(gst_adder_setcaps));

	/*
	 * add pad to collect pads object
	 */

	data = gstlal_collect_pads_add_pad(adder->collect, newpad, sizeof(*data));
	if(!data) {
		GST_DEBUG_OBJECT(adder, "could not add pad to collectpads object");
		goto could_not_add_to_collectpads;
	}

	/*
	 * FIXME: hacked way to override/extend the event function of
	 * GstCollectPads;  because it sets its own event function giving
	 * the element (us) no access to events
	 */

	adder->collect_event = (GstPadEventFunction) GST_PAD_EVENTFUNC(newpad);
	gst_pad_set_event_function(newpad, GST_DEBUG_FUNCPTR(gst_adder_sink_event));

	/*
	 * add pad to element (takes ownership of the pad).
	 */

	if(!gst_element_add_pad(GST_ELEMENT(adder), newpad)) {
		GST_DEBUG_OBJECT(adder, "could not add pad to element");
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
	gstlal_collect_pads_remove_pad(adder->collect, newpad);
could_not_add_to_collectpads:
	gst_object_unref(newpad);
not_sink:
	return NULL;
}


static void gst_adder_release_pad(GstElement * element, GstPad * pad)
{
	GstAdder *adder = GST_ADDER(element);

	GST_DEBUG_OBJECT(adder, "release pad %s:%s", GST_DEBUG_PAD_NAME(pad));

	gstlal_collect_pads_remove_pad(adder->collect, pad);
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


static GstFlowReturn gst_adder_collected(GstCollectPads * pads, gpointer user_data)
{
	GstAdder *adder = GST_ADDER(user_data);
	GSList *collected = NULL;
	guint64 earliest_input_offset, earliest_input_offset_end;
	guint64 length;
	GstBuffer *outbuf = NULL;
	gpointer outbytes = NULL;
	GstFlowReturn result;

	/*
	 * this is fatal
	 */

	if(G_UNLIKELY(!adder->func)) {
		GST_ELEMENT_ERROR(adder, STREAM, FORMAT, (NULL), ("Unknown data received, not negotiated"));
		result = GST_FLOW_NOT_NEGOTIATED;
		goto error;
	}

	if(adder->flush_stop_pending) {
		gst_pad_push_event(adder->srcpad, gst_event_new_flush_stop());
		adder->flush_stop_pending = FALSE;
	}

	/*
	 * check for new segment
	 */

	if(adder->segment_pending) {
		GstSegment *segment = gstlal_collect_pads_get_segment(adder->collect);
		if(!segment) {
			/* FIXME:  failure getting bounding segment, do
			 * something about it */
		}
		adder->segment = *segment;
		gst_segment_free(segment);
		adder->offset = GST_BUFFER_OFFSET_NONE;
	}

	/*
	 * get the range of offsets (in the output stream) spanned by the
	 * available input buffers.
	 */

	if(adder->synchronous) {
		/*
		 * when doing synchronous adding, determine the offsets for
		 * real.
		 */
		
		if(!gstlal_collect_pads_get_earliest_offsets(adder->collect, &earliest_input_offset, &earliest_input_offset_end, adder->rate, adder->segment.start)) {
			GST_ERROR_OBJECT(adder, "cannot deduce input timestamp offset information");
			result = GST_FLOW_ERROR;
			goto error;
		}

		/*
		 * check for EOS
		 */

		if(earliest_input_offset == GST_BUFFER_OFFSET_NONE)
			goto eos;

		/*
		 * don't let time go backwards.  in principle we could be
		 * smart and handle this, but the audiorate element can be
		 * used to correct screwed up time series so there is no
		 * point in re-inventing its capabilities here.
		 */

		if((adder->offset != GST_BUFFER_OFFSET_NONE) && (earliest_input_offset < adder->offset)) {
			GST_ERROR_OBJECT(adder, "detected time reversal in at least one input stream:  expected nothing earlier than offset %lu, found sample at offset %lu", adder->offset, earliest_input_offset);
			result = GST_FLOW_ERROR;
			goto error;
		}
	} else {
		/*
		 * when not doing synchronous adding use the element's
		 * output offset counter and the number of bytes available.
		 * gst_collect_pads_available() returns 0 on EOS, which
		 * we'll figure out later when no pads produce buffers.
		 */

		earliest_input_offset = adder->offset == GST_BUFFER_OFFSET_NONE ? 0 : adder->offset;
		earliest_input_offset_end = earliest_input_offset + gst_collect_pads_available(pads) / adder->unit_size;
	}

	/*
	 * compute the number of samples for which all sink pads can
	 * contribute information.  0 does not necessarily mean EOS.
	 */

	length = earliest_input_offset_end - earliest_input_offset;

	/*
	 * loop over input pads, getting chunks of data from each in turn.
	 */

	GST_LOG_OBJECT(adder, "cycling through channels, offsets %lu--%lu available", earliest_input_offset, earliest_input_offset_end);
	for(collected = pads->data; collected; collected = g_slist_next(collected)) {
		GstLALCollectData *data = collected->data;
		GstBuffer *inbuf;
		size_t gap;
		size_t len;

		/*
		 * (try to) get a buffer upto the desired end offset.
		 */

		if(adder->synchronous)
			inbuf = gstlal_collect_pads_take_buffer(pads, data, earliest_input_offset_end, adder->rate, adder->segment.start);
		else
			inbuf = gst_collect_pads_take_buffer(pads, (GstCollectData *) data, length * adder->unit_size);

		/*
		 * NULL means EOS.
		 */

		if(!inbuf) {
			GST_LOG_OBJECT(adder, "channel %p: no bytes available (EOS)", data);
			continue;
		}

		/*
		 * determine the buffer's location relative to the desired
		 * range of offsets.  we've checked above that time hasn't
		 * gone backwards on any input buffer so gap can't be
		 * negative.  if not doing synchronous mixing, the buffer
		 * starts now.
		 */

		gap = adder->synchronous ? (GST_BUFFER_OFFSET(inbuf) - earliest_input_offset) * adder->unit_size : 0;
		len = GST_BUFFER_SIZE(inbuf);

		/*
		 * add to output
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
			 * */

			GST_LOG_OBJECT(adder, "requesting output buffer of %lu samples", length);
			result = gst_pad_alloc_buffer(adder->srcpad, earliest_input_offset, length * adder->unit_size, GST_PAD_CAPS(adder->srcpad), &outbuf);
			if(result != GST_FLOW_OK) {
				/* FIXME: handle failure */
				outbuf = NULL;
			}
			outbytes = GST_BUFFER_DATA(outbuf);

			/*
			 * buffer is empty until something goes in it
			 */

			GST_BUFFER_FLAG_SET(outbuf, GST_BUFFER_FLAG_GAP);

			/*
			 * if the input buffer isn't a gap and has non-zero
			 * length copy it into the output buffer and mark
			 * as non-empty, otherwise memset the new output
			 * buffer to 0
			 */

			if(!GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP) && len) {
				GST_LOG_OBJECT(adder, "channel %p: copying %d bytes from data %p", data, len, GST_BUFFER_DATA(inbuf));
				memset(outbytes, 0, gap);
				memcpy(outbytes + gap, GST_BUFFER_DATA(inbuf), len);
				memset(outbytes + gap + len, 0, GST_BUFFER_SIZE(outbuf) - len - gap);
				GST_BUFFER_FLAG_UNSET(outbuf, GST_BUFFER_FLAG_GAP);
			} else {
				GST_LOG_OBJECT(adder, "channel %p: zeroing %d bytes from data %p", data, GST_BUFFER_SIZE(outbuf), GST_BUFFER_DATA(inbuf));
				memset(outbytes, 0, GST_BUFFER_SIZE(outbuf));
			}
		} else {
			/*
			 * if buffer is not a gap and has non-zero length
			 * add to previous data and mark output as
			 * non-empty, otherwise do nothing.
			 */

			if(!GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP) && len) {
				GST_LOG_OBJECT(adder, "channel %p: mixing %d bytes from data %p", data, len, GST_BUFFER_DATA(inbuf));
				adder->func(outbytes + gap, GST_BUFFER_DATA(inbuf), len);
				GST_BUFFER_FLAG_UNSET(outbuf, GST_BUFFER_FLAG_GAP);
			} else
				GST_LOG_OBJECT(adder, "channel %p: skipping %d bytes from data %p", data, len, GST_BUFFER_DATA(inbuf));
		}

		gst_buffer_unref(inbuf);
	}

	/*
	 * can only happen when no pads to collect or all EOS
	 */

	if(!outbuf)
		goto eos;

	/*
	 * precede the buffer with a new_segment event if one is pending
	 */
	/* FIXME, use rate/applied_rate as set on all sinkpads.  currently
	 * we just set rate as received from last seek-event We could
	 * potentially figure out the duration as well using the current
	 * segment positions and the stated stop positions. */

	if(adder->segment_pending) {
		/* FIXME:  the segment start time is almost certainly
		 * incorrect */
		GstEvent *event = gst_event_new_new_segment_full(FALSE, adder->segment.rate, 1.0, GST_FORMAT_TIME, adder->segment.start, adder->segment.stop, output_timestamp_from_offset(adder, GST_BUFFER_OFFSET(outbuf)));

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

		gst_pad_push_event(adder->srcpad, event);
		adder->segment_pending = FALSE;
	}

	/*
	 * check for discontinuity.
	 */

	if((adder->offset == GST_BUFFER_OFFSET_NONE) || (adder->offset != GST_BUFFER_OFFSET(outbuf)))
		GST_BUFFER_FLAG_SET(outbuf, GST_BUFFER_FLAG_DISCONT);

	/*
	 * set the timestamp, end offset, and duration.  computing the
	 * duration the way we do here ensures that if some downstream
	 * element adds all the buffer durations together they'll stay in
	 * sync with the timestamp.  the end offset is saved for comparison
	 * against the next start offset to watch for discontinuities.
	 */

	adder->offset = GST_BUFFER_OFFSET_END(outbuf) = GST_BUFFER_OFFSET(outbuf) + length;
	GST_BUFFER_TIMESTAMP(outbuf) = output_timestamp_from_offset(adder, GST_BUFFER_OFFSET(outbuf));
	GST_BUFFER_DURATION(outbuf) = output_timestamp_from_offset(adder, GST_BUFFER_OFFSET_END(outbuf)) - GST_BUFFER_TIMESTAMP(outbuf);

	/*
	 * push the buffer downstream.
	 */

	GST_LOG_OBJECT(adder, "pushing outbuf, timestamp %" GST_TIME_FORMAT ", offset %lu", GST_TIME_ARGS(GST_BUFFER_TIMESTAMP(outbuf)), GST_BUFFER_OFFSET(outbuf));
	return gst_pad_push(adder->srcpad, outbuf);

	/*
	 * ERRORS
	 */

eos:
	GST_DEBUG_OBJECT(adder, "no data available (EOS)");
	gst_pad_push_event(adder->srcpad, gst_event_new_eos());
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


static GstStaticPadTemplate gst_adder_src_template =
	GST_STATIC_PAD_TEMPLATE(
		"src",
		GST_PAD_SRC,
		GST_PAD_ALWAYS,
		GST_STATIC_CAPS(CAPS)
	);


static GstStaticPadTemplate gst_adder_sink_template =
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


static GstStateChangeReturn gst_adder_change_state(GstElement * element, GstStateChange transition)
{
	GstAdder *adder = GST_ADDER(element);

	switch(transition) {
	case GST_STATE_CHANGE_NULL_TO_READY:
		break;

	case GST_STATE_CHANGE_READY_TO_PAUSED:
		adder->segment_pending = TRUE;
		gst_segment_init(&adder->segment, GST_FORMAT_UNDEFINED);
		adder->offset = GST_BUFFER_OFFSET_NONE;
		gst_collect_pads_start(adder->collect);
		break;

	case GST_STATE_CHANGE_PAUSED_TO_PLAYING:
		break;

	case GST_STATE_CHANGE_PAUSED_TO_READY:
		/* need to unblock the collectpads before calling the
		 * parent change_state so that streaming can finish */
		gst_collect_pads_stop(adder->collect);
		break;

	default:
		break;
	}

	return GST_ELEMENT_CLASS(parent_class)->change_state(element, transition);
}


/*
 * free internal memory
 */


static void gst_adder_finalize(GObject * object)
{
	GstAdder *adder = GST_ADDER(object);

	gst_object_unref(adder->collect);
	adder->collect = NULL;

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


/*
 * class init
 */


static void gst_adder_class_init(gpointer klass, gpointer class_data)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *gstelement_class = GST_ELEMENT_CLASS(klass);
	GstAdderClass *gst_adder_class = GST_ADDER_CLASS(klass);

	gobject_class->set_property = set_property;
	gobject_class->get_property = get_property;
	gobject_class->finalize = GST_DEBUG_FUNCPTR(gst_adder_finalize);

	g_object_class_install_property(gobject_class, ARG_SYNCHRONOUS, g_param_spec_boolean("sync", "Synchronous", "Align the time stamps of input streams", FALSE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

	gst_element_class_add_pad_template(gstelement_class, gst_static_pad_template_get(&gst_adder_src_template));
	gst_element_class_add_pad_template(gstelement_class, gst_static_pad_template_get(&gst_adder_sink_template));
	gst_element_class_set_details_simple(gstelement_class, "Adder", "Generic/Audio", "Add N audio channels together", "Thomas <thomas@apestaart.org>");

	parent_class = g_type_class_peek_parent(gst_adder_class);

	gstelement_class->request_new_pad = GST_DEBUG_FUNCPTR(gst_adder_request_new_pad);
	gstelement_class->release_pad = GST_DEBUG_FUNCPTR(gst_adder_release_pad);
	gstelement_class->change_state = GST_DEBUG_FUNCPTR(gst_adder_change_state);
}


/*
 * instance init
 */


static void gst_adder_init(GTypeInstance * object, gpointer class)
{
	GstAdder *adder = GST_ADDER(object);
	GstPadTemplate *template = NULL;

	template = gst_static_pad_template_get(&gst_adder_src_template);
	adder->srcpad = gst_pad_new_from_template(template, "src");
	gst_object_unref(template);

	gst_pad_set_getcaps_function(adder->srcpad, GST_DEBUG_FUNCPTR(gst_pad_proxy_getcaps));
	gst_pad_set_setcaps_function(adder->srcpad, GST_DEBUG_FUNCPTR(gst_adder_setcaps));
	gst_pad_set_query_function(adder->srcpad, GST_DEBUG_FUNCPTR(gst_adder_query));
	gst_pad_set_event_function(adder->srcpad, GST_DEBUG_FUNCPTR(gst_adder_src_event));
	gst_element_add_pad(GST_ELEMENT(object), adder->srcpad);

	adder->padcount = 0;

	adder->collect = gst_collect_pads_new();
	gst_collect_pads_set_function(adder->collect, GST_DEBUG_FUNCPTR(gst_adder_collected), adder);

	adder->rate = 0;
	adder->unit_size = 0;
	adder->func = NULL;
}


/*
 * create/get type ID
 */


GType gst_adder_get_type(void)
{
	static GType adder_type = 0;

	if(G_UNLIKELY(adder_type == 0)) {
		static const GTypeInfo adder_info = {
			.class_size = sizeof(GstAdderClass),
			.class_init = gst_adder_class_init,
			.instance_size = sizeof(GstAdder),
			.instance_init = gst_adder_init,
		};

		adder_type = g_type_register_static(GST_TYPE_ELEMENT, "LALGstAdder", &adder_info, 0);
		GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "adder", 0, "audio channel mixing element");
	}

	return adder_type;
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
	return gst_element_register(plugin, "adder", GST_RANK_NONE, GST_TYPE_ADDER);
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, "adder", "Adds multiple streams", plugin_init, VERSION, "LGPL", GST_PACKAGE_NAME, GST_PACKAGE_ORIGIN)
#endif

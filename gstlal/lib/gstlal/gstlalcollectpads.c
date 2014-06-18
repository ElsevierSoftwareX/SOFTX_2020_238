/*
 * Custom GstCollectPads class to assist with combining input streams
 * synchronously
 *
 * Copyright (C) 2008 Kipp Cannon <kipp.cannon@ligo.org>
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


/**
 * SECTION:gstlalcollectpads
 * @short_description:  Custom #GstCollectPads to assist with combining
 * input streams synchronously.
 *
 * Custom GstCollectData structure with extra metadata to facilitate
 * synchronous mixing of input streams.  In fact, only the #GstCollectData
 * structure is customized, replaced here with the #GstLALCollectData
 * structure;  however, a few shim functions are required to adapt function
 * signatures to accept and return pointers to the custom type.
 */


/*
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


#include <math.h>


#include <gst/gst.h>
#include <gst/base/gstcollectpads.h>
#include <gstlalcollectpads.h>
#include <gstlal_debug.h>


/*
 * ============================================================================
 *
 *                            Add/Remove Sink Pad
 *
 * ============================================================================
 */


/**
 * gstlal_collect_pads_add_pad_full:
 * @pads:  passed to #gst_collect_pads_add_pad()
 * @pad:  passed to #gst_collect_pads_add_pad()
 * @size:  passed to #gst_collect_pads_add_pad()
 * @destroy_notify:  passed to#gst_collect_pads_add_pad()
 *
 * Wraps #gst_collect_pads_add_pad_full(), initializing the additional
 * fields in the custom #GstLALCollectData object.
 *
 * Returns:  #GstLALCollectData associated with the #GstPad.
 */


GstLALCollectData *gstlal_collect_pads_add_pad_full(GstCollectPads *pads, GstPad *pad, guint size, GstCollectDataDestroyNotify destroy_notify)
{
	GstLALCollectData *data;

	/*
	 * add pad to collect pads object
	 */

	data = (GstLALCollectData *) gst_collect_pads_add_pad_full(pads, pad, size, destroy_notify);
	if(!data) {
		GST_ERROR_OBJECT(pads, "could not add pad to collectpads object");
		return NULL;
	}

	/*
	 * initialize our own extra contents of the GstLALCollectData
	 * structure
	 */

	data->unit_size = 0;

	/*
	 * done
	 */

	return data;
}


/**
 * gstlal_collect_pads_add_pad:
 * @pads:  passed to #gstlal_collect_pads_add_pad_full()
 * @pad:  passed to #gstlal_collect_pads_add_pad_full()
 * @size:  passed to #gstlal_collect_pads_add_pad_full()
 *
 * Equivalent to #gst_collect_pads_add_pad().
 *
 * Returns:  #GstLALCollectData associated with the #GstPad.
 */


GstLALCollectData *gstlal_collect_pads_add_pad(GstCollectPads *pads, GstPad *pad, guint size)
{
	return gstlal_collect_pads_add_pad_full(pads, pad, size, NULL);
}


/**
 * gstlal_collect_pads_remove_pad:
 * @pads:  passed to #gst_collect_pads_remove_pad()
 * @pad:  passed to #gst_collect_pads_remove_pad()
 *
 * Equivalent to #gst_collect_pads_remove_pad().
 *
 * Returns:  TRUE if #GstPad was removed successfully, FALSE if not.
 */


gboolean gstlal_collect_pads_remove_pad(GstCollectPads *pads, GstPad *pad)
{
	return gst_collect_pads_remove_pad(pads, pad);
}


/*
 * ============================================================================
 *
 *                               Data Retrieval
 *
 * ============================================================================
 */


/**
 * gstlal_collect_pads_set_unit_size:
 * @pad:  the #GstPad whose unit size is to be set
 * @unit_size:  the size in bytes of one unit
 *
 * Set the number of bytes per unit (e.g., sample, frame, etc.) on the
 * given input stream.
 *
 * Should be called with the #GstCollectPads' lock held (e.g., from the
 * collected() method).
 */


void gstlal_collect_pads_set_unit_size(GstPad *pad, guint unit_size)
{
	GstLALCollectData *data = gst_pad_get_element_private(pad);

	g_return_if_fail(data != NULL);

	data->unit_size = unit_size;
}


/**
 * gstlal_collect_pads_get_unit_size:
 * @pad:  the #GstPad whose unit size is to be retrieved
 *
 * Get the number of bytes per unit (e.g., sample, frame, etc.) on the
 * given input stream.
 *
 * Should be called with the #GstCollectPads' lock held (i.e., from the
 * collected() method).
 *
 * Returns:  unit size in bytes.
 */


guint gstlal_collect_pads_get_unit_size(GstPad *pad)
{
	GstLALCollectData *data = gst_pad_get_element_private(pad);

	g_return_val_if_fail(data != NULL, -1);

	return data->unit_size;
}


/**
 * gstlal_collect_pads_set_rate:
 * @pad:  the #GstPad whose unit rate (in Hertz) is to be set
 * @rate:  the number of units per second.
 *
 * Set the unit rate (e.g., sample rate, frame rate, etc.) on the given
 * input stream.
 *
 * Should be called with the #GstCollectPads' lock held (e.g., from the
 * collected() method).
 */


void gstlal_collect_pads_set_rate(GstPad *pad, gint rate)
{
	GstLALCollectData *data = gst_pad_get_element_private(pad);

	g_return_if_fail(data != NULL);

	data->rate = rate;
}


/**
 * gstlal_collect_pads_get_rate:
 * @pad:  the #GstPad whose unit rate is to be retrieved
 *
 * Get the unit rate (e.g., sample rate, frame rate, etc.) on the given
 * input stream.
 *
 * Should be called with the #GstCollectPads' lock held (i.e., from the
 * collected() method).
 *
 * Returns:  unit rate in Hertz.
 */


gint gstlal_collect_pads_get_rate(GstPad *pad)
{
	GstLALCollectData *data = gst_pad_get_element_private(pad);

	g_return_val_if_fail(data != NULL, -1);

	return data->rate;
}


/**
 * gstlal_collect_pads_get_segment:
 * @pads:  #GstCollectPads
 *
 * Compute the smallest segment that contains the segments (from the most
 * recent newsegment events) of all pads.  The segments must be in the same
 * format on all pads.  The return value is a newly allocated #GstSegment
 * owned by the calling code.
 *
 * Should be called with the #GstCollectPads' lock held (i.e., from the
 * collected() method).
 *
 * Returns:  newly-allocated #GstSegment.  #gst_segment_free() when no
 * longer needed.
 */


GstSegment *gstlal_collect_pads_get_segment(GstCollectPads *pads)
{
	GSList *collectdatalist = NULL;
	GstSegment *segment = NULL;

	g_return_val_if_fail(pads != NULL, NULL);
	g_return_val_if_fail(GST_IS_COLLECT_PADS(pads), NULL);

	for(collectdatalist = pads->data; collectdatalist; collectdatalist = g_slist_next(collectdatalist)) {
		/* really a pointer to a GstLALCollectData object, casting
		 * to GstCollectData */
		GstCollectData *data = collectdatalist->data;

		/*
		 * start by copying the segment from the first collect pad
		 */

		if(!segment) {
			segment = gst_segment_copy(&data->segment);
			if(!segment) {
				GST_ERROR_OBJECT(pads, "%" GST_PTR_FORMAT ": failure copying segment", data->pad);
				goto done;
			}
			continue;
		}

		/*
		 * check for format/rate mismatch
		 */

		if(segment->format != data->segment.format || segment->applied_rate != data->segment.applied_rate) {
			GST_ERROR_OBJECT(pads, "%" GST_PTR_FORMAT ": mismatch in segment format and/or applied rate", data->pad);
			gst_segment_free(segment);
			segment = NULL;
			goto done;
		}

		/*
		 * expand start and stop
		 */

		GST_DEBUG_OBJECT(pads, "%" GST_PTR_FORMAT ": have segment [%" G_GUINT64_FORMAT ", %" G_GUINT64_FORMAT ")", data->pad, data->segment.start, data->segment.stop);
		if(segment->start == -1 || segment->start > data->segment.start)
			segment->start = data->segment.start;
		if(segment->stop == -1 || segment->stop < data->segment.stop)
			segment->stop = data->segment.stop;
	}
	if(segment)
		GST_DEBUG_OBJECT(pads, "returning segment [%" G_GUINT64_FORMAT ", %" G_GUINT64_FORMAT ")", segment->start, segment->stop);
	else
		GST_DEBUG_OBJECT(pads, "no segment available");

done:
	return segment;
}


/**
 * gstlal_collect_pads_get_earliest_times:
 * @pads:  #GstCollectPads
 * @t_start:  address of #GstClockTime where start time will be stored
 * @t_end:  address of #GstClockTime where end time will be stored
 *
 * Computes the earliest of the start and of the end times of the
 * #GstCollectPad's input buffers.
 *
 * Upon the successful completion of this function, both time parameters
 * will be set to #GST_CLOCK_TIME_NONE if all input streams are at EOS.
 * Otherwise, if at least one stream is not at EOS, the times are set to
 * the earliest interval spanned by all the buffers that are available.
 *
 * Note that if no input pads have data available, this condition is
 * interpreted as EOS.  EOS is, therefore, indistinguishable from the
 * initial state, wherein no data has yet arrived.  It is assumed this
 * function will only be invoked from within the collected() method, and
 * therefore only after at least one pad has received a buffer, and
 * therefore the "no data available" condition is only seen at EOS.
 *
 * <table><title>Return Values</title>
 * <tgroup cols='3' align='center'>
 * <thead>
 *	<row>
 *		<entry>condition</entry>
 *		<entry>return value</entry>
 *		<entry>t_end, t_start</entry>
 *	</row>
 * </thead>
 * <tbody>
 *	<row>
 *		<entry>bad input</entry>
 *		<entry>FALSE</entry>
 *		<entry>undefined</entry>
 *	</row>
 *	<row>
 *		<entry>EOS</entry>
 *		<entry>TRUE</entry>
 *		<entry>#GST_CLOCK_TIME_NONE</entry>
 *	</row>
 *	<row>
 *		<entry>success</entry>
 *		<entry>TRUE</entry>
 *		<entry>&ge;0</entry>
 *	</row>
 * </tbody>
 * </tgroup>
 * </table>
 *
 * Should be called with the #GstCollectPads' lock held (i.e., from the
 * collected() method).
 *
 * Returns:  TRUE indicates the function was able to procede to a
 * successful conclusion, FALSE indicates that one or more errors occured
 * (see above).
 */


static GstClockTime compute_t_start(GstLALCollectData *data, GstBuffer *buf)
{
	/* FIXME:  could use GST_FRAMES_TO_CLOCK_TIME() but that macro is
	 * defined in gst-plugins-base */
	return  GST_BUFFER_TIMESTAMP(buf) + gst_util_uint64_scale_int_round(((GstCollectData *) data)->pos / data->unit_size, GST_SECOND, data->rate);
}


static GstClockTime compute_t_end(GstLALCollectData *data, GstBuffer *buf)
{
	return GST_BUFFER_TIMESTAMP(buf) + GST_BUFFER_DURATION(buf);
}


gboolean gstlal_collect_pads_get_earliest_times(GstCollectPads *pads, GstClockTime *t_start, GstClockTime *t_end)
{
	gboolean all_eos = TRUE;
	GSList *collectdatalist;

	/*
	 * initilize
	 */

	g_return_val_if_fail(t_start != NULL, FALSE);
	g_return_val_if_fail(t_end != NULL, FALSE);

	*t_start = *t_end = G_MAXUINT64;

	g_return_val_if_fail(pads != NULL, FALSE);
	g_return_val_if_fail(GST_IS_COLLECT_PADS(pads), FALSE);

	/*
	 * loop over sink pads
	 */

	for(collectdatalist = pads->data; collectdatalist; collectdatalist = g_slist_next(collectdatalist)) {
		GstLALCollectData *data = collectdatalist->data;
		GstBuffer *buf;
		GstClockTime buf_t_start, buf_t_end;

		/*
		 * check for uninitialized GstLALCollectData
		 */

		g_return_val_if_fail(data->unit_size != 0, FALSE);

		/*
		 * check for EOS
		 */

		buf = gst_collect_pads_peek(pads, (GstCollectData *) data);
		if(!buf) {
			GST_DEBUG_OBJECT(pads, "%" GST_PTR_FORMAT ": EOS", ((GstCollectData *) data)->pad);
			continue;
		}

		/*
		 * require a valid start offset and timestamp
		 */

		if(!GST_BUFFER_OFFSET_IS_VALID(buf) || !GST_BUFFER_OFFSET_END_IS_VALID(buf)) {
			GST_ERROR_OBJECT(pads, "%" GST_PTR_FORMAT ": %" GST_PTR_FORMAT " does not have valid offsets", ((GstCollectData *) data)->pad, buf);
			gst_buffer_unref(buf);
			return FALSE;
		}

		if(!GST_BUFFER_TIMESTAMP_IS_VALID(buf) || !GST_BUFFER_DURATION_IS_VALID(buf)) {
			GST_ERROR_OBJECT(pads, "%" GST_PTR_FORMAT ": %" GST_PTR_FORMAT " does not have a valid timestamp and/or duration", ((GstCollectData *) data)->pad, buf);
			gst_buffer_unref(buf);
			return FALSE;
		}

		/*
		 * compute this buffer's start and end times
		 */

		buf_t_start = compute_t_start(data, buf);
		buf_t_end = compute_t_end(data, buf);
		gst_buffer_unref(buf);

		GST_DEBUG_OBJECT(pads, "%" GST_PTR_FORMAT ": buffer spans [%" GST_TIME_SECONDS_FORMAT ", %" GST_TIME_SECONDS_FORMAT ")", ((GstCollectData *) data)->pad, GST_TIME_SECONDS_ARGS(buf_t_start), GST_TIME_SECONDS_ARGS(buf_t_end));

		if(buf_t_end < buf_t_start) {
			GST_ERROR_OBJECT(pads, "%" GST_PTR_FORMAT ": buffer has negative length", ((GstCollectData *) data)->pad);
			return FALSE;
		}

		/*
		 * update the minima
		 */

		if(buf_t_start < *t_start)
			*t_start = buf_t_start;
		if(buf_t_end < *t_end)
			*t_end = buf_t_end;

		/*
		 * with at least one valid pair of times, we can return
		 * meaningful numbers.
		 */

		all_eos = FALSE;
	}

	/*
	 * found at least one buffer?
	 */

	if(all_eos)
		*t_start = *t_end = GST_CLOCK_TIME_NONE;
	GST_DEBUG_OBJECT(pads, "earliest common spanned interval [%" GST_TIME_SECONDS_FORMAT ", %" GST_TIME_SECONDS_FORMAT ")", GST_TIME_SECONDS_ARGS(*t_start), GST_TIME_SECONDS_ARGS(*t_end));

	return TRUE;
}


/**
 * gstlal_collect_pads_take_buffer_sync:
 * @pads:  #GstCollectPads
 * @data:  #GstLALCollectData associated with the #GstPad from which to
 * take the data
 * @t_end:  the #GstClockTime up to which to retrieve data
 *
 * Wrapper for #gst_collect_pads_take_buffer().  Returns a #GstBuffer
 * containing the samples taken from the start of the current buffer upto
 * (not including) the offset corresponding to t_end.  The buffer returned
 * might be shorter if the pad does not have data upto the requested time.
 * The buffer returned by this function has its offset and offset_end set
 * to indicate its location in the input stream.  Calling this function has
 * the effect of flushing the pad upto the offset corresponding to t_end or
 * the upper bound of the available data, whichever comes first.
 *
 * If the pad has no data available then NULL is returned, this indicates
 * EOS.  If the pad has data available but it is subsequent to the
 * requested interval then a zero-length buffer is returned.
 *
 * Should be called with the #GstCollectPads' lock held (i.e., from the
 * collected() method).
 *
 * Returns:  #GstBuffer.  #gst_buffer_unref() when no longer needed.
 */


GstBuffer *gstlal_collect_pads_take_buffer_sync(GstCollectPads *pads, GstLALCollectData *data, GstClockTime t_end)
{
	GstBuffer *buf;
	guint64 offset;
	GstClockTime buf_t_start, buf_t_end;
	gboolean is_gap, is_malloced;
	guint64 units;

	/*
	 * checks
	 */

	g_return_val_if_fail(pads != NULL, NULL);
	g_return_val_if_fail(GST_IS_COLLECT_PADS(pads), NULL);
	g_return_val_if_fail(data != NULL, NULL);
	g_return_val_if_fail(data->unit_size != 0, NULL);
	g_return_val_if_fail(data->rate != 0, NULL);

	/*
	 * retrieve the start and end time of the next buffer to be
	 * dequeued.
	 */

	buf = gst_collect_pads_peek(pads, (GstCollectData *) data);
	if(!buf)
		/*
		 * EOS
		 */
		return NULL;
	offset = GST_BUFFER_OFFSET(buf) + ((GstCollectData *) data)->pos / data->unit_size;
	buf_t_start = compute_t_start(data, buf);
	buf_t_end = compute_t_end(data, buf);
	is_gap = GST_BUFFER_FLAG_IS_SET(buf, GST_BUFFER_FLAG_GAP);
	is_malloced = GST_BUFFER_DATA(buf) != NULL;
	gst_buffer_unref(buf);

	/*
	 * more checks
	 */

	g_return_val_if_fail(t_end <= buf_t_end, NULL);

	/*
	 * compute the number of units to request from the queued buffer.
	 * if the requested end time precedes the queued buffer then set
	 * the number of units to 0 to return an empty buffer.
	 */

	/* FIXME:  could use GST_CLOCK_TIME_TO_FRAMES() but that macro is
	 * defined in gst-plugins-base */
	units = t_end <= buf_t_start ? 0 : gst_util_uint64_scale_int_round(t_end - buf_t_start, data->rate, GST_SECOND);
	GST_DEBUG_OBJECT(GST_PAD_PARENT(((GstCollectData *) data)->pad), "(%s): requesting %" G_GUINT64_FORMAT " units\n", GST_PAD_NAME(((GstCollectData *) data)->pad), units);

	/*
	 * retrieve a buffer
	 */

	if(is_gap && !is_malloced) {
		/* FIXME:  the underlying GstCollectData class should
		 * handle this itself.  the need to do so is independent of
		 * the synchronization-related work going on here.  this
		 * probably starts by teaching gst_buffer_create_sub() to
		 * do the right things with non-malloc()ed buffers */
		GstBuffer *source = gst_collect_pads_peek(pads, (GstCollectData *) data);
		buf = gst_buffer_new();
		gst_buffer_copy_metadata(buf, source, GST_BUFFER_COPY_ALL);
		((GstCollectData *) data)->pos += units * data->unit_size;
		if(((GstCollectData *) data)->pos / data->unit_size >= GST_BUFFER_OFFSET_END(source) - GST_BUFFER_OFFSET(source)) {
			gst_buffer_unref(gst_collect_pads_pop(pads, (GstCollectData *) data));
			/*gst_collect_pads_clear (pads, (GstCollectData *) data);*/
		}
		gst_buffer_unref(source);
	} else {
		buf = gst_collect_pads_take_buffer(pads, (GstCollectData *) data, units * data->unit_size);
		/* this would normally indicate EOS, but it's impossible
		 * here because we would've seen this already up above */
		g_assert(buf != NULL);
		/* it should be impossible to not get what we asked for */
		g_assert_cmpuint(GST_BUFFER_SIZE(buf), ==, units * data->unit_size);
	}
	g_assert(GST_BUFFER_FLAG_IS_SET(buf, GST_BUFFER_FLAG_GAP) == is_gap);

	/*
	 * make sure its caps are set.  for some reason we can get buffers
	 * with NULL caps, but setcaps() must have been called on the sink
	 * pad so we should be able to copy the caps from there.  I think
	 * gst_buffer_create_sub() maybe doesn't copy the caps?
	 */
	/* FIXME:  I'm going to open a PR for this, and this code should be
	 * removed when we can rely on a gstreamer new-enough to contain
	 * the fix */

	if(!GST_BUFFER_CAPS(buf))
		gst_buffer_set_caps(buf, GST_PAD_CAPS(((GstCollectData *) data)->pad));
	g_assert(GST_BUFFER_CAPS(buf) != NULL);

	/*
	 * set the buffer's start and end offsets and time stamp and
	 * duration relative to the input stream
	 */

	buf = gst_buffer_make_metadata_writable(buf);
	GST_BUFFER_OFFSET(buf) = offset;
	GST_BUFFER_OFFSET_END(buf) = offset + units;
	GST_BUFFER_TIMESTAMP(buf) = buf_t_start;
	/* FIXME:  could use GST_FRAMES_TO_CLOCK_TIME() but that macro is
	 * defined in gst-plugins-base */
	GST_BUFFER_DURATION(buf) = gst_util_uint64_scale_int_round(units, GST_SECOND, data->rate);

	GST_DEBUG_OBJECT(GST_PAD_PARENT(((GstCollectData *) data)->pad), "(%s): returning %" GST_BUFFER_BOUNDARIES_FORMAT, GST_PAD_NAME(((GstCollectData *) data)->pad), GST_BUFFER_BOUNDARIES_ARGS(buf));

	return buf;
}

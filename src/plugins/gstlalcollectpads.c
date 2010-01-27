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
#include <gstlal.h>


/*
 * ============================================================================
 *
 *                            Add/Remove Sink Pad
 *
 * ============================================================================
 */


/**
 * Wraps gst_collect_pads_add_pad(), initializing the additional fields in
 * the custom GstLALCollectData object and installing a custom event
 * handler.
 */


GstLALCollectData *gstlal_collect_pads_add_pad_full(GstCollectPads *pads, GstPad *pad, guint size, GstCollectDataDestroyNotify destroy_notify)
{
	GstLALCollectData *data;

	/*
	 * add pad to collect pads object
	 */

	data = (GstLALCollectData *) gst_collect_pads_add_pad_full(pads, pad, size, destroy_notify);
	if(!data) {
		GST_DEBUG_OBJECT(pads, "could not add pad to collectpads object");
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


GstLALCollectData *gstlal_collect_pads_add_pad(GstCollectPads *pads, GstPad *pad, guint size)
{
	return gstlal_collect_pads_add_pad_full(pads, pad, size, NULL);
}


/**
 * Wraps gst_collect_pads_remove_pad().
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
 * Record the number of bytes per unit (e.g., sample, frame, etc.) on the
 * given input stream.
 *
 * Should be called with the GstCollectPads' lock held (e.g., from the
 * collected() method).
 */


void gstlal_collect_pads_set_unit_size(GstPad *pad, guint unit_size)
{
	GstLALCollectData *data = gst_pad_get_element_private(pad);

	g_return_if_fail(data != NULL);

	data->unit_size = unit_size;
}


/**
 * Retrieve the number of bytes per unit (e.g., sample, frame, etc.) on the
 * given input stream.
 *
 * Should be called with the GstCollectPads' lock held (i.e., from the
 * collected() method).
 */


guint gstlal_collect_pads_get_unit_size(GstPad *pad)
{
	GstLALCollectData *data = gst_pad_get_element_private(pad);

	g_return_val_if_fail(data != NULL, -1);

	return data->unit_size;
}


/**
 * Compute the smallest segment that contains the segments (from the most
 * recent newsegment events) of all pads.  The segments must be in the same
 * format on all pads.  The return value is a newly allocated GstSegment
 * owned by the calling code.
 *
 * Should be called with the GstCollectPads' lock held (i.e., from the
 * collected() method).
 */


GstSegment *gstlal_collect_pads_get_segment(GstCollectPads *pads)
{
	GSList *collectdatalist = NULL;
	GstSegment *segment = NULL;

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
				GST_ERROR_OBJECT(pads, "failure copying segment");
				goto done;
			}
			continue;
		}

		/*
		 * check for format/rate mismatch
		 */

		if(segment->format != data->segment.format || segment->applied_rate != data->segment.applied_rate) {
			GST_ERROR_OBJECT(pads, "mismatch in segment format and/or applied rate");
			gst_segment_free(segment);
			segment = NULL;
			goto done;
		}

		/*
		 * expand start and stop
		 */

		if(segment->start == -1 || segment->start > data->segment.start)
			segment->start = data->segment.start;
		if(segment->stop == -1 || segment->stop < data->segment.stop)
			segment->stop = data->segment.stop;
	}

done:
	return segment;
}


/**
 * Computes the earliest of the start and of the end times of the
 * GstCollectPad's input buffers.
 *
 * Both times are set to GST_CLOCK_TIME_NONE if one or more input buffers
 * has invalid timestamps and/or offsets or all pads report EOS.  The
 * return value is FALSE if at least one input buffer had an invalid
 * timestamp and/or offsets.  The calling code should interpret this to
 * indicate the presence of invalid input on at least one pad.  The return
 * value is TRUE if no input buffer had invalid metadata (including when
 * there are 0 input buffers, as when all pads report EOS).
 *
 * Summary:
 *
 * condition   return value   times
 * ----------------------------------
 * bad input   FALSE          GST_CLOCK_TIME_NONE
 * EOS         TRUE           GST_CLOCK_TIME_NONE
 * success     TRUE           >= 0
 *
 * Should be called with the GstCollectPads' lock held (i.e., from the
 * collected() method).
 */


static GstClockTime compute_t_start(GstLALCollectData *data, GstBuffer *buf, gint rate)
{
	/* FIXME:  could use GST_FRAMES_TO_CLOCK_TIME() but that macro is
	 * defined in gst-plugins-base */
	return  GST_BUFFER_TIMESTAMP(buf) + gst_util_uint64_scale_int_round(((GstCollectData *) data)->pos / data->unit_size, GST_SECOND, rate);
}


static GstClockTime compute_t_end(GstLALCollectData *data, GstBuffer *buf, gint rate)
{
	/* FIXME:  could use GST_FRAMES_TO_CLOCK_TIME() but that macro is
	 * defined in gst-plugins-base */
	return GST_BUFFER_TIMESTAMP(buf) + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET_END_IS_VALID(buf) ? GST_BUFFER_OFFSET_END(buf) - GST_BUFFER_OFFSET(buf) : GST_BUFFER_SIZE(buf) / data->unit_size, GST_SECOND, rate);
}


gboolean gstlal_collect_pads_get_earliest_times(GstCollectPads *pads, GstClockTime *t_start, GstClockTime *t_end, gint rate)
{
	gboolean valid = FALSE;
	GSList *collectdatalist = NULL;

	/*
	 * initilize
	 */

	*t_start = *t_end = G_MAXUINT64;

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
			GST_LOG("%p: EOS\n", data);
			continue;
		}

		/*
		 * require a valid start offset and timestamp
		 */

		if(!GST_BUFFER_OFFSET_IS_VALID(buf)) {
			GST_LOG("%p: input buffer does not have a valid offset\n", data);
			gst_buffer_unref(buf);
			return FALSE;
		}

		if(!GST_BUFFER_TIMESTAMP_IS_VALID(buf)) {
			GST_LOG("%p: input buffer does not have a valid timestamp\n", data);
			gst_buffer_unref(buf);
			return FALSE;
		}

		/*
		 * compute this buffer's start and end times
		 */

		buf_t_start = compute_t_start(data, buf, rate);
		buf_t_end = compute_t_end(data, buf, rate);
		gst_buffer_unref(buf);

		GST_DEBUG_OBJECT(GST_PAD_PARENT(((GstCollectData *) data)->pad), "(%s): time = [%" GST_TIME_SECONDS_FORMAT ", %" GST_TIME_SECONDS_FORMAT ")\n", GST_PAD_NAME(((GstCollectData *) data)->pad), GST_TIME_SECONDS_ARGS(buf_t_start), GST_TIME_SECONDS_ARGS(buf_t_end));

		if(buf_t_end < buf_t_start) {
			GST_LOG("%p: input buffer appears to have negative length\n", data);
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

		valid = TRUE;
	}

	/*
	 * found at least one buffer?
	 */

	if(!valid)
		*t_start = *t_end = GST_CLOCK_TIME_NONE;
	GST_DEBUG("%p: time = [%" GST_TIME_SECONDS_FORMAT ", %" GST_TIME_SECONDS_FORMAT ")\n", pads, GST_TIME_SECONDS_ARGS(*t_start), GST_TIME_SECONDS_ARGS(*t_end));

	return TRUE;
}


/**
 * Wrapper for gst_collect_pads_take_buffer().  Returns a buffer containing
 * the samples taken from the start of the current buffer upto (not
 * including) the offset corresponding to t_end.  The buffer returned might
 * be shorter if the pad does not have data upto the requested time.  The
 * buffer returned by this function has its offset and offset_end set to
 * indicate its location in the input stream.  Calling this function has
 * the effect of flushing the pad upto the offset corresponding to t_end or
 * the upper bound of the available data, whichever comes first.
 *
 * If the pad has no data available then NULL is returned, this indicates
 * EOS.  If the pad has data available but it is subsequent to the
 * requested interval then a zero-length buffer is returned.
 *
 * Should be called with the GstCollectPads' lock held (i.e., from the
 * collected() method).
 */


GstBuffer *gstlal_collect_pads_take_buffer(GstCollectPads *pads, GstLALCollectData *data, GstClockTime t_end, gint rate)
{
	GstBuffer *buf;
	guint64 offset;
	GstClockTime t_start;
	guint64 units;

	/*
	 * check for uninitialized GstLALCollectData
	 */

	g_return_val_if_fail(data->unit_size != 0, NULL);

	/*
	 * retrieve the start time of the next buffer to be dequeued.
	 */

	buf = gst_collect_pads_peek(pads, (GstCollectData *) data);
	if(!buf)
		/*
		 * EOS
		 */
		return NULL;
	offset = GST_BUFFER_OFFSET(buf) + ((GstCollectData *) data)->pos / data->unit_size;
	t_start = compute_t_start(data, buf, rate);
	gst_buffer_unref(buf);

	/*
	 * compute the number of units to request from the queued buffer.
	 * if the requested end time precedes the queued buffer then set
	 * the number of units to 0 to return an empty buffer.
	 */

	/* FIXME:  could use GST_CLOCK_TIME_TO_FRAMES() but that macro is
	 * defined in gst-plugins-base */
	units = t_end < t_start ? 0 : gst_util_uint64_scale_int_round(t_end - t_start, rate, GST_SECOND);
	GST_DEBUG_OBJECT(GST_PAD_PARENT(((GstCollectData *) data)->pad), "(%s): requesting %" G_GUINT64_FORMAT " units\n", GST_PAD_NAME(((GstCollectData *) data)->pad), units);

	/*
	 * retrieve a buffer
	 */

	buf = gst_collect_pads_take_buffer(pads, (GstCollectData *) data, units * data->unit_size);
	if(!buf)
		/*
		 * EOS or no data (probably impossible, because we would've
		 * detected this above, but might as well check again)
		 */
		return NULL;

	/*
	 * set the buffer's start and end offsets and time stamp and
	 * duration relative to the input stream
	 */

	buf = gst_buffer_make_metadata_writable(buf);
	GST_BUFFER_OFFSET(buf) = offset;
	GST_BUFFER_OFFSET_END(buf) = offset + GST_BUFFER_SIZE(buf) / data->unit_size;
	GST_BUFFER_TIMESTAMP(buf) = t_start;
	/* FIXME:  could use GST_FRAMES_TO_CLOCK_TIME() but that macro is
	 * defined in gst-plugins-base */
	GST_BUFFER_DURATION(buf) = gst_util_uint64_scale_int_round(units, GST_SECOND, rate);

	GST_DEBUG_OBJECT(GST_PAD_PARENT(((GstCollectData *) data)->pad), "(%s): returning [%" GST_TIME_SECONDS_FORMAT ", %" GST_TIME_SECONDS_FORMAT ")\n", GST_PAD_NAME(((GstCollectData *) data)->pad), GST_TIME_SECONDS_ARGS(GST_BUFFER_TIMESTAMP(buf)), GST_TIME_SECONDS_ARGS(GST_BUFFER_TIMESTAMP(buf) + GST_BUFFER_DURATION(buf)));

	return buf;
}

/*
 * Custom GstCollectPads class to assist with combining input streams
 * synchronously
 *
 * Copyright (C) 2008 Kipp Cannon <kcannon@ligo.caltech.edu>
 *
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Library General Public License as published
 * by the Free Software Foundation; either version 2 of the License, or (at
 * your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307,
 * USA.
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
#include "gstlalcollectpads.h"


/*
 * ============================================================================
 *
 *                                   Events
 *
 * ============================================================================
 */


/**
 * sink pad event handler.  this is hacked in as an override of the collect
 * pads object's own event handler so that we can detect new segments and
 * flush stop events arriving on sink pads.  the real event handling is
 * accomplished by chaining to the original event handler installed by the
 * collect pads object.
 */


static gboolean gstlal_collect_pads_sink_event(GstPad *pad, GstEvent *event)
{
	/* FIXME:  the collect pads stores the address of the
	 * GstCollectData object in the pad's element private.  this is
	 * undocumented behaviour, but we rely on it! */
	GstLALCollectData *data = gst_pad_get_element_private(pad);

	g_return_val_if_fail(data != NULL, FALSE);

	/*
	 * handle events
	 */

	switch (GST_EVENT_TYPE(event)) {
	case GST_EVENT_NEWSEGMENT:
		break;

	default:
		break;
	}

	/*
	 * now chain to GstCollectPads handler to take care of the rest.
	 */

	return data->collect_event_func(pad, event);
}


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
	 * FIXME: hacked way to override/extend the event function of
	 * GstCollectPads;  because it sets its own event function giving
	 * the element (us) no access to events
	 */

	data->collect_event_func = (GstPadEventFunction) GST_PAD_EVENTFUNC(pad);
	gst_pad_set_event_function(pad, GST_DEBUG_FUNCPTR(gstlal_collect_pads_sink_event));

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
 * Record the number of bytes per sample on the given input stream.
 */


void gstlal_collect_pads_set_unit_size(GstPad *pad, guint unit_size)
{
	GstLALCollectData *data = gst_pad_get_element_private(pad);

	g_return_if_fail(data != NULL);

	data->unit_size = unit_size;
}


/**
 * Retrieve the number of bytes per sample on the given input stream.
 */


guint gstlal_collect_pads_get_unit_size(GstPad *pad)
{
	GstLALCollectData *data = gst_pad_get_element_private(pad);

	g_return_val_if_fail(data != NULL, -1);

	return data->unit_size;
}


/**
 * Compute the smallest segment that contains the segments of all pads.
 * The segments must be in the same format on all pads.
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
				return NULL;
			}
			continue;
		}

		/*
		 * check for format/rate mismatch
		 */

		if(segment->format != data->segment.format || segment->applied_rate != data->segment.applied_rate) {
			GST_ERROR_OBJECT(pads, "mismatch in segment format and/or applied rate");
			gst_segment_free(segment);
			return NULL;
		}

		/*
		 * expand start and stop
		 */

		if(segment->start == -1 || segment->start > data->segment.start)
			segment->start = data->segment.start;
		if(segment->stop == -1 || segment->stop < data->segment.stop)
			segment->stop = data->segment.stop;
	}

	return segment;
}


/**
 * Computes the earliest of the offsets and of the upper bounds of the
 * offsets spanned by the GstCollectPad's input buffers.  All offsets are
 * converted to their equivalent offsets in the output stream (if this
 * results in a negative offset, then it is replaced with 0).  Sets both
 * offsets to GST_BUFFER_OFFSET_NONE if one or more input buffers has
 * invalid offsets or all pads report EOS.  The return value is FALSE if at
 * least one input buffer had invalid offsets.  The calling code should
 * interpret this to indicate the presence of invalid input on at least one
 * pad.  The return value is TRUE if no input buffer had invalid offsets
 * (when all sink pads report EOS there are zero input buffers, the return
 * value is TRUE).
 *
 * Summary:
 *
 * condition   return value   offsets
 * ----------------------------------
 * bad input   FALSE          GST_BUFFER_OFFSET_NONE
 * EOS         TRUE           GST_BUFFER_OFFSET_NONE
 * success     TRUE           >= 0
 *
 * Requires the collect pad's lock to be held (use from within the callback
 * handler).
 */


static gint64 compute_offset_offset(GstBuffer *buf, gint rate, GstClockTime output_timestamp_at_zero_offset)
{
	/*
	 * subtract buffer's offset from its offset in the output stream.
	 */

	/* FIXME:  the floating-point versions work better than the scale
	 * functions, should I tell the gstreamer people? */
	if(GST_BUFFER_TIMESTAMP(buf) >= output_timestamp_at_zero_offset)
		return (gint64) round((double) (GST_BUFFER_TIMESTAMP(buf) - output_timestamp_at_zero_offset) * rate / GST_SECOND) - (gint64) GST_BUFFER_OFFSET(buf);
		/*return (gint64) gst_util_uint64_scale_int(GST_BUFFER_TIMESTAMP(buf) - output_timestamp_at_zero_offset, rate, GST_SECOND) - (gint64) GST_BUFFER_OFFSET(buf);*/
	else
		return -(gint64) round((double) (output_timestamp_at_zero_offset - GST_BUFFER_TIMESTAMP(buf)) * rate / GST_SECOND) - (gint64) GST_BUFFER_OFFSET(buf);
		/*return -(gint64) gst_util_uint64_scale_int(output_timestamp_at_zero_offset - GST_BUFFER_TIMESTAMP(buf), rate, GST_SECOND) - (gint64) GST_BUFFER_OFFSET(buf);*/
}


/* FIXME:  it would be nice not to have to pass in the rate and timestamp
 * metadata. */


gboolean gstlal_collect_pads_get_earliest_offsets(GstCollectPads *pads, guint64 *offset, guint64 *offset_end, gint rate, GstClockTime output_timestamp_at_zero_offset)
{
	gboolean valid = FALSE;
	GSList *collectdatalist = NULL;

	/*
	 * safety
	 */

	*offset = *offset_end = G_MAXUINT64;

	/*
	 * loop over sink pads
	 */

	for(collectdatalist = pads->data; collectdatalist; collectdatalist = g_slist_next(collectdatalist)) {
		GstLALCollectData *data = collectdatalist->data;
		GstBuffer *buf;
		gint64 offset_offset;
		guint64 this_offset;
		guint64 this_offset_end;

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
			GST_LOG("%p: input buffer does not have valid offset\n", data);
			gst_buffer_unref(buf);
			return FALSE;
		}

		if(!GST_BUFFER_TIMESTAMP_IS_VALID(buf)) {
			GST_LOG("%p: input buffer does not have valid timestamp\n", data);
			gst_buffer_unref(buf);
			return FALSE;
		}

		/*
		 * compute the next sub-buffer's start and end offsets
		 */

		this_offset = GST_BUFFER_OFFSET(buf) + data->as_gstcollectdata.pos / data->unit_size;

		if(GST_BUFFER_OFFSET_END_IS_VALID(buf))
			this_offset_end = GST_BUFFER_OFFSET_END(buf);
		else
			/*
			 * end offset is not valid, derive the end offset
			 * from the start offset, buffer size, and unit
			 * size
			 */
			this_offset_end = GST_BUFFER_OFFSET(buf) + GST_BUFFER_SIZE(buf) / data->unit_size;

		/*
		 * convert the start and end offsets to start and end
		 * offsets in the output stream
		 */

		offset_offset = compute_offset_offset(buf, rate, output_timestamp_at_zero_offset);
		this_offset = (offset_offset < 0 && this_offset < (guint64) -offset_offset) ? 0 : this_offset + offset_offset;
		this_offset_end = (offset_offset < 0 && this_offset_end < (guint64) -offset_offset) ? 0 : this_offset_end + offset_offset;

		GST_DEBUG_OBJECT(GST_PAD_PARENT(data->as_gstcollectdata.pad), "(%s): offset_offset = %ld, offset = %lu, offset_end = %lu, size = %u, unit_size = %d\n", GST_PAD_NAME(data->as_gstcollectdata.pad), offset_offset, GST_BUFFER_OFFSET(buf), GST_BUFFER_OFFSET_END(buf), GST_BUFFER_SIZE(buf), data->unit_size);
		GST_DEBUG_OBJECT(GST_PAD_PARENT(data->as_gstcollectdata.pad), "(%s): this_offset = %lu, this_offset_end = %lu\n", GST_PAD_NAME(data->as_gstcollectdata.pad), this_offset, this_offset_end);

		gst_buffer_unref(buf);

		if(this_offset_end < this_offset) {
			GST_LOG("%p: input buffer appears to have negative length\n", data);
			return FALSE;
		}

		/*
		 * update the minima
		 */

		if(this_offset < *offset)
			*offset = this_offset;
		if(this_offset_end < *offset_end)
			*offset_end = this_offset_end;

		/*
		 * with at least one valid pair of offsets, we can return
		 * meaningful numbers.
		 */

		valid = TRUE;
	}

	/*
	 * found at least one buffer?
	 */

	if(!valid)
		*offset = *offset_end = GST_BUFFER_OFFSET_NONE;
	GST_DEBUG("%p: rate = %d, t0 = %lu, offset = %lu, offset_end = %lu\n", pads, rate, output_timestamp_at_zero_offset, *offset, *offset_end);

	return TRUE;
}


/**
 * wrapper for gst_collect_pads_take_buffer().  Returns a buffer containing
 * the samples taken from the start of the current buffer upto (not
 * including) an offset of offset_end.  The buffer returned might be
 * shorter if the pad does not have data upto the requested offset.  The
 * buffer returned by this function has its offset and offset_end set to
 * properly indicate its location in the output stream.  Calling this
 * function has the effect of flushing the pad upto the offset_end or the
 * upper bound of the available data, whichever comes first.
 *
 * If the pad has no data available then NULL is returned, this indicates
 * EOS.  If the pad has data available but it is subsequent to the
 * requested interval then a zero-length buffer is returned.
 */


GstBuffer *gstlal_collect_pads_take_buffer(GstCollectPads *pads, GstLALCollectData *data, guint64 offset_end, gint rate, GstClockTime output_timestamp_at_zero_offset)
{
	GstBuffer *buf;
	guint64 dequeued_offset;
	guint64 length;

	/*
	 * check for uninitialized GstLALCollectData
	 */

	g_return_val_if_fail(data->unit_size != 0, NULL);

	/*
	 * retrieve the offset (in the output stream) of the next buffer to
	 * be dequeued.
	 */

	buf = gst_collect_pads_peek(pads, (GstCollectData *) data);
	if(!buf)
		/*
		 * EOS
		 */
		return NULL;
	/* FIXME:  is a negative answer possible? */
	dequeued_offset = (GST_BUFFER_OFFSET(buf) + data->as_gstcollectdata.pos / data->unit_size) + compute_offset_offset(buf, rate, output_timestamp_at_zero_offset);
	gst_buffer_unref(buf);

	/*
	 * compute the number of samples to request from the queued buffer.
	 * if the output stream has not yet advanced into the queued buffer
	 * then set the length to 0 to return an empty buffer.
	 */

	length = offset_end >= dequeued_offset ? offset_end - dequeued_offset : 0;
	GST_DEBUG_OBJECT(GST_PAD_PARENT(data->as_gstcollectdata.pad), "(%s): dequeued offset = %lu, length = %lu\n", GST_PAD_NAME(data->as_gstcollectdata.pad), dequeued_offset, length);

	/*
	 * retrieve a buffer
	 */

	buf = gst_collect_pads_take_buffer(pads, (GstCollectData *) data, length * data->unit_size);
	if(!buf)
		/*
		 * EOS or no data (probably impossible, because we would've
		 * detected this above, but might as well check again)
		 */
		return NULL;

	/*
	 * set the buffer's start and end offsets and time stamp and
	 * duration in the output stream
	 */

	buf = gst_buffer_make_metadata_writable(buf);
	GST_BUFFER_OFFSET(buf) = dequeued_offset;
	GST_BUFFER_OFFSET_END(buf) = dequeued_offset + GST_BUFFER_SIZE(buf) / data->unit_size;
	GST_BUFFER_TIMESTAMP(buf) = output_timestamp_at_zero_offset + gst_util_uint64_scale_int(GST_BUFFER_OFFSET(buf), GST_SECOND, rate);
	GST_BUFFER_DURATION(buf) = gst_util_uint64_scale_int(GST_BUFFER_OFFSET_END(buf), GST_SECOND, rate) - gst_util_uint64_scale_int(GST_BUFFER_OFFSET(buf), GST_SECOND, rate);
	GST_DEBUG_OBJECT(GST_PAD_PARENT(data->as_gstcollectdata.pad), "(%s): returning %lu -- %lu, %lu -- %lu\n", GST_PAD_NAME(data->as_gstcollectdata.pad), GST_BUFFER_OFFSET(buf), GST_BUFFER_OFFSET_END(buf), GST_BUFFER_TIMESTAMP(buf), GST_BUFFER_TIMESTAMP(buf) + GST_BUFFER_DURATION(buf));

	return buf;
}

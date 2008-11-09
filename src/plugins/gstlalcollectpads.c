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

	/*
	 * handle events
	 */

	switch (GST_EVENT_TYPE(event)) {
	case GST_EVENT_NEWSEGMENT:
		/*
		 * flag the offset_offset as invalid to force a resync
		 */

		data->offset_offset_valid = FALSE;
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


GstLALCollectData *gstlal_collect_pads_add_pad(GstCollectPads *pads, GstPad *pad, guint size)
{
	GstLALCollectData *data;

	/*
	 * add pad to collect pads object
	 */

	data = (GstLALCollectData *) gst_collect_pads_add_pad(pads, pad, size);
	if(!data) {
		GST_DEBUG_OBJECT(pads, "could not add pad to collectpads object");
		return NULL;
	}

	/*
	 * initialize our own extra contents of the GstAdderCollectData
	 * structure
	 */

	data->offset_offset_valid = FALSE;
	data->offset_offset = 0;

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
 *
 * This function is where the offset_offset for each sink pad is
 * set/updated.
 */


/* FIXME:  it would be nice not to have to pass in the rate,
 * bytes_per_sample, and timestamp metadata. */


gboolean gstlal_collect_pads_get_earliest_offsets(GstCollectPads *pads, guint64 *offset, guint64 *offset_end, gint rate, gint bytes_per_sample, GstClockTime output_timestamp_at_zero_offset)
{
	/* internally we work with signed values so that we can handle
	 * negative offsets without confusion */
	gint64 _offset = G_MAXINT64;
	gint64 _offset_end = G_MAXINT64;
	gboolean valid = FALSE;
	GSList *collected;

	/*
	 * safety
	 */

	*offset = *offset_end = GST_BUFFER_OFFSET_NONE;

	/*
	 * loop over sink pads
	 */

	for(collected = pads->data; collected; collected = g_slist_next(collected)) {
		GstLALCollectData *data = collected->data;
		GstBuffer *buf = gst_collect_pads_peek(pads, &data->collectdata);
		gint64 this_offset;
		gint64 this_offset_end;

		/*
		 * check for EOS
		 */

		if(!buf) {
			GST_LOG("%p: EOS\n", data);
			continue;
		}

		/*
		 * require a valid start offset
		 */

		if(!GST_BUFFER_OFFSET_IS_VALID(buf)) {
			GST_LOG("%p: input buffer does not have valid offset\n", data);
			gst_buffer_unref(buf);
			return FALSE;
		}

		/*
		 * (re)set this pad's offset_offset if this buffer is
		 * flagged as a discontinuity and we have not yet extracted
		 * data from it, or if this pad's offset_offset is not yet
		 * valid.
		 */

		if((GST_BUFFER_IS_DISCONT(buf) && !data->collectdata.pos) || !data->offset_offset_valid) {
			/*
			 * require a valid timestamp
			 */

			if(!GST_BUFFER_TIMESTAMP_IS_VALID(buf)) {
				GST_LOG("%p: input buffer does not have valid timestamp\n", data);
				data->offset_offset_valid = FALSE;
				gst_buffer_unref(buf);
				return FALSE;
			}

			/*
			 * this buffer's offset in the input stream.
			 */

			data->offset_offset = GST_BUFFER_OFFSET(buf);

			/*
			 * subtract from that this buffer's offset in the
			 * output stream.
			 */

			data->offset_offset -= ((gint64) GST_BUFFER_TIMESTAMP(buf) - (gint64) output_timestamp_at_zero_offset) * rate / GST_SECOND;

			/*
			 * offset_offset is now valid.
			 */

			data->offset_offset_valid = TRUE;
		} else {
			/* FIXME:  add a sanity check? that the current
			 * offset_offset does not disagree with the
			 * buffer's timestamp by more than X samples? */
		}

		/*
		 * compute this buffer's start and end offsets in the
		 * output stream
		 */

		this_offset = (gint64) GST_BUFFER_OFFSET(buf) + data->collectdata.pos / bytes_per_sample - data->offset_offset;

		if(GST_BUFFER_OFFSET_END_IS_VALID(buf)) {
			this_offset_end = (gint64) GST_BUFFER_OFFSET_END(buf) - data->offset_offset;
		} else {
			/*
			 * end offset is not valid, but start offset is
			 * valid (see above) so derive the end offset from
			 * the start offset, buffer size, and bytes /
			 * sample
			 */

			this_offset_end = (gint64) (GST_BUFFER_OFFSET(buf) + GST_BUFFER_SIZE(buf) / bytes_per_sample) - data->offset_offset;
		}
		gst_buffer_unref(buf);

		/*
		 * update the minima
		 */

		if(this_offset < _offset)
			_offset = this_offset;
		if(this_offset_end < _offset_end)
			_offset_end = this_offset_end;

		/*
		 * with at least one valid pair of offsets, we can return
		 * meaningful numbers.
		 */

		valid = TRUE;
	}

	/*
	 * found at least one buffer?
	 */

	if(valid) {
		/*
		 * store results in (unsigned) output variables
		 */

		*offset = _offset < 0 ? 0 : (guint64) _offset;
		*offset_end = _offset_end < (gint64) *offset ? *offset : (guint64) _offset_end;
	}

	return TRUE;
}


/**
 * wrapper for gst_collect_pads_take_buffer().  Returns a buffer with its
 * offset set to properly indicate its location in the output stream.  The
 * offset and length parameters indicate the range of offsets the calling
 * code would like the buffer to span.  The buffer returned by this
 * function may span an interval preceding the requested interval, but will
 * never span an interval subsequent to that requested.  Calling this
 * function has the effect of flushing the pad upto the upper bound of the
 * requested interval or the upper bound of the available data, whichever
 * comes first.
 *
 * If the pad has no data available then NULL is returned, this indicates
 * EOS.  If the pad has data available but none of it spans the requested
 * interval then a zero-length buffer is returned.
 */


GstBuffer *gstlal_collect_pads_take_buffer(GstCollectPads *pads, GstLALCollectData *data, guint64 offset, gint64 length, size_t bytes_per_sample)
{
	guint64 dequeued_offset;
	GstBuffer *buf;

	/*
	 * can't do anything unless we understand the relationship between
	 * this input stream's offsets and the output stream's offsets
	 */

	if(!data->offset_offset_valid)
		return NULL;

	/*
	 * retrieve the offset (in the output stream) of the next buffer to
	 * be dequeued.
	 */

	buf = gst_collect_pads_peek(pads, &data->collectdata);
	if(!buf)
		/*
		 * EOS
		 */
		return NULL;
	dequeued_offset = GST_BUFFER_OFFSET(buf) + data->collectdata.pos / bytes_per_sample - data->offset_offset;
	gst_buffer_unref(buf);

	/*
	 * compute the number of samples to request from the queued buffer.
	 * if negative then the output stream has not yet advanced into the
	 * queued buffer --> set the length to 0 to return an empty buffer.
	 */

	length += (gint64) offset - (gint64) dequeued_offset;
	if(length < 0)
		length = 0;

	/*
	 * retrieve a buffer
	 */

	buf = gst_collect_pads_take_buffer(pads, &data->collectdata, length * bytes_per_sample);
	if(!buf)
		/*
		 * EOS or no data (probably impossible, because we would've
		 * detected this above, but might as well check again)
		 */
		return NULL;

	/*
	 * set the buffer's offset
	 */

	buf = gst_buffer_make_metadata_writable(buf);
	GST_BUFFER_OFFSET(buf) = dequeued_offset;

	return buf;
}

/*
 * GstAudioAdapter
 *
 * Copyright (C) 2011  Kipp Cannon
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


#include <string.h>
#include <glib.h>
#include <gst/gst.h>


#include <gstaudioadapter.h>


/*
 * ============================================================================
 *
 *                             Exported Interface
 *
 * ============================================================================
 */


struct gstlal_input_queue *gstlal_input_queue_create(gint unit_size)
{
	struct gstlal_input_queue *new;

	new = g_malloc(sizeof(*new));
	if(!new)
		goto error_no_mem;

	new->queue = g_queue_new();
	if(!new->queue)
		goto error_no_queue;

	new->unit_size = unit_size;
	new->size = 0;
	new->skip = 0;

	return new;

error_no_queue:
	g_free(new);
error_no_mem:
	return NULL;
}


void gstlal_input_queue_drain(struct gstlal_input_queue *input_queue)
{
	GstBuffer *buf;
	while((buf = g_queue_pop_head(input_queue->queue)))
		gst_buffer_unref(buf);
	input_queue->size = 0;
	input_queue->skip = 0;
}


void gstlal_input_queue_free(struct gstlal_input_queue *input_queue)
{
	if(input_queue) {
		gstlal_input_queue_drain(input_queue);
		g_queue_free(input_queue->queue);
		input_queue->queue = NULL;
	}
	g_free(input_queue);
}


gint gstlal_input_queue_get_size(const struct gstlal_input_queue *input_queue)
{
	return input_queue->size;
}


gint gstlal_input_queue_get_unit_size(const struct gstlal_input_queue *input_queue)
{
	return input_queue->unit_size;
}


void gstlal_input_queue_set_unit_size(struct gstlal_input_queue *input_queue, gint unit_size)
{
	if(unit_size != input_queue->unit_size) {
		gstlal_input_queue_drain(input_queue);
		input_queue->unit_size = unit_size;
	}
}


void gstlal_input_queue_push(struct gstlal_input_queue *input_queue, GstBuffer *buf)
{
	g_assert(GST_BUFFER_OFFSET_IS_VALID(buf));
	g_assert(GST_BUFFER_OFFSET_END_IS_VALID(buf));
	g_queue_push_tail(input_queue->queue, buf);
	input_queue->size += GST_BUFFER_OFFSET_END(buf) - GST_BUFFER_OFFSET(buf);
}


gboolean gstlal_input_queue_is_gap(struct gstlal_input_queue *input_queue)
{
	GList *head;

	for(head = g_queue_peek_head_link(input_queue->queue); head; head = g_list_next(head))
		if(!GST_BUFFER_FLAG_IS_SET(head->data, GST_BUFFER_FLAG_GAP))
			return FALSE;

	return TRUE;
}


void gstlal_input_queue_copy(struct gstlal_input_queue *input_queue, void *dst, guint samples, gboolean *copied_gap, gboolean *copied_nongap)
{
	GList *head = g_queue_peek_head_link(input_queue->queue);
	gboolean gap = FALSE;
	gboolean nongap = FALSE;
	guint n = GST_BUFFER_OFFSET_END(head->data) - GST_BUFFER_OFFSET(head->data) - input_queue->skip;

	if(samples < n) {
		if(GST_BUFFER_FLAG_IS_SET(head->data, GST_BUFFER_FLAG_GAP)) {
			memset(dst, 0, samples * input_queue->unit_size);
			gap = TRUE;
		} else {
			memcpy(dst, GST_BUFFER_DATA(head->data) + input_queue->skip * input_queue->unit_size, samples * input_queue->unit_size);
			nongap = TRUE;
		}
		goto done;
	} else {
		if(GST_BUFFER_FLAG_IS_SET(head->data, GST_BUFFER_FLAG_GAP)) {
			memset(dst, 0, n * input_queue->unit_size);
			gap = TRUE;
		} else {
			memcpy(dst, GST_BUFFER_DATA(head->data) + input_queue->skip * input_queue->unit_size, n * input_queue->unit_size);
			nongap = TRUE;
		}
		dst += n * input_queue->unit_size;
		samples -= n;
	}

	while(samples) {
		head = g_list_next(head);
		n = GST_BUFFER_OFFSET_END(head->data) - GST_BUFFER_OFFSET(head->data);

		if(samples < n) {
			if(GST_BUFFER_FLAG_IS_SET(head->data, GST_BUFFER_FLAG_GAP)) {
				memset(dst, 0, samples * input_queue->unit_size);
				gap = TRUE;
			} else {
				memcpy(dst, GST_BUFFER_DATA(head->data), samples * input_queue->unit_size);
				nongap = TRUE;
			}
			goto done;
		} else {
			if(GST_BUFFER_FLAG_IS_SET(head->data, GST_BUFFER_FLAG_GAP)) {
				memset(dst, 0, n * input_queue->unit_size);
				gap = TRUE;
			} else {
				memcpy(dst, GST_BUFFER_DATA(head->data), n * input_queue->unit_size);
				nongap = TRUE;
			}
		}

		dst += n * input_queue->unit_size;
		samples -= n;
	}

done:
	if(copied_gap)
		*copied_gap = gap;
	if(copied_nongap)
		*copied_nongap = nongap;
	return;
}


void gstlal_input_queue_flush(struct gstlal_input_queue *input_queue, guint samples)
{
	while(samples) {
		GstBuffer *head = g_queue_peek_head(input_queue->queue);
		guint n = GST_BUFFER_OFFSET_END(head) - GST_BUFFER_OFFSET(head) - input_queue->skip;

		if(samples < n) {
			input_queue->skip += samples;
			input_queue->size -= samples;
			goto done;
		} else {
			input_queue->skip = 0;
			input_queue->size -= n;
			samples -= n;
			gst_buffer_unref(g_queue_pop_head(input_queue->queue));
		}
	}

done:
	return;
}

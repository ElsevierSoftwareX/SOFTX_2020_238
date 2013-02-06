/*
 * GstAudioAdapter
 *
 * Copyright (C) 2011,2012  Kipp Cannon
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
 *                                Boilerplate
 *
 * ============================================================================
 */


GST_BOILERPLATE(GstAudioAdapter, gst_audioadapter, GObject, G_TYPE_OBJECT);


enum property {
	PROP_UNITSIZE = 1,
	PROP_SIZE
};


/*
 * ============================================================================
 *
 *                               Internal Code
 *
 * ============================================================================
 */


static guint samples_remaining(GstBuffer *buf, guint skip)
{
	guint n = GST_BUFFER_OFFSET_END(buf) - GST_BUFFER_OFFSET(buf);
	g_assert_cmpuint(skip, <=, n);
	return n - skip;
}


static GstClockTime expected_timestamp(GstAudioAdapter *adapter)
{
	GstBuffer *buf = GST_BUFFER(g_queue_peek_tail(adapter->queue));
	g_assert(GST_BUFFER_TIMESTAMP_IS_VALID(buf));
	g_assert(GST_BUFFER_DURATION_IS_VALID(buf));
	return GST_BUFFER_TIMESTAMP(buf) + GST_BUFFER_DURATION(buf);
}


static guint64 expected_offset(GstAudioAdapter *adapter)
{
	GstBuffer *buf = GST_BUFFER(g_queue_peek_tail(adapter->queue));
	g_assert(GST_BUFFER_OFFSET_END_IS_VALID(buf));
	return GST_BUFFER_OFFSET_END(buf);
}


/*
 * ============================================================================
 *
 *                                Exported API
 *
 * ============================================================================
 */


gboolean gst_audioadapter_is_empty(GstAudioAdapter *adapter)
{
	return g_queue_is_empty(adapter->queue);
}


GstClockTime gst_audioadapter_expected_timestamp(GstAudioAdapter *adapter)
{
	return g_queue_is_empty(adapter->queue) ? GST_CLOCK_TIME_NONE : expected_timestamp(adapter);
}


guint64 gst_audioadapter_expected_offset(GstAudioAdapter *adapter)
{
	return g_queue_is_empty(adapter->queue) ? GST_BUFFER_OFFSET_NONE : expected_offset(adapter);
}


void gst_audioadapter_clear(GstAudioAdapter *adapter)
{
	void *buf;
	while((buf = g_queue_pop_head(adapter->queue)))
		gst_buffer_unref(GST_BUFFER(buf));
	adapter->size = 0;
	adapter->skip = 0;
	g_object_notify(G_OBJECT(adapter), "size");
}


void gst_audioadapter_push(GstAudioAdapter *adapter, GstBuffer *buf)
{
	g_assert(GST_BUFFER_OFFSET_IS_VALID(buf));
	g_assert(GST_BUFFER_OFFSET_END_IS_VALID(buf));
	g_queue_push_tail(adapter->queue, buf);
	adapter->size += GST_BUFFER_OFFSET_END(buf) - GST_BUFFER_OFFSET(buf);
	g_object_notify(G_OBJECT(adapter), "size");
}


gboolean gst_audioadapter_is_gap(GstAudioAdapter *adapter)
{
	GList *head;

	for(head = g_queue_peek_head_link(adapter->queue); head; head = g_list_next(head)) {
		GstBuffer *buf = GST_BUFFER(head->data);
		if(!GST_BUFFER_FLAG_IS_SET(buf, GST_BUFFER_FLAG_GAP) && (GST_BUFFER_OFFSET_END(buf) - GST_BUFFER_OFFSET(buf)))
			return FALSE;
	}

	return TRUE;
}


guint gst_audioadapter_head_gap_length(GstAudioAdapter *adapter)
{
	guint length = 0;
	GList *head;

	for(head = g_queue_peek_head_link(adapter->queue); head && (GST_BUFFER_FLAG_IS_SET(GST_BUFFER(head->data), GST_BUFFER_FLAG_GAP) || !(GST_BUFFER_OFFSET_END(head->data) - GST_BUFFER_OFFSET(head->data))); head = g_list_next(head))
		length += GST_BUFFER_OFFSET_END(head->data) - GST_BUFFER_OFFSET(head->data);
	if(length) {
		g_assert_cmpuint(length, >=, adapter->skip);
		length -= adapter->skip;
	}
	g_assert_cmpuint(length, <=, adapter->size);

	return length;
}


guint gst_audioadapter_tail_gap_length(GstAudioAdapter *adapter)
{
	guint length = 0;
	GList *tail;

	for(tail = g_queue_peek_tail_link(adapter->queue); tail && (GST_BUFFER_FLAG_IS_SET(GST_BUFFER(tail->data), GST_BUFFER_FLAG_GAP) || !(GST_BUFFER_OFFSET_END(tail->data) - GST_BUFFER_OFFSET(tail->data))); tail = g_list_previous(tail))
		length += GST_BUFFER_OFFSET_END(tail->data) - GST_BUFFER_OFFSET(tail->data);

	return MIN(length, adapter->size);
}


guint gst_audioadapter_head_nongap_length(GstAudioAdapter *adapter)
{
	guint length = 0;
	GList *head;

	for(head = g_queue_peek_head_link(adapter->queue); head && (!GST_BUFFER_FLAG_IS_SET(GST_BUFFER(head->data), GST_BUFFER_FLAG_GAP) || !(GST_BUFFER_OFFSET_END(head->data) - GST_BUFFER_OFFSET(head->data))); head = g_list_next(head))
		length += GST_BUFFER_OFFSET_END(head->data) - GST_BUFFER_OFFSET(head->data);
	if(length) {
		g_assert_cmpuint(length, >=, adapter->skip);
		length -= adapter->skip;
	}
	g_assert_cmpuint(length, <=, adapter->size);

	return length;
}


guint gst_audioadapter_tail_nongap_length(GstAudioAdapter *adapter)
{
	guint length = 0;
	GList *tail;

	for(tail = g_queue_peek_tail_link(adapter->queue); tail && (!GST_BUFFER_FLAG_IS_SET(GST_BUFFER(tail->data), GST_BUFFER_FLAG_GAP) || !(GST_BUFFER_OFFSET_END(tail->data) - GST_BUFFER_OFFSET(tail->data))); tail = g_list_previous(tail))
		length += GST_BUFFER_OFFSET_END(tail->data) - GST_BUFFER_OFFSET(tail->data);

	return MIN(length, adapter->size);
}


void gst_audioadapter_copy_samples(GstAudioAdapter *adapter, void *dst, guint samples, gboolean *copied_gap, gboolean *copied_nongap)
{
	GList *head = g_queue_peek_head_link(adapter->queue);
	GstBuffer *buf;
	gboolean _copied_gap = FALSE;
	gboolean _copied_nongap = FALSE;
	guint n;

	if(!samples)
		goto done;
	g_assert_cmpuint(samples, <=, adapter->size);

	/* first buffer might need to have some samples skipped so it needs
	 * to be handled separately */
	buf = GST_BUFFER(head->data);
	n = MIN(samples, samples_remaining(buf, adapter->skip));
	if(GST_BUFFER_FLAG_IS_SET(buf, GST_BUFFER_FLAG_GAP)) {
		memset(dst, 0, n * adapter->unit_size);
		_copied_gap = TRUE;
	} else {
		memcpy(dst, GST_BUFFER_DATA(buf) + adapter->skip * adapter->unit_size, n * adapter->unit_size);
		_copied_nongap = TRUE;
	}
	dst += n * adapter->unit_size;
	samples -= n;

	while(samples) {
		buf = GST_BUFFER((head = g_list_next(head))->data);

		n = MIN(samples, GST_BUFFER_OFFSET_END(buf) - GST_BUFFER_OFFSET(buf));
		if(GST_BUFFER_FLAG_IS_SET(buf, GST_BUFFER_FLAG_GAP)) {
			memset(dst, 0, n * adapter->unit_size);
			_copied_gap = TRUE;
		} else {
			memcpy(dst, GST_BUFFER_DATA(buf), n * adapter->unit_size);
			_copied_nongap = TRUE;
		}
		dst += n * adapter->unit_size;
		samples -= n;
	}

done:
	if(copied_gap)
		*copied_gap = _copied_gap;
	if(copied_nongap)
		*copied_nongap = _copied_nongap;
	return;
}


#undef gst_audioadapter_copy
void gst_audioadapter_copy(GstAudioAdapter *adapter, void *dst, guint samples, gboolean *copied_gap, gboolean *copied_nongap)
{
	/* compatibility stub */
	gst_audioadapter_copy_samples(adapter, dst, samples, copied_gap, copied_nongap);
}


GList *gst_audioadapter_get_list_samples(GstAudioAdapter *adapter, guint samples)
{
	GList *head;
	GstBuffer *buf;
	guint n;
	GList *result = NULL;

	g_assert_cmpuint(samples, <=, adapter->size);

	if(g_queue_is_empty(adapter->queue) || !samples)
		goto done;

	buf = GST_BUFFER((head = g_queue_peek_head_link(adapter->queue))->data);
	n = samples_remaining(buf, adapter->skip);
	if(adapter->skip || samples < n) {
		GstBuffer *newbuf;
		n = MIN(samples, n);

		newbuf = gst_buffer_create_sub(buf, adapter->skip * adapter->unit_size, n * adapter->unit_size);

		GST_BUFFER_OFFSET(newbuf) = GST_BUFFER_OFFSET(buf) + adapter->skip;
		GST_BUFFER_OFFSET_END(newbuf) = GST_BUFFER_OFFSET(newbuf) + n;

		/* FIXME:  check to make sure gst_buffer_create_sub()
		 * copies caps.  remove this when gstreamer can be relied
		 * on to do this */
		if(!GST_BUFFER_CAPS(newbuf))
			gst_buffer_set_caps(newbuf, GST_BUFFER_CAPS(buf));
		g_assert(GST_BUFFER_CAPS(newbuf) != NULL);

		result = g_list_append(result, newbuf);
	} else {
		gst_buffer_ref(buf);
		result = g_list_append(result, buf);
	}
	samples -= n;
	if(!samples)
		goto done;

	for(head = g_list_next(head); head; head = g_list_next(head)) {
		buf = GST_BUFFER(head->data);
		n = GST_BUFFER_OFFSET_END(buf) - GST_BUFFER_OFFSET(buf);
		if(samples < n) {
			GstBuffer *newbuf = gst_buffer_create_sub(buf, 0, samples * adapter->unit_size);
			GST_BUFFER_OFFSET_END(newbuf) = GST_BUFFER_OFFSET(newbuf) + samples;

			/* FIXME:  check to make sure gst_buffer_create_sub()
			 * copies caps.  remove this when gstreamer can be relied
			 * on to do this */
			if(!GST_BUFFER_CAPS(newbuf))
				gst_buffer_set_caps(newbuf, GST_BUFFER_CAPS(buf));
			g_assert(GST_BUFFER_CAPS(newbuf) != NULL);

			result = g_list_append(result, newbuf);
			samples = 0;
			break;
		} else {
			gst_buffer_ref(buf);
			result = g_list_append(result, buf);
			samples -= n;
		}
	}

done:
	return result;
}


#undef gst_audioadapter_get_list
GList *gst_audioadapter_get_list(GstAudioAdapter *adapter, guint samples)
{
	/* compatibility stub */
	return gst_audioadapter_get_list_samples(adapter, samples);
}


void gst_audioadapter_flush_samples(GstAudioAdapter *adapter, guint samples)
{
	g_assert_cmpuint(samples, <=, adapter->size);

	while(samples) {
		GstBuffer *head = GST_BUFFER(g_queue_peek_head(adapter->queue));
		guint n = samples_remaining(head, adapter->skip);

		if(samples < n) {
			adapter->skip += samples;
			adapter->size -= samples;
			break;
		} else {
			/* want the samples == n case to go this way so
			 * that the buffer is removed from the queue */
			adapter->skip = 0;
			adapter->size -= n;
			samples -= n;
			/* we've already tested the conversion to GstBuffer above */
			gst_buffer_unref(g_queue_pop_head(adapter->queue));
		}
	}

	g_object_notify(G_OBJECT(adapter), "size");
	return;
}


#undef gst_audioadapter_flush
void gst_audioadapter_flush(GstAudioAdapter *adapter, guint samples)
{
	/* compatibility stub */
	gst_audioadapter_flush_samples(adapter, samples);
}


/*
 * ============================================================================
 *
 *                              GObject Methods
 *
 * ============================================================================
 */


static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec)
{
	GstAudioAdapter *adapter = GST_AUDIOADAPTER(object);

	switch(id) {
	case PROP_UNITSIZE: {
		guint unit_size = g_value_get_uint(value);
		if(unit_size != adapter->unit_size) {
			gst_audioadapter_clear(adapter);
			adapter->unit_size = unit_size;
		}
	}
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}
}


static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *pspec)
{
	GstAudioAdapter *adapter = GST_AUDIOADAPTER(object);

	switch(id) {
	case PROP_UNITSIZE:
		g_value_set_uint(value, adapter->unit_size);
		break;

	case PROP_SIZE:
		g_value_set_uint(value, adapter->size);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}
}


static void dispose(GObject *object)
{
	GstAudioAdapter *adapter = GST_AUDIOADAPTER(object);

	gst_audioadapter_clear(adapter);

	G_OBJECT_CLASS(parent_class)->dispose(object);
}


static void finalize(GObject *object)
{
	GstAudioAdapter *adapter = GST_AUDIOADAPTER(object);

	g_queue_free(adapter->queue);
	adapter->queue = NULL;

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


static void gst_audioadapter_base_init(gpointer g_class)
{
	/* no-op */
}


static void gst_audioadapter_class_init(GstAudioAdapterClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

	gobject_class->set_property = set_property;
	gobject_class->get_property = get_property;
	gobject_class->dispose = dispose;
	gobject_class->finalize = finalize;

	g_object_class_install_property(
		gobject_class,
		PROP_UNITSIZE,
		g_param_spec_uint(
			"unit-size",
			"Unit size",
			"The size in bytes of one \"frame\" (one sample from all channels).",
			1, G_MAXUINT, 1,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		PROP_SIZE,
		g_param_spec_uint(
			"size",
			"size",
			"The number of frames in the adapter.",
			0, G_MAXUINT, 0,
			G_PARAM_READABLE | G_PARAM_STATIC_STRINGS
		)
	);
}


static void gst_audioadapter_init(GstAudioAdapter *adapter, GstAudioAdapterClass *g_class)
{
	adapter->queue = g_queue_new();
	adapter->unit_size = 0;
	adapter->size = 0;
	adapter->skip = 0;
}

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
 *                                Exported API
 *
 * ============================================================================
 */


void gst_audioadapter_drain(GstAudioAdapter *adapter)
{
	GstBuffer *buf;
	while((buf = g_queue_pop_head(adapter->queue)))
		gst_buffer_unref(GST_BUFFER(buf));
	adapter->size = 0;
	adapter->skip = 0;
}


void gst_audioadapter_push(GstAudioAdapter *adapter, GstBuffer *buf)
{
	g_assert(GST_BUFFER_OFFSET_IS_VALID(buf));
	g_assert(GST_BUFFER_OFFSET_END_IS_VALID(buf));
	g_queue_push_tail(adapter->queue, buf);
	adapter->size += GST_BUFFER_OFFSET_END(buf) - GST_BUFFER_OFFSET(buf);
}


gboolean gst_audioadapter_is_gap(GstAudioAdapter *adapter)
{
	GList *head;

	for(head = g_queue_peek_head_link(adapter->queue); head; head = g_list_next(head))
		if(!GST_BUFFER_FLAG_IS_SET(GST_BUFFER(head->data), GST_BUFFER_FLAG_GAP))
			return FALSE;

	return TRUE;
}


void gst_audioadapter_copy(GstAudioAdapter *adapter, void *dst, guint samples, gboolean *copied_gap, gboolean *copied_nongap)
{
	GList *head = g_queue_peek_head_link(adapter->queue);
	gboolean gap = FALSE;
	gboolean nongap = FALSE;
	guint n = GST_BUFFER_OFFSET_END(GST_BUFFER(head->data)) - GST_BUFFER_OFFSET(GST_BUFFER(head->data)) - adapter->skip;

	if(samples < n) {
		if(GST_BUFFER_FLAG_IS_SET(GST_BUFFER(head->data), GST_BUFFER_FLAG_GAP)) {
			memset(dst, 0, samples * adapter->unit_size);
			gap = TRUE;
		} else {
			memcpy(dst, GST_BUFFER_DATA(GST_BUFFER(head->data)) + adapter->skip * adapter->unit_size, samples * adapter->unit_size);
			nongap = TRUE;
		}
		goto done;
	} else {
		if(GST_BUFFER_FLAG_IS_SET(GST_BUFFER(head->data), GST_BUFFER_FLAG_GAP)) {
			memset(dst, 0, n * adapter->unit_size);
			gap = TRUE;
		} else {
			memcpy(dst, GST_BUFFER_DATA(GST_BUFFER(head->data)) + adapter->skip * adapter->unit_size, n * adapter->unit_size);
			nongap = TRUE;
		}
		dst += n * adapter->unit_size;
		samples -= n;
	}

	while(samples) {
		head = g_list_next(head);
		n = GST_BUFFER_OFFSET_END(GST_BUFFER(head->data)) - GST_BUFFER_OFFSET(GST_BUFFER(head->data));

		if(samples < n) {
			if(GST_BUFFER_FLAG_IS_SET(GST_BUFFER(head->data), GST_BUFFER_FLAG_GAP)) {
				memset(dst, 0, samples * adapter->unit_size);
				gap = TRUE;
			} else {
				memcpy(dst, GST_BUFFER_DATA(GST_BUFFER(head->data)), samples * adapter->unit_size);
				nongap = TRUE;
			}
			goto done;
		} else {
			if(GST_BUFFER_FLAG_IS_SET(GST_BUFFER(head->data), GST_BUFFER_FLAG_GAP)) {
				memset(dst, 0, n * adapter->unit_size);
				gap = TRUE;
			} else {
				memcpy(dst, GST_BUFFER_DATA(GST_BUFFER(head->data)), n * adapter->unit_size);
				nongap = TRUE;
			}
		}

		dst += n * adapter->unit_size;
		samples -= n;
	}

done:
	if(copied_gap)
		*copied_gap = gap;
	if(copied_nongap)
		*copied_nongap = nongap;
	return;
}


void gst_audioadapter_flush(GstAudioAdapter *adapter, guint samples)
{
	while(samples) {
		GstBuffer *head = GST_BUFFER(g_queue_peek_head(adapter->queue));
		guint n = GST_BUFFER_OFFSET_END(head) - GST_BUFFER_OFFSET(head) - adapter->skip;

		if(samples < n) {
			adapter->skip += samples;
			adapter->size -= samples;
			goto done;
		} else {
			adapter->skip = 0;
			adapter->size -= n;
			samples -= n;
			/* we've already tested the conversion to GstBuffer above */
			gst_buffer_unref(g_queue_pop_head(adapter->queue));
		}
	}

done:
	return;
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
			gst_audioadapter_drain(adapter);
			adapter->unit_size = unit_size;
		}
	}
		break;

	default:
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
		break;
	}
}


static void dispose(GObject *object)
{
	GstAudioAdapter *adapter = GST_AUDIOADAPTER(object);

	gst_audioadapter_drain(adapter);
}


static void finalize(GObject *object)
{
	GstAudioAdapter *adapter = GST_AUDIOADAPTER(object);

	g_queue_free(adapter->queue);
	adapter->queue = NULL;

	parent_class->finalize(object);
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

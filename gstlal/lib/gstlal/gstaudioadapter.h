/*
 * GstAudioAdapter
 *
 * Copyright (C) 2011--2013  Kipp Cannon
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


#ifndef __GSTAUDIOADAPTER_H__
#define __GSTAUDIOADAPTER_H__


/*
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


#include <glib.h>
#include <gst/gst.h>


G_BEGIN_DECLS


/*
 * ============================================================================
 *
 *                                    Type
 *
 * ============================================================================
 */


#define GST_TYPE_AUDIOADAPTER \
	(gst_audioadapter_get_type())
#define GST_AUDIOADAPTER(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_AUDIOADAPTER, GstAudioAdapter))
#define GST_AUDIOADAPTER_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_AUDIOADAPTER, GstAudioAdapterClass))
#define GST_AUDIODAPATER_GET_CLASS(obj) \
	(G_TYPE_INSTANCE_GET_CLASS((obj), GST_TYPE_AUDIOADAPTER, GstAudioAdapterClass))
#define GST_IS_AUDIOADAPTER(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_AUDIOADAPTER))
#define GST_IS_AUDIOADAPTER_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_AUDIOADAPTER))


typedef struct _GstAudioAdapter GstAudioAdapter;
typedef struct _GstAudioAdapterClass GstAudioAdapterClass;


/**
 * GstAudioAdapter:
 *
 * The opaque #GstAudioAdapter instance structure.
 */


struct _GstAudioAdapter {
	GObject object;

	/*< private >*/
	GQueue *queue;
	guint unit_size;	/* bytes */
	guint size;		/* samples (units) */
	guint skip;		/* samples (units) */
};


/**
 * GstAudioAdapterClass:
 */


struct _GstAudioAdapterClass {
	GObjectClass parent_class;
};


/*
 * ============================================================================
 *
 *                                Exported API
 *
 * ============================================================================
 */


GType gst_audioadapter_get_type(void);


gboolean gst_audioadapter_is_empty(GstAudioAdapter *adapter);
GstClockTime gst_audioadapter_expected_timestamp(GstAudioAdapter *adapter);
guint64 gst_audioadapter_expected_offset(GstAudioAdapter *adapter);
void gst_audioadapter_clear(GstAudioAdapter *adapter);
void gst_audioadapter_push(GstAudioAdapter *adapter, GstBuffer *buf);
gboolean gst_audioadapter_is_gap(GstAudioAdapter *adapter);
guint gst_audioadapter_head_gap_length(GstAudioAdapter *adapter);
guint gst_audioadapter_tail_gap_length(GstAudioAdapter *adapter);
guint gst_audioadapter_head_nongap_length(GstAudioAdapter *adapter);
guint gst_audioadapter_tail_nongap_length(GstAudioAdapter *adapter);
void gst_audioadapter_copy_samples(GstAudioAdapter *adapter, void *dst, guint samples, gboolean *copied_gap, gboolean *copied_nongap);
GList *gst_audioadapter_get_list_samples(GstAudioAdapter *adapter, guint samples);
void gst_audioadapter_flush_samples(GstAudioAdapter *adapter, guint samples);


G_END_DECLS


#endif	/* __GSTAUDIOADAPTER_H__ */

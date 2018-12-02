/*
 * Copyright (C) 2017 Leo Tsukada <tsukada@resceu.s.u-tokyo.ac.jp>
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

#ifndef __GST_LAL_TDWHITEN_H__
#define __GST_LAL_TDWHITEN_H__

#include <glib.h>
#include <gst/gst.h>
#include <gst/audio/audio.h>
#include <gst/base/gstadapter.h>
#include <gst/base/gstbasetransform.h>


G_BEGIN_DECLS

#define GSTLAL_TDWHITEN_TYPE \
	(gstlal_tdwhiten_get_type())
#define GSTLAL_TDWHITEN(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_TDWHITEN_TYPE, GSTLALTDwhiten))
#define GSTLAL_TDWHITEN_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_TDWHITEN_TYPE, GSTLALTDwhitenClass))
#define GST_IS_GSTLAL_TDWHITEN(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_TDWHITEN_TYPE))
#define GST_IS_GSTLAL_TDWHITEN_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_TDWHITEN_TYPE))


typedef struct {
	GstBaseTransformClass parent_class;

	void (*rate_changed)(GstElement *, gint, void *);
} GSTLALTDwhitenClass;


typedef struct {
	GstBaseTransform element;

	/*
	 * input stream
	 */

	GstAudioInfo audio_info;
	GstAudioAdapter *adapter;

	/*
	 * kernels
	 */

	guint32 taper_length;
	GQueue *kernels;
	GQueue *waiting_kernels;
	gint64 latency;
	guint64 kernel_endtime;
	GMutex kernel_lock;

	/*
	 * timestamp book-keeping
	 */

	GstClockTime t0;
	GstClockTime next_pts;
	guint64 offset0;
	guint64 next_out_offset;
	guint64 next_in_offset;
	gboolean need_discont;
} GSTLALTDwhiten;


GType gstlal_tdwhiten_get_type(void);


G_END_DECLS

#endif /* __GST_LAL_TDWHITEN_H__ */


/*
 * framecpp channel multiplexor
 *
 * Copyright (C) 2012  Kipp Cannon, Ed Maros
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */


#ifndef __FRAMECPP_CHANNELMUX_H__
#define __FRAMECPP_CHANNELMUX_H__


#include <glib.h>
#include <gst/gst.h>


#include <muxcollectpads.h>


G_BEGIN_DECLS

/*
 *  framecpp_channelmux_compression_scheme enum type
 */


#define FRAMECPP_CHANNELMUX_COMPRESSION_SCHEME_TYPE \
        (framecpp_channelmux_compression_scheme_get_type())


GType framecpp_channelmux_compression_scheme_get_type(void);

/*
 * framecpp_channelmux element
 */


#define FRAMECPP_CHANNELMUX_TYPE \
	(framecpp_channelmux_get_type())
#define FRAMECPP_CHANNELMUX(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), FRAMECPP_CHANNELMUX_TYPE, GstFrameCPPChannelMux))
#define FRAMECPP_CHANNELMUX_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), FRAMECPP_CHANNELMUX_TYPE, GstFrameCPPChannelMuxClass))
#define GST_IS_FRAMECPP_CHANNELMUX(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), FRAMECPP_CHANNELMUX_TYPE))
#define GST_IS_FRAMECPP_CHANNELMUX_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), FRAMECPP_CHANNELMUX_TYPE))


typedef struct {
	GstElementClass parent_class;
} GstFrameCPPChannelMuxClass;


typedef struct {
	GstElement element;

	FrameCPPMuxCollectPads *collect;
	GstPad *srcpad;

	GHashTable *instruments;
	gboolean need_discont;
	gboolean need_tag_list;
	guint64 next_out_offset;

	guint compression_scheme;
	guint compression_level;

	GstClockTime frame_duration;
	guint frames_per_file;

	gchar *frame_name;
	gint frame_run;
	guint frame_number;
	GValueArray *frame_history;
} GstFrameCPPChannelMux;


GType framecpp_channelmux_get_type(void);


G_END_DECLS


#endif	/* __FRAMECPP_CHANNELMUX_H__ */

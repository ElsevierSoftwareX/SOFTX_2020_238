/*
 * framecpp channel demultiplexor
 *
 * Copyright (C) 2011  Kipp Cannon, Ed Maros
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


#ifndef __FRAMECPP_CHANNELDEMUX_H__
#define __FRAMECPP_CHANNELDEMUX_H__


#include <glib.h>
#include <gst/gst.h>


G_BEGIN_DECLS


/*
 * framecpp_channeldemux element
 */


#define FRAMECPP_CHANNELDEMUX_TYPE \
	(framecpp_channeldemux_get_type())
#define FRAMECPP_CHANNELDEMUX(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), FRAMECPP_CHANNELDEMUX_TYPE, GstFrameCPPChannelDemux))
#define FRAMECPP_CHANNELDEMUX_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), FRAMECPP_CHANNELDEMUX_TYPE, GstFrameCPPChannelDemuxClass))
#define GST_IS_FRAMECPP_CHANNELDEMUX(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), FRAMECPP_CHANNELDEMUX_TYPE))
#define GST_IS_FRAMECPP_CHANNELDEMUX_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), FRAMECPP_CHANNELDEMUX_TYPE))


typedef struct {
	GstElementClass parent_class;
} GstFrameCPPChannelDemuxClass;


typedef struct {
	GstElement element;

	GstEvent *last_new_segment_event;
	GstSegment segment;	/* undefined if last_new_segment_event is NULL */

	gboolean do_file_checksum;
	gboolean skip_bad_files;
	GHashTable *channel_list;
	GstTagList *tag_list;

	guint frame_format_version;
	guint frame_library_version;
	gchar *frame_library_name;
	gchar *frame_name;
	gint frame_run;
	guint frame_number;
	GValueArray *frame_history;
} GstFrameCPPChannelDemux;


GType framecpp_channeldemux_get_type(void);


G_END_DECLS


#endif	/* __FRAMECPP_CHANNELDEMUX_H__ */

/*
 * LAL online h(t) src element
 *
 * Copyright (C) 2008  Kipp C. Cannon
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


#ifndef __GSTLAL_ONLINEHOFTSRC_H__
#define __GSTLAL_ONLINEHOFTSRC_H__


#include <gst/gst.h>
#include <gst/base/gstpushsrc.h>

#include <lal/FrameStream.h>
#include <lal/LALDatatypes.h>
#include <lal/Units.h>


G_BEGIN_DECLS


#define GSTLAL_ONLINEHOFTSRC_TYPE \
	(gstlal_onlinehoftsrc_get_type())
#define GSTLAL_ONLINEHOFTSRC(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_ONLINEHOFTSRC_TYPE, GSTLALOnlineHoftSrc))
#define GSTLAL_ONLINEHOFTSRC_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_ONLINEHOFTSRC_TYPE, GSTLALOnlineHoftSrcClass))
#define GST_IS_GSTLAL_ONLINEHOFTSRC(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_ONLINEHOFTSRC_TYPE))
#define GST_IS_GSTLAL_ONLINEHOFTSRC_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_ONLINEHOFTSRC_TYPE))


typedef struct {
	GstBaseSrcClass parent_class;
} GSTLALOnlineHoftSrcClass;


typedef struct {
	GstBaseSrc basesrc;

	char *location;
	char *instrument;
	char *channel_name;
	char *full_channel_name;
	gint rate;
	gint width;

	FrStream *stream;
	LALTYPECODE series_type;
	LALUnit units;
} GSTLALOnlineHoftSrc;


GType gstlal_onlinehoftsrc_get_type(void);


G_END_DECLS


#endif	/* __GSTLAL_ONLINEHOFTSRC_H__ */

/*
 *  NDS2-based frame src element
 *
 *  Copyright (C) 2008  Leo Singer
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


#ifndef __GSTLAL_NDS2_FRAMESRC_H__
#define __GSTLAL_NDS2_FRAMESRC_H__


#include <gst/gst.h>
#include <gst/base/gstpushsrc.h>

#include <lal/FrameStream.h>
#include <lal/LALDatatypes.h>
#include <lal/Units.h>


G_BEGIN_DECLS


#define GSTLAL_NDS2_FRAMESRC_TYPE \
	(gstlal_nds2_framesrc_get_type())
#define GSTLAL_NDS2_FRAMESRC(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_NDS2_FRAMESRC_TYPE, GSTLALNDS2FrameSrc))
#define GSTLAL_NDS2_FRAMESRC_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_NDS2_FRAMESRC_TYPE, GSTLALNDS2FrameSrcClass))
#define GST_IS_GSTLAL_NDS2_FRAMESRC(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_NDS2_FRAMESRC_TYPE))
#define GST_IS_GSTLAL_NDS2_FRAMESRC_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_NDS2_FRAMESRC_TYPE))


typedef struct {
	GstBaseSrcClass parent_class;
} GSTLALNDS2FrameSrcClass;


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
} GSTLALNDS2FrameSrc;


GType gstlal_nds2_framesrc_get_type(void);


G_END_DECLS


#endif	/* __GSTLAL_NDS2_FRAMESRC_H__ */

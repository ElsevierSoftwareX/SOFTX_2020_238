/*
 * multi downsample written in one element
 *
 * Copyright (C) 2014 Qi Chu
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


#ifndef __MULTI_DOWNSAMPLE_H__
#define __MULTI_DOWNSAMPLE_H__


#include <glib.h>
#include <gst/gst.h>


G_BEGIN_DECLS


/*
 * multi_downsample element
 */


#define GSTLAL_MULTI_DOWNSAMPLE_TYPE \
	(gstlal_multi_downsample_get_type())
#define GSTLAL_MULTI_DOWNSAMPLE(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_MULTI_DOWNSAMPLE_TYPE, GstlalMultiDownsample))
#define GSTLAL_MULTI_DOWNSAMPLE_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_MULTI_DOWNSAMPLE_TYPE, GstlalMultiDownsampleClass))
#define GST_IS_GSTLAL_MULTI_DOWNSAMPLE(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_MULTI_DOWNSAMPLE_TYPE))
#define GST_IS_GSTLAL_MULTI_DOWNSAMPLE_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_MULTI_DOWNSAMPLE_TYPE))


typedef struct {
	GstElementClass parent_class;
} GstlalMultiDownsampleClass;


typedef struct {
	GstElement element;

	gint inrate;
	gint depth;

} GstlalMultiDownsample;


GType gstlal_multidownsample_get_type(void);


G_END_DECLS


#endif	/* __FRAMECPP_CHANNELDEMUX_H__ */

/*
 * A tab-separated values dumper to produce files for plotting
 *
 * Copyright (C) 2008--2012  Kipp Cannon, Chad Hanna
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


#ifndef __GST_TSVENC_H__
#define __GST_TSVENC_H__


#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>


G_BEGIN_DECLS
#define GST_TSVENC_TYPE \
	(gst_tsvenc_get_type())
#define GST_TSVENC(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TSVENC_TYPE, GstTSVEnc))
#define GST_TSVENC_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GST_TSVENC_TYPE, GstTSVEncClass))
#define GST_IS_GST_TSVENC(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TSVENC_TYPE))
#define GST_IS_GST_TSVENC_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GST_TSVENC_TYPE))


typedef struct
{
  GstBaseTransformClass parent_class;
} GstTSVEncClass;


typedef struct
{
  GstBaseTransform element;

  gint rate;
  gint channels;
  int (*printsample) (char *, const void **);

  GstClockTime start_time;
  GstClockTime stop_time;
} GstTSVEnc;


GType gst_tsvenc_get_type(void);


G_END_DECLS
#endif                          /* __GST_TSVENC_H__ */

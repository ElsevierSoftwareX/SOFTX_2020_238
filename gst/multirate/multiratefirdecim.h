/* 
 * GStreamer
 * Copyright (C) 2011 Leo Singer <leo.singer@ligo.org>
 * Copyright (C) 2007 Sebastian Dröge <slomo@circular-chaos.org>
 * Copyright (C) 2006 Stefan Kost <ensonic@users.sf.net>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#ifndef __GST_MULTIRATE_FIR_DECIM_H__
#define __GST_MULTIRATE_FIR_DECIM_H__

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <gst/base/gstadapter.h>

G_BEGIN_DECLS
#define GST_TYPE_MULTIRATE_FIR_DECIM            (gst_multirate_fir_decim_get_type())
#define GST_MULTIRATE_FIR_DECIM(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_MULTIRATE_FIR_DECIM,GstMultirateFirDecim))
#define GST_IS_MULTIRATE_FIR_DECIM(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_MULTIRATE_FIR_DECIM))
#define GST_MULTIRATE_FIR_DECIM_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass) ,GST_TYPE_MULTIRATE_FIR_DECIM,GstMultirateFirDecimClass))
#define GST_IS_MULTIRATE_FIR_DECIM_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass) ,GST_TYPE_MULTIRATE_FIR_DECIM))
#define GST_MULTIRATE_FIR_DECIM_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj) ,GST_TYPE_MULTIRATE_FIR_DECIM,GstMultirateFirDecimClass))
typedef struct _GstMultirateFirDecim GstMultirateFirDecim;
typedef struct _GstMultirateFirDecimClass GstMultirateFirDecimClass;

typedef void (*GstMultirateFirDecimProcessFunc) (GstMultirateFirDecim *, void *, guint);

struct _GstMultirateFirDecim
{
  GstBaseTransform audiofilter;

  double *kernel;
  guint kernel_length;
  guint64 lag;

  /* < private > */
  GstClockTime t0;
  guint64 offset0;
  guint64 samples;
  gboolean needs_timestamp;

  gint inrate, outrate, channels;
  GstAdapter *adapter;
  guint downsample_factor;
};

struct _GstMultirateFirDecimClass
{
  GstBaseTransformClass parent;
};

GType gst_multirate_fir_decim_get_type (void);

G_END_DECLS
#endif /* __GST_MULTIRATE_FIR_DECIM_H__ */

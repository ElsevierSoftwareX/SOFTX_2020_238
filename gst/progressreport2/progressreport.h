/* GStreamer Progress Report Element
 * Copyright (C) <1999> Erik Walthinsen <omega@cse.ogi.edu>
 * Copyright (C) <2003> David Schleef <ds@schleef.org>
 * Copyright (C) <2004> Jan Schmidt <thaytan@mad.scientist.com>
 * Copyright (C) <2006> Tim-Philipp Müller <tim centricular net>
 *
 * Modified by Leo Singer <leo.singer@ligo.org> on 2011-07-28;
 * added properties 'high-precision' and 'absolute-time.'
 * Renamed to 'progressreport2' to prevent name collision with
 * original element in gst-plugins-good.
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

#ifndef __GST_PROGRESS_REPORT_2_H__
#define __GST_PROGRESS_REPORT_2_H__

#include <gst/base/gstbasetransform.h>

G_BEGIN_DECLS
#define GST_TYPE_PROGRESS_REPORT_2 \
  (gst_progress_report_2_get_type())
#define GST_PROGRESS_REPORT_2(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_PROGRESS_REPORT_2,GstProgressReport2))
#define GST_PROGRESS_REPORT_2_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_PROGRESS_REPORT_2,GstProgressReport2Class))
#define GST_IS_PROGRESS_REPORT_2(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_PROGRESS_REPORT_2))
#define GST_IS_PROGRESS_REPORT_2_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_PROGRESS_REPORT_2))
typedef struct _GstProgressReport2 GstProgressReport2;
typedef struct _GstProgressReport2Class GstProgressReport2Class;

struct _GstProgressReport2
{
  GstBaseTransform basetransform;

  GstMessage *pending_msg;

  gint update_freq;
  gboolean silent;
  gboolean do_query;
  gboolean high_precision;
  gboolean absolute_time;
  GTimeVal start_time;
  GTimeVal last_report;

  /* Format used for querying. Using a string here because the
   * format might not be registered yet when the property is set */
  gchar *format;
};

struct _GstProgressReport2Class
{
  GstBaseTransformClass parent_class;
};

GType gst_progress_report_2_get_type (void);

G_END_DECLS
#endif /* __GST_PROGRESS_REPORT_2_H__ */

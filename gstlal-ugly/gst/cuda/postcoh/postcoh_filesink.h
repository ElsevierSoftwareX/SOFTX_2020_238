/* 
 * Copyright (C) 2014 Qi Chu <qi.chu@ligo.org>
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


#ifndef __POSTCOH_FILESINK_H__
#define __POSTCOH_FILESINK_H__

#include <stdio.h>

#include <gst/gst.h>
#include <gst/base/gstbasesink.h>

#include <LIGOLw_xmllib/LIGOLwHeader.h>

G_BEGIN_DECLS

#define POSTCOH_TYPE_FILESINK \
  (postcoh_filesink_get_type())
#define POSTCOH_FILESINK(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),POSTCOH_TYPE_FILESINK,PostcohFilesink))
#define POSTCOH_FILESINK_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),POSTCOH_TYPE_FILESINK,PostcohFilesinkClass))
#define GST_IS_POSTCOH_FILESINK(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),POSTCOH_TYPE_FILESINK))
#define GST_IS_POSTCOH_FILESINK_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),POSTCOH_TYPE_FILESINK))

typedef struct _PostcohFilesink PostcohFilesink;
typedef struct _PostcohFilesinkClass PostcohFilesinkClass;

/**
 * PostcohFilesink:
 *
 * Opaque #PostcohFilesink structure.
 */
struct _PostcohFilesink {
  GstBaseSink parent;

  /*< private >*/
  gchar *filename;
  gchar *uri;
  FILE *file;

  gint compress;
  gint snapshot_interval;
  xmlTextWriterPtr writer;
  XmlTable *xtable;
  
  GstClockTime t_start;
  GstClockTime t_end;
  GString *cur_filename;
};

struct _PostcohFilesinkClass {
  GstBaseSinkClass parent_class;
};

G_GNUC_INTERNAL GType postcoh_filesink_get_type (void);

G_END_DECLS

#endif

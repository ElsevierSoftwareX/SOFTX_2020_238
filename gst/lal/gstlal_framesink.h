/*
 * Copyright (C) 2010 Jordi Burguet-Castell <jordi.burguet-castell@ligo.org>
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

/*
 * Copied from gstreamer's gstfilesink:

 cp ../gstreamer/plugins/elements/gstfilesink.h src/plugins/gstlalframesink.h
 cp ../gstreamer/plugins/elements/gstfilesink.c src/plugins/gstlalframesink.c

 * And then used:

 (defun file2frame ()
  (interactive)
  (save-excursion
   (replace-string "GstFileSink" "GstLalframeSink" nil (point-min) (point-max))
   (replace-string "file_" "lalframe_" nil (point-min) (point-max))
   (replace-string "filesink" "lalframesink" nil (point-min) (point-max))))

 * and (beautify)
 */

/*
 * Note on naming: Instead of calling things like gstlal_framesink, I
 * call things like gst_lalframe_sink (or gst_lalframesink), which
 * seems to be more consistent with the other gstreamer elements.
 *
 * Instead of writing things like GSTLALFrameSink we write GstLalframeSink
 */

#ifndef __GST_LALFRAME_SINK_H__
#define __GST_LALFRAME_SINK_H__

#include <stdio.h>

#include <gst/gst.h>
#include <gst/base/gstbasesink.h>

#include <lal/LALFrameIO.h>


G_BEGIN_DECLS

#define GST_TYPE_LALFRAME_SINK                  \
    (gst_lalframe_sink_get_type())
#define GST_LALFRAME_SINK(obj)                                          \
    (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_LALFRAME_SINK, GstLalframeSink))
#define GST_LALFRAME_SINK_CLASS(klass)                                  \
    (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_LALFRAME_SINK, GstLalframeSinkClass))
#define GST_IS_LALFRAME_SINK(obj)                               \
    (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_LALFRAME_SINK))
#define GST_IS_LALFRAME_SINK_CLASS(klass)                       \
    (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_LALFRAME_SINK))

typedef struct _GstLalframeSink GstLalframeSink;
typedef struct _GstLalframeSinkClass GstLalframeSinkClass;

/**
 * GstLalframeSink:
 *
 * Opaque #GstLalframeSink structure.
 */
struct _GstLalframeSink {
    GstBaseSink parent;

    GstPad *sinkpad;

    /*< private >*/
    gchar *filename;
    gchar *uri;
    FrameH *frame;
    gchar *instrument;
    gchar *channel_name;
    gchar *units;

    gboolean seekable;
    guint64 current_pos;

    gint    buffer_mode;
    guint   buffer_size;
    gchar  *buffer;
};

struct _GstLalframeSinkClass {
    GstBaseSinkClass parent_class;
};

GType gst_lalframe_sink_get_type(void);

G_END_DECLS

#endif /* __GST_LALFRAME_SINK_H__ */

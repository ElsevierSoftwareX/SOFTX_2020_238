/*
 * Copyright (C) 2011 Jordi Burguet-Castell <jordi.burguet-castell@ligo.org>
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

#ifndef __GST_LALPAD_H__
#define __GST_LALPAD_H__

#include <stdio.h>

#include <gst/gst.h>

G_BEGIN_DECLS


/*
 * Different types of supported units.
 */
typedef enum {
    GST_LALPAD_UNIT_SAMPLES,
    GST_LALPAD_UNIT_TIME,
} GstLalpadUnit;


/* Definition of structure storing data for this element. */
typedef struct _GstLalpad {
    GstElement parent;

    /* parameters */
    guint64 pre, post;
    GstLalpadUnit unit;

    /*< private >*/
    GstPad *sinkpad, *srcpad;
    guint64 saved_offset, saved_offset_end;
    GstClockTime saved_timestamp, saved_duration;
    gboolean first_buffer;
    GstCaps* caps;
} GstLalpad;

/* Standard definition defining a class for this element. */
typedef struct _GstLalpadClass {
    GstElementClass parent_class;
} GstLalpadClass;

/* Standard macros for defining types for this element.  */
#define GST_TYPE_LALPAD                          \
    (gst_lalpad_get_type())
#define GST_LALPAD(obj)                                          \
    (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_LALPAD, GstLalpad))
#define GST_LALPAD_CLASS(klass)                                  \
    (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_LALPAD, GstLalpadClass))
#define GST_IS_LALPAD(obj)                               \
    (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_LALPAD))
#define GST_IS_LALPAD_CLASS(klass)                       \
    (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_LALPAD))

/* Standard function returning type information. */
GType gst_lalpad_get_type(void);

G_END_DECLS

#endif /* __GST_LALPAD_H__ */

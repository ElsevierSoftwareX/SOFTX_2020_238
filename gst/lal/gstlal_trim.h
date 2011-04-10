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

#ifndef __GST_LALTRIM_H__
#define __GST_LALTRIM_H__

#include <stdio.h>

#include <gst/gst.h>

G_BEGIN_DECLS

/* Definition of structure storing data for this element. */
typedef struct _GstLaltrim {
    GstElement parent;

    GstPad *sinkpad, *srcpad;

    guint64 initial_offset, final_offset;
} GstLaltrim;

/* Standard definition defining a class for this element. */
typedef struct _GstLaltrimClass {
    GstElementClass parent_class;
} GstLaltrimClass;

/* Standard macros for defining types for this element.  */
#define GST_TYPE_LALTRIM                          \
    (gst_laltrim_get_type())
#define GST_LALTRIM(obj)                                          \
    (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_LALTRIM, GstLaltrim))
#define GST_LALTRIM_CLASS(klass)                                  \
    (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_LALTRIM, GstLaltrimClass))
#define GST_IS_LALTRIM(obj)                               \
    (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_LALTRIM))
#define GST_IS_LALTRIM_CLASS(klass)                       \
    (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_LALTRIM))

/* Standard function returning type information. */
GType gst_laltrim_get_type(void);

G_END_DECLS

#endif /* __GST_LALTRIM_H__ */

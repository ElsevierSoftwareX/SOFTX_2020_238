/*
 * Copyright (C) 2016  DASWG <daswg@ligo.org>
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation; either version 2 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */


#include <gst/gst.h>

#define TYPE_SPLIT_COUNTER \
  (split_counter_get_type())
#define SPLIT_COUNTER(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),TYPE_SPLIT_COUNTER,Splitcounter))
#define SPLIT_COUNTER_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),TYPE_SPLIT_COUNTER,SplitcounterClass))
#define IS_PLUGIN_TEMPLATE(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),TYPE_SPLIT_COUNTER))
#define IS_PLUGIN_TEMPLATE_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),TYPE_SPLIT_COUNTER))

typedef struct _Splitcounter      Splitcounter;
typedef struct _SplitcounterClass SplitcounterClass;

struct _Splitcounter
{
  GstElement element;

  GstPad *sinkpad, *srcpad;

  /* properties */
  char *filename;
};

struct _SplitcounterClass 
{
  GstElementClass parent_class;
};

GType split_counter_get_type (void);



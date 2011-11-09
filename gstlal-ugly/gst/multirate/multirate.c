/* GStreamer
 * Copyright (C) 2007 David Schleef <leo.singer@ligo.org>
 * Copyright (C) 2007 David Schleef <ds@schleef.org>
 *
 * Modified from gstapp.c from gst-plugins-base
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
/**
 * SECTION:element-multiratefirdecim
 *
 * Apply an FIR decimation filter to a stream using a direct form polyphase
 * implementation.
 */
/**
 * SECTION:element-multiratefirinterp
 *
 * Apply an FIR interpolation filter to a stream using a direct form polyphase
 * implementation.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gst/gst.h>

#include "multiratefirdecim.h"
#include "multiratefirinterp.h"

static gboolean
plugin_init (GstPlugin * plugin)
{
  return (gst_element_register (plugin, "multiratefirdecim", GST_RANK_NONE,
          GST_TYPE_MULTIRATE_FIR_DECIM) &&
      gst_element_register (plugin, "multiratefirinterp", GST_RANK_NONE,
          GST_TYPE_MULTIRATE_FIR_INTERP));
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    "multirate",
    "Multirate filters",
    plugin_init, VERSION, GST_LICENSE, GST_PACKAGE_NAME, GST_PACKAGE_ORIGIN)

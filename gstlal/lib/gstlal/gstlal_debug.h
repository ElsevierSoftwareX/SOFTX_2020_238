/*
 * Various bits of LAL wrapped in gstreamer elements.
 *
 * Copyright (C) 2008--2011  Kipp Cannon
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


#ifndef __GSTLAL_DEBUG_H__
#define __GSTLAL_DEBUG_H__


/**
 * SECTION:gstlal_debug
 * @title: Debugging
 * @include: gstlal/gstlal_debug.h
 * @short_description:  Debugging macros.
 */


#include <glib.h>
#include <gst/gst.h>


G_BEGIN_DECLS


/**
 * GST_TIME_SECONDS_FORMAT:
 *
 * printf() style format specifier for displaying a #GstClockTime in
 * seconds.
 *
 * See also:  #GST_TIME_SECONDS_ARGS
 */


/**
 * GST_TIME_SECONDS_ARGS:
 * @t: #GstClockTime
 *
 * Macro to prepare printf() style parameters from a #GstClockTime for
 * ingestion by the format described by #GST_TIME_SECONDS_FORMAT.  NOTE:
 * this macro evaluates its argument multiple times.
 *
 * See also:  #GST_TIME_SECONDS_FORMAT
 */


/*
 * FIXME:  there is a patch in gnats to add these to gstreamer proper,
 * delete when that happens.  (hence the #ifdef protection)
 */


#ifndef GST_TIME_SECONDS_FORMAT
#define GST_TIME_SECONDS_FORMAT G_GINT64_FORMAT ".%09" G_GUINT64_FORMAT " s"
#define GST_TIME_SECONDS_ARGS(t) (GST_CLOCK_TIME_IS_VALID(t) ? (gint64)((t) / GST_SECOND) : -1), (GST_CLOCK_TIME_IS_VALID(t) ? (t) % GST_SECOND : 0)
#endif /* GST_TIME_SECONDS_FORMAT */


/**
 * GST_BUFFERS_BOUNDARIES_FORMAT:
 *
 * printf() style format specifier for displaying the time and offset
 * intervals spanned by a #GstBuffer.
 *
 * See also:  #GST_BUFFERS_BOUNDARIES_ARGS
 */


/**
 * GST_BUFFER_BOUNDARIES_ARGS:
 * @buf:  #GstBuffer
 *
 * Macro to prepare the printf() style parameters from a #GstBuffer for
 * ingestion by the format described by #GST_BUFFER_BOUNDARIES_FORMAT.
 * NOTE:  this macro evalutes its argument multiple times.
 *
 * See also:  #GST_BUFFER_BOUNDARIES_FORMAT
 */


#ifndef GST_BUFFER_BOUNDARIES_FORMAT
#define GST_BUFFER_BOUNDARIES_FORMAT ".d[%" GST_TIME_SECONDS_FORMAT ", %" GST_TIME_SECONDS_FORMAT ") = offsets [%" G_GUINT64_FORMAT ", %" G_GUINT64_FORMAT ")"
#define GST_BUFFER_BOUNDARIES_ARGS(buf) 0, GST_TIME_SECONDS_ARGS(GST_BUFFER_TIMESTAMP(buf)), GST_TIME_SECONDS_ARGS(GST_BUFFER_TIMESTAMP(buf) + GST_BUFFER_DURATION(buf)), GST_BUFFER_OFFSET(buf), GST_BUFFER_OFFSET_END(buf)
#endif /* GST_BUFFER_BOUNDARIES_FORMAT */


G_END_DECLS


#endif	/* __GSTLAL_DEBUG_H__ */

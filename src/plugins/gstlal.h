/*
 * Various bits of LAL wrapped in gstreamer elements.
 *
 * Copyright (C) 2000,2001,2008  Kipp C. Cannon
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


#ifndef __GSTLAL_H__
#define __GSTLAL_H__


#include <glib.h>
#include <gst/gst.h>


#include <lal/LALDatatypes.h>
#include <lal/Units.h>


G_BEGIN_DECLS


/*
 * Hack to work on ancient CentOS
 */


#ifndef G_PARAM_STATIC_STRINGS
#define G_PARAM_STATIC_STRINGS (G_PARAM_STATIC_NAME | G_PARAM_STATIC_NICK | G_PARAM_STATIC_BLURB)
#endif


/*
 * Custom tags
 */


#define GSTLAL_TAG_INSTRUMENT "instrument"
#define GSTLAL_TAG_CHANNEL_NAME "channel-name"
#define GSTLAL_TAG_UNITS "units"


/*
 * Data
 */


extern GMutex *gstlal_fftw_lock;


/*
 * Function prototypes
 */


char *gstlal_build_full_channel_name(const char *, const char *);
REAL8TimeSeries *gstlal_REAL8TimeSeries_from_buffer(GstBuffer *, const char *, const char *, const char *);
LALUnit gstlal_lalStrainPerADCCount(void);
LALUnit gstlal_lalStrainSquaredPerHertz(void);
REAL8FrequencySeries *gstlal_read_reference_psd(const char *);
REAL8FrequencySeries *gstlal_get_reference_psd(const char *, double, double, size_t);


/*
 * Debugging helpers
 */


#define GST_BUFFER_BOUNDARIES_FMT "[%" G_GUINT64_FORMAT ".%09" G_GUINT64_FORMAT " s -- %" G_GUINT64_FORMAT ".%09" G_GUINT64_FORMAT " s) = samples [%" G_GUINT64_FORMAT " -- %" G_GUINT64_FORMAT ")"
#define GST_BUFFER_BOUNDARIES_ARGS(buf) GST_BUFFER_TIMESTAMP(buf) / 1000000000, GST_BUFFER_TIMESTAMP(buf) % 1000000000, (GST_BUFFER_TIMESTAMP(buf) + GST_BUFFER_DURATION(buf)) / 1000000000, (GST_BUFFER_TIMESTAMP(buf) + GST_BUFFER_DURATION(buf)) % 1000000000, GST_BUFFER_OFFSET(buf), GST_BUFFER_OFFSET_END(buf)


G_END_DECLS


#endif	/* __GSTLAL_H__ */

/*
 * GstLALGPSSystemClock
 *
 * Copyright (C) 2012  Kipp Cannon
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */


#ifndef __GSTLAL_GPS_SYSTEM_CLOCK_H__
#define __GSTLAL_GPS_SYSTEM_CLOCK_H__


/*
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


#include <glib.h>
#include <gst/gst.h>


G_BEGIN_DECLS


/*
 * ============================================================================
 *
 *                                    Type
 *
 * ============================================================================
 */


#define GSTLAL_GPS_SYSTEM_CLOCK_TYPE \
	(gstlal_gps_system_clock_get_type())
#define GSTLAL_GPS_SYSTEM_CLOCK(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_GPS_SYSTEM_CLOCK_TYPE, GstLALGPSSystemClock))
#define GSTLAL_GPS_SYSTEM_CLOCK_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_GPS_SYSTEM_CLOCK_TYPE, GstLALGPSSystemClockClass))
#define GSTLAL_GPS_SYSTEM_CLOCK_GET_CLASS(obj) \
	(G_TYPE_INSTANCE_GET_CLASS((obj), GSTLAL_GPS_SYSTEM_CLOCK_TYPE, GstLALGPSSystemClockClass))
#define GST_IS_LAL_GPS_SYSTEM_CLOCK(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_GPS_SYSTEM_CLOCK_TYPE))
#define GST_IS_LAL_GPS_SYSTEM_CLOCK_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_GPS_SYSTEM_CLOCK_TYPE))


typedef struct _GstLALGPSSystemClockClass GstLALGPSSystemClockClass;
typedef struct _GstLALGPSSystemClock GstLALGPSSystemClock;


struct _GstLALGPSSystemClockClass {
	GstSystemClockClass parent_class;
};


/**
 * GstLALGPSSystemClock
 */


struct _GstLALGPSSystemClock {
	GstSystemClock systemclock;
};


/*
 * ============================================================================
 *
 *                                Exported API
 *
 * ============================================================================
 */


GType gstlal_gps_system_clock_get_type(void);


G_END_DECLS


#endif	/* __GSTLAL_GPS_SYSTEM_CLOCK_H__ */

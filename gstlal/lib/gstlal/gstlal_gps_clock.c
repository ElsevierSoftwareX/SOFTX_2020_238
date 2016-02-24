/*
 * GstLALGPSSystemClock
 *
 * Copyright (C) 2012,2013  Kipp Cannon
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


/**
 * SECTION:gstlal_gps_clock
 * @include: gstlal/gstlal_gps_clock.h
 * @short_description:  GstSystemClock sub-class that reports GPS time.
 *
 * The #GstLALGPSSystemClock class is a sub-class of #GstSystemClock that
 * converts the times reported from Unix time to GPS seconds.
 */


/*
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


#include <glib.h>
#include <gst/gst.h>


#include <lal/Date.h>


#include <gstlal_gps_clock.h>


/*
 * ============================================================================
 *
 *                                Boilerplate
 *
 * ============================================================================
 */


G_DEFINE_TYPE(GstLALGPSSystemClock, gstlal_gps_system_clock, GST_TYPE_SYSTEM_CLOCK);


/*
 * ============================================================================
 *
 *                              GstClock Methods
 *
 * ============================================================================
 */


static GstClockTime get_internal_time(GstClock *clock)
{
	GstClockTime t = GST_CLOCK_CLASS(gstlal_gps_system_clock_parent_class)->get_internal_time(clock);

	return t - (XLAL_EPOCH_UNIX_GPS - XLALGPSLeapSeconds(GST_TIME_AS_SECONDS(t))) * GST_SECOND;
}


/*
 * ============================================================================
 *
 *                              GObject Methods
 *
 * ============================================================================
 */


static void gstlal_gps_system_clock_base_init(gpointer klass)
{
}


static void gstlal_gps_system_clock_class_init(GstLALGPSSystemClockClass *klass)
{
	GstClockClass *clock_class = GST_CLOCK_CLASS(klass);

	clock_class->get_internal_time = GST_DEBUG_FUNCPTR(get_internal_time);
}


static void gstlal_gps_system_clock_init(GstLALGPSSystemClock *object)
{
	/*
	 * Kipp:  I have empirically determined that "REALTIME" means Unix
	 * time
	 */

	g_object_set(object, "clock-type", GST_CLOCK_TYPE_REALTIME, NULL);
}

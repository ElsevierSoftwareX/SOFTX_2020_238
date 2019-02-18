/*
 * Various bits of LAL wrapped in gstreamer elements.
 *
 * Copyright (C) 2012  Kipp Cannon, Chad Hanna
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


/*
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


/*
 * Stuff from GStreamer
 */


#include <gst/gst.h>


/*
 * Stuff from LAL
 */


#include <lal/XLALError.h>


/*
 * Our own stuff
 */


#include <gstlal/gstlal_tags.h>
#include <gstlal_itacac.h>
/*
#include <gstlal_coinc.h>
#include <gstlal_skymap.h>
#include <gstlal_triggergen.h>
#include <gstlal_timeslicechisq.h>
#include <gstlal_blcbc_triggergen.h>
#include <gstlal_itac.h>
*/


/*
 * ============================================================================
 *
 *                             Plugin Entry Point
 *
 * ============================================================================
 */


static gboolean plugin_init(GstPlugin *plugin)
{
	struct {
		const gchar *name;
		GType type;
	} *element, elements[] = {
		/*{"lal_coinc", GSTLAL_COINC_TYPE},
		{"lal_skymap", GSTLAL_SKYMAP_TYPE},
		{"lal_triggergen", GSTLAL_TRIGGERGEN_TYPE},
		{"lal_timeslicechisq", GSTLAL_TIMESLICECHISQUARE_TYPE},
		{"lal_blcbctriggergen", GSTLAL_BLCBC_TRIGGERGEN_TYPE},
		{"lal_itac", GSTLAL_ITAC_TYPE}, */
		{"lal_itacac", GSTLAL_ITACAC_TYPE},
		{NULL, 0},
	};

	/*
	 * Set the LAL error handler.
	 */

	XLALSetSilentErrorHandler();

	/*
	 * Tell GStreamer about the custom tags.
	 */

	gstlal_register_tags();

	/*
	 * Tell GStreamer about the elements.
	 */

	for(element = elements; element->name; element++)
		if(!gst_element_register(plugin, element->name, GST_RANK_SECONDARY, element->type))
			return FALSE;

	/*
	 * Done.
	 */

	return TRUE;
}


/*
 * This is the structure that gst-register looks for.
 */


GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, gstlalinspiral, "Various bits of the LIGO Algorithm Library wrapped in gstreamer elements", plugin_init, PACKAGE_VERSION, "GPL", PACKAGE_NAME, "http://www.lsc-group.phys.uwm.edu/daswg")

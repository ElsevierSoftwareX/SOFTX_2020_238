/*
 * Various bits of LAL wrapped in gstreamer elements.
 *
 * Copyright (C) 2008  Kipp Cannon, Chad Hanna
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


/*
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


#include <gst/gst.h>


#include <gstlal.h>
#include <gstlal_framesrc.h>
#include <gstlal_templatebank.h>
#include <gstlal_multiscope.h>
#include <gstlal_simulation.h>
#include <gstlal_whiten.h>


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
		GType (*type)(void);
	} *element, elements[] = {
		{"lal_framesrc", gstlal_framesrc_get_type},
		{"lal_templatebank", gstlal_templatebank_get_type},
		{"lal_multiscope", gstlal_multiscope_get_type},
		{"lal_simulation", gstlal_simulation_get_type},
		{"lal_whiten", gstlal_whiten_get_type},
		{NULL, NULL},
	};

	/* tell gstreamer about the elements */
	for(element = elements; element->name; element++)
		if(!gst_element_register(plugin, element->name, GST_RANK_NONE, element->type()))
			return FALSE;
	return TRUE;
}


/*
 * This is the structure that gst-register looks for.
 */


GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, "gstlal", "Various bits of the LIGO Algorithm Library wrapped in gstreamer elements", plugin_init, PACKAGE_VERSION, "GPL", PACKAGE_NAME, "http://www.lsc-group.phys.uwm.edu/daswg")

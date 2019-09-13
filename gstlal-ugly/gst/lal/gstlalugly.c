/*
 * Various bits of LAL wrapped in gstreamer elements.
 *
 * Copyright (C) 2008  Kipp Cannon, Chad Hanna
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
#include <gstlal_aggregator.h>
#include <gstlal_iirbank.h>
#include <gstlal_interpolator.h>
#include <gstlal_tdwhiten.h>
/*#include <gstlal_specgram.h>
#include <gstlal_pad.h>
#include <gstlal_trim.h>*/
#include <gstlal_bitvectorgen.h>
#include <audioratefaker.h>
#include <gstlal_latency.h>
#include <gstlaldeglitchfilter.h>


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
		{"lal_aggregator", GSTLAL_AGGREGATOR_TYPE},
		{"lal_iirbank", GSTLAL_IIRBANK_TYPE},
		{"lal_interpolator", GSTLAL_INTERPOLATOR_TYPE},
		{"lal_tdwhiten", GSTLAL_TDWHITEN_TYPE},
		/*{"lal_specgram", GSTLAL_SPECGRAM_TYPE},*/
		/*{"lal_pad", GST_TYPE_LALPAD},
		{"lal_trim", GST_TYPE_LALTRIM},*/
		{"lal_bitvectorgen", GSTLAL_BITVECTORGEN_TYPE},
		{"audioratefaker", GST_TYPE_AUDIO_RATE_FAKER},
		{"lal_latency", GSTLAL_LATENCY_TYPE},
		{"lal_deglitcher", GST_TYPE_LALDEGLITCHFILTER},
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
		if(!gst_element_register(plugin, element->name, GST_RANK_NONE, element->type))
			return FALSE;

	/*
	 * Done.
	 */

	return TRUE;
}


/*
 * This is the structure that gst-register looks for.
 */


GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, gstlalugly, "Various bits of the LIGO Algorithm Library wrapped in gstreamer elements", plugin_init, PACKAGE_VERSION, "GPL", PACKAGE_NAME, "http://www.lsc-group.phys.uwm.edu/daswg")

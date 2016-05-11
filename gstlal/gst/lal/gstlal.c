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
#include <gstlal/gstlal.h>
#include <gstlal_audioundersample.h>
#include <gstlal_autochisq.h>
#include <gstlal_cachesrc.h>
#include <gstlal_drop.h>
#include <gstlal_firbank.h>
#include <gstlal_gate.h>
#include <gstlal_matrixmixer.h>
#include <gstlal_nofakedisconts.h>
#include <gstlal_nxydump.h>
#include <gstlal_peak.h>
#include <gstlal_reblock.h>
#include <gstlal_segmentsrc.h>
#include <gstlal_simulation.h>
#include <gstlal_statevector.h>
#include <gstlal_sumsquares.h>
#include <gstlal_togglecomplex.h>
#include <gstlal_whiten.h>
#include <gstlal_shift.h>


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
		{"lal_audioundersample", GSTLAL_AUDIOUNDERSAMPLE_TYPE},
		{"lal_autochisq", GSTLAL_AUTOCHISQ_TYPE},
		{"lal_cachesrc", GSTLAL_CACHESRC_TYPE},
		{"lal_drop", GSTLAL_DROP_TYPE},
		{"lal_firbank", GSTLAL_FIRBANK_TYPE},
		{"lal_gate", GSTLAL_GATE_TYPE},
		{"lal_matrixmixer", GSTLAL_MATRIXMIXER_TYPE},
		{"lal_nofakedisconts", GSTLAL_NOFAKEDISCONTS_TYPE},
		{"lal_nxydump", GST_TSVENC_TYPE},
		{"lal_peak", GSTLAL_PEAK_TYPE},
		{"lal_reblock", GSTLAL_REBLOCK_TYPE},
		{"lal_segmentsrc", GSTLAL_SEGMENTSRC_TYPE},
		{"lal_shift", GSTLAL_SHIFT_TYPE},
		{"lal_simulation", GSTLAL_SIMULATION_TYPE},
		{"lal_statevector", GSTLAL_STATEVECTOR_TYPE},
		{"lal_sumsquares", GSTLAL_SUMSQUARES_TYPE},
		{"lal_togglecomplex", GSTLAL_TOGGLECOMPLEX_TYPE},
		{"lal_whiten", GSTLAL_WHITEN_TYPE},
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
	 * Load FFTW wisdom
	 */

	gstlal_load_fftw_wisdom();

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


GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, gstlal, "Various bits of the LIGO Algorithm Library wrapped in gstreamer elements", plugin_init, PACKAGE_VERSION, "GPL", PACKAGE_NAME, "http://www.lsc-group.phys.uwm.edu/daswg")

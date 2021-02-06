/*
 * Various bits of LAL wrapped in gstreamer elements.
 *
 * Copyright (C) 2014  Madeline Wade
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
 *				  Preamble
 *
 * ============================================================================
 */


#include <Python.h>


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
#include <gstlal_add_constant.h>
#include <gstlal_wings.h>
#include <gstlal_complexfirbank.h>
#include <gstlal_smoothcalibfactors.h>
#include <gstlal_smoothkappas.h>
#include <gstlal_constantupsample.h>
#include <gstlal_resample.h>
#include <gstlal_logicalundersample.h>
#include <gstlal_demodulate.h>
#include <gstlal_insertgap.h>
#include <gstlal_fccupdate.h>
#include <gstlal_transferfunction.h>
#include <gstlal_trackfrequency.h>
#include <gstlal_adaptivefirfilt.h>
#include <gstlal_dqtukey.h>
#include <gstlal_property.h>
#include <gstlal_typecast.h>
#include <gstlal_matrixsolver.h>
#include <gstlal_sensingtdcfs.h>
#include <gstlal_makediscont.h>
#include <gstlal_randreplace.h>
#include <gstlal_stdev.h>
#include <gstlal_minmax.h>


/*
 * ============================================================================
 *
 *			     Plugin Entry Point
 *
 * ============================================================================
 */


static gboolean plugin_init(GstPlugin *plugin)
{
	struct {
		const gchar *name;
		GType type;
	} *element, elements[] = {
		{"lal_add_constant", GSTLAL_ADD_CONSTANT_TYPE},
		{"lal_wings", GST_TYPE_LALWINGS},
		{"lal_complexfirbank", GSTLAL_COMPLEXFIRBANK_TYPE},
		{"lal_smoothcalibfactors", GSTLAL_SMOOTHCALIBFACTORS_TYPE},
		{"lal_smoothkappas", GSTLAL_SMOOTHKAPPAS_TYPE},
		{"lal_constantupsample", GSTLAL_CONSTANTUPSAMPLE_TYPE},
		{"lal_resample", GSTLAL_RESAMPLE_TYPE},
		{"lal_logicalundersample", GSTLAL_LOGICALUNDERSAMPLE_TYPE},
		{"lal_demodulate", GSTLAL_DEMODULATE_TYPE},
		{"lal_insertgap", GSTLAL_INSERTGAP_TYPE},
		{"lal_fcc_update", GSTLAL_FCC_UPDATE_TYPE},
		{"lal_transferfunction", GSTLAL_TRANSFERFUNCTION_TYPE},
		{"lal_trackfrequency", GSTLAL_TRACKFREQUENCY_TYPE},
		{"lal_adaptivefirfilt", GSTLAL_ADAPTIVEFIRFILT_TYPE},
		{"lal_dqtukey", GSTLAL_DQTUKEY_TYPE},
		{"lal_property", GSTLAL_PROPERTY_TYPE},
		{"lal_typecast", GSTLAL_TYPECAST_TYPE},
		{"lal_matrixsolver", GSTLAL_MATRIXSOLVER_TYPE},
		{"lal_sensingtdcfs", GSTLAL_SENSINGTDCFS_TYPE},
		{"lal_makediscont", GSTLAL_MAKEDISCONT_TYPE},
		{"lal_randreplace", GSTLAL_RANDREPLACE_TYPE},
		{"lal_stdev", GSTLAL_STDEV_TYPE},
		{"lal_minmax", GSTLAL_MINMAX_TYPE},
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


GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, gstlalcalibration, "Various bits of the LIGO Algorithm Library wrapped in gstreamer elements", plugin_init, PACKAGE_VERSION, "GPL", PACKAGE_NAME, "http://www.lsc-group.phys.uwm.edu/daswg")

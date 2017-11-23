/*
 * gpu accelerated elements.
 *
 * Copyright (C) 2013  Qi Chu
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
 * Our own stuff
 */


#include <spiir/spiir.h>
//#include <audioresample/cuda_gstaudioresample.h>
//#include <multidownsample/gstlal_multidownsample.h>
#include <multiratespiir/multiratespiir.h>
#include <postcoh/postcoh.h>
#include <postcoh/postcoh_filesink.h>
#include <cohfar/cohfar_accumbackground.h>
#include <cohfar/cohfar_assignfar.h>


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
		{"cuda_iirbank", CUDA_IIRBANK_TYPE},
//		{"cuda_audioresample", CUDA_AUDIO_RESAMPLE_TYPE},
//		{"gstlal_multidownsample", GSTLAL_MULTI_DOWNSAMPLE_TYPE},
		{"cuda_multiratespiir", CUDA_TYPE_MULTIRATE_SPIIR},
		{"cuda_postcoh", CUDA_TYPE_POSTCOH},
		{"postcoh_filesink", POSTCOH_TYPE_FILESINK},
		{"cohfar_accumbackground", COHFAR_ACCUMBACKGROUND_TYPE},
		{"cohfar_assignfar", COHFAR_ASSIGNFAR_TYPE},
		{NULL, 0},
	};


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


GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, cuda, "gpu accelerated elements", plugin_init, PACKAGE_VERSION, "GPL", PACKAGE_NAME, "http://www.lsc-group.phys.uwm.edu/daswg")

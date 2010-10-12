/*
 * Copyright (C) 2010 Leo Singer
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
 * Stuff from the C library
 */


#include <math.h>
#include <stdio.h>


/*
 * Stuff from GStreamer
 */


#include <gst/gst.h>


/*
 * Our own stuff
 */


#include <cairovis_base.h>
#include <cairovis_lineseries.h>
#include <cairovis_waterfall.h>


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
		{"cairovis_base", CAIROVIS_BASE_TYPE},
		{"cairovis_lineseries", CAIROVIS_LINESERIES_TYPE},
		{"cairovis_waterfall", CAIROVIS_WATERFALL_TYPE},
		{NULL, 0},
	};
	struct {
		const gchar *name;
		GstTagFlag flag;
		GType type;
		const gchar *nick;
		const gchar *blurb;
		GstTagMergeFunc func;
	} *tagarg, tagargs[] = {
		{NULL,},
	};

	/*
	 * Tell GStreamer about the elements.
	 */

	for(element = elements; element->name; element++)
		if(!gst_element_register(plugin, element->name, GST_RANK_NONE, element->type))
			return FALSE;

	/*
	 * Tell GStreamer about the custom tags.
	 */

	for(tagarg = tagargs; tagarg->name; tagarg++)
		gst_tag_register(tagarg->name, tagarg->flag, tagarg->type, tagarg->nick, tagarg->blurb, tagarg->func);

	/*
	 * Done.
	 */

	return TRUE;
}


/*
 * This is the structure that gst-register looks for.
 */


GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, "cairovis", "Cairo visualization elements", plugin_init, PACKAGE_VERSION, "GPL", PACKAGE_NAME, "http://www.lsc-group.phys.uwm.edu/daswg")

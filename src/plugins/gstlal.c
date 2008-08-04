/*
 * Various bits of LAL wrapped in gstreamer elements.
 *
 * Copyright (C) 2008  Kipp C. Cannon
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


/*
 * ============================================================================
 *
 *                                    Data
 *
 * ============================================================================
 */


/*
 * types is a 0-terminated array of type constants.  e.g.,
 *
 * caps = gstlal_get_template_caps({G_TYPE_INT, 0});
 */


GstCaps *gstlal_get_template_caps(const GType *types)
{
	GstCaps *caps = gst_caps_new_empty();

	/*
	 * 1073741824 = 2^30, the highest power of two that won't roll-over
	 * a 32-bit signed integer
	 */

	while(*types) {
		switch(*types++) {
		case G_TYPE_FLOAT:
			gst_caps_append(caps, gst_caps_new_simple(
				"audio/x-raw-float",
				"rate", GST_TYPE_INT_RANGE, 1, 1073741824,
				"channels", G_TYPE_INT, 1,
				"endianness", G_TYPE_INT, G_BYTE_ORDER,
				"width", G_TYPE_INT, 32,
				NULL
			));
			break;

		case G_TYPE_DOUBLE:
			gst_caps_append(caps, gst_caps_new_simple(
				"audio/x-raw-float",
				"rate", GST_TYPE_INT_RANGE, 1, 1073741824,
				"channels", G_TYPE_INT, 1,
				"endianness", G_TYPE_INT, G_BYTE_ORDER,
				"width", G_TYPE_INT, 64,
				NULL
			));
			break;

		case G_TYPE_INT:
			gst_caps_append(caps, gst_caps_new_simple(
				"audio/x-raw-int",
				"rate", GST_TYPE_INT_RANGE, 1, 1073741824,
				"channels", G_TYPE_INT, 1,
				"endianness", G_TYPE_INT, G_BYTE_ORDER,
				"width", G_TYPE_INT, 32,
				"depth", G_TYPE_INT, 32,
				"signed", G_TYPE_BOOLEAN, TRUE,
				NULL
			));
			break;
		}
	}

	return caps;
}


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

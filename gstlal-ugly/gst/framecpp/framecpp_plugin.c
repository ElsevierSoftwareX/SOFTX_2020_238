/*
 * framecpp wrapped in gstreamer elements.
 *
 * Copyright (C) 2011  Kipp Cannon, Ed Maros
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


#include <string.h>


/*
 * Stuff from GStreamer
 */


#include <gst/gst.h>


/*
 * Our own stuff
 */


#include <gstlal/gstlal_tags.h>
#include <framecpp_channeldemux.h>
#include <framecpp_channelmux.h>
#include <framecpp_filesink.h>
#if HAVE_GST_BASEPARSE
#include <framecpp_igwdparse.h>
#endif /* HAVE_GST_BASEPARSE */


/*
 * ============================================================================
 *
 *                              TypeFind Support
 *
 * ============================================================================
 */


/*
 * See LIGO-T970130 section 4.3.1, "File Header".
 *
 * The test used to identify a frame file is to extract the 40 byte header,
 * and then require:
 *
 * - bytes 0 -- 4 to be {'I', 'G', 'W', 'D', '\0'},
 *
 * If test passes, we claim it's an IGWD frame file with 100% certainty.
 * The probability that 40 randomly-select bits would equal this sequence
 * is 1 in 1099511627776.  The probability that 32 null-terminated bits
 * would equal this sequence is 1 in 4294967296.  The probability that 4
 * null-terminated upper-case ASCII letters would equal this sequence is 1
 * in 456976.
 */


static void typefind(GstTypeFind *find, gpointer data)
{
	guint8 *header = gst_type_find_peek(find, 0, 40);

	if(!header)
		GST_DEBUG("unable to retrieve 40 byte header");
	else if(memcmp(header, "IGWD", 5))
		GST_DEBUG("bytes 0--4 are not {'I', 'G', 'W', 'D', '\\0'}");
	else
		gst_type_find_suggest(find, GST_TYPE_FIND_MAXIMUM, gst_caps_new_simple(
			"application/x-igwd-frame",
			"framed", G_TYPE_BOOLEAN, FALSE,
			NULL
		));
}


static gboolean register_typefind(GstPlugin *plugin)
{
	static gchar *extensions[] = {"gwf", NULL};

	return gst_type_find_register(
		plugin,
		"framecpp_typefind",
		GST_RANK_PRIMARY,
		typefind,
		extensions,
		gst_caps_from_string(
			"application/x-igwd-frame, " \
			"framed = (boolean) false"
		),
		NULL,
		NULL
	);
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
		GstRank rank;
		GType type;
	} *element, elements[] = {
		{"framecpp_channeldemux", GST_RANK_SECONDARY, FRAMECPP_CHANNELDEMUX_TYPE},
		{"framecpp_channelmux", GST_RANK_SECONDARY, FRAMECPP_CHANNELMUX_TYPE},
        {"framecpp_filesink", GST_RANK_SECONDARY, FRAMECPP_FILESINK_TYPE},
#if HAVE_GST_BASEPARSE
		{"framecpp_igwdparse", GST_RANK_SECONDARY, FRAMECPP_IGWDPARSE_TYPE},
#endif
		{NULL, 0, 0},
	};

	/*
	 * Register tags.
	 */

	gstlal_register_tags();

	/*
	 * Tell GStreamer about the elements.
	 */

	for(element = elements; element->name; element++)
		if(!gst_element_register(plugin, element->name, element->rank, element->type))
			return FALSE;

	/*
	 * Regsister type finder.
	 */

	if(!register_typefind(plugin))
		return FALSE;

	/*
	 * Done.
	 */

	return TRUE;
}


/*
 * This is the structure that gst-register looks for.
 */


GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, "framecpp", "framecpp wrapped in gstreamer elements", plugin_init, PACKAGE_VERSION, "GPL", PACKAGE_NAME, "http://www.lsc-group.phys.uwm.edu/daswg")

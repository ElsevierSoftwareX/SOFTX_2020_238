/*
 * Copyright (C) 2008  Kipp Cannon, Chad Hanna
 * Copyrigth (C) 2010  Leo Singer
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
 * Adapted from gst-plugins-base/gst-libs/gst/tag/tags.c
 */


#include "gstlal_tags.h"


static gpointer register_tags(gpointer unused)
{
	gst_tag_register(GSTLAL_TAG_INSTRUMENT, GST_TAG_FLAG_META, G_TYPE_STRING, "instrument", "The short name of the instrument or observatory where these data were recorded, e.g., \"H1\"", gst_tag_merge_strings_with_comma);
	gst_tag_register(GSTLAL_TAG_CHANNEL_NAME, GST_TAG_FLAG_META, G_TYPE_STRING, "channel name", "The name of this channel, e.g., \"LSC-STRAIN\"", gst_tag_merge_strings_with_comma);
	gst_tag_register(GSTLAL_TAG_UNITS, GST_TAG_FLAG_META, G_TYPE_STRING, "units", "The units for this channel (as encoded by LAL), e.g., \"strain\".", NULL);
	return NULL;
}


void gstlal_register_tags(void)
{
	static GOnce mb_once = G_ONCE_INIT;

	g_once(&mb_once, register_tags, NULL);
}

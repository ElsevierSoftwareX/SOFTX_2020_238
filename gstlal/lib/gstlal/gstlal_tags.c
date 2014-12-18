/*
 * Copyright (C) 2008,2011,2013,2014  Kipp Cannon, Chad Hanna
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


/**
 * SECTION:gstlal_tags
 * @title: Tags
 * @include: gstlal/gstlal_tags.h
 * @short_description:  Extra tags to help describe gravitational-wave data
 * streams.
 */


/*
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


#include "gstlal_tags.h"


/*
 * ============================================================================
 *
 *                               Internal Code
 *
 * ============================================================================
 */


static gpointer register_tags(gpointer unused)
{
	gst_tag_register(GSTLAL_TAG_INSTRUMENT, GST_TAG_FLAG_META, G_TYPE_STRING, "instrument", "The short name of the instrument or observatory where these data were recorded, e.g., \"H1\"", gst_tag_merge_strings_with_comma);
	gst_tag_register(GSTLAL_TAG_CHANNEL_NAME, GST_TAG_FLAG_META, G_TYPE_STRING, "channel name", "The name of this channel, e.g., \"LSC-STRAIN\"", gst_tag_merge_strings_with_comma);
	gst_tag_register(GSTLAL_TAG_BIAS, GST_TAG_FLAG_META, G_TYPE_FLOAT, "bias", "DC bias on channel:  units @ count = 0.", NULL);
	gst_tag_register(GSTLAL_TAG_SLOPE, GST_TAG_FLAG_META, G_TYPE_FLOAT, "slope", "ADC calibration:  units/count.", NULL);
	gst_tag_register(GSTLAL_TAG_UNITS, GST_TAG_FLAG_META, G_TYPE_STRING, "units", "The units for this channel (as encoded by LAL), e.g., \"strain\".", NULL);
	return NULL;
}


/*
 * ============================================================================
 *
 *                                Exported API
 *
 * ============================================================================
 */


/**
 * gstlal_register_tags:
 *
 * Register the tags with the GStreamer tags system.  This function must be
 * invoked at least once before the tags can be used.  It is OK to call
 * this function more than once.  The gstlal plugin calls this function
 * when loaded, so applications using GSTLAL elements need not invoke this
 * function.
 */


void gstlal_register_tags(void)
{
	static GOnce mb_once = G_ONCE_INIT;

	g_once(&mb_once, register_tags, NULL);
}

/*
 * Copyright (C) 2016 Kipp Cannon
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
 * Adapted from gst-plugins-base/gst-libs/gst/audio/audio-info.c
 */


/**
 * SECTION:gstlal_audio_info
 * @title: Audio info with support for complex data
 * @include: gstlal/gstlal_audio_info.h
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


#include <string.h>


#include <gst/gst.h>
#include <gst/audio/audio.h>


#include "gstlal_audio_info.h"


/*
 * ============================================================================
 *
 *                               Internal Code
 *
 * ============================================================================
 */


/*
 * static GstAudioFormatInfo structures for Z64 and Z128 native endianness
 */


static struct _GstAudioFormatInfo formatinfo[] = {
	{
		.format = GST_AUDIO_FORMAT_Z64,
		.name = GST_AUDIO_NE(Z64),
		.description = "single-precision complex float audio",
		.flags = GST_AUDIO_FORMAT_FLAG_FLOAT | GST_AUDIO_FORMAT_FLAG_COMPLEX,
		.endianness = G_BYTE_ORDER,
		.width = 64,
		.depth = 64,
		.silence = {0, 0, 0, 0, 0, 0, 0, 0},
		.unpack_format = GST_AUDIO_FORMAT_Z64,
		.unpack_func = NULL,
		.pack_func = NULL,
	},
	{
		.format = GST_AUDIO_FORMAT_Z128,
		.name = GST_AUDIO_NE(Z128),
		.description = "double-precision complex float audio",
		.flags = GST_AUDIO_FORMAT_FLAG_FLOAT | GST_AUDIO_FORMAT_FLAG_COMPLEX,
		.endianness = G_BYTE_ORDER,
		.width = 128,
		.depth = 128,
		.silence = {0, 0, 0, 0, 0, 0, 0, 0},	/* NOTE:  not big enough */
		.unpack_format = GST_AUDIO_FORMAT_Z128,
		.unpack_func = NULL,
		.pack_func = NULL,
	}
};


/*
 * ============================================================================
 *
 *                                Exported API
 *
 * ============================================================================
 */


/**
 * gstlal_audio_info_from_caps:
 *
 * Wraps gst_audio_info_from_caps() adding support for complex-valued
 * floating point formats (Z64 AND Z128).
 */


gboolean
gstlal_audio_info_from_caps (GstAudioInfo *info, const GstCaps *caps)
{
	GstStructure *str;
	const gchar *s;

	/*
	 * first see if it's a complex-valued time series
	 */

	if((str = gst_caps_get_structure(caps, 0)) && gst_structure_has_name(str, "audio/x-raw") && (s = gst_structure_get_string(str, "format")) && s[0] == 'Z') {
		/*
		 * it looks like it might be.  pretend it's
		 * native-endianness double-precision floats and try
		 * parsing the caps again
		 */

		GstCaps *pretend = gst_caps_copy(caps);
		gboolean success;
		gst_structure_set(gst_caps_get_structure(pretend, 0), "format", G_TYPE_STRING, GST_AUDIO_NE(F64), NULL);
		success = gst_audio_info_from_caps(info, pretend);
		gst_caps_unref(pretend);

		/*
		 * if that didn't work, this isn't a valid format
		 */

		if(!success)
			return success;

		/*
		 * if it did, we need to make a few adjustments to the info
		 * structure
		 */

		if(!strcmp(s, GST_AUDIO_NE(Z64))) {
			info->finfo = &formatinfo[0];
			return success;
		} else if(!strcmp(s, GST_AUDIO_NE(Z128))) {
			info->finfo = &formatinfo[1];
			info->bpf *= 2;
			return success;
		}

		/* oops, not really a format we understand.  let the stock
		 * implementationt try.  it will surely fail (because we
		 * know the format string starts with a "Z" which it won't
		 * understand) but it will put the appropriate error
		 * message into the debug log for us */
	}

	/*
	 * now let the stock implementation try
	 */

	return gst_audio_info_from_caps(info, caps);
}

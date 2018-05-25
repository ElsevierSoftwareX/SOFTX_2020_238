/*
 * Copyright (C) 2010 Leo Singer
 * Copyright (C) 2016 Aaron Viets
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

/*
 * Start here for explanation of class:
 * https://gstreamer.freedesktop.org/data/doc/gstreamer/head/gstreamer-libs/html/GstBaseTransform.html#GstBaseTransform-struct
 */


#include <gst/audio/audio.h>
#include <cmath_base.h>


/*
 * ============================================================================
 *
 *				Type Support
 *
 * ============================================================================
 */


#define CAPS \
		"audio/x-raw, " \
		"format = (string) {"GST_AUDIO_NE(F32)", "GST_AUDIO_NE(F64)", "GST_AUDIO_NE(Z64)", "GST_AUDIO_NE(Z128)"}, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) [1, MAX], " \
		"layout = (string) {interleaved, non-interleaved}, " \
		"channel-mask = (bitmask) 0"


static void
base_init(gpointer klass)
{
	GstElementClass* gstelement_class = GST_ELEMENT_CLASS(klass);
	GstBaseTransformClass* gstbasetransform_class = GST_BASE_TRANSFORM_CLASS(klass);

	gst_element_class_set_details_simple(gstelement_class,
		"CMath operation base class",
		"Filter/Audio",
		"Base class for elements that provide complex arithmetic operations",
		"Aaron Viets <aaron.viets@ligo.org>, Leo Singer <leo.singer@ligo.org>");

	gstbasetransform_class -> set_caps = set_caps;

	static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE("sink",
		GST_PAD_SINK,
		GST_PAD_ALWAYS,
		GST_STATIC_CAPS(CAPS)
	);

	static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE("src",
		GST_PAD_SRC,
		GST_PAD_ALWAYS,
		GST_STATIC_CAPS(CAPS)
	);

	gstbasetransform_class -> set_caps = set_caps;

	gst_element_class_add_pad_template(gstelement_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(gstelement_class, gst_static_pad_template_get(&sink_factory));
}

GType
cmath_base_get_type(void)
{
	static GType type = 0;

	if(!type) {
		static const GTypeInfo info = {
			.class_size = sizeof(CMathBaseClass),
			.base_init = base_init,
			.instance_size = sizeof(CMathBase),
		};
		type =
		g_type_register_static(GST_TYPE_BASE_TRANSFORM, "CMathBase", &info, 0);
	}

	return type;
}


/*
 * ============================================================================
 *
 *				Caps Negotiation
 *
 * ============================================================================
 */

/*
 * set_caps() is called when the caps are re-negotiated. Can return
 * false if we do not like what we see.
 */

gboolean set_caps(GstBaseTransform *trans, GstCaps *incaps,
		GstCaps *outcaps)
{
	gboolean success = TRUE;
	static char *formats[] = {GST_AUDIO_NE(Z128), GST_AUDIO_NE(Z64), GST_AUDIO_NE(F64), GST_AUDIO_NE(F32)};
	static int is_complex[] = {1, 1, 0, 0};
	static int bits[] = {128, 64, 64, 32};
	const gchar *format;
	const gchar *interleave;
	gint channels;
	gint rate;

	CMathBase *element = CMATH_BASE(trans);

	/* Check the incaps to see if it contains e.g. Z128, etc. */
	GstStructure *str = gst_caps_get_structure(incaps, 0);
	g_assert(str);

	if(gst_structure_has_field(str, "format")) {
		format = gst_structure_get_string(str, "format");
	} else {
		GST_ERROR_OBJECT(trans, "No format! Cannot set element caps.\n");
		return 0;
	}

	guint i;
	for(i = 0; i < sizeof(formats) / sizeof(*formats); i++) {
		if(!strcmp(format, formats[i])) {
			element->is_complex = is_complex[i];
			element->bits = bits[i];
			goto found_format;
		}
	}

	GST_ERROR_OBJECT(trans, "Could not find format [%s].\n", format);
	element->is_complex = -1;
	element->bits = -1;
	element->channels = -1;
	element->interleave = NULL;
	return 0;

	found_format:

	if(gst_structure_has_field(str, "layout")) {
		interleave = gst_structure_get_string(str, "layout");
	} else {
		GST_ERROR_OBJECT(trans, "No layout! Cannot set element caps.\n");
		return 0;
	}

	element->interleave = interleave;

	if(gst_structure_has_field(str, "channels")) {
		gst_structure_get_int(str, "channels", &channels);
	} else {
		GST_ERROR_OBJECT(trans, "No channels! Cannot set element caps.\n");
		return 0;
	}

	element->channels = channels;

	if(gst_structure_has_field(str, "rate")) {
		gst_structure_get_int(str, "rate", &rate);
	} else {
		GST_ERROR_OBJECT(trans, "No rate! Cannot set element caps.\n");
		return 0;
	}

	element->rate = rate;

	return success;
}

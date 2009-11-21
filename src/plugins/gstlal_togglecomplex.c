/*
 * Copyright (C) 2009 Kipp Cannon <kipp.cannon@ligo.org>
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * Alternatively, the contents of this file may be used under the GNU
 * Lesser General Public License Version 2.1 (the "LGPL"), in which case
 * the following provisions apply instead of the ones mentioned above:
 *
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Library General Public License as published
 * by the Free Software Foundation; either version 2 of the License, or (at
 * your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307,
 * USA.
 */


/*
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


/*
 *  stuff from the C library
 */


#include <string.h>


/*
 *  stuff from gobject/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include "gstlal_togglecomplex.h"


/*
 * ============================================================================
 *
 *                           GStreamer Boiler Plate
 *
 * ============================================================================
 */


static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE(
	"sink",
	GST_PAD_SINK,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		"audio/x-raw-float, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) [1, MAX], " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) {32, 64}; " \
		"audio/x-raw-complex, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) [1, MAX], " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) {64, 128}"
	)
);


static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE(
	"src",
	GST_PAD_SRC,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		"audio/x-raw-float, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) [1, MAX], " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) {32, 64}; " \
		"audio/x-raw-complex, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) [1, MAX], " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) {64, 128}"
	)
);


GST_BOILERPLATE(
	GSTLALToggleComplex,
	gstlal_togglecomplex,
	GstBaseTransform,
	GST_TYPE_BASE_TRANSFORM
);


/*
 * ============================================================================
 *
 *                     GstBaseTransform Method Overrides
 *
 * ============================================================================
 */


/*
 * get_unit_size()
 */


static gboolean get_unit_size(GstBaseTransform *trans, GstCaps *caps, guint *size)
{
	GstStructure *str;
	gint channels, width;

	str = gst_caps_get_structure(caps, 0);
	if(!gst_structure_get_int(str, "channels", &channels)) {
		GST_DEBUG_OBJECT(trans, "unable to parse channels from %" GST_PTR_FORMAT, caps);
		return FALSE;
	}
	if(!gst_structure_get_int(str, "width", &width)) {
		GST_DEBUG_OBJECT(trans, "unable to parse width from %" GST_PTR_FORMAT, caps);
		return FALSE;
	}

	*size = width / 8 * channels;

	return TRUE;
}


/*
 * set_caps()
 */


static gboolean set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps)
{
	GSTLALToggleComplex *element = GSTLAL_TOGGLECOMPLEX(trans);

	if(element->incaps)
		gst_caps_unref(element->incaps);
	element->incaps = incaps;
	gst_caps_ref(element->incaps);
	if(element->outcaps)
		gst_caps_unref(element->outcaps);
	element->outcaps = outcaps;
	gst_caps_ref(element->outcaps);

	return TRUE;
}


/*
 * transform_caps()
 */


static GstCaps *transform_caps(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps)
{
	guint n;

	caps = gst_caps_copy(caps);

	switch(direction) {
	case GST_PAD_SRC:
	case GST_PAD_SINK:
		for(n = 0; n < gst_caps_get_size(caps); n++) {
			GstStructure *str = gst_caps_get_structure(caps, n);
			const gchar *name;
			gint channels;
			gint width;
			if(!gst_structure_get_int(str, "channels", &channels)) {
				GST_DEBUG_OBJECT(trans, "unable to parse channels from %" GST_PTR_FORMAT, caps);
				goto error;
			}
			if(!gst_structure_get_int(str, "width", &width)) {
				GST_DEBUG_OBJECT(trans, "unable to parse width from %" GST_PTR_FORMAT, caps);
				goto error;
			}
			name = gst_structure_get_name(str);
			if(name && !strcmp(name, "audio/x-raw-float")) {
				gst_structure_set_name(str, "audio/x-raw-complex");
				gst_structure_set(str, "channels", G_TYPE_INT, channels / 2, NULL);
				gst_structure_set(str, "width", G_TYPE_INT, width * 2, NULL);
			} else if(name && !strcmp(name, "audio/x-raw-complex")) {
				gst_structure_set_name(str, "audio/x-raw-float");
				gst_structure_set(str, "channels", G_TYPE_INT, channels * 2, NULL);
				gst_structure_set(str, "width", G_TYPE_INT, width / 2, NULL);
			} else {
				GST_DEBUG_OBJECT(trans, "unrecognized format %s in %" GST_PTR_FORMAT, name ? name : "(NULL)", caps);
				goto error;
			}
		}
		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		goto error;
	}

	return caps;

error:
	gst_caps_unref(caps);
	return NULL;
}


/*
 * transform()
 */


static GstFlowReturn transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf)
{
	/*
	 * no-op
	 */

	return GST_FLOW_OK;
}


/*
 * prepare_output_buffer()
 */


GstFlowReturn prepare_output_buffer(GstBaseTransform *trans, GstBuffer *input, gint size, GstCaps *caps, GstBuffer **buf)
{
	/*
	 * create sub-buffer from input with writeable metadata
	 */

	gst_buffer_ref(input);
	*buf = gst_buffer_make_metadata_writable(input);
	if(!*buf) {
		GST_DEBUG_OBJECT(trans, "failure creating sub-buffer from input");
		return GST_FLOW_ERROR;
	}

	/*
	 * replace caps with output caps
	 */

	gst_buffer_set_caps(*buf, caps);

	/*
	 * done
	 */

	return GST_FLOW_OK;
}


/*
 * ============================================================================
 *
 *                          GObject Method Overrides
 *
 * ============================================================================
 */


/*
 * finalize()
 */


static void finalize(GObject *object)
{
	GSTLALToggleComplex *element = GSTLAL_TOGGLECOMPLEX(object);

	if(element->incaps)
		gst_caps_unref(element->incaps);
	if(element->outcaps)
		gst_caps_unref(element->outcaps);

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


/*
 * base_init()
 */


static void gstlal_togglecomplex_base_init(gpointer gclass)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(gclass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(gclass);

	gst_element_class_set_details_simple(element_class, "Toggle Complex", "Filter/Audio", "Replace float caps with complex (with half the channels), complex with float (with twice the channels).", "Kipp Cannon <kipp.cannon@ligo.org>");

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));

	transform_class->get_unit_size = GST_DEBUG_FUNCPTR(get_unit_size);
	transform_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
	transform_class->transform = GST_DEBUG_FUNCPTR(transform);
	transform_class->prepare_output_buffer = GST_DEBUG_FUNCPTR(prepare_output_buffer);
	transform_class->transform_caps = GST_DEBUG_FUNCPTR(transform_caps);
}


/*
 * class_init()
 */


static void gstlal_togglecomplex_class_init(GSTLALToggleComplexClass *klass)
{
	GObjectClass *gobject_class = (GObjectClass *) klass;

	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);
}


/*
 * init()
 */


static void gstlal_togglecomplex_init(GSTLALToggleComplex *filter, GSTLALToggleComplexClass *kclass)
{
	gst_base_transform_set_in_place(GST_BASE_TRANSFORM(filter), TRUE);
	gst_base_transform_set_qos_enabled(GST_BASE_TRANSFORM(filter), TRUE);
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(filter), TRUE);
}

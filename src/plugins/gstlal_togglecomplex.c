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
		"channels = (int) [2, MAX], " \
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
		"channels = (int) [2, MAX], " \
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
 * transform_caps()
 */


static gint scale_int(gint x, double factor, gint min, gint max)
{
	if(factor >= 1)
		return x < max / factor ? x * factor : max;
	return x > min / factor ? x * factor : min;
}


static GValue *g_value_scale_int(const GValue *src, GValue *dst, double factor)
{
	if(G_VALUE_HOLDS_INT(src)) {
		g_value_init(dst, G_TYPE_INT);
		g_value_set_int(dst, scale_int(g_value_get_int(src), factor, 1, G_MAXINT));
	} else if(GST_VALUE_HOLDS_INT_RANGE(src)) {
		g_value_init(dst, GST_TYPE_INT_RANGE);
		gst_value_set_int_range(dst, scale_int(gst_value_get_int_range_min(src), factor, 1, G_MAXINT), scale_int(gst_value_get_int_range_max(src), factor, 1, G_MAXINT));
	} else if(GST_VALUE_HOLDS_LIST(src)) {
		guint i;
		g_value_init(dst, GST_TYPE_LIST);
		for(i = 0; i < gst_value_list_get_size(src); i++) {
			GValue x = {0};
			gst_value_init_and_copy(&x, gst_value_list_get_value(src, i));
			g_assert(G_VALUE_HOLDS_INT(&x));
			g_value_set_int(&x, scale_int(g_value_get_int(&x), factor, 1, G_MAXINT));
			gst_value_list_append_value(dst, &x);
			g_value_unset(&x);
		}
	} else {
		g_assert_not_reached();
		return NULL;
	}
	return dst;
}


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
			GValue channels = {0};
			GValue width = {0};
			name = gst_structure_get_name(str);
			if(name && !strcmp(name, "audio/x-raw-float")) {
				/* FIXME: should confirm that the channel count is even */
				gst_structure_set_name(str, "audio/x-raw-complex");
				g_value_scale_int(gst_structure_get_value(str, "channels"), &channels, 0.5);
				g_value_scale_int(gst_structure_get_value(str, "width"), &width, 2.0);
			} else if(name && !strcmp(name, "audio/x-raw-complex")) {
				gst_structure_set_name(str, "audio/x-raw-float");
				g_value_scale_int(gst_structure_get_value(str, "channels"), &channels, 2.0);
				g_value_scale_int(gst_structure_get_value(str, "width"), &width, 0.5);
			} else {
				GST_DEBUG_OBJECT(trans, "unrecognized format %s in %" GST_PTR_FORMAT, name ? name : "(NULL)", caps);
				goto error;
			}
			gst_structure_set_value(str, "channels", &channels);
			gst_structure_set_value(str, "width", &width);
			g_value_unset(&channels);
			g_value_unset(&width);
		}
		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		goto error;
	}

	return caps;

error:
	gst_caps_unref(caps);
	return GST_CAPS_NONE;
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
 * base_init()
 */


static void gstlal_togglecomplex_base_init(gpointer gclass)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(gclass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(gclass);

	gst_element_class_set_details_simple(
		element_class,
		"Toggle Complex",
		"Filter/Audio",
		"Replace float caps with complex (with half the channels), complex with float (with twice the channels).",
		"Kipp Cannon <kipp.cannon@ligo.org>"
	);

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));

	transform_class->get_unit_size = GST_DEBUG_FUNCPTR(get_unit_size);
	transform_class->transform = GST_DEBUG_FUNCPTR(transform);
	transform_class->prepare_output_buffer = GST_DEBUG_FUNCPTR(prepare_output_buffer);
	transform_class->transform_caps = GST_DEBUG_FUNCPTR(transform_caps);
}


/*
 * class_init()
 */


static void gstlal_togglecomplex_class_init(GSTLALToggleComplexClass *klass)
{
}


/*
 * init()
 */


static void gstlal_togglecomplex_init(GSTLALToggleComplex *element, GSTLALToggleComplexClass *kclass)
{
	gst_base_transform_set_in_place(GST_BASE_TRANSFORM(element), TRUE);
	gst_base_transform_set_qos_enabled(GST_BASE_TRANSFORM(element), TRUE);
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(element), TRUE);
}

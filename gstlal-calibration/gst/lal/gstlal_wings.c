/*
 * Copyright (C) 2011, 2014 Jordi Burguet-Castell <jordi.burguet-castell@ligo.org>, Madeline Wade <madeline.wade@ligo.org>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */


/**
 * SECTION:element-lal_wings
 *
 * Only lets data between certain offsets pass thru.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch audiotestsrc wave=sine num-buffers=100 ! \
 *   lal_wings initial-offset=10240 final-offset=20480 ! alsasink
 * ]| Only let data between 1024-2048 thru.
 * </refsect2>
 */

static const char gst_lalwings_doc[] =
    "Pass data only inside a region and mark everything else as gaps.\n"
    "\n"
    "The \"offsets\" are media-type specific. For audio buffers, it's the "
    "number of samples produced so far. For video buffers, it's generally "
    "the frame number. For compressed data, it could be the byte offset in "
    "a source or destination file.\n"
    "\n"
    "If \"inverse=true\" is set, only data *outside* of the specified "
    "region will pass, and data in the inside will be marked as gaps.\n"
    "\n"
    "Example launch line:\n"
    "  gst-launch audiotestsrc wave=sine num-buffers=100 ! lal_wings "
    "initial-offset=10240 final-offset=20480 ! alsasink\n"
    "Another example, saving data to disk:\n"
    "  gst-launch audiotestsrc num-buffers=10 ! audio/x-raw-float,"
    "rate=16384,width=64 ! lal_wings initial-offset=1024 final-offset=9216 "
    "inverse=true ! lal_nxydump ! filesink location=borders.txt\n";


#include <math.h>  /* to use round() */

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include "gstlal_wings.h"


/*
 * ============================================================================
 *
 *                                 Transform
 *
 * ============================================================================
 */


/*
 * Where data processing takes place.
 */
static GstFlowReturn transform_ip(GstBaseTransform *trans, GstBuffer *buf)
{
    GstLalwings *elem = GST_LALWINGS(trans);
    guint64 b0, b1, c0, c1;
    gboolean inv = elem->inverse;

    /* Short notation */ 
    if (elem->timestamp) {
        b0 = GST_BUFFER_TIMESTAMP(buf);
        b1 = GST_BUFFER_TIMESTAMP(buf) + GST_BUFFER_DURATION(buf);
        c0 = elem->initial_timestamp;
        c1 = elem->final_timestamp;
    }
    else {
        b0 = GST_BUFFER_OFFSET(buf);
        b1 = GST_BUFFER_OFFSET_END(buf);
        c0 = elem->initial_offset;
        c1 = elem->final_offset;
    }        

    /*                  c0-------c1                   c0-------c1
     *    b0xxxxxxb1                      or                        b0xxxxxxb1
     */
    if (c0 >= b1 || c1 <= b0) { /* buffer completely outside the region */
        if (inv) /* keep */
            return GST_FLOW_OK;
        /* discard */
        gst_buffer_unref(buf);
        return GST_BASE_TRANSFORM_FLOW_DROPPED;
    }

    /*                  c0-------c1
     *                     b0xb1
     */
    if (c0 <= b0 && b1 <= c1) {  /* buffer completely inside the region */
        if (!inv) /* keep */
            return GST_FLOW_OK;
        /* discard */
        gst_buffer_unref(buf);
        return GST_BASE_TRANSFORM_FLOW_DROPPED;
    }

    /*                  c0-------c1
     *              b0xxxxxxb1
     */
    if (b0 <= c0 && b1 <= c1) {
        guint64 num = c0 - b0;
        guint64 den = b1 - b0;
        guint64 byte_offset = gst_util_uint64_scale_int_round (gst_buffer_get_size (buf), num, den);
	guint64 offsets = gst_util_uint64_scale_int_round (GST_BUFFER_OFFSET_END (buf) - GST_BUFFER_OFFSET (buf), num, den);
        GstClockTime duration = gst_util_uint64_scale_int_round (GST_BUFFER_DURATION (buf), num, den);
        if (inv) { /* keep 1st part */
            gst_buffer_resize (buf, 0, byte_offset);
            GST_BUFFER_DURATION (buf) = duration;
            GST_BUFFER_OFFSET_END (buf) = GST_BUFFER_OFFSET (buf) + offsets;
        }
        else {	/* keep 2nd part */
            gst_buffer_resize (buf, byte_offset, gst_buffer_get_size (buf));
            GST_BUFFER_TIMESTAMP (buf) += GST_BUFFER_DURATION (buf) - duration;
            GST_BUFFER_DURATION (buf) = duration;
            GST_BUFFER_OFFSET (buf) = GST_BUFFER_OFFSET_END (buf) - offsets;
        }
        return GST_FLOW_OK;
    }

    /*                  c0-------c1
     *                       b0xxxxxxb1
     */
    if (c0 <= b0 && c1 <= b1) {
        guint64 num = c1 - b0;
        guint64 den = b1 - b0;
        guint64 byte_offset = gst_util_uint64_scale_int_round (gst_buffer_get_size (buf), num, den);
	guint64 offsets = gst_util_uint64_scale_int_round (GST_BUFFER_OFFSET_END (buf) - GST_BUFFER_OFFSET (buf), num, den);
        GstClockTime duration = gst_util_uint64_scale_int_round (GST_BUFFER_DURATION (buf), num, den);
        if (!inv) { /* keep 1st part */
            gst_buffer_resize (buf, 0, byte_offset);
            GST_BUFFER_DURATION (buf) = duration;
            GST_BUFFER_OFFSET_END (buf) = GST_BUFFER_OFFSET (buf) + offsets;
        }
        else {	/* keep 2nd part */
            gst_buffer_resize (buf, byte_offset, gst_buffer_get_size (buf));
            GST_BUFFER_TIMESTAMP (buf) += GST_BUFFER_DURATION (buf) - duration;
            GST_BUFFER_DURATION (buf) = duration;
            GST_BUFFER_OFFSET (buf) = GST_BUFFER_OFFSET_END (buf) - offsets;
        }
        return GST_FLOW_OK;
    }

    /*                  c0-------c1
     *              b0xxxxxxxxxxxxxxxb1
     */
    gsize start_byte_offset = gst_util_uint64_scale_int_round (gst_buffer_get_size (buf), c0 - b0, b1 - b0);
    gsize end_byte_offset = gst_util_uint64_scale_int_round (gst_buffer_get_size (buf), c1 - b0, b1 - b0);

    if (!inv) {	/* keep middle part of buffer */
        gst_buffer_resize (buf, start_byte_offset, end_byte_offset - start_byte_offset);
        GST_BUFFER_TIMESTAMP (buf) += gst_util_uint64_scale_int_round (GST_BUFFER_DURATION (buf), c0 - b0, b1 - b0);
        GST_BUFFER_DURATION (buf) = gst_util_uint64_scale_int_round (GST_BUFFER_DURATION (buf), c1 - c0, b1 - b0);
        /* FIXME: offsets */
        return GST_FLOW_OK;
    }

    /* push 1st part and keep last part for calling code */
    GstBuffer *new = gst_buffer_copy_region (buf, GST_BUFFER_COPY_FLAGS | GST_BUFFER_COPY_MEMORY | GST_BUFFER_COPY_TIMESTAMPS, 0, start_byte_offset);
    GST_BUFFER_DURATION (new) = gst_util_uint64_scale_int_round (GST_BUFFER_DURATION (buf), c0 - b0, b1 - b0);
    GST_BUFFER_OFFSET_END (new) = GST_BUFFER_OFFSET (buf) + gst_util_uint64_scale_int_round (GST_BUFFER_OFFSET_END (buf) - GST_BUFFER_OFFSET (buf), c0 - b0, b1 - b0);
    GstFlowReturn ret = gst_pad_push(GST_BASE_TRANSFORM_SRC_PAD(trans), new);
    if (ret != GST_FLOW_OK)
        return ret;

    gst_buffer_resize (buf, end_byte_offset, gst_buffer_get_size (buf) - end_byte_offset);
    GST_BUFFER_TIMESTAMP (buf) += gst_util_uint64_scale_int_round (GST_BUFFER_DURATION (buf), c1 - b0, b1 - b0);
    GST_BUFFER_OFFSET (buf) += gst_util_uint64_scale_int_round (GST_BUFFER_OFFSET_END (buf) - GST_BUFFER_OFFSET (buf), c1 - b0, b1 - b0);
    return GST_FLOW_OK;
}


/*
 * ============================================================================
 *
 *                                 Properties
 *
 * ============================================================================
 */


enum {
    PROP_0,
    PROP_INITIAL_OFFSET,
    PROP_FINAL_OFFSET,
    PROP_INITIAL_TIMESTAMP,
    PROP_FINAL_TIMESTAMP,
    PROP_INVERSE,
    PROP_TIMESTAMP
};


static void set_property(GObject *object, guint prop_id,
                         const GValue *value, GParamSpec *pspec)
{
    GstLalwings *elem = GST_LALWINGS(object);

    switch (prop_id) {
    case PROP_INITIAL_OFFSET:
        elem->initial_offset = g_value_get_uint64(value);
        break;
    case PROP_FINAL_OFFSET:
        elem->final_offset = g_value_get_uint64(value);
        break;
    case PROP_INITIAL_TIMESTAMP:
        elem->initial_timestamp = g_value_get_uint64(value);
        break;
    case PROP_FINAL_TIMESTAMP:
        elem->final_timestamp = g_value_get_uint64(value);
        break;
    case PROP_INVERSE:
        elem->inverse = g_value_get_boolean(value);
        break;
    case PROP_TIMESTAMP:
        elem->timestamp = g_value_get_boolean(value);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}


static void
get_property(GObject *object, guint prop_id, GValue *value, GParamSpec *pspec)
{
    GstLalwings *elem = GST_LALWINGS(object);

    switch (prop_id) {
    case PROP_INITIAL_OFFSET:
        g_value_set_uint64(value, elem->initial_offset);
        break;
    case PROP_FINAL_OFFSET:
        g_value_set_uint64(value, elem->final_offset);
        break;
    case PROP_INITIAL_TIMESTAMP:
        g_value_set_uint64(value, elem->initial_timestamp);
        break;
    case PROP_FINAL_TIMESTAMP:
        g_value_set_uint64(value, elem->final_timestamp);
        break;
    case PROP_INVERSE:
        g_value_set_boolean(value, elem->inverse);
        break;
    case PROP_TIMESTAMP:
        g_value_set_boolean(value, elem->timestamp);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}


/*
 * ============================================================================
 *
 *                                Type Support
 *
 * ============================================================================
 */


#define GST_CAT_DEFAULT gst_lalwings_debug      // set as default
GST_DEBUG_CATEGORY_STATIC(gst_lalwings_debug);  // define category (statically)


G_DEFINE_TYPE_WITH_CODE(
    GstLalwings,
    gst_lalwings,
    GST_TYPE_BASE_TRANSFORM,
    GST_DEBUG_CATEGORY_INIT(gst_lalwings_debug, "lal_wings", 0,
                            "gstlal wings element")
 );


// See http://library.gnome.org/devel/gstreamer/unstable/gstreamer-GstInfo.html

/*
 * Class init function.
 *
 * Specify properties ("arguments").
 */
static void gst_lalwings_class_init(GstLalwingsClass *klass)
{
    GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
    GstElementClass *gstelement_class = GST_ELEMENT_CLASS(klass);
    GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(klass);

    /* Element description. */
    gst_element_class_set_details_simple(
        gstelement_class,
        "Trim data",
        "Filter-like",
        gst_lalwings_doc,
        "Madeline Wade <madeline.wade@ligo.org>");

    gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
    gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);

    transform_class->transform_ip = GST_DEBUG_FUNCPTR(transform_ip);

    /* Pad description. */
    gst_element_class_add_pad_template(
        gstelement_class,
        gst_pad_template_new(
            GST_BASE_TRANSFORM_SINK_NAME,
            GST_PAD_SINK,
            GST_PAD_ALWAYS,
            GST_CAPS_ANY));

    gst_element_class_add_pad_template(
        gstelement_class,
        gst_pad_template_new(
            GST_BASE_TRANSFORM_SRC_NAME,
            GST_PAD_SRC,
            GST_PAD_ALWAYS,
            GST_CAPS_ANY));

    /* Specify properties. See:
     * http://developer.gnome.org/gobject/unstable/gobject-The-Base-Object-Type.html#g-object-class-install-property
     * and the collection of g_param_spec_*() at
     * http://developer.gnome.org/gobject/unstable/gobject-Standard-Parameter-and-Value-Types.html
     */
    g_object_class_install_property(
        gobject_class, PROP_INITIAL_OFFSET,
        g_param_spec_uint64(
            "initial-offset", "initial offset.",
            "Only let data with offset bigger than this value pass.",
            0, G_MAXUINT64, 0,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property(
        gobject_class, PROP_FINAL_OFFSET,
        g_param_spec_uint64(
            "final-offset", "final offset.",
            "Only let data with offset smaller than this value pass",
            0, G_MAXUINT64, 0,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
    
    g_object_class_install_property(
        gobject_class, PROP_INITIAL_TIMESTAMP,
        g_param_spec_uint64(
            "initial-timestamp", "initial timestamp.",
            "Only let data with timestamp bigger than this value pass",
            0, G_MAXUINT64, 0,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property(
        gobject_class, PROP_FINAL_TIMESTAMP,
        g_param_spec_uint64(
            "final-timestamp", "final timestamp.",
            "Only let data with timestamp smaller than this value pass",
            0, G_MAXUINT64, 0,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property(
        gobject_class, PROP_INVERSE,
        g_param_spec_boolean(
            "inverse", "inverse.",
            "If set only data *outside* the region will pass.",
            FALSE,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property(
        gobject_class, PROP_TIMESTAMP,
        g_param_spec_boolean(
            "timestamp", "timestamp.",
            "If set use timestamps to determine data boundaries.",
            FALSE,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
}


/*
 * Instance init function.
 *
 * Initialize instance. Give default values.
 */
static void gst_lalwings_init(GstLalwings *elem)
{
    /* Properties initial value */
    elem->initial_offset = 0;
    elem->final_offset = 0;
    elem->initial_timestamp = 0;
    elem->final_timestamp = 0;
    elem->inverse = FALSE;
    elem->timestamp = FALSE;
}

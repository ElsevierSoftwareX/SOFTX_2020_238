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
#include "gstlal_wings.h"


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
                         const GValue *value, GParamSpec *pspec);
static void get_property(GObject *object, guint prop_id,
                         GValue *value, GParamSpec *pspec);

static GstFlowReturn chain(GstPad *pad, GstBuffer *buf);


/*
 * ============================================================================
 *
 *                                Type Support
 *
 * ============================================================================
 */

GST_DEBUG_CATEGORY_STATIC(gst_lalwings_debug);  // define category (statically)
#define GST_CAT_DEFAULT gst_lalwings_debug      // set as default

static void additional_initializations(GType type)
{
    GST_DEBUG_CATEGORY_INIT(gst_lalwings_debug, "lal_wings", 0,
                            "gstlal wings element");
}

GST_BOILERPLATE_FULL(GstLalwings, gst_lalwings, GstElement,
                     GST_TYPE_ELEMENT, additional_initializations);

// See http://library.gnome.org/devel/gstreamer/unstable/gstreamer-GstInfo.html

/*
 * Base init function.
 *
 * Element description and pads description.
 */
static void gst_lalwings_base_init(gpointer g_class)
{
    GstElementClass *gstelement_class = GST_ELEMENT_CLASS(g_class);

    /* Element description. */
    gst_element_class_set_details_simple(
        gstelement_class,
        "Trim data",
        "Filter-like",
        gst_lalwings_doc,
        "Madeline Wade <madeline.wade@ligo.org>");

    /* Pad description. */
    gst_element_class_add_pad_template(
        gstelement_class,
        gst_pad_template_new(
            "sink",
            GST_PAD_SINK,
            GST_PAD_ALWAYS,
            gst_caps_from_string("ANY")));

    gst_element_class_add_pad_template(
        gstelement_class,
        gst_pad_template_new(
            "src",
            GST_PAD_SRC,
            GST_PAD_ALWAYS,
            gst_caps_from_string("ANY")));
}


/*
 * Class init function.
 *
 * Specify properties ("arguments").
 */
static void gst_lalwings_class_init(GstLalwingsClass *klass)
{
    GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

    gobject_class->set_property = set_property;
    gobject_class->get_property = get_property;

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
static void gst_lalwings_init(GstLalwings *elem, GstLalwingsClass *g_class)
{
    gst_element_create_all_pads(GST_ELEMENT(elem));

    /* Pad through which data comes in to the element (sink pad) */
    elem->sinkpad = gst_element_get_static_pad(GST_ELEMENT(elem), "sink");
    gst_pad_set_chain_function(elem->sinkpad, chain);

    /* Pad through which data goes out of the element (src pad) */
    elem->srcpad = gst_element_get_static_pad(GST_ELEMENT(elem), "src");

    /* Properties initial value */
    elem->initial_offset = 0;
    elem->final_offset = 0;
    elem->initial_timestamp = 0;
    elem->final_timestamp = 0;
    elem->inverse = FALSE;
    elem->timestamp = FALSE;
}


/*
 * ============================================================================
 *
 *                                 Properties
 *
 * ============================================================================
 */


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
 *                                Chain
 *
 * ============================================================================
 */


/*
 * Auxiliary function used in chain(). Push a subbuffer into pad,
 * constructed from buffer template, that spans from the fractions of
 * the original buffer specified in f0, f1.
 *
 * f0, f1: fractions of the original buffer that are going to be
 * sent. For example, if sending the middle half, f0=0.25, f1=0.75.
 *
 * If gap==TRUE, send subbuffer as a gap.
 *
 * As with gst_pad_push(), in all cases, success or failure, the caller
 * loses its reference to "template" after calling this function.
 */
static GstFlowReturn push_subbuf(GstPad *pad, GstBuffer *template,
                                 gdouble f0, gdouble f1, gboolean gap)
{
    GstFlowReturn result = GST_FLOW_OK;
    GstBuffer *buf;

    if (f1-f0 < 1e-14) {  /* if negative interval, or really small... */
        gst_buffer_unref(template);
        return GST_FLOW_OK;  /* don't bother with them! */
    }

    guint size = GST_BUFFER_SIZE(template);
    guint64 s0 = GST_BUFFER_OFFSET(template);      /* first sample */
    guint64 s1 = GST_BUFFER_OFFSET_END(template);  /* last sample + 1 */
    GstClockTime t = GST_BUFFER_TIMESTAMP(template);
    GstClockTime dt = GST_BUFFER_DURATION(template);

    if (gap) {
        result = gst_pad_alloc_buffer(pad, GST_BUFFER_OFFSET_NONE,
                                      0, GST_PAD_CAPS(pad), &buf);
    }
    else {
        buf = gst_buffer_create_sub(template, (guint) round(f0 * size),
                                    (guint) round((f1-f0) * size));
        result = (buf != NULL) ? GST_FLOW_OK : GST_FLOW_ERROR;
    }

    if (result != GST_FLOW_OK) {
        GST_ERROR("gst_pad_alloc_buffer() failed allocating buffer");
        gst_buffer_unref(template);
        return result;
    }

    gst_buffer_copy_metadata(buf, template,
                             GST_BUFFER_COPY_FLAGS | GST_BUFFER_COPY_CAPS);
    if (gap)
        GST_BUFFER_FLAG_SET(buf, GST_BUFFER_FLAG_GAP);

    GST_BUFFER_OFFSET(buf)     = s0 + (guint64) round(f0 * (s1 - s0));
    GST_BUFFER_OFFSET_END(buf) = s0 + (guint64) round(f1 * (s1 - s0));
    if (GST_BUFFER_TIMESTAMP_IS_VALID(template)) {
        GST_BUFFER_TIMESTAMP(buf) = t + (GstClockTime) round(f0 * dt);
        GST_BUFFER_DURATION(buf) = (GstClockTime) round((f1 - f0) * dt);
    }
    else {
        GST_BUFFER_TIMESTAMP(buf) = GST_CLOCK_TIME_NONE;
        GST_BUFFER_DURATION(buf) = GST_CLOCK_TIME_NONE;
    }

    result = gst_pad_push(pad, buf);
    if (result != GST_FLOW_OK) {
        GST_ERROR("gst_pad_push() failed pushing gap buffer");
        gst_buffer_unref(template);
        return result;
    }

    /* Unref the template buffer, because an element should either
     * unref the buffer or push it out on a src pad using gst_pad_push()
     */
    gst_buffer_unref(template);

    return GST_FLOW_OK;
}


/*
 * Where data processing takes place.
 *
 * http://gstreamer.freedesktop.org/data/doc/gstreamer/head/pwg/html/chapter-building-chainfn.html
 */
static GstFlowReturn chain(GstPad *pad, GstBuffer *buf)
{
    GstLalwings *elem = GST_LALWINGS(GST_OBJECT_PARENT(pad));
    guint64 b0, b1, c0, c1;
    gboolean inv = elem->inverse;
    gboolean ts = elem->timestamp;

    /* Short notation */ 
    if (ts) {
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
    if (b1 < c0 || c1 < b0)  /* buffer out of the region we cut */
        return push_subbuf(elem->srcpad, buf, 0.0, 1.0, TRUE ^ inv);

    /*                  c0-------c1
     *                     b0xb1
     */
    if (c0 <= b0 && b1 <= c1)  /* buffer completely inside the region we cut */
        return push_subbuf(elem->srcpad, buf, 0.0, 1.0, FALSE ^ inv);

    /*                  c0-------c1
     *              b0xxxxxxb1
     */
    if (b0 <= c0 && b1 <= c1) {
        gdouble f = (c0 - b0) / (gdouble) (b1 - b0);
        gst_buffer_ref(buf);
        /* because we will unref() it twice by calling push_subbuf() */

        if (push_subbuf(elem->srcpad, buf, 0.0, f, TRUE ^ inv) != GST_FLOW_OK)
            return GST_FLOW_ERROR;

        return push_subbuf(elem->srcpad, buf, f, 1.0, FALSE ^ inv);
    }

    /*                  c0-------c1
     *                       b0xxxxxxb1
     */
    if (c0 <= b0 && c1 <= b1) {
        gdouble f = (c1 - b0) / (gdouble) (b1 - b0);
        gst_buffer_ref(buf);
        /* because we will unref() it twice by calling push_subbuf() */

        if (push_subbuf(elem->srcpad, buf, 0.0, f, FALSE ^ inv) != GST_FLOW_OK)
            return GST_FLOW_ERROR;

        return push_subbuf(elem->srcpad, buf, f, 1.0, TRUE ^ inv);
    }

    /*                  c0-------c1
     *              b0xxxxxxxxxxxxxxxb1
     */
    gdouble f0 = (c0 - b0) / (gdouble) (b1 - b0);
    gdouble f1 = (c1 - b0) / (gdouble) (b1 - b0);

    gst_buffer_ref(buf);
    /* because we will unref() it *3 times* by calling push_subbuf() */

    if (push_subbuf(elem->srcpad, buf, 0.0, f0, TRUE ^ inv) != GST_FLOW_OK)
        return GST_FLOW_ERROR;

    gst_buffer_ref(buf);  /* still one to go */

    if (push_subbuf(elem->srcpad, buf, f0, f1, FALSE ^ inv) != GST_FLOW_OK)
        return GST_FLOW_ERROR;

    return push_subbuf(elem->srcpad, buf, f1, 1.0, TRUE ^ inv);
}

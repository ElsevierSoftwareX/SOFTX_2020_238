/*
 * Copyright (C) 2011 Jordi Burguet-Castell <jordi.burguet-castell@ligo.org>
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

/*
 * TODO:
 * See function push_gap() and how it is used in gstlal_nxydump.c, to
 * do something similar here.
 *
 * See
 * http://gstreamer.freedesktop.org/data/doc/gstreamer/head/pwg/html/chapter-building-pads.html
 */

/**
 * SECTION:element-lal_trim
 *
 * Only lets data between certain offsets pass thru.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch audiotestsrc wave=sine ! \
 *   lal_trim initial-offset=1024 final-offset=2048 ! alsasink
 * ]| Only let data between 1024-2048 thru.
 * </refsect2>
 */

static const char gst_laltrim_doc[] =
    "Prints the offset values of the buffers that pass thru it.\n"
    "\n"
    "Example launch line:\n"
    "  gst-launch audiotestsrc ! lal_trim "
    "initial_offset=1024 final-offset=2048 ! alsasink\n";


#include <gst/gst.h>
#include "gstlal_trim.h"


enum {
    PROP_0,
    PROP_INITIAL_OFFSET,
    PROP_FINAL_OFFSET,
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

GST_DEBUG_CATEGORY_STATIC(gst_laltrim_debug);  // define category (statically)
#define GST_CAT_DEFAULT gst_laltrim_debug      // set as default

static void additional_initializations(GType type)
{
    GST_DEBUG_CATEGORY_INIT(gst_laltrim_debug, "laltrim", 0,
                            "laltrim element");
}

GST_BOILERPLATE_FULL(GstLaltrim, gst_laltrim, GstElement,
                     GST_TYPE_ELEMENT, additional_initializations);

// See http://library.gnome.org/devel/gstreamer/unstable/gstreamer-GstInfo.html

/*
 * Base init function.
 *
 * Element description and pads description.
 */
static void gst_laltrim_base_init(gpointer g_class)
{
    GstElementClass *gstelement_class = GST_ELEMENT_CLASS(g_class);

    /* Element description. */
    gst_element_class_set_details_simple(
        gstelement_class,
        "Trim data",
        "Filter-like",
        gst_laltrim_doc,
        "Jordi Burguet-Castell <jordi.burguet-castell@ligo.org>");

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
static void gst_laltrim_class_init(GstLaltrimClass *klass)
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
}


/*
 * Instance init function.
 *
 * Initialize instance. Give default values.
 */
static void gst_laltrim_init(GstLaltrim *elem, GstLaltrimClass *g_class)
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
    GstLaltrim *elem = GST_LALTRIM(object);

    switch (prop_id) {
    case PROP_INITIAL_OFFSET:
        elem->initial_offset = g_value_get_uint64(value);
        break;
    case PROP_FINAL_OFFSET:
        elem->final_offset = g_value_get_uint64(value);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}


static void
get_property(GObject *object, guint prop_id, GValue *value, GParamSpec *pspec)
{
    GstLaltrim *elem = GST_LALTRIM(object);

    switch (prop_id) {
    case PROP_INITIAL_OFFSET:
        g_value_set_uint64(value, elem->initial_offset);
        break;
    case PROP_FINAL_OFFSET:
        g_value_set_uint64(value, elem->final_offset);
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
 * Where data processing takes place.
 *
 * http://gstreamer.freedesktop.org/data/doc/gstreamer/head/pwg/html/chapter-building-chainfn.html
 */
static GstFlowReturn chain(GstPad *pad, GstBuffer *buf)
{
    GstLaltrim *elem = GST_LALTRIM(GST_OBJECT_PARENT(pad));
    guint64 t0_buf, t1_buf, t0_cut, t1_cut;

    /* Short notation */
    t0_buf = GST_BUFFER_OFFSET(buf);
    t1_buf = GST_BUFFER_OFFSET_END(buf);
    t0_cut = elem->initial_offset;
    t1_cut = elem->final_offset;


   /* The funny pictures below represent the conditions checked:
     *
     *  __________-------_____    <--- times/offsets to trim
     *  ..xxxxxx..............    <--- times/offsets of the buffer
     *
     */

    /*            -------
     *    xxxxxx
     */
    if (t1_buf < t0_cut) {
        GstBuffer *outbuf;
        outbuf = gst_buffer_create_sub(buf, 0, 0);
        GST_BUFFER_DURATION(outbuf) = 0;
        GST_BUFFER_OFFSET_END(outbuf) = 0;
        GstCaps *caps;
        if ((caps = GST_BUFFER_CAPS(buf)))
            gst_caps_ref(caps);
        GST_BUFFER_CAPS(outbuf) = caps;
        gst_buffer_unref(buf);
        return gst_pad_push(elem->srcpad, outbuf);
    }

    /*            -------
     *                      xxxxxx
     */
    if (t1_cut < t0_buf) {
        gst_element_send_event(GST_ELEMENT(GST_OBJECT_PARENT(elem->sinkpad)),
                               gst_event_new_eos());
        return GST_FLOW_OK;
    }

    /*            -------
     *             xxxxx
     */
    if (t0_cut <= t0_buf && t1_buf <= t1_cut) {
        return gst_pad_push(elem->srcpad, buf);
    }

    /*            -------
     *         xxxxxxxxxxxxx
     */
    if (t0_buf <= t0_cut && t1_cut <= t1_buf) {
        GstBuffer* outbuf;
        guint size = GST_BUFFER_SIZE(buf);
        guint new_size = size * (t1_cut - t0_cut) / (t1_buf - t0_buf);
        printf("size, newsize = %ud %ud\n", size, new_size);
        outbuf = gst_buffer_create_sub(buf, t0_cut - t0_buf, new_size);
        GST_BUFFER_DURATION(outbuf) = new_size;
        GST_BUFFER_OFFSET_END(outbuf) = GST_BUFFER_OFFSET_NONE;
        GstCaps *caps;
        if ((caps = GST_BUFFER_CAPS(buf)))
            gst_caps_ref(caps);
        GST_BUFFER_CAPS(outbuf) = caps;
        gst_buffer_unref(buf);
        return gst_pad_push(elem->srcpad, outbuf);
    }

    /*            -------
     *         xxxxxx
     */
    if (t0_buf <= t0_cut && t1_buf <= t1_cut) {
        GstBuffer* outbuf;
        guint size = GST_BUFFER_SIZE(buf);
        guint new_size = size * (t1_buf - t0_cut) / (t1_buf - t0_buf);
        printf("size, new_size = %ud %ud\n", size, new_size); ///
        outbuf = gst_buffer_create_sub(buf, t0_cut - t0_buf, new_size);
        GST_BUFFER_DURATION(outbuf) = new_size;
        GST_BUFFER_OFFSET_END(outbuf) = GST_BUFFER_OFFSET_END(buf);
        GstCaps *caps;
        if ((caps = GST_BUFFER_CAPS(buf)))
            gst_caps_ref(caps);
        GST_BUFFER_CAPS(outbuf) = caps;
        gst_buffer_unref(buf);
        return gst_pad_push(elem->srcpad, outbuf);
    }

    /*            -------
     *                xxxxxx
     */
    if (t0_cut <= t0_buf && t1_cut <= t1_buf) {
        GstBuffer* outbuf;
        guint size = GST_BUFFER_SIZE(buf);
        guint new_size = size * (t1_cut - t0_buf) / (t1_buf - t0_buf);
        outbuf = gst_buffer_create_sub(buf, 0, new_size);
        GST_BUFFER_DURATION(outbuf) = new_size;
        GST_BUFFER_OFFSET_END(outbuf) = GST_BUFFER_OFFSET_NONE;
        GstCaps *caps;
        if ((caps = GST_BUFFER_CAPS(buf)))
            gst_caps_ref(caps);
        GST_BUFFER_CAPS(outbuf) = caps;
        gst_buffer_unref(buf);
        return gst_pad_push(elem->srcpad, outbuf);
    }


    // Shouldn't get to this point!
    printf("Really?\n");
    return gst_pad_push(elem->srcpad, buf);
}

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
 * TODO: negotiate caps even when we are not passing the first buffer
 * (probably we will have to define setcaps()
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
            gst_caps_from_string(
                "audio/x-raw-int, "                  // int
                "rate = (int) [1, MAX], "
                "channels = (int) 1, "
                "endianness = (int) BYTE_ORDER, "
                "width = (int) {8, 16, 32, 64}, "
                "depth = (int) {8, 16, 32, 64}, "
                "signed = (boolean) {true, false}; "
                "audio/x-raw-float, "                // float, double
                "rate = (int) [1, MAX], "
                "channels = (int) 1, "
                "endianness = (int) BYTE_ORDER, "
                "width = (int) {32, 64}")));

    gst_element_class_add_pad_template(
        gstelement_class,
        gst_pad_template_new(
            "src",
            GST_PAD_SRC,
            GST_PAD_ALWAYS,
            gst_caps_from_string(
                "audio/x-raw-int, "                  // int
                "rate = (int) [1, MAX], "
                "channels = (int) 1, "
                "endianness = (int) BYTE_ORDER, "
                "width = (int) {8, 16, 32, 64}, "
                "depth = (int) {8, 16, 32, 64}, "
                "signed = (boolean) {true, false}; "
                "audio/x-raw-float, "                // float, double
                "rate = (int) [1, MAX], "
                "channels = (int) 1, "
                "endianness = (int) BYTE_ORDER, "
                "width = (int) {32, 64}")));
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
    GstElementClass *base_class = GST_ELEMENT_CLASS(g_class);

    /* Pad through which data comes in to the element */
    elem->sinkpad = gst_pad_new_from_template(
        gst_element_class_get_pad_template(base_class, "sink"), "sink");
    gst_pad_set_chain_function(elem->sinkpad, chain);

    gst_element_add_pad(GST_ELEMENT(elem), elem->sinkpad);

    /* Pad through which data goes out of the element */
    elem->srcpad = gst_pad_new_from_template (
        gst_element_class_get_pad_template(base_class, "src"), "src");

    gst_element_add_pad(GST_ELEMENT(elem), elem->srcpad);

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
    guint64 start, end;
    guint64 trim_start, trim_end;

    start = GST_BUFFER_OFFSET(buf);
    end = GST_BUFFER_OFFSET_END(buf);

    trim_start = elem->initial_offset;
    trim_end   = elem->final_offset;   // so we don't write so much

   /* The funny pictures below represent the conditions checked:
     *
     *  __________-------_____    <--- times/offsets to trim
     *  ..xxxxxx..............    <--- times/offsets of the buffer
     *
     */

    printf("start, end   trim_start, trim_end = %lld, %lld  %lld, %lld\n",
           start, end, trim_start, trim_end);

    /*            -------
     *    xxxxxx
     */
    if (end < trim_start) {
        printf("              -------\n"
               "      xxxxxx\n\n");
        return GST_FLOW_OK;  // don't pass the buffer
    }

    /*            -------
     *                      xxxxxx
     */
    if (trim_end < start) {
        printf("              -------\n"
               "                        xxxxxx\n\n");
        gst_element_send_event(GST_ELEMENT(GST_OBJECT_PARENT(elem->sinkpad)),
                               gst_event_new_eos());
        return GST_FLOW_OK;
    }

    /*            -------
     *             xxxxx
     */
    if (trim_start <= start && end <= trim_end) {
        printf("              -------\n"
               "               xxxxx\n\n");
        return gst_pad_push(elem->srcpad, buf);
    }

    /*            -------
     *         xxxxxxxxxxxxx
     */
    if (start <= trim_start && trim_end <= end) {
        printf("              -------\n"
               "           xxxxxxxxxxxx\n\n");
        GstBuffer* outbuf;
        guint size = GST_BUFFER_SIZE(buf);
        guint new_size = size * (trim_end - trim_start) / (end - start);
        printf("size, newsize = %ud %ud\n", size, new_size);
        outbuf = gst_buffer_create_sub(buf, trim_start - start, new_size);
//        outbuf->duration = buf->duration * (trim_end - trim_start) / (end - start);
//        outbuf->offset = 0;
        gst_buffer_unref(buf);
        return gst_pad_push(elem->srcpad, outbuf);
    }

    /*            -------
     *         xxxxxx
     */
    if (start <= trim_start && end <= trim_end) {
        printf("              -------\n"
               "           xxxxxx\n\n");
        GstBuffer* outbuf;
        guint size = GST_BUFFER_SIZE(buf);
        guint new_size = size * (end - trim_start) / (end - start);
        printf("size, new_size = %ud %ud\n", size, new_size); ///
        outbuf = gst_buffer_create_sub(buf, trim_start - start, new_size);
//        outbuf->duration = buf->duration * (end - trim_start) / (end - start);
//        outbuf->offset = 0;
        gst_buffer_unref(buf);
        return gst_pad_push(elem->srcpad, outbuf);
    }

    /*            -------
     *                xxxxxx
     */
    if (trim_start <= start && trim_end <= end) {
        printf("              -------\n"
               "                  xxxxxx\n\n");
        GstBuffer* outbuf;
        guint size = GST_BUFFER_SIZE(buf);
        guint new_size = size * (trim_end - start) / (end - start);
        outbuf = gst_buffer_create_sub(buf, 0, new_size);
        gst_buffer_unref(buf);
        return gst_pad_push(elem->srcpad, outbuf);
    }


    // Shouldn't get to this point!
    printf("Really?\n");
    return gst_pad_push(elem->srcpad, buf);
}

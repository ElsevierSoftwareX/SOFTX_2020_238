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


/**
 * SECTION:element-lal_pad
 *
 * Pad stream - prepend and append gaps.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch audiotestsrc wave=sine
 *   timestamp-offset=1000000000 num-buffers=100 ! \
 *   lal_pad pre=1024 post=2048 ! alsasink
 * ]|
 * </refsect2>
 */

static const char gst_lalpad_doc[] =
    "Prepend and append (\"pad\") gaps to a stream.\n"
    "\n"
    "The \"pre\" and \"post\" properties can be offsets or durations, "
    "depending on the \"units\" property. By default they are offsets "
    "--number of samples in the case of audio buffers, and frame number "
    "for video buffers.\n"
    "\n"
    "The element works this way:\n"
    "  * When the first buffer arrives it saves the caps and sends a gap (of "
    "size \"pre\") before it.\n"
    "  * When an EOS event arrives, it sends a gap (of size \"post\") using "
    "the saved caps.\n"
    "\n"
    "Example launch line:\n"
    "  gst-launch audiotestsrc wave=sine "
    "timestamp-offset=1000000000 num-buffers=5 ! lal_pad pre=1024 post=2048 ! "
    "alsasink\n";

#include <gst/gst.h>
#include "gstlal_pad.h"

#define A_X_B__C(a, b, c)  gst_util_uint64_scale_int_round(a, b, c)


enum {
    PROP_0,
    PROP_PRE,
    PROP_POST,
    PROP_UNITS,
};


static void dispose(GObject *object);

static void set_property(GObject *object, guint prop_id,
                         const GValue *value, GParamSpec *pspec);
static void get_property(GObject *object, guint prop_id,
                         GValue *value, GParamSpec *pspec);

static gboolean event(GstPad *pad, GstEvent *event);
static GstFlowReturn chain(GstPad *pad, GstBuffer *buf);
static gboolean setcaps(GstPad *pad, GstCaps *caps);


/* Copied from gstaudiotestsrc.c */
#define GST_TYPE_LALPAD_UNIT (gst_lalpad_unit_get_type())
static GType gst_lalpad_unit_get_type()
{
    static GType lalpad_unit_type = 0;
    static const GEnumValue lalpad_units[] = {
        {GST_LALPAD_UNIT_SAMPLES, "Samples", "samples"},
        {GST_LALPAD_UNIT_TIME, "Time in nanoseconds", "nanoseconds"},
        {0, NULL, NULL},
    };

    if (G_UNLIKELY(lalpad_unit_type == 0)) {
        lalpad_unit_type = g_enum_register_static("GstLalpadUnit",
                                                  lalpad_units);
    }
    return lalpad_unit_type;
}


/* Signals */
guint signal_rate_changed;

static void rate_changed(GstElement* elem, gint rate, gpointer data)
{
    // do nothing, but we are necessary for g_signal_new()
}


/*
 * ============================================================================
 *
 *                                Type Support
 *
 * ============================================================================
 */

GST_DEBUG_CATEGORY_STATIC(gst_lalpad_debug);  // define category (statically)
#define GST_CAT_DEFAULT gst_lalpad_debug      // set as default

static void additional_initializations(GType type)
{
    GST_DEBUG_CATEGORY_INIT(gst_lalpad_debug, "lal_pad", 0,
                            "gstlal pad element");
}

GST_BOILERPLATE_FULL(GstLalpad, gst_lalpad, GstElement,
                     GST_TYPE_ELEMENT, additional_initializations);

// See http://library.gnome.org/devel/gstreamer/unstable/gstreamer-GstInfo.html

/*
 * Base init function.
 *
 * Element description and pads description.
 */
static void gst_lalpad_base_init(gpointer g_class)
{
    GstElementClass *gstelement_class = GST_ELEMENT_CLASS(g_class);

    /* Element description. */
    gst_element_class_set_details_simple(
        gstelement_class,
        "Pad data",
        "Filter-like",
        gst_lalpad_doc,
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
static void gst_lalpad_class_init(GstLalpadClass *klass)
{
    GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

    gobject_class->dispose = dispose;

    gobject_class->set_property = set_property;
    gobject_class->get_property = get_property;

    klass->rate_changed = rate_changed;

    /* Specify properties. See:
     * http://developer.gnome.org/gobject/unstable/gobject-The-Base-Object-Type.html#g-object-class-install-property
     * and the collection of g_param_spec_*() at
     * http://developer.gnome.org/gobject/unstable/gobject-Standard-Parameter-and-Value-Types.html
     */
    g_object_class_install_property(
        gobject_class, PROP_PRE,
        g_param_spec_uint64(
            "pre", "prepended padding.",
            "Number of gap samples to prepend to the stream",
            0, G_MAXUINT64, 0,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property(
        gobject_class, PROP_POST,
        g_param_spec_uint64(
            "post", "appended padding.",
            "Number of gap samples to append to the stream",
            0, G_MAXUINT64, 0,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property(
        gobject_class, PROP_UNITS,
        g_param_spec_enum(
            "units", "Units",
            "Units in which the \"pre\" and \"post\" properties are given",
            GST_TYPE_LALPAD_UNIT, GST_LALPAD_UNIT_SAMPLES,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    signal_rate_changed = g_signal_new(
        "rate-changed", G_TYPE_FROM_CLASS(klass), G_SIGNAL_RUN_FIRST,
        G_STRUCT_OFFSET(GstLalpadClass, rate_changed),
        NULL, NULL, g_cclosure_marshal_VOID__INT, G_TYPE_NONE, 1, G_TYPE_INT);
}


/*
 * Instance init function.
 *
 * Initialize instance. Give default values.
 */
static void gst_lalpad_init(GstLalpad *elem, GstLalpadClass *g_class)
{
    gst_element_create_all_pads(GST_ELEMENT(elem));

    /* Pad through which data comes in to the element (sink pad) */
    elem->sinkpad = gst_element_get_static_pad(GST_ELEMENT(elem), "sink");
    gst_pad_set_event_function(elem->sinkpad, event);
    gst_pad_set_chain_function(elem->sinkpad, chain);
    gst_pad_set_setcaps_function(elem->sinkpad, setcaps);

    /* Pad through which data goes out of the element (src pad) */
    elem->srcpad = gst_element_get_static_pad(GST_ELEMENT(elem), "src");

    /* Properties initial value */
    elem->pre = 0;
    elem->post = 0;
    elem->unit = GST_LALPAD_UNIT_SAMPLES;

    /* Info about the buffers being processed */
    elem->saved_offset = 0;
    elem->saved_offset_end = 0;
    elem->saved_timestamp = 0;
    elem->saved_duration = 0;
    elem->first_buffer = TRUE;
    elem->caps = NULL;
    elem->rate = 0;
}


/*
 * Free resources.
 */
static void dispose(GObject *object)
{
    GstLalpad *elem = GST_LALPAD(object);

    if (elem->caps != NULL) {
        gst_caps_unref(elem->caps);
        elem->caps = NULL;
    }

    G_OBJECT_CLASS(parent_class)->dispose(object);
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
    GstLalpad *elem = GST_LALPAD(object);

    switch (prop_id) {
    case PROP_PRE:
        elem->pre = g_value_get_uint64(value);
        break;
    case PROP_POST:
        elem->post = g_value_get_uint64(value);
        break;
    case PROP_UNITS:
        elem->unit = g_value_get_enum(value);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}


static void get_property(GObject *object, guint prop_id,
                         GValue *value, GParamSpec *pspec)
{
    GstLalpad *elem = GST_LALPAD(object);

    switch (prop_id) {
    case PROP_PRE:
        g_value_set_uint64(value, elem->pre);
        break;
    case PROP_POST:
        g_value_set_uint64(value, elem->post);
        break;
    case PROP_UNITS:
        g_value_set_enum(value, elem->unit);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}



/*
 * ============================================================================
 *
 *                                Utilities
 *
 * ============================================================================
 */

/* Types of gap buffers: pre(pended) or post (appended) */
enum buf_type {
    TYPE_PRE,
    TYPE_POST,
};

/*
 * Auxiliary function. Push a gap into the source pad, of nsamples
 * number of samples.
 *
 * If pre==TRUE, it is a buffer that before the next one. Else, it
 * comes after the last one.
 */
static GstFlowReturn push_gap(GstPad *pad, enum buf_type type)
{
    GstFlowReturn result = GST_FLOW_OK;
    GstLalpad *elem = GST_LALPAD(GST_PAD_PARENT(pad));
    GstBuffer *buf;

    /* First check that we are in condition to create the gap */
    if (elem->caps == NULL) {
        GST_ERROR_OBJECT(elem, "caps not setted yet");
        return GST_FLOW_ERROR;
    }

    /* Allocate a 0-sized buffer */
    result = gst_pad_alloc_buffer(pad, GST_BUFFER_OFFSET_NONE,
                                  0, elem->caps, &buf);

    if (result != GST_FLOW_OK) {
        GST_ERROR_OBJECT(elem, "failed allocating gap buffer");
        return result;
    }

    /* Fill all the buffer metadata */
    /* -- caps and gap flag */
    gst_buffer_set_caps(buf, elem->caps);

    GST_BUFFER_FLAG_SET(buf, GST_BUFFER_FLAG_GAP);

    /* -- offsets and timestamps */
    /* ---- short notation */
    guint64 offset = elem->saved_offset, offset_end = elem->saved_offset_end;
    GstClockTime t = elem->saved_timestamp, dt = elem->saved_duration;

    /* ---- compute the number of samples and the duration of the gap */
    guint64 nsamples = 0;
    GstClockTime duration = 0;

    if (elem->unit == GST_LALPAD_UNIT_SAMPLES) {
        if (type == TYPE_PRE)
            nsamples = elem->pre;
        else if (type == TYPE_POST)
            nsamples = elem->post;
        duration = A_X_B__C(nsamples, GST_SECOND, elem->rate);
    }
    else if (elem->unit == GST_LALPAD_UNIT_TIME) {
        if (type == TYPE_PRE)
            duration = elem->pre;
        else if (type == TYPE_POST)
            duration = elem->post;
        nsamples = A_X_B__C(duration, elem->rate, GST_SECOND);
    }

    if (nsamples == 0) {
        gst_buffer_unref(buf);
        return GST_FLOW_OK;  // nothing to send if no samples!
    }

    /* ---- finally set offsets and timestamps, for PRE or POST */
    if (type == TYPE_PRE) {
        GST_BUFFER_OFFSET(buf) = offset - nsamples;
        GST_BUFFER_OFFSET_END(buf) = offset;
        GST_BUFFER_DURATION(buf) = duration;
        GST_BUFFER_TIMESTAMP(buf) = t - duration;
    }
    else if (type == TYPE_POST) {
        GST_BUFFER_OFFSET(buf) = offset_end;
        GST_BUFFER_OFFSET_END(buf) = offset_end + nsamples;
        GST_BUFFER_DURATION(buf) = duration;
        GST_BUFFER_TIMESTAMP(buf) = t + dt;
    }

    /* If we wanted to fill a buffer with something instead of
     * sending a gap, we can compute the size using:
     *
     *   gint width;
     *   gst_structure_get_int(str, "width", &width);   // read width
     *   size = elem->pre * width / 8;
     */

    /* Push it and be done */
    result = gst_pad_push(pad, buf);
    if (result != GST_FLOW_OK) {
        GST_ERROR("gst_pad_push() failed pushing gap buffer");
        return result;
    }

    return GST_FLOW_OK;
}


/*
 * ============================================================================
 *
 *                                Events
 *
 * ============================================================================
 */

/*
 * Gets called if the Data passed to this element is an Event (the
 * only other option being a Buffer).
 *
 * Events contain a subtype indicating the type of the contained
 * event.
 *
 * Set first_buffer=TRUE for NEWSEGMENTs, and push post-padding if
 * necessary for the previous block. Also push post-padding at EOS.
 */
static gboolean event(GstPad *pad, GstEvent *event)
{
    GstLalpad *elem = GST_LALPAD(GST_OBJECT_PARENT(pad));

    GST_DEBUG_OBJECT(elem, "Got an event of type %s",
                     gst_event_type_get_name(GST_EVENT_TYPE(event)));

    switch (GST_EVENT_TYPE(event)) {
    case GST_EVENT_NEWSEGMENT:
    {
        /* If this is not the first buffer, put the padding for the previous */
        if (!elem->first_buffer)
            if (push_gap(elem->srcpad, TYPE_POST) != GST_FLOW_OK)
                return FALSE;

        elem->first_buffer = TRUE;
        break;
    }
    case GST_EVENT_EOS:
    {
        if (!elem->first_buffer)  // put padding for previous block
            if (push_gap(elem->srcpad, TYPE_POST) != GST_FLOW_OK)
                return FALSE;
        break;
    }
    default:
        break;
    }

    return gst_pad_push_event(elem->srcpad, event);
}



/*
 * ============================================================================
 *
 *                                Chain
 *
 * ============================================================================
 */

static GstFlowReturn chain(GstPad *pad, GstBuffer *buf)
{
    GstLalpad *elem = GST_LALPAD(GST_OBJECT_PARENT(pad));

    /* Save info from buffer in case it is the first one or the last one */
    elem->saved_offset = GST_BUFFER_OFFSET(buf);
    elem->saved_offset_end = GST_BUFFER_OFFSET_END(buf);
    elem->saved_timestamp = GST_BUFFER_TIMESTAMP(buf);
    elem->saved_duration = GST_BUFFER_DURATION(buf);

    /* Send the pre-pad before the buffer, for the 1st buffer */
    if (elem->first_buffer) {
        elem->caps = gst_buffer_get_caps(buf);
        if (push_gap(elem->srcpad, TYPE_PRE) != GST_FLOW_OK)
            return GST_FLOW_ERROR;
        elem->first_buffer = FALSE;
    }

    /* Send the buffer */
    return gst_pad_push(elem->srcpad, buf);
}


static gboolean setcaps(GstPad *pad, GstCaps *caps)
{
    GstLalpad *elem = GST_LALPAD(GST_OBJECT_PARENT(pad));
    GstStructure *str;
    gint rate;

    /* Forward-negotiate */
    if (!gst_pad_set_caps(elem->srcpad, caps))
        return FALSE;

    /* Negotiation succeeded, so now configure ourselves */
    str = gst_caps_get_structure(caps, 0);
    gst_structure_get_int(str, "rate", &rate);

    /* Signal if change of rate */
    if (rate != elem->rate) {
        g_signal_emit(G_OBJECT(elem), signal_rate_changed,
                      0, rate, NULL);
        elem->rate = rate;
    }

    return TRUE;
}

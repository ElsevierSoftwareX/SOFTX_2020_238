/*
 * framecpp filesink
 *
 * Copyright (C) 2013  Branson Stephens, Kipp Cannon
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
 * ========================================================================
 *
 *                                Preamble
 *
 * ========================================================================
 */


/* 
 * stuff from standard library
 */


#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/*
 * stuff from glib/gobject/gstreamer
 */ 


#include <glib.h>
#include <glib-object.h>
#include <gst/gst.h>
#include <gst/base/gstbasesink.h>

/*
 * our own stuff
 */


#include <gstlal/gstlal_tags.h>
#include <framecpp_filesink.h>


/*
 * ============================================================================
 *
 *                                Boilerplate
 *
 * ============================================================================
 */


G_DEFINE_TYPE(
        FRAMECPPFilesink,
        framecpp_filesink,
        GST_TYPE_BIN
);


/*
 * ============================================================================
 *
 *                                 Parameters
 *
 * ============================================================================
 */


#define DEFAULT_FRAME_TYPE  "test_frame"
#define DEFAULT_INSTRUMENT NULL
#define DEFAULT_PATH "."
#define DEFAULT_TIMESTAMP 0


/*
 * ============================================================================
 *
 *                                 Utilities
 *
 * ============================================================================
 */


/*
 * convert a string of the form "H1,V1,H2" into a string of the form "HV".
 * the return value must be freed with g_free() when no longer needed.
 */


int strCmpWrap(const void *pa, const void *pb){
    /* Incoming args are actually pointers to pointers to char.
       Whereas g_strcmp0 expects pointers to char. 
       Thus, we cast and dereference once. */
    return g_strcmp0(*(char * const *)pa, *(char * const *)pb);
}

static gchar *observatory_from_instruments(const gchar *instruments)
{
    gchar *observatory;
    gchar **in, **out;
    /* split comma-delimited string */
    gchar **split_instruments = g_strsplit(instruments, ",", 0);
    /* strip whitespace */
    for(in = split_instruments; *in; in++) g_strstrip(*in);

    /* sort */
    qsort(split_instruments, g_strv_length(split_instruments),
        sizeof(*split_instruments), strCmpWrap);

    /* null-terminate each instrument after 1st character */
    for(in = split_instruments; *in; in++) {
        if(strlen(*in) > 1)
                (*in)[1] = '\0';
    }

    /* "remove" duplicates by setting them to 0 length */
    for(in = out = split_instruments; *out; out = in) {
        for(in++; !g_strcmp0(*in, *out); in++)
                (*in)[0] = '\0';
    }

    /* concatenate */
    observatory = g_strjoinv(NULL, split_instruments);
    g_strfreev(split_instruments);

    /* done */
    return observatory;
}


/*
 * pad probe handlers
 */


static GstPadProbeReturn probeEventHandler(GstPad *pad, GstPadProbeInfo *info, gpointer data) {
    FRAMECPPFilesink *element = FRAMECPP_FILESINK(gst_pad_get_parent(pad));
    GstEvent *event = GST_EVENT(info->data);
    GstTagList *tag_list;
    gchar *instrumentList = NULL;
    gchar *observatoryString = NULL;

    g_assert(gst_pad_is_linked(pad));

    if (GST_EVENT_TYPE(event)==GST_EVENT_TAG) {
        gst_event_parse_tag(event, &tag_list);
        if (gst_tag_list_get_string(tag_list, GSTLAL_TAG_INSTRUMENT, &instrumentList)){
            observatoryString = observatory_from_instruments(instrumentList);
            GST_DEBUG("setting instrument to %s", observatoryString);
            g_object_set(G_OBJECT(element), "instrument", observatoryString, NULL);
        }
    }

    gst_object_unref(element);
    g_free(instrumentList);
    g_free(observatoryString);
    return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn probeBufferHandler(GstPad *pad, GstPadProbeInfo *info, gpointer data) {
    FRAMECPPFilesink *element = FRAMECPP_FILESINK(gst_pad_get_parent(pad));
    GstBuffer *buffer = GST_BUFFER(info->data);
    guint timestamp, end_time, duration;
    gchar *filename, *location;

    g_assert(gst_pad_is_linked(pad));

    /* Buffer looks good, else die. */
    g_assert(GST_BUFFER_TIMESTAMP_IS_VALID(buffer));
    g_assert(GST_BUFFER_DURATION_IS_VALID(buffer));

    /* Set the element timestamp property */
    element->timestamp = GST_BUFFER_TIMESTAMP(buffer);
    g_object_notify(G_OBJECT(element), "timestamp");

    if (!(element->instrument)) {
        /* Instrument should have come from via the stream, hence STREAM error. */
        GST_ELEMENT_ERROR(element, STREAM, TYPE_NOT_FOUND, (NULL), ("instrument not set in framecpp_filesink element."));
        goto done;
    } else if (!(element->frame_type)) {
        /* frame_type is an input parameter, hence RESOURCE error. */
        GST_ELEMENT_ERROR(element, RESOURCE, NOT_FOUND, (NULL), ("frame_type not set in framecpp_filesink element."));
        goto done;
    }

    timestamp = GST_BUFFER_TIMESTAMP(buffer)/GST_SECOND;
    end_time = gst_util_uint64_scale_ceil(GST_BUFFER_TIMESTAMP(buffer) + GST_BUFFER_DURATION(buffer), 1, GST_SECOND);
    duration = end_time - timestamp;
    /* The interval indicated by the filename should "cover" the actual 
    data interval. */
    g_assert_cmpuint(duration*GST_SECOND, >=, GST_BUFFER_DURATION(buffer));
    filename = g_strdup_printf("%s-%s-%d-%d.gwf", element->instrument, 
        element->frame_type, timestamp, duration); 
    location = g_build_path(G_DIR_SEPARATOR_S, element->path, filename, NULL);
    g_free(filename);

    if(element->mfs) {
        GST_DEBUG("Setting write location to %s", location);
        g_object_set(G_OBJECT(element->mfs), "location", location, NULL);
    }

    g_free(location);

done:
    gst_object_unref(element);
    return GST_PAD_PROBE_OK;
}


/*
 * ============================================================================
 *
 *                             GObject Overrides
 *
 * ============================================================================
 */


/*
 * properties
 */


enum {
    PROP_0,
    PROP_PATH,
    PROP_FRAME_TYPE,
    PROP_INSTRUMENT,
    PROP_TIMESTAMP,
};


static void set_property(GObject *object, guint prop_id, 
                         const GValue *value, GParamSpec *pspec)
{
    FRAMECPPFilesink *sink = FRAMECPP_FILESINK(object);
    GST_OBJECT_LOCK(object);
    switch (prop_id) {
    case PROP_FRAME_TYPE:
        g_free(sink->frame_type);
        sink->frame_type = g_strdup(g_value_get_string(value));
        break;
    case PROP_INSTRUMENT:
        g_free(sink->instrument);
        sink->instrument = g_strdup(g_value_get_string(value));
        break;
    case PROP_PATH:
        g_free(sink->path);
        sink->path = g_strdup(g_value_get_string(value));
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        g_assert_not_reached();
        break;
    }
    GST_OBJECT_UNLOCK(object);
}


static void get_property(GObject *object, guint prop_id, 
                         GValue *value, GParamSpec *pspec)
{
    FRAMECPPFilesink *sink = FRAMECPP_FILESINK(object);
    GST_OBJECT_LOCK(object);
    switch (prop_id) {
    case PROP_FRAME_TYPE:
        g_value_set_string(value, sink->frame_type);
        break;
    case PROP_INSTRUMENT:
        g_value_set_string(value, sink->instrument);
        break;
    case PROP_PATH:
        g_value_set_string(value, sink->path);
        break;
    case PROP_TIMESTAMP:
        g_value_set_uint64(value, sink->timestamp);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        g_assert_not_reached();
        break;
    }
    GST_OBJECT_UNLOCK(object);
}


/*
 * Instance dispose function.
 */


static void dispose(GObject *object)
{
    FRAMECPPFilesink *element = FRAMECPP_FILESINK(object);

    if (element->mfs) {
        gst_object_unref(element->mfs);
        element->mfs = NULL;
    }

    G_OBJECT_CLASS(framecpp_filesink_parent_class)->dispose(object);
}


/*
 * Instance finalize function.
 */


static void finalize(GObject *object)
{
    FRAMECPPFilesink *element = FRAMECPP_FILESINK(object);

    g_free(element->frame_type);
    element->frame_type = NULL;
    g_free(element->instrument);
    element->instrument = NULL;
    g_free(element->path);
    element->path = NULL;

    G_OBJECT_CLASS(framecpp_filesink_parent_class)->finalize(object);
}


/*
 * Pad template.
 */


static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE(
        "sink",
        GST_PAD_SINK,
        GST_PAD_ALWAYS,
        GST_STATIC_CAPS(
            "application/x-igwd-frame, " \
            "framed = (boolean) true" 
        )
);


/*
 * class_init()
 */

static void framecpp_filesink_class_init(FRAMECPPFilesinkClass *klass)
{
    GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
    GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

    gst_element_class_set_details_simple(element_class, "Write frame files from muxer", "Sink/File", "Comment", "Branson Stephens <stephenb@uwm.edu>");
    gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));

    gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
    gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
    gobject_class->dispose = GST_DEBUG_FUNCPTR(dispose);
    gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

    g_object_class_install_property(
        gobject_class, PROP_FRAME_TYPE,
        g_param_spec_string(
            "frame-type", "Frame type.",
            "Type of frame, a description of its contents", DEFAULT_FRAME_TYPE,
            (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)
            )
        );

    g_object_class_install_property(
        gobject_class, PROP_INSTRUMENT,
        g_param_spec_string(
            "instrument", "Observatory string.",
            "The IFO, like H1, L1, V1, etc.", DEFAULT_INSTRUMENT,
            (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)
            )
        );

    g_object_class_install_property(
        gobject_class, PROP_PATH,
        g_param_spec_string(
            "path", "Write path.",
            "The directory where the frames should be written.", DEFAULT_PATH,
            (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)
            )
        );

    g_object_class_install_property(
        gobject_class, PROP_TIMESTAMP,
        g_param_spec_uint64(
            "timestamp", "Buffer timestamp.",
            "Timestamp of the current buffer in nanoseconds.", 0, G_MAXUINT64, 
            DEFAULT_TIMESTAMP,
            (GParamFlags) (G_PARAM_READABLE | G_PARAM_STATIC_STRINGS)
            )
        );

}


/*
 * instance_init()
 */


static void framecpp_filesink_init(FRAMECPPFilesink *element)
{
    gboolean retval;

    /* initialize element timestamp property */
    element->timestamp = GST_CLOCK_TIME_NONE;

    /* Create the multifilesink element. */
    element->mfs = gst_element_factory_make("multifilesink", "multifilesink");
    g_object_set(G_OBJECT(element->mfs), "sync", FALSE, "async", FALSE, NULL);

    /* Add the multifilesink to the bin (consumes one reference) */
    gst_object_ref(element->mfs);
    retval = gst_bin_add(GST_BIN(element), element->mfs);
    g_assert(retval == TRUE);

    /* Add the ghostpad */
    GstPad *sink = gst_element_get_static_pad(element->mfs, "sink");
    GstPad *sink_ghost = gst_ghost_pad_new_from_template("sink", sink, gst_element_class_get_pad_template(GST_ELEMENT_CLASS(G_OBJECT_GET_CLASS(element)),"sink"));
    retval = gst_element_add_pad(GST_ELEMENT(element), sink_ghost);
    g_assert(retval == TRUE);

    gst_object_unref(sink);

    /* add event and buffer probes */
    gst_pad_add_probe(sink_ghost, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM, probeEventHandler, NULL, NULL);
    gst_pad_add_probe(sink_ghost, GST_PAD_PROBE_TYPE_BUFFER, probeBufferHandler, NULL, NULL);
}



/*
 * Copyright (C) 2010 Jordi Burguet-Castell <jordi.burguet-castell@ligo.org>
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
 * SECTION:element-lalframesink
 * @see_also: #GSTLALFrameSrc
 *
 * Write incoming data to a sequence of GWF files in the local file system.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch audiotestsrc wave=sine num-buffers=1000 ! \
 *   taginject tags="instrument=H1,channel-name=H1:LSC-STRAIN,units=strain" ! \
 *   audio/x-raw-float,rate=16384,width=64 ! \
 *   lal_framesink path=. frame-type=hoft
 * ]| Save wave into a sequence of gwf files of the form ./H-hoft-TIME-SPAN.gwf.
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif


#include <lal/LALFrameIO.h>
#include <lal/TimeSeries.h>
#include <lal/LALDetectors.h>  // LAL_LHO_4K_DETECTOR_BIT et al
#include <lal/Units.h>         // lalDimensionlessUnit

#include <gstlal.h>
#include <gstlal_tags.h>

#include <gst/gst.h>
#include <stdio.h>              /* for fseeko() */
#ifdef HAVE_STDIO_EXT_H
#include <stdio_ext.h>          /* for __fbufsize, for debugging */
#endif
#include <errno.h>
#include "gstlal_framesink.h"
#include <string.h>
#include <sys/types.h>

#include <sys/stat.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif


GST_DEBUG_CATEGORY_STATIC(gst_lalframe_sink_debug);
#define GST_CAT_DEFAULT gst_lalframe_sink_debug

enum {
    PROP_0,
    PROP_PATH,
    PROP_FRAME_TYPE,
    PROP_DURATION,
};


static void dispose(GObject *object);

static void set_property(GObject *object, guint prop_id,
                         const GValue *value, GParamSpec *pspec);
static void get_property(GObject *object, guint prop_id,
                         GValue *value, GParamSpec *pspec);

static gboolean event(GstBaseSink *basesink, GstEvent *event);

static gboolean start(GstBaseSink *basesink);
static gboolean stop(GstBaseSink *basesink);
static gboolean query(GstPad *pad, GstQuery *query);
static GstFlowReturn render(GstBaseSink *basesink, GstBuffer *buffer);

static gboolean write_frame(GstLalframeSink *sink, guint nbytes);


/*
 * ============================================================================
 *
 *                                Type Support
 *
 * ============================================================================
 */


GST_BOILERPLATE(GstLalframeSink, gst_lalframe_sink, GstBaseSink,
                GST_TYPE_BASE_SINK);


/*
 * Base init function.
 *
 * Element description and pads description.
 */
static void gst_lalframe_sink_base_init(gpointer g_class)
{
    GstElementClass *gstelement_class = GST_ELEMENT_CLASS(g_class);

    /* Element description. */
    gst_element_class_set_details_simple(
        gstelement_class,
        "GWF Frame File Sink",
        "Sink/GWF",
        "Write data to a frame file",
        "Jordi Burguet-Castell <jordi.burguet-castell@ligo.org>");

    /* Pad description. */
    gst_element_class_add_pad_template(
        gstelement_class,
        gst_pad_template_new(
            "sink",
            GST_PAD_SINK,
            GST_PAD_ALWAYS,
            gst_caps_from_string(
                "audio/x-raw-float, "
                "rate = (int) [1, MAX], "
                "channels = (int) 1, "
                "endianness = (int) BYTE_ORDER, "
                "width = (int) {32, 64}; "
                "audio/x-raw-int, "
                "rate = (int) [1, MAX], "
                "channels = (int) 1, "
                "endianness = (int) BYTE_ORDER, "
                "width = (int) 32, "
                "depth = (int) 32, "
                "signed = (boolean) true")));
}


/*
 * Class init function.
 *
 * Specify properties ("arguments").
 */
static void gst_lalframe_sink_class_init(GstLalframeSinkClass *klass)
{
    GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
    GstBaseSinkClass *gstbasesink_class = GST_BASE_SINK_CLASS(klass);

    gobject_class->dispose = dispose;

    gobject_class->set_property = set_property;
    gobject_class->get_property = get_property;

    g_object_class_install_property(
        gobject_class, PROP_PATH,
        g_param_spec_string(
            "path", "Path to files.",
            "Directory where the frame files will be written", ".",
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property(
        gobject_class, PROP_FRAME_TYPE,
        g_param_spec_string(
            "frame-type", "Frame type.",
            "Type of frame, kind of description of its contents", "test_frame",
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property(
        gobject_class, PROP_DURATION,
        g_param_spec_double(
            "duration", "Duration",
            "Time span (in s) stored in each frame file", 0, G_MAXDOUBLE, 64,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    gstbasesink_class->get_times = NULL;  /* no sync */
    gstbasesink_class->start = GST_DEBUG_FUNCPTR(start);
    gstbasesink_class->stop = GST_DEBUG_FUNCPTR(stop);
    gstbasesink_class->render = GST_DEBUG_FUNCPTR(render);
    gstbasesink_class->event = GST_DEBUG_FUNCPTR(event);

    if (sizeof(off_t) < 8) {
        GST_LOG("No large file support, sizeof(off_t) = %" G_GSIZE_FORMAT "!",
                sizeof(off_t));
    }
}


/*
 * Instance init function.
 *
 * Create caps.
 */
static void gst_lalframe_sink_init(GstLalframeSink *sink,
                                   GstLalframeSinkClass *g_class)
{
    GstPad *pad;

    pad = GST_BASE_SINK_PAD(sink);

    gst_pad_set_query_function(pad, GST_DEBUG_FUNCPTR(query));

    sink->path = g_strdup(".");
    sink->frame_type = g_strdup("test_frame");
    sink->instrument = NULL;
    sink->channel_name = NULL;
    sink->units = NULL;
    sink->duration = 64;
    sink->adapter = gst_adapter_new();

    sink->rate = 0;
    sink->width = 0;
    sink->type = NULL;

    gst_base_sink_set_sync(GST_BASE_SINK(sink), FALSE);
}


static void dispose(GObject *object)
{
    GstLalframeSink *sink = GST_LALFRAME_SINK(object);

    if (sink->adapter) {
        g_object_unref(sink->adapter);
        sink->adapter = NULL;
    }
    g_free(sink->units);
    sink->units = NULL;
    g_free(sink->channel_name);
    sink->channel_name = NULL;
    g_free(sink->instrument);
    sink->instrument = NULL;
    g_free(sink->frame_type);
    sink->frame_type = NULL;
    g_free(sink->path);
    sink->path = NULL;

    g_free(sink->type);
    sink->type = NULL;

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
    GstLalframeSink *sink = GST_LALFRAME_SINK(object);

    switch (prop_id) {
    case PROP_PATH:
        g_free(sink->path);
        sink->path = g_strdup(g_value_get_string(value));
        break;
    case PROP_FRAME_TYPE:
        g_free(sink->frame_type);
        sink->frame_type = g_strdup(g_value_get_string(value));
        break;
    case PROP_DURATION:
        sink->duration = g_value_get_double(value);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}


static void
get_property(GObject *object, guint prop_id, GValue *value, GParamSpec *pspec)
{
    GstLalframeSink *sink = GST_LALFRAME_SINK(object);

    switch (prop_id) {
    case PROP_PATH:
        g_value_set_string(value, sink->path);
        break;
    case PROP_FRAME_TYPE:
        g_value_set_string(value, sink->frame_type);
        break;
    case PROP_DURATION:
        g_value_set_double(value, sink->duration);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}


/*
 * ============================================================================
 *
 *                                Events
 *
 * ============================================================================
 */


static inline void extract(GstLalframeSink *sink, GstTagList *taglist,
                           const char *tagname, gchar **dest)
{
    gchar *tmp;
    if (!gst_tag_list_get_string(taglist, tagname, &tmp)) {
        GST_WARNING_OBJECT(sink, "Unable to parse \"%s\" from %" GST_PTR_FORMAT,
                           tagname, taglist);
        return;
    }
    g_free(*dest);
    *dest = tmp;
    GST_DEBUG_OBJECT(sink, "Found tag \"%s\"=\"%s\"", tagname, *dest);
}


static gboolean event(GstBaseSink *basesink, GstEvent *event)
{
    GstLalframeSink *sink = GST_LALFRAME_SINK(basesink);

    switch (GST_EVENT_TYPE(event)) {
    case GST_EVENT_TAG:  /* from gstlal_simulation.c */
    {
        GstTagList *taglist;
        gst_event_parse_tag(event, &taglist);
        extract(sink, taglist, GSTLAL_TAG_INSTRUMENT, &sink->instrument);
        extract(sink, taglist, GSTLAL_TAG_CHANNEL_NAME, &sink->channel_name);
        extract(sink, taglist, GSTLAL_TAG_UNITS, &sink->units);
        break;
    }
    case GST_EVENT_NEWSEGMENT:
    {
        gint64 start, stop, pos;
        GstFormat format;
        GstPad *pad;
        GstStructure *str;

        gst_event_parse_new_segment(event, NULL, NULL, &format, &start,
                                    &stop, &pos);

        if (format == GST_FORMAT_BYTES) {
            /* only try to seek and fail when we are going to a different
             * position */
            if (sink->current_pos != (guint64) start)
                goto seek_failed;
            else
                GST_DEBUG_OBJECT(sink, "Ignored NEWSEGMENT, no seek needed");
        }
        else {
            GST_DEBUG_OBJECT(
                sink,
                "Ignored NEWSEGMENT event of format %u (%s)", (guint) format,
                gst_format_get_name(format));
        }

        /* Keep info about the data stream */
        pad = gst_element_get_static_pad(GST_ELEMENT(sink), "sink");
        str = gst_caps_get_structure(GST_PAD_CAPS(pad), 0);

        gst_structure_get_int(str, "width", &sink->width);
        gst_structure_get_int(str, "rate", &sink->rate);
        g_free(sink->type);
        sink->type = g_strdup(gst_structure_get_name(str));

        break;
    }
    case GST_EVENT_EOS:
    {
        guint nbytes = gst_adapter_available(sink->adapter);
        if (nbytes > 0) {
            if (!write_frame(sink, nbytes))
                goto flush_failed;
            gst_adapter_flush(sink->adapter, nbytes);
        }
        break;
    }
    default:
        break;
    }

    return TRUE;

    /* ERRORS */
seek_failed:
    {
        GST_ELEMENT_ERROR(
            sink, RESOURCE, SEEK,
            ("Error while seeking in function %s", __FUNCTION__),
            GST_ERROR_SYSTEM);
        return FALSE;
    }
flush_failed:
    {
        GST_ELEMENT_ERROR(
            sink, RESOURCE, WRITE,
            ("Error while writing in function %s", __FUNCTION__),
            GST_ERROR_SYSTEM);
        return FALSE;
    }
}


/*
 * ============================================================================
 *
 *                          Start, Stop, Query, Render
 *
 * ============================================================================
 */


static gboolean start(GstBaseSink *basesink)
{
    // We may want to open files or other resources...
    return TRUE;
}


static gboolean stop(GstBaseSink *basesink)
{
    // And we may want to close resources too...
    return TRUE;
}


static gboolean query(GstPad *pad, GstQuery *query)
{
    GstFormat format;
    GstLalframeSink *sink = GST_LALFRAME_SINK(GST_PAD_PARENT(pad));

    switch (GST_QUERY_TYPE(query)) {
    case GST_QUERY_POSITION:
        gst_query_parse_position(query, &format, NULL);
        switch (format) {
        case GST_FORMAT_DEFAULT:
        case GST_FORMAT_BYTES:
            gst_query_set_position(query, GST_FORMAT_BYTES, sink->current_pos);
            return TRUE;
        default:
            return FALSE;
        }

    case GST_QUERY_FORMATS:
        gst_query_set_formats(query, 2, GST_FORMAT_DEFAULT, GST_FORMAT_BYTES);
        return TRUE;

    default:
        return gst_pad_query_default(pad, query);
    }
}


/* This is the most important one. It calls write_frame() */
static GstFlowReturn render(GstBaseSink *basesink, GstBuffer *buffer)
{
    GstLalframeSink *sink = GST_LALFRAME_SINK(basesink);
    guint nbytes = sink->duration * sink->rate * sink->width / 8;

    gst_buffer_ref(buffer);  /* a reference is lost in GstBaseSink's render */

    gst_adapter_push(sink->adapter, buffer);  /* put buffer into adapter */

    while (gst_adapter_available(sink->adapter) >= nbytes) {
        if (!write_frame(sink, nbytes))
            return GST_FLOW_ERROR;

        gst_adapter_flush(sink->adapter, nbytes);
    }

    return GST_FLOW_OK;
}


/*
 * ============================================================================
 *
 *                                Utilities
 *
 * ============================================================================
 */


static gboolean write_frame(GstLalframeSink *sink, guint nbytes)
{
    double duration = nbytes*8.0 / (sink->rate * sink->width);
    double deltaT = 1.0 / sink->rate;  /* to write in the TimeSeries */
    int ifo_flags;
    LIGOTimeGPS epoch;
    GstClockTime timestamp;
    FrameH *frame;
    double f0 = 0;  /* kind of dummy, to write in the TimeSeries */
    char filename[1024];

    if (sink->instrument == NULL || sink->path == NULL ||
        sink->frame_type == NULL || sink->channel_name == NULL)
        goto handle_error;

    /* Get detector flags */
    if      (strcmp(sink->instrument, "H1") == 0)
        ifo_flags = LAL_LHO_4K_DETECTOR_BIT;
    else if (strcmp(sink->instrument, "H2") == 0)
        ifo_flags = LAL_LHO_2K_DETECTOR_BIT;
    else if (strcmp(sink->instrument, "L1") == 0)
        ifo_flags = LAL_LLO_4K_DETECTOR_BIT;
    else if (strcmp(sink->instrument, "V1") == 0)
        ifo_flags = LAL_VIRGO_DETECTOR_BIT;
    else
        ifo_flags = -1;

    /* Get timestamp from adapter */
    timestamp = gst_adapter_prev_timestamp(sink->adapter, NULL);
    epoch.gpsSeconds     = timestamp / GST_SECOND;
    epoch.gpsNanoSeconds = timestamp % GST_SECOND;

    frame = XLALFrameNew(&epoch, duration, "LIGO", 0, 1, ifo_flags);

    if (strcmp(sink->type, "audio/x-raw-float") == 0) {
        if (sink->width == 64) {
            REAL8TimeSeries *ts = XLALCreateREAL8TimeSeries(
                sink->channel_name, &epoch, f0, deltaT,
                &lalDimensionlessUnit, nbytes/8);

            gst_adapter_copy(sink->adapter, (guint8 *) ts->data->data,
                             0, nbytes);

            XLALFrameAddREAL8TimeSeriesProcData(frame, ts);
        }
        else if (sink->width == 32) {
            REAL4TimeSeries *ts = XLALCreateREAL4TimeSeries(
                sink->channel_name, &epoch, f0, deltaT,
                &lalDimensionlessUnit, nbytes/4);

            gst_adapter_copy(sink->adapter, (guint8 *) ts->data->data,
                             0, nbytes);

            XLALFrameAddREAL4TimeSeriesProcData(frame, ts);
        }
    }
    else if (strcmp(sink->type, "audio/x-raw-int") == 0) {
        INT4TimeSeries *ts = XLALCreateINT4TimeSeries(
            sink->channel_name, &epoch, f0, deltaT,
            &lalDimensionlessUnit, nbytes/4);

        gst_adapter_copy(sink->adapter, (guint8 *) ts->data->data, 0, nbytes);

        XLALFrameAddINT4TimeSeriesProcData(frame, ts);
    }

    snprintf(filename, sizeof(filename), "%s/%c-%s-%.15g-%.15g.gwf",
             sink->path, sink->instrument[0], sink->frame_type,
             epoch.gpsSeconds + epoch.gpsNanoSeconds*1e-9, duration);
    if (XLALFrameWrite(frame, filename, -1) != 0)
        goto handle_error;

    sink->current_pos += nbytes;

    return TRUE;

handle_error:
    {
        switch (errno) {
        case ENOSPC: {
            GST_ELEMENT_ERROR(sink, RESOURCE, NO_SPACE_LEFT, (NULL), (NULL));
            break;
        }
        default: {
            GST_ELEMENT_ERROR(sink, RESOURCE, WRITE,
                ("Error in function %s", __FUNCTION__),
                ("%s", g_strerror(errno)));
        }
        }
        return FALSE;
    }
}

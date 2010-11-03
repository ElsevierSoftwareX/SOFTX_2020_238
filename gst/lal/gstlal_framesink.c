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
 * gst-launch audiotestsrc wave=sine num-buffers=100 ! \
 *   taginject tags="instrument=H1,channel-name=H1:LSC-STRAIN,units=strain" ! \
 *   audio/x-raw-float,rate=16384,width=64 ! lal_framesink prefix=./out
 * ]| Save wave into a sequence of gwf files of the form ./out-TIME-SPAN.gwf.
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
    PROP_PREFIX,
    PROP_INSTRUMENT,
    PROP_CHANNEL_NAME,
    PROP_UNITS,
};


static void gst_lalframe_sink_dispose(GObject *object);

static void gst_lalframe_sink_set_property(GObject *object, guint prop_id,
                                           const GValue *value,
                                           GParamSpec *pspec);
static void gst_lalframe_sink_get_property(GObject *object, guint prop_id,
                                           GValue *value, GParamSpec *pspec);

static gboolean gst_lalframe_sink_start(GstBaseSink *sink);
static gboolean gst_lalframe_sink_stop(GstBaseSink *sink);
static gboolean gst_lalframe_sink_event(GstBaseSink *sink, GstEvent *event);
static GstFlowReturn gst_lalframe_sink_render(GstBaseSink *sink,
                                              GstBuffer *buffer);

static gboolean gst_lalframe_sink_query(GstPad *pad, GstQuery *query);



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
static void
gst_lalframe_sink_base_init(gpointer g_class)
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
static void
gst_lalframe_sink_class_init(GstLalframeSinkClass *klass)
{
    GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
    GstBaseSinkClass *gstbasesink_class = GST_BASE_SINK_CLASS(klass);

    gobject_class->dispose = gst_lalframe_sink_dispose;

    gobject_class->set_property = gst_lalframe_sink_set_property;
    gobject_class->get_property = gst_lalframe_sink_get_property;

    g_object_class_install_property(
        gobject_class, PROP_PREFIX,
        g_param_spec_string(
            "prefix", "Path and file prefix",
            "Prefix of the files PREFIX-TIME-SPAN.gwf to write", NULL,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property(
        gobject_class, PROP_INSTRUMENT,
        g_param_spec_string(
            "instrument", "Instrument",
            "Name of the interferometer (H1, H2, L1, V1)", NULL,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property(
        gobject_class, PROP_CHANNEL_NAME,
        g_param_spec_string(
            "channel-name", "Channel name",
            "Name of the channel as will appear in the file", NULL,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property(
        gobject_class, PROP_UNITS,
        g_param_spec_string(
            "units", "Units",
            "Units of the data. Not used yet.", NULL,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    gstbasesink_class->get_times = NULL;  //// ?
    gstbasesink_class->start = GST_DEBUG_FUNCPTR(gst_lalframe_sink_start);
    gstbasesink_class->stop = GST_DEBUG_FUNCPTR(gst_lalframe_sink_stop);
    gstbasesink_class->render = GST_DEBUG_FUNCPTR(gst_lalframe_sink_render);
    gstbasesink_class->event = GST_DEBUG_FUNCPTR(gst_lalframe_sink_event);

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
static void
gst_lalframe_sink_init(GstLalframeSink *sink,
                       GstLalframeSinkClass *g_class)
{
    GstPad *pad;

    pad = GST_BASE_SINK_PAD(sink);

    gst_pad_set_query_function(pad, GST_DEBUG_FUNCPTR(gst_lalframe_sink_query));

    sink->prefix = NULL;
    sink->instrument = NULL;
    sink->channel_name = NULL;
    sink->units = NULL;
    sink->adapter = gst_adapter_new();

    /* retrieve (and ref) sink pad */
    sink->sinkpad = gst_element_get_static_pad(GST_ELEMENT(sink), "sink");

    gst_base_sink_set_sync(GST_BASE_SINK(sink), FALSE);
}

static void
gst_lalframe_sink_dispose(GObject *object)
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
    g_free(sink->prefix);
    sink->prefix = NULL;

    G_OBJECT_CLASS(parent_class)->dispose(object);
}


/*
 * ============================================================================
 *
 *                                 Properties
 *
 * ============================================================================
 */

static void
gst_lalframe_sink_set_property(GObject *object, guint prop_id,
                               const GValue *value, GParamSpec *pspec)
{
    GstLalframeSink *sink = GST_LALFRAME_SINK(object);

    switch (prop_id) {
    case PROP_PREFIX:
        g_free(sink->prefix);
        sink->prefix = g_strdup(g_value_get_string(value));
        break;
    case PROP_INSTRUMENT:
        g_free(sink->instrument);
        sink->instrument = g_strdup(g_value_get_string(value));
        break;
    case PROP_CHANNEL_NAME:
        g_free(sink->channel_name);
        sink->channel_name = g_strdup(g_value_get_string(value));
        break;
    case PROP_UNITS:
        g_free(sink->units);
        sink->units = g_strdup(g_value_get_string(value));
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

static void
gst_lalframe_sink_get_property(GObject *object, guint prop_id, GValue *value,
                               GParamSpec *pspec)
{
    GstLalframeSink *sink = GST_LALFRAME_SINK(object);

    switch (prop_id) {
    case PROP_PREFIX:
        g_value_set_string(value, sink->prefix);
        break;
    case PROP_INSTRUMENT:
        g_value_set_string(value, sink->instrument);
        break;
    case PROP_CHANNEL_NAME:
        g_value_set_string(value, sink->channel_name);
        break;
    case PROP_UNITS:
        g_value_set_string(value, sink->units);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}


/*
 * ============================================================================
 *
 *                           Open, query, seek, event
 *
 * ============================================================================
 */


static gboolean
gst_lalframe_sink_query(GstPad *pad, GstQuery *query)
{
    GstLalframeSink *self;
    GstFormat format;

    self = GST_LALFRAME_SINK(GST_PAD_PARENT(pad));

    switch (GST_QUERY_TYPE(query)) {
    case GST_QUERY_POSITION:
        gst_query_parse_position(query, &format, NULL);
        switch (format) {
        case GST_FORMAT_DEFAULT:
        case GST_FORMAT_BYTES:
            gst_query_set_position(query, GST_FORMAT_BYTES, self->current_pos);
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


/* handle events(search) */

static gboolean
taglist_extract_string(GstLalframeSink *element, GstTagList *taglist,
                       const char *tagname, gchar **dest)
{
    if (!gst_tag_list_get_string(taglist, tagname, dest)) {
        GST_WARNING_OBJECT(element, "unable to parse \"%s\" from %"
                           GST_PTR_FORMAT, tagname, taglist);
        return FALSE;
    }
    return TRUE;
}

static gboolean
gst_lalframe_sink_event(GstBaseSink *base_sink, GstEvent *event)
{
    gboolean success;

    GstLalframeSink *sink;

    sink = GST_LALFRAME_SINK(base_sink);

    switch (GST_EVENT_TYPE(event)) {
    case GST_EVENT_TAG:  /* from gstlal_simulation.c */
    {
        GstTagList *taglist;
        gchar *instrument, *channel_name, *units;
        gst_event_parse_tag(event, &taglist);
        success = taglist_extract_string(sink, taglist, GSTLAL_TAG_INSTRUMENT, &instrument);
        success &= taglist_extract_string(sink, taglist, GSTLAL_TAG_CHANNEL_NAME, &channel_name);
        success &= taglist_extract_string(sink, taglist, GSTLAL_TAG_UNITS, &units);
        if (success) {
            GST_DEBUG_OBJECT(
                sink, "found tags \"%s\"=\"%s\" \"%s\"=\"%s\" \"%s\"=\"%s\"",
                GSTLAL_TAG_INSTRUMENT, instrument, GSTLAL_TAG_CHANNEL_NAME,
                channel_name, GSTLAL_TAG_UNITS, units);
            g_free(sink->instrument);
            sink->instrument = instrument;
            g_free(sink->channel_name);
            sink->channel_name = channel_name;
            g_free(sink->units);
            sink->units = units;
        }
        break;
    }
    case GST_EVENT_NEWSEGMENT:
    {
        gint64 start, stop, pos;
        GstFormat format;

        gst_event_parse_new_segment(event, NULL, NULL, &format, &start,
                                    &stop, &pos);

        if (format == GST_FORMAT_BYTES) {
            /* only try to seek and fail when we are going to a different
             * position */
            if (sink->current_pos != (guint64) start) {
                goto seek_failed;
            } else {
                GST_DEBUG_OBJECT(
                    sink, "Ignored NEWSEGMENT, no seek needed");
            }
        } else {
            GST_DEBUG_OBJECT(
                sink,
                "Ignored NEWSEGMENT event of format %u(%s)", (guint) format,
                gst_format_get_name(format));
        }
        break;
    }
    case GST_EVENT_EOS:
        // FIXME: Fill the last frame if there is something to save
        if (0)  // whatever is appropiate
            goto flush_failed;
        break;
    default:
        break;
    }

    return TRUE;

    /* ERRORS */
seek_failed:
    {
        GST_ELEMENT_ERROR(
            sink, RESOURCE, SEEK,
            ("Error while seeking in file \"%s-TIME-SPAN.gwf\".", sink->prefix),
            GST_ERROR_SYSTEM);
        return FALSE;
    }
flush_failed:
    {
        GST_ELEMENT_ERROR(
            sink, RESOURCE, WRITE,
            ("Error while writing to file \"%s-TIME-SPAN.gwf\".", sink->prefix),
            GST_ERROR_SYSTEM);
        return FALSE;
    }
}

static GstFlowReturn
gst_lalframe_sink_render(GstBaseSink *base_sink, GstBuffer *buffer)
{
    GstLalframeSink *sink;
//    guint size;
    double *data;  // people normally address it as  guint8 *data; ...
    /* Number of expected bytes. 16 s, 16*1024 Hz, 8 bytes/double */
    const guint N_EXP_BYTES = 16 * 16*1024 * sizeof(double);

    sink = GST_LALFRAME_SINK(base_sink);

//    size = GST_BUFFER_SIZE(buffer);

    gst_buffer_ref(buffer); //// FIXME: really? where do I lose the ref??
    gst_adapter_push(sink->adapter, buffer);
    while (gst_adapter_available(sink->adapter) >= N_EXP_BYTES) {
        data = (double *) gst_adapter_peek(sink->adapter, N_EXP_BYTES);

        FrameH *frame;
        double duration = N_EXP_BYTES / (sizeof(double)*16.0*1024);
        int detectorFlags;
        LIGOTimeGPS epoch;
        REAL8TimeSeries *series;  //// FIXME: pick type depending on input
        double f0 = 0;
        double deltaT = 1.0/(16*1024);  ///// FIXME: take sample rate from the buffer's caps
        guint i;
        char name[256];

        if      (strcmp(sink->instrument, "H1") == 0)
            detectorFlags = LAL_LHO_4K_DETECTOR_BIT;
        else if (strcmp(sink->instrument, "H2") == 0)
            detectorFlags = LAL_LHO_2K_DETECTOR_BIT;
        else if (strcmp(sink->instrument, "L1") == 0)
            detectorFlags = LAL_LLO_4K_DETECTOR_BIT;
        else if (strcmp(sink->instrument, "V1") == 0)
            detectorFlags = LAL_VIRGO_DETECTOR_BIT;
        else
            detectorFlags = -1;

   /* /\* get timestamp of the current adapter byte *\/ */

   /*  timestamp = gst_adapter_prev_timestamp(sink->adapter, &dist); */

        epoch.gpsSeconds = GST_BUFFER_TIMESTAMP(buffer) / GST_SECOND;
        epoch.gpsNanoSeconds = GST_BUFFER_TIMESTAMP(buffer) % GST_SECOND;

        if (sink->channel_name == NULL)
            goto handle_error;

        frame = XLALFrameNew(&epoch, duration, "LIGO", 0, 1, detectorFlags);

        series = XLALCreateREAL8TimeSeries(sink->channel_name,
                                           &epoch, f0, deltaT,
                                           &lalDimensionlessUnit,
                                           N_EXP_BYTES/sizeof(double));

        /* copy buffer contents to timeseries */
        for (i = 0; i < N_EXP_BYTES/sizeof(double); i++)
            series->data->data[i] = data[i];

        XLALFrameAddREAL8TimeSeriesProcData(frame, series);

        snprintf(name, sizeof(name), "%s-%d-%g.gwf",
                 sink->prefix, epoch.gpsSeconds, duration);
        if (XLALFrameWrite(frame, name, -1) != 0)
            goto handle_error;

        sink->current_pos += N_EXP_BYTES;

        gst_adapter_flush(sink->adapter, N_EXP_BYTES);
    }

    return GST_FLOW_OK;

handle_error:
    {
        switch (errno) {
        case ENOSPC: {
            GST_ELEMENT_ERROR(
                sink, RESOURCE, NO_SPACE_LEFT, (NULL), (NULL));
            break;
        }
        default: {
            GST_ELEMENT_ERROR(
                sink, RESOURCE, WRITE,
                ("Error while writing to file \"%s-GPSTIME-SPAN.gwf\".",
                 sink->prefix),
                ("%s", g_strerror(errno)));
        }
        }
        return GST_FLOW_ERROR;
    }
}

static gboolean
gst_lalframe_sink_start(GstBaseSink *basesink)
{
    // We may want to open files or other resources...
    return TRUE;
}

static gboolean
gst_lalframe_sink_stop(GstBaseSink *basesink)
{
    // And we may want to close resources too...
    return TRUE;
}

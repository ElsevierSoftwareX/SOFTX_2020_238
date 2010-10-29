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
 * Write incoming data to a GWF file in the local file system.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch audiotestsrc wave=sine num_buffers=100 ! audio/x-raw-float,rate=16384,width=64 ! lal_framesink location=out.gwf
 * ]| Save a sine wave into a gwf file.
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif


#include <lal/LALFrameIO.h>
#include <lal/TimeSeries.h>
#include <lal/LALDetectors.h>  // LAL_LHO_4K_DETECTOR_BIT
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


#define GST_TYPE_BUFFER_MODE (buffer_mode_get_type())
static GType
buffer_mode_get_type(void)
{
    static GType buffer_mode_type = 0;
    static const GEnumValue buffer_mode[] = {
        {-1, "Default buffering", "default"},
        {_IOFBF, "Fully buffered", "full"},
        {_IOLBF, "Line buffered", "line"},
        {_IONBF, "Unbuffered", "unbuffered"},
        {0, NULL, NULL},
    };

    if (!buffer_mode_type) {
        buffer_mode_type =
            g_enum_register_static("GstLalframeSinkBufferMode", buffer_mode);
    }
    return buffer_mode_type;
}

GST_DEBUG_CATEGORY_STATIC(gst_lalframe_sink_debug);
#define GST_CAT_DEFAULT gst_lalframe_sink_debug

#define DEFAULT_BUFFER_MODE     -1
#define DEFAULT_BUFFER_SIZE     64 * 1024

enum
{
    PROP_0,
    PROP_LOCATION,
    PROP_INSTRUMENT,
    PROP_CHANNEL_NAME,
    PROP_UNITS,
    PROP_BUFFER_MODE,
    PROP_BUFFER_SIZE
};


static void gst_lalframe_sink_dispose(GObject * object);

static void gst_lalframe_sink_set_property(GObject * object, guint prop_id,
                                           const GValue * value,
                                           GParamSpec * pspec);
static void gst_lalframe_sink_get_property(GObject * object, guint prop_id,
                                           GValue * value, GParamSpec * pspec);

static gboolean gst_lalframe_sink_open_file(GstLalframeSink * sink);
static void gst_lalframe_sink_close_file(GstLalframeSink * sink);

static gboolean gst_lalframe_sink_start(GstBaseSink * sink);
static gboolean gst_lalframe_sink_stop(GstBaseSink * sink);
static gboolean gst_lalframe_sink_event(GstBaseSink * sink, GstEvent * event);
static GstFlowReturn gst_lalframe_sink_render(GstBaseSink * sink,
                                              GstBuffer * buffer);

static gboolean gst_lalframe_sink_query(GstPad * pad, GstQuery * query);

static void gst_lalframe_sink_uri_handler_init(gpointer g_iface,
                                               gpointer iface_data);



/*
 * ============================================================================
 *
 *                                Type Support
 *
 * ============================================================================
 */

static void
_do_init(GType lalframesink_type)
{
    static const GInterfaceInfo urihandler_info = {
        gst_lalframe_sink_uri_handler_init,
        NULL,
        NULL
    };

    g_type_add_interface_static(lalframesink_type, GST_TYPE_URI_HANDLER,
                                &urihandler_info);
    GST_DEBUG_CATEGORY_INIT(gst_lalframe_sink_debug, "lalframesink", 0,
                            "lalframesink element");
}

GST_BOILERPLATE_FULL(GstLalframeSink, gst_lalframe_sink, GstBaseSink,
                     GST_TYPE_BASE_SINK, _do_init);


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
gst_lalframe_sink_class_init(GstLalframeSinkClass * klass)
{
    GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
    GstBaseSinkClass *gstbasesink_class = GST_BASE_SINK_CLASS(klass);

    gobject_class->dispose = gst_lalframe_sink_dispose;

    gobject_class->set_property = gst_lalframe_sink_set_property;
    gobject_class->get_property = gst_lalframe_sink_get_property;

    g_object_class_install_property(
        gobject_class, PROP_LOCATION,
        g_param_spec_string(
            "location", "File Location",
            "Location of the file to write", NULL,
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
    
    g_object_class_install_property(
        gobject_class, PROP_BUFFER_MODE,
        g_param_spec_enum(
            "buffer-mode", "Buffering mode",
            "The buffering mode to use", GST_TYPE_BUFFER_MODE,
            DEFAULT_BUFFER_MODE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property(
        gobject_class, PROP_BUFFER_SIZE,
        g_param_spec_uint(
            "buffer-size", "Buffering size",
            "Size of buffer in number of bytes for line or full buffer-mode", 0,
            G_MAXUINT, DEFAULT_BUFFER_SIZE,
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
gst_lalframe_sink_init(GstLalframeSink * lalframesink,
                       GstLalframeSinkClass * g_class)
{
    GstPad *pad;

    pad = GST_BASE_SINK_PAD(lalframesink);

    gst_pad_set_query_function(pad, GST_DEBUG_FUNCPTR(gst_lalframe_sink_query));

    lalframesink->filename = NULL;
    lalframesink->frame = NULL;
    lalframesink->buffer_mode = DEFAULT_BUFFER_MODE;
    lalframesink->buffer_size = DEFAULT_BUFFER_SIZE;
    lalframesink->buffer = NULL;
    lalframesink->instrument = NULL;
    lalframesink->channel_name = NULL;
    lalframesink->units = NULL;

    /* retrieve (and ref) src pad */
    lalframesink->srcpad = gst_element_get_static_pad(GST_ELEMENT(lalframesink), "src");

    gst_base_sink_set_sync(GST_BASE_SINK(lalframesink), FALSE);
}

static void
gst_lalframe_sink_dispose(GObject * object)
{
    GstLalframeSink *sink = GST_LALFRAME_SINK(object);

    G_OBJECT_CLASS(parent_class)->dispose(object);

    g_free(sink->units);
    sink->units = NULL;
    g_free(sink->channel_name);
    sink->channel_name = NULL;
    g_free(sink->instrument);
    sink->instrument = NULL;
    g_free(sink->uri);
    sink->uri = NULL;
    g_free(sink->filename);
    sink->filename = NULL;
    g_free(sink->buffer);
    sink->buffer = NULL;
    sink->buffer_size = 0;
}


/*
 * ============================================================================
 *
 *                                 Utilities
 *
 * ============================================================================
 */

static gboolean
gst_lalframe_sink_set_location(GstLalframeSink * sink, const gchar * location)
{
    if (sink->frame)
        goto was_open;

    g_free(sink->filename);
    g_free(sink->uri);
    if (location != NULL) {
        /* we store the filename as we received it from the application */
        sink->filename = g_strdup(location);
        sink->uri = gst_uri_construct("file", sink->filename);
    } else {
        sink->filename = NULL;
        sink->uri = NULL;
    }

    return TRUE;

    /* ERRORS */
was_open:
    {
        g_warning("Changing the `location' property on lalframesink when a "
                  "file is open is not supported.");
        return FALSE;
    }
}

static gboolean
gst_lalframe_sink_set_instrument(GstLalframeSink * sink,
                                 const gchar * instrument)
{
    if (sink->frame)
        goto was_open;

    g_free(sink->instrument);
    if (instrument != NULL) {
        /* we store the filename as we received it from the application */
        sink->instrument = g_strdup(instrument);
    } else {
        sink->instrument = NULL;
    }

    return TRUE;

    /* ERRORS */
was_open:
    {
        g_warning("Changing the `instrument' property on lalframesink when a "
                  "file is open is not supported.");
        return FALSE;
    }
}

static gboolean
gst_lalframe_sink_set_channel_name(GstLalframeSink * sink,
                                   const gchar * channel_name)
{
    if (sink->frame)
        goto was_open;

    g_free(sink->channel_name);
    if (channel_name != NULL) {
        /* we store the filename as we received it from the application */
        sink->channel_name = g_strdup(channel_name);
    } else {
        sink->channel_name = NULL;
    }

    return TRUE;

    /* ERRORS */
was_open:
    {
        g_warning("Changing the `channel-name' property on lalframesink when a "
                  "file is open is not supported.");
        return FALSE;
    }
}

static gboolean
gst_lalframe_sink_set_units(GstLalframeSink * sink, const gchar * units)
{
    if (sink->frame)
        goto was_open;

    g_free(sink->units);
    if (units != NULL) {
        /* we store the filename as we received it from the application */
        sink->units = g_strdup(units);
    } else {
        sink->units = NULL;
    }

    return TRUE;

    /* ERRORS */
was_open:
    {
        g_warning("Changing the `units' property on lalframesink when a "
                  "file is open is not supported.");
        return FALSE;
    }
}


/*
 * ============================================================================
 *
 *                                 Properties
 *
 * ============================================================================
 */

static void
gst_lalframe_sink_set_property(GObject * object, guint prop_id,
                               const GValue * value, GParamSpec * pspec)
{
    GstLalframeSink *sink = GST_LALFRAME_SINK(object);

    switch (prop_id) {
    case PROP_LOCATION:
        gst_lalframe_sink_set_location(sink, g_value_get_string(value));
        break;
    case PROP_INSTRUMENT:
        gst_lalframe_sink_set_instrument(sink, g_value_get_string(value));
        break;
    case PROP_CHANNEL_NAME:
        gst_lalframe_sink_set_channel_name(sink, g_value_get_string(value));
        break;
    case PROP_UNITS:
        gst_lalframe_sink_set_units(sink, g_value_get_string(value));
        break;
    case PROP_BUFFER_MODE:
        sink->buffer_mode = g_value_get_enum(value);
        break;
    case PROP_BUFFER_SIZE:
        sink->buffer_size = g_value_get_uint(value);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

static void
gst_lalframe_sink_get_property(GObject * object, guint prop_id, GValue * value,
                               GParamSpec * pspec)
{
    GstLalframeSink *sink = GST_LALFRAME_SINK(object);

    switch (prop_id) {
    case PROP_LOCATION:
        g_value_set_string(value, sink->filename);
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
    case PROP_BUFFER_MODE:
        g_value_set_enum(value, sink->buffer_mode);
        break;
    case PROP_BUFFER_SIZE:
        g_value_set_uint(value, sink->buffer_size);
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
gst_lalframe_sink_open_file(GstLalframeSink * sink)
{
    gint mode;

    /* open the file */
    if (sink->filename == NULL || sink->filename[0] == '\0')
        goto no_filename;

    {
        LIGOTimeGPS epoch;
        const int duration = 60;
        const int run = 0;
        const int frnum = 1;
        const int detectorFlags = LAL_LHO_4K_DETECTOR_BIT;

        /* Initialization */
        epoch.gpsSeconds = 600000000;
        epoch.gpsNanoSeconds = 0;

        sink->frame = XLALFrameNew(&epoch, duration, "LIGO", run, frnum, detectorFlags);
    /* This evidently has to be properly filled! Get epoch, duration
     * and detector from metadata or something. */
    }

    if (sink->frame == NULL)
        goto open_failed;

    /* see if we are asked to perform a specific kind of buffering */
    if ((mode = sink->buffer_mode) != -1) {
        GST_WARNING_OBJECT(
            sink, "warning: asked for buffering or something, that I don't know how to do!", g_strerror(errno));
    }

    sink->current_pos = 0;
    sink->seekable = FALSE;

    GST_DEBUG_OBJECT(sink, "opened file %s, seekable %d",
                     sink->filename, sink->seekable);

    return TRUE;

    /* ERRORS */
no_filename:
    {
        GST_ELEMENT_ERROR(
            sink, RESOURCE, NOT_FOUND,
            ("No file name specified for writing."), (NULL));
        return FALSE;
    }
open_failed:
    {
        GST_ELEMENT_ERROR(
            sink, RESOURCE, OPEN_WRITE,
            ("Could not open file \"%s\" for writing.", sink->filename),
            GST_ERROR_SYSTEM);
        return FALSE;
    }
}

static void
gst_lalframe_sink_close_file(GstLalframeSink * sink)
{
    if (sink->frame) {
        if (XLALFrameWrite(sink->frame, sink->filename, -1) != 0)
            goto close_failed;

        GST_DEBUG_OBJECT(sink, "closed file");
        sink->frame = NULL;

        g_free(sink->buffer);
        sink->buffer = NULL;
    }
    return;

    /* ERRORS */
close_failed:
    {
        GST_ELEMENT_ERROR(
            sink, RESOURCE, CLOSE,
            ("Error closing file \"%s\".", sink->filename), GST_ERROR_SYSTEM);
        return;
    }
}

static gboolean
gst_lalframe_sink_query(GstPad * pad, GstQuery * query)
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

    case GST_QUERY_URI:
        gst_query_set_uri(query, self->uri);
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
gst_lalframe_sink_event(GstBaseSink * base_sink, GstEvent * event)
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
            success = gst_pad_push_event(sink->srcpad, event);
            /* FIXME:  flush the cache of injection timeseries */
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
            if (sink->current_pos != start) {
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
        if (XLALFrameWrite(sink->frame, sink->filename, -1) != 0)
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
            ("Error while seeking in file \"%s\".", sink->filename),
            GST_ERROR_SYSTEM);
        return FALSE;
    }
flush_failed:
    {
        GST_ELEMENT_ERROR(
            sink, RESOURCE, WRITE,
            ("Error while writing to file \"%s\".", sink->filename),
            GST_ERROR_SYSTEM);
        return FALSE;
    }
}

static GstFlowReturn
gst_lalframe_sink_render(GstBaseSink * sink, GstBuffer * buffer)
{
    GstLalframeSink *lalframesink;
    guint size;
    guint8 *data;

    lalframesink = GST_LALFRAME_SINK(sink);

    size = GST_BUFFER_SIZE(buffer);
    data = GST_BUFFER_DATA(buffer);

    GST_DEBUG_OBJECT(lalframesink, "writing %u bytes at %" G_GUINT64_FORMAT,
                     size, lalframesink->current_pos);

    if (size > 0 && data != NULL) {
        LIGOTimeGPS epoch;
        REAL8TimeSeries *series;  //// FIXME: pick type depending on input
        double f0 = 0;        ////  FIXME
        double deltaT = 1.0;  ///// FIXME

        epoch.gpsSeconds = 600000000;   //// FIXME
        epoch.gpsNanoSeconds = 0;       //// FIXME

        if (lalframesink->channel_name == NULL)
            goto handle_error;

        series = XLALCreateREAL8TimeSeries(lalframesink->channel_name,
                                           &epoch, f0, deltaT,
                                           &lalDimensionlessUnit, size);

        {  /* copy buffer contents to timeseries */
            size_t i;
            for (i = 0; i < size; i++)
                series->data->data[i] = data[i];
        }

        XLALFrameAddREAL8TimeSeriesProcData(lalframesink->frame, series);

        ////
        /* gchar* tmpname = g_strconcat("temp_", lalframesink->frame); */
        /* if (XLALFrameWrite(lalframesink->frame, tmpname, -1) != 0) */
        /*     goto handle_error; */

        lalframesink->current_pos += size;
    }

    return GST_FLOW_OK;

handle_error:
    {
        switch (errno) {
        case ENOSPC: {
            GST_ELEMENT_ERROR(
                lalframesink, RESOURCE, NO_SPACE_LEFT, (NULL), (NULL));
            break;
        }
        default: {
            GST_ELEMENT_ERROR(
                lalframesink, RESOURCE, WRITE,
                ("Error while writing to file \"%s\".", lalframesink->filename),
                ("%s", g_strerror(errno)));
        }
        }
        return GST_FLOW_ERROR;
    }
}

static gboolean
gst_lalframe_sink_start(GstBaseSink * basesink)
{
    return gst_lalframe_sink_open_file(GST_LALFRAME_SINK(basesink));
}

static gboolean
gst_lalframe_sink_stop(GstBaseSink * basesink)
{
    gst_lalframe_sink_close_file(GST_LALFRAME_SINK(basesink));
    return TRUE;
}

/*** GSTURIHANDLER INTERFACE *************************************************/

static GstURIType
gst_lalframe_sink_uri_get_type(void)
{
    return GST_URI_SINK;
}

static gchar **
gst_lalframe_sink_uri_get_protocols(void)
{
    static gchar *protocols[] = { "file", NULL };

    return protocols;
}

static const gchar *
gst_lalframe_sink_uri_get_uri(GstURIHandler * handler)
{
    GstLalframeSink *sink = GST_LALFRAME_SINK(handler);

    return sink->uri;
}

static gboolean
gst_lalframe_sink_uri_set_uri(GstURIHandler * handler, const gchar * uri)
{
    gchar *protocol, *location;
    gboolean ret;
    GstLalframeSink *sink = GST_LALFRAME_SINK(handler);

    protocol = gst_uri_get_protocol(uri);
    if (strcmp(protocol, "file") != 0) {
        g_free(protocol);
        return FALSE;
    }
    g_free(protocol);

    /* allow file://localhost/foo/bar by stripping localhost but fail
     * for every other hostname */
    if (g_str_has_prefix(uri, "file://localhost/")) {
        char *tmp;

        /* 16 == strlen("file://localhost") */
        tmp = g_strconcat("file://", uri + 16, NULL);
        /* we use gst_uri_get_location() although we already have the
         * "location" with uri + 16 because it provides unescaping */
        location = gst_uri_get_location(tmp);
        g_free(tmp);
    } else if (strcmp(uri, "file://") == 0) {
        /* Special case for "file://" as this is used by some applications
         *  to test with gst_element_make_from_uri if there's an element
         *  that supports the URI protocol. */
        gst_lalframe_sink_set_location(sink, NULL);
        return TRUE;
    } else {
        location = gst_uri_get_location(uri);
    }

    if (!location)
        return FALSE;
    if (!g_path_is_absolute(location)) {
        g_free(location);
        return FALSE;
    }

    ret = gst_lalframe_sink_set_location(sink, location);
    g_free(location);

    return ret;
}

static void
gst_lalframe_sink_uri_handler_init(gpointer g_iface, gpointer iface_data)
{
    GstURIHandlerInterface *iface = (GstURIHandlerInterface *) g_iface;

    iface->get_type = gst_lalframe_sink_uri_get_type;
    iface->get_protocols = gst_lalframe_sink_uri_get_protocols;
    iface->get_uri = gst_lalframe_sink_uri_get_uri;
    iface->set_uri = gst_lalframe_sink_uri_set_uri;
}

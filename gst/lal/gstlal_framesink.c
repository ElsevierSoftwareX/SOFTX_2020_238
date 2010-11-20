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
 *   taginject tags="instrument=H1,channel-name=LSC-STRAIN,units=strain" ! \
 *   audio/x-raw-float,rate=16384,width=64 ! \
 *   lal_framesink path=. frame-type=hoft
 * ]| Save wave into a sequence of gwf files of the form ./H-hoft-TIME-SPAN.gwf.
 * </refsect2>
 */

static const char gst_lalframe_sink_doc[] =
    "Write data to frame files.\n"
    "\n"
    "This element will read a stream of either int, float or double data and "
    "save it into frame files in subdirectories. The files will look like:\n"
    "  ./H-test_frame-9888/H-test_frame-988889100-64.gwf\n"
    "\n"
    "It requires the following tags to be present:\n"
    "  instrument   - e.g. G1, H1, H2, L1, V1\n"
    "  channel-name - name of channel in the frames, e.g. LSC-STRAIN\n"
    "  units        - currently not used\n"
    "\n"
    "The duration property controls the length of each frame file. The "
    "timestamps of the files are a multiple of this duration, unless the "
    "property clean-timestamps is false.\n"
    "\n"
    "The dir-digits property controls how many digits appear in the name of "
    "the subdirectories, by specifying how many of the least significant "
    "digits should be removed. If 0, no subdirectories are created.\n";

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
#include "gstlal_framesink.h"
#include <math.h>

#include <sys/stat.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

/* Geez, I don't wanna write so much! */
#define A_X_B__C(a, b, c)  gst_util_uint64_scale_int_round(a, b, c)
/* This is the same as
 *    #define A_X_B__C(a, b, c)  a * b / c
 * except it takes better care of possible over/underflows. It seems
 * better to use gst_util_uint64_scale_int_round() but audiotestsrc
 * computes the timestamps of its buffers with gst_..._int() <shrugs>
 */


enum {
    PROP_0,
    PROP_PATH,
    PROP_FRAME_TYPE,
    PROP_DURATION,
    PROP_CLEAN_TIMESTAMPS,
    PROP_DIR_DIGITS,
};


#define GST_CAT_DEFAULT gstlal_framesink_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


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


static void additional_initializations(GType type)
{
    GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "framesink", 0,
                            "framesink element");
}


GST_BOILERPLATE_FULL(GstLalframeSink, gst_lalframe_sink, GstBaseSink,
                     GST_TYPE_BASE_SINK, additional_initializations);


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
        gst_lalframe_sink_doc,
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

    g_object_class_install_property(
        gobject_class, PROP_CLEAN_TIMESTAMPS,
        g_param_spec_boolean(
            "clean-timestamps", "Clean timestamps",
            "Files start at a multiple of \"duration\"", TRUE,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_property(
        gobject_class, PROP_DIR_DIGITS,
        g_param_spec_int(
            "dir-digits", "Directory digits to remove",
            "Number of least significant digits dropped from subdir names",
            0, G_MAXINT, 5,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    gstbasesink_class->get_times = NULL;  // no sync
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
 * Initialize instance. Give default values.
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
    sink->clean_timestamps = TRUE;
    sink->dir_digits = 5;
    sink->adapter = gst_adapter_new();

    sink->rate = 0;
    sink->width = 0;
    sink->type = NULL;
    sink->start = TRUE;
    sink->t0_ns = 0;

    gst_base_sink_set_sync(GST_BASE_SINK(sink), FALSE);
}


/*
 * Free resources.
 */
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
    case PROP_CLEAN_TIMESTAMPS:
        sink->clean_timestamps = g_value_get_boolean(value);
        break;
    case PROP_DIR_DIGITS:
        sink->dir_digits = g_value_get_int(value);
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
    case PROP_CLEAN_TIMESTAMPS:
        g_value_set_boolean(value, sink->clean_timestamps);
        break;
    case PROP_DIR_DIGITS:
        g_value_set_int(value, sink->dir_digits);
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


/*
 * Helper function used in event(). Get a string from a tag and put it
 * in dest. And track what's going on with warnings or debug info.
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


/*
 * Gets called if the Data passed to this element is an Event (the
 * only other option being a Buffer).
 *
 * Events contain a subtype indicating the type of the contained event. For:
 *   tags received: save the values in instrument, channel_name and units
 *   new segment: save the type of data coming (int/float, rate, width)
 *   end of stream: flush remaining data into a (shorter) frame
 */
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

        gst_structure_get_int(str, "width", &sink->width);   // read width
        gst_structure_get_int(str, "rate", &sink->rate);     // read rate
        g_free(sink->type);
        sink->type = g_strdup(gst_structure_get_name(str));  // mime type

        break;
    }
    case GST_EVENT_EOS:
    {
        guint nbytes = gst_adapter_available(sink->adapter);
        if (nbytes > 0) {
            if (!write_frame(sink, nbytes))  // write any remaining data
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
    GstLalframeSink *sink = GST_LALFRAME_SINK(basesink);
    sink->current_pos = 0;
    return TRUE;
}


static gboolean stop(GstBaseSink *basesink)
{
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


/*
 * This is the most important one. It calls write_frame()
 */
static GstFlowReturn render(GstBaseSink *basesink, GstBuffer *buffer)
{
    GstLalframeSink *sink = GST_LALFRAME_SINK(basesink);
    guint byterate = sink->rate * sink->width / 8;
    guint nbytes = sink->duration * byterate;

    GST_DEBUG_OBJECT(
        sink, "Got timestamp=%llu, duration=%llu, offset=%llu, offset_end=%llu",
        GST_BUFFER_TIMESTAMP(buffer), GST_BUFFER_DURATION(buffer),
        GST_BUFFER_OFFSET(buffer), GST_BUFFER_OFFSET_END(buffer));

    /* Get t0_ns from the very first buffer */
    if (sink->start) {
        sink->t0_ns = GST_BUFFER_TIMESTAMP(buffer);
        sink->start = FALSE;
    }

    /* Check for discontinuities (see gstlal_firbank.c at ~ 909) */
    guint available = gst_adapter_available(sink->adapter);

    if (GST_BUFFER_IS_DISCONT(buffer) || GST_BUFFER_TIMESTAMP(buffer) !=
        sink->t0_ns + A_X_B__C(available, GST_SECOND, byterate)) {
        GST_INFO_OBJECT(sink, "Detected discontinuity");
        /* Flush previous data to a frame */
        if (available > 0) {
            if (!write_frame(sink, available)) { // write any remaining data
                GST_ELEMENT_ERROR(
                    sink, RESOURCE, WRITE,
                    ("Error while writing in function %s", __FUNCTION__),
                    GST_ERROR_SYSTEM);
                return GST_FLOW_ERROR;
            }

            /* Flush adapter and restart counting timestamps */
            gst_adapter_flush(sink->adapter, available);
            sink->t0_ns = GST_BUFFER_TIMESTAMP(buffer);
        }
    }

    /* Compensate for reference lost in GstBaseSink's render */
    gst_buffer_ref(buffer);

    /* Put buffer into adapter */
    gst_adapter_push(sink->adapter, buffer);

    /* Write a first frame if needed to align timestamps in the filenames */
    if (sink->clean_timestamps) {
        GstClockTime dur_ns = (GstClockTime) (sink->duration * GST_SECOND);

        if (sink->t0_ns % dur_ns != 0) {  // if beginning of data is not aligned
            gdouble dt = (dur_ns - sink->t0_ns % dur_ns) / (gdouble) GST_SECOND;
            gdouble n = dt * byterate;  // bytes to save
            if (fabs(n - (guint) n) > 1e-12) {
                GST_ELEMENT_ERROR(
                    sink, STREAM, FAILED,
                    ("Impossible to do clean timestamps. Current timestamp (%"
                     G_GUINT64_FORMAT " ns) and rate (%d Hz) will not produce "
                     "a timestamp multiple of duration (%.14g s)",
                     sink->t0_ns, sink->rate, sink->duration), (NULL));
                return GST_FLOW_ERROR;
            }

            if (gst_adapter_available(sink->adapter) < n)
                return GST_FLOW_OK;  // not enough data to write yet, fine

            /* Write only a few bytes so next frame starts at proper times */
            if (!write_frame(sink, n))
                return GST_FLOW_ERROR;

            gst_adapter_flush(sink->adapter, n);
            sink->t0_ns += A_X_B__C(n, GST_SECOND, byterate);
        }
    }

    /* Keep writing frames of requested duration while we have data */
    while (gst_adapter_available(sink->adapter) >= nbytes) {
        if (!write_frame(sink, nbytes))
            return GST_FLOW_ERROR;

        gst_adapter_flush(sink->adapter, nbytes);
        sink->t0_ns += A_X_B__C(nbytes, GST_SECOND, byterate);
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


/*
 * Write a single frame file of size nbytes.
 */
static gboolean write_frame(GstLalframeSink *sink, guint nbytes)
{
    guint byterate = sink->rate * sink->width / 8;
    double duration = nbytes / (double) byterate;
    double deltaT = 1.0 / sink->rate;  // to write in the TimeSeries
    int ifo_flags;
    LIGOTimeGPS epoch;
    FrameH *frame;
    double f0 = 0;  // kind of dummy, to write in the TimeSeries
    gchar *channame, *dirname, *filename;

    if (sink->instrument == NULL || sink->path == NULL ||
        sink->frame_type == NULL || sink->channel_name == NULL) {
        GST_ELEMENT_ERROR(
            sink, STREAM, FAILED,
            ("instrument, path, frame-type, or channel-name not set"), (NULL));
        return FALSE;
    }

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
    epoch.gpsSeconds     = sink->t0_ns / GST_SECOND;
    epoch.gpsNanoSeconds = sink->t0_ns % GST_SECOND;

    /* Create subdirectories with nice names if needed */
    if (sink->dir_digits > 0) {
        int pre = epoch.gpsSeconds / (int) pow(10, sink->dir_digits);
        dirname = g_strdup_printf("%s/%c-%s-%d",
                                  sink->path, sink->instrument[0],
                                  sink->frame_type, pre);
        if (mkdir(dirname, 0777) != 0 && errno != EEXIST) {
            GST_ELEMENT_ERROR(
                sink, RESOURCE, WRITE,
                ("Could not create directory: %s", dirname), GST_ERROR_SYSTEM);
            g_free(dirname);
            return FALSE;
        }
    }
    else {
        dirname = g_strdup(sink->path);
    }

    filename = g_strdup_printf("%s/%c-%s-%.15g-%.15g.gwf",
                               dirname, sink->instrument[0], sink->frame_type,
                               epoch.gpsSeconds + epoch.gpsNanoSeconds*1e-9,
                               duration);
    g_free(dirname);

    channame = g_strdup_printf("%s:%s", sink->instrument, sink->channel_name);

    /* Create frame file */
    frame = XLALFrameNew(&epoch, duration, "LIGO", 0, 1, ifo_flags);

    /* Create and store the proper timeseries depending on data mime type */
    if (strcmp(sink->type, "audio/x-raw-float") == 0) {
        if (sink->width == 64) {  // 64 bits = 8 bytes ... a double
            REAL8TimeSeries *ts = XLALCreateREAL8TimeSeries(
                channame, &epoch, f0, deltaT, &lalDimensionlessUnit, nbytes/8);

            if (ts == NULL)
                goto memory_failed;

            gst_adapter_copy(sink->adapter, (guint8 *) ts->data->data,
                             0, nbytes);

            XLALFrameAddREAL8TimeSeriesProcData(frame, ts);

            if (XLALFrameWrite(frame, filename, -1) != 0)
                goto write_failed;

            XLALDestroyREAL8TimeSeries(ts);
        }
        else if (sink->width == 32) {  // 32 bits = 4 bytes... a float
            REAL4TimeSeries *ts = XLALCreateREAL4TimeSeries(
                channame, &epoch, f0, deltaT, &lalDimensionlessUnit, nbytes/4);

            if (ts == NULL)
                goto memory_failed;

            gst_adapter_copy(sink->adapter, (guint8 *) ts->data->data,
                             0, nbytes);

            XLALFrameAddREAL4TimeSeriesProcData(frame, ts);

            if (XLALFrameWrite(frame, filename, -1) != 0)
                goto write_failed;

            XLALDestroyREAL4TimeSeries(ts);
        }
    }
    else if (strcmp(sink->type, "audio/x-raw-int") == 0) {
        INT4TimeSeries *ts = XLALCreateINT4TimeSeries(
            channame, &epoch, f0, deltaT, &lalDimensionlessUnit, nbytes/4);

        if (ts == NULL)
            goto memory_failed;

        gst_adapter_copy(sink->adapter, (guint8 *) ts->data->data, 0, nbytes);

        XLALFrameAddINT4TimeSeriesProcData(frame, ts);

        if (XLALFrameWrite(frame, filename, -1) != 0)
            goto write_failed;

        XLALDestroyINT4TimeSeries(ts);
    }

    sink->current_pos += nbytes;
    g_free(channame);
    g_free(filename);

    return TRUE;

memory_failed:
    {
        GST_ELEMENT_ERROR(
            sink, RESOURCE, NO_SPACE_LEFT,
            ("Error when creating timeseries in function %s", __FUNCTION__),
            GST_ERROR_SYSTEM);
        g_free(channame);
        g_free(filename);
        return FALSE;
    }

write_failed:
    {
        GST_ELEMENT_ERROR(
            sink, RESOURCE, WRITE,
            ("Could not write frame file: %s", filename),
            GST_ERROR_SYSTEM);
        g_free(channame);
        g_free(filename);
        return FALSE;
    }
}

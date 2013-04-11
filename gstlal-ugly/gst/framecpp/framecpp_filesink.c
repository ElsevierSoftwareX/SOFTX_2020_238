#include <glib.h>
#include <glib-object.h>
#include <stdio.h>
#include <math.h>
#include <gst/gst.h>
#include <gstlal/gstlal_tags.h>
#include <framecpp_filesink.h>

// grabbed from lalframe_sink
enum {
    PROP_0,
    PROP_PATH,
    PROP_FRAME_TYPE,
    PROP_DURATION,
    PROP_CLEAN_TIMESTAMPS,
    PROP_STRICT_TIMESTAMPS,
    PROP_DIR_DIGITS,
    PROP_INSTRUMENT,
};

static void set_property(GObject *object, guint prop_id,
                         const GValue *value, GParamSpec *pspec);
static void get_property(GObject *object, guint prop_id,
                         GValue *value, GParamSpec *pspec);

static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE(
        "sink",
        GST_PAD_SINK,
        GST_PAD_ALWAYS,
        GST_STATIC_CAPS(
            "application/x-igwd-frame, " \
            "framed = (boolean) true" 
        )
);

GST_BOILERPLATE(
        FRAMECPPFilesink,
        framecpp_filesink,
        GstBin,
        GST_TYPE_BIN
);

/*
* base_init()
*/

static void framecpp_filesink_base_init(gpointer gclass)
{
}

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

    g_object_class_install_property(
        gobject_class, PROP_FRAME_TYPE,
        g_param_spec_string(
            "frame-type", "Frame type.",
            "Type of frame, a description of its contents", "test_frame",
            (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)
            )
        );

    g_object_class_install_property(
        gobject_class, PROP_INSTRUMENT,
        g_param_spec_string(
            "instrument", "Observatory string.",
            "The IFO, like H1, L1, V1, etc.", "H1",
            (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)
            )
        );

}

// XXX It might decide to pass the probe hanlder context as a void **.
// Hopefully this will just do a cast.
static gboolean probeEventHandler(GstPad *pad, GstEvent *event, gpointer data) {
    FRAMECPPFilesink *element = FRAMECPP_FILESINK(gst_pad_get_parent(pad));

    if (GST_EVENT_TYPE(event)==GST_EVENT_TAG) {
        GstTagList *tag_list;
        gst_event_parse_tag(event, &tag_list);
        gchar *value = NULL;
        if (gst_tag_list_get_string(tag_list, GSTLAL_TAG_INSTRUMENT, &value)){
            GST_DEBUG("setting instrument to %s", value);
            g_object_set(G_OBJECT(element), "instrument", value, NULL);
        }
    }

    gst_object_unref(element);

    return TRUE;
}

static gboolean probeBufferHandler(GstPad *pad, GstBuffer *buffer, gpointer data) {
    FRAMECPPFilesink *element = FRAMECPP_FILESINK(gst_pad_get_parent(pad));
    guint timestamp, end_time, duration;
    gchar *newloc;
    gchar *instrument, *frame_type;

    g_object_get(G_OBJECT(element), "instrument", &instrument,
        "frame_type", &frame_type, NULL);

    if (!instrument || !frame_type) {
        // XXX Error message or event or something.
        GST_INFO("returning false.");
        return FALSE;
    }
    timestamp = GST_BUFFER_TIMESTAMP(buffer)/GST_SECOND;
    end_time = gst_util_uint64_scale_ceil(GST_BUFFER_TIMESTAMP(buffer) + GST_BUFFER_DURATION(buffer), 1, GST_SECOND);
    duration = end_time - timestamp;
    newloc = g_strdup_printf("%s-%s-%d-%d.gwf", instrument, 
        frame_type, timestamp, duration); 

    // Must cast the multifilesink as a GObject first.  
    GST_DEBUG("setting write location to %s", newloc);
    g_object_set(G_OBJECT(element->mfs), "location", newloc, NULL);

    g_free(newloc);

    gst_object_unref(element);

    return TRUE;
}

/*
 * The initializer
 */
static void framecpp_filesink_init(FRAMECPPFilesink *element, FRAMECPPFilesinkClass *kclass)
{
    // Create the multifilesink element.
    GstElement *multifilesink = gst_element_factory_make("multifilesink", NULL);
    g_object_set(G_OBJECT(multifilesink), "sync", FALSE, "async", FALSE, NULL);
    // Set the framecpp_filesink element's mfs property.
    element->mfs = multifilesink;

    // Add the multifilesink to the bin.
    gst_bin_add(GST_BIN(element), multifilesink);

    // Add the ghostpad 
    GstPad *sink = gst_element_get_static_pad(multifilesink, "sink");
    GstPad *sink_ghost = gst_ghost_pad_new_from_template("sink", sink, gst_element_class_get_pad_template(GST_ELEMENT_CLASS(G_OBJECT_GET_CLASS(element)),"sink"));
    gst_element_add_pad(GST_ELEMENT(element), sink_ghost);

    gst_object_unref(sink);

    // add event and buffer probes
    gst_pad_add_event_probe(sink_ghost, G_CALLBACK(probeEventHandler), NULL);
    gst_pad_add_buffer_probe(sink_ghost, G_CALLBACK(probeBufferHandler), NULL); 
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
    // XXX Check this.
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
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
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
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
    GST_OBJECT_UNLOCK(object);
}

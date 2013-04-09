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
            "framed = (bool) true" 
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
    // XXX Not using bin_class?
    GstBinClass *bin_class = GST_BIN_CLASS(klass);

    gst_element_class_set_details_simple(element_class, "Write frame files from muxer", "Sink/File", "Comment", "Branson Stephens <stephenb@uwm.edu>");
    gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));

    GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

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

}

// XXX Not sure if this struct thingy is going to work.
// XXX Should the struct actually be in the .h file?
typedef struct 
{
    // How do you declare a string in c?
    gchar *instrument;
    gchar *frame_type;
    // To store the multifilesink object.
    GstElement *mfs;
} probe_handler_context;

// XXX It might decide to pass the probe hanlder context as a void **.
// Hopefully this will just do a cast.
static gint probeEventHandler(GstEvent *event, gpointer c) {
    // Cast the user data context object.
    probe_handler_context *hc;
    hc = (probe_handler_context *) c;
    
    if (GST_EVENT_TYPE(event)==GST_EVENT_TAG) {
        GstTagList *tag_list;
        gst_event_parse_tag(event, &tag_list);
        gchar *value = NULL;
        if (gst_tag_list_get_string(tag_list, GSTLAL_TAG_INSTRUMENT, &value)){
            hc->instrument = value;  
        }
    }
    return 1;
}

static gint probeBufferHandler(GstBuffer *buffer, gpointer c) {
    // Cast the user data context object.
    probe_handler_context *hc;
    hc = (probe_handler_context *) c;

    if (!(hc->instrument) || !(hc->frame_type)) {
        // XXX Error message or event or something.
        return 1;
    }
    guint offset_diff, timestamp, end_time, duration;
    gchar *newloc;
    offset_diff = buffer->offset_end - buffer->offset;
    timestamp = buffer->timestamp/GST_SECOND;
    end_time = buffer->timestamp + buffer->duration;
    end_time = ceil(((float)end_time)/((float)GST_SECOND));
    duration = end_time - timestamp;
    // Need to store the new location as a GValue
    sprintf(newloc, "%s-%s-%d-%d.gwf", hc->instrument, 
        hc->frame_type, timestamp, duration);
    GValue newLocation = {0};
    g_value_init(&newLocation, G_TYPE_STRING);
    g_value_set_string(&newLocation, newloc);

    // Must cast the multifilesink as a GObject first.  
    g_object_set_property(G_OBJECT(hc->mfs), "location", &newLocation);
    g_value_unset(&newLocation);

    return 1;
}

/*
 * The initializer
 */
static void framecpp_filesink_init(FRAMECPPFilesink *element, FRAMECPPFilesinkClass *kclass)
{
    // Note: mkmultifilesink adds properties sync = False, async = False
    GstElement *multifilesink = gst_element_factory_make("multifilesink", NULL);
    GValue gf = {0};
    gboolean gbf = FALSE;
    g_value_init(&gf, G_TYPE_BOOLEAN);
    g_value_set_boolean(&gf, gbf);
    g_object_set_property(G_OBJECT(multifilesink), "sync", &gf);
    g_object_set_property(G_OBJECT(multifilesink), "async", &gf);
    GstPad *sink = gst_element_get_static_pad(multifilesink, "sink");
    GstPad *sink_ghost = gst_ghost_pad_new_from_template("sink", sink, gst_element_class_get_pad_template(GST_ELEMENT_CLASS(G_OBJECT_GET_CLASS(element)),"sink"));

    gst_object_unref(sink);

    gst_bin_add(GST_BIN(element), multifilesink);

    gst_element_add_pad(GST_ELEMENT(element), sink_ghost);

    // XXX Now I've got a pad connected to a multifilesink
    // add event and buffer probes

    // Instantiate probe hander context.
    probe_handler_context c;
    c.instrument = NULL;
    c.frame_type = element->frame_type;
    // Add the multifilesink object to the probe_handler_context.
    // Note.  These are both pointers to a GstElement.
    c.mfs = multifilesink;
    // Add probes.
    probe_handler_context c2;
    c2 = c;
    gst_pad_add_event_probe(sink_ghost, G_CALLBACK(probeEventHandler), &c);
    gst_pad_add_buffer_probe(sink_ghost, G_CALLBACK(probeBufferHandler), &c2); 
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
    switch (prop_id) {
    case PROP_FRAME_TYPE:
        g_free(sink->frame_type);
        sink->frame_type = g_strdup(g_value_get_string(value));
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

static void get_property(GObject *object, guint prop_id, 
                         GValue *value, GParamSpec *pspec)
{
    FRAMECPPFilesink *sink = FRAMECPP_FILESINK(object);

    switch (prop_id) {
    case PROP_FRAME_TYPE:
        g_value_set_string(value, sink->frame_type);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

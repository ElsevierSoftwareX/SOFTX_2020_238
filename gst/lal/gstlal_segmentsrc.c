/*
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */

/*
 * stuff from the C library
 */


/*
 * stuff from gobject/gstreamer
 */

#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasesrc.h>

/*
 * stuff from gstlal
 */

#include <gstlal.h>
#include <gstlal_segmentsrc.h>


/*
 * ============================================================================
 *
 *                           GStreamer Boiler Plate
 *
 * ============================================================================
 */

static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE(
    "src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS(
        "audio/x-raw-int, " \
        "rate = (int) [1, MAX], " \
        "channels = (int) 1, " \
        "endianness = (int) BYTE_ORDER, " \
        "width = (int) 8," \
        "depth = (int) 1," \
        "signed = false"
    )
);

GST_BOILERPLATE(
    GSTLALSegmentSrc,
    gstlal_segmentsrc,
    GstBaseSrc,
    GST_TYPE_BASE_SRC
);

enum property {
    ARG_SEGMENT_LIST = 1
};

/*
 * ============================================================================
 *
 *                          GObject Method Overrides
 *
 * ============================================================================
 */

/*
 * set_property()
 */

static void set_property(GObject *object, enum property prop_id, const GValue *value, GParamSpec *pspec)
{
    GSTLALSegmentSrc        *element = GSTLAL_SEGMENTSRC(object);

    GST_OBJECT_LOCK(element);

    switch (prop_id) {
        case ARG_SEGMENT_LIST:
            g_value_array_free(element->segment_list);
            element->segment_list = g_value_get_boxed(value);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;

    }

    GST_OBJECT_UNLOCK(element);
}


/*
 * get_property()
 */

static void get_property(GObject *object, enum property prop_id, GValue *value, GParamSpec *pspec)
{
    GSTLALSegmentSrc        *element = GSTLAL_SEGMENTSRC(object);

    GST_OBJECT_LOCK(element);

    switch (prop_id) {
        case ARG_SEGMENT_LIST:
            g_value_set_boxed(value, element->segment_list);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;

    }

    GST_OBJECT_UNLOCK(element);
}


/*
 * finalize()
 */

static void finalize(GObject *object)
{
    GSTLALSegmentSrc        *element = GSTLAL_SEGMENTSRC(object);

    /*
     * free resources
     */
    g_value_array_free(element->segment_list);


    /*
     * chain to parent class' finalize() method
     */

    G_OBJECT_CLASS(parent_class)->finalize(object);
}


/*
 * base_init()
 */

static void gstlal_segmentsrc_base_init(gpointer gclass)
{
    GstElementClass     *element_class = GST_ELEMENT_CLASS(gclass);

    gst_element_class_set_details_simple(element_class, "List of on times and off times", "Source/Audio", "The output is a buffer of boolean values specifying when a list of segments are on and off.", "Collin Capano <collin.capano@ligo.org>");
    gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
}


/*
 * class_init()
 */

static void gstlal_segmentsrc_class_init(GSTLALSegmentSrcClass *klass)
{
    GObjectClass        *gobject_class = G_OBJECT_CLASS(klass);

    gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
    gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
    gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

    g_object_class_install_property(
        gobject_class,
        ARG_SEGMENT_LIST,
        g_param_spec_value_array(
            "segment-list",
            "Segment List",
            "List of Segments. This is a Nx2 array where N (the rows) is the number of segments. The columns are the start and stop times of each segment.",
			g_param_spec_value_array(
				"segment",
				"[start, stop)",
				"Start and stop time of segment.",
				g_param_spec_uint64(
					"time",
					"Time",
					"Time (in nanoseconds)",
					0, G_MAXUINT64, 0,
					G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
				),
				G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
			),
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);
}


/*
 * init()
 */

static void gstlal_segmentsrc_init(GSTLALSegmentSrc *segment_src, GSTLALSegmentSrcClass *klass)
{
    segment_src->segment_list = g_value_array_new(0);
}

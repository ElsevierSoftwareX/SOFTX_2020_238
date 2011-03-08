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

#include <string.h>

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
        "depth = (int) 8," \
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
    ARG_SEGMENT_LIST = 1,
    ARG_INVERT_OUTPUT
};



static GstCaps *segments_to_caps(gint rate)
{
    GstCaps *caps;

    /* 
     * FIXME, it would be nice to get say, the rate from the segments...
     * but that seems hard.  So for now everything is hardcoded
     * FIXME this function isn't even used yet, we get caps from downstream
     */ 

    caps = gst_caps_new_simple(
        "audio/x-raw-int",
        "rate", G_TYPE_INT, rate,
        "channels", G_TYPE_INT, 1,
        "endianness", G_TYPE_INT, G_BYTE_ORDER,
        "width", G_TYPE_INT, 8,
        "depth", G_TYPE_INT, 8,
	"signed", G_TYPE_BOOLEAN, FALSE,
        NULL
    );
    return caps;
}


/*
 * ============================================================================
 *
 *                        GstBaseSrc Method Overrides
 *
 * ============================================================================
 */


/*
 * start()
 */


static gboolean start(GstBaseSrc *object)
{
    return TRUE;
}


/*
 * stop()
 */


static gboolean stop(GstBaseSrc *object)
{
    //GSTLALSegmentSrc        *element = GSTLAL_SEGMENTSRC(object);

    return TRUE;
}


static guint sample_size(GstBaseSrc *basesrc)
{
    //GSTLALSegmentSrc        *element = GSTLAL_SEGMENTSRC(basesrc);
    return 8; /* FIXME The only width supported, unsigned char */
}

/*
 * create()
 */

static int mark_segments(GstBaseSrc *basesrc, GstBuffer *buffer)
{
    //GSTLALSegmentSrc        *element = GSTLAL_SEGMENTSRC(basesrc);
    return 0;
}

static GstFlowReturn create(GstBaseSrc *basesrc, guint64 offset, guint size, GstBuffer **buffer)
{
    GSTLALSegmentSrc        *element = GSTLAL_SEGMENTSRC(basesrc);
    GstFlowReturn result;
    gulong blocksize = gst_base_src_get_blocksize(basesrc);
    guint samplesize = sample_size(basesrc);
    guint64 numsamps = blocksize / samplesize;

    /*
     * Bail if the requested block size doesn't correspond to integer samples 
     */

    if (blocksize % samplesize) {
        GST_ERROR_OBJECT(element,
            "block size %lu is not an integer multiple of the sample size %lu",
            blocksize, samplesize);
        return GST_FLOW_ERROR;
    }

    /*
     * Allocate the buffer of ones or zeros depending on the invert property
     */

    result = gst_pad_alloc_buffer(GST_BASE_SRC_PAD(basesrc), basesrc->offset, blocksize, GST_PAD_CAPS(GST_BASE_SRC_PAD(basesrc)), buffer);

    if (element->invert_output)
        memset(GST_BUFFER_DATA(*buffer), 0, blocksize);
    else {
        guint8 * d = (guint8 *) GST_BUFFER_DATA(*buffer);
        for (guint32 i = 0; i < numsamps; i++) d[i] = G_MAXUINT8;
    }

    /* 
     * Mark the buffer according to the segments
     */
   
    mark_segments(basesrc, *buffer);

    /* 
     * update the offsets, timestamps etc 
     */

    GST_BUFFER_OFFSET_END(*buffer) = GST_BUFFER_OFFSET(*buffer) + numsamps;

    GST_BUFFER_TIMESTAMP(*buffer) = basesrc->segment.start
        + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET(*buffer), GST_SECOND, element->rate);

    GST_BUFFER_DURATION(*buffer) = basesrc->segment.start
        + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET_END(*buffer), GST_SECOND, element->rate)
        - GST_BUFFER_TIMESTAMP(*buffer);

    /* FIXME Huh? */
    if(basesrc->offset == 0)
        GST_BUFFER_FLAG_SET(*buffer, GST_BUFFER_FLAG_DISCONT);

    basesrc->offset += numsamps;

    return GST_FLOW_OK;
}


/*
 * is_seekable()
 */


static gboolean is_seekable(GstBaseSrc *object)
{
    return TRUE;
}


/*
 * do_seek()
 */


static gboolean do_seek(GstBaseSrc *basesrc, GstSegment *segment)
{
    GSTLALSegmentSrc        *element = GSTLAL_SEGMENTSRC(basesrc);
    GstClockTime epoch;

    /*
     * Parse the segment
     */

    if((GstClockTime) segment->start == GST_CLOCK_TIME_NONE) {
        GST_ELEMENT_ERROR(element, RESOURCE, SEEK, ("seek failed:  start time is required"), (NULL));
        return FALSE;
    }
    epoch = segment->start;

    /*
     * Try doing the seek
     */

    /*
     * Done
     */

    basesrc->offset = 0;
    return TRUE;
}


/*
 * query
 */


static gboolean query(GstBaseSrc *basesrc, GstQuery *query)
{
    //GSTLALSegmentSrc        *element = GSTLAL_SEGMENTSRC(basesrc);

    /* FIXME:  this is copy-and-pasted from frammesrc and needs to be reworked for this element */
#if 0
	switch(GST_QUERY_TYPE(query)) {
	case GST_QUERY_FORMATS:
		gst_query_set_formats(query, 5, GST_FORMAT_DEFAULT, GST_FORMAT_BYTES, GST_FORMAT_TIME, GST_FORMAT_BUFFERS, GST_FORMAT_PERCENT);
		break;

	case GST_QUERY_CONVERT: {
		GstFormat src_format, dest_format;
		gint64 src_value, dest_value;
		guint64 offset;

		gst_query_parse_convert(query, &src_format, &src_value, &dest_format, &dest_value);

		switch(src_format) {
		case GST_FORMAT_DEFAULT:
		case GST_FORMAT_TIME:
			if(src_value < basesrc->segment.start) {
				GST_DEBUG("requested time precedes start of segment, clipping to start of segment");
				offset = 0;
			} else
				offset = gst_util_uint64_scale_int_round(src_value - basesrc->segment.start, element->rate, GST_SECOND);
			break;

		case GST_FORMAT_BYTES:
			offset = src_value / (element->width / 8);
			break;

		case GST_FORMAT_BUFFERS:
			offset = gst_util_uint64_scale_int_round(src_value, gst_base_src_get_blocksize(basesrc), element->width / 8);
			break;

		case GST_FORMAT_PERCENT:
			if(src_value < 0) {
				GST_DEBUG("requested percentage < 0, clipping to 0");
				offset = 0;
			} else if(src_value > 100) {
				GST_DEBUG("requested percentage > 100, clipping to 100");
				offset = basesrc->segment.stop - basesrc->segment.start;
			} else
				offset = gst_util_uint64_scale_int_round(basesrc->segment.stop - basesrc->segment.start, src_value, 100);
			offset = gst_util_uint64_scale_int_round(offset, element->rate, GST_SECOND);
			break;

		default:
			g_assert_not_reached();
			return FALSE;
		}
		switch(dest_format) {
		case GST_FORMAT_DEFAULT:
		case GST_FORMAT_TIME:
			dest_value = basesrc->segment.start + gst_util_uint64_scale_int_round(offset, GST_SECOND, element->rate);
			break;

		case GST_FORMAT_BYTES:
			dest_value = offset * (element->width / 8);
			break;

		case GST_FORMAT_BUFFERS:
			dest_value = gst_util_uint64_scale_int_ceil(offset, element->width / 8, gst_base_src_get_blocksize(basesrc));
			break;

		case GST_FORMAT_PERCENT:
			dest_value = gst_util_uint64_scale_int_round(offset, 100, gst_util_uint64_scale_int_round(basesrc->segment.stop - basesrc->segment.start, element->rate, GST_SECOND));
			break;

		default:
			g_assert_not_reached();
			return FALSE;
		}

		gst_query_set_convert(query, src_format, src_value, dest_format, dest_value);

		break;
	}

	default:
		return parent_class->query(basesrc, query);
	}
#endif

	return TRUE;
}


/*
 * check_get_range()
 */


static gboolean check_get_range(GstBaseSrc *basesrc)
{
	return TRUE;
}

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
        case ARG_INVERT_OUTPUT:
            element->invert_output = g_value_get_boolean(value);
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
        case ARG_INVERT_OUTPUT:
            g_value_set_boolean(value, element->invert_output);
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
 * set_caps()
 */


static gboolean set_caps(GstBaseSrc *src, GstCaps *caps)
{
    GSTLALSegmentSrc *element = GSTLAL_SEGMENTSRC(src);
    GstStructure *s;
    gint rate;
    gboolean success = TRUE;

    s = gst_caps_get_structure(caps, 0);

    if(!gst_structure_get_int(s, "rate", &rate)) {
        GST_DEBUG_OBJECT(element, "unable to parse rate from %" GST_PTR_FORMAT,
                         caps);
        success = FALSE;
    }

    if(success) {
        if(rate != element->rate)
		GST_DEBUG_OBJECT(element, "rate changed, but no signal was emitted because it is not implmented: %d -> %d ", element->rate, rate);
        element->rate = rate;
    }

	return success;
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
    GstBaseSrcClass *gstbasesrc_class = GST_BASE_SRC_CLASS(klass);

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

    g_object_class_install_property(
        gobject_class,
        ARG_INVERT_OUTPUT,
        g_param_spec_boolean(
            "invert-output",
            "Invert output",
            "False = output is high in segments (default), True = output is low in segments",
            FALSE,
            G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
        )
    );

    /*
     * GstBaseSrc method overrides
     */

    gstbasesrc_class->start = GST_DEBUG_FUNCPTR(start);
    gstbasesrc_class->stop = GST_DEBUG_FUNCPTR(stop);
    gstbasesrc_class->create = GST_DEBUG_FUNCPTR(create);
    gstbasesrc_class->is_seekable = GST_DEBUG_FUNCPTR(is_seekable);
    gstbasesrc_class->do_seek = GST_DEBUG_FUNCPTR(do_seek);
    /* FIXME, write a proper query function */
    /* gstbasesrc_class->query = GST_DEBUG_FUNCPTR(query); */
    gstbasesrc_class->check_get_range = GST_DEBUG_FUNCPTR(check_get_range);
    gstbasesrc_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
}


/*
 * init()
 */

static void gstlal_segmentsrc_init(GSTLALSegmentSrc *segment_src, GSTLALSegmentSrcClass *klass)
{
    segment_src->segment_list = g_value_array_new(0);
    segment_src->rate = 0;
}

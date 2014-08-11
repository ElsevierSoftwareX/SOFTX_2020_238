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

#include <gstlal/gstlal.h>
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
        "signed = true"
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


/*
 * ======================================================================
 *
 * utility functions
 *
 * ======================================================================
 */


/*
 * return the sample size (this is always 8 for now)
 */


static guint sample_size(GstBaseSrc *basesrc)
{
    //GSTLALSegmentSrc        *element = GSTLAL_SEGMENTSRC(basesrc);
    return 1; /* FIXME The only size supported, guint8*/
}


/*
 * round a time to the nearest sample based on sample rate
 */


static guint64 round_to_nearest_sample(GstBaseSrc *basesrc, guint64 val)
{
    GSTLALSegmentSrc        *element = GSTLAL_SEGMENTSRC(basesrc);
    return gst_util_uint64_scale_int_round(val, 1, element->rate * GST_SECOND) * element->rate * GST_SECOND;
}


/* FIXME a place holder to somehow infer caps from segments??? maybe useless */
#if 0
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
        "signed", G_TYPE_BOOLEAN, TRUE,
        NULL
    );
    return caps;
}
#endif


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
    return TRUE;
}


/*
 * Mark buffer according to segment
 */


static int mark_segment(GstBaseSrc *basesrc, GstBuffer *buffer, guint64 start, guint64 stop)
{
    GSTLALSegmentSrc *element = GSTLAL_SEGMENTSRC(basesrc);
    guint startix=0;
    guint stopix =0;
    gint8 *data = NULL;

    if (start > GST_BUFFER_TIMESTAMP(buffer))
        startix = round_to_nearest_sample(basesrc, start - GST_BUFFER_TIMESTAMP(buffer)) / element->rate / GST_SECOND;
    else
        startix = 0;

    if (stop > GST_BUFFER_TIMESTAMP(buffer))
        stopix = round_to_nearest_sample(basesrc, stop - GST_BUFFER_TIMESTAMP(buffer)) / element->rate / GST_SECOND;
    else
        stopix = 0;

    data = (gint8 *) GST_BUFFER_DATA(buffer);

    if (element->invert_output)
        for (guint32 i = startix; i < stopix; i++) data[i] = 0;
    else
        for (guint32 i = startix; i < stopix; i++) data[i] = G_MAXINT8;

    return 0;
}


/*
 * Mark buffer according to segment list
 */


static int mark_segments(GstBaseSrc *basesrc, GstBuffer *buffer, guint64 start, guint64 stop)
{
    GSTLALSegmentSrc        *element = GSTLAL_SEGMENTSRC(basesrc);
    struct gstlal_segment_list *seglist = element->seglist;
    if (!seglist) return 0; /* FIXME handle no segment lists */
    guint rows = seglist->length;
    guint64 segstart, segstop;

    /* FIXME provide a bailout and a sensible starting point if you have sorted and coalesced segents */
    /* This is ridiculous, but doesn't require sorted or coalesced segments.  Could some fancy data structure help? */
    /* FIXME switch to using gstlal segment routines */
    for (guint i = 0; i < rows; i++) {
        segstart = seglist->segments[i].start;
        segstop = seglist->segments[i].stop;
        if ((segstart >= start) && (segstart < stop) && (segstop < stop))
            mark_segment(basesrc, buffer, segstart, segstop);
        if ((segstart >= start) && (segstart < stop) && (segstop >= stop))
            mark_segment(basesrc, buffer, segstart, stop);
        if ((segstop >= start) && (segstop < stop) && (segstart < start))
            mark_segment(basesrc, buffer, start, segstop);
        if ((segstart <= start) && (segstop >= stop))
            mark_segment(basesrc, buffer, start, stop);
    }

    return 0;
}


/*
 * create()
 */


static GstFlowReturn create(GstBaseSrc *basesrc, guint64 offset, guint size, GstBuffer **buffer)
{
    GSTLALSegmentSrc *element = GSTLAL_SEGMENTSRC(basesrc);
    GstFlowReturn result = GST_FLOW_OK;
    gulong blocksize = gst_base_src_get_blocksize(basesrc);
    guint samplesize = sample_size(basesrc);
    guint64 numsamps = blocksize / samplesize;
    guint64 start, stop;

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
    if(result != GST_FLOW_OK)
        return result;

    gint8 *d = (gint8 *) GST_BUFFER_DATA(*buffer);
    if (element->invert_output)
        for (guint32 i = 0; i < numsamps; i++) d[i] = G_MAXINT8;
    else {
        for (guint32 i = 0; i < numsamps; i++) d[i] = 0;
    }

    /*
     * update the offsets, timestamps etc
     */

    GST_BUFFER_OFFSET_END(*buffer) = GST_BUFFER_OFFSET(*buffer) + numsamps;

    start = GST_BUFFER_TIMESTAMP(*buffer) = basesrc->segment.start
        + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET(*buffer), GST_SECOND, element->rate);

    stop = basesrc->segment.start
        + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET_END(*buffer), GST_SECOND, element->rate);

    GST_BUFFER_DURATION(*buffer) = stop - start;
    
    /*
     * Mark the buffer according to the segments
     */

    mark_segments(basesrc, *buffer, start, stop);

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

    /*
     * Parse the segment
     */

    if((GstClockTime) segment->start == GST_CLOCK_TIME_NONE) {
        GST_ELEMENT_ERROR(element, RESOURCE, SEEK, ("seek failed:  start time is required"), (NULL));
        return FALSE;
    }

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
	
	GSTLALSegmentSrc        *element = GSTLAL_SEGMENTSRC(basesrc);

	switch(GST_QUERY_TYPE(query)) {

		case GST_QUERY_FORMATS:
			gst_query_set_formats(query, 5, GST_FORMAT_DEFAULT, GST_FORMAT_TIME, GST_FORMAT_BYTES, GST_FORMAT_BUFFERS, GST_FORMAT_PERCENT);
			break;

		case GST_QUERY_CONVERT: {
			GstFormat src_format, dest_format;
			gint64 src_value, dest_value;
			guint64 offset;

			gst_query_parse_convert(query, &src_format, &src_value, &dest_format, &dest_value);
	

			if (src_format == dest_format) {
				dest_value = src_value;
				break;
			}
			
			switch(src_format) {
				case GST_FORMAT_DEFAULT:
				case GST_FORMAT_TIME:
					if(src_value < basesrc->segment.start) {
						GST_DEBUG("requested time precedes start of segment, clipping to start of segment");
						offset = 0;
					} else
						offset = gst_util_uint64_scale_int_round(src_value, element->rate, GST_SECOND);
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
					} else if(src_value > GST_FORMAT_PERCENT_MAX) {
						GST_DEBUG("requested percentage > 100, clipping to 100");
						offset = basesrc->segment.stop - basesrc->segment.start;
					} else
						offset = gst_util_uint64_scale_int_round(basesrc->segment.stop - basesrc->segment.start, src_value, GST_FORMAT_PERCENT_MAX);
					offset = gst_util_uint64_scale_int_round(offset, element->rate, GST_SECOND);
					break;

				default:
					g_assert_not_reached();
					return FALSE;
				}
			
			switch(dest_format) {
				case GST_FORMAT_DEFAULT:
				case GST_FORMAT_TIME:
					dest_value = gst_util_uint64_scale_int_round(offset, GST_SECOND, element->rate);
					break;

				case GST_FORMAT_BYTES:
					dest_value = offset * (element->width / 8);
					break;

				case GST_FORMAT_BUFFERS:
					dest_value = gst_util_uint64_scale_int_ceil(offset, element->width / 8, gst_base_src_get_blocksize(basesrc));
					break;

				case GST_FORMAT_PERCENT:
					dest_value = gst_util_uint64_scale_int_round(offset, GST_FORMAT_PERCENT_MAX, gst_util_uint64_scale_int_round(basesrc->segment.stop - basesrc->segment.start, element->rate, GST_SECOND));
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
 * Compare function to sort segment list
 */


gint seg_compare_func(gconstpointer a, gconstpointer b)
{
    GValueArray *rowa = (GValueArray *) g_value_get_boxed((GValue *) a);
    GValueArray *rowb = (GValueArray *) g_value_get_boxed((GValue *) b);
    guint64 astart = g_value_get_uint64(g_value_array_get_nth(rowa, 0));
    guint64 bstart = g_value_get_uint64(g_value_array_get_nth(rowb, 0));

    if (astart <  bstart) return -1;
    if (astart == bstart) return 0;
    if (astart >  bstart) return 1;
    return 0;
}


/*
 * set_property()
 */


static void set_property(GObject *object, enum property prop_id, const GValue *value, GParamSpec *pspec)
{
    GSTLALSegmentSrc        *element = GSTLAL_SEGMENTSRC(object);

    GST_OBJECT_LOCK(element);

    switch (prop_id) {
        case ARG_SEGMENT_LIST:
            element->seglist = gstlal_segment_list_from_g_value_array(g_value_get_boxed(value));
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
            g_mutex_lock(element->segment_matrix_lock);
            if(element->seglist)
                g_value_take_boxed(value, g_value_array_from_gstlal_segment_list(element->seglist));
            /* FIXME:  else? */
            g_mutex_unlock(element->segment_matrix_lock);
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

    if (element->seglist) gstlal_segment_list_free(element->seglist);
    g_mutex_free(element->segment_matrix_lock);

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
}


/*
 * class_init()
 */

static void gstlal_segmentsrc_class_init(GSTLALSegmentSrcClass *klass)
{
    GstBaseSrcClass *gstbasesrc_class = GST_BASE_SRC_CLASS(klass);
    GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
    GObjectClass    *gobject_class = G_OBJECT_CLASS(klass);

    gst_element_class_set_details_simple(element_class, "List of on times and off times", "Source/Audio", "The output is a buffer of boolean values specifying when a list of segments are on and off.", "Collin Capano <collin.capano@ligo.org>");

    gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
    gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
    gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

    gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));

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
    gstbasesrc_class->query = GST_DEBUG_FUNCPTR(query);
    gstbasesrc_class->check_get_range = GST_DEBUG_FUNCPTR(check_get_range);
    gstbasesrc_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
}


/*
 * init()
 */

static void gstlal_segmentsrc_init(GSTLALSegmentSrc *segment_src, GSTLALSegmentSrcClass *klass)
{
    segment_src->seglist = NULL;
    segment_src->rate = 0;
    /* FIXME hardcoded width */
    segment_src->width = 8;
    segment_src->segment_matrix_lock = g_mutex_new();
    gst_base_src_set_format(GST_BASE_SRC(segment_src), GST_FORMAT_TIME);
}

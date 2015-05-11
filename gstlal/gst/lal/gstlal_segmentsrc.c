/*
 * lal_segmentsrc
 *
 * Copyright (C) ??
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


/**
 * SECTION:gstlal_segmentsrc
 * @short_description:  The output is a buffer of boolean values specifying when a list of segments are on and off.
 *
 * Generates a one-channel boolean-valued stream from the #segment-list
 * property, which must be set to an array of two-element arrays of
 * start/stop time pairs.  If #invert-output is %False the start/stop pairs
 * are taken to give intervals when the output is %True, otherwise they are
 * taken to gve intervals when the output is %False.
 *
 * The element can be seeked, but when seeked the requested start time must
 * be set.
 *
 * Reviewed:  a2d52f933cd71abc2effa66b46d030ee605e7cea 2014-08-13 K.
 * Cannon, J.  Creighton, B. Sathyaprakash.
 */


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
    ARG_SEGMENT_LIST = 1,
    ARG_INVERT_OUTPUT
};


/*
 * ============================================================================
 *
 *                        GstBaseSrc Method Overrides
 *
 * ============================================================================
 */


/*
 * Mark buffer according to segment list
 */


static int mark_segments(GSTLALSegmentSrc *element, GstBuffer *buffer)
{
    guint8 *data = GST_BUFFER_DATA(buffer);
    GstClockTime start = GST_BUFFER_TIMESTAMP(buffer);
    GstClockTime stop = GST_BUFFER_TIMESTAMP(buffer) + GST_BUFFER_DURATION(buffer);
    gint i;

    /* This is ridiculous, but doesn't require sorted or coalesced
     * segments.  Could some fancy data structure help? */
    for (i = 0; i < element->seglist->length; i++) {
    	/* clip segment to buffer */
        GstClockTime segstart = CLAMP(element->seglist->segments[i].start, start, stop);
        GstClockTime segstop  = CLAMP(element->seglist->segments[i].stop,  start, stop);

	/* convert to samples */
	guint64 startix = gst_util_uint64_scale_int_round(segstart - start, element->rate, GST_SECOND);
	guint64 stopix  = gst_util_uint64_scale_int_round(segstop  - start, element->rate, GST_SECOND);

	/* set samples */
	for(; startix < stopix; startix++)
		data[startix] = element->invert_output ? 0 : 0x80;
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
    guint64 numsamps = blocksize;
    GstClockTime start = basesrc->segment.start + gst_util_uint64_scale_int_round(basesrc->offset, GST_SECOND, element->rate);
    GstClockTime stop = basesrc->segment.start + gst_util_uint64_scale_int_round(basesrc->offset + numsamps, GST_SECOND, element->rate);

    *buffer = NULL;	/* just in case */

    /*
     * Check for EOS
     */

    if(GST_CLOCK_TIME_IS_VALID(basesrc->segment.stop) && start >= (GstClockTime) basesrc->segment.stop)
        return GST_FLOW_UNEXPECTED;

    /*
     * Allocate the buffer of ones or zeros depending on the invert property
     */

    result = gst_pad_alloc_buffer(GST_BASE_SRC_PAD(basesrc), basesrc->offset, blocksize, GST_PAD_CAPS(GST_BASE_SRC_PAD(basesrc)), buffer);
    if(result != GST_FLOW_OK)
        return result;

    memset(GST_BUFFER_DATA(*buffer), element->invert_output ? 0x80 : 0x00, GST_BUFFER_SIZE(*buffer));

    /*
     * update the offsets, timestamps etc
     */

    GST_BUFFER_OFFSET_END(*buffer) = GST_BUFFER_OFFSET(*buffer) + numsamps;
    GST_BUFFER_TIMESTAMP(*buffer) = start;
    GST_BUFFER_DURATION(*buffer) = stop - start;

    /*
     * Mark the buffer according to the segments
     */

    mark_segments(element, *buffer);
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
					/* width is 8 bits */
					offset = src_value;
					break;

				case GST_FORMAT_BUFFERS:
					/* width is 8 bits */
					offset = src_value * gst_base_src_get_blocksize(basesrc);
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
					/* width is 8 bits */
					dest_value = offset;
					break;

				case GST_FORMAT_BUFFERS:
					/* width is 8 bits */
					dest_value = gst_util_uint64_scale_int_ceil(offset, 1, gst_base_src_get_blocksize(basesrc));
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
 * set_property()
 */


static void set_property(GObject *object, enum property prop_id, const GValue *value, GParamSpec *pspec)
{
    GSTLALSegmentSrc        *element = GSTLAL_SEGMENTSRC(object);

    GST_OBJECT_LOCK(element);

    switch (prop_id) {
        case ARG_SEGMENT_LIST:
            g_mutex_lock(element->segment_matrix_lock);
            gstlal_segment_list_free(element->seglist);
            element->seglist = gstlal_segment_list_from_g_value_array(g_value_get_boxed(value));
            g_mutex_unlock(element->segment_matrix_lock);
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
    GSTLALSegmentSrc *element = GSTLAL_SEGMENTSRC(object);

    /*
     * free resources
     */

    gstlal_segment_list_free(element->seglist);
    element->seglist = NULL;
    g_mutex_free(element->segment_matrix_lock);
    element->segment_matrix_lock = NULL;

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

    if(success)
        element->rate = rate;

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
            "List of Segments.  This is an Nx2 array where N (the rows) is the number of segments. The columns are the start and stop times of each segment.",
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
    segment_src->segment_matrix_lock = g_mutex_new();
    gst_base_src_set_format(GST_BASE_SRC(segment_src), GST_FORMAT_TIME);
}

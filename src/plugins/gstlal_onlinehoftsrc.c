/*
 * LAL online h(t) src element
 *
 * Copyright (C) 2008  Leo Singer
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


/*
 * ========================================================================
 *
 *                                  Preamble
 *
 * ========================================================================
 */


/*
 * stuff from the C library
 */


#include <string.h>


/*
 * stuff from gstreamer
 */


#include <gst/gst.h>
#include <gst/base/gstbasesrc.h>


/*
 * our own stuff
 */


#include <gstlal.h>
#include <gstlal_onlinehoftsrc.h>


/*
 * Parent class.
 */


static GstBaseSrcClass *parent_class = NULL;



/*
 * ============================================================================
 *
 *                                 Properties
 *
 * ============================================================================
 */


enum property {
	ARG_SRC_INSTRUMENT = 1
};


static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec)
{
	GSTLALOnlineHoftSrc *element = GSTLAL_ONLINEHOFTSRC(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_SRC_INSTRUMENT:
		g_free(element->instrument);
		element->instrument = g_value_dup_string(value);
		onlinehoft_destroy(element->tracker);
		element->tracker = NULL;
		element->needs_seek = TRUE;
		break;

	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *pspec)
{
	GSTLALOnlineHoftSrc *element = GSTLAL_ONLINEHOFTSRC(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_SRC_INSTRUMENT:
		g_value_set_string(value, element->instrument);
		break;

	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * ============================================================================
 *
 *                        GstBaseSrc Method Overrides
 *
 * ============================================================================
 */


/*
 * create()
 */


static GstFlowReturn create(GstBaseSrc *basesrc, guint64 offset, guint size, GstBuffer **buffer)
{
	GSTLALOnlineHoftSrc *element = GSTLAL_ONLINEHOFTSRC(basesrc);

	// If the online h(t) tracker has not been created (e.g. if the instrument
	// property has been changed), then try to create it.
	if (!element->tracker)
	{
		GST_INFO_OBJECT(element, "onlinehoft_create(\"%s\")", element->instrument);
		element->tracker = onlinehoft_create(element->instrument);

		if (!element->tracker)
		{
			GST_ERROR_OBJECT(element, "onlinehoft_create(\"%s\")", element->instrument);
			return GST_FLOW_ERROR;
		}

		// Send new instrument tag
		if (!gst_pad_push_event(
			GST_BASE_SRC_PAD(basesrc),
			gst_event_new_tag(
				gst_tag_list_new_full(
					GSTLAL_TAG_INSTRUMENT, element->instrument,
					GSTLAL_TAG_CHANNEL_NAME, onlinehoft_get_channelname(element->tracker),
					NULL
				)
			)
		)) {
			GST_ERROR_OBJECT(element, "Failed to push taglist");
		}
	}

	if (element->needs_seek)
	{
		// Do seek
		guint64 seek_start_seconds = basesrc->segment.start / GST_SECOND;
		GST_INFO_OBJECT(element, "onlinehoft_seek(tracker, %u)", seek_start_seconds);
		onlinehoft_seek(element->tracker, seek_start_seconds);
		element->needs_seek = FALSE;
	}

	uint16_t segment_mask;
	FrVect* frVect = onlinehoft_next_vect(element->tracker, &segment_mask);
	if (!frVect) return GST_FLOW_ERROR;

	GST_INFO_OBJECT(element, "segment_mask=0x%02X", segment_mask);

	guint64 gps_start_time = (guint64)frVect->GTime;

	gboolean was_nongap = segment_mask & 1;
	gboolean is_discontinuous = onlinehoft_was_discontinuous(element->tracker);
	uint8_t segment_num, last_segment_num;
	for (segment_num = 1, last_segment_num = 0; segment_num < 16; segment_num++)
	{
		segment_mask >>= 1;
		gboolean is_nongap = segment_mask & 1;
		if (is_nongap ^ was_nongap)
		{
			// Push buffer [last_segment_num, segment_num)
			guint64 offset = 16384 * (gps_start_time + last_segment_num);
			gint size = 16384 * (segment_num - last_segment_num) * sizeof(double);
			GstBuffer* buf;
			GstFlowReturn result = gst_pad_alloc_buffer(
				GST_BASE_SRC_PAD(basesrc), offset, size,
				GST_PAD_CAPS(GST_BASE_SRC_PAD(basesrc)), &buf);
			if (result != GST_FLOW_OK)
			{
				FrVectFree(frVect);
				return result;
			}
			memcpy(GST_BUFFER_DATA(buf), &((double*)frVect->data)[16384 * last_segment_num], size);
			GST_BUFFER_OFFSET(buf) = offset;
			GST_BUFFER_OFFSET_END(buf) = 16384 * (gps_start_time + segment_num);
			GST_BUFFER_DURATION(buf) = GST_SECOND * (segment_num - last_segment_num);
			GST_BUFFER_TIMESTAMP(buf) = GST_SECOND *  (gps_start_time + last_segment_num);
			if (!was_nongap)
			{
				GST_BUFFER_FLAG_SET(buf, GST_BUFFER_FLAG_GAP);
				GST_INFO_OBJECT(element, "Setting GST_BUFFER_FLAG_GAP");
			}
			if (is_discontinuous)
			{
				GST_BUFFER_FLAG_SET(buf, GST_BUFFER_FLAG_DISCONT);
				is_discontinuous = FALSE;
			}
			GST_INFO_OBJECT(element, "pushing frame spanning [%llu, %llu) (extra frame because of change in data quality)",
				gps_start_time + last_segment_num,
				gps_start_time + segment_num
			);
			result = gst_pad_push(GST_BASE_SRC_PAD(basesrc), buf);
			if (result != GST_FLOW_OK)
				return result;

			last_segment_num = segment_num;
		}
		was_nongap = is_nongap;
	}

	{
		// create buffer [last_segment_num, segment_num) to return
		guint64 offset = 16384 * (gps_start_time + last_segment_num);
		gint size = 16384 * (segment_num - last_segment_num) * sizeof(double);
		GstBuffer* buf;
		GstFlowReturn result = gst_pad_alloc_buffer(
			GST_BASE_SRC_PAD(basesrc), offset, size,
			GST_PAD_CAPS(GST_BASE_SRC_PAD(basesrc)), &buf);
		if (result != GST_FLOW_OK)
		{
			FrVectFree(frVect);
			return result;
		}
		memcpy(GST_BUFFER_DATA(buf), &((double*)frVect->data)[16384 * last_segment_num], size);
		GST_BUFFER_OFFSET(buf) = offset;
		GST_BUFFER_OFFSET_END(buf) = 16384 * (gps_start_time + segment_num);
		GST_BUFFER_DURATION(buf) = GST_SECOND * (segment_num - last_segment_num);
		GST_BUFFER_TIMESTAMP(buf) = GST_SECOND *  (gps_start_time + last_segment_num);
		if (!was_nongap)
		{
			GST_BUFFER_FLAG_SET(buf, GST_BUFFER_FLAG_GAP);
			GST_INFO_OBJECT(element, "Setting GST_BUFFER_FLAG_GAP");
		}
		if (is_discontinuous)
			GST_BUFFER_FLAG_SET(buf, GST_BUFFER_FLAG_DISCONT);
		*buffer = buf;
	}

	FrVectFree(frVect);

	GST_INFO_OBJECT(element, "pushing frame spanning [%llu, %llu)",
		gps_start_time + last_segment_num,
		gps_start_time + segment_num
	);

	return GST_FLOW_OK;
}



/*
 * check_get_range()
 */


static gboolean check_get_range(GstBaseSrc *object)
{
	// FIXME: This element doesn't really support random access, so we should 
	// return FALSE here, but it seems like gstlal_inspiral doesn't like that
	return TRUE;
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
	GSTLALOnlineHoftSrc *element = GSTLAL_ONLINEHOFTSRC(basesrc);

	GST_INFO_OBJECT(basesrc, "do_seek got segment: [%ld, %ld)", segment->start, segment->stop);

	if (segment->flags & GST_SEEK_FLAG_KEY_UNIT)
	{
		segment->start = gst_util_uint64_scale(gst_util_uint64_scale(segment->start, 1, 16 * GST_SECOND), 16 * GST_SECOND, 1);
		segment->last_stop = gst_util_uint64_scale(gst_util_uint64_scale(segment->last_stop, 1, 16 * GST_SECOND), 16 * GST_SECOND, 1);
		segment->time = gst_util_uint64_scale(gst_util_uint64_scale(segment->time, 1, 16 * GST_SECOND), 16 * GST_SECOND, 1);
		segment->stop = gst_util_uint64_scale_ceil(gst_util_uint64_scale_ceil(segment->stop, 1, 16 * GST_SECOND), 16 * GST_SECOND, 1);
		GST_INFO_OBJECT(basesrc, "do_seek modified key unit seek segment: [%ld, %ld)", segment->start, segment->stop);
	}

	GST_INFO_OBJECT(element, "in do_seek: %u", segment->start / GST_SECOND);
	element->needs_seek = TRUE;
	return TRUE;
}



/*
 * query()
 */


static gboolean query(GstBaseSrc *basesrc, GstQuery *query)
{
	switch (GST_QUERY_TYPE(query))
	{
		case GST_QUERY_FORMATS: {
			gst_query_set_formats(query, 4, GST_FORMAT_DEFAULT, GST_FORMAT_BYTES, GST_FORMAT_TIME, GST_FORMAT_BUFFERS);
			return TRUE;
		} break;

		case GST_QUERY_CONVERT: {
			GstFormat src_format, dest_format;
			gint64 src_value, dest_value;
			guint64 num = 1, den = 1;
			
			gst_query_parse_convert(query, &src_format, &src_value, &dest_format, &dest_value);
			
			switch (src_format)
			{
				case GST_FORMAT_DEFAULT:
				case GST_FORMAT_TIME:
					break;
				case GST_FORMAT_BYTES:
					den *= (8 /*bytes per sample*/) * (16384 /*samples per second*/);
					num *= (GST_SECOND /*nanoseconds per second*/);
					break;
				case GST_FORMAT_BUFFERS:
					num *= (16 /*seconds per buffer*/) * (GST_SECOND /*nanoseconds per second*/);
					break;
				default:
					g_assert_not_reached();
					return FALSE;
			}
			switch (dest_format)
			{
				case GST_FORMAT_DEFAULT:
				case GST_FORMAT_TIME:
					break;
				case GST_FORMAT_BYTES:
					num *= (8 /*bytes per sample*/) * (16384 /*samples per second*/);
					den *= (GST_SECOND /*nanoseconds per second*/);
					break;
				case GST_FORMAT_BUFFERS:
					den *= (16 /*seconds per buffer*/) * (GST_SECOND /*nanoseconds per second*/);
					break;
				default:
					g_assert_not_reached();
					return FALSE;
			}
			
			dest_value = gst_util_uint64_scale(src_value, num, den);
			gst_query_set_convert(query, src_format, src_value, dest_format, dest_value);
			return TRUE;
		} break;

		default:
			return parent_class->query(basesrc, query);
	}
}



/*
 * ============================================================================
 *
 *                                Type Support
 *
 * ============================================================================
 */


/*
 * Instance finalize function.  See ???
 */


static void finalize(GObject *object)
{
	GSTLALOnlineHoftSrc *element = GSTLAL_ONLINEHOFTSRC(object);

	g_free(element->instrument);
	element->instrument = NULL;
	onlinehoft_destroy(element->tracker);
	element->tracker = NULL;

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


/*
 * Base init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GBaseInitFunc
 */


static void base_init(gpointer class)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(class);

	gst_element_class_set_details_simple(
		element_class,
		"Online h(t) Source",
		"Source",
		"LAL online h(t) source",
		"Leo Singer <leo.singer@ligo.org>"
	);

	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"src",
			GST_PAD_SRC,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
				"audio/x-raw-float, " \
				"rate = (int) 16384, " \
				"channels = (int) 1, " \
				"endianness = (int) BYTE_ORDER, " \
				"width = (int) 64 " \
			)
		)
	);
}


/*
 * Class init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GClassInitFunc
 */


static void class_init(gpointer class, gpointer class_data)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(class);
	GstBaseSrcClass *gstbasesrc_class = GST_BASE_SRC_CLASS(class);

	parent_class = g_type_class_ref(GST_TYPE_BASE_SRC);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	g_object_class_install_property(
		gobject_class,
		ARG_SRC_INSTRUMENT,
		g_param_spec_string(
			"instrument",
			"Instrument",
			"Instrument name (e.g., \"H1\").",
			NULL,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);

	/*
	 * GstBaseSrc method overrides
	 */

	gstbasesrc_class->create = GST_DEBUG_FUNCPTR(create);
	gstbasesrc_class->is_seekable = GST_DEBUG_FUNCPTR(is_seekable);
	gstbasesrc_class->check_get_range = GST_DEBUG_FUNCPTR(check_get_range);
	gstbasesrc_class->do_seek = GST_DEBUG_FUNCPTR(do_seek);
	gstbasesrc_class->query = GST_DEBUG_FUNCPTR(query);
}


/*
 * Instance init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GInstanceInitFunc
 */


static void instance_init(GTypeInstance *object, gpointer class)
{
	GstBaseSrc *basesrc = GST_BASE_SRC(object);
	GSTLALOnlineHoftSrc *element = GSTLAL_ONLINEHOFTSRC(object);

	gst_pad_use_fixed_caps(GST_BASE_SRC_PAD(basesrc));

	basesrc->offset = 0;
	element->instrument = NULL;
	element->tracker = NULL;
	element->needs_seek = FALSE;

	gst_base_src_set_blocksize(basesrc, 16384 * 16 * 8);
	gst_base_src_set_do_timestamp(basesrc, FALSE);
	gst_base_src_set_format(GST_BASE_SRC(object), GST_FORMAT_TIME);
}


/*
 * gstlal_onlinehoftsrc_get_type().
 */


GType gstlal_onlinehoftsrc_get_type(void)
{
	static GType type = 0;

	if(!type) {
		static const GTypeInfo info = {
			.class_size = sizeof(GSTLALOnlineHoftSrcClass),
			.class_init = class_init,
			.base_init = base_init,
			.instance_size = sizeof(GSTLALOnlineHoftSrc),
			.instance_init = instance_init,
		};
		type = g_type_register_static(GST_TYPE_BASE_SRC, "lal_onlinehoftsrc", &info, 0);
	}

	return type;
}

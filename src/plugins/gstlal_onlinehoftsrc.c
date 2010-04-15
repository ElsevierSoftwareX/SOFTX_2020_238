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
#include <gst/base/gstpushsrc.h>


/*
 * stuff from LAL
 */


#include <lal/LALDatatypes.h>
#include <lal/Aggregation.h>


/*
 * our own stuff
 */


#include <gstlal.h>
#include <gstlal_onlinehoftsrc.h>


/*
 * Parent class.
 */


static GstPushSrcClass *parent_class = NULL;



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
    
    LIGOTimeGPS timeGPS = { (element->gps_remainder) << 4, 0};
    
    REAL8TimeSeries* series
        = XLALAggregationStrainDataWait(element->instrument, &timeGPS, 16, 32);
    
    if (!series)
        return GST_FLOW_ERROR;
    
    size_t data_size = series->data->length * sizeof(*series->data->data);
    
    GstFlowReturn result = gst_pad_alloc_buffer(GST_BASE_SRC_PAD(basesrc), basesrc->offset, data_size, GST_PAD_CAPS(GST_BASE_SRC_PAD(basesrc)), buffer);
    
    if (result != GST_FLOW_OK)
    {
        XLALDestroyREAL8TimeSeries(series);
        return result;
    }
    
    memcpy((char*)GST_BUFFER_DATA(*buffer), series->data->data, data_size);
    basesrc->offset += series->data->length;
    GST_BUFFER_OFFSET_END(*buffer) = basesrc->offset;
    GST_BUFFER_TIMESTAMP(*buffer) = GST_SECOND * series->epoch.gpsSeconds + series->epoch.gpsNanoSeconds;
    GST_BUFFER_DURATION(*buffer) = series->data->length * series->deltaT * GST_SECOND;
    XLALDestroyREAL8TimeSeries(series);
    
    return GST_FLOW_OK; /// TODO
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
    
    element->gps_remainder = ((guint32) (segment->start / GST_SECOND)) >> 4;
    return TRUE;
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

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


/*
 * Base init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GBaseInitFunc
 */


static void base_init(gpointer class)
{
	static const GstElementDetails plugin_details = {
		"Online h(t) Source",
		"Source",
		"LAL online h(t) source",
		"Leo Singer <leo.singer@ligo.org>"
	};
	GstElementClass *element_class = GST_ELEMENT_CLASS(class);

	gst_element_class_set_details(element_class, &plugin_details);

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

	parent_class = g_type_class_ref(GST_TYPE_PUSH_SRC);

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
	gstbasesrc_class->do_seek = GST_DEBUG_FUNCPTR(do_seek);
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
    element->gps_remainder = 0; // TODO: should set to current time

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
		type = g_type_register_static(GST_TYPE_PUSH_SRC, "lal_onlinehoftsrc", &info, 0);
	}

	return type;
}

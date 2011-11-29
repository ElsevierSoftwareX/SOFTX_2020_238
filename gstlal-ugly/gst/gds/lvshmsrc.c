/*
 * GDS LIGO-Virgo Shared Memory frame file source element
 *
 * Copyright (C) 2011  Kipp Cannon
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
 * stuff from gds
 */


#include <gds/lvshmapi.h>


/*
 * our own stuff
 */


#include <gstlal/gstlal_debug.h>
#include <lvshmsrc.h>


GST_DEBUG_CATEGORY(gds_lvshmsrc_debug);


/*
 * Parent class.
 */


static GstBaseSrcClass *parent_class = NULL;


/*
 * ========================================================================
 *
 *                                 Parameters
 *
 * ========================================================================
 */


#define GST_CAT_DEFAULT gds_lvshmsrc_debug


enum property {
	ARG_NAME = 1,
	ARG_MASK,
	ARG_WAIT_TIME
};


#define DEFAULT_NAME "Fast_Data"
#define DEFAULT_MASK -1	/* FIXME:  what does this mean? */
#define DEFAULT_WAIT_TIME -1.0	/* wait indefinitely */


/*
 * ========================================================================
 *
 *                             Utility Functions
 *
 * ========================================================================
 */


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
	GDSLVSHMSrc *element = GDS_LVSHMSRC(object);

	g_assert(element->name != NULL);
	element->handle = lvshm_init(element->name, element->mask);
	lvshm_setWaitTime(element->handle, element->wait_time);

	return TRUE;
}


/*
 * stop()
 */


static gboolean stop(GstBaseSrc *object)
{
	GDSLVSHMSrc *element = GDS_LVSHMSRC(object);

	/* FIXME:  can I pass NULL to deaccess()? */
	if(element->handle) {
		lvshm_deaccess(element->handle);
		element->handle = NULL;
	}

	return TRUE;
}


/*
 * create()
 */


static GstFlowReturn create(GstBaseSrc *basesrc, guint64 offset, guint size, GstBuffer **buffer)
{
	GDSLVSHMSrc *element = GDS_LVSHMSRC(basesrc);
	int flags = 0;	/* LVSHM_NOWAIT is not set = respect wait time */
	const char *data;
	unsigned length;
	GstFlowReturn result = GST_FLOW_OK;

	/*
	 * just in case
	 */

	*buffer = NULL;

	/*
	 * retrieve next frame file from the lvshm library.  all error
	 * paths after this succedes must include a call to
	 * lvshm_releaseDataBuffer()
	 */

	data = lvshm_getNextBuffer(element->handle, flags);
	if(!data) {
		GST_ELEMENT_ERROR(element, RESOURCE, READ, (NULL), ("unknown failure retrieving buffer from GDS shared memory.  possible causes include:  timeout, interupted by signal, no data available."));
		/* indicate end-of-stream */
		return GST_FLOW_UNEXPECTED;
	}
	length = lvshm_dataLength(element->handle);
	GST_LOG_OBJECT(element, "retrieved %u byte frame file labeled %lu", length, (unsigned long) lvshm_bufferGPS(element->handle));

	/*
	 * copy into a GstBuffer
	 */

	result = gst_pad_alloc_buffer(GST_BASE_SRC_PAD(basesrc), offset, length, GST_PAD_CAPS(GST_BASE_SRC_PAD(basesrc)), buffer);
	if(result != GST_FLOW_OK) {
		GST_ELEMENT_ERROR(element, RESOURCE, READ, (NULL), ("gst_pad_alloc_buffer() returned %d (%s)", result, gst_flow_get_name(result)));
		goto done;
	}
	if(GST_BUFFER_SIZE(*buffer) != length) {
		GST_ELEMENT_ERROR(element, RESOURCE, READ, (NULL), ("gst_pad_alloc_buffer(): requested buffer size %u, got buffer size %u", length, GST_BUFFER_SIZE(*buffer)));
		gst_buffer_unref(*buffer);
		*buffer = NULL;
		result = GST_FLOW_ERROR;
		goto done;
	}
	memcpy(GST_BUFFER_DATA(*buffer), data, length);
	GST_BUFFER_TIMESTAMP(*buffer) = lvshm_bufferGPS(element->handle) * GST_SECOND;
	GST_BUFFER_DURATION(*buffer) = GST_CLOCK_TIME_NONE;
	GST_BUFFER_OFFSET(*buffer) = offset;
	GST_BUFFER_OFFSET_END(*buffer) = GST_BUFFER_OFFSET_NONE;

	/*
	 * Done
	 */

done:
	lvshm_releaseDataBuffer(element->handle);
	return result;
}


/*
 * ============================================================================
 *
 *                          GObject Method Overrides
 *
 * ============================================================================
 */


static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec)
{
	GDSLVSHMSrc *element = GDS_LVSHMSRC(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_NAME:
		g_free(element->name);
		element->name = g_value_dup_string(value);
		break;

	case ARG_MASK:
		element->mask = g_value_get_uint(value);
		break;

	case ARG_WAIT_TIME:
		element->wait_time = g_value_get_double(value);
		if(element->handle)
			lvshm_setWaitTime(element->handle, element->wait_time);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *pspec)
{
	GDSLVSHMSrc *element = GDS_LVSHMSRC(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_NAME:
		g_value_set_string(value, element->name);
		break;

	case ARG_MASK:
		g_value_set_uint(value, element->mask);
		break;

	case ARG_WAIT_TIME:
		g_value_set_double(value, element->wait_time);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * Instance finalize function.
 */


static void finalize(GObject *object)
{
	GDSLVSHMSrc *element = GDS_LVSHMSRC(object);

	g_free(element->name);
	element->name = NULL;

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
		"GDS LVSHM Frame File Source",
		"Source",
		"LIGO-Virgo shared memory frame file source element",
		"Kipp Cannon <kipp.cannon@ligo.org>"
	);

	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"src",
			GST_PAD_SRC,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
				"application/x-igwd-frame"
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
		ARG_NAME,
		g_param_spec_string(
			"shm-name",
			"Name",
			"Shared memory partition name.",
			DEFAULT_NAME,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_MASK,
		g_param_spec_uint(
			"mask",
			"Mask",
			"Shared memory data type mask.",
			0, G_MAXUINT, DEFAULT_MASK,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_WAIT_TIME,
		g_param_spec_double(
			"wait-time",
			"Wait time",
			"Wait time in seconds (<0 = wait indefinitely, 0 = never wait).",
			-G_MAXDOUBLE, G_MAXDOUBLE, DEFAULT_WAIT_TIME,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);

	/*
	 * GstBaseSrc method overrides
	 */

	gstbasesrc_class->start = GST_DEBUG_FUNCPTR(start);
	gstbasesrc_class->stop = GST_DEBUG_FUNCPTR(stop);
	gstbasesrc_class->create = GST_DEBUG_FUNCPTR(create);
}


/*
 * Instance init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GInstanceInitFunc
 */


static void instance_init(GTypeInstance *object, gpointer class)
{
	GstBaseSrc *basesrc = GST_BASE_SRC(object);
	GDSLVSHMSrc *element = GDS_LVSHMSRC(object);

	gst_base_src_set_live(basesrc, TRUE);
	gst_base_src_set_format(basesrc, GST_FORMAT_TIME);

	element->handle = NULL;
}


/*
 * gsd_lvshmsrc_get_type().
 */


GType gsd_lvshmsrc_get_type(void)
{
	static GType type = 0;

	if(!type) {
		static const GTypeInfo info = {
			.class_size = sizeof(GDSLVSHMSrcClass),
			.class_init = class_init,
			.base_init = base_init,
			.instance_size = sizeof(GDSLVSHMSrc),
			.instance_init = instance_init,
		};
		type = g_type_register_static(GST_TYPE_BASE_SRC, "gds_lvshmsrc", &info, 0);
	}

	return type;
}

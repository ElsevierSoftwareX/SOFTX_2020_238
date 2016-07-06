/*
 * gds lvshm (LIGO-Virgo Shared Memory) sink element
 *
 * Copyright (C) 2012,2013  Kipp Cannon
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
 * Completed Action
 * - Wrote pass-through unit test
 *
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
 * stuff from glib/gobject/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasesink.h>


/*
 * stuff from gds
 */


#include <gds/lsmp_prod.hh>


/*
 * our own stuff
 */


#include <gstlal/gstlal_debug.h>
#include <lvshmsink.h>


/*
 * ========================================================================
 *
 *                                Boilerplate
 *
 * ========================================================================
 */


#define GST_CAT_DEFAULT gds_lvshmsink_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


G_DEFINE_TYPE_WITH_CODE(
	GDSLVSHMSink,
	gds_lvshmsink,
	GST_TYPE_BASE_SINK,
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "gds_lvshmsink", 0, "gds_lvshmsink element")
);


/*
 * buffer-mode enum type
 */


GType gds_lvshmsink_buffer_mode_get_type(void)
{
	static GType type = 0;

	if(!type) {
		static GEnumValue values[] = {
			{GDS_LVSHMSINK_BUFFER_MODE_0, "GDS_LVSHMSINK_BUFFER_MODE_0", "A buffer is released (moved to free list) any time is not reserved and not in use."},
			{GDS_LVSHMSINK_BUFFER_MODE_1, "GDS_LVSHMSINK_BUFFER_MODE_1", "A buffer is released after it has been seen by at least one consumer and is no longer reserved or in use."},
			{GDS_LVSHMSINK_BUFFER_MODE_2, "GDS_LVSHMSINK_BUFFER_MODE_2", "A buffer is not released except if it is unreserved, not in use and needed to fill a producer request."},
			{GDS_LVSHMSINK_BUFFER_MODE_3, "GDS_LVSHMSINK_BUFFER_MODE_3", "All incoming buffers are reserved for all consumers.  The buffer is entered into the free list after being released by all consumers."},
			{0, NULL, NULL}
		};

		type = g_enum_register_static("gds_lvshmsink_buffer_mode", values);
	}

	return type;
}


/*
 * ========================================================================
 *
 *                                 Parameters
 *
 * ========================================================================
 */


#define DEFAULT_BLOCKSIZE (1<<20)	/* 1 MiB */
#define DEFAULT_SHM_NAME NULL
#define DEFAULT_NUM_BUFFERS 1
#define DEFAULT_MASK -1
#define DEFAULT_BUFFER_MODE 0
#define DEFAULT_LOCK FALSE


/*
 * ============================================================================
 *
 *                                 Utilities
 *
 * ============================================================================
 */


#define lsmp_partition(element) ((LSMP_PROD *) ((element)->partition))


/*
 * ============================================================================
 *
 *                        GstBaseSink Method Overrides
 *
 * ============================================================================
 */


/*
 * get_times()
 */


static void get_times(GstBaseSink *sink, GstBuffer *buffer, GstClockTime *start, GstClockTime *end)
{
	*start = GST_BUFFER_PTS(buffer);
	*end = GST_BUFFER_PTS(buffer) + GST_BUFFER_DURATION(buffer);
}


/*
 * start()
 */


static gboolean start(GstBaseSink *sink)
{
	GDSLVSHMSink *element = GDS_LVSHMSINK(sink);
	gboolean success = TRUE;

	if(!element->name) {
		GST_ELEMENT_ERROR(element, RESOURCE, FAILED, (NULL), ("shm-name not set"));
		success = FALSE;
		goto done;
	}

	element->partition = new LSMP_PROD(element->name, element->num_buffers, gst_base_sink_get_blocksize(sink));
	if(!element->partition) {
		GST_ELEMENT_ERROR(element, RESOURCE, OPEN_WRITE, (NULL), ("unknown failure accessing shared-memory partition"));
		success = FALSE;
		goto done;
	}
	if(!lsmp_partition(element)->valid()) {
		GST_ELEMENT_ERROR(element, RESOURCE, OPEN_WRITE, (NULL), ("failure accessing shared-memory partition: %s", lsmp_partition(element)->Error()));
		delete lsmp_partition(element);
		element->partition = NULL;
		success = FALSE;
		goto done;
	}

	GST_DEBUG_OBJECT(element, "gained access to shared-memory partition \"%s\"", element->name);

	lsmp_partition(element)->keep(false);
	if(element->lock) {
		if(lsmp_partition(element)->lock(true))
			GST_ELEMENT_WARNING(element, RESOURCE, OPEN_WRITE, (NULL), ("failure locking shared-memory partition into memory: %s", lsmp_partition(element)->Error()));
		else
			GST_DEBUG_OBJECT(element, "locked shared-memory partition \"%s\" to main memory", element->name);
	}

	GST_DEBUG_OBJECT(element, "setting shared-memory partition \"%s\" buffer allocation mode to %d", element->name, element->buffer_mode);
	lsmp_partition(element)->bufmode(element->buffer_mode);

done:
	return success;
}


/*
 * stop()
 */


static gboolean stop(GstBaseSink *sink)
{
	GDSLVSHMSink *element = GDS_LVSHMSINK(sink);
	gboolean success = TRUE;

	delete lsmp_partition(element);
	element->partition = NULL;

	GST_DEBUG_OBJECT(element, "de-accessed shared-memory partition \"%s\"", element->name);

	return success;
}


/*
 * render()
 */


static GstFlowReturn render(GstBaseSink *sink, GstBuffer *buffer)
{
	GDSLVSHMSink *element = GDS_LVSHMSINK(sink);
	GstMapInfo mapinfo;
	GstFlowReturn result = GST_FLOW_OK;
	int flags = 0;	/* NOWAIT = don't wait */
	char *dest;

	g_assert(element->partition != NULL);

	GST_DEBUG_OBJECT(element, "have buffer spanning %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(buffer));

	gst_buffer_map(buffer, &mapinfo, GST_MAP_READ);

	dest = lsmp_partition(element)->get_buffer(flags);
	if(!dest) {
		GST_ELEMENT_ERROR(element, RESOURCE, WRITE, (NULL), ("unable to obtain shared-memory buffer"));
		result = GST_FLOW_ERROR;
		goto done;
	}
	GST_DEBUG_OBJECT(element, "have shared-memory buffer %p", dest);
	if(mapinfo.size > (guint64) lsmp_partition(element)->getBufferLength()) {
		GST_ELEMENT_ERROR(element, RESOURCE, WRITE, (NULL), ("data (%" G_GUINT64_FORMAT " bytes) too large for shared-memry buffer (%" G_GUINT64_FORMAT " bytes)", mapinfo.size, (guint64) lsmp_partition(element)->getBufferLength()));
		lsmp_partition(element)->return_buffer();
		result = GST_FLOW_ERROR;
		goto done;
	}
	memcpy(dest, mapinfo.data, mapinfo.size);
	memset(dest + mapinfo.size, 0, gst_base_sink_get_blocksize(sink) - mapinfo.size);
	lsmp_partition(element)->SetID(GST_BUFFER_PTS(buffer) / GST_SECOND);
	GST_DEBUG_OBJECT(element, "shared-memory buffer %p ID set to %" G_GUINT64_FORMAT, dest, GST_BUFFER_PTS(buffer) / GST_SECOND);
	lsmp_partition(element)->release(mapinfo.size, element->mask, flags);
	GST_DEBUG_OBJECT(element, "shared-memory buffer %p released", dest);

done:
	gst_buffer_unmap(buffer, &mapinfo);
	return result;
}


/*
 * ============================================================================
 *
 *                          GObject Method Overrides
 *
 * ============================================================================
 */


/*
 * properties
 */


enum property {
	ARG_SHM_NAME = 1,
	ARG_NUM_BUFFERS,
	ARG_MASK,
	ARG_BUFFER_MODE,
	ARG_LOCK,
};


static void set_property(GObject *object, guint id, const GValue *value, GParamSpec *pspec)
{
	GDSLVSHMSink *element = GDS_LVSHMSINK(object);

	GST_OBJECT_LOCK(element);

	switch((enum property) id) {
	case ARG_SHM_NAME:
		g_free(element->name);
		element->name = g_value_dup_string(value);
		break;

	case ARG_NUM_BUFFERS:
		element->num_buffers = g_value_get_uint(value);
		break;

	case ARG_MASK:
		element->mask = g_value_get_uint(value);
		break;

	case ARG_BUFFER_MODE:
		element->buffer_mode = (enum gds_lvshmsink_buffer_mode) g_value_get_enum(value);
		break;

	case ARG_LOCK:
		element->lock = g_value_get_boolean(value);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, guint id, GValue *value, GParamSpec *pspec)
{
	GDSLVSHMSink *element = GDS_LVSHMSINK(object);

	GST_OBJECT_LOCK(element);

	switch((enum property) id) {
	case ARG_SHM_NAME:
		g_value_set_string(value, element->name);
		break;

	case ARG_NUM_BUFFERS:
		g_value_set_uint(value, element->num_buffers);
		break;

	case ARG_MASK:
		g_value_set_uint(value, element->mask);
		break;

	case ARG_BUFFER_MODE:
		g_value_set_enum(value, element->buffer_mode);
		break;

	case ARG_LOCK:
		g_value_set_boolean(value, element->lock);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * finalize()
 */


static void finalize(GObject *object)
{
	GDSLVSHMSink *element = GDS_LVSHMSINK(object);

	g_free(element->name);
	element->name = NULL;
	if(lsmp_partition(element)) {
		lsmp_partition(element)->lock(false);
		delete lsmp_partition(element);
		element->partition = NULL;
	}

	G_OBJECT_CLASS(gds_lvshmsink_parent_class)->finalize(object);
}


/*
 * class_init()
 */


static void gds_lvshmsink_class_init(GDSLVSHMSinkClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GstBaseSinkClass *gstbasesink_class = GST_BASE_SINK_CLASS(klass);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	gstbasesink_class->get_times = GST_DEBUG_FUNCPTR(get_times);
	gstbasesink_class->start = GST_DEBUG_FUNCPTR(start);
	gstbasesink_class->stop = GST_DEBUG_FUNCPTR(stop);
	gstbasesink_class->render = GST_DEBUG_FUNCPTR(render);

	G_PARAM_SPEC_UINT(g_object_class_find_property(gobject_class, "blocksize"))->default_value = DEFAULT_BLOCKSIZE;

	gst_element_class_set_details_simple(
		element_class,
		"GDS LVSHM Frame File Sink",
		"Sink",
		"LIGO-Virgo shared memory frame file sink element",
		"Kipp Cannon <kipp.cannon@ligo.org>"
	);

	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"sink",
			GST_PAD_SINK,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
				"application/x-igwd-frame, " \
				"framed = (boolean) true"
			)
		)
	);

	g_object_class_install_property(
		gobject_class,
		ARG_SHM_NAME,
		g_param_spec_string(
			"shm-name",
			"Name",
			"Shared memory partition name.  Suggestions:  \"LHO_Data\", \"LLO_Data\", \"VIRGO_Data\".",
			DEFAULT_SHM_NAME,
			(GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_NUM_BUFFERS,
		g_param_spec_uint(
			"num-buffers",
			"Number of buffers",
			"Number of buffers in shared-memory partiion.",
			1, G_MAXUINT, DEFAULT_NUM_BUFFERS,
			(GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_MASK,
		g_param_spec_uint(
			"mask",
			"Trigger mask",
			"To receive buffers, consumers must use a mask whose bitwise logical and with this mask is non-zero.",
			0, G_MAXUINT, DEFAULT_MASK,
			(GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_BUFFER_MODE,
		g_param_spec_enum(
			"buffer-mode",
			"Buffer mode",
			"Buffer allocation mode.",
			GDS_LVSHMSINK_BUFFER_MODE_TYPE,
			DEFAULT_BUFFER_MODE,
			(GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_LOCK,
		g_param_spec_boolean(
			"lock",
			"Lock shared-memory",
			"Lock shared-memory into main memory.  Process must be able to set effective user ID to 0.",
			DEFAULT_LOCK,
			(GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)
		)
	);
}


/*
 * init()
 */


static void gds_lvshmsink_init(GDSLVSHMSink *element)
{
	gst_base_sink_set_blocksize(GST_BASE_SINK(element), DEFAULT_BLOCKSIZE);

	/*
	 * internal data
	 */

	element->name = NULL;
	element->partition = NULL;
}

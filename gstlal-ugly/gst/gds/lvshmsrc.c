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


#include <pthread.h>
#include <string.h>


/*
 * stuff from glib/gobject/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstpushsrc.h>


/*
 * stuff from gds
 */


#include <gds/lvshmapi.h>
#include <gds/tconv.h>


/*
 * our own stuff
 */


#include <gstlal/gstlal_debug.h>
#include <lvshmsrc.h>


#define GST_CAT_DEFAULT gds_lvshmsrc_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


/*
 * Parent class.
 */


static GstPushSrcClass *parent_class = NULL;


/*
 * ========================================================================
 *
 *                                 Parameters
 *
 * ========================================================================
 */


enum property {
	ARG_NAME = 1,
	ARG_MASK,
	ARG_WAIT_TIME
};


#define DEFAULT_NAME NULL
#define DEFAULT_MASK -1	/* FIXME:  what does this mean? */
#define DEFAULT_WAIT_TIME -1.0	/* wait indefinitely */


/*
 * ============================================================================
 *
 *                                 Utilities
 *
 * ============================================================================
 */


static GstClockTime GPSNow(void)
{
	/* FIXME:  why does TAInow() return the GPS time? */
	return gst_util_uint64_scale_int_round(TAInow(), GST_SECOND, _ONESEC);
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
	GDSLVSHMSrc *element = GDS_LVSHMSRC(object);

	if(!element->name) {
		GST_ELEMENT_ERROR(element, RESOURCE, READ, (NULL), ("shm-name not set"));
		return FALSE;
	}
	GST_LOG_OBJECT(element, "lvshm_init()");
	element->handle = lvshm_init(element->name, element->mask);
	if(!element->handle) {
		GST_ELEMENT_ERROR(element, RESOURCE, READ, (NULL), ("lvshm_init() failed"));
		return FALSE;
	}
	lvshm_setWaitTime(element->handle, element->wait_time);

	element->need_new_segment = TRUE;

	return TRUE;
}


/*
 * stop()
 */


static gboolean stop(GstBaseSrc *object)
{
	GDSLVSHMSrc *element = GDS_LVSHMSRC(object);

	GST_LOG_OBJECT(element, "lvshm_deaccess()");
	lvshm_deaccess(element->handle);
	element->handle = NULL;

	element->max_latency = element->min_latency = GST_CLOCK_TIME_NONE;

	return TRUE;
}


/*
 * unlock()
 */


static gboolean unlock(GstBaseSrc *basesrc)
{
	GDSLVSHMSrc *element = GDS_LVSHMSRC(basesrc);
	gboolean success = TRUE;

	element->unblocked = TRUE;

	if(!g_mutex_trylock(element->create_thread_lock))
		success = !pthread_kill(element->create_thread, SIGALRM);
	else
		g_mutex_unlock(element->create_thread_lock);

	return success;
}


/*
 * unlock_stop()
 */


static gboolean unlock_stop(GstBaseSrc *basesrc)
{
	GDSLVSHMSrc *element = GDS_LVSHMSRC(basesrc);
	gboolean success = TRUE;

	element->unblocked = FALSE;

	return success;
}


/*
 * create()
 */


static GstFlowReturn create(GstBaseSrc *basesrc, guint64 offset, guint size, GstBuffer **buffer)
{
	GDSLVSHMSrc *element = GDS_LVSHMSRC(basesrc);
	GstClockTime t_before, t_after;
	int flags = 0;	/* LVSHM_NOWAIT is not set = respect wait time */
	const char *data;
	unsigned length;
	GstClockTime timestamp;
	GstFlowReturn result = GST_FLOW_OK;

	/*
	 * just in case
	 */

	*buffer = NULL;

	/*
	 * retrieve next frame file from the lvshm interface.  disabled if
	 * element is "unlocked".
	 */

	if(element->unblocked) {
		GST_DEBUG_OBJECT(element, "unlock() called, no buffer created");
		return GST_FLOW_UNEXPECTED;
	}

	element->create_thread = pthread_self();
	g_mutex_lock(element->create_thread_lock);
	t_before = GPSNow();
	data = lvshm_getNextBuffer(element->handle, flags);
	t_after = GPSNow();
	g_mutex_unlock(element->create_thread_lock);
	if(!data) {
		/*
		 * data retrieval failed.  guess cause.
		 * FIXME:  we need an API that can tell us the cause
		 */

		if(element->unblocked) {
			/*
			 * assume reason for failure was we were killed by
			 * a signal, and assume that is because we were
			 * unblocked.  indicate end-of-stream
			 */

			GST_DEBUG_OBJECT(element, "unlock() called, no buffer created");
			return GST_FLOW_UNEXPECTED;
		} else if(element->wait_time > 0. && (GstClockTimeDiff) (t_after - t_before) >= (GstClockTimeDiff) (element->wait_time * GST_SECOND)) {
			/*
			 * assume reason for failure was a timeout.  create
			 * a 0-length buffer with a guess as to the
			 * timestamp of the missing data.  guess:  the time
			 * when we started waiting for the data adjusted by
			 * the most recently measured latency
			 *
			 *
			 * FIXME:  we need an API that can tell us the
			 * timestamp of the missing data
			 */

			GST_DEBUG_OBJECT(element, "timeout occured, creating 0-length heartbeat buffer");

			*buffer = gst_buffer_new();
			gst_buffer_set_caps(*buffer, GST_PAD_CAPS(GST_BASE_SRC_PAD(basesrc)));
			GST_BUFFER_TIMESTAMP(*buffer) = t_before;
			if(GST_CLOCK_TIME_IS_VALID(element->max_latency))
				GST_BUFFER_TIMESTAMP(*buffer) -= element->max_latency;
			GST_DEBUG_OBJECT(element, "heartbeat timestamp = %" GST_TIME_SECONDS_FORMAT, GST_TIME_SECONDS_ARGS(GST_BUFFER_TIMESTAMP(*buffer)));
			GST_BUFFER_DURATION(*buffer) = GST_CLOCK_TIME_NONE;
			GST_BUFFER_OFFSET(*buffer) = offset;
			GST_BUFFER_OFFSET_END(*buffer) = GST_BUFFER_OFFSET_NONE;
			return result;
		} else {
			/*
			 * reason for failure is not known.  indicate
			 * end-of-stream
			 */

			GST_ELEMENT_ERROR(element, RESOURCE, READ, (NULL), ("unknown failure retrieving buffer from GDS shared memory.  possible causes include:  timeout, interupted by signal, no data available."));
			return GST_FLOW_UNEXPECTED;
		}
	}
	/*
	 * we have successfully retrieved data.  all code paths from this
	 * point must include a call to lvshm_releaseDataBuffer()
	 */
	length = lvshm_dataLength(element->handle);
	timestamp = lvshm_bufferGPS(element->handle) * GST_SECOND;
	GST_LOG_OBJECT(element, "retrieved %u byte frame file for GPS %" GST_TIME_SECONDS_FORMAT, length, GST_TIME_SECONDS_ARGS(timestamp));

	/*
	 * copy into a GstBuffer
	 */

	if(!length) {
		GST_ELEMENT_ERROR(element, RESOURCE, READ, (NULL), ("received 0-byte frame file"));
		result = GST_FLOW_UNEXPECTED;
		goto done;
	}
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
	GST_BUFFER_TIMESTAMP(*buffer) = timestamp;
	GST_BUFFER_DURATION(*buffer) = GST_CLOCK_TIME_NONE;
	GST_BUFFER_OFFSET(*buffer) = offset;
	GST_BUFFER_OFFSET_END(*buffer) = GST_BUFFER_OFFSET_NONE;

	element->max_latency = GPSNow() - GST_BUFFER_TIMESTAMP(*buffer);
	element->min_latency = element->max_latency - GST_BUFFER_DURATION(*buffer);

	/*
	 * adjust segment
	 */

	if(element->need_new_segment) {
		gst_base_src_new_seamless_segment(basesrc, GST_BUFFER_TIMESTAMP(*buffer), GST_CLOCK_TIME_NONE, GST_BUFFER_TIMESTAMP(*buffer));
		element->need_new_segment = FALSE;
	}

	/*
	 * Done
	 */

done:
	lvshm_releaseDataBuffer(element->handle);
	GST_LOG_OBJECT(element, "released frame file");
	return result;
}


/*
 * query()
 */


static gboolean query(GstBaseSrc *basesrc, GstQuery *query)
{
	GDSLVSHMSrc *element = GDS_LVSHMSRC(basesrc);
	gboolean success = TRUE;

	switch(GST_QUERY_TYPE(query)) {
	case GST_QUERY_LATENCY:
		gst_query_set_latency(query, gst_base_src_is_live(basesrc), element->min_latency, element->max_latency);
		break;

	default:
		success = GST_BASE_SRC_CLASS(parent_class)->query(basesrc, query);
		break;
	}

	return success;
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
	g_mutex_free(element->create_thread_lock);
	element->create_thread_lock = NULL;
	if(element->handle) {
		GST_WARNING_OBJECT(element, "parent class failed to invoke stop() method.  doing lvshm_deaccess() in finalize() instead.");
		GST_LOG_OBJECT(element, "lvshm_deaccess()");
		lvshm_deaccess(element->handle);
		element->handle = NULL;
	}

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
				"application/x-igwd-frame, " \
				"framed = (boolean) true"
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
			"Shared memory partition name.  Suggestions:  \"LHO_Data\", \"LLO_Data\", \"VIRGO_Data\".",
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
	gstbasesrc_class->unlock = GST_DEBUG_FUNCPTR(unlock);
	gstbasesrc_class->unlock_stop = GST_DEBUG_FUNCPTR(unlock_stop);
	gstbasesrc_class->create = GST_DEBUG_FUNCPTR(create);
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
	GDSLVSHMSrc *element = GDS_LVSHMSRC(object);

	gst_base_src_set_live(basesrc, TRUE);
	gst_base_src_set_format(basesrc, GST_FORMAT_TIME);

	/*
	 * internal data
	 */

	element->name = NULL;
	element->max_latency = element->min_latency = GST_CLOCK_TIME_NONE;
	element->unblocked = FALSE;
	element->create_thread_lock = g_mutex_new();
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
		GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "gds_lvshmsrc", 0, "gds_lvshmsrc element");
		type = g_type_register_static(GST_TYPE_PUSH_SRC, "gds_lvshmsrc", &info, 0);
	}

	return type;
}

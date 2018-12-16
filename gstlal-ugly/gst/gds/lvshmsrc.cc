/*
 * GDS LIGO-Virgo Shared Memory frame file source element
 *
 * Copyright (C) 2011--2013  Kipp Cannon
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
 * SECTION:lvshmsrc
 * @short_description:  LIGO-Virgo shared memory frame file source element.
 *
 * Reviewed:  00d65a70accca228bb76bd07e89b3ec07c78f736 2014-08-13 K.
 * Cannon, J.  Creighton, B. Sathyaprakash.
 *
 * Actions:
 * - Please add output of tests that were done at the f2f meeting to the review page
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


#include <gds/lsmp_con.hh>
#include <gds/tconv.h>
#include <gds/SysError.hh>


/*
 * our own stuff
 */


#include <gstlal/gstlal_debug.h>
#include <lvshmsrc.h>


/*
 * ========================================================================
 *
 *                               GstURIHandler
 *
 * ========================================================================
 */


#define URI_SCHEME "lvshm"

static int repair_lsmp(GDSLVSHMSrc *element)
{
    LSMP_ACCESS *part = new LSMP_ACCESS(element->name);
	if (!part) {
        GST_WARNING_OBJECT(element, "Unable to construct partition accessor for [%s], ignoring.", element->name);
    }
	else if (!part->valid()) {
        GST_WARNING_OBJECT(element, "Unable to attach partition [%s], error: [%s], ignoring.", element->name, part->Error());
        delete part;
    }
    else {
        /* try this next statement and see if it works - otherwise, remove it */
        part->bufmode(5);
        /* more generally, we could make some new arguments for this element and set them here, e.g.: */
        /* if(element->repair_bufmode >= 0) part->bufmode(element->repair_bufmode); */
        /* if(element->repair_keep >= 0) part->keep(element->repair_keep); */
        part->Repair();
        delete part;
    }
    return(0);
}

static GstURIType uri_get_type(GType type)
{
	return GST_URI_SRC;
}


/* 1.0:  this becomes static const gchar *const * */
static gchar **uri_get_protocols(GType type)
{
	/* 1.0:  this becomes
	static const gchar *protocols[] = {URI_SCHEME, NULL};
	*/
	static gchar *protocols[] = {(gchar *) URI_SCHEME, NULL};

	return protocols;
}


/* 1.0:  this becomes static gchar * */
static const gchar *uri_get_uri(GstURIHandler *handler)
{
	GDSLVSHMSrc *element = GDS_LVSHMSRC(handler);

	/* 1.0:  this won't be a memory leak */
	return g_strdup_printf(URI_SCHEME "://%s", element->name);
}


/* 1.0:  this gets a GError ** argument */
static gboolean uri_set_uri(GstURIHandler *handler, const gchar *uri)
{
	GDSLVSHMSrc *element = GDS_LVSHMSRC(handler);
	gchar *scheme = g_uri_parse_scheme(uri);
	gchar name[strlen(uri)];
	gboolean success = TRUE;

	success = !strcmp(scheme, URI_SCHEME);
	if(success)
		success &= sscanf(uri, URI_SCHEME "://%[^/]", name) == 1;
	if(success)
		g_object_set(G_OBJECT(element), "shm-name", name, NULL);

	g_free(scheme);
	return success;
}


static void uri_handler_init(gpointer g_iface, gpointer iface_data)
{
	GstURIHandlerInterface *iface = (GstURIHandlerInterface *) g_iface;

	iface->get_uri = GST_DEBUG_FUNCPTR(uri_get_uri);
	iface->set_uri = GST_DEBUG_FUNCPTR(uri_set_uri);
	/* 1.0:  this is ->get_type */
	iface->get_type_full = GST_DEBUG_FUNCPTR(uri_get_type);
	/* 1.0:  this is ->get_protocols */
	iface->get_protocols_full = GST_DEBUG_FUNCPTR(uri_get_protocols);
}


/*
 * ========================================================================
 *
 *                                Boilerplate
 *
 * ========================================================================
 */


#define GST_CAT_DEFAULT gds_lvshmsrc_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


static void additional_initializations(GType type)
{
	static const GInterfaceInfo uri_handler_info = {
		uri_handler_init,
		NULL,
		NULL
	};
	g_type_add_interface_static(type, GST_TYPE_URI_HANDLER, &uri_handler_info);

	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "gds_lvshmsrc", 0, "gds_lvshmsrc element");
}


GST_BOILERPLATE_FULL(GDSLVSHMSrc, gds_lvshmsrc, GstBaseSrc, GST_TYPE_BASE_SRC, additional_initializations);


/*
 * ========================================================================
 *
 *                                 Parameters
 *
 * ========================================================================
 */


#define DEFAULT_SHM_NAME NULL
#define DEFAULT_MASK -1
#define DEFAULT_WAIT_TIME -1.0	/* wait indefinitely */
#define DEFAULT_ASSUMED_DURATION 4


/*
 * ============================================================================
 *
 *                                 Utilities
 *
 * ============================================================================
 */


#define lsmp_partition(element) ((LSMP_CON *) ((element)->partition))


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
	gboolean success = TRUE;

	if(!element->name) {
		GST_ELEMENT_ERROR(element, RESOURCE, READ, (NULL), ("shm-name not set"));
		success = FALSE;
		goto done;
	}
	repair_lsmp(element); // repair the lvshm first as suggested by Patrick Brockill
	element->partition = new LSMP_CON(element->name, 0 /* nbuf */, element->mask);
	if(!element->partition) {
		GST_ELEMENT_ERROR(element, RESOURCE, READ, (NULL), ("unknown failure accessing shared-memory parition \"%s\"", element->name));
		success = FALSE;
		goto done;
	}
	if(!lsmp_partition(element)->isConnected()) {
		GST_ELEMENT_ERROR(element, RESOURCE, OPEN_READ, (NULL), ("failure connecting to shared-memory partition \"%s\": %s", element->name, lsmp_partition(element)->Error()));
		delete lsmp_partition(element);
		element->partition = NULL;
		success = FALSE;
		goto done;
	}
	GST_DEBUG_OBJECT(element, "connected to shared-memory partition \"%s\"", element->name);

	lsmp_partition(element)->setTimeout(element->wait_time);

	element->need_new_segment = TRUE;
	element->next_timestamp = 0;

done:
	return success;
}


/*
 * stop()
 */


static gboolean stop(GstBaseSrc *object)
{
	GDSLVSHMSrc *element = GDS_LVSHMSRC(object);

	delete lsmp_partition(element);
	element->partition = NULL;
	GST_DEBUG_OBJECT(element, "de-accessed shared-memory partition \"%s\"", element->name);

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
	GstClockTime t_before;
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
	while(1) {
		int catch_error = 0;
		int save_errno = 0;
		g_mutex_lock(element->create_thread_lock);
		t_before = GPSNow();
		try {
			/*
			* LSMP_CON::get_buffer API: https://bugs.ligo.org/redmine/issues/6225
			*/
			data = lsmp_partition(element) -> get_buffer(flags);
		}
		catch (const SysError & e) {
			GST_WARNING_OBJECT(element, "Caught SysError description: [%s]", e.what());
			catch_error = 1;
		}
		catch (const std::exception & e) {
			GST_WARNING_OBJECT(element, "Caught error with description: [%s]", e.what());
			catch_error = 2;
		}
		save_errno = errno;
		g_mutex_unlock(element->create_thread_lock);
		if(catch_error) {
			switch(save_errno) {
			case EIDRM:
				GST_ELEMENT_ERROR(element, RESOURCE, READ, (NULL), ("EIDRM received in gst_buffer(), shared memory partition [%s] possibly removed.", element->name));
				return GST_FLOW_UNEXPECTED;
			default:
				GST_ELEMENT_ERROR(element, RESOURCE, READ, (NULL), ("Caught error in gst_buffer(), errno: [%d]", save_errno));
				return GST_FLOW_UNEXPECTED;
			}
		}
		if(!data) {
			/*
			 * data retrieval failed.  guess cause.
			 * FIXME:  we need an API that can tell us the
			 * cause
			 */

			if(element->unblocked) {
				/*
				 * assume reason for failure was we were
				 * killed by a signal, and assume that is
				 * because we were unblocked.  indicate
				 * end-of-stream
				 */

				GST_DEBUG_OBJECT(element, "unlock() called, no buffer created");
				return GST_FLOW_UNEXPECTED;
			}
			else {
				switch(save_errno) {
				case EAGAIN:
					/*
					 * assume reason for failure was a timeout.
					 * create a 0-length buffer with a guess as
					 * to the timestamp of the missing data.
					 * guess:  the time when we started waiting
					 * for the data adjusted by the most
					 * recently measured latency
					 *
					 * FIXME:  we need an API that can tell us
					 * the timestamp of the missing data, e.g.,
					 * when we receive data tell us what its
					 * duration is so when know how much we've
					 * received.
					 */

					GST_DEBUG_OBJECT(element, "timeout occured, creating 0-length heartbeat buffer");

					*buffer = gst_buffer_new();
					gst_buffer_set_caps(*buffer, GST_PAD_CAPS(GST_BASE_SRC_PAD(basesrc)));
					GST_BUFFER_TIMESTAMP(*buffer) = t_before;
					if(GST_CLOCK_TIME_IS_VALID(element->max_latency))
						GST_BUFFER_TIMESTAMP(*buffer) -= element->max_latency;
					if(GST_BUFFER_TIMESTAMP(*buffer) < element->next_timestamp) {
						GST_LOG_OBJECT(element, "time reversal.	 skipping buffer.");
						gst_buffer_unref(*buffer);
						*buffer = NULL;
						continue;
					}
					GST_DEBUG_OBJECT(element, "heartbeat timestamp = %" GST_TIME_SECONDS_FORMAT, GST_TIME_SECONDS_ARGS(GST_BUFFER_TIMESTAMP(*buffer)));
					GST_BUFFER_DURATION(*buffer) = 0;
					GST_BUFFER_OFFSET(*buffer) = offset;
					GST_BUFFER_OFFSET_END(*buffer) = GST_BUFFER_OFFSET_NONE;
					element->next_timestamp = GST_BUFFER_TIMESTAMP(*buffer) + GST_BUFFER_DURATION(*buffer);
					return result;
				case EINTR:
					GST_WARNING_OBJECT(element, "EINTR received from gst_buffer(), could be due to a signal or failure to get write control over the partition.");
					/*
					 * Wait a bit and try again just in case
					 */
					usleep(5 * 1000);
					break;
				case EBUSY:
					GST_ELEMENT_ERROR(element, RESOURCE, READ, (NULL), ("EBUSY received from gst_buffer(), the consumer already has a buffer assigned."));
					return GST_FLOW_UNEXPECTED;
				case EINVAL:
					GST_ELEMENT_ERROR(element, RESOURCE, READ, (NULL), ("EINVAL received from gst_buffer(), the consumer instance is not attached to a partition."));
					return GST_FLOW_UNEXPECTED;
				case ENOENT:
					GST_ELEMENT_ERROR(element, RESOURCE, READ, (NULL), ("ENOENT received from gst_buffer(), findDataID could not find a buffer with the requested ID."));
					return GST_FLOW_UNEXPECTED;
				default:
					/*
					 * reason for failure is not known.
					 * indicate end-of-stream
					 */
					GST_ELEMENT_ERROR(element, RESOURCE, READ, (NULL), ("unknown failure retrieving buffer from GDS shared memory. errno: [%d]", save_errno));
					return GST_FLOW_UNEXPECTED;
				}
			}
		}

		/*
		 * we have successfully retrieved data.  NOTE:  all code
		 * paths from this point *must* include a call to
		 * lsmp_partition(element)->free_buffer()
		 */

		length = lsmp_partition(element)->getLength();
		timestamp = lsmp_partition(element)->getEvtID() * GST_SECOND;
		GST_LOG_OBJECT(element, "retrieved %u byte shared-memory buffer %p for GPS %" GST_TIME_SECONDS_FORMAT, length, data, GST_TIME_SECONDS_ARGS(timestamp));
		if(timestamp >= element->next_timestamp)
			break;
		GST_LOG_OBJECT(element, "time reversal.  skipping buffer.");
		lsmp_partition(element)->free_buffer();
	}

	/*
	 * copy into a GstBuffer
	 */

	if(!length) {
		GST_ELEMENT_ERROR(element, RESOURCE, READ, (NULL), ("received 0-byte shared-memory buffer"));
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
	GST_BUFFER_DURATION(*buffer) = element->assumed_duration * GST_SECOND;	/* FIXME:  we need to know this! */
	GST_BUFFER_OFFSET(*buffer) = offset;
	GST_BUFFER_OFFSET_END(*buffer) = GST_BUFFER_OFFSET_NONE;
	element->next_timestamp = GST_BUFFER_TIMESTAMP(*buffer) + GST_BUFFER_DURATION(*buffer);

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
	lsmp_partition(element)->free_buffer();
	GST_LOG_OBJECT(element, "released shared-memory buffer %p", data);
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
	case GST_QUERY_FORMATS:
		gst_query_set_formats(query, 1, GST_FORMAT_TIME);
		break;

	case GST_QUERY_LATENCY:
		gst_query_set_latency(query, gst_base_src_is_live(basesrc), element->min_latency, element->max_latency);
		break;

	case GST_QUERY_POSITION:
		/* timestamp of next buffer */
		gst_query_set_position(query, GST_FORMAT_TIME, element->next_timestamp);
		break;

#if 0
	case GST_QUERY_SEGMENT:
		gst_query_set_segment(query, 1.0, GST_FORMAT_TIME, GST_CLOCK_TIME_NONE, GST_CLOCK_TIME_NONE);
		break;
#endif

	default:
		success = GST_BASE_SRC_CLASS(parent_class)->query(basesrc, query);
		break;
	}

	if(success)
		GST_DEBUG_OBJECT(element, "result: %" GST_PTR_FORMAT, query);
	else
		GST_ERROR_OBJECT(element, "query failed");

	return success;
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
	ARG_MASK,
	ARG_WAIT_TIME,
	ARG_ASSUMED_DURATION
};


static void set_property(GObject *object, guint id, const GValue *value, GParamSpec *pspec)
{
	GDSLVSHMSrc *element = GDS_LVSHMSRC(object);

	GST_OBJECT_LOCK(element);

	switch((enum property) id) {
	case ARG_SHM_NAME:
		g_free(element->name);
		element->name = g_value_dup_string(value);
		break;

	case ARG_MASK:
		element->mask = g_value_get_uint(value);
		break;

	case ARG_WAIT_TIME:
		element->wait_time = g_value_get_double(value);
		if(element->partition)
			lsmp_partition(element)->setTimeout(element->wait_time);
		break;

	case ARG_ASSUMED_DURATION:
		element->assumed_duration = g_value_get_uint(value);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, guint id, GValue *value, GParamSpec *pspec)
{
	GDSLVSHMSrc *element = GDS_LVSHMSRC(object);

	GST_OBJECT_LOCK(element);

	switch((enum property) id) {
	case ARG_SHM_NAME:
		g_value_set_string(value, element->name);
		break;

	case ARG_MASK:
		g_value_set_uint(value, element->mask);
		break;

	case ARG_WAIT_TIME:
		g_value_set_double(value, element->wait_time);
		break;

	case ARG_ASSUMED_DURATION:
		g_value_set_uint(value, element->assumed_duration);
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
	GDSLVSHMSrc *element = GDS_LVSHMSRC(object);

	g_free(element->name);
	element->name = NULL;
	g_mutex_free(element->create_thread_lock);
	element->create_thread_lock = NULL;
	if(element->partition) {
		GST_WARNING_OBJECT(element, "parent class failed to invoke stop() method.  doing shared-memory de-access in finalize() instead.");
		delete lsmp_partition(element);
		element->partition = NULL;
	}

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


/*
 * base_init()
 */


static void gds_lvshmsrc_base_init(gpointer klass)
{
}


/*
 * class_init()
 */


static void gds_lvshmsrc_class_init(GDSLVSHMSrcClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GstBaseSrcClass *gstbasesrc_class = GST_BASE_SRC_CLASS(klass);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	gstbasesrc_class->start = GST_DEBUG_FUNCPTR(start);
	gstbasesrc_class->stop = GST_DEBUG_FUNCPTR(stop);
	gstbasesrc_class->unlock = GST_DEBUG_FUNCPTR(unlock);
	gstbasesrc_class->unlock_stop = GST_DEBUG_FUNCPTR(unlock_stop);
	gstbasesrc_class->create = GST_DEBUG_FUNCPTR(create);
	gstbasesrc_class->query = GST_DEBUG_FUNCPTR(query);

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
		ARG_MASK,
		g_param_spec_uint(
			"mask",
			"Trigger mask",
			"Buffers will be received only from producers whose masks' bit-wise logical and's with this value are non-zero.",
			0, G_MAXUINT, DEFAULT_MASK,
			(GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)
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
			(GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_ASSUMED_DURATION,
		g_param_spec_uint(
			"assumed-duration",
			"Assumed duration",
			"Assume all files span this much time in seconds.",
			1, G_MAXUINT, DEFAULT_ASSUMED_DURATION,
			(GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)
		)
	);
}


/*
 * init()
 */


static void gds_lvshmsrc_init(GDSLVSHMSrc *element, GDSLVSHMSrcClass *klass)
{
	GstBaseSrc *basesrc = GST_BASE_SRC(element);

	gst_base_src_set_live(basesrc, TRUE);
	gst_base_src_set_format(basesrc, GST_FORMAT_TIME);

	/*
	 * internal data
	 */

	element->name = NULL;
	element->max_latency = element->min_latency = GST_CLOCK_TIME_NONE;
	element->unblocked = FALSE;
	element->create_thread_lock = g_mutex_new();
	element->partition = NULL;
}
 

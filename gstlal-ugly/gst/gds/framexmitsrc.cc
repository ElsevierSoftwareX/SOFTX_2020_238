/*
 * GDS framexmit broadcast receiver source element
 *
 * Copyright (C) 2012.2013  Kipp Cannon
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
 * SECTION:framexmitsrc
 * @short_description:  GDS framexmit broadcast receiver source element.
 *
 * Reviewed:  00d65a70accca228bb76bd07e89b3ec07c78f736 2014-08-13 K.
 * Cannon, J.  Creighton, B. Sathyaprakash.
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
#include <stdio.h>
#include <stdlib.h>
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


#include <gds/framexmit/framerecv.hh>
#include <gds/tconv.h>


/*
 * our own stuff
 */


#include <gstlal/gstlal_debug.h>
#include <framexmitsrc.h>


#define FRAMERCV(element) ((framexmit::frameRecv *) element->frameRecv)


/*
 * ========================================================================
 *
 *                               GstURIHandler
 *
 * ========================================================================
 */


#define URI_SCHEME "framexmit"


static GstURIType uri_get_type(GType type)
{
	return GST_URI_SRC;
}


static const gchar *const *uri_get_protocols(GType type)
{
	static const gchar *protocols[] = {URI_SCHEME, NULL};

	return protocols;
}


static gchar *uri_get_uri(GstURIHandler *handler)
{
	GstGDSFramexmitSrc *element = GDS_FRAMEXMITSRC(handler);

	return g_strdup_printf(URI_SCHEME "://%s:%d", element->group, element->port);
}


static gboolean uri_set_uri(GstURIHandler *handler, const gchar *uri, GError **err)
{
	/* FIXME:  report errors via GError argument */
	GstGDSFramexmitSrc *element = GDS_FRAMEXMITSRC(handler);
	gchar *scheme = g_uri_parse_scheme(uri);
	gchar group[strlen(uri)];
	gint port;
	gboolean success = TRUE;

	success = !strcmp(scheme, URI_SCHEME);
	if(success)
		success &= sscanf(uri, URI_SCHEME "://%[^:]:%d", group, &port) == 2;
	if(success)
		g_object_set(G_OBJECT(element), "multicast-group", group, "port", port, NULL);

	g_free(scheme);
	return success;
}


static void uri_handler_init(gpointer g_iface, gpointer iface_data)
{
	GstURIHandlerInterface *iface = (GstURIHandlerInterface *) g_iface;

	iface->get_uri = GST_DEBUG_FUNCPTR(uri_get_uri);
	iface->set_uri = GST_DEBUG_FUNCPTR(uri_set_uri);
	iface->get_type = GST_DEBUG_FUNCPTR(uri_get_type);
	iface->get_protocols = GST_DEBUG_FUNCPTR(uri_get_protocols);
}


/*
 * ========================================================================
 *
 *                                Boilerplate
 *
 * ========================================================================
 */


#define GST_CAT_DEFAULT gds_framexmitsrc_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


static void additional_initializations(GType type)
{
	static const GInterfaceInfo uri_handler_info = {
		uri_handler_init,
		NULL,
		NULL
	};
	g_type_add_interface_static(type, GST_TYPE_URI_HANDLER, &uri_handler_info);

	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "gds_framexmitsrc", 0, "gds_framexmitsrc element");
}


G_DEFINE_TYPE_WITH_CODE(
	GstGDSFramexmitSrc,
	gds_framexmitsrc,
	GST_TYPE_PUSH_SRC,
	additional_initializations(g_define_type_id)
);


/*
 * ========================================================================
 *
 *                                 Parameters
 *
 * ========================================================================
 */


#define DEFAULT_MULTICAST_IFACE NULL
#define DEFAULT_MULTICAST_GROUP "0.0.0.0"
#define DEFAULT_PORT 0
#define DEFAULT_QOS 2
#define DEFAULT_WAIT_TIME -1.0


/*
 * ========================================================================
 *
 *                         GDS_FRAMEXMITSRC_QOS_TYPE
 *
 * ========================================================================
 */


GType gds_framexmitsrc_qos_get_type(void)
{
	static GType type = 0;

	if(!type) {
		static GEnumValue values[] = {
			{GDS_FRAMEXMITSRC_QOS_30, "GDS_FRAMEXMITSRC_QOS_30", "Retransmission rate <= 30%"},
			{GDS_FRAMEXMITSRC_QOS_10, "GDS_FRAMEXMITSRC_QOS_10", "Retransmission rate <= 10%"},
			{GDS_FRAMEXMITSRC_QOS_3, "GDS_FRAMEXMITSRC_QOS_3", "Retransmission rate <= 3%"},
			{0, NULL, NULL}
		};

		type = g_enum_register_static("gds_framexmitsrc_qos", values);
	}

	return type;
}


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
 *                               Receive Thread
 *
 * ============================================================================
 */


static void *receive_thread(void *arg)
{
	GstGDSFramexmitSrc *element = GDS_FRAMEXMITSRC(arg);

	pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL);

	/* FIXME:  this will leak memory if killed at the wrong spot */
	while(TRUE) {
		GstBuffer *buffer;
		char *data = NULL;
		int len;
		unsigned sequence, timestamp, duration;

		len = FRAMERCV(element)->receive(data, 0, &sequence, &timestamp, &duration);
		if(len < 0) {
			free(data);
			GST_ELEMENT_ERROR(element, RESOURCE, FAILED, (NULL), ("framexmit::frameRecv.receive() failed"));
			g_mutex_lock(&element->buffer_lock);
			element->recv_status = GST_FLOW_ERROR;
			g_cond_signal(&element->received_buffer);
			g_mutex_unlock(&element->buffer_lock);
			break;
		}
		GST_DEBUG_OBJECT(element, "recieved %d byte buffer (seq. #%u) for [%u s, %u s)", len, sequence, timestamp, timestamp + duration);

		buffer = gst_buffer_new_wrapped_full(
			GST_MEMORY_FLAG_PHYSICALLY_CONTIGUOUS,
			data,
			len,
			0, len,
			data, free
		);
		GST_BUFFER_TIMESTAMP(buffer) = timestamp * GST_SECOND;
		GST_BUFFER_DURATION(buffer) = duration * GST_SECOND;
		GST_BUFFER_OFFSET(buffer) = sequence;
		GST_BUFFER_OFFSET_END(buffer) = sequence + 1;

		g_mutex_lock(&element->buffer_lock);
		if(element->buffer) {
			GST_WARNING_OBJECT(element, "receive thread overrun, dropping %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(element->buffer));
			gst_buffer_unref(element->buffer);
		}
		element->buffer = buffer;
		g_cond_signal(&element->received_buffer);
		g_mutex_unlock(&element->buffer_lock);
	}

	return NULL;
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
	GstGDSFramexmitSrc *element = GDS_FRAMEXMITSRC(object);
	int retval;
	gboolean success = TRUE;

	g_assert(element->buffer == NULL);

	element->frameRecv = new framexmit::frameRecv(element->qos);
	success = FRAMERCV(element)->open(element->group, element->iface, element->port);
	if(!success) {
		GST_ELEMENT_ERROR(element, RESOURCE, FAILED, (NULL), ("framexmit::frameRecv.open(group = \"%s\", iface = \"%s\", port = %d) failed", element->group, element->iface, element->port));
		FRAMERCV(element)->close();
		element->frameRecv = NULL;
		goto done;
	} else
		GST_DEBUG_OBJECT(element, "framexmit::frameRecv.open(group = \"%s\", iface = \"%s\", port = %d) succeeded", element->group, element->iface, element->port);

	element->recv_status = GST_FLOW_OK;
	retval = pthread_create(&element->recv_thread, NULL, receive_thread, element);
	success = retval == 0;
	if(!success) {
		GST_ELEMENT_ERROR(element, RESOURCE, FAILED, (NULL), ("failure creating receiver thread (%d): %s", retval, strerror(retval)));
		FRAMERCV(element)->close();
		element->frameRecv = NULL;
		goto done;
	}

	element->need_new_segment = TRUE;
	element->next_timestamp = GST_CLOCK_TIME_NONE;

done:
	gst_base_src_start_complete(object, success ? GST_FLOW_OK : GST_FLOW_ERROR);
	return success;
}


/*
 * stop()
 */


static gboolean stop(GstBaseSrc *object)
{
	GstGDSFramexmitSrc *element = GDS_FRAMEXMITSRC(object);
	gboolean success = TRUE;

	success &= pthread_cancel(element->recv_thread) == 0;
	success &= pthread_join(element->recv_thread, NULL) == 0;
	g_mutex_lock(&element->buffer_lock);
	if(element->buffer) {
		gst_buffer_unref(element->buffer);
		element->buffer = NULL;
	}
	g_mutex_unlock(&element->buffer_lock);

	FRAMERCV(element)->close();
	delete FRAMERCV(element);
	element->frameRecv = NULL;
	GST_DEBUG_OBJECT(element, "framexmit::frameRecv.close()");

	element->max_latency = element->min_latency = GST_CLOCK_TIME_NONE;

	return success;
}


/*
 * unlock()
 */


static gboolean unlock(GstBaseSrc *basesrc)
{
	GstGDSFramexmitSrc *element = GDS_FRAMEXMITSRC(basesrc);
	gboolean success = TRUE;

	g_mutex_lock(&element->buffer_lock);
	element->unblocked = TRUE;
	g_cond_signal(&element->received_buffer);
	g_mutex_unlock(&element->buffer_lock);

	return success;
}


/*
 * unlock_stop()
 */


static gboolean unlock_stop(GstBaseSrc *basesrc)
{
	GstGDSFramexmitSrc *element = GDS_FRAMEXMITSRC(basesrc);
	gboolean success = TRUE;

	g_mutex_lock(&element->buffer_lock);
	element->unblocked = FALSE;
	g_mutex_unlock(&element->buffer_lock);

	return success;
}


/*
 * create()
 */


static GstFlowReturn create(GstBaseSrc *basesrc, guint64 offset, guint size, GstBuffer **buffer)
{
	GstGDSFramexmitSrc *element = GDS_FRAMEXMITSRC(basesrc);
	GstClockTime t_before;
	gboolean timeout;
	GstFlowReturn result = GST_FLOW_OK;

	/*
	 * retrieve data
	 */

try_again:
	g_mutex_lock(&element->buffer_lock);
	timeout = FALSE;
	t_before = GST_CLOCK_TIME_NONE;
	while(!element->buffer && !element->unblocked && !timeout && element->recv_status == GST_FLOW_OK) {
		t_before = GPSNow();
		GST_DEBUG_OBJECT(element, "waiting for data at %" GST_TIME_SECONDS_FORMAT, GST_TIME_SECONDS_ARGS(t_before));
		if(element->wait_time >= 0)
			timeout = !g_cond_wait_until(&element->received_buffer, &element->buffer_lock, g_get_monotonic_time() + element->wait_time * G_TIME_SPAN_SECOND);
		else
			g_cond_wait(&element->received_buffer, &element->buffer_lock);
	}
	*buffer = element->buffer;
	element->buffer = NULL;
	g_mutex_unlock(&element->buffer_lock);

	/*
	 * if no data, try to guess cause
	 */

	if(!*buffer) {
		GST_DEBUG_OBJECT(element, "no data available");

		/*
		 * failure in receive thread?
		 */

		if(element->recv_status != GST_FLOW_OK) {
			GST_ERROR_OBJECT(element, "reason: error in receive thread");
			result = element->recv_status;
			goto done;
		}

		/*
		 * application shutting us down?
		 */

		else if(element->unblocked) {
			GST_DEBUG_OBJECT(element, "reason: application invoked unlock()");
			result = GST_FLOW_EOS;
			goto done;
		}

		/*
		 * timeout?  create a 0-length buffer with a guess as to
		 * the timestamp of the missing data.  guess:  the time
		 * when we started waiting for the data adjusted by the
		 * most recently measured latency
		 */

		else if(timeout) {
			GST_WARNING_OBJECT(element, "reason: timeout");
			g_assert(GST_CLOCK_TIME_IS_VALID(t_before));

			*buffer = gst_buffer_new();
			GST_BUFFER_TIMESTAMP(*buffer) = t_before;
			if(GST_CLOCK_TIME_IS_VALID(element->max_latency))
				GST_BUFFER_TIMESTAMP(*buffer) -= element->max_latency;
			GST_BUFFER_DURATION(*buffer) = 0;
			GST_BUFFER_OFFSET(*buffer) = GST_BUFFER_OFFSET_NONE;
			GST_BUFFER_OFFSET_END(*buffer) = GST_BUFFER_OFFSET_NONE;

			GST_DEBUG_OBJECT(element, "created 0-length heartbeat buffer with timestamp = %" GST_TIME_SECONDS_FORMAT, GST_TIME_SECONDS_ARGS(GST_BUFFER_TIMESTAMP(*buffer)));
			if(GST_CLOCK_TIME_IS_VALID(element->next_timestamp) && GST_BUFFER_TIMESTAMP(*buffer) < element->next_timestamp) {
				GST_LOG_OBJECT(element, "timestamp reversal.  trying again to wait for data.");
				gst_buffer_unref(*buffer);
				*buffer = NULL;
				goto try_again;
			}
		}

		/*
		 * there are no other possible causes of failure
		 */

		else
			g_assert_not_reached();
	} else
		/*
		 * check for logical impossibility
		 */

		g_assert(!timeout);

	/*
	 * check for disconts
	 */

	if(!GST_CLOCK_TIME_IS_VALID(element->next_timestamp) || GST_BUFFER_TIMESTAMP(*buffer) != element->next_timestamp) {
		GST_BUFFER_FLAG_SET(*buffer, GST_BUFFER_FLAG_DISCONT);
		GST_WARNING_OBJECT(element, "discontinuity @ %" GST_TIME_SECONDS_FORMAT, GST_TIME_SECONDS_ARGS(GST_BUFFER_TIMESTAMP(*buffer)));
	}
	element->next_timestamp = GST_BUFFER_TIMESTAMP(*buffer) + GST_BUFFER_DURATION(*buffer);

	/*
	 * update latency
	 */

	if(!timeout) {
		element->max_latency = GST_CLOCK_DIFF(GST_BUFFER_TIMESTAMP(*buffer), GPSNow());
		element->min_latency = element->max_latency - GST_BUFFER_DURATION(*buffer);
	}
	GST_DEBUG_OBJECT(element, "latency = [%" GST_TIME_SECONDS_FORMAT ", %" GST_TIME_SECONDS_FORMAT ")", GST_TIME_SECONDS_ARGS(element->min_latency), GST_TIME_SECONDS_ARGS(element->max_latency));

	/*
	 * adjust segment
	 */

	if(element->need_new_segment) {
		gst_base_src_new_seamless_segment(basesrc, GST_BUFFER_TIMESTAMP(*buffer), GST_CLOCK_TIME_NONE, GST_BUFFER_TIMESTAMP(*buffer));
		element->need_new_segment = FALSE;
	}

	/*
	 * done
	 */

done:
	return result;
}


/*
 * query()
 */


static gboolean query(GstBaseSrc *basesrc, GstQuery *query)
{
	GstGDSFramexmitSrc *element = GDS_FRAMEXMITSRC(basesrc);
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
		success = GST_BASE_SRC_CLASS(gds_framexmitsrc_parent_class)->query(basesrc, query);
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
	ARG_MULTICAST_IFACE = 1,
	ARG_MULTICAST_GROUP,
	ARG_PORT,
	ARG_QOS,
	ARG_WAIT_TIME,
};


static void set_property(GObject *object, guint id, const GValue *value, GParamSpec *pspec)
{
	GstGDSFramexmitSrc *element = GDS_FRAMEXMITSRC(object);

	GST_OBJECT_LOCK(element);

	switch((enum property) id) {
	case ARG_MULTICAST_IFACE: {
		const gchar *iface;
		g_free(element->iface);
		if((iface = g_value_get_string(value)))
			element->iface = g_strdup(iface);
		else
			element->iface = g_strdup(DEFAULT_MULTICAST_IFACE);
		break;
	}

	case ARG_MULTICAST_GROUP: {
		const gchar *group;
		g_free(element->group);
		if((group = g_value_get_string(value)))
			element->group = g_strdup(group);
		else
			element->group = g_strdup(DEFAULT_MULTICAST_GROUP);
		break;
	}

	case ARG_PORT:
		element->port = g_value_get_int(value);
		break;

	case ARG_QOS:
		element->qos = (enum gds_framexmitsrc_qos) g_value_get_enum(value);
		break;

	case ARG_WAIT_TIME:
		element->wait_time = g_value_get_double(value);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, guint id, GValue *value, GParamSpec *pspec)
{
	GstGDSFramexmitSrc *element = GDS_FRAMEXMITSRC(object);

	GST_OBJECT_LOCK(element);

	switch((enum property) id) {
	case ARG_MULTICAST_IFACE:
		g_value_set_string(value, element->iface);
		break;

	case ARG_MULTICAST_GROUP:
		g_value_set_string(value, element->group);
		break;

	case ARG_PORT:
		g_value_set_int(value, element->port);
		break;

	case ARG_QOS:
		g_value_set_enum(value, element->qos);
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
 * finalize()
 */


static void finalize(GObject *object)
{
	GstGDSFramexmitSrc *element = GDS_FRAMEXMITSRC(object);

	if(element->buffer) {
		gst_buffer_unref(element->buffer);
		element->buffer = NULL;
	}
	g_mutex_clear(&element->buffer_lock);
	g_cond_clear(&element->received_buffer);
	g_free(element->iface);
	element->iface = NULL;
	g_free(element->group);
	element->group = NULL;

	G_OBJECT_CLASS(gds_framexmitsrc_parent_class)->finalize(object);
}


/*
 * class_init()
 */


static void gds_framexmitsrc_class_init(GstGDSFramexmitSrcClass *klass)
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
		"GDS Framexmit Receiver Source",
		"Source",
		"GDS framexmit broadcast receiver source element",
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
		ARG_MULTICAST_IFACE,
		g_param_spec_string(
			"multicast-iface",
			"IP address",
			"The network interface on which to join the multicast group.  If the interface is obmitted, the default interface will be used.  In general, one can use the subnet address as the interface address argument. The receiver will go through the list of all local interfaces and determine the closest match.",
			DEFAULT_MULTICAST_IFACE,
			(GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_MULTICAST_GROUP,
		g_param_spec_string(
			"multicast-group",
			"IP address",
			"The address of multicast group to join.  If no multicast address is supplied, the receiver will listen for UDP/IP broadcast transmissions at the specified port.",
			DEFAULT_MULTICAST_GROUP,
			(GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_PORT,
		g_param_spec_int(
			"port",
			"Port",
			"The local port on which to receive broadcasts (0 = allocate).  These ports can be reused by multiple applications.",
			0, 65535, DEFAULT_PORT,
			(GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_QOS,
		g_param_spec_enum(
			"qos",
			"QoS",
			"Quality of service limit.",
			GDS_FRAMEXMITSRC_QOS_TYPE, DEFAULT_QOS,
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
}


/*
 * init()
 */


static void gds_framexmitsrc_init(GstGDSFramexmitSrc *element)
{
	GstBaseSrc *basesrc = GST_BASE_SRC(element);

	gst_base_src_set_live(basesrc, TRUE);
	gst_base_src_set_format(basesrc, GST_FORMAT_TIME);

	/*
	 * receive thread
	 */

	element->buffer = NULL;
	g_mutex_init(&element->buffer_lock);
	g_cond_init(&element->received_buffer);
	element->unblocked = FALSE;

	/*
	 * internal data
	 */

	element->iface = NULL;
	element->group = NULL;
	element->max_latency = element->min_latency = GST_CLOCK_TIME_NONE;
	element->frameRecv = NULL;
}

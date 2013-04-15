/*
 * GDS framexmit broadcast receiver source element
 *
 * Copyright (C) 2012  Kipp Cannon
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


#include <signal.h>
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


#define URI_SCHEME "framexmit"


#define FRAMERCV(element) ((framexmit::frameRecv *) element->frameRecv)


/*
 * ========================================================================
 *
 *                               GstURIHandler
 *
 * ========================================================================
 */


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
	GstGDSFramexmitSrc *element = GDS_FRAMEXMITSRC(handler);

	/* 1.0:  this won't be a memory leak */
	return g_strdup_printf(URI_SCHEME "://%s:%d", element->group, element->port);
}


/* 1.0:  this gets a GError ** argument */
static gboolean uri_set_uri(GstURIHandler *handler, const gchar *uri)
{
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


GST_BOILERPLATE_FULL(GstGDSFramexmitSrc, gds_framexmitsrc, GstPushSrc, GST_TYPE_PUSH_SRC, additional_initializations);


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
	gboolean success = TRUE;

	element->frameRecv = new framexmit::frameRecv(element->qos);
	success = FRAMERCV(element)->open(element->group, element->iface, element->port);
	if(!success) {
		GST_ELEMENT_ERROR(element, RESOURCE, FAILED, (NULL), ("framexmit::frameRecv.open(group = \"%s\", iface = \"%s\", port = %d) failed", element->group, element->iface, element->port));
		FRAMERCV(element)->close();
		element->frameRecv = NULL;
		goto done;
	}

	element->need_new_segment = TRUE;
	element->next_timestamp = GST_CLOCK_TIME_NONE;

done:
	return success;
}


/*
 * stop()
 */


static gboolean stop(GstBaseSrc *object)
{
	GstGDSFramexmitSrc *element = GDS_FRAMEXMITSRC(object);

	FRAMERCV(element)->close();
	delete FRAMERCV(element);
	element->frameRecv = NULL;

	element->max_latency = element->min_latency = GST_CLOCK_TIME_NONE;

	return TRUE;
}


/*
 * unlock()
 */


static gboolean unlock(GstBaseSrc *basesrc)
{
	GstGDSFramexmitSrc *element = GDS_FRAMEXMITSRC(basesrc);
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
	GstGDSFramexmitSrc *element = GDS_FRAMEXMITSRC(basesrc);
	gboolean success = TRUE;

	element->unblocked = FALSE;

	return success;
}


/*
 * create()
 */


static GstFlowReturn create(GstBaseSrc *basesrc, guint64 offset, guint size, GstBuffer **buffer)
{
	GstGDSFramexmitSrc *element = GDS_FRAMEXMITSRC(basesrc);
	char *data = NULL;
	int len;
	unsigned sequence, timestamp, duration;
	GstFlowReturn result = GST_FLOW_OK;

	if(element->unblocked) {
		GST_DEBUG_OBJECT(element, "unlock() called, no buffer created");
		result = GST_FLOW_UNEXPECTED;
		goto done;
	}

	/*
	 * retrieve data
	 */

	element->create_thread = pthread_self();
	g_mutex_lock(element->create_thread_lock);
	len = FRAMERCV(element)->receive(data, 0, &sequence, &timestamp, &duration);
	g_mutex_unlock(element->create_thread_lock);

	if(len < 0) {
		if(element->unblocked) {
			GST_DEBUG_OBJECT(element, "unlock() called, no buffer created");
			result = GST_FLOW_UNEXPECTED;
		} else {
			GST_ELEMENT_ERROR(element, RESOURCE, FAILED, (NULL), ("framexmit::frameRecv.receive() failed"));
			result = GST_FLOW_ERROR;
		}
		goto done;
	} else
		GST_DEBUG_OBJECT(element, "recieved %d byte buffer for [%u s, %u s)", len, timestamp, timestamp + duration);

	/*
	 * prepare output
	 */

	*buffer = gst_buffer_new();
	g_assert(*buffer != NULL);
	gst_buffer_set_caps(*buffer, GST_PAD_CAPS(GST_BASE_SRC_PAD(basesrc)));
	GST_BUFFER_DATA(*buffer) = GST_BUFFER_MALLOCDATA(*buffer) = (guint8 *) data;
	GST_BUFFER_SIZE(*buffer) = len;
	GST_BUFFER_TIMESTAMP(*buffer) = timestamp * GST_SECOND;
	GST_BUFFER_DURATION(*buffer) = duration * GST_SECOND;
	GST_BUFFER_OFFSET(*buffer) = sequence;
	GST_BUFFER_OFFSET_END(*buffer) = sequence + 1;
	if(!GST_CLOCK_TIME_IS_VALID(element->next_timestamp) || GST_BUFFER_TIMESTAMP(*buffer) != element->next_timestamp)
		GST_BUFFER_FLAG_SET(*buffer, GST_BUFFER_FLAG_DISCONT);

	element->next_timestamp = GST_BUFFER_TIMESTAMP(*buffer) + GST_BUFFER_DURATION(*buffer);

	/*
	 * update latency
	 */

	element->max_latency = GST_CLOCK_DIFF(GST_BUFFER_TIMESTAMP(*buffer), GPSNow());
	element->min_latency = element->max_latency - GST_BUFFER_DURATION(*buffer);
	GST_DEBUG_OBJECT(element, "latency = %" G_GINT64_FORMAT " ns -- %" G_GINT64_FORMAT " ns", element->min_latency, element->max_latency);

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


/*
 * properties
 */


enum property {
	ARG_MULTICAST_IFACE = 1,
	ARG_MULTICAST_GROUP,
	ARG_PORT,
	ARG_QOS
};


static void set_property(GObject *object, guint id, const GValue *value, GParamSpec *pspec)
{
	GstGDSFramexmitSrc *element = GDS_FRAMEXMITSRC(object);

	GST_OBJECT_LOCK(element);

	switch((enum property) id) {
	case ARG_MULTICAST_IFACE: {
		const gchar *group;
		g_free(element->iface);
		if((group = g_value_get_string(value)))
			element->iface = g_strdup(group);
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
	GstGDSFramexmitSrc *element = GDS_FRAMEXMITSRC(object);

	g_free(element->iface);
	element->iface = NULL;
	g_free(element->group);
	element->group = NULL;
	g_mutex_free(element->create_thread_lock);
	element->create_thread_lock = NULL;

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


/*
 * Base init function.
 */


static void gds_framexmitsrc_base_init(gpointer klass)
{
}


/*
 * Class init function.
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
			"The network interface on which to join the multicast group.",
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
			"The address of multicast group to join.",
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
			"The port to receive the packets from (0 = allocate).",
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
}


/*
 * Instance init function.
 */


static void gds_framexmitsrc_init(GstGDSFramexmitSrc *element, GstGDSFramexmitSrcClass *klass)
{
	GstBaseSrc *basesrc = GST_BASE_SRC(element);

	gst_base_src_set_live(basesrc, TRUE);
	gst_base_src_set_format(basesrc, GST_FORMAT_TIME);

	/*
	 * internal data
	 */

	element->iface = NULL;
	element->group = NULL;
	element->max_latency = element->min_latency = GST_CLOCK_TIME_NONE;
	element->unblocked = FALSE;
	element->create_thread_lock = g_mutex_new();
	element->frameRecv = NULL;
}

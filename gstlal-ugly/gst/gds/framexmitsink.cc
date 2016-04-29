/*
 * GDS framexmit broadcast transmitter sink element
 *
 * Copyright (C) 2012--2013  Kipp Cannon
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
#include <stdio.h>
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


#include <gds/framexmit/framesend.hh>
#include <gds/tconv.h>


/*
 * our own stuff
 */


#include <gstlal/gstlal_debug.h>
#include <framexmitsink.h>


#define FRAMESEND(element) ((framexmit::frameSend *) element->frameSend)


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
	return GST_URI_SINK;
}


static const gchar *const *uri_get_protocols(GType type)
{
	static const gchar *protocols[] = {URI_SCHEME, NULL};

	return protocols;
}


static gchar *uri_get_uri(GstURIHandler *handler)
{
	GstGDSFramexmitSink *element = GDS_FRAMEXMITSINK(handler);

	return g_strdup_printf(URI_SCHEME "://%s:%d", element->group, element->port);
}


static gboolean uri_set_uri(GstURIHandler *handler, const gchar *uri, GError **error)
{
	/* FIXME:  report errors via err argument */
	GstGDSFramexmitSink *element = GDS_FRAMEXMITSINK(handler);
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


#define GST_CAT_DEFAULT gds_framexmitsink_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


static void additional_initializations(GType type)
{
	static const GInterfaceInfo uri_handler_info = {
		uri_handler_init,
		NULL,
		NULL
	};
	g_type_add_interface_static(type, GST_TYPE_URI_HANDLER, &uri_handler_info);

	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "gds_framexmitsink", 0, "gds_framexmitsink element");
}


G_DEFINE_TYPE_WITH_CODE(
	GstGDSFramexmitSink,
	gds_framexmitsink,
	GST_TYPE_BASE_SINK,
	additional_initializations(g_define_type_id)
);


/*
 * ========================================================================
 *
 *                                 Parameters
 *
 * ========================================================================
 */


#define DEFAULT_MAX_SIZE_BUFFERS 8
#define DEFAULT_MULTICAST_IFACE NULL
#define DEFAULT_MULTICAST_GROUP "0.0.0.0"
#define DEFAULT_PORT 0


/*
 * ============================================================================
 *
 *                        GstBaseSink Method Overrides
 *
 * ============================================================================
 */


/*
 * start()
 */


static gboolean start(GstBaseSink *object)
{
	GstGDSFramexmitSink *element = GDS_FRAMEXMITSINK(object);
	gboolean success = TRUE;

	element->frameSend = new framexmit::frameSend(element->max_size_buffers);
	success = FRAMESEND(element)->open(element->group, element->iface, element->port);
	if(!success) {
		GST_ELEMENT_ERROR(element, RESOURCE, FAILED, (NULL), ("framexmit::frameSend.open(group = \"%s\", iface = \"%s\", port = %d) failed", element->group, element->iface, element->port));
		FRAMESEND(element)->close();
		element->frameSend = NULL;
		goto done;
	} else
		GST_DEBUG_OBJECT(element, "framexmit::frameSend.open(group = \"%s\", iface = \"%s\", port = %d) succeeded", element->group, element->iface, element->port);

done:
	return success;
}


/*
 * stop()
 */


static gboolean stop(GstBaseSink *object)
{
	GstGDSFramexmitSink *element = GDS_FRAMEXMITSINK(object);

	FRAMESEND(element)->close();
	delete FRAMESEND(element);
	element->frameSend = NULL;
	GST_DEBUG_OBJECT(element, "framexmit::frameSend.close()");

	return TRUE;
}


/*
 * render()
 */


static GstFlowReturn render(GstBaseSink *basesink, GstBuffer *buffer)
{
	GstGDSFramexmitSink *element = GDS_FRAMEXMITSINK(basesink);
	GstMapInfo mapinfo;
	GstFlowReturn result = GST_FLOW_OK;

	/*
	 * send data
	 */

	gst_buffer_map(buffer, &mapinfo, GST_MAP_READ);

	GST_DEBUG_OBJECT(element, "sending %" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(buffer));
	if(!FRAMESEND(element)->send((char *) mapinfo.data, mapinfo.size, NULL, TRUE, GST_BUFFER_TIMESTAMP(buffer) / GST_SECOND, GST_BUFFER_DURATION(buffer) / GST_SECOND)) {
		GST_ELEMENT_ERROR(element, RESOURCE, FAILED, (NULL), ("framexmit::frameSend.send() failed"));
		result = GST_FLOW_ERROR;
	}

	gst_buffer_unmap(buffer, &mapinfo);

	/*
	 * done
	 */

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
	ARG_MAX_SIZE_BUFFERS = 1,
	ARG_MULTICAST_IFACE,
	ARG_MULTICAST_GROUP,
	ARG_PORT,
};


static void set_property(GObject *object, guint id, const GValue *value, GParamSpec *pspec)
{
	GstGDSFramexmitSink *element = GDS_FRAMEXMITSINK(object);

	GST_OBJECT_LOCK(element);

	switch((enum property) id) {
	case ARG_MAX_SIZE_BUFFERS:
		element->max_size_buffers = g_value_get_int(value);
		break;

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

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, guint id, GValue *value, GParamSpec *pspec)
{
	GstGDSFramexmitSink *element = GDS_FRAMEXMITSINK(object);

	GST_OBJECT_LOCK(element);

	switch((enum property) id) {
	case ARG_MAX_SIZE_BUFFERS:
		g_value_set_int(value, element->max_size_buffers);
		break;

	case ARG_MULTICAST_IFACE:
		g_value_set_string(value, element->iface);
		break;

	case ARG_MULTICAST_GROUP:
		g_value_set_string(value, element->group);
		break;

	case ARG_PORT:
		g_value_set_int(value, element->port);
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
	GstGDSFramexmitSink *element = GDS_FRAMEXMITSINK(object);

	g_free(element->iface);
	element->iface = NULL;
	g_free(element->group);
	element->group = NULL;

	G_OBJECT_CLASS(gds_framexmitsink_parent_class)->finalize(object);
}


/*
 * class_init()
 */


static void gds_framexmitsink_class_init(GstGDSFramexmitSinkClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GstBaseSinkClass *gstbasesink_class = GST_BASE_SINK_CLASS(klass);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	gstbasesink_class->start = GST_DEBUG_FUNCPTR(start);
	gstbasesink_class->stop = GST_DEBUG_FUNCPTR(stop);
	gstbasesink_class->render = GST_DEBUG_FUNCPTR(render);

	gst_element_class_set_details_simple(
		element_class,
		"GDS Framexmit Transmitter Sink",
		"Sink",
		"GDS framexmit broadcast transmitter sink element",
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
		ARG_MAX_SIZE_BUFFERS,
		g_param_spec_int(
			"max-size-buffers",
			"Buffers",
			"The maximum number of buffers to hold for retransmission requests.",
			0, G_MAXINT, DEFAULT_MAX_SIZE_BUFFERS,
			(GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_MULTICAST_IFACE,
		g_param_spec_string(
			"multicast-iface",
			"IP address",
			"The interface on which to join the multicast group.  If the interface is omitted, the default interface will be used.  In general, one can use the subnet address as the interface address.  The transmitter will go through the list of all local interfaces and determine the closest match.",
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
			"The address of the multicast group to transmit to.  If the address is not a multicast address, the transmitter will UDP/IP broadcast instead of multicast.",
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
			"The local port from which to transmit the packets from (0 = allocate).",
			0, 65535, DEFAULT_PORT,
			(GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)
		)
	);
}


/*
 * init()
 */


static void gds_framexmitsink_init(GstGDSFramexmitSink *element)
{
	/*
	 * internal data
	 */

	element->iface = NULL;
	element->group = NULL;
	element->frameSend = NULL;
}

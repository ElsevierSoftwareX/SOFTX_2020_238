/*
 * Copyright (C) 2009 Kipp Cannon <kipp.cannon@ligo.org>, Chad Hanna
 * <chad.hanna@ligo.caltech.edu>
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
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasesink.h>
#include <gstlal.h>
#include <gstlal_triggerxmlwriter.h>
#include <lal/LIGOMetadataTables.h>
#include <lal/LIGOLwXML.h>
#include <lal/LALStdlib.h>



/*
 * ============================================================================
 *
 *                             Trigger XML Writer
 *
 * ============================================================================
 */


static gboolean xmlwriter_start(GstBaseSink *sink)
{
	GSTLALTriggerXMLWriter *element = GSTLAL_TRIGGERXMLWRITER(sink);
	LALStatus status;
	memset(&status, 0, sizeof(status));

	g_assert(element->location != NULL);

	element->xml = XLALOpenLIGOLwXMLFile(element->location);
	LALBeginLIGOLwXMLTable(&status, element->xml, sngl_inspiral_table);

	return TRUE;
}


static gboolean xmlwriter_stop(GstBaseSink *sink)
{
	GSTLALTriggerXMLWriter *element = GSTLAL_TRIGGERXMLWRITER(sink);
	LALStatus status;
	memset(&status, 0, sizeof(status));

	if(element->xml) {
		LALEndLIGOLwXMLTable(&status, element->xml);
		XLALCloseLIGOLwXMLFile(element->xml);
		element->xml = NULL;
	}

	return TRUE;
}


static GstFlowReturn xmlwriter_render(GstBaseSink *sink, GstBuffer *buffer)
{
	GSTLALTriggerXMLWriter *element = GSTLAL_TRIGGERXMLWRITER(sink);
	LALStatus status;
	MetadataTable table = {
		.snglInspiralTable = (SnglInspiralTable *) GST_BUFFER_DATA(buffer)
	};
	memset(&status, 0, sizeof(status));

	if(!GST_BUFFER_FLAG_IS_SET(buffer, GST_BUFFER_FLAG_GAP))
		LALWriteLIGOLwXMLTable(&status, element->xml, table, sngl_inspiral_table);

	return GST_FLOW_OK;
}


enum xmlwriter_property {
	PROP_LOCATION = 1
};


static void xmlwriter_set_property(GObject * object, enum xmlwriter_property id, const GValue * value, GParamSpec * pspec)
{
	GSTLALTriggerXMLWriter *element = GSTLAL_TRIGGERXMLWRITER(object);

	GST_OBJECT_LOCK(element);
	switch(id) {
	case PROP_LOCATION:
		g_free(element->location);
		element->location = g_value_dup_string(value);
		break;
	}
	GST_OBJECT_UNLOCK(element);
}


static void xmlwriter_get_property(GObject * object, enum xmlwriter_property id, GValue * value, GParamSpec * pspec)
{
	GSTLALTriggerXMLWriter *element = GSTLAL_TRIGGERXMLWRITER(object);

	GST_OBJECT_LOCK(element);
	switch(id) {
	case PROP_LOCATION:
		g_value_set_string(value,element->location);
		break;
	}
	GST_OBJECT_UNLOCK(element);
}


static GstBaseSink *xmlwriter_parent_class = NULL;


static void xmlwriter_base_init(gpointer g_class)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(g_class);

	gst_element_class_set_details_simple(
		element_class,
		"Trigger XML Writer",
		"Sink/File",
		"Writes LAL's SnglInspiralTable C structures to an XML file",
		"Kipp Cannon <kipp.cannon@ligo.org>, Chad Hanna <channa@ligo.caltech.edu>"
	);

	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"sink",
			GST_PAD_SINK,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
				"application/x-lal-snglinspiral"
			)
		)
	);
}


static void xmlwriter_class_init(gpointer klass, gpointer class_data)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstBaseSinkClass *gstbasesink_class = GST_BASE_SINK_CLASS(klass);

	xmlwriter_parent_class = g_type_class_ref(GST_TYPE_BASE_SINK);
	gobject_class->set_property = GST_DEBUG_FUNCPTR(xmlwriter_set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(xmlwriter_get_property);
	gstbasesink_class->start = GST_DEBUG_FUNCPTR(xmlwriter_start);
	gstbasesink_class->stop = GST_DEBUG_FUNCPTR(xmlwriter_stop);
	gstbasesink_class->render = GST_DEBUG_FUNCPTR(xmlwriter_render);

	g_object_class_install_property(
		gobject_class,
		PROP_LOCATION,
		g_param_spec_string(
			"location",
			"filename",
			"Path to output file",
			NULL,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);
}


static void xmlwriter_instance_init(GTypeInstance *object, gpointer klass)
{
	GSTLALTriggerXMLWriter *element = GSTLAL_TRIGGERXMLWRITER(object);

	/*
	 * Internal data
	 */

	element->location = NULL;
	element->xml = NULL;
}


GType gstlal_triggerxmlwriter_get_type(void)
{
	static GType type = 0;

	if(!type) {
		static const GTypeInfo info = {
			.class_size = sizeof(GSTLALTriggerXMLWriterClass),
			.class_init = xmlwriter_class_init,
			.base_init = xmlwriter_base_init,
			.instance_size = sizeof(GSTLALTriggerXMLWriter),
			.instance_init = xmlwriter_instance_init,
		};
		type = g_type_register_static(GST_TYPE_BASE_SINK, "lal_triggerxmlwriter", &info, 0);
	}

	return type;
}

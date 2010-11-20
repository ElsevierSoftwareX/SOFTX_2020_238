/*
 * Copyright (C) 2010 Leo Singer
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


#include <unary_base.h>

#include <math.h>

GType unary_log_get_type(void);

#define UNARY_LOG_TYPE \
	(unary_log_get_type())
#define UNARY_LOG(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), UNARY_LOG_TYPE, UnaryLog))


typedef struct {
	GstAudioFilter audiofilter;
	double base;
	double invlogbase;
} UnaryLog;



static GstFlowReturn transform_ip(GstBaseTransform *trans, GstBuffer *buf)
{
	UnaryLog *element = UNARY_LOG(trans);
	GstAudioFilter *audiofilter = GST_AUDIO_FILTER(trans);
	GstBufferFormat format = audiofilter->format.format;

	gpointer data = GST_BUFFER_DATA(buf);
	gpointer data_end = GST_BUFFER_DATA(buf) + GST_BUFFER_SIZE(buf);

	const double n = element->invlogbase;

	if (format >= GST_FLOAT64_LE)
	{
		double *ptr, *end = data_end;
		for (ptr = data; ptr < end; ptr++)
			*ptr = log(*ptr) * n;
	} else if (format >= GST_FLOAT32_LE) {
		float *ptr, *end = data_end;
		for (ptr = data; ptr < end; ptr++)
			*ptr = logf(*ptr) * n;
	} else {
		g_assert_not_reached();
	}

	return GST_FLOW_OK;
}


/*
 * ============================================================================
 *
 *                                Type Support
 *
 * ============================================================================
 */


enum property {
	PROP_BASE = 1,
};


static void set_property(GObject * object, enum property id, const GValue * value, GParamSpec * pspec)
{
	UnaryLog *element = UNARY_LOG(object);
	
	GST_OBJECT_LOCK(element);
	
	switch (id) {
		case PROP_BASE:
			element->base = g_value_get_double(value);
			element->invlogbase = 1. / log(element->base);
			break;
	}
	
	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject * object, enum property id, GValue * value, GParamSpec * pspec)
{
	UnaryLog *element = UNARY_LOG(object);
	
	GST_OBJECT_LOCK(element);
	
	switch (id) {
		case PROP_BASE:
			g_value_set_double(value, element->base);
			break;
	}
	
	GST_OBJECT_UNLOCK(element);
}


static void base_init(gpointer class)
{
	gst_element_class_set_details_simple(
		GST_ELEMENT_CLASS(class),
		"Natural logarithm",
		"Filter/Audio",
		"Calculate natural logarithm, y = log_k x",
		"Leo Singer <leo.singer@ligo.org>"
	);
}


static void class_init(gpointer class, gpointer class_data)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(class);
	GstBaseTransformClass *basetransform_class = GST_BASE_TRANSFORM_CLASS(class);

	basetransform_class->transform_ip = GST_DEBUG_FUNCPTR(transform_ip);

	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);

	g_object_class_install_property(
		gobject_class,
		PROP_BASE,
		g_param_spec_double(
			"base",
			"Base",
			"Base of logarithm",
			G_MINDOUBLE, G_MAXDOUBLE, 10.,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
}


GType unary_log_get_type(void)
{
	static GType type = 0;

	if(!type) {
		static const GTypeInfo info = {
			.class_size = sizeof(UnaryBaseClass),
			.base_init = base_init,
			.class_init = class_init,
			.instance_size = sizeof(UnaryLog),
		};
		type = g_type_register_static(UNARY_BASE_TYPE, "UnaryLog", &info, 0);
	}

	return type;
}

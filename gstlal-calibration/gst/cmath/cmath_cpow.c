/*
 * Copyright (C) 2010 Leo Singer
 * Copyright (C) 2016 Aaron Viets
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */


#include <cmath_base.h>

#define TYPE_CMATH_CPOW \
	(cmath_cpow_get_type())
#define CMATH_CPOW(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj),TYPE_CMATH_CPOW,CMathCPow))
#define CMATH_CPOW_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass),TYPE_CMATH_CPOW,CMathCPowClass))
#define IS_PLUGIN_TEMPLATE(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj),TYPE_CMATH_CPOW))
#define IS_PLUGIN_TEMPLATE_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass),TYPE_CMATH_CPOW))

typedef struct _CMathCPow CMathCPow;
typedef struct _CMathCPowClass CMathCPowClass;

GType
cmath_cpow_get_type(void);

struct _CMathCPow
{
	CMathBase cmath_base;
	double exponent;
};

struct _CMathCPowClass 
{
	CMathBaseClass parent_class;
};


/*
 * ============================================================================
 *
 *			GstBaseTransform vmethod Implementations
 *
 * ============================================================================
 */

/* An in-place transform really does the same thing as the chain function */

static GstFlowReturn transform_ip(GstBaseTransform *trans, GstBuffer *buf)
{
	CMathCPow* element = CMATH_CPOW(trans);
	int bits = element -> cmath_base.bits;
	int is_complex = element -> cmath_base.is_complex;

	/*
	 * Debugging
	 *
	 * GstObject* element_gstobj = GST_OBJECT(trans);
	 * int channels = element -> cmath_base.channels;
	 * int rate = element -> cmath_base.rate;
	 * g_print("[%s]: passing GstBuffer: ", element_gstobj->name);
	 * g_print("%d channels, ", channels);
	 * g_print("%d bits, ", bits);
	 * g_print("rate: %d, ", rate);
	 */

	GstMapInfo info;
	if(!gst_buffer_map(buf, &info, GST_MAP_READWRITE)) {
		GST_ERROR_OBJECT(trans, "gst_buffer_map failed\n");
	}
	gpointer data = info.data;
	gpointer data_end = data + info.size;

	const double n = element -> exponent;

	if(is_complex == 1) {

		if(bits == 128) {
			double complex *ptr, *end = data_end;
			for(ptr = data; ptr < end; ptr++) {
				*ptr = cpow(*ptr, n);
			}
		} else if(bits == 64) {
			float complex *ptr, *end = data_end;
			for(ptr = data; ptr < end; ptr++) {
				*ptr = cpowf(*ptr, n);
			}
		} else {
			g_assert_not_reached();
		}
	} else if(is_complex == 0) {

		if(bits == 64) {
			double *ptr, *end = data_end;
			for(ptr = data; ptr < end; ptr++) {
				*ptr = pow(*ptr, n);
			}
		} else if(bits == 32) {
			float *ptr, *end = data_end;
			for(ptr = data; ptr < end; ptr++) {
				*ptr = powf(*ptr, n);
			}
		} else {
			g_assert_not_reached();
		}
	} else {
		g_assert_not_reached();
	}
	gst_buffer_unmap(buf, &info);
	return GST_FLOW_OK;
}


/*
 * ============================================================================
 *
 *				Type Support
 *
 * ============================================================================
 */

/* Set the exponent */
enum property
{
	PROP_EXPONENT = 1,
};

static void
set_property(GObject * object, enum property id, const GValue * value,
	GParamSpec * pspec)
{
	CMathCPow *element = CMATH_CPOW(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case PROP_EXPONENT:
		element->exponent = g_value_get_double(value);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}

static void
get_property(GObject * object, enum property id, GValue * value,
	GParamSpec * pspec)
{
	CMathCPow *element = CMATH_CPOW(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case PROP_EXPONENT:
		g_value_set_double(value, element->exponent);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}

/* Initialize the plugin's class */
static void
cmath_cpow_class_init(gpointer klass, gpointer klass_data)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstBaseTransformClass *basetransform_class = GST_BASE_TRANSFORM_CLASS(klass);

	gst_element_class_set_details_simple(GST_ELEMENT_CLASS(klass),
		"Raise input to a power",
		"Filter/Audio",
		"Calculate input raised to the power n, y = x^n",
		"Aaron Viets <aaron.viets@ligo.org>, Leo Singer <leo.singer@ligo.org>");

	basetransform_class -> transform_ip = GST_DEBUG_FUNCPTR(transform_ip);
	basetransform_class -> set_caps = GST_DEBUG_FUNCPTR(set_caps);

	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);

	g_object_class_install_property(gobject_class,
		PROP_EXPONENT,
		g_param_spec_double("exponent",
			"Exponent",
			"Exponent",
			-G_MAXDOUBLE, G_MAXDOUBLE, 2.,
			 G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT));
}

GType
cmath_cpow_get_type(void)
{
	static GType type = 0;

	if(!type) {
		static const GTypeInfo info = {
			.class_size = sizeof(CMathBaseClass),
			.class_init = cmath_cpow_class_init,
			.instance_size = sizeof(CMathCPow),
		};
		type = g_type_register_static(CMATH_BASE_TYPE, "CMathCPow", &info, 0);
	}

	return type;
}

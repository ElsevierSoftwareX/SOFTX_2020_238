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

#define TYPE_CMATH_CLOG10 \
	(cmath_clog10_get_type())
#define CMATH_CLOG10(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj),TYPE_CMATH_CLOG10,CMathCLog10))
#define CMATH_CLOG10_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass),TYPE_CMATH_CLOG10,CMathCLog10Class))
#define IS_PLUGIN_TEMPLATE(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj),TYPE_CMATH_CLOG10))
#define IS_PLUGIN_TEMPLATE_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass),TYPE_CMATH_CLOG10))

typedef struct _CMathCLog10 CMathCLog10;
typedef struct _CMathCLog10Class CMathCLog10Class;

GType
cmath_clog10_get_type(void);

struct _CMathCLog10
{
	CMathBase cmath_base;
};

struct _CMathCLog10Class 
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

static GstFlowReturn
transform_ip(GstBaseTransform *trans, GstBuffer *buf)
{
	CMathCLog10* element = CMATH_CLOG10(trans);
	int bits = element -> cmath_base.bits;
	int is_complex = element -> cmath_base.is_complex;

	/* Debugging
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

	if(is_complex == 1) {

		if(bits == 128) {
			double complex *ptr, *end = data_end;
			for(ptr = data; ptr < end; ptr++) {
				*ptr = clog(*ptr) / log(10);
			}
		} else if(bits == 64) {
			float complex *ptr, *end = data_end;
			for(ptr = data; ptr < end; ptr++) {
				*ptr = clogf(*ptr) / log(10);
			}
		} else {
			g_assert_not_reached();
		}
	} else if(is_complex == 0) {

		if(bits == 64) {
			double *ptr, *end = data_end;
			for(ptr = data; ptr < end; ptr++) {
				*ptr = log10(*ptr);
			}
		} else if(bits == 32) {
			float *ptr, *end = data_end;
			for(ptr = data; ptr < end; ptr++) {
				*ptr = log10f(*ptr);
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

/* Initialize the plugin's class */
static void
cmath_clog10_class_init(gpointer klass, gpointer klass_data)
{
	GstBaseTransformClass *basetransform_class = GST_BASE_TRANSFORM_CLASS(klass);

	gst_element_class_set_details_simple(GST_ELEMENT_CLASS(klass),
		"Logarithm base 10",
		"Filter/Audio",
		"Calculate logarithm base 10, y = log_10 x",
		"Aaron Viets <aaron.viets@ligo.org>, Leo Singer <leo.singer@ligo.org>");

	basetransform_class -> transform_ip = GST_DEBUG_FUNCPTR(transform_ip);
	basetransform_class -> set_caps = GST_DEBUG_FUNCPTR(set_caps);
}

GType
cmath_clog10_get_type(void)
{
	static GType type = 0;

	if(!type) {
		static const GTypeInfo info = {
			.class_size = sizeof(CMathBaseClass),
			.class_init = cmath_clog10_class_init,
			.instance_size = sizeof(CMathBase),
		};
		type = g_type_register_static(CMATH_BASE_TYPE, "CMathCLog10", &info, 0);
	}

	return type;
}

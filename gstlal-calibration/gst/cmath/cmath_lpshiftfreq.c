/*
 * Copyright (C) 2018 Aaron Viets
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

#define TYPE_CMATH_LPSHIFTFREQ \
	(cmath_lpshiftfreq_get_type())
#define CMATH_LPSHIFTFREQ(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj),TYPE_CMATH_LPSHIFTFREQ,CMathLPShiftFreq))
#define CMATH_LPSHIFTFREQ_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass),TYPE_CMATH_LPSHIFTFREQ,CMathLPShiftFreqClass))
#define IS_PLUGIN_TEMPLATE(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj),TYPE_CMATH_LPSHIFTFREQ))
#define IS_PLUGIN_TEMPLATE_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass),TYPE_CMATH_LPSHIFTFREQ))

typedef struct _CMathLPShiftFreq CMathLPShiftFreq;
typedef struct _CMathLPShiftFreqClass CMathLPShiftFreqClass;

GType
cmath_lpshiftfreq_get_type(void);

struct _CMathLPShiftFreq
{
	CMathBase cmath_base;
	double frequency_ratio;
};

struct _CMathLPShiftFreqClass 
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
	CMathLPShiftFreq* element = CMATH_LPSHIFTFREQ(trans);
	int bits = element -> cmath_base.bits;
	int is_complex = element -> cmath_base.is_complex;

	GstMapInfo info;
	if(!gst_buffer_map(buf, &info, GST_MAP_READWRITE)) {
		GST_ERROR_OBJECT(trans, "gst_buffer_map failed\n");
	}
	gpointer data = info.data;
	gpointer data_end = data + info.size;

	const double n = element->frequency_ratio;

	if(is_complex == 1) {

		if(bits == 128) {
			double complex *ptr, *end = data_end;
			for(ptr = data; ptr < end; ptr++) {
				*ptr = cabs(*ptr) * cexp(I * carg(*ptr) * n);
			}
		} else if(bits == 64) {
			float complex *ptr, *end = data_end;
			for(ptr = data; ptr < end; ptr++) {
				*ptr = cabsf(*ptr) * cexpf(I * cargf(*ptr) * (float) n);
			}
		} else {
			g_assert_not_reached();
		}
	} else if(is_complex == 0) {

		/* Really, this is not the purpose of this element, but we'll just multiply by the frequency ratio */
		if(bits == 64) {
			double *ptr, *end = data_end;
			for(ptr = data; ptr < end; ptr++) {
				*ptr = *ptr * n;
			}
		} else if(bits == 32) {
			float *ptr, *end = data_end;
			for(ptr = data; ptr < end; ptr++) {
				*ptr = *ptr * (float) n;
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

/* Set the frequency ratio */
enum property
{
	PROP_FREQUENCY_RATIO = 1,
};

static void
set_property(GObject * object, enum property id, const GValue * value,
	GParamSpec * pspec)
{
	CMathLPShiftFreq *element = CMATH_LPSHIFTFREQ(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case PROP_FREQUENCY_RATIO:
		element->frequency_ratio = g_value_get_double(value);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}

static void
get_property(GObject * object, enum property id, GValue * value,
	GParamSpec * pspec)
{
	CMathLPShiftFreq *element = CMATH_LPSHIFTFREQ(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case PROP_FREQUENCY_RATIO:
		g_value_set_double(value, element->frequency_ratio);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}

/* Initialize the plugin's class */
static void
cmath_lpshiftfreq_class_init(gpointer klass, gpointer klass_data)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstBaseTransformClass *basetransform_class = GST_BASE_TRANSFORM_CLASS(klass);

	gst_element_class_set_details_simple(GST_ELEMENT_CLASS(klass),
		"Multiply phase of a complex number",
		"Filter/Audio",
		"This element reads in a complex stream and multiplies the phase by a\n\t\t\t   "
		"constant without affecting the magnitude. This can be useful when a\n\t\t\t   "
		"function whose phase is proportional to frequency has a known value at a\n\t\t\t   "
		"particular frequency, and the user wishes to evaluate it at a different\n\t\t\t   "
		"frequency.",
		"Aaron Viets <aaron.viets@ligo.org>");

	basetransform_class -> transform_ip = GST_DEBUG_FUNCPTR(transform_ip);
	basetransform_class -> set_caps = GST_DEBUG_FUNCPTR(set_caps);

	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);

	g_object_class_install_property(gobject_class,
		PROP_FREQUENCY_RATIO,
		g_param_spec_double("frequency-ratio",
			"Frequency Ratio",
			"The factor by which you want to multiply the phase. In the above example,\n\t\t\t"
			"this is equal to the ratio of the frequencies.",
			-G_MAXDOUBLE, G_MAXDOUBLE, 2.,
			 G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT));
}

GType
cmath_lpshiftfreq_get_type(void)
{
	static GType type = 0;

	if(!type) {
		static const GTypeInfo info = {
			.class_size = sizeof(CMathBaseClass),
			.class_init = cmath_lpshiftfreq_class_init,
			.instance_size = sizeof(CMathLPShiftFreq),
		};
		type = g_type_register_static(CMATH_BASE_TYPE, "CMathLPShiftFreq", &info, 0);
	}

	return type;
}

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


#include <gst/audio/audio.h>
#include <cmath_base.h>

#define TYPE_CMATH_CIMAG \
	(cmath_cimag_get_type())
#define CMATH_CIMAG(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj),TYPE_CMATH_CIMAG,CMathCImag))
#define CMATH_CIMAG_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass),TYPE_CMATH_CIMAG,CMathCImagClass))
#define IS_PLUGIN_TEMPLATE(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj),TYPE_CMATH_CIMAG))
#define IS_PLUGIN_TEMPLATE_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass),TYPE_CMATH_CIMAG))

typedef struct _CMathCImag CMathCImag;
typedef struct _CMathCImagClass CMathCImagClass;

GType
cmath_cimag_get_type(void);

struct _CMathCImag
{
	CMathBase cmath_base;
};

struct _CMathCImagClass 
{
	CMathBaseClass parent_class;
};


/*
 * ============================================================================
 *
 *		     GstBaseTransform vmethod Implementations
 *
 * ============================================================================
 */

static GstCaps *transform_caps(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, GstCaps *filter)
{
	guint n;

	caps = gst_caps_normalize(gst_caps_copy(caps));
	GstCaps *othercaps = gst_caps_new_empty();

	switch(direction) {
	case GST_PAD_SRC:
		/* There are two possible sink pad formats for each src pad format, so the sink pad caps has twice as many structures */
		for(n = 0; n < gst_caps_get_size(caps); n++) {
			gst_caps_append(othercaps, gst_caps_copy_nth(caps, n));
			gst_caps_append(othercaps, gst_caps_copy_nth(caps, n));

			GstStructure *str = gst_caps_get_structure(othercaps, 2 * n);
			const gchar *format = gst_structure_get_string(str, "format");

			if(!format) {
				GST_DEBUG_OBJECT(trans, "unrecognized caps %" GST_PTR_FORMAT, othercaps);
				goto error;
			} else if(!strcmp(format, GST_AUDIO_NE(F32)) || !strcmp(format, GST_AUDIO_NE(Z64)))
				gst_structure_set(str, "format", G_TYPE_STRING, GST_AUDIO_NE(Z64), NULL);
			else if(!strcmp(format, GST_AUDIO_NE(F64)) || !strcmp(format, GST_AUDIO_NE(Z128)))
				gst_structure_set(str, "format", G_TYPE_STRING, GST_AUDIO_NE(Z128), NULL);
			else {
				GST_DEBUG_OBJECT(trans, "unrecognized format %s in %" GST_PTR_FORMAT, format, othercaps);
				goto error;
			}
		}
		break;

	case GST_PAD_SINK:
		for(n = 0; n < gst_caps_get_size(caps); n++) {
			gst_caps_append(othercaps, gst_caps_copy_nth(caps, n));
			GstStructure *str = gst_caps_get_structure(othercaps, n);
			const gchar *format = gst_structure_get_string(str, "format");

			if(!format) {
				GST_DEBUG_OBJECT(trans, "unrecognized caps %" GST_PTR_FORMAT, othercaps);
				goto error;
			} else if(!strcmp(format, GST_AUDIO_NE(F32)) || !strcmp(format, GST_AUDIO_NE(Z64)))
				gst_structure_set(str, "format", G_TYPE_STRING, GST_AUDIO_NE(F32), NULL);
			else if(!strcmp(format, GST_AUDIO_NE(F64)) || !strcmp(format, GST_AUDIO_NE(Z128)))
				gst_structure_set(str, "format", G_TYPE_STRING, GST_AUDIO_NE(F64), NULL);
			else {
				GST_DEBUG_OBJECT(trans, "unrecognized format %s in %" GST_PTR_FORMAT, format, othercaps);
				goto error;
			}
		}
		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		goto error;
	}

	if(filter) {
		GstCaps *intersection = gst_caps_intersect(othercaps, filter);
		gst_caps_unref(othercaps);
		othercaps = intersection;
	}
	gst_caps_unref(caps);
	return gst_caps_simplify(othercaps);

error:
	gst_caps_unref(caps);
	gst_caps_unref(othercaps);
	return GST_CAPS_NONE;
}

static gboolean transform_size(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, gsize size, GstCaps *othercaps, gsize *othersize)
{
	CMathCImag* element = CMATH_CIMAG(trans);

	int is_complex = element->cmath_base.is_complex;

	switch(direction) {
	case GST_PAD_SRC:
		/*
		 * We have the size of the output buffer, and we set the size of the input buffer,
		 * which depends on the sink pad caps.
		 */

		if(is_complex) {
			/* input buffer is twice as large as output buffer since it is complex */
			*othersize = 2 * size;
		} else {
			/* input and output buffers are the same size */
			*othersize = size;
		}
		break;

	case GST_PAD_SINK:
		/*
		 * We have the size of the input buffer, and we set the size of the output buffer,
		 * which depends on the sink pad caps..
		 */

		if(is_complex) {
			/* output buffer is half as large as input buffer since it is complex */
			*othersize = size / 2;
		} else {
			/* input and output buffers are the same size */
			*othersize = size;
		}
		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		return FALSE;
	}

	return TRUE;
	
}

/* A transform really does the same thing as the chain function */

static GstFlowReturn transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf)
{
	CMathCImag* element = CMATH_CIMAG(trans);
	int bits = element -> cmath_base.bits;
	int is_complex = element -> cmath_base.is_complex;

	GstMapInfo inmap, outmap;
	gst_buffer_map(inbuf, &inmap, GST_MAP_READ);
	gst_buffer_map(outbuf, &outmap, GST_MAP_WRITE);

	gpointer indata = inmap.data;
	gpointer outdata = outmap.data;
	gpointer indata_end = indata + inmap.size;

	if(is_complex == 1) {

		if(bits == 128) {
			double complex *ptr, *end = indata_end;
			double *outptr = outdata;
			for(ptr = indata; ptr < end; ptr++, outptr++)
				*outptr = cimag(*ptr);
		} else if(bits == 64) {
			float complex *ptr, *end = indata_end;
			float *outptr = outdata;
			for(ptr = indata; ptr < end; ptr++, outptr++)
				*outptr = cimagf(*ptr);
		} else {
			g_assert_not_reached();
		}
	} else if(is_complex == 0) {

		if(bits == 64) {
			double *ptr, *end = indata_end;
			double *outptr = outdata;
			for(ptr = indata; ptr < end; ptr++, outptr++)
				*outptr = 0;
		} else if(bits == 32) {
			float *ptr, *end = indata_end;
			float *outptr = outdata;
			for(ptr = indata; ptr < end; ptr++, outptr++)
				*outptr = 0;
		} else {
			g_assert_not_reached();
		}
	} else {
		g_assert_not_reached();
	}

	gst_buffer_unmap(inbuf, &inmap);
	gst_buffer_unmap(outbuf, &outmap);

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
cmath_cimag_class_init(gpointer klass, gpointer klass_data)
{
	GstBaseTransformClass *basetransform_class = GST_BASE_TRANSFORM_CLASS(klass);

	gst_element_class_set_details_simple(GST_ELEMENT_CLASS(klass),
		"Imaginary",
		"Filter/Audio",
		"Calculate imanginary part of a complex number, y = Im(x)", 
		"Aaron Viets <aaron.viets@ligo.org>");

	basetransform_class -> transform = GST_DEBUG_FUNCPTR(transform);
	basetransform_class -> set_caps = GST_DEBUG_FUNCPTR(set_caps);
	basetransform_class -> transform_caps = GST_DEBUG_FUNCPTR(transform_caps);
	basetransform_class -> transform_size = GST_DEBUG_FUNCPTR(transform_size);
}

GType
cmath_cimag_get_type(void)
{
	static GType type = 0;

	if(!type) {
		static const GTypeInfo info = {
			.class_size = sizeof(CMathBaseClass),
			.class_init = cmath_cimag_class_init,
			.instance_size = sizeof(CMathBase),
		};
		type = g_type_register_static(CMATH_BASE_TYPE, "CMathCImag", &info, 0);
	}

	return type;
}

/*
 * Copyright (C) 2019  Aaron Viets <aaron.viets@ligo.org>
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation; either version 2 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */


/*
 * =============================================================================
 *
 *			       Preamble
 *
 * =============================================================================
 */


/*
 * stuff from C
 */


#include <string.h>
#include <math.h>


/*
 * stuff from gobject/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/audio/audio.h>
#include <gst/base/gstbasetransform.h>
#include <gstlal/gstlal.h>
#include <gstlal/gstlal_audio_info.h>
#include <gstlal_typecast.h>


/*
 * ============================================================================
 *
 *			       Utilities
 *
 * ============================================================================
 */


#define DEFINE_TYPECAST_BUFFER(INTYPE, INCOMPLEX, OUTTYPE, OUTCOMPLEX) \
static void typecast_buffer_ ## INCOMPLEX ## INTYPE ## _to_ ## OUTCOMPLEX ## OUTTYPE(const INCOMPLEX INTYPE *src, guint64 src_size, OUTCOMPLEX OUTTYPE *dst) { \
 \
	guint64 i; \
	for(i = 0; i < src_size; i++) \
		dst[i] = (OUTCOMPLEX OUTTYPE) src[i]; \
 \
	return; \
}


DEFINE_TYPECAST_BUFFER(gint8, , gint8, );
DEFINE_TYPECAST_BUFFER(gint8, , gint16, );
DEFINE_TYPECAST_BUFFER(gint8, , gint32, );
DEFINE_TYPECAST_BUFFER(gint8, , guint8, );
DEFINE_TYPECAST_BUFFER(gint8, , guint16, );
DEFINE_TYPECAST_BUFFER(gint8, , guint32, );
DEFINE_TYPECAST_BUFFER(gint8, , float, );
DEFINE_TYPECAST_BUFFER(gint8, , double, );
DEFINE_TYPECAST_BUFFER(gint8, , float, complex);
DEFINE_TYPECAST_BUFFER(gint8, , double, complex);
DEFINE_TYPECAST_BUFFER(gint16, , gint8, );
DEFINE_TYPECAST_BUFFER(gint16, , gint16, );
DEFINE_TYPECAST_BUFFER(gint16, , gint32, );
DEFINE_TYPECAST_BUFFER(gint16, , guint8, );
DEFINE_TYPECAST_BUFFER(gint16, , guint16, );
DEFINE_TYPECAST_BUFFER(gint16, , guint32, );
DEFINE_TYPECAST_BUFFER(gint16, , float, );
DEFINE_TYPECAST_BUFFER(gint16, , double, );
DEFINE_TYPECAST_BUFFER(gint16, , float, complex);
DEFINE_TYPECAST_BUFFER(gint16, , double, complex);
DEFINE_TYPECAST_BUFFER(gint32, , gint8, );
DEFINE_TYPECAST_BUFFER(gint32, , gint16, );
DEFINE_TYPECAST_BUFFER(gint32, , gint32, );
DEFINE_TYPECAST_BUFFER(gint32, , guint8, );
DEFINE_TYPECAST_BUFFER(gint32, , guint16, );
DEFINE_TYPECAST_BUFFER(gint32, , guint32, );
DEFINE_TYPECAST_BUFFER(gint32, , float, );
DEFINE_TYPECAST_BUFFER(gint32, , double, );
DEFINE_TYPECAST_BUFFER(gint32, , float, complex);
DEFINE_TYPECAST_BUFFER(gint32, , double, complex);
DEFINE_TYPECAST_BUFFER(guint8, , gint8, );
DEFINE_TYPECAST_BUFFER(guint8, , gint16, );
DEFINE_TYPECAST_BUFFER(guint8, , gint32, );
DEFINE_TYPECAST_BUFFER(guint8, , guint8, );
DEFINE_TYPECAST_BUFFER(guint8, , guint16, );
DEFINE_TYPECAST_BUFFER(guint8, , guint32, );
DEFINE_TYPECAST_BUFFER(guint8, , float, );
DEFINE_TYPECAST_BUFFER(guint8, , double, );
DEFINE_TYPECAST_BUFFER(guint8, , float, complex);
DEFINE_TYPECAST_BUFFER(guint8, , double, complex);
DEFINE_TYPECAST_BUFFER(guint16, , gint8, );
DEFINE_TYPECAST_BUFFER(guint16, , gint16, );
DEFINE_TYPECAST_BUFFER(guint16, , gint32, );
DEFINE_TYPECAST_BUFFER(guint16, , guint8, );
DEFINE_TYPECAST_BUFFER(guint16, , guint16, );
DEFINE_TYPECAST_BUFFER(guint16, , guint32, );
DEFINE_TYPECAST_BUFFER(guint16, , float, );
DEFINE_TYPECAST_BUFFER(guint16, , double, );
DEFINE_TYPECAST_BUFFER(guint16, , float, complex);
DEFINE_TYPECAST_BUFFER(guint16, , double, complex);
DEFINE_TYPECAST_BUFFER(guint32, , gint8, );
DEFINE_TYPECAST_BUFFER(guint32, , gint16, );
DEFINE_TYPECAST_BUFFER(guint32, , gint32, );
DEFINE_TYPECAST_BUFFER(guint32, , guint8, );
DEFINE_TYPECAST_BUFFER(guint32, , guint16, );
DEFINE_TYPECAST_BUFFER(guint32, , guint32, );
DEFINE_TYPECAST_BUFFER(guint32, , float, );
DEFINE_TYPECAST_BUFFER(guint32, , double, );
DEFINE_TYPECAST_BUFFER(guint32, , float, complex);
DEFINE_TYPECAST_BUFFER(guint32, , double, complex);
DEFINE_TYPECAST_BUFFER(float, , gint8, );
DEFINE_TYPECAST_BUFFER(float, , gint16, );
DEFINE_TYPECAST_BUFFER(float, , gint32, );
DEFINE_TYPECAST_BUFFER(float, , guint8, );
DEFINE_TYPECAST_BUFFER(float, , guint16, );
DEFINE_TYPECAST_BUFFER(float, , guint32, );
DEFINE_TYPECAST_BUFFER(float, , float, );
DEFINE_TYPECAST_BUFFER(float, , double, );
DEFINE_TYPECAST_BUFFER(float, , float, complex);
DEFINE_TYPECAST_BUFFER(float, , double, complex);
DEFINE_TYPECAST_BUFFER(double, , gint8, );
DEFINE_TYPECAST_BUFFER(double, , gint16, );
DEFINE_TYPECAST_BUFFER(double, , gint32, );
DEFINE_TYPECAST_BUFFER(double, , guint8, );
DEFINE_TYPECAST_BUFFER(double, , guint16, );
DEFINE_TYPECAST_BUFFER(double, , guint32, );
DEFINE_TYPECAST_BUFFER(double, , float, );
DEFINE_TYPECAST_BUFFER(double, , double, );
DEFINE_TYPECAST_BUFFER(double, , float, complex);
DEFINE_TYPECAST_BUFFER(double, , double, complex);
DEFINE_TYPECAST_BUFFER(float, complex, gint8, );
DEFINE_TYPECAST_BUFFER(float, complex, gint16, );
DEFINE_TYPECAST_BUFFER(float, complex, gint32, );
DEFINE_TYPECAST_BUFFER(float, complex, guint8, );
DEFINE_TYPECAST_BUFFER(float, complex, guint16, );
DEFINE_TYPECAST_BUFFER(float, complex, guint32, );
DEFINE_TYPECAST_BUFFER(float, complex, float, );
DEFINE_TYPECAST_BUFFER(float, complex, double, );
DEFINE_TYPECAST_BUFFER(float, complex, float, complex);
DEFINE_TYPECAST_BUFFER(float, complex, double, complex);
DEFINE_TYPECAST_BUFFER(double, complex, gint8, );
DEFINE_TYPECAST_BUFFER(double, complex, gint16, );
DEFINE_TYPECAST_BUFFER(double, complex, gint32, );
DEFINE_TYPECAST_BUFFER(double, complex, guint8, );
DEFINE_TYPECAST_BUFFER(double, complex, guint16, );
DEFINE_TYPECAST_BUFFER(double, complex, guint32, );
DEFINE_TYPECAST_BUFFER(double, complex, float, );
DEFINE_TYPECAST_BUFFER(double, complex, double, );
DEFINE_TYPECAST_BUFFER(double, complex, float, complex);
DEFINE_TYPECAST_BUFFER(double, complex, double, complex);


/*
 * set the metadata on an output buffer
 */


static void set_metadata(GSTLALTypeCast *element, GstBuffer *buf, guint64 outsamples, gboolean gap) {

	GST_BUFFER_OFFSET(buf) = element->next_out_offset;
	element->next_out_offset += outsamples;
	GST_BUFFER_OFFSET_END(buf) = element->next_out_offset;
	GST_BUFFER_PTS(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET(buf) - element->offset0, GST_SECOND, element->rate);
	GST_BUFFER_DURATION(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET_END(buf) - element->offset0, GST_SECOND, element->rate) - GST_BUFFER_PTS(buf);
	if(G_UNLIKELY(element->need_discont)) {
		GST_BUFFER_FLAG_SET(buf, GST_BUFFER_FLAG_DISCONT);
		element->need_discont = FALSE;
	}
	if(gap)
		GST_BUFFER_FLAG_SET(buf, GST_BUFFER_FLAG_GAP);
	else
		GST_BUFFER_FLAG_UNSET(buf, GST_BUFFER_FLAG_GAP);
}


/*
 * ============================================================================
 *
 *			   GStreamer Boiler Plate
 *
 * ============================================================================
 */


static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SINK_NAME,
	GST_PAD_SINK,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		"audio/x-raw, " \
		"rate = " GST_AUDIO_RATE_RANGE ", " \
		"channels = (int) [1, MAX], " \
		"format = (string) {" GST_AUDIO_NE(S8) ", " GST_AUDIO_NE(S16) ", " GST_AUDIO_NE(S32) ", " GST_AUDIO_NE(U8) ", " GST_AUDIO_NE(U16) ", " GST_AUDIO_NE(U32) ", " GST_AUDIO_NE(F32) ", " GST_AUDIO_NE(F64) ", " GST_AUDIO_NE(Z64) ", " GST_AUDIO_NE(Z128) "}, " \
		"layout = (string) interleaved, " \
		"channel-mask = (bitmask) 0"
	)
);


static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SRC_NAME,
	GST_PAD_SRC,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		"audio/x-raw, " \
		"rate = " GST_AUDIO_RATE_RANGE ", " \
		"channels = (int) [1, MAX], " \
		"format = (string) {" GST_AUDIO_NE(S8) ", " GST_AUDIO_NE(S16) ", " GST_AUDIO_NE(S32) ", " GST_AUDIO_NE(U8) ", " GST_AUDIO_NE(U16) ", " GST_AUDIO_NE(U32) ", " GST_AUDIO_NE(F32) ", " GST_AUDIO_NE(F64) ", " GST_AUDIO_NE(Z64) ", " GST_AUDIO_NE(Z128) "}, " \
		"layout = (string) interleaved, " \
		"channel-mask = (bitmask) 0"
	)
);


G_DEFINE_TYPE(
	GSTLALTypeCast,
	gstlal_typecast,
	GST_TYPE_BASE_TRANSFORM
);


/*
 * ============================================================================
 *
 *		     GstBaseTransform Method Overrides
 *
 * ============================================================================
 */


/*
 * get_unit_size()
 */


static gboolean get_unit_size(GstBaseTransform *trans, GstCaps *caps, gsize *size) {

	GstAudioInfo info;
	gboolean success = TRUE;

	success &= gstlal_audio_info_from_caps(&info, caps);

	if(success) {
		*size = GST_AUDIO_INFO_BPF(&info);
	} else
		GST_WARNING_OBJECT(trans, "unable to parse caps %" GST_PTR_FORMAT, caps);

	return success;
}


/*
 * transform_caps()
 */


static GstCaps *transform_caps(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, GstCaps *filter) {

	guint n;
	static char *formats[] = {GST_AUDIO_NE(S8), GST_AUDIO_NE(S16), GST_AUDIO_NE(S32), GST_AUDIO_NE(U8), GST_AUDIO_NE(U16), GST_AUDIO_NE(U32), GST_AUDIO_NE(F32), GST_AUDIO_NE(F64), GST_AUDIO_NE(Z64), GST_AUDIO_NE(Z128)};

	caps = gst_caps_normalize(gst_caps_copy(caps));
	GstCaps *othercaps = gst_caps_new_empty();

	switch(direction) {
	case GST_PAD_SRC:
	case GST_PAD_SINK:
		/*
		 * There are 10 possible formats in othercaps for each format in caps, so the
		 * othercaps has 10 times as many structures. Otherwise the sink pad and source
		 * pad caps are the same.
		 */
		for(n = 0; n < 10 * gst_caps_get_size(caps); n++) {
			gst_caps_append(othercaps, gst_caps_copy_nth(caps, n / 10));
			GstStructure *str = gst_caps_get_structure(othercaps, n);
			gst_structure_set(str, "format", G_TYPE_STRING, formats[n % 10], NULL);
		}
		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		gst_caps_unref(caps);
		return GST_CAPS_NONE;
	}

	othercaps = gst_caps_simplify(othercaps);

	if(filter) {
		GstCaps *intersection = gst_caps_intersect(othercaps, filter);
		gst_caps_unref(othercaps);
		othercaps = intersection;
	}

	gst_caps_unref(caps);
	return othercaps;
}


/*
 * set_caps()
 */


static gboolean set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps) {

	guint i, test;
	GSTLALTypeCast *element = GSTLAL_TYPECAST(trans);
	gsize unit_size_in, unit_size_out;
	const gchar *format_in;
	const gchar *format_out;
	static char *formats[] = {GST_AUDIO_NE(S8), GST_AUDIO_NE(S16), GST_AUDIO_NE(S32), GST_AUDIO_NE(U8), GST_AUDIO_NE(U16), GST_AUDIO_NE(U32), GST_AUDIO_NE(F32), GST_AUDIO_NE(F64), GST_AUDIO_NE(Z64), GST_AUDIO_NE(Z128)};
	gboolean complexes[] = {FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE};
	gboolean floats[] = {FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE};
	gboolean signs[] = {TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE};

	/* Find unit sizes of input and output */
	if(!get_unit_size(trans, incaps, &unit_size_in)) {
		GST_DEBUG_OBJECT(element, "failed to get unit size from input caps %" GST_PTR_FORMAT, incaps);
		return FALSE;
	}
	if(!get_unit_size(trans, outcaps, &unit_size_out)) {
		GST_DEBUG_OBJECT(element, "failed to get unit size from output caps %" GST_PTR_FORMAT, outcaps);
		return FALSE;
	}

	element->unit_size_in = unit_size_in;
	element->unit_size_out = unit_size_out;

	/* Get the caps structure from the incaps */
	GstStructure *str_in = gst_caps_get_structure(incaps, 0);
	g_assert(str_in);

	/* Number of channels, which is the same for input and output */
	if(!gst_structure_get_int(str_in, "channels", &element->channels)) {
		GST_DEBUG_OBJECT(element, "failed to get channels from input caps %" GST_PTR_FORMAT, incaps);
		return FALSE;
	}

	/* Rate, which is the same for input and output */
	if(!gst_structure_get_int(str_in, "rate", &element->rate)) {
		GST_DEBUG_OBJECT(element, "failed to get rate from input caps %" GST_PTR_FORMAT, incaps);
		return FALSE;
	}

	/* Check the incaps to see if it contains S32, etc. */
	if(gst_structure_has_field(str_in, "format")) {
		format_in = gst_structure_get_string(str_in, "format");
	} else {
		GST_ERROR_OBJECT(element, "No incaps format! Cannot set element caps.\n");
		return FALSE;
	}
	test = 0;
	for(i = 0; i < sizeof(formats) / sizeof(*formats); i++) {
		if(!strcmp(format_in, formats[i])) {
			element->complex_in = complexes[i];
			element->float_in = floats[i];
			element->sign_in = signs[i];
			test++;
		}			
	}
	if(test != 1) {
		GST_ERROR_OBJECT(element, "element caps not properly set");
		return FALSE;
	}

	/* Check the outcaps to see if it contains S32, etc. */
	GstStructure *str_out = gst_caps_get_structure(outcaps, 0);
	g_assert(str_out);

	if(gst_structure_has_field(str_out, "format")) {
		format_out = gst_structure_get_string(str_out, "format");
	} else {
		GST_ERROR_OBJECT(element, "No outcaps format! Cannot set element caps.\n");
		return FALSE;
	}
	test = 0;
	for(i = 0; i < sizeof(formats) / sizeof(*formats); i++) {
		if(!strcmp(format_out, formats[i])) {
			element->complex_out = complexes[i];
			element->float_out = floats[i];
			element->sign_out = signs[i];
			test++;
		}
	}
	if(test != 1) {
		GST_ERROR_OBJECT(element, "element->sign_out and/or element->float_out not properly set");
		return FALSE;
	}

	return TRUE;
}


/*
 * transform_size()
 */


static gboolean transform_size(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, gsize size, GstCaps *othercaps, gsize *othersize) {

	GSTLALTypeCast *element = GSTLAL_TYPECAST(trans);

	switch(direction) {
	case GST_PAD_SRC:
		/*
		 * We know the size of the output buffer, and we will compute the size of the
		 * input buffer.  The only difference is the unit size.
		 */
		if(G_UNLIKELY(size % element->unit_size_out)) {
			GST_DEBUG_OBJECT(element, "buffer size %" G_GSIZE_FORMAT " is not a multiple of unit_size %" G_GSIZE_FORMAT, size, (gsize) element->unit_size_out);
			return FALSE;
		}

		*othersize = (gsize) (size * element->unit_size_in / element->unit_size_out);

		break;

	case GST_PAD_SINK:
		/*
		 * We know the size of the input buffer, and we will compute the size of the
		 * output buffer.  The only difference is the unit size.
		 */
		if(G_UNLIKELY(size % element->unit_size_in)) {
			GST_DEBUG_OBJECT(element, "buffer size %" G_GSIZE_FORMAT " is not a multiple of unit_size %" G_GSIZE_FORMAT, size, (gsize) element->unit_size_in);
			return FALSE;
		}

		*othersize = (gsize) (size * element->unit_size_out / element->unit_size_in);

		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		return FALSE;
	}

	return TRUE;
}


/*
 * transform()
 */


static GstFlowReturn transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf) {

	GSTLALTypeCast *element = GSTLAL_TYPECAST(trans);
	GstMapInfo inmap, outmap;
	GstFlowReturn result = GST_FLOW_OK;

	/*
	 * check for discontinuity
	 */

	if(G_UNLIKELY(GST_BUFFER_IS_DISCONT(inbuf) || GST_BUFFER_OFFSET(inbuf) != element->next_in_offset || !GST_CLOCK_TIME_IS_VALID(element->t0))) {
		GST_DEBUG_OBJECT(element, "pushing discontinuous buffer at input timestamp %lu", (long unsigned) GST_TIME_AS_SECONDS(GST_BUFFER_PTS(inbuf)));
		element->t0 = GST_BUFFER_PTS(inbuf);
		element->offset0 = GST_BUFFER_OFFSET(inbuf);
		element->next_out_offset = GST_BUFFER_OFFSET(inbuf);
		element->need_discont = TRUE;
	}

	element->next_in_offset = GST_BUFFER_OFFSET_END(inbuf);

	/*
	 * process buffer
	 */

	gst_buffer_map(inbuf, &inmap, GST_MAP_READ);
	gst_buffer_map(outbuf, &outmap, GST_MAP_WRITE);

	g_assert_cmpuint(inmap.size % element->unit_size_in, ==, 0);
	g_assert_cmpuint(outmap.size % element->unit_size_out, ==, 0);

	guint64 src_size = inmap.size * element->channels / element->unit_size_in;

	if(element->complex_in) {
		switch(element->unit_size_in) {
		case 8:
			if(element->complex_out) {
				switch(element->unit_size_out) {
				case 8:
					typecast_buffer_complexfloat_to_complexfloat((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 16:
					typecast_buffer_complexfloat_to_complexdouble((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				default:
					g_assert_not_reached();
				}
			} else if(element->float_out) {
				switch(element->unit_size_out) {
				case 4:
					typecast_buffer_complexfloat_to_float((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 8:
					typecast_buffer_complexfloat_to_double((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				default:
					g_assert_not_reached();
				}
			} else if(element->sign_out) {
				switch(element->unit_size_out) {
				case 1:
					typecast_buffer_complexfloat_to_gint8((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 2:
					typecast_buffer_complexfloat_to_gint16((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 4:
					typecast_buffer_complexfloat_to_gint32((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				default:
					g_assert_not_reached();
				}
			} else {
				switch(element->unit_size_out) {
				case 1:
					typecast_buffer_complexfloat_to_guint8((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 2:
					typecast_buffer_complexfloat_to_guint16((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 4:
					typecast_buffer_complexfloat_to_guint32((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				default:
					g_assert_not_reached();
				}
			}
			break;
		case 16:
			if(element->complex_out) {
				switch(element->unit_size_out) {
				case 8:
					typecast_buffer_complexdouble_to_complexfloat((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 16:
					typecast_buffer_complexdouble_to_complexdouble((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				default:
					g_assert_not_reached();
				}
			} else if(element->float_out) {
				switch(element->unit_size_out) {
				case 4:
					typecast_buffer_complexdouble_to_float((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 8:
					typecast_buffer_complexdouble_to_double((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				default:
					g_assert_not_reached();
				}
			} else if(element->sign_out) {
				switch(element->unit_size_out) {
				case 1:
					typecast_buffer_complexdouble_to_gint8((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 2:
					typecast_buffer_complexdouble_to_gint16((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 4:
					typecast_buffer_complexdouble_to_gint32((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				default:
					g_assert_not_reached();
				}
			} else {
				switch(element->unit_size_out) {
				case 1:
					typecast_buffer_complexdouble_to_guint8((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 2:
					typecast_buffer_complexdouble_to_guint16((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 4:
					typecast_buffer_complexdouble_to_guint32((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				default:
					g_assert_not_reached();
				}
			}
			break;
		default:
			g_assert_not_reached();
		}
	} else if(element->float_in) {
		switch(element->unit_size_in) {
		case 4:
			if(element->complex_out) {
				switch(element->unit_size_out) {
				case 8:
					typecast_buffer_float_to_complexfloat((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 16:
					typecast_buffer_float_to_complexdouble((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				default:
					g_assert_not_reached();
				}
			} else if(element->float_out) {
				switch(element->unit_size_out) {
				case 4:
					typecast_buffer_float_to_float((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 8:
					typecast_buffer_float_to_double((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				default:
					g_assert_not_reached();
				}
			} else if(element->sign_out) {
				switch(element->unit_size_out) {
				case 1:
					typecast_buffer_float_to_gint8((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 2:
					typecast_buffer_float_to_gint16((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 4:
					typecast_buffer_float_to_gint32((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				default:
					g_assert_not_reached();
				}
			} else {
				switch(element->unit_size_out) {
				case 1:
					typecast_buffer_float_to_guint8((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 2:
					typecast_buffer_float_to_guint16((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 4:
					typecast_buffer_float_to_guint32((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				default:
					g_assert_not_reached();
				}
			}
			break;
		case 8:
			if(element->complex_out) {
				switch(element->unit_size_out) {
				case 8:
					typecast_buffer_double_to_complexfloat((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 16:
					typecast_buffer_double_to_complexdouble((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				default:
					g_assert_not_reached();
				}
			} else if(element->float_out) {
				switch(element->unit_size_out) {
				case 4:
					typecast_buffer_double_to_float((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 8:
					typecast_buffer_double_to_double((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				default:
					g_assert_not_reached();
				}
			} else if(element->sign_out) {
				switch(element->unit_size_out) {
				case 1:
					typecast_buffer_double_to_gint8((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 2:
					typecast_buffer_double_to_gint16((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 4:
					typecast_buffer_double_to_gint32((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				default:
					g_assert_not_reached();
				}
			} else {
				switch(element->unit_size_out) {
				case 1:
					typecast_buffer_double_to_guint8((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 2:
					typecast_buffer_double_to_guint16((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 4:
					typecast_buffer_double_to_guint32((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				default:
					g_assert_not_reached();
				}
			}
			break;
		default:
			g_assert_not_reached();
		}
	} else if(element->sign_in) {
		switch(element->unit_size_in) {
		case 1:
			if(element->complex_out) {
				switch(element->unit_size_out) {
				case 8:
					typecast_buffer_gint8_to_complexfloat((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 16:
					typecast_buffer_gint8_to_complexdouble((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				default:
					g_assert_not_reached();
				}
			} else if(element->float_out) {
				switch(element->unit_size_out) {
				case 4:
					typecast_buffer_gint8_to_float((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 8:
					typecast_buffer_gint8_to_double((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				default:
					g_assert_not_reached();
				}
			} else if(element->sign_out) {
				switch(element->unit_size_out) {
				case 1:
					typecast_buffer_gint8_to_gint8((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 2:
					typecast_buffer_gint8_to_gint16((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 4:
					typecast_buffer_gint8_to_gint32((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				default:
					g_assert_not_reached();
				}
			} else {
				switch(element->unit_size_out) {
				case 1:
					typecast_buffer_gint8_to_guint8((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 2:
					typecast_buffer_gint8_to_guint16((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 4:
					typecast_buffer_gint8_to_guint32((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				default:
					g_assert_not_reached();
				}
			}
			break;
		case 2:
			if(element->complex_out) {
				switch(element->unit_size_out) {
				case 8:
					typecast_buffer_gint16_to_complexfloat((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 16:
					typecast_buffer_gint16_to_complexdouble((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				default:
					g_assert_not_reached();
				}
			} else if(element->float_out) {
				switch(element->unit_size_out) {
				case 4:
					typecast_buffer_gint16_to_float((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 8:
					typecast_buffer_gint16_to_double((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				default:
					g_assert_not_reached();
				}
			} else if(element->sign_out) {
				switch(element->unit_size_out) {
				case 1:
					typecast_buffer_gint16_to_gint8((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 2:
					typecast_buffer_gint16_to_gint16((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 4:
					typecast_buffer_gint16_to_gint32((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				default:
					g_assert_not_reached();
				}
			} else {
				switch(element->unit_size_out) {
				case 1:
					typecast_buffer_gint16_to_guint8((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 2:
					typecast_buffer_gint16_to_guint16((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 4:
					typecast_buffer_gint16_to_guint32((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				default:
					g_assert_not_reached();
				}
			}
			break;
		case 4:
			if(element->complex_out) {
				switch(element->unit_size_out) {
				case 8:
					typecast_buffer_gint32_to_complexfloat((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 16:
					typecast_buffer_gint32_to_complexdouble((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				default:
					g_assert_not_reached();
				}
			} else if(element->float_out) {
				switch(element->unit_size_out) {
				case 4:
					typecast_buffer_gint32_to_float((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 8:
					typecast_buffer_gint32_to_double((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				default:
					g_assert_not_reached();
				}
			} else if(element->sign_out) {
				switch(element->unit_size_out) {
				case 1:
					typecast_buffer_gint32_to_gint8((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 2:
					typecast_buffer_gint32_to_gint16((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 4:
					typecast_buffer_gint32_to_gint32((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				default:
					g_assert_not_reached();
				}
			} else {
				switch(element->unit_size_out) {
				case 1:
					typecast_buffer_gint32_to_guint8((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 2:
					typecast_buffer_gint32_to_guint16((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 4:
					typecast_buffer_gint32_to_guint32((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				default:
					g_assert_not_reached();
				}
			}
			break;
		default:
			g_assert_not_reached();
		}
	} else {
		switch(element->unit_size_in) {
		case 1:
			if(element->complex_out) {
				switch(element->unit_size_out) {
				case 8:
					typecast_buffer_guint8_to_complexfloat((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 16:
					typecast_buffer_guint8_to_complexdouble((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				default:
					g_assert_not_reached();
				}
			} else if(element->float_out) {
				switch(element->unit_size_out) {
				case 4:
					typecast_buffer_guint8_to_float((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 8:
					typecast_buffer_guint8_to_double((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				default:
					g_assert_not_reached();
				}
			} else if(element->sign_out) {
				switch(element->unit_size_out) {
				case 1:
					typecast_buffer_guint8_to_gint8((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 2:
					typecast_buffer_guint8_to_gint16((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 4:
					typecast_buffer_guint8_to_gint32((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				default:
					g_assert_not_reached();
				}
			} else {
				switch(element->unit_size_out) {
				case 1:
					typecast_buffer_guint8_to_guint8((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 2:
					typecast_buffer_guint8_to_guint16((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 4:
					typecast_buffer_guint8_to_guint32((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				default:
					g_assert_not_reached();
				}
			}
			break;
		case 2:
			if(element->complex_out) {
				switch(element->unit_size_out) {
				case 8:
					typecast_buffer_guint16_to_complexfloat((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 16:
					typecast_buffer_guint16_to_complexdouble((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				default:
					g_assert_not_reached();
				}
			} else if(element->float_out) {
				switch(element->unit_size_out) {
				case 4:
					typecast_buffer_guint16_to_float((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 8:
					typecast_buffer_guint16_to_double((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				default:
					g_assert_not_reached();
				}
			} else if(element->sign_out) {
				switch(element->unit_size_out) {
				case 1:
					typecast_buffer_guint16_to_gint8((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 2:
					typecast_buffer_guint16_to_gint16((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 4:
					typecast_buffer_guint16_to_gint32((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				default:
					g_assert_not_reached();
				}
			} else {
				switch(element->unit_size_out) {
				case 1:
					typecast_buffer_guint16_to_guint8((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 2:
					typecast_buffer_guint16_to_guint16((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 4:
					typecast_buffer_guint16_to_guint32((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				default:
					g_assert_not_reached();
				}
			}
			break;
		case 4:
			if(element->complex_out) {
				switch(element->unit_size_out) {
				case 8:
					typecast_buffer_guint32_to_complexfloat((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 16:
					typecast_buffer_guint32_to_complexdouble((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				default:
					g_assert_not_reached();
				}
			} else if(element->float_out) {
				switch(element->unit_size_out) {
				case 4:
					typecast_buffer_guint32_to_float((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 8:
					typecast_buffer_guint32_to_double((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				default:
					g_assert_not_reached();
				}
			} else if(element->sign_out) {
				switch(element->unit_size_out) {
				case 1:
					typecast_buffer_guint32_to_gint8((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 2:
					typecast_buffer_guint32_to_gint16((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 4:
					typecast_buffer_guint32_to_gint32((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				default:
					g_assert_not_reached();
				}
			} else {
				switch(element->unit_size_out) {
				case 1:
					typecast_buffer_guint32_to_guint8((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 2:
					typecast_buffer_guint32_to_guint16((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				case 4:
					typecast_buffer_guint32_to_guint32((const void *) inmap.data, src_size, (void *) outmap.data);
					break;
				default:
					g_assert_not_reached();
				}
			}
			break;
		default:
			g_assert_not_reached();
		}
	}

	set_metadata(element, outbuf, outmap.size / element->unit_size_out, GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP));
	gst_buffer_unmap(outbuf, &outmap);
	gst_buffer_unmap(inbuf, &inmap);

	/*
	 * done
	 */

	return result;
}


/*
 * ============================================================================
 *
 *			  GObject Method Overrides
 *
 * ============================================================================
 */


/*
 * class_init()
 */


static void gstlal_typecast_class_init(GSTLALTypeCastClass *klass) {

	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(klass);

	transform_class->transform_caps = GST_DEBUG_FUNCPTR(transform_caps);
	transform_class->transform_size = GST_DEBUG_FUNCPTR(transform_size);
	transform_class->get_unit_size = GST_DEBUG_FUNCPTR(get_unit_size);
	transform_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
	transform_class->transform = GST_DEBUG_FUNCPTR(transform);
	transform_class->passthrough_on_same_caps = TRUE;

	gst_element_class_set_details_simple(element_class,
		"TypeCast",
		"Filter/Audio",
		"Convert the data type of a buffer using a simple type cast",
		"Aaron Viets <aaron.viets@ligo.org>"
	);

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));
}


/*
 * init()
 */


static void gstlal_typecast_init(GSTLALTypeCast *element) {

	element->t0 = GST_CLOCK_TIME_NONE;
	element->offset0 = GST_BUFFER_OFFSET_NONE;
	element->next_in_offset = GST_BUFFER_OFFSET_NONE;
	element->next_out_offset = GST_BUFFER_OFFSET_NONE;
	element->need_discont = TRUE;
	element->unit_size_in = 0;
	element->unit_size_out = 0;
	element->channels = 0;
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(element), TRUE);
}


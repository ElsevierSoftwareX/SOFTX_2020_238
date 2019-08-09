/*
 * Copyright (C) 2019  Aaron Viets <aaron.viets@ligo.org>
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
 *				  Preamble
 *
 * ============================================================================
 */


/*
 * stuff from C
 */


#include <string.h>
#include <math.h>
#include <complex.h>


/*
 *  stuff from gobject/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/audio/audio.h>
#include <gst/base/gstbasetransform.h>


/*
 * our own stuff
 */


#include <gstlal/gstlal_audio_info.h>
#include <gstlal_sensingtdcfs.h>


/*
 * ============================================================================
 *
 *			   GStreamer Boiler Plate
 *
 * ============================================================================
 */


#define INCAPS \
	"audio/x-raw, " \
	"format = (string) {"GST_AUDIO_NE(Z64)", "GST_AUDIO_NE(Z128)"}, " \
	"rate = " GST_AUDIO_RATE_RANGE ", " \
	"channels = (int) [4, 5], " \
	"layout = (string) interleaved, " \
	"channel-mask = (bitmask) 0"

#define OUTCAPS \
	"audio/x-raw, " \
	"format = (string) {"GST_AUDIO_NE(F32)", "GST_AUDIO_NE(F64)"}, " \
	"rate = " GST_AUDIO_RATE_RANGE ", " \
	"channels = (int) [4, 6], " \
	"layout = (string) interleaved, " \
	"channel-mask = (bitmask) 0"


static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SINK_NAME,
	GST_PAD_SINK,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(INCAPS)
);


static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SRC_NAME,
	GST_PAD_SRC,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(OUTCAPS)
);


G_DEFINE_TYPE(
	GSTLALSensingTDCFs,
	gstlal_sensingtdcfs,
	GST_TYPE_BASE_TRANSFORM
);


/*
 * ============================================================================
 *
 *				Utilities
 *
 * ============================================================================
 */


/*
 * set the metadata on an output buffer
 */


static void set_metadata(GSTLALSensingTDCFs *element, GstBuffer *buf, guint64 outsamples, gboolean gap) {

	GST_BUFFER_OFFSET(buf) = element->next_out_offset;
	element->next_out_offset += outsamples;
	GST_BUFFER_OFFSET_END(buf) = element->next_out_offset;
	GST_BUFFER_TIMESTAMP(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET(buf) - element->offset0, GST_SECOND, element->rate);
	GST_BUFFER_DURATION(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET_END(buf) - element->offset0, GST_SECOND, element->rate) - GST_BUFFER_TIMESTAMP(buf);
	GST_BUFFER_FLAG_UNSET(buf, GST_BUFFER_FLAG_GAP);
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
 * Macro functions to solve for kappa_C using three different sensing function models: 0, 1, and 2
 */


#define DEFINE_KAPPA_C_0(DTYPE, F_OR_NOT) \
DTYPE kappa_C_0_ ## DTYPE(DTYPE Gres1_real, DTYPE Gres1_imag, DTYPE Gres2_real, DTYPE Gres2_imag, DTYPE Y1_real, DTYPE Y1_imag, DTYPE Y2_real, DTYPE Y2_imag, DTYPE f1, DTYPE f2) { \
 \
	DTYPE H1, H2, I1, I2, Xi, Zeta, a, b, c, d, e, Delta0, Delta1, p, q, R0; \
	complex DTYPE Q0, S0; \
	H1 = f1 * f1 * (Gres1_real - Y1_real); \
	H2 = f2 * f2 * (Gres2_real - Y2_real); \
	I1 = f1 * f1 * (Gres1_imag - Y1_imag); \
	I2 = f2 * f2 * (Gres2_imag - Y2_imag); \
	Xi = (f2 * f2 * H1 - f1 * f1 * H2) / (f1 * f1 - f2 * f2); \
	Zeta = I1 / f1 - I2 / f2; \
	a = f2 * f2 * Xi * (H2 + Xi) * Zeta * Zeta; \
	b = f2 * (f2 * f2 - f1 * f1) * (H2 + Xi) * Zeta * I2 + pow ## F_OR_NOT(f2, 4) * (H2 + 2 * Xi) * Zeta * Zeta; \
	c = pow ## F_OR_NOT(f2, 3) * (f2 * f2 - f1 * f1) * Zeta * I2 + pow ## F_OR_NOT(f2 * f2 - f1 * f1, 2) * pow ## F_OR_NOT(H2 + Xi, 2) + pow ## F_OR_NOT(f2, 6) * Zeta * Zeta; \
	d = 2 * f2 * f2 * pow ## F_OR_NOT(f2 * f2 - f1 * f1, 2) * (H2 + Xi); \
	e = pow ## F_OR_NOT(f2, 4) * pow ## F_OR_NOT(f2 * f2 - f1 * f1, 2); \
	Delta0 = c * c - 3 * b * d + 12 * a * e; \
	Delta1 = 2 * pow ## F_OR_NOT(c, 3) - 9 * b * c * d + 27 * b * b * e + 27 * a * d * d - 72 * a * c * e; \
	p = (8 * a * c - 3 * b * b) / (8 * a * a); \
	q = (pow ## F_OR_NOT(b, 3) - 4 * a * b * c + 8 * a * a * d) / (8 * pow ## F_OR_NOT(a, 3)); \
	Q0 = cpow ## F_OR_NOT(0.5 * ((complex DTYPE) Delta1 + cpow ## F_OR_NOT((complex DTYPE) (Delta1 * Delta1 - 4 * pow ## F_OR_NOT(Delta0, 3)), 0.5)), 1.0 / 3.0); \
	S0 = 0.5 * cpow ## F_OR_NOT((-2 * p) / 3.0 + (Q0 + Delta0 / Q0) / (3 * a), 0.5); \
	R0 = pow ## F_OR_NOT(b, 3) + 8 * d * a * a - 4 * a * b * c; \
	if(R0 <= 0.0) \
		return creal ## F_OR_NOT(-b / (4 * a) - S0 + 0.5 * cpow ## F_OR_NOT(-4 * S0 * S0 - 2 * p + q / S0, 0.5)); \
	else \
		return creal ## F_OR_NOT(-b / (4 * a) + S0 + 0.5 * cpow ## F_OR_NOT(-4 * S0 * S0 - 2 * p - q / S0, 0.5)); \
}


DEFINE_KAPPA_C_0(float, f);
DEFINE_KAPPA_C_0(double, );


/*
 * Macro functions to solve for f_cc using three different sensing function models: 0, 1, and 2
 */


#define DEFINE_F_CC_0(DTYPE) \
DTYPE f_cc_0_ ## DTYPE(DTYPE Gres1_imag, DTYPE Gres2_imag, DTYPE Y1_imag, DTYPE Y2_imag, DTYPE f1, DTYPE f2, DTYPE kappa_C) { \
 \
	return (f2 * f2 - f1 * f1) / (kappa_C * (f1 * (Gres1_imag - Y1_imag) - f2 * (Gres2_imag - Y2_imag))); \
}


DEFINE_F_CC_0(float);
DEFINE_F_CC_0(double);


/*
 * Macro functions to solve for f_s^2 using three different sensing function models: 0, 1, and 2
 */


#define DEFINE_F_S_SQUARED_0(DTYPE) \
DTYPE f_s_squared_0_ ## DTYPE(DTYPE Gres1_real, DTYPE Gres2_real, DTYPE Y1_real, DTYPE Y2_real, DTYPE f1, DTYPE f2, DTYPE kappa_C) { \
 \
	return kappa_C * (Y2_real - Gres2_real - Y1_real + Gres1_real) / (1.0 / (f2 * f2) - 1.0 / (f1 * f1)); \
}


DEFINE_F_S_SQUARED_0(float);
DEFINE_F_S_SQUARED_0(double);


/*
 * Macro functions to solve for f_s / Q using three different sensing function models: 0, 1, and 2
 */


#define DEFINE_FS_OVER_Q_0(DTYPE) \
DTYPE f_s_over_Q_0_ ## DTYPE(DTYPE Gres1_real, DTYPE Y1_real, DTYPE f1, DTYPE kappa_C, DTYPE f_cc, DTYPE f_s_squared) { \
 \
	return f_cc * (kappa_C * Y1_real - 1.0 - f_s_squared / (f1 * f1) - kappa_C * Gres1_real); \
}


DEFINE_FS_OVER_Q_0(float);
DEFINE_FS_OVER_Q_0(double);


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


static gboolean get_unit_size(GstBaseTransform *trans, GstCaps *caps, gsize *size)
{
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
	caps = gst_caps_normalize(gst_caps_copy(caps));

	switch(direction) {
	case GST_PAD_SRC:
		/* 
		 * We know the caps on the source pad, and we want to put constraints on
		 * the sink pad caps.  The sink pad caps are complex, while the source pad
		 * caps are real.  If there are 4 channels on the source pad, there should
		 * be 4 on the sink pad as well.  If there are 6 on the source pad, there
		 * should be 5 on the sink pad.  There are no other possibilities.
		 */
		for(n = 0; n < gst_caps_get_size(caps); n++) {
			GstStructure *str = gst_caps_get_structure(caps, n);
			const GValue *v = gst_structure_get_value(str, "channels");
			if(GST_VALUE_HOLDS_INT_RANGE(v)) {
				gint channels_out_min, channels_out_max;
				channels_out_min = gst_value_get_int_range_min(v);
				channels_out_max = gst_value_get_int_range_max(v);
				g_assert_cmpint(channels_out_min, <=, 6);
				g_assert_cmpint(channels_out_max, >=, 4);
				if(channels_out_min >= 5)
					/* Then we know there must be 5 on the sink pad */
					gst_structure_set(str, "channels", G_TYPE_INT, 5, NULL);
				else if(channels_out_max <=5)
					/* Then we know there must be 4 on the sink pad */
					gst_structure_set(str, "channels", G_TYPE_INT, 4, NULL);
				else
					gst_structure_set(str, "channels", GST_TYPE_INT_RANGE, 4, 5, NULL);

			} else if(G_VALUE_HOLDS_INT(v)) {
				gint channels_out = g_value_get_int(v);
				if(channels_out == 4)
					gst_structure_set(str, "channels", G_TYPE_INT, 4, NULL);
				else if(channels_out == 6)
					gst_structure_set(str, "channels", G_TYPE_INT, 5, NULL);
				else
					GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid number of channels in caps"));

			} else
				GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid type for channels in caps"));

			const gchar *format = gst_structure_get_string(str, "format");
			if(!format) {
				GST_DEBUG_OBJECT(trans, "unrecognized caps %" GST_PTR_FORMAT, caps);
				goto error;
			} else if(!strcmp(format, GST_AUDIO_NE(F32)))
				gst_structure_set(str, "format", G_TYPE_STRING, GST_AUDIO_NE(Z64), NULL);
			else if(!strcmp(format, GST_AUDIO_NE(F64)))
				gst_structure_set(str, "format", G_TYPE_STRING, GST_AUDIO_NE(Z128), NULL);
			else {
				GST_DEBUG_OBJECT(trans, "unrecognized format %s in %" GST_PTR_FORMAT, format, caps);
				goto error;
			}
		}
		break;

	case GST_PAD_SINK:
		/*
		 * We know the caps on the sink pad, and we want to put constraints on
		 * the source pad caps.  The sink pad caps are complex, while the source pad
		 * caps are real.  If there are 4 channels on the sink pad, there should
		 * be 4 on the source pad as well.  If there are 5 on the sink pad, there
		 * should be 6 on the source pad.  There are no other possibilities.
		 */
		for(n = 0; n < gst_caps_get_size(caps); n++) {
			GstStructure *str = gst_caps_get_structure(caps, n);
			const GValue *v = gst_structure_get_value(str, "channels");
			if(GST_VALUE_HOLDS_INT_RANGE(v)) {
				gint channels_in_min, channels_in_max;
				channels_in_min = gst_value_get_int_range_min(v);
				channels_in_max = gst_value_get_int_range_max(v);
				g_assert_cmpint(channels_in_min, <=, 5);
				g_assert_cmpint(channels_in_max, >=, 4);
				if(channels_in_min == 5)
					/* Then we know there must be 6 channels on the source pad */
					gst_structure_set(str, "channels", G_TYPE_INT, 6, NULL);
				else if(channels_in_max == 4)
					/* Then we know there must be 4 channels on the source pad */
					gst_structure_set(str, "channels", G_TYPE_INT, 4, NULL);
				else
					gst_structure_set(str, "channels", GST_TYPE_INT_RANGE, 4, 6, NULL);

			} else if(G_VALUE_HOLDS_INT(v)) {
				gint channels_in;
				channels_in = g_value_get_int(v);
				if(channels_in == 4)
					gst_structure_set(str, "channels", G_TYPE_INT, 4, NULL);
				else if(channels_in == 5)
					gst_structure_set(str, "channels", G_TYPE_INT, 6, NULL);
				else
					GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid number of channels in caps"));

			} else
				GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid type for channels in caps"));

			const gchar *format = gst_structure_get_string(str, "format");
			if(!format) {
				GST_DEBUG_OBJECT(trans, "unrecognized caps %" GST_PTR_FORMAT, caps);
				goto error;
			} else if(!strcmp(format, GST_AUDIO_NE(Z64)))
				gst_structure_set(str, "format", G_TYPE_STRING, GST_AUDIO_NE(F32), NULL);
			else if(!strcmp(format, GST_AUDIO_NE(Z128)))
				gst_structure_set(str, "format", G_TYPE_STRING, GST_AUDIO_NE(F64), NULL);
			else {
				GST_DEBUG_OBJECT(trans, "unrecognized format %s in %" GST_PTR_FORMAT, format, caps);
				goto error;
			}
		}
		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		goto error;
	default:
		g_assert_not_reached();
	}

	if(filter) {
		GstCaps *intersection = gst_caps_intersect(caps, filter);
		gst_caps_unref(caps);
		caps = intersection;
	}
	return gst_caps_simplify(caps);

error:
	gst_caps_unref(caps);
	return GST_CAPS_NONE;
}


/*
 * set_caps()
 */


static gboolean set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps)
{
	GSTLALSensingTDCFs *element = GSTLAL_SENSINGTDCFS(trans);
	gint rate, channels_in, channels_out;
	gsize unit_size_in, unit_size_out;

	/*
 	 * parse the caps
 	 */

	GstStructure *outstr = gst_caps_get_structure(outcaps, 0);
	GstStructure *instr = gst_caps_get_structure(incaps, 0);
	const gchar *name = gst_structure_get_string(outstr, "format");
	if(!name) {
		GST_DEBUG_OBJECT(element, "unable to parse format from %" GST_PTR_FORMAT, outcaps);
		return FALSE;
	}
	if(!get_unit_size(trans, outcaps, &unit_size_out)) {
		GST_DEBUG_OBJECT(element, "function 'get_unit_size' failed");
		return FALSE;
	}
	if(!get_unit_size(trans, incaps, &unit_size_in)) {
		GST_DEBUG_OBJECT(element, "function 'get_unit_size' failed");
		return FALSE;
	}
	if(!gst_structure_get_int(outstr, "rate", &rate)) {
		GST_DEBUG_OBJECT(element, "unable to parse rate from %" GST_PTR_FORMAT, outcaps);
		return FALSE;
	}
	if(!gst_structure_get_int(outstr, "channels", &channels_out)) {
		GST_DEBUG_OBJECT(element, "unable to parse channels from %" GST_PTR_FORMAT, outcaps);
		return FALSE;
	}
	if(!gst_structure_get_int(instr, "channels", &channels_in)) {
		GST_DEBUG_OBJECT(element, "unable to parse channels from %" GST_PTR_FORMAT, incaps);
		return FALSE;
	}

	/* Requirements for channels */
	if(channels_in == 4) {
		g_assert_cmpint(channels_out, ==, 4);
		if(element->sensing_model != 0 && element->sensing_model != 1) {
			GST_WARNING_OBJECT(element, "When there are 4 input channels, sensing-model must be either 0 or 1.  Resetting sensing-model to 0.");
			element->sensing_model = 0;
		}
	} else if(channels_in == 5) {
		g_assert_cmpint(channels_out, ==, 6);
		if(element->sensing_model != 2) {
			GST_WARNING_OBJECT(element, "When there are 5 input channels, sensing-model must be 2.  Resetting sensing-model to 2.");
			element->sensing_model = 2;
		}
	} else
		g_assert_not_reached();

	/*
 	 * record stream parameters
 	 */

	if(!strcmp(name, GST_AUDIO_NE(F32))) {
		element->data_type = GSTLAL_SENSINGTDCFS_FLOAT;
		g_assert_cmpuint(unit_size_out, ==, 4 * (guint) channels_out);
		g_assert_cmpuint(unit_size_in, ==, 4 * (guint) channels_in);
	} else if(!strcmp(name, GST_AUDIO_NE(F64))) {
		element->data_type = GSTLAL_SENSINGTDCFS_DOUBLE;
		g_assert_cmpuint(unit_size_out, ==, 8 * (guint) channels_out);
		g_assert_cmpuint(unit_size_in, ==, 8 * (guint) channels_in);
	} else
		g_assert_not_reached();

	element->rate = rate;
	element->channels_out = channels_out;
	element->channels_in = channels_in;
	element->unit_size_out = unit_size_out;
	element->unit_size_in = unit_size_in;

	return TRUE;
}


/*
 * transform_size{}
 */


static gboolean transform_size(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, gsize size, GstCaps *othercaps, gsize *othersize) {

	GSTLALSensingTDCFs *element = GSTLAL_SENSINGTDCFS(trans);

	/*
	 * The data types of inputs and outputs are the same, but the number of channels differs.
	 * For N output channels, there are N(N+1) input channels.
	 */

	switch(direction) {
	case GST_PAD_SRC:
		/*
		 * We know the size of the output buffer and want to compute the size of the input buffer.
		 * The size of the output buffer should be a multiple of the unit_size.
		 */

		if(G_UNLIKELY(size % element->unit_size_out)) {
			GST_DEBUG_OBJECT(element, "buffer size %" G_GSIZE_FORMAT " is not a multiple of %" G_GSIZE_FORMAT, size, (gsize) element->unit_size_out);
			return FALSE;
		}

		*othersize = size * element->unit_size_in / element->unit_size_out;

		break;

	case GST_PAD_SINK:
		/*
		 * We know the size of the input buffer and want to compute the size of the output buffer.
		 * The size of the output buffer should be a multiple of unit_size * (N+1).
		 */

		if(G_UNLIKELY(size % element->unit_size_in)) {
			GST_ERROR_OBJECT(element, "buffer size %" G_GSIZE_FORMAT " is not a multiple of %" G_GSIZE_FORMAT, size, (gsize) element->unit_size_in);
			return FALSE;
		}

		*othersize = size * element->unit_size_out / element->unit_size_in;

		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		return FALSE;
	}

	return TRUE;
}


/*
 * start()
 */


static gboolean start(GstBaseTransform *trans) {

	GSTLALSensingTDCFs *element = GSTLAL_SENSINGTDCFS(trans);

	element->t0 = GST_CLOCK_TIME_NONE;
	element->offset0 = GST_BUFFER_OFFSET_NONE;
	element->next_in_offset = GST_BUFFER_OFFSET_NONE;
	element->next_out_offset = GST_BUFFER_OFFSET_NONE;
	element->need_discont = TRUE;

	/* Sanity checks */
	if(element->freq1 == G_MAXDOUBLE)
		GST_WARNING_OBJECT(element, "freq1 was not set. It must be set in order to produce sensible output.");
	if(element->freq2 == G_MAXDOUBLE)
		GST_WARNING_OBJECT(element, "freq2 was not set. It must be set in order to produce sensible output.");
	if(element->sensing_model == 2 && element->freq2 == G_MAXDOUBLE)
		GST_WARNING_OBJECT(element, "freq4 was not set. When using sensing-model is 2, freq4 must be set in order to produce sensible output.");

	return TRUE;
}


/*
 * transform()
 */


static GstFlowReturn transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf) {

	GSTLALSensingTDCFs *element = GSTLAL_SENSINGTDCFS(trans);
	GstMapInfo inmap, outmap;
	GstFlowReturn result = GST_FLOW_OK;

	/*
	 * check for discontinuity
	 */

	if(G_UNLIKELY(GST_BUFFER_IS_DISCONT(inbuf) || GST_BUFFER_OFFSET(inbuf) != element->next_in_offset || !GST_CLOCK_TIME_IS_VALID(element->t0))) {
		GST_DEBUG_OBJECT(element, "pushing discontinuous buffer at input timestamp %lu", (long unsigned) GST_TIME_AS_SECONDS(GST_BUFFER_PTS(inbuf)));
		element->t0 = GST_BUFFER_PTS(inbuf);
		element->offset0 = element->next_out_offset = GST_BUFFER_OFFSET(inbuf);
		element->need_discont = TRUE;
	}
	element->next_in_offset = GST_BUFFER_OFFSET_END(inbuf);

	/*
	 * process buffer
	 */

	if(!GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP)) {

		/*
		 * input is not gap.
		 */

		gst_buffer_map(inbuf, &inmap, GST_MAP_READ);
		gst_buffer_map(outbuf, &outmap, GST_MAP_WRITE);
		if(element->data_type == GSTLAL_SENSINGTDCFS_FLOAT) {
			complex float *indata = (complex float *) inmap.data;
			float *outdata = (float *) outmap.data;
			guint64 i, samples = outmap.size / element->unit_size_out;
			float f1 = (float) element->freq1;
			float f2 = (float) element->freq2;
			float kappa_C, f_cc;
			switch(element->sensing_model) {
			case 0: ;
				float f_s_squared, f_s_over_Q;
				for(i = 0; i < samples; i++) {
					kappa_C = kappa_C_0_float(crealf(indata[4 * i]), cimagf(indata[4 * i]), crealf(indata[4 * i + 1]), cimagf(indata[4 * i + 1]), crealf(indata[4 * i + 2]), cimagf(indata[4 * i + 2]), crealf(indata[4 * i + 3]), cimagf(indata[4 * i + 3]), f1, f2);
					f_cc = f_cc_0_float(cimagf(indata[4 * i]), cimagf(indata[4 * i + 1]), cimagf(indata[4 * i + 2]), cimagf(indata[4 * i + 3]), f1, f2, kappa_C);
					f_s_squared = f_s_squared_0_float(crealf(indata[4 * i]), crealf(indata[4 * i + 1]), crealf(indata[4 * i + 2]), crealf(indata[4 * i + 3]), f1, f2, kappa_C);
					f_s_over_Q = f_s_over_Q_0_float(crealf(indata[4 * i]), crealf(indata[4 * i + 2]), f1, kappa_C, f_cc, f_s_squared);
					outdata[4 * i] = kappa_C;
					outdata[4 * i + 1] = f_cc;
					outdata[4 * i + 2] = f_s_squared;
					outdata[4 * i + 3] = f_s_over_Q;
				}
				break;
			case 1:
				/* Solution not developed */
				break;
			case 2:
				/* Solution not developed */
				break;
			default:
				g_assert_not_reached();
			}
		} else if (element->data_type == GSTLAL_SENSINGTDCFS_DOUBLE) {
			complex double *indata = (complex double *) inmap.data;
			double *outdata = (double *) outmap.data;
			guint64 i, samples = outmap.size / element->unit_size_out;
			double f1 = element->freq1;
			double f2 = element->freq2;
			double kappa_C, f_cc;
			switch(element->sensing_model) {
			case 0: ;
				double f_s_squared, f_s_over_Q;
				for(i = 0; i < samples; i++) {
					kappa_C = kappa_C_0_double(crealf(indata[4 * i]), cimagf(indata[4 * i]), crealf(indata[4 * i + 1]), cimagf(indata[4 * i + 1]), crealf(indata[4 * i + 2]), cimagf(indata[4 * i + 2]), crealf(indata[4 * i + 3]), cimagf(indata[4 * i + 3]), f1, f2);
					f_cc = f_cc_0_double(cimagf(indata[4 * i]), cimagf(indata[4 * i + 1]), cimagf(indata[4 * i + 2]), cimagf(indata[4 * i + 3]), f1, f2, kappa_C);
					f_s_squared = f_s_squared_0_double(crealf(indata[4 * i]), crealf(indata[4 * i + 1]), crealf(indata[4 * i + 2]), crealf(indata[4 * i + 3]), f1, f2, kappa_C);
					f_s_over_Q = f_s_over_Q_0_double(crealf(indata[4 * i]), crealf(indata[4 * i + 2]), f1, kappa_C, f_cc, f_s_squared);
					outdata[4 * i] = kappa_C;
					outdata[4 * i + 1] = f_cc;
					outdata[4 * i + 2] = f_s_squared;
					outdata[4 * i + 3] = f_s_over_Q;
				}
				break;
			case 1:
				/* Solution not developed */
				break;
			case 2:
				/* Solution not developed */
				break;
			default:
				g_assert_not_reached();
			}
		} else {
			g_assert_not_reached();
		}
		set_metadata(element, outbuf, outmap.size / element->unit_size_out, FALSE);
		gst_buffer_unmap(outbuf, &outmap);
		gst_buffer_unmap(inbuf, &inmap);

	} else {

		/*
		 * input is gap.
		 */

		gst_buffer_map(outbuf, &outmap, GST_MAP_WRITE);
		memset(outmap.data, 0, outmap.size);
		set_metadata(element, outbuf, outmap.size / element->unit_size_out, TRUE);
		gst_buffer_unmap(outbuf, &outmap);
	}

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
 * properties
 */


enum property {
	ARG_SENSING_MODEL = 1,
	ARG_FREQ1,
	ARG_FREQ2,
	ARG_FREQ4
};


static void set_property(GObject *object, enum property prop_id, const GValue *value, GParamSpec *pspec) {

	GSTLALSensingTDCFs *element = GSTLAL_SENSINGTDCFS(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_SENSING_MODEL:
		element->sensing_model = g_value_get_int(value);
		break;
	case ARG_FREQ1:
		element->freq1 = g_value_get_double(value);
		break;
	case ARG_FREQ2:
		element->freq2 = g_value_get_double(value);
		break;
	case ARG_FREQ4:
		element->freq4 = g_value_get_double(value);
		break;
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);	
}


static void get_property(GObject *object, enum property prop_id, GValue *value, GParamSpec *pspec) {

	GSTLALSensingTDCFs *element = GSTLAL_SENSINGTDCFS(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_SENSING_MODEL:
		g_value_set_int(value, element->sensing_model);
		break;
	case ARG_FREQ1:
		g_value_set_double(value, element->freq1);
		break;
	case ARG_FREQ2:
		g_value_set_double(value, element->freq2);
		break;
	case ARG_FREQ4:
		g_value_set_double(value, element->freq4);
		break;
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * class_init()
 */


static void gstlal_sensingtdcfs_class_init(GSTLALSensingTDCFsClass *klass)
{
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

	gst_element_class_set_details_simple(
		element_class,
		"Sensing TDCFs",
		"Filter/Audio",
		"Solves for the time-dependent correction factors of the sensing function using\n\t\t\t   "
		"the solution described in LIGO DCC document P1900052, Section 5.2.6.  It takes\n\t\t\t   "
		"the complex inputs Gres^1, Gres^2, Y^1, and Y^2 (and Y^3 if sensing-model is 2),\n\t\t\t   "
		"in that order.  The outputs are kappa_C, f_cc, f_s, and Q, in that order.\n\t\t\t   "
		"Currently, sensing-model=0 is the only sensing model that has been developed.",
		"Aaron Viets <aaron.viets@ligo.org>"
	);

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));

	transform_class->get_unit_size = GST_DEBUG_FUNCPTR(get_unit_size);
	transform_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
	transform_class->transform_caps = GST_DEBUG_FUNCPTR(transform_caps);
	transform_class->transform_size = GST_DEBUG_FUNCPTR(transform_size);
	transform_class->start = GST_DEBUG_FUNCPTR(start);
	transform_class->transform = GST_DEBUG_FUNCPTR(transform);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);

	g_object_class_install_property(
		gobject_class,
		ARG_SENSING_MODEL,
		g_param_spec_int(
			"sensing-model",
			"Sensing Model",
			"Which model of the sensing function to use.  Each model is described below:\n\n\t\t\t"
			"sensing-model=0:\n\t\t\t"
			"    kappa_C\t\t    f^2\n\t\t\t"
			" -------------- * ------------------------- * C_res(f)\n\t\t\t"
			" 1 + i f / f_cc   f^2 + f_s^2 - i f f_s / Q\n\n\t\t\t"
			"Complex inputs: Gres1, Gres2, Y1, Y2.  See P1900052, Sec. 5.2.6.\n\t\t\t"
			"Real outputs: kappa_C, f_cc, f_s^2, f_s / Q\n\n\t\t\t"
			"sensing-model=1:\n\t\t\t"
			"    kappa_C\n\t\t\t"
			" -------------- * (\?\?\?)... not yet modeled\n\t\t\t"
			" 1 + i f / f_cc\n\n\t\t\t"
			"Complex inputs: Gres1, Gres2, Y1, Y2.\n\t\t\t"
			"Real outputs: kappa_C, f_cc, \?\?, \?\?\n\n\t\t\t"
			"sensing-model=2:\n\t\t\t"
			"    kappa_C\t\t    f^2\n\t\t\t"
			" -------------- * ------------------------- * (\?\?\?)... not yet modeled.\n\t\t\t"
			" 1 + i f / f_cc   f^2 + f_s^2 - i f f_s / Q\n\n\t\t\t"
			"Complex inputs: Gres1, Gres2, Y1, Y2, Y3\?\n\t\t\t"
			"Real outputs: kappa_C, f_cc, f_s^2, f_s / Q, \?\?, \?\?",
			0, 2, 0,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_FREQ1,
		g_param_spec_double(
			"freq1",
			"Frequency 1",
			"First Pcal line frequency, typically around 15-20 Hz.",
			-G_MAXDOUBLE, G_MAXDOUBLE, G_MAXDOUBLE,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_FREQ2,
		g_param_spec_double(
			"freq2",
			"Frequency 2",
			"Second Pcal line frequency, typically around 400 Hz.",
			-G_MAXDOUBLE, G_MAXDOUBLE, G_MAXDOUBLE,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_FREQ4,
		g_param_spec_double(
			"freq4",
			"Frequency 4",
			"Fourth Pcal line frequency, typically around 10 Hz.",
			-G_MAXDOUBLE, G_MAXDOUBLE, G_MAXDOUBLE,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
}


/*
 * init()
 */


static void gstlal_sensingtdcfs_init(GSTLALSensingTDCFs *element)
{
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(element), TRUE);
	element->rate = 0;
	element->channels_in = 0;
	element->channels_out = 0;
	element->unit_size_in = 0;
	element->unit_size_out = 0;
}


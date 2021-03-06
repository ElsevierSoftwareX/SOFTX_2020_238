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
 * stuff from GSL
 */


#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_permutation.h>


/*
 * our own stuff
 */


#include <gstlal/gstlal_audio_info.h>
#include <gstlal_matrixsolver.h>


/*
 * ============================================================================
 *
 *			   GStreamer Boiler Plate
 *
 * ============================================================================
 */


#define INCAPS \
	"audio/x-raw, " \
	"format = (string) {"GST_AUDIO_NE(F32)", "GST_AUDIO_NE(F64)", "GST_AUDIO_NE(Z64)", "GST_AUDIO_NE(Z128)"}, " \
	"rate = " GST_AUDIO_RATE_RANGE ", " \
	"channels = (int) [1, MAX], " \
	"layout = (string) interleaved, " \
	"channel-mask = (bitmask) 0"

#define OUTCAPS \
	"audio/x-raw, " \
	"format = (string) {"GST_AUDIO_NE(F32)", "GST_AUDIO_NE(F64)", "GST_AUDIO_NE(Z64)", "GST_AUDIO_NE(Z128)"}, " \
	"rate = " GST_AUDIO_RATE_RANGE ", " \
	"channels = (int) [1, MAX], " \
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
	GSTLALMatrixSolver,
	gstlal_matrixsolver,
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


static void set_metadata(GSTLALMatrixSolver *element, GstBuffer *buf, guint64 outsamples, gboolean gap) {

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
 * Four functions to be used in the macro below
 */


static double make_gsl_input(double x) {
	return x;
}


static gsl_complex make_gsl_inputcomplex(complex double z) {
	return gsl_complex_rect(creal(z), cimag(z));
}


static double get_double_from_gsl_vector(gsl_vector *vec, int i) {
	return gsl_vector_get(vec, i);
}


static complex double get_complexdouble_from_gsl_vector(gsl_vector_complex *vec, int i) {
	gsl_complex gslz = gsl_vector_complex_get(vec, i);
	return GSL_REAL(gslz) + I * GSL_IMAG(gslz);
}


/*
 * Macro for solving systems of linear equations for floats, doubles, and complex float and doubles
 */


#define DEFINE_SOLVE_SYSTEM(COMPLEX, DTYPE, UNDERSCORE) \
static void solve_system_ ## COMPLEX ## DTYPE(const COMPLEX DTYPE *src, COMPLEX DTYPE *dst, guint64 dst_size, int channels_in, int channels_out, gsl_vector ## UNDERSCORE ## COMPLEX *invec, gsl_vector ## UNDERSCORE ## COMPLEX *outvec, gsl_matrix ## UNDERSCORE ## COMPLEX *matrix, gsl_permutation *permutation) { \
 \
	guint64 i; \
	int j, signum; \
 \
	for(i = 0; i < dst_size; i++) { \
		/* Set the elements of the GSL vector invec using the first N channels of input data */ \
		for(j = 0; j < channels_out; j++) \
			gsl_vector_ ## COMPLEX ## UNDERSCORE ## set(invec, j, make_gsl_input ## COMPLEX((COMPLEX double) src[channels_in * i + j])); \
 \
		/* Set the elements of the GSL matrix using the remaining channels of input data */ \
		for(j = 0; j < channels_in - channels_out; j++) \
			gsl_matrix_ ## COMPLEX ## UNDERSCORE ## set(matrix, j / channels_out, j % channels_out, make_gsl_input ## COMPLEX((COMPLEX double) src[channels_in * i + channels_out + j])); \
 \
		/* Now solve [matrix] [outvec] = [invec] for [outvec] using gsl */ \
		gsl_linalg_ ## COMPLEX ## UNDERSCORE ## LU_decomp(matrix, permutation, &signum); \
		if(!isinf(gsl_linalg_ ## COMPLEX ## UNDERSCORE ## LU_lndet(matrix))) { \
			gsl_linalg_ ## COMPLEX ## UNDERSCORE ## LU_solve(matrix, permutation, invec, outvec); \
 \
			/* Put the solutions into the output buffer */ \
			for(j = 0; j < channels_out; j++) \
				dst[i * channels_out + j] = get_ ## COMPLEX ## double_from_gsl_vector(outvec, j); \
		} else { \
			/* Fill output buffer with zeros */ \
			for(j = 0; j < channels_out; j++) \
				dst[i * channels_out + j] = 0.0; \
		} \
	} \
}


DEFINE_SOLVE_SYSTEM(, float, );
DEFINE_SOLVE_SYSTEM(, double, );
DEFINE_SOLVE_SYSTEM(complex, float, _);
DEFINE_SOLVE_SYSTEM(complex, double, _);


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
		 * the sink pad caps.  The sink pad caps are the same as the source pad
		 * caps except that there are N(N+1) input channels for N output channels.
		 */
		for(n = 0; n < gst_caps_get_size(caps); n++) {
			GstStructure *str = gst_caps_get_structure(caps, n);
			const GValue *v = gst_structure_get_value(str, "channels");
			if(GST_VALUE_HOLDS_INT_RANGE(v)) {
				gint channels_in_min, channels_out_min, channels_out_max;
				guint64 channels_in_max;
				channels_out_min = gst_value_get_int_range_min(v);
				channels_out_max = gst_value_get_int_range_max(v);
				channels_in_min = channels_out_min * (channels_out_min + 1);
				channels_in_max = channels_out_max * ((guint64) channels_out_max + 1);
				/* In case channels_in_max is greater than G_MAXINT */
				channels_in_max = channels_in_max < (guint64) G_MAXINT ? channels_in_max : (guint64) G_MAXINT;
				gst_structure_set(str, "channels", GST_TYPE_INT_RANGE, channels_in_min, (gint) channels_in_max, NULL);
			} else if(G_VALUE_HOLDS_INT(v)) {
				gint channels_in, channels_out;
				channels_out = g_value_get_int(v);
				channels_in = channels_out * (channels_out + 1);
				gst_structure_set(str, "channels", G_TYPE_INT, channels_in, NULL);
			} else
				GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid type for channels in caps"));
		}
		break;

	case GST_PAD_SINK:
		/*
		 * We know the caps on the sink pad, and we want to put constraints on
		 * the source pad caps.  The source pad caps are the same as the sink pad
		 * caps except that there are N output channels for N(N+1) input channels.
		 */
		for(n = 0; n < gst_caps_get_size(caps); n++) {
			GstStructure *str = gst_caps_get_structure(caps, n);
			const GValue *v = gst_structure_get_value(str, "channels");
			if(GST_VALUE_HOLDS_INT_RANGE(v)) {
				int channels_in_min, channels_in_max, channels_out_min, channels_out_max;
				channels_in_min = gst_value_get_int_range_min(v);
				channels_in_max = gst_value_get_int_range_max(v);
				channels_out_min = (int) pow((double) channels_in_min, 0.5);
				channels_out_max = (int) pow((double) channels_in_max, 0.5);
				gst_structure_set(str, "channels", GST_TYPE_INT_RANGE, channels_out_min, channels_out_max, NULL);
			} else if(G_VALUE_HOLDS_INT(v)) {
				int channels_in, channels_out;
				channels_in = g_value_get_int(v);
				channels_out = (int) pow((double) channels_in, 0.5);
				gst_structure_set(str, "channels", G_TYPE_INT, channels_out, NULL);
			} else
				GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid type for channels in caps"));
		}
		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		gst_caps_unref(caps);
		return GST_CAPS_NONE;
	default:
		g_assert_not_reached();
	}

	if(filter) {
		GstCaps *intersection = gst_caps_intersect(caps, filter);
		gst_caps_unref(caps);
		caps = intersection;
	}
	return gst_caps_simplify(caps);
}


/*
 * set_caps()
 */


static gboolean set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps)
{
	GSTLALMatrixSolver *element = GSTLAL_MATRIXSOLVER(trans);
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

	/* Require that there are N(N+1) input channels for N output channels */
	if(channels_in != channels_out * (channels_out + 1))
		GST_ERROR_OBJECT(element, "For N output channels, there must be N(N+1) input channels.  input caps = %" GST_PTR_FORMAT " output caps = %" GST_PTR_FORMAT, incaps, outcaps);

	/* Require the input unit size to be larger than the output unit size by a factor N+1 */
	if(unit_size_in != unit_size_out * (channels_out + 1))
		GST_ERROR_OBJECT(element, "Input unit size must be N+1 times as large as output unit size, where N is the number of output channels.  input caps = %" GST_PTR_FORMAT " output caps = %" GST_PTR_FORMAT, incaps, outcaps);

	/*
 	 * record stream parameters
 	 */

	if(!strcmp(name, GST_AUDIO_NE(F32))) {
		element->data_type = GSTLAL_MATRIXSOLVER_F32;
		g_assert_cmpuint(unit_size_out, ==, 4 * (guint) channels_out);
	} else if(!strcmp(name, GST_AUDIO_NE(F64))) {
		element->data_type = GSTLAL_MATRIXSOLVER_F64;
		g_assert_cmpuint(unit_size_out, ==, 8 * (guint) channels_out);
	} else if(!strcmp(name, GST_AUDIO_NE(Z64))) {
		element->data_type = GSTLAL_MATRIXSOLVER_Z64;
		g_assert_cmpuint(unit_size_out, ==, 8 * (guint) channels_out);
	} else if(!strcmp(name, GST_AUDIO_NE(Z128))) {
		element->data_type = GSTLAL_MATRIXSOLVER_Z128;
		g_assert_cmpuint(unit_size_out, ==, 16 * (guint) channels_out);
	} else
		g_assert_not_reached();

	element->rate = rate;
	element->channels_out = channels_out;
	element->channels_in = channels_in;
	element->unit_size_out = unit_size_out;

	/* Allocate memory for gsl vectors, matrices, and permutations */
	if(element->data_type == GSTLAL_MATRIXSOLVER_F32 || element->data_type == GSTLAL_MATRIXSOLVER_F64) {
		if(element->workspace.real.invec) {
			gsl_vector_free(element->workspace.real.invec);
			element->workspace.real.invec = NULL;
		}
		element->workspace.real.invec = gsl_vector_alloc(channels_out);
		if(element->workspace.real.outvec) {
			gsl_vector_free(element->workspace.real.outvec);
			element->workspace.real.outvec = NULL;
		}
		element->workspace.real.outvec = gsl_vector_alloc(channels_out);
		if(element->workspace.real.matrix) {
			gsl_matrix_free(element->workspace.real.matrix);
			element->workspace.real.matrix = NULL;
		}
		element->workspace.real.matrix = gsl_matrix_alloc(channels_out, channels_out);
	} else if (element->data_type == GSTLAL_MATRIXSOLVER_Z64 || element->data_type == GSTLAL_MATRIXSOLVER_Z128) {
		if(element->workspace.cplx.invec) {
			gsl_vector_complex_free(element->workspace.cplx.invec);
			element->workspace.cplx.invec = NULL;
		}
		element->workspace.cplx.invec = gsl_vector_complex_alloc(channels_out);
		if(element->workspace.cplx.outvec) {
			gsl_vector_complex_free(element->workspace.cplx.outvec);
			element->workspace.cplx.outvec = NULL;
		}
		element->workspace.cplx.outvec = gsl_vector_complex_alloc(channels_out);
		if(element->workspace.cplx.matrix) {
			gsl_matrix_complex_free(element->workspace.cplx.matrix);
			element->workspace.cplx.matrix = NULL;
		}
		element->workspace.cplx.matrix = gsl_matrix_complex_alloc(channels_out, channels_out);
	} else
		g_assert_not_reached();

	if(element->permutation) {
		gsl_permutation_free(element->permutation);
		element->permutation = NULL;
	}
	element->permutation = gsl_permutation_alloc(channels_out);

	return TRUE;
}


/*
 * transform_size{}
 */


static gboolean transform_size(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, gsize size, GstCaps *othercaps, gsize *othersize) {

	GSTLALMatrixSolver *element = GSTLAL_MATRIXSOLVER(trans);

	gsize unit_size;

	if(!get_unit_size(trans, caps, &unit_size)) {
		GST_DEBUG_OBJECT(element, "function 'get_unit_size' failed");
		return FALSE;
	}

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

		if(G_UNLIKELY(size % unit_size)) {
			GST_DEBUG_OBJECT(element, "buffer size %" G_GSIZE_FORMAT " is not a multiple of %" G_GSIZE_FORMAT, size, unit_size);
			return FALSE;
		}

		*othersize = size * (element->channels_out + 1);

		break;

	case GST_PAD_SINK:
		/*
		 * We know the size of the input buffer and want to compute the size of the output buffer.
		 * The size of the output buffer should be a multiple of unit_size * (N+1).
		 */

		if(G_UNLIKELY(size % unit_size)) {
			GST_ERROR_OBJECT(element, "buffer size %" G_GSIZE_FORMAT " is not a multiple of %" G_GSIZE_FORMAT, size, unit_size);
			return FALSE;
		}

		g_assert_cmpint(unit_size, ==, element->unit_size_out * (element->channels_out + 1));

		*othersize = size / (element->channels_out + 1);

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

	GSTLALMatrixSolver *element = GSTLAL_MATRIXSOLVER(trans);

	element->t0 = GST_CLOCK_TIME_NONE;
	element->offset0 = GST_BUFFER_OFFSET_NONE;
	element->next_in_offset = GST_BUFFER_OFFSET_NONE;
	element->next_out_offset = GST_BUFFER_OFFSET_NONE;
	element->need_discont = TRUE;

	return TRUE;
}


/*
 * transform()
 */


static GstFlowReturn transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf) {

	GSTLALMatrixSolver *element = GSTLAL_MATRIXSOLVER(trans);
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
		switch(element->data_type) {
		case GSTLAL_MATRIXSOLVER_F32:
			solve_system_float((const void *) inmap.data, (void *) outmap.data, outmap.size / element->unit_size_out, element->channels_in, element->channels_out, element->workspace.real.invec, element->workspace.real.outvec, element->workspace.real.matrix, element->permutation);
			break;
		case GSTLAL_MATRIXSOLVER_F64:
			solve_system_double((const void *) inmap.data, (void *) outmap.data, outmap.size / element->unit_size_out, element->channels_in, element->channels_out, element->workspace.real.invec, element->workspace.real.outvec, element->workspace.real.matrix, element->permutation);
			break;
		case GSTLAL_MATRIXSOLVER_Z64:
			solve_system_complexfloat((const void *) inmap.data, (void *) outmap.data, outmap.size / element->unit_size_out, element->channels_in, element->channels_out, element->workspace.cplx.invec, element->workspace.cplx.outvec, element->workspace.cplx.matrix, element->permutation);
			break;
		case GSTLAL_MATRIXSOLVER_Z128:
			solve_system_complexdouble((const void *) inmap.data, (void *) outmap.data, outmap.size / element->unit_size_out, element->channels_in, element->channels_out, element->workspace.cplx.invec, element->workspace.cplx.outvec, element->workspace.cplx.matrix, element->permutation);
			break;
		default:
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
 * finalize()
 */


static void finalize(GObject *object) {

	GSTLALMatrixSolver *element = GSTLAL_MATRIXSOLVER(object);

	/*
	 * free resources
	 */

	if(element->data_type == GSTLAL_MATRIXSOLVER_F32 || element->data_type == GSTLAL_MATRIXSOLVER_F64) {
		if(element->workspace.real.invec) {
			gsl_vector_free(element->workspace.real.invec);
			element->workspace.real.invec = NULL;
		}
		if(element->workspace.real.outvec) {
			gsl_vector_free(element->workspace.real.outvec);
			element->workspace.real.outvec = NULL;
		}
		if(element->workspace.real.matrix) {
			gsl_matrix_free(element->workspace.real.matrix);
			element->workspace.real.matrix = NULL;
		}
	} else if(element->data_type == GSTLAL_MATRIXSOLVER_Z64 || element->data_type == GSTLAL_MATRIXSOLVER_Z128) {
		if(element->workspace.cplx.invec) {
			gsl_vector_complex_free(element->workspace.cplx.invec);
			element->workspace.cplx.invec = NULL;
		}
		if(element->workspace.cplx.outvec) {
			gsl_vector_complex_free(element->workspace.cplx.outvec);
			element->workspace.cplx.outvec = NULL;
		}
		if(element->workspace.cplx.matrix) {
			gsl_matrix_complex_free(element->workspace.cplx.matrix);
			element->workspace.cplx.matrix = NULL;
		}
	}

	if(element->permutation) {
		gsl_permutation_free(element->permutation);
		element->permutation = NULL;
	}

	/*
	 * chain to parent class' finalize() method
	 */

	G_OBJECT_CLASS(gstlal_matrixsolver_parent_class)->finalize(object);
}


/*
 * class_init()
 */


static void gstlal_matrixsolver_class_init(GSTLALMatrixSolverClass *klass)
{
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

	gst_element_class_set_details_simple(
		element_class,
		"Matrix Solver",
		"Filter/Audio",
		"Solves a system of N linear equations with N unknowns by solving the\n\t\t\t   "
		"matrix equation\n\t\t\t   "
		"--\t\t\t\t      --  --      --     --      --\n\t\t\t   "
		"|  x[N]    x[N+1]   ...  x[2N-1]    |  | y[0]   |     | x[0]   |\n\t\t\t   "
		"|  x[2N]   x[2N+1]  ...  x[3N-1]    |  | y[1]   |     | x[1]   |\n\t\t\t   "
		"|   .\t    .\t .\t       |  |  .     |  =  |  .     |\n\t\t\t   "
		"|   .\t       .      .\t       |  |  .     |     |  .     |\n\t\t\t   "
		"|   .\t\t  .   .\t       |  |  .     |     |  .     |\n\t\t\t   "
		"|  x[N^2]  x[N^2+1] ...  x[N^2+N-1] |  | y[N-1] |     | x[N-1] |\n\t\t\t   "
		"--\t\t\t\t      --  --      --     --      --\n\t\t\t   "
		"for the y[j].  x[i] are the N(N+1) input channels and y[j] are the N\n\t\t\t   "
		"output channels.",
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
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);
}


/*
 * init()
 */


static void gstlal_matrixsolver_init(GSTLALMatrixSolver *element)
{
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(element), TRUE);
	element->rate = 0;
	element->channels_in = 0;
	element->channels_out = 0;
	element->unit_size_out = 0;
	element->permutation = NULL;
}

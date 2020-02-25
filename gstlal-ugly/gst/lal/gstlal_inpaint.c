/*
 * Inpaints.
 *
 * Copyright (C) 2020 Cody Messick
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
 *
 */
		   

/*
 * ========================================================================
 *
 *				  Preamble
 *
 * ========================================================================
 */


/*
 * stuff from the C library
 */


#include <math.h>
#include <stdint.h>
#include <string.h>

/*
 * stuff from gsl
 */


#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>

/*
 * stuff from gstreamer
 */


#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>

/*
 * our own stuff
 */


#include <gstlal/gstlal.h>
#include <gstlal/gstlal_tags.h>
#include <gstlal_inpaint.h>
#include <gstlal/gstaudioadapter.h>

/*
 * stuff from LAL
 */

#include <lal/Date.h>
#include <lal/TimeSeries.h>
#include <lal/FrequencySeries.h>
#include <lal/TimeFreqFFT.h>
#include <lal/Units.h>
#include <lal/Window.h>

// FIXME Figure out why I need this
static const LIGOTimeGPS GPS_ZERO = {0, 0};

#define DEFAULT_FFT_LENGTH_SECONDS 8.0

/*
 * ============================================================================
 *
 *                                Boilerplate
 *
 * ============================================================================
 */


#define GST_CAT_DEFAULT gstlal_inpaint_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


G_DEFINE_TYPE_WITH_CODE(
	GSTLALInpaint,
	gstlal_inpaint,
	GST_TYPE_BASE_TRANSFORM,
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "lal_inpaint", 0, "lal_inpaint element")
);


/*
 * ============================================================================
 *
 *                           Utility Functions
 *
 * ============================================================================
 */

// FIXME This function needs to be called each time a new psd is passed in
static void inv_fft_psd(GSTLALInpaint *inpaint) {
	// Set up the units
	// FIXME Add these to inpaint struct
	LALUnit strain_units = lalStrainUnit;
	LALUnit sample_units, sample_inv_units, psd_inv_units;
	XLALUnitMultiply(&sample_units, &strain_units, &strain_units);
	XLALUnitInvert(&sample_inv_units, &sample_units);
	LALUnit psd_units = gstlal_lalUnitSquaredPerHertz(lalDimensionlessUnit);
	XLALUnitInvert(&psd_inv_units, &psd_units);

	COMPLEX16FrequencySeries *complex_inv_psd = XLALCreateCOMPLEX16FrequencySeries("Complex PSD", &GPS_ZERO, 0.0,  1.0 / inpaint->fft_length_seconds, &psd_inv_units, inpaint->psd->data->length);
	REAL8TimeSeries *inv_cov_series = XLALCreateREAL8TimeSeries("Covariance", &GPS_ZERO, 0.0, 1.0 / (double) inpaint->rate, &sample_inv_units, inpaint->rate * (guint) inpaint->fft_length_seconds);

	guint i;
	for(i=0; i < inpaint->psd->data->length; i++)
		complex_inv_psd->data->data[i] = (COMPLEX16) 1. / inpaint->psd->data->data[i];

	REAL8FFTPlan *revplan = XLALCreateReverseREAL8FFTPlan(inpaint->rate * (guint) inpaint->fft_length_seconds, 1);
	if(XLALREAL8FreqTimeFFT(inv_cov_series, complex_inv_psd, revplan)){
		GST_ERROR_OBJECT(inpaint, "XLALREAL8FreqTimeFFT() failed: %s", XLALErrorString(XLALGetBaseErrno()));
		XLALClearErrno();
	}

	// Multiply by sum of squares of window function. The covariance matrix
	// would need be multiplied by 1 over this factor, so the inverse
	// covariance matrix should be multiplied by the factor
	// NOTE NOTE NOTE This will depend on the method used to construct the
	// psd, the numbers used here assume the FIR whitener was not used
	// FIXME Pass in zero pad length instead of hardcoding it
	REAL8Window *hann_window = XLALCreateHannREAL8Window((guint) inpaint->fft_length_seconds * inpaint->rate - ((guint) inpaint->fft_length_seconds * inpaint->rate) / 2 + 1);
	double window_sum_of_squares = 0;
	for(i = 0; i < hann_window->data->length; i++)
		window_sum_of_squares += pow(hann_window->data->data[i], 2);

	for(i = 0; i < inv_cov_series->data->length; i++)
		inv_cov_series->data->data[i] *= window_sum_of_squares;

	inpaint->inv_cov_series = inv_cov_series;
	XLALDestroyCOMPLEX16FrequencySeries(complex_inv_psd);
	XLALDestroyREAL8Window(hann_window);
}

static gboolean taglist_extract_string(GstObject *object, GstTagList *taglist, const char *tagname, gchar **dest) {
	if(!gst_tag_list_get_string(taglist, tagname, dest)) {
		GST_WARNING_OBJECT(object, "unable to parse \"%s\" from %" GST_PTR_FORMAT, tagname, taglist);
		return FALSE;
	}
        return TRUE;
}

static guint gst_audioadapter_available_samples(GstAudioAdapter *adapter) {
	guint size;
	g_object_get(adapter, "size", &size, NULL);
	return size;
}

/*
 * ============================================================================
 *
 *                         GstBaseTransform Overrides
 *
 * ============================================================================
 */


/*
 * sink_event()
 */


static gboolean gstlal_inpaint_sink_event(GstBaseTransform *trans, GstEvent *event) {
	GSTLALInpaint *inpaint = GSTLAL_INPAINT(trans);
	gboolean result = TRUE;

	GST_DEBUG_OBJECT(trans, "Got %s event on sink pad", GST_EVENT_TYPE_NAME (event));

	switch (GST_EVENT_TYPE(event)) {
		case GST_EVENT_CAPS:
		{
			GstCaps *caps;
			gint rate;
			gst_event_parse_caps(event, &caps);
			GstStructure *str = gst_caps_get_structure(caps, 0);
			gst_structure_get_int(str, "rate", &rate);
			inpaint->rate = (guint) rate;
			// FIXME Move elsewhere
			if(inpaint->psd != NULL)
				inv_fft_psd(inpaint);
			break;
		}
		case GST_EVENT_TAG:
		{
			GstTagList *taglist;
			gchar *instrument = NULL, *channel_name = NULL, *units = NULL;

			/*
			 * attempt to extract all 3 tags from the event's taglist
			 */

			gst_event_parse_tag(event, &taglist);
			result = taglist_extract_string(GST_OBJECT(trans), taglist, GSTLAL_TAG_INSTRUMENT, &instrument);
			result &= taglist_extract_string(GST_OBJECT(trans), taglist, GSTLAL_TAG_CHANNEL_NAME, &channel_name);
			result &= taglist_extract_string(GST_OBJECT(trans), taglist, GSTLAL_TAG_UNITS, &units);

			if(result) {
				GST_DEBUG_OBJECT(inpaint, "found tags \"%s\"=\"%s\", \"%s\"=\"%s\"", GSTLAL_TAG_INSTRUMENT, instrument, GSTLAL_TAG_CHANNEL_NAME, channel_name);
				free(inpaint->instrument);
				inpaint->instrument = instrument;
				free(inpaint->channel_name);
				inpaint->channel_name = channel_name;
				free(inpaint->units);
				inpaint->units = units;
			}
                        break;
		}
		default:
			break;
	}

	if(!result) 
		gst_event_unref(event);
	else 
		result = GST_BASE_TRANSFORM_CLASS(gstlal_inpaint_parent_class)->sink_event(trans, event);

	return result;
}


/*
 * transform_size()
 */

static gboolean gstlal_inpaint_transform_size(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, gsize size, GstCaps *othercaps, gsize *othersize) {
	GSTLALInpaint *inpaint = GSTLAL_INPAINT(trans);
	gsize unit_size = sizeof(double);

	switch(direction) {
	case GST_PAD_SRC:
		// Keep the sample count the same

		*othersize = size;
		break;

	case GST_PAD_SINK:
		/* number of samples available */
		*othersize = size / unit_size + gst_audioadapter_available_samples(inpaint->adapter);
		/* number of output bytes to be generated */
		// FIXME Dont hardcode
		// FIXME Will have to think about this more carefully for
		// general use. In theory, the procedure depends on exactly how
		// you whiten the data, so e.g. for the non-FIR whitener the
		// data may need to be inpainted in 2 steps and combined after
		// (just like how the data are whitened). In practice, it *may*
		// turn out that this is a negligible high level effect. For
		// test case, make sure data being inpainted are in the center
		// of the buffer
		if(*othersize < inpaint->rate * (guint) inpaint->fft_length_seconds / 2)
			*othersize = 0;
		else
			*othersize *= sizeof(double);
		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		return FALSE;
	}

	return TRUE;

}

/*
 * inpainting algorithm
 */
static GstFlowReturn gstlal_inpaint_process(GSTLALInpaint *inpaint, guint data_start, guint data_end, guint hole_start, guint hole_end) {
	fprintf(stderr, "in gstlal_inpaint_process...\n");
	GstFlowReturn result = GST_FLOW_OK;
	g_assert(inpaint->inv_cov_series->data->length % 2 == 0);
	g_assert(hole_end > hole_start);
	gsl_matrix *F_trans_mat = gsl_matrix_alloc(inpaint->inv_cov_series->data->length / 2, inpaint->inv_cov_series->data->length / 2);
	if(F_trans_mat == NULL) {
		GST_ERROR_OBJECT(GST_ELEMENT(inpaint), "failure allocating memory");
		result = GST_FLOW_ERROR;
		return result;
	}

	gint i, j;
	guint k;
	// The covariance series we got from the fft is the length of the fft
	// in sample points (N) while the length of the psd is N/2+1. The
	// covariance series is symmetric around N/2+1, though it has one
	// additional point before it than after it (i.e. index 1 and index N
	// are the same, index 2 and index N-1 are the same, etc). It is a
	// circulant matrix since the noise is stationary.  The value of a
	// given element in the covariance matrix only depends on the
	// difference between the row and column indices, with no difference
	// corresponding to the middle (N/2+1) of the covariance series.
	// F_trans_mat will store the inverse covariance matrix until the end,
	// when it will store the transformation matrix F.
	fprintf(stderr, "setting inverse covariance matrix\n");
	for(i = 0; i < (gint) F_trans_mat->size1; i++) {
		for(j = 0; j < (gint) F_trans_mat->size2; j++) {
			k = (guint) fabs(i - j) + inpaint->inv_cov_series->data->length / 2;
			gsl_matrix_set(F_trans_mat, (guint) i, (guint) j, inpaint->inv_cov_series->data->data[k]);
		}
	}

	// Matrix A from inpainting paper
	// A is size Nd x Nh, where Nd is the number of sample points in the
	// data being transformed, and Nh is the number of holes to inpaint. A
	// is zero except at points corresponding to a hole, where it is 1.
	// e.g. If there are 8 sample points where points 5 and 6 are the holes
	// to inpaint, then A is 1 at (4,0) and (5,1) and zero everywhere else:
	// 0 0
	// 0 0
	// 0 0
	// 0 0
	// 1 0
	// 0 1
	// 0 0
	// 0 0
	gsl_matrix *A_hole_mat = gsl_matrix_calloc(F_trans_mat->size1, hole_end - hole_start);
	if(A_hole_mat == NULL) {
		GST_ERROR_OBJECT(GST_ELEMENT(inpaint), "failure allocating memory");
		result = GST_FLOW_ERROR;
		return result;
	}

	j = 0;
	fprintf(stderr, "setting the a hole matrix\n");
	for(i = (gint) hole_start; i < (gint) hole_end; i++)
		gsl_matrix_set(A_hole_mat, (guint) i, (guint) j++, 1);

	// The matrices needed here are huge, so will allocate enough space for
	// 2, one of which will be discarded in the end, and the other will
	// store the final transformation matrix in the end.
	// FIXME Need to find a less memory intensive way to do this.
	gsl_matrix *mat_workspace1 = gsl_matrix_alloc(A_hole_mat->size1, A_hole_mat->size2);
	if(mat_workspace1 == NULL) {
		GST_ERROR_OBJECT(GST_ELEMENT(inpaint), "failure allocating memory");
		result = GST_FLOW_ERROR;
		return result;
	}
	gsl_matrix *M_trans_mat = gsl_matrix_alloc(A_hole_mat->size2, A_hole_mat->size2);
	if(M_trans_mat == NULL) {
		GST_ERROR_OBJECT(GST_ELEMENT(inpaint), "failure allocating memory");
		result = GST_FLOW_ERROR;
		return result;
	}
	gsl_matrix *M_inv_trans_mat = gsl_matrix_alloc(A_hole_mat->size2, A_hole_mat->size2);
	if(M_inv_trans_mat == NULL) {
		GST_ERROR_OBJECT(GST_ELEMENT(inpaint), "failure allocating memory");
		result = GST_FLOW_ERROR;
		return result;
	}

	// M = A^T * C^{-1} * A. This will be stored in mat_workspace1 and then inverted in-place
	fprintf(stderr, "performing C^{-1} * A\n");
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, F_trans_mat, A_hole_mat, 0., mat_workspace1);
	fprintf(stderr, "performing A^T*C^{-1}*A\n");
	gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1., A_hole_mat, mat_workspace1, 0., M_trans_mat);

	// Perform an LU decomposition of M, which is required to compute its
	// inverse using the gsl function gsl_linalg_LU_invert
	int signum;
	gsl_permutation *permutation = gsl_permutation_alloc(M_trans_mat->size1);
	if(permutation == NULL) {
		GST_ERROR_OBJECT(GST_ELEMENT(inpaint), "failure allocating memory");
		result = GST_FLOW_ERROR;
		return result;
	}
	fprintf(stderr, "Performing LU decomposition of M\n");
	gsl_linalg_LU_decomp(M_trans_mat, permutation, &signum);
	fprintf(stderr, "inverting M\n");
	gsl_linalg_LU_invert(M_trans_mat, permutation, M_inv_trans_mat);
	gsl_matrix_free(M_trans_mat);
	gsl_permutation_free(permutation);

	// Perform A*M^{-1}*A^T*C^{-1}
	gsl_matrix *mat_workspace2 = gsl_matrix_alloc(F_trans_mat->size1, F_trans_mat->size1);
	if(mat_workspace2 == NULL) {
		GST_ERROR_OBJECT(GST_ELEMENT(inpaint), "failure allocating memory");
		result = GST_FLOW_ERROR;
		return result;
	}
	fprintf(stderr, "performing A*M^{-1}\n");
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1., A_hole_mat, M_inv_trans_mat, 0., mat_workspace1);
	gsl_matrix_free(M_inv_trans_mat);
	fprintf(stderr, "performing A*M^{-1}*A^T\n");
	gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1., mat_workspace1, A_hole_mat, 0., mat_workspace2);
	gsl_matrix_free(A_hole_mat);
	gsl_matrix_free(mat_workspace1);
	mat_workspace1 = gsl_matrix_alloc(F_trans_mat->size1, F_trans_mat->size1);
	if(mat_workspace1 == NULL) {
		GST_ERROR_OBJECT(GST_ELEMENT(inpaint), "failure allocating memory");
		result = GST_FLOW_ERROR;
		return result;
	}
	fprintf(stderr, "Performing A*M^{-1}*A^T*C^{-1}\n");
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1., mat_workspace2, F_trans_mat, 0., mat_workspace1);
	gsl_matrix_free(mat_workspace2);

	// Subtract from identity to form F = 1 - A*M^{-1}*A^T*C^{-1}
	gsl_matrix_free(F_trans_mat);
	/*
	gsl_matrix_set_identity(F_trans_mat);
	fprintf(stderr, "Computing F\n");
	gsl_matrix_sub(F_trans_mat, mat_workspace1);
	gsl_matrix_free(mat_workspace1);
	*/

	//gsl_blas_dgemv(CBLAS_TRANSPOSE_t TransA, double alpha, const gsl_matrix * A, const gsl_vector * x, double beta, gsl_vector * y)
	// Load data into vector and then perform transformation
	gsl_vector *hoft = gsl_vector_alloc(data_end - data_start);
	if(hoft == NULL) {
		GST_ERROR_OBJECT(GST_ELEMENT(inpaint), "failure allocating memory");
		result = GST_FLOW_ERROR;
		return result;
	}
	gsl_vector *hoft_transformed = gsl_vector_alloc(data_end - data_start);
	if(hoft_transformed == NULL) {
		GST_ERROR_OBJECT(GST_ELEMENT(inpaint), "failure allocating memory");
		result = GST_FLOW_ERROR;
		return result;
	}
	fprintf(stderr, "setting hoft vector\n");
	for(i = 0; i < (gint) hoft->size; i++)
		gsl_vector_set(hoft, (guint) i, inpaint->transformed_data[(guint) i + data_start]);

	fprintf(stderr, "computing Fd\n");
	gsl_blas_dgemv(CblasNoTrans, 1., mat_workspace1, hoft, 0.0, hoft_transformed);
	gsl_matrix_free(mat_workspace1);
	gsl_vector_free(hoft);
	fprintf(stderr, "copying inpainted data\n");
	for(i = (gint) data_start; i < (gint) (data_end - data_start); i++)
		inpaint->transformed_data[(guint) i]  -= hoft_transformed->data[(guint) i];
	gsl_vector_free(hoft_transformed);
	fprintf(stderr, "gstlal_inpaint_process done\n");
	return result;

}

/*
 * transform()
 */

static GstFlowReturn gstlal_inpaint_transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf) {
	GSTLALInpaint *inpaint = GSTLAL_INPAINT(trans);
	GstFlowReturn result = GST_FLOW_OK;
	LIGOTimeGPS gate_start, gate_end, t0_GPS, t_idx;
	gint idx;

	// Prototype: Just set h(t) during GW170817 gate to zero in L1 (test to
	// make sure I understand how these pieces fit together since this is
	// my first transform element)
	// GW170817 gate: 1187008881.37875 to 1187008881.44125

	if(inpaint->t0 == GST_CLOCK_TIME_NONE) {
		inpaint->t0 = GST_BUFFER_PTS(inbuf);
		inpaint->initial_offset = GST_BUFFER_OFFSET(inbuf);
	}
	gst_buffer_ref(inbuf); // If this is not called, buffer will be unref'd by calling code
	gst_audioadapter_push(inpaint->adapter, inbuf);

	gint n_samples = (gint) gst_audioadapter_available_samples(inpaint->adapter);
	if(n_samples < (gint) inpaint->rate * (gint) inpaint->fft_length_seconds / 2) {
		gst_buffer_set_size(outbuf,  0);
		GST_BUFFER_OFFSET(outbuf) = inpaint->initial_offset;
		GST_BUFFER_OFFSET_END(outbuf) = inpaint->initial_offset;
		GST_BUFFER_PTS(outbuf) = inpaint->t0;
		GST_BUFFER_DURATION(outbuf) = (GstClockTime) gst_util_uint64_scale_int_round(GST_SECOND, 0, inpaint->rate);
		return result;
	}
	guint outsamples = inpaint->rate * (guint) inpaint->fft_length_seconds / 2;

	inpaint->outbuf_length = outsamples;
	inpaint->transformed_data = calloc(outsamples, sizeof(double));
	if(!inpaint->transformed_data) {
		GST_ERROR_OBJECT(GST_ELEMENT(trans), "failure allocating memory");
		result = GST_FLOW_ERROR;
		return result;
	}
	//FIXME Dont hardcode everything for specific test case
	gst_audioadapter_copy_samples(inpaint->adapter, inpaint->transformed_data, outsamples, NULL, NULL);

	//FIXME Dont hardcode everything for specific test case
	XLALGPSSet(&gate_start, 1187008881, 378750000);
	XLALGPSSet(&gate_end, 1187008881, 441250000);
	XLALGPSSet(&t0_GPS, (INT4) (inpaint->t0 / 1000000000), 0);

	//FIXME Dont hardcode everything for specific test case
	double dt = 1./ (double) inpaint->rate;
	guint gate_min = G_MAXUINT;
	guint gate_max = 0;
	for(idx=0; idx < (gint) outsamples; idx++) {
		if(idx == 0)
			t_idx = t0_GPS;
		else
			XLALGPSAdd(&t_idx, dt);

		if(XLALGPSCmp(&t_idx, &gate_start) >= 0 && XLALGPSCmp(&t_idx, &gate_end) == -1) {
			gate_min = gate_min < (guint) idx ? gate_min : (guint) idx;
			gate_max = (guint) idx;
		}
	}

	if(gate_min < G_MAXUINT) {
		fprintf(stderr, "n_samples = %u, outsamples = %u, gate_min = %u, gate_max = %u, t0=%u + 1e-9*%u\n", n_samples, outsamples, gate_min, gate_max + 1, (guint) (inpaint->t0 / 1000000000), (guint) ( inpaint->t0 - 1000000000*(inpaint->t0 / 1000000000)));
		result = gstlal_inpaint_process(inpaint, 0, outsamples, gate_min, ++gate_max);
	}

	GstMapInfo mapinfo;
	gst_buffer_map(outbuf, &mapinfo, GST_MAP_WRITE);
	memcpy(mapinfo.data, inpaint->transformed_data, outsamples * sizeof(double));
	gst_buffer_unmap(outbuf, &mapinfo);

	gst_audioadapter_flush_samples(inpaint->adapter, outsamples);
	GST_BUFFER_OFFSET(outbuf) = inpaint->initial_offset;
	GST_BUFFER_OFFSET_END(outbuf) = inpaint->initial_offset + inpaint->outbuf_length;
	GST_BUFFER_PTS(outbuf) = inpaint->t0;
	GST_BUFFER_DURATION(outbuf) = (GstClockTime) gst_util_uint64_scale_int_round(GST_SECOND, inpaint->outbuf_length, inpaint->rate);
	inpaint->t0 += GST_BUFFER_DURATION(outbuf);


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
 * Properties
 */


enum property {
	ARG_FFT_LENGTH = 1,
	ARG_PSD
};


static void gstlal_inpaint_set_property(GObject * object, enum property id, const GValue * value, GParamSpec * pspec) {
	GSTLALInpaint *inpaint = GSTLAL_INPAINT(object);

	GST_OBJECT_LOCK(inpaint);

	switch (id) {
	case ARG_FFT_LENGTH: {
		double fft_length_seconds = g_value_get_double(value);
		if(fft_length_seconds != inpaint->fft_length_seconds) {
			/*
			 * record new value
			 */

			inpaint->fft_length_seconds = fft_length_seconds;

			// FIXME Set up notification handlers to deal with
			// fft_length changing, since other elements (e.g.
			// lal_whiten) allow for this to happen
		}
		break;
	}

	case ARG_PSD: {
		// FIXME GValueArray is deprecated, switch to GArray once the rest of gstlal does
		GValueArray *va = g_value_get_boxed(value);

		// FIXME add units to inpaint struct
		LALUnit psd_units = gstlal_lalUnitSquaredPerHertz(lalDimensionlessUnit);
		inpaint->psd = XLALCreateREAL8FrequencySeries("PSD", &GPS_ZERO, 0.0, 1.0 / inpaint->fft_length_seconds, &psd_units, va->n_values);
		if(!inpaint->psd) {
			GST_ERROR("XLALCreateREAL8FrequencySeries() failed: %s", XLALErrorString(XLALGetBaseErrno()));
			XLALClearErrno();
		}
		gstlal_doubles_from_g_value_array(va, inpaint->psd->data->data, NULL);
		// FIXME Move elsewhere
		if(inpaint->rate != 0)
			inv_fft_psd(inpaint);
		break;
	}

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(inpaint);
}


static void gstlal_inpaint_get_property(GObject * object, enum property id, GValue * value, GParamSpec * pspec) {
	GSTLALInpaint *inpaint = GSTLAL_INPAINT(object);

	GST_OBJECT_LOCK(inpaint);

	switch (id) {
	case ARG_FFT_LENGTH:
		g_value_set_double(value, inpaint->fft_length_seconds);
		break;

	case ARG_PSD:
		if(inpaint->psd)
			g_value_take_boxed(value, gstlal_g_value_array_from_doubles(inpaint->psd->data->data, inpaint->psd->data->length));
		else
			// FIXME Switch from g_value_array_new once gstlal moves from the deprecated GValueArray to GValue
			g_value_take_boxed(value, g_value_array_new(0));
			//g_value_take_boxed(value, g_array_sized_new(TRUE, TRUE, sizeof(double), 0));
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(inpaint);
}


/*
 * finalize()
 */


static void gstlal_inpaint_finalize(GObject * object) {
	GSTLALInpaint *inpaint = GSTLAL_INPAINT(object);

	free(inpaint->instrument);
	free(inpaint->channel_name);
	free(inpaint->units);

	gst_audioadapter_clear(inpaint->adapter);
	g_object_unref(inpaint->adapter);
	inpaint->adapter = NULL;

	XLALDestroyREAL8FrequencySeries(inpaint->psd);
	inpaint->psd = NULL;
	XLALDestroyREAL8TimeSeries(inpaint->inv_cov_series);
	inpaint->inv_cov_series = NULL;

	free(inpaint->transformed_data);
	inpaint->transformed_data = NULL;

	G_OBJECT_CLASS(gstlal_inpaint_parent_class)->finalize(object);
}


/*
 * class_init()
 */


#define CAPS \
	"audio/x-raw, " \
	"format = (string) " GST_AUDIO_NE(F64) ", " \
	"rate = " GST_AUDIO_RATE_RANGE ", " \
	"channels = (int) 1, " \
	"layout = (string) interleaved, " \
	"channel-mask = (bitmask) 0"


static void gstlal_inpaint_class_init(GSTLALInpaintClass *klass) {
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(klass);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(gstlal_inpaint_set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(gstlal_inpaint_get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(gstlal_inpaint_finalize);

	transform_class->sink_event = GST_DEBUG_FUNCPTR(gstlal_inpaint_sink_event);
	transform_class->transform = GST_DEBUG_FUNCPTR(gstlal_inpaint_transform);
	transform_class->transform_size = GST_DEBUG_FUNCPTR(gstlal_inpaint_transform_size);

	gst_element_class_set_metadata(
		element_class,
		"Inpaint",
		"Filter",
		"A routine that replaces replaces glitchy data with data based on the surrounding times.",
		"Cody Messick <cody.messick@ligo.org>"
	);

	g_object_class_install_property(
		gobject_class,
		ARG_FFT_LENGTH,
		g_param_spec_double(
			"fft-length",
			"FFT length",
			"Total length of the FFT convolution (including zero padding) in seconds",
			0, G_MAXDOUBLE, DEFAULT_FFT_LENGTH_SECONDS,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);

	g_object_class_install_property(
		gobject_class,
		ARG_PSD,
		g_param_spec_value_array(
			"psd",
			"PSD",
			"Power spectral density that describes the data at the time of the hole being inpainted.  First bin is at 0 Hz, last bin is at f-nyquist, bin spacing is delta-f.",
			g_param_spec_double(
				"bin",
				"Bin",
				"Power spectral density bin",
				0, G_MAXDOUBLE, 1.0,
				G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
			),
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);

	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"sink",
			GST_PAD_SINK,
			GST_PAD_ALWAYS,
			gst_caps_from_string(CAPS)
		)
	);

	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"src",
			GST_PAD_SRC,
			GST_PAD_ALWAYS,
			gst_caps_from_string(CAPS)
		)
	);
}


/*
 * instance init
 */


static void gstlal_inpaint_init(GSTLALInpaint *inpaint) {
	inpaint->instrument = NULL;
	inpaint->channel_name = NULL;
	inpaint->units = NULL;
	inpaint->adapter = g_object_new(GST_TYPE_AUDIOADAPTER, "unit-size", sizeof(double), NULL);
	inpaint->transformed_data = NULL;
	inpaint->rate = 0;

	inpaint->initial_offset = 0;
	inpaint->t0 = GST_CLOCK_TIME_NONE;
	inpaint->outbuf_length = 0;

	inpaint->fft_length_seconds = DEFAULT_FFT_LENGTH_SECONDS;
	inpaint->psd = NULL;
	inpaint->inv_cov_series = NULL;
}

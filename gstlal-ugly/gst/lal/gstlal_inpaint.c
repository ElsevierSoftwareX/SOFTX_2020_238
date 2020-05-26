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
#include <lal/Sequence.h>
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

static void free_workspace(GSTLALInpaint* inpaint) {

	if(inpaint->hoft_workspace != NULL) {
		XLALDestroyREAL8TimeSeries(inpaint->hoft_workspace);
		inpaint->hoft_workspace = NULL;
	}

	if(inpaint->output_hoft != NULL) {
		XLALDestroyREAL8TimeSeries(inpaint->output_hoft);
		inpaint->output_hoft = NULL;
	}

	g_free(inpaint->inv_cov_mat_workspace);
	inpaint->inv_cov_mat_workspace = NULL;

	if(inpaint->M_trans_mat_workspace != NULL) {
		free(inpaint->M_trans_mat_workspace);
		inpaint->M_trans_mat_workspace = NULL;
	}

	if(inpaint->inv_M_trans_mat_workspace != NULL) {
		free(inpaint->inv_M_trans_mat_workspace);
		inpaint->inv_M_trans_mat_workspace = NULL;
	}

	if(inpaint->F_trans_mat_workspace != NULL) {
		free(inpaint->F_trans_mat_workspace);
		inpaint->F_trans_mat_workspace = NULL;
	}

	if(inpaint->hann_window != NULL) {
		XLALDestroyREAL8Window(inpaint->hann_window);
		inpaint->hann_window = NULL;
	}
}


//FIXME Figure out best way to handle errors here, currently I believe it will flag an error available in some GST_DEBUG
//FIXME Need to call this when any psd properties change
static void make_workspace(GSTLALInpaint *inpaint) {
	// Always start by clearing workspace, just in case
	free_workspace(inpaint);
	LALUnit strain_units = lalStrainUnit;

	inpaint->fft_length_samples = (guint) (inpaint->rate * (guint) inpaint->fft_length_seconds);

	inpaint->hoft_workspace = XLALCreateREAL8TimeSeries(NULL, &GPS_ZERO, 0.0, (double) 1.0 / inpaint->rate, &strain_units, inpaint->fft_length_samples);
	if(inpaint->hoft_workspace == NULL)
		GST_ERROR_OBJECT(GST_ELEMENT(inpaint), "failure creating REAL8TimeSeries");

	inpaint->output_hoft = XLALCreateREAL8TimeSeries(NULL, &GPS_ZERO, 0.0, (double) 1.0 / inpaint->rate, &strain_units, inpaint->fft_length_samples);
	if(inpaint->output_hoft == NULL)
		GST_ERROR_OBJECT(GST_ELEMENT(inpaint), "failure creating REAL8TimeSeries");


	inpaint->inv_cov_mat_workspace = g_malloc(inpaint->fft_length_samples * inpaint->fft_length_samples * sizeof(double));
	if(inpaint->inv_cov_mat_workspace == NULL)
		GST_ERROR_OBJECT(GST_ELEMENT(inpaint), "failure allocating memory");

	//FIXME Figure out max duration that can be inpainted, which will set the max size of this matrix and its inverse
	inpaint->M_trans_mat_workspace = calloc(inpaint->fft_length_samples * inpaint->fft_length_samples / 4, sizeof(double));
	if(inpaint->M_trans_mat_workspace == NULL)
		GST_ERROR_OBJECT(GST_ELEMENT(inpaint), "failure allocating memory");

	inpaint->inv_M_trans_mat_workspace = calloc(inpaint->fft_length_samples * inpaint->fft_length_samples / 4, sizeof(double));
	if(inpaint->inv_M_trans_mat_workspace == NULL)
		GST_ERROR_OBJECT(GST_ELEMENT(inpaint), "failure allocating memory");

	inpaint->F_trans_mat_workspace = calloc(inpaint->fft_length_samples * inpaint->fft_length_samples / 4, sizeof(double));
	if(inpaint->F_trans_mat_workspace == NULL)
		GST_ERROR_OBJECT(GST_ELEMENT(inpaint), "failure allocating memory");

	inpaint->hann_window = XLALCreateHannREAL8Window(inpaint->fft_length_samples / 2 + 1);
	if(inpaint->hann_window == NULL) {
		GST_ERROR_OBJECT(GST_ELEMENT(inpaint), "failure creating Hann window: %s", XLALErrorString(XLALGetBaseErrno()));
		XLALClearErrno();
	}

	// safety check in case numerical issues lead to the edge bins
	// being not exactly 0.  we require them to be exactly 0 in order
	// to affect the reset of the workspace between iterations
	// NOTE This is the same as is done in the whitening element
	g_assert_cmpfloat(inpaint->hann_window->data->data[0], ==, 0);
	g_assert_cmpfloat(inpaint->hann_window->data->data[inpaint->hann_window->data->length - 1], ==, 0);

	if(!XLALResizeREAL8Sequence(inpaint->hann_window->data, -1 * (gint) inpaint->fft_length_samples / 4, inpaint->fft_length_samples)) {
		GST_ERROR_OBJECT(GST_ELEMENT(inpaint), "failure resizing Hann window: %s", XLALErrorString(XLALGetBaseErrno()));
		XLALClearErrno();
	}
}

// FIXME This function needs to be called each time a new psd is passed in
static void fft_psd(GSTLALInpaint *inpaint) {
	// Set up the units
	// FIXME Add these to inpaint struct
	LALUnit strain_units = lalStrainUnit;
	LALUnit cov_units, inv_cov_units, inv_psd_units;
	XLALUnitMultiply(&cov_units, &strain_units, &strain_units);
	XLALUnitInvert(&inv_cov_units, &cov_units);
	LALUnit psd_units = gstlal_lalUnitSquaredPerHertz(lalDimensionlessUnit);
	XLALUnitInvert(&inv_psd_units, &psd_units);

	COMPLEX16FrequencySeries *complex_inv_psd = XLALCreateCOMPLEX16FrequencySeries("Complex PSD", &GPS_ZERO, 0.0,  1.0 / inpaint->fft_length_seconds, &inv_psd_units, inpaint->psd->data->length);
	REAL8TimeSeries *inv_cov_series = XLALCreateREAL8TimeSeries("Inverse Covariance", &GPS_ZERO, 0.0, 1.0 / (double) inpaint->rate, &inv_cov_units, inpaint->rate * (guint) inpaint->fft_length_seconds);
	//REAL8TimeSeries *tmp = XLALCreateREAL8TimeSeries("tmp", &GPS_ZERO, 0.0, 1.0 / (double) inpaint->rate, &inv_cov_units, inpaint->rate * (guint) inpaint->fft_length_seconds);

	guint i;
	for(i=0; i < inpaint->psd->data->length; i++)
		complex_inv_psd->data->data[i] = (COMPLEX16) (1.0 / inpaint->psd->data->data[i]);

	REAL8FFTPlan *revplan = XLALCreateReverseREAL8FFTPlan(inpaint->rate * (guint) inpaint->fft_length_seconds, 1);
	//if(XLALREAL8FreqTimeFFT(tmp, complex_inv_psd, revplan)){
	if(XLALREAL8FreqTimeFFT(inv_cov_series, complex_inv_psd, revplan)){
		GST_ERROR_OBJECT(inpaint, "XLALREAL8FreqTimeFFT() failed: %s", XLALErrorString(XLALGetBaseErrno()));
		XLALClearErrno();
	}

	// Rearrange autocovariance to go from -FFT LENGTH/2 to FFT LENGTH/2 instead of 0 to FFT LENGTH
	/*
	for(i=0; i < inv_cov_series->data->length / 2; i++)
		inv_cov_series->data->data[i] = tmp->data->data[i+tmp->data->length/2];
	for(; i < inv_cov_series->data->length; i++)
		inv_cov_series->data->data[i] = tmp->data->data[i - tmp->data->length/2];
	*/

	inpaint->inv_cov_series = inv_cov_series;
	XLALDestroyCOMPLEX16FrequencySeries(complex_inv_psd);
	//XLALDestroyREAL8TimeSeries(tmp);
	XLALDestroyREAL8FFTPlan(revplan);
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
			if(inpaint->psd != NULL) {
				fft_psd(inpaint);
				make_workspace(inpaint);
			}
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
		// number of samples available
		*othersize = size / unit_size + gst_audioadapter_available_samples(inpaint->adapter);
		// number of output bytes to be generated
		// FIXME Dont hardcode
		// FIXME Will have to think about this more carefully for
		// general use. In theory, the procedure depends on exactly how
		// you whiten the data, so e.g. for the non-FIR whitener the
		// data may need to be inpainted in 2 steps and combined after
		// (just like how the data are whitened). In practice, it *may*
		// turn out that this is a negligible high level effect. For
		// test case, make sure data being inpainted are in the center
		// of the buffer
		if(*othersize < inpaint->rate * (guint) inpaint->fft_length_seconds / 4)
			*othersize = 0;
		else
			*othersize = inpaint->rate * (guint) inpaint->fft_length_seconds * sizeof(double) / 4;
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
static GstFlowReturn gstlal_inpaint_process(GSTLALInpaint *inpaint, guint hole_start, guint hole_length) {
	fprintf(stderr, "in gstlal_inpaint_process...\n");
	GstFlowReturn result = GST_FLOW_OK;
	g_assert(inpaint->inv_cov_series->data->length % 2 == 0);
	gsl_matrix_view inv_cov_mat_view = gsl_matrix_view_array(inpaint->inv_cov_mat_workspace, inpaint->fft_length_samples, inpaint->fft_length_samples);

	guint i;
	// The covariance and inverse convariance matrices are symmetric and
	// circulant.  The value of a given element in the inverse covariance
	// matrix only depends on the difference between the row and column
	// indices, with no difference corresponding to the first entry of the
	// covariance series.
	fprintf(stderr, "Setting inverse covariance matrix\n");
	// FIXME Move to some function (other than make_workspace) so that this can be done only once when inpainting multiple windows (Also add fft_psd function call to this function)
	gsl_vector_view cov_series_view, cov_matrix_row_view;
	for(i = 0; i < inpaint->fft_length_samples; i++) {
		cov_series_view = gsl_vector_view_array(inpaint->inv_cov_series->data->data, inpaint->fft_length_samples - i);
		cov_matrix_row_view = gsl_vector_view_array(inpaint->inv_cov_mat_workspace + i * (inpaint->fft_length_samples + 1), inpaint->fft_length_samples - i);
		gsl_blas_dcopy(&cov_series_view.vector, &cov_matrix_row_view.vector);
		if(i > 0) {
			cov_series_view = gsl_vector_view_array(inpaint->inv_cov_series->data->data + inpaint->fft_length_samples - i, i);
			cov_matrix_row_view = gsl_vector_view_array(inpaint->inv_cov_mat_workspace + i * inpaint->fft_length_samples, i);
			gsl_blas_dcopy(&cov_series_view.vector, &cov_matrix_row_view.vector);

		}
	}

	// M is a subset of C^{-1} and is comprised of the rows and columns of
	// C^{-1} that correspond to the holes we want to inpaint. e.g. If we
	// had 8 samples wanted to inpaint the third through the fifth samples,
	// M would be
	// C^{-1}_{22} C^{-1}_{23} C^{-1}_{24}
	// C^{-1}_{32} C^{-1}_{33} C^{-1}_{34}
	// C^{-1}_{42} C^{-1}_{43} C^{-1}_{44}
	gsl_matrix_view M_trans_mat_view = gsl_matrix_view_array(inpaint->M_trans_mat_workspace, hole_length, hole_length);
	gsl_matrix_view inv_cov_submat_view = gsl_matrix_submatrix(&inv_cov_mat_view.matrix, hole_start, hole_start, hole_length, hole_length);
	gsl_matrix_memcpy(&M_trans_mat_view.matrix, &inv_cov_submat_view.matrix);

	gsl_matrix_view inv_M_trans_mat_view = gsl_matrix_view_array(inpaint->inv_M_trans_mat_workspace,M_trans_mat_view.matrix.size1, M_trans_mat_view.matrix.size2);
	// Perform an LU decomposition of M, which is required to compute its
	// inverse using the gsl function gsl_linalg_LU_invert
	int signum;
	gsl_permutation *permutation = gsl_permutation_alloc(hole_length);
	if(permutation == NULL) {
		GST_ERROR_OBJECT(GST_ELEMENT(inpaint), "failure allocating memory");
		result = GST_FLOW_ERROR;
		return result;
	}

	fprintf(stderr, "Performing LU decomposition of M\n");
	gsl_linalg_LU_decomp(&M_trans_mat_view.matrix, permutation, &signum);
	fprintf(stderr, "inverting M\n");
	gsl_linalg_LU_invert(&M_trans_mat_view.matrix, permutation, &inv_M_trans_mat_view.matrix);
	fprintf(stderr, "done inverting M\n");
	gsl_permutation_free(permutation);

	// A M^{-1} A^T is a block diagonal matrix with the same dimensions as
	// C^{-1}, where the only nonzero components are the submatrix M^{-1},
	// whose upper left element is the row and column that correspond to
	// the first sample that is being inpainted. Thus the only non-zero
	// components of A M^{-1} A^T C^{-1} can be computed using just M^{-1}.
	// Furthermore we know that a submatrix of the resulting matrix
	// will be the identity matrix, but we only actually need the columns
	// outside of this identity submatrix because the result will be
	// subtracted from an identity matrix (which is the size of C^{-1}).
	// For example, if you have 8 samples and you want to inpaint the third
	// through fifth samples, then A M^{-1} A^T C^{-1} will have the form
	// 0 0 0 0 0 0 0 0
	// 0 0 0 0 0 0 0 0
	// * * 1 0 0 * * *
	// * * 0 1 0 * * *
	// * * 0 0 1 * * *
	// 0 0 0 0 0 0 0 0
	// 0 0 0 0 0 0 0 0
	// 0 0 0 0 0 0 0 0
	// where '*' represents non-zero values that are not generally unity
	// (technically '*' could be zero, but it's unlikely unless the FFT
	// length is very large). We can compute these values by computing the
	// product of M^{-1} and submatrices of C^{-1}, one for the columns to
	// the left of the identity submatrix and another submatrix for the
	// columns to the right of the identity submatrix.

	gsl_matrix_view F_trans_mat_view = gsl_matrix_view_array(inpaint->F_trans_mat_workspace, hole_length, inpaint->fft_length_samples - hole_length);
	gsl_vector_view inpainted_hoft_view = gsl_vector_view_array(inpaint->output_hoft->data->data + hole_start, hole_length);
	gsl_vector_view relevant_hoft_view;
	// First compute the columns to the left of the identity submatrix, if
	// there are any, and then use those to partially inpaint the holes
	if(hole_start > 0) {
		gsl_matrix_view left_inv_cov_submat = gsl_matrix_submatrix(&inv_cov_mat_view.matrix, hole_start, 0, hole_length, hole_start);
		gsl_matrix_view left_F_submat = gsl_matrix_submatrix(&F_trans_mat_view.matrix, 0, 0, hole_length, hole_start);
		// Multiply result by -1 to account for subtracting A M^{-1} A^T C^{-1} from identity matrix
		fprintf(stderr, "Calculating left columns of F\n");
		gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1., &inv_M_trans_mat_view.matrix, &left_inv_cov_submat.matrix, 0., &left_F_submat.matrix);

		// The inpaint transformation matrix F only modifies the inpainted
		// samples, replacing them with linear combinations of the other
		// samples.
		relevant_hoft_view = gsl_vector_view_array(inpaint->output_hoft->data->data, hole_start);

		fprintf(stderr, "Inpainting data stage 1\n");
		gsl_blas_dgemv(CblasNoTrans, 1., &left_F_submat.matrix, &relevant_hoft_view.vector, 0.0, &inpainted_hoft_view.vector);
	}
	// Now compute the columns to the right of the identity submatrix, if
	// there are any, and then use those to finish inpainting the holes
	if(hole_start + hole_length + 1 < inpaint->fft_length_samples) {
		gsl_matrix_view right_inv_cov_submat = gsl_matrix_submatrix(&inv_cov_mat_view.matrix, hole_start, hole_start + hole_length, hole_length, inpaint->fft_length_samples - hole_start - hole_length);
		gsl_matrix_view right_F_submat = gsl_matrix_submatrix(&F_trans_mat_view.matrix, 0, hole_start, hole_length, F_trans_mat_view.matrix.size2 - hole_start);
		// Multiply result by -1 to account for subtracting A M^{-1} A^T C^{-1} from identity matrix
		fprintf(stderr, "Calculating right columns of F\n");
		gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1., &inv_M_trans_mat_view.matrix, &right_inv_cov_submat.matrix, 0., &right_F_submat.matrix);
		relevant_hoft_view = gsl_vector_view_array(inpaint->output_hoft->data->data + hole_start + hole_length, inpaint->fft_length_samples - hole_start - hole_length);
		// If there were columns to the left of the identify submatrix,
		// add the result of the inpainting process to the partially
		// inpainted samples. If not, replace the samples being
		// inpainted.
		double b;
		if(hole_start > 0)
			b = 1.0;
		else
			b = 0.0;
		fprintf(stderr, "Inpainting data stage 2\n");
		gsl_blas_dgemv(CblasNoTrans, 1., &right_F_submat.matrix, &relevant_hoft_view.vector, b, &inpainted_hoft_view.vector);
	}

	fprintf(stderr, "gstlal_inpaint_process done\n");
	return result;

}

/*
 * transform()
 */

// FIXME Need to address problem of gaps...
static GstFlowReturn gstlal_inpaint_transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf) {
	GSTLALInpaint *inpaint = GSTLAL_INPAINT(trans);
	GstFlowReturn result = GST_FLOW_OK;
	//LIGOTimeGPS gate_start, gate_end, t0_GPS, t_idx;
	LIGOTimeGPS gate_start, gate_end, t_idx;
	gint idx;
	guint outsamples;

	// FIXME Need to add logic to look for disconts and such
	if(inpaint->t0 == GST_CLOCK_TIME_NONE) {
		inpaint->t0 = GST_BUFFER_PTS(inbuf);
		inpaint->initial_offset = GST_BUFFER_OFFSET(inbuf);
	}
	gst_buffer_ref(inbuf); // If this is not called, buffer will be unref'd by calling code
	gst_audioadapter_push(inpaint->adapter, inbuf);

	gint n_samples = (gint) gst_audioadapter_available_samples(inpaint->adapter);
	if(n_samples < (gint) inpaint->fft_length_samples / 2) {
		gst_buffer_set_size(outbuf,  0);
		GST_BUFFER_OFFSET(outbuf) = inpaint->initial_offset;
		GST_BUFFER_OFFSET_END(outbuf) = inpaint->initial_offset;
		GST_BUFFER_PTS(outbuf) = inpaint->t0;
		GST_BUFFER_DURATION(outbuf) = (GstClockTime) gst_util_uint64_scale_int_round(GST_SECOND, 0, inpaint->rate);
		return result;
	}

	for(outsamples = 0; gst_audioadapter_available_samples(inpaint->adapter) >= inpaint->fft_length_samples / 2;) {
		//FIXME Dont hardcode everything for specific test case
		//FIXME Need to be a lot smarter about tracking time
		gst_audioadapter_copy_samples(inpaint->adapter, inpaint->hoft_workspace->data->data + inpaint->fft_length_samples / 4, inpaint->fft_length_samples / 2, NULL, NULL);
		XLALINT8NSToGPS(&inpaint->hoft_workspace->epoch, inpaint->t0);
		XLALGPSAdd(&inpaint->hoft_workspace->epoch, (double) ((gint) outsamples - (gint) inpaint->fft_length_samples / 4));

		//FIXME Dont hardcode everything for specific test case
		// GW170817 gate: 1187008881.37875 to 1187008881.44125
		XLALGPSSet(&gate_start, 1187008881, 378750000);
		XLALGPSSet(&gate_end, 1187008881, 441250000);
		//XLALGPSSet(&t0_GPS, (INT4) (inpaint->t0 / 1000000000), 0);

		//FIXME Dont hardcode everything for specific test case
		double dt = 1./ (double) inpaint->rate;
		guint gate_min = G_MAXUINT;
		guint gate_max = 0;
		for(idx=0; idx < (gint) inpaint->fft_length_samples; idx++) {
			if(idx == 0)
				t_idx = inpaint->hoft_workspace->epoch;
			else
				XLALGPSAdd(&t_idx, dt);

			if(XLALGPSCmp(&t_idx, &gate_start) >= 0 && XLALGPSCmp(&t_idx, &gate_end) == -1) {
				gate_min = gate_min < (guint) idx ? gate_min : (guint) idx;
				gate_max = (guint) idx;
			}
		}

		if(gate_min < G_MAXUINT) {

			// Apply Hann window to data
			if(!XLALUnitaryWindowREAL8Sequence(inpaint->hoft_workspace->data, inpaint->hann_window)) {
				GST_ERROR_OBJECT(GST_ELEMENT(inpaint), "XLALUnitaryWindowREAL8Sequence() failed: %s", XLALErrorString(XLALGetBaseErrno()));
				XLALClearErrno();
				return GST_FLOW_ERROR;
			}

			fprintf(stderr, "n_samples = %u, outsamples = %u, gate_min = %u, gate_max = %u, t0=%u + 1e-9*%u\n", n_samples, inpaint->fft_length_samples / 4, gate_min, gate_max + 1, (guint) (inpaint->t0 / 1000000000), (guint) ( inpaint->t0 - 1000000000*(inpaint->t0 / 1000000000)));
			if(gstlal_inpaint_process(inpaint, gate_min, gate_max - gate_min + 1) == GST_FLOW_ERROR)
				return GST_FLOW_ERROR;

			// if the gate doesn't begin until a quarter way
			// through the window, then we aren't outputting any
			// inpainted samples in the next output, so just copy
			// the original samples into the output. Only copy the
			// samples that are going to be output.
			gsl_vector_view hoft_workspace_view, output_hoft_view;
			// Mix into output
			// 0) ________****************________
			// 1)         ________****************________
			// 2)                 ________****************________
			// 3)                         ________****************________
			// 4)                                 ________****************________
			// 5)                                         ________****************________
			//
			// Normal state of windowed h(t)
			// 0s represent unaltered times
			// 1s represent times that *may* be inpainted (assuming at least one sample is inpainted)
			// |<-----fft_length_samples----->|
			// |zeropad|<----data---->|zeropad|
			// |  1/4  |      1/2     |  1/4  |
			// ________****************________
			// 00000000000000000000000000000000
			// |<----workspace epoch, sample index 0
			//         |<----output epoch, sample index fft_length_samples / 4
			//                |<--last sample of output, sample index fft_length_samples / 2 - 1
			//
			// 1)
			// |<-----fft_length_samples----->|
			// |zeropad|<----data---->|zeropad|
			// |  1/4  |      1/2     |  1/4  |
			// ________****************________
			// 00000000000000000000000011111111
			// |<--workspace epoch, sample index 0, time t0
			//         |<--output epoch, sample index fft_length_samples / 4
			//                |<--last sample of output, sample index fft_length_samples / 2 - 1
			//
			// 2)
			// |<-----fft_length_samples----->|
			// |zeropad|<----data---->|zeropad|
			// |  1/4  |      1/2     |  1/4  |
			// ________****************________
			// 00000000000000001111111111111111
			// |<--workspace epoch, sample index 0, time t0+fft_length_samples/4
			//         |<--output epoch, sample index fft_length_samples / 4
			//                |<--last sample of output, sample index fft_length_samples / 2 - 1
			//
			// 3)
			// |<-----fft_length_samples----->|
			// |zeropad|<----data---->|zeropad|
			// |  1/4  |      1/2     |  1/4  |
			// ________****************________
			// 00000000111111111111111111111110
			// |<--workspace epoch, sample index 0, time t0+fft_length_samples/2
			// |<--output epoch, sample index 0 (no samples output this iteration)
			//
			// 4)
			// |<-----fft_length_samples----->|
			// |zeropad|<----data---->|zeropad|
			// |  1/4  |      1/2     |  1/4  |
			// --------****************--------
			// 11111111111111111111111000000000
			// |<--workspace epoch, sample index 0, time t0+3/4*fft_length_samples
			// |<--output epoch, sample index 0
			//        |<--last sample of output, sample index fft_length_samples / 4 - 1
			//
			// 5)
			// |<-----fft_length_samples----->|
			// |zeropad|<----data---->|zeropad|
			// |  1/4  |      1/2     |  1/4  |
			// --------****************--------
			// 11111111000000000000000000000000
			// |<--workspace epoch, sample index 0, time t0+fft_length_samples
			// |<--output epoch, sample index 0
			//        |<--last sample of output, sample index fft_length_samples / 4 - 1
			if(gate_min >= 3 * inpaint->fft_length_samples / 4) {
				// First iteration where the workspace time-series includes times to inpaint
				// Copy original samples to output, ensuring they are unchanged by inpainting
				// Copy over first iteration if inpainted samples
				// |<-----fft_length_samples----->|
				// |zeropad|<----data---->|zeropad|
				// |  1/4  |      1/2     |  1/4  |
				// ________****************________
				// 00000000000000000000000011111111
				// |<--workspace epoch, sample index 0, sample time inpaint->t0 - fft_length-samples / 4
				//         |<--output epoch, sample index fft_length_samples / 4
				//                |<--last sample of output, sample index fft_length_samples / 2 - 1
				gst_audioadapter_copy_samples(inpaint->adapter, inpaint->output_hoft->data->data, inpaint->fft_length_samples / 4, NULL, NULL);
				hoft_workspace_view = gsl_vector_view_array(inpaint->hoft_workspace->data->data + gate_min, 1 + gate_max - gate_min);
				output_hoft_view = gsl_vector_view_array(inpaint->output_hoft->data->data + gate_min, 1 + gate_max - gate_min);
				gsl_blas_dcopy(&hoft_workspace_view.vector, &output_hoft_view.vector);
				gst_audioadapter_flush_samples(inpaint->adapter, inpaint->fft_length_samples / 4);
				XLALINT8NSToGPS(&inpaint->output_hoft->epoch, inpaint->t0);
			} else if(gate_min >= inpaint->fft_length_samples / 2) {
				// Second iteration where the workspace time-series includes times to inpaint
				// Copy original samples to output, ensuring they are unchanged by inpainting
				// Add inpainted samples from this iteration to inpainted samples from previous iteration
				// |<-----fft_length_samples----->|
				// |zeropad|<----data---->|zeropad|
				// |  1/4  |      1/2     |  1/4  |
				// ________****************________
				// 00000000000000001111111111111111
				// |<--workspace epoch, sample index 0, sample time inpaint->t0 - fft_length-samples / 4
				//         |<--inpaint->t0
				//         |<--output epoch, sample index fft_length_samples / 4
				//                |<--last sample of output, sample index fft_length_samples / 2 - 1
				gst_audioadapter_copy_samples(inpaint->adapter, inpaint->output_hoft->data->data, inpaint->fft_length_samples / 4, NULL, NULL);
				hoft_workspace_view = gsl_vector_view_array(inpaint->hoft_workspace->data->data + gate_min, 1 + gate_max - gate_min);
				output_hoft_view = gsl_vector_view_array(inpaint->output_hoft->data->data + gate_min, 1 + gate_max - gate_min);
				gsl_blas_daxpy(1.0, &hoft_workspace_view.vector, &output_hoft_view.vector);
				gst_audioadapter_flush_samples(inpaint->adapter, inpaint->fft_length_samples / 4);
				XLALINT8NSToGPS(&inpaint->output_hoft->epoch, inpaint->t0);
			} else if(gate_min >= inpaint->fft_length_samples / 4) {
				// Third iteration where the workspace time-series includes times to inpaint
				// Copy original samples (if there are any) to output, ensuring they are unchanged by inpainting
				// Add inpainted samples from this iteration to inpainted samples from previous iteration
				// FIXME FIXME FIXME Haven't implemented the second copy in this loop yet, should maybe rethink it?
				// If end of inpainted times occurs in the first half of the data considered, copy original
				// samples after the inpainted times to output, ensuring they are unchanged by inpainting
				// |<-----fft_length_samples----->|
				// |zeropad|<----data---->|zeropad|
				// |  1/4  |      1/2     |  1/4  |
				// ________****************________
				// 00000000111111111111111111111110
				// |<--workspace epoch, sample index 0, sample index 0, sample time inpaint->t0 - fft_length-samples / 4
				// |<--output epoch, sample index 0 (no samples to output this iteration)
				gst_audioadapter_copy_samples(inpaint->adapter, inpaint->output_hoft->data->data + inpaint->fft_length_samples/4, gate_min - inpaint->fft_length_samples / 4, NULL, NULL);
				hoft_workspace_view = gsl_vector_view_array(inpaint->hoft_workspace->data->data + gate_min, 1 + gate_max - gate_min);
				output_hoft_view = gsl_vector_view_array(inpaint->output_hoft->data->data + gate_min, 1 + gate_max - gate_min);
				gsl_blas_daxpy(1.0, &hoft_workspace_view.vector, &output_hoft_view.vector);
				if(gate_max + 1 >= inpaint->fft_length_samples / 2)
					// End of inpainted time is not contained in the data being flushed
					gst_audioadapter_flush_samples(inpaint->adapter, inpaint->fft_length_samples / 4);
				else {
					// End of inpainted time is contained in data being flushed, thus need to copy non-inpainted samples out of the adapter
					gst_audioadapter_flush_samples(inpaint->adapter, 1 + gate_max - inpaint->fft_length_samples / 4);
					gst_audioadapter_copy_samples(inpaint->adapter, inpaint->output_hoft->data->data + gate_max + 1, inpaint->fft_length_samples / 2 - 1 - gate_max, NULL, NULL);
					gst_audioadapter_flush_samples(inpaint->adapter, inpaint->fft_length_samples / 2 - 1 - gate_max);
				}
				XLALINT8NSToGPS(&inpaint->output_hoft->epoch, inpaint->t0);
				XLALGPSAdd(&inpaint->output_hoft->epoch, -1.0 * (double) inpaint->fft_length_samples / 4);
			} else if(gate_max + 1 >= inpaint->fft_length_samples / 2) {
				// One possibility for the fourth iteration where the workspace time-series includes times to inpaint
				// Already copied original samples (if there are any) to output in last iteration
				// Add inpainted samples from this iteration to inpainted samples from previous iteration
				// |<-----fft_length_samples----->|
				// |zeropad|<----data---->|zeropad|
				// |  1/4  |      1/2     |  1/4  |
				// --------****************--------
				// 11111111111111111111111000000000
				// |<--workspace epoch, sample index 0, sample time inpaint->t0
				// |<--output epoch, sample index 0
				//        |<--last sample of output, sample index fft_length_samples / 4 - 1
				hoft_workspace_view = gsl_vector_view_array(inpaint->hoft_workspace->data->data + gate_min, 1 + gate_max - gate_min);
				output_hoft_view = gsl_vector_view_array(inpaint->output_hoft->data->data + gate_min, 1 + gate_max - gate_min);
				gsl_blas_daxpy(1.0, &hoft_workspace_view.vector, &output_hoft_view.vector);
				gst_audioadapter_flush_samples(inpaint->adapter, inpaint->fft_length_samples / 4);
				XLALINT8NSToGPS(&inpaint->output_hoft->epoch, inpaint->t0);
			} else if(gate_max + 1 >= inpaint->fft_length_samples / 4) {
				// Either the fourth iteration or fifth iteration where the workspace time-series includes times to inpaint
				// Already copied untouched samples from before inpainted samples to output in last iteration
				// Add inpainted samples from this iteration to inpainted samples from previous iteration
				// Copy original samples from after inpainted samples to output
				// |<-----fft_length_samples----->|
				// |zeropad|<----data---->|zeropad|
				// |  1/4  |      1/2     |  1/4  |
				// --------****************--------
				// 11111111111111100000000000000000
				// |<--workspace epoch, sample index 0, sample time inpaint->t0
				// |<--output epoch, sample index 0
				//        |<--last sample of output, sample index fft_length_samples / 4 - 1
				hoft_workspace_view = gsl_vector_view_array(inpaint->hoft_workspace->data->data + gate_min, 1 + gate_max - gate_min);
				output_hoft_view = gsl_vector_view_array(inpaint->output_hoft->data->data + gate_min, 1 + gate_max - gate_min);
				gsl_blas_daxpy(1.0, &hoft_workspace_view.vector, &output_hoft_view.vector);
				// First need to flush the samples that are being inpainted
				gst_audioadapter_flush_samples(inpaint->adapter, gate_max + 1 - inpaint->fft_length_samples / 4);
				// Copy the non-inpainted samples
				gst_audioadapter_copy_samples(inpaint->adapter, inpaint->output_hoft + gate_max + 1, inpaint->fft_length_samples / 2 - 1 - gate_max, NULL, NULL);
				// Flush more to have flushed a total of fft_length_samples/4
				gst_audioadapter_flush_samples(inpaint->adapter, inpaint->fft_length_samples / 2 - 1 - gate_max);
				XLALINT8NSToGPS(&inpaint->output_hoft->epoch, inpaint->t0);
			} else {
				// The last iteration where the workspace time-series includes times to inpaint
				// Already copied untouched samples from before inpainted samples to output in last iteration
				// Add inpainted samples from this iteration to inpainted samples from previous iteration
				// |<-----fft_length_samples----->|
				// |zeropad|<----data---->|zeropad|
				// |  1/4  |      1/2     |  1/4  |
				// --------****************--------
				// 11111110000000000000000000000000
				// |<--workspace epoch, sample index 0, sample time inpaint->t0
				// |<--output epoch, sample index 0
				//        |<--last sample of output, sample index fft_length_samples / 4 - 1
				hoft_workspace_view = gsl_vector_view_array(inpaint->hoft_workspace->data->data + gate_min, 1 + gate_max - gate_min);
				output_hoft_view = gsl_vector_view_array(inpaint->output_hoft->data->data + gate_min, 1 + gate_max - gate_min);
				gsl_blas_daxpy(1.0, &hoft_workspace_view.vector, &output_hoft_view.vector);
				XLALINT8NSToGPS(&inpaint->output_hoft->epoch, inpaint->t0);
			}
		} else {
			// FIXME Find a workaround to only do one copy when not inpainting any samples, currently doing 3...
			gsl_vector_view hoft_workspace_view = gsl_vector_view_array(inpaint->hoft_workspace->data->data + inpaint->fft_length_samples / 4, inpaint->fft_length_samples / 4);
			gsl_vector_view output_hoft_view = gsl_vector_view_array(inpaint->output_hoft->data->data, inpaint->fft_length_samples);
			gsl_blas_dcopy(&hoft_workspace_view.vector, &output_hoft_view.vector);
			XLALINT8NSToGPS(&inpaint->output_hoft->epoch, inpaint->t0);
		}

		LIGOTimeGPS t_cmp;
		if(XLALGPSCmp(&inpaint->output_hoft->epoch, XLALINT8NSToGPS(&t_cmp, inpaint->t0)) == 0) {
			outsamples += inpaint->fft_length_samples / 4;
			GstMapInfo mapinfo;
			gst_buffer_map(outbuf, &mapinfo, GST_MAP_WRITE);
			// Make sure we havent already filled the outbuf
			g_assert_cmpuint(outsamples, <=, mapinfo.size / sizeof(*mapinfo.data));
			memcpy(mapinfo.data, inpaint->output_hoft->data->data, inpaint->fft_length_samples / 4 * sizeof(double));
			gst_buffer_unmap(outbuf, &mapinfo);

			gst_audioadapter_flush_samples(inpaint->adapter, outsamples);
			GST_BUFFER_OFFSET(outbuf) = inpaint->initial_offset;
			inpaint->initial_offset += (guint64) outsamples;
			GST_BUFFER_OFFSET_END(outbuf) = inpaint->initial_offset;
			GST_BUFFER_PTS(outbuf) = inpaint->t0;
			GST_BUFFER_DURATION(outbuf) = (GstClockTime) gst_util_uint64_scale_int_round(GST_SECOND, outsamples, inpaint->rate);
			inpaint->t0 += GST_BUFFER_DURATION(outbuf);

		}
		// FIXME Dont hardcode inpainted times
		if(gate_min < G_MAXUINT) {
			memmove(inpaint->output_hoft->data->data, inpaint->output_hoft->data->data + inpaint->fft_length_samples / 4, 3*inpaint->fft_length_samples / 4 * sizeof(double));
			XLALGPSAdd(&inpaint->output_hoft->epoch, (double) (inpaint->fft_length_samples / 4));
			gst_audioadapter_flush_samples(inpaint->adapter, inpaint->fft_length_samples / 4);
		}
	}


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
		if(inpaint->rate != 0) {
			fft_psd(inpaint);
			make_workspace(inpaint);
		}
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

	free_workspace(inpaint);

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
	inpaint->rate = 0;

	inpaint->initial_offset = 0;
	inpaint->t0 = GST_CLOCK_TIME_NONE;

	inpaint->fft_length_seconds = DEFAULT_FFT_LENGTH_SECONDS;
	inpaint->psd = NULL;
	inpaint->inv_cov_series = NULL;

	free_workspace(inpaint);
}

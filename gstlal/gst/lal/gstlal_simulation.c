/*
 * An interface to LALSimulation.  
 *
 * Copyright (C) 2008--2015  Kipp Cannon, Chad Hanna, Drew Keppel
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
 * Actions:
 *
 * - Sort out extra padding for injection series, simulation series, etc.
 * - Make it possible to get all the injections from a frame file.
 * - Why is a nano-second taken out at the beginning and added at the end?
 * - Get lal to do a conditional taper at the start of the waveform and run it through
 * a high-pass filter (need to check if this is really needed for BNS).
 * - Should the conditional tapering be the responsibility of waveform
 * developers or could it be done outside waveform generation?
 * - consider patching lal to remove start/stop parameters from
 * XML loading functions so that they just load everything
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
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <complex.h>
#include <hdf5.h>


/*
 * stuff from gstreamer
 */


#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <gst/audio/audio.h>


/*
 * stuff from LAL
 */
#include <lal/PrintVector.h>
#include <lal/Date.h>
#include <lal/FindChirp.h>
#include <lal/FrequencySeries.h>
#include <lal/GenerateBurst.h>
#include <lal/LALConfig.h>
#include <lal/LALDatatypes.h>
#include <lal/LALInspiral.h>
#include <lal/LALSimInspiral.h>
#include <lal/LALSimulation.h>
#include <lal/LALStdlib.h>
#include <lal/LIGOLwXML.h>
#include <lal/LIGOMetadataTables.h>
#include <lal/LIGOMetadataUtils.h>
#include <lal/LIGOMetadataInspiralUtils.h>
#include <lal/SnglBurstUtils.h>
#include <lal/TimeSeries.h>
#include <lal/Units.h>
#include <lal/VectorOps.h>
#include <lal/TimeFreqFFT.h>
#include <lal/LALDetectors.h>
#include <lal/XLALError.h>
#include <lal/RealFFT.h>
#include <lal/LIGOLwXMLRead.h>
#include <lal/H5FileIO.h>
#include <lal/LIGOMetadataInspiralUtils.h>


/*
 * our own stuff
 */


#include <gstlal/ezligolw.h>
#include <gstlal/gstlal.h>
#include <gstlal/gstlal_tags.h>
#include <gstlal_simulation.h>

/*
 * files to run Salvo's hdf5 read function
 */

#include <lal/LALInferenceVCSInfo.h>
#include <lal/LALVCSInfoType.h>
#include <lal/LALInferenceHDF5.h>
#include <hdf5_hl.h>
#include <assert.h>

#include <assert.h>
#include <errno.h>
#include <lal/GenerateInspiral.h>
#include <lal/LALInference.h>
#include <lal/StringInput.h>
#include <lal/LALInferencePrior.h>
#include <lal/LALInferenceTemplate.h>
#include <lal/LALInferenceProposal.h>
#include <lal/LALInferenceLikelihood.h>
#include <lal/LALInferenceReadData.h>
#include <lal/LALInferenceInit.h>
#include <lal/LALInferenceCalibrationErrors.h>
#include <lal/LALInferenceHDF5.h>
#include <lal/LALSimNeutronStar.h>

#include <lal/LIGOMetadataUtils.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>

double PI = M_PI;

/*
 * ============================================================================
 *
 *                                Boilerplate
 *
 * ============================================================================
 */


#define GST_CAT_DEFAULT gstlal_simulation_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


G_DEFINE_TYPE_WITH_CODE(
	GSTLALSimulation,
	gstlal_simulation,
	GST_TYPE_BASE_TRANSFORM,
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "lal_simulation", 0, "lal_simulation element")
);


/*
 * ============================================================================
 *
 *	                  Calibration Errors Input
 *
 * ============================================================================
 */

int H5DatasetToREAL8Vector(LALH5Dataset *dataset, REAL8Vector **vector, UINT4 *Ndim)
{

    size_t type_size = XLALH5TableQueryRowSize(dataset);
    size_t Ndims = XLALH5TableQueryNColumns(dataset);
    int ret;
    size_t nbytes = XLALH5DatasetQueryNBytes(dataset);
    char *data = XLALMalloc(nbytes);
    assert(data);
    ret = XLALH5DatasetQueryData(data, dataset);
    assert(ret == 0);
    REAL8Vector *vec = NULL;
    vec = XLALCreateREAL8Vector(Ndims);

    /* Read the group dataset in as a vector */
    memcpy(vec->data, data, type_size);
    XLALFree(data);

    /* Construct the array of LALInferenceVariables */
    for(UINT4 i = 0; i < Ndims; i++)
    {
	(*vector)->data[i] = vec->data[i];
    }
    *Ndim = Ndims;
    XLALDestroyREAL8Vector(vec);
    return(XLAL_SUCCESS);
}

int H5DatasetToREAL8VectorArray(LALH5Dataset *dataset, REAL8Vector ***array, UINT4 *Nrow, UINT4 *Ncol)
{
    size_t type_size = XLALH5TableQueryRowSize(dataset);
    size_t Ncols = XLALH5TableQueryNColumns(dataset);
    int ret;
    size_t nbytes = XLALH5DatasetQueryNBytes(dataset);
    char *data = XLALMalloc(nbytes);
    assert(data);
    ret = XLALH5DatasetQueryData(data, dataset);
    assert(ret == 0);
    REAL8Vector **arr = NULL;
    UINT4 Nrows = XLALH5DatasetQueryNPoints(dataset);
    arr = XLALCalloc(Nrows, sizeof(REAL8Vector *));
    for (size_t i = 0; i < Nrows; i++)
        arr[i] = XLALCreateREAL8Vector(Ncols);
   /* Read the group datasets in as arrays */
   for (UINT4 i = 0; i < Nrows; i++)
        memcpy(arr[i]->data , data + type_size * i, type_size);
    XLALFree(data);

    /* Construct the array of LALInferenceVariables */
    //*array = arr;
    for (UINT4 i = 0; i < Nrows; i++)
    {
	for(UINT4 j = 0; j < Ncols; j++)
        {
		(*array)[i]->data[j] = arr[i]->data[j];
	}
    }
    *Nrow = Nrows;
    *Ncol = Ncols;
    for (UINT4 i = 0; i < Nrows; i++){
	XLALDestroyREAL8Vector(arr[i]);
    }
    return(XLAL_SUCCESS);
}

static void destroy_calib_errors_document(struct calib_errors_document *doc)
{
	if(doc) {
		for(UINT4 i=0; i<doc->Nrows; i++)
		{
			XLALDestroyREAL8Vector(doc->calib_errors_amplitude[i]);
			doc->calib_errors_amplitude[i] = NULL;
			XLALDestroyREAL8Vector(doc->calib_errors_phase[i]);
			doc->calib_errors_phase[i] = NULL;
		}
		XLALDestroyREAL8Vector(doc->calib_errors_freqs);
		doc->calib_errors_freqs = NULL;
	}
	g_free(doc);
}

static struct calib_errors_document *load_calib_errors_document(const char *filename, const char *instrument){

	struct calib_errors_document *out;

	g_assert(filename != NULL);

	/*
	 * allocate the document
	 */

	out = g_new0(struct calib_errors_document, 1);
	if(!out) {
		XLALPrintError("%s(): malloc() failed\n", __func__);
		goto allocfailed;
	}

	/*
	 * figure out which hdf5 to load from filename
	 */
	FILE *fp = NULL;
	fp = fopen(filename, "r"); // open file that should contain list of paths to HDF5 files
	LALH5File *calib_errors_file = NULL;
	if(fp != NULL){
		//FIXME: There's probably a better way to organize the input than a file with a list of hdf5 file paths in it - maybe use the frame cache tools? or does an xml make sense?
		char h5filename[1000];
		char ifo[2];
		int match = 0; // track if we ever get a match in IFOs
		while(fscanf(fp, "%s %s", ifo, h5filename) != EOF){ // loop through all the hdf5 files until find the one that matches the relevant detector
			calib_errors_file = XLALH5FileOpen(h5filename, "r");
			if(calib_errors_file == NULL){
				XLALErrorHandler = XLALExitErrorHandler;
				XLALPrintError("Input file error. Please check that the specified path exists. (in %s, line %d)\n",__FILE__, __LINE__);
			}
			if(strcmp(ifo, instrument) == 0){
				printf("Loading calibration errors for %s\n", instrument);
				match = 1;
				break;
			}
			else{
				XLALH5FileClose(calib_errors_file);
			}
		}
		if(match == 0){
			// FIXME: Does this exit the code or just the function?
			printf("No calibration error files provided match IFO.\n");
			XLALErrorHandler = XLALExitErrorHandler;
			XLALPrintError("No calibration error files provided that match IFO.");
			goto allocfailed;
		}
	}
	else{
		XLALPrintError("%s(): fopen() failed\n", __func__);
		goto fopenfailed;
	}
	fclose(fp);
	/*
	 * Load the calibration error magnitude, phase, and frequencies int REAL8Vectors
	 */

	LALH5Dataset *dataset = NULL;
	LALH5File *group = XLALH5GroupOpen(calib_errors_file, "deltaR");
	XLALH5FileClose(calib_errors_file);

	UINT4 n_cal_samps, n_cal_freqs, n;

	// Figure out the size of the arrays for amplitude and phase error
	dataset = XLALH5DatasetRead(group, "draws_amp_rel");
	size_t Ncols = XLALH5TableQueryNColumns(dataset);
	UINT4 Nrows = XLALH5DatasetQueryNPoints(dataset);
	XLALH5DatasetFree(dataset);

	// Allocate memory for each of these arrays
	// FIXME: I don't know if I need to allocate memory for the arrays being passed to the read file.  I should try commenting this stuff out and see if it works.
	out->calib_errors_freqs = XLALCalloc(1, sizeof(REAL8Vector*));
	out->calib_errors_freqs = XLALCreateREAL8Vector(Ncols);
	out->calib_errors_amplitude = XLALCalloc(Nrows, sizeof(REAL8Vector*));
	out->calib_errors_phase = XLALCalloc(Nrows, sizeof(REAL8Vector*));
	for(UINT4 i=0; i<Nrows; i++)
	{
		out->calib_errors_amplitude[i] = XLALCreateREAL8Vector(Ncols);
		out->calib_errors_phase[i] = XLALCreateREAL8Vector(Ncols);
	}

	/*Get freqs (assumes same number for amp and phase)*/
	dataset = XLALH5DatasetRead(group, "freq");
	H5DatasetToREAL8Vector(dataset, &(out->calib_errors_freqs), &n_cal_freqs);
	XLALH5DatasetFree(dataset);

	/* Get amps and phases*/
	dataset = XLALH5DatasetRead(group, "draws_amp_rel");
	H5DatasetToREAL8VectorArray(dataset, &(out->calib_errors_amplitude), &n_cal_samps, &n);
	XLALH5DatasetFree(dataset);

	dataset = XLALH5DatasetRead(group, "draws_phase");
	H5DatasetToREAL8VectorArray(dataset, &(out->calib_errors_phase), &n_cal_samps, &n);
        XLALH5DatasetFree(dataset);

	// Save the size of the errors arrys
	out->Ncols = Ncols;
	out->Nrows = Nrows;

	// Close the file because done with reading things in
	XLALH5FileClose(group);

	return out;

allocfailed:
	destroy_calib_errors_document(out);
	return NULL;
fopenfailed:
	destroy_calib_errors_document(out);
	fclose(fp);
	return NULL;
}

static int add_calib_errors_to_strain(REAL8TimeSeries *h, struct calib_errors_document* calib_errors_document, int random_draw){

	/*
	 * Fourier transform strain into frequency domain to apply calibration errors
	 */

	REAL8FFTPlan *fwdplan = NULL;
        REAL8FFTPlan *revplan = NULL;

	REAL8 duration = h->data->length*h->deltaT;
	REAL8 df = 1./duration;
	REAL8 f0 = 0.0;
        COMPLEX16FrequencySeries *tilde_h = NULL;
        tilde_h = XLALCreateCOMPLEX16FrequencySeries(NULL, &h->epoch, f0, df, &lalDimensionlessUnit, h->data->length / 2 + 1);

        /* Create a plan for Forward and Reverse FFT */
	fwdplan = XLALCreateForwardREAL8FFTPlan(h->data->length, 0);
        revplan = XLALCreateReverseREAL8FFTPlan(h->data->length, 0);

        if(!tilde_h || !fwdplan || !revplan){
           XLALDestroyCOMPLEX16FrequencySeries(tilde_h);
           XLALDestroyREAL8FFTPlan(fwdplan);
           XLALDestroyCOMPLEX16FrequencySeries(tilde_h);
           XLAL_ERROR(XLAL_EFUNC);
        }

	/* FFT the strain into the freq domain */
        if(XLALREAL8TimeFreqFFT(tilde_h, h, fwdplan)){
           XLALDestroyCOMPLEX16FrequencySeries(tilde_h);
           XLALDestroyREAL8FFTPlan(fwdplan);
           XLALDestroyCOMPLEX16FrequencySeries(tilde_h);
           XLAL_ERROR(XLAL_EFUNC);
	}

	/*
	 * Interpolate the calibration errors to match the waveform
	 */

	// Figure out the frequency spacing and starting/ending frequencies for interpolated vector
	REAL8 calib_errors_f0 = 0.0;
	while(calib_errors_f0 < calib_errors_document->calib_errors_freqs->data[0]){
		calib_errors_f0 += df;
	}
	REAL8 calib_errors_fNyq = calib_errors_f0;
	while(calib_errors_fNyq < (calib_errors_document->calib_errors_freqs->data[(calib_errors_document->Ncols)-1]-df)){
		calib_errors_fNyq += df;
	}
	UINT4 len_after_interp = (calib_errors_fNyq - calib_errors_f0)/df + 1;

	/* Set up vectors to store interpolated values */
	REAL8Vector *interp_calib_errors_freqs;
	REAL8Vector *interp_calib_errors_amplitude;
	REAL8Vector *interp_calib_errors_phase;
	interp_calib_errors_amplitude = XLALCalloc(1, sizeof(REAL8Vector*));
	interp_calib_errors_phase = XLALCalloc(1, sizeof(REAL8Vector*));
	interp_calib_errors_freqs = XLALCalloc(1, sizeof(REAL8Vector*));
	interp_calib_errors_amplitude = XLALCreateREAL8Vector(len_after_interp);
	interp_calib_errors_phase = XLALCreateREAL8Vector(len_after_interp);
	interp_calib_errors_freqs = XLALCreateREAL8Vector(len_after_interp);

	/* Make interpolated calibration frequency array */
	interp_calib_errors_freqs->data[0] = calib_errors_f0;
	for(UINT4 i=1; i < len_after_interp; i++){
		interp_calib_errors_freqs->data[i]=interp_calib_errors_freqs->data[i-1]+df;
	}

	/* Use gsl methods to interpolate */
	gsl_interp_accel *acc_amplitude = gsl_interp_accel_alloc();
	gsl_interp_accel *acc_phase = gsl_interp_accel_alloc();
	gsl_spline *spline_amplitude = gsl_spline_alloc(gsl_interp_cspline, calib_errors_document->Ncols);
	gsl_spline *spline_phase = gsl_spline_alloc(gsl_interp_cspline, calib_errors_document->Ncols);

	gsl_spline_init(spline_amplitude, calib_errors_document->calib_errors_freqs->data, calib_errors_document->calib_errors_amplitude[random_draw]->data, calib_errors_document->Ncols);
	gsl_spline_init(spline_phase, calib_errors_document->calib_errors_freqs->data, calib_errors_document->calib_errors_phase[random_draw]->data, calib_errors_document->Ncols);

	for(UINT4 i = 0; i < len_after_interp; i++)
        {
		interp_calib_errors_amplitude->data[i] = gsl_spline_eval(spline_amplitude, interp_calib_errors_freqs->data[i], acc_amplitude);
		interp_calib_errors_phase->data[i] = gsl_spline_eval(spline_phase, interp_calib_errors_freqs->data[i], acc_phase);
        }

	gsl_spline_free (spline_amplitude);
        gsl_interp_accel_free (acc_amplitude);
	gsl_spline_free (spline_phase);
        gsl_interp_accel_free (acc_phase);

	/*One pad the interpolated cal error so it is the same length as the tilde_h*/

	REAL8Vector *padded_interp_calib_errors_amplitude;
	REAL8Vector *padded_interp_calib_errors_phase;
	padded_interp_calib_errors_amplitude = XLALCalloc(1, sizeof(REAL8Vector*));
	padded_interp_calib_errors_phase = XLALCalloc(1, sizeof(REAL8Vector*));
	padded_interp_calib_errors_amplitude = XLALCreateREAL8Vector(tilde_h->data->length);
	padded_interp_calib_errors_phase = XLALCreateREAL8Vector(tilde_h->data->length);

	UINT4 pad_front_length = (calib_errors_f0 - h->f0)/df;
	for(UINT4 i=0; i < tilde_h->data->length; ++i){
		if (i < pad_front_length){
			padded_interp_calib_errors_amplitude->data[i] = 1.0;
			padded_interp_calib_errors_phase->data[i] = 0.0;
		}
		else if(i >= pad_front_length && i < len_after_interp){
			padded_interp_calib_errors_amplitude->data[i] = interp_calib_errors_amplitude->data[i];
			padded_interp_calib_errors_phase->data[i] = interp_calib_errors_phase->data[i];
		}
		else{
			padded_interp_calib_errors_amplitude->data[i] = 1.0;
			padded_interp_calib_errors_phase->data[i] = 0.0;
		}
	}

	XLALDestroyREAL8Vector(interp_calib_errors_amplitude);
	XLALDestroyREAL8Vector(interp_calib_errors_phase);
	XLALDestroyREAL8Vector(interp_calib_errors_freqs);

	// Next need to fix this part.  I can just use XLALZZVectorMultiply, but first I need to make a COMPLEX16 vecotr out of the padded, interpoalted calibration errors.  Then I need to make a new COMPLEX16 vector that will be the output of the multiplication.  Then I can iFFT this output - or maybe just store the data from the output into tilde_h - not sure if it matters which one I do.
        COMPLEX16Vector *tilde_calib_errors;
	COMPLEX16Vector *tilde_h_with_errors;
	tilde_calib_errors = XLALCalloc(1, sizeof(COMPLEX16Vector*));
	tilde_h_with_errors = XLALCalloc(1, sizeof(COMPLEX16Vector*));
        tilde_calib_errors = XLALCreateCOMPLEX16Vector(tilde_h->data->length);
        tilde_h_with_errors = XLALCreateCOMPLEX16Vector(tilde_h->data->length);
	for(UINT4 i = 0; i < tilde_calib_errors->length; ++i){
		tilde_calib_errors->data[i] = padded_interp_calib_errors_amplitude->data[i]*cexp(I*padded_interp_calib_errors_phase->data[i]);
	}
	XLALDestroyREAL8Vector(padded_interp_calib_errors_amplitude);
	XLALDestroyREAL8Vector(padded_interp_calib_errors_phase);

	XLALZZVectorMultiply(tilde_h_with_errors, tilde_calib_errors, tilde_h->data);
	for(UINT4 i = 0; i < tilde_h->data->length; ++i){
		tilde_h->data->data[i] = tilde_h_with_errors->data[i];
	}
	XLALDestroyCOMPLEX16Vector(tilde_calib_errors);
	XLALDestroyCOMPLEX16Vector(tilde_h_with_errors);

	/* Zero the DC and Nyquist Components */
	tilde_h->data->data[0] = 0.0;
        tilde_h->data->data[tilde_h->data->length - 1] = 0.0;

	/* return to time domain */

	/* Calls on the already created XLAL Freq -> Time routine */
	REAL8TimeSeries *h_new = NULL;
	size_t h_size = h->data->length;
	h_new = XLALCreateREAL8TimeSeries(h->name, &h->epoch, h->f0, h->deltaT, &h->sampleUnits, h_size);
        if(XLALREAL8FreqTimeFFT(h_new, tilde_h, revplan)){
           XLALDestroyREAL8TimeSeries(h_new);
           XLALDestroyREAL8FFTPlan(revplan);
           XLAL_ERROR(XLAL_EFUNC);
	}

	/*Copy the data to h so it gets seen by the rest of the code*/
	for(UINT4 i=0; i<h_size; i++){
		h->data->data[i]=h_new->data->data[i];
	}

	/* Clean up */
	XLALDestroyREAL8FFTPlan(revplan);
	XLALDestroyREAL8FFTPlan(fwdplan);
        XLALDestroyCOMPLEX16FrequencySeries(tilde_h);
	XLALDestroyREAL8TimeSeries(h_new);

	return 0;
}


/*
 * ============================================================================
 *
 *				 XML Input
 *
 * ============================================================================
 */


/*
 * Clean up
 */


static void destroy_injection_document(struct injection_document *doc)
{
	if(doc) {
		XLALDestroySimBurstTable(doc->sim_burst_table_head);
		doc->sim_burst_table_head = NULL;
		XLALDestroyTimeSlideTable(doc->time_slide_table_head);
		doc->time_slide_table_head = NULL;
		while(doc->sim_inspiral_table_head) {
			SimInspiralTable *next = doc->sim_inspiral_table_head->next;
			XLALFree(doc->sim_inspiral_table_head);
			doc->sim_inspiral_table_head = next;
		}
	}
	g_free(doc);
}


/*
 * Load document
 */

static int sim_burst_row_callback(struct ligolw_table *table, struct ligolw_table_row row, void *data)
{
	int result_code;
	SimBurst **head = data;
	SimBurst *new = XLALCreateSimBurst();
	struct ligolw_unpacking_spec spec[] = {
		{&new->process_id, "process_id", ligolw_cell_type_int_8s, LIGOLW_UNPACKING_REQUIRED},
		{NULL, "waveform", ligolw_cell_type_lstring, LIGOLW_UNPACKING_REQUIRED},
		{&new->ra, "ra", ligolw_cell_type_real_8, LIGOLW_UNPACKING_REQUIRED},
		{&new->dec, "dec", ligolw_cell_type_real_8, LIGOLW_UNPACKING_REQUIRED},
		{&new->psi, "psi", ligolw_cell_type_real_8, LIGOLW_UNPACKING_REQUIRED},
		{&new->time_geocent_gps.gpsSeconds, "time_geocent_gps", ligolw_cell_type_int_4s, LIGOLW_UNPACKING_REQUIRED},
		{&new->time_geocent_gps.gpsNanoSeconds, "time_geocent_gps_ns", ligolw_cell_type_int_4s, LIGOLW_UNPACKING_REQUIRED},
		{&new->time_geocent_gmst, "time_geocent_gmst", ligolw_cell_type_real_8, LIGOLW_UNPACKING_REQUIRED},
		{&new->duration, "duration", ligolw_cell_type_real_8, LIGOLW_UNPACKING_REQUIRED},
		{&new->frequency, "frequency", ligolw_cell_type_real_8, LIGOLW_UNPACKING_REQUIRED},
		{&new->bandwidth, "bandwidth", ligolw_cell_type_real_8, LIGOLW_UNPACKING_REQUIRED},
		{&new->q, "q", ligolw_cell_type_real_8, LIGOLW_UNPACKING_REQUIRED},
		{&new->pol_ellipse_angle, "pol_ellipse_angle", ligolw_cell_type_real_8, LIGOLW_UNPACKING_REQUIRED},
		{&new->pol_ellipse_e, "pol_ellipse_e", ligolw_cell_type_real_8, LIGOLW_UNPACKING_REQUIRED},
		{&new->amplitude, "amplitude", ligolw_cell_type_real_8, LIGOLW_UNPACKING_REQUIRED},
		{&new->hrss, "hrss", ligolw_cell_type_real_8, LIGOLW_UNPACKING_REQUIRED},
		{&new->egw_over_rsquared, "egw_over_rsquared", ligolw_cell_type_real_8, LIGOLW_UNPACKING_REQUIRED},
		{&new->waveform_number, "waveform_number", ligolw_cell_type_int_8u, LIGOLW_UNPACKING_REQUIRED},
		{&new->time_slide_id, "time_slide_id", ligolw_cell_type_int_8s, LIGOLW_UNPACKING_REQUIRED},
		{&new->simulation_id, "simulation_id", ligolw_cell_type_int_8s, LIGOLW_UNPACKING_REQUIRED},
		{NULL, NULL, -1, 0}
	};

	/* check for memory allocation failure.  remember to clean up row's
	 * memory. */
	if(!new) {
		XLALPrintError("memory allocation failure\n");
		free(row.cells);
		return -1;
	}

	/* unpack.  have to do the strings manually because they get copied
	 * by value rather than reference.  ligolw_unpacking_row_builder()
	 * cleans up row's memory for us. */
	strncpy(new->waveform, ligolw_row_get_cell(row, "waveform").as_string, LIGOMETA_WAVEFORM_MAX - 1);
	new->waveform[LIGOMETA_WAVEFORM_MAX - 1] = '\0';

	result_code = ligolw_unpacking_row_builder(table, row, spec);
	if(result_code > 0) {
		/* missing required column */
		XLALPrintError("failure parsing row: missing column \"%s\"\n", spec[result_code - 1].name);
		free(new);
		return -1;
	} else if(result_code < 0) {
		/* column type mismatch */
		XLALPrintError("failure parsing row: incorrect type for column \"%s\"\n", spec[-result_code - 1].name);
		free(new);
		return -1;
	}

	/* add new sim to head of linked list.  yes, this means the table's
	 * rows get reversed.  so what. */
	new->next = *head;
	*head = new;

	/* success */
	return 0;
}


static int time_slide_row_callback(struct ligolw_table *table, struct ligolw_table_row row, void *data)
{
	int result_code;
	TimeSlide **head = data;
	TimeSlide *new = XLALCreateTimeSlide();
	struct ligolw_unpacking_spec spec[] = {
		{&new->process_id, "process_id", ligolw_cell_type_int_8s, LIGOLW_UNPACKING_REQUIRED},
		{&new->time_slide_id, "time_slide_id", ligolw_cell_type_int_8s, LIGOLW_UNPACKING_REQUIRED},
		{NULL, "instrument", ligolw_cell_type_lstring, LIGOLW_UNPACKING_REQUIRED},
		{&new->offset, "offset", ligolw_cell_type_real_8, LIGOLW_UNPACKING_REQUIRED},
		{NULL, NULL, -1, 0}
	};

	/* check for memory allocation failure.  remember to clean up row's
	 * memory. */
	if(!new) {
		XLALPrintError("memory allocation failure\n");
		free(row.cells);
		return -1;
	}

	/* unpack.  have to do the strings manually because they get copied
	 * by value rather than reference.  ligolw_unpacking_row_builder()
	 * cleans up row's memory for us. */
	strncpy(new->instrument, ligolw_row_get_cell(row, "instrument").as_string, LIGOMETA_STRING_MAX - 1);
	new->instrument[LIGOMETA_WAVEFORM_MAX - 1] = '\0';

	result_code = ligolw_unpacking_row_builder(table, row, spec);
	if(result_code > 0) {
		/* missing required column */
		XLALPrintError("failure parsing row: missing column \"%s\"\n", spec[result_code - 1].name);
		free(new);
		return -1;
	} else if(result_code < 0) {
		/* column type mismatch */
		XLALPrintError("failure parsing row: incorrect type for column \"%s\"\n", spec[-result_code - 1].name);
		free(new);
		return -1;
	}

	/* add new sim to head of linked list.  yes, this means the table's
	 * rows get reversed.  so what. */
	new->next = *head;
	*head = new;

	/* success */
	return 0;
}


static int sim_inspiral_row_callback(struct ligolw_table *table, struct ligolw_table_row row, void *data)
{
	int result_code;
	SimInspiralTable **head = data;
	SimInspiralTable *new = LALCalloc(1, sizeof(*new));	/* ugh, lal */
	struct ligolw_unpacking_spec spec[] = {
		{&new->process_id, "process_id", ligolw_cell_type_int_8s, LIGOLW_UNPACKING_REQUIRED},
		{NULL, "waveform", ligolw_cell_type_lstring, LIGOLW_UNPACKING_REQUIRED},
		{&new->geocent_end_time.gpsSeconds, "geocent_end_time", ligolw_cell_type_int_4s, LIGOLW_UNPACKING_REQUIRED},
		{&new->geocent_end_time.gpsNanoSeconds, "geocent_end_time_ns", ligolw_cell_type_int_4s, LIGOLW_UNPACKING_REQUIRED},
		/* don't load detector end times:  they're stupid */
		{NULL, "source", ligolw_cell_type_lstring, LIGOLW_UNPACKING_REQUIRED},
		{&new->mass1, "mass1", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->mass2, "mass2", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->mchirp, "mchirp", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->eta, "eta", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->distance, "distance", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->longitude, "longitude", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->latitude, "latitude", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->inclination, "inclination", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->coa_phase, "coa_phase", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->polarization, "polarization", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->psi0, "psi0", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->psi3, "psi3", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->alpha, "alpha", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->alpha1, "alpha1", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->alpha2, "alpha2", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->alpha3, "alpha3", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->alpha4, "alpha4", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->alpha5, "alpha5", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->alpha6, "alpha6", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->beta, "beta", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->spin1x, "spin1x", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->spin1y, "spin1y", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->spin1z, "spin1z", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->spin2x, "spin2x", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->spin2y, "spin2y", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->spin2z, "spin2z", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->theta0, "theta0", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->phi0, "phi0", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->f_lower, "f_lower", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->f_final, "f_final", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		/* don't load effective distances:  they're stupid */
		{&new->numrel_mode_min, "numrel_mode_min", ligolw_cell_type_int_4s, LIGOLW_UNPACKING_REQUIRED},
		{&new->numrel_mode_max, "numrel_mode_max", ligolw_cell_type_int_4s, LIGOLW_UNPACKING_REQUIRED},
		{NULL, "numrel_data", ligolw_cell_type_lstring, LIGOLW_UNPACKING_REQUIRED},
		{&new->amp_order, "amp_order", ligolw_cell_type_int_4s, LIGOLW_UNPACKING_REQUIRED},
		{NULL, "taper", ligolw_cell_type_lstring, LIGOLW_UNPACKING_REQUIRED},
		{&new->bandpass, "bandpass", ligolw_cell_type_int_4s, LIGOLW_UNPACKING_REQUIRED},
		{&new->simulation_id, "simulation_id", ligolw_cell_type_int_8s, LIGOLW_UNPACKING_REQUIRED},
		{NULL, NULL, -1, 0}
	};

	/* check for memory allocation failure.  remember to clean up row's
	 * memory. */
	if(!new) {
		XLALPrintError("memory allocation failure\n");
		free(row.cells);
		return -1;
	}

	/* unpack.  have to do the strings manually because they get copied
	 * by value rather than reference.  ligolw_unpacking_row_builder()
	 * cleans up row's memory for us. */
	strncpy(new->waveform, ligolw_row_get_cell(row, "waveform").as_string, LIGOMETA_WAVEFORM_MAX - 1);
	new->waveform[LIGOMETA_WAVEFORM_MAX - 1] = '\0';
	strncpy(new->source, ligolw_row_get_cell(row, "source").as_string, LIGOMETA_SOURCE_MAX - 1);
	new->source[LIGOMETA_SOURCE_MAX - 1] = '\0';
	strncpy(new->numrel_data, ligolw_row_get_cell(row, "numrel_data").as_string, LIGOMETA_STRING_MAX - 1);
	new->numrel_data[LIGOMETA_STRING_MAX - 1] = '\0';
	strncpy(new->taper, ligolw_row_get_cell(row, "taper").as_string, LIGOMETA_INSPIRALTAPER_MAX - 1);
	new->taper[LIGOMETA_INSPIRALTAPER_MAX - 1] = '\0';

	result_code = ligolw_unpacking_row_builder(table, row, spec);
	if(result_code > 0) {
		/* missing required column */
		XLALPrintError("failure parsing row: missing column \"%s\"\n", spec[result_code - 1].name);
		LALFree(new);
		return -1;
	} else if(result_code < 0) {
		/* column type mismatch */
		XLALPrintError("failure parsing row: incorrect type for column \"%s\"\n", spec[-result_code - 1].name);
		LALFree(new);
		return -1;
	}

	/* add new sim to head of linked list.  yes, this means the table's
	 * rows get reversed.  so what. */
	new->next = *head;
	*head = new;

	/* success */
	return 0;
}


static struct injection_document *load_injection_document(const char *filename, LIGOTimeGPS start, LIGOTimeGPS end, double longest_injection)
{
	ezxml_t xmldoc;
	ezxml_t elem;
	struct ligolw_table *table;
	struct injection_document *new;

	g_assert(filename != NULL);

	/*
	 * allocate the document
	 */

	new = g_new0(struct injection_document, 1);
	if(!new) {
		XLALPrintError("%s(): malloc() failed\n", __func__);
		goto allocfailed;
	}

	/*
	 * adjust start and end times
	 */

	XLALGPSAdd(&start, -longest_injection);
	XLALGPSAdd(&end, longest_injection);

	/* parse the document */
	xmldoc = ezxml_parse_file(filename);
	if(!xmldoc) {
		XLALPrintError("%s(): error parsing \"%s\"\n", __func__, filename);
		goto parsefailed;
	}

	/*
	 * load optional (sim_burst + time_slide) tables
	 */

	elem = ligolw_table_get(xmldoc, "sim_burst");
	if(elem) {
		table = ligolw_table_parse(elem, sim_burst_row_callback, &new->sim_burst_table_head);
		if(!table) {
			XLALPrintError("%s(): failure parsing sim_burst table in \"%s\"\n", __func__, filename);
			goto simburstfailed;
		}
		ligolw_table_free(table);
		new->has_sim_burst_table = 1;
	} else {
		new->has_sim_burst_table = 0;
		new->sim_burst_table_head = NULL;
	}

	elem = ligolw_table_get(xmldoc, "time_slide");
	if(elem) {
		table = ligolw_table_parse(elem, time_slide_row_callback, &new->time_slide_table_head);
		if(!table) {
			XLALPrintError("%s(): failure parsing time_slide table in \"%s\"\n", __func__, filename);
			goto timeslidefailed;
		}
		ligolw_table_free(table);
		new->has_time_slide_table = 1;
	} else if(new->has_sim_burst_table) {
		/* document is required to have a time_slide table if it
		 * has a sim_burst table */
		XLALPrintError("%s(): sim_burst table requires time_slide table in \"%s\"\n", __func__, filename);
		goto timeslidefailed;
	} else {
		new->has_time_slide_table = 0;
		new->time_slide_table_head = NULL;
	}

	/*
	 * load optional sim_inspiral table.  subsequent code requires it
	 * to be ordered by geocenter end time.
	 */

	elem = ligolw_table_get(xmldoc, "sim_inspiral");
	if(elem) {
		table = ligolw_table_parse(elem, sim_inspiral_row_callback, &new->sim_inspiral_table_head);
		if(!table) {
			XLALPrintError("%s(): failure parsing sim_inspiral table in \"%s\"\n", __func__, filename);
			goto siminspiralfailed;
		}
		ligolw_table_free(table);
		new->has_sim_inspiral_table = 1;
		XLALSortSimInspiral(&new->sim_inspiral_table_head, XLALCompareSimInspiralByGeocentEndTime);
	} else {
		new->has_sim_inspiral_table = 0;
		new->sim_inspiral_table_head = NULL;
	}

	/*
	 * clean up
	 */

	ezxml_free(xmldoc);

	/*
	 * success
	 */

	return new;

	/*
	 * error
	 */

siminspiralfailed:
timeslidefailed:
simburstfailed:
	ezxml_free(xmldoc);
parsefailed:
allocfailed:
	destroy_injection_document(new);
	return NULL;
}


/*
 * Create detector strain from sim_inspiral
 */


static int sim_inspiral_strain(REAL8TimeSeries **strain, SimInspiralTable *sim_inspiral, double deltaT, LALDetector detector, struct calib_errors_document **calib_errors_document)
{
	REAL8TimeSeries *hplus = NULL;
	REAL8TimeSeries *hcross = NULL;
	/*
	 * create waveform using lalinspiral's wrapping of lalsimulation.
	 * XLALInspiralTDWaveformFromSimInspiral() translates the
	 * sim_inspiral row into a function call into lalsimulation,
	 * collecting and returning a conditioned waveform suitable for
	 * injection into an h(t) stream (after projection onto an antenna
	 * respose).
	 */

	GST_CAT_INFO(GST_CAT_DEFAULT, "Generating injection with parameters: m1=%e m2=%e spin1x=%e spin1y=%e spin1z=%e spin2x=%e spin2y=%e spin2z=%e geocent_end_time=%d geocent_end_time_ns=%d waveform=%s", sim_inspiral->mass1, sim_inspiral->mass2, sim_inspiral->spin1x, sim_inspiral->spin1y, sim_inspiral->spin1z, sim_inspiral->spin2x, sim_inspiral->spin2y, sim_inspiral->spin2z, sim_inspiral->geocent_end_time.gpsSeconds, sim_inspiral->geocent_end_time.gpsNanoSeconds, sim_inspiral->waveform);
	if(XLALInspiralTDWaveformFromSimInspiral(&hplus, &hcross, sim_inspiral, deltaT) == XLAL_FAILURE)
		XLAL_ERROR(XLAL_EFUNC);


	/* add the time of the injection at the geocentre to the
	 * start times of the h+ and hx time series.  after this,
	 * their epochs mark the start of those time series at the
	 * geocentre. */

	XLALGPSAddGPS(&hcross->epoch, &sim_inspiral->geocent_end_time);
	XLALGPSAddGPS(&hplus->epoch, &sim_inspiral->geocent_end_time);

	/*
	 * project waveform onto detector
	 */

	*strain = XLALSimDetectorStrainREAL8TimeSeries(hplus, hcross, sim_inspiral->longitude, sim_inspiral->latitude, sim_inspiral->polarization, &detector);
	XLALDestroyREAL8TimeSeries(hplus);
	XLALDestroyREAL8TimeSeries(hcross);
	if(!(*strain))
		XLAL_ERROR(XLAL_EFUNC);

	if(*calib_errors_document != NULL){
		printf("Applying calibration errors\n");
		//Use the current geocent_end_time to seed a random number generator to select a random calibration uncertainty draw
		srand((UINT4)sim_inspiral->geocent_end_time.gpsSeconds);
		UINT4 random_draw = (rand()%((*calib_errors_document)->Nrows-1));
		if(add_calib_errors_to_strain(*strain, *calib_errors_document, random_draw) < 0) {
			XLAL_ERROR(XLAL_EFUNC);
		}
	}
	else{
		printf("Calibraiton errors not applied\n");
	}

	return XLAL_SUCCESS;
}


static int update_simulation_series(REAL8TimeSeries *h, GSTLALSimulation *element, const COMPLEX16FrequencySeries *response)
{
	double injTime, DeltaT, startMinusEnd, endMinusStart;
	REAL8 tmpREAL8;
	LIGOTimeGPS hStartTime, hEndTime, injStartTime, injEndTime;
	SimInspiralTable *thisSimInspiral;
	SimInspiralTable *prevSimInspiral = NULL;
	/* skip inspiral injections whose estimated geocentre start times are more than this many
	 * seconds outside of the target time series */
	const double injection_window = 100.0;

	/* to be deduced from the h's channel name */
	const LALDetector *detector;

	/*
	 * get detector information
	 */

	detector = XLALDetectorPrefixToLALDetector(h->name);
	if(!detector)
		XLAL_ERROR(XLAL_EFUNC);

	/*
	 * calculate h segment boundaries, h = [hStartTime, hEndTime)
	 */

	hStartTime = h->epoch;
	hEndTime = hStartTime;
	XLALGPSAdd(&hEndTime, h->data->length * h->deltaT);

	/*
	 * initialize simulation_series if NULL
	 */

	//printf("PSD \n");
	if(element->simulation_series == NULL) {
		size_t series_length = 0;
		LIGOTimeGPS startEpoch = hStartTime;
		XLALGPSAdd(&startEpoch, -injection_window);
		tmpREAL8 = XLALGPSDiff(&hStartTime, &startEpoch) / h->deltaT;
		tmpREAL8 -= floor(tmpREAL8);
		XLALGPSAdd(&startEpoch, -tmpREAL8);
		series_length += XLALGPSDiff(&hEndTime, &startEpoch) / h->deltaT;
		series_length += ceil(injection_window / h->deltaT);
		element->simulation_series = XLALCreateREAL8TimeSeries(h->name, &startEpoch, h->f0, h->deltaT, &h->sampleUnits, series_length);
		memset(element->simulation_series->data->data, 0, element->simulation_series->data->length * sizeof(*element->simulation_series->data->data));
	}

	/*
	 * resize simulation_series to cover this time
	 */

	DeltaT = XLALGPSDiff(&element->simulation_series->epoch, &hEndTime);
	DeltaT += element->simulation_series->deltaT * element->simulation_series->data->length;
	if(DeltaT < injection_window)
		element->simulation_series = XLALResizeREAL8TimeSeries(element->simulation_series, 0, element->simulation_series->data->length + (int) ceil((injection_window - DeltaT) / element->simulation_series->deltaT));
	if(!element->simulation_series)
		XLAL_ERROR(XLAL_EFUNC);

	/*
	 * loop over injections in file
	 */

	for(thisSimInspiral = element->injection_document->sim_inspiral_table_head; thisSimInspiral; ) {
		REAL8TimeSeries *inspiral_series = NULL;

		/*
		 * calculate start and end times for this series containing
		 * this injection.  NOTE:  this just uses the 0 PN formula
		 * and pads it to be safe
		 */

		/* the padding will no longer be sufficient at this mass */
		g_assert((thisSimInspiral->mass1 + thisSimInspiral->mass2) < 1e4);

		injTime = 1.0 + XLALSimInspiralTaylorF2ReducedSpinChirpTime(thisSimInspiral->f_lower, thisSimInspiral->mass1 * LAL_MSUN_SI, thisSimInspiral->mass2 * LAL_MSUN_SI, 0, 0);
		injStartTime = injEndTime = thisSimInspiral->geocent_end_time;
		XLALGPSAdd(&injStartTime, -1.9*injTime - 1.0);
		XLALGPSAdd(&injEndTime, 0.1*injTime + 1.0);

		/*
		 * check whether injection segment intersects h
		 */

		startMinusEnd = XLALGPSDiff(&injStartTime, &hEndTime);
		endMinusStart = XLALGPSDiff(&injEndTime, &hStartTime);

		if(endMinusStart < -injection_window) /* injection ends before h */ {
			if(prevSimInspiral)
				prevSimInspiral->next = thisSimInspiral->next;
			if(thisSimInspiral == element->injection_document->sim_inspiral_table_head)
				element->injection_document->sim_inspiral_table_head = thisSimInspiral->next;
			SimInspiralTable *tmpSimInspiral = thisSimInspiral;
			thisSimInspiral = thisSimInspiral->next;
			XLALFree(tmpSimInspiral);
			continue;
		}

		if(startMinusEnd > injection_window) /* injection starts after h */ {
			prevSimInspiral = thisSimInspiral;
			thisSimInspiral = thisSimInspiral->next;
			continue;
		}

		/*
		 * compute injection waveform
		 */
		if(sim_inspiral_strain(&inspiral_series, thisSimInspiral, h->deltaT, *detector, &(element->calib_errors_document))){
			XLAL_ERROR(XLAL_EFUNC);}

		/*
		 * resize simulation_series to cover this time
		 */

		DeltaT = XLALGPSDiff(&inspiral_series->epoch, &element->simulation_series->epoch);
		DeltaT += inspiral_series->data->length * inspiral_series->deltaT + injection_window;
		DeltaT -= element->simulation_series->data->length * element->simulation_series->deltaT;
		if(DeltaT > 0.)
			element->simulation_series = XLALResizeREAL8TimeSeries(element->simulation_series, 0, element->simulation_series->data->length + ceil(DeltaT / element->simulation_series->deltaT));
		if(!element->simulation_series) {
			XLALDestroyREAL8TimeSeries(inspiral_series);
			XLAL_ERROR(XLAL_EFUNC);
		}

		/*
		 * add detector strain to simulation_series
		 */

		if(XLALSimAddInjectionREAL8TimeSeries(element->simulation_series, inspiral_series, response)) {
			XLALDestroyREAL8TimeSeries(inspiral_series);
			XLAL_ERROR(XLAL_EFUNC);
		}
		XLALDestroyREAL8TimeSeries(inspiral_series);

		/*
		 * remove injection from list and continue
		 */

		if(prevSimInspiral)
			prevSimInspiral->next = thisSimInspiral->next;
		{
		SimInspiralTable *tmpSimInspiral = thisSimInspiral;
		thisSimInspiral = thisSimInspiral->next;
		if (tmpSimInspiral == element->injection_document->sim_inspiral_table_head)
			element->injection_document->sim_inspiral_table_head = thisSimInspiral;
		XLALFree(tmpSimInspiral);
		}
	}

	/*
	 * add burst injections to simulation_series
	 */

	if(element->injection_document->sim_burst_table_head) {
		/*
		 * create a buffer to store burst injections.  we put the
		 * injections into this (zeroed) buffer and then from there
		 * into h(t) so that we don't inject the same burst
		 * injection into the data more than once (these
		 * intermediate buffers are disjoint).
		 * XLALBurstInjectSignals() will skip injections too far
		 * outside of the boundaries of its target series, so in
		 * this way we also control which injections are generated
		 * in each iteration.
		 */

		REAL8TimeSeries *burst_series = XLALCreateREAL8TimeSeries(h->name, &h->epoch, h->f0, h->deltaT, &h->sampleUnits, h->data->length);
		if(!burst_series)
			XLAL_ERROR(XLAL_EFUNC);
		memset(burst_series->data->data, 0, burst_series->data->length * sizeof(*burst_series->data->data));

		/*
		 * inject waveforms into that buffer
		 */

		if(XLALBurstInjectSignals(burst_series, element->injection_document->sim_burst_table_head, element->injection_document->time_slide_table_head, response))
			XLAL_ERROR(XLAL_EFUNC);

		/*
		 * add waveforms buffer into simulation_series
		 */

		if(!XLALAddREAL8TimeSeries(element->simulation_series, burst_series)) {
			XLALDestroyREAL8TimeSeries(burst_series);
			XLAL_ERROR(XLAL_EFUNC);
		}
		XLALDestroyREAL8TimeSeries(burst_series);		
	}

	return 0;
}


/*
 * This is the function that gets called.
 */


static int add_simulation_series(REAL8TimeSeries *h, const GSTLALSimulation *element, const COMPLEX16FrequencySeries *response)
{
	size_t startIdx;

	/*
	 * add simulation_series to h
	 */

	if(!XLALAddREAL8TimeSeries(h, element->simulation_series))
		XLAL_ERROR(XLAL_EFUNC);

	/*
	 * shrink simulation_series removing anything before the end of h
	 */

	startIdx = (size_t) (XLALGPSDiff(&h->epoch, &element->simulation_series->epoch) / h->deltaT);
	startIdx += h->data->length;
	if(!XLALShrinkREAL8TimeSeries(element->simulation_series, startIdx, element->simulation_series->data->length - startIdx))
		XLAL_ERROR(XLAL_EFUNC);

	/*
	 * done
	 */

	return 0;
}


/*
 * ============================================================================
 *
 *                         GstBaseTransform Overrides
 *
 * ============================================================================
 */


/* FIXME:  or maybe as a source element, and let the adder do the mixing work */


/*
 * sink_event()
 */


static gboolean sink_event(GstBaseTransform *trans, GstEvent *event)
{
	GSTLALSimulation *element = GSTLAL_SIMULATION(trans);

	/*
	 * extract metadata from tags
	 */

	switch(GST_EVENT_TYPE(event)) {
	case GST_EVENT_TAG: {
		GstTagList *taglist;
		gchar *instrument = NULL, *channel_name = NULL, *units = NULL;

		/*
		 * attempt to extract all 3 tags from the event's taglist
		 */

		gst_event_parse_tag(event, &taglist);
		gst_tag_list_get_string(taglist, GSTLAL_TAG_INSTRUMENT, &instrument);
		gst_tag_list_get_string(taglist, GSTLAL_TAG_CHANNEL_NAME, &channel_name);
		gst_tag_list_get_string(taglist, GSTLAL_TAG_UNITS, &units);

		if(instrument || channel_name || units) {
			/*
			 * if any of the 3 tags were provided discard
			 * all current values
			 */

			g_free(element->instrument);
			element->instrument = NULL;
			g_free(element->channel_name);
			element->channel_name = NULL;
			g_free(element->units);
			element->units = NULL;

			if(instrument && channel_name && units) {
				/*
				 * if all 3 tags were provided, save as the
				 * new values
				 */

				element->instrument = instrument;
				element->channel_name = channel_name;
				element->units = units;
			}

			/*
			 * do notifies
			 */

			g_object_notify(G_OBJECT(element), "instrument");
			g_object_notify(G_OBJECT(element), "channel-name");
			g_object_notify(G_OBJECT(element), "units");
		}
		break;
	}

	default:
		break;
	}

	return GST_BASE_TRANSFORM_CLASS(gstlal_simulation_parent_class)->sink_event(trans, event);
}


/*
 * chain()
 */


static GstFlowReturn transform_ip(GstBaseTransform *trans, GstBuffer *buf)
{
	GSTLALSimulation *element = GSTLAL_SIMULATION(trans);
	GstFlowReturn result = GST_FLOW_OK;
	GstMapInfo info;
	REAL8TimeSeries *h;

	/*
	 * If no injection list, reduce to pass-through
	 */

	if(!element->xml_location)
		goto done;

	/*
	 * Load injections if needed
	 */

	if(!element->injection_document) {
		LIGOTimeGPS start = {0, 0};
		LIGOTimeGPS end = {+2000000000, 0};
		/* earliest and latest possible LIGOTimeGPS */
		/* FIXME:  hard-coded = BAD BAD BAD */
		/*XLALINT8NSToGPS(&start, (INT8) 1 << 63);
		XLALINT8NSToGPS(&end, ((INT8) 1 << 63) - 1);*/
		element->injection_document = load_injection_document(element->xml_location, start, end, 0.0);
		if(!element->injection_document) {
			GST_ELEMENT_ERROR(element, RESOURCE, READ, (NULL), ("error loading \"%s\"", element->xml_location));
			result = GST_FLOW_ERROR;
			goto done;
		}
	}

	/*
	 * Load calibration errors document if needed
	 */
	if(element->calib_errors_location){
		if(!element->calib_errors_document){
			printf("Calibration errors document is being loaded.\n");
			element->calib_errors_document = load_calib_errors_document(element->calib_errors_location, element->instrument);
			if(!element->calib_errors_document) {
				GST_ELEMENT_ERROR(element, RESOURCE, READ, (NULL), ("error loading \"%s\"", element->calib_errors_location));
				result = GST_FLOW_ERROR;
				goto done;
			}
			printf("Calibraiton errors document successfully loaded.\n");
		}
	}
	else{
		element->calib_errors_document = NULL;
	}


	/*
	 * Make sure stream tags are sufficient
	 */

	if(!element->instrument || !element->channel_name || !element->units) {
		GST_ELEMENT_ERROR(element, STREAM, FORMAT, (NULL), ("stream metadata not available:  must receive tags \"%s\", \"%s\", \"%s\"", GSTLAL_TAG_INSTRUMENT, GSTLAL_TAG_CHANNEL_NAME, GSTLAL_TAG_UNITS));
		result = GST_FLOW_ERROR;
		goto done;
	}

	/*
	 * Wrap buffer in a LAL REAL8TimeSeries.
	 * gstlal_buffer_map_REAL8TimeSeries() consumes the reference
	 * returned by gst_pad_get_current_caps().
	 */

	h = gstlal_buffer_map_REAL8TimeSeries(buf, gst_pad_get_current_caps(GST_BASE_TRANSFORM_SINK_PAD(trans)), &info, element->instrument, element->channel_name, element->units);
	if(!h) {
		GST_ELEMENT_ERROR(element, LIBRARY, FAILED, (NULL), ("failure wrapping buffer in REAL8TimeSeries"));
		result = GST_FLOW_ERROR;
		goto done;
	}

	/*
	 * Update simulation_series
	 * FIXME: needs to update simulation_series when new injection file provided
	 */

	/* FIXME: do we need a series mutex lock here? */

	if(update_simulation_series(h, element, NULL) < 0) {
		GST_ELEMENT_ERROR(element, LIBRARY, FAILED, (NULL), ("failure updating simulation_series"));
		result = GST_FLOW_ERROR;
		goto release_h;
	}

	/*
	 * Add injections
	 */

	if(add_simulation_series(h, element, NULL) < 0) {
		GST_ELEMENT_ERROR(element, LIBRARY, FAILED, (NULL), ("failure performing injections"));
		result = GST_FLOW_ERROR;
		goto release_h;
	}

	/* FIXME: do we need a series mutex lock free here? */

	/*
	 * Free the wrapping.  Setting the data pointer to NULL prevents
	 * XLALDestroyREAL8TimeSeries() from free()ing the buffer's data.
	 */

release_h:
	gstlal_buffer_unmap_REAL8TimeSeries(buf, &info, h);

	/*
	 * Done
	 */

done:
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
	ARG_XML_LOCATION = 1,
	ARG_INSTRUMENT,
	ARG_CHANNEL_NAME,
	ARG_UNITS,
	ARG_CALIB_ERRORS_LOCATION
};


static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec)
{

	GSTLALSimulation *element = GSTLAL_SIMULATION(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_XML_LOCATION:
		g_free(element->xml_location);
		element->xml_location = g_value_dup_string(value);
		destroy_injection_document(element->injection_document);
		element->injection_document = NULL;
		break;

	case ARG_CALIB_ERRORS_LOCATION:
		g_free(element->calib_errors_location);
		element->calib_errors_location = g_value_dup_string(value);
		destroy_calib_errors_document(element->calib_errors_document);
		element->calib_errors_document = NULL;
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *pspec)
{
	GSTLALSimulation *element = GSTLAL_SIMULATION(object);

	GST_OBJECT_LOCK(element);

	switch(id) {
	case ARG_XML_LOCATION:
		g_value_set_string(value, element->xml_location);
		break;

	case ARG_INSTRUMENT:
		g_value_set_string(value, element->instrument);
		break;

	case ARG_CHANNEL_NAME:
		g_value_set_string(value, element->channel_name);
		break;

	case ARG_UNITS:
		g_value_set_string(value, element->units);
		break;

	case ARG_CALIB_ERRORS_LOCATION:
		g_value_set_string(value, element->calib_errors_location);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * finalize()
 */


static void finalize(GObject * object)
{
	GSTLALSimulation *element = GSTLAL_SIMULATION(object);

	g_free(element->xml_location);
	element->xml_location = NULL;
	destroy_injection_document(element->injection_document);
	element->injection_document = NULL;
	g_free(element->instrument);
	element->instrument = NULL;
	g_free(element->channel_name);
	element->channel_name = NULL;
	g_free(element->units);
	element->units = NULL;
	g_free(element->calib_errors_location);
	element->calib_errors_location = NULL;
	destroy_calib_errors_document(element->calib_errors_document);
	element->calib_errors_document = NULL;
	XLALDestroyREAL8TimeSeries(element->simulation_series);
	element->simulation_series = NULL;

	G_OBJECT_CLASS(gstlal_simulation_parent_class)->finalize(object);
}


/*
 * class_init()
 */


#define CAPS \
	"audio/x-raw, " \
	"rate = " GST_AUDIO_RATE_RANGE ", " \
	"channels = (int) 1, " \
	"format = (string) " GST_AUDIO_NE(F64) ", " \
	"layout = (string) interleaved, " \
	"channel-mask = (bitmask) 0"


static void gstlal_simulation_class_init(GSTLALSimulationClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(klass);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	transform_class->sink_event = GST_DEBUG_FUNCPTR(sink_event);
	transform_class->transform_ip = GST_DEBUG_FUNCPTR(transform_ip);

	gst_element_class_set_details_simple(
		element_class,
		"Simulation",
		"Filter",
		"An injection routine calling lalsimulation waveform generators",
		"Kipp Cannon <kipp.cannon@ligo.org>, Chad Hanna <channa@ligo.caltech.edu>, Drew Keppel <drew.keppel@ligo.org>"
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

	g_object_class_install_property(
		gobject_class,
		ARG_XML_LOCATION,
		g_param_spec_string(
			"xml-location",
			"XML Location",
			"Name of LIGO Light Weight XML file containing list(s) of software injections",
			NULL,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_INSTRUMENT,
		g_param_spec_string(
			"instrument",
			"Instrument",
			"Name of instrument for which the injections are being simulated",
			NULL,
			G_PARAM_READABLE | G_PARAM_STATIC_STRINGS
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_CHANNEL_NAME,
		g_param_spec_string(
			"channel-name",
			"Channel name",
			"Name of the channel for which the injections are being simulated",
			NULL,
			G_PARAM_READABLE | G_PARAM_STATIC_STRINGS
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_UNITS,
		g_param_spec_string(
			"units",
			"Units",
			"Units in which the injections are being computed.  Units are a string in the format used by the LAL units package.",
			NULL,
			G_PARAM_READABLE | G_PARAM_STATIC_STRINGS
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_CALIB_ERRORS_LOCATION,
		g_param_spec_string(
			"calib-errors-location",
			"Calibration Errors Location",
			"File containing list of HDF5 file location for documents containing calibration errors to be applied to software injections",
			NULL,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);
}


/*
 * instance init
 */


static void gstlal_simulation_init(GSTLALSimulation *element)
{
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(element), TRUE);

	element->xml_location = NULL;
	element->injection_document = NULL;
	element->instrument = NULL;
	element->channel_name = NULL;
	element->units = NULL;
	element->calib_errors_location = NULL;
	element->calib_errors_document = NULL;
	element->simulation_series = NULL;
}

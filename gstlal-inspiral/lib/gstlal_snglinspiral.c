/*
 * Copyright (C) 2012,2013,2015,2016  Chad Hanna
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation; either version 2 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */


#include <glib.h>
#include <glib-object.h>
#include <gst/gst.h>
#include <gstlal/ezligolw.h>
#include <gstlal/gstlal_peakfinder.h>
#include <complex.h>
#include <string.h>
#include <math.h>
#include <lal/Date.h>
#include <lal/LIGOMetadataTables.h>
#include <lal/LALStdlib.h>
#include <snglinspiralrowtype.h>
#include <gsl/gsl_matrix_float.h>
#include <gsl/gsl_blas.h>

/**
 * SECTION:gstlal_snglinspiral.c
 * @short_description:  Utilities for sngl inspiral 
 * 
 * Reviewed: 38c65535fc96d6cc3dee76c2de9d3b76b47d5283 2015-05-14 
 * K. Cannon, J. Creighton, C. Hanna, F. Robinett 
 * 
 * Actions:
 * 67,79: outside of loop
 * 144: add bankarray end time here to take into account the IMR waveform shifts
 * 152: figure out how to use a more accurate sigmasq calculation
 *
 * Complete Actions:
 *  56: free() should be LALFree()
 */

double gstlal_eta(double m1, double m2)
{
	return m1 * m2 / pow(m1 + m2, 2);
}


double gstlal_mchirp(double m1, double m2)
{
	return pow(m1 * m2, 0.6) / pow(m1 + m2, 0.2);
}


double gstlal_effective_distance(double snr, double sigmasq)
{
	return sqrt(sigmasq) / snr;
}


static int sngl_inspiral_row_callback(struct ligolw_table *table, struct ligolw_table_row row, void *data)
{
	int result_code;
	SnglInspiralTable **head = data;
	SnglInspiralTable *new = LALCalloc(1, sizeof(*new));
	struct ligolw_unpacking_spec spec[] = {
		{&new->process_id, "process_id", ligolw_cell_type_int_8s, LIGOLW_UNPACKING_REQUIRED},
		{&new->event_id, "event_id", ligolw_cell_type_int_8s, LIGOLW_UNPACKING_REQUIRED},
		{&new->mass1, "mass1", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->mass2, "mass2", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->mtotal, "mtotal", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->mchirp, "mchirp", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->eta, "eta", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->spin1x, "spin1x", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->spin1y, "spin1y", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->spin1z, "spin1z", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->spin2x, "spin2x", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->spin2y, "spin2y", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->spin2z, "spin2z", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->chi, "chi", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->f_final, "f_final", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->template_duration, "template_duration", ligolw_cell_type_real_8, LIGOLW_UNPACKING_REQUIRED},
		{&new->ttotal, "ttotal", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{NULL, "search", ligolw_cell_type_lstring, LIGOLW_UNPACKING_REQUIRED},
		{NULL, "ifo", ligolw_cell_type_lstring, LIGOLW_UNPACKING_REQUIRED},
		{NULL, "channel", ligolw_cell_type_lstring, LIGOLW_UNPACKING_REQUIRED},
		{&new->sigmasq, "sigmasq", ligolw_cell_type_real_8, LIGOLW_UNPACKING_REQUIRED},
		{&new->snr, "snr", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->coa_phase, "coa_phase", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->eff_distance, "eff_distance", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->amplitude, "amplitude", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->end.gpsSeconds, "end_time", ligolw_cell_type_int_4s, LIGOLW_UNPACKING_REQUIRED},
		{&new->end.gpsNanoSeconds, "end_time_ns", ligolw_cell_type_int_4s, LIGOLW_UNPACKING_REQUIRED},
		{&new->end_time_gmst, "end_time_gmst", ligolw_cell_type_real_8, LIGOLW_UNPACKING_REQUIRED},
		{&new->impulse_time.gpsSeconds, "impulse_time", ligolw_cell_type_int_4s, LIGOLW_UNPACKING_REQUIRED},
		{&new->impulse_time.gpsNanoSeconds, "impulse_time_ns", ligolw_cell_type_int_4s, LIGOLW_UNPACKING_REQUIRED},
		{&new->bank_chisq, "bank_chisq", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->bank_chisq_dof, "bank_chisq_dof", ligolw_cell_type_int_4s, LIGOLW_UNPACKING_REQUIRED},
		{&new->chisq, "chisq", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->chisq_dof, "chisq_dof", ligolw_cell_type_int_4s, LIGOLW_UNPACKING_REQUIRED},
		{&new->cont_chisq, "cont_chisq", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->cont_chisq_dof, "cont_chisq_dof", ligolw_cell_type_int_4s, LIGOLW_UNPACKING_REQUIRED},
		{&new->event_duration, "event_duration", ligolw_cell_type_real_8, LIGOLW_UNPACKING_REQUIRED},
		{&new->rsqveto_duration, "rsqveto_duration", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->alpha, "alpha", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->alpha1, "alpha1", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->alpha2, "alpha2", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->alpha3, "alpha3", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->alpha4, "alpha4", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->alpha5, "alpha5", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->alpha6, "alpha6", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->beta, "beta", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->kappa, "kappa", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->tau0, "tau0", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->tau2, "tau2", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->tau3, "tau3", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->tau4, "tau4", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->tau5, "tau5", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->psi0, "psi0", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->psi3, "psi3", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->Gamma[0], "Gamma0", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->Gamma[1], "Gamma1", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->Gamma[2], "Gamma2", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->Gamma[3], "Gamma3", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->Gamma[4], "Gamma4", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->Gamma[5], "Gamma5", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->Gamma[6], "Gamma6", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->Gamma[7], "Gamma7", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->Gamma[8], "Gamma8", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
		{&new->Gamma[9], "Gamma9", ligolw_cell_type_real_4, LIGOLW_UNPACKING_REQUIRED},
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
	strncpy(new->search, ligolw_row_get_cell(row, "search").as_string, LIGOMETA_SEARCH_MAX - 1);
	new->search[LIGOMETA_SEARCH_MAX - 1] = '\0';
	strncpy(new->ifo, ligolw_row_get_cell(row, "ifo").as_string, LIGOMETA_IFO_MAX - 1);
	new->ifo[LIGOMETA_IFO_MAX - 1] = '\0';
	strncpy(new->channel, ligolw_row_get_cell(row, "channel").as_string, LIGOMETA_CHANNEL_MAX - 1);
	new->channel[LIGOMETA_CHANNEL_MAX - 1] = '\0';

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

	/* add new object to head of linked list.  the linked list is
	 * reversed with respect to the file's contents.  it will be
	 * reversed again below */
	new->next = *head;
	*head = new;

	/* success */
	return 0;
}


int gstlal_snglinspiral_array_from_file(const char *filename, SnglInspiralTable **bankarray)
{
	SnglInspiralTable *head = NULL;
	SnglInspiralTable *row;
	ezxml_t xmldoc;
	ezxml_t elem;
	struct ligolw_table *table;
	int num = 0;

	/*
	 * so there's no confusion in case of error
	 */

	*bankarray = NULL;

	/*
	 * parse the document
	 */

	g_assert(filename != NULL);
	xmldoc = ezxml_parse_file(filename);
	if(!xmldoc) {
		XLALPrintError("%s(): error parsing \"%s\"\n", __func__, filename);
		goto parsefailed;
	}

	/*
	 * load sngl_inspiral table.
	 */

	elem = ligolw_table_get(xmldoc, "sngl_inspiral");
	if(elem) {
		table = ligolw_table_parse(elem, sngl_inspiral_row_callback, &head);
		if(!table) {
			XLALPrintError("%s(): failure parsing sngl_inspiral table in \"%s\"\n", __func__, filename);
			goto snglinspiralfailed;
		}
		ligolw_table_free(table);
	} else {
		XLALPrintError("%s(): no sngl_inspiral table in \"%s\"\n", __func__, filename);
		goto snglinspiralfailed;
	}

	/*
	 * clean up
	 */

	ezxml_free(xmldoc);

	/*
	 * count rows.  can't use table->n_rows because the callback
	 * interecepted the rows, and the table object is empty
	 */

	for(num = 0, row = head; row; row = row->next, num++);

	/*
	 * copy the linked list of templates into the template array in
	 * reverse order.  the linked list is reversed with respect to the
	 * contents of the file, so this constructs an array of templates
	 * in the order in which they appear in the file.
	 */

	*bankarray = calloc(num, sizeof(**bankarray));
	for(row = &(*bankarray)[num - 1]; head; row--) {
		SnglInspiralTable *next = head->next;

		/* fix broken columns */
		head->snr = 0;
		head->sigmasq = 0;
		head->mtotal = head->mass1 + head->mass2;
		head->mchirp = gstlal_mchirp(head->mass1, head->mass2);
		head->eta = gstlal_eta(head->mass1, head->mass2);

		*row = *head;
		LALFree(head);
		head = next;
	}

	/*
	 * success
	 */

	return num;

	/*
	 * error
	 */

snglinspiralfailed:
	ezxml_free(xmldoc);
parsefailed:
	return -1;
}


void gstlal_snglinspiral_array_free(SnglInspiralTable *bankarray)
{
	free(bankarray);
}

int gstlal_set_channel_in_snglinspiral_array(SnglInspiralTable *bankarray, int length, char *channel)
{
	if(channel)
		for(; length > 0; bankarray++, length--) {
			strncpy(bankarray->channel, channel, LIGOMETA_CHANNEL_MAX);
			bankarray->channel[LIGOMETA_CHANNEL_MAX - 1] = 0;
		}
	return 0;
}

int gstlal_set_instrument_in_snglinspiral_array(SnglInspiralTable *bankarray, int length, char *instrument)
{
	if(instrument)
		for(; length > 0; bankarray++, length--) {
			strncpy(bankarray->ifo, instrument, LIGOMETA_IFO_MAX);
			bankarray->ifo[LIGOMETA_IFO_MAX - 1] = 0;
		}
	return 0;
}

int gstlal_set_sigmasq_in_snglinspiral_array(SnglInspiralTable *bankarray, int length, double *sigmasq)
{
	for(; length > 0; bankarray++, sigmasq++, length--)
		bankarray->sigmasq = *sigmasq;
	return 0;
}

int gstlal_set_min_offset_in_snglinspiral_array(SnglInspiralTable *bankarray, int length, GstClockTimeDiff *timediff)
{
	int i;
	gint64 gpsns = G_MAXINT64;
	gint64 offset = 0;
	for (i = 0; i < length; i++) {
		offset = XLALGPSToINT8NS(&(bankarray[i].end));
		if (offset < gpsns)
			gpsns = offset;
	}
	/*
	 * FIXME FIXME FIXME This should be one sample at the sample rate, but
	 * unfortunately we don't have that in this function, so it is hardcoded to a
	 * very conservative value of 32 samples per second
	 */
	*timediff = gpsns - gst_util_uint64_scale_int_round(1, GST_SECOND, 32);
	return 0;
}

int populate_snglinspiral_buffer(GstBuffer *srcbuf, struct gstlal_peak_state *input, SnglInspiralTable *bankarray, GstPad *pad, guint64 length, GstClockTime time, guint rate, void *chi2, gsl_matrix_complex_float_view *H1_snr_matrix_view, gsl_matrix_complex_float_view *K1_snr_matrix_view, gsl_matrix_complex_float_view *L1_snr_matrix_view, gsl_matrix_complex_float_view *V1_snr_matrix_view)
{
	guint channel;
	guint L1_snr_timeseries_length, H1_snr_timeseries_length, V1_snr_timeseries_length, K1_snr_timeseries_length;
	for(channel = 0; channel < input->channels; channel++) {
		struct GSTLALSnglInspiral *event;
		SnglInspiralTable *parent;
		double complex maxdata_channel = 0;

		switch (input->type)
		{
			case GSTLAL_PEAK_COMPLEX:
			maxdata_channel = (double complex) input->interpvalues.as_float_complex[channel];
			break;

			case GSTLAL_PEAK_DOUBLE_COMPLEX:
			maxdata_channel = (double complex) input->interpvalues.as_double_complex[channel];
			break;

			default:
			g_assert(input->type == GSTLAL_PEAK_COMPLEX || input->type == GSTLAL_PEAK_DOUBLE_COMPLEX);
		}

		if (!maxdata_channel && !input->no_peaks_past_threshold)
			continue;

		/*
		 * allocate new event structure
		 */


		/*
		 * Populate the SNR snippet if available
		 * FIXME: only supported for single precision at the moment
		 */
		gsl_vector_complex_float_view H1_snr_vector_view, K1_snr_vector_view, L1_snr_vector_view, V1_snr_vector_view;
		gsl_vector_complex_float_view H1_snr_series_view, K1_snr_series_view, L1_snr_series_view, V1_snr_series_view;
		if ((H1_snr_matrix_view || K1_snr_matrix_view || L1_snr_matrix_view || V1_snr_matrix_view) && !input->no_peaks_past_threshold)
		{
			/* Allocate a set of empty time series. The event takes ownership, so no need to free it*/
			/* Get the columns of SNR we are interested in */
			if(H1_snr_matrix_view != NULL) {
				H1_snr_vector_view = gsl_matrix_complex_float_column(&(H1_snr_matrix_view->matrix), channel);
				H1_snr_timeseries_length = H1_snr_vector_view.vector.size;
			} else
				H1_snr_timeseries_length = 0;
			if(K1_snr_matrix_view != NULL) {
				K1_snr_vector_view = gsl_matrix_complex_float_column(&(K1_snr_matrix_view->matrix), channel);
				K1_snr_timeseries_length = K1_snr_vector_view.vector.size;
			} else
				K1_snr_timeseries_length = 0;
			if(L1_snr_matrix_view != NULL) {
				L1_snr_vector_view = gsl_matrix_complex_float_column(&(L1_snr_matrix_view->matrix), channel);
				L1_snr_timeseries_length = L1_snr_vector_view.vector.size;
			} else
				L1_snr_timeseries_length = 0;
			if(V1_snr_matrix_view != NULL) {
				V1_snr_vector_view = gsl_matrix_complex_float_column(&(V1_snr_matrix_view->matrix), channel);
				V1_snr_timeseries_length = V1_snr_vector_view.vector.size;
			} else
				V1_snr_timeseries_length = 0;

			event = gstlal_snglinspiral_new(H1_snr_timeseries_length, K1_snr_timeseries_length, L1_snr_timeseries_length, V1_snr_timeseries_length);

			if(H1_snr_matrix_view != NULL) {
				/* Make a GSL view of the time series array data */
				H1_snr_series_view = gsl_vector_complex_float_view_array((float *) event->snr, event->H1_length);
				/* Use BLAS to do the copy */
				gsl_blas_ccopy (&(H1_snr_vector_view.vector), &(H1_snr_series_view.vector));
			}
			if(K1_snr_matrix_view != NULL) {
				/* Make a GSL view of the time series array data */
				K1_snr_series_view = gsl_vector_complex_float_view_array((float *) &(event->snr[event->H1_length]), event->K1_length);
				/* Use BLAS to do the copy */
				gsl_blas_ccopy (&(K1_snr_vector_view.vector), &(K1_snr_series_view.vector));
			}
			if(L1_snr_matrix_view != NULL) {
				/* Make a GSL view of the time series array data */
				L1_snr_series_view = gsl_vector_complex_float_view_array((float *) &(event->snr[event->H1_length + event->K1_length]), event->L1_length);
				/* Use BLAS to do the copy */
				gsl_blas_ccopy (&(L1_snr_vector_view.vector), &(L1_snr_series_view.vector));
			}
			if(V1_snr_matrix_view != NULL) {
				/* Make a GSL view of the time series array data */
				V1_snr_series_view = gsl_vector_complex_float_view_array((float *) &(event->snr[event->H1_length + event->K1_length + event->L1_length]), event->V1_length);
				/* Use BLAS to do the copy */
				gsl_blas_ccopy (&(V1_snr_vector_view.vector), &(V1_snr_series_view.vector));
			}
		} else
			event = gstlal_snglinspiral_new(0,0,0,0);

		if (!event) {
			/* FIXME handle error */
		}
		/*
		 * populate
		 */

		parent = (SnglInspiralTable *) event;
		*parent = bankarray[channel];
		parent->snr = cabs(maxdata_channel);
		parent->coa_phase = carg(maxdata_channel);


		XLALINT8NSToGPS(&event->epoch, time);
		XLALGPSAddGPS(&event->epoch, &parent->end);
		parent->end = event->epoch;
		XLALGPSAdd(&parent->end, (double) input->interpsamples[channel] / rate);
		XLALGPSAdd(&event->epoch, ((gint) input->samples[channel] - (gint) input->pad) / (double) rate);
		event->deltaT = 1. / rate;

		parent->end_time_gmst = XLALGreenwichMeanSiderealTime(&parent->end);
		parent->eff_distance = gstlal_effective_distance(parent->snr, parent->sigmasq);
		/* populate chi squared if we have it */
		parent->chisq = 0.0;
		parent->chisq_dof = 1;
		switch (input->type)
		{
			case GSTLAL_PEAK_COMPLEX:
			if (chi2) parent->chisq = (double) *(((float *) chi2 ) + channel);
			break;

			case GSTLAL_PEAK_DOUBLE_COMPLEX:
			if (chi2) parent->chisq = (double) *(((double *) chi2 ) + channel);
			break;

			default:
			g_assert(input->type == GSTLAL_PEAK_COMPLEX || input->type == GSTLAL_PEAK_DOUBLE_COMPLEX);
		}


		/*
		 * add to buffer
		 */

		gst_buffer_append_memory(
			srcbuf,
			gst_memory_new_wrapped(
				GST_MEMORY_FLAG_READONLY | GST_MEMORY_FLAG_PHYSICALLY_CONTIGUOUS,
				event,
				sizeof(*event) + (event->H1_length + event->K1_length + event->L1_length + event->V1_length) * sizeof(event->snr[0]),
				0,
				sizeof(*event) + (event->H1_length + event->K1_length + event->L1_length + event->V1_length) * sizeof(event->snr[0]),
				event,
				(GDestroyNotify) gstlal_snglinspiral_free
			)
		);
	}
	return 0;
}

GstBuffer *gstlal_snglinspiral_new_buffer_from_peak(struct gstlal_peak_state *input, SnglInspiralTable *bankarray, GstPad *pad, guint64 offset, guint64 length, GstClockTime time, guint rate, void *chi2, gsl_matrix_complex_float_view *H1_snr_matrix_view, gsl_matrix_complex_float_view *K1_snr_matrix_view, gsl_matrix_complex_float_view *L1_snr_matrix_view, gsl_matrix_complex_float_view *V1_snr_matrix_view, GstClockTimeDiff timediff)
{
	GstBuffer *srcbuf = gst_buffer_new();

	if (!srcbuf) {
		GST_ERROR_OBJECT(pad, "Could not allocate sngl-inspiral buffer");
		return srcbuf;
	}

	if (input->is_gap)
	/* if (input->num_events == 0) FIXME this was what it used to do */
		GST_BUFFER_FLAG_SET(srcbuf, GST_BUFFER_FLAG_GAP);

	/* set the offset */
	GST_BUFFER_OFFSET(srcbuf) = offset;
	GST_BUFFER_OFFSET_END(srcbuf) = offset + length;

	/* set the time stamps */
	GST_BUFFER_PTS(srcbuf) = time + timediff;
	GST_BUFFER_DURATION(srcbuf) = (GstClockTime) gst_util_uint64_scale_int_round(GST_SECOND, length, rate);

	if (input->num_events || input->no_peaks_past_threshold) {
		populate_snglinspiral_buffer(srcbuf, input, bankarray, pad, length, time, rate, chi2, H1_snr_matrix_view, K1_snr_matrix_view, L1_snr_matrix_view, V1_snr_matrix_view);
	}
	return srcbuf;
}

int gstlal_snglinspiral_append_peak_to_buffer(GstBuffer *srcbuf, struct gstlal_peak_state *input, SnglInspiralTable *bankarray, GstPad *pad, guint64 offset, guint64 length, GstClockTime time, guint rate, void *chi2, gsl_matrix_complex_float_view *H1_snr_matrix_view, gsl_matrix_complex_float_view *K1_snr_matrix_view, gsl_matrix_complex_float_view *L1_snr_matrix_view, gsl_matrix_complex_float_view *V1_snr_matrix_view)
{
	//
	// Add peak information to a buffer, GST_BUFFER_OFFSET cannot be
	// changed but GST_BUFFER_OFFSET_END can
	//

	/* Update the offset end and duration */
	if(offset+length > GST_BUFFER_OFFSET_END(srcbuf)) {
		GST_BUFFER_OFFSET_END(srcbuf) = offset + length;
		GST_BUFFER_DURATION(srcbuf) = (GstClockTime) gst_util_uint64_scale_int_round(GST_SECOND, GST_BUFFER_OFFSET_END(srcbuf) - GST_BUFFER_OFFSET(srcbuf), rate);
	}

	populate_snglinspiral_buffer(srcbuf, input, bankarray, pad, length, time, rate, chi2, H1_snr_matrix_view, K1_snr_matrix_view, L1_snr_matrix_view, V1_snr_matrix_view);

	return 0;
}

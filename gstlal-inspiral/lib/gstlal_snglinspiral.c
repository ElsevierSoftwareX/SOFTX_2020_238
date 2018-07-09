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
#include <gstlal/gstlal_peakfinder.h>
#include <complex.h>
#include <string.h>
#include <math.h>
#include <lal/Date.h>
#include <lal/LIGOMetadataTables.h>
#include <lal/LIGOLwXMLInspiralRead.h>
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

int gstlal_snglinspiral_array_from_file(const char *bank_filename, SnglInspiralTable **bankarray)
{
	SnglInspiralTable *this = NULL;
	SnglInspiralTable *bank = NULL;
	int num;
	num = LALSnglInspiralTableFromLIGOLw(&this, bank_filename, -1, -1);

	*bankarray = bank = (SnglInspiralTable *) calloc(num, sizeof(SnglInspiralTable));

	/* FIXME do some basic sanity checking */

	/*
	 * copy the linked list of templates constructed by
	 * LALSnglInspiralTableFromLIGOLw() into the template array.
	 */

	while (this) {
		SnglInspiralTable *next = this->next;
		this->snr = 0;
		this->sigmasq = 0;
		this->mtotal = this->mass1 + this->mass2;
		this->mchirp = gstlal_mchirp(this->mass1, this->mass2);
		this->eta = gstlal_eta(this->mass1, this->mass2);
		*bank = *this;
		bank->next = NULL;
		bank++;
		LALFree(this);
		this = next;
	}

	return num;
}

int gstlal_set_channel_in_snglinspiral_array(SnglInspiralTable *bankarray, int length, char *channel)
{
	int i;
	for (i = 0; i < length; i++) {
		if (channel) {
			strncpy(bankarray[i].channel, (const char*) channel, LIGOMETA_CHANNEL_MAX);
			bankarray[i].channel[LIGOMETA_CHANNEL_MAX - 1] = 0;
		}
	}
	return 0;
}

int gstlal_set_instrument_in_snglinspiral_array(SnglInspiralTable *bankarray, int length, char *instrument)
{
	int i;
	for (i = 0; i < length; i++) {
		if (instrument) {
			strncpy(bankarray[i].ifo, (const char*) instrument, LIGOMETA_IFO_MAX);
			bankarray[i].ifo[LIGOMETA_IFO_MAX - 1] = 0;
		}
	}
	return 0;
}

int gstlal_set_sigmasq_in_snglinspiral_array(SnglInspiralTable *bankarray, int length, double *sigmasq)
{
	int i;
	for (i = 0; i < length; i++) {
		bankarray[i].sigmasq = sigmasq[i];
	}
	return 0;
}

int gstlal_set_min_offset_in_snglinspiral_array(SnglInspiralTable *bankarray, int length, GstClockTimeDiff *timediff)
{
	int i;
	gint64 gpsns = 0;
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

GstBuffer *gstlal_snglinspiral_new_buffer_from_peak(struct gstlal_peak_state *input, SnglInspiralTable *bankarray, GstPad *pad, guint64 offset, guint64 length, GstClockTime time, guint rate, void *chi2, gsl_matrix_complex_float_view *snr_matrix_view, GstClockTimeDiff timediff)
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

	if (input->num_events) {
		guint channel;
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

			if (!maxdata_channel)
				continue;

			/*
			 * allocate new event structure
			 */


			/*
			 * Populate the SNR snippet if available
			 * FIXME: only supported for single precision at the moment
			 */
			if (snr_matrix_view)
			{
				/* Get the column of SNR we are interested in */
				gsl_vector_complex_float_view snr_vector_view = gsl_matrix_complex_float_column(&(snr_matrix_view->matrix), channel);
				/* Allocate an empty time series to hold it. The event takes ownership, so no need to free it*/
				event = gstlal_snglinspiral_new(snr_vector_view.vector.size);
				/* Make a GSL view of the time series array data */
				gsl_vector_complex_float_view snr_series_view = gsl_vector_complex_float_view_array((float *) event->snr, event->length);
				/* Use BLAS to do the copy */
				gsl_blas_ccopy (&(snr_vector_view.vector), &(snr_series_view.vector));
			} else
				event = gstlal_snglinspiral_new(0);

			parent = (SnglInspiralTable *) event;
			if (!event) {
				/* FIXME handle error */
			}

			/*
			 * populate
			 */

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
					sizeof(*event) + sizeof(event->snr[0]) * event->length,
					0,
					sizeof(*event) + sizeof(event->snr[0]) * event->length,
					event,
					(GDestroyNotify) gstlal_snglinspiral_free
				)
			);
		}
	}

	return srcbuf;
}

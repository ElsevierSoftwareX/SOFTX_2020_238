/*
 * Copyright (C) 2012,2013,2015,2016  Chad Hanna
 *                         2017-2018  Duncan Meacher
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
#include <sngltriggerrowtype.h>
#include <gsl/gsl_matrix_float.h>
#include <gsl/gsl_blas.h>
#include <gstlal_sngltrigger.h>


GstBuffer *gstlal_sngltrigger_new_buffer_from_peak(struct gstlal_peak_state *input, char *channel_name, GstPad *pad, guint64 offset, guint64 length, GstClockTime time, guint rate, void *chi2, gsl_matrix_complex_float_view *snr_matrix_view, GstClockTimeDiff timediff, gboolean max_snr)
{
	GstBuffer *srcbuf = gst_buffer_new();

	int max_snr_index = -1;

	if (!srcbuf) {
		GST_ERROR_OBJECT(pad, "Could not allocate sngl-trigger buffer");
		return srcbuf;
	}

	if (input->num_events == 0)
		GST_BUFFER_FLAG_SET(srcbuf, GST_BUFFER_FLAG_GAP);

	/* set the offset */
	GST_BUFFER_OFFSET(srcbuf) = offset;
	GST_BUFFER_OFFSET_END(srcbuf) = offset + length;

	/* set the time stamps */
	GST_BUFFER_PTS(srcbuf) = time + timediff;
	GST_BUFFER_DURATION(srcbuf) = (GstClockTime) gst_util_uint64_scale_int_round(GST_SECOND, length, rate);

	if (input->num_events) {

		/* create a single buffer from max snr channel */
		if(max_snr) {
			max_snr_index = gstlal_peak_max_over_channels(input);

			/* Type casting unsigned int (guint) to int */
			g_assert(max_snr_index >= 0 && max_snr_index < (int)input->channels);

			/* create a single buffer from max snr channel */
			add_buffer_from_channel(input, channel_name, pad, length, time, rate, chi2, snr_matrix_view, max_snr_index, srcbuf);

		/* add a buffer from each channel */
		} else {

			guint channel;
			for(channel = 0; channel < input->channels; channel++) {
				add_buffer_from_channel(input, channel_name, pad, length, time, rate, chi2, snr_matrix_view, channel, srcbuf);
			}
		}
	}

	return srcbuf;
}

void add_buffer_from_channel(struct gstlal_peak_state *input, char *channel_name, GstPad *pad, guint64 length, GstClockTime time, guint rate, void *chi2, gsl_matrix_complex_float_view *snr_matrix_view, int channel, GstBuffer *srcbuf)
{
	struct GSTLALSnglTrigger *event;
	SnglTriggerTable *parent;
	double complex maxdata_channel = 0;

	switch (input->type)
	{
		case GSTLAL_PEAK_COMPLEX:
		maxdata_channel = (double complex) input->values.as_float_complex[channel];
		break;

		case GSTLAL_PEAK_DOUBLE_COMPLEX:
		maxdata_channel = (double complex) input->values.as_double_complex[channel];
		break;

		default:
		g_assert(input->type == GSTLAL_PEAK_COMPLEX || input->type == GSTLAL_PEAK_DOUBLE_COMPLEX);
	}

	if (!maxdata_channel)
		return;

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
		event = gstlal_sngltrigger_new(snr_vector_view.vector.size);
		/* Make a GSL view of the time series array data */
		gsl_vector_complex_float_view snr_series_view = gsl_vector_complex_float_view_array((float *) event->snr, event->length);
		/* Use BLAS to do the copy */
		gsl_blas_ccopy (&(snr_vector_view.vector), &(snr_series_view.vector));
	} else
		event = gstlal_sngltrigger_new(0);

	parent = (SnglTriggerTable *) event;
	if (!event) {
		/* FIXME handle error */
	}

	/*
	 * populate
	 */

	//*parent = bankarray[channel];
	strcpy(parent->channel, channel_name);
	parent->snr = cabs(maxdata_channel);
	parent->phase = carg(maxdata_channel);
	parent->channel_index = channel;

	XLALINT8NSToGPS(&event->epoch, time);
	{
		LIGOTimeGPS end_time = event->epoch;
		XLALGPSAdd(&end_time, (double) input->samples[channel] / rate);
		XLALGPSAddGPS(&parent->end, &end_time);
	}
	XLALGPSAdd(&event->epoch, (double) (input->samples[channel] - input->pad) / rate);
	event->deltaT = 1. / rate;

	/* populate chi squared if we have it */
	parent->chisq = 0.0;
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
			(GDestroyNotify) gstlal_sngltrigger_free
		)
	);
}

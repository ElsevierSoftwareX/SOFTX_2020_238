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

/**
 * SECTION:gstlal_snglinspiral.c
 * @short_description:  Utilities for sngl inspiral 
 * 
 * Reviewed: 38c65535fc96d6cc3dee76c2de9d3b76b47d5283 2015-05-14 
 * K. Cannon, J. Creighton, C. Hanna, F. Robinett 
 * 
 * Actions:
 *  56: free() should be LALFree()
 * 67,79: outside of loop
 * 144: add bankarray end time here to take into account the IMR waveform shifts
 * 152: figure out how to use a more accurate sigmasq calculation
 */



GstBuffer *gstlal_snglinspiral_new_buffer_from_peak_spearman(struct gstlal_peak_state *input, SnglInspiralTable *bankarray, GstPad *pad, guint64 offset, guint64 length, GstClockTime time, guint rate, void *chi2, void *pval)
{
	/* FIXME check errors */

	GstBuffer *srcbuf = NULL;
	GstCaps *caps = GST_PAD_CAPS(pad);
	GstFlowReturn result = gst_pad_alloc_buffer(pad, offset, sizeof(*bankarray) * input->num_events, caps, &srcbuf);
	guint channel;
	double complex maxdata_channel = 0;

	if (result != GST_FLOW_OK) {
		GST_ERROR_OBJECT(pad, "Could not allocate sngl-inspiral buffer %d", result);
		return srcbuf;
		}

	if (input->num_events == 0)
		GST_BUFFER_FLAG_SET(srcbuf, GST_BUFFER_FLAG_GAP);

	/* set the offset */
        GST_BUFFER_OFFSET(srcbuf) = offset;
        GST_BUFFER_OFFSET_END(srcbuf) = offset + length;

        /* set the time stamps */
        GST_BUFFER_TIMESTAMP(srcbuf) = time;
        GST_BUFFER_DURATION(srcbuf) = (GstClockTime) gst_util_uint64_scale_int_round(GST_SECOND, length, rate);

	/* FIXME do error checking */
	if (input->num_events) {
		SnglInspiralTable *output = (SnglInspiralTable *) GST_BUFFER_DATA(srcbuf);
		for(channel = 0; channel < input->channels; channel++) {
	
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

			if ( maxdata_channel ) {
				LIGOTimeGPS end_time;
				XLALINT8NSToGPS(&end_time, time);
				XLALGPSAdd(&end_time, (double) input->interpsamples[channel] / rate);
				memcpy(output, &(bankarray[channel]), sizeof(*bankarray));
				output->snr = cabs(maxdata_channel);
				output->coa_phase = carg(maxdata_channel);
				output->chisq = 0.0;
				output->chisq_dof = 1;
				output->end_time = end_time;
				output->end_time_gmst = XLALGreenwichMeanSiderealTime(&end_time);
				output->eff_distance = gstlal_effective_distance(output->snr, output->sigmasq);
				/* populate chi squared if we have it */
				switch (input->type)
					{
					case GSTLAL_PEAK_COMPLEX:
					if (chi2) output->chisq = (double) *(((float *) chi2 ) + channel);
					if (pval) output->bank_chisq = (double) *(((float *) pval ) + channel);
					break;
		
					case GSTLAL_PEAK_DOUBLE_COMPLEX:
					if (chi2) output->chisq = (double) *(((double *) chi2 ) + channel);
					break;

					default:
					g_assert(input->type == GSTLAL_PEAK_COMPLEX || input->type == GSTLAL_PEAK_DOUBLE_COMPLEX);
					}
				output++;
			}
		}
		g_assert_cmpuint(output - (SnglInspiralTable *) GST_BUFFER_DATA(srcbuf), ==, input->num_events);
	}

	return srcbuf;
}

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

int gstlal_snglinspiral_array_from_file(char *bank_filename, SnglInspiralTable **bankarray)
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
		free(this);
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

GstBuffer *gstlal_snglinspiral_new_buffer_from_peak(struct gstlal_peak_state *input, SnglInspiralTable *bankarray, GstPad *pad, guint64 offset, guint64 length, GstClockTime time, guint rate, void *chi2)
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

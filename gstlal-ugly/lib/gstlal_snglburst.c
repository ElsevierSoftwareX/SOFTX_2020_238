#include <glib.h>
#include <glib-object.h>
#include <gst/gst.h>
#include <gstlal/gstlal_peakfinder.h>
#include <complex.h>
#include <string.h>
#include <math.h>
#include <lal/Date.h>
#include <lal/LIGOMetadataTables.h>
#include <lal/LIGOMetadataBurstUtils.h>
#include <lal/LIGOLwXMLBurstRead.h>
#include <lal/LALStdlib.h>


int gstlal_snglburst_array_from_file(char *bank_filename, SnglBurst **bankarray)
{
	SnglBurst *this = NULL;
	SnglBurst *bank = NULL;
	int num = 0;
	bank = this = XLALSnglBurstTableFromLIGOLw(bank_filename);
	/* count the rows */
	while (bank) {
		num++;
		bank = bank->next;
		}

	*bankarray = bank = (SnglBurst *) calloc(num, sizeof(SnglBurst));

	/* FIXME do some basic sanity checking */

	/*
	 * copy the linked list of templates constructed into the template
	 * array.
	 */

	while (this) {
		SnglBurst *next = this->next;
		this->snr = 0;
		*bank = *this;
		bank->next = NULL;
		bank++;
		free(this);
		this = next;
	}

	return num;
}

int gstlal_set_channel_in_snglburst_array(SnglBurst *bankarray, int length, char *channel)
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

int gstlal_set_instrument_in_snglburst_array(SnglBurst *bankarray, int length, char *instrument)
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

GstBuffer *gstlal_snglburst_new_double_buffer_from_peak(struct gstlal_peak_state *input, SnglBurst *bankarray, GstPad *pad, guint64 offset, guint64 length, GstClockTime time, guint rate, guint64 *count)
{
	/* FIXME check errors */

	/* size is length in samples times number of channels times number of bytes per sample */
	gint size = sizeof(SnglBurst) * input->num_events;
	GstBuffer *srcbuf = NULL;
	GstCaps *caps = GST_PAD_CAPS(pad);
	GstFlowReturn result = gst_pad_alloc_buffer(pad, offset, size, caps, &srcbuf);
	if (result != GST_FLOW_OK)
		return srcbuf;

	SnglBurst *output = (SnglBurst *) GST_BUFFER_DATA(srcbuf);
	guint channel;
	double *maxdata = input->values.as_double;
	guint *maxsample = input->samples;

	if (input->num_events == 0)
		GST_BUFFER_FLAG_SET(srcbuf, GST_BUFFER_FLAG_GAP);

	/* set the offset */
  GST_BUFFER_OFFSET(srcbuf) = offset;
  GST_BUFFER_OFFSET_END(srcbuf) = offset + length;

  /* set the time stamps */
  GST_BUFFER_TIMESTAMP(srcbuf) = time;
  GST_BUFFER_DURATION(srcbuf) = (GstClockTime) gst_util_uint64_scale_int_round(GST_SECOND, length, rate);

	/* FIXME do error checking */
	if (srcbuf && size) {
		for(channel = 0; channel < input->channels; channel++) {
			if ( maxdata[channel] ) {
				memcpy(output, &(bankarray[channel]), sizeof(SnglBurst));
				LIGOTimeGPS peak_time;
				XLALINT8NSToGPS(&peak_time, time);
				XLALGPSAdd(&peak_time, (double) maxsample[channel] / rate);
				LIGOTimeGPS start_time = peak_time;
				XLALGPSAdd(&start_time, -output->duration/2);
				output->snr = fabs(maxdata[channel]);
				output->start_time = start_time;
				output->peak_time = peak_time;
				//FIXME: Process ID
				XLALSnglBurstAssignIDs( output, 0, *count );
				(*count)++;
				output++;
			}
		}
	}

	return srcbuf;
}

GstBuffer *gstlal_snglburst_new_buffer_from_peak(struct gstlal_peak_state *input, SnglBurst *bankarray, GstPad *pad, guint64 offset, guint64 length, GstClockTime time, guint rate, guint64 *count)
{
	/* FIXME check errors */

	/* size is length in samples times number of channels times number of bytes per sample */
	gint size = sizeof(SnglBurst) * input->num_events;
	GstBuffer *srcbuf = NULL;
	GstCaps *caps = GST_PAD_CAPS(pad);
	GstFlowReturn result = gst_pad_alloc_buffer(pad, offset, size, caps, &srcbuf);
	SnglBurst *output = (SnglBurst *) GST_BUFFER_DATA(srcbuf);
	guint channel;
	double complex *maxdata = input->values.as_double_complex;
	guint *maxsample = input->samples;

	if (result != GST_FLOW_OK)
		return srcbuf;

	if (input->num_events == 0)
		GST_BUFFER_FLAG_SET(srcbuf, GST_BUFFER_FLAG_GAP);

	/* set the offset */
        GST_BUFFER_OFFSET(srcbuf) = offset;
        GST_BUFFER_OFFSET_END(srcbuf) = offset + length;

        /* set the time stamps */
        GST_BUFFER_TIMESTAMP(srcbuf) = time;
        GST_BUFFER_DURATION(srcbuf) = (GstClockTime) gst_util_uint64_scale_int_round(GST_SECOND, length, rate);

	/* FIXME do error checking */
	if (srcbuf && size) {
		for(channel = 0; channel < input->channels; channel++) {
			if ( maxdata[channel] ) {
				memcpy(output, &(bankarray[channel]), sizeof(SnglBurst));
				LIGOTimeGPS peak_time;
				XLALINT8NSToGPS(&peak_time, time);
				XLALGPSAdd(&peak_time, (double) maxsample[channel] / rate);
				LIGOTimeGPS start_time = peak_time;
				XLALGPSAdd(&start_time, -output->duration/2);
				output->snr = cabs(maxdata[channel]);
				output->start_time = start_time;
				output->peak_time = peak_time;
				//FIXME: Process ID
				XLALSnglBurstAssignIDs( output, 0, *count );
				(*count)++;
				output++;
			}
		}
	}

	return srcbuf;
}

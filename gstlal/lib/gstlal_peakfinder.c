#include <glib.h>
#include <glib-object.h>
#include <gst/gst.h>
#include <gstlal_peakfinder.h>
#include <complex.h>
#include <string.h>
#include <math.h>

/*
 * Double precision
 */

/* Init a structure to hold peak times and values */
struct gstlal_double_peak_samples_and_values *gstlal_double_peak_samples_and_values_new(guint channels)
{
	struct gstlal_double_peak_samples_and_values *new = g_new0(struct gstlal_double_peak_samples_and_values,1);
	if (!new) return NULL;
	new->channels = channels;
	new->samples = g_malloc0(sizeof(guint) * channels);
	new->values = g_malloc0(sizeof(double) * channels);
	new->num_events = 0;
	new->pad = 0;
	new->thresh = 0;
	return new;
}

/* Clear a structure to hold peak times and values */
int gstlal_double_peak_samples_and_values_clear(struct gstlal_double_peak_samples_and_values *val)
{
	memset(val->samples, 0.0, val->channels * sizeof(guint));
	memset(val->values, 0.0, val->channels * sizeof(double));
	val->num_events = 0;
	return 0;
}

/* Simple peak over window algorithm */
int gstlal_double_peak_over_window(struct gstlal_double_peak_samples_and_values *output, const double *data, guint64 length)
{
	guint sample, channel;
	double *maxdata = output->values;
	guint *maxsample = output->samples;
	
	/* clear the output array */
	gstlal_double_peak_samples_and_values_clear(output);
	
	/* Find maxima of the data */
	for(sample = 0; sample < length; sample++) {
		for(channel = 0; channel < output->channels; channel++) {
			if(fabs(*data) > fabs(maxdata[channel]) && fabs(*data) > output->thresh) {
				/* only increment events if the previous value was 0 */
				if (fabs(maxdata[channel]) == 0)
					output->num_events += 1;
				maxdata[channel] = *data;
				maxsample[channel] = sample;
			}
		data++;
		}
	}
	
	return 0;
}

/* simple function to fill a buffer with the max values */
int gstlal_double_fill_output_with_peak(struct gstlal_double_peak_samples_and_values *input, double *data, guint64 length)
{

	guint channel, index;
	double *maxdata = input->values;
	guint *maxsample = input->samples;
	
	/* clear the output data */
	memset(data, 0.0, length * sizeof(double));

	/* Decide if there are any events to keep */
	for(channel = 0; channel < input->channels; channel++) {
		if ( maxdata[channel] ) {
			index = maxsample[channel] * input->channels + channel;
			data[index] = maxdata[channel];
		}
	}
	return 0;
}

/* A convenience function to make a new buffer of doubles and populate it with peaks */
GstBuffer *gstlal_double_new_buffer_from_peak(struct gstlal_double_peak_samples_and_values *input, GstPad *pad, guint64 offset, guint64 length, GstClockTime time, guint rate)
{
	/* FIXME check errors */
	
	/* size is length in samples times number of channels times number of bytes per sample */
	gint size = sizeof(double) * length * input->channels;
	GstBuffer *srcbuf = NULL;
	GstCaps *caps = GST_PAD_CAPS(pad);
	GstFlowReturn result = gst_pad_alloc_buffer(pad, offset, size, caps, &srcbuf);

	/* FIXME someday with better gap support don't actually allocate data
	 * in this case.  For now we just mark it as a gap but let the rest of
	 * this function do its thing so that we get a buffer allocated with
	 * zeros 
	 */
	
	if (input->num_events == 0)
		GST_BUFFER_FLAG_SET(srcbuf, GST_BUFFER_FLAG_GAP);

       	if (result != GST_FLOW_OK)
		return srcbuf;

	/* set the offset */
        GST_BUFFER_OFFSET(srcbuf) = offset;
        GST_BUFFER_OFFSET_END(srcbuf) = offset + length;

        /* set the time stamps */
        GST_BUFFER_TIMESTAMP(srcbuf) = time;
        GST_BUFFER_DURATION(srcbuf) = (GstClockTime) gst_util_uint64_scale_int_round(GST_SECOND, length, rate);
	
	if (srcbuf)
		gstlal_double_fill_output_with_peak(input, (double *) GST_BUFFER_DATA(srcbuf), length);

	return srcbuf;
}

/*
 * Complex double precision
 */

/* Init a structure to hold peak times and values */
struct gstlal_double_complex_peak_samples_and_values *gstlal_double_complex_peak_samples_and_values_new(guint channels)
{
	struct gstlal_double_complex_peak_samples_and_values *new = g_new0(struct gstlal_double_complex_peak_samples_and_values,1);
	if (!new) return NULL;
	new->channels = channels;
	new->samples = g_malloc0(sizeof(guint) * channels);
	new->values = g_malloc0(sizeof(double complex) * channels);
	new->num_events = 0;
	new->pad = 0;
	new->thresh = 0;
	return new;
}

/* Clear a structure to hold peak times and values */
int gstlal_double_complex_peak_samples_and_values_clear(struct gstlal_double_complex_peak_samples_and_values *val)
{
	memset(val->samples, 0.0, val->channels * sizeof(guint));
	memset(val->values, 0.0, val->channels * sizeof(double complex));
	val->num_events = 0;
	return 0;
}

/* Simple peak over window algorithm */
int gstlal_double_complex_peak_over_window(struct gstlal_double_complex_peak_samples_and_values *output, const double complex *data, guint64 length)
{
	guint sample, channel;
	double complex *maxdata = output->values;
	guint *maxsample = output->samples;
	
	/* clear the output array */
	gstlal_double_complex_peak_samples_and_values_clear(output);
	
	/* Find maxima of the data */
	for(sample = 0; sample < length; sample++) {
		for(channel = 0; channel < output->channels; channel++) {
			if(cabs(*data) > cabs(maxdata[channel]) && cabs(*data) > output->thresh) {
				/* only increment events if the previous value was 0 */
				if (cabs(maxdata[channel]) == 0)
					output->num_events += 1;
				maxdata[channel] = *data;
				maxsample[channel] = sample;
			}
		data++;
		}
	}
	
	return 0;
}

/* Assumes that you can index the data being given, if not expect a segfault or
 * worse.  Data pointer must exist be valid outputmat->size2 / 2 samples in past and future of the time over which the peak was computed
 */

int gstlal_double_complex_series_around_peak(struct gstlal_double_complex_peak_samples_and_values *input, double complex *data, double complex *outputmat, guint n)
{
	guint channel, sample;
	gint index;
	guint *maxsample = input->samples;
	double complex *maxdata = input->values;
	double complex *peakdata = NULL;
	memset(outputmat, 0, sizeof(double) * input->channels * (2 * n + 1));

	for (channel = 0; channel < input->channels; channel++) {
		if (maxdata[channel]) {
			index = (maxsample[channel] - n) * input->channels + channel;
			for (sample = 0, peakdata = data+index; sample < (2*n + 1); sample++, peakdata += input->channels)
				outputmat[sample * input->channels + channel] = *peakdata;
			}
		}

	return 0;
}



/* simple function to fill a buffer with the max values */
int gstlal_double_complex_fill_output_with_peak(struct gstlal_double_complex_peak_samples_and_values *input, double complex *data, guint64 length)
{

	guint channel, index;
	double complex *maxdata = input->values;
	guint *maxsample = input->samples;
	
	/* clear the output data */
	memset(data, 0.0, length * sizeof(double complex));

	/* Decide if there are any events to keep */
	for(channel = 0; channel < input->channels; channel++) {
		if ( maxdata[channel] ) {
			index = maxsample[channel] * input->channels + channel;
			data[index] = maxdata[channel];
		}
	}
	return 0;
}

/* A convenience function to make a new buffer of doubles and populate it with peaks */
GstBuffer *gstlal_double_complex_new_buffer_from_peak(struct gstlal_double_complex_peak_samples_and_values *input, GstPad *pad, guint64 offset, guint64 length, GstClockTime time, guint rate)
{
	/* FIXME check errors */
	
	/* size is length in samples times number of channels times number of bytes per sample */
	gint size = sizeof(double complex) * length * input->channels;
	GstBuffer *srcbuf = NULL;
	GstCaps *caps = GST_PAD_CAPS(pad);
	GstFlowReturn result = gst_pad_alloc_buffer(pad, offset, size, caps, &srcbuf);

	/* FIXME someday with better gap support don't actually allocate data
	 * in this case.  For now we just mark it as a gap but let the rest of
	 * this function do its thing so that we get a buffer allocated with
	 * zeros 
	 */
	
	if (input->num_events == 0)
		GST_BUFFER_FLAG_SET(srcbuf, GST_BUFFER_FLAG_GAP);

       	if (result != GST_FLOW_OK)
		return srcbuf;

	/* set the offset */
        GST_BUFFER_OFFSET(srcbuf) = offset;
        GST_BUFFER_OFFSET_END(srcbuf) = offset + length;

        /* set the time stamps */
        GST_BUFFER_TIMESTAMP(srcbuf) = time;
        GST_BUFFER_DURATION(srcbuf) = (GstClockTime) gst_util_uint64_scale_int_round(GST_SECOND, length, rate);
	
	if (srcbuf)
		gstlal_double_complex_fill_output_with_peak(input, (double complex *) GST_BUFFER_DATA(srcbuf), length);

	return srcbuf;
}


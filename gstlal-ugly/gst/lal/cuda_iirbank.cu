#ifdef __cplusplus
extern "C" {
#endif

#include <cuda_iirbank.h>

#ifdef __cplusplus
}
#endif

/*
 * ============================================================================
 *
 *                                 Utilities
 *
 * ============================================================================
 */

/*
 * return the number of IIR channels
 */
unsigned iir_channels(const GSTLALIIRBankCuda *element)
{
	if(element->a1)
		return 2 * element->a1->size1;
	return 0;
}

/*
 * the number of samples available in the adapter
 */


guint64 get_available_samples(GSTLALIIRBankCuda *element)
{
	return gst_adapter_available(element->adapter) / sizeof(double);
}



/*
 * set the metadata on an output buffer. 
 */


void set_metadata(GSTLALIIRBankCuda *element, GstBuffer *buf, guint64 outsamples, gboolean gap)
{
        GST_BUFFER_SIZE(buf) = outsamples * iir_channels(element) * sizeof(double);
        GST_BUFFER_OFFSET(buf) = element->next_out_offset;
	element->next_out_offset += outsamples;
	GST_BUFFER_OFFSET_END(buf) = element->next_out_offset;
	GST_BUFFER_TIMESTAMP(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET(buf) - element->offset0, GST_SECOND, element->rate);
	GST_BUFFER_DURATION(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET_END(buf) - element->offset0, GST_SECOND, element->rate) - GST_BUFFER_TIMESTAMP(buf);
	if(element->need_discont) {
		GST_BUFFER_FLAG_SET(buf, GST_BUFFER_FLAG_DISCONT);
		element->need_discont = FALSE;
	}
	if(gap)
		GST_BUFFER_FLAG_SET(buf, GST_BUFFER_FLAG_GAP);
	else
		GST_BUFFER_FLAG_UNSET(buf, GST_BUFFER_FLAG_GAP);
}


/*
 * transform input samples to output samples using a time-domain algorithm
 */


GstFlowReturn filter(GSTLALIIRBankCuda *element, GstBuffer *outbuf)
{
	unsigned available_length;
	unsigned output_length;
	double * restrict input;
	complex double * restrict output;
	complex double * restrict last_output;
	complex double * restrict last_filter;
	complex double ytemp;
	int dmax, dmin;
	complex double * restrict y, * restrict a1, * restrict b0;
	int * restrict d;
	uint size1, size2;

	y = (complex double *) gsl_matrix_complex_ptr(element->y, 0, 0);
	a1 = (complex double *) gsl_matrix_complex_ptr(element->a1, 0, 0);
	b0 = (complex double *) gsl_matrix_complex_ptr(element->b0, 0, 0);
	d = gsl_matrix_int_ptr(element->delay, 0, 0);

	/*
	 * how much data is available?
	 */

	gsl_matrix_int_minmax(element->delay, &dmin, &dmax);
	dmin = 0;
	available_length = get_available_samples(element);
	output_length = available_length - (dmax - dmin);

	if(!output_length)
		return GST_BASE_TRANSFORM_FLOW_DROPPED;

	/*
	 * wrap the adapter's contents in a GSL vector view.
	 */

	input = (double *) gst_adapter_peek(element->adapter, available_length * sizeof(double));

	/*
	 * wrap output buffer in a complex double array.
	 */

	output = (complex double *) GST_BUFFER_DATA(outbuf);
	g_assert(output_length * iir_channels(element) / 2 * sizeof(complex double) <= GST_BUFFER_SIZE(outbuf));

	memset(output, 0, output_length * iir_channels(element) / 2 * sizeof(*output));

	size1 = element->a1->size1;
	size2 = element->a1->size2;

	for (last_output = output + size1; output < last_output; output++) {
		for (last_filter = y + size2; y < last_filter; y++) {
			ytemp = *y;
			complex double *out = output;
			double *in_last, *in = &input[dmax -*d];

			for(in_last = in + output_length; in < in_last; in++, out += size1) { /* sample # */
				ytemp = *a1 * ytemp + *b0 * *in;
				*out += ytemp;
			}
			*y = ytemp;
			a1++;
			b0++;
			d++;
		}
	}

	/*
	 * flush the data from the adapter
	 */

	gst_adapter_flush(element->adapter, output_length * sizeof(double));
	if(element->zeros_in_adapter > available_length - output_length)
		/*
		 * some trailing zeros have been flushed from the adapter
		 */

		element->zeros_in_adapter = available_length - output_length;

	/*
	 * set buffer metadata
	 */

	set_metadata(element, outbuf, output_length, FALSE);

	/*
	 * done
	 */

	return GST_FLOW_OK;
}


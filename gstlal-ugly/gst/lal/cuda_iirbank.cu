/*
  TODO : copyright Yuan Liu, Qi Chu
 
 */

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
//	printf("duration : %d \n", GST_BUFFER_DURATION(buf));
}


/*
 * transform input samples to output samples using a time-domain algorithm
 */



int bank_init(iirBank **pbank, GSTLALIIRBankCuda *element)
{

	int dmax, dmin;
	complex double * restrict y, * restrict a1, * restrict b0;
	int * restrict d;
	uint size1, size2;

	gsl_matrix_int_minmax(element->delay, &dmin, &dmax);
	dmin = 0;
	size1 = element->a1->size1;
	size2 = element->a1->size2;

	y = (complex double *) gsl_matrix_complex_ptr(element->y, 0, 0);
	a1 = (complex double *) gsl_matrix_complex_ptr(element->a1, 0, 0);
	b0 = (complex double *) gsl_matrix_complex_ptr(element->b0, 0, 0);
	d = gsl_matrix_int_ptr(element->delay, 0, 0);

	/*
	 * how much data is available?
	 */
	
	(*pbank) = (iirBank *)malloc( sizeof(iirBank) );
	(*pbank)->a1_f = (COMPLEX8_F *)malloc( size1 * size2 * sizeof( COMPLEX8_F) );
	(*pbank)->b0_f = (COMPLEX8_F *)malloc( size1 * size2 * sizeof( COMPLEX8_F) * size1 * size2 );
	(*pbank)->y_f = (COMPLEX8_F *)malloc( size1 * size2 * sizeof( COMPLEX8_F) * size1 * size2 );
	(*pbank)->d_i = d;
	(*pbank)->input_f = NULL;
	(*pbank)->output_f = NULL;

	(*pbank)->num_templates = size1;
	(*pbank)->num_filters = size2;
	(*pbank)->dmax = dmax;
	(*pbank)->dmin = dmin;
	(*pbank)->rate = element->rate;
	(*pbank)->pre_input_length = 0;
	(*pbank)->pre_output_length = 0;

	/*
	 * prepare a1, b0, y, convert to float 
	 */

	int i;
	for(i=0; i < size1*size2; i++)
	{
		((*pbank)->a1_f)[i].re = float(creal(a1[i]));
		((*pbank)->a1_f)[i].im = float(cimag(a1[i]));
		((*pbank)->b0_f)[i].re = float(creal(b0[i]));
		((*pbank)->b0_f)[i].im = float(cimag(b0[i]));
		((*pbank)->y_f)[i].re = float(creal(y[i]));
		((*pbank)->y_f)[i].im = float(cimag(y[i]));
	}
	return BANK_INIT_SUCCESS;


}

int bank_free(iirBank **pbank, GSTLALIIRBankCuda *element)
{
	if((*pbank)->a1_f) {
		free((*pbank)->a1_f);
		(*pbank)->a1_f = NULL;
	}
	if((*pbank)->b0_f) {
		free((*pbank)->b0_f);
		(*pbank)->b0_f = NULL;
	}
	if((*pbank)->y_f) {
		free((*pbank)->y_f);
		(*pbank)->y_f = NULL;
	}
	if((*pbank)->d_i) {
		free((*pbank)->d_i);
		(*pbank)->d_i = NULL;
		(element->delay)->data = NULL;
	}
	if((*pbank)->input_f) {
		free((*pbank)->input_f);
		(*pbank)->input_f = NULL;
	}
	if((*pbank)->output_f) {
		free((*pbank)->output_f);
		(*pbank)->output_f = NULL;
	}
	return BANK_FREE_SUCCESS;


}

GstFlowReturn filter(GSTLALIIRBankCuda *element, GstBuffer *outbuf)
{
	double * restrict input;
	complex double * restrict output;

	unsigned available_length;
	unsigned output_length;
	available_length = get_available_samples(element);
	input = (double *) gst_adapter_peek(element->adapter, available_length * sizeof(double));
	/*
	 * initialize bank
	 */
	static iirBank * bank;
	static gboolean first_call = TRUE;
	static gboolean last_call = FALSE;

	/*
	 *	TODO : Cuda variables definition, some are static
	 *
	 */

	if (first_call)
	{
		bank_init(&bank, element);
	/*
	 *
	 *
	 *	TODO : transfer the static data CPU -> GPU : bank->a1, b0, y, d
	 *
	 *
	 */
		first_call = FALSE;
	}

	output_length = available_length - (bank->dmax - bank->dmin);
	if(!output_length)
		return GST_BASE_TRANSFORM_FLOW_DROPPED;

	if(output_length != bank->pre_output_length)
	{
		if(bank->input_f)
		{
			free(bank->input_f);
			bank->input_f = NULL;
		}	
		bank->input_f = (float *)malloc( available_length * sizeof(float));
		memset(bank->input_f, 0, available_length * sizeof(float));
		if(bank->output_f)
		{
			free(bank->output_f);
			bank->output_f = NULL;
		}	
		bank->output_f = (COMPLEX8_F *)malloc(iir_channels(element)/2 * output_length * sizeof(COMPLEX8_F) );
		memset(bank->output_f, 0, output_length * iir_channels(element) / 2 * sizeof(COMPLEX8_F));
	}

	int i;
	for(i=0; i<available_length; i++)
		(bank->input_f)[i] = float(input[i]);

	/*
	 *
	 *
	 *	TODO : transfer the input data to CPU -> GPU
	 *
	 *
	 */

	/*
	 *
	 *
	 *	TODO : cuda kernel
	 *
	 *
	 */

	/*
	 *
	 *
	 *	TODO : transfer the output data to GPU -> CPU
	 *
	 *
	 */


	/*
	 * wrap output buffer in a complex double array.
	 */
	output = (complex double *) GST_BUFFER_DATA(outbuf);
	g_assert(output_length * iir_channels(element) / 2 * sizeof(complex double) <= GST_BUFFER_SIZE(outbuf));

	memset(output, 0, output_length * iir_channels(element) / 2 * sizeof(*output));
	for(i=0; i < output_length * iir_channels(element)/2; i++)
	{
		*(output +i) = double((bank->output_f)[i].re) + ( double((bank->output_f)[i].im) ) * _Complex_I;
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

	bank->pre_input_length = available_length;
	bank->pre_output_length = output_length;

	/*
	 *
	 *
	 *	TODO  for Chichi : free data. How to know if it is the last call
	 *
	 *
	 */
	if (last_call)
	{
		bank_free(&bank, element);
	}

	/*
	 * done
	 */

	return GST_FLOW_OK;
}


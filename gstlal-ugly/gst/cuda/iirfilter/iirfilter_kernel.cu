/*
 * Copyright (C) 2012 Yuan Liu, Qi Chu <chicsheep@gmail.com>
 * Copyright (C) 2013 Qi Chu <chicsheep@gmail.com>
 * 
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */


/*
 * ============================================================================
 *
 *				Preamble
 *
 * ============================================================================
 */


#ifdef __cplusplus
extern "C" {
#endif
#include <stdio.h>
#include <iirfilter/iirfilter_kernel.h>

#ifdef __cplusplus
}
#endif

// for gpu debug
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
/*
 * ============================================================================
 *
 *				 Utilities
 *
 * ============================================================================
*/

/*
 * get the stream ID-- deprecated
 */

/*
int get_stream_id(GSTLALIIRBankCuda *elem)
{
	
	switch (elem->rate)
	{
	case RATE_8:	
		return 8;
	case RATE_7:	
		return 7;
	case RATE_6:	
		return 6;
	case RATE_5:
		return 5;
	case RATE_4:
		return 4;
	case RATE_3:
		return 3;
	case RATE_2:
		return 2;
	case RATE_1:
		return 1;
	default:
		return 1;
	}
}
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
	return gst_adapter_available(element->adapter) / ( element->width / 8 );
}



/*
 * set the metadata on an output buffer. 
 */

void set_metadata(GSTLALIIRBankCuda *element, GstBuffer *buf, guint64 outsamples, gboolean gap)
{
        GST_BUFFER_SIZE(buf) = outsamples * iir_channels(element) * element->width / 8;
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



int cuda_bank_init(GSTLALIIRBankCuda *element)
{

	int dmax, dmin;
	complex double * restrict y, * restrict a1, * restrict b0;
	COMPLEX8_F *h_a1_s, *h_b0_s, *h_y_s;
	int * restrict d;
	uint size1, size2;

	IIRBankCuda_s *bank;

	gsl_matrix_int_minmax(element->delay, &dmin, &dmax);
	dmin = 0;
	size1 = element->a1->size1;
	size2 = element->a1->size2;

	y = (complex double *) gsl_matrix_complex_ptr(element->y, 0, 0);
	a1 = (complex double *) gsl_matrix_complex_ptr(element->a1, 0, 0);
	b0 = (complex double *) gsl_matrix_complex_ptr(element->b0, 0, 0);
	d = gsl_matrix_int_ptr(element->delay, 0, 0);

	/*
	 * malloc intermediate cpu memory
	*/

	h_a1_s = (COMPLEX8_F *)malloc( size1 * size2 * sizeof( COMPLEX8_F) );
	h_b0_s = (COMPLEX8_F *)malloc( size1 * size2 * sizeof( COMPLEX8_F) );
	h_y_s = (COMPLEX8_F *)malloc( size1 * size2 * sizeof( COMPLEX8_F) );

	/*
	 * malloc memory in IIRBankCuda
	 */

	element->bank = (IIRBankCuda_s *)malloc( sizeof(IIRBankCuda_s) );
	bank = element->bank;

	
	cudaMalloc((void **)&bank->d_a1_s, size1 * size2 *sizeof(COMPLEX8_F));
	cudaMalloc((void **)&bank->d_b0_s, size1 * size2 *sizeof(COMPLEX8_F));
	cudaMalloc((void **)&bank->d_y_s, size1 * size2 *sizeof(COMPLEX8_F));
	cudaMalloc((void **)&bank->d_d_i, size1 * size2 *sizeof(int));

	/*
	 * set other fileds in IIRBankCuda
	*/

	bank->h_input_s = NULL;
	bank->d_input_s = NULL;
	bank->h_output_s = NULL;
	bank->d_output_s = NULL;

	bank->num_templates = size1;
	bank->num_filters = size2;
	bank->dmax = dmax;
	bank->dmin = dmin;
	bank->rate = element->rate;
	bank->pre_input_length = 0;
	bank->pre_output_length = 0;

	/*
	 * prepare a1, b0, y, convert to float 
	 */
	int i;
	for(i=0; i < size1*size2; i++)
	{
		h_a1_s[i].re = float(creal(a1[i]));
		h_a1_s[i].im = float(cimag(a1[i]));
		h_b0_s[i].re = float(creal(b0[i]));
		h_b0_s[i].im = float(cimag(b0[i]));
		h_y_s[i].re = float(creal(y[i]));
		h_y_s[i].im = float(cimag(y[i]));
	}

	cudaStreamCreate(&element->stream); 
	/*
	 * copy to GPU memory
	*/

	gpuErrchk(cudaPeekAtLastError());
	cudaMemcpyAsync(bank->d_a1_s, h_a1_s, size1 * size2 * sizeof(COMPLEX8_F), cudaMemcpyHostToDevice, element->stream);
	cudaMemcpyAsync(bank->d_b0_s, h_b0_s, size1 * size2 * sizeof(COMPLEX8_F), cudaMemcpyHostToDevice, element->stream);
	cudaMemcpyAsync(bank->d_y_s, h_y_s, size1 * size2 * sizeof(COMPLEX8_F), cudaMemcpyHostToDevice, element->stream);
	cudaMemcpyAsync(bank->d_d_i, d, size1 * size2 * sizeof(int), cudaMemcpyHostToDevice, element->stream);

	gpuErrchk(cudaPeekAtLastError());
	free(h_a1_s);
	free(h_b0_s);
	free(h_y_s);

	return BANK_INIT_SUCCESS;
}

int cuda_bank_free(GSTLALIIRBankCuda *element)
{

	cudaStreamDestroy(element->stream);

	if((element->bank)->d_a1_s) {
		cudaFree((element->bank)->d_a1_s);
		(element->bank)->d_a1_s = NULL;
	}
	if((element->bank)->d_b0_s) {
		cudaFree((element->bank)->d_b0_s);
		(element->bank)->d_b0_s = NULL;
	}
	if((element->bank)->d_y_s) {
		cudaFree((element->bank)->d_y_s);
		(element->bank)->d_y_s = NULL;
	}
	if((element->bank)->d_d_i) {
		cudaFree((element->bank)->d_d_i);
		(element->bank)->d_d_i = NULL;
	}
	if((element->bank)->d_input_s) {
		cudaFree((element->bank)->d_input_s);
		(element->bank)->d_input_s = NULL;
	}
	if((element->bank)->d_output_s) {
		cudaFree((element->bank)->d_output_s);
		(element->bank)->d_output_s = NULL;
	}
	if((element->bank)->h_input_s) {
		free((element->bank)->h_input_s);
		(element->bank)->h_input_s = NULL;
	}
	if((element->bank)->h_output_s) {
		free((element->bank)->h_output_s);
		(element->bank)->h_output_s = NULL;
	}
	if((element->bank)) {
		free((element->bank));
		(element->bank) = NULL;
	}
	return BANK_FREE_SUCCESS;
}

/*
 * cuda filter kernel
 */
extern __shared__ char sharedMem[];

texture<float, 1, cudaReadModeElementType> texRef;

__global__ void cuda_iir_filter_kernel( COMPLEX8_F *cudaA1, 
					COMPLEX8_F *cudaB0, int *cudaShift, 
					COMPLEX8_F *cudaPrevSnr,
					float *cudaData, COMPLEX8_F *cudaSnr, 
					unsigned int numFilters, 
					unsigned int step_points, int delay_max, unsigned int nb)
{
	unsigned int i,j;

	COMPLEX8_F a1, b0;
	int shift;
	unsigned int tx = threadIdx.x;
	//unsigned int bx = blockIdx.x;
	unsigned int by = blockIdx.y;
	
	unsigned int threads = blockDim.x;
	float data;
	COMPLEX8_F *gPrevSnr;
	COMPLEX8_F previousSnr;
	COMPLEX8_F *snr;
	COMPLEX8_F snrVal;
	unsigned int numSixtnGrp;
	numSixtnGrp = (numFilters + 16 -1)/16;
	
	volatile float *fltrOutptReal = (float *)sharedMem;
	volatile float *fltrOutptImag = &(fltrOutptReal[numFilters+8]);
	float *grpOutptReal = (float *)&(fltrOutptImag[numFilters+8]);	
	float *grpOutptImag = &(grpOutptReal[numSixtnGrp*nb]);
	
	unsigned int tx_2 = tx%16;
	unsigned int tx_3 = tx/16;
	for (i = tx; i < 8; i += threads)
	{
		fltrOutptReal[numFilters+i] = 0.0f;
		fltrOutptImag[numFilters+i] = 0.0f;
	}
	__syncthreads();
	if( tx < numFilters ) 
	{

		gPrevSnr = &(cudaPrevSnr[by * numFilters + tx]);
		previousSnr = *gPrevSnr;
		
		a1 = cudaA1[by * numFilters + tx];
		b0 = cudaB0[by * numFilters + tx];
		shift = delay_max - cudaShift[by * numFilters + tx];
		if(tx < nb)
		{
			snr = &(cudaSnr[by*step_points+tx]);
		}
		for( i = 0; i < step_points; i+=nb )
		{
			for(j = 0; j < nb; ++j)
			{ 
				//data = 0.01f;
				//data = tex1Dfetch(texRef, shift+i+j);		//use texture, abandon now
				data = cudaData[shift+i+j];
				fltrOutptReal[tx] = a1.re * previousSnr.re - a1.im * previousSnr.im + b0.re * data;
		 
				fltrOutptImag[tx] = a1.re * previousSnr.im + a1.im * previousSnr.re + b0.im * data;
			 
				previousSnr.re = fltrOutptReal[tx];
				previousSnr.im = fltrOutptImag[tx];
			
				fltrOutptReal[tx] += fltrOutptReal[tx+8];
				fltrOutptImag[tx] += fltrOutptImag[tx+8];
				fltrOutptReal[tx] += fltrOutptReal[tx+4];
				fltrOutptImag[tx] += fltrOutptImag[tx+4];
				fltrOutptReal[tx] += fltrOutptReal[tx+2];
				fltrOutptImag[tx] += fltrOutptImag[tx+2];
				fltrOutptReal[tx] += fltrOutptReal[tx+1];
				fltrOutptImag[tx] += fltrOutptImag[tx+1];
				if(tx_2 == 0)
				{
					grpOutptReal[tx_3*nb+j] = fltrOutptReal[tx];
					grpOutptImag[tx_3*nb+j] = fltrOutptImag[tx];
				}
			}
			__syncthreads();
			if(tx < nb)
			{
				snrVal.re = 0.0f;
				snrVal.im = 0.0f;
				for(j = 0; j < numSixtnGrp; ++j)
				{
						snrVal.re += grpOutptReal[j*nb+tx];
						snrVal.im += grpOutptImag[j*nb+tx];
				}
				snr[i] = snrVal;
			}	 
			__syncthreads();
		}
		*gPrevSnr = previousSnr;			//store previousSnr for next step
	}
}


/*
 * transform input samples to output samples using a time-domain algorithm -- double precision
 */

GstFlowReturn filter_d(GSTLALIIRBankCuda *element, GstBuffer *outbuf)
{
	double * restrict input;
	complex double * restrict output;
	unsigned int i, j;

	unsigned available_length;
	unsigned output_length;
	available_length = get_available_samples(element);
	input = (double *) gst_adapter_peek(element->adapter, available_length * sizeof(double));

	/* GPU setup 	
	cudaSetDevice(0);
	// static streams deprecated
	// streames numbers set to allow processing 5 bank file, 5 files * 3 detectors * 8 rates = 120
	static cudaStream_t streams[128]; //= (cudaStream_t*)malloc(NSTREAMS * sizeof (cudaStream_t));
	static int streamID = 0;
	*/
	static int streamID = 0;
	static GMutex *stream_lock = g_mutex_new();

	if (element->bank == NULL)
	{
		g_mutex_lock(stream_lock);
		streamID = streamID + 1;
		int deviceCount;
		cudaGetDeviceCount(&deviceCount);
		element->deviceID = streamID % deviceCount;	
		cudaSetDevice(element->deviceID);
		cuda_bank_init(element);
		//printf("streamID is %d, device ID is %d\n", streamID, element->deviceID);
		g_mutex_unlock(stream_lock);
	}

	//cudaSetDevice(element->deviceID);

	/*
	 * local bank equals element->bank for conveniency
	 */

	IIRBankCuda_s *bank;
	bank = element->bank;

	output_length = available_length - (bank->dmax - bank->dmin);

	if(!output_length)
		return GST_BASE_TRANSFORM_FLOW_DROPPED;

	if(output_length != bank->pre_output_length)
	{
		// set nb
		i = NB_MAX;
		if (bank->num_filters < NB_MAX) 
		{
			i = bank->num_filters;
		}

		for (; i > 0; --i)
		{
			if (output_length % i == 0)
			{
				break;
			}
		}
		bank->nb = i;
	}

	// only need to realloc memory when output_length gets bigger

	if(output_length > bank->pre_output_length)
	{
		// realloc cpu input and output buf
		if(bank->h_input_s)
		{
			cudaFreeHost(bank->h_input_s);
		}	
			cudaMallocHost(&bank->h_input_s, available_length * sizeof(float));
		//bank->h_input_s = (float *)malloc( available_length * sizeof(float));
		if(bank->h_output_s)
		{
			cudaFreeHost(bank->h_output_s);
		}	
			cudaMallocHost(&bank->h_output_s, bank->num_templates * output_length * sizeof(COMPLEX8_F));
		//bank->h_output_s = (COMPLEX8_F *)malloc(bank->num_templates * output_length * sizeof(COMPLEX8_F) );

		// realloc gpu input and output buf
		if (bank->d_input_s)
		{
			cudaFree(bank->d_input_s);
		}
		cudaMalloc((void **)&bank->d_input_s, available_length * sizeof(float));
		if (bank->d_output_s)
		{
			cudaFree(bank->d_output_s);
		}
		cudaMalloc((void **)&bank->d_output_s, output_length * bank->num_templates * sizeof(COMPLEX8_F));

	}

	/*
	 * transfer input data to gpu memory
	 */
	for(i=0; i<available_length; i++)
		bank->h_input_s[i] = float(input[i]);

	
	cudaMemcpyAsync(bank->d_input_s, bank->h_input_s, available_length * sizeof(float), cudaMemcpyHostToDevice, element->stream);
	/*
	 *	cuda kernel
	 */
	dim3 block(bank->num_filters, 1, 1);
	dim3 grid(1, bank->num_templates, 1);
	uint share_mem_sz = (block.x+8 + (bank->num_filters+16-1)/16*bank->nb) * sizeof(float) * 2;

	// using mutex to make sure that kernel launch is right after texture binding
	//g_mutex_lock(element->cuTex_lock);
	//Set up texture.
	/*cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>(); 
	texRef.addressMode[0] = cudaAddressModeWrap;
	texRef.filterMode	= cudaFilterModeLinear;
	texRef.normalized	= false;
	cudaBindTexture(0, texRef, bank->d_input_s, channelDesc, available_length * sizeof(float));
	*/
	cuda_iir_filter_kernel<<<grid, block, share_mem_sz, element->stream>>>(bank->d_a1_s, bank->d_b0_s, bank->d_d_i, bank->d_y_s, bank->d_input_s, bank->d_output_s, bank->num_filters, output_length, bank->dmax, bank->nb);
	//g_mutex_unlock(element->cuTex_lock);

	/*
	 * transfer the output data to GPU -> CPU
	 */
	cudaMemcpyAsync(bank->h_output_s, bank->d_output_s, output_length * bank->num_templates * sizeof(COMPLEX8_F), cudaMemcpyDeviceToHost, element->stream); 
	cudaStreamSynchronize(element->stream);
	/*
	 * wrap output buffer in a complex double array.
	 */
	output = (complex double *) GST_BUFFER_DATA(outbuf);
	g_assert(output_length * iir_channels(element) / 2 * sizeof(complex double) <= GST_BUFFER_SIZE(outbuf));

	COMPLEX8_F temp;
	for (i = 0; i < output_length; ++i)
	{
		for (j = 0; j < bank->num_templates; ++j)
		{
			temp = bank->h_output_s[j*output_length+i];
			output[i*bank->num_templates+j] =	(double)temp.re + ((double)temp.im)*_Complex_I;
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

	bank->pre_input_length = available_length;
	bank->pre_output_length = output_length;

	/*
	 * done
	 */
	bank = NULL;

	return GST_FLOW_OK;
}


GstFlowReturn filter_s(GSTLALIIRBankCuda *element, GstBuffer *outbuf)
{
	float * restrict input;
	complex float * restrict output;
	unsigned int i;

	unsigned available_length;
	unsigned output_length;
	available_length = get_available_samples(element);
	input = (float *) gst_adapter_peek(element->adapter, available_length * element->width / 8 );


	/* GPU setup 	
	cudaSetDevice(0);
	// static streams deprecated
	// streames numbers set to allow processing 5 bank file, 5 files * 3 detectors * 8 rates = 120
	static cudaStream_t streams[128]; //= (cudaStream_t*)malloc(NSTREAMS * sizeof (cudaStream_t));
	static int streamID = 0;
	*/
	static int streamID = 0;
	static GMutex *stream_lock = g_mutex_new();

	if (element->bank == NULL)
	{
		g_mutex_lock(stream_lock);
		streamID = streamID + 1;
		int deviceCount;
		cudaGetDeviceCount(&deviceCount);
		element->deviceID = streamID % deviceCount;	
		cudaSetDevice(element->deviceID);
		cuda_bank_init(element);
		//printf("streamID is %d, device ID is %d\n", streamID, element->deviceID);
		g_mutex_unlock(stream_lock);
	}

	//cudaSetDevice(element->deviceID);
	/*
	 * local bank equals element->bank for conveniency
	 */

	IIRBankCuda_s *bank;
	bank = element->bank;

	output_length = available_length - (bank->dmax - bank->dmin);

	if(!output_length)
		return GST_BASE_TRANSFORM_FLOW_DROPPED;

	if(output_length != bank->pre_output_length)
	{
		// set nb
		i = NB_MAX;
		if (bank->num_filters < NB_MAX) 
		{
			i = bank->num_filters;
		}

		for (; i > 0; --i)
		{
			if (output_length % i == 0)
			{
				break;
			}
		}
		bank->nb = i;
	}

	// only need to realloc memory when output_length gets bigger

	if(output_length > bank->pre_output_length)
	{
		// realloc cpu input and output buf
		if(bank->h_input_s)
		{
			cudaFreeHost(bank->h_input_s);
		}	
		cudaMallocHost(&bank->h_input_s, available_length * sizeof(float));
		//bank->h_input_s = (float *)malloc( available_length * sizeof(float));
		if(bank->h_output_s)
		{
			cudaFreeHost(bank->h_output_s);
		}	
		cudaMallocHost(&bank->h_output_s, bank->num_templates * output_length * sizeof(COMPLEX8_F));
		//bank->h_output_s = (COMPLEX8_F *)malloc(bank->num_templates * output_length * sizeof(COMPLEX8_F) );

		// realloc gpu input and output buf
		if (bank->d_input_s)
		{
			cudaFree(bank->d_input_s);
		}
		cudaMalloc((void **)&bank->d_input_s, available_length * sizeof(float));
		if (bank->d_output_s)
		{
			cudaFree(bank->d_output_s);
		}
		cudaMalloc((void **)&bank->d_output_s, output_length * bank->num_templates * sizeof(COMPLEX8_F));

	}

	/*
	 * transfer input data to gpu memory
	 */
	for(i=0; i<available_length; i++)
		bank->h_input_s[i] = float(input[i]);

	cudaMemcpyAsync(bank->d_input_s, bank->h_input_s, available_length * sizeof(float), cudaMemcpyHostToDevice, element->stream);

	/*
	 *	cuda kernel
	 */
	dim3 block(bank->num_filters, 1, 1);
	dim3 grid(1, bank->num_templates, 1);
	uint share_mem_sz = (block.x+8 + (bank->num_filters+16-1)/16*bank->nb) * sizeof(float) * 2;

	// using mutex to make sure that kernel launch is right after texture binding
	//g_mutex_lock(element->cuTex_lock);
	//Set up texture.
	/*cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>(); 
	texRef.addressMode[0] = cudaAddressModeWrap;
	texRef.filterMode	= cudaFilterModeLinear;
	texRef.normalized	= false;
	cudaBindTexture(0, texRef, bank->d_input_s, channelDesc, available_length * sizeof(float));
	*/

	cuda_iir_filter_kernel<<<grid, block, share_mem_sz, element->stream>>>(bank->d_a1_s, bank->d_b0_s, bank->d_d_i, bank->d_y_s, bank->d_input_s, bank->d_output_s, bank->num_filters, output_length, bank->dmax, bank->nb);
	//g_mutex_unlock(element->cuTex_lock);

	/*
	 * transfer the output data to GPU -> CPU
	 */
	cudaMemcpyAsync(bank->h_output_s, bank->d_output_s, output_length * bank->num_templates * sizeof(COMPLEX8_F), cudaMemcpyDeviceToHost, element->stream); 
	cudaStreamSynchronize(element->stream);
//	gpuErrchk(cudaPeekAtLastError());


	/*
	 * wrap output buffer in a complex float array.
	 */
	output = (complex float *) GST_BUFFER_DATA(outbuf);
	g_assert(output_length * iir_channels(element) / 2 * sizeof(complex float) <= GST_BUFFER_SIZE(outbuf));


	memset(output, 0, output_length * iir_channels(element) / 2 * sizeof(*output));
	COMPLEX8_F temp;
	int j;
	for (i = 0; i < output_length; ++i)
	{
		for (j = 0; j < bank->num_templates; ++j)
		{
			temp = bank->h_output_s[ j*output_length+i ];
			output[ i*bank->num_templates+j ] = temp.re + temp.im*_Complex_I;
		}
	}
	/*
	 * flush the data from the adapter
	 */


	gst_adapter_flush(element->adapter, output_length * sizeof(float));
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
	 * done
	 */
	bank = NULL;

	return GST_FLOW_OK;
}


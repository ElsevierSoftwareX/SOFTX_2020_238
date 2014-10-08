
#ifdef __cplusplus
extern "C" {
#endif
#include <stdio.h>
#include <glib.h>
#include <gst/gst.h>


#include "multiratespiir.h"
#include "multiratespiir_utils.h"
#include "spiir_state_macro.h"

#ifdef __cplusplus
}
#endif

#define THREADSPERBLOCK 256
#define NB_MAX 32

// for gpu debug
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
static void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      GST_LOG ("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

extern __shared__ char sharedMem[];

__global__ void downsample2x (const float amplifier,
			      const int times,
  			      float *sinc, 
			      const int filt_len, 
			      int last_sample,
			      float *mem, 
			      const int len, 
			      float *queue_in, 
			      float *queue_out
)
{
	unsigned int tx = threadIdx.x;
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int tdx = blockDim.x;
	unsigned int bdx = gridDim.x;
	unsigned int bx = blockIdx.x;
	int tx2, tid2, tdx2,  in_start, in_end, tmp_mem_end;
	int tail_len = last_sample;
	int i;

	tx2 = 2 * tx;
	tid2 = 2 * tid;
	tdx2 = 2 * tdx;

	volatile float *tmp_mem = (float *)sharedMem;
	volatile float *tmp_sinc = &(tmp_mem[tdx2 + 3*filt_len]);
	float tmp = 0.0;

	/*
	 * copy the sinc table to shared mem
	 */

	for (i=tx; i < filt_len; i+=tdx)
		tmp_sinc[i] = sinc[i];

	if (bx < 1) {
		for (i=tx;  i<filt_len-1; i+=tdx) 
			tmp_mem[i] = mem[i];
	}
	else {
		in_start = 2 * bx * tdx;
		for (i=tx;  i<filt_len-1; i+=tdx) 
			tmp_mem[i] = queue_in[ in_start - (filt_len - 1) + i] ;
	}

	if (tid < len) {
	/*
	 * grab filt_len-1 data,
	 * for the first block, grab data from mem 
	 */


		tmp_mem[tx2 + filt_len - 1] = queue_in[tid2];
		tmp_mem[tx2 + filt_len]  = queue_in[tid2 + 1];

	}
	if (bx == bdx - 1) {
		tmp_mem_end = (len - bx * tdx) * 2;
		in_end = len * 2;
	}
	else {
		tmp_mem_end = tdx2;
		in_end = 2*(tdx + bx*tdx);
	}


	for (i=tx;  i<tail_len; i+=tdx) 
		tmp_mem[tmp_mem_end + filt_len - 1 + i] = queue_in[in_end + i];
	__syncthreads();	

	if (tid < len) {

		for (int j = 0; j < filt_len; ++j) {
		  tmp += tmp_mem[tx2 + last_sample + j] * tmp_sinc[j];
		}

		queue_out[tid] 	 = tmp * amplifier;
//		printf("in device %d, mem %f in %f\n", tx_loc+1, mem[filt_len-1+tx_loc+1], queue_in[tx_loc+1]);
	}


	// copy last to first (filt_len-1) mem data
	if (bx == bdx - 1) {
		for (i=tx;  i<filt_len - 1; i+=tdx) {
			mem[i] = tmp_mem[tmp_mem_end + last_sample + i];
		}
	}
}


__global__ void reload_queue_spiir (float *queue,
				 float *queue_spiir,
				 int num_inchunk,
				 int num_left,
				 gint queue_spiir_start,
				 gint queue_spiir_len)
{
	unsigned int tx = threadIdx.x, tdx = blockDim.x;
	int i, j;

	for (i=tx; i< num_inchunk; i+=tdx) {
	  j = (queue_spiir_start + i) % queue_spiir_len;
	  queue_spiir[j] = queue[i];
	}
    /*
     * shift shiftInput  samples of queue;
     * update the effective length, down_start of this spstate;
     */

	float *tmp = (float *) sharedMem;

	for (i=tx; i< num_left; i+=tdx) 
		tmp[i] = queue[i+num_inchunk];
	__syncthreads();

	for (i=tx; i< num_left; i+=tdx) 
		queue[i] = tmp[i];
#if 0
	/ should be removed ?
	for (i=tx; i< num_inchunk; i+=tdx) {
	  	j = (queue_spiir_start + i) % queue_spiir_len;
	}
#endif
	

}
/*
 * cuda filter kernel
 */

texture<float, 1, cudaReadModeElementType> texRef;

__global__ void cuda_iir_filter_kernel( COMPLEX_F *cudaA1, 
					COMPLEX_F *cudaB0, int *cudaShift, 
					COMPLEX_F *cudaPrevSnr,
					float *cudaData, 
					float *cudaSnr, gint mem_len, 
					gint filt_len, gint delay_max,
					gint step_points, 
					guint nb, 
					gint queue_spiir_last_sample,
					gint queue_spiir_len)
{
	unsigned int i,j;

	COMPLEX_F a1, b0;
	int shift;
	unsigned int tx = threadIdx.x;
	//unsigned int bx = blockIdx.x;
	unsigned int by = blockIdx.y;
	
	unsigned int threads = blockDim.x, numFilters = blockDim.x;
	float data;
	COMPLEX_F *gPrevSnr;
	COMPLEX_F previousSnr;
	float *snr_real, *snr_imag;
	COMPLEX_F snrVal;
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
			snr_real = &(cudaSnr[2*by*mem_len+filt_len-1+tx]);
			snr_imag = &(cudaSnr[(2*by+1)*mem_len+filt_len-1+tx]);
		}

		for( i = 0; i < step_points; i+=nb )
		{
			for(j = 0; j < nb; ++j)
			{ 
				//data = 0.01f;
				//data = tex1Dfetch(texRef, shift+i+j);		//use texture, abandon now
				data = cudaData[(shift+i+j+queue_spiir_last_sample)%queue_spiir_len];
//				printf ("channelid %d, data %d %f\n", by, i+j, data);
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
				snr_real[i] = snrVal.re;
				snr_imag[i] = snrVal.im;
			}	 
			__syncthreads();
		}
		*gPrevSnr = previousSnr;			//store previousSnr for next step
	}
}

extern __shared__ char sharedMem[];


__global__ void upsample2x_and_add (
  			      float *sinc, 
			      const gint filt_len, 
			      gint last_sample,
			      const gint len,
			      float *mem_in, 
			      float *mem_out,
			      const gint mem_in_len,
			      const gint mem_out_len)
{
	volatile float *tmp_sinc = (float *) sharedMem;
	float tmp0 = 0.0, tmp1 = 0.0, tmp_in;
	unsigned int tx = threadIdx.x, tdx = blockDim.x;
	unsigned int by = blockIdx.y;
	int  pos, pos_in_start = mem_in_len * by + last_sample, pos_out_start = mem_out_len * by + filt_len - 1;
	float *in, *out;
	int i;
	for (i=tx; i<filt_len * 2; i+=tdx)
		tmp_sinc[i] = sinc[i];

	for (i=tx; i< len; i+=tdx) {
	  tmp0 = 0.0;
	  tmp1 = 0.0;
	  pos = i;
		in = &(mem_in[pos_in_start + pos]);

		for (int j = 0; j < filt_len; ++j) {
			tmp_in = in[j];
			tmp0 += tmp_in * tmp_sinc[j];
			tmp1 += tmp_in * tmp_sinc[j + filt_len];
		}
		out = &(mem_out[pos_out_start + 2 * pos]);

		out[0] += tmp0;
		out[1] += tmp1;
	}

	__syncthreads();

	// copy last to first (filt_len-1) mem data
	for (i=tx; i<filt_len - 1; i+=tdx) {
		in = &(mem_in[mem_in_len * by + i]);
		in[0] = in[len];
	}

}

__global__ void iir_filter (
			    float *in,
			    float *out)
{
	int tx = threadIdx.x + blockIdx.x * blockDim.x;
	out[tx] = in[tx];
}

__global__ void flush_queue (
			    float *queue,
			    int num_flush)
{
	int tx = threadIdx.x;
	queue[tx] = queue[tx + num_flush];
}


gint multi_downsample (SpiirState **spstate, float *in_multidown, gint num_in_multidown, gint num_depths, cudaStream_t stream)
{
  float *pos_inqueue, *pos_outqueue;
  gint i, out_processed;
  gint num_inchunk = num_in_multidown;

  GST_LOG ("multi downsample %d samples", num_inchunk);
  g_assert (SPSTATE(0)->queue_eff_len + num_inchunk <= SPSTATE(0)->queue_len);
  pos_inqueue = SPSTATE(0)->d_queue + SPSTATE(0)->queue_eff_len ;
  /* 
   * copy inbuf data to first queue
   */
  cudaMemcpyAsync(pos_inqueue, in_multidown, num_inchunk * sizeof(float), cudaMemcpyHostToDevice, stream);

  SPSTATE(0)->queue_eff_len += num_inchunk;

  for (i=0; i<num_depths-1; i++) {
    // predicted output length of downsample this round
    out_processed = (num_inchunk - SPSTATEDOWN(i)->last_sample)/2;
    //g_assert ((SPSTATE(i-1)->queue_eff_len - SPSTATE(i-1)->queue_down_start >= num_inchunk;
    /*
     * downsample 2x of number of filt samples
     */
    // make sure lower depth mem is large enough to store queue data.
    g_assert (num_inchunk <= SPSTATEDOWN(i)->mem_len - SPSTATEDOWN(i)->filt_len + 1 );
    // make sure current depth queue is large enough to store output data
    g_assert (out_processed <= SPSTATE(i+1)->queue_len - SPSTATE(i+1)->queue_eff_len);
    pos_inqueue = SPSTATE(i)->d_queue + SPSTATE(i)->queue_down_start;
    pos_outqueue = SPSTATE(i+1)->d_queue + SPSTATE(i+1)->queue_down_start;

    /* 
     * CUDA downsample2x 
     */

    
    dim3 block(1, 1, 1);
    dim3 grid(1, 1, 1);
    block.x = MIN(THREADSPERBLOCK, out_processed); 
    grid.x = out_processed % block.x == 0 ? out_processed/block.x : (int)out_processed/block.x + 1;

    uint share_mem_sz = (2 * block.x + 4 * SPSTATEDOWN(i)->sinc_len) * sizeof (float);
    GST_LOG ("downsample threads %d, blocks %d, amplifier %f", block.x, grid.x, SPSTATEDOWN(i)->amplifier);

    downsample2x <<<grid, block, share_mem_sz, stream>>> (SPSTATEDOWN(i)->amplifier,
						    2, 
						    SPSTATEDOWN(i)->d_sinc_table,
						    SPSTATEDOWN(i)->sinc_len, 
						    SPSTATEDOWN(i)->last_sample, 
						    SPSTATEDOWN(i)->d_mem, 
						    out_processed, 
						    pos_inqueue, 
						    pos_outqueue);

    gpuErrchk (cudaPeekAtLastError ());
    /* The following code is used for comparison with gstreamer downsampler2x 
     * quality=6; */

#if 0
//    if (i+1 == 2) {
	    int j;
	    float tmp_mem[num_inchunk + SPSTATEDOWN(i)->sinc_len -1 ];
	    float tmp[num_inchunk + SPSTATEDOWN(i)->sinc_len -1 ];
	    float tmp_sinc[SPSTATEDOWN(i)->sinc_len];
	    float tmp_q[num_inchunk];

	    printf ("depth %d, last sample %d\n", i, SPSTATEDOWN(i)->last_sample);
      	    cudaMemcpy(tmp_sinc, SPSTATEDOWN(i)->d_sinc_table, SPSTATEDOWN(i)->sinc_len * sizeof(float), cudaMemcpyDeviceToHost);
	//	for (j=0; j<SPSTATEDOWN(i)->sinc_len; j++)
	//	    printf("depth %d sinc [%d] = %e\n", i, j, tmp_sinc[j]);

      	    cudaMemcpy(tmp, pos_outqueue, out_processed * sizeof(float), cudaMemcpyDeviceToHost);
	    for (j=0; j<out_processed; j++)
		    printf("depth %d down out[%d] = %e\n", i+1, j, tmp[j]);

      	    cudaMemcpy(tmp_mem, SPSTATEDOWN(i)->d_mem, (num_inchunk + SPSTATEDOWN(i)->sinc_len -1 ) * sizeof(float), cudaMemcpyDeviceToHost);
      	    cudaMemcpy(tmp_q, pos_inqueue, (num_inchunk ) * sizeof(float), cudaMemcpyDeviceToHost);

	    for (j=0; j<(num_inchunk); j++) 
		    printf("depth %d mem[%d] = %f in %f\n", i, j, tmp_mem[j + SPSTATEDOWN(i)->sinc_len -1], tmp_q[j]);

            gpuErrchk (cudaPeekAtLastError ());

//    }
#endif

    /* 
     * the only possible situation to discard some samples is 
     * when at the end of a segment
     */

    /* 
     * if the number of input samples is odd, discard the last input 
     * sample. We do not expect this affect accuracy much.
     * if (num_inchunk % 2 == 1)
     * SPSTATE(i)->queue_eff_len -= 1;
     */

    /*
     * filter finish, update the next expected down start of upper 
     * spstate; update the effective length of this spstate;
     */
    SPSTATE(i)->queue_down_start = SPSTATE(i)->queue_eff_len;
    SPSTATEDOWN(i)->last_sample = 0 ;
    SPSTATE(i+1)->queue_eff_len += out_processed;
    num_inchunk = out_processed;
    GST_LOG ("%dth depth: queue eff len %d", i, SPSTATE(i)->queue_eff_len);
  }
  GST_LOG ("%dth depth: queue eff len %d", i, SPSTATE(i)->queue_eff_len);
  SPSTATE(num_depths-1)->queue_down_start = SPSTATE(num_depths-1)->queue_eff_len;
  GST_LOG ("multi downsample out processed %d samples", out_processed);

#if 0
  for (i=0; i<out_processed; i++) {
    printf ("in[%d] = %e\n", i, in_multidown[i]);
    printf ("out[%d] = %e\n", i, SPSTATE(num_depths-1)->queue[i]);
  }
#endif
  return out_processed;
}

void update_nb (SpiirState **spstate, gint new_processed, gint depth)
{
 
    if(new_processed != SPSTATE(depth)->pre_out_spiir_len)
    {
      // set nb
      guint nb = NB_MAX;
      if (SPSTATE(depth)->num_filters < NB_MAX) 
        nb = SPSTATE(depth)->num_filters;

      for (; nb > 0; --nb)
        if (new_processed % nb == 0)
          break;

      SPSTATE(depth)->nb = nb;
      SPSTATE(depth)->pre_out_spiir_len = new_processed;
    }
}


gint spiirup (SpiirState **spstate, gint num_in_multiup, gint num_depths, float *out, cudaStream_t stream)
{
  gint num_inchunk = num_in_multiup;

  gint i;
  // FIXME: 0 is used;

  /* 
   * SPIIR filter for the lowest depth 
   */

  GST_LOG ("spiirup %d samples", num_inchunk);

  i = num_depths - 1;
  gint num_left = SPSTATE(i)->queue_eff_len - num_inchunk;
  gint numThreads = MAX(num_inchunk, num_left);
  dim3 block(1, 1, 1);
  dim3 grid(1, 1, 1);
  block.x = MIN (THREADSPERBLOCK, num_inchunk);
  guint nb = numThreads % block.x == 0 ? numThreads/block.x : (int)numThreads/block.x + 1;
  uint share_mem_sz = num_left * sizeof (float);

  GST_LOG ("%dth depth: reload threads %d, nb %d, block.x %d, block.y %d, block.z %d, grid.x %d, grid.y %d, grid.z %d", i, numThreads, nb, block.x, block.y, block.z, grid.x, grid.y, grid.z);

  reload_queue_spiir <<<grid, block, share_mem_sz, stream>>> (SPSTATE(i)->d_queue,
				 SPSTATE(i)->d_queue_spiir,
				 num_inchunk,
				 num_left,
				 SPSTATE(i)->d_max + SPSTATE(i)->queue_spiir_last_sample,
				 SPSTATE(i)->queue_spiir_len);
  gpuErrchk (cudaPeekAtLastError ());

  SPSTATE(i)->queue_eff_len -= num_inchunk;
  SPSTATE(i)->queue_down_start -= num_inchunk;


  update_nb (spstate, num_inchunk, num_depths-1);
/*
 *	cuda kernel
 */
  block.x = SPSTATE(i)->num_filters;
  grid.y = SPSTATE(i)->num_templates;
  share_mem_sz = (block.x+8 + (SPSTATE(i)->num_filters+16-1)/16*SPSTATE(i)->nb) * 2 * sizeof(float);

  // using mutex to make sure that kernel launch is right after texture binding
  //g_mutex_lock(element->cuTex_lock);
  //Set up texture.
  /*cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>(); 
  texRef.addressMode[0] = cudaAddressModeWrap;
  texRef.filterMode	= cudaFilterModeLinear;
  texRef.normalized	= false;
  cudaBindTexture(0, texRef, SPSTATE(i)->d_input_s, channelDesc, available_length * sizeof(float));
  */

  GST_LOG ("%dth depth: processed %d, nb %d, SPIIR num of filters %d, number of templates %d, block.x %d, block.y %d, block.z %d, grid.x %d, grid.y %d, grid.z %d", i, num_inchunk, SPSTATE(i)->nb, SPSTATE(i)->num_filters, SPSTATE(i)->num_templates, block.x, block.y, block.z, grid.x, grid.y, grid.z);

  cuda_iir_filter_kernel<<<grid, block, share_mem_sz, stream>>>(SPSTATE(i)->d_a1,
							SPSTATE(i)->d_b0, 
							SPSTATE(i)->d_d, 
							SPSTATE(i)->d_y, 
						 	SPSTATE(i)->d_queue_spiir,
							SPSTATEUP(i)->d_mem, 
							SPSTATEUP(i)->mem_len,
							SPSTATEUP(i)->filt_len,
							SPSTATE(i)->d_max,
							num_inchunk, 
							SPSTATE(i)->nb,
							SPSTATE(i)->queue_spiir_last_sample,
							SPSTATE(i)->queue_spiir_len);
  //g_mutex_unlock(element->cuTex_lock);

  cudaDeviceSynchronize();
  gpuErrchk (cudaPeekAtLastError ());

  SPSTATE(i)->queue_spiir_last_sample = (SPSTATE(i)->queue_spiir_last_sample + num_inchunk) % SPSTATE(i)->queue_spiir_len;

#if 0
//    if (i+1 == 2) {
	    int j, k;
	    float tmp_mem[(SPSTATEUP(i)->mem_len)*SPSTATE(i)->num_templates*2];

      //	    cudaMemcpy(tmp_sinc, SPSTATEUP(i)->d_sinc_table, SPSTATEUP(i)->sinc_len * sizeof(float), cudaMemcpyDeviceToHost);
	//	for (j=0; j<SPSTATEDOWN(i)->sinc_len; j++)
	//	    printf("depth %d sinc [%d] = %e\n", i, j, tmp_sinc[j]);

      	    cudaMemcpy(tmp_mem, SPSTATEUP(i)->d_mem, (SPSTATEUP(i)->mem_len * SPSTATE(i)->num_templates*2) * sizeof(float), cudaMemcpyDeviceToHost);

	    for (j=0; j<(num_inchunk); j++) 
		    for (k=0; k<(SPSTATE(i)->num_templates*2); k++) 
		    printf("depth %d mem[%d, %d] = %e \n", i, k, j, tmp_mem[SPSTATEUP(i)->mem_len * k + SPSTATEUP(i)->filt_len -1 + j]);

            gpuErrchk (cudaPeekAtLastError ());

//    }
#endif

  GST_LOG ("%dth depth: queue eff len %d", i, SPSTATE(i)->queue_eff_len);
  gint resample_processed, spiir_processed;

  for (i=num_depths-2; i>=0; i--) {
    //g_assert ((SPSTATE(i-1)->queue_eff_len - SPSTATE(i-1)->queue_down_start >= num_inchunk;

    resample_processed = num_inchunk - SPSTATEUP(i+1)->last_sample;
    spiir_processed = resample_processed * 2;

    num_left = SPSTATE(i)->queue_eff_len - spiir_processed;
    numThreads = MAX(spiir_processed, num_left);
    block.x = MIN (THREADSPERBLOCK, spiir_processed);
    grid.x = 1;
    grid.y = 1;
    nb = numThreads % block.x == 0 ? numThreads/block.x : (int)numThreads/block.x + 1;

    share_mem_sz = num_left * sizeof (float);

    GST_LOG ("%dth depth: reload threads %d, nb %d, block.x %d, block.y %d, block.z %d, grid.x %d, grid.y %d, grid.z %d", i, numThreads, nb, block.x, block.y, block.z, grid.x, grid.y, grid.z);

    reload_queue_spiir <<<grid, block, share_mem_sz, stream>>> (SPSTATE(i)->d_queue,
							 SPSTATE(i)->d_queue_spiir,
							 spiir_processed,
							 num_left,
							 SPSTATE(i)->d_max + SPSTATE(i)->queue_spiir_last_sample,
							 SPSTATE(i)->queue_spiir_len);
    gpuErrchk (cudaPeekAtLastError ());

    SPSTATE(i)->queue_eff_len -= spiir_processed;
    SPSTATE(i)->queue_down_start -= spiir_processed;


    update_nb(spstate, spiir_processed, i);
  /*
   *	cuda kernel
   */
    block.x = SPSTATE(i)->num_filters;
    grid.y =  SPSTATE(i)->num_templates;
    share_mem_sz = (block.x+8 + (SPSTATE(i)->num_filters+16-1)/16*SPSTATE(i)->nb) * 2 * sizeof(float);

    GST_LOG ("%dth depth: up processed %d, nb %d, SPIIR num of filters %d, number of templates %d, block.x %d, block.y %d, block.z %d, grid.x %d, grid.y %d, grid.z %d", i, spiir_processed, SPSTATE(i)->nb, SPSTATE(i)->num_filters, SPSTATE(i)->num_templates, block.x, block.y, block.z, grid.x, grid.y, grid.z);
    // using mutex to make sure that kernel launch is right after texture binding
    //g_mutex_lock(element->cuTex_lock);
    //Set up texture.
    /*cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>(); 
    texRef.addressMode[0] = cudaAddressModeWrap;
    texRef.filterMode	= cudaFilterModeLinear;
    texRef.normalized	= false;
    cudaBindTexture(0, texRef, SPSTATE(i)->d_input_s, channelDesc, available_length * sizeof(float));
    */
#if 0
    if (SPSTATE(i)->num_filters > 3) 
#endif
    {
    cuda_iir_filter_kernel<<<grid, block, share_mem_sz, stream>>>(SPSTATE(i)->d_a1,
							SPSTATE(i)->d_b0, 
							SPSTATE(i)->d_d, 
							SPSTATE(i)->d_y, 
						 	SPSTATE(i)->d_queue_spiir,
							SPSTATEUP(i)->d_mem, 
							SPSTATEUP(i)->mem_len,
							SPSTATEUP(i)->filt_len,
							SPSTATE(i)->d_max,
							spiir_processed, 
							SPSTATE(i)->nb,
							SPSTATE(i)->queue_spiir_last_sample,
							SPSTATE(i)->queue_spiir_len);
    }
    //g_mutex_unlock(element->cuTex_lock);

  

    gpuErrchk (cudaPeekAtLastError ());

    SPSTATE(i)->queue_spiir_last_sample = (SPSTATE(i)->queue_spiir_last_sample + spiir_processed) % SPSTATE(i)->queue_spiir_len;

    block.x = MIN(THREADSPERBLOCK, resample_processed);
    nb = resample_processed % block.x == 0 ? resample_processed/block.x : (int)resample_processed/block.x + 1;
    grid.y = SPSTATEUP(i)->channels;
    share_mem_sz = SPSTATEUP(i)->sinc_len * sizeof(float);

    GST_LOG("depth %d, queue_spiir_last_sample %d\n", i, SPSTATE(i)->queue_spiir_last_sample);
    GST_LOG ("%dth depth: resample processed %d, block.x %d, block.y %d, block.z %d, grid.x %d, grid.y %d, grid.z %d, channels %d", i, resample_processed, block.x, block.y, block.z, grid.x, grid.y, grid.z, SPSTATEUP(i)->channels);
    /*
     * upsample 2x and add 
     */

    upsample2x_and_add <<<grid, block, share_mem_sz, stream>>>(SPSTATEUP(i+1)->d_sinc_table, 
					SPSTATEUP(i+1)->filt_len, 
					SPSTATEUP(i+1)->last_sample, 
					resample_processed, 
					SPSTATEUP(i+1)->d_mem, 
					SPSTATEUP(i)->d_mem,
					SPSTATEUP(i+1)->mem_len, 
					SPSTATEUP(i)->mem_len);

    gpuErrchk (cudaPeekAtLastError ());
    SPSTATEUP(i+1)->last_sample = 0;
    num_inchunk = spiir_processed; 
    /* The following code is used for comparison with gstreamer downsampler2x 
     * quality=6; */
#if 0
    if (i == 2) {
	    int tmp_len = (SPSTATEUP(i)->mem_len) * SPSTATEUP(i)->channels;
	    float tmp[tmp_len];
      	    cudaMemcpy(tmp, SPSTATEUP(i)->d_mem, tmp_len * sizeof(float), cudaMemcpyDeviceToHost);
	    for (int j=0; j<spiir_processed; j++)
	    for (int k=0; j<SPSTATEUP(i)->channels; j++)
		    printf("depth %d up out channel %d, [%d] = %e\n", i, j, tmp[SPSTATEUP(i)->mem_len * k + SPSTATEUP(i)->filt_len-1 + j]);
//      	    cudaMemcpy(tmp, SPSTATEDOWN(i)->d_mem, (SPSTATEDOWN(i)->sinc_len -1 + out_processed) * sizeof(float), cudaMemcpyDeviceToHost);
//	    for (int j=0; j<(SPSTATEDOWN(i)->sinc_len -1+ out_processed); j++)
//		    printf("depth %d mem[%d] = %e\n", i, j, tmp[j]);

    }
#endif


  }
  num_remains = SPSTATE(0)->queue_eff_len - num_inchunk;
  cudaMemcpy(SPSTATE(0)->d_queue, SPSTATE(0)->d_queue + num_inchunk, num_remains * sizeof(float), cudaMemcpyDeviceToDevice);

 
  GST_LOG ("spiirup out processed %d samples", num_inchunk);
  cudaMemcpyAsync(out, SPSTATEUP(0)->d_mem,  SPSTATEUP(0)->channels * (SPSTATEUP(0)->mem_len) * sizeof(float), cudaMemcpyDeviceToHost, stream);
  GST_LOG ("out %f d mem addr %p ", out[0], SPSTATEUP(0)->d_mem);
  gpuErrchk (cudaPeekAtLastError ());
  return spiir_processed;
}

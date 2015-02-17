/* GStreamer
 * Copyright (C) 2014 Qi Chu, NIMS, Yuan Liu
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more deroll-offss.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */


/*
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


#ifdef __cplusplus
extern "C" {
#endif
#include <stdio.h>
#include <glib.h>
#include <gst/gst.h>


#include "multiratespiir.h"
#include "multiratespiir_utils.h"
#include "spiir_state_macro.h"
#include <cuda_debug.h>
#ifdef __cplusplus
}
#endif

#define THREADSPERBLOCK 256
#define NB_MAX 32

// #define ORIGINAL
#define CUT_FILTERS 0 // set to 0 to keep all the filtering results

// for gpu debug
#define gpuErrchk(stream) { gpuAssert(stream, __FILE__, __LINE__); }
static void gpuAssert(cudaStream_t stream, char *file, int line, bool abort=true)
{
    cudaStreamSynchronize(stream);
    cudaError_t code = cudaPeekAtLastError ();

   if (code != cudaSuccess) 
   {
      printf ("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
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
			      const int last_sample_in,
			      const int len_in,
			      float *queue_out,
			      const int last_sample_out,
			      const int len_out
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
		in_start = 2 * bx * tdx + last_sample_in;
		for (i=tx;  i<filt_len-1; i+=tdx) 
			tmp_mem[i] = queue_in[ (in_start - (filt_len - 1) + i) % len_in] ;
	}

	if (tid < len) {
	/*
	 * grab filt_len-1 data,
	 * for the first block, grab data from mem 
	 */


		tmp_mem[tx2 + filt_len - 1] = queue_in[(tid2 + last_sample_in) % len_in];
		tmp_mem[tx2 + filt_len]  = queue_in[(tid2 + 1 + last_sample_in) % len_in];

	}
	if (bx == bdx - 1) {
		tmp_mem_end = (len - bx * tdx) * 2;
		in_end = len * 2 + last_sample_in;
	}
	else {
		tmp_mem_end = tdx2;
		in_end = 2*(tdx + bx*tdx) + last_sample_in;
	}


	for (i=tx;  i<tail_len; i+=tdx) 
		tmp_mem[tmp_mem_end + filt_len - 1 + i] = queue_in[(in_end + i) % len_in];
	__syncthreads();	

	if (tid < len) {

		for (int j = 0; j < filt_len; ++j) {
		  tmp += tmp_mem[tx2 + last_sample + j] * tmp_sinc[j];
		}

		queue_out[(tid + last_sample_out) % len_out] = tmp * amplifier;
//		printf("in device %d, mem %f in %f\n", tx_loc+1, mem[filt_len-1+tx_loc+1], queue_in[tx_loc+1]);
	}


	// copy last to first (filt_len-1) mem data
	if (bx == bdx - 1) {
		for (i=tx;  i<filt_len - 1; i+=tdx) {
			mem[i] = tmp_mem[tmp_mem_end + last_sample + i];
		}
	}
}

/*
 * cuda filter kernel
 */

//texture<float, 1, cudaReadModeElementType> texRef;

#define GRANU       32
#define LOGGRANU    5

#define WARPSIZE	32
#define LOGWARPSIZE	5

__global__ void cuda_iir_filter_kernel_coarse
(
	COMPLEX_F *cudaA1, 
	COMPLEX_F *cudaB0, 
	int *cudaShift,
	COMPLEX_F *cudaPrevSnr, 
	float *cudaData, 
	float *cudaSnr,
	gint mem_len, 
	gint filt_len, 
	gint delay_max,
	gint len, 
	gint queue_first_sample, 
	gint queue_len,
	gint numFilters,
	gint numTemplates	
)
{
	// This is a coarse-grained version of the spiir kernel, it's used to calculate SNRs
	// where number of filters of each template (numFilters) is larger than 32.
    // 1 block is assigned to calculate 1 template, 1 template has numFilters filters.
	// There are numTemplate blocks launched, each block has at least numFilters threads,
	// and block size is a multiple of WARPSIZE

    int tLane = threadIdx.x & (WARPSIZE - 1);

    float fltrOutptReal;
    float fltrOutptImag;
    float data;
    int shift;

    COMPLEX_F a1, b0;
    COMPLEX_F previousSnr;

    a1.re = 0.0f;
    a1.im = 0.0f;
    b0.re = 0.0f;
    b0.im = 0.0f;
    shift = 0;
    previousSnr.re = 0.0f;
    previousSnr.im = 0.0f;
    fltrOutptReal = 0.0f;
    fltrOutptImag = 0.0f;

    if (threadIdx.x < numFilters)
    {
        a1 = cudaA1[blockIdx.x * numFilters + threadIdx.x];
        b0 = cudaB0[blockIdx.x * numFilters + threadIdx.x];
        shift = delay_max - cudaShift[blockIdx.x * numFilters + threadIdx.x];
        previousSnr = cudaPrevSnr[blockIdx.x * numFilters + threadIdx.x];
    }

    for (int i = 0; i < len; ++i)
    {
        data = cudaData[(shift + i + queue_first_sample) % queue_len];

        fltrOutptReal = a1.re * previousSnr.re - a1.im * previousSnr.im + b0.re * data;
        fltrOutptImag = a1.re * previousSnr.im + a1.im * previousSnr.re + b0.im * data;

        previousSnr.re = fltrOutptReal; 
        previousSnr.im = fltrOutptImag; 

        // Inter-Warp Reduction
        for (int off = WARPSIZE >> 1; off > 0; off = off >> 1)
        {
            fltrOutptReal += __shfl_down(fltrOutptReal, off, 2 * off);
            fltrOutptImag += __shfl_down(fltrOutptImag, off, 2 * off);
        }

        if (tLane == 0)
        {
            atomicAdd(cudaSnr + (2 * blockIdx.x + 0) * mem_len + filt_len - 1 + i, fltrOutptReal);
            atomicAdd(cudaSnr + (2 * blockIdx.x + 1) * mem_len + filt_len - 1 + i, fltrOutptImag);
        }
    }

    if (threadIdx.x < numFilters)
        cudaPrevSnr[blockIdx.x * numFilters + threadIdx.x] = previousSnr;
}

__global__ void cuda_iir_filter_kernel_fine
(
	COMPLEX_F *cudaA1, 
	COMPLEX_F *cudaB0, 
	int *cudaShift,
	COMPLEX_F *cudaPrevSnr, 
	float *cudaData, 
	float *cudaSnr,
	gint mem_len, 
	gint filt_len, 
	gint delay_max,
	gint len, 
	gint queue_first_sample, 
	gint queue_len,
	gint numFilters,
	gint numTemplates,	
	gint cu,
	gint logcu
)
{
	// This is a fine-grained version of the spiir kernel, it's used to calculate SNRs
	// where the number of filters per template is less than or equal to 32.
	// 1 compute unit (CU) is used to calculate 1 template, the value of CU is specified
	// by variable "cu", "logcu" is log2(cu)

    int tLane   = threadIdx.x & (WARPSIZE - 1);
    int cuIdG   = (threadIdx.x + blockIdx.x * blockDim.x) >> logcu;
    int tIdL    = threadIdx.x & (cu - 1);

    float fltrOutptReal;
    float fltrOutptImag;
    int shift;
    float data;

    COMPLEX_F a1, b0;
    COMPLEX_F previousSnr;

    a1.re = 0.0f;
    a1.im = 0.0f;
    b0.re = 0.0f;
    b0.im = 0.0f;
    shift = 0;
    previousSnr.re = 0.0f;
    previousSnr.im = 0.0f;
    fltrOutptReal = 0.0f;
    fltrOutptImag = 0.0f;

	// Total number of CUs should exceed numTemplates
    if (cuIdG < numTemplates)
    {
        if (tIdL < numFilters)
        {
            a1 = cudaA1[cuIdG * numFilters + tIdL]; 
            b0 = cudaB0[cuIdG * numFilters + tIdL];
            shift = delay_max - cudaShift[cuIdG * numFilters + tIdL];
            previousSnr = cudaPrevSnr[cuIdG * numFilters + tIdL];
        }

        for (int i = 0; i < len; ++i)
        {
            data = cudaData[(shift + i + queue_first_sample) % queue_len];

            fltrOutptReal = a1.re * previousSnr.re - a1.im * previousSnr.im + b0.re * data;
            fltrOutptImag = a1.re * previousSnr.im + a1.im * previousSnr.re + b0.im * data;

            previousSnr.re = fltrOutptReal;
            previousSnr.im = fltrOutptImag;

            // Inter-CU Reduction
            for (int off = cu >> 1; off > 0; off = off >> 1)
            {
                fltrOutptReal += __shfl(fltrOutptReal, tLane + off, 2 * off);
                fltrOutptImag += __shfl(fltrOutptImag, tLane + off, 2 * off);
            }

            if (tIdL == 0)
            {
                cudaSnr[(2 * cuIdG + 0) * mem_len + filt_len - 1 + i] = fltrOutptReal;
                cudaSnr[(2 * cuIdG + 1) * mem_len + filt_len - 1 + i] = fltrOutptImag;
            }
        }

        if (tIdL < numFilters)
        {
            cudaPrevSnr[cuIdG * numFilters + tIdL] = previousSnr;
        }
    }
}

__global__ void cuda_iir_filter_kernel( COMPLEX_F *cudaA1, 
					COMPLEX_F *cudaB0, int *cudaShift, 
					COMPLEX_F *cudaPrevSnr,
					float *cudaData, 
					float *cudaSnr, gint mem_len, 
					gint filt_len, gint delay_max,
					gint len, 
					guint nb, 
					gint queue_first_sample,
					gint queue_len)
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
//	volatile float *fltrOutptImag = &(fltrOutptReal[numFilters+8]);
//	float *grpOutptReal = (float *)&(fltrOutptImag[numFilters+8]);	

	volatile float *fltrOutptImag = &(fltrOutptReal[numFilters+8]);
	float *grpOutptReal = (float *)&(fltrOutptImag[numFilters+8]);	
	float *grpOutptImag = &(grpOutptReal[numSixtnGrp*nb]);
	
	unsigned int tx_2 = tx%16;
	unsigned int tx_3 = tx/16;
//	for (i = tx; i < 8; i += threads)
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

		for( i = 0; i < len; i+=nb )
		{
			for(j = 0; j < nb; ++j)
			{ 
				//data = 0.01f;
				//data = tex1Dfetch(texRef, shift+i+j);		//use texture, abandon now
				data = cudaData[(shift+i+j+queue_first_sample)%queue_len];
//				printf ("channelid %d, data %d %f\n", by, i+j, data);
				fltrOutptReal[tx] = a1.re * previousSnr.re - a1.im * previousSnr.im + b0.re * data;
		 
				fltrOutptImag[tx] = a1.re * previousSnr.im + a1.im * previousSnr.re + b0.im * data;
			 
		//		__syncthreads();
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
#if 0
					for (int iter=1; iter<16; iter++) {
						fltrOutptReal[tx] += fltrOutptReal[tx + iter];
						fltrOutptImag[tx] += fltrOutptImag[tx + iter];
					}
#endif
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
#if 0
  /* the following is to compare result with cpu */
  if (tx < 1 && by < 1)
    printf("delay_max, %d, snr.real[0] %.10f, snr.imag[0] %.10f, nb %f\n", delay_max, snr_real[0], snr_imag[0], (float)nb);
#endif
}



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

	__syncthreads();

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

#if 0
	if (tx < 1 && by < 1)
		printf("mem_out_len %d, mem_in[filt_len-1] %f, mem_out[filt_len-1] %f\n", mem_out_len, mem_in[filt_len-1], mem_out[filt_len-1]);
#endif 


	// copy last to first (filt_len-1) mem data
	for (i=tx; i<filt_len - 1; i+=tdx) {
		in = &(mem_in[mem_in_len * by + i]);
		in[0] = in[len];
	}
}


gint multi_downsample (SpiirState **spstate, float *in_multidown, gint num_in_multidown, guint num_depths, cudaStream_t stream)
{
  float *pos_inqueue, *pos_outqueue;
  gint i, out_processed;
  gint num_inchunk = num_in_multidown;

  GST_LOG ("multidownsample: start. in %d samples\n", num_inchunk);
  /* make sure that unspiired samples + incoming samples won't exceed the physical queue length */
 // g_assert (SPSTATE(0)->queue_last_sample + num_inchunk <= SPSTATE(0)->queue_len);

  /* 
   * copy inbuf data to the end of queue
   */


  if (SPSTATE(0)->queue_last_sample + num_inchunk <= SPSTATE(0)->queue_len) {

    pos_inqueue = SPSTATE(0)->d_queue + SPSTATE(0)->queue_last_sample;
    CUDA_CHECK(cudaMemcpyAsync(pos_inqueue, in_multidown, num_inchunk * sizeof(float), cudaMemcpyHostToDevice, stream));

  } else {

    int num_tail_samples = SPSTATE(0)->queue_len - SPSTATE(0)->queue_last_sample;
    pos_inqueue = SPSTATE(0)->d_queue + SPSTATE(0)->queue_last_sample;
    CUDA_CHECK(cudaMemcpyAsync(pos_inqueue, in_multidown, num_tail_samples * sizeof(float), cudaMemcpyHostToDevice, stream));

    int num_head_samples = num_inchunk - num_tail_samples;
    pos_inqueue = SPSTATE(0)->d_queue;
    CUDA_CHECK(cudaMemcpyAsync(pos_inqueue, in_multidown + num_tail_samples, num_head_samples * sizeof(float), cudaMemcpyHostToDevice, stream));
  }

  /* the following parameters should be updated each time of downsample :
   * queue_last_sample,
   * last_sample.
   */


  for (i=0; i<num_depths-1; i++) {

    /* predicted output length of downsample this round,
     * we already ganrantee earlier that the length in samples 
     * will be even 
     */
    out_processed = (num_inchunk - SPSTATEDOWN(i)->last_sample)/2;
    /*
     * downsample 2x of number of filt samples
     */

    /* make sure lower depth mem is large enough to store queue data. */
    g_assert (num_inchunk <= SPSTATEDOWN(i)->mem_len - SPSTATEDOWN(i)->filt_len + 1 );

    pos_inqueue = SPSTATE(i)->d_queue; 
    pos_outqueue = SPSTATE(i+1)->d_queue;

    /* 
     * CUDA downsample2x 
     */

    
    dim3 block(1, 1, 1);
    dim3 grid(1, 1, 1);
    block.x = MIN(THREADSPERBLOCK, out_processed); 
    grid.x = out_processed % block.x == 0 ? out_processed/block.x : (int)out_processed/block.x + 1;

    uint share_mem_sz = (2 * block.x + 4 * SPSTATEDOWN(i)->sinc_len) * sizeof (float);
    GST_LOG ("downsample: threads %d, blocks %d, amplifier %f, share_mem_sz %d", block.x, grid.x, SPSTATEDOWN(i)->amplifier, share_mem_sz);

    downsample2x <<<grid, block, share_mem_sz, stream>>> (SPSTATEDOWN(i)->amplifier,
						    2, 
						    SPSTATEDOWN(i)->d_sinc_table,
						    SPSTATEDOWN(i)->sinc_len, 
						    SPSTATEDOWN(i)->last_sample, 
						    SPSTATEDOWN(i)->d_mem, 
						    out_processed, 
						    pos_inqueue,
						    SPSTATE(i)->queue_last_sample,
						    SPSTATE(i)->queue_len,
						    pos_outqueue,
						    SPSTATE(i+1)->queue_last_sample,
						    SPSTATE(i+1)->queue_len
						    );

 //   gpuErrchk (stream);

    /* 
     * FIXME: the only possible situation to discard some samples is 
     * when at the end of a segment. BY tests, this situation could be 
     * ignorable.
     * if the number of input samples is odd, discard the last input 
     * sample. We do not expect this affect accuracy much.
     */
   if (num_inchunk % 2 == 1)
    SPSTATE(i)->queue_last_sample = (SPSTATE(i)->queue_last_sample + num_inchunk - 1) % SPSTATE(i)->queue_len;
   else
    SPSTATE(i)->queue_last_sample = (SPSTATE(i)->queue_last_sample + num_inchunk) % SPSTATE(i)->queue_len;

    /*
     * filter finish, update the next expected down start of upper 
     * spstate; update the effective length of this spstate;
     */
    SPSTATEDOWN(i)->last_sample = 0 ;
    num_inchunk = out_processed;
  }
  SPSTATE(num_depths - 1)->queue_last_sample = (SPSTATE(i)->queue_last_sample + out_processed) % SPSTATE(i)->queue_len;
  GST_LOG ("multidownsample: finished. out processed %d samples", out_processed);

#if 0
  for (i=0; i<out_processed; i++) {
    printf ("in[%d] = %e\n", i, in_multidown[i]);
    printf ("out[%d] = %e\n", i, SPSTATE(num_depths-1)->queue[i]);
  }
#endif
  return out_processed;
}

void update_nb (SpiirState **spstate, gint new_processed, guint depth)
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


gint spiirup (SpiirState **spstate, gint num_in_multiup, guint num_depths, float *out, cudaStream_t stream)
{
  gint num_inchunk = num_in_multiup;

  gint i;
  // FIXME: 0 is used;

  /* 
   * SPIIR filter for the lowest depth 
   */

  GST_LOG ("spiirup: start. in %d samples\n", num_inchunk);

  i = num_depths - 1;

  update_nb (spstate, num_inchunk, num_depths-1);
/*
 *	cuda kernel
 */

  dim3 block(1, 1, 1);
  dim3 grid(1, 1, 1);
  uint share_mem_sz;

  if (SPSTATE(i)->num_filters > CUT_FILTERS) {

  #ifdef ORIGINAL
  block.x = SPSTATE(i)->num_filters;
  grid.y = SPSTATE(i)->num_templates;
//  share_mem_sz = (block.x+8 + (SPSTATE(i)->num_filters+16-1)/16*SPSTATE(i)->nb) * 2 * sizeof(float);
  share_mem_sz = (block.x+32 + (SPSTATE(i)->num_filters+16-1)/16*SPSTATE(i)->nb) * 2 * sizeof(float);

  // using mutex to make sure that kernel launch is right after texture binding
  //g_mutex_lock(element->cuTex_lock);
  //Set up texture.
  /*cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>(); 
  texRef.addressMode[0] = cudaAddressModeWrap;
  texRef.filterMode	= cudaFilterModeLinear;
  texRef.normalized	= false;
  cudaBindTexture(0, texRef, SPSTATE(i)->d_input_s, channelDesc, available_length * sizeof(float));
  */

  GST_LOG ("spiir_kernel: depth %d. processed %d, nb %d, num of (templates: %d,filters: %d). block.size (%d, %d, %d), grid.size (%d, %d, %d)", i, num_inchunk, SPSTATE(i)->nb, SPSTATE(i)->num_templates, SPSTATE(i)->num_filters, block.x, block.y, block.z, grid.x, grid.y, grid.z);

  cuda_iir_filter_kernel<<<grid, block, share_mem_sz, stream>>>(SPSTATE(i)->d_a1,
							SPSTATE(i)->d_b0, 
							SPSTATE(i)->d_d, 
							SPSTATE(i)->d_y, 
						 	SPSTATE(i)->d_queue,
							SPSTATEUP(i)->d_mem, 
							SPSTATEUP(i)->mem_len,
							SPSTATEUP(i)->filt_len,
							SPSTATE(i)->delay_max,
							num_inchunk, 
							SPSTATE(i)->nb,
							SPSTATE(i)->queue_first_sample,
							SPSTATE(i)->queue_len);
  //g_mutex_unlock(element->cuTex_lock);

//  gpuErrchk (stream);
  #else
    GST_LOG ("spiir_kernel: depth %d. processed %d, nb %d, num of (templates: %d,filters: %d). block.size (%d, %d, %d), grid.size (%d, %d, %d)", i, num_inchunk, SPSTATE(i)->nb, SPSTATE(i)->num_templates, SPSTATE(i)->num_filters, block.x, block.y, block.z, grid.x, grid.y, grid.z);
	if (SPSTATE(i)->num_filters > 32)
	{
		// Use Coarse-Grained Kernel
		int numTemplates = SPSTATE(i)->num_templates;
		int numFilters = SPSTATE(i)->num_filters;
		int mem_len = SPSTATEUP(i)->mem_len;
		int spiir_grid	= numTemplates;	
		int spiir_block	= ((numFilters + WARPSIZE - 1) >> LOGWARPSIZE) * WARPSIZE;
        CUDA_CHECK(cudaMemset(SPSTATEUP(i)->d_mem, 0, sizeof(COMPLEX_F) * mem_len * numTemplates));
		cuda_iir_filter_kernel_coarse<<<spiir_grid, spiir_block, 0, stream>>>
		(
			SPSTATE(i)->d_a1,
			SPSTATE(i)->d_b0,
			SPSTATE(i)->d_d,
			SPSTATE(i)->d_y,
			SPSTATE(i)->d_queue,
			SPSTATEUP(i)->d_mem,
			SPSTATEUP(i)->mem_len,
			SPSTATEUP(i)->filt_len,
			SPSTATE(i)->delay_max,
			num_inchunk, 
			SPSTATE(i)->queue_first_sample,
			SPSTATE(i)->queue_len,
			numFilters,
			numTemplates
		);
	}
	else
	{
		// Use Fine-Grained Kernel
		int numTemplates = SPSTATE(i)->num_templates;
		int numFilters = SPSTATE(i)->num_filters;
		int mem_len = SPSTATEUP(i)->mem_len;
		int t = numFilters - 1;
		int c = 0;
		while (t >>= 1)
		{
			++c;
		}
		int cu = 2 << c;
		int logcu = c + 1;
		int spiir_block = 512;
        int cuPerBlock = spiir_block >> logcu;
        int spiir_grid = (numTemplates + cuPerBlock - 1) / cuPerBlock;
		cuda_iir_filter_kernel_fine<<<spiir_grid, spiir_block, 0, stream>>>
		(
			SPSTATE(i)->d_a1,
			SPSTATE(i)->d_b0,
			SPSTATE(i)->d_d,
			SPSTATE(i)->d_y,
			SPSTATE(i)->d_queue,
			SPSTATEUP(i)->d_mem,
			SPSTATEUP(i)->mem_len,
			SPSTATEUP(i)->filt_len,
			SPSTATE(i)->delay_max,
			num_inchunk,
			SPSTATE(i)->queue_first_sample,
			SPSTATE(i)->queue_len,
			numFilters,
			numTemplates,
			cu,
			logcu
		);
	}	
  #endif

  }
  else {
          CUDA_CHECK(cudaMemsetAsync(SPSTATEUP(i)->d_mem, 0, sizeof(COMPLEX_F) * SPSTATEUP(i)->mem_len * SPSTATE(i)->num_templates, stream));
  }

  SPSTATE(i)->queue_first_sample = (SPSTATE(i)->queue_first_sample + num_inchunk) % SPSTATE(i)->queue_len;

  gint resample_processed, spiir_processed;

  for (i=num_depths-2; i>=0; i--) {

    resample_processed = num_inchunk - SPSTATEUP(i+1)->last_sample;
    spiir_processed = resample_processed * 2;

    update_nb(spstate, spiir_processed, i);
    if (SPSTATE(i)->num_filters > CUT_FILTERS) 
    {

  /*
   *	cuda kernel
   */

	#ifdef ORIGINAL 
    block.x = SPSTATE(i)->num_filters;
    grid.y =  SPSTATE(i)->num_templates;
   // share_mem_sz = (block.x+8 + (SPSTATE(i)->num_filters+16-1)/16*SPSTATE(i)->nb) * 2 * sizeof(float);
    share_mem_sz = (block.x+32 + (SPSTATE(i)->num_filters+16-1)/16*SPSTATE(i)->nb) * 2 * sizeof(float);

    GST_LOG ("spiir_kernel: depth %d. processed %d, nb %d, num of (templates: %d,filters: %d). block.size (%d, %d, %d), grid.size (%d, %d, %d)", i, num_inchunk, SPSTATE(i)->nb, SPSTATE(i)->num_templates, SPSTATE(i)->num_filters, block.x, block.y, block.z, grid.x, grid.y, grid.z);
    // using mutex to make sure that kernel launch is right after texture binding
    //g_mutex_lock(element->cuTex_lock);
    //Set up texture.
    /*cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>(); 
    texRef.addressMode[0] = cudaAddressModeWrap;
    texRef.filterMode	= cudaFilterModeLinear;
    texRef.normalized	= false;
    cudaBindTexture(0, texRef, SPSTATE(i)->d_input_s, channelDesc, available_length * sizeof(float));
    */
    cuda_iir_filter_kernel<<<grid, block, share_mem_sz, stream>>>(SPSTATE(i)->d_a1,
							SPSTATE(i)->d_b0, 
							SPSTATE(i)->d_d, 
							SPSTATE(i)->d_y, 
						 	SPSTATE(i)->d_queue,
							SPSTATEUP(i)->d_mem, 
							SPSTATEUP(i)->mem_len,
							SPSTATEUP(i)->filt_len,
							SPSTATE(i)->delay_max,
							spiir_processed, 
							SPSTATE(i)->nb,
							SPSTATE(i)->queue_first_sample,
							SPSTATE(i)->queue_len);
	#else
    GST_LOG ("spiir_kernel: depth %d. processed %d, nb %d, num of (templates: %d,filters: %d). block.size (%d, %d, %d), grid.size (%d, %d, %d)", i, num_inchunk, SPSTATE(i)->nb, SPSTATE(i)->num_templates, SPSTATE(i)->num_filters, block.x, block.y, block.z, grid.x, grid.y, grid.z);
	if (SPSTATE(i)->num_filters > 32)
	{
		// Use Coarse-Grained Kernel
		int numTemplates = SPSTATE(i)->num_templates;
		int numFilters = SPSTATE(i)->num_filters;
		int mem_len = SPSTATEUP(i)->mem_len;
		int spiir_grid	= numTemplates;	
		int spiir_block	= ((numFilters + WARPSIZE - 1) >> LOGWARPSIZE) * WARPSIZE;
        CUDA_CHECK(cudaMemset(SPSTATEUP(i)->d_mem, 0, sizeof(COMPLEX_F) * mem_len * numTemplates));
		cuda_iir_filter_kernel_coarse<<<spiir_grid, spiir_block, 0, stream>>>
		(
			SPSTATE(i)->d_a1,
			SPSTATE(i)->d_b0,
			SPSTATE(i)->d_d,
			SPSTATE(i)->d_y,
			SPSTATE(i)->d_queue,
			SPSTATEUP(i)->d_mem,
			SPSTATEUP(i)->mem_len,
			SPSTATEUP(i)->filt_len,
			SPSTATE(i)->delay_max,
			spiir_processed,
			SPSTATE(i)->queue_first_sample,
			SPSTATE(i)->queue_len,
			numFilters,
			numTemplates
		);
	}
	else
	{
		// Use Fine-Grained Kernel
		int numTemplates = SPSTATE(i)->num_templates;
		int numFilters = SPSTATE(i)->num_filters;
		int mem_len = SPSTATEUP(i)->mem_len;
		int t = numFilters - 1;
		int c = 0;
		while (t >>= 1)
		{
			++c;
		}
		int cu = 2 << c;
		int logcu = c + 1;
		int spiir_block = 512;
        int cuPerBlock = spiir_block >> logcu;
        int spiir_grid = (numTemplates + cuPerBlock - 1) / cuPerBlock;
		cuda_iir_filter_kernel_fine<<<spiir_grid, spiir_block, 0, stream>>>
		(
			SPSTATE(i)->d_a1,
			SPSTATE(i)->d_b0,
			SPSTATE(i)->d_d,
			SPSTATE(i)->d_y,
			SPSTATE(i)->d_queue,
			SPSTATEUP(i)->d_mem,
			SPSTATEUP(i)->mem_len,
			SPSTATEUP(i)->filt_len,
			SPSTATE(i)->delay_max,
			spiir_processed,
			SPSTATE(i)->queue_first_sample,
			SPSTATE(i)->queue_len,
			numFilters,
			numTemplates,
			cu,
			logcu
		);
	}	
	#endif
    }
  else {
          CUDA_CHECK(cudaMemsetAsync(SPSTATEUP(i)->d_mem, 0, sizeof(COMPLEX_F) * SPSTATEUP(i)->mem_len * SPSTATE(i)->num_templates, stream));
  }

    //g_mutex_unlock(element->cuTex_lock);

  //  gpuErrchk (stream);


    SPSTATE(i)->queue_first_sample = (SPSTATE(i)->queue_first_sample + spiir_processed) % SPSTATE(i)->queue_len;

    block.x = MIN(THREADSPERBLOCK, resample_processed);
    grid.y = SPSTATEUP(i)->channels;
    share_mem_sz = SPSTATEUP(i)->sinc_len * sizeof(float);

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

  //  gpuErrchk (stream);
    SPSTATEUP(i+1)->last_sample = 0;
    num_inchunk = spiir_processed; 

  }

 
  CUDA_CHECK(cudaMemcpyAsync(out, SPSTATEUP(0)->d_mem,  SPSTATEUP(0)->channels * (SPSTATEUP(0)->mem_len) * sizeof(float), cudaMemcpyDeviceToHost, stream));
//  gpuErrchk (stream);
  return spiir_processed;
}


#ifdef __cplusplus
extern "C" {
#endif
#include <glib.h>
#include <gst/gst.h>


#include "multiratespiir.h"
#include "multiratespiir_utils.h"
#include "spiir_state_macro.h"

#ifdef __cplusplus
}
#endif

#define THREADSPERBLOCK 256

__global__ void downsample2x (const float amplifier,
			      const int times,
  			      float *sinc, 
			      const int sinc_len, 
			      int last_sample,
			      float *mem, 
			      const int len, 
			      float *queue_in, 
			      float *queue_out)
{



	float tmp = 0.0;
	int len_loc = len * times;
	int tx_loc = 0;
	int tx = threadIdx.x + blockIdx.x * blockDim.x;

	// grab data from queue
	if (tx < len) {
		tx_loc = tx * times;
		mem[sinc_len-1 + tx_loc] = queue_in[tx_loc];
		mem[sinc_len-1 + tx_loc + 1] = queue_in[tx_loc + 1 ];
		if (tx < last_sample)
			mem[sinc_len-1 +  len_loc + tx] = 
				queue_in[ len_loc + tx];

		__syncthreads();
		for (int j = 0; j < sinc_len; ++j) {
		  tmp += mem[tx_loc + j + last_sample] * sinc[j];
		}

		queue_out[tx] = tmp;
	}
	__syncthreads();
	

	// copy last to first (sinc_len-1) mem data
	if (tx < sinc_len -1)
		mem[tx] = mem[tx + len_loc + last_sample];
}


__global__ void upsample2x_and_add (const int resolution,
			      const int times,
  			      float *sinc, 
			      const int filt_len, 
			      int last_sample,
			      const int len, 
			      float *mem_in, 
			      float *mem_out)
{
	float tmp = 0.0;
	int tx = threadIdx.x + blockIdx.x * blockDim.x;

	for (int j = 0; j < filt_len; ++j)
		tmp += mem_in[(tx * 2 + 1) / times + j] * sinc[j + filt_len];

	mem_out[ tx * 2] += mem_in[tx];
	mem_out[tx * 2 + 1] += tmp;

	// copy last to first (filt_len-1) mem data
	if (tx < filt_len -1)
		mem_in[tx] = mem_in[tx + len];

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


gint multi_downsample (SpiirState **spstate, float *in_multidown, gint num_in_multidown, gint num_depths)
{
  float *pos_inqueue, *pos_outqueue;
  gint i, out_processed;
  gint num_inchunk = num_in_multidown;
  int threadsPerBlock, numBlocks;

  GST_LOG ("multi downsample %d samples", num_inchunk);
  g_assert (SPSTATE(0)->queue_eff_len + num_inchunk <= SPSTATE(0)->queue_len);
  pos_inqueue = SPSTATE(0)->d_queue + SPSTATE(0)->queue_eff_len ;
  /* 
   * copy inbuf data to first queue
   */
  cudaMemcpy(pos_inqueue, in_multidown, num_inchunk * sizeof(float), cudaMemcpyHostToDevice);
  float tmp_in[num_inchunk];
  int j;
  cudaMemcpy(tmp_in, pos_inqueue, num_inchunk * sizeof(float), cudaMemcpyDeviceToHost);
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

    
    threadsPerBlock = THREADSPERBLOCK; 
    numBlocks = out_processed % threadsPerBlock == 0 ? out_processed/threadsPerBlock : (int)out_processed/threadsPerBlock + 1;
    downsample2x <<<numBlocks, threadsPerBlock>>>(SPSTATEDOWN(i)->amplifier, 2, SPSTATEDOWN(i)->d_sinc_table, SPSTATEDOWN(i)->sinc_len, SPSTATEDOWN(i)->last_sample, SPSTATEDOWN(i)->d_mem, out_processed, pos_inqueue, pos_outqueue);

    /* The following code is used for comparison with gstreamer downsampler2x quality=6; */
    #if 0
    if (i == 0) {
	    float tmp[SPSTATEDOWN(i)->sinc_len + out_processed];
      	    cudaMemcpy(tmp, pos_outqueue, out_processed * sizeof(float), cudaMemcpyDeviceToHost);
	    for (j=0; j<out_processed; j++)
		    printf("out[%d] = %e\n", j, tmp[j]);
      	    cudaMemcpy(tmp, SPSTATEDOWN(i)->d_mem, (SPSTATEDOWN(i)->sinc_len + out_processed) * sizeof(float), cudaMemcpyDeviceToHost);
	    for (j=0; j<(SPSTATEDOWN(i)->sinc_len + out_processed); j++)
		    printf("mem[%d] = %e\n", j, tmp[j]);

    }
    #endif

    /* never discard any samples, we already prevent this situation from happening by providing a good size of num_cover_samples*/
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
#if 0
static void
upsample2x(float *in, const gint num_inchunk, float *out, gint *out_processed){
}

static void
iir_filter (float *in, const gint num_inchunk, float *out)
{
	gint i;
	for (i=0; i<num_inchunk; i++)
	{
		out[i] = in[i];
	}
}
#endif 
gint spiirup (SpiirState **spstate, gint num_in_multiup, gint num_depths, float *out)
{
  float *pos_out_spiir;
  gint num_inchunk = num_in_multiup, num_remains;

  gint i, up_spiir_processed, low_processed;

  /* 
   * SPIIR filter for the lowest depth 
   */

  GST_LOG ("spiirup %d samples", num_inchunk);
  int threadsPerBlock = num_inchunk;
  int numBlocks = 1;
  threadsPerBlock = THREADSPERBLOCK; 
  numBlocks = num_inchunk % threadsPerBlock == 0 ? num_inchunk / threadsPerBlock : (int)num_inchunk / threadsPerBlock + 1;

  pos_out_spiir = SPSTATEUP(num_depths-1)->d_mem + SPSTATEUP(num_depths-1)->filt_len - 1;
  iir_filter <<<numBlocks, threadsPerBlock>>>(SPSTATE(num_depths-1)->d_queue, pos_out_spiir);

//  spiir_state_flush_queue (spstate, num_depths-1, num_inchunk);

  for (i=num_depths-1; i>=1; i--) {
    //g_assert ((SPSTATE(i-1)->queue_eff_len - SPSTATE(i-1)->queue_down_start >= num_inchunk;
    /*
     * upsample 2x and add 
     */


    low_processed = num_inchunk - SPSTATEUP(i)->last_sample;
    up_spiir_processed = (num_inchunk - SPSTATEUP(i)->last_sample) * 2;

    threadsPerBlock = up_spiir_processed;

    pos_out_spiir = SPSTATEUP(i-1)->d_mem + SPSTATEUP(i-1)->filt_len - 1;
    iir_filter <<<numBlocks, threadsPerBlock>>>(SPSTATE(i-1)->d_queue, pos_out_spiir);

    threadsPerBlock = THREADSPERBLOCK; 
    numBlocks = low_processed % threadsPerBlock == 0 ? low_processed / threadsPerBlock : (int)low_processed/threadsPerBlock + 1;


    upsample2x_and_add <<<numBlocks, threadsPerBlock>>>(2, 2, SPSTATEUP(i)->d_sinc_table, SPSTATEUP(i)->filt_len, SPSTATEUP(i)->last_sample, low_processed, SPSTATEUP(i)->d_mem, pos_out_spiir);
    /*
     * filter finish, flush num_inchunk samples of queue;
     * update the effective length, down_start of this spstate;
     */
    num_remains = SPSTATE(i)->queue_eff_len - num_inchunk;
    cudaMemcpy(SPSTATE(i)->d_queue, SPSTATE(i)->d_queue + num_inchunk, num_remains * sizeof(float), cudaMemcpyDeviceToDevice);
    SPSTATE(i)->queue_eff_len -= num_inchunk;
    SPSTATE(i)->queue_down_start -= num_inchunk;
    SPSTATE(i)->queue_len -= num_inchunk;
    SPSTATEUP(i)->last_sample = 0;
    num_inchunk = up_spiir_processed; 
    GST_LOG ("%dth depth: queue eff len %d", i, SPSTATE(i)->queue_eff_len);
  }
  num_remains = SPSTATE(0)->queue_eff_len - num_inchunk;
  cudaMemcpy(SPSTATE(0)->d_queue, SPSTATE(0)->d_queue + num_inchunk, num_remains * sizeof(float), cudaMemcpyDeviceToDevice);
  SPSTATE(0)->queue_eff_len -= num_inchunk;
  SPSTATE(0)->queue_down_start -= num_inchunk;
  SPSTATE(0)->queue_len -= num_inchunk;
  GST_LOG ("%dth depth: queue eff len %d", i, SPSTATE(i)->queue_eff_len);
 
  GST_LOG ("spiirup out processed %d samples", num_inchunk);
  cudaMemcpy(out, pos_out_spiir, num_inchunk * sizeof(float), cudaMemcpyDeviceToHost);
  return up_spiir_processed;
}


#ifdef __cplusplus
extern "C" {
#endif

#include "multiratespiir.h"
#include "multiratespiir_utils.h"
#include "spiir_state_macro.h"
#include <glib.h>
#include <gst/gst.h>


#ifdef __cplusplus
}
#endif


__global__ void downsample2x (const int resolution,
			      const int times,
  			      float *sinc, 
			      const int sinc_len, 
			      int last_sample,
			      float *mem, 
			      const int len, 
			      float *queue_in, 
			      float *queue_out)
{


	const int gap			= resolution / times;

	double tmp = 0.0;
	int tx = threadIdx.x;

	// grab data from queue
	mem[sinc_len-1 + tx * times] = queue_in[tx * times];
	mem[sinc_len-1 + tx * times + 1] = queue_in[tx * times + 1 ];
	if (tx < last_sample)
		mem[sinc_len-1 + blockDim.x * times + tx] = 
			queue_in[blockDim.x * times + tx];

	for (int j = 0; j < sinc_len; ++j) {
	  tmp += mem[tx * times + j + last_sample] * sinc[j * gap];
	}

	queue_out[tx] = tmp / times;
	__syncthreads();

	// copy last to first (sinc_len-1) mem data
	if (tx < sinc_len -1)
		mem[tx] = mem[tx + len];
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
	int tx = threadIdx.x * 2;

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
	int tx = threadIdx.x;
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

  SPSTATE(0)->queue_eff_len += num_inchunk;

  for (i=1; i<num_depths; i++) {
    // predicted output length of downsample this round
    out_processed = (num_inchunk - SPSTATEDOWN(i-1)->last_sample)/2;
    //g_assert ((SPSTATE(i-1)->queue_eff_len - SPSTATE(i-1)->queue_down_start >= num_inchunk;
    /*
     * downsample 2x of number of filt samples
     */
    // make sure lower depth mem is large enough to store queue data.
    g_assert (num_inchunk <= SPSTATEDOWN(i-1)->mem_len - SPSTATEDOWN(i-1)->filt_len + 1 );
    // make sure current depth queue is large enough to store output data
    g_assert (out_processed <= SPSTATE(i)->queue_len - SPSTATE(i)->queue_eff_len);
    pos_inqueue = SPSTATE(i-1)->d_queue + SPSTATE(i-1)->queue_down_start;
    pos_outqueue = SPSTATE(i)->d_queue + SPSTATE(i)->queue_down_start;
    // CUDA downsample2x
    threadsPerBlock = out_processed;
    numBlocks = 1;
    downsample2x <<<numBlocks, threadsPerBlock>>>(2, 2, SPSTATEDOWN(i-1)->d_sinc_table, SPSTATEDOWN(i-1)->sinc_len, SPSTATEDOWN(i-1)->last_sample, SPSTATEDOWN(i-1)->d_mem, num_inchunk, pos_inqueue, pos_outqueue);
    /* 
     * if the number of input samples is odd, discard the last input 
     * sample. We do not expect this affect accuracy much.
     */
    if (num_inchunk % 2 == 1)
      SPSTATE(i-1)->queue_eff_len -= 1;
    /*
     * filter finish, update the next expected down start of upper 
     * spstate; update the effective length of this spstate;
     */
    SPSTATE(i-1)->queue_down_start = SPSTATE(i-1)->queue_eff_len;
    SPSTATEDOWN(i-1)->last_sample = 0 ;
    SPSTATE(i)->queue_eff_len += out_processed;
    num_inchunk = out_processed;
  }
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
  float *pos_inqueue, *pos_out_spiir , *pos_out_up;
  gint num_inchunk = num_in_multiup;

  gint i, spiir_processed, up_processed;

  int threadsPerBlock = num_inchunk;
  int numBlocks = 1;
  pos_out_spiir = SPSTATEUP(num_depths-1)->d_mem + SPSTATEUP(num_depths-1)->filt_len - 1;
  iir_filter <<<numBlocks, threadsPerBlock>>>(SPSTATE(num_depths-1)->d_queue, pos_out_spiir);

//  spiir_state_flush_queue (spstate, num_depths-1, num_inchunk);

  for (i=num_depths-2; i>=0; i--) {
    //g_assert ((SPSTATE(i-1)->queue_eff_len - SPSTATE(i-1)->queue_down_start >= num_inchunk;
    /*
     * upsample 2x and add 
     */


    up_processed = num_inchunk - SPSTATEUP(i+1)->last_sample;
    spiir_processed = (num_inchunk - SPSTATEUP(i+1)->last_sample) * 2;

    threadsPerBlock = spiir_processed;

    pos_out_spiir = SPSTATEUP(i)->d_mem + SPSTATEUP(i)->filt_len - 1;
    iir_filter <<<numBlocks, threadsPerBlock>>>(SPSTATE(i)->d_queue, pos_out_spiir);

    threadsPerBlock = up_processed;

    upsample2x_and_add <<<numBlocks, threadsPerBlock>>>(2, 2, SPSTATEUP(i+1)->d_sinc_table, SPSTATEUP(i+1)->filt_len, SPSTATEUP(i+1)->last_sample, up_processed, SPSTATEUP(i+1)->d_mem, pos_out_spiir);
    /*
     * filter finish, flush num_inchunk samples of queue;
     * update the effective length, down_start of this spstate;
     */
    threadsPerBlock = SPSTATE(i+1)->queue_eff_len - num_inchunk;
    flush_queue <<<numBlocks, threadsPerBlock>>>(SPSTATE(i+1)->d_queue, spiir_processed);
    SPSTATEUP(i+1)->last_sample = 0;
    num_inchunk = spiir_processed;
  }

  pos_out_spiir = SPSTATEUP(0)->d_mem + SPSTATEUP(i)->filt_len - 1;
  out = (float *)malloc(num_inchunk * sizeof(float));
  cudaMemcpy(out, pos_out_spiir, num_inchunk * sizeof(float), cudaMemcpyDeviceToHost);
  return spiir_processed;
}

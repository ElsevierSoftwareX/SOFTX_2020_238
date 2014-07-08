#include <string.h>
#include <math.h>

#include "resampler_state_macro.h"
#include "resampler_state_utils.h"
#include <cuda_runtime.h>

int sinc_function(float *sinc, gint sinc_len, gint times)
{
	if (sinc_len & 0x1)
		return -1;

	gint i;
	for (i = -(sinc_len/2); i < (sinc_len/2); ++i)
		sinc[sinc_len/2 + i] = sin(i * (1.0 / times) * M_PI) / (i * (1.0 / times) * M_PI);
	sinc[sinc_len / 2] = 1;

	return 0;
}

void psinc_function(float *sinc_table, int filtersize, int times)
{
	int c = 1 - times;
	int sincsize = filtersize * times;
	float *sinc = (float*)malloc(sizeof(float)*sincsize);
	sinc_function(sinc, sincsize, times);

	for (int i = 0; i < filtersize; ++i) {
		for (int j = times - 1; j >= 0; --j) {
			if (c < 0) {
				sinc_table[j * filtersize + i] = 0.0;
				++c;	
			} else {
				sinc_table[j * filtersize + i] = sinc[c++];
			}
		}	
	}
}


ResamplerState *
resampler_state_init (gint inrate, gint outrate, gint channels, gint num_exe_samples)
{
	gint mem_alloc_size, sinc_len;
	gint resolution = 2;
	ResamplerState *state = (ResamplerState *)malloc(sizeof(ResamplerState));
	state->inrate = inrate;
	state->outrate = outrate;
	if (inrate > outrate){
	  state->filt_len = DOWN_FILT_LEN * 2;
	  state->sinc_len = state->filt_len;
  	  sinc_len = state->filt_len * resolution;
	  cudaMalloc((void **) &(state->d_sinc_table), sizeof(float) * sinc_len);
	  float *sinc_table = (float *)malloc (sizeof(float) * sinc_len);
	  /* Sinc function Generator */
	  sinc_function(sinc_table, state->filt_len, resolution);
          cudaMemcpy(state->d_sinc_table, sinc_table, sinc_len * sizeof(float), cudaMemcpyHostToDevice);
  	  free(sinc_table);
	  sinc_table = NULL;

	} else {
	  state->filt_len = UP_FILT_LEN;
	  state->sinc_len = state->filt_len * 2;

	  sinc_len = state->filt_len * resolution;
	  cudaMalloc((void **) &(state->d_sinc_table), sizeof(float) * sinc_len);
	  float *sinc_table = (float *)malloc (sizeof(float) * sinc_len);
	  /* Sinc function Generator */
	  psinc_function(sinc_table, state->filt_len, resolution);
          cudaMemcpy(state->d_sinc_table, sinc_table, sinc_len * sizeof(float), cudaMemcpyHostToDevice);
 	  free(sinc_table);
	  sinc_table = NULL;


	}

	state->mem_len = state->filt_len - 1 + num_exe_samples;
	mem_alloc_size = state->mem_len * sizeof(float);
	cudaMalloc((void **) &(state->d_mem), mem_alloc_size);

//	state->mem = (float *)malloc(mem_alloc_size);
	cudaMemset(state->d_mem, 0, mem_alloc_size);
	state->last_sample = state->sinc_len/2;
	return state;
}

void 
resampler_state_reset (ResamplerState *state)
{
	gint mem_alloc_size = state->mem_len * sizeof(float);
	cudaMemset(state->d_mem, 0, mem_alloc_size);
	state->last_sample = state->filt_len/2;

}
void
resampler_state_destroy (ResamplerState *state)
{
  if (state->d_sinc_table)
    cudaFree(state->d_sinc_table) ;
  cudaFree(state->d_mem) ;
}



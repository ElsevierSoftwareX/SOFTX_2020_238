
#include <math.h>
#include "resampler_state_utils.h"
#include "spiir_state_utils.h"
#include "spiir_state_macro.h"
#include <cuda_runtime.h>

SpiirState ** 
spiir_state_init (gint num_depths, gint num_cover_samples,
		gint num_exe_samples, gint width, gint rate, gint outchannels)
{
	gint i, tmp_len, inrate, outrate, queue_alloc_size;
	SpiirState ** spstate = (SpiirState **)malloc(num_depths * sizeof(SpiirState*));

	for(i=0; i<num_depths; i++)
	{
		SPSTATE(i) = (SpiirState *)malloc(sizeof(SpiirState));
		SPSTATE(i)->depth = i;
		SPSTATE(i)->queue_len = (num_cover_samples + num_exe_samples) / pow (2, i) + 1; 
		SPSTATE(i)->queue_eff_len = 0;
		SPSTATE(i)->queue_down_start = 0;
		SPSTATE(i)->queue_up_start = 0;
		queue_alloc_size = SPSTATE(i)->queue_len* sizeof(float);
		cudaMalloc((void **) &(SPSTATE(i)->d_queue), queue_alloc_size);
		cudaMemset(SPSTATE(i)->d_queue, 0, queue_alloc_size);
		inrate = rate/pow(2, i);
		outrate = inrate / 2;
  		tmp_len = num_exe_samples / pow(2, i);
//		SPSTATE(i)->out_spiir = (float*)malloc(tmp_len * sizeof(float));
//		tmp_len *=2;
//		SPSTATE(i)->out_up = (float*)malloc(tmp_len * sizeof(float));

		SPSTATEDOWN(i) = resampler_state_init (inrate, outrate, outchannels, num_exe_samples);
		SPSTATEUP(i) = resampler_state_init (outrate, inrate, outchannels, num_exe_samples);
	}
	return spstate;
}

void 
spiir_state_destroy (SpiirState ** spstate, gint num_depths)
{
	gint i; for(i=0; i<num_depths; i++)
	{
		resampler_state_destroy (SPSTATEDOWN(i));
		resampler_state_destroy (SPSTATEUP(i));
		free(SPSTATEDOWN(i));
		free(SPSTATEUP(i));
		cudaFree(SPSTATE(i)->d_queue);
		free(SPSTATE(i));

	}
}

void
spiir_state_reset (SpiirState **spstate, gint num_depths)
{
  int i;
  for(i=0; i<num_depths; i++)
  {
    SPSTATE(i)->queue_eff_len = 0;
    SPSTATE(i)->queue_down_start = 0;
    resampler_state_reset(SPSTATEDOWN(i));
    resampler_state_reset(SPSTATEUP(i));
  }
}

#if 0
/* DEPRECATED */
void
spiir_state_flush_queue (SpiirState **spstate, gint depth, gint
		num_flush_samples)
{
  int i;
  gint queue_len = SPSTATE(depth)->queue_len;
  float *pos_queue = SPSTATE(depth)->queue;

  for (i=0; i<queue_len - num_flush_samples; i++) 
	  pos_queue[i] = pos_queue[i + num_flush_samples];

  SPSTATE(depth)->queue_len = SPSTATE(depth)->queue_len - num_flush_samples;
  SPSTATE(depth)->queue_eff_len = SPSTATE(depth)->queue_eff_len - num_flush_samples;
  SPSTATE(depth)->queue_down_start = SPSTATE(depth)->queue_down_start - num_flush_samples;
}
#endif

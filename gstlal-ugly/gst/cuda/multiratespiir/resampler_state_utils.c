#include "resampler_state_utils.h"

ResamplerState *
resampler_state_init (gint inrate, gint outrate, gint channels)
{
	gint mem_alloc_size;
	ResamplerState *state = (ResamplerState *)malloc(sizeof(ResamplerState));
	state->inrate = inrate;
	state->outrate = outrate;
	if (inrate > outrate){
	  state->filt_len = DOWN_FILT_LEN;
	  state->sinc_table = NULL; // FIXME
	} else {
	  state->filt_len = UP_FILT_LEN;
	  state->sinc_table = NULL;
	}

	state->mem_len = state->filt_len - 1 + inrate;
	mem_alloc_size = state->mem_len * sizeof(float);
	state->mem = (float *)malloc(mem_alloc_size);
	memset(state->mem, 0, mem_alloc_size);
	state->last_sample = state->filt_len/2;
	return state;
}

void 
resampler_state_reset (ResamplerState *state)
{
	gint mem_alloc_size = state->mem_len * sizeof(float);
	memset(state->mem, 0, mem_alloc_size);
	state->last_sample = state->filt_len/2;

}

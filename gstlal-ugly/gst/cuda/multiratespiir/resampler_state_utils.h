#include "multiratespiir.h"


ResamplerState *
resampler_state_create (gint inrate, gint outrate, gint channels, gint num_exe_samples, gint num_cover_samples, gint depth, cudaStream_t stream);

void 
resampler_state_reset (ResamplerState *state, cudaStream_t stream);

void 
resampler_state_destroy (ResamplerState *state);


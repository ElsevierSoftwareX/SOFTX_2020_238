#include "multiratespiir.h"


ResamplerState *
resampler_state_init (gint inrate, gint outrate, gint channels, gint num_exe_samples, gint num_cover_samples, gint depth);

void 
resampler_state_reset (ResamplerState *state);

void 
resampler_state_destroy (ResamplerState *state);


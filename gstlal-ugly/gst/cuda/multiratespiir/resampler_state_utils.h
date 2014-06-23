#include "multiratespiir.h"

#define DOWN_FILT_LEN 64
#define UP_FILT_LEN 16


ResamplerState *
resampler_state_init (gint inrate, gint outrate, gint channels);


void 
resampler_state_reset (ResamplerState *state);


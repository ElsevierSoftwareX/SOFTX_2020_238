
#include "multiratespiir.h"

ResamplerState *
resampler_state_create (gint inrate, gint outrate, gint channels, gint num_exe_samples, gint num_cover_samples, gint depth, cudaStream_t stream);

void 
resampler_state_reset (ResamplerState *state, cudaStream_t stream);

void 
resampler_state_destroy (ResamplerState *state);

SpiirState **
spiir_state_create (const gchar *bank_fname, guint ndepth, guint rate, guint num_head_cover_samples,
		gint num_exe_samples, cudaStream_t stream);

void 
spiir_state_destroy (SpiirState ** spstate, guint num_depths);

void
spiir_state_reset (SpiirState **spstate, guint num_depths, cudaStream_t stream);

gint
spiir_state_get_outlen (SpiirState **spstate, gint in_len, guint num_depths);




#include "multiratespiir.h"


SpiirState ** 
spiir_state_init (gdouble *bank, gint bank_len, gint num_cover_samples,
		gint num_exe_samples, gint width, gint rate, cudaStream_t stream);

void 
spiir_state_destroy (SpiirState ** spstate, gint num_depths);

void
spiir_state_reset (SpiirState **spstate, gint num_depths, cudaStream_t stream);

void
spiir_state_flush_queue (SpiirState **spstate, gint depth, 
		gint num_flush_samples);

gint
spiir_state_get_outlen (SpiirState **spstate, gint in_len, gint num_depths);


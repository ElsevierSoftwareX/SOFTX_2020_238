
#include "multiratespiir.h"


SpiirState ** 
spiir_state_init (gint num_depths, gint num_cover_samples,
		gint num_exe_samples, gint width, gint rate, gint outchannels);

void 
spiir_state_destroy (SpiirState ** spstate, gint num_depths);

void
spiir_state_reset (SpiirState **spstate, gint num_depths);

void
spiir_state_flush_queue (SpiirState **spstate, gint depth, 
		gint num_flush_samples);

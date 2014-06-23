
#include <math.h>
#include "multiratespiir.h"
#include "resampler_state_utils.h"


#define SPSTATE(i) (*(spstate+i)) 
#define SPSTATEDOWN(i) (SPSTATE(i)->downstate)
#define SPSTATEUP(i) (SPSTATE(i)->upstate)

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


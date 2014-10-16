
#include "multiratespiir.h"

SpiirState ** 
spiir_state_create (gdouble *bank, gint bank_len, guint num_cover_samples,
		guint num_exe_samples, gint width, guint rate, cudaStream_t stream);

void 
spiir_state_destroy (SpiirState ** spstate, guint num_depths);

void
spiir_state_reset (SpiirState **spstate, guint num_depths, cudaStream_t stream);

gint
spiir_state_get_outlen (SpiirState **spstate, gint in_len, guint num_depths);

/* DEPRECATED
void
spiir_state_flush_queue (SpiirState **spstate, gint depth, 
		gint num_flush_samples);
*/


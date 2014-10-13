/* 
 * utils for multi rate spiir
 */
#include "multiratespiir.h"

#define RESAMPLER_NUM_DEPTHS_MIN 0 
#define RESAMPLER_NUM_DEPTHS_MAX 7 
#define RESAMPLER_NUM_DEPTHS_DEFAULT 7 
#define MATRIX_DEFAULT 1

//#define MIN(a, b) if a > b ? b : a

//#define ELESPSTATE(ele, i) (*(ele->spstate+i))
// guint get_num_outsamples(SpiirState **pspstate, guint insamples
//

void 
cuda_multirate_spiir_init_cover_samples (gint *num_head_cover_samples, 
		gint *num_tail_cover_samples, gint rate, gint num_depths, 
		gint down_filtlen, gint up_filtlen);

void 
cuda_multirate_spiir_update_exe_samples (gint *num_exe_samples, gint new_value);

gboolean 
cuda_multirate_spiir_parse_bank (gdouble *bank, gint *num_depths, gint *
		outchannels);

gint 
cuda_multirate_spiir_get_outchannels (CudaMultirateSPIIR *element);

gint 
cuda_multirate_spiir_get_num_head_cover_samples (CudaMultirateSPIIR *element);

guint64 
cuda_multirate_spiir_get_available_samples (CudaMultirateSPIIR *element);

gint
multi_downsample (SpiirState **spstate, float *in_multidown, 
		gint num_in_multidown, gint num_depths, cudaStream_t stream);

gint
spiirup (SpiirState **spstate, gint num_in_multiup, gint num_depths, float *out, cudaStream_t stream);




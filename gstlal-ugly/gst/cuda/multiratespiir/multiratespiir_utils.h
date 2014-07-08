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

gint cuda_multirate_spiir_init_cover_samples (gint rate, gint num_depths, gint down_filtlen, gint up_filtlen);

gint cuda_multirate_spiir_get_num_templates(CudaMultirateSPIIR *element);

gint cuda_multirate_spiir_get_num_cover_samples(CudaMultirateSPIIR *element);
guint64 cuda_multirate_spiir_get_available_samples(CudaMultirateSPIIR *element);

void cuda_multirate_spiir_add_two_data(float *data1, float *data2, gint len);


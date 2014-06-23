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

gint init_cover_samples (gint rate, gint num_depths, gint down_filtlen, gint up_filtlen);

gint get_num_templates(CudaMultirateSPIIR *element);

gint get_num_cover_samples(CudaMultirateSPIIR *element);
guint64 get_available_samples(CudaMultirateSPIIR *element);

void add_two_data(float *data1, float *data2, gint len);





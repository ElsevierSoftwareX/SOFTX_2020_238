#ifdef __cplusplus
extern "C" {
#endif

#include <postcoh/postcoh.h>

#ifdef __cplusplus
}
#endif

#define POSTCOH_PARAMS_NOT_INIT -1
#define POSTCOH_PARAMS_INIT 1

void cuda_device_print(int deviceCount);

PeakList *create_peak_list(PostcohState *state, cudaStream_t stream);

void
cuda_postcoh_map_from_xml(char *fname, PostcohState *state, cudaStream_t stream);

void
cuda_postcoh_autocorr_from_xml(char *fname, PostcohState *state, cudaStream_t stream);

void
cuda_postcoh_sngl_tmplt_from_xml(char *fname, SnglInspiralTable **psngl_table);

void
peakfinder(PostcohState *state, int iifo, cudaStream_t stream);
void
state_destroy(PostcohState *state);

void
peak_list_destroy(PeakList *pklist);

void
state_reset_npeak(PeakList *pklist);

void cohsnr_and_chisq(PostcohState *state, int iifo, int gps_idx, int output_skymap, cudaStream_t stream);

void cohsnr_and_chisq_background(PostcohState *state, int iifo, int hist_trials , int gps_idx);

void transpose_snglsnr(COMPLEX_F* idata, COMPLEX_F* odata, int offset, int copy_snglsnr_len, int snglsnr_len, int tmplt_len, cudaStream_t stream);

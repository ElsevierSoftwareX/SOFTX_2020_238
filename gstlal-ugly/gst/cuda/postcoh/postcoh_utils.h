#include "postcoh.h"

PeakList *create_peak_list(PostcohState *state);

void
cuda_postcoh_map_from_xml(char *fname, PostcohState *state);

void
cuda_postcoh_autocorr_from_xml(char *fname, PostcohState *state);

void
peakfinder(PostcohState *state, int iifo);
void
state_destroy(PostcohState *state);

void
peak_list_destroy(PeakList *pklist);

void
state_reset_npeak(PeakList *pklist);

void cohsnr_and_chisq(PostcohState *state, int iifo, int gps_idx);

void cohsnr_and_chisq_background(PostcohState *state, int iifo, int hist_trials , int gps_idx);

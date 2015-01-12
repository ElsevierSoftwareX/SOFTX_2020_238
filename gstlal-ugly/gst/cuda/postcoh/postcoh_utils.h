#include "postcoh.h"

PeakList *create_peak_list(PostcohState *state, int iifo);

void
cuda_postcoh_map_from_xml(char *fname, PostcohState *state);
void
peakfinder(PostcohState *state, int iifo);
void
state_destroy(PostcohState *state);

void
state_reset_npeak(PeakList *pklist);

void cohsnr_and_chi2(PostcohState *state, int iifo, int gps_idx);

void cohsnr_and_chi2_background(PostcohState *state, int iifo, int hist_trials , int gps_idx);

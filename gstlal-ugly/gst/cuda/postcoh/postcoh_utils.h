#include "postcoh.h"

PeakList *create_peak_list(int exe_len);

void
cuda_postcoh_map_from_xml(char *fname, PostcohState *state);
void
peakfinder(PostcohState *state, int iifo);
void
state_destroy(PostcohState *state);

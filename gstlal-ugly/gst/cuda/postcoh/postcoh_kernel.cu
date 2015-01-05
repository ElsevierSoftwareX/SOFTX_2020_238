#include "postcoh.h"

void peakfinder(COMPLEX_F *one_d_snglsnr, int iifo, PostcohState *state)
{

	COMPLEX_F *d_snglsnr = one_d_snglsnr + state->head_len * state->ntmplt;
//	kel_step1(d_snglsnr, state->ntmplt, state->exe_len, state->peak_list[iifo]->maxsnr, state->peak_list[iifo]->tmplt_index)
}



#ifdef __cplusplus
extern "C" {
#endif
#include <stdio.h>

#include "postcoh.h"
#include "postcoh_utils.h"

#ifdef __cplusplus
}
#endif

#define WARP_SIZE 		32
#define LOG_WARP_SIZE	5

__device__ static inline float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void kel_max_snglsnr
(
    COMPLEX_F*  snr,        // INPUT: snr
    int     templateN,  // INPUT: template number
    int     timeN,      // INPUT: time sample number
    float*  ressnr,     // OUTPUT: maximum snr
    int*    restemplate,     // OUTPUT: max snr time
    int*	ressample
)
{
    int wIDl    = threadIdx.x & (WARP_SIZE - 1); 
    int wIDg    = (threadIdx.x + blockIdx.x * blockDim.x) >> LOG_WARP_SIZE;
    int wNg     = (gridDim.x * blockDim.x) >> LOG_WARP_SIZE;

    float   max_snr_sq         = - 1;
    int     max_template    = 0; 
    float   temp_snr_sq;
    int     temp_template;
    int	index = 0;

    for (int i = wIDg; i < timeN; i += wNg)
    {
        // one warp is used to find the max snr for one time
        for (int j = wIDl; j < templateN; j += WARP_SIZE)
        {
	    index = i * templateN + j;
            temp_snr_sq        = snr[index].re * snr[index].re + snr[index].im * snr[index].im;

            max_template    = (j + max_template) + (j - max_template) * (2 * (temp_snr_sq > max_snr_sq) - 1);
            max_template    = max_template >> 1;

            max_snr_sq     = (max_snr_sq + temp_snr_sq) * 0.5f  + (max_snr_sq - temp_snr_sq) * ((max_snr_sq > temp_snr_sq) - 0.5f);
        }

        // inter-warp reduction to find the max snr among threads in the warp
        for (int j = WARP_SIZE / 2; j > 0; j = j >> 1)
        {

            temp_snr_sq    = __shfl(max_snr_sq , wIDl + j, 2 * j);

            temp_template = __shfl(max_template, wIDl + j, 2 * j); 
            max_template = (max_template + temp_template) + (max_template - temp_template) * (2 * (max_snr_sq > temp_snr_sq) - 1);
            max_template = max_template >> 1;

            max_snr_sq = (max_snr_sq + temp_snr_sq) * 0.5f + (max_snr_sq - temp_snr_sq) * ((max_snr_sq > temp_snr_sq) - 0.5f);
        }

        if (wIDl == 0)
        {
            ressnr[i]   = sqrt(max_snr_sq); 
            restemplate[i]  = max_template;
            ressample[i]  = i;
        }
    }
}

__global__ void
kel_remove_duplicate_mix
(
    float*  ressnr,
    int*    restemplate,
    int     timeN,
    int     templateN,
    float*  peak
)
{
    // there is only 1 thread block 

    float   snr;
    int     templ;

    for (int i = threadIdx.x; i < timeN; i += blockDim.x)
    {
        snr     = ressnr[i];
        templ   = restemplate[i];
        atomicMax(peak + templ, snr);
    }
    __syncthreads();
     
    float   max_snr;
    for (int i = threadIdx.x; i < timeN; i += blockDim.x)
    {
        snr     = ressnr[i];
        templ   = restemplate[i];
        max_snr = peak[templ];

        restemplate[i]  = ((-1 + templ) + (-1 - templ) * ((max_snr > snr) * 2 - 1)) >> 1;
        ressnr[i]       = (-1.0f + snr) * 0.5 + (-1.0f - snr) * ((max_snr > snr) - 0.5);
    }
}

__global__ void 
kel_remove_duplicate_find_peak
(
    float*  ressnr,
    int*    restemplate,
    int     timeN,
    int     templateN,
    float*  peak            // peak is used for storing intermediate result, it is of size templateN
)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int tn  = blockDim.x * gridDim.x; 

    float   snr;
    int     templ;
    for (int i = tid; i < timeN; i += tn)
    {
        snr     = ressnr[i]; 
        templ   = restemplate[i];
        atomicMax(peak + templ, snr);        
    }
}

__global__ void
kel_remove_duplicate_scan
(
    float*  ressnr,
    int*    restemplate,
    int     timeN,
    int     templateN,
    float*  peak
)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int tn  = blockDim.x * gridDim.x;

    float   max_snr;
    float   snr;
    int     templ;
    for (int i = tid; i < timeN; i += tn)
    {
        snr     = ressnr[i];
        templ   = restemplate[i];
        max_snr = peak[templ];

        restemplate[i]  = ((-1 + templ) + (-1 - templ) * ((max_snr > snr) * 2 - 1)) >> 1;
        ressnr[i]       = (-1.0f + snr) * 0.5 + (-1.0f - snr) * ((max_snr > snr) - 0.5);
	printf("tmplt %d, snr %f\n", restemplate[i], ressnr[i]);
    }
}

void peakfinder(COMPLEX_F *one_d_snglsnr, int iifo, PostcohState *state)
{

	printf("start peakfinder\n");
	COMPLEX_F *d_snglsnr = one_d_snglsnr + state->head_len * state->ntmplt;
    int THREAD_BLOCK    = 256;
    int GRID            = (state->exe_len * 32 + THREAD_BLOCK - 1) / THREAD_BLOCK;
	kel_max_snglsnr<<<GRID, THREAD_BLOCK>>>(d_snglsnr, state->ntmplt, state->exe_len, state->peak_list[iifo]->d_maxsnglsnr, state->peak_list[iifo]->d_tmplt_index, state->peak_list[iifo]->d_sample_index);
	
    GRID = (state->exe_len + THREAD_BLOCK - 1) / THREAD_BLOCK;
    float *peak_tmplt;
    cudaMalloc((void **)&peak_tmplt, sizeof(float) * state->ntmplt);
    kel_remove_duplicate_find_peak<<<GRID, THREAD_BLOCK>>>(state->peak_list[iifo]->d_maxsnglsnr, state->peak_list[iifo]->tmplt_index, state->exe_len, state->ntmplt, peak_tmplt);
    kel_remove_duplicate_scan<<<GRID, THREAD_BLOCK>>>(state->peak_list[iifo]->d_maxsnglsnr, state->peak_list[iifo]->tmplt_index, state->exe_len, state->ntmplt, peak_tmplt);
    cudaFree(peak_tmplt);
}

/* calculate cohsnr, null stream, chi2 of a peak list and copy it back */
void cohsnr_and_chi2(int iifo, PostcohState *state)
{
}

/* calculate cohsnr, null stream, chi2 of a peak list and copy it back */
void cohsnr_and_chi2_background(int iifo, PostcohState *state)
{
}


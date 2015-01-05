#include "postcoh.h"

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

__global__ void kel_maxsnr
(
    float*  snr,        // INPUT: snr
    int     templateN,  // INPUT: template number
    int     timeN,      // INPUT: time sample number
    float*  ressnr,     // OUTPUT: maximum snr
    int*    restemplate     // OUTPUT: max snr time
)
{
    int wIDl    = threadIdx.x & (WARP_SIZE - 1); 
    int wIDg    = (threadIdx.x + blockIdx.x * blockDim.x) >> LOG_WARP_SIZE;
    int wNg     = (gridDim.x * blockDim.x) >> LOG_WARP_SIZE;

    float   max_snr         = - 1;
    int     max_template    = 0; 
    float   temp_snr;
    int     temp_template;

    for (int i = wIDg; i < timeN; i += wNg)
    {
        // one warp is used to find the max snr for one time
        for (int j = wIDl; j < templateN; j += WARP_SIZE)
        {
            temp_snr        = snr[i * templateN + j];

            max_template    = (j + max_template) + (j - max_template) * (2 * (temp_snr > max_snr) - 1);
            max_template    = max_template >> 1;

            max_snr     = (max_snr + temp_snr) * 0.5f  + (max_snr - temp_snr) * ((max_snr > temp_snr) - 0.5f);
        }

        // inter-warp reduction to find the max snr among threads in the warp
        for (int j = WARP_SIZE / 2; j > 0; j = j >> 1)
        {

            temp_snr    = __shfl(max_snr , wIDl + j, 2 * j);

            temp_template = __shfl(max_template, wIDl + j, 2 * j); 
            max_template = (max_template + temp_template) + (max_template - temp_template) * (2 * (max_snr > temp_snr) - 1);
            max_template = max_template >> 1;

            max_snr = (max_snr + temp_snr) * 0.5f + (max_snr - temp_snr) * ((max_snr > temp_snr) - 0.5f);
        }

        if (wIDl == 0)
        {
            ressnr[i]   = max_snr; 
            restemplate[i]  = max_template;
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
    }
}

void peakfinder(COMPLEX_F *one_d_snglsnr, int iifo, PostcohState *state)
{

	COMPLEX_F *d_snglsnr = one_d_snglsnr + state->head_len * state->ntmplt;
    int THREAD_BLOCK    = 256;
    int GRID            = (timeN * 32 + THREAD_BLOCK - 1) / THREAD_BLOCK;
	kel_maxsnr(d_snglsnr, state->ntmplt, state->exe_len, state->peak_list[iifo]->maxsnr, state->peak_list[iifo]->tmplt_index)
	
    GRID = (timeN + THREAD_BLOCK - 1) / THREAD_BLOCK;
    kel_remove_duplicate_find_peak<<<GRID, THREAD_BLOCK>>>(ressnr, restemplate, timeN, templateN, peak);
    kel_remove_duplicate_scan<<<GRID, THREAD_BLOCK>>>(ressnr, restemplate, timeN, templateN, peak);
}


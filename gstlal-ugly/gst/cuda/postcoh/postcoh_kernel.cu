
#ifdef __cplusplus
extern "C" {
#endif
#include <stdio.h>

#include "postcoh.h"
#include "postcoh_utils.h"

#ifdef __cplusplus
}
#endif

const int GAMMA_ITMAX = 50;
// const float GAMMA_EPS = 2.22e-16;


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

__global__ void ker_max_snglsnr
(
    COMPLEX_F**  snr,        // INPUT: snr
    int iifo,
    int start_len,
    int len,
    int     templateN,  // INPUT: template number
    int     timeN,      // INPUT: time sample number
    float*  ressnr,     // OUTPUT: maximum snr
    int*    restemplate     // OUTPUT: max snr time
)
{
    int wIDl    = threadIdx.x & (WARP_SIZE - 1); 
    int wIDg    = (threadIdx.x + blockIdx.x * blockDim.x) >> LOG_WARP_SIZE;
    int wNg     = (gridDim.x * blockDim.x) >> LOG_WARP_SIZE;

    float   max_snr_sq         = - 1;
    int     max_template    = 0; 
    float   temp_snr_sq;
    int     temp_template;
    int	index = 0, start_idx = 0;

    for (int i = wIDg; i < timeN; i += wNg)
    {
	    start_idx = (i + start_len) % len;
        // one warp is used to find the max snr for one time
        for (int j = wIDl; j < templateN; j += WARP_SIZE)
        {
	    index = start_idx * templateN + j;
	
            temp_snr_sq        = snr[iifo][index].re * snr[iifo][index].re + snr[iifo][index].im * snr[iifo][index].im;

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
        }
    }
}

__global__ void
ker_remove_duplicate_mix
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
ker_remove_duplicate_find_peak
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
ker_remove_duplicate_scan
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

__device__ float gser (float x, float a)
{
    float ap, del, sum;
    int i;

    ap = a;
    del = 1.0 / a;
    sum = del;

    for (i = 1; i <= GAMMA_ITMAX; ++i)
    {   
        ap += 1.0;
        del *= x / ap; 
        sum += del;
        // if (fabs(del) < fabs(sum) * GAMMA_EPS)
        // {   
        //     return sum * exp(-x + a * log(x) - lgamma(a));
        // }   
    }   
    return sum * exp(-x + a * log(x) - lgamma(a));
}

__global__ void ker_coh_sky_map
(
	float	*coh_snr,			/* OUTPUT, of size (num_triggers * num_sky_directions) */
	float	*coh_nullstream,	/* OUTPUT, of size (num_triggers * num_sky_directions) */
	COMPLEX_F	*snr_all,			/* INPUT, (2, 3) * data_points */	
	int		iifo,			/* INPUT, detector we are considering */
	int		nifo,			/* INPUT, all detectors that are in this coherent analysis */
	int		*triggers_offset0,	/* INPUT, place the location of the trigger */
	int		num_triggers,		/* INPUT, number of triggers */
	float	*u_map,				/* INPUT, u matrix map */
	int		*toa_diff_map,		/* INPUT, time of arrival difference map */
	int		num_sky_directions,	/* INPUT, # of sky directions */
	int		exe_len				/* INPUT, execution time length */
)
{
	int bid	= blockIdx.x;
	int bn	= gridDim.x;

	// float	*mu;	// matrix u for certain sky direction
	int		peak_cur;
	COMPLEX_F	dk[6];
	int		NtOff;
	int		map_idx;
	float	real, imag;
	float	utdka[6];
	float	al_all = 0.0f;

	for (int l = bid; l < num_triggers; l += bn)
	{
		peak_cur = triggers_offset0[l] - 1;

		for (int i = threadIdx.x; i < num_sky_directions; i += blockDim.x)
		{
			for (int j = 0; j < nifo; ++j)
			{
				if (iifo !=  j) 
				{
					if (iifo < j) 
					{
						map_idx = nifo * iifo + j - ((iifo + 3) * iifo >> 1) - 1;		
						NtOff = toa_diff_map[map_idx * num_sky_directions + i];
					} 
					else 
					{
						map_idx = nifo * j + iifo - ((j + 3) * j >> 1) - 1;	
						NtOff = -toa_diff_map[map_idx * num_sky_directions + i];
					}
					dk[j] = snr_all[j * exe_len + peak_cur + NtOff]; 	
				} 
				else 
				{
					dk[j] = snr_all[j * exe_len + peak_cur];
				}
			}

			// do the matrix vector multiplication
			for (int j = 0; j < nifo; ++j)
			{
				real = 0.0f;
				imag = 0.0f;
				for (int k = 0; k < nifo; ++k)
				{
					/*
					real += mu[j * nifo + k] * dk[k].x;
					imag += mu[j * nifo + k] * dk[k].y;
					*/
					real += u_map[(j * nifo + k) * num_sky_directions + i] * dk[k].re;
					imag += u_map[(j * nifo + k) * num_sky_directions + i] * dk[k].im;
				}
				utdka[j] = real * real + imag * imag;	
			}		

			coh_snr[l * num_sky_directions + i] = (2 * (utdka[0] + utdka[1]) - 4) / sqrt(8.0f);

			al_all = 0.0f;
			for (int j = 2; j < nifo; ++j)
				al_all += utdka[j];	
			coh_nullstream[l * num_sky_directions + i] = 1 - gser(al_all, 1.0f);
		}
	}
}

__global__ void ker_coh_sky_map_max
(
	float		*coh_snr,		/* OUTPUT, only save the max one, of size (num_triggers) */
	float		*coh_nullstream,	/* OUTPUT, only save the max one, of size (num_triggers) */
	int		*pix_idx,		/* OUTPUT, store the index of the sky_direction, of size (num_triggers) */
	float		*chi2,			/* OUTPUT, chisq value */
	COMPLEX_F	**snr,			/* INPUT, (2, 3) * data_points */	
	int		iifo,			/* INPUT, detector we are considering */
	int		nifo,			/* INPUT, all detectors that are in this coherent analysis */
	int		*tmplt_idx,		/* INPUT, the tmplt index of triggers	*/
	int		*peak_pos,	/* INPUT, place the location of the trigger */
	int		npeak,		/* INPUT, number of triggers */
	float		*u_map,				/* INPUT, u matrix map */
	float		*toa_diff_map,		/* INPUT, time of arrival difference map */
	int		num_sky_directions,	/* INPUT, # of sky directions */
	int		len,				/* INPUT, snglsnr length */
	int		start_exe,				/* INPUT, snglsnr start exe position */
	float		dt,			/* INPUT, 1/ sampling rate */
	int		some_trial		/* INPUT, trial number */
)
{
	int bid	= blockIdx.x;
	int bn	= gridDim.x;

	int wn	= blockDim.x >> LOG_WARP_SIZE;

	// store snr_max, nullstream_max and sky_idx, each has (blockDim.x / WARP_SIZE) elements
	extern __shared__ float smem[];
	volatile float *snr_shared = &smem[0];
	volatile float *nullstream_shared = &snr_shared[wn];
	volatile int *sky_idx_shared = (int*)&nullstream_shared[wn];

	// float	*mu;	// matrix u for certain sky direction
	int		peak_cur, tmplt_cur;
	COMPLEX_F	dk[6];
	int		NtOff;
	int		map_idx;
	float	real, imag;
	float	utdka[6];
	float	al_all = 0.0f;

	float	snr_max			= 0.0f;
	float	nullstream_max;
	int		sky_idx;	
	float	snr_tmp;

	for (int l = bid; l < npeak; l += bn)
	{
		peak_cur = peak_pos[l];
		tmplt_cur = tmplt_idx[peak_cur];

		for (int ipix = threadIdx.x; ipix < num_sky_directions; ipix += blockDim.x)
		{
			// matrix u is stored in column order
			// mu = u_map + nifo * nifo * i;			

			for (int j = 0; j < nifo; ++j)
			{
				/* this is a simplified algorithm to get map_idx */
				map_idx = iifo * nifo + j;
				NtOff = round (toa_diff_map[map_idx * num_sky_directions + ipix] / dt);
				dk[j] = snr[j][tmplt_cur * len + (start_exe + peak_cur + NtOff) % len]; 	
			}

			for (int j = 0; j < nifo; ++j)
			{
				real = 0.0f;
				imag = 0.0f;
				for (int k = 0; k < nifo; ++k)
				{
					// real += mu[j * nifo + k] * dk[k].x;
					// imag += mu[j * nifo + k] * dk[k].y;
					real += u_map[(j * nifo + k) * num_sky_directions + ipix] * dk[k].re;
					imag += u_map[(j * nifo + k) * num_sky_directions + ipix] * dk[k].im;
				}
				utdka[j] = real * real + imag * imag;	
			}		

			// coh_snr[l * num_sky_directions + i] = (2 * (utdka[0] + utdka[1]) - 4)	/ sqrt(8.0f);
			snr_tmp = utdka[0] + utdka[1];

			if (snr_tmp > snr_max)
			{
				snr_max = snr_tmp;
				al_all = 0.0f;
				for (int j = 2; j < nifo; ++j)
					al_all += utdka[j];	
				nullstream_max = 1 - gser(al_all, 1.0f);;
				sky_idx = ipix;
			}
		}

		int srcLane = threadIdx.x & 0x1f;

		for (int i = WARP_SIZE / 2; i > 0; i = i >> 1)
		{
			snr_tmp = __shfl(snr_max, srcLane + i, i * 2);
			if (snr_tmp > snr_max)
			{
				snr_max = snr_tmp;
				nullstream_max = __shfl(nullstream_max, srcLane + i, i * 2);
				sky_idx = __shfl(sky_idx, srcLane + i, i * 2);
			}
		}

		if (srcLane == 0)
		{
			snr_shared[threadIdx.x >> LOG_WARP_SIZE]		= snr_max;
			nullstream_shared[threadIdx.x >> LOG_WARP_SIZE]	= nullstream_max;
			sky_idx_shared[threadIdx.x >> LOG_WARP_SIZE]	= sky_idx;
		}
		__syncthreads();

		for (int i = wn >> 1; i > 0; i = i >> 1)
		{
			if (threadIdx.x < i)
			{
				snr_tmp = snr_shared[threadIdx.x + i];
				snr_max = snr_shared[threadIdx.x];

				if (snr_tmp > snr_max)
				{
					snr_shared[threadIdx.x] = snr_tmp;
					nullstream_shared[threadIdx.x] = nullstream_shared[threadIdx.x + i];
					sky_idx_shared[threadIdx.x] = sky_idx_shared[threadIdx.x + i];
				}
			}	
			__syncthreads();
		}

		if (threadIdx.x == 0)
		{
			coh_snr[l]			= snr_shared[0];
			coh_nullstream[l]	= nullstream_shared[0];
			pix_idx[l]		= sky_idx_shared[0]; 			
		}
	}
}


void peakfinder(PostcohState *state, int iifo)
{

	printf("start peakfinder\n");
    int THREAD_BLOCK    = 256;
    int GRID            = (state->exe_len * 32 + THREAD_BLOCK - 1) / THREAD_BLOCK;
    PeakList *pklist = state->peak_list[iifo];
	ker_max_snglsnr<<<GRID, THREAD_BLOCK>>>(state->d_snglsnr, 
						iifo,
						state->snglsnr_start_exe,
						state->snglsnr_len,
						state->ntmplt, 
						state->exe_len, 
						pklist->d_maxsnglsnr, 
						pklist->d_tmplt_idx);
	
    GRID = (state->exe_len + THREAD_BLOCK - 1) / THREAD_BLOCK;
    ker_remove_duplicate_find_peak<<<GRID, THREAD_BLOCK>>>(	pklist->d_maxsnglsnr, 
		    						pklist->d_tmplt_idx, 
								state->exe_len, 
								state->ntmplt, 
								pklist->d_peak_tmplt);

    ker_remove_duplicate_scan<<<GRID, THREAD_BLOCK>>>(	pklist->d_maxsnglsnr, 
		    					pklist->d_tmplt_idx, 
							state->exe_len, 
							state->ntmplt, 
							pklist->d_peak_tmplt);
}

/* calculate cohsnr, null stream, chi2 of a peak list and copy it back */
void cohsnr_and_chi2(PostcohState *state, int iifo, int gps_idx)
{
	int threads = 1024;
	int sharedmem	 = 3 * threads / WARP_SIZE * sizeof(float);
	PeakList *pklist = state->peak_list[iifo];
	ker_coh_sky_map_max<<<pklist->npeak[0], threads, sharedmem>>>(	pklist->d_cohsnr,
									pklist->d_nullsnr,
									pklist->d_pix_idx,
									pklist->d_chi2,
									state->d_snglsnr,
									iifo,	
									state->nifo,
									pklist->d_tmplt_idx,
									pklist->d_peak_pos,
									pklist->npeak[0],
									state->d_U_map[gps_idx],
									state->d_diff_map[gps_idx],
									state->npix,
									state->snglsnr_len,
									state->snglsnr_start_exe,
									state->dt,
									0
	);
}


/* calculate cohsnr, null stream, chi2 of a peak list and copy it back */
void cohsnr_and_chi2_background(PostcohState *state, int iifo, int hist_trials , int gps_idx)
{
}



#ifdef __cplusplus
extern "C" {
#endif
#include <stdio.h>

#include "postcoh_utils.h"
#include <cuda_debug.h>

#ifdef __cplusplus
}
#endif

const int GAMMA_ITMAX = 50;
// const float GAMMA_EPS = 2.22e-16;


#define WARP_SIZE 		32
#define LOG_WARP_SIZE	5

#define MIN_EPSILON 1e-7
#define MAXIFOS 6


// for gpu debug
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
static void gpuAssert(cudaError_t code, char *file, int line)
{
   if (code != cudaSuccess) 
   {
      printf ("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      exit(code);
   }
}

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
    int start_exe,
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
	    start_idx = (i + start_exe) % len;
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
    int*	npeak,			// Needs to be initialize to 0
    int*	peak_pos,
    float*  ressnr,
    int*    restemplate,
    int     timeN,
    int     templateN,
    float*  peak,
    float   snglsnr_thresh
)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int tn  = blockDim.x * gridDim.x;

    float   max_snr;
    float   snr;
    int     templ;

    unsigned int index;
    for (int i = tid; i < timeN; i += tn)
    {
        snr     = ressnr[i];
        templ   = restemplate[i];
        max_snr = peak[templ];

		if (abs(max_snr - snr) < MIN_EPSILON && max_snr > snglsnr_thresh)
		{
			index = atomicInc((unsigned *)npeak, timeN);
			peak_pos[index] = i;		
			/* FIXME: could be many peak positions for one peak */
//			peak[templ] = -1;
//			printf("peak tmplt %d, time %d, maxsnr %f, snr %f\n", templ, i, max_snr, snr);
		}
        // restemplate[i]  = ((-1 + templ) + (-1 - templ) * ((max_snr > snr) * 2 - 1)) >> 1;
        // ressnr[i]       = (-1.0f + snr) * 0.5 + (-1.0f - snr) * ((max_snr > snr) - 0.5);
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

__global__ void ker_coh_skymap
(
	float	*cohsnr_skymap,			/* OUTPUT, of size (num_triggers * num_sky_directions) */
	float	*nullsnr_skymap,	/* OUTPUT, of size (num_triggers * num_sky_directions) */
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
	int		exe_len,				/* INPUT, snglsnr length */
	int		start_exe,				/* INPUT, snglsnr start exe position */
	float		dt,			/* INPUT, 1/ sampling rate */
	int		templateN		/* INPUT, number of templates */
)
{
	int bid	= blockIdx.x;
	int bn	= gridDim.x;

	// float	*mu;	// matrix u for certain sky direction
	int		peak_cur, tmplt_cur;
	COMPLEX_F	dk[6];
	int		NtOff;
	int		map_idx;
	float	real, imag;
	float	utdka[6];
	float	al_all = 0.0f;

	for (int ipeak = bid; ipeak < npeak; ipeak += bn)
	{
		peak_cur = peak_pos[ipeak];
		tmplt_cur = tmplt_idx[peak_cur];

		for (int ipix = threadIdx.x; ipix < num_sky_directions; ipix += blockDim.x)
		{
			for (int j = 0; j < nifo; ++j)
			{
				/* this is a simplified algorithm to get map_idx */
				map_idx = iifo * nifo + j;
				NtOff = round (toa_diff_map[map_idx * num_sky_directions + ipix] / dt);
				/* NOTE: The snr is stored channel-wise */
				dk[j] = snr[j][((start_exe + peak_cur + NtOff + len) % len) * templateN + tmplt_cur ]; 	

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
					real += u_map[(j * nifo + k) * num_sky_directions + ipix] * dk[k].re;
					imag += u_map[(j * nifo + k) * num_sky_directions + ipix] * dk[k].im;
				}
				utdka[j] = real * real + imag * imag;	
			}		

			//cohsnr_skymap[ipeak * num_sky_directions + ipix] = (2 * (utdka[0] + utdka[1]) - 4) / sqrt(8.0f);
			cohsnr_skymap[ipeak * num_sky_directions + ipix] = utdka[0] + utdka[1];

			al_all = 0.0f;
			for (int j = 2; j < nifo; ++j)
				al_all += utdka[j];	
			nullsnr_skymap[ipeak * num_sky_directions + ipix] = 1 - gser(al_all, 1.0f);
		}
	}
}

__global__ void ker_coh_max_and_chisq
(
	float		*coh_snr,		/* OUTPUT, only save the max one, of size (num_triggers) */
	float		*coh_nullstream,	/* OUTPUT, only save the max one, of size (num_triggers) */
	float		*chisq,			/* OUTPUT, chisq value */
	int		*pix_idx,		/* OUTPUT, store the index of the sky_direction, of size (num_triggers) */
	COMPLEX_F	**snr,			/* INPUT, (2, 3) * data_points */	
	int		iifo,			/* INPUT, detector we are considering */
	int		nifo,			/* INPUT, all detectors that are in this coherent analysis */
	float		*maxsnglsnr,		/* INPUT, maximum single snr	*/
	int		*tmplt_idx,		/* INPUT, the tmplt index of triggers	*/
	int		*peak_pos,	/* INPUT, place the location of the trigger */
	int		npeak,		/* INPUT, number of triggers */
	float		*u_map,				/* INPUT, u matrix map */
	float		*toa_diff_map,		/* INPUT, time of arrival difference map */
	int		num_sky_directions,	/* INPUT, # of sky directions */
	int		len,				/* INPUT, snglsnr length */
	int		exe_len,				/* INPUT, snglsnr length */
	int		start_exe,				/* INPUT, snglsnr start exe position */
	float		dt,			/* INPUT, 1/ sampling rate */
	int		templateN,		/* INPUT, number of templates */
	int		autochisq_len,		/* INPUT, auto-chisq length */
	COMPLEX_F	**autocorr_matrix,	/* INPUT, autocorrelation matrix for all templates */
	float		**autocorr_norm,	/* INPUT, autocorrelation normalization matrix for all templates */
	int		hist_trials		/* INPUT, trial number */
)
{
	int bid	= blockIdx.x;
	int bn	= gridDim.x;

	int wn	= blockDim.x >> LOG_WARP_SIZE;
	int wID = threadIdx.x >> LOG_WARP_SIZE;		

	int srcLane = threadIdx.x & 0x1f;
	// store snr_max, nullstream_max and sky_idx, each has (blockDim.x / WARP_SIZE) elements
	extern __shared__ float smem[];
	volatile float *snr_shared = &smem[0];
	volatile float *nullstream_shared = &snr_shared[wn];
	volatile int *sky_idx_shared = (int*)&nullstream_shared[wn];

	// float	*mu;	// matrix u for certain sky direction
	int		peak_cur, tmplt_cur;
	COMPLEX_F	dk[MAXIFOS];
	int		NtOff;
	int		map_idx;
	float	real, imag;
	float	utdka[MAXIFOS];
	float	al_all = 0.0f;

	float	snr_max			= 0.0f, snr_tmp;
	float	nullstream_max, nullstream_max_tmp;
	int	sky_idx, sky_idx_tmp;	
	int	i, i2, itrial, trial_offset;

	for (int ipeak = bid; ipeak < npeak; ipeak += bn)
	{
		peak_cur = peak_pos[ipeak];
		tmplt_cur = tmplt_idx[peak_cur];

		for(itrial=0; itrial<=hist_trials; itrial++) {
			snr_max = 0.0;
			trial_offset = itrial * exe_len;
		for (int ipix = threadIdx.x; ipix < num_sky_directions; ipix += blockDim.x)
		{
			// matrix u is stored in column order
			// mu = u_map + nifo * nifo * i;			

			for (int j = 0; j < nifo; ++j)
			{
				/* this is a simplified algorithm to get map_idx */
				map_idx = iifo * nifo + j;
				NtOff = round (toa_diff_map[map_idx * num_sky_directions + ipix] / dt);
				/* NOTE: The snr is stored channel-wise */
				if (NtOff == 0)
					dk[j] = snr[j][((start_exe + peak_cur + len) % len) * templateN + tmplt_cur ]; 	
				else
					dk[j] = snr[j][((start_exe + peak_cur + NtOff - trial_offset + len) % len) * templateN + tmplt_cur ]; 	
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
#if 0
			if (ipix < 10) {
			printf("ipix %d, dk[0].re %f, dk[0].im %f," 
					"dk[1].re %f, dk[1].im %f, dk[2].re %f, dk[2].im %f,"
					"snr %f\n", ipix, dk[0].re, dk[0].im,
				       	dk[1].re, dk[1].im, dk[2].re, dk[2].im, snr_tmp);
			}
#endif
	
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

		for (i = WARP_SIZE / 2; i > 0; i = i >> 1)
		{
			i2 = i * 2;
			snr_tmp = __shfl(snr_max, srcLane + i, i2);
			nullstream_max_tmp = __shfl(nullstream_max, srcLane + i, i2);
			sky_idx_tmp = __shfl(sky_idx, srcLane + i, i2);

			if (snr_tmp > snr_max)
			{
				snr_max = snr_tmp;
				nullstream_max = nullstream_max_tmp;
				sky_idx = sky_idx_tmp;
			}
		}

		if (srcLane == 0)
		{
			snr_shared[wID]		= snr_max;
			nullstream_shared[wID]	= nullstream_max;
			sky_idx_shared[wID]	= sky_idx;
		}
		__syncthreads();

		for (i = wn >> 1; i > 0; i = i >> 1)
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
			coh_snr[peak_cur + trial_offset]			= snr_shared[0];
			coh_nullstream[peak_cur + trial_offset]	= nullstream_shared[0];
			pix_idx[peak_cur + trial_offset]		= sky_idx_shared[0]; 			

		}
		__syncthreads();

		/*
		COMPLEX_F data;
		chisq[peak_cur] = 0.0;
		int autochisq_half_len = autochisq_len /2;
		for (int j = 0; j < nifo; ++j)
		{	data = 0;
			// this is a simplified algorithm to get map_idx 
			map_idx = iifo * nifo + j;
			NtOff = round (toa_diff_map[map_idx * num_sky_directions + ipix] / dt);
			for(int ishift=-autochisq_half_len; ishift<=autochisq_half_len; ishift++)
			{

			// NOTE: The snr is stored channel-wise 
			data += snr[j][((start_exe + peak_cur + NtOff + ishift) % len) * templateN + tmplt_cur] - maxsnglsnr[peak_cur] * autocorr_matrix[j][ tmplt_cur * autochisq_len + ishift + autochisq_half_len]; 	
			}
			chisq[peak_cur] += (data.re * data.re + data.im * data.im) / autocorr_norm[j][tmplt_cur];
		}
		*/
#if 1
		COMPLEX_F data, tmp_snr, tmp_autocorr;
		float laneChi2 = 0.0f, tmp_maxsnr;
		int autochisq_half_len = autochisq_len /2, peak_pos_tmp;

		if (wID == 0)
		{
			for (int j = 0; j < nifo; ++j)
			{
				data.re = data.im = 0.0f; 
				/* this is a simplified algorithm to get map_idx */
				map_idx = iifo * nifo + j;
				NtOff = round (toa_diff_map[map_idx * num_sky_directions + pix_idx[peak_cur]] / dt);
				if (NtOff == 0)
					peak_pos_tmp = start_exe + peak_cur;
				else
					peak_pos_tmp = start_exe + peak_cur + NtOff - trial_offset + len;

				for (int ishift = srcLane - autochisq_half_len; ishift <= autochisq_half_len; ishift += WARP_SIZE)
				{
					/* NOTE: The snr is stored channel-wise */
					tmp_snr = snr[j][((peak_pos_tmp + ishift + len) % len) * templateN + tmplt_cur];
					tmp_autocorr = autocorr_matrix[j][ tmplt_cur * autochisq_len + ishift + autochisq_half_len];
					tmp_maxsnr =  maxsnglsnr[peak_cur]; 	
					data.re += tmp_snr.re - tmp_maxsnr * tmp_autocorr.re;
					data.im += tmp_snr.im - tmp_maxsnr * tmp_autocorr.im;
//					printf("autocorr_matrix %d, tmplt %d, [%d]: %f, %f\n", j, tmplt_cur, ishift + autochisq_half_len, tmp_autocorr.re, tmp_autocorr.im);
				}
				laneChi2 += (data.re * data.re + data.im * data.im) / autocorr_norm[j][tmplt_cur];	
//				printf("autocorr addr %p, autocorr_norm %d: addr %p, %f\n", autocorr_matrix[j], j, autocorr_norm[j], autocorr_norm[j][tmplt_cur]);
			}

			for (int j = WARP_SIZE >> 1; j > 0; j = j >> 1)
			{
				laneChi2 += __shfl_xor(laneChi2, j, WARP_SIZE);
			}

			if (srcLane == 0)
			{

				chisq[peak_cur + trial_offset] = laneChi2;
	//			printf("peak %d, itrial %d, cohsnr %f, nullstream %f, ipix %d, chisq %f\n", ipeak, itrial, coh_snr[peak_cur + trial_offset], coh_nullstream[peak_cur + trial_offset], pix_idx[peak_cur + trial_offset], chisq[peak_cur + trial_offset]);
			}
		}

		__syncthreads();
#endif
	}
	}
}

void peakfinder(PostcohState *state, int iifo)
{

    int THREAD_BLOCK    = 256;
    int GRID            = (state->exe_len * 32 + THREAD_BLOCK - 1) / THREAD_BLOCK;
    PeakList *pklist = state->peak_list[iifo];
    state_reset_npeak(pklist);

    ker_max_snglsnr<<<GRID, THREAD_BLOCK>>>(state->dd_snglsnr, 
						iifo,
						state->snglsnr_start_exe,
						state->snglsnr_len,
						state->ntmplt, 
						state->exe_len, 
						pklist->d_maxsnglsnr, 
						pklist->d_tmplt_idx);
   // gpuErrchk(cudaPeekAtLastError());

    GRID = (state->exe_len + THREAD_BLOCK - 1) / THREAD_BLOCK;
    ker_remove_duplicate_find_peak<<<GRID, THREAD_BLOCK>>>(	pklist->d_maxsnglsnr, 
		    						pklist->d_tmplt_idx, 
								state->exe_len, 
								state->ntmplt, 
								pklist->d_peak_tmplt);
    //gpuErrchk(cudaPeekAtLastError());

    ker_remove_duplicate_scan<<<GRID, THREAD_BLOCK>>>(	pklist->d_npeak,
	    						pklist->d_peak_pos,	    
		    					pklist->d_maxsnglsnr, 
		    					pklist->d_tmplt_idx, 
							state->exe_len, 
							state->ntmplt, 
							pklist->d_peak_tmplt,
							state->snglsnr_thresh);
  // gpuErrchk(cudaPeekAtLastError());
}

/* calculate cohsnr, null stream, chisq of a peak list and copy it back */
void cohsnr_and_chisq(PostcohState *state, int iifo, int gps_idx)
{
	int threads = 1024;
	int sharedmem	 = 3 * threads / WARP_SIZE * sizeof(float);
	PeakList *pklist = state->peak_list[iifo];
	int npeak = pklist->npeak[0];
	int mem_alloc_size = sizeof(float) * npeak * state->npix * 2;
//	printf("alloc cohsnr_skymap size %d\n", mem_alloc_size);
	CUDA_CHECK(cudaMalloc((void **)&(pklist->d_cohsnr_skymap), mem_alloc_size));
//	CUDA_CHECK(cudaMemset(pklist->d_cohsnr_skymap, 0, mem_alloc_size));

	pklist->d_nullsnr_skymap = pklist->d_cohsnr_skymap + npeak * state->npix;
	pklist->cohsnr_skymap = (float *)malloc(mem_alloc_size);
	pklist->nullsnr_skymap = pklist->cohsnr_skymap + npeak * state->npix;

	ker_coh_skymap<<<npeak, threads, sharedmem>>>(			pklist->d_cohsnr_skymap,
									pklist->d_nullsnr_skymap,
									state->dd_snglsnr,
									iifo,	
									state->nifo,
									pklist->d_tmplt_idx,
									pklist->d_peak_pos,
									pklist->npeak[0],
									state->d_U_map[gps_idx],
									state->d_diff_map[gps_idx],
									state->npix,
									state->snglsnr_len,
									state->exe_len,
									state->snglsnr_start_exe,
									state->dt,
									state->ntmplt);
						
//	gpuErrchk(cudaPeekAtLastError());

	ker_coh_max_and_chisq<<<npeak, threads, sharedmem>>>(	pklist->d_cohsnr,
									pklist->d_nullsnr,
									pklist->d_chisq,
									pklist->d_pix_idx,
									state->dd_snglsnr,
									iifo,	
									state->nifo,
									pklist->d_maxsnglsnr,
									pklist->d_tmplt_idx,
									pklist->d_peak_pos,
									pklist->npeak[0],
									state->d_U_map[gps_idx],
									state->d_diff_map[gps_idx],
									state->npix,
									state->snglsnr_len,
									state->exe_len,
									state->snglsnr_start_exe,
									state->dt,
									state->ntmplt,
									state->autochisq_len,
									state->dd_autocorr_matrix,
									state->dd_autocorr_norm,
									state->hist_trials);

//	gpuErrchk(cudaPeekAtLastError());
	/* copy the snr, cohsnr, nullsnr, chisq out */
	CUDA_CHECK(cudaMemcpy(	pklist->tmplt_idx, 
			pklist->d_tmplt_idx, 
			sizeof(int) * (pklist->peak_intlen), 
			cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaMemcpy(	pklist->maxsnglsnr, 
			pklist->d_maxsnglsnr, 
			sizeof(float) * (pklist->peak_floatlen), 
			cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaMemcpy(	pklist->cohsnr_skymap, 
			pklist->d_cohsnr_skymap, 
			mem_alloc_size,
			cudaMemcpyDeviceToHost));
}


/* calculate cohsnr, null stream, chisq of a peak list and copy it back */
void cohsnr_and_chisq_background(PostcohState *state, int iifo, int hist_trials , int gps_idx)
{
}


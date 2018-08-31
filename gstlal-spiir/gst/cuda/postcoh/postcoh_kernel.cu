/* 
 * Copyright (C) 2014 Xiaoyang Guo, Xiangyu Guo, Qi Chu <qi.chu@ligo.org>
 * 
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */


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
#define WARP_MASK		31
#define LOG_WARP_SIZE	5

#define MIN_EPSILON 1e-7
#define MAXIFOS 6
#define NSKY_REDUCE_RATIO 4


#if 0
// deprecated: we have cuda_debug.h for gpu debug now
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
static void gpuAssert(cudaError_t code, char *file, int line)
{
   if (code != cudaSuccess) 
   {
      printf ("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      exit(code);
   }
}
#endif

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
    int     ntmplt,  // INPUT: template number
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
        for (int j = wIDl; j < ntmplt; j += WARP_SIZE)
        {
	    index = start_idx * ntmplt + j;
	
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
    int     ntmplt,
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
    int     ntmplt,
    float*  peak            // peak is used for storing intermediate result, it is of size ntmplt
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
    int     ntmplt,
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
    /* make sure npeak is 0 at the beginning */
    if (tid == 0)
	    npeak[0] = 0;
    __syncthreads();

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
        // NOTE: Bottleneck, previous code, del *= x / ap;, use __fdividef for fast math
        del *= __fdividef(x , ap); 
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
	int		*peak_pos,	/* INPUT, place the location of the trigger */
	int		npeak,		/* INPUT, number of triggers */
	float		*u_map,				/* INPUT, u matrix map */
	float		*toa_diff_map,		/* INPUT, time of arrival difference map */
	int		num_sky_directions,	/* INPUT, # of sky directions */
	int		max_npeak,				/* INPUT, max number of peaks */
	int		len,				/* INPUT, snglsnr length */
	int		start_exe,				/* INPUT, snglsnr start exe position */
	float		dt,			/* INPUT, 1/ sampling rate */
	int		ntmplt		/* INPUT, number of templates */
)
{
    int wn  = blockDim.x >> LOG_WARP_SIZE;
    int wID = threadIdx.x >> LOG_WARP_SIZE;     

    int     peak_cur, tmplt_cur, len_cur;
    COMPLEX_F   dk[MAXIFOS];
    int     NtOff;
    int     map_idx, ipix, i;
    float   real, imag;
    float   snr_tmp, al_all = 0.0f;


	if (npeak > 0)
    {
        peak_cur = peak_pos[0];
        // find the len_cur from len_idx
        len_cur = peak_pos[peak_cur + max_npeak];
        // find the tmplt_cur from tmplt_idx
        tmplt_cur = peak_pos[peak_cur + 2 * max_npeak];

        for (int seed_pix = threadIdx.x; seed_pix < num_sky_directions; seed_pix += blockDim.x)
        {
            snr_tmp = 0.0;
            al_all = 0.0;
            // matrix u is stored in column order
            // mu = u_map + nifo * nifo * i;            
            ipix = seed_pix;

            for (int j = 0; j < nifo; ++j)
            {
                /* this is a simplified algorithm to get map_idx */
                map_idx = iifo * nifo + j;
                NtOff = round (toa_diff_map[map_idx * num_sky_directions + ipix] / dt);
                NtOff = (j == iifo ? 0 : NtOff);
                // dk[j] = snr[j][((start_exe + len_cur + NtOff + len) % len) * ntmplt + tmplt_cur ];  
                dk[j] = snr[j][tmplt_cur * len + ((start_exe + len_cur + NtOff + len) % len)];  
            }

            for (int j = 0; j < nifo; ++j)
            {
                real = 0.0f;
                imag = 0.0f;
                for (int k = 0; k < nifo; ++k)
                {
                    real += u_map[(j * nifo + k) * num_sky_directions + ipix] * dk[k].re;
                    imag += u_map[(j * nifo + k) * num_sky_directions + ipix] * dk[k].im;
                }
                (j < 2 ? snr_tmp: al_all) += real * real + imag * imag;   
            }   

			cohsnr_skymap[ipix] = snr_tmp;
			nullsnr_skymap[ipix] = al_all;
		}
	}
}

__global__ void ker_coh_max_and_chisq
(
    COMPLEX_F   **snr,          /* INPUT, (2, 3) * data_points */   
    int     iifo,           /* INPUT, detector we are considering */
    int     nifo,           /* INPUT, all detectors that are in this coherent analysis */
    int     *write_ifo_mapping,  /* INPUT, write-ifo-mapping */
    int     *peak_pos,  /* INPUT, place the location of the trigger */
    float       *snglsnr_H,     /* INPUT, maximum single snr    */
    float       *snglsnr_bg_H,      /* INPUT, maximum single snr    */
    float       *cohsnr_skymap,
    float       *nullsnr_skymap,
	int			output_skymap,	/* INPUT, whether to write to cohsnr_skymap and nullsnr_skymap */
    int     npeak,      /* INPUT, number of triggers */
    float       *u_map,             /* INPUT, u matrix map */
    float       *toa_diff_map,      /* INPUT, time of arrival difference map */
    int     num_sky_directions, /* INPUT, # of sky directions */
    int     len,                /* INPUT, snglsnr length */
    int     max_npeak,              /* INPUT, snglsnr length */
    int     start_exe,              /* INPUT, snglsnr start exe position */
    float       dt,         /* INPUT, 1/ sampling rate */
    int     ntmplt,     /* INPUT, number of templates */
    int     autochisq_len,      /* INPUT, auto-chisq length */
    COMPLEX_F   **autocorr_matrix,  /* INPUT, autocorrelation matrix for all templates */
    float       **autocorr_norm,    /* INPUT, autocorrelation normalization matrix for all templates */
    int     hist_trials,        /* INPUT, trial number */
    int     trial_sample_inv        /* INPUT, trial interval in samples */
)
{
    int bid = blockIdx.x;
    int bn  = gridDim.x;

    int wn  = blockDim.x >> LOG_WARP_SIZE;
    int wID = threadIdx.x >> LOG_WARP_SIZE;     

    int srcLane = threadIdx.x & 0x1f, ipix; // binary: 11111, decimal 31
    // store snr_max, nullstream_max and sky_idx, each has (blockDim.x / WARP_SIZE) elements
    extern __shared__ float smem[];
    volatile float *stat_shared = &smem[0];
    volatile float *snr_shared = &stat_shared[wn];
    volatile float *nullstream_shared = &snr_shared[wn];
    volatile int *sky_idx_shared = (int*)&nullstream_shared[wn];

    // float    *mu;    // matrix u for certain sky direction
    int     peak_cur, tmplt_cur, ipeak_max = 0;
    COMPLEX_F   dk[MAXIFOS];
    int     NtOff;
    int     map_idx;
    float   real, imag;
    float   al_all = 0.0f, chisq_cur;

    float   stat_max, stat_tmp;
    float   snr_max, snr_tmp;
    float   nullstream_max, nullstream_max_tmp;
    int sky_idx = 0, sky_idx_tmp = 0;   
    int i, itrial, trial_offset, output_offset, len_cur;
    int *pix_idx = peak_pos + 3 * max_npeak;
    int *pix_idx_bg = peak_pos + 4 * max_npeak;
    float   *cohsnr = snglsnr_H + 9 * max_npeak;
    float   *nullsnr = snglsnr_H + 10 * max_npeak;
    float   *cmbchisq = snglsnr_H + 11 * max_npeak;
    float   *cohsnr_bg = snglsnr_H + (12 + 9*hist_trials) * max_npeak;
    float   *nullsnr_bg = snglsnr_H + (12 + 10*hist_trials) * max_npeak;
    float   *cmbchisq_bg = snglsnr_H + (12 + 11*hist_trials) * max_npeak;

    for (int ipeak = bid; ipeak < npeak; ipeak += bn)
    {
        peak_cur = peak_pos[ipeak];
        // find the len_cur from len_idx
        len_cur = peak_pos[peak_cur + max_npeak];
        // find the tmplt_cur from tmplt_idx
        tmplt_cur = peak_pos[peak_cur + 2 * max_npeak];

        itrial = 0;
        stat_max = 0.0;
        snr_max = 0.0;
        nullstream_max = 0.0f;
        sky_idx = 0;

        for (int seed_pix = threadIdx.x; seed_pix < num_sky_directions/NSKY_REDUCE_RATIO; seed_pix += blockDim.x)
        {
            ipix = seed_pix * NSKY_REDUCE_RATIO;

            snr_tmp = 0.0;
            al_all = 0.0;
            // matrix u is stored in column order
            // mu = u_map + nifo * nifo * i;            

            for (int j = 0; j < nifo; ++j)
            {
                /* this is a simplified algorithm to get map_idx */
                map_idx = iifo * nifo + j;
                NtOff = round (toa_diff_map[map_idx * num_sky_directions + ipix] / dt);
                NtOff = (j == iifo ? 0 : NtOff);
                // dk[j] = snr[j][((start_exe + len_cur + NtOff + len) % len) * ntmplt + tmplt_cur ];  
                dk[j] = snr[j][tmplt_cur * len + ((start_exe + len_cur + NtOff + len) % len)];  
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
                (j < 2 ? snr_tmp: al_all) += real * real + imag * imag;   
            }   
#if 0
            if (ipix < 3072) {
            printf("ipeak %d, ipix %d, trial %d, dk[0].re %f, dk[0].im %f," 
                    "dk[1].re %f, dk[1].im %f, dk[2].re %f, dk[2].im %f,"
                    "snr %f\n", ipeak, ipix, itrial, dk[0].re, dk[0].im,
                        dk[1].re, dk[1].im, dk[2].re, dk[2].im, snr_max);
            }
#endif

            stat_tmp = snr_tmp - 0.0;
            if (stat_tmp > stat_max)
            {
                stat_max = stat_tmp;
                snr_max = snr_tmp;
                nullstream_max = al_all;
                sky_idx = ipix;
            }
        }

        for (i = WARP_SIZE / 2; i > 0; i = i >> 1)
        {
            stat_tmp = __shfl_xor(stat_max, i);
            snr_tmp = __shfl_xor(snr_max, i);
            nullstream_max_tmp = __shfl_xor(nullstream_max, i);
            sky_idx_tmp = __shfl_xor(sky_idx, i);

            if (stat_tmp > stat_max)
            {
                stat_max = stat_tmp;
                snr_max = snr_tmp;
                nullstream_max = nullstream_max_tmp;
                sky_idx = sky_idx_tmp;
            }
        }

        if (srcLane == 0)
        {
            stat_shared[wID]    = stat_max;
            snr_shared[wID]     = snr_max;
            nullstream_shared[wID]  = nullstream_max;
            sky_idx_shared[wID] = sky_idx;
        }
        __syncthreads();

        for (i = wn >> 1; i > 0; i = i >> 1)
        {
            if (threadIdx.x < i)
            {
                stat_tmp = stat_shared[threadIdx.x + i];
                stat_max = stat_shared[threadIdx.x];

                if (stat_tmp > stat_max)
                {
                    stat_shared[threadIdx.x] = stat_tmp;
                    snr_shared[threadIdx.x] = snr_shared[threadIdx.x + i];
                    nullstream_shared[threadIdx.x] = nullstream_shared[threadIdx.x + i];
                    sky_idx_shared[threadIdx.x] = sky_idx_shared[threadIdx.x + i];
                }

            }   
            __syncthreads();

        }
        if (threadIdx.x == 0)
        {
            cohsnr[peak_cur]    = snr_shared[0];
			nullsnr[peak_cur]   = nullstream_shared[0];
            pix_idx[peak_cur]   = sky_idx_shared[0];            
        }
        __syncthreads();

		/* chisq calculation */

        COMPLEX_F data, tmp_snr, tmp_autocorr, tmp_maxsnr;
        float laneChi2 = 0.0f;
        int autochisq_half_len = autochisq_len /2, peak_pos_tmp;

        cmbchisq[peak_cur] = 0.0;
        
        for (int j = 0; j < nifo; ++j)
        {
            laneChi2 = 0.0f;
            /* this is a simplified algorithm to get map_idx */
            map_idx = iifo * nifo + j;
            NtOff = round (toa_diff_map[map_idx * num_sky_directions + pix_idx[peak_cur]] / dt);
            peak_pos_tmp = start_exe + len_cur + (j == iifo ? 0 : NtOff + len);

            // tmp_maxsnr = snr[j][((peak_pos_tmp + len) % len) * ntmplt + tmplt_cur];
            tmp_maxsnr = snr[j][len * tmplt_cur + ((peak_pos_tmp + len) % len)];
            /* set the ntoff; snglsnr; and coa_phase for each detector */
            /* set the ntoff; this is actually d_ntoff_*  */
            peak_pos[peak_cur + (4 + hist_trials + write_ifo_mapping[j]) * max_npeak] = NtOff;
            /* set the d_snglsnr_* */
            snglsnr_H[peak_cur + write_ifo_mapping[j] * max_npeak] =  sqrt(tmp_maxsnr.re * tmp_maxsnr.re + tmp_maxsnr.im * tmp_maxsnr.im);
            /* set the d_coa_phase_* */
            snglsnr_H[peak_cur + (3 + write_ifo_mapping[j]) * max_npeak] = atan(tmp_maxsnr.re / tmp_maxsnr.im);
            
            for (int ishift = threadIdx.x - autochisq_half_len; ishift <= autochisq_half_len; ishift += blockDim.x)
            {
                tmp_snr = snr[j][len * tmplt_cur + ((peak_pos_tmp + ishift + len) % len)];
                tmp_autocorr = autocorr_matrix[j][ tmplt_cur * autochisq_len + ishift + autochisq_half_len];
                data.re = tmp_snr.re - tmp_maxsnr.re * tmp_autocorr.re + tmp_maxsnr.im * tmp_autocorr.im;
                data.im = tmp_snr.im - tmp_maxsnr.re * tmp_autocorr.im - tmp_maxsnr.im * tmp_autocorr.re;
                laneChi2 += (data.re * data.re + data.im * data.im);    
            }
            for (int k = WARP_SIZE >> 1; k > 0; k = k >> 1)
            {
                laneChi2 += __shfl_xor(laneChi2, k, WARP_SIZE);
            }
            if (srcLane == 0) {
                snr_shared[wID] = laneChi2; 
            }
            __syncthreads();
            if (threadIdx.x < wn) {
                laneChi2 = snr_shared[srcLane];
                for (i = wn / 2; i > 0; i = i >> 1) {
                    laneChi2 += __shfl_xor(laneChi2, i);
                }
                if (srcLane == 0)
                {
                    chisq_cur = laneChi2/ autocorr_norm[j][tmplt_cur];
                    // the location of chisq_* is indexed from maxsnglsnr
                    snglsnr_H[peak_cur + (6 + write_ifo_mapping[j]) * max_npeak] = chisq_cur;

                    cmbchisq[peak_cur] += chisq_cur;
                    //printf("peak %d, itrial %d, cohsnr %f, nullstream %f, ipix %d, chisq %f\n", ipeak, itrial, cohsnr[peak_cur], nullsnr[peak_cur], pix_idx[peak_cur], cmbchisq[peak_cur]);
                }
            }
            __syncthreads();
        }

        __syncthreads();

        /*
         *
         * Generate background cohsnr; nullsnr; chisq
         *
         */

        int ipix = 0, rand_range = trial_sample_inv * hist_trials -1;
        for(itrial=1+threadIdx.x/WARP_SIZE; itrial<=hist_trials; itrial+=blockDim.x/WARP_SIZE) {
            snr_max = 0.0;
            nullstream_max = 0.0;
            sky_idx = 0;
            
            // FIXME: try using random offset like the following
            //trial_offset = rand()% rand_range + 1;
            trial_offset = itrial * trial_sample_inv;
            output_offset = peak_cur + (itrial - 1)* max_npeak;
        for (int seed_pix = srcLane; seed_pix < num_sky_directions/NSKY_REDUCE_RATIO; seed_pix += WARP_SIZE)
        {
            snr_tmp = 0.0;
            al_all = 0.0;
            // matrix u is stored in column order
            // mu = u_map + nifo * nifo * i;            

            ipix = (seed_pix * NSKY_REDUCE_RATIO) + (itrial & (NSKY_REDUCE_RATIO - 1));
            for (int j = 0; j < nifo; ++j)
            {
                /* this is a simplified algorithm to get map_idx */
                map_idx = iifo * nifo + j;
    
                NtOff = round (toa_diff_map[map_idx * num_sky_directions + ipix] / dt);
                // The background cohsnr should be obtained coherently as well.
                int offset = (j == iifo ? 0 : NtOff - trial_offset);
                // dk[j] = snr[j][((start_exe + len_cur + offset + len) % len) * ntmplt + tmplt_cur ];     
                dk[j] = snr[j][len * tmplt_cur + ((start_exe + len_cur + offset + len) % len)];     
            }

            for (int j = 0; j < nifo; ++j)
            {
                real = 0.0f;
                imag = 0.0f;
                for (int k = 0; k < nifo; ++k)
                {
                    real += u_map[(j * nifo + k) * num_sky_directions + ipix] * dk[k].re;
                    imag += u_map[(j * nifo + k) * num_sky_directions + ipix] * dk[k].im;
                }
                (j < 2 ? snr_tmp: al_all) += real * real + imag * imag;   
            }   
#if 0
            if (itrial==1 && threadIdx.x == 1) {
            printf("iifo %d, ipeak %d, ipix %d, trial %d, trial_offset %d, trial_sample_inv %d, dk[0].re %f, dk[0].im %f," 
                    "dk[1].re %f, dk[1].im %f, dk[2].re %f, dk[2].im %f,"
                    "snr %f\n", iifo, ipeak, ipix, itrial, trial_offset, trial_sample_inv, dk[0].re, dk[0].im,
                        dk[1].re, dk[1].im, dk[2].re, dk[2].im, snr_tmp);
            }
#endif
    
            stat_tmp = snr_tmp - 0.0;
            if (stat_tmp > stat_max)
            {
                stat_max = stat_tmp;
                snr_max = snr_tmp;
                nullstream_max = al_all;
                sky_idx = ipix;
            }
        }

        for (i = WARP_SIZE / 2; i > 0; i = i >> 1)
        {
            stat_tmp = __shfl_xor(stat_max, i);
            snr_tmp = __shfl_xor(snr_max, i);
            nullstream_max_tmp = __shfl_xor(nullstream_max, i);
            sky_idx_tmp = __shfl_xor(sky_idx, i);

            if (stat_tmp > stat_max)
            {
                stat_max = stat_tmp;
                snr_max = snr_tmp;
                nullstream_max = nullstream_max_tmp;
                sky_idx = sky_idx_tmp;
            }
        }
        if (srcLane == 0)
        {
            cohsnr_bg[output_offset]        = snr_max;
			nullsnr_bg[output_offset]   = nullstream_max;
            /* background need this for Ntoff */
            pix_idx_bg[output_offset]       = sky_idx;          

        }
        __syncthreads();

        /*
        COMPLEX_F data;
        chisq[peak_cur] = 0.0;
        int autochisq_half_len = autochisq_len /2;
        for (int j = 0; j < nifo; ++j)
        {   data = 0;
            // this is a simplified algorithm to get map_idx 
            map_idx = iifo * nifo + j;
            NtOff = round (toa_diff_map[map_idx * num_sky_directions + ipix] / dt);
            for(int ishift=-autochisq_half_len; ishift<=autochisq_half_len; ishift++)
            {

            data += snr[j][((start_exe + peak_cur + NtOff + ishift) % len) * ntmplt + tmplt_cur] - maxsnglsnr[peak_cur] * autocorr_matrix[j][ tmplt_cur * autochisq_len + ishift + autochisq_half_len];     
            }
            chisq[peak_cur] += (data.re * data.re + data.im * data.im) / autocorr_norm[j][tmplt_cur];
        }
        */
#if 1
        COMPLEX_F data, tmp_snr, tmp_autocorr, tmp_maxsnr;
        float laneChi2 = 0.0f;
        int autochisq_half_len = autochisq_len /2, peak_pos_tmp;

        cmbchisq_bg[output_offset] = 0.0;
        
        for (int j = 0; j < nifo; ++j)
        {
            laneChi2 = 0.0f;
            /* this is a simplified algorithm to get map_idx */
            map_idx = iifo * nifo + j;
            NtOff = round (toa_diff_map[map_idx * num_sky_directions + pix_idx_bg[output_offset]] / dt);

            peak_pos_tmp = start_exe + len_cur + (j == iifo ? 0 : NtOff - trial_offset + len);

            // tmp_maxsnr = snr[j][((peak_pos_tmp + len) % len) * ntmplt + tmplt_cur];
            tmp_maxsnr = snr[j][len * tmplt_cur + ((peak_pos_tmp + len) % len)];
            /* set the d_snglsnr_* */
            snglsnr_bg_H[output_offset + write_ifo_mapping[j] * hist_trials * max_npeak] =  sqrt(tmp_maxsnr.re * tmp_maxsnr.re + tmp_maxsnr.im * tmp_maxsnr.im);
            /* set the d_coa_phase_* */
            snglsnr_bg_H[output_offset + (3 + write_ifo_mapping[j]) * hist_trials * max_npeak] = atan(tmp_maxsnr.re / tmp_maxsnr.im);

#if 0
            if (threadIdx.x == 1 && ipeak == 0)
            printf("ipeak %d, itrial %d, trial_offset %d, NtOff %d, maxsnr.re %f, maxsnr.im %f\n", ipeak, itrial, trial_offset, NtOff, tmp_maxsnr.re, tmp_maxsnr.im);
#endif

            for (int ishift = srcLane - autochisq_half_len; ishift <= autochisq_half_len; ishift += WARP_SIZE)
            {
                // tmp_snr = snr[j][((peak_pos_tmp + ishift + len) % len) * ntmplt + tmplt_cur];
                tmp_snr = snr[j][len * tmplt_cur + ((peak_pos_tmp + ishift + len) % len)];
                tmp_autocorr = autocorr_matrix[j][ tmplt_cur * autochisq_len + ishift + autochisq_half_len];
                data.re = tmp_snr.re - tmp_maxsnr.re * tmp_autocorr.re + tmp_maxsnr.im * tmp_autocorr.im;
                data.im = tmp_snr.im - tmp_maxsnr.re * tmp_autocorr.im - tmp_maxsnr.im * tmp_autocorr.re;
                laneChi2 += (data.re * data.re + data.im * data.im);    
            }
            for (int k = WARP_SIZE >> 1; k > 0; k = k >> 1)
            {
                laneChi2 += __shfl_xor(laneChi2, k, WARP_SIZE);
            }

            if (srcLane == 0)
            {
                chisq_cur = laneChi2/ autocorr_norm[j][tmplt_cur];
                // set d_chisq_bg_* from snglsnr_bg_H
                snglsnr_bg_H[output_offset + (6 + write_ifo_mapping[j]) * hist_trials * max_npeak] = chisq_cur;


                cmbchisq_bg[output_offset] += chisq_cur;
#if 0
                if (ipeak == 0)
                printf("iifo%d, jifo %d, peak %d, start %d, itrial %d, trial_offset %d, ntoff %d, off_pos %d, len %d, cohsnr %f, nullstream %f, ipix %d, cmbchisq %f\n", iifo, j, ipeak, start_exe+len_cur, itrial, trial_offset, NtOff, peak_pos_tmp, len, cohsnr[peak_cur], nullsnr[peak_cur], pix_idx[peak_cur], cmbchisq[peak_cur]);
#endif
            }
            __syncthreads();
        }

        __syncthreads();
#endif
    }
    }
	/* find maximum cohsnr and swope to the first pos */
    volatile float *cohsnr_shared = &smem[0];
    volatile float *ipeak_shared = &smem[blockDim.x];
	float cohsnr_max = 0.0, cohsnr_cur;

	if (bid == 0 && npeak > 0) {
		/* clean up smem history */
		cohsnr_shared[threadIdx.x] = 0.0;
		ipeak_shared[threadIdx.x] = 0;
		__syncthreads();
	    for (i = threadIdx.x; i < npeak; i += blockDim.x) {
			peak_cur = peak_pos[i];
			cohsnr_cur = cohsnr[peak_cur];
			if (cohsnr_cur > cohsnr_max) {
				cohsnr_shared[threadIdx.x] = cohsnr_cur;
				ipeak_shared[threadIdx.x] = i;
				cohsnr_max = cohsnr_cur;
			}
		}
		__syncthreads();
	    for (i = wn >> 1; i > 0; i = i >> 1)
	    {
	        if (threadIdx.x < i)
	        {
				cohsnr_cur = cohsnr_shared[threadIdx.x + i];
	            cohsnr_max = cohsnr_shared[threadIdx.x];
	
	            if (cohsnr_cur > cohsnr_max)
	            {
	                cohsnr_shared[threadIdx.x] = cohsnr_cur;
	                ipeak_shared[threadIdx.x] = ipeak_shared[threadIdx.x + i];
	            }
	
	        }   
	            __syncthreads();
	    }
	
		/* swope the first and max peak_cur in peak_pos */
	
	    if (threadIdx.x == 0)
	    {
			ipeak_max = ipeak_shared[0];
			peak_cur = peak_pos[ipeak_max];
			peak_pos[ipeak_max] = peak_pos[0];
			peak_pos[0] = peak_cur;
	    }

	} 
}

#define TRANSPOSE_TILE_DIM 32
#define TRANSPOSE_BLOCK_ROWS 8

__global__ void transpose_matrix(COMPLEX_F* idata, 
                COMPLEX_F* odata, 
                int in_x_offset, int in_y_offset, 
                int in_width, int in_height, 
                int out_x_offset, int out_y_offset,
                int out_width, int out_height,
                int copy_width, int copy_height){  // for in matrix
                
    // sizeof(COMPLEX_F) == 8 bytes
    // 32 shared memory banks
    __shared__ COMPLEX_F tile[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM + 1];

    int x = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.y;

    if (x < copy_width) {
        for (int j = 0; j < TRANSPOSE_TILE_DIM && y + j < copy_height; j += TRANSPOSE_BLOCK_ROWS)
            tile[threadIdx.y + j][threadIdx.x] = idata[((in_y_offset + y + j) % in_height) * in_width + (in_x_offset + x) % in_width];
    }
    
    __syncthreads();

    x = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.x;  // transpose block offset
    y = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.y;

    if (x < copy_height) {
        for (int j = 0; j < TRANSPOSE_TILE_DIM && y + j < copy_width; j += TRANSPOSE_BLOCK_ROWS)
            odata[((out_y_offset + y + j) % out_height) * out_width + (out_x_offset + x) % out_width] = tile[threadIdx.x][threadIdx.y + j];    
    }
}

void transpose_snglsnr(COMPLEX_F* idata, COMPLEX_F* odata, int offset, int copy_snglsnr_len, int snglsnr_len, int tmplt_len, cudaStream_t stream) {
    // input shape [height, width] [copy_snglsnr_len, tmplt_len]
    // output shape [snglsnr_len, tmplt_len]^T
    dim3 grid((tmplt_len + TRANSPOSE_TILE_DIM - 1) / TRANSPOSE_TILE_DIM, (copy_snglsnr_len + TRANSPOSE_TILE_DIM - 1) / TRANSPOSE_TILE_DIM);
    dim3 block(TRANSPOSE_TILE_DIM, TRANSPOSE_BLOCK_ROWS);
    transpose_matrix<<<grid, block, 0, stream>>>(idata, odata, 
                                                0, 0, tmplt_len, copy_snglsnr_len, 
                                                offset, 0, snglsnr_len, tmplt_len,
                                                tmplt_len, copy_snglsnr_len);
}

void peakfinder(PostcohState *state, int iifo, cudaStream_t stream)
{

    int THREAD_BLOCK    = 256;
    int GRID            = (state->exe_len * 32 + THREAD_BLOCK - 1) / THREAD_BLOCK;
    PeakList *pklist = state->peak_list[iifo];
//    state_reset_npeak(pklist);

    ker_max_snglsnr<<<GRID, THREAD_BLOCK, 0, stream>>>(state->dd_snglsnr, 
						iifo,
						state->snglsnr_start_exe,
						state->snglsnr_len,
						state->ntmplt, 
						state->exe_len, 
						pklist->d_maxsnglsnr, 
						pklist->d_tmplt_idx);
    cudaStreamSynchronize(stream);
    CUDA_CHECK(cudaPeekAtLastError());

    GRID = (state->exe_len + THREAD_BLOCK - 1) / THREAD_BLOCK;
    ker_remove_duplicate_find_peak<<<GRID, THREAD_BLOCK, 0, stream>>>(
						 	    pklist->d_maxsnglsnr, 
		    						pklist->d_tmplt_idx, 
								state->exe_len, 
								state->ntmplt, 
								pklist->d_peak_tmplt);

    cudaStreamSynchronize(stream);
    CUDA_CHECK(cudaPeekAtLastError());

    ker_remove_duplicate_scan<<<GRID, THREAD_BLOCK, 0, stream>>>(	pklist->d_npeak,
	    						pklist->d_peak_pos,	    
		    					pklist->d_maxsnglsnr, 
		    					pklist->d_tmplt_idx, 
							state->exe_len, 
							state->ntmplt, 
							pklist->d_peak_tmplt,
							state->snglsnr_thresh);

    cudaStreamSynchronize(stream);
    CUDA_CHECK(cudaPeekAtLastError());
}


/* calculate cohsnr, null stream, chisq of a peak list and copy it back */
void cohsnr_and_chisq(PostcohState *state, int iifo, int gps_idx, int output_skymap, cudaStream_t stream)
{
	size_t freemem;
	size_t totalmem;

	int threads = 256;
    int sharedsize     = MAX(2*threads*sizeof(float), 4 * threads / WARP_SIZE * sizeof(float));
	PeakList *pklist = state->peak_list[iifo];
	int npeak = pklist->npeak[0];

	ker_coh_max_and_chisq<<<npeak, threads, sharedsize, stream>>>(
									state->dd_snglsnr,
									iifo,	
									state->nifo,
									state->d_write_ifo_mapping,
									pklist->d_peak_pos,
									pklist->d_snglsnr_H,
									pklist->d_snglsnr_bg_H,
									pklist->d_cohsnr_skymap,
									pklist->d_nullsnr_skymap,
									output_skymap,
									pklist->npeak[0],
									state->d_U_map[gps_idx],
									state->d_diff_map[gps_idx],
									state->npix,
									state->snglsnr_len,
									state->max_npeak,
									state->snglsnr_start_exe,
									state->dt,
									state->ntmplt,
									state->autochisq_len,
									state->dd_autocorr_matrix,
									state->dd_autocorr_norm,
									state->hist_trials,
									state->trial_sample_inv);

	CUDA_CHECK(cudaStreamSynchronize(stream));
	CUDA_CHECK(cudaPeekAtLastError());

	if(output_skymap && state->snglsnr_max > MIN_OUTPUT_SKYMAP_SNR)
	{
		ker_coh_skymap<<<1, threads, sharedsize, stream>>>(
									pklist->d_cohsnr_skymap,
									pklist->d_nullsnr_skymap,
									state->dd_snglsnr,
									iifo,	
									state->nifo,
									pklist->d_peak_pos,
									pklist->npeak[0],
									state->d_U_map[gps_idx],
									state->d_diff_map[gps_idx],
									state->npix,
									state->max_npeak,
									state->snglsnr_len,
									state->snglsnr_start_exe,
									state->dt,
									state->ntmplt
									);

		CUDA_CHECK(cudaStreamSynchronize(stream));
		CUDA_CHECK(cudaPeekAtLastError());

		CUDA_CHECK(cudaMemcpyAsync(pklist->cohsnr_skymap,
			pklist->d_cohsnr_skymap,
			sizeof(float) * state->npix * 2,
			cudaMemcpyDeviceToHost,
			stream));

	}

	/* copy the snr, cohsnr, nullsnr, chisq out */
	CUDA_CHECK(cudaMemcpyAsync(	pklist->snglsnr_H, 
			pklist->d_snglsnr_H, 
			sizeof(float) * (pklist->peak_floatlen), 
			cudaMemcpyDeviceToHost,
			stream));

	CUDA_CHECK(cudaMemcpyAsync(	pklist->npeak, 
			pklist->d_npeak, 
			sizeof(int) * (pklist->peak_intlen), 
			cudaMemcpyDeviceToHost,
			stream));


	cudaStreamSynchronize(stream);
}


/* calculate cohsnr, null stream, chisq of a peak list and copy it back */
void cohsnr_and_chisq_background(PostcohState *state, int iifo, int hist_trials , int gps_idx)
{
}


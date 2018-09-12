/* 
 * Copyright (C) 2014 Qi Chu
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more deroll-offss.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#ifndef __CUDA_MULTIRATESPIIR_UTILS_H__
#define __CUDA_MULTIRATESPIIR_UTILS_H__

#include <multiratespiir/multiratespiir.h>
#define RESAMPLER_NUM_DEPTHS_MIN 0 
#define RESAMPLER_NUM_DEPTHS_MAX 7 
#define RESAMPLER_NUM_DEPTHS_DEFAULT 7 
#define MATRIX_DEFAULT 1

#define SPSTATE(i) (*(spstate+i)) 
#define SPSTATEDOWN(i) (SPSTATE(i)->downstate)
#define SPSTATEUP(i) (SPSTATE(i)->upstate)

/* The quality of downsampler should be choosen carefully. Tests show that
 * quality = 6 will produce 1.5 times single events than quality = 9
 * The real filter length = base_len * sample_ratio.
 * e.g. if we downsample 2x, filt_len = 192 * 2 = 384
 */
#define DOWN_FILT_LEN 192 
#define DOWN_QUALITY 9

/* The quality of upsampler can be as small as 1. It won't affect the
 * number of single events
 */
#define UP_FILT_LEN 16 
#define UP_QUALITY 1


typedef enum {
SP_OK = 0,
SP_RESAMPLER_NOT_INITED = -1,
SP_BANK_LOAD_ERR = -2
} SpInitReturn;


ResamplerState *
resampler_state_create (gint inrate, gint outrate, gint channels, gint num_exe_samples, gint num_cover_samples, gint depth, cudaStream_t stream);

void 
resampler_state_reset (ResamplerState *state, cudaStream_t stream);

void 
resampler_state_destroy (ResamplerState *state);

SpiirState **
spiir_state_create (const gchar *bank_fname, guint ndepth, guint rate, guint num_head_cover_samples,
		gint num_exe_samples, cudaStream_t stream);

void 
spiir_state_destroy (SpiirState ** spstate, guint num_depths);

void
spiir_state_reset (SpiirState **spstate, guint num_depths, cudaStream_t stream);

gint
spiir_state_get_outlen (SpiirState **spstate, gint in_len, guint num_depths);


void
cuda_multiratespiir_read_bank_id(const char *fname, gint *bank_id);

void
cuda_multiratespiir_read_ndepth_and_rate(const char *fname, guint *num_depths, gint *rate);

void 
cuda_multiratespiir_init_cover_samples (guint *num_head_cover_samples, 
		guint *num_tail_cover_samples, gint rate, guint num_depths, 
		gint down_filtlen, gint up_filtlen);

void 
cuda_multiratespiir_update_exe_samples (gint *num_exe_samples, gint new_value);

gboolean 
cuda_multiratespiir_parse_bank (gdouble *bank, guint *num_depths, gint *
		outchannels);

guint 
cuda_multiratespiir_get_outchannels (CudaMultirateSPIIR *element);

guint 
cuda_multiratespiir_get_num_head_cover_samples (CudaMultirateSPIIR *element);

guint64 
cuda_multiratespiir_get_available_samples (CudaMultirateSPIIR *element);

#endif

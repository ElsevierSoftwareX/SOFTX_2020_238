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

#ifndef __CUDA_MULTIRATESPIIR_KERNEL_H__
#define __CUDA_MULTIRATESPIIR_KERNEL_H__

#include <multiratespiir/multiratespiir.h>
#include <multiratespiir/multiratespiir_utils.h>
gint
multi_downsample (SpiirState **spstate, float *in_multidown, 
		gint num_in_multidown, guint num_depths, cudaStream_t stream);

gint
spiirup (SpiirState **spstate, gint num_in_multiup, guint num_depths, float *out, cudaStream_t stream);
#endif



/*
 * Copyright (C) 2012-2013 Qi Chu <chicsheep@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */


/*
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


#include <gstlal_iirbankCuda.h>
#define BANK_INIT_SUCCESS 1
#define BANK_INIT_FAILED -1
#define BANK_FREE_SUCCESS 1


#define NB_MAX 32

/* deprecated
#define RATE_1 4096
#define RATE_2 2048
#define RATE_3 1024
#define RATE_4 512
#define RATE_5 256
#define RATE_6 128
#define RATE_7 64
#define RATE_8 32
*/

/*
 * ============================================================================
 *
 * 				    Utilities
 *
 * ============================================================================
 */

/* deprecated
int get_stream_id(GSTLALIIRBankCuda *element);
*/

int cuda_bank_init(GSTLALIIRBankCuda *element);

int cuda_bank_free(GSTLALIIRBankCuda *element);

unsigned iir_channels(const GSTLALIIRBankCuda *element);

guint64 get_available_samples(GSTLALIIRBankCuda *element);

void set_metadata(GSTLALIIRBankCuda *element, GstBuffer *buf, guint64 outsamples, gboolean gap);


/*
 * transform double precision input samples to double precision output samples using GPU
 */

GstFlowReturn filter_d(GSTLALIIRBankCuda *element, GstBuffer *outbuf);

/*
 * transform single precision input samples to single precision output samples using GPU
 */


GstFlowReturn filter_s(GSTLALIIRBankCuda *element, GstBuffer *outbuf);


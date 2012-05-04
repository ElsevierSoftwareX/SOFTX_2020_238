/*
 * Copyright (C) 2010 Qi Chu <chicsheep@gmail.com>
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
#define BANK_FREE_SUCCESS 1

typedef struct _Complex8_F
{
	float re;
	float im;
} COMPLEX8_F;

typedef struct _Complex8_D
{
	double re;
	double im;
} COMPLEX8_D;

// all input/ output are in float fasion

typedef struct _iirBank
{
	COMPLEX8_F *a1_f;
	COMPLEX8_F *b0_f;
	COMPLEX8_F *y_f;

	int *d_i;
	float *input_f;
	COMPLEX8_F *output_f;

	unsigned int num_templates;
	unsigned int num_filters;
	int dmax, dmin;
	unsigned int rate;
	unsigned int pre_input_length;
	unsigned int pre_output_length;

} iirBank;

/*
 * ============================================================================
 *
 * 				    Utilities
 *
 * ============================================================================
 */

int bank_init(iirBank **pbank, GSTLALIIRBankCuda *element);
int bank_free(iirBank **pbank, GSTLALIIRBankCuda *element);

unsigned iir_channels(const GSTLALIIRBankCuda *element);

guint64 get_available_samples(GSTLALIIRBankCuda *element);

void set_metadata(GSTLALIIRBankCuda *element, GstBuffer *buf, guint64 outsamples, gboolean gap);


/*
 * transform input samples to output samples using IIR in Gpu
 */

GstFlowReturn filter(GSTLALIIRBankCuda *element, GstBuffer *outbuf);


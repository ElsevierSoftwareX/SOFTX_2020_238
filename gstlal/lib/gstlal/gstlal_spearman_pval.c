/*
 * Copyright (C) 2015  Linqing Wen <linqing.wen@ligo.org>
 *
 * Copyright (C) 2009--2011 Mireia Crispin Ortuzar <mcrispin@caltech.edu>,
 * Kipp Cannon <kipp.cannon@ligo.org>, Chad Hanna <chad.hanna@ligo.org>
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

/**
 * SECTION:gstlal_itac
 * @short_description:  Compute inspiral triggers
 *
 * Reviewed: 38c65535fc96d6cc3dee76c2de9d3b76b47d5283 2015-05-14 
 * K. Cannon, J. Creighton, C. Hanna, F. Robinett 
 *
 * Actions:
 * 
 * line 282: could be more efficient
 * lines 496, 501: assert the length retuned by gstlal_autocorrelation_chi2_float is exactly 1.
 *
 */

/*
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


/*
 * stuff from the C library
 */


#include <complex.h>
#include <math.h>


/*
 * stuff from glib/gobject
 */


#include <glib.h>


/*
 * stuff from GSL
 */


#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_statistics_float.h>
#include <gsl/gsl_sf.h>

/*
 * our own stuff
 */


#include <gstlal_spearman_pval.h>
#include <gstlal_autocorrelation_chi2.h>

/*
 * ============================================================================
 *
 *                                 Parameters
 *
 * ============================================================================
 */


#define CHI2_USES_REAL_ONLY FALSE


/*
 * ============================================================================
 *
 *                               Internal Code
 *
 * ============================================================================
 */

/*                                                                                                                                                         
 * return the number of autocorrelation vectors                                                                                                            
 */


static unsigned autocorrelation_channels(const gsl_matrix_complex *autocorrelation)
{
  return autocorrelation->size1;
}


/*                                                                                                                                                         
 * return the number of samples in the autocorrelation vectors                                                                                             
 */


static unsigned autocorrelation_length(const gsl_matrix_complex *autocorrelation)
{
  return autocorrelation->size2;
}

/*
 * Single precision version only
 */



unsigned gstlal_spearman_float(
	float *output,	/* pointer to start of output buffer */
	const float complex *input,	/* pointer to start of input buffer */
	unsigned input_length,	/* how many samples of the input to process */
	int latency,	/* latency offset */
	double snr_threshold,	/* only compute \chi^{2} values for input samples at or above this SNR (set to 0.0 to compute all \chi^{2} values) */
	const gsl_matrix_complex *autocorrelation_matrix,	/* autocorrelation function matrix.  autocorrelation vectors are rows */
	const gsl_matrix_int *autocorrelation_mask_matrix,	/* autocorrelation mask matrix or NULL to disable mask feature TEST WEN: Not used, kept for interface*/
	const gsl_vector *autocorrelation_norm	/* autocorrelation norms TEST WEN: Not used */
)
{
	unsigned channels = autocorrelation_channels(autocorrelation_matrix);
	unsigned output_length;
	float *output_end;

	float *data1;
	float *data2;
	double *work;

        unsigned i;
        unsigned data_len = autocorrelation_length(autocorrelation_matrix);



	/*
	 * safety checks
	 */
	g_assert(autocorrelation_matrix->tda == data_len);
	/*
	 * initialize
	 */

	/* the +1 is because when there is 1 correlation-length of data in
	 * the adapter then we can produce 1 output sample, not 0. */
	output_length = input_length - autocorrelation_length(autocorrelation_matrix) + 1;
	output_end = output + output_length * channels;

	/* temperory data vectors */
        data1 = (float *)malloc(data_len*sizeof(float));
        data2 = (float *)malloc(data_len*sizeof(float));
        work = (double *)malloc(2*data_len*sizeof(double));





	/*
	 * compute output samples.  note:  we assume that gsl_complex can
	 * be aliased to complex double.  I think it says somewhere in the
	 * documentation that this is true.
	 */

	while(output < output_end) {
		const complex double *autocorrelation = (const complex double *) gsl_matrix_complex_const_ptr(autocorrelation_matrix, 0, 0);
		unsigned channel;

		for(channel = 0; channel < channels; channel++) {
			/*
			 * start of input data block to be used for this
			 * output sample
			 */

			const float complex *indata = input;

			/*
			 * the input sample by which the autocorrelation
			 * funcion will be scaled
			 */
			/* DEBUG WEN: latency here is 1/2 * correlation length */
			complex double snr = input[((gint) data_len - 1 + latency) * channels];

			if(cabs(snr) >= snr_threshold) {
				/*
				 * end of this channel's row in the autocorrelation
				 * matrix
				 */
			  
			        double temp;
				double rs;
				double probrs;
			        const complex double *autocorrelation_end = autocorrelation + data_len;
				for(i=0; autocorrelation < autocorrelation_end; autocorrelation++, indata += channels, i++) {
				       complex double z;
				       z = (const complex double) *indata;
				       data1[i] = (float)  creal(z) ;
				       z = *autocorrelation *snr;  /* mainly get sign right */ 
				       /* DEBUG WEN: use correlation of real only */
				       data2[i] = (float) creal(z);
				}
				rs = gsl_stats_float_spearman(data1, 1, data2,1, data_len,work); 
				if (abs(rs) < 1 ){
				  /* calculate Student's t and its significance */
				  temp = rs *sqrt((data_len -2)/(1+rs)/(1-rs));
				  probrs = gsl_sf_beta_inc(0.5*(data_len-2), 0.5, (data_len-2)/(data_len-2+pow(temp,2)));
				  *output = (float) probrs;
				}
				else{
				  *output =0; 
				}

			} else {
			        autocorrelation += autocorrelation_length(autocorrelation_matrix);
				*output = 0;
			}

			/*
			 * advance to next sample
			 */

			output++;
			input++;
		}
	}

	/*
	 * done
	 */

	return output_length;
}


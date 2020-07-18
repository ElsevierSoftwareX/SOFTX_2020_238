/*
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
#include <string.h>


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


/*
 * our own stuff
 */


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
 * ============================================================================
 *
 *                                Exported API
 *
 * ============================================================================
 */


/*
 * return the number of autocorrelation vectors
 */


unsigned gstlal_autocorrelation_chi2_autocorrelation_channels(const gsl_matrix_complex *autocorrelation)
{
	return autocorrelation_channels(autocorrelation);
}


/*
 * return the number of samples in the autocorrelation vectors
 */


unsigned gstlal_autocorrelation_chi2_autocorrelation_length(const gsl_matrix_complex *autocorrelation)
{
	return autocorrelation_length(autocorrelation);
}


/*
 * compute autocorrelation norms --- the expectation value in noise.
 */


gsl_vector *gstlal_autocorrelation_chi2_compute_norms(const gsl_matrix_complex *autocorrelation_matrix, const gsl_matrix_int *autocorrelation_mask_matrix)
{
	gsl_vector *norm;
	unsigned channel;

	if(autocorrelation_mask_matrix && (autocorrelation_channels(autocorrelation_matrix) != autocorrelation_mask_matrix->size1 || autocorrelation_length(autocorrelation_matrix) != autocorrelation_mask_matrix->size2)) {
		/* FIXME:  report errors how? */
		/*GST_ELEMENT_ERROR(element, STREAM, FAILED, ("array size mismatch"), ("autocorrelation matrix (%dx%d) and mask matrix (%dx%d) do not have the same size", autocorrelation_channels(autocorrelation_matrix), autocorrelation_length(autocorrelation_matrix), autocorrelation_mask_matrix->size1, autocorrelation_mask_matrix->size2));*/
		return NULL;
	}

	norm = gsl_vector_alloc(autocorrelation_channels(autocorrelation_matrix));

	for(channel = 0; channel < autocorrelation_channels(autocorrelation_matrix); channel++) {
		gsl_vector_complex_const_view row = gsl_matrix_complex_const_row(autocorrelation_matrix, channel);
		gsl_vector_int_const_view mask = autocorrelation_mask_matrix ? gsl_matrix_int_const_row(autocorrelation_mask_matrix, channel) : (gsl_vector_int_const_view) {{0}};
		unsigned sample;
		double n = 0;
		
		for(sample = 0; sample < row.vector.size; sample++) {
			if(autocorrelation_mask_matrix && !gsl_vector_int_get(&mask.vector, sample))
				continue;
#if CHI2_USES_REAL_ONLY
			n += 1 - pow(GSL_REAL(gsl_vector_complex_get(&row.vector, sample)), 2);
#else
			n += 2 - gsl_complex_abs2(gsl_vector_complex_get(&row.vector, sample));
#endif
		}
		gsl_vector_set(norm, channel, n);
	}

	return norm;
}


/*
 * transform input samples to output samples using a time-domain algorithm
 */


unsigned gstlal_autocorrelation_chi2(
	double *output,	/* pointer to start of output buffer */
	const complex double *input,	/* pointer to start of input buffer */
	unsigned input_length,	/* how many samples of the input to process */
	int latency,	/* latency offset */
	double snr_threshold,	/* only compute \chi^{2} values for input samples at or above this SNR (set to 0.0 to compute all \chi^{2} values) */
	const gsl_matrix_complex *autocorrelation_matrix,	/* autocorrelation function matrix.  autocorrelation vectors are rows */
	const gsl_matrix_int *autocorrelation_mask_matrix,	/* autocorrelation mask matrix or NULL to disable mask feature */
	const gsl_vector *autocorrelation_norm	/* autocorrelation norms */
)
{
	unsigned channels = autocorrelation_channels(autocorrelation_matrix);
	unsigned output_length;
	double *output_end;

	/*
	 * safety checks
	 */

	g_assert(autocorrelation_matrix->tda == autocorrelation_length(autocorrelation_matrix));
	if(autocorrelation_mask_matrix) {
		g_assert(autocorrelation_channels(autocorrelation_matrix) == autocorrelation_mask_matrix->size1);
		g_assert(autocorrelation_length(autocorrelation_matrix) == autocorrelation_mask_matrix->size2);
		g_assert(autocorrelation_mask_matrix->tda == autocorrelation_length(autocorrelation_matrix));
	}

	/*
	 * initialize
	 */

	/* the +1 is because when there is 1 correlation-length of data in
	 * the adapter then we can produce 1 output sample, not 0. */
	output_length = input_length - autocorrelation_length(autocorrelation_matrix) + 1;
	output_end = output + output_length * channels;

	/*
	 * compute output samples.  note:  we assume that gsl_complex can
	 * be aliased to complex double.  I think it says somewhere in the
	 * documentation that this is true.
	 */

	while(output < output_end) {
		const complex double *autocorrelation = (const complex double *) gsl_matrix_complex_const_ptr(autocorrelation_matrix, 0, 0);
		const int *autocorrelation_mask = autocorrelation_mask_matrix ? (const int *) gsl_matrix_int_const_ptr(autocorrelation_mask_matrix, 0, 0) : NULL;
		unsigned channel;

		for(channel = 0; channel < channels; channel++) {
			/*
			 * start of input data block to be used for this
			 * output sample
			 */

			const complex double *indata = input;

			/*
			 * the input sample by which the autocorrelation
			 * funcion will be scaled
			 */

			complex double snr = input[((gint) autocorrelation_length(autocorrelation_matrix) - 1 + latency) * channels];

			if(cabs(snr) >= snr_threshold) {
#if CHI2_USES_REAL_ONLY
				/*
				 * multiplying snr by this makes it real
				 */

				complex double invsnrphase = cexp(-I*carg(snr));
#endif

				/*
				 * end of this channel's row in the autocorrelation
				 * matrix
				 */

				const complex double *autocorrelation_end = autocorrelation + autocorrelation_length(autocorrelation_matrix);

				/*
				 * \chi^{2} sum
				 */

				double chisq;

				/*
				 * compute \sum_{i} (A_{i} * \rho_{0} - \rho_{i})^{2}
				 */

				if(autocorrelation_mask) {
					for(chisq = 0; autocorrelation < autocorrelation_end; autocorrelation++, autocorrelation_mask++, indata += channels) {
						complex double z;
						if(!*autocorrelation_mask)
							continue;
						z = *autocorrelation * snr - *indata;
#if CHI2_USES_REAL_ONLY
						chisq += pow(creal(z * invsnrphase), 2);
#else
						chisq += pow(creal(z), 2) + pow(cimag(z), 2);
#endif
					}
				} else {
					for(chisq = 0; autocorrelation < autocorrelation_end; autocorrelation++, indata += channels) {
						complex double z = *autocorrelation * snr - *indata;
#if CHI2_USES_REAL_ONLY
						chisq += pow(creal(z * invsnrphase), 2);
#else
						chisq += pow(creal(z), 2) + pow(cimag(z), 2);
#endif
					}
				}

				/*
				 * record \chi^{2} sum, advance to next output sample
				 */

				*output = chisq / gsl_vector_get(autocorrelation_norm, channel);
			} else {
				autocorrelation += autocorrelation_length(autocorrelation_matrix);
				if(autocorrelation_mask)
					autocorrelation_mask += autocorrelation_length(autocorrelation_matrix);
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

/*
 * Single precision version
 */

unsigned gstlal_autocorrelation_chi2_float(
	float *output,	/* pointer to start of output buffer */
	const float complex *input,	/* pointer to start of input buffer */
	unsigned input_length,	/* how many samples of the input to process */
	int latency,	/* latency offset */
	double snr_threshold,	/* only compute \chi^{2} values for input samples at or above this SNR (set to 0.0 to compute all \chi^{2} values) */
	const gsl_matrix_complex *autocorrelation_matrix,	/* autocorrelation function matrix.  autocorrelation vectors are rows */
	const gsl_matrix_int *autocorrelation_mask_matrix,	/* autocorrelation mask matrix or NULL to disable mask feature */
	const gsl_vector *autocorrelation_norm	/* autocorrelation norms */
)
{
	unsigned channels = autocorrelation_channels(autocorrelation_matrix);
	unsigned output_length;
	float *output_end;

	/*
	 * safety checks
	 */

	g_assert(autocorrelation_matrix->tda == autocorrelation_length(autocorrelation_matrix));
	if(autocorrelation_mask_matrix) {
		g_assert(autocorrelation_channels(autocorrelation_matrix) == autocorrelation_mask_matrix->size1);
		g_assert(autocorrelation_length(autocorrelation_matrix) == autocorrelation_mask_matrix->size2);
		g_assert(autocorrelation_mask_matrix->tda == autocorrelation_length(autocorrelation_matrix));
	}

	/*
	 * initialize
	 */

	/* the +1 is because when there is 1 correlation-length of data in
	 * the adapter then we can produce 1 output sample, not 0. */
	output_length = input_length - autocorrelation_length(autocorrelation_matrix) + 1;
	output_end = output + output_length * channels;

	/*
	 * compute output samples.  note:  we assume that gsl_complex can
	 * be aliased to complex double.  I think it says somewhere in the
	 * documentation that this is true.
	 */

	while(output < output_end) {
		const complex double *autocorrelation = (const complex double *) gsl_matrix_complex_const_ptr(autocorrelation_matrix, 0, 0);
		const int *autocorrelation_mask = autocorrelation_mask_matrix ? (const int *) gsl_matrix_int_const_ptr(autocorrelation_mask_matrix, 0, 0) : NULL;
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

			complex double snr = input[((gint) autocorrelation_length(autocorrelation_matrix) - 1 + latency) * channels];

			if(cabs(snr) >= snr_threshold) {
#if CHI2_USES_REAL_ONLY
				/*
				 * multiplying snr by this makes it real
				 */

				complex double invsnrphase = complex cexp(-I*carg(snr));
#endif

				/*
				 * end of this channel's row in the autocorrelation
				 * matrix
				 */

				const complex double *autocorrelation_end = autocorrelation + autocorrelation_length(autocorrelation_matrix);

				/*
				 * \chi^{2} sum
				 */

				double chisq;

				/*
				 * compute \sum_{i} (A_{i} * \rho_{0} - \rho_{i})^{2}
				 */

				if(autocorrelation_mask) {
					for(chisq = 0; autocorrelation < autocorrelation_end; autocorrelation++, autocorrelation_mask++, indata += channels) {
						complex double z;
						if(!*autocorrelation_mask)
							continue;
						z = *autocorrelation * snr - (const complex double) *indata;
#if CHI2_USES_REAL_ONLY
						chisq += pow(creal(z * invsnrphase), 2);
#else
						chisq += pow(creal(z), 2) + pow(cimag(z), 2);
#endif
					}
				} else {
					for(chisq = 0; autocorrelation < autocorrelation_end; autocorrelation++, indata += channels) {
						complex double z = *autocorrelation * snr - (const complex double) *indata;
#if CHI2_USES_REAL_ONLY
						chisq += pow(creal(z * invsnrphase), 2);
#else
						chisq += pow(creal(z), 2) + pow(cimag(z), 2);
#endif
					}
				}

				/*
				 * record \chi^{2} sum, advance to next output sample
				 */

				*output = (float) chisq / gsl_vector_get(autocorrelation_norm, channel);
			} else {
				autocorrelation += autocorrelation_length(autocorrelation_matrix);
				if(autocorrelation_mask)
					autocorrelation_mask += autocorrelation_length(autocorrelation_matrix);
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

gsl_vector *gstlal_bankcorrelation_chi2_compute_norms(const gsl_matrix_complex *bankcorrelation_matrix)
{
	gsl_vector *norm;
	unsigned i;
	unsigned channel;
	unsigned channels = bankcorrelation_matrix->size1;
	float complex cij;
	float norms;

	norm = gsl_vector_alloc(channels);

	for(channel = 0; channel < channels; channel++) {
		norms = 2*channels;
		for(i = 0; i < channels; i++) {
			cij = ((float) GSL_REAL(gsl_matrix_complex_get(bankcorrelation_matrix, i, channel)) + (float) GSL_IMAG(gsl_matrix_complex_get(bankcorrelation_matrix, i, channel)) * I);
			norms -= 0.5*conjf(cij)*cij;
		}
		gsl_vector_set(norm, channel, norms);
	}

	return norm;
}
//FIXME make this like the float version
unsigned gstlal_bankcorrelation_chi2_from_peak(double *out, struct gstlal_peak_state *state, const gsl_matrix_complex *bcmat, const gsl_vector *bcnorm, const complex double *data, guint pad)
{
	// FIXME add something
	return 0;
}

unsigned gstlal_bankcorrelation_chi2_from_peak_float(float *out, struct gstlal_peak_state *state, const gsl_matrix_complex *bcmat, const gsl_vector *bcnorm, const complex float *data, guint pad)
{
	unsigned i,j;
	float complex *snr = state->values.as_float_complex;
	float complex cij;
	float complex xij;
	float complex snrj;
	int index;

	for (i = 0; i < state->channels; i++)
	{
		out[i] = 0.0;
		/*
		 * Don't bother computing if the event is below threshold, i.e.
		 * set to 0
		 */
		if (snr[i] == 0) continue;
		for (j = 0; j < state->channels; j++)
		{
			index = (state->samples[i]) * state->channels + j;
			snrj = *(data + index);
			// Normalization of cij is in svd_bank.py. Phase must be (+).
			cij = ((float) GSL_REAL(gsl_matrix_complex_get(bcmat, i, j)) + (float) GSL_IMAG(gsl_matrix_complex_get(bcmat, i, j)) * I);
			// 0.5 is necessary due to the template normalization gstlal uses.
			xij = 0.5 * snr[i] * cij - snrj;
			out[i] += conjf(xij) * xij;
		}
		out[i] /= (float) gsl_vector_get(bcnorm, i);
	}
	return 0;
}

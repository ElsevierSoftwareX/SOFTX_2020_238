/*
 * Copyright (C) 2020  Aaron Viets
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation; either version 2 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */


#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include <complex.h>
#include <glib.h>
#include <time.h>


#include <gstlal_firtools.h>


/*
 * Below are functions to compute FFTs and inverse FFTs, including the special cases
 * of purely real input, at long double precision.  The actual precision depends on
 * your machine's platform, but it is often 80 bits (double has 64 bits).
 */


/* Typecast an array */
#define TYPECAST_TYPEIN_TO_TYPEOUT(LONGIN, DTYPEIN, LONGOUT, DTYPEOUT, COMPLEX) \
LONGOUT COMPLEX DTYPEOUT *typecast_ ## LONGIN ## COMPLEX ## DTYPEIN ## _to_ ## LONGOUT ## COMPLEX ## DTYPEOUT(LONGIN COMPLEX DTYPEIN *data, guint N) { \
 \
	COMPLEX LONGOUT DTYPEOUT *outdata = g_malloc(N * sizeof(LONGOUT COMPLEX DTYPEOUT)); \
	guint i; \
	for(i = 0; i < N; i++) \
		outdata[i] = (LONGOUT COMPLEX DTYPEOUT) data[i]; \
 \
	g_free(data); \
 \
	return outdata; \
}


TYPECAST_TYPEIN_TO_TYPEOUT(, float, , float, );
TYPECAST_TYPEIN_TO_TYPEOUT(, float, , double, );
TYPECAST_TYPEIN_TO_TYPEOUT(, float, long, double, );
TYPECAST_TYPEIN_TO_TYPEOUT(, double, , double, );
TYPECAST_TYPEIN_TO_TYPEOUT(, double, , float, );
TYPECAST_TYPEIN_TO_TYPEOUT(, double, long, double, );
TYPECAST_TYPEIN_TO_TYPEOUT(long, double, long, double, );
TYPECAST_TYPEIN_TO_TYPEOUT(long, double, , float, );
TYPECAST_TYPEIN_TO_TYPEOUT(long, double, , double, );
TYPECAST_TYPEIN_TO_TYPEOUT(, float, , float, complex);
TYPECAST_TYPEIN_TO_TYPEOUT(, float, , double, complex);
TYPECAST_TYPEIN_TO_TYPEOUT(, float, long, double, complex);
TYPECAST_TYPEIN_TO_TYPEOUT(, double, , double, complex);
TYPECAST_TYPEIN_TO_TYPEOUT(, double, , float, complex);
TYPECAST_TYPEIN_TO_TYPEOUT(, double, long, double, complex);
TYPECAST_TYPEIN_TO_TYPEOUT(long, double, long, double, complex);
TYPECAST_TYPEIN_TO_TYPEOUT(long, double, , float, complex);
TYPECAST_TYPEIN_TO_TYPEOUT(long, double, , double, complex);


#define SUM_ARRAY(LONG, COMPLEX, DTYPE) \
LONG COMPLEX DTYPE sum_array_ ## LONG ## COMPLEX ## DTYPE(LONG COMPLEX DTYPE *array, guint N, guint cadence) { \
 \
	LONG COMPLEX DTYPE sum = 0.0; \
	LONG COMPLEX DTYPE *ptr, *end = array + N; \
	for(ptr = array; ptr < end; ptr += cadence) \
		sum += *ptr; \
 \
	return sum; \
}


SUM_ARRAY(, , float);
SUM_ARRAY(, , double);
SUM_ARRAY(long, , double);
SUM_ARRAY(, complex, float);
SUM_ARRAY(, complex, double);
SUM_ARRAY(long, complex, double);


#define LONG_SUM_ARRAY(LONG, COMPLEX, DTYPE) \
long COMPLEX double long_sum_array_ ## LONG ## COMPLEX ## DTYPE(LONG COMPLEX DTYPE *array, guint N, guint cadence) { \
 \
	long COMPLEX double sum = 0.0; \
	LONG COMPLEX DTYPE *ptr, *end = array + N; \
	for(ptr = array; ptr < end; ptr += cadence) \
		sum += *ptr; \
 \
	return sum; \
}


LONG_SUM_ARRAY(, , float);
LONG_SUM_ARRAY(, complex, float);
LONG_SUM_ARRAY(, , double);
LONG_SUM_ARRAY(, complex, double);
LONG_SUM_ARRAY(long, , double);
LONG_SUM_ARRAY(long, complex, double);


#define ARRAY_SUBSET(LONG, COMPLEX, DTYPE) \
long COMPLEX double *array_subset_ ## LONG ## COMPLEX ## DTYPE(LONG COMPLEX DTYPE *array, guint N, guint cadence) { \
 \
	guint N_sub = (N + cadence - 1) / cadence; \
	long COMPLEX double *subset = g_malloc(N_sub * sizeof(long COMPLEX double)); \
	long COMPLEX double *ptr, *end = subset + N_sub; \
	for(ptr = subset; ptr < end; ptr++, array += cadence) \
		*ptr = *array; \
 \
	return subset; \
}


ARRAY_SUBSET(, , float);
ARRAY_SUBSET(, complex, float);
ARRAY_SUBSET(, , double);
ARRAY_SUBSET(, complex, double);
ARRAY_SUBSET(long, , double);
ARRAY_SUBSET(long, complex, double);


#define ARRAY_SUBSET_MOD_N(LONG, COMPLEX, DTYPE) \
long COMPLEX double *array_subset_mod_n_ ## LONG ## COMPLEX ## DTYPE(LONG COMPLEX DTYPE *array, guint N, guint n) { \
 \
	guint mod_max = n / 2; \
	guint N_sub = N / n * (mod_max + 1) + (N % n < mod_max + 1 ? N % n : mod_max + 1); \
	long COMPLEX double *subset = g_malloc(N_sub * sizeof(long COMPLEX double)); \
	long COMPLEX double *ptr, *end = subset + N_sub; \
	guint i = 0; \
	for(ptr = subset; ptr < end; ptr++, i++, i += (i % n <= mod_max ? 0 : mod_max)) \
		*ptr = array[i]; \
 \
	return subset; \
}


ARRAY_SUBSET_MOD_N(, , float);
ARRAY_SUBSET_MOD_N(, complex, float);
ARRAY_SUBSET_MOD_N(, , double);
ARRAY_SUBSET_MOD_N(, complex, double);
ARRAY_SUBSET_MOD_N(long, , double);
ARRAY_SUBSET_MOD_N(long, complex, double);


#define ARRAY_SUBSET_CONJ(LONG, DTYPE, LLL, FF) \
long complex double *array_subset_conj_ ## LONG ## DTYPE(LONG complex DTYPE *array, guint start, guint N_in, guint N_total, guint cadence) { \
 \
	guint i, j; \
	guint N_sub = (N_total - start + cadence - 1) / cadence; \
	long complex double *subset = g_malloc(N_sub * sizeof(long complex double)); \
	long complex double *ptr = subset; \
	for(i = start; i < N_in; i += cadence, ptr++) \
		*ptr = array[i]; \
 \
	for(j = 1 + i - N_in + (N_total + 1) % 2; j < N_in; j += cadence, ptr++) \
		*ptr = conj ## LLL ## FF(array[N_in - j]); \
 \
	return subset; \
}


ARRAY_SUBSET_CONJ(, float, , f);
ARRAY_SUBSET_CONJ(, double, , );
ARRAY_SUBSET_CONJ(long, double, l, );


#define CONJ_ARRAY(LONG, DTYPE, LLL, FF) \
LONG complex DTYPE *conj_array_ ## LONG ## DTYPE(LONG complex DTYPE *array, guint N) { \
 \
	guint i; \
	LONG complex DTYPE *conjugate = g_malloc(N * sizeof(LONG complex DTYPE)); \
	for(i = 0; i < N; i++) \
		conjugate[i] = conj ## LLL ## FF(array[i]); \
 \
	return conjugate; \
}


CONJ_ARRAY(, float, , f);
CONJ_ARRAY(, double, , );
CONJ_ARRAY(long, double, l, );


#define PAD_ZEROS_A(LONG, COMPLEX, DTYPE, LLL, FF) \
long complex double *pad_zeros_A ## LONG ## COMPLEX ## DTYPE(LONG COMPLEX DTYPE *data, guint N, guint N_conj, long complex double *exp_array, guint M) { \
 \
	guint i; \
	long complex double *A_n = g_malloc0(M * sizeof(long complex double)); \
	for(i = 0; i < N; i++) \
		A_n[i] = data[i] * exp_array[i]; \
	guint N_tot = N + N_conj; \
	for(i = N; i < N_tot; i++) \
		A_n[i] = conj ## LLL(data[N_tot - i]) * exp_array[i]; \
 \
	return A_n; \
}


PAD_ZEROS_A(, , float, , f);
PAD_ZEROS_A(, , double, , );
PAD_ZEROS_A(long, , double, l, );
PAD_ZEROS_A(, complex, float, , f);
PAD_ZEROS_A(, complex, double, , );
PAD_ZEROS_A(long, complex, double, l, );


#define PAD_ZEROS_B(LONG, LLL) \
LONG complex double *pad_zeros_B ## LONG(LONG complex double *b_n, guint N, guint N_out, guint M) { \
 \
	guint i; \
	LONG complex double *B_n = g_malloc0(M * sizeof(LONG complex double)); \
	for(i = 0; i < N_out; i++) \
		B_n[i] = b_n[i]; \
	for(i = 1; i < N ; i++) \
		B_n[M - i] = b_n[i]; \
 \
	return B_n; \
}


PAD_ZEROS_B(, );
PAD_ZEROS_B(long, l);


/* A function to find prime factors of N, the size of the input data */
guint *find_prime_factors(guint N, guint *num_factors) {

	/* Also find the number of prime factors, including 1. */
	*num_factors = 0;

	/* Allocate as much memory as we could possibly need */
	guint *prime_factors = g_malloc(((guint) log2((double) N) + 1) * sizeof(guint));
	guint *ptr = prime_factors;
	guint factor = 2;
	while(factor <= N) {
		if(N % factor)
			factor += 1;
		else {
			*ptr = factor;
			ptr++;
			(*num_factors)++;
			N /= factor;
		}
	}
	*ptr = 1;
	(*num_factors)++;

	return prime_factors;
}


/*
 * For Bluestein's algorithm.  Find a good padded length.  Also, for efficiency,
 * compute new prime factors
 */
guint *find_prime_factors_M(guint M_min, guint *M, guint *num_factors) {

	*M = (guint) pow(2.0, 1.0 + (guint) log2(M_min - 1.0));
	*num_factors = (guint) log2((double) *M) + 1;

	guint i, *prime_factors;

	if(9 * *M >= 16 * M_min && *M >= 16) {
		*num_factors -= 2;
		*M = *M * 9 / 16;
		prime_factors = g_malloc(*num_factors * sizeof(guint));
		prime_factors[*num_factors - 3] = prime_factors[*num_factors - 2] = 3;
		prime_factors[*num_factors - 1] = 1;
		for(i = 0; i < *num_factors - 3; i++)
			prime_factors[i] = 2;
		return prime_factors;
	} else if(5 * *M >= 8 * M_min && *M >= 8) {
		*num_factors -= 2;
		*M = *M * 5 / 8;
		prime_factors = g_malloc(*num_factors * sizeof(guint));
		prime_factors[*num_factors - 2] = 5;
		prime_factors[*num_factors - 1] = 1;
		for(i = 0; i < *num_factors - 2; i++)
			prime_factors[i] = 2;
		return prime_factors;
	} else if(3 * *M >= 4 * M_min && *M >= 4) {
		*num_factors -= 1;
		*M = *M * 3 / 4;
		prime_factors = g_malloc(*num_factors * sizeof(guint));
		prime_factors[*num_factors - 2] = 3;
		prime_factors[*num_factors - 1] = 1;
		for(i = 0; i < *num_factors - 2; i++)
			prime_factors[i] = 2;
		return prime_factors;
	} else if(7 * *M >= 8 * M_min && *M >= 8) {
		*num_factors -= 2;
		*M = *M * 7 / 8;
		prime_factors = g_malloc(*num_factors * sizeof(guint));
		prime_factors[*num_factors - 2] = 7;
		prime_factors[*num_factors - 1] = 1;
		for(i = 0; i < *num_factors - 2; i++)
			prime_factors[i] = 2;
		return prime_factors;
	} else if(15 * *M >= 16 * M_min && *M >= 16) {
		*num_factors -= 2;
		*M = *M * 15 / 16;
		prime_factors = g_malloc(*num_factors * sizeof(guint));
		prime_factors[*num_factors - 3] = 3;
		prime_factors[*num_factors - 2] = 5;
		prime_factors[*num_factors - 1] = 1;
		for(i = 0; i < *num_factors - 3; i++)
			prime_factors[i] = 2;
		return prime_factors;
	} else {
		prime_factors = g_malloc(*num_factors * sizeof(guint));
		prime_factors[*num_factors - 1] = 1;
		for(i = 0; i < *num_factors - 1; i++)
			prime_factors[i] = 2;
		return prime_factors;
	}
}


/* A function to compute the array of exponentials */
long complex double *find_exp_array(guint N, gboolean inverse) {

	long complex double *exp_array = g_malloc(N * sizeof(long complex double));

	/* If this is the inverse DFT, just don't negate 2*pi */
	long complex double prefactor;
	if(inverse)
		prefactor = 2 * PI * I;
	else
		prefactor = -2 * PI * I;

	guint n;
	if(!(N % 4)) {
		/* It's a multiple of 4, so we know these values right away: */
		exp_array[0] = 1.0;
		exp_array[N / 2] = -1.0;
		if(inverse) {
			exp_array[N / 4] = I;
			exp_array[3 * N / 4] = -I;
		} else {
			exp_array[N / 4] = -I;
			exp_array[3 * N / 4] = I;
		}
		/* Only compute one fourth of the array, and use symmetry for the rest. */
		for(n = 1; n < N / 4; n++) {
			exp_array[n] = cexpl(prefactor * n / N);
			exp_array[N / 2 - n] = -conjl(exp_array[n]);
			exp_array[N / 2 + n] = -exp_array[n];
			exp_array[N - n] = conjl(exp_array[n]);
		}
	} else if(!(N % 2)) {
		/* It's a multiple of 2, so we know these values right away: */
		exp_array[0] = 1.0;
		exp_array[N / 2] = -1.0;

		/* Only compute one fourth of the array, and use symmetry for the rest. */
		for(n = 1; n <= N / 4; n++) {
			exp_array[n] = cexpl(prefactor * n / N);
			exp_array[N / 2 - n] = -conjl(exp_array[n]);
			exp_array[N / 2 + n] = -exp_array[n];
			exp_array[N - n] = conjl(exp_array[n]);
		}
	} else {
		/* It's odd, but we still know this: */
		exp_array[0] = 1.0;

		/* Only compute half of the array, and use symmetry for the rest. */
		for(n = 1; n <= N / 2; n++) {
			exp_array[n] = cexpl(prefactor * n / N);
			exp_array[N - n] = conjl(exp_array[n]);
		}
	}
	return exp_array;
}


/* A function to compute the array of exponentials for Bluestein's algorithm */
long complex double *find_exp_array2(guint N, gboolean inverse) {

	/* First compute the usual fft array */
	long complex double *exp_array = find_exp_array(2 * N, inverse);

	/* Rearrange it */
	long complex double *exp_array2 = g_malloc(N * sizeof(long complex double));
	guint i;
	for(i = 0; i < N; i++)
		exp_array2[i] = exp_array[(guint) pow(i, 2) % (2 * N)];
	return exp_array2;
}


/* A discrete Fourier transform, evaluated according to the definition */
#define GSTLAL_DFT(LONG, DTYPE) \
LONG complex DTYPE *gstlal_dft_ ## LONG ## DTYPE(LONG complex DTYPE *td_data, guint N, long complex double *exp_array, gboolean inverse, long complex double *fd_data) { \
 \
	gboolean exp_array_need_freed = FALSE; \
	if(exp_array == NULL) { \
		/*
		 * Make array of exp(-2 pi i f t) to multiply.  This is expensive, so only make
		 * the code do it once.
		 */ \
		exp_array = find_exp_array(N, inverse); \
		exp_array_need_freed = TRUE; \
	} \
	if(fd_data == NULL) \
		fd_data = g_malloc0(N * sizeof(long complex double)); \
 \
	/* The first term is the DC component, which is just the sum. */ \
	fd_data[0] = long_sum_array_ ## LONG ## complex ## DTYPE(td_data, N, 1); \
 \
	/*
	 * Since this function is most often called by gstlal_fft_(), N is most likely a prime, so assume
	 * there are no more trivial multiplications
	 */ \
	if(N == 2) { \
		fd_data[1] += td_data[0]; \
		fd_data[1] -= td_data[1]; \
	} else { \
		guint i, j; \
		for(i = 1; i < N; i++) { \
			fd_data[i] += td_data[0]; \
			for(j = 1; j < N; j++) \
				fd_data[i] += td_data[j] * exp_array[i * j % N]; \
		} \
	} \
	g_free(td_data); \
	if(exp_array_need_freed) \
		g_free(exp_array); \
 \
	if(sizeof(LONG DTYPE) < sizeof(long double)) \
		return typecast_longcomplexdouble_to_ ## LONG ## complex ## DTYPE(fd_data, N); \
	else \
		return (LONG complex DTYPE *) fd_data; \
}


GSTLAL_DFT(, float);
GSTLAL_DFT(, double);
GSTLAL_DFT(long, double);


/* A discrete inverse Fourier transform, evaluated according to the definition */
#define GSTLAL_IDFT(LONG, DTYPE) \
LONG complex DTYPE *gstlal_idft_ ## LONG ## DTYPE(LONG complex DTYPE *fd_data, guint N, gboolean normalize) { \
 \
	LONG complex DTYPE *td_data = gstlal_dft_ ## LONG ## DTYPE(fd_data, N, NULL, TRUE, NULL); \
 \
	if(normalize) { \
		guint i; \
		for(i = 0; i < N; i++) \
			td_data[i] /= N; \
	} \
 \
	return td_data; \
}


GSTLAL_IDFT(, float);
GSTLAL_IDFT(, double);
GSTLAL_IDFT(long, double);


/*
 * If the input is real, the output is conjugate-symmetric: fd_data[n] = conj(fd_data[N - n]).
 * We can reduce the number of operations by a factor of ~2.  Also, we have the option to only
 * output half of the result, since the second half is redundant.
 */
#define GSTLAL_RDFT(LONG, DTYPE) \
LONG complex DTYPE *gstlal_rdft_ ## LONG ## DTYPE(LONG DTYPE *td_data, guint N, long complex double *exp_array, gboolean return_full, long double complex *fd_data) { \
 \
	guint N_out = N / 2 + 1; \
 \
	gboolean exp_array_need_freed = FALSE; \
	if(exp_array == NULL) { \
		/*
		 * Make array of exp(-2 pi i f t) to multiply.  This is expensive, so only make
		 * the code do it once.
		 */ \
		exp_array = find_exp_array(N, FALSE); \
		exp_array_need_freed = TRUE; \
	} \
	if(fd_data == NULL) \
		fd_data = g_malloc0((return_full ? N : N_out) * sizeof(long double complex)); \
 \
	/* The first term is the DC component, which is just the sum. */ \
	fd_data[0] = long_sum_array_ ## LONG ## DTYPE(td_data, N, 1); \
 \
	/*
	 * Since this function is most often called by gstlal_fft_(), N is most likely a prime, so assume
	 * there are no more trivial multiplications
	 */ \
	if(N == 2) { \
		fd_data[1] += td_data[0]; \
		fd_data[1] -= td_data[1]; \
	} else { \
		guint i, j; \
		for(i = 1; i < N_out; i++) { \
			fd_data[i] += td_data[0]; \
			for(j = 1; j < N; j++) \
				fd_data[i] += td_data[j] * exp_array[i * j % N]; \
		} \
	} \
	g_free(td_data); \
	if(exp_array_need_freed) \
		g_free(exp_array); \
 \
	if(return_full && N > 2) { \
		/* Then fill in the second half */ \
		guint i; \
		for(i = 1; i <= N - N_out; i++) \
			fd_data[N - i] = conjl(fd_data[i]); \
	} \
	if(sizeof(LONG DTYPE) < sizeof(long double)) \
		return typecast_longcomplexdouble_to_ ## LONG ## complex ## DTYPE(fd_data, return_full ? N : N_out); \
	else \
		return (LONG complex DTYPE *) fd_data; \
}


GSTLAL_RDFT(, float);
GSTLAL_RDFT(, double);
GSTLAL_RDFT(long, double);


/*
 * Inverse of the above real-input DFT.  So the output of this is real and the input is assumed
 * to be shortened to N / 2 + 1 samples to avoid redundancy.
 */
#define GSTLAL_IRDFT(LONG, DTYPE, LLL, FF) \
LONG DTYPE *gstlal_irdft_ ## LONG ## DTYPE(LONG complex DTYPE *fd_data, guint N_in, guint *N, long complex double *exp_array, gboolean normalize, long double *td_data) { \
 \
	if(!N) { \
		N = g_malloc(sizeof(guint)); \
		*N = 0; \
	} \
 \
	if(*N == 0) { \
		/*
		 * Find N, the original number of samples. If the imaginary part of the last
		 * sample is zero, assume N was even.
		 */ \
		if(!cimag ## LLL ## FF(fd_data[N_in - 1])) \
			*N = (N_in - 1) * 2; \
		else if(!creal ## LLL ## FF(fd_data[N_in - 1])) \
			*N = N_in * 2 - 1; \
		else if(fabs ## LLL ## FF(cimag ## LLL ## FF(fd_data[N_in - 1]) / creal ## LLL ## FF(fd_data[N_in - 1])) < 1e-10) \
			*N = (N_in - 1) * 2; \
		else \
			*N = N_in * 2 - 1; \
	} \
	gboolean exp_array_need_freed = FALSE; \
	if(exp_array == NULL) { \
		/*
		 * Make array of exp(-2 pi i f t) to multiply.  This is expensive, so only make
		 * the code do it once.
		 */ \
		exp_array = find_exp_array(*N, TRUE); \
		exp_array_need_freed = TRUE; \
	} \
	if(td_data == NULL) \
		td_data = g_malloc(*N * sizeof(long double)); \
 \
	/* The first term is the DC component, which is just the sum. */ \
	td_data[0] = creall(long_sum_array_ ## LONG ## complex ## DTYPE(fd_data, N_in, 1)) + creall(long_sum_array_ ## LONG ## complex ## DTYPE(fd_data + 1, *N - N_in, 1)); \
 \
	/*
	 * Since this function is most often called by gstlal_irfft_(), N is most likely a prime, so assume
	 * there are no more trivial multiplications
	 */ \
	if(*N == 2) { \
		td_data[1] += creal ## LLL ## FF(fd_data[0]); \
		td_data[1] -= creal ## LLL ## FF(fd_data[1]); \
 \
	} else { \
		guint i, j; \
		for(i = 1; i < *N; i++) { \
			td_data[i] += creal ## LLL ## FF(fd_data[0]); \
			for(j = 1; j < N_in; j++) \
				td_data[i] += creall(fd_data[j] * exp_array[i * j % *N]); \
			for(j = N_in; j < *N; j++) \
				td_data[i] += creall(conj ## LLL ## FF(fd_data[*N - j]) * exp_array[i * j % *N]); \
		} \
	} \
 \
	g_free(fd_data); \
	if(exp_array_need_freed) \
		g_free(exp_array); \
 \
	if(normalize) { \
		guint i; \
		for(i = 0; i < *N; i++) \
			td_data[i] /= *N; \
	} \
 \
	if(sizeof(LONG DTYPE) < sizeof(long double)) \
		return typecast_longdouble_to_ ## LONG ## DTYPE(td_data, *N); \
	else \
		return (LONG DTYPE *) td_data; \
}


GSTLAL_IRDFT(, float, , f);
GSTLAL_IRDFT(, double, , );
GSTLAL_IRDFT(long, double, l, );


/*
 * A fast Fourier transform using the Cooley-Tukey algorithm, which
 * factors the length N to break up the transform into smaller transforms
 */
#define GSTLAL_FFT(LONG, DTYPE) \
LONG complex DTYPE *gstlal_fft_ ## LONG ## DTYPE(LONG complex DTYPE *td_data, guint N, guint *prime_factors, guint num_factors, long complex double *exp_array, gboolean inverse, guint M, guint *M_prime_factors, guint M_num_factors, long complex double *M_exp_array2, long complex double *M_exp_array, long complex double *fd_data, gboolean free_input) { \
 \
	if(N < 2) \
		return td_data; \
 \
	gboolean prime_factors_need_freed = FALSE; \
	gboolean M_prime_factors_need_freed = FALSE; \
	gboolean exp_array_need_freed = FALSE; \
 \
	if(prime_factors == NULL) { \
		/* Find prime factors */ \
		prime_factors = find_prime_factors(N, &num_factors); \
		prime_factors_need_freed = TRUE; \
 \
		/* Check if we will need to use gstlal_prime_fft_() for this */ \
		if(prime_factors[num_factors - 2] >= 107) { \
			/* Find the first member greater than or equal to 107 */ \
			guint i = 0; \
			while(prime_factors[i] < 107) \
				i += 1; \
 \
			/* Compute a good padded length for Bluestein's algorithm */ \
			guint j, M_min = 2; \
			for(j = i; j < num_factors; j++) \
				M_min *= prime_factors[j]; \
			M_min--; \
			M_prime_factors = find_prime_factors_M(M_min, &M, &M_num_factors); \
 \
			/* Find the array of exponentials for Bluestein's algorithm */ \
			M_exp_array2 = find_exp_array2((M_min + 1) / 2, inverse); \
			M_exp_array = find_exp_array(M, FALSE); \
 \
			M_prime_factors_need_freed = TRUE; \
		} \
	} \
	if(exp_array == NULL && prime_factors[0] < 107) { \
		/*
		 * Make array of exp(-2 pi i f t) to multiply.  This is expensive, so only make
		 * the code do it once.
		 */ \
		exp_array = find_exp_array(N, inverse); \
		exp_array_need_freed = TRUE; \
	} \
 \
	if(fd_data == NULL) \
		fd_data = g_malloc0(N * sizeof(long complex double)); \
 \
	if(prime_factors[0] >= 107) \
		/* Use Bluestein's algorithm for a prime-length fft */ \
		gstlal_prime_fft_ ## LONG ## DTYPE(td_data, N, inverse, M_exp_array2, M, M_prime_factors, M_num_factors, M_exp_array, fd_data); \
 \
	else if(prime_factors[0] == N) \
		/* Do an ordinary DFT */ \
		gstlal_dft_ ## LONG ## DTYPE(td_data, N, exp_array, inverse, fd_data); \
 \
	else { \
		/* We will break this up into smaller Fourier transforms */ \
		guint i, num_ffts = prime_factors[0]; \
		guint N_mini = N / num_ffts; \
		long complex double *exp_array_subset = array_subset_longcomplexdouble(exp_array, N, num_ffts); \
		for(i = 0; i < num_ffts; i++) \
			gstlal_fft_longdouble(array_subset_ ## LONG ## complex ## DTYPE(td_data + i, N, num_ffts), N_mini, prime_factors + 1, num_factors - 1, exp_array_subset, inverse, M, M_prime_factors, M_num_factors, M_exp_array2, M_exp_array, fd_data + i * N_mini, TRUE); \
		if(free_input) \
			g_free(td_data); \
		g_free(exp_array_subset); \
 \
		/* Now we need to "mix" the output appropriately.  First, copy all but the first fft. */ \
		long complex double *fd_data_copy = array_subset_longcomplexdouble(fd_data + N_mini, N - N_mini, 1); \
 \
		/* Apply phase rotations to all but the first fft */ \
		guint exp_index; \
		for(i = N_mini; i < N; i++) { \
			exp_index = (i * (i / N_mini)) % N; \
			/* Do a multiplication only if we have to */ \
			if(exp_index) \
				fd_data[i] *= exp_array[exp_index]; \
		} \
		/* Add the first fft to all the others */ \
		for(i = N_mini; i < N; i++) \
			fd_data[i] += fd_data[i % N_mini]; \
 \
		/* Now we have to use the copied data.  Apply phase rotations and add to all other locations. */ \
		guint copy_index, j; \
		for(i = N_mini; i < N; i++) { \
			copy_index = i - N_mini; \
			/* Note that we skip j == i below, since we took care of that contribution 2 for loops ago */ \
			for(j = i % N_mini; j < N; j += N_mini, j += (j == i ? N_mini : 0)) { \
				exp_index = (j * (i / N_mini)) % N; \
				/* Do a multiplication only if we have to */ \
				if(exp_index) { \
					fd_data[j] += fd_data_copy[copy_index] * exp_array[exp_index]; \
				} else { \
					fd_data[j] += fd_data_copy[copy_index]; \
				} \
			} \
		} \
	} \
 \
	/* Done */ \
	if(exp_array_need_freed) \
		g_free(exp_array); \
	if(prime_factors_need_freed) \
		g_free(prime_factors); \
	if(M_prime_factors_need_freed) { \
		g_free(M_prime_factors); \
		g_free(M_exp_array); \
		g_free(M_exp_array2); \
	} \
	if(sizeof(LONG DTYPE) < sizeof(long double)) \
		return typecast_longcomplexdouble_to_ ## LONG ## complex ## DTYPE(fd_data, N); \
	else \
		return (LONG complex DTYPE *) fd_data; \
}


GSTLAL_FFT(, float);
GSTLAL_FFT(, double);
GSTLAL_FFT(long, double);


/*
 * An inverse fast Fourier transform that factors the length N to break up the
 * transform into smaller transforms
 */
#define GSTLAL_IFFT(LONG, DTYPE) \
LONG complex DTYPE *gstlal_ifft_ ## LONG ## DTYPE(LONG complex DTYPE *fd_data, guint N, gboolean normalize, gboolean free_input) { \
 \
	LONG complex DTYPE *td_data = gstlal_fft_ ## LONG ## DTYPE(fd_data, N, NULL, 0, NULL, TRUE, 0, NULL, 0, NULL, NULL, NULL, free_input); \
 \
	if(normalize) { \
		guint i; \
		for(i = 0; i < N; i++) \
			td_data[i] /= N; \
	} \
 \
	return td_data; \
}


GSTLAL_IFFT(, float);
GSTLAL_IFFT(, double);
GSTLAL_IFFT(long, double);


/*
 * If the input is real, the output is conjugate-symmetric: fd_data[n] = conj(fd_data[N - n]).
 * We can reduce the number of operations by a factor of ~2.  Also, we have the option to only
 * output half of the result, since the second half is redundant.
 */
#define GSTLAL_RFFT(LONG, DTYPE) \
LONG complex DTYPE *gstlal_rfft_ ## LONG ## DTYPE(LONG DTYPE *td_data, guint N, guint *prime_factors, guint num_factors, long complex double *exp_array, gboolean return_full, guint M, guint *M_prime_factors, guint M_num_factors, long complex double *M_exp_array2, long complex double *M_exp_array, long complex double *fd_data, gboolean free_input) { \
 \
	guint N_out = N / 2 + 1; \
 \
	if(N < 2) \
		return (LONG complex DTYPE *) td_data; \
 \
	gboolean prime_factors_need_freed = FALSE; \
	gboolean M_prime_factors_need_freed = FALSE; \
	gboolean exp_array_need_freed = FALSE; \
 \
	if(prime_factors == NULL) { \
		/* Find prime factors */ \
		prime_factors = find_prime_factors(N, &num_factors); \
		prime_factors_need_freed = TRUE; \
 \
		/* Check if we will need to use gstlal_prime_fft_() for this */ \
		if(prime_factors[num_factors - 2] >= 607) { \
			/* Find the first member greater than or equal to 607 */ \
			guint i = 0; \
			while(prime_factors[i] < 607) \
				i += 1; \
 \
			/* Compute a good padded length for Bluestein's algorithm */ \
			guint j, M_out, M_in = prime_factors[i]; \
			for(j = i + 1; j < num_factors; j++) \
				M_in *= prime_factors[j]; \
			M_out = M_in / 2 + 1;; \
			M_prime_factors = find_prime_factors_M(M_in + M_out - 1, &M, &M_num_factors); \
 \
			/* Find the array of exponentials for Bluestein's algorithm */ \
			M_exp_array2 = find_exp_array2(M_in, FALSE); \
			M_exp_array = find_exp_array(M, FALSE); \
 \
			M_prime_factors_need_freed = TRUE; \
		} \
	} \
	if(exp_array == NULL && prime_factors[0] < 607) { \
		/*
		 * Make array of exp(-2 pi i f t) to multiply.  This is expensive, so only make
		 * the code do it once.
		 */ \
		exp_array = find_exp_array(N, FALSE); \
		exp_array_need_freed = TRUE; \
	} \
 \
	if(prime_factors[0] >= 607) { \
		/* Use Bluestein's algorithm for a prime-length fft */ \
		if(fd_data == NULL) \
			fd_data = g_malloc0((return_full ? N : N_out) * sizeof(long complex double)); \
		gstlal_prime_rfft_ ## LONG ## DTYPE(td_data, N, return_full, M_exp_array2, M, M_prime_factors, M_num_factors, M_exp_array, fd_data); \
	} else if(prime_factors[0] == N) { \
		/* Do an ordinary DFT */ \
		if(fd_data == NULL) \
			fd_data = g_malloc0((return_full ? N : N_out) * sizeof(long complex double)); \
		gstlal_rdft_ ## LONG ## DTYPE(td_data, N, exp_array, return_full, fd_data); \
	} else { \
		/*
		 * We will break this up into smaller Fourier transforms.  Therefore, we still
		 * need to allocate enough memory for N elements.
		 */ \
		if(fd_data == NULL) \
			fd_data = g_malloc0(N * sizeof(long complex double)); \
		guint i, num_ffts = prime_factors[0]; \
		guint N_mini = N / num_ffts; \
		guint N_mini_out = N_mini / 2 + 1; \
		long complex double *exp_array_subset = array_subset_longcomplexdouble(exp_array, N, num_ffts); \
		for(i = 0; i < num_ffts; i++) \
			gstlal_rfft_longdouble(array_subset_ ## LONG ## DTYPE(td_data + i, N, num_ffts), N_mini, prime_factors + 1, num_factors - 1, exp_array_subset, TRUE, M, M_prime_factors, M_num_factors, M_exp_array2, M_exp_array, fd_data + i * N_mini, TRUE); \
		if(free_input) \
			g_free(td_data); \
		g_free(exp_array_subset); \
 \
		/* Now we need to "mix" the output appropriately.  First, copy all but the first fft. */ \
		guint N_fd_data_copy = (num_ffts - 1) * N_mini_out; \
		long complex double *fd_data_copy = array_subset_mod_n_longcomplexdouble(fd_data + N_mini, N - N_mini, N_mini); \
		/* Apply phase rotations to all but the first fft */ \
		guint exp_index; \
		for(i = N_mini; i < N_out; i++) { \
			exp_index = (i * (i / N_mini)) % N; \
			/* Do a multiplication only if we have to */ \
			if(exp_index) \
				fd_data[i] *= exp_array[exp_index]; \
		} \
		/* Add the first fft to all the others */ \
		for(i = N_mini; i < N_out; i++) \
			fd_data[i] += fd_data[i % N_mini]; \
 \
		/* Now we have to use the copied data.  Apply phase rotations and add to all other locations. */ \
		guint original_index, j; \
		for(i = 0; i < N_fd_data_copy; i++) { \
			original_index = N_mini + i / N_mini_out * N_mini + i % N_mini_out; \
			/*
			 * Note that we skip j == original_index below, since we took care of
			 * that contribution 2 for loops ago
			 */ \
			for(j = original_index % N_mini; j < N_out; j += N_mini, j += (j == original_index ? N_mini : 0)) { \
				exp_index = (j * (original_index / N_mini)) % N; \
				/* Do a multiplication only if we have to */ \
				if(exp_index) \
					fd_data[j] += fd_data_copy[i] * exp_array[exp_index]; \
				else \
					fd_data[j] += fd_data_copy[i]; \
			} \
			if(original_index % N_mini && original_index % N_mini < (N_mini + 1) / 2) { \
				/* Then handle the contribution from the complex conjugate */ \
				original_index += N_mini - 2 * (original_index % N_mini); \
				/* Again, skip j == original_index */ \
				for(j = original_index % N_mini; j < N_out; j += N_mini, j += (j == original_index ? N_mini : 0)) { \
					exp_index = (j * (original_index / N_mini)) % N; \
					/* Do a multiplication only if we have to */ \
					if(exp_index) \
						fd_data[j] += conjl(fd_data_copy[i]) * exp_array[exp_index]; \
					else \
						fd_data[j] += conjl(fd_data_copy[i]); \
				} \
			} \
		} \
		if(!(N % 2)) \
			/* The Nyquist component is real */ \
			fd_data[N_out - 1] = creall(fd_data[N_out - 1]); \
 \
		if(return_full && N > 2) { \
			/* Then fill in the second half */ \
			guint i; \
			for(i = 1; i <= N - N_out; i++) \
				fd_data[N - i] = conjl(fd_data[i]); \
		} \
	} \
 \
	/* Done */ \
	if(exp_array_need_freed) \
		g_free(exp_array); \
	if(prime_factors_need_freed) \
		g_free(prime_factors); \
	if(M_prime_factors_need_freed) { \
		g_free(M_prime_factors); \
		g_free(M_exp_array); \
		g_free(M_exp_array2); \
	} \
	if(sizeof(LONG DTYPE) < sizeof(long double)) \
		return typecast_longcomplexdouble_to_ ## LONG ## complex ## DTYPE(fd_data, return_full ? N : N_out); \
	else \
		return (LONG complex DTYPE *) fd_data; \
}


GSTLAL_RFFT(, float);
GSTLAL_RFFT(, double);
GSTLAL_RFFT(long, double);


/*
 * Inverse of the above real-input FFT.  So the output of this is real and the input is assumed
 * to be shortened to N / 2 + 1 samples to avoid redundancy.
 */
#define GSTLAL_IRFFT(LONG, DTYPE, LLL, FF) \
LONG DTYPE *gstlal_irfft_ ## LONG ## DTYPE(LONG complex DTYPE *fd_data, guint N_in, guint *N, guint *prime_factors, guint num_factors, long complex double *exp_array, gboolean normalize, guint M_fft, guint *M_fft_prime_factors, guint M_fft_num_factors, long complex double *M_fft_exp_array2, long complex double *M_fft_exp_array, guint M_irfft, guint *M_irfft_prime_factors, guint M_irfft_num_factors, long complex double *M_irfft_exp_array2, long complex double *M_irfft_exp_array, long double *td_data, gboolean free_input) { \
 \
	if(N_in < 2) { \
		LONG DTYPE *out = g_malloc(N_in * sizeof(LONG DTYPE)); \
		*out = (LONG DTYPE) creal ## LLL ## FF(*fd_data); \
		return out; \
	} \
 \
	if(!N) \
		N = g_malloc(sizeof(guint)); \
 \
	gboolean prime_factors_need_freed = FALSE; \
	gboolean M_irfft_prime_factors_need_freed = FALSE; \
	gboolean M_fft_prime_factors_need_freed = FALSE; \
	gboolean exp_array_need_freed = FALSE; \
 \
	if(prime_factors == NULL) { \
		/*
		 * First, find N, the original number of samples. If the imaginary part of the last
		 * sample is zero, assume N was even
		 */ \
		if(cimag ## LLL ## FF(fd_data[N_in - 1]) == 0) \
			*N = (N_in - 1) * 2; \
		else if(creal ## LLL ## FF(fd_data[N_in - 1]) == 0) \
			*N = N_in * 2 - 1; \
		else if(fabs ## LLL ## FF(cimag ## LLL ## FF(fd_data[N_in - 1]) / creal ## LLL ## FF(fd_data[N_in - 1])) < 1e-10) \
			*N = (N_in - 1) * 2; \
		else \
			*N = N_in * 2 - 1; \
		/* Find prime factors */ \
		prime_factors = find_prime_factors(*N, &num_factors); \
		prime_factors_need_freed = TRUE; \
 \
		/* Check if we will need to use gstlal_prime_irfft_() for this */ \
		if(prime_factors[num_factors - 2] >= 113) { \
			/* Find the first member greater than or equal to 113 */ \
			guint i = 0; \
			while(prime_factors[i] < 113) \
				i++; \
 \
			/* Compute a good padded length for Bluestein's algorithm */ \
			guint j, M_min = 2; \
			for(j = i; j < num_factors; j++) \
				M_min *= prime_factors[j]; \
			M_min--; \
			M_irfft_prime_factors = find_prime_factors_M(M_min, &M_irfft, &M_irfft_num_factors); \
 \
			/* Find the array of exponentials for Bluestein's algorithm */ \
			M_irfft_exp_array2 = find_exp_array2((M_min + 1) / 2, TRUE); \
			M_irfft_exp_array = find_exp_array(M_irfft, FALSE); \
 \
			M_irfft_prime_factors_need_freed = TRUE; \
		} \
		/* Check if we will need to use gstlal_prime_fft_() for this */ \
		if(prime_factors[num_factors - 2] >= 107) { \
			/* Find the first member greater than or equal to 107 */ \
			guint i = 0; \
			while(prime_factors[i] < 107) \
				i++; \
 \
			/* Compute a good padded length for Bluestein's algorithm */ \
			guint j, M_min = 2; \
			for(j = i; j < num_factors; j++) \
				M_min *= prime_factors[j]; \
			M_min--; \
			M_fft_prime_factors = find_prime_factors_M(M_min, &M_fft, &M_fft_num_factors); \
 \
			/* Find the array of exponentials for Bluestein's algorithm */ \
			M_fft_exp_array2 = find_exp_array2((M_min + 1) / 2, TRUE); \
			M_fft_exp_array = find_exp_array(M_fft, FALSE); \
 \
			M_fft_prime_factors_need_freed = TRUE; \
		} \
	} \
 \
	if(exp_array == NULL && prime_factors[0] < 113) { \
		/*
		 * Make array of exp(-2 pi i f t) to multiply.  This is expensive, so only make
		 * the code do it once.
		 */ \
		exp_array = find_exp_array(*N, TRUE); \
		exp_array_need_freed = TRUE; \
	} \
 \
	if(td_data == NULL) \
		td_data = g_malloc0(*N * sizeof(long double)); \
 \
	if(prime_factors[0] >= 113) \
		/* Use Bluestein's algorithm for a prime-length fft */ \
		gstlal_prime_irfft_ ## LONG ## DTYPE(fd_data, N_in, N, normalize, M_irfft_exp_array2, M_irfft, M_irfft_prime_factors, M_irfft_num_factors, M_irfft_exp_array, td_data); \
	else if(prime_factors[0] == *N) \
		/* Do an ordinary DFT */ \
		gstlal_irdft_ ## LONG ## DTYPE(fd_data, N_in, N, exp_array, normalize, td_data); \
	else { \
		/* We will break this up into smaller Fourier transforms */ \
		guint num_ffts = prime_factors[0]; \
		guint N_in_mini = (N_in + num_ffts - 1) / num_ffts; \
		guint N_mini = *N / num_ffts; \
		long complex double *exp_array_subset = array_subset_longcomplexdouble(exp_array, *N, num_ffts); \
		gstlal_irfft_longdouble(array_subset_ ## LONG ## complex ## DTYPE(fd_data, N_in, num_ffts), N_in_mini, &N_mini, prime_factors + 1, num_factors - 1, exp_array_subset, FALSE, M_fft, M_fft_prime_factors, M_fft_num_factors, M_fft_exp_array2, M_fft_exp_array, M_irfft, M_irfft_prime_factors, M_irfft_num_factors, M_irfft_exp_array2, M_irfft_exp_array, td_data, TRUE); \
 \
		/* The rest of the transforms will, in general, produce complex output */ \
		long complex double *td_data_complex = g_malloc0((*N - N_mini) * sizeof(long complex double)); \
		guint i; \
		for(i = 1; i < num_ffts; i++) \
			gstlal_fft_longdouble(array_subset_conj_ ## LONG ## DTYPE(fd_data, i, N_in, *N, num_ffts), N_mini, prime_factors + 1, num_factors - 1, exp_array_subset, FALSE, M_fft, M_fft_prime_factors, M_fft_num_factors, M_fft_exp_array2, M_fft_exp_array, td_data_complex + (i - 1) * N_mini, TRUE); \
 \
		if(free_input) \
			g_free(fd_data); \
		g_free(exp_array_subset); \
 \
		/* Now we need to "mix" the output appropriately.  Start by adding the first ifft to the others. */ \
		for(i = N_mini; i < *N; i++) \
			td_data[i] += td_data[i % N_mini]; \
 \
		/* Now use the complex data.  Apply phase rotations and add real parts to all other locations. */ \
		guint j, complex_index, exp_index; \
		for(i = N_mini; i < *N; i++) { \
			complex_index = i - N_mini; \
			for(j = i % N_mini; j < *N; j += N_mini) { \
				exp_index = (j * (i / N_mini)) % *N; \
				/* Do a multiplication only if we have to */ \
				if(exp_index) \
					td_data[j] += creall(td_data_complex[complex_index] * exp_array[exp_index]); \
				else \
					td_data[j] += creall(td_data_complex[complex_index]); \
			} \
		} \
		if(normalize) { \
			for(i = 0; i < *N; i++) \
				td_data[i] /= *N; \
		} \
	} \
	/* Done */ \
	if(exp_array_need_freed) \
		g_free(exp_array); \
	if(prime_factors_need_freed) \
		g_free(prime_factors); \
	if(M_irfft_prime_factors_need_freed) { \
		g_free(M_irfft_prime_factors); \
		g_free(M_irfft_exp_array); \
		g_free(M_irfft_exp_array2); \
	} \
	if(M_fft_prime_factors_need_freed) { \
		g_free(M_fft_prime_factors); \
		g_free(M_fft_exp_array); \
		g_free(M_fft_exp_array2); \
	} \
 \
	if(sizeof(LONG DTYPE) < sizeof(long double)) \
		return typecast_longdouble_to_ ## LONG ## DTYPE(td_data, *N); \
	else \
		return (LONG DTYPE *) td_data; \
}


GSTLAL_IRFFT(, float, , f);
GSTLAL_IRFFT(, double, , );
GSTLAL_IRFFT(long, double, l, );


/*
 * Bluestein's algorithm for FFTs of prime length, for which the Cooley-Tukey algorithm is
 * ineffective.  Make the replacement nk -> -(k - n)^2 / 2 + n^2 / 2 + k^2 / 2.
 * Then X_k = sum_(n=0)^(N-1) x_n * exp(-2*pi*i*n*k/N)
 *	   = exp(-pi*i*k^2/N) * sum_(n=0)^(N-1) x_n * exp(-pi*i*n^2/N) * exp(pi*i*(k-n)^2/N)
 * This can be done as a cyclic convolution between the sequences a_n = x_n * exp(-pi*i*n^2/N)
 * and b_n = exp(pi*i*n^2/N), with the output multiplied by conj(b_k).
 * a_n and b_n can be padded with zeros to make their lengths a power of 2.  The zero-padding
 * for a_n is done simply by adding zeros at the end, but since the index k - n can be negative
 * and b_{-n} = b_n, the padding has to be done differently.  Since k - n can take on 2N - 1
 * values, it is necessary to make the new arrays a length N' >= 2N - 1.  The new arrays are
 *
 *	  |--
 *	  | a_n,	0 <= n < N
 * A_n = -|
 *	  | 0,		N <= n < N'
 *	  |--
 *
 *	  |--
 *	  | b_n,	0 <= n < N
 * B_n = -| 0,		N <= n <= N' - N
 *	  | b_{N'-n},	N' - N <= n < N'
 *	  |--
 *
 * The convolution of A_n and B_n can be evaluated using the convolution theorem and the
 * Cooley-Tukey FFT algorithm:
 * X_k = conj(b_k) * ifft(fft(A_n) * fft(B_n))[:N]
 */


#define GSTLAL_PRIME_FFT(LONG, DTYPE) \
LONG complex DTYPE *gstlal_prime_fft_ ## LONG ## DTYPE(LONG complex DTYPE *td_data, guint N, gboolean inverse, long complex double *exp_array2, guint M, guint *prime_factors, guint num_factors, long complex double *exp_array, long complex double *fd_data) { \
 \
	gboolean exp_array2_need_freed = FALSE; \
	gboolean exp_array_need_freed = FALSE; \
	gboolean prime_factors_need_freed = FALSE; \
 \
	/* Find the array of exponentials. */ \
	if(exp_array2 == NULL) { \
		exp_array2 = find_exp_array2(N, inverse); \
		exp_array2_need_freed = TRUE; \
	} \
 \
	/* Find the sequences we need, padding with zeros as necessary. */ \
	if(M == 0 || prime_factors == NULL) { \
		prime_factors = find_prime_factors_M(2 * N - 1, &M, &num_factors); \
		prime_factors_need_freed = TRUE; \
	} \
 \
	if(exp_array == NULL) { \
		exp_array = find_exp_array(M, FALSE); \
		exp_array_need_freed = TRUE; \
	} \
 \
	long complex double *A_n = pad_zeros_A ## LONG ## complex ## DTYPE(td_data, N, 0, exp_array2, M); \
	long complex double *b_n = conj_array_longdouble(exp_array2, N); \
	long complex double *B_n = pad_zeros_Blong(b_n, N, N, M); \
 \
	/*
	 * Do the convolution using the convolution theorem and the Cooley-Tukey algorithm, and
	 * multiply by exp_array2.
	 */ \
	long complex double *A_n_fft = g_malloc0(M * sizeof(long complex double)); \
	long complex double *B_n_fft = g_malloc0(M * sizeof(long complex double)); \
	long complex double *A_n_conv_B_n = g_malloc0(M * sizeof(long complex double)); \
	long complex double *conj_exp_array = conj_array_longdouble(exp_array, M); \
	gstlal_fft_longdouble(A_n, M, prime_factors, num_factors, exp_array, FALSE, 0, NULL, 0, NULL, NULL, A_n_fft, TRUE); \
	gstlal_fft_longdouble(B_n, M, prime_factors, num_factors, exp_array, FALSE, 0, NULL, 0, NULL, NULL, B_n_fft, TRUE); \
 \
	guint i; \
	for(i = 0; i < M; i++) \
		A_n_fft[i] *= B_n_fft[i]; \
 \
	gstlal_fft_longdouble(A_n_fft, M, prime_factors, num_factors, conj_exp_array, TRUE, 0, NULL, 0, NULL, NULL, A_n_conv_B_n, TRUE); \
 \
	if(fd_data == NULL) \
		fd_data = g_malloc(N * sizeof(long complex double)); \
 \
	for(i = 0; i < N; i++) \
		fd_data[i] = exp_array2[i] * A_n_conv_B_n[i] / M; \
 \
	/* Done */ \
	g_free(b_n); \
	g_free(B_n_fft); \
	g_free(A_n_conv_B_n); \
	g_free(conj_exp_array); \
	if(exp_array2_need_freed) \
		g_free(exp_array2); \
	if(exp_array_need_freed) \
		g_free(exp_array); \
	if(prime_factors_need_freed) { \
		g_free(prime_factors); \
	} \
 \
	if(sizeof(LONG DTYPE) < sizeof(long double)) \
		return typecast_longcomplexdouble_to_ ## LONG ## complex ## DTYPE(fd_data, N); \
	else \
		return (LONG complex DTYPE *) fd_data; \
}


GSTLAL_PRIME_FFT(, float);
GSTLAL_PRIME_FFT(, double);
GSTLAL_PRIME_FFT(long, double);


/*
 * If the input is real, the output is conjugate-symmetric: fd_data[n] = conj(fd_data[N - n]).
 * We can reduce the number of operations by a factor of ~2.  Also, we have the option to only
 * output half of the result, since the second half is redundant.
 */
#define GSTLAL_PRIME_RFFT(LONG, DTYPE) \
LONG complex DTYPE *gstlal_prime_rfft_ ## LONG ## DTYPE(LONG DTYPE *td_data, guint N, gboolean return_full, long complex double *exp_array2, guint M, guint *prime_factors, guint num_factors, long complex double *exp_array, long complex double *fd_data) { \
 \
	guint N_out = N / 2 + 1; \
 \
	gboolean exp_array2_need_freed = FALSE; \
	gboolean exp_array_need_freed = FALSE; \
	gboolean prime_factors_need_freed = FALSE; \
 \
	/* Find the array of exponentials. */ \
	if(exp_array2 == NULL) { \
		exp_array2 = find_exp_array2(N, FALSE); \
		exp_array2_need_freed = TRUE; \
	} \
 \
	/* Find the sequences we need, padding with zeros as necessary. */ \
	if(M == 0 || prime_factors == NULL) { \
		prime_factors = find_prime_factors_M(N + N_out - 1, &M, &num_factors); \
		prime_factors_need_freed = TRUE; \
	} \
 \
	if(exp_array == NULL) { \
		exp_array = find_exp_array(M, FALSE); \
		exp_array_need_freed = TRUE; \
	} \
 \
	long complex double *A_n = pad_zeros_A ## LONG ## DTYPE(td_data, N, 0, exp_array2, M); \
	long complex double *b_n = conj_array_longdouble(exp_array2, N); \
	long complex double *B_n = pad_zeros_Blong(b_n, N, N_out, M); \
 \
	/*
	 * Do the convolution using the convolution theorem and the Cooley-Tukey algorithm, and
	 * multiply by exp_array2.
	 */ \
	long complex double *A_n_fft = g_malloc0(M * sizeof(long complex double)); \
	long complex double *B_n_fft = g_malloc0(M * sizeof(long complex double)); \
	long complex double *A_n_conv_B_n = g_malloc0(M * sizeof(long complex double)); \
	long complex double *conj_exp_array = conj_array_longdouble(exp_array, M); \
	gstlal_fft_longdouble(A_n, M, prime_factors, num_factors, exp_array, FALSE, 0, NULL, 0, NULL, NULL, A_n_fft, TRUE); \
	gstlal_fft_longdouble(B_n, M, prime_factors, num_factors, exp_array, FALSE, 0, NULL, 0, NULL, NULL, B_n_fft, TRUE); \
 \
	guint i; \
	for(i = 0; i < M; i++) \
		A_n_fft[i] *= B_n_fft[i]; \
 \
	gstlal_fft_longdouble(A_n_fft, M, prime_factors, num_factors, conj_exp_array, TRUE, 0, NULL, 0, NULL, NULL, A_n_conv_B_n, TRUE); \
 \
	if(fd_data == NULL) \
		fd_data = g_malloc((return_full ? N : N_out) * sizeof(long complex double)); \
 \
	for(i = 0; i < N_out; i++) \
		fd_data[i] = exp_array2[i] * A_n_conv_B_n[i] / M; \
 \
	if(return_full && N > 2) { \
		/* Then fill in the second half */ \
		for(i = 1; i <= N - N_out; i++) \
			fd_data[N - i] = conjl(fd_data[i]); \
	} \
 \
	/* Done */ \
	g_free(b_n); \
	g_free(B_n_fft); \
	g_free(A_n_conv_B_n); \
	g_free(conj_exp_array); \
	if(exp_array2_need_freed) \
		g_free(exp_array2); \
	if(exp_array_need_freed) \
		g_free(exp_array); \
	if(prime_factors_need_freed) \
		g_free(prime_factors); \
 \
	if(sizeof(LONG DTYPE) < sizeof(long double)) \
		return typecast_longcomplexdouble_to_ ## LONG ## complex ## DTYPE(fd_data, return_full ? N : N_out); \
	else \
		return (LONG complex DTYPE *) fd_data; \
}


GSTLAL_PRIME_RFFT(, float);
GSTLAL_PRIME_RFFT(, double);
GSTLAL_PRIME_RFFT(long, double);


/*
 * Inverse of the above real-input FFT.  So the output of this is real and the input is assumed
 * to be shortened to N / 2 + 1 samples to avoid redundancy.
 */
#define GSTLAL_PRIME_IRFFT(LONG, DTYPE, LLL, FF) \
LONG DTYPE *gstlal_prime_irfft_ ## LONG ## DTYPE(LONG complex DTYPE *fd_data, guint N_in, guint *N, gboolean normalize, long complex double *exp_array2, guint M, guint *prime_factors, guint num_factors, long complex double *exp_array, long double *td_data) { \
 \
	if(!N) { \
		N = g_malloc(sizeof(guint)); \
		*N = 0; \
	} \
	if(*N == 0) { \
		/*
		 * Find N, the original number of samples. If the imaginary part of the last
		 * sample is zero, assume N was even
		 */ \
		if(cimag ## LLL ## FF(fd_data[N_in - 1]) == 0) \
			*N = (N_in - 1) * 2; \
		else if(creal ## LLL ## FF(fd_data[N_in - 1]) == 0) \
			*N = N_in * 2 - 1; \
		else if(cabs ## LLL ## FF(cimag ## LLL ## FF(fd_data[N_in - 1]) / creal ## LLL ## FF(fd_data[N_in - 1])) < 1e-10) \
			*N = (N_in - 1) * 2; \
		else \
			*N = N_in * 2 - 1; \
	} \
 \
	gboolean exp_array2_need_freed = FALSE; \
	gboolean exp_array_need_freed = FALSE; \
	gboolean prime_factors_need_freed = FALSE; \
 \
	/* Find the array of exponentials. */ \
	if(exp_array2 == NULL) { \
		exp_array2 = find_exp_array2(*N, TRUE); \
		exp_array2_need_freed = TRUE; \
	} \
 \
	/* Find the sequences we need, padding with zeros as necessary. */ \
	if(M == 0 || prime_factors == NULL) { \
		prime_factors = find_prime_factors_M(2 * *N - 1, &M, &num_factors); \
		prime_factors_need_freed = TRUE; \
	} \
 \
	if(exp_array == NULL) { \
		exp_array = find_exp_array(M, FALSE); \
		exp_array_need_freed = TRUE; \
	} \
 \
	long complex double *A_n = pad_zeros_A ## LONG  ## complex ## DTYPE(fd_data, N_in, *N - N_in, exp_array2, M); \
	long complex double *b_n = conj_array_longdouble(exp_array2, *N); \
	long complex double *B_n = pad_zeros_Blong(b_n, *N, *N, M); \
 \
	/*
	 * Do the convolution using the convolution theorem and the Cooley-Tukey algorithm, and
	 * multiply by exp_array2.
	 */ \
	long complex double *A_n_fft = g_malloc0(M * sizeof(long complex double)); \
	long complex double *B_n_fft = g_malloc0(M * sizeof(long complex double)); \
	long complex double *A_n_conv_B_n = g_malloc0(M * sizeof(long complex double)); \
	long complex double *conj_exp_array = conj_array_longdouble(exp_array, M); \
	gstlal_fft_longdouble(A_n, M, prime_factors, num_factors, exp_array, FALSE, 0, NULL, 0, NULL, NULL, A_n_fft, TRUE); \
	gstlal_fft_longdouble(B_n, M, prime_factors, num_factors, exp_array, FALSE, 0, NULL, 0, NULL, NULL, B_n_fft, TRUE); \
 \
	guint i; \
	for(i = 0; i < M; i++) \
		A_n_fft[i] *= B_n_fft[i]; \
 \
	gstlal_fft_longdouble(A_n_fft, M, prime_factors, num_factors, conj_exp_array, TRUE, 0, NULL, 0, NULL, NULL, A_n_conv_B_n, TRUE); \
 \
	if(td_data == NULL) \
		td_data = g_malloc(*N * sizeof(long double)); \
 \
	for(i = 0; i < *N; i++) \
		td_data[i] = creall(exp_array2[i] * A_n_conv_B_n[i]) / M; \
 \
	if(normalize) { \
		for(i = 0; i < *N; i++) \
			td_data[i] /= *N; \
	} \
 \
	/* Done */ \
	g_free(b_n); \
	g_free(B_n_fft); \
	g_free(A_n_conv_B_n); \
	g_free(conj_exp_array); \
	if(exp_array2_need_freed) \
		g_free(exp_array2); \
	if(exp_array_need_freed) \
		g_free(exp_array); \
	if(prime_factors_need_freed) \
		g_free(prime_factors); \
 \
	if(sizeof(LONG DTYPE) < sizeof(long double)) \
		return typecast_longdouble_to_ ## LONG ## DTYPE(td_data, *N); \
	else \
		return (LONG DTYPE *) td_data; \
}


GSTLAL_PRIME_IRFFT(, float, , f);
GSTLAL_PRIME_IRFFT(, double, , );
GSTLAL_PRIME_IRFFT(long, double, l, );


#define RAND_ARRAY(LONG, COMPLEX) \
LONG COMPLEX double *rand_array ## LONG ## COMPLEX(guint N) { \
 \
	LONG COMPLEX double *array = g_malloc0(N * sizeof(LONG COMPLEX double)); \
	guint i; \
	time_t t; \
	srand((unsigned) time(&t)); \
	for(i = 0; i < N; i++) \
		array[i] += (rand() - RAND_MAX / 2.0) / RAND_MAX; \
		/* * pow(10.0, (rand() - RAND_MAX / 2.0) * 5.0 / RAND_MAX); */ \
	if(sizeof(COMPLEX double) > 8) { \
		for(i = 0; i < N; i++) \
			array[i] += I * (rand() - RAND_MAX / 2.0) / RAND_MAX; \
			/* * pow(10.0, (rand() - RAND_MAX / 2.0) * 5.0 / RAND_MAX); */ \
	} \
	return array; \
}


RAND_ARRAY(, );
RAND_ARRAY(long, );
RAND_ARRAY(, complex);
RAND_ARRAY(long, complex);


#define COMPARE_SPEED(FT_TYPE, INVERSE, COMPLEX_IN, COMPLEX_OUT, COMPLEX_R) \
void compare_speed_ ## INVERSE ## FT_TYPE ## fft(guint N, guint iterations) { \
 \
	/* N should be odd for this function */ \
	N += (N + 1) % 2; \
 \
	struct timespec spec_start, spec_end; \
	double dft_diff = 0; \
	double primefft_diff = 0; \
 \
	guint N_in = sizeof(COMPLEX_IN double) > sizeof(COMPLEX_OUT double) ? N / 2 + 1 : N; \
	guint N_out = sizeof(COMPLEX_OUT double) > sizeof(COMPLEX_IN double) ? N / 2 + 1 : N; \
 \
	/* Generate some stuff the fft functions will use */ \
	long complex double *exp_array = find_exp_array(N, FALSE); \
	long complex double *exp_array2 = find_exp_array2(N, FALSE); \
	guint M, M_num_factors; \
	guint *M_prime_factors = find_prime_factors_M(N + N_out - 1, &M, &M_num_factors); \
	long complex double *M_exp_array = find_exp_array(M, FALSE); \
 \
	/* Do some dft's */ \
	guint i; \
	if(sizeof(COMPLEX_OUT double) <= 8) { \
 \
		/* It's an inverse real FT */ \
		long complex double *input; \
		long double *output = g_malloc0(N * sizeof(long double)); \
		for(i = 0; i < iterations; i++) { \
			/* Generate a random array */ \
			input = rand_arraylongcomplex(N_in); \
 \
			/* get start time for dft */ \
			clock_gettime(CLOCK_REALTIME, &spec_start); \
			/* do the FT */ \
			gstlal_irdft_longdouble(input, N_in, &N, exp_array, FALSE, output); \
			/* get end time for dft */ \
			clock_gettime(CLOCK_REALTIME, &spec_end); \
 \
			/* Add time elapsed in seconds */ \
			dft_diff += ((intmax_t) spec_end.tv_sec + ((double) spec_end.tv_nsec) / 1e9) - ((intmax_t) spec_start.tv_sec + ((double) spec_start.tv_nsec) / 1e9); \
		} \
		g_free(output); \
	} else { \
 \
		/* It is not an inverse FT */ \
		long COMPLEX_R double *input; \
		long complex double *output = g_malloc0(N * sizeof(long complex double)); \
		for(i = 0; i < iterations; i++) { \
			/* Generate a random array */ \
			input = rand_arraylong ## COMPLEX_R(N_in); \
 \
			/* get start time for dft */ \
			clock_gettime(CLOCK_REALTIME, &spec_start); \
			/* do the FT */ \
			gstlal_ ## FT_TYPE ## dft_longdouble(input, N, exp_array, TRUE, output); \
			/* get end time for dft */ \
			clock_gettime(CLOCK_REALTIME, &spec_end); \
 \
			/* Add time elapsed in seconds */ \
			dft_diff += ((intmax_t) spec_end.tv_sec + ((double) spec_end.tv_nsec) / 1e9) - ((intmax_t) spec_start.tv_sec + ((double) spec_start.tv_nsec) / 1e9); \
		} \
		g_free(output); \
	} \
 \
	/* get start time for prime fft */ \
	clock_gettime(CLOCK_REALTIME, &spec_start); \
 \
	/* Do some prime fft's */ \
	if(sizeof(COMPLEX_OUT double) <= 8) { \
 \
		/* It's an inverse real FT */ \
		long complex double *input; \
		long double *output = g_malloc0(N * sizeof(long double)); \
		for(i = 0; i < iterations; i++) { \
			/* Generate a random array */ \
			input = rand_arraylongcomplex(N_in); \
 \
			/* get start time for prime fft */ \
			clock_gettime(CLOCK_REALTIME, &spec_start); \
			/* do the FT */ \
			gstlal_prime_irfft_longdouble(input, N_in, &N, FALSE, exp_array2, M, M_prime_factors, M_num_factors, M_exp_array, output); \
			/* get end time for prime fft */ \
			clock_gettime(CLOCK_REALTIME, &spec_end); \
 \
			/* Add time elapsed in seconds */ \
			primefft_diff += ((intmax_t) spec_end.tv_sec + ((double) spec_end.tv_nsec) / 1e9) - ((intmax_t) spec_start.tv_sec + ((double) spec_start.tv_nsec) / 1e9); \
		} \
		g_free(output); \
	} else { \
 \
		/* It is not an inverse FT */ \
		long COMPLEX_R double *input; \
		long complex double *output = g_malloc0(N * sizeof(long complex double)); \
		for(i = 0; i < iterations; i++) { \
			/* Generate a random array */ \
			input = rand_arraylong ## COMPLEX_R(N_in); \
 \
			/* get start time for prime fft */ \
			clock_gettime(CLOCK_REALTIME, &spec_start); \
			/* do the FT */ \
			gstlal_prime_ ## FT_TYPE ## fft_longdouble(input, N, TRUE, exp_array2, M, M_prime_factors, M_num_factors, M_exp_array, output); \
			/* get end time for dft */ \
			clock_gettime(CLOCK_REALTIME, &spec_end); \
 \
			/* Add time elapsed in seconds */ \
			primefft_diff += ((intmax_t) spec_end.tv_sec + ((double) spec_end.tv_nsec) / 1e9) - ((intmax_t) spec_start.tv_sec + ((double) spec_start.tv_nsec) / 1e9); \
		} \
		g_free(output); \
	} \
 \
	/* Done */ \
	g_free(exp_array); \
	g_free(exp_array2); \
	g_free(M_prime_factors); \
	g_free(M_exp_array); \
	g_print("N = %u: time(primefft) / time(dft) = %f\n", N, primefft_diff / dft_diff); \
 \
	return; \
}


COMPARE_SPEED(, , complex, complex, complex);
COMPARE_SPEED(r, , , complex, );
COMPARE_SPEED(r, i, complex, , );


void fft_test_inverse(guint N_start, guint N_end, guint cadence) {

	guint i, j, max_error_index = 0;
	long double max_error = 0.0;
	long double avg_error = 0.0;

	g_print("\n====================================================================\n");
	g_print("===================== TESTING IFFT(FFT(x)) = x =====================\n");
	g_print("============= REPORTING FRACTIONAL ERRORS IN MAGNITUDE =============\n");
	g_print("====================================================================\n\n");

	for(i = N_start; i < N_end; i += cadence) {
		long complex double *input = rand_arraylongcomplex(i);
		long complex double *copy = g_malloc(i * sizeof(long complex double));
		memcpy(copy, input, i * sizeof(long complex double));
		long complex double *output = gstlal_ifft_longdouble(gstlal_fft_longdouble(input, i, NULL, 0, NULL, FALSE, 0, NULL, 0, NULL, NULL, NULL, TRUE), i, TRUE, TRUE);
		for(j = 0; j < i; j++) {
			output[j] /= copy[j];
			output[j] -= 1;
			max_error_index = cabsl(output[j]) > max_error ? j : max_error_index;
			max_error = cabsl(output[j]) > max_error ? cabsl(output[j]) : max_error;
			avg_error += cabsl(output[j]);
		}
		avg_error /= i;

		g_print("--------------------\n");
		g_print("Results for N = %u:\n", i);
		g_print("avg error: %Le\n", avg_error);
		g_print("max error: %Le index: %u\n", max_error, max_error_index);
		g_print("--------------------\n\n");

		g_free(output);
		g_free(copy);
	}
	return;
}


void rfft_test_inverse(guint N_start, guint N_end, guint cadence) {

	guint i, j, max_error_index = 0;
	long double max_error = 0.0;
	long double avg_error = 0.0;

	g_print("\n====================================================================\n");
	g_print("==================== TESTING IRFFT(RFFT(x)) = x ====================\n");
	g_print("============= REPORTING FRACTIONAL ERRORS IN MAGNITUDE =============\n");
	g_print("====================================================================\n\n");

	for(i = N_start; i < N_end; i += cadence) {
		long double *input = rand_arraylong(i);
		long double *copy = g_malloc(i * sizeof(long double));
		memcpy(copy, input, i * sizeof(long double));
		long double *output = gstlal_irfft_longdouble(gstlal_rfft_longdouble(input, i, NULL, 0, NULL, FALSE, 0, NULL, 0, NULL, NULL, NULL, TRUE), i / 2 + 1, &i, NULL, 0, NULL, TRUE, 0, NULL, 0, NULL, NULL, 0, NULL, 0, NULL, NULL, NULL, TRUE);
		for(j = 0; j < i; j++) {
			output[j] /= copy[j];
			output[j] -= 1;
			max_error_index = fabsl(output[j]) > max_error ? j : max_error_index;
			max_error = fabsl(output[j]) > max_error ? fabsl(output[j]) : max_error;
			avg_error += fabsl(output[j]);
		}
		avg_error /= i;

		g_print("--------------------\n");
		g_print("Results for N = %u:\n", i);
		g_print("avg error: %Le\n", avg_error);
		g_print("max error: %Le index: %u\n", max_error, max_error_index);
		g_print("--------------------\n\n");
	
		g_free(output);
		g_free(copy);
	}
	return;
}


void compare_fft(guint N_start, guint N_end, guint cadence) {

	guint i, j, max_error_index_dft, max_error_index_prime;
	long double test, max_error_dft, avg_error_dft, max_error_prime, avg_error_prime;

	g_print("\n====================================================================\n");
	g_print("================ COMPARING DFT, FFT, AND PRIME_FFT =================\n");
	g_print("========== REPORTING FRACTIONAL DIFFERENCES IN MAGNITUDE ===========\n");
	g_print("====================================================================\n\n");

	for(i = N_start; i < N_end; i += cadence) {
		long complex double *dftinput = rand_arraylongcomplex(i);
		long complex double *fftinput = g_malloc(i * sizeof(long complex double));
		long complex double *primefftinput = g_malloc(i * sizeof(long complex double));
		memcpy(fftinput, dftinput, i * sizeof(long complex double));
		memcpy(primefftinput, dftinput, i * sizeof(long complex double));
		long complex double *dft = gstlal_dft_longdouble(dftinput, i, NULL, FALSE, NULL);
		long complex double *fft = gstlal_fft_longdouble(fftinput, i, NULL, 0, NULL, FALSE, 0, NULL, 0, NULL, NULL, NULL, TRUE);
		long complex double *prime_fft = gstlal_prime_fft_longdouble(primefftinput, i, FALSE, NULL, 0, NULL, 0, NULL, NULL);
		max_error_index_dft = 0;
		max_error_index_prime = 0;
		max_error_dft = 0.0;
		avg_error_dft = 0.0;
		max_error_prime = 0.0;
		avg_error_prime = 0.0;
		for(j = 0; j < i; j++) {
			test = cabsl(dft[j] / fft[j] - 1);
			max_error_index_dft = test > max_error_dft ? j : max_error_index_dft;
			max_error_dft = test > max_error_dft ? test : max_error_dft;
			avg_error_dft += test;

			test = cabsl(prime_fft[j] / fft[j] - 1);
			max_error_index_prime = test > max_error_prime ? j : max_error_index_prime;
			max_error_prime = test > max_error_prime ? test : max_error_prime;
			avg_error_prime += test;
		}
		avg_error_dft /= i;
		avg_error_prime /= i;

		g_print("--------------------\n");
		g_print("Results for N = %u:\n\n", i);
		g_print("DFT vs FFT:\n");
		g_print("avg difference: %Le\n", avg_error_dft);
		g_print("max difference: %Le index: %u\n\n", max_error_dft, max_error_index_dft);
		g_print("PRIME_FFT vs FFT:\n");
		g_print("avg difference: %Le\n", avg_error_prime);
		g_print("max difference: %Le index: %u\n", max_error_prime, max_error_index_prime);
		g_print("--------------------\n\n");

		g_free(dft);
		g_free(fft);
		g_free(prime_fft);
	}
	return;
}


void compare_rfft(guint N_start, guint N_end, guint cadence) {

	guint i, N_out, j, max_error_index_dft = 0;
	long double test;
	long double max_error_dft = 0.0;
	long double avg_error_dft = 0.0;
	guint max_error_index_prime = 0;
	long double max_error_prime = 0.0;
	long double avg_error_prime = 0.0;

	g_print("\n====================================================================\n");
	g_print("=============== COMPARING RDFT, RFFT, AND PRIME_RFFT ===============\n");
	g_print("========== REPORTING FRACTIONAL DIFFERENCES IN MAGNITUDE ===========\n");
	g_print("====================================================================\n\n");

	for(i = N_start; i < N_end; i += cadence) {
		N_out = i / 2 + 1;
		long double *dftinput = rand_arraylong(i);
		long double *fftinput = g_malloc(i * sizeof(long double));
		long double *primefftinput = g_malloc(i * sizeof(long double));
		memcpy(fftinput, dftinput, i * sizeof(long double));
		memcpy(primefftinput, dftinput, i * sizeof(long double));
		long complex double *dft = gstlal_rdft_longdouble(dftinput, i, NULL, FALSE, NULL);
		long complex double *fft = gstlal_rfft_longdouble(fftinput, i, NULL, 0, NULL, FALSE, 0, NULL, 0, NULL, NULL, NULL, TRUE);
		long complex double *prime_fft = gstlal_prime_rfft_longdouble(primefftinput, i, FALSE, NULL, 0, NULL, 0, NULL, NULL);
		for(j = 0; j < N_out; j++) {
			test = cabsl(dft[j] / fft[j] - 1);
			max_error_index_dft = test > max_error_dft ? j : max_error_index_dft;
			max_error_dft = test > max_error_dft ? test : max_error_dft;
			avg_error_dft += test;

			test = cabsl(prime_fft[j] / fft[j] - 1);
			max_error_index_prime = test > max_error_prime ? j : max_error_index_prime;
			max_error_prime = test > max_error_prime ? test : max_error_prime;
			avg_error_prime += test;
		}
		avg_error_dft /= N_out;
		avg_error_prime /= N_out;

		g_print("--------------------\n");
		g_print("Results for N = %u:\n\n", i);
		g_print("RDFT vs RFFT:\n");
		g_print("avg difference: %Le\n", avg_error_dft);
		g_print("max difference: %Le index: %u\n\n", max_error_dft, max_error_index_dft);
		g_print("PRIME_RFFT vs RFFT:\n");
		g_print("avg difference: %Le\n", avg_error_prime);
		g_print("max difference: %Le index: %u\n", max_error_prime, max_error_index_prime);
		g_print("--------------------\n\n");

		g_free(dft);
		g_free(fft);
		g_free(prime_fft);
	}
	return;
}


/*
 * Below are several useful window functions, all to long double precision.
 */


/*
 * DPSS window
 */


/*
 * Compute a discrete prolate spheroidal sequence (DPSS) window,
 * which maximizes the energy concentration in the central lobe
 */


/*
 * A function to multiply a symmetric Toeplitz matrix times a vector and normalize.
 * Assume that only the first row of the matrix is stored, to save memory.
 * Assume that only half of the vector is stored, due to symmetry.
 */
#define MAT_TIMES_VEC(LONG, COMPLEX, DTYPE) \
LONG COMPLEX DTYPE *mat_times_vec_ ## LONG ## COMPLEX ## DTYPE(LONG COMPLEX DTYPE *mat, guint N, LONG COMPLEX DTYPE *vec, guint n) { \
 \
	LONG COMPLEX DTYPE *outvec = g_malloc0(n * sizeof(LONG COMPLEX DTYPE)); \
	guint i, j; \
	gint j1, j2; \
	for(i = 0; i < n; i++) { \
		j1 = N - 1 - i; \
		j2 = i; \
		for(j = 0; j < N - n; j++, j1--, j2--) \
			outvec[i] += vec[j] * (mat[j1] + mat[j2 >= 0 ? j2 : -j2]); \
		if(2 * n > N) \
			outvec[i] += vec[j] * mat[j1]; \
	} \
 \
	g_free(vec); \
 \
	/* Normalize */ \
	for(i = 0; i < n; i++) \
		outvec[i] /= outvec[n - 1]; \
 \
	return outvec; \
}


MAT_TIMES_VEC(, , float);
MAT_TIMES_VEC(, , double);
MAT_TIMES_VEC(long, , double);
MAT_TIMES_VEC(, complex, float);
MAT_TIMES_VEC(, complex, double);
MAT_TIMES_VEC(long, complex, double);


#define DPSS(LONG, DTYPE) \
LONG DTYPE *dpss_ ## LONG ## DTYPE(guint N, double alpha, double max_time, LONG DTYPE *data, gboolean half_window) { \
 \
	/*
	 * Estimate how long each process should take.  This is based on data taken from
	 * ldas-pcdev5 on the LHO cluster in June 2020.
	 */ \
	double seconds_per_iteration_double = 2.861e-10 * N * N - 9.134e-9 * N; \
	double seconds_per_iteration_longdouble = 9.422e-10 * N * N + 2.403e-9 * N; \
 \
	guint double_iterations = (guint) (max_time / 2.0 / seconds_per_iteration_double); \
	guint longdouble_iterations = (guint) (max_time / 2.0 / seconds_per_iteration_longdouble); \
 \
	/*
	 * Start with ordinary double precision to make it run faster.
	 * Angular cutoff frequency times sample period
	 */ \
	double omega_c_Ts = (double) (2 * PI * alpha / N); \
 \
	/*
	 * The DPSS window is the eigenvector associated with the largest eigenvalue of the symmetric
	 * Toeplitz matrix (Toeplitz means all elements along negative sloping diagonals are equal),
	 * where the zeroth column and row are the sampled sinc function below:
	 */ \
	double *sinc = g_malloc(N * sizeof(double)); \
	sinc[0] = omega_c_Ts; \
	guint i; \
	for(i = 1; i < N; i++) \
		sinc[i] = sin(omega_c_Ts * i) / i; \
 \
	/*
	 * Start by approximating the DPSS window with a Kaiser window with the same value of alpha.
	 * Note that kaiser() takes beta = pi * alpha as an argument.  Due to symmetry, we need to
	 * use only half of the window.
	 */ \
	double *dpss = kaiser_double(N, (double) (PI * alpha), NULL, FALSE); \
 \
	/*
	 * Now use power iteration to get our approximation closer to the true DPSS window.  This
	 * entails simply applying the eigenvalue equation over and over until we are satisfied
	 * with the accuracy of the eigenvector.  This method assumes the existance of an eigenvalue
	 * that is larger in magnitude than all the other eigenvalues.
	 */ \
 \
	guint n = N / 2 + N % 2; \
 \
	for(i = 0; i < double_iterations; i++) \
		dpss = mat_times_vec_double(sinc, N, dpss, n); \
 \
	g_free(sinc); \
 \
	/* Now do this with extra precision */ \
	long double *long_dpss = g_malloc(n * sizeof(long double)); \
	for(i = 0; i < n; i++) \
		long_dpss[i] = (long double) dpss[i]; \
 \
	g_free(dpss); \
 \
	long double long_omega_c_Ts = 2.0L * PI * alpha / N; \
	long double *long_sinc = g_malloc(N * sizeof(long double)); \
	long_sinc[0] = long_omega_c_Ts; \
	for(i = 1; i < N; i++) \
		long_sinc[i] = sinl(long_omega_c_Ts * i) / i; \
 \
	for(i = 0; i < longdouble_iterations; i++) \
		long_dpss = mat_times_vec_longdouble(long_sinc, N, long_dpss, n); \
 \
	g_free(long_sinc); \
 \
	LONG DTYPE *new_dpss; \
	if(sizeof(LONG DTYPE) < sizeof(long double)) \
		new_dpss = typecast_longdouble_to_ ## LONG ## DTYPE(long_dpss, n); \
	else \
		new_dpss = (LONG DTYPE *) long_dpss; \
 \
	LONG DTYPE *full_dpss = g_malloc(N * sizeof(LONG DTYPE)); \
	memcpy(full_dpss, new_dpss, n * sizeof(LONG DTYPE)); \
	for(i = 0; i < N - n; i++) \
		full_dpss[N - 1 - i] = full_dpss[i]; \
 \
	g_free(new_dpss); \
 \
	guint start = half_window ? N / 2 : 0; \
	LONG DTYPE *dpss_out; \
        if(half_window) { \
                dpss_out = g_malloc((N - start) * sizeof(LONG DTYPE)); \
                for(i = 0; i < N - start; i++) \
                        dpss_out[i] = full_dpss[start + i]; \
                g_free(full_dpss); \
        } else \
                dpss_out = full_dpss; \
 \
	if(data != NULL) { \
		for(i = 0; i < N - start; i++) \
			data[i] *= dpss_out[i]; \
		g_free(dpss_out); \
		return data; \
 \
	} else \
		return dpss_out; \
}


DPSS(, float);
DPSS(, double);
DPSS(long, double);


/*
 * Kaiser window, an approximation to the DPSS window
 */

/* Modified Bessel function of the first kind */
long double I0(long double x, long double *factorials_inv2) {
	long double out = 1.0L;
	guint i;
	for(i = 1; i < 35; i++)
		out += powl((x / 2), 2.0L * i) * factorials_inv2[i];

	return out;
}


#define KAISER(LONG, DTYPE) \
LONG DTYPE *kaiser_ ## LONG ## DTYPE(guint N, double beta, LONG DTYPE *data, gboolean half_window) { \
 \
	long double *factorials_inv2 = g_malloc(35 * sizeof(long double)); \
	factorials_inv2[0] = 1.0L; \
	guint i; \
	__uint128_t current = 1; \
	for(i = 1; i < 35; i++) { \
		current *= i; \
		factorials_inv2[i] = 1.0L / (long double) current / (long double) current; \
	} \
 \
	long double *win = g_malloc(N * sizeof(long double)); \
	long double denom = I0((long double) beta, factorials_inv2); \
	for(i = 0; i < N; i++) \
		win[i] = I0(beta * sqrtl(1.0L - powl(2.0L * i / (N - 1) - 1.0L, 2.0L)), factorials_inv2) / denom; \
 \
	LONG DTYPE *kwin; \
	if(sizeof(LONG DTYPE) < sizeof(long double)) \
		kwin = typecast_longdouble_to_ ## LONG ## DTYPE(win, N); \
	else \
		kwin = (LONG DTYPE *) win; \
 \
	guint start = half_window ? N / 2 : 0; \
	LONG DTYPE *kwin_out; \
	if(half_window) { \
		kwin_out = g_malloc((N - start) * sizeof(LONG DTYPE)); \
		for(i = 0; i < N - start; i++) \
			kwin_out[i] = kwin[start + i]; \
		g_free(kwin); \
	} else \
		kwin_out = kwin; \
 \
	if(data != NULL) { \
		for(i = 0; i < N - start; i++) \
			data[i] *= kwin_out[i]; \
		g_free(kwin_out); \
		return data; \
	} else \
		return kwin_out; \
}


KAISER(, float);
KAISER(, double);
KAISER(long, double);


/*
 * Dolph-Chebyshev window
 */


long double compute_Tn(long double x, guint n) {
	if(x < -1.0L) {
		if(n % 2)
			return -coshl(n * acoshl(-x));
		else
			return coshl(n * acoshl(-x));
	} else if(x <= 1.0L)
		return cosl(n * acosl(x));
	else
		return coshl(n * acoshl(x));
}


long complex double *compute_W0_lagged(guint N, double alpha) {

	guint n = N / 2 + 1;
	long double beta = coshl(acoshl(powl(10.0L, alpha)) / (N - 1));

	long complex double *W0 = g_malloc(n * sizeof(long complex double));
	long double denominator = powl(10.0L, alpha);
	guint k;
	long complex double factor = -PI * I * (N - 1) / N;
	for(k = 0; k < n; k++)
		W0[k] = cexpl(factor * k) * compute_Tn(beta * cosl(PI * k / N), N - 1) / denominator;

	/* If we want an even window length, the Nyquist component must be real. */
	if(!(N % 2))
		W0[n - 1] = cabsl(W0[n - 1]);

	return W0;
}


#define DOLPH_CHEBYSHEV(LONG, DTYPE) \
LONG DTYPE *DolphChebyshev_ ## LONG ## DTYPE(guint N, double alpha, LONG DTYPE *data, gboolean half_window) { \
 \
	guint n = N / 2 + 1; \
	long complex double *W0 = compute_W0_lagged(N, alpha); \
	long double *win = gstlal_irfft_longdouble(W0, n, &N, NULL, 0, NULL, FALSE, 0, NULL, 0, NULL, NULL, 0, NULL, 0, NULL, NULL, NULL, TRUE); \
	long double normalization = win[n - 1]; \
	guint i; \
	for(i = 0; i < N; i++) \
		win[i] /= normalization; \
 \
	LONG DTYPE *dcwin; \
	if(sizeof(LONG DTYPE) < sizeof(long double)) \
		dcwin = typecast_longdouble_to_ ## LONG ## DTYPE(win, N); \
	else \
		dcwin = (LONG DTYPE *) win; \
 \
	guint start = half_window ? N / 2 : 0; \
	if(data != NULL) { \
		for(i = 0; i < N - start; i++) \
			data[i] *= dcwin[start + i]; \
		g_free(dcwin); \
		return data; \
 \
	} else \
		return dcwin + start; \
}


DOLPH_CHEBYSHEV(, float);
DOLPH_CHEBYSHEV(, double);
DOLPH_CHEBYSHEV(long, double)


/*
 * Blackman Window
 */


#define BLACKMAN(LONG, DTYPE, LLL, FF) \
LONG DTYPE *blackman_ ## LONG ## DTYPE(guint N, LONG DTYPE *data, gboolean half_window) { \
 \
	guint i, N_half = half_window ? (N + 1) / 2 : N; \
 \
	LONG DTYPE *win = g_malloc(N_half * sizeof(LONG DTYPE)); \
	for(i = 0; i < N_half; i++) \
		win[i] = 0.42 - 0.5 * cos ## LLL ## FF((LONG DTYPE) ((2 * PI * (i - N_half + N)) / (N - 1))) + 0.08 * cos ## LLL ## FF((LONG DTYPE) ((4 * PI * (i - N_half + N)) / (N - 1))); \
 \
	if(data != NULL) { \
		for(i = 0; i < N_half; i++) \
			data[i] *= win[i]; \
		g_free(win); \
		return data; \
	 } else \
		return win; \
}


BLACKMAN(, float, , f);
BLACKMAN(, double, , );
BLACKMAN(long, double, l, );


/*
 * Hann Window
 */


#define HANN(LONG, DTYPE, LLL, FF) \
LONG DTYPE *hann_ ## LONG ## DTYPE(guint N, LONG DTYPE *data, gboolean half_window) { \
 \
	guint i, N_half = half_window ? (N + 1) / 2 : N; \
 \
	LONG DTYPE a; \
	LONG DTYPE *win = g_malloc(N_half * sizeof(LONG DTYPE)); \
	for(i = 0; i < N_half; i++) { \
		a = sin ## LLL ## FF((LONG DTYPE) ((PI * (i - N_half + N)) / (N - 1))); \
		win[i] = a * a; \
	} \
 \
	if(data != NULL) { \
		for(i = 0; i < N_half; i++) \
			data[i] *= win[i]; \
		g_free(win); \
		return data; \
	 } else \
		return win; \
}


HANN(, float, , f);
HANN(, double, , );
HANN(long, double, l, );


/*
 * A resampler
 */


#define FIR_RESAMPLE(LONG, COMPLEX, DTYPE, LLL, FF) \
LONG COMPLEX DTYPE *fir_resample_ ## LONG ## COMPLEX ## DTYPE(LONG COMPLEX DTYPE *data, guint N_in, guint N_out) { \
 \
	/* Max and min */ \
	guint N_max = N_in > N_out ? N_in : N_out; \
	guint N_min = N_in < N_out ? N_in : N_out; \
 \
	if(N_in < 1 || N_in == N_out) \
		return data; \
 \
	if(N_out < 1) { \
		g_free(data); \
		return NULL; \
	} \
 \
	guint i, j; \
	LONG COMPLEX DTYPE *resampled = g_malloc0(N_out * sizeof(LONG COMPLEX DTYPE)); \
	if(N_in == 1) { \
		/* Then just copy the input N_out times. */ \
		for(i = 0; i < N_out; i++) \
			resampled[i] = *data; \
		g_free(data); \
		return resampled; \
	} \
	if(N_out == 1) { \
		/* Return the average */ \
		*resampled = long_sum_array_ ## LONG ## COMPLEX ## DTYPE(data, N_in, 1); \
		g_free(data); \
		return resampled; \
	} \
	if(N_in == 2) { \
		/* Linear interpolation.  If we've reached this point, we know that N_out >= 3. */ \
		LONG COMPLEX DTYPE diff = data[1] - *data; \
		for(i = 0; i < N_out; i++) \
			resampled[i] = diff * i / (N_out - 1) + *data; \
 \
		g_free(data); \
		return resampled; \
	} \
 \
	/* If we've reached this point, we need to use a sinc function to resample */ \
	/* Are we upsampling or downsampling? */ \
	gboolean up = N_in < N_out ? TRUE : FALSE; \
 \
	/*
	 * Find the least common multiple of input and output lengths to determine the
	 * lenth of the sinc array.
	 */ \
	guint short_length = N_min - 1; \
	guint long_length = N_max - 1; \
	guint64 LCM = (guint64) long_length; \
	while(LCM % short_length) \
		LCM += long_length; \
 \
	/* Number of sinc taps per sample at the higher sample rate */ \
	guint sinc_taps_per_sample = (guint) (LCM / long_length); \
	/* Number of sinc taps per sample at the lower sample rate */ \
	guint long_sinc_taps_per_sample = (guint) (LCM / short_length); \
 \
	guint sinc_length = (N_min < 192 ? N_min : 192) * long_sinc_taps_per_sample; \
	sinc_length -= (sinc_length + 1) % 2; \
	LONG DTYPE *sinc = g_malloc(sinc_length * sizeof(LONG DTYPE)); \
	sinc[sinc_length / 2] = (LONG DTYPE) 1.0L; \
 \
	/* Frequency resolution in units of frequency bins of sinc */ \
	double alpha = (1 + (192 < N_min ? 192 : N_min) / 24.0); \
	/* Low-pass cutoff frequency as a fraction of the sampling frequency of sinc */ \
	LONG DTYPE f_cut = (LONG DTYPE) 0.5L / long_sinc_taps_per_sample - alpha / sinc_length; \
	if(f_cut > 0) { \
		LONG DTYPE two_pi = (LONG DTYPE) (2 * PI); \
		for(i = 1; i < sinc_length / 2 + 1; i++) \
			sinc[sinc_length / 2 + i] = sinc[sinc_length / 2 - i] = sin ## FF ## LLL(two_pi * ((f_cut * i) - (guint) (f_cut * i))) / (two_pi * f_cut * i); \
	} else { \
		for(i = 1; i < sinc_length; i++) \
			sinc[i] = (LONG DTYPE) 1.0L; \
	} \
 \
	/*
	 * Apply a Kaiser window.  Note that the chosen cutoff frequency is below the
	 * lower Nyquist rate just enough to be at the end of the main lobe.
	 */ \
	sinc = kaiser_ ## LONG ## DTYPE(sinc_length, (double) (PI * alpha), sinc, FALSE); \
 \
	/*
	 * Normalize the sinc filter.  Since, in general, not every tap gets used for
	 * each output sample, the normalization has to be done this way:
	 */ \
	guint taps_per_input, taps_per_output; \
	if(!up) { \
		taps_per_input = sinc_taps_per_sample; \
		taps_per_output = long_sinc_taps_per_sample; \
	} else { \
		taps_per_input = long_sinc_taps_per_sample; \
		taps_per_output = sinc_taps_per_sample; \
	} \
	LONG DTYPE normalization; \
	for(i = 0; i < taps_per_input; i++) { \
		normalization = sum_array_ ## LONG ## DTYPE(sinc + i, sinc_length - i, taps_per_input); \
		for(j = i; j < sinc_length; j += taps_per_input) \
			sinc[j] /= normalization; \
	} \
 \
	/* Extend the input array at the ends to prepare for filtering */ \
	guint half_sinc_length = sinc_length / 2; \
	guint N_ext = half_sinc_length / taps_per_input; \
	LONG DTYPE *ext_data = g_malloc((N_in + 2 * N_ext) * sizeof(LONG DTYPE)); \
	LONG DTYPE real_start2 = 2 * creal ## LLL ## FF(data[0]); \
	LONG DTYPE real_end2 = 2 * creal ## LLL ## FF(data[-1]); \
	for(i = 0; i < N_ext; i++) \
		ext_data[i] = real_start2 - data[N_ext - i]; \
	for(i = 0; i < N_in; i++) \
		ext_data[N_ext + i] = data[i]; \
	for(i = 0; i < N_ext; i++) \
		ext_data[N_ext + N_in + i] = real_end2 - data[N_in - i - 2]; \
 \
	g_free(data); \
 \
	/*
	 * Filter.  The center of sinc should line up with the first and last input
	 * at the first and last output, respectively.
	 */ \
	guint sinc_start, data_start, ntaps; \
	for(i = 0; i < N_out; i++) { \
		sinc_start = (half_sinc_length - i * taps_per_output % taps_per_input) % taps_per_input; \
		data_start = (i * taps_per_output - half_sinc_length + N_ext * taps_per_input + taps_per_input - 1) / taps_per_input; \
		ntaps = (sinc_length - sinc_start + taps_per_input - 1) / taps_per_input; \
		for(j = 0; j < ntaps; j++) \
			resampled[i] += sinc[sinc_start + j * taps_per_input] * ext_data[data_start + j]; \
	} \
 \
	g_free(ext_data); \
	g_free(sinc); \
 \
	return resampled; \
}


FIR_RESAMPLE(, , float, , f);
FIR_RESAMPLE(, , double, , );
FIR_RESAMPLE(long, , double, l, );
FIR_RESAMPLE(, complex, float, , f);
FIR_RESAMPLE(, complex, double, , );
FIR_RESAMPLE(long, complex, double, l, );


/*
 * A function to get the frequency-responce of an FIR filter, showing the lobes
 */


#define FREQRESP(LONG, DTYPE) \
LONG complex DTYPE *freqresp_ ## LONG ## DTYPE(LONG DTYPE *filt, guint N, guint delay_samples, guint samples_per_lobe) { \
 \
	if(N == 0) \
		return NULL; \
 \
	/*
	 * Make a longer version of the filter so that we can
	 * get better frequency resolution.
	 */ \
	guint N_prime = samples_per_lobe * N; \
 \
	/* Start with zeros */ \
	LONG DTYPE *filt_prime = g_malloc0(N_prime * sizeof(LONG DTYPE)); \
 \
	/* The beginning and end have filter coefficients */ \
	guint i; \
	for(i = 0; i < N - delay_samples; i++) \
		filt_prime[i] = filt[delay_samples + i]; \
 \
	for(i = 0; i < delay_samples; i++) \
		filt_prime[N - delay_samples + i] = filt[i]; \
 \
	/* Now take an FFT */ \
	return gstlal_rfft_ ## LONG ## DTYPE(filt_prime, N_prime, NULL, 0, NULL, FALSE, 0, NULL, 0, NULL, NULL, NULL, TRUE); \
}


FREQRESP(, float);
FREQRESP(, double);
FREQRESP(long, double);



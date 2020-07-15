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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */


/*
 * ============================================================================
 *
 *				  Preamble
 *
 * ============================================================================
 */


#ifndef __GSTLAL_FIRTOOLS_H__
#define __GSTLAL_FIRTOOLS_H__


#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include <complex.h>
#include <glib.h>
#include <time.h>


G_BEGIN_DECLS


/* pi to long double precision. */
#define PI 3.1415926535897932384626433832795L


float *typecast_float_to_float(float *data, guint N);

float *typecast_double_to_float(double *data, guint N);

float *typecast_longdouble_to_float(long double *data, guint N);

double *typecast_float_to_double(float *data, guint N);

double *typecast_double_to_double(double *data, guint N);

double *typecast_longdouble_to_double(long double *data, guint N);

long double *typecast_float_to_longdouble(float *data, guint N);

long double *typecast_double_to_longdouble(double *data, guint N);

long double *typecast_longdouble_to_longdouble(long double *data, guint N);

complex float *typecast_complexfloat_to_complexfloat(complex float *data, guint N);

complex float *typecast_complexdouble_to_complexfloat(complex double *data, guint N);

complex float *typecast_longcomplexdouble_to_complexfloat(long complex double *data, guint N);

complex double *typecast_complexfloat_to_complexdouble(complex float *data, guint N);

complex double *typecast_complexdouble_to_complexdouble(complex double *data, guint N);

complex double *typecast_longcomplexdouble_to_complexdouble(long complex double *data, guint N);

long complex double *typecast_complexfloat_to_longcomplexdouble(complex float *data, guint N);

long complex double *typecast_complexdouble_to_longcomplexdouble(complex double *data, guint N);

long complex double *typecast_longcomplexdouble_to_longcomplexdouble(long complex double *data, guint N);

float sum_array_float(float *array, guint N, guint cadence);

double sum_array_double(double *array, guint N, guint cadence);

long double sum_array_longdouble(long double *array, guint N, guint cadence);

complex float sum_array_complexfloat(complex float *array, guint N, guint cadence);

complex double sum_array_complexdouble(complex double *array, guint N, guint cadence);

long complex double sum_array_longcomplexdouble(long complex double *array, guint N, guint cadence);

long double long_sum_array_float(float *array, guint N, guint cadence);

long double long_sum_array_double(double *array, guint N, guint cadence);

long double long_sum_array_longdouble(long double *array, guint N, guint cadence);

long complex double long_sum_array_complexfloat(complex float *array, guint N, guint cadence);

long complex double long_sum_array_complexdouble(complex double *array, guint N, guint cadence);

long complex double long_sum_array_longcomplexdouble(long complex double *array, guint N, guint cadence);

long double *array_subset_float(float *array, guint N, guint cadence);

long double *array_subset_double(double *array, guint N, guint cadence);

long double *array_subset_longdouble(long double *array, guint N, guint cadence);

long complex double *array_subset_complexfloat(complex float *array, guint N, guint cadence);

long complex double *array_subset_complexdouble(complex double *array, guint N, guint cadence);

long complex double *array_subset_longcomplexdouble(long complex double *array, guint N, guint cadence);

long double *array_subset_mod_n_float(float *array, guint N, guint n);

long double *array_subset_mod_n_double(double *array, guint N, guint n);

long double *array_subset_mod_n_longdouble(long double *array, guint N, guint n);

long complex double *array_subset_mod_n_complexfloat(complex float *array, guint N, guint n);

long complex double *array_subset_mod_n_complexdouble(complex double *array, guint N, guint n);

long complex double *array_subset_mod_n_longcomplexdouble(long complex double *array, guint N, guint n);

long complex double *array_subset_conj_float(complex float *array, guint start, guint N_in, guint N_total, guint cadence);

long complex double *array_subset_conj_double(complex double *array, guint start, guint N_in, guint N_total, guint cadence);

long complex double *array_subset_conj_longdouble(long complex double *array, guint start, guint N_in, guint N_total, guint cadence);

complex float *conj_array_float(complex float *array, guint N);

complex double *conj_array_double(complex double *array, guint N);

long complex double *conj_array_longdouble(long complex double *array, guint N);

long complex double *pad_zeros_Afloat(float *data, guint N, guint N_conj, long complex double *exp_array, guint M);

long complex double *pad_zeros_Adouble(double *data, guint N, guint N_conj, long complex double *exp_array, guint M);

long complex double *pad_zeros_Alongdouble(long double *data, guint N, guint N_conj, long complex double *exp_array, guint M);

long complex double *pad_zeros_Acomplexfloat(complex float *data, guint N, guint N_conj, long complex double *exp_array, guint M);

long complex double *pad_zeros_Acomplexdouble(complex double *data, guint N, guint N_conj, long complex double *exp_array, guint M);

long complex double *pad_zeros_Alongcomplexdouble(long complex double *data, guint N, guint N_conj, long complex double *exp_array, guint M);

complex double *pad_zeros_B(complex double *b_n, guint N, guint N_out, guint M);

long complex double *pad_zeros_Blong(long complex double *b_n, guint N, guint N_out, guint M);

guint *find_prime_factors(guint N, guint *num_factors);

guint *find_prime_factors_M(guint M_min, guint *M, guint *num_factors);

long complex double *find_exp_array(guint N, gboolean inverse);

long complex double *find_exp_array2(guint N, gboolean inverse);

complex float *gstlal_dft_float(complex float *td_data, guint N, long complex double *exp_array, gboolean inverse, long complex double *fd_data);

complex double *gstlal_dft_double(complex double *td_data, guint N, long complex double *exp_array, gboolean inverse, long complex double *fd_data);

long complex double *gstlal_dft_longdouble(long complex double *td_data, guint N, long complex double *exp_array, gboolean inverse, long complex double *fd_data);

complex float *gstlal_idft_float(complex float *fd_data, guint N, gboolean normalize);

complex double *gstlal_idft_double(complex double *fd_data, guint N, gboolean normalize);

long complex double *gstlal_idft_longdouble(long complex double *fd_data, guint N, gboolean normalize);

complex float *gstlal_rdft_float(float *td_data, guint N, long complex double *exp_array, gboolean return_full, long double complex *fd_data);

complex double *gstlal_rdft_double(double *td_data, guint N, long complex double *exp_array, gboolean return_full, long double complex *fd_data);

long complex double *gstlal_rdft_longdouble(long double *td_data, guint N, long complex double *exp_array, gboolean return_full, long double complex *fd_data);

float *gstlal_irdft_float(complex float *fd_data, guint N_in, guint *N, long complex double *exp_array, gboolean normalize, long double *td_data);

double *gstlal_irdft_double(complex double *fd_data, guint N_in, guint *N, long complex double *exp_array, gboolean normalize, long double *td_data);

long double *gstlal_irdft_longdouble(long complex double *fd_data, guint N_in, guint *N, long complex double *exp_array, gboolean normalize, long double *td_data);

complex float *gstlal_fft_float(complex float *td_data, guint N, guint *prime_factors, guint num_factors, long complex double *exp_array, gboolean inverse, guint M, guint *M_prime_factors, guint M_num_factors, long complex double *M_exp_array2, long complex double *M_exp_array, long complex double *fd_data, gboolean free_input);

complex double *gstlal_fft_double(complex double *td_data, guint N, guint *prime_factors, guint num_factors, long complex double *exp_array, gboolean inverse, guint M, guint *M_prime_factors, guint M_num_factors, long complex double *M_exp_array2, long complex double *M_exp_array, long complex double *fd_data, gboolean free_input);

long complex double *gstlal_fft_longdouble(long complex double *td_data, guint N, guint *prime_factors, guint num_factors, long complex double *exp_array, gboolean inverse, guint M, guint *M_prime_factors, guint M_num_factors, long complex double *M_exp_array2, long complex double *M_exp_array, long complex double *fd_data, gboolean free_input);

complex float *gstlal_ifft_float(complex float *fd_data, guint N, gboolean normalize, gboolean free_input);

complex double *gstlal_ifft_double(complex double *fd_data, guint N, gboolean normalize, gboolean free_input);

long complex double *gstlal_ifft_longdouble(long complex double *fd_data, guint N, gboolean normalize, gboolean free_input);

complex float *gstlal_rfft_float(float *td_data, guint N, guint *prime_factors, guint num_factors, long complex double *exp_array, gboolean return_full, guint M, guint *M_prime_factors, guint M_num_factors, long complex double *M_exp_array2, long complex double *M_exp_array, long complex double *fd_data, gboolean free_input);

complex double *gstlal_rfft_double(double *td_data, guint N, guint *prime_factors, guint num_factors, long complex double *exp_array, gboolean return_full, guint M, guint *M_prime_factors, guint M_num_factors, long complex double *M_exp_array2, long complex double *M_exp_array, long complex double *fd_data, gboolean free_input);

long complex double *gstlal_rfft_longdouble(long double *td_data, guint N, guint *prime_factors, guint num_factors, long complex double *exp_array, gboolean return_full, guint M, guint *M_prime_factors, guint M_num_factors, long complex double *M_exp_array2, long complex double *M_exp_array, long complex double *fd_data, gboolean free_input);

float *gstlal_irfft_float(complex float *fd_data, guint N_in, guint *N, guint *prime_factors, guint num_factors, long complex double *exp_array, gboolean normalize, guint M_fft, guint *M_fft_prime_factors, guint M_fft_num_factors, long complex double *M_fft_exp_array2, long complex double *M_fft_exp_array, guint M_irfft, guint *M_irfft_prime_factors, guint M_irfft_num_factors, long complex double *M_irfft_exp_array2, long complex double *M_irfft_exp_array, long double *td_data, gboolean free_input);

double *gstlal_irfft_double(complex double *fd_data, guint N_in, guint *N, guint *prime_factors, guint num_factors, long complex double *exp_array, gboolean normalize, guint M_fft, guint *M_fft_prime_factors, guint M_fft_num_factors, long complex double *M_fft_exp_array2, long complex double *M_fft_exp_array, guint M_irfft, guint *M_irfft_prime_factors, guint M_irfft_num_factors, long complex double *M_irfft_exp_array2, long complex double *M_irfft_exp_array, long double *td_data, gboolean free_input);

long double *gstlal_irfft_longdouble(long complex double *fd_data, guint N_in, guint *N, guint *prime_factors, guint num_factors, long complex double *exp_array, gboolean normalize, guint M_fft, guint *M_fft_prime_factors, guint M_fft_num_factors, long complex double *M_fft_exp_array2, long complex double *M_fft_exp_array, guint M_irfft, guint *M_irfft_prime_factors, guint M_irfft_num_factors, long complex double *M_irfft_exp_array2, long complex double *M_irfft_exp_array, long double *td_data, gboolean free_input);

complex float *gstlal_prime_fft_float(complex float *td_data, guint N, gboolean inverse, long complex double *exp_array2, guint M, guint *prime_factors, guint num_factors, long complex double *exp_array, long complex double *fd_data);

complex double *gstlal_prime_fft_double(complex double *td_data, guint N, gboolean inverse, long complex double *exp_array2, guint M, guint *prime_factors, guint num_factors, long complex double *exp_array, long complex double *fd_data);

long complex double *gstlal_prime_fft_longdouble(long complex double *td_data, guint N, gboolean inverse, long complex double *exp_array2, guint M, guint *prime_factors, guint num_factors, long complex double *exp_array, long complex double *fd_data);

complex float *gstlal_prime_rfft_float(float *td_data, guint N, gboolean return_full, long complex double *exp_array2, guint M, guint *prime_factors, guint num_factors, long complex double *exp_array, long complex double *fd_data);

complex double *gstlal_prime_rfft_double(double *td_data, guint N, gboolean return_full, long complex double *exp_array2, guint M, guint *prime_factors, guint num_factors, long complex double *exp_array, long complex double *fd_data);

long complex double *gstlal_prime_rfft_longdouble(long double *td_data, guint N, gboolean return_full, long complex double *exp_array2, guint M, guint *prime_factors, guint num_factors, long complex double *exp_array, long complex double *fd_data);

float *gstlal_prime_irfft_float(complex float *fd_data, guint N_in, guint *N, gboolean normalize, long complex double *exp_array2, guint M, guint *prime_factors, guint num_factors, long complex double *exp_array, long double *td_data);

double *gstlal_prime_irfft_double(complex double *fd_data, guint N_in, guint *N, gboolean normalize, long complex double *exp_array2, guint M, guint *prime_factors, guint num_factors, long complex double *exp_array, long double *td_data);

long double *gstlal_prime_irfft_longdouble(long complex double *fd_data, guint N_in, guint *N, gboolean normalize, long complex double *exp_array2, guint M, guint *prime_factors, guint num_factors, long complex double *exp_array, long double *td_data);

double *rand_array(guint N);

long double *rand_arraylong(guint N);

complex double *rand_arraycomplex(guint N);

long complex double *rand_arraylongcomplex(guint N);

void compare_speed_fft(guint N, guint iterations);

void compare_speed_rfft(guint N, guint iterations);

void compare_speed_irfft(guint N, guint iterations);

void fft_test_inverse(guint N_start, guint N_end, guint cadence);

void rfft_test_inverse(guint N_start, guint N_end, guint cadence);

void compare_fft(guint N_start, guint N_end, guint cadence);

void compare_rfft(guint N_start, guint N_end, guint cadence);

float *mat_times_vec_float(float *mat, guint N, float *vec, guint n);

double *mat_times_vec_double(double *mat, guint N, double *vec, guint n);

long double *mat_times_vec_longdouble(long double *mat, guint N, long double *vec, guint n);

complex float *mat_times_vec_complexfloat(complex float *mat, guint N, complex float *vec, guint n);

complex double *mat_times_vec_complexdouble(complex double *mat, guint N, complex double *vec, guint n);

long complex double *mat_times_vec_longcomplexdouble(long complex double *mat, guint N, long complex double *vec, guint n);

float *dpss_float(guint N, double alpha, double compute_time, float *data, gboolean half_window, gboolean free_warehouse);

double *dpss_double(guint N, double alpha, double compute_time, double *data, gboolean half_window, gboolean free_warehouse);

long double *dpss_longdouble(guint N, double alpha, double compute_time, long double *data, gboolean half_window, gboolean free_warehouse);

long double I0(long double x, long double *factorials_inv2);

float *kaiser_float(guint N, double beta, float *data, gboolean half_window);

double *kaiser_double(guint N, double beta, double *data, gboolean half_window);

long double *kaiser_longdouble(guint N, double beta, long double *data, gboolean half_window);

long double compute_Tn(long double x, guint n);

long complex double *compute_W0_lagged(guint N, double alpha);

float *DolphChebyshev_float(guint N, double alpha, float *data, gboolean half_window);

double *DolphChebyshev_double(guint N, double alpha, double *data, gboolean half_window);

long double *DolphChebyshev_longdouble(guint N, double alpha, long double *data, gboolean half_window);

float *blackman_float(guint N, float *data, gboolean half_window);

double *blackman_double(guint N, double *data, gboolean half_window);

long double *blackman_longdouble(guint N, long double *data, gboolean half_window);

float *hann_float(guint N, float *data, gboolean half_window);

double *hann_double(guint N, double *data, gboolean half_window);

long double *hann_longdouble(guint N, long double *data, gboolean half_window);

float *fir_resample_float(float *data, guint N_in, guint N_out);

double *fir_resample_double(double *data, guint N_in, guint N_out);

long double *fir_resample_longdouble(long double *data, guint N_in, guint N_out);

complex float *fir_resample_complexfloat(complex float *data, guint N_in, guint N_out);

complex double *fir_resample_complexdouble(complex double *data, guint N_in, guint N_out);

long complex double *fir_resample_longcomplexdouble(long complex double *data, guint N_in, guint N_out);

complex float *freqresp_float(float *filt, guint N, guint delay_samples, guint samples_per_lobe);

complex double *freqresp_double(double *filt, guint N, guint delay_samples, guint samples_per_lobe);

long complex double *freqresp_longdouble(long double *filt, guint N, guint delay_samples, guint samples_per_lobe);


G_END_DECLS
#endif	/* __GSTLAL_FIRTOOLS_H__ */


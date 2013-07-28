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


/*
 * stuff from GSL
 */


#include <gsl/gsl_complex.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>


/*
 * ============================================================================
 *
 *                                 Prototypes
 *
 * ============================================================================
 */


unsigned gstlal_autocorrelation_chi2_autocorrelation_channels(const gsl_matrix_complex *);
unsigned gstlal_autocorrelation_chi2_autocorrelation_length(const gsl_matrix_complex *);
gsl_vector *gstlal_autocorrelation_chi2_compute_norms(const gsl_matrix_complex *, const gsl_matrix_int *);
unsigned gstlal_autocorrelation_chi2(double *, const complex double *, unsigned, int, double, const gsl_matrix_complex *, const gsl_matrix_int *, const gsl_vector *);
unsigned gstlal_autocorrelation_chi2_float(float *, const float complex *, unsigned, int, double, const gsl_matrix_complex *, const gsl_matrix_int *, const gsl_vector *);

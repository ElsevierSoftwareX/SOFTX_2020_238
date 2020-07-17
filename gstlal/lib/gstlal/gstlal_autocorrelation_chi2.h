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
 * Our own stuff
 */


#include <gstlal/gstlal_peakfinder.h>


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

/*
 * Bank veto functions
 */

gsl_vector *gstlal_bankcorrelation_chi2_compute_norms(const gsl_matrix_complex *bankcorrelation_matrix);
unsigned gstlal_bankcorrelation_chi2_from_peak(double *, struct gstlal_peak_state *, const gsl_matrix_complex *, const gsl_vector *, const complex double *, guint);
unsigned gstlal_bankcorrelation_chi2_from_peak_float(float *, struct gstlal_peak_state *, const gsl_matrix_complex *, const gsl_vector *, const complex float *, guint);

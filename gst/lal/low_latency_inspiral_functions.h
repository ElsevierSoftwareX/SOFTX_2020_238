/*
 * Code to generate an inspiral bank and svd for portions of tempates at
 * the requested sample rate.  This is temporary code and will be moved to
 * the LSC Algorithm Library.
 *
 * Copyright (C) 2008  Kipp Cannon, Chad Hanna
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#include <stdio.h>
#include <glib.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <lal/FindChirp.h>
#include <lal/LALDatatypes.h>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>


int generate_bank_and_svd(
                      gsl_matrix **U,
                      gsl_vector **S, 
                      gsl_matrix **V,
                      gsl_vector **chifacs,
		      gsl_matrix_complex **A,
                      const char *xml_bank_filename,
                      REAL8FrequencySeries *psd,
                      int base_sample_rate,
                      int down_samp_fac, 
                      double t_start,
                      double t_end, 
                      double tmax, 
                      double tolerance,
		      GMutex *fftw_lock,
                      int verbose);

int generate_bank(
                      gsl_matrix **U,
                      gsl_vector **chifacs,
		      gsl_matrix_complex **A,
                      const char *xml_bank_filename,
                      REAL8FrequencySeries *psd,
                      int base_sample_rate,
                      int down_samp_fac, 
                      double t_start,
                      double t_end, 
                      double tmax, 
		      GMutex *fftw_lock,
                      int verbose);

int compute_time_frequency_boundaries_from_bank(char * bank_name,
                                                double min_subtemplate_samples,
                                                double base_sample_rate,
                                                double f_lower,
                                                gsl_vector **sample_rates,
                                                gsl_vector **start_times,
                                                gsl_vector **stop_times,
                                                int verbose);

double normalize_template(double M, double ts, double duration,
                                int fsamp);

int trim_matrix(gsl_matrix **U, gsl_matrix **V, gsl_vector **S,
                        double tolerance);

int not_gsl_matrix_chop(gsl_matrix **M, size_t m, size_t n);

int not_gsl_vector_chop(gsl_vector **V, size_t m);

int not_gsl_matrix_transpose(gsl_matrix **m);

double gstlal_spa_chirp_time(REAL8 m,
                             REAL8 eta,
                             REAL8 fLower,
                             LALPNOrder order);

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
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
/*#include "gstlal.h"*/
#include <lal/FindChirp.h>


int create_template_from_sngl_inspiral(
                       InspiralTemplate *bankHead,
                       gsl_matrix *U, 
                       gsl_vector *chifacs,
                       double duration, 
                       int fsamp,
                       int downsampfac, 
                       double t_start, 
                       double t_end,
                       int U_column,
                       FindChirpFilterInput *fcFilterInput,
                       FindChirpTmpltParams *fcTmpltParams,
                       REAL8TimeSeries *template,
                       COMPLEX16FrequencySeries *fft_template,
                       REAL8FFTPlan *fwdplan,
                       REAL8FFTPlan *revplan,
                       REAL8FrequencySeries *psd
                       );

/*int generate_bank_svd(gsl_matrix **U, gsl_vector **S, gsl_matrix **V,
                           gsl_vector **chifacs,
                           char * bank_name, int base_sample_rate,
                           int down_samp_fac, double t_start,
                           double t_end, double tmax, double tolerance,
                           int verbose);*/

int generate_bank_svd(
                      gsl_matrix **U,
                      gsl_vector **S, 
                      gsl_matrix **V,
                      gsl_vector **chifacs,
                      char * bank_name, 
                      int base_sample_rate,
                      int down_samp_fac, 
                      double t_start,
                      double t_end, 
                      double tmax, 
                      double tolerance,
                      int verbose);

double normalize_template(double M, double ts, double duration,
                                int fsamp);

int trim_matrix(gsl_matrix **U, gsl_matrix **V, gsl_vector **S,
                        double tolerance);

int not_gsl_matrix_chop(gsl_matrix **M, size_t m, size_t n);

int not_gsl_vector_chop(gsl_vector **V, size_t m);

void not_gsl_matrix_transpose(gsl_matrix **m);


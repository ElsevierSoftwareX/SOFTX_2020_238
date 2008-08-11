#include <stdio.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>


int generate_bank_svd(gsl_matrix **U, gsl_vector **S, gsl_matrix **V,
                gsl_vector **chifacs,
		double chirp_mass_start, int base_sample_rate,
		int down_samp_fac, unsigned numtemps, double t_start,
		double t_end, double tmax, double tolerance, int verbose);

double normalize_template(double M, double ts, double duration,
                                int fsamp);

int trim_matrix(gsl_matrix **U, gsl_matrix **V, gsl_vector **S,
                        double tolerance);

int not_gsl_matrix_chop(gsl_matrix **M, size_t m, size_t n);

int not_gsl_vector_chop(gsl_vector **V, size_t m);

void not_gsl_matrix_transpose(gsl_matrix **m);


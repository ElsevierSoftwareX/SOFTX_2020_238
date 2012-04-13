#include <cheby_interp.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
/*#include <spa.c>*/
/*
 * Data structure methods
 */


int free_waveform_interp_objects(struct twod_waveform_interpolant_array * interps) {
	free(interps->interp);
	free(interps);
	return 0;
	}


struct twod_waveform_interpolant_array * new_waveform_interpolant_array(int size) {
	struct twod_waveform_interpolant_array * output = (struct twod_waveform_interpolant_array *) calloc(1, sizeof(struct twod_waveform_interpolant_array));
	output-> size = size;
	output->interp = (struct twod_waveform_interpolant *) calloc(size, sizeof(struct twod_waveform_interpolant));
	return output;
	}


/*
 * Formula (3) in http://arxiv.org/pdf/1108.5618v1.pdf
 */ 

static gsl_complex projection_coefficient(gsl_vector_complex *svd_basis, gsl_vector_complex *spa_waveform){
	/*project svd basis onto SPA's to get coefficients*/
	gsl_complex M;
	/* Note that svd_basis should have 0 imaginary part */
	gsl_blas_zdotu(svd_basis, spa_waveform, &M);
	return M;
}


/*
 * Formula (4) in http://arxiv.org/pdf/1108.5618v1.pdf
 */

static double chebyshev_node(int j, int J_max) {
	return cos(M_PI * (2.*j + 1.) / J_max / 2.);
}

/* 
 * Formula (5) in http://arxiv.org/pdf/1108.5618v1.pdf
 */

static double onedCheby(double x, int J, int J_max) {
	double w;

	if (J == 0)
		w = sqrt(2. * (J_max + 1) / 2.);
	else
		w = sqrt(1. * (J_max + 1) / 2.);
		
	return (pow(x - sqrt(x*x - 1.0), J) + pow(x + sqrt(x*x - 1.0), J)) / w;
}

static double twodCheby(double x, int K, int K_max, double y, int L, int L_max) {
	return onedCheby(x, K, K_max) * onedCheby(y, L, L_max);
}


/* 
 * Formula (7) in http://arxiv.org/pdf/1108.5618v1.pdf
 */


/* you do this for every mu */
static gsl_matrix_complex * compute_C_KL(gsl_vector *x_k, gsl_vector *y_l, gsl_matrix_complex *M) {
	int K, L, k, l;
	gsl_complex out;

	int k_max = x_k->size;
	int l_max = y_l->size;
	gsl_matrix_complex *C_KL = gsl_matrix_complex_calloc(k_max, l_max);
	gsl_complex tmp;

	for (K = 0; K < k_max; K++) {
		for (L = 0; L < l_max; L++) {
			GSL_SET_COMPLEX(&out, 0, 0);
			for (k = 0; k < k_max; k++) {
				for (l = 0; l < l_max; l++) {
					tmp = gsl_complex_mul_real(gsl_matrix_complex_get(M, k, l), twodCheby(gsl_vector_get(x_k, k), K, k_max, gsl_vector_get(y_l, l), L, l_max));
					out = gsl_complex_add(out, tmp);
				}
			}
		gsl_matrix_complex_set(C_KL, K, L, out);
		}
	}

	return C_KL;
}


/* 
 * Formula (8) in http://arxiv.org/pdf/1108.5618v1.pdf
 */

static gsl_complex compute_M_xy(gsl_matrix_complex *C_KL, double x, double y) {
	int K_max, L_max;
	int K, L;
	gsl_complex M; 
	gsl_complex tmp;
	
	K_max = C_KL->size1;
	L_max = C_KL->size2;

	GSL_SET_COMPLEX(&M, 0, 0);

	for (K = 0; K < K_max; K++) {
		for (L = 0; L < L_max; L++) {
			/* FIXME, is this correct?? */
			tmp =gsl_complex_mul_real(gsl_matrix_complex_get(C_KL, K, L),twodCheby(x, K, K_max, y, L, L_max));
			M = gsl_complex_add(M, tmp);
			
		}

	}	

	return M;
}

/*
 * generic cheby utilities
 */

static double map_coordinate_to_cheby(double c_min, double c_max, double c) {
	return 2. * ( c - c_min) / (c_max - c_min) - 1.;
}

/*
 * High level functions
 */

/* FIXME use a better name */
static gsl_vector_complex *interpolate_waveform_from_mchirp_and_eta(struct twod_waveform_interpolant_array *interps, double mchirp, double eta) { 
	int i;
	gsl_complex M;
	double deltaF, x, y;
	struct twod_waveform_interpolant *interp = interps->interp;
	gsl_vector_complex *h_f = gsl_vector_complex_calloc(interp->svd_basis->size);
	
	for (i = 0; i < interps->size; i++, interp++) {
		x = map_coordinate_to_cheby(interp->p1_min, interp->p1_max, mchirp);
		y = map_coordinate_to_cheby(interp->p2_min, interp->p2_max, eta);
		M = compute_M_xy(interp->C_KL, x, y);
		/* don't do this, we need an add and scale function so as to not destroy svd_basis */
		/* gsl_vector_scale(interp->svd_basis, M); */
		gsl_blas_zaxpy(M, h_f, interp->svd_basis); 
		}
	free_waveform_interp_objects(interps);

	return h_f;
}


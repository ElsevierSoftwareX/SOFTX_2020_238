#include <interp.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <complex.h>

/*
 * Data structure methods
 */


int free_waveform_interp_objects(struct waveform_interpolants * interps) {
	free(interps->interp);
	free(interps);
	return 0;
	}


struct waveform_interpolants * new_waveform_interpolants(int size) {
	struct waveform_interpolants * output = (struct waveform_interpolants *) calloc(1, sizeof(waveform_interpolants));
	output-> size = size;
	output->interp = (struct waveform_interpolant *) calloc(size, sizeof(struct waveform_interpolant));
	return output;
	}


/*
 * Formula (3) in http://arxiv.org/pdf/1108.5618v1.pdf
 */ 

gsl_complex projection_coefficient(gsl_vector_complex *svd_basis, gsl_vector_complex *spa_waveform){
	/*project svd basis onto SPA's to get coefficients*/
	gsl_complex M; //Should equal double complex and can be safely cast
	/* Note that svd_basis should have 0 imaginary part */
	gsl_blas_zdotu(svd_basis, spa_waveform, &M);
	return M
}


/*
 * Formula (4) in http://arxiv.org/pdf/1108.5618v1.pdf
 */

double chebyshev_node(int j, int J_max) {
	return cos(M_PI * (2.*j + 1.) / J_max / 2.);
}

/* 
 * Formula (5) in http://arxiv.org/pdf/1108.5618v1.pdf
 */

double 1dCheby(double x, int J, int J_max) {
	double w;

	if (J == 0)
		w = sqrt(2. * (J_max + 1) / 2.);
	else
		w = sqrt(1. * (J_max + 1) / 2.);
		
	return (pow(x - sqrt(x*x - 1.0), J) + pow(x + sqrt(x*x - 1.0), J)) / w;

double 2dCheby(double x, int K, int K_max, double y, int L, int L_max) {
	return 1dCheby(x, K, K_max) * 1dCheby(y, L, L_max);
}


/* 
 * Formula (7) in http://arxiv.org/pdf/1108.5618v1.pdf
 */


/* you do this for every mu */
gsl_matrix_complex * compute_C_KL(gsl_vector *x_k, gsl_vector *y_l, gsl_matrix_complex *M) {
	int K, L, k, l;
	double complex out = 0.0;
	k_max = x_k->size;
	l_max = y_l->size;
	gsl_matrix_complex *C_KL = gsl_matrix_complex_calloc(k_max, l_max);

	for (K = 0; K < k_max; K++) {
		for (L = 0; L < l_max; L++) {
			out = 0.;
			for (k = 0; k < k_max; k++) {
				for (l = 0; l < l_max; l++) {
					out += (double complex) (2dCheby(gsl_vector_get(x_k, k), K, k_max, gsl_vector_get(y_l, l), L, l_max) + 0.*I) * (double complex) gsl_matrix_complex_get(M, k, l);
				}
			}
		gsl_matrix_set(C_KL, K, L, (gsl_complex) out);
		}
	}

	return C_KL;
}


/* 
 * Formula (8) in http://arxiv.org/pdf/1108.5618v1.pdf
 */

gsl_complex compute_M_xy(gsl_matrix_complex *C_KL, double x, double y) {
	int K, L;
	K_max = C_KL->size1;
	L_max = C_KL->size2;
	double complex M = 0. + 0.I;
	for (K = 0; K < K_max; K++) {
		for (L = 0; L < L_max; L++) {
			//FIXME, is this correct??
			M += (double complex) gsl_matrix_complex_get(C_KL, K, L) * (double complex) (2dCheby(x, K, K_max, y, L, L_max) + 0.I);
		}
	}	

	return (gsl_complex) M;
}

/*
 * generic cheby utilities
 */

double map_coordinate_to_cheby(double c_min, double c_max, double c) {
	return 2. * ( c - c_min) / (c_max - c_min) - 1.;
}

/*
 * High level functions
 */

/* FIXME use a better name */
gsl_vector *interpolate_waveform_from_mchirp_and_eta(struct 2d_waveform_interpolant_array *interps, double mchirp, double eta) { /* FIXME: the interpolator never uses mc and eta directly; we have to map an (mc,eta) to a point on the intervals ( [-1,1], [-1,1] ) */

	int i;
	double complex M;
	double deltaF, x, y;
	gsl_vector *h_f = gsl_vector_calloc(interps[0]->svd_basis->size);
	struct 2d_waveform_interpolant *interp = interps->interp;
	
	for (i = 0; i < interps->size; i++, interp++) {
		x = map_coordinate_to_cheby(interp->p1_min, interp->p1_max, mchirp);
		y = map_coordinate_to_cheby(interp->p2_min, interp->p2_max, eta);
		M = compute_M_xy(interp->C_KL, x, y)
		// don't do this, we need an add and scale function so as to not destroy svd_basis
		//gsl_vector_scale(interp->svd_basis, M);
		gsl_vector_add(h_f, interp->svd_basis);		
		gsl_blas_zaxpy (M, h_f, interp->svd_basis) // requires M be a const gsl_complex
		}
	free_waveform_interp_objects(* interps);

	return h_f
}


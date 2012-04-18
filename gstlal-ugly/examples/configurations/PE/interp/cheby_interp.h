#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

struct twod_waveform_interpolant {

	gsl_vector_view svd_basis; /* Imaginary part must be zero */

	/* See http://arxiv.org/pdf/1108.5618v1.pdf  This represents the C
 	 * matrix of formula (8) without mu.  Note that you specify a separate waveform
	 * interpolant object for each mu 
	 */

	gsl_matrix_complex *C_KL;

		
};
	
struct twod_waveform_interpolant_array {
	struct twod_waveform_interpolant *interp;
	int size;
	double param1_min;
	double param1_max;
	double param2_min;
	double param2_max;
	
};

int free_waveform_interp_objects(struct twod_waveform_interpolant_array *);

struct twod_waveform_interpolant_array * new_waveform_interpolant_array_from_svd_bank(gsl_matrix *svd_bank, param1_min, param2_min, param1_max, param2_max);

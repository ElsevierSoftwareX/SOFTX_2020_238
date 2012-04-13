#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

struct twod_waveform_interpolant {

	gsl_vector_complex *svd_basis; /* Imaginary part must be zero */

	/* See http://arxiv.org/pdf/1108.5618v1.pdf  This represents the C
 	 * matrix of formula (8) without mu.  Note that you specify a separate waveform
	 * interpolant object for each mu 
	 */

	gsl_matrix_complex *C_KL;

	double p1_min;
	double p1_max;
	double p2_min;
	double p2_max;
		
};
	
struct twod_waveform_interpolant_array {
	struct twod_waveform_interpolant *interp;
	int size;
	
};

int free_waveform_interp_objects(struct twod_waveform_interpolant_array *);

struct twod_waveform_interpolant_array * new_twod_waveform_interpolant_array(int size);



struct 2d_waveform_interpolant {
	gsl_vector_complex *svd_basis;
	/* See http://arxiv.org/pdf/1108.5618v1.pdf  This represents the C
 	 * matrix of formula (8) without mu.  Note that you specify a separate waveform
	 * interpolant object for each mu 
	 */
	gsl_matrix *C_KL;

	double p1_min;
	double p1_max;
	double p2_min;
	double p2_max;
		
	}
	
struct 2d_waveform_interpolant_array {
	struct waveform_interpolant *interp;
	int size;
	
}





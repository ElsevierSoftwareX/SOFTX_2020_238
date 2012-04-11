struct waveform_interpolant {
	gsl_vector *svd_basis;
	/* See http://arxiv.org/pdf/1108.5618v1.pdf  This represents the C
 *  	 * matrix of formula (8) without mu.  Note that you specify a separate waveform
 *  	 	 * interpolant object for each mu 
 *  	 	 	 */
	gsl_matrix *Ckl;
	
		
	}
	
struct waveform_interpolants {
	struct waveform_interpolant *interp;
	int size;
	
	}

struct spa_waves {
	struct spa_mc *waveform_mc;

	}

struct spa_mc {
	struct spa_eta *waveform_eta;
	}
struct spa_eta {
	gsl_vector *spa_waveform;
	}




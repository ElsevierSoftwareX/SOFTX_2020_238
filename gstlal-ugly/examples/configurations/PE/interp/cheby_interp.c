#include <interp.h>



int free_waveform_interp_objects(struct waveform_interpolants * interps) {
	free(interps->interp);
	free(interps);
	return 0;
	}

double 2dCheby(double x, int k, int K_max, double y, int l, int L_max) {
	double ux;
	double uy;
	int thisk;
	int thisl;
	int w;

	for (thisk=0; thisk < (k/2+1); thisk++):
		ux += (x**2.-1.)**(thisk) * x**(k-2*thisk) * factorial(k)/factorial(2*thisk)/factorial(k-2*thisk)	
	for (thisl=0; thisl < (l/2+1); thisl++):
		uy += (y**2.-1.)**(thisl) * y**(l-2*thisl) * factorial(l)/factorial(2*thisl)/factorial(l-2*thisl)

	w = 1.
	if (k){ //not sure what the correct c-syntax is for this
		w *= K_max/2. }
	else{
		w *= K_max}
	if (l){
		w *= L_max/2.}
	else{
		w *= L_max}

	return ux*uy/w**.5;	
}

gsl_complex M_from_interp(struct waveform_interpolant *interp, double mchirp, double eta) {
	int K, L;
	gsl_complex M;

	gsl_matrix *Ckl = interp->Ckl;
	for (K = 0; K < Ckl->size1; K++) {
		for (L = 0; L < Ckl->size2; L++) {
			
			M += gsl_matrix_get(*Ckl,K,L)*2dCheby(mchirp, K, Ckl->size1, eta, L, Ckl->size2)
		}
	}
	
	return M;	
}

struct waveform_interpolants * new_waveform_interpolants(int size) {
	struct waveform_interpolants * output = (struct waveform_interpolants *) calloc(1, sizeof(waveform_interpolants));
	output-> size = size;
	output->interp = (struct waveform_interpolant *) calloc(size, sizeof(struct waveform_interpolant));
	return output;
	}

struct spa_waves * make_spa_waveforms(M,N){
	struct spa_waves* = (struct spa_waves *) calloc(1, sizeof(spa_waves));
	spa_waves -> waveform_mc = (struct spa_mc *) calloc(M, sizeof(struct spa_mc));
	spa_waves -> waveform_mc -> waveform_eta = (struct spa_eta *) calloc(N, sizeof(struct spa_eta));
	return spa_waves;
	}

double  colocation_coefficients(gsl_vector svd_basis, gsl_vector spa_waveform){
		
	/*project svd basis onto SPA's to get coefficients*/
	double M = dot(svd_basis,spa_waveform); //FIXME
	return M
}


struct waveform_interpolants * get_basis_vectors_and_projection_matrices(int size, int M, int N){ 	
	int idx, jdy, k, l, K, L;
	double x;
	interps = new_waveform_interpolants(size);

	x_cheby = gsl_vector_calloc(N);	
	y_cheby = gsl_vector_calloc(M);
	
	mc_colocation =  gsl_vector_calloc(M);	
	eta_colocation =  gsl_vector_calloc(N);

	for(idx=0; idx < M; idx++){
		gsl_vector_set(* x_cheby, idx, cos(pi*(2*idx+1)/N1/2));
	}
	for(jdy=0; jdy < N; jdy++){
		gsl_vector_set(* y_cheby, jdy, cos(pi*(2*jdy+1)/N1/2));
	}

	for(idx=0; idx < M; idx++){
		gsl_vector_set(* mc_colocation, idx, (mc_range[-1]-mc_range[0])*(x_cheby[idx]+1.)/2. + mc_range[0]);
	}
	gsl_vector_reverse (mc_colocation) ;

	for(jdy=0; jdy < N; jdy++){
		gsl_vector_set(* eta_colocation, jdy, (eta_range[-1]-eta_range[0])*(y_cheby[jdy]+1.)/2. + eta_range[0]);
	}
	gsl_vector_reverse(eta_colocation);

	/* FIXME: compute SPA waveforms at colocation points (mc_colocation, eta_colocation): store in some structure so waveforms can be called*/

	for (i = 0; i < size; i++) {
		interps[i]->svd_basis = svd_basis_vectors[i];
		interps[i]->Ckl = gsl_matrix_alloc(M, N);
		for  (K = 0; K < M; K++){
			for (L =0; L < N; L++){
				for (k = 0; k < M; k++){
					for(l = 0; l < N; l++){
					
						x += 2dCheby(x_cheby[k], K, M, y_cheby[l], L, N) * get_coefficients_on_mc_and_eta_colocations(interps[i]->svd_basis, spa_waves->waveform_mc[k]->waveform_eta[l]->spa_waveform);												
					}
				}			
			gsl_matrix_set(* interps[i]->Ckl, K, L, x);
			x=0;
  			}
		}

	}

}

/* FIXME use a better name */
gsl_vector interpolate_waveform(struct waveform_interpolants *interps, double mchirp, double eta) { /* FIXME: the interpolator never uses mc and eta directly; we have to map an (mc,eta) to a point on the intervals ( [-1,1], [-1,1] ) */

	int i;
	double complex M;
	double deltaF;

	h_f = gsl_vector_calloc(sizeof interps[0]->svd_basis);

	gsl_vector svd_basis;
	gsl_vector output = gsl_matrix_calloc(correct size);
	for (i = 0; i < interps->size; i++) {
		M = M_from_interp(interps[i], mchirp, eta);
		svd_basis = interps[i]->svd_basis;
		gsl_vector_scale(svd_basis, M);
		gsl_vector_add(* h_f, * svd_basis);		
		 
		}
	free_waveform_interp_objects(* interps);

	return h_f
}


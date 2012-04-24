/* Use old lal complex structures for now */
#define LAL_USE_OLD_COMPLEX_STRUCTS

#include <cheby_interp.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_spline.h>
/* LAL includes */
#include <lal/LALDatatypes.h>
#include <lal/Units.h>
#include <lal/TimeFreqFFT.h>
#include <lal/ComplexFFT.h>
#include <lal/TimeSeries.h>
#include <lal/FrequencySeries.h>
#include <lal/LALConstants.h>
#include <lal/Sequence.h>
#include <assert.h>
/*#include <spa.c>*/
/*
 * Data structure methods
 */


int free_waveform_interp_objects(struct twod_waveform_interpolant_array * interps) {
	int i;
	struct twod_waveform_interpolant *interp = interps->interp;
	/* if they exist free the C_KL matrices */
	for (i = 0; i < interps->size; i++, interp++) {
		if (interp->C_KL) gsl_matrix_complex_free(interp->C_KL);
	}
	free(interps->interp);
	free(interps);
	return 0;
	}


struct twod_waveform_interpolant_array * new_waveform_interpolant_array_from_svd_bank(gsl_matrix *svd_bank,  double param1_min, double param2_min, double param1_max, double param2_max)
{
	int i;
	struct twod_waveform_interpolant_array * output = (struct twod_waveform_interpolant_array *) calloc(1, sizeof(struct twod_waveform_interpolant_array));
	output->size = svd_bank->size1; 
	output->interp = (struct twod_waveform_interpolant *) calloc(output->size, sizeof(struct twod_waveform_interpolant));
	/* FIXME: how should CKL be allocated? */
	output->param1_min = param1_min;
	output->param2_min = param2_min;
	output->param1_max = param1_max;
	output->param2_max = param2_max;
	for (i = 0; i < output->size; i++) {
		output->interp[i].svd_basis = gsl_matrix_row(svd_bank, i);
		output->interp[i].C_KL = NULL;
	}
	return output;
}


/*
 * Formula (3) in http://arxiv.org/pdf/1108.5618v1.pdf
 */ 

static gsl_matrix_complex *projection_coefficient(gsl_vector *svd_basis, gsl_matrix *template_bank, int N_mc, int M_eta){
	int i,j;
	/*project svd basis onto SPA's to get coefficients*/
	gsl_complex M;
	double M_real, M_imag;
	/* compute M_ky at fixed mu */
	gsl_matrix_complex *M_xy = gsl_matrix_complex_calloc(N_mc, M_eta);

	assert(N_mc * M_eta == template_bank->size1 / 2);

	for (unsigned int k =0; k < template_bank->size1 / 2; k++){
	
		i = floor(k/N_mc);
		j = k - floor(k/N_mc)*N_mc; /* indices for M_xy */

		gsl_vector_view spa_waveform_real = gsl_matrix_column(template_bank, 2*k);
		gsl_vector_view spa_waveform_imag = gsl_matrix_column(template_bank, 2*k+1);

		gsl_blas_ddot(&spa_waveform_real.vector, svd_basis, &M_real);
	        gsl_blas_ddot(&spa_waveform_imag.vector, svd_basis, &M_imag);
	
		GSL_SET_COMPLEX(&M, M_real, M_imag);
		gsl_matrix_complex_set(M_xy, i, j, M);
	}


	return M_xy;
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


/* waveform template stuff */

double mc2mass1(double mc, double eta)
/* mass 1 (the smaller one) for given mass ratio & chirp mass */
{
 double root = sqrt(0.25-eta);
 double fraction = (0.5+root) / (0.5-root);
 return mc * (pow(1+fraction,0.2) / pow(fraction,0.6));
}


double mc2mass2(double mc, double eta)
/* mass 2 (the greater one) for given mass ratio & chirp mass */
{
 double root = sqrt(0.25-eta);
 double inversefraction = (0.5-root) / (0.5+root);
 return mc * (pow(1+inversefraction,0.2) / pow(inversefraction,0.6));
}

static int SPAWaveformReduceSpin (double mass1, double mass2, double chi, 
        int order, double startTime, double phi0, double deltaF,
        double fLower, double fFinal, int numPoints, COMPLEX16* hOfF) {

    double m = mass1 + mass2;
    double eta = mass1 * mass2 / m / m;
    
    double psi2 = 0., psi3 = 0., psi4 = 0., psi5 = 0., psi6 = 0., psi6L = 0., psi7 = 0.; 
    double psi3S = 0., psi4S = 0., psi5S = 0., psi0; 
    double alpha2 = 0., alpha3 = 0., alpha4 = 0., alpha5 = 0., alpha6 = 0., alpha6L = 0.;
    double alpha7 = 0., alpha3S = 0., alpha4S = 0., alpha5S = 0.; 
    double f, v, v2, v3, v4, v5, v6, v7, Psi, amp, shft, amp0, d_eff; 
    int k, kmin, kmax; 

    double mSevenBySix = -7./6.;
    double piM = LAL_PI*m*LAL_MTSUN_SI;
    double oneByThree = 1./3.;
    double piBy4 = LAL_PI/4.;

    COMPLEX16 H;

    /************************************************************************/
    /* spin terms in the ampl & phase in terms of the 'reduced-spin' param. */
    /************************************************************************/
    psi3S = 113.*chi/3.;
    psi4S = 63845.*(-81. + 4.*eta)*chi*chi/(8.*pow(-113. + 76.*eta, 2.));  
    psi5S = -565.*(-146597. + 135856.*eta + 17136.*eta*eta)*chi/(2268.*(-113. + 76.*eta)); 

    alpha3S = (113*chi)/24.; 
    alpha4S = (12769*pow(chi,2)*(-81 + 4*eta))/(32.*pow(-113 + 76*eta,2)); 
    alpha5S = (-113*chi*(502429 - 591368*eta + 1680*pow(eta,2)))/(16128.*(-113 + 76*eta)); 

    /* coefficients of the phase at PN orders from 0 to 3.5PN */
    psi0 = 3./(128.*eta);

    /************************************************************************/
    /* set the amplitude and phase coefficients according to the PN order   */
    /************************************************************************/
    switch (order) {
        case 7: 
            psi7 = (77096675.*LAL_PI)/254016. + (378515.*LAL_PI*eta)/1512.  
                     - (74045.*LAL_PI*eta*eta)/756.;
            alpha7 = (-5111593*LAL_PI)/2.709504e6 - (72221*eta*LAL_PI)/24192. - 
                        (1349*pow(eta,2)*LAL_PI)/24192.; 
        case 6:
            psi6 = 11583231236531./4694215680. - (640.*LAL_PI*LAL_PI)/3. - (6848.*LAL_GAMMA)/21. 
                     + (-5162.983708047263 + 2255.*LAL_PI*LAL_PI/12.)*eta 
                     + (76055.*eta*eta)/1728. - (127825.*eta*eta*eta)/1296.;
            psi6L = -6848./21.;
            alpha6 = -58.601030974347324 + (3526813753*eta)/2.7869184e7 - 
                        (1041557*pow(eta,2))/258048. + (67999*pow(eta,3))/82944. + 
                    	(10*pow(LAL_PI,2))/3. - (451*eta*pow(LAL_PI,2))/96.; 
            alpha6L = 856/105.; 
        case 5:
            psi5 = (38645.*LAL_PI/756. - 65.*LAL_PI*eta/9. + psi5S);
            alpha5 = (-4757*LAL_PI)/1344. + (57*eta*LAL_PI)/16. + alpha5S; 
        case 4:
            psi4 = 15293365./508032. + 27145.*eta/504. + 3085.*eta*eta/72. + psi4S;
            alpha4 = 0.8939214212884228 + (18913*eta)/16128. + (1379*pow(eta,2))/1152. + alpha4S; 
        case 3:
            psi3 = psi3S - 16.*LAL_PI;
            alpha3 = -2*LAL_PI + alpha3S; 
        case 2:
            psi2 = 3715./756. + 55.*eta/9.;
            alpha2 = 1.1056547619047619 + (11*eta)/8.; 
        default:
            break;
    }

    /* compute the amplitude assuming effective dist. of 1 Mpc */
    d_eff = 1e6*LAL_PC_SI/LAL_C_SI;  /*1 Mpc in seconds */
    amp0 = sqrt(5.*eta/24.)*pow(m*LAL_MTSUN_SI, 5./6.)/(d_eff*pow(LAL_PI, 2./3.));

    shft = 2.*LAL_PI *startTime;

    /* zero output */    
	kmin = fLower / deltaF > 1 ? fLower / deltaF : 1;
	kmax = fFinal / deltaF < numPoints  ? fFinal / deltaF : numPoints ;

    /************************************************************************/
    /*          now generate the waveform at all frequency bins             */
    /************************************************************************/
    for (k = kmin; k < kmax; k++) {

        /* fourier frequency corresponding to this bin */
      	f = k * deltaF;
        v = pow(piM*f, oneByThree);

        v2 = v*v;   v3 = v2*v;  v4 = v3*v;  v5 = v4*v;  v6 = v5*v;  v7 = v6*v;

        /* compute the phase and amplitude */
        if ((f < fLower) || (f > fFinal)) {
            amp = 0.;
            Psi = 0.;
        }
        else {

            Psi = psi0*pow(v, -5.)*(1. 
                    + psi2*v2 + psi3*v3 + psi4*v4 
                    + psi5*v5*(1.+3.*log(v)) 
                    + (psi6 + psi6L*log(4.*v))*v6 + psi7*v7); 

            amp = amp0*pow(f, mSevenBySix)*(1. 
                    + alpha2*v2 + alpha3*v3 + alpha4*v4 
                    + alpha5*v5 + (alpha6 + alpha6L*(LAL_GAMMA+log(4.*v)) )*v6 
                    + alpha7*v7); 

        }

	H.re =  amp * (cos(Psi+shft*f+phi0+piBy4));
	H.im =  -amp*sin(Psi+shft*f+phi0+piBy4);
        
	/* generate the waveform */
	hOfF[k] = H;
    }    

	return 0;
}






static double chirp_time (double m1, double m2, double fLower, int order, double chi)
	{

	/* variables used to compute chirp time */
	double c0T, c2T, c3T, c4T, c5T, c6T, c6LogT, c7T;
	double xT, x2T, x3T, x4T, x5T, x6T, x7T, x8T;
	double m = m1 + m2;
	double eta = m1 * m2 / m / m;

	c0T = c2T = c3T = c4T = c5T = c6T = c6LogT = c7T = 0.;

	/* Switch on PN order, set the chirp time coeffs for that order */
	switch (order)
		{
		case 8:
		case 7:
			c7T = LAL_PI * (14809.0 * eta * eta - 75703.0 * eta / 756.0 - 15419335.0 / 127008.0);
		case 6:
			c6T = LAL_GAMMA * 6848.0 / 105.0 - 10052469856691.0 / 23471078400.0 + LAL_PI * LAL_PI * 128.0 / 3.0 + eta * (3147553127.0 / 3048192.0 - LAL_PI * LAL_PI * 451.0 / 12.0) - eta * eta * 15211.0 / 1728.0 + eta * eta * eta * 25565.0 / 1296.0 + log (4.0) * 6848.0 / 105.0;
     			c6LogT = 6848.0 / 105.0;
		case 5:
			c5T = 13.0 * LAL_PI * eta / 3.0 - 7729.0 / 252.0 - (0.4*565.*(-146597. + 135856.*eta + 17136.*eta*eta)*chi/(2268.*(-113. + 76.*eta))); /* last term is 0 if chi is 0*/
		case 4:
			c4T = 3058673.0 / 508032.0 + eta * (5429.0 / 504.0 + eta * 617.0 / 72.0) + (0.4*63845.*(-81. + 4.*eta)*chi*chi/(8.*pow(-113. + 76.*eta, 2.))); /* last term is 0 if chi is 0*/
			c3T = -32.0 * LAL_PI / 5.0 + (0.4*113.*chi/3.); /* last term is 0 if chi is 0*/
			c2T = 743.0 / 252.0 + eta * 11.0 / 3.0;
			c0T = 5.0 * m * LAL_MTSUN_SI / (256.0 * eta);
			break;
		default:
			fprintf (stderr, "ERROR!!!\n");
			break;
		}

	/* This is the PN parameter v evaluated at the lower freq. cutoff */
	xT = pow (LAL_PI * m * LAL_MTSUN_SI * fLower, 1.0 / 3.0);
	x2T = xT * xT;
	x3T = xT * x2T;
	x4T = x2T * x2T;
	x5T = x2T * x3T;
	x6T = x3T * x3T;
	x7T = x3T * x4T;
	x8T = x4T * x4T;

	/* Computes the chirp time as tC = t(v_low)    */
	/* tC = t(v_low) - t(v_upper) would be more    */
	/* correct, but the difference is negligble.   */

	/* This formula works for any PN order, because */
	/* higher order coeffs will be set to zero.     */

	return c0T * (1 + c2T * x2T + c3T * x3T + c4T * x4T + c5T * x5T + (c6T + c6LogT * log (xT)) * x6T + c7T * x7T) / x8T;
}

static double ffinal(double m1, double m2){
	
	/* Compute frequency at Schwarzschild ISCO */

	double f_isco;
	
	pow(2., ceil( log( (1./LAL_PI)*( pow(6.,-3./2.) )*( pow(m1+m2,-1.) ) ) ) / log(2) ); /* Next highest power of 2 of f_isco */
	
	return f_isco;
}

static int generate_template(double m1, double m2, double duration, double f_low, double f_high, double order, COMPLEX16FrequencySeries *hOfF){
	SPAWaveformReduceSpin(m1, m2, 0, order, 0, 0, 1.0 / duration, f_low, f_high, hOfF->data->length, hOfF->data->data);
	return 0;
}

static int get_psd_from_file(REAL8FrequencySeries *series, char *fname){
	
	FILE *fp = fopen(fname, "r");
	double deltaF = series->deltaF;
	double f0 = series->f0;
	double f;
	const gsl_interp_type *t = gsl_interp_linear;
	gsl_interp *g_interp = gsl_interp_alloc (t, 16384);
	gsl_interp_accel *acc = gsl_interp_accel_alloc();
	double freq[16384];
	double psd[16384];
	
	int i = 0;
	while (!feof(fp)) {
		if (i >= 16384) break;
		fscanf(fp, "%lf %le\n", &freq[i], &psd[i]);
		i++;
	}

	gsl_interp_init(g_interp, freq, psd, 16384);

	for (i = 0; i < series->data->length; i++) {
		f = f0 + i * deltaF;
		series->data->data[i] = gsl_interp_eval(g_interp, freq, psd, f, acc);
	}

	gsl_interp_free(g_interp);
	gsl_interp_accel_free (acc);
	fclose(fp);
	return 0;
}

static int freq_to_time_fft(COMPLEX16FrequencySeries *fseries, REAL8FrequencySeries* psd, COMPLEX16TimeSeries *tseries, COMPLEX16FFTPlan *revplan){

	XLALWhitenCOMPLEX16FrequencySeries(fseries, psd);
	XLALCOMPLEX16FreqTimeFFT(tseries, fseries, revplan);
	
	return 0;
}

/*
 * High level functions
 */

static int compute_max_chirp_time_and_max_frequency(double mc_min, double mc_max, double eta_min, double eta_max, double f_min, double *f_max, double *t_max) {
	
	/* largest frequency is the largest total mass which is the largest mchirp and eta */
	double m1max = mc2mass1(mc_max, eta_max);
	double m2max = mc2mass2(mc_max, eta_max);

	/* FIXME longest chirp time comes from the smallest chirp mass and eta corrections which we say comes from the smallest eta but have not verified... */
	double m1min = mc2mass1(mc_min, eta_min);
	double m2min = mc2mass2(mc_min, eta_min);
	
	*f_max = ffinal(m1max, m2max);
	*t_max = chirp_time(m1min, m2min, f_min, 7, 0);
	
	return 0;
}

static int compute_working_length_and_sample_rate(double chirp_time, double f_max, unsigned int *working_length, double *sample_rate) {
	
	double duration = pow(2., ceil(log(chirp_time) / log(2.))); /* see SPADocstring in _spawaveform.c */
	/* double length_max = 0; */
	
	*sample_rate = pow(2., ceil(log(2.* f_max) / log(2.)));
	unsigned int length_max =  round(*sample_rate * duration);
	*working_length = (unsigned int) round(pow(2., ceil(log(length_max + round(32.0 * *(sample_rate))) / log(2.))));

	return 0;
}

static gsl_matrix *create_templates_from_mc_and_eta(double mc_min, double mc_max, double N_mc, double eta_min, double eta_max, double M_eta, double f_min, REAL8FrequencySeries* psd, COMPLEX16TimeSeries* tseries, COMPLEX16FrequencySeries* fseries, COMPLEX16FFTPlan* revplan, int node_flag){
       /*
 	* N_mc is number of points on M_c grid
 	* viceversa for M_eta	
 	*/ 
	int i,j,k=0;
	double sample_rate, deltaT;
	double working_duration;
        unsigned int working_length;
	double eta, mc, m1, m2;
	gsl_matrix *A = NULL;
	
	/* fseries, tseries and revplan are used elsewhere so we create them once in main and pass them as arguments
	COMPLEX16FrequencySeries *fseries;
	COMPLEX16TimeSeries *tseries;
	LIGOTimeGPS epoch = LIGOTIMEGPSZERO;
	COMPLEX16FFTPlan *revplan;
	*/
	working_length = psd->data->length;

	/* gsl_matrix *A will contain template bank */
	A = gsl_matrix_calloc(2*N_mc*M_eta, working_length); 

        /*working_duration = 1.0 / psd->deltaF;
	sample_rate = working_length / working_duration;
	deltaT = 1. / sample_rate;
	*/

	/* allocate the frequency and time series once and for all */
	/*tseries = XLALCreateCOMPLEX16TimeSeries(NULL, &epoch, 0., deltaT, &lalStrainUnit, working_length);
	fseries = XLALCreateCOMPLEX16FrequencySeries(NULL, &epoch, 0, 1. / working_duration, &lalDimensionlessUnit, working_length);
	revplan = XLALCreateReverseCOMPLEX16FFTPlan(fseries->data->length, 1);
	*/
	if (node_flag == 0) {

	for ( i = 0; i < N_mc ; i++){
		k+=i;
		for ( j = 0; j < M_eta ; j++){
			k+=j;

			eta = eta_min + (j/(M_eta-1))*(eta_max - eta_min);
			mc = mc_min + (i/(N_mc-1))*(mc_max - mc_min);

                        m1 = mc2mass1(mc, eta);
                        m2 = mc2mass2(mc, eta);

			generate_template(m1, m2, 1.0 / psd->deltaF, f_min, working_length / working_duration / (2*1.05), 7, fseries);
			freq_to_time_fft(fseries, psd, tseries, revplan); /* return whitened complex time series */	
			for (i = 0; i < tseries->data->length; i++) {
				gsl_matrix_set(A, 2*k, i, tseries->data->data[i].re);
				gsl_matrix_set(A, 2*k+1, i, tseries->data->data[i].im);
			}
		}
	}
	
	/*XLALDestroyCOMPLEX16TimeSeries(tseries);
	XLALDestroyCOMPLEX16FrequencySeries(fseries);
	XLALDestroyReverseCOMPLEX16FFTPlan(revplan);
	*/

	return A;

	}

	else if (node_flag !=0){
        
	for ( i = 0; i < N_mc ; i++){
                k+=i;
                for ( j = 0; j < M_eta ; j++){
                        k+=j;
                        eta = eta_min + ( ( chebyshev_node(j,M_eta)+1. )/2.)*(eta_max - eta_min);
                        mc = mc_min + ( ( chebyshev_node(i,N_mc)+1. )/2.)*(mc_max - mc_min);
                        m1 = mc2mass1(mc, eta);
                        m2 = mc2mass2(mc, eta);

                        generate_template(m1, m2, 1.0 / psd->deltaF, f_min, working_length / working_duration / (2*1.05), 7, fseries);
                        freq_to_time_fft(fseries, psd, tseries, revplan); /* return whitened complex time series */
                        for (i = 0; i < tseries->data->length; i++) {
                                gsl_matrix_set(A, 2*k, i, tseries->data->data[i].re);
                                gsl_matrix_set(A, 2*k+1, i, tseries->data->data[i].im);
                        }
                }
        }
		
	return A;

	}



}

static gsl_matrix *create_svd_basis_from_template_bank(gsl_matrix* template_bank){
	
	double tolerance;
	double norm_s;
	double sum_s = 0.;
	int n;
	gsl_matrix *output;
	gsl_matrix_view template_view;
	gsl_matrix *V;
	gsl_vector *S; 

	/* Work space matrices */
	gsl_matrix *gX;
	gsl_vector *gW;

	gX = gsl_matrix_calloc(template_bank->size2, template_bank->size2);
 	gW = gsl_vector_calloc(template_bank->size2);	

	V = gsl_matrix_calloc(template_bank->size2, template_bank->size2);
	S = gsl_vector_calloc(template_bank->size2);

	gsl_linalg_SV_decomp_mod(template_bank, gX, V, S, gW);

	/* FIXME, add the svd truncation, replace template_bank_ptr with the
 	 * new truncated matrix and free the template_bank matrix, populate the M
	 * matrix and free V S gX and gW when you are done
	 */
	
	/*Note: We don't actually need the coefficients that come from the calculation of the basis template */

	tolerance = 1e-6;
	norm_s = gsl_blas_dnrm2(S);
	
	/*FIXME make this more sophisticated if you care */
	for (n = 0; n < S->size; n++) {
		sum_s += gsl_vector_get(S, n) * gsl_vector_get(S, n);
		if (sqrt(sum_s / norm_s) >= tolerance) break;
		}

	template_view = gsl_matrix_submatrix(template_bank, 0, 0, n, template_bank->size2);
	output = gsl_matrix_calloc(n, template_bank->size2);
	fprintf(stderr, "output size %d %d\n", output->size1, output->size2);
	gsl_matrix_memcpy(output, &template_view.matrix);
	return output;
}

/* FIXME use a better name */
static gsl_vector_complex *interpolate_waveform_from_mchirp_and_eta(struct twod_waveform_interpolant_array *interps, double mchirp, double eta) { 
	int i;
	gsl_complex M;
	double deltaF, x, y;
	struct twod_waveform_interpolant *interp = interps->interp;
	gsl_vector_complex *h_t = gsl_vector_complex_calloc(interp->svd_basis.vector.size);
	gsl_vector_view h_t_real = gsl_vector_complex_real(h_t); 
	gsl_vector_view h_t_imag = gsl_vector_complex_imag(h_t);

	x = map_coordinate_to_cheby(interps->param1_min, interps->param1_max, mchirp);
	y = map_coordinate_to_cheby(interps->param2_min, interps->param2_max, eta);
	gsl_complex z_tmp;	
	/* this is the loop over mu */
	for (i = 0; i < interps->size; i++, interp++) {
			M = compute_M_xy(interp->C_KL, x, y);
			gsl_blas_daxpy (GSL_REAL(M), &interp->svd_basis.vector, &h_t_real.vector);
			gsl_blas_daxpy (GSL_IMAG(M), &interp->svd_basis.vector, &h_t_imag.vector);
		}
	return h_t;
}

/* example usage */

int main() {

	/* FIXME set these */
	double mc_min = 7.0;
	double eta_min = 0.1;
	double mc_max = 7.6;
	double eta_max = 0.25;
	double N_mc = 40;
	double M_eta = 40;
	double f_min = 40.0;
	double t_max = 0;
	double f_max = 0;
	double eta, mc, m1, m2;
	unsigned int working_length;
	double sample_rate, working_duration;
	double New_N_mc, New_M_eta;
	double deltaT;
	double Overlap;

	gsl_complex dotc1,dotc2,dotc3; /* used for products in overlap function */
	gsl_complex z_tmp;
	gsl_vector_complex *Tseries=NULL;
	gsl_vector_complex* h_t = NULL;
	gsl_vector *x_nodes = gsl_vector_calloc(N_mc);
	gsl_vector *y_nodes = gsl_vector_calloc(M_eta);
	gsl_matrix_complex *M_xy=NULL;
	gsl_matrix *templates = NULL;
	gsl_matrix *svd_basis = NULL;
	gsl_matrix *templates_at_nodes=NULL;

	struct twod_waveform_interpolant_array *interps = NULL;

	REAL8FrequencySeries *psd = NULL;
	LIGOTimeGPS epoch = LIGOTIMEGPSZERO;
        COMPLEX16FrequencySeries *fseries;
        COMPLEX16TimeSeries *tseries;
	COMPLEX16FFTPlan *revplan;

	compute_max_chirp_time_and_max_frequency(mc_min, mc_max, eta_min, eta_max, f_min, &f_max, &t_max);
	compute_working_length_and_sample_rate(t_max, f_max, &working_length, &sample_rate);

	deltaT = 1. / sample_rate;
        working_duration = 1. / (working_length * sample_rate);

	psd = XLALCreateREAL8FrequencySeries(NULL, &epoch, 0, 1. / (working_duration), &lalDimensionlessUnit, working_length);
	get_psd_from_file(psd, "reference_psd.txt");

	/* templates and psd is allocated by this function make sure to free them */
	/* allocate tseries, fseries and revplan here for use throughout main */

	tseries = XLALCreateCOMPLEX16TimeSeries(NULL, &epoch, 0., deltaT, &lalStrainUnit, working_length);
        fseries = XLALCreateCOMPLEX16FrequencySeries(NULL, &epoch, 0, 1. / working_duration, &lalDimensionlessUnit, working_length);
        revplan = XLALCreateReverseCOMPLEX16FFTPlan(fseries->data->length, 1);


	templates = create_templates_from_mc_and_eta(mc_min, mc_max, N_mc, eta_min, eta_max, M_eta, f_min, psd, tseries, fseries, revplan, 0);
	
	/* FIXME HERE DOWN: still need to compute node points */
	
	svd_basis = create_svd_basis_from_template_bank(templates);

	
	/* FIXME: CKL needs to be set in new_waveform_interpolant_array_from_svd_bank */
	interps = new_waveform_interpolant_array_from_svd_bank(svd_basis, mc_min, eta_min, mc_max, eta_max );	

	/*FIXME: Calculation of waveforms at the coolcation points:*/
	/* Compute new template bank at colocation points-> project onto basis vectors
 	 * to get matrix of coefficients for C_KL computation */

	for (unsigned int i=0; i < N_mc; i++){
		gsl_vector_set(x_nodes, i, chebyshev_node(i,N_mc) ); 
	}

	for (unsigned int j=0; j < M_eta ; j++){
		gsl_vector_set(x_nodes, j, chebyshev_node(j,M_eta) ); 
	}

	templates_at_nodes = create_templates_from_mc_and_eta(mc_min, mc_max, N_mc, eta_min, eta_max, M_eta, f_min, psd, tseries, fseries, revplan, 1);	

	for (unsigned int i = 0; i < interps->size; i++, interps->interp++) {	
		M_xy = projection_coefficient(&interps->interp->svd_basis.vector, templates_at_nodes, N_mc, M_eta);
		interps->interp->C_KL = compute_C_KL(x_nodes, y_nodes, M_xy);		
		
	}

	New_N_mc = 100;
	New_M_eta = 100;

	for (unsigned int i =0; i <  New_N_mc; i++){
		for (unsigned int j =0; j <  New_N_mc; j++){
			eta = eta_min + (j/(New_M_eta-1))*(eta_max - eta_min);
			mc = mc_min + (i/(New_N_mc-1))*(mc_max - mc_min);
			
			h_t = interpolate_waveform_from_mchirp_and_eta(interps, mc, eta);
			
                        m1 = mc2mass1(mc, eta);
                        m2 = mc2mass2(mc, eta);

                        generate_template(m1, m2, working_duration, f_min, sample_rate / (2*1.05), 7, fseries);
			freq_to_time_fft(fseries, psd, tseries, revplan);
			

			/* calculate overlap of waveforms */	
			//
			for (unsigned int l =0; l <  working_length; l++){
					
				GSL_SET_COMPLEX(&z_tmp, tseries->data->data[l].re, tseries->data->data[l].im);
				gsl_vector_complex_set(Tseries,l, z_tmp);

			gsl_blas_zdotc(Tseries, h_t, &dotc1);
			gsl_blas_zdotc(h_t, h_t, &dotc2);	
			gsl_blas_zdotc(Tseries, Tseries, &dotc3);

		 	Overlap = ( gsl_complex_abs(dotc1) / gsl_complex_abs( gsl_complex_sqrt(dotc2) ) / gsl_complex_abs( gsl_complex_sqrt(dotc3) ) );

			}
		}		
	}

	gsl_matrix_free(templates);
	gsl_vector_complex_free(h_t);
	gsl_vector_complex_free(Tseries);

	XLALDestroyCOMPLEX16TimeSeries(tseries);
        XLALDestroyCOMPLEX16FrequencySeries(fseries);
        XLALDestroyCOMPLEX16FFTPlan(revplan);
	XLALDestroyREAL8FrequencySeries(psd);
	
	return 0;
}


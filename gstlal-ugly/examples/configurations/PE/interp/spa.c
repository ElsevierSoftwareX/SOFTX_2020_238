#include <lal/LALDatatypes.h>
#include <lal/LALStdlib.h>
#include <lal/Units.h>
#include <lal/LALInspiral.h>


gsl_vector_complex SPAWaveformReduceSpin (double mass1, double mass2, double chi, 
        int order, double startTime, double phi0, double deltaF,
        double fLower, double fFinal, int numPoints) {

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

    /* zero outout */    
    hOfF = gsl_complex_vector gsl_vector_complex_calloc(hOfF, numPoints * sizeof (complex double));

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
	gsl_complex Z;
	GSL_SET_COMPLEX(&Z, 0, -amp*sin(Psi+shft*f+phi0+piBy4))
        /* generate the waveform */
	gsl_vector_complex_set(&hOfF, k, gsl_complex_add_real(Z, amp*cos(Psi+shft*f+phi0+piBy4)) )

    }    

	return hOfF;
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
			c5T = 13.0 * LAL_PI * eta / 3.0 - 7729.0 / 252.0 - (0.4*565.*(-146597. + 135856.*eta + 17136.*eta*eta)*chi/(2268.*(-113. + 76.*eta))); // last term is 0 if chi is 0;
		case 4:
			c4T = 3058673.0 / 508032.0 + eta * (5429.0 / 504.0 + eta * 617.0 / 72.0) + (0.4*63845.*(-81. + 4.*eta)*chi*chi/(8.*pow(-113. + 76.*eta, 2.))); // last term is 0 if chi is 0;
			c3T = -32.0 * LAL_PI / 5.0 + (0.4*113.*chi/3.); // last term is 0 if chi is 0;
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
	for (k = kmin; k < kmax; ++k)
		{
		double x = x1 * pow ((double) k, -1.0 / 3.0);
		double psi = c0 * (x * (c20 + x * (c15 + x * (c10 + x * x))) + c25 - c25Log * log (x) + (1.0 / x) * (c30 - c30Log * log (x) + (1.0 / x) * (c35 - (1.0 / x) * c40P * log (x))));

		double psi1 = psi + psi0;
		double psi2;

		/* range reduction of psi1 */
		while (psi1 < -LAL_PI)
			{
			psi1 += 2 * LAL_PI;
			psi0 += 2 * LAL_PI;
			}
		while (psi1 > LAL_PI)
			{
			psi1 -= 2 * LAL_PI;
			psi0 -= 2 * LAL_PI;
			}

		/* compute approximate sine and cosine of psi1 */
		if (psi1 < -LAL_PI / 2)
			{
			psi1 = -LAL_PI - psi1;
			psi2 = psi1 * psi1;
			/* XXX minus sign added because of new sign convention for fft */
			/* FIXME minus sign put back because it makes a reverse chirp with scipy's ifft */
			value = psi1 * (1 + psi2 * (s2 + psi2 * s4)) +  I * (0. - 1. - psi2 * (c2 + psi2 * c4));
			expPsi[k]  = value;
			}
		else if (psi1 > LAL_PI / 2)
			{
			psi1 = LAL_PI - psi1;
			psi2 = psi1 * psi1;
			/* XXX minus sign added because of new sign convention for fft */
			/* FIXME minus sign put back because it makes a reverse chirp with scipy's ifft */
			value = psi1 * (1 + psi2 * (s2 + psi2 * s4)) + I * (0. - 1. - psi2 * (c2 + psi2 * c4));
			expPsi[k] = value;
			}
		else
			{
			psi2 = psi1 * psi1;
			/* XXX minus sign added because of new sign convention for fft */
			/* FIXME minus sign put back because it makes a reverse chirp with scipy's ifft */
			value = psi1 * (1 + psi2 * (s2 + psi2 * s4)) + I * (1. + psi2 * (c2 + psi2 * c4));
			expPsi[k] = value;
			}
		/* put in the first order amplitude factor */
		expPsi[k] *= pow(k*deltaF, -7.0 / 6.0) * tNorm;
		}
	return 0;
	}


gsl_vector_complex* generate_template(double mass1, double mass2, double sample_rate, double duration, double f_low, double f_high, double order = 7){
      /*
      *	Generate a single frequency-domain template, which
      *	 (1) is band-limited between f_low and f_high,
      *	 (2) has an IFFT which is duration seconds long and
      *	 (3) has an IFFT which is sampled at sample_rate Hz
      */
	gsl_vector_complex z = gsl_vector_complex_calloc(sample_rate * duration)
	//if approximant=="FindChirpSP" or approximant=="TaylorF2":
	SPAWaveformReduceSpin(mass1, mass2, order, 1.0 / duration, 1.0 / sample_rate, f_low, f_high, z, template_bank_row.chi)
	
     /*	elif approximant=="IMRPhenomB":
     *	 	#FIXME a better plan than multiplying flow by 0.5 should be done...
     *		spawaveform.imrwaveform(template_bank_row.mass1, template_bank_row.mass2, 1.0/duration, 0.5 * f_low, z, template_bank_row.chi)
     *	else:
     *		raise ValueError, "Unsupported approximant given"
     */
	return COMPLEX16FrequencySeries(
		name = "template",
		epoch = LIGOTimeGPS(0),
		f0 = 0.0,
		deltaF = 1.0 / duration,
		sampleUnits = LALUnit("strain"),
		data = z[:len(z)  2 + 1]
	)

}


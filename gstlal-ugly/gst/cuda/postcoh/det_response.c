
/* This is modified from lalsuite DetResponse.c */
/**
 * An implementation of the detector response formulae in Anderson et al
 * PRD 63 042003 (2001) \cite ABCF2001.
 *
 * Computes F+ and Fx for a source at a specified sky position,
 * polarization angle, and sidereal time.  Also requires the detector's
 * response matrix which is defined by Eq. (B6) of [ABCF] using either
 * Table 1 of \cite ABCF2001 or Eqs. (B11)--(B17) to compute the arm
 * direction unit vectors.
 */
void DetAMResponseMatrix(
	double *fplus,		/**< Returned value of F+ */
	double *fcross,		/**< Returned value of Fx */
	const REAL4 D[3][3],	/**< Detector response 3x3 matrix */
	const double ra,	/**< Right ascention of source (radians) */
	const double dec,	/**< Declination of source (radians) */
	const double psi,	/**< Polarization angle of source (radians) */
	const double gmst	/**< Greenwich mean sidereal time (radians) */
)
{
	int i;
	double X[3];
	double Y[3];

	/* Greenwich hour angle of source (radians). */
	const double gha = gmst - ra;

	/* pre-compute trig functions */
	const double cosgha = cos(gha);
	const double singha = sin(gha);
	const double cosdec = cos(dec);
	const double sindec = sin(dec);
	const double cospsi = cos(psi);
	const double sinpsi = sin(psi);

	/* Eq. (B4) of [ABCF].  Note that dec = pi/2 - theta, and gha =
	 * -phi where theta and phi are the standard spherical coordinates
	 * used in that paper. */
	X[0] = -cospsi * singha - sinpsi * cosgha * sindec;
	X[1] = -cospsi * cosgha + sinpsi * singha * sindec;
	X[2] =  sinpsi * cosdec;

	/* Eq. (B5) of [ABCF].  Note that dec = pi/2 - theta, and gha =
	 * -phi where theta and phi are the standard spherical coordinates
	 * used in that paper. */
	Y[0] =  sinpsi * singha - cospsi * cosgha * sindec;
	Y[1] =  sinpsi * cosgha + cospsi * singha * sindec;
	Y[2] =  cospsi * cosdec;

	/* Now compute Eq. (B7) of [ABCF] for each polarization state, i.e.,
	 * with s+=1 and sx=0 to get F+, with s+=0 and sx=1 to get Fx */
	*fplus = *fcross = 0.0;
	for(i = 0; i < 3; i++) {
		const double DX = D[i][0] * X[0] + D[i][1] * X[1] + D[i][2] * X[2];
		const double DY = D[i][0] * Y[0] + D[i][1] * Y[1] + D[i][2] * Y[2];
		*fplus  += X[i] * DX - Y[i] * DY;
		*fcross += X[i] * DY + Y[i] * DX;
	}
}



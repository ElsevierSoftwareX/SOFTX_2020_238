
/* for each sky direction, compute the coherent snr */

coh_sky_map(coh_snr, /* OUTPUT */
	coh_nullstream, /* OUTPUT */
	theta, /* INPUT, sky_direction */
	phi, /* INPUT, sky_direction */
	snr_all, /*INPUT, 2*data_points*/,
	detector_turn, /* INPUT, which detector we are considering */,
	offset0 /*INPUT place where the location of the trigger*/) 

	u(theta, phi); /* matrix 2*2, 3*3 */
	offset1(theta, phi);
	

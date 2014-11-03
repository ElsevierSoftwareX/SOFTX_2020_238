
/* for each sky direction, compute the coherent snr */

coh_sky_map(coh_snr, /* OUTPUT */
	coh_nullstream, /* OUTPUT */
	snr_all, /* INPUT, 2*data_points*/
	detector_turn, /* INPUT, which detector we are considering */
	offset0, /* INPUT place where the location of the trigger */
	need_update,
	sky_map) /* INPUT, map to get the sky direction */


	if need_update
	  for all sky directions in sky_map
	    update u matrix(index) = u(theta, phi, current_time);
	    update time_delay_ij matrix(index) (1 <= i, j <= num_detectors);
	else 
	    do nothing;
		
	get u(index); /* matrix 2*2, 3*3 */
	get time_delay_ij(index);
	interploate offset_ij;

	calc coh_snr, coh_nullstream;

	

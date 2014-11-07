
/* for each sky direction, compute the coherent snr */

coh_sky_map(float *coh_snr, /* OUTPUT */
	float *coh_nullstream, /* OUTPUT */
	complex_float *snr_all, /* INPUT, (2, 3)*data_points*/
	det_label, /* INPUT, which detector we are considering */
	detectors, /* INPUT, all the detectors that are in this coherent analysis */
	int *tiggers_offset0, /* INPUT place where the location of the trigger */
	int num_triggers, /* INPUT number of triggers */
	float *u_map, /* INPUT u matrix map, each u matrix corresponds to one sky direction  */
	float *toa_diff_map, /* INPUT, time of arrival difference map*/
	int num_sky_directions) 

	get u(index); /* matrix 2*2, 3*3 */
	get toa_diff(ij, index);

	calc coh_snr, coh_nullstream;

	

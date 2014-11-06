
/* for each sky direction, compute the coherent snr */

coh_sky_map(coh_snr, /* OUTPUT */
	coh_nullstream, /* OUTPUT */
	snr_all, /* INPUT, 2*data_points*/
	det_label, /* INPUT, which detector we are considering */
	detectors, /* INPUT, all the detectors that are in this coherent analysis */
	offset0, /* INPUT place where the location of the trigger */
	float *u_map, /* INPUT u matrix map, each u matrix corresponds to one sky direction  */
	float *toa_diff_map, ) /* INPUT, time of arrival difference map*/

	get u(index); /* matrix 2*2, 3*3 */
	get toa_diff(ij, index);

	calc coh_snr, coh_nullstream;

	

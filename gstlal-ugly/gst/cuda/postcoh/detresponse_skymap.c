#include <lal/Date.h>
#include <lal/Detectors.h>
#include <lal/DetResponse.h>
#include <lal/TimeDelay.h>
#include <lalsimulation/lalsimulation.h>

#include <chealpix.h>

void create_skylocs(unsigned char order) {
	unsigned long nside = (unsigned long) 1 << order;
	unsigned long npix = nside2npix(nside);
}

/* get u matirx for each sky direction from detector response for each sky 
 * direction at every minute
 */
LALDector* create_detectors_from_name(
		char **detectors_name,
		int num_detectors,
		)
{
	LALDetector *detectors = (LALDetector *)malloc(sizeof(LALDector) * num_detectors);
	for (i=0; i<num_detectors; i++) 
		detectors[i] = XLALDetectorPrefixToLALDetector(detectors_name[i]);
	return detectors;
}

void create_detresponse_skymap(
		LALDector *detectors,
		int num_detectors;
		double *horizons,
		)
{	// since 0h UTC 6 Jan 1980
	LIGOTimeGPS gpstime_start = {0, 0}; 

	LIGOTimeGPS gpstime_end = {24*3600, 0}; 

	double gmst_start = XLALGreenwichMeanSiderealTime(&gpstime_start);
	double gmst_end = XLALGreenwichMeanSiderealTime(&gpstime_end);
	double gmst_step = 60;

	for (double gmst=gmst_start; gmst<gmst_end; gmst+=gmst_step) {
		for (ipix; ipix<npix; ipix++) {
			for (i=0; i<num_detectors; i++) {
	
			pix2ang_nest(nside, ipix, &theta, &phi)
		
			/* get fplus, fcross from lalsuite DetResponse.c */

			XLALComputeDetAMResponse(*fplus, *fcross, detectors[i].response, ra, dec, psi, gmst);
		
			A_matrix[i][1] = *fplus;
			A_matrix[i][2] = *fcross;
			}
				
			u_matrix = svd(A_matrix);
			u_map[gmst_index][ipix] = u_matrix;

		}
	}
}

void create_time_delay_skymap(
		LALDetector * detectors,
		)
{
	XLALArrivalTimeDiff(detectors[i].location, detectors[j].location, ra, dec, gpstime);


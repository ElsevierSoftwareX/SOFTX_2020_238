#include <lal/Date.h>
#include <lal/Detectors.h>
#include <lal/DetResponse.h>
#include <lal/TimeDelay.h>
#include <lalsimulation/lalsimulation.h>

#include <chealpix.h>

typedef struct _DetSkymap {
	char **ifos,
	int nifo;
	double gmst_step;
	unsigned order;
	double *u_matrix;
	double *diff_matrix;
	int matrix_size[3];
} DetSkymap;



/* get u matirx for each sky direction from detector response for each sky 
 * direction at every minute
 */
LALDector** create_detectors_from_name(
		char **ifos,
		int ndetector,
		)
{
	LALDetector **detectors = (LALDetector **)malloc(sizeof(LALDector*) * ndetector);
	for (i=0; i<ndetector; i++) 
		/* LALSimulation.c returns lalCachedDetectors defined in LALDetectors.h, NOTE: external type */
		detectors[i] = XLALDetectorPrefixToLALDetector(ifos[i]);
	return detectors;
}

void create_detresponse_skymap(
		char **ifos,
		int nifo,
		int ndetector,
		double *horizons,
		double gmst_step,
		unsigned order
		)
{
	DetSkymap * det_map = (DetSkymap *)malloc(sizeof(DetSkymap));
	// from 0h UTC 6 Jan 1980
	LIGOTimeGPS gpstime_start = {0, 0}; 

	LIGOTimeGPS gpstime_end = {24*3600, 0}; 

	double gmst_start = XLALGreenwichMeanSiderealTime(&gpstime_start);
	double gmst_end = XLALGreenwichMeanSiderealTime(&gpstime_end);
	det_map->gmst_step = gmst_step;
	det_map->matrix_size[0] = (int)(gmst_end - gmst_start) / gmst_step;

	det_map->order = order;
	unsigned long nside = (unsigned long) 1 << order;
	det_map->matrix_size[1] = nside2npix(nside);
	det_map->u_matrix = 

	LALDector **detectors = create_detectors_from_name(ifos, nifo);
	det_map->nifo = nifo;
	det_map->matrix_size[2] = nifo * nifo;

	double tmp_Atmp_u[nifo*nifo], tmp_diff[nifo*nifo];
	double fplus, fcross;
	unsigned long ipix;

	for (double gmst=gmst_start; gmst<gmst_end; gmst+=gmst_step) {
		for (ipix=0; ipix<npix; ipix++) {
			for (i=0; i<ndetector; i++) {

			/* ra = phi, dec = 2pi - theta */	
			pix2ang_nest(nside, ipix, &theta, &phi)
		
			/* get fplus, fcross from lalsuite DetResponse.c */

			XLALComputeDetAMResponse(*fplus, *fcross, detectors[i].response, phi, M_PI_2, 0, gmst);
		
			A_matrix[i][1] = *fplus;
			A_matrix[i][2] = *fcross;
			}
				
			tmp_u = svd_u(A_matrix);
			det_map[gmst_index][ipix] = u_matrix;

		}
	}
}

void create_time_delay_skymap(
		LALDetector * detectors,
		)
{
	/* TimeDelay.c */
	double diff = XLALArrivalTimeDiff(detectors[i].location, detectors[j].location, ra, dec, gpstime);


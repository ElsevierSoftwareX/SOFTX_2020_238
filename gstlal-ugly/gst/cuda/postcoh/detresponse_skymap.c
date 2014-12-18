/* GStreamer
 * Copyright (C) Qi Chu,
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more deroll-offss.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */



#include <lal/Date.h>
#include <lal/LALDetectors.h>
#include <lal/DetResponse.h>
#include <lal/TimeDelay.h>
#include <lal/LALSimulation.h>

#include <lapacke/lapacke_config.h>
#include <lapacke/lapacke.h>

#include <chealpix.h>

#include <math.h>
#define min(a,b) ((a)>(b)?(b):(a))

typedef struct _DetSkymap {
	char **ifos;
	int nifo;
	double gps_step;
	unsigned order;
	float *U_map;
	float *diff_map;
	int matrix_size[3];
} DetSkymap;



/* get u matirx for each sky direction from detector response for each sky 
 * direction at every minute
 */
LALDetector* const* create_detectors_from_name(
		char **ifos,
		int nifo
		)
{
	LALDetector ** detectors = (LALDetector **)malloc(sizeof(LALDetector*) * nifo);
	const LALDetector *detector;
	int i;
	for (i=0; i<nifo; i++) {
		/* LALSimulation.c returns lalCachedDetectors defined in LALDetectors.h, NOTE: external type */
		printf("ifo%d %s\n", i, ifos[i]);
		detector = XLALDetectorPrefixToLALDetector(ifos[i]);
		detectors[i] = (LALDetector *)malloc(sizeof(LALDetector));
		memcpy(detectors[i], detector, sizeof(LALDetector));
	}
	return detectors;
}

DetSkymap *
create_detresponse_skymap(
		char **ifos,
		int nifo,
		double *horizons,
		double ingps_step,
		unsigned order
		)
{
	DetSkymap * det_map = (DetSkymap *)malloc(sizeof(DetSkymap));
	// from 0h UTC 6 Jan 1980
	LIGOTimeGPS gps_start = {0, 0}; 
	LIGOTimeGPS gps_end = {24*3600, 0};
	LIGOTimeGPS gps_step = {ingps_step, 0}; 
	LIGOTimeGPS gps_cur = {0, 0}; 

	int ngps;
	ngps = (int)(gps_end.gpsSeconds - gps_start.gpsSeconds) / gps_step.gpsSeconds;

	det_map->gps_step = ingps_step;
	det_map->matrix_size[0] = ngps;

	unsigned long nside = (unsigned long) 1 << order, npix = nside2npix(nside);
	det_map->order = order;
	det_map->matrix_size[1] = npix;

	LALDetector* const* detectors = create_detectors_from_name(ifos, nifo);
	det_map->nifo = nifo;
	det_map->matrix_size[2] = nifo * nifo;

       	int i, j;

	unsigned long U_len = (unsigned long) det_map->matrix_size[0] * (det_map->matrix_size)[1] * (det_map->matrix_size)[2];
	printf("u len %lu \n", U_len);
	det_map->U_map = (float *)malloc(sizeof(float) * U_len);
	det_map->diff_map = (float *)malloc(sizeof(float) * U_len);

	double theta, phi, fplus, fcross, gmst;
	unsigned long ipix;

	float *U_map = det_map->U_map, *diff_map = det_map->diff_map;
	double U[nifo*nifo], VT[2*2], S[2], diff[nifo*nifo], A[nifo*2], superb[min(nifo,2)-1];
	lapack_int lda = 2, ldu = nifo, ldvt = 2, info;


	unsigned long index = 0;
	for (gps_cur.gpsSeconds=gps_start.gpsSeconds; gps_cur.gpsSeconds<gps_end.gpsSeconds; XLALGPSAdd(&gps_cur, ingps_step)) {
		for (ipix=0; ipix<npix; ipix++) {
			for (i=0; i<nifo; i++) {

				/* ra = phi, dec = 2pi - theta */	
				pix2ang_nest(nside, ipix, &theta, &phi);
		
				/* get fplus, fcross from lalsuite DetResponse.c */

				gmst = XLALGreenwichMeanSiderealTime(&gps_cur);
				XLALComputeDetAMResponse(&fplus, &fcross, detectors[i]->response, phi, M_PI_2-theta, 0, gmst);
	
				A[i*2] = fplus;
				A[i*2 + 1] = fcross;
			}
				
			info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', nifo, 2, A, lda, S, U, ldu, VT, ldvt, superb);

			if(info > 0) {
				printf("SVD of A matrix failed to converge. \n");
				exit(1);
			}

			/* TimeDelay.c */
			for (i=0; i<nifo; i++) 
				for (j=0; j<nifo; j++) 
					diff[i*nifo+j] = XLALArrivalTimeDiff(detectors[i]->location, detectors[j]->location, phi, M_PI_2-theta, &gps_cur);

			for (i=0; i<nifo*nifo; i++) {

				U_map[index] = (float) U[i];
//				printf("index %d U %f\n", index, U[i]);
				diff_map[index] = (float) diff[i];
				index = index + 1;
			}

		}

	}
	return det_map;
}

static int to_xml(DetSkymap *det_map) 
{

return 0;
}

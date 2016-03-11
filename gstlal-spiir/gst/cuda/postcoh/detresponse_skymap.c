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

//#include <LIGOLw_xmllib/LIGOLwHeader.h>
#include "../LIGOLw_xmllib/LIGOLwHeader.h"
#include <math.h>
#define min(a,b) ((a)>(b)?(b):(a))

typedef struct _DetSkymap {
	char **ifos;
	int nifo;
	int gps_step;
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
		int ingps_step,
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

	LALDetector* const* detectors = create_detectors_from_name(ifos, nifo);
	det_map->nifo = nifo;
	det_map->matrix_size[1] = nifo * nifo;

	unsigned long nside = (unsigned long) 1 << order, npix = nside2npix(nside);
	det_map->order = order;
	det_map->matrix_size[2] = npix;

       	int iifo, jifo;

	unsigned long Umap_len = (unsigned long) det_map->matrix_size[0] * (det_map->matrix_size)[1] * (det_map->matrix_size)[2];
	unsigned long Umatrix_len = (unsigned long) (det_map->matrix_size)[1] * (det_map->matrix_size)[2];
	printf("u len %lu \n", Umap_len);
	det_map->U_map = (float *)malloc(sizeof(float) * Umap_len);
	memset(det_map->U_map, 0, sizeof(float) * Umap_len);
	det_map->diff_map = (float *)malloc(sizeof(float) * Umap_len);
	memset(det_map->diff_map, 0, sizeof(float) * Umap_len);

	double theta, phi, fplus, fcross, gmst;
	unsigned long ipix;

	float *U_map = det_map->U_map, *diff_map = det_map->diff_map;
	double U[nifo*nifo], VT[2*2], S[2], diff[nifo*nifo], A[nifo*2], superb[min(nifo,2)-1];
	lapack_int lda = 2, ldu = nifo, ldvt = 2, info;


	unsigned long index = 0, igps = 0;

	for (igps=0, gps_cur.gpsSeconds=gps_start.gpsSeconds; gps_cur.gpsSeconds<gps_end.gpsSeconds; XLALGPSAdd(&gps_cur, ingps_step), igps++) {
		for (ipix=0; ipix<npix; ipix++) {
			for (iifo=0; iifo<nifo; iifo++) {

				/* ra = phi, dec = 2pi - theta */	
				pix2ang_nest(nside, ipix, &theta, &phi);
		
				/* get fplus, fcross from lalsuite DetResponse.c */

				gmst = XLALGreenwichMeanSiderealTime(&gps_cur);
				XLALComputeDetAMResponse(&fplus, &fcross, detectors[iifo]->response, phi, M_PI_2-theta, 0, gmst);
	
				A[iifo*2] = fplus*horizons[iifo];
				A[iifo*2 + 1] = fcross*horizons[iifo];
			}
#if 0
			for (i=0; i<6; i++)
				printf("ipix %d, ra %f, dec %f, A[%d] %f\n", ipix, phi, M_PI_2-theta, i, A[i]);
#endif
				
			info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', nifo, 2, A, lda, S, U, ldu, VT, ldvt, superb);

			if(info > 0) {
				printf("SVD of A matrix failed to converge. \n");
				exit(1);
			}

			/* TimeDelay.c */
			for (iifo=0; iifo<nifo; iifo++) 
				for (jifo=0; jifo<nifo; jifo++) 
					diff[iifo*nifo+jifo] = XLALArrivalTimeDiff(detectors[iifo]->location, detectors[jifo]->location, phi, M_PI_2-theta, &gps_cur);

			for (iifo=0; iifo<nifo*nifo; iifo++) {

				index = igps*Umatrix_len + iifo*npix +ipix;
				U_map[index] = (float) U[iifo];
				diff_map[index] = (float) diff[iifo];
		//		printf("index %d diff %f\n", index, diff[i]);
			}

		}

	}
	return det_map;
}

static int to_xml(DetSkymap *det_map, const char *detrsp_fname, const char *detrsp_header_string, int compression) 
{
    int rc;
    xmlTextWriterPtr writer;
    xmlChar *tmp;

    XmlArray *tmp_array = (XmlArray *)malloc(sizeof(XmlArray));

    tmp_array->ndim = 2;
    tmp_array->dim[0] = det_map->matrix_size[1];
    tmp_array->dim[1] = det_map->matrix_size[2];
    int array_len = tmp_array->dim[0] * tmp_array->dim[1];
    int array_size = sizeof(float) * array_len;

    tmp_array->data = (float*) malloc(array_size);
    /* Create a new XmlWriter for uri, with no compression. */
    writer = xmlNewTextWriterFilename(detrsp_fname, compression);
    if (writer == NULL) {
        printf("Error creating the xml writer\n");
        return;
    }

    rc = xmlTextWriterSetIndent(writer, 1);
    rc = xmlTextWriterSetIndentString(writer, BAD_CAST "\t");

    /* Start the document with the xml default for the version,
     * encoding utf-8 and the default for the standalone
     * declaration. */
    rc = xmlTextWriterStartDocument(writer, NULL, MY_ENCODING, NULL);
    if (rc < 0) {
        printf
            ("testXmlwriterFilename: Error at xmlTextWriterStartDocument\n");
        return;
    }

    rc = xmlTextWriterWriteDTD(writer, BAD_CAST "LIGO_LW", NULL, BAD_CAST "http://ldas-sw.ligo.caltech.edu/doc/ligolwAPI/html/ligolw_dtd.txt", NULL);
    if (rc < 0) {
        printf
            ("testXmlwriterFilename: Error at xmlTextWriterWriteDTD\n");
        return;
    }

    /* Start an element named "LIGO_LW". Since thist is the first
     * element, this will be the root element of the document. */
    rc = xmlTextWriterStartElement(writer, BAD_CAST "LIGO_LW");
    if (rc < 0) {
        printf
            ("testXmlwriterFilename: Error at xmlTextWriterStartElement\n");
        return;
    }

    /* Start an element named "LIGO_LW" as child of EXAMPLE. */
    rc = xmlTextWriterStartElement(writer, BAD_CAST "LIGO_LW");
    if (rc < 0) {
        printf
            ("testXmlwriterFilename: Error at xmlTextWriterStartElement\n");
        return;
    }

    /* Add an attribute with name "Name" and value detrsp_header_string to LIGO_LW. */
    rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "Name",
                                     BAD_CAST detrsp_header_string);
    if (rc < 0) {
        printf
            ("testXmlwriterFilename: Error at xmlTextWriterWriteAttribute\n");
        return;
    }

    XmlParam tmp_param;
    tmp_param.data = malloc(sizeof(int));
    *((int *)tmp_param.data) = det_map->gps_step;
    ligoxml_write_Param(writer, &tmp_param, BAD_CAST "int_4s", BAD_CAST "gps_step:param");

    free(tmp_param.data);
    tmp_param.data = NULL;

    tmp_param.data = malloc(sizeof(int));
    *((int *)tmp_param.data) = det_map->order;
    ligoxml_write_Param(writer, &tmp_param, BAD_CAST "int_4s", BAD_CAST "chealpix_order:param");

    free(tmp_param.data);
    tmp_param.data = NULL;

    gchar gps_name[40];
    int i;
    for(i=0; i<det_map->matrix_size[0]; i++) {

	memcpy(tmp_array->data, det_map->U_map + i*array_len, array_size);
	sprintf(gps_name, "U_map_gps_%d:array", det_map->gps_step*i);
    	ligoxml_write_Array(writer, tmp_array, BAD_CAST "real_4", BAD_CAST " ", BAD_CAST gps_name);
	memcpy(tmp_array->data, det_map->diff_map + i*array_len, array_size);
	sprintf(gps_name, "diff_map_gps_%d:array", det_map->gps_step*i);
    	ligoxml_write_Array(writer, tmp_array, BAD_CAST "real_4", BAD_CAST " ", BAD_CAST gps_name);

    }

    rc = xmlTextWriterEndDocument(writer);
    if (rc < 0) {
        printf
            ("testXmlwriterFilename: Error at xmlTextWriterEndDocument\n");
        return;
    }

    xmlFreeTextWriter(writer);

	return 0;
}

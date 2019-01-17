/* 
 * Copyright (C) 2014 Qi Chu <qi.chu@ligo.org>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

/* This element will synchronize the snr sequencies from all detectors, find 
 * peaks from all detectors and for each peak, do null stream analysis.
 */
#include <stdlib.h>
#include <getopt.h>
#include <postcoh/postcoh_utils.h>
#include <gst/gst.h>
#include <math.h>
#include <pipe_macro.h>
//

#include <lal/LIGOMetadataTables.h> // SnglInspiralTable
#include <LIGOLwHeader.h>
#include <chealpix.h>
#include <lal/Date.h>
#include <lal/LALDetectors.h>
#include <lal/DetResponse.h>
#include <lal/TimeDelay.h>
#include <lal/LALSimulation.h>

#ifdef SPIIR_HAVE_LAPACKE
#include <lapacke/lapacke_config.h>
#include <lapacke/lapacke.h>
#else
#include <lapacke_config.h>
#include <lapacke.h>
#endif

//#include <LIGOLw_xmllib/LIGOLwHeader.h>
#include <math.h>
#define min(a,b) ((a)>(b)?(b):(a))

typedef struct _RspSkymap {
	char **ifos;
	int nifo;
	int gps_step;
	long gps_start;
	unsigned order;
	float *U_map;
	float *diff_map;
	float *Det_map; // det(G.T*G) = s1^2*s2^2  
	int matrix_size[3];
} RspSkymap;


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
		// printf("ifo%d %s\n", i, ifos[i]);
		detector = XLALDetectorPrefixToLALDetector(ifos[i]);
		detectors[i] = (LALDetector *)malloc(sizeof(LALDetector));
		memcpy(detectors[i], detector, sizeof(LALDetector));
	}
	return detectors;
}

RspSkymap *
create_detresponse_skymap(
		char **ifos,
		int nifo,
		double *horizons,
		int ingps_step,
		unsigned order,
		long gps
		)
{
	RspSkymap * rsp_map = (RspSkymap *)malloc(sizeof(RspSkymap));
	// from 0h UTC 6 Jan 1980
	// LIGOTimeGPS gps_cur = {0, 0}; 
	LIGOTimeGPS gps_cur = {gps, 0}; 
	// current time
	//LIGOTimeGPS *out_cur = XLALGPSTimeNow(&gps_cur);
	//if (out_cur == NULL)
	//	printf("can not find current gps time");
	// printf("current gps time %d, %d for detector response\n", gps_cur.gpsSeconds, gps_cur.gpsNanoSeconds);
	LIGOTimeGPS gps_start = {gps_cur.gpsSeconds, 0}; 
	LIGOTimeGPS gps_end = {gps_cur.gpsSeconds + 24*3600, 0};
	LIGOTimeGPS gps_step = {ingps_step, 0}; 

	int ngps;
	ngps = (int)(gps_end.gpsSeconds - gps_start.gpsSeconds) / gps_step.gpsSeconds;

	rsp_map->gps_start = gps_cur.gpsSeconds;
	rsp_map->gps_step = ingps_step;
	rsp_map->matrix_size[0] = ngps;

	LALDetector* const* detectors = create_detectors_from_name(ifos, nifo);
	rsp_map->nifo = nifo;
	rsp_map->matrix_size[1] = nifo * nifo;

	unsigned long nside = (unsigned long) 1 << order, npix = nside2npix(nside);
	rsp_map->order = order;
	rsp_map->matrix_size[2] = npix;

    int iifo, jifo;

	unsigned long Umap_len = (unsigned long) rsp_map->matrix_size[0] * (rsp_map->matrix_size)[1] * (rsp_map->matrix_size)[2];
	unsigned long Umatrix_len = (unsigned long) (rsp_map->matrix_size)[1] * (rsp_map->matrix_size)[2];
	unsigned long Detmap_len = (unsigned long) (rsp_map->matrix_size)[0] * (rsp_map->matrix_size)[2];
	// printf("u len %lu \n", Umap_len);
	rsp_map->U_map = (float *)malloc(sizeof(float) * Umap_len);
	memset(rsp_map->U_map, 0, sizeof(float) * Umap_len);
	rsp_map->diff_map = (float *)malloc(sizeof(float) * Umap_len);
	memset(rsp_map->diff_map, 0, sizeof(float) * Umap_len);
	rsp_map->Det_map = (float *)malloc(sizeof(float) * Detmap_len);
	memset(rsp_map->Det_map, 0, sizeof(float) * Detmap_len);


	double theta, phi, fplus, fcross, gmst;
	unsigned long ipix;

	double U[nifo*nifo], VT[2*2], S[2], diff[nifo*nifo], A[nifo*2], superb[min(nifo,2)-1];
	lapack_int lda = 2, ldu = nifo, ldvt = 2, info;


	unsigned long index = 0, igps = 0;

	for (igps=0, gps_cur.gpsSeconds=gps_start.gpsSeconds; gps_cur.gpsSeconds<gps_end.gpsSeconds; XLALGPSAdd(&gps_cur, ingps_step), igps++) {
		for (ipix=0; ipix<npix; ipix++) {
			for (iifo=0; iifo<nifo; iifo++) {

				/* ra = phi, dec = pi/2 - theta , phi: longitude 0-2PI, theta colatitude 0-PI, see bayestar lalinference plot.py healpix_lookup */	
				pix2ang_nest(nside, ipix, &theta, &phi);
		
				/* get fplus, fcross from lalsuite DetResponse.c */

				gmst = XLALGreenwichMeanSiderealTime(&gps_cur);
				/* polarization angle---psi must be set to zero */
				XLALComputeDetAMResponse(&fplus, &fcross, detectors[iifo]->response, phi, M_PI_2-theta, 0, gmst);
	
				A[iifo*2] = fplus*horizons[iifo];
				A[iifo*2 + 1] = fcross*horizons[iifo];
				// printf("igps %d, ipix %d, ra %f, dec %f, iifo %d, fplus %f, fcross %f, horizon %f \n", gps_cur.gpsSeconds, ipix, phi, M_PI_2-theta, iifo, fplus, fcross, horizons[iifo]);


			}

			/* TimeDelay.c */
			/* note that in the c file, the function is calculating arrival_time from input1 - arrival_time from input2 */
			for (iifo=0; iifo<nifo; iifo++) 
				for (jifo=0; jifo<nifo; jifo++) 
					diff[iifo*nifo+jifo] = XLALArrivalTimeDiff(detectors[jifo]->location, detectors[iifo]->location, phi, M_PI_2-theta, &gps_cur);
			// printf("igps %d, ipix %d, ra %f, dec %f, A %f, %f, %f, %f, %f, %f, diff %f, %f, %f, %f, %f, %f\n", gps_cur.gpsSeconds, ipix, phi, M_PI_2-theta, A[0], A[1], A[2], A[3], A[4], A[5], diff[0], diff[1], diff[2], diff[3], diff[4], diff[5]);
		
			/* compute SVD of A */
			info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', nifo, 2, A, lda, S, U, ldu, VT, ldvt, superb);

			if(info > 0) {
				printf("SVD of A matrix failed to converge. \n");
				exit(1);
			}


			/* save the U and diff into U_map and diff_map */
			for (iifo=0; iifo<nifo*nifo; iifo++) {

				index = igps*Umatrix_len + iifo*npix +ipix;
				rsp_map->U_map[index] = (float) U[iifo];
				rsp_map->diff_map[index] = (float) diff[iifo];
				//printf("index %d diff %f\n", index, diff[i]);
			}
			/* save Det map */
			rsp_map->Det_map[igps*npix + ipix] = S[0]*S[0]*S[1]*S[1]; // det(G.T*G)

		}

	}
	return rsp_map;
}

static int to_xml(RspSkymap *rsp_map, const char *detrsp_fname, const char *detrsp_header_string, int is_coh, int compression) 
{
    int rc;
    xmlTextWriterPtr writer;
    xmlChar *tmp;

    XmlArray *tmp_array = (XmlArray *)malloc(sizeof(XmlArray));

    tmp_array->ndim = 2;
    tmp_array->dim[0] = rsp_map->matrix_size[1]; // nifo^2
    tmp_array->dim[1] = rsp_map->matrix_size[2]; // npix
    int U_array_len = tmp_array->dim[0] * tmp_array->dim[1];
    int U_array_size = sizeof(float) * U_array_len;
	int Det_array_len = tmp_array->dim[1]; // npix
    int Det_array_size = sizeof(float) * Det_array_len;

    tmp_array->data = (float*) malloc(U_array_size);
    /* Create a new XmlWriter for uri, with no compression. */
    writer = xmlNewTextWriterFilename(detrsp_fname, compression);
    if (writer == NULL) {
        printf("Error creating the xml writer\n");
        return -1;
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
        return rc;
    }

    rc = xmlTextWriterWriteDTD(writer, BAD_CAST "LIGO_LW", NULL, BAD_CAST "http://ldas-sw.ligo.caltech.edu/doc/ligolwAPI/html/ligolw_dtd.txt", NULL);
    if (rc < 0) {
        printf
            ("testXmlwriterFilename: Error at xmlTextWriterWriteDTD\n");
        return rc;
    }

    /* Start an element named "LIGO_LW". Since thist is the first
     * element, this will be the root element of the document. */
    rc = xmlTextWriterStartElement(writer, BAD_CAST "LIGO_LW");
    if (rc < 0) {
        printf
            ("testXmlwriterFilename: Error at xmlTextWriterStartElement\n");
        return rc;
    }

    /* Start an element named "LIGO_LW" as child of EXAMPLE. */
    rc = xmlTextWriterStartElement(writer, BAD_CAST "LIGO_LW");
    if (rc < 0) {
        printf
            ("testXmlwriterFilename: Error at xmlTextWriterStartElement\n");
        return rc;
    }

    /* Add an attribute with name "Name" and value detrsp_header_string to LIGO_LW. */
    rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "Name",
                                     BAD_CAST detrsp_header_string);
    if (rc < 0) {
        printf
            ("testXmlwriterFilename: Error at xmlTextWriterWriteAttribute\n");
        return rc;
    }

    /* gps time is half way through int_4 limit, use int_8 for this */
    XmlParam tmp_param;
    tmp_param.data = malloc(sizeof(long));
    *((long *)tmp_param.data) = rsp_map->gps_start;
    ligoxml_write_Param(writer, &tmp_param, BAD_CAST "int_8s", BAD_CAST "gps_start:param");

    free(tmp_param.data);
    tmp_param.data = NULL;


    tmp_param.data = malloc(sizeof(int));
    *((int *)tmp_param.data) = rsp_map->gps_step;
    ligoxml_write_Param(writer, &tmp_param, BAD_CAST "int_4s", BAD_CAST "gps_step:param");

    free(tmp_param.data);
    tmp_param.data = NULL;

    tmp_param.data = malloc(sizeof(int));
    *((int *)tmp_param.data) = rsp_map->order;
    ligoxml_write_Param(writer, &tmp_param, BAD_CAST "int_4s", BAD_CAST "chealpix_order:param");

    free(tmp_param.data);
    tmp_param.data = NULL;

    gchar gps_name[40];
    int i;
	if (1 == is_coh){
	    for(i=0; i<rsp_map->matrix_size[0]; i++) { // igps
	
			tmp_array->dim[0] = rsp_map->matrix_size[1];
			memcpy(tmp_array->data, rsp_map->U_map + i*U_array_len, U_array_size);
			sprintf(gps_name, "U_map_gps_%d:array", rsp_map->gps_step*i);
		    ligoxml_write_Array(writer, tmp_array, BAD_CAST "real_4", BAD_CAST " ", BAD_CAST gps_name);
			memcpy(tmp_array->data, rsp_map->diff_map + i*U_array_len, U_array_size);
			sprintf(gps_name, "diff_map_gps_%d:array", rsp_map->gps_step*i);
		    ligoxml_write_Array(writer, tmp_array, BAD_CAST "real_4", BAD_CAST " ", BAD_CAST gps_name);
		}
	} else {
	    for(i=0; i<rsp_map->matrix_size[0]; i++) { // igps
			tmp_array->dim[0] = 1;
			memcpy(tmp_array->data, rsp_map->Det_map + i*Det_array_len, Det_array_size);
			sprintf(gps_name, "Det_map_gps_%d:array", rsp_map->gps_step*i);
			ligoxml_write_Array(writer, tmp_array, BAD_CAST "real_4", BAD_CAST " ", BAD_CAST gps_name);
		}
    }

    rc = xmlTextWriterEndDocument(writer);
    if (rc < 0) {
        printf
            ("testXmlwriterFilename: Error at xmlTextWriterEndDocument\n");
        return rc;
    }

    xmlFreeTextWriter(writer);

	return 0;
}
static void parse_opts(int argc, char *argv[], gchar **pin, gchar **pnorder, gchar **pgps, gchar **pout_coh, gchar **pout_prob)
{
	int option_index = 0;
	struct option long_opts[] =
	{
		{"ifo-horizons",	required_argument,	0,	'i'},
		{"chealpix-order",	required_argument,	0,	'n'},
		{"output-coh-coeff",	required_argument,	0,	'c'},
		{"output-prob-coeff",	required_argument,	0,	'p'},
		{"gps-time",		required_argument,	0,	'g'},
		{0, 0, 0, 0}
	};
	int opt;
	while ((opt = getopt_long(argc, argv, "i:n:o:g:", long_opts, &option_index)) != -1) {
		switch (opt) {
			case 'i':
				*pin = g_strdup((gchar *)optarg);
				break;
			case 'n':
				*pnorder = g_strdup((gchar *)optarg);
				break;
			case 'g':
				*pgps = g_strdup((gchar *)optarg);
				break;
			case 'c':
				*pout_coh = g_strdup((gchar *)optarg);
				break;
			case 'p':
				*pout_prob = g_strdup((gchar *)optarg);
				break;
			default:
				exit(0);
		}
	}
}

int main(int argc, char *argv[])
{
	gchar **pin = (gchar **)malloc(sizeof(gchar *));
	gchar **pnorder = (gchar **)malloc(sizeof(gchar *));
	gchar **pgps = (gchar **)malloc(sizeof(gchar *));
	gchar **pout_coh = (gchar **)malloc(sizeof(gchar *));
	gchar **pout_prob = (gchar **)malloc(sizeof(gchar *));

	parse_opts(argc, argv, pin, pnorder, pgps, pout_coh, pout_prob);
	
	gchar ** in_ifo_strings = g_strsplit(*pin, ",", -1);
	gchar ** one_ifo_string = NULL;
	int nifo = 0;
	for (one_ifo_string = in_ifo_strings; *one_ifo_string; one_ifo_string++)
		nifo++;

	char **ifo_names = (char**)malloc(nifo*sizeof(char*));
	/* equivalent to sigma in Chichi's thesis */
	double *horizons = (double*)malloc(nifo*sizeof(double));
	char ** tmp_str = NULL;
	GString *mapname = g_string_new(DETRSP_XML_ID_NAME);
	int iifo = 0;
	/* decode the ifo names and horizons from input e.g. H1:26,L1:52,V1:10 */
	for(iifo=0;iifo<nifo;iifo++) {
		ifo_names[iifo] = (char*)malloc(sizeof(char)*4);
		tmp_str = g_strsplit(in_ifo_strings[iifo], ":", 2);
		/* FIXME: hard-coded copy buffer size =4 */
		g_strlcpy(ifo_names[iifo], tmp_str[0], 4);
		horizons[iifo] = (double) atoi(tmp_str[1]);
	}

	/* GPS interval = 1800 seconds.
	 * norder is 4, for depth of number of pixels created  */
	int norder = atoi(*pnorder);
	// FIXME: can atoi convert to long ?
	long gps = atoi(*pgps);
	RspSkymap *rsp_map = create_detresponse_skymap(ifo_names, nifo, horizons, 1800 ,norder, gps);

   	GString *tmp_fname_coh = g_string_new(*pout_coh);
    g_string_append_printf(tmp_fname_coh, "_next");
	int is_coh = 1;
 
	if (to_xml(rsp_map, tmp_fname_coh->str, mapname->str, is_coh, 0) < 0)
		return -1;

	if (g_rename(tmp_fname_coh->str, *pout_coh) != 0) {
		fprintf(stderr, "unable to rename to %s", *pout_coh);
		return -1;
	}

   	GString *tmp_fname_prob = g_string_new(*pout_prob);
    g_string_append_printf(tmp_fname_prob, "_next");

	is_coh = 0;
	if (to_xml(rsp_map, tmp_fname_prob->str, mapname->str, is_coh, 0) < 0)
		return -1;


	if (g_rename(tmp_fname_prob->str, *pout_prob) != 0) {
		fprintf(stderr, "unable to rename to %s", *pout_prob);
		return -1;
	}

	g_string_free(mapname, TRUE);
	g_string_free(tmp_fname_coh, TRUE);
	g_string_free(tmp_fname_prob, TRUE);

	for(iifo=0;iifo<nifo;iifo++) 
		free(ifo_names[iifo]);

	free(ifo_names);
	free(horizons);

	return 0;
}


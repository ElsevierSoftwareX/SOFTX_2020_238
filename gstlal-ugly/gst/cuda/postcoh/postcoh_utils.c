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



#include <lal/LIGOMetadataTables.h> // SnglInspiralTable
#include <LIGOLw_xmllib/LIGOLwHeader.h>
#include <chealpix.h>
#include <postcoh/postcoh_utils.h>
#include <cuda_debug.h>

char* IFO_MAP[] = {"L1", "H1", "V1"};
#define __DEBUG__ 1
#define NSNGL_TMPLT_COLS 12

#include <lal/Date.h>
#include <lal/LALDetectors.h>
#include <lal/DetResponse.h>
#include <lal/TimeDelay.h>
#include <lal/LALSimulation.h>

#include <lapacke/lapacke_config.h>
#include <lapacke/lapacke.h>

//#include <LIGOLw_xmllib/LIGOLwHeader.h>
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

				/* ra = phi, dec = pi/2 - theta , phi: longitude 0-2PI, theta colatitude 0-PI, see bayestar lalinference plot.py healpix_lookup */	
				pix2ang_nest(nside, ipix, &theta, &phi);
		
				/* get fplus, fcross from lalsuite DetResponse.c */

				gmst = XLALGreenwichMeanSiderealTime(&gps_cur);
				/* polarization angle---psi must be set to zero */
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
PeakList *create_peak_list(PostcohState *state, cudaStream_t stream)
{
		int hist_trials = state->hist_trials;
		g_assert(hist_trials != -1);
		int max_npeak = state->max_npeak;
		PeakList *pklist = (PeakList *)malloc(sizeof(PeakList));

		int peak_intlen = (7 + hist_trials) * max_npeak + 1;
		int peak_floatlen = (12 + hist_trials * 12 ) * max_npeak;
		pklist->peak_intlen = peak_intlen;
		pklist->peak_floatlen = peak_floatlen;
		
		CUDA_CHECK(cudaMalloc((void **) &(pklist->d_npeak), sizeof(int) * peak_intlen ));
		CUDA_CHECK(cudaMemsetAsync(pklist->d_npeak, 0, sizeof(int) * peak_intlen, stream));
		pklist->d_peak_pos = pklist->d_npeak + 1;
		pklist->d_len_idx = pklist->d_npeak + 1 + max_npeak;
		pklist->d_tmplt_idx = pklist->d_npeak + 1 + 2 * max_npeak;
		pklist->d_pix_idx = pklist->d_npeak + 1 + 3 * max_npeak;
		pklist->d_pix_idx_bg = pklist->d_npeak + 1 + 4 * max_npeak;
		pklist->d_ntoff_L = pklist->d_npeak + 1 + (4 + hist_trials) * max_npeak;
		pklist->d_ntoff_H = pklist->d_npeak + 1 + (5 + hist_trials) * max_npeak;
		pklist->d_ntoff_V = pklist->d_npeak + 1 + (6 + hist_trials) * max_npeak;

		//printf("d_npeak %p\n", pklist->d_npeak);
		//CUDA_CHECK(cudaMemsetAsync(pklist->d_npeak, 0, sizeof(int), stream));

		CUDA_CHECK(cudaMalloc((void **) &(pklist->d_snglsnr_L), sizeof(float) * peak_floatlen));
		CUDA_CHECK(cudaMemsetAsync(pklist->d_snglsnr_L, 0, sizeof(float) * peak_floatlen, stream));
		pklist->d_snglsnr_H = pklist->d_snglsnr_L + max_npeak;
		pklist->d_snglsnr_V = pklist->d_snglsnr_L + 2 * max_npeak;
		pklist->d_coaphase_L = pklist->d_snglsnr_L + 3 * max_npeak;
		pklist->d_coaphase_H = pklist->d_snglsnr_L + 4 * max_npeak;
		pklist->d_coaphase_V = pklist->d_snglsnr_L + 5 * max_npeak;
		pklist->d_chisq_L = pklist->d_snglsnr_L + 6 * max_npeak;
		pklist->d_chisq_H = pklist->d_snglsnr_L + 7 * max_npeak;
		pklist->d_chisq_V = pklist->d_snglsnr_L + 8 * max_npeak;
		pklist->d_cohsnr = pklist->d_snglsnr_L + 9 * max_npeak;
		pklist->d_nullsnr = pklist->d_snglsnr_L + 10 * max_npeak;
		pklist->d_cmbchisq = pklist->d_snglsnr_L + 11 * max_npeak;
	
		pklist->d_snglsnr_bg_L = pklist->d_snglsnr_L + 12 * max_npeak;
		pklist->d_snglsnr_bg_H = pklist->d_snglsnr_L + (12 + hist_trials) * max_npeak;
		pklist->d_snglsnr_bg_V = pklist->d_snglsnr_L + (12 + 2* hist_trials) * max_npeak;
		pklist->d_coaphase_bg_L = pklist->d_snglsnr_L + (12 + 3*hist_trials) * max_npeak;
		pklist->d_coaphase_bg_H = pklist->d_snglsnr_L + (12 + 4*hist_trials) * max_npeak;
		pklist->d_coaphase_bg_V = pklist->d_snglsnr_L + (12 + 5*hist_trials) * max_npeak;
		pklist->d_chisq_bg_L = pklist->d_snglsnr_L + (12 + 6*hist_trials) * max_npeak;
		pklist->d_chisq_bg_H = pklist->d_snglsnr_L + (12 + 7*hist_trials) * max_npeak;
		pklist->d_chisq_bg_V = pklist->d_snglsnr_L + (12 + 8*hist_trials) * max_npeak;
		pklist->d_cohsnr_bg = pklist->d_snglsnr_L + (12 + 9*hist_trials) * max_npeak;
		pklist->d_nullsnr_bg = pklist->d_snglsnr_L + (12 + 10*hist_trials) * max_npeak;
		pklist->d_cmbchisq_bg = pklist->d_snglsnr_L + (12 + 11 * hist_trials) * max_npeak;

		//pklist->npeak = (int *)malloc(sizeof(int) * peak_intlen);
		CUDA_CHECK(cudaMallocHost((void **) &(pklist->npeak), sizeof(int) * peak_intlen));
		memset(pklist->npeak, 0, sizeof(int) * peak_intlen);
		pklist->peak_pos = pklist->npeak + 1;
		pklist->len_idx = pklist->npeak + 1 + max_npeak;
		pklist->tmplt_idx = pklist->npeak + 1 + 2 * max_npeak;
		pklist->pix_idx = pklist->npeak + 1 + 3 * max_npeak;
		pklist->pix_idx_bg = pklist->npeak + 1 + 4 * max_npeak;
		pklist->ntoff_L = pklist->npeak + 1 + (4 + hist_trials) * max_npeak;
		pklist->ntoff_H = pklist->npeak + 1 + (5 + hist_trials) * max_npeak;
		pklist->ntoff_V = pklist->npeak + 1 + (6 + hist_trials) * max_npeak;

		//pklist->snglsnr_L = (float *)malloc(sizeof(float) * peak_floatlen);
		CUDA_CHECK(cudaMallocHost((void **) &(pklist->snglsnr_L), sizeof(float) * peak_floatlen));
		memset(pklist->snglsnr_L, 0, sizeof(float) * peak_floatlen);
		pklist->snglsnr_H = pklist->snglsnr_L + max_npeak;
		pklist->snglsnr_V = pklist->snglsnr_L + 2 * max_npeak;
		pklist->coaphase_L = pklist->snglsnr_L + 3 * max_npeak;
		pklist->coaphase_H = pklist->snglsnr_L + 4 * max_npeak;
		pklist->coaphase_V = pklist->snglsnr_L + 5 * max_npeak;
		pklist->chisq_L = pklist->snglsnr_L + 6 * max_npeak;
		pklist->chisq_H = pklist->snglsnr_L + 7 * max_npeak;
		pklist->chisq_V = pklist->snglsnr_L + 8 * max_npeak;
		pklist->cohsnr = pklist->snglsnr_L + 9 * max_npeak;
		pklist->nullsnr = pklist->snglsnr_L + 10 * max_npeak;
		pklist->cmbchisq = pklist->snglsnr_L + 11 * max_npeak;
//	
		pklist->snglsnr_bg_L = pklist->snglsnr_L + 12 * max_npeak;
		pklist->snglsnr_bg_H = pklist->snglsnr_L + (12 + hist_trials) * max_npeak;
		pklist->snglsnr_bg_V = pklist->snglsnr_L + (12 + 2* hist_trials) * max_npeak;
		pklist->coaphase_bg_L = pklist->snglsnr_L + (12 + 3*hist_trials) * max_npeak;
		pklist->coaphase_bg_H = pklist->snglsnr_L + (12 + 4*hist_trials) * max_npeak;
		pklist->coaphase_bg_V = pklist->snglsnr_L + (12 + 5*hist_trials) * max_npeak;
		pklist->chisq_bg_L = pklist->snglsnr_L + (12 + 6*hist_trials) * max_npeak;
		pklist->chisq_bg_H = pklist->snglsnr_L + (12 + 7*hist_trials) * max_npeak;
		pklist->chisq_bg_V = pklist->snglsnr_L + (12 + 8*hist_trials) * max_npeak;
		pklist->cohsnr_bg = pklist->snglsnr_L + (12 + 9*hist_trials) * max_npeak;
		pklist->nullsnr_bg = pklist->snglsnr_L + (12 + 10*hist_trials) * max_npeak;
		pklist->cmbchisq_bg = pklist->snglsnr_L + (12 + 11 * hist_trials) * max_npeak;

//		printf("set peak addr %p, d_npeak addr %p\n", pklist, pklist->d_npeak);
		//printf("hist trials %d, peak_intlen %d, peak_floatlen %d\n", hist_trials, peak_intlen, peak_floatlen);
		/* temporary struct to store tmplt max in one max_npeak data */
		CUDA_CHECK(cudaMalloc((void **)&(pklist->d_peak_tmplt), sizeof(float) * state->ntmplt));
		CUDA_CHECK(cudaMemsetAsync(pklist->d_peak_tmplt, 0, sizeof(float) * state->ntmplt, stream));

		pklist->d_cohsnr_skymap = NULL;
		pklist->cohsnr_skymap = NULL;
		return pklist;
}

void
cuda_postcoh_map_from_xml(char *fname, PostcohState *state, cudaStream_t stream)
{
	// FIXME: sanity check that the size of U matrix and diff matrix for
	// each sky pixel is consistent with number of detectors
	//printf("read map from xml\n");
	/* first get the params */
	XmlNodeStruct *xns = (XmlNodeStruct *)malloc(sizeof(XmlNodeStruct) * 2);
	XmlParam param_gps = {0, NULL};
	XmlParam param_order = {0, NULL};

	sprintf((char *)xns[0].tag, "gps_step:param");
	xns[0].processPtr = readParam;
	xns[0].data = &param_gps;

	sprintf((char *)xns[1].tag, "chealpix_order:param");
	xns[1].processPtr = readParam;
	xns[1].data = &param_order;

	parseFile(fname, xns, 2);
	/*
	 * Cleanup function for the XML library.
	 */
	xmlCleanupParser();
	/*
	 * this is to debug memory for regression tests
	 */
	xmlMemoryDump();


	//printf("test\n");
	printf("%s \n", xns[0].tag);

	printf("%p\n", param_gps.data);
	state->gps_step = *((int *)param_gps.data);
	printf("gps_step %d\n", state->gps_step);
	unsigned long nside = (unsigned long) 1 << *((int *)param_order.data);
	state->nside = nside;
	state->npix = nside2npix(nside);
	free(param_gps.data);
	param_gps.data = NULL;
	//printf("test\n");
	free(param_order.data);
	param_order.data = NULL;
	free(xns);


	int gps = 0, gps_start = 0, gps_end = 24*3600;
	int ngps = gps_end/(state->gps_step);

	xns = (XmlNodeStruct *)malloc(sizeof(XmlNodeStruct) * 2* ngps);
	state->d_U_map = (float**)malloc(sizeof(float *) * ngps);
	state->d_diff_map = (float**)malloc(sizeof(float *) * ngps);

	int i;
	XmlArray *array_u = (XmlArray *)malloc(sizeof(XmlArray) * ngps);
	XmlArray *array_diff = (XmlArray *)malloc(sizeof(XmlArray) * ngps);

	for (i=0; i<ngps; i++) {

		sprintf((char *)xns[i].tag, "U_map_gps_%d:array", gps);
		//printf("%s\n", xns[i].tag);
		xns[i].processPtr = readArray;
		xns[i].data = &(array_u[i]);

		sprintf((char *)xns[i+ngps].tag, "diff_map_gps_%d:array", gps);
		xns[i+ngps].processPtr = readArray;
		xns[i+ngps].data = &(array_diff[i]);
		gps += state->gps_step; 
	}

	parseFile(fname, xns, 2*ngps);

	int mem_alloc_size = sizeof(float) * array_u[0].dim[0] * array_u[0].dim[1];
	for (i=0; i<ngps; i++) {
		CUDA_CHECK(cudaMalloc((void **)&(state->d_U_map[i]), mem_alloc_size));
		CUDA_CHECK(cudaMemcpyAsync(state->d_U_map[i], array_u[i].data, mem_alloc_size, cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMalloc((void **)&(state->d_diff_map[i]), mem_alloc_size));
		CUDA_CHECK(cudaMemcpyAsync(state->d_diff_map[i], array_diff[i].data, mem_alloc_size, cudaMemcpyHostToDevice, stream));

	}
	/*
	 * Cleanup function for the XML library.
	 */
	xmlCleanupParser();
	/*
	 * this is to debug memory for regression tests
	 */
	xmlMemoryDump();

	for (i=0; i<ngps; i++) {
		free(array_u[i].data);
		free(array_diff[i].data);
	}
}

void
cuda_postcoh_autocorr_from_xml(char *fname, PostcohState *state, cudaStream_t stream)
{
	//printf("read autocorr from xml\n");

	int ntoken = 0;

	char *end_ifo, *fname_cpy = (char *)malloc(sizeof(char) * strlen(fname));
	strcpy(fname_cpy, fname);
	char *token = strtok_r(fname, ",", &end_ifo);
	int mem_alloc_size = 0, autochisq_len = 0, ntmplt = 0, match_ifo = 0;

	/* parsing for nifo */
	while (token != NULL) {
		token = strtok_r(NULL, ",", &end_ifo);
		ntoken++;
	}

	int nifo = ntoken;
	XmlNodeStruct *xns = (XmlNodeStruct *)malloc(sizeof(XmlNodeStruct) * 2);
	XmlArray *array_autocorr = (XmlArray *)malloc(sizeof(XmlArray) * 2);

	COMPLEX_F *tmp_autocorr = NULL;
	float *tmp_norm = NULL;
	COMPLEX_F **autocorr = (COMPLEX_F **)malloc(sizeof(COMPLEX_F *) * nifo );
	float **autocorr_norm = (float **)malloc(sizeof(float *) * nifo );
	cudaMalloc((void **)&(state->dd_autocorr_matrix), sizeof(COMPLEX_F *) * nifo);
	cudaMalloc((void **)&(state->dd_autocorr_norm), sizeof(float *) * nifo);

	end_ifo = NULL;
	token = strtok_r(fname_cpy, ",", &end_ifo);
	//printf("fname_cpy %s\n", fname_cpy);
	sprintf((char *)xns[0].tag, "autocorrelation_bank_real:array");
	xns[0].processPtr = readArray;
	xns[0].data = &(array_autocorr[0]);

	sprintf((char *)xns[1].tag, "autocorrelation_bank_imag:array");
	xns[1].processPtr = readArray;
	xns[1].data = &(array_autocorr[1]);

	/* start parsing again */
	while (token != NULL) {
		char *end_token;
		char *token_bankname = strtok_r(token, ":", &end_token);
		token_bankname = strtok_r(NULL, ":", &end_token);

		parseFile(token_bankname, xns, 2);

		for (int i=0; i<nifo; i++) {
			if (strncmp(token, IFO_MAP[i], 2) == 0) {
				match_ifo = i;
				break;
			}
		
		}

		ntmplt = array_autocorr[0].dim[1];
		autochisq_len = array_autocorr[0].dim[0];

		//printf("parse match ifo %d, %s, ntmplt %d, auto_len %d\n", match_ifo, token_bankname, ntmplt, autochisq_len);
		mem_alloc_size = sizeof(COMPLEX_F) * ntmplt * autochisq_len;
		CUDA_CHECK(cudaMalloc((void **)&(autocorr[match_ifo]), mem_alloc_size));
		CUDA_CHECK(cudaMalloc((void **)&(autocorr_norm[match_ifo]), sizeof(float) * ntmplt));

		if (tmp_autocorr == NULL) {
			tmp_autocorr = (COMPLEX_F *)malloc(mem_alloc_size);
			tmp_norm = (float *)malloc(sizeof(float) * ntmplt);
		}

		float tmp_re = 0.0, tmp_im = 0.0;

		memset(tmp_norm, 0, sizeof(float) * ntmplt);
		for (int j=0; j<ntmplt; j++) {
			for (int k=0; k<autochisq_len; k++) {
				tmp_re = (float)((double *)(array_autocorr[0].data))[k * ntmplt + j];
				tmp_im = (float)((double *)(array_autocorr[1].data))[k * ntmplt + j];
				tmp_autocorr[j * autochisq_len + k].re = tmp_re;
				tmp_autocorr[j * autochisq_len + k].im = tmp_im;
				tmp_norm[j] += 2 - (tmp_re * tmp_re + tmp_im * tmp_im);
			}
//			printf("match ifo %d, norm %d: %f\n", match_ifo, j, tmp_norm[j]);
		}

		CUDA_CHECK(cudaMemcpyAsync(autocorr[match_ifo], tmp_autocorr, mem_alloc_size, cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(&(state->dd_autocorr_matrix[match_ifo]), &(autocorr[match_ifo]), sizeof(COMPLEX_F *), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(autocorr_norm[match_ifo], tmp_norm, sizeof(float) * ntmplt, cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(&(state->dd_autocorr_norm[match_ifo]), &(autocorr_norm[match_ifo]), sizeof(float *), cudaMemcpyHostToDevice, stream));

		freeArraydata(array_autocorr);
		freeArraydata(array_autocorr+1);
		token = strtok_r(NULL, ",", &end_ifo);
		/*
		 * Cleanup function for the XML library.
		 */
		xmlCleanupParser();
		/*
		 * this is to debug memory for regression tests
		 */
		xmlMemoryDump();

//		printf("next token %s \n", token);

	}

	free(tmp_autocorr);
	free(tmp_norm);
	state->autochisq_len = autochisq_len;
}

char * ColNames[] = {
	"sngl_inspiral:template_duration", 
	"sngl_inspiral:mass1",
	"sngl_inspiral:mass2",
	"sngl_inspiral:mchirp",
	"sngl_inspiral:mtotal",
	"sngl_inspiral:spin1x",
	"sngl_inspiral:spin1y",
	"sngl_inspiral:spin1z",
	"sngl_inspiral:spin2x",
	"sngl_inspiral:spin2y",
	"sngl_inspiral:spin2z",
	"sngl_inspiral:eta"
	};

void
cuda_postcoh_sngl_tmplt_from_xml(char *fname, SnglInspiralTable **psngl_table)
{

	XmlNodeStruct *xns = (XmlNodeStruct *) malloc(sizeof(XmlNodeStruct));
	XmlTable *xtable = (XmlTable *) malloc(sizeof(XmlTable));

	xtable->names = NULL;
	xtable->type_names = NULL;

	strncpy((char *) xns->tag, "sngl_inspiral:table", XMLSTRMAXLEN);
	xns->processPtr = readTable;
	xns->data = xtable;

	parseFile(fname, xns, 1);

    /*
     * Cleanup function for the XML library.
     */
    xmlCleanupParser();
    /*
     * this is to debug memory for regression tests
     */
    xmlMemoryDump();

	GHashTable *hash = xtable->hashContent;
	GString **col_names = (GString **) malloc(sizeof(GString *) * NSNGL_TMPLT_COLS);
	unsigned icol, jlen;
	for (icol=0; icol<NSNGL_TMPLT_COLS; icol++) {
		col_names[icol] = g_string_new(ColNames[icol]);
	}
	XmlHashVal *val = g_hash_table_lookup(hash, (gpointer) col_names[0]);
	*psngl_table = (SnglInspiralTable *) malloc(sizeof(SnglInspiralTable) * (val->data->len));
	SnglInspiralTable *sngl_table = *psngl_table;
	for (jlen=0; jlen<val->data->len; jlen++)
		sngl_table[jlen].template_duration = g_array_index(val->data, double, jlen);

	val = g_hash_table_lookup(hash, col_names[1]);
	for (jlen=0; jlen<val->data->len; jlen++)
		sngl_table[jlen].mass1 = g_array_index(val->data, float, jlen);

	val = g_hash_table_lookup(hash, col_names[2]);
	for (jlen=0; jlen<val->data->len; jlen++)
		sngl_table[jlen].mass2 = g_array_index(val->data, float, jlen);

	val = g_hash_table_lookup(hash, col_names[3]);
	for (jlen=0; jlen<val->data->len; jlen++)
		sngl_table[jlen].mchirp = g_array_index(val->data, float, jlen);


	val = g_hash_table_lookup(hash, col_names[4]);
	for (jlen=0; jlen<val->data->len; jlen++)
		sngl_table[jlen].mtotal = g_array_index(val->data, float, jlen);

	val = g_hash_table_lookup(hash, col_names[5]);
	for (jlen=0; jlen<val->data->len; jlen++)
		sngl_table[jlen].spin1x = g_array_index(val->data, float, jlen);

	val = g_hash_table_lookup(hash, col_names[6]);
	for (jlen=0; jlen<val->data->len; jlen++)
		sngl_table[jlen].spin1y = g_array_index(val->data, float, jlen);

	val = g_hash_table_lookup(hash, col_names[7]);
	for (jlen=0; jlen<val->data->len; jlen++)
		sngl_table[jlen].spin1z = g_array_index(val->data, float, jlen);

	val = g_hash_table_lookup(hash, col_names[8]);
	for (jlen=0; jlen<val->data->len; jlen++)
		sngl_table[jlen].spin2x = g_array_index(val->data, float, jlen);

	val = g_hash_table_lookup(hash, col_names[9]);
	for (jlen=0; jlen<val->data->len; jlen++)
		sngl_table[jlen].spin2y = g_array_index(val->data, float, jlen);

	val = g_hash_table_lookup(hash, col_names[10]);
	for (jlen=0; jlen<val->data->len; jlen++) 
		sngl_table[jlen].spin2z = g_array_index(val->data, float, jlen);

	val = g_hash_table_lookup(hash, col_names[11]);
	for (jlen=0; jlen<val->data->len; jlen++) 
		sngl_table[jlen].eta = g_array_index(val->data, float, jlen);

	for (icol=0; icol<NSNGL_TMPLT_COLS; icol++) 
		g_string_free(col_names[icol], TRUE);

	// FIXME: XmlTable destroy not implemented yet.	
	// freeTable(xtable);
	free(xns);
}

void
state_destroy(PostcohState *state)
{
	int i;
	if(state->is_member_init != NOT_INIT) {
		for(i=0; i<state->nifo; i++) {
			CUDA_CHECK(cudaFree(state->dd_snglsnr[i]));
			CUDA_CHECK(cudaFree(state->dd_autocorr_matrix[i]));
			CUDA_CHECK(cudaFree(state->dd_autocorr_norm[i]));
		}

		CUDA_CHECK(cudaFree(state->dd_snglsnr));
		CUDA_CHECK(cudaFree(state->dd_autocorr_matrix));
		CUDA_CHECK(cudaFree(state->dd_autocorr_norm));

		int gps_end = 24*3600;
		int ngps = gps_end/(state->gps_step);
		for(i=0; i<ngps; i++) {
			CUDA_CHECK(cudaFree(state->d_U_map[i]));
			CUDA_CHECK(cudaFree(state->d_diff_map[i]));
		}
		for(i=0; i<state->nifo; i++) {
			peak_list_destroy(state->peak_list[i]);
			free(state->peak_list[i]);
		}
	}

}

void
peak_list_destroy(PeakList *pklist)
{
	
	CUDA_CHECK(cudaFree(pklist->d_npeak));
	CUDA_CHECK(cudaFree(pklist->d_snglsnr_L));
	CUDA_CHECK(cudaFree(pklist->d_peak_tmplt));

	CUDA_CHECK(cudaFreeHost(pklist->npeak));
	CUDA_CHECK(cudaFreeHost(pklist->snglsnr_L));
}

void
state_reset_npeak(PeakList *pklist)
{
	//printf("d_npeak %p\n", pklist->d_npeak);
	CUDA_CHECK(cudaMemset(pklist->d_npeak, 0, sizeof(int)));
	pklist->npeak[0] = 0;
}

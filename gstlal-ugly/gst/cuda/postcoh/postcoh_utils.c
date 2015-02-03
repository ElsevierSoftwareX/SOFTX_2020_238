#include <LIGOLw_xmllib/LIGOLwHeader.h>
#include <chealpix.h>
#include "postcoh_utils.h"

char* IFO_MAP[] = {"L1", "H1", "V1"};

PeakList *create_peak_list(PostcohState *state, gint hist_trials)
{
		int exe_len = state->exe_len;
		PeakList *pklist = (PeakList *)malloc(sizeof(PeakList));

		int peak_intlen = 3 * exe_len + 1;
		int peak_floatlen = (4 + hist_trials * 3 ) * exe_len;
		pklist->peak_intlen = peak_intlen;
		pklist->peak_floatlen = peak_floatlen;
		
		cudaMalloc((void **) &(pklist->d_tmplt_idx), sizeof(int) * peak_intlen );
		cudaMemset(pklist->d_tmplt_idx, 0, sizeof(int) * peak_intlen);
		pklist->d_pix_idx = pklist->d_tmplt_idx + exe_len;
		pklist->d_peak_pos = pklist->d_pix_idx + exe_len;
		pklist->d_npeak = pklist->d_peak_pos + exe_len;

		cudaMalloc((void **) &(pklist->d_maxsnglsnr), sizeof(float) * peak_floatlen);
		cudaMemset(pklist->d_maxsnglsnr, 0, sizeof(float) * peak_floatlen);
		pklist->d_cohsnr = pklist->d_maxsnglsnr + exe_len;
		pklist->d_nullsnr = pklist->d_cohsnr + exe_len;
		pklist->d_chisq = pklist->d_nullsnr + exe_len;
		pklist->d_cohsnr_bg = pklist->d_chisq + exe_len;
		pklist->d_nullsnr_bg = pklist->d_cohsnr_bg + hist_trials * exe_len;
		pklist->d_chisq_bg = pklist->d_nullsnr_bg + hist_trials * exe_len;

		pklist->tmplt_idx = (int *)malloc(sizeof(int) * peak_intlen);
		memset(pklist->tmplt_idx, 0, sizeof(int) * peak_intlen);
		pklist->pix_idx = pklist->tmplt_idx + exe_len;
		pklist->peak_pos = pklist->pix_idx + exe_len;
		pklist->npeak = pklist->peak_pos + exe_len;

		pklist->maxsnglsnr = (float *)malloc(sizeof(float) * peak_floatlen);
		memset(pklist->maxsnglsnr, 0, sizeof(float) * peak_floatlen);
		pklist->cohsnr = pklist->maxsnglsnr + exe_len;
		pklist->nullsnr = pklist->cohsnr + exe_len;
		pklist->chisq = pklist->nullsnr + exe_len;
		pklist->cohsnr_bg = pklist->chisq + exe_len;
		pklist->nullsnr_bg = pklist->cohsnr_bg + hist_trials * exe_len;
		pklist->chisq_bg = pklist->nullsnr_bg + hist_trials * exe_len;

		printf("set peak addr %p, npeak addr %p\n", pklist, pklist->npeak);

		/* temporary struct to store tmplt max in one exe_len data */
		cudaMalloc((void **)&(pklist->d_peak_tmplt), sizeof(float) * state->ntmplt);
		cudaMemset(pklist->d_peak_tmplt, 0, sizeof(float) * state->ntmplt);

		return pklist;
}

void
cuda_postcoh_map_from_xml(char *fname, PostcohState *state)
{
	// FIXME: sanity check that the size of U matrix and diff matrix for
	// each sky pixel is consistent with number of detectors
	printf("read map from xml\n");
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


	printf("test\n");
	printf("%s \n", xns[0].tag);

	printf("%p\n", param_gps.data);
	state->gps_step = *((int *)param_gps.data);
	printf("gps_step %d\n", state->gps_step);
	unsigned long nside = (unsigned long) 1 << *((int *)param_order.data);
	state->npix = nside2npix(nside);
	free(param_gps.data);
	param_gps.data = NULL;
	printf("test\n");
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
		printf("%s\n", xns[i].tag);
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
		cudaMalloc((void **)&(state->d_U_map[i]), mem_alloc_size);
		cudaMemcpy(state->d_U_map[i], array_u[i].data, mem_alloc_size, cudaMemcpyHostToDevice);
		cudaMalloc((void **)&(state->d_diff_map[i]), mem_alloc_size);
		cudaMemcpy(state->d_diff_map[i], array_diff[i].data, mem_alloc_size, cudaMemcpyHostToDevice);

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
cuda_postcoh_autocorr_from_xml(char *fname, PostcohState *state)
{
	printf("read autocorr from xml\n");

	int ntoken = 0;

	char *end_ifo, *fname_cpy = (char *)malloc(sizeof(char) * strlen(fname));
	strcpy(fname_cpy, fname);
	char *token = strtok_r(fname, ",", &end_ifo);
	int mem_alloc_size = 0, autocorr_len = 0, ntmplt = 0, match_ifo;

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
	printf("fname_cpy %s\n", fname_cpy);
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
		autocorr_len = array_autocorr[0].dim[0];

		printf("parse match ifo %d, %s, ntmplt %d, auto_len %d\n", match_ifo, token_bankname, ntmplt, autocorr_len);
		mem_alloc_size = sizeof(COMPLEX_F) * ntmplt * autocorr_len;
		cudaMalloc((void **)&(autocorr[match_ifo]), mem_alloc_size);
		cudaMalloc((void **)&(autocorr_norm[match_ifo]), sizeof(float) * ntmplt);

		if (tmp_autocorr == NULL) {
			tmp_autocorr = (COMPLEX_F *)malloc(mem_alloc_size);
			tmp_norm = (float *)malloc(sizeof(float) * ntmplt);
		}

		float tmp_re = 0.0, tmp_im = 0.0;

		memset(tmp_norm, 0, sizeof(float) * ntmplt);
		for (int j=0; j<ntmplt; j++) {
			for (int k=0; k<autocorr_len; k++) {
				tmp_re = (float)((double *)(array_autocorr[0].data))[k * ntmplt + j];
				tmp_im = (float)((double *)(array_autocorr[1].data))[k * ntmplt + j];
				tmp_autocorr[j * autocorr_len + k].re = tmp_re;
				tmp_autocorr[j * autocorr_len + k].im = tmp_im;
				tmp_norm[j] += 2 - (tmp_re * tmp_re + tmp_im * tmp_im);
			}
//			printf("match ifo %d, norm %d: %f\n", match_ifo, j, tmp_norm[j]);
		}

		cudaMemcpy(autocorr[match_ifo], tmp_autocorr, mem_alloc_size, cudaMemcpyHostToDevice);
		cudaMemcpy(&(state->dd_autocorr_matrix[match_ifo]), &(autocorr[match_ifo]), sizeof(COMPLEX_F *), cudaMemcpyHostToDevice);
		cudaMemcpy(autocorr_norm[match_ifo], tmp_norm, sizeof(float) * ntmplt, cudaMemcpyHostToDevice);
		cudaMemcpy(&(state->dd_autocorr_norm[match_ifo]), &(autocorr_norm[match_ifo]), sizeof(float *), cudaMemcpyHostToDevice);

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
	state->autochisq_len = autocorr_len;
}



void
state_destroy(PostcohState *state)
{
	int i;
	if(state->dd_snglsnr) {
		for(i=0; i<state->nifo; i++)
			cudaFree(state->dd_snglsnr[i]);
		cudaFree(state->dd_snglsnr);
	}
}

void
state_reset_npeak(PeakList *pklist)
{
	cudaMemset(pklist->d_npeak, 0, sizeof(int));
	pklist->npeak[0] = 0;
}

#include <LIGOLw_xmllib/LIGOLwHeader.h>
#include <chealpix.h>
#include "postcoh_utils.h"


PeakList *create_peak_list(PostcohState *state, gint hist_trials)
{
		int exe_len = state->exe_len;
		PeakList *tmp_peak_list = (PeakList *)malloc(sizeof(PeakList));
		
		cudaMalloc((void **) &(tmp_peak_list->d_tmplt_idx), sizeof(int) * (3 * exe_len + 1 ));
		cudaMemset(tmp_peak_list->d_tmplt_idx, 0, sizeof(int) * (3 *exe_len + 1));
		tmp_peak_list->d_pix_idx = tmp_peak_list->d_tmplt_idx + exe_len;
		tmp_peak_list->d_peak_pos = tmp_peak_list->d_pix_idx + exe_len;
		tmp_peak_list->d_npeak = tmp_peak_list->d_peak_pos + exe_len;

		cudaMalloc((void **) &(tmp_peak_list->d_maxsnglsnr), sizeof(float) * (4 + 3*hist_trials)* exe_len);
		cudaMemset(tmp_peak_list->d_maxsnglsnr, 0, sizeof(float) * 4 * exe_len);

		tmp_peak_list->d_cohsnr = tmp_peak_list->d_maxsnglsnr + exe_len;
		tmp_peak_list->d_nullsnr = tmp_peak_list->d_cohsnr + exe_len;
		tmp_peak_list->d_chi2 = tmp_peak_list->d_nullsnr + exe_len;


		tmp_peak_list->tmplt_idx = (int *)malloc(sizeof(int) * (3 *exe_len + 1));
		memset(tmp_peak_list->tmplt_idx, 0, sizeof(int) * (3 * exe_len + 1));
		tmp_peak_list->pix_idx = tmp_peak_list->tmplt_idx + exe_len;
		tmp_peak_list->peak_pos = tmp_peak_list->pix_idx + exe_len;
		tmp_peak_list->npeak = tmp_peak_list->peak_pos + exe_len;
		tmp_peak_list->maxsnglsnr = (float *)malloc(sizeof(float) * 4 * exe_len);
		memset(tmp_peak_list->maxsnglsnr, 0, sizeof(float) * (4 + 3*hist_trials) * exe_len);
		tmp_peak_list->cohsnr = tmp_peak_list->maxsnglsnr + exe_len;
		tmp_peak_list->nullsnr = tmp_peak_list->cohsnr + exe_len;
		tmp_peak_list->chi2 = tmp_peak_list->nullsnr + exe_len;
    		;
		cudaMalloc((void **)&(tmp_peak_list->d_peak_tmplt), sizeof(float) * state->ntmplt);

		cudaMemset(tmp_peak_list->d_peak_tmplt, 0, sizeof(float) * state->ntmplt);
		return tmp_peak_list;
}

void
cuda_postcoh_map_from_xml(char *fname, PostcohState *state)
{
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
		/* initialisation , a must */
		array_u[i].ndim = 0;
		xns[i].data = &(array_u[i]);

		sprintf((char *)xns[i+ngps].tag, "diff_map_gps_%d:array", gps);
		xns[i+ngps].processPtr = readArray;
		/* initialisation , a must */
		array_diff[i].ndim = 0;
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
state_destroy(PostcohState *state)
{
	int i;
	if(state->d_snglsnr)
		for(i=0; i<state->nifo; i++)
			cudaFree(state->d_snglsnr[i]);
}

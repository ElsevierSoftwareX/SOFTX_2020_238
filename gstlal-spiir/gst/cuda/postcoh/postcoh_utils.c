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


#include <gst/gst.h>
#include <LIGOLwHeader.h>
#include <postcohtable.h>
#include <postcoh/postcoh_utils.h>
#include <pipe_macro.h> // for IFOComboMap
#include <cuda_debug.h>
#include <cuda_runtime.h>

//#define __DEBUG__ 0
#define NSNGL_TMPLT_COLS 14

void cuda_device_print(int deviceCount)
{  
	int dev, driverVersion = 0, runtimeVersion = 0;


  for (dev = 0; dev < deviceCount; ++dev) {
    cudaSetDevice(dev);
#ifdef __cplusplus
    cudaDeviceProp deviceProp;
#else // !__cplusplus
    struct cudaDeviceProp deviceProp;
#endif

	cudaGetDeviceProperties(&deviceProp, dev);

    printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

    // Console log
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
           driverVersion / 1000, (driverVersion % 100) / 10,
           runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
           deviceProp.major, deviceProp.minor);

    printf("  (%2d) Multiprocessors \n",
           deviceProp.multiProcessorCount);

    printf(
        "  GPU Max Clock rate:                            %.0f MHz (%0.2f "
        "GHz)\n",
        deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

    printf(
        "  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, "
        "%d), 3D=(%d, %d, %d)\n",
        deviceProp.maxTexture1D, deviceProp.maxTexture2D[0],
        deviceProp.maxTexture2D[1], deviceProp.maxTexture3D[0],
        deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
    printf(
        "  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
        deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
    printf(
        "  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d "
        "layers\n",
        deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
        deviceProp.maxTexture2DLayered[2]);

    printf("  Total amount of constant memory:               %lu bytes\n",
           deviceProp.totalConstMem);
    printf("  Total amount of shared memory per block:       %lu bytes\n",
           deviceProp.sharedMemPerBlock);
    printf("  Total number of registers available per block: %d\n",
           deviceProp.regsPerBlock);
    printf("  Warp size:                                     %d\n",
           deviceProp.warpSize);
    printf("  Maximum number of threads per multiprocessor:  %d\n",
           deviceProp.maxThreadsPerMultiProcessor);
    printf("  Maximum number of threads per block:           %d\n",
           deviceProp.maxThreadsPerBlock);
    printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
           deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
           deviceProp.maxThreadsDim[2]);
    printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
           deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
           deviceProp.maxGridSize[2]);
    printf("  Maximum memory pitch:                          %lu bytes\n",
           deviceProp.memPitch);
    printf("  Texture alignment:                             %lu bytes\n",
           deviceProp.textureAlignment);
    printf(
        "  Concurrent copy and kernel execution:          %s with %d copy "
        "engine(s)\n",
        (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
    printf("  Run time limit on kernels:                     %s\n",
           deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
    printf("  Integrated GPU sharing Host Memory:            %s\n",
           deviceProp.integrated ? "Yes" : "No");
    printf("  Support host page-locked memory mapping:       %s\n",
           deviceProp.canMapHostMemory ? "Yes" : "No");
    printf("  Alignment requirement for Surfaces:            %s\n",
           deviceProp.surfaceAlignment ? "Yes" : "No");
    printf("  Device has ECC support:                        %s\n",
           deviceProp.ECCEnabled ? "Enabled" : "Disabled");
    printf("  Device supports Unified Addressing (UVA):      %s\n",
           deviceProp.unifiedAddressing ? "Yes" : "No");
    printf("  Device supports Compute Preemption:            %s\n",
           deviceProp.computePreemptionSupported ? "Yes" : "No");
    printf("  Supports Cooperative Kernel Launch:            %s\n",
           deviceProp.cooperativeLaunch ? "Yes" : "No");
    printf("  Supports MultiDevice Co-op Kernel Launch:      %s\n",
           deviceProp.cooperativeMultiDeviceLaunch ? "Yes" : "No");
    printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n",
           deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);

    const char *sComputeMode[] = {
        "Default (multiple host threads can use ::cudaSetDevice() with device "
        "simultaneously)",
        "Exclusive (only one host thread in one process is able to use "
        "::cudaSetDevice() with this device)",
        "Prohibited (no host thread can use ::cudaSetDevice() with this "
        "device)",
        "Exclusive Process (many threads in one process is able to use "
        "::cudaSetDevice() with this device)",
        "Unknown",
        NULL};
    printf("  Compute Mode:\n");
    printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);
  }

}


/* get ifo indices of a given combo in IFOMap
 * e.g. HV: 0, 2
 */
void get_write_ifo_mapping(char *ifo_combo, int nifo, int *write_ifo_mapping)
{
	int iifo, jifo;
	for (iifo=0; iifo<nifo; iifo++)
		for (jifo=0; jifo<MAX_NIFO; jifo++)
			if (strncmp(ifo_combo+iifo*IFO_LEN, IFOMap[jifo].name, IFO_LEN) == 0 ) {
				write_ifo_mapping[iifo] = jifo;
				break;

			}
#ifdef __DEBUG__
	for (iifo=0; iifo<nifo; iifo++)
		printf("write_ifo_mapping %d->%d\n", iifo, write_ifo_mapping[iifo]);
#endif
}
PeakList *create_peak_list(PostcohState *state, cudaStream_t stream)
{
		int hist_trials = state->hist_trials;
		g_assert(hist_trials != -1);
		int max_npeak = state->max_npeak;
#ifdef __DEBUG__
		printf("max_npeak %d\n", max_npeak);
#endif
		PeakList *pklist = (PeakList *)malloc(sizeof(PeakList));

		int peak_intlen = (7 + hist_trials) * max_npeak + 1;
		int peak_floatlen = (12 + hist_trials * 12 ) * max_npeak;
		pklist->peak_intlen = peak_intlen;
		pklist->peak_floatlen = peak_floatlen;
	
		/* create device space for peak list for int-type variables */	
		CUDA_CHECK(cudaMalloc((void **) &(pklist->d_npeak), sizeof(int) * peak_intlen ));
		CUDA_CHECK(cudaMemsetAsync(pklist->d_npeak, 0, sizeof(int) * peak_intlen, stream));
		pklist->d_peak_pos = pklist->d_npeak + 1;
		pklist->d_len_idx = pklist->d_npeak + 1 + max_npeak;
		pklist->d_tmplt_idx = pklist->d_npeak + 1 + 2 * max_npeak;
		pklist->d_pix_idx = pklist->d_npeak + 1 + 3 * max_npeak;
		pklist->d_pix_idx_bg = pklist->d_npeak + 1 + 4 * max_npeak;
		pklist->d_ntoff_H = pklist->d_npeak + 1 + (4 + hist_trials) * max_npeak;
		pklist->d_ntoff_L = pklist->d_npeak + 1 + (5 + hist_trials) * max_npeak;
		pklist->d_ntoff_V = pklist->d_npeak + 1 + (6 + hist_trials) * max_npeak;

		//printf("d_npeak %p\n", pklist->d_npeak);
		//CUDA_CHECK(cudaMemsetAsync(pklist->d_npeak, 0, sizeof(int), stream));

		/* create device space for peak list for float-type variables */	
		CUDA_CHECK(cudaMalloc((void **) &(pklist->d_snglsnr_H), sizeof(float) * peak_floatlen));
		CUDA_CHECK(cudaMemsetAsync(pklist->d_snglsnr_H, 0, sizeof(float) * peak_floatlen, stream));
		pklist->d_snglsnr_L = pklist->d_snglsnr_H + max_npeak;
		pklist->d_snglsnr_V = pklist->d_snglsnr_H + 2 * max_npeak;
		pklist->d_coaphase_H = pklist->d_snglsnr_H + 3 * max_npeak;
		pklist->d_coaphase_L = pklist->d_snglsnr_H + 4 * max_npeak;
		pklist->d_coaphase_V = pklist->d_snglsnr_H + 5 * max_npeak;
		pklist->d_chisq_H = pklist->d_snglsnr_H + 6 * max_npeak;
		pklist->d_chisq_L = pklist->d_snglsnr_H + 7 * max_npeak;
		pklist->d_chisq_V = pklist->d_snglsnr_H + 8 * max_npeak;
		pklist->d_cohsnr = pklist->d_snglsnr_H + 9 * max_npeak;
		pklist->d_nullsnr = pklist->d_snglsnr_H + 10 * max_npeak;
		pklist->d_cmbchisq = pklist->d_snglsnr_H + 11 * max_npeak;
	
		pklist->d_snglsnr_bg_H = pklist->d_snglsnr_H + 12 * max_npeak;
		pklist->d_snglsnr_bg_L = pklist->d_snglsnr_H + (12 + hist_trials) * max_npeak;
		pklist->d_snglsnr_bg_V = pklist->d_snglsnr_H + (12 + 2* hist_trials) * max_npeak;
		pklist->d_coaphase_bg_H = pklist->d_snglsnr_H + (12 + 3*hist_trials) * max_npeak;
		pklist->d_coaphase_bg_L = pklist->d_snglsnr_H + (12 + 4*hist_trials) * max_npeak;
		pklist->d_coaphase_bg_V = pklist->d_snglsnr_H + (12 + 5*hist_trials) * max_npeak;
		pklist->d_chisq_bg_H = pklist->d_snglsnr_H + (12 + 6*hist_trials) * max_npeak;
		pklist->d_chisq_bg_L = pklist->d_snglsnr_H + (12 + 7*hist_trials) * max_npeak;
		pklist->d_chisq_bg_V = pklist->d_snglsnr_H + (12 + 8*hist_trials) * max_npeak;
		pklist->d_cohsnr_bg = pklist->d_snglsnr_H + (12 + 9*hist_trials) * max_npeak;
		pklist->d_nullsnr_bg = pklist->d_snglsnr_H + (12 + 10*hist_trials) * max_npeak;
		pklist->d_cmbchisq_bg = pklist->d_snglsnr_H + (12 + 11 * hist_trials) * max_npeak;

		/* create host space for peak list for int-type variables */	
		//pklist->npeak = (int *)malloc(sizeof(int) * peak_intlen);
		CUDA_CHECK(cudaMallocHost((void **) &(pklist->npeak), sizeof(int) * peak_intlen));
		memset(pklist->npeak, 0, sizeof(int) * peak_intlen);
		pklist->peak_pos = pklist->npeak + 1;
		pklist->len_idx = pklist->npeak + 1 + max_npeak;
		pklist->tmplt_idx = pklist->npeak + 1 + 2 * max_npeak;
		pklist->pix_idx = pklist->npeak + 1 + 3 * max_npeak;
		pklist->pix_idx_bg = pklist->npeak + 1 + 4 * max_npeak;
		pklist->ntoff_H = pklist->npeak + 1 + (4 + hist_trials) * max_npeak;
		pklist->ntoff_L = pklist->npeak + 1 + (5 + hist_trials) * max_npeak;
		pklist->ntoff_V = pklist->npeak + 1 + (6 + hist_trials) * max_npeak;

		/* create host space for peak list for float-type variables */	
		//pklist->snglsnr_L = (float *)malloc(sizeof(float) * peak_floatlen);
		CUDA_CHECK(cudaMallocHost((void **) &(pklist->snglsnr_H), sizeof(float) * peak_floatlen));
		memset(pklist->snglsnr_H, 0, sizeof(float) * peak_floatlen);
		pklist->snglsnr_L = pklist->snglsnr_H + max_npeak;
		pklist->snglsnr_V = pklist->snglsnr_H + 2 * max_npeak;
		pklist->coaphase_H = pklist->snglsnr_H + 3 * max_npeak;
		pklist->coaphase_L = pklist->snglsnr_H + 4 * max_npeak;
		pklist->coaphase_V = pklist->snglsnr_H + 5 * max_npeak;
		pklist->chisq_H = pklist->snglsnr_H + 6 * max_npeak;
		pklist->chisq_L = pklist->snglsnr_H + 7 * max_npeak;
		pklist->chisq_V = pklist->snglsnr_H + 8 * max_npeak;
		pklist->cohsnr = pklist->snglsnr_H + 9 * max_npeak;
		pklist->nullsnr = pklist->snglsnr_H + 10 * max_npeak;
		pklist->cmbchisq = pklist->snglsnr_H + 11 * max_npeak;
//	
		pklist->snglsnr_bg_H = pklist->snglsnr_H + 12 * max_npeak;
		pklist->snglsnr_bg_L = pklist->snglsnr_H + (12 + hist_trials) * max_npeak;
		pklist->snglsnr_bg_V = pklist->snglsnr_H + (12 + 2* hist_trials) * max_npeak;
		pklist->coaphase_bg_H = pklist->snglsnr_H + (12 + 3*hist_trials) * max_npeak;
		pklist->coaphase_bg_L = pklist->snglsnr_H + (12 + 4*hist_trials) * max_npeak;
		pklist->coaphase_bg_V = pklist->snglsnr_H + (12 + 5*hist_trials) * max_npeak;
		pklist->chisq_bg_H = pklist->snglsnr_H + (12 + 6*hist_trials) * max_npeak;
		pklist->chisq_bg_L = pklist->snglsnr_H + (12 + 7*hist_trials) * max_npeak;
		pklist->chisq_bg_V = pklist->snglsnr_H + (12 + 8*hist_trials) * max_npeak;
		pklist->cohsnr_bg = pklist->snglsnr_H + (12 + 9*hist_trials) * max_npeak;
		pklist->nullsnr_bg = pklist->snglsnr_H + (12 + 10*hist_trials) * max_npeak;
		pklist->cmbchisq_bg = pklist->snglsnr_H + (12 + 11 * hist_trials) * max_npeak;

//		printf("set peak addr %p, d_npeak addr %p\n", pklist, pklist->d_npeak);
		//printf("hist trials %d, peak_intlen %d, peak_floatlen %d\n", hist_trials, peak_intlen, peak_floatlen);
		/* temporary struct to store tmplt max in one max_npeak data */
		CUDA_CHECK(cudaMalloc((void **)&(pklist->d_peak_tmplt), sizeof(float) * state->ntmplt));
		CUDA_CHECK(cudaMemsetAsync(pklist->d_peak_tmplt, 0, sizeof(float) * state->ntmplt, stream));
		
		// add for new postcoh kernel optimized by Xiaoyang Guo
		pklist->d_snglsnr_buffer = NULL;
		pklist->len_snglsnr_buffer = 0;

		int mem_alloc_size = sizeof(float) * state->npix * 2;
		printf("alloc cohsnr_skymap size %f MB\n", (float) mem_alloc_size/1000000);

		CUDA_CHECK(cudaMalloc((void **)&(pklist->d_cohsnr_skymap), mem_alloc_size));
		CUDA_CHECK(cudaMemsetAsync(pklist->d_cohsnr_skymap, 0, mem_alloc_size, stream));
		pklist->d_nullsnr_skymap = pklist->d_cohsnr_skymap + state->npix;

		CUDA_CHECK(cudaMallocHost((void **) &(pklist->cohsnr_skymap), mem_alloc_size));
		memset(pklist->cohsnr_skymap, 0, mem_alloc_size);
		pklist->nullsnr_skymap = pklist->cohsnr_skymap + state->npix;

		CUDA_CHECK(cudaStreamSynchronize(stream));
		CUDA_CHECK(cudaPeekAtLastError());

		return pklist;
}

void
cuda_postcoh_sigmasq_from_xml(char *fname, PostcohState *state)
{

	gchar ** isigma, **sigma_fnames = g_strsplit(fname, ",", -1);
	int mem_alloc_size = 0, ntmplt = 0, match_ifo = 0, nifo = 0;

	/* parsing for nifo */
	for (isigma = sigma_fnames; *isigma; isigma++)
		nifo++;

	#ifdef __DEBUG__
	printf("sigma from %d ifo\n", nifo);
	#endif

	XmlNodeStruct *xns = (XmlNodeStruct *)malloc(sizeof(XmlNodeStruct));
	XmlArray *array_sigmasq = (XmlArray *)malloc(sizeof(XmlArray));

	state->sigmasq = (double **)malloc(sizeof(double *) * nifo );
	double **sigmasq = state->sigmasq;
	sigmasq[0] = NULL;

	sprintf((char *)xns[0].tag, "sigmasq:array");
	xns[0].processPtr = readArray;
	xns[0].data = &(array_sigmasq[0]);
	/* used for sanity check the lengths of sigmasq arrays should be equal */
	int last_dimension = -1;

	/* parsing for all_ifos */
	for (isigma = sigma_fnames; *isigma; isigma++) {
		gchar ** this_ifo_split = g_strsplit(*isigma, ":", -1);

		for (int i=0; i<nifo; i++) {
			if (strncmp(this_ifo_split[0], IFOMap[i].name ,IFO_LEN) == 0) {
				match_ifo = i;
				break;
			}
		
		}

		parseFile(this_ifo_split[1], xns, 1);

		ntmplt = array_sigmasq[0].dim[0];
		// combos like HL, match_ifo will still be like 0:H,1:L
		// combos like HV, match_ifo will still be like 0:H,1:V
		#ifdef __DEBUG__
		printf("this sigma %s, this ifo %s, match ifo %d,ntmplt %d \n", *isigma, this_ifo_split[0], match_ifo, ntmplt);
		#endif

		/* check if the lengths of sigmasq arrays are equal for all detectors */
		if (last_dimension == -1){
			/* allocate memory for sigmasq for all detectors upon the first detector ntmplt */
			last_dimension = ntmplt;
			mem_alloc_size = sizeof(double) * ntmplt;
			for (int i = 0; i < nifo; i++) {
				sigmasq[i] = (double *)malloc(mem_alloc_size);
				memset(sigmasq[i], 0, mem_alloc_size);
			}
		}
		else
			if (last_dimension != ntmplt) {
				fprintf(stderr, "reading different lengths of sigmasq arrays from different detectors, should exit\n");
				exit(0);
			}


		for (int j=0; j<ntmplt; j++) {
			sigmasq[match_ifo][j] = (double)((double *)(array_sigmasq[0].data))[j];
			#ifdef __DEBUG__
			printf("match ifo %d, template %d: %f\n", match_ifo, j, sigmasq[match_ifo][j]);
			#endif
		}

		freeArraydata(array_sigmasq);
		/*
		 * Cleanup function for the XML library.
		 */
		xmlCleanupParser();
		/*
		 * this is to debug memory for regression tests
		 */
		xmlMemoryDump();

		g_strfreev(this_ifo_split);

	}
	/* free memory */
	g_strfreev(sigma_fnames);
	free(array_sigmasq);
	free(xns);
}

void
cuda_postcoh_map_from_xml(char *fname, PostcohState *state, cudaStream_t stream)
{
	// FIXME: sanity check that the size of U matrix and diff matrix for
	// each sky pixel is consistent with number of detectors
	//printf("read map from xml\n");
	/* first get the params */
	XmlNodeStruct *xns = (XmlNodeStruct *)malloc(sizeof(XmlNodeStruct) * 3);
	XmlParam param_gps_step = {0, NULL};
	XmlParam param_gps_start = {0, NULL};
	XmlParam param_order = {0, NULL};

	sprintf((char *)xns[0].tag, "gps_step:param");
	xns[0].processPtr = readParam;
	xns[0].data = &param_gps_step;

	sprintf((char *)xns[1].tag, "gps_start:param");
	xns[1].processPtr = readParam;
	xns[1].data = &param_gps_start;

	sprintf((char *)xns[2].tag, "chealpix_order:param");
	xns[2].processPtr = readParam;
	xns[2].data = &param_order;
	// printf("read in detrsp map from xml %s\n", fname);

	parseFile(fname, xns, 3);
	/*
	 * Cleanup function for the XML library.
	 */
	xmlCleanupParser();
	/*
	 * this is to debug memory for regression tests
	 */
	xmlMemoryDump();


	/* assign basic information to state */
	int gps_step_new = *((int *)param_gps_step.data);
	if (state->npix != POSTCOH_PARAMS_NOT_INIT && state->gps_step != gps_step_new) {
	  fprintf(stderr, "detrsp map has a different configuration than last read, aboring!");
	  exit(0);
	}
	state->gps_step = *((int *)param_gps_step.data);
	state->gps_start = *((long *)param_gps_start.data);
	unsigned long nside = (unsigned long) 1 << *((int *)param_order.data);
	state->nside = nside;

	/* free memory */
	free(param_gps_step.data);
	free(param_gps_start.data);
	param_gps_step.data = NULL;
	param_gps_start.data = NULL;
	free(param_order.data);
	param_order.data = NULL;
	free(xns);


	int gps = 0, gps_end = 24*3600;
	int ngps = gps_end/(state->gps_step);

	xns = (XmlNodeStruct *)malloc(sizeof(XmlNodeStruct) * 2* ngps);
	if (state->npix == POSTCOH_PARAMS_NOT_INIT) {
	  state->d_U_map = (float**)malloc(sizeof(float *) * ngps);
	  state->d_diff_map = (float**)malloc(sizeof(float *) * ngps);
	}

	int i;
	XmlArray *array_u = (XmlArray *)malloc(sizeof(XmlArray) * ngps);
	XmlArray *array_diff = (XmlArray *)malloc(sizeof(XmlArray) * ngps);

	for (i=0; i<ngps; i++) {

		sprintf((char *)xns[i].tag, "U_map_gps_%d:array", gps);
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
		if (state->npix == POSTCOH_PARAMS_NOT_INIT) {
		CUDA_CHECK(cudaMalloc((void **)&(state->d_U_map[i]), mem_alloc_size));
		CUDA_CHECK(cudaMalloc((void **)&(state->d_diff_map[i]), mem_alloc_size));
		}
		CUDA_CHECK(cudaMemcpyAsync(state->d_U_map[i], array_u[i].data, mem_alloc_size, cudaMemcpyHostToDevice, stream));
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

	/* label that the map has been initialized, no longer POSTCOH_PARAMS_NOT_INIT */
	state->npix = nside2npix(nside);

	/* free memory */
	for (i=0; i<ngps; i++) {
		free(array_u[i].data);
		free(array_diff[i].data);
	}
	free(array_u);
	free(array_diff);
	free(xns);
}

void
cuda_postcoh_autocorr_from_xml(char *fname, PostcohState *state, cudaStream_t stream)
{
	#ifdef __DEBUG__
	printf("read in autocorr from xml %s\n", fname);
	#endif

	gchar ** iauto, **auto_fnames = g_strsplit(fname, ",", -1);
	int mem_alloc_size = 0, autochisq_len = 0, ntmplt = 0, match_ifo = 0, nifo = 0;

	/* parsing for nifo */
	for (iauto = auto_fnames; *iauto; iauto++)
		nifo++;

	#ifdef __DEBUG__
	printf("autocorr from %d ifo\n", nifo);
	#endif

	XmlNodeStruct *xns = (XmlNodeStruct *)malloc(sizeof(XmlNodeStruct) * 2);
	XmlArray *array_autocorr = (XmlArray *)malloc(sizeof(XmlArray) * 2);

	COMPLEX_F *tmp_autocorr = NULL;
	float *tmp_norm = NULL;
	/* allocate memory for host autocorr list pointer and device auto list pointer*/
	COMPLEX_F **autocorr = (COMPLEX_F **)malloc(sizeof(COMPLEX_F *) * nifo );
	float **autocorr_norm = (float **)malloc(sizeof(float *) * nifo );
	CUDA_CHECK(cudaMalloc((void **)&(state->dd_autocorr_matrix), sizeof(COMPLEX_F *) * nifo));
	CUDA_CHECK(cudaMalloc((void **)&(state->dd_autocorr_norm), sizeof(float *) * nifo));

	sprintf((char *)xns[0].tag, "autocorrelation_bank_real:array");
	xns[0].processPtr = readArray;
	xns[0].data = &(array_autocorr[0]);

	sprintf((char *)xns[1].tag, "autocorrelation_bank_imag:array");
	xns[1].processPtr = readArray;
	xns[1].data = &(array_autocorr[1]);

	
	/* parsing for all_ifos */
	for (iauto = auto_fnames; *iauto; iauto++) {
		gchar ** this_ifo_split = g_strsplit(*iauto, ":", -1);

		for (int i=0; i<nifo; i++) {
			if (strncmp(this_ifo_split[0], IFOMap[i].name ,IFO_LEN) == 0) {
				match_ifo = i;
				break;
			}
		
		}

		parseFile(this_ifo_split[1], xns, 2);
		ntmplt = array_autocorr[0].dim[1];
		autochisq_len = array_autocorr[0].dim[0];

		#ifdef __DEBUG__
		printf("this auto %s, this ifo %s, match ifo %d,ntmplt %d,auto len %d \n", *iauto, this_ifo_split[0], match_ifo, ntmplt, autochisq_len);
		#endif


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
			#ifdef __DEBUG__
			printf("match ifo %d, norm %d: %f\n", match_ifo, j, tmp_norm[j]);
			#endif
		}
		/* copy the autocorr array to GPU device;
		 * copy the array address to GPU device */
		CUDA_CHECK(cudaMemcpyAsync(autocorr[match_ifo], tmp_autocorr, mem_alloc_size, cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(&(state->dd_autocorr_matrix[match_ifo]), &(autocorr[match_ifo]), sizeof(COMPLEX_F *), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(autocorr_norm[match_ifo], tmp_norm, sizeof(float) * ntmplt, cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(&(state->dd_autocorr_norm[match_ifo]), &(autocorr_norm[match_ifo]), sizeof(float *), cudaMemcpyHostToDevice, stream));

		freeArraydata(array_autocorr);
		freeArraydata(array_autocorr+1);
		/*
		 * Cleanup function for the XML library.
		 */
		xmlCleanupParser();
		/*
		 * this is to debug memory for regression tests
		 */
		xmlMemoryDump();
		g_strfreev(this_ifo_split);

	}

	state->autochisq_len = autochisq_len;

	/* free memory */
	g_strfreev(auto_fnames);
	free(array_autocorr);
	free(tmp_autocorr);
	free(tmp_norm);
	free(autocorr);
	free(autocorr_norm);
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
	"sngl_inspiral:eta",
	"sngl_inspiral:end_time", 
	"sngl_inspiral:end_time_ns"
	};

void
cuda_postcoh_sngl_tmplt_from_xml(char *fname, SnglInspiralTable **psngl_table)
{
	#ifdef __DEBUG__
	printf("read in sngl tmplt from %s\n", fname);
	#endif

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

	val = g_hash_table_lookup(hash, col_names[12]);
	for (jlen=0; jlen<val->data->len; jlen++) 
		sngl_table[jlen].end.gpsSeconds = g_array_index(val->data, int, jlen);

	val = g_hash_table_lookup(hash, col_names[13]);
	for (jlen=0; jlen<val->data->len; jlen++) {
		sngl_table[jlen].end.gpsNanoSeconds = g_array_index(val->data, int, jlen);
		#ifdef __DEBUG__
		printf("read %d, end gps %d, end nano %d\n", jlen, sngl_table[jlen].end.gpsSeconds, sngl_table[jlen].end.gpsNanoSeconds);
		#endif
	}

	/* free memory */
	freeTable(xtable);
	free(xtable);
	free(xns);
	for (icol=0; icol<NSNGL_TMPLT_COLS; icol++) 
		g_string_free(col_names[icol], TRUE);
	free(col_names);
}

static void
sigmasq_destroy(PostcohState *state){
  int iifo;
  if (state->sigmasq != NULL) {
    for (iifo=0; iifo<state->nifo; iifo++) {
      if (state->sigmasq[iifo] != NULL)
	free(state->sigmasq[iifo]);
    }
    free(state->sigmasq);
  }
}
 
static void
map_destroy(PostcohState *state){
  int i, ngps = 24*3600/ state->gps_step;
	for (i=0; i<ngps; i++) {
		CUDA_CHECK(cudaFree(state->d_U_map[i]));
		CUDA_CHECK(cudaFree(state->d_diff_map[i]));
	}
	
    free(state->d_U_map);
    free(state->d_diff_map);
}
   

static void
autocorr_destroy(PostcohState *state)
{
  int iifo;
  if (state->dd_autocorr_matrix != NULL) {
    for (iifo=0; iifo<state->nifo; iifo++) {
      if (state->dd_autocorr_matrix[iifo] != NULL)
	cudaFree(state->dd_autocorr_matrix[iifo]);
	cudaFree(state->dd_autocorr_norm[iifo]);
    }
    cudaFree(state->dd_autocorr_matrix);
    cudaFree(state->dd_autocorr_norm);
  }

}

void
peak_list_destroy(PeakList *pklist)
{
	
	CUDA_CHECK(cudaFree(pklist->d_npeak));
	CUDA_CHECK(cudaFree(pklist->d_snglsnr_L));
	CUDA_CHECK(cudaFree(pklist->d_peak_tmplt));
	CUDA_CHECK(cudaFree(pklist->d_cohsnr_skymap));

	CUDA_CHECK(cudaFreeHost(pklist->npeak));
	CUDA_CHECK(cudaFreeHost(pklist->snglsnr_L));
	CUDA_CHECK(cudaFreeHost(pklist->cohsnr_skymap));
}

void
state_destroy(PostcohState *state)
{
	int i;
	if(state->is_member_init != POSTCOH_PARAMS_NOT_INIT) {
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
	sigmasq_destroy(state);
	autocorr_destroy(state);
	map_destroy(state);
	}
}

void
state_reset_npeak(PeakList *pklist)
{
	//printf("d_npeak %p\n", pklist->d_npeak);
	CUDA_CHECK(cudaMemset(pklist->d_npeak, 0, sizeof(int)));
	pklist->npeak[0] = 0;
}

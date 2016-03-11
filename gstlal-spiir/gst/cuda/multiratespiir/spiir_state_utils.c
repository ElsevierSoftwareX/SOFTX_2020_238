/* GStreamer
 * Copyright (C) 2014 Qi Chu
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


/*
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */



#ifdef __cplusplus
extern "C" {
#endif

#include <glib.h>
#include <math.h>
#include "multiratespiir.h"
#include "spiir_state_utils.h"
#include "spiir_state_macro.h"
#include <LIGOLw_xmllib/LIGOLwHeader.h>
#include <cuda_runtime.h>
#include <cuda_debug.h>

#ifdef __cplusplus
}
#endif

#if 0
// deprecated: we have cuda_debug.h for gpu debug now
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
static void gpuAssert(cudaError_t code, char *file, int line)
{
   if (code != cudaSuccess) 
   {
      printf ("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      exit(code);
   }
}
#endif
/*
 * ============================================================================
 *
 * 		The following part is copied/ rewritten from 
 * 		gstreamer-base/ audioresample/ resample.c
 *
 * ============================================================================
 */

static double kaiser12_table[68] = {
  0.99859849, 1.00000000, 0.99859849, 0.99440475, 0.98745105, 0.97779076,
  0.96549770, 0.95066529, 0.93340547, 0.91384741, 0.89213598, 0.86843014,
  0.84290116, 0.81573067, 0.78710866, 0.75723148, 0.72629970, 0.69451601,
  0.66208321, 0.62920216, 0.59606986, 0.56287762, 0.52980938, 0.49704014,
  0.46473455, 0.43304576, 0.40211431, 0.37206735, 0.34301800, 0.31506490,
  0.28829195, 0.26276832, 0.23854851, 0.21567274, 0.19416736, 0.17404546,
  0.15530766, 0.13794294, 0.12192957, 0.10723616, 0.09382272, 0.08164178,
  0.07063950, 0.06075685, 0.05193064, 0.04409466, 0.03718069, 0.03111947,
  0.02584161, 0.02127838, 0.01736250, 0.01402878, 0.01121463, 0.00886058,
  0.00691064, 0.00531256, 0.00401805, 0.00298291, 0.00216702, 0.00153438,
  0.00105297, 0.00069463, 0.00043489, 0.00025272, 0.00013031, 0.0000527734,
  0.00001000, 0.00000000
};

/*
static double kaiser12_table[36] = {
   0.99440475, 1.00000000, 0.99440475, 0.97779076, 0.95066529, 0.91384741,
   0.86843014, 0.81573067, 0.75723148, 0.69451601, 0.62920216, 0.56287762,
   0.49704014, 0.43304576, 0.37206735, 0.31506490, 0.26276832, 0.21567274,
   0.17404546, 0.13794294, 0.10723616, 0.08164178, 0.06075685, 0.04409466,
   0.03111947, 0.02127838, 0.01402878, 0.00886058, 0.00531256, 0.00298291,
   0.00153438, 0.00069463, 0.00025272, 0.0000527734, 0.00000500, 0.00000000};
*/
static double kaiser10_table[36] = {
  0.99537781, 1.00000000, 0.99537781, 0.98162644, 0.95908712, 0.92831446,
  0.89005583, 0.84522401, 0.79486424, 0.74011713, 0.68217934, 0.62226347,
  0.56155915, 0.50119680, 0.44221549, 0.38553619, 0.33194107, 0.28205962,
  0.23636152, 0.19515633, 0.15859932, 0.12670280, 0.09935205, 0.07632451,
  0.05731132, 0.04193980, 0.02979584, 0.02044510, 0.01345224, 0.00839739,
  0.00488951, 0.00257636, 0.00115101, 0.00035515, 0.00000000, 0.00000000
};

static double kaiser8_table[36] = {
  0.99635258, 1.00000000, 0.99635258, 0.98548012, 0.96759014, 0.94302200,
  0.91223751, 0.87580811, 0.83439927, 0.78875245, 0.73966538, 0.68797126,
  0.63451750, 0.58014482, 0.52566725, 0.47185369, 0.41941150, 0.36897272,
  0.32108304, 0.27619388, 0.23465776, 0.19672670, 0.16255380, 0.13219758,
  0.10562887, 0.08273982, 0.06335451, 0.04724088, 0.03412321, 0.02369490,
  0.01563093, 0.00959968, 0.00527363, 0.00233883, 0.00050000, 0.00000000
};

static double kaiser6_table[36] = {
  0.99733006, 1.00000000, 0.99733006, 0.98935595, 0.97618418, 0.95799003,
  0.93501423, 0.90755855, 0.87598009, 0.84068475, 0.80211977, 0.76076565,
  0.71712752, 0.67172623, 0.62508937, 0.57774224, 0.53019925, 0.48295561,
  0.43647969, 0.39120616, 0.34752997, 0.30580127, 0.26632152, 0.22934058,
  0.19505503, 0.16360756, 0.13508755, 0.10953262, 0.08693120, 0.06722600,
  0.05031820, 0.03607231, 0.02432151, 0.01487334, 0.00752000, 0.00000000
};

struct FuncDef
{
  double *table;
  int oversample;
};

static struct FuncDef _KAISER12 = { kaiser12_table, 64 };

#define KAISER12 (&_KAISER12)
/*static struct FuncDef _KAISER12 = {kaiser12_table, 32};
#define KAISER12 (&_KAISER12)*/
static struct FuncDef _KAISER10 = { kaiser10_table, 32 };

#define KAISER10 (&_KAISER10)
static struct FuncDef _KAISER8 = { kaiser8_table, 32 };

#define KAISER8 (&_KAISER8)
static struct FuncDef _KAISER6 = { kaiser6_table, 32 };

#define KAISER6 (&_KAISER6)

struct QualityMapping
{
  int base_length;
  int oversample;
  float downsample_bandwidth;
  float upsample_bandwidth;
  struct FuncDef *window_func;
};


/* This table maps conversion quality to internal parameters. There are two
   reasons that explain why the up-sampling bandwidth is larger than the 
   down-sampling bandwidth:
   1) When up-sampling, we can assume that the spectrum is already attenuated
      close to the Nyquist rate (from an A/D or a previous resampling filter)
   2) Any aliasing that occurs very close to the Nyquist rate will be masked
      by the sinusoids/noise just below the Nyquist rate (guaranteed only for
      up-sampling).
*/
static const struct QualityMapping quality_map[11] = {
  {8, 4, 0.830f, 0.860f, KAISER6},      /* Q0 */
  {16, 4, 0.850f, 0.880f, KAISER6},     /* Q1 */
  {32, 4, 0.882f, 0.910f, KAISER6},     /* Q2 *//* 82.3% cutoff ( ~60 dB stop) 6  */
  {48, 8, 0.895f, 0.917f, KAISER8},     /* Q3 *//* 84.9% cutoff ( ~80 dB stop) 8  */
  {64, 8, 0.921f, 0.940f, KAISER8},     /* Q4 *//* 88.7% cutoff ( ~80 dB stop) 8  */
  {80, 16, 0.922f, 0.940f, KAISER10},   /* Q5 *//* 89.1% cutoff (~100 dB stop) 10 */
  {96, 16, 0.940f, 0.945f, KAISER10},   /* Q6 *//* 91.5% cutoff (~100 dB stop) 10 */
  {128, 16, 0.950f, 0.950f, KAISER10},  /* Q7 *//* 93.1% cutoff (~100 dB stop) 10 */
  {160, 16, 0.960f, 0.960f, KAISER10},  /* Q8 *//* 94.5% cutoff (~100 dB stop) 10 */
  {192, 32, 0.968f, 0.968f, KAISER12},  /* Q9 *//* 95.5% cutoff (~100 dB stop) 10 */
  {256, 32, 0.975f, 0.975f, KAISER12},  /* Q10 *//* 96.6% cutoff (~100 dB stop) 10 */
};
static double
compute_func (float x, struct FuncDef *func)
{
  float y, frac;
  double interp[4];
  int ind;
  y = x * func->oversample;
  ind = (int) floor (y);
  frac = (y - ind);
  /* CSE with handle the repeated powers */
  interp[3] = -0.1666666667 * frac + 0.1666666667 * (frac * frac * frac);
  interp[2] = frac + 0.5 * (frac * frac) - 0.5 * (frac * frac * frac);
  /*interp[2] = 1.f - 0.5f*frac - frac*frac + 0.5f*frac*frac*frac; */
  interp[0] =
      -0.3333333333 * frac + 0.5 * (frac * frac) -
      0.1666666667 * (frac * frac * frac);
  /* Just to make sure we don't have rounding problems */
  interp[1] = 1.f - interp[3] - interp[2] - interp[0];

  /*sum = frac*accum[1] + (1-frac)*accum[2]; */
  return interp[0] * func->table[ind] + interp[1] * func->table[ind + 1] +
      interp[2] * func->table[ind + 2] + interp[3] * func->table[ind + 3];
}


/*8,24,40,56,80,104,128,160,200,256,320*/
static float
sinc (float cutoff, float x, int N, struct FuncDef *window_func)
{
  /*fprintf (stderr, "%f ", x); */
  float xx = x * cutoff;
  if (fabs (x) < 1e-6)
    return cutoff;
  else if (fabs (x) > .5 * N)
    return 0;
  /*FIXME: Can it really be any slower than this? */
  return cutoff * sin (M_PI * xx) / (M_PI * xx) * compute_func (fabs (2. * x /
          N), window_func);
}

int sinc_function (float *sinc_table, gint filt_len, float cutoff, gint den_rate, gint quality)
{
	int i, j;
	for (i = 0; i < den_rate; i++) {
		for (j = 0; j < filt_len; j++) {
			sinc_table[i * filt_len + j] =
            sinc (cutoff, ((j -  filt_len / 2 + 1) -
                ((float) i) / den_rate), filt_len,
            quality_map[quality].window_func);
//	    printf("%s init sinc[%d] = %e\n", quality > 1 ? "down" : "up", i * filt_len + j, sinc_table[i * filt_len + j]);
      		}
	}
	return 0;

}

/*
 * ============================================================================
 *
 *			End of the copied part 
 *
 * ============================================================================
 */



/* CURRENTLY DEPRECATED : This is the code written by NIMS for a simpler sinc table*/
#if 0
int sinc_function(float *sinc, gint sinc_len, gint times)
{
	if (sinc_len & 0x1)
		return -1;

	gint i;
	for (i = -(sinc_len/2); i < (sinc_len/2); ++i)
		sinc[sinc_len/2 + i] = sin(i * (1.0 / times) * M_PI) / (i * (1.0 / times) * M_PI);
	sinc[sinc_len / 2] = 1;

	return 0;
}

void psinc_function(float *sinc_table, int filtersize, int times)
{
	int c = 1 - times;
	int sincsize = filtersize * times;
	float *sinc = (float*)malloc(sizeof(float)*sincsize);
	sinc_function(sinc, sincsize, times);

	for (int i = 0; i < filtersize; ++i) {
		for (int j = times - 1; j >= 0; --j) {
			if (c < 0) {
				sinc_table[j * filtersize + i] = 0.0;
				++c;	
			} else {
				sinc_table[j * filtersize + i] = sinc[c++];
			}
		}	
	}
}
#endif

/* 
 * See gstlal/ python/ pipeparts/ __init__.py : audioresample_variance_gain
 * for more information
 */
static const float amplifier_down_map[11] = {
			0.7224862140943990596,
			0.7975021342935247892,
			0.8547537598970208483,
			0.8744072146753004704,
			0.9075294214410336568,
			0.9101523813406768859,
			0.9280549396020538744,
			0.9391809530012216189,
			0.9539276644089494939,
			0.9623083437067311285,
			0.9684700588501590213,
			};

static const float amplifier_up_map[11] = {
			0.7539740617648067467,
			0.8270076656536116122,
			0.8835072979478705291,
			0.8966758456219333651,
			0.9253434087537378838,
			0.9255866674042573239,
			0.9346487800036394900,
			0.9415331868209220190,
			0.9524608799160205752,
			0.9624372769883490220,
			0.9704505626409354324,
			};


float resampler_state_amplifier_init (gint quality, gint inrate, gint outrate, gint depth)
{
	float amplifier = 0.0;
		
	if (inrate > outrate) {
		if (depth > 0)
			amplifier = (float) outrate / (float) inrate;
		else
			amplifier = amplifier_down_map[quality] * (float) outrate / (float) inrate; 
	}
       	else if (inrate < outrate)
		amplifier = amplifier_up_map[quality];
	else
		amplifier = 1.0;

	return 1/sqrt(amplifier);
}

ResamplerState *
resampler_state_create (gint inrate, gint outrate, gint channels, gint num_exe_samples, gint num_cover_samples, gint depth, cudaStream_t stream)
{
	gint mem_alloc_size, num_alloc_samples; 
	gint den_rate; // denominator rate, = outrate / gcd (inrate, outrate) 
	gint times = 2; // resampler times
	float cutoff;
	ResamplerState *state = (ResamplerState *)malloc(sizeof(ResamplerState));
	state->inrate = inrate;
	state->outrate = outrate;
	if (inrate > outrate){
	  cutoff = quality_map[DOWN_QUALITY].downsample_bandwidth / times;
	  den_rate = 1;

	  state->filt_len = quality_map[DOWN_QUALITY].base_length;
	  state->filt_len = state->filt_len * times;
	  state->sinc_len = state->filt_len * den_rate;


	  CUDA_CHECK(cudaMalloc((void **) &(state->d_sinc_table), sizeof(float) * state->sinc_len));
	  float *sinc_table = (float *)malloc (sizeof(float) * state->sinc_len);
	  /* Sinc function Generator */
	  sinc_function(sinc_table, state->filt_len, cutoff, den_rate, DOWN_QUALITY);
          CUDA_CHECK(cudaMemcpyAsync(state->d_sinc_table, sinc_table, state->sinc_len * sizeof(float), cudaMemcpyHostToDevice, stream));
  	  free(sinc_table);
	  sinc_table = NULL;

	  state->amplifier = resampler_state_amplifier_init (DOWN_QUALITY, inrate, outrate, depth);
	} else {
	  cutoff = quality_map[UP_QUALITY].upsample_bandwidth;
	  den_rate = 2;

       	  state->filt_len = quality_map[UP_QUALITY].base_length;
	  state->sinc_len = state->filt_len * den_rate;

	  CUDA_CHECK(cudaMalloc((void **) &(state->d_sinc_table), sizeof(float) * state->sinc_len));
	  float *sinc_table = (float *)malloc (sizeof(float) * state->sinc_len);
	  /* Sinc function Generator */
	  sinc_function(sinc_table, state->filt_len, cutoff, den_rate, UP_QUALITY);
	  //psinc_function(sinc_table, state->filt_len, resolution);
          CUDA_CHECK(cudaMemcpyAsync(state->d_sinc_table, sinc_table, state->sinc_len * sizeof(float), cudaMemcpyHostToDevice, stream));
 	  free(sinc_table);
	  sinc_table = NULL;

	  state->amplifier = resampler_state_amplifier_init (UP_QUALITY, inrate, outrate, depth);

	}
	state->channels = channels;
	num_alloc_samples = MAX(num_exe_samples, num_cover_samples)/pow(2, depth) + 1;// prevent overflow

	state->mem_len = state->filt_len - 1 + num_alloc_samples;
	mem_alloc_size = state->mem_len * channels * sizeof(float);
	CUDA_CHECK(cudaMalloc((void **) &(state->d_mem), mem_alloc_size));

	//state->mem = (float *)malloc(mem_alloc_size);
	CUDA_CHECK(cudaMemsetAsync(state->d_mem, 0, mem_alloc_size, stream));
	state->last_sample = state->filt_len/2;
	//GST_LOG ("flit len:%d, sinc len %d, amplifier %d, mem len %d%d", state->filt_len, state->sinc_len, state->amplifier, state->mem_len, state->channels);
	//printf("inrate %d, outrate %d, amplifier %f\n", inrate, outrate, state->amplifier);
	return state;
}

void 
resampler_state_reset (ResamplerState *state, cudaStream_t stream)
{
	gint mem_alloc_size = state->mem_len * state->channels * sizeof(float);
	CUDA_CHECK(cudaMemsetAsync(state->d_mem, 0, mem_alloc_size, stream));
	state->last_sample = state->filt_len/2;

}
void
resampler_state_destroy (ResamplerState *state)
{
  if (state->d_sinc_table)
    cudaFree(state->d_sinc_table) ;
  cudaFree(state->d_mem) ;
}



static COMPLEX_F *
spiir_state_workspace_realloc_complex (COMPLEX_F ** workspace, int * len,
    int new_len)
{
  COMPLEX_F *new;
  if (new_len <= *len)
    /* no need to resize */
    return *workspace;
  new = (COMPLEX_F *)realloc (*workspace, new_len * sizeof (COMPLEX_F));
  if (!new)
    /* failure (re)allocating memeory */
    return NULL;
  /* success */
  *workspace = new;
  *len = new_len;
  return *workspace;
}

static int *
spiir_state_workspace_realloc_int (int ** workspace, int * len,
    int new_len)
{
  int *new;
  if (new_len <= *len)
    /* no need to resize */
    return *workspace;
  new = (int *)realloc (*workspace, new_len * sizeof (int));
  if (!new)
    /* failure (re)allocating memeory */
    return NULL;
  /* success */
  *workspace = new;
  *len = new_len;
  return *workspace;
}

void
spiir_state_load_bank( SpiirState **spstate, const char *filename, guint ndepth, gint maxrate, cudaStream_t stream)
{

	XmlNodeStruct	*inxns		= (XmlNodeStruct*)malloc(sizeof(XmlNodeStruct)*ndepth*3);
	XmlArray	*d_array	= (XmlArray*)malloc(sizeof(XmlArray)*ndepth);
	XmlArray	*a_array	= (XmlArray*)malloc(sizeof(XmlArray)*ndepth);
	XmlArray	*b_array	= (XmlArray*)malloc(sizeof(XmlArray)*ndepth);
	guint i;
	for (i = 0; i < ndepth; ++i)
	{
		// configure d_array 
		d_array[i].ndim = 0;
		sprintf((char *)inxns[i + 0 * ndepth].tag, "d_%d:array", maxrate >> i);
		inxns[i + 0 * ndepth].processPtr = readArray;
		inxns[i + 0 * ndepth].data = d_array + i;

		// configure a_array
		a_array[i].ndim = 0;
		sprintf((char *)inxns[i + 1 * ndepth].tag, "a_%d:array", maxrate >> i);
		inxns[i + 1 * ndepth].processPtr = readArray;
		inxns[i + 1 * ndepth].data = a_array + i;	

		// configure b_array
		b_array[i].ndim = 0;
		sprintf((char *)inxns[i + 2 * ndepth].tag, "b_%d:array", maxrate >> i);
		inxns[i + 2 * ndepth].processPtr = readArray;
		inxns[i + 2 * ndepth].data = b_array + i;	
	}

	// start parsing xml file, get the requested array
	parseFile(filename, inxns, ndepth * 3);


	// free array memory 
	int num_filters, num_templates;
	COMPLEX_F *tmp_a1 = NULL, *tmp_b0 = NULL;
       	int *tmp_d = NULL, tmp_max = 0, cur_d = 0;
	int a1_len = 0, b0_len = 0, d_len = 0;
	int eff_len = 0;

	int j, k;
	for (i = 0; i < ndepth; ++i)
	{
		num_filters		= (gint)d_array[i].dim[0];
		num_templates	= (gint)d_array[i].dim[1];
		eff_len = num_filters * num_templates;
		spiir_state_workspace_realloc_complex (&tmp_a1, &a1_len, eff_len);
		spiir_state_workspace_realloc_complex (&tmp_b0, &b0_len, eff_len);
		spiir_state_workspace_realloc_int (&tmp_d, &d_len, eff_len);

		// spstate[i]->d_d = (long*)inxns[i].data;
		//printf("%d - d_dim: (%d, %d) a_dim: (%d, %d) b_dim: (%d, %d)\n", i, d_array[i].dim[0], d_array[i].dim[1],
		//		a_array[i].dim[0], a_array[i].dim[1], b_array[i].dim[0], b_array[i].dim[1]);

		//printf("eff_len %d\n", eff_len);
		spstate[i]->num_filters		= num_filters;
		spstate[i]->num_templates	= num_templates;

		for (j = 0; j < num_filters; ++j)
		{
			for (k = 0; k < num_templates; ++k)
			{
				tmp_d[k * num_filters + j] = (int)(((long*)(d_array[i].data))[j * num_templates + k]);
				cur_d = tmp_d[k * num_filters + j] ;
				tmp_max = cur_d > tmp_max ? cur_d : tmp_max;
				tmp_a1[k * num_filters + j].re = (float)(((double*)(a_array[i].data))[j * 2 * num_templates + k]);
				tmp_a1[k * num_filters + j].im = (float)(((double*)(a_array[i].data))[j * 2 * num_templates + num_templates + k]);
				tmp_b0[k * num_filters + j].re = (float)(((double*)(b_array[i].data))[j * 2 * num_templates + k]);
				tmp_b0[k * num_filters + j].im = (float)(((double*)(b_array[i].data))[j * 2 * num_templates + num_templates + k]);
			}
		}
		spstate[i]->delay_max = tmp_max;
		CUDA_CHECK(cudaMalloc((void **) &(spstate[i]->d_a1), eff_len * sizeof (COMPLEX_F)));
		CUDA_CHECK(cudaMemcpyAsync(spstate[i]->d_a1, tmp_a1, eff_len * sizeof(COMPLEX_F), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMalloc((void **) &(spstate[i]->d_b0), eff_len * sizeof (COMPLEX_F)));
		CUDA_CHECK(cudaMemcpyAsync(spstate[i]->d_b0, tmp_b0, eff_len * sizeof(COMPLEX_F), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMalloc((void **) &(spstate[i]->d_d), eff_len * sizeof (int)));
		CUDA_CHECK(cudaMemcpyAsync(spstate[i]->d_d, tmp_d, eff_len * sizeof(int), cudaMemcpyHostToDevice, stream));

		CUDA_CHECK(cudaMalloc((void **) &(spstate[i]->d_y), eff_len * sizeof (COMPLEX_F)));

		CUDA_CHECK(cudaMemsetAsync(spstate[i]->d_y, 0, eff_len * sizeof(COMPLEX_F), stream));
	
		freeArray(d_array + i);
		freeArray(a_array + i);
		freeArray(b_array + i);

		//printf("2st a: (%.3f + %.3fi) 2st b: (%.3f + %.3fi) 2st d: %d\n", tmp_a1[1].re, tmp_a1[1].im,
			//	tmp_b0[1].re, tmp_b0[1].im, tmp_d[1]);
		CUDA_CHECK(cudaPeekAtLastError());
	}
	
	free(inxns);
	free(tmp_a1);
	free(tmp_b0);
	free(tmp_d);

    xmlCleanupParser();
    xmlMemoryDump();
}

SpiirState **
spiir_state_create (const gchar *bank_fname, guint ndepth, guint rate, guint num_head_cover_samples,
		gint num_exe_samples, cudaStream_t stream)
{

	//printf("init spstate\n");
	guint i;	
	gint inrate, outrate, queue_alloc_size;
	SpiirState ** spstate = (SpiirState **)malloc(ndepth * sizeof(SpiirState*));

	for(i=0; i<ndepth; i++)
	{
		SPSTATE(i) = (SpiirState *)malloc(sizeof(SpiirState));
		SPSTATE(i)->depth = i;

	}

	spiir_state_load_bank(spstate, bank_fname, ndepth, rate, stream);

	gint outchannels = spstate[0]->num_templates * 2;
	for(i=0; i<ndepth; i++)
	{
		inrate = rate/pow(2, i);
		outrate = inrate / 2;

		SPSTATEDOWN(i) = resampler_state_create (inrate, outrate, 1, num_exe_samples, num_head_cover_samples, i, stream);
		SPSTATEUP(i) = resampler_state_create (outrate, inrate, outchannels, num_exe_samples, num_head_cover_samples, i, stream);
		g_assert (SPSTATEDOWN(i) != NULL);
		g_assert (SPSTATEUP(i) != NULL);

	}

	for(i=0; i<ndepth; i++) {

		SPSTATE(i)->nb = 0;
		SPSTATE(i)->pre_out_spiir_len = 0;
		SPSTATE(i)->queue_len = (2 * num_head_cover_samples + num_exe_samples) / pow (2, i) + 1 + SPSTATE(i)->delay_max; 
		SPSTATE(i)->queue_first_sample = 0;
		SPSTATE(i)->queue_last_sample = SPSTATE(i)->delay_max;
		queue_alloc_size = SPSTATE(i)->queue_len* sizeof(float);
		CUDA_CHECK(cudaMalloc((void **) &(SPSTATE(i)->d_queue), queue_alloc_size));
		CUDA_CHECK(cudaMemsetAsync(SPSTATE(i)->d_queue, 0, queue_alloc_size, stream));

	}
	// FIXME: this d_out will cost large GPU memory, find a way to avoid it
	int out_alloc_size = MAX(num_exe_samples, num_head_cover_samples) * outchannels * sizeof(float);

	//printf("out_alloc_size %d\n", out_alloc_size);

	CUDA_CHECK(cudaMalloc((void **) &(SPSTATE(0)->d_out), out_alloc_size)); // for the output
	CUDA_CHECK(cudaMemsetAsync(SPSTATE(0)->d_out, 0, out_alloc_size, stream));
	CUDA_CHECK(cudaPeekAtLastError());
	return spstate;
}

void 
spiir_state_destroy (SpiirState ** spstate, guint num_depths)
{
	guint i;
       	for(i=0; i<num_depths; i++)
	{
		resampler_state_destroy (SPSTATEDOWN(i));
		resampler_state_destroy (SPSTATEUP(i));
		free(SPSTATEDOWN(i));
		free(SPSTATEUP(i));
		cudaFree(SPSTATE(i)->d_queue);
		free(SPSTATE(i));

	}
}

void
spiir_state_reset (SpiirState **spstate, guint num_depths, cudaStream_t stream)
{
  guint i = 0;
  for(i=0; i<num_depths; i++)
  {
    SPSTATE(i)->pre_out_spiir_len = 0;
    int eff_len = SPSTATE(i)->num_filters * SPSTATE(i)->num_templates;

    CUDA_CHECK(cudaMemsetAsync(SPSTATE(i)->d_queue, 0, SPSTATE(i)->queue_len * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(SPSTATE(i)->d_y, 0, eff_len * sizeof(COMPLEX_F), stream));

    SPSTATE(i)->queue_first_sample = 0;
    SPSTATE(i)->queue_last_sample = SPSTATE(i)->delay_max;

    resampler_state_reset(SPSTATEDOWN(i), stream);
    CUDA_CHECK(cudaPeekAtLastError());
    resampler_state_reset(SPSTATEUP(i), stream);
    CUDA_CHECK(cudaPeekAtLastError());
  }
}

gint
spiir_state_get_outlen (SpiirState **spstate, gint in_len, guint num_depths) {
  guint i;
  for (i=0; i<num_depths-1; i++) 
   in_len = (in_len - SPSTATEDOWN(i)->last_sample)/2; 

  for (i=num_depths-1; i>0; i--) 
   in_len = (in_len - SPSTATEUP(i)->last_sample)*2; 

  return in_len;
}


/* DEPRECATED */
#if 0
void
spiir_state_load_bank (SpiirState **spstate, guint num_depths, gdouble *bank, gint bank_len, cudaStream_t stream)
{

	COMPLEX_F *tmp_a1 = NULL, *tmp_b0 = NULL;
       	int *tmp_d = NULL, tmp_max = 0;
	int a1_len = 0, b0_len = 0, d_len = 0;
	int a1_eff_len = 0, b0_eff_len = 0, d_eff_len = 0;
	gint i, depth;
	gint pos = 1; //start position, the 0 is for number of depths

	for (depth = num_depths - 1; depth >= 0; depth--) {
		SPSTATE(depth)->num_templates = (gint) bank[pos];
		SPSTATE(depth)->num_filters = (gint) bank[pos+1]/2;
		//printf("depth %d, ntemplates %d, nfilters %d\n", depth, SPSTATE(depth)->num_templates, SPSTATE(depth)->num_filters);

		/* 
		 * initiate coefficient a1
		 */
		if (SPSTATE(depth)->num_templates > 0) {
			//printf("dpt %d\n",depth);
		a1_eff_len = (gint) bank[pos] * bank[pos+1]/2;	
		pos = pos + 2;
		spiir_state_workspace_realloc_complex (&tmp_a1, &a1_len, a1_eff_len);

		for (i=0; i<a1_eff_len; i++) {
			tmp_a1[i].re = (float) bank[pos++];
//			printf("a matrix %d, re %e \n", i, tmp_a1[i].re);
			tmp_a1[i].im = (float) bank[pos++];
//			printf("a matrix %d, im %e \n", i, tmp_a1[i].im);
		}

		cudaMalloc((void **) &(SPSTATE(depth)->d_a1), a1_eff_len * sizeof (COMPLEX_F));

		cudaMemcpyAsync(SPSTATE(depth)->d_a1, tmp_a1, a1_eff_len * sizeof(COMPLEX_F), cudaMemcpyHostToDevice, stream);
		/* 
		 * initiate coefficient b0
		 */
		b0_eff_len = (gint) bank[pos] * bank[pos+1]/2;	
		pos = pos + 2;
		spiir_state_workspace_realloc_complex (&tmp_b0, &b0_len, b0_eff_len);

		for (i=0; i<b0_eff_len; i++) {
			tmp_b0[i].re = (float) bank[pos++];
			tmp_b0[i].im = (float) bank[pos++];
		}

		cudaMalloc((void **) &(SPSTATE(depth)->d_b0), b0_eff_len * sizeof (COMPLEX_F));

		cudaMemcpyAsync(SPSTATE(depth)->d_b0, tmp_b0, b0_eff_len * sizeof(COMPLEX_F), cudaMemcpyHostToDevice, stream);
		/* 
		 * initiate coefficient d (delay)
		 */

		d_eff_len = (gint) bank[pos] * bank[pos+1];

		pos = pos + 2;
		spiir_state_workspace_realloc_int (&tmp_d, &d_len, d_eff_len);

		tmp_max = (int)bank[pos];
		for (i=0; i<d_eff_len; i++) {
			tmp_d[i] = (int) bank[pos++];
			tmp_max = tmp_d[i] > tmp_max ? tmp_d[i] : tmp_max;
		}

		SPSTATE(depth)->delay_max = tmp_max;
		cudaMalloc((void **) &(SPSTATE(depth)->d_d), d_eff_len * sizeof (int));

		cudaMemcpyAsync(SPSTATE(depth)->d_d, tmp_d, d_eff_len * sizeof(int), cudaMemcpyHostToDevice, stream);
		/* 
		 * initiate previous output y
		 */

		cudaMalloc((void **) &(SPSTATE(depth)->d_y), a1_eff_len * sizeof (COMPLEX_F));

		cudaMemsetAsync(SPSTATE(depth)->d_y, 0, a1_eff_len * sizeof(COMPLEX_F), stream);
		} else {
			SPSTATE(depth)->d_a1 = NULL;
			SPSTATE(depth)->d_b0 = NULL;
			SPSTATE(depth)->d_d = NULL;
			SPSTATE(depth)->d_y = NULL;
			SPSTATE(depth)->delay_max = 0;
		}
	}
	if (tmp_a1)
		free (tmp_a1);
	if (tmp_b0)
		free (tmp_b0);
	if (tmp_d)
		free (tmp_d);
	//g_assert (pos == bank_len);
        //gpuErrchk (cudaPeekAtLastError ());


}

SpiirState ** 
spiir_state_create (gdouble *bank, gint bank_len, guint num_head_cover_samples,
		guint num_exe_samples, gint width, guint rate, cudaStream_t stream)
{

	//printf("init spstate\n");
	gint i, inrate, outrate, queue_alloc_size;
	gint num_depths = (gint) bank[0];
	gint outchannels = (gint) bank[1] * 2;
	SpiirState ** spstate = (SpiirState **)malloc(num_depths * sizeof(SpiirState*));

	for(i=0; i<num_depths; i++)
	{
		SPSTATE(i) = (SpiirState *)malloc(sizeof(SpiirState));
		SPSTATE(i)->depth = i;
		inrate = rate/pow(2, i);
		outrate = inrate / 2;

		SPSTATEDOWN(i) = resampler_state_create (inrate, outrate, 1, num_exe_samples, num_head_cover_samples, i, stream);
		SPSTATEUP(i) = resampler_state_create (outrate, inrate, outchannels, num_exe_samples, num_head_cover_samples, i, stream);
		g_assert (SPSTATEDOWN(i) != NULL);
	    g_assert (SPSTATEUP(i) != NULL);

	}
	spiir_state_load_bank (spstate, num_depths, bank, bank_len, stream);

	for(i=0; i<num_depths; i++) {

		SPSTATE(i)->nb = 0;
		SPSTATE(i)->pre_out_spiir_len = 0;
		SPSTATE(i)->queue_len = (2 * num_head_cover_samples + num_exe_samples) / pow (2, i) + 1 + SPSTATE(i)->delay_max; 
		SPSTATE(i)->queue_first_sample = 0;
		SPSTATE(i)->queue_last_sample = SPSTATE(i)->delay_max;
		queue_alloc_size = SPSTATE(i)->queue_len* sizeof(float);
		cudaMalloc((void **) &(SPSTATE(i)->d_queue), queue_alloc_size);
		cudaMemsetAsync(SPSTATE(i)->d_queue, 0, queue_alloc_size, stream);

	}

	return spstate;
}

void
spiir_state_flush_queue (SpiirState **spstate, gint depth, gint
		num_flush_samples)
{
  int i;
  gint queue_len = SPSTATE(depth)->queue_len;
  float *pos_queue = SPSTATE(depth)->queue;

  for (i=0; i<queue_len - num_flush_samples; i++) 
	  pos_queue[i] = pos_queue[i + num_flush_samples];

  SPSTATE(depth)->queue_len = SPSTATE(depth)->queue_len - num_flush_samples;
  SPSTATE(depth)->queue_last_sample = SPSTATE(depth)->queue_last_sample - num_flush_samples;
}
#endif

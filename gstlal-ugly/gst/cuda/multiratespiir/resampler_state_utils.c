#ifdef __cplusplus
extern "C" {
#endif

#include <string.h>
#include <math.h>
#include <stdio.h>

#include "resampler_state_macro.h"
#include "resampler_state_utils.h"
#include <glib.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
}
#endif

/* 
 * The following part is copied/ rewritten from resample.c
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

}
/* 
 * End of the copied part 
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


	  cudaMalloc((void **) &(state->d_sinc_table), sizeof(float) * state->sinc_len);
	  float *sinc_table = (float *)malloc (sizeof(float) * state->sinc_len);
	  /* Sinc function Generator */
	  sinc_function(sinc_table, state->filt_len, cutoff, den_rate, DOWN_QUALITY);
          cudaMemcpyAsync(state->d_sinc_table, sinc_table, state->sinc_len * sizeof(float), cudaMemcpyHostToDevice, stream);
  	  free(sinc_table);
	  sinc_table = NULL;

	  state->amplifier = resampler_state_amplifier_init (DOWN_QUALITY, inrate, outrate, depth);
	} else {
	  cutoff = quality_map[UP_QUALITY].upsample_bandwidth;
	  den_rate = 2;

       	  state->filt_len = quality_map[UP_QUALITY].base_length;
	  state->sinc_len = state->filt_len * den_rate;

	  cudaMalloc((void **) &(state->d_sinc_table), sizeof(float) * state->sinc_len);
	  float *sinc_table = (float *)malloc (sizeof(float) * state->sinc_len);
	  /* Sinc function Generator */
	  sinc_function(sinc_table, state->filt_len, cutoff, den_rate, UP_QUALITY);
//	  psinc_function(sinc_table, state->filt_len, resolution);
          cudaMemcpyAsync(state->d_sinc_table, sinc_table, state->sinc_len * sizeof(float), cudaMemcpyHostToDevice, stream);
 	  free(sinc_table);
	  sinc_table = NULL;

	  state->amplifier = resampler_state_amplifier_init (UP_QUALITY, inrate, outrate, depth);

	}
	state->channels = channels;
	num_alloc_samples = MAX(num_exe_samples, num_cover_samples)/pow(2, depth) + 1;// prevent overflow

	state->mem_len = state->filt_len - 1 + num_alloc_samples;
	mem_alloc_size = state->mem_len * channels * sizeof(float);
	cudaMalloc((void **) &(state->d_mem), mem_alloc_size);

//	state->mem = (float *)malloc(mem_alloc_size);
	cudaMemsetAsync(state->d_mem, 0, mem_alloc_size, stream);
	state->last_sample = state->sinc_len/2;
//	GST_LOG ("flit len:%d, sinc len %d, amplifier %d, mem len %d%d", state->filt_len, state->sinc_len, state->amplifier, state->mem_len, state->channels);
//	printf("inrate %d, outrate %d, amplifier %f\n", inrate, outrate, state->amplifier);
	return state;
}

void 
resampler_state_reset (ResamplerState *state, cudaStream_t stream)
{
	gint mem_alloc_size = state->mem_len * state->channels * sizeof(float);
	cudaMemsetAsync(state->d_mem, 0, mem_alloc_size, stream);
	state->last_sample = state->filt_len/2;

}
void
resampler_state_destroy (ResamplerState *state)
{
  if (state->d_sinc_table)
    cudaFree(state->d_sinc_table) ;
  cudaFree(state->d_mem) ;
}



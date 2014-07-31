#ifdef __cplusplus
extern "C" {
#endif

#include <glib.h>
#include <math.h>
#include "resampler_state_utils.h"
#include "multiratespiir.h"
#include "spiir_state_utils.h"
#include "spiir_state_macro.h"
#include <cuda_runtime.h>

#ifdef __cplusplus
}
#endif

// for gpu debug
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
static void gpuAssert(cudaError_t code, char *file, int line)
{
   if (code != cudaSuccess) 
   {
      printf ("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      exit(code);
   }
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
spiir_state_load_bank (SpiirState **spstate, gint num_depths, gdouble *bank, gint bank_len, cudaStream_t stream)
{
	cudaSetDevice(1);

	COMPLEX_F *tmp_a1 = NULL, *tmp_b0 = NULL;
       	int *tmp_d = NULL, tmp_max = 0;
	int a1_len = 0, b0_len = 0, d_len = 0;
	int a1_eff_len = 0, b0_eff_len = 0, d_eff_len = 0;
	gint i, depth;
	gint pos = 1; //start position, the 0 is for number of depths

	for (depth = num_depths - 1; depth >= 0; depth--) {
		SPSTATE(depth)->num_templates = (gint) bank[pos];
		SPSTATE(depth)->num_filters = (gint) bank[pos+1]/2;

		/* 
		 * initiate coefficient a1
		 */
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

		SPSTATE(depth)->d_max = tmp_max;
		cudaMalloc((void **) &(SPSTATE(depth)->d_d), d_eff_len * sizeof (int));

		cudaMemcpyAsync(SPSTATE(depth)->d_d, tmp_d, d_eff_len * sizeof(int), cudaMemcpyHostToDevice, stream);
		/* 
		 * initiate previous output y
		 */

		cudaMalloc((void **) &(SPSTATE(depth)->d_y), a1_eff_len * sizeof (COMPLEX_F));

		cudaMemsetAsync(SPSTATE(depth)->d_y, 0, a1_eff_len * sizeof(COMPLEX_F), stream);
	}
	free (tmp_a1);
	free (tmp_b0);
	free (tmp_d);
	g_assert (pos == bank_len);
        gpuErrchk (cudaPeekAtLastError ());


}

SpiirState ** 
spiir_state_init (gdouble *bank, gint bank_len, gint num_cover_samples,
		gint num_exe_samples, gint width, gint rate, cudaStream_t stream)
{
	cudaSetDevice(1);

	printf("init spstate\n");
	gint i, inrate, outrate, queue_alloc_size;
	gint num_depths = (gint) bank[0];
	gint outchannels = (gint) bank[1] * 2;

	SpiirState ** spstate = (SpiirState **)malloc(num_depths * sizeof(SpiirState*));

	for(i=0; i<num_depths; i++)
	{
		SPSTATE(i) = (SpiirState *)malloc(sizeof(SpiirState));
		SPSTATE(i)->depth = i;
		SPSTATE(i)->queue_len = (2 * num_cover_samples + num_exe_samples) / pow (2, i) + 1; 
		SPSTATE(i)->queue_eff_len = 0;
		SPSTATE(i)->queue_down_start = 0;
		queue_alloc_size = SPSTATE(i)->queue_len* sizeof(float);
		cudaMalloc((void **) &(SPSTATE(i)->d_queue), queue_alloc_size);
		cudaMemsetAsync(SPSTATE(i)->d_queue, 0, queue_alloc_size, stream);

        gpuErrchk (cudaPeekAtLastError ());
		inrate = rate/pow(2, i);
		outrate = inrate / 2;
//		SPSTATE(i)->out_spiir = (float*)malloc(tmp_len * sizeof(float));
//		tmp_len *=2;
//		SPSTATE(i)->out_up = (float*)malloc(tmp_len * sizeof(float));

		SPSTATEDOWN(i) = resampler_state_init (inrate, outrate, 1, num_exe_samples, num_cover_samples, i, stream);
		SPSTATEUP(i) = resampler_state_init (outrate, inrate, outchannels, num_exe_samples, num_cover_samples, i, stream);
		g_assert (SPSTATEDOWN(i) != NULL);
	       	g_assert (SPSTATEUP(i) != NULL);

	}
	spiir_state_load_bank (spstate, num_depths, bank, bank_len, stream);
	for(i=0; i<num_depths; i++) {
		SPSTATE(i)->nb = 0;
		SPSTATE(i)->pre_out_spiir_len = 0;
		SPSTATE(i)->queue_spiir_last_sample = 0;
		SPSTATE(i)->queue_spiir_len = SPSTATE(i)->queue_len + SPSTATE(i)->d_max;
		cudaMalloc((void **) &(SPSTATE(i)->d_queue_spiir), SPSTATE(i)->queue_spiir_len * sizeof(float));
		cudaMemsetAsync(SPSTATE(i)->d_queue_spiir, 0, SPSTATE(i)->queue_spiir_len * sizeof(float), stream);
	}

	return spstate;
}

void 
spiir_state_destroy (SpiirState ** spstate, gint num_depths)
{
	gint i; for(i=0; i<num_depths; i++)
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
spiir_state_reset (SpiirState **spstate, gint num_depths, cudaStream_t stream)
{
  int i;
  for(i=0; i<num_depths; i++)
  {
  SPSTATE(i)->pre_out_spiir_len = 0;
  SPSTATE(i)->queue_spiir_last_sample = 0;

    SPSTATE(i)->queue_eff_len = 0;
    SPSTATE(i)->queue_down_start = 0;
    resampler_state_reset(SPSTATEDOWN(i), stream);
    resampler_state_reset(SPSTATEUP(i), stream);
  }
}

gint
spiir_state_get_outlen (SpiirState **spstate, gint in_len, gint num_depths) {
  int i;
  for (i=0; i<num_depths-1; i++) 
   in_len = (in_len - SPSTATEDOWN(i)->last_sample)/2; 

  for (i=num_depths-1; i>0; i--) 
   in_len = (in_len - SPSTATEUP(i)->last_sample)*2; 

  return in_len;
}


#if 0
/* DEPRECATED */
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
  SPSTATE(depth)->queue_eff_len = SPSTATE(depth)->queue_eff_len - num_flush_samples;
  SPSTATE(depth)->queue_down_start = SPSTATE(depth)->queue_down_start - num_flush_samples;
}
#endif

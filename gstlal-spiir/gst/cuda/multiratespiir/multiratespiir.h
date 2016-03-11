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


#ifndef __CUDA_MULTIRATE_SPIIR_H__
#define __CUDA_MULTIRATE_SPIIR_H__

#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <gst/base/gstadapter.h>

#include <cuda_runtime.h>


G_BEGIN_DECLS

#define CUDA_TYPE_MULTIRATE_SPIIR \
  (cuda_multirate_spiir_get_type())
#define CUDA_MULTIRATE_SPIIR(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),CUDA_TYPE_MULTIRATE_SPIIR,CudaMultirateSPIIR))
#define CUDA_MULTIRATE_SPIIR_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),CUDA_TYPE_MULTIRATE_SPIIR,CudaMultirateSPIIRClass))
#define GST_IS_CUDA_MULTIRATE_SPIIR(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),CUDA_TYPE_MULTIRATE_SPIIR))
#define GST_IS_CUDA_MULTIRATE_SPIIR_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),CUDA_TYPE_MULTIRATE_SPIIR))

typedef struct _CudaMultirateSPIIR CudaMultirateSPIIR;
typedef struct _CudaMultirateSPIIRClass CudaMultirateSPIIRClass;

#ifndef DEFINED_COMPLEX_F
#define DEFINED_COMPLEX_F 

typedef struct _Complex_F{
	float re;
	float im;
} COMPLEX_F;

#else
#endif


typedef struct _ResamplerState{
  float *d_sinc_table;
  float *d_mem; 		/* fixed length to store input */
  gint channels;
  gint mem_len;
  gint last_sample;
  gint filt_len;
  gint sinc_len;
  gint inrate;
  gint outrate;
  float amplifier; 		/* correction factor for resampling */
} ResamplerState;

typedef struct _SpiirState {
  COMPLEX_F *d_a1;
  COMPLEX_F *d_b0;
  int *d_d;
  gint delay_max;
  COMPLEX_F *d_y;

  guint nb;
  gint num_filters;
  gint num_templates;

  gint depth; 			/* supposed to be 0-6 */
  ResamplerState *downstate, *upstate;
  float *d_queue; 		/* circular buffer (or ring buffer) for downsampler and spiir */
  float *d_out;			/* only apply to 0 depth */
  gint queue_len;
  gint queue_first_sample;  	/* queue start position */
  gint queue_last_sample;  	/* queue end position */
  gint pre_out_spiir_len;	/* previous output length for spiir filtering */
} SpiirState;

/* single-precision bank */
typedef struct _SpiirBank_s {
  float *a1_s;
  float *b0_s;
  int *d_s;

  unsigned int num_templates;
  unsigned int num_filters;
  unsigned int rate;
  unsigned int depth;
} SpiirBank_s;


/**
 * CudaMultirateSPIIR:
 *
 * Opaque data structure.
 */
struct _CudaMultirateSPIIR {
  GstBaseTransform element;

  /* <private> */

  GstAdapter *adapter;

  gboolean need_discont;
  guint num_depths;
  guint num_head_cover_samples; /* number of samples needed to produce the first buffer */
  guint num_tail_cover_samples; /* number of samples needed to produce the last buffer */
  gint num_exe_samples; 	/* number of samples executed every time after first buffer */

  GstClockTime t0;
  guint64 offset0;
  guint64 samples_in;
  guint64 samples_out;
  guint64 next_in_offset;
  
  guint64 num_gap_samples;
  gboolean need_tail_drain;

  gint outchannels; 		/* = number of templates */
  gint rate;
  gint width;
  // SpiirBank_s **bank;
  //gdouble *bank;
  //gint bank_len;
  gchar *bank_fname;
  GMutex *iir_bank_lock;
  GCond *iir_bank_available;
  SpiirState **spstate;
  gboolean spstate_initialised;

  gint stream_id;
  gint deviceID;
  cudaStream_t stream;

  gint gap_handle;
};

struct _CudaMultirateSPIIRClass {
  GstBaseTransformClass parent_class;
};

GType cuda_multirate_spiir_get_type(void);

G_END_DECLS

#endif /* __CUDA_MULTIRATE_SPIIR_H__ */

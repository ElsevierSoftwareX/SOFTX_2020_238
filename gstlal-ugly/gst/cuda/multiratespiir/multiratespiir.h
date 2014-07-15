/* GStreamer
 * Copyright (C) <1999> Erik Walthinsen <omega@cse.ogi.edu>
 * Copyright (C) <2007-2008> Sebastian Dröge <sebastian.droege@collabora.co.uk>
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

typedef struct _ResamplerState{
  float *d_sinc_table;
  float *d_mem; // fixed length to store input
  gint channels;
  gint mem_len;
  gint last_sample;
  gint filt_len;
  gint sinc_len;
  gint inrate;
  gint outrate;
  float amplifier;
} ResamplerState;

typedef struct _SpiirState {
//	a0;
//	b1;
//	d;
  int depth; // 0-6
  ResamplerState *downstate, *upstate;
  float *d_queue; // fixed length structure, to store the intermediate result from downsample, this is the input for upsample
//  float *d_out_spiir;
  int queue_len;
  int queue_eff_len;  // effective length
  int queue_down_start;
  int queue_up_start;
} SpiirState;

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
  gint num_depths;
  gint num_cover_samples; // number of samples needed to produce the first buffer
  gint num_exe_samples; // number of samples executed every time

  GstClockTime t0;
  guint64 offset0;
  guint64 samples_in;
  guint64 samples_out;
  guint64 next_in_offset;
  
  guint64 num_gap_samples;

  gint outchannels; // equals number of templates
  gint inchannels;
  gint rate;
  gint width;
  gboolean matrix_initialised;
  gboolean spstate_initialised;
  SpiirState **spstate;
};

struct _CudaMultirateSPIIRClass {
  GstBaseTransformClass parent_class;
};

GType cuda_multirate_spiir_get_type(void);

G_END_DECLS

#endif /* __CUDA_MULTIRATE_SPIIR_H__ */

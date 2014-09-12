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


#ifndef __AUDIO_RESAMPLECUDA_H__
#define __AUDIO_RESAMPLE_H__

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <gst/audio/audio.h>

#include "speex_resampler_wrapper.h"

G_BEGIN_DECLS

#define CUDA_AUDIO_RESAMPLE_TYPE \
  (cuda_audio_resample_get_type())
#define CUDA_AUDIO_RESAMPLE(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),CUDA_AUDIO_RESAMPLE_TYPE,GstAudioResampleCuda))
#define CUDA_AUDIO_RESAMPLE_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),CUDA_AUDIO_RESAMPLE_TYPE,GstAudioResampleCudaClass))
#define GST_IS_CUDA_AUDIO_RESAMPLE(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),CUDA_AUDIO_RESAMPLE_TYPE))
#define GST_IS_CUDA_AUDIO_RESAMPLE_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),CUDA_AUDIO_RESAMPLE_TYPE))

typedef struct _GstAudioResampleCuda GstAudioResampleCuda;
typedef struct _GstAudioResampleCudaClass GstAudioResampleCudaClass;

/**
 * GstAudioResample:
 *
 * Opaque data structure.
 */
struct _GstAudioResampleCuda {
  GstBaseTransform element;

  /* <private> */

  GstCaps *srccaps, *sinkcaps;

  gboolean need_discont;

  GstClockTime t0;
  guint64 in_offset0;
  guint64 out_offset0;
  guint64 samples_in;
  guint64 samples_out;
  
  guint64 num_gap_samples;
  guint64 num_nongap_samples;

  gint channels;
  gint inrate;
  gint outrate;
  gint quality;
  gint width;
  gboolean fp;

  guint8 *tmp_in;
  guint tmp_in_size;

  guint8 *tmp_out;
  guint tmp_out_size;

  SpeexResamplerState *state;
  const SpeexResampleFuncs *funcs;
};

struct _GstAudioResampleCudaClass {
  GstBaseTransformClass parent_class;
};

GType cuda_audio_resample_get_type(void);

G_END_DECLS

#endif /* __AUDIO_RESAMPLE_H__ */

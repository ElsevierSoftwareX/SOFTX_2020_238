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


#ifndef __CUDA_POSTCOH_H__
#define __CUDA_POSTCOH_H__

#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstcollectpads.h>
#include <gst/base/gstadapter.h>
#include <cuda_runtime.h>


G_BEGIN_DECLS

#define CUDA_TYPE_POSTCOH \
  (cuda_postcoh_get_type())
#define CUDA_POSTCOH(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),CUDA_TYPE_POSTCOH,CudaPostcoh))
#define CUDA_POSTCOH_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),CUDA_TYPE_POSTCOH,CudaPostcohClass))
#define GST_IS_CUDA_POSTCOH(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),CUDA_TYPE_POSTCOH))
#define GST_IS_CUDA_POSTCOH_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),CUDA_TYPE_POSTCOH))

typedef struct _CudaPostcoh CudaPostcoh;
typedef struct _CudaPostcohClass CudaPostcohClass;

#ifndef DEFINED_COMPLEX_F
#define DEFINED_COMPLEX_F 

typedef struct _Complex_F
{
	float re;
	float im;
} COMPLEX_F;

#else
#endif


typedef struct _GstPostcohCollectData GstPostcohCollectData;
typedef void (*CudaPostcohPeakfinder) (gpointer d_snglsnr, gint size);

struct _GstPostcohCollectData {
	GstCollectData data;
	gchar *ifo_name;
	GstAdapter *adapter;
  	double offset_per_nanosecond;
	gint channels;
	gboolean is_aligned;
	guint64 aligned_offset0;
	GstCollectDataDestroyNotify destroy_notify;
};

typedef struct _PeakList {
	/* data in the same type are allocated together */
	int *sample_index;
	int *tmplt_index;
	int *pix_index;
	float *maxsnglsnr;
	float *cohsnr;
	float *nullsnr;
	float *chi2;
	
	int *d_sample_index;
	int *d_tmplt_index;
	int *d_pix_index;
	float *d_maxsnglsnr;
	float *d_cohsnr;
	float *d_nullsnr;
	float *d_chi2;
} PeakList;

typedef struct _PostcohState {
  COMPLEX_F **d_snglsnr;
  gint nifo;
  gint8 *ifo_mapping;
  float **d_U_map;
  float **d_diff_map;
  int gps_step;
  int order;
  PeakList **peak_list;
  int *npeak;
  int head_len;
  int exe_len;
  int ntmplt;
  float *chi2_norm;
} PostcohState;

/**
 * CudaPostcoh:
 *
 * Opaque data structure.
 */
struct _CudaPostcoh {
  GstElement element;

  /* <private> */
  GstPad *srcpad;
  GstCollectPads *collect;

  gint rate;
  gint channels;
  gint width;
  gint bps;

  char *detrsp_fname;
  char *autocorrelation_fname;
  gint autocorrelation_len;
  gint exe_size;
  gint preserved_size;
  gboolean set_starttime;
  gboolean is_all_aligned;
  double offset_per_nanosecond;

  GstClockTime in_t0;
  GstClockTime out_t0;
  guint64 out_offset0;
  guint64 samples_in;
  guint64 samples_out;

  PostcohState *state;
};

struct _CudaPostcohClass {
  GstElementClass parent_class;
};

GType cuda_postcoh_get_type(void);

G_END_DECLS

#endif /* __CUDA_POSTCOH_H__ */

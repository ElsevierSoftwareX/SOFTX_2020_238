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

#ifndef MAX_ALLIFO_LEN
#define MAX_ALLIFO_LEN 14
#endif
typedef struct _CudaPostcoh CudaPostcoh;
typedef struct _CudaPostcohClass CudaPostcohClass;

extern char* IFO_MAP[];

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
  int peak_intlen;
  int peak_floatlen;

	/* data in the same type are allocated together */
	int *tmplt_idx;
	int *pix_idx;
	int *pix_idx_bg;
	int *peak_pos;
	int *npeak;
	float *maxsnglsnr;
	float *cohsnr;
	float *cohsnr_bg;
	float *nullsnr;
	float *nullsnr_bg;
	float *chisq;
	float *chisq_bg;
	float *cohsnr_skymap;
	float *nullsnr_skymap;

	int *d_tmplt_idx;
	int *d_pix_idx;
	int *d_pix_idx_bg;
	int *d_peak_pos;
	int *d_npeak;
	float *d_maxsnglsnr;
	float *d_cohsnr;
	float *d_cohsnr_bg;
	float *d_nullsnr;
	float *d_nullsnr_bg;
	float *d_chisq;
	float *d_chisq_bg;
	float *d_cohsnr_skymap;
	float *d_nullsnr_skymap;

	float *d_peak_tmplt;

} PeakList;

typedef struct _PostcohState {
  COMPLEX_F **d_snglsnr;
  COMPLEX_F **dd_snglsnr;
  COMPLEX_F **dd_autocorr_matrix;
  float **dd_autocorr_norm;
  int autochisq_len;
  int snglsnr_len;
  int snglsnr_start_load;
  int snglsnr_start_exe;
  gint nifo;
  gint8 *ifo_mapping;
  float **d_U_map;
  float **d_diff_map;
  int gps_step;
  int npix;
  PeakList **peak_list;
  int head_len;
  int exe_len;
  int ntmplt;
  float dt;
  float snglsnr_thresh;
  gint hist_trials;
  gint trial_sample_inv;
  char cur_ifos[MAX_ALLIFO_LEN];
  gint cur_nifo;
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
  char *autocorr_fname;
  gint exe_len;
  gint preserved_len;
  float max_dt;
  gboolean set_starttime;
  gboolean is_all_aligned;
  double offset_per_nanosecond;

  GstClockTime in_t0;
  GstClockTime out_t0;
  GstClockTime next_t;
  guint64 out_offset0;
  guint64 samples_in;
  guint64 samples_out;

  PostcohState *state;
  float snglsnr_thresh;
  GMutex *prop_lock;
  GCond *prop_avail;
  gint hist_trials;
  float trial_interval;
  gint trial_interval_in_samples;
  gint output_skymap;

  /* sink event handling */
  GstPadEventFunction collect_event;
};

struct _CudaPostcohClass {
  GstElementClass parent_class;
};

GType cuda_postcoh_get_type(void);

G_END_DECLS

#endif /* __CUDA_POSTCOH_H__ */

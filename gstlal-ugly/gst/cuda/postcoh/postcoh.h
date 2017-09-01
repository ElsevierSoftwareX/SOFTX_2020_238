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

// FIXME: hack for cuda-6.5 and lal header to work
#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS
#endif

#include <lal/LIGOMetadataTables.h>

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

#ifndef MIN_OUTPUT_SKYMAP_SNR
#define MIN_OUTPUT_SKYMAP_SNR 11
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
  	guint64 next_offset;
	GstCollectDataDestroyNotify destroy_notify;
};

// FIXME: consider more flxible structure for PeakList
typedef struct _PeakList {
	int peak_intlen;
	int peak_floatlen;

	/* data in the same type are allocated together */
	int *npeak;
	int *peak_pos;
	int *len_idx;
	int *tmplt_idx;
	int *pix_idx;
	int *pix_idx_bg; // background Ntoff needs this, do not remove
	int *ntoff_L;
	int *ntoff_H;
	int *ntoff_V;

	float *snglsnr_L;
	float *snglsnr_H;
	float *snglsnr_V;
	float *coaphase_L;
	float *coaphase_H;
	float *coaphase_V;
	float *chisq_L;
	float *chisq_H;
	float *chisq_V;

	float *snglsnr_bg_L;
	float *snglsnr_bg_H;
	float *snglsnr_bg_V;
	float *coaphase_bg_L;
	float *coaphase_bg_H;
	float *coaphase_bg_V;
	float *chisq_bg_L;
	float *chisq_bg_H;
	float *chisq_bg_V;


	float *cohsnr;
	float *nullsnr;
	float *cmbchisq;

	float *cohsnr_bg;
	float *nullsnr_bg;
	float *cmbchisq_bg;
	
	float *cohsnr_skymap;
	float *nullsnr_skymap;
	
	/* structure on GPU device */
	int *d_npeak;
	int *d_peak_pos;
	int *d_len_idx;
	int *d_tmplt_idx;
	int *d_pix_idx;
	int *d_pix_idx_bg; // background Ntoff needs this, do not remove
	int *d_ntoff_L;
	int *d_ntoff_H;
	int *d_ntoff_V;

	float *d_snglsnr_L;
	float *d_snglsnr_H;
	float *d_snglsnr_V;
	float *d_coaphase_L;
	float *d_coaphase_H;
	float *d_coaphase_V;
	float *d_chisq_L;
	float *d_chisq_H;
	float *d_chisq_V;

	float *d_snglsnr_bg_L;
	float *d_snglsnr_bg_H;
	float *d_snglsnr_bg_V;
	float *d_coaphase_bg_L;
	float *d_coaphase_bg_H;
	float *d_coaphase_bg_V;
	float *d_chisq_bg_L;
	float *d_chisq_bg_H;
	float *d_chisq_bg_V;

	float *d_cohsnr;
	float *d_nullsnr;
	float *d_cmbchisq;

	float *d_cohsnr_bg;
	float *d_nullsnr_bg;
	float *d_cmbchisq_bg;
	
	float *d_cohsnr_skymap;
	float *d_nullsnr_skymap;

	float *d_peak_tmplt;
	float *d_maxsnglsnr; // for cuda peakfinder, not used now

	float *d_snglsnr_buffer;  // we need to copy data from CPU memory to this buffer; then do transpose for new postcoh kernel optimized by Xiaoyang Guo
	int len_snglsnr_buffer;
} PeakList;

typedef struct _PostcohState {
  /* parent pointer in host device, each children pointer is in host device, 
   * pointing to a detector snglsnr array in GPU device */
  COMPLEX_F **d_snglsnr;
  /* parent pointer in host device, each children pointer is in GPU device,
   * pointing to a detector snglsnr array in GPU device*/
  COMPLEX_F **dd_snglsnr;
  /* parent pointer in host device, each children pointer is in GPU device,
   * pointing to a detector autocorrelation array in GPU device*/
  COMPLEX_F **dd_autocorr_matrix;
  /* parent pointer in host device, each children pointer is in GPU device,
   * pointing to a detector autocorrealtion norm value in GPU device*/
  float **dd_autocorr_norm;
  int autochisq_len;
  int snglsnr_len;
  int snglsnr_start_load;
  int snglsnr_start_exe;
  gint nifo;
  gint8 *ifo_mapping;
  /* sigmasq read from bank to compute effective distance */
  double **sigmasq;
  /* parent pointer in host device, each children pointer is in host device,
   * pointing to the coherent U map of a certain time in GPU device*/
  float **d_U_map;
  /* parent pointer in host device, each children pointer is in host device,
   * pointing to the coherent time arrival diff map of a certain time in GPU device*/
  float **d_diff_map;
  int gps_step;
  /* be careful that long has different length in different machines */
  long gps_start;
  unsigned long nside;
  int npix;
  PeakList **peak_list;
  int head_len;
  int exe_len;
  int max_npeak;
  int ntmplt;
  float dt;
  float snglsnr_thresh;
  gint hist_trials;
  gint trial_sample_inv;
  char cur_ifos[MAX_ALLIFO_LEN];
  gint cur_nifo;
  gint is_member_init;
  float snglsnr_max;
  float *tmp_maxsnr;
  int *tmp_tmpltidx;
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
  char *spiir_bank_fname;
  gint exe_len;
  gint preserved_len;
  float max_dt;
  gboolean set_starttime;
  gboolean is_all_aligned;
  double offset_per_nanosecond;

  GstClockTime t0;
  GstClockTime next_exe_t;
  guint64 offset0;
  guint64 samples_in;
  guint64 samples_out;

  PostcohState *state;
  float snglsnr_thresh;
  float cohsnr_thresh;
  GMutex *prop_lock;
  GCond *prop_avail;
  gint hist_trials;
  float trial_interval;
  gint trial_interval_in_samples;
  gint output_skymap;

  char *sngl_tmplt_fname;
  SnglInspiralTable *sngl_table;

  /* sink event handling */
  GstPadEventFunction collect_event;

  gint stream_id;
  gint device_id;
  /* book-keeping */
  long process_id;
  long cur_event_id;
  cudaStream_t stream;
};

struct _CudaPostcohClass {
  GstElementClass parent_class;
};

GType cuda_postcoh_get_type(void);

G_END_DECLS

#endif /* __CUDA_POSTCOH_H__ */

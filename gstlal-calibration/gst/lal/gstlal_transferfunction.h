/*
 * Copyright (C) 2018  Aaron Viets <aaron.viets@ligo.org>
 *
 *  This program is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU General Public License as published by the
 *  Free Software Foundation; either version 2 of the License, or (at your
 *  option) any later version.
 *
 *  This program is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */


#ifndef __GSTLAL_TRANSFERFUNCTION_H__
#define __GSTLAL_TRANSFERFUNCTION_H__


#include <complex.h>

#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasesink.h>

#include <fftw3.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_permutation.h>


G_BEGIN_DECLS


/*
 * gstlal_transferfunction_window_type enum
 */


enum gstlal_transferfunction_window_type {
        GSTLAL_TRANSFERFUNCTION_DPSS = 0,
        GSTLAL_TRANSFERFUNCTION_KAISER,
        GSTLAL_TRANSFERFUNCTION_DOLPH_CHEBYSHEV
};


#define GSTLAL_TRANSFERFUNCTION_WINDOW_TYPE  \
        (gstlal_transferfunction_window_get_type())


GType gstlal_transferfunction_window_get_type(void);


/*
 * lal_transferfunction element
 */


#define GSTLAL_TRANSFERFUNCTION_TYPE \
	(gstlal_transferfunction_get_type())
#define GSTLAL_TRANSFERFUNCTION(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_TRANSFERFUNCTION_TYPE, GSTLALTransferFunction))
#define GSTLAL_TRANSFERFUNCTION_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_TRANSFERFUNCTION_TYPE, GSTLALTransferFunctionClass))
#define GST_IS_GSTLAL_TRANSFERFUNCTION(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_TRANSFERFUNCTION_TYPE))
#define GST_IS_GSTLAL_TRANSFERFUNCTION_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_TRANSFERFUNCTION_TYPE))


typedef struct _GSTLALTransferFunction GSTLALTransferFunction;
typedef struct _GSTLALTransferFunctionClass GSTLALTransferFunctionClass;


/**
 * GSTLALTransferFunction:
 */


struct _GSTLALTransferFunction {
	GstBaseSink basesink;

	/* stream info */
	gint rate;
	gint unit_size;
	gint channels;
	gint64 sample_count;
	gint64 gap_samples;
	int num_tfs_since_gap;
	enum gstlal_transferfunction_data_type {
		GSTLAL_TRANSFERFUNCTION_F32 = 0,
		GSTLAL_TRANSFERFUNCTION_F64,
	} data_type;

	/* timestamp bookkeeping */
	GstClockTime t0;
	guint64 offset0;
	guint64 next_in_offset;

	/* Internal state */
	gboolean computed_full_tfs;
	double t_start_tf;

	/* transfer function work space */
	union {
		struct {
			float *fft_window;
			float *sinc_table;
			gint64 sinc_length;
			gint64 sinc_taps_per_df;
			float *fd_fir_window;
			double *fir_window;
			float *leftover_data;
			gint64 num_leftover;
			complex float *ffts;
			gint64 num_ffts_in_avg;
			gint64 num_ffts_dropped;
			complex float *autocorrelation_matrix;
			float *autocorrelation_median_real;
			gint64 *index_median_real;
			float *autocorrelation_median_imag;
			gint64 *index_median_imag;

			/* gsl stuff */
			gsl_vector_complex *transfer_functions_at_f;
			gsl_vector_complex *transfer_functions_solved_at_f;
			gsl_matrix_complex *autocorrelation_matrix_at_f;
			gsl_permutation *permutation;

			/* fftwf stuff */
			complex float *fft;
			fftwf_plan plan;
			complex float *fir_filter;
			fftwf_plan fir_plan;
		} wspf;  /* workspace single-precision float */
		struct {
			double *fft_window;
			double *sinc_table;
			gint64 sinc_length;
			gint64 sinc_taps_per_df;
			double *fd_fir_window;
			double *fir_window;
			double *leftover_data;
			gint64 num_leftover;
			complex double *ffts;
			gint64 num_ffts_in_avg;
			gint64 num_ffts_dropped;
			complex double *autocorrelation_matrix;
			double *autocorrelation_median_real;
			gint64 *index_median_real;
			double *autocorrelation_median_imag;
			gint64 *index_median_imag;

			/* gsl stuff */
			gsl_vector_complex *transfer_functions_at_f;
			gsl_vector_complex *transfer_functions_solved_at_f;
			gsl_matrix_complex *autocorrelation_matrix_at_f;
			gsl_permutation *permutation;

			/* fftw stuff */
			complex double *fft;
			fftw_plan plan;
			complex double *fir_filter;
			fftw_plan fir_plan;
		} wdpf;  /* workspace double-precision float */
	} workspace;

	/* properties */
	gint64 fft_length;
	gint64 fft_overlap;
	gint64 num_ffts;
	gint64 min_ffts;
	gboolean use_median;
	gint64 update_samples;
	gboolean update_after_gap;
	gint64 use_first_after_gap;
	gint64 update_delay_samples;
	gboolean parallel_mode;
	gboolean write_to_screen;
	char *filename;
	double make_fir_filters;
	gint64 fir_length;
	double frequency_resolution;
	int high_pass;
	int low_pass;
	double *notch_frequencies;
	gint64 *notch_indices;
	int num_notches;
	gint64 fir_timeshift;
	complex double *post_gap_transfer_functions;
	double *post_gap_fir_filters;
	complex double *transfer_functions;
	double *fir_filters;
	guint64 fir_endtime;
	enum gstlal_transferfunction_window_type window;
};


/**
 * GSTLALTransferFunctionClass:
 * @parent_class:  the parent class
 */


struct _GSTLALTransferFunctionClass {
	GstBaseSinkClass parent_class;
};


GType gstlal_transferfunction_get_type(void);


G_END_DECLS


#endif	/* __GSTLAL_TRANSFERFUNCTION_H__ */

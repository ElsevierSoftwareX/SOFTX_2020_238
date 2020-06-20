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


#ifndef __GSTLAL_ADAPTIVEFIRFILT_H__
#define __GSTLAL_ADAPTIVEFIRFILT_H__


#include <complex.h>

#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasesink.h>

#include <fftw3.h>


G_BEGIN_DECLS


/*
 * gstlal_adaptivefirfilt_window_type enum
 */


enum gstlal_adaptivefirfilt_window_type {
	GSTLAL_ADAPTIVEFIRFILT_DPSS = 0,
	GSTLAL_ADAPTIVEFIRFILT_KAISER,
	GSTLAL_ADAPTIVEFIRFILT_DOLPH_CHEBYSHEV
};


#define GSTLAL_ADAPTIVEFIRFILT_WINDOW_TYPE  \
	(gstlal_adaptivefirfilt_window_get_type())


GType gstlal_adaptivefirfilt_window_get_type(void);


/*
 * lal_adaptivefirfilt element
 */


#define GSTLAL_ADAPTIVEFIRFILT_TYPE \
	(gstlal_adaptivefirfilt_get_type())
#define GSTLAL_ADAPTIVEFIRFILT(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_ADAPTIVEFIRFILT_TYPE, GSTLALAdaptiveFIRFilt))
#define GSTLAL_ADAPTIVEFIRFILT_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_ADAPTIVEFIRFILT_TYPE, GSTLALAdaptiveFIRFiltClass))
#define GST_IS_GSTLAL_ADAPTIVEFIRFILT(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_ADAPTIVEFIRFILT_TYPE))
#define GST_IS_GSTLAL_ADAPTIVEFIRFILT_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_ADAPTIVEFIRFILT_TYPE))


typedef struct _GSTLALAdaptiveFIRFilt GSTLALAdaptiveFIRFilt;
typedef struct _GSTLALAdaptiveFIRFiltClass GSTLALAdaptiveFIRFiltClass;


/**
 * GSTLALAdaptiveFIRFilt:
 */


struct _GSTLALAdaptiveFIRFilt {
	GstBaseSink basesink;

	/* stream info */
	gint rate;
	gint unit_size;
	gint channels;
	enum gstlal_adaptivefirfilt_data_type {
		GSTLAL_ADAPTIVEFIRFILT_Z64 = 0,
		GSTLAL_ADAPTIVEFIRFILT_Z128,
	} data_type;

	/* timestamp bookkeeping */
	GstClockTime t0;
	guint64 offset0;
	guint64 next_in_offset;

	/* FIR filter parameters */
	complex double *input_average;
	gint64 num_in_avg;
	gboolean filter_has_gain;
	double complex *variable_filter;
	fftw_plan variable_filter_plan;

	/* properties */
	gint64 update_samples;
	gint64 average_samples;
	int num_zeros;
	int num_poles;
	complex double *static_zeros;
	int num_static_zeros;
	complex double *static_poles;
	int num_static_poles;
	double phase_measurement_frequency;
	double *static_filter;
	gint64 static_filter_length;
	gint64 variable_filter_length;
	gboolean minimize_filter_length;
	double *adaptive_filter;
	gint64 adaptive_filter_length;
	double *window;
	double frequency_resolution;
	gint filter_sample_rate;
	gint64 filter_timeshift;
	guint64 filter_endtime;
	gboolean write_to_screen;
	char *filename;
	enum gstlal_adaptivefirfilt_window_type window_type;
};


/**
 * GSTLALAdaptiveFIRFiltClass:
 * @parent_class:  the parent class
 */


struct _GSTLALAdaptiveFIRFiltClass {
	GstBaseSinkClass parent_class;
};


GType gstlal_adaptivefirfilt_get_type(void);


G_END_DECLS


#endif	/* __GSTLAL_ADAPTIVEFIRFILT_H__ */

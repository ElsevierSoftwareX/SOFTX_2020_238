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


#ifndef __GSTLAL_LPFILTER_H__
#define __GSTLAL_LPFILTER_H__


#include <complex.h>

#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasesink.h>

#include <fftw3.h>


G_BEGIN_DECLS


#define GSTLAL_LPFILTER_TYPE \
	(gstlal_lpfilter_get_type())
#define GSTLAL_LPFILTER(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_LPFILTER_TYPE, GSTLALLPFilter))
#define GSTLAL_LPFILTER_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_LPFILTER_TYPE, GSTLALLPFilterClass))
#define GST_IS_GSTLAL_LPFILTER(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_LPFILTER_TYPE))
#define GST_IS_GSTLAL_LPFILTER_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_LPFILTER_TYPE))


typedef struct _GSTLALLPFilter GSTLALLPFilter;
typedef struct _GSTLALLPFilterClass GSTLALLPFilterClass;


/**
 * GSTLALLPFilter:
 */


struct _GSTLALLPFilter {
	GstBaseSink basesink;

	/* stream info */
	gint rate;
	gint unit_size;
	enum gstlal_lpfilter_data_type {
		GSTLAL_LPFILTER_Z64 = 0,
		GSTLAL_LPFILTER_Z128,
	} data_type;

	/* timestamp bookkeeping */
	GstClockTime t0;
	guint64 offset0;
	guint64 next_in_offset;

	/* FIR filter parameters */
	complex double input_average;
	gint64 num_in_avg;

	/* properties */
	double measurement_frequency;
	gint64 update_samples;
	gint64 average_samples;
	gboolean write_to_screen;
	char *filename;
	gint64 fir_length;
	gint fir_sample_rate;
	complex double *fir_filter;
	fftw_plan fir_plan;
};


/**
 * GSTLALLPFilterClass:
 * @parent_class:  the parent class
 */


struct _GSTLALLPFilterClass {
	GstBaseSinkClass parent_class;
};


GType gstlal_lpfilter_get_type(void);


G_END_DECLS


#endif	/* __GSTLAL_LPFILTER_H__ */

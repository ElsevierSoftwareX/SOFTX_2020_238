/*
 * A multi-channel scope.
 *
 * Copyright (C) 2008  Kipp Cannon, Chad Hanna
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */


#ifndef __GSTLAL_MULTISCOPE_H__
#define __GSTLAL_MULTISCOPE_H__


#include <gst/gst.h>
#include <gst/base/gstadapter.h>


#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>


G_BEGIN_DECLS


#define GSTLAL_MULTISCOPE_TYPE \
	(gstlal_multiscope_get_type())
#define GSTLAL_MULTISCOPE(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_MULTISCOPE_TYPE, GSTLALMultiScope))
#define GSTLAL_MULTISCOPE_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_MULTISCOPE_TYPE, GSTLALMultiScopeClass))
#define GST_IS_GSTLAL_MULTISCOPE(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_MULTISCOPE_TYPE))
#define GST_IS_GST_MULTISCOPE_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_MULTISCOPE_TYPE))


typedef struct {
	GstElementClass parent_class;
} GSTLALMultiScopeClass;


typedef struct {
	GstElement element;

	GstPad *srcpad;

	GstAdapter *adapter;

	int channels;
	int sample_rate;
	double trace_duration;
	double frame_interval;
	double vertical_scale_sigmas;

	unsigned long next_sample;
	GstClockTime adapter_head_timestamp;

	double mean;
	double variance;
	double average_interval;
	gboolean do_timestamp;
} GSTLALMultiScope;


GType gstlal_multiscope_get_type(void);


G_END_DECLS


#endif	/* __GSTLAL_MULTISCOPE_H__ */

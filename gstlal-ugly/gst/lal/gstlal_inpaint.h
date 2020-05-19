/*
 * Inpaints.
 *
 * Copyright (C) 2020 Cody Messick
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 */


#ifndef __GSTLAL_INPAINT_H__
#define __GSTLAL_INPAINT_H__

/*
 * stuff from gstreamer
 */

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>

/*
 * our own stuff
 */

#include <gstlal/gstaudioadapter.h>

/*
 * stuff from LAL
 */

#include <lal/TimeSeries.h>
#include <lal/FrequencySeries.h>

G_BEGIN_DECLS


#define GSTLAL_INPAINT_TYPE \
	(gstlal_inpaint_get_type())
#define GSTLAL_INPAINT(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_INPAINT_TYPE, GSTLALInpaint))
#define GSTLAL_INPAINT_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_INPAINT_TYPE, GSTLALInpaintClass))
#define GST_IS_GSTLAL_INPAINT(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_INPAINT_TYPE))
#define GST_IS_GSTLAL_INPAINT_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_INPAINT_TYPE))



/**
 * GSTLALInpaint:
 * @parent:  the parent structure
 */


typedef struct {
	GstBaseTransform parent;

	char *instrument;
	char *channel_name;
	char *units;
	guint rate;
	GstAudioAdapter *adapter;
	double *output_hoft;

	/*
	 * Buffer time tracking
	 */
	guint64 initial_offset;
	guint64 outbuf_length;
	GstClockTime t0;

	/*
	 * PSD stuff
	 */
	double fft_length_seconds;
	guint fft_length_samples;
	REAL8FrequencySeries *psd;
	REAL8TimeSeries *inv_cov_series;

	/*
	 * Matrix workspace
	 */
	double *inv_cov_mat_workspace;
	double *M_trans_mat_workspace;
	double *inv_M_trans_mat_workspace;
	double *F_trans_mat_workspace;

} GSTLALInpaint;


/**
 * GSTLALInpaintClass:
 * @parent_class:  the parent class
 */


typedef struct {
	GstBaseTransformClass parent_class;
} GSTLALInpaintClass;


GType gstlal_inpaint_get_type(void);


G_END_DECLS


#endif	/* __GSTLAL_INPAINT_H__ */

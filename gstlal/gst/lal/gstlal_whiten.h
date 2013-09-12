/*
 * PSD Estimation and whitener
 *
 * Copyright (C) 2008  Chad Hanna, Kipp Cannon
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


#ifndef __GSTLAL_WHITEN_H__
#define __GSTLAL_WHITEN_H__


#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <gstlal/gstaudioadapter.h>


#include <lal/LALDatatypes.h>
#include <lal/TimeFreqFFT.h>
#include <lal/Units.h>


G_BEGIN_DECLS


/*
 * gstlal_psdmode_t enum
 */


enum gstlal_psdmode_t {
	GSTLAL_PSDMODE_RUNNING_AVERAGE,
	GSTLAL_PSDMODE_FIXED
};


#define GSTLAL_PSDMODE_TYPE  \
	(gstlal_psdmode_get_type())


GType gstlal_psdmode_get_type(void);


/*
 * lal_whiten element
 */


#define GSTLAL_WHITEN_TYPE \
	(gstlal_whiten_get_type())
#define GSTLAL_WHITEN(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_WHITEN_TYPE, GSTLALWhiten))
#define GSTLAL_WHITEN_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_WHITEN_TYPE, GSTLALWhitenClass))
#define GST_IS_GSTLAL_WHITEN(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_WHITEN_TYPE))
#define GST_IS_GSTLAL_WHITEN_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_WHITEN_TYPE))


typedef struct _GSTLALWhiten GSTLALWhiten;
typedef struct _GSTLALWhitenClass GSTLALWhitenClass;


/**
 * GSTLALWhiten:
 */


struct _GSTLALWhiten {
	GstBaseTransform element;

	/*
	 * input stream
	 */

	LALUnit sample_units;
	gint sample_rate;

	GstAudioAdapter *input_queue;

	/*
	 * psd output stream
	 */

	GstPad *mean_psd_pad;

	/*
	 * time stamp book-keeping
	 */

	gboolean need_discont;
	GstClockTime t0;
	guint64 offset0;
	guint64 next_offset_in;
	guint64 next_offset_out;

	/*
	 * PSD estimation parameters
	 */

	double zero_pad_seconds;
	double fft_length_seconds;
	enum gstlal_psdmode_t psdmode;

	/*
	 * work space
	 */

	REAL8Window *hann_window;
	REAL8Window *tukey_window;
	REAL8FFTPlan *fwdplan;
	REAL8FFTPlan *revplan;
	REAL8TimeSeries *tdworkspace;
	COMPLEX16FrequencySeries *fdworkspace;

	/*
	 * output stream
	 */

	REAL8Sequence *output_history;
	guint64 output_history_offset;
	guint nonzero_output_history_length;
	gboolean expand_gaps;

	/*
	 * PSD state
	 */

	LALPSDRegressor *psd_regressor;
	REAL8FrequencySeries *psd;
};


/**
 * GSTLALWhitenClass:
 */


struct _GSTLALWhitenClass {
	GstBaseTransformClass parent_class;
};


GType gstlal_whiten_get_type(void);


G_END_DECLS


#endif	/* __GSTLAL_WHITEN_H__ */

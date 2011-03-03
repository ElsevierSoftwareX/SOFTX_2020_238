/*
 * A time-slice-based \Chi^{2} element for the inspiral pipeline.
 *
 * Copyright (C) 2011  Kipp Cannon, Chad Hanna, Drew Keppel
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation; either version 2 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */


#ifndef __GSTLAL_TIMESLICECHISQUARE_H__
#define __GSTLAL_TIMESLICECHISQUARE_H__


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstcollectpads.h>
#include <gstlalcollectpads.h>


#include <gsl/gsl_matrix.h>


G_BEGIN_DECLS


#define GSTLAL_TIMESLICECHISQUARE_TYPE \
	(gstlal_timeslicechisquare_get_type())
#define GSTLAL_TIMESLICECHISQUARE(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_TIMESLICECHISQUARE_TYPE, GSTLALTimeSliceChiSquare))
#define GSTLAL_TIMESLICECHISQUARE_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_TIMESLICECHISQUARE_TYPE, GSTLALTimeSliceChiSquareClass))
#define GST_IS_GSTLAL_TIMESLICECHISQUARE(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_TIMESLICECHISQUARE_TYPE))
#define GST_IS_GSTLAL_TIMESLICECHISQUARE_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_TIMESLICECHISQUARE_TYPE))


typedef struct {
	GstElementClass parent_class;
} GSTLALTimeSliceChiSquareClass;


typedef struct {
	GstElement element;

	GstCollectPads *collect;

	GstPad *timeslicesnrpad;
	GstLALCollectData *timeslicesnrcollectdata;
	GstPad *snrpad;
	GstLALCollectData *snrcollectdata;
	GstPad *srcpad;

	/* stream format */
	gint rate;

	GMutex *coefficients_lock;
	GCond *coefficients_available;
	gsl_vector *chifacs;	

	/* counters to keep track of timestamps. */
	gboolean segment_pending;
	GstSegment segment;
	guint64 offset;
} GSTLALTimeSliceChiSquare;


GType gstlal_timeslicechisquare_get_type(void);


G_END_DECLS


#endif	/* __GSTLAL_TIMESLICECHISQUARE_H__ */

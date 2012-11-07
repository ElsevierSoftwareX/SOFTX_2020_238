/* GStreamer
 * Copyright (C) 1999,2000 Erik Walthinsen <omega@cse.ogi.edu>
 *                    2000 Wim Taymans <wtay@chello.be>
 *                    2008 Kipp Cannon <kipp.cannon@ligo.org>
 *                    2011 Drew Keppel <drew.keppel@ligo.org>
 *
 * gstlal_timeslicechisq.h: Header for GstlalTimeSliceChisq element
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


#ifndef __GSTLAL_TIMESLICECHISQUARE_H__
#define __GSTLAL_TIMESLICECHISQUARE_H__


#include <gst/gst.h>
#include <gst/base/gstcollectpads.h>
#include <gsl/gsl_matrix.h>


G_BEGIN_DECLS
#define GSTLAL_TIMESLICECHISQUARE_TYPE (gstlal_timeslicechisquare_get_type())
#define GSTLAL_TIMESLICECHISQUARE(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_TIMESLICECHISQUARE_TYPE, GSTLALTimeSliceChiSquare))
#define GST_IS_GSTLAL_TIMESLICECHISQUARE(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_TIMESLICECHISQUARE_TYPE))
#define GSTLAL_TIMESLICECHISQUARE_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_TIMESLICECHISQUARE_TYPE, GSTLALTimeSliceChiSquareClass))
#define GST_IS_GSTLAL_TIMESLICECHISQUARE_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_TIMESLICECHISQUARE_TYPE))
#define GSTLAL_TIMESLICECHISQUARE_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS((obj), GSTLAL_TIMESLICECHISQUARE_TYPE, GSTLALTimeSliceChiSquareClass))


typedef void (*GSTLALTimeSliceChiSquareFunction) (gpointer out, const gpointer in, size_t size);


/**
 * GSTLALTimeSliceChiSquare:
 *
 * The gstlal_timeslicechisq object structure.
 */


typedef struct _GSTLALTimeSliceChiSquare {
	GstElement element;

	GstPad *srcpad;
	GstCollectPads *collect;
	/* pad counter, used for creating unique request pads */
	gint padcount;

	/* stream format */
	gint rate;
	guint float_unit_size; /* = 8 * channels */
	guint complex_unit_size; /* = 16 * channels */
	gint channels;

	/* chifacs coefficients */
	GMutex *coefficients_lock;
	GCond *coefficients_available;
	gsl_matrix *chifacs;

	/* counters to keep track of timestamps */
	GstClockTime    timestamp;
	guint64         offset;
	gboolean        synchronous;

	/* sink event handling */
	GstPadEventFunction collect_event;
	GstSegment segment;
	gboolean segment_pending;

	/* src event handling */
	gboolean flush_stop_pending;

	/* Pending inline events */
	GList *pending_events;
} GSTLALTimeSliceChiSquare;


/**
 * GSTLALTimeSliceChiSquareClass:
 *
 * The gstlal_timeslicechisq class structure.
 */


typedef struct _GSTLALTimeSliceChiSquareClass {
	GstElementClass parent_class;
} GSTLALTimeSliceChiSquareClass;


GType gstlal_timeslicechisquare_get_type(void);


G_END_DECLS


#endif	/* __GSTLAL_TIMESLICECHISQUARE_H__ */

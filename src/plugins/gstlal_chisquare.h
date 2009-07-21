/*
 * A \Chi^{2} element for the inspiral pipeline.
 *
 * Copyright (C) 2008  Kipp Cannon, Chad Hanna
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


#ifndef __GSTLAL_CHISQUARE_H__
#define __GSTLAL_CHISQUARE_H__


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstcollectpads.h>
#include <gstlalcollectpads.h>


#include <gsl/gsl_matrix.h>


G_BEGIN_DECLS


#define GSTLAL_CHISQUARE_TYPE \
	(gstlal_chisquare_get_type())
#define GSTLAL_CHISQUARE(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_CHISQUARE_TYPE, GSTLALChiSquare))
#define GSTLAL_CHISQUARE_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_CHISQUARE_TYPE, GSTLALChiSquareClass))
#define GST_IS_GSTLAL_CHISQUARE(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_CHISQUARE_TYPE))
#define GST_IS_GSTLAL_CHISQUARE_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_CHISQUARE_TYPE))


typedef struct {
	GstElementClass parent_class;
} GSTLALChiSquareClass;


typedef struct {
	GstElement element;

	GstCollectPads *collect;

	GstPad *matrixpad;
	GstPad *chifacspad;
	GstPad *orthosnrpad;
	GstLALCollectData *orthosnrcollectdata;
	GstPad *snrpad;
	GstLALCollectData *snrcollectdata;
	GstPad *srcpad;

	/* max dof to use */
	gint max_dof;

	/* stream format */
	gint rate;

	GCond *coefficients_available;

	GstBuffer *mixmatrix_buf;
	gsl_matrix_view mixmatrix;
	GstBuffer *chifacs_buf;
	gsl_vector_view chifacs;

	/* counters to keep track of timestamps. */
	gboolean segment_pending;
	GstSegment segment;
	guint64 offset;
} GSTLALChiSquare;


GType gstlal_chisquare_get_type(void);


G_END_DECLS


#endif	/* __GSTLAL_CHISQUARE_H__ */

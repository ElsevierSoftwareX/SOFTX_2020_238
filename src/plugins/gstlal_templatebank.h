/*
 * A template bank.
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


#ifndef __GSTLAL_TEMPLATEBANK_H__
#define __GSTLAL_TEMPLATEBANK_H__


#include <gst/gst.h>
#include <gst/base/gstadapter.h>


#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>


G_BEGIN_DECLS


#define GSTLAL_TEMPLATEBANK_TYPE \
	(gstlal_templatebank_get_type())
#define GSTLAL_TEMPLATEBANK(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_TEMPLATEBANK_TYPE, GSTLALTemplateBank))
#define GSTLAL_TEMPLATEBANK_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_TEMPLATEBANK_TYPE, GSTLALTemplateBankClass))
#define GST_IS_GSTLAL_TEMPLATEBANK(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_TEMPLATEBANK_TYPE))
#define GST_IS_GST_TEMPLATEBANK_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_TEMPLATEBANK_TYPE))


typedef struct {
	GstElementClass parent_class;
} GSTLALTemplateBankClass;


typedef struct {
	GstElement element;

	GList *srcpads;

	GstAdapter *adapter;

	unsigned int t_start;
	unsigned int t_end;

	long next_sample;
	GstClockTime next_sample_time;

	gsl_matrix *U;
	gsl_vector *S;
	gsl_matrix *V;
	gsl_vector *chifacs;
} GSTLALTemplateBank;


GType gstlal_templatebank_get_type(void);


G_END_DECLS


#endif	/* __GSTLAL_TEMPLATEBANK_H__ */

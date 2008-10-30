/*
 * An SNR time series sink that produces LIGOLwXML files of triggers.
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


#ifndef __GSTLAL_TRIGGERGEN_H__
#define __GSTLAL_TRIGGERGEN_H__


#include <gst/gst.h>
#include <gst/base/gstbasesink.h>


G_BEGIN_DECLS


#define GSTLAL_TRIGGERGEN_TYPE \
	(gstlal_triggergen_get_type())
#define GSTLAL_TRIGGERGEN(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_TRIGGERGEN_TYPE, GSTLALTriggerGen))
#define GSTLAL_TRIGGERGEN_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_TRIGGERGEN_TYPE, GSTLALTriggerGenClass))
#define GST_IS_GSTLAL_TRIGGERGEN(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_TRIGGERGEN_TYPE))
#define GST_IS_GSTLAL_TRIGGERGEN_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_TRIGGERGEN_TYPE))


typedef struct {
	GstBaseSinkClass parent_class;
} GSTLALTriggerGenClass;


typedef struct {
	GstBaseSink element;
	double snr_thresh;
	double *mass1;
	double *mass2;
	/* only one additional parameter is necessary to derive tau0,tau3 from
	 * mass, the lower cutoff frequency.  But since that isn't available 
	 * directly I'll just store tau0 and tau3;
	 */
	double *tau0;
	double *tau3;
	double *sigmasq;
	double *Gamma;
	char *bank_filename;
} GSTLALTriggerGen;


GType gstlal_triggergen_get_type(void);


G_END_DECLS


#endif	/* __GSTLAL_TRIGGERGEN_H__ */

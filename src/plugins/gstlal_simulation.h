/*
 * An interface to LALSimulation.  
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

#ifndef __GSTLAL_SIMULATION_H__
#define __GSTLAL_SIMULATION_H__


#include <gst/gst.h>
#include <gst/base/gstadapter.h>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

G_BEGIN_DECLS


#define GSTLAL_SIMULATION_TYPE \
        (gstlal_simulation_get_type())
	#define GSTLAL_SIMULATION(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_SIMULATION_TYPE, GSTLALSimulation))
#define GSTLAL_TEMPLATEBANK_CLASS(klass) \
        (G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_SIMULATION_TYPE, GSTLALSimulationClass))
#define GST_IS_GSTLAL_SIMULATION(obj) \
        (G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_SIMULATION_TYPE))
#define GST_IS_GST_SIMULATION_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_SIMULATION_TYPE))

typedef struct {
        GstElementClass parent_class;
	} GSTLALSimulationClass;


typedef struct {
        GstElement element;

        GList *srcpads;

        GstAdapter *adapter;
	
	double right_ascension;
	double declination;
	double psi;
	double phic;
	double m1;
	double m2;
	double fmin;
	double r;
	double i;
	int amplitudeO;
	int phaseO;
	LALDetector *detector;
	LIGOTimeGPS *tc;
	REAL8TimeSeries *h; /* This is the computed detector strain */

} GSTLALSimulation;

GType gstlal_simulation_get_type(void);

G_END_DECLS

#endif  /* __GSTLAL_SIMULATION_H__ */



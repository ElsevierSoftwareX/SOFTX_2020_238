/*
 * An interface to LALSimulation.  
 *
 * Copyright (C) 2008--2012  Chad Hanna, Kipp Cannon, Drew Keppel
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
#include <gst/base/gstbasetransform.h>


#include <lal/LIGOMetadataTables.h>


G_BEGIN_DECLS


#define GSTLAL_SIMULATION_TYPE \
	(gstlal_simulation_get_type())
#define GSTLAL_SIMULATION(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_SIMULATION_TYPE, GSTLALSimulation))
#define GSTLAL_SIMULATION_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_SIMULATION_TYPE, GSTLALSimulationClass))
#define GST_IS_GSTLAL_SIMULATION(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_SIMULATION_TYPE))
#define GST_IS_GSTLAL_SIMULATION_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_SIMULATION_TYPE))


typedef struct {
	GstBaseTransformClass parent_class;
} GSTLALSimulationClass;


typedef struct {
	GstBaseTransform parent;

	char *xml_location;

	struct injection_document {
		int has_sim_burst_table;
		SimBurst *sim_burst_table_head;
		int has_sim_inspiral_table;
		SimInspiralTable *sim_inspiral_table_head;
		int has_time_slide_table;
		TimeSlide *time_slide_table_head;
	} *injection_document;

	char *instrument;
	char *channel_name;
	char *units;

	REAL8TimeSeries *simulation_series;
} GSTLALSimulation;


GType gstlal_simulation_get_type(void);


G_END_DECLS


#endif	/* __GSTLAL_SIMULATION_H__ */

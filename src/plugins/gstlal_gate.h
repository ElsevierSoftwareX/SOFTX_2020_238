/*
 * An element to flag buffers in a stream as silence or not based on the
 * value of a control input.
 *
 * Copyright (C) 2008  Kipp Cannon, Chad Hanna
 *
 *  This program is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU General Public License as published by the
 *  Free Software Foundation; either version 2 of the License, or (at your
 *  option) any later version.
 *
 *  This program is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */


#ifndef __GSTLAL_GATE_H__
#define __GSTLAL_GATE_H__


#include <glib.h>
#include <gst/gst.h>


G_BEGIN_DECLS


#define GSTLAL_GATE_TYPE \
	(gstlal_gate_get_type())
#define GSTLAL_GATE(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_GATE_TYPE, GSTLALGate))
#define GSTLAL_GATE_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_GATE_TYPE, GSTLALGateClass))
#define GST_IS_GSTLAL_GATE(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_GATE_TYPE))
#define GST_IS_GSTLAL_GATE_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_GATE_TYPE))


typedef struct {
	GstElementClass parent_class;
} GSTLALGateClass;


typedef struct _GSTLALGate {
	GstElement element;

	GstPad *controlpad;
	GstPad *sinkpad;
	GstPad *srcpad;

	GCond *control_available;
	GCond *control_flushed;
	GstBuffer *control_buf;
	GstClockTime control_end;
	double (*control_sample_func)(const struct _GSTLALGate *, size_t);

	double threshold;

	gint rate;
	gint bytes_per_sample;
	gint control_rate;
} GSTLALGate;


GType gstlal_gate_get_type(void);


G_END_DECLS


#endif	/* __GSTLAL_GATE_H__ */

/*
 * An element to flag buffers in a stream as silence or not based on the
 * value of a control input.
 *
 * Copyright (C) 2008--2012  Kipp Cannon, Chad Hanna
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


typedef struct _GSTLALGateClass GSTLALGateClass;
typedef struct _GSTLALGate GSTLALGate;


struct _GSTLALGateClass {
	GstElementClass parent_class;

	void (*rate_changed)(GSTLALGate *, gint, void *);
	void (*start)(GSTLALGate *, guint64, void *);
	void (*stop)(GSTLALGate *, guint64, void *);
};


struct _GSTLALGate {
	GstElement element;

	GstPad *controlpad;
	GstPad *sinkpad;
	GstPad *srcpad;

	GMutex *control_lock;
	gboolean control_eos;
	gboolean sink_eos;
	GstClockTime t_sink_head;
	GArray *control_segments;
	GCond *control_queue_head_changed;
	gdouble (*control_sample_func)(const gpointer, guint64);

	gboolean emit_signals;
	gboolean default_state;
	gint last_state;
	gdouble threshold;
	gint64 attack_length;
	gint64 hold_length;
	gboolean leaky;
	gboolean invert_control;

	gint rate;
	gint unit_size;
	gint control_rate;
	gboolean need_discont;
};


GType gstlal_gate_get_type(void);


G_END_DECLS


#endif	/* __GSTLAL_GATE_H__ */

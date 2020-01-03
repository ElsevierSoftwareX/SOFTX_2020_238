/*
 * Copyright (C) 2019 Aaron Viets <aaron.viets@ligo.org>
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation; either version 2 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */


#include <gst/gst.h>

#define GSTLAL_MAKEDISCONT_TYPE \
	(gstlal_makediscont_get_type())
#define GSTLAL_MAKEDISCONT(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj),GSTLAL_MAKEDISCONT_TYPE,GSTLALMakeDiscont))
#define GSTLAL_MAKEDISCONT_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass),GSTLAL_MAKEDISCONT_TYPE,GSTLALMakeDiscontClass))
#define IS_PLUGIN_TEMPLATE(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj),GSTLAL_MAKEDISCONT_TYPE))
#define IS_PLUGIN_TEMPLATE_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass),GSTLAL_MAKEDISCONT_TYPE))

typedef struct _GSTLALMakeDiscont GSTLALMakeDiscont;
typedef struct _GSTLALMakeDiscontClass GSTLALMakeDiscontClass;

struct _GSTLALMakeDiscont {

	GstElement element;
	GstPad *sinkpad, *srcpad;

	/* properties */
	guint64 dropout_time;
	guint64 data_time;
	guint64 heartbeat_time;
	double switch_probability;
	guint64 sleep_time;

	/* filter memory */
	int current_data_state;
	guint64 next_switch_time;

	/* timestamp book-keeping */
	GstClockTime t0;
	guint64 offset0;
	guint64 next_in_offset;
};

struct _GSTLALMakeDiscontClass {
	GstElementClass parent_class;
};

GType gstlal_makediscont_get_type (void);



/*
 * gds lvshm (LIGO-Virgo Shared Memory) sink element
 *
 * Copyright (C) 2012  Kipp C. Cannon
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


#ifndef __GDS_LVSHMSINK_H__
#define __GDS_LVSHMSINK_H__

/*
 * the GDS lvshmapi library is only availale on Unix-like systems where
 * pthread.h will be available, and therefore on systems where glib's
 * threading library will be wrapping pthreads.  so we don't bother
 * conditionally-compiling the pthread-related code.  the entire gds plugin
 * will be disabled on systems where this won't work.
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasesink.h>


G_BEGIN_DECLS


/*
 * gds_lvshmsink_buffer_mode enum type
 */


enum gds_lvshmsink_buffer_mode {
	GDS_LVSHMSINK_BUFFER_MODE_0 = 0,
	GDS_LVSHMSINK_BUFFER_MODE_1 = 1,
	GDS_LVSHMSINK_BUFFER_MODE_2 = 2,
	GDS_LVSHMSINK_BUFFER_MODE_3 = 3,
};


#define GDS_LVSHMSINK_BUFFER_MODE_TYPE \
	(gds_lvshmsink_buffer_mode_get_type())


GType gds_lvshmsink_buffer_mode_get_type(void);


/*
 * GDSLVSHMSink type
 */


#define GDS_LVSHMSINK_TYPE \
	(gds_lvshmsink_get_type())
#define GDS_LVSHMSINK(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GDS_LVSHMSINK_TYPE, GDSLVSHMSink))
#define GDS_LVSHMSINK_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GDS_LVSHMSINK_TYPE, GDSLVSHMSinkClass))
#define GST_IS_GDS_LVSHMSINK(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GDS_LVSHMSINK_TYPE))
#define GST_IS_GDS_LVSHMSINK_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GDS_LVSHMSINK_TYPE))


typedef struct {
	GstBaseSinkClass parent_class;
} GDSLVSHMSinkClass;


typedef struct {
	GstBaseSink basesrc;

	/*
	 * properties
	 */

	char *name;
	guint num_buffers;
	guint mask;
	enum gds_lvshmsink_buffer_mode buffer_mode;
	gboolean lock;

	/*< private >*/

	/*
	 * LSMP_PROD producer interface.  this is declared void * here and
	 * type casts are used in the module proper so that other code that
	 * uses this header can be compiled without loading the lsmp
	 * headers from gds.  the lsmp headers require C++, and there is no
	 * pkg-config file available for them.
	 */

	void *partition;
} GDSLVSHMSink;


GType gds_lvshmsink_get_type(void);


G_END_DECLS


#endif	/* __GDS_LVSHMSINK_H__ */

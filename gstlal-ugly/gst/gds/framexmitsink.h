/*
 * GDS framexmit broadcast transmitter sink element
 *
 * Copyright (C) 2012--2013  Kipp Cannon
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


#ifndef __GDS_FRAMEXMITSINK_H__
#define __GDS_FRAMEXMITSINK_H__


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasesink.h>


G_BEGIN_DECLS


/*
 * gds_framexmitsink element
 */


#define GDS_FRAMEXMITSINK_TYPE \
	(gds_framexmitsink_get_type())
#define GDS_FRAMEXMITSINK(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GDS_FRAMEXMITSINK_TYPE, GstGDSFramexmitSink))
#define GDS_FRAMEXMITSINK_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GDS_FRAMEXMITSINK_TYPE, GstGDSFramexmitSinkClass))
#define GDS_FRAMEXMITSINK_GET_CLASS(obj) \
	(G_TYPE_INSTANCE_GET_CLASS((obj), GDS_FRAMEXMITSINK_TYPE, GstGDSFramexmitSinkClass))
#define GST_IS_GDS_FRAMEXMITSINK(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GDS_FRAMEXMITSINK_TYPE))
#define GST_IS_GDS_FRAMEXMITSINK_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GDS_FRAMEXMITSINK_TYPE))


typedef struct _GstGDSFramexmitSinkClass GstGDSFramexmitSinkClass;
typedef struct _GstGDSFramexmitSink GstGDSFramexmitSink;


struct _GstGDSFramexmitSinkClass {
	GstBaseSinkClass parent_class;
};


struct _GstGDSFramexmitSink {
	GstBaseSink element;

	/*
	 * properties
	 */

	gint max_size_buffers;
	gchar *iface;
	gchar *group;
	gint port;

	/*
	 * opaque frame transmitter
	 */

	void *frameSend;
};


GType gds_framexmitsink_get_type(void);


G_END_DECLS


#endif	/* __GDS_FRAMEXMITSINK_H__ */

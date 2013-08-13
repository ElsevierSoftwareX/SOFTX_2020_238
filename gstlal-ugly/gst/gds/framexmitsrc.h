/*
 * GDS framexmit broadcast receiver source element
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


#ifndef __GDS_FRAMEXMITSRC_H__
#define __GDS_FRAMEXMITSRC_H__


#include <pthread.h>


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstpushsrc.h>


G_BEGIN_DECLS


/*
 * gds_framexmitsrc_qos enum
 */


enum gds_framexmitsrc_qos {
	GDS_FRAMEXMITSRC_QOS_30 = 0,
	GDS_FRAMEXMITSRC_QOS_10 = 1,
	GDS_FRAMEXMITSRC_QOS_3 = 2
};


#define GDS_FRAMEXMITSRC_QOS_TYPE (gds_framexmitsrc_qos_get_type())


GType gds_framexmitsrc_qos_get_type(void);


/*
 * gds_framexmitsrc element
 */


#define GDS_FRAMEXMITSRC_TYPE \
	(gds_framexmitsrc_get_type())
#define GDS_FRAMEXMITSRC(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GDS_FRAMEXMITSRC_TYPE, GstGDSFramexmitSrc))
#define GDS_FRAMEXMITSRC_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GDS_FRAMEXMITSRC_TYPE, GstGDSFramexmitSrcClass))
#define GDS_FRAMEXMITSRC_GET_CLASS(obj) \
	(G_TYPE_INSTANCE_GET_CLASS((obj), GDS_FRAMEXMITSRC_TYPE, GstGDSFramexmitSrcClass))
#define GST_IS_GDS_FRAMEXMITSRC(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GDS_FRAMEXMITSRC_TYPE))
#define GST_IS_GDS_FRAMEXMITSRC_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GDS_FRAMEXMITSRC_TYPE))


typedef struct _GstGDSFramexmitSrcClass GstGDSFramexmitSrcClass;
typedef struct _GstGDSFramexmitSrc GstGDSFramexmitSrc;


struct _GstGDSFramexmitSrcClass {
	GstPushSrcClass parent_class;
};


struct _GstGDSFramexmitSrc {
	GstPushSrc element;

	/*
	 * receive thread
	 */

	pthread_t recv_thread;
	GstFlowReturn recv_status;
	GstBuffer *buffer;
	GMutex *buffer_lock;
	GCond *received_buffer;
	gboolean unblocked;

	/*
	 * properties
	 */

	gchar *iface;
	gchar *group;
	gint port;
	enum gds_framexmitsrc_qos qos;
	gdouble wait_time;

	/*
	 * latency
	 */

	GstClockTimeDiff max_latency;
	GstClockTimeDiff min_latency;

	/*
	 * state
	 */

	gboolean need_new_segment;
	GstClockTime next_timestamp;

	/*< private >*/

	/*
	 * framexmit::frameRecv interface.  this is declared void * here
	 * and type casts are used in the module proper so that other code
	 * that uses this header can be compiled without loading the
	 * framexmit headers from gds.  the framexmit headers require C++
	 * and there is no pkg-config file available for them.
	 */

	void *frameRecv;
};


GType gds_framexmitsrc_get_type(void);


G_END_DECLS


#endif	/* __GDS_FRAMEXMITSRC_H__ */

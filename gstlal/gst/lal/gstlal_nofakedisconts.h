/*
 * An element to fix discontinuity flags
 *
 * Copyright (C) 2009  Kipp Cannon
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


#ifndef __GSTLAL_NOFAKEDISCONTS_H__
#define __GSTLAL_NOFAKEDISCONTS_H__


#include <glib.h>
#include <gst/gst.h>


G_BEGIN_DECLS


#define GSTLAL_NOFAKEDISCONTS_TYPE \
	(gstlal_nofakedisconts_get_type())
#define GSTLAL_NOFAKEDISCONTS(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_NOFAKEDISCONTS_TYPE, GSTLALNoFakeDisconts))
#define GSTLAL_NOFAKEDISCONTS_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_NOFAKEDISCONTS_TYPE, GSTLALNoFakeDiscontsClass))
#define GST_IS_GSTLAL_NOFAKEDISCONTS(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_NOFAKEDISCONTS_TYPE))
#define GST_IS_GSTLAL_NOFAKEDISCONTS_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_NOFAKEDISCONTS_TYPE))


typedef struct _GSTLALNoFakeDisconts GSTLALNoFakeDisconts;
typedef struct _GSTLALNoFakeDiscontsClass GSTLALNoFakeDiscontsClass;


/**
 * GSTLALNoFakeDisconts:
 */


struct _GSTLALNoFakeDisconts {
	GstElement element;

	GstPad *sinkpad;
	GstPad *srcpad;

	guint64 next_offset;
	guint64 next_timestamp;
	gboolean silent;
};


/**
 * GSTLALNoFakeDiscontsClass:
 */


struct _GSTLALNoFakeDiscontsClass {
	GstElementClass parent_class;
};


GType gstlal_nofakedisconts_get_type(void);


G_END_DECLS


#endif	/* __GSTLAL_NOFAKEDISCONTS_H__ */

/*
 * IGWD frame file parser
 *
 * Copyright (C) 2012  Kipp Cannon
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


#ifndef __FRAMECPP_IGWDPARSE_H__
#define __FRAMECPP_IGWDPARSE_H__


#include <gst/gst.h>
#include <gst/base/gstbaseparse.h>


G_BEGIN_DECLS


/*
 * framecpp_igwdparse element
 */


#define FRAMECPP_IGWDPARSE_TYPE \
	(framecpp_igwdparse_get_type())
#define FRAMECPP_IGWDPARSE(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), FRAMECPP_IGWDPARSE_TYPE, GSTFrameCPPIGWDParse))
#define FRAMECPP_IGWDPARSE_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), FRAMECPP_IGWDPARSE_TYPE, GSTFrameCPPIGWDParseClass))
#define GST_IS_FRAMECPP_IGWDPARSE(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), FRAMECPP_IGWDPARSE_TYPE))
#define GST_IS_FRAMECPP_IGWDPARSE_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), FRAMECPP_IGWDPARSE_TYPE))


typedef struct {
	GstBaseParseClass parent_class;
} GSTFrameCPPIGWDParseClass;


typedef struct {
	GstBaseParse element;

	gint endianness;
	guint sizeof_int_2;
	guint sizeof_int_4;
	guint sizeof_int_8;
	guint sizeof_table_6;
	guint16 frameh_klass;
	guint16 eof_klass;

	GstClockTime file_start_time;
	GstClockTime file_stop_time;
	size_t offset;
} GSTFrameCPPIGWDParse;


GST_DEBUG_CATEGORY_EXTERN(framecpp_igwdparse_debug);
GType framecpp_igwdparse_get_type(void);


G_END_DECLS


#endif	/* __FRAMECPP_IGWDPARSE_H__ */

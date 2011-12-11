/*
 * gds lvshm (LIGO-Virgo Shared Memory) source element
 *
 * Copyright (C) 2011  Kipp C. Cannon
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


#ifndef __GDS_LVSHMSRC_H__
#define __GDS_LVSHMSRC_H__


#include <gst/gst.h>
#include <gst/base/gstpushsrc.h>

#include <gds/lvshmapi.h>


G_BEGIN_DECLS


#define GDS_LVSHMSRC_TYPE \
	(gsd_lvshmsrc_get_type())
#define GDS_LVSHMSRC(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GDS_LVSHMSRC_TYPE, GDSLVSHMSrc))
#define GDS_LVSHMSRC_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GDS_LVSHMSRC_TYPE, GDSLVSHMSrcClass))
#define GST_IS_GDS_LVSHMSRC(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GDS_LVSHMSRC_TYPE))
#define GST_IS_GDS_LVSHMSRC_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GDS_LVSHMSRC_TYPE))


typedef struct {
	GstPushSrcClass parent_class;
} GDSLVSHMSrcClass;


typedef struct {
	GstPushSrc basesrc;

	/*
	 * properties
	 */

	char *name;
	lvshm_mask mask;
	double wait_time;

	/*
	 * state
	 */

	lvshm_handle handle;
	gboolean need_new_segment;
} GDSLVSHMSrc;


GST_DEBUG_CATEGORY_EXTERN(gds_lvshmsrc_debug);
GType gsd_lvshmsrc_get_type(void);


G_END_DECLS


#endif	/* __GDS_LVSHMSRC_H__ */

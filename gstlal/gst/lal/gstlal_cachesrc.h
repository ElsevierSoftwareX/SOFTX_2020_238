/*
 * GstLALCacheSrc
 *
 * Copyright (C) 2012  Kipp Cannon
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */


#ifndef __GSTLAL_CACHESRC_H__
#define __GSTLAL_CACHESRC_H__


/*
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasesrc.h>


#include <lal/LALCache.h>


G_BEGIN_DECLS


/*
 * ============================================================================
 *
 *                                    Type
 *
 * ============================================================================
 */


#define GSTLAL_CACHESRC_TYPE \
	(gstlal_cachesrc_get_type())
#define GSTLAL_CACHESRC(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_CACHESRC_TYPE, GstLALCacheSrc))
#define GSTLAL_CACHESRC_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_CACHESRC_TYPE, GstLALCacheSrcClass))
#define GSTLAL_CACHESRC_GET_CLASS(obj) \
	(G_TYPE_INSTANCE_GET_CLASS((obj), GSTLAL_CACHESRC_TYPE, GstLALCacheSrcClass))
#define GST_IS_LAL_CACHESRC(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_CACHESRC_TYPE))
#define GST_IS_LAL_CACHESRC_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_CACHESRC_TYPE))


typedef struct _GstLALCacheSrc GstLALCacheSrc;
typedef struct _GstLALCacheSrcClass GstLALCacheSrcClass;


/**
 * GstLALCacheSrc:
 */


struct _GstLALCacheSrc {
	GstBaseSrc basesrc;

	gchar *location;
	gchar *cache_src_regex;
	gchar *cache_dsc_regex;
	gboolean use_mmap;

	LALCache *cache;
	guint index;
	gboolean need_discont;
};


/**
 * GstLALCacheSrcClass:
 */


struct _GstLALCacheSrcClass {
	GstBaseSrcClass parent_class;
};


/*
 * ============================================================================
 *
 *                                Exported API
 *
 * ============================================================================
 */


GType gstlal_cachesrc_get_type(void);


G_END_DECLS


#endif	/* __GSTLAL_CACHESRC_H__ */

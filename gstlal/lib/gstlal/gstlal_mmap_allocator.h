/*
 * GstLALMMapAllocator
 *
 * Copyright (C) 2012 Kipp Cannon
 * Copyright (C) 2016 Cody Messick
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


#ifndef __GSTLAL_MMAP_ALLOCATOR_H__
#define __GSTLAL_MMAP_ALLOCATOR_H__


/*
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


#include <glib.h>
#include <gst/gst.h>


G_BEGIN_DECLS


/*
 * ============================================================================
 *
 *                                    Type
 *
 * ============================================================================
 */


#define GSTLAL_MMAP_ALLOCATOR_TYPE (gstlal_mmapallocator_get_type())
#define GSTLAL_MMAP_ALLOCATOR(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_MMAP_ALLOCATOR_TYPE, GstLALMmapAllocator))
#define GSTLAL_MMAP_ALLOCATOR_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_MMAP_ALLOCATOR_TYPE, GstLALMmapAllocatorClass))
#define GSTLAL_MMAP_ALLOCATOR_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS((obj), GSTLAL_MMAP_ALLOCATOR_TYPE, GstLALMmapAllocatorClass))
#define GST_IS_LAL_MMAP_ALLOCATOR(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_MMAP_ALLOCATOR_TYPE))
#define GST_IS_LAL_MMAP_ALLOCATOR_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_MMAP_ALLOCATOR_TYPE))


typedef struct _GstLALMmapAllocator GstLALMmapAllocator;
typedef struct _GstLALMmapAllocatorClass GstLALMmapAllocatorClass;

/**
 * GST_ALLOCATOR_SYSMEM:
 *
 * The allocator name for the default system memory allocator
 */
#define GSTLAL_MMAP_ALLOCATOR_NAME   "MmapAllocator"

/**
 * GstLALMmapAllocator:
 */
struct _GstLALMmapAllocator {
	GstAllocator allocator;
	int fd;
	gsize size;
};


/**
 * GstLALMmapAllocatorClass:
 * @parent_class:  the parent class
 */
struct _GstLALMmapAllocatorClass {
	GstAllocatorClass parent_class;
};

/*
 * ============================================================================
 *
 *                                Exported API
 *
 * ============================================================================
 */


GType gstlal_mmapallocator_get_type(void);


G_END_DECLS


#endif	/* __GSTLAL_MMAP_ALLOCATOR_H__ */

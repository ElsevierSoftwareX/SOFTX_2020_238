/*
 * GstLALMmapAllocator
 *
 * Copyright (C) 2012,2013  Kipp Cannon
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


/**
 * SECTION:gstlal_mmap_allocator
 * @include: gstlal/gstlal_mmap_allocator.h
 * @short_description:  GstAllocator sub-class that reports GPS time.
 *
 * The #GstLALMmapAllocator class is a sub-class of #GstAllocator that
 * converts uses mmap to allocate space and a custom wraper of munmap 
 * to free space
 */


/*
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


#include <glib.h>
#include <gst/gst.h>
#include <sys/mman.h>


#include <gstlal_mmap_allocator.h>


/*
 * ============================================================================
 *
 *                                Boilerplate
 *
 * ============================================================================
 */


G_DEFINE_TYPE(GstLALMmapAllocator, gstlal_mmap_allocator, GSTLAL_MMAP_ALLOCATOR_TYPE);


/*
 * ============================================================================
 *
 *                       GstAllocate Replacement Methods
 *
 * ============================================================================
 */


/**
 * mmap_alloc:
 * @allocator: an instance of #GstAllocator 
 * @size: size of memory to allocate
 * @params: Optional #GstAllocationParams
 */
static GstMemory *mmap_alloc(GstAllocator *allocator, gsize size, GstAllocationParams *params)
{
	//FIXME Make more general so that user can choose mmap flags
	
	GstMemory *memory;	

	g_assert (allocator != NULL);
	g_assert(GST_IS_LAL_MMAP_ALLOCATOR(allocator));
	
	// FIXME Check to make sure call to mmap was successful, implying that
	memory = (GstMemory *) mmap(NULL, size, PROT_READ, MAP_PRIVATE | MAP_NORESERVE, ((GstLALMmapAllocator*) allocator)->fd, 0);
	return memory;
}


/**
 * munmap_alloc:
 * @allocator: #GstAllocator instance
 * @memory: #GstMemory instance
 */
static void munmap_free(GstAllocator *allocator, GstMemory *memory)
{
	if(memory) 
	{
		g_assert (allocator != NULL);
		g_assert(memory->allocator == (GstAllocator*) allocator);
		g_assert(GST_IS_LAL_MMAP_ALLOCATOR(allocator));
		munmap(memory, memory->size);
        }
}


/*
 * ============================================================================
 *
 *                              GObject Methods
 *
 * ============================================================================
 */

enum
{
	PROP_0,
	PROP_FD
};

static void gstlal_mmap_allocator_set_property(GObject * object, guint property_id, const GValue * value, GParamSpec * pspec)
{
	GstLALMmapAllocator *allocator = GSTLAL_MMAP_ALLOCATOR(object);

	switch(property_id)
	{
		case PROP_FD:
			// FIXME Add logic to prevent fd from being changed mmap is called
			allocator->fd = g_value_get_int(value);
			break;
	}
}

static void gstlal_mmap_allocator_get_property(GObject * object, guint property_id, GValue * value, GParamSpec * pspec)
{
	GstLALMmapAllocator *allocator = GSTLAL_MMAP_ALLOCATOR(object);

	switch(property_id)
	{
		case PROP_FD:
			g_value_set_int(value, allocator->fd);
			break;
	}
}

static void gstlal_mmap_allocator_class_init(GstLALMmapAllocatorClass *klass)
{
	GstAllocatorClass *allocator_class = GST_ALLOCATOR_CLASS(klass);
	GObjectClass *object_class = G_OBJECT_CLASS (klass);

	allocator_class->alloc = mmap_alloc;
	allocator_class->free = munmap_free;

	object_class->set_property = gstlal_mmap_allocator_set_property;
	object_class->get_property = gstlal_mmap_allocator_get_property;

	g_object_class_install_property (object_class, PROP_FD, g_param_spec_int("fd", 
		"File Descriptor", "It's a file descriptor, what else do you need to know?", 
		0, G_MAXINT, 0, G_PARAM_READABLE | G_PARAM_WRITABLE));
}


static void gstlal_mmap_allocator_init(GstLALMmapAllocator *object)
{
	// Initilize the fd property to an impossible value so that mmap will
	// fail when called if fd hasnt been set
	object->fd = -1; 
}

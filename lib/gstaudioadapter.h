/*
 * GstAudioAdapter
 *
 * Copyright (C) 2011  Kipp Cannon
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


#ifndef __GSTAUDIOADAPTER_H__
#define __GSTAUDIOADAPTER_H__


/*
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


#include <glib.h>
#include <gst/gst.h>


/*
 * ============================================================================
 *
 *                                    Type
 *
 * ============================================================================
 */


struct gstlal_input_queue {
	GQueue *queue;
	gint unit_size;
	gint size;
	gint skip;
};


/*
 * ============================================================================
 *
 *                            Function Prototypes
 *
 * ============================================================================
 */


struct gstlal_input_queue *gstlal_input_queue_create(gint);
void gstlal_input_queue_drain(struct gstlal_input_queue *);
void gstlal_input_queue_free(struct gstlal_input_queue *);
gint gstlal_input_queue_get_size(const struct gstlal_input_queue *);
gint gstlal_input_queue_get_unit_size(const struct gstlal_input_queue *);
void gstlal_input_queue_set_unit_size(struct gstlal_input_queue *, gint);
void gstlal_input_queue_push(struct gstlal_input_queue *, GstBuffer *);
gboolean gstlal_input_queue_is_gap(struct gstlal_input_queue *);
void gstlal_input_queue_copy(struct gstlal_input_queue *, void *, guint, gboolean *, gboolean *);
void gstlal_input_queue_flush(struct gstlal_input_queue *, guint);


#endif	/* __GSTAUDIOADAPTER_H__ */

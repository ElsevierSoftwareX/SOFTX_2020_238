/*
 * Copyright (C) 2010 Leo Singer
 * Copyright (C) 2016 Aaron Viets
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */


#ifndef __CMATH_BASE_H__
#define __CMATH_BASE_H__


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <stdio.h>
#include <string.h>
#include <complex.h>
#include <stdlib.h>
#include <math.h>

G_BEGIN_DECLS
#define CMATH_BASE_TYPE \
	(cmath_base_get_type())
#define CMATH_BASE(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), CMATH_BASE_TYPE, CMathBase))
#define CMATH_BASE_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), CMATH_BASE_TYPE, CMathBaseClass))
#define GST_IS_CMATH_BASE(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), CMATH_BASE_TYPE))
#define GST_IS_CMATH_BASE_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), CMATH_BASE_TYPE))

typedef struct _CMathBase CMathBase;
typedef struct _CMathBaseClass CMathBaseClass;

struct _CMathBase
{
	GstBaseTransform element;

	int is_complex;
	int bits;
	int channels;
	int rate;
	const gchar *interleave;
};

struct _CMathBaseClass
{
	GstBaseTransformClass parent_class;
};

GType cmath_base_get_type(void);

/*
 * set_caps() is called when the caps are re-negotiated. Can return
 * false if we do not like what we see.
 */

gboolean set_caps(GstBaseTransform *trans, GstCaps *incaps,
	GstCaps *outcaps);


G_END_DECLS
#endif /* __CMATH_BASE_H__ */

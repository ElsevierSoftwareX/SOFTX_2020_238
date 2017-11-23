/*
 * GstLALPyFuncSrc
 *
 * Copyright (C) 2016  Kipp Cannon
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


#ifndef __GSTLAL_PYFUNCSRC_H__
#define __GSTLAL_PYFUNCSRC_H__


/*
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/audio/audio.h>
#include <gst/base/gstbasesrc.h>


#include <Python.h>


G_BEGIN_DECLS


/*
 * ============================================================================
 *
 *                                    Type
 *
 * ============================================================================
 */


#define GSTLAL_PYFUNCSRC_TYPE \
	(gstlal_pyfuncsrc_get_type())
#define GSTLAL_PYFUNCSRC(obj) \
	(G_TYPE_CHECK_INSTANCE_CAST((obj), GSTLAL_PYFUNCSRC_TYPE, GstLALPyFuncSrc))
#define GSTLAL_PYFUNCSRC_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_CAST((klass), GSTLAL_PYFUNCSRC_TYPE, GstLALPyFuncSrcClass))
#define GSTLAL_PYFUNCSRC_GET_CLASS(obj) \
	(G_TYPE_INSTANCE_GET_CLASS((obj), GSTLAL_PYFUNCSRC_TYPE, GstLALPyFuncSrcClass))
#define GST_IS_LAL_PYFUNCSRC(obj) \
	(G_TYPE_CHECK_INSTANCE_TYPE((obj), GSTLAL_PYFUNCSRC_TYPE))
#define GST_IS_LAL_PYFUNCSRC_CLASS(klass) \
	(G_TYPE_CHECK_CLASS_TYPE((klass), GSTLAL_PYFUNCSRC_TYPE))


typedef struct _GstLALPyFuncSrc GstLALPyFuncSrc;
typedef struct _GstLALPyFuncSrcClass GstLALPyFuncSrcClass;


/**
 * GstLALPyFuncSrc:
 */


struct _GstLALPyFuncSrc {
	GstBaseSrc basesrc;

	gchar *expression;
	PyCodeObject *code;
	PyObject *globals;

	GstAudioInfo audioinfo;

	GstSegment segment;
	guint64 offset;
};


/**
 * GstLALPyFuncSrcClass:
 * @parent_class:  the parent class
 */


struct _GstLALPyFuncSrcClass {
	GstBaseSrcClass parent_class;
};


/*
 * ============================================================================
 *
 *                                Exported API
 *
 * ============================================================================
 */


GType gstlal_pyfuncsrc_get_type(void);


G_END_DECLS


#endif	/* __GSTLAL_PYFUNCSRC_H__ */

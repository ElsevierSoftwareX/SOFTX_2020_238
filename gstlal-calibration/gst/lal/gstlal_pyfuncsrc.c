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


/*
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


/* FIXME:  only used for dlopen(), is this really how to do this? */
#include <dlfcn.h>


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasesrc.h>


#include <Python.h>


#include <gstlal/gstlal_audio_info.h>
#include <gstlal/gstlal_debug.h>
#include <gstlal_pyfuncsrc.h>


/*
 * ============================================================================
 *
 *                                Boilerplate
 *
 * ============================================================================
 */


#define GST_CAT_DEFAULT gstlal_pyfuncsrc_debug
GST_DEBUG_CATEGORY_STATIC(GST_CAT_DEFAULT);


G_DEFINE_TYPE_WITH_CODE(
	GstLALPyFuncSrc,
	gstlal_pyfuncsrc,
	GST_TYPE_BASE_SRC,
	GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "lal_pyfuncsrc", 0, "lal_pyfuncsrc element");
);


/*
 * ============================================================================
 *
 *                                 Parameters
 *
 * ============================================================================
 */


#define DEFAULT_EXPRESSION "0.01 * (sin(2. * pi * 256. * t) + sin(2. * pi * 440. * t))\n\t\t\t# quiet middle C and A"


/*
 * ============================================================================
 *
 *                             Internal Functions
 *
 * ============================================================================
 */


/*
 * eval()
 *
 * evalute the compiled expression at t.  must be called with the global
 * interpreter lock held
 */


static PyObject *eval(GstLALPyFuncSrc *element, GstClockTime t)
{
	PyObject *locals, *result;

	locals = Py_BuildValue("{s:d}", "t", t / (gdouble) GST_SECOND);
	if(!locals) {
		PyErr_Print();
		return NULL;
	}

	result = PyEval_EvalCode(element->code, element->globals, locals);
	if(!result)
		PyErr_Print();

	Py_DECREF(locals);
	return result;
}


/*
 * get_caps_filter()
 *
 * restrict allowed caps using current expression
 */


static GstCaps *get_caps_filter(GstLALPyFuncSrc *element)
{
	GstCaps *filter = NULL;
	const gchar *format;
	gint channels;
	PyObject *val;
	gboolean success = TRUE;

	/* evaluate at t = 0 */
	val = eval(element, 0);
	if(!val)
		goto done;

	if(PyFloat_Check(val)) {
		/* result is a scalar float */
		format = GST_AUDIO_NE(F64);
		channels = 1;
	} else if(PyComplex_Check(val)) {
		/* result is a scalar complex */
		format = GST_AUDIO_NE(Z128);
		channels = 1;
	} else if(PySequence_Check(val) && PySequence_Length(val) > 0) {
		/* result is an array-like value with at least 1 entry */
		PyObject *elem = PySequence_ITEM(val, 0);

		channels = PySequence_Length(val);

		if(PyFloat_Check(elem)) {
			/* elements are floats */
			format = GST_AUDIO_NE(F64);
		} else if(PyComplex_Check(elem)) {
			/* elements are complex */
			format = GST_AUDIO_NE(Z128);
		} else {
			/* unsupported element type */
			success = FALSE;
		}

		Py_DECREF(elem);
	} else {
		/* unsupported type */
		success = FALSE;
	}

	Py_DECREF(val);

	if(success) {
		filter = gst_caps_new_simple("audio/x-raw", "format", G_TYPE_STRING, format, "channels", G_TYPE_INT, channels, NULL);
	} else
		GST_ELEMENT_ERROR(element, STREAM, FORMAT, (NULL), ("expression \"%s\" returned unsupported type", element->expression));

done:
	return filter;
}


/*
 * unpack()
 *
 * unpack the Python object into the the given memory location.  takes
 * ownership of val.
 */


static gboolean unpack(void *dst, gboolean is_real, gint channels, PyObject *val)
{
	gboolean success = TRUE;

	if(PySequence_Check(val)) {
		gint sequence_length = PySequence_Length(val);
		gint i;
		g_assert_cmpint(sequence_length, ==, channels);
		for(i = 0; i < sequence_length && !PyErr_Occurred(); i++) {
			PyObject *item = PySequence_ITEM(val, i);
			if(is_real)
				((gdouble *) dst)[i] = PyFloat_AsDouble(item);
			else
				((Py_complex *) dst)[i] = PyComplex_AsCComplex(item);
			Py_DECREF(item);
		}
	} else if(is_real)
		*(gdouble *) dst = PyFloat_AsDouble(val);
	else
		*(Py_complex *) dst = PyComplex_AsCComplex(val);

	if(PyErr_Occurred()) {
		PyErr_Print();
		success = FALSE;
	}

	Py_DECREF(val);
	return success;
}


/*
 * ============================================================================
 *
 *                             GstBaseSrc Methods
 *
 * ============================================================================
 */


/*
 * get_caps()
 */


static GstCaps *get_caps(GstBaseSrc *basesrc, GstCaps *filter)
{
	GstLALPyFuncSrc *element = GSTLAL_PYFUNCSRC(basesrc);
	/* start with template caps */
	GstCaps *caps = gst_pad_get_pad_template_caps(GST_BASE_SRC_PAD(basesrc));

	/* intersect with calling code's filter if supplied */
	if(filter) {
		GstCaps *intersect = gst_caps_intersect(caps, filter);
		gst_caps_unref(caps);
		caps = intersect;
	}

	/* intersect with the format allowed by the current expression */
	filter = get_caps_filter(element);
	if(filter) {
		GstCaps *intersect = gst_caps_intersect(caps, filter);
		gst_caps_unref(caps);
		gst_caps_unref(filter);
		caps = intersect;
	}

	/* done */
	return caps;
}


/*
 * set_caps()
 */


static gboolean set_caps(GstBaseSrc *basesrc, GstCaps *caps)
{
	GstLALPyFuncSrc *element = GSTLAL_PYFUNCSRC(basesrc);

	return gstlal_audio_info_from_caps(&element->audioinfo, caps);
}


/*
 * is_seekable()
 */


static gboolean is_seekable(GstBaseSrc *basesrc)
{
	return TRUE;
}


/*
 * fill()
 */


static GstFlowReturn fill(GstBaseSrc *basesrc, guint64 offset, guint size, GstBuffer *buf)
{
	GstLALPyFuncSrc *element = GSTLAL_PYFUNCSRC(basesrc);
	/*PyGILState_STATE gilstate = PyGILState_Ensure();*/
	GstMapInfo mapinfo;
	gboolean is_real = !strncmp(GST_AUDIO_INFO_NAME(&element->audioinfo), "F64", 3);
	gint channels = GST_AUDIO_INFO_CHANNELS(&element->audioinfo);
	gsize i, len;
	guint8 *data;
	GstFlowReturn result = GST_FLOW_OK;

	gst_buffer_map(buf, &mapinfo, GST_MAP_WRITE);

	data = mapinfo.data;
	len = mapinfo.size / GST_AUDIO_INFO_BPF(&element->audioinfo);

	for(i = 0; i < len; i++) {
		GstClockTime t = element->segment.start + gst_util_uint64_scale_int_round(element->offset + i, GST_SECOND, GST_AUDIO_INFO_RATE(&element->audioinfo));
		PyObject *val = eval(element, t);
		if(!val || !unpack(data, is_real, channels, val)) {
			result = GST_FLOW_ERROR;
			goto done;
		}
		data += GST_AUDIO_INFO_BPF(&element->audioinfo);
	}

	GST_BUFFER_OFFSET(buf) = element->offset;
	element->offset += len;
	GST_BUFFER_OFFSET_END(buf) = element->offset;
	GST_BUFFER_TIMESTAMP(buf) = element->segment.start + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET(buf), GST_SECOND, GST_AUDIO_INFO_RATE(&element->audioinfo));
	element->segment.position = element->segment.start + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET_END(buf), GST_SECOND, GST_AUDIO_INFO_RATE(&element->audioinfo));
	GST_BUFFER_DURATION(buf) = element->segment.position - GST_BUFFER_TIMESTAMP(buf);

	GST_DEBUG_OBJECT(element, "%" GST_BUFFER_BOUNDARIES_FORMAT, GST_BUFFER_BOUNDARIES_ARGS(buf));

done:
	gst_buffer_unmap(buf, &mapinfo);
	/*PyGILState_Release(gilstate);*/
	return result;
}


/*
 * do_seek()
 */


static gboolean do_seek(GstBaseSrc *basesrc, GstSegment *segment)
{
	GstLALPyFuncSrc *element = GSTLAL_PYFUNCSRC(basesrc);
	gboolean success = TRUE;

	GST_DEBUG_OBJECT(element, "requested segment is [%" GST_TIME_SECONDS_FORMAT ", %" GST_TIME_SECONDS_FORMAT "), stream time %" GST_TIME_SECONDS_FORMAT ", position %" GST_TIME_SECONDS_FORMAT ", duration %" GST_TIME_SECONDS_FORMAT, GST_TIME_SECONDS_ARGS(segment->start), GST_TIME_SECONDS_ARGS(segment->stop), GST_TIME_SECONDS_ARGS(segment->time), GST_TIME_SECONDS_ARGS(segment->position), GST_TIME_SECONDS_ARGS(segment->duration));

	/*
	 * do the seek
	 */

	gst_segment_copy_into(segment, &element->segment);
	element->offset = element->segment.start + gst_util_uint64_scale_int_round(element->segment.position - element->segment.start, GST_AUDIO_INFO_RATE(&element->audioinfo), GST_SECOND);

	/*
	 * done
	 */

	return success;
}


/*
 * query()
 */


static gboolean query(GstBaseSrc *basesrc, GstQuery *query)
{
	GstLALPyFuncSrc *element = GSTLAL_PYFUNCSRC(basesrc);
	gboolean success = TRUE;

	GST_DEBUG_OBJECT(element, "%" GST_PTR_FORMAT, query);

	switch(GST_QUERY_TYPE(query)) {
	case GST_QUERY_FORMATS:
		gst_query_set_formats(query, 1, GST_FORMAT_TIME);
		break;

	case GST_QUERY_POSITION:
		gst_query_set_position(query, GST_FORMAT_TIME, element->segment.position);
		break;

	default:
		success = GST_BASE_SRC_CLASS(gstlal_pyfuncsrc_parent_class)->query(basesrc, query);
		break;
	}

	if(success)
		GST_DEBUG_OBJECT(element, "result: %" GST_PTR_FORMAT, query);
	else
		GST_WARNING_OBJECT(element, "query failed");
	return success;
}


/*
 * ============================================================================
 *
 *                              GObject Methods
 *
 * ============================================================================
 */


enum property {
	PROP_EXPRESSION = 1,
};


static void set_property(GObject *object, enum property id, const GValue *value, GParamSpec *pspec)
{
	GstLALPyFuncSrc *element = GSTLAL_PYFUNCSRC(object);

	GST_OBJECT_LOCK(object);

	switch(id) {
	case PROP_EXPRESSION: {
		/*PyGILState_STATE gilstate = PyGILState_Ensure();*/

		g_free(element->expression);
		element->expression = g_value_dup_string(value);

		Py_XDECREF(element->code);
		element->code = (PyCodeObject *) Py_CompileString(element->expression, "lal_pyfuncsrc", Py_eval_input);
		if(!element->code)
			PyErr_Print();

		gst_pad_mark_reconfigure(GST_BASE_SRC_PAD(GST_BASE_SRC(object)));

		/*PyGILState_Release(gilstate);*/
		break;
	}

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(object);
}


static void get_property(GObject *object, enum property id, GValue *value, GParamSpec *pspec)
{
	GstLALPyFuncSrc *element = GSTLAL_PYFUNCSRC(object);

	GST_OBJECT_LOCK(object);

	switch(id) {
	case PROP_EXPRESSION:
		g_value_set_string(value, element->expression);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(object);
}


static void finalize(GObject *object)
{
	GstLALPyFuncSrc *element = GSTLAL_PYFUNCSRC(object);
	/*PyGILState_STATE gilstate = PyGILState_Ensure();*/

	g_free(element->expression);
	element->expression = NULL;
	Py_XDECREF(element->code);
	element->code = NULL;
	Py_DECREF(element->globals);
	element->globals = NULL;

	/*PyGILState_Release(gilstate);*/

	G_OBJECT_CLASS(gstlal_pyfuncsrc_parent_class)->finalize(object);
}


static void gstlal_pyfuncsrc_class_init(GstLALPyFuncSrcClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GstBaseSrcClass *gstbasesrc_class = GST_BASE_SRC_CLASS(klass);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	gstbasesrc_class->get_caps = GST_DEBUG_FUNCPTR(get_caps);
	gstbasesrc_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
	gstbasesrc_class->is_seekable = GST_DEBUG_FUNCPTR(is_seekable);
	gstbasesrc_class->fill = GST_DEBUG_FUNCPTR(fill);
	gstbasesrc_class->do_seek = GST_DEBUG_FUNCPTR(do_seek);
	gstbasesrc_class->query = GST_DEBUG_FUNCPTR(query);

	gst_element_class_set_details_simple(
		element_class,
		"Python Function Source",
		"Source",
		"Generate a time series by repeatedly evaluating a Python expression.",
		"Kipp Cannon <kipp.cannon@ligo.org>"
	);

	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"src",
			GST_PAD_SRC,
			GST_PAD_ALWAYS,
			gst_caps_from_string(
				GST_AUDIO_CAPS_MAKE("{ " GST_AUDIO_NE(F64) ", " GST_AUDIO_NE(Z128) " }") ", " \
				"layout = (string) interleaved, " \
				"channel-mask = (bitmask) 0"
			)
		)
	);

	g_object_class_install_property(
		gobject_class,
		PROP_EXPRESSION,
		g_param_spec_string(
			"expression",
			"Expression",
			"Expression to evaluate.  The namespace will include \"from numpy input *\"\n\t\t\t"
			"and a variable \"t\" containing the current time.",
			DEFAULT_EXPRESSION,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);

/* FIXME:  WTF? */
	dlopen("libpython2.7.so", RTLD_LAZY | RTLD_GLOBAL);
	Py_Initialize();
	PyEval_InitThreads();
}


static void gstlal_pyfuncsrc_init(GstLALPyFuncSrc *element)
{
	/*PyGILState_STATE gilstate = PyGILState_Ensure();*/
	PyObject *numpy;

	gst_base_src_set_format(GST_BASE_SRC(element), GST_FORMAT_TIME);

	element->expression = NULL;
	element->code = NULL;

	numpy = PyImport_ImportModule("numpy");
	if(!numpy) {
		PyErr_Print();
		element->globals = Py_None;
		Py_INCREF(Py_None);
	} else {
		/* _GetDict() returns borrowed reference */
		element->globals = PyModule_GetDict(numpy);
		if(!element->globals) {
			PyErr_Print();
			element->globals = Py_None;
		}
		Py_INCREF(element->globals);
		Py_DECREF(numpy);
	}

	/*PyGILState_Release(gilstate);*/

	gst_segment_init(&element->segment, GST_FORMAT_TIME);
	element->offset = 0;
}

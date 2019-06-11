/*
 * Copyright (C) 2010-2013,2015,2016  Kipp Cannon
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation; either version 2 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */


/*
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>
#include <lal/Date.h>


#include <snglburstrowtype.h>


static PyObject *LIGOTimeGPSType = NULL;


/*
 * ============================================================================
 *
 *                                    Type
 *
 * ============================================================================
 */


typedef struct {
	PyObject_HEAD
	SnglBurst row;
} gstlal_GSTLALSnglBurst;


/*
 * Member access
 */


static struct PyMemberDef members[] = {
	{"start_time", T_INT, offsetof(gstlal_GSTLALSnglBurst, row.start_time.gpsSeconds), 0, "start_time"},
	{"start_time_ns", T_INT, offsetof(gstlal_GSTLALSnglBurst, row.start_time.gpsNanoSeconds), 0, "start_time_ns"},
	{"peak_time", T_INT, offsetof(gstlal_GSTLALSnglBurst, row.peak_time.gpsSeconds), 0, "peak_time"},
	{"peak_time_ns", T_INT, offsetof(gstlal_GSTLALSnglBurst, row.peak_time.gpsNanoSeconds), 0, "peak_time_ns"},
	{"duration", T_FLOAT, offsetof(gstlal_GSTLALSnglBurst, row.duration), 0, "event_duration"},
	{"central_freq", T_FLOAT, offsetof(gstlal_GSTLALSnglBurst, row.central_freq), 0, "central frequency"},
	{"bandwidth", T_FLOAT, offsetof(gstlal_GSTLALSnglBurst, row.bandwidth), 0, "bandwidth"},
	{"amplitude", T_FLOAT, offsetof(gstlal_GSTLALSnglBurst, row.amplitude), 0, "amplitude"},
	{"snr", T_FLOAT, offsetof(gstlal_GSTLALSnglBurst, row.snr), 0, "snr"},
	{"confidence", T_FLOAT, offsetof(gstlal_GSTLALSnglBurst, row.confidence), 0, "confidence"},
	{"chisq", T_DOUBLE, offsetof(gstlal_GSTLALSnglBurst, row.chisq), 0, "chisq"},
	{"chisq_dof", T_DOUBLE, offsetof(gstlal_GSTLALSnglBurst, row.chisq_dof), 0, "chisq_dof"},
	{"process_id", T_LONG, offsetof(gstlal_GSTLALSnglBurst, row.process_id), 0, "process_id (long)"},
	{"event_id", T_LONG, offsetof(gstlal_GSTLALSnglBurst, row.event_id), 0, "event_id (long)"},
	{NULL,}
};


struct pylal_inline_string_description {
	Py_ssize_t offset;
	Py_ssize_t length;
};


static PyObject *pylal_inline_string_get(PyObject *obj, void *data)
{
	const struct pylal_inline_string_description *desc = data;
	char *s = (void *) obj + desc->offset;

	if((ssize_t) strlen(s) >= desc->length) {
		/* something's wrong, obj probably isn't a valid address */
	}

	return PyString_FromString(s);
}


static int pylal_inline_string_set(PyObject *obj, PyObject *val, void *data)
{
	const struct pylal_inline_string_description *desc = data;
	char *v = PyString_AsString(val);
	char *s = (void *) obj + desc->offset;

	if(!v)
		return -1;
	if((ssize_t) strlen(v) >= desc->length) {
		PyErr_Format(PyExc_ValueError, "string too long \'%s\'", v);
		return -1;
	}

	strncpy(s, v, desc->length - 1);
	s[desc->length - 1] = '\0';

	return 0;
}


static struct PyGetSetDef getset[] = {
	{"ifo", pylal_inline_string_get, pylal_inline_string_set, "ifo", &(struct pylal_inline_string_description) {offsetof(gstlal_GSTLALSnglBurst, row.ifo), LIGOMETA_IFO_MAX}},
	{"search", pylal_inline_string_get, pylal_inline_string_set, "search", &(struct pylal_inline_string_description) {offsetof(gstlal_GSTLALSnglBurst, row.search), LIGOMETA_SEARCH_MAX}},
	{"channel", pylal_inline_string_get, pylal_inline_string_set, "channel", &(struct pylal_inline_string_description) {offsetof(gstlal_GSTLALSnglBurst, row.channel), LIGOMETA_CHANNEL_MAX}},
	{NULL,}
};


/*
 * Methods
 */


static PyObject *__new__(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	gstlal_GSTLALSnglBurst *new = (gstlal_GSTLALSnglBurst *) PyType_GenericNew(type, args, kwds);

	if(!new)
		return NULL;

	/* done */
	return (PyObject *) new;
}


static void __del__(PyObject *self)
{
	Py_TYPE(self)->tp_free(self);
}


static PyObject *from_buffer(PyObject *cls, PyObject *args)
{
	const char *data;
	Py_ssize_t length;
	PyObject *result;

	if(!PyArg_ParseTuple(args, "s#", (const char **) &data, &length))
		return NULL;
	const char *const end = data + length;

	result = PyList_New(0);
	if(!result)
		return NULL;
	while (data < end) {
		PyObject *item = PyType_GenericNew((PyTypeObject *) cls, NULL, NULL);
		if(!item) {
			Py_DECREF(result);
			return NULL;
		}
		/* memcpy sngl_burst row */ /*FIXME this should be done in a much simpler way for Burst?*/
		const struct GSTLALSnglBurst *gstlal_snglburst = (const struct GSTLALSnglBurst *) data;
		data += sizeof(*gstlal_snglburst);
		if (data > end)
		{
			Py_DECREF(item);
			Py_DECREF(result);
			PyErr_SetString(PyExc_ValueError, "buffer overrun while copying sngl_burst row");
			return NULL;
		}
		((gstlal_GSTLALSnglBurst*)item)->row = gstlal_snglburst->parent;

		PyList_Append(result, item);
		Py_DECREF(item);
	}

	if (data != end)
	{
		Py_DECREF(result);
		PyErr_SetString(PyExc_ValueError, "did not consume entire buffer");
		return NULL;
	}

	PyObject *tuple = PyList_AsTuple(result);
	Py_DECREF(result);
	return tuple;
}


static struct PyMethodDef methods[] = {
	{"from_buffer", from_buffer, METH_VARARGS | METH_CLASS, "Construct a tuple of GSTLALSnglBurst objects from a buffer object.  The buffer is interpreted as a C array of GSTLALSnglBurst structures.  All data is copied, the buffer can be deallocated afterwards."},
	{NULL,}
};


/*
 * comparison is defined specifically for the coincidence code, allowing a
 * bisection search of a sorted trigger list to be used to identify the
 * subset of triggers that fall within a time interval
 */


static PyObject *richcompare(PyObject *self, PyObject *other, int op_id)
{
	PyObject *converted = PyObject_CallFunctionObjArgs(LIGOTimeGPSType, other, NULL);
	PyObject *attr;
	PyObject *result;
	LIGOTimeGPS t_other;
	int cmp;

	if(!converted)
		return NULL;

	attr = PyObject_GetAttrString(converted, "gpsSeconds");
	if(!attr) {
		Py_DECREF(converted);
		return NULL;
	}
	t_other.gpsSeconds = PyInt_AsLong(attr);
	Py_DECREF(attr);
	attr = PyObject_GetAttrString(converted, "gpsNanoSeconds");
	if(!attr) {
		Py_DECREF(converted);
		return NULL;
	}
	t_other.gpsNanoSeconds = PyInt_AsLong(attr);
	Py_DECREF(attr);
	Py_DECREF(converted);

	cmp = XLALGPSCmp(&((gstlal_GSTLALSnglBurst *) self)->row.peak_time, &t_other);

	switch(op_id) {
	case Py_LT:
		result = (cmp < 0) ? Py_True : Py_False;
		break;

	case Py_LE:
		result = (cmp <= 0) ? Py_True : Py_False;
		break;

	case Py_EQ:
		result = (cmp == 0) ? Py_True : Py_False;
		break;

	case Py_NE:
		result = (cmp != 0) ? Py_True : Py_False;
		break;

	case Py_GE:
		result = (cmp >= 0) ? Py_True : Py_False;
		break;

	case Py_GT:
		result = (cmp > 0) ? Py_True : Py_False;
		break;

	default:
		PyErr_BadInternalCall();
		return NULL;
	}

	Py_INCREF(result);
	return result;
}


/*
 * Type
 */


static PyTypeObject gstlal_GSTLALSnglBurst_Type = {
	PyObject_HEAD_INIT(NULL)
	.tp_basicsize = sizeof(gstlal_GSTLALSnglBurst),
	.tp_doc = "GstLAL's GSTLALSnglBurst type",
	.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_CHECKTYPES,
	.tp_members = members,
	.tp_methods = methods,
	.tp_getset = getset,
	.tp_name = MODULE_NAME ".GSTLALSnglBurst",
	.tp_new = __new__,
	.tp_dealloc = __del__,
	.tp_richcompare = richcompare,
};


/*
 * ============================================================================
 *
 *                            Module Registration
 *
 * ============================================================================
 */


PyMODINIT_FUNC init_snglbursttable(void)
{
	PyObject *module = Py_InitModule3(MODULE_NAME, NULL, "Low-level wrapper for GSTLALSnglBurst type.");

	/* LIGOTimeGPS */

	{
	PyObject *lal = PyImport_ImportModule("lal");
	if(!lal)
		return;
	LIGOTimeGPSType = PyDict_GetItemString(PyModule_GetDict(lal), "LIGOTimeGPS");
	if(!LIGOTimeGPSType) {
		Py_DECREF(lal);
		return;
	}
	Py_INCREF(LIGOTimeGPSType);
	Py_DECREF(lal);
	}

	/* SnglBurst */
	if(PyType_Ready(&gstlal_GSTLALSnglBurst_Type) < 0)
		return;
	Py_INCREF(&gstlal_GSTLALSnglBurst_Type);
	PyModule_AddObject(module, "GSTLALSnglBurst", (PyObject *) &gstlal_GSTLALSnglBurst_Type);
}

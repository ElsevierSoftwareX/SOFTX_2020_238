/*
 * Copyright (C) 2013  Kipp Cannon
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


#include <Python.h>
#include <gstfrhistory.h>


#define MODULE_NAME "gstlal._gstfrhistory"


/*
 * ============================================================================
 *
 *                                    Type
 *
 * ============================================================================
 */


/*
 * Structure
 */


typedef struct {
	PyObject_HEAD
	GstFrHistory *history;
} PyGstFrHistory;


static PyTypeObject PyGstFrHistory_Type;


/*
 * Member access
 */


static PyObject *get_name(PyObject *obj, void *data)
{
	const char *str = gst_frhistory_get_name(((PyGstFrHistory *) obj)->history);
	if(str)
		return PyString_FromString(str);
	Py_INCREF(Py_None);
	return Py_None;
}


static PyObject *get_comment(PyObject *obj, void *data)
{
	const char *str = gst_frhistory_get_comment(((PyGstFrHistory *) obj)->history);
	if(str)
		return PyString_FromString(str);
	Py_INCREF(Py_None);
	return Py_None;
}


static int set_comment(PyObject *obj, PyObject *val, void *data)
{
	gst_frhistory_set_comment(((PyGstFrHistory *) obj)->history, PyString_AsString(val));
	return 0;
}


static PyObject *get_timestamp(PyObject *obj, void *data)
{
	return PyLong_FromLong(gst_frhistory_get_timestamp(((PyGstFrHistory *) obj)->history));
}


static int set_timestamp(PyObject *obj, PyObject *val, void *data)
{
	gst_frhistory_set_timestamp(((PyGstFrHistory *) obj)->history, PyLong_AsLong(val));
	return 0;
}


static struct PyGetSetDef getset[] = {
	{"name", get_name, NULL, "name (string or None, read-only)", NULL},
	{"comment", get_comment, set_comment, "comment (string or None)", NULL},
	{"timestamp", get_timestamp, set_timestamp, "timestamp (long)", NULL},
	{NULL,}
};


/*
 * Methods
 */


static PyObject *__new__(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	/* call the generic __new__() */
	PyGstFrHistory *new = (PyGstFrHistory *) PyType_GenericNew(type, args, kwds);
	char *name;

	if(!new || !PyArg_ParseTuple(args, "z:GstFrHistory", &name)) {
		Py_XDECREF(new);
		return NULL;
	}

	new->history = gst_frhistory_new(name);

	/* done */
	return (PyObject *) new;
}


static void __del__(PyObject *obj)
{
	gst_frhistory_free(((PyGstFrHistory *) obj)->history);
	((PyGstFrHistory *) obj)->history = NULL;

	PyGstFrHistory_Type.tp_free(obj);
}


static PyObject *__str__(PyObject *obj)
{
	char *str = gst_frhistory_to_string(((PyGstFrHistory *) obj)->history);
	PyObject *result = PyString_FromString(str);
	free(str);
	return result;
}


/*
 * Type
 */


static PyTypeObject PyGstFrHistory_Type = {
	PyObject_HEAD_INIT(NULL)
	.tp_basicsize = sizeof(PyGstFrHistory),
	.tp_doc = "gstlal.GstFrHistory",
	.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_CHECKTYPES,
	.tp_getset = getset,
	.tp_name = MODULE_NAME ".GstFrHistory",
	.tp_new = __new__,
	.tp_dealloc = __del__,
	.tp_str = __str__,
};


/*
 * ============================================================================
 *
 *                            Module Registration
 *
 * ============================================================================
 */


static struct PyMethodDef functions[] = {
	{NULL,}
};


PyMODINIT_FUNC init_gstfrhistory(void)
{
	PyObject *module = Py_InitModule3(MODULE_NAME, functions, "Wrapper for GstFrHistory type.");

	if(PyType_Ready(&PyGstFrHistory_Type) < 0)
		return;
	Py_INCREF((PyObject *) &PyGstFrHistory_Type);
	PyModule_AddObject(module, "GstFrHistory", (PyObject *) &PyGstFrHistory_Type);
}

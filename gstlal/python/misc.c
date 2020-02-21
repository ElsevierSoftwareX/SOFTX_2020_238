/*
 * Copyright (C) 2010,2011 Kipp Cannon <kipp.cannon@ligo.org>
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


#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <stdlib.h>


#include <gstlal/gstlal_cdf_weighted_chisq_P.h>


#define MODULE_NAME "gstlal._misc"


/*
 * ============================================================================
 *
 *                                 Utilities
 *
 * ============================================================================
 */


/*
 * gets a 1-D continuous PyArrayObject from a PyArrayObject
 */


static PyArrayObject *get_continuous_1d_pyarray(PyArrayObject *input, int type)
{
	if(!PyArray_ISCARRAY(input)) {
		PyErr_SetObject(PyExc_TypeError, (PyObject *) input);
		return NULL;
	}
	return (PyArrayObject *) PyArray_ContiguousFromAny((PyObject *) input, type, 1, 1);
}


/*
 * ============================================================================
 *
 *                                 Functions
 *
 * ============================================================================
 */


static PyObject *gstlal_cdf_weighted_chisq_P_wrapper(PyObject *self, PyObject *args)
{
	PyArrayObject *A, *noncent, *dof;
	int *dof_local;
	double var, c, accuracy, result;
	int N, lim;
	int i;

	if(!PyArg_ParseTuple(args, "O!O!O!ddid", &PyArray_Type, &A, &PyArray_Type, &noncent, &PyArray_Type, &dof, &var, &c, &lim, &accuracy))
		return NULL;
	A = get_continuous_1d_pyarray(A, NPY_DOUBLE);
	noncent = get_continuous_1d_pyarray(noncent, NPY_DOUBLE);
	dof = get_continuous_1d_pyarray(dof, NPY_LONG);
	if(!A || !noncent || !dof) {
		Py_XDECREF(A); Py_XDECREF(noncent); Py_XDECREF(dof);
		return NULL;
	}

	N = PyArray_SIZE(A);
	if(N != PyArray_SIZE(noncent) || N != PyArray_SIZE(dof)) {
		PyErr_SetString(PyExc_ValueError, "array size mismatch");
		Py_DECREF(A); Py_DECREF(noncent); Py_DECREF(dof);
		return NULL;
	}

	dof_local = malloc(N * sizeof(*dof_local));
	if(!dof_local) {
		Py_DECREF(A); Py_DECREF(noncent); Py_DECREF(dof);
		return PyErr_NoMemory();
	}
	for(i = 0; i < N; i++)
		dof_local[i] = ((long *) PyArray_DATA(dof))[i];

	result = gstlal_cdf_weighted_chisq_P((double *) PyArray_DATA(A), (double *) PyArray_DATA(noncent), dof_local, N, var, c, lim, accuracy, NULL, NULL);

	free(dof_local);
	Py_DECREF(A); Py_DECREF(noncent); Py_DECREF(dof);

	return PyFloat_FromDouble(result);
}


/*
 * ============================================================================
 *
 *                                Entry Point
 *
 * ============================================================================
 */


static struct PyMethodDef methods[] = {
	{"cdf_weighted_chisq_P", gstlal_cdf_weighted_chisq_P_wrapper, METH_VARARGS, NULL},
	{NULL, }
};


#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC init_misc(void); /* Silence -Wmissing-prototypes */
PyMODINIT_FUNC init_misc(void)
#else
PyMODINIT_FUNC PyInit__misc(void); /* Silence -Wmissing-prototypes */
PyMODINIT_FUNC PyInit__misc(void)
#endif
{
#if PY_MAJOR_VERSION < 3
	(void) Py_InitModule(MODULE_NAME, methods);
#else
	static struct PyModuleDef modef = {
		PyModuleDef_HEAD_INIT,
		.m_name = MODULE_NAME,
		.m_size = -1,
		.m_methods = methods,
	};
	PyObject *module = PyModule_Create(&modef);
#endif

	import_array();

#if PY_MAJOR_VERSION < 3
	return;
#else
	return module;
#endif
}

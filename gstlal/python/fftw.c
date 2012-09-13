/*
 * Copyright (C) 2012 Kipp Cannon <kipp.cannon@ligo.org>
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
#include <stdlib.h>


#include <gstlal.h>


/*
 * ============================================================================
 *
 *                                 Functions
 *
 * ============================================================================
 */


static PyObject *lock(PyObject *self, PyObject *args)
{
	gstlal_fftw_lock();

	Py_INCREF(Py_None);
	return Py_None;
}


static PyObject *unlock(PyObject *self, PyObject *args)
{
	gstlal_fftw_unlock();

	Py_INCREF(Py_None);
	return Py_None;
}


/*
 * ============================================================================
 *
 *                                Entry Point
 *
 * ============================================================================
 */


static struct PyMethodDef methods[] = {
	{"lock", lock, METH_NOARGS, NULL},
	{"unlock", unlock, METH_NOARGS, NULL},
	{NULL, }
};


void initfftw(void)
{
	/*PyObject *module =*/ Py_InitModule("gstlal.fftw", methods);
}

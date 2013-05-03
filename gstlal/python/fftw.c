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


#include <lal/LALConfig.h>	/* only needed for LAL_PTHREAD_LOCK */
#include <gstlal.h>


/*
 * ============================================================================
 *
 *                                 Functions
 *
 * ============================================================================
 */


/*
 * FIXME:  these functions are only here so that Python code that calls LAL
 * FFT plan creation functions can lock the wisdom.  if LAL is compiled
 * with locking enabled, then the use of these functions in python code
 * outside of the plan creation functions would deadlock, therefore, if LAL
 * is compiled with locking enabled THESE FUNCTIONS ARE NO-OPs.  don't rely
 * on them in python code to actually acquire and release the gstlal fftw
 * wisdom lock.  that's not what they're for.  delete them when LAL can be
 * trusted to have locking enabled.
 */


static PyObject *lock(PyObject *self, PyObject *args)
{
#ifndef LAL_PTHREAD_LOCK
	gstlal_fftw_lock();
#endif

	Py_INCREF(Py_None);
	return Py_None;
}


static PyObject *unlock(PyObject *self, PyObject *args)
{
#ifndef LAL_PTHREAD_LOCK
	gstlal_fftw_unlock();
#endif

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

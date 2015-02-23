/*
 * Copyright (C) 2014  Kipp C. Cannon
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


#include <math.h>
#include <stdlib.h>


#include <Python.h>
#include <numpy/arrayobject.h>


/*
 * ============================================================================
 *
 *                               Internal Code
 *
 * ============================================================================
 */


/*
 * input conditioning.  sort ln_f_over_b array in ascending order.
 */


static int conditioning_compare(const void *a, const void *b)
{
	double A = *(double *) a, B = *(double *) b;

	return A > B ? +1 : A < B ? -1 : 0;
}


static void condition(double *ln_f_over_b, int n)
{
	qsort(ln_f_over_b, n, sizeof(*ln_f_over_b), conditioning_compare);
}


/*
 * compute the log probability density of the foreground and background
 * rates given by equation (21) in Farr et al., "Counting and Confusion:
 * Bayesian Rate Estimation With Multiple Populations", arXiv:1302.5341.
 * the prior is that specified in the paper.
 */


static double log_prior(double Rf, double Rb)
{
	return -0.5 * log(Rf * Rb);
}


static double log_posterior(const double *ln_f_over_b, int n, double Rf, double Rb)
{
	double ln_Rf_over_Rb = log(Rf / Rb);
	int i;
	double ln_P = 0.;

	if(Rf < 0. || Rb < 0.)
		return atof("-inf");

	#pragma omp parallel for reduction(+:ln_P)
	for(i = 0; i < n; i++) {
		/*
		 * need to add log(Rf f / (Rb b) + 1) to sum.  if x = Rf f
		 * / (Rb b) is large, we approximate ln(x + 1) with
		 *
		 *	ln(x + 1) = ln x + 1 / x
		 *
		 * for x > 1000 this gives 7 correct digits or more.  for x
		 * > 10^15, the 1/x correction becomes irrelevant.
		 */

		double ln_x = ln_Rf_over_Rb + ln_f_over_b[i];
		if(ln_x > 35.)	/* ~= log(10^15) */
			ln_P += ln_x;
		else if(ln_x > 6.9)	/* ~= log(1000) */
			ln_P += ln_x + exp(-ln_x);
		else
			ln_P += log1p(exp(ln_x));
	}
	ln_P += n * log(Rb) - (Rf + Rb);

	return ln_P + log_prior(Rf, Rb);
}


/*
 * ============================================================================
 *
 *                    Rate Estimation --- Posterior Class
 *
 * ============================================================================
 */


/*
 * Structure
 */


struct posterior {
	PyObject_HEAD

	/*
	 * array of P(L | signal) / P(L | noise) for the L's of all events
	 * in the experiment's results
	 */

	double *ln_f_over_b;
	int ln_f_over_b_len;
};


/*
 * __del__() method
 */


static void __del__(PyObject *self)
{
	struct posterior *posterior = (struct posterior *) self;

	free(posterior->ln_f_over_b);
	posterior->ln_f_over_b = NULL;
	posterior->ln_f_over_b_len = 0;

	self->ob_type->tp_free(self);
}


/*
 * __init__() method
 */


static int __init__(PyObject *self, PyObject *args, PyObject *kwds)
{
	struct posterior *posterior = (struct posterior *) self;
	PyArrayObject *arr;
	int i;

	if(!PyArg_ParseTuple(args, "O!", &PyArray_Type, &arr))
		return -1;

	if(PyArray_NDIM(arr) != 1) {
		PyErr_SetString(PyExc_ValueError, "wrong number of dimensions");
		return -1;
	}
	if(!PyArray_ISFLOAT(arr) || !PyArray_ISPYTHON(arr)) {
		PyErr_SetObject(PyExc_TypeError, (PyObject *) arr);
		return -1;
	}

	posterior->ln_f_over_b_len = *PyArray_DIMS(arr);
	posterior->ln_f_over_b = malloc(posterior->ln_f_over_b_len * sizeof(*posterior->ln_f_over_b));

	for(i = 0; i < posterior->ln_f_over_b_len; i++)
		posterior->ln_f_over_b[i] = *(double *) PyArray_GETPTR1(arr, i);

	condition(posterior->ln_f_over_b, posterior->ln_f_over_b_len);

	return 0;
}


/*
 * __call__() method
 */


static PyObject *__call__(PyObject *self, PyObject *args, PyObject *kw)
{
	struct posterior *posterior = (struct posterior *) self;
	double Rf, Rb;

	if(kw) {
		PyErr_SetString(PyExc_ValueError, "unexpected keyword arguments");
		return NULL;
	}
	if(!PyArg_ParseTuple(args, "(dd)", &Rf, &Rb))
		return NULL;

	/*
	 * return log_posterior() / 2.  to improve the measurement of the
	 * tails of the PDF using the MCMC sampler, we draw from the square
	 * root of the PDF and then correct the histogram of the samples.
	 */

	return PyFloat_FromDouble(log_posterior(posterior->ln_f_over_b, posterior->ln_f_over_b_len, Rf, Rb) / 2.);
}


/*
 * Type information
 */


static PyTypeObject posterior_Type = {
	PyObject_HEAD_INIT(NULL)
	.tp_basicsize = sizeof(struct posterior),
	.tp_call = __call__,
	.tp_dealloc = __del__,
	.tp_doc = "",
	.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_CHECKTYPES,
	.tp_init = __init__,
	.tp_name = MODULE_NAME ".posterior",
	.tp_new = PyType_GenericNew,
};


/*
 * ============================================================================
 *
 *                                Entry Point
 *
 * ============================================================================
 */


void init_rate_estimation(void)
{
	PyObject *module = Py_InitModule3(MODULE_NAME, NULL, "");

	import_array();

	if(PyType_Ready(&posterior_Type) < 0)
		return;
	Py_INCREF(&posterior_Type);
	PyModule_AddObject(module, "posterior", (PyObject *) &posterior_Type);
}

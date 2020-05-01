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

/**
 * SECTION:rate_estimation.c
 * @short_description: C code for event rate intervals
 * 
 * Reviewed: 2fb185eda0edb9d49d79b8185f7b35457cafa06b 2015-05-14 
 * K. Cannon, J. Creighton, C. Hanna, F. Robinett
 *
 * Actions:
 *  - Improve comments (some out of date)
 *
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


#include <gsl/gsl_integration.h>


#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>


#define GSL_WORKSPACE_SIZE  8


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
 * the natural logarithm (up to an unknown additive constant) of the prior.
 * see equation (17) of Farr et al., "Counting and Confusion: Bayesian Rate
 * Estimation With Multiple Populations", arXiv:1302.5341.
 */


static double log_prior(double Rf, double Rb)
{
	return -0.5 * log(Rf * Rb);
}


/*
 * compute the logarithm (up to an unknown additive constant) of the joint
 * probability density of the foreground and background rates given by
 * equation (21) in Farr et al., "Counting and Confusion: Bayesian Rate
 * Estimation With Multiple Populations", arXiv:1302.5341.
 */


struct integrand_data_t {
	const double *ln_f_over_b;
	double ln_Rf_over_Rb;
};


static double integrand(double i, void *data)
{
	const struct integrand_data_t *integrand_data = data;
	double ln_x = integrand_data->ln_Rf_over_Rb + integrand_data->ln_f_over_b[(int) floor(i)];
	if(ln_x > 33.)	/* x ~= 10^14 */
		return ln_x;
	return log1p(exp(ln_x));
}


static double log_posterior(const double *ln_f_over_b, int n, double Rf, double Rb, gsl_integration_cquad_workspace *workspace)
{
	double ln_Rf_over_Rb = log(Rf / Rb);
	int i;
	double ln_P = 0.;

	if(Rf < 0. || Rb < 0.)
		return atof("-inf");

	/*
	 * need to compute sum of log(Rf f / (Rb b) + 1).  if x = Rf f /
	 * (Rb b) is larger than about 10^14 we consider the +1 to be
	 * irrelevant and approximate ln(x + 1) with ln(x).  for smaller x
	 * we use log1p(x) to evaluate the exprsesion.
	 *
	 * experience shows that the array of f/b values contains many very
	 * similar entries at the lower end of its range, so we sort the
	 * array and by treating the array as a function of its index use a
	 * numerical integration scheme to obtain the sum.  we do this for
	 * the bottom 99% of the array, and the top 1% is treated
	 * explicitly to capture the more rapid sample-to-sample variation.
	 */

	/*
	 * first do the numerical interal for the bottom part of the array
	 */

	i = 0.99 * n;
	if(i) {
		/*
		 * for these entries, compute the sum of log(Rf f / (Rb b)
		 * + 1) by approximating it with a numerical integration of
		 * the addend (evaluated using the approximations described
		 * above)
		 */
		gsl_function _integrand = {
			.function = integrand,
			.params = &(struct integrand_data_t) {
				.ln_f_over_b = ln_f_over_b,
				.ln_Rf_over_Rb = ln_Rf_over_Rb
			}
		};
		gsl_integration_cquad(&_integrand, 0, i * (1. - 1e-16), 0., 1e-8, workspace, &ln_P, NULL, NULL);
	}

	/*
	 * now explicitly compute the sum of log(Rf f / (Rb b) + 1) for the
	 * remaining entries
	 */

	for(; i < n; i++) {
		double ln_x = ln_Rf_over_Rb + ln_f_over_b[i];
		if(ln_x > 33.)	/* x ~= log(10^14) */
			ln_P += ln_x;
		else
			ln_P += log1p(exp(ln_x));
	}

	/*
	 * multiply by the remaining factors
	 */

	ln_P += n * log(Rb) - (Rf + Rb);

	/*
	 * finally multiply by the prior
	 */

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


struct LogPosterior {
	PyObject_HEAD

	/*
	 * array of P(L | signal) / P(L | noise) for the L's of all events
	 * in the experiment's results
	 */

	double *ln_f_over_b;
	int ln_f_over_b_len;
	gsl_integration_cquad_workspace *workspace;
};


/*
 * __del__() method
 */


static void __del__(PyObject *self)
{
	struct LogPosterior *posterior = (struct LogPosterior *) self;

	free(posterior->ln_f_over_b);
	posterior->ln_f_over_b = NULL;
	posterior->ln_f_over_b_len = 0;
	gsl_integration_cquad_workspace_free(posterior->workspace);
	posterior->workspace = NULL;

	self->ob_type->tp_free(self);
}


/*
 * __init__() method
 */


static int __init__(PyObject *self, PyObject *args, PyObject *kwds)
{
	struct LogPosterior *posterior = (struct LogPosterior *) self;
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

	posterior->workspace = gsl_integration_cquad_workspace_alloc(GSL_WORKSPACE_SIZE);

	return 0;
}


/*
 * __call__() method
 */


static PyObject *__call__(PyObject *self, PyObject *args, PyObject *kw)
{
	struct LogPosterior *posterior = (struct LogPosterior *) self;
	double Rf, Rb;

	if(kw) {
		PyErr_SetString(PyExc_ValueError, "unexpected keyword arguments");
		return NULL;
	}
	if(!PyArg_ParseTuple(args, "(dd)", &Rf, &Rb))
		return NULL;

	/*
	 * return log_posterior()
	 */

	return PyFloat_FromDouble(log_posterior(posterior->ln_f_over_b, posterior->ln_f_over_b_len, Rf, Rb, posterior->workspace));
}


/*
 * Type information
 */


static PyTypeObject LogPosterior_Type = {
	PyObject_HEAD_INIT(NULL)
	.tp_basicsize = sizeof(struct LogPosterior),
	.tp_call = __call__,
	.tp_dealloc = __del__,
	.tp_doc = "",
	.tp_flags = Py_TPFLAGS_DEFAULT,
	.tp_init = __init__,
	.tp_name = MODULE_NAME ".LogPosterior",
	.tp_new = PyType_GenericNew,
};


/*
 * ============================================================================
 *
 *                                Entry Point
 *
 * ============================================================================
 */

static struct PyModuleDef RateEstimationModule = {
	PyModuleDef_HEAD_INIT,
	MODULE_NAME,
	"",
	-1,
	NULL
};

PyMODINIT_FUNC PyInit__rate_estimation(void)
{
	PyObject *module = PyModule_Create(&RateEstimationModule);

	import_array();

	if(PyType_Ready(&LogPosterior_Type) < 0)
		return;
	Py_INCREF((PyObject *) &LogPosterior_Type);
	PyModule_AddObject(module, "LogPosterior", (PyObject *) &LogPosterior_Type);

	return module;
}

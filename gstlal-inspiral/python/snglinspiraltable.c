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
#include <numpy/ndarrayobject.h>
#include <structmember.h>
#include <lal/Date.h>
#include <lal/TimeSeries.h>
#include <lal/Units.h>


#include <snglinspiralrowtype.h>
#include <gst/gst.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix_float.h>


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
	SnglInspiralTable row;
	COMPLEX8TimeSeries *G1_snr;
	COMPLEX8TimeSeries *H1_snr;
	COMPLEX8TimeSeries *K1_snr;
	COMPLEX8TimeSeries *L1_snr;
	COMPLEX8TimeSeries *V1_snr;
} gstlal_GSTLALSnglInspiral;


/*
 * Member access
 */


static struct PyMemberDef members[] = {
	{"end_time", T_INT, offsetof(gstlal_GSTLALSnglInspiral, row.end.gpsSeconds), 0, "end_time"},
	{"end_time_ns", T_INT, offsetof(gstlal_GSTLALSnglInspiral, row.end.gpsNanoSeconds), 0, "end_time_ns"},
	{"end_time_gmst", T_DOUBLE, offsetof(gstlal_GSTLALSnglInspiral, row.end_time_gmst), 0, "end_time_gmst"},
	{"impulse_time", T_INT, offsetof(gstlal_GSTLALSnglInspiral, row.impulse_time.gpsSeconds), 0, "impulse_time"},
	{"impulse_time_ns", T_INT, offsetof(gstlal_GSTLALSnglInspiral, row.impulse_time.gpsNanoSeconds), 0, "impulse_time_ns"},
	{"template_duration", T_DOUBLE, offsetof(gstlal_GSTLALSnglInspiral, row.template_duration), 0, "template_duration"},
	{"event_duration", T_DOUBLE, offsetof(gstlal_GSTLALSnglInspiral, row.event_duration), 0, "event_duration"},
	{"amplitude", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.amplitude), 0, "amplitude"},
	{"eff_distance", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.eff_distance), 0, "eff_distance"},
	{"coa_phase", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.coa_phase), 0, "coa_phase"},
	{"mass1", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.mass1), 0, "mass1"},
	{"mass2", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.mass2), 0, "mass2"},
	{"mchirp", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.mchirp), 0, "mchirp"},
	{"mtotal", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.mtotal), 0, "mtotal"},
	{"eta", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.eta), 0, "eta"},
	{"kappa", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.kappa), 0, "kappa"},
	{"chi", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.chi), 0, "chi"},
	{"tau0", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.tau0), 0, "tau0"},
	{"tau2", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.tau2), 0, "tau2"},
	{"tau3", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.tau3), 0, "tau3"},
	{"tau4", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.tau4), 0, "tau4"},
	{"tau5", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.tau5), 0, "tau5"},
	{"ttotal", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.ttotal), 0, "ttotal"},
	{"psi0", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.psi0), 0, "psi0"},
	{"psi3", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.psi3), 0, "psi3"},
	{"alpha", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.alpha), 0, "alpha"},
	{"alpha1", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.alpha1), 0, "alpha1"},
	{"alpha2", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.alpha2), 0, "alpha2"},
	{"alpha3", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.alpha3), 0, "alpha3"},
	{"alpha4", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.alpha4), 0, "alpha4"},
	{"alpha5", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.alpha5), 0, "alpha5"},
	{"alpha6", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.alpha6), 0, "alpha6"},
	{"beta", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.beta), 0, "beta"},
	{"f_final", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.f_final), 0, "f_final"},
	{"snr", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.snr), 0, "snr"},
	{"chisq", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.chisq), 0, "chisq"},
	{"chisq_dof", T_INT, offsetof(gstlal_GSTLALSnglInspiral, row.chisq_dof), 0, "chisq_dof"},
	{"bank_chisq", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.bank_chisq), 0, "bank_chisq"},
	{"bank_chisq_dof", T_INT, offsetof(gstlal_GSTLALSnglInspiral, row.bank_chisq_dof), 0, "bank_chisq_dof"},
	{"cont_chisq", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.cont_chisq), 0, "cont_chisq"},
	{"cont_chisq_dof", T_INT, offsetof(gstlal_GSTLALSnglInspiral, row.cont_chisq_dof), 0, "cont_chisq_dof"},
	{"sigmasq", T_DOUBLE, offsetof(gstlal_GSTLALSnglInspiral, row.sigmasq), 0, "sigmasq"},
	{"rsqveto_duration", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.rsqveto_duration), 0, "rsqveto_duration"},
	{"Gamma0", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.Gamma[0]), 0, "Gamma0"},
	{"Gamma1", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.Gamma[1]), 0, "Gamma1"},
	{"Gamma2", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.Gamma[2]), 0, "Gamma2"},
	{"Gamma3", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.Gamma[3]), 0, "Gamma3"},
	{"Gamma4", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.Gamma[4]), 0, "Gamma4"},
	{"Gamma5", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.Gamma[5]), 0, "Gamma5"},
	{"Gamma6", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.Gamma[6]), 0, "Gamma6"},
	{"Gamma7", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.Gamma[7]), 0, "Gamma7"},
	{"Gamma8", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.Gamma[8]), 0, "Gamma8"},
	{"Gamma9", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.Gamma[9]), 0, "Gamma9"},
        {"spin1x", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.spin1x), 0, "spin1x"},
	{"spin1y", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.spin1y), 0, "spin1y"},
	{"spin1z", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.spin1z), 0, "spin1z"},
	{"spin2x", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.spin2x), 0, "spin2x"},
	{"spin2y", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.spin2y), 0, "spin2y"},
	{"spin2z", T_FLOAT, offsetof(gstlal_GSTLALSnglInspiral, row.spin2z), 0, "spin2z"},
	{"process_id", T_LONG, offsetof(gstlal_GSTLALSnglInspiral, row.process_id), 0, "process_id (long)"},
	{"event_id", T_LONG, offsetof(gstlal_GSTLALSnglInspiral, row.event_id), 0, "event_id (long)"},
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

	return PyUnicode_FromString(s);
}


static int pylal_inline_string_set(PyObject *obj, PyObject *val, void *data)
{
	const struct pylal_inline_string_description *desc = data;
	char *v = PyUnicode_AsUTF8(val);
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


static PyObject *end_get(PyObject *obj, void *null)
{
	return PyObject_CallFunction(LIGOTimeGPSType, "ii", ((gstlal_GSTLALSnglInspiral *) obj)->row.end.gpsSeconds, ((gstlal_GSTLALSnglInspiral *) obj)->row.end.gpsNanoSeconds);
}


static int end_set(PyObject *obj, PyObject *val, void *null)
{
	int end_time, end_time_ns;
	PyObject *converted = PyObject_CallFunctionObjArgs(LIGOTimeGPSType, val, NULL);
	PyObject *attr = NULL;;

	if(!converted)
		goto error;

	attr = PyObject_GetAttrString(converted, "gpsSeconds");
	if(!attr)
		goto error;
	end_time = PyLong_AsLong(attr);
	Py_DECREF(attr);
	attr = PyObject_GetAttrString(converted, "gpsNanoSeconds");
	if(!attr)
		goto error;
	end_time_ns = PyLong_AsLong(attr);
	Py_DECREF(attr);
	Py_DECREF(converted);

	XLALGPSSet(&((gstlal_GSTLALSnglInspiral *) obj)->row.end, end_time, end_time_ns);

	return 0;

error:
	Py_XDECREF(converted);
	Py_XDECREF(attr);
	return -1;
}


static PyObject *template_id_get(PyObject *obj, void *null)
{
	return PyLong_FromLong(((gstlal_GSTLALSnglInspiral *) obj)->row.Gamma[0]);
}


static int template_id_set(PyObject *obj, PyObject *val, void *null)
{
	int template_id = PyLong_AsLong(val);

	if(template_id == -1 && PyErr_Occurred())
		return -1;

	((gstlal_GSTLALSnglInspiral *) obj)->row.Gamma[0] = template_id;

	return 0;
}

static PyObject *G1_snr_component_get(PyObject *obj, void *data)
{
	COMPLEX8TimeSeries *G1_snr = ((gstlal_GSTLALSnglInspiral *) obj)->G1_snr;
	const char *name = data;

	if(!G1_snr) {
		PyErr_SetString(PyExc_ValueError, "no snr time series available");
		return NULL;
	}
	if(!strcmp(name, "_G1_snr_name")) {
		return PyUnicode_FromString(G1_snr->name);
	} else if(!strcmp(name, "_G1_snr_epoch_gpsSeconds")) {
		return PyLong_FromLong(G1_snr->epoch.gpsSeconds);
	} else if(!strcmp(name, "_G1_snr_epoch_gpsNanoSeconds")) {
		return PyLong_FromLong(G1_snr->epoch.gpsNanoSeconds);
	} else if(!strcmp(name, "_G1_snr_f0")) {
		return PyFloat_FromDouble(G1_snr->f0);
	} else if(!strcmp(name, "_G1_snr_deltaT")) {
		return PyFloat_FromDouble(G1_snr->deltaT);
	} else if(!strcmp(name, "_G1_snr_sampleUnits")) {
		char *s = XLALUnitToString(&G1_snr->sampleUnits);
		PyObject *result = PyUnicode_FromString(s);
		XLALFree(s);
		return result;
	} else if(!strcmp(name, "_G1_snr_data_length")) {
		return PyLong_FromLong(G1_snr->data->length);
	} else if(!strcmp(name, "_G1_snr_data")) {
		npy_intp dims[] = {G1_snr->data->length};
		PyObject *array = PyArray_SimpleNewFromData(1, dims, NPY_CFLOAT, G1_snr->data->data);
		if(!array)
			return NULL;
		Py_INCREF(obj);
		PyArray_SetBaseObject((PyArrayObject *) array, obj);
		return array;
	}
	PyErr_BadArgument();
	return NULL;
}

static PyObject *H1_snr_component_get(PyObject *obj, void *data)
{
	COMPLEX8TimeSeries *H1_snr = ((gstlal_GSTLALSnglInspiral *) obj)->H1_snr;
	const char *name = data;

	if(!H1_snr) {
		PyErr_SetString(PyExc_ValueError, "no snr time series available");
		return NULL;
	}
	if(!strcmp(name, "_H1_snr_name")) {
		return PyUnicode_FromString(H1_snr->name);
	} else if(!strcmp(name, "_H1_snr_epoch_gpsSeconds")) {
		return PyLong_FromLong(H1_snr->epoch.gpsSeconds);
	} else if(!strcmp(name, "_H1_snr_epoch_gpsNanoSeconds")) {
		return PyLong_FromLong(H1_snr->epoch.gpsNanoSeconds);
	} else if(!strcmp(name, "_H1_snr_f0")) {
		return PyFloat_FromDouble(H1_snr->f0);
	} else if(!strcmp(name, "_H1_snr_deltaT")) {
		return PyFloat_FromDouble(H1_snr->deltaT);
	} else if(!strcmp(name, "_H1_snr_sampleUnits")) {
		char *s = XLALUnitToString(&H1_snr->sampleUnits);
		PyObject *result = PyUnicode_FromString(s);
		XLALFree(s);
		return result;
	} else if(!strcmp(name, "_H1_snr_data_length")) {
		return PyLong_FromLong(H1_snr->data->length);
	} else if(!strcmp(name, "_H1_snr_data")) {
		npy_intp dims[] = {H1_snr->data->length};
		PyObject *array = PyArray_SimpleNewFromData(1, dims, NPY_CFLOAT, H1_snr->data->data);
		if(!array)
			return NULL;
		Py_INCREF(obj);
		PyArray_SetBaseObject((PyArrayObject *) array, obj);
		return array;
	}
	PyErr_BadArgument();
	return NULL;
}

static PyObject *K1_snr_component_get(PyObject *obj, void *data)
{
	COMPLEX8TimeSeries *K1_snr = ((gstlal_GSTLALSnglInspiral *) obj)->K1_snr;
	const char *name = data;

	if(!K1_snr) {
		PyErr_SetString(PyExc_ValueError, "no snr time series available");
		return NULL;
	}
	if(!strcmp(name, "_K1_snr_name")) {
		return PyUnicode_FromString(K1_snr->name);
	} else if(!strcmp(name, "_K1_snr_epoch_gpsSeconds")) {
		return PyLong_FromLong(K1_snr->epoch.gpsSeconds);
	} else if(!strcmp(name, "_K1_snr_epoch_gpsNanoSeconds")) {
		return PyLong_FromLong(K1_snr->epoch.gpsNanoSeconds);
	} else if(!strcmp(name, "_K1_snr_f0")) {
		return PyFloat_FromDouble(K1_snr->f0);
	} else if(!strcmp(name, "_K1_snr_deltaT")) {
		return PyFloat_FromDouble(K1_snr->deltaT);
	} else if(!strcmp(name, "_K1_snr_sampleUnits")) {
		char *s = XLALUnitToString(&K1_snr->sampleUnits);
		PyObject *result = PyUnicode_FromString(s);
		XLALFree(s);
		return result;
	} else if(!strcmp(name, "_K1_snr_data_length")) {
		return PyLong_FromLong(K1_snr->data->length);
	} else if(!strcmp(name, "_K1_snr_data")) {
		npy_intp dims[] = {K1_snr->data->length};
		PyObject *array = PyArray_SimpleNewFromData(1, dims, NPY_CFLOAT, K1_snr->data->data);
		if(!array)
			return NULL;
		Py_INCREF(obj);
		PyArray_SetBaseObject((PyArrayObject *) array, obj);
		return array;
	}
	PyErr_BadArgument();
	return NULL;
}
static PyObject *L1_snr_component_get(PyObject *obj, void *data)
{
	COMPLEX8TimeSeries *L1_snr = ((gstlal_GSTLALSnglInspiral *) obj)->L1_snr;
	const char *name = data;

	if(!L1_snr) {
		PyErr_SetString(PyExc_ValueError, "no snr time series available");
		return NULL;
	}
	if(!strcmp(name, "_L1_snr_name")) {
		return PyUnicode_FromString(L1_snr->name);
	} else if(!strcmp(name, "_L1_snr_epoch_gpsSeconds")) {
		return PyLong_FromLong(L1_snr->epoch.gpsSeconds);
	} else if(!strcmp(name, "_L1_snr_epoch_gpsNanoSeconds")) {
		return PyLong_FromLong(L1_snr->epoch.gpsNanoSeconds);
	} else if(!strcmp(name, "_L1_snr_f0")) {
		return PyFloat_FromDouble(L1_snr->f0);
	} else if(!strcmp(name, "_L1_snr_deltaT")) {
		return PyFloat_FromDouble(L1_snr->deltaT);
	} else if(!strcmp(name, "_L1_snr_sampleUnits")) {
		char *s = XLALUnitToString(&L1_snr->sampleUnits);
		PyObject *result = PyUnicode_FromString(s);
		XLALFree(s);
		return result;
	} else if(!strcmp(name, "_L1_snr_data_length")) {
		return PyLong_FromLong(L1_snr->data->length);
	} else if(!strcmp(name, "_L1_snr_data")) {
		npy_intp dims[] = {L1_snr->data->length};
		PyObject *array = PyArray_SimpleNewFromData(1, dims, NPY_CFLOAT, L1_snr->data->data);
		if(!array)
			return NULL;
		Py_INCREF(obj);
		PyArray_SetBaseObject((PyArrayObject *) array, obj);
		return array;
	}
	PyErr_BadArgument();
	return NULL;
}

static PyObject *V1_snr_component_get(PyObject *obj, void *data)
{
	COMPLEX8TimeSeries *V1_snr = ((gstlal_GSTLALSnglInspiral *) obj)->V1_snr;
	const char *name = data;

	if(!V1_snr) {
		PyErr_SetString(PyExc_ValueError, "no snr time series available");
		return NULL;
	}
	if(!strcmp(name, "_V1_snr_name")) {
		return PyUnicode_FromString(V1_snr->name);
	} else if(!strcmp(name, "_V1_snr_epoch_gpsSeconds")) {
		return PyLong_FromLong(V1_snr->epoch.gpsSeconds);
	} else if(!strcmp(name, "_V1_snr_epoch_gpsNanoSeconds")) {
		return PyLong_FromLong(V1_snr->epoch.gpsNanoSeconds);
	} else if(!strcmp(name, "_V1_snr_f0")) {
		return PyFloat_FromDouble(V1_snr->f0);
	} else if(!strcmp(name, "_V1_snr_deltaT")) {
		return PyFloat_FromDouble(V1_snr->deltaT);
	} else if(!strcmp(name, "_V1_snr_sampleUnits")) {
		char *s = XLALUnitToString(&V1_snr->sampleUnits);
		PyObject *result = PyUnicode_FromString(s);
		XLALFree(s);
		return result;
	} else if(!strcmp(name, "_V1_snr_data_length")) {
		return PyLong_FromLong(V1_snr->data->length);
	} else if(!strcmp(name, "_V1_snr_data")) {
		npy_intp dims[] = {V1_snr->data->length};
		PyObject *array = PyArray_SimpleNewFromData(1, dims, NPY_CFLOAT, V1_snr->data->data);
		if(!array)
			return NULL;
		Py_INCREF(obj);
		PyArray_SetBaseObject((PyArrayObject *) array, obj);
		return array;
	}
	PyErr_BadArgument();
	return NULL;
}


static struct PyGetSetDef getset[] = {
	{"ifo", pylal_inline_string_get, pylal_inline_string_set, "ifo", &(struct pylal_inline_string_description) {offsetof(gstlal_GSTLALSnglInspiral, row.ifo), LIGOMETA_IFO_MAX}},
	{"search", pylal_inline_string_get, pylal_inline_string_set, "search", &(struct pylal_inline_string_description) {offsetof(gstlal_GSTLALSnglInspiral, row.search), LIGOMETA_SEARCH_MAX}},
	{"channel", pylal_inline_string_get, pylal_inline_string_set, "channel", &(struct pylal_inline_string_description) {offsetof(gstlal_GSTLALSnglInspiral, row.channel), LIGOMETA_CHANNEL_MAX}},
	{"end", end_get, end_set, "end", NULL},
	{"template_id", template_id_get, template_id_set, "template_id", NULL},
	{"_G1_snr_name", G1_snr_component_get, NULL, ".G1_snr.name", "_G1_snr_name"},
	{"_H1_snr_name", H1_snr_component_get, NULL, ".H1_snr.name", "_H1_snr_name"},
	{"_K1_snr_name", K1_snr_component_get, NULL, ".K1_snr.name", "_K1_snr_name"},
	{"_L1_snr_name", L1_snr_component_get, NULL, ".L1_snr.name", "_L1_snr_name"},
	{"_V1_snr_name", V1_snr_component_get, NULL, ".V1_snr.name", "_V1_snr_name"},
	{"_G1_snr_epoch_gpsSeconds", G1_snr_component_get, NULL, ".G1_snr.epoch.gpsSeconds", "_G1_snr_epoch_gpsSeconds"},
	{"_H1_snr_epoch_gpsSeconds", H1_snr_component_get, NULL, ".H1_snr.epoch.gpsSeconds", "_H1_snr_epoch_gpsSeconds"},
	{"_K1_snr_epoch_gpsSeconds", K1_snr_component_get, NULL, ".K1_snr.epoch.gpsSeconds", "_K1_snr_epoch_gpsSeconds"},
	{"_L1_snr_epoch_gpsSeconds", L1_snr_component_get, NULL, ".L1_snr.epoch.gpsSeconds", "_L1_snr_epoch_gpsSeconds"},
	{"_V1_snr_epoch_gpsSeconds", V1_snr_component_get, NULL, ".V1_snr.epoch.gpsSeconds", "_V1_snr_epoch_gpsSeconds"},
	{"_G1_snr_epoch_gpsNanoSeconds", G1_snr_component_get, NULL, ".G1_snr.epoch.gpsNanoSeconds", "_G1_snr_epoch_gpsNanoSeconds"},
	{"_H1_snr_epoch_gpsNanoSeconds", H1_snr_component_get, NULL, ".H1_snr.epoch.gpsNanoSeconds", "_H1_snr_epoch_gpsNanoSeconds"},
	{"_K1_snr_epoch_gpsNanoSeconds", K1_snr_component_get, NULL, ".K1_snr.epoch.gpsNanoSeconds", "_K1_snr_epoch_gpsNanoSeconds"},
	{"_L1_snr_epoch_gpsNanoSeconds", L1_snr_component_get, NULL, ".L1_snr.epoch.gpsNanoSeconds", "_L1_snr_epoch_gpsNanoSeconds"},
	{"_V1_snr_epoch_gpsNanoSeconds", V1_snr_component_get, NULL, ".V1_snr.epoch.gpsNanoSeconds", "_V1_snr_epoch_gpsNanoSeconds"},
	{"_G1_snr_f0", G1_snr_component_get, NULL, ".G1_snr.f0", "_G1_snr_f0"},
	{"_H1_snr_f0", H1_snr_component_get, NULL, ".H1_snr.f0", "_H1_snr_f0"},
	{"_K1_snr_f0", K1_snr_component_get, NULL, ".K1_snr.f0", "_K1_snr_f0"},
	{"_L1_snr_f0", L1_snr_component_get, NULL, ".L1_snr.f0", "_L1_snr_f0"},
	{"_V1_snr_f0", V1_snr_component_get, NULL, ".V1_snr.f0", "_V1_snr_f0"},
	{"_G1_snr_deltaT", G1_snr_component_get, NULL, ".G1_snr.deltaT", "_G1_snr_deltaT"},
	{"_H1_snr_deltaT", H1_snr_component_get, NULL, ".H1_snr.deltaT", "_H1_snr_deltaT"},
	{"_K1_snr_deltaT", K1_snr_component_get, NULL, ".K1_snr.deltaT", "_K1_snr_deltaT"},
	{"_L1_snr_deltaT", L1_snr_component_get, NULL, ".L1_snr.deltaT", "_L1_snr_deltaT"},
	{"_V1_snr_deltaT", V1_snr_component_get, NULL, ".V1_snr.deltaT", "_V1_snr_deltaT"},
	{"_G1_snr_sampleUnits", G1_snr_component_get, NULL, ".G1_snr.sampleUnits", "_G1_snr_sampleUnits"},
	{"_H1_snr_sampleUnits", H1_snr_component_get, NULL, ".H1_snr.sampleUnits", "_H1_snr_sampleUnits"},
	{"_K1_snr_sampleUnits", K1_snr_component_get, NULL, ".K1_snr.sampleUnits", "_K1_snr_sampleUnits"},
	{"_L1_snr_sampleUnits", L1_snr_component_get, NULL, ".L1_snr.sampleUnits", "_L1_snr_sampleUnits"},
	{"_V1_snr_sampleUnits", V1_snr_component_get, NULL, ".V1_snr.sampleUnits", "_V1_snr_sampleUnits"},
	{"_G1_snr_data_length", G1_snr_component_get, NULL, ".G1_snr.data.length", "_G1_snr_data_length"},
	{"_H1_snr_data_length", H1_snr_component_get, NULL, ".H1_snr.data.length", "_H1_snr_data_length"},
	{"_K1_snr_data_length", K1_snr_component_get, NULL, ".K1_snr.data.length", "_K1_snr_data_length"},
	{"_L1_snr_data_length", L1_snr_component_get, NULL, ".L1_snr.data.length", "_L1_snr_data_length"},
	{"_V1_snr_data_length", V1_snr_component_get, NULL, ".V1_snr.data.length", "_V1_snr_data_length"},
	{"_G1_snr_data", G1_snr_component_get, NULL, ".G1_snr.data", "_G1_snr_data"},
	{"_H1_snr_data", H1_snr_component_get, NULL, ".H1_snr.data", "_H1_snr_data"},
	{"_K1_snr_data", K1_snr_component_get, NULL, ".K1_snr.data", "_K1_snr_data"},
	{"_L1_snr_data", L1_snr_component_get, NULL, ".L1_snr.data", "_L1_snr_data"},
	{"_V1_snr_data", V1_snr_component_get, NULL, ".V1_snr.data", "_V1_snr_data"},
	{NULL,}
};


/*
 * Methods
 */


static PyObject *__new__(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	gstlal_GSTLALSnglInspiral *new = (gstlal_GSTLALSnglInspiral *) PyType_GenericNew(type, args, kwds);

	if(!new)
		return NULL;

	/* done */
	return (PyObject *) new;
}


static void __del__(PyObject *self)
{
	if(((gstlal_GSTLALSnglInspiral *) self)->G1_snr != NULL)
		XLALDestroyCOMPLEX8TimeSeries(((gstlal_GSTLALSnglInspiral *) self)->G1_snr);
	if(((gstlal_GSTLALSnglInspiral *) self)->H1_snr != NULL)
		XLALDestroyCOMPLEX8TimeSeries(((gstlal_GSTLALSnglInspiral *) self)->H1_snr);
	if(((gstlal_GSTLALSnglInspiral *) self)->K1_snr != NULL)
		XLALDestroyCOMPLEX8TimeSeries(((gstlal_GSTLALSnglInspiral *) self)->K1_snr);
	if(((gstlal_GSTLALSnglInspiral *) self)->L1_snr != NULL)
		XLALDestroyCOMPLEX8TimeSeries(((gstlal_GSTLALSnglInspiral *) self)->L1_snr);
	if(((gstlal_GSTLALSnglInspiral *) self)->V1_snr != NULL)
		XLALDestroyCOMPLEX8TimeSeries(((gstlal_GSTLALSnglInspiral *) self)->V1_snr);
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
		/* memcpy sngl_inspiral row */
		const struct GSTLALSnglInspiral *gstlal_snglinspiral = (const struct GSTLALSnglInspiral *) data;
		data += sizeof(*gstlal_snglinspiral);
		if (data > end)
		{
			Py_DECREF(item);
			Py_DECREF(result);
			PyErr_SetString(PyExc_ValueError, "buffer overrun while copying sngl_inspiral row");
			return NULL;
		}
		((gstlal_GSTLALSnglInspiral*)item)->row = gstlal_snglinspiral->parent;

		/* duplicate the SNR time series */
		if(gstlal_snglinspiral->G1_length)
		{
			const size_t G1_nbytes = sizeof(gstlal_snglinspiral->snr[0]) * gstlal_snglinspiral->G1_length;
			if (data + G1_nbytes > end)
			{
				Py_DECREF(item);
				Py_DECREF(result);
				PyErr_SetString(PyExc_ValueError, "buffer overrun while copying G1 SNR time series");
				return NULL;
			}
			COMPLEX8TimeSeries *series = XLALCreateCOMPLEX8TimeSeries("snr", &gstlal_snglinspiral->epoch, 0., gstlal_snglinspiral->deltaT, &lalDimensionlessUnit, gstlal_snglinspiral->G1_length);
			if (!series)
			{
				Py_DECREF(item);
				Py_DECREF(result);
				PyErr_SetString(PyExc_MemoryError, "out of memory");
				return NULL;
			}

			memcpy(series->data->data, gstlal_snglinspiral->snr, G1_nbytes);
			data += G1_nbytes;
			((gstlal_GSTLALSnglInspiral*)item)->G1_snr = series;
		} else
			((gstlal_GSTLALSnglInspiral*)item)->G1_snr = NULL;


		if(gstlal_snglinspiral->H1_length)
		{
			const size_t H1_nbytes = sizeof(gstlal_snglinspiral->snr[0]) * gstlal_snglinspiral->H1_length;
			if (data + H1_nbytes > end)
			{
				Py_DECREF(item);
				Py_DECREF(result);
				PyErr_SetString(PyExc_ValueError, "buffer overrun while copying H1 SNR time series");
				return NULL;
			}
			COMPLEX8TimeSeries *series = XLALCreateCOMPLEX8TimeSeries("snr", &gstlal_snglinspiral->epoch, 0., gstlal_snglinspiral->deltaT, &lalDimensionlessUnit, gstlal_snglinspiral->H1_length);
			if (!series)
			{
				Py_DECREF(item);
				Py_DECREF(result);
				PyErr_SetString(PyExc_MemoryError, "out of memory");
				return NULL;
			}

			memcpy(series->data->data, &(gstlal_snglinspiral->snr[gstlal_snglinspiral->G1_length]), H1_nbytes);
			data += H1_nbytes;
			((gstlal_GSTLALSnglInspiral*)item)->H1_snr = series;
		} else
			((gstlal_GSTLALSnglInspiral*)item)->H1_snr = NULL;


		if(gstlal_snglinspiral->K1_length)
		{
			const size_t K1_nbytes = sizeof(gstlal_snglinspiral->snr[0]) * gstlal_snglinspiral->K1_length;
			if (data + K1_nbytes > end)
			{
				Py_DECREF(item);
				Py_DECREF(result);
				PyErr_SetString(PyExc_ValueError, "buffer overrun while copying K1 SNR time series");
				return NULL;
			}
			COMPLEX8TimeSeries *series = XLALCreateCOMPLEX8TimeSeries("snr", &gstlal_snglinspiral->epoch, 0., gstlal_snglinspiral->deltaT, &lalDimensionlessUnit, gstlal_snglinspiral->K1_length);
			if (!series)
			{
				Py_DECREF(item);
				Py_DECREF(result);
				PyErr_SetString(PyExc_MemoryError, "out of memory");
				return NULL;
			}

			memcpy(series->data->data, &(gstlal_snglinspiral->snr[gstlal_snglinspiral->G1_length + gstlal_snglinspiral->H1_length]), K1_nbytes);
			data += K1_nbytes;
			((gstlal_GSTLALSnglInspiral*)item)->K1_snr = series;
		} else
			((gstlal_GSTLALSnglInspiral*)item)->K1_snr = NULL;


		if(gstlal_snglinspiral->L1_length)
		{
			const size_t L1_nbytes = sizeof(gstlal_snglinspiral->snr[0]) * gstlal_snglinspiral->L1_length;
			if (data + L1_nbytes > end)
			{
				Py_DECREF(item);
				Py_DECREF(result);
				PyErr_SetString(PyExc_ValueError, "buffer overrun while copying SNR time series");
				return NULL;
			}

			COMPLEX8TimeSeries *series = XLALCreateCOMPLEX8TimeSeries("snr", &gstlal_snglinspiral->epoch, 0., gstlal_snglinspiral->deltaT, &lalDimensionlessUnit, gstlal_snglinspiral->L1_length);
			if (!series)
			{
				Py_DECREF(item);
				Py_DECREF(result);
				PyErr_SetString(PyExc_MemoryError, "out of memory");
				return NULL;
			}

			memcpy(series->data->data, &(gstlal_snglinspiral->snr[gstlal_snglinspiral->G1_length + gstlal_snglinspiral->H1_length + gstlal_snglinspiral->K1_length]), L1_nbytes);
			data += L1_nbytes;
			((gstlal_GSTLALSnglInspiral*)item)->L1_snr = series;
		} else
			((gstlal_GSTLALSnglInspiral*)item)->L1_snr = NULL;

		if(gstlal_snglinspiral->V1_length)
		{
			const size_t V1_nbytes = sizeof(gstlal_snglinspiral->snr[0]) * gstlal_snglinspiral->V1_length;
			if (data + V1_nbytes > end)
			{
				Py_DECREF(item);
				Py_DECREF(result);
				PyErr_SetString(PyExc_ValueError, "buffer overrun while copying SNR time series");
				return NULL;
			}
			COMPLEX8TimeSeries *series = XLALCreateCOMPLEX8TimeSeries("snr", &gstlal_snglinspiral->epoch, 0., gstlal_snglinspiral->deltaT, &lalDimensionlessUnit, gstlal_snglinspiral->V1_length);
			if (!series)
			{
				Py_DECREF(item);
				Py_DECREF(result);
				PyErr_SetString(PyExc_MemoryError, "out of memory");
				return NULL;
			}

			memcpy(series->data->data, &(gstlal_snglinspiral->snr[gstlal_snglinspiral->G1_length + gstlal_snglinspiral->H1_length + gstlal_snglinspiral->K1_length + gstlal_snglinspiral->L1_length]), V1_nbytes);
			data += V1_nbytes;
			((gstlal_GSTLALSnglInspiral*)item)->V1_snr = series;
		} else
			((gstlal_GSTLALSnglInspiral*)item)->V1_snr = NULL;


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


static PyObject *_G1_snr_time_series_deleter(PyObject *self, PyObject *args)
{
	XLALDestroyCOMPLEX8TimeSeries(((gstlal_GSTLALSnglInspiral *) self)->G1_snr);
	((gstlal_GSTLALSnglInspiral *) self)->G1_snr = NULL;
	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject *_H1_snr_time_series_deleter(PyObject *self, PyObject *args)
{
	XLALDestroyCOMPLEX8TimeSeries(((gstlal_GSTLALSnglInspiral *) self)->H1_snr);
	((gstlal_GSTLALSnglInspiral *) self)->H1_snr = NULL;
	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject *_K1_snr_time_series_deleter(PyObject *self, PyObject *args)
{
	XLALDestroyCOMPLEX8TimeSeries(((gstlal_GSTLALSnglInspiral *) self)->K1_snr);
	((gstlal_GSTLALSnglInspiral *) self)->K1_snr = NULL;
	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject *_L1_snr_time_series_deleter(PyObject *self, PyObject *args)
{
	XLALDestroyCOMPLEX8TimeSeries(((gstlal_GSTLALSnglInspiral *) self)->L1_snr);
	((gstlal_GSTLALSnglInspiral *) self)->L1_snr = NULL;
	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject *_V1_snr_time_series_deleter(PyObject *self, PyObject *args)
{
	XLALDestroyCOMPLEX8TimeSeries(((gstlal_GSTLALSnglInspiral *) self)->V1_snr);
	((gstlal_GSTLALSnglInspiral *) self)->V1_snr = NULL;
	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject *random_obj(PyObject *cls, PyObject *args)
{
	gstlal_GSTLALSnglInspiral *new = (gstlal_GSTLALSnglInspiral *) __new__((PyTypeObject *) cls, NULL, NULL);
	unsigned i;

	new->G1_snr = XLALCreateCOMPLEX8TimeSeries("", &(LIGOTimeGPS) {0, 0}, 0., 1. / 16384, &lalDimensionlessUnit, 16384);
	new->H1_snr = XLALCreateCOMPLEX8TimeSeries("", &(LIGOTimeGPS) {0, 0}, 0., 1. / 16384, &lalDimensionlessUnit, 16384);
	new->K1_snr = XLALCreateCOMPLEX8TimeSeries("", &(LIGOTimeGPS) {0, 0}, 0., 1. / 16384, &lalDimensionlessUnit, 16384);
	new->L1_snr = XLALCreateCOMPLEX8TimeSeries("", &(LIGOTimeGPS) {0, 0}, 0., 1. / 16384, &lalDimensionlessUnit, 16384);
	new->V1_snr = XLALCreateCOMPLEX8TimeSeries("", &(LIGOTimeGPS) {0, 0}, 0., 1. / 16384, &lalDimensionlessUnit, 16384);

	for(i = 0; i < new->G1_snr->data->length; i++)
		new->G1_snr->data->data[i] = 0.;

	for(i = 0; i < new->H1_snr->data->length; i++)
		new->H1_snr->data->data[i] = 0.;

	for(i = 0; i < new->K1_snr->data->length; i++)
		new->K1_snr->data->data[i] = 0.;

	for(i = 0; i < new->L1_snr->data->length; i++)
		new->L1_snr->data->data[i] = 0.;

	for(i = 0; i < new->V1_snr->data->length; i++)
		new->V1_snr->data->data[i] = 0.;

	return (PyObject *) new;
}


static struct PyMethodDef methods[] = {
	{"from_buffer", from_buffer, METH_VARARGS | METH_CLASS, "Construct a tuple of GSTLALSnglInspiral objects from a buffer object.  The buffer is interpreted as a C array of GSTLALSnglInspiral structures.  All data is copied, the buffer can be deallocated afterwards."},
	{"_G1_snr_time_series_deleter", _G1_snr_time_series_deleter, METH_NOARGS, "Release the G1 SNR time series attached to the GSTLALSnglInspiral object."},
	{"_H1_snr_time_series_deleter", _H1_snr_time_series_deleter, METH_NOARGS, "Release the H1 SNR time series attached to the GSTLALSnglInspiral object."},
	{"_K1_snr_time_series_deleter", _K1_snr_time_series_deleter, METH_NOARGS, "Release the K1 SNR time series attached to the GSTLALSnglInspiral object."},
	{"_L1_snr_time_series_deleter", _L1_snr_time_series_deleter, METH_NOARGS, "Release the L1 SNR time series attached to the GSTLALSnglInspiral object."},
	{"_V1_snr_time_series_deleter", _V1_snr_time_series_deleter, METH_NOARGS, "Release the V1 SNR time series attached to the GSTLALSnglInspiral object."},
	{"random", random_obj, METH_NOARGS | METH_CLASS, "Make a GSTLALSnglInspiral with an SNR time series attached to it for G1, H1, K1, L1, and V1 to assist with writing test code."},
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
	t_other.gpsSeconds = PyLong_AsLong(attr);
	Py_DECREF(attr);
	attr = PyObject_GetAttrString(converted, "gpsNanoSeconds");
	if(!attr) {
		Py_DECREF(converted);
		return NULL;
	}
	t_other.gpsNanoSeconds = PyLong_AsLong(attr);
	Py_DECREF(attr);
	Py_DECREF(converted);

	cmp = XLALGPSCmp(&((gstlal_GSTLALSnglInspiral *) self)->row.end, &t_other);

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


static PyTypeObject gstlal_GSTLALSnglInspiral_Type = {
	PyObject_HEAD_INIT(NULL)
	.tp_basicsize = sizeof(gstlal_GSTLALSnglInspiral),
	.tp_doc = "GstLAL's GSTLALSnglInspiral type",
	.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
	.tp_members = members,
	.tp_methods = methods,
	.tp_getset = getset,
	.tp_name = MODULE_NAME ".GSTLALSnglInspiral",
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

static struct PyModuleDef SnglInspiralTableModule = {
	PyModuleDef_HEAD_INIT,
	MODULE_NAME,
	"Low-level wrapper for GSTLALSnglInspiral type.",
	-1,
	NULL
};

PyMODINIT_FUNC PyInit__snglinspiraltable(void)
{
	PyObject *module = PyModule_Create(&SnglInspiralTableModule);

	import_array();

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

	/* SnglInspiralTable */
	if(PyType_Ready(&gstlal_GSTLALSnglInspiral_Type) < 0)
		return;
	Py_INCREF(&gstlal_GSTLALSnglInspiral_Type);
	PyModule_AddObject(module, "GSTLALSnglInspiral", (PyObject *) &gstlal_GSTLALSnglInspiral_Type);

	return module;
}

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
#include <lal/TimeSeries.h>
#include <lal/Units.h>


#include <snglinspiralrowtype.h>


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
	COMPLEX8TimeSeries *snr;
	/* FIXME:  this should be incorporated into the LAL structure */
	EventIDColumn event_id;
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
	{"_process_id", T_LONG, offsetof(gstlal_GSTLALSnglInspiral, row.process_id), 0, "process_id (long)"},
	{"_event_id", T_LONG, offsetof(gstlal_GSTLALSnglInspiral, event_id.id), 0, "event_id (long)"},
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


static PyObject *snr_component_get(PyObject *obj, void *data)
{
	COMPLEX8TimeSeries *snr = ((gstlal_GSTLALSnglInspiral *) obj)->snr;
	const char *name = data;

	if(!snr) {
		PyErr_SetString(PyExc_ValueError, "no snr time series available");
		return NULL;
	}
	if(!strcmp(name, "_snr_name")) {
		return PyString_FromString(snr->name);
	} else if(!strcmp(name, "_snr_epoch_gpsSeconds")) {
		return PyInt_FromLong(snr->epoch.gpsSeconds);
	} else if(!strcmp(name, "_snr_epoch_gpsNanoSeconds")) {
		return PyInt_FromLong(snr->epoch.gpsNanoSeconds);
	} else if(!strcmp(name, "_snr_f0")) {
		return PyFloat_FromDouble(snr->f0);
	} else if(!strcmp(name, "_snr_deltaT")) {
		return PyFloat_FromDouble(snr->deltaT);
	} else if(!strcmp(name, "_snr_sampleUnits")) {
		char *s = XLALUnitToString(&snr->sampleUnits);
		PyObject *result = PyString_FromString(s);
		XLALFree(s);
		return result;
	} else if(!strcmp(name, "_snr_data_length")) {
		return PyInt_FromLong(snr->data->length);
	} else if(!strcmp(name, "_snr_data")) {
		npy_intp dims[] = {snr->data->length};
		PyObject *array = PyArray_SimpleNewFromData(1, dims, NPY_CFLOAT, snr->data->data);
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
	{"_snr_name", snr_component_get, NULL, ".snr.name", "_snr_name"},
	{"_snr_epoch_gpsSeconds", snr_component_get, NULL, ".snr.epoch.gpsSeconds", "_snr_epoch_gpsSeconds"},
	{"_snr_epoch_gpsNanoSeconds", snr_component_get, NULL, ".snr.epoch.gpsNanoSeconds", "_snr_epoch_gpsNanoSeconds"},
	{"_snr_f0", snr_component_get, NULL, ".snr.f0", "_snr_f0"},
	{"_snr_deltaT", snr_component_get, NULL, ".snr.deltaT", "_snr_deltaT"},
	{"_snr_sampleUnits", snr_component_get, NULL, ".snr.sampleUnits", "_snr_sampleUnits"},
	{"_snr_data_length", snr_component_get, NULL, ".snr.data.length", "_snr_data_length"},
	{"_snr_data", snr_component_get, NULL, ".snr.data", "_snr_data"},
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

	/* link the event_id pointer in the sngl_inspiral row structure
	 * to the event_id structure */
	new->row.event_id = &new->event_id;
	new->event_id.id = 0;

	/* done */
	return (PyObject *) new;
}


static void __del__(PyObject *self)
{
	XLALDestroyCOMPLEX8TimeSeries(((gstlal_GSTLALSnglInspiral *) self)->snr);
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
		/* repoint event_id to event_id structure */
		((gstlal_GSTLALSnglInspiral*)item)->row.event_id = &((gstlal_GSTLALSnglInspiral*)item)->event_id;
		/* duplicate the SNR time series */
		if(gstlal_snglinspiral->length)
		{
			const size_t nbytes = sizeof(gstlal_snglinspiral->snr[0]) * gstlal_snglinspiral->length;
			if (data + nbytes > end)
			{
				Py_DECREF(item);
				Py_DECREF(result);
				PyErr_SetString(PyExc_ValueError, "buffer overrun while copying SNR time series");
				return NULL;
			}
			COMPLEX8TimeSeries *series = XLALCreateCOMPLEX8TimeSeries("snr", &gstlal_snglinspiral->epoch, 0., gstlal_snglinspiral->deltaT, &lalDimensionlessUnit, gstlal_snglinspiral->length);
			if (!series)
			{
				Py_DECREF(item);
				Py_DECREF(result);
				PyErr_SetString(PyExc_MemoryError, "out of memory");
				return NULL;
			}
			memcpy(series->data->data, gstlal_snglinspiral->snr, nbytes);
			data += nbytes;
			((gstlal_GSTLALSnglInspiral*)item)->snr = series;
		}

		PyList_Append(result, item);
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


static PyObject *_snr_time_series_deleter(PyObject *self, PyObject *args)
{
	XLALDestroyCOMPLEX8TimeSeries(((gstlal_GSTLALSnglInspiral *) self)->snr);
	((gstlal_GSTLALSnglInspiral *) self)->snr = NULL;
	Py_INCREF(Py_None);
	return Py_None;
}


static PyObject *random_obj(PyObject *cls, PyObject *args)
{
	gstlal_GSTLALSnglInspiral *new = (gstlal_GSTLALSnglInspiral *) __new__((PyTypeObject *) cls, NULL, NULL);
	unsigned i;

	new->snr = XLALCreateCOMPLEX8TimeSeries("", &(LIGOTimeGPS) {0, 0}, 0., 1. / 16384, &lalDimensionlessUnit, 16384);

	for(i = 0; i < new->snr->data->length; i++)
		new->snr->data->data[i] = 0.;

	return (PyObject *) new;
}


static struct PyMethodDef methods[] = {
	{"from_buffer", from_buffer, METH_VARARGS | METH_CLASS, "Construct a tuple of GSTLALSnglInspiral objects from a buffer object.  The buffer is interpreted as a C array of GSTLALSnglInspiral structures.  All data is copied, the buffer can be deallocated afterwards."},
	{"_snr_time_series_deleter", _snr_time_series_deleter, METH_NOARGS, "Release the SNR time series attached to the GSTLALSnglInspiral object."},
	{"random", random_obj, METH_NOARGS | METH_CLASS, "Make a GSTLALSnglInspiral with an SNR time series attached to assist with writing test code."},
	{NULL,}
};


/*
 * Type
 */


static PyTypeObject gstlal_GSTLALSnglInspiral_Type = {
	PyObject_HEAD_INIT(NULL)
	.tp_basicsize = sizeof(gstlal_GSTLALSnglInspiral),
	.tp_doc = "GstLAL's GSTLALSnglInspiral type",
	.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_CHECKTYPES,
	.tp_members = members,
	.tp_methods = methods,
	.tp_getset = getset,
	.tp_name = MODULE_NAME ".GSTLALSnglInspiral",
	.tp_new = __new__,
	.tp_dealloc = __del__,
};


/*
 * ============================================================================
 *
 *                            Module Registration
 *
 * ============================================================================
 */


PyMODINIT_FUNC init_snglinspiraltable(void)
{
	PyObject *module = Py_InitModule3(MODULE_NAME, NULL, "Low-level wrapper for GSTLALSnglInspiral type.");

	import_array();

	/* SnglInspiralTable */
	if(PyType_Ready(&gstlal_GSTLALSnglInspiral_Type) < 0)
		return;
	Py_INCREF(&gstlal_GSTLALSnglInspiral_Type);
	PyModule_AddObject(module, "GSTLALSnglInspiral", (PyObject *) &gstlal_GSTLALSnglInspiral_Type);
}

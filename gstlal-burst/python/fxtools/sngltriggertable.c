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

#include <gstlal-burst/gstlal_sngltrigger.h>
#include <gstlal-burst/sngltriggerrowtype.h>


/*
 * ============================================================================
 *
 *                                    Type
 *
 * ============================================================================
 */


typedef struct {
	PyObject_HEAD
	SnglTriggerTable row;
	COMPLEX8TimeSeries *snr;
	/* FIXME:  this should be incorporated into the LAL structure */
	EventIDColumn event_id;
} gstlal_GSTLALSnglTrigger;


/*
 * Member access
 */


// Modified
static struct PyMemberDef members[] = {
	{"end_time", T_INT, offsetof(gstlal_GSTLALSnglTrigger, row.end.gpsSeconds), 0, "end_time"},
	{"end_time_ns", T_INT, offsetof(gstlal_GSTLALSnglTrigger, row.end.gpsNanoSeconds), 0, "end_time_ns"},
	{"channel_index", T_INT, offsetof(gstlal_GSTLALSnglTrigger, row.channel_index), 0, "channel_index"},
	{"phase", T_FLOAT, offsetof(gstlal_GSTLALSnglTrigger, row.phase), 0, "phase"},
	{"snr", T_FLOAT, offsetof(gstlal_GSTLALSnglTrigger, row.snr), 0, "snr"},
	{"chisq", T_FLOAT, offsetof(gstlal_GSTLALSnglTrigger, row.chisq), 0, "chisq"},
	{"sigmasq", T_DOUBLE, offsetof(gstlal_GSTLALSnglTrigger, row.sigmasq), 0, "sigmasq"},
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


// Modified
static PyObject *snr_component_get(PyObject *obj, void *data)
{
	COMPLEX8TimeSeries *snr = ((gstlal_GSTLALSnglTrigger *) obj)->snr;
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


// Modified
static struct PyGetSetDef getset[] = {
	{"ifo", pylal_inline_string_get, pylal_inline_string_set, "ifo", &(struct pylal_inline_string_description) {offsetof(gstlal_GSTLALSnglTrigger, row.ifo), LIGOMETA_IFO_MAX}},
	{"channel", pylal_inline_string_get, pylal_inline_string_set, "channel", &(struct pylal_inline_string_description) {offsetof(gstlal_GSTLALSnglTrigger, row.channel), LIGOMETA_CHANNEL_MAX}},
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

	//{"search", pylal_inline_string_get, pylal_inline_string_set, "search", &(struct pylal_inline_string_description) {offsetof(gstlal_GSTLALSnglTrigger, row.search), LIGOMETA_SEARCH_MAX}},

/*
 * Methods
 */


// Modified
static PyObject *__new__(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	gstlal_GSTLALSnglTrigger *new = (gstlal_GSTLALSnglTrigger *) PyType_GenericNew(type, args, kwds);

	if(!new)
		return NULL;

	/* link the event_id pointer in the sngl_trigger row structure
	 * to the event_id structure */
	//new->row.event_id = &new->event_id;
	//new->event_id.id = 0;

	/* done */
	return (PyObject *) new;
}


// Modified
static void __del__(PyObject *self)
{
	XLALDestroyCOMPLEX8TimeSeries(((gstlal_GSTLALSnglTrigger *) self)->snr);
	Py_TYPE(self)->tp_free(self);
}


// Modified
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
		/* memcpy sngl_trigger row */
		const struct GSTLALSnglTrigger *gstlal_sngltrigger = (const struct GSTLALSnglTrigger *) data;
		data += sizeof(*gstlal_sngltrigger);
		if (data > end)
		{
			Py_DECREF(item);
			Py_DECREF(result);
			PyErr_SetString(PyExc_ValueError, "buffer overrun while copying sngl_trigger row");
			return NULL;
		}
		((gstlal_GSTLALSnglTrigger*)item)->row = gstlal_sngltrigger->parent;
		/* repoint event_id to event_id structure */
		//((gstlal_GSTLALSnglTrigger*)item)->row.event_id = &((gstlal_GSTLALSnglTrigger*)item)->event_id;
		/* duplicate the SNR time series */
		if(gstlal_sngltrigger->length)
		{
			const size_t nbytes = sizeof(gstlal_sngltrigger->snr[0]) * gstlal_sngltrigger->length;
			if (data + nbytes > end)
			{
				Py_DECREF(item);
				Py_DECREF(result);
				PyErr_SetString(PyExc_ValueError, "buffer overrun while copying SNR time series");
				return NULL;
			}
			COMPLEX8TimeSeries *series = XLALCreateCOMPLEX8TimeSeries("snr", &gstlal_sngltrigger->epoch, 0., gstlal_sngltrigger->deltaT, &lalDimensionlessUnit, gstlal_sngltrigger->length);
			if (!series)
			{
				Py_DECREF(item);
				Py_DECREF(result);
				PyErr_SetString(PyExc_MemoryError, "out of memory");
				return NULL;
			}
			memcpy(series->data->data, gstlal_sngltrigger->snr, nbytes);
			data += nbytes;
			((gstlal_GSTLALSnglTrigger*)item)->snr = series;
		}

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


// Modified
static PyObject *_snr_time_series_deleter(PyObject *self, PyObject *args)
{
	XLALDestroyCOMPLEX8TimeSeries(((gstlal_GSTLALSnglTrigger *) self)->snr);
	((gstlal_GSTLALSnglTrigger *) self)->snr = NULL;
	Py_INCREF(Py_None);
	return Py_None;
}


// Modified
static PyObject *random_obj(PyObject *cls, PyObject *args)
{
	gstlal_GSTLALSnglTrigger *new = (gstlal_GSTLALSnglTrigger *) __new__((PyTypeObject *) cls, NULL, NULL);
	unsigned i;

	new->snr = XLALCreateCOMPLEX8TimeSeries("", &(LIGOTimeGPS) {0, 0}, 0., 1. / 16384, &lalDimensionlessUnit, 16384);

	for(i = 0; i < new->snr->data->length; i++)
		new->snr->data->data[i] = 0.;

	return (PyObject *) new;
}


// Modified
static struct PyMethodDef methods[] = {
	{"from_buffer", from_buffer, METH_VARARGS | METH_CLASS, "Construct a tuple of GSTLALSnglTrigger objects from a buffer object.  The buffer is interpreted as a C array of GSTLALSnglTrigger structures.  All data is copied, the buffer can be deallocated afterwards."},
	{"_snr_time_series_deleter", _snr_time_series_deleter, METH_NOARGS, "Release the SNR time series attached to the GSTLALSnglTrigger object."},
	{"random", random_obj, METH_NOARGS | METH_CLASS, "Make a GSTLALSnglTrigger with an SNR time series attached to assist with writing test code."},
	{NULL,}
};


/*
 * Type
 */


// Modified
static PyTypeObject gstlal_GSTLALSnglTrigger_Type = {
	PyObject_HEAD_INIT(NULL)
	.tp_basicsize = sizeof(gstlal_GSTLALSnglTrigger),
	.tp_doc = "GstLAL's GSTLALSnglTrigger type",
	.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_CHECKTYPES,
	.tp_members = members,
	.tp_methods = methods,
	.tp_getset = getset,
	.tp_name = MODULE_NAME ".GSTLALSnglTrigger",
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


// Modified
PyMODINIT_FUNC init_sngltriggertable(void)
{
	PyObject *module = Py_InitModule3(MODULE_NAME, NULL, "Low-level wrapper for GSTLALSnglTrigger type.");

	import_array();

	/* SnglTriggerTable */
	if(PyType_Ready(&gstlal_GSTLALSnglTrigger_Type) < 0)
		return;
	Py_INCREF(&gstlal_GSTLALSnglTrigger_Type);
	PyModule_AddObject(module, "GSTLALSnglTrigger", (PyObject *) &gstlal_GSTLALSnglTrigger_Type);
}

/*
 * Copyright (C) 2010  Kipp Cannon
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

#include <postcohtable.h>

/*
 * ============================================================================
 *
 *                                    Type
 *
 * ============================================================================
 */


/*
 * Cached ID types
 */

typedef struct {
  PyObject_HEAD
  PostcohInspiralTable row;
  COMPLEX8TimeSeries *snr;
} gstlal_GSTLALPostcohInspiral;

//static PyObject *row_event_id_type = NULL;
//static PyObject *process_id_type = NULL;


/*
 * Member access
 */


static struct PyMemberDef members[] = {
	{"end_time", T_INT, offsetof(gstlal_GSTLALPostcohInspiral, row.end_time.gpsSeconds), 0, "end_time"},
	{"end_time_ns", T_INT, offsetof(gstlal_GSTLALPostcohInspiral, row.end_time.gpsNanoSeconds), 0, "end_time_ns"},
	{"end_time_L", T_INT, offsetof(gstlal_GSTLALPostcohInspiral, row.end_time_L.gpsSeconds), 0, "end_time_L"},
	{"end_time_ns_L", T_INT, offsetof(gstlal_GSTLALPostcohInspiral, row.end_time_L.gpsNanoSeconds), 0, "end_time_ns_L"},
	{"end_time_H", T_INT, offsetof(gstlal_GSTLALPostcohInspiral, row.end_time_H.gpsSeconds), 0, "end_time_H"},
	{"end_time_ns_H", T_INT, offsetof(gstlal_GSTLALPostcohInspiral, row.end_time_H.gpsNanoSeconds), 0, "end_time_ns_H"},
	{"end_time_V", T_INT, offsetof(gstlal_GSTLALPostcohInspiral, row.end_time_V.gpsSeconds), 0, "end_time_V"},
	{"end_time_ns_V", T_INT, offsetof(gstlal_GSTLALPostcohInspiral, row.end_time_V.gpsNanoSeconds), 0, "end_time_ns_V"},
	{"snglsnr_L", T_FLOAT, offsetof(gstlal_GSTLALPostcohInspiral, row.snglsnr_L), 0, "snglsnr_L"},
	{"snglsnr_H", T_FLOAT, offsetof(gstlal_GSTLALPostcohInspiral, row.snglsnr_H), 0, "snglsnr_H"},
	{"snglsnr_V", T_FLOAT, offsetof(gstlal_GSTLALPostcohInspiral, row.snglsnr_V), 0, "snglsnr_V"},
	{"coaphase_L", T_FLOAT, offsetof(gstlal_GSTLALPostcohInspiral, row.coaphase_L), 0, "coaphase_L"},
	{"coaphase_H", T_FLOAT, offsetof(gstlal_GSTLALPostcohInspiral, row.coaphase_H), 0, "coaphase_H"},
	{"coaphase_V", T_FLOAT, offsetof(gstlal_GSTLALPostcohInspiral, row.coaphase_V), 0, "coaphase_V"},
	{"chisq_L", T_FLOAT, offsetof(gstlal_GSTLALPostcohInspiral, row.chisq_L), 0, "chisq_L"},
	{"chisq_H", T_FLOAT, offsetof(gstlal_GSTLALPostcohInspiral, row.chisq_H), 0, "chisq_H"},
	{"chisq_V", T_FLOAT, offsetof(gstlal_GSTLALPostcohInspiral, row.chisq_V), 0, "chisq_V"},
	{"is_background", T_INT, offsetof(gstlal_GSTLALPostcohInspiral, row.is_background), 0, "is_background"},
	{"livetime", T_INT, offsetof(gstlal_GSTLALPostcohInspiral, row.livetime), 0, "livetime"},
	{"tmplt_idx", T_INT, offsetof(gstlal_GSTLALPostcohInspiral, row.tmplt_idx), 0, "tmplt_idx"},
	{"bankid", T_INT, offsetof(gstlal_GSTLALPostcohInspiral, row.bankid), 0, "bankid"},
	{"pix_idx", T_INT, offsetof(gstlal_GSTLALPostcohInspiral, row.pix_idx), 0, "pix_idx"},
	{"cohsnr", T_FLOAT, offsetof(gstlal_GSTLALPostcohInspiral, row.cohsnr), 0, "cohsnr"},
	{"nullsnr", T_FLOAT, offsetof(gstlal_GSTLALPostcohInspiral, row.nullsnr), 0, "nullsnr"},
	{"cmbchisq", T_FLOAT, offsetof(gstlal_GSTLALPostcohInspiral, row.cmbchisq), 0, "cmbchisq"},
	{"spearman_pval", T_FLOAT, offsetof(gstlal_GSTLALPostcohInspiral, row.spearman_pval), 0, "spearman_pval"},
	{"fap", T_FLOAT, offsetof(gstlal_GSTLALPostcohInspiral, row.fap), 0, "fap"},
	{"far_2h", T_FLOAT, offsetof(gstlal_GSTLALPostcohInspiral, row.far_2h), 0, "far_2h"},
	{"far_1d", T_FLOAT, offsetof(gstlal_GSTLALPostcohInspiral, row.far_1d), 0, "far_1d"},
	{"far_1w", T_FLOAT, offsetof(gstlal_GSTLALPostcohInspiral, row.far_1w), 0, "far_1w"},
	{"far", T_FLOAT, offsetof(gstlal_GSTLALPostcohInspiral, row.far), 0, "far"},
	{"far_h", T_FLOAT, offsetof(gstlal_GSTLALPostcohInspiral, row.far_h), 0, "far_h"},
	{"far_l", T_FLOAT, offsetof(gstlal_GSTLALPostcohInspiral, row.far_l), 0, "far_l"},
	{"far_v", T_FLOAT, offsetof(gstlal_GSTLALPostcohInspiral, row.far_v), 0, "far_v"},
	{"far_h_1w", T_FLOAT, offsetof(gstlal_GSTLALPostcohInspiral, row.far_h_1w), 0, "far_h_1w"},
	{"far_l_1w", T_FLOAT, offsetof(gstlal_GSTLALPostcohInspiral, row.far_l_1w), 0, "far_l_1w"},
	{"far_v_1w", T_FLOAT, offsetof(gstlal_GSTLALPostcohInspiral, row.far_v_1w), 0, "far_v_1w"},
	{"far_h_1d", T_FLOAT, offsetof(gstlal_GSTLALPostcohInspiral, row.far_h_1d), 0, "far_h_1d"},
	{"far_l_1d", T_FLOAT, offsetof(gstlal_GSTLALPostcohInspiral, row.far_l_1d), 0, "far_l_1d"},
	{"far_v_1d", T_FLOAT, offsetof(gstlal_GSTLALPostcohInspiral, row.far_v_1d), 0, "far_v_1d"},
	{"far_h_2h", T_FLOAT, offsetof(gstlal_GSTLALPostcohInspiral, row.far_h_2h), 0, "far_h_2h"},
	{"far_l_2h", T_FLOAT, offsetof(gstlal_GSTLALPostcohInspiral, row.far_l_2h), 0, "far_l_2h"},
	{"far_v_2h", T_FLOAT, offsetof(gstlal_GSTLALPostcohInspiral, row.far_v_2h), 0, "far_v_2h"},
	{"rank", T_DOUBLE, offsetof(gstlal_GSTLALPostcohInspiral, row.rank), 0, "rank"},
	{"template_duration", T_DOUBLE, offsetof(gstlal_GSTLALPostcohInspiral, row.template_duration), 0, "template_duration"},
	{"mass1", T_FLOAT, offsetof(gstlal_GSTLALPostcohInspiral, row.mass1), 0, "mass1"},
	{"mass2", T_FLOAT, offsetof(gstlal_GSTLALPostcohInspiral, row.mass2), 0, "mass2"},
	{"mchirp", T_FLOAT, offsetof(gstlal_GSTLALPostcohInspiral, row.mchirp), 0, "mchirp"},
	{"mtotal", T_FLOAT, offsetof(gstlal_GSTLALPostcohInspiral, row.mtotal), 0, "mtotal"},
	{"eta", T_FLOAT, offsetof(gstlal_GSTLALPostcohInspiral, row.eta), 0, "eta"},
	{"spin1x", T_FLOAT, offsetof(gstlal_GSTLALPostcohInspiral, row.spin1x), 0, "spin1x"},
	{"spin1y", T_FLOAT, offsetof(gstlal_GSTLALPostcohInspiral, row.spin1y), 0, "spin1y"},
	{"spin1z", T_FLOAT, offsetof(gstlal_GSTLALPostcohInspiral, row.spin1z), 0, "spin1z"},
	{"spin2x", T_FLOAT, offsetof(gstlal_GSTLALPostcohInspiral, row.spin2x), 0, "spin2x"},
	{"spin2y", T_FLOAT, offsetof(gstlal_GSTLALPostcohInspiral, row.spin2y), 0, "spin2y"},
	{"spin2z", T_FLOAT, offsetof(gstlal_GSTLALPostcohInspiral, row.spin2z), 0, "spin2z"},
	{"ra", T_DOUBLE, offsetof(gstlal_GSTLALPostcohInspiral, row.ra), 0, "ra"},
	{"dec", T_DOUBLE, offsetof(gstlal_GSTLALPostcohInspiral, row.dec), 0, "dec"},
	{"deff_L", T_DOUBLE, offsetof(gstlal_GSTLALPostcohInspiral, row.deff_L), 0, "deff_L"},
	{"deff_H", T_DOUBLE, offsetof(gstlal_GSTLALPostcohInspiral, row.deff_H), 0, "deff_H"},
	{"deff_V", T_DOUBLE, offsetof(gstlal_GSTLALPostcohInspiral, row.deff_V), 0, "deff_V"},
	{"_process_id", T_LONG, offsetof(gstlal_GSTLALPostcohInspiral, row.process_id), 0, "process_id (long)"},
	{"_event_id", T_LONG, offsetof(gstlal_GSTLALPostcohInspiral, row.event_id), 0, "event_id (long)"},
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
	COMPLEX8TimeSeries *snr = ((gstlal_GSTLALPostcohInspiral *) obj)->snr;
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
	{"ifos", pylal_inline_string_get, pylal_inline_string_set, "ifos", &(struct pylal_inline_string_description) {offsetof(gstlal_GSTLALPostcohInspiral, row.ifos), MAX_ALLIFO_LEN}},
	{"pivotal_ifo", pylal_inline_string_get, pylal_inline_string_set, "pivotal_ifo", &(struct pylal_inline_string_description) {offsetof(gstlal_GSTLALPostcohInspiral, row.pivotal_ifo), MAX_IFO_LEN}},
	{"skymap_fname", pylal_inline_string_get, pylal_inline_string_set, "skymap_fname", &(struct pylal_inline_string_description) {offsetof(gstlal_GSTLALPostcohInspiral, row.skymap_fname), MAX_SKYMAP_FNAME_LEN}},
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


//static Py_ssize_t getreadbuffer(PyObject *self, Py_ssize_t segment, void **ptrptr)
//{
//	if(segment) {
//		PyErr_SetString(PyExc_SystemError, "bad segment");
//		return -1;
//	}
//	*ptrptr = &((gstlal_GSTLALPostcohInspiral*)self)->row;
//	return sizeof(((gstlal_GSTLALPostcohInspiral*)self)->row);
//}
//
//
//static Py_ssize_t getsegcount(PyObject *self, Py_ssize_t *lenp)
//{
//	if(lenp)
//		*lenp = sizeof(((gstlal_GSTLALPostcohInspiral*)self)->row);
//	return 1;
//}
//
//
//static PyBufferProcs as_buffer = {
//	.bf_getreadbuffer = getreadbuffer,
//	.bf_getsegcount = getsegcount,
//	.bf_getwritebuffer = NULL,
//	.bf_getcharbuffer = NULL
//};
//

/*
 * Methods
 */


static PyObject *__new__(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	gstlal_GSTLALPostcohInspiral *new = (gstlal_GSTLALPostcohInspiral *) PyType_GenericNew(type, args, kwds);

	if(!new)
		return NULL;

	/* link the event_id pointer in the row table structure
	 * to the event_id structure */
	//new->row->event_id = new->event_id_i;

	//new->process_id_i = 0;
	//new->event_id_i = 0;

	/* done */
	return (PyObject *) new;
}

static void __del__(PyObject *self)
{
	if (((gstlal_GSTLALPostcohInspiral *) self)->snr)
	  XLALDestroyCOMPLEX8TimeSeries(((gstlal_GSTLALPostcohInspiral *) self)->snr);
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
		/* memcpy postcoh row */
		const PostcohInspiralTable *gstlal_postcohinspiral = (const PostcohInspiralTable *) data;
		data += sizeof(*gstlal_postcohinspiral);
		/* if the data read in is less then expected amount */
		if (data > end)
		{
			Py_DECREF(item);
			Py_DECREF(result);
			PyErr_SetString(PyExc_ValueError, "buffer overrun while copying postcoh row");
			return NULL;
		}
		/* sorb the PostcohInspiralTable entry from the pipeline to the gstlal_GSTLALPostcohInspiral item*/
		((gstlal_GSTLALPostcohInspiral*)item)->row = (PostcohInspiralTable)*gstlal_postcohinspiral;
		/* duplicate the SNR time series if we have length? */
		if(gstlal_postcohinspiral->snr_length)
		{
			const size_t nbytes = sizeof(gstlal_postcohinspiral->snr[0]) * gstlal_postcohinspiral->snr_length;
			if (data + nbytes > end)
			{
				Py_DECREF(item);
				Py_DECREF(result);
				PyErr_SetString(PyExc_ValueError, "buffer overrun while copying SNR time series");
				return NULL;
			}
			COMPLEX8TimeSeries *series = XLALCreateCOMPLEX8TimeSeries("snr", &gstlal_postcohinspiral->epoch, 0., gstlal_postcohinspiral->deltaT, &lalDimensionlessUnit, gstlal_postcohinspiral->snr_length);
			if (!series)
			{
				Py_DECREF(item);
				Py_DECREF(result);
				PyErr_SetString(PyExc_MemoryError, "out of memory");
				return NULL;
			}
			memcpy(series->data->data, gstlal_postcohinspiral->snr, nbytes);
			data += nbytes;
			((gstlal_GSTLALPostcohInspiral*)item)->snr = series;
		} else
			((gstlal_GSTLALPostcohInspiral*)item)->snr = NULL;

		if (PyList_Append(result, item))
		  printf("append failure");
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
	{"from_buffer", from_buffer, METH_VARARGS | METH_CLASS, "Construct a tuple of PostcohInspiralTable objects from a buffer object.  The buffer is interpreted as a C array of PostcohInspiralTable structures."},
	{NULL,}
};


/*
 * Type
 */


static PyTypeObject gstlal_GSTLALPostcohInspiral_Type = {
	PyObject_HEAD_INIT(NULL)
	.tp_basicsize = sizeof(gstlal_GSTLALPostcohInspiral),
	.tp_doc = "LAL's PostcohInspiral structure",
	.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_CHECKTYPES,
	.tp_members = members,
	.tp_methods = methods,
	.tp_getset = getset,
	.tp_name = MODULE_NAME ".GSTLALPostcohInspiral",
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


PyMODINIT_FUNC init_postcohtable(void)
{
	PyObject *module = Py_InitModule3(MODULE_NAME, NULL, "Wrapper for LAL's PostcohInspiralTable type.");

	import_array();

	/* Cached ID types */
	//process_id_type = pylal_get_ilwdchar_class("process", "process_id");
	//row_event_id_type = pylal_get_ilwdchar_class("postcoh", "event_id");

	/* PostcohInspiralTable */
	//_gstlal_GSTLALPostcohInspiral_Type = &pylal_postcohinspiraltable_type;
	if(PyType_Ready(&gstlal_GSTLALPostcohInspiral_Type) < 0)
		return;
	Py_INCREF(&gstlal_GSTLALPostcohInspiral_Type);
	PyModule_AddObject(module, "GSTLALPostcohInspiral", (PyObject *) &gstlal_GSTLALPostcohInspiral_Type);
}

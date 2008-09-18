/*
 * PSD Estimation and whitener
 *
 * Copyright (C) 2008  Kipp Cannon, Chad Hanna
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */


/*
 * ========================================================================
 *
 *                                  Preamble
 *
 * ========================================================================
 */


/*
 * stuff from the C library
 */


#include <math.h>


/*
 * stuff from gstreamer
 */


#include <gst/gst.h>
#include <gst/base/gstadapter.h>


/*
 * stuff from LAL
 */


#include <lal/LALDatatypes.h>
#include <lal/LALStdlib.h>
#include <lal/Date.h>
#include <lal/Sequence.h>
#include <lal/TimeSeries.h>
#include <lal/FrequencySeries.h>
#include <lal/TimeFreqFFT.h>
#include <lal/LALNoiseModels.h>
#include <lal/Units.h>
#include <lal/LALComplex.h>
#include <lal/Window.h>


/*
 * our own stuff
 */


#include <gstlal_whiten.h>


/*
 * ============================================================================
 *
 *                                 Parameters
 *
 * ============================================================================
 */


#define DEFAULT_FILTER_LENGTH 8.0
#define DEFAULT_CONVOLUTION_LENGTH 64.0
#define DEFAULT_AVERAGE_SAMPLES 16
#define DEFAULT_PSDMODE GSTLAL_PSDMODE_INITIAL_LIGO_SRD


/*
 * ============================================================================
 *
 *                                Custom Types
 *
 * ============================================================================
 */


/*
 * PSD mode enum
 */


GType gstlal_psdmode_get_type(void)
{
	static GType type = 0;

	if(!type) {
		static GEnumValue values[] = {
			{GSTLAL_PSDMODE_INITIAL_LIGO_SRD, "GSTLAL_PSDMODE_INITIAL_LIGO_SRD", "Use Initial LIGO SRD for PSD"},
			{GSTLAL_PSDMODE_RUNNING_AVERAGE, "GSTLAL_PSDMODE_RUNNING_AVERAGE", "Use running average for PSD"},
			{0, NULL, NULL}
		};

		type = g_enum_register_static("GSTLAL_PSDMODE", values);
	}

	return type;
}


/*
 * ============================================================================
 *
 *                                Support Code
 *
 * ============================================================================
 */


static LALUnit lalStrainSquaredPerHertz(void)
{
	LALUnit unit;

	return *XLALUnitMultiply(&unit, XLALUnitSquare(&unit, &lalStrainUnit), &lalSecondUnit);
}


static int make_window(GSTLALWhiten *element)
{
	unsigned transient = trunc(element->filter_length * element->sample_rate + 0.5);
	unsigned hann_length = trunc(element->convolution_length * element->sample_rate + 0.5) - 2 * transient;

	element->window = XLALCreateHannREAL8Window(hann_length);
	if(!element->window) {
		GST_ERROR("failure creating Hann window");
		return -1;
	}
	if(!XLALResizeREAL8Sequence(element->window->data, -transient, hann_length + 2 * transient)) {
		GST_ERROR("failure resizing Hann window");
		return -1;
	}

	return 0;
}


static int make_fft_plans(GSTLALWhiten *element)
{
	unsigned fft_length = trunc(element->convolution_length * element->sample_rate + 0.5);

	XLALDestroyREAL8FFTPlan(element->fwdplan);
	XLALDestroyREAL8FFTPlan(element->revplan);

	element->fwdplan = XLALCreateForwardREAL8FFTPlan(fft_length, 1);
	element->revplan = XLALCreateReverseREAL8FFTPlan(fft_length, 1);

	if(!element->fwdplan || !element->revplan) {
		GST_ERROR("failure creating FFT plans");
		XLALDestroyREAL8FFTPlan(element->fwdplan);
		XLALDestroyREAL8FFTPlan(element->revplan);
		element->fwdplan = NULL;
		element->revplan = NULL;
		return -1;
	}

	return 0;
}


static REAL8FrequencySeries *make_empty_psd(const GSTLALWhiten *element)
{
	LIGOTimeGPS gps_zero = {0, 0};
	LALUnit strain_squared_per_hertz = lalStrainSquaredPerHertz();
	unsigned segment_length = trunc(element->convolution_length * element->sample_rate + 0.5);
	unsigned psd_length = segment_length / 2 + 1;
	REAL8FrequencySeries *psd;

	psd = XLALCreateREAL8FrequencySeries("PSD", &gps_zero, 0.0, 1.0 / element->convolution_length, &strain_squared_per_hertz, psd_length);

	if(!psd)
		GST_ERROR("XLALCreateREAL8FrequencySeries() failed");

	return psd;
}


static REAL8FrequencySeries *make_iligo_psd(const GSTLALWhiten *element)
{
	REAL8FrequencySeries *psd = make_empty_psd(element);
	unsigned i;

	if(!psd)
		return NULL;

	for(i = 0; i < psd->data->length; i++)
		psd->data->data[i] = XLALLIGOIPsd(psd->f0 + i * psd->deltaF);

	return psd;
}


static int get_psd(GSTLALWhiten *element)
{
	XLALDestroyREAL8FrequencySeries(element->psd);
	element->psd = NULL;

	switch(element->psdmode) {
	case GSTLAL_PSDMODE_INITIAL_LIGO_SRD:
		element->psd = make_iligo_psd(element);
		if(!element->psd)
			return -1;
		break;

	case GSTLAL_PSDMODE_RUNNING_AVERAGE:
		if(!element->psd_regressor->n_samples) {
			/* no data for the average yet, seed psd regressor
			 * with initial LIGO SRD */
			REAL8FrequencySeries *psd = make_iligo_psd(element);
			if(!psd)
				return -1;
			if(XLALPSDRegressorSetPSD(element->psd_regressor, psd, element->psd_regressor->max_samples)) {
				GST_ERROR("XLALPSDRegressorSetPSD() failed");
				XLALDestroyREAL8FrequencySeries(psd);
				return -1;
			}
			XLALDestroyREAL8FrequencySeries(psd);
		}
		element->psd = XLALPSDRegressorGetPSD(element->psd_regressor);
		if(!element->psd) {
			GST_ERROR("XLALPSDRegressorGetPSD() failed");
			return -1;
		}
		break;
	}

	return 0;
}


/*
 * ============================================================================
 *
 *                                  The Guts
 *
 * ============================================================================
 */


/*
 * Properties
 */


enum property {
	ARG_PSDMODE = 1,
	ARG_FILTER_LENGTH,
	ARG_CONVOLUTION_LENGTH,
	ARG_AVERAGE_SAMPLES
};


static void set_property(GObject * object, enum property id, const GValue * value, GParamSpec * pspec)
{

	GSTLALWhiten *element = GSTLAL_WHITEN(object);

	switch (id) {
	case ARG_PSDMODE:
		element->psdmode = g_value_get_enum(value);
		break;

	case ARG_FILTER_LENGTH:
		element->filter_length = g_value_get_double(value);
		break;

	case ARG_CONVOLUTION_LENGTH:
		element->convolution_length = g_value_get_double(value);
		break;

	case ARG_AVERAGE_SAMPLES:
		element->psd_regressor->max_samples = g_value_get_int(value);
		break;
	}
}


static void get_property(GObject * object, enum property id, GValue * value, GParamSpec * pspec)
{
	GSTLALWhiten *element = GSTLAL_WHITEN(object);

	switch (id) {
	case ARG_PSDMODE:
		g_value_set_enum(value, element->psdmode);
		break;

	case ARG_FILTER_LENGTH:
		g_value_set_double(value, element->filter_length);
		break;

	case ARG_CONVOLUTION_LENGTH:
		g_value_set_double(value, element->convolution_length);
		break;

	case ARG_AVERAGE_SAMPLES:
		g_value_set_int(value, element->psd_regressor->max_samples);
		break;
	}
}


/*
 * setcaps()
 */


static gboolean setcaps(GstPad *pad, GstCaps *caps)
{
	GSTLALWhiten *element = GSTLAL_WHITEN(gst_pad_get_parent(pad));
	gboolean result = TRUE;

	/* FIXME:  this element doesn't handle the caps changing in mid
	 * stream, but it could */

	element->sample_rate = g_value_get_int(gst_structure_get_value(gst_caps_get_structure(caps, 0), "rate"));

	result = gst_pad_set_caps(element->srcpad, caps);

	gst_object_unref(element);
	return result;
}


/*
 * chain()
 */


static GstFlowReturn chain(GstPad *pad, GstBuffer *sinkbuf)
{
	GSTLALWhiten *element = GSTLAL_WHITEN(gst_pad_get_parent(pad));
	GstCaps *caps = gst_buffer_get_caps(sinkbuf);
	GstFlowReturn result = GST_FLOW_OK;
	gboolean is_discontinuity = FALSE;
	unsigned segment_length = trunc(element->convolution_length * element->sample_rate + 0.5);
	unsigned transient = trunc(element->filter_length * element->sample_rate + 0.5);
	REAL8TimeSeries *segment = NULL;
	COMPLEX16FrequencySeries *tilde_segment = NULL;

	/*
	 * Push the incoming buffer into the adapter.  If the buffer is a
	 * discontinuity, first clear the adapter and reset the clock
	 */

	if(GST_BUFFER_IS_DISCONT(sinkbuf)) {
		/* FIXME:  if there is tail data left over, maybe it should
		 * be pushed downstream? */
		is_discontinuity = TRUE;
		gst_adapter_clear(element->adapter);
		element->adapter_head_timestamp = GST_BUFFER_TIMESTAMP(sinkbuf);
	}

	gst_adapter_push(element->adapter, sinkbuf);

	/*
	 * Make sure we've got a Hann window and FFT plans
	 */

	if(!element->window) {
		if(make_window(element)) {
			result = GST_FLOW_ERROR;
			goto done;
		}
	}
	if(!element->fwdplan || !element->revplan) {
		if(make_fft_plans(element)) {
			result = GST_FLOW_ERROR;
			goto done;
		}
	}

	/*
	 * Create holding area and make sure we've got a holding place for
	 * the tail.
	 */

	segment = XLALCreateREAL8TimeSeries(NULL, &(LIGOTimeGPS) {0, 0}, 0.0, (double) 1.0 / element->sample_rate, &lalStrainUnit, segment_length);
	tilde_segment = XLALCreateCOMPLEX16FrequencySeries(NULL, &(LIGOTimeGPS) {0, 0}, 0, 0, &lalDimensionlessUnit, segment_length / 2 + 1);
	if(!segment || !tilde_segment) {
		GST_ERROR("failure creating holding area");
		result = GST_FLOW_ERROR;
		goto done;
	}
	if(!element->tail) {
		element->tail = XLALCutREAL8Sequence(segment->data, 0, segment_length / 2 - transient);
		if(!element->tail) {
			result = GST_FLOW_ERROR;
			goto done;
		}
		memset(element->tail->data, 0, element->tail->length * sizeof(*element->tail->data));
	}

	/*
	 * Iterate over the available data
	 */

	while(gst_adapter_available(element->adapter) / sizeof(*segment->data->data) >= segment_length) {
		GstBuffer *srcbuf;
		unsigned i;

		/*
		 * Make sure we've got an up-to-date PSD
		 */

		if(!element->psd || element->psdmode == GSTLAL_PSDMODE_RUNNING_AVERAGE) {
			if(get_psd(element)) {
				result = GST_FLOW_ERROR;
				goto done;
			}
		}

		/*
		 * Copy data from adapter into holding area
		 */

		memcpy(segment->data->data, gst_adapter_peek(element->adapter, segment_length * sizeof(*segment->data->data)), segment_length * sizeof(*segment->data->data));

		/*
		 * Transform to frequency domain
		 */

		if(!XLALUnitaryWindowREAL8Sequence(segment->data, element->window)) {
			GST_ERROR("XLALUnitaryWindowREAL8Sequence() failed");
			result = GST_FLOW_ERROR;
			goto done;
		}
		if(XLALREAL8TimeFreqFFT(tilde_segment, segment, element->fwdplan)) {
			GST_ERROR("XLALREAL8TimeFreqFFT() failed");
			result = GST_FLOW_ERROR;
			goto done;
		}

		/*
		 * Add frequency domain data to spectrum averager
		 */

		if(XLALPSDRegressorAdd(element->psd_regressor, tilde_segment)) {
			GST_ERROR("XLALPSDRegressorAdd() failed");
			result = GST_FLOW_ERROR;
			goto done;
		}

		/*
		 * Remove lines and whiten.
		 */

		if(XLALPSDRegressorRemoveLines(element->psd_regressor, tilde_segment, 4.0)) {
			GST_ERROR("XLALPSDRegressorRemoveLines() failed");
			result = GST_FLOW_ERROR;
			goto done;
		}
		if(!XLALWhitenCOMPLEX16FrequencySeries(tilde_segment, element->psd)) {
			GST_ERROR("XLALWhitenCOMPLEX16FrequencySeries() failed");
			result = GST_FLOW_ERROR;
			goto done;
		}

		/*
		 * Transform to time domain
		 */

		if(XLALREAL8FreqTimeFFT(segment, tilde_segment, element->revplan)) {
			GST_ERROR("XLALREAL8FreqTimeFFT() failed");
			result = GST_FLOW_ERROR;
			goto done;
		}

		/*
		 * Get a buffer from the downstream peer (note the size is
		 * half of the holding area minus the transient)
		 */

		result = gst_pad_alloc_buffer(element->srcpad, element->next_sample, (segment_length / 2 - transient) * sizeof(*segment->data->data), GST_PAD_CAPS(element->srcpad), &srcbuf);
		if(result != GST_FLOW_OK)
			goto done;
		if(is_discontinuity) {
			GST_BUFFER_FLAG_SET(srcbuf, GST_BUFFER_FLAG_DISCONT);
			is_discontinuity = FALSE;
		}
		GST_BUFFER_OFFSET_END(srcbuf) = GST_BUFFER_OFFSET(srcbuf) + segment_length - 1;
		GST_BUFFER_TIMESTAMP(srcbuf) = element->adapter_head_timestamp + transient * GST_SECOND / element->sample_rate;
		GST_BUFFER_DURATION(srcbuf) = (GstClockTime) (segment_length / 2 - transient) * GST_SECOND / element->sample_rate;

		/*
		 * Copy the first half of the time series into the buffer,
		 * removing the transient from the start, and adding the
		 * contents of the tail
		 */

		for(i = 0; i < element->tail->length; i++)
			((double *) GST_BUFFER_DATA(srcbuf))[i] = segment->data->data[transient + i] + element->tail->data[i];

		/*
		 * Push the buffer downstream
		 */

		result = gst_pad_push(element->srcpad, srcbuf);
		if(result != GST_FLOW_OK)
			goto done;

		/*
		 * Save the second half of time series data minus the final
		 * transient in the tail
		 */

		memcpy(element->tail->data, &segment->data->data[transient + i], element->tail->length * sizeof(*element->tail->data));

		/*
		 * Flush the adapter and advance the sample count and
		 * adapter clock
		 */

		gst_adapter_flush(element->adapter, (segment_length / 2 - transient) * sizeof(*segment->data->data));
		element->next_sample += segment_length / 2 - transient;
		element->adapter_head_timestamp += (GstClockTime) (segment_length / 2 - transient) * GST_SECOND / element->sample_rate;
	}

	/*
	 * Done
	 */

done:
	XLALDestroyREAL8TimeSeries(segment);
	XLALDestroyCOMPLEX16FrequencySeries(tilde_segment);
	gst_caps_unref(caps);
	gst_object_unref(element);
	return result;
}


/*
 * Parent class.
 */


static GstElementClass *parent_class = NULL;


/*
 * Instance dispose function.  See ???
 */


static void dispose(GObject * object)
{
	GSTLALWhiten *element = GSTLAL_WHITEN(object);

	g_object_unref(element->adapter);
	element->adapter = NULL;
	gst_object_unref(element->srcpad);
	element->srcpad = NULL;
	XLALDestroyREAL8Window(element->window);
	XLALDestroyREAL8FFTPlan(element->fwdplan);
	XLALDestroyREAL8FFTPlan(element->revplan);
	XLALPSDRegressorFree(element->psd_regressor);
	XLALDestroyREAL8FrequencySeries(element->psd);
	XLALDestroyREAL8Sequence(element->tail);

	G_OBJECT_CLASS(parent_class)->dispose(object);
}


/*
 * Base init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GBaseInitFunc
 */


static void base_init(gpointer class)
{
	static GstElementDetails plugin_details = {
		"Whiten",
		"Filter",
		"A PSD estimator and time series whitener",
		"Kipp Cannon <kcannon@ligo.caltech.edu>, Chan Hanna <channa@ligo.caltech.edu>"
	};
	GstElementClass *element_class = GST_ELEMENT_CLASS(class);

	gst_element_class_set_details(element_class, &plugin_details);

	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"sink",
			GST_PAD_SINK,
			GST_PAD_ALWAYS,
			gst_caps_new_simple(
				"audio/x-raw-float",
				"rate", GST_TYPE_INT_RANGE, 1, G_MAXINT,
				"channels", G_TYPE_INT, 1,
				"endianness", G_TYPE_INT, G_BYTE_ORDER,
				"width", G_TYPE_INT, 64,
				NULL
			)
		)
	);

	gst_element_class_add_pad_template(
		element_class,
		gst_pad_template_new(
			"src",
			GST_PAD_SRC,
			GST_PAD_ALWAYS,
			gst_caps_new_simple(
				"audio/x-raw-float",
				"rate", GST_TYPE_INT_RANGE, 1, G_MAXINT,
				"channels", G_TYPE_INT, 1,
				"endianness", G_TYPE_INT, G_BYTE_ORDER,
				"width", G_TYPE_INT, 64,
				NULL
			)
		)
	);
}


/*
 * Class init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GClassInitFunc
 */


static void class_init(gpointer class, gpointer class_data)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(class);

	parent_class = g_type_class_ref(GST_TYPE_ELEMENT);

	gobject_class->set_property = set_property;
	gobject_class->get_property = get_property;
	gobject_class->dispose = dispose;

	g_object_class_install_property(gobject_class, ARG_PSDMODE, g_param_spec_enum("psd-mode", "PSD mode", "PSD estimation mode", GSTLAL_PSDMODE_TYPE, DEFAULT_PSDMODE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	g_object_class_install_property(gobject_class, ARG_FILTER_LENGTH, g_param_spec_double("filter-length", "Filter length", "Length of the whitening filter in seconds", 0, G_MAXDOUBLE, DEFAULT_FILTER_LENGTH, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	g_object_class_install_property(gobject_class, ARG_CONVOLUTION_LENGTH, g_param_spec_double("convolution-length", "Convolution length", "Length of the FFT convolution in seconds", 0, G_MAXDOUBLE, DEFAULT_CONVOLUTION_LENGTH, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	g_object_class_install_property(gobject_class, ARG_AVERAGE_SAMPLES, g_param_spec_int("average-samples", "Average samples", "Number of convolution-length intervals used in PSD average", 1, G_MAXINT, DEFAULT_AVERAGE_SAMPLES, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
}


/*
 * Instance init function.  See
 *
 * http://developer.gnome.org/doc/API/2.0/gobject/gobject-Type-Information.html#GInstanceInitFunc
 */


static void instance_init(GTypeInstance * object, gpointer class)
{
	GSTLALWhiten *element = GSTLAL_WHITEN(object);
	GstPad *pad;

	gst_element_create_all_pads(GST_ELEMENT(element));

	/* configure sink pad */
	pad = gst_element_get_static_pad(GST_ELEMENT(element), "sink");
	gst_pad_set_setcaps_function(pad, setcaps);
	gst_pad_set_chain_function(pad, chain);
	gst_object_unref(pad);

	/* retrieve (and ref) src pad */
	element->srcpad = gst_element_get_static_pad(GST_ELEMENT(object), "src");

	/* internal data */
	element->adapter = gst_adapter_new();
	element->next_sample = 0;
	element->adapter_head_timestamp = 0;
	element->filter_length = DEFAULT_FILTER_LENGTH;
	element->convolution_length = DEFAULT_CONVOLUTION_LENGTH;
	element->psdmode = DEFAULT_PSDMODE;
	element->sample_rate = 0;
	element->window = NULL;
	element->fwdplan = NULL;
	element->revplan = NULL;
	element->psd_regressor = XLALPSDRegressorNew(DEFAULT_AVERAGE_SAMPLES);
	element->psd = NULL;
	element->tail = NULL;
}


/*
 * gstlal_whiten_get_type().
 */


GType gstlal_whiten_get_type(void)
{
	static GType type = 0;

	if(!type) {
		static const GTypeInfo info = {
			.class_size = sizeof(GSTLALWhitenClass),
			.class_init = class_init,
			.base_init = base_init,
			.instance_size = sizeof(GSTLALWhiten),
			.instance_init = instance_init,
		};
		type = g_type_register_static(GST_TYPE_ELEMENT, "lal_whiten", &info, 0);
	}

	return type;
}

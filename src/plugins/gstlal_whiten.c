/*
 * PSD Estimation and whitener
 *
 * Copyright (C) 2008  Kipp Cannon, Chad Hanna, Drew Keppel
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
 * stuff from glib/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
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
#include <lal/Units.h>
#include <lal/LALComplex.h>
#include <lal/Window.h>
#include <lal/Units.h>


/*
 * our own stuff
 */


#include <gstlal.h>
#include <gstlal_plugins.h>
#include <gstlal_whiten.h>


static const LIGOTimeGPS GPS_ZERO = {0, 0};


/*
 * ============================================================================
 *
 *                                 Parameters
 *
 * ============================================================================
 */


#define DEFAULT_ZERO_PAD_SECONDS 2.0
#define DEFAULT_FFT_LENGTH_SECONDS 8.0
#define DEFAULT_AVERAGE_SAMPLES 32
#define DEFAULT_MEDIAN_SAMPLES 9
#define DEFAULT_PSDMODE GSTLAL_PSDMODE_RUNNING_AVERAGE


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
			{GSTLAL_PSDMODE_RUNNING_AVERAGE, "GSTLAL_PSDMODE_RUNNING_AVERAGE", "Use running average for PSD"},
			{GSTLAL_PSDMODE_FIXED, "GSTLAL_PSDMODE_FIXED", "Use fixed spectrum for PSD"},
			{0, NULL, NULL}
		};

		type = g_enum_register_static("GSTLAL_PSDMODE", values);
	}

	return type;
}


/*
 * ============================================================================
 *
 *                                 Utilities
 *
 * ============================================================================
 */


static int fft_length(const GSTLALWhiten *element)
{
	return round(element->fft_length_seconds * element->sample_rate);
}


static int zero_pad_length(const GSTLALWhiten *element)
{
	return round(element->zero_pad_seconds * element->sample_rate);
}


static void reset_workspace_metadata(GSTLALWhiten *element)
{
	element->tdworkspace->deltaT = (double) 1.0 / element->sample_rate;
	element->tdworkspace->sampleUnits = element->sample_units;
	element->fdworkspace->deltaF = (double) 1.0 / (element->tdworkspace->deltaT * fft_length(element));
}


static int make_window_and_fft_plans(GSTLALWhiten *element)
{
	/*
	 * build a Hann window with zero-padding.  both fft_length and
	 * zero_pad are an even number of samples (enforced in the caps
	 * negotiation phase).  we need a Hann window with an odd number of
	 * samples so that there is a middle sample (= 1) to overlap the
	 * end sample (= 0) of the next window.  we achieve this by adding
	 * 1 to the length of the envelope, and then clipping the last
	 * sample.  the result is a sequence of windows that fit together
	 * as shown below:
	 *
	 * 1.0 --------A-------B-------C-------
	 *     ------A---A---B---B---C---C-----
	 * 0.5 ----A-------A-------B-------C---
	 *     --A-------B---A---C---B-------C-
	 * 0.0 A-------B-------C---------------
	 *
	 * i.e., A is "missing" its last sample, which is where C begins,
	 * and B's first sample starts on A's middle sample, and the sum of
	 * the windows is identically 1 everywhere.
	 */

	XLALDestroyREAL8Window(element->window);
	element->window = XLALCreateHannREAL8Window(fft_length(element) - 2 * zero_pad_length(element) + 1);
	if(!element->window) {
		GST_ERROR_OBJECT(element, "failure creating Hann window: %s", XLALErrorString(XLALGetBaseErrno()));
		XLALClearErrno();
		return -1;
	}
	if(!XLALResizeREAL8Sequence(element->window->data, -zero_pad_length(element), fft_length(element))) {
		GST_ERROR_OBJECT(element, "failure resizing Hann window: %s", XLALErrorString(XLALGetBaseErrno()));
		XLALDestroyREAL8Window(element->window);
		element->window = NULL;
		XLALClearErrno();
		return -1;
	}

	/*
	 * allocate a tail buffer
	 */

	XLALDestroyREAL8Sequence(element->tail);
	element->tail = XLALCreateREAL8Sequence(fft_length(element) / 2 - zero_pad_length(element));
	if(!element->tail) {
		GST_ERROR_OBJECT(element, "failure allocating tail buffer: %s", XLALErrorString(XLALGetBaseErrno()));
		XLALClearErrno();
		return -1;
	}
	memset(element->tail->data, 0, element->tail->length * sizeof(*element->tail->data));

	/*
	 * construct FFT plans
	 */

	g_mutex_lock(gstlal_fftw_lock);
	XLALDestroyREAL8FFTPlan(element->fwdplan);
	XLALDestroyREAL8FFTPlan(element->revplan);

	element->fwdplan = XLALCreateForwardREAL8FFTPlan(fft_length(element), 1);
	element->revplan = XLALCreateReverseREAL8FFTPlan(fft_length(element), 1);
	g_mutex_unlock(gstlal_fftw_lock);

	if(!element->fwdplan || !element->revplan) {
		GST_ERROR_OBJECT(element, "failure creating FFT plans: %s", XLALErrorString(XLALGetBaseErrno()));
		g_mutex_lock(gstlal_fftw_lock);
		XLALDestroyREAL8FFTPlan(element->fwdplan);
		XLALDestroyREAL8FFTPlan(element->revplan);
		g_mutex_unlock(gstlal_fftw_lock);
		element->fwdplan = NULL;
		element->revplan = NULL;
		XLALClearErrno();
		return -1;
	}

	/*
	 * construct work spaces
	 */

	XLALDestroyREAL8TimeSeries(element->tdworkspace);
	element->tdworkspace = XLALCreateREAL8TimeSeries(NULL, &GPS_ZERO, 0.0, (double) 1.0 / element->sample_rate, &element->sample_units, fft_length(element));
	if(!element->tdworkspace) {
		GST_ERROR_OBJECT(element, "failure creating time-domain workspace: %s", XLALErrorString(XLALGetBaseErrno()));
		XLALClearErrno();
		return -1;
	}
	XLALDestroyCOMPLEX16FrequencySeries(element->fdworkspace);
	element->fdworkspace = XLALCreateCOMPLEX16FrequencySeries(NULL, &GPS_ZERO, 0.0, (double) 1.0 / (element->tdworkspace->deltaT * fft_length(element)), &lalDimensionlessUnit, fft_length(element) / 2 + 1);
	if(!element->fdworkspace) {
		GST_ERROR_OBJECT(element, "failure creating frequency-domain workspace: %s", XLALErrorString(XLALGetBaseErrno()));
		XLALClearErrno();
		return -1;
	}

	/*
	 * reset PSD regressor
	 */

	XLALPSDRegressorReset(element->psd_regressor);

	/*
	 * done
	 */

	return 0;
}


static unsigned get_available_samples(GSTLALWhiten *element)
{
	return gst_adapter_available(element->adapter) / sizeof(*element->tdworkspace->data->data);
}


static REAL8FrequencySeries *make_empty_psd(double f0, double deltaF, int length, LALUnit sample_units)
{
	REAL8FrequencySeries *psd;

	sample_units = gstlal_lalUnitSquaredPerHertz(sample_units);
	psd = XLALCreateREAL8FrequencySeries("PSD", &GPS_ZERO, f0, deltaF, &sample_units, length);
	if(!psd) {
		GST_ERROR("XLALCreateREAL8FrequencySeries() failed: %s", XLALErrorString(XLALGetBaseErrno()));
		XLALClearErrno();
	}

	return psd;
}


static REAL8FrequencySeries *make_psd_from_fseries(const COMPLEX16FrequencySeries *fseries)
{
	LALUnit unit;
	REAL8FrequencySeries *psd;
	unsigned i;

	/*
	 * reconstruct the time-domain sample units from the sample units
	 * of the frequency series
	 */

	XLALUnitMultiply(&unit, &fseries->sampleUnits, &lalHertzUnit);

	/*
	 * build the PSD
	 */

	psd = make_empty_psd(fseries->f0, fseries->deltaF, fseries->data->length, unit);
	if(!psd)
		return NULL;
	for(i = 0; i < psd->data->length; i++)
		psd->data->data[i] = XLALCOMPLEX16Abs2(fseries->data->data[i]) * (2 * psd->deltaF);

	/*
	 * zero the DC and Nyquist components
	 */

	if(psd->f0 == 0)
		psd->data->data[0] = 0;
	psd->data->data[psd->data->length - 1] = 0;

	return psd;
}


static REAL8FrequencySeries *get_psd(GSTLALWhiten *element)
{
	REAL8FrequencySeries *psd;

	switch(element->psdmode) {
	case GSTLAL_PSDMODE_RUNNING_AVERAGE:
		if(!element->psd_regressor->n_samples) {
			/*
			 * No data for the average yet, seed psd regressor
			 * with current frequency series.
			 */

			psd = make_psd_from_fseries(element->fdworkspace);
			if(!psd)
				return NULL;
			if(XLALPSDRegressorSetPSD(element->psd_regressor, psd, 1)) {
				GST_ERROR_OBJECT(element, "XLALPSDRegressorSetPSD() failed: %s", XLALErrorString(XLALGetBaseErrno()));
				XLALDestroyREAL8FrequencySeries(psd);
				XLALClearErrno();
				return NULL;
			}
		} else {
			psd = XLALPSDRegressorGetPSD(element->psd_regressor);
			if(!psd) {
				GST_ERROR_OBJECT(element, "XLALPSDRegressorGetPSD() failed: %s", XLALErrorString(XLALGetBaseErrno()));
				XLALClearErrno();
				return NULL;
			}
		}
		break;

	case GSTLAL_PSDMODE_FIXED:
		psd = element->psd;
		if(!psd) {
			GST_ERROR_OBJECT(element, "mode %s requires PSD", g_enum_get_value(G_ENUM_CLASS(g_type_class_peek(GSTLAL_PSDMODE_TYPE)), GSTLAL_PSDMODE_FIXED)->value_name);
			return NULL;
		}
		break;

	default:
		psd = NULL;
		g_assert_not_reached();
	}

	psd->epoch = element->tdworkspace->epoch;

	/*
	 * done
	 */

	return psd;
}


static GstMessage *psd_message_new(GSTLALWhiten *element, REAL8FrequencySeries *psd)
{
	GValueArray *va = gstlal_g_value_array_from_doubles(psd->data->data, psd->data->length);
	char units[50];
	GstStructure *s = gst_structure_new(
		"spectrum",
		"timestamp", G_TYPE_UINT64, (guint64) XLALGPSToINT8NS(&psd->epoch),
		"delta-f", G_TYPE_DOUBLE, psd->deltaF,
		"sample-units", G_TYPE_STRING, XLALUnitAsString(units, sizeof(units), &psd->sampleUnits),
		"magnitude", G_TYPE_VALUE_ARRAY, va,
		NULL
	);
	g_value_array_free(va);

	return gst_message_new_element(GST_OBJECT(element), s);
}


static GstFlowReturn push_psd(GstPad *psd_pad, const REAL8FrequencySeries *psd)
{
	GstBuffer *buffer;
	GstFlowReturn result;
	GstCaps *caps = gst_caps_new_simple(
		"audio/x-raw-float",
		"channels", G_TYPE_INT, 1,
		"delta-f", G_TYPE_DOUBLE, psd->deltaF,
		"endianness", G_TYPE_INT, G_BYTE_ORDER,
		"width", G_TYPE_INT, 64
	);

	gst_pad_set_caps(psd_pad, caps);
	gst_caps_unref(caps);

	result = gst_pad_alloc_buffer(psd_pad, GST_BUFFER_OFFSET_NONE, psd->data->length * sizeof(*psd->data->data), GST_PAD_CAPS(psd_pad), &buffer);
	if(result != GST_FLOW_OK)
		return result;

	memcpy(GST_BUFFER_DATA(buffer), psd->data->data, GST_BUFFER_SIZE(buffer));

	GST_BUFFER_OFFSET_END(buffer) = GST_BUFFER_OFFSET_NONE;
	GST_BUFFER_TIMESTAMP(buffer) = XLALGPSToINT8NS(&psd->epoch);
	GST_BUFFER_DURATION(buffer) = GST_CLOCK_TIME_NONE;

	result = gst_pad_push(psd_pad, buffer);

	return result;
}


static void set_metadata(GSTLALWhiten *element, GstBuffer *buf, guint64 outsamples)
{
	GST_BUFFER_SIZE(buf) = outsamples * sizeof(*element->tdworkspace->data->data);
	GST_BUFFER_OFFSET(buf) = element->next_offset_out;
	element->next_offset_out += outsamples;
	GST_BUFFER_OFFSET_END(buf) = element->next_offset_out;
	GST_BUFFER_TIMESTAMP(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET(buf) - element->offset0, GST_SECOND, element->sample_rate);
	GST_BUFFER_DURATION(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET_END(buf) - element->offset0, GST_SECOND, element->sample_rate) - GST_BUFFER_TIMESTAMP(buf);
	if(element->next_is_discontinuity) {
		GST_BUFFER_FLAG_SET(buf, GST_BUFFER_FLAG_DISCONT);
		element->next_is_discontinuity = FALSE;
	}
}


static GstFlowReturn whiten(GSTLALWhiten *element, GstBuffer *outbuf)
{
	guint64 zero_pad = zero_pad_length(element);
	double *dst = (double *) GST_BUFFER_DATA(outbuf);
	unsigned block_number;

	/*
	 * Iterate over the available data
	 */

	for(block_number = 0; get_available_samples(element) >= element->tdworkspace->data->length - 2 * zero_pad; block_number++) {
		REAL8FrequencySeries *newpsd;
		unsigned i;

		/*
		 * Reset the workspace's metadata that gets modified
		 * through each iteration of this loop.
		 */

		reset_workspace_metadata(element);

		/*
		 * Copy data from adapter into time-domain workspace.  No
		 * need to explicitly zero-pad the time series because the
		 * window function will do it for us.
		 *
		 * Note:  the time series' epoch is set to the timestamp of
		 * the data taken from the adapter, not the timestamp of
		 * the start of the series (which is zero_pad samples
		 * earlier).
		 */

		memcpy(element->tdworkspace->data->data + zero_pad * sizeof(*element->tdworkspace->data->data), gst_adapter_peek(element->adapter, (element->tdworkspace->data->length - 2 * zero_pad) * sizeof(*element->tdworkspace->data->data)), (element->tdworkspace->data->length - 2 * zero_pad) * sizeof(*element->tdworkspace->data->data));
		XLALINT8NSToGPS(&element->tdworkspace->epoch, element->t0);
		XLALGPSAdd(&element->tdworkspace->epoch, (double) (element->next_offset_out - element->offset0) / element->sample_rate);

		/*
		 * Transform to frequency domain
		 */

		if(!XLALUnitaryWindowREAL8Sequence(element->tdworkspace->data, element->window)) {
			GST_ERROR_OBJECT(element, "XLALUnitaryWindowREAL8Sequence() failed: %s", XLALErrorString(XLALGetBaseErrno()));
			XLALClearErrno();
			return GST_FLOW_ERROR;
		}
		if(XLALREAL8TimeFreqFFT(element->fdworkspace, element->tdworkspace, element->fwdplan)) {
			GST_ERROR_OBJECT(element, "XLALREAL8TimeFreqFFT() failed: %s", XLALErrorString(XLALGetBaseErrno()));
			XLALClearErrno();
			return GST_FLOW_ERROR;
		}

		/*
		 * Retrieve the PSD.
		 */

		newpsd = get_psd(element);
		if(!newpsd)
			return GST_FLOW_ERROR;
		if(newpsd != element->psd) {
			XLALDestroyREAL8FrequencySeries(element->psd);
			element->psd = newpsd;
			gst_element_post_message(GST_ELEMENT(element), psd_message_new(element, element->psd));
			if(element->mean_psd_pad) {
				GstFlowReturn result = push_psd(element->mean_psd_pad, element->psd);
				if(result != GST_FLOW_OK)
					return result;
			}
		}

		/*
		 * Add frequency domain data to spectrum averager
		 */

		if(XLALPSDRegressorAdd(element->psd_regressor, element->fdworkspace)) {
			GST_ERROR_OBJECT(element, "XLALPSDRegressorAdd() failed: %s", XLALErrorString(XLALGetBaseErrno()));
			XLALClearErrno();
			return GST_FLOW_ERROR;
		}

		/*
		 * Whiten.  After this, the frequency bins should be unit
		 * variance zero mean complex Gaussian random variables.
		 * They are *not* independent random variables because the
		 * source time series data was windowed before conversion
		 * to the frequency domain.
		 */

		if(!XLALWhitenCOMPLEX16FrequencySeries(element->fdworkspace, element->psd)) {
			GST_ERROR_OBJECT(element, "XLALWhitenCOMPLEX16FrequencySeries() failed: %s", XLALErrorString(XLALGetBaseErrno()));
			XLALClearErrno();
			return GST_FLOW_ERROR;
		}

		/*
		 * Transform to time domain.
		 */

		if(XLALREAL8FreqTimeFFT(element->tdworkspace, element->fdworkspace, element->revplan)) {
			GST_ERROR_OBJECT(element, "XLALREAL8FreqTimeFFT() failed: %s", XLALErrorString(XLALGetBaseErrno()));
			XLALClearErrno();
			return GST_FLOW_ERROR;
		}

		/* 
		 * Normalize the time series.
		 *
		 * After inverse transforming the frequency series to the
		 * time domain, the variance of the time series is
		 *
		 * <x_{j}^{2}> = w_{j}^{2} / (\Delta t^{2} \sigma^{2})
		 *
		 * where \sigma^{2} is the sum-of-squares of the window
		 * function, \sigma^{2} = \sum_{j} w_{j}^{2}
		 *
		 * The time series has a j-dependent variance, but we
		 * normalize it so that the variance is 1 where w_{j} = 1
		 * in the middle of the window.
		 */

		for(i = 0; i < element->tdworkspace->data->length; i++)
			element->tdworkspace->data->data[i] *= element->tdworkspace->deltaT * sqrt(element->window->sumofsquares);
		/* normalization constant has units of seconds */
		XLALUnitMultiply(&element->tdworkspace->sampleUnits, &element->tdworkspace->sampleUnits, &lalSecondUnit);

		/*
		 * Verify the result is dimensionless.
		 */

		if(XLALUnitCompare(&lalDimensionlessUnit, &element->tdworkspace->sampleUnits)) {
			char units[100];
			XLALUnitAsString(units, sizeof(units), &element->tdworkspace->sampleUnits);
			GST_ERROR_OBJECT(element, "whitening process failed to produce dimensionless time series: result has units \"%s\"", units);
			return GST_FLOW_ERROR;
		}

		/*
		 * Copy the first half of the time series minus the
		 * zero_pad into the output buffer, removing the zero_pad
		 * from the start, and adding the contents of the tail.
		 * When we add the two time series (the first half of the
		 * piece we have just whitened and the contents of the tail
		 * buffer), we do so overlapping the Hann windows so that
		 * the sum of the windows is 1.
		 */

		for(i = 0; i < element->tail->length; i++)
			dst[i] = element->tdworkspace->data->data[zero_pad + i] + element->tail->data[i];

		/*
		 * Save the second half of time series data minus the final
		 * zero_pad in the tail
		 */

		memcpy(element->tail->data, &element->tdworkspace->data->data[zero_pad + element->tail->length], element->tail->length * sizeof(*element->tail->data));

		/*
		 * flush the adapter, advance the output pointer
		 */

		gst_adapter_flush(element->adapter, element->tail->length * sizeof(*element->tdworkspace->data->data));
		dst += element->tail->length;
	}

	/*
	 * check for no-op
	 */

	if(!block_number)
		return GST_BASE_TRANSFORM_FLOW_DROPPED;

	/*
	 * set output metadata
	 */

	set_metadata(element, outbuf, dst - (double *) GST_BUFFER_DATA(outbuf));

	/*
	 * done
	 */

	return GST_FLOW_OK;
}


/*
 * ============================================================================
 *
 *                                  Signals
 *
 * ============================================================================
 */


static void delta_f_changed(GObject *object, GParamSpec *pspec, gpointer user_data)
{
	/* FIXME:  what if this fails?  should that be indicated somehow?
	 * return non-void? */
	make_window_and_fft_plans(GSTLAL_WHITEN(object));
}


/*
 * ============================================================================
 *
 *                                  PSD Pad
 *
 * ============================================================================
 */


static GstPad *request_new_pad(GstElement *element, GstPadTemplate *template, const gchar *name)
{
	GstPad *pad = gst_pad_new_from_template(template, name);

	gst_pad_use_fixed_caps(pad);

	gst_element_add_pad(element, pad);
	gst_object_ref(pad);	/* for the reference in GSTLALWhiten */
	GSTLAL_WHITEN(element)->mean_psd_pad = pad;

	return pad;
}


static void release_pad(GstElement *element, GstPad *pad)
{
	GSTLALWhiten *whiten = GSTLAL_WHITEN(element);

	if(pad != whiten->mean_psd_pad)
		/* !?  don't know about this pad ... */
		return;

	gst_object_unref(pad);
	whiten->mean_psd_pad = NULL;
}


/*
 * ============================================================================
 *
 *                           GStreamer Boiler Plate
 *
 * ============================================================================
 */


static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SINK_NAME,
	GST_PAD_SINK,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		"audio/x-raw-float, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) 1, " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) 64"
	)
);


static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SRC_NAME,
	GST_PAD_SRC,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(
		"audio/x-raw-float, " \
		"rate = (int) [1, MAX], " \
		"channels = (int) 1, " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) 64"
	)
);


static GstStaticPadTemplate psd_factory = GST_STATIC_PAD_TEMPLATE(
	"mean-psd",
	GST_PAD_SRC,
	GST_PAD_REQUEST,
	GST_STATIC_CAPS(
		"audio/x-raw-float, " \
		"channels = (int) 1, " \
		"delta-f = (double) [0, MAX], " \
		"endianness = (int) BYTE_ORDER, " \
		"width = (int) 64"
	)
);


GST_BOILERPLATE(
	GSTLALWhiten,
	gstlal_whiten,
	GstBaseTransform,
	GST_TYPE_BASE_TRANSFORM
);


enum property {
	ARG_PSDMODE = 1,
	ARG_ZERO_PAD_SECONDS,
	ARG_FFT_LENGTH,
	ARG_AVERAGE_SAMPLES,
	ARG_MEDIAN_SAMPLES,
	ARG_DELTA_F,
	ARG_F_NYQUIST,
	ARG_MEAN_PSD
};


/*
 * ============================================================================
 *
 *                     GstBaseTransform Method Overrides
 *
 * ============================================================================
 */


/*
 * get_unit_size()
 */


static gboolean get_unit_size(GstBaseTransform *trans, GstCaps *caps, guint *size)
{
	GstStructure *str;
	gint channels;

	str = gst_caps_get_structure(caps, 0);
	if(!gst_structure_get_int(str, "channels", &channels)) {
		GST_DEBUG_OBJECT(trans, "unable to parse channels from %" GST_PTR_FORMAT, caps);
		return FALSE;
	}

	*size = sizeof(double) * channels;

	return TRUE;
}


/*
 * set_caps()
 */


static gboolean set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps)
{
	GSTLALWhiten *element = GSTLAL_WHITEN(trans);
	GstStructure *s;
	gint channels;
	gint rate;
	gboolean success = TRUE;

	/*
	 * extract the channel count and sample rate, and check that they
	 * are allowed
	 */

	s = gst_caps_get_structure(incaps, 0);
	if(!gst_structure_get_int(s, "channels", &channels)) {
		GST_DEBUG_OBJECT(element, "unable to parse channels from %" GST_PTR_FORMAT, incaps);
		success = FALSE;
	} else if(!gst_structure_get_int(s, "rate", &rate)) {
		GST_DEBUG_OBJECT(element, "unable to parse channels from %" GST_PTR_FORMAT, incaps);
		success = FALSE;
	} else if((int) round(element->fft_length_seconds * rate) & 1 || (int) round(element->zero_pad_seconds * rate) & 1) {
		GST_ERROR_OBJECT(element, "bad sample rate: FFT length and/or zero-padding is an odd number of samples (must be even)");
		success = FALSE;
	}

	/*
	 * record the sample rate, make a new Hann window, new FFT plans,
	 * and workspaces
	 */

	if(success && (rate != element->sample_rate)) {
		element->sample_rate = rate;

		/*
		 * let everybody know the PSD Nyquist has changed
		 */

		g_object_notify(G_OBJECT(trans), "f-nyquist");
	}

	/*
	 * done
	 */

	return success;
}


/*
 * event()
 *
 * FIXME:  handle flusing and eos (i.e. flush the adapter and send the last
 * bit of data downstream)
 */


static gboolean event(GstBaseTransform *trans, GstEvent *event)
{
	GSTLALWhiten *element = GSTLAL_WHITEN(trans);

	switch(GST_EVENT_TYPE(event)) {
	case GST_EVENT_TAG: {
		GstTagList *taglist;
		gchar *units;

		gst_event_parse_tag(event, &taglist);
		if(gst_tag_list_get_string(taglist, GSTLAL_TAG_UNITS, &units)) {
			/*
			 * tag list contains a units tag;  replace with
			 * equivalent of "dimensionless" before sending
			 * downstream
			 */

			LALUnit sample_units;

			if(!XLALParseUnitString(&sample_units, units)) {
				GST_ERROR_OBJECT(element, "cannot parse units \"%s\"", units);
				sample_units = lalDimensionlessUnit;
				/*
				 * re-use the event
				 */

				gst_event_ref(event);
			} else {
				gchar dimensionless_units[16];	/* argh hard-coded length = BAD BAD BAD */
				XLALUnitAsString(dimensionless_units, sizeof(dimensionless_units), &lalDimensionlessUnit);

				/*
				 * create a new event with a new taglist
				 * object (don't corrupt the original
				 * event, 'cause we don't own it)
				 */

				taglist = gst_tag_list_copy(taglist);
				/* FIXME:  gstreamer doesn't like empty strings */
				gst_tag_list_add(taglist, GST_TAG_MERGE_REPLACE, GSTLAL_TAG_UNITS, " "/*dimensionless_units*/, NULL);
				event = gst_event_new_tag(taglist);
			}

			g_free(units);
			element->sample_units = sample_units;
		} else
			/*
			 * re-use the event
			 */

			gst_event_ref(event);

		/*
		 * gst_pad_push_event() consumes the reference count
		 */

		if(element->mean_psd_pad) {
			gst_event_ref(event);
			gst_pad_push_event(element->mean_psd_pad, event);
		}
		gst_pad_push_event(GST_BASE_TRANSFORM_SRC_PAD(trans), event);

		/*
		 * don't forward the event (we did it)
		 */

		return FALSE;
	}

	default:
		/*
		 * gst_pad_push_event() consumes the reference count
		 */

		if(element->mean_psd_pad) {
			gst_event_ref(event);
			gst_pad_push_event(element->mean_psd_pad, event);
		}

		/*
		 * forward the event
		 */

		return TRUE;
	}
}


/*
 * transform_size()
 */


static gboolean transform_size(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, guint size, GstCaps *othercaps, guint *othersize)
{
	GSTLALWhiten *element = GSTLAL_WHITEN(trans);
	guint quantum = fft_length(element) / 2 - zero_pad_length(element);
	guint unit_size;
	guint other_unit_size;

	if(!get_unit_size(trans, caps, &unit_size))
		return FALSE;
	if(size % unit_size) {
		GST_DEBUG_OBJECT(element, "size not a multiple of %u", unit_size);
		return FALSE;
	}
	if(!get_unit_size(trans, othercaps, &other_unit_size))
		return FALSE;

	switch(direction) {
	case GST_PAD_SRC:
		/*
		 * just keep the sample count the same
		 */

		*othersize = (size / unit_size) * other_unit_size;
		break;

	case GST_PAD_SINK:
		/*
		 * upper bound of sample count on source pad is input
		 * sample count plus the number of samples in the adapter
		 * rounded down to an integer multiple of 1/2 the fft
		 * quantum but only if there's enough data for at least 1
		 * full fft.
		 */

		*othersize = (size / unit_size + get_available_samples(element)) / quantum;
		if(*othersize >= 2)
			*othersize = (*othersize - 1) * quantum * other_unit_size;
		else
			*othersize = 0;
		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		return FALSE;
	}

	return TRUE;
}


/*
 * transform()
 */


static GstFlowReturn transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf)
{
	GSTLALWhiten *element = GSTLAL_WHITEN(trans);
	GstFlowReturn result = GST_FLOW_OK;

	/*
	 * If the incoming buffer is a discontinuity, clear the adapter and
	 * reset the clock
	 */

	if((GST_BUFFER_OFFSET(inbuf) != element->next_offset_in) || GST_BUFFER_IS_DISCONT(inbuf)) {
		gst_adapter_clear(element->adapter);
		element->next_is_discontinuity = TRUE;
		element->t0 = GST_BUFFER_TIMESTAMP(inbuf);
		element->offset0 = GST_BUFFER_OFFSET(inbuf);
		element->next_offset_out = GST_BUFFER_OFFSET(inbuf);
	}
	element->next_offset_in = GST_BUFFER_OFFSET_END(inbuf);

	/*
	 * Push the incoming buffer into the adapter.  Process adapter
	 * contents into output buffer
	 */

	gst_buffer_ref(inbuf);	/* don't let the adapter free it */
	gst_adapter_push(element->adapter, inbuf);
	result = whiten(element, outbuf);

	/*
	 * Done
	 */

	return result;
}


/*
 * ============================================================================
 *
 *                          GObject Method Overrides
 *
 * ============================================================================
 */


/*
 * set_property()
 */


static void set_property(GObject * object, enum property id, const GValue * value, GParamSpec * pspec)
{

	GSTLALWhiten *element = GSTLAL_WHITEN(object);

	GST_OBJECT_LOCK(element);

	switch (id) {
	case ARG_PSDMODE:
		element->psdmode = g_value_get_enum(value);
		break;

	case ARG_ZERO_PAD_SECONDS: {
		double zero_pad_seconds = g_value_get_double(value);
		if(zero_pad_seconds != element->zero_pad_seconds) {
			/*
			 * set sink pad's caps to NULL to force
			 * renegotiation == check that the rate is still
			 * OK, and rebuild windows and FFT plans
			 */

			gst_pad_set_caps(GST_BASE_TRANSFORM_SINK_PAD(GST_BASE_TRANSFORM(object)), NULL);
		}
		element->zero_pad_seconds = zero_pad_seconds;
		break;
	}

	case ARG_FFT_LENGTH: {
		double fft_length_seconds = g_value_get_double(value);
		if(fft_length_seconds != element->fft_length_seconds) {
			/*
			 * set sink pad's caps to NULL to force
			 * renegotiation == check that the rate is still
			 * OK, and rebuild windows and FFT plans
			 */

			gst_pad_set_caps(GST_BASE_TRANSFORM_SINK_PAD(GST_BASE_TRANSFORM(object)), NULL);
		}
		element->fft_length_seconds = fft_length_seconds;
		break;
	}

	case ARG_AVERAGE_SAMPLES:
		XLALPSDRegressorSetAverageSamples(element->psd_regressor, g_value_get_uint(value));
		break;

	case ARG_MEDIAN_SAMPLES:
		XLALPSDRegressorSetMedianSamples(element->psd_regressor, g_value_get_uint(value));
		break;

	case ARG_DELTA_F:
	case ARG_F_NYQUIST:
		/* read-only */
		g_assert_not_reached();
		break;

	case ARG_MEAN_PSD: {
		GValueArray *va = g_value_get_boxed(value);
		REAL8FrequencySeries *psd;
		psd = make_empty_psd(0.0, 1.0 / element->fft_length_seconds, va->n_values, element->sample_units);
		gstlal_doubles_from_g_value_array(va, psd->data->data, NULL);
		if(XLALPSDRegressorSetPSD(element->psd_regressor, psd, XLALPSDRegressorGetAverageSamples(element->psd_regressor))) {
			GST_ERROR_OBJECT(element, "XLALPSDRegressorSetPSD() failed: %s", XLALErrorString(XLALGetBaseErrno()));
			XLALClearErrno();
			XLALDestroyREAL8FrequencySeries(psd);
		} else {
			XLALDestroyREAL8FrequencySeries(element->psd);
			element->psd = psd;
		}
		break;
	}
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * get_property()
 */


static void get_property(GObject * object, enum property id, GValue * value, GParamSpec * pspec)
{
	GSTLALWhiten *element = GSTLAL_WHITEN(object);

	GST_OBJECT_LOCK(element);

	switch (id) {
	case ARG_PSDMODE:
		g_value_set_enum(value, element->psdmode);
		break;

	case ARG_ZERO_PAD_SECONDS:
		g_value_set_double(value, element->zero_pad_seconds);
		break;

	case ARG_FFT_LENGTH:
		g_value_set_double(value, element->fft_length_seconds);
		break;

	case ARG_AVERAGE_SAMPLES:
		g_value_set_uint(value, XLALPSDRegressorGetAverageSamples(element->psd_regressor));
		break;

	case ARG_MEDIAN_SAMPLES:
		g_value_set_uint(value, XLALPSDRegressorGetMedianSamples(element->psd_regressor));
		break;

	case ARG_DELTA_F:
		g_value_set_double(value, 1.0 / element->fft_length_seconds);
		break;

	case ARG_F_NYQUIST:
		g_value_set_double(value, element->sample_rate / 2.0);
		break;

	case ARG_MEAN_PSD:
		if(element->psd)
			g_value_take_boxed(value, gstlal_g_value_array_from_doubles(element->psd->data->data, element->psd->data->length));
		else
			g_value_take_boxed(value, g_value_array_new(0));
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * finalize()
 */


static void finalize(GObject * object)
{
	GSTLALWhiten *element = GSTLAL_WHITEN(object);

	if(element->mean_psd_pad)
		gst_object_unref(element->mean_psd_pad);
	g_object_unref(element->adapter);
	XLALDestroyREAL8Window(element->window);
	g_mutex_lock(gstlal_fftw_lock);
	XLALDestroyREAL8FFTPlan(element->fwdplan);
	XLALDestroyREAL8FFTPlan(element->revplan);
	g_mutex_unlock(gstlal_fftw_lock);
	XLALPSDRegressorFree(element->psd_regressor);
	XLALDestroyREAL8FrequencySeries(element->psd);
	XLALDestroyREAL8TimeSeries(element->tdworkspace);
	XLALDestroyCOMPLEX16FrequencySeries(element->fdworkspace);
	XLALDestroyREAL8Sequence(element->tail);

	G_OBJECT_CLASS(parent_class)->finalize(object);
}


/*
 * base_init()
 */


static void gstlal_whiten_base_init(gpointer gclass)
{
	GstElementClass *element_class = GST_ELEMENT_CLASS(gclass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(gclass);

	gst_element_class_set_details_simple(
		element_class,
		"Whiten",
		"Filter/Audio",
		"A PSD estimator and time series whitener.",
		"Kipp Cannon <kipp.cannon@ligo.org>, Chad Hanna <channa@ligo.caltech.edu>, Drew Keppel <dkeppel@ligo.caltech.edu>"
	);

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&psd_factory));

	transform_class->get_unit_size = GST_DEBUG_FUNCPTR(get_unit_size);
	transform_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
	transform_class->event = GST_DEBUG_FUNCPTR(event);
	transform_class->transform_size = GST_DEBUG_FUNCPTR(transform_size);
	transform_class->transform = GST_DEBUG_FUNCPTR(transform);
}


/*
 * class_init()
 */


static void gstlal_whiten_class_init(GSTLALWhitenClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	element_class->request_new_pad = GST_DEBUG_FUNCPTR(request_new_pad);
	element_class->release_pad = GST_DEBUG_FUNCPTR(release_pad);

	g_object_class_install_property(
		gobject_class,
		ARG_PSDMODE,
		g_param_spec_enum(
			"psd-mode",
			"PSD mode",
			"PSD estimation mode",
			GSTLAL_PSDMODE_TYPE,
			DEFAULT_PSDMODE,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_ZERO_PAD_SECONDS,
		g_param_spec_double(
			"zero-pad",
			"Zero-padding",
			"Length of the zero-padding to include on both sides of the FFT in seconds",
			0, G_MAXDOUBLE, DEFAULT_ZERO_PAD_SECONDS,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_FFT_LENGTH,
		g_param_spec_double(
			"fft-length",
			"FFT length",
			"Total length of the FFT convolution in seconds",
			0, G_MAXDOUBLE, DEFAULT_FFT_LENGTH_SECONDS,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_AVERAGE_SAMPLES,
		g_param_spec_uint(
			"average-samples",
			"Average samples",
			"Number of FFTs used in PSD average",
			1, G_MAXUINT, DEFAULT_AVERAGE_SAMPLES,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_MEDIAN_SAMPLES,
		g_param_spec_uint(
			"median-samples",
			"Median samples",
			"Number of FFTs used in PSD median history",
			1, G_MAXUINT, DEFAULT_MEDIAN_SAMPLES,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_DELTA_F,
		g_param_spec_double(
			"delta-f",
			"Delta f",
			"PSD frequency resolution in Hz",
			0, G_MAXDOUBLE, 0,
			G_PARAM_READABLE | G_PARAM_STATIC_STRINGS
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_F_NYQUIST,
		g_param_spec_double(
			"f-nyquist",
			"Nyquist Frequency",
			"Nyquist frequency in Hz",
			0, G_MAXDOUBLE, 0,
			G_PARAM_READABLE | G_PARAM_STATIC_STRINGS
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_MEAN_PSD,
		g_param_spec_value_array(
			"mean-psd",
			"Mean PSD",
			"Mean power spectral density being used to whiten the data.  First bin is at 0 Hz, last bin is at f-nyquist, bin spacing is delta-f.",
			g_param_spec_double(
				"bin",
				"Bin",
				"Power spectral density bin",
				-G_MAXDOUBLE, G_MAXDOUBLE, 1.0,
				G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
			),
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS
		)
	);
}


/*
 * init()
 */


static void gstlal_whiten_init(GSTLALWhiten *element, GSTLALWhitenClass *klass)
{
	g_signal_connect(G_OBJECT(element), "notify::f-nyquist", G_CALLBACK(delta_f_changed), NULL);

	element->mean_psd_pad = NULL;
	element->adapter = gst_adapter_new();
	element->next_is_discontinuity = FALSE;
	element->t0 = GST_CLOCK_TIME_NONE;
	element->offset0 = GST_BUFFER_OFFSET_NONE;
	element->next_offset_in = GST_BUFFER_OFFSET_NONE;
	element->next_offset_out = GST_BUFFER_OFFSET_NONE;
	element->zero_pad_seconds = DEFAULT_ZERO_PAD_SECONDS;
	element->fft_length_seconds = DEFAULT_FFT_LENGTH_SECONDS;
	element->psdmode = DEFAULT_PSDMODE;
	element->sample_units = lalDimensionlessUnit;
	element->sample_rate = 0;
	element->window = NULL;
	element->fwdplan = NULL;
	element->revplan = NULL;
	element->psd_regressor = XLALPSDRegressorNew(DEFAULT_AVERAGE_SAMPLES, DEFAULT_MEDIAN_SAMPLES);
	element->psd = NULL;
	element->tdworkspace = NULL;
	element->fdworkspace = NULL;
	element->tail = NULL;
}

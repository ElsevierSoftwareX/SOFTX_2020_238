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
 * stuff from glib/gstreamer
 */


#include <glib.h>
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
#include <lal/LIGOLwXML.h>
#include <lal/LIGOLwXMLArray.h>
#include <lal/Units.h>
#include <lal/TFTransform.h>	/* FIXME:  remove when XLALREAL8WindowTwoPointSpectralCorrelation() migrated to fft package */


/*
 * our own stuff
 */


#include <gstlal.h>
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
			{GSTLAL_PSDMODE_REFERENCE, "GSTLAL_PSDMODE_REFERENCE", "Use reference spectrum for PSD"},
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


static void reset_workspace_metadata(GSTLALWhiten *element)
{
	element->tdworkspace->deltaT = (double) 1.0 / element->sample_rate;
	element->tdworkspace->sampleUnits = element->sample_units;
	element->fdworkspace->deltaF = (double) 1.0 / (element->tdworkspace->deltaT * element->window->data->length);
}


static int make_window_and_fft_plans(GSTLALWhiten *element)
{
	int fft_length = round(element->fft_length_seconds * element->sample_rate);
	int zero_pad = round(element->zero_pad_seconds * element->sample_rate);

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
	element->window = XLALCreateHannREAL8Window(fft_length - 2 * zero_pad + 1);
	if(!element->window) {
		GST_ERROR_OBJECT(element, "failure creating Hann window");
		XLALClearErrno();
		return -1;
	}
	if(!XLALResizeREAL8Sequence(element->window->data, -zero_pad, fft_length)) {
		GST_ERROR_OBJECT(element, "failure resizing Hann window");
		XLALDestroyREAL8Window(element->window);
		element->window = NULL;
		XLALClearErrno();
		return -1;
	}

	/*
	 * allocate a tail buffer
	 */

	XLALDestroyREAL8Sequence(element->tail);
	element->tail = XLALCreateREAL8Sequence(element->window->data->length / 2 - zero_pad);
	if(!element->tail) {
		GST_ERROR_OBJECT(element, "failure allocating tail buffer");
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

	element->fwdplan = XLALCreateForwardREAL8FFTPlan(element->window->data->length, 1);
	element->revplan = XLALCreateReverseREAL8FFTPlan(element->window->data->length, 1);
	g_mutex_unlock(gstlal_fftw_lock);

	if(!element->fwdplan || !element->revplan) {
		GST_ERROR_OBJECT(element, "failure creating FFT plans");
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
	element->tdworkspace = XLALCreateREAL8TimeSeries(NULL, &GPS_ZERO, 0.0, (double) 1.0 / element->sample_rate, &element->sample_units, element->window->data->length);
	if(!element->tdworkspace) {
		GST_ERROR_OBJECT(element, "failure creating time-domain workspace");
		XLALClearErrno();
		return -1;
	}
	XLALDestroyCOMPLEX16FrequencySeries(element->fdworkspace);
	element->fdworkspace = XLALCreateCOMPLEX16FrequencySeries(NULL, &GPS_ZERO, 0.0, (double) 1.0 / (element->tdworkspace->deltaT * element->window->data->length), &lalDimensionlessUnit, element->window->data->length / 2 + 1);
	if(!element->fdworkspace) {
		GST_ERROR_OBJECT(element, "failure creating frequency-domain workspace");
		XLALClearErrno();
		return -1;
	}

	/*
	 * done
	 */

	return 0;
}


static REAL8FrequencySeries *make_empty_psd(double f0, double deltaF, int length)
{
	LALUnit strain_squared_per_hertz = gstlal_lalStrainSquaredPerHertz();
	REAL8FrequencySeries *psd = XLALCreateREAL8FrequencySeries("PSD", &GPS_ZERO, f0, deltaF, &strain_squared_per_hertz, length);

	if(!psd) {
		GST_ERROR("XLALCreateREAL8FrequencySeries() failed");
		XLALClearErrno();
	}

	return psd;
}


static REAL8FrequencySeries *make_iligo_psd(double f0, double deltaF, int length)
{
	REAL8FrequencySeries *psd = make_empty_psd(f0, deltaF, length);
	unsigned i;

	if(!psd)
		return NULL;

	/*
	 * Use LAL's LIGO I PSD function to populate the frequency series.
	 */

	for(i = 0; i < psd->data->length; i++) {
		psd->data->data[i] = XLALLIGOIPsd(psd->f0 + i * psd->deltaF) * (2 * psd->deltaF);

		/*
		 * Replace any infs with 0.  the whiten function that
		 * applies the PSD treats a 0 in the PSD as an inf (it
		 * zeros the bin instead of allowing a floating point
		 * divide-by-zero error), so algebraically this works out.
		 * More importantly, it allows an average over time to work
		 * out (otherwise a bin at +inf stays there forever).
		 */

		if(isinf(psd->data->data[i]))
			psd->data->data[i] = 0;
	}

	/*
	 * Zero the DC and Nyquist components
	 */

	if(psd->f0 == 0)
		psd->data->data[0] = 0;
	psd->data->data[psd->data->length - 1] = 0;

	return psd;
}


static REAL8FrequencySeries *make_psd_from_fseries(const COMPLEX16FrequencySeries *fseries)
{
	REAL8FrequencySeries *psd = make_empty_psd(fseries->f0, fseries->deltaF, fseries->data->length);
	unsigned i;

	if(!psd)
		return NULL;

	for(i = 0; i < psd->data->length; i++)
		psd->data->data[i] = XLALCOMPLEX16Abs2(fseries->data->data[i]) * (2 * psd->deltaF);

	/*
	 * Zero the DC and Nyquist components
	 */

	if(psd->f0 == 0)
		psd->data->data[0] = 0;
	psd->data->data[psd->data->length - 1] = 0;

	return psd;
}


static REAL8FrequencySeries *get_psd(GSTLALWhiten *element)
{
	REAL8FrequencySeries *psd;

	/*
	 * Take this opportunity to retrieve the reference PSD.
	 */

	if(element->reference_psd_filename) {
		/*
		 * If a reference spectrum is not available, or the
		 * reference spectrum does not match the current frequency
		 * band and resolution, load a new reference spectrum.
		 */

		if(!element->reference_psd || element->reference_psd->f0 != element->fdworkspace->f0 || element->reference_psd->deltaF != element->fdworkspace->deltaF || element->reference_psd->data->length != element->fdworkspace->data->length) {
			XLALDestroyREAL8FrequencySeries(element->reference_psd);
			element->reference_psd = gstlal_get_reference_psd(element->reference_psd_filename, 0.0, element->fdworkspace->deltaF, element->fdworkspace->data->length);
			if(!element->reference_psd) {
				GST_ERROR_OBJECT(element, "gstlal_get_reference_psd() failed");
				return NULL;
			}
			GST_INFO_OBJECT(element, "loaded reference PSD from \"%s\" with %d samples at %.16g Hz resolution spanning the frequency band %.16g Hz -- %.16g Hz", element->reference_psd_filename, element->reference_psd->data->length, element->reference_psd->deltaF, element->reference_psd->f0, element->reference_psd->f0 + (element->reference_psd->data->length - 1) * element->reference_psd->deltaF);
		}

		/*
		 * If there's no data in the spectrum averager yet, use the
		 * reference spectrum to initialize it.
		 */

		if(!element->psd_regressor->n_samples) {
			if(XLALPSDRegressorSetPSD(element->psd_regressor, element->reference_psd, XLALPSDRegressorGetAverageSamples(element->psd_regressor))) {
				GST_ERROR_OBJECT(element, "XLALPSDRegressorSetPSD() failed");
				/*
				 * erase the reference PSD to force this
				 * code path to be tried again
				 */
				XLALDestroyREAL8FrequencySeries(element->reference_psd);
				element->reference_psd = NULL;
				XLALClearErrno();
				return NULL;
			}
		}
	}

	switch(element->psdmode) {
	case GSTLAL_PSDMODE_INITIAL_LIGO_SRD:
		psd = make_iligo_psd(0.0, element->fdworkspace->deltaF, element->fdworkspace->data->length);
		if(!psd)
			return NULL;
		break;

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
				GST_ERROR("XLALPSDRegressorSetPSD() failed");
				XLALDestroyREAL8FrequencySeries(psd);
				XLALClearErrno();
				return NULL;
			}
		} else {
			psd = XLALPSDRegressorGetPSD(element->psd_regressor);
			if(!psd) {
				GST_ERROR("XLALPSDRegressorGetPSD() failed");
				XLALClearErrno();
				return NULL;
			}
		}
		break;

	case GSTLAL_PSDMODE_REFERENCE:
		psd = XLALCutREAL8FrequencySeries(element->reference_psd, 0, element->reference_psd->data->length);
		break;
	}

	/*
	 * done
	 */

	return psd;
}


/*
 * ============================================================================
 *
 *                             GStreamer Element
 *
 * ============================================================================
 */


/* FIXME:  try rewriting this as a subclass of the base transform class */


/*
 * Properties
 */


enum property {
	ARG_PSDMODE = 1,
	ARG_ZERO_PAD_SECONDS,
	ARG_FFT_LENGTH,
	ARG_AVERAGE_SAMPLES,
	ARG_MEDIAN_SAMPLES,
	ARG_XML_FILENAME,
	ARG_REFERENCE_PSD
};


static void set_property(GObject * object, enum property id, const GValue * value, GParamSpec * pspec)
{

	GSTLALWhiten *element = GSTLAL_WHITEN(object);

	GST_OBJECT_LOCK(element);

	switch (id) {
	case ARG_PSDMODE:
		element->psdmode = g_value_get_enum(value);
		break;

	case ARG_ZERO_PAD_SECONDS:
		element->zero_pad_seconds = g_value_get_double(value);
		break;

	case ARG_FFT_LENGTH:
		element->fft_length_seconds = g_value_get_double(value);
		break;

	case ARG_AVERAGE_SAMPLES:
		XLALPSDRegressorSetAverageSamples(element->psd_regressor, g_value_get_uint(value));
		break;

	case ARG_MEDIAN_SAMPLES:
		XLALPSDRegressorSetMedianSamples(element->psd_regressor, g_value_get_uint(value));
		break;

	case ARG_XML_FILENAME:
		free(element->xml_filename);
		if(element->xml_stream) {
			XLALCloseLIGOLwXMLFile(element->xml_stream);
			element->xml_stream = NULL;
		}
		element->xml_filename = g_value_dup_string(value);
		if(element->xml_filename) {
			element->xml_stream = XLALOpenLIGOLwXMLFile(element->xml_filename);
			if(!element->xml_stream) {
				GST_ERROR_OBJECT(element, "XLALOpenLIGOLwXMLFile() failed");
				free(element->xml_filename);
				element->xml_filename = NULL;
				XLALClearErrno();
			}
		}
		break;

	case ARG_REFERENCE_PSD:
		/*
		 * A reload of the reference PSD occurs when the PSD
		 * filename is non-NULL and the PSD frequency series itself
		 * is NULL, so we just set it up that way to induce a
		 * reload
		 */

		XLALDestroyREAL8FrequencySeries(element->reference_psd);
		element->reference_psd = NULL;
		free(element->reference_psd_filename);
		element->reference_psd_filename = g_value_dup_string(value);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


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

	case ARG_XML_FILENAME:
		g_value_set_string(value, element->xml_filename);
		break;

	case ARG_REFERENCE_PSD:
		g_value_set_string(value, element->reference_psd_filename);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * get_caps()
 */


static GstCaps *get_caps(GstPad *pad)
{
	GSTLALWhiten *element = GSTLAL_WHITEN(gst_pad_get_parent(pad));
	GstCaps *caps, *peercaps;

	/*
	 * start by retrieving our own caps.  use get_fixed_caps_func() to
	 * avoid recursing back into this function.
	 */

	caps = gst_pad_get_fixed_caps_func(pad);

	/*
	 * now compute the intersection of the caps with the downstream
	 * peer's caps if known.
	 */

	peercaps = gst_pad_peer_get_caps(element->srcpad);
	if(peercaps) {
		GstCaps *result = gst_caps_intersect(peercaps, caps);
		gst_caps_unref(caps);
		gst_caps_unref(peercaps);
		caps = result;
	}

	/*
	 * done
	 */

	gst_object_unref(element);
	return caps;
}


/*
 * set_caps()
 */


static gboolean set_caps(GstPad *pad, GstCaps *caps)
{
	GSTLALWhiten *element = GSTLAL_WHITEN(gst_pad_get_parent(pad));
	GstStructure *structure;
	int sample_rate;
	LALUnit sample_units;
	char units[100];	/* FIXME:  argh, hard-coded length = BAD BAD BAD */
	gboolean result = TRUE;

	/*
	 * extract the sample rate, and check that it is allowed
	 */

	structure = gst_caps_get_structure(caps, 0);
	sample_rate = g_value_get_int(gst_structure_get_value(structure, "rate"));
	if((int) round(element->fft_length_seconds * sample_rate) & 1 || (int) round(element->zero_pad_seconds * sample_rate) & 1) {
		GST_ERROR_OBJECT(element, "FFT length and/or Zero-padding is an odd number of samples (must be even)");
		result = FALSE;
		goto done;
	}

	/*
	 * extract the sample units
	 */

	if(!XLALParseUnitString(&sample_units, g_value_get_string(gst_structure_get_value(structure, "units")))) {
		GST_ERROR_OBJECT(element, "cannot parse units");
		result = FALSE;
		goto done;
	}

	/*
	 * get a modifiable copy of the caps, set the caps' units to
	 * dimensionless, and try setting the new caps on the downstream
	 * peer.  if this succeeds, doing this here means we don't have to
	 * do this repeatedly in the chain function.
	 * gst_caps_make_writable() unref()s its argument so we have to
	 * ref() it first to keep it valid.
	 */

	gst_caps_ref(caps);
	caps = gst_caps_make_writable(caps);

	XLALUnitAsString(units, sizeof(units), &lalDimensionlessUnit);
	/* FIXME:  gstreamer doesn't like empty strings */
	gst_caps_set_simple(caps, "units", G_TYPE_STRING, " "/*units*/, NULL);

	result = gst_pad_set_caps(element->srcpad, caps);

	gst_caps_unref(caps);

	if(!result)
		goto done;

	/*
	 * record the sample rate and units
	 */

	element->sample_rate = sample_rate;
	element->sample_units = sample_units;

	/*
	 * make a new Hann window, new FFT plans, and workspaces
	 */

	if(make_window_and_fft_plans(element)) {
		result = FALSE;
		goto done;
	}

	/*
	 * done
	 */

done:
	gst_object_unref(element);
	return result;
}


/* FIXME:  add an event handler to handle flusing and eos (i.e. flush the
 * adapter and send the last bit of data downstream) */


/*
 * chain()
 */


static GstFlowReturn chain(GstPad *pad, GstBuffer *sinkbuf)
{
	GSTLALWhiten *element = GSTLAL_WHITEN(gst_pad_get_parent(pad));
	GstFlowReturn result = GST_FLOW_OK;
	unsigned zero_pad = round(element->zero_pad_seconds * element->sample_rate);

	/*
	 * Confirm that set_caps() has successfully configured everything
	 */

	if(!element->window || !element->tail || !element->fwdplan || !element->revplan || !element->tdworkspace || !element->fdworkspace) {
		result = GST_FLOW_NOT_NEGOTIATED;
		goto done;
	}

	/*
	 * Push the incoming buffer into the adapter.  If the buffer is a
	 * discontinuity, first clear the adapter and reset the clock
	 */

	if(GST_BUFFER_IS_DISCONT(sinkbuf)) {
		/* FIXME:  if there is tail data left over, maybe it should
		 * be pushed downstream? */
		gst_adapter_clear(element->adapter);
		element->next_is_discontinuity = TRUE;
		element->segment_start = GST_BUFFER_TIMESTAMP(sinkbuf);
		element->offset0 = GST_BUFFER_OFFSET(sinkbuf);
		element->offset = 0;
	}

	gst_adapter_push(element->adapter, sinkbuf);

	/*
	 * Iterate over the available data
	 */

	while(gst_adapter_available(element->adapter) >= element->tdworkspace->data->length * sizeof(*element->tdworkspace->data->data)) {
		REAL8FrequencySeries *newpsd;
		GstBuffer *srcbuf;
		unsigned i;

		/*
		 * Reset the workspace's metadata that gets modified
		 * through each iteration of this loop.
		 */

		reset_workspace_metadata(element);

		/*
		 * Copy data from adapter into time-domain workspace.
		 */

		memcpy(element->tdworkspace->data->data, gst_adapter_peek(element->adapter, element->tdworkspace->data->length * sizeof(*element->tdworkspace->data->data)), element->tdworkspace->data->length * sizeof(*element->tdworkspace->data->data));
		XLALINT8NSToGPS(&element->tdworkspace->epoch, element->segment_start);
		XLALGPSAdd(&element->tdworkspace->epoch, (double) element->offset / element->sample_rate);

		/*
		 * Transform to frequency domain
		 */

		if(!XLALUnitaryWindowREAL8Sequence(element->tdworkspace->data, element->window)) {
			GST_ERROR_OBJECT(element, "XLALUnitaryWindowREAL8Sequence() failed");
			result = GST_FLOW_ERROR;
			XLALClearErrno();
			goto done;
		}
		if(XLALREAL8TimeFreqFFT(element->fdworkspace, element->tdworkspace, element->fwdplan)) {
			GST_ERROR_OBJECT(element, "XLALREAL8TimeFreqFFT() failed");
			result = GST_FLOW_ERROR;
			XLALClearErrno();
			goto done;
		}

		/*
		 * Retrieve the PSD.
		 */

		newpsd = get_psd(element);
		if(!newpsd) {
			result = GST_FLOW_ERROR;
			goto done;
		}
		/* FIXME:  compare the new PSD to the old PSD and tell the
		 * world about it if it has changed according to some
		 * metric */
		XLALDestroyREAL8FrequencySeries(element->psd);
		element->psd = newpsd;
		{
		/* FIXME:  if more than one whitener is in the pipeline,
		 * this counter is probably shared between them.  bad */
		static int n = 0;
		if(!(n++ % (XLALPSDRegressorGetAverageSamples(element->psd_regressor) / 2)) && element->xml_stream) {
			GST_INFO_OBJECT(element, "writing PSD snapshot");
			if(XLALWriteLIGOLwXMLArrayREAL8FrequencySeries(element->xml_stream, "Recorded by GSTLAL element lal_whiten", element->psd)) {
				GST_ERROR_OBJECT(element, "XLALWriteLIGOLwXMLArrayREAL8FrequencySeries() failed");
				XLALClearErrno();
			}
		}
		}

		/*
		 * Add frequency domain data to spectrum averager
		 */

		if(XLALPSDRegressorAdd(element->psd_regressor, element->fdworkspace)) {
			GST_ERROR_OBJECT(element, "XLALPSDRegressorAdd() failed");
			result = GST_FLOW_ERROR;
			XLALClearErrno();
			goto done;
		}

		/*
		 * Whiten.  After this, the frequency bins should be unit
		 * variance zero mean complex Gaussian random variables.
		 * They are *not* independent random variables because the
		 * source time series data was windowed before conversion
		 * to the frequency domain.
		 */

		if(!XLALWhitenCOMPLEX16FrequencySeries(element->fdworkspace, element->psd)) {
			GST_ERROR_OBJECT(element, "XLALWhitenCOMPLEX16FrequencySeries() failed");
			result = GST_FLOW_ERROR;
			XLALClearErrno();
			goto done;
		}

		/*
		 * Transform to time domain.
		 */

		if(XLALREAL8FreqTimeFFT(element->tdworkspace, element->fdworkspace, element->revplan)) {
			GST_ERROR_OBJECT(element, "XLALREAL8FreqTimeFFT() failed");
			result = GST_FLOW_ERROR;
			XLALClearErrno();
			goto done;
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
		 * normalize it so that the variance is 1 in the middle of
		 * the window.
		 */

		for(i = 0; i < element->tdworkspace->data->length; i++)
			element->tdworkspace->data->data[i] *= element->tdworkspace->deltaT * sqrt(element->window->sumofsquares);
		XLALUnitDivide(&element->tdworkspace->sampleUnits, &element->tdworkspace->sampleUnits, &lalHertzUnit);

		/*
		 * Verify the result is dimensionless.
		 */

		if(XLALUnitCompare(&lalDimensionlessUnit, &element->tdworkspace->sampleUnits)) {
			char units[100];
			XLALUnitAsString(units, sizeof(units), &element->tdworkspace->sampleUnits);
			GST_ERROR_OBJECT(element, "whitening process failed to produce dimensionless time series: result has units \"%s\"", units);
			result = GST_FLOW_ERROR;
			goto done;
		}

		/*
		 * Get a buffer from the downstream peer.
		 */

		result = gst_pad_alloc_buffer(element->srcpad, element->offset0 + element->offset + zero_pad, element->tail->length * sizeof(*element->tdworkspace->data->data), GST_PAD_CAPS(element->srcpad), &srcbuf);
		if(result != GST_FLOW_OK)
			goto done;
		if(element->next_is_discontinuity) {
			GST_BUFFER_FLAG_SET(srcbuf, GST_BUFFER_FLAG_DISCONT);
			element->next_is_discontinuity = FALSE;
		}
		GST_BUFFER_OFFSET_END(srcbuf) = GST_BUFFER_OFFSET(srcbuf) + element->tail->length;
		GST_BUFFER_TIMESTAMP(srcbuf) = element->segment_start + gst_util_uint64_scale_int(element->offset + zero_pad, GST_SECOND, element->sample_rate);
		GST_BUFFER_DURATION(srcbuf) = gst_util_uint64_scale_int(element->offset + zero_pad + element->tail->length, GST_SECOND, element->sample_rate) - gst_util_uint64_scale_int(element->offset + zero_pad, GST_SECOND, element->sample_rate);

		/*
		 * Copy the first half of the time series into the buffer,
		 * removing the zero_pad from the start, and adding the
		 * contents of the tail.  When we add the two time series
		 * (the first half of the piece we have just whitened and
		 * the contents of the tail buffer), we do so overlapping
		 * the Hann windows so that the sum of the windows is 1.
		 */

		for(i = 0; i < element->tail->length; i++)
			((double *) GST_BUFFER_DATA(srcbuf))[i] = element->tdworkspace->data->data[zero_pad + i] + element->tail->data[i];

		/*
		 * Push the buffer downstream
		 */

		result = gst_pad_push(element->srcpad, srcbuf);
		if(result != GST_FLOW_OK)
			goto done;

		/*
		 * Save the second half of time series data minus the final
		 * zero_pad in the tail
		 */

		memcpy(element->tail->data, &element->tdworkspace->data->data[zero_pad + element->tail->length], element->tail->length * sizeof(*element->tail->data));

		/*
		 * Flush the adapter and advance the sample count and
		 * adapter clock
		 */

		gst_adapter_flush(element->adapter, element->tail->length * sizeof(*element->tdworkspace->data->data));
		element->offset += element->tail->length;
	}

	/*
	 * Done
	 */

done:
	gst_object_unref(element);
	return result;
}


/*
 * Parent class.
 */


static GstElementClass *parent_class = NULL;


/*
 * Instance finalize function.  See ???
 */


static void finalize(GObject * object)
{
	GSTLALWhiten *element = GSTLAL_WHITEN(object);

	g_object_unref(element->adapter);
	gst_object_unref(element->srcpad);
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
	free(element->xml_filename);
	XLALCloseLIGOLwXMLFile(element->xml_stream);
	free(element->reference_psd_filename);
	XLALDestroyREAL8FrequencySeries(element->reference_psd);

	G_OBJECT_CLASS(parent_class)->finalize(object);
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
		"Kipp Cannon <kcannon@ligo.caltech.edu>, Chad Hanna <channa@ligo.caltech.edu>"
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
	gobject_class->finalize = finalize;

	g_object_class_install_property(gobject_class, ARG_PSDMODE, g_param_spec_enum("psd-mode", "PSD mode", "PSD estimation mode", GSTLAL_PSDMODE_TYPE, DEFAULT_PSDMODE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	g_object_class_install_property(gobject_class, ARG_ZERO_PAD_SECONDS, g_param_spec_double("zero-pad", "Zero-padding", "Length of the zero-padding to include on both sides of the FFT in seconds", 0, G_MAXDOUBLE, DEFAULT_ZERO_PAD_SECONDS, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	g_object_class_install_property(gobject_class, ARG_FFT_LENGTH, g_param_spec_double("fft-length", "FFT length", "Total length of the FFT convolution in seconds", 0, G_MAXDOUBLE, DEFAULT_FFT_LENGTH_SECONDS, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	g_object_class_install_property(gobject_class, ARG_AVERAGE_SAMPLES, g_param_spec_uint("average-samples", "Average samples", "Number of FFTs used in PSD average", 1, G_MAXUINT, DEFAULT_AVERAGE_SAMPLES, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	g_object_class_install_property(gobject_class, ARG_MEDIAN_SAMPLES, g_param_spec_uint("median-samples", "Median samples", "Number of FFTs used in PSD median history", 1, G_MAXUINT, DEFAULT_MEDIAN_SAMPLES, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	g_object_class_install_property(gobject_class, ARG_XML_FILENAME, g_param_spec_string("xml-filename", "XML Filename", "Name of file into which will be dumped PSD snapshots (null = disable).", NULL, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	g_object_class_install_property(gobject_class, ARG_REFERENCE_PSD, g_param_spec_string("reference-psd", "Filename", "Name of text file from which to read reference spectrum (null = disable).", NULL, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
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
	gst_pad_set_getcaps_function(pad, get_caps);
	gst_pad_set_setcaps_function(pad, set_caps);
	gst_pad_set_chain_function(pad, chain);
	gst_object_unref(pad);

	/* retrieve (and ref) src pad */
	element->srcpad = gst_element_get_static_pad(GST_ELEMENT(object), "src");

	/* internal data */
	element->adapter = gst_adapter_new();
	element->next_is_discontinuity = FALSE;
	element->segment_start = 0;
	element->offset0 = 0;
	element->offset = 0;
	element->zero_pad_seconds = DEFAULT_ZERO_PAD_SECONDS;
	element->fft_length_seconds = DEFAULT_FFT_LENGTH_SECONDS;
	element->psdmode = DEFAULT_PSDMODE;
	element->sample_rate = 0;
	element->sample_units = lalDimensionlessUnit;
	element->window = NULL;
	element->fwdplan = NULL;
	element->revplan = NULL;
	element->psd_regressor = XLALPSDRegressorNew(DEFAULT_AVERAGE_SAMPLES, DEFAULT_MEDIAN_SAMPLES);
	element->psd = NULL;
	element->tdworkspace = NULL;
	element->fdworkspace = NULL;
	element->tail = NULL;
	element->xml_filename = NULL;
	element->xml_stream = NULL;
	element->reference_psd_filename = NULL;
	element->reference_psd = NULL;
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

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


/*
 * our own stuff
 */


#include <gstlal.h>
#include <gstlal_whiten.h>


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
#define DEFAULT_MEDIAN_SAMPLES 13
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


static int make_window_and_fft_plans(GSTLALWhiten *element)
{
	int fft_length = floor(element->fft_length_seconds * element->sample_rate + 0.5);
	int zero_pad = floor(element->zero_pad_seconds * element->sample_rate + 0.5);

	/*
	 * build a Hann window with zero-padding
	 */

	XLALDestroyREAL8Window(element->window);
	element->window = XLALCreateHannREAL8Window(fft_length - 2 * zero_pad);
	if(!element->window) {
		GST_ERROR_OBJECT(element, "failure creating Hann window");
		return -1;
	}
	if(!XLALResizeREAL8Sequence(element->window->data, -zero_pad, fft_length)) {
		GST_ERROR_OBJECT(element, "failure resizing Hann window");
		XLALDestroyREAL8Window(element->window);
		element->window = NULL;
		return -1;
	}

	/*
	 * allocate a tail buffer
	 */

	XLALDestroyREAL8Sequence(element->tail);
	element->tail = XLALCreateREAL8Sequence(element->window->data->length / 2 - zero_pad);
	if(!element->tail) {
		GST_ERROR_OBJECT(element, "failure allocating tail buffer");
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
		return -1;
	}

	return 0;
}


static REAL8FrequencySeries *make_empty_psd(double f0, double deltaF, int length)
{
	LIGOTimeGPS gps_zero = {0, 0};
	LALUnit strain_squared_per_hertz = gstlal_lalStrainSquaredPerHertz();
	REAL8FrequencySeries *psd = XLALCreateREAL8FrequencySeries("PSD", &gps_zero, f0, deltaF, &strain_squared_per_hertz, length);

	if(!psd)
		GST_ERROR("XLALCreateREAL8FrequencySeries() failed");

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


static REAL8FrequencySeries *get_psd(enum gstlal_psdmode_t psdmode, LALPSDRegressor *psd_regressor, const COMPLEX16FrequencySeries *fseries)
{
	REAL8FrequencySeries *psd;

	switch(psdmode) {
	case GSTLAL_PSDMODE_INITIAL_LIGO_SRD:
		psd = make_iligo_psd(fseries->f0, fseries->deltaF, fseries->data->length);
		if(!psd)
			return NULL;
		break;

	case GSTLAL_PSDMODE_RUNNING_AVERAGE:
		if(!psd_regressor->n_samples) {
			/*
			 * No data for the average yet, seed psd regressor
			 * with current frequency series.
			 */

			psd = make_psd_from_fseries(fseries);
			if(!psd)
				return NULL;
			if(XLALPSDRegressorSetPSD(psd_regressor, psd, psd_regressor->average_samples)) {
				GST_ERROR("XLALPSDRegressorSetPSD() failed");
				XLALDestroyREAL8FrequencySeries(psd);
				return NULL;
			}
		} else {
			psd = XLALPSDRegressorGetPSD(psd_regressor);
			if(!psd) {
				GST_ERROR("XLALPSDRegressorGetPSD() failed");
				return NULL;
			}
		}
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
	ARG_XML_FILENAME,
	ARG_COMPENSATION_PSD
};


static void set_property(GObject * object, enum property id, const GValue * value, GParamSpec * pspec)
{

	GSTLALWhiten *element = GSTLAL_WHITEN(object);

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
		element->psd_regressor->average_samples = g_value_get_int(value);
		break;

	case ARG_XML_FILENAME:
		free(element->xml_filename);
		element->xml_filename = g_value_dup_string(value);
		XLALCloseLIGOLwXMLFile(element->xml_stream);
		if(element->xml_filename) {
			element->xml_stream = XLALOpenLIGOLwXMLFile(element->xml_filename);
			if(!element->xml_stream) {
				GST_ERROR_OBJECT(element, "XLALOpenLIGOLwXMLFile() failed");
				free(element->xml_filename);
				element->xml_filename = NULL;
			}
		} else
			element->xml_stream = NULL;
		break;

	case ARG_COMPENSATION_PSD:
		/*
		 * A reload of the reference PSD occurs when the PSD
		 * filename is non-NULL and the PSD frequency series itself
		 * is NULL, so we just set it up that way to induce a
		 * reload
		 */

		XLALDestroyREAL8FrequencySeries(element->compensation_psd);
		element->compensation_psd = NULL;
		free(element->compensation_psd_filename);
		element->compensation_psd_filename = g_value_dup_string(value);
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

	case ARG_ZERO_PAD_SECONDS:
		g_value_set_double(value, element->zero_pad_seconds);
		break;

	case ARG_FFT_LENGTH:
		g_value_set_double(value, element->fft_length_seconds);
		break;

	case ARG_AVERAGE_SAMPLES:
		g_value_set_int(value, element->psd_regressor->average_samples);
		break;

	case ARG_XML_FILENAME:
		g_value_set_string(value, element->xml_filename);
		break;

	case ARG_COMPENSATION_PSD:
		g_value_set_string(value, element->compensation_psd_filename);
		break;
	}
}


/*
 * getcaps()
 */


static GstCaps *getcaps(GstPad *pad)
{
	GSTLALWhiten *element = GSTLAL_WHITEN(GST_PAD_PARENT(pad));
	GstCaps *caps, *peercaps;

	GST_OBJECT_LOCK(element);

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

	GST_OBJECT_UNLOCK(element);
	return caps;
}


/*
 * setcaps()
 */


static gboolean setcaps(GstPad *pad, GstCaps *caps)
{
	GSTLALWhiten *element = GSTLAL_WHITEN(gst_pad_get_parent(pad));
	int sample_rate;
	char units[100];	/* FIXME:  argh, hard-coded length = BAD BAD BAD */
	gboolean result = TRUE;

	/*
	 * extract the sample rate, and check that it is allowed
	 */

	sample_rate = g_value_get_int(gst_structure_get_value(gst_caps_get_structure(caps, 0), "rate"));

	if((int) floor(element->fft_length_seconds * sample_rate + 0.5) & 1) {
		GST_ERROR_OBJECT(element, "FFT length is an odd number of samples");
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
	gst_caps_set_simple(caps, "units", G_TYPE_STRING, units, NULL);

	result = gst_pad_set_caps(element->srcpad, caps);
	gst_caps_unref(caps);

	if(!result)
		goto done;

	/*
	 * record the sample rate
	 */

	element->sample_rate = sample_rate;

	/*
	 * make a new Hann window, new tail buffer, and new FFT plans
	 */

	if(make_window_and_fft_plans(element)) {
		result = FALSE;
		goto done;
	}

	/*
	 * erase the contents of the adapter
	 */

	gst_adapter_clear(element->adapter);

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
	unsigned zero_pad = floor(element->zero_pad_seconds * element->sample_rate + 0.5);
	REAL8TimeSeries *segment = NULL;
	COMPLEX16FrequencySeries *tilde_segment = NULL;

	/*
	 * Confirm that setcaps() has successfully configured everything
	 */

	if(!element->window || !element->tail || !element->fwdplan || !element->revplan) {
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
		element->next_sample = GST_BUFFER_OFFSET(sinkbuf);
		element->adapter_head_timestamp = GST_BUFFER_TIMESTAMP(sinkbuf);
	}

	gst_adapter_push(element->adapter, sinkbuf);

	/*
	 * Create workspace
	 */

	segment = XLALCreateREAL8TimeSeries(NULL, &(LIGOTimeGPS) {0, 0}, 0.0, (double) 1.0 / element->sample_rate, &lalStrainUnit, element->window->data->length);
	tilde_segment = XLALCreateCOMPLEX16FrequencySeries(NULL, &(LIGOTimeGPS) {0, 0}, 0, 0, &lalDimensionlessUnit, element->window->data->length / 2 + 1);
	if(!segment || !tilde_segment) {
		GST_ERROR_OBJECT(element, "failure creating workspace");
		result = GST_FLOW_ERROR;
		goto done;
	}

	/*
	 * Iterate over the available data
	 */

	while(gst_adapter_available(element->adapter) / sizeof(*segment->data->data) >= segment->data->length) {
		GstBuffer *srcbuf;
		unsigned i;

		/*
		 * Copy data from adapter into time-domain workspace.
		 * Reset the workspace's metadata that gets modified
		 * through each iteration of this loop.
		 */

		memcpy(segment->data->data, gst_adapter_peek(element->adapter, segment->data->length * sizeof(*segment->data->data)), segment->data->length * sizeof(*segment->data->data));
		segment->deltaT = (double) 1.0 / element->sample_rate;
		segment->sampleUnits = lalStrainUnit;
		XLALINT8NSToGPS(&segment->epoch, element->adapter_head_timestamp);

		/*
		 * Transform to frequency domain
		 */

		if(!XLALUnitaryWindowREAL8Sequence(segment->data, element->window)) {
			GST_ERROR_OBJECT(element, "XLALUnitaryWindowREAL8Sequence() failed");
			result = GST_FLOW_ERROR;
			goto done;
		}
		if(XLALREAL8TimeFreqFFT(tilde_segment, segment, element->fwdplan)) {
			GST_ERROR_OBJECT(element, "XLALREAL8TimeFreqFFT() failed");
			result = GST_FLOW_ERROR;
			goto done;
		}

		/*
		 * If we're compensating for a reference spectrum, make
		 * sure we have one that's up-to-date.
		 */

		if(element->compensation_psd_filename) {
			/*
			 * If a reference spectrum is already available,
			 * confirm that it matches the current frequency
			 * resolution, otherwise delete it to induce a new
			 * one to be loaded.
			 */

			if(element->compensation_psd) {
				if(element->compensation_psd->f0 != tilde_segment->f0 || element->compensation_psd->deltaF != tilde_segment->deltaF || element->compensation_psd->data->length != tilde_segment->data->length) {
					XLALDestroyREAL8FrequencySeries(element->compensation_psd);
					element->compensation_psd = NULL;
				}
			}

			/*
			 * Load a reference spectrum if one is not
			 * available.
			 */

			if(!element->compensation_psd) {
				element->compensation_psd = gstlal_get_reference_psd(element->compensation_psd_filename, tilde_segment->f0, tilde_segment->deltaF, tilde_segment->data->length);
				if(!element->compensation_psd) {
					result = GST_FLOW_ERROR;
					goto done;
				}
				GST_INFO_OBJECT(element, "loaded reference PSD from \"%s\" with %d samples at %.16g Hz resolution spanning the frequency band %.16g Hz -- %.16g Hz", element->compensation_psd_filename, element->compensation_psd->data->length, element->compensation_psd->deltaF, element->compensation_psd->f0, element->compensation_psd->f0 + (element->compensation_psd->data->length - 1) * element->compensation_psd->deltaF);
			}

			/*
			 * If there's no data in the spectrum averager yet,
			 * use the reference spectrum to initialize it.
			 */

			if(!element->psd_regressor->n_samples) {
				if(XLALPSDRegressorSetPSD(element->psd_regressor, element->compensation_psd, element->psd_regressor->average_samples)) {
					GST_ERROR_OBJECT(element, "XLALPSDRegressorSetPSD() failed");
					result = GST_FLOW_ERROR;
					goto done;
				}
			}
		}

		/*
		 * Make sure we've got an up-to-date PSD.
		 */

		if(!element->psd || element->psdmode == GSTLAL_PSDMODE_RUNNING_AVERAGE) {
			/* FIXME:  if more than one whitener is in the
			 * pipeline, this counter is probably shared
			 * between them.  bad */
			static int n = 0;
			XLALDestroyREAL8FrequencySeries(element->psd);
			element->psd = get_psd(element->psdmode, element->psd_regressor, tilde_segment);
			if(!element->psd) {
				result = GST_FLOW_ERROR;
				goto done;
			}
			if(!(n++ % (element->psd_regressor->average_samples / 2)) && element->xml_stream) {
				static int n = 1;
				GST_INFO_OBJECT(element, "writing PSD snapshot %d", n++);
				if(XLALWriteLIGOLwXMLArrayREAL8FrequencySeries(element->xml_stream, "Recorded by GSTLAL element lal_whiten", element->psd))
					GST_ERROR_OBJECT(element, "XLALWriteLIGOLwXMLArrayREAL8FrequencySeries() failed");
			}
		}

		/*
		 * Add frequency domain data to spectrum averager
		 */

		if(XLALPSDRegressorAdd(element->psd_regressor, tilde_segment)) {
			GST_ERROR_OBJECT(element, "XLALPSDRegressorAdd() failed");
			result = GST_FLOW_ERROR;
			goto done;
		}

		/*
		 * Remove lines and whiten.  After this, the frequency bins
		 * should be unit variance zero mean complex Gaussian
		 * random variables.  They are *not* independent random
		 * variables because the source time series data was
		 * windowed before conversion to the frequency domain.
		 */

		if(!XLALWhitenCOMPLEX16FrequencySeries(tilde_segment, element->psd)) {
			GST_ERROR_OBJECT(element, "XLALWhitenCOMPLEX16FrequencySeries() failed");
			result = GST_FLOW_ERROR;
			goto done;
		}

		/*
		 * Compensate for the use of a mis-matched PSD for
		 * whitening the templates in a subsequent matched-filter
		 * stage.  Given data h(f), template s(f), and PSD S(f),
		 * the inner product of the data with the template is
		 *
		 * 	    h(f)     s*(f)
		 * 	---------------------
		 * 	sqrt(S(f)) sqrt(S(f))
		 *
		 * which we can write as
		 *
		 * 	(    h(f)      sqrt(W(f)) )     s*(f)
		 * 	( ---------- * ---------- ) * ----------
		 * 	( sqrt(S(f))   sqrt(S(f)) )   sqrt(W(f))
		 *
		 * for some approximate PSD W(f).  Our frequency series
		 * contains h(f)/sqrt(S(f)), and we now multiply by the
		 * additional factor of sqrt(W(f) / S(f)) to provide the
		 * compensation required for s*(f) being divided by the
		 * wrong PSD.
		 *
		 * There is a complication in that s(f) is normalized to
		 * the PSD by requiring that |s(f)|^2/S(f) = 1, but if W(f)
		 * is used to normalize s(f) instead of the correct PSD,
		 * then the overall normalization of s(f) will be
		 * incorrect.  We adjust for that here as well by dividing
		 * by the RMS of sqrt(W(f)/S(f)).  This correction factor
		 * is only exact when s(f) itself has a flat spectrum, if
		 * s(f) samples the spectrum non-uniformly then the
		 * normalization adjustment factor will be wrong.  However,
		 * if W(f) is a good approximation of S(f) then the effect
		 * will be small, and the approximate normalization
		 * correction factor will be close to the exact value.
		 *
		 * (I don't really know if that last bit is true, but when
		 * W(f) is horribly different from S(f), for example when 1
		 * is used for W(f), then applying the approximate
		 * correction is much better than not applying it at all
		 * because then at least the SNR comes out close to the
		 * correct order of magnitude).
		 */

		if(element->compensation_psd) {
			double rms = 0;
			for(i = 0; i < tilde_segment->data->length; i++) {
				if(element->psd->data->data[i] == 0) {
					rms += 1;
					tilde_segment->data->data[i] = LAL_COMPLEX16_ZERO;
				} else {
					double psd_ratio = element->compensation_psd->data->data[i] / element->psd->data->data[i];
					rms += psd_ratio;
					tilde_segment->data->data[i] = XLALCOMPLEX16MulReal(tilde_segment->data->data[i], sqrt(psd_ratio));
				}
			}
			rms = sqrt(rms / tilde_segment->data->length);
			GST_LOG_OBJECT(element, "PSD compensation filter's RMS = %.16g\n", rms);
			for(i = 0; i < tilde_segment->data->length; i++)
				tilde_segment->data->data[i] = XLALCOMPLEX16MulReal(tilde_segment->data->data[i], 1.0 / rms);
		}

		/*
		 * Transform to time domain.
		 */

		if(XLALREAL8FreqTimeFFT(segment, tilde_segment, element->revplan)) {
			GST_ERROR_OBJECT(element, "XLALREAL8FreqTimeFFT() failed");
			result = GST_FLOW_ERROR;
			goto done;
		}

		/* 
		 * Divide by \Delta F \sqrt{N} to yield a time series of
		 * unit variance zero mean Gaussian random variables.
		 *
		 * Note: that because the original time series had been
		 * windowed (and the frequency components rendered
		 * non-indepedent as a result) the time series here still
		 * retains the shape of the original window.  The mean
		 * square is not only an ensemble average but an average
		 * over the segment.  The mean square for any given sample
		 * in the segment can be computed from the window function
		 * knowing that the average over the segment is 1, and is
		 *
		 * <x_{j}^{2}> = w_{j}^2 * (N / sum-of-squares)
		 *
		 * where N is the length of the window (and the segment)
		 * and sum-of-squares is the sum of the squares of the
		 * window.
		 *
		 * Also note that the result will *not*, in general, be a
		 * unit variance random process if the PSD compensation
		 * filter was applied for that implies a normalization
		 * adjustment.
		 *
		 * FIXME:  the normalization factor used here is only
		 * correct when the frequency bins are independent random
		 * variables, which they aren't.  The correct normalization
		 * requires making use of the two-point spectral covariance
		 * function which is derived from the Fourier transform of
		 * the Hann-like window applied to the data.
		 */

		for(i = 0; i < segment->data->length; i++)
			segment->data->data[i] /= tilde_segment->deltaF * sqrt(segment->data->length);
		XLALUnitDivide(&segment->sampleUnits, &segment->sampleUnits, &lalHertzUnit);

		/*
		 * Verify the result is dimensionless.
		 */

		if(XLALUnitCompare(&lalDimensionlessUnit, &segment->sampleUnits)) {
			char units[100];
			XLALUnitAsString(units, sizeof(units), &segment->sampleUnits);
			GST_ERROR_OBJECT(element, "whitening process failed to produce dimensionless time series: result has units \"%s\"", units);
			result = GST_FLOW_ERROR;
			goto done;
		}

		/*
		 * Get a buffer from the downstream peer.
		 */

		result = gst_pad_alloc_buffer(element->srcpad, element->next_sample + zero_pad, element->tail->length * sizeof(*segment->data->data), GST_PAD_CAPS(element->srcpad), &srcbuf);
		if(result != GST_FLOW_OK)
			goto done;
		if(element->next_is_discontinuity) {
			GST_BUFFER_FLAG_SET(srcbuf, GST_BUFFER_FLAG_DISCONT);
			element->next_is_discontinuity = FALSE;
		}
		GST_BUFFER_OFFSET_END(srcbuf) = GST_BUFFER_OFFSET(srcbuf) + element->tail->length;
		GST_BUFFER_TIMESTAMP(srcbuf) = element->adapter_head_timestamp + gst_util_uint64_scale_int(zero_pad, GST_SECOND, element->sample_rate);
		GST_BUFFER_DURATION(srcbuf) = gst_util_uint64_scale_int(element->tail->length, GST_SECOND, element->sample_rate);

		/*
		 * Copy the first half of the time series into the buffer,
		 * removing the zero_pad from the start, and adding the
		 * contents of the tail.  We want the result to be a unit
		 * variance random process.  When we add the two time
		 * series (the first half of the piece we have just
		 * whitened and the contents of the tail buffer), we do so
		 * overlapping the Hann windows so that the sum of the
		 * windows is 1.  This leaves the mean square of the
		 * samples equal to N / sum-of-squares (see the comments
		 * above about the sample-to-sample variation of the mean
		 * square).  We remove this factor leaving us with a unit
		 * variance random process.
		 */

		for(i = 0; i < element->tail->length; i++)
			((double *) GST_BUFFER_DATA(srcbuf))[i] = (segment->data->data[zero_pad + i] + element->tail->data[i]) / sqrt(element->window->data->length / element->window->sumofsquares);

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

		memcpy(element->tail->data, &segment->data->data[zero_pad + i], element->tail->length * sizeof(*element->tail->data));

		/*
		 * Flush the adapter and advance the sample count and
		 * adapter clock
		 */

		gst_adapter_flush(element->adapter, element->tail->length * sizeof(*segment->data->data));
		element->next_sample += element->tail->length;
		/* FIXME:  this accumulates round-off, the time stamp
		 * should be calculated directly somehow */
		element->adapter_head_timestamp += gst_util_uint64_scale_int(element->tail->length, GST_SECOND, element->sample_rate);
	}

	/*
	 * Done
	 */

done:
	XLALDestroyREAL8TimeSeries(segment);
	XLALDestroyCOMPLEX16FrequencySeries(tilde_segment);
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
	XLALDestroyREAL8Sequence(element->tail);
	free(element->xml_filename);
	XLALCloseLIGOLwXMLFile(element->xml_stream);
	free(element->compensation_psd_filename);
	XLALDestroyREAL8FrequencySeries(element->compensation_psd);

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
	g_object_class_install_property(gobject_class, ARG_AVERAGE_SAMPLES, g_param_spec_int("average-samples", "Average samples", "Number of FFTs used in PSD average", 1, G_MAXINT, DEFAULT_AVERAGE_SAMPLES, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	g_object_class_install_property(gobject_class, ARG_XML_FILENAME, g_param_spec_string("xml-filename", "XML Filename", "Name of file into which will be dumped PSD snapshots (null = disable).", NULL, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	g_object_class_install_property(gobject_class, ARG_COMPENSATION_PSD, g_param_spec_string("compensation-psd", "Filename", "Name of text file from which to read reference spectrum to be compensated for by over-whitening (null = disable).", NULL, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
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
	gst_pad_set_getcaps_function(pad, getcaps);
	gst_pad_set_setcaps_function(pad, setcaps);
	gst_pad_set_chain_function(pad, chain);
	gst_object_unref(pad);

	/* retrieve (and ref) src pad */
	element->srcpad = gst_element_get_static_pad(GST_ELEMENT(object), "src");

	/* internal data */
	element->adapter = gst_adapter_new();
	element->next_is_discontinuity = FALSE;
	element->next_sample = 0;
	element->adapter_head_timestamp = 0;
	element->zero_pad_seconds = DEFAULT_ZERO_PAD_SECONDS;
	element->fft_length_seconds = DEFAULT_FFT_LENGTH_SECONDS;
	element->psdmode = DEFAULT_PSDMODE;
	element->sample_rate = 0;
	element->window = NULL;
	element->fwdplan = NULL;
	element->revplan = NULL;
	element->psd_regressor = XLALPSDRegressorNew(DEFAULT_AVERAGE_SAMPLES, DEFAULT_MEDIAN_SAMPLES);
	element->psd = NULL;
	element->tail = NULL;
	element->xml_filename = NULL;
	element->xml_stream = NULL;
	element->compensation_psd_filename = NULL;
	element->compensation_psd = NULL;
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

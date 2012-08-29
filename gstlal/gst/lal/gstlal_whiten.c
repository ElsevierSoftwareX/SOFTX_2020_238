/*
 * PSD Estimation and whitener
 *
 * Copyright (C) 2008-2011  Kipp Cannon, Chad Hanna, Drew Keppel
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
#include <lal/TFTransform.h>


/*
 * our own stuff
 */


#include <gstlal.h>
#include <gstaudioadapter.h>
#include <gstlal_tags.h>
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


static guint fft_length(const GSTLALWhiten *element)
{
	return round(element->fft_length_seconds * element->sample_rate);
}


static guint zero_pad_length(const GSTLALWhiten *element)
{
	return round(element->zero_pad_seconds * element->sample_rate);
}


static guint get_available_samples(GSTLALWhiten *element)
{
	guint size;

	g_object_get(element->input_queue, "size", &size, NULL);

	return size;
}


/*
 * work space
 */


static void zero_output_history(GSTLALWhiten *element)
{
	if(element->output_history)
		memset(element->output_history->data, 0, element->output_history->length * sizeof(*element->output_history->data));
	element->nonzero_output_history_length = 0;
	element->output_history_offset = element->next_offset_out - zero_pad_length(element);
}


static int make_workspace(GSTLALWhiten *element)
{
	REAL8Window *hann_window = NULL;
	REAL8Window *tukey_window = NULL;
	REAL8FFTPlan *fwdplan = NULL;
	REAL8FFTPlan *revplan = NULL;
	REAL8TimeSeries *tdworkspace = NULL;
	COMPLEX16FrequencySeries *fdworkspace = NULL;
	REAL8Sequence *output_history = NULL;

	/*
	 * safety checks
	 */

	g_assert_cmpint(element->sample_rate, >, 0);
	g_assert_cmpfloat(element->zero_pad_seconds, >=, 0);
	g_assert_cmpfloat(element->fft_length_seconds, >, 0);
	g_assert_cmpuint(fft_length(element), >, 2 * zero_pad_length(element));

	/*
	 * construct FFT plans and build a Hann window with zero-padding.
	 * both fft_length and zero_pad are an even number of samples
	 * (enforced in the caps negotiation phase).  we need a Hann window
	 * with an odd number of samples so that there is a middle sample
	 * (= 1) to overlap the end sample (= 0) of the next window.  we
	 * achieve this by adding 1 to the length of the envelope, and then
	 * clipping the last sample.  the result is a sequence of windows
	 * that fit together as shown below:
	 *
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

	if(fft_length(element) & 1) {
		GST_ERROR_OBJECT(element, "bad sample rate: FFT length is an odd number of samples, must be even");
		goto error;
	}
	if((element->zero_pad_seconds > 0) && zero_pad_length(element) % (fft_length(element) / 2 - zero_pad_length(element))) {
		GST_ERROR_OBJECT(element, "bad zero-pad length:  must be a multiple of 1/2 the non-zero-pad length");
		goto error;
	}

	hann_window = XLALCreateHannREAL8Window(fft_length(element) - 2 * zero_pad_length(element) + 1);
	if(!hann_window) {
		GST_ERROR_OBJECT(element, "failure creating Hann window: %s", XLALErrorString(XLALGetBaseErrno()));
		goto error;
	}
	if(!XLALResizeREAL8Sequence(hann_window->data, -zero_pad_length(element), fft_length(element))) {
		GST_ERROR_OBJECT(element, "failure resizing Hann window: %s", XLALErrorString(XLALGetBaseErrno()));
		goto error;
	}

	/*
	 * build FFT plans
	 */

	g_static_mutex_lock(gstlal_fftw_lock);
	fwdplan = XLALCreateForwardREAL8FFTPlan(fft_length(element), 1);
	revplan = XLALCreateReverseREAL8FFTPlan(fft_length(element), 1);
	g_static_mutex_unlock(gstlal_fftw_lock);
	if(!fwdplan || !revplan) {
		GST_ERROR_OBJECT(element, "failure creating FFT plans: %s", XLALErrorString(XLALGetBaseErrno()));
		goto error;
	}

	/*
	 * construct work space vectors
	 */

	tdworkspace = XLALCreateREAL8TimeSeries(NULL, &GPS_ZERO, 0.0, (double) 1.0 / element->sample_rate, &element->sample_units, fft_length(element));
	if(!tdworkspace) {
		GST_ERROR_OBJECT(element, "failure creating time-domain workspace vector: %s", XLALErrorString(XLALGetBaseErrno()));
		goto error;
	}
	fdworkspace = XLALCreateCOMPLEX16FrequencySeries(NULL, &GPS_ZERO, 0.0, (double) 1.0 / (tdworkspace->deltaT * fft_length(element)), &lalDimensionlessUnit, fft_length(element) / 2 + 1);
	if(!fdworkspace) {
		GST_ERROR_OBJECT(element, "failure creating frequency-domain workspace vector: %s", XLALErrorString(XLALGetBaseErrno()));
		goto error;
	}

	/*
	 * allocate an output history buffer
	 */

	output_history = XLALCreateREAL8Sequence(fft_length(element));
	if(!output_history) {
		GST_ERROR_OBJECT(element, "failure allocating output history buffer: %s", XLALErrorString(XLALGetBaseErrno()));
		goto error;
	}
	if(zero_pad_length(element)) {
		tukey_window = XLALCreateTukeyREAL8Window(fft_length(element), 2.0 * zero_pad_length(element) / fft_length(element));
		if(!tukey_window) {
			GST_ERROR_OBJECT(element, "failure creating Tukey window: %s", XLALErrorString(XLALGetBaseErrno()));
			goto error;
		}
	}

	/*
	 * done
	 */

	element->hann_window = hann_window;
	element->tukey_window = tukey_window;
	element->fwdplan = fwdplan;
	element->revplan = revplan;
	element->tdworkspace = tdworkspace;
	element->fdworkspace = fdworkspace;
	element->output_history = output_history;
	/* this is really just for safety;  this function must be called
	 * again after next_offset_out is initialized at the start of the
	 * stream */
	zero_output_history(element);
	g_object_notify(G_OBJECT(element), "sigma-squared");
	g_object_notify(G_OBJECT(element), "spectral-correlation");
	return 0;

error:
	XLALDestroyREAL8Window(hann_window);
	XLALDestroyREAL8Window(tukey_window);
	g_static_mutex_lock(gstlal_fftw_lock);
	XLALDestroyREAL8FFTPlan(fwdplan);
	XLALDestroyREAL8FFTPlan(revplan);
	g_static_mutex_unlock(gstlal_fftw_lock);
	XLALDestroyREAL8TimeSeries(tdworkspace);
	XLALDestroyCOMPLEX16FrequencySeries(fdworkspace);
	XLALDestroyREAL8Sequence(output_history);
	XLALClearErrno();
	return -1;
}


static void reset_workspace_metadata(GSTLALWhiten *element)
{
	element->tdworkspace->deltaT = (double) 1.0 / element->sample_rate;
	element->tdworkspace->sampleUnits = element->sample_units;
	element->fdworkspace->deltaF = (double) 1.0 / (element->tdworkspace->deltaT * fft_length(element));
}


static void free_workspace(GSTLALWhiten *element)
{
	XLALDestroyREAL8Window(element->hann_window);
	element->hann_window = NULL;
	XLALDestroyREAL8Window(element->tukey_window);
	element->tukey_window = NULL;
	g_static_mutex_lock(gstlal_fftw_lock);
	XLALDestroyREAL8FFTPlan(element->fwdplan);
	element->fwdplan = NULL;
	XLALDestroyREAL8FFTPlan(element->revplan);
	element->revplan = NULL;
	g_static_mutex_unlock(gstlal_fftw_lock);
	XLALDestroyREAL8TimeSeries(element->tdworkspace);
	element->tdworkspace = NULL;
	XLALDestroyCOMPLEX16FrequencySeries(element->fdworkspace);
	element->fdworkspace = NULL;
	XLALDestroyREAL8Sequence(element->output_history);
	element->output_history = NULL;
}


/*
 * psd-related
 */


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


/*
 * make the PSD corresponding to zero-mean unit-variance Gaussian noise.
 * LAL's normalization is such that the integral of the PSD yields the
 * variance in the time domain, therefore PSD = 1 / (n \Delta f).
 */

#if 0	/* not used, commented out to silence build warning */
static REAL8FrequencySeries *make_unit_psd(double f0, double deltaF, int length, LALUnit sample_units)
{
	REAL8FrequencySeries *psd = make_empty_psd(f0, deltaF, length, sample_units);
	double f_nyquist = f0 + length + deltaF;
	int n = round(f_nyquist / deltaF) + 1;
	unsigned i;

	if(psd)
		return NULL;

	for(i = 0; i < psd->data->length; i++)
		psd->data->data[i] = 1 / (n * deltaF);

	return psd;
}
#endif


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
		psd->data->data[i] = pow(cabs(fseries->data->data[i]), 2) * (2 * psd->deltaF);

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
			 * No data for the average yet, fake a PSD with
			 * current frequency series.
			 */

			psd = make_psd_from_fseries(element->fdworkspace);
			if(!psd)
				return NULL;
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
		g_assert_not_reached();
		return NULL;
	}

	/*
	 * place the epoch at the mid-point of the interval of data used to
	 * produce the PSD.  some rounding is done to help ensure the
	 * calculation corresponds to an integer number of samples
	 */

	psd->epoch = element->tdworkspace->epoch;
	XLALGPSAdd(&psd->epoch, round(element->sample_rate / psd->deltaF) / 2 / element->sample_rate);

	/*
	 * done
	 */

	return psd;
}


/*
 * psd I/O
 */


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


static GstFlowReturn push_psd(GstPad *psd_pad, const REAL8FrequencySeries *psd, gint period_N, gint period_D)
{
	GstBuffer *buffer = NULL;
	GstFlowReturn result;
	GstCaps *caps = gst_caps_new_simple(
		"audio/x-raw-float",
		"channels", G_TYPE_INT, 1,
		"delta-f", G_TYPE_DOUBLE, psd->deltaF,
		"endianness", G_TYPE_INT, G_BYTE_ORDER,
		"width", G_TYPE_INT, 64,
		"rate", GST_TYPE_FRACTION, period_D, period_N,
		NULL
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


/*
 * output buffer
 */


static void set_metadata(GSTLALWhiten *element, GstBuffer *buf, guint64 outsamples, gboolean is_gap)
{
	GST_BUFFER_SIZE(buf) = outsamples * sizeof(*element->tdworkspace->data->data);
	GST_BUFFER_OFFSET(buf) = element->next_offset_out;
	element->next_offset_out += outsamples;
	GST_BUFFER_OFFSET_END(buf) = element->next_offset_out;
	GST_BUFFER_TIMESTAMP(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET(buf) - element->offset0, GST_SECOND, element->sample_rate);
	GST_BUFFER_DURATION(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET_END(buf) - element->offset0, GST_SECOND, element->sample_rate) - GST_BUFFER_TIMESTAMP(buf);
	if(element->need_discont) {
		GST_BUFFER_FLAG_SET(buf, GST_BUFFER_FLAG_DISCONT);
		element->need_discont = FALSE;
	}
	if(is_gap)
		GST_BUFFER_FLAG_SET(buf, GST_BUFFER_FLAG_GAP);
	else
		GST_BUFFER_FLAG_UNSET(buf, GST_BUFFER_FLAG_GAP);
}


static GstFlowReturn whiten(GSTLALWhiten *element, GstBuffer *outbuf, guint *outsamples, gboolean *output_is_gap)
{
	guint zero_pad = zero_pad_length(element);
	guint hann_length = fft_length(element) - 2 * zero_pad;
	double *dst = (double *) GST_BUFFER_DATA(outbuf);
	guint dst_size = GST_BUFFER_SIZE(outbuf) / sizeof(*dst);

	/*
	 * safety checks
	 */

	g_assert(element->tdworkspace != NULL);
	g_assert_cmpuint(element->tdworkspace->data->length, ==, fft_length(element));
	g_assert_cmpuint(element->hann_window->data->length, ==, element->tdworkspace->data->length);
	g_assert_cmpuint(element->output_history->length, ==, element->tdworkspace->data->length);
	g_assert_cmpuint(sizeof(*element->output_history->data), ==, sizeof(*element->tdworkspace->data->data));
	g_assert_cmpuint(sizeof(*element->output_history->data), ==, sizeof(*dst));

	/*
	 * Iterate over the available data
	 */

	*output_is_gap = TRUE;
	for(*outsamples = 0; get_available_samples(element) >= hann_length;) {
		REAL8FrequencySeries *newpsd;
		gboolean block_contains_gaps;
		gboolean block_contains_nongaps;
		unsigned i;

		/*
		 * safety checks
		 */

		g_assert_cmpuint((*outsamples + hann_length / 2) * sizeof(*element->tdworkspace->data->data), <=, GST_BUFFER_SIZE(outbuf));

		/*
		 * Reset the workspace's metadata that gets modified
		 * through each iteration of this loop.
		 */

		reset_workspace_metadata(element);

		/*
		 * Copy data from input queue into time-domain workspace.
		 * No need to explicitly zero-pad the time series because
		 * the window function will do it for us.
		 *
		 * Note:  the workspace's epoch is set to the timestamp of
		 * the workspace's first sample, not the first sample of
		 * the data taken from the input queue (which is zero_pad
		 * samples later).
		 */

		gst_audioadapter_copy(element->input_queue, &element->tdworkspace->data->data[zero_pad], hann_length, &block_contains_gaps, &block_contains_nongaps);
		XLALINT8NSToGPS(&element->tdworkspace->epoch, element->t0);
		XLALGPSAdd(&element->tdworkspace->epoch, (double) ((gint64) (element->next_offset_out + *outsamples - element->offset0) - (gint64) zero_pad) / element->sample_rate);

		/*
		 * Apply (zero-padded) Hann window.
		 */

		/*{ unsigned kk; double s = 0; for(kk = 0; kk < element->tdworkspace->data->length; kk++) s += pow(element->tdworkspace->data->data[kk], 2); fprintf(stderr, "mean square before window = %.16g\n", s / kk); }*/
		if(!XLALUnitaryWindowREAL8Sequence(element->tdworkspace->data, element->hann_window)) {
			GST_ERROR_OBJECT(element, "XLALUnitaryWindowREAL8Sequence() failed: %s", XLALErrorString(XLALGetBaseErrno()));
			XLALClearErrno();
			return GST_FLOW_ERROR;
		}
		/*{ unsigned kk; double s = 0; for(kk = 0; kk < element->tdworkspace->data->length; kk++) s += pow(element->tdworkspace->data->data[kk], 2); fprintf(stderr, "mean square after window = %.16g\n", s / kk); }*/

		/*
		 * The next steps can be skipped if all we have are zeros
		 */

		if(block_contains_nongaps && !(element->expand_gaps && block_contains_gaps)) {
			/*
			 * Transform to frequency domain
			 */

			if(XLALREAL8TimeFreqFFT(element->fdworkspace, element->tdworkspace, element->fwdplan)) {
				GST_ERROR_OBJECT(element, "XLALREAL8TimeFreqFFT() failed: %s", XLALErrorString(XLALGetBaseErrno()));
				XLALClearErrno();
				return GST_FLOW_ERROR;
			}
			/*{ unsigned kk; double s = 0; for(kk = 0; kk < element->fdworkspace->data->length; kk++) s += pow(cabs(element->fdworkspace->data->data[kk]), 2); fprintf(stderr, "mean square after FFT = %.16g\n", s / kk); }*/

			/*
			 * Retrieve the PSD.
			 */

			newpsd = get_psd(element);
			if(!newpsd)
				return GST_FLOW_ERROR;
			if(newpsd != element->psd) {
				XLALDestroyREAL8FrequencySeries(element->psd);
				element->psd = newpsd;

				/*
				 * let everybody know about the new PSD:  gobject's
				 * notify mechanism, post a gst message on the
				 * message bus, and push a buffer containing the
				 * PSD out the psd pad.
				 */

				g_object_notify(G_OBJECT(element), "mean-psd");
				gst_element_post_message(GST_ELEMENT(element), psd_message_new(element, element->psd));
				if(element->mean_psd_pad) {
					/* fft_length is sure to be even */
					GstFlowReturn result = push_psd(element->mean_psd_pad, element->psd, fft_length(element) / 2 - zero_pad_length(element), element->sample_rate);
					if(result != GST_FLOW_OK)
						return result;
				}
			}

			/*
			 * Add frequency domain data to spectrum averager
			 * if not contaminated by gaps
			 */

			if(!block_contains_gaps)
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
			/*{ unsigned kk; double s = 0; for(kk = 0; kk < element->fdworkspace->data->length; kk++) s += pow(cabs(element->fdworkspace->data->data[kk]), 2); fprintf(stderr, "mean square after whiten = %.16g\n", s / kk); }*/

			/*
			 * Transform to time domain.
			 */

			if(XLALREAL8FreqTimeFFT(element->tdworkspace, element->fdworkspace, element->revplan)) {
				GST_ERROR_OBJECT(element, "XLALREAL8FreqTimeFFT() failed: %s", XLALErrorString(XLALGetBaseErrno()));
				XLALClearErrno();
				return GST_FLOW_ERROR;
			}
			/*{ unsigned kk; double s = 0; for(kk = 0; kk < element->tdworkspace->data->length; kk++) s += pow(element->tdworkspace->data->data[kk], 2); fprintf(stderr, "mean square after IFFT = %.16g\n", s / kk); }*/

			/* 
			 * Normalize the time series.
			 *
			 * After inverse transforming the frequency series
			 * to the time domain, the variance of the time
			 * series is
			 *
			 * <x_{j}^{2}> = w_{j}^{2} / (\Delta t^{2}
			 * \sigma^{2})
			 *
			 * where \sigma^{2} is the sum-of-squares of the
			 * window function, \sigma^{2} = \sum_{j} w_{j}^{2}
			 *
			 * The time series has a j-dependent variance, but
			 * we normalize it so that the variance is 1 where
			 * w_{j} = 1 in the middle of the window.
			 */

			for(i = 0; i < element->tdworkspace->data->length; i++)
				element->tdworkspace->data->data[i] *= element->tdworkspace->deltaT * sqrt(element->hann_window->sumofsquares);
			/* normalization constant has units of seconds */
			XLALUnitMultiply(&element->tdworkspace->sampleUnits, &element->tdworkspace->sampleUnits, &lalSecondUnit);
			/*{ unsigned kk; double s = 0; for(kk = 0; kk < element->tdworkspace->data->length; kk++) s += pow(element->tdworkspace->data->data[kk], 2); fprintf(stderr, "mean square after normalization = %.16g\n", s / kk); }*/

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
			 * Mix the results into the output history buffer
			 */

			if(element->tukey_window) {
				for(i = 0; i < element->output_history->length; i++)
					element->output_history->data[i] += element->tdworkspace->data->data[i] * element->tukey_window->data->data[i];
			} else {
				for(i = 0; i < element->output_history->length; i++)
					element->output_history->data[i] += element->tdworkspace->data->data[i];
			}
			element->nonzero_output_history_length = element->output_history->length;
		} else
			element->tdworkspace->sampleUnits = lalDimensionlessUnit;

		/*
		 * copy new result into output buffer, shift output history
		 * buffer.
		 */

		g_assert_cmpint((gint64) element->output_history_offset, <=, (gint64) (element->next_offset_out + *outsamples));
		if(element->output_history_offset == element->next_offset_out + *outsamples) {
			g_assert_cmpuint(*outsamples + hann_length / 2, <=, dst_size);
			memcpy(&dst[*outsamples], &element->output_history->data[0], hann_length / 2 * sizeof(*element->output_history->data));
			*outsamples += hann_length / 2;
			if(element->nonzero_output_history_length != 0)
				*output_is_gap = FALSE;
		}
		memmove(&element->output_history->data[0], &element->output_history->data[hann_length / 2], (element->output_history->length - hann_length / 2) * sizeof(*element->output_history->data));
		memset(&element->output_history->data[element->output_history->length - hann_length / 2], 0, hann_length / 2 * sizeof(*element->output_history->data));
		element->output_history_offset += hann_length / 2;
		element->nonzero_output_history_length = element->nonzero_output_history_length > hann_length / 2 ? element->nonzero_output_history_length - hann_length / 2 : 0;

		/*
		 * flush the input queue
		 */

		gst_audioadapter_flush(element->input_queue, hann_length / 2);
	}

	/*
	 * done
	 */

	return GST_FLOW_OK;
}


/*
 * ============================================================================
 *
 *                              Signal Handlers
 *
 * ============================================================================
 */


static void rebuild_workspace_and_reset(GObject *object, GParamSpec *pspec, gpointer user_data)
{
	GSTLALWhiten *element = GSTLAL_WHITEN(object);

	/*
	 * free old work space
	 */

	free_workspace(element);

	/*
	 * build work space if we know the number of samples in the FFT
	 * (requires fft length to be set and the sample rate to be known)
	 */

	if(fft_length(element))
		/*
		 * we ignore this function's return code.  all failure
		 * paths within the function emit appropriate gstreamer
		 * errors.
		 */

		make_workspace(element);

	/*
	 * reset PSD regressor
	 */

	XLALPSDRegressorReset(element->psd_regressor);
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

	whiten->mean_psd_pad = NULL;
	gst_object_unref(pad);
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
		"width = (int) 64, " \
		"rate = (fraction) [0/1, MAX]"
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
	ARG_MEAN_PSD,
	ARG_SIGMA_SQUARED,
	ARG_SPECTRAL_CORRELATION,
	ARG_EXPAND_GAPS
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
	gboolean success = TRUE;

	str = gst_caps_get_structure(caps, 0);
	success &= gst_structure_get_int(str, "channels", &channels);

	if(success)
		*size = sizeof(double) * channels;
	else
		GST_WARNING_OBJECT(trans, "unable to parse caps %" GST_PTR_FORMAT, caps);

	return success;
}


/*
 * set_caps()
 */


static gboolean set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps)
{
	GSTLALWhiten *element = GSTLAL_WHITEN(trans);
	GstStructure *s;
	gint width;
	gint channels;
	gint rate;
	gboolean success = TRUE;

	/*
	 * extract the channel count and sample rate, and check that they
	 * are allowed
	 */

	s = gst_caps_get_structure(incaps, 0);
	success &= gst_structure_get_int(s, "width", &width);
	success &= gst_structure_get_int(s, "channels", &channels);
	success &= gst_structure_get_int(s, "rate", &rate);

	/*
	 * record the sample rate
	 */

	if(!success)
		GST_ERROR_OBJECT(element, "unable to parse caps %" GST_PTR_FORMAT, incaps);
	else if(rate != element->sample_rate) {
		element->sample_rate = rate;

		/*
		 * let everybody know the PSD's Nyquist has changed.  this
		 * triggers the building of a new Hann window, new FFT
		 * plans, and new workspaces
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
 * FIXME:  handle flusing and eos (i.e. flush the input queue and send the
 * last bit of data downstream)
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
	/* 1/2 the hann window */
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
		 * sample count plus the number of samples in the input
		 * queue rounded down to an integer multiple of 1/2 the
		 * Hann window size, but only if there's enough data for at
		 * least 1 full Hann window.
		 */

		/* number of samples available */
		*othersize = size / unit_size + get_available_samples(element);
		/* number of quanta available */
		*othersize /= quantum;
		/* number of output bytes to be generated */
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
 * start()
 */


static gboolean start(GstBaseTransform *trans)
{
	GSTLALWhiten *element = GSTLAL_WHITEN(trans);

	element->input_queue = g_object_new(GST_TYPE_AUDIOADAPTER, "unit-size", (guint) sizeof(*element->tdworkspace->data->data), NULL);

	/*
	 * an invalid t0 trips the "this buffer is a discont" behaviour in
	 * the transform() method, causing the timestamp book-keeping to
	 * reset and zeroing the output history.  the rest of the
	 * initialization being done here is mostly for safety.
	 */

	element->t0 = GST_CLOCK_TIME_NONE;
	element->offset0 = GST_BUFFER_OFFSET_NONE;
	element->next_offset_in = GST_BUFFER_OFFSET_NONE;
	element->next_offset_out = GST_BUFFER_OFFSET_NONE;
	element->need_discont = TRUE;

	element->output_history_offset = GST_BUFFER_OFFSET_NONE;
	element->nonzero_output_history_length = 0;

	return TRUE;
}


/*
 * stop()
 */


static gboolean stop(GstBaseTransform *trans)
{
	GSTLALWhiten *element = GSTLAL_WHITEN(trans);

	g_object_unref(element->input_queue);
	element->input_queue = NULL;

	return TRUE;
}


/*
 * transform()
 */


static GstFlowReturn transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf)
{
	GSTLALWhiten *element = GSTLAL_WHITEN(trans);
	GstFlowReturn result = GST_FLOW_OK;
	guint outsamples;
	gboolean output_is_gap;

	/*
	 * If the incoming buffer is a discontinuity, clear the input queue
	 * and reset the clock
	 */

	if(GST_BUFFER_IS_DISCONT(inbuf) || GST_BUFFER_OFFSET(inbuf) != element->next_offset_in || !GST_CLOCK_TIME_IS_VALID(element->t0)) {
		/*
		 * clear input queue
		 */

		gst_audioadapter_clear(element->input_queue);

		/*
		 * (re)sync timestamp and offset book-keeping
		 */

		g_assert(GST_BUFFER_TIMESTAMP_IS_VALID(inbuf));
		g_assert(GST_BUFFER_OFFSET_IS_VALID(inbuf));
		element->t0 = GST_BUFFER_TIMESTAMP(inbuf);
		element->offset0 = GST_BUFFER_OFFSET(inbuf);
		element->next_offset_out = GST_BUFFER_OFFSET(inbuf);

		/*
		 * next output is a discontinuity
		 */

		element->need_discont = TRUE;

		/*
		 * clear the output history.  this must be done after
		 * setting next_offset_out.
		 */

		zero_output_history(element);
	} else
		g_assert_cmpuint(GST_BUFFER_TIMESTAMP(inbuf), ==, gst_audioadapter_expected_timestamp(element->input_queue));
	element->next_offset_in = GST_BUFFER_OFFSET_END(inbuf);

	/*
	 * process data
	 */
	/* FIXME:  what this code isn't smart enough to do is recognize
	 * when there is no non-zero history in the adapter and just not
	 * process inbuf at all.  it's also not smart enough to know when
	 * inbuf is a large enough gap that it will drain whatever non-zero
	 * history there is so that only the first bit needs to be
	 * processed and the rest should be flagged as a gap. */

	gst_buffer_ref(inbuf);	/* don't let calling code free buffer */
	gst_audioadapter_push(element->input_queue, inbuf);
	result = whiten(element, outbuf, &outsamples, &output_is_gap);
	if(result != GST_FLOW_OK)
		goto done;

	/*
	 * set output metadata
	 */

	set_metadata(element, outbuf, outsamples, output_is_gap);

	/*
	 * done
	 */

done:
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

	case ARG_ZERO_PAD_SECONDS:
		element->zero_pad_seconds = g_value_get_double(value);

		/*
		 * it is now necessary to re-build the work spaces.  we
		 * affect this by hooking the workspace re-build function
		 * onto the zero-pad notification signal.  see
		 * gstlal_whiten_init()
		 */
		break;

	case ARG_FFT_LENGTH: {
		double fft_length_seconds = g_value_get_double(value);
		if(fft_length_seconds != element->fft_length_seconds) {
			/*
			 * record new value
			 */

			element->fft_length_seconds = fft_length_seconds;

			/*
			 * let everybody know the PSD's \Delta f has
			 * changed.  this affects a rebuild of the
			 * workspace.  see gstlal_whiten_init().
			 */

			g_object_notify(object, "delta-f");
		}
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
	case ARG_SIGMA_SQUARED:
	case ARG_SPECTRAL_CORRELATION:
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

	case ARG_EXPAND_GAPS:
		element->expand_gaps = g_value_get_boolean(value);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
		break;
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
		if(element->fft_length_seconds != 0)
			g_value_set_double(value, 1.0 / element->fft_length_seconds);
		else
			/* prevent divide-by-zero FPE;  if the fft length
			 * is 0, it doesn't matter what the delta f is */
			g_value_set_double(value, 0.0);
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

	case ARG_SIGMA_SQUARED:
		if(element->hann_window)
			g_value_set_double(value, element->hann_window->sumofsquares / element->hann_window->data->length);
		else
			g_value_set_double(value, 0.0);
		break;

	case ARG_SPECTRAL_CORRELATION:
		if(!element->hann_window)
			g_value_take_boxed(value, g_value_array_new(0));
		else {
			REAL8Sequence *correlation = XLALREAL8WindowTwoPointSpectralCorrelation(element->hann_window, element->fwdplan);
			g_value_take_boxed(value, gstlal_g_value_array_from_doubles(correlation->data, correlation->length));
			XLALDestroyREAL8Sequence(correlation);
		}
		break;

	case ARG_EXPAND_GAPS:
		g_value_set_boolean(value, element->expand_gaps);
		break;

	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
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

	if(element->mean_psd_pad) {
		gst_object_unref(element->mean_psd_pad);
		element->mean_psd_pad = NULL;
	}
	XLALPSDRegressorFree(element->psd_regressor);
	element->psd_regressor = NULL;
	XLALDestroyREAL8FrequencySeries(element->psd);
	element->psd = NULL;

	free_workspace(element);

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
	transform_class->start = GST_DEBUG_FUNCPTR(start);
	transform_class->stop = GST_DEBUG_FUNCPTR(stop);
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
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
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
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_FFT_LENGTH,
		g_param_spec_double(
			"fft-length",
			"FFT length",
			"Total length of the FFT convolution (including zero padding) in seconds",
			0, G_MAXDOUBLE, DEFAULT_FFT_LENGTH_SECONDS,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
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
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
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
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
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
	g_object_class_install_property(
		gobject_class,
		ARG_SIGMA_SQUARED,
		g_param_spec_double(
			"sigma-squared",
			"sigma^{2}",
			"FFT window mean square",
			0, G_MAXDOUBLE, 0,
			G_PARAM_READABLE | G_PARAM_STATIC_STRINGS
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_SPECTRAL_CORRELATION,
		g_param_spec_value_array(
			"spectral-correlation",
			"Two-point Spectral Correlation",
			"Two-point spectral correlation function for output stream.  Bin index is |k - k'|.",
			g_param_spec_double(
				"bin",
				"Bin",
				"Two-point spectral correlation bin",
				-G_MAXDOUBLE, G_MAXDOUBLE, 1.0,
				G_PARAM_READABLE | G_PARAM_STATIC_STRINGS
			),
			G_PARAM_READABLE | G_PARAM_STATIC_STRINGS
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_EXPAND_GAPS,
		g_param_spec_boolean(
			"expand-gaps",
			"expand gaps",
			"expand gaps to fill entire fft length",
			FALSE,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
}


/*
 * init()
 */


static void gstlal_whiten_init(GSTLALWhiten *element, GSTLALWhitenClass *klass)
{
	g_signal_connect(G_OBJECT(element), "notify::f-nyquist", G_CALLBACK(rebuild_workspace_and_reset), NULL);
	g_signal_connect(G_OBJECT(element), "notify::zero-pad", G_CALLBACK(rebuild_workspace_and_reset), NULL);
	g_signal_connect(G_OBJECT(element), "notify::delta-f", G_CALLBACK(rebuild_workspace_and_reset), NULL);

	element->mean_psd_pad = NULL;

	element->sample_units = lalDimensionlessUnit;
	element->sample_rate = 0;
	element->input_queue = NULL;

	element->hann_window = NULL;
	element->tukey_window = NULL;
	element->fwdplan = NULL;
	element->revplan = NULL;
	element->tdworkspace = NULL;
	element->fdworkspace = NULL;
	element->output_history = NULL;

	element->psd_regressor = XLALPSDRegressorNew(DEFAULT_AVERAGE_SAMPLES, DEFAULT_MEDIAN_SAMPLES);
	element->psd = NULL;

	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(element), TRUE);
}

/*
 * Copyright (C) 2017  Aaron Viets <aaron.viets@ligo.org>
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation; either version 2 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */


/*
 * =============================================================================
 *
 *				 Preamble
 *
 * =============================================================================
 */


/*
 * stuff from C
 */


#include <string.h>
#include <math.h>
#include <complex.h>


/*
 * stuff from gobject/gstreamer
 */


#include <glib.h>
#include <gst/gst.h>
#include <gst/audio/audio.h>
#include <gst/base/gstbasetransform.h>


/*
 * our own stuff
 */


#include <gstlal/gstlal.h>
#include <gstlal-calibration/gstlal_firtools.h>
#include <gstlal_resample.h>


#define SHORT_SINC_LENGTH 33
#define LONG_SINC_LENGTH 193


/*
 * ============================================================================
 *
 *				Custom Types
 *
 * ============================================================================
 */


/*
 * window type enum
 */


GType gstlal_resample_window_get_type(void) {

	static GType type = 0;

	if(!type) {
		static GEnumValue values[] = {
			{GSTLAL_RESAMPLE_DPSS, "GSTLAL_RESAMPLE_DPSS", "Maximize energy concentration in main lobe"},
			{GSTLAL_RESAMPLE_KAISER, "GSTLAL_RESAMPLE_KAISER", "Simple approximtion to DPSS window"},
			{GSTLAL_RESAMPLE_DOLPH_CHEBYSHEV, "GSTLAL_RESAMPLE_DOLPH_CHEBYSHEV", "Attenuate all side lobes equally"},
			{0, NULL, NULL}
		};

		type = g_enum_register_static("GSTLAL_RESAMPLE_WINDOW", values);
	}

	return type;
}


/*
 * ============================================================================
 *
 *				 Utilities
 *
 * ============================================================================
 */


/*
 * First, the constant upsample functions, which just copy inputs to n outputs 
 */
#define DEFINE_CONST_UPSAMPLE(size) \
static void const_upsample_ ## size(const gint ## size *src, gint ## size *dst, guint64 src_size, gint32 cadence) { \
 \
	const gint ## size *src_end; \
	gint32 i; \
 \
	for(src_end = src + src_size; src < src_end; src++) { \
		for(i = 0; i < cadence; i++, dst++) \
			*dst = *src; \
	} \
}

DEFINE_CONST_UPSAMPLE(8)
DEFINE_CONST_UPSAMPLE(16)
DEFINE_CONST_UPSAMPLE(32)
DEFINE_CONST_UPSAMPLE(64)


static void const_upsample_other(const gint8 *src, gint8 *dst, guint64 src_size, gint unit_size, gint32 cadence) {

	const gint8 *src_end;
	gint32 i;

	for(src_end = src + src_size * unit_size; src < src_end; src += unit_size) {
		for(i = 0; i < cadence; i++, dst += unit_size)
			memcpy(dst, src, unit_size);
	}
}


/*
 * Linear upsampling functions, in which upsampled output samples 
 * lie on lines connecting input samples 
 */
#define DEFINE_LINEAR_UPSAMPLE(DTYPE, COMPLEX) \
static void linear_upsample_ ## DTYPE ## COMPLEX(const DTYPE COMPLEX *src, DTYPE COMPLEX *dst, guint64 src_size, gint32 cadence, DTYPE COMPLEX *end_samples, gint32 *num_end_samples) { \
 \
	/* First, fill in previous data using the last sample of the previous input buffer */ \
	DTYPE COMPLEX slope; /* first derivative between points we are connecting */ \
	gint32 i; \
	if(*num_end_samples > 0) { \
		slope = *src - *end_samples; \
		*dst = *end_samples; \
		dst++; \
		for(i = 1; i < cadence; i++, dst++) \
			*dst = *end_samples + slope * i / cadence; \
	} \
 \
	/* Now, process the current input buffer */ \
	const DTYPE COMPLEX *src_end; \
	for(src_end = src + src_size - 1; src < src_end; src++) { \
		slope = *(src + 1) - *src; \
		*dst = *src; \
		dst++; \
		for(i = 1; i < cadence; i++, dst++) \
			*dst = *src + slope * i / cadence; \
	} \
 \
	/* Save the last input sample for the next buffer, so that we can find the slope */ \
	*end_samples = *src; \
	*num_end_samples = 1; \
}

DEFINE_LINEAR_UPSAMPLE(float, )
DEFINE_LINEAR_UPSAMPLE(double, )
DEFINE_LINEAR_UPSAMPLE(float, complex)
DEFINE_LINEAR_UPSAMPLE(double, complex)


/*
 * Qaudratic spline interpolating functions, just for fun. The curve
 * connecting two points depends on those two points and the previous point. 
 */
#define DEFINE_QUADRATIC_UPSAMPLE(DTYPE, COMPLEX) \
static void quadratic_upsample_ ## DTYPE ## COMPLEX(const DTYPE COMPLEX *src, DTYPE COMPLEX *dst, guint64 src_size, gint32 cadence, DTYPE COMPLEX *end_samples, gint32 *num_end_samples) { \
 \
	/* First, fill in previous data using the last samples of the previous input buffer */ \
	DTYPE COMPLEX dxdt0 = 0.0, half_d2xdt2 = 0.0; /* first derivative and half of second derivative at initial point */ \
	gint32 i; \
	if(*num_end_samples > 1) { \
		g_assert(end_samples); \
		dxdt0 = (*src - end_samples[1]) / 2.0; \
		half_d2xdt2 = *src - end_samples[0] - dxdt0; \
		*dst = end_samples[0]; \
		dst++; \
		for(i = 1; i < cadence; i++, dst++) \
			*dst = end_samples[0] + dxdt0 * i / cadence + (i * i * half_d2xdt2) / (cadence * cadence); \
	} \
 \
	/* This needs to happen even if the first section was skipped */ \
	if(*num_end_samples >= 1) { \
		if(*num_end_samples < 2) { \
			/* In this case, we also must fill in data from end_samples to the start of src, assuming an initial slope of zero */ \
			half_d2xdt2 = *src - end_samples[0] - dxdt0; \
			*dst = end_samples[0]; \
			dst++; \
			for(i = 1; i < cadence; i++, dst++) \
				*dst = end_samples[0] + (i * i * half_d2xdt2) / (cadence * cadence); \
		} \
		if(src_size > 1) { \
			dxdt0 = (*(src + 1) - end_samples[0]) / 2.0; \
			half_d2xdt2 = *(src + 1) - *src - dxdt0; \
			*dst = *src; \
			dst++; \
			for(i = 1; i < cadence; i++, dst++) \
				*dst = *src + dxdt0 * i / cadence + (i * i * half_d2xdt2) / (cadence * cadence); \
			src++; \
		} \
 \
	} else { \
		/* This function should not be called if there is not enough data to make an output buffer */ \
		g_assert(src_size > 1); \
		/* If this is the first input data or follows a discontinuity, assume the initial slope is zero */ \
		half_d2xdt2 = *(src + 1) - *src; \
		*dst = *src; \
		dst++; \
		for(i = 1; i < cadence; i++, dst++) \
			*dst = *src + (i * i * half_d2xdt2) / (cadence * cadence); \
		src++; \
	} \
 \
	/* Now, process the current input buffer */ \
	const DTYPE COMPLEX *src_end; \
	for(src_end = src + src_size - 2; src < src_end; src++) { \
		dxdt0 = (*(src + 1) - *(src - 1)) / 2.0; \
		half_d2xdt2 = *(src + 1) - *src - dxdt0; \
		*dst = *src; \
		dst++; \
		for(i = 1; i < cadence; i++, dst++) \
			*dst = *src + dxdt0 * i / cadence + (i * i * half_d2xdt2) / (cadence * cadence); \
	} \
 \
	/* Save the last two samples for the next buffer */ \
	if(src_size == 1 && *num_end_samples >= 1) { \
		*num_end_samples = 2; \
		end_samples[1] = end_samples[0]; \
	} else if(src_size > 1) { \
		*num_end_samples = 2; \
		end_samples[1] = *(src - 1); \
	} else \
		*num_end_samples = 1; \
	end_samples[0] = *src; \
}

DEFINE_QUADRATIC_UPSAMPLE(float, )
DEFINE_QUADRATIC_UPSAMPLE(double, )
DEFINE_QUADRATIC_UPSAMPLE(float, complex)
DEFINE_QUADRATIC_UPSAMPLE(double, complex)


/*
 * Cubic spline interpolating functions. The curve connecting two points 
 * depends on those two points and the previous and following point. 
 */
#define DEFINE_CUBIC_UPSAMPLE(DTYPE, COMPLEX) \
static void cubic_upsample_ ## DTYPE ## COMPLEX(const DTYPE COMPLEX *src, DTYPE COMPLEX *dst, guint64 src_size, gint32 cadence, DTYPE COMPLEX *dxdt0, DTYPE COMPLEX *end_samples, gint32 *num_end_samples) { \
 \
	/* First, fill in previous data using the last samples of the previous input buffer */ \
	DTYPE COMPLEX dxdt1, half_d2xdt2_0, sixth_d3xdt3; /* first derivative at end point, half of second derivative and one sixth of third derivative at initial point */ \
	gint32 i; \
	if(*num_end_samples > 1) { \
		g_assert(end_samples); \
		dxdt1 = (*src - end_samples[1]) / 2.0; \
		half_d2xdt2_0 =  3 * (end_samples[0] - end_samples[1]) - dxdt1 - 2 * *dxdt0; \
		sixth_d3xdt3 = 2 * (end_samples[1] - end_samples[0]) + dxdt1 + *dxdt0; \
		*dst = end_samples[1]; \
		dst++; \
		for(i = 1; i < cadence; i++, dst++) \
			*dst = end_samples[1] + *dxdt0 * i / cadence + (i * i * half_d2xdt2_0) / (cadence * cadence) + (i * i * i * sixth_d3xdt3) / (cadence * cadence* cadence); \
		/* Save the slope at the end point as the slope at the next initial point */ \
		*dxdt0 = dxdt1; \
	} \
 \
	if(*num_end_samples > 0 && src_size > 1) { \
		dxdt1 = (*(src + 1) - end_samples[0]) / 2.0; \
		half_d2xdt2_0 =  3 * (*src - end_samples[0]) - dxdt1 - 2 * *dxdt0; \
		sixth_d3xdt3 = 2 * (end_samples[0] - *src) + dxdt1 + *dxdt0; \
		*dst = end_samples[0]; \
		dst++; \
		for(i = 1; i < cadence; i++, dst++) \
			*dst = end_samples[0] + *dxdt0 * i / cadence + (i * i * half_d2xdt2_0) / (cadence * cadence) + (i * i * i * sixth_d3xdt3) / (cadence * cadence* cadence); \
		/* Save the slope at the end point as the slope at the next initial point */ \
		*dxdt0 = dxdt1; \
	} \
 \
	/* Now, process the current input buffer */ \
	const DTYPE COMPLEX *src_end; \
	for(src_end = src + src_size - 2; src < src_end; src++) { \
		dxdt1 = (*(src + 2) - *src) / 2.0; \
		half_d2xdt2_0 =  3 * (*(src + 1) - *src) - dxdt1 - 2 * *dxdt0; \
		sixth_d3xdt3 = 2 * (*src - *(src + 1)) + dxdt1 + *dxdt0; \
		*dst = *src; \
		dst++; \
		for(i = 1; i < cadence; i++, dst++) \
			*dst = *src + *dxdt0 * i / cadence + (i * i * half_d2xdt2_0) / (cadence * cadence) + (i * i * i * sixth_d3xdt3) / (cadence * cadence* cadence); \
		/* Save the slope at the end point as the slope at the next initial point */ \
		*dxdt0 = dxdt1; \
	} \
 \
	/* Save the last two samples for the next buffer */ \
	if(src_size == 1 && *num_end_samples > 0) { \
		end_samples[1] = end_samples[0]; \
		end_samples[0] = *src; \
		*num_end_samples = 2; \
	} else if(src_size == 1) { \
		end_samples[0] = *src; \
		*num_end_samples = 1; \
	} else { \
		end_samples[1] = *src; \
		end_samples[0] = *(src + 1); \
		*num_end_samples = 2; \
	} \
}

DEFINE_CUBIC_UPSAMPLE(float, )
DEFINE_CUBIC_UPSAMPLE(double, )
DEFINE_CUBIC_UPSAMPLE(float, complex)
DEFINE_CUBIC_UPSAMPLE(double, complex)


/*
 * Upsampling functions that reduce aliasing by filtering the inputs
 * with a sinc table [ sin(pi * x - phase) / (pi * x) ]
 */
#define DEFINE_SINC_UPSAMPLE(DTYPE, COMPLEX) \
static void sinc_upsample_ ## DTYPE ## COMPLEX(const DTYPE COMPLEX *src, DTYPE COMPLEX *dst, gint64 src_size, gint64 dst_size, gint32 cadence, DTYPE COMPLEX *end_samples, gint32 *num_end_samples, gint32 *index_end_samples, gint32 max_end_samples, gint32 sinc_length, double *sinc_table, gboolean *produced_outbuf) { \
 \
	/*
	 * If this is the start of stream or right after a discont, set the location
	 * of the first end sample to the start of end_samples
	 * and set the location of the latest end sample to the end of end_samples.
	 */ \
	if(!(*produced_outbuf)) { \
		*index_end_samples = 0; \
		index_end_samples[1] = *num_end_samples - 1; \
	} \
 \
	/* move the pointer to the element of end_samples corresponding to the next output sample */ \
	end_samples += *index_end_samples; \
	gint32 i, j, i_start, j_start, i_stop, j_stop, sinc_index, end_samples_index, dst_index, dst_shift; \
	dst_index = 0; \
 \
	/* start dst with zeros to simplify the following algorithms */ \
	memset(dst, 0, dst_size * sizeof(DTYPE COMPLEX)); \
 \
	if(dst_size && !(*produced_outbuf)) { \
		/* We have enough input to produce output and this is the first output buffer (or first after a discontinuity)... */ \
 \
		/* dependence of output on the end_samples that we will replace, if any */ \
		i_stop = src_size >= max_end_samples ? *num_end_samples : *num_end_samples - max_end_samples + src_size; \
		for(i = 0; i < i_stop; i++, end_samples++, dst_index += cadence) { \
			dst[dst_index] += *sinc_table * *end_samples; \
			j_stop = i * cadence < sinc_length / 2 ? i * cadence : sinc_length / 2; \
			for(j = 1; j <= j_stop; j++) { \
				dst[dst_index + j] += sinc_table[j] * *end_samples; \
				dst[dst_index - j] += sinc_table[j] * *end_samples; \
			} \
			j_start = j_stop + 1; \
			j_stop = sinc_length / 2; \
			for(j = j_start; j <= j_stop; j++) \
				dst[dst_index + j] += sinc_table[j] * *end_samples; \
		} \
 \
		/* dependence of output on end_samples that we need to keep for the next input buffer, if any */ \
		i_stop = *num_end_samples - i_stop; \
		for(i = 0; i < i_stop; i++, end_samples++, dst_index += cadence) { \
			if(i_stop - i + src_size > max_end_samples / 2) \
				dst[dst_index] += *sinc_table * *end_samples; \
			j_stop = sinc_length / 2 - i * cadence < (i_stop - i + src_size) * cadence - sinc_length / 2 ? sinc_length / 2 - i * cadence : (i_stop - i + src_size) * cadence - sinc_length / 2; \
			for(j = 1; j < j_stop; j++) \
				dst[dst_index + j] += sinc_table[j] * *end_samples; \
			j_start = 1 > 1 - j_stop ? 1 : 1 - j_stop; \
			j_stop = sinc_length / 2 < (*num_end_samples - i_stop + i) * cadence ? sinc_length / 2 : (*num_end_samples - i_stop + i) * cadence; \
			for(j = j_start; j <= j_stop; j++) \
				dst[dst_index - j] += sinc_table[j] * *end_samples; \
		} \
 \
		/* dependence of output on src samples that we don't need to store for the next input buffer, if any */ \
		i_stop = src_size - max_end_samples; \
		for(i = 0; i < i_stop; i++, src++, dst_index += cadence) { \
			dst[dst_index] += *sinc_table * *src; \
			j_stop = *num_end_samples + i * cadence < sinc_length / 2 ? *num_end_samples + i * cadence : sinc_length / 2; \
			for(j = 1; j <= j_stop; j++) { \
				dst[dst_index + j] += sinc_table[j] * *src; \
				dst[dst_index - j] += sinc_table[j] * *src; \
			} \
			j_start = j_stop + 1; \
			j_stop = sinc_length / 2; \
			for(j = j_start; j <= j_stop; j++) \
				dst[dst_index + j] += sinc_table[j] * *src; \
		} \
 \
		/* dependence of output on src samples that we need to keep for the next input buffer (guaranteed to exist) */ \
		i_stop = max_end_samples < src_size ? max_end_samples : src_size; \
		for(i = 0; i < i_stop; i++, src++, dst_index += cadence) { \
			if(i_stop - i > max_end_samples / 2) \
				dst[dst_index] += *sinc_table * *src; \
 \
			/* dst samples before src */ \
			j_start = 0 > (max_end_samples / 2 - i_stop + i) * cadence + (max_end_samples % 2 ? cadence / 2 : 0) ? 1 : (max_end_samples / 2 - i_stop + i) * cadence + (max_end_samples % 2 ? cadence / 2 : 0) + 1; \
			j_stop = (*num_end_samples + (src_size > max_end_samples ? src_size - max_end_samples : 0) + i) * cadence; \
			j_stop = j_stop < sinc_length / 2 ? j_stop : sinc_length / 2; \
			for(j = j_start; j <= j_stop; j++) \
				dst[dst_index - j] += sinc_table[j] * *src; \
 \
			/* dst samples after src */ \
			j_stop = sinc_length / 2 - (max_end_samples < src_size ? i : max_end_samples - src_size + i) * cadence; \
			for(j = 1; j < j_stop; j++) \
				dst[dst_index + j] += sinc_table[j] * *src; \
		} \
	} else if(dst_size) { \
		/* We have enough input to produce output and this is not the first output buffer after a discontinuity */ \
 \
		/* First, deal with dependence of output on end_samples that come before first output in dst */ \
		i_stop = (1 + max_end_samples) / 2; \
		for(i = 0; i < i_stop; i++) { \
			sinc_index = sinc_length / 2 - i * cadence; \
			j_stop = dst_size <= i * cadence ? dst_size : 1 + i * cadence; \
			end_samples_index = i + *index_end_samples < max_end_samples ? i : i - max_end_samples; \
			for(j = 0; j < j_stop; j++) \
				dst[j] += sinc_table[sinc_index + j] * end_samples[end_samples_index]; \
		} \
 \
		/* dependence of output on the rest of end_samples (we know that there are max_end_samples end samples) */ \
		/* first, shift the pointer to dst to a point aligned in time with an input sample */ \
		dst_shift = (max_end_samples % 2) * cadence / 2; \
		dst += dst_shift; \
		i_start = i_stop; \
		i_stop = max_end_samples; \
		for(i = i_start; i < i_stop; i++, dst_index += cadence) { \
			end_samples_index = i + *index_end_samples < max_end_samples ? i : i - max_end_samples; \
			if(max_end_samples + src_size - i > max_end_samples / 2) \
				dst[dst_index] += *sinc_table * end_samples[end_samples_index]; \
			j_stop = dst_size - dst_shift - (i - i_start) * cadence <= sinc_length / 2 ? dst_size - dst_shift - (i - i_start) * cadence - 1 : sinc_length / 2; \
			j_stop = j_stop < dst_shift + (i - i_start) * cadence ? j_stop : dst_shift + (i - i_start) * cadence; \
			for(j = 1; j <= j_stop; j++) { \
				dst[dst_index + j] += sinc_table[j] * end_samples[end_samples_index]; \
				dst[dst_index - j] += sinc_table[j] * end_samples[end_samples_index]; \
			} \
			/* handle remaining "one-sided" dependence */ \
			j_start = j_stop >= 0 ? j_stop + 1 : -j_stop; \
			if(j_stop == dst_shift + (i - i_start) * cadence) { \
				j_stop = dst_size - dst_shift - (i - i_start) * cadence <= sinc_length / 2 ? dst_size - dst_shift - (i - i_start) * cadence - 1 : sinc_length / 2; \
				for(j = j_start; j <= j_stop; j++) \
					dst[dst_index + j] += sinc_table[j] * end_samples[end_samples_index]; \
			} else { \
				j_stop = sinc_length / 2 < dst_shift + (i - i_start) * cadence ? sinc_length / 2 : dst_shift + (i - i_start) * cadence; \
				for(j = j_start; j <= j_stop; j++) \
					dst[dst_index - j] += sinc_table[j] * end_samples[end_samples_index]; \
			} \
		} \
 \
		/* dependence of output on src samples */ \
		for(i = 0; i < src_size; i++, src++, dst_index += cadence) { \
			if(src_size - i > max_end_samples / 2) \
				dst[dst_index] += *sinc_table * *src; \
			j_stop = dst_size - dst_shift - (max_end_samples / 2 + i) * cadence <= sinc_length / 2 ? dst_size - dst_shift - (max_end_samples / 2 + i) * cadence - 1 : sinc_length / 2; \
			for(j = 1; j <= j_stop; j++) { \
				dst[dst_index + j] += sinc_table[j] * *src; \
				dst[dst_index - j] += sinc_table[j] * *src; \
			} \
			j_start = j_stop >= -j_stop ? j_stop + 1 : -j_stop; \
			j_stop = sinc_length / 2; \
			for(j = j_start; j <= j_stop; j++) \
				dst[dst_index - j] += sinc_table[j] * *src; \
		} \
 \
	} \
 \
	/* move end_samples pointer back to beginning */ \
	if(dst_size && *produced_outbuf) \
		end_samples -= *index_end_samples; \
	else if(dst_size) { \
		end_samples -= index_end_samples[1] + 1; \
		*produced_outbuf = TRUE; \
	} \
	/* find new locations in end_samples where oldest and newest sample will be stored */ \
	index_end_samples[1] = (index_end_samples[1] + src_size) % max_end_samples; \
	if(dst_size) \
		*index_end_samples = (index_end_samples[1] + 1) % max_end_samples; \
 \
	/* Move pointer to end of src, so we can store the last samples */ \
	if(dst_size) \
		src--; \
	else \
		src += src_size - 1; \
 \
	/* Store current input samples we will need later in end_samples */ \
	i_stop = index_end_samples[1] + 1 < src_size ? 0 : index_end_samples[1] + 1 - src_size; \
	for(i = index_end_samples[1]; i >= i_stop; i--, src--) \
		end_samples[i] = *src; \
 \
	/* A second loop is necessary in case we hit the boundary of end_samples before storing all the end samples */ \
	i_stop = max_end_samples - ((max_end_samples < src_size ? max_end_samples : src_size) - index_end_samples[1] - 1); \
	for(i = max_end_samples - 1; i >= i_stop; i--, src--) \
		end_samples[i] = *src; \
 \
	/* record how many samples are stored in end_samples */ \
	*num_end_samples += src_size; \
	if(*num_end_samples > max_end_samples) \
		*num_end_samples = max_end_samples; \
}


DEFINE_SINC_UPSAMPLE(float, )
DEFINE_SINC_UPSAMPLE(double, )
DEFINE_SINC_UPSAMPLE(float, complex)
DEFINE_SINC_UPSAMPLE(double, complex)


/*
 * Simple downsampling functions that just pick every nth value 
 */
#define DEFINE_DOWNSAMPLE(size) \
static void downsample_ ## size(const gint ## size *src, gint ## size *dst, guint64 dst_size, gint32 inv_cadence, gint32 leading_samples) { \
 \
	/* increnent the pointer to the input buffer data to point to the first outgoing sample */ \
	src += leading_samples; \
	const gint ## size *dst_end; \
	for(dst_end = dst + dst_size; dst < dst_end; dst++, src += inv_cadence) \
		*dst = *src; \
 \
}

DEFINE_DOWNSAMPLE(8)
DEFINE_DOWNSAMPLE(16)
DEFINE_DOWNSAMPLE(32)
DEFINE_DOWNSAMPLE(64)


static void downsample_other(const gint8 *src, gint8 *dst, guint64 dst_size, gint unit_size, gint32 inv_cadence, gint32 leading_samples) {

	/* increment the pointer to the input buffer data to point to the first outgoing sample */	
	src += unit_size * leading_samples; \
	const gint8 *dst_end;

	for(dst_end = dst + dst_size * unit_size; dst < dst_end; dst += unit_size, src += unit_size * inv_cadence)
		memcpy(dst, src, unit_size);
}


/*
 * Downsampling functions that average n samples, where the 
 * middle sample has the timestamp of the outgoing sample 
 */
#define DEFINE_AVG_DOWNSAMPLE(DTYPE, COMPLEX) \
static void avg_downsample_ ## DTYPE ## COMPLEX(const DTYPE COMPLEX *src, DTYPE COMPLEX *dst, guint64 src_size, guint64 dst_size, gint32 inv_cadence, gint32 leading_samples, DTYPE COMPLEX *end_samples, gint32 *num_end_samples) { \
 \
	/*
	 * If inverse cadence (rate in / rate out) is even, we take inv_cadence/2 samples
	 * from before and after the middle sample (which is timestamped with the outgoing timestamp).
	 * We then sum 1/2 first sample + 1/2 last sample + all other samples,
	 * and divide by inv_cadence. Technically, this is a Tukey window,
	 * but for large inv_cadence, it is almost an average.
	 */ \
	if(!(inv_cadence % 2) && dst_size != 0) { \
		/* Produce the first output sample, which may have a contribution from the leftover samples */ \
		if(*num_end_samples > 0) \
			*dst = *end_samples; \
		else \
			*dst = 0.0; \
		const DTYPE COMPLEX *src_end; \
		int num_src_samples = leading_samples + (*num_end_samples + leading_samples < inv_cadence ?  inv_cadence / 2 : -(inv_cadence / 2)); \
		for(src_end = src + num_src_samples; src < src_end; src++) \
			*dst += *src; \
		*dst += *src / 2; \
		*dst /= inv_cadence; \
		dst++; \
 \
		/* Process current buffer */ \
		const DTYPE COMPLEX *dst_end; \
		gint32 i; \
		for(dst_end = dst + dst_size - 1; dst < dst_end; dst++) { \
			*dst = *src / 2; \
			src++; \
			for(i = 0; i < inv_cadence - 1; i++, src++) \
				*dst += *src; \
			*dst += *src / 2; \
			*dst /= inv_cadence; \
		} \
 \
		/* Save the sum of the unused samples in end_samples and the number of unused samples in num_end_samples */ \
		*num_end_samples = 1 + (src_size + inv_cadence / 2 - leading_samples - 1) % inv_cadence; \
		*end_samples = *src / 2; \
		src++; \
		for(i = 1; i < *num_end_samples; i++, src++) \
			*end_samples += *src; \
 \
	/*
	 * If inverse cadence (rate in / rate out) is odd, we take the average of samples starting
	 * at inv_cadence/2 - 1 samples before the middle sample (which is timestamped with the
	 * outgoing timestamp) and ending at inv_cadence/2 - 1 samples after the middle sample.
	 */ \
	} else if(inv_cadence % 2 && dst_size != 0) { \
		/* Produce the first output sample, which may have a contribution from the leftover samples */ \
		if(*num_end_samples > 0) \
			*dst = *end_samples; \
		else \
			*dst = 0.0; \
		const DTYPE COMPLEX *src_end; \
		int num_src_samples = leading_samples + (*num_end_samples + leading_samples < inv_cadence ? 1 + inv_cadence / 2 : -(inv_cadence / 2)); \
		for(src_end = src + num_src_samples; src < src_end; src++) \
			*dst += *src; \
		*dst /= inv_cadence; \
		dst++; \
 \
		/* Process current buffer */ \
		const DTYPE COMPLEX *dst_end; \
		gint32 i; \
		for(dst_end = dst + dst_size - 1; dst < dst_end; dst++) { \
			for(i = 0; i < inv_cadence; i++, src++) \
				*dst += *src; \
			*dst /= inv_cadence; \
		} \
 \
		/* Save the sum of the unused samples in end_samples and the number of unused samples in num_end_samples */ \
		*num_end_samples = (src_size + inv_cadence / 2 - leading_samples) % inv_cadence; \
		*end_samples = *src; \
		src++; \
		for(i = 1; i < *num_end_samples; i++, src++) \
			*end_samples += *src; \
 \
	/*
	 * If the size of the outgoing buffer has been computed to be zero, all we want to
	 * do is store the additional data from the input buffer in end_samples and num_end_samples.
	 */ \
	} else { \
		guint64 i; \
		if(*num_end_samples == 0 && !(inv_cadence % 2)) { \
			/* Then the first input sample should be divided by two, since it is the first to affect the next output sample. */ \
			*end_samples = *src / 2; \
			src++; \
			for(i = 1; i < src_size; i++, src++) \
				*end_samples += *src; \
		} else { \
			/* Then each sample contributes its full value to end_samples. */ \
			for(i = 0; i < src_size; i++, src++) \
				*end_samples += *src; \
		} \
		*num_end_samples += src_size; \
	} \
}

DEFINE_AVG_DOWNSAMPLE(float, )
DEFINE_AVG_DOWNSAMPLE(double, )
DEFINE_AVG_DOWNSAMPLE(float, complex)
DEFINE_AVG_DOWNSAMPLE(double, complex)


/*
 * Downsampling functions that reduce aliasing by filtering the inputs
 * with a sinc table [ sin(pi * x * cadence) / (pi * x * cadence) ]
 */
#define DEFINE_SINC_DOWNSAMPLE(DTYPE, COMPLEX) \
static void sinc_downsample_ ## DTYPE ## COMPLEX(const DTYPE COMPLEX *src, DTYPE COMPLEX *dst, guint64 src_size, guint64 dst_size, gint32 inv_cadence, gint32 leading_samples, DTYPE COMPLEX *end_samples, gint32 *num_end_samples, gint32 *index_end_samples, gint32 max_end_samples, double *sinc_table) { \
 \
	/*
	 * If this is the start of stream or right after a discont, record the location
	 * corresponding to the first output sample produced relative to the start of end_samples
	 * and set the location of the latest end sample to the end of end_samples.
	 */ \
	if(*index_end_samples == -1) { \
		*index_end_samples = leading_samples; \
		index_end_samples[1] = max_end_samples - 1; \
	} \
 \
	DTYPE COMPLEX *end_samples_end; \
	end_samples_end = end_samples + index_end_samples[1]; \
 \
	/* move the pointer to the element of end_samples corresponding to the next output sample */ \
	end_samples += *index_end_samples; \
	guint64 i; \
	gint32 j, k; \
	/* If we have enough input to produce output and this is the first output buffer (or first after a discontinuity)... */ \
	if(dst_size && *num_end_samples < max_end_samples) { \
		DTYPE COMPLEX *ptr_before, *ptr_after, *ptr_end; \
		for(i = 0; i < dst_size; i++, dst++) { \
			/*
			 * In this case, the inputs in end_samples are in chronological order, so it is simple
			 * to determine whether the sinc table is centered in end_samples or src
			 */ \
			*dst = (DTYPE) *sinc_table * (*index_end_samples < *num_end_samples ? *end_samples : *(src + *index_end_samples - *num_end_samples)); \
			sinc_table++; \
 \
			/*
			 * First, deal with dependence of output sample on elements of end_samples before and
			 * after center of sinc table. j is the number of steps to reach a boundary of *end_samples
			 */ \
			j = *index_end_samples < *num_end_samples - *index_end_samples - 1 ? *index_end_samples : *num_end_samples - *index_end_samples - 1; \
			ptr_end = end_samples + j; \
			for(ptr_before = end_samples - 1, ptr_after = end_samples + 1; ptr_after <= ptr_end; ptr_after++, ptr_before--, sinc_table++) \
				*dst += (DTYPE) *sinc_table * (*ptr_before + *ptr_after); \
 \
			/* Next, deal with potential "one-sided" dependence of output sample on elements of end_samples after center of sinc table. */ \
			j = *num_end_samples - *index_end_samples - 1 < max_end_samples / 2 ? *num_end_samples - *index_end_samples - 1 : max_end_samples / 2; \
			ptr_end = end_samples + j; \
			for(ptr_after = end_samples + *index_end_samples + 1; ptr_after <= ptr_end; ptr_after++, sinc_table++) \
				*dst += (DTYPE) *sinc_table * *ptr_after; \
 \
			/* Next, deal with dependence of output sample on current input samples before and after center of sinc table */ \
			j = *index_end_samples - *num_end_samples < max_end_samples / 2 ? *index_end_samples - *num_end_samples : max_end_samples / 2; \
			ptr_end = (DTYPE COMPLEX *) src + *index_end_samples - *num_end_samples + j; \
			for(ptr_before = (DTYPE COMPLEX *) src + *index_end_samples - *num_end_samples - 1, ptr_after = ptr_end - j + 1; ptr_after <= ptr_end; ptr_after++, ptr_before--, sinc_table++) \
				*dst += (DTYPE) *sinc_table * (*ptr_before + *ptr_after); \
 \
			/* Next, deal with dependence of output sample on current input samples after and end_samples before center of sinc table, which is in end_samples */ \
			if(*index_end_samples < *num_end_samples) { \
				j = *index_end_samples < max_end_samples / 2 ? 2 * *index_end_samples - *num_end_samples + 1 : max_end_samples / 2 - *num_end_samples + *index_end_samples + 1; \
				ptr_end = (DTYPE COMPLEX *) src + j - 1; \
				for(ptr_before = end_samples - (*num_end_samples - *index_end_samples), ptr_after = (DTYPE COMPLEX *) src; ptr_after <= ptr_end; ptr_after++, ptr_before--, sinc_table++) \
					*dst += (DTYPE) *sinc_table * (*ptr_before + *ptr_after); \
			} else { \
				/* Next, deal with dependence of output sample on current input samples after and end_samples before center of sinc table, which is in src */ \
				j = *num_end_samples < max_end_samples / 2 - (*index_end_samples - *num_end_samples) ? *num_end_samples : max_end_samples / 2 - (*index_end_samples - *num_end_samples); \
				ptr_end = (DTYPE COMPLEX *) src + 2 * (*index_end_samples - *num_end_samples) + j; \
				for(ptr_before = end_samples_end, ptr_after = ptr_end - j + 1; ptr_after <= ptr_end; ptr_after++, ptr_before--, sinc_table++) \
					*dst += (DTYPE) *sinc_table * (*ptr_before + *ptr_after); \
			} \
 \
			/* Next, deal with potential "one-sided" dependence of output sample on current input samples after center of sinc table. */ \
			j = max_end_samples / 2 - (2 * *index_end_samples > *num_end_samples - 1 ? *index_end_samples : (*num_end_samples - *index_end_samples - 1)); \
			ptr_end = (DTYPE COMPLEX *) src + *index_end_samples - *num_end_samples + max_end_samples / 2; \
			for(ptr_after = ptr_end - j + 1; ptr_after <= ptr_end; ptr_after++, sinc_table++) \
				*dst += (DTYPE) *sinc_table * *ptr_after; \
 \
			/* We've now reached the end of the sinc table. Move the pointer back. */ \
			sinc_table -= (1 + max_end_samples / 2); \
 \
			/* Also need to increment end_samples. *index_end_samples is used to find our place in *src, so it is reduced later */ \
			end_samples += (inv_cadence - ((*index_end_samples % max_end_samples) + inv_cadence < max_end_samples ? 0 : max_end_samples)); \
			*index_end_samples += inv_cadence; \
		} \
 \
	} else if(dst_size) { \
		/* We have enough input to produce output and this is not the first output buffer since a discont */ \
		g_assert_cmpint(*num_end_samples, ==, max_end_samples); \
		/* artificially increase index_end_samples[1] so that comparison to *index_end_samples tells us whether the sinc table is centered in end_samples or src. */ \
		index_end_samples[1] += *index_end_samples > index_end_samples[1] ? max_end_samples : 0; \
		DTYPE COMPLEX *ptr_before, *ptr_after, *ptr_end; \
		gint32 j1, j2, j3; \
		for(i = 0; i < dst_size; i++, dst++) { \
			if(*index_end_samples <= index_end_samples[1]) { \
				/*
			 	 * sinc table is centered in end_samples. First, deal with dependence of output sample on
			 	 * elements of end_samples before and after center of sinc table. There are 2 possible end
			 	 * points for a for loop: we hit the boundary of end_samples in either chronology or memory.
				 */ \
				*dst = (DTYPE) *sinc_table * *end_samples; \
				sinc_table++; \
				j1 = index_end_samples[1] - *index_end_samples; \
				j2 = *index_end_samples % max_end_samples; \
				j3 = max_end_samples - *index_end_samples % max_end_samples - 1; \
				/* Number of steps in for loop is minimum of above */ \
				j = j1 < (j2 < j3 ? j2 : j3) ? j1 : (j2 < j3 ? j2 : j3); \
				ptr_end = end_samples + j; \
				for(ptr_before = end_samples - 1, ptr_after = end_samples + 1; ptr_after <= ptr_end; ptr_after++, ptr_before--, sinc_table++) \
					*dst += (DTYPE) *sinc_table * (*ptr_before + *ptr_after); \
				if(j2 <= j1 && j2 < j3) { \
					ptr_before += max_end_samples; \
					j = j1 - j2; \
				} else if(j3 <= j1 && j3 < j2) { \
					ptr_after -= max_end_samples; \
					j = j1 - j3; \
				} else \
					j = 0; \
				ptr_end = ptr_after + j; \
				while(ptr_after < ptr_end) { \
					*dst += (DTYPE) *sinc_table * (*ptr_before + *ptr_after); \
					ptr_after++; \
					ptr_before--; \
					sinc_table++; \
				} \
				/* Now deal with dependence of output sample on current input samples after and end_samples before center of sinc table */ \
				j2 = 1 + (j2 - j1 - 1 + max_end_samples) % max_end_samples; \
				j1 = max_end_samples / 2 - j1; \
				j = j1 < j2 ? j1 : j2; \
				ptr_after = (DTYPE COMPLEX *) src; \
				ptr_end = ptr_after + j; \
				while(ptr_after < ptr_end) { \
					*dst += (DTYPE) *sinc_table * (*ptr_before + *ptr_after); \
					ptr_after++; \
					ptr_before--; \
					sinc_table++; \
				} \
 \
				j = j1 - j2; \
				ptr_before += max_end_samples; \
				ptr_end += j; \
				while(ptr_after < ptr_end) { \
					*dst += (DTYPE) *sinc_table * (*ptr_before + *ptr_after); \
					ptr_after++; \
					ptr_before--; \
					sinc_table++; \
				} \
			} else { \
				/*
				 * sinc table is centered in src. First, deal with dependence of output samples on current
				 * input samples before and after center of sinc table.
				 */ \
				*dst = (DTYPE) *sinc_table * *(src + *index_end_samples - index_end_samples[1] - 1); \
				sinc_table++; \
				j1 = max_end_samples / 2; \
				j2 = *index_end_samples - index_end_samples[1] - 1; \
				j = j1 < j2 ? j1 : j2; \
				ptr_end = (DTYPE COMPLEX *) src + *index_end_samples - index_end_samples[1] - 1 + j; \
				for(ptr_before = ptr_end - j - 1, ptr_after = ptr_end - j + 1; ptr_after <= ptr_end; ptr_after++, ptr_before--, sinc_table++) \
					*dst += (DTYPE) *sinc_table * (*ptr_before + *ptr_after); \
 \
				/* Now deal with dependence of output sample on current input samples after and end_samples before center of sinc table */ \
				j1 -= j2; \
				j2 = 1 + index_end_samples[1] % max_end_samples; \
				j = j1 < j2 ? j1 : j2; \
				ptr_before = end_samples_end; \
				ptr_end = ptr_after + j; \
				while(ptr_after < ptr_end) { \
					*dst += (DTYPE) *sinc_table * (*ptr_before + *ptr_after); \
					ptr_after++; \
					ptr_before--; \
					sinc_table++; \
				} \
 \
				j = j1 - j2; \
				ptr_before += max_end_samples; \
				ptr_end += j; \
				while(ptr_after < ptr_end) { \
					*dst += (DTYPE) *sinc_table * (*ptr_before + *ptr_after); \
					ptr_after++; \
					ptr_before--; \
					sinc_table++; \
				} \
			} \
			/* We've now reached the end of the sinc table. Move the pointer back. */ \
			sinc_table -= (1 + max_end_samples / 2); \
 \
			/* Also need to increment end_samples. *index_end_samples is used to find our place in src, so it is reduced later */ \
			end_samples += (inv_cadence - ((*index_end_samples % max_end_samples) + inv_cadence < max_end_samples ? 0 : max_end_samples)); \
			*index_end_samples += inv_cadence; \
		} \
	} \
	/* Move *index_end_samples back to the appropriate location within end_samples */ \
	*index_end_samples %= max_end_samples; \
 \
	/* Store current input samples that we will need later in end_samples */ \
	end_samples_end += ((index_end_samples[1] + src_size) % max_end_samples - index_end_samples[1] % max_end_samples); \
	index_end_samples[1] = (index_end_samples[1] + src_size) % max_end_samples; \
	src += (src_size - 1); \
	j = index_end_samples[1] + 1 < (gint32) src_size ? index_end_samples[1] + 1 : (gint32) src_size; \
	for(k = 0; k < j; k++, src--, end_samples_end--) \
		*end_samples_end = *src; \
 \
	/* A second for loop is necessary in case we hit the boundary of end_samples before we've stored the end samples */ \
	end_samples_end += max_end_samples; \
	j = index_end_samples[1] + 1 < (gint32) src_size ? ((gint32) src_size < max_end_samples ? (gint32) src_size : max_end_samples)  - index_end_samples[1] - 1 : 0; \
	for(k = 0; k < j; k++, src--, end_samples_end--) \
		*end_samples_end = *src; \
 \
	/* record how many samples are stored in end_samples */ \
	*num_end_samples += src_size; \
	if(*num_end_samples > max_end_samples) \
		*num_end_samples = max_end_samples; \
}


DEFINE_SINC_DOWNSAMPLE(float, )
DEFINE_SINC_DOWNSAMPLE(double, )
DEFINE_SINC_DOWNSAMPLE(float, complex)
DEFINE_SINC_DOWNSAMPLE(double, complex)


/* Based on given parameters, this function calls the proper resampling function */
static void resample(const void *src, guint64 src_size, void *dst, guint64 dst_size, gint unit_size, enum gstlal_resample_data_type data_type, gint32 cadence, gint32 inv_cadence, guint quality, void *dxdt0, void *end_samples, gint32 leading_samples, gint32 *num_end_samples, gint32 *index_end_samples, gint32 max_end_samples, gint32 sinc_length, double *sinc_table, gboolean *produced_outbuf) {

	/* Sanity checks */
	g_assert_cmpuint(src_size % unit_size, ==, 0);
	g_assert_cmpuint(dst_size % unit_size, ==, 0);
	g_assert(cadence > 1 || inv_cadence > 1);

	/* convert buffer sizes to number of samples */
	src_size /= unit_size;
	dst_size /= unit_size;

	/* 
	 * cadence is # of output samples per input sample, so if cadence > 1, we are upsampling.
	 * quality (0 - 3) is the degree of the polynomial used to interpolate between points,
	 * so quality = 0 means we are just copying input samples n times.
	 */
	if(cadence > 1 && quality == 0) {
		switch(unit_size) {
		case 1:
			const_upsample_8(src, dst, src_size, cadence);
			break;
		case 2:
			const_upsample_16(src, dst, src_size, cadence);
			break;
		case 4:
			const_upsample_32(src, dst, src_size, cadence);
			break;
		case 8:
			const_upsample_64(src, dst, src_size, cadence);
			break;
		default:
			const_upsample_other(src, dst, src_size, unit_size, cadence);
			break;
		}

	} else if(cadence > 1 && quality == 1) {
		switch(data_type) {
		case GSTLAL_RESAMPLE_F32:
			linear_upsample_float(src, dst, src_size, cadence, end_samples, num_end_samples);
			break;
		case GSTLAL_RESAMPLE_F64:
			linear_upsample_double(src, dst, src_size, cadence, end_samples, num_end_samples);
			break;
		case GSTLAL_RESAMPLE_Z64:
			linear_upsample_floatcomplex(src, dst, src_size, cadence, end_samples, num_end_samples);
			break;
		case GSTLAL_RESAMPLE_Z128:
			linear_upsample_doublecomplex(src, dst, src_size, cadence, end_samples, num_end_samples);
			break;
		default:
			g_assert_not_reached();
			break;
		}

	} else if(cadence > 1 && quality == 2) {
		switch(data_type) {
		case GSTLAL_RESAMPLE_F32:
			quadratic_upsample_float(src, dst, src_size, cadence, end_samples, num_end_samples);
			break;
		case GSTLAL_RESAMPLE_F64:
			quadratic_upsample_double(src, dst, src_size, cadence, end_samples, num_end_samples);
			break;
		case GSTLAL_RESAMPLE_Z64:
			quadratic_upsample_floatcomplex(src, dst, src_size, cadence, end_samples, num_end_samples);
			break;
		case GSTLAL_RESAMPLE_Z128:
			quadratic_upsample_doublecomplex(src, dst, src_size, cadence, end_samples, num_end_samples);
			break;
		default:
			g_assert_not_reached();
			break;
		}

	} else if(cadence > 1 && quality == 3) {
		switch(data_type) {
		case GSTLAL_RESAMPLE_F32:
			cubic_upsample_float(src, dst, src_size, cadence, dxdt0, end_samples, num_end_samples);
			break;
		case GSTLAL_RESAMPLE_F64:
			cubic_upsample_double(src, dst, src_size, cadence, dxdt0, end_samples, num_end_samples);
			break;
		case GSTLAL_RESAMPLE_Z64:
			cubic_upsample_floatcomplex(src, dst, src_size, cadence, dxdt0, end_samples, num_end_samples);
			break;
		case GSTLAL_RESAMPLE_Z128:
			cubic_upsample_doublecomplex(src, dst, src_size, cadence, dxdt0, end_samples, num_end_samples);
			break;
		default:
			g_assert_not_reached();
			break;
		}

	} else if(cadence > 1 && quality > 3) {
		switch(data_type) {
		case GSTLAL_RESAMPLE_F32:
			sinc_upsample_float(src, dst, (gint64) src_size, (gint64) dst_size, cadence, end_samples, num_end_samples, index_end_samples, max_end_samples, sinc_length, sinc_table, produced_outbuf);
			break;
		case GSTLAL_RESAMPLE_F64:
			sinc_upsample_double(src, dst, (gint64) src_size, (gint64) dst_size, cadence, end_samples, num_end_samples, index_end_samples, max_end_samples, sinc_length, sinc_table, produced_outbuf);
			break;
		case GSTLAL_RESAMPLE_Z64:
			sinc_upsample_floatcomplex(src, dst, (gint64) src_size, (gint64) dst_size, cadence, end_samples, num_end_samples, index_end_samples, max_end_samples, sinc_length, sinc_table, produced_outbuf);
			break;
		case GSTLAL_RESAMPLE_Z128:
			sinc_upsample_doublecomplex(src, dst, (gint64) src_size, (gint64) dst_size, cadence, end_samples, num_end_samples, index_end_samples, max_end_samples, sinc_length, sinc_table, produced_outbuf);
			break;
		default:
			g_assert_not_reached();
			break;
		}

	/* 
	 * inv_cadence is # of input samples per output sample, so if inv_cadence > 1, we are downsampling.
	 * The meaning of "quality" when downsampling is: if quality = 0, we pick every nth input sample,
	 * if 1 <= quality <= 3, we take an average, and if quality >= 4, we filter inputs with a sinc table.
	 */

	} else if(inv_cadence > 1 && quality == 0) {
		switch(unit_size) {
		case 1:
			downsample_8(src, dst, dst_size, inv_cadence, leading_samples);
			break;
		case 2:
			downsample_16(src, dst, dst_size, inv_cadence, leading_samples);
			break;
		case 4:
			downsample_32(src, dst, dst_size, inv_cadence, leading_samples);
			break;
		case 8:
			downsample_64(src, dst, dst_size, inv_cadence, leading_samples);
			break;
		default:
			downsample_other(src, dst, dst_size, unit_size, inv_cadence, leading_samples);
			break;
		}

	} else if(inv_cadence > 1 && quality > 0 && quality < 4) {
		switch(data_type) {
		case GSTLAL_RESAMPLE_F32:
			avg_downsample_float(src, dst, src_size, dst_size, inv_cadence, leading_samples, end_samples, num_end_samples);
			break;
		case GSTLAL_RESAMPLE_F64:
			avg_downsample_double(src, dst, src_size, dst_size, inv_cadence, leading_samples, end_samples, num_end_samples);
			break;
		case GSTLAL_RESAMPLE_Z64:
			avg_downsample_floatcomplex(src, dst, src_size, dst_size, inv_cadence, leading_samples, end_samples, num_end_samples);
			break;
		case GSTLAL_RESAMPLE_Z128:
			avg_downsample_doublecomplex(src, dst, src_size, dst_size, inv_cadence, leading_samples, end_samples, num_end_samples);
			break;
		default:
			g_assert_not_reached();
			break;
		}
	} else if(inv_cadence > 1 && quality > 3) {
		switch(data_type) {
		case GSTLAL_RESAMPLE_F32:
			sinc_downsample_float(src, dst, src_size, dst_size, inv_cadence, leading_samples, end_samples, num_end_samples, index_end_samples, max_end_samples, sinc_table);
			break;
		case GSTLAL_RESAMPLE_F64:
			sinc_downsample_double(src, dst, src_size, dst_size, inv_cadence, leading_samples, end_samples, num_end_samples, index_end_samples, max_end_samples, sinc_table);
			break;
		case GSTLAL_RESAMPLE_Z64:
			sinc_downsample_floatcomplex(src, dst, src_size, dst_size, inv_cadence, leading_samples, end_samples, num_end_samples, index_end_samples, max_end_samples, sinc_table);
			break;
		case GSTLAL_RESAMPLE_Z128:
			sinc_downsample_doublecomplex(src, dst, src_size, dst_size, inv_cadence, leading_samples, end_samples, num_end_samples, index_end_samples, max_end_samples, sinc_table);
			break;
		default:
			g_assert_not_reached();
			break;
		}
	} else
		g_assert_not_reached();
}


/*
 * Prepare the element for processing data.  This is not the start() function
 * of the gstbasetransform class.  This function is to be called after the
 * element knows about the caps and properties
 */


static void prepare_element(GSTLALResample *element) {

	/* First, free stuff if it has already been allocated. */
	if(element->sinc_table) {
		g_free(element->sinc_table);
		element->sinc_table = NULL;
	}
	if(element->end_samples) {
		g_free(element->end_samples);
		element->end_samples = NULL;
	}
	if(element->index_end_samples) {
		g_free(element->index_end_samples);
		element->index_end_samples = NULL;
	}

	gint32 cadence = element->rate_out / element->rate_in;
	gint32 inv_cadence = element->rate_in / element->rate_out;

	if(element->rate_in > element->rate_out && element->quality > 3) {
		/*
		 * In this case, we are filtering inputs with a sinc table and then downsampling.
		 * max_end_samples is the maximum number of samples that could need to be stored
		 * between buffers. It is one less than the length of the sinc table in samples.
		 * The sinc table is tapered at the ends using a DPSS window, a Kaiser window, or
		 * a Dolph-Chebyshev window, depending on the choice of the user. The cutoff
		 * frequency is below the Nyquist frequency by half the width of the main lobe of
		 * the window function, in order to minimize aliasing.
		 */
		int sinc_length_at_low_rate = (int) (element->quality == 4) * SHORT_SINC_LENGTH + (int) (element->quality == 5) * LONG_SINC_LENGTH;
		element->max_end_samples = (((gint32) sinc_length_at_low_rate * inv_cadence) / 2) * 2;
		/* end_samples stores input samples needed to produce output with the next buffer(s) */
		element->end_samples = g_malloc(element->max_end_samples * element->unit_size);

		/* index_end_samples records locations in end_samples of the next output sample and the newest sample in end_samples. */
		element->index_end_samples = g_malloc(2 * sizeof(gint32));

		/* To save memory, we use symmetry and record only half of the sinc table */
		element->sinc_length = element->max_end_samples + 1;
		element->sinc_table = g_malloc((1 + element->sinc_length / 2) * sizeof(double));
		*(element->sinc_table) = 1.0;
		gint32 i;
		double sin_arg;
		/* Frequency resolution in units of frequency bins of the sinc table */ \
		double alpha = (1 + sinc_length_at_low_rate / 24.0); \
		/* Low-pass cutoff frequency as a fraction of the sampling frequency of the sinc table */ \
		double f_cut = 0.5 / inv_cadence - alpha / element->sinc_length; \
		for(i = 1; i <= element->sinc_length / 2; i++) {
			sin_arg = 2 * M_PI * f_cut * i;
			element->sinc_table[i] = sin(sin_arg) / sin_arg;
		}

		/* Apply a window function */
		switch(element->window) {
		case GSTLAL_RESAMPLE_DPSS:
			dpss_double(element->sinc_length, alpha, 5.0, element->sinc_table, TRUE);
			break;
		case GSTLAL_RESAMPLE_KAISER:
			kaiser_double(element->sinc_length, M_PI * alpha, element->sinc_table, TRUE);
			break;
		case GSTLAL_RESAMPLE_DOLPH_CHEBYSHEV:
			DolphChebyshev_double(element->sinc_length, alpha, element->sinc_table, TRUE);
			break;
		default:
			GST_ERROR_OBJECT(element, "Invalid window type.  See properties for appropriate window types.");
			g_assert_not_reached();
			break;
		}

		/* normalize sinc_table to make the DC gain exactly 1 */
		double normalization = 1.0;
		for(i = 1; i <= element->max_end_samples / 2; i++)
			normalization += 2 * element->sinc_table[i];

		for(i = 0; i <= element->max_end_samples / 2; i++)
			element->sinc_table[i] /= normalization;

		/* tell the downsampling function about start of stream (or caps update) */
		*element->index_end_samples = -1;

	} else if(element->rate_in < element->rate_out && element->quality > 3) {
		/*
		 * In this case, we are filtering inputs with a sinc table and upsampling.
		 * max_end_samples is the maximum number of samples that could need to be stored
		 * between buffers. It is slightly shorter than the length of the sinc table in time.
		 * The sinc table is tapered at the ends using a DPSS window, a Kaiser window, or
		 * a Dolph-Chebyshev window, depending on the choice of the user. The cutoff
		 * frequency is below the Nyquist frequency by half the width of the main lobe of the
		 * window function. To upsample, the inputs are shifted relative to the sinc filter
		 * by one sample period of the upsampled rate for each consecutive output sample.
		 * This creates a smooth data stream that contains the original frequency content
		 * without adding additional frequency content due to imaging effects.
		 */
		int sinc_length_at_low_rate = (int) (element->quality == 4) * SHORT_SINC_LENGTH + (int) (element->quality == 5) * LONG_SINC_LENGTH;
		element->max_end_samples = (gint32) (element->quality == 4) * SHORT_SINC_LENGTH + (gint32) (element->quality == 5) * LONG_SINC_LENGTH;
		element->sinc_length = (gint32) (1 + element->max_end_samples * cadence);

		/* end_samples stores input samples needed to produce output with the next buffer(s) */
		element->end_samples = g_malloc(element->max_end_samples * element->unit_size);

		/* index_end_samples records locations in end_samples of the next output sample and the newest sample in end_samples. */
		element->index_end_samples = g_malloc(2 * sizeof(gint32));

		/* To save memory, we use symmetry and record only half of the sinc table */
		element->sinc_table = g_malloc((1 + element->sinc_length / 2) * sizeof(double));
		*(element->sinc_table) = 1.0;
		gint32 i, j;
		double sin_arg;
		/* Frequency resolution in units of frequency bins of the sinc table */ \
		double alpha = (1 + sinc_length_at_low_rate / 24.0); \
		/* Low-pass cutoff frequency as a fraction of the sampling frequency of the sinc table */ \
		double f_cut = 0.5 / cadence - alpha / element->sinc_length; \
		for(i = 1; i <= element->sinc_length / 2; i++) {
			sin_arg = 2 * M_PI * f_cut * i;
			element->sinc_table[i] = sin(sin_arg) / sin_arg;
		}
		/* Since sinc_table's length is one more than a multiple of cadence, we need to account for times when an extra input is being filtered */
		element->sinc_table[element->sinc_length / 2] /= 2;

		/* Apply a window function */
		switch(element->window) {
		case GSTLAL_RESAMPLE_DPSS:
			dpss_double(element->sinc_length, alpha, 5.0, element->sinc_table, TRUE);
			break;
		case GSTLAL_RESAMPLE_KAISER:
			kaiser_double(element->sinc_length, M_PI * alpha, element->sinc_table, TRUE);
			break;
		case GSTLAL_RESAMPLE_DOLPH_CHEBYSHEV:
			DolphChebyshev_double(element->sinc_length, alpha, element->sinc_table, TRUE);
			break;
		default:
			GST_ERROR_OBJECT(element, "Invalid window type.  See properties for appropriate window types.");
			g_assert_not_reached();
			break;
		}

		/*
		 * Normalize sinc_table to make the DC gain exactly 1. We need to account for the fact
		 * that the density of input samples is less than the density of samples in the sinc table
		 */
		double normalization;
		for(i = 0; i < (cadence + 1) / 2; i++) {
			normalization = 0.0;
			for(j = i; j <= element->sinc_length / 2; j += cadence)
				normalization += element->sinc_table[j];
			for(j = cadence - i; j <= element->sinc_length / 2; j += cadence)
				normalization += element->sinc_table[j];
			for(j = i; j <= element->sinc_length / 2; j += cadence)
				element->sinc_table[j] /= normalization;
			if(i) {
				for(j = cadence - i; j <= element->sinc_length / 2; j += cadence)
					element->sinc_table[j] /= normalization;
			}
		}
		/* If cadence is even, we need to account for one more normalization without "over-normalizing." */
		if(!((cadence) % 2)) {
			normalization = 0.0;
			for(j = cadence / 2; j <= element->sinc_length / 2; j += cadence)
				normalization += 2 * element->sinc_table[j];
			for(j = cadence / 2; j <= element->sinc_length / 2; j += cadence)
				element->sinc_table[j] /= normalization;
		}

	} else if(element->rate_out > element->rate_in && (element->quality == 2 || element->quality == 3) && !element->end_samples)
		element->end_samples = g_malloc(2 * element->unit_size);
	else if(element->quality > 0 && !element->end_samples)
		element->end_samples = g_malloc(element->unit_size);
}


/*
 * If transform_size() is unable to correctly compute the size of an
 * outgoing buffer, this function is called.
 */


static gssize get_outbuf_size(GSTLALResample *element, guint64 inbuf_size) {

	/* Convert from bytes to samples */
	inbuf_size /= element->unit_size;

	gint32 cadence = element->rate_out / element->rate_in;
	gint32 inv_cadence = element->rate_in / element->rate_out;

	gint64 outbuf_size;
	if(element->rate_out > element->rate_in) {

		/*
		 * We are upsampling.  If using any interpolation, each input buffer leaves one or
		 * two samples at the end to add to the next buffer.  If these are absent, we need
		 * to reduce the output buffer size.
		 */

		outbuf_size = cadence * inbuf_size;

		switch(element->quality) {
		case 0:
			break;

		case 1:
		case 2:
			if(element->num_end_samples == 0)
				outbuf_size -= cadence;
			break;

		case 3:
			if(element->num_end_samples == 0)
				outbuf_size -= 2 * cadence;
			else if(element->num_end_samples == 1)
				outbuf_size -= cadence;
			break;

		case 4:
		case 5:
			/* We are filtering with a sinc function */
			if(!element->produced_outbuf) {
				gint32 total_samples = element->num_end_samples + (gint32) inbuf_size;
				if(total_samples > element->max_end_samples) {
					gint32 half_sinc_length = element->max_end_samples * cadence / 2;
					outbuf_size = total_samples * cadence - half_sinc_length;
				} else
					outbuf_size = 0;
			}
			break;

		default:
			g_assert_not_reached();
		}

	} else if(element->rate_in > element->rate_out) {
		/*
		 * We are downsampling.  Assuming we are simply picking every nth sample, the number
		 * of samples in the output buffer is the number of samples in the input buffer that
		 * carry timestamps that are multiples of the output sampling period.
		 */
		outbuf_size = (inbuf_size - element->leading_samples + inv_cadence - 1) / inv_cadence;

		/* We now adjust the size if we are applying a Tukey window when downsampling */
		if(element->quality >= 1 && element->quality <= 3) {
			/*
			 * There must be enough input samples that come after the last timestamp that
			 * is a multiple of the output sampling period.  If not, we need to remove a
			 * sample from the output buffer.
			 */
			if(outbuf_size > 0 && (gint32) (inbuf_size - element->leading_samples - 1) % inv_cadence < inv_cadence / 2)
				outbuf_size--;
			/*
			 * Check if there will be an outgoing sample on this buffer before the
			 * presentation timestamp of the input buffer.  If so, add a sample.
			 */
			if(element->num_end_samples > inv_cadence / 2 && element->num_end_samples + (gint32) inbuf_size > inv_cadence / 2 * 2)
				outbuf_size += 1;
		} else if(element->quality >= 4 && element->num_end_samples == element->max_end_samples)
			outbuf_size = (inbuf_size + inv_cadence - (element->max_end_samples / 2 + element->leading_samples) % inv_cadence - 1) / inv_cadence;
		else if(element->quality >= 4) {
			/* produce output only if enough data is available to completely fill the sinc table */
			if((gint32) inbuf_size + element->num_end_samples >= element->max_end_samples)
				outbuf_size = (inbuf_size + element->num_end_samples - element->first_leading_samples - element->max_end_samples / 2 + inv_cadence - 1) / inv_cadence;
			else
				outbuf_size = 0;
		}
	} else
		/* input and output rates are equal, so the element runs on passthrough mode */
		outbuf_size = inbuf_size;

	if(outbuf_size < 0)
		outbuf_size = 0;

	/* Convert from samples to bytes */
	outbuf_size *= element->unit_size;

	return (gssize) outbuf_size;
}


/*
 * set the metadata on an output buffer
 */


static void set_metadata(GSTLALResample *element, GstBuffer *buf, guint64 outsamples, gboolean gap) {

	GST_BUFFER_OFFSET(buf) = element->next_out_offset;
	element->next_out_offset += outsamples;
	GST_BUFFER_OFFSET_END(buf) = element->next_out_offset;
	if(element->zero_latency) {
		GST_BUFFER_PTS(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET(buf) - element->offset0 + (element->rate_out * element->max_end_samples / 2) / element->rate_in, GST_SECOND, element->rate_out);
		GST_BUFFER_DURATION(buf) = gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET_END(buf) - element->offset0, GST_SECOND, element->rate_out) - gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET(buf) - element->offset0, GST_SECOND, element->rate_out);
	} else {
		GST_BUFFER_PTS(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET(buf) - element->offset0, GST_SECOND, element->rate_out);
		GST_BUFFER_DURATION(buf) = element->t0 + gst_util_uint64_scale_int_round(GST_BUFFER_OFFSET_END(buf) - element->offset0, GST_SECOND, element->rate_out) - GST_BUFFER_PTS(buf);
	}
	if(G_UNLIKELY(element->need_discont)) {
		GST_BUFFER_FLAG_SET(buf, GST_BUFFER_FLAG_DISCONT);
		element->need_discont = FALSE;
	}
	if(gap)
		GST_BUFFER_FLAG_SET(buf, GST_BUFFER_FLAG_GAP);
	else
		GST_BUFFER_FLAG_UNSET(buf, GST_BUFFER_FLAG_GAP);
}


/*
 * ============================================================================
 *
 *			   GStreamer Boiler Plate
 *
 * ============================================================================
 */


#define CAPS \
	"audio/x-raw, " \
	"rate = (int) [1, MAX], " \
	"channels = (int) 1, " \
	"format = (string) {"GST_AUDIO_NE(F32)", "GST_AUDIO_NE(F64)", "GST_AUDIO_NE(Z64)", "GST_AUDIO_NE(Z128)"}, " \
	"layout = (string) interleaved, " \
	"channel-mask = (bitmask) 0"


static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SINK_NAME,
	GST_PAD_SINK,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(CAPS)
);


static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE(
	GST_BASE_TRANSFORM_SRC_NAME,
	GST_PAD_SRC,
	GST_PAD_ALWAYS,
	GST_STATIC_CAPS(CAPS)
);


G_DEFINE_TYPE(
	GSTLALResample,
	gstlal_resample,
	GST_TYPE_BASE_TRANSFORM
);


/*
 * ============================================================================
 *
 *		     GstBaseTransform Method Overrides
 *
 * ============================================================================
 */


/*
 * get_unit_size()
 */


static gboolean get_unit_size(GstBaseTransform *trans, GstCaps *caps, gsize *size) {

	/*
	 * It seems that the function gst_audio_info_from_caps() does not work for gstlal's complex formats.
	 * Therefore, a different method is used below to parse the caps.
	 */
	const gchar *format;
	static char *formats[] = {"F32LE", "F32BE", "F64LE", "F64BE", "Z64LE", "Z64BE", "Z128LE", "Z128BE"};
	gint sizes[] = {4, 4, 8, 8, 8, 8, 16, 16};

	GstStructure *str = gst_caps_get_structure(caps, 0);
	g_assert(str);

	if(gst_structure_has_field(str, "format")) {
		format = gst_structure_get_string(str, "format");
	} else {
		GST_ERROR_OBJECT(trans, "No format! Cannot infer unit size.\n");
		return FALSE;
	}
	int test = 0;
	for(unsigned int i = 0; i < sizeof(formats) / sizeof(*formats); i++) {
		if(!strcmp(format, formats[i])) {
			*size = sizes[i];
			test++;
		}
	}
	if(test != 1)
		GST_WARNING_OBJECT(trans, "unit size not properly set");

	return TRUE;
}


/*
 * transform_caps()
 */


static GstCaps *transform_caps(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, GstCaps *filter) {

	guint n;

	caps = gst_caps_copy(caps);

	switch(direction) {

	case GST_PAD_SRC:
	case GST_PAD_SINK:
		/*
		 * Source and sink pad formats are the same except that
		 * the rate can change to any integer value going in either 
		 * direction. (Really needs to be either an integer multiple
		 * or an integer divisor of the rate on the other pad, but 
		 * that requirement is not enforced here).
		 */

		for(n = 0; n < gst_caps_get_size(caps); n++) {
			GstStructure *s = gst_caps_get_structure(caps, n);
			const GValue *v = gst_structure_get_value(s, "rate");

			if(!(GST_VALUE_HOLDS_INT_RANGE(v) || G_VALUE_HOLDS_INT(v)))
				GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid type for rate in caps"));
			gst_structure_set(s, "rate", GST_TYPE_INT_RANGE, 1, G_MAXINT, NULL);
		}
		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		gst_caps_unref(caps);
		return GST_CAPS_NONE;

	default:
		g_assert_not_reached();
	}
	return caps;
}


/*
 * set_caps()
 */


static gboolean set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps) {

	GSTLALResample *element = GSTLAL_RESAMPLE(trans);
	gboolean success = TRUE;
	gint32 rate_in, rate_out;
	gsize unit_size;

	/*
	 * parse the caps
	 */

	success &= get_unit_size(trans, incaps, &unit_size);
	GstStructure *str = gst_caps_get_structure(incaps, 0);
	const gchar *name = gst_structure_get_string(str, "format");
	success &= (name != NULL);
	success &= gst_structure_get_int(str, "rate", &rate_in);
	success &= gst_structure_get_int(gst_caps_get_structure(outcaps, 0), "rate", &rate_out);
	if(!success)
		GST_ERROR_OBJECT(element, "unable to parse caps.  input caps = %" GST_PTR_FORMAT " output caps = %" GST_PTR_FORMAT, incaps, outcaps);

	/* require the output rate to be an integer multiple or divisor of the input rate */
	success &= !(rate_out % rate_in) || !(rate_in % rate_out);
	if((rate_out % rate_in) && (rate_in % rate_out))
		GST_ERROR_OBJECT(element, "output rate is not an integer multiple or divisor of input rate.  input caps = %" GST_PTR_FORMAT " output caps = %" GST_PTR_FORMAT, incaps, outcaps);

	if(success) {

		/*
		 * record stream parameters
		 */

		if(!strcmp(name, GST_AUDIO_NE(F32))) {
			element->data_type = GSTLAL_RESAMPLE_F32;
			g_assert_cmpuint(unit_size, ==, 4);
		} else if(!strcmp(name, GST_AUDIO_NE(F64))) {
			element->data_type = GSTLAL_RESAMPLE_F64;
			g_assert_cmpuint(unit_size, ==, 8);
		} else if(!strcmp(name, GST_AUDIO_NE(Z64))) {
			element->data_type = GSTLAL_RESAMPLE_Z64;
			g_assert_cmpuint(unit_size, ==, 8);
		} else if(!strcmp(name, GST_AUDIO_NE(Z128))) {
			element->data_type = GSTLAL_RESAMPLE_Z128;
			g_assert_cmpuint(unit_size, ==, 16);
		} else
			g_assert_not_reached();

		element->rate_in = rate_in;
		element->rate_out = rate_out;
		element->unit_size = unit_size;

		/*
		 * record everything else that depends on caps (as well as properties, which should be set by now)
		 */

		prepare_element(element);
	}
	return success;
}


/*
 * transform_size()
 */


static gboolean transform_size(GstBaseTransform *trans, GstPadDirection direction, GstCaps *caps, gsize size, GstCaps *othercaps, gsize *othersize) {

	GSTLALResample *element = GSTLAL_RESAMPLE(trans);
	gint32 cadence = element->rate_out / element->rate_in;
	gint32 inv_cadence = element->rate_in / element->rate_out;

	gsize unit_size;
	if(!get_unit_size(trans, caps, &unit_size))
		return FALSE;

	if(element->unit_size == 0)
		element->unit_size = unit_size;
	else
		g_assert_cmpint((gint) unit_size, ==, element->unit_size);

	if(G_UNLIKELY(size % unit_size)) {
		GST_DEBUG_OBJECT(element, "buffer size %" G_GSIZE_FORMAT " is not a multiple of %" G_GSIZE_FORMAT, size, unit_size);
		return FALSE;
	}

	switch(direction) {
	case GST_PAD_SRC:
		/*
		 * compute samples needed on sink pad from sample count on source pad.
		 * size = # of samples needed on source pad
		 * cadence = # of output samples per input sample
		 * inv_cadence = # of input samples per output sample
		 */

		if(inv_cadence > 1)
			*othersize = size * inv_cadence;
		else
			*othersize = size / cadence;
		break;

	case GST_PAD_SINK:
		/*
		 * compute samples to be produced on source pad from sample
		 * count available on sink pad.
		 * size = # of samples available on sink pad
		 * cadence = # of output samples per input sample
		 * inv_cadence = # of input samples per output sample
		 */

		*othersize = get_outbuf_size(element, (guint64) size);

		/*
		 * If we are downsampling and we haven't gotten any input data yet, we need to know
		 * the presentation timestamp of the first input buffer to compute the size of the
		 * output buffer.  So we'll have to figure the output buffer size when transform()
		 * gets called.  For now, just allocate as much memory as we could possibly need.
		 */

		if(element->rate_in > element->rate_out && !GST_CLOCK_TIME_IS_VALID(element->t0))
			*othersize = size / inv_cadence + (gsize) (element->unit_size * (element->num_end_samples / inv_cadence + 2));
		break;

	case GST_PAD_UNKNOWN:
		GST_ELEMENT_ERROR(trans, CORE, NEGOTIATION, (NULL), ("invalid direction GST_PAD_UNKNOWN"));
		return FALSE;

	default:
		g_assert_not_reached();
	}

	return TRUE;
}


/*
 * transform()
 */


static GstFlowReturn transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf) {

	GSTLALResample *element = GSTLAL_RESAMPLE(trans);
	GstMapInfo inmap, outmap;
	GstFlowReturn result = GST_FLOW_OK;

	gint32 cadence = element->rate_out / element->rate_in;
	gint32 inv_cadence = element->rate_in / element->rate_out;

	/*
	 * check for discontinuity
	 */

	if(G_UNLIKELY(GST_BUFFER_IS_DISCONT(inbuf) || GST_BUFFER_OFFSET(inbuf) != element->next_in_offset || !GST_CLOCK_TIME_IS_VALID(element->t0))) {
		element->t0 = GST_BUFFER_PTS(inbuf);
		if(element->rate_in > element->rate_out && abs(GST_BUFFER_PTS(inbuf) - gst_util_uint64_scale_round(gst_util_uint64_scale_round(GST_BUFFER_PTS(inbuf), element->rate_out, 1000000000), 1000000000, element->rate_out)) >= 500000000 / element->rate_in)
			element->t0 = gst_util_uint64_scale_round(gst_util_uint64_scale_ceil(GST_BUFFER_PTS(inbuf), element->rate_out, 1000000000), 1000000000, element->rate_out);
		else
			element->t0 = gst_util_uint64_scale_round(gst_util_uint64_scale_round(GST_BUFFER_PTS(inbuf), element->rate_out, 1000000000), 1000000000, element->rate_out);
		element->offset0 = element->next_out_offset = gst_util_uint64_scale_ceil(GST_BUFFER_OFFSET(inbuf), element->rate_out, element->rate_in);
		element->need_discont = TRUE;
		element->dxdt0 = 0.0;
		element->num_end_samples = 0;
		element->produced_outbuf = FALSE;
		if(element->rate_in > element->rate_out) {
			/* leading_samples is the number of input samples that come before the first timestamp that is a multiple of the output sampling period */
			element->leading_samples = gst_util_uint64_scale_int_round(GST_BUFFER_PTS(inbuf), element->rate_in, 1000000000) % inv_cadence;
			element->leading_samples = (inv_cadence - element->leading_samples) % inv_cadence;
		}
		element->first_leading_samples = element->leading_samples;

		/* We need to adjust the output buffer size */
		gst_buffer_set_size(outbuf, get_outbuf_size(element, (guint64) gst_buffer_get_size(inbuf)));
	}

	element->next_in_offset = GST_BUFFER_OFFSET_END(inbuf);

	/*
	 * process buffer
	 */

	gst_buffer_map(inbuf, &inmap, GST_MAP_READ);

	if(!GST_BUFFER_FLAG_IS_SET(inbuf, GST_BUFFER_FLAG_GAP) && inmap.size != 0) {

		/*
		 * input data is relevant.
		 */

		gst_buffer_map(outbuf, &outmap, GST_MAP_WRITE);
		resample(inmap.data, inmap.size, outmap.data, outmap.size, element->unit_size, element->data_type, cadence, inv_cadence, element->quality, (void *) &element->dxdt0, (void *) element->end_samples, element->leading_samples, &element->num_end_samples, element->index_end_samples, element->max_end_samples, element->sinc_length, element->sinc_table, &element->produced_outbuf);
		set_metadata(element, outbuf, outmap.size / element->unit_size, FALSE);
		gst_buffer_unmap(outbuf, &outmap);
	} else {
		/*
		 * input is gap.
		 */

		GST_BUFFER_FLAG_SET(outbuf, GST_BUFFER_FLAG_GAP);
		gst_buffer_map(outbuf, &outmap, GST_MAP_WRITE);
		memset(outmap.data, 0, outmap.size);
		set_metadata(element, outbuf, outmap.size / element->unit_size, TRUE);
		gst_buffer_unmap(outbuf, &outmap);
	}

	gst_buffer_unmap(inbuf, &inmap);

	if(element->rate_in > element->rate_out) {
		/* set leading_samples for the next input buffer */
		element->leading_samples = gst_util_uint64_scale_int_round(GST_BUFFER_PTS(inbuf) + GST_BUFFER_DURATION(inbuf), element->rate_in, 1000000000) % inv_cadence;
		element->leading_samples = (inv_cadence - element->leading_samples) % inv_cadence;
	}

	/*
	 * done
	 */

	return result;
}


/*
 * ============================================================================
 *
 *			  GObject Method Overrides
 *
 * ============================================================================
 */


/*
 * properties
 */


enum property {
	ARG_QUALITY = 1,
	ARG_ZERO_LATENCY,
	ARG_WINDOW
};


static void set_property(GObject *object, enum property prop_id, const GValue *value, GParamSpec *pspec) {

	GSTLALResample *element = GSTLAL_RESAMPLE(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_QUALITY:
		element->quality = g_value_get_uint(value);
		if(element->rate_in > 0)
			/* then we have already called set_caps() */
			prepare_element(element);
		break;
	case ARG_ZERO_LATENCY:
		element->zero_latency = g_value_get_boolean(value);
		break;
	case ARG_WINDOW:
		element->window = g_value_get_enum(value);
		break;
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


static void get_property(GObject *object, enum property prop_id, GValue *value, GParamSpec *pspec) {

	GSTLALResample *element = GSTLAL_RESAMPLE(object);

	GST_OBJECT_LOCK(element);

	switch (prop_id) {
	case ARG_QUALITY:
		g_value_set_uint(value, element->quality);
		break;
	case ARG_ZERO_LATENCY:
		g_value_set_boolean(value, element->zero_latency);
		break;
	case ARG_WINDOW:
		g_value_set_enum(value, element->window);
		break;
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
	}

	GST_OBJECT_UNLOCK(element);
}


/*
 * finalize()
 */


static void finalize(GObject *object) {

	GSTLALResample *element = GSTLAL_RESAMPLE(object);

	/*
	 * free resources
	 */

	if(element->sinc_table) {
		g_free(element->sinc_table);
		element->sinc_table = NULL;
	}
	if(element->end_samples) {
		g_free(element->end_samples);
		element->end_samples = NULL;
	}
	if(element->index_end_samples) {
		g_free(element->index_end_samples);
		element->index_end_samples = NULL;
	}

	/*
	 * chain to parent class' finalize() method
	 */

	G_OBJECT_CLASS(gstlal_resample_parent_class)->finalize(object);
}


/*
 * class_init()
 */


static void gstlal_resample_class_init(GSTLALResampleClass *klass) {

	GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
	GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(klass);

	transform_class->transform_caps = GST_DEBUG_FUNCPTR(transform_caps);
	transform_class->transform_size = GST_DEBUG_FUNCPTR(transform_size);
	transform_class->get_unit_size = GST_DEBUG_FUNCPTR(get_unit_size);
	transform_class->set_caps = GST_DEBUG_FUNCPTR(set_caps);
	transform_class->transform = GST_DEBUG_FUNCPTR(transform);
	transform_class->passthrough_on_same_caps = TRUE;

	gst_element_class_set_details_simple(element_class,
		"Resamples a data stream",
		"Filter/Audio",
		"Resamples a stream with adjustable quality.",
		"Aaron Viets <aaron.viets@ligo.org>"
	);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(get_property);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(finalize);

	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&src_factory));
	gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_factory));

	g_object_class_install_property(
		gobject_class,
		ARG_QUALITY,
		g_param_spec_uint(
			"quality",
			"Quality of Resampling",
			"Higher quality will reduce aliasing and imaging effects for an increased\n\t\t\t"
			"computational cost. Refer to the table below for details. The sinc table\n\t\t\t"
			"lengths given are measured at the lower sample rate.\n\n\t\t\t"
			"quality\t\tupsampling\t\tdownsampling \n\n\t\t\t"
			"0\t\tconstant\t\tpick every nth\n\t\t\t"
			"1\t\tlinear\t\t\taverage nearest n samples\n\t\t\t"
			"2\t\tquadratic spline\taverage nearest n samples\n\t\t\t"
			"3\t\tcubic spline\t\taverage nearest n samples\n\t\t\t"
			"4\t\tsinc table: 33 samples\tsinc table: 33 samples\n\t\t\t"
			"5\t\tsinc table: 193 samples\tsinc table: 193 samples\n",
			0, 5, 0,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_ZERO_LATENCY,
		g_param_spec_boolean(
			"zero-latency",
			"Zero Latency",
			"If set to true, applies a timestamp shift in order to make the latency zero.\n\t\t\t"
			"Default is false",
			FALSE,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
	g_object_class_install_property(
		gobject_class,
		ARG_WINDOW,
		g_param_spec_enum(
			"window",
			"Window Function",
			"What window function to apply to the sinc table",
			GSTLAL_RESAMPLE_WINDOW_TYPE,
			GSTLAL_RESAMPLE_DPSS,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT
		)
	);
}


/*
 * init()
 */


static void gstlal_resample_init(GSTLALResample *element) {

	element->rate_in = 0;
	element->rate_out = 0;
	element->unit_size = 0;
	element->t0 = GST_CLOCK_TIME_NONE;
	element->offset0 = GST_BUFFER_OFFSET_NONE;
	element->next_in_offset = GST_BUFFER_OFFSET_NONE;
	element->next_out_offset = GST_BUFFER_OFFSET_NONE;
	element->need_discont = TRUE;
	element->dxdt0 = 0.0;
	element->end_samples = NULL;
	element->sinc_table = NULL;
	element->index_end_samples = NULL;
	element->max_end_samples = 0;
	element->sinc_length = 0;
	element->num_end_samples = 0;
	element->produced_outbuf = FALSE;
	element->leading_samples = 0;
	gst_base_transform_set_gap_aware(GST_BASE_TRANSFORM(element), TRUE);
}


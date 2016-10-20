# Copyright (C) 2012 Madeline Wade
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

## @file

## @package coherent_null

#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#

import scipy.fftpack
import numpy
import math

import lal

#
# =============================================================================
#
#                                  Functions
#
# =============================================================================
#

def factors_to_fir_kernel(coh_facs):
	"""
	Compute a finite impulse-response filter kernel from a power
	spectral density conforming to the LAL normalization convention,
	such that if zero-mean unit-variance Gaussian random noise is fed
	into an FIR filter using the kernel the filter's output will
	possess the given PSD.  The PSD must be provided as a
	REAL8FrequencySeries object (see lal's swig binding documentation).

	The return value is the tuple (kernel, latency, sample rate).  The
	kernel is a numpy array containing the filter kernel, the latency
	is the filter latency in samples and the sample rate is in Hz.
        """
	#
	# this could be relaxed with some work
	#

	assert coh_facs.f0 == 0.0

	#
	# extract the PSD bins and determine sample rate for kernel
	#

	data = coh_facs.data.data
	sample_rate = 2 * int(round(coh_facs.f0 + len(data) * coh_facs.deltaF))

	#
	# compute the FIR kernel.  it always has an odd number of samples
	# and no DC offset.
	#

	data[0] = data[-1] = 0.0
	tmp = numpy.zeros((2 * len(data) - 1,), dtype = data.dtype)
	tmp[0] = data[0]
	tmp[1::2] = data[1:]
	data = tmp
	del tmp
	kernel = scipy.fftpack.irfft(data)
	kernel = numpy.roll(kernel, (len(kernel) - 1) / 2)

	#
	# apply a Tukey window whose flat bit is 50% of the kernel.
	# preserve the FIR kernel's square magnitude
	#

	norm_before = numpy.dot(kernel, kernel)
	kernel *= lal.CreateTukeyREAL8Window(len(kernel), .5).data.data
	kernel *= math.sqrt(norm_before / numpy.dot(kernel, kernel))

	#
	# the kernel's latency
	#

	latency = (len(kernel) + 1) / 2 - 1

	#
	# done
	#

	return kernel, latency, sample_rate

def psd_to_impulse_response(PSD1, PSD2):

	assert PSD1.f0 == PSD2.f0
	assert PSD1.deltaF == PSD2.deltaF
	assert len(PSD1.data.data) == len(PSD2.data.data)

	coh_facs_H1 = lal.CreateREAL8FrequencySeries(
		name = "",
		epoch = PSD1.epoch,
		f0 = PSD1.f0,
		deltaF = PSD1.deltaF,
		sampleUnits = lal.DimensionlessUnit,
		length = len(PSD1.data.data)
	)
	coh_facs_H1.data.data = PSD2.data.data / (PSD1.data.data + PSD2.data.data)
	coh_facs_H1.data.data[0] = coh_facs_H1.data.data[-1] = 0.0

	coh_facs_H2 = lal.REAL8FrequencySeries(
		name = "",
		epoch = PSD2.epoch,
		f0 = PSD2.f0,
		deltaF = PSD2.deltaF,
		sampleUnits = lal.DimensionlessUnit,
		length = len(PSD2.data.data)
	)
	coh_facs_H2.data = PSD1.data.data / (PSD1.data.data + PSD2.data.data)
	coh_facs_H2.data[0] = coh_facs_H2.data[-1] = 0.0

	#
	# set up fir filter
	#

	H1_impulse, H1_latency, H1_srate = factors_to_fir_kernel(coh_facs_H1)
	H2_impulse, H2_latency, H2_srate = factors_to_fir_kernel(coh_facs_H2)

	assert H1_srate == H2_srate

	return H1_impulse, H1_latency, H2_impulse, H2_latency, H1_srate

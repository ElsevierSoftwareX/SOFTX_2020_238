# Copyright (C) 2009--2013  LIGO Scientific Collaboration
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
#
# ### Review Status
#
# | Names                                          | Hash                                        | Date       |
# | -------------------------------------------    | ------------------------------------------- | ---------- |
# | Florent, Sathya, Duncan Me, Jolien, Kipp, Chad | 7536db9d496be9a014559f4e273e1e856047bf71    | 2014-04-29 |
#
# #### Actions
#
# - Performance of time slices could be improved by e.g., not using powers of 2 for time slice sample size. This might also simplify the code
# 

## @package templates

#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


import numpy
import math

import sys
from pylal import datatypes as laltypes
from pylal import lalfft
from pylal import spawaveform
import lalsimulation as lalsim


__author__ = "Kipp Cannon <kipp.cannon@ligo.org>, Chad Hanna <chad.hanna@ligo.org>, Drew Keppel <drew.keppel@ligo.org>"
__version__ = "FIXME"
__date__ = "FIXME"


#
# =============================================================================
#
#                                    Stuff
#
# =============================================================================
#

gstlal_FD_approximants = set(('IMRPhenomB', 'TaylorF2', 'TaylorF2RedSpin', 'TaylorF2RedSpinTidal'))
gstlal_TD_approximants = set(('TaylorT1', 'TaylorT2', 'TaylorT3', 'TaylorT4', 'EOBNRv2'))
gstlal_approximants = gstlal_FD_approximants | gstlal_TD_approximants


def gstlal_valid_approximant(appx_str):
	try:
		lalsim.GetApproximantFromString(str(appx_str))
	except RuntimeError:
		raise ValueError("Approximant not supported by lalsimulation: %s" % appx_str)
	if appx_str not in gstlal_approximants:
		raise ValueError("Approximant not currently supported by gstlal, but is supported by lalsimulation: %s. Please consider preparing a patch" % appx_str)


def add_quadrature_phase(fseries, n):
	"""
	From the Fourier transform of a real-valued function of time,
	compute and return the Fourier transform of the complex-valued
	function of time whose real component is the original time series
	and whose imaginary component is the quadrature phase of the real
	part.  fseries is a LAL COMPLEX16FrequencySeries and n is the
	number of samples in the original time series.
	"""
	#
	# prepare output frequency series
	#

	out_fseries = laltypes.COMPLEX16FrequencySeries(
		name = fseries.name,
		epoch = fseries.epoch,
		f0 = fseries.f0,	# caution: only 0 is supported
		deltaF = fseries.deltaF,
		sampleUnits = fseries.sampleUnits
	)

	#
	# positive frequencies include Nyquist if n is even
	#

	have_nyquist = not (n % 2)

	#
	# shuffle frequency bins
	#

	positive_frequencies = fseries.data
	positive_frequencies[0] = 0	# set DC to zero
	if have_nyquist:
		positive_frequencies[-1] = 0	# set Nyquist to 0
	zeros = numpy.zeros((len(positive_frequencies),), dtype = "cdouble")
	if have_nyquist:
		# complex transform never includes positive Nyquist
		positive_frequencies = positive_frequencies[:-1]

	out_fseries.data = numpy.concatenate((zeros, 2 * positive_frequencies[1:]))

	return out_fseries


class QuadraturePhase(object):
	"""
	A tool for generating the quadrature phase of a real-valued
	template.

	Example:

	>>> import numpy
	>>> from pylal.datatypes import REAL8TimeSeries
	>>> q = QuadraturePhase(128) # initialize for 128-sample templates
	>>> inseries = REAL8TimeSeries(deltaT = 1.0 / 128, data = numpy.cos(numpy.arange(128, dtype = "double") * 2 * numpy.pi / 128)) # one cycle of cos(t)
	>>> outseries = q(inseries) # output has cos(t) in real part, sin(t) in imaginary part
	"""

	def __init__(self, n):
		"""
		Initialize.  n is the size, in samples, of the templates to
		be processed.  This is used to pre-allocate work space.
		"""
		self.n = n
		self.fwdplan = lalfft.XLALCreateForwardREAL8FFTPlan(n, 1)
		self.revplan = lalfft.XLALCreateReverseCOMPLEX16FFTPlan(n, 1)
		self.in_fseries = lalfft.prepare_fseries_for_real8tseries(laltypes.REAL8TimeSeries(deltaT = 1.0, data = numpy.zeros((n,), dtype = "double")))

	def __call__(self, tseries):
		"""
		Transform the real-valued time series stored in tseries
		into a complex-valued time series.  The return value is a
		newly-allocated complex time series.  The input time series
		is stored in the real part of the output time series, and
		the complex part stores the quadrature phase.
		"""
		#
		# transform to frequency series
		#

		lalfft.XLALREAL8TimeFreqFFT(self.in_fseries, tseries, self.fwdplan)

		#
		# transform to complex time series
		#

		tseries = laltypes.COMPLEX16TimeSeries(data = numpy.zeros((self.n,), dtype = "cdouble"))
		lalfft.XLALCOMPLEX16FreqTimeFFT(tseries, add_quadrature_phase(self.in_fseries, self.n), self.revplan)

		#
		# done
		#

		return tseries


def normalized_autocorrelation(fseries, revplan):
	data = fseries.data
	fseries = laltypes.COMPLEX16FrequencySeries(
		name = fseries.name,
		epoch = fseries.epoch,
		f0 = fseries.f0,
		deltaF = fseries.deltaF,
		sampleUnits = fseries.sampleUnits,
		data = data * numpy.conj(data)
	)
	tseries = laltypes.COMPLEX16TimeSeries(
		data = numpy.empty((len(data),), dtype = "cdouble")
	)
	lalfft.XLALCOMPLEX16FreqTimeFFT(tseries, fseries, revplan)
	data = tseries.data
	tseries.data = data / data[0]
	return tseries
	

# Round a number up to the nearest power of 2
def ceil_pow_2(x):
	x = int(math.ceil(x))
	x -= 1
	n = 1
	while n and (x & (x + 1)):
		x |= x >> n
		n *= 2
	return x + 1


def time_slices(
	sngl_inspiral_rows,
	flow = 40,
	fhigh = 900,
	padding = 1.5,
	samples_min = 1024,
	samples_max_256 = 1024,
	samples_max_64 = 2048,
	samples_max = 4096,
	verbose = False
):
	"""
	The function time_frequency_boundaries splits a template bank up by
	times for which different sampling rates are appropriate.

	The function returns a list of 3-tuples of the form (rate,begin,end)
	where rate is the sampling rate and begin/end mark the boundaries
	during which the given rate is guaranteed to be appropriate (no
	template exceeds a frequency of Nyquist/padding during these times).

	@param sngl_inspiral_rows The snigle inspiral rows for this sub bank
	@param flow The low frequency to generate the waveform from
	@param fhigh The maximum frequency for generating the waveform
	@param padding Allow a template to be generated up to Nyquist / padding for any slice
	@param samples_min The minimum number of samples to use in any time slice
	@param samples_max_256 The maximum number of samples to have in any time slice greater than or equal to 256 Hz
	@param samples_max_64 The maximum number of samples to have in any time slice greater than or equal to 64 Hz
	@param samples_max The maximum number of samples in any time slice below 64 Hz
	@param verbose Be verbose
	"""

	#
	# DETERMINE A SET OF ALLOWED SAMPLE RATES
	#

	# We only allow sample rates that are powers of two.  The upper limit
	# is set by the sampling rate for h(t), which is 16384Hz.  The lower
	# limit is set arbitrarily to be 32Hz.
	allowed_rates = [16384,8192,4096,2048,1024,512,256,128,64,32]

	# Remove too-small and too-big sample rates base on input params.
	sample_rate_min = ceil_pow_2( 2 * padding * flow )
	sample_rate_max = ceil_pow_2( 2 * fhigh )
	while allowed_rates[-1] < sample_rate_min:
		allowed_rates.pop(-1)
	while allowed_rates[0] > sample_rate_max:
		allowed_rates.pop(0)

	# make sure the second highest rate is less than ringdown frequency
	allowed_rates = [allowed_rates[0]] + [r for r in allowed_rates if r < min(spawaveform.imrffinal(row.mass1, row.mass2, spawaveform.computechi(row.mass1, row.mass2, row.spin1z, row.spin2z), 'ringdown') for row in sngl_inspiral_rows)]

	#
	# FIND TIMES WHEN THESE SAMPLE RATES ARE OK TO USE
	#

	# How many sample points should be included in a chunk?
	# We need to balance the need to have few chunks with the
	# need to have small chunks.
	# We choose the min size such that the template matrix
	# has its time dimension at least as large as its template dimension.
	# The max size is chosen based on experience, which shows that
	# SVDs of matrices bigger than m x 8192 are very slow.
	segment_samples_min = max(ceil_pow_2( 2*len(sngl_inspiral_rows) ),samples_min)

	# For each allowed sampling rate with associated Nyquist frequency fN,
	# determine the greatest amount of time any template in the bank spends
	# between fN/2 and fhigh.
	# fN/2 is given by sampling_rate/4 and is the next lowest Nyquist frequency
	# We pad this so that we only start a new lower frequency sampling rate
	# when all the waveforms are a fraction (padding-1) below the fN/2.
	time_freq_boundaries = []
	accum_time = 0

	# get the lower rate boundaries
	rates_below = [r for r in allowed_rates[1:]] + [flow]

	for rate_below, rate in zip(rates_below, allowed_rates):
		if rate >= 256:
			segment_samples_max = samples_max_256
		elif rate >= 64:
			segment_samples_max = samples_max_64
		else:
			segment_samples_max = samples_max
	
		if segment_samples_min > segment_samples_max:
			raise ValueError("The input template bank must have fewer than %d templates, but had %d." % (segment_samples_max, 2 * len(sngl_inspiral_rows)))

		# Factor of 2 goes from sample rate to Nyquist freq
		this_flow = float(rate_below)/(2*padding)
		longest_chirp = max(spawaveform.imrchirptime(row.mass1,row.mass2,this_flow,spawaveform.computechi(row.mass1, row.mass2, row.spin1z, row.spin2z)) for row in sngl_inspiral_rows)

		# Do any of the templates go beyond the accumulated time?
		# If so, we need to add some blocks at this sampling rate.
		# If not, we can skip this sampling rate and move on to the next lower one.
		if longest_chirp > accum_time:
			# How many small/large blocks are needed to cover this band?
			number_large_blocks = int(numpy.floor( rate*(longest_chirp-accum_time) / segment_samples_max ))
			number_small_blocks = int(numpy.ceil( rate*(longest_chirp-accum_time) / segment_samples_min )
						  - number_large_blocks*segment_samples_max/segment_samples_min)

			# Add small blocks first, since the templates change more rapidly
			# near the end of the template and therefore have more significant
			# components in the SVD.
			exponent = 0
			while number_small_blocks > 0:
				if number_small_blocks % 2 == 1:
					time_freq_boundaries.append((rate,
								     accum_time,
								     accum_time+float(2**exponent)*segment_samples_min/rate))
					accum_time += float(2**exponent)*segment_samples_min/rate
				exponent += 1
				number_small_blocks = number_small_blocks >> 1 # bit shift to right, dropping LSB


			# Now add the big blocks
			for idx in range(number_large_blocks):
				time_freq_boundaries.append((rate,
							     accum_time,
							     accum_time+(1./rate)*segment_samples_max))
				accum_time += (1./rate)*segment_samples_max

	if verbose:
		print>> sys.stderr, "Time freq boundaries: "
		print>> sys.stderr, time_freq_boundaries

	return numpy.array(time_freq_boundaries,dtype=[('rate','int'),('begin','float'),('end','float')])

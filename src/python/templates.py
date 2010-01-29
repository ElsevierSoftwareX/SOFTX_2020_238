# Copyright (C) 2009  LIGO Scientific Collaboration
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


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


import numpy


from pylal import datatypes as laltypes
from pylal import lalfft
from pylal import spawaveform
from glue.ligolw import lsctables
from glue.ligolw import utils


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


def add_quadrature_phase(fseries, n):
	"""
	From the Fourier transform of a real-valued function of time,
	compute and return the Fourier transform of the complex-valued
	function of time whose real component is the original time series
	and whose imaginary component is the quandrature phase of the real
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
	#negative_frequencies = numpy.conj(positive_frequencies[::-1])
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
	>>> input = REAL8TimeSeries(deltaT = 1.0 / 128, data = numpy.cos(numpy.arange(128, dtype = "double") * 2 * numpy.pi / 128)) # one cycle of cos(t)
	>>> output = q(input) # output has cos(t) in real part, sin(t) in imaginary part
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


def time_frequency_boundaries(
	template_bank_filename,
	segment_samples = 4096,
	flow = 64,
	sample_rate_max = 2048,
	padding = 0.9
):
	"""
	The function time_frequency_boundaries splits a template bank up by
	times for which different sampling rates are appropriate.  The
	function returns a list of 3-tuples of the form (rate,begin,end)
	where rate is the sampling rate and begin/end mark the boundaries
	during which the given rate is guaranteed to be appropriate (no
	template exceeds a frequency of padding*Nyquist during these times
	and no lower sampling rate would work).  For computational
	purposes, no time interval exceeds max_samples_per_segment.  The
	same rate may therefore apply to more than one segment.
	"""
	# Round a number up to the nearest power of 2
	def ceil_pow_2( number ):
		return 2**(numpy.ceil(numpy.log2( number )))

	# Load template bank mass params
	template_bank_table = (
		lsctables.table.get_table(
			utils.load_filename(
				template_bank_filename,
				gz=template_bank_filename.endswith("gz") ),
			"sngl_inspiral") )
        mass1 = template_bank_table.get_column('mass1')
        mass2 = template_bank_table.get_column('mass2')

	# We only allow sample rates that are powers of two.
	#
	# h(t) is sampled at 16384Hz, which sets the upper limit
	# and advligo will likely not reach below 10Hz, which 
	# sets the lower limit (32Hz = ceil_pow_2(2*10)Hz )
	#
	allowed_rates = [32,64,128,256,512,1024,2048,4096,8192,16384]

	#
	# Adjust the allowed_rates to fit with the template bank
	#
	ffinal_max = max(spawaveform.ffinal(m1,m2,'schwarz_isco') for m1,m2 in zip(mass1,mass2) )
	ffinal_max = min(padding*allowed_rates[-1]/2,ffinal_max)

	# Refine the list of allowed rates
	while allowed_rates[-1] > min(2*ffinal_max/padding,sample_rate_max):
		allowed_rates.pop(-1)

	sample_rate_min = ceil_pow_2( 2*(1./padding)* flow )
	while allowed_rates[0] < sample_rate_min:
		allowed_rates.pop(0)

	#
	# Split up templates by time
	#

	# FIXME: what happens if padding*sample_rate_min/2 == flow?
	# Best to look at high rates first
	allowed_rates.reverse()
	time_freq_boundaries = [(sample_rate_max,0,(1./sample_rate_max)*segment_samples)]
	accum_time = (1./sample_rate_max)*segment_samples
	for rate in allowed_rates:
		longest_chirp = max(spawaveform.chirptime(m1,m2,7,padding*rate/2,sample_rate_max/2) for m1,m2 in zip(mass1,mass2))
		print "longest chirp",longest_chirp
		print "accum time",accum_time
		if longest_chirp < accum_time:
			allowed_rates.remove(rate)
			continue
		while accum_time <= longest_chirp:
			segment_num = len(time_freq_boundaries)
			time_freq_boundaries.append((rate,accum_time,accum_time+(1./rate)*segment_samples))
			accum_time += (1./rate)*segment_samples

	print "Time freq boundaries: ",time_freq_boundaries

	return time_freq_boundaries

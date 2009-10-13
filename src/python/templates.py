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


class QuadraturePhase(object):
	"""
	A tool for generating the quadrature phase of a real-valued
	template.

	Example:

	>>> import numpy
	>>> from pylal.datatypes import REAL8TimeSeries
	>>> q = QuadraturePhase(128) # initialize for 128-sample templates
	>>> in = REAL8TimeSeries(deltaT = 1.0 / 128, data = numpy.sin(numpy.arange(128, dtype = "double") * 2 * math.pi / 128)) # one cycle of sin(t)
	>>> out = q.process(in) # sin(t) in real part, cos(t) in imaginary part
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
		self.out_fseries = lalfft.prepare_fseries_for_complex16tseries(laltypes.COMPLEX16TimeSeries(deltaT = 1.0, data = numpy.zeros((n,), dtype = "cdouble")))

	def process(self, tseries):
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
		# copy into expanded frequency series to generate an
		# imaginary component.  after setting the DC and Nyquist
		# components to zero, the components of the input frequency
		# series (which are the non-negative frequency components)
		# are doubled, conjugated, and copied into the negative
		# frequency components of the output frequency series.  the
		# positive frequency components of the output frequency
		# series are zeroed.
		#

		self.out_fseries.name = self.in_fseries.name
		self.out_fseries.epoch = self.in_fseries.epoch
		self.out_fseries.f0 = self.in_fseries.f0
		self.out_fseries.deltaF = self.in_fseries.deltaF
		self.out_fseries.sampleUnits = self.in_fseries.sampleUnits
		indata = self.in_fseries.data
		indata[0] = indata[-1] = 0
		self.out_fseries.data = numpy.concatenate((2 * numpy.conj(indata[::-1]), numpy.zeros((self.n - len(indata),), dtype = "cdouble")))

		#
		# transform to complex time series
		#

		tseries = laltypes.COMPLEX16TimeSeries(data = numpy.zeros((self.n,), dtype = "cdouble"))
		lalfft.XLALCOMPLEX16FreqTimeFFT(tseries, self.out_fseries, self.revplan)

		#
		# done
		#

		return tseries

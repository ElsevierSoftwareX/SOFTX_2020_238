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


import math
import cmath
import numpy
from scipy import interpolate
from scipy import linalg
import sys


from pylal import datatypes as laltypes
from pylal import lalfft
from pylal import spawaveform


import templates


__author__ = "Kipp Cannon <kipp.cannon@ligo.org>, Chad Hanna <chad.hanna@ligo.org>, Drew Keppel <drew.keppel@ligo.org>"
__version__ = "FIXME"
__date__ = "FIXME"


#
# =============================================================================
#
#                           Inspiral Template Stuff
#
# =============================================================================
#


def interpolate_psd(psd, deltaF):
	#
	# no-op?
	#

	if deltaF == psd.deltaF:
		return psd

	#
	# interpolate PSD by clipping/zero-padding time-domain impulse
	# response of equivalent whitening filter
	#

	#from scipy import fftpack
	#psd_data = psd.data
	#x = numpy.zeros((len(psd_data) * 2 - 2,), dtype = "double")
	#psd_data = numpy.where(psd_data, psd_data, float("inf"))
	#x[0] = 1 / psd_data[0]**.5
	#x[1::2] = 1 / psd_data[1:]**.5
	#x = fftpack.irfft(x)
	#if deltaF < psd.deltaF:
	#	x *= numpy.cos(numpy.arange(len(x)) * math.pi / (len(x) + 1))**2
	#	x = numpy.concatenate((x[:(len(x) / 2)], numpy.zeros((int(round(len(x) * psd.deltaF / deltaF)) - len(x),), dtype = "double"), x[(len(x) / 2):]))
	#else:
	#	x = numpy.concatenate((x[:(int(round(len(x) * psd.deltaF / deltaF)) / 2)], x[-(int(round(len(x) * psd.deltaF / deltaF)) / 2):]))
	#	x *= numpy.cos(numpy.arange(len(x)) * math.pi / (len(x) + 1))**2
	#x = 1 / fftpack.rfft(x)**2
	#psd_data = numpy.concatenate(([x[0]], x[1::2]))

	#
	# interpolate PSD with linear interpolator
	#

	#psd_data = psd.data
	#f = psd.f0 + numpy.arange(len(psd_data)) * psd.deltaF
	#interp = interpolate.interp1d(f, psd_data, bounds_error = False)
	#f = psd.f0 + numpy.arange(round(len(psd_data) * psd.deltaF / deltaF)) * deltaF
	#psd_data = interp(f)

	#
	# interpolate log(PSD) with cubic spline.  note that the PSD is
	# clipped at 1e-300 to prevent nan's in the interpolator (which
	# doesn't seem to like the occasional sample being -inf)
	#

	psd_data = psd.data
	psd_data = numpy.where(psd_data, psd_data, 1e-300)
	f = psd.f0 + numpy.arange(len(psd_data)) * psd.deltaF
	interp = interpolate.splrep(f, numpy.log(psd_data), s = 0)
	f = psd.f0 + numpy.arange(round(len(psd_data) * psd.deltaF / deltaF)) * deltaF
	psd_data = numpy.exp(interpolate.splev(f, interp, der = 0))

	#
	# return result
	#

	return laltypes.REAL8FrequencySeries(
		name = psd.name,
		epoch = psd.epoch,
		f0 = psd.f0,
		deltaF = deltaF,
		sampleUnits = psd.sampleUnits,
		data = psd_data
	)


def generate_template(template_bank_row, f_low, sample_rate, duration, order = 7, end_freq = "light_ring"):
	z = numpy.empty(int(round(sample_rate * duration)), "cdouble")

	spawaveform.waveform(template_bank_row.mass1, template_bank_row.mass2, order, 1.0 / duration, 1.0 / sample_rate, f_low, spawaveform.ffinal(template_bank_row.mass1, template_bank_row.mass2, end_freq), z)

	return laltypes.COMPLEX16FrequencySeries(
		name = "template",
		epoch = laltypes.LIGOTimeGPS(0),
		f0 = 0.0,
		deltaF = 1.0 / duration,
		sampleUnits = laltypes.LALUnit("strain"),
		data = z[:len(z) // 2 + 1]
	)


def generate_templates(template_table, psd, f_low, sample_rate, duration, autocorrelation_length = None, verbose = False):
	length = int(round(duration * sample_rate))

	working_length = int(round(2**math.ceil(math.log(length + round(32.0 * sample_rate), 2))))	# add 32 seconds for PSD ringing, round up to power of 2 count of samples
	working_duration = float(working_length) / sample_rate

	if psd is not None:
		psd = interpolate_psd(psd, 1.0 / working_duration)

	revplan = lalfft.XLALCreateReverseCOMPLEX16FFTPlan(working_length, 1)
	tseries = laltypes.COMPLEX16TimeSeries(
		data = numpy.zeros((working_length,), dtype = "cdouble")
	)

	template_bank = numpy.zeros((2 * len(template_table), length), dtype = "double")
	if autocorrelation_length is not None:
		if not (autocorrelation_length % 2):
			raise ValueError, "autocorrelation_length must be odd (got %d)" % autocorrelation_length
		autocorrelation_bank = numpy.zeros((len(template_table), autocorrelation_length), dtype = "cdouble")
	else:
		autocorrelation_bank = None

	for i, row in enumerate(template_table):
		if verbose:
			print >>sys.stderr, "generating template %d/%d:  m1 = %g, m2 = %g" % (i + 1, len(template_table), row.mass1, row.mass2)

		#
		# generate "cosine" component of frequency-domain template
		#

		fseries = generate_template(row, f_low, sample_rate, working_duration)

		#
		# whiten and add quadrature phase ("sine" component)
		#

		if psd is not None:
			lalfft.XLALWhitenCOMPLEX16FrequencySeries(fseries, psd)
		fseries = templates.add_quadrature_phase(fseries, working_length)

		#
		# compute time-domain autocorrelation function
		#

		if autocorrelation_bank is not None:
			autocorrelation = templates.normalized_autocorrelation(fseries, revplan).data
			autocorrelation_bank[i, ::-1] = numpy.concatenate((autocorrelation[-(autocorrelation_length // 2):], autocorrelation[:(autocorrelation_length // 2  + 1)]))

		#
		# transform template to time domain
		#

		lalfft.XLALCOMPLEX16FreqTimeFFT(tseries, fseries, revplan)

		#
		# extract the portion to be used for filtering
		#

		data = tseries.data[-length:]

		#
		# normalize so that inner product of template with itself
		# is 2
		#

		data *= cmath.sqrt(2 / numpy.dot(data, numpy.conj(data)))

		#
		# copy real and imaginary parts into adjacent (real-valued)
		# rows of template bank
		#

		template_bank[(2 * i + 0), :] = data.real
		template_bank[(2 * i + 1), :] = data.imag

	return template_bank, autocorrelation_bank


def decompose_templates(template_bank, tolerance, identity = False):
	#
	# sum-of-squares for each template (row).
	#

	chifacs = (template_bank * template_bank).sum(1)

	#
	# this turns this function into a no-op:  the output "basis
	# vectors" are exactly the input templates and the reconstruction
	# matrix is the identity matrix
	#

	if identity:
		return template_bank, numpy.ones(template_bank.shape[0], dtype = "double"), numpy.identity(template_bank.shape[0], dtype = "double"), chifacs

	#
	# adjust tolerance according to local norm
	#

	tolerance = 1 - (1 - tolerance) / chifacs.max()

	#
	# S.V.D.
	#

	U, s, Vh = linalg.svd(template_bank.T)

	#
	# determine component count
	#

	residual = numpy.sqrt((s * s).cumsum() / numpy.dot(s, s))
	n = min(residual.searchsorted(tolerance) + 1, len(s))

	#
	# clip decomposition, pre-multiply Vh by s
	#

	U = U[:,:n]
	Vh = numpy.dot(numpy.diag(s), Vh)[:n,:]
	s = s[:n]

	#
	# done.
	#

	return U.T, s, Vh, chifacs

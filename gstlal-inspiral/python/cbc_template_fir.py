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

## @file

## @package cbc_template_fir

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
import scipy
import sys


from pylal import datatypes as laltypes
from pylal import lalfft
from pylal import spawaveform
import lal
import lalsimulation as lalsim


from gstlal.reference_psd import interpolate_psd, psd_to_fir_kernel


from gstlal import templates


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


def tukeywindow(data):
	# Taper the edges unless it goes over 50%
	samps = 200.
	if len(data) >= 2 * samps:
		tp = samps / len(data)
	else:
		tp = 0.50
	return lal.CreateTukeyREAL8Window(len(data), tp).data.data



def generate_template(template_bank_row, approximant, sample_rate, duration, f_low, f_high, amporder = 0, order = 7, fwdplan = None, fworkspace = None):
	"""
	Generate a single frequency-domain template, which
	 (1) is band-limited between f_low and f_high,
	 (2) has an IFFT which is duration seconds long and
	 (3) has an IFFT which is sampled at sample_rate Hz
	"""
	if approximant in templates.gstlal_FD_approximants:

		# FIXME use hcross somday?
		# We don't here because it is not guaranteed to be orthogonal
		# and we add orthogonal phase later

		hplus,hcross = lalsim.SimInspiralChooseFDWaveform(
			0., # phase
			1.0 / duration, # sampling interval
			lal.LAL_MSUN_SI * template_bank_row.mass1,
			lal.LAL_MSUN_SI * template_bank_row.mass2,
			template_bank_row.spin1x,
			template_bank_row.spin1y,
			template_bank_row.spin1z,
			template_bank_row.spin2x,
			template_bank_row.spin2y,
			template_bank_row.spin2z,
			f_low,
			f_high,
			1.e6 * lal.LAL_PC_SI, # distance
			0., # inclination
			0., # tidal deformability lambda 1
			0., # tidal deformability lambda 2
			None, # waveform flags
			None, # Non GR params
			amporder,
			order,
			lalsim.GetApproximantFromString(str(approximant))
			)

		# NOTE assumes fhigh is the Nyquist frequency!!!
		assert len(hplus.data.data) == int(round(sample_rate * duration))//2 +1
		z = hplus.data.data

	elif approximant in templates.gstlal_TD_approximants:

		# FIXME use hcross somday?
		# We don't here because it is not guaranteed to be orthogonal
		# and we add orthogonal phase later

		hplus,hcross = lalsim.SimInspiralChooseTDWaveform(
			0., # phase
			1.0 / sample_rate, # sampling interval
			lal.LAL_MSUN_SI * template_bank_row.mass1,
			lal.LAL_MSUN_SI * template_bank_row.mass2,
			template_bank_row.spin1x,
			template_bank_row.spin1y,
			template_bank_row.spin1z,
			template_bank_row.spin2x,
			template_bank_row.spin2y,
			template_bank_row.spin2z,
			f_low,
			0, # reference frequency?
			1.e6 * lal.LAL_PC_SI, # distance
			0., # inclination
			0., # tidal deformability lambda 1
			0., # tidal deformability lambda 2
			None, # waveform flags
			None, # Non GR params
			amporder,
			order,
			lalsim.GetApproximantFromString(str(approximant))
			)

		hplus.data.data *= tukeywindow(hplus.data.data)
		data = numpy.zeros((sample_rate * duration,))
		data[-hplus.data.length:] = hplus.data.data

		tseries = laltypes.REAL8TimeSeries(
			name = "template",
			epoch = laltypes.LIGOTimeGPS(0),
			f0 = 0.0,
			deltaT = 1.0 / sample_rate,
			sampleUnits = laltypes.LALUnit("strain"),
			data = data
		)
		
		lalfft.XLALREAL8TimeFreqFFT(fworkspace, tseries, fwdplan)
		z = numpy.copy(fworkspace.data)

	else:
		raise ValueError("Unsupported approximant given %s" % approximant)

	return laltypes.COMPLEX16FrequencySeries(
		name = "template",
		epoch = laltypes.LIGOTimeGPS(0),
		f0 = 0.0,
		deltaF = 1.0 / duration,
		sampleUnits = laltypes.LALUnit("strain"),
		data = z
	)

def condition_imr_template(approximant, data, sample_rate_max, max_mass):

	max_index = numpy.argmax(numpy.abs(data))
	phase = numpy.arctan2(data[max_index].real, data[max_index].imag)
	data *= numpy.exp(1.j * phase)
	target_index = len(data) - int(sample_rate_max * max_mass * 100 * 5e-6) # 100 M for the max mass to leave room for ringdown
	data = numpy.roll(data, target_index - max_index)
	target_tukey_percentage = 2 * (1. - float(target_index) / len(data))
	data *= lal.CreateTukeyREAL8Window(len(data), target_tukey_percentage).data.data
	return data


def compute_autocorrelation_mask( autocorrelation ):
	'''
	Given an autocorrelation time series, estimate the optimal
	autocorrelation length to use and return a matrix which masks
	out the unwanted elements. FIXME TODO for now just returns
	ones
	'''
	return numpy.ones( autocorrelation.shape, dtype="int" )


def movingmedian(interval, window_size):
	tmp = numpy.copy(interval)
	for i in range(window_size, len(interval)-window_size):
		tmp[i] = numpy.median(interval[i-window_size:i+window_size])
	return tmp


def movingaverage(interval, window_size):
	window = lal.CreateTukeyREAL8Window(window_size, 0.5).data.data
	return numpy.convolve(interval, window, 'same')


def condition_psd(psd, newdeltaF, minf = 40.0, avgwindow = 64):
	assert newdeltaF < psd.deltaF
	psddata = psd.data
	psddata[int(0.85*len(psddata)):] = max(psddata)
	psd.data = movingmedian(psddata, avgwindow)
	psd.data = movingaverage(psd.data, avgwindow)
	half_impulse = len(psd.data) - 1
	kernel, latency, sample_rate = psd_to_fir_kernel(psd)
	# FIXME is this a no-op? Is the latency returned correct?
	d = kernel[latency-half_impulse:latency+half_impulse+1]
	# FIXME check even/odd
	length = int(round(psd.deltaF / newdeltaF * 2 * half_impulse))
	# FIXME maybe don't do the uwrapping??
	out = numpy.zeros((length,))
	out[-half_impulse-1:] = d[:half_impulse+1]
	out[:half_impulse] = d[half_impulse+1:]
	newdata = scipy.fft(out)
	newdata = abs(newdata)**2
	newdata = newdata[:length//2+1] / len(newdata)**.5 * (newdeltaF / psd.deltaF)**.5
	newdata[int(0.85*len(newdata)):] = max(newdata)
	newdata[:int(minf / newdeltaF)] = max(newdata)

	return laltypes.REAL8FrequencySeries(
		name = psd.name,
		epoch = psd.epoch,
		f0 = psd.f0,
		deltaF = newdeltaF,
		sampleUnits = psd.sampleUnits,
		data = newdata
		)


def generate_templates(template_table, approximant, psd, f_low, time_slices, autocorrelation_length = None, verbose = False):
	"""!
	Generate a bank of templates, which are
	 (1) broken up into time slice,
	 (2) down-sampled in each time slice and
	 (3) whitened with the given psd.
	"""
	sample_rate_max = max(time_slices['rate'])
	duration = max(time_slices['end'])
	length_max = int(round(duration * sample_rate_max))

	# Add 32 seconds to template length for PSD ringing, round up to power of 2 count of samples
	working_length = templates.ceil_pow_2(length_max + round(32.0 * sample_rate_max))
	working_duration = float(working_length) / sample_rate_max

	# Give the PSD the same frequency spacing as the waveforms we are about to generate
	if psd is not None:
		psd_initial_deltaF = psd.deltaF # store for normalization later
		psd = condition_psd(psd, 1.0 / working_duration, minf = f_low)

	revplan = lalfft.XLALCreateReverseCOMPLEX16FFTPlan(working_length, 1)
	fwdplan = lalfft.XLALCreateForwardREAL8FFTPlan(working_length, 1)
	tseries = laltypes.COMPLEX16TimeSeries(
		data = numpy.zeros((working_length,), dtype = "cdouble")
	)
	fworkspace = laltypes.COMPLEX16FrequencySeries(
		name = "template",
		epoch = laltypes.LIGOTimeGPS(0),
		f0 = 0.0,
		deltaF = 1.0 / working_duration,
		data = numpy.zeros((working_length//2 + 1,), dtype = "cdouble")
	)

	# Check parity of autocorrelation length
	if autocorrelation_length is not None:
		if not (autocorrelation_length % 2):
			raise ValueError, "autocorrelation_length must be odd (got %d)" % autocorrelation_length
		autocorrelation_bank = numpy.zeros((len(template_table), autocorrelation_length), dtype = "cdouble")
		autocorrelation_mask = compute_autocorrelation_mask( autocorrelation_bank )
	else:
		autocorrelation_bank = None
		autocorrelation_mask = None

	# Multiply by 2 * length of the number of sngl_inspiral rows to get the sine/cosine phases.
	template_bank = [numpy.zeros((2 * len(template_table), int(round(rate*(end-begin)))), dtype = "double") for rate,begin,end in time_slices]

	# Store the original normalization of the waveform.  After whitening, the waveforms
	# are normalized.  Use the sigmasq factors to get back the original waveform.
	sigmasq = []

	# Generate each template, downsampling as we go to save memory
	max_mass = max([row.mass1+row.mass2 for row in template_table])
	for i, row in enumerate(template_table):
		if verbose:
			print >>sys.stderr, "generating template %d/%d:  m1 = %g, m2 = %g, s1x = %g, s1y = %g, s1z = %g, s2x = %g, s2y = %g, s2z = %g" % (i + 1, len(template_table), row.mass1, row.mass2, row.spin1x, row.spin1y, row.spin1z, row.spin2x, row.spin2y, row.spin2z)

		#
		# generate "cosine" component of frequency-domain template
		#

		fseries = generate_template(row, approximant, sample_rate_max, working_duration, f_low, sample_rate_max / 2., fwdplan = fwdplan, fworkspace = fworkspace)

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

		data = tseries.data[-length_max:]

		#
		# condition the template if necessary (e.g. line up IMR waveforms by peak amplitude)
		#
		if approximant in ("IMRPhenomB", "EOBNRv2"):
			data = condition_imr_template(approximant, data, sample_rate_max, max_mass)
		else:
			data *= tukeywindow(data)
		#
		# normalize so that inner product of template with itself
		# is 2
		#

		norm = abs(numpy.dot(data, numpy.conj(data)))
		data *= cmath.sqrt(2 / norm)

		#
		# definition of sigmasq is: sigmasq = \int h(f) h^*(f) df the
		# norm we have computed so far is missing df, with a factor of
		# 2 from the psd definition NOTE!!! We have to use the original
		# psd spacing, the interpolation does *not* preserve the
		# integral properly
		#

		sigmasq.append(2*norm)

		#
		# copy real and imaginary parts into adjacent (real-valued)
		# rows of template bank
		#

		for frag_num,slice in enumerate(time_slices):
			# start and end times are measured *backwards* from
			# template end;  subtract from n to convert to
			# start and end index;  end:start is the slice to
			# extract (argh!  Chad!)
			begin_index = length_max - int(round(slice['begin'] * sample_rate_max))
			end_index = length_max - int(round(slice['end'] * sample_rate_max))
			stride = int(round(sample_rate_max / slice['rate']))

			# extract every stride-th sample.  we multiply by
			# \sqrt{stride} to maintain inner product
			# normalization so that the templates still appear
			# to be unit vectors at the reduced sample rate.
			# note that the svd returns unit basis vectors
			# regardless so this factor has no effect on the
			# normalization of the basis vectors used for
			# filtering but it ensures that the chifacs values
			# have the correct relative normalization.
			template_bank[frag_num][(2*i+0),:] = data.real[end_index:begin_index:stride] * math.sqrt(stride)
			template_bank[frag_num][(2*i+1),:] = data.imag[end_index:begin_index:stride] * math.sqrt(stride)

	return template_bank, autocorrelation_bank, autocorrelation_mask, sigmasq


def decompose_templates(template_bank, tolerance, identity = False):
	#
	# sum-of-squares for each template (row).
	#

	chifacs = (template_bank * template_bank).sum(1)

	#
	# this turns this function into a no-op:  the output "basis
	# vectors" are exactly the input templates and the reconstruction
	# matrix is absent (triggers pipeline construction code to omit
	# matrixmixer element)
	#

	if identity:
		return template_bank, None, None, chifacs

	#
	# adjust tolerance according to local norm
	#

	tolerance = 1 - (1 - tolerance) / chifacs.max()

	#
	# S.V.D.
	#

	U, s, Vh = spawaveform.svd(template_bank.T,mod=True,inplace=True)

	#
	# determine component count
	#

	residual = numpy.sqrt((s * s).cumsum() / numpy.dot(s, s))
	# FIXME in an ad hoc way force at least 6 principle components
	n = max(min(residual.searchsorted(tolerance) + 1, len(s)), 6)

	#
	# clip decomposition, pre-multiply Vh by s
	#

	U = U[:,:n]
	Vh = numpy.dot(numpy.diag(s), Vh)[:n,:]
	s = s[:n]

	#
	# renormalize the truncated SVD approximation of these template
	# waveform slices making sure their squares still add up to chifacs.
	# This is done by renormalizing the sum of the square of the
	# singular value weighted reconstruction coefficients associated with
	# each template.
	#

	V2 = (Vh * Vh).sum(0)
	for idx,v2 in enumerate(V2):
		Vh[:, idx] *= numpy.sqrt(chifacs[idx] / v2)

	#
	# done.
	#

	return U.T, s, Vh, chifacs

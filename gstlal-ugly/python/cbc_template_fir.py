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
import sys


from pylal import datatypes as laltypes
from pylal import lalfft
from pylal import spawaveform


from gstlal.reference_psd import interpolate_psd


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


def generate_template(template_bank_row, approximant, sample_rate, duration, f_low, f_high, order = 7):
	"""
	Generate a single frequency-domain template, which
	 (1) is band-limited between f_low and f_high,
	 (2) has an IFFT which is duration seconds long and
	 (3) has an IFFT which is sampled at sample_rate Hz
	"""
	z = numpy.empty(int(round(sample_rate * duration)), "cdouble")
	if approximant=="FindChirpSP" or approximant=="TaylorF2":
		spawaveform.waveform(template_bank_row.mass1, template_bank_row.mass2, order, 1.0 / duration, 1.0 / sample_rate, f_low, f_high, z, template_bank_row.chi)
	elif approximant=="IMRPhenomB":
		#FIXME a better plan than multiplying flow by 0.5 should be done...
		spawaveform.imrwaveform(template_bank_row.mass1, template_bank_row.mass2, 1.0/duration, 0.5 * f_low, z, template_bank_row.chi)
	else:
		raise ValueError, "Unsupported approximant given"

	return laltypes.COMPLEX16FrequencySeries(
		name = "template",
		epoch = laltypes.LIGOTimeGPS(0),
		f0 = 0.0,
		deltaF = 1.0 / duration,
		sampleUnits = laltypes.LALUnit("strain"),
		data = z[:len(z) // 2 + 1]
	)


def generate_templates(template_table, approximant, psd, f_low, time_slices, autocorrelation_length = None, verbose = False):
	"""
	Generate a bank of templates, which are
	 (1) broken up into time slice,
	 (2) optimally down-sampled in each time slice and
	 (3) whitened with the given psd.
	"""
	sample_rate_max = max(time_slices['rate'])
	duration = max(time_slices['end'])
	length_max = int(round(duration * sample_rate_max))
	length = int(round(sum(rate*(end-begin) for rate,begin,end in time_slices)))

	# Add 32 seconds to template length for PSD ringing, round up to power of 2 count of samples
	working_length = int(round(2**math.ceil(math.log(length_max + round(32.0 * sample_rate_max), 2))))
	working_duration = float(working_length) / sample_rate_max

	# Give the PSD the same frequency spacing as the waveforms we are about to generate
	psd_initial_deltaF = psd.deltaF # store for normalization later
	if psd is not None:
		psd = interpolate_psd(psd, 1.0 / working_duration)

	# Generate a plan for IFFTing the waveform and make space for the time-domain waveform
	revplan = lalfft.XLALCreateReverseCOMPLEX16FFTPlan(working_length, 1)
	tseries = laltypes.COMPLEX16TimeSeries(
		data = numpy.zeros((working_length,), dtype = "cdouble")
	)

	# Check parity of autocorrelation length
	if autocorrelation_length is not None:
		if not (autocorrelation_length % 2):
			raise ValueError, "autocorrelation_length must be odd (got %d)" % autocorrelation_length
		autocorrelation_bank = numpy.zeros((len(template_table), autocorrelation_length), dtype = "cdouble")
	else:
		autocorrelation_bank = None

	# Have one template bank for each time_slice
	template_bank = [numpy.zeros((2 * len(template_table), int(round(rate*(end-begin)))), dtype = "double") for rate,begin,end in time_slices]

	# Store the original normalization of the waveform.  After whitening, the waveforms
	# are normalized.  Use the sigmasq factors to get back the original waveform.
	sigmasq = []

	# Generate each template, downsampling as we go to save memory
	for i, row in enumerate(template_table):
		if verbose:
			print >>sys.stderr, "generating template %d/%d:  m1 = %g, m2 = %g, chi = %g" % (i + 1, len(template_table), row.mass1, row.mass2, row.chi)

		#
		# generate "cosine" component of frequency-domain template
		#

		fseries = generate_template(row, approximant, sample_rate_max, working_duration, f_low, sample_rate_max / (2*1.05)) # pad the nyquist rate by 5%

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

		sigmasq.append(norm * psd_initial_deltaF / 2.)

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

	return template_bank, autocorrelation_bank, sigmasq


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

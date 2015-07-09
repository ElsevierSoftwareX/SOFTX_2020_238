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
# The python module to implement SVD decomposed FIR filtering
#
# ### Review Status
#
# STATUS: reviewed with actions
#
# | Names                                               | Hash                                     | Date       | Diff to Head of Master      |
# | --------------------------------------------------- | ---------------------------------------- | ---------- | --------------------------- |
# | Florent, Sathya, Duncan Me, Jolien, Kipp, Chad      | 7536db9d496be9a014559f4e273e1e856047bf71 | 2014-04-30 | --------------------------- |
# | Florent, Surabhi, Tjonnie, Kent, Jolien, Kipp, Chad | d84a8446a056ce92625b042148c2d9ef9cd8bb0d | 2015-05-12 | <a href="@gstlal_inspiral_cgit_diff/python/cbc_template_fir.py?id=HEAD&id2=d84a8446a056ce92625b042148c2d9ef9cd8bb0d">cbc_template_fir.py</a> |
#
# #### Action items
#
# - Consider changing the order of interpolation and smoothing the PSD
# - Remove Jolien's function and get the new flow from lalsimulation to use XLALSimInspiralChirpStartFrequencyBound() and friends
# - move sigma squared calculation somewhere and get them updated dynamically
# - possibly use ROM stuff, possibly use low-order polynomial approx computed on the fly from the template as it's generated
# - remove lefttukeywindow()
# - use template_bank_row.coa_phase == 0. in SimInspiralFD() call, make sure itac adjusts the phase it assigns to triggers from the template coa_phase
# - change "assumes fhigh" to "asserts fhigh"
# - move assert epoch_time into condition_imr_waveform(), should be assert -len(data) <= epoch_time * sample_rate < 0
#
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


from gstlal.reference_psd import interpolate_psd, horizon_distance


from gstlal import templates
from gstlal import chirptime


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


def tukeywindow(data, samps = 200.):
	assert (len(data) >= 2 * samps) # make sure that the user is requesting something sane
	tp = float(samps) / len(data)
	return lal.CreateTukeyREAL8Window(len(data), tp).data.data


def generate_template(template_bank_row, approximant, sample_rate, duration, f_low, f_high, amporder = 0, order = 7, fwdplan = None, fworkspace = None):
	"""
	Generate a single frequency-domain template, which
	 (1) is band-limited between f_low and f_high,
	 (2) has an IFFT which is duration seconds long and
	 (3) has an IFFT which is sampled at sample_rate Hz
	"""
	if approximant in templates.gstlal_approximants:

		# FIXME use hcross somday?
		# We don't here because it is not guaranteed to be orthogonal
		# and we add orthogonal phase later

		hplus,hcross = lalsim.SimInspiralFD(
			0., # phase
			1.0 / duration, # sampling interval
			lal.MSUN_SI * template_bank_row.mass1,
			lal.MSUN_SI * template_bank_row.mass2,
			template_bank_row.spin1x,
			template_bank_row.spin1y,
			template_bank_row.spin1z,
			template_bank_row.spin2x,
			template_bank_row.spin2y,
			template_bank_row.spin2z,
			f_low,
			f_high,
			0., #FIXME chosen until suitable default value for f_ref is defined
			1.e6 * lal.PC_SI, # distance
			0., # redshift
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
		z = hplus.data.data
		assert len(z) == int(round(sample_rate * duration))//2 +1

	else:
		raise ValueError("Unsupported approximant given %s" % approximant)

	return laltypes.COMPLEX16FrequencySeries(
		name = "template",
		epoch = laltypes.LIGOTimeGPS(hplus.epoch.gpsSeconds, hplus.epoch.gpsNanoSeconds),
		f0 = 0.0,
		deltaF = 1.0 / duration,
		sampleUnits = laltypes.LALUnit("strain"),
		data = z
	)

def condition_imr_template(approximant, data, epoch_time, sample_rate_max, max_ringtime):
	assert -len(data) / sample_rate_max <= epoch_time < 0.0, "Epoch returned follows a different convention"
	# find the index for the peak sample using the epoch returned by
	# the waveform generator
	epoch_index = -int(epoch_time*sample_rate_max) - 1
	# align the peaks according to an overestimate of max rinddown
	# time for a given split bank
	target_index = len(data)-1 - int(sample_rate_max * max_ringtime)
	# rotate phase so that sample with peak amplitude is real
	phase = numpy.arctan2(data[epoch_index].imag, data[epoch_index].real)
	data *= numpy.exp(-1.j * phase)
	data = numpy.roll(data, target_index-epoch_index)
	# re-taper the ends of the waveform that got cyclically permuted
	# around the ring
	tukey_beta = 2. * abs(target_index - epoch_index) / float(len(data))
	assert 0. <= tukey_beta <= 1., "waveform got rolled WAY too much"
	data *= lal.CreateTukeyREAL8Window(len(data), tukey_beta).data.data
	# done
	return data, target_index


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


def condition_psd(psd, newdeltaF, minfs = (35.0, 40.0), maxfs = (1800., 2048.), smoothing_frequency = 4.):
	"""
	A function to condition the psd that will be used to whiten waveforms

	@param psd A real8 frequency series containing the psd
	@param newdeltaF (Hz) The target delta F to interpolate to
	@param minfs (Hz) The frequency boundaries over which to taper the spectrum to infinity.  i.e., frequencies below the first item in the tuple will have an infinite spectrum, the second item in the tuple will not be changed.  A taper from 0 to infinity is applied in between.  The PSD is also tapered from 0.85 * Nyquist to Nyquist.
	@param smoothing_frequency (Hz) The target frequency resolution after smoothing.  Lines with bandwidths << smoothing_frequency are removed via a median calculation.  Remaining features will be blurred out to this resolution.

	returns a conditioned psd
	"""

	#
	# store the psd horizon before conditioning
	#

	horizon_before = horizon_distance(psd, 1.4, 1.4, 8.0, minfs[1], maxfs[0])
	
	#
	# interpolate to new \Delta f
	#

	psd = interpolate_psd(psd, newdeltaF)

	#
	# Smooth the psd
	#

	psddata = psd.data
	avgwindow = int(smoothing_frequency / newdeltaF)
	psddata = movingmedian(psddata, avgwindow)
	psddata = movingaverage(psddata, avgwindow)

	#
	# Taper to infinity to turn this psd into an effective band pass filter
	#

	kmin = int(minfs[0] / newdeltaF)
	kmax = int(minfs[1] / newdeltaF)
	psddata[:kmin] = float('Inf')
	psddata[kmin:kmax] /= numpy.sin(numpy.arange(kmax-kmin) / (kmax-kmin-1.) * numpy.pi / 2.0)**4
	
	kmin = int(maxfs[0] / newdeltaF)
	kmax = int(maxfs[1] / newdeltaF)
	psddata[kmax:] = float('Inf')
	psddata[kmin:kmax] /= numpy.cos(numpy.arange(kmax-kmin) / (kmax-kmin-1.) * numpy.pi / 2.0)**4

	psd.data = psddata
	
	#
	# compute the psd horizon after conditioning and renormalize
	#

	horizon_after = horizon_distance(psd, 1.4, 1.4, 8.0, minfs[1], maxfs[0])

	psddata = psd.data
	psd.data =  psddata * (horizon_after / horizon_before)**2

	#
	# done
	#

	return psd


def joliens_function(f, template_table):
	"""
	A function to compute the padding needed to gaurantee well behaved waveforms

	@param f The target low frequency starting point
	@param template_table The sngl_inspiral table containing all the template parameters for this bank

	Returns the extra padding in time (tx) and the new low frequency cut
	off that should be used (f_new) as a tuple: (tx, f_new)
	"""
	def _chirp_duration(f, m1, m2):
		"""
		@returns the Newtonian chirp duration in seconds
		@param f the starting frequency in Hertz
		@param m1 mass of one component in solar masses
		@param m2 mass of the other component in solar masses
		"""
		G = 6.67e-11
		c = 3e8
		m1 *= 2e30 # convert to kg
		m2 *= 2e30 # convert to kg
		m1 *= G / c**3 # convert to s
		m2 *= G / c**3 # convert to s
		M = m1 + m2
		nu = m1 * m2 / M**2
		v = (numpy.pi * M * f)**(1.0/3.0)
		return 5.0 * M / (256.0 * nu * v**8)

	def _chirp_start_frequency(t, m1, m2):
		"""
		@returns the Newtonian chirp start frequency in Hertz
		@param t the time before coalescence in seconds
		@param m1 mass of one component in solar masses
		@param m2 mass of the other component in solar masses
		"""
		G = 6.67e-11
		c = 3e8
		m1 *= 2e30 # convert to kg
		m2 *= 2e30 # convert to kg
		m1 *= G / c**3 # convert to s
		m2 *= G / c**3 # convert to s
		M = m1 + m2
		nu = m1 * m2 / M**2
		theta = (nu * t / (5.0 * M))**(-1.0/8.0)
		return theta**3 / (8.0 * numpy.pi * M)

	minmc = min(template_table.getColumnByName('mchirp'))
	row = [t for t in template_table if t.mchirp == minmc][0]
	m1, m2 = row.mass1, row.mass2
	tc = _chirp_duration(f, m1, m2)
	tx = .1 * tc + 1.0
	f_new = _chirp_start_frequency(tx + tc, m1, m2)
	return tx, f_new
	

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

	# working f_low to actually use for generating the waveform
	working_f_low_extra_time, working_f_low = joliens_function(f_low, template_table)

	# Add duration of PSD to template length for PSD ringing, round up to power of 2 count of samples
	working_length = templates.ceil_pow_2(length_max + round(1./psd.deltaF * sample_rate_max))
	working_duration = float(working_length) / sample_rate_max

	# Smooth the PSD and interpolate to required resolution
	if psd is not None:
		psd = condition_psd(psd, 1.0 / working_duration, minfs = (working_f_low, f_low), maxfs = (sample_rate_max / 2.0 * 0.90, sample_rate_max / 2.0))

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

	# Store the original normalization of the waveform.  After
	# whitening, the waveforms are normalized.  Use the sigmasq factors
	# to get back the original waveform.
	sigmasq = []

	# Generate each template, downsampling as we go to save memory
	max_ringtime = max([chirptime.ringtime(row.mass1*lal.MSUN_SI + row.mass2*lal.MSUN_SI, chirptime.overestimate_j_from_chi(max(row.spin1z, row.spin2z))) for row in template_table])
	for i, row in enumerate(template_table):
		if verbose:
			print >>sys.stderr, "generating template %d/%d:  m1 = %g, m2 = %g, s1x = %g, s1y = %g, s1z = %g, s2x = %g, s2y = %g, s2z = %g" % (i + 1, len(template_table), row.mass1, row.mass2, row.spin1x, row.spin1y, row.spin1z, row.spin2x, row.spin2y, row.spin2z)

		#
		# generate "cosine" component of frequency-domain template.
		# waveform is generated for a canonical distance of 1 Mpc.
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

		data = tseries.data
		epoch_time = fseries.epoch.seconds + fseries.epoch.nanoseconds*1.e-9
		#
		# extract the portion to be used for filtering
		#


		#
		# condition the template if necessary (e.g. line up IMR
		# waveforms by peak amplitude)
		#

		if approximant in templates.gstlal_IMR_approximants:
			data, target_index = condition_imr_template(approximant, data, epoch_time, sample_rate_max, max_ringtime)
			# record the new end times for the waveforms (since we performed the shifts)
			row.set_end(laltypes.LIGOTimeGPS(float(target_index-(len(data) - 1.))/sample_rate_max))
		else:
			data *= tukeywindow(data, samps = 32)

		data = data[-length_max:]
		#
		# normalize so that inner product of template with itself
		# is 2
		#

		norm = abs(numpy.dot(data, numpy.conj(data)))
		data *= cmath.sqrt(2 / norm)

		#
		# sigmasq = 2 \sum_{k=0}^{N-1} |\tilde{h}_{k}|^2 / S_{k} \Delta f
		#
		# XLALWhitenCOMPLEX16FrequencySeries() computed
		#
		# \tilde{h}'_{k} = \sqrt{2 \Delta f} \tilde{h}_{k} / \sqrt{S_{k}}
		#
		# and XLALCOMPLEX16FreqTimeFFT() computed
		#
		# h'_{j} = \Delta f \sum_{k=0}^{N-1} exp(2\pi i j k / N) \tilde{h}'_{k}
		#
		# therefore, "norm" is
		#
		# \sum_{j} |h'_{j}|^{2} = (\Delta f)^{2} \sum_{j} \sum_{k=0}^{N-1} \sum_{k'=0}^{N-1} exp(2\pi i j (k-k') / N) \tilde{h}'_{k} \tilde{h}'^{*}_{k'}
		#                       = (\Delta f)^{2} \sum_{k=0}^{N-1} \sum_{k'=0}^{N-1} \tilde{h}'_{k} \tilde{h}'^{*}_{k'} \sum_{j} exp(2\pi i j (k-k') / N)
		#                       = (\Delta f)^{2} N \sum_{k=0}^{N-1} |\tilde{h}'_{k}|^{2}
		#                       = (\Delta f)^{2} N 2 \Delta f \sum_{k=0}^{N-1} |\tilde{h}_{k}|^{2} / S_{k}
		#                       = (\Delta f)^{2} N sigmasq
		#
		# and \Delta f = 1 / (N \Delta t), so "norm" is
		#
		# \sum_{j} |h'_{j}|^{2} = 1 / (N \Delta t^2) sigmasq
		#
		# therefore
		#
		# sigmasq = norm * N * (\Delta t)^2
		#

		sigmasq.append(norm * len(data) / sample_rate_max**2.)

		#
		# copy real and imaginary parts into adjacent (real-valued)
		# rows of template bank
		#

		for j, time_slice in enumerate(time_slices):
			# start and end times are measured *backwards* from
			# template end;  subtract from n to convert to
			# start and end index;  end:start is the slice to
			# extract, but there's also an amount added equal
			# to 1 less than the stride that I can't explain
			# but probaby has something to do with the reversal
			# of the open/closed boundary conditions through
			# all of this (argh!  Chad!)
			stride = int(round(sample_rate_max / time_slice['rate']))
			begin_index = length_max - int(round(time_slice['begin'] * sample_rate_max)) + stride - 1
			end_index = length_max - int(round(time_slice['end'] * sample_rate_max)) + stride - 1
			# make sure the rates are commensurate
			assert stride * time_slice['rate'] == sample_rate_max

			# extract every stride-th sample.  we multiply by
			# \sqrt{stride} to maintain inner product
			# normalization so that the templates still appear
			# to be unit vectors at the reduced sample rate.
			# note that the svd returns unit basis vectors
			# regardless so this factor has no effect on the
			# normalization of the basis vectors used for
			# filtering but it ensures that the chifacs values
			# have the correct relative normalization.
			template_bank[j][(2*i+0),:] = data.real[end_index:begin_index:stride] * math.sqrt(stride)
			template_bank[j][(2*i+1),:] = data.imag[end_index:begin_index:stride] * math.sqrt(stride)

	return template_bank, autocorrelation_bank, autocorrelation_mask, sigmasq, psd


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

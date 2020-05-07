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


__doc__= """
The python module to implement SVD decomposed FIR filtering

**Review Status**

STATUS: reviewed with actions

+-----------------------------------------------------+------------------------------------------+------------+
| Names                                               | Hash                                     | Date       |
+=====================================================+==========================================+============+
| Florent, Sathya, Duncan Me, Jolien, Kipp, Chad      | 7536db9d496be9a014559f4e273e1e856047bf71 | 2014-04-30 |
+-----------------------------------------------------+------------------------------------------+------------+
| Florent, Surabhi, Tjonnie, Kent, Jolien, Kipp, Chad | d84a8446a056ce92625b042148c2d9ef9cd8bb0d | 2015-05-12 |
+-----------------------------------------------------+------------------------------------------+------------+

**Action items**

- Consider changing the order of interpolation and smoothing the PSD
- move sigma squared calculation somewhere and get them updated dynamically
- possibly use ROM stuff, possibly use low-order polynomial approx computed on the fly from the template as it's generated
- remove lefttukeywindow()
- use template_bank_row.coa_phase == 0. in SimInspiralFD() call, make sure itac adjusts the phase it assigns to triggers from the template coa_phase
- change "assumes fhigh" to "asserts fhigh"
- move assert epoch_time into condition_imr_waveform(), should be assert -len(data) <= epoch_time * sample_rate < 0
"""


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


import bisect
import cmath
import math
import numpy
import sys
import os


import lal
from lal import LIGOTimeGPS
import lalsimulation as lalsim


from gstlal import spawaveform


from gstlal import chirptime
from gstlal import reference_psd
from gstlal import templates

try: ### optional dependency for movingmedian
	import pandas
except ImportError:
	pandas = None


__author__ = "Kipp Cannon <kipp.cannon@ligo.org>, Chad Hanna <chad.hanna@ligo.org>, Drew Keppel <drew.keppel@ligo.org>"
__version__ = "FIXME"
__date__ = "FIXME"

# a macro to switch between a conventional whitener and a fir whitener below
try:
	if int(os.environ["GSTLAL_FIR_WHITEN"]):
		FIR_WHITENER = True
	else:
		FIR_WHITENER = False
except KeyError:
	print("You must set the environment variable GSTLAL_FIR_WHITEN to either 0 or 1.  1 enables causal whitening. 0 is the traditional acausal whitening filter", file=sys.stderr)
	raise

#
# =============================================================================
#
#                           Inspiral Template Stuff
#
# =============================================================================
#

def create_FIR_whitener_kernel(length, duration, sample_rate, psd):
	assert psd
	#
	# Add another COMPLEX16TimeSeries and COMPLEX16FrequencySeries for kernel's FFT (Leo)
	#

	# Add another FFT plan for kernel FFT (Leo)
	fwdplan_kernel = lal.CreateForwardCOMPLEX16FFTPlan(length, 1)
	kernel_tseries = lal.CreateCOMPLEX16TimeSeries(
		name = "timeseries of whitening kernel",
		epoch = LIGOTimeGPS(0.),
		f0 = 0.,
		deltaT = 1.0 / sample_rate,
		length = length,
		sampleUnits = lal.Unit("strain")
	)
	kernel_fseries = lal.CreateCOMPLEX16FrequencySeries(
		name = "freqseries of whitening kernel",
		epoch = LIGOTimeGPS(0),
		f0 = 0.0,
		deltaF = 1.0 / duration,
		length = length,
		sampleUnits = lal.Unit("strain s")
	)

	#
	# Obtain a kernel of zero-latency whitening filter and
	# adjust its length (Leo)
	#

	psd_fir_kernel = reference_psd.PSDFirKernel()
	(kernel, latency, fir_rate) = psd_fir_kernel.psd_to_linear_phase_whitening_fir_kernel(psd, nyquist = sample_rate / 2.0)
	(kernel, theta) = psd_fir_kernel.linear_phase_fir_kernel_to_minimum_phase_whitening_fir_kernel(kernel, fir_rate)
	kernel = kernel[-1::-1]
	# FIXME this is off by one sample, but shouldn't be. Look at the miminum phase function
	# assert len(kernel) == length
	if len(kernel) < length:
		kernel = numpy.append(kernel, numpy.zeros(length - len(kernel)))
	else:
		kernel = kernel[:length]

	kernel_tseries.data.data = kernel

	#
	# FFT of the kernel
	#

	lal.COMPLEX16TimeFreqFFT(kernel_fseries, kernel_tseries, fwdplan_kernel) #FIXME

	return kernel_fseries

def taperzero_fseries(fseries, minfs = (35.0, 40.0), maxfs = (1800., 2048.)):
	"""
	A function to taper the psd to infinity for given min/max frequencies

	:psd: A real8 frequency series containing the psd
	:newdeltaF (Hz): The target delta F to interpolate to
	:minfs (Hz): The frequency boundaries over which to taper the spectrum to infinity.  i.e., frequencies below the first item in the tuple will have an infinite spectrum, the second item in the tuple will not be changed.  A taper from 0 to infinity is applied in between.  The PSD is also tapered from 0.85 * Nyquist to Nyquist.
	:smoothing_frequency (Hz): The target frequency resolution after smoothing.  Lines with bandwidths << smoothing_frequency are removed via a median calculation.  Remaining features will be blurred out to this resolution.

	:returns: A tapered psd

	"""

	#
	# store the psd horizon before tapering
	#

	data = fseries.data.data
	norm_before = numpy.dot(data.conj(), data).real

	#
	# taper to infinity to turn this psd into an effective band pass filter
	#

	deltaF = fseries.deltaF
	kmin = int(minfs[0] / deltaF)
	kmax = int(minfs[1] / deltaF)
	data[(len(data)//2 + 1) - kmin + 1:(len(data)//2 + 1) + kmin] = 0.
	data[(len(data)//2 + 1) + kmin:(len(data)//2 + 1) + kmax] *= numpy.sin(numpy.arange(kmax-kmin) / (kmax-kmin-1.) * numpy.pi / 2.0)**4
	data[(len(data)//2 + 1) - kmax:(len(data)//2 + 1) - kmin] *= numpy.cos(numpy.arange(kmax-kmin) / (kmax-kmin-1.) * numpy.pi / 2.0)**4

	kmin = int(maxfs[0] / deltaF) - 1
	kmax = int(maxfs[1] / deltaF) - 1
	data[(len(data)//2 + 1) + kmax:] = data[:-(len(data)//2 + 1) - kmax] = 0.
	data[(len(data)//2 + 1) + kmin:(len(data)//2 + 1) + kmax] *= numpy.cos(numpy.arange(kmax-kmin) / (kmax-kmin-1.) * numpy.pi / 2.0)**4
	data[(len(data)//2 + 1) - kmax:(len(data)//2 + 1) - kmin] *= numpy.sin(numpy.arange(kmax-kmin) / (kmax-kmin-1.) * numpy.pi / 2.0)**4

	#
	# renormalize after tapering
	#

	fseries.data.data = data * math.sqrt(norm_before / numpy.dot(data.conj(), data).real)

	#
	# done
	#

	return fseries


def tukeywindow(data, samps = 200.):
	assert (len(data) >= 2 * samps) # make sure that the user is requesting something sane
	tp = float(samps) / len(data)
	return lal.CreateTukeyREAL8Window(len(data), tp).data.data


def generate_template(template_bank_row, approximant, sample_rate, duration, f_low, f_high, amporder = 0, order = 7, fwdplan = None, fworkspace = None):
	"""
	Generate a single frequency-domain template, which
	1. is band-limited between f_low and f_high,
	2. has an IFFT which is duration seconds long and
	3. has an IFFT which is sampled at sample_rate Hz
	"""
	if approximant not in templates.gstlal_approximants:
		raise ValueError("Unsupported approximant given %s" % approximant)
	assert f_high <= sample_rate // 2

	# FIXME use hcross somday?
	# We don't here because it is not guaranteed to be orthogonal
	# and we add orthogonal phase later

	parameters = {}
	parameters['m1'] = lal.MSUN_SI * template_bank_row.mass1
	parameters['m2'] = lal.MSUN_SI * template_bank_row.mass2
	parameters['S1x'] = template_bank_row.spin1x
	parameters['S1y'] = template_bank_row.spin1y
	parameters['S1z'] = template_bank_row.spin1z
	parameters['S2x'] = template_bank_row.spin2x
	parameters['S2y'] = template_bank_row.spin2y
	parameters['S2z'] = template_bank_row.spin2z
	parameters['distance'] = 1.e6 * lal.PC_SI
	parameters['inclination'] = 0.
	parameters['phiRef'] = 0.
	parameters['longAscNodes'] = 0.
	parameters['eccentricity'] = 0.
	parameters['meanPerAno'] = 0.
	parameters['deltaF'] = 1.0 / duration
	parameters['f_min'] = f_low
	parameters['f_max'] = f_high
	parameters['f_ref'] = 0.
	parameters['LALparams'] = None
	parameters['approximant'] = lalsim.GetApproximantFromString(str(approximant))

	hplus, hcross = lalsim.SimInspiralFD(**parameters)
	assert len(hplus.data.data) == int(round(f_high * duration)) +1
	# pad the output vector if the sample rate was higher than the
	# requested final frequency
	if f_high < sample_rate / 2:
		fseries = lal.CreateCOMPLEX16FrequencySeries(
			name = hplus.name,
			epoch = hplus.epoch,
			f0 = hplus.f0,
			deltaF = hplus.deltaF,
			length = int(round(sample_rate * duration))//2 +1,
			sampleUnits = hplus.sampleUnits
		)
		fseries.data.data = numpy.zeros(fseries.data.length)
		fseries.data.data[:hplus.data.length] = hplus.data.data[:]
		hplus = fseries
	return hplus


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


def condition_ear_warn_template(approximant, data, epoch_time, sample_rate_max, max_shift_time):
	assert -len(data) / sample_rate_max <= epoch_time < 0.0, "Epoch returned follows a different convention"
	# find the index for the peak sample using the epoch returned by
	# the waveform generator
	epoch_index = -int(epoch_time*sample_rate_max) - 1
	# move the early warning waveforms forward according to the waveform
	# that spends the longest in going from fhigh to ISCO in a given
	# split bank. This effectively ends some waveforms at f < fhigh
	target_index = int(sample_rate_max * max_shift_time)
	data = numpy.roll(data, target_index-epoch_index)
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
	if pandas: ### use pandas implementation if available
		tmp = numpy.copy(interval)
		if pandas.__version__ >= '0.18.1':
			tmp[window_size : len(interval) - window_size] = numpy.array(pandas.Series(tmp).rolling(2 * window_size).median()[2 * window_size - 1 : -1])
		else:
			tmp[window_size : len(interval) - window_size] = pandas.rolling_median(tmp, 2 * window_size)[2 * window_size - 1 : -1]
	else:
		interval = list(interval)
		tmp = numpy.copy(interval)
		A = None
		As = None
		prev = None
		for i in range(window_size, len(interval)-window_size):
			if A is None:
				A = interval[i-window_size:i+window_size]
				ix = numpy.argsort(A)
				As = list(numpy.array(A)[ix])
			else:
				newdata = interval[i+window_size-1]
				A = A + [newdata]
				bisect.insort(As, newdata)
			if len(As) % 2:
				tmp[i] = As[len(As)//2]
			else:
				tmp[i] = (As[len(As)//2-1] + As[len(As)//2]) / 2.
			prev = A.pop(0)
			del As[bisect.bisect_left(As, prev)]
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

	horizon_distance = reference_psd.HorizonDistance(minfs[1], maxfs[0], psd.deltaF, 1.4, 1.4)
	horizon_before = horizon_distance(psd, 8.0)[0]

	#
	# interpolate to new \Delta f
	#

	psd = reference_psd.interpolate_psd(psd, newdeltaF)

	#
	# Smooth the psd
	#

	psddata = psd.data.data
	avgwindow = int(smoothing_frequency / newdeltaF)
	psddata = movingmedian(psddata, avgwindow)
	psddata = movingaverage(psddata, avgwindow)
	psd.data.data = psddata

	#
	# Tapering psd in either side up to infinity if a frequency-domain whitener is used, returns a psd without tapering otherwise.
	# For a time-domain whitener, the tapering is effectively done as a part of deriving a frequency series of the FIR-whitner kernel
	#
	if not FIR_WHITENER:
		#
		# Taper to infinity to turn this psd into an effective band pass filter
		#

		psddata = psd.data.data
		kmin = int(minfs[0] / newdeltaF)
		kmax = int(minfs[1] / newdeltaF)
		psddata[:kmin] = float('Inf')
		psddata[kmin:kmax] /= numpy.sin(numpy.arange(kmax-kmin) / (kmax-kmin-1.) * numpy.pi / 2.0)**4

		kmin = int(maxfs[0] / newdeltaF)
		kmax = int(maxfs[1] / newdeltaF)
		psddata[kmax:] = float('Inf')
		psddata[kmin:kmax] /= numpy.cos(numpy.arange(kmax-kmin) / (kmax-kmin-1.) * numpy.pi / 2.0)**4

		psd.data.data = psddata

	#
	# compute the psd horizon after conditioning and renormalize
	#

	horizon_after = horizon_distance(psd, 8.0)[0]

	psddata = psd.data.data
	psd.data.data = psddata * (horizon_after / horizon_before)**2

	#
	# done
	#

	return psd


class templates_workspace(object):
	def __init__(self, template_table, approximant, psd, f_low, time_slices, autocorrelation_length = None, fhigh = None):
		self.template_table = template_table
		self.approximant = approximant
		self.f_low = f_low
		self.time_slices = time_slices
		self.autocorrelation_length = autocorrelation_length
		self.fhigh = fhigh
		self.sample_rate_max = max(time_slices["rate"])
		self.duration = max(time_slices["end"])
		self.length_max = int(round(self.duration * self.sample_rate_max))

		if self.fhigh is None:
			self.fhigh = self.sample_rate_max / 2.
		# Some input checking to avoid incomprehensible error messages
		if not self.template_table:
			raise ValueError("template list is empty")
		if self.f_low < 0.:
			raise ValueError("f_low must be >= 0. %s" % repr(self.f_low))

		# working f_low to actually use for generating the waveform.  pick
		# template with lowest chirp mass, compute its duration starting
		# from f_low;  the extra time is 10% of this plus 3 cycles (3 /
		# f_low);  invert to obtain f_low corresponding to desired padding.
		# NOTE:  because SimInspiralChirpStartFrequencyBound() does not
		# account for spin, we set the spins to 0 in the call to
		# SimInspiralChirpTimeBound() regardless of the component's spins.
		template = min(self.template_table, key = lambda row: row.mchirp)
		tchirp = lalsim.SimInspiralChirpTimeBound(self.f_low, template.mass1 * lal.MSUN_SI, template.mass2 * lal.MSUN_SI, 0., 0.)
		self.working_f_low = lalsim.SimInspiralChirpStartFrequencyBound(1.1 * tchirp + 3. / self.f_low, template.mass1 * lal.MSUN_SI, template.mass2 * lal.MSUN_SI)

		# Add duration of PSD to template length for PSD ringing, round up to power of 2 count of samples
		self.working_length = templates.ceil_pow_2(self.length_max + round(1./psd.deltaF * self.sample_rate_max))
		self.working_duration = float(self.working_length) / self.sample_rate_max

		if psd is not None:
			# Smooth the PSD and interpolate to required resolution
			self.psd = condition_psd(psd, 1.0 / self.working_duration, minfs = (self.working_f_low, self.f_low), maxfs = (self.sample_rate_max / 2.0 * 0.90, self.sample_rate_max / 2.0))

		if FIR_WHITENER:
			# Compute a frequency response of the time-domain whitening kernel and effectively taper the psd by zero-ing some elements for a FIR kernel
			self.kernel_fseries = taperzero_fseries(create_FIR_whitener_kernel(self.working_length, self.working_duration, self.sample_rate_max, self.psd),\
													  minfs = (self.working_f_low, self.f_low),\
													  maxfs = (self.sample_rate_max / 2.0 * 0.90, self.sample_rate_max / 2.0)\
													 )

		self.revplan = lal.CreateReverseCOMPLEX16FFTPlan(self.working_length, 1)
		self.fwdplan = lal.CreateForwardREAL8FFTPlan(self.working_length, 1)
		self.tseries = lal.CreateCOMPLEX16TimeSeries(
			name = "timeseries",
			epoch = LIGOTimeGPS(0.),
			f0 = 0.,
			deltaT = 1.0 / self.sample_rate_max,
			length = self.working_length,
			sampleUnits = lal.Unit("strain")
		)
		self.fworkspace = lal.CreateCOMPLEX16FrequencySeries(
			name = "template",
			epoch = LIGOTimeGPS(0),
			f0 = 0.0,
			deltaF = 1.0 / self.working_duration,
			length = self.working_length // 2 + 1,
			sampleUnits = lal.Unit("strain s")
		)

		# Calculate the maximum ring down time or maximum shift time
		if approximant in templates.gstlal_IMR_approximants:
			self.max_ringtime = max([chirptime.ringtime(row.mass1*lal.MSUN_SI + row.mass2*lal.MSUN_SI, chirptime.overestimate_j_from_chi(max(row.spin1z, row.spin2z))) for row in self.template_table])
		else:
			if self.sample_rate_max > 2. * self.fhigh:
			# Calculate the maximum time we need to shift the early warning
			# waveforms forward by, calculated by the 3.5 approximation from
			# fhigh to ISCO.
				self.max_shift_time = max([spawaveform.chirptime(row.mass1, row.mass2, 7, fhigh, 0., spawaveform.computechi(row.mass1, row.mass2, row.spin1z, row.spin2z)) for row in self.template_table])

			#
			# Generate each template, downsampling as we go to save memory
			# generate "cosine" component of frequency-domain template.
			# waveform is generated for a canonical distance of 1 Mpc.
			#

	def make_whitened_template(self, template_table_row):
		# FIXME: This is won't work
		#assert template_table_row in self.template_table, "The input Sngl_Inspiral:Table is not found in the workspace."

		# Create template
		fseries = generate_template(template_table_row, self.approximant, self.sample_rate_max, self.working_duration, self.f_low, self.fhigh, fwdplan = self.fwdplan, fworkspace = self.fworkspace)

		if FIR_WHITENER:
			#
			# Compute a product of freq series of the whitening kernel and the template (convolution in time domain) then add quadrature phase
			#
			assert (len(self.kernel_fseries.data.data) // 2 + 1) == len(fseries.data.data), "the size of whitening kernel freq series does not match with a given format of COMPLEX16FrequencySeries."
			fseries.data.data *= self.kernel_fseries.data.data[len(self.kernel_fseries.data.data) // 2 - 1:]
			fseries = templates.QuadraturePhase.add_quadrature_phase(fseries, self.working_length)
		else:
			#
			# whiten and add quadrature phase ("sine" component)
			#

			if self.psd is not None:
				lal.WhitenCOMPLEX16FrequencySeries(fseries, self.psd)
				fseries = templates.QuadraturePhase.add_quadrature_phase(fseries, self.working_length)

		#
		# compute time-domain autocorrelation function
		#

		if self.autocorrelation_length is not None:
			autocorrelation = templates.normalized_autocorrelation(fseries, self.revplan).data.data
		else:
			autocorrelation = None

		#
		# transform template to time domain
		#

		lal.COMPLEX16FreqTimeFFT(self.tseries, fseries, self.revplan)

		data = self.tseries.data.data
		epoch_time = fseries.epoch.gpsSeconds + fseries.epoch.gpsNanoSeconds*1.e-9

		#
		# extract the portion to be used for filtering
		#

		#
		# condition the template if necessary (e.g. line up IMR
		# waveforms by peak amplitude)
		#

		if self.approximant in templates.gstlal_IMR_approximants:
			data, target_index = condition_imr_template(self.approximant, data, epoch_time, self.sample_rate_max, self.max_ringtime)
			# record the new end times for the waveforms (since we performed the shifts)
			template_table_row.end = LIGOTimeGPS(float(target_index-(len(data) - 1.))/self.sample_rate_max)
		else:
			if self.sample_rate_max > self.fhigh*2.:
				data, target_index = condition_ear_warn_template(self.approximant, data, epoch_time, self.sample_rate_max, self.max_shift_time)
				data *= tukeywindow(data, samps = 32)
				# record the new end times for the waveforms (since we performed the shifts)
				template_table_row.end = LIGOTimeGPS(float(target_index)/self.sample_rate_max)
			else:
				data *= tukeywindow(data, samps = 32)

		data = data[-self.length_max:]

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

		sigmasq = norm * len(data) / self.sample_rate_max**2.

		return data, autocorrelation, sigmasq


def generate_templates(template_table, approximant, psd, f_low, time_slices, autocorrelation_length = None, fhigh = None, verbose = False):
	# Create workspace for making template bank
	workspace = templates_workspace(template_table, approximant, psd, f_low, time_slices, autocorrelation_length = autocorrelation_length, fhigh = fhigh)

	# Check parity of autocorrelation length
	if autocorrelation_length is not None:
		if not (autocorrelation_length % 2):
			raise ValueError("autocorrelation_length must be odd (got %d)".format(autocorrelation_length))
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

	for i, row in enumerate(template_table):
		if verbose:
			print("generating template %d/%d:  m1 = %g, m2 = %g, s1x = %g, s1y = %g, s1z = %g, s2x = %g, s2y = %g, s2z = %g" % (i + 1, len(template_table), row.mass1, row.mass2, row.spin1x, row.spin1y, row.spin1z, row.spin2x, row.spin2y, row.spin2z), file=sys.stderr)
		# FIXME: ensure the row is in template_table
		template, autocorrelation, this_sigmasq = workspace.make_whitened_template(row)

		sigmasq.append(this_sigmasq)

		if autocorrelation is not None:
			autocorrelation_bank[i, ::-1] = numpy.concatenate((autocorrelation[-(autocorrelation_length // 2):], autocorrelation[:(autocorrelation_length // 2  + 1)]))

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

			stride = int(round(workspace.sample_rate_max / time_slice['rate']))
			begin_index = workspace.length_max - int(round(time_slice['begin'] * workspace.sample_rate_max)) + stride - 1
			end_index = workspace.length_max - int(round(time_slice['end'] * workspace.sample_rate_max)) + stride - 1
			# make sure the rates are commensurate
			assert stride * time_slice['rate'] == workspace.sample_rate_max

			# extract every stride-th sample.  we multiply by
			# \sqrt{stride} to maintain inner product
			# normalization so that the templates still appear
			# to be unit vectors at the reduced sample rate.
			# note that the svd returns unit basis vectors
			# regardless so this factor has no effect on the
			# normalization of the basis vectors used for
			# filtering but it ensures that the chifacs values
			# have the correct relative normalization.
			template_bank[j][(2*i+0),:] = template.real[end_index:begin_index:stride] * math.sqrt(stride)
			template_bank[j][(2*i+1),:] = template.imag[end_index:begin_index:stride] * math.sqrt(stride)

	return template_bank, autocorrelation_bank, autocorrelation_mask, sigmasq, workspace


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

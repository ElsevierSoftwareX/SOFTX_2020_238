# Copyright (C) 2017-2018  Patrick Godwin
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


import bisect
from collections import defaultdict

import numpy


class TemplateGenerator(object):
	"""Generate templates within a parameter space.

	Parameters
	----------
	parameters : Dict[Tuple[float, float]]
		Specified in the form {'parameter': (param_low, param_high)}.
		NOTE: 'frequency' parameter is required.
	rates : List[int]
		Sampling rates used when performing matched filtering.
	mismatch : float, optional
		Minimal mismatch for adjacent templates to be placed.
	tolerance : float, optional
		Maximum value at the waveform edges, used to mitigate
		edge effects when matched filtering.
	downsample_factor : float, optional
		Used to shift the distribution of templates in a particular
		frequency band downwards to avoid placing templates too close
		to where we lose power due to due to low-pass rolloff.

	"""
	_default_parameters = ['frequency']

	def __init__(
		self,
		parameters,
		rates,
		bins,
		mismatch=0.2,
		tolerance=5e-3,
		downsample_factor=0.8
	):

		# set parameter range
		self.parameter_ranges = dict(parameters)
		self.parameter_ranges['frequency'] = (
			downsample_factor * self.parameter_ranges['frequency'][0],
			downsample_factor * self.parameter_ranges['frequency'][1]
		)
		self.parameter_names = self._default_parameters
		self.phases = [0., numpy.pi / 2.]

		# define grid spacing and edge rolloff for templates
		self.mismatch = mismatch
		self.tolerance = tolerance

		# determine how to scale down templates considered
		# in frequency band to deal with low pass rolloff
		self.downsample_factor = downsample_factor

		# determine frequency bands and bins considered
		self.rates = sorted(set(rates))
		self.bins = bins

		# determine lower frequency limits for rate
		# conversion based on downsampling factor
		self._breakpoints = [self.downsample_factor * rate / 2. for rate in self.rates[:-1]]

		# define grid and template duration
		self.parameter_grid = defaultdict(list)
		for point in self.generate_grid():
			rate = point[0]
			params = point[1:]
			template = {name: param for name, param in zip(self.parameter_names, params)}
			template['duration'] = self.duration(*params)
			self.parameter_grid[rate].append(template)

		# specify time samples, number of sample points, latencies,
		# filter durations associated with each sampling rate
		self._times = {}
		self._latency = {}
		self._sample_pts = {}
		self._filter_duration = {}

		# generate binning indices for each sampling rate
		# store both the mapping and inverse mapping
		self._index_by_bin = {rate: [[] for bin_ in range(len(self.bins))] for rate in self.rates}
		self._idx_to_waveform = {rate: [[] for bin_ in range(len(self.bins))] for rate in self.rates}
		self._bin_mixer_coeffs = {rate: {bin_: {} for bin_ in range(len(self.bins))} for rate in self.rates}

		for rate in self.rates:
			for idx, waveform in enumerate(self.parameter_grid[rate]):
				freq_idx = self.bins[waveform['frequency']]
				self._index_by_bin[rate][freq_idx].append(idx)
				self._idx_to_waveform[rate][freq_idx].append(waveform)

		# set up mixing coefficients
		for rate in self.rates:
			for bin_idx in range(len(self.bins)):
				waveform_indices = self._index_by_bin[rate][bin_idx]
				num_cols = len(waveform_indices)
				num_rows = len(self.parameter_grid[rate])
				mixer_coeffs = numpy.zeros((num_rows, num_cols))
				for col_idx, row_idx in enumerate(waveform_indices):
					mixer_coeffs[row_idx, col_idx] = 1
					#self._idx_to_waveform[rate][freq_idx].append(waveform)

				self._bin_mixer_coeffs[rate][bin_idx] = mixer_coeffs

	def duration(self, *params):
		"""
		Return the duration of a waveform such that its
		edges will die out to tolerance of the peak.
		"""
		return NotImplementedError

	def generate_templates(self, rate, quadrature=True, sampling_rate=None):
		"""
		Generate all templates corresponding to a parameter
		range for a given sampling rate.

		If quadrature is set, yield two templates corresponding
		to in-phase and quadrature components of the waveform.
		"""
		return NotImplementedError

	def waveform(self, rate, phase, *params):
		"""
		Construct waveforms that taper to tolerance at edges of window.
		"""
		return NotImplementedError

	def generate_grid(self):
		"""
		Generates (rate, *param) pairs based on parameter ranges.
		"""
		return NotImplementedError

	def latency(self, rate):
		"""
		Return the latency in sample points associated with
		waveforms with a particular sampling rate.
		"""
		return NotImplementedError

	def times(self, rate):
		"""
		Return the time samples associated with waveforms
		with a particular sampling rate.
		"""
		return NotImplementedError

	def sample_pts(self, rate):
		"""
		Return the number of sample points associated with
		waveforms with a particular sampling rate.
		"""
		max_duration = max(template['duration'] for template in self.parameter_grid[rate])
		return self._sample_pts.get(rate, self._round_to_next_odd(max_duration * rate))

	def filter_duration(self, rate):
		"""
		Return the filter duration associated with waveforms
		with a particular sampling rate.
		"""
		return self._filter_duration.get(rate, self.times(rate)[-1] - self.times(rate)[0])

	def frequency2rate(self, frequency):
		"""
		Maps a frequency to the correct sampling rate considered.
		"""
		idx = bisect.bisect(self._breakpoints, frequency)
		return self.rates[idx]

	def index_by_bin(self, rate):
		"""
		Given a sampling rate, returns the indices of waveforms
		whose central frequencies fall within each frequency bin.
		"""
		return self._index_by_bin[rate]

	def bin_mixer_coeffs(self, rate, bin_idx):
		"""
		Gives matrix mixing coefficients to split up streams based
		on a frequency binning and sampling rate.
		"""
		return self._bin_mixer_coeffs[rate][bin_idx]

	def index_to_waveform(self, rate, bin_idx, row_idx):
		"""
		Maps a sampling rate and bin/row indices to a particular waveform.
		This is useful in conjunction with index_by_bin() to provide
		an inverse mapping to the index_by_bin() operation.
		"""
		return self._idx_to_waveform[rate][bin_idx][row_idx]

	def _round_to_next_odd(self, n):
		return int(numpy.ceil(n) // 2 * 2 + 1)


class HalfSineGaussianGenerator(TemplateGenerator):
	"""
	Generates half-Sine-Gaussian templates based on
	a f, Q range and a sampling frequency.
	"""
	_default_parameters = ['frequency', 'q']

	def duration(self, f, q):
		"""
		return the duration of a half sine-gaussian waveform such
		that its edges will die out to tolerance of the peak.
		"""
		return 0.5 * (q / (2. * numpy.pi * f)) * numpy.log(1. / self.tolerance)

	def generate_templates(self, rate, quadrature=True, sampling_rate=None):
		"""
		generate all half sine gaussian templates corresponding
		to a parameter range and template duration for a given
		sampling rate.
		"""
		if not sampling_rate:
			sampling_rate = rate

		for template in self.parameter_grid[rate]:
			if quadrature:
				for phase in self.phases:
					yield self.waveform(sampling_rate, phase, template['frequency'], template['q'])
			else:
				yield self.waveform(sampling_rate, self.phases[0], template['frequency'], template['q'])

	def waveform(self, rate, phase, f, q):
		"""
		construct half sine gaussian waveforms that taper to
		tolerance at edges of window.
		f is the central frequency of the waveform
		"""
		assert f < rate / 2.

		# phi is the central frequency of the sine gaussian
		tau = q / (2. * numpy.pi * f)
		template = numpy.cos(2. * numpy.pi * f * self.times(rate) + phase)
		template *= numpy.exp(-1. * self.times(rate)**2. / tau**2.)

		# normalize to have unit length in their vector space
		inner_product = numpy.sum(template * template)
		norm_factor = 1. / inner_product**0.5

		return norm_factor * template

	def generate_grid(self):
		"""
		Generates (f_samp, f, Q) pairs based on f, Q ranges
		"""
		q_min, q_max = self.parameter_ranges['q']
		f_min, f_max = self.parameter_ranges['frequency']
		for q in self._generate_q_values(q_min, q_max):
			num_f = self._num_f_templates(f_min, f_max, q)
			for l in range(num_f):
				f = f_min * (f_max / f_min)**((0.5 + l) / num_f)
				rate = self.frequency2rate(f)
				if f < (rate / 2) / (1 + (numpy.sqrt(11) / q)):
					yield rate, f, q
				elif rate != max(self.rates):
					yield (2 * rate), f, q

	def latency(self, rate):
		"""
		Return the latency in sample points associated
		with half-Sine-Gaussians with a particular sampling rate.
		"""
		return 0

	def times(self, rate):
		"""
		Return the time samples associated with half-Sine-Gaussians
		with a particular sampling rate.
		"""
		if rate not in self._times:
			t0 = -float(self.sample_pts(rate) - 1) / rate
			self._times[rate] = numpy.linspace(t0, 0, self.sample_pts(rate), endpoint=True)
		return self._times[rate]

	def _num_q_templates(self, q_min, q_max):
		"""
		Minimum number of distinct Q values to generate based on Q_min, Q_max, and mismatch.
		"""
		mismatch_factor = 1. / (2. * numpy.sqrt(self.mismatch / 3.))
		return int(numpy.ceil(mismatch_factor * (1. / numpy.sqrt(2)) * numpy.log(q_max / q_min)))

	def _num_f_templates(self, f_min, f_max, q):
		"""
		Minimum number of distinct frequency values to generate based on f_min, f_max, and mismatch.
		"""
		mismatch_factor = 1. / (2. * numpy.sqrt(self.mismatch / 3.))
		return int(numpy.ceil(mismatch_factor * (numpy.sqrt(2. + q**2.) / 2.) * numpy.log(f_max / f_min)))

	def _generate_q_values(self, q_min, q_max):
		"""
		List of Q values to generate based on Q_min, Q_max, and mismatch params.
		"""
		num_q = self._num_q_templates(q_min, q_max)
		return [q_min * (q_max / q_min)**((0.5 + q) / num_q) for q in range(num_q)]


class SineGaussianGenerator(HalfSineGaussianGenerator):
	"""
	Generates sine gaussian templates based on a f, Q range
	and a sampling frequency.
	"""
	_default_parameters = ['frequency', 'q']

	def duration(self, f, q):
		"""
		return the duration of a sine-gaussian waveform such
		that its edges will die out to tolerance of the peak.
		"""
		return 2 * super(SineGaussianGenerator, self).duration(f, q)

	def latency(self, rate):
		"""
		Return the latency in sample points associated with
		half-Sine-Gaussians with a particular sampling rate.
		"""
		if not rate in self._latency:
			self._latency[rate] = int((self.sample_pts(rate) - 1) / 2)
		return self._latency[rate]

	def times(self, rate):
		"""
		Return the time samples associated with half-Sine-Gaussians
		with a particular sampling rate.
		"""
		if not rate in self._times:
			dt = ((self.sample_pts(rate) - 1) / 2.) / rate
			self._times[rate] = numpy.linspace(-dt, dt, self.sample_pts(rate), endpoint=True)
		return self._times[rate]


class TaperedSineGaussianGenerator(HalfSineGaussianGenerator):
	"""
	Generates tapered sine-Gaussian templates based on
	f, Q range and a sampling frequency.

	Tapering is based off of a 'max_latency' kwarg that
	a sine-Gaussian template should incur.
	"""
	_default_parameters = ['frequency', 'q']

	def __init__(
		self,
		parameters,
		rates,
		bins,
		mismatch=0.2,
		tolerance=5e-3,
		downsample_factor=0.8,
		max_latency=1
	):
		self.max_latency = max_latency
		super(TaperedSineGaussianGenerator, self).__init__(
			parameters,
			rates,
			bins,
			mismatch=mismatch,
			tolerance=tolerance,
			downsample_factor=downsample_factor
		)

	def waveform(self, rate, phase, f, q):
		"""
		construct tapered sine-Gaussian waveforms that taper
		to tolerance at edges of window
		f is the central frequency of the waveform
		"""
		assert f < rate/2.

		# phi is the central frequency of the sine gaussian
		tau = q / (2. * numpy.pi * f)
		damping_factor = numpy.log(self.tolerance) / self.max_latency
		template = numpy.cos(2. * numpy.pi * f * self.times(rate) + phase)
		template *= numpy.exp(-1. * self.times(rate)**2. / tau**2.)
		template *= numpy.minimum(1, numpy.exp(damping_factor * self.times(rate)))

		# normalize sine gaussians to have unit length in their vector space
		inner_product = numpy.sum(template * template)
		norm_factor = 1. / inner_product**0.5

		return norm_factor * template

	def duration(self, f, q):
		"""
		return the duration of a tapered sine-gaussian waveform such that
		its edges will die out to tolerance of the peak.
		"""
		hsg_duration = super(TaperedSineGaussianGenerator, self).duration(f, q)
		return hsg_duration + min(self.max_latency, hsg_duration)

	def latency(self, rate):
		"""
		Return the latency in sample points associated with
		tapered Sine-Gaussians with a particular sampling rate.
		"""
		if not rate in self._latency:
			t0 = int((self.sample_pts(rate) - 1) / 2)
			tf = int(self.max_latency * rate)
			self._latency[rate] = min(t0, tf)
		return self._latency[rate]

	def times(self, rate):
		"""
		Return the time samples associated with tapered
		Sine-Gaussians with a particular sampling rate.
		"""
		if not rate in self._times:
			t0 = -((self.sample_pts(rate) - 1) / 2.) / rate
			tf = float(self.latency(rate)) / rate
			self._times[rate] = numpy.linspace(t0, tf, self.sample_pts(rate), endpoint = True)
		return self._times[rate]

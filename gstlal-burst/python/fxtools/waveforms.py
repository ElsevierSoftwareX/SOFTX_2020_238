#!/usr/bin/env python
#
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



####################
# 
#     preamble
#
#################### 


import bisect
from collections import defaultdict

import numpy


####################
#
#     classes
#
####################

#----------------------------------
### structures to generate basis waveforms

class TemplateGenerator(object):
	"""
	Base class to generate templates based on the parameter space, given the following parameters:

	  Required:

        * parameter_ranges: A dictionary of tuples in the form {'parameter': (param_low, param_high)}.
                            Note that 'frequency' will always be a required parameter.
        * sampling_rates: A list of sampling rates that are used when performing the matched filtering
                          against these templates.

      Optional:

        * mismatch: A minimal mismatch for adjacent templates to be placed.
        * tolerance: Maximum value at the waveform edges, used to mitigate edge effects when matched filtering.
        * downsample_factor: Used to shift the distribution of templates in a particular frequency band downwards
                             to avoid placing templates too close to where we lose power due to due to low-pass rolloff.
                             This parameter doesn't need to be modified in general use.
	"""
	_default_parameters = ['frequency']

	def __init__(self, parameter_ranges, sampling_rates, mismatch=0.2, tolerance=5e-3, downsample_factor=0.8):

		### set parameter range
		self.parameter_ranges = dict(parameter_ranges)
		self.parameter_ranges['frequency'] = (downsample_factor * self.parameter_ranges['frequency'][0], downsample_factor * self.parameter_ranges['frequency'][1])
		self.parameter_names = self._default_parameters
		self.phases = [0., numpy.pi/2.]

		### define grid spacing and edge rolloff for templates
		self.mismatch = mismatch
		self.tolerance = tolerance

		### determine how to scale down templates considered
		### in frequency band to deal with low pass rolloff
		self.downsample_factor = downsample_factor

		### determine frequency bands considered
		self.rates = sorted(set(sampling_rates))

		### determine lower frequency limits for rate conversion based on downsampling factor
		self._freq_breakpoints = [self.downsample_factor * rate / 2. for rate in self.rates[:-1]]

		### define grid and template duration
		self.parameter_grid = defaultdict(list)
		for point in self.generate_grid():
			rate = point[0]
			params = point[1:]
			template = {param_name: param for param_name, param in zip(self.parameter_names, params)}
			template['duration'] = self.duration(*params)
			self.parameter_grid[rate].append(template)

		### specify time samples, number of sample points, latencies,
		### filter durations associated with each sampling rate
		self._times = {}
		self._latency = {}
		self._sample_pts = {}
		self._filter_duration = {}

	def duration(self, *params):
		"""
		Return the duration of a waveform such that its edges will die out to tolerance of the peak.
		"""
		return NotImplementedError

	def generate_templates(self, rate, quadrature = True, sampling_rate = None):
		"""
		Generate all templates corresponding to a parameter range for a given sampling rate.
		If quadrature is set, yield two templates corresponding to in-phase and quadrature components of the waveform.
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
		Return the latency in sample points associated with waveforms with a particular sampling rate.
		"""
		return NotImplementedError

	def times(self, rate):
		"""
		Return the time samples associated with waveforms with a particular sampling rate.
		"""
		return NotImplementedError

	def sample_pts(self, rate):
		"""
		Return the number of sample points associated with waveforms with a particular sampling rate.
		"""
		return self._sample_pts.get(rate, self._round_to_next_odd(max(template['duration'] for template in self.parameter_grid[rate]) * rate))

	def filter_duration(self, rate):
		"""
		Return the filter duration associated with waveforms with a particular sampling rate.
		"""
		return self._filter_duration.get(rate, self.times(rate)[-1] - self.times(rate)[0])

	def frequency2rate(self, frequency):
		"""
		Maps a frequency to the correct sampling rate considered.
		"""
		idx = bisect.bisect(self._freq_breakpoints, frequency)
		return self.rates[idx]

	def _round_to_next_odd(self, n):
		return int(numpy.ceil(n) // 2 * 2 + 1)

class HalfSineGaussianGenerator(TemplateGenerator):
	"""
	Generates half-Sine-Gaussian templates based on a f, Q range and a sampling frequency.
	"""
	_default_parameters = ['frequency', 'q']

	def duration(self, f, q):
		"""
		return the duration of a half sine-gaussian waveform such that its edges will die out to tolerance of the peak.
		"""
		return 0.5 * (q/(2.*numpy.pi*f)) * numpy.log(1./self.tolerance)

	def generate_templates(self, rate, quadrature = True, sampling_rate = None):
		"""
		generate all half sine gaussian templates corresponding to a parameter range and template duration
		for a given sampling rate.
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
		construct half sine gaussian waveforms that taper to tolerance at edges of window
		f is the central frequency of the waveform
		"""
		assert f < rate/2.

		# phi is the central frequency of the sine gaussian
		tau = q/(2.*numpy.pi*f)
		template = numpy.cos(2.*numpy.pi*f*self.times(rate) + phase)*numpy.exp(-1.*self.times(rate)**2./tau**2.)

		# normalize sine gaussians to have unit length in their vector space
		inner_product = numpy.sum(template*template)
		norm_factor = 1./(inner_product**0.5)

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
				f = f_min * (f_max/f_min)**( (0.5+l) /num_f)
				rate = self.frequency2rate(f)
				if f < (rate/2) / (1 + (numpy.sqrt(11)/q)):
					yield rate, f, q
				elif rate != max(self.rates):
					yield (2 * rate), f, q

	def latency(self, rate):
		"""
		Return the latency in sample points associated with half-Sine-Gaussians with a particular sampling rate.
		"""
		return 0

	def times(self, rate):
		"""
		Return the time samples associated with half-Sine-Gaussians with a particular sampling rate.
		"""
		return self._times.get(rate, numpy.linspace(-float(self.sample_pts(rate) - 1) / rate, 0, self.sample_pts(rate), endpoint=True))
	
	def _num_q_templates(self, q_min, q_max):
		"""
		Minimum number of distinct Q values to generate based on Q_min, Q_max, and mismatch params.
		"""
		return int(numpy.ceil(1./(2.*numpy.sqrt(self.mismatch/3.))*(1./numpy.sqrt(2))*numpy.log(q_max/q_min)))
	
	def _num_f_templates(self, f_min, f_max, q):
		"""
		Minimum number of distinct frequency values to generate based on f_min, f_max, and mismatch params.
		"""
		return int(numpy.ceil(1./(2.*numpy.sqrt(self.mismatch/3.))*(numpy.sqrt(2.+q**2.)/2.)*numpy.log(f_max/f_min)))
	
	def _generate_q_values(self, q_min, q_max):
		"""
		List of Q values to generate based on Q_min, Q_max, and mismatch params.
		"""
		num_q = self._num_q_templates(q_min, q_max)
		return [q_min*(q_max/q_min)**((0.5+q)/num_q) for q in range(num_q)]

class SineGaussianGenerator(HalfSineGaussianGenerator):
	"""
	Generates sine gaussian templates based on a f, Q range and a sampling frequency.
	"""
	_default_parameters = ['frequency', 'q']

	def duration(self, f, q):
		"""
		return the duration of a sine-gaussian waveform such that its edges will die out to tolerance of the peak.
		"""
		return 2 * super(SineGaussianGenerator, self).duration(f, q)

	def latency(self, rate):
		"""
		Return the latency in sample points associated with half-Sine-Gaussians with a particular sampling rate.
		"""
		return self._latency.get(rate, int((self.sample_pts(rate) - 1)  / 2))

	def times(self, rate):
		"""
		Return the time samples associated with half-Sine-Gaussians with a particular sampling rate.
		"""
		return self._times.get(rate, numpy.linspace(-((self.sample_pts(rate) - 1)  / 2.) / rate, ((self.sample_pts(rate) - 1)  / 2.) / rate, self.sample_pts(rate), endpoint=True))

class TaperedSineGaussianGenerator(HalfSineGaussianGenerator):
	"""
	Generates tapered sine-Gaussian templates based on a f, Q range and a sampling frequency.

	Tapering is based off of a 'max_latency' kwarg that a sine-Gaussian template should incur.
	"""
	_default_parameters = ['frequency', 'q']

	def __init__(self, parameter_ranges, sampling_rates, mismatch=0.2, tolerance=5e-3, downsample_factor=0.8, max_latency=1):
		self.max_latency = max_latency
		super(TaperedSineGaussianGenerator, self).__init__(parameter_ranges, sampling_rates, mismatch=mismatch, tolerance=tolerance, downsample_factor=downsample_factor)

	def waveform(self, rate, phase, f, q):
		"""
		construct tapered sine-Gaussian waveforms that taper to tolerance at edges of window
		f is the central frequency of the waveform
		"""
		assert f < rate/2.

		# phi is the central frequency of the sine gaussian
		tau = q/(2.*numpy.pi*f)
		damping_factor = numpy.log(self.tolerance) / self.max_latency
		template = numpy.cos(2.*numpy.pi*f*self.times(rate) + phase)*numpy.exp(-1.*self.times(rate)**2./tau**2.) * numpy.minimum(1, numpy.exp(damping_factor * self.times(rate)))

		# normalize sine gaussians to have unit length in their vector space
		inner_product = numpy.sum(template*template)
		norm_factor = 1./(inner_product**0.5)

		return norm_factor * template

	def duration(self, f, q):
		"""
		return the duration of a sine-gaussian waveform such that its edges will die out to tolerance of the peak.
		"""
		hsg_duration = super(TaperedSineGaussianGenerator, self).duration(f, q)
		return hsg_duration + min(self.max_latency, hsg_duration)

	def latency(self, rate):
		"""
		Return the latency in sample points associated with half-Sine-Gaussians with a particular sampling rate.
		"""
		return min(self._latency.get(rate, int((self.sample_pts(rate) - 1)  / 2)), int(self.max_latency * rate))

	def times(self, rate):
		"""
		Return the time samples associated with half-Sine-Gaussians with a particular sampling rate.
		"""
		return self._times.get(rate, numpy.linspace(-((self.sample_pts(rate) - 1)  / 2.) / rate, float(self.latency(rate)) / rate, self.sample_pts(rate), endpoint=True))

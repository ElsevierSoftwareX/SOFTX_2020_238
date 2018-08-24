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
from collections import Counter, defaultdict, deque
import glob
import logging
import os
import sys

import h5py
import numpy

from lal import gpstime
from gstlal import aggregator


####################
# 
#    functions
#
####################

#----------------------------------
### hdf5 utilities

def get_dataset(path, base, name = 'data', group = None):
	"""
	open a dataset at @param path with name @param base and return the data
	"""
	fname = os.path.join(path, "%s.h5" % base)
	try:
		with h5py.File(fname, 'r') as hfile:
			if group:
				data = numpy.array(hfile[group][name])
			else:
				data = numpy.array(hfile[name])
		return fname, data
	except IOError:
		return fname, []

def create_new_dataset(path, base, data, name = 'data', group = None, tmp = False):
	"""
	A function to create a new dataset with data @param data.
	The data will be stored in an hdf5 file at path @param path with
	base name @param base.  You can also make a temporary file.
	"""
	if tmp:
		fname = os.path.join(path, "%s.h5.tmp" % base)
	else:
		fname = os.path.join(path, "%s.h5" % base)

	# create dir if it doesn't exist
	if not os.path.exists(path):
		aggregator.makedir(path)

	# save data to hdf5
	with h5py.File(fname, 'a') as hfile:
		if group:
			if group not in hfile:
				hfile.create_group(group)
			hfile[group].create_dataset(name, data=data)
		else:
			hfile.create_dataset(name, data=data)

	return fname

#----------------------------------
### gps time utilities

def in_new_epoch(new_gps_time, prev_gps_time, gps_epoch):
	"""
	Returns whether new and old gps times are in different
	epochs.

	>>> in_new_epoch(1234561200, 1234560000, 1000)
	True
	>>> in_new_epoch(1234561200, 1234560000, 10000)
	False

	"""
	return (new_gps_time - floor_div(prev_gps_time, gps_epoch)) >= gps_epoch

def floor_div(x, n):
	"""
	Floor an integer by removing its remainder
	from integer division by another integer n.

	>>> floor_div(163, 10)
	160
	>>> floor_div(158, 10)
	150

	"""
	assert n > 0
	return (x / n) * n

#----------------------------------
### pathname utilities

def to_trigger_path(rootdir, basename, start_time, job_id, subset_id):
	"""
	Given a basepath, instrument, description, start_time, job_id, will return a
	path pointing to a directory structure in the form::

		${rootdir}/${basename}/${basename}-${start_time_mod1e5}/${basename}-${job_id}-${subset_id}/
	"""
	start_time_mod1e5 = str(start_time).zfill(10)[:5]
	return os.path.join(rootdir, basename, '-'.join([basename, start_time_mod1e5]), '-'.join([basename, job_id, subset_id]))

def to_trigger_filename(basename, start_time, duration, suffix, tmp=False):
	"""
	Given an instrument, description, start_time, and duration, will return a
	filename suitable with the T050017 file naming convention, in the form::

		${basename}-${start_time}-{duration}.${suffix}

	or if a temporary file is requested::

		${basename}-${start_time}-{duration}.${suffix}.tmp
	"""
	if tmp:
		return '%s-%d-%d.%s.tmp' % (basename, start_time, duration, suffix)
	else:
		return '%s-%d-%d.%s' % (basename, start_time, duration, suffix)

def latency_name(stage_name, stage_num, channel, rate=None):
	"""
	Returns a properly formatted latency element name based on stage,
	channel, and rate information.
	"""
	if rate:
		return 'stage%d_%s_%s_%s' % (stage_num, stage_name, str(rate).zfill(5), channel)
	else:
		return 'stage%d_%s_%s' % (stage_num, stage_name, channel)

#----------------------------------
### logging utilities

# FIXME: shamelessly copied from iDQ's logs module, until this dependency is added in to gstlal-iDQ proper.

def get_logger(logname, log_level=10, rootdir='.', verbose=False):
    '''
    standardize how we instantiate loggers
    '''
    logger = logging.getLogger(logname)
    logger.setLevel(log_level)

    # set up FileHandler for output file
    log_path = os.path.join(rootdir, logname+'.log')
    handlers = [logging.FileHandler(log_path)]

    # set up handler for stdout
    if verbose:
        handlers.append( logging.StreamHandler() )

    # add handlers to logger
    formatter = gen_formatter()
    for handler in handlers:
        handler.setFormatter( formatter )
        logger.addHandler( handler )

    return logger

def gen_formatter():
    """
    standarizes formatting for loggers
    returns an instance of logging.Formatter
    """
    return logging.Formatter('%(asctime)s | %(name)s : %(levelname)s : %(message)s')

####################
#
#     classes
#
####################

#----------------------------------
### Feature I/O structures

class FeatureData(object):
	"""
	Base class for saving feature data.
	Extend for a specific file-based implementation.
	"""
	def __init__(self, columns, keys = None, **kwargs):
		self.keys = keys
		self.columns = columns
		self.feature_data = {}

	def dump(self, path):
		raise NotImplementedError

	def append(self, key, value):
		raise NotImplementedError

	def clear(self):
		raise NotImplementedError

class HDF5TimeseriesFeatureData(FeatureData):
	"""
	Saves feature data to hdf5 as regularly sampled timeseries.
	"""
	def __init__(self, columns, keys, **kwargs):
		super(HDF5TimeseriesFeatureData, self).__init__(columns, keys = keys, **kwargs)
		self.cadence = kwargs.pop('cadence')
		self.sample_rate = kwargs.pop('sample_rate')
		self.dtype = [(column, '<f8') for column in self.columns]
		self.feature_data = {key: numpy.empty((self.cadence * self.sample_rate,), dtype = self.dtype) for key in keys}
		self.last_save_time = 0
		self.clear()

	def dump(self, path, base, start_time, tmp = False):
		"""
		Saves the current cadence of features to disk and clear out data
		"""
		name = "%d_%d" % (start_time, self.cadence)
		for key in self.keys:
			create_new_dataset(path, base, self.feature_data[key], name=name, group=key, tmp=tmp)
		self.clear()

	def append(self, timestamp, features):
		"""
		Append a feature buffer to data structure
		"""
		self.last_save_time = floor_div(timestamp, self.cadence)
		time_idx = (timestamp - self.last_save_time) * self.sample_rate

		for key in features.keys():
			for row_idx, row in enumerate(features[key]):
				if row:
					idx = time_idx + row_idx
					self.feature_data[key][idx] = numpy.array(tuple(row[col] for col in self.columns), dtype=self.dtype)

	def clear(self):
		for key in self.keys:
			self.feature_data[key][:] = numpy.nan

class HDF5ETGFeatureData(FeatureData):
	"""!
	Saves feature data with varying dataset lengths (when run in ETG mode) to hdf5.
	"""
	def __init__(self, columns, keys, **kwargs):
		super(HDF5ETGFeatureData, self).__init__(columns, keys = keys, **kwargs)
		self.cadence = kwargs.pop('cadence')
		self.dtype = [(column, '<f8') for column in self.columns]
		self.feature_data = {key: [] for key in keys}
		self.clear()

	def dump(self, path, base, start_time, tmp = False):
		"""
		Saves the current cadence of gps triggers to disk and clear out data
		"""
		name = "%d_%d" % (start_time, self.cadence)
		for key in self.keys:
			create_new_dataset(path, base, numpy.array(self.feature_data[key], dtype=self.dtype), name=name, group=key, tmp=tmp)
		self.clear()

	def append(self, timestamp, features):
		"""
		Append a trigger row to data structure

		NOTE: timestamp arg is here purely to match API, not used in append
		"""
		for key in features.keys():
			for row in features[key]:
				self.feature_data[key].append(tuple(row[col] for col in self.columns))

	def clear(self):
		for key in self.keys:
			self.feature_data[key] = []

class TimeseriesFeatureQueue(object):
	"""
	Class for storing regularly sampled feature data.
	NOTE: assumes that ingested features are time ordered.

	Example:
		>>> # create the queue
		>>> columns = ['trigger_time', 'snr']
		>>> channels = ['channel1']
		>>> queue = TimeseriesFeatureQueue(channels, columns, sample_rate=1)
		>>> # add features
		>>> queue.append(123450, 'channel1', {'trigger_time': 123450.3, 'snr': 3.0})
		>>> queue.append(123451, 'channel1', {'trigger_time': 123451.7, 'snr': 6.5})
		>>> queue.append(123452, 'channel1', {'trigger_time': 123452.4, 'snr': 5.2})
		>>> # get oldest feature
		>>> queue.pop()
		{'timestamp': 123450, 'features': {'channel1': [{'snr': 3.0, 'trigger_time': 123450.3}]}}
		>>> # flush queue and get rest of features
		>>> queue.flush()
		>>> len(queue)
		2
		>>> queue.pop()
		{'timestamp': 123451, 'features': {'channel1': [{'snr': 6.5, 'trigger_time': 123451.7}]}}
		>>> queue.pop()
		{'timestamp': 123452, 'features': {'channel1': [{'snr': 5.2, 'trigger_time': 123452.4}]}}

	"""
	def __init__(self, channels, columns, **kwargs):
		self.channels = channels
		self.columns = columns
		self.sample_rate = kwargs.pop('sample_rate')
		self.out_queue = deque(maxlen = 5)
		self.in_queue = {}
		self.counter = Counter()
		self.last_timestamp = 0
		self.effective_latency = 2

	def append(self, timestamp, channel, row):
		if timestamp > self.last_timestamp:
			### create new buffer if one isn't available for new timestamp
			if timestamp not in self.in_queue:
				self.in_queue[timestamp] = self._create_buffer()
			self.counter[timestamp] += 1

			### store row, aggregating if necessary
			idx = self._idx(row['trigger_time'])
			if not self.in_queue[timestamp][channel][idx] or (row['snr'] > self.in_queue[timestamp][channel][idx]['snr']):
				self.in_queue[timestamp][channel][idx] = row

			### check if there's enough new samples that the oldest sample needs to be pushed
			if len(self.counter) > self.effective_latency:
				oldest_timestamp = min(self.counter.keys())
				self.last_timestamp = oldest_timestamp
				self.out_queue.append({'timestamp': oldest_timestamp, 'features': self.in_queue.pop(oldest_timestamp)})
				del self.counter[oldest_timestamp]

	def pop(self):
		if len(self):
			return self.out_queue.popleft()

	def flush(self):
		while self.in_queue:
			oldest_timestamp = min(self.counter.keys())
			del self.counter[oldest_timestamp]
			self.out_queue.append({'timestamp': oldest_timestamp, 'features': self.in_queue.pop(oldest_timestamp)})

	def _create_buffer(self):
		return {channel: [None for x in range(self.sample_rate)] for channel in self.channels}

	def _idx(self, timestamp):
		return int(numpy.floor((timestamp % 1) * self.sample_rate))

	def __len__(self):
		return len(self.out_queue)

class ETGFeatureQueue(object):
	"""
	Class for storing feature data when pipeline is running in ETG mode, i.e. report all triggers above an SNR threshold.
	NOTE: assumes that ingested features are time ordered.
	"""
	def __init__(self, channels, columns, **kwargs):
		self.channels = channels
		self.columns = columns
		self.out_queue = deque(maxlen = 5)
		self.in_queue = {}
		self.counter = Counter()
		self.last_timestamp = 0
		self.effective_latency = 2

	def append(self, timestamp, channel, row):
		if timestamp > self.last_timestamp:
			### create new buffer if one isn't available for new timestamp
			if timestamp not in self.in_queue:
				self.in_queue[timestamp] = self._create_buffer()
			self.counter[timestamp] += 1

			### store row
			self.in_queue[timestamp][channel].append(row)

			### check if there's enough new samples that the oldest sample needs to be pushed
			if len(self.counter) > self.effective_latency:
				oldest_timestamp = min(self.counter.keys())
				self.last_timestamp = oldest_timestamp
				self.out_queue.append({'timestamp': oldest_timestamp, 'features': self.in_queue.pop(oldest_timestamp)})
				del self.counter[oldest_timestamp]

	def pop(self):
		if len(self):
			return self.out_queue.popleft()

	def flush(self):
		while self.in_queue:
			oldest_timestamp = min(self.counter.keys())
			del self.counter[oldest_timestamp]
			self.out_queue.append({'timestamp': oldest_timestamp, 'features': self.in_queue.pop(oldest_timestamp)})

	def _create_buffer(self):
		return {channel: [] for channel in self.channels}

	def __len__(self):
		return len(self.out_queue)

#----------------------------------
### structures to generate basis waveforms

class HalfSineGaussianGenerator(object):
	"""
	Generates half sine gaussian templates based on a f, Q range and a sampling frequency.
	"""
	def __init__(self, f_range, q_range, rates, mismatch=0.2, tolerance=5e-3, downsample_factor=0.8):

		### define parameter range
		self.f_low, self.f_high = (downsample_factor * f_range[0], downsample_factor * f_range[1])
		self.q_low, self.q_high = q_range
		self.phases = [0., numpy.pi/2.]

		### define grid spacing and edge rolloff for templates
		self.mismatch = mismatch
		self.tol = tolerance

		### determine how to scale down templates considered
		### in frequency band to deal with low pass rolloff
		self.downsample_factor = downsample_factor

		### determine frequency bands considered
		self.rates = sorted(set(rates))

		### determine lower frequency limits for rate conversion based on downsampling factor
		self.freq_breakpoints = [self.downsample_factor * rate / 2. for rate in self.rates[:-1]]

		### define grid and template duration
		self.parameter_grid = defaultdict(list)
		for rate, f, q in self.generate_f_q_grid(self.f_low, self.f_high, self.q_low, self.q_high):
			self.parameter_grid[rate].append((f, q, self.duration(f, q)))

		self.sample_pts = {rate: self.round_to_next_odd(max(duration for f, q, duration in parameters) * rate) for rate, parameters in self.parameter_grid.items()}

		### determine timing properties
		self.times = {rate: numpy.linspace(-float(self.sample_pts[rate] - 1) / rate, 0, self.sample_pts[rate], endpoint=True) for rate in self.rates}
		self.latency = {rate: 0 for rate in self.rates}
		self.filter_duration = {rate: (self.times[rate][-1] - self.times[rate][0]) for rate in self.rates}

	def frequency_to_rate(self, frequency):
		"""
		Maps a frequency to the correct sampling rate considered.
		"""
		idx = bisect.bisect(self.freq_breakpoints, frequency)
		return self.rates[idx]

	def round_to_next_odd(self, n):
			return int(numpy.ceil(n) // 2 * 2 + 1)

	def generate_templates(self, rate, quadrature = True):
		"""
		generate all half sine gaussian templates corresponding to a parameter range and template duration
		for a given sampling rate.
		"""
		for f, q, _ in self.parameter_grid[rate]:
			if quadrature:
				for phase in self.phases:
					yield self.waveform(f, q, phase, rate)
			else:
				yield self.waveform(f, q, self.phases[0], rate)

	def duration(self, f, q):
		"""
		return the duration of a half sine-gaussian waveform such that its edges will die out to tolerance of the peak.
		"""
		return 0.5 * (q/(2.*numpy.pi*f)) * numpy.log(1./self.tol)
	
	def waveform(self, f, q, phase, rate):
		"""
		construct half sine gaussian waveforms that taper to tolerance at edges of window
		f is the central frequency of the waveform
		"""
		assert f < rate/2. 
	
		# phi is the central frequency of the sine gaussian
		tau = q/(2.*numpy.pi*f)
		template = numpy.cos(2.*numpy.pi*f*self.times[rate] + phase)*numpy.exp(-1.*self.times[rate]**2./tau**2.)
	
		# normalize sine gaussians to have unit length in their vector space
		inner_product = numpy.sum(template*template)
		norm_factor = 1./(inner_product**0.5)
	
		return norm_factor * template
	
	def num_q_templates(self, q_min, q_max):
		"""
		Minimum number of distinct Q values to generate based on Q_min, Q_max, and mismatch params.
		"""
		return int(numpy.ceil(1./(2.*numpy.sqrt(self.mismatch/3.))*(1./numpy.sqrt(2))*numpy.log(q_max/q_min)))
	
	def num_f_templates(self, f_min, f_max, q):
		"""
		Minimum number of distinct frequency values to generate based on f_min, f_max, and mismatch params.
		"""
		return int(numpy.ceil(1./(2.*numpy.sqrt(self.mismatch/3.))*(numpy.sqrt(2.+q**2.)/2.)*numpy.log(f_max/f_min)))
	
	def generate_q_values(self, q_min, q_max):
		"""
		List of Q values to generate based on Q_min, Q_max, and mismatch params.
		"""
		num_q = self.num_q_templates(q_min, q_max)
		return [q_min*(q_max/q_min)**((0.5+q)/num_q) for q in range(num_q)]
	
	def generate_f_q_grid(self, f_min, f_max, q_min, q_max):
		"""
		Generates (f_samp, f, Q) pairs based on f, Q ranges
		"""
		for q in self.generate_q_values(q_min, q_max):
			num_f = self.num_f_templates(f_min, f_max, q)
			for l in range(num_f):
				f = f_min * (f_max/f_min)**( (0.5+l) /num_f)
				rate = self.frequency_to_rate(f)
				if f < (rate/2) / (1 + (numpy.sqrt(11)/q)):
					yield rate, f, q
				elif rate != max(self.rates):
					yield (2 * rate), f, q

class SineGaussianGenerator(HalfSineGaussianGenerator):
	"""
	Generates sine gaussian templates based on a f, Q range and a sampling frequency.
	"""
	def __init__(self, f_range, q_range, rates, mismatch=0.2, tolerance=5e-3, downsample_factor=0.8):
		super(SineGaussianGenerator, self).__init__(f_range, q_range, rates, mismatch=mismatch, tolerance=tolerance, downsample_factor=0.8)
		self.times = {rate: numpy.linspace(-((self.sample_pts[rate] - 1)  / 2.) / rate, ((self.sample_pts[rate] - 1)  / 2.) / rate, self.sample_pts[rate], endpoint=True) for rate in self.rates}
		self.latency = {rate: int((self.sample_pts[rate] - 1)  / 2) for rate in self.rates}
		self.filter_duration = {rate: (self.times[rate][-1] - self.times[rate][0]) for rate in self.rates}

	def duration(self, f, q):
		"""
		return the duration of a sine-gaussian waveform such that its edges will die out to tolerance of the peak.
		"""
		return 2 * super(SineGaussianGenerator, self).duration(f, q)
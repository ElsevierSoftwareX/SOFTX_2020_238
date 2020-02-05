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


from collections import Counter, defaultdict, deque
import glob
import itertools
import logging
import operator
import os
import sys
import timeit

import h5py
import numpy

from lal import gpstime
from lal.utils import CacheEntry

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

def create_new_dataset(path, base, data, name = 'data', group = None, tmp = False, metadata = None):
	"""
	A function to create a new dataset with data @param data.
	The data will be stored in an hdf5 file at path @param path with
	base name @param base.  You can also make a temporary file.
	If specified, will also save metadata given as key value pairs.

	Returns the filename where the dataset was created.
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

		# set global metadata if specified
		if metadata and not hfile.attrs:
			for key, value in metadata.items():
				hfile.attrs.create(key, value)

		# create dataset
		if group:
			if group not in hfile:
				hfile.create_group(group)
			hfile[group].create_dataset(name, data=data)
		else:
			hfile.create_dataset(name, data=data)

	return fname

def feature_dtype(columns):
	"""
	given a set of columns, returns back numpy dtypes associated with those
	columns. All time-based columns are double-precision, others are stored
	in single-precision.
	"""
	return [(column, numpy.float64) if 'time' in column else (column, numpy.float32) for column in columns]

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
	from the nearest value n.

	>>> floor_div(163, 10)
	160
	>>> floor_div(158, 10)
	150

	"""
	assert n > 0
	return (x // n) * n

def gps2latency(gps_time):
	"""
	Given a gps time, measures the latency to ms precision relative to now.
	"""
	current_gps_time = float(gpstime.tconvert('now')) + (timeit.default_timer() % 1)
	return round(current_gps_time - gps_time, 3)

#----------------------------------
### pathname utilities

def to_trigger_path(rootdir, basename, start_time, job_id=None, subset_id=None):
	"""
	Given a basepath, instrument, description, start_time, will return a
	path pointing to a directory structure in the form::

		${rootdir}/${basename}/${basename}-${start_time_mod1e5}/

    and if the optional job_id, subset_id kwargs are given, the path will be of the form::

		${rootdir}/${basename}/${basename}-${start_time_mod1e5}/${basename}-${job_id}-${subset_id}/
	"""
	start_time_mod1e5 = str(start_time).zfill(10)[:5]
	if job_id and subset_id:
		trigger_path = os.path.join(rootdir, basename, '-'.join([basename, start_time_mod1e5]), '-'.join([basename, job_id, subset_id]))
	else:
		trigger_path = os.path.join(rootdir, basename, '-'.join([basename, start_time_mod1e5]))
	return trigger_path

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

def get_logger(logname, verbose=False):
	"""
	standardize how we instantiate loggers
	"""
	logger = logging.getLogger(logname)
	logger.setLevel(logging.DEBUG if verbose else logging.INFO)

	# set up handler for stdout
	handlers = [logging.StreamHandler()]

	# add handlers to logger
	formatter = logging.Formatter('%(asctime)s | %(name)s : %(levelname)s : %(message)s')
	for handler in handlers:
		handler.setFormatter(formatter)
		logger.addHandler(handler)

	return logger

#----------------------------------
### cache utilities

def path2cache(rootdir, pathname):
	"""
	given a rootdir and a glob-compatible pathname that may contain shell-style wildcards,
	will find all files that match and populate a Cache.
	NOTE: this will only work with files that comply with the T050017 file convention.
	"""
	return [CacheEntry.from_T050017(file_) for file_ in glob.iglob(os.path.join(rootdir, pathname))]

#----------------------------------
### other utilities

def group_indices(indices):
	"""
	Given a list of indices, groups up indices into contiguous groups.
	"""
	for k, group in itertools.groupby(enumerate(indices), lambda (i,x):i-x):
		yield map(operator.itemgetter(1), group)


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
		self.cadence = kwargs['cadence']
		self.sample_rate = kwargs['sample_rate']
		self.waveform = kwargs['waveform']
		self.metadata = dict(**kwargs)
		self.dtype = feature_dtype(self.columns)
		self.feature_data = {key: numpy.empty((self.cadence * self.sample_rate,), dtype = self.dtype) for key in keys}
		self.last_save_time = 0
		self.clear()

	def dump(self, path, base, start_time, tmp = False):
		"""
		Saves the current cadence of features to disk and clear out data
		"""
		for key in self.keys:
			nonnan_indices = list(numpy.where(numpy.isfinite(self.feature_data[key]['time']))[0])

			### split up and save datasets into contiguous segments
			for idx_group in group_indices(nonnan_indices):
				start_idx, end_idx = idx_group[0], idx_group[-1]
				start = start_time + float(start_idx) / self.sample_rate
				end = start_time + float(end_idx + 1) / self.sample_rate
				name = "%.6f_%.6f" % (float(start), float(end - start))
				create_new_dataset(path, base, self.feature_data[key][start_idx:end_idx], name=name, group=key, tmp=tmp, metadata=self.metadata)

		### clear out current features
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
		self.cadence = kwargs['cadence']
		self.waveform = kwargs['waveform']
		self.metadata = dict(**kwargs)
		self.dtype = feature_dtype(self.columns)
		self.feature_data = {key: [] for key in keys}
		self.clear()

	def dump(self, path, base, start_time, tmp = False):
		"""
		Saves the current cadence of gps triggers to disk and clear out data
		"""
		name = "%d_%d" % (start_time, self.cadence)
		for key in self.keys:
			create_new_dataset(path, base, numpy.array(self.feature_data[key], dtype=self.dtype), name=name, group=key, tmp=tmp, metadata=self.metadata)
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
		>>> columns = ['time', 'snr']
		>>> channels = ['channel1']
		>>> queue = TimeseriesFeatureQueue(channels, columns, sample_rate=1, buffer_size=1)
		>>> # add features
		>>> queue.append(123450, 'channel1', {'time': 123450.3, 'snr': 3.0})
		>>> queue.append(123451, 'channel1', {'time': 123451.7, 'snr': 6.5})
		>>> queue.append(123452, 'channel1', {'time': 123452.4, 'snr': 5.2})
		>>> # get oldest feature
		>>> row = queue.pop()
		>>> row['timestamp']
		123450
		>>> row['features']['channel1']
		[{'snr': 3.0, 'time': 123450.3}]

	"""
	def __init__(self, channels, columns, **kwargs):
		self.channels = channels
		self.columns = columns
		self.sample_rate = kwargs.pop('sample_rate')
		self.buffer_size = kwargs.pop('buffer_size')
		self.out_queue = deque(maxlen = 5)
		self.in_queue = {}
		self.counter = Counter()
		self.last_timestamp = 0
		self.effective_latency = 2 # NOTE: set so that late features are not dropped

	def append(self, timestamp, channel, row):
		if timestamp > self.last_timestamp:
			### create new buffer if one isn't available for new timestamp
			if timestamp not in self.in_queue:
				self.in_queue[timestamp] = self._create_buffer()
			self.counter[timestamp] += 1

			### store row, aggregating if necessary
			idx = self._idx(row['time'])
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
		return defaultdict(lambda: [None for ii in range(int(self.sample_rate * self.buffer_size))])

	def _idx(self, timestamp):
		return int(numpy.floor(((timestamp / self.buffer_size) % 1) * self.buffer_size *self.sample_rate))

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
		return defaultdict(list)

	def __len__(self):
		return len(self.out_queue)

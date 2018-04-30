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
	"""
	return (new_gps_time - floor_div(prev_gps_time, gps_epoch)) >= gps_epoch

def floor_div(x, n):
	"""
	Floor an integer by removing its remainder
	from integer division by another integer n.
	e.g. floor_div(163, 10) = 160
	e.g. floor_div(158, 10) = 150
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
	start_time_mod1e5 = str(start_time)[:5]
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

class HDF5FeatureData(FeatureData):
	"""
	Saves feature data to hdf5.
	"""
	def __init__(self, columns, keys, **kwargs):
		super(HDF5FeatureData, self).__init__(columns, keys = keys, **kwargs)
		self.cadence = kwargs.pop('cadence')
		self.feature_data = {key: numpy.empty((self.cadence,), dtype = [(column, 'f8') for column in self.columns]) for key in keys}
		self.clear()

	def dump(self, path, base, start_time, key = None, tmp = False):
		"""
		Saves the current cadence of gps triggers to disk and clear out data
		"""
		name = "%d_%d" % (start_time, self.cadence)
		if key:
			group = os.path.join(str(key[0]), str(key[1]).zfill(4))
			create_new_dataset(path, base, self.feature_data[key], name=name, group=group, tmp=tmp)
			self.clear(key)
		else:
			for key in self.keys:
				group = os.path.join(str(key[0]), str(key[1]).zfill(4))
				create_new_dataset(path, base, self.feature_data[key], name=name, group=group, tmp=tmp)
			self.clear()

	def append(self, value, key = None, buftime = None):
		"""
		Append a trigger row to data structure
		"""
		if buftime and key:
			idx = buftime % self.cadence
			self.feature_data[key][idx] = numpy.array([value[column] for column in self.columns])

	def clear(self, key = None):
		if key:
			self.feature_data[key][:] = numpy.nan
		else:
			for key in self.keys:
				self.feature_data[key][:] = numpy.nan

class HDF5SeriesFeatureData(FeatureData):
	"""!
	Saves feature data with varying dataset lengths to hdf5.
	"""
	def __init__(self, columns, keys, **kwargs):
		super(HDF5SeriesFeatureData, self).__init__(columns, keys = keys, **kwargs)
		self.cadence = kwargs.pop('cadence')
		self.dtype = [(column, '<f8') for column in self.columns]
		self.feature_data = {key: [] for key in keys}
		self.clear()

	def dump(self, path, base, start_time, key = None, tmp = False):
		"""
		Saves the current cadence of gps triggers to disk and clear out data
		"""
		name = "%d_%d" % (start_time, self.cadence)
		if key:
			group = os.path.join(str(key[0]), str(key[1]).zfill(4))
			create_new_dataset(path, base, numpy.array(self.feature_data[key], dtype=self.dtype), name=name, group=group, tmp=tmp)
			self.clear(key)
		else:
			for key in self.keys:
				group = os.path.join(str(key[0]), str(key[1]).zfill(4))
				create_new_dataset(path, base, numpy.array(self.feature_data[key], dtype=self.dtype), name=name, group=group, tmp=tmp)
			self.clear()

	def append(self, value, key = None, buftime = None):
		"""
		Append a trigger row to data structure
		"""
		if buftime and key:
			self.feature_data[key].append(value)

	def clear(self, key = None):
		if key:
			self.feature_data[key] = []
		else:
			for key in self.keys:
				self.feature_data[key] = []

#----------------------------------
### structures to generate basis waveforms

class HalfSineGaussianGenerator(object):
	"""
	Generates half sine gaussian templates based on a f, Q range and a sampling frequency.
	"""
	def __init__(self, f_range, q_range, f_samp, mismatch=0.2, tolerance=5e-3):
		### define parameter range
		self.f_low, self.f_high = f_range
		self.q_low, self.q_high = q_range
		self.f_samp = f_samp
		self.mismatch = mismatch
		self.tol = tolerance

		### define grid and template duration
		self.parameter_grid = [(f, q, self.duration(f, q)) for f, q in self.generate_f_q_grid(self.f_low, self.f_high, self.q_low, self.q_high)]
		self.max_duration = max(duration for f, q, duration in self.parameter_grid)

		self.phases = [0., numpy.pi/2.]
		self.times = numpy.linspace(-self.max_duration, 0, int(numpy.ceil(self.max_duration*self.f_samp)), endpoint=True)
		self.latency = 0

	def generate_templates(self, quadrature = True):
		"""
		generate all half sine gaussian templates corresponding to a parameter range and template duration
		"""
		for f, q in self.generate_f_q_grid(self.f_low, self.f_high, self.q_low, self.q_high):
			if quadrature:
				for phase in self.phases:
					yield self.waveform(f, q, phase)
			else:
				yield self.waveform(f, q, self.phases[0])

	def duration(self, f, q):
		"""
		return the duration of a half sine-gaussian waveform such that its edges will die out to tolerance of the peak.
		"""
		return 0.5 * (q/(2.*numpy.pi*f)) * numpy.log(1./self.tol)
	
	def waveform(self, f, q, phase):
		"""
		construct half sine gaussian waveforms that taper to tolerance at edges of window
		f is the central frequency of the waveform
		"""
		dt = self.times[1] - self.times[0]
		rate = 1./dt
		assert f < rate/2. 
	
		# phi is the central frequency of the sine gaussian
		tau = q/(2.*numpy.pi*f)
		template = numpy.cos(2.*numpy.pi*f*self.times + phase)*numpy.exp(-1.*self.times**2./tau**2.)
	
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
		Generates (f, Q) pairs based on f, Q ranges
		"""
		for q in self.generate_q_values(q_min, q_max):
			num_f = self.num_f_templates(f_min, f_max, q)
			for l in range(num_f):
				f = f_min * (f_max/f_min)**( (0.5+l) /num_f)
				if f < f_max / (1 + (numpy.sqrt(11)/q)):
					yield (f, q)

class SineGaussianGenerator(HalfSineGaussianGenerator):
	"""
	Generates sine gaussian templates based on a f, Q range and a sampling frequency.
	"""
	def __init__(self, f_range, q_range, f_samp, mismatch=0.2, tolerance=5e-3):
		super(SineGaussianGenerator, self).__init__(f_range, q_range, f_samp, mismatch=mismatch, tolerance=tolerance)
		self.times = numpy.linspace(-self.max_duration/2., self.max_duration/2., int(numpy.ceil(self.max_duration*self.f_samp)), endpoint=True)
		self.latency = self.max_duration / 2.

	def duration(self, f, q):
		"""
		return the duration of a sine-gaussian waveform such that its edges will die out to tolerance of the peak.
		"""
		return 2 * super(SineGaussianGenerator, self).duration(f, q)

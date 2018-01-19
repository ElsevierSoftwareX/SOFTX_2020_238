#!/usr/bin/env python
#
# Copyright (C) 2017  Patrick Godwin
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


import os
import glob
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

def get_dataset(path, base, name = 'data', group = None):
	"""!
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
	"""!
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

def in_new_epoch(new_gps_time, prev_gps_time, gps_epoch):
	"""!
	Returns whether new and old gps times are in different
	epochs.
	"""
	return (new_gps_time - floor_div(prev_gps_time, gps_epoch)) >= gps_epoch

def floor_div(x, n):
	"""!
	Floor an integer by removing its remainder
	from integer division by another integer n.
	e.g. floor_div(163, 10) = 160
	e.g. floor_div(158, 10) = 150
	"""
	assert n > 0
	return (x / n) * n



####################
#
#     classes
#
####################

class FeatureData(object):
	"""!
	Base class for saving feature data.
	Extend for a specific file-based implementation.
	"""
	def __init__(self, columns, keys = None, **kwargs):
		self.keys = keys
		self.columns = columns
		self.etg_data = {}

	def dump(self, path):
		raise NotImplementedError

	def load(self, path):
		raise NotImplementedError

	def pop(self):
		raise NotImplementedError

	def append(self, key, value):
		raise NotImplementedError

	def clear(self):
		raise NotImplementedError

class HDF5FeatureData(FeatureData):
	"""!
	Saves feature data to hdf5.
	"""
	def __init__(self, columns, keys, **kwargs):
		super(HDF5FeatureData, self).__init__(columns, keys = keys, **kwargs)
		self.cadence = kwargs.pop('cadence')
		self.etg_data = {key: numpy.empty((self.cadence,), dtype = [(column, 'f8') for column in self.columns]) for key in keys}
		self.clear()

	def dump(self, path, base, start_time, key = None, tmp = False):
		"""
		Saves the current cadence of gps triggers to disk and clear out data
		"""
		name = "%d_%d" % (start_time, self.cadence)
		if key:
			group = os.path.join(str(key[0]), str(key[1]).zfill(4))
			create_new_dataset(path, base, self.etg_data[key], name=name, group=group, tmp=tmp)
			self.clear(key)
		else:
			for key in self.keys:
				group = os.path.join(str(key[0]), str(key[1]).zfill(4))
				create_new_dataset(path, base, self.etg_data[key], name=name, group=group, tmp=tmp)
			self.clear()

	def load(self, path):
		raise NotImplementedError

	def pop(self):
		raise NotImplementedError

	def append(self, value, key = None, buftime = None):
		"""
		Append a trigger row to data structure
		"""
		if buftime and key:
			idx = buftime % self.cadence
			self.etg_data[key][idx] = numpy.array([value[column] for column in self.columns])

	def clear(self, key = None):
		if key:
			self.etg_data[key][:] = numpy.nan
		else:
			for key in self.keys:
				self.etg_data[key][:] = numpy.nan

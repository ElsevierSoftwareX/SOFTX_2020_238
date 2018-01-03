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

def get_dataset(path, base):
	"""!
	open a dataset at @param path with name @param base and return the data
	"""
	fname = os.path.join(path, "%s.hdf5" % base)
	try:
		f = h5py.File(fname, "r")
		fields = f.keys()
		data = zip(*[f[field] for field in fields])
		f.close()
		d_types = [(str(field), 'f8') for field in fields]
		data = numpy.array(data, dtype=d_types)
		return fname, data
	except IOError:
		return fname, []

def create_new_dataset(path, base, fields, data = None, tmp = False):
	"""!
	A function to create a new dataset with data @param data.
	The data will be stored in an hdf5 file at path @param path with
	base name @param base.  You can also make a temporary file.
	"""
	if tmp:
		fname = os.path.join(path, "%s.hdf5.tmp" % base)
	else:
		# A non temp dataset should not be overwritten
		fname = os.path.join(path, "%s.hdf5" % base)
		if os.path.exists(fname):
			return fname
	# create dir if it doesn't exist
	if not os.path.exists(path):
		aggregator.makedir(path)
	# save data to hdf5
	f = h5py.File(fname, "w")
	for field in fields:
		if data is None:
			f.create_dataset(field, (0,), dtype="f8")
		else:
			f.create_dataset(field, (len(data[field]),), dtype="f8")
			f[field][...] = data[field]

	f.close()
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
		self.cadence = kwargs.pop("cadence")
		self.etg_data = {key: numpy.empty((self.cadence,), dtype = [(column, 'f8') for column in self.columns]) for key in keys}
		for key in keys:
			self.etg_data[key][:] = numpy.nan

	def dump(self, path, base):
		for key in self.keys:
			key_path = os.path.join(path, str(key[0]), str(key[1]).zfill(4))
			create_new_dataset(key_path, base, self.columns, data = self.etg_data[key])
		self.clear()

	def load(self, path):
		raise NotImplementedError

	def pop(self):
		raise NotImplementedError

	def append(self, value, key = None, buftime = None):
		if buftime and key:
			idx = buftime % self.cadence
			self.etg_data[key][idx] = numpy.array([value[column] for column in self.columns])

	def clear(self):
		for key in self.keys:
			self.etg_data[key][:] = numpy.nan

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

import numpy
import pandas

from lal import gpstime
from gstlal import aggregator


####################
# 
#    functions
#
####################

def reduce_data(data, gps_start, gps_end, path, reduce_count, aggregate = 'max', level = 0):
	"""!
	This function does a data reduction recursively as needed
	by powers of 10 where level specifies the power.
	Default minimum is 1 e.g., reduction by 1 order of magnitude.
	"""
	if (reduce_count % (10 ** level) == 0 and level < aggregator.DIRS):
		if level == 0:
			agg_data = aggregate_data(data, gps_start, gps_end, aggregate = aggregate)
		else:
			agg_data = aggregate_data(get_dataset_by_range(gps_start, gps_end, path, aggregate = aggregate, level = level - 1), gps_start, gps_end, level = level)
			path = update_agg_path(path, gps_start, cur_level = level - 1, new_level = level)
		if agg_data is not None:
			create_new_dataset(path, 'aggregates', agg_data, aggregate = aggregate)
		reduce_data(data, gps_start, gps_end, path, reduce_count, aggregate = aggregate, level = level + 1)

def aggregate_data(data, gps_start, gps_end, column = 'snr', aggregate = 'max', level = 0):
	"""!
	Reduces data of a given level for a given gps range,
	column, and aggregate. Returns the aggregated data.
	"""
	gps_start_idx = floor_div(gps_start, 10 ** (level+1))
	gps_end_idx = floor_div(gps_end, 10 ** (level+1))
	if aggregate == 'max':
		max_index = get_dataframe_subset(gps_start_idx, gps_end_idx, data, level = level).groupby(pandas.TimeGrouper('%ds' % (10 ** (level+1))))[column].idxmax().dropna().values
	else:
		raise NotImplementedError
	if max_index.size > 0:
		return data.loc[max_index]
	else:
		return None

def get_dataset(path, base, aggregate = None):
	"""!
	open a dataset at @param path with name @param base and return the data
	"""
	fname = os.path.join(path, "%s.h5" % base)
	with pandas.HDFStore(fname) as store:
		if aggregate is None:
			return store.select('data')
		else:
			return store.select(aggregate)

def create_new_dataset(path, base, data, aggregate = None, tmp = False):
	"""!
	A function to create a new dataset with data @param data.
	The data will be stored in an hdf5 file at path @param path with
	base name @param base.  You can also make a temporary file.
	"""
	if not os.path.exists(path):
		aggregator.makedir(path)
	if tmp:
		fname = os.path.join(path, "%s.h5.tmp" % base)
	else:
		fname = os.path.join(path, "%s.h5" % base)
	with pandas.HDFStore(fname) as store:
		if aggregate is None:
			store.append('data', data)
		else:
			store.append(aggregate, data)
	return fname

def get_dataset_by_range(gps_start, gps_end, path, aggregate = None, level = 0):
	"""!
	Returns a dataset for a given aggregation level and gps range.
	"""
	global_start_index = floor_div(gps_start, 10 ** (level+1))
	global_end_index = floor_div(gps_end, 10 ** (level+1))
	gps_epochs = (floor_div(t, aggregator.MIN_TIME_QUANTA * (10 ** level)) for t in range(global_start_index, global_end_index, aggregator.MIN_TIME_QUANTA * (10 ** level)))
	# generate all relevant files and combine to one dataset
	# FIXME: more efficient to filter relevant datasets first
	#        rather than overpopulating dataset and then filtering
	paths = (update_agg_path(path, gps_epoch, cur_level = level, new_level = level) for gps_epoch in gps_epochs)
	if aggregate is None:
		pattern = '[0-9]' * 10 + '.h5'
		filelist = (glob.glob(os.path.join(path, pattern)) for path in paths)
		# flatten list
		files = (file_ for sublist in filelist for file_ in sublist)
	else:
		files = (os.path.join(path, 'aggregates.h5') for path in paths)
	datasets = (pandas.read_hdf(file_, aggregate) for file_ in files)
	return get_dataframe_subset(global_start_index, global_end_index, pandas.concat(datasets), level = level)

def get_dataframe_subset(gps_start, gps_end, dataframe, level = 0):
	start_index = gps_to_index(gps_start)
	end_index = gps_to_index(gps_end - (10 ** level))
	return dataframe.loc[start_index:end_index]

def create_dataframe_index(gps_time):
	"""
	Returns a datetime index based on gps times
	per minimum aggregator quanta.
	"""
	start_time = gps_to_index(floor_div(gps_time, aggregator.MIN_TIME_QUANTA))
	return pandas.date_range(start_time, periods = aggregator.MIN_TIME_QUANTA, freq = 's')

def gps_to_index(gps_time):
	# FIXME: round microseconds is a workaround for possible bug in the
	#        gpstime module in lal that adds a microsecond 
	return pandas.Timestamp(gpstime.gps_to_utc(gps_time).replace(microsecond=0))

def gps_from_index(timestamp):
	return int(gpstime.utc_to_gps(timestamp))

def to_agg_path(base_path, tag, gps_time, channel, rate, level = 0):
	"""
	Returns a hierarchical gps time path based on
	channel rate and level in hierarchy.
	e.g. level 0: base/tag/channel/1/2/3/4/5/6/rate/
	e.g. level 2: base/tag/channel/1/2/3/4/rate/
	"""
	path = os.path.join(base_path, tag)
	if channel is not None:
		path = os.path.join(path, channel)
	path = os.path.join(path, aggregator.gps_to_leaf_directory(gps_time, level = level))
	if rate is not None:
		path = os.path.join(path, str(rate).zfill(4))
	return path

def update_agg_path(path, gps_time, cur_level = 0, new_level = 0):
	"""
	Returns an updated aggregator path based on
	an existing path and a gps time.
	"""
	path, rate = os.path.split(os.path.abspath(path))
	for agg_level in range(max(0, cur_level), aggregator.DIRS):
		path, _ = os.path.split(path)
	return os.path.join(path, os.path.join(aggregator.gps_to_leaf_directory(gps_time, level = new_level), rate))

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

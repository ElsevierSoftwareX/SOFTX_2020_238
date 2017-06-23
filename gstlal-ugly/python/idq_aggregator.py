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

import numpy
import pandas

from gstlal import aggregator


####################
# 
#    functions
#
####################


# FIXME: works for level 1 aggregation only, not tested for other levels
def reduce_data(data, gps_start, gps_end, path, reduce_count, aggregate = 'max', level = 1):
	"""!
	This function does a data reduction recursively as needed
	by powers of 10 where level specifies the power.
	Default minimum is 1 e.g., reduction by 1 order of magnitude.
	"""
	if level == 1 or (reduce_count % (10 ** level) == 0 and level <= aggregator.DIRS):
		if level == 1:
			agg_data = aggregate_data(data, gps_start, gps_end, aggregate = aggregate)
		else:
			agg_data = aggregate_data(get_dataset_by_range(gps_start, gps_end, path, aggregate = aggregate, level = level), gps_start, gps_end, level = level)
		if agg_data is not None:
			create_new_dataset(path, 'aggregates', agg_data, aggregate = aggregate)
		path = update_agg_path(path, gps_start, level = level)
		reduce_data(data, gps_start, gps_end, path, reduce_count, aggregate = aggregate, level = level + 1)

# FIXME: works for level 1 aggregation only, not tested for other levels
def aggregate_data(data, gps_start, gps_end, column = 'snr', aggregate = 'max', level = 1):
	"""!
	Reduces data of a given level for a given gps range,
	column, and aggregate. Returns the aggregated data.
	"""
	gps_start_index = floor_div(gps_start, 10 ** level)
	gps_end_index = floor_div(gps_end, 10 ** level) - (10 ** level)
	if aggregate == 'max':
		max_index = data.loc[gps_start_index:gps_end_index].groupby(level='gps_level%d' % level)[column].idxmax().dropna().values
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

# FIXME: function not currently working yet
def get_dataset_by_range(gps_start, gps_end, path, aggregate = None, level = 0):
	"""!
	Returns a dataset for a given aggregation level and gps range.
	"""
	global_start_index = floor_div(gps_start, 10 ** (level+1))
	global_end_index = floor_div(gps_end, 10 ** (level+1))
	gps_epochs = (floor_div(t, aggregator.MIN_TIME_QUANTA * (10 ** level)) for t in range(global_start_index, global_end_index, aggregator.MIN_TIME_QUANTA * (10 ** level)))
	dataset = pandas.DataFrame()
	for gps_epoch in gps_epochs:
		path = update_agg_path(path, gps_epoch, level = level)
		gps_start_index = max(global_start_index, gps_epoch)
		gps_end_index = min(global_end_index, gps_epoch + aggregator.MIN_TIME_QUANTA * (10 ** level))
		if aggregate is None:
			for gps_index in range(gps_start_index, gps_end_index, 10 ** (level+1)):
				dataset = dataset.append(get_dataset(path, str(gps_index), aggregate = aggregate))
		else:
			dataset = dataset.append(get_dataset(path, 'aggregates', aggregate = aggregate))
	return dataset

def gps_to_index(gps_time, n_levels = 2):
	"""!
	Returns a tuple containing a multilevel gps index.
	"""
	index = [gps_time]
	for level in range(1, n_levels):
		index.append(floor_div(gps_time, 10 ** level))
	return tuple(index[::-1])

def create_dataframe_index(gps_time, n_levels = 2):
	"""
	Returns an n level index based on gps times
	per minimum aggregator quanta.
	"""
	index_t = numpy.arange(floor_div(gps_time, aggregator.MIN_TIME_QUANTA), floor_div(gps_time, aggregator.MIN_TIME_QUANTA) + aggregator.MIN_TIME_QUANTA, dtype = numpy.int)
	indices = [index_t]
	index_names = ['gps_level0']
	for level in range(1, n_levels):
		indices.append(numpy.fromiter(( floor_div(x, 10 ** level) for x in index_t), dtype = numpy.int))
		index_names.append('gps_level%d' % level)
	return pandas.MultiIndex.from_tuples(list(zip(*(indices[::-1]))), names = index_names[::-1])

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

def update_agg_path(path, gps_time, level = 0):
	"""
	Returns an updated aggregator path based on
	an existing path and a gps time.
	"""
	path, rate = os.path.split(path)
	for agg_level in range(aggregator.MIN_TIME_QUANTA):
		path, _ = os.path.split(path)
	return os.path.join(path, aggregator.gps_to_leaf_directory(gps_time, level = level), rate)

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


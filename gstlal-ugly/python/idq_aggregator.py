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


#def reduce_data(data_path, gps_start, reduce_count, reduce_cadence, level = 1):
#	"""!
#	This function does a data reduction recursively as needed 
#	by powers of 10 where level specifies the power.  
#	Default minimum is 1 e.g., reduction by 1 order of magnitude.
#	"""
#	if level == aggregator.DIRS:
#		# reduce
#	elif reduce_count % (2 ** level):
#		# reduce
#		reduce_data(data_path, gps_start, reduce_count, reduce_cadence, level + 1)
#
#def aggregate_data(gps_start, gps_end, level = 0):
#
#
#def get_dataset(path, base):
#	"""!
#	open a dataset at @param path with name @param base and return the data
#	"""
#	fname = os.path.join(path, "%s.h5" % base)

def to_dataframe_index(gps_time):
	"""
	Returns an n level index based on gps times
	per minimum aggregator quanta.
	"""
	index_t = numpy.arange(truncate_int(gps_time, aggregator.MIN_TIME_QUANTA), truncate_int(gps_time, aggregator.MIN_TIME_QUANTA) + aggregator.MIN_TIME_QUANTA, dtype = numpy.int)
	index_cadence = numpy.fromiter(( truncate_int(x, 10) for x in index_t), dtype = numpy.int)
	return pandas.MultiIndex.from_tuples(list(zip(index_cadence, index_t)), names = ['gps_time_cadence', 'gps_time'])

def to_agg_path(base_path, tag, gps_time, channel, rate, level = 0):
	"""
	Returns a hierarchical gps time path based on
	channel rate and level in hierarchy.
	e.g. level 0: base/description/channel/1/2/3/4/5/6/rate/
	e.g. level 2: base/description/channel/1/2/3/4/rate/
	"""
	path = os.path.join(base_path, tag)
	if channel is not None:
		path = os.path.join(path, channel)
	path = os.path.join(path, aggregator.gps_to_leaf_directory(gps_time, level = level))
	if rate is not None:
		path = os.path.join(path, str(rate).zfill(4))
	return path

def truncate_int(x, n):
	"""
	Truncates an integer by removing its remainder
	from integer division by another integer n.
	e.g. truncate_int(163, 10) = 160
	"""
	assert n > 0
	return (x / n) * n


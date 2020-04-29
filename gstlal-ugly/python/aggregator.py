#!/usr/bin/env python
#
# Copyright (C) 2016  Kipp Cannon, Chad Hanna
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


import h5py
import numpy
import sys, os
import itertools
import argparse
import lal
from lal import LIGOTimeGPS
import time
from gi.repository import GLib
import logging
import subprocess
import urllib2
import shutil
import collections
from multiprocessing import Pool

MIN_TIME_QUANTA = 10000
DIRS = 6


#
# =============================================================================
#
#                                 Utility functions
#
# =============================================================================
#


def median(l):
	"""!
	Return the median of a list on nearest value
	"""
	return sorted(l)[len(l)//2]


def now():
	"""!
	A convenience function to return the current gps time
	"""
	return LIGOTimeGPS(lal.UTCToGPS(time.gmtime()), 0)


def get_url(url,d):
	"""!
	A function to pull data from @param url where @param d specifies a
	specific route.  FIXME it assumes that the routes end in .txt
	"""
	f = "%s%s.txt" % (url, d)
	try:
		jobdata = urllib2.urlopen(f).read().split("\n")
	except urllib2.HTTPError as e:
		logging.error("%s : %s" % (f, str(e)))
		return
	except urllib2.URLError as e:
		logging.error("%s : %s" % (f, str(e)))
		return
	data = []
	for line in jobdata:
		if line:
			data.append([float(x) for x in line.split()])
	data = numpy.array(data)
	out = []
	if data.shape != (0,):
		for i in range(data.shape[1]):
			out.append(data[:,i])
	return out


def get_data_from_kafka(jobs, routes, kafka_consumer, req_all = False, timeout = 300):
	"""!
	A function to pull data from kafka for a set of jobs (topics) and
	routes (keys in the incoming json messages)
	"""
	timedata = dict(((j,r), []) for j in jobs for r in routes)
	datadata = dict(((j,r), []) for j in jobs for r in routes)
	# pick a time to bail out of this loop
	bailout_time = now()
	# FIXME this will block forever if it runs out of events unless
	# consumer_timeout_ms (int) is set.  Should we let it block?
	maxt = 0
	for message in kafka_consumer:
		if message.topic not in jobs:
			continue
		for route in routes:
			for x in message.value[route].split("\n"):
				try:
					t,d = [float(y) for y in x.split()]
					timedata[(message.topic, route)].append(t)
					datadata[(message.topic, route)].append(d)
					maxt = max(maxt, t)
				except ValueError as e:
					logging.info("couldn't process %s for %s: %s" % (route, message.topic, e))
		if maxt > bailout_time and (not req_all or all(timedata.values()) or now() > bailout_time + timeout):
			break
	for k in timedata:
		timedata[k] = numpy.array(timedata[k])
		datadata[k] = numpy.array(datadata[k])
	return timedata, datadata


def reduce_data(xarr, yarr, func, level = 0):
	"""!
	This function does a data reduction by powers of 10 where level
	specifies the power.  Default is 0 e.g., data reduction over 1 second
	"""
	datadict = collections.OrderedDict()
	assert len(yarr) == len(xarr)
	for x,y in zip(xarr, yarr):
		# reduce to this level
		key = int(x) // (10**level)
		# we want to sort on y not x
		datadict.setdefault(key, []).append((y,x))
	reduced_data = [func(value) for value in datadict.values()]
	reduced_time = [x[1] for x in reduced_data]
	reduced_data = [x[0] for x in reduced_data]
	assert len(reduced_data) == len(reduced_time)
	idx = numpy.argsort(reduced_time)
	
	return list(numpy.array(reduced_time)[idx]), list(numpy.array(reduced_data)[idx])


def makedir(path):
	"""!
	A convenience function to make new directories and trap errors
	"""
	try:
		os.makedirs(path)
	except IOError:
		pass
	except OSError:
		pass


def create_new_dataset(path, base, timedata = None, data = None, tmp = False):
	"""!
	A function to create a new dataset with time @param timedata and data
	@param data.  The data will be stored in an hdf5 file at path @param path with
	base name @param base.  You can also make a temporary file.
	"""
	tmpfname = "/dev/shm/%s_%s" % (path.replace("/","_"), "%s.hdf5.tmp" % base)
	fname = os.path.join(path, "%s.hdf5" % base)
	if not tmp and os.path.exists(fname):
		return tmpfname, fname
	f = h5py.File(tmpfname if tmp else fname, "w")
	if timedata is None and data is None:
		f.create_dataset("time", (0,), dtype="f8")
		f.create_dataset("data", (0,), dtype="f8")
	else:
		if len(timedata) != len(data):
			raise ValueError("time data %d data %d" % (len(timedata), len(data)))
		f.create_dataset("time", (len(timedata),), dtype="f8")
		f.create_dataset("data", (len(data),), dtype="f8")
		f["time"][...] = timedata
		f["data"][...] = data

	f.close()
	return tmpfname, fname


def get_dataset(path, base):
	"""!
	open a dataset at @param path with name @param base and return the data
	"""
	fname = os.path.join(path, "%s.hdf5" % base)
	try:
		f = h5py.File(fname, "r")
		x,y = numpy.array(f["time"]), numpy.array(f["data"])
		f.close()
		return fname, x,y
	except IOError:
		tmpfname, fname = create_new_dataset(path, base, timedata = None, data = None, tmp = False)
		return fname, numpy.array([]), numpy.array([])


def gps_to_minimum_time_quanta(gpstime):
	"""!
	given a gps time return the minimum time quanta, e.g., 123456789 ->
	123456000.
	"""
	return int(gpstime) // MIN_TIME_QUANTA * MIN_TIME_QUANTA


def gps_to_leaf_directory(gpstime, level = 0):
	"""!
	get the leaf directory for a given gps time
	"""
	return "/".join(str(gps_to_minimum_time_quanta(gpstime) // MIN_TIME_QUANTA // (10**level)))


def gps_to_sub_directories(gpstime, level, basedir):
	"""!
	return the entire relevant directory structure for a given GPS time
	"""
	root = os.path.join(basedir, gps_to_leaf_directory(gpstime, level))
	out = []
	for i in range(10):
		path = os.path.join(root,str(i))
		if os.path.exists(path):
			out.append(str(i))
	return out


def setup_dir_by_job_and_level(gpstime, typ, job, route, base_dir, verbose = True, level = 0):
	"""!
	Given a gps time, the number of jobs and data types produce an
	appropriate data structure for storing the hierarchical data.
	"""
	str_time = str(gpstime).split(".")[0]
	str_time = str_time[:(len(str_time)-int(numpy.log10(MIN_TIME_QUANTA))-level)]
	directory = "%s/%s/by_job/%s/%s" % (base_dir, "/".join(str_time), job, typ) 
	makedir(directory)
	tmpfname, fname = create_new_dataset(directory, route)


def setup_dir_across_job_by_level(gpstime, typ, route, base_dir, verbose = True, level = 0):
	"""!
	Given a gps time, the number of jobs and data types produce an
	appropriate data structure for storing the hierarchical data.
	"""
	str_time = str(gpstime).split(".")[0]
	str_time = str_time[:(len(str_time)-int(numpy.log10(MIN_TIME_QUANTA))-level)]
	directory = "%s/%s/%s" % (base_dir, "/".join(str_time), typ) 
	makedir(directory)
	tmpfname, fname = create_new_dataset(directory, route)


def gps_range(jobtime):
	gpsblocks = set((gps_to_minimum_time_quanta(t) for t in jobtime))
	if not gpsblocks:
		return [], []
	min_t, max_t = min(gpsblocks), max(gpsblocks)
	return range(min_t, max_t+MIN_TIME_QUANTA, MIN_TIME_QUANTA), range(min_t+MIN_TIME_QUANTA, max_t+2*MIN_TIME_QUANTA, MIN_TIME_QUANTA)


def update_lowest_level_data_by_job_type_and_route(job, route, start, end, typ, base_dir, jobtime, jobdata, func):
	path = "/".join([base_dir, gps_to_leaf_directory(start), "by_job", job, typ])
	try:
		fname, prev_times, prev_data = get_dataset(path, route)
	except:
		setup_dir_by_job_and_level(start, typ, job, route, base_dir, verbose = True, level = 0)
		fname, prev_times, prev_data = get_dataset(path, route)
	# only get new data and assume that everything is time ordered
	if prev_times.size:
		this_time_ix = numpy.logical_and(jobtime > max(start-1e-16, prev_times[-1]), jobtime < end)
	else:
		this_time_ix = numpy.logical_and(jobtime >= start, jobtime < end)
	this_time = numpy.concatenate((jobtime[this_time_ix], prev_times))
	this_data = numpy.concatenate((jobdata[this_time_ix], prev_data))
	# shortcut if there are no updates
	if len(this_time) == len(prev_times) and len(this_data) == len(prev_data):
		return []
	reduced_time, reduced_data = reduce_data(this_time, this_data, func, level = 0)
	#logging.info("processing job %s for data %s in span [%d,%d] of type %s: found %d" % (job, route, start, end, typ, len(reduced_time)))
	tmpfname, _ = create_new_dataset(path, route, reduced_time, reduced_data, tmp = True)
	# copy the tmp file over the original
	shutil.move(tmpfname, fname)
	return [start, end]


def job_expanse(dataspan):
	if dataspan:
		min_t, max_t = min(dataspan), max(dataspan)
		return range(min_t, max_t+MIN_TIME_QUANTA, MIN_TIME_QUANTA), range(min_t+MIN_TIME_QUANTA, max_t+2*MIN_TIME_QUANTA, MIN_TIME_QUANTA)
	else:
		return [], []

def reduce_data_from_lower_level_by_job_type_and_route(level, base_dir, job, typ, route, func, start, end):

	this_level_dir = "/".join([base_dir, gps_to_leaf_directory(start, level = level)])

	agg_data = numpy.array([])
	agg_time = numpy.array([])

	# FIXME iterate over levels instead.
	for subdir in gps_to_sub_directories(start, level, base_dir):
		path = "/".join([this_level_dir, subdir, "by_job", job, typ])
		try:
			fname, x, y = get_dataset(path, route)
			agg_time = numpy.concatenate((agg_time, x))
			agg_data = numpy.concatenate((agg_data, y))
		except IOError as ioerr:
			makedir(path)
			# make an empty data set
			create_new_dataset(path, route)
			pass
	reduced_time, reduced_data = reduce_data(agg_time, agg_data, func, level=level)
	path = "/".join([this_level_dir, "by_job", job, typ])
	makedir(path)
	#logging.info("processing reduced data %s for job %s  in span [%d,%d] of type %s at level %d: found %d/%d" % (d, job, s, e, typ, level, len(reduced_time), len(agg_time)))
	tmpfname, fname = create_new_dataset(path, route, reduced_time, reduced_data, tmp = True)
	# FIXME don't assume we can get the non temp file name this way
	shutil.move(tmpfname, fname)


def reduce_across_jobs(jobs, this_level_dir, typ, route, func, level, start, end):
	# Process this level
	agg_data = numpy.array([])
	agg_time = numpy.array([])
	for job  in sorted(jobs):
		path = "/".join([this_level_dir, "by_job", job, typ])
		try:
			fname, x, y = get_dataset(path, route)
			agg_time = numpy.concatenate((agg_time, x))
			agg_data = numpy.concatenate((agg_data, y))
		except IOError as ioerr:
			makedir(path)
			create_new_dataset(path, route)
			pass
	reduced_time, reduced_data = reduce_data(agg_time, agg_data, func, level=level)
	#logging.info("processing reduced data %s in span [%d,%d] of type %s at level %d: found %d/%d" % (route, start, end, typ, level, len(reduced_time), len(agg_time)))
	path = "/".join([this_level_dir, typ])
	tmpfname, fname = create_new_dataset(path, route, reduced_time, reduced_data, tmp = True)
	# FIXME don't assume we can get the non temp file name this way
	shutil.move(tmpfname, fname)


def get_data_from_job_and_reduce(job, job_tag, routes, datatypes, prevdataspan, base_dir, jobs, timedata, datadata):
	# get the url
	with open(os.path.join(job_tag, "%s_registry.txt" % job)) as f:
		url = f.readline().strip()
	update_time = []
	reduce_time = []
	dataspan = set()
	for route in routes:
		#logging.info("processing job %s for route %s" % (job, route))
		# FIXME assumes always two columns
		if timedata is None:
			full_data = get_url(url, route)
			if full_data:
				jobtime, jobdata = full_data[0], full_data[1]
			else:
				jobtime, jobdata = [], []
		else:
			jobtime, jobdata = timedata[(job,route)], datadata[(job,route)]
		gps1, gps2 = gps_range(jobtime)
		for start, end in zip(gps1, gps2):
			# shortcut to not reprocess data that has already been
			# processed.  Dataspan was the only thing that was
			# previously determined to be needing to be updated
			# anything before that is pointless
			if prevdataspan and end < min(prevdataspan):
				#logging.info("no new data for job %s and route %s" % (job, route))
				continue
			for (typ, func) in datatypes:
				now = time.time()
				for processed_time in update_lowest_level_data_by_job_type_and_route(job, route, start, end, typ, base_dir, jobtime, jobdata, func):
					dataspan.add(processed_time)
				update_time.append(time.time()-now)
				# descend down through the levels
				now = time.time()
				for level in range(1,DIRS):
					reduce_data_from_lower_level_by_job_type_and_route(level, base_dir, job, typ, route, func, start, end)
				reduce_time.append(time.time()-now)
	#print "Updated %d with average time %f; Reduced %d with average time %f" % (len(update_time), numpy.mean(update_time), len(reduce_time), numpy.mean(reduce_time))
	return dataspan

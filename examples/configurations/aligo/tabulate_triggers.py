#!/usr/bin/env python
"""Gather all .sqlite files in a directory together and plot ROC curve."""
__author__ = "Leo Singer <leo.singer@ligo.org>"


from pylal.low_latency_clustering import clustered


try:
	import sqlite3
except ImportError:
        # pre 2.5.x
	from pysqlite2 import dbapi2 as sqlite3

try:
	from heapq import merge
except:
	# pre 2.6

	from heapq import heapify, heappop, heapreplace

	# From Python 2.6 standard library, which is
	# Copyright (c) 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010
	# Python Software Foundation; All Rights Reserved

	def merge(*iterables):
		'''Merge multiple sorted inputs into a single sorted output.

		Similar to sorted(itertools.chain(*iterables)) but returns a generator,
		does not pull the data into memory all at once, and assumes that each of
		the input streams is already sorted (smallest to largest).

		>>> list(merge([1,3,5,7], [0,2,4,8], [5,10,15,20], [], [25]))
		[0, 1, 2, 3, 4, 5, 5, 7, 8, 10, 15, 20, 25]

		'''
		_heappop, _heapreplace, _StopIteration = heappop, heapreplace, StopIteration

		h = []
		h_append = h.append
		for itnum, it in enumerate(map(iter, iterables)):
			try:
				next = it.next
				h_append([next(), itnum, next])
			except _StopIteration:
				pass
		heapify(h)

		while 1:
			try:
				while 1:
					v, itnum, next = s = h[0]   # raises IndexError when h is empty
					yield v
					s[0] = next()               # raises StopIteration when exhausted
					_heapreplace(h, s)          # restore heap condition
			except _StopIteration:
				_heappop(h)                     # remove empty iterator
			except IndexError:
				return

try:
	from itertools import chain
	chain_from_iterable = chain.from_iterable
except:
	# Pre-Python 2.6 compatibility.
	def chain_from_iterable(it):
		for it in iterables:
			for element in it:
				yield element

try:
	from collections import namedtuple
except:
	# Pre-Python 2.6 compatibility.
	from gstlal.namedtuple import namedtuple

import os.path
from bisect import bisect_left, bisect_right, insort
from glob import glob
from itertools import groupby, takewhile
from progress import ProgressBar
from optparse import Option, OptionParser
from glue.ligolw import utils, table, lsctables


opts, args = OptionParser(
	description = __doc__,
	usage = "%prog [options] DIRECTORY"
).parse_args()


try:
	dir = args[0]
except IndexError:
	raise ValueError("No directory provided")

filenames = glob(os.path.join(dir, '*.xml.sqlite'))

seen_times = {}
injections = []
noninjections = []


progress = ProgressBar()
progress.max = len(filenames)


DBInfo = namedtuple("DBInfo", "out_start_time out_end_time injections filename svd_bank")

def info_for_db(filename):
	progress.update(text = "examining database %d" % progress.value)
	progress.value += 1

	db = sqlite3.connect(filename)
	try:
		out_start_time, out_end_time = db.execute("""
			SELECT
				CAST(1e9 AS INT) * out_start_time + out_start_time_ns,
				CAST(1e9 AS INT) * out_end_time + out_end_time_ns
			FROM search_summary
		""").fetchone()

		svd_bank, = db.execute("SELECT value FROM process_params WHERE param = '--svd-bank'").fetchone()

		injections = db.execute("SELECT value FROM process_params WHERE param = '--injections'").fetchone()
		if injections is not None:
			injections = injections[0]
	finally:
		db.close()
	return DBInfo(out_start_time=out_start_time, out_end_time=out_end_time, injections=injections, filename=filename, svd_bank=svd_bank)


def iterate_triggers_from_filename(filename, start_time, end_time):
	progress.update()
	progress.value += 1

	return sqlite3.connect(filename).execute("""
		SELECT
			CAST(1e9 AS INT) * end_time + end_time_ns AS end_time_nanos,
			snr
		FROM sngl_inspiral
		WHERE end_time_nanos BETWEEN ? AND ? ORDER BY end_time_nanos
	""", (start_time, end_time))


def chain_triggers_all_times(iterable):
	last_end_time = None
	for dbinfo in sorted(iterable, key = lambda x: x.out_start_time):
		if last_end_time is None:
			last_end_time = dbinfo.out_start_time
		for element in iterate_triggers_from_filename(dbinfo.filename, last_end_time, dbinfo.out_end_time):
			yield element
		last_end_time = dbinfo.out_end_time


def chain_triggers_all_svds(iterable):
	return merge(*[chain_triggers_all_times(list(group)) for svd_bank, group in groupby(sorted(iterable, key = lambda x: x.svd_bank), key = lambda x: x.svd_bank)])

n_total_injections = 0
found_inj_snrs = []
false_snrs = []


for injections, group in groupby(sorted((info_for_db(filename) for filename in filenames), key = lambda x: x.injections), key = lambda x: x.injections):
	group = list(group)
	if injections is None:
		injection_times_distances = None
		progress.text = 'noninjections'
	else:
		progress.update(-1, 'reading injections')
		injection_times_distances = [
			(int(1e9) * sim.h_end_time + sim.h_end_time_ns, sim.distance) for sim in
			table.get_table(
				utils.load_filename(injections),
				lsctables.SimInspiralTable.tableName
			)
		]
		n_total_injections += len(injection_times_distances)
		progress.text = os.path.split(injections)[-1]
	progress.value = 0
	progress.max = len(group)
	last_end_time = 0
	clustered_triggers = [(end_time, snr) for end_time, snr in clustered(chain_triggers_all_svds(group), (lambda x:x[0]), (lambda x:x[1]), int(2e9))]
	if injections is None:
		false_snrs += [snr for end_time, snr in clustered_triggers]
	else:
		for inj_time, inj_dist in injection_times_distances:
			left_index = bisect_left(clustered_triggers, (inj_time,))
			right_index = bisect_right(clustered_triggers, (inj_time,))
			snrs = [clustered_triggers[i][1] for i in range(max(left_index - 1, 0), min(right_index + 1, len(clustered_triggers))) if abs(clustered_triggers[i][0] - inj_time) <= int(2e9)]
			if len(snrs) > 0:
				found_inj_snrs.append((snrs[0], inj_dist))
			else:
				found_inj_snrs.append((0, inj_dist))

import numpy
numpy.save('noninjections.npy', numpy.array(sorted(false_snrs)))
numpy.save('injections.npy', numpy.array(sorted(found_inj_snrs)))

#!/usr/bin/env python3
#
# Copyright (C) 2013--2015  Kipp Cannon
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


import itertools


from glue import iterutils
from ligo import segments
from lal import LIGOTimeGPS


default_timestamp_fuzz = 1e-9	# seconds
default_sample_fuzz = 1e-15	# relative


#
# flags
#


# when comparing time series, require gap intervals to be identical
COMPARE_FLAGS_EXACT_GAPS               = 1
# consider samples that are all 0 also to be gaps
COMPARE_FLAGS_ZERO_IS_GAP              = 2
# don't require the two time series to start and stop at the same time
COMPARE_FLAGS_ALLOW_STARTSTOP_MISALIGN = 4

# the default flags for comparing time series
COMPARE_FLAGS_DEFAULT = 0


#
# tools
#


def load_file(fobj, transients = (0.0, 0.0)):
	stream = (line.strip() for line in fobj)
	stream = (line.split() for line in stream if line and not line.startswith("#"))
	lines = [(LIGOTimeGPS(line[0]),) + tuple(map(float, line[1:])) for line in stream]
	assert lines, "no data"
	channel_count_plus_1 = len(lines[0])
	assert all(len(line) == channel_count_plus_1 for line in lines), "not all lines have the same channel count"
	for t1, t2 in itertools.izip((line[0] for line in lines), (line[0] for line in lines[1:])):
		assert t2 > t1, "timestamps not in order @ t = %s s" % str(t2)
	start = lines[0][0] + transients[0]
	stop = lines[-1][0] - transients[-1]
	iterutils.inplace_filter(lambda line: start <= line[0] <= stop, lines)
	assert lines, "transients remove all data"
	return lines


def max_abs_sample(lines):
	# return the largest of the absolute values of the samples
	return max(max(abs(x) for x in line[1:]) for line in lines)


def identify_gaps(lines, timestamp_fuzz = default_timestamp_fuzz, sample_fuzz = default_sample_fuzz, flags = COMPARE_FLAGS_DEFAULT):
	# assume the smallest interval bewteen samples indicates the true
	# sample rate, and correct for possible round-off by assuming true
	# sample rate is an integer number of Hertz
	dt = min(float(line1[0] - line0[0]) for line0, line1 in itertools.izip(lines, lines[1:]))
	dt = 1.0 / round(1.0 / dt)

	# convert to absolute fuzz (but don't waste time with this if we
	# don't need it)
	if flags & COMPARE_FLAGS_ZERO_IS_GAP:
		sample_fuzz *= max_abs_sample(lines)

	gaps = segments.segmentlist()
	for i, line in enumerate(lines):
		if i and (line[0] - lines[i - 1][0]) - dt > timestamp_fuzz * 2:
			# clock skip.  interpret missing timestamps as a
			# gap
			gaps.append(segments.segment((lines[i - 1][0] + dt, line[0])))
		if flags & COMPARE_FLAGS_ZERO_IS_GAP and all(abs(x) <= sample_fuzz for x in line[1:]):
			# all samples are "0".  the current sample is a gap
			gaps.append(segments.segment((line[0], lines[i + 1][0] if i + 1 < len(lines) else line[0] + dt)))
	return gaps.protract(timestamp_fuzz).coalesce()


def compare_fobjs(fobj1, fobj2, transients = (0.0, 0.0), timestamp_fuzz = default_timestamp_fuzz, sample_fuzz = default_sample_fuzz, flags = COMPARE_FLAGS_DEFAULT):
	timestamp_fuzz = LIGOTimeGPS(timestamp_fuzz)

	# load dump files with transients removed
	lines1 = load_file(fobj1, transients = transients)
	lines2 = load_file(fobj2, transients = transients)
	assert len(lines1[0]) == len(lines2[0]), "files do not have same channel count"

	# trim lead-in and lead-out if requested
	if flags & COMPARE_FLAGS_ALLOW_STARTSTOP_MISALIGN:
		lines1 = [line for line in lines1 if lines2[0][0] <= line[0] <= lines2[-1][0]]
		assert lines1, "time intervals do not overlap"
		lines2 = [line for line in lines2 if lines1[0][0] <= line[0] <= lines1[-1][0]]
		assert lines2, "time intervals do not overlap"

	# construct segment lists indicating gap intervals
	gaps1 = identify_gaps(lines1, timestamp_fuzz = timestamp_fuzz, sample_fuzz = sample_fuzz, flags = flags)
	gaps2 = identify_gaps(lines2, timestamp_fuzz = timestamp_fuzz, sample_fuzz = sample_fuzz, flags = flags)
	if flags & COMPARE_FLAGS_EXACT_GAPS:
		difference = gaps1 ^ gaps2
		iterutils.inplace_filter(lambda seg: abs(seg) > timestamp_fuzz, difference)
		assert not difference, "gap discrepancy: 1 ^ 2 = %s" % str(difference)

	# convert relative sample fuzz to absolute
	sample_fuzz *= max_abs_sample(itertools.chain(lines1, lines2))

	lines1 = iter(lines1)
	lines2 = iter(lines2)
	# guaranteeed to be at least 1 line in both lists
	line1 = lines1.next()
	line2 = lines2.next()
	while True:
		try:
			if abs(line1[0] - line2[0]) <= timestamp_fuzz:
				for val1, val2 in zip(line1[1:], line2[1:]):
					assert abs(val1 - val2) <= sample_fuzz, "values disagree @ t = %s s" % str(line1[0])
				line1 = lines1.next()
				line2 = lines2.next()
			elif line1[0] < line2[0] and line1[0] in gaps2:
				line1 = lines1.next()
			elif line2[0] < line1[0] and line2[0] in gaps1:
				line2 = lines2.next()
			else:
				raise AssertionError("timestamp misalignment @ %s s and %s s" % (str(line1[0]), str(line2[0])))
		except StopIteration:
			break
	# FIXME:  should check that we're at the end of both series


def compare(filename1, filename2, *args, **kwargs):
	try:
		compare_fobjs(open(filename1), open(filename2), *args, **kwargs)
	except AssertionError as e:
		raise AssertionError("%s <--> %s: %s" % (filename1, filename2, str(e)))


#
# main()
#


if __name__ == "__main__":
	from optparse import OptionParser
	parser = OptionParser()
	parser.add_option("--compare-exact-gaps", action = "store_const", const = COMPARE_FLAGS_EXACT_GAPS, default = 0)
	parser.add_option("--compare-zero-is-gap", action = "store_const", const = COMPARE_FLAGS_ZERO_IS_GAP, default = 0)
	parser.add_option("--compare-allow-startstop-misalign", action = "store_const", const = COMPARE_FLAGS_ALLOW_STARTSTOP_MISALIGN, default = 0)
	parser.add_option("--timestamp-fuzz", metavar = "seconds", type = "float", default = default_timestamp_fuzz)
	parser.add_option("--sample-fuzz", metavar = "fraction", type = "float", default = default_sample_fuzz)
	options, (filename1, filename2) = parser.parse_args()
	compare(filename1, filename2, timestamp_fuzz = options.timestamp_fuzz, sample_fuzz = options.sample_fuzz, flags = options.compare_exact_gaps | options.compare_zero_is_gap | options.compare_allow_startstop_misalign)

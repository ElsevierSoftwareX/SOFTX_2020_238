#!/usr/bin/python
"""
Plot latency from progressreport elements in GStreamer pipelines.

The progressreport elements' names must start with the string "progress_".
"""
__author__ = "Leo Singer <leo.singer@ligo.org>"


# Use OptionParser to generate help line, otherwise this is kind of pointless
from optparse import OptionParser
(options, args) = OptionParser(
	usage='%prog filename',
	description='Plot latency from progressreport elements in GStreamer pipelines'
).parse_args()


# If a positional argument is given, try to open the file with that name.
# Otherwise, read from stdin.
if len(args) > 0:
	file = open(args[0], 'r')
else:
	import sys
	file = sys.stdin


# Regex matching "progress_(??:??:??) ????????? seconds"
import re
regex = re.compile(r"^progress_([^ ]+) \((\d\d):(\d\d):(\d\d)\): (\d+) seconds$")


# Read file, keeping only lines that match the regex
trends = {}
for line in file:
	m = regex.match(line)
	if m is not None:
		name, hours, minutes, seconds, stream_time = m.groups()
		if name not in trends.keys():
			trends[name] = []
		trends[name].append((int(hours) * 3600 + int(minutes) * 60 + int(seconds), int(stream_time)))
file.close()


# Plot
import pylab
for k, v in trends.iteritems():
	data = pylab.array(v)
	pylab.plot(data[:,0], data[:,1], label=k)

pylab.grid()
pylab.legend(loc='lower right')
pylab.xlabel('running time (seconds)')
pylab.ylabel('stream time (seconds)')
pylab.show()


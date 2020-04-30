#!/usr/bin/env python3
"""
Plot latency from progressreport elements in GStreamer pipelines.

The progressreport elements' names must start with the string "progress_".
"""
__author__ = "Leo Singer <leo.singer@ligo.org>"


# Use OptionParser to generate help line, otherwise this is kind of pointless
from optparse import OptionParser, Option
(options, args) = OptionParser(
	usage='%prog [options] filename [image filename]',
	description='Plot latency from progressreport elements in GStreamer pipelines',
	option_list=[
		Option('--disable-legend', action='store_true', default=False, help='Disable figure legend.'),
	]
).parse_args()


# If a positional argument is given, try to open the file with that name.
# Otherwise, read from stdin.
if len(args) > 0:
	filename = args[0]
	file = open(filename, 'r')
else:
	import sys
	file = sys.stdin
	filename = '/dev/stdin'

# If a second argument is given, use it as the filename to write to.
if len(args) > 1:
	out_filename = args[1]
else:
	out_filename = None


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


gps_start_time = min(trend[0][1] for trend in trends.values())
gps_end_time = max(trend[-1][1] for trend in trends.values())
gps_duration = gps_end_time - gps_start_time
running_duration = max(trend[-1][0] for trend in trends.values())

# Plot
if out_filename is not None:
	import matplotlib
	matplotlib.use('Agg')
import pylab

# Plot diagonal grid
t0 = max(gps_duration, running_duration)
for t in range(-t0, t0, t0 / 20):
	lines = pylab.plot((0, t0), (t, t+t0), color='#cccccc')
lines[0].set_label('constant lag')

for k, v in sorted(trends.iteritems()):
	data = pylab.array(v)
	pylab.plot(data[:,0], data[:,1] - gps_start_time, label=k)

pylab.xlim((0, running_duration))
pylab.ylim((0, gps_duration))
pylab.gca().set_aspect('equal')
if not options.disable_legend:
	pylab.legend(loc='lower right')
pylab.xlabel('running time (seconds)')
pylab.ylabel('stream time - %d (seconds)' % gps_start_time)
pylab.title('Progress report for %s' % filename)
if out_filename is None:
	pylab.show()
else:
	pylab.savefig(out_filename)

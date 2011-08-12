#!/usr/bin/env python
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
regex = re.compile(r"^progress_([^ ]+) \((\d+):(\d\d):(\d\d)\.(\d\d\d\d\d\d)\): (\d+) nanoseconds$")


# Read file, keeping only lines that match the regex
trends = {}
for line in file:
	m = regex.match(line)
	if m is not None:
		name, hours, minutes, seconds, useconds, stream_time = m.groups()
		if name not in trends.keys():
			trends[name] = []
		trends[name].append(((int(hours) * 3600000000 + int(minutes) * 60000000 + int(seconds) * 1000000 + int(useconds)) * 1e-6, int(stream_time) * 1e-9))
file.close()


gps_start_time = min(min(t[1] for t in trend) for trend in trends.values())
gps_end_time = max(max(t[1] for t in trend) for trend in trends.values())
gps_duration = gps_end_time - gps_start_time
start_time = min(min(t[0] for t in trend) for trend in trends.values())
end_time = max(max(t[0] for t in trend) for trend in trends.values())
duration = end_time - start_time

# Plot
if out_filename is not None:
	import matplotlib
	matplotlib.use('Agg')
import pylab

# Plot diagonal grid
t0 = max(gps_duration, duration)
for t in pylab.arange(-t0, t0, t0 / 20):
	lines = pylab.plot((0, t0), (t, t+t0), color='#cccccc')
lines[0].set_label('constant lag')

for k, v in sorted(trends.iteritems()):
	data = pylab.array(v)
	pylab.plot(data[:,0] - start_time, data[:,1] - gps_start_time, label=k)
	print k, pylab.mean((data[:, 0] - start_time) - (data[:, 1] - gps_start_time))

pylab.xlim((0, duration))
pylab.ylim((0, gps_duration))
pylab.gca().set_aspect('equal')
if not options.disable_legend:
	pylab.legend(loc='lower right')
pylab.xlabel('running time - %d (seconds)' % start_time)
pylab.ylabel('stream time - %d (seconds)' % gps_start_time)
pylab.title('Progress report for %s' % filename)
if out_filename is None:
	pylab.show()
else:
	pylab.savefig(out_filename)


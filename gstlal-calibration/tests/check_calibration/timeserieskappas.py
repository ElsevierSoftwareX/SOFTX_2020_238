import matplotlib as mpl; mpl.use('Agg')
from matplotlib import pyplot as mpl
from gwpy.timeseries import TimeSeries
from gwpy.timeseries import TimeSeriesDict
from gwpy.segments import DataQualityFlag
from optparse import OptionParser, Option

parser = OptionParser()

parser.add_option("--gps-start-time", metavar = "seconds", help = "Set the GPS start time.")
parser.add_option("--gps-end-time", metavar = "seconds", help = "Set the GPS end time.")
parser.add_option("--ifo", metavar = "name", help = "Interferometer to perform the analysis on")
parser.add_option("--frame-cache", metavar = "name", help = "Filename for frame cache to be analyzed.")
parser.add_option("--channel-list", metavar = "list", help = "List of channels to be plotted.")

options, filenames = parser.parse_args()

start = int(options.gps_start_time)
end = int(options.gps_end_time)
ifo = options.ifo

channel_list = []
if options.channel_list is not None:
	channels = options.channel_list.split(',')
	for channel in channels:
		channel_list.append((ifo, channel))
else:
	raise ValueError('Channel list option must be set.')

data = TimeSeriesDict.read(options.frame_cache, channels = map("%s:%s".__mod__, channel_list), start = start, end = end)

segs = DataQualityFlag.query('%s:DMT-CALIBRATED:1' % ifo, start, end)

for n, channel in enumerate(channels):
	plot = TimeSeries.plot(data["%s:%s" % (ifo, channel)])
	mpl.ylabel('Correction value')
	#title = item
	#title = title.replace('_', '\_')
	mpl.title(channel.replace('_', '\_'))
	plot.add_state_segments(segs, plotargs=dict(label='Calibrated'))
	plot.savefig('%s_%s_%s_plot_%s.png' % (ifo, options.gps_start_time, options.gps_end_time, channel))

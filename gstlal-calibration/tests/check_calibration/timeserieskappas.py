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
parser.add_option("--raw-frame-cache", metavar = "name", help = "Filename for raw frame cache to be analyzed.")
parser.add_option("--raw-channel-list", metavar = "list", help = "List of raw channels to be plotted.")

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

raw_channel_list = []
if options.raw_channel_list is not None:
	raw_channels = options.raw_channel_list.split(',')
	for raw_channel in raw_channels:
		raw_channel_list.append((ifo, raw_channel))
else:
	raise ValueError('Raw channel list option must be set.')

raw_data = TimeSeriesDict.read(options.raw_frame_cache, map("%s:%s".__mod__, raw_channel_list), start = start, end = end)
data = TimeSeriesDict.read(options.frame_cache, map("%s:%s".__mod__, channel_list), start = start, end = end)

segs = DataQualityFlag.query('%s:DMT-CALIBRATED:1' % ifo, start, end)

for n in range(0, len(channels)):
	plot = TimeSeries.plot(data["%s:%s" % (ifo, channels[n])], label = 'GDS')
	mpl.ylabel('Correction value')
	ax = plot.gca()
	ax.plot(raw_data["%s:%s" % (ifo, raw_channels[n])], label='CAL-CS')
	ax.legend()
	#title = item
	#title = title.replace('_', '\_')
	mpl.title(channels[n].replace('_', '\_'))
	plot.add_state_segments(segs, plotargs=dict(label='Calibrated'))
	plot.savefig('%s_%s_%s_plot_%s.png' % (ifo, options.gps_start_time, options.gps_end_time, channels[n]))


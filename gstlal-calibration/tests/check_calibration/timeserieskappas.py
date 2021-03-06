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
parser.add_option("--front-end-channel-list", metavar = "list", help = "List of channels to be plotted from the front-end.")
parser.add_option("--analyze-additional-hoft", action = "store_true", help = "Set this to analyze an additional h(t) channel.")
parser.add_option("--additional-hoft-frames-cache", metavar = "name", help = "If desired, provide an additional frame cache for a secondary h(t) data stream to be analyzed.")
parser.add_option("--additional-channel-list", metavar = "list", help = "List of channels to be plotted from additional h(t) frames.")

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

if options.analyze_additional_hoft:
	additional_channel_list = []
	if options.additional_channel_list is not None:
		additional_channels = options.additional_channel_list.split(',')
		for channel in additional_channels:
			additional_channel_list.append((ifo, channel))
		plot_additional_hoft = True
	else:
		plot_additional_hoft = False


front_end_channel_list = []
if options.front_end_channel_list is not None:
	front_end_channels = options.front_end_channel_list.split(',')
	for channel in front_end_channels:
		front_end_channel_list.append((ifo, channel))
	plot_front_end = True
else:
	plot_front_end = False

data = TimeSeriesDict.read(options.frame_cache, list(map("%s:%s".__mod__, channel_list), start = start, end = end))
if plot_front_end:
	front_end_data = TimeSeriesDict.fetch(list(map("%s:%s".__mod__, front_end_channel_list), start = start, end = end))
if plot_additional_hoft:
	additional_hoft_data = TimeSeriesDict.read(options.additional_hoft_frames_cache, list(map("%s:%s".__mod__, additional_channel_list)), start = start, end = end)

print(map("%s:%s".__mod__, front_end_channel_list))

segs = DataQualityFlag.query('%s:DMT-CALIBRATED:1' % ifo, start, end)

for n, channel in enumerate(channels):
	plot = TimeSeries.plot(data["%s:%s" % (ifo, channel)])
	ax = plot.gca()
	if plot_front_end:
		ax.plot(front_end_data["%s:%s" % (ifo, front_end_channels[n])])
	if plot_additional_hoft:
		ax.plot(additional_hoft_data["%s:%s" % (ifo, additional_channels[n])])
	ax.set_ylabel('Correction value')
	plot.gca().legend()
	#title = item
	#title = title.replace('_', '\_')
	ax.set_title(channel.replace('_', '\_'))
	if 'F_S_SQUARED' in channel:
		ax.set_ylim(-100,100)
	elif 'SRC_Q_INVERSE' in channel:
		ax.set_ylim(-2,2)
	#plot.add_state_segments(segs, plotargs=dict(label='Calibrated'))
	#plot.legend([r'GDS value', r'front-end value'])
	plot.savefig('%s_%s_%s_plot_%s.png' % (ifo, options.gps_start_time, options.gps_end_time, channel))

import matplotlib as mpl; mpl.use('Agg')
from matplotlib import pyplot as mpl
from gwpy.timeseries import TimeSeries
from gwpy.segments import DataQualityFlag
from optparse import OptionParser, Option

parser = OptionParser()

parser.add_option("--gps_start_time", metavar = "seconds", help = "Set the GPS start time.")
parser.add_option("--gps_end_time", metavar = "seconds", help = "Set the GPS end time.")
parser.add_option("--channel_name1", metavar = "name", help = "_")
parser.add_option("--channel_name2", metavar = "name", help = "_")
parser.add_option("--channel_name3", metavar = "name", help = "_")
parser.add_option("--channel_name4", metavar = "name", help = "_")
parser.add_option("--channel_name5", metavar = "name", help = "_")
parser.add_option("--channel_name6", metavar = "name", help = "_")

options, filenames = parser.parse_args()

start = int(options.gps_start_time)
end = int(options.gps_end_time)
channel_name1 = str(options.channel_name1)
channel_name2 = str(options.channel_name2)
channel_name3 = str(options.channel_name3)
channel_name4 = str(options.channel_name4)
channel_name5 = str(options.channel_name5)
channel_name6 = str(options.channel_name6)

data1 = TimeSeries.get(channel_name1, start, end)
data2 = TimeSeries.get(channel_name2, start, end)
data3 = TimeSeries.get(channel_name3, start, end)
data4 = TimeSeries.get(channel_name4, start, end)
data5 = TimeSeries.get(channel_name5, start, end)
data6 = TimeSeries.get(channel_name6, start, end)

segs = DataQualityFlag.query('L1:DMT-CALIBRATED:1', start, end)

plot1 = TimeSeries.plot(data1)
mpl.ylabel('Correction value')
title1 = channel_name1
title1 = title1.replace('_', '\_')
mpl.title(title1)
plot1.add_state_segments(segs, plotargs=dict(label='Calibrated'))
plot1.savefig('plot_'+channel_name1+'.png')

plot2 = TimeSeries.plot(data2)
mpl.ylabel('Correction value')
title2 = channel_name2
title2 = title2.replace('_', '\_')
mpl.title(title2)
plot2.add_state_segments(segs, plotargs=dict(label='Calibrated'))
plot2.savefig('plot_'+channel_name2+'.png')

plot3 = TimeSeries.plot(data3)
mpl.ylabel('Correction value')
title3 = channel_name3
title3 = title3.replace('_', '\_')
mpl.title(title3)
plot3.add_state_segments(segs, plotargs=dict(label='Calibrated'))
plot3.savefig('plot_'+channel_name3+'.png')

plot4 = TimeSeries.plot(data4)
mpl.ylabel('Correction value')
title4 = channel_name4
title4 = title4.replace('_', '\_')
mpl.title(title4)
plot4.add_state_segments(segs, plotargs=dict(label='Calibrated'))
plot4.savefig('plot_'+channel_name4+'.png')

plot5 = TimeSeries.plot(data5)
mpl.ylabel('Correction value')
title5 = channel_name5
title5 = title5.replace('_', '\_')
mpl.title(title5)
plot5.add_state_segments(segs, plotargs=dict(label='Calibrated'))
plot5.savefig('plot_'+channel_name5+'.png')

plot6 = TimeSeries.plot(data6)
mpl.ylabel('Correction value')
title6 = channel_name2
title6 = title6.replace('_', '\_')
mpl.title(title6)
plot6.add_state_segments(segs, plotargs=dict(label='Calibrated'))
plot6.savefig('plot_'+channel_name6+'.png')


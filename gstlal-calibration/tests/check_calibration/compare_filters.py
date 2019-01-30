##########################################################
# Create a Bode plot of the response function as derived #
# from different h(t) pipelines				 #
##########################################################

import matplotlib as mpl; mpl.use('Agg')
from gwpy.plotter import BodePlot
from gwpy.timeseries import TimeSeriesDict
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
import numpy
import sys
from optparse import OptionParser, Option

parser = OptionParser()

parser.add_option("--filter-file1", metavar = "name", help = "Set the path to the first filters file to be compared.")
parser.add_option("--filter-file2", metavar = "name", help = "Set the path to the second filters file to be compared.")
parser.add_option("--filter-name", metavar = "name", help = "Name of filter to be plotted.")
parser.add_option("--plot-name", metavar = "name", help = "Name of plot prefix.")
parser.add_option("--dt", metavar = "float", help = "Time spacing of FIR filter.")

options, filenames = parser.parse_args()

filters1 = numpy.load(options.filter_file1)
filters2 = numpy.load(options.filter_file2)

filter1 = filters1[options.filter_name].flatten()
filter2 = filters2[options.filter_name].flatten()

filter1_fd = numpy.fft.rfft(filter1)
filter2_fd = numpy.fft.rfft(filter2)
freqs1 = numpy.fft.rfftfreq(len(filter1), d = float(options.dt))
freqs2 = numpy.fft.rfftfreq(len(filter2), d = float(options.dt))
df1 = freqs1[1]-freqs1[0]
df2 = freqs2[1]-freqs2[0]
f0 = 0
print(len(filter1))
print(len(filter2))

filter1_fs = FrequencySeries(filter1_fd, f0=f0, df=df1) # packagae as a FrequecySeries object
filter2_fs = FrequencySeries(filter2_fd, f0=f0, df=df2) # packagae as a FrequecySeries object

plot = BodePlot(filter1_fs, frequencies=freqs1, dB = False, linewidth=2)
plot.add_frequencyseries(filter2_fs, dB = False, color='#ee0000',linewidth=2)
# FIXME: Figure out how to make the legend and title appropriate and flexible
plot.maxes.set_yscale('log')
plot.paxes.set_yscale("linear")
plot.save('filter_comparison.pdf')

diff = filter1_fs / filter2_fs
plot = BodePlot(diff, frequencies = freqs1, dB = False, linewidth = 2)
plot.maxes.set_yscale('log')
plot.paxes.set_yscale('linear')
plot.maxes.set_ylim(.5, 1.5)
plot.maxes.set_xlim(10, 5000)
plot.paxes.set_ylim(-10, 10)
plot.paxes.set_xlim(10, 5000)
plot.save("filter_difference.pdf")


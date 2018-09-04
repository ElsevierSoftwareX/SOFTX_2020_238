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
import ConfigParser
import sys
from optparse import OptionParser, Option

parser = OptionParser()

parser.add_option("--gps_start_time", metavar = "seconds", help = "Set the GPS start time.")
parser.add_option("--gps_end_time", metavar = "seconds", help = "Set the GPS end time.")
parser.add_option("--dt", metavar = "seconds", help = "Sampling time interval.")
parser.add_option("--ifo", metavar = "name", help = "Name of the IFO")
parser.add_option("--c00_hoft_frames_cache", metavar = "name", help = "_")
parser.add_option("--c01_hoft_frames_cache", metavar = "name", help = "_")
parser.add_option("--raw_frames_cache", metavar = "name", help = "_")
parser.add_option("--darm_err_channel_name", metavar = "name", help = "_")
parser.add_option("--c00_hoft_channel_name", metavar = "name", help = "_")
parser.add_option("--c01_hoft_channel_name", metavar = "name", help = "_")
parser.add_option("--calcs_hoft_channel_name", metavar = "name", help = "_")
parser.add_option("--response_file", metavar = "name", help = "_")

options, filenames = parser.parse_args()

start = int(options.gps_start_time)
end = int(options.gps_end_time)
dt = float(options.dt)
ifo = options.ifo
C00_hoft_frames_cache = options.c00_hoft_frames_cache
C01_hoft_frames_cache = options.c01_hoft_frames_cache
raw_frames_cache = options.raw_frames_cache
DARM_ERR_channel_name = options.darm_err_channel_name
C00_hoft_channel_name = options.c00_hoft_channel_name
C01_hoft_channel_name = options.c01_hoft_channel_name
CALCS_hoft_channel_name = options.calcs_hoft_channel_name
response_file = numpy.load(options.response_file)

#response_real = response_file['response_real_function']
response_real = response_file['response_function'][1]
#response_imag = response_file['response_imaginary_function']
response_imag = response_file['response_function'][2]
# I had to save the real and imaginary parts separately (there might be a way to save the complex number
# into an npz file from Matlab, but I couldn't figure that out)
response = response_real + 1j*response_imag
# This is the same frequency vector as we'll use below for the h(t) data
#freqs = response_file['frequency_vector']
freqs = response_file['response_function'][0]
f0 = freqs[0]
df = freqs[1]-freqs[0]

response[0] = 0 # zero out the DC component
response = numpy.ndarray.flatten(response) # flatten to a 1D array

response_fs = FrequencySeries(response, f0=f0, df=df) # packagae as a FrequecySeries object

# Read in DARM_ERR data and compensate for the model jump delay be reading in data from start+dt to end+dt
#DARM_ERR_data = TimeSeriesDict.read(raw_frames_cache, channels = ["%s:%s" % (ifo, DARM_ERR_channel_name)], start = start+dt, end = end+dt) 
DARM_ERR_data = TimeSeries.read(raw_frames_cache, "%s:%s" % (ifo, DARM_ERR_channel_name), start = start+dt, end = end+dt)
#DARM_ERR_data = TimeSeriesDict.read(raw_frames_cache, "%s:%s" % (ifo, DARM_ERR_channel_name), start+dt, end+dt)
# Read in CALCS data
#CALCS_data = TimeSeriesDict.read(raw_frames_cache, channels = ["%s:%s" % (ifo, CALCS_hoft_channel_name)], start = start, end = end) 
CALCS_data = TimeSeries.read(raw_frames_cache, "%s:%s" % (ifo, CALCS_hoft_channel_name), start = start, end = end)
# Read in the C00 data without kappas applied
#C00_data = TimeSeriesDict.read(C00_hoft_frames_cache, channels = ["%s:%s" % (ifo, C00_hoft_channel_name)], start = start, end = end)
C00_data = TimeSeries.read(C00_hoft_frames_cache, "%s:%s" % (ifo, C00_hoft_channel_name), start = start, end = end)
# Read in the C01 data without kappas applied
#C01_data = TimeSeriesDict.read(C01_hoft_frames_cache, channels = ["%s:%s" % (ifo, C01_hoft_channel_name)], start = start, end = end)
C01_data = TimeSeries.read(C01_hoft_frames_cache, "%s:%s" % (ifo, C01_hoft_channel_name), start = start, end = end)

# Pick out channel from dictionary
#DARM_ERR_data = DARM_ERR_data["%s:%s" % (ifo, DARM_ERR_channel_name)]
#CALCS_data = CALCS_data["%s:%s" % (ifo, CALCS_hoft_channel_name)]
#DARM_ERR_data = DARM_ERR_data["%s:%s" % (ifo, DARM_ERR_channel_name)]

# We want \Delta L, not strain, to find the response function
#C00_data = C00_data["%s:%s" % (ifo, C00_hoft_channel_name)] * 3995.1
#C01_data = C01_data["%s:%s" % (ifo, C01_hoft_channel_name)] * 3995.1

dur = end - start
averaging_time = 16
chunk_start = start
chunk_end = start + averaging_time

CALCS_tf_data = numpy.zeros(len(response)) + 1j*numpy.zeros(len(response))
C00_tf_data = numpy.zeros(len(response)) + 1j*numpy.zeros(len(response))
C01_tf_data = numpy.zeros(len(response)) + 1j*numpy.zeros(len(response))

N = 0

while (chunk_end <= end):
	# Correct for model jump delay in darm_err      
	DARM_ERR_chunk = DARM_ERR_data.crop(chunk_start+dt, chunk_end+dt, True)
	DARM_ERR_chunk = DARM_ERR_chunk.detrend()
	DARM_ERR_chunk_fft = DARM_ERR_chunk.average_fft(4, 2, window = 'hann')

	CALCS_chunk = CALCS_data.crop(chunk_start, chunk_end, True)
	CALCS_chunk = CALCS_chunk.detrend()
	CALCS_chunk_fft = CALCS_chunk.average_fft(4, 2, window = 'hann')
	CALCS_chunk_fft = CALCS_chunk_fft.filter([30]*6, [0.3]*6, 1e-12)

	C00_chunk = C00_data.crop(chunk_start, chunk_end, True)
	C00_chunk = C00_chunk.detrend()
	C00_chunk_fft = C00_chunk.average_fft(4, 2, window = 'hann')

	C01_chunk = C01_data.crop(chunk_start, chunk_end, True)
	C01_chunk = C01_chunk.detrend()
	C01_chunk_fft = C01_chunk.average_fft(4, 2, window = 'hann')

	CALCS_chunk_tf = CALCS_chunk_fft / DARM_ERR_chunk_fft
	C00_chunk_tf = C00_chunk_fft / DARM_ERR_chunk_fft
	C01_chunk_tf = C01_chunk_fft / DARM_ERR_chunk_fft

	CALCS_tf_data += CALCS_chunk_tf.value
	C00_tf_data += C00_chunk_tf.value
	C01_tf_data += C01_chunk_tf.value

	chunk_start += averaging_time
	chunk_end += averaging_time
	N += 1


CALCS_tf_data = CALCS_tf_data / N
CALCS_tf = FrequencySeries(CALCS_tf_data, f0=f0, df=df)
C00_tf_data = C00_tf_data / N
C00_tf = FrequencySeries(C00_tf_data, f0=f0, df=df)
C01_tf_data = C01_tf_data / N
C01_tf = FrequencySeries(C01_tf_data, f0=f0, df=df)

# Make plot that compares all of the derived response functions and the model response
plot = BodePlot(response_fs, frequencies=freqs, dB = False, linewidth=2)
plot.add_frequencyseries(CALCS_tf, dB=False, color='#4ba6ff', linewidth=2)
plot.add_frequencyseries(C00_tf*3995.1, dB = False, color='#ee0000',linewidth=2)
plot.add_frequencyseries(C01_tf*3995.1, dB = False, color="#94ded7", linewidth=2)
plot.add_legend([r'Reference model response function', r'Front-end response function', r'Low-latency \texttt{gstlal} response function', r'High-latency \texttt{gstlal} response function'], loc='upper right', fontsize='x-small')
plot.maxes.set_yscale('log')
plot.paxes.set_yscale("linear")
plot.save('%s_all_tf.pdf' % ifo)

# Make a plot that compares the ratios of each derived response to the model
ratio_CALCS = CALCS_tf / response_fs
ratio_C00 = C00_tf / response_fs
ratio_C01 = C01_tf / response_fs
plot = BodePlot(ratio_CALCS, frequencies = freqs, dB = False, color='#4ba6ff', linewidth=2)
plot.add_frequencyseries(ratio_C00*3995.1, dB = False, color='#ee0000',linewidth=2)
plot.add_frequencyseries(ratio_C01*3995.1, dB = False, color="#94ded7", linewidth=2)
plot.add_legend([r'Front-end response / Reference model response', r'Low-latency \texttt{gstlal} response / Reference model response', r'High-latency \texttt{gstlal} response / Reference model response'], loc='upper right', fontsize='small')
plot.maxes.set_yscale('linear')
plot.paxes.set_yscale('linear')
plot.save('%s_all_tf_ratio.pdf' % ifo)

plot = BodePlot(ratio_CALCS, frequencies = freqs, dB = False, color='#4ba6ff', linewidth=2)
plot.add_frequencyseries(ratio_C00*3995.1, dB = False, color='#ee0000',linewidth=2)
plot.add_frequencyseries(ratio_C01*3995.1, dB = False, color="#94ded7", linewidth=2)
plot.add_legend([r'Front-end response / Reference model response', r'Low-latency \texttt{gstlal} response / Reference model response', r'High-latency \texttt{gstlal} response / Reference model response'], loc='upper right', fontsize='small')
plot.maxes.set_yscale('linear')
plot.paxes.set_yscale('linear')
plot.maxes.set_ylim(.9, 1.1)
plot.maxes.set_xlim(10, 5000)
plot.paxes.set_ylim(-5, 5)
plot.paxes.set_xlim(10, 5000)
plot.save('%s_all_tf_ratio_zoomed.pdf' % ifo)



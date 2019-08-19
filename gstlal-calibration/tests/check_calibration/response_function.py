##########################################################
# Create a Bode plot of the response function as derived #
# from different h(t) pipelines				 #
##########################################################

import matplotlib as mpl; mpl.use('Agg')
from gwpy.plot import BodePlot
from gwpy.timeseries import TimeSeriesDict
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
import numpy
import ConfigParser
import sys
from optparse import OptionParser, Option

parser = OptionParser()

parser.add_option("--gps-start-time", metavar = "seconds", help = "Set the GPS start time.")
parser.add_option("--gps-end-time", metavar = "seconds", help = "Set the GPS end time.")
parser.add_option("--dt", metavar = "seconds", help = "Sampling time interval.")
parser.add_option("--ifo", metavar = "name", help = "Name of the IFO")
parser.add_option("--hoft-frames-cache", metavar = "name", help = "Frame cache file for h(t) data to be analyzed")
parser.add_option("--analyze-additional-hoft", action = "store_true", help = "Set this to analyze an additional h(t) channel.")
parser.add_option("--additional-hoft-frames-cache", metavar = "name", help = "If desired, provide an additional frame cache for a secondary h(t) data stream to be analyzed.")
parser.add_option("--raw-frames-cache", metavar = "name", help = "Frame cache for raw data.")
parser.add_option("--darm-err-channel-name", metavar = "name", default = "CAL-DARM_ERR_DBL_DQ", help = "DARM_ERR channel name (default = CAL-DARM_ERR_DBL_DQ)")
parser.add_option("--hoft-channel-name", metavar = "name", default = "GDS-CALIB_STRAIN", help = "h(t) channel name (default = GDS-CALIB_STRAIN")
parser.add_option("--additional-hoft-channel-name", metavar = "name", help = "Additional h(t) channel name, if provided")
parser.add_option("--analyze-calcs-hoft", action = "store_true", help = "Set this to analyze CALCS h(t) data")
parser.add_option("--calcs-deltal-channel-name", metavar = "name", default = "CAL-DELTAL_EXTERNAL_DQ", help = "CALCS \delta L channel name (default = CAL-DELTAL_EXTERNAL_DQ)")
parser.add_option("--response-file", metavar = "name", help = "Name of .npz file containing response as derived from DARM model.")

options, filenames = parser.parse_args()

start = int(options.gps_start_time)
end = int(options.gps_end_time)
dt = float(options.dt)
ifo = options.ifo
hoft_frames_cache = options.hoft_frames_cache
if options.analyze_additional_hoft:
	additional_hoft_frames_cache = options.additional_hoft_frames_cache
	additional_hoft_channel_name = options.additional_hoft_channel_name
raw_frames_cache = options.raw_frames_cache
DARM_ERR_channel_name = options.darm_err_channel_name
hoft_channel_name = options.hoft_channel_name
if options.analyze_calcs_hoft:
	CALCS_deltal_channel_name = options.calcs_deltal_channel_name
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
if options.analyze_calcs_hoft:
	CALCS_data = TimeSeries.read(raw_frames_cache, "%s:%s" % (ifo, CALCS_deltal_channel_name), start = start, end = end)
# Read in the h(t) data without kappas applied
#C00_data = TimeSeriesDict.read(C00_hoft_frames_cache, channels = ["%s:%s" % (ifo, C00_hoft_channel_name)], start = start, end = end)
hoft_data = TimeSeries.read(hoft_frames_cache, "%s:%s" % (ifo, hoft_channel_name), start = start, end = end)
# If needed, read in the additional h(t) data without kappas applied
#C01_data = TimeSeriesDict.read(C01_hoft_frames_cache, channels = ["%s:%s" % (ifo, C01_hoft_channel_name)], start = start, end = end)
if options.analyze_additional_hoft:
	additional_hoft_data = TimeSeries.read(additional_hoft_frames_cache, "%s:%s" % (ifo, additional_hoft_channel_name), start = start, end = end)

# Pick out channel from dictionary
#DARM_ERR_data = DARM_ERR_data["%s:%s" % (ifo, DARM_ERR_channel_name)]
#CALCS_data = CALCS_data["%s:%s" % (ifo, CALCS_hoft_channel_name)]
#DARM_ERR_data = DARM_ERR_data["%s:%s" % (ifo, DARM_ERR_channel_name)]

# We want \Delta L, not strain, to find the response function
#C00_data = C00_data["%s:%s" % (ifo, C00_hoft_channel_name)] * 3995.1
#C01_data = C01_data["%s:%s" % (ifo, C01_hoft_channel_name)] * 3995.1

dur = end - start
averaging_time = 8
chunk_start = start
chunk_end = start + averaging_time

if options.analyze_calcs_hoft:
	CALCS_tf_data = numpy.zeros(len(response)) + 1j*numpy.zeros(len(response))
hoft_tf_data = numpy.zeros(len(response)) + 1j*numpy.zeros(len(response))
if options.analyze_additional_hoft:
	additional_hoft_tf_data = numpy.zeros(len(response)) + 1j*numpy.zeros(len(response))

N = 0

while (chunk_end <= end):
	# Correct for model jump delay in darm_err      
	DARM_ERR_chunk = DARM_ERR_data.crop(chunk_start+dt, chunk_end+dt, True)
	DARM_ERR_chunk = DARM_ERR_chunk.detrend()
	DARM_ERR_chunk_fft = DARM_ERR_chunk.average_fft(4, 2, window = 'hann')

	hoft_chunk = hoft_data.crop(chunk_start, chunk_end, True)
	hoft_chunk = hoft_chunk.detrend()
	hoft_chunk_fft = hoft_chunk.average_fft(4, 2, window = 'hann')
	
	hoft_chunk_tf = hoft_chunk_fft / DARM_ERR_chunk_fft
	hoft_tf_data += hoft_chunk_tf.value[:len(hoft_tf_data)]

	if options.analyze_calcs_hoft:
		CALCS_chunk = CALCS_data.crop(chunk_start, chunk_end, True)
		CALCS_chunk = CALCS_chunk.detrend()
		CALCS_chunk_fft = CALCS_chunk.average_fft(4, 2, window = 'hann')
		CALCS_chunk_fft = CALCS_chunk_fft.filter([30]*6, [0.3]*6, 1e-12)

		CALCS_chunk_tf = CALCS_chunk_fft / DARM_ERR_chunk_fft
		CALCS_tf_data += CALCS_chunk_tf.value[:len(hoft_tf_data)]

	if options.analyze_additional_hoft:
		additional_hoft_chunk = additional_hoft_data.crop(chunk_start, chunk_end, True)
		additional_hoft_chunk = additional_hoft_chunk.detrend()
		additional_hoft_chunk_fft = additional_hoft_chunk.average_fft(4, 2, window = 'hann')

		additional_hoft_chunk_tf = additional_hoft_chunk_fft / DARM_ERR_chunk_fft
		additional_hoft_tf_data += additional_hoft_chunk_tf.value[:len(hoft_tf_data)]

	chunk_start += averaging_time
	chunk_end += averaging_time
	N += 1

hoft_tf_data = hoft_tf_data / N
hoft_tf = FrequencySeries(hoft_tf_data, f0=f0, df=df)

if options.analyze_calcs_hoft:
	CALCS_tf_data = CALCS_tf_data / N
	CALCS_tf = FrequencySeries(CALCS_tf_data, f0=f0, df=df)
if options.analyze_additional_hoft:
	additional_hoft_tf_data = additional_hoft_tf_data / N
	additional_hoft_tf = FrequencySeries(additional_hoft_tf_data, f0=f0, df=df)

# Make plot that compares all of the derived response functions and the model response
plot = BodePlot(response_fs, frequencies=freqs, dB = False, linewidth=2)
plot.add_frequencyseries(hoft_tf*3995.1, dB = False, color='#ee0000',linewidth=2)
if options.analyze_calcs_hoft:
	plot.add_frequencyseries(CALCS_tf, dB=False, color='#4ba6ff', linewidth=2)
if options.analyze_additional_hoft:
	plot.add_frequencyseries(additional_hoft_tf*3995.1, dB = False, color="#94ded7", linewidth=2)
plot.legend([r'Reference model response function', r'GDS h(t) derived response function', r'CALCS h(t) derived response function', r'DCS h(t) derived response function'], loc='upper right', fontsize='x-small')
# FIXME: Figure out how to make the legend and title appropriate and flexible
plot.maxes.set_yscale('log')
plot.paxes.set_yscale("linear")
plot.save('%s_%s_%s_all_tf.png' % (ifo, options.gps_start_time, options.gps_end_time))

# Make a plot that compares the ratios of each derived response to the model
ratio_hoft = hoft_tf / response_fs
if options.analyze_calcs_hoft:
	ratio_CALCS = CALCS_tf / response_fs
if options.analyze_additional_hoft:
	ratio_additional_hoft = additional_hoft_tf / response_fs

plot = BodePlot(ratio_hoft*3995.1, frequencies = freqs, dB = False, color='#ee0000', linewidth=2)
if options.analyze_calcs_hoft:
	plot.add_frequencyseries(ratio_CALCS, dB = False, color='#4ba6ff',linewidth=2)
if options.analyze_additional_hoft:
	plot.add_frequencyseries(ratio_additional_hoft*3995.1, dB = False, color="#94ded7", linewidth=2)
plot.legend([r'GDS h(t) derived response / Reference model response', r'CALCS h(t) derived response / Reference model response', r'DCS h(t) derived response / Reference model response'], loc='upper right', fontsize='small')
plot.maxes.set_yscale('linear')
plot.paxes.set_yscale('linear')
plot.maxes.set_ylim(0,2)
plot.save('%s_%s_%s_all_tf_ratio.png' % (ifo, options.gps_start_time, options.gps_end_time))

plot = BodePlot(ratio_hoft*3995.1, frequencies = freqs, dB = False, color='#ee0000', linewidth=2)
if options.analyze_calcs_hoft:
	plot.add_frequencyseries(ratio_CALCS, dB = False, color='#4ba6ff',linewidth=2)
if options.analyze_additional_hoft:
	plot.add_frequencyseries(ratio_additional_hoft*3995.1, dB = False, color="#94ded7", linewidth=2)
plot.legend([r'GDS h(t) derived response / Reference model response', r'CALCS h(t) derived response / Reference model response', r'DCS h(t) derived response / Reference model response'], loc='upper right', fontsize='small')
plot.maxes.set_yscale('linear')
plot.paxes.set_yscale('linear')
plot.maxes.set_ylim(.9, 1.1)
plot.maxes.set_xlim(10, 5000)
plot.paxes.set_ylim(-5, 5)
plot.paxes.set_xlim(10, 5000)
plot.save('%s_%s_%s_all_tf_ratio_zoomed.png' % (ifo, options.gps_start_time, options.gps_end_time))



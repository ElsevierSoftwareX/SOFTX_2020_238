import matplotlib as mpl; mpl.use('Agg')
from gwpy.timeseries import StateVector
import numpy
import sys
from optparse import OptionParser, Option

parser = OptionParser()

parser.add_option("--gps-start-time", metavar = "seconds", help = "Set the GPS start time.")
parser.add_option("--gps-end-time", metavar = "seconds", help = "Set the GPS end time.")
parser.add_option("--ifo", metavar = "name", help = "Name of the IFO")
parser.add_option("--hoft-frames-cache", metavar = "name", help = "Frame cache file for h(t) data to be analyzed")
#parser.add_option("--raw-frames-cache", metavar = "name", help = "Frame cache for raw data.")
parser.add_option("--calib-state-vector-channel-name", metavar = "name", default = "GDS-CALIB_STATE_VECTOR", help = "Calibration state vector channel name (default = GDS-CALIB_STATE_VECTOR")
#parser.add_option("--analyze-calcs-hoft", action = "store_true", help = "Set this to analyze CALCS h(t) data")
#parser.add_option("--calcs-deltal-channel-name", metavar = "name", default = "CAL-DELTAL_EXTERNAL_DQ", help = "CALCS \delta L channel name (default = CAL-DELTAL_EXTERNAL_DQ)")

options, filenames = parser.parse_args()

start = int(options.gps_start_time)
end = int(options.gps_end_time)
ifo = options.ifo
hoft_frames_cache = options.hoft_frames_cache
calib_state_channel_name = options.calib_state_vector_channel_name

calib_state_vector = StateVector.read(hoft_frames_cache, "%s:%s" % (ifo, calib_state_channel_name), start = start, end = end)

plot = calib_state_vector.plot(format='segments')
ax = plot.gca()
ax.set_xscale('seconds')
ax.set_title("Calibration state vector")
plot.save("%s_%s_%s_calib_state_vector.pdf" % (ifo, options.gps_start_time, options.gps_end_time))

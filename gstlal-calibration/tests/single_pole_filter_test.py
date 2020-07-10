#!/usr/bin/env python3
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

#
# =============================================================================
#
#				   Preamble
#
# =============================================================================
#


import numpy
import sys
from gstlal import pipeparts
from gstlal import calibration_parts
import test_common
from gi.repository import Gst


#
# =============================================================================
#
#				  Pipelines
#
# =============================================================================
#

# For your reference:
# See gstlal/gstlal-calibration/python/calibration_parts.py
# See gstlal/gstlal-calibration/bin/gstlal_compute_strain
# See gstlal/gstlal/python/pipeparts/__init__.py
# See gstlal/gstlal-calibration/gst/lal/gstlal_adaptivefirfilt.c
# To implement your solution (once you have something that works), you will need to make changes around line 1717 and around line 2060 in gstlal_compute_strain.
# Quite possibly, you may need to make changes in the function starting at line 327 in gstlal_adaptivefirfilt.c.  There is already code in place to make a 2-tap filter for fcc (or an N - 1 tap filter for N zeros), but I have not tested it.  This is where Jolien's solution could go, if what's currently there is not correct.
# Also - just a warning - I just typed this testing script up, and it may have problems that you will need to debug.

def single_pole_filter_test(pipeline, name, line_sep = 0.5):

	rate = 16384		# Hz
	buffer_length = 1.0	# seconds
	test_duration = 1000	# seconds
	fcc_rate = 16		# Hz
	gain = 1.1
	fcc = 430		# Hz
	update_time = 20	# seconds

	#
	# build pipeline
	#

	# Make a source of fake data to act as input values for a gain and a zero
	gain_fcc_data = test_common.test_src(pipeline, rate = fcc_rate, test_duration = test_duration, wave = 5, src_suffix = "_gain_fcc_data")
	# Take to the power of 0 to make it a stream of ones
	gain_fcc_data = calibration_parts.mkpow(pipeline, gain_fcc_data, exponent = 0.0)
	# Make a copy which we can use to make both the gain and the fcc data
	gain_fcc_data = pipeparts.mktee(pipeline, gain_fcc_data)
	# Make the gain data by multiplying ones by the gain
	gain_data = pipeparts.mkaudioamplify(pipeline, gain_fcc_data, gain)
	# Make the fcc data by multiplying ones by fcc
	fcc_data = pipeparts.mkaudioamplify(pipeline, gain_fcc_data, fcc)
	# Now, this needs to be used in the element lal_adaptivefirfilt to turn it into a FIR filter.
	# lal_adaptivefirfilt takes as inputs (in this order) zeros, poles, a complex factor containing gain and phase.
	# Type "$ gst-inspect-1.0 lal_adaptivefirfilt" for info about the element.
	# Each of the inputs must be a complex data stream, so we first must make these real streams complex
	gain_data = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, gain_data, matrix = [[1.0, 0.0]]))
	fcc_data = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, fcc_data, matrix = [[1.0, 0.0]]))
	# Now we must interleave these streams, since lal_adaptivefirfilt takes a single multi-channel stream of interleaved data.
	# The fcc data (the zero frequency) must come first so that it is channel 0; that way lal_adaptivefirfilt recognizes it as such.
	filter_data = calibration_parts.mkinterleave(pipeline, [fcc_data, gain_data], complex_data = True)
	# Finally, send the interleaved data to lal_adaptivefirfilt, which will make a FIR filter.
	# Note that it needs to know the number of zeros and poles.
	# update_samples tells it how often to send a new filter to the filtering element
	# minimize_filter_length must be True for it to use a 2-tap filter.  Otherwise, it makes a longer FIR filter using and iFFT from a frequency-domain model.
	# This will also write the filter coefficients to file.
	adaptive_invsens_filter = calibration_parts.mkadaptivefirfilt(pipeline, filter_data, update_samples = int(update_time * fcc_rate), average_samples = 1, num_zeros = 1, num_poles = 0, filter_sample_rate = rate, minimize_filter_length = True, filename = "%s_FIR_filter.txt" % name)


	# Now make time series source of white noise to be filtered (wave = 5 means white noise)
	in_signal = test_common.test_src(pipeline, rate = rate, test_duration = test_duration, wave = 5, src_suffix = "_in_signal")
	# Make a copy of input data so that we can write it to file
	in_signal = pipeparts.mktee(pipeline, in_signal)
	# Write input time series to file
	pipeparts.mknxydumpsink(pipeline, in_signal, "%s_in.txt" % name)
	# Filter the data using lal_tdwhiten, which handles smooth filter updates
	# The property "kernel" is the FIR filter we will update.  To start, give a trivial default value.
	# The "taper_length" is the number of samples over which to handle filter transitions.  It is set to be 1 second.
	out_signal = pipeparts.mkgeneric(pipeline, in_signal, "lal_tdwhiten", kernel = [0, 1], latency = 0, taper_length = rate)
	# Hook up the adaptive filter from lal_adaptivefirfilt to lal_tdwhiten so that the filter gets updated
	adaptive_invsens_filter.connect("notify::adaptive-filter", calibration_parts.update_filter, out_signal, "adaptive_filter", "kernel")
	# Correct the 1/2-sample shift in timestamps by applying a linear-phase FIR filter
	# The last argument here (0.5) is the number of samples worth of timestamp advance to apply to the data.  You might want to try -0.5 as well, since I often get advances and delays mixed up.
	out_signal = calibration_parts.linear_phase_filter(pipeline, out_signal, 0.5)
	# Make a copy, so that we can write the time series to file and send it to lal_transferfunction
	out_signal = pipeparts.mktee(pipeline, out_signal)
	# Finally, write the output to file
	pipeparts.mknxydumpsink(pipeline, out_signal, "%s_out.txt" % name)


	# Now, take the input and output and compute a transfer function.
	# First, we need to interleave the data for use by lal_transferfunction. The numerator comes first
	tf_input = calibration_parts.mkinterleave(pipeline, [out_signal, in_signal])
	# Remove some initial samples in case they were filtered by the default dummy filter [0, 1]
	tf_input = calibration_parts.mkinsertgap(pipeline, tf_input, chop_length = Gst.SECOND * 50) # Removing 50 s of initial data
	# Send to lal_transferfunction, which will compute the frequency-domain transfer function between the input and output data and write it to file
	calibration_parts.mktransferfunction(pipeline, tf_input, fft_length = 16 * rate, fft_overlap = 8 * rate, num_ffts = (test_duration - 50) / 8 - 3, use_median = True, update_samples = 1e15, filename = "%s_filter_transfer_function.txt" % name)

	# You could, for instance, compare this transfer function to what you expect, i.e., gain * (1 + i f / fcc), and plot the comparison in the frequency-domain.  I'm guessing there will be a fair amount of work involved in getting everything to work and getting the result correct.

	#
	# done
	#
	
	return pipeline
	
#
# =============================================================================
#
#				     Main
#
# =============================================================================
#

test_common.build_and_run(single_pole_filter_test, "single_pole_filter_test")



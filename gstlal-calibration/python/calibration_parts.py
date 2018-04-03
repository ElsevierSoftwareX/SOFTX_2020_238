#!/usr/bin/env python
#
# Copyright (C) 2015 Madeline Wade
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

from gstlal import pipeparts
import numpy

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject
from gi.repository import Gst
GObject.threads_init()
Gst.init(None)

#
# Shortcut functions for common element combos/properties
#

def mkqueue(pipeline, head, length = 0, min_length = 0):
	if length < 0:
		return head
	else:
		return pipeparts.mkqueue(pipeline, head, max_size_time = int(1000000000 * length), max_size_buffers = 0, max_size_bytes = 0, min_threshold_time = min_length)

def mkcomplexqueue(pipeline, head, length = 0, min_length = 0):
	head = pipeparts.mktogglecomplex(pipeline, head)
	head = mkqueue(pipeline, head, length = length, min_length = min_length)
	head = pipeparts.mktogglecomplex(pipeline, head)
	return head
	
def mkinsertgap(pipeline, head, bad_data_intervals = [-1e35, -1e-35, 1e-35, 1e35], insert_gap = False, remove_gap = True, replace_value = 0, fill_discont = True, block_duration = Gst.SECOND, chop_length = 0):
	return pipeparts.mkgeneric(pipeline, head, "lal_insertgap", bad_data_intervals = bad_data_intervals, insert_gap = insert_gap, remove_gap = remove_gap, replace_value = replace_value, fill_discont = fill_discont, block_duration = int(block_duration), chop_length = int(chop_length))

#def mkupsample(pipeline, head, new_caps):
#	head = pipeparts.mkgeneric(pipeline, head, "lal_constantupsample")
#	head = pipeparts.mkcapsfilter(pipeline, head, new_caps)
#	return head

def mkstockresample(pipeline, head, caps):
	head = pipeparts.mkresample(pipeline, head, quality = 9)
	head = pipeparts.mkcapsfilter(pipeline, head, caps)
	return head

def mkresample(pipeline, head, quality, zero_latency, caps):
	head = pipeparts.mkgeneric(pipeline, head, "lal_resample", quality = quality, zero_latency = zero_latency)
	head = pipeparts.mkcapsfilter(pipeline, head, caps)
	return head

def mkcomplexfirbank(pipeline, src, latency = None, fir_matrix = None, time_domain = None, block_stride = None):
	properties = dict((name, value) for name, value in zip(("latency", "fir_matrix", "time_domain", "block_stride"), (latency, fir_matrix, time_domain, block_stride)) if value is not None)
	return pipeparts.mkgeneric(pipeline, src, "lal_complexfirbank", **properties)

def mkcomplexfirbank2(pipeline, src, latency = None, fir_matrix = None, time_domain = None, block_stride = None):
	properties = dict((name, value) for name, value in zip(("latency", "fir_matrix", "time_domain", "block_stride"), (latency, fir_matrix, time_domain, block_stride)) if value is not None)
	return pipeparts.mkgeneric(pipeline, src, "lal_complexfirbank2", **properties)

def mkmultiplier(pipeline, srcs, sync = True, queue_length = 0):
	elem = pipeparts.mkgeneric(pipeline, None, "lal_adder", sync=sync, mix_mode="product")
	if srcs is not None:
		for src in srcs:
			mkqueue(pipeline, src, length = queue_length).link(elem)
	return elem

def mkadder(pipeline, srcs, sync = True, queue_length = 0):
	elem = pipeparts.mkgeneric(pipeline, None, "lal_adder", sync=sync)
	if srcs is not None:
		for src in srcs:
			mkqueue(pipeline, src, length = queue_length).link(elem)
	return elem

def mkgate(pipeline, src, control, threshold, queue_length = 0, **properties):
	elem = pipeparts.mkgate(pipeline, mkqueue(pipeline, src, length = queue_length), control = mkqueue(pipeline, control, length = queue_length), threshold = threshold, **properties)
	return elem

def mkinterleave(pipeline, srcs):
	num_srcs = len(srcs)
	i = 0
	mixed_srcs = []
	for src in srcs:
		matrix = [numpy.zeros(num_srcs)]
		matrix[0][i] = 1
		mixed_srcs.append(pipeparts.mkmatrixmixer(pipeline, src, matrix=matrix))
		i += 1
	elem = mkadder(pipeline, tuple(mixed_srcs))

	#chan1 = pipeparts.mkmatrixmixer(pipeline, src1, matrix=[[1,0]])
	#chan2 = pipeparts.mkmatrixmixer(pipeline, src2, matrix=[[0,1]])
	#elem = mkadder(pipeline, list_srcs(pipeline, chan1, chan2)) 

	#elem = pipeparts.mkgeneric(pipeline, None, "interleave")
	#if srcs is not None:
	#	for src in srcs:
	#		pipeparts.mkqueue(pipeline, src).link(elem)
	return elem

def mkdeinterleave(pipeline, src, num_channels):
	head = pipeparts.mktee(pipeline, src)
	streams = []
	for i in range(0, num_channels):
		matrix = numpy.transpose([numpy.zeros(num_channels)])
		matrix[i][0] = 1.0
		streams.append(pipeparts.mkmatrixmixer(pipeline, head, matrix = matrix))

	return tuple(streams)


#
# Write a pipeline graph function
#

def write_graph(demux, pipeline, name):
	pipeparts.write_dump_dot(pipeline, "%s.%s" % (name, "PLAYING"), verbose = True)

#
# Common element combo functions
#

def hook_up(pipeline, demux, channel_name, instrument, buffer_length):
	if channel_name.endswith("UNCERTAINTY"):
		head = mkinsertgap(pipeline, None, block_duration = 1000000000 * buffer_length, replace_value = 1)
	else:
		head = mkinsertgap(pipeline, None, block_duration = 1000000000 * buffer_length)
	pipeparts.src_deferred_link(demux, "%s:%s" % (instrument, channel_name), head.get_static_pad("sink"))

	return head

def caps_and_progress(pipeline, head, caps, progress_name):
	head = pipeparts.mkaudioconvert(pipeline, head, caps)
	head = pipeparts.mkprogressreport(pipeline, head, "progress_src_%s" % progress_name)

	return head


#
# Function to make a list of heads to pass to, i.e. the multiplier or adder
#

def list_srcs(pipeline, *args):
	out = []
	for src in args:
		out.append(src)
	return tuple(out)

#
# Various filtering functions
#

def demodulate(pipeline, head, freq, td, caps, integration_samples, delay, chop_length, prefactor_real = 1.0, prefactor_imag = 0.0):
	# demodulate input at a given frequency freq

	head = pipeparts.mkgeneric(pipeline, head, "lal_demodulate", line_frequency = freq, prefactor_real = prefactor_real, prefactor_imag = prefactor_imag)
	head = mkresample(pipeline, head, 5, True, caps)
	head = mkcomplexfirbank(pipeline, head, fir_matrix=[numpy.hanning(integration_samples + 1) * 2 / integration_samples], time_domain = td, latency = delay)
	if chop_length != 0:
		head = pipeparts.mkgeneric(pipeline, head, "lal_insertgap", chop_length = chop_length)

	return head

def remove_lines(pipeline, head, freq, caps, filter_length):
	# remove any line(s) from a spectrum. filter length for demodulation (given in seconds) is adjustable
	# function argument caps must be complex caps

	integration_samples = filter_length * 16
	if type(freq) is not list:
		freq = [freq]

	head = pipeparts.mktee(pipeline, head)
	elem = pipeparts.mkgeneric(pipeline, None, "lal_adder", sync = True)
	mkqueue(pipeline, head).link(elem)
	for f in freq:
		line = pipeparts.mkgeneric(pipeline, head, "lal_demodulate", line_frequency = f)
		line = mkresample(pipeline, line, 5, False, "audio/x-raw,rate=16")
		line = mkcomplexfirbank(pipeline, line, latency = integration_samples / 2, fir_matrix = [numpy.hanning(integration_samples + 1) * 2 / integration_samples], time_domain = True)
		line = mkresample(pipeline, line, 3, False, caps)
		line = pipeparts.mkgeneric(pipeline, line, "lal_demodulate", line_frequency = -1.0 * f, prefactor_real = -2.0)
		real, imag = split_into_real(pipeline, line)
		pipeparts.mkfakesink(pipeline, imag)
		mkqueue(pipeline, real).link(elem)

	return elem

def remove_lines_with_witness(pipeline, signal, witness, freq, compute_rate = 16, rate_out = 16384, integration_samples = 320, obsready = None, latency_samples = 0, N_median = 2048, N_avg = 160):
	# remove line(s) from a spectrum. filter length for demodulation (given in seconds) is adjustable
	# function argument caps must be complex caps

	if latency_samples == 0:
		zero_latency = True
	else:
		zero_latency = False
	if type(freq) is not list:
		freq = [freq]

	if len(freq) > 1:
		witness = pipeparts.mktee(pipeline, witness)

	signal = pipeparts.mktee(pipeline, signal)
	signal_minus_lines = [signal]
	for f in freq:
		# Find amplitude and phase of line in witness channel
		line_in_witness = pipeparts.mkgeneric(pipeline, witness, "lal_demodulate", line_frequency = f)
		line_in_witness = mkresample(pipeline, line_in_witness, 5, zero_latency, "audio/x-raw,rate=%d" % compute_rate)
		line_in_witness = mkcomplexfirbank(pipeline, line_in_witness, latency = latency_samples, fir_matrix = [numpy.hanning(integration_samples + 1) * 2 / integration_samples], time_domain = True)
		line_in_witness = pipeparts.mktee(pipeline, line_in_witness)

		# Find amplitude and phase of line in signal
		line_in_signal = pipeparts.mkgeneric(pipeline, signal, "lal_demodulate", line_frequency = f)
		line_in_signal = mkresample(pipeline, line_in_signal, 5, zero_latency, "audio/x-raw,rate=%d" % compute_rate)
		line_in_signal = mkcomplexfirbank(pipeline, line_in_signal, latency = latency_samples, fir_matrix = [numpy.hanning(integration_samples + 1) * 2 / integration_samples], time_domain = True)

		# Find transfer function between witness channel and signal at this frequency
		tf_at_f = complex_division(pipeline, line_in_signal, line_in_witness)
		# Remove worthless data from computation of transfer function if we can
		if obsready is not None:
			tf_at_f = mkgate(pipeline, tf_at_f, obsready, 1, attack_length = -integration_samples)
		# Take a running median and average of transfer function
		tf_at_f = pipeparts.mkgeneric(pipeline, tf_at_f, "lal_smoothkappas", default_kappa_re = 0, array_size = N_median, avg_array_size = N_avg, default_to_median = True)

		# Use gated, averaged transfer function to reconstruct the sinusoid as it appears in the signal from the witness channel
		reconstructed_line_at_signal = mkmultiplier(pipeline, list_srcs(tf_at_f, line_in_witness))
		reconstructed_line_at_signal = mkresample(pipeline, reconstructed_line_at_signal, 3, False, "audio/x-raw,rate=%d" % rate_out)
		reconstructed_line_at_signal = pipeparts.mkgeneric(pipeline, reconstructed_line_at_signal, "lal_demodulate", line_frequency = -1.0 * f, prefactor_real = -2.0)
		reconstructed_line_at_signal, imag = split_into_real(pipeline, reconstructed_line_at_signal)
		pipeparts.mkfakesink(pipeline, imag)

		signal_minus_lines.append(reconstructed_line_at_signal)

	return mkadder(pipeline, tuple(signal_minus_lines))

def removeDC(pipeline, head, caps):
	head = pipeparts.mktee(pipeline, head)
	DC = mkresample(pipeline, head, 4, True, "audio/x-raw, rate=16")
	#DC = pipeparts.mkgeneric(pipeline, DC, "lal_smoothkappas", default_kappa_re = 0, array_size = 1, avg_array_size = 64)
	DC = mkresample(pipeline, DC, 4, True, caps)
	DC = pipeparts.mkaudioamplify(pipeline, DC, -1)

	return mkadder(pipeline, list_srcs(pipeline, head, DC))

def lowpass(pipeline, head, rate, length = 1.0, fcut = 500):
	length = int(length * rate)
	if not length % 2:
		length += 1 # Make sure the filter length is odd

	# Compute a low-pass filter.
	lowpass = numpy.sinc(2 * float(fcut) / rate * (numpy.arange(length) - (length - 1) / 2))
	lowpass *= numpy.blackman(length)
	lowpass /= numpy.sum(lowpass)

	# Now apply the filter
	return pipeparts.mkfirbank(pipeline, head, latency = int((length - 1) / 2), fir_matrix = [lowpass], time_domain = True)

def highpass(pipeline, head, rate, length = 1.0, fcut = 9.0):
	length = int(length * rate)
	if not length % 2:
		length += 1 # Make sure the filter length is odd

	# Compute a low-pass filter.
	lowpass = numpy.sinc(2 * float(fcut) / rate * (numpy.arange(length) - (length - 1) / 2))
	lowpass *= numpy.blackman(length)
	lowpass /= numpy.sum(lowpass)

	# Create a high-pass filter from the low-pass filter through spectral inversion.
	highpass = -lowpass
	highpass[int((length - 1) / 2)] += 1

	# Now apply the filter
	return pipeparts.mkfirbank(pipeline, head, latency = int((length - 1) / 2), fir_matrix = [highpass], time_domain = True)

def bandpass(pipeline, head, rate, length = 1.0, f_low = 100, f_high = 400):
	length = int(length * rate / 2)
	if not length % 2:
		length += 1 # Make sure the filter length is odd

	# Compute a temporary low-pass filter.
	lowpass = numpy.sinc(2 * float(f_low) / rate * (numpy.arange(length) - (length - 1) / 2))
	lowpass *= numpy.blackman(length)
	lowpass /= numpy.sum(lowpass)

	# Create the high-pass filter from the low-pass filter through spectral inversion.
	highpass = -lowpass
	highpass[(length - 1) / 2] += 1

	# Compute the low-pass filter.
	lowpass = numpy.sinc(2 * float(f_high) / rate * (numpy.arange(length) - (length - 1) / 2))
	lowpass *= numpy.blackman(length)
	lowpass /= numpy.sum(lowpass)

	# Convolve the high-pass and low-pass filters to make a band-pass filter
	bandpass = numpy.convolve(highpass, lowpass)

	# Now apply the filter
	return pipeparts.mkfirbank(pipeline, head, latency = int(length - 1), fir_matrix = [bandpass], time_domain = True)

def bandstop(pipeline, head, rate, length = 1.0, f_low = 100, f_high = 400):
	length = int(length * rate / 2)
	if not length % 2:
		length += 1 # Make sure the filter length is odd

	# Compute a temporary low-pass filter.
	lowpass = numpy.sinc(2 * float(f_low) / rate * (numpy.arange(length) - (length - 1) / 2))
	lowpass *= numpy.blackman(length)
	lowpass /= numpy.sum(lowpass)

	# Create a high-pass filter from the low-pass filter through spectral inversion.
	highpass = -lowpass
	highpass[(length - 1) / 2] += 1

	# Compute a low-pass filter.
	lowpass = numpy.sinc(2 * float(f_high) / rate * (numpy.arange(length) - (length - 1) / 2))
	lowpass *= numpy.blackman(length)
	lowpass /= numpy.sum(lowpass)

	# Convolve the high-pass and low-pass filters to make a temporary band-pass filter
	bandpass = numpy.convolve(highpass, lowpass)

	# Create a band-stop filter from the band-pass filter through spectral inversion.
	bandstop = -bandpass
	bandstop[length - 1] += 1

	# Now apply the filter
	return pipeparts.mkfirbank(pipeline, head, latency = int(length - 1), fir_matrix = [bandstop], time_domain = True)

def compute_rms(pipeline, head, rate, average_time, f_min = None, f_max = None, zero_latency = True, rate_out = 16):
	# Find the root mean square amplitude of a signal between two frequencies
	# Downsample to save computational cost
	head = mkresample(pipeline, head, 5, zero_latency, "audio/x-raw,rate=%d" % rate)

	# Remove any frequency content we don't care about
	if (f_min is not None) and (f_max is not None):
		head = bandpass(pipeline, head, rate, f_low = f_min, f_high = f_max)
	elif f_min is not None:
		head = highpass(pipeline, head, fcut = f_min)
	elif f_max is not None:
		head = lowpass(pipeline, head, fcut = f_max)

	# Square it
	head = pipeparts.mkpow(pipeline, head, exponent = 2.0)

	# Downsample again to save computational cost
	head = mkresample(pipeline, head, 3, zero_latency, "audio/x-raw,rate=%d" % rate_out)

	# Compute running average
	head = pipeparts.mkgeneric(pipeline, head, "lal_smoothkappas", default_kappa_re = 0.0, array_size = 1, avg_array_size = average_time * rate_out)

	return head

#
# Calibration factor related functions
#

def smooth_kappas_no_coherence(pipeline, head, var, expected, N, Nav, default_to_median):
	# Find median of calibration factors array with size N and smooth out medians with an average over Nav samples
	# Use the maximum_offset_re property to determine whether input kappas are good or not
	head = pipeparts.mkgeneric(pipeline, head, "lal_smoothkappas", maximum_offset_re = var, default_kappa_re = expected, array_size = N, avg_array_size = Nav, default_to_median = default_to_median)
	return head

def smooth_complex_kappas_no_coherence(pipeline, head, real_var, imag_var, real_expected, imag_expected, N, Nav, default_to_median):
	# Find median of complex calibration factors array with size N, split into real and imaginary parts, and smooth out medians with an average over Nav samples
	# Use the maximum_offset_re and maximum_offset_im properties to determine whether input kappas are good or not
	head = pipeparts.mkgeneric(pipeline, head, "lal_smoothkappas", maximum_offset_re = real_var, maximum_offset_im = imag_var, default_kappa_re = real_expected, default_kappa_im = imag_expected, array_size = N, avg_array_size = Nav, default_to_median = default_to_median)
	re, im = split_into_real(pipeline, head)
	return re, im

def smooth_kappas(pipeline, head, expected, N, Nav, default_to_median):
	# Find median of calibration factors array with size N and smooth out medians with an average over Nav samples
	# Assume input was previously gated with coherence uncertainty to determine if input kappas are good or not
	head = pipeparts.mkgeneric(pipeline, head, "lal_smoothkappas", default_kappa_re = expected, array_size = N, avg_array_size = Nav, default_to_median = default_to_median)
	return head

def smooth_complex_kappas(pipeline, head, real_expected, imag_expected, N, Nav, default_to_median):
	# Find median of complex calibration factors array with size N and smooth out medians with an average over Nav samples
	# Assume input was previously gated with coherence uncertainty to determine if input kappas are good or not

	head = pipeparts.mkgeneric(pipeline, head, "lal_smoothkappas", default_kappa_re = real_expected, default_kappa_im = imag_expected, array_size = N, avg_array_size = Nav, default_to_median = default_to_median)
	re, im = split_into_real(pipeline, head)
	return re, im

def track_bad_kappas_no_coherence(pipeline, head, var, expected, N, Nav, default_to_median):
	# Produce output of 1's or 0's that correspond to median not corrupted (1) or corrupted (0) based on whether median of input array is defualt value.
	head = pipeparts.mkgeneric(pipeline, head, "lal_smoothkappas", maximum_offset_re = var, default_kappa_re = expected, array_size = N, avg_array_size = Nav if default_to_median else 1, track_bad_kappa = True, default_to_median = default_to_median)
	return head

def track_bad_complex_kappas_no_coherence(pipeline, head, real_var, imag_var, real_expected, imag_expected, N, Nav, default_to_median):
	# Produce output of 1's or 0's that correspond to median not corrupted (1) or corrupted (0) based on whether median of input array is defualt value.
	# Real and imaginary parts are done separately (outputs of lal_smoothkappas can be 1+i, 1, i, or 0)
	head = pipeparts.mkgeneric(pipeline, head, "lal_smoothkappas", maximum_offset_re = real_var, maximum_offset_im = imag_var, default_kappa_re = real_expected, default_kappa_im = imag_expected, array_size = N, avg_array_size = Nav if default_to_median else 1, track_bad_kappa = True, default_to_median = default_to_median)
	re, im = split_into_real(pipeline, head)
	return re, im

def track_bad_kappas(pipeline, head, expected, N, Nav, default_to_median):
	# Produce output of 1's or 0's that correspond to median not corrupted (1) or corrupted (0) based on whether median of input array is defualt value.
	head = pipeparts.mkgeneric(pipeline, head, "lal_smoothkappas", default_kappa_re = expected, array_size = N, avg_array_size = Nav if default_to_median else 1, track_bad_kappa = True, default_to_median = default_to_median)
	return head

def track_bad_complex_kappas(pipeline, head, real_expected, imag_expected, N, Nav, default_to_median):
	# Produce output of 1's or 0's that correspond to median not corrupted (1) or corrupted (0) based on whether median of input array is defualt value.
	# Real and imaginary parts are done separately (outputs of lal_smoothkappas can be 1+i, 1, i, or 0)

	head = pipeparts.mkgeneric(pipeline, head, "lal_smoothkappas", default_kappa_re = real_expected, default_kappa_im = imag_expected, array_size = N, avg_array_size = Nav if default_to_median else 1, track_bad_kappa = True, default_to_median = default_to_median)
	re, im = split_into_real(pipeline, head)
	return re, im

def smooth_kappas_no_coherence_test(pipeline, head, var, expected, N, Nav, default_to_median):
	# Find median of calibration factors array with size N and smooth out medians with an average over Nav samples
	head = pipeparts.mktee(pipeline, head)
	pipeparts.mknxydumpsink(pipeline, head, "raw_kappatst.txt")
	head = pipeparts.mkgeneric(pipeline, head, "lal_smoothkappas", maximum_offset_re = var, default_kappa_re = expected, array_size = N, avg_array_size = Nav, default_to_median = default_to_median)
	head = pipeparts.mktee(pipeline, head)
	pipeparts.mknxydumpsink(pipeline, head, "smooth_kappatst.txt")
	return head

def compute_kappa_bits(pipeline, smoothR, smoothI, dqR, dqI, expected_real, expected_imag, real_ok_var, imag_ok_var, min_samples, status_out_smooth = 1, status_out_median = 1, starting_rate=16, ending_rate=16):

	smoothRInRange = mkinsertgap(pipeline, smoothR, bad_data_intervals = [expected_real - real_ok_var, expected_real, expected_real, expected_real + real_ok_var], insert_gap = True, remove_gap = False)
	smoothRInRange = pipeparts.mkbitvectorgen(pipeline, smoothRInRange, nongap_is_control = True, bit_vector = status_out_smooth)
	smoothRInRange = pipeparts.mkcapsfilter(pipeline, smoothRInRange, "audio/x-raw, format=U32LE, rate=%d, channel-mask=(bitmask)0x0" % starting_rate)
	if starting_rate != ending_rate:
		smoothRInRange = pipeparts.mkgeneric(pipeline, smoothRInRange, "lal_logicalundersample", required_on = status_out_smooth, status_out = status_out_smooth)
		smoothRInRange = pipeparts.mkcapsfilter(pipeline, smoothRInRange, "audio/x-raw, format=U32LE, rate=%d, channel-mask=(bitmask)0x0" % ending_rate)
	smoothRInRangetee = pipeparts.mktee(pipeline, smoothRInRange)

	smoothIInRange = mkinsertgap(pipeline, smoothI, bad_data_intervals = [expected_imag - imag_ok_var, expected_imag, expected_imag, expected_imag + imag_ok_var], insert_gap = True, remove_gap = False)
	smoothIInRange = pipeparts.mkbitvectorgen(pipeline, smoothIInRange, nongap_is_control = True, bit_vector = status_out_smooth)
	smoothIInRange = pipeparts.mkcapsfilter(pipeline, smoothIInRange, "audio/x-raw, format=U32LE, rate=%d, channel-mask=(bitmask)0x0" % starting_rate)
	if starting_rate != ending_rate:
		smoothIInRange = pipeparts.mkgeneric(pipeline, smoothIInRange, "lal_logicalundersample", required_on = status_out_smooth, status_out = status_out_smooth)
		smoothIInRange = pipeparts.mkcapsfilter(pipeline, smoothIInRange, "audio/x-raw, format=U32LE, rate=%d, channel-mask=(bitmask)0x0" % ending_rate)

	smoothInRange = mkgate(pipeline, smoothRInRangetee, mkadder(pipeline, list_srcs(pipeline, smoothIInRange, smoothRInRangetee)), status_out_smooth * 2, attack_length = -min_samples)
	smoothInRange = pipeparts.mkbitvectorgen(pipeline, smoothInRange, nongap_is_control = True, bit_vector = status_out_smooth)
	smoothInRange = pipeparts.mkcapsfilter(pipeline, smoothInRange, "audio/x-raw, format=U32LE, rate=%d, channel-mask=(bitmask)0x0" % ending_rate)

	dqtotal = mkadder(pipeline, list_srcs(pipeline, dqR, dqI))
	medianUncorrupt = pipeparts.mkbitvectorgen(pipeline, dqtotal, threshold = 2, bit_vector = status_out_median)
	medianUncorrupt = pipeparts.mkcapsfilter(pipeline, medianUncorrupt, "audio/x-raw, format=U32LE, rate=%d, channel-mask=(bitmask)0x0" % starting_rate)
	if starting_rate != ending_rate:
		medianUncorrupt = pipeparts.mkgeneric(pipeline, medianUncorrupt, "lal_logicalundersample", required_on = status_out_median, status_out = status_out_median)
		medianUncorrupt = pipeparts.mkcapsfilter(pipeline, medianUncorrupt, "audio/x-raw, format=U32LE, rate = %d, channel-mask=(bitmask)0x0" % ending_rate)

	return smoothInRange, medianUncorrupt

def compute_kappa_bits_only_real(pipeline, smooth, dq, expected, ok_var, min_samples, status_out_smooth = 1, status_out_median = 1, starting_rate=16, ending_rate=16):

	smoothInRange = mkinsertgap(pipeline, smooth, bad_data_intervals = [expected - ok_var, expected, expected, expected + ok_var], insert_gap = True, remove_gap = False)
	smoothInRange = pipeparts.mkbitvectorgen(pipeline, smoothInRange, nongap_is_control = True, bit_vector = status_out_smooth)
	smoothInRange = pipeparts.mkcapsfilter(pipeline, smoothInRange, "audio/x-raw, format=U32LE, rate=%d, channel-mask=(bitmask)0x0" % starting_rate)
	if starting_rate != ending_rate:
		smoothInRange = pipeparts.mkgeneric(pipeline, smoothInRange, "lal_logicalundersample", required_on = status_out_smooth, status_out = status_out_smooth)
		smoothInRange = pipeparts.mkcapsfilter(pipeline, smoothInRange, "audio/x-raw, format=U32LE, rate=%d, channel-mask=(bitmask)0x0" % ending_rate)
	smoothInRangetee = pipeparts.mktee(pipeline, smoothInRange)
	smoothInRange = mkgate(pipeline, smoothInRangetee, smoothInRangetee, status_out_smooth, attack_length = -min_samples)
	smoothInRange = pipeparts.mkbitvectorgen(pipeline, smoothInRange, nongap_is_control = True, bit_vector = status_out_smooth)

	medianUncorrupt = pipeparts.mkbitvectorgen(pipeline, dq, threshold = 1, bit_vector = status_out_median)
	medianUncorrupt = pipeparts.mkcapsfilter(pipeline, medianUncorrupt, "audio/x-raw, format=U32LE, rate=%d, channel-mask=(bitmask)0x0" % starting_rate)
	if starting_rate != ending_rate:
		medianUncorrupt = pipeparts.mkgeneric(pipeline, medianUncorrupt, "lal_logicalundersample", required_on = status_out_median, status_out = status_out_median)
		medianUncorrupt = pipeparts.mkcapsfilter(pipeline, medianUncorrupt, "audio/x-raw, format=U32LE, rate=%d, channel-mask=(bitmask)0x0" % ending_rate)

	return smoothInRange, medianUncorrupt

def merge_into_complex(pipeline, real, imag):
	# Merge real and imag into one complex channel with complex caps
	head = mkinterleave(pipeline, list_srcs(pipeline, real, imag))
	head = pipeparts.mktogglecomplex(pipeline,head)
	return head

def split_into_real(pipeline, complex_chan):
	# split complex channel with complex caps into two channels (real and imag) with real caps

	elem = pipeparts.mktogglecomplex(pipeline, complex_chan)
	(real, imag) = mkdeinterleave(pipeline, elem, 2)

#	elem = pipeparts.mkgeneric(pipeline, elem, "deinterleave", keep_positions=True)
#	real = pipeparts.mkgeneric(pipeline, None, "identity")
#	pipeparts.src_deferred_link(elem, "src_0", real.get_static_pad("sink"))
#	imag = pipeparts.mkgeneric(pipeline, None, "identity")
#	pipeparts.src_deferred_link(elem, "src_1", imag.get_static_pad("sink"))

	return real, imag

def complex_audioamplify(pipeline, chan, WR, WI):
	# Multiply a complex channel chan by a complex number WR+I WI
	# Re[out] = -chanI*WI + chanR*WR
	# Im[out] = chanR*WI + chanI*WR

	head = pipeparts.mktogglecomplex(pipeline, chan)
	head = pipeparts.mkmatrixmixer(pipeline, head, matrix=[[WR, WI],[-WI, WR]])
	head = pipeparts.mktogglecomplex(pipeline, head)

	return head

def complex_inverse(pipeline, head):
	# Invert a complex number (1/z)

	head = pipeparts.mktogglecomplex(pipeline, head)
	head = pipeparts.mkgeneric(pipeline, head, "complex_pow", exponent = -1)
	head = pipeparts.mktogglecomplex(pipeline, head)

	return head

def complex_division(pipeline, a, b):
	# Perform complex division of c = a/b and output the complex quotient c

	bInv = complex_inverse(pipeline, b)
	c = mkmultiplier(pipeline, list_srcs(pipeline, a, bInv))

	return c

def compute_kappatst_from_filters_file(pipeline, derrfesd, tstexcfesd, pcalfdarm, derrfdarm, ktstfacR, ktstfacI):

	#	       
	# \kappa_TST = ktstfac * (derrfesd/tstexcfesd) * (pcalfdarm/derrfdarm)
	# ktstfac = -(1/A0fesd) * (C0fdarm/(1+G0fdarm)) * ((1+G0fesd)/C0fesd)
	#

	derrfdarminv = complex_inverse(pipeline, derrfdarm)
	tstexcfesdinv = complex_inverse(pipeline, tstexcfesd)
	ktst = mkmultiplier(pipeline, list_srcs(pipeline, pcalfdarm, derrfdarminv, tstexcfesdinv, derrfesd))
	ktst = complex_audioamplify(pipeline, ktst, ktstfacR, ktstfacI)

	return ktst

def compute_kappatst(pipeline, derrfesd, tstexcfesd, pcalfdarm, derrfdarm, ktstfac):

	#	       
	# \kappa_TST = ktstfac * (derrfesd/tstexcfesd) * (pcalfdarm/derrfdarm)
	# ktstfac = -(1/A0fesd) * (C0fdarm/(1+G0fdarm)) * ((1+G0fesd)/C0fesd)
	#

	derrfdarminv = complex_inverse(pipeline, derrfdarm)
	tstexcfesdinv = complex_inverse(pipeline, tstexcfesd)
	ktst = mkmultiplier(pipeline, list_srcs(pipeline, ktstfac, pcalfdarm, derrfdarminv, tstexcfesdinv, derrfesd))

	return ktst

def compute_afctrl_from_filters_file(pipeline, derrfpu, excfpu, pcalfdarm, derrfdarm, afctrlfacR, afctrlfacI):

	#
	# A(f_ctrl) = -afctrlfac * (derrfpu/excfpu) * (pcalfdarm/derrfdarm)
	# afctrlfac = C0fpcal/(1+G0fpcal) * (1+G0fctrl)/C0fctrl
	#

	derrfdarminv = complex_inverse(pipeline, derrfdarm)
	excfpuinv = complex_inverse(pipeline, excfpu)
	afctrl = mkmultiplier(pipeline, list_srcs(pipeline, pcalfdarm, derrfdarminv, excfpuinv, derrfpu))
	afctrl = complex_audioamplify(pipeline, afctrl, -1.0*afctrlfacR, -1.0*afctrlfacI)

	return afctrl
	

def compute_afctrl(pipeline, derrfpu, excfpu, pcalfdarm, derrfdarm, afctrlfac):

	#
	# A(f_ctrl) = -afctrlfac * (derrfpu/excfpu) * (pcalfdarm/derrfdarm)
	# afctrlfac = EP2 = C0fpcal/(1+G0fpcal) * (1+G0fctrl)/C0fctrl
	#

	derrfdarminv = complex_inverse(pipeline, derrfdarm)
	excfpuinv = complex_inverse(pipeline, excfpu)
	afctrl = mkmultiplier(pipeline, list_srcs(pipeline, afctrlfac, pcalfdarm, derrfdarminv, excfpuinv, derrfpu))
	afctrl = complex_audioamplify(pipeline, afctrl, -1.0, 0.0)

	return afctrl

def compute_kappapu_from_filters_file(pipeline, EP3R, EP3I, afctrl, ktst, EP4R, EP4I):

	#
	# \kappa_pu = EP3 * (afctrl - ktst * EP4)
	#

	kpu = complex_audioamplify(pipeline, mkadder(pipeline, list_srcs(pipeline, afctrl, complex_audioamplify(pipeline, ktst, -1.0*EP4R, -1.0*EP4I))), EP3R, EP3I)	

	return kpu

def compute_kappapu(pipeline, EP3, afctrl, ktst, EP4):
	
	#
	# \kappa_pu = EP3 * (afctrl - ktst * EP4)
	#

	ep4_kappatst = mkmultiplier(pipeline, list_srcs(pipeline, ktst, complex_audioamplify(pipeline, EP4, -1.0, 0.0)))
	afctrl_kappatst = mkadder(pipeline, list_srcs(pipeline, afctrl, ep4_kappatst))
	kpu = mkmultiplier(pipeline, list_srcs(pipeline, EP3, afctrl_kappatst))

	return kpu

def compute_kappaa_from_filters_file(pipeline, afctrl, EP4R, EP4I, EP5R, EP5I):

	#
	#\kappa_a = afctrl / (EP4+EP5)
	#

	facR = (EP4R + EP5R) / ((EP4R + EP5R)**2 + (EP4I + EP5I)**2)
	facI = -(EP4I + EP5I) / ((EP4R + EP5R)**2 + (EP4I + EP5I)**2)

	ka = complex_audioamplify(pipeline, afctrl, facR, facI) 

	return ka

def compute_kappaa(pipeline, afctrl, EP4, EP5):

	#
	#\kappa_a = afctrl / (EP4 + EP5)
	#

	ka = complex_division(pipeline, afctrl, mkadder(pipeline, list_srcs(pipeline, EP4, EP5)))

	return ka


def compute_S_from_filters_file(pipeline, EP6R, EP6I, pcalfpcal2, derrfpcal2, EP7R, EP7I, ktst, EP8R, EP8I, kpu, EP9R, EP9I):

	#	
	# S = 1/EP6 * ( pcalfpcal2/derrfpcal2 - EP7*(ktst*EP8 + kpu*EP9) ) ^ (-1)
	#

	pcal_over_derr = complex_division(pipeline, pcalfpcal2, derrfpcal2)
	ep8_kappatst = complex_audioamplify(pipeline, ktst, EP8R, EP8I)
	ep9_kappapu = complex_audioamplify(pipeline, kpu, EP9R, EP9I)
	kappatst_kappapu = mkadder(pipeline, list_srcs(pipeline, ep8_kappatst, ep9_kappapu))
	kappatst_kappapu = complex_audioamplify(pipeline, kappatst_kappapu,  -1.0*EP7R, -1.0*EP7I)
	Sinv = mkadder(pipeline, list_srcs(pipeline, pcal_over_derr, kappatst_kappapu))
	Sinv = complex_audioamplify(pipeline, Sinv, EP6R, EP6I)
	S = complex_inverse(pipeline, Sinv)
	
	return S

def compute_S(pipeline, EP6, pcalfpcal2, derrfpcal2, EP7, ktst, EP8, kpu, EP9):

	#	
	# S = 1/EP6 * ( pcalfpcal2/derrfpcal2 - EP7*(ktst*EP8 + kpu*EP9) ) ^ (-1)
	#

	pcal_over_derr = complex_division(pipeline, pcalfpcal2, derrfpcal2)
	ep8_kappatst = mkmultiplier(pipeline, list_srcs(pipeline, ktst, EP8))
	ep9_kappapu = mkmultiplier(pipeline, list_srcs(pipeline, kpu, EP9))
	kappatst_kappapu = mkadder(pipeline, list_srcs(pipeline, ep8_kappatst, ep9_kappapu))
	kappatst_kappapu = mkmultiplier(pipeline, list_srcs(pipeline, complex_audioamplify(pipeline, EP7, -1.0, 0.0), kappatst_kappapu))
	Sinv = mkadder(pipeline, list_srcs(pipeline, pcal_over_derr, kappatst_kappapu))
	Sinv = mkmultiplier(pipeline, list_srcs(pipeline, EP6, Sinv))
	S = complex_inverse(pipeline, Sinv)

	return S

def compute_kappac(pipeline, SR, SI):

	#
	# \kappa_C = |S|^2 / Re[S]
	#

	SR = pipeparts.mktee(pipeline, SR)
	S2 = mkadder(pipeline, list_srcs(pipeline, pipeparts.mkpow(pipeline, SR, exponent=2.0), pipeparts.mkpow(pipeline, SI, exponent=2.0)))
	kc = mkmultiplier(pipeline, list_srcs(pipeline, S2, pipeparts.mkpow(pipeline, SR, exponent=-1.0)))
	return kc

def compute_fcc(pipeline, SR, SI, fpcal2):

	#
	# f_cc = - (Re[S]/Im[S]) * fpcal2
	#

	fcc = mkmultiplier(pipeline, list_srcs(pipeline, pipeparts.mkaudioamplify(pipeline, SR, -1.0*fpcal2), pipeparts.mkpow(pipeline, SI, exponent=-1.0)))
	return fcc

def compute_Xi_from_filters_file(pipeline, pcalfpcal4, darmfpcal4, fpcal4, EP11_real, EP11_imag, EP12_real, EP12_imag, EP13_real, EP13_imag, EP14_real, EP14_imag, ktst, kpu, kc, fcc):

	#
	# Xi = -1 + ((EP11*kc) / (1 + i * f_src/f_cc)) * (pcalfpcal4/derrfpcal4 - EP12*(ktst*EP13 + kpu*EP14))
	#

	Atst = complex_audioamplify(pipeline, ktst, EP13_real, EP13_imag)
	Apu = complex_audioamplify(pipeline, kpu, EP14_real, EP14_imag) 
	A = mkadder(pipeline, list_srcs(pipeline, Atst, Apu))
	minusAD = complex_audioamplify(pipeline, A, -1.0 * EP12_real, -1.0 * EP12_imag)
	pcal_over_derr = complex_division(pipeline, pcalfpcal4, darmfpcal4)
	pcal_over_derr_res = mkadder(pipeline, list_srcs(pipeline, pcal_over_derr, minusAD))
	fpcal4_over_fcc = pipeparts.mkaudioamplify(pipeline, pipeparts.mkpow(pipeline, fcc, exponent = -1.0), fpcal4)
	i_fpcal4_over_fcc = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, fpcal4_over_fcc, matrix = [[0, 1]]))
	i_fpcal4_over_fcc_plus_one = pipeparts.mkgeneric(pipeline, i_fpcal4_over_fcc, "lal_add_constant", value = 1.0)
	i_fpcal4_over_fcc_plus_one_inv = complex_inverse(pipeline, i_fpcal4_over_fcc_plus_one)
	kc_EP11 = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, kc, matrix = [[EP11_real, EP11_imag]]))
	Xi_plus_one = mkmultiplier(pipeline, list_srcs(pipeline, kc_EP11, i_fpcal4_over_fcc_plus_one_inv, pcal_over_derr_res))
	Xi = pipeparts.mkgeneric(pipeline, Xi_plus_one, "lal_add_constant", value = -1.0)

	return Xi

def compute_Xi(pipeline, pcalfpcal4, darmfpcal4, fpcal4, EP11, EP12, EP13, EP14, ktst, kpu, kc, fcc):

	#
	# Xi = -1 + ((EP11*kc) / (1 + i * f_src/f_cc)) * (pcalfpcal4/derrfpcal4 - EP12*(ktst*EP13 + kpu*EP14))
	#

	complex_kc = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, kc, matrix=[[1,0]]))
	Atst = mkmultiplier(pipeline, list_srcs(pipeline, EP13, ktst))
	Apu = mkmultiplier(pipeline, list_srcs(pipeline, EP14, kpu))
	A = mkadder(pipeline, list_srcs(pipeline, Atst, Apu))
	minusAD = mkmultiplier(pipeline, list_srcs(pipeline, complex_audioamplify(pipeline, EP12, -1.0, 0.0), A))
	pcal_over_derr = complex_division(pipeline, pcalfpcal4, darmfpcal4)
	pcal_over_derr_res = mkadder(pipeline, list_srcs(pipeline, pcal_over_derr, minusAD))
	fpcal4_over_fcc = pipeparts.mkaudioamplify(pipeline, pipeparts.mkpow(pipeline, fcc, exponent = -1.0), fpcal4)
	i_fpcal4_over_fcc = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, fpcal4_over_fcc, matrix = [[0, 1]]))
	i_fpcal4_over_fcc_plus_one = pipeparts.mkgeneric(pipeline, i_fpcal4_over_fcc, "lal_add_constant", value = 1.0)
	i_fpcal4_over_fcc_plus_one_inv = complex_inverse(pipeline, i_fpcal4_over_fcc_plus_one)
	Xi_plus_one = mkmultiplier(pipeline, list_srcs(pipeline, EP11, complex_kc, i_fpcal4_over_fcc_plus_one_inv, pcal_over_derr_res))
	Xi = pipeparts.mkgeneric(pipeline, Xi_plus_one, "lal_add_constant", value = -1.0)

	return Xi

def update_filter(filter_maker, arg, filter_taker, maker_prop_name, taker_prop_name, filter_number):
	firfilter = filter_maker.get_property(maker_prop_name)[filter_number][::-1]
	filter_taker.set_property(taker_prop_name, firfilter)

def clean_data(pipeline, signal, signal_rate, witnesses, witness_rate, fft_length, fft_overlap, num_ffts, update_samples, fir_length, frequency_resolution, obsready = None, filename = None):

	#
	# Note: this function can cause pipelines to lock up. Adding queues does not seem to help.
	# What does seem to help is one of two things: either replace lal_transferfunction with
	# another sink element, or be sure to give it inputs that feed into this function before
	# going anywhere else (e.g., if there is a tee, hook this function to the tee before
	# anything else.
	#

	default_fir_filter = numpy.zeros(fir_length)

	signal_tee = pipeparts.mktee(pipeline, signal)
	witnesses = list(witnesses)
	witness_tees = []
	for i in range(0, len(witnesses)):
		witnesses[i] = mkresample(pipeline, witnesses[i], 5, False, "audio/x-raw,rate=%d" % witness_rate)
		witnesses[i] = highpass(pipeline, witnesses[i], witness_rate)
		witness_tees.append(pipeparts.mktee(pipeline, witnesses[i]))

	resampled_signal = mkresample(pipeline, signal_tee, 5, False, "audio/x-raw,rate=%d" % witness_rate)
	transfer_functions = mkinterleave(pipeline, numpy.insert(witness_tees, 0, resampled_signal, axis = 0))
	if obsready is not None:
		transfer_functions = mkgate(pipeline, transfer_functions, obsready, 1)
	transfer_functions = pipeparts.mkgeneric(pipeline, transfer_functions, "lal_transferfunction", fft_length = fft_length, fft_overlap = fft_overlap, num_ffts = num_ffts, update_samples = update_samples, make_fir_filters = -1, fir_length = fir_length, frequency_resolution = frequency_resolution, high_pass = 9, update_after_gap = True, filename = filename)
	signal_minus_noise = [signal_tee]
	for i in range(0, len(witnesses)):
		minus_noise = pipeparts.mkgeneric(pipeline, witness_tees[i], "lal_tdwhiten", kernel = default_fir_filter, latency = fir_length / 2, taper_length = 20 * fir_length)
		transfer_functions.connect("notify::fir-filters", update_filter, minus_noise, "fir_filters", "kernel", i)
		signal_minus_noise.append(mkresample(pipeline, minus_noise, 5, False, "audio/x-raw,rate=%d" % signal_rate))

	return mkadder(pipeline, tuple(signal_minus_noise))




#!/usr/bin/env python
#
# Copyright (C) 2015 Madeline Wade, Aaron Viets
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
		return pipeparts.mkqueue(pipeline, head, max_size_time = int(1000000000 * length), max_size_buffers = 0, max_size_bytes = 0, min_threshold_time = int(1000000000 * min_length))

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
	if type(caps) is int:
		caps = "audio/x-raw,rate=%d,channel-mask=(bitmask)0x0" % caps
	head = pipeparts.mkresample(pipeline, head, quality = 9)
	head = pipeparts.mkcapsfilter(pipeline, head, caps)
	return head

def mkresample(pipeline, head, quality, zero_latency, caps):
	if type(caps) is int:
		caps = "audio/x-raw,rate=%d,channel-mask=(bitmask)0x0" % caps
	head = pipeparts.mkgeneric(pipeline, head, "lal_resample", quality = quality, zero_latency = zero_latency)
	head = pipeparts.mkcapsfilter(pipeline, head, caps)
	return head

def mkcomplexfirbank(pipeline, src, latency = None, fir_matrix = None, time_domain = None, block_stride = None):
	properties = dict((name, value) for name, value in zip(("latency", "fir_matrix", "time_domain", "block_stride"), (latency, fir_matrix, time_domain, block_stride)) if value is not None)
	return pipeparts.mkgeneric(pipeline, src, "lal_complexfirbank", **properties)

def mkcomplexfirbank2(pipeline, src, latency = None, fir_matrix = None, time_domain = None, block_stride = None):
	properties = dict((name, value) for name, value in zip(("latency", "fir_matrix", "time_domain", "block_stride"), (latency, fir_matrix, time_domain, block_stride)) if value is not None)
	return pipeparts.mkgeneric(pipeline, src, "lal_complexfirbank2", **properties)

def mkpow(pipeline, src, **properties):
	return pipeparts.mkgeneric(pipeline, src, "cpow", **properties)

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

def mkinterleave(pipeline, srcs, complex_data = False):
	complex_factor = 1 + int(complex_data)
	num_srcs = complex_factor * len(srcs)
	i = 0
	mixed_srcs = []
	for src in srcs:
		matrix = [numpy.zeros(num_srcs)]
		matrix[0][i] = 1
		mixed_srcs.append(pipeparts.mkmatrixmixer(pipeline, src, matrix=matrix))
		i += complex_factor
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

def demodulate(pipeline, head, freq, td, rate, filter_time, filter_latency, prefactor_real = 1.0, prefactor_imag = 0.0):
	# demodulate input at a given frequency freq

	head = pipeparts.mkgeneric(pipeline, head, "lal_demodulate", line_frequency = freq, prefactor_real = prefactor_real, prefactor_imag = prefactor_imag)
	head = mkresample(pipeline, head, 5, filter_latency == 0.0, rate)
	head = lowpass(pipeline, head, rate, length = filter_time, fcut = 0, filter_latency = filter_latency, td = td)

	return head

def remove_harmonics(pipeline, signal, f0, num_harmonics, f0_var, filter_latency, compute_rate = 16, rate_out = 16384):
	# remove any line(s) from a spectrum. filter length for demodulation (given in seconds) is adjustable
	# function argument caps must be complex caps

	filter_param = 0.0625
	head = pipeparts.mktee(pipeline, head)
	elem = pipeparts.mkgeneric(pipeline, None, "lal_adder", sync = True)
	mkqueue(pipeline, head).link(elem)
	for i in range(1, num_harmonics + 1):
		line = pipeparts.mkgeneric(pipeline, head, "lal_demodulate", line_frequency = i * f0)
		line = mkresample(pipeline, line, 5, filter_latency == 0, compute_rate)
		line_in_witness = lowpass(pipeline, line_in_witness, compute_rate, length = filter_param / (f0_var * i), fcut = 0, filter_latency = filter_latency)
		line = mkresample(pipeline, line, 3, filter_latency == 0.0, rate_out)
		line = pipeparts.mkgeneric(pipeline, line, "lal_demodulate", line_frequency = -1.0 * i * f0, prefactor_real = -2.0)
		line = pipeparts.mkgeneric(pipeline, line, "creal")
		mkqueue(pipeline, line).link(elem)

	return elem

def remove_harmonics_with_witness(pipeline, signal, witness, f0, num_harmonics, f0_var, filter_latency, compute_rate = 16, rate_out = 16384, num_avg = 2048, obsready = None):
	# remove line(s) from a spectrum. filter length for demodulation (given in seconds) is adjustable
	# function argument caps must be complex caps

	filter_param = 0.0625
	downsample_quality = 4
	upsample_quality = 4
	resample_shift = 16.0 + 16.5
	zero_latency = filter_latency == 0.0
	filter_latency = min(0.5, filter_latency)

	witness = pipeparts.mktee(pipeline, witness)
	signal = pipeparts.mktee(pipeline, signal)

	# If f0 strays from its nominal value and there is a timestamp shift in the signal
	# (e.g., to achieve zero latency), we need to correct the phase in the reconstructed
	# signal. To do this, we measure the frequency in the witness and find the beat
	# frequency between that and the nominal frequency f0.
	if filter_latency != 0.5:
		# The low-pass and resampling filters are not centered in time
		f0_measured = pipeparts.mkgeneric(pipeline, witness, "lal_trackfrequency", num_halfcycles = int(round((filter_param / f0_var / 2 + resample_shift / compute_rate) * f0)))
		f0_measured = mkresample(pipeline, f0_measured, 3, zero_latency, compute_rate)
		f0_measured = pipeparts.mkgeneric(pipeline, f0_measured, "lal_smoothkappas", array_size = 1, avg_array_size = int(round((filter_param / f0_var / 2 * compute_rate + resample_shift) / 2)), default_kappa_re = 0, default_to_median = True, filter_latency = filter_latency)
		f0_beat_frequency = pipeparts.mkgeneric(pipeline, f0_measured, "lal_add_constant", value = -f0)
		f0_beat_frequency = pipeparts.mktee(pipeline, f0_beat_frequency)

	signal_minus_lines = [signal]

	for n in range(1, num_harmonics + 1):
		# Length of low-pass filter
		filter_length = filter_param / (f0_var * n)
		filter_samples = int(filter_length * compute_rate) + (1 - int(filter_length * compute_rate) % 2)
		sample_shift = filter_samples / 2 - int((filter_samples - 1) * filter_latency + 0.5)
		# shift of timestamp relative to data
		time_shift = float(sample_shift) / compute_rate + zero_latency * resample_shift / compute_rate
		two_n_pi_delta_t = 2 * n * numpy.pi * time_shift

		# Only do this if we have to
		if filter_latency != 0.5:
			# Find phase shift due to timestamp shift for each harmonic
			phase_shift = pipeparts.mkmatrixmixer(pipeline, f0_beat_frequency, matrix=[[0, two_n_pi_delta_t]])
			phase_shift = pipeparts.mktogglecomplex(pipeline, phase_shift)
			phase_factor = pipeparts.mkgeneric(pipeline, phase_shift, "cexp")

		# Find amplitude and phase of each harmonic in the witness channel
		line_in_witness = pipeparts.mkgeneric(pipeline, witness, "lal_demodulate", line_frequency = n * f0)
		line_in_witness = mkresample(pipeline, line_in_witness, downsample_quality, zero_latency, compute_rate)
		line_in_witness = lowpass(pipeline, line_in_witness, compute_rate, length = filter_length, fcut = 0, filter_latency = filter_latency)
		line_in_witness = pipeparts.mktee(pipeline, line_in_witness)

		# Find amplitude and phase of line in signal
		line_in_signal = pipeparts.mkgeneric(pipeline, signal, "lal_demodulate", line_frequency = n * f0)
		line_in_signal = mkresample(pipeline, line_in_signal, downsample_quality, zero_latency, compute_rate)
		line_in_signal = lowpass(pipeline, line_in_signal, compute_rate, length = filter_length, fcut = 0, filter_latency = filter_latency)

		# Find transfer function between witness channel and signal at this frequency
		tf_at_f = complex_division(pipeline, line_in_signal, line_in_witness)
		# It may be necessary to remove the first few samples since denominator samples may have arrived before numerator samples, in which case the adder assumes the numerator is one.
		tf_at_f = pipeparts.mkgeneric(pipeline, tf_at_f, "lal_insertgap", chop_length = 1000000000) # Removing one second

		# Remove worthless data from computation of transfer function if we can
		if obsready is not None:
			tf_at_f = mkgate(pipeline, tf_at_f, obsready, 1, attack_length = -((1.0 - filter_latency) * filter_samples))
		tf_at_f = pipeparts.mkgeneric(pipeline, tf_at_f, "lal_smoothkappas", default_kappa_re = 0, array_size = 1, avg_array_size = num_avg, default_to_median = True, filter_latency = filter_latency)

		# Use gated, averaged transfer function to reconstruct the sinusoid as it appears in the signal from the witness channel
		if filter_latency == 0.5:
			reconstructed_line_in_signal = mkmultiplier(pipeline, list_srcs(pipeline, tf_at_f, line_in_witness))
		else:
			reconstructed_line_in_signal = mkmultiplier(pipeline, list_srcs(pipeline, tf_at_f, line_in_witness, phase_factor))
		# It may be necessary to remove the first few samples since line_in_witness may have arrived first, in which case the result of the above multiplication would be line_in_witness
		reconstructed_line_in_signal = pipeparts.mkgeneric(pipeline, reconstructed_line_in_signal, "lal_insertgap", chop_length = 1000000000)
		reconstructed_line_in_signal = mkresample(pipeline, reconstructed_line_in_signal, upsample_quality, zero_latency, rate_out)
		reconstructed_line_in_signal = pipeparts.mkgeneric(pipeline, reconstructed_line_in_signal, "lal_demodulate", line_frequency = -1.0 * n * f0, prefactor_real = -2.0)
		reconstructed_line_in_signal = pipeparts.mkgeneric(pipeline, reconstructed_line_in_signal, "creal")

		signal_minus_lines.append(reconstructed_line_in_signal)

	clean_signal = mkadder(pipeline, tuple(signal_minus_lines))

	return clean_signal

def removeDC(pipeline, head, rate):
	head = pipeparts.mktee(pipeline, head)
	DC = mkresample(pipeline, head, 4, True, 16)
	#DC = pipeparts.mkgeneric(pipeline, DC, "lal_smoothkappas", default_kappa_re = 0, array_size = 1, avg_array_size = 64)
	DC = mkresample(pipeline, DC, 4, True, rate)
	DC = pipeparts.mkaudioamplify(pipeline, DC, -1)

	return mkadder(pipeline, list_srcs(pipeline, head, DC))

def lowpass(pipeline, head, rate, length = 1.0, fcut = 500, filter_latency = 0.5, td = True):
	length = int(length * rate)
	if not length % 2:
		length += 1 # Make sure the filter length is odd

	# Compute a low-pass filter.
	lowpass = numpy.sinc(2 * float(fcut) / rate * (numpy.arange(length) - (length - 1) / 2))
	lowpass *= numpy.blackman(length)
	lowpass /= numpy.sum(lowpass)

	# Now apply the filter
	return mkcomplexfirbank(pipeline, head, latency = int((length - 1) * filter_latency + 0.5), fir_matrix = [lowpass], time_domain = td)

def highpass(pipeline, head, rate, length = 1.0, fcut = 9.0, filter_latency = 0.5, td = True):
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
	return mkcomplexfirbank(pipeline, head, latency = int((length - 1) * filter_latency + 0.5), fir_matrix = [highpass], time_domain = td)

def bandpass(pipeline, head, rate, length = 1.0, f_low = 100, f_high = 400, filter_latency = 0.5, td = True):
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
	return mkcomplexfirbank(pipeline, head, latency = int((length - 1) * 2 * filter_latency + 0.5), fir_matrix = [bandpass], time_domain = td)

def bandstop(pipeline, head, rate, length = 1.0, f_low = 100, f_high = 400, filter_latency = 0.5, td = True):
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
	return mkcomplexfirbank(pipeline, head, latency = int((length - 1) * 2 * filter_latency + 0.5), fir_matrix = [bandstop], time_domain = td)

def compute_rms(pipeline, head, rate, average_time, f_min = None, f_max = None, filter_latency = 0.5, rate_out = 16, td = True):
	# Find the root mean square amplitude of a signal between two frequencies
	# Downsample to save computational cost
	head = mkresample(pipeline, head, 5, filter_latency == 0.0, rate)

	# Remove any frequency content we don't care about
	if (f_min is not None) and (f_max is not None):
		head = bandpass(pipeline, head, rate, f_low = f_min, f_high = f_max, filter_latency = filter_latency, td = td)
	elif f_min is not None:
		head = highpass(pipeline, head, fcut = f_min, filter_latency = filter_latency, td = td)
	elif f_max is not None:
		head = lowpass(pipeline, head, fcut = f_max, filter_latency = filter_latency, td = td)

	# Square it
	head = mkpow(pipeline, head, exponent = 2.0)

	# Downsample again to save computational cost
	head = mkresample(pipeline, head, 4, filter_latency == 0.0, rate_out)

	# Compute running average
	head = pipeparts.mkgeneric(pipeline, head, "lal_smoothkappas", default_kappa_re = 0.0, array_size = 1, avg_array_size = average_time * rate_out, filter_latency = filter_latency)

	return head

#
# Calibration factor related functions
#

def smooth_kappas_no_coherence(pipeline, head, var, expected, N, Nav, default_to_median, filter_latency):
	# Find median of calibration factors array with size N and smooth out medians with an average over Nav samples
	# Use the maximum_offset_re property to determine whether input kappas are good or not
	head = pipeparts.mkgeneric(pipeline, head, "lal_smoothkappas", maximum_offset_re = var, default_kappa_re = expected, array_size = N, avg_array_size = Nav, default_to_median = default_to_median, filter_latency = filter_latency)
	return head

def smooth_complex_kappas_no_coherence(pipeline, head, real_var, imag_var, real_expected, imag_expected, N, Nav, default_to_median, filter_latency):
	# Find median of complex calibration factors array with size N, split into real and imaginary parts, and smooth out medians with an average over Nav samples
	# Use the maximum_offset_re and maximum_offset_im properties to determine whether input kappas are good or not
	head = pipeparts.mkgeneric(pipeline, head, "lal_smoothkappas", maximum_offset_re = real_var, maximum_offset_im = imag_var, default_kappa_re = real_expected, default_kappa_im = imag_expected, array_size = N, avg_array_size = Nav, default_to_median = default_to_median, filter_latency = filter_latency)
	return head

def smooth_kappas(pipeline, head, expected, N, Nav, default_to_median, filter_latency):
	# Find median of calibration factors array with size N and smooth out medians with an average over Nav samples
	# Assume input was previously gated with coherence uncertainty to determine if input kappas are good or not
	head = pipeparts.mkgeneric(pipeline, head, "lal_smoothkappas", default_kappa_re = expected, array_size = N, avg_array_size = Nav, default_to_median = default_to_median, filter_latency = filter_latency)
	return head

def smooth_complex_kappas(pipeline, head, real_expected, imag_expected, N, Nav, default_to_median, filter_latency):
	# Find median of complex calibration factors array with size N and smooth out medians with an average over Nav samples
	# Assume input was previously gated with coherence uncertainty to determine if input kappas are good or not

	head = pipeparts.mkgeneric(pipeline, head, "lal_smoothkappas", default_kappa_re = real_expected, default_kappa_im = imag_expected, array_size = N, avg_array_size = Nav, default_to_median = default_to_median, filter_latency = filter_latency)
	return head

def track_bad_kappas_no_coherence(pipeline, head, var, expected, N, Nav, default_to_median, filter_latency):
	# Produce output of 1's or 0's that correspond to median not corrupted (1) or corrupted (0) based on whether median of input array is defualt value.
	head = pipeparts.mkgeneric(pipeline, head, "lal_smoothkappas", maximum_offset_re = var, default_kappa_re = expected, array_size = N, avg_array_size = Nav if default_to_median else 1, track_bad_kappa = True, default_to_median = default_to_median, filter_latency = filter_latency)
	return head

def track_bad_complex_kappas_no_coherence(pipeline, head, real_var, imag_var, real_expected, imag_expected, N, Nav, default_to_median, filter_latency):
	# Produce output of 1's or 0's that correspond to median not corrupted (1) or corrupted (0) based on whether median of input array is defualt value.
	# Real and imaginary parts are done separately (outputs of lal_smoothkappas can be 1+i, 1, i, or 0)
	head = pipeparts.mkgeneric(pipeline, head, "lal_smoothkappas", maximum_offset_re = real_var, maximum_offset_im = imag_var, default_kappa_re = real_expected, default_kappa_im = imag_expected, array_size = N, avg_array_size = Nav if default_to_median else 1, track_bad_kappa = True, default_to_median = default_to_median, filter_latency = filter_latency)
	re, im = split_into_real(pipeline, head)
	return re, im

def track_bad_kappas(pipeline, head, expected, N, Nav, default_to_median, filter_latency):
	# Produce output of 1's or 0's that correspond to median not corrupted (1) or corrupted (0) based on whether median of input array is defualt value.
	head = pipeparts.mkgeneric(pipeline, head, "lal_smoothkappas", default_kappa_re = expected, array_size = N, avg_array_size = Nav if default_to_median else 1, track_bad_kappa = True, default_to_median = default_to_median, filter_latency = filter_latency)
	return head

def track_bad_complex_kappas(pipeline, head, real_expected, imag_expected, N, Nav, default_to_median, filter_latency):
	# Produce output of 1's or 0's that correspond to median not corrupted (1) or corrupted (0) based on whether median of input array is defualt value.
	# Real and imaginary parts are done separately (outputs of lal_smoothkappas can be 1+i, 1, i, or 0)

	head = pipeparts.mkgeneric(pipeline, head, "lal_smoothkappas", default_kappa_re = real_expected, default_kappa_im = imag_expected, array_size = N, avg_array_size = Nav if default_to_median else 1, track_bad_kappa = True, default_to_median = default_to_median, filter_latency = filter_latency)
	re, im = split_into_real(pipeline, head)
	return re, im

def smooth_kappas_no_coherence_test(pipeline, head, var, expected, N, Nav, default_to_median, filter_latency):
	# Find median of calibration factors array with size N and smooth out medians with an average over Nav samples
	head = pipeparts.mktee(pipeline, head)
	pipeparts.mknxydumpsink(pipeline, head, "raw_kappatst.txt")
	head = pipeparts.mkgeneric(pipeline, head, "lal_smoothkappas", maximum_offset_re = var, default_kappa_re = expected, array_size = N, avg_array_size = Nav, default_to_median = default_to_median, filter_latency = filter_latency)
	head = pipeparts.mktee(pipeline, head)
	pipeparts.mknxydumpsink(pipeline, head, "smooth_kappatst.txt")
	return head

def compute_kappa_bits(pipeline, smoothR, smoothI, expected_real, expected_imag, real_ok_var, imag_ok_var, min_samples, status_out_smooth = 1, starting_rate=16, ending_rate=16):

	if type(real_ok_var) is list:
		smoothRInRange = mkinsertgap(pipeline, smoothR, bad_data_intervals = [real_ok_var[0], expected_real, expected_real, real_ok_var[1]], insert_gap = True, remove_gap = False)
	else:
		smoothRInRange = mkinsertgap(pipeline, smoothR, bad_data_intervals = [expected_real - real_ok_var, expected_real, expected_real, expected_real + real_ok_var], insert_gap = True, remove_gap = False)
	smoothRInRange = pipeparts.mkbitvectorgen(pipeline, smoothRInRange, nongap_is_control = True, bit_vector = status_out_smooth)
	smoothRInRange = pipeparts.mkcapsfilter(pipeline, smoothRInRange, "audio/x-raw, format=U32LE, rate=%d, channel-mask=(bitmask)0x0" % starting_rate)
	if starting_rate != ending_rate:
		smoothRInRange = pipeparts.mkgeneric(pipeline, smoothRInRange, "lal_logicalundersample", required_on = status_out_smooth, status_out = status_out_smooth)
		smoothRInRange = pipeparts.mkcapsfilter(pipeline, smoothRInRange, "audio/x-raw, format=U32LE, rate=%d, channel-mask=(bitmask)0x0" % ending_rate)
	smoothRInRangetee = pipeparts.mktee(pipeline, smoothRInRange)

	if type(imag_ok_var) is list:
		smoothIInRange = mkinsertgap(pipeline, smoothI, bad_data_intervals = [imag_ok_var[0], expected_imag, expected_imag, imag_ok_var[1]], insert_gap = True, remove_gap = False)
	else:
		smoothIInRange = mkinsertgap(pipeline, smoothI, bad_data_intervals = [expected_imag - imag_ok_var, expected_imag, expected_imag, expected_imag + imag_ok_var], insert_gap = True, remove_gap = False)
	smoothIInRange = pipeparts.mkbitvectorgen(pipeline, smoothIInRange, nongap_is_control = True, bit_vector = status_out_smooth)
	smoothIInRange = pipeparts.mkcapsfilter(pipeline, smoothIInRange, "audio/x-raw, format=U32LE, rate=%d, channel-mask=(bitmask)0x0" % starting_rate)
	if starting_rate != ending_rate:
		smoothIInRange = pipeparts.mkgeneric(pipeline, smoothIInRange, "lal_logicalundersample", required_on = status_out_smooth, status_out = status_out_smooth)
		smoothIInRange = pipeparts.mkcapsfilter(pipeline, smoothIInRange, "audio/x-raw, format=U32LE, rate=%d, channel-mask=(bitmask)0x0" % ending_rate)

	smoothInRange = mkgate(pipeline, smoothRInRangetee, mkadder(pipeline, list_srcs(pipeline, smoothIInRange, smoothRInRangetee)), status_out_smooth * 2, attack_length = -min_samples)
	smoothInRange = pipeparts.mkbitvectorgen(pipeline, smoothInRange, nongap_is_control = True, bit_vector = status_out_smooth)
	smoothInRange = pipeparts.mkcapsfilter(pipeline, smoothInRange, "audio/x-raw, format=U32LE, rate=%d, channel-mask=(bitmask)0x0" % ending_rate)

	return smoothInRange

def compute_kappa_bits_only_real(pipeline, smooth, expected, ok_var, min_samples, status_out_smooth = 1, starting_rate=16, ending_rate=16):

	if type(ok_var) is list:
		smoothInRange = mkinsertgap(pipeline, smooth, bad_data_intervals = [ok_var[0], expected, expected, ok_var[1]], insert_gap = True, remove_gap = False)
	else:
		smoothInRange = mkinsertgap(pipeline, smooth, bad_data_intervals = [expected - ok_var, expected, expected, expected + ok_var], insert_gap = True, remove_gap = False)
	smoothInRange = pipeparts.mkbitvectorgen(pipeline, smoothInRange, nongap_is_control = True, bit_vector = status_out_smooth)
	smoothInRange = pipeparts.mkcapsfilter(pipeline, smoothInRange, "audio/x-raw, format=U32LE, rate=%d, channel-mask=(bitmask)0x0" % starting_rate)
	if starting_rate != ending_rate:
		smoothInRange = pipeparts.mkgeneric(pipeline, smoothInRange, "lal_logicalundersample", required_on = status_out_smooth, status_out = status_out_smooth)
		smoothInRange = pipeparts.mkcapsfilter(pipeline, smoothInRange, "audio/x-raw, format=U32LE, rate=%d, channel-mask=(bitmask)0x0" % ending_rate)
	smoothInRangetee = pipeparts.mktee(pipeline, smoothInRange)
	smoothInRange = mkgate(pipeline, smoothInRangetee, smoothInRangetee, status_out_smooth, attack_length = -min_samples)
	smoothInRange = pipeparts.mkbitvectorgen(pipeline, smoothInRange, nongap_is_control = True, bit_vector = status_out_smooth)

	return smoothInRange

def merge_into_complex(pipeline, real, imag):
	# Merge real and imag into one complex channel with complex caps
	head = mkinterleave(pipeline, list_srcs(pipeline, real, imag))
	head = pipeparts.mktogglecomplex(pipeline, head)
	return head

def split_into_real(pipeline, complex_chan):
	# split complex channel with complex caps into two channels (real and imag) with real caps

	complex_chan = pipeparts.mktee(pipeline, complex_chan)
	real = pipeparts.mkgeneric(pipeline, complex_chan, "creal")
	imag = pipeparts.mkgeneric(pipeline, complex_chan, "cimag")

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

	head = mkpow(pipeline, head, exponent = -1)

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
	S2 = mkadder(pipeline, list_srcs(pipeline, mkpow(pipeline, SR, exponent=2.0), mkpow(pipeline, SI, exponent=2.0)))
	kc = mkmultiplier(pipeline, list_srcs(pipeline, S2, mkpow(pipeline, SR, exponent=-1.0)))
	return kc

def compute_fcc(pipeline, SR, SI, fpcal2):

	#
	# f_cc = - (Re[S]/Im[S]) * fpcal2
	#

	fcc = mkmultiplier(pipeline, list_srcs(pipeline, pipeparts.mkaudioamplify(pipeline, SR, -1.0*fpcal2), mkpow(pipeline, SI, exponent=-1.0)))
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
	fpcal4_over_fcc = pipeparts.mkaudioamplify(pipeline, mkpow(pipeline, fcc, exponent = -1.0), fpcal4)
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
	fpcal4_over_fcc = pipeparts.mkaudioamplify(pipeline, mkpow(pipeline, fcc, exponent = -1.0), fpcal4)
	i_fpcal4_over_fcc = pipeparts.mktogglecomplex(pipeline, pipeparts.mkmatrixmixer(pipeline, fpcal4_over_fcc, matrix = [[0, 1]]))
	i_fpcal4_over_fcc_plus_one = pipeparts.mkgeneric(pipeline, i_fpcal4_over_fcc, "lal_add_constant", value = 1.0)
	i_fpcal4_over_fcc_plus_one_inv = complex_inverse(pipeline, i_fpcal4_over_fcc_plus_one)
	Xi_plus_one = mkmultiplier(pipeline, list_srcs(pipeline, EP11, complex_kc, i_fpcal4_over_fcc_plus_one_inv, pcal_over_derr_res))
	Xi = pipeparts.mkgeneric(pipeline, Xi_plus_one, "lal_add_constant", value = -1.0)

	return Xi

def update_filter(filter_maker, arg, filter_taker, maker_prop_name, taker_prop_name):
	firfilter = filter_maker.get_property(maker_prop_name)[::-1]
	filter_taker.set_property(taker_prop_name, firfilter)

def update_filters(filter_maker, arg, filter_taker, maker_prop_name, taker_prop_name, filter_number):
	firfilter = filter_maker.get_property(maker_prop_name)[filter_number][::-1]
	filter_taker.set_property(taker_prop_name, firfilter)

def clean_data(pipeline, signal, signal_rate, witnesses, witness_rate, fft_length, fft_overlap, num_ffts, update_samples, fir_length, frequency_resolution, obsready = None, attack_length = 0, chop_time = 0.0, wait_time = 0.0, filename = None):

	#
	# Use witness channels that monitor the environment to remove environmental noise
	# from a signal of interest.  This function accounts for potential correlation
	# between witness channels.
	#

	default_fir_filter = numpy.zeros(fir_length)

	signal_tee = pipeparts.mktee(pipeline, signal)
	witnesses = list(witnesses)
	witness_tees = []
	for i in range(0, len(witnesses)):
		witnesses[i] = mkresample(pipeline, witnesses[i], 5, False, witness_rate)
		witnesses[i] = highpass(pipeline, witnesses[i], witness_rate)
		witness_tees.append(pipeparts.mktee(pipeline, witnesses[i]))

	resampled_signal = mkresample(pipeline, signal_tee, 5, False, witness_rate)
	transfer_functions = mkinterleave(pipeline, numpy.insert(witness_tees, 0, resampled_signal, axis = 0))
	if obsready is not None:
		transfer_functions = mkgate(pipeline, transfer_functions, obsready, 1, attack_length = attack_length)
	if(chop_time):
		transfer_functions = pipeparts.mkgeneric(pipeline, transfer_functions, "lal_insertgap", chop_length = int(1000000000 * chop_time))
	transfer_functions = pipeparts.mkgeneric(pipeline, transfer_functions, "lal_transferfunction", fft_length = fft_length, fft_overlap = fft_overlap, num_ffts = num_ffts, update_samples = update_samples, make_fir_filters = -1, fir_length = fir_length, frequency_resolution = frequency_resolution, high_pass = 9, update_after_gap = True, filename = filename)
	signal_minus_noise = [signal_tee]
	for i in range(0, len(witnesses)):
		minus_noise = pipeparts.mkgeneric(pipeline, mkqueue(pipeline, witness_tees[i], min_length = wait_time + 0.5), "lal_tdwhiten", kernel = default_fir_filter, latency = fir_length / 2, taper_length = 20 * fir_length)
		transfer_functions.connect("notify::fir-filters", update_filters, minus_noise, "fir_filters", "kernel", i)
		signal_minus_noise.append(mkresample(pipeline, minus_noise, 5, False, signal_rate))

	return mkadder(pipeline, tuple(signal_minus_noise))




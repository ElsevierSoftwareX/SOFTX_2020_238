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

def mkqueue(pipeline, head, length, min_length = 0):
	if length < 0:
		return head
	else:
		return pipeparts.mkqueue(pipeline, head, max_size_time = int(1000000000 * length), max_size_buffers = 0, max_size_bytes = 0, min_threshold_time = min_length)

def mkcomplexqueue(pipeline, head, length, min_length = 0):
	head = pipeparts.mktogglecomplex(pipeline, head)
	head = mkqueue(pipeline, head, length, min_length = min_length)
	head = pipeparts.mktogglecomplex(pipeline, head)
	return head
	
def mkinsertgap(pipeline, head, bad_data_intervals = [-1e35, -1e-35, 1e-35, 1e35], insert_gap = False, remove_gap = True, replace_value = 0, fill_discont = True, block_duration = Gst.SECOND):
	return pipeparts.mkgeneric(pipeline, head, "lal_insertgap", bad_data_intervals = bad_data_intervals, insert_gap = insert_gap, remove_gap = remove_gap, replace_value = replace_value, fill_discont = fill_discont, block_duration = int(block_duration))

def mkupsample(pipeline, head, new_caps):
	head = pipeparts.mkgeneric(pipeline, head, "lal_constantupsample")
	head = pipeparts.mkcapsfilter(pipeline, head, new_caps)
	return head

def mkresample(pipeline, head, caps):
	head = pipeparts.mkresample(pipeline, head, quality = 9)
	head = pipeparts.mkcapsfilter(pipeline, head, caps)
	return head

def mkmultiplier(pipeline, srcs, sync = True):
	elem = pipeparts.mkgeneric(pipeline, None, "lal_adder", sync=sync, mix_mode="product")
	if srcs is not None:
		for src in srcs:
			src.link(elem)
	return elem

def mkadder(pipeline, srcs, sync = True):
	elem = pipeparts.mkgeneric(pipeline, None, "lal_adder", sync=sync)
	if srcs is not None:
		for src in srcs:
			src.link(elem)
	return elem

def mkinterleave(pipeline, src1, src2, queue_length1, queue_length2):
	chan1 = pipeparts.mkmatrixmixer(pipeline, src1, matrix=[[1,0]])
	chan2 = pipeparts.mkmatrixmixer(pipeline, src2, matrix=[[0,1]])
	elem = mkadder(pipeline, list_srcs(pipeline, mkqueue(pipeline, chan1, queue_length1), mkqueue(pipeline, chan2, queue_length2))) 
	#elem = pipeparts.mkgeneric(pipeline, None, "interleave")
	#if srcs is not None:
	#	for src in srcs:
	#		pipeparts.mkqueue(pipeline, src).link(elem)
	return elem


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
#	head = pipeparts.mkgeneric(pipeline, head, "splitcounter", name = channel_name)

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

def compute_kappa_bits(pipeline, smoothR, smoothI, dqR, dqI, expected_real, expected_imag, real_ok_var, imag_ok_var, queue_length1, queue_length2, status_out_smooth = 1, status_out_median = 1, starting_rate=16, ending_rate=16):

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

	smoothInRange = pipeparts.mkgate(pipeline, mkqueue(pipeline, smoothRInRangetee, queue_length1), threshold = status_out_smooth * 2, control = mkqueue(pipeline, mkadder(pipeline, list_srcs(pipeline, mkqueue(pipeline, smoothIInRange, queue_length2), mkqueue(pipeline, smoothRInRangetee, queue_length1))), queue_length2))
	smoothInRange = pipeparts.mkbitvectorgen(pipeline, smoothInRange, nongap_is_control = True, bit_vector = status_out_smooth)
	smoothInRange = pipeparts.mkcapsfilter(pipeline, smoothInRange, "audio/x-raw, format=U32LE, rate=%d, channel-mask=(bitmask)0x0" % ending_rate)

	dqtotal = mkadder(pipeline, list_srcs(pipeline, mkqueue(pipeline, dqR, queue_length1), mkqueue(pipeline, dqI, queue_length2)))
	medianUncorrupt = pipeparts.mkbitvectorgen(pipeline, dqtotal, threshold = 2, bit_vector = status_out_median)
	medianUncorrupt = pipeparts.mkcapsfilter(pipeline, medianUncorrupt, "audio/x-raw, format=U32LE, rate=%d, channel-mask=(bitmask)0x0" % starting_rate)
	if starting_rate != ending_rate:
		medianUncorrupt = pipeparts.mkgeneric(pipeline, medianUncorrupt, "lal_logicalundersample", required_on = status_out_median, status_out = status_out_median)
		medianUncorrupt = pipeparts.mkcapsfilter(pipeline, medianUncorrupt, "audio/x-raw, format=U32LE, rate = %d, channel-mask=(bitmask)0x0" % ending_rate)

	return smoothInRange, medianUncorrupt

def compute_kappa_bits_only_real(pipeline, smooth, dq, expected, ok_var, status_out_smooth = 1, status_out_median = 1, starting_rate=16, ending_rate=16):

	smoothInRange = mkinsertgap(pipeline, smooth, bad_data_intervals = [expected - ok_var, expected, expected, expected + ok_var], insert_gap = True, remove_gap = False)
	smoothInRange = pipeparts.mkbitvectorgen(pipeline, smoothInRange, nongap_is_control = True, bit_vector = status_out_smooth)
	smoothInRange = pipeparts.mkcapsfilter(pipeline, smoothInRange, "audio/x-raw, format=U32LE, rate=%d, channel-mask=(bitmask)0x0" % starting_rate)
	if starting_rate != ending_rate:
		smoothInRange = pipeparts.mkgeneric(pipeline, smoothInRange, "lal_logicalundersample", required_on = status_out_smooth, status_out = status_out_smooth)
		smoothInRange = pipeparts.mkcapsfilter(pipeline, smoothInRange, "audio/x-raw, format=U32LE, rate=%d, channel-mask=(bitmask)0x0" % ending_rate)

	medianUncorrupt = pipeparts.mkbitvectorgen(pipeline, dq, threshold = 1, bit_vector = status_out_median)
	medianUncorrupt = pipeparts.mkcapsfilter(pipeline, medianUncorrupt, "audio/x-raw, format=U32LE, rate=%d, channel-mask=(bitmask)0x0" % starting_rate)
	if starting_rate != ending_rate:
		medianUncorrupt = pipeparts.mkgeneric(pipeline, medianUncorrupt, "lal_logicalundersample", required_on = status_out_median, status_out = status_out_median)
		medianUncorrupt = pipeparts.mkcapsfilter(pipeline, medianUncorrupt, "audio/x-raw, format=U32LE, rate=%d, channel-mask=(bitmask)0x0" % ending_rate)

	return smoothInRange, medianUncorrupt

def merge_into_complex(pipeline, real, imag, queue_length1, queue_length2):
	# Merge real and imag into one complex channel with complex caps
	head = mkinterleave(pipeline, real, imag, queue_length1, queue_length2)
	head = pipeparts.mktogglecomplex(pipeline,head)
	return head

def split_into_real(pipeline, complex_chan):
	# split complex channel with complex caps into two channels (real and imag) with real caps
	elem = pipeparts.mktogglecomplex(pipeline, complex_chan)
	elem = pipeparts.mkgeneric(pipeline, elem, "deinterleave", keep_positions=True)
	real = pipeparts.mkgeneric(pipeline, None, "identity")
	pipeparts.src_deferred_link(elem, "src_0", real.get_static_pad("sink"))
	
	imag = pipeparts.mkgeneric(pipeline, None, "identity")
	pipeparts.src_deferred_link(elem, "src_1", imag.get_static_pad("sink"))
	return real, imag

def demodulate(pipeline, head, freq, td, caps, integration_samples, queue_length1, queue_length2):
	# demodulate input at a given frequency freq

	head = pipeparts.mkgeneric(pipeline, head, "lal_demodulate", line_frequency = freq)
	headR, headI = split_into_real(pipeline, head)
	headR = pipeparts.mkresample(pipeline, headR)
	headR = pipeparts.mkcapsfilter(pipeline, headR, caps)
	headR = pipeparts.mkfirbank(pipeline, headR, fir_matrix=[numpy.hanning(integration_samples + 1)], time_domain = td)
	headI = pipeparts.mkresample(pipeline, headI)
	headI = pipeparts.mkcapsfilter(pipeline, headI, caps)
	headI = pipeparts.mkfirbank(pipeline, headI, fir_matrix=[numpy.hanning(integration_samples + 1)], time_domain = td)
	head = merge_into_complex(pipeline, headR, headI, queue_length1, queue_length2)

	# headR, headI = split_into_real(pipeline, headtee)
	# headR = pipeparts.mkresample(pipeline, headR)
	# headR = pipeparts.mkcapsfilter(pipeline, headR, "audio/x-raw, rate=512")
	# headR = pipeparts.mkaudioamplify(pipeline, headR, amplification = 1.0/16384.0);
	# headR = pipeparts.mkfirbank(pipeline, headR, fir_matrix=[numpy.hanning(integration_samples+1)], time_domain = td)
	# headI = pipeparts.mkresample(pipeline, headI)
	# headI = pipeparts.mkcapsfilter(pipeline, headI, "audio/x-raw, rate=512")
	# headI = pipeparts.mkaudioamplify(pipeline, headI, amplification = -1.0/16384.0);
	# headI = pipeparts.mkfirbank(pipeline, headI, fir_matrix=[numpy.hanning(integration_samples+1)], time_domain = td)
	# #pipeparts.mknxydumpsink(pipeline, headR, "real_new.txt")
	# #pipeparts.mknxydumpsink(pipeline, headI, "imag_new.txt")
	# head2 = merge_into_complex(pipeline, headR, headI)
	# pipeparts.mknxydumpsink(pipeline, head2, "old_method_demod.dump")

	return head

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

def complex_division(pipeline, a, b, queue_length1, queue_length2):
	# Perform complex division of c = a/b and output the complex quotient c

	bInv = complex_inverse(pipeline, b)
	c = mkmultiplier(pipeline, list_srcs(pipeline, mkqueue(pipeline, a, queue_length1), mkqueue(pipeline, bInv, queue_length2)))

	return c

def compute_kappatst_from_filters_file(pipeline, derrfesd, tstexcfesd, pcalfdarm, derrfdarm, ktstfacR, ktstfacI, queue_length1, queue_length2):

	#	       
	# \kappa_TST = ktstfac * (derrfesd/tstexcfesd) * (pcalfdarm/derrfdarm)
	# ktstfac = -(1/A0fesd) * (C0fdarm/(1+G0fdarm)) * ((1+G0fesd)/C0fesd)
	#

	derrfdarminv = complex_inverse(pipeline, derrfdarm)
	tstexcfesdinv = complex_inverse(pipeline, tstexcfesd)
	ktst = mkmultiplier(pipeline, list_srcs(pipeline, mkqueue(pipeline, pcalfdarm, queue_length2), mkqueue(pipeline, derrfdarminv, queue_length1), mkqueue(pipeline, tstexcfesdinv, queue_length1), mkqueue(pipeline, derrfesd, queue_length1)))
	ktst = complex_audioamplify(pipeline, ktst, ktstfacR, ktstfacI)

	return ktst

def compute_kappatst(pipeline, derrfesd, tstexcfesd, pcalfdarm, derrfdarm, ktstfac, queue_length1, queue_length2):

	#	       
	# \kappa_TST = ktstfac * (derrfesd/tstexcfesd) * (pcalfdarm/derrfdarm)
	# ktstfac = -(1/A0fesd) * (C0fdarm/(1+G0fdarm)) * ((1+G0fesd)/C0fesd)
	#

	derrfdarminv = complex_inverse(pipeline, derrfdarm)
	tstexcfesdinv = complex_inverse(pipeline, tstexcfesd)
	ktst = mkmultiplier(pipeline, list_srcs(pipeline, mkqueue(pipeline, ktstfac, queue_length1), mkqueue(pipeline, pcalfdarm, queue_length2), mkqueue(pipeline, derrfdarminv, queue_length1), mkqueue(pipeline, tstexcfesdinv, queue_length1), mkqueue(pipeline, derrfesd, queue_length1)))

	return ktst

def compute_afctrl_from_filters_file(pipeline, derrfpu, excfpu, pcalfdarm, derrfdarm, afctrlfacR, afctrlfacI, queue_length1, queue_length2):
	
	#
	# A(f_ctrl) = -afctrlfac * (derrfpu/excfpu) * (pcalfdarm/derrfdarm)
	# afctrlfac = C0fpcal/(1+G0fpcal) * (1+G0fctrl)/C0fctrl
	#

	derrfdarminv = complex_inverse(pipeline, derrfdarm)
	excfpuinv = complex_inverse(pipeline, excfpu)
	afctrl = mkmultiplier(pipeline, list_srcs(pipeline, mkqueue(pipeline, pcalfdarm, queue_length2), mkqueue(pipeline, derrfdarminv, queue_length1), mkqueue(pipeline, excfpuinv, queue_length1), mkqueue(pipeline, derrfpu, queue_length1)))
	afctrl = complex_audioamplify(pipeline, afctrl, -1.0*afctrlfacR, -1.0*afctrlfacI)

	return afctrl
	

def compute_afctrl(pipeline, derrfpu, excfpu, pcalfdarm, derrfdarm, afctrlfac, queue_length1, queue_length2):

	#
	# A(f_ctrl) = -afctrlfac * (derrfpu/excfpu) * (pcalfdarm/derrfdarm)
	# afctrlfac = EP2 = C0fpcal/(1+G0fpcal) * (1+G0fctrl)/C0fctrl
	#

	derrfdarminv = complex_inverse(pipeline, derrfdarm)
	excfpuinv = complex_inverse(pipeline, excfpu)
	afctrl = mkmultiplier(pipeline, list_srcs(pipeline, mkqueue(pipeline, afctrlfac, queue_length1), mkqueue(pipeline, pcalfdarm, queue_length2), mkqueue(pipeline, derrfdarminv, queue_length1), mkqueue(pipeline, excfpuinv, queue_length1), mkqueue(pipeline, derrfpu, queue_length1)))
	afctrl = complex_audioamplify(pipeline, afctrl, -1.0, 0.0)

	return afctrl

def compute_kappapu_from_filters_file(pipeline, EP3R, EP3I, afctrl, ktst, EP4R, EP4I, queue_length1, queue_length2):

	#
	# \kappa_pu = EP3 * (afctrl - ktst * EP4)
	#

	kpu = complex_audioamplify(pipeline, mkadder(pipeline, list_srcs(pipeline, mkqueue(pipeline, afctrl, queue_length2), mkqueue(pipeline, complex_audioamplify(pipeline, ktst, -1.0*EP4R, -1.0*EP4I), queue_length1))), EP3R, EP3I)	

	return kpu

def compute_kappapu(pipeline, EP3, afctrl, ktst, EP4, queue_length1, queue_length2):
	
	#
	# \kappa_pu = EP3 * (afctrl - ktst * EP4)
	#

	ep4_kappatst = mkmultiplier(pipeline, list_srcs(pipeline, mkqueue(pipeline, ktst, queue_length2), mkqueue(pipeline, complex_audioamplify(pipeline, EP4, -1.0, 0.0), queue_length1)))
	afctrl_kappatst = mkadder(pipeline, list_srcs(pipeline, mkqueue(pipeline, afctrl, queue_length2), mkqueue(pipeline, ep4_kappatst, queue_length1)))
	kpu = mkmultiplier(pipeline, list_srcs(pipeline, mkqueue(pipeline, EP3, queue_length1), mkqueue(pipeline, afctrl_kappatst, queue_length2)))

	return kpu

def compute_kappaa_from_filters_file(pipeline, afctrl, EP4R, EP4I, EP5R, EP5I):

	#
	#\kappa_a = afctrl / (EP4+EP5)
	#

	facR = (EP4R + EP5R) / ((EP4R + EP5R)**2 + (EP4I + EP5I)**2)
	facI = -(EP4I + EP5I) / ((EP4R + EP5R)**2 + (EP4I + EP5I)**2)

	ka = complex_audioamplify(pipeline, afctrl, facR, facI) 

	return ka

def compute_kappaa(pipeline, afctrl, EP4, EP5, queue_length1, queue_length2):

	#
	#\kappa_a = afctrl / (EP4 + EP5)
	#

	ka = complex_division(pipeline, afctrl, mkadder(pipeline, list_srcs(pipeline, mkqueue(pipeline, EP4, queue_length1), mkqueue(pipeline, EP5, queue_length2))), queue_length2, queue_length1)

	return ka


def compute_S_from_filters_file(pipeline, EP6R, EP6I, pcalfpcal2, derrfpcal2, EP7R, EP7I, ktst, EP8R, EP8I, kpu, EP9R, EP9I, queue_length1, queue_length2):
	
	#	
	# S = 1/EP6 * ( pcalfpcal2/derrfpcal2 - EP7*(ktst*EP8 + kpu*EP9) ) ^ (-1)
	#

	pcal_over_derr = complex_division(pipeline, pcalfpcal2, derrfpcal2, queue_length2, queue_length1)
	ep8_kappatst = complex_audioamplify(pipeline, ktst, EP8R, EP8I)
	ep9_kappapu = complex_audioamplify(pipeline, kpu, EP9R, EP9I)
	kappatst_kappapu = mkadder(pipeline, list_srcs(pipeline, mkqueue(pipeline, ep8_kappatst, queue_length1), mkqueue(pipeline, ep9_kappapu, queue_length2)))
	kappatst_kappapu = complex_audioamplify(pipeline, kappatst_kappapu,  -1.0*EP7R, -1.0*EP7I)
	Sinv = mkadder(pipeline, list_srcs(pipeline, mkqueue(pipeline, pcal_over_derr, queue_length2), mkqueue(pipeline, kappatst_kappapu, queue_length1)))
	Sinv = complex_audioamplify(pipeline, Sinv, EP6R, EP6I)
	S = complex_inverse(pipeline, Sinv)
	
	return S

def compute_S(pipeline, EP6, pcalfpcal2, derrfpcal2, EP7, ktst, EP8, kpu, EP9, queue_length1, queue_length2):

	#	
	# S = 1/EP6 * ( pcalfpcal2/derrfpcal2 - EP7*(ktst*EP8 + kpu*EP9) ) ^ (-1)
	#

	pcal_over_derr = complex_division(pipeline, pcalfpcal2, derrfpcal2, queue_length2, queue_length1)
	ep8_kappatst = mkmultiplier(pipeline, list_srcs(pipeline, mkqueue(pipeline, ktst, queue_length2), mkqueue(pipeline, EP8, queue_length1)))
	ep9_kappapu = mkmultiplier(pipeline, list_srcs(pipeline, mkqueue(pipeline, kpu, queue_length2), mkqueue(pipeline, EP9, queue_length1)))
	kappatst_kappapu = mkadder(pipeline, list_srcs(pipeline, mkqueue(pipeline, ep8_kappatst, queue_length1), mkqueue(pipeline, ep9_kappapu, queue_length2)))
	kappatst_kappapu = mkmultiplier(pipeline, list_srcs(pipeline, mkqueue(pipeline, complex_audioamplify(pipeline, EP7, -1.0, 0.0), queue_length1), mkqueue(pipeline, kappatst_kappapu, queue_length2)))
	Sinv = mkadder(pipeline, list_srcs(pipeline, mkqueue(pipeline, pcal_over_derr, queue_length2), mkqueue(pipeline, kappatst_kappapu, queue_length1)))
	Sinv = mkmultiplier(pipeline, list_srcs(pipeline, mkqueue(pipeline, EP6, queue_length1), mkqueue(pipeline, Sinv, queue_length2)))
	S = complex_inverse(pipeline, Sinv)

	return S

def compute_kappac(pipeline, SR, SI, queue_length1, queue_length2):

	#
	# \kappa_C = |S|^2 / Re[S]
	#

	SR = pipeparts.mktee(pipeline, SR)
	S2 = mkadder(pipeline, list_srcs(pipeline, mkqueue(pipeline, pipeparts.mkpow(pipeline, SR, exponent=2.0), queue_length1), mkqueue(pipeline, pipeparts.mkpow(pipeline, SI, exponent=2.0), queue_length2)))
	kc = mkmultiplier(pipeline, list_srcs(pipeline, mkqueue(pipeline, S2, queue_length2), mkqueue(pipeline, pipeparts.mkpow(pipeline, SR, exponent=-1.0), queue_length1)))
	return kc

def compute_fcc(pipeline, SR, SI, fpcal2, queue_length1, queue_length2):
	#
	# f_cc = - (Re[S]/Im[S]) * fpcal2
	#

	fcc = mkmultiplier(pipeline, list_srcs(pipeline, mkqueue(pipeline, pipeparts.mkaudioamplify(pipeline, SR, -1.0*fpcal2), queue_length1), mkqueue(pipeline, pipeparts.mkpow(pipeline, SI, exponent=-1.0), queue_length2)))
	return fcc


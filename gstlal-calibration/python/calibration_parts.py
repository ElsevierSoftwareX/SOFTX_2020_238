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
# Functions for making sure no channels are missing from the frames
#

def gate_strain_for_output_frames(pipeline, head, control):
	controltee = pipeparts.mktee(pipeline, control)
	control = mkaudiorate(pipeline, mkqueue(pipeline, controltee))
	head = pipeparts.mkgate(pipeline, mkqueue(pipeline, head), control = control, threshold = 0, leaky = True)
	control = mkqueue(pipeline, controltee)
	return head, control

def gate_other_with_strain(pipeline, other, strain):
	other = pipeparts.mkgate(pipeline, mkqueue(pipeline, other), control = mkqueue(pipeline, strain), threshold=0, leaky = True)
	other = mkaudiorate(pipeline, other)
	return other

#
# Shortcut functions for common element combos/properties
#

def mkqueue(pipeline, head):
	return pipeparts.mkqueue(pipeline, head, max_size_time = 0, max_size_buffers = 0, max_size_bytes = 0)

def mkcomplexqueue(pipeline, head):
	head = pipeparts.mktogglecomplex(pipeline, head)
	head = mkqueue(pipeline, head)
	head = pipeparts.mktogglecomplex(pipeline, head)
	return head
	
def mkaudiorate(pipeline, head):
	head = pipeparts.mkaudiorate(pipeline, head, skip_to_first = True, silent = False)
	head = mkreblock(pipeline, head)
	return head

def mkreblock(pipeline, head):
	return pipeparts.mkreblock(pipeline, head, block_duration = Gst.SECOND)

def mkupsample(pipeline, head, new_caps):
	head = pipeparts.mkgeneric(pipeline, head, "lal_constantupsample")
	head = pipeparts.mkcapsfilter(pipeline, head, new_caps)
	#head = mkaudiorate(pipeline, head)
	#head = pipeparts.mktee(pipeline, head)
	return head

def mkresample(pipeline, head, caps):
	head = pipeparts.mkresample(pipeline, head, quality = 9)
	head = pipeparts.mkcapsfilter(pipeline, head, caps)
	#head = mkaudiorate(pipeline, head)
	return head

def mkmultiplier(pipeline, srcs, sync = True, Complex = False):
	elem = pipeparts.mkgeneric(pipeline, None, "lal_adder", sync=sync, mix_mode="product")
	if srcs is not None:
		for src in srcs:
			if not Complex:
				mkqueue(pipeline, src).link(elem)
			else:
				mkcomplexqueue(pipeline, src).link(elem)
	return elem

def mkinterleave(pipeline, src1, src2):
	chan1 = pipeparts.mkmatrixmixer(pipeline, src1, matrix=[[1,0]])
	chan2 = pipeparts.mkmatrixmixer(pipeline, src2, matrix=[[0,1]])
	elem = mkadder(pipeline, list_srcs(pipeline, mkqueue(pipeline, chan1), mkqueue(pipeline, chan2))) 
	#elem = pipeparts.mkgeneric(pipeline, None, "interleave")
	#if srcs is not None:
	#	for src in srcs:
	#		pipeparts.mkqueue(pipeline, src).link(elem)
	#elem = pipeparts.mkaudiorate(pipeline, elem)
	return elem

def mkadder(pipeline, srcs, sync = True, Complex = False):
	elem = pipeparts.mkgeneric(pipeline, None, "lal_adder", sync=sync)
	if srcs is not None:
		for src in srcs:
			if not Complex:
				mkqueue(pipeline, src).link(elem)
			else:
				mkcomplexqueue(pipeline, src).link(elem)
	return elem

#
# Write a pipeline graph function
#

def write_graph(demux, pipeline, name):
	pipeparts.write_dump_dot(pipeline, "%s.%s" % (name, "PLAYING"), verbose = True)

#
# Common element combo functions
#

def hook_up_and_reblock(pipeline, demux, channel_name, instrument):
	head = mkqueue(pipeline, None)
	pipeparts.src_deferred_link(demux, "%s:%s" % (instrument, channel_name), head.get_static_pad("sink"))
	head = mkreblock(pipeline, head)
	return head

def caps_and_progress(pipeline, head, caps, progress_name):
	head = pipeparts.mkaudioconvert(pipeline, head)
	head = pipeparts.mkcapsfilter(pipeline, head, caps)
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

def smooth_kappas(pipeline, head, var, expected, Nav, N):
	# Find median of calibration factors array with size N and smooth out medians with an average over Nav samples
	head = mkaudiorate(pipeline, head)
	head = pipeparts.mkgeneric(pipeline, head, "lal_smoothkappas", maximum_offset = var, kappa_ceiling = 0.01, default_kappa = expected, array_size = N)
	head = pipeparts.mkfirbank(pipeline, head, fir_matrix = [numpy.ones(Nav)/Nav])
	return head

def compute_kappa_bits(pipeline, averageok, raw, smoothR, smoothI, expected_real, expected_imag, real_ok_var, imag_ok_var, caps, status_out_raw = 1, status_out_smooth = 1, status_out_overall = 1, starting_rate=16, ending_rate=16):
	rawR, rawI = split_into_real(pipeline, raw, caps)

	rawRInRange = pipeparts.mkgeneric(pipeline, rawR, "lal_add_constant", value = -expected_real)
	rawRInRange = pipeparts.mkbitvectorgen(pipeline, rawRInRange, threshold = real_ok_var, invert_control = True, bit_vector = status_out_raw)
	rawRInRange = pipeparts.mkcapsfilter(pipeline, rawRInRange, "audio/x-raw, format=U32LE, rate=%d" % starting_rate)
	rawRInRange = pipeparts.mkgeneric(pipeline, rawRInRange, "lal_logicalundersample", required_on = status_out_raw, status_out = status_out_raw)
	rawRInRange = pipeparts.mkcapsfilter(pipeline, rawRInRange, "audio/x-raw, format=U32LE, rate=%d" % ending_rate)
	rawRInRangetee = pipeparts.mktee(pipeline, rawRInRange)

	rawIInRange = pipeparts.mkgeneric(pipeline, rawI, "lal_add_constant", value = -expected_imag)
	rawIInRange = pipeparts.mkbitvectorgen(pipeline, rawIInRange, threshold = imag_ok_var, invert_control = True, bit_vector = status_out_raw)
	rawIInRange = pipeparts.mkcapsfilter(pipeline, rawIInRange, "audio/x-raw, format=U32LE, rate=%d" % starting_rate)
	rawIInRange = pipeparts.mkgeneric(pipeline, rawIInRange, "lal_logicalundersample", required_on = status_out_raw, status_out = status_out_raw)
	rawIInRange = pipeparts.mkcapsfilter(pipeline, rawIInRange, "audio/x-raw, format=U32LE, rate=%d" % ending_rate)

	rawInRange = pipeparts.mkgate(pipeline, mkqueue(pipeline, rawRInRangetee), threshold = status_out_raw * 2, control = mkadder(pipeline, list_srcs(pipeline, rawIInRange, mkqueue(pipeline, rawRInRange))))

	smoothRInRange = pipeparts.mkgeneric(pipeline, smoothR, "lal_add_constant", value = -expected_real)
	smoothRInRange = pipeparts.mkbitvectorgen(pipeline, smoothRInRange, threshold = real_ok_var, invert_control = True, bit_vector = status_out_smooth)
	smoothRInRange = pipeparts.mkcapsfilter(pipeline, smoothRInRange, "audio/x-raw, format=U32LE, rate=%d" % starting_rate)
	smoothRInRange = pipeparts.mkgeneric(pipeline, smoothRInRange, "lal_logicalundersample", required_on = status_out_smooth, status_out = status_out_smooth)
	smoothRInRange = pipeparts.mkcapsfilter(pipeline, smoothRInRange, "audio/x-raw, format=U32LE, rate=%d" % ending_rate)
	smoothRInRangetee = pipeparts.mktee(pipeline, smoothRInRange)

	smoothIInRange = pipeparts.mkgeneric(pipeline, smoothI, "lal_add_constant", value = -expected_imag)
	smoothIInRange = pipeparts.mkbitvectorgen(pipeline, smoothIInRange, threshold = imag_ok_var, invert_control = True, bit_vector = status_out_smooth)
	smoothIInRange = pipeparts.mkcapsfilter(pipeline, smoothIInRange, "audio/x-raw, format=U32LE, rate=%d" % starting_rate)
	smoothIInRange = pipeparts.mkgeneric(pipeline, smoothIInRange, "lal_logicalundersample", required_on = status_out_smooth, status_out = status_out_smooth)
	smoothIInRange = pipeparts.mkcapsfilter(pipeline, smoothIInRange, "audio/x-raw, format=U32LE, rate=%d" % ending_rate)

	smoothInRange = pipeparts.mkgate(pipeline, mkqueue(pipeline, smoothRInRangetee), threshold = status_out_raw * 2, control = mkadder(pipeline, list_srcs(pipeline, smoothIInRange, mkqueue(pipeline, smoothRInRange))))

	overall = pipeparts.mkgate(pipeline, mkqueue(pipeline, smoothRInRangetee), threshold = status_out_raw + status_out_smooth + 1, control = mkadder(pipeline, list_srcs(pipeline, rawInRange, smoothInRange, averageok)))
	overall = pipeparts.mkbitvectorgen(pipeline, overall, threshold = status_out_smooth, bit_vector = status_out_overall)

	return rawInRange, smoothInRange, overall

def compute_kappa_bits_only_real(pipeline, averageok, raw, smooth, expected, ok_var, status_out_raw = 1, status_out_smooth = 1, status_out_overall = 1, starting_rate=16, ending_rate=16):

	rawInRange = pipeparts.mkgeneric(pipeline, raw, "lal_add_constant", value = -expected)
	rawInRange = pipeparts.mkbitvectorgen(pipeline, rawInRange, threshold = ok_var, invert_control = True, bit_vector = status_out_raw)
	rawInRange = pipeparts.mkcapsfilter(pipeline, rawInRange, "audio/x-raw, format=U32LE, rate=%d" % starting_rate)
	rawInRange = pipeparts.mkgeneric(pipeline, rawInRange, "lal_logicalundersample", required_on = status_out_raw, status_out = status_out_raw)
	rawInRange = pipeparts.mkcapsfilter(pipeline, rawInRange, "audio/x-raw, format=U32LE, rate=%d" % ending_rate)
	rawInRangetee = pipeparts.mktee(pipeline, rawInRange)

	smoothInRange = pipeparts.mkgeneric(pipeline, smooth, "lal_add_constant", value = -expected)
	smoothInRange = pipeparts.mkbitvectorgen(pipeline, smoothInRange, threshold = ok_var, invert_control = True, bit_vector = status_out_smooth)
	smoothInRange = pipeparts.mkcapsfilter(pipeline, smoothInRange, "audio/x-raw, format=U32LE, rate=%d" % starting_rate)
	smoothInRange = pipeparts.mkgeneric(pipeline, smoothInRange, "lal_logicalundersample", required_on = status_out_smooth, status_out = status_out_smooth)
	smoothInRange = pipeparts.mkcapsfilter(pipeline, smoothInRange, "audio/x-raw, format=U32LE, rate=%d" % ending_rate)
	smoothInRangetee = pipeparts.mktee(pipeline, smoothInRange)

	overall = pipeparts.mkgate(pipeline, mkqueue(pipeline, smoothInRangetee), threshold = status_out_raw + status_out_smooth + 1, control = mkadder(pipeline, list_srcs(pipeline, rawInRange, smoothInRange, averageok)))
	overall = pipeparts.mkbitvectorgen(pipeline, overall, threshold = status_out_smooth, bit_vector = status_out_overall)

	return rawInRange, smoothInRangetee, overall

def merge_into_complex(pipeline, real, imag):
	# Merge real and imag into one complex channel with complex caps
	head = mkinterleave(pipeline, real, imag)
	head = pipeparts.mktogglecomplex(pipeline,head)
	return head

def split_into_real(pipeline, complex_chan, real_caps):
	# split complex channel with complex caps into two channels (real and imag) with real caps
	elem = pipeparts.mktogglecomplex(pipeline, complex_chan)
	elem = pipeparts.mkgeneric(pipeline, elem, "deinterleave", keep_positions=True)
	real = pipeparts.mkaudiorate(pipeline, None)
	pipeparts.src_deferred_link(elem, "src_0", real.get_static_pad("sink"))
	real = pipeparts.mkcapsfilter(pipeline, real, real_caps)
	
	imag = pipeparts.mkaudiorate(pipeline, None)
	pipeparts.src_deferred_link(elem, "src_1", imag.get_static_pad("sink"))
	imag = pipeparts.mkcapsfilter(pipeline, imag, real_caps)
	return real, imag

def demodulate(pipeline, head, freq, td, caps):
	# demodulate input at a given frequency freq

	head = pipeparts.mkgeneric(pipeline, head, "lal_demodulate", line_frequency = freq)
	headtee = pipeparts.mktee(pipeline, head)

	head = pipeparts.mktogglecomplex(pipeline, headtee)
	head = pipeparts.mkresample(pipeline, head)
	head = pipeparts.mkcapsfilter(pipeline, head, caps)
	head = pipeparts.mkgeneric(pipeline, head, "audiocheblimit", cutoff = 0.05)
	head = pipeparts.mkmatrixmixer(pipeline, head, matrix=[[1.0,0.0],[0.0,-1.0]])
	head = pipeparts.mktogglecomplex(pipeline, head)

	""""
	headR, headI = split_into_real(pipeline, headtee)
	headR = pipeparts.mkresample(pipeline, headR)
	headR = pipeparts.mkcapsfilter(pipeline, headR, "audio/x-raw, rate=512")
	headR = pipeparts.mkaudioamplify(pipeline, headR, amplification = 1.0/16384.0);
	headR = pipeparts.mkfirbank(pipeline, headR, fir_matrix=[numpy.hanning(integration_samples+1)], time_domain = td)
	headI = pipeparts.mkresample(pipeline, headI)
	headI = pipeparts.mkcapsfilter(pipeline, headI, "audio/x-raw, rate=512")
	headI = pipeparts.mkaudioamplify(pipeline, headI, amplification = -1.0/16384.0);
	headI = pipeparts.mkfirbank(pipeline, headI, fir_matrix=[numpy.hanning(integration_samples+1)], time_domain = td)
	#pipeparts.mknxydumpsink(pipeline, headR, "real_new.txt")
	#pipeparts.mknxydumpsink(pipeline, headI, "imag_new.txt")
	head2 = merge_into_complex(pipeline, headR, headI)
	pipeparts.mknxydumpsink(pipeline, head2, "old_method_demod.dump")
	"""

	return head

def complex_audioamplify(pipeline, chan, WR, WI):
	# Multiply a complex channel chan by a complex number WR+I WI
	# Re[out] = -chanI*WI + chanR*WR
	# Im[out] = chanR*WI + chanI*WR

	head = pipeparts.mktogglecomplex(pipeline, chan)
	head = pipeparts.mkmatrixmixer(pipeline, head, matrix=[[WR, WI],[-WI, WR]])
	head = pipeparts.mktogglecomplex(pipeline, head)

	return head

def complex_division(pipeline, a, b):
	# Perform complex division of c = a/b and output the complex quotient c

	#b = pipeparts.mkcapsfilter(pipeline, b, "audio/x-raw, format=Z128LE, rate=16384, channels=1, width=128, endianness=1234")
	b = pipeparts.mktogglecomplex(pipeline, b)
	#bInv = b
	bInv = pipeparts.mkgeneric(pipeline, b, "complex_pow", exponent = -1)
	bInv = pipeparts.mkcapsfilter(pipeline, bInv, "audio/x-raw, format=F64LE, rate=16384, channels=2")
	bInv = pipeparts.mktogglecomplex(pipeline, bInv)
	#bInv = pipeparts.mkcapsfilter(pipeline, bInv, "audio/x-raw, format=Z128LE, rate=16384")

	c = mkmultiplier(pipeline, list_srcs(pipeline, a, bInv))

	return c

def compute_kappatst_from_filters_file(pipeline, derrfesd, tstexcfesd, pcalfdarm, derrfdarm, ktstfacR, ktstfacI):

	#               
	# \kappa_TST = ktstfac * (derrfesd/tstexcfesd) * (pcalfdarm/derrfdarm)
	# ktstfac = -(1/A0fesd) * (C0fdarm/(1+G0fdarm)) * ((1+G0fesd)/C0fesd)
	#

	ktst = mkmultiplier(pipeline, list_srcs(pipeline, complex_audioamplify(pipeline, complex_division(pipeline, derrfesd, tstexcfesd), ktstfacR, ktstfacI), complex_division(pipeline, pcalfdarm, derrfdarm)))

	return ktst

def compute_kappatst(pipeline, derrfesd, tstexcfesd, pcalfdarm,  derrfdarm, ktstfac):

	#               
	# \kappa_TST = ktstfac * (derrfesd/tstexcfesd) * (pcalfdarm/derrfdarm)
	# ktstfac = -(1/A0fesd) * (C0fdarm/(1+G0fdarm)) * ((1+G0fesd)/C0fesd)
	#

	ktst = mkmultiplier(pipeline, list_srcs(pipeline, ktstfac, complex_division(pipeline, derrfesd, tstexcfesd), complex_division(pipeline, pcalfdarm, derrfdarm)))

	return ktst

def compute_afctrl_from_filters_file(pipeline, derrfpu, excfpu, pcalfdarm, derrfdarm, afctrlfacR, afctrlfacI):
	
	#
	# A(f_ctrl) = -afctrlfac * (derrfpu/excfpu) * (pcalfdarm/derrfdarm)
	# afctrlfac = C0fpcal/(1+G0fpcal) * (1+G0fctrl)/C0fctrl
	#

	afctrl = mkmultiplier(pipeline, list_srcs(pipeline, complex_audioamplify(pipeline, complex_division(pipeline, derrfpu, excfpu), -1.0*afctrlfacR, -1.0*afctrlfacI), complex_division(pipeline, pcalfdarm, derrfdarm)))


	return afctrl
	

def compute_afctrl(pipeline, derrfpu, excfpu, pcalfdarm, derrfdarm, afctrlfac):

	#
	# A(f_ctrl) = -afctrlfac * (derrfpu/excfpu) * (pcalfdarm/derrfdarm)
	# afctrlfac = C0fpcal/(1+G0fpcal) * (1+G0fctrl)/C0fctrl
	#

	afctrl = mkmultiplier(pipeline, list_srcs(pipeline, complex_audioamplify(pipeline, afctrlfac, -1.0, 0.0), complex_division(pipeline, derrfpu, excfpu), complex_division(pipeline, pcalfdarm, derrfdarm)))

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

	kpu = mkmultiplier(pipeline, list_srcs(pipeline, EP3, mkadder(pipeline, list_srcs(pipeline, afctrl, mkmultiplier(pipeline, list_srcs(pipeline, ktst, complex_audioamplify(pipeline, EP4, -1.0, -0.0)))))))

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

	Sinv = complex_audioamplify(pipeline, mkadder(pipeline, list_srcs(pipeline, complex_division(pipeline, pcalfpcal2, derrfpcal2), complex_audioamplify(pipeline, mkadder(pipeline, list_srcs(pipeline, complex_audioamplify(pipeline, ktst, EP8R, EP8I), complex_audioamplify(pipeline, kpu, EP9R, EP9I))), -1.0*EP7R, -1.0*EP7I))), EP6R, EP6I)
	Sinv = pipeparts.mktogglecomplex(pipeline, Sinv)
	S = pipeparts.mkgeneric(pipeline, Sinv, "complex_pow", exponent = -1)
	S = pipeparts.mktogglecomplex(pipeline, S)
	
	return S

def compute_S(pipeline, EP6, pcalfpcal2, derrfpcal2, EP7, ktst, EP8, kpu, EP9):

	#	
	# S = 1/EP6 * ( pcalfpcal2/derrfpcal2 - EP7*(ktst*EP8 + kpu*EP9) ) ^ (-1)
	#

	Sinv = mkmultiplier(pipeline, list_srcs(pipeline, EP6, mkadder(pipeline, list_srcs(pipeline, complex_division(pipeline, pcalfpcal2, derrfpcal2), mkmultiplier(pipeline, list_srcs(pipeline, complex_audioamplify(pipeline, EP7, -1.0, 0.0), mkadder(pipeline, list_srcs(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, ktst, EP8)), mkmultiplier(pipeline, list_srcs(pipeline, kpu, EP9))))))))))
	Sinv = pipeparts.mktogglecomplex(pipeline, Sinv)
	S = pipeparts.mkgeneric(pipeline, Sinv, "complex_pow", exponent = -1.0)
	S = pipeparts.mktogglecomplex(pipeline, S) 

	return S

def compute_kappac(pipeline, SR, SI):

	#
	# \kappa_C = |S|^2 / Re[S]
	#

	SR = pipeparts.mktee(pipeline, SR)
	SI = pipeparts.mktee(pipeline, SI)
	# FIXME: The pow element behaving very strangely... I think there might be an issue where it is modifying buffers or parts of memory space that it shouldn't be touching. This is why I've avoided using the pow element via some very strange tricsk below.
	S2 = mkadder(pipeline, list_srcs(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, SR, SR)), mkmultiplier(pipeline, list_srcs(pipeline, SI, SI))))
	#S2 = mkadder(pipeline, list_srcs(pipeline, pipeparts.mkpow(pipeline, SR, exponent=2.0), pipeparts.mkpow(pipeline, SI, exponent=2.0)))
	SRinv = pipeparts.mkmatrixmixer(pipeline, SR, matrix=[[1.0, 0.0]])
	SRinv = pipeparts.mkgeneric(pipeline, SRinv, "complex_pow", exponent = -1.0)
	SRinv = pipeparts.mkmatrixmixer(pipeline, SRinv, matrix=[[1.0],[0.0]])
	#kc = mkmultiplier(pipeline, list_srcs(pipeline, S2, pipeparts.mkpow(pipeline, pipeparts.mkqueue(pipeline, SR), exponent=-1.0)))
	kc = mkmultiplier(pipeline, list_srcs(pipeline, S2, SRinv))
	return kc

def compute_fcc(pipeline, SR, SI, fpcal2):
	#
	# f_cc = - (Re[S]/Im[S]) * fpcal2
	#

	SIinv = pipeparts.mkmatrixmixer(pipeline, SI, matrix = [[1.0, 0.0]])
	SIinv = pipeparts.mkgeneric(pipeline, SIinv, "complex_pow", exponent = -1.0)
	SIinv = pipeparts.mkmatrixmixer(pipeline, SIinv, matrix=[[1.0],[0.0]])

	#fcc = mkmultiplier(pipeline, list_srcs(pipeline, pipeparts.mkaudioamplify(pipeline, SR, -1.0), pipeparts.mkpow(pipeline, pipeparts.mkqueue(pipeline, SI), exponent=-1.0)))
	fcc = mkmultiplier(pipeline, list_srcs(pipeline, pipeparts.mkaudioamplify(pipeline, SR, -1.0*fpcal2), SIinv))
	return fcc

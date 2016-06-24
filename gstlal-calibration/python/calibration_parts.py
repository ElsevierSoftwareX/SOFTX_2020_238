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
	
def mkaudiorate(pipeline, head):
	head = pipeparts.mkaudiorate(pipeline, head, skip_to_first = True, silent = False)
	head = mkreblock(pipeline, head)
	return head

def mkreblock(pipeline, head):
	return pipeparts.mkreblock(pipeline, head, block_duration = Gst.SECOND)

def mkupsample(pipeline, head, new_caps):
	head = pipeparts.mkgeneric(pipeline, head, "lal_constant_upsample")
	head = pipeparts.mkcapsfilter(pipeline, head, new_caps)
	head = mkaudiorate(pipeline, head)
	#head = pipeparts.mktee(pipeline, head)
	return head

def mkresample(pipeline, head, caps):
	head = pipeparts.mkresample(pipeline, head, quality = 9)
	head = pipeparts.mkcapsfilter(pipeline, head, caps)
	#head = mkaudiorate(pipeline, head)
	return head

def mkmultiplier(pipeline, srcs, caps, sync = True):
	#elem = pipeparts.mkgeneric(pipeline, None, "lal_adder", sync=sync, caps=Gst.Caps(caps), mix_mode="product")
	elem = pipeparts.mkgeneric(pipeline, None, "lal_adder", sync=sync, mix_mode="product")
	if srcs is not None:
		for src in srcs:
			mkqueue(pipeline, src).link(elem)
	elem = pipeparts.mkcapsfilter(pipeline, elem, caps)
	return elem

def mkinterleave(pipeline, srcs):
	elem = pipeparts.mkgeneric(pipeline, None, "interleave")
	if srcs is not None:
		for src in srcs:
			pipeparts.mkqueue(pipeline, src).link(elem)
	return elem

def mkadder(pipeline, srcs, caps, sync = True):
	elem = pipeparts.mkgeneric(pipeline, None, "lal_adder", sync=sync)
	if srcs is not None:
		for src in srcs:
			mkqueue(pipeline, src).link(elem)
	elem = pipeparts.mkcapsfilter(pipeline, elem, caps)
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
	#head = pipeparts.mkprogressreport(pipeline, head, name="progress_src_%s" % progress_name)
	head = pipeparts.mkprogressreport(pipeline, head, "test")
	return head


#
# Function to make a list of heads to pass to, i.e. the multiplier or adder
#

def list_srcs(pipeline, *args):
	out = []
	for src in args:
		out.append(mkqueue(pipeline, src))
	return tuple(out)

#
# Calibration factor related functions
#

def average_calib_factors(pipeline, head, var, expected, N, caps, Nav):
	# Find median of calibration factors array and smooth out medians with an average over Nav samples
	head = pipeparts.mkaudioconvert(pipeline, head)
	head = pipeparts.mkcapsfilter(pipeline, head, caps)
	head = mkaudiorate(pipeline, head)
	head = pipeparts.mkgeneric(pipeline, head, "lal_smoothcalibfactors", max_value = expected + var, min_value = expected-var, default_val = expected, max_size = N)
	head = pipeparts.mkfirbank(pipeline, head, fir_matrix = [numpy.ones(Nav)/Nav])
	return head

def merge_into_complex(pipeline, real, imag):
	# Merge real and imag into one complex channel with complex caps
	head = mkinterleave(pipeline, list_srcs(pipeline, real, imag))
	head = mkaudiorate(pipeline, head) # This audiorate is necessary! Probaly once lal_interleave gets fixed it won't be
	head = pipeparts.mktogglecomplex(pipeline,head)
	return head

def split_into_real(pipeline, complex_chan, real_caps, complex_caps):
	# split complex channel with complex caps into two channels (real and imag) with real caps
	elem = pipeparts.mktogglecomplex(pipeline, complex_chan)
	elem = pipeparts.mkgeneric(pipeline, elem, "deinterleave")
	real = pipeparts.mkaudioconvert(pipeline, None)
	pipeparts.src_deferred_link(elem, "src0", real.get_static_pad("sink"))
	real = pipeparts.mkcapsfilter(pipeline, real, real_caps)
	
	imag = pipeparts.mkaudioconvert(pipeline, None)
	pipeparts.src_deferred_link(elem, "src1", imag.get_static_pad("sink"))
	imag = pipeparts.mkcapsfilter(pipeline, imag, real_caps)
	return real, imag

def demodulate(pipeline, head, sr, freq, orig_caps, new_caps, integration_samples, td):
	# demodulate input at a given frequency freq

	head = pipeparts.mkgeneric(pipeline, head, "lal_demodulate", line_frequency = freq)
	head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw, format=Z128LE")

	#headR = pipeparts.mkfirbank(pipeline, headR, fir_matrix=[numpy.hanning(integration_samples+1)], time_domain = td)

	return head

def complex_audioamplify(pipeline, chan, WR, WI):
	# Multiply a complex channel chan by a complex number WR+I WI
	# Re[out] = -chanI*WI + chanR*WR
	# Im[out] = chanR*WI + chanI*WR

	head = pipeparts.mktogglecomplex(pipeline, chan)
	head = pipeparts.mkmatrixmixer(pipeline, head, matrix=[[WR, WI],[-WI, WR]])
	head = pipeparts.mktogglecomplex(pipeline, head)

	return head

def complex_division(pipeline, a, b, caps):
	# Perform complex division of c = a/b and output the complex quotient c

	b = pipeparts.mktogglecomplex(pipeline, b)
	bInv = pipeparts.mkgeneric(pipeline, b, "complex_pow", exponent = -1)
	bInv = pipeparts.mktogglecomplex(pipeline, bInv)

	c = mkmultiplier(pipeline, list_srcs(pipeline, a, bInv), caps)

	return c

def compute_kappatst_from_filters_file(pipeline, derrfesd, tstexcfesd, pcalfdarm, derrfdarm, ktstfacR, ktstfacI, complex_caps):

	#               
	# \kappa_TST = ktstfac * (derrfesd/tstexcfesd) * (pcalfdarm/derrfdarm)
	# ktstfac = -(1/A0fesd) * (C0fdarm/(1+G0fdarm)) * ((1+G0fesd)/C0fesd)
	#

	ktst = mkmultiplier(pipeline, list_srcs(pipeline, complex_audioamplify(pipeline, complex_division(pipeline, derrfesd, tstexcfesd, complex_caps), ktstfacR, ktstfacI), complex_division(pipeline, pcalfdarm, derrfdarm, complex_caps)), complex_caps)

	return ktst

def compute_kappatst(pipeline, derrfesd, tstexcfesd, pcalfdarm,  derrfdarm, ktstfac, complex_caps):

	#               
	# \kappa_TST = ktstfac * (derrfesd/tstexcfesd) * (pcalfdarm/derrfdarm)
	# ktstfac = -(1/A0fesd) * (C0fdarm/(1+G0fdarm)) * ((1+G0fesd)/C0fesd)
	#

	ktst = mkmultiplier(pipeline, list_srcs(pipeline, ktstfac, complex_division(pipeline, derrfesd, tstexcfesd, complex_caps), complex_division(pipeline, pcalfdarm, derrfdarm, complex_caps)), complex_caps)

	return ktst

def compute_afctrl_from_filters_file(pipeline, derrfpu, excfpu, pcalfdarm, derrfdarm, afctrlfacR, afctrlfacI, complex_caps):
	
	#
	# A(f_ctrl) = -afctrlfac * (derrfpu/excfpu) * (pcalfdarm/derrfdarm)
	# afctrlfac = C0fpcal/(1+G0fpcal) * (1+G0fctrl)/C0fctrl
	#

	afctrl = mkmultiplier(pipeline, list_srcs(pipeline, complex_audioamplify(pipeline, complex_division(pipeline, derrfpu, excfpu, complex_caps), -1.0*afctrlR, -1.0*afctrlI), complex_division(pipeline, pcalfdarm, derrfdarm, complex_caps)), complex_caps)


	return afctrl
	

def compute_afctrl(pipeline, derrfpu, excfpu, pcalfdarm, derrfdarm, afctrlfac, complex_caps):

	#
	# A(f_ctrl) = -afctrlfac * (derrfpu/excfpu) * (pcalfdarm/derrfdarm)
	# afctrlfac = C0fpcal/(1+G0fpcal) * (1+G0fctrl)/C0fctrl
	#

	afctrl = mkmultiplier(pipeline, list_srcs(pipeline, complex_audioamplify(pipeline, afctrlfac, -1.0, 0.0), complex_division(pipeline, derrfpu, excfpu, complex_caps), complex_division(pipeline, pcalfdarm, derrfdarm, complex_caps)), complex_caps)

	return afctrl

def compute_kappapu_from_filters_file(pipeline, EP3R, EP3I, afctrl, ktst, EP4R, EP4I, complex_caps):

	#
	# \kappa_pu = EP3 * (afctrl - ktst * EP4)
	#

	kpu = complex_audioamplify(pipeline, mkadder(pipeline, list_srcs(pipeline, afctrl, complex_audioamplify(pipeline, ktst, -1.0*EP4R, -1.0*EP4I)), complex_caps), EP3R, EP3I)	

	return kpu

def compute_kappapu(pipeline, EP3, afctrl, ktst, EP4,  complex_caps):
	
	#
	# \kappa_pu = EP3 * (afctrl - ktst * EP4)
	#

	kpu = mkmultiplier(pipeline, list_srcs(pipeline, EP3, mkadder(pipeline, list_srcs(pipeline, afctrl, mkmultiplier(pipeline, list_srcs(pipeline, ktst, complex_audioamplify(pipeline, EP4, -1.0, -0.0)), complex_caps)), complex_caps)), complex_caps)

	return kpu

def compute_kappaa_from_filters_file(pipeline, afctrl, EP4R, EP4I, EP5R, EP5I):

	#
	#\kappa_a = afctrl / (EP4+EP5)
	#

	facR = (EP4R + EP5R) / ((EP4R + EP5R)**2 + (EP4I + EP5I)**2)
	facI = -(EP4I + EP5I) / ((EP4R + EP5R)**2 + (EP4I + EP5I)**2)

	ka = complex_audioamplify(pipeline, afctrl, facR, facI) 

	return ka

def compute_kappaa(pipeline, afctrl, EP4, EP5,  complex_caps):

	#
	#\kappa_a = afctrl / (EP4 + EP5)
	#

	ka = complex_division(pipeline, afctrl, mkadder(pipeline, list_srcs(pipeline, EP4, EP5), complex_caps), complex_caps)

	return ka


def compute_S_from_filters_file(pipeline, EP6R, EP6I, pcalfpcal2, derrfpcal2, EP7R, EP7I, ktst, EP8R, EP8I, kpu, EP9R, EP9I, complex_caps):
	
	#	
	# S = 1/EP6 * ( pcalfpcal2/derrfpcal2 - EP7*(ktst*EP8 + kpu*EP9) ) ^ (-1)
	#

	Sinv = complex_audioamplify(pipeline, mkadder(pipeline, list_srcs(pipeline, complex_division(pipeline, pcalfpcal2, derrfpcal2, complex_caps), complex_audioamplify(pipeline, mkadder(pipeline, list_srcs(pipeline, complex_audioamplify(pipeline, ktst, EP8R, EP8I), complex_audioamplify(pipeline, kpu, EP9R, EP9I)), complex_caps), -1.0*EP7R, -1.0*EP7I)), complex_caps), EP6R, EP6I)
	Sinv = pipeparts.mktogglecomplex(pipeline, Sinv)
	S = pipeparts.mkgeneric(pipeline, Sinv, "complex_pow", exponent = -1)
	S = pipeparts.mktogglecomplex(pipeline, S)
	
	return S

def compute_S(pipeline, EP6, pcalfpcal2, derrfpcal2, EP7, ktst, EP8, kpu, EP9, complex_caps):

	#	
	# S = 1/EP6 * ( pcalfpcal2/derrfpcal2 - EP7*(ktst*EP8 + kpu*EP9) ) ^ (-1)
	#

	Sinv = mkmultiplier(pipeline, list_srcs(pipeline, EP6, mkadder(pipeline, list_srcs(pipeline, complex_division(pipeline, pcalfpcal2, derrfpcal2, complex_caps), mkmultiplier(pipeline, list_srcs(pipeline, complex_audioamplify(pipeline, EP7, -1.0, 0.0), mkadder(pipeline, list_srcs(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, ktst, EP8), complex_caps), mkmultiplier(pipeline, list_srcs(pipeline, kpu, EP9), complex_caps)), complex_caps)), complex_caps)), complex_caps)), complex_caps)
	Sinv = pipeparts.mktogglecomplex(pipeline, Sinv)
	S = pipeparts.mkgeneric(pipeline, Sinv, "complex_pow", exponent = -1)
	S = pipeparts.mktogglecomplex(pipeline, S) 

	return S

def compute_kappac(pipeline, SR, SI, caps):

	#
	# \kappa_C = |S|^2 / Re[S]
	#

	SR = pipeparts.mktee(pipeline, SR)
	S2 = mkadder(pipeline, list_srcs(pipeline, pipeparts.mkpow(pipeline, SR, exponent=2.0), pipeparts.mkpow(pipeline, SI, exponent=2.0)), caps)
	kc = mkmultiplier(pipeline, list_srcs(pipeline, S2, pipeparts.mkpow(pipeline, SR, exponent=-1.0)), caps)
	return kc

def compute_fcc(pipeline, SR, SI, fpcal2, caps):
	#
	# f_cc = - (Re[S]/Im[S]) * fpcal2
	#
	fcc = mkmultiplier(pipeline, list_srcs(pipeline, pipeparts.mkaudioamplify(pipeline, SR, -1.0), pipeparts.mkpow(pipeline, SI, exponent=-1.0)), caps)
	fcc = pipeparts.mkaudioamplify(pipeline, fcc, fpcal2)
	return fcc

def average_and_check_range(pipeline, factor, variance_real, variance_imag, expected_value_real, expected_value_imag, median_array_size, median_smoothing_samples, caps, complex_caps):

	real, imag = split_into_real(pipeline, factor, caps, complex_caps)

	# Produce a channel that says whether lal_check_calib_factors will compute a good value or not.  Use this in the statevector bit for this \kappa
	realInRange = pipeparts.mkgeneric(pipeline, mkaudiorate(pipeline, mkqueue(pipeline, real)), "lal_smoothcalibfactors", max_value = expected_value_real + variance_real, min_value = expected_value_real - variance_real, default_val = expected_value_real, statevector = True, max_size = median_array_size)
	imagInRange = pipeparts.mkgeneric(pipeline, mkaudiorate(pipeline, mkqueue(pipeline, real)), "lal_smoothcalibfactors", max_value = expected_value_imag + variance_imag, min_value = expected_value_imag - variance_imag, default_val = expected_value_imag, statevector = True, max_size = median_array_size)

	realOut = average_calib_factors(pipeline, mkqueue(pipeline, real), variance_real, expected_value_real, median_array_size, caps, median_smoothing_samples)
	imagOut = average_calib_factors(pipeline, mkqueue(pipeline, imag), variance_imag, expected_value_imag, median_array_size, caps, median_smoothing_samples)

	realOuttee = pipeparts.mktee(pipeline, realOut)
	imagOuttee = pipeparts.mktee(pipeline, imagOut)

	return realInRange, imagInRange, realOuttee, imagOuttee

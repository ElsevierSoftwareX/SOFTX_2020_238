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
import gst
import numpy

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

def mkqueue(pipeline, head):
	return pipeparts.mkqueue(pipeline, head, max_size_time = 0, max_size_buffers = 0, max_size_bytes = 0)
	
def mkaudiorate(pipeline, head):
	head = pipeparts.mkaudiorate(pipeline, head, skip_to_first = True, silent = False)
	head = mkreblock(pipeline, head)
	return head

def mkreblock(pipeline, head):
	return pipeparts.mkreblock(pipeline, head, block_duration = gst.SECOND)

def write_graph(demux, pipeline, name):
	pipeparts.write_dump_dot(pipeline, "%s.%s" % (name, "PLAYING"), verbose = True)

def hook_up_and_reblock(pipeline, demux, channel_name, instrument):
	head = mkqueue(pipeline, None)
	pipeparts.src_deferred_link(demux, "%s:%s" % (instrument, channel_name), head.get_pad("sink"))
	head = mkreblock(pipeline, head)
	return head

def caps_and_progress(pipeline, head, caps, progress_name):
	head = pipeparts.mkaudioconvert(pipeline, head)
	head = pipeparts.mkcapsfilter(pipeline, head, caps)
	head = pipeparts.mkprogressreport(pipeline, head, "progress_src_%s" % progress_name)
	return head

def caps_and_progress_and_resample(pipeline, head, caps, progress_name, new_caps):
	head = pipeparts.mkaudioconvert(pipeline, head)
	head = pipeparts.mkcapsfilter(pipeline, head, caps)
	head = pipeparts.mkprogressreport(pipeline, head, "progress_src_%s" % progress_name)
	head = pipeparts.mkresample(pipeline, head, quality = 9)
	head = pipeparts.mkcapsfilter(pipeline, head, new_caps)
	head = mkaudiorate(pipeline, head)
	return head

def caps_and_progress_and_upsample(pipeline, head, caps, progress_name, new_caps):
	head = pipeparts.mkaudioconvert(pipeline, head)
	head = pipeparts.mkcapsfilter(pipeline, head, caps)
	head = pipeparts.mkprogressreport(pipeline, head, "progress_src_%s" % progress_name)
	head = pipeparts.mkgeneric(pipeline, head, "lal_constant_upsample")
	head = pipeparts.mkcapsfilter(pipeline, head, new_caps)
	head = mkaudiorate(pipeline, head)
	head = pipeparts.mktee(pipeline, head)
	return head

def resample(pipeline, head, caps):
	head = pipeparts.mkresample(pipeline, head, quality = 9)
	head = pipeparts.mkcapsfilter(pipeline, head, caps)
	head = mkaudiorate(pipeline, head)
	#head = mkreblock(pipeline, head)
	return head

def mkmultiplier(pipeline, srcs, caps, sync = True):
	elem = pipeparts.mkgeneric(pipeline, None, "lal_multiplier", sync=sync)
	if srcs is not None:
		for src in srcs:
			pipeparts.mkcapsfilter(pipeline, src, caps).link(elem)
	elem = pipeparts.mkcapsfilter(pipeline, elem, caps)
	return elem

def mkinterleave(pipeline, srcs, caps):
	elem = pipeparts.mkgeneric(pipeline, None, "lal_interleave", sync = True)
	if srcs is not None:
		for src in srcs:
			pipeparts.mkcapsfilter(pipeline, src, caps).link(elem)
	return elem

def mkadder(pipeline, srcs, caps, sync = True):
	elem = pipeparts.mkgeneric(pipeline, None, "lal_adder", sync=sync, caps=gst.Caps(caps))
	if srcs is not None:
		for src in srcs:
			pipeparts.mkcapsfilter(pipeline, src, caps).link(elem)
	elem = pipeparts.mkcapsfilter(pipeline, elem, caps)
	return elem

def list_srcs(pipeline, *args):
	out = []
	for src in args:
		out.append(mkqueue(pipeline, src))
	return tuple(out)

def average_calib_factors(pipeline, head, var, expected, N, caps, hold_time, Nav):
	head = pipeparts.mkaudioconvert(pipeline, head)
	head = pipeparts.mkcapsfilter(pipeline, head, caps)
	head = pipeparts.mkgeneric(pipeline, head, "lal_check_calib_factors", variance = var, default = expected, wait_time_to_new_expected = hold_time, median_array_size = N)
	head = pipeparts.mkfirbank(pipeline, head, fir_matrix = [numpy.ones(Nav)/Nav])
	return head

def merge_into_complex(pipeline, real, imag, real_caps, complex_caps):
	head = mkinterleave(pipeline, list_srcs(pipeline, real, imag), real_caps)
	head = pipeparts.mkaudioconvert(pipeline, head)
	head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw-float")
	head = mkaudiorate(pipeline, head) # This audiorate is necessary! Probaly once lal_interleave gets fixed it won't be
	head = pipeparts.mktogglecomplex(pipeline,head)
	head = pipeparts.mkcapsfilter(pipeline, head, complex_caps)
	return head

def split_into_real(pipeline, complex, real_caps, complex_caps):
	elem = pipeparts.mkcapsfilter(pipeline, complex, complex_caps)
	elem = pipeparts.mktogglecomplex(pipeline, elem)
	elem = pipeparts.mkcapsfilter(pipeline, elem, "audio/x-raw-float, channels=2")
	elem = pipeparts.mkgeneric(pipeline, elem, "deinterleave")
	real = pipeparts.mkaudioconvert(pipeline, None)
	pipeparts.src_deferred_link(elem, "src0", real.get_pad("sink"))
	real = pipeparts.mkcapsfilter(pipeline, real, real_caps)
	
	imag = pipeparts.mkaudioconvert(pipeline, None)
	pipeparts.src_deferred_link(elem, "src1", imag.get_pad("sink"))
	imag = pipeparts.mkcapsfilter(pipeline, imag, real_caps)
	return real, imag

def demodulate(pipeline, head, sr, freq, orig_caps, new_caps, integration_samples, td):
	headtee = pipeparts.mktee(pipeline, head)
	deltat = 1.0/float(sr)
	cos = pipeparts.mkgeneric(pipeline, mkqueue(pipeline, headtee), "lal_numpy_fx_transform", expression = "%f * cos(2.0 * 3.1415926535897931 * %f * t)" % (deltat, freq))
	sin = pipeparts.mkgeneric(pipeline, mkqueue(pipeline, headtee), "lal_numpy_fx_transform", expression = "-1.0 * %f * sin(2.0 * 3.1415926535897931 * %f * t)" % (deltat, freq))

	headR = mkmultiplier(pipeline, (mkqueue(pipeline, headtee), cos), orig_caps)
	headR = resample(pipeline, headR, new_caps)
	headR = pipeparts.mkfirbank(pipeline, headR, fir_matrix=[numpy.hanning(integration_samples+1)], time_domain = td)

	headI = mkmultiplier(pipeline, (mkqueue(pipeline, headtee), sin), orig_caps)
	headI = resample(pipeline, headI, new_caps)
	headI = pipeparts.mkfirbank(pipeline, headI, fir_matrix=[numpy.hanning(integration_samples+1)], time_domain = td)

	return headR, headI

def filter_at_line(pipeline, chanR, chanI, WR, WI, caps):
	# Apply a filter to a demodulated channel at a specific frequency, where the filter at that frequency is Re[W] = WR and Im[W] = WI
	# Re[out] = -chanI*WI + chanR*WR
	# Im[out] = chanR*WI + chanI*WR

	chanR = pipeparts.mktee(pipeline, chanR)
	chanI = pipeparts.mktee(pipeline, chanI)

	chanI_WI = pipeparts.mkaudioamplify(pipeline, chanI, -1.0*WI)
	chanR_WR = pipeparts.mkaudioamplify(pipeline, chanR, WR)
	chanR_WI = pipeparts.mkaudioamplify(pipeline, chanR, WI)
	chanI_WR = pipeparts.mkaudioamplify(pipeline, chanI, WR)

	outR = mkadder(pipeline, list_srcs(pipeline, chanI_WI, chanR_WR), caps)
	outI = mkadder(pipeline, list_srcs(pipeline, chanR_WI, chanI_WR), caps)
	return outR, outI

def compute_pcalfp_over_derrfp(pipeline, derrfpR, derrfpI, pcalfpR, pcalfpI, caps):

	pcalfpRtee = pipeparts.mktee(pipeline, pipeparts.mkcapsfilter(pipeline, pcalfpR, caps))
	pcalfpItee = pipeparts.mktee(pipeline, pipeparts.mkcapsfilter(pipeline, pcalfpI, caps))
	derrfpRtee = pipeparts.mktee(pipeline, pipeparts.mkcapsfilter(pipeline, derrfpR, caps))
	derrfpItee = pipeparts.mktee(pipeline, pipeparts.mkcapsfilter(pipeline, derrfpI, caps))
	derrfp2 = pipeparts.mktee(pipeline, mkadder(pipeline, list_srcs(pipeline, pipeparts.mkpow(pipeline, derrfpRtee, exponent=2.0), pipeparts.mkpow(pipeline, derrfpItee, exponent=2.0)), caps))

	cR1 = mkmultiplier(pipeline, list_srcs(pipeline, derrfpItee, pcalfpItee), caps)
	cR2 = mkmultiplier(pipeline, list_srcs(pipeline, derrfpRtee, pcalfpRtee), caps)
	cR = mkmultiplier(pipeline, list_srcs(pipeline, mkadder(pipeline, list_srcs(pipeline, cR1, cR2), caps), pipeparts.mkpow(pipeline, derrfp2, exponent=-1.0)), caps)
	cI1 = mkmultiplier(pipeline, list_srcs(pipeline, derrfpRtee, pcalfpItee), caps)
	cI2 = mkmultiplier(pipeline, list_srcs(pipeline, derrfpItee, pcalfpRtee), caps)
	cI = mkmultiplier(pipeline, list_srcs(pipeline, mkadder(pipeline, list_srcs(pipeline, cI1, pipeparts.mkaudioamplify(pipeline, cI2, -1.0)), caps), pipeparts.mkpow(pipeline, derrfp2, exponent=-1.0)), caps)
	return cR, cI

def compute_gamma(pipeline, excR, excI, ctrlR, ctrlI, olgR, olgI, WR, WI, real_caps):

	ctrlR, ctrlI = filter_at_line(pipeline, ctrlR, ctrlI, WR, WI, real_caps)

	ctrlR = pipeparts.mktee(pipeline, ctrlR)
	ctrlI = pipeparts.mktee(pipeline, ctrlI)

	exc_over_ctrlR, exc_over_ctrlI = compute_pcalfp_over_derrfp(pipeline, ctrlR, ctrlI, excR, excI, real_caps)
	exc_over_ctrlR = pipeparts.mkaudioconvert(pipeline, exc_over_ctrlR)
	exc_over_ctrlR = pipeparts.mkcapsfilter(pipeline, exc_over_ctrlR, real_caps)

	exc_over_ctrlI = pipeparts.mkaudioconvert(pipeline, exc_over_ctrlI)
	exc_over_ctrlI = pipeparts.mkcapsfilter(pipeline, exc_over_ctrlI, real_caps)

	exc_over_ctrlR_minus_one = pipeparts.mkgeneric(pipeline, exc_over_ctrlR, "lal_add_constant", constant = -1.0)
	exc_over_ctrlR_minus_one = pipeparts.mkaudioconvert(pipeline, exc_over_ctrlR_minus_one)
	exc_over_ctrlR_minus_one = pipeparts.mkcapsfilter(pipeline, exc_over_ctrlR_minus_one, real_caps)

	olginvR = olgR / (olgR*olgR + olgI*olgI)
	olginvI = -olgI /(olgR*olgR + olgI*olgI)

	gammaR, gammaI = filter_at_line(pipeline, exc_over_ctrlR_minus_one, exc_over_ctrlI, olginvR, olginvI, real_caps)
	gammaR = pipeparts.mkaudioconvert(pipeline, gammaR)
	gammaR = pipeparts.mkcapsfilter(pipeline, gammaR, real_caps)
	gammaI = pipeparts.mkaudioconvert(pipeline, gammaI)
	gammaI = pipeparts.mkcapsfilter(pipeline, gammaI, real_caps)

	return gammaR, gammaI
	
def multiply_complex_channel_complex_number(pipeline, channelR, channelI, numberR, numberI, caps):
	channelR = pipeparts.mktee(pipeline, channelR)
	channelI = pipeparts.mktee(pipeline, channelI)
	outR = mkadder(pipeline, list_srcs(pipeline, pipeparts.mkaudioamplify(pipeline, channelR, numberR), pipeparts.mkaudioamplify(pipeline, channelI, -1.0 * numberI)), caps)
	outI = mkadder(pipeline, list_srcs(pipeline, pipeparts.mkaudioamplify(pipeline, channelR, numberI), pipeparts.mkaudioamplify(pipeline, channelI, numberR)), caps)
	return outR, outI

def compute_kappatst_from_filters_file(pipeline, derrfxR, derrfxI, excfxR, excfxI, pcalfp_derrfpR, pcalfp_derrfpI,  ktstfacR, ktstfacI, real_caps, complex_caps):

	derrfx_over_excfxR, derrfx_over_excfxI = compute_pcalfp_over_derrfp(pipeline, excfxR, excfxI, derrfxR, derrfxI, real_caps)
	pcalfp_over_derrfp = merge_into_complex(pipeline, pcalfp_derrfpR, pcalfp_derrfpI, real_caps, complex_caps)

	# 	     
	# \kappa_TST = ktstfac * (derrfx/excfx) * (pcalfp/derrfp)
	# ktstfac = -(1/A0fx) * (C0fp/(1+G0fp)) * ((1+G0fx)/C0fx)
	#

	part1R, part1I = multiply_complex_channel_complex_number(pipeline, derrfx_over_excfxR, derrfx_over_excfxI, ktstfacR, ktstfacI, real_caps)
	part1 = merge_into_complex(pipeline, part1R, part1I, real_caps, complex_caps)
	ktst = mkmultiplier(pipeline, list_srcs(pipeline, part1, pcalfp_over_derrfp), complex_caps)
	ktstR, ktstI = split_into_real(pipeline, ktst, real_caps, complex_caps)
	return ktstR, ktstI

def compute_kappatst(pipeline, derrfxR, derrfxI, excfxR, excfxI, pcalfp_derrfpR, pcalfp_derrfpI,  ktstfacR, ktstfacI, real_caps, complex_caps):

	derrfx_over_excfxR, derrfx_over_excfxI = compute_pcalfp_over_derrfp(pipeline, excfxR, excfxI, derrfxR, derrfxI, real_caps)
	derrfx_over_excfx = merge_into_complex(pipeline, derrfx_over_excfxR, derrfx_over_excfxI, real_caps, complex_caps)
	pcalfp_over_derrfp = merge_into_complex(pipeline, pcalfp_derrfpR, pcalfp_derrfpI, real_caps, complex_caps)
	ktstfac = merge_into_complex(pipeline, ktstfacR, ktstfacI, real_caps, complex_caps)

	# 	     
	# \kappa_TST = ktstfac * (derrfx/excfx) * (pcalfp/derrfp)
	# ktstfac = -(1/A0fx) * (C0fp/(1+G0fp)) * ((1+G0fx)/C0fx)
	#

	ktst = mkmultiplier(pipeline, list_srcs(pipeline, ktstfac, derrfx_over_excfx, pcalfp_over_derrfp), complex_caps)
	ktstR, ktstI = split_into_real(pipeline, ktst, real_caps, complex_caps)
	return ktstR, ktstI

def compute_kappapu_from_filters_file(pipeline, A0pufxinvR, A0pufxinvI, AfctrlR, AfctrlI, ktstR, ktstI, A0tstfxR, A0tstfxI, real_caps, complex_caps):

	ktstfacs = -1.0*(A0tstfxR+1j*A0tstfxI)*(A0pufxinvR+1j*A0pufxinvI)
	ktstfacsR = numpy.real(ktstfacs)
	ktstfacsI = numpy.imag(ktstfacs)

	ktstA0tstfxR, ktstA0tstfxI = multiply_complex_channel_complex_number(pipeline, ktstR, ktstI, ktstfacsR, ktstfacsI, real_caps)
	AfxA0pufxinvR, AfxA0pufxinvI = multiply_complex_channel_complex_number(pipeline, AfctrlR, AfctrlI, A0pufxinvR, A0pufxinvI, real_caps)
	
	# \kappa_pu = (1/A0pufx) * (Afx - ktst * A0tstfx)
	kpuR = mkadder(pipeline, list_srcs(pipeline, AfxA0pufxinvR, ktstA0tstfxR), real_caps)
	kpuI = mkadder(pipeline, list_srcs(pipeline, AfxA0pufxinvI, ktstA0tstfxI), real_caps)	

	return kpuR, kpuI

def compute_kappapu(pipeline, A0pufxinvR, A0pufxinvI, AfctrlR, AfctrlI, ktstR, ktstI, A0tstfxR, A0tstfxI, real_caps, complex_caps):
	ktst = merge_into_complex(pipeline, pipeparts.mkaudioamplify(pipeline, ktstR, -1.0), pipeparts.mkaudioamplify(pipeline, ktstI, -1.0), real_caps, complex_caps)
	A0tstfx = merge_into_complex(pipeline, A0tstfxR, A0tstfxI, real_caps, complex_caps)
	A0pufxinv = merge_into_complex(pipeline, A0pufxinvR, A0pufxinvI, real_caps, complex_caps)
	Afx = merge_into_complex(pipeline, AfctrlR, AfctrlI, real_caps, complex_caps)
	
	# \kappa_pu = (1/A0pufx) * (Afx - ktst * A0tstfx)
	kpu = mkmultiplier(pipeline, list_srcs(pipeline, A0pufxinv, mkadder(pipeline, list_srcs(pipeline, Afx, mkmultiplier(pipeline, list_srcs(pipeline, ktst, A0tstfx), complex_caps)), complex_caps)), complex_caps)
	kpuR, kpuI = split_into_real(pipeline, kpu, real_caps, complex_caps)

	return kpuR, kpuI

def compute_kappaa_from_filters_file(pipeline, AfxR, AfxI, A0tstfxR, A0tstfxI, A0pufxR, A0pufxI,real_caps, complex_caps):

	#\kappa_a = A0fx / (A0tstfx + A0pufx)

	den = 1/(A0tstfxR+A0tstfxI*1j+A0pufxR+A0pufxI*1j)
	facsR = numpy.real(den)
	facsI = numpy.imag(den)
	kaR, kaI = multiply_complex_channel_complex_number(pipeline, AfxR, AfxI, facsR, facsI, real_caps)
	return kaR, kaI

def compute_kappaa(pipeline, AfxR, AfxI, A0tstfxR, A0tstfxI, A0pufxR, A0pufxI,real_caps, complex_caps):
	Afx = merge_into_complex(pipeline, AfxR, AfxI, real_caps, complex_caps)

	#\kappa_a = A0fx / (A0tstfx + A0pufx)

	A0tstfx_plus_A0pufxR = mkadder(pipeline, list_srcs(pipeline, A0tstfxR, A0pufxR), real_caps)
	A0tstfx_plus_A0pufxI = mkadder(pipeline, list_srcs(pipeline, A0tstfxI, A0pufxI), real_caps)
	A0tstfx_plus_A0pufxR = pipeparts.mktee(pipeline, A0tstfx_plus_A0pufxR)
	A0tstfx_plus_A0pufxI = pipeparts.mktee(pipeline, A0tstfx_plus_A0pufxI)
	den2 = mkadder(pipeline, list_srcs(pipeline, pipeparts.mkpow(pipeline, A0tstfx_plus_A0pufxR, exponent=2.0), pipeparts.mkpow(pipeline, A0tstfx_plus_A0pufxI, exponent=2.0)), real_caps)
	den2 = pipeparts.mktee(pipeline, pipeparts.mkpow(pipeline, den2, exponent = -1.0))
	denR = mkmultiplier(pipeline, list_srcs(pipeline, A0tstfx_plus_A0pufxR, den2), real_caps)
	denI = mkmultiplier(pipeline, list_srcs(pipeline, pipeparts.mkaudioamplify(pipeline, A0tstfx_plus_A0pufxI, -1.0), den2), real_caps)

	den = merge_into_complex(pipeline, denR, denI, real_caps, complex_caps)
	ka = mkmultiplier(pipeline, list_srcs(pipeline, Afx, den), complex_caps)
	kaR, kaI = split_into_real(pipeline, ka, real_caps, complex_caps)
	
	return kaR, kaI


def compute_S_from_filters_file(pipeline, CresR, CresI, pcal_derrR, pcal_derrI, ktstR, ktstI, kpuR, kpuI, D0R, D0I, A0tstR, A0tstI, A0puR, A0puI, real_caps, complex_caps):
	
	# S = 1/Cres * ( pcal/derr - D0*(ktst*A0tst + kpu*A0pu) ) ^ (-1)
	
	CresD0A0tst = -1.0*(D0R+1j*D0I)*(A0tstR+1j*A0tstI)*(CresR+1j*CresI)
	CresD0A0tstR = numpy.real(CresD0A0tst)
	CresD0A0tstI = numpy.imag(CresD0A0tst)

	CresD0A0pu = -1.0*(CresR+1j*CresI)*(D0R+1j*D0I)*(A0puR+1j*A0puI)
	CresD0A0puR = numpy.real(CresD0A0pu)
	CresD0A0puI = numpy.imag(CresD0A0pu)
	
	ktstfacsR, ktstfacsI = multiply_complex_channel_complex_number(pipeline, ktstR, ktstI, CresD0A0tstR, CresD0A0tstI, real_caps)
	kpufacsR, kpufacsI = multiply_complex_channel_complex_number(pipeline, kpuR, kpuI, CresD0A0puR, CresD0A0puI, real_caps)
	pcal_derr_facsR, pcal_derr_facsI = multiply_complex_channel_complex_number(pipeline, pcal_derrR, pcal_derrI, CresR, CresI, real_caps)

	ktstfacs = merge_into_complex(pipeline, ktstfacsR, ktstfacsI, real_caps, complex_caps)
	kpufacs = merge_into_complex(pipeline, kpufacsR, kpufacsI, real_caps, complex_caps)
	pcalderrfacs = merge_into_complex(pipeline, pcal_derr_facsR, pcal_derr_facsI, real_caps, complex_caps)
	Sinv = mkadder(pipeline, list_srcs(pipeline, ktstfacs, kpufacs, pcalderrfacs), complex_caps)
	SinvR, SinvI = split_into_real(pipeline, Sinv, real_caps, complex_caps)
	SinvR = pipeparts.mktee(pipeline, SinvR)
	SinvI = pipeparts.mktee(pipeline, SinvI)
	Sinv2 = pipeparts.mktee(pipeline, mkadder(pipeline, list_srcs(pipeline, pipeparts.mkpow(pipeline, SinvR, exponent = 2.0), pipeparts.mkpow(pipeline, SinvI, exponent = 2.0)), real_caps))
	SR = mkmultiplier(pipeline, list_srcs(pipeline, SinvR, pipeparts.mkpow(pipeline, Sinv2, exponent = -1.0)), real_caps)
	SI = mkmultiplier(pipeline, list_srcs(pipeline, pipeparts.mkaudioamplify(pipeline, SinvI, -1.0), pipeparts.mkpow(pipeline, Sinv2, exponent = -1.0)), real_caps)
	return SR, SI

def compute_S(pipeline, CresR, CresI, pcal_derrR, pcal_derrI, ktstR, ktstI, kpuR, kpuI, D0R, D0I, A0tstR, A0tstI, A0puR, A0puI, real_caps, complex_caps):
	Cres = merge_into_complex(pipeline, CresR, CresI, real_caps, complex_caps)
	pcal_derr = merge_into_complex(pipeline, pcal_derrR, pcal_derrI, real_caps, complex_caps)
	ktst = merge_into_complex(pipeline, ktstR, ktstI, real_caps, complex_caps)
	kpu = merge_into_complex(pipeline, kpuR, kpuI, real_caps, complex_caps)
	D0 = merge_into_complex(pipeline, pipeparts.mkaudioamplify(pipeline, D0R, -1.0), pipeparts.mkaudioamplify(pipeline, D0I, -1.0), real_caps, complex_caps)
	A0tst = merge_into_complex(pipeline, A0tstR, A0tstI, real_caps, complex_caps)
	A0pu = merge_into_complex(pipeline, A0puR, A0puI, real_caps, complex_caps)

	#	
	# S = 1/Cres * ( pcal/derr - D0*(ktst*A0tst + kpu*A0pu) ) ^ (-1)

	Sinv = mkmultiplier(pipeline, list_srcs(pipeline, Cres, mkadder(pipeline, list_srcs(pipeline, pcal_derr, mkmultiplier(pipeline, list_srcs(pipeline, D0, mkadder(pipeline, list_srcs(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, ktst, A0tst), complex_caps), mkmultiplier(pipeline, list_srcs(pipeline, kpu, A0pu), complex_caps)), complex_caps)), complex_caps)), complex_caps)), complex_caps)
	SinvR, SinvI = split_into_real(pipeline, Sinv, real_caps, complex_caps)
	SinvR = pipeparts.mktee(pipeline, SinvR)
	SinvI = pipeparts.mktee(pipeline, SinvI)

	Sinv2 = pipeparts.mktee(pipeline, mkadder(pipeline, list_srcs(pipeline, pipeparts.mkpow(pipeline, SinvR, exponent = 2.0), pipeparts.mkpow(pipeline, SinvI, exponent = 2.0)), real_caps))
	SR = mkmultiplier(pipeline, list_srcs(pipeline, SinvR, pipeparts.mkpow(pipeline, Sinv2, exponent = -1.0)), real_caps)
	SI = mkmultiplier(pipeline, list_srcs(pipeline, pipeparts.mkaudioamplify(pipeline, SinvI, -1.0), pipeparts.mkpow(pipeline, Sinv2, exponent = -1.0)), real_caps)

	return SR, SI

def compute_kappac(pipeline, SR, SI, caps):
	#
	# \kappa_C = |S|^2 / Re[S]
	#
	SR = pipeparts.mktee(pipeline, SR)
	S2 = mkadder(pipeline, list_srcs(pipeline, pipeparts.mkpow(pipeline, SR, exponent=2.0), pipeparts.mkpow(pipeline, SI, exponent=2.0)), caps)
	kc = mkmultiplier(pipeline, list_srcs(pipeline, S2, pipeparts.mkpow(pipeline, SR, exponent=-1.0)), caps)
	return kc

def compute_fcc(pipeline, SR, SI, fpcal, caps):
	#
	# f_cc = - (Re[S]/Im[S]) * fpcal
	#
	fcc = mkmultiplier(pipeline, list_srcs(pipeline, pipeparts.mkaudioamplify(pipeline, SR, -1.0), pipeparts.mkpow(pipeline, SI, exponent=-1.0)), caps)
	fcc = pipeparts.mkaudioamplify(pipeline, fcc, fpcal)
	return fcc

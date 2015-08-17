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

def write_graph(demux, pipeline, name):
	pipeparts.write_dump_dot(pipeline, "%s.%s" % (name, "PLAYING"), verbose = True)

def hook_up_and_reblock(pipeline, demux, channel_name, instrument):
	head = pipeparts.mkqueue(pipeline, None, max_size_buffers = 0, max_size_time = gst.SECOND * 100)
	pipeparts.src_deferred_link(demux, "%s:%s" % (instrument, channel_name), head.get_pad("sink"))
	head = pipeparts.mkreblock(pipeline, head, block_duration = gst.SECOND)
	return head

def caps_and_progress(pipeline, head, caps, progress_name):
	head = pipeparts.mkaudiorate(pipeline, head, skip_to_first = True, silent = False)
	head = pipeparts.mkaudioconvert(pipeline, head)
	head = pipeparts.mkcapsfilter(pipeline, head, caps)
	head = pipeparts.mkprogressreport(pipeline, head, "progress_src_%s" % progress_name)
	return head

def caps_and_progress_and_resample(pipeline, head, caps, progress_name, new_caps):
	head = pipeparts.mkaudiorate(pipeline, head, skip_to_first = True, silent = False)
	head = pipeparts.mkaudioconvert(pipeline, head)
	head = pipeparts.mkcapsfilter(pipeline, head, caps)
	head = pipeparts.mkprogressreport(pipeline, head, "progress_src_%s" % progress_name)
	head = pipeparts.mkresample(pipeline, head, quality = 9)
	head = pipeparts.mkcapsfilter(pipeline, head, new_caps)
	return head

def caps_and_progress_and_upsample(pipeline, head, caps, progress_name, new_caps):
	head = pipeparts.mkaudiorate(pipeline, head, skip_to_first = True, silent = False)
	head = pipeparts.mkaudioconvert(pipeline, head)
	head = pipeparts.mkcapsfilter(pipeline, head, caps)
	head = pipeparts.mkprogressreport(pipeline, head, "progress_src_%s" % progress_name)
	head = pipeparts.mkgeneric(pipeline, head, "lal_constant_upsample")
	head = pipeparts.mkcapsfilter(pipeline, head, new_caps)
	return head

def resample(pipeline, head, caps):
	head = pipeparts.mkresample(pipeline, head, quality = 9)
	head = pipeparts.mkcapsfilter(pipeline, head, caps)
	head = pipeparts.mkaudiorate(pipeline, head, skip_to_first = True, silent = False)
	return head

def mkmultiplier(pipeline, srcs, caps, sync = True):
	elem = pipeparts.mkgeneric(pipeline, None, "lal_multiplier", sync=sync)
	if srcs is not None:
		for src in srcs:
			pipeparts.mkcapsfilter(pipeline, src, caps).link(elem)
	elem = pipeparts.mkcapsfilter(pipeline, elem, caps)
	return elem

def mkinterleave(pipeline, srcs, caps):
	elem = pipeparts.mkgeneric(pipeline, None, "lal_interleave")
	if srcs is not None:
		for src in srcs:
			pipeparts.mkcapsfilter(pipeline, src, caps).link(elem)
	return elem

def mkadder(pipeline, srcs, caps, sync = True):
	elem = pipeparts.mkgeneric(pipeline, None, "lal_adder", sync=sync, caps=gst.Caps(caps))
	if srcs is not None:
		for src in srcs:
			pipeparts.mkcapsfilter(pipeline, src, caps).link(elem)
			#src.link(elem)
	elem = pipeparts.mkcapsfilter(pipeline, elem, caps)
	return elem

def list_srcs(pipeline, *args):
	out = []
	for src in args:
		out.append(pipeparts.mkqueue(pipeline, src, max_size_time = gst.SECOND * 100))
	return tuple(out)

def merge_into_complex(pipeline, real, imag, real_caps, complex_caps):
	head = mkinterleave(pipeline, list_srcs(pipeline, real, imag), real_caps)
	head = pipeparts.mkaudioconvert(pipeline, head)
	head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw-float")
	head = pipeparts.mkaudiorate(pipeline, head, skip_to_first = True, silent = False)
	head = pipeparts.mktogglecomplex(pipeline,head)
	head = pipeparts.mkcapsfilter(pipeline, head, complex_caps)
	return head

def split_into_real(pipeline, complex, real_caps, complex_caps):
	elem = pipeparts.mkcapsfilter(pipeline, complex, complex_caps)
	elem = pipeparts.mktogglecomplex(pipeline, elem)
	elem = pipeparts.mkcapsfilter(pipeline, elem, "audio/x-raw-float")
	elem = pipeparts.mkgeneric(pipeline, elem, "deinterleave")
	real = pipeparts.mkaudioconvert(pipeline, None)
	pipeparts.src_deferred_link(elem, "src0", real.get_pad("sink"))
	real = pipeparts.mkcapsfilter(pipeline, real, real_caps)
	real = pipeparts.mkaudiorate(pipeline, real, skip_to_first = True, silent = False)
	imag = pipeparts.mkaudioconvert(pipeline, None)
	pipeparts.src_deferred_link(elem, "src1", imag.get_pad("sink"))
	imag = pipeparts.mkcapsfilter(pipeline, imag, real_caps)
	imag = pipeparts.mkaudiorate(pipeline, imag, skip_to_first = True, silent = False)
	return real, imag

def demodulate(pipeline, head, sr, freq, orig_caps, new_caps, integration_samples):
	headtee = pipeparts.mktee(pipeline, head)
	deltat = 1.0/float(sr)
	cos = pipeparts.mkgeneric(pipeline, pipeparts.mkqueue(pipeline, headtee, max_size_time = gst.SECOND * 100), "lal_numpy_fx_transform", expression = "%f * cos(2.0 * 3.1415926535897931 * %f * t)" % (deltat, freq))
	sin = pipeparts.mkgeneric(pipeline, pipeparts.mkqueue(pipeline, headtee, max_size_time = gst.SECOND * 100), "lal_numpy_fx_transform", expression = "-1.0 * %f * sin(2.0 * 3.1415926535897931 * %f * t)" % (deltat, freq))

	headR = mkmultiplier(pipeline, (pipeparts.mkqueue(pipeline, headtee, max_size_time = gst.SECOND * 100), cos), orig_caps)
	headR = pipeparts.mkresample(pipeline, headR, quality=9)
	headR = pipeparts.mkcapsfilter(pipeline, headR, new_caps)
	headR = pipeparts.mkfirbank(pipeline, headR, fir_matrix=[numpy.hanning(integration_samples+1)], time_domain = True)
	headR = pipeparts.mkaudiorate(pipeline, headR, skip_to_first = True, silent = False)

	headI = mkmultiplier(pipeline, (pipeparts.mkqueue(pipeline, headtee, max_size_time = gst.SECOND * 100), sin), orig_caps)
	headI = pipeparts.mkresample(pipeline, headI, quality=9)
	headI = pipeparts.mkcapsfilter(pipeline, headI, new_caps)
	headI = pipeparts.mkfirbank(pipeline, headI, fir_matrix=[numpy.hanning(integration_samples+1)], time_domain = True)
	headI = pipeparts.mkaudiorate(pipeline, headI, skip_to_first = True, silent = False)

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

	outR = mkadder(pipeline, (pipeparts.mkaudioamplify(pipeline, pipeparts.mkqueue(pipeline, chanI, max_size_time = gst.SECOND * 100), -1.0 * WI), pipeparts.mkaudioamplify(pipeline, pipeparts.mkqueue(pipeline, chanR, max_size_time = gst.SECOND * 100), WR)), caps)
	outI = mkadder(pipeline, (pipeparts.mkaudioamplify(pipeline, pipeparts.mkqueue(pipeline, chanR, max_size_time = gst.SECOND * 100), WI), pipeparts.mkaudioamplify(pipeline, pipeparts.mkqueue(pipeline, chanI, max_size_time = gst.SECOND * 100), WR)), caps)
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

def compute_kappapu(pipeline, A0pufxinvR, A0pufxinvI, AfctrlR, AfctrlI, ktstR, ktstI, A0tstfxR, A0tstfxI, real_caps, complex_caps):
	ktst = merge_into_complex(pipeline, pipeparts.mkaudioamplify(pipeline, ktstR, -1.0), pipeparts.mkaudioamplify(pipeline, ktstI, -1.0), real_caps, complex_caps)
	A0tstfx = merge_into_complex(pipeline, A0tstfxR, A0tstfxI, real_caps, complex_caps)
	A0pufxinv = merge_into_complex(pipeline, A0pufxinvR, A0pufxinvI, real_caps, complex_caps)
	Afx = merge_into_complex(pipeline, AfctrlR, AfctrlI, real_caps, complex_caps)
	
	# \kappa_pu = (1/A0pufx) * (Afx - ktst * A0tstfx)
	kpu = mkmultiplier(pipeline, list_srcs(pipeline, A0pufxinv, mkadder(pipeline, list_srcs(pipeline, Afx, mkmultiplier(pipeline, list_srcs(pipeline, ktst, A0tstfx), complex_caps)), complex_caps)), complex_caps)
	kpuR, kpuI = split_into_real(pipeline, kpu, real_caps, complex_caps)

	return kpuR, kpuI

def compute_kappaa(pipeline, AfxR, AfxI, A0tstfxR, A0tstfxI, A0pufxR, A0pufxI,real_caps, complex_caps):
	Afx = merge_into_complex(pipeline, AfxR, AfxI, real_caps, complex_caps)
	A0tstfx = merge_into_complex(pipeline, A0tstfxR, A0tstfxI, real_caps, complex_caps)
	A0pufx = merge_into_complex(pipeline, pipeparts.mkaudioamplify(pipeline, A0pufxR, -1.0), pipeparts.mkaudioamplify(pipeline, A0pufxI, -1.0), real_caps, complex_caps)

	#\kappa_a = A0fx / (A0tstfx - A0pufx)

	A0tstfx_minus_A0pufx = mkadder(pipeline, list_srcs(pipeline, A0tstfx, A0pufx), complex_caps)
	A0tstfx_minus_A0pufxR, A0tstfx_minus_A0pufxI = split_into_real(pipeline, A0tstfx_minus_A0pufx, real_caps, complex_caps)
	A0tstfx_minus_A0pufxR = pipeparts.mktee(pipeline, A0tstfx_minus_A0pufxR)
	A0tstfx_minus_A0pufxI = pipeparts.mktee(pipeline, A0tstfx_minus_A0pufxI)
	den2 = mkadder(pipeline, list_srcs(pipeline, pipeparts.mkpow(pipeline, A0tstfx_minus_A0pufxR, exponent=2.0), pipeparts.mkpow(pipeline, A0tstfx_minus_A0pufxI, exponent=2.0)), real_caps)
	den2 = pipeparts.mktee(pipeline, pipeparts.mkpow(pipeline, den2, exponent = -1.0))
	denR = mkmultiplier(pipeline, list_srcs(pipeline, A0tstfx_minus_A0pufxR, den2), real_caps)
	denI = mkmultiplier(pipeline, list_srcs(pipeline, pipeparts.mkaudioamplify(pipeline, A0tstfx_minus_A0pufxI, -1.0), den2), real_caps)

	den = merge_into_complex(pipeline, denR, denI, real_caps, complex_caps)
	ka = mkmultiplier(pipeline, list_srcs(pipeline, Afx, den), complex_caps)
	kaR, kaI = split_into_real(pipeline, ka, real_caps, complex_caps)
	
	return kaR, kaI

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

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
	head = pipeparts.mkreblock(pipeline, None, block_duration = gst.SECOND)
	#head = pipeparts.mkqueue(pipeline, None, max_size_buffers = 0, max_size_time = gst.SECOND * 100)
	pipeparts.src_deferred_link(demux, "%s:%s" % (instrument, channel_name), head.get_pad("sink"))
	#head = pipeparts.mkreblock(pipeline, head, block_duration = gst.SECOND)
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
	#head = pipeparts.mkgeneric(pipeline, head, "lal_constant_upsample")
	head = pipeparts.mkresample(pipeline, head, quality=9)
	head = pipeparts.mkcapsfilter(pipeline, head, new_caps)
	return head

def resample(pipeline, head, caps):
	head = pipeparts.mkresample(pipeline, head, quality = 9)
	head = pipeparts.mkcapsfilter(pipeline, head, caps)
	#head = pipeparts.mkaudiorate(pipeline, head, skip_to_first = True, silent = False)
	return head

def mkmultiplier(pipeline, srcs, caps, sync = True, **properties):
	elem = pipeparts.mkgeneric(pipeline, None, "lal_multiplier", sync=sync, **properties)
	if srcs is not None:
		for src in srcs:
			src.link(elem)
	elem = pipeparts.mkcapsfilter(pipeline, elem, caps)
	return elem

def list_srcs(pipeline, *args):
	out = []
	for src in args:
		out.append(pipeparts.mkqueue(pipeline, src, max_size_time = gst.SECOND * 100))
	return tuple(out)

def demodulate(pipeline, head, sr, freq, orig_caps, new_caps, integration_samples):
	headtee = pipeparts.mktee(pipeline, head)
	deltat = 1.0/float(sr)
	cos = pipeparts.mkgeneric(pipeline, pipeparts.mkqueue(pipeline, headtee, max_size_time = gst.SECOND * 100), "lal_numpy_fx_transform", expression = "%f * cos(2.0 * 3.1415926535897931 * %f * t)" % (deltat, freq))
	cos = pipeparts.mkaudiorate(pipeline, cos, skip_to_first = True, silent = False)
	sin = pipeparts.mkgeneric(pipeline, pipeparts.mkqueue(pipeline, headtee, max_size_time = gst.SECOND * 100), "lal_numpy_fx_transform", expression = "-1.0 * %f * sin(2.0 * 3.1415926535897931 * %f * t)" % (deltat, freq))

	headR = mkmultiplier(pipeline, (pipeparts.mkqueue(pipeline, headtee, max_size_time = gst.SECOND * 100), cos), orig_caps)
	headR = pipeparts.mkresample(pipeline, headR, quality=9)
	headR = pipeparts.mkcapsfilter(pipeline, headR, new_caps)
	headR = pipeparts.mkfirbank(pipeline, headR, fir_matrix=[numpy.hanning(integration_samples+1)], time_domain = True)
	#headR = pipeparts.mkaudiorate(pipeline, headR, skip_to_first = True, silent = False)

	headI = mkmultiplier(pipeline, (pipeparts.mkqueue(pipeline, headtee, max_size_time = gst.SECOND * 100), sin), orig_caps)
	headI = pipeparts.mkresample(pipeline, headI, quality=9)
	headI = pipeparts.mkcapsfilter(pipeline, headI, new_caps)
	headI = pipeparts.mkfirbank(pipeline, headI, fir_matrix=[numpy.hanning(integration_samples+1)], time_domain = True)
	#headI = pipeparts.mkaudiorate(pipeline, headI, skip_to_first = True, silent = False)

	return headR, headI

def filter_at_line(pipeline, chanR, chanI, WR, WI):
	# Apply a filter to a demodulated channel at a specific frequency, where the filter at that frequency is Re[W] = WR and Im[W] = WI
	# Re[out] = -chanI*WI + chanR*WR
	# Im[out] = chanR*WI + chanI*WR

	chanR = pipeparts.mktee(pipeline, chanR)
	chanI = pipeparts.mktee(pipeline, chanI)

	chanI_WI = pipeparts.mkaudioamplify(pipeline, chanI, -1.0*WI)
	chanR_WR = pipeparts.mkaudioamplify(pipeline, chanR, WR)
	chanR_WI = pipeparts.mkaudioamplify(pipeline, chanR, WI)
	chanI_WR = pipeparts.mkaudioamplify(pipeline, chanI, WR)

	outR = pipeparts.mkadder(pipeline, (pipeparts.mkaudioamplify(pipeline, pipeparts.mkqueue(pipeline, chanI, max_size_time = gst.SECOND * 100), -1.0 * WI), pipeparts.mkaudioamplify(pipeline, pipeparts.mkqueue(pipeline, chanR, max_size_time = gst.SECOND * 100), WR)))
	outI = pipeparts.mkadder(pipeline, (pipeparts.mkaudioamplify(pipeline, pipeparts.mkqueue(pipeline, chanR, max_size_time = gst.SECOND * 100), WI), pipeparts.mkaudioamplify(pipeline, pipeparts.mkqueue(pipeline, chanI, max_size_time = gst.SECOND * 100), WR)))
	return outR, outI

def compute_pcalfp_over_derrfp(pipeline, derrfpR, derrfpI, pcalfpR, pcalfpI, caps):

	pcalfpRtee = pipeparts.mktee(pipeline, pcalfpR)
	pcalfpItee = pipeparts.mktee(pipeline, pcalfpI)
	derrfpRtee = pipeparts.mktee(pipeline, derrfpR)
	derrfpItee = pipeparts.mktee(pipeline, derrfpI)
	derrfp2 = pipeparts.mktee(pipeline, pipeparts.mkadder(pipeline, list_srcs(pipeline, pipeparts.mkpow(pipeline, derrfpRtee, exponent=2.0), pipeparts.mkpow(pipeline, derrfpItee, exponent=2.0))))
	cR1 = mkmultiplier(pipeline, list_srcs(pipeline, derrfpItee, pcalfpItee), caps)
	cR2 = mkmultiplier(pipeline, list_srcs(pipeline, derrfpRtee, pcalfpRtee), caps)
	cR = pipeparts.mktee(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, pipeparts.mkadder(pipeline, list_srcs(pipeline, cR1, cR2)), pipeparts.mkpow(pipeline, derrfp2, exponent=-1.0)), caps))
	cI1 = mkmultiplier(pipeline, list_srcs(pipeline, derrfpRtee, pcalfpItee), caps)
	cI2 = mkmultiplier(pipeline, list_srcs(pipeline, derrfpItee, pcalfpRtee), caps)
	cI = pipeparts.mktee(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, pipeparts.mkadder(pipeline, list_srcs(pipeline, cI1, pipeparts.mkaudioamplify(pipeline, cI2, -1.0))), pipeparts.mkpow(pipeline, derrfp2, exponent=-1.0)), caps))
	return cR, cI

def compute_kappatst(pipeline, derrfxR, derrfxI, excfxR, excfxI, pcalfp_derrfpR, pcalfp_derrfpI,  ktstfacR, ktstfacI, caps):
	derrfxRtee = pipeparts.mktee(pipeline, derrfxR)
	derrfxItee = pipeparts.mktee(pipeline, derrfxI)
	excfxRtee = pipeparts.mktee(pipeline, excfxR)
	excfxItee = pipeparts.mktee(pipeline, excfxI)

	# 	     
	# \kappa_TST = -ktstfac * (derrfx/excfx) * (pcalfp/derrfp)
	# ktstfac = (1/A0fx) * (C0fp/(1+G0fp)) * ((1+G0fx)/C0fx)
	#	   = a * b * c
	#
	# a = -ktstfac
	#	Re[a] = -ktstfacR
	#	Im[a] = -ktstfacI
	# b = derrfx / excfx
	#	|excfx|^2 = excfxR^2 + excfxI^2
	#	Re[b] = [(derrfxI * excfxI) + (derrfxR * excfxR)] / |excfx|^2
	#	Im[b] = [-(derrfxR * excfxI) + (derrfxI * excfxR)] / |excfx|^2
	# c = pcalfp / derrfp
	#	|derrfp|^2 = derrfpR^2 + derrfpI^2
	# 	Re[c] = [(derrfpI * pcalfpI) + (derrfpR * pcalfpR)] / |derrfp|^2
	#	Im[c] = [(derrfpR * pcalfpI) - (derrfpI * pcalfpR)] / |derrfp|^2

	# Compute real and imaginary parts of a
	aR = pipeparts.mktee(pipeline, pipeparts.mkaudioamplify(pipeline, ktstfacR, -1.0))
	aI = pipeparts.mktee(pipeline, pipeparts.mkaudioamplify(pipeline, ktstfacI, -1.0))

	# Compute the real and imaginary parts of b
	excfx2 = pipeparts.mktee(pipeline, pipeparts.mkadder(pipeline, (pipeparts.mkpow(pipeline, excfxRtee, exponent=2.0), pipeparts.mkpow(pipeline, excfxItee, exponent=2.0))))
	bR1 = mkmultiplier(pipeline, list_srcs(pipeline, derrfxItee, excfxItee), caps)
	bR2 = mkmultiplier(pipeline, list_srcs(pipeline, derrfxRtee, excfxRtee), caps)
	bR = pipeparts.mktee(pipeline, mkmultiplier(pipeline, (pipeparts.mkadder(pipeline, (bR1, bR2)), pipeparts.mkpow(pipeline, excfx2, exponent=-1.0)), caps))
	bI1 = mkmultiplier(pipeline, list_srcs(pipeline, derrfxRtee, excfxItee), caps)
	bI2 = mkmultiplier(pipeline, list_srcs(pipeline, derrfxItee, excfxRtee), caps)
	bI = pipeparts.mktee(pipeline, mkmultiplier(pipeline, (pipeparts.mkadder(pipeline, (pipeparts.mkaudioamplify(pipeline, bI1, -1.0), bI2)), pipeparts.mkpow(pipeline, excfx2, exponent=-1.0)), caps))

	# Compute the real and imaginary parts of c
	cR = pipeparts.mktee(pipeline, pcalfp_derrfpR)
	cI = pipeparts.mktee(pipeline, pcalfp_derrfpI)

	# Combine all the pieces to form Re[\kappa_A] and Im[\kappa_A]
	# Re[\kappa_tst] = -aI * (bR*cI + bI*cR) + aR * (-bI*cI + bR*cR)
	# Im[\kappa_tst] = -aI*bI*cI + aR*bR*cI + aR*bI*cR + aI*bR*cR

	bR_cI = pipeparts.mktee(pipeline, pipeparts.mkqueue(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, bR, cI), caps), max_size_time = gst.SECOND * 100))
	bI_cR = pipeparts.mktee(pipeline, pipeparts.mkqueue(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, bI, cR), caps), max_size_time = gst.SECOND * 100))
	bI_cI = pipeparts.mktee(pipeline, pipeparts.mkqueue(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, bI, cI), caps), max_size_time = gst.SECOND * 100))
	bR_cR = pipeparts.mktee(pipeline, pipeparts.mkqueue(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, bR, cR), caps), max_size_time = gst.SECOND * 100))

	ktstR1 = mkmultiplier(pipeline, (pipeparts.mkadder(pipeline, list_srcs(pipeline, bR_cI, bI_cR)), pipeparts.mkaudioamplify(pipeline, aI, -1.0)), caps)
	ktstR2 = mkmultiplier(pipeline, (pipeparts.mkadder(pipeline, (pipeparts.mkaudioamplify(pipeline, bI_cI, -1.0), pipeparts.mkqueue(pipeline, bR_cR))), pipeparts.mkqueue(pipeline, aR)), caps)
	ktstR = pipeparts.mkadder(pipeline, (ktstR1, ktstR2))

	ktstI1 = mkmultiplier(pipeline, (pipeparts.mkqueue(pipeline, bI_cI, max_size_time = gst.SECOND * 100), pipeparts.mkaudioamplify(pipeline, aI, -1.0)), caps)
	ktstI2 = mkmultiplier(pipeline, list_srcs(pipeline, bR_cI, aR), caps)
	ktstI3 = mkmultiplier(pipeline, list_srcs(pipeline, bI_cR, aR), caps)
	ktstI4 = mkmultiplier(pipeline, list_srcs(pipeline, bR_cR, aI), caps)
	ktstI = pipeparts.mkadder(pipeline, (ktstI1, ktstI2, ktstI3, ktstI4))

	return ktstR, ktstI

def compute_kappapu(pipeline, A0pufxR, A0pufxI, AfctrlR, AfctrlI, ktstR, ktstI, A0tstfxR, A0tstfxI, caps):
	A0pufxR = pipeparts.mktee(pipeline, A0pufxR)
	A0pufxI = pipeparts.mktee(pipeline, A0pufxI)
	ktstR = pipeparts.mktee(pipeline, ktstR)
	ktstI = pipeparts.mktee(pipeline, ktstI)
	A0tstfxR = pipeparts.mktee(pipeline, A0tstfxR)
	A0tstfxI = pipeparts.mktee(pipeline, A0tstfxI)
	
	# \kappa_pu = (1/A0pufx) * (Afx - ktst * A0tstfx)
	#      	    = a * (b - c)
	# a = 1/A0pufx
	#	|A0pufx|^2 = A0pufxR^2 + A0pufxI^2
	#	Re[a] = A0pufxR / |A0pufx|^2
	#	Im[a] = -A0pufxI / |A0pufx|^2
	# b = Afx
	#	Re[b] = AfxR
	#	Im[b] = AfxI
	# c = ktst * A0tstfx
	#	Re[c] = A0tstfxR*ktstR - A0tstfxI*ktstI
	#	Im[c] = A0tstfxR*ktstI + A0tstfI*ktstR

	A0pufx2 = pipeparts.mktee(pipeline, pipeparts.mkadder(pipeline, (pipeparts.mkpow(pipeline, A0pufxR, exponent=2.0), pipeparts.mkpow(pipeline, A0pufxI, exponent = 2.0))))
	aR = pipeparts.mktee(pipeline, mkmultiplier(pipeline, (pipeparts.mkqueue(pipeline, A0pufxR), pipeparts.mkpow(pipeline, A0pufx2, exponent=-1.0)), caps))
	aI = pipeparts.mktee(pipeline, mkmultiplier(pipeline, (pipeparts.mkaudioamplify(pipeline, A0pufxI, -1.0), pipeparts.mkpow(pipeline, A0pufx2, exponent=-1.0)), caps))
	
	bR = pipeparts.mktee(pipeline, AfctrlR)
	bI = pipeparts.mktee(pipeline, AfctrlI)

	A0tstfxR_ktstR = mkmultiplier(pipeline, list_srcs(pipeline, A0tstfxR, ktstR), caps)
	A0tstfxI_ktstI = mkmultiplier(pipeline, list_srcs(pipeline, A0tstfxI, ktstI), caps)
	A0tstfxR_ktstI = mkmultiplier(pipeline, list_srcs(pipeline, A0tstfxR, ktstI), caps)
	A0tstfxI_ktstR = mkmultiplier(pipeline, list_srcs(pipeline, A0tstfxI, ktstI), caps)
	cR = pipeparts.mktee(pipeline, pipeparts.mkadder(pipeline, (pipeparts.mkqueue(pipeline, A0tstfxR_ktstR), pipeparts.mkaudioamplify(pipeline, A0tstfxI_ktstI, -1.0))))
	cI = pipeparts.mktee(pipeline, pipeparts.mkadder(pipeline, (A0tstfxR_ktstI, A0tstfxI_ktstR)))

	# combine parts to form kpuR and kpuI
	# Re[kpu] = aI * (cI - bI) + aR * (bR - cR)	
	# Im[kpu] = aI * (bI - cI) + aI * (bR - cR)
	
	kpuR = pipeparts.mkadder(pipeline, (mkmultiplier(pipeline, (pipeparts.mkqueue(pipeline, aI), pipeparts.mkadder(pipeline, (pipeparts.mkqueue(pipeline, cI), pipeparts.mkaudioamplify(pipeline, bI, -1.0)))), caps), mkmultiplier(pipeline, (pipeparts.mkqueue(pipeline, aR), pipeparts.mkadder(pipeline, (pipeparts.mkqueue(pipeline, bR), pipeparts.mkaudioamplify(pipeline, cR, -1.0)))), caps)))
	kpuI = pipeparts.mkadder(pipeline, (mkmultiplier(pipeline, (pipeparts.mkqueue(pipeline, aI), pipeparts.mkadder(pipeline, (pipeparts.mkqueue(pipeline, bI), pipeparts.mkaudioamplify(pipeline, cI, -1.0)))), caps), mkmultiplier(pipeline, (pipeparts.mkqueue(pipeline, aI), pipeparts.mkadder(pipeline, (pipeparts.mkqueue(pipeline, bR), pipeparts.mkaudioamplify(pipeline, cR, -1.0)))), caps)))
	return kpuR, kpuI

def compute_kappaa(pipeline, AfxR, AfxI, A0tstfxR, A0tstfxI, A0pufxR, A0pufxI, caps):
	AfxR = pipeparts.mktee(pipeline, AfxR)
	AfxI = pipeparts.mktee(pipeline, AfxI)
	A0tstfxR = pipeparts.mktee(pipeline, A0tstfxR)
	A0tstfxI = pipeparts.mktee(pipeline, A0tstfxI)
	A0pufxR = pipeparts.mktee(pipeline, A0pufxR)
	A0pufxI = pipeparts.mktee(pipeline, A0pufxI)

	#\kappa_a = A0fx / (A0tstfx - A0pufx)
	# Re[ka] = [(-A0pufxI + A0tstfxI) AfxI + (-A0pufxR + A0tstfxR)*AfxR] / [(A0pufxI - A0tstfxI)^2 + (A0pufxR - A0tstfxR)^2]
	# Im[ka] = [(-A0pufxR + A0tstfxR) AfxI + (A0pufxI - A0tstfxI) AfxR] / [(A0pufxI - A0tstfxI)^2 + (A0pufxR - A0tstfxR)^2]
	
	den1 = pipeparts.mkpow(pipeline, pipeparts.mkadder(pipeline, (pipeparts.mkqueue(pipeline, A0pufxI), pipeparts.mkaudioamplify(pipeline, A0tstfxI, -1.0))), exponent = 2.0)
	den2 = pipeparts.mkpow(pipeline, pipeparts.mkadder(pipeline, (pipeparts.mkqueue(pipeline, A0pufxR), pipeparts.mkaudioamplify(pipeline, A0tstfxR, -1.0))), exponent = 2.0)
	den =  pipeparts.mktee(pipeline, pipeparts.mkadder(pipeline, (den1, den2)))

	kaR1 = mkmultiplier(pipeline, (pipeparts.mkqueue(pipeline, AfxI), pipeparts.mkadder(pipeline, (pipeparts.mkaudioamplify(pipeline, A0pufxI, -1.0), pipeparts.mkqueue(pipeline, A0tstfxI)))), caps)
	kaR2 = mkmultiplier(pipeline, (pipeparts.mkqueue(pipeline, AfxR), pipeparts.mkadder(pipeline, (pipeparts.mkaudioamplify(pipeline, A0pufxR, -1.0), pipeparts.mkqueue(pipeline, A0tstfxR)))), caps)
	kaR = mkmultiplier(pipeline, (pipeparts.mkadder(pipeline, (kaR1, kaR2)), pipeparts.mkpow(pipeline, den, exponent = -1.0)), caps)

	kaI1 = mkmultiplier(pipeline, (pipeparts.mkqueue(pipeline, AfxI), pipeparts.mkadder(pipeline, (pipeparts.mkaudioamplify(pipeline, A0pufxR, -1.0), pipeparts.mkqueue(pipeline, A0tstfxR)))), caps)
	kaI2 = mkmultiplier(pipeline, (pipeparts.mkqueue(pipeline, AfxR), pipeparts.mkadder(pipeline, (pipeparts.mkaudioamplify(pipeline, A0tstfxI, -1.0), pipeparts.mkqueue(pipeline, A0pufxI)))), caps)
	kaI = mkmultiplier(pipeline, (pipeparts.mkadder(pipeline, (kaI1, kaI2)), pipeparts.mkpow(pipeline, den, exponent = -1.0)), caps)
	return kaR, kaI

def compute_S(pipeline, CresR, CresI, pcal_derrR, pcal_derrI, ktstR, ktstI, kpuR, kpuI, D0R, D0I, A0tstR, A0tstI, A0puR, A0puI, caps):
	CresR = pipeparts.mktee(pipeline, CresR)
	CresI = pipeparts.mktee(pipeline, CresI)
	ktstR = pipeparts.mktee(pipeline, ktstR)
	kpuR = pipeparts.mktee(pipeline, kpuR)
	ktstI = pipeparts.mktee(pipeline, ktstI)
	kpuI = pipeparts.mktee(pipeline, kpuI)
	A0tstR = pipeparts.mktee(pipeline, A0tstR)
	A0puR = pipeparts.mktee(pipeline, A0puR)
	A0tstI = pipeparts.mktee(pipeline, A0tstI)
	A0puI = pipeparts.mktee(pipeline, A0puI)

	#	
	# S = 1/Cres * ( pcal/derr - D0*(ktst*A0tst + kpu*A0pu) ) ^ (-1)
	#   =    a   / (b - c)
	# 
	# a = 1/Cres
	#	|Cres|^2 = CresI^2 + CresR^2
	#	Re[a] = CresR / |Cres|^2
	#	Im[a] = -CresI / |Cres|^2
	# b = pcal/derr 
	#	Re[b] = pcal_derrR
	#	Im[b] = pcal_derrI
	# c = D0*(ktst*A0tst + kpu*A0pu)
	#   = d * e
	# 	d = D0
	#		Re[d] = D0R
	#		Im[d] = D0I
	#	e = ktst*A0tst + kpu*A0pu
	#		Re[e] = -A0puI kpuI + A0puR kpuR - A0tstI ktstI + A0tstR ktstR
	#		Im[e] = A0puR kpuI + A0puI kpuR + A0tstR ktstI + A0tstI ktstR
	#	Re[c] = -dI eI + dR eR
	#	Im[c] = dR eI + dI eR
	
	# Compute the real and imaginary parts of a
	Cres2 = pipeparts.mktee(pipeline, pipeparts.mkadder(pipeline, (pipeparts.mkpow(pipeline, CresR, exponent=2.0), pipeparts.mkpow(pipeline, CresI, exponent=2.0))))
	aR = pipeparts.mktee(pipeline, mkmultiplier(pipeline, (pipeparts.mkqueue(pipeline, CresR), pipeparts.mkpow(pipeline, Cres2, exponent=-1.0)), caps))
	aI = pipeparts.mktee(pipeline, mkmultiplier(pipeline, (pipeparts.mkaudioamplify(pipeline, CresI, -1.0), pipeparts.mkpow(pipeline, Cres2, exponent=-1.0)), caps))

	# Compute the real and imaginary parts of b
	bR = pipeparts.mktee(pipeline, pcal_derrR)
	bI = pipeparts.mktee(pipeline, pcal_derrI)

	# Compute the real and imaginary parts of c
	dR = pipeparts.mktee(pipeline, D0R)
	dI = pipeparts.mktee(pipeline, D0I)
	
	A0puI_kpuI = mkmultiplier(pipeline, list_srcs(pipeline, A0puI, kpuI), caps)
	A0puR_kpuR = mkmultiplier(pipeline, list_srcs(pipeline, A0puR, kpuR), caps)
	A0tstI_ktstI = mkmultiplier(pipeline, list_srcs(pipeline, A0tstI, ktstI), caps)
	A0tstR_ktstR = mkmultiplier(pipeline, list_srcs(pipeline, A0tstR, ktstR), caps)
	A0puR_kpuI = mkmultiplier(pipeline, list_srcs(pipeline, A0puR, kpuI), caps)
	A0puI_kpuR = mkmultiplier(pipeline, list_srcs(pipeline, A0puI, kpuR), caps)
	A0tstR_ktstI = mkmultiplier(pipeline, list_srcs(pipeline, A0tstR, ktstI), caps)
	A0tstI_ktstR = mkmultiplier(pipeline, list_srcs(pipeline, A0tstI, ktstR), caps)
	eR = pipeparts.mktee(pipeline, pipeparts.mkadder(pipeline, (pipeparts.mkaudioamplify(pipeline, A0puI_kpuI, -1.0), A0puR_kpuR, pipeparts.mkaudioamplify(pipeline, A0tstI_ktstI, -1.0), A0tstR_ktstR)))
	eI = pipeparts.mktee(pipeline, pipeparts.mkadder(pipeline, (A0puR_kpuI, A0puI_kpuR, A0tstR_ktstI, A0tstI_ktstR)))
	
	cR = pipeparts.mktee(pipeline, pipeparts.mkadder(pipeline, (pipeparts.mkaudioamplify(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, dI, eI), caps), -1.0), mkmultiplier(pipeline, list_srcs(pipeline, dR, eR), caps))))
	cI = pipeparts.mktee(pipeline, pipeparts.mkadder(pipeline, (mkmultiplier(pipeline, list_srcs(pipeline, dR, eI), caps), mkmultiplier(pipeline, list_srcs(pipeline, dI, eR), caps))))

	# Combine parts to make the real and imaginary parts of S
	# Re[S] = [aI * (bI-cI) + aR * (bR-cR)] / [ (bI-cI)^2 + (bR-cR)^2]
	# Im[S] = [aR * (cI-bI) + aI * (bR-cR)] / [ (bI-cI)^2 + (bR-cR)^2]

	den = pipeparts.mktee(pipeline, pipeparts.mkadder(pipeline, (pipeparts.mkpow(pipeline, pipeparts.mkadder(pipeline, (pipeparts.mkqueue(pipeline, bI), pipeparts.mkaudioamplify(pipeline, cI, -1.0))), exponent=2.0), pipeparts.mkpow(pipeline, pipeparts.mkadder(pipeline, (pipeparts.mkqueue(pipeline, bR), pipeparts.mkaudioamplify(pipeline, cR, -1.0))), exponent=2.0))))
	SR = mkmultiplier(pipeline, (pipeparts.mkadder(pipeline, (mkmultiplier(pipeline, (pipeparts.mkqueue(pipeline, aI), pipeparts.mkadder(pipeline, (pipeparts.mkqueue(pipeline, bI), pipeparts.mkaudioamplify(pipeline, cI, -1.0)))), caps), mkmultiplier(pipeline, (pipeparts.mkqueue(pipeline, aR), pipeparts.mkadder(pipeline, (pipeparts.mkqueue(pipeline, bR), pipeparts.mkaudioamplify(pipeline, cR, -1.0)))), caps))), pipeparts.mkpow(pipeline, den, exponent=-1.0)), caps)
	SI = mkmultiplier(pipeline, (pipeparts.mkadder(pipeline, (mkmultiplier(pipeline, (pipeparts.mkqueue(pipeline, aR), pipeparts.mkadder(pipeline, (pipeparts.mkqueue(pipeline, cI), pipeparts.mkaudioamplify(pipeline, bI, -1.0)))), caps), mkmultiplier(pipeline, (pipeparts.mkqueue(pipeline, aI), pipeparts.mkadder(pipeline, (pipeparts.mkqueue(pipeline, bR), pipeparts.mkaudioamplify(pipeline, cR, -1.0)))), caps))), pipeparts.mkpow(pipeline, den, exponent=-1.0)), caps)
	return SR, SI

def compute_kappac(pipeline, SR, SI, caps):
	#
	# \kappa_C = |S|^2 / Re[S]
	#
	SR = pipeparts.mktee(pipeline, SR)
	S2 = pipeparts.mkadder(pipeline, (pipeparts.mkpow(pipeline, SR, exponent=2.0), pipeparts.mkpow(pipeline, SI, exponent=2.0)))
	kc = mkmultiplier(pipeline, (S2, pipeparts.mkpow(pipeline, SR, exponent=-1.0)), caps)
	return kc

def compute_fcc(pipeline, SR, SI, fpcal, caps):
	#
	# f_cc = - (Re[S]/Im[S]) * fpcal
	#
	fcc = mkmultiplier(pipeline, (pipeparts.mkaudioamplify(pipeline, SR, -1.0), pipeparts.mkpow(pipeline, SI, exponent=-1.0)), caps)
	fcc = pipeparts.mkaudioamplify(pipeline, fcc, fpcal)
	return fcc

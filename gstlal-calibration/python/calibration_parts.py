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

def write_graph(demux, name):
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
	head = pipeparts.mkaudiorate(pipeline, head, skip_to_first = True, silent = False)
	return head

def resample(pipeline, head, caps):
	head = pipeparts.mkresample(pipeline, head, quality = 9)
	head = pipeparts.mkcapsfilter(pipeline, head, caps)
	head = pipeparts.mkaudiorate(pipeline, head, skip_to_first = True, silent = False)
	return head

def mkmultiplier(pipeline, srcs, sync = True, **properties):
	elem = pipeparts.mkgeneric(pipeline, None, "lal_multiplier", sync=sync, **properties)
	if srcs is not None:
		for src in srcs:
			src.link(elem)
	return elem

def list_srcs(pipeline, *args):
	out = []
	for src in args:
		out.append(pipeparts.mkqueue(pipeline, src, max_size_time = gst.SECOND * 100))
	return tuple(out)

def demodulate(pipeline, head, sr, freq, caps, integration_samples):
	headtee = pipeparts.mktee(pipeline, head)
	deltat = 1.0/float(sr)
	cos = pipeparts.mkgeneric(pipeline, pipeparts.mkqueue(pipeline, headtee, max_size_time = gst.SECOND * 100), "lal_numpy_fx_transform", expression = "%f * cos(2.0 * 3.1415926535897931 * %f * t)" % (deltat, freq))
	cos = pipeparts.mkaudiorate(pipeline, cos, skip_to_first = True, silent = False)
	sin = pipeparts.mkgeneric(pipeline, pipeparts.mkqueue(pipeline, headtee, max_size_time = gst.SECOND * 100), "lal_numpy_fx_transform", expression = "-1.0 * %f * sin(2.0 * 3.1415926535897931 * %f * t)" % (deltat, freq))

	headR = mkmultiplier(pipeline, (pipeparts.mkqueue(pipeline, headtee), pipeparts.mkqueue(pipeline, cos)))
	headR = pipeparts.mkresample(pipeline, headR, quality=9)
	headR = pipeparts.mkcapsfilter(pipeline, headR, caps)
	headR = pipeparts.mkfirbank(pipeline, headR, fir_matrix=[numpy.hanning(integration_samples+1)], time_domain = True)

	headI = mkmultiplier(pipeline, (pipeparts.mkqueue(pipeline, headtee), pipeparts.mkqueue(pipeline, sin)))
	headI = pipeparts.mkresample(pipeline, headI, quality=9)
	headI = pipeparts.mkcapsfilter(pipeline, headI, caps)
	headI = pipeparts.mkfirbank(pipeline, headI, fir_matrix=[numpy.hanning(integration_samples+1)], time_domain = True)

	return headR, headI

def compute_olg_from_ACD(pipeline, actR, sensR, darmR, actI, sensI, darmI):
	actRtee = pipeparts.mktee(pipeline, actR)
	actItee = pipeparts.mktee(pipeline, actI)
	sensRtee = pipeparts.mktee(pipeline, sensR)
	sensItee = pipeparts.mktee(pipeline, sensI)
	darmRtee = pipeparts.mktee(pipeline, darmR)
	darmItee = pipeparts.mktee(pipeline, darmI)
	
	# Compute the real part of the open loop gain
	# olgR = -actR * sensI * darmI - actI * sensR * darmI - actI * sensI * darmR + actR * sensR * darmR
	actR_sensI_darmI = mkmultiplier(pipeline, (pipeparts.mkqueue(pipeline, actRtee, max_size_time = gst.SECOND * 100), pipeparts.mkqueue(pipeline, sensItee, max_size_time = gst.SECOND * 100), pipeparts.mkqueue(pipeline, darmItee, max_size_time = gst.SECOND * 100)))

	actI_sensR_darmI = mkmultiplier(pipeline, (pipeparts.mkqueue(pipeline, actItee, max_size_time = gst.SECOND * 100), pipeparts.mkqueue(pipeline, sensRtee, max_size_time = gst.SECOND * 100), pipeparts.mkqueue(pipeline, darmItee, max_size_time = gst.SECOND * 100)))

	actI_sensI_darmR = mkmultiplier(pipeline, (pipeparts.mkqueue(pipeline, actItee, max_size_time = gst.SECOND * 100), pipeparts.mkqueue(pipeline, sensItee, max_size_time = gst.SECOND * 100), pipeparts.mkqueue(pipeline, darmRtee, max_size_time = gst.SECOND * 100)))

	actR_sensR_darmR = mkmultiplier(pipeline, (pipeparts.mkqueue(pipeline, actRtee, max_size_time = gst.SECOND * 100), pipeparts.mkqueue(pipeline, sensRtee, max_size_time = gst.SECOND * 100), pipeparts.mkqueue(pipeline, darmRtee, max_size_time = gst.SECOND * 100)))

	olgR = pipeparts.mkadder(pipeline, (pipeparts.mkqueue(pipeline, pipeparts.mkaudioamplify(pipeline, actR_sensI_darmI, -1.0), max_size_time = gst.SECOND * 100), pipeparts.mkqueue(pipeline, pipeparts.mkaudioamplify(pipeline, actI_sensR_darmI, -1.0), max_size_time = gst.SECOND * 100), pipeparts.mkqueue(pipeline, pipeparts.mkaudioamplify(pipeline, actI_sensI_darmR, -1.0), max_size_time = gst.SECOND * 100), pipeparts.mkqueue(pipeline, actR_sensR_darmR, max_size_time = gst.SECOND * 100)))

	# Compute the imaginary part of the open loop gain
	# olgI = -actI * sensI * darmI + actR * sensR * darmI + actR * sensI * darmR + actI * sensR * darmR
	actI_sensI_darmI = mkmultiplier(pipeline, (pipeparts.mkqueue(pipeline, actItee, max_size_time = gst.SECOND * 100), pipeparts.mkqueue(pipeline, sensItee, max_size_time = gst.SECOND * 100), pipeparts.mkqueue(pipeline, darmItee, max_size_time = gst.SECOND * 100)))

	actR_sensR_darmI = mkmultiplier(pipeline, (pipeparts.mkqueue(pipeline, actRtee, max_size_time = gst.SECOND * 100), pipeparts.mkqueue(pipeline, sensRtee, max_size_time = gst.SECOND * 100), pipeparts.mkqueue(pipeline, darmItee, max_size_time = gst.SECOND * 100)))

	actR_sensI_darmR = mkmultiplier(pipeline, (pipeparts.mkqueue(pipeline, actRtee, max_size_time = gst.SECOND * 100), pipeparts.mkqueue(pipeline, sensItee, max_size_time = gst.SECOND * 100), pipeparts.mkqueue(pipeline, darmRtee, max_size_time = gst.SECOND * 100)))

	actI_sensR_darmR = mkmultiplier(pipeline, (pipeparts.mkqueue(pipeline, actItee, max_size_time = gst.SECOND * 100), pipeparts.mkqueue(pipeline, sensRtee, max_size_time = gst.SECOND * 100), pipeparts.mkqueue(pipeline, darmRtee, max_size_time = gst.SECOND * 100)))

	olgI = pipeparts.mkadder(pipeline, (pipeparts.mkqueue(pipeline, pipeparts.mkaudioamplify(pipeline, actI_sensI_darmI, -1.0), max_size_time = gst.SECOND * 100), pipeparts.mkqueue(pipeline, actR_sensR_darmI, max_size_time = gst.SECOND * 100), pipeparts.mkqueue(pipeline, actR_sensI_darmR, max_size_time = gst.SECOND * 100), pipeparts.mkqueue(pipeline, actI_sensR_darmR, max_size_time = gst.SECOND * 100)))
	
	return olgR, olgI

def compute_kappaa(pipeline, A0fxR, A0fxI, derrfxR, derrfxI, derrfpR, derrfpI, excfxR, excfxI, pcalfpR, pcalfpI, C0fxR, C0fxI, G0fxR, G0fxI, G0fpR, G0fpI, C0fpR, C0fpI, WfxR, WfxI, WfpR, WfpI):
	A0fxRtee = pipeparts.mktee(pipeline, A0fxR)
	A0fxItee = pipeparts.mktee(pipeline, A0fxI)
	derrfxRtee = pipeparts.mktee(pipeline, derrfxR)
	derrfxItee = pipeparts.mktee(pipeline, derrfxI)
	excfxRtee = pipeparts.mktee(pipeline, excfxR)
	excfxItee = pipeparts.mktee(pipeline, excfxI)
	pcalfpRtee = pipeparts.mktee(pipeline, pcalfpR)
	pcalfpItee = pipeparts.mktee(pipeline, pcalfpI)
	C0fxRtee = pipeparts.mktee(pipeline, C0fxR)
	C0fxItee = pipeparts.mktee(pipeline, C0fxI)
	G0fxRtee = pipeparts.mktee(pipeline, G0fxR)
	G0fxItee = pipeparts.mktee(pipeline, G0fxI)
	G0fpRtee = pipeparts.mktee(pipeline, G0fpR)
	G0fpItee = pipeparts.mktee(pipeline, G0fpI)
	C0fpRtee = pipeparts.mktee(pipeline, C0fpR)
	C0fpItee = pipeparts.mktee(pipeline, C0fpI)
	derrfpRtee = pipeparts.mktee(pipeline, derrfpR)
	derrfpItee = pipeparts.mktee(pipeline, derrfpI)

	# 	     
	# \kappa_A = -(1/A0fx) * ((Wfx * derrfx)/excfx) * (pcalfp/(Wfp * derrfp)) * (C0fp/(1+G0fp)) * ((1+G0fx)/C0fx)
	#	   = a * b * c * d * e
	#
	# a = -1/A0fx
	#	|A0fx|^2 = A0fxR^2 + A0fxI^2
	#	Re[a] = -A0fxR / |A0fx|^2
	#	Im[a] = A0fxI / |A0fx|^2
	# b = (Wfx * derrfx) / excfx
	#	|excfx|^2 = excfxR^2 + excfxI^2
	#	Re[b] = [(derrfxR * excfxI * WfxI) - (derrfxI * excfxR * WfxI) + (derrfxI * excfxI * WfxR) + (derrfxR * excfxR * WfxR)] / |excfx|^2
	#	Im[b] = [(derrfxI * excfxI * WfxI) + (derrfxR * excfxR * wfxI) - (derrfxR * excfxI * WfxR) + (derrfxI * excfxR * WfxR)] / |excfx|^2
	# c = pcalfp / (Wfp * derrfp)
	#	|derrfp|^2 = derrfpR^2 + derrfpI^2
	#	|Wfp|^2 = WfpR^2 + WfpI^2
	# 	Re[c] = [(derrfpR * pcalfpI * WfpI) - (derrfpI * pcalfpR * WfpI) + (derrfpI * pcalfpI * WfpR) + (derrfpR * pcalfpR * WfpR)] / (|derrfp|^2 |Wfp|^2)
	#	Im[c] = [(-derrfpI * pcalfpI * WfpI) - (derrfpR * pcalfpR * WfpI) + (derrfpR * pcalfpI * WfpR) - (derrfpI * pcalfpR * WfpR)] / (|derrfp|^2 |Wfp|^2)
	# d = C0fp / (1 + G0fp)
	#	|1+G0fp|^2 = (G0fpI^2 + (1 + G0fpR)^2)
	#	Re[d] = [C0fpR + (C0fpI * G0fpI) + (C0fpR * G0fpR)] / |1+G0fp|^2
	#	Im[d] = [C0fpI - (C0fpR * G0fpI) + (C0fpI * G0fpR)] / |1+G0fp|^2
	# e = (1 + G0fx) / C0fx
	#	|C0fx|^2 = C0fxI^2 + C0fxR^2
	#  	Re[e] = [C0fxR + (C0fxI * G0fxI) + (C0fxR * G0fxR)] / |C0fx|^2
	#	Im[e] = [-C0fxI + (C0fxR * G0fxI) - (C0fxI * G0fxR)] / |C0fx|^2

	# Compute real and imaginary parts of a
	A0fx2 = pipeparts.mktee(pipeline, pipeparts.mkadder(pipeline, list_srcs(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, A0fxRtee, A0fxRtee)), mkmultiplier(pipeline, list_srcs(pipeline, A0fxItee, A0fxItee)))))
	aR = pipeparts.mktee(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, pipeparts.mkaudioamplify(pipeline, A0fxRtee, amplification=-1.0), pipeparts.mkpow(pipeline, A0fx2, exponent=-1.0))))
	aI = pipeparts.mktee(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, A0fxItee, pipeparts.mkpow(pipeline, A0fx2, exponent=-1.0))))

	# Compute the real and imaginary parts of b
	excfx2 = pipeparts.mktee(pipeline, pipeparts.mkadder(pipeline, list_srcs(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, excfxRtee, excfxRtee)), mkmultiplier(pipeline, list_srcs(pipeline, excfxItee, excfxItee)))))
	bR1 = pipeparts.mkaudioamplify(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, derrfxRtee, excfxItee)), amplification=WfxI)
	bR2 = pipeparts.mkaudioamplify(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, derrfxItee, excfxRtee)), amplification=WfxI)
	bR3 = pipeparts.mkaudioamplify(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, derrfxItee, excfxItee)), amplification=WfxR)
	bR4 = pipeparts.mkaudioamplify(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, derrfxRtee, excfxRtee)), amplification=WfxR)
	bR = pipeparts.mktee(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, pipeparts.mkadder(pipeline, list_srcs(pipeline, bR1, pipeparts.mkaudioamplify(pipeline, bR2, amplification=-1.0), bR3, bR4)), pipeparts.mkpow(pipeline, excfx2, exponent=-1.0))))
	bI1 = pipeparts.mkaudioamplify(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, derrfxItee, excfxItee)), amplification=WfxI)
	bI2 = pipeparts.mkaudioamplify(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, derrfxRtee, excfxRtee)), WfxI)
	bI3 = pipeparts.mkaudioamplify(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, derrfxRtee, excfxItee)), WfxR)
	bI4 = pipeparts.mkaudioamplify(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, derrfxItee, excfxRtee)), WfxR)
	bI = pipeparts.mktee(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, pipeparts.mkadder(pipeline, list_srcs(pipeline, bI1, bI2, pipeparts.mkaudioamplify(pipeline, bI3, -1.0), bI4)), pipeparts.mkpow(pipeline, excfx2, exponent=-1.0))))

	# Compute the real and imaginary parts of c
	derrfp2 = pipeparts.mkadder(pipeline, list_srcs(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, derrfpRtee, derrfpRtee)), mkmultiplier(pipeline, list_srcs(pipeline, derrfpItee, derrfpItee))))
	Wfp2 = WfpR * WfpR + WfpI * WfpI
	derrfp2Wfp2 = pipeparts.mktee(pipeline, pipeparts.mkaudioamplify(pipeline, derrfp2, Wfp2))
	cR1 = pipeparts.mkaudioamplify(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, derrfpRtee, pcalfpItee)), WfpI)
	cR2 = pipeparts.mkaudioamplify(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, derrfpItee, pcalfpRtee)), WfpI)
	cR3 = pipeparts.mkaudioamplify(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, derrfpItee, pcalfpItee)), WfpR)
	cR4 = pipeparts.mkaudioamplify(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, derrfpRtee, pcalfpRtee)), WfpR)
	cR = pipeparts.mktee(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, pipeparts.mkadder(pipeline, list_srcs(pipeline, cR1, pipeparts.mkaudioamplify(pipeline, cR2, -1.0), cR3, cR4)), pipeparts.mkpow(pipeline, derrfp2Wfp2, exponent=-1.0))))
	cI1 = pipeparts.mkaudioamplify(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, derrfpItee, pcalfpItee)), WfpI)
	cI2 = pipeparts.mkaudioamplify(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, derrfpRtee, pcalfpRtee)), WfpI)
	cI3 = pipeparts.mkaudioamplify(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, derrfpRtee, pcalfpItee)), WfpR)
	cI4 = pipeparts.mkaudioamplify(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, derrfpItee, pcalfpRtee)), WfpR)
	cI = pipeparts.mktee(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, pipeparts.mkadder(pipeline, list_srcs(pipeline, pipeparts.mkaudioamplify(pipeline, cI1, -1.0), pipeparts.mkaudioamplify(pipeline, cI2, -1.0), cI3, pipeparts.mkaudioamplify(pipeline, cI4, -1.0))), pipeparts.mkpow(pipeline, derrfp2Wfp2, exponent=-1.0))))

	# Compute the real and imaginary parts of d
	oneplusG0fpR = pipeparts.mktee(pipeline, pipeparts.mkgeneric(pipeline, G0fpRtee, "lal_add_constant", constant = 1.0))
	onepG0fp2 = pipeparts.mktee(pipeline, pipeparts.mkadder(pipeline, list_srcs(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, G0fpItee, G0fpItee)), mkmultiplier(pipeline, list_srcs(pipeline, oneplusG0fpR, oneplusG0fpR)))))
	dR = pipeparts.mktee(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, pipeparts.mkadder(pipeline, list_srcs(pipeline, C0fpRtee, mkmultiplier(pipeline, list_srcs(pipeline, C0fpItee, G0fpItee)), mkmultiplier(pipeline, list_srcs(pipeline, C0fpRtee, G0fpRtee)))), pipeparts.mkpow(pipeline, onepG0fp2, exponent=-1.0))))
	dI = pipeparts.mktee(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, pipeparts.mkadder(pipeline, list_srcs(pipeline, C0fpItee, pipeparts.mkaudioamplify(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, C0fpRtee, G0fpItee)), -1.0), mkmultiplier(pipeline, list_srcs(pipeline, C0fpItee, G0fpRtee)))), pipeparts.mkpow(pipeline, onepG0fp2, exponent=-1.0))))

	# Compute the real and imaginary parts of e
	C0fx2 = pipeparts.mktee(pipeline, pipeparts.mkadder(pipeline, list_srcs(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, C0fxItee, C0fxItee)), mkmultiplier(pipeline, list_srcs(pipeline, C0fxRtee, C0fxRtee))))) 
	eR = pipeparts.mktee(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, pipeparts.mkadder(pipeline, list_srcs(pipeline, C0fxRtee, mkmultiplier(pipeline, list_srcs(pipeline, C0fxItee, G0fxItee)), mkmultiplier(pipeline, list_srcs(pipeline, C0fxRtee, G0fxRtee)))), pipeparts.mkpow(pipeline, C0fx2, exponent=-1.0))))
	eI = pipeparts.mktee(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, pipeparts.mkadder(pipeline, list_srcs(pipeline, pipeparts.mkaudioamplify(pipeline, C0fxItee, -1.0), mkmultiplier(pipeline, list_srcs(pipeline, C0fxRtee, G0fxItee)), pipeparts.mkaudioamplify(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, C0fxItee, G0fxRtee)), -1.0))), pipeparts.mkpow(pipeline, C0fx2, exponent=-1.0))))

	# Combine all the pieces to form Re[\kappa_A] and Im[\kappa_A]
	# Re[\kappa_A] = -aR * eI * (-bI*cI*dI + bR*cR*dI + bR*cI*dR + bI*cR*dR) + 
	# 		  aI * eI * (bR*cI*dI + bI*cR*dI + bI*cI*dR - bR*cR*dR) -
	#		  aI * eR * (-bI*cI*dI + bR*cR*dI + bR*cI*dR + bI*cR*dR) -
	#		  aR * eR * (bR*cI*dI + bI*cR*dI + bI*cI*dR - bR*cR*dR)
	# Im[\kappa_A] = -aI * eI * (-bI*cI*dI + bR*cR*dI + bR*cI*dR + bI*cR*dR) - 
	#		  aI * eR * (bR*cI*dI + bI*cR*dI + bI*cI*dR - bR*cR*dR) -
	#		  aR * eI * (bR*cI*dI + bI*cR*dI + bI*cI*dR - bR*cR*dR) +
	#		  aR * eR * (-bI*cI*dI + bR*cR*dI + bR*cI*dR + bI*cR*dR)

	bI_cI_dI = mkmultiplier(pipeline, list_srcs(pipeline, bI, cI, dI))
	bR_cR_dI = mkmultiplier(pipeline, list_srcs(pipeline, bR, cR, dI))
	bR_cI_dR = mkmultiplier(pipeline, list_srcs(pipeline, bR, cI, dR))
	bI_cR_dR = mkmultiplier(pipeline, list_srcs(pipeline, bI, cR, dR))
	bR_cI_dI = mkmultiplier(pipeline, list_srcs(pipeline, bR, cI, dI))
	bI_cR_dI = mkmultiplier(pipeline, list_srcs(pipeline, bI, cR, dI))
	bI_cI_dR = mkmultiplier(pipeline, list_srcs(pipeline, bI, cI, dR))
	bR_cR_dR = mkmultiplier(pipeline, list_srcs(pipeline, bR, cR, dR))

	combo1 = pipeparts.mktee(pipeline, pipeparts.mkadder(pipeline, list_srcs(pipeline, pipeparts.mkaudioamplify(pipeline, bI_cI_dI, -1.0), bR_cR_dI, bR_cI_dR, bI_cR_dR)))
	combo2 = pipeparts.mktee(pipeline, pipeparts.mkadder(pipeline, list_srcs(pipeline, bR_cI_dI, bI_cR_dI, bI_cI_dR, pipeparts.mkaudioamplify(pipeline, bR_cR_dR, -1.0))))

	kaR = pipeparts.mkadder(pipeline, list_srcs(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, pipeparts.mkaudioamplify(pipeline, aR, -1.0), eI, combo1)), mkmultiplier(pipeline, list_srcs(pipeline, aI, eI, combo2)), mkmultiplier(pipeline, list_srcs(pipeline, pipeparts.mkaudioamplify(pipeline, aI, -1.0), eR, combo1)), mkmultiplier(pipeline, list_srcs(pipeline, pipeparts.mkaudioamplify(pipeline, aR, -1.0), eR, combo2))))
	kaI = pipeparts.mkadder(pipeline, list_srcs(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, pipeparts.mkaudioamplify(pipeline, aI, -1.0), eI, combo1)), mkmultiplier(pipeline, list_srcs(pipeline, pipeparts.mkaudioamplify(pipeline, aI, -1.0), eR, combo2)), mkmultiplier(pipeline, list_srcs(pipeline, pipeparts.mkaudioamplify(pipeline, aR, -1.0), eI, combo2)), mkmultiplier(pipeline, list_srcs(pipeline, aR, eR, combo1))))

	return kaR, kaI
	
def compute_S(pipeline, CresR, CresI, pcalR, pcalI, derrR, derrI, kaR, kaI, D0R, D0I, A0R, A0I, WR, WI):
	CresR = pipeparts.mktee(pipeline, CresR)
	CresI = pipeparts.mktee(pipeline, CresI)
	pcalR = pipeparts.mktee(pipeline, pcalR)
	pcalI = pipeparts.mktee(pipeline, pcalI)
	derrR = pipeparts.mktee(pipeline, derrR)
	derrI = pipeparts.mktee(pipeline, derrI)
	kaR = pipeparts.mktee(pipeline, kaR)
	kaI = pipeparts.mktee(pipeline, kaI)
	D0R = pipeparts.mktee(pipeline, D0R)
	D0I = pipeparts.mktee(pipeline, D0I)
	A0R = pipeparts.mktee(pipeline, A0R)
	A0I = pipeparts.mktee(pipeline, A0I)

	#	
	# S = 1/Cres * ( pcal/(W*derr) - ka*D0*A0 ) ^ (-1)
	#   =    a   /  b
	# 
	# a = 1/Cres
	#	|Cres|^2 = CresI^2 + CresR^2
	#	Re[a] = CresR / |Cres|^2
	#	Im[a] = -CresI / |Cres|^2
	# b = pcal/(W*derr) - ka*D0*A0
	#	|derr|^2 = derrR^2 + derrI^2
	#	|W|^2 = WR^2 + WI^2
	#	Re[b] = A0R*D0I*kaI + A0I*D0R*kaI + A0I*D0I*kaR - A0R*D0R*kaR + (derrR*pcalI*WI - derrI*pcalR*WI +derrI*pcalI*WR + derrR*pcalR*WR) / (|derr|^2 |W|^2)
	#	Im[b] = -A0R*D0R*kaI - A0R*D0I*kaR + A0I*D0I*kaI - A0I*D0R*kaR - (derrI*pcalI*WI + derrR*pcalR*WI - derrR*pcalI*WR + derrI*pcalR*WR) / (|derr|^2 |W|^2)
	
	# Compute the real and imaginary parts of a
	Cres2 = pipeparts.mktee(pipeline, pipeparts.mkadder(pipeline, list_srcs(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, CresR, CresR)), mkmultiplier(pipeline, list_srcs(pipeline, CresI, CresI)))))
	aR = pipeparts.mktee(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, CresR, pipeparts.mkpow(pipeline, Cres2, exponent=-1.0))))
	aI = pipeparts.mktee(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, pipeparts.mkaudioamplify(pipeline, CresI, -1.0), pipeparts.mkpow(pipeline, Cres2, exponent=-1.0))))

	# Compute the real and imaginary parts of b
	derr2 = pipeparts.mkadder(pipeline, list_srcs(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, derrR, derrR)), mkmultiplier(pipeline, list_srcs(pipeline, derrI, derrI))))
	W2 = WR * WR + WI * WI
	derr2W2 = pipeparts.mktee(pipeline, pipeparts.mkaudioamplify(pipeline, derr2, W2))
	A0R_D0I_kaI = mkmultiplier(pipeline, list_srcs(pipeline, A0R, D0I, kaI))
	A0I_D0R_kaI = mkmultiplier(pipeline, list_srcs(pipeline, A0I, D0R, kaI))
	A0I_D0I_kaR = mkmultplier(pipeline, list_srcs(pipeline, A0I, D0I, kaR))
	A0R_D0R_kaR = mkmultiplier(pipeline, list_srcs(pipeline, A0R, D0R, kaR))
	derrR_pcalI_WI = pipeparts.mkaudioamplify(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, derrR, pcalI)), WI)
	derrI_pcalR_WI = pipeparts.mkaudioamplify(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, derrI, pcalR)), WI)
	derrI_pcalI_WR = pipeparts.mkaudioamplify(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, derrI, pcalI)), WR)
	derrR_pcalR_WR = pipeparts.mkaudioamplify(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, derrR, pcalR)), WR)
	bR = pipeparts.mktee(pipeline, pipeparts.mkadder(pipeline, list_srcs(pipeline, A0R_D0I_kaI, A0I_D0R_kaI, A0I_D0I_kaR, pipeparts.mkaudioamplify(pipeline, A0R_D0R_kaR, -1.0), mkmultiplier(pipeline, list_srcs(pipeline, pipeparts.mkadder(pipeline, list_srcs(pipeline, derrR_pcalI_WI, pipeparts.mkaudioamplify(pipeline, derrI_pcalR_WI, -1.0), derrI_pcalI_WR, derrR_pcalR_WR)), pipeparts.mkpow(pipeline, derr2W2, exponent=-1.0))))))

	A0R_D0R_kaI = mkmultiplier(pipeline, list_srcs(pipeline, A0R, D0R, kaI))
	A0R_D0I_kaR = mkmultiplier(pipeline, list_srcs(pipeline, A0R, D0I, kaR))
	A0I_D0I_kaI = mkmultiplier(pipeline, list_srcs(pipeline, A0I, D0I, kaI))
	A0I_D0R_kaR = mkmultiplier(pipeline, list_srcs(pipeline, A0I, D0R, kaR))
	derrI_pcalI_WI = pipeparts.mkaudioamplify(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, derrI, pcalI)), WI)
	derrR_pcalR_WI = pipeparts.mkaudioamplify(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, derrR, pcalR)), WI)
	derrR_pcalI_WR = pipeparts.mkaudioamplify(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, derrR, pcalI)), WR)
	derrI_pcalR_WR = pipeparts.mkaudioamplify(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, derrI, pcalR)), WR)
	bI = pipeparts.mktee(pipeline, pipeparts.mkadder(pipeline, list_srcs(pipeline, pipeparts.mkaudioamplify(pipeline, A0R_D0R_kaI, -1.0), pipeparts.mkaudiamplify(pipeline, A0R_D0I_kaR, -1.0), A0I_D0I_kaI, pipeparts.mkaudioamplify(pipeline, A0I_D0R_kaR, -1.0), pipeparts.mkaudioamplify(pipeline, mkmultplier(pipeline, list_srcs(pipeline, pipeparts.mkadder(pipeline, list_srcs(pipeline, derrI_pcalI_WI, derrR_pcalR_WI, pipeparts.mkaudioamplify(pipeline, derrR_pcalI_WR, -1.0), derrI_pcalR_WR)), pipeparts.mkpow(pipeline, derr2W2, exponent=-1.0))), -1.0))))

	# Combine parts to make the real and imaginary parts of S
	# Re[S] = (aI*bI + aR*bR) / |b|^2
	# Im[S] = (aI*bR - aR*bI) / |b|^2

	b2 = pipeparts.mktee(pipeline, pipeparts.mkadder(pipeline, list_srcs(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, bR, bR)), mkmultiplier(pipeline, list_srcs(pipeline, bI, bI)))))
	aI_bI = mkmultiplier(pipeline, list_srcs(pipeline, aI, bI))
	aR_bR = mkmultiplier(pipeline, list_srcs(pipeline, aR, bR))
	aI_bR = mkmultiplier(pipeline, list_srcs(pipeline, aI, bR))
	aR_bI = mkmultiplier(pipeline, list_srcs(pipeline, aR, bI))
	SR = mkmultiplier(pipeline, list_srcs(pipeline, pipeparts.mkadder(pipeline, list_srcs(pipeline, aI_bI, aR_bR)), pipeparts.mkpow(pipeline, b2, exponent=-1.0)))
	SI = mkmultiplier(pipeline, list_srcs(pipeline, pipeparts.mkadder(pipeline, list_srcs(pipeline, aI_bR, pipeparts.mkaudioamplify(pipeline, aR_bI, -1.0))), pipeparts.mkpow(pipeline, b1, exponent=-1.0)))

	return SR, SI

def compute_kappac(pipeline, SR, SI):
	#
	# \kappa_C = |S|^2 / Re[S]
	#
	SR = pipeparts.mktee(pipeline, SR)
	SI = pipeparts.mktee(pipeline, SI)
	S2 = pipeparts.mkadder(pipeline, list_srcs(pipeline, mkmultiplier(pipeline, list_srcs(pipeline, SR, SR)), mkmultiplier(pipeline, list_srcs(pipeline, SI, SI))))
	kc = mkmultiplier(pipeline, list_srcs(pipeline, S2, pipeparts.mkpow(pipeline, SR, exponent=-1.0)))
	return kc

def compute_fcc(pipeline, SR, SI, fpcal):
	#
	# f_cc = - (Re[S]/Im[S]) * fpcal
	#
	fcc = mkmultiplier(pipeline, list_srcs(pipeline, pipeparts.mkaudioamplify(pipeline, SR, -1.0), pipeparts.mkpow(pipeline, SI, exponent=-1.0)))
	fcc = pipeparts.mkaudioamplify(pipeline, fcc, fpcal)
	return fcc

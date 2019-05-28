# Copyright (C) 2009--2013  Kipp Cannon, Chad Hanna, Drew Keppel
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

##
# @file
#
# A file that contains the lloidparts module code; Roughly speaking it
# implements the algorithm described in <a
# href=http://arxiv.org/abs/1107.2665>ApJ 748 136 (2012)</a>
#
#
# Review Status
#
# | Names                                 | Hash                                     | Date       | Diff to Head of Master      |
# | ------------------------------------- | ---------------------------------------- | ---------- | --------------------------- |
# | Sathya, Duncan Me, Jolien, Kipp, Chad | 2f5f73f15a1903dc7cc4383ef30a4187091797d1 | 2014-05-02 | <a href="@gstlal_inspiral_cgit_diff/python/lloidparts.py?id=HEAD&id2=2f5f73f15a1903dc7cc4383ef30a4187091797d1">lloidparts.py</a> |
#
# #### Actions
# - Feature request: do checkpointing when instruments are down
# - Feature request: need to hook up equivalent of "CAT 2" vetoes to online analysis when ready, and HW inj veto, etc.
# - Document the parameters in mkLLOIDmulti()
# - Check if bank ids are the same across different instruments
# - Feature request: Make time-frequency videos for interesting events
# - Inject signals of known, high SNR and measure loss
# - move the makesegmentsrcgate to before the matrix mixer, not in the current sum-of-squares control loop
# - Make conditional branches in the graph gray boxes
# - consider zero padding the beginning of jobs to get rid of mkdrop()
# - Decide how to properly normalize SNRs for incomplete filter segments (currently filters are not renormalized so the loss in SNR is bigger than might be expected)
# - Check and possibly fix the function that switches between time-domain and FFT convolution based on stride and number of samples
# - Consider if quality = 9 is right for downsampling (it should be plenty good, but maybe excessive?)
#
#
# #### Functions/classes not reviewed since they will be moved
# - DetectorData
# - mkSPIIRmulti
# - mkSPIIRhoftToSnrSlices
# - mkLLOIDSnrSlicesToTimeSliceChisq
# - mkLLOIDSnrChisqToTriggers

##
# @package lloidparts
#
# a module for building gstreamer graphs of the LLOID algorithm
#


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


import math
import numpy


import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
GObject.threads_init()
Gst.init(None)


from glue import iterutils
from gstlal import datasource
from gstlal import multirate_datasource
from gstlal import pipeparts
from gstlal import pipeio


#
# =============================================================================
#
#                              Pipeline Elements
#
# =============================================================================
#

##
# A "sum-of-squares" aggregator
#
# _Gstreamer graph describing this function:_
#
# @dot
# digraph G {
#	rankdir="LR";
#
#	// nodes
#	node [shape=box, style=rounded];
#
#	lal_adder
#	lal_peak [URL="\ref pipeparts.mkpeak()", style=filled, color=grey, label="lal_peak\niff control_peak_samples > 0"];
#	capsfilter [URL = "\ref pipeparts.mkcapsfilter()"];
#	"mksegmentsrcgate()" [URL="\ref datasource.mksegmentsrcgate()"];
#	tee [URL="\ref pipeparts.mktee()"];
#	lal_checktimestamps [URL="\ref pipeparts.mkchecktimestamps()"];
#
#	// connections
#	"? sink 1" -> lal_adder;
#	"? sink 2" -> lal_adder;
#	"? sink N" -> lal_adder;
#	lal_adder -> capsfilter;
#	capsfilter -> lal_peak;
#	lal_peak -> lal_checktimestamps;
#	lal_checktimestamps -> tee;
#	tee -> "? src";
# }
# @enddot
#
#
def mkcontrolsnksrc(pipeline, rate, verbose = False, suffix = None, control_peak_samples = None):
	"""!
	This function implements a portion of a gstreamer graph to provide a
	control signal for deciding when to reconstruct physical SNRS

	@param pipeline A reference to the gstreamer pipeline in which to add this graph
	@param rate An integer representing the target sample rate of the resulting src
	@param verbose Make verbose
	@param suffix Log name for verbosity
	@param control_peak_samples If nonzero, this would do peakfinding on the control signal with the window specified by this parameter.  The peak finding would give a single sample of "on" state at the peak.   This will cause far less CPU to be used if you only want to reconstruct SNR around the peak of the control signal.
	"""
	#
	# start with an adder and caps filter to select a sample rate
	#

	snk = pipeparts.mkadder(pipeline, None)
	src = pipeparts.mkcapsfilter(pipeline, snk, "audio/x-raw, rate=%d" % rate)

	#
	# Add a peak finder on the control signal sample number
	#

	if control_peak_samples > 0:
		src = pipeparts.mkpeak(pipeline, src, control_peak_samples)

	#
	# verbosity and a tee
	#

	logname = suffix and "_%s" % suffix or ""
	if verbose:
		src = pipeparts.mkprogressreport(pipeline, src, "progress_sumsquares%s" % logname)
	src = pipeparts.mkchecktimestamps(pipeline, src, "timestamps%s_sumsquares" % logname)
	src = pipeparts.mktee(pipeline, src)

	#
	# return the adder and tee
	#

	return snk, src


##
# _Gstreamer graph describing this function:_
#
# @dot
# digraph G {
#	rankdir="LR";
#
#	// nodes
#	node [shape=box, style=rounded];
#
# 	lal_firbank [URL="\ref pipeparts.mkfirbank()"];
#	lal_checktimestamps1 [URL="\ref pipeparts.mkchecktimestamps()"];
#	lal_checktimestamps2 [URL="\ref pipeparts.mkchecktimestamps()"];
#	queue [URL="\ref pipeparts.mkqueue()"];
#	matrixmixer [URL="\ref pipeparts.mkmatrixmixer()", label="lal_matrixmixer\niff bank.mix_matrix", style=filled, color=grey];
#
#	"? sink" -> lal_firbank;
#	lal_firbank -> lal_checktimestamps1;
#
#	// without control
#
#	lal_checktimestamps1 -> queue [label="iff control_snk, control_src are None"];
#	queue -> matrixmixer;
#	matrixmixer -> lal_checktimestamps2;
#	lal_checktimestamps2 -> "? src";
#
#	// with control
#
#	tee [URL="\ref pipeparts.mktee()"];
#	queue2 [URL="\ref pipeparts.mkqueue()"];
#	queue3 [URL="\ref pipeparts.mkqueue()"];
#	queue4 [URL="\ref pipeparts.mkqueue()"];
#	lal_checktimestamps3 [URL="\ref pipeparts.mkchecktimestamps()"];
#	lal_checktimestamps4 [URL="\ref pipeparts.mkchecktimestamps()"];
#	lal_checktimestamps5 [URL="\ref pipeparts.mkchecktimestamps()"];
#	capsfilter [URL="\ref pipeparts.mkcapsfilter()"];
#	gate [URL="\ref pipeparts.mkgate()"];
#	"mkcontrolsnksrc()" [URL="\ref mkcontrolsnksrc()"];
#	lal_sumsquares [URL="\ref pipeparts.mksumsquares()"];
#	audioresample [URL="\ref pipeparts.mkresample()"];
#
#
#	lal_checktimestamps1 -> tee [label="iff control_snk, control_src are not None"];
#	tee -> lal_sumsquares -> queue2;
#	queue2 -> lal_checktimestamps3;
#	lal_checktimestamps3 -> audioresample;
#	audioresample -> capsfilter;
#	capsfilter -> lal_checktimestamps4;
#	lal_checktimestamps4 -> "mkcontrolsnksrc()"
#	"mkcontrolsnksrc()" -> queue3;
#	queue3 -> gate;
#	tee -> queue4 -> gate;
#	gate -> lal_checktimestamps5;
#	lal_checktimestamps5 -> "mksegmentsrcgate()";
#	"mksegmentsrcgate()" -> matrixmixer;
#
# }
# @enddot
#
#
def mkLLOIDbranch(pipeline, src, bank, bank_fragment, (control_snk, control_src), gate_attack_length, gate_hold_length, block_duration, nxydump_segment = None, fir_stride = None, control_peak_time = None, reconstruction_segment_list = None):
	"""!
	Make a single slice of one branch of the lloid graph, e.g. one instrument and one
	template bank fragment. For details see: http://arxiv.org/abs/1107.2665

	Specifically this implements the filtering of multirate svd basis and
	(conditional) resampling and reconstruction of the physical SNR

	@param pipeline The gstreamer pipeline in which to place this graph
	@param src The source of data for this graph provided by a gstreamer element
	@param bank The template bank class
	@param bank_fragment The specific fragment (time slice) of the template bank in question
	@param (control_snk, control_src) An optional tuple of the sink and source elements for a graph that will construct a control time series for the gate which aggregates the orthogonal snrs from each template slice. This is used to conditionally reconstruct the physical SNR of interesting times
	@param gate_attack_length The attack length in samples for the lal_gate element that controls the reconstruction of physical SNRs
	@param gate_hold_length The hold length in samples for the lal_gate element that controls the reconstruction of physical SNRs
	@param block_duration The characteristic buffer size that is passed around, which is useful for constructing queues.
	@param nxydump_segment Not used
	@param fir_stride The target length of output buffers from lal_firbank in seconds.  Directly effects latency.  Making this short will force time-domain convolution. Otherwise FFT convolution will be done to save CPU cycles, but at higher latency.
	@param control_peak_time The window over which to find peaks in the control signal.  Shorter windows increase computational cost but probably also detection efficiency.
	@param reconstruction_segment_list A segment list object that describes when the control signal should be on.  This can be useful in e.g., only reconstructing physical SNRS around the time of injections, which can save an enormous amount of CPU time.
	"""
	logname = "%s_%.2f.%.2f" % (bank.logname, bank_fragment.start, bank_fragment.end)

	#
	# FIR filter bank.  low frequency branches use time-domain
	# convolution, high-frequency branches use FFT convolution with a
	# block stride given by fir_stride.
	#

	latency = -int(round(bank_fragment.start * bank_fragment.rate))
	block_stride = int(fir_stride * bank_fragment.rate)

	# we figure an fft costs ~5 logN flops where N is duration + block
	# stride.  Time domain costs N * block_stride. So if block stride is
	# less than about 5logN you might as well do time domain filtering
	# FIXME This calculation should probably be made more rigorous
	time_domain = 5 * numpy.log2((bank_fragment.end - bank_fragment.start) * bank_fragment.rate + block_stride) > block_stride

	src = pipeparts.mkfirbank(pipeline, src, latency = latency, fir_matrix = bank_fragment.orthogonal_template_bank, block_stride = block_stride, time_domain = time_domain)
	src = pipeparts.mkchecktimestamps(pipeline, src, "timestamps_%s_after_firbank" % logname)
	# uncomment reblock if you ever use really big ffts and want to cut them down a bit
	#src = pipeparts.mkreblock(pipeline, src, block_duration = control_peak_time * gst.SECOND)
	#src = pipeparts.mkchecktimestamps(pipeline, src, "timestamps_%s_after_firbank_reblock" % logname)
	#src = pipeparts.mktee(pipeline, src)	# comment-out the tee below if this is uncommented
	#pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, src), "orthosnr_%s.dump" % logname, segment = nxydump_segment)

	#pipeparts.mkvideosink(pipeline, pipeparts.mkcapsfilter(pipeline, pipeparts.mkhistogramplot(pipeline, src), "video/x-raw-rgb, width=640, height=480, framerate=1/4"))
	#pipeparts.mkogmvideosink(pipeline, pipeparts.mkcapsfilter(pipeline, pipeparts.mkchannelgram(pipeline, pipeparts.mkqueue(pipeline, src), plot_width = .125), "video/x-raw-rgb, width=640, height=480, framerate=64/1"), "orthosnr_channelgram_%s.ogv" % logname, verbose = True)

	#
	# compute weighted sum-of-squares, feed to sum-of-squares
	# aggregator
	#

	if control_snk is not None and control_src is not None:
		src = pipeparts.mktee(pipeline, src)	# comment-out if the tee above is uncommented
		elem = pipeparts.mkqueue(pipeline, pipeparts.mksumsquares(pipeline, src, weights = bank_fragment.sum_of_squares_weights), max_size_buffers = 0, max_size_bytes = 0, max_size_time = block_duration)
		elem = pipeparts.mkchecktimestamps(pipeline, elem, "timestamps_%s_after_sumsquare" % logname)
		elem = pipeparts.mkresample(pipeline, elem, quality = 9)
		elem = pipeparts.mkchecktimestamps(pipeline, elem, "timestamps_%s_after_sumsquare_resampler" % logname)
		elem.link(control_snk)

		#
		# use sum-of-squares aggregate as gate control for
		# orthogonal SNRs
		#
		# FIXME the queuing in this code is broken.  enabling this
		# causes lock-ups.  there is latency in the construction of
		# the composite detection statistic due to the need for
		# resampling and because of the peak finding, therefore the
		# source stream going into the gate needs to be queued
		# until the composite detection statistic can catch up, but
		# we don't know by how much (what we've got here doesn't
		# work).  there should not be a need to buffer the control
		# stream at all, nor is there a need for the queuing to
		# accomodate different latencies for different SNR slices,
		# but we do require that all elements correctly modify
		# segments events to reflect their latency and the actual
		# time stamps of the data stream they will produce.  it
		# might be that not all elements are doing that correctly.
		#
		# FIXME we ignore the bank.gate_threshold parameter and
		# just use 1e-100.  this change was made when peak finding
		# was put into the composite detector

		src = pipeparts.mkgate(
			pipeline,
			pipeparts.mkqueue(pipeline, src, max_size_buffers = 0, max_size_bytes = 0, max_size_time = 1 * int((2. * control_peak_time + float(abs(gate_attack_length) + abs(gate_hold_length)) / bank_fragment.rate) * Gst.SECOND)),
			threshold = 1e-100,
			attack_length = gate_attack_length,
			hold_length = gate_hold_length,
			control = control_src
		)
		src = pipeparts.mkchecktimestamps(pipeline, src, "timestamps_%s_after_gate" % logname)

	#
	# optionally add a segment src and gate to only reconstruct around
	# injections
	#
	# FIXME:  set the names of these gates so their segments can be
	# collected later?  or else propagate this segment list into the
	# output some other way.

	if reconstruction_segment_list is not None:
		src = datasource.mksegmentsrcgate(pipeline, src, reconstruction_segment_list, invert_output = False)

	#
	# reconstruct physical SNRs
	#

	if bank_fragment.mix_matrix is not None:
		src = pipeparts.mkmatrixmixer(pipeline, src, matrix = bank_fragment.mix_matrix)
		src = pipeparts.mkchecktimestamps(pipeline, src, "timestamps_%s_after_matrixmixer" % logname)

	#
	# done
	#
	# FIXME:  find a way to use less memory without this hack

	del bank_fragment.orthogonal_template_bank
	del bank_fragment.sum_of_squares_weights
	del bank_fragment.mix_matrix

	return src


def mkLLOIDhoftToSnrSlices(pipeline, hoftdict, bank, control_snksrc, block_duration, verbose = False, logname = "", nxydump_segment = None, fir_stride = None, control_peak_time = None, snrslices = None, reconstruction_segment_list = None):
	"""!
	Build the pipeline fragment that creates the SnrSlices associated with
	different sample rates from hoft.

	@param reconstruction_segment_list A segment list object that describes when the control signal should be on.  This can be useful in e.g., only reconstructing physical SNRS around the time of injections, which can save an enormous amount of CPU time.
	"""
	#
	# parameters
	#

	rates = sorted(bank.get_rates())
	output_rate = max(rates)

	# work out the upsample factors for the attack and hold calculations below
	upsample_factor = dict((rate, rates[i+1] / rate) for i, rate in list(enumerate(rates))[:-1])
	upsample_factor[output_rate] = 0

	autocorrelation_length = bank.autocorrelation_bank.shape[1]
	assert autocorrelation_length % 2 == 1
	autocorrelation_latency = -(autocorrelation_length - 1) / 2

	#
	# loop over template bank slices
	#

	branch_heads = dict((rate, set()) for rate in rates)
	for bank_fragment in bank.bank_fragments:
		# The attack and hold width has three parts
		#
		# 1) The audio resampler filter: 16 comes from the size of
		# the audioresampler filter in samples for the next highest
		# rate at quality 1. Note it must then be converted to the size
		# at the current rate using the upsample_factor dictionary
		# (which is 0 if you are at the max rate).
		#
		# 2) The chisq latency.  You must have at least latency number
		# of points before and after (at the maximum sample rate) to
		# compute the chisq
		#
		# 3) A fudge factor to get the width of the peak.  FIXME this
		# is just set to 1/8th of a second
		peak_half_width = upsample_factor[bank_fragment.rate] * 16 + int(math.ceil(-autocorrelation_latency * (float(bank_fragment.rate) / output_rate))) + int(math.ceil(bank_fragment.rate / 8.))
		branch_heads[bank_fragment.rate].add(mkLLOIDbranch(
			pipeline,
			# FIXME:  the size isn't ideal:  the correct value
			# depends on how much data is accumulated in the
			# firbank element, and the value here is only
			# approximate and not tied to the fir bank
			# parameters so might not work if those change
			pipeparts.mkqueue(pipeline, hoftdict[bank_fragment.rate], max_size_bytes = 0, max_size_buffers = 0, max_size_time = int((1 * fir_stride + int(math.ceil(bank.filter_length))) * Gst.SECOND)),
			bank,
			bank_fragment,
			control_snksrc,
			peak_half_width,
			peak_half_width,
			block_duration,
			nxydump_segment = nxydump_segment,
			fir_stride = fir_stride,
			control_peak_time = control_peak_time,
			reconstruction_segment_list = reconstruction_segment_list
		))

	#
	# if the calling code has requested copies of the snr
	# slices, sum together the highest sample rate streams and tee
	# them off here.  this needs to be done before constructing the
	# adder network below in order to have access to this snr slice by
	# itself.  if we put this off until later it'll have all the other
	# snrs added into it before we get a chance to tee it off
	#

	if snrslices is not None:
		rate, heads = output_rate, branch_heads[output_rate]
		if len(heads) > 1:
			#
			# this sample rate has more than one snr stream.
			# sum them together in an adder, which becomes the
			# head of the stream at this sample rate
			#

			branch_heads[rate] = pipeparts.mkadder(pipeline, (pipeparts.mkqueue(pipeline, head, max_size_bytes = 0, max_size_buffers = 0, max_size_time = block_duration) for head in heads))
		else:
			#
			# this sample rate has only one stream.  it's the
			# head of the stream at this sample rate
			#

			branch_heads[rate], = heads
		branch_heads[rate] = pipeparts.mktee(pipeline, branch_heads[rate])
		snrslices[rate] = pipeparts.mktogglecomplex(pipeline, branch_heads[rate])

		#
		# the code below expects an interable of elements
		#

		branch_heads[rate] = set([branch_heads[rate]])

	#
	# sum the snr streams
	#

	if True:	# FIXME:  make conditional on time-slice \chi^{2}
		next_rate = dict(zip(rates, rates[1:]))
	else:
		next_rate = dict((rate, output_rate) for rate in rates if rate != output_rate)

	for rate, heads in sorted(branch_heads.items()):
		if len(heads) > 1:
			#
			# this sample rate has more than one snr stream.
			# sum them together in an adder, which becomes the
			# head of the stream at this sample rate
			#

			branch_heads[rate] = pipeparts.mkadder(pipeline, heads)
			branch_heads[rate] = pipeparts.mkchecktimestamps(pipeline, branch_heads[rate], "timestamps_%s_after_%d_snr_adder" % (logname, rate))
		else:
			#
			# this sample rate has only one stream.  it's the
			# head of the stream at this sample rate
			#

			branch_heads[rate], = heads

		#
		# resample if needed
		#

		if rate in next_rate:
			# NOTE: quality = 1 requires that the template
			# slices are padded such that the Nyquist frequency
			# is 1.5 times the highest frequency of the time
			# slice.  NOTE: the adder (that comes downstream of
			# this) isn't quite smart enough to negotiate a
			# common format among its upstream peers so the
			# capsfilter is still required.
			# NOTE uncomment this line to restore audioresample for
			# upsampling
			#branch_heads[rate] = pipeparts.mkcapsfilter(pipeline, pipeparts.mkresample(pipeline, branch_heads[rate], quality = 1), "audio/x-raw, rate=%d" % next_rate[rate])
			branch_heads[rate] = pipeparts.mkcapsfilter(pipeline, pipeparts.mkinterpolator(pipeline, branch_heads[rate]), "audio/x-raw, rate=%d" % next_rate[rate])
			branch_heads[rate] = pipeparts.mkchecktimestamps(pipeline, branch_heads[rate], "timestamps_%s_after_%d_to_%d_snr_resampler" % (logname, rate, next_rate[rate]))

		#
		# if the calling code has requested copies of the snr
		# slices, tee that off here.  remember that we've already
		# got the highest sample rate slice from above
		#

		if snrslices is not None and rate != output_rate:
			branch_heads[rate] = pipeparts.mktee(pipeline, branch_heads[rate])
			snrslices[rate] = pipeparts.mktogglecomplex(pipeline, branch_heads[rate])

		#
		# chain to next adder if this isn't the final answer
		#

		if rate in next_rate:
			branch_heads[next_rate[rate]].add(branch_heads.pop(rate))

	#
	# done
	#

	snr, = branch_heads.values()	# make sure we've summed down to one stream
	return pipeparts.mktogglecomplex(pipeline, snr)


def mkLLOIDSnrSlicesToTimeSliceChisq(pipeline, branch_heads, bank, block_duration):
	"""!
	Build pipeline fragment that computes the TimeSliceChisq from SnrSlices.
	"""
	#
	# parameters
	#

	rates = sorted(bank.get_rates())

	#
	# compute the chifacs for each rate, store in ascending order in rate
	#

	chifacsdict = dict((rate, []) for rate in rates)
	for bank_fragment in bank.bank_fragments:
		chifacsdict[bank_fragment.rate].append(bank_fragment.chifacs)
	chifacs = []
	for rate, facs in sorted(chifacsdict.items()):
		chifacs.append(facs[0][0::2])
		chifacs[-1] += facs[0][1::2]
		for fac in facs[1:]:
			chifacs[-1] += fac[0::2]
			chifacs[-1] += fac[1::2]
		chifacs[-1] /= 2.

	#
	# create timeslicechisq element and add chifacs as a property
	#

	chisq = Gst.ElementFactory.make("lal_timeslicechisq", None)
	pipeline.add(chisq)

	#
	# link the snrslices to the timeslicechisq element in ascending order in rate
	#

	for rate, snrslice in sorted(branch_heads.items()):
		pipeparts.mkqueue(pipeline, snrslice,  max_size_bytes = 0, max_size_buffers = 0, max_size_time = block_duration).link(chisq)

	#
	# set chifacs-matrix property, needs to be done after snrslices are linked in
	#

	chisq.set_property("chifacs-matrix", chifacs)

	return pipeparts.mkqueue(pipeline, chisq, max_size_bytes = 0, max_size_buffers = 0, max_size_time = block_duration)


def mkLLOIDSnrChisqToTriggers(pipeline, snr, chisq, bank, verbose = False, nxydump_segment = None, logname = ""):
	"""!
	Build pipeline fragment that converts single detector SNR and Chisq
	into triggers.
	"""
	#
	# trigger generator and progress report
	#

	head = pipeparts.mktriggergen(pipeline, snr, chisq, template_bank_filename = bank.template_bank_filename, snr_threshold = bank.snr_threshold, sigmasq = bank.sigmasq)
	# FIXME:  add ability to choose this
	# "lal_blcbctriggergen", {"bank-filename": bank.template_bank_filename, "snr-thresh": bank.snr_threshold, "sigmasq": bank.sigmasq}
	if verbose:
		head = pipeparts.mkprogressreport(pipeline, head, "progress_xml_%s" % logname)

	#
	# done
	#

	return head


#
# many instruments, many template banks
#


def mkLLOIDmulti(pipeline, detectors, banks, psd, psd_fft_length = 32, ht_gate_threshold = float("inf"), veto_segments = None, verbose = False, nxydump_segment = None, chisq_type = 'autochisq', track_psd = False, fir_stride = 16, control_peak_time = 2, block_duration = Gst.SECOND, reconstruction_segment_list = None):
	"""!
	The multiple instrument, multiple bank LLOID algorithm
	"""

	#
	# check for unrecognized chisq_types, non-unique bank IDs
	#

	if chisq_type not in ['autochisq', 'timeslicechisq']:
		raise ValueError("chisq_type must be either 'autochisq' or 'timeslicechisq', given %s" % chisq_type)
	if any(tuple(iterutils.nonuniq(bank.bank_id for bank in banklist)) for banklist in banks.values()):
		raise ValueError("bank IDs are not unique: %s" % "; ".join("for %s: %s" % (instrument, iterutils.nonuniq(bank.bank_id for bank in banklist)) for instrument, banklist in banks.items()))

	#
	# construct dictionaries of whitened, conditioned, down-sampled
	# h(t) streams.  NOTE:  we assume all banks for each instrument
	# were generated with the same processed PSD for that instrument
	# and just extract the first without checking that this assumption
	# is correct
	#

	assert psd_fft_length % 4 == 0, "psd_fft_length (= %g) must be multiple of 4" % psd_fft_length
	hoftdicts = {}
	for instrument in detectors.channel_dict:
		src, statevector, dqvector = datasource.mkbasicsrc(pipeline, detectors, instrument, verbose)
		hoftdicts[instrument] = multirate_datasource.mkwhitened_multirate_src(
			pipeline,
			src = src,
			rates = set(rate for bank in banks[instrument] for rate in bank.get_rates()),
			instrument = instrument,
			psd = psd[instrument],
			psd_fft_length = psd_fft_length,
			ht_gate_threshold = ht_gate_threshold,
			veto_segments = veto_segments[instrument] if veto_segments is not None else None,
			nxydump_segment = nxydump_segment,
			track_psd = track_psd,
			width = 32,
			statevector = statevector,
			dqvector = dqvector,
			fir_whiten_reference_psd = banks[instrument][0].processed_psd
		)

	#
	# build gate control branches
	#

	if control_peak_time > 0:
		control_branch = {}
		for instrument, bank in [(instrument, bank) for instrument, banklist in banks.items() for bank in banklist]:
			suffix = "%s%s" % (instrument, (bank.logname and "_%s" % bank.logname or ""))
			if instrument != "H2":
				control_branch[(instrument, bank.bank_id)] = mkcontrolsnksrc(pipeline, max(bank.get_rates()), verbose = verbose, suffix = suffix, control_peak_samples = control_peak_time * max(bank.get_rates()))
				#pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, control_branch[(instrument, bank.bank_id)][1]), "control_%s.dump" % suffix, segment = nxydump_segment)
	else:
		control_branch = None

	#
	# construct trigger generators
	#

	itacac_dict = {}
	for i, (instrument, bank) in enumerate([(instrument, bank) for instrument, banklist in banks.items() for bank in banklist]):
		suffix = "%s%s" % (instrument, (bank.logname and "_%s" % bank.logname or ""))
		if control_branch is not None:
			if instrument != "H2":
				control_snksrc = control_branch[(instrument, bank.bank_id)]
			else:
				control_snksrc = (None, control_branch[("H1", bank.bank_id)][1])
		else:
			control_snksrc = (None, None)
		if chisq_type == 'timeslicechisq':
			snrslices = {}
		else:
			snrslices = None
		snr = mkLLOIDhoftToSnrSlices(
			pipeline,
			hoftdicts[instrument],
			bank,
			control_snksrc,
			block_duration,
			verbose = verbose,
			logname = suffix,
			nxydump_segment = nxydump_segment,
			control_peak_time = control_peak_time,
			fir_stride = fir_stride,
			snrslices = snrslices,
			reconstruction_segment_list = reconstruction_segment_list
		)
		snr = pipeparts.mkchecktimestamps(pipeline, snr, "timestamps_%s_snr" % suffix)
		# uncomment this tee if the diagnostic sinks below are
		# needed
		#snr = pipeparts.mktee(pipeline, snr)
		if chisq_type == 'autochisq':
			# FIXME don't hardcode
			# peak finding window (n) in samples is 1/4 second at max rate, ie max(rates) / 4
			# NOTE the snr min set in the diststats file is 3.5,
			# but 4 is about the lowest we can do stably for
			# coincidence online...
			#nsamps_window = max(max(bank.get_rates()) / 4, 256) # FIXME stupid hack
			nsamps_window = 1 * max(bank.get_rates())
			if bank.bank_id not in itacac_dict:
				itacac_dict[bank.bank_id] = pipeparts.mkgeneric(pipeline, None, "lal_itacac")

			head = itacac_dict[bank.bank_id]
			pad = head.get_request_pad("sink%d" % len(head.sinkpads))
			if instrument == 'H1' or instrument == 'L1':
				for prop, val in [("n", nsamps_window), ("snr-thresh", 4.0), ("bank_filename", bank.template_bank_filename), ("sigmasq", bank.sigmasq), ("autocorrelation_matrix", pipeio.repack_complex_array_to_real(bank.autocorrelation_bank)), ("autocorrelation_mask", bank.autocorrelation_mask)]:
					pad.set_property(prop, val)
				snr.srcpads[0].link(pad)
			else:
				for prop, val in [("n", nsamps_window), ("snr-thresh", 4.0), ("bank_filename", bank.template_bank_filename), ("sigmasq", bank.sigmasq), ("autocorrelation_matrix", pipeio.repack_complex_array_to_real(bank.autocorrelation_bank)), ("autocorrelation_mask", bank.autocorrelation_mask)]:
					pad.set_property(prop, val)
				snr.srcpads[0].link(pad)
		else:
			raise NotImplementedError("Currently only 'autochisq' is supported")
		# FIXME:  find a way to use less memory without this hack
		del bank.autocorrelation_bank
		#pipeparts.mknxydumpsink(pipeline, pipeparts.mktogglecomplex(pipeline, pipeparts.mkqueue(pipeline, snr)), "snr_%s.dump" % suffix, segment = nxydump_segment)
		#pipeparts.mkogmvideosink(pipeline, pipeparts.mkcapsfilter(pipeline, pipeparts.mkchannelgram(pipeline, pipeparts.mkqueue(pipeline, snr), plot_width = .125), "video/x-raw-rgb, width=640, height=480, framerate=64/1"), "snr_channelgram_%s.ogv" % suffix, audiosrc = pipeparts.mkaudioamplify(pipeline, pipeparts.mkqueue(pipeline, hoftdict[max(bank.get_rates())], max_size_time = 2 * int(math.ceil(bank.filter_length)) * gst.SECOND), 0.125), verbose = True)

	#
	# done
	#

	assert any(itacac_dict.values())
	if verbose:
		for bank_id, head in itacac_dict.items():
			itacac_dict[bank_id] = pipeparts.mkprogressreport(pipeline, head, "progress_xml_bank_%s" % bank_id)
	return itacac_dict

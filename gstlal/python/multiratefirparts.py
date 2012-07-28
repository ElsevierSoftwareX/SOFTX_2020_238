# Copyright (C) 2009--2012  Kipp Cannon, Chad Hanna, Drew Keppel
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


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


import math
import sys
import numpy


# The following snippet is taken from http://gstreamer.freedesktop.org/wiki/FAQ#Mypygstprogramismysteriouslycoredumping.2Chowtofixthis.3F
import pygtk
pygtk.require("2.0")
import gobject
gobject.threads_init()
import pygst
pygst.require('0.10')
import gst


from glue import iterutils
from gstlal import bottle
from gstlal import pipeparts
from gstlal import reference_psd
from gstlal import simulation
from gstlal import lloidparts

#
# many instruments, many template banks, no SVD
#


def mkFIRbranch(pipeline, src, bank, bank_fragment, nxydump_segment = None, fir_stride = None, block_duration = None):
	logname = "%s_%.2f.%.2f" % (bank.logname, bank_fragment.start, bank_fragment.end)

	#
	# FIR filter bank.  low frequency branches use time-domain
	# convolution, high-frequency branches use FFT convolution with a
	# block stride of 4 s.
	#
	# FIXME:  why the -1?  without it the pieces don't match but I
	# don't understand where this offset comes from.  it might really
	# need to be here, or it might be a symptom of a bug elsewhere.
	# figure this out.

	latency = -int(round(bank_fragment.start * bank_fragment.rate)) - 1
	block_stride = fir_stride * bank_fragment.rate
	
	# we figure an fft costs ~5 logN flops where N is duration + block
	# stride.  For each chunk you have to do a forward and a reverse fft.
	# Time domain costs N * block_stride. So if block stride is less than
	# about 10logN you might as well do time domain filtering
	# FIXME This calculation should probably be made more rigorous
	time_domain = 10 * numpy.log2((bank_fragment.end - bank_fragment.start) * bank_fragment.rate + block_stride) > block_stride

	src = pipeparts.mkfirbank(pipeline, src, latency = latency, fir_matrix = bank_fragment.orthogonal_template_bank, block_stride = block_stride, time_domain = time_domain)
	src = pipeparts.mkchecktimestamps(pipeline, src, "timestamps_%s_after_firbank" % logname)

	# FIXME:  teach the collectpads object not to wait for buffers on pads
	# whose segments have not yet been reached by the input on the other
	# pads.  then this large queue buffer will not be required because
	# streaming can begin through the downstream adders without waiting for
	# input from all upstream elements.

	src = pipeparts.mkqueue(pipeline, src, max_size_buffers = 0, max_size_bytes = 0, max_size_time = 2 * block_duration)

	#
	# done
	#
	# FIXME:  find a way to use less memory without this hack

	del bank_fragment.orthogonal_template_bank
	del bank_fragment.sum_of_squares_weights
	del bank_fragment.mix_matrix

	return src



def mkhoftToSnrSlices(pipeline, hoftdict, bank, verbose = False, logname = "", nxydump_segment = None, fir_stride = None, block_duration = None, snrslices = None):
	"""Build the pipeline fragment that creates the SnrSlices associated with different sample rates from hoft."""
	#
	# parameters
	#

	rates = sorted(bank.get_rates())
	output_rate = max(rates)

	autocorrelation_length = bank.autocorrelation_bank.shape[1]
	autocorrelation_latency = -(autocorrelation_length - 1) / 2

	#
	# loop over template bank slices
	#

	branch_heads = dict((rate, set()) for rate in rates)
	for bank_fragment in bank.bank_fragments:
		branch_heads[bank_fragment.rate].add(mkFIRbranch(
			pipeline,
			# FIXME:  the size isn't ideal:  the correct value
			# depends on how much data is accumulated in the
			# firbank element, and the value here is only
			# approximate and not tied to the fir bank
			# parameters so might not work if those change
			pipeparts.mkqueue(pipeline, pipeparts.mkdrop(pipeline, hoftdict[bank_fragment.rate], int(round((bank.filter_length - bank_fragment.end) * bank_fragment.rate))), max_size_bytes = 0, max_size_buffers = 0, max_size_time = (1 * fir_stride + int(math.ceil(bank.filter_length))) * gst.SECOND),
			bank,
			bank_fragment,
			nxydump_segment = nxydump_segment,
			fir_stride = fir_stride,
			block_duration = block_duration
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

			branch_heads[rate] = gst.element_factory_make("lal_adder")
			branch_heads[rate].set_property("sync", True)
			pipeline.add(branch_heads[rate])
			for head in heads:
				pipeparts.mkqueue(pipeline, head, max_size_bytes = 0, max_size_buffers = 0, max_size_time = block_duration).link(branch_heads[rate])
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

			branch_heads[rate] = gst.element_factory_make("lal_adder")
			branch_heads[rate].set_property("sync", True)
			pipeline.add(branch_heads[rate])
			for head in heads:
				pipeparts.mkqueue(pipeline, head, max_size_bytes = 0, max_size_buffers = 0, max_size_time = 1 * block_duration).link(branch_heads[rate])
			# FIXME capsfilter shouldn't be needed remove when adder is fixed
			branch_heads[rate] = pipeparts.mkcapsfilter(pipeline, branch_heads[rate], "audio/x-raw-float, rate=%d" % rate)
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
			branch_heads[rate] = pipeparts.mkcapsfilter(pipeline, pipeparts.mkresample(pipeline, branch_heads[rate], quality = 4), "audio/x-raw-float, rate=%d" % next_rate[rate])
			branch_heads[rate] = pipeparts.mknofakedisconts(pipeline, branch_heads[rate])	# FIXME:  remove when resampler is patched
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


def mkFIRmulti(pipeline, seekevent, detectors, banks, psd, psd_fft_length = 8, data_source = None, injection_filename = None, ht_gate_threshold = None, veto_segments = None, verbose = False, nxydump_segment = None, frame_segments = None, chisq_type = 'autochisq', track_psd = False, fir_stride = 16, block_duration = gst.SECOND, state_vector_on_off_dict = {"H1" : (0x7, 0x160), "L1" : (0x7, 0x160), "V1" : (0x67, 0x100)}):
	#
	# check for unrecognized chisq_types, non-unique bank IDs
	#

	if chisq_type not in ['autochisq', 'timeslicechisq']:
		raise ValueError, "chisq_type must be either 'autochisq' or 'timeslicechisq', given %s" % (chisq_type)
	# FIXME:  uncomment when glue.iterutils.nonuniq is available
	#if tuple(iterutils.nonuniq(bank.bank_id for bank in banks)):
	#	raise ValueError("bank IDs %s are not unique" % ", ".join(iterutils.nonuniq(bank.bank_id for bank in banks)))

	#
	# extract segments from the injection file for selected reconstruction
	#

	if injection_filename is not None:
		inj_seg_list = simulation.sim_inspiral_to_segment_list(injection_filename)
	else:
		inj_seg_list = None

	#
	# construct dictionaries of whitened, conditioned, down-sampled
	# h(t) streams
	#

	hoftdicts = {}
	for instrument in detectors:
		rates = set(rate for bank in banks[instrument] for rate in bank.get_rates()) # FIXME what happens if the rates are not the same?
		src = lloidparts.mkLLOIDbasicsrc(pipeline, seekevent, instrument, detectors[instrument], data_source = data_source, injection_filename = injection_filename, frame_segments = frame_segments[instrument], state_vector_on_off_dict = state_vector_on_off_dict, verbose = verbose)
		# let the frame reader and injection code run in a
		# different thread than the whitener, etc.,
		src = pipeparts.mkqueue(pipeline, src, max_size_bytes = 0, max_size_buffers = 0, max_size_time = block_duration)
		if veto_segments is not None:
			hoftdicts[instrument] = lloidparts.mkLLOIDsrc(pipeline, src, rates, instrument, psd = psd[instrument], psd_fft_length = psd_fft_length, seekevent = seekevent, ht_gate_threshold = ht_gate_threshold, veto_segments = veto_segments[instrument], nxydump_segment = nxydump_segment, track_psd = track_psd, block_duration = block_duration)
		else:
			hoftdicts[instrument] = lloidparts.mkLLOIDsrc(pipeline, src, rates, instrument, psd = psd[instrument], psd_fft_length = psd_fft_length, seekevent = seekevent, ht_gate_threshold = ht_gate_threshold, nxydump_segment = nxydump_segment, track_psd = track_psd, block_duration = block_duration)

	#
	# construct trigger generators
	#

	triggersrcs = set()
	for instrument, bank in [(instrument, bank) for instrument, banklist in banks.items() for bank in banklist]:
		suffix = "%s%s" % (instrument, (bank.logname and "_%s" % bank.logname or ""))
		if chisq_type == 'timeslicechisq':
			snrslices = {}
		else:
			snrslices = None
		snr = mkhoftToSnrSlices(
			pipeline,
			hoftdicts[instrument],
			bank,
			verbose = verbose,
			logname = suffix,
			nxydump_segment = nxydump_segment,
			fir_stride = fir_stride,
			block_duration = block_duration,
			snrslices = snrslices
		)
		snr = pipeparts.mkchecktimestamps(pipeline, snr, "timestamps_%s_snr" % suffix)
		# FIXME you get a different trigger generator depending on the chisq calculation :/
		if chisq_type == 'autochisq':
			# FIXME don't hardcode
			# peak finding window (n) in samples is one second at max rate, ie max(rates)
			head = pipeparts.mkitac(pipeline, snr, max(rates), bank.template_bank_filename, autocorrelation_matrix = bank.autocorrelation_bank, snr_thresh = bank.snr_threshold, sigmasq = bank.sigmasq)
			if verbose:
				head = pipeparts.mkprogressreport(pipeline, head, "progress_xml_%s" % suffix)
			triggersrcs.add(head)
		else:
			chisq = mkLLOIDSnrSlicesToTimeSliceChisq(pipeline, snrslices, bank, block_duration)
			triggersrcs.add(lloidparts.mkLLOIDSnrChisqToTriggers(
				pipeline,
				pipeparts.mkqueue(pipeline, snr, max_size_bytes = 0, max_size_buffers = 0, max_size_time = 1 * block_duration),
				chisq,
				bank,
				verbose = verbose,
				nxydump_segment = nxydump_segment,
				logname = suffix
			))
		# FIXME:  find a way to use less memory without this hack
		del bank.autocorrelation_bank

	#
	# if there is more than one trigger source, synchronize the streams
	# with a multiqueue then use an n-to-1 adapter to combine into a
	# single stream
	#

	assert len(triggersrcs) > 0
	return triggersrcs



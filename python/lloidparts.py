# Copyright (C) 2010  Kipp Cannon, Chad Hanna
# Copyright (C) 2009  Kipp Cannon, Chad Hanna
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


from gstlal import pipeparts
from gstlal import pipeio
from gstlal import cbc_template_fir
from gstlal import simulation
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

def mkelems_fast(bin, *pipedesc):
	elems = []
	elem = None
	for arg in pipedesc:
		if isinstance(arg, dict):
			for tup in arg.iteritems():
				elem.set_property(*tup)
		else:
			if elem is not None:
				if elem.get_parent() is None:
					bin.add(elem)
				if len(elems) > 0:
					elems[-1].link(elem)
				elems.append(elem)
			if isinstance(arg, gst.Element):
				elem = arg
			else:
				elem = gst.element_factory_make(arg)
	if elem is not None:
		if elem.get_parent() is None:
			bin.add(elem)
		if len(elems) > 0:
			elems[-1].link(elem)
		elems.append(elem)
	return elems


#
# =============================================================================
#
#                              Pipeline Metadata
#
# =============================================================================
#


class DetectorData(object):
	# default block_size = 16384 samples/second * 8 bytes/sample * 8
	# second
	def __init__(self, frame_cache, channel, block_size = 16384 * 8 * 512):
		self.frame_cache = frame_cache
		self.channel = channel
		self.block_size = block_size


#
# =============================================================================
#
#                              Pipeline Elements
#
# =============================================================================
#

#
# gate controlled by a segment source
#

def mksegmentsrcgate(pipeline, src, segment_list, threshold, seekevent = None, invert_output = False):
	segsrc = pipeparts.mksegmentsrc(pipeline, segment_list, invert_output=invert_output)
	if seekevent is not None:
		if segsrc.set_state(gst.STATE_READY) != gst.STATE_CHANGE_SUCCESS:
			raise RuntimeError, "Element %s did not want to enter ready state" % segsrc.get_name()
		if not segsrc.send_event(seekevent):
			raise RuntimeError, "Element %s did not handle seek event" % segsrc.get_name()
	return pipeparts.mkgate(pipeline, src, threshold = threshold, control = pipeparts.mkqueue(pipeline, segsrc))


#
# LLOID Pipeline handler
#


class LLOIDHandler(object):
	def __init__(self, mainloop, pipeline):
		self.mainloop = mainloop
		self.pipeline = pipeline

		bus = pipeline.get_bus()
		bus.add_signal_watch()
		bus.connect("message", self.on_message)

	def on_message(self, bus, message):
		if message.type == gst.MESSAGE_EOS:
			self.pipeline.set_state(gst.STATE_NULL)
			self.mainloop.quit()
		elif message.type == gst.MESSAGE_ERROR:
			gerr, dbgmsg = message.parse_error()
			self.pipeline.set_state(gst.STATE_NULL)
			self.mainloop.quit()
			sys.exit("error (%s:%d '%s'): %s" % (gerr.domain, gerr.code, gerr.message, dbgmsg))


def seek_event_for_gps(gps_start_time, gps_end_time, flags = 0):
	"""Create a new seek event for a given gps_start_time and gps_end_time,
	with optional flags.  gps_start_time and gps_end_time may be provided as
	instances of LIGOTimeGPS, as doubles, or as floats."""

	def seek_args_for_gps(gps_time):
		"""Convenience routine to convert a GPS time to a seek type and a
		GStreamer timestamp."""

		if gps_time is None or gps_time == -1:
			return (gst.SEEK_TYPE_NONE, -1) # -1 == gst.CLOCK_TIME_NONE
		elif hasattr(gps_time, 'ns'):
			return (gst.SEEK_TYPE_SET, gps_time.ns())
		else:
			return (gst.SEEK_TYPE_SET, long(float(gps_time) * gst.SECOND))

	start_type, start_time = seek_args_for_gps(gps_start_time)
	stop_type, stop_time   = seek_args_for_gps(gps_end_time)

	return gst.event_new_seek(1., gst.FORMAT_TIME, flags,
		start_type, start_time, stop_type, stop_time)


#
# sum-of-squares aggregator
#


def mkcontrolsnksrc(pipeline, rate, verbose = False, suffix = None):
	args = (
		"lal_adder", {"sync": True},
		"capsfilter", {"caps": gst.Caps("audio/x-raw-float, rate=%d" % rate)}
	)
	if verbose:
		args += ("progressreport", {"name": "progress_sumsquares%s" % (suffix and "_%s" % suffix or "")})
	args += ("tee",)

	elems = mkelems_fast(pipeline, *args)
	return elems[0], elems[-1]


#
# data source
#

def mkLLOIDbasicsrc(pipeline, seekevent, instrument, detector, fake_data = False, online_data = False, injection_filename = None, verbose = False):
	#
	# data source and progress report
	#

	if fake_data:
		args = ("lal_fakeligosrc", {"instrument": instrument, "channel-name": detector.channel, "blocksize": detector.block_size})
	elif online_data:
		args = ("lal_onlinehoftsrc", {"instrument": instrument})
	else:
		args = ("lal_framesrc", {"instrument": instrument, "blocksize": detector.block_size, "location": detector.frame_cache, "channel-name": detector.channel})
	args += ("audioconvert",)

	if verbose:
		args += ("progressreport", {"name": "progress_src_%s" % instrument})

	if injection_filename is not None:
		args += ("lal_simulation", {"xml-location": injection_filename})

	elems = mkelems_fast(pipeline, *args)

	if elems[0].set_state(gst.STATE_READY) != gst.STATE_CHANGE_SUCCESS:
		raise RuntimeError, "Element %s did not want to enter ready state" % elems[0].get_name()
	if not elems[0].send_event(seekevent):
		raise RuntimeError, "Element %s did not handle seek event" % elems[0].get_name()

	return elems[-1]


def mkLLOIDsrc(pipeline, src, rates, psd=None, psd_fft_length=8, veto_segments=None, seekevent=None):
	"""Build pipeline stage to whiten and downsample h(t)."""

	#
	# down-sample to highest of target sample rates.  note:  there is
	# no check that this is, infact, *down*-sampling.  if the source
	# time series has a lower sample rate this will up-sample the data.
	# up-sampling will probably interact poorly with the whitener as it
	# will likely add (possibly significant) numerical noise when it
	# amplifies the non-existant high-frequency components
	#

	source_rate = max(rates)
	elems = mkelems_fast(pipeline,
		src,
		"queue", # FIXME: I think we can remove this queue.
		"audioresample", {"quality": 9},
		"capsfilter", {"caps": gst.Caps("audio/x-raw-float, rate=%d" % source_rate)},
		"lal_checktimestamps", {"name": "timestamps_before_whitener"},
		"lal_whiten", {"fft-length": psd_fft_length, "zero-pad": 0, "average-samples": 64, "median-samples": 7},
		"lal_nofakedisconts", {"silent": True}
	)

	if psd is None:
		# use running average PSD
		elems[-2].set_property("psd-mode", 0)
	else:
		# use fixed PSD
		elems[-2].set_property("psd-mode", 1)

		#
		# install signal handler to retrieve \Delta f when it is
		# known, resample the user-supplied PSD, and install it
		# into the whitener.
		#

		def psd_resolution_changed(elem, pspec, psd):
			# get frequency resolution and number of bins
			delta_f = elem.get_property("delta-f")
			n = int(round(elem.get_property("f-nyquist") / delta_f) + 1)
			# interpolate and install PSD
			psd = cbc_template_fir.interpolate_psd(psd, delta_f)
			elem.set_property("mean-psd", psd.data[:n])

		elems[-2].connect_after("notify::f-nyquist", psd_resolution_changed, psd)
		elems[-2].connect_after("notify::delta-f", psd_resolution_changed, psd)

	head = elems[-1]

	# optionally add vetoes
	if veto_segments is not None:
		head = mksegmentsrcgate(pipeline, head, veto_segments, threshold=0.1, seekevent=seekevent, invert_output=True)

	# put in the final tee
	elems = mkelems_fast(pipeline, head, "tee")

	
	#
	# down-sample whitened time series to remaining target sample rates
	# while applying an amplitude correction to adjust for low-pass
	# filter roll-off.  we also scale by \sqrt{original rate / new
	# rate}.  this is done to preserve the square magnitude of the time
	# series --- the inner product of the time series with itself.
	# really what we want is for
	#
	#	\int v_{1}(t) v_{2}(t) \diff t
	#		\approx \sum v_{1}(t) v_{2}(t) \Delta t
	#
	# to be preserved across different sample rates, i.e. for different
	# \Delta t.  what we do is rescale the time series and ignore
	# \Delta t, so we put 1/2 factor of the ratio of the \Delta t's
	# into the h(t) time series here, and, later, another 1/2 factor
	# into the template when it gets downsampled.
	#
	# By design, the input to the orthogonal filter
	# banks is pre-whitened, so it is unit variance over short periods of time.
	# However, resampling it reduces the variance by a small, sample rate
	# dependent factor.  The audioamplify element applies a correction factor
	# that restores the input's unit variance.
	#

	quality = 9
	head = {source_rate: elems[-1]}
	for rate in sorted(rates, reverse = True)[1:]:	# all but the highest rate
		head[rate] = mkelems_fast(pipeline,
			elems[-1],
			"audioamplify", {"clipping-method": 3, "amplification": 1/math.sqrt(pipeparts.audioresample_variance_gain(quality, source_rate, rate))},
			"audioresample", {"quality": quality},
			"capsfilter", {"caps": gst.Caps("audio/x-raw-float, rate=%d" % rate)},
			"lal_checktimestamps", {"name": "timestamps_after_downsample_to_%d" % rate},
			"tee"
		)[-1]

	#
	# done.  return value is a dictionary of tee elements indexed by
	# sample rate
	#

	#for rate, elem in head.items():
	#	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, elem), "src_%d.dump" % rate, segment = nxydump_segment)
	return head


#
# one instrument, one template bank
#


def mkLLOIDbranch(pipeline, src, bank, bank_fragment, (control_snk, control_src), gate_attack_length, gate_hold_length, seekevent=None, inj_seg_list=None):
	logname = "%s_%d_%d" % (bank.logname, bank_fragment.start, bank_fragment.end)

	#
	# FIR filter bank
	#
	# FIXME:  why the -1?  without it the pieces don't match but I
	# don't understand where this offset comes from.  it might really
	# need to be here, or it might be a symptom of a bug elsewhere.
	# figure this out.

	src = mkelems_fast(pipeline,
		src,
		"lal_firbank", {"block-length-factor": 10, "latency": -int(round(bank_fragment.start * bank_fragment.rate)) - 1, "fir-matrix": bank_fragment.orthogonal_template_bank},
		"lal_nofakedisconts", {"silent": True},
		"lal_reblock",
		"tee"
	)[-1]
	#pipeparts.mkvideosink(pipeline, pipeparts.mkcapsfilter(pipeline, pipeparts.mkhistogram(pipeline, src), "video/x-raw-rgb, width=640, height=480, framerate=1/4"))
	#pipeparts.mkogmvideosink(pipeline, pipeparts.mkcapsfilter(pipeline, pipeparts.mkchannelgram(pipeline, pipeparts.mkqueue(pipeline, src), plot_width = .125), "video/x-raw-rgb, width=640, height=480, framerate=64/1"), "orthosnr_channelgram_%s.ogv" % logname, verbose = True)

	#
	# compute weighted sum-of-squares, feed to sum-of-squares
	# aggregator
	#

	mkelems_fast(pipeline,
		src,
		"lal_sumsquares", {"weights": bank_fragment.sum_of_squares_weights},
		"queue",
		"audioresample", {"quality": 9},
		"lal_checktimestamps", {"name": "timestamps_%s_after_sumsquare_resampler" % logname},
		control_snk
	)

	#
	# use sum-of-squares aggregate as gate control for orthogonal SNRs
	#

	elems = mkelems_fast(pipeline,
		"lal_gate", {"threshold": bank.gate_threshold, "attack-length": gate_attack_length, "hold-length": gate_hold_length},
		"lal_checktimestamps", {"name": "timestamps_%s_after_gate" % logname},

		#
		# buffer orthogonal SNRs
		#
		# FIXME:  teach the collectpads object not to wait for buffers on
		# pads whose segments have not yet been reached by the input on the
		# other pads.  then this large queue buffer will not be required
		# because streaming can begin through the downstream adders without
		# waiting for input from all upstream elements.

		"queue", {"max-size-buffers": 0, "max-size-bytes": 0, "max-size-time": 2 * gst.SECOND},

		#
		# reconstruct physical SNRs
		#

		"lal_matrixmixer", {"matrix": bank_fragment.mix_matrix},
	)

	#
	# optionally add a segment src and gate to only reconstruct around injections
	#

	if inj_seg_list is not None:
		control_src = mksegmentsrcgate(pipeline, pipeparts.mkqueue(pipeline, control_src), inj_seg_list, threshold=0.1, seekevent=seekevent, invert_output=False)

	mkelems_fast(pipeline, src, "queue", {"max-size-buffers": 0, "max-size-bytes": 0, "max-size-time": 5 * gst.SECOND})[-1].link_pads("src", elems[0], "sink")
	mkelems_fast(pipeline, control_src, "queue", {"max-size-buffers": 0, "max-size-bytes": 0, "max-size-time": 1 * gst.SECOND})[-1].link_pads("src", elems[0], "control")

	#
	# done
	#
	# FIXME:  find a way to use less memory without this hack

	del bank_fragment.orthogonal_template_bank
	del bank_fragment.sum_of_squares_weights
	del bank_fragment.mix_matrix

	return elems[-1]


def mkLLOIDhoftToSnr(pipeline, hoftdict, instrument, bank, control_snksrc, verbose = False, nxydump_segment = None, seekevent=None, inj_seg_list=None):
	"""Build pipeline fragment that converts h(t) to SNR."""

	logname = "%s%s" % (instrument, (bank.logname and "_%s" % bank.logname or ""))

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
		branch_heads[bank_fragment.rate].add(mkLLOIDbranch(
			pipeline,
			# FIXME:  the size isn't ideal:  the correct value
			# depends on how much data is accumulated in the
			# firbank element, and the value here is only
			# approximate and not tied to the fir bank
			# parameters so might not work if those change
			mkelems_fast(pipeline,
				hoftdict[bank_fragment.rate],
				"lal_delay", {"delay": int(round((bank.filter_length - bank_fragment.end) * bank_fragment.rate))},
				"queue", {"max-size-bytes": 0, "max-size-buffers": 0, "max-size-time": 4 * int(math.ceil(bank.filter_length)) * gst.SECOND}
			)[-1],
			bank,
			bank_fragment,
			control_snksrc,
			int(math.ceil(-autocorrelation_latency * (float(bank_fragment.rate) / output_rate))),
			int(math.ceil(-autocorrelation_latency * (float(bank_fragment.rate) / output_rate))),
			seekevent = seekevent,
			inj_seg_list = inj_seg_list
		))

	#
	# sum snrs with common sample rates
	#

	snr = None
	next_rate = dict(zip(rates, rates[1:]))
	for rate, heads in sorted(branch_heads.items()):
		#
		# hook matrix mixers to an adder
		#

		branchsnr = mkelems_fast(pipeline, "lal_adder", {"sync": True})[-1]
		for head in heads:
			pipeparts.mkqueue(pipeline, head, max_size_time = gst.SECOND).link(branchsnr)
		#logname = "%s_%d_%d" % (bank.logname, bank_fragment.start, bank_fragment.end)
		#branchsnr = pipeparts.mktee(pipeline, branchsnr)
		#pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, branchsnr), "snr_%s_%02d.dump" % (logname, bank_fragment.start), segment = nxydump_segment)

		#
		# if this isn't the highest sample rate, attach a resampler
		# and add to heads for next highest sample rate.  otherwise
		# this adder's output is the fully-reconstructed snr stream
		#

		if rate in next_rate:
			branchsnr = pipeparts.mkresample(pipeline, branchsnr, quality = 4)
			#branchsnr = pipeparts.mkchecktimestamps(pipeline, branchsnr, "timestamps_%s_after_snr_resampler" % logname)
			branch_heads[next_rate[rate]].add(branchsnr)
		else:
			assert snr is None	# only one snr element allowed
			snr = branchsnr
	assert snr is not None	# did we identify the snr element?

	#
	# snr
	#

	return pipeparts.mktogglecomplex(pipeline, snr)


def mkLLOIDsnrToTriggers(pipeline, snr, bank, verbose = False, nxydump_segment = None, logname = None):
	"""Build pipeline fragment that converts single detector SNR into triggers."""
	#
	# parameters
	#

	output_rate = max(bank.get_rates())
	autocorrelation_length = bank.autocorrelation_bank.shape[1]
	autocorrelation_latency = -(autocorrelation_length - 1) / 2

	snr = pipeparts.mktee(pipeline, snr)

	#pipeparts.mknxydumpsink(pipeline, pipeparts.mktogglecomplex(pipeline, pipeparts.mkqueue(pipeline, snr)), "snr_%s.dump" % logname, segment = nxydump_segment)
	#pipeparts.mkogmvideosink(pipeline, pipeparts.mkcapsfilter(pipeline, pipeparts.mkchannelgram(pipeline, pipeparts.mkqueue(pipeline, snr), plot_width = .125), "video/x-raw-rgb, width=640, height=480, framerate=64/1"), "snr_channelgram_%s.ogv" % logname, audiosrc = pipeparts.mkaudioamplify(pipeline, pipeparts.mkqueue(pipeline, hoftdict[output_rate], max_size_time = 2 * int(math.ceil(bank.filter_length)) * gst.SECOND), 0.125), verbose = True)

	#
	# \chi^{2}
	#

	chisq = mkelems_fast(pipeline,
		snr,
		"queue",
		"lal_autochisq", {"autocorrelation-matrix": pipeio.repack_complex_array_to_real(bank.autocorrelation_bank), "latency": autocorrelation_latency, "snr-thresh": bank.snr_threshold}
	)[-1]
	#chisq = pipeparts.mktee(pipeline, chisq)
	#pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, chisq), "chisq_%s.dump" % logname, segment = nxydump_segment)
	# FIXME:  find a way to use less memory without this hack
	del bank.autocorrelation_bank

	#
	# trigger generator and progress report
	#

	head = mkelems_fast(pipeline,
		chisq,
		"lal_triggergen", {"bank-filename": bank.template_bank_filename, "snr-thresh": bank.snr_threshold, "sigmasq": bank.sigmasq},
		#"lal_blcbctriggergen", {"bank-filename": bank.template_bank_filename, "snr-thresh": bank.snr_threshold, "sigmasq": bank.sigmasq},
	)[-1]
	mkelems_fast(pipeline, snr, "queue", head)
	if verbose:
		head = mkelems_fast(pipeline, head, "progressreport", {"name": "progress_xml_%s" % logname})[-1]

	#
	# done
	#

	return head


#
# many instruments, many template banks
#


def mkLLOIDmulti(pipeline, seekevent, detectors, banks, psd, psd_fft_length = 8, fake_data = False, online_data = False, injection_filename = None, veto_segments=None, verbose = False, nxydump_segment = None):
	#
	# xml stream aggregator
	#

	# Input selector breaks seeks.  For a single detector and single template bank,
	# we don't need an input selector.  This is an ugly kludge to make seeks work
	# in this special (and very high priority) case.
	needs_input_selector = (len(detectors) * len(banks) > 1)
	if needs_input_selector:
		nto1 = mkelems_fast(pipeline, "input-selector", {"select-all": True})[-1]

	#
	# extract segments from the injection file for selected reconstruction
	#

	if injection_filename is not None:
		#inj_seg_list = simulation.sim_inspiral_to_segment_list(injection_filename)
		inj_seg_list = None
	else:
		inj_seg_list = None

	#
	# loop over instruments and template banks
	#

	for instrument in detectors:
		rates = set(rate for bank in banks for rate in bank.get_rates())
		head = mkLLOIDbasicsrc(pipeline, seekevent, instrument, detectors[instrument], fake_data=fake_data, online_data=online_data, injection_filename=injection_filename, verbose=verbose)
		
		#
		# check to see if we have veto segments, if so extract the segments for the current instrument
		#

		if veto_segments:
			hoftdict = mkLLOIDsrc(pipeline, head, rates, psd=psd, psd_fft_length=psd_fft_length, seekevent=seekevent, veto_segments=veto_segments[instrument])
		else:
			hoftdict = mkLLOIDsrc(pipeline, head, rates, psd=psd, psd_fft_length=psd_fft_length, veto_segments=None)

		for bank in banks:
			control_snksrc = mkcontrolsnksrc(pipeline, max(bank.get_rates()), verbose = verbose, suffix = "%s%s" % (instrument, (bank.logname and "_%s" % bank.logname or "")))
			#pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, control_snksrc[1]), "control_%s.dump" % bank.logname, segment = nxydump_segment)
			snr = mkLLOIDhoftToSnr(
				pipeline,
				hoftdict,
				instrument,
				bank,
				control_snksrc,
				verbose = verbose,
				nxydump_segment = nxydump_segment,
				seekevent=seekevent,
				inj_seg_list = inj_seg_list
			)
			head = mkLLOIDsnrToTriggers(
				pipeline,
				snr,
				bank,
				verbose = verbose,
				nxydump_segment = nxydump_segment,
				logname = "%s%s" % (instrument, (bank.logname and "_%s" % bank.logname or ""))
			)
			if needs_input_selector:
				mkelems_fast(pipeline, head, "queue", nto1)

	#
	# done
	#

	if needs_input_selector:
		return nto1
	else:
		return mkelems_fast(pipeline, head, "queue")[-1]

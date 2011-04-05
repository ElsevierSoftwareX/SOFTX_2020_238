# Copyright (C) 2009--2011  Kipp Cannon, Chad Hanna
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
from gstlal import cbc_template_fir
import math
import sys


# The following snippet is taken from http://gstreamer.freedesktop.org/wiki/FAQ#Mypygstprogramismysteriouslycoredumping.2Chowtofixthis.3F
import pygtk
pygtk.require("2.0")
import gobject
gobject.threads_init()
import pygst
pygst.require('0.10')
import gst


#
# =============================================================================
#
#                              Pipeline Metadata
#
# =============================================================================
#


class DetectorData(object):
	# default block_size = 16384 samples/second * 8 bytes/sample * 512
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

	return gst.event_new_seek(1., gst.FORMAT_TIME, flags, start_type, start_time, stop_type, stop_time)


#
# sum-of-squares aggregator
#


def mkcontrolsnksrc(pipeline, rate, verbose = False, suffix = None):
	snk = gst.element_factory_make("lal_adder")
	snk.set_property("sync", True)
	pipeline.add(snk)
	src = pipeparts.mkcapsfilter(pipeline, snk, "audio/x-raw-float, rate=%d" % rate)
	if verbose:
		src = pipeparts.mkprogressreport(pipeline, src, "progress_sumsquares%s" % (suffix and "_%s" % suffix or ""))
	src = pipeparts.mktee(pipeline, src)
	return snk, src


#
# data source
#


def mkLLOIDbasicsrc(pipeline, seekevent, instrument, detector, fake_data = False, online_data = False, injection_filename = None, verbose = False):
	#
	# data source
	#

	if fake_data:
		assert not online_data
		src = pipeparts.mkfakeLIGOsrc(pipeline, instrument = instrument, channel_name = detector.channel, blocksize = detector.block_size)
	elif online_data:
		assert not fake_data
		src = pipeparts.mkonlinehoftsrc(pipeline, instrument)
	else:
		src = pipeparts.mkframesrc(pipeline, location = detector.frame_cache, instrument = instrument, channel_name = detector.channel, blocksize = detector.block_size)

	#
	# seek the data source
	#

	if src.set_state(gst.STATE_READY) != gst.STATE_CHANGE_SUCCESS:
		raise RuntimeError, "Element %s did not want to enter ready state" % src.get_name()
	if not src.send_event(seekevent):
		raise RuntimeError, "Element %s did not handle seek event" % src.get_name()

	#
	# convert single precision streams to double precision if needed
	#

	src = pipeparts.mkaudioconvert(pipeline, src)

	#
	# progress report
	#

	if verbose:
		src = pipeparts.mkprogressreport(pipeline, src, "progress_src_%s" % instrument)

	#
	# optional injections
	#

	if injection_filename is not None:
		src = pipeparts.mkinjections(pipeline, src, injection_filename)

	#
	# done
	#

	return src


def mkLLOIDsrc(pipeline, src, rates, psd=None, psd_fft_length = 8, veto_segments = None, seekevent = None, nxydump_segment = None):
	"""Build pipeline stage to whiten and downsample h(t)."""

	#
	# down-sample to highest of target sample rates.  we include a caps
	# filter upstream of the resampler to ensure that this is, infact,
	# *down*-sampling.  if the source time series has a lower sample
	# rate than the highest target sample rate the resampler will
	# become an upsampler, and the result will likely interact poorly
	# with the whitener as it tries to ampify the non-existant
	# high-frequency components, possibly adding significant numerical
	# noise to its output.  if you see errors about being unable to
	# negotiate a format from this stage in the pipeline, it is because
	# you are asking for output sample rates that are higher than the
	# sample rate of your data source.
	#
	# note that when downsampling it might be possible for the
	# audioresampler to produce spurious DISCONT flags.  if it receives
	# an input buffer too small to produce any new output samples, it
	# might emit a DISCONT flag on the next output buffer (this
	# behaviour has not been confirmed, but it is a known quirk of the
	# base class from which it is derived).  however, in this
	# application the buffer sizes produced by the data source are too
	# large to trigger this behaviour whether or not the resampler
	# might ever do it
	#

	quality = 9
	head = pipeparts.mkcapsfilter(pipeline, src, "audio/x-raw-float, rate=[%d,MAX]" % max(rates))
	head = pipeparts.mkresample(pipeline, head, quality = quality)
	head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw-float, rate=%d" % max(rates))

	#
	# construct whitener.  this element must be followed by a
	# nofakedisconts element.
	#

	head = pipeparts.mkwhiten(pipeline, head, fft_length = psd_fft_length, zero_pad = 0, average_samples = 64, median_samples = 7)
	if psd is None:
		# use running average PSD
		head.set_property("psd-mode", 0)
	else:
		# use fixed PSD
		head.set_property("psd-mode", 1)

		#
		# install signal handler to retrieve \Delta f and
		# f_{Nyquist} whenever they are known and/or change,
		# resample the user-supplied PSD, and install it into the
		# whitener.
		#

		def psd_resolution_changed(elem, pspec, psd):
			# get frequency resolution and number of bins
			delta_f = elem.get_property("delta-f")
			n = int(round(elem.get_property("f-nyquist") / delta_f) + 1)
			# interpolate and install PSD
			psd = cbc_template_fir.interpolate_psd(psd, delta_f)
			elem.set_property("mean-psd", psd.data[:n])

		head.connect_after("notify::f-nyquist", psd_resolution_changed, psd)
		head.connect_after("notify::delta-f", psd_resolution_changed, psd)
	head = pipeparts.mknofakedisconts(pipeline, head, silent = True)

	#
	# optionally add vetoes
	#

	if veto_segments is not None:
		segsrc = pipeparts.mksegmentsrc(pipeline, veto_segments, invert_output=True)
		if seekevent is not None:
			if segsrc.set_state(gst.STATE_READY) != gst.STATE_CHANGE_SUCCESS:
				raise RuntimeError, "Element %s did not want to enter ready state" % segsrc.get_name()
			if not segsrc.send_event(seekevent):
				raise RuntimeError, "Element %s did not handle seek event" % segsrc.get_name()
		q = pipeparts.mkqueue(pipeline, segsrc, max_size_buffers=0, max_size_bytes=0, max_size_time=(1 * gst.SECOND))
		head = pipeparts.mkgate(pipeline, head, threshold = 0.1, control = q)

	#
	# tee for highest sample rate stream
	#

	head = {max(rates): pipeparts.mktee(pipeline, head)}

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
	# by design, the output of the whitener is a unit-variance time
	# series.  however, downsampling it reduces the variance due to the
	# removal of some frequency components.  we require the input to
	# the orthogonal filter banks to be unit variance, therefore a
	# correction factor is applied via an audio amplify element to
	# adjust for the reduction in variance due to the downsampler.
	#

	for rate in sorted(set(rates))[:-1]:	# all but the highest rate
		head[rate] = pipeparts.mktee(
			pipeline,
			pipeparts.mkcapsfilter(
				pipeline,
				pipeparts.mkresample(
					pipeline,
					pipeparts.mkaudioamplify(
						pipeline,
						head[max(rates)],
						1/math.sqrt(pipeparts.audioresample_variance_gain(quality, max(rates), rate))
					),
					quality = quality
				),
				caps = "audio/x-raw-float, rate=%d" % rate
			)
		)

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


def mkLLOIDbranch(pipeline, src, bank, bank_fragment, (control_snk, control_src), gate_attack_length, gate_hold_length, nxydump_segment = None):
	logname = "%s_%d_%d" % (bank.logname, bank_fragment.start, bank_fragment.end)

	#
	# FIR filter bank
	#
	# FIXME:  why the -1?  without it the pieces don't match but I
	# don't understand where this offset comes from.  it might really
	# need to be here, or it might be a symptom of a bug elsewhere.
	# figure this out.

	src = pipeparts.mkfirbank(pipeline, src, latency = -int(round(bank_fragment.start * bank_fragment.rate)) - 1, fir_matrix = bank_fragment.orthogonal_template_bank)
	src = pipeparts.mkreblock(pipeline, src, block_duration = 1 * gst.SECOND)
	src = pipeparts.mktee(pipeline, src)
	#pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, src), "orthosnr_%s.dump" % logname, segment = nxydump_segment)

	#pipeparts.mkvideosink(pipeline, pipeparts.mkcapsfilter(pipeline, pipeparts.mkhistogram(pipeline, src), "video/x-raw-rgb, width=640, height=480, framerate=1/4"))
	#pipeparts.mkogmvideosink(pipeline, pipeparts.mkcapsfilter(pipeline, pipeparts.mkchannelgram(pipeline, pipeparts.mkqueue(pipeline, src), plot_width = .125), "video/x-raw-rgb, width=640, height=480, framerate=64/1"), "orthosnr_channelgram_%s.ogv" % logname, verbose = True)

	#
	# compute weighted sum-of-squares, feed to sum-of-squares
	# aggregator
	#

	elem = pipeparts.mkresample(pipeline, pipeparts.mkqueue(pipeline, pipeparts.mksumsquares(pipeline, src, weights = bank_fragment.sum_of_squares_weights)), quality = 9)
	elem = pipeparts.mkchecktimestamps(pipeline, elem, "timestamps_%s_after_sumsquare_resampler" % logname)
	elem.link(control_snk)

	#
	# use sum-of-squares aggregate as gate control for orthogonal SNRs
	#

	src = pipeparts.mkgate(pipeline, pipeparts.mkqueue(pipeline, src), threshold = bank.gate_threshold, attack_length = gate_attack_length, hold_length = gate_hold_length, control = pipeparts.mkqueue(pipeline, control_src, max_size_buffers = 0, max_size_bytes = 0, max_size_time = 1 * gst.SECOND))
	src = pipeparts.mkchecktimestamps(pipeline, src, "timestamps_%s_after_gate" % logname)

	#
	# buffer orthogonal SNRs
	#
	# FIXME:  teach the collectpads object not to wait for buffers on
	# pads whose segments have not yet been reached by the input on the
	# other pads.  then this large queue buffer will not be required
	# because streaming can begin through the downstream adders without
	# waiting for input from all upstream elements.

	src = pipeparts.mkqueue(pipeline, src, max_size_buffers = 0, max_size_bytes = 0, max_size_time = 2 * gst.SECOND)

	#
	# reconstruct physical SNRs
	#

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


def mkLLOIDhoftToSnr(pipeline, hoftdict, bank, control_snksrc, verbose = False, logname = "", nxydump_segment = None):
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
			pipeparts.mkqueue(pipeline, pipeparts.mkdelay(pipeline, hoftdict[bank_fragment.rate], int(round((bank.filter_length - bank_fragment.end) * bank_fragment.rate))), max_size_bytes = 0, max_size_buffers = 0, max_size_time = 4 * int(math.ceil(bank.filter_length)) * gst.SECOND),
			bank,
			bank_fragment,
			control_snksrc,
			int(math.ceil(-autocorrelation_latency * (float(bank_fragment.rate) / output_rate))),
			int(math.ceil(-autocorrelation_latency * (float(bank_fragment.rate) / output_rate))),
			nxydump_segment = nxydump_segment
		))

	#
	# sum the snr streams, adding resamplers where needed.  at the
	# start of this loop, branch_heads is a dictionary mapping sample
	# rates to sets of matrix mixer elements.  we loop over the
	# contents of the dictionary from lowest to highest sample rate
	#

	next_rate = dict(zip(rates, rates[1:]))
	for rate, heads in sorted(branch_heads.items()):
		#
		# hook all matrix mixers that share a common sample rate to
		# an adder.  the adder replaces the set of matrix mixers as
		# the new "head" associated with the sample rate
		#

		branch_heads[rate] = gst.element_factory_make("lal_adder")
		branch_heads[rate].set_property("sync", True)
		pipeline.add(branch_heads[rate])
		for head in heads:
			pipeparts.mkqueue(pipeline, head, max_size_time = gst.SECOND).link(branch_heads[rate])

		#
		# if the reconstructed snr upto and including this sample
		# rate is needed for something, like an early warning
		# detection statistic, or to dump it to a file, it can be
		# tee'ed off here
		#

		#branch_heads[rate] = pipeparts.mktee(pipeline, branch_heads[rate])
		#pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, branch_heads[rate]), "snr_%s_%d.dump" % (logname, rate), segment = nxydump_segment)

		#
		# if this isn't the highest sample rate, attach a resampler
		# to the adder and add the resampler to the set of heads
		# for next highest sample rate.
		#

		if rate in next_rate:
			branch_heads[next_rate[rate]].add(pipeparts.mkresample(pipeline, branch_heads[rate], quality = 4))

	#
	# the adder for the highest sample rate provides the final
	# reconstructed snr
	#

	return pipeparts.mktogglecomplex(pipeline, branch_heads[max(rates)])


def mkLLOIDsnrToTriggers(pipeline, snr, bank, verbose = False, nxydump_segment = None, logname = ""):
	"""Build pipeline fragment that converts single detector SNR into triggers."""
	#
	# parameters
	#

	output_rate = max(bank.get_rates())
	autocorrelation_length = bank.autocorrelation_bank.shape[1]
	autocorrelation_latency = -(autocorrelation_length - 1) / 2

	#
	# \chi^{2}
	#

	snr = pipeparts.mktee(pipeline, snr)
	chisq = pipeparts.mkautochisq(pipeline, pipeparts.mkqueue(pipeline, snr), autocorrelation_matrix = bank.autocorrelation_bank, latency = autocorrelation_latency, snr_thresh = bank.snr_threshold)
	# FIXME:  find a way to use less memory without this hack
	del bank.autocorrelation_bank

	#chisq = pipeparts.mktee(pipeline, chisq)
	#pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, chisq), "chisq_%s.dump" % logname, segment = nxydump_segment)

	#
	# trigger generator and progress report
	#

	head = pipeparts.mktriggergen(pipeline, pipeparts.mkqueue(pipeline, snr), chisq, template_bank_filename = bank.template_bank_filename, snr_threshold = bank.snr_threshold, sigmasq = bank.sigmasq)
	if verbose:
		head = pipeparts.mkprogressreport(pipeline, head, "progress_xml_%s" % logname)

	#
	# done
	#

	return head


#
# many instruments, many template banks
#


def mkLLOIDmulti(pipeline, seekevent, detectors, banks, psd, psd_fft_length = 8, fake_data = False, online_data = False, injection_filename = None, veto_segments=None, verbose = False, nxydump_segment = None):
	#
	# loop over instruments and template banks
	#

	rates = set(rate for bank in banks for rate in bank.get_rates())
	triggersrc = set()
	for instrument in detectors:
		src = mkLLOIDbasicsrc(pipeline, seekevent, instrument, detectors[instrument], fake_data = fake_data, online_data = online_data, injection_filename = injection_filename, verbose = verbose)
		# let the frame reader and injection code run in a
		# different thread than the whitener, etc.,
		src = pipeparts.mkqueue(pipeline, src)
		if veto_segments:
			hoftdict = mkLLOIDsrc(pipeline, src, rates, psd = psd, psd_fft_length = psd_fft_length, seekevent = seekevent, veto_segments = veto_segments[instrument], nxydump_segment = nxydump_segment)
		else:
			hoftdict = mkLLOIDsrc(pipeline, src, rates, psd = psd, psd_fft_length = psd_fft_length, veto_segments = None, nxydump_segment = nxydump_segment)
		for bank in banks:
			suffix = "%s%s" % (instrument, (bank.logname and "_%s" % bank.logname or ""))
			control_snksrc = mkcontrolsnksrc(pipeline, max(bank.get_rates()), verbose = verbose, suffix = suffix)
			#pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, control_snksrc[1]), "control_%s.dump" % suffix, segment = nxydump_segment)
			head = mkLLOIDhoftToSnr(
				pipeline,
				hoftdict,
				bank,
				control_snksrc,
				verbose = verbose,
				logname = suffix,
				nxydump_segment = nxydump_segment
			)
			head = pipeparts.mkchecktimestamps(pipeline, head, "timestamps_%s_snr" % suffix)
			#head = pipeparts.mktee(pipeline, head)
			#pipeparts.mknxydumpsink(pipeline, pipeparts.mktogglecomplex(pipeline, pipeparts.mkqueue(pipeline, head)), "snr_%s.dump" % suffix, segment = nxydump_segment)
			#pipeparts.mkogmvideosink(pipeline, pipeparts.mkcapsfilter(pipeline, pipeparts.mkchannelgram(pipeline, pipeparts.mkqueue(pipeline, head), plot_width = .125), "video/x-raw-rgb, width=640, height=480, framerate=64/1"), "snr_channelgram_%s.ogv" % suffix, audiosrc = pipeparts.mkaudioamplify(pipeline, pipeparts.mkqueue(pipeline, hoftdict[max(rates)], max_size_time = 2 * int(math.ceil(bank.filter_length)) * gst.SECOND), 0.125), verbose = True)
			triggersrc.add(mkLLOIDsnrToTriggers(
				pipeline,
				head,
				bank,
				verbose = verbose,
				nxydump_segment = nxydump_segment,
				logname = suffix
			))

	#
	# if there is more than one trigger source, use an n-to-1 adapter
	# to combine into a single stream
	#
	# FIXME:  it has been reported that the input selector breaks
	# seeks.  confirm and fix if needed

	assert len(triggersrc) > 0
	if len(triggersrc) > 1:
		# FIXME:  input-selector in 0.10.32 no longer has the
		# "select-all" feature.  need to get this re-instated
		assert False	# force crash until input-selector problem is fixed
		nto1 = gst.element_factory_make("input-selector")
		nto1.set_property("select-all", True)
		pipeline.add(nto1)
		for head in triggersrc:
			pipeparts.mkqueue(pipeline, head).link(nto1)
		triggersrc = nto1
	else:
		# len(triggersrc) == 1
		triggersrc, = triggersrc

	#
	# done
	#

	return triggersrc

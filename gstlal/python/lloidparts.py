# Copyright (C) 2009--2011  Kipp Cannon, Chad Hanna, Drew Keppel
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
# gate controlled by a segment source
#


def mksegmentsrcgate(pipeline, src, segment_list, threshold, seekevent = None, invert_output = False):
	segsrc = pipeparts.mksegmentsrc(pipeline, segment_list, invert_output=invert_output)
	if seekevent is not None:
		if segsrc.set_state(gst.STATE_READY) != gst.STATE_CHANGE_SUCCESS:
			raise RuntimeError("Element %s did not want to enter ready state" % segsrc.get_name())
		if not segsrc.send_event(seekevent):
			raise RuntimeError("Element %s did not handle seek event" % segsrc.get_name())
	return pipeparts.mkgate(pipeline, src, threshold = threshold, control = pipeparts.mkqueue(pipeline, segsrc))

#
# gate controlled by h(t)
#


def mkhtgate(pipeline, src, threshold = 8.0, attack_length = -128, hold_length = -128):
	# FIXME someday explore a good bandpass filter
	# src = pipeparts.mkaudiochebband(pipeline, src, low_frequency, high_frequency)
	src = pipeparts.mktee(pipeline, src)
	control = pipeparts.mkqueue(pipeline, src, max_size_time = 0, max_size_bytes = 0, max_size_buffers = 0)
	input = pipeparts.mkqueue(pipeline, src, max_size_time = gst.SECOND, max_size_bytes = 0, max_size_buffers = 0)
	return pipeparts.mkgate(pipeline, input, threshold = threshold, control = control, attack_length = attack_length, hold_length = hold_length, invert_control = True)


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
		elif message.type == gst.MESSAGE_INFO:
			gerr, dbgmsg = message.parse_info()
			print >>sys.stderr, "info (%s:%d '%s'): %s" % (gerr.domain, gerr.code, gerr.message, dbgmsg)
		elif message.type == gst.MESSAGE_WARNING:
			gerr, dbgmsg = message.parse_warning()
			print >>sys.stderr, "warning (%s:%d '%s'): %s" % (gerr.domain, gerr.code, gerr.message, dbgmsg)
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


def mkcontrolsnksrc(pipeline, rate, verbose = False, suffix = None, inj_seg_list = None, seekevent = None, control_peak_samples = None):
	#
	# start with an adder and caps filter to select a sample rate
	#

	snk = gst.element_factory_make("lal_adder")
	snk.set_property("sync", True)
	pipeline.add(snk)
	src = pipeparts.mkcapsfilter(pipeline, snk, "audio/x-raw-float, rate=%d" % rate)

	#
	# Add a peak finder on the control signal sample number
	#

	if control_peak_samples is not None:
		src = pipeparts.mkpeak(pipeline, src, control_peak_samples)

	#
	# optionally add a segment src and gate to only reconstruct around
	# injections
	#

	if inj_seg_list is not None:
		src = mksegmentsrcgate(pipeline, src, inj_seg_list, threshold = 0.1, seekevent = seekevent, invert_output = False)

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


#
# data source
#


class get_gate_state(object):
	# monitor state vector transitions, export via web
	# interface
	def __init__(self, elem, msg = None, verbose = False):
		self.verbose = verbose
		self.current_segment = "unknown"
		self.segment_start = "unknown"
		if msg is not None:
			self.msg = "%s: " % msg
		else:
			self.msg = ""
		elem.connect("start", self.sighandler, "on")
		elem.connect("stop", self.sighandler, "off")

	def sighandler(self, elem, timestamp, segment_type):
		self.current_segment = segment_type
		self.segment_start = "%.9f" % (timestamp / 1e9)
		if self.verbose:
			print >>sys.stderr, "%s: %sstate transition: %s" % (elem.get_name(), self.msg, self.text().strip())

	def text(self):
		return "%s @ %s\n" % (self.current_segment, self.segment_start)


def mkLLOIDbasicsrc(pipeline, seekevent, instrument, detector, data_source = "frames", injection_filename = None, frame_segments = None, state_vector_on_off_dict = {"H1" : (0x7, 0x160), "L1" : (0x7, 0x160), "V1" : (0x67, 0x100)}, verbose = False):
	#
	# data source
	#

	# First process fake data or frame data
	if data_source == "white":
		# seek events have to be given to these since the element returned is a tag inject
		src = pipeparts.mkfakesrcseeked(pipeline, instrument, detector.channel, seekevent, blocksize = detector.block_size)
	elif data_source == "silence":
		# seek events have to be given to these since the element returned is a tag inject
		src = pipeparts.mkfakesrcseeked(pipeline, instrument, detector.channel, seekevent, blocksize = detector.block_size, wave = 4)
	elif data_source == 'LIGO':
		src = pipeparts.mkfakeLIGOsrc(pipeline, instrument = instrument, channel_name = detector.channel, blocksize = detector.block_size)
	elif data_source == 'AdvLIGO':
		src = pipeparts.mkfakeadvLIGOsrc(pipeline, instrument = instrument, channel_name = detector.channel, blocksize = detector.block_size)
	elif data_source == 'AdvVirgo':
		src = pipeparts.mkfakeadvvirgosrc(pipeline, instrument = instrument, channel_name = detector.channel, blocksize = detector.block_size)
	elif data_source == "frames":
		if instrument == "V1":
			#FIXME Hack because virgo often just uses "V" in the file names rather than "V1".  We need to sieve on "V"
			src = pipeparts.mkframesrc(pipeline, location = detector.frame_cache, instrument = instrument, cache_src_regex = "V", channel_name = detector.channel, blocksize = detector.block_size, segment_list = frame_segments)
		else:
			src = pipeparts.mkframesrc(pipeline, location = detector.frame_cache, instrument = instrument, cache_dsc_regex = instrument, channel_name = detector.channel, blocksize = detector.block_size, segment_list = frame_segments)
	# Next process online data, fake data must be None for this to have gotten this far
	elif data_source == "online":
		# See https://wiki.ligo.org/DAC/ER2DataDistributionPlan#LIGO_Online_DQ_Channel_Specifica
		state_vector_on_bits, state_vector_off_bits = state_vector_on_off_dict[instrument]

		# FIXME:  be careful hard-coding shared-memory partition
		# FIXME make wait_time adjustable through web interface or command line or both
		src = pipeparts.mklvshmsrc(pipeline, shm_name = {"H1": "LHO_Data", "H2": "LHO_Data", "L1": "LLO_Data", "V1": "VIRGO_Data"}[instrument], wait_time = 120)
		src = pipeparts.mkframecppchanneldemux(pipeline, src, do_file_checksum = True, skip_bad_files = True)

		# strain
		strain = pipeparts.mkqueue(pipeline, None, max_size_buffers = 0, max_size_bytes = 0, max_size_time = gst.SECOND * 60 * 10) # 10 minutes of buffering
		pipeparts.src_deferred_link(src, "%s:%s" % (instrument, detector.channel), strain.get_pad("sink"))
		strain = pipeparts.mkaudiorate(pipeline, strain, skip_to_first = True, silent = False)
		@bottle.route("/%s/strain_add_drop.txt" % instrument)
		def strain_add(elem = strain):
			import time
			from pylal.date import XLALUTCToGPS
			t = float(XLALUTCToGPS(time.gmtime()))
			add = elem.get_property("add")
			drop = elem.get_property("drop")
			# FIXME don't hard code the sample rate
			return "%.9f %d %d" % (t, add / 16384., drop / 16384.)

		# state vector
		statevector = gst.element_factory_make("queue")
		statevector.set_property("max_size_buffers", 0)
		statevector.set_property("max_size_bytes", 0)
		statevector.set_property("max_size_time", gst.SECOND * 60 * 10) # 10 minutes of buffering
		pipeline.add(statevector)
		# FIXME:  don't hard-code channel name
		pipeparts.src_deferred_link(src, "%s:%s" % (instrument, "LLD-DQ_VECTOR"), statevector.get_pad("sink"))
		# FIXME we don't add a signal handler to the statevector audiorate, I assume it should report the same missing samples?
		statevector = pipeparts.mkaudiorate(pipeline, statevector, skip_to_first = True)
		statevector = pipeparts.mkstatevector(pipeline, statevector, required_on = state_vector_on_bits, required_off = state_vector_off_bits)
		@bottle.route("/%s/state_vector_on_off_gap.txt" % instrument)
		def state_vector_state(elem = statevector):
			import time
			from pylal.date import XLALUTCToGPS
			t = float(XLALUTCToGPS(time.gmtime()))
			on = elem.get_property("on-samples")
			off = elem.get_property("off-samples")
			gap = elem.get_property("gap-samples")
			return "%.9f %d %d %d" % (t, on, off, gap)

		# use state vector to gate strain
		src = pipeparts.mkgate(pipeline, strain, threshold = 1, control = statevector)
		# export state vector state
		src.set_property("emit-signals", True)
		# FIXME:  let the state vector messages going to stderr be
		# controled somehow
		bottle.route("/%s/current_segment.txt" % instrument)(get_gate_state(src, msg = instrument, verbose = True).text)
	
	else:
		raise ValueError("invalid data_source: %s" % data_source)

	# seek some non-live sources FIXME someday this should go away and seeks
	# should only be done on the pipeline that is why this is separated
	# here
	if data_source in ("LIGO", "AdvLIGO", "AdvVirgo", "frames"):
		#
		# seek the data source if not live
		#

		if src.set_state(gst.STATE_READY) != gst.STATE_CHANGE_SUCCESS:
			raise RuntimeError("Element %s did not want to enter ready state" % src.get_name())
		if not src.send_event(seekevent):
			raise RuntimeError("Element %s did not handle seek event" % src.get_name())

	#
	# provide an audioconvert element to allow Virgo data (which is
	# single-precision) to be adapted into the pipeline
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


def mkLLOIDsrc(pipeline, src, rates, instrument, psd = None, psd_fft_length = 8, ht_gate_threshold = None, veto_segments = None, seekevent = None, nxydump_segment = None, track_psd = False, block_duration = None, zero_pad = 0):
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

	quality = 9
	head = pipeparts.mkcapsfilter(pipeline, src, "audio/x-raw-float, rate=[%d,MAX]" % max(rates))
	head = pipeparts.mkcapsfilter(pipeline, pipeparts.mkresample(pipeline, head, quality = quality), "audio/x-raw-float, rate=%d" % max(rates))
	head = pipeparts.mknofakedisconts(pipeline, head)	# FIXME:  remove when resampler is patched
	head = pipeparts.mkchecktimestamps(pipeline, head, "%s_timestamps_%d_hoft" % (instrument, max(rates)))

	#
	# add a reblock element.  to reduce disk I/O gstlal_inspiral asks
	# framesrc to provide enormous buffers, and it helps reduce the RAM
	# pressure of the pipeline by slicing them up.  also, the
	# whitener's gap support isn't 100% yet and giving it smaller input
	# buffers works around the remaining weaknesses (namely that when
	# it sees a gap buffer large enough to drain its internal history,
	# it doesn't know enough to produce a short non-gap buffer to drain
	# its history followed by a gap buffer, it just produces one huge
	# non-gap buffer that's mostly zeros).
	#

	head = pipeparts.mkreblock(pipeline, head, block_duration = block_duration)

	#
	# construct whitener.
	#

	head = pipeparts.mkwhiten(pipeline, head, fft_length = psd_fft_length, zero_pad = zero_pad, average_samples = 64, median_samples = 7, expand_gaps = True, name = "lal_whiten_%s" % instrument)
	# export PSD in ascii text format
	# FIXME:  also make them available in XML format as a single document
	@bottle.route("/%s/psd.txt" % instrument)
	def get_psd_txt(elem = head):
		delta_f = elem.get_property("delta-f")
		yield "# frequency\tspectral density\n"
		for i, value in enumerate(elem.get_property("mean-psd")):
			yield "%.16g %.16g\n" % (i * delta_f, value)
	if psd is None:
		# use running average PSD
		head.set_property("psd-mode", 0)
	else:
		# use running psd
		if track_psd:
			head.set_property("psd-mode", 0)
		# use fixed PSD
		else:
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
			psd = reference_psd.interpolate_psd(psd, delta_f)
			elem.set_property("mean-psd", psd.data[:n])

		head.connect_after("notify::f-nyquist", psd_resolution_changed, psd)
		head.connect_after("notify::delta-f", psd_resolution_changed, psd)
	head = pipeparts.mkchecktimestamps(pipeline, head, "%s_timestamps_%d_whitehoft" % (instrument, max(rates)))

	#
	# optionally add vetoes
	#

	if veto_segments is not None:
		head = mksegmentsrcgate(pipeline, head, veto_segments, threshold=0.1, seekevent=seekevent, invert_output=True)

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

	# FIXME this for loop was reworked to allow the h(t) gate to go after
	# audioresamplers.  There is apparently a cornercase in the
	# audioresample element that is causing a problem
	for rate in sorted(set(rates)):
		if rate < max(rates): # downsample
			head[rate] = pipeparts.mkaudioamplify(pipeline, head[max(rates)], 1/math.sqrt(pipeparts.audioresample_variance_gain(quality, max(rates), rate)))
			head[rate] = pipeparts.mkcapsfilter(pipeline, pipeparts.mkresample(pipeline, head[rate], quality = quality), caps = "audio/x-raw-float, rate=%d" % rate)
			head[rate] = pipeparts.mknofakedisconts(pipeline, head[rate])	# FIXME:  remove when resampler is patched
			head[rate] = pipeparts.mkchecktimestamps(pipeline, head[rate], "%s_timestamps_%d_whitehoft" % (instrument, rate))
	
		#
		# optional gate on whitened h(t) amplitude
		#

		if ht_gate_threshold is not None:
			head[rate] = mkhtgate(pipeline, head[rate], threshold = ht_gate_threshold, hold_length = -rate // 4, attack_length = -rate //4)
			# export ht gate state
			head[rate].set_property("emit-signals", True)
			# FIXME:  let the state messages going to stderr be
			# controled somehow
			bottle.route("/%s/%d/current_ht_gate_segment.txt" % (instrument, rate))(get_gate_state(head[rate], msg = "%s_%d_ht_gate" % (instrument, rate), verbose = True).text)

		head[rate] = pipeparts.mktee(pipeline, head[rate])

	#
	# done.  return value is a dictionary of tee elements indexed by
	# sample rate
	#

	#for rate, elem in head.items():
	#	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, elem), "src_%d.dump" % rate)
	return head


#
# one instrument, one template bank
#


def mkLLOIDbranch(pipeline, src, bank, bank_fragment, (control_snk, control_src), gate_attack_length, gate_hold_length, nxydump_segment = None, fir_stride = None, control_peak_time = None, block_duration = None):
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

	if control_snk is not None:
		src = pipeparts.mktee(pipeline, src)	# comment-out if the tee above is uncommented
		elem = pipeparts.mkqueue(pipeline, pipeparts.mksumsquares(pipeline, src, weights = bank_fragment.sum_of_squares_weights), max_size_buffers = 0, max_size_bytes = 0, max_size_time = block_duration)
		elem = pipeparts.mkchecktimestamps(pipeline, elem, "timestamps_%s_after_sumsquare" % logname)
		# FIXME:  the capsfilter shouldn't be needed, the adder
		# should intersect it's downstream peer's format with the
		# sink format
		elem = pipeparts.mkcapsfilter(pipeline, pipeparts.mkresample(pipeline, elem, quality = 9), "audio/x-raw-float, rate=%d" % max(bank.get_rates()))
		elem = pipeparts.mknofakedisconts(pipeline, elem)	# FIXME:  remove when resampler is patched
		elem = pipeparts.mkchecktimestamps(pipeline, elem, "timestamps_%s_after_sumsquare_resampler" % logname)
		elem.link(control_snk)

	#
	# use sum-of-squares aggregate as gate control for orthogonal SNRs
	#
	# FIXME This queue has to be large for the peak finder on the control
	# signal if that element gets smarter maybe this could be made smaller
	# It should be > 1 * control_peak_time * gst.SECOND + 4 * block_duration
	#

	# if control_peak_time is None, set it to 0
	if control_peak_time is None:
		control_peak_time = 0

	src = pipeparts.mkgate(pipeline, pipeparts.mkqueue(pipeline, src, max_size_buffers = 0, max_size_bytes = 0, max_size_time = (1 * control_peak_time * gst.SECOND + 10 * block_duration)), threshold = bank.gate_threshold, attack_length = gate_attack_length, hold_length = gate_hold_length, control = pipeparts.mkqueue(pipeline, control_src, max_size_buffers = 0, max_size_bytes = 0, max_size_time = (1 * control_peak_time * gst.SECOND + 10 * block_duration)))
	src = pipeparts.mkchecktimestamps(pipeline, src, "timestamps_%s_after_gate" % logname)

	#
	# buffer orthogonal SNRs
	#
	# FIXME:  teach the collectpads object not to wait for buffers on pads
	# whose segments have not yet been reached by the input on the other
	# pads.  then this large queue buffer will not be required because
	# streaming can begin through the downstream adders without waiting for
	# input from all upstream elements.

	src = pipeparts.mkqueue(pipeline, src, max_size_buffers = 0, max_size_bytes = 0, max_size_time = 2 * block_duration)

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


def mkLLOIDhoftToSnrSlices(pipeline, hoftdict, bank, control_snksrc, verbose = False, logname = "", nxydump_segment = None, fir_stride = None, control_peak_time = None, block_duration = None, snrslices = None):
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
		branch_heads[bank_fragment.rate].add(mkLLOIDbranch(
			pipeline,
			# FIXME:  the size isn't ideal:  the correct value
			# depends on how much data is accumulated in the
			# firbank element, and the value here is only
			# approximate and not tied to the fir bank
			# parameters so might not work if those change
			pipeparts.mkqueue(pipeline, pipeparts.mkdrop(pipeline, hoftdict[bank_fragment.rate], int(round((bank.filter_length - bank_fragment.end) * bank_fragment.rate))), max_size_bytes = 0, max_size_buffers = 0, max_size_time = (1 * fir_stride + int(math.ceil(bank.filter_length))) * gst.SECOND),
			bank,
			bank_fragment,
			control_snksrc,
			32 + 2 * int(math.ceil(-autocorrelation_latency * (float(bank_fragment.rate) / output_rate))),#32 is for 1/2 the audioresample filter with qual=4 FIXME tune these windows
			32 + 2 * int(math.ceil(-autocorrelation_latency * (float(bank_fragment.rate) / output_rate))),#32 is for 1/2 the audioresample filter with qual=4
			nxydump_segment = nxydump_segment,
			fir_stride = fir_stride,
			control_peak_time = control_peak_time,
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
	#return pipeparts.mkcapsfilter(pipeline, pipeparts.mktogglecomplex(pipeline, pipeparts.mkcapsfilter(pipeline, snr, "audio/x-raw-float, rate=%d" % output_rate)), "audio/x-raw-complex, rate=%d" % output_rate)


def mkLLOIDSnrSlicesToTimeSliceChisq(pipeline, branch_heads, bank, block_duration):
	"""Build pipeline fragment that computes the TimeSliceChisq from SnrSlices."""
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

	chisq = gst.element_factory_make("lal_timeslicechisq")
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


def mkLLOIDSnrToAutoChisq(pipeline, snr, bank):
	"""Build pipeline fragment that computes the AutoChisq from single detector SNR."""
	#
	# parameters
	#

	autocorrelation_length = bank.autocorrelation_bank.shape[1]
	autocorrelation_latency = -(autocorrelation_length - 1) / 2

	# FIXME something like this could be tried.
	#mask_matrix = numpy.ones(bank.autocorrelation_bank.shape, numpy.int)
	#stix = autocorrelation_latency - 10
	#mask_matrix[:,stix:] = 0

	#
	# \chi^{2}
	#

	chisq = pipeparts.mkautochisq(pipeline, snr, autocorrelation_matrix = bank.autocorrelation_bank, mask_matrix = None, latency = autocorrelation_latency, snr_thresh = bank.snr_threshold)
	chisq = pipeparts.mkchecktimestamps(pipeline, chisq, "timestamps_%s_after_chisq" % bank.logname)

	#chisq = pipeparts.mktee(pipeline, chisq)
	#pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, chisq), "chisq_%s.dump" % logname, segment = nxydump_segment)

	return chisq


def mkLLOIDSnrChisqToTriggers(pipeline, snr, chisq, bank, verbose = False, nxydump_segment = None, logname = ""):
	"""Build pipeline fragment that converts single detector SNR and Chisq into triggers."""
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


def mkLLOIDmulti(pipeline, seekevent, detectors, banks, psd, psd_fft_length = 8, data_source = None, injection_filename = None, ht_gate_threshold = None, veto_segments = None, verbose = False, nxydump_segment = None, frame_segments = None, chisq_type = 'autochisq', track_psd = False, fir_stride = 16, control_peak_time = 16, block_duration = gst.SECOND, state_vector_on_off_dict = {"H1" : (0x7, 0x160), "L1" : (0x7, 0x160), "V1" : (0x67, 0x100)}):
	#
	# check for unrecognized chisq_types, non-unique bank IDs
	#

	if chisq_type not in ['autochisq', 'timeslicechisq']:
		raise ValueError("chisq_type must be either 'autochisq' or 'timeslicechisq', given %s" % chisq_type)
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
		src = mkLLOIDbasicsrc(pipeline, seekevent, instrument, detectors[instrument], data_source = data_source, injection_filename = injection_filename, frame_segments = frame_segments[instrument], state_vector_on_off_dict = state_vector_on_off_dict, verbose = verbose)
		# let the frame reader and injection code run in a
		# different thread than the whitener, etc.,
		src = pipeparts.mkqueue(pipeline, src, max_size_bytes = 0, max_size_buffers = 0, max_size_time = block_duration)
		if veto_segments is not None:
			hoftdicts[instrument] = mkLLOIDsrc(pipeline, src, rates, instrument, psd = psd[instrument], psd_fft_length = psd_fft_length, seekevent = seekevent, ht_gate_threshold = ht_gate_threshold, veto_segments = veto_segments[instrument], nxydump_segment = nxydump_segment, track_psd = track_psd, block_duration = block_duration)
		else:
			hoftdicts[instrument] = mkLLOIDsrc(pipeline, src, rates, instrument, psd = psd[instrument], psd_fft_length = psd_fft_length, seekevent = seekevent, ht_gate_threshold = ht_gate_threshold, nxydump_segment = nxydump_segment, track_psd = track_psd, block_duration = block_duration)

	#
	# build gate control branches
	#

	control_branch = {}
	for instrument, bank in [(instrument, bank) for instrument, banklist in banks.items() for bank in banklist]:
		suffix = "%s%s" % (instrument, (bank.logname and "_%s" % bank.logname or ""))
		if instrument != "H2":
			control_branch[(instrument, bank.bank_id)] = mkcontrolsnksrc(pipeline, max(bank.get_rates()), verbose = verbose, suffix = suffix, inj_seg_list= inj_seg_list, seekevent = seekevent, control_peak_samples = control_peak_time * max(bank.get_rates()))
			#pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, control_branch[(instrument, bank.bank_id)][1]), "control_%s.dump" % suffix, segment = nxydump_segment)

	#
	# construct trigger generators
	#

	triggersrcs = set()
	for instrument, bank in [(instrument, bank) for instrument, banklist in banks.items() for bank in banklist]:
		suffix = "%s%s" % (instrument, (bank.logname and "_%s" % bank.logname or ""))
		if instrument != "H2":
			control_snksrc = control_branch[(instrument, bank.bank_id)]
		else:
			control_snksrc = (None, control_branch[("H1", bank.bank_id)][1])
		if chisq_type == 'timeslicechisq':
			snrslices = {}
		else:
			snrslices = None
		snr = mkLLOIDhoftToSnrSlices(
			pipeline,
			hoftdicts[instrument],
			bank,
			control_snksrc,
			verbose = verbose,
			logname = suffix,
			nxydump_segment = nxydump_segment,
			control_peak_time = control_peak_time,
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
			# old way
			# snr = pipeparts.mktee(pipeline, snr)
			# chisq = mkLLOIDSnrToAutoChisq(pipeline, pipeparts.mkqueue(pipeline, snr, max_size_buffers = 0, max_size_bytes = 0, max_size_time = 1 * block_duration), bank)
		else:
			chisq = mkLLOIDSnrSlicesToTimeSliceChisq(pipeline, snrslices, bank, block_duration)
			triggersrcs.add(mkLLOIDSnrChisqToTriggers(
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
		#pipeparts.mknxydumpsink(pipeline, pipeparts.mktogglecomplex(pipeline, pipeparts.mkqueue(pipeline, snr)), "snr_%s.dump" % suffix, segment = nxydump_segment)
		#pipeparts.mkogmvideosink(pipeline, pipeparts.mkcapsfilter(pipeline, pipeparts.mkchannelgram(pipeline, pipeparts.mkqueue(pipeline, snr), plot_width = .125), "video/x-raw-rgb, width=640, height=480, framerate=64/1"), "snr_channelgram_%s.ogv" % suffix, audiosrc = pipeparts.mkaudioamplify(pipeline, pipeparts.mkqueue(pipeline, hoftdict[max(bank.get_rates())], max_size_time = 2 * int(math.ceil(bank.filter_length)) * gst.SECOND), 0.125), verbose = True)

	#
	# if there is more than one trigger source, synchronize the streams
	# with a multiqueue then use an n-to-1 adapter to combine into a
	# single stream
	#


	assert len(triggersrcs) > 0
	return triggersrcs

#
# SPIIR many instruments, many template banks
#


def mkSPIIRmulti(pipeline, seekevent, detectors, banks, psd, psd_fft_length = 8, data_source = None, injection_filename = None, ht_gate_threshold = None, veto_segments = None, verbose = False, nxydump_segment = None, frame_segments = None, chisq_type = 'autochisq', track_psd = False, block_duration = gst.SECOND, state_vector_on_off_dict = {"H1" : (0x7, 0x160), "L1" : (0x7, 0x160), "V1" : (0x67, 0x100)}):
	#
	# check for recognized value of chisq_type
	#

	if chisq_type not in ['autochisq']:
		raise ValueError("chisq_type must be either 'autochisq', given %s" % chisq_type)

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
		print instrument
		rates = set(rate for bank in banks[instrument] for rate in bank.get_rates()) # FIXME what happens if the rates are not the same?
		src = mkLLOIDbasicsrc(pipeline, seekevent, instrument, detectors[instrument], data_source = data_source, injection_filename = injection_filename, frame_segments = frame_segments[instrument], state_vector_on_off_dict = state_vector_on_off_dict, verbose = verbose)
		# let the frame reader and injection code run in a
		# different thread than the whitener, etc.,
		src = pipeparts.mkqueue(pipeline, src, max_size_bytes = 0, max_size_buffers = 0, max_size_time = block_duration)
		if veto_segments is not None:
			hoftdicts[instrument] = mkLLOIDsrc(pipeline, src, rates, instrument, psd = psd[instrument], psd_fft_length = psd_fft_length, seekevent = seekevent, ht_gate_threshold = ht_gate_threshold, veto_segments = veto_segments[instrument], nxydump_segment = nxydump_segment, track_psd = track_psd, block_duration = block_duration)
		else:
			hoftdicts[instrument] = mkLLOIDsrc(pipeline, src, rates, instrument, psd = psd[instrument], psd_fft_length = psd_fft_length, seekevent = seekevent, ht_gate_threshold = ht_gate_threshold, nxydump_segment = nxydump_segment, track_psd = track_psd, block_duration = block_duration)

	#
	# construct trigger generators
	#

	triggersrcs = set()
	for instrument, bank in [(instrument, bank) for instrument, banklist in banks.items() for bank in banklist]:
		suffix = "%s%s" % (instrument, (bank.logname and "_%s" % bank.logname or ""))

		snr = mkSPIIRhoftToSnrSlices(
			pipeline,
			hoftdicts[instrument],
			bank,
			instrument,
			verbose = verbose,
			nxydump_segment = nxydump_segment,
			quality = 4,
		)
		snr = pipeparts.mkchecktimestamps(pipeline, snr, "timestamps_%s_snr" % suffix)

		snr = pipeparts.mktogglecomplex(pipeline, snr)
		snr = pipeparts.mktee(pipeline, snr)
		# FIXME you get a different trigger generator depending on the chisq calculation :/
		if chisq_type == 'autochisq':
			# FIXME don't hardcode
			# peak finding window (n) in samples is one second at max rate, ie max(rates)
			head = pipeparts.mkitac(pipeline, snr, max(rates), bank.template_bank_filename, autocorrelation_matrix = bank.autocorrelation_bank, snr_thresh = bank.snr_threshold, sigmasq = bank.sigmasq)
			if verbose:
				head = pipeparts.mkprogressreport(pipeline, head, "progress_xml_%s" % suffix)
			triggersrcs.add(head)
			# old way
			# chisq = mkLLOIDSnrToAutoChisq(pipeline, pipeparts.mkqueue(pipeline, snr, max_size_buffers = 0, max_size_bytes = 0, max_size_time = 1 * block_duration), bank)
		# FIXME:  find a way to use less memory without this hack
		del bank.autocorrelation_bank
		#pipeparts.mknxydumpsink(pipeline, pipeparts.mktogglecomplex(pipeline, pipeparts.mkqueue(pipeline, snr)), "snr_%s.dump" % suffix, segment = nxydump_segment)
		#pipeparts.mkogmvideosink(pipeline, pipeparts.mkcapsfilter(pipeline, pipeparts.mkchannelgram(pipeline, pipeparts.mkqueue(pipeline, snr), plot_width = .125), "video/x-raw-rgb, width=640, height=480, framerate=64/1"), "snr_channelgram_%s.ogv" % suffix, audiosrc = pipeparts.mkaudioamplify(pipeline, pipeparts.mkqueue(pipeline, hoftdict[max(bank.get_rates())], max_size_time = 2 * int(math.ceil(bank.filter_length)) * gst.SECOND), 0.125), verbose = True)
## 		triggersrc.add(mkLLOIDSnrChisqToTriggers(
## 			pipeline,
## 			pipeparts.mkqueue(pipeline, snr, max_size_bytes = 0, max_size_buffers = 0, max_size_time = 1 * block_duration),
## 			chisq,
## 			bank,
## 			verbose = verbose,
## 			nxydump_segment = nxydump_segment,
## 			logname = suffix
## 		))

	#
	# if there is more than one trigger source, synchronize the streams
	# with a multiqueue then use an n-to-1 adapter to combine into a
	# single stream
	#

	assert len(triggersrcs) > 0
## 	if len(triggersrc) > 1:
## 		# send all streams through a multiqueue
## 		queue = gst.element_factory_make("multiqueue")
## 		pipeline.add(queue)
## 		for head in triggersrc:
## 			head.link(queue)
## 		triggersrc = queue
## 		# FIXME:  it has been reported that the input selector
## 		# breaks seeks.  confirm and fix if needed
## 		# FIXME:  input-selector in 0.10.32 no longer has the
## 		# "select-all" feature.  need to get this re-instated
## 		#nto1 = gst.element_factory_make("input-selector")
## 		#nto1.set_property("select-all", True)
## 		#pipeline.add(nto1)
## 		#for pad in queue.src_pads():
## 		#	pad.link(nto1)
## 		#triggersrc = nto1
## 	else:
## 		# len(triggersrc) == 1
## 		triggersrc, = triggersrc

	#
	# done
	#

	return triggersrcs


def mkSPIIRhoftToSnrSlices(pipeline, src, bank, instrument, verbose = None, nxydump_segment = None, quality = 4, sample_rates = None, max_rate = None):
	if sample_rates is None:
		sample_rates = sorted(bank.get_rates())
	else:
		sample_rates = sorted(sample_rates)
	#FIXME don't upsample everything to a common rate
	if max_rate is None:
		max_rate = max(sample_rates)
	prehead = None

	for sr in sample_rates:
		head = pipeparts.mkqueue(pipeline, src[sr], max_size_time=gst.SECOND * 10, max_size_buffers=0, max_size_bytes=0)
		head = pipeparts.mkreblock(pipeline, head)
		head = pipeparts.mkiirbank(pipeline, head, a1 = bank.A[sr], b0 = bank.B[sr], delay = bank.D[sr], name = "gstlaliirbank_%d_%s_%s" % (sr, instrument, bank.logname))
		head = pipeparts.mkqueue(pipeline, head, max_size_time=gst.SECOND * 10, max_size_buffers=0, max_size_bytes=0)
		if prehead is not None:
			adder = gst.element_factory_make("lal_adder")
			adder.set_property("sync", True)
			pipeline.add(adder)
			head.link(adder)
			prehead.link(adder)
			head = adder
		# FIXME:  this should get a nofakedisconts after it until the resampler is patched
		head = pipeparts.mkresample(pipeline, head, quality = quality)
		if sr == max_rate:
			head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw-float, rate=%d" % max_rate)
		else:
			head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw-float, rate=%d" % (2 * sr))
		prehead = head

	return head

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


# The following snippet is taken from http://gstreamer.freedesktop.org/wiki/FAQ#Mypygstprogramismysteriouslycoredumping.2Chowtofixthis.3F
import pygtk
pygtk.require("2.0")
import gobject
gobject.threads_init()
import pygst
pygst.require('0.10')
import gst


from glue import iterutils
from glue import segments
from glue.ligolw import lsctables
from pylal import ligolw_thinca
from pylal.xlal.datatypes.ligotimegps import LIGOTimeGPS


from gstlal import pipeparts
from gstlal import cbc_template_fir
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
			raise RuntimeError, "Element %s did not want to enter ready state" % segsrc.get_name()
		if not segsrc.send_event(seekevent):
			raise RuntimeError, "Element %s did not handle seek event" % segsrc.get_name()
	return pipeparts.mkgate(pipeline, src, threshold = threshold, control = pipeparts.mkqueue(pipeline, segsrc))

#
# gate controlled by h(t)
#


def mkhtgate(pipeline, src, threshold = 1.0, attack_length = 1024, hold_length = 1024, invert_control = True, low_frequency=40, high_frequency=1000):
	src = pipeparts.mkqueue(pipeline, src)
	t = pipeparts.mktee(pipeline, src)
	q1 = pipeparts.mkqueue(pipeline, t)
	ss = pipeparts.mkaudiochebband(pipeline, q1, low_frequency, high_frequency)
	q2 = pipeparts.mkqueue(pipeline, t)
	return pipeparts.mkgate(pipeline, q2, threshold = threshold, control = ss, attack_length = attack_length, hold_length = hold_length, invert_control = invert_control)


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


def mkcontrolsnksrc(pipeline, rate, verbose = False, suffix = None, inj_seg_list = None, seekevent = None, control_peak_time = None, block_duration = None):
	#
	# start with an adder and caps filter to select a sample rate
	#

	snk = gst.element_factory_make("lal_adder")
	snk.set_property("sync", True)
	pipeline.add(snk)
	src = pipeparts.mkcapsfilter(pipeline, snk, "audio/x-raw-float, rate=%d" % rate)

	#
	# Add a peak finder on the control signal sample number = 3 seconds at 2048 Hz
	# FIXME don't assume 2048 Hz
	#
	
	if control_peak_time is not None:
		src = pipeparts.mkreblock(pipeline, pipeparts.mkpeak(pipeline, src, 2048 * control_peak_time), block_duration = block_duration)
	
	src = pipeparts.mkqueue(pipeline, src, max_size_buffers = 0, max_size_bytes = 0, max_size_time = block_duration)
	
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


def mkLLOIDbasicsrc(pipeline, seekevent, instrument, detector, fake_data = False, online_data = False, injection_filename = None, frame_segments = None, verbose = False):
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
		src = pipeparts.mkframesrc(pipeline, location = detector.frame_cache, instrument = instrument, channel_name = detector.channel, blocksize = detector.block_size, segment_list = frame_segments)

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


def mkLLOIDsrc(pipeline, src, rates, instrument, psd = None, psd_fft_length = 8, ht_gate_threshold = None, veto_segments = None, seekevent = None, nxydump_segment = None, track_psd = False, block_duration = None):
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
	# might ever do it therefore no nofakediscont elements are required
	# here
	#

	quality = 9
	head = pipeparts.mkcapsfilter(pipeline, src, "audio/x-raw-float, rate=[%d,MAX]" % max(rates))
	head = pipeparts.mkresample(pipeline, head, quality = quality)
	head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw-float, rate=%d" % max(rates))

	#
	# add a reblock element.  to reduce disk I/O gstlal_inspiral asks
	# framesrc to provide enormous buffers, and it helps reduce the RAM
	# pressure of the pipeline by slicing them up.  also, the
	# whitener's gap support isn't 100% yet and giving it smaller input
	# buffers works around the remaining weaknesses (namely that when
	# it sees a gap buffer large enough to drain its internal history,
	# it doesn't know enough to produce a short non-gap buffer to drain
	# its history followed by a gap buffer, it just produces one huge
	# non-gap buffer).
	#

	head = pipeparts.mkreblock(pipeline, head, block_duration = block_duration)

	#
	# construct whitener.  this element must be followed by a
	# nofakedisconts element.
	#

	head = pipeparts.mkwhiten(pipeline, head, fft_length = psd_fft_length, zero_pad = 0, average_samples = 64, median_samples = 7)
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
			psd = cbc_template_fir.interpolate_psd(psd, delta_f)
			elem.set_property("mean-psd", psd.data[:n])

		head.connect_after("notify::f-nyquist", psd_resolution_changed, psd)
		head.connect_after("notify::delta-f", psd_resolution_changed, psd)
	head = pipeparts.mknofakedisconts(pipeline, head, silent = True)
	head = pipeparts.mkchecktimestamps(pipeline, head, "%s_timestamps_%d_whitehoft" % (instrument, max(rates)))

	#
	# optional gate on h(t) amplitude
	#

	if ht_gate_threshold is not None:
		head = mkhtgate(pipeline, head, ht_gate_threshold, high_frequency = max(rates) * 0.45)

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

	for rate in sorted(set(rates))[:-1]:	# all but the highest rate
		head[rate] = pipeparts.mkaudioamplify(pipeline, head[max(rates)], 1/math.sqrt(pipeparts.audioresample_variance_gain(quality, max(rates), rate)))
		head[rate] = pipeparts.mkcapsfilter(pipeline, pipeparts.mkresample(pipeline, head[rate], quality = quality), caps = "audio/x-raw-float, rate=%d" % rate)
		head[rate] = pipeparts.mknofakedisconts(pipeline, head[rate], silent = True)
		head[rate] = pipeparts.mkchecktimestamps(pipeline, head[rate], "%s_timestamps_%d_whitehoft" % (instrument, rate))
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
	logname = "%s_%d_%d" % (bank.logname, bank_fragment.start, bank_fragment.end)

	#
	# FIR filter bank.  low frequency branches use time-domain
	# convolution, high-frequency branches use FFT convolution with a
	# block stride of 4 s.
	#
	# FIXME:  why the -1?  without it the pieces don't match but I
	# don't understand where this offset comes from.  it might really
	# need to be here, or it might be a symptom of a bug elsewhere.
	# figure this out.

	src = pipeparts.mkfirbank(pipeline, src, latency = -int(round(bank_fragment.start * bank_fragment.rate)) - 1, fir_matrix = bank_fragment.orthogonal_template_bank, block_stride = fir_stride * bank_fragment.rate, time_domain = max(bank.get_rates()) / bank_fragment.rate >= 32)
	src = pipeparts.mkchecktimestamps(pipeline, src, "timestamps_%s_after_firbank" % logname)
	src = pipeparts.mkreblock(pipeline, src, block_duration = block_duration)
	src = pipeparts.mktee(pipeline, src)
	#pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, src), "orthosnr_%s.dump" % logname, segment = nxydump_segment)

	#pipeparts.mkvideosink(pipeline, pipeparts.mkcapsfilter(pipeline, pipeparts.mkhistogram(pipeline, src), "video/x-raw-rgb, width=640, height=480, framerate=1/4"))
	#pipeparts.mkogmvideosink(pipeline, pipeparts.mkcapsfilter(pipeline, pipeparts.mkchannelgram(pipeline, pipeparts.mkqueue(pipeline, src), plot_width = .125), "video/x-raw-rgb, width=640, height=480, framerate=64/1"), "orthosnr_channelgram_%s.ogv" % logname, verbose = True)

	#
	# compute weighted sum-of-squares, feed to sum-of-squares
	# aggregator
	#

	elem = pipeparts.mkresample(pipeline, pipeparts.mkqueue(pipeline, pipeparts.mksumsquares(pipeline, src, weights = bank_fragment.sum_of_squares_weights),max_size_buffers = 0, max_size_bytes = 0, max_size_time = block_duration), quality = 9)
	elem = pipeparts.mknofakedisconts(pipeline, elem, silent = True)
	elem = pipeparts.mkchecktimestamps(pipeline, elem, "timestamps_%s_after_sumsquare_resampler" % logname)
	elem.link(control_snk)

	#
	# use sum-of-squares aggregate as gate control for orthogonal SNRs
	#
	# FIXME This queue has to be large for the peak finder on the control
	# signal if that element gets smarter maybe this could be made smaller
	# It should be > 3 * control_peak_time * gst.SECOND + 4 * block_duration
	#

	# if control_peak_time is None, set it to 0
	if control_peak_time is None:
		control_peak_time = 0

	src = pipeparts.mkgate(pipeline, pipeparts.mkqueue(pipeline, src, max_size_buffers = 0, max_size_bytes = 0, max_size_time = (3 * control_peak_time * gst.SECOND + 5 * block_duration)), threshold = bank.gate_threshold, attack_length = gate_attack_length, hold_length = gate_hold_length, control = pipeparts.mkqueue(pipeline, control_src, max_size_buffers = 0, max_size_bytes = 0, max_size_time = 2 * block_duration))
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


def mkLLOIDhoftToSnrSlices(pipeline, hoftdict, bank, control_snksrc, verbose = False, logname = "", nxydump_segment = None, fir_stride = None, control_peak_time = None, block_duration = None):
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
			int(math.ceil(-autocorrelation_latency * (float(bank_fragment.rate) / output_rate))),
			int(math.ceil(-autocorrelation_latency * (float(bank_fragment.rate) / output_rate))),
			nxydump_segment = nxydump_segment,
			fir_stride = fir_stride,
			control_peak_time = control_peak_time,
			block_duration = block_duration
		))

	#
	# sum the snr streams of the same sample rate
	#

	output_heads = {}
	for rate, heads in sorted(branch_heads.items()):
		#
		# include an adder when more than one stream at a given sample rate
		#

		if len(heads) > 1:
			#
			# hook all matrix mixers that share a common sample rate to
			# an adder.  the adder replaces the set of matrix mixers as
			# the new "head" associated with the sample rate
			#

			output_head = gst.element_factory_make("lal_adder")
			output_head.set_property("sync", True)
			pipeline.add(output_head)
			for head in heads:
				pipeparts.mkqueue(pipeline, head, max_size_bytes = 0, max_size_buffers = 0, max_size_time = block_duration).link(output_head)
		else:
			output_head = list(heads)[0]

		#
		# resample this to the highest sample rate
		#

		# FIXME quality = 1 seems to be okay and we could save some flops potentially...
		output_head = pipeparts.mkresample(pipeline, output_head, quality = 4)
		output_head = pipeparts.mkcapsfilter(pipeline, output_head, "audio/x-raw-float, rate=%d" % output_rate)
		output_head = pipeparts.mktogglecomplex(pipeline, output_head)
		output_heads[rate] = output_head

	return output_heads


def mkLLOIDSnrSlicesToSnr(pipeline, branch_heads, block_duration):
	"""Build the pipeline fragment to compute the single detector SNR from SnrSlices associated with different saple rates."""
	#
	# if more than one SnrSlice, add them together
	#

	if len(branch_heads) > 1:
		snr = gst.element_factory_make("lal_adder")
		snr.set_property("sync", True)
		pipeline.add(snr)
		for rate, head in branch_heads.items():
			pipeparts.mkqueue(pipeline, head, max_size_bytes = 0, max_size_buffers = 0, max_size_time = block_duration).link(snr)
	else:
		snr = branch_heads.values()[0]

	return snr


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


def mkLLOIDSnrToAutoChisq(pipeline, snr, bank, block_duration):
	"""Build pipeline fragment that computes the AutoChisq from single detector SNR."""
	#
	# parameters
	#

	autocorrelation_length = bank.autocorrelation_bank.shape[1]
	autocorrelation_latency = -(autocorrelation_length - 1) / 2

	#
	# \chi^{2}
	#

	chisq = pipeparts.mkautochisq(pipeline, pipeparts.mkqueue(pipeline, snr, max_size_bytes = 0, max_size_buffers = 0, max_size_time = block_duration), autocorrelation_matrix = bank.autocorrelation_bank, latency = autocorrelation_latency, snr_thresh = bank.snr_threshold)

	#chisq = pipeparts.mktee(pipeline, chisq)
	#pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, chisq), "chisq_%s.dump" % logname, segment = nxydump_segment)

	return chisq


def mkLLOIDSnrChisqToTriggers(pipeline, snr, chisq, bank, verbose = False, nxydump_segment = None, logname = "", block_duration = None):
	"""Build pipeline fragment that converts single detector SNR and Chisq into triggers."""
	#
	# trigger generator and progress report
	#

	head = pipeparts.mktriggergen(pipeline, pipeparts.mkqueue(pipeline, snr, max_size_bytes = 0, max_size_buffers = 0, max_size_time = block_duration), pipeparts.mkqueue(pipeline, chisq, max_size_bytes = 0, max_size_buffers = 0, max_size_time = block_duration), template_bank_filename = bank.template_bank_filename, snr_threshold = bank.snr_threshold, sigmasq = bank.sigmasq)
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


def mkLLOIDmulti(pipeline, seekevent, detectors, banks, psd, psd_fft_length = 8, fake_data = False, online_data = False, injection_filename = None, ht_gate_threshold = None, veto_segments = None, verbose = False, nxydump_segment = None, frame_segments = None, chisq_type = 'autochisq', track_psd = False, fir_stride = 10, control_peak_time = 10, block_duration = gst.SECOND):
	#
	# check for recognized value of chisq_type
	#

	if chisq_type not in ['autochisq', 'timeslicechisq']:
		raise ValueError, "chisq_type must be either 'autochisq' or 'timeslicechisq', given %s" % (chisq_type)


	#
	# extract segments from the injection file for selected reconstruction
	#

	if injection_filename is not None:
		inj_seg_list = simulation.sim_inspiral_to_segment_list(injection_filename)
	else:
		inj_seg_list = None

	#
	# loop over instruments and template banks
	#

	triggersrc = set()
	for instrument in detectors:
		rates = set(rate for bank in banks[instrument] for rate in bank.get_rates()) # FIXME what happens if the rates are not the same?
		src = mkLLOIDbasicsrc(pipeline, seekevent, instrument, detectors[instrument], fake_data = fake_data, online_data = online_data, injection_filename = injection_filename, frame_segments = frame_segments[instrument], verbose = verbose)
		# let the frame reader and injection code run in a
		# different thread than the whitener, etc.,
		src = pipeparts.mkqueue(pipeline, src, max_size_bytes = 0, max_size_buffers = 0, max_size_time = block_duration)
		if veto_segments is not None:
			hoftdict = mkLLOIDsrc(pipeline, src, rates, instrument, psd = psd[instrument], psd_fft_length = psd_fft_length, seekevent = seekevent, ht_gate_threshold = ht_gate_threshold, veto_segments = veto_segments[instrument], nxydump_segment = nxydump_segment, track_psd = track_psd, block_duration = block_duration)
		else:
			hoftdict = mkLLOIDsrc(pipeline, src, rates, instrument, psd = psd[instrument], psd_fft_length = psd_fft_length, seekevent = seekevent, ht_gate_threshold = ht_gate_threshold, nxydump_segment = nxydump_segment, track_psd = track_psd)
		for bank in banks[instrument]:
			suffix = "%s%s" % (instrument, (bank.logname and "_%s" % bank.logname or ""))
			control_snksrc = mkcontrolsnksrc(pipeline, max(bank.get_rates()), verbose = verbose, suffix = suffix, inj_seg_list= inj_seg_list, seekevent = seekevent, control_peak_time = control_peak_time, block_duration = block_duration)
			#pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, control_snksrc[1]), "control_%s.dump" % suffix, segment = nxydump_segment)
			snrslices = mkLLOIDhoftToSnrSlices(
				pipeline,
				hoftdict,
				bank,
				control_snksrc,
				verbose = verbose,
				logname = suffix,
				nxydump_segment = nxydump_segment,
				control_peak_time = control_peak_time,
				fir_stride = fir_stride,
				block_duration = block_duration
			)
			if chisq_type == 'timeslicechisq':
				for rate, snrslice in snrslices.items():
					snrslices[rate] = pipeparts.mktee(pipeline, snrslice)
			snr = mkLLOIDSnrSlicesToSnr(
				pipeline,
				snrslices,
				block_duration
			)
			snr = pipeparts.mkchecktimestamps(pipeline, snr, "timestamps_%s_snr" % suffix)
			if chisq_type == 'autochisq':
				snr = pipeparts.mktee(pipeline, snr)
				chisq = mkLLOIDSnrToAutoChisq(pipeline, snr, bank, block_duration)
			else:
				chisq = mkLLOIDSnrSlicesToTimeSliceChisq(pipeline, snrslices, bank, block_duration)
			# FIXME:  find a way to use less memory without this hack
			del bank.autocorrelation_bank
			#pipeparts.mknxydumpsink(pipeline, pipeparts.mktogglecomplex(pipeline, pipeparts.mkqueue(pipeline, snr)), "snr_%s.dump" % suffix, segment = nxydump_segment)
			#pipeparts.mkogmvideosink(pipeline, pipeparts.mkcapsfilter(pipeline, pipeparts.mkchannelgram(pipeline, pipeparts.mkqueue(pipeline, snr), plot_width = .125), "video/x-raw-rgb, width=640, height=480, framerate=64/1"), "snr_channelgram_%s.ogv" % suffix, audiosrc = pipeparts.mkaudioamplify(pipeline, pipeparts.mkqueue(pipeline, hoftdict[max(rates)], max_size_time = 2 * int(math.ceil(bank.filter_length)) * gst.SECOND), 0.125), verbose = True)
			triggersrc.add(mkLLOIDSnrChisqToTriggers(
				pipeline,
				snr,
				chisq,
				bank,
				verbose = verbose,
				nxydump_segment = nxydump_segment,
				logname = suffix,
				block_duration = block_duration
			))

	#
	# if there is more than one trigger source, synchronize the streams
	# with a multiqueue then use an n-to-1 adapter to combine into a
	# single stream
	#

	assert len(triggersrc) > 0
	if len(triggersrc) > 1:
		# send all streams through a multiqueue
		queue = gst.element_factory_make("multiqueue")
		pipeline.add(queue)
		for head in triggersrc:
			head.link(queue)
		triggersrc = queue
		# FIXME:  it has been reported that the input selector
		# breaks seeks.  confirm and fix if needed
		# FIXME:  input-selector in 0.10.32 no longer has the
		# "select-all" feature.  need to get this re-instated
		#nto1 = gst.element_factory_make("input-selector")
		#nto1.set_property("select-all", True)
		#pipeline.add(nto1)
		#for pad in queue.src_pads():
		#	pad.link(nto1)
		#triggersrc = nto1
	else:
		# len(triggersrc) == 1
		triggersrc, = triggersrc

	#
	# done
	#

	return triggersrc


#
# on-the-fly thinca implementation
#


class StreamThinca(object):
	def __init__(self, dataobj, e_thinca_parameter, coincidence_back_off, thinca_size = 100.0):
		self.dataobj = dataobj
		self.process_table = lsctables.New(lsctables.ProcessTable)
		self.process_params_table = lsctables.New(lsctables.ProcessParamsTable)
		self.sngl_inspiral_table = lsctables.New(lsctables.SnglInspiralTable, dataobj.sngl_inspiral_table.columnnames)
		self.coinc_event_map_table = lsctables.New(lsctables.CoincMapTable)
		self.last_boundary = -segments.infinity()
		self.e_thinca_parameter = e_thinca_parameter
		self.coincidence_back_off = coincidence_back_off
		self.thinca_size = thinca_size
		self.ids = set()

	def run_coincidence(self, boundary):
		# wait until we've accumulated thinca_size seconds
		if self.last_boundary + self.thinca_size > boundary:
			return

		# remove triggers that are too old to be useful
		discard_boundary = self.last_boundary - self.coincidence_back_off
		iterutils.inplace_filter(lambda row: row.get_end() >= discard_boundary, self.sngl_inspiral_table)

		# replace tables with our versions
		self.dataobj.xmldoc.childNodes[-1].replaceChild(self.process_table, self.dataobj.process_table)
		self.dataobj.xmldoc.childNodes[-1].replaceChild(self.process_params_table, self.dataobj.process_params_table)
		self.dataobj.xmldoc.childNodes[-1].replaceChild(self.coinc_event_map_table, self.dataobj.coinc_event_map_table)

		# find coincs.  gstlal_inspiral's triggers cause a
		# divide-by-zero error in the effective SNR function used
		# for lalapps_inspiral triggers, so we replace it with one
		# that works for the duration of the ligolw_thinca() call.
		def ntuple_comparefunc(events, offset_vector, seg = segments.segment(self.last_boundary, boundary)):
			return ligolw_thinca.coinc_inspiral_end_time(events, offset_vector) not in seg
		def get_effective_snr(self, fac):
			return self.snr / self.chisq
		orig_get_effective_snr, ligolw_thinca.SnglInspiral.get_effective_snr = ligolw_thinca.SnglInspiral.get_effective_snr, get_effective_snr
		ligolw_thinca.ligolw_thinca(
			self.dataobj.xmldoc,
			process_id = self.dataobj.process.process_id,
			EventListType = ligolw_thinca.InspiralEventList,
			CoincTables = ligolw_thinca.InspiralCoincTables,
			coinc_definer_row = ligolw_thinca.InspiralCoincDef,
			event_comparefunc = ligolw_thinca.inspiral_coinc_compare_exact,
			thresholds = self.e_thinca_parameter,
			ntuple_comparefunc = ntuple_comparefunc
		)
		ligolw_thinca.SnglInspiral.get_effective_snr = orig_get_effective_snr

		# put the original table objects back
		self.dataobj.xmldoc.childNodes[-1].replaceChild(self.dataobj.process_table, self.process_table)
		self.dataobj.xmldoc.childNodes[-1].replaceChild(self.dataobj.process_params_table, self.process_params_table)
		self.dataobj.xmldoc.childNodes[-1].replaceChild(self.dataobj.coinc_event_map_table, self.coinc_event_map_table)
		del self.process_table[:]
		del self.process_params_table[:]

		# record boundary
		self.last_boundary = boundary

	def appsink_new_buffer(self, elem, dataobj):
		# make sure we've been passed the correct object
		assert dataobj is self.dataobj

		# replace the sngl_inspiral table with our version.  in
		# addition to replacing the table object in the xml tree,
		# we also need to replace the attribute in the dataobj
		# because that's what appsink_new_buffer() will write to
		self.dataobj.xmldoc.childNodes[-1].replaceChild(self.sngl_inspiral_table, self.dataobj.sngl_inspiral_table)
		orig_sngl_inspiral_table, self.dataobj.sngl_inspiral_table = self.dataobj.sngl_inspiral_table, self.sngl_inspiral_table

		# chain to normal function in pipeparts.  after this, the
		# new triggers will have been appended to our
		# sngl_inspiral_table
		prev_len = len(self.sngl_inspiral_table)
		pipeparts.appsink_new_buffer(elem, self.dataobj)

		# convert the new row objects to the type required by
		# ligolw_thinca()
		for i in range(prev_len, len(self.sngl_inspiral_table)):
			old = self.sngl_inspiral_table[i]
			self.sngl_inspiral_table[i] = new = ligolw_thinca.SnglInspiral()
			for col in self.sngl_inspiral_table.columnnames:
				setattr(new, col, getattr(old, col))

		# coincidence
		if self.sngl_inspiral_table:
			# since the triggers are appended to the table, we
			# can rely on the last one to provide an estimate
			# of the most recent time stamps to come out of the
			# pipeline
			self.run_coincidence(self.sngl_inspiral_table[-1].get_end() - self.coincidence_back_off)

		# put the original sngl_inspiral table back
		self.dataobj.sngl_inspiral_table = orig_sngl_inspiral_table
		self.dataobj.xmldoc.childNodes[-1].replaceChild(self.dataobj.sngl_inspiral_table, self.sngl_inspiral_table)

		# copy triggers into real output document
		if self.coinc_event_map_table:
			newids = set(self.coinc_event_map_table.getColumnByName("event_id")) - self.ids
			self.ids |= newids
			while self.coinc_event_map_table:
				self.dataobj.coinc_event_map_table.append(self.coinc_event_map_table.pop())
			for row in self.sngl_inspiral_table:
				if row.event_id in newids:
					self.dataobj.sngl_inspiral_table.append(row)

	def flush(self):
		# replace the sngl_inspiral table with our version
		self.dataobj.xmldoc.childNodes[-1].replaceChild(self.sngl_inspiral_table, self.dataobj.sngl_inspiral_table)

		# coincidence
		self.run_coincidence(segments.infinity())

		# put the original sngl_inspiral table back
		self.dataobj.xmldoc.childNodes[-1].replaceChild(self.dataobj.sngl_inspiral_table, self.sngl_inspiral_table)

		# copy triggers into real output document
		newids = set(self.coinc_event_map_table.getColumnByName("event_id")) - self.ids
		self.ids |= newids
		while self.coinc_event_map_table:
			self.dataobj.coinc_event_map_table.append(self.coinc_event_map_table.pop())
		while self.sngl_inspiral_table:
			row = self.sngl_inspiral_table.pop()
			if row.event_id in newids:
				self.dataobj.sngl_inspiral_table.append(row)

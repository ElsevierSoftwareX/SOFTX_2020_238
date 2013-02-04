# Copyright (C) 2009--2011  LIGO Scientific Collaboration
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


import sys, os
import numpy
import threading

# The following snippet is taken from http://gstreamer.freedesktop.org/wiki/FAQ#Mypygstprogramismysteriouslycoredumping.2Chowtofixthis.3F
import pygtk
pygtk.require("2.0")
import gobject
gobject.threads_init()
import pygst
pygst.require('0.10')
import gst


from glue import segments
from gstlal import pipeio


__author__ = "Kipp Cannon <kipp.cannon@ligo.org>, Chad Hanna <chad.hanna@ligo.org>, Drew Keppel <drew.keppel@ligo.org>"
__version__ = "FIXME"
__date__ = "FIXME"


#
# =============================================================================
#
#                             Generic Constructors
#
# =============================================================================
#


#
# Applications should use the element-specific wrappings that follow below.
# The generic constructors are only intended to simplify the writing of
# those wrappings, they are not meant to be how applications create
# elements in pipelines.
#


def mkgeneric(pipeline, src, elem_type_name, **properties):
	if "name" in properties:
		elem = gst.element_factory_make(elem_type_name, properties.pop("name"))
	else:
		elem = gst.element_factory_make(elem_type_name)
	for name, value in properties.items():
		elem.set_property(name.replace("_", "-"), value)
	pipeline.add(elem)
	if isinstance(src, gst.Pad):
		src.get_parent_element().link_pads(src, elem, None)
	elif src is not None:
		src.link(elem)
	return elem


#
# deferred link helper
#


class src_deferred_link(object):
	def __init__(self, src, srcpadname, sinkpad):
		no_more_pads_handler_id = src.connect("no-more-pads", self.no_more_pads, srcpadname)
		src.connect("pad-added", self.pad_added, (srcpadname, sinkpad, no_more_pads_handler_id))

	@staticmethod
	def pad_added(element, pad, (srcpadname, sinkpad, no_more_pads_handler_id)):
		if pad.get_name() == srcpadname:
			pad.get_parent().handler_disconnect(no_more_pads_handler_id)
			pad.link(sinkpad)

	@staticmethod
	def no_more_pads(element, srcpadname):
		raise ValueError("<%s>: no pad named '%s'" % (element.get_name(), srcpadname))


#
# =============================================================================
#
#                                Pipeline Parts
#
# =============================================================================
#


def mkchannelgram(pipeline, src, **properties):
	return mkgeneric(pipeline, src, "lal_channelgram", **properties)


def mkspectrumplot(pipeline, src, **properties):
	return mkgeneric(pipeline, src, "lal_spectrumplot", **properties)


def mkhistogram(pipeline, src):
	return mkgeneric(pipeline, src, "lal_histogramplot")


def mksegmentsrc(pipeline, segment_list, blocksize = 4096 * 1 * 1, invert_output = False):
	# default blocksize is 4096 seconds of unsigned integers at
	# 1 Hz, e.g. segments without nanoseconds
	return mkgeneric(pipeline, None, "lal_segmentsrc", blocksize = blocksize, segment_list = segments.segmentlist(segments.segment(a.ns(), b.ns()) for a, b in segment_list), invert_output = invert_output)


def mklalcachesrc(pipeline, location, **properties):
	return mkgeneric(pipeline, None, "lal_cachesrc", location = location, **properties)


def mkframesrc(pipeline, location, instrument, channel_name, blocksize = 16384 * 8 * 1, cache_src_regex = None, cache_dsc_regex = None, segment_list = None):
	# default blocksize is 1 second of double precision floats at
	# 16384 Hz, e.g., LIGO h(t)
	properties = {}
	if segment_list is not None:
		properties["segment_list"] = segments.segmentlist(segments.segment(a.ns(), b.ns()) for a, b in segment_list)
	if cache_src_regex is not None:
		properties["cache_src_regex"] = cache_src_regex
	if cache_dsc_regex is not None:
		properties["cache_dsc_regex"] = cache_dsc_regex
	return mkgeneric(pipeline, None, "lal_framesrc", blocksize = blocksize, location = location, instrument = instrument, channel_name = channel_name, **properties)


def mklvshmsrc(pipeline, **properties):
	return mkgeneric(pipeline, None, "gds_lvshmsrc", **properties)


def mkigwdparse(pipeline, src, **properties):
	return mkgeneric(pipeline, src, "framecpp_igwdparse", **properties)


def mkframecppchanneldemux(pipeline, src, **properties):
	return mkgeneric(pipeline, src, "framecpp_channeldemux", **properties)


def mkframecppchannelmux(pipeline, src, **properties):
	return mkgeneric(pipeline, src, "framecpp_channelmux", **properties)


def mkmultifilesink(pipeline, src, next_file = 0, **properties):
	return mkgeneric(pipeline, src, "multifilesink", next_file = next_file, sync = False, async = False, **properties)


def mkframesink(pipeline, src, **properties):
	return mkgeneric(pipeline, src, "lal_framesink", sync = False, async = False, **properties)


def mkndssrc(pipeline, host, instrument, channel_name, channel_type, blocksize = 16384 * 8 * 1, port = 31200):
	# default blocksize is 1 second of double precision floats at
	# 16384 Hz, e.g., LIGO h(t)
	return mkgeneric(pipeline, None, "ndssrc", blocksize = blocksize, port = port, host = host, channel_name = "%s:%s" % (instrument, channel_name), channel_type = channel_type)


def mkonlinehoftsrc(pipeline, instrument):
	# This function lacks the "channel_name" argument because with the
	# online h(t) source "onlinehoftsrc" knows the channel names that are needed
	# for each instrument.
	#
	# It also lacks the "blocksize" argument because the blocksize for an
	# "onlinehoftsrc" is not adjustable.
	elem = gst.element_factory_make("lal_onlinehoftsrc")
	elem.set_property("instrument", instrument)
	pipeline.add(elem)
	return elem


def mkcapsfilter(pipeline, src, caps):
	return mkgeneric(pipeline, src, "capsfilter", caps = gst.Caps(caps))


def mkcapssetter(pipeline, src, caps, **properties):
	return mkgeneric(pipeline, src, "capssetter", caps = gst.Caps(caps), **properties)


def mkstatevector(pipeline, src, **properties):
	return mkgeneric(pipeline, src, "lal_statevector", **properties)


def mktaginject(pipeline, src, tags):
	return mkgeneric(pipeline, src, "taginject", tags = tags)


def mkaudiotestsrc(pipeline, **properties):
	return mkgeneric(pipeline, None, "audiotestsrc", **properties)


def mkfakesrc(pipeline, instrument, channel_name, blocksize = 16384 * 8 * 1, volume = 1e-20, is_live = False, wave = 9, rate = 16384):
	# default blocksize is 1 second of double precision floats at
	# 16384 Hz, e.g., h(t)
	return mktaginject(pipeline, mkcapsfilter(pipeline, mkaudiotestsrc(pipeline, samplesperbuffer = blocksize / 8, wave = wave, volume = volume, is_live = is_live), "audio/x-raw-float, width=64, rate=%d" % rate), "instrument=%s,channel-name=%s,units=strain" % (instrument, channel_name))


def mkfirfilter(pipeline, src, kernel, latency, **properties):
	properties.update((name, val) for name, val in (("kernel", kernel), ("latency", latency)) if val is not None)
	return mkgeneric(pipeline, src, "audiofirfilter", **properties)


def mkiirfilter(pipeline, src, a, b):
	# convention is z = \exp(-i 2 \pi f / f_{\rm sampling})
	# H(z) = (\sum_{j=0}^{N} a_j z^{-j}) / (\sum_{j=0}^{N} (-1)^{j} b_j z^{-j})
	return mkgeneric(pipeline, src, "audioiirfilter", a = a, b = b)


def mkshift(pipeline, src, **properties):
	return mkgeneric(pipeline, src, "lal_shift", **properties)


def mkfakeLIGOsrc(pipeline, location = None, instrument = None, channel_name = None, blocksize = 16384 * 8 * 1):
	properties = {"blocksize": blocksize}
	if instrument is not None:
		properties["instrument"] = instrument
	if channel_name is not None:
		properties["channel_name"] = channel_name
	return mkgeneric(pipeline, None, "lal_fakeligosrc", **properties)


def mkfakeadvLIGOsrc(pipeline, location = None, instrument = None, channel_name = None, blocksize = 16384 * 8 * 1):
	properties = {"blocksize": blocksize}
	if instrument is not None:
		properties["instrument"] = instrument
	if channel_name is not None:
		properties["channel_name"] = channel_name
	return mkgeneric(pipeline, None, "lal_fakeadvligosrc", **properties)


def mkfakeadvvirgosrc(pipeline, location = None, instrument = None, channel_name = None, blocksize = 16384 * 8 * 1):
	properties = {"blocksize": blocksize}
	if instrument is not None:
		properties["instrument"] = instrument
	if channel_name is not None:
		properties["channel_name"] = channel_name
	return mkgeneric(pipeline, None, "lal_fakeadvvirgosrc", **properties)


def mkprogressreport(pipeline, src, name):
	return mkgeneric(pipeline, src, "progressreport", do_query = False, name = name)


def mkinjections(pipeline, src, filename):
	return mkgeneric(pipeline, src, "lal_simulation", xml_location = filename)


def mkaudiochebband(pipeline, src, lower_frequency, upper_frequency, poles = 8):
	return mkgeneric(pipeline, src, "audiochebband", lower_frequency = lower_frequency, upper_frequency = upper_frequency, poles = poles)


def mkaudiocheblimit(pipeline, src, cutoff, mode = 0, poles = 8):
	return mkgeneric(pipeline, src, "audiocheblimit", cutoff = cutoff, mode = mode, poles = poles)


def mkaudioamplify(pipeline, src, amplification):
	return mkgeneric(pipeline, src, "audioamplify", clipping_method = 3, amplification = amplification)


def mkaudioundersample(pipeline, src):
	return mkgeneric(pipeline, src, "lal_audioundersample")


def mkresample(pipeline, src, **properties):
	return mkgeneric(pipeline, src, "audioresample", **properties)


def mkwhiten(pipeline, src, psd_mode = 0, zero_pad = 0, fft_length = 8, average_samples = 64, median_samples = 7, **properties):
	return mkgeneric(pipeline, src, "lal_whiten", psd_mode = psd_mode, zero_pad = zero_pad, fft_length = fft_length, average_samples = average_samples, median_samples = median_samples, **properties)


def mktee(pipeline, src):
	return mkgeneric(pipeline, src, "tee")


def mkqueue(pipeline, src, **properties):
	return mkgeneric(pipeline, src, "queue", **properties)


def mkdrop(pipeline, src, drop_samples = 0):
	return mkgeneric(pipeline, src, "lal_drop", drop_samples = drop_samples)


def mkdelay(pipeline, src, delay = 0):
	return mkgeneric(pipeline, src, "lal_delay", delay = delay)


def mknofakedisconts(pipeline, src, silent = True):
	return mkgeneric(pipeline, src, "lal_nofakedisconts", silent = silent)


def mkfirbank(pipeline, src, latency = None, fir_matrix = None, time_domain = None, block_stride = None):
	properties = dict((name, value) for name, value in zip(("latency", "fir_matrix", "time_domain", "block_stride"), (latency, fir_matrix, time_domain, block_stride)) if value is not None)
	return mkgeneric(pipeline, src, "lal_firbank", **properties)


def mkiirbank(pipeline, src, a1, b0, delay, name=None):
	properties = {}
	if name is not None:
		properties["name"] = name
	if a1 is not None:
		properties["a1_matrix"] = pipeio.repack_complex_array_to_real(a1)
	if b0 is not None:
		properties["b0_matrix"] = pipeio.repack_complex_array_to_real(b0)
	if delay is not None:
		properties["delay_matrix"] = delay
	elem = mkgeneric(pipeline, src, "lal_iirbank", **properties)
	elem = mknofakedisconts(pipeline, elem)	# FIXME:  remove after basetransform behaviour fixed
	return elem


def mktrim(pipeline, src, initial_offset = None, final_offset = None, inverse = None):
	properties = dict((name, value) for name, value in zip(("initial-offset", "final-offset", "inverse"), (initial_offset,final_offset,inverse)) if value is not None)
	return mkgeneric(pipeline, src, "lal_trim", **properties)


def mkmean(pipeline, src, **properties):
	return mkgeneric(pipeline, src, "lal_mean", **properties)


def mkabs(pipeline, src, **properties):
	return mkgeneric(pipeline, src, "abs", **properties)


def mkpow(pipeline, src, **properties):
	return mkgeneric(pipeline, src, "pow", **properties)


def mkreblock(pipeline, src, **properties):
	return mkgeneric(pipeline, src, "lal_reblock", **properties)


def mksumsquares(pipeline, src, weights = None):
	if weights is not None:
		return mkgeneric(pipeline, src, "lal_sumsquares", weights = weights)
	else:
		return mkgeneric(pipeline, src, "lal_sumsquares")


def mkgate(pipeline, src, threshold = None, control = None, **properties):
	if threshold is not None:
		elem = mkgeneric(pipeline, None, "lal_gate", threshold = threshold, **properties)
	else:
		elem = mkgeneric(pipeline, None, "lal_gate", **properties)
	for peer, padname in ((src, "sink"), (control, "control")):
		if isinstance(peer, gst.Pad):
			peer.get_parent_element().link_pads(peer, elem, padname)
		elif peer is not None:
			peer.link_pads(None, elem, padname)
	return elem


def mkbitvectorgen(pipeline, src, bit_vector, **properties):
	return mkgeneric(pipeline, src, "lal_bitvectorgen", bit_vector = bit_vector, **properties)


def mkmatrixmixer(pipeline, src, matrix = None):
	if matrix is not None:
		return mkgeneric(pipeline, src, "lal_matrixmixer", matrix = matrix)
	else:
		return mkgeneric(pipeline, src, "lal_matrixmixer")


def mktogglecomplex(pipeline, src):
	return mkgeneric(pipeline, src, "lal_togglecomplex")


def mkautochisq(pipeline, src, autocorrelation_matrix = None, mask_matrix = None, latency = 0, snr_thresh=0):
	properties = {}
	if autocorrelation_matrix is not None:
		properties.update({
			"autocorrelation_matrix": pipeio.repack_complex_array_to_real(autocorrelation_matrix),
			"latency": latency,
			"snr_thresh": snr_thresh
		})
	if mask_matrix is not None:
		properties["autocorrelation_mask_matrix"] = mask_matrix
	return mkgeneric(pipeline, src, "lal_autochisq", **properties)


def mkfakesink(pipeline, src):
	return mkgeneric(pipeline, src, "fakesink", sync = False, async = False)


def mkfilesink(pipeline, src, filename):
	return mkgeneric(pipeline, src, "filesink", sync = False, async = False, buffer_mode = 2, location = filename)


def mknxydumpsink(pipeline, src, filename, segment = None):
	if segment is not None:
		elem = mkgeneric(pipeline, src, "lal_nxydump", start_time = segment[0].ns(), stop_time = segment[1].ns())
	else:
		elem = mkgeneric(pipeline, src, "lal_nxydump")
	return mkfilesink(pipeline, elem, filename)


def mknxydumpsinktee(pipeline, src, *args, **properties):
	t = mktee(pipeline, src)
	mknxydumpsink(pipeline, mkqueue(pipeline, t), *args, **properties)
	return t


def mkblcbctriggergen(pipeline, snr, chisq, template_bank_filename, snr_threshold, sigmasq):
	# snr is complex and chisq is real so the correct source and sink
	# pads will be selected automatically
	elem = mkgeneric(pipeline, snr, "lal_blcbctriggergen", bank_filename = template_bank_filename, snr_thresh = snr_threshold, sigmasq = sigmasq)
	chisq.link(elem)
	return elem


def mktriggergen(pipeline, snr, chisq, template_bank_filename, snr_threshold, sigmasq):
	# snr is complex and chisq is real so the correct source and sink
	# pads will be selected automatically
	elem = mkgeneric(pipeline, snr, "lal_triggergen", bank_filename = template_bank_filename, snr_thresh = snr_threshold, sigmasq = sigmasq)
	chisq.link(elem)
	return elem


def mktriggerxmlwritersink(pipeline, src, filename):
	return mkgeneric(pipeline, src, "lal_triggerxmlwriter", sync = False, async = False, location = filename)


def mkwavenc(pipeline, src):
	return mkgeneric(pipeline, src, "wavenc")


def mkcolorspace(pipeline, src):
	return mkgeneric(pipeline, src, "ffmpegcolorspace")


def mktheoraenc(pipeline, src, **properties):
	return mkgeneric(pipeline, src, "theoraenc", **properties)


def mkmpeg4enc(pipeline, src, **properties):
	return mkgeneric(pipeline, src, "ffenc_mpeg4", **properties)


def mkoggmux(pipeline, src):
	return mkgeneric(pipeline, src, "oggmux")


def mkavimux(pipeline, src):
	return mkgeneric(pipeline, src, "avimux")


def mkaudioconvert(pipeline, src, caps_string = None):
	elem = mkgeneric(pipeline, src, "audioconvert")
	if caps_string is not None:
		elem = mkcapsfilter(pipeline, elem, caps_string)
	return elem


def mkaudiorate(pipeline, src, **properties):
	return mkgeneric(pipeline, src, "audiorate", **properties)


def mkflacenc(pipeline, src, quality = 0, **properties):
	return mkgeneric(pipeline, src, "flacenc", quality = quality, **properties)


def mkogmvideosink(pipeline, videosrc, filename, audiosrc = None, verbose = False):
	src = mkcolorspace(pipeline, videosrc)
	src = mkcapsfilter(pipeline, src, "video/x-raw-yuv, format=(fourcc)I420")
	src = mktheoraenc(pipeline, src, border = 2, quality = 48, quick = False)
	src = mkoggmux(pipeline, src)
	if audiosrc is not None:
		mkflacenc(pipeline, mkcapsfilter(pipeline, mkaudioconvert(pipeline, audiosrc), "audio/x-raw-int, width=32, depth=24")).link(src)
	if verbose:
		src = mkprogressreport(pipeline, src, filename)
	mkfilesink(pipeline, src, filename)


def mkvideosink(pipeline, src):
	return mkgeneric(pipeline, mkcolorspace(pipeline, src), "autovideosink")


def mkplaybacksink(pipeline, src, amplification = 0.1):
	elems = (
		gst.element_factory_make("audioconvert"),
		gst.element_factory_make("capsfilter"),
		gst.element_factory_make("audioamplify"),
		gst.element_factory_make("audioconvert"),
		gst.element_factory_make("queue"),
		gst.element_factory_make("autoaudiosink")
	)
	elems[1].set_property("caps", gst.Caps("audio/x-raw-float, width=64"))
	elems[2].set_property("amplification", amplification)
	elems[4].set_property("max-size-time", 1 * gst.SECOND)
	pipeline.add(*elems)
	gst.element_link_many(src, *elems)


def mkappsink(pipeline, src, max_buffers = 1, drop = False, **properties):
	return mkgeneric(pipeline, src, "appsink", sync = False, async = False, emit_signals = True, max_buffers = max_buffers, drop = drop, **properties)


class AppSync(object):
	def __init__(self, appsink_new_buffer, appsinks = []):
		self.lock = threading.Lock()
		self.appsink_new_buffer = appsink_new_buffer
		self.appsinks = {}
		self.at_eos = set()
		for elem in appsinks:
			if elem in self.appsinks:
				raise ValueError("duplicate appsinks")
			elem.connect("new-buffer", self.appsink_handler, False)
			elem.connect("eos", self.appsink_handler, True)
			self.appsinks[elem] = None

	def add_sink(self, pipeline, src, drop = False, **properties):
		# NOTE that max buffers must be 1 for this to work
		assert "max_buffers" not in properties
		elem = mkappsink(pipeline, src, max_buffers = 1, drop = drop, **properties)
		elem.connect("new-buffer", self.appsink_handler, False)
		elem.connect("eos", self.appsink_handler, True)
		self.appsinks[elem] = None
		return elem

	def appsink_handler(self, elem, eos):
		self.lock.acquire()
		try:
			# update eos status, and retrieve buffer timestamp
			if eos:
				self.at_eos.add(elem)
			else:
				self.at_eos.discard(elem)
				assert self.appsinks[elem] is None
				self.appsinks[elem] = elem.get_last_buffer().timestamp

			# keep looping while we can process buffers
			while True:
				# retrieve the timestamps of all elements that
				# aren't at eos and all elements at eos that still
				# have buffers in them
				timestamps = [(t, e) for e, t in self.appsinks.items() if e not in self.at_eos or t is not None]
				# nothing to do if all elements are at eos and do
				# not have buffers
				if not timestamps:
					break
				# find the element with the oldest timestamp.  None
				# compares as less than everything, so we'll find
				# any element (that isn't at eos) that doesn't yet
				# have a buffer (elements at eos and that are
				# without buffers aren't in the list)
				timestamp, elem_with_oldest = min(timestamps)
				# if there's an element without a buffer, do
				# nothing --- we require all non-eos elements to
				# have buffers before proceding
				if timestamp is None:
					break
				# pass element to handler func and clear timestamp
				self.appsink_new_buffer(elem_with_oldest)
				self.appsinks[elem_with_oldest] = None
		finally:
			self.lock.release()


def connect_appsink_dump_dot(pipeline, appsinks, basename, verbose = False):

	"""
	add a signal handler to write a pipeline graph upon receipt of the
	first trigger buffer.  the caps in the pipeline graph are not fully
	negotiated until data comes out the end, so this version of the graph
	shows the final formats on all links
	"""

	class AppsinkDumpDot(object):
		# data shared by all instances
		# number of times execute method has been invoked, and a mutex
		n_lock = threading.Lock()
		n = 0

		def __init__(self, pipeline, write_after, basename, verbose = False):
			self.pipeline = pipeline
			self.handler_id = None
			self.write_after = write_after
			self.filestem = "%s.%s" % (basename, "TRIGGERS")
			self.verbose = verbose

		def execute(self, elem):
			self.n_lock.acquire()
			try:
				type(self).n += 1
				if self.n >= self.write_after:
					write_dump_dot(self.pipeline, self.filestem, verbose = self.verbose)
			finally:
				self.n_lock.release()
			elem.disconnect(self.handler_id)

	for sink in appsinks:
		appsink_dump_dot = AppsinkDumpDot(pipeline, len(appsinks), basename = basename, verbose = verbose)
		appsink_dump_dot.handler_id = sink.connect_after("new-buffer", appsink_dump_dot.execute)


def mkchecktimestamps(pipeline, src, name = None, silent = True, timestamp_fuzz = 1):
	return mkgeneric(pipeline, src, "lal_checktimestamps", name = name, silent = silent, timestamp_fuzz = timestamp_fuzz)


def mkpeak(pipeline, src, n):
	return mkgeneric(pipeline, src, "lal_peak", n = n)


def mkitac(pipeline, src, n, bank, autocorrelation_matrix = None, snr_thresh = 0, sigmasq = None):
	properties = {
		"n": n,
		"bank_filename": bank,
		"snr_thresh": snr_thresh
	}
	if autocorrelation_matrix is not None:
		properties["autocorrelation_matrix"] = pipeio.repack_complex_array_to_real(autocorrelation_matrix)
	if sigmasq is not None:
		properties["sigmasq"] = sigmasq
	return mkgeneric(pipeline, src, "lal_itac", **properties)


def mklhocoherentnull(pipeline, H1src, H2src, H1_impulse, H1_latency, H2_impulse, H2_latency, srate):
	elem = mkgeneric(pipeline, None, "lal_lho_coherent_null", block_stride = srate, H1_impulse = H1_impulse, H2_impulse = H2_impulse, H1_latency = H1_latency, H2_latency = H2_latency)
	for peer, padname in ((H1src, "H1sink"), (H2src, "H2sink")):
		if isinstance(peer, gst.Pad):
			peer.get_parent_element().link_pads(peer, elem, padname)
		elif peer is not None:
			peer.link_pads(None, elem, padname)
	return elem


def mkbursttriggergen(pipeline, src, n, bank):
	return mkgeneric(pipeline, src, "lal_bursttriggergen", n = n, bank_filename = bank)

def mkodctodqv(pipeline, src, **properties):
	return mkgeneric(pipeline, src, "lal_odc_to_dqv", **properties)

def mktcpserversink(pipeline, src, **properties):
	# units_soft_max = 1 GB
	# FIXME:  are these sensible defaults?
	return mkgeneric(pipeline, src, "tcpserversink", sync = True, sync_method = "latest-keyframe", recover_policy = "keyframe", unit_type = "bytes", units_soft_max = 1024**3, **properties)


def audioresample_variance_gain(quality, num, den):
	"""Calculate the output gain of GStreamer's stock audioresample element.

	The audioresample element has a frequency response of unity "almost" all the
	way up the Nyquist frequency.  However, for an input of unit variance
	Gaussian noise, the output will have a variance very slighly less than 1.
	The return value is the variance that the filter will produce for a given
	"quality" setting and sample rate.

	@param den The denomenator of the ratio of the input and output sample rates
	@param num The numerator of the ratio of the input and output sample rates
	@return The variance of the output signal for unit variance input

	The following example shows how to apply the correction factor using an
	audioamplify element.

	>>> from gstlal.pipeutil import *
	>>> from gstlal.pipeparts import audioresample_variance_gain
	>>> import gstlal.pipeio as pipeio
	>>> import numpy
	>>> nsamples = 2 ** 17
	>>> num = 2
	>>> den = 1
	>>> def handoff_handler(element, buffer, pad, (quality, filt_len, num, den)):
	...		out_latency = numpy.ceil(float(den) / num * filt_len)
	...		buf = pipeio.array_from_audio_buffer(buffer).flatten()
	...		std = numpy.std(buf[out_latency:-out_latency])
	...		print "quality=%2d, filt_len=%3d, num=%d, den=%d, stdev=%.2f" % (
	...			quality, filt_len, num, den, std)
	...
	>>> for quality in range(11):
	...		pipeline = gst.Pipeline()
	...		correction = 1/numpy.sqrt(audioresample_variance_gain(quality, num, den))
	...		elems = mkelems_in_bin(pipeline,
	...			('audiotestsrc', {'wave':'gaussian-noise','volume':1}),
	...			('capsfilter', {'caps':gst.Caps('audio/x-raw-float,width=64,rate=%d' % num)}),
	...			('audioresample', {'quality':quality}),
	...			('capsfilter', {'caps':gst.Caps('audio/x-raw-float,width=64,rate=%d' % den)}),
	...			('audioamplify', {'amplification':correction,'clipping-method':'none'}),
	...			('fakesink', {'signal-handoffs':True, 'num-buffers':1})
	...		)
	...		filt_len = elems[2].get_property('filter-length')
	...		elems[0].set_property('samplesperbuffer', 2 * filt_len + nsamples)
	...		if elems[-1].connect_after('handoff', handoff_handler, (quality, filt_len, num, den)) < 1:
	...			raise RuntimeError
	...		try:
	...			if pipeline.set_state(gst.STATE_PLAYING) is not gst.STATE_CHANGE_ASYNC:
	...				raise RuntimeError
	...			if not pipeline.get_bus().poll(gst.MESSAGE_EOS, -1):
	...				raise RuntimeError
	...		finally:
	...			if pipeline.set_state(gst.STATE_NULL) is not gst.STATE_CHANGE_SUCCESS:
	...				raise RuntimeError
	...
	quality= 0, filt_len=  8, num=2, den=1, stdev=1.00
	quality= 1, filt_len= 16, num=2, den=1, stdev=1.00
	quality= 2, filt_len= 32, num=2, den=1, stdev=1.00
	quality= 3, filt_len= 48, num=2, den=1, stdev=1.00
	quality= 4, filt_len= 64, num=2, den=1, stdev=1.00
	quality= 5, filt_len= 80, num=2, den=1, stdev=1.00
	quality= 6, filt_len= 96, num=2, den=1, stdev=1.00
	quality= 7, filt_len=128, num=2, den=1, stdev=1.00
	quality= 8, filt_len=160, num=2, den=1, stdev=1.00
	quality= 9, filt_len=192, num=2, den=1, stdev=1.00
	quality=10, filt_len=256, num=2, den=1, stdev=1.00
	"""

	# These constants were measured with 2**22 samples.

	if num > den: # downsampling
		return den * (
			0.7224862140943990596,
			0.7975021342935247892,
			0.8547537598970208483,
			0.8744072146753004704,
			0.9075294214410336568,
			0.9101523813406768859,
			0.9280549396020538744,
			0.9391809530012216189,
			0.9539276644089494939,
			0.9623083437067311285,
			0.9684700588501590213
			)[quality] / num
	elif num < den: # upsampling
		return (
			0.7539740617648067467,
			0.8270076656536116122,
			0.8835072979478705291,
			0.8966758456219333651,
			0.9253434087537378838,
			0.9255866674042573239,
			0.9346487800036394900,
			0.9415331868209220190,
			0.9524608799160205752,
			0.9624372769883490220,
			0.9704505626409354324
			)[quality]
	else: # no change in sample rate
		return 1.


#
# =============================================================================
#
#                                Debug utilities
#
# =============================================================================
#


def write_dump_dot(pipeline, filestem, verbose = False):
	"""
	This function needs the environment variable GST_DEBUG_DUMP_DOT_DIR
	to be set.   The filename will be

	os.path.join($GST_DEBUG_DUMP_DOT_DIR, filestem + ".dot")

	If verbose is True, a message will be written to stderr.
	"""
	if "GST_DEBUG_DUMP_DOT_DIR" not in os.environ:
		raise ValueError("cannot write pipeline, environment variable GST_DEBUG_DUMP_DOT_DIR is not set")
	gst.DEBUG_BIN_TO_DOT_FILE(pipeline, gst.DEBUG_GRAPH_SHOW_ALL, filestem)
	if verbose:
		print >>sys.stderr, "Wrote pipeline to %s" % os.path.join(os.environ["GST_DEBUG_DUMP_DOT_DIR"], "%s.dot" % filestem)

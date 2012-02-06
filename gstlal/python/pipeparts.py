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
import time

# The following snippet is taken from http://gstreamer.freedesktop.org/wiki/FAQ#Mypygstprogramismysteriouslycoredumping.2Chowtofixthis.3F
import pygtk
pygtk.require("2.0")
import gobject
gobject.threads_init()
import pygst
pygst.require('0.10')
import gst


from glue import segments

import pipeio


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
# Applications should use the element-specific wrappings below.  The
# generic constructors are only intended to simplify the writing of those
# wrappings, they are not meant to be how applications create elements in
# pipelines.
#


def mkgeneric(pipeline, src, elem_type_name, **properties):
	if "name" in properties:
		elem = gst.element_factory_make(elem_type_name, properties.pop("name"))
	else:
		elem = gst.element_factory_make(elem_type_name)
	for name, value in properties.items():
		elem.set_property(name.replace("_", "-"), value)
	pipeline.add(elem)
	if src is not None:
		src.link(elem)
	return elem


#
# deferred link helper
#


def src_deferred_link(src, srcpadname, sinkpad):
	def pad_added(element, pad, (srcpadname, sinkpad)):
		if pad.get_name() == srcpadname:
			pad.link(sinkpad)
	src.connect("pad-added", pad_added, (srcpadname, sinkpad))


#
# =============================================================================
#
#                                Pipeline Parts
#
# =============================================================================
#


def mkchannelgram(pipeline, src, **properties):
	return mkgeneric(pipeline, src, "lal_channelgram", **properties)


def mkspectrumplot(pipeline, src, pad = None, **properties):
	elem = gst.element_factory_make("lal_spectrumplot")
	for name, value in properties.items():
		elem.set_property(name.replace("_", "-"), value)
	pipeline.add(elem)
	if pad is not None:
		src.link_pads(pad, elem, "sink")
	else:
		src.link(elem)
	return elem


def mkhistogram(pipeline, src):
	return mkgeneric(pipeline, src, "lal_histogramplot")


def mksegmentsrc(pipeline, segment_list, blocksize = 4096 * 1 * 1, invert_output=False):
	# default blocksize is 4096 seconds of unsigned integers at
	# 1 Hz, e.g. segments without nanoseconds
	elem = gst.element_factory_make("lal_segmentsrc")
	elem.set_property("blocksize", blocksize)
	elem.set_property("segment-list", segments.segmentlist(segments.segment(a.ns(), b.ns()) for a, b in segment_list))
	elem.set_property("invert-output", invert_output)
	pipeline.add(elem)
	return elem


def mkframesrc(pipeline, location, instrument, channel_name, blocksize = 16384 * 8 * 1, cache_src_regex = None, cache_dsc_regex = None, segment_list = None):
	# default blocksize is 1 second of double precision floats at
	# 16384 Hz, e.g., LIGO h(t)
	elem = gst.element_factory_make("lal_framesrc")
	elem.set_property("blocksize", blocksize)
	elem.set_property("location", location)
	elem.set_property("instrument", instrument)
	elem.set_property("channel-name", channel_name)
	if segment_list is not None:
		elem.set_property("segment-list", segments.segmentlist(segments.segment(a.ns(), b.ns()) for a, b in segment_list))
	if cache_src_regex is not None:
		elem.set_property("cache-src-regex", cache_src_regex)
	if cache_dsc_regex is not None:
		elem.set_property("cache-dsc-regex", cache_dsc_regex)
	pipeline.add(elem)
	return elem


def mklvshmsrc(pipeline, **properties):
	elem = gst.element_factory_make("gds_lvshmsrc")
	for name, value in properties.items():
		elem.set_property(name.replace("_", "-"), value)
	pipeline.add(elem)
	return elem


def mkframecppchanneldemux(pipeline, src, **properties):
	return mkgeneric(pipeline, src, "framecpp_channeldemux", **properties)


def mkframesink(pipeline, src, **properties):
	elem = gst.element_factory_make("lal_framesink")
	elem.set_property("sync", False)
	elem.set_property("async", False)
	for name, value in properties.items():
		elem.set_property(name.replace("_", "-"), value)
	pipeline.add(elem)
	src.link(elem)


def mkndssrc(pipeline, host, instrument, channel_name, blocksize = 16384 * 8 * 1):
	# default blocksize is 1 second of double precision floats at
	# 16384 Hz, e.g., LIGO h(t)
	elem = gst.element_factory_make("ndssrc")
	elem.set_property("blocksize", blocksize)
	elem.set_property("host", host)
	elem.set_property("channel-name", "%s:%s" % (instrument, channel_name))
	pipeline.add(elem)
	return elem


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


def mkstatevector(pipeline, src, **properties):
	return mkgeneric(pipeline, src, "lal_statevector", **properties)


def mktaginject(pipeline, src, tags):
	return mkgeneric(pipeline, src, "taginject", tags = tags)


def mkaudiotestsrc(pipeline, **properties):
	elem = gst.element_factory_make("audiotestsrc")
	for name, value in properties.items():
		elem.set_property(name.replace("_", "-"), value)
	pipeline.add(elem)
	return elem


def mkfakesrc(pipeline, instrument, channel_name, blocksize = 16384 * 8 * 1, volume = 1e-20, is_live = False, wave = 9):
	# default blocksize is 1 second of double precision floats at
	# 16384 Hz, e.g., h(t)
	return mktaginject(pipeline, mkcapsfilter(pipeline, mkaudiotestsrc(pipeline, samplesperbuffer = blocksize / 8, wave = wave, volume = volume, is_live = is_live), "audio/x-raw-float, width=64, rate=16384"), "instrument=%s,channel-name=%s,units=strain" % (instrument, channel_name))


def mkfakesrcseeked(pipeline, instrument, channel_name, seekevent, blocksize = 16384 * 8 * 1, volume = 1e-20, is_live = False, wave = 9):
	# default blocksize is 1 second of double precision floats at
	# 16384 Hz, e.g., h(t)
	src = mkaudiotestsrc(pipeline, samplesperbuffer = blocksize / 8, wave = wave, volume = volume, is_live = is_live)
	# attempt to seek the element
	if src.set_state(gst.STATE_READY) != gst.STATE_CHANGE_SUCCESS:
		raise RuntimeError, "Element %s did not want to enter ready state" % src.get_name()
	if not src.send_event(seekevent):
		raise RuntimeError, "Element %s did not handle seek event" % src.get_name()
	return mktaginject(pipeline, mkcapsfilter(pipeline, src, "audio/x-raw-float, width=64, rate=16384"), "instrument=%s,channel-name=%s,units=strain" % (instrument, channel_name))


def mkfirfilter(pipeline, src, kernel, latency, **properties):
	properties.update((name, val) for name, val in (("kernel", kernel), ("latency", latency)) if val is not None)
	return mkgeneric(pipeline, src, "audiofirfilter", **properties)


def mkiirfilter(pipeline, src, a, b):
	# convention is z = \exp(-i 2 \pi f / f_{\rm sampling})
	# H(z) = (\sum_{j=0}^{N} a_j z^{-j}) / (\sum_{j=0}^{N} (-1)^{j} b_j z^{-j})
	return mkgeneric(pipeline, src, "audioiirfilter", a = a, b = b)


def mkfakeLIGOsrc(pipeline, location=None, instrument=None, channel_name=None, blocksize=16384 * 8 * 1):
	head = gst.element_factory_make("lal_fakeligosrc")
	if instrument is not None:
		head.set_property("instrument", instrument)
	if channel_name is not None:
		head.set_property("channel-name", channel_name)
	head.set_property("blocksize", blocksize)
	pipeline.add(head)
	return head


def mkfakeadvLIGOsrc(pipeline, location=None, instrument=None, channel_name=None, blocksize=16384 * 8 * 1):
	head = gst.element_factory_make("lal_fakeadvligosrc")
	if instrument is not None:
		head.set_property("instrument", instrument)
	if channel_name is not None:
		head.set_property("channel-name", channel_name)
	head.set_property("blocksize", blocksize)
	pipeline.add(head)
	return head


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


def mkresample(pipeline, src, pad_name = None, **properties):
	elem = gst.element_factory_make("audioresample")
	for name, value in properties.items():
		elem.set_property(name.replace("_", "-"), value)
	pipeline.add(elem)
	if pad_name is None:
		src.link(elem)
	else:
		src.link_pads(pad_name, elem, "sink")
	return elem


def mkwhiten(pipeline, src, psd_mode = 0, zero_pad = 0, fft_length = 8, average_samples = 64, median_samples = 7, **kwargs):
	return mkgeneric(pipeline, src, "lal_whiten", psd_mode = psd_mode, zero_pad = zero_pad, fft_length = fft_length, average_samples = average_samples, median_samples = median_samples, **kwargs)


def mktee(pipeline, src, pad_name = None):
	elem = gst.element_factory_make("tee")
	pipeline.add(elem)
	if pad_name is None:
		src.link(elem)
	else:
		src.link_pads(pad_name, elem, "sink")
	return elem


def mkqueue(pipeline, src, pad_name = None, **properties):
	elem = gst.element_factory_make("queue")
	for name, value in properties.items():
		elem.set_property(name.replace("_", "-"), value)
	pipeline.add(elem)
	if pad_name is None:
		src.link(elem)
	else:
		src.link_pads(pad_name, elem, "sink")
	return elem


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


def mkmean(pipeline, src, **properties):
	return mkgeneric(pipeline, src, "lal_mean", **properties)


def mkreblock(pipeline, src, **properties):
	return mkgeneric(pipeline, src, "lal_reblock", **properties)


def mksumsquares(pipeline, src, weights = None):
	if weights is not None:
		return mkgeneric(pipeline, src, "lal_sumsquares", weights = weights)
	else:
		return mkgeneric(pipeline, src, "lal_sumsquares")


def mkgate(pipeline, src, threshold = None, control = None, **properties):
	elem = gst.element_factory_make("lal_gate")
	if threshold is not None:
		elem.set_property("threshold", threshold)
	pipeline.add(elem)
	src.link_pads("src", elem, "sink")
	if control is not None:
		control.link_pads("src", elem, "control")
	for name, value in properties.items():
		elem.set_property(name.replace("_", "-"), value)
	return elem


def mkmatrixmixer(pipeline, src, matrix = None):
	if matrix is not None:
		return mkgeneric(pipeline, src, "lal_matrixmixer", matrix = matrix)
	else:
		return mkgeneric(pipeline, src, "lal_matrixmixer")


def mktogglecomplex(pipeline, src):
	return mkgeneric(pipeline, src, "lal_togglecomplex")


def mkautochisq(pipeline, src, autocorrelation_matrix = None, mask_matrix = None, latency = 0, snr_thresh=0):
	elem = gst.element_factory_make("lal_autochisq")
	if autocorrelation_matrix is not None:
		elem.set_property("autocorrelation-matrix", pipeio.repack_complex_array_to_real(autocorrelation_matrix))
		elem.set_property("latency", latency)
		elem.set_property("snr-thresh", snr_thresh)
	if mask_matrix is not None:
		elem.set_property("autocorrelation-mask-matrix", mask_matrix)
	pipeline.add(elem)
	src.link(elem)
	return elem


def mkfakesink(pipeline, src, pad = None):
	elem = gst.element_factory_make("fakesink")
	elem.set_property("sync", False)
	elem.set_property("async", False)
	pipeline.add(elem)
	if pad is not None:
		src.link_pads(pad, elem, "sink")
	else:
		src.link(elem)


def mkfilesink(pipeline, src, filename):
	return mkgeneric(pipeline, src, "filesink", sync = False, async = False, buffer_mode = 2, location = filename)


def mknxydumpsink(pipeline, src, filename, segment = None):
	elem = gst.element_factory_make("lal_nxydump")
	if segment is not None:
		if type(segment[0]) is not segments.infinity:
			elem.set_property("start-time", segment[0].ns())
		if type(segment[1]) is not segments.infinity:
			elem.set_property("stop-time", segment[1].ns())
	pipeline.add(elem)
	src.link(elem)
	mkfilesink(pipeline, elem, filename)


def mknxydumpsinktee(pipeline, src, *args, **kwargs):
	t = mktee(pipeline, src)
	mknxydumpsink(pipeline, mkqueue(pipeline, t), *args, **kwargs)
	return t


def mkblcbctriggergen(pipeline, snr, chisq, template_bank_filename, snr_threshold, sigmasq):
	elem = gst.element_factory_make("lal_blcbctriggergen")
	elem.set_property("bank-filename", template_bank_filename)
	elem.set_property("snr-thresh", snr_threshold)
	elem.set_property("sigmasq", sigmasq)
	pipeline.add(elem)
	# snr is complex and chisq is real so the correct source and sink
	# pads will be selected automatically
	snr.link(elem)
	chisq.link(elem)
	return elem


def mktriggergen(pipeline, snr, chisq, template_bank_filename, snr_threshold, sigmasq):
	elem = gst.element_factory_make("lal_triggergen")
	elem.set_property("bank-filename", template_bank_filename)
	elem.set_property("snr-thresh", snr_threshold)
	elem.set_property("sigmasq", sigmasq)
	pipeline.add(elem)
	# snr is complex and chisq is real so the correct source and sink
	# pads will be selected automatically
	snr.link(elem)
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


def mkaudiorate(pipeline, src, pad_name = None, **properties):
	elem = gst.element_factory_make("audiorate")
	pipeline.add(elem)
	for name, value in properties.items():
		elem.set_property(name, value)
	if pad_name is None:
		src.link(elem)
	else:
		src.link_pads(pad_name, elem, "sink")
	return elem


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


def mkappsink(pipeline, src, pad_name = None, max_buffers = 1, drop = False, **properties):
	elem = gst.element_factory_make("appsink")
	elem.set_property("sync", False)
	elem.set_property("async", False)
	elem.set_property("emit-signals", True)
	elem.set_property("max-buffers", max_buffers)
	elem.set_property("drop", drop)
	for name, value in properties.items():
		elem.set_property(name, value)
	pipeline.add(elem)
	if pad_name is not None:
		src.link_pads(pad_name, elem, "sink")
	elif src is not None:
		src.link(elem)
	return elem


class AppSync(object):
	def __init__(self, appsink_new_buffer, appsinks = []):
		self.lock = threading.Lock()
		self.appsink_new_buffer = appsink_new_buffer
		self.appsinks = {}
		self.at_eos = set()
		for elem in appsinks:
			if elem in self.appsinks:
				raise ValueError, "duplicate appsinks"
			elem.connect("new-buffer", self.appsink_handler, False)
			elem.connect("eos", self.appsink_handler, True)
			self.appsinks[elem] = None

	def add_sink(self, pipeline, src, drop = False, **properties):
		# NOTE that max buffers must be 1 for this to work
		elem = mkappsink(pipeline, src, max_buffers = 1, drop = drop, **properties)
		elem.connect("new-buffer", self.appsink_handler, False)
		elem.connect("eos", self.appsink_handler, True)
		self.appsinks[elem] = None
		return elem

	def appsink_handler(self, elem, eos):
		self.lock.acquire()

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

		self.lock.release()


def mkchecktimestamps(pipeline, src, name = None, silent = True, timestamp_fuzz = 1):
	return mkgeneric(pipeline, src, "lal_checktimestamps", name = name, silent = silent, timestamp_fuzz = timestamp_fuzz)


def mkpeak(pipeline, src, n):
	return mkgeneric(pipeline, src, "lal_peak", n = n)


def mkitac(pipeline, src, n, bank, autocorrelation_matrix = None, snr_thresh = 0, sigmasq = None):
	elem = gst.element_factory_make("lal_itac")
	elem.set_property("n", n)
	elem.set_property("bank-filename", bank)
	if autocorrelation_matrix is not None:
		elem.set_property("autocorrelation-matrix", pipeio.repack_complex_array_to_real(autocorrelation_matrix))
	if sigmasq is not None:
		elem.set_property("sigmasq", sigmasq)
	elem.set_property("snr-thresh", snr_thresh)
	pipeline.add(elem)
	src.link(elem)
	return elem


def mkbursttriggergen(pipeline, src, n, bank):
	return mkgeneric(pipeline, src, "lal_bursttriggergen", n = n, bank_filename = bank)


def mksyncsink(pipeline, srcs):
	"""
	add streams together and dump to a fake sink.  this can be used to
	synchronize streams. It returns tee'd off versions of srcs
	"""
	adder = gst.element_factory_make("lal_adder")
	adder.set_property("sync", True)
	pipeline.add(adder)
	outsrcs = []
	for src in srcs:
		outsrcs.append(mktee(pipeline,src))
		mkqueue(pipeline,outsrcs[-1],max_size_time=0, max_size_buffers=1, max_size_bytes=0).link(adder)
	mkfakesink(pipeline, adder)
	return outsrcs


def mktcpserversink(pipeline, src, **properties):
	elem = gst.element_factory_make("tcpserversink")
	# FIXME:  are these sensible defaults?
	elem.set_property("sync", True)
	elem.set_property("sync-method", "latest-keyframe")
	elem.set_property("recover-policy", "keyframe")
	elem.set_property("unit-type", "bytes")
	elem.set_property("unist-soft-max", 1024**3)	# 1 GB
	for name, value in properties.items():
		elem.set_property(name.replace("_", "-"), value)
	pipeline.add(elem)
	src.link(elem)


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
		raise ValueError, "cannot write pipeline, environment variable GST_DEBUG_DUMP_DOT_DIR is not set"
	gst.DEBUG_BIN_TO_DOT_FILE(pipeline, gst.DEBUG_GRAPH_SHOW_ALL, filestem)
	if verbose:
		print >>sys.stderr, "Wrote pipeline to %s" % os.path.join(os.environ["GST_DEBUG_DUMP_DOT_DIR"], "%s.dot" % filestem)

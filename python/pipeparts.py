# Copyright (C) 2009  LIGO Scientific Collaboration
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


import numpy


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
from elements.channelgram import mkchannelgram
from elements.histogram import mkhistogram
from elements.spectrum import mkspectrumplot


__author__ = "Kipp Cannon <kipp.cannon@ligo.org>, Chad Hanna <chad.hanna@ligo.org>, Drew Keppel <drew.keppel@ligo.org>"
__version__ = "FIXME"
__date__ = "FIXME"


#
# =============================================================================
#
#                                Pipeline Parts
#
# =============================================================================
#


def mksegmentsrc(pipeline, segment_list, blocksize = 4096 * 1 * 1, invert_output=False):
	# default blocksize is 4096 seconds of unsigned integers at
	# 1 Hz, e.g. segments without nanoseconds
	elem = gst.element_factory_make("lal_segmentsrc")
	elem.set_property("blocksize", blocksize)
	elem.set_property("segment-list", segments.segmentlist(segments.segment(a.ns(), b.ns()) for a, b in segment_list))
	elem.set_property("invert-output", invert_output)
	pipeline.add(elem)
	return elem


def mkframesrc(pipeline, location, instrument, channel_name, blocksize = 16384 * 8 * 1, segment_list = None):
	# default blocksize is 1 second of double precision floats at
	# 16384 Hz, e.g., LIGO h(t)
	elem = gst.element_factory_make("lal_framesrc")
	elem.set_property("blocksize", blocksize)
	elem.set_property("location", location)
	elem.set_property("instrument", instrument)
	elem.set_property("channel-name", channel_name)
	if segment_list is not None:
		elem.set_property("segment-list", segments.segmentlist(segments.segment(a.ns(), b.ns()) for a, b in segment_list))
	pipeline.add(elem)
	return elem


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
	elem = gst.element_factory_make("capsfilter")
	elem.set_property("caps", gst.Caps(caps))
	pipeline.add(elem)
	src.link(elem)
	return elem


def mktaginject(pipeline, src, tags):
	elem = gst.element_factory_make("taginject")
	elem.set_property("tags", tags)
	pipeline.add(elem)
	src.link(elem)
	return elem


def mkaudiotestsrc(pipeline, **properties):
	elem = gst.element_factory_make("audiotestsrc")
	for name, value in properties.items():
		elem.set_property(name.replace("_", "-"), value)
	pipeline.add(elem)
	return elem


def mkfakesrc(pipeline, location, instrument, channel_name, blocksize = 16384 * 8 * 1, volume = 1e-20):
	# default blocksize is 1 second of double precision floats at
	# 16384 Hz, e.g., h(t)
	return mktaginject(pipeline, mkcapsfilter(pipeline, mkaudiotestsrc(pipeline, samplesperbuffer = blocksize / 8, wave = 9, volume = volume), "audio/x-raw-float, width=64, rate=16384"), "instrument=%s,channel-name=%s,units=strain" % (instrument, channel_name))


def mkiirfilter(pipeline, src, a, b):
	# convention is z = \exp(-i 2 \pi f / f_{\rm sampling})
	# H(z) = (\sum_{j=0}^{N} a_j z^{-j}) / (\sum_{j=0}^{N} (-1)^{j} b_j z^{-j})
	elem = gst.element_factory_make("audioiirfilter")
	elem.set_property("a", a)
	elem.set_property("b", b)
	pipeline.add(elem)
	src.link(elem)
	return elem


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
	elem = gst.element_factory_make("progressreport", name)
	elem.set_property("do-query", False)
	pipeline.add(elem)
	src.link(elem)
	return elem


def mkinjections(pipeline, src, filename):
	elem = gst.element_factory_make("lal_simulation")
	elem.set_property("xml-location", filename)
	pipeline.add(elem)
	src.link(elem)
	return elem


def mkaudiochebband(pipeline, src, lower_frequency, upper_frequency, poles = 8):
	elem = gst.element_factory_make("audiochebband")
	elem.set_property("lower-frequency", lower_frequency)
	elem.set_property("upper-frequency", upper_frequency)
	elem.set_property("poles", poles)
	pipeline.add(elem)
	src.link(elem)
	return elem


def mkaudiocheblimit(pipeline, src, cutoff, mode = 0, poles = 8):
	elem = gst.element_factory_make("audiocheblimit")
	elem.set_property("cutoff", cutoff)
	elem.set_property("mode", mode)
	elem.set_property("poles", poles)
	pipeline.add(elem)
	src.link(elem)
	return elem


def mkaudioamplify(pipeline, src, amplification):
	elem = gst.element_factory_make("audioamplify")
	elem.set_property("clipping-method", 3)
	elem.set_property("amplification", amplification)
	pipeline.add(elem)
	src.link(elem)
	return elem


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


def mkwhiten(pipeline, src, psd_mode = 0, zero_pad = 0, fft_length = 8, average_samples = 64, median_samples = 7):
	elem = gst.element_factory_make("lal_whiten")
	elem.set_property("psd-mode", psd_mode)
	elem.set_property("zero-pad", zero_pad)
	elem.set_property("fft-length", fft_length)
	elem.set_property("average-samples", average_samples)
	elem.set_property("median-samples", median_samples)
	pipeline.add(elem)
	src.link(elem)
	return elem


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
	elem = gst.element_factory_make("lal_drop")
	elem.set_property("drop-samples",drop_samples)
	pipeline.add(elem)
	src.link(elem)
	return elem


def mkdelay(pipeline, src, delay = 0):
	elem = gst.element_factory_make("lal_delay")
	elem.set_property("delay",delay)
	pipeline.add(elem)
	src.link(elem)
	return elem


def mknofakedisconts(pipeline, src, silent = True):
	elem = gst.element_factory_make("lal_nofakedisconts")
	elem.set_property("silent", silent)
	pipeline.add(elem)
	src.link(elem)
	return elem


def mkfirbank(pipeline, src, latency = None, fir_matrix = None, time_domain = None, block_stride = None):
	elem = gst.element_factory_make("lal_firbank")
	if latency is not None:
		elem.set_property("latency", latency)
	if fir_matrix is not None:
		elem.set_property("fir-matrix", fir_matrix)
	if time_domain is not None:
		elem.set_property("time-domain", time_domain)
	if block_stride is not None:
		elem.set_property("block-stride", block_stride)
	pipeline.add(elem)
	src.link(elem)
	elem = mknofakedisconts(pipeline, elem)	# FIXME:  remove after basetransform behaviour fixed
	return elem

def mkiirbank(pipeline, src, a1, b0, delay):
	elem = gst.element_factory_make("lal_iirbank")

	if a1 is not None:
		elem.set_property("a1-matrix", pipeio.repack_complex_array_to_real(a1))
	if b0 is not None:
		elem.set_property("b0-matrix", pipeio.repack_complex_array_to_real(b0))
	if delay is not None:
		elem.set_property("delay-matrix", delay)
	pipeline.add(elem)
	src.link(elem)
	elem = mknofakedisconts(pipeline, elem)	# FIXME:  remove after basetransform behaviour fixed
	return elem


def mkmean(pipeline, src, **properties):
	elem = gst.element_factory_make("lal_mean")
	for name, value in properties.items():
		elem.set_property(name.replace("_", "-"), value)
	pipeline.add(elem)
	src.link(elem)
	return elem


def mkreblock(pipeline, src, **properties):
	elem = gst.element_factory_make("lal_reblock")
	for name, value in properties.items():
		elem.set_property(name.replace("_", "-"), value)
	pipeline.add(elem)
	src.link(elem)
	return elem


def mksumsquares(pipeline, src, weights = None):
	elem = gst.element_factory_make("lal_sumsquares")
	if weights is not None:
		elem.set_property("weights", weights)
	pipeline.add(elem)
	src.link(elem)
	return elem


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
	elem = gst.element_factory_make("lal_matrixmixer")
	if matrix is not None:
		elem.set_property("matrix", matrix)
	pipeline.add(elem)
	src.link(elem)
	return elem


def mktogglecomplex(pipeline, src):
	elem = gst.element_factory_make("lal_togglecomplex")
	pipeline.add(elem)
	src.link(elem)
	return elem


def mkautochisq(pipeline, src, autocorrelation_matrix = None, latency = 0, snr_thresh=0):
	elem = gst.element_factory_make("lal_autochisq")
	if autocorrelation_matrix is not None:
		elem.set_property("autocorrelation-matrix", pipeio.repack_complex_array_to_real(autocorrelation_matrix))
		elem.set_property("latency", latency)
		elem.set_property("snr-thresh", snr_thresh)
	pipeline.add(elem)
	src.link(elem)
	elem = mknofakedisconts(pipeline, elem)	# FIXME:  remove after basetransform behaviour fixed
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
	elem = gst.element_factory_make("filesink")
	elem.set_property("sync", False)
	elem.set_property("async", False)
	elem.set_property("buffer-mode", 2)
	elem.set_property("location", filename)
	pipeline.add(elem)
	src.link(elem)


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


def mknxydumpsinktee(pipeline, src, filename, segment = None):
	t = mktee(pipeline, src)
	q = mkqueue(pipeline,t)
	mknxydumpsink(pipeline,q,filename,segment)
	return t
	

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
	elem = gst.element_factory_make("lal_triggerxmlwriter")
	elem.set_property("location", filename)
	elem.set_property("sync", False)
	elem.set_property("async", False)
	pipeline.add(elem)
	src.link(elem)


def mkwavenc(pipeline, src):
	elem = gst.element_factory_make("wavenc")
	pipeline.add(elem)
	src.link(elem)
	return elem


def mkcolorspace(pipeline, src):
	elem = gst.element_factory_make("ffmpegcolorspace")
	pipeline.add(elem)
	src.link(elem)
	return elem


def mktheoraenc(pipeline, src, **properties):
	elem = gst.element_factory_make("theoraenc")
	for name, value in properties.items():
		elem.set_property(name.replace("_", "-"), value)
	pipeline.add(elem)
	src.link(elem)
	return elem


def mkmpeg4enc(pipeline, src, **properties):
	elem = gst.element_factory_make("ffenc_mpeg4")
	for name, value in properties.items():
		elem.set_property(name.replace("_", "-"), value)
	pipeline.add(elem)
	src.link(elem)
	return elem


def mkoggmux(pipeline, src):
	elem = gst.element_factory_make("oggmux")
	pipeline.add(elem)
	src.link(elem)
	return elem


def mkavimux(pipeline, src):
	elem = gst.element_factory_make("avimux")
	pipeline.add(elem)
	src.link(elem)
	return elem


def mkaudioconvert(pipeline, src, caps_string = None):
	elem = gst.element_factory_make("audioconvert")
	pipeline.add(elem)
	src.link(elem)
	src = elem
	if caps_string is not None:
		src = mkcapsfilter(pipeline, src, caps_string)
	return src


def mkflacenc(pipeline, src, quality = 0, **properties):
	elem = gst.element_factory_make("flacenc")
	elem.set_property("quality", quality)
	for name, value in properties.items():
		elem.set_property(name, value)
	pipeline.add(elem)
	src.link(elem)
	return elem


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
	src = mkcolorspace(pipeline, src)
	elem = gst.element_factory_make("autovideosink")
	pipeline.add(elem)
	src.link(elem)


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
	else:
		src.link(elem)
	return elem


class AppSync(object):
	def __init__(self, appsinks=[]):
		self.appsinks = {}
		for a in appsinks:
			self.appsinks[a] = None

	def add_sink(self, pipeline, src, pad_name = None, drop = False, **properties):
		# NOTE that max buffers must be 1 for this to work
		elem = mkappsink(pipeline, src, pad_name, 1, drop, **properties)
		self.appsinks[elem] = None
		return elem

	def times(self):
		return [a.get_last_buffer().timestamp for a in self.appsinks.keys() if a.get_last_buffer() is not None]

	def num_first_buffers(self):
		return len([a for a in self.appsinks.values() if a is not None])

	def earliest_time(self):
		t=self.times()
		return min(t)

	def earliest_appsink(self):
		out = [k for k in self.appsinks.keys() if k.get_last_buffer() is not None and k.get_last_buffer().timestamp == self.earliest_time()]
		return out


def pull_appsinks_in_order(appsink,appsync):

	# get the earliest buffers
	eap = appsync.earliest_appsink()

	# mark that this one cannot emit another buffer signal (default)
	appsync.appsinks[appsink] = 1

	# This buffer has never been pulled. Pull it and mark that it could
	# emit another signal Return when we are done
	if appsync.appsinks[appsink] is None:
		appsync.appsinks[appsink] = 0
		buf = appsink.emit('pull-buffer')
		return

	# we must wait until we have at least one buffer from each appsink This
	# thread should block, return when we are done
	if appsync.num_first_buffers() != len(appsync.appsinks):
		return

	# if this is among the earliest, pull and mark that it could emit
	# another signal Return when we are done
	if appsink in eap:
		appsync.appsinks[appsink] = 0
		buf = appsink.emit('pull-buffer')
		return
	else:
		pass

	# If we have gotten this far, it is not the earliest, but we should
	# check to see if the others are, and are blocked If everything is
	# blocked we could get a deadlock. We can only unblock one appsync or
	# else crazy things could happen We should return after finding one
	# "earliest"
	not_earliest = []
	if sum([v for v in appsync.appsinks.values()]) == len(appsync.appsinks):
		for k,v in appsync.appsinks.items():
			if v == 1:
				if k in eap:
					appsync.appsinks[k] = 0
					buf = k.emit('pull-buffer')
					return
				else:
					not_earliest.append(k)

	# One last check.  If none of the buffers are "earliest", then we need
	# to check again, that could have changed.  We should updated our
	# earliest list at this point.
	eap = appsync.earliest_appsink()
	if len(not_earliest) == len(appsync.appsinks):
		not_earliest = []
		for k,v in appsync.appsinks.items():
			if v == 1:
				if k in eap:
					appsync.appsinks[k] = 0
					buf = k.emit('pull-buffer')
					return
				else:
					not_earliest.append(k)

	return


def mkchecktimestamps(pipeline, src, name = None, silent = True):
	elem = gst.element_factory_make("lal_checktimestamps")
	if name is not None:
		elem.set_property("name", name)
	elem.set_property("silent", silent)
	pipeline.add(elem)
	src.link(elem)
	return elem


def mkpeak(pipeline, src, n):
	elem = gst.element_factory_make("lal_peak")
	elem.set_property("n", n)
	pipeline.add(elem)
	src.link(elem)
	return elem


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

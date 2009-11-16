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


import math


import gobject
import pygst
pygst.require("0.10")
import gst


from elements import channelgram, histogram, spectrum


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


def mkframesrc(pipeline, location, instrument, channel_name, blocksize = 16384 * 1 * 8):
	elem = gst.element_factory_make("lal_framesrc")
	elem.set_property("blocksize", blocksize)
	elem.set_property("location", location)
	elem.set_property("instrument", instrument)
	elem.set_property("channel-name", channel_name)
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


def mkfakesrc(pipeline, location, instrument, channel_name, blocksize = 16384 * 1 * 8, volume = 1e-20):
	elem = gst.element_factory_make("audiotestsrc")
	elem.set_property("samplesperbuffer", blocksize / 8)
	elem.set_property("wave", 9)
	elem.set_property("volume", volume)
	pipeline.add(elem)
	return mktaginject(pipeline, mkcapsfilter(pipeline, elem, "audio/x-raw-float, width=64, rate=16384"), "instrument=%s,channel-name=%s,units=strain" % (instrument, channel_name))


def mkiirfilter(pipeline, src, a, b):
	elem = gst.element_factory_make("audioiirfilter")
	elem.set_property("a", a)
	elem.set_property("b", b)
	pipeline.add(elem)
	src.link(elem)
	return elem


def mkfakeLIGOsrc(pipeline, location, instrument, channel_name, blocksize = 16384 * 1 * 8):
	head1 = mkfakesrc(pipeline, location = location, instrument = instrument, channel_name = channel_name, blocksize = blocksize, volume = 5.03407936516e-17)
	a = [1.87140685e-05, 3.74281370e-05, 1.87140685e-05]
	b = [1., 1.98861643, -0.98869215]
	for idx in range(14):
		head1 = mkiirfilter(pipeline, head1, a, b)

	head2 = mkfakesrc(pipeline, location = location, instrument = instrument, channel_name = channel_name, blocksize = blocksize, volume = 1.39238913312e-20)
	a = [9.17933667e-07, 1.83586733e-06, 9.17933667e-07]
	b = [1., 1.99728828, -0.99729195]
	head2 = mkiirfilter(pipeline, head2, a, b)

	head3 = mkfakesrc(pipeline, location = location, instrument = instrument, channel_name = channel_name, blocksize = blocksize, volume = 2.16333076528e-23)

	head4 = mkfakesrc(pipeline, location = location, instrument = instrument, channel_name = channel_name, blocksize = blocksize, volume = 1.61077910675e-20)
	a = [0.03780506, -0.03780506]
	b = [1.0, -0.9243905]
	head4 = mkiirfilter(pipeline, head4, a, b)

	head = gst.element_factory_make("adder")
	pipeline.add(head)
	head1.link(head)
	head2.link(head)
	head3.link(head)
	head4.link(head)
	return mkaudioamplify(pipeline, head, 16384.**.5)


def mkprogressreport(pipeline, src, name):
	elem = gst.element_factory_make("progressreport", name)
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


def mkaudioamplify(pipeline, src, amplification):
	elem = gst.element_factory_make("audioamplify")
	elem.set_property("clipping-method", 3)
	elem.set_property("amplification", amplification)
	pipeline.add(elem)
	src.link(elem)
	return elem


def mkresample(pipeline, src, pad_name = None, **properties):
	elem = gst.element_factory_make("audioresample")
	elem.set_property("gap-aware", True)
	for name, value in properties.items():
		elem.set_property(name.replace("_", "-"), value)
	pipeline.add(elem)
	if pad_name is None:
		src.link(elem)
	else:
		src.link_pads(pad_name, elem, "sink")
	return elem


def mkwhiten(pipeline, src):
	elem = gst.element_factory_make("lal_whiten")
	elem.set_property("psd-mode", 0)
	elem.set_property("zero-pad", 0)
	elem.set_property("fft-length", 8)
	elem.set_property("average-samples", 64)
	elem.set_property("median-samples", 7)
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


def mkfirbank(pipeline, src, latency = None, fir_matrix = None):
	elem = gst.element_factory_make("lal_firbank")
	if latency is not None:
		elem.set_property("latency", latency)
	if fir_matrix is not None:
		elem.set_property("fir-matrix", fir_matrix)
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


def mkgate(pipeline, src, threshold = None, control = None):
	elem = gst.element_factory_make("lal_gate")
	if threshold is not None:
		elem.set_property("threshold", threshold)
	pipeline.add(elem)
	src.link_pads("src", elem, "sink")
	if control is not None:
		control.link_pads("src", elem, "control")
	return elem


def mkmatrixmixer(pipeline, src, matrix = None):
	elem = gst.element_factory_make("lal_matrixmixer")
	if matrix is not None:
		elem.set_property("matrix", matrix)
	pipeline.add(elem)
	src.link(elem)
	return elem


def mkLLOIDbranch(pipeline, src, bank, bank_fragment, control_snk, control_src):
	# FIXME:  latency, fir_matrix
	src = mktee(pipeline, mkfirbank(pipeline, src, latency = None, fir_matrix = None))

	# FIXME:  weights
	mkresample(pipeline, mkqueue(pipeline, mksumsquares(pipeline, src, weights = None))).link(control_snk)

	src = mkgate(pipeline, mkqueue(pipeline, src), control = mkqueue(pipeline, control_src), threshold = bank.gate_threshold)

	# FIXME:  teach the collectpads object not to wait for buffers on
	# pads whose segments have not yet been reached by the input on the
	# other pads.  then this large queue buffer will not be required
	# because streaming can begin through the downstream adders without
	# waiting for input from all upstream elements.
	src = mkqueue(pipeline, src, max_size_buffers = 0, max_size_bytes = 0, max_size_time = 2 * int(math.ceil(bank.filter_length)) * 1000000000)

	# FIXME:  matrix
	return mkresample(pipeline, mkmatrixmixer(pipeline, src, matrix = None), quality = 0)


def mkfakesink(pipeline, src, pad = None):
	elem = gst.element_factory_make("fakesink")
	elem.set_property("sync", False)
	elem.set_property("preroll-queue-len", 1)
	pipeline.add(elem)
	if pad is not None:
		src.link_pads(pad, elem, "sink")
	else:
		src.link(elem)


def mkfilesink(pipeline, src, filename):
	elem = gst.element_factory_make("filesink")
	elem.set_property("sync", False)
	elem.set_property("preroll-queue-len", 1)
	elem.set_property("buffer-mode", 2)
	elem.set_property("location", filename)
	pipeline.add(elem)
	src.link(elem)


def mknxydumpsink(pipeline, src, filename):
	elem = gst.element_factory_make("lal_nxydump")
	if False:
		# output for hardware injection @ 874107078.149271066
		elem.set_property("start-time", 874107068000000000)
		elem.set_property("stop-time", 874107088000000000)
	elif False:
		# output for impulse injection @ 873337860
		elem.set_property("start-time", 873337850000000000)
		elem.set_property("stop-time", 873337960000000000)
	elif False:
		# output for use with software injections:
		# bns_injections.xml = 874107198.405080859, impulse =
		# 874107189
		elem.set_property("start-time", 874107188000000000)
		elem.set_property("stop-time", 874107258000000000)
	elif False:
		# FIXME:  what's at this time?
		elem.set_property("start-time", 873248760000000000)
		elem.set_property("stop-time", 873248960000000000)
	elif False:
		# output to dump lots and lots of data (the whole cache)
		elem.set_property("start-time", 873247860000000000)
		elem.set_property("stop-time", 873424754000000000)
	else:
		pass
	pipeline.add(elem)
	src.link(elem)
	# FIXME:  add bz2enc element from plugins-bad to compress text
	# streams.
	#src = elem
	#elem = gst.element_factory_make("bz2enc")
	#pipeline.add(elem)
	#src.link(elem)
	mkfilesink(pipeline, elem, filename)


def mktriggergen(pipeline, snr, chisq, template_bank_filename, snr_threshold):
	elem = gst.element_factory_make("lal_triggergen")
	elem.set_property("bank-filename", template_bank_filename)
	elem.set_property("snr-thresh", snr_threshold)
	pipeline.add(elem)
	snr.link_pads("src", elem, "snr")
	chisq.link(elem)
	return elem


def mktriggerxmlwritersink(pipeline, src, filename):
	elem = gst.element_factory_make("lal_triggerxmlwriter")
	elem.set_property("location", filename)
	elem.set_property("sync", False)
	elem.set_property("preroll-queue-len", 1)
	pipeline.add(elem)
	src.link(elem)


def mkchannelgram(pipeline, src):
	elem = channelgram.Channelgram()
	pipeline.add(elem)
	src.link(elem)
	return elem


def mkhistogram(pipeline, src):
	elem = histogram.Histogram()
	pipeline.add(elem)
	src.link(elem)
	return elem


def mkspectrumplot(pipeline, src, pad = None):
	elem = spectrum.Spectrum()
	pipeline.add(elem)
	if pad is not None:
		src.link_pads(pad, elem, "sink")
	else:
		src.link(elem)
	return elem


def mkcolorspace(pipeline, src):
	elem = gst.element_factory_make("ffmpegcolorspace")
	pipeline.add(elem)
	src.link(elem)
	return elem


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
		gst.element_factory_make("audioresample"),
		gst.element_factory_make("audioconvert"),
		gst.element_factory_make("queue"),
		gst.element_factory_make("alsasink")
	)
	elems[1].set_property("caps", gst.Caps("audio/x-raw-float, width=64"))
	elems[2].set_property("amplification", amplification)
	elems[5].set_property("max-size-time", 1 * gst.SECOND)
	pipeline.add(*elems)
	gst.element_link_many(src, *elems)

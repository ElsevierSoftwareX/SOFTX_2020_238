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
import numpy


import gobject
import pygst
pygst.require("0.10")
import gst


from pylal import datatypes as laltypes


__author__ = "Kipp Cannon <kipp.cannon@ligo.org>, Chad Hanna <chad.hanna@ligo.org>, Drew Keppel <drew.keppel@ligo.org>"
__version__ = "FIXME"
__date__ = "FIXME"


#
# =============================================================================
#
#                                   Messages
#
# =============================================================================
#


def parse_spectrum_message(message):
	"""
	Parse a "spectrum" message from the lal_whiten element, return a
	LAL REAL8FrequencySeries containing the strain spectral density.
	"""
	return laltypes.REAL8FrequencySeries(
		name = "PSD",
		epoch = laltypes.LIGOTimeGPS(0, message.structure["timestamp"]),
		f0 = 0.0,
		deltaF = message.structure["delta-f"],
		sampleUnits = laltypes.LALUnit(message.structure["sample-units"]),
		data = numpy.array(message.structure["magnitude"])
	)


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
	head1 = mkfakesrc(pipeline, location = location, instrument = instrument, channel_name = channel_name, blocksize = blocksize, volume = 1.0)
	a = [1.87140685e-05, 3.74281370e-05, 1.87140685e-05]
	b = [1., 1.98861643, -0.98869215]
	for idx in range(14):
		head1 = mkiirfilter(pipeline, head1, a, b)
	head1 = mkaudioamplify(pipeline, head1, 5.03407936516e-17)

	head2 = mkfakesrc(pipeline, location = location, instrument = instrument, channel_name = channel_name, blocksize = blocksize, volume = 1.0)
	a = [9.17933667e-07, 1.83586733e-06, 9.17933667e-07]
	b = [1., 1.99728828, -0.99729195]
	head2 = mkiirfilter(pipeline, head2, a, b)
	head2 = mkaudioamplify(pipeline, head2, 1.39238913312e-20)

	head3 = mkfakesrc(pipeline, location = location, instrument = instrument, channel_name = channel_name, blocksize = blocksize, volume = 1.0)
	head3 = mkaudioamplify(pipeline, head3, 2.16333076528e-23)

	head4 = mkfakesrc(pipeline, location = location, instrument = instrument, channel_name = channel_name, blocksize = blocksize, volume = 1.0)
	a = [0.03780506, -0.03780506]
	b = [1.0, -0.9243905]
	head4 = mkiirfilter(pipeline, head4, a, b)
	head4 = mkaudioamplify(pipeline, head4, 1.61077910675e-20)

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

def mkbank_oldchisq(pipeline, src, bank, bank_fragment, reference_psd_filename, control_snk, control_src):
	elem = gst.element_factory_make("lal_templatebank")
	elem.set_property("t-start", bank_fragment.start)
	elem.set_property("t-end", bank_fragment.end)
	elem.set_property("t-total-duration", bank.filter_length)
	elem.set_property("snr-length", bank_fragment.blocksize)
	elem.set_property("template-bank", bank.template_bank_filename)
	elem.set_property("reference-psd", reference_psd_filename)
	pipeline.add(elem)
	src.link(elem)

	matrix = mktee(pipeline, elem, pad_name = "matrix")

	mkqueue(pipeline, mkresample(pipeline, elem, pad_name = "sumofsquares")).link(control_snk)

	gate = gst.element_factory_make("lal_gate")
	gate.set_property("threshold", bank.gate_threshold)
	pipeline.add(gate)
	mkqueue(pipeline, elem, pad_name = "src").link_pads("src", gate, "sink")
	mkqueue(pipeline, control_src).link_pads("src", gate, "control")
	# FIXME:  teach the collectpads object not to wait for buffers on
	# pads whose segments have not yet been reached by the input on the
	# other pads.  then this large queue buffer will not be required
	# because streaming can begin through the downstream adders without
	# waiting for input from all upstream elements.
	orthosnr = mktee(pipeline, mkqueue(pipeline, gate, pad_name = "src", max_size_buffers = 0, max_size_bytes = 0, max_size_time = 2 * int(math.ceil(bank.filter_length)) * 1000000000))

	#mknxydumpsink(pipeline, mkqueue(pipeline, orthosnr), "orthosnr_%s.txt" % elem.get_name())

	snr = gst.element_factory_make("lal_matrixmixer")
	pipeline.add(snr)
	mkqueue(pipeline, matrix).link_pads("src", snr, "matrix")
	mkqueue(pipeline, orthosnr).link(snr)
	snr = mktee(pipeline, snr)

	chisq = gst.element_factory_make("lal_chisquare")
	pipeline.add(chisq)
	mkqueue(pipeline, matrix).link_pads("src", chisq, "matrix")
	mkqueue(pipeline, elem, pad_name = "chifacs").link_pads("src", chisq, "chifacs")
	mkqueue(pipeline, orthosnr).link_pads("src", chisq, "orthosnr")
	mkqueue(pipeline, snr).link_pads("src", chisq, "snr")

	return mkresample(pipeline, snr, quality = 0), mkresample(pipeline, chisq, quality = 0)

def mkbank_newchisq(pipeline, src, bank, bank_fragment, reference_psd_filename, control_snk, control_src):
	elem = gst.element_factory_make("lal_templatebank")
	elem.set_property("t-start", bank_fragment.start)
	elem.set_property("t-end", bank_fragment.end)
	elem.set_property("t-total-duration", bank.filter_length)
	elem.set_property("snr-length", bank_fragment.blocksize)
	elem.set_property("template-bank", bank.template_bank_filename)
	elem.set_property("reference-psd", reference_psd_filename)
	pipeline.add(elem)
	src.link(elem)

	# chifacs not needed by new \chi^{2} element
	mkfakesink(pipeline, mkqueue(pipeline, elem, pad_name = "chifacs"))

	mkqueue(pipeline, mkresample(pipeline, elem, pad_name = "sumofsquares")).link(control_snk)

	gate = gst.element_factory_make("lal_gate")
	gate.set_property("threshold", bank.gate_threshold)
	pipeline.add(gate)
	mkqueue(pipeline, elem, pad_name = "src").link_pads("src", gate, "sink")
	mkqueue(pipeline, control_src).link_pads("src", gate, "control")
	# FIXME:  teach the collectpads object not to wait for buffers on
	# pads whose segments have not yet been reached by the input on the
	# other pads.  then this large queue buffer will not be required
	# because streaming can begin through the downstream adders without
	# waiting for input from all upstream elements.
	orthosnr = mkqueue(pipeline, gate, pad_name = "src", max_size_buffers = 0, max_size_bytes = 0, max_size_time = 2 * int(math.ceil(bank.filter_length)) * 1000000000)

	snr = gst.element_factory_make("lal_matrixmixer")
	pipeline.add(snr)
	mkqueue(pipeline, elem, pad_name = "matrix").link_pads("src", snr, "matrix")
	orthosnr.link(snr)

	return mkresample(pipeline, snr, quality = 0)

def mkfakesink(pipeline, src):
	elem = gst.element_factory_make("fakesink")
	elem.set_property("sync", False)
	elem.set_property("preroll-queue-len", 1)
	pipeline.add(elem)
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
	elif True:
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
	else:
		# output to dump lots and lots of data (the whole cache)
		elem.set_property("start-time", 873247860000000000)
		elem.set_property("stop-time", 873424754000000000)
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

def mkscopesink(pipeline, src):
	elems = (
		gst.element_factory_make("lal_multiscope"),
		gst.element_factory_make("ffmpegcolorspace"),
		gst.element_factory_make("cairotimeoverlay"),
		gst.element_factory_make("autovideosink")
	)
	elems[0].set_property("trace-duration", 4.0)
	elems[0].set_property("frame-interval", 1.0 / 16)
	elems[0].set_property("average-interval", 32.0)
	elems[0].set_property("do-timestamp", False)
	pipeline.add(*elems)
	gst.element_link_many(mkqueue(pipeline, src), *elems)

def mkplaybacksink(pipeline, src):
	elems = (
		gst.element_factory_make("adder"),
		gst.element_factory_make("audioresample"),
		gst.element_factory_make("audioconvert"),
		gst.element_factory_make("capsfilter"),
		gst.element_factory_make("audioamplify"),
		gst.element_factory_make("audioconvert"),
		gst.element_factory_make("queue"),
		gst.element_factory_make("alsasink")
	)
	elems[3].set_property("caps", gst.Caps("audio/x-raw-float, width=32"))
	elems[4].set_property("amplification", 5e-2)
	elems[6].set_property("max-size-time", 3000000000)
	pipeline.add(*elems)
	gst.element_link_many(*elems)

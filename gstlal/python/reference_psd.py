#!/usr/bin/env python
#
# Copyright (C) 2010  Kipp Cannon, Chad Hanna, Leo Singer
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
import scipy
import scipy.fftpack
from scipy import interpolate
import sys
import signal


# The following snippet is taken from http://gstreamer.freedesktop.org/wiki/FAQ#Mypygstprogramismysteriouslycoredumping.2Chowtofixthis.3F
import pygtk
pygtk.require("2.0")
import gobject
gobject.threads_init()
import pygst
pygst.require("0.10")
import gst


from glue.ligolw import ligolw
from glue.ligolw import utils
from glue.ligolw import param
from glue.ligolw import types as ligolw_types
from pylal import datatypes as laltypes
from pylal import series as lalseries
from pylal import window


from gstlal import pipeparts
from gstlal import pipeio


#
# =============================================================================
#
#                               PSD Measurement
#
# =============================================================================
#


def measure_psd(instrument, seekevent, detector, seg, rate, fake_data = None, online_data = False, injection_filename = None, psd_fft_length = 8, frame_segments = None, verbose = False):
	# FIXME:  why can't this be done at the top with the other imports?
	# yes it creates a cyclic dependency, but there's no reason why it
	# shouldn't work that I can see.
	from gstlal import lloidparts

	#
	# pipeline handler for PSD measurement
	#

	class PSDHandler(lloidparts.LLOIDHandler):
		def on_message(self, bus, message):
			if message.type == gst.MESSAGE_ELEMENT and message.structure.get_name() == "spectrum":
				self.psd = pipeio.parse_spectrum_message(message)
			else:
				super(type(self), self).on_message(bus, message)

	#
	# 8 FFT-lengths is just a ball-parky estimate of how much data is
	# needed for a good PSD, this isn't a requirement of the code (the
	# code requires a minimum of 1)
	#

	if float(abs(seg)) < 8 * psd_fft_length:
		raise ValueError, "segment %s too short" % str(seg)

	#
	# build pipeline
	#

	if verbose:
		print >>sys.stderr, "measuring PSD in segment %s" % str(seg)
		print >>sys.stderr, "building pipeline ..."
	mainloop = gobject.MainLoop()
	pipeline = gst.Pipeline("psd")
	handler = PSDHandler(mainloop, pipeline)

	head = lloidparts.mkLLOIDbasicsrc(pipeline, seekevent, instrument, detector, fake_data = fake_data, online_data = online_data, injection_filename = injection_filename, frame_segments = frame_segments, verbose = verbose)
	head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw-float, rate=[%d,MAX]" % rate)	# disallow upsampling
	head = pipeparts.mkresample(pipeline, head, quality = 9)
	head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw-float, rate=%d" % rate)
	head = pipeparts.mkqueue(pipeline, head, max_size_buffers = 8)
	head = pipeparts.mkwhiten(pipeline, head, psd_mode = 0, zero_pad = 0, fft_length = psd_fft_length, average_samples = int(round(float(abs(seg)) / (psd_fft_length / 2) - 1)), median_samples = 7)
	pipeparts.mkfakesink(pipeline, head)

	#
	# process segment
	#

	if verbose:
		print >>sys.stderr, "putting pipeline into playing state ..."
	pipeline.set_state(gst.STATE_PLAYING)
	if verbose:
		print >>sys.stderr, "running pipeline ..."

	class SigData(object):
		def __init__(self):
			self.has_been_signaled = False

	sigdata = SigData()

	def signal_handler(signal, frame, pipeline = pipeline, sigdata = sigdata):
		if not sigdata.has_been_signaled:
			print >>sys.stderr, "*** SIG %d attempting graceful shutdown... ***" % (signal,)
			# override file name with approximate interval
			bus = pipeline.get_bus()
			bus.post(gst.message_new_eos(pipeline))
			sigdata.has_been_signaled = True
		else:
			print >>sys.stderr, "*** received SIG %d, but already handled... ***" % (signal,)

	# this is how the program could stop gracefully
	signal.signal(signal.SIGINT, signal_handler)
	signal.signal(signal.SIGTERM, signal_handler)

	mainloop.run()

	#
	# done
	#

	if verbose:
		print >>sys.stderr, "PSD measurement complete"
	return handler.psd


def read_psd(filename, verbose = False):
	return dict((param.get_pyvalue(elem, u"instrument"), lalseries.parse_REAL8FrequencySeries(elem)) for elem in utils.load_filename(filename, verbose = verbose).getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") and elem.getAttribute(u"Name") == u"REAL8FrequencySeries")


def write_psd(filename, psddict, verbose = False):
	xmldoc = ligolw.Document()
	lw = xmldoc.appendChild(ligolw.LIGO_LW())
	for instrument, psd in psddict.items():
		fs = lw.appendChild(lalseries.build_REAL8FrequencySeries(psd))
		if instrument is not None:
			fs.appendChild(param.from_pyvalue(u"instrument", instrument))
	utils.write_filename(xmldoc, filename, gz = (filename or "stdout").endswith(".gz"), verbose = verbose)


#
# =============================================================================
#
#                                PSD Utilities
#
# =============================================================================
#


def psd_to_fir_kernel(psd):
	"""
	Compute a finite impulse-response filter kernel from a power
	spectral density conforming to the LAL normalization convention,
	such that if zero-mean unit-variance Gaussian random noise is fed
	into an FIR filter using the kernel the filter's output will
	possess the given PSD.  The PSD must be provided as a
	REAL8FrequencySeries object (see
	pylal.xlal.datatypes.real8frequencyseries).

	The return value is the tuple (kernel, latency, sample rate).  The
	kernel is a numpy array containing the filter kernel, the latency
	is the filter latency in samples and the sample rate is in Hz.  The
	kernel and latency can be used, for example, with gstreamer's stock
	audiofirfilter element.
	"""
	#
	# this could be relaxed with some work
	#

	assert psd.f0 == 0.0

	#
	# extract the PSD bins and determine sample rate for kernel
	#

	data = psd.data / 2
	sample_rate = 2 * int(round(psd.f0 + len(data) * psd.deltaF))

	#
	# compute the FIR kernel.  it always has an odd number of samples
	# and no DC offset.
	#

	data[0] = data[-1] = 0.0
	try:
		kernel = scipy.fftpack.idct(numpy.sqrt(data), type = 1) * math.sqrt(sample_rate) / (2 * (len(data) - 1))
		kernel = numpy.hstack((kernel[::-1], kernel[1:]))
	except AttributeError:
		# this computer's scipy.fftpack is missing idct()
		# repack data:  data[0], data[1], 0, data[2], 0, ....
		tmp = numpy.zeros((2 * (len(data) - 1),), dtype = data.dtype)
		tmp[0:len(data)] = data
		tmp[len(data):] = data[-2:0:-1]
		kernel = numpy.roll(numpy.real( scipy.ifft((tmp)**0.5)),(len(data) - 1)) * sample_rate**.5
		kernel = numpy.hstack((kernel,[0]))


	#
	# apply a Tukey window whose flat bit is 50% of the kernel.
	# preserve the FIR kernel's square magnitude
	#

	norm_before = numpy.dot(kernel, kernel)
	kernel *= window.XLALCreateTukeyREAL8Window(len(kernel), .5).data
	kernel *= math.sqrt(norm_before / numpy.dot(kernel, kernel))

	#
	# the kernel's latency
	#

	latency = (len(kernel) + 1) / 2

	#
	# done
	#

	return kernel, latency, sample_rate


def interpolate_psd(psd, deltaF):
	#
	# no-op?
	#

	if deltaF == psd.deltaF:
		return psd

	#
	# interpolate PSD by clipping/zero-padding time-domain impulse
	# response of equivalent whitening filter
	#

	#from scipy import fftpack
	#psd_data = psd.data
	#x = numpy.zeros((len(psd_data) * 2 - 2,), dtype = "double")
	#psd_data = numpy.where(psd_data, psd_data, float("inf"))
	#x[0] = 1 / psd_data[0]**.5
	#x[1::2] = 1 / psd_data[1:]**.5
	#x = fftpack.irfft(x)
	#if deltaF < psd.deltaF:
	#	x *= numpy.cos(numpy.arange(len(x)) * math.pi / (len(x) + 1))**2
	#	x = numpy.concatenate((x[:(len(x) / 2)], numpy.zeros((int(round(len(x) * psd.deltaF / deltaF)) - len(x),), dtype = "double"), x[(len(x) / 2):]))
	#else:
	#	x = numpy.concatenate((x[:(int(round(len(x) * psd.deltaF / deltaF)) / 2)], x[-(int(round(len(x) * psd.deltaF / deltaF)) / 2):]))
	#	x *= numpy.cos(numpy.arange(len(x)) * math.pi / (len(x) + 1))**2
	#x = 1 / fftpack.rfft(x)**2
	#psd_data = numpy.concatenate(([x[0]], x[1::2]))

	#
	# interpolate PSD with linear interpolator
	#

	#psd_data = psd.data
	#f = psd.f0 + numpy.arange(len(psd_data)) * psd.deltaF
	#interp = interpolate.interp1d(f, psd_data, bounds_error = False)
	#f = psd.f0 + numpy.arange(round(len(psd_data) * psd.deltaF / deltaF)) * deltaF
	#psd_data = interp(f)

	#
	# interpolate log(PSD) with cubic spline.  note that the PSD is
	# clipped at 1e-300 to prevent nan's in the interpolator (which
	# doesn't seem to like the occasional sample being -inf)
	#

	psd_data = psd.data
	psd_data = numpy.where(psd_data, psd_data, 1e-300)
	f = psd.f0 + numpy.arange(len(psd_data)) * psd.deltaF
	interp = interpolate.splrep(f, numpy.log(psd_data), s = 0)
	f = psd.f0 + numpy.arange(round((len(psd_data) - 1) * psd.deltaF / deltaF) + 1) * deltaF
	psd_data = numpy.exp(interpolate.splev(f, interp, der = 0))

	#
	# return result
	#

	return laltypes.REAL8FrequencySeries(
		name = psd.name,
		epoch = psd.epoch,
		f0 = psd.f0,
		deltaF = deltaF,
		sampleUnits = psd.sampleUnits,
		data = psd_data
	)


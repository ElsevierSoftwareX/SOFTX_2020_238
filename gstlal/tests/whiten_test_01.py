#!/usr/bin/env python3
# Copyright (C) 2009,2010  Kipp Cannon
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
import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
GObject.threads_init()
Gst.init(None)


from gstlal import pipeparts
import cmp_nxydumps
import test_common


#
# =============================================================================
#
#                                  Pipelines
#
# =============================================================================
#


#
# is the whiten element an identity transform when given a unit PSD?  in
# and out timeseries should be identical modulo FFT precision and start-up
# and shut-down transients.
#


def whiten_test_01a(pipeline, name):
	#
	# signal handler to construct a new unit PSD (with LAL's
	# normalization) whenever the frequency resolution or Nyquist
	# frequency changes.  LAL's normalization is such that the integral
	# of the PSD yields the variance in the time domain, therefore
	#
	# PSD =  1 / (n \Delta f)
	#

	def psd_resolution_changed(elem, pspec, ignored):
		delta_f = elem.get_property("delta-f")
		f_nyquist = elem.get_property("f-nyquist")
		n = int(round(f_nyquist / delta_f) + 1)
		elem.set_property("mean-psd", numpy.ones((n,), dtype = "double") / (n * delta_f))

	#
	# try changing these.  test should still work!
	#

	rate = 2048	# Hz
	zero_pad = 0.0		# seconds
	fft_length = 4.0	# seconds
	buffer_length = 1.0	# seconds
	test_duration = 100.0	# seconds

	#
	# build pipeline
	#

	head = test_common.test_src(pipeline, buffer_length = buffer_length, rate = rate, test_duration = test_duration, wave = 9)
	head = pipeparts.mkgeneric(pipeline, head, "audiocheblimit", mode = 1, cutoff = 0.25)
	head = tee = pipeparts.mktee(pipeline, head)
	head = pipeparts.mkwhiten(pipeline, head, psd_mode = 1, zero_pad = zero_pad, fft_length = fft_length)
	head.connect_after("notify::f-nyquist", psd_resolution_changed, None)
	head.connect_after("notify::delta-f", psd_resolution_changed, None)
	head = pipeparts.mkchecktimestamps(pipeline, head)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, head), "%s_out.dump" % name)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, tee, max_size_time = int(fft_length * Gst.SECOND)), "%s_in.dump" % name)

	#
	# done
	#

	return pipeline


#
# does the whitener turn coloured Gaussian noise into zero-mean,
# unit-variance stationary white Gaussian noise?
#


def whiten_test_01b(pipeline, name):
	#
	# try changing these.  test should still work!
	#

	rate = 2048	# Hz
	zero_pad = 0.0		# seconds
	fft_length = 4.0	# seconds
	buffer_length = 1.0	# seconds
	test_duration = 10000.0	# seconds

	#
	# build pipeline
	#

	head = test_common.test_src(pipeline, buffer_length = buffer_length, rate = rate, test_duration = test_duration, wave = 6)
	head = pipeparts.mkwhiten(pipeline, head, psd_mode = 0, zero_pad = zero_pad, fft_length = fft_length)
	head = pipeparts.mkchecktimestamps(pipeline, head)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, head), "%s_out.dump" % name)

	#
	# done
	#

	return pipeline


#
# =============================================================================
#
#                                     Main
#
# =============================================================================
#


test_common.build_and_run(whiten_test_01a, "whiten_test_01a")
test_common.build_and_run(whiten_test_01b, "whiten_test_01b")

cmp_nxydumps.compare("whiten_test_01a_in.dump", "whiten_test_01a_out.dump", transients = (2.0, 2.0), sample_fuzz = 1e-2)

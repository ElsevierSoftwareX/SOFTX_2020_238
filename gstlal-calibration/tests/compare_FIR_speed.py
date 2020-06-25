#!/usr/bin/env python
# Copyright (C) 2020  Aaron Viets
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
#				   Preamble
#
# =============================================================================
#


import numpy
import sys
from gstlal import pipeparts
from gstlal import calibration_parts
import test_common
from gi.repository import Gst


#
# =============================================================================
#
#				  Utilities
#
# =============================================================================
#


def fir_matrix_update(elem, arg, filtered):
	filtered.set_property("kernel", elem.get_property("fir_matrix")[0][::-1])
	print("fir matrix updated")


#
# =============================================================================
#
#				  Pipelines
#
# =============================================================================
#


rate = 16384			# Hz
test_duration = 100.0		# seconds
buffer_length = 1.0		# seconds
filt_dur = 2.0			# seconds
fir_filt = numpy.ones(int(numpy.floor(rate * filt_dur / 2.0 + 1) * 2.0 - 2.0))
latency = int(rate * filt_dur / 2.0 + 1)


def lal_complexfirbank_01(pipeline, name):

	#
	# This test passes ones through lal_complexfirbank
	#

	src = test_common.test_src(pipeline, buffer_length = buffer_length, wave = 0, freq = 0, rate = rate, test_duration = test_duration, width = 64)
	head = pipeparts.mkaudioamplify(pipeline, src, 0.0)
	head = pipeparts.mkgeneric(pipeline, head, "lal_add_constant", value = 1.0)
	head = pipeparts.mktee(pipeline, head)
	pipeparts.mknxydumpsink(pipeline, head, "%s_in.txt" % name)
	head = calibration_parts.mkcomplexfirbank(pipeline, head, latency = latency, fir_matrix = [fir_filt], time_domain = True)
	pipeparts.mknxydumpsink(pipeline, head, "%s_out.txt" % name)

	#
	# done
	#
	
	return pipeline


def lal_tdwhiten_01(pipeline, name):

	#
	# This test passes ones through lal_tdwhiten
	#

	src = test_common.test_src(pipeline, buffer_length = buffer_length, wave = 0, freq = 0, rate = rate, test_duration = test_duration, width = 64)
	head = pipeparts.mkaudioamplify(pipeline, src, 0.0)
	head = pipeparts.mkgeneric(pipeline, head, "lal_add_constant", value = 1.0)
	head = pipeparts.mktee(pipeline, head)
	pipeparts.mknxydumpsink(pipeline, head, "%s_in.txt" % name)
	head = pipeparts.mkgeneric(pipeline, head, "lal_tdwhiten", kernel = fir_filt[::-1], latency = latency)
	pipeparts.mknxydumpsink(pipeline, head, "%s_out.txt" % name)

	#
	# done
	#

	return pipeline


#
# =============================================================================
#
#				     Main
#
# =============================================================================
#


#test_common.build_and_run(lal_complexfirbank_01, "lal_complexfirbank_01")
test_common.build_and_run(lal_tdwhiten_01, "lal_tdwhiten_01")





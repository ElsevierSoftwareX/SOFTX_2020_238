#!/usr/bin/env python3
# Copyright (C) 2017  Aaron Viets
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
#				  Pipelines
#
# =============================================================================
#

def audioresample_01(pipeline, name):

	#
	# This test adds various noise into a stream and uses audioresample to remove it
	#

	rate = 16384		# Hz
	buffer_length = 1.0	# seconds
	test_duration = 2000.0	# seconds
	width = 64		# bits

	#
	# build pipeline
	#

	src = test_common.test_src(pipeline, buffer_length = buffer_length, wave = 5, volume = 1, rate = rate, test_duration = test_duration, width = width, verbose = False)
	tee = pipeparts.mktee(pipeline, src)
	identity = pipeparts.mkgeneric(pipeline, tee, "identity")
	head = pipeparts.mkgeneric(pipeline, tee, "lal_resample", quality = 5)
	head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw,format=F64LE,rate=2048")
	head = pipeparts.mkgeneric(pipeline, head, "audioresample", quality = 9)
	head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw,format=F64LE,rate=16384")
	head = calibration_parts.mkinterleave(pipeline, [head, identity])
	#pipeparts.mknxydumpsink(pipeline, head, "resampled_data.txt")
	#head = pipeparts.mkgeneric(pipeline, head, "splitcounter")
	pipeparts.mkgeneric(pipeline, head, "lal_transferfunction", fft_length = rate, fft_overlap = rate / 2, num_ffts = 1000, update_samples = rate * test_duration, filename = "audioresample_tf.txt")

	#
	# done
	#
	
	return pipeline

def lal_resample_01(pipeline, name):

	#
	# This test adds various noise into a stream and uses audioresample to remove it
	#

	rate = 16384		# Hz
	buffer_length = 1.0	# seconds
	test_duration = 2000.0	# seconds
	width = 64		# bits

	#
	# build pipeline
	#

	src = test_common.test_src(pipeline, buffer_length = buffer_length, wave = 5, volume = 1, rate = rate, test_duration = test_duration, width = width, verbose = False)
	tee = pipeparts.mktee(pipeline, src)
	identity = pipeparts.mkgeneric(pipeline, tee, "identity")
	head = pipeparts.mkgeneric(pipeline, tee, "lal_resample", quality = 5)
	head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw,format=F64LE,rate=2048")
	head = pipeparts.mkgeneric(pipeline, head, "lal_resample", quality = 5)
	head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw,format=F64LE,rate=16384")
	head = calibration_parts.mkinterleave(pipeline, [head, identity])
	#pipeparts.mknxydumpsink(pipeline, head, "resampled_data.txt")
	#head = pipeparts.mkgeneric(pipeline, head, "splitcounter")
	pipeparts.mkgeneric(pipeline, head, "lal_transferfunction", fft_length = rate, fft_overlap = rate / 2, num_ffts = 1000, update_samples = rate * test_duration, filename = "lal_resample_tf.txt")

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


test_common.build_and_run(audioresample_01, "audioresample_01")
#test_common.build_and_run(lal_resample_01, "lal_resample_01")





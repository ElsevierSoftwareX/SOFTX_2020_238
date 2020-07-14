#!/usr/bin/env python3
# Copyright (C) 2016  Aaron Viets
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


import random
import sys
import os
import numpy
import time
from math import pi
import resource
import datetime
import time
import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 22
matplotlib.rcParams['legend.fontsize'] = 18
matplotlib.rcParams['mathtext.default'] = 'regular'
matplotlib.use('Agg')
import glob
import matplotlib.pyplot as plt

from optparse import OptionParser, Option

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
GObject.threads_init()
Gst.init(None)

import lal
from lal import LIGOTimeGPS

from gstlal import pipeparts
from gstlal import calibration_parts
from gstlal import simplehandler
from gstlal import datasource

from ligo import segments
from gstlal import test_common


#
# =============================================================================
#
#				  Pipelines
#
# =============================================================================
#

def fill_silence_test_01(pipeline, name, line_sep = 0.5):
	#
	# This test is intended to help get rid of error messages associated with adding two streams that have different start times
	#

	rate = 16		# Hz
	buffer_length = 1.0	# seconds
	test_duration = 300.0	# seconds
	filter_latency = 1.0

	#
	# build pipeline
	#

	head = test_common.test_src(pipeline, rate = rate, test_duration = test_duration, wave = 5)
	head = pipeparts.mktee(pipeline, head)
	smooth = calibration_parts.mkcomplexfirbank(pipeline, head, latency = int((rate * 40 - 1) * filter_latency + 0.5), fir_matrix = [numpy.ones(rate * 40)], time_domain = True)
	smooth = calibration_parts.mkcomplexfirbank(pipeline, smooth, latency = 23, fir_matrix = [numpy.ones(45)], time_domain = True)
	#smooth = pipeparts.mkgeneric(pipeline, head, "lal_smoothkappas", array_size = rate * 128, avg_array_size = rate * 10, filter_latency = 1)
	#smooth = pipeparts.mkgeneric(pipeline, smooth, "splitcounter", name = "smooth")
	#head = pipeparts.mkgeneric(pipeline, head, "splitcounter", name = "unsmooth")
	#channelmux_input_dict = {}
	#channelmux_input_dict["unsmooth"] = calibration_parts.mkqueue(pipeline, head)
	#channelmux_input_dict["smooth"] = calibration_parts.mkqueue(pipeline, smooth)
	#mux = pipeparts.mkframecppchannelmux(pipeline, channelmux_input_dict, frame_duration = 64, frames_per_file = 1, compression_scheme = 6, compression_level = 3)
	head = calibration_parts.mkadder(pipeline, calibration_parts.list_srcs(pipeline, head, smooth))
	#mux = pipeparts.mkgeneric(pipeline, mux, "splitcounter", name = "sum")
	#head = calibration_parts.mkgate(pipeline, smooth, head, 0)
	pipeparts.mknxydumpsink(pipeline, head, "%s_out.txt" % name)
	#pipeparts.mkframecppfilesink(pipeline, mux, frame_type = "H1DCS", instrument = "H1")

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

test_common.build_and_run(fill_silence_test_01, "fill_silence_test_01")



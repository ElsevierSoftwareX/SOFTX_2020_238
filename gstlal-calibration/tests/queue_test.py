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


import sys
import os
import numpy
import time
import resource

from optparse import OptionParser, Option

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
GObject.threads_init()
Gst.init(None)

import lal

from gstlal import pipeparts
from gstlal import calibration_parts
from gstlal import simplehandler
from gstlal import datasource

from ligo import segments
import test_common


#
# =============================================================================
#
#				  Pipelines
#
# =============================================================================
#

def queue_01(pipeline, name):

	#
	# This test is intended to probe data flow with a demuxer
	#

	channel1 = "CAL-CS_TDEP_SUS_LINE1_UNCERTAINTY"
	channel2 = "CAL-CS_TDEP_PCALY_LINE1_UNCERTAINTY"
	channel3 = "CAL-CS_TDEP_PCALY_LINE2_UNCERTAINTY"
	channel4 = "CAL-CS_TDEP_DARM_LINE1_UNCERTAINTY"
	channel5 = "CAL-DARM_ERR_WHITEN_OUT_DBL_DQ"

	channel_list = [("L1", channel1), ("L1", channel2), ("L1", channel3), ("L1", channel4), ("L1", channel5)]

	#
	# build pipeline
	#

	src = pipeparts.mklalcachesrc(pipeline, location = "L1_raw_frames.cache", cache_dsc_regex = "L1")
	demux = pipeparts.mkframecppchanneldemux(pipeline, src, do_file_checksum = False, skip_bad_files = True, channel_list = map("%s:%s".__mod__, channel_list))
	channel1 = calibration_parts.hook_up_and_queue(pipeline, demux, channel1, "L1", 1.0)
	channel2 = calibration_parts.hook_up_and_queue(pipeline, demux, channel2, "L1", 1.0)
	channel3 = calibration_parts.hook_up_and_queue(pipeline, demux, channel3, "L1", 1.0)
	channel4 = calibration_parts.hook_up_and_queue(pipeline, demux, channel4, "L1", 1.0)
	channel5 = calibration_parts.hook_up_and_queue(pipeline, demux, channel5, "L1", 1.0)
	channel1 = pipeparts.mkgeneric(pipeline, channel1, "splitcounter", name = "channel1_1")
	channel2 = pipeparts.mkgeneric(pipeline, channel2, "splitcounter", name = "channel2_1")
	channel3 = pipeparts.mkgeneric(pipeline, channel3, "splitcounter", name = "channel3_1")
	channel4 = pipeparts.mkgeneric(pipeline, channel4, "splitcounter", name = "channel4_1")
	channel5 = pipeparts.mkgeneric(pipeline, channel5, "splitcounter", name = "channel5_1")
	pipeparts.mknxydumpsink(pipeline, channel1, "%s_channel1.dump" % name)
	pipeparts.mknxydumpsink(pipeline, channel2, "%s_channel2.dump" % name)
	pipeparts.mknxydumpsink(pipeline, channel3, "%s_channel3.dump" % name)
	pipeparts.mknxydumpsink(pipeline, channel4, "%s_channel4.dump" % name)
	pipeparts.mknxydumpsink(pipeline, channel5, "%s_channel5.dump" % name)

	#
	# done
	#

	return pipeline


def queue_02(pipeline, name):

	#
	# This test is intended to probe data flow with tees, firbanks, and queues
	#

	rate = 16384		# Hz
	buffer_length = 1.0	# seconds
	test_duration = 100.0	# seconds

	#
	# build pipeline
	#

	src = test_common.test_src(pipeline, buffer_length = buffer_length, rate = rate, test_duration = test_duration, width = 32)
#	tee = pipeparts.mktee(pipeline, src)
	head = pipeparts.mkgeneric(pipeline, src, "splitcounter", name = "before_resample")
	head = calibration_parts.mkresample(pipeline, head, "audio/x-raw, format=F32LE,rate=16")#fir_matrix=[numpy.hanning(600)], latency = 300, time_domain = True)
#	head = calibration_parts.mkinsertgap(pipeline, head, bad_data_intervals = [-1e35, 1e35], block_duration = buffer_length * 1000000000, remove_gap = False)
	head = pipeparts.mkgeneric(pipeline, head, "splitcounter", name = "after_resample")
#	head2 = pipeparts.mkfirbank(pipeline, tee, fir_matrix=[numpy.hanning(1000)], latency = 500, time_domain = True)
#	head2 = calibration_parts.mkinsertgap(pipeline, head2, bad_data_intervals = [-1e35, 1e35], block_duration = buffer_length * 1000000000, remove_gap = False)
#	head2 = pipeparts.mkgeneric(pipeline, tee, "splitcounter", name = "tee_to_adder")
#	head = calibration_parts.mkqueue(pipeline, head, 1)
#	head2 = calibration_parts.mkqueue(pipeline, head2, 1)
#	adder = calibration_parts.mkadder(pipeline, calibration_parts.list_srcs(pipeline, head, head2))
#	head = pipeparts.mkgeneric(pipeline, adder, "splitcounter", name = "before_final_dump")
	pipeparts.mknxydumpsink(pipeline, head, "%s_out.dump" % name)

	#
	# done
	#
	
	return pipeline

def queue_03(pipeline, name):

	#
	# Very simple test of data storage in queues
	#

	rate = 1000	     # Hz
	buffer_length = 1.0     # seconds
	test_duration = 100.0   # seconds

	#
	# build pipeline
	#

	src = test_common.test_src(pipeline, buffer_length = buffer_length, rate = rate, test_duration = test_duration, width = 32)
	head = pipeparts.mkgeneric(pipeline, src, "splitcounter", name = "before_queue")
	head = calibration_parts.mkqueue(pipeline, head, 50)
	head = pipeparts.mkgeneric(pipeline, head, "splitcounter", name = "after_queue")
	pipeparts.mkfakesink(pipeline, head)

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


#test_common.build_and_run(queue_01, "queue_01")
test_common.build_and_run(queue_02, "queue_02")
#test_common.build_and_run(queue_03, "queue_03")

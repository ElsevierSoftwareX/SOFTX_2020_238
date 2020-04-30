#!/usr/bin/env python3
# Copyright (C) 2013  Kipp Cannon
# Copyright (C) 2014  Chad Hanna
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
import test_common

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject
from gi.repository import Gst
GObject.threads_init()
Gst.init(None)

from gstlal import pipeparts
from gstlal import simplehandler
import signal
import time

## @file lal_checktimestamps_test_01.py
# A program to intentionally break a running stream in order to test lal_checktimestamps; see lal_checktimestamps_test_01 for more details

#
# Setup a signal hander to introduce a shift with ctrl+C
#

## @package lal_checktimestamps_test_01
#
# ### USAGE
#
# - This test program dynamically adds a one nanosecond time shift every time the user hits ctrl+C.  You need to do kill -9 to stop the program ;) Here is an example session
#
#		./lal_checktimestamps_test_01.py 
#		src (00:00:05): 5 seconds
#		src (00:00:10): 10 seconds
#		^Cshifting by 1 ns
#		lal_checktimestamps+lal_checktimestamps0: discontinuity:  timestamp = 12.000682382 s, offset = 24576;  would have been 12.000682381 s, offset = 24576
#		src (00:00:15): 15 seconds
#		^Cshifting by 2 ns
#		lal_checktimestamps+lal_checktimestamps0: discontinuity:  timestamp = 17.000682383 s, offset = 34816;  would have been 17.000682382 s, offset = 34816
#		src (00:00:20): 20 seconds
#		^Cshifting by 3 ns
#		lal_checktimestamps+lal_checktimestamps0: discontinuity:  timestamp = 22.000682384 s, offset = 45056;  would have been 22.000682383 s, offset = 45056
#		^Z
#


class SigHandler(object):
	def __init__(self, pipeline):
		self.pipeline = pipeline
		self.shift = 1
		signal.signal(signal.SIGINT, self)

	def __call__(self, signum, frame):
		print "shifting by %d ns" % self.shift
		self.pipeline.get_by_name("shift").set_property("shift", self.shift)
		self.shift += 1

# setup the pipeline and event loop
mainloop = GObject.MainLoop(context = GObject.MainContext())
pipeline = Gst.Pipeline()

# setup the test pipeline
head = test_common.test_src(pipeline, test_duration = 100.0, is_live = True)
head = pipeparts.mkshift(pipeline, head, shift = 0, name = "shift") # in nanoseconds
head = pipeparts.mkchecktimestamps(pipeline, head)
pipeparts.mkfakesink(pipeline, head)

# setup the pipeline handlers and start it running
handler = simplehandler.Handler(mainloop, pipeline)
sighand = SigHandler(pipeline)
if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
	raise RuntimeError("pipeline failed to enter PLAYING state")

# run the event loop
mainloop.run()

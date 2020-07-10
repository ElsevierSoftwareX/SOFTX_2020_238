#!/usr/bin/env python3
# Copyright (C) 2018  Aaron Viets
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

def lal_dqtukey_01(pipeline, name):

	#
	# This test passes a random series of integers through lal_dqtukey
	#

	rate_in = 4		# Hz
	rate_out = 16384	# Hz
	buffer_length = 1.0	# seconds
	test_duration = 2.0	# seconds
	transition_samples = 997

	#
	# build pipeline
	#

	head = test_common.int_test_src(pipeline, buffer_length = buffer_length, rate = rate_in, width = 32, test_duration = test_duration, wave = 5, freq = 0)
	head = pipeparts.mktee(pipeline, head)
	pipeparts.mknxydumpsink(pipeline, head, "%s_in.txt" % name)
	normal = pipeparts.mkgeneric(pipeline, head, "lal_dqtukey", transition_samples = transition_samples, required_on = 2, required_off = 1, invert_control = False)
	invert = pipeparts.mkgeneric(pipeline, head, "lal_dqtukey", transition_samples = transition_samples, required_on = 2, required_off = 1, invert_control = True)
	invwin = pipeparts.mkgeneric(pipeline, head, "lal_dqtukey", transition_samples = transition_samples, required_on = 2, required_off = 1, invert_control = False, invert_window = True)
	invboth = pipeparts.mkgeneric(pipeline, head, "lal_dqtukey", transition_samples = transition_samples, required_on = 2, required_off = 1, invert_control = True, invert_window = True)
	normal = pipeparts.mkcapsfilter(pipeline, normal, "audio/x-raw,rate=%s,format=F64LE" % rate_out)
	invert = pipeparts.mkcapsfilter(pipeline, invert, "audio/x-raw,rate=%s,format=F64LE" % rate_out)
	invwin = pipeparts.mkcapsfilter(pipeline, invwin, "audio/x-raw,rate=%s,format=F64LE" % rate_out)
	invboth = pipeparts.mkcapsfilter(pipeline, invboth, "audio/x-raw,rate=%s,format=F64LE" % rate_out)
	pipeparts.mknxydumpsink(pipeline, normal, "%s_normal_out.txt" % name)
	pipeparts.mknxydumpsink(pipeline, invert, "%s_invert_out.txt" % name)
	pipeparts.mknxydumpsink(pipeline, invwin, "%s_invwin_out.txt" % name)
	pipeparts.mknxydumpsink(pipeline, invboth, "%s_invboth_out.txt" % name)

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


test_common.build_and_run(lal_dqtukey_01, "lal_dqtukey_01")



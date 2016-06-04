#!/usr/bin/env python
# Copyright (C) 2016 Aaron Viets <aaron.viets@ligo.org>
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
import test_common


#
# =============================================================================
#
#				  Pipelines
#
# =============================================================================
#

def lal_logicalundersample_01(pipeline, name):
	#
	# This is a simple test that the undersample element outputs "1" if a pair of two consecutive inputs are odd and "0" otherwise.
	# Note that the pairing must be such that the two inputs are combined in the logical operation.
	#

	in_rate = 1024	  	# Hz
	out_rate = 512	  	# Hz
	buffer_length = 1.0     # seconds
	test_duration = 10.0    # seconds

	#
	# build pipeline
	#

	src = pipeparts.mkaudiotestsrc(pipeline, num_buffers = int(test_duration / buffer_length))
	capsfilter1 = pipeparts.mkcapsfilter(pipeline, src, "audio/x-raw, format=S32LE, rate=%d" % int(in_rate))
	tee1 = pipeparts.mktee(pipeline, capsfilter1)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, tee1), "%s_in.dump" % name)
	undersample = pipeparts.mkgeneric(pipeline, tee1, "lal_logicalundersample")
	capsfilter2 = pipeparts.mkcapsfilter(pipeline, undersample, "audio/x-raw, format=U32LE, rate=%d" % int(out_rate))
	#checktimestamps = pipeparts.mkchecktimestamps(pipeline, capsfilter2)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, capsfilter2), "%s_out.dump" % name)

	#
	# done
	#

	return pipeline

def lal_logicalundersample_02(pipeline, name):
	#
	# This is similar to the one above, but here, we wish to see how logicalundersample handles unsigned input data.
	# There is quite possibly a better way to do this...
	#

	in_rate = 1024    	# Hz
	out_rate = 512	   	# Hz
	odd_inputs = 137		# the odd unsigned int's that, when occurring in pairs, should cause an output of "1".
	buffer_length = 1.0     # seconds
	test_duration = 10.0    # seconds

	#
	# build the pipeline
	#

	src = pipeparts.mkaudiotestsrc(pipeline, num_buffers = int(test_duration / buffer_length))
	capsfilter1 = pipeparts.mkcapsfilter(pipeline, src, "audio/x-raw, format=S32LE, rate=%d" % int(in_rate))
	undersample1 = pipeparts.mkgeneric(pipeline, capsfilter1, "lal_logicalundersample", status_out = odd_inputs)
	capsfilter2 = pipeparts.mkcapsfilter(pipeline, undersample1, "audio/x-raw, format=U32LE, rate=%d" % int(in_rate))
	tee1 = pipeparts.mktee(pipeline, capsfilter2)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, tee1), "%s_in.dump" % name)
	undersample2 = pipeparts.mkgeneric(pipeline, tee1, "lal_logicalundersample")
	capsfilter3 = pipeparts.mkcapsfilter(pipeline, undersample2, "audio/x-raw, format=U32LE, rate=%d" % int(out_rate))
	#checktimestamps = pipeparts.mkchecktimestamps(pipeline, capsfilter2)
	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, capsfilter3), "%s_out.dump" % name)

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


test_common.build_and_run(lal_logicalundersample_01, "lal_logicalundersample_01")
#test_common.build_and_run(lal_logicalundersample_02, "lal_logicalundersample_02")

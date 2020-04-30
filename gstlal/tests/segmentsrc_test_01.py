#!/usr/bin/env python3
# Copyright (C) 2015  Kipp Cannon
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
from ligo import segments
from gstlal import pipeio
from gstlal import pipeparts
from lal import LIGOTimeGPS

import test_common
import cmp_nxydumps


#
# =============================================================================
#
#                                  Pipelines
#
# =============================================================================
#


def segmentsrc_test_01(pipeline, name, seg):
	segs = segments.segmentlist([segments.segment(LIGOTimeGPS(100), LIGOTimeGPS(200)), segments.segment(LIGOTimeGPS(250), LIGOTimeGPS(300))])

	head = pipeparts.mksegmentsrc(pipeline, segs, blocksize = 1)
	head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw, format=(string)U8, rate=(int)4, channels=(int)1, layout=(string)interleaved, channel-mask=(bitmask)0")
	head = pipeparts.mknxydumpsink(pipeline, head, "%s_out.dump" % name)

	f = open("%s_in.dump" % name, "w")
	for t in numpy.arange(float(seg[0]), float(seg[1]), 0.25):
		print >>f, "%g\t%d" % (t, 128 if  t in segs else 0)

	return pipeline


#
# =============================================================================
#
#                                     Main
#
# =============================================================================
#


seg = segments.segment(LIGOTimeGPS(0), LIGOTimeGPS(350))
test_common.build_and_run(segmentsrc_test_01, segment = seg, seg = seg, name = "segmentsrc_test_01a")
cmp_nxydumps.compare("segmentsrc_test_01a_in.dump", "segmentsrc_test_01a_out.dump")

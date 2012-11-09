# Copyright (C) 2010  Kipp Cannon (kipp.cannon@ligo.org)
# Copyright (C) 2010 Chad Hanna (chad.hanna@ligo.org)
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 51
# Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


"""
DAG construction tools.
"""


import math
import os
import sys
import time


from glue import iterutils
from glue import segments
from glue import segmentsUtils
from glue import pipeline
from glue.lal import CacheEntry
from pylal.datatypes import LIGOTimeGPS
from pylal import ligolw_tisi
from pylal import llwapp
from lalapps import power


__author__ = "Kipp Cannon <kipp.cannon@ligo.org>, Chad Hanna <chad.hanna@ligo.org>"
__date__ = "$Date$" #FIXME
__version__ = "$Revision$" #FIXME


def breakupseg(seg, maxextent, overlap):
	if maxextent <= 0:
		raise ValueError, "maxextent must be positive, not %s" % repr(maxextent)

	# adjust maxextent so that segments are divided roughly equally
	maxextent = max(int(abs(seg) / (int(abs(seg)) // int(maxextent) + 1)), overlap)

	seglist = segments.segmentlist()

	while abs(seg) > maxextent:
		seglist.append(segments.segment(seg[0], seg[0] + maxextent))
		seg = segments.segment(seglist[-1][1] - overlap, seg[1])

	seglist.append(seg)

	return seglist


def breakupsegs(seglist, maxextent, overlap):
	newseglist = segments.segmentlist()
	for bigseg in seglist:
		newseglist.extend(breakupseg(bigseg, maxextent, overlap))
	return newseglist
	

def breakupseglists(seglists, maxextent, overlap):
	for instrument, seglist in seglists.iteritems():
		newseglist = segments.segmentlist()
	        for bigseg in seglist:
			newseglist.extend(breakupseg(bigseg, maxextent, overlap))
	        seglists[instrument] = newseglist

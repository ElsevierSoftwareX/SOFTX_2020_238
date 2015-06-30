#
# Copyright (C) 2010
# Chad Hanna <chad.hanna@ligo.org>
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


from glue import segments
from glue.ligolw import ligolw
from glue.ligolw import lsctables
from glue.ligolw import utils as ligolw_utils
from pylal.datatypes import LIGOTimeGPS


## @file
# the simulation module
#
# Review status
#
# | Names                                          | Hash                                        | Date       |
# | -------------------------------------------    | ------------------------------------------- | ---------- |
# | Sathya, Duncan Me, Jolien, Kipp, Chad          | 7536db9d496be9a014559f4e273e1e856047bf71    | 2014-05-02 |
#
#
# | Review Hash                              | Diff                |
# | ---------------------------------------- | ------------------- |
# | 7536db9d496be9a014559f4e273e1e856047bf71 | <a href="@gstlal_cgit_diff/python/simulation.py?id=HEAD&id2=7536db9d496be9a014559f4e273e1e856047bf71">simulation.py</a> |
#


## @package python.simulation
# The simulation module code

class ContentHandler(ligolw.LIGOLWContentHandler):
	pass
lsctables.use_in(ContentHandler)


#
# open ligolw_xml file containing sim_inspiral and create a segment list
#

## Turn a file containing a sim inspiral into a segment list
def sim_inspiral_to_segment_list(fname, pad=3, verbose=False):
	"""!
	Given an xml file create a segment list that marks the time of an
	injection with padding

	- fname: the xml file name
	- pad: duration in seconds to pad the coalescence time when producint a segment, e.g., [tc-pad, tc+pad)
	"""

	# initialization

	seglist = segments.segmentlist()

	# Parse the XML file

	xmldoc = ligolw_utils.load_filename(fname, contenthandler=ContentHandler, verbose=verbose)

	# extract the padded geocentric end times into segment lists

	for row in lsctables.SimInspiralTable.get_table(xmldoc):
		t = LIGOTimeGPS(row.get_time_geocent())
		seglist.append(segments.segment(t-pad, t+pad))

	# help the garbage collector

	xmldoc.unlink()

	return seglist.coalesce()

# Copyright (C) 2015 Cody Messick
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

## @file

## @package plotsegments

import datetime
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import numpy

from ligo import segments
from ligo.lw.ligolw import LIGOLWContentHandler
from ligo.lw import array as ligolw_array
from ligo.lw import lsctables
from ligo.lw import param as ligolw_param
from ligo.lw import utils as ligolw_utils
from ligo.lw.utils import segments as ligolw_segments
from gstlal import far
import lal

#
# =============================================================================
#
#                           ligo.lw Content Handlers
#
# =============================================================================
#


class ligolwcontenthandler(LIGOLWContentHandler):
	pass
ligolw_array.use_in(ligolwcontenthandler)
ligolw_param.use_in(ligolwcontenthandler)
lsctables.use_in(ligolwcontenthandler)

def parse_segments_xml(path):
	llwsegments = ligolw_segments.LigolwSegments(ligolw_utils.load_filename(path,contenthandler=ligolwcontenthandler))
	seglistdicts = { #'frame gate': llwsegments.get_by_name(u'framesegments')}
	'h(t) gate': llwsegments.get_by_name(u'whitehtsegments'),
	'state vector': llwsegments.get_by_name(u'statevectorsegments'),
	'trigger buffers': llwsegments.get_by_name(u'triggersegments') }
	# FIXME This needs to be generalized to more than two IFOs
	seglistdicts['joint segments'] = {'H1L1': seglistdicts['state vector'].intersection(['H1','L1'])}
	return seglistdicts


def plot_segments_history(seglistdicts, segments_to_plot = ['trigger buffers', 'h(t) gate', 'state vector'], t_max = None, length = 86400., labelspacing = 10800., colors = {'H1': numpy.array((1.0, 0.0, 0.0)), 'L1':  numpy.array((0.0, 0.8, 0.0)), 'V1':  numpy.array((1.0, 0.0, 1.0)), 'H1L1': numpy.array((.5, .5, .5))}, fig = None, axes = None):
	if fig is None:
		fig = plt.figure(figsize=(15,5),)
	if axes is None:
		axes = fig.add_subplot(111)
	# If t_max is specified, cut the segments so they end at t_max,
	# otherwise set it to the current time
	if t_max is None:
		t_max = float(lal.GPSTimeNow())
	else:
		seglist_to_drop = segments.segmentlist([segments.segment(lal.LIGOTimeGPS(t_max),segments.PosInfinity)])
		for seglistdict in seglistdicts.values():
			for seglist in seglistdict.values():
				seglist -= seglist_to_drop
	t_min = t_max - length
	bottom = []
	width = []
	left_edge = []
	y_ticks = []
	y_tick_labels = []
	color_list = []
	color_dict = {'H1': numpy.array((1.0, 0.0, 0.0)), 'L1':  numpy.array((0.0, 0.8, 0.0)), 'V1':  numpy.array((1.0, 0.0, 1.0)), 'H1L1': numpy.array((.5, .5, .5))}
	x_format = tkr.FuncFormatter(lambda x, pos: datetime.datetime(*lal.GPSToUTC(int(x))[:7]).strftime('%Y-%m-%d, %H:%M:%S UTC'))
	x_ticks = numpy.arange(t_min,t_max+labelspacing, labelspacing)

	for j, segtype in enumerate(segments_to_plot):
		for row, (ifo, segmentlist) in enumerate(seglistdicts[segtype].items()):
			y_ticks.append(row + 2*j + 0.5)
			y_tick_labels.append('%s %s' % (ifo, segtype))
			bottom.extend([row + 2*j + 0.25] * len(segmentlist))
			color_list.extend([color_dict[ifo]] * len(segmentlist))
			for segment in segmentlist:
				width.append(float(segment[1]) - float(segment[0]))
				left_edge.append(float(segment[0]))

	edgecolor_list = ['w'] * len(bottom)
	fig.patch.set_alpha(0.0)
	axes.barh(bottom, width, height=0.5, left=left_edge, color=color_list, edgecolor=edgecolor_list)
	plt.xlim([t_min, t_max])
	axes.xaxis.set_major_formatter(x_format)
	plt.yticks(y_ticks, y_tick_labels)
	plt.xticks(x_ticks, rotation=10.)
	axes.tick_params(axis='y',which='both',left='off',right='off')
	axes.grid(color=(0.1,0.4,0.5), linewidth=2)
	# FIXME use this grid() when we have a newer matplotlib
	# axes.grid(color=(0.1,0.4,0.5), linewidth=2, which='both', axis='x')
	# FIXME Switch to tight_layout() when matplotlib is updated
	try:
		fig.tight_layout(pad = .8)
		return fig, axes
	except AttributeError:
                return fig, axes

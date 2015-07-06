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
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import numpy

from glue.ligolw import utils as ligolw_utils
from gstlal import far
import lal


def parse_segments_xml(path):
	# FIXME Switch to this code once glue is fixed
	#from glue.ligolw.utils import segments as ligolw_segments
	#ligolwsegments = ligolw_segments.LigolwSegments(ligolw_utils.load_filename(path,contenthandler=far.ThincaCoincParamsDistributions.LIGOLWContentHandler))
	#return {'h(t) gate': ligolwsegments.get_by_name(u'whitehtsegments'), 'state vector': ligolwsegments.get_by_name(u'statevectorsegments'), 'trigger buffers': ligolwsegments.get_by_name(u'triggersegments'), 'frame gate': ligolwsegments.get_by_name(u'framesegments')}
	xmldoc = ligolw_utils.load_filename(path,contenthandler=far.ThincaCoincParamsDistributions.LIGOLWContentHandler)
	return { #'frame gate': ligolw_utils.segments.segmenttable_get_by_name(xmldoc, u'framesegments'),
		'h(t) gate': ligolw_utils.segments.segmenttable_get_by_name(xmldoc, u'whitehtsegments'), 
		'state vector': ligolw_utils.segments.segmenttable_get_by_name(xmldoc, u'statevectorsegments'), 
		'trigger buffers': ligolw_utils.segments.segmenttable_get_by_name(xmldoc, u'triggersegments')}
	

def plot_segments_history(seglistdicts, length = 86400., labelspacing = 10800.):
	bottom = []
	width = []
	left_edge = []
	y_ticks = []
	y_tick_labels = []
	colors = []
	color_dict = {'H1': numpy.array((1.0, 0.0, 0.0)), 'L1':  numpy.array((0.0, 0.8, 0.0)), 'V1':  numpy.array((1.0, 0.0, 1.0))}
	t_max = float(lal.GPSTimeNow())
	t_min = t_max - length
	x_format = tkr.FuncFormatter(lambda x, pos: datetime.datetime(*lal.GPSToUTC(int(x))[:7]).strftime("%Y-%m-%d, %H:%M:%S UTC"))
	x_ticks = numpy.arange(t_min,t_max+labelspacing, labelspacing)

	for j, segtype in enumerate(['trigger buffers', 'h(t) gate', 'state vector']):#, 'frame gate']):
		for row, (ifo, segmentlist) in enumerate(seglistdicts[segtype].items()):
			y_ticks.append(row + 2*j + 0.5)
			y_tick_labels.append('%s %s' % (ifo, segtype))
			bottom.extend([row + 2*j + 0.25] * len(segmentlist))
			colors.extend([color_dict[ifo]] * len(segmentlist))
			for segment in segmentlist:
				width.append(float(segment[1]) - float(segment[0]))
				left_edge.append(float(segment[0]))

	edgecolors = ['w'] * len(bottom)
	fig = plt.figure(figsize=(15,5),)
	fig.patch.set_alpha(0.0)
	h = fig.add_subplot(111)
	h.barh(bottom, width, height=0.5, left=left_edge, color=colors, edgecolor=edgecolors)
	plt.xlim([t_min, t_max])
	h.xaxis.set_major_formatter(x_format)
	plt.yticks(y_ticks, y_tick_labels)
	plt.xticks(x_ticks, rotation=10.)
	h.tick_params(axis='y',which='both',left='off',right='off')
	h.grid(color=(0.1,0.4,0.5), linewidth=2)
	# FIXME use this grid() when we have a newer matplotlib
	# h.grid(color=(0.1,0.4,0.5), linewidth=2, which='both', axis='x')
	# FIXME Switch to tight_layout() when matplotlib is updated
	try:
		fig.tight_layout(pad = .8)
		return fig
        except AttributeError:
                return fig

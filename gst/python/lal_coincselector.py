# Copyright (C) 2010 Leo Singer, Nickolas Fotopoulos
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
"""
Select interesting coincidences based on combined effective SNR. This element
uses the same algorithm as lalinspiral's CoincInspiralUtils.c's
XLALClusterCoincInspiralTable. In the future, we would like the selection to be
done via network IFAR.
"""
__author__ = "Leo Singer <leo.singer@ligo.org>, Nickolas Fotopoulos <nickolas.fotopoulos@ligo.org>"

import operator

from gstlal.pipeutil import *
from gstlal.pipeio import sngl_inspiral_groups_from_buffer, sngl_inspiral_groups_to_buffer
from gstlal.ligolw_output import combined_effective_snr
try:
	all
except NameError:
	# Python < 2.5 compatibility
	from glue.iterutils import all
import traceback

class SnglCoinc(object):
	"""
	A useful intermediate coinc representation to avoid recomputing stat and time repeatedly.
	"""
	def __init__(self, sngl_group, stat):
		self.sngl_group = sngl_group
		self.stat = stat
		self.time = min(row.end_time * gst.SECOND + row.end_time_ns for row in sngl_group)

def cluster_coincs(coincs, cluster_window):
	"""
	Return a list of clustered coincs from the given input list. The coincs
	should actually just be a time-ordered list of SnglCoinc objects.
	"""
	if len(coincs) <= 1:
		return coincs
	
	clustered_coincs = []
	previous = coincs[0]
	for coinc in coincs[1:]:
		if coinc.time - previous.time > cluster_window:  # we have a cluster
			clustered_coincs.append(previous)
			previous = coinc
		else:
			if coinc.stat == previous.stat:
				raise ValueError, "Equal stats! How do I cluster them!?"
			if coinc.stat > previous.stat:  # current is louder, so keep it
				previous = coinc
			# else, leave the previous alone and drop this coinc
	# hold on to the in-progress coinc, even if it's not yet definitively clustered
	if (len(clustered_coincs) == 0) or (previous != clustered_coincs[-1]):
		clustered_coincs.append(previous)
	return clustered_coincs

class lal_coincselector(gst.Element):
	__gstdetails__ = (
		'Coincidence Selector',
		'Generic',
		__doc__,
		__author__
	)
	__gproperties__ = {
		#'min-ifar': (
		#	gobject.TYPE_UINT64,
		#	'Minimum net inverse false alarm rate (IFAR)',
		#	'Minimum network IFAR (inverse false alarm rate, also known as extrapolated waiting time) in nanoseconds.  Coincidences with an estimated IFAR less than this value will be dropped.',
		#	0, gst.CLOCK_TIME_NONE, 3600 * gst.SECOND,
		#	gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		#),
		'min-combined-eff-snr': (
			gobject.TYPE_DOUBLE,
			'Minimum combined effective SNR',
			'Minimum value of combined effective SNR (RMS effective SNR across all detectors).  Coincidences with a combined effective SNR less than this value will be dropped.',
			0, gobject.G_MAXDOUBLE, 15,
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		),
		'min-waiting-time': (
			gobject.TYPE_UINT64,
			'Minimum waiting time',
			'Minimum waiting time between coincidences (nanoseconds).',
			0, gst.CLOCK_TIME_NONE, gst.SECOND,
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		),
		#'dt': ( # FIXME: could the coinc and coincselector elements share this piece of information?
		#	gobject.TYPE_UINT64,
		#	'Coincidence window',
		#	'Coincidence window in nanoseconds.',
		#	0, gst.CLOCK_TIME_NONE, 50 * gst.MSECOND,
		#	gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		#),
	}
	__gsttemplates__ = (
		gst.PadTemplate("sink",
			gst.PAD_SINK, gst.PAD_ALWAYS,
			gst.caps_from_string("""
				application/x-lal-snglinspiral,
				channels = [2, MAX]
			""")
		),
		gst.PadTemplate("src",
			gst.PAD_SRC, gst.PAD_ALWAYS,
			gst.caps_from_string("""
				application/x-lal-snglinspiral,
				channels = [2, MAX]
			""")
		)
	)


	def __init__(self):
		super(lal_coincselector, self).__init__()
		self.create_all_pads()
		self.__srcpad = self.get_static_pad('src')
		self.__sinkpad = self.get_static_pad('sink')
		self.__srcpad.use_fixed_caps() # FIXME: better to use proxycaps?
		self.__sinkpad.use_fixed_caps() # FIXME: better to use proxycaps?
		self.__sinkpad.set_chain_function(self.chain)
		self.__last_coinc = None  # the last coinc seen


	def do_set_property(self, prop, val):
		#if prop.name == 'min-ifar':
		#	self.__min_stat = val
		if prop.name == 'min-combined-eff-snr':
			self.__min_stat = val
		elif prop.name == 'min-waiting-time':
			self.__min_waiting_time = val
		#elif prop.name == 'dt':
		#	self.__dt = val


	def do_get_property(self, prop):
		#if prop.name == 'min-ifar':
		#	return self.__min_stat
		if prop.name == 'min-combined-eff-snr':
			return self.__min_stat
		elif prop.name == 'min-waiting-time':
			return self.__min_waiting_time
		#elif prop.name == 'dt':
		#	return self.__dt

	def chain(self, pad, inbuf):
		try: # FIXME: apparently the gst.Pad wrapper silences exceptions from chain() routines.
			# Eligible coincs include __last_coinc and new coincs over threshold.
			coincs = []
			if self.__last_coinc is not None:
				coincs = [self.__last_coinc]
			for sngl_group in sngl_inspiral_groups_from_buffer(inbuf):
				# FIXME: switch back to network IFAR
				# stat = net_ifar((float(gst.SECOND) / row.alpha for row in group), float(self.__dt))
				stat = combined_effective_snr(sngl_group)
				if stat > self.__min_stat:
					coincs.append(SnglCoinc(sngl_group, stat))
			coincs.sort(key=operator.attrgetter("time"))

			# Cluster
			clustered_coincs = cluster_coincs(coincs, self.__min_waiting_time)

			# The last coinc may not be definitively clustered yet.
			if (len(clustered_coincs) > 0) and (inbuf.timestamp + inbuf.duration < clustered_coincs[-1].time + self.__min_waiting_time):
				self.__last_coinc = clustered_coincs.pop()
			else:
				self.__last_coinc = None

			# Push clusters.
			outbuf = sngl_inspiral_groups_to_buffer([coinc.sngl_group for coinc in clustered_coincs], inbuf.caps[0]['channels'])
			outbuf.timestamp = inbuf.timestamp
			outbuf.duration = inbuf.duration
			outbuf.offset = gst.BUFFER_OFFSET_NONE
			outbuf.offset_end = gst.BUFFER_OFFSET_NONE
			outbuf.caps = inbuf.caps
			return self.__srcpad.push(outbuf)
		except:
			self.error(traceback.format_exc())
			return gst.FLOW_ERROR

# Register element class
gstlal_element_register(lal_coincselector)

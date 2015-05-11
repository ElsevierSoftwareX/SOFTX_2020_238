# Copyright (C) 2010  Leo Singer
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
Fake trigger generator
"""
__author__ = "Leo Singer <leo.singer@ligo.org>"


from gstlal.pipeutil import *
from gst.extend.pygobject import gproperty, with_construct_properties
from random import randint
from collections import deque
from pylal.xlal.datatypes.snglinspiraltable import SnglInspiralTable
from glue.ligolw import lsctables
from glue.ligolw import utils
import lal


def sngl_inspiral_pylal_from_glue(glue_sngl):
	pylal_sngl = SnglInspiralTable()
	for key in glue_sngl.__slots__:
		setattr(pylal_sngl, key, getattr(glue_sngl, key))
	return pylal_sngl


class lal_faketriggersrc(gst.BaseSrc):

	__gstdetails__ = (
		"Fake trigger source",
		"Source",
		__doc__.strip(),
		__author__
	)
	gproperty(
		gobject.TYPE_STRING,
		"instrument",
		'Instrument name (e.g., "H1")',
		None,
		construct=True # FIXME if gst.extend.pygobject provided gst.PARAM_MUTABLE_READY it would be a good idea to set this here
	)
	gproperty(
		gobject.TYPE_STRING,
		"xml-location",
		"Name of LIGO Light Weight XML file containing list of templates",
		None,
		construct=True # FIXME if gst.extend.pygobject provided gst.PARAM_MUTABLE_READY it would be a good idea to set this here
	)
	gproperty(
		gobject.TYPE_UINT64,
		"min-waiting-time",
		"Minimum waiting time between per-template triggers (nanoseconds)",
		0, gst.CLOCK_TIME_NONE, 500 * gst.MSECOND,
		construct=True # FIXME if gst.extend.pygobject provided gst.PARAM_MUTABLE_READY it would be a good idea to set this here
	)
	gproperty(
		gobject.TYPE_UINT64,
		"max-waiting-time",
		"Minimum waiting time between per-template triggers (nanoseconds)",
		0, gst.CLOCK_TIME_NONE, 1 * gst.SECOND,
		construct=True # FIXME if gst.extend.pygobject provided gst.PARAM_MUTABLE_READY it would be a good idea to set this here
	)
	gproperty(
		gobject.TYPE_UINT64,
		"buffer-duration",
		"Duration of each buffer (nanoseconds)",
		1, gst.CLOCK_TIME_NONE, 4 * gst.SECOND,
		construct=True # FIXME if gst.extend.pygobject provided gst.PARAM_MUTABLE_READY it would be a good idea to set this here
	)
	gproperty(
		gobject.TYPE_UINT64,
		"start-time",
		"Time from which to start playback (nanoseconds)",
		0, gst.CLOCK_TIME_NONE, 0,
		construct=True # FIXME if gst.extend.pygobject provided gst.PARAM_MUTABLE_READY it would be a good idea to set this here
	)
	gproperty(
		gobject.TYPE_UINT64,
		"duration",
		"Duration for which to produce triggers (nanoseconds)",
		0, gst.CLOCK_TIME_NONE, 20 * gst.SECOND,
		construct=True # FIXME if gst.extend.pygobject provided gst.PARAM_MUTABLE_READY it would be a good idea to set this here
	)
	__gsttemplates__ = (
		gst.PadTemplate("src",
			gst.PAD_SRC, gst.PAD_ALWAYS,
			gst.caps_from_string("""
				application/x-lal-snglinspiral,
				channels = (int) 1
			""")
		),
	)


	@staticmethod
	def make_trigger_times(start_time, end_time, min_waiting_time, max_waiting_time):
		times = deque()
		t = start_time
		while t < end_time:
			t += randint(min_waiting_time, max_waiting_time)
			times.append(t)
		return times


	def __init__(self):
		super(lal_faketriggersrc, self).__init__()
		self.set_do_timestamp(False)
		self.set_format(gst.FORMAT_TIME)
		self.src_pads().next().use_fixed_caps()
		for prop in self.props:
			self.set_property(prop.name, prop.default_value)


	def do_start(self):
		"""GstBaseSrc->start virtual method"""

		xml_location = self.get_property("xml-location")
		if xml_location is None:
			self.error("xml-location property is unset, cannot load template bank")
			return False

		self.__templates = list(
			lsctables.SnglInspiralTable.get_table(
				utils.load_filename(xml_location)
			)
		)

		start_time = self.get_property("start-time")
		duration = self.get_property("duration")
		end_time = start_time + duration
		min_waiting_time = self.get_property("min-waiting-time")
		max_waiting_time = self.get_property("max-waiting-time")

		self.__last_time = start_time
		self.__stream_end_time = end_time
		self.__triggertimes = [
			self.make_trigger_times(start_time, end_time, min_waiting_time, max_waiting_time)
			for i in range(len(self.__templates))
		]
		self.__ntriggers = 0

		#self.src_pads().next().push_event(gst.event_new_new_segment(False, 1.0, gst.FORMAT_TIME, start_time, end_time, start_time))

		return True


	def do_stop(self):
		"""GstBaseSrc->stop virtual method"""
		self.__templates = None
		self.__triggertimes = None
		return True


	def do_check_get_range(self):
		"""GstBaseSrc->check_get_range virtual method"""
		return True


	def do_is_seekable(self):
		"""GstBaseSrc->is_seekable virtual method"""
		return False


	def do_create(self, offset, size):
		"""GstBaseSrc->create virtual method"""

		instrument = self.get_property("instrument")
		if instrument is None:
			self.error("instrument property is unset")
			return (gst.FLOW_ERROR, None)

		buffer_duration = self.get_property("buffer-duration")
		timestamp = self.__last_time

		if timestamp >= self.__stream_end_time:
			gst.info('timestamp %d is greater than stream end time %d, sending EOS' % (timestamp, self.__stream_end_time))
			self.src_pads().next().push_event(gst.event_new_eos())
			return (gst.FLOW_UNEXPECTED, None)

		end_time = timestamp + buffer_duration
		if end_time > self.__stream_end_time:
			end_time = self.__stream_end_time
			buffer_duration = end_time - timestamp


		# Construct triggers for all templates
		triggers_by_template = []
		for template, triggertimes in zip(self.__templates, self.__triggertimes):
			new_triggers = deque()
			while len(triggertimes) > 0 and triggertimes[0] < end_time:
				triggertime = triggertimes.popleft()
				triggertime = lal.LIGOTimeGPS(triggertime / gst.SECOND, triggertime % gst.SECOND)
				sngl = sngl_inspiral_pylal_from_glue(template)
				sngl.ifo = instrument
				sngl.channel = "FAKE_TRIGGER"
				sngl.end_time = triggertime.gpsSeconds
				sngl.end_time_ns = triggertime.gpsNanoSeconds
				sngl.end_time_gmst = lal.GreenwichMeanSiderealTime(triggertime)
				sngl.snr = 5
				sngl.coa_phase = 0
				sngl.chisq = 1
				sngl.chisq_dof = 1
				sngl.sigmasq = 1.0
				new_triggers.append(sngl)
			if len(new_triggers) > 0:
				triggers_by_template.append(new_triggers)

		ntriggers = sum([len(x) for x in triggers_by_template])
		recordsize = len(buffer(SnglInspiralTable()))
		gst.info('emitting %d triggers for %d templates' % (ntriggers, len(self.__templates)))

		bufsize = len(buffer(SnglInspiralTable())) * ntriggers
		pad = self.src_pads().next()
		(retval, buf) = pad.alloc_buffer(0, bufsize, pad.get_property("caps")) # TODO set offset

		if retval != gst.FLOW_OK:
			return (retval, None)

		bufidx = 0
		# Randomly rearrange triggers
		while len(triggers_by_template) > 0:
			idx = randint(0, len(triggers_by_template)-1)
			template = triggers_by_template[idx].popleft()
			if len(triggers_by_template[idx]) == 0:
				del triggers_by_template[idx]
			buf[bufidx:bufidx+recordsize] = buffer(template)
			bufidx += recordsize

		# TODO set buffer offset and offset-end
		buf.timestamp = timestamp
		buf.duration = buffer_duration
		buf.offset = self.__ntriggers
		self.__ntriggers += ntriggers
		buf.offset_end = self.__ntriggers

		self.__last_time = end_time
		return (gst.FLOW_OK, buf)


# Register element class
gstlal_element_register(lal_faketriggersrc)

from glue.ligolw import ilwd
from glue.ligolw import lsctables
import lal
from gstlal import _snglbursttable


__all__ = ["GSTLALSnglBurst"]


class GSTLALSnglBurst(_snglbursttable.GSTLALSnglBurst):
	__slots__ = ()

	process_id_type = ilwd.get_ilwdchar_class("process", "process_id")
	event_id_type = ilwd.get_ilwdchar_class("sngl_burst", "event_id")

	start = lsctables.gpsproperty("start_time", "start_time_ns")
	peak = lsctables.gpsproperty("peak_time", "peak_time_ns")

	@property
	def process_id(self):
		return self.process_id_type(self._process_id)

	@process_id.setter
	def process_id(self, val):
		self._process_id = int(val)

	@property
	def event_id(self):
		return self.event_id_type(self._event_id)

	@event_id.setter
	def event_id(self, val):
		self._event_id = int(val)

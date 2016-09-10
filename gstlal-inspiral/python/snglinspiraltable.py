from glue.ligolw import ilwd
from glue.ligolw import lsctables
import lal
from . import _snglinspiraltable


__all__ = ["GSTLALSnglInspiral"]


class GSTLALSnglInspiral(_snglinspiraltable.GSTLALSnglInspiral):
	__slots__ = ()

	process_id_type = ilwd.get_ilwdchar_class("process", "process_id")
	event_id_type = ilwd.get_ilwdchar_class("sngl_inspiral", "event_id")

	spin1 = lsctables.SnglInspiral.spin1
	spin2 = lsctables.SnglInspiral.spin2
	__eq__ = lsctables.SnglInspiral.__eq__

	@property
	def end(self):
		return lal.LIGOTimeGPS(self.end_time, self.end_time_ns)

	@end.setter
	def end(self, val):
		if val is None:
			self.end_time = self.end_time_ns = 0
		else:
			self.end_time = val.gpsSeconds
			self.end_time_ns = val.gpsNanoSeconds

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

	@property
	def snr(self):
		try:
			name = self._snr_name
		except ValueError:
			# C interface raises ValueError if the internal snr
			# pointer is NULL
			return None
		series = lal.CreateCOMPLEX16TimeSeries(
			name,
			lal.LIGOTimeGPS(self._snr_epoch_gpsSeconds, self._snr_epoch_gpsNanoSeconds),
			self._snr_f0,
			self._snr_deltaT,
			lal.Unit(self._snr_sampleUnits),
			self._snr_data_length
		)
		series.data.data = self._snr_data
		return series

from glue.ligolw import ilwd
from glue.ligolw import lsctables
import lal
from . import _snglinspiraltable


__all__ = ["GSTLALSnglInspiral"]


class GSTLALSnglInspiral(_snglinspiraltable.GSTLALSnglInspiral):
	__slots__ = ()

	process_id_type = ilwd.get_ilwdchar_class("process", "process_id")
	event_id_type = ilwd.get_ilwdchar_class("sngl_inspiral", "event_id")

	end = lsctables.gpsproperty("end_time", "end_time_ns")
	spin1 = lsctables.SnglInspiral.spin1
	spin2 = lsctables.SnglInspiral.spin2

	def __eq__(self, other):
		return not cmp(
			(self.ifo, self.end, self.mass1, self.mass2, self.spin1, self.spin2, self.search),
			(other.ifo, other.end, other.mass1, other.mass2, other.spin1, other.spin2, other.search)
		)

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
	def snr_time_series(self):
		try:
			name = self._snr_name
		except ValueError:
			# C interface raises ValueError if the internal snr
			# pointer is NULL
			return None
		series = lal.CreateCOMPLEX8TimeSeries(
			name,
			lal.LIGOTimeGPS(self._snr_epoch_gpsSeconds, self._snr_epoch_gpsNanoSeconds),
			self._snr_f0,
			self._snr_deltaT,
			lal.Unit(self._snr_sampleUnits),
			self._snr_data_length
		)
		# we want to be able to keep the table row object in memory
		# for an extended period of time so we need to be able to
		# release the memory used by the SNR time series when we no
		# longer need it, and so we copy the data here instead of
		# holding a reference to the original memory.  if we
		# allowed references to the original memory to leak out
		# into Python land we could never know if it's safe to free
		# it
		series.data.data[:] = self._snr_data
		return series

	@snr_time_series.deleter
	def snr_time_series(self):
		self._snr_time_series_deleter()

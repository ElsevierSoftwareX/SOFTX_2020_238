from ligo.lw import lsctables
import lal
from gstlal import _sngltriggertable


__all__ = ["GSTLALSnglTrigger"]


class GSTLALSnglTrigger(_sngltriggertable.GSTLALSnglTrigger):
	__slots__ = ()

#	def __eq__(self, other):
#		return not cmp(
#			(self.ifo, self.end, self.mass1, self.mass2, self.spin1, self.spin2, self.search),
#			(other.ifo, other.end, other.mass1, other.mass2, other.spin1, other.spin2, other.search)
#		)

	def __cmp__(self, other):
		return cmp(self.end, other)

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

# Copyright (C) 2016,2017  Kipp Cannon, Leo Singer
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


from ligo.lw import lsctables
import lal
from . import _snglinspiraltable


__all__ = ["GSTLALSnglInspiral"]


class GSTLALSnglInspiral(_snglinspiraltable.GSTLALSnglInspiral):
	__slots__ = ()

	spin1 = lsctables.SnglInspiral.spin1
	spin2 = lsctables.SnglInspiral.spin2

	@property
	def G1_snr_time_series(self):
		try:
			name = self._G1_snr_name
		except ValueError:
			# C interface raises ValueError if the internal snr
			# pointer is NULL
			return None
		series = lal.CreateCOMPLEX8TimeSeries(
			name,
			lal.LIGOTimeGPS(self._G1_snr_epoch_gpsSeconds, self._G1_snr_epoch_gpsNanoSeconds),
			self._G1_snr_f0,
			self._G1_snr_deltaT,
			lal.Unit(self._G1_snr_sampleUnits),
			self._G1_snr_data_length
		)
		# we want to be able to keep the table row object in memory
		# for an extended period of time so we need to be able to
		# release the memory used by the SNR time series when we no
		# longer need it, and so we copy the data here instead of
		# holding a reference to the original memory.  if we
		# allowed references to the original memory to leak out
		# into Python land we could never know if it's safe to free
		# it
		series.data.data[:] = self._G1_snr_data
		return series

	@property
	def H1_snr_time_series(self):
		try:
			name = self._H1_snr_name
		except ValueError:
			# C interface raises ValueError if the internal snr
			# pointer is NULL
			return None
		series = lal.CreateCOMPLEX8TimeSeries(
			name,
			lal.LIGOTimeGPS(self._H1_snr_epoch_gpsSeconds, self._H1_snr_epoch_gpsNanoSeconds),
			self._H1_snr_f0,
			self._H1_snr_deltaT,
			lal.Unit(self._H1_snr_sampleUnits),
			self._H1_snr_data_length
		)
		# we want to be able to keep the table row object in memory
		# for an extended period of time so we need to be able to
		# release the memory used by the SNR time series when we no
		# longer need it, and so we copy the data here instead of
		# holding a reference to the original memory.  if we
		# allowed references to the original memory to leak out
		# into Python land we could never know if it's safe to free
		# it
		series.data.data[:] = self._H1_snr_data
		return series

	@property
	def L1_snr_time_series(self):
		try:
			name = self._L1_snr_name
		except ValueError:
			# C interface raises ValueError if the internal snr
			# pointer is NULL
			return None
		series = lal.CreateCOMPLEX8TimeSeries(
			name,
			lal.LIGOTimeGPS(self._L1_snr_epoch_gpsSeconds, self._L1_snr_epoch_gpsNanoSeconds),
			self._L1_snr_f0,
			self._L1_snr_deltaT,
			lal.Unit(self._L1_snr_sampleUnits),
			self._L1_snr_data_length
		)
		# we want to be able to keep the table row object in memory
		# for an extended period of time so we need to be able to
		# release the memory used by the SNR time series when we no
		# longer need it, and so we copy the data here instead of
		# holding a reference to the original memory.  if we
		# allowed references to the original memory to leak out
		# into Python land we could never know if it's safe to free
		# it
		series.data.data[:] = self._L1_snr_data
		return series

	@property
	def V1_snr_time_series(self):
		try:
			name = self._V1_snr_name
		except ValueError:
			# C interface raises ValueError if the internal snr
			# pointer is NULL
			return None
		series = lal.CreateCOMPLEX8TimeSeries(
			name,
			lal.LIGOTimeGPS(self._V1_snr_epoch_gpsSeconds, self._V1_snr_epoch_gpsNanoSeconds),
			self._V1_snr_f0,
			self._V1_snr_deltaT,
			lal.Unit(self._V1_snr_sampleUnits),
			self._V1_snr_data_length
		)
		# we want to be able to keep the table row object in memory
		# for an extended period of time so we need to be able to
		# release the memory used by the SNR time series when we no
		# longer need it, and so we copy the data here instead of
		# holding a reference to the original memory.  if we
		# allowed references to the original memory to leak out
		# into Python land we could never know if it's safe to free
		# it
		series.data.data[:] = self._V1_snr_data
		return series

	@property
	def K1_snr_time_series(self):
		try:
			name = self._K1_snr_name
		except ValueError:
			# C interface raises ValueError if the internal snr
			# pointer is NULL
			return None
		series = lal.CreateCOMPLEX8TimeSeries(
			name,
			lal.LIGOTimeGPS(self._K1_snr_epoch_gpsSeconds, self._K1_snr_epoch_gpsNanoSeconds),
			self._K1_snr_f0,
			self._K1_snr_deltaT,
			lal.Unit(self._K1_snr_sampleUnits),
			self._K1_snr_data_length
		)
		# we want to be able to keep the table row object in memory
		# for an extended period of time so we need to be able to
		# release the memory used by the SNR time series when we no
		# longer need it, and so we copy the data here instead of
		# holding a reference to the original memory.  if we
		# allowed references to the original memory to leak out
		# into Python land we could never know if it's safe to free
		# it
		series.data.data[:] = self._K1_snr_data
		return series

	@G1_snr_time_series.deleter
	def G1_snr_time_series(self):
		self._G1_snr_time_series_deleter()

	@H1_snr_time_series.deleter
	def H1_snr_time_series(self):
		self._H1_snr_time_series_deleter()

	@L1_snr_time_series.deleter
	def L1_snr_time_series(self):
		self._L1_snr_time_series_deleter()

	@V1_snr_time_series.deleter
	def V1_snr_time_series(self):
		self._V1_snr_time_series_deleter()

	@K1_snr_time_series.deleter
	def K1_snr_time_series(self):
		self._K1_snr_time_series_deleter()

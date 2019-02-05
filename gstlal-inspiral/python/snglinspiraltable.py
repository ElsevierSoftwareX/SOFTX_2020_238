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

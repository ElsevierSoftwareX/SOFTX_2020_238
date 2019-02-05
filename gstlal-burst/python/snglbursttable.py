from ligo.lw import lsctables
import lal
from gstlal import _snglbursttable


__all__ = ["GSTLALSnglBurst"]


class GSTLALSnglBurst(_snglbursttable.GSTLALSnglBurst):
	__slots__ = ()

	start = lsctables.gpsproperty("start_time", "start_time_ns")
	peak = lsctables.gpsproperty("peak_time", "peak_time_ns")

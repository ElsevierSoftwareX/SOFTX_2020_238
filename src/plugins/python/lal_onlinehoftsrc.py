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

Online calibrated h(t) source, following conventions established in S6.

The LIGO online frames are described at
<https://www.lsc-group.phys.uwm.edu/daswg/wiki/S6OnlineGroup/CalibratedData>.

The Virgo online frames are described at
<https://workarea.ego-gw.it/ego2/virgo/data-analysis/calibration-reconstruction/online-h-t-reconstruction-hrec>.

The environment variable ONLINEHOFT must be set and must point to the online
frames directory, which has subfolders for H1, H2, L1, V1, ... .

Online frames are 16 seconds in duration, and start on 16 second boundaries.
They contain up to three channels:
 - IFO:DMT-STRAIN (16384 Hz), online calibrated h(t)
 - IFO:DMT-STATE_VECTOR (16 Hz), state vector
 - IFO:DMT-DATA_QUALITY_VECTOR (1 Hz), data quality flags

This element features user-programmable data vetos at 1 second resolution.
Gaps (GStreamer buffers marked as containing neutral data) will be created
whenever the state vector mask and data quality mask flag properties are
not met.

Gaps will also be created whenever an anticipated frame file is missing.

"""
__author__ = "Leo Singer <leo.singer@ligo.org>"
__version__ = "FIXME"
__date__ = "FIXME"
__all__ = ("lal_onlinehoftsrc", "directory_poller")


import errno
import os
import os.path
import sys
import time
import bisect
from _onlinehoftsrc import *
try:
	from collections import namedtuple
except:
	# Pre-Python 2.6 compatibility.
	from gstlal.namedtuple import namedtuple
from gstlal.pipeutil import *


def gps_now():
	import pylal.xlal.date
	return int(pylal.xlal.date.XLALUTCToGPS(time.gmtime()))


def safe_getvect(filename, channel, start, duration, fs):
	"""Ultra-paranoid frame reading function."""
	from pylal.Fr import frgetvect1d
	vect_data, vect_start, vect_x0, vect_df, vect_unit_x, vect_unit_y = frgetvect1d(filename, channel, start, duration)
	if vect_start != start:
		raise ValueError, "channel %s: expected start time %d, but got %f" % (channel, start, vec_start)
	if vect_x0 != 0:
		raise ValueError, "channel %s: expected offset 0, but got %f" % (channel, vect_x0)
	if vect_df != 1.0 / fs:
		raise ValueError, "channel %s: expected sample rate %d, but got %f" % (channel, fs, 1.0 / vect_df)
	if len(vect_data) != duration * fs:
		raise ValueError, "channel %s: expected %d samples, but got %d" % (channel, duration * fs, len(vect_data))
	return vect_data


class dir_cache(object):
	def __init__(self, path, expires):
		self.path = path
		self.expires = expires
		self.mtime = 0
		self.refresh()
	def refresh(self):
		old_mtime = self.mtime
		self.mtime = os.stat(self.path)[8]
		if self.path is None or self.mtime != old_mtime:
			self.items = self.decorate(os.listdir(self.path))


class dir_cache_top(dir_cache):
	def __init__(self, top, nameprefix):
		self.nameprefix = nameprefix
		super(dir_cache_top, self).__init__(top, 0)
	def decorate(self, filenames):
		items = []
		for filename in filenames:
			if filename != 'latest':
				if filename.startswith(self.nameprefix):
					try:
						item = int(filename[len(self.nameprefix):])
					except:
						print >>sys.stderr, "lal_onlinehoftsrc: %s: invalid epoch name" % filename
					else:
						items.append(item)
				else:
					print >>sys.stderr, "lal_onlinehoftsrc: %s: invalid epoch name" % filename
		return sorted(items)


class dir_cache_epoch(dir_cache):
	def __init__(self, top, nameprefix, namesuffix, epoch):
		self.nameprefix = nameprefix
		self.namesuffix = namesuffix
		path = os.path.join(top, "%s%u" % (nameprefix, epoch))
		super(dir_cache_epoch, self).__init__(path, epoch * 100000)
	def decorate(self, filenames):
		items = []
		for filename in filenames:
			if filename.startswith(self.nameprefix) and filename.endswith(self.namesuffix):
				try:
					item = int(filename[len(self.nameprefix):-len(self.namesuffix)])
				except:
					print >>sys.stderr, "lal_onlinehoftsrc: %s: invalid file name" % filename
				else:
					items.append(item)
			else:
				print >>sys.stderr, "lal_onlinehoftsrc: %s: invalid file name" % filename
		return sorted(items)


class directory_poller(object):
	"""Iterate over file descriptors from a directory tree of GPS-timestamped
	files, like the $ONLINEHOFT or $ONLINEDQ directories on LSC clusters.
	"""

	def __init__(self, top, nameprefix, namesuffix):
		self.top = top
		self.nameprefix = nameprefix
		self.namesuffix = namesuffix
		self.__time = 0
		self.stride = 16
		self.latency = 70
		self.timeout = 2
		self.top_cache = None
		self.epoch_caches = {}


	def get_time(self):
		return self.__time
	def set_time(self, time):
		for key in self.epoch_caches.keys():
			if self.epoch_caches[key].expires < time:
				del self.epoch_caches[key]
		self.__time = time
	time = property(get_time, set_time)


	def __iter__(self):
		return self


	def next(self):
		while True:
			epoch = self.time / 100000
			epoch_path = os.path.join(self.top, "%s%u" % (self.nameprefix, epoch))
			filename = "%s%d%s" % (self.nameprefix, self.time, self.namesuffix)
			filepath = os.path.join(epoch_path, filename)

			try:
				# Attempt to open the file.
				fd = os.open(filepath, os.O_RDONLY)
			except OSError, (err, strerror):
				# Opening the file failed.
				if err == errno.ENOENT:
					# Opening the file failed because it did not exist.
					if gps_now() - self.time < self.latency:
						# The requested time is too recent, so just wait
						# a bit and then try again.
						gst.warning("lal_onlinehoftsrc: sleeping because requested time is too new")
						time.sleep(self.timeout)
					else:
						# The requested time is old enough that it is possible that
						# there is a missing file.  Look through the directory tree
						# to find the next available file.

						gst.warning("lal_onlinehoftsrc: %s: late or missing file suspected" % filepath)

						# We need to scan the directory tree successfully twice
						# in succession to avoid a race condition where the
						# frame builder writes files faster than we move through
						# the directory tree.
						num_tries_remaining = 2
						while num_tries_remaining > 0:

							# Try to get the top level directory listing,
							# and refresh it if its mtime has changed.
							try:
								if self.top_cache is None:
									self.top_cache = dir_cache_top(self.top, self.nameprefix)
								else:
									self.top_cache.refresh()
							except OSError, (err, strerror):
								# We couldn't read the top level directory.
								# This is very bad, so let's complain about
								# it, sleep for a moment, and then go back
								# to the outer loop.
								print >>sys.stderr, "lal_onlinehoftsrc: %s: %s" % (self.top, strerror)
								time.sleep(self.timeout)
								break

							# Loop over the epochs until we find a file that
							# is at least as new as the anticipated GPS time.
							new_file_found = False
							for other_epoch in self.top_cache.items[bisect.bisect_left(self.top_cache.items, epoch):]:
								if other_epoch not in self.epoch_caches.keys():
									try:
										cache = dir_cache_epoch(self.top, self.nameprefix, self.namesuffix, other_epoch)
									except OSError, (err, strerror):
										print >>sys.stderr, "lal_onlinehoftsrc: %s: %s" % (self.top, strerror)
										continue
									self.epoch_caches[other_epoch] = cache
								else:
									cache = self.epoch_caches[other_epoch]
									try:
										cache.refresh()
									except OSError, (err, strerror):
										print >>sys.stderr, "lal_onlinehoftsrc: %s: %s" % (self.top, strerror)
										continue
								idx = bisect.bisect_left(cache.items, self.time)
								if idx < len(cache.items):
									if self.time == cache.items[idx]:
										# The file that we wanted has just appeared,
										# so we can try to read it right away.
										# Go back to outer loop.
										num_tries_remaining = 0
									else:
										num_tries_remaining -= 1
										if num_tries_remaining == 0:
											# We have found a new file a second time,
											# so go back to outer loop.
											gst.warning("lal_onlinehoftsrc: files skipped")
											self.time = cache.items[idx]
									# Go back to outer loop.
									new_file_found = True
									break
							if not new_file_found:
								num_tries_remaining = 2
								gst.warning("lal_onlinehoftsrc: files are very late")
								time.sleep(self.timeout)
				else:
					# Opening file failed for some reason *other* than that it did
					# not exist, so we assume that we will never be able to open it.
					# Print an error message and try the next file.
					self.time += self.stride
					print >>sys.stderr, "lal_onlinehoftsrc: %s: %s" % (filepath, strerror)
			else:
				# Opening the file succeeded, so return the new file descriptor.
				self.time += self.stride
				return ((self.time - self.stride), fd)


ifodesc = namedtuple("ifodesc", "ifo nameprefix namesuffix channelname state_channelname dq_channelname")

ifodescs = {
	"H1": ifodesc("H1", "H-H1_DMT_C00_L2-", "-16.gwf", "H1:DMT-STRAIN", "H1:DMT-STATE_VECTOR", "H1:DMT-DATA_QUALITY_VECTOR"),
	"H2": ifodesc("H2", "H-H2_DMT_C00_L2-", "-16.gwf", "H2:DMT-STRAIN", "H2:DMT-STATE_VECTOR", "H2:DMT-DATA_QUALITY_VECTOR"),
	"L1": ifodesc("L1", "L-L1_DMT_C00_L2-", "-16.gwf", "L1:DMT-STRAIN", "L1:DMT-STATE_VECTOR", "L1:DMT-DATA_QUALITY_VECTOR"),
	"V1": ifodesc("V1", "V-V1_DMT_HREC-", "-16.gwf", "V1:h_16384Hz", None, "V1:Hrec_Veto_dataQuality")
}


class lal_onlinehoftsrc(gst.BaseSrc):

	__gstdetails__ = (
		"Online h(t) Source",
		"Source",
		__doc__,
		__author__
	)
	__gproperties__ = {
		"instrument": (
			gobject.TYPE_STRING,
			"instrument",
			'Instrument name (e.g., "H1")',
			None,
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		),
		"state-require": (
			StateFlags,
			"state-require",
			"State vector flags that must be TRUE",
			STATE_SCI | STATE_CON | STATE_UP | STATE_EXC,
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		),
		"state-deny": (
			StateFlags,
			"state-deny",
			"State vector flags that must be FALSE",
			0,
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		),
		"data-quality-require": (
			DQFlags,
			"data-quality-require",
			"Data quality flags that must be TRUE",
			DQ_SCIENCE | DQ_UP | DQ_CALIBRATED | DQ_LIGHT,
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		),
		"data-quality-deny": (
			DQFlags,
			"data-quality-deny",
			"Data quality flags that must be FALSE",
			DQ_BADGAMMA,
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		),
		"virgo-data-quality": (
			VirgoDQFlags,
			"virgo-data-quality",
			"Data quality value that must be present in Virgo data",
			VIRGO_DQ_12,
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		),
		"is-live": (
			gobject.TYPE_BOOLEAN,
			"is-live",
			"Whether to act as a live source, starting playback from current GPS time",
			False,
			gobject.PARAM_READWRITE | gobject.PARAM_CONSTRUCT
		)
	}
	__gsttemplates__ = (
		gst.PadTemplate("src",
			gst.PAD_SRC, gst.PAD_ALWAYS,
			gst.caps_from_string("""
				audio/x-raw-float,
				channels = (int) 1,
				endianness = (int) BYTE_ORDER,
				width = (int) {32, 64},
				rate = (int) 16384
			""")
		),
	)


	__float32_caps = gst.caps_from_string("""
		audio/x-raw-float,
		channels = (int) 1,
		endianness = (int) BYTE_ORDER,
		width = (int) 32,
		rate = (int) 16384
	""")


	__float64_caps = gst.caps_from_string("""
		audio/x-raw-float,
		channels = (int) 1,
		endianness = (int) BYTE_ORDER,
		width = (int) 64,
		rate = (int) 16384
	""")


	def __init__(self):
		super(lal_onlinehoftsrc, self).__init__()
		self.set_property('blocksize', 16384 * 16 * 8)
		self.set_do_timestamp(False)
		self.set_format(gst.FORMAT_TIME)
		self.src_pads().next().use_fixed_caps()
		self.__needs_seek = False


	def do_set_property(self, prop, val):
		setattr(self, '_' + prop.name.replace('-', '_'), val)
		if prop.name == 'is-live':
			self.set_live(val)
			if val:
				self.seek(1.0, gst.FORMAT_TIME, gst.SEEK_FLAG_KEY_UNIT,
						gst.SEEK_TYPE_SET, (gps_now() - 70) * gst.SECOND,
						gst.SEEK_TYPE_NONE, -1)


	def do_get_property(self, prop):
		return getattr(self, '_' + prop.name.replace('-', '_'))


	def do_start(self):
		"""GstBaseSrc->start virtual method"""
		self.__last_successful_gps_end = None

		# Look up instrument
		if self._instrument not in ifodescs:
			self.error("unknown instrument: %s" % self._instrument)
			return False
		self.__ifodesc = ifodescs[self._instrument]

		# Create instance of directory_poller
		self.__poller = directory_poller(
			os.path.join(os.getenv('ONLINEHOFT'), self._instrument),
			self.__ifodesc.nameprefix, self.__ifodesc.namesuffix
		)

		# Send tags
		taglist = gst.TagList()
		taglist["instrument"] = self._instrument
		taglist["channel-name"] = self.__ifodesc.channelname.split(":")[-1]
		taglist["units"] = "strain" # FIXME: can we get this value from the frame file itself?

		if not self.send_event(gst.event_new_tag(taglist)):
			self.error("tags rejected")
			return False

		# Done! OK to start playing now.
		return True


	def do_stop(self):
		"""GstBaseSrc->stop virtual method"""
		self.__ifodesc = None
		self.__poller = None
		self.__needs_seek = False
		return True


	def do_check_get_range(self):
		"""GstBaseSrc->check_get_range virtual method"""
		return True


	def do_is_seekable(self):
		"""GstBaseSrc->is_seekable virtual method"""
		return not(self._is_live)


	def do_do_seek(self, segment):
		"""GstBaseSrc->do_seek virtual method"""
		if segment.flags & gst.SEEK_FLAG_KEY_UNIT:
			# If necessary, extend the segment to the nearest "key frame",
			# playback can only start or stop on boundaries of 16 seconds.
			if segment.start == -1:
				start = -1
				start_seek_type = gst.SEEK_TYPE_NONE
			else:
				start = gst.util_uint64_scale(gst.util_uint64_scale(segment.start, 1, 16 * gst.SECOND), 16 * gst.SECOND, 1)
				start_seek_type = gst.SEEK_TYPE_SET
			if segment.stop == -1:
				stop = -1
				stop_seek_type = gst.SEEK_TYPE_NONE
			else:
				stop = gst.util_uint64_scale_ceil(gst.util_uint64_scale_ceil(segment.stop, 1, 16 * gst.SECOND), 16 * gst.SECOND, 1)
				stop_seek_type = gst.SEEK_TYPE_SET
			segment.set_seek(segment.rate, segment.format, segment.flags, start_seek_type, start, stop_seek_type, stop)
		self.__seek_time = (segment.start / gst.SECOND / 16) * 16
		self.__needs_seek = True
		return True


	def do_query(self, query):
		"""GstBaseSrc->query virtual method"""

		if query.type == gst.QUERY_FORMATS:
			query.set_formats(gst.FORMAT_DEFAULT, gst.FORMAT_TIME)
			return True
		elif query.type == gst.QUERY_CONVERT:
			src_format, src_value, dest_format, dest_value = query.parse_convert()
			if src_format not in (gst.FORMAT_DEFAULT, gst.FORMAT_TIME):
				return False
			if dest_format not in (gst.FORMAT_DEFAULT, gst.FORMAT_TIME):
				return False
			dest_value = src_value
			query.set_convert(src_format, src_value, dest_format, dest_value)
			return True
		else:
			return gst.BaseSrc.do_query(self, query)


	def do_create(self, offset, size):
		"""GstBaseSrc->create virtual method"""

		# Seek if needed.
		if self.__needs_seek:
			self.__poller.time = self.__seek_time
			self.__needs_seek = False
			self.__last_successful_gps_end = None

		# Loop over available buffers until we reach one that is not corrupted.
		for (gps_start, fd) in self.__poller:
			# FIXME: Merge this try-finally and try-except-else block into a
			# single try-except-else-finally block.  Unified try-except-finally
			# statements were added in Python 2.5 (see PEP 341).  CentOS still
			# ships with Python 2.4.
			try:
				try:
					filename = "/dev/fd/%d" % fd
					hoft_array = safe_getvect(filename, self.__ifodesc.channelname, gps_start, 16, 16384)
					if self.__ifodesc.ifo != 'V1':
						os.lseek(fd, 0, 0) # FIXME: use os.SEEK_SET (added to API in Python 2.5) for last argument
						state_array = safe_getvect(filename, self.__ifodesc.state_channelname, gps_start, 16, 16)
					os.lseek(fd, 0, 0) # FIXME: use os.SEEK_SET (added to API in Python 2.5) for last argument
					dq_array = safe_getvect(filename, self.__ifodesc.dq_channelname, gps_start, 16, 1)
				finally:
					os.close(fd)
			except Exception, e:
				self.warning(str(e))
			else:
				break

		# Look up our src pad and its caps.
		pad = self.src_pads().next()
		if hoft_array.dtype.name == 'float64':
			caps = self.__float64_caps
		elif hoft_array.dtype.name == 'float32':
			caps = self.__float32_caps
		else:
			self.error("h(t) channel has unrecognized dtype %s" % hoft_array.dtype.name)
			return (gst.FLOW_ERROR, None)

		# Compute "good data" segment mask.
		if self.__ifodesc.ifo == "V1":
			segment_mask = (dq_array >= int(self._virgo_data_quality))
		else:
			dq_require = int(self._data_quality_require)
			dq_deny = int(self._data_quality_deny)
			state_require = int(self._state_require)
			state_deny = int(self._state_deny)
			state_array = state_array.astype(int).reshape((16, 16))
			segment_mask = (
				(state_array & state_require == state_require).all(1) &
				(~state_array & state_deny == state_deny).all(1) &
				(dq_array & dq_require == dq_require) & 
				(~dq_array & dq_deny == dq_deny)
			)
		self.info('good data mask is ' + ''.join([str(x) for x in segment_mask.astype('int')]))

		# If necessary, create gap for skipped frames.
		if self.__last_successful_gps_end is not None and self.__last_successful_gps_end != gps_start:
			offset = 16384 * self.__last_successful_gps_end
			size = 16384 * (gps_start - self.__last_successful_gps_end) * len(hoft_array.data[:1].data)
			buf = gst.buffer_new_and_alloc(size)
			buf.caps = caps
			buf.offset = offset
			buf.offset_end = 16384 * gps_start
			buf.duration = gst.SECOND * (gps_start - self.__last_successful_gps_end)
			buf.timestamp = gst.SECOND * self.__last_successful_gps_end
			buf.flag_set(gst.BUFFER_FLAG_GAP)
			self.warning("pushing buffer spanning [%u, %u) (nongap=1, SKIPPED frames)"
				% (self.__last_successful_gps_end, gps_start))
			result = pad.push(buf)
			if result != gst.FLOW_OK:
				return (retval, None)
		self.__last_successful_gps_end = gps_start + 16
			
		# Loop over 1-second chunks in current buffer, and push extra buffers
		# as needed when a transition betwen gap and nongap has to occur.
		was_nongap = segment_mask[0]
		last_segment_num = 0
		for segment_num, is_nongap in enumerate(segment_mask):
			if is_nongap ^ was_nongap:
				offset = 16384 * (gps_start + last_segment_num)
				hoft_data = hoft_array[(16384*last_segment_num):(16384*segment_num)].data
				size = len(hoft_data)
				buf = gst.buffer_new_and_alloc(size)
				buf.caps = caps
				buf[0:size] = hoft_data
				buf.offset = offset
				buf.offset_end = 16384 * (gps_start + segment_num)
				buf.duration = gst.SECOND * (segment_num - last_segment_num)
				buf.timestamp = gst.SECOND * (gps_start + last_segment_num)
				if not was_nongap:
					buf.flag_set(gst.BUFFER_FLAG_GAP)
				self.info("pushing buffer spanning [%u, %u) (nongap=%d, extra frame)"
					% (gps_start + last_segment_num, gps_start + segment_num, was_nongap))
				result = pad.push(buf)
				if result != gst.FLOW_OK:
					return (retval, None)
				last_segment_num = segment_num
			was_nongap = is_nongap

		# Finish off current frame.
		segment_num = 16
		offset = 16384 * (gps_start + last_segment_num)
		hoft_data = hoft_array[(16384*last_segment_num):(16384*segment_num)].data
		size = len(hoft_data)
		buf = gst.buffer_new_and_alloc(size)
		buf.caps = caps
		buf[0:size] = hoft_data
		buf.offset = offset
		buf.offset_end = 16384 * (gps_start + segment_num)
		buf.duration = gst.SECOND * (segment_num - last_segment_num)
		buf.timestamp = gst.SECOND * (gps_start + last_segment_num)
		if not was_nongap:
			buf.flag_set(gst.BUFFER_FLAG_GAP)
		self.info("pushing buffer spanning [%u, %u) (nongap=%d)"
			% (gps_start + last_segment_num, gps_start + segment_num, was_nongap))

		# Don't need to push this buffer, just return it.
		return (gst.FLOW_OK, buf)


# Register element class
gstlal_element_register(lal_onlinehoftsrc)


if __name__ == '__main__':
	# Pipeline to demonstrate a veto kicking in.
	# Conlog says:
	#
	# H1-1792  19771 s    959175240- 959195011   2010 05/29 13:33:45 - 05/29 19:03:16 utc
	# H1-1793  13342 s    959196562- 959209904   2010 05/29 19:29:07 - 05/29 23:11:29 utc
	#
	# so after a few frames you'll see gaps start appearing, then about a hundred
	# frames later you'll see data again.
	#
	pipeline = gst.Pipeline("lal_onlinehoftsrc_example")
	elems = mkelems_in_bin(pipeline,
		("lal_onlinehoftsrc", {"instrument":"H1"}),
		("progressreport",),
		("fakesink",)
	)
	print elems[0].set_state(gst.STATE_READY)
	if not elems[0].seek(1.0, gst.FORMAT_TIME, gst.SEEK_FLAG_KEY_UNIT,
		gst.SEEK_TYPE_SET, (959195011 - 16*3)*gst.SECOND, gst.SEEK_TYPE_NONE, -1):
		raise RuntimeError, "Seek failed"
	print pipeline.set_state(gst.STATE_PLAYING)
	mainloop = gobject.MainLoop()
	mainloop.run()

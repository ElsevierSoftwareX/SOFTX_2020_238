# Copyright (C) 2013  Kipp Cannon, Chad Hanna, Drew Keppel
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


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


import os
import tempfile


import pygtk
pygtk.require("2.0")
import gobject
gobject.threads_init()
import pygst
pygst.require("0.10")
import gst


from gstlal import datasource
from gstlal import pipeparts
from gstlal import simplehandler


#
# =============================================================================
#
#                           File Clean-Up Machinery
#
# =============================================================================
#


class tempcache(list):
	"""
	List-like object to hold lal.CacheEntry objects, and run
	os.unlink() on the .path of each as they are removed from the list
	or when the list is garbage collected.  All errors during file
	removal are ignored.

	Note that there is no way to remove a CacheEntry from this list
	without the file it represents being deleted.  If, after adding a
	CacheEntry to this list it is decided the file must not be deleted,
	then instead of removing it from the list it must be replaced with
	something else, e.g. None, and that item can then be removed from
	the list.

	Example:

	>>> from glue.lal import CacheEntry
	>>> # create a cache, and add an entry
	>>> cache = tempcache()
	>>> cache.append(CacheEntry("- - - - file://localhost/tmp/blah.txt"))
	>>> # now remove it without the file being deleted
	>>> cache[-1] = None
	>>> del cache[-1]
	"""
	def __delitem__(self, i):
		try:
			os.unlink(self[i].path)
		except:
			pass
		super(tempcache, self).__delitem__(i)

	def __delslice__(self, i, j):
		try:
			for entry in self[i:j]:
				try:
					os.unlink(entry.path)
				except:
					pass
		except:
			pass
		super(tempcache, self).__delslice__(i, j)

	def __del__(self):
		del self[:]


#
# =============================================================================
#
#                                   Handler
#
# =============================================================================
#


class Handler(simplehandler.Handler):
	def __init__(self, *args, **kwargs):
		super(Handler, self).__init__(*args, **kwargs)
		self.cache = tempcache()

	def do_on_message(self, bus, message):
		if message.type == gst.MESSAGE_ELEMENT and message.structure.get_name() == "GstMultiFileSink":
			self.cache.append(pipeparts.framecpp_filesink_cache_entry_from_mfs_message(message))
			return True
		return False

	def write_cache(self, fileobj):
		for cacheentry in self.cache:
			print >>fileobj, str(cacheentry)


#
# =============================================================================
#
#                                   Pipeline
#
# =============================================================================
#


def build_pipeline(pipeline, data_source_info, output_path = tempfile.gettempdir(), sample_rate = None, description = "TMPFILE_DELETE_ME", channel_comment = None, frame_duration = 1, frames_per_file = 2048, verbose = False):
	#
	# get instrument and channel name (requires exactly one
	# instrument+channel)
	#

	(instrument, channel_name), = data_source_info.channel_dict.items()

	#
	# retrieve h(t)
	#

	src = datasource.mkbasicsrc(pipeline, data_source_info, instrument, verbose = verbose)

	#
	# optionally resample
	#

	if sample_rate is not None:
		# make sure we're *down*sampling
		src = pipeparts.mkcapsfilter(pipeline, src, "audio/x-raw-float, rate=[%d,MAX]" % sample_rate)
		src = pipeparts.mkresample(pipeline, src, quality = 9)
		src = pipeparts.mkcapsfilter(pipeline, src, "audio/x-raw-float, rate=%d" % sample_rate)

	#
	# pack into frame files for output
	#

	src = pipeparts.mkframecppchannelmux(pipeline, {"%s:%s" % (instrument, channel_name): src}, frame_duration = frame_duration, frames_per_file = frames_per_file)
	for pad in src.sink_pads():
		if channel_comment is not None:
			pad.set_property("comment", channel_comment)
		pad.set_property("pad-type", "FrProcData")
	pipeparts.mkframecppfilesink(pipeline, src, frame_type = description, path = output_path)

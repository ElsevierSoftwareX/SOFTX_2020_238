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

## @file

## @package hoftcache

#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


import os
import sys
import tempfile
import uuid


import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
GObject.threads_init()
Gst.init(None)


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
	List-like object to hold CacheEntry objects, and run os.unlink() on
	the .path of each as they are removed from the list or when the
	list is garbage collected.  All errors during file removal are
	ignored.

	Note that there is no way to remove a CacheEntry from this list
	without the file it represents being deleted.  If, after adding a
	CacheEntry to this list it is decided the file must not be deleted,
	then instead of removing it from the list it must be replaced with
	something else, e.g. None, and that item can then be removed from
	the list.

	Example:

	>>> from lal.utils import CacheEntry
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
		if message.type == Gst.MessageType.ELEMENT and message.get_structure().get_name() == "GstMultiFileSink":
			self.cache.append(pipeparts.framecpp_filesink_cache_entry_from_mfs_message(message))
			return True
		return False


#
# =============================================================================
#
#               Modified Version of mkbasicsrc from datasource.py
#
# =============================================================================
#


def mkbasicsrc(pipeline, gw_data_source_info, instrument, verbose = False):
	if gw_data_source_info.data_source == "frames":
		if instrument == "V1":
			#FIXME Hack because virgo often just uses "V" in the file names rather than "V1".  We need to sieve on "V"
			src = pipeparts.mklalcachesrc(pipeline, blocksize = 1048576, use_mmap = False, location = gw_data_source_info.frame_cache, cache_src_regex = "V")
		else:
			src = pipeparts.mklalcachesrc(pipeline, blocksize = 1048576, use_mmap = False, location = gw_data_source_info.frame_cache, cache_src_regex = instrument[0], cache_dsc_regex = instrument)
		demux = pipeparts.mkframecppchanneldemux(pipeline, src, do_file_checksum = False, channel_list = list(map("%s:%s".__mod__, gw_data_source_info.channel_dict.items())))
		pipeparts.framecpp_channeldemux_set_units(demux, dict.fromkeys(demux.get_property("channel-list"), "strain"))
		# allow frame reading and decoding to occur in a diffrent thread
		src = pipeparts.mkqueue(pipeline, None, max_size_buffers = 0, max_size_bytes = 0, max_size_time = 8 * Gst.SECOND)
		pipeparts.src_deferred_link(demux, "%s:%s" % (instrument, gw_data_source_info.channel_dict[instrument]), src.get_static_pad("sink"))
		# FIXME:  remove this when pipeline can handle disconts
		src = pipeparts.mkaudiorate(pipeline, src, skip_to_first = True, silent = False)
	else:
		raise ValueError("invalid data_source: %s" % gw_data_source_info.data_source)

	# provide an audioconvert element to allow Virgo data (which is single-precision) to be adapted into the pipeline
	src = pipeparts.mkaudioconvert(pipeline, src)

	# progress report
	if verbose:
		src = pipeparts.mkprogressreport(pipeline, src, "progress_src_%s" % instrument)

	# optional injections
	if gw_data_source_info.injection_filename is not None:
		src = pipeparts.mkinjections(pipeline, src, gw_data_source_info.injection_filename)
		# let the injection code run in a different thread than the whitener, etc.,
		src = pipeparts.mkqueue(pipeline, src, max_size_bytes = 0, max_size_buffers = 0, max_size_time = Gst.SECOND * 64)


	return src


#
# =============================================================================
#
#                                   Pipeline
#
# =============================================================================
#


def build_pipeline(pipeline, data_source_info, output_path = tempfile.gettempdir(), sample_rate = None, description = "TMPFILE_DELETE_ME_%s" % uuid.uuid4().hex, channel_comment = None, frame_duration = 1, frames_per_file = 1024, verbose = False):
	#
	# get instrument and channel name (requires exactly one
	# instrument+channel)
	#

	channelmux_input_dict = {}

	for instrument, channel_name in data_source_info.channel_dict.items():
		#
		# retrieve h(t)
		#

		src = mkbasicsrc(pipeline, data_source_info, instrument, verbose = verbose)

		#
		# optionally resample
		#

		if sample_rate is not None:
			# make sure we're *down*sampling
			src = pipeparts.mkcapsfilter(pipeline, src, "audio/x-raw, rate=[%d,MAX]" % sample_rate)
			src = pipeparts.mkresample(pipeline, src, quality = 9)
			src = pipeparts.mkcapsfilter(pipeline, src, "audio/x-raw, rate=%d" % sample_rate)

		#
		# pack into frame files for output
		#

		src = pipeparts.mkframecppchannelmux(pipeline, {"%s:%s" % (instrument, channel_name): src}, frame_duration = frame_duration, frames_per_file = frames_per_file)
		for pad in src.sinkpads:
			if channel_comment is not None:
				pad.set_property("comment", channel_comment)
			pad.set_property("pad-type", "FrProcData")
		pipeparts.mkframecppfilesink(pipeline, src, frame_type = "%s_%s" % (instrument, description), path = output_path)


#
# =============================================================================
#
#                            Collect and Cache h(t)
#
# =============================================================================
#


def cache_hoft(data_source_info, channel_comment = "cached h(t) for inspiral search", verbose = False, **kwargs):
	#
	# build pipeline
	#


	mainloop = GObject.MainLoop()
	pipeline = Gst.Pipeline(name="pipeline")
	handler = Handler(mainloop, pipeline)


	if verbose:
		print("assembling pipeline ...", file=sys.stderr)
	build_pipeline(pipeline, data_source_info, channel_comment = channel_comment, verbose = verbose, **kwargs)
	if verbose:
		print("done", file=sys.stderr)


	#
	# seek and run pipeline
	#

	if pipeline.set_state(Gst.State.READY) != Gst.StateChangeReturn.SUCCESS:
		raise RuntimeError("pipeline did not enter ready state")
	datasource.pipeline_seek_for_gps(pipeline, *data_source_info.seg)
	if verbose:
		print("setting pipeline state to playing ...", file=sys.stderr)
	if pipeline.set_state(Gst.State.PLAYING) != Gst.StateChangeReturn.SUCCESS:
		raise RuntimeError("pipeline did not enter playing state")

	if verbose:
		print("running pipeline ...", file=sys.stderr)
	mainloop.run()


	#
	# return tempcache object.  when this object is garbage collected
	# the frame files will be deleted.  keep a reference alive as long
	# as you wish to preserve the files.
	#


	return handler.cache

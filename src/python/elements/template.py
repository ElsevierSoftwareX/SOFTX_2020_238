# Copyright (C) FIXME(year)  FIXME(your name)
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


import numpy


import gobject
import pygst
pygst.require('0.10')
import gst


from gstlal import pipeio


__author__ = "FIXME"
__version__ = "FIXME"
__date__ = "FIXME"


#
# =============================================================================
#
#                                   Element
#
# =============================================================================
#


#
# a skeleton to get started.  this shows the ingredients for an element
# that transforms input data to output data out-of-place;  the
# basetransform class allows for other possibilities, review the
# documentation for more information
#


class Template(gst.BaseTransform):	# FIXME:  change class name

	#
	# the BaseTransform class requires the element to have two pads
	# named "sink" and "src";  these are described here by two pad
	# templates which the baseclass will use to create the pads
	# itself
	#

	__gsttemplates__ = (
		gst.PadTemplate("sink",
			gst.PAD_SINK,
			gst.PAD_ALWAYS,
			gst.caps_from_string(
				"audio/x-raw-float, " +
				"rate = (int) [1, MAX], " +
				"channels = (int) [1, MAX], " +
				"endianness = (int) BYTE_ORDER, " +
				"width = (int) 64"
			)
		),
		gst.PadTemplate("src",
			gst.PAD_SRC,
			gst.PAD_ALWAYS,
			gst.caps_from_string(
				"audio/x-raw-float, " +
				"rate = (int) [1, MAX], " +
				"channels = (int) [1, MAX], " +
				"endianness = (int) BYTE_ORDER, " +
				"width = (int) 64"
			)
		)
	)


	def __init__(self):
		gst.BaseTransform.__init__(self)

		#
		# this method is invoked when an instance of this class is
		# created, and it should initialize internal data.  ideally
		# it should be possible for an application to call this
		# method again, later, to re-initialize the internal data
		#


	def do_set_caps(self, snkcaps, srccaps):
		#
		# this method is invoked when the base class has chosen
		# caps for the sink and source pads, and can be used by
		# your code to record information about the format that has
		# been selected.  the example below shows how to extract
		# the channel count and sample rate from the sink (input)
		# caps
		#

		self.channels = snkcaps[0]["channels"]
		self.rate = snkcaps[0]["rate"]

		#
		# this method should return True if the caps are acceptable
		# or False if not
		#

		return True


	def do_get_unit_size(self, caps):
		#
		# this method is used by the base class to compute the unit
		# size for caps.  the unit size is the number of bytes per
		# unit of offset.  for audio streams it's the number of
		# bytes per sample (= width / 8 * channels), for video
		# streams it's the number of bytes per frame.  a utility
		# function in the pipeio module handles the common cases
		# for gstlal
		#

		return pipeio.get_unit_size(caps)


	def do_event(self, event):
		#
		# this optional method is invoked by the base class when an
		# event is received.  events that often require gstlal
		# elements to perform special actions are shown here.
		#

		if event.type == gst.EVENT_TAG:
			#
			# the framesrc element transmits additional stream
			# metadata downstream in the tag event.  a utility
			# function in the pipeio module can be used to
			# parse the information out of the event
			#

			tags = pipeio.parse_framesrc_tags(event.parse_tag())
			self.instrument = tags["instrument"]
			self.channel_name = tags["channel-name"]
			self.sample_units = tags["sample-units"]

		elif event.type == gst.EVENT_NEWSEGMENT:
			#
			# the new-segment event indicates the start of a
			# new data stream and carries information about the
			# time interval spanned by the data to follow.
			# this event is guaranteed to arrive in advance of
			# data so elements can rely on this event to
			# initialize internal segment timestamp information
			# if needed.  new-segment events are also used to
			# seek the pipeline --- to move the place where
			# processing is occuring to a new timestamp or
			# offset --- so elements with internal history
			# should be sure to handle this event and reset
			# themselves appropriately
			#

			pass

		elif event.type == gst.EVENT_EOS:
			#
			# the end-of-stream event indicates the end of a
			# data stream.  this event is guaranteed to arrive
			# after the last data in the segment, and can be
			# used by elements with internal state to "finish
			# off" their processing
			#

			pass

		#
		# returning True tells the base class to forward the event
		# downstream;  returning False tells the base class to not
		# forward the event downstream
		#

		return True


	def do_transform(self, inbuf, outbuf):
		#
		# this method is invoked by the base class to transform an
		# input buffer into an output buffer when not doing
		# in-place transforms (where the input buffer is modified
		# in-place to generate the output).  inbuf contains the
		# input buffer and outbuf is a pre-allocated buffer into
		# which the output should be placed.  the
		# .do_transform_size() method should be used to tell the
		# base class what size the output buffer needs to be if the
		# default algorithm is not correct
		#

		#
		# a utility function in the pipeio module can be used to
		# convert a gstreamer buffer containing multi-channel audio
		# data into a numpy matrix
		#

		indata = pipeio.array_from_audio_buffer(inbuf)

		#
		# use the .data attribute of a numpy array to retrieve the
		# in-ram data
		#

		outbuf[0:len(indata.data)] = indata.data
		outbuf.datasize = len(indata.data)

		#
		# set metadata on output buffer
		#

		outbuf.offset = inbuf.offset
		outbuf.offset_end = inbuf.offset_end
		outbuf.timestamp = inbuf.timestamp
		outbuf.duration = inbuf.duration

		#
		# done.  the return values are FLOW_OK for "ok", FLOW_ERROR
		# on failure, or FLOW_CUSTOM_SUCCESS if no output data was
		# generated but there was no error (this is equivalent to
		# the C GST_BASE_TRANSFORM_FLOW_DROPPED return value)
		#

		return gst.FLOW_OK


	def do_transform_caps(self, direction, caps):
		#
		# this optional method is used by the base class to convert
		# an input data format to an output data format and vice
		# versa as part of the format negotiation.  if this method
		# is not provided then the base class assumes the input and
		# output formats must be the same.  the example here does
		# the same
		#

		if direction == gst.PAD_SRC:
			#
			# convert src pad's caps to sink pad's
			#

			return caps

		elif direction == gst.PAD_SINK:
			#
			# convert sink pad's caps to src pad's
			#

			return caps

		raise ValueError, direction


	def do_transform_size(self, direction, caps, size, othercaps):
		#
		# this optional method is used by the base class to compute
		# the number of bytes that are required on the opposite pad
		# given a buffer of the given size on the given pad.  if
		# this method is not provided the base class computes the
		# number of bytes by keeping the unit count constant (using
		# .do_get_unit_size() to translate byte count to unit
		# count).  the example here does the same
		#

		if direction == gst.PAD_SRC:
			#
			# compute the number of bytes that must be received
			# on the sink pad to generate "size" bytes of data
			# on the source pad
			#

			return size / pipeio.get_unit_size(caps) * pipeio.get_unit_size(othercaps)

		elif direction == gst.PAD_SINK:
			#
			# compute the number of bytes that will be
			# generated on the source pad if a buffer
			# containing "size" bytes arrives on the sink pad
			#

			return size / pipeio.get_unit_size(caps) * pipeio.get_unit_size(othercaps)

		raise ValueError, direction


gobject.type_register(Template)	# FIXME:  change class name


def mktemplate(pipeline, src, **kwargs):	# FIXME:  change function name
	elem = Template()	# FIXME:  change class name
	for name, value in kwargs.items():
		elem.set_property(name.replace("_", "-"), value)
	pipeline.add(elem)
	src.link(elem)
	return elem

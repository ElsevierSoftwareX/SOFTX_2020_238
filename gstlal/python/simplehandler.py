# Copyright (C) 2009--2013  Kipp Cannon, Chad Hanna, Drew Keppel
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


import sys


import pygtk
pygtk.require("2.0")
import gobject
gobject.threads_init()
import pygst
pygst.require('0.10')
import gst
import signal


## @file
# The simplehander module

## @package python.simplehandler
# The simplehander module

#
# =============================================================================
#
#                               Pipeline Handler
#
# =============================================================================
#


class Handler(object):
	"""!
	A simple handler that prints pipeline error messages to stderr, and
	stops the pipeline and terminates the mainloop at EOS.  Complex
	applications will need to write their own pipeline handler, but for
	most simple applications this will suffice, and it's easier than
	copy-and-pasting this all over the place.
	"""
	def __init__(self, mainloop, pipeline):
		self.mainloop = mainloop
		self.pipeline = pipeline

		bus = pipeline.get_bus()
		bus.add_signal_watch()
		bus.connect("message", self.on_message)

	def do_on_message(self, bus, message):
		"""!
		Add extra message handling by overriding this in your
		subclass.  If this method returns True, no further message
		handling is performed.  If this method returns False,
		message handling continues with default cases or EOS, INFO,
		WARNING and ERROR messages.
		"""
		return False

	def on_message(self, bus, message):
		if self.do_on_message(bus, message):
			pass
		elif message.type == gst.MESSAGE_EOS:
			self.pipeline.set_state(gst.STATE_NULL)
			self.mainloop.quit()
		elif message.type == gst.MESSAGE_INFO:
			gerr, dbgmsg = message.parse_info()
			print >>sys.stderr, "info (%s:%d '%s'): %s" % (gerr.domain, gerr.code, gerr.message, dbgmsg)
		elif message.type == gst.MESSAGE_WARNING:
			gerr, dbgmsg = message.parse_warning()
			print >>sys.stderr, "warning (%s:%d '%s'): %s" % (gerr.domain, gerr.code, gerr.message, dbgmsg)
		elif message.type == gst.MESSAGE_ERROR:
			gerr, dbgmsg = message.parse_error()
			# FIXME:  this deadlocks.  shouldn't we be doing this?
			#self.pipeline.set_state(gst.STATE_NULL)
			self.mainloop.quit()
			sys.exit("error (%s:%d '%s'): %s" % (gerr.domain, gerr.code, gerr.message, dbgmsg))


class OneTimeSignalHandler(object):
	"""!
	A helper class for application signal handling.  Use this to help your
	application to cleanly shutdown gstreamer pipelines when responding to e.g.,
	ctrl+c.
	"""
	def __init__(self, pipeline, signals = [signal.SIGINT, signal.SIGTERM]):
		self.pipeline = pipeline
		self.count = 0
		for sig in signals:
			signal.signal(sig, self)

	def do_on_call(self, signum, frame):
		"""!
		Over ride this for your subclass
		"""
		pass

	def __call__(self, signum, frame):
		self.count += 1
		if self.count == 1:
			print >>sys.stderr, "*** SIG %d attempting graceful shutdown (this might take several minutes) ... ***" % signum
			try:
				self.do_on_call(signum, frame)
				if not self.pipeline.send_event(gst.event_new_eos()):
					raise Exception("pipeline.send_event(EOS) returned failure")
			except Exception, e:
				print >>sys.stderr, "graceful shutdown failed: %s\naborting." % str(e)
				os._exit(1)
		else:
				print >>sys.stderr, "*** received SIG %d %d times... ***" % (signum, self.count)

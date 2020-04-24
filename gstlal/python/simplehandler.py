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
import os


import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
Gst.init(None)
import signal

__doc__="""

**Review Status**

+-------------------------------------------------+------------------------------------------+------------+
| Names                                           | Hash                                     | Date       |
+=================================================+==========================================+============+
| Florent, Sathya, Duncan Me., Jolien, Kipp, Chad | b3ef077fe87b597578000f140e4aa780f3a227aa | 2014-05-01 |
+-------------------------------------------------+------------------------------------------+------------+

"""


#
# =============================================================================
#
#                               Pipeline Handler
#
# =============================================================================
#


class Handler(object):
	"""
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
		self.on_message_handler_id = bus.connect("message", self.on_message)

		def excepthook(*args):
			# system exception hook that forces hard exit.  without this,
			# exceptions that occur inside python code invoked as a call-back
			# from the gstreamer pipeline just stop the pipeline, they don't
			# cause gstreamer to exit.

			# FIXME:  they probably *would* cause if we could figure out why
			# element errors and the like simply stop the pipeline instead of
			# crashing it, as well.  Perhaps this should be removed when/if the
			# "element error's don't crash program" problem is fixed
			sys.__excepthook__(*args)
			os._exit(1)

		sys.excepthook = excepthook


	def quit(self, bus):
		"""
		Decouple this object from the Bus object to allow the Bus'
		reference count to drop to 0, and .quit() the mainloop
		object.  This method is invoked by the default EOS and
		ERROR message handlers.
		"""
		bus.disconnect(self.on_message_handler_id)
		del self.on_message_handler_id
		bus.remove_signal_watch()
		self.mainloop.quit()

	def do_on_message(self, bus, message):
		"""
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
		elif message.type == Gst.MessageType.EOS:
			self.pipeline.set_state(Gst.State.NULL)
			self.quit(bus)
		elif message.type == Gst.MessageType.INFO:
			gerr, dbgmsg = message.parse_info()
			print("info (%s:%d '%s'): %s" % (gerr.domain, gerr.code, gerr.message, dbgmsg), file=sys.stderr)
		elif message.type == Gst.MessageType.WARNING:
			gerr, dbgmsg = message.parse_warning()
			print("warning (%s:%d '%s'): %s" % (gerr.domain, gerr.code, gerr.message, dbgmsg), file=sys.stderr)
		elif message.type == Gst.MessageType.ERROR:
			gerr, dbgmsg = message.parse_error()
			# FIXME:  this deadlocks.  shouldn't we be doing this?
			#self.pipeline.set_state(gst.STATE_NULL)
			self.quit(bus)
			sys.exit("error (%s:%d '%s'): %s" % (gerr.domain, gerr.code, gerr.message, dbgmsg))


class OneTimeSignalHandler(object):
	"""
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
		"""
		Over ride this for your subclass
		"""
		pass

	def __call__(self, signum, frame):
		self.count += 1
		if self.count == 1:
			print("*** SIG %d attempting graceful shutdown (this might take several minutes) ... ***" % signum, file=sys.stderr)
			try:
				self.do_on_call(signum, frame)
				if not self.pipeline.send_event(Gst.Event.new_eos()):
					raise Exception("pipeline.send_event(EOS) returned failure")
			except Exception as e:
				print("graceful shutdown failed: %s\naborting." % str(e), file=sys.stderr)
				os._exit(1)
		else:
				print("*** received SIG %d %d times... ***" % (signum, self.count), file=sys.stderr)

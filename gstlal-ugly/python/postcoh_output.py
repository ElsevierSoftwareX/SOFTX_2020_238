import threading
import sys

# The following snippet is taken from http://gstreamer.freedesktop.org/wiki/FAQ#Mypygstprogramismysteriouslycoredumping.2Chowtofixthis.3F
import pygtk
pygtk.require("2.0")
import gobject
gobject.threads_init()
import pygst
pygst.require('0.10')
import gst

try:
	from ligo import gracedb
except ImportError:
	print >>sys.stderr, "warning: gracedb import failed, program will crash if gracedb uploads are attempted"

from glue import iterutils
from glue import segments
from glue.ligolw import ligolw
from glue.ligolw import dbtables
from glue.ligolw import ilwd
from glue.ligolw import lsctables
from glue.ligolw import array as ligolw_array
from glue.ligolw import param as ligolw_param
from glue.ligolw import utils as ligolw_utils
import lal
from pylal.datatypes import LALUnit
from pylal.datatypes import LIGOTimeGPS
from pylal.datatypes import REAL8FrequencySeries

lsctables.LIGOTimeGPS = LIGOTimeGPS

class Data(object):
	def __init__(self, output_prefix, cluster_window = 1, snapshot_interval = None, gracedb_far_threshold = None, gracedb_group = "Test", gracedb_search = "LowMass", gracedb_pipeline = "gstlal_spiir", gracedb_service_url = "https://gracedb.ligo.org/api/", verbose = False):
	#
	# initialize
	#
		self.lock = threading.Lock()
		self.verbose = verbose
		self.output_prefix = output_prefix
		self.cluster_window = cluster_window
		self.gracedb_far_threshold = gracedb_far_threshold
		self.gracedb_group = gracedb_group
		self.gracedb_search = gracedb_search
		self.gracedb_pipeline = gracedb_pipeline
		self.gracedb_service_url = gracedb_service_url

		#self.postcoh_document = PostcohDocument()

		self.t_roll_start = None

	def appsink_new_buffer(self, elem):
		with self.lock:
			buf = elem.emit("pull-buffer")
			buf_timestamp = LIGOTimeGPS(0, buf.timestamp)
			if self.t_roll_start is None:
				self.t_roll_start = buf_timestamp
			print buf_timestamp
			#events = PostcohFromBuf(buf)

			#self.cluster(events, self.cluster_window)

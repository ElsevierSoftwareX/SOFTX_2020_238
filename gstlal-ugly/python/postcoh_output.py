import threading
import sys
import pdb

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
from glue.ligolw import table
from glue.ligolw import lsctables
from glue.ligolw import array as ligolw_array
from glue.ligolw import param as ligolw_param
from glue.ligolw import utils as ligolw_utils
import lal
from pylal.datatypes import LALUnit
from pylal.datatypes import LIGOTimeGPS
from pylal.datatypes import REAL8FrequencySeries
from pylal.xlal.datatypes import postcohinspiraltable

lsctables.LIGOTimeGPS = LIGOTimeGPS

# defined in postcohinspiral_table.h
class PostcohInspiralTable(table.Table):
	tableName = "postcoh:table"
	validcolumns = {
			"end_time":	"int_4s",
			"end_time_ns":	"int_4s",
			"is_background":	"int_4s",
			"livetime":	"int_4s",
			"ifos":		"lstring",
			"pivotal_ifo":	"lstring",
			"tmplt_idx":	"int_4s",
			"pix_idx":	"int_4s",
			"maxsnglsnr":	"real_4",
			"cohsnr":	"real_4",
			"nullsnr":	"real_4",
			"chisq":	"real_4",
			"spearman_pval":	"real_4",
			"fap":		"real_4",
			"far":		"real_4",
			"skymap_fname":	"lstring",
			"template_duration": "real_8",
			"mass1":	"real_4",
			"mass2":	"real_4",
			"mchirp":	"real_4",
			"mtotal":	"real_4",
			"spin1x":	"real_4",
			"spin1y":	"real_4",
			"spin1z":	"real_4",
			"spin2x":	"real_4",
			"spin2y":	"real_4",
			"spin2z":	"real_4",
			"ra":		"real_8",
			"dec":		"real_8"
	}


class PostcohDocument(object):
	def __init__(self, verbose = False):
		self.get_another = lambda: PostcohDocument(verbose = verbose)

		self.filename = None

		#
		# build the XML document
		#

		self.xmldoc = ligolw.Document()
		self.xmldoc.appendChild(ligolw.LIGO_LW())

		# FIXME: process table, search summary table
		# FIXME: should be implemented as lsctables.PostcohInspiralTable
		self.xmldoc.childNodes[-1].appendChild(lsctables.New(PostcohInspiralTable))

	def set_filename(self, filename):
		self.filename = filename

	def write_output_file(self, verbose = False):
		assert self.filename is not None
		ligolw_utils.write_filename(self.xmldoc, self.filename, gz = (self.filename or "stdout").endswith(".gz"), verbose = verbose, trap_signals = None)


class Data(object):
	def __init__(self, pipeline, data_output_prefix, cluster_window = 1, data_snapshot_interval = None, gracedb_far_threshold = None, gracedb_group = "Test", gracedb_search = "LowMass", gracedb_pipeline = "gstlal_spiir", gracedb_service_url = "https://gracedb.ligo.org/api/", verbose = False):
	#
	# initialize
	#
		self.lock = threading.Lock()
		self.verbose = verbose
		self.pipeline = pipeline
		self.data_output_prefix = data_output_prefix
		self.data_snapshot_interval = data_snapshot_interval
		self.cluster_window = cluster_window
		self.gracedb_far_threshold = gracedb_far_threshold
		self.gracedb_group = gracedb_group
		self.gracedb_search = gracedb_search
		self.gracedb_pipeline = gracedb_pipeline
		self.gracedb_service_url = gracedb_service_url

		self.postcoh_document = PostcohDocument()
		self.postcoh_table = PostcohInspiralTable.get_table(self.postcoh_document.xmldoc)

		self.t_roll_start = None
		self.duration_roll = None

	def appsink_new_buffer(self, elem):
		#pdb.set_trace()
		with self.lock:
			buf = elem.emit("pull-buffer")
			buf_timestamp = LIGOTimeGPS(0, buf.timestamp)
			if self.t_roll_start is None:
				self.t_roll_start = buf_timestamp
			#print buf_timestamp
			events = postcohinspiraltable.PostcohInspiralTable.from_buffer(buf)
			self.postcoh_table.extend(events)
			self.duration_roll = buf_timestamp - self.t_roll_start
			if self.data_snapshot_interval is not None and self.duration_roll > self.data_snapshot_interval:
				snapshot_filename = self.get_output_filename(self.output_data_prefix, self.t_roll_start, self.duration_roll)
				self.snapshot_output_file(snapshot_filename)
				self.t_roll_start = buf_timestamp
			#self.cluster(events, self.cluster_window)
	def get_output_filename(self, output_data_prefix, t_roll_start, duration_roll):
			fname = "%s_%d_%d.xml.gz" % (output_data_prefix, t_roll_start, duration_roll)
			return fname

	def snapshot_output_file(self, filename, verbose = False):
		with self.lock:
			postcoh_document = self.postcoh_document.get_another()
			self.__write_output_file(filename = filename, verbose = verbose)
			del self.postcoh_document
			self.postcoh_document = postcoh_document

	def write_output_file(self, filename = None, verbose = False):
		self.__write_output_file(filename)

	def __write_output_file(self, filename = None, verbose = False):
		if filename is not None:
			self.postcoh_document.set_filename(filename)
		self.postcoh_document.write_output_file(verbose = verbose)

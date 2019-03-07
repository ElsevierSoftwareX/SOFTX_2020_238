"""
Short cutting gstlal inspiral pipeline to produce SNR for template(s)
"""

import sys
import numpy
import threading

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstAudio', '1.0')
from gi.repository import GObject, Gst, GstAudio
GObject.threads_init()
Gst.init(None)

from gstlal import datasource
from gstlal import lloidparts
from gstlal import multirate_datasource
from gstlal import pipeparts
from gstlal import pipeio
from gstlal import reference_psd
from gstlal import simplehandler

import lal
from lal import LIGOTimeGPS
import lal.series

from ligo.lw import array as ligolw_array
from ligo.lw import ligolw
from ligo.lw import param as ligolw_param
from ligo.lw import utils as ligolw_utils

@ligolw_array.use_in
@ligolw_param.use_in
class SNRContentHandler(ligolw.LIGOLWContentHandler):
	pass

class LLOID_SNR(object):
	"""
	 The options for SNR calculation, please refer to multirate_datasource.mkwhitened_src()
	 and llloidparts.mkLLOIDhoftToSnrSlices() for more information. Defaults are:

		"psd_fft_length": 32,
		"ht_gate_threshold": None,
		"veto_segments": None,
		"track_psd": False,
		"width": 32,
		"verbose": False
	"""

	def __init__(self, psd_fft_length = 32, ht_gate_threshold = None, veto_segments = None, track_psd = False, width = 32, verbose = False):
		self.psd_fft_length = psd_fft_length
		self.ht_gate_threshold = ht_gate_threshold
		self.veto_segments = veto_segments
		self.track_psd = track_psd
		self.width = width
		self.verbose = verbose

		self.lock = threading.Lock()
		self.snr_info = {
			"timestamps": [],
			"instrument": None,
			"deltaT": None,
			"data": [],
		}

	def __call__(self, gw_data_source_info, bank, instrument, psd = None):
		pipeline = Gst.Pipeline(name = "gstlal_inspiral_LLOID_SNR")
		mainloop = GObject.MainLoop()
		handler = simplehandler.Handler(mainloop, pipeline)
		self.snr_info["instrument"] = instrument

		# sanity check
		if psd is not None:
			assert instrument in set(psd)
		assert instrument in set(gw_data_source_info.channel_dict)

		if self.verbose:
			sys.stderr.write("Building pipeline to calculate SNR...\n")

		src, statevector, dqvector = datasource.mkbasicsrc(pipeline, gw_data_source_info, instrument, self.verbose)

		hoftdict = multirate_datasource.mkwhitened_multirate_src(
			pipeline,
			src = src,
			rates = set(rate for rate in bank.get_rates()),
			instrument = instrument,
			psd = psd[instrument],
			psd_fft_length = self.psd_fft_length,
			ht_gate_threshold = self.ht_gate_threshold,
			veto_segments = self.veto_segments,
			track_psd = self.track_psd,
			width = self.width,
			statevector = statevector,
			dqvector = dqvector,
			fir_whiten_reference_psd = bank.processed_psd
			)

		snr = lloidparts.mkLLOIDhoftToSnrSlices(
			pipeline,
			hoftdict = hoftdict,
			bank = bank,
			control_snksrc = (None, None),
			block_duration = 8 * Gst.SECOND,
			fir_stride = 16,
			verbose = self.verbose,
			logname = instrument
			)

		appsink = pipeparts.mkappsink(pipeline, snr, drop = False)
		handler_id = appsink.connect("new-preroll", self.new_preroll_handler)
		assert handler_id > 0
		handler_id = appsink.connect("new-sample", self.pull_buffer)
		assert handler_id > 0
		handler_id = appsink.connect("eos", self.pull_buffer)
		assert handler_id > 0

		if self.verbose:
			sys.stderr.write("Setting pipeline state to READY...\n")
		if pipeline.set_state(Gst.State.READY) != Gst.StateChangeReturn.SUCCESS:
			raise RuntimeError("pipeline cannot enter ready state.")

		datasource.pipeline_seek_for_gps(pipeline, *gw_data_source_info.seg)

		if self.verbose:
			sys.stderr.write("Seting pipeline state to PLAYING...\n")
		if pipeline.set_state(Gst.State.PLAYING) != Gst.StateChangeReturn.SUCCESS:
			raise RuntimeError("pipeline cannot enter playing state.")
		if self.verbose:
			sys.stderr.write("Calculating SNR...\n")

		mainloop.run()

		if self.verbose:
			sys.stderr.write("Calculation done.\n")
		if pipeline.set_state(Gst.State.NULL) != Gst.StateChangeReturn.SUCCESS:
			raise RuntimeError("pipeline could not be set to NULL.")

		del pipeline, mainloop, handler

		# return snr_info containing information to construct snr for all template in the template bank
		# see make_SNR_series() to make SNR LAL Series from snr_info
		self.snr_info["data"] = numpy.concatenate(numpy.array(self.snr_info["data"]), axis = 0)
		return self.snr_info

	#===============================================================================================
	#
	#									internal functions
	#
	#===============================================================================================
	def new_preroll_handler(self, elem):
		with self.lock:
			# ignore preroll buffers
			elem.emit("pull-preroll")
			return Gst.FlowReturn.OK

	def pull_buffer(self, elem):
		with self.lock:
			sample = elem.emit("pull-sample")
			if sample is None:
				return Gst.FlowReturn.OK
			else:
				success, rate = sample.get_caps().get_structure(0).get_int("rate")

				assert success == True
				# make sure the sampling rate is the same for all data
				if self.snr_info["deltaT"] is None:
					self.snr_info["deltaT"] = 1. / rate
				else:
					assert self.snr_info["deltaT"] == 1. / rate, "data have different sampling rate."

				buf = sample.get_buffer()
				if buf.mini_object.flags & Gst.BufferFlags.GAP or buf.n_memory() == 0:
					return Gst.FlowReturn.OK
				# FIXME: check timestamps
				data = pipeio.array_from_audio_sample(sample)
				if data is not None:
					self.snr_info["data"].append(data)
					self.snr_info["timestamps"].append(LIGOTimeGPS(0, sample.get_buffer().pts))
				return Gst.FlowReturn.OK

class FIR_SNR(object):
	"""
	Required arguments:
	-gw_data_source_info:
	-template:
	-psd:
	-instrument:

	Optional arguments:
	-psd_fft_length:
	-zero_pad: Hanning Window's zero padding (seconds)
	-average_samples:
	-median_samples:
	-rate:
	-verbose:
	"""

	def __init__(self, rate, psd_fft_length = 32, zero_pad = 0, average_samples = 64, median_samples = 7, width = 32, track_psd = False, verbose = False):
		self.average_samples = average_samples
		self.lock = threading.Lock()
		self.median_samples = median_samples
		self.psd_fft_length = psd_fft_length
		self.rate = rate
		self.track_psd = track_psd
		self.verbose = verbose
		self.width = width
		self.zero_pad = zero_pad

		self.snr_info = {
			"timestamps": [],
			"instrument": None,
			"deltaT": 1./rate,
			"data": [],
		}

	def __call__(self, gw_data_source_info, template, instrument, latency, psd = None):
		self.snr_info["instrument"] = instrument

		pipeline = Gst.Pipeline("gstlal_inspiral_simple_SNR")
		mainloop = GObject.MainLoop()
		handler = simplehandler.Handler(mainloop, pipeline)

		if self.verbose:
			sys.stderr.write("Building pipeline to calculate SNR\n")

		src, statevector, dqvector = datasource.mkbasicsrc(pipeline, gw_data_source_info, instrument, verbose = self.verbose)

		hoftdict = multirate_datasource.mkwhitened_multirate_src(
					pipeline,
					src = src,
					rates = [self.rate],
					instrument = instrument,
					psd = psd[instrument],
					psd_fft_length = self.psd_fft_length,
					track_psd = self.track_psd,
					width = self.width,
					statevector = statevector,
					dqvector = dqvector
					)

		#FIXME: how to set latency
		head = pipeparts.mkfirbank(pipeline, hoftdict[self.rate], latency = latency, fir_matrix = [template], block_stride = 16 * self.rate, time_domain = False)

		appsink = pipeparts.mkappsink(pipeline, head, drop = False)
		handler_id = appsink.connect("new-preroll", self.new_preroll_handler)
		assert handler_id > 0
		handler_id = appsink.connect("new-sample", self.pull_buffer)
		assert handler_id > 0
		handler_id = appsink.connect("eos", self.pull_buffer)
		assert handler_id > 0

		if self.verbose:
			sys.stderr.write("Setting pipeline state to READY...\n")
		if pipeline.set_state(Gst.State.READY) != Gst.StateChangeReturn.SUCCESS:
			raise RuntimeError("pipeline cannot enter ready state.")

		datasource.pipeline_seek_for_gps(pipeline, *gw_data_source_info.seg)

		if self.verbose:
			sys.stderr.write("Setting pipeline state to PLAYING...\n")
		if pipeline.set_state(Gst.State.PLAYING) != Gst.StateChangeReturn.SUCCESS:
			raise RuntimeError("pipeline cannot enter playing state.")
		if self.verbose:
			sys.stderr.write("Calculating SNR...\n")

		mainloop.run()

		if self.verbose:
			sys.stderr.write("Calculation done.\n")
		if pipeline.set_state(Gst.State.NULL) != Gst.StateChangeReturn.SUCCESS:
			raise RuntimeError("pipeline could not be set to NULL.")

		del pipeline, mainloop, handler

		self.snr_info["data"] = numpy.concatenate(numpy.array(self.snr_info["data"]), axis = 0)
		return self.snr_info

	#===============================================================================================
	#
	#									internal functions
	#
	#===============================================================================================

	def new_preroll_handler(self, elem):
		with self.lock:
			# ignore preroll buffers
			elem.emit("pull-preroll")
			return Gst.FlowReturn.OK

	def pull_buffer(self, elem):
		with self.lock:
			sample = elem.emit("pull-sample")
			if sample is None:
				return Gst.FlowReturn.OK
			else:
				success, rate = sample.get_caps().get_structure(0).get_int("rate")

				assert success == True
				# make sure the sampling rate is the same for all data
				if self.snr_info["deltaT"] is not None:
					assert self.snr_info["deltaT"] == 1. / rate, "data have different sampling rate."
				self.snr_info["deltaT"] = 1. / rate

				buf = sample.get_buffer()
				if buf.mini_object.flags & Gst.BufferFlags.GAP or buf.n_memory() == 0:
					return Gst.FlowReturn.OK

				data = pipeio.array_from_audio_sample(sample)
				if data is not None:
					self.snr_info["data"].append(data)
					self.snr_info["timestamps"].append(LIGOTimeGPS(0, sample.get_buffer().pts))
				return Gst.FlowReturn.OK

#=============================================================================================
#
# 										Output Utilities
#
#=============================================================================================
def make_snr_series(snr_info, row_number = 0, drop_first = 0, drop_last = 0):
	assert drop_first >= 0, "must drop positive number of data"
	assert drop_last >= 0, "must drop positive number of data"
	bps = drop_first * int(round(1 / snr_info["deltaT"]))
	bpe = -drop_last * int(round(1 / snr_info["deltaT"])) if drop_last != 0 else None

	data = numpy.abs(snr_info["data"])[:,row_number][bps:bpe]

	if data.dtype == numpy.float32:
		tseries = lal.CreateREAL4TimeSeries(
				name = snr_info["instrument"],
				epoch = snr_info["timestamps"][0] + drop_first,
				deltaT = snr_info["deltaT"],
				f0 = 0,
				sampleUnits = lal.DimensionlessUnit,
				length = len(data)
				)
		tseries.data.data = data
	elif data.dtype == numpy.float64:
		tseries = lal.CreateREAL8TimeSeries(
				name = snr_info["instrument"],
				epoch = snr_info["timestamps"][0] + drop_first,
				deltaT = snr_info["deltaT"],
				f0 = 0,
				sampleUnits = lal.DimensionlessUnit,
				length = len(data)
				)
		tseries.data.data = data
	else:
		raise ValueError("unsupported type : %s " % data.dtype)

	return tseries

def make_xmldoc(snrdict, xmldoc = None, root_name = u"gstlal_inspiral_snr"):
	if xmldoc is None:
		xmldoc = ligolw.Document()

	root = xmldoc.appendChild(ligolw.LIGO_LW())
	root.Name = root_name
	for instrument, snr in snrdict.items():
		if snr.data.data.dtype == numpy.float32:
			tseries = root.appendChild(lal.series.build_REAL4TimeSeries(snr))
		elif snr.data.data.dtype == numpy.float64:
			tseries = root.appendChild(lal.series.build_REAL8TimeSeries(snr))
		else:
			raise ValueError("unsupported type : %s" % snr.data.data.dtype)
		if instrument is not None:
			tseries.appendChild(ligolw_param.Param.from_pyvalue(u"instrument", instrument))

	return xmldoc

def read_xmldoc(xmldoc, root_name = u"gstlal_inspiral_snr"):
	if root_name is not None:
		xmldoc, = (elem for elem in xmldoc.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") if elem.Name == root_name)

	result = []
	for elem in (elem for elem in xmldoc.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name")):
		if elem.Name == u"REAL4TimeSeries":
			result.append((ligolw_param.get_pyvalue(elem, u"instrument"), lal.series.parse_REAL4TimeSeries(elem)))
		elif elem.Name == u"REAL8TimeSeries":
			result.append((ligolw_param.get_pyvalue(elem, u"instrument"), lal.series.parse_REAL8TimeSeries(elem)))

	assert result is not None, "xmldoc contains no LAL Series or LAL Series is unsupported"

	return dict(result)

# wrapper for writing snr series to URL
def write_url(xmldoc, filename, verbose = False):
	ligolw_utils.write_filename(xmldoc, filename, gz = filename.endswith(".gz"), verbose = verbose)

# wrapper for reading snr series from URL
def read_url(filename, contenthandler, verbose = False):
	return ligolw_utils.load_url(filename, verbose = verbose, contenthandler = contenthandler)

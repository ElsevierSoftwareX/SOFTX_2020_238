"""
Short cutting gstlal inspiral pipeline to produce SNR for gstlal_svd_bank.
A gstlal-based direct matched filter in time domain is also implemented.
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

class SNR_Pipeline(object):
	def __init__(self, name = "gstlal_inspiral_SNR", verbose = False):
		self.pipeline = Gst.Pipeline(name = name)
		self.mainloop = GObject.MainLoop()
		self.handler = simplehandler.Handler(self.mainloop, self.pipeline)
		self.verbose = verbose
		self.lock = threading.Lock()
		self.snr_info = {
			"epoch": None,
			"instrument": None,
			"deltaT": None,
			"data": [],
		}

	def run(self, segments):
		if self.verbose:
			sys.stderr.write("Setting pipeline state to READY...\n")
		if self.pipeline.set_state(Gst.State.READY) != Gst.StateChangeReturn.SUCCESS:
			raise RuntimeError("pipeline cannot enter ready state.")

		datasource.pipeline_seek_for_gps(self.pipeline, *segments)

		if self.verbose:
			sys.stderr.write("Seting pipeline state to PLAYING...\n")
		if self.pipeline.set_state(Gst.State.PLAYING) != Gst.StateChangeReturn.SUCCESS:
			raise RuntimeError("pipeline cannot enter playing state.")
		if self.verbose:
			sys.stderr.write("Calculating SNR...\n")

		self.mainloop.run()

		if self.verbose:
			sys.stderr.write("Calculation done.\n")
		if self.pipeline.set_state(Gst.State.NULL) != Gst.StateChangeReturn.SUCCESS:
			raise RuntimeError("pipeline could not be set to NULL.")

	def make_series(self, data):
		if data.dtype == numpy.float32:
			tseries = lal.CreateREAL4TimeSeries(
					name = self.snr_info["instrument"],
					epoch = self.snr_info["epoch"],
					deltaT = self.snr_info["deltaT"],
					f0 = 0,
					sampleUnits = lal.DimensionlessUnit,
					length = len(data)
					)
			tseries.data.data = data
		elif data.dtype == numpy.float64:
			tseries = lal.CreateREAL8TimeSeries(
					name = self.snr_info["instrument"],
					epoch = self.snr_info["epoch"],
					deltaT = self.snr_info["deltaT"],
					f0 = 0,
					sampleUnits = lal.DimensionlessUnit,
					length = len(data)
					)
			tseries.data.data = data
		elif data.dtype == numpy.complex64:
			tseries = lal.CreateCOMPLEX8TimeSeries(
					name = self.snr_info["instrument"],
					epoch = self.snr_info["epoch"],
					deltaT = self.snr_info["deltaT"],
					f0 = 0,
					sampleUnits = lal.DimensionlessUnit,
					length = len(data)
					)
			tseries.data.data = data
		elif data.dtype == numpy.complex128:
			tseries = lal.CreateCOMPLEX16TimeSeries(
					name = self.snr_info["instrument"],
					epoch = self.snr_info["epoch"],
					deltaT = self.snr_info["deltaT"],
					f0 = 0,
					sampleUnits = lal.DimensionlessUnit,
					length = len(data)
					)
			tseries.data.data = data

		else:
			raise ValueError("unsupported type : %s " % data.dtype)

		return tseries

	def get_snr_series(self, COMPLEX = False, row_number = None, start = None, end = None):
		gps_start = self.snr_info["epoch"].gpsSeconds + self.snr_info["epoch"].gpsNanoSeconds * 10.**-9
		gps = gps_start + numpy.arange(len(self.snr_info["data"])) * self.snr_info["deltaT"]
		if start and end:
			if start >= end:
				raise ValueError("Start time must be less than end time.")

			if start - gps[0] >= 0 and start - gps[-1] <= 0:
				s = abs(gps - start).argmin()
			else:
				raise ValueError("Invalid choice of start time %f." % start)

			if end - gps[0] >= 0 and end - gps[-1] <= 0:
				e = abs(gps - end).argmin()
			else:
				raise ValueError("Invalid choice of end time %f." % end)

			self.snr_info["epoch"] = gps[s]
			self.snr_info["data"] = self.snr_info["data"][s:e].T
		else:
			self.snr_info["epoch"] = gps[0]
			self.snr_info["data"] = self.snr_info["data"].T

		if row_number is None:
			temp = []
			if COMPLEX:
				for data in self.snr_info["data"]:
					temp.append(self.make_series(data))
				return temp
			else:
				for data in self.snr_info["data"]:
					temp.append(self.make_series(numpy.abs(data)))
				return temp
		else:
			self.snr_info["data"] = self.snr_info["data"][row_number]
			if COMPLEX:
				return [self.make_series(self.snr_info["data"])]
			else:
				return [self.make_series(numpy.abs(self.snr_info["data"]))]


        def new_preroll_handler(self, elem):
                with self.lock:
                        # ignore preroll buffers
                        elem.emit("pull-preroll")
                        return Gst.FlowReturn.OK

        def pull_snr_buffer(self, elem):
                with self.lock:
                        sample = elem.emit("pull-sample")
                        if sample is None:
                                return Gst.FlowReturn.OK

			success, rate = sample.get_caps().get_structure(0).get_int("rate")

			assert success == True
			# make sure the sampling rate is the same for all data
			if self.snr_info["deltaT"] is None:
				self.snr_info["deltaT"] = 1. / rate
			else:
				assert self.snr_info["deltaT"] == 1. / rate, "data have different sampling rate."

			# record the first timestamp
			if self.snr_info["epoch"] is None:
				self.snr_info["epoch"] = LIGOTimeGPS(0, sample.get_buffer().pts)

			buf = sample.get_buffer()
			if buf.mini_object.flags & Gst.BufferFlags.GAP or buf.n_memory() == 0:
				return Gst.FlowReturn.OK
			# FIXME: check timestamps
			data = pipeio.array_from_audio_sample(sample)
			if data is not None:
				self.snr_info["data"].append(data)
			return Gst.FlowReturn.OK

class LLOID_SNR(SNR_Pipeline):
	def __init__(self, gw_data_source_info, bank, instrument, psd = None, psd_fft_length = 32, ht_gate_threshold = float("inf"), veto_segments = None, track_psd = False, width = 32, verbose = False):
		SNR_Pipeline.__init__(self, name = "gstlal_inspiral_lloid_snr", verbose = verbose)
		self.snr_info["instrument"] = instrument

		# sanity check
		if psd is not None:
			if not (instrument in set(psd)):
				raise ValueError("No psd for instrument %s." % instrument)

		if self.verbose:
			sys.stderr.write("Building pipeline to calculate SNR...\n")

		src, statevector, dqvector = datasource.mkbasicsrc(self.pipeline, gw_data_source_info, instrument, self.verbose)

		hoftdict = multirate_datasource.mkwhitened_multirate_src(
			self.pipeline,
			src = src,
			rates = set(rate for rate in bank.get_rates()),
			instrument = instrument,
			psd = psd[instrument],
			psd_fft_length = psd_fft_length,
			ht_gate_threshold = ht_gate_threshold,
			veto_segments = veto_segments,
			track_psd = track_psd,
			width = width,
			statevector = statevector,
			dqvector = dqvector,
			fir_whiten_reference_psd = bank.processed_psd
			)

		snr = lloidparts.mkLLOIDhoftToSnrSlices(
			self.pipeline,
			hoftdict = hoftdict,
			bank = bank,
			control_snksrc = (None, None),
			block_duration = 8 * Gst.SECOND,
			fir_stride = 16,
			verbose = self.verbose,
			logname = instrument
			)

		appsink = pipeparts.mkappsink(self.pipeline, snr, drop = False)
		handler_id = appsink.connect("new-preroll", self.new_preroll_handler)
		assert handler_id > 0
		handler_id = appsink.connect("new-sample", self.pull_snr_buffer)
		assert handler_id > 0
		handler_id = appsink.connect("eos", self.pull_snr_buffer)
		assert handler_id > 0

		self.run(gw_data_source_info.seg)
                self.snr_info["data"] = numpy.concatenate(numpy.array(self.snr_info["data"]), axis = 0)

	def __call__(self, COMPLEX = False, row_number = 0, start = None, end = None):
		return self.get_snr_series(COMPLEX, row_number, start, end)

class FIR_SNR(SNR_Pipeline):
	def __init__(self, gw_data_source_info, template, instrument, rate, latency, psd = None, psd_fft_length = 32, ht_gate_threshold = float("inf"), veto_segments = None, width = 32, track_psd = False, verbose = False):
		SNR_Pipeline.__init__(self, name = "gstlal_inspiral_fir_snr", verbose = verbose)
		self.snr_info["instrument"] = instrument

		# sanity check
		if psd is not None:
			if not (instrument in set(psd)):
				raise ValueError("No psd for instrument %s." % instrument)

		if self.verbose:
			sys.stderr.write("Building pipeline to calculate SNR\n")

		src, statevector, dqvector = datasource.mkbasicsrc(self.pipeline, gw_data_source_info, instrument, verbose = self.verbose)

		hoftdict = multirate_datasource.mkwhitened_multirate_src(
					self.pipeline,
					src = src,
					rates = [rate],
					instrument = instrument,
					psd = psd[instrument],
					psd_fft_length = psd_fft_length,
					ht_gate_threshold = ht_gate_threshold,
					veto_segments = veto_segments,
					track_psd = track_psd,
					width = width,
					statevector = statevector,
					dqvector = dqvector
					)

		#FIXME: how to set latency
		head = pipeparts.mkfirbank(self.pipeline, hoftdict[rate], latency = latency, fir_matrix = [template.real, template.imag], block_stride = 16 * rate, time_domain = False)

		appsink = pipeparts.mkappsink(self.pipeline, head, drop = False)
		handler_id = appsink.connect("new-preroll", self.new_preroll_handler)
		assert handler_id > 0
		handler_id = appsink.connect("new-sample", self.pull_snr_buffer)
		assert handler_id > 0
		handler_id = appsink.connect("eos", self.pull_snr_buffer)
		assert handler_id > 0

		self.run(gw_data_source_info.seg)
                self.snr_info["data"] = numpy.concatenate(numpy.array(self.snr_info["data"]), axis = 0)
		self.snr_info["data"] = numpy.vectorize(complex)(self.snr_info["data"][:,0], self.snr_info["data"][:,1])
		self.snr_info["data"].shape = len(self.snr_info["data"]), 1

	def __call__(self, COMPLEX = False, row_number = 0 , start = None, end = None):
		return self.get_snr_series(COMPLEX, row_number, start, end)

#=============================================================================================
#
# 					Output Utilities
#
#=============================================================================================

def make_xmldoc(snrdict, xmldoc = None, root_name = u"gstlal_inspiral_snr"):
	if xmldoc is None:
		xmldoc = ligolw.Document()

	root = xmldoc.appendChild(ligolw.LIGO_LW())
	root.Name = root_name
	for instrument, snrs in snrdict.items():
		for snr in snrs:
			if snr.data.data.dtype == numpy.float32:
				tseries = root.appendChild(lal.series.build_REAL4TimeSeries(snr))
			elif snr.data.data.dtype == numpy.float64:
				tseries = root.appendChild(lal.series.build_REAL8TimeSeries(snr))
			elif snr.data.data.dtype == numpy.complex64:
				tseries = root.appendChild(lal.series.build_COMPLEX8TimeSeries(snr))
			elif snr.data.data.dtype == numpy.complex128:
				tseries = root.appendChild(lal.series.build_COMPLEX16TimeSeries(snr))
			else:
				raise ValueError("unsupported type : %s" % snr.data.data.dtype)
	return xmldoc

def read_xmldoc(xmldoc, root_name = u"gstlal_inspiral_snr"):
	if root_name is not None:
		xmldoc, = (elem for elem in xmldoc.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") if elem.Name == root_name)

	result = {}
	temp = []
	for elem in (elem for elem in xmldoc.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name")):
		if elem.Name == u"REAL4TimeSeries":
			tseries = lal.series.parse_REAL4TimeSeries(elem)
			temp.append([tseries.name, tseries])
		elif elem.Name == u"REAL8TimeSeries":
			tseries = lal.series.parse_REAL8TimeSeries(elem)
			temp.append([tseries.name, tseries])
		elif elem.Name == u"COMPLEX8TimeSeries":
			tseries = lal.series.parse_COMPLEX8TimeSeries(elem)
			temp.append([tseries.name, tseries])
		elif elem.Name == u"COMPLEX16TimeSeries":
			tseries = lal.series.parse_COMPLEX16TimeSeries(elem)
			temp.append([tseries.name, tseries])

	for i in temp:
		if i[0] in result.keys():
			result[i[0]].append(i[1])
		else:
			result[i[0]] = [i[1]]

	assert result is not None, "xmldoc contains no LAL Series or LAL Series is unsupported"

	return result

# wrapper for writing snr series to URL
def write_url(xmldoc, filename, verbose = False):
	ligolw_utils.write_filename(xmldoc, filename, gz = filename.endswith(".gz"), verbose = verbose)

# wrapper for reading snr series from URL
def read_url(filename, contenthandler = SNRContentHandler, verbose = False):
	return ligolw_utils.load_url(filename, verbose = verbose, contenthandler = contenthandler)

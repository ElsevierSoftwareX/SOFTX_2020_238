"""
Short cutting gstlal inspiral pipeline to produce SNR for gstlal_svd_bank.
A gstlal-based direct matched filter in time domain is also implemented.
"""

import os
import sys
import shutil
import numpy
import threading

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstAudio', '1.0')
from gi.repository import GObject, Gst, GstAudio
GObject.threads_init()
Gst.init(None)

from gstlal import cbc_template_fir
from gstlal import datasource
from gstlal import inspiral
from gstlal import lloidparts
from gstlal import lvalert_helper
from gstlal import multirate_datasource
from gstlal import pipeparts
from gstlal import pipeio
from gstlal import reference_psd
from gstlal import simplehandler

import gwdatafind

import lal
from lal.utils import CacheEntry
from lal import LIGOTimeGPS
import lal.series
import lalsimulation as lalsim

from ligo.lw import array as ligolw_array
from ligo.lw import ligolw
from ligo.lw import param as ligolw_param
from ligo.lw import utils as ligolw_utils
from ligo.lw import lsctables
from ligo.gracedb import rest as gracedb

@ligolw_array.use_in
@ligolw_param.use_in
class SNRContentHandler(ligolw.LIGOLWContentHandler):
	pass

class SNR_Pipeline(object):
	def __init__(self, row_number, start, end, name = "gstlal_inspiral_SNR", verbose = False):
		self.pipeline = Gst.Pipeline(name = name)
		self.mainloop = GObject.MainLoop()
		self.handler = simplehandler.Handler(self.mainloop, self.pipeline)
		self.verbose = verbose
		self.lock = threading.Lock()
                self.row_number = row_number
                self.start = start
                self.end = end
		self.snr_info = {
			"epoch": None,
			"instrument": None,
			"deltaT": None,
			"data": [],
		}
                if self.start >= self.end:
                        raise ValueError("Start time must be less than end time.")

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
                para = {"name" : self.snr_info["instrument"],
                        "epoch" : self.snr_info["epoch"],
                        "deltaT" : self.snr_info["deltaT"],
                        "f0": 0,
                        "sampleUnits" : lal.DimensionlessUnit,
                        "length" : len(data)}
		if data.dtype == numpy.float32:
			tseries = lal.CreateREAL4TimeSeries(**para)
		elif data.dtype == numpy.float64:
			tseries = lal.CreateREAL8TimeSeries(**para)
		elif data.dtype == numpy.complex64:
			tseries = lal.CreateCOMPLEX8TimeSeries(**para)
		elif data.dtype == numpy.complex128:
			tseries = lal.CreateCOMPLEX16TimeSeries(**para)
		else:
			raise ValueError("unsupported type : %s " % data.dtype)

		tseries.data.data = data
		return tseries

	def get_snr_series(self, COMPLEX = False):
                assert self.snr_info["epoch"] is not None, "No SNRs are obtained, check your start time."
		gps_start = self.snr_info["epoch"].gpsSeconds + self.snr_info["epoch"].gpsNanoSeconds * 10.**-9
		gps = gps_start + numpy.arange(len(self.snr_info["data"])) * self.snr_info["deltaT"]

                if self.start - gps[0] < 0 or self.end - gps[-1] > 0:
                        raise ValueError("Invalid choice of start time or end time. The data spans from %f to %f." % (gps[0], gps[-1]))
                else:
                        s = abs(gps - self.start).argmin()
                        e = abs(gps - self.end).argmin()

                self.snr_info["epoch"] = gps[s]
                self.snr_info["data"] = self.snr_info["data"][s:e].T

		if self.row_number is None:
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
			self.snr_info["data"] = self.snr_info["data"][self.row_number]
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

                        buf = sample.get_buffer()
                        if buf.mini_object.flags & Gst.BufferFlags.GAP or buf.n_memory() == 0:
                                return Gst.FlowReturn.OK

                        # drop snrs that are irrelevant
                        cur_time_stamp = LIGOTimeGPS(0, sample.get_buffer().pts)
                        if self.start >= cur_time_stamp and self.end > cur_time_stamp:
                                # record the first timestamp closet to start time
                                self.snr_info["epoch"] = cur_time_stamp
                                # FIXME: check timestamps
                                self.snr_info["data"] = [pipeio.array_from_audio_sample(sample)]
                        elif self.start <= cur_time_stamp < self.end:
				self.snr_info["data"].append(pipeio.array_from_audio_sample(sample))
                        else:
                                Gst.FlowReturn.OK

			return Gst.FlowReturn.OK

class LLOID_SNR(SNR_Pipeline):
	def __init__(self, gw_data_source_info, bank, instrument, row_number, start, end, psd = None, psd_fft_length = 32, ht_gate_threshold = float("inf"), veto_segments = None, track_psd = False, width = 32, verbose = False):
		SNR_Pipeline.__init__(self, row_number, start, end, name = "gstlal_inspiral_lloid_snr", verbose = verbose)
		self.snr_info["instrument"] = instrument

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

	def __call__(self, COMPLEX = False):
		return self.get_snr_series(COMPLEX)

class FIR_SNR(SNR_Pipeline):
	def __init__(self, gw_data_source_info, template, instrument, rate, latency, start, end,  psd = None, psd_fft_length = 32, ht_gate_threshold = float("inf"), veto_segments = None, width = 32, track_psd = False, verbose = False):
		SNR_Pipeline.__init__(self, 0, start, end, name = "gstlal_inspiral_fir_snr", verbose = verbose)
		self.snr_info["instrument"] = instrument

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

	@staticmethod
	def make_template(template_table, template_psd, sample_rate, approximant, instrument, f_low, f_high = None, autocorrelation_length = None, verbose = False):
		row = [row for row in template_table]
		if len(row) != 1 :
			raise ValueError("Expecting only one template or cannot find any template.")

		template_psd = lal.series.read_psd_xmldoc(ligolw_utils.load_url(template_psd, contenthandler = lal.series.PSDContentHandler))
		if instrument not in set(template_psd):
			raise ValueError("No such instrument: %s in template psd: (%s)"% (instrument, ", ".join(set(template_psd))))

		# work around for building a single whitened template
		template_duration = lalsim.SimInspiralChirpTimeBound(f_low, row[0].mass1 * lal.MSUN_SI, row[0].mass2 * lal.MSUN_SI, 0., 0.)
		time_slice = numpy.array([(sample_rate, 0, template_duration)], dtype = [("rate", "int"),("begin", "float"), ("end", "float")])
		workspace = cbc_template_fir.templates_workspace(template_table, approximant, template_psd[instrument], f_low, time_slice, autocorrelation_length = None, fhigh = f_high)
		template, autocorrelation, sigma = workspace.make_whitened_template(row[0])

		return template, row[0].end

	def __call__(self, COMPLEX = False):
		return self.get_snr_series(COMPLEX)

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


#=============================================================================================
#
# 					Gracedb Events Utilities
#
#=============================================================================================

def svd_banks_from_event(gid, outdir = ".", save = True, verbose = False):
	gracedb_client = gracedb.GraceDb()
	coinc_xmldoc = lvalert_helper.get_coinc_xmldoc(gracedb_client, gid)
	eventid_trigger_dict = dict((row.event_id, row) for row in lsctables.SnglInspiralTable.get_table(coinc_xmldoc))

	assert len(set([row.template_id for row in eventid_trigger_dict.values()])) == 1, "Templates should have the same template_id."

	try:
		bank_urls = inspiral.parse_svdbank_string([row.value for row in lsctables.ProcessParamsTable.get_table(coinc_xmldoc) if row.param == "--svd-bank"].pop())
		banks_dict = inspiral.parse_bank_files(bank_urls, verbose = verbose)
	except IOError:
		sys.stderr.write("Files Not Found! Make sure you are on the LIGO-Caltech Computing Cluster or check if file exist.\nAbortting...\n")
		sys.exit()

	if save:
		try:
			for bank_url in bank_urls.values():
				outname =os.path.join(outdir, os.path.basename(bank_url))
				if verbose:
					sys.stderr.write("saving SVD bank file to %s  ...\n" % outname)
				shutil.copyfile(bank_url, outname)
		# FIXME: in python > 2.7, OSError will be raised if destination is not writable.
		except IOError as e:
			raise e

	# Just get one of the template bank from any instrument,
	# the templates should all have the same template_id because they are exact-matched.
	sub_bank_id = None
        for i, bank in enumerate(banks_dict.values()[0]):
                for j, row in enumerate(bank.sngl_inspiral_table):
                        if row.template_id == eventid_trigger_dict.values()[0].template_id:
                                sub_bank_id = i
                                row_number = j
                                break
                if sub_bank_id is not None:
                        break

	return banks_dict, sub_bank_id, row_number

def framecache_from_event(gid, observatories, frame_types, time_span = 1000, outdir = ".", filename = "frame.cache", verbose = False):
	assert time_span >= 1000., "Please use time_span larger or equal to 1000."

	obs2ifo = {"H": "H1", "L": "L1", "V": "V1"}

	observatories = set(observatories)
	frame_types = set(frame_types)

	if len(observatories) != len(frame_types):
		raise ValueError("Must have as many frame_types as observatories.")
	# FIXME: This is not reliable, have a better way to map frame_type to observatory?
	obs_type_dict = dict([(obs, frame_type) for obs in observatories for frame_type in frame_types if obs == frame_type[0]])

        gracedb_client = gracedb.GraceDb()
        coinc_xmldoc = lvalert_helper.get_coinc_xmldoc(gracedb_client, gid)
        eventid_trigger_dict = dict((row.ifo, row) for row in lsctables.SnglInspiralTable.get_table(coinc_xmldoc))
	channel_names_dict = dict([(row.value.split("=")[0], row.value) for row in lsctables.ProcessParamsTable.get_table(coinc_xmldoc) if row.param == "--channel-name"])

	gwdata_metavar_headers = ["instruments", "trigger_times", "gps_start_time", "gps_end_time", "channels_name"]
	gwdata_metavar_values = []
	urls = []
        for observatory, frame_type in obs_type_dict.items():
		trigger_time = eventid_trigger_dict[obs2ifo[observatory]].end
		gps_start_time = int(trigger_time - time_span)
		gps_end_time = int(trigger_time + time_span)
                gwdata_metavar_values.append((obs2ifo[observatory], trigger_time, gps_start_time, gps_end_time, channel_names_dict[obs2ifo[observatory]]))

		urls += gwdatafind.find_urls(observatory, frame_type, gps_start_time, gps_end_time)

	with open(os.path.join(outdir, "frame.cache"), "w") as cache:
		for url in urls:
			filename = str(CacheEntry.from_T050017(url))
			cache.write("%s\n" % filename)
			if verbose:
				sys.stderr.write("writing %s to %s\n" % (filename, os.path.join(outdir, "frame.cache")))
		if verbose:
			sys.stderr.write("Done.\n")

	return dict(zip(gwdata_metavar_headers, zip(*gwdata_metavar_values)))

def psd_from_event(gid, outdir = ".", save = True, filename = "psd.xml.gz", verbose = False):
	gracedb_client = gracedb.GraceDb()
	psd_fileobj = lvalert_helper.get_filename(gracedb_client, gid, filename)
	xmldoc = ligolw_utils.load_fileobj(psd_fileobj, contenthandler = lal.series.PSDContentHandler)
	if save:
		if verbose:
			sys.stderr.write("saving psd file to %s ...\n" % os.path.join(outdir, filename))
		ligolw_utils.write_filename(xmldoc, filename, gz = filename.endswith("gz"))
	return lal.series.read_psd_xmldoc(xmldoc)

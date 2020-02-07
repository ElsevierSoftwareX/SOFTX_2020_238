"""
Short cutting gstlal inspiral pipeline to produce SNR for gstlal_svd_bank.
A gstlal-based direct matched filter in time domain is also implemented.
"""

from collections import defaultdict
import numpy
import os
import shutil
import sys
import threading

from gstlal import cbc_template_fir
from gstlal import templates
from gstlal import inspiral
from gstlal import lloidhandler
from gstlal import lvalert_helper
from gstlal import pipeio
from gstlal import streamthinca
from gstlal import simplehandler
from gstlal import svd_bank
from gstlal.lloidhandler import SegmentsTracker
from gstlal.snglinspiraltable import GSTLALSnglInspiral as SnglInspiral

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstAudio', '1.0')
from gi.repository import GObject, Gst, GstAudio
GObject.threads_init()
Gst.init(None)

import gwdatafind

from lal import LIGOTimeGPS
from lal.utils import CacheEntry
import lal
import lal.series
import lalsimulation as lalsim

from ligo.gracedb import rest as gracedb
from ligo.lw import array as ligolw_array
from ligo.lw import ligolw
from ligo.lw import lsctables
from ligo.lw import param as ligolw_param
from ligo.lw import utils as ligolw_utils

@ligolw_array.use_in
@ligolw_param.use_in
class SNRContentHandler(ligolw.LIGOLWContentHandler):
	pass


@ligolw_param.use_in
@ligolw_array.use_in
@lsctables.use_in
class ContentHandler(ligolw.LIGOLWContentHandler):
        pass

#=============================================================================================
#
#				Signal to Noise Ratio Document
#
#=============================================================================================
class SignalNoiseRatioDocument(object):
	"""LIGO_LW xml document for Signal to Noise Ratio.

	This xml document contains the SNRs timeseries and their corresponding templates
	autocorrelation. Some meta data are recorded in the xml document for
	"""
        def __init__(self, snrdict, banks_dict, verbose=False):
		self.verbose = verbose
		self.snrdict = snrdict
		self.banks_dict = banks_dict
		self.bank_number = snrdict.values()[0].bank_number
		self.bank_id = banks_dict.values()[0][self.bank_number].bank_id
		self.template_ids = [row.template_id for row in snrdict.values()[0].sngl_inspiral_table]

	def write_output_url(self, outdir, row_number=None, root_name="gstlal_inspiral_snr"):
		"""Writing the LIGO_LW xmldoc to disk.

		Args:
		    outdir (str): The output diretory.
		    row_number (int, default=None): The row number of the SNR to be outputed. Default=None is to output all.
		    root_name (str, default="gstlal_inspiral_snr"): The root name of the xml document.

		Return:
		    xmldoc: The file object representing the xmldoc.

		"""
		for instrument, snrs in self.snrdict.items():
			# create root
			xmldoc = ligolw.Document()
			root = xmldoc.appendChild(ligolw.LIGO_LW())
			root.Name = root_name
			root.appendChild(ligolw_param.Param.from_pyvalue('bank_filename', self.banks_dict[instrument][0].template_bank_filename))
			root.appendChild(ligolw_param.Param.from_pyvalue('bank_number', self.bank_number))
			root.appendChild(ligolw_param.Param.from_pyvalue('bank_id', self.bank_id))
			root.appendChild(ligolw_param.Param.from_pyvalue('instrument', instrument))

			# add SNR and autocorrelation branches
			self._append_content(root, snrs, instrument, row_number=row_number)

			if row_number is None:
				outname = "%s-%s_SNR_%d-%d-%d.xml.gz" % (instrument, snrs.method, snrs.bank_number, snrs.start, snrs.duration)
				write_url(xmldoc, os.path.join(outdir, outname), verbose = self.verbose)
			else:
				outname = "%s-%s_SNR_%d_%d-%d-%d.xml.gz" % (instrument, snrs.method, snrs.bank_number, row_number, snrs.start, snrs.duration)
				write_url(xmldoc, os.path.join(outdir, outname), verbose = self.verbose)
		return xmldoc

	def _append_content(self, root, snrs, instrument, row_number=None):
		"""For internal use only."""
		if row_number is None:
			for row, template_id, snr in zip(range(len(snrs)), self.template_ids, snrs):
				branch = root.appendChild(ligolw.LIGO_LW())
				branch.Name = "SNR_and_autocorrelation"

				# append timeseries and templates autocorrelation
				if snr.data.data.dtype == numpy.float32:
					tseries = branch.appendChild(lal.series.build_REAL4TimeSeries(snr))
				elif snr.data.data.dtype == numpy.float64:
					tseries = branch.appendChild(lal.series.build_REAL8TimeSeries(snr))
				elif snr.data.data.dtype == numpy.complex64:
					tseries = branch.appendChild(lal.series.build_COMPLEX8TimeSeries(snr))
				elif snr.data.data.dtype == numpy.complex128:
					tseries = branch.appendChild(lal.series.build_COMPLEX16TimeSeries(snr))
				else:
					raise ValueError("unsupported type : %s" % snr.data.data.dtype)

				# append template_id and autocorrelation_bank
				branch.appendChild(ligolw_param.Param.from_pyvalue('template_id', template_id))
				branch.appendChild(ligolw_array.Array.build('autocorrelation_bank', self.banks_dict[instrument][self.bank_number].autocorrelation_bank[row]))
		else:
			branch = root.appendChild(ligolw.LIGO_LW())
			branch.Name = "SNR_and_autocorrelation"

			# append timeseries and template autocorrelation
			snr = snrs[row_number]
			if snr.data.data.dtype == numpy.float32:
				tseries = branch.appendChild(lal.series.build_REAL4TimeSeries(snr))
			elif snr.data.data.dtype == numpy.float64:
				tseries = branch.appendChild(lal.series.build_REAL8TimeSeries(snr))
			elif snr.data.data.dtype == numpy.complex64:
				tseries = branch.appendChild(lal.series.build_COMPLEX8TimeSeries(snr))
			elif snr.data.data.dtype == numpy.complex128:
				tseries = branch.appendChild(lal.series.build_COMPLEX16TimeSeries(snr))
			else:
				raise ValueError("unsupported type : %s" % snr.data.data.dtype)

			# append template_id and autocorrelation_bank
			branch.appendChild(ligolw_param.Param.from_pyvalue('template_id', self.template_ids[row_number]))
			branch.appendChild(ligolw_array.Array.build('autocorrelation_bank', self.banks_dict[instrument][self.bank_number].autocorrelation_bank[row_number]))

		return branch

#=============================================================================================
#
#					Pipeline Handler
#
#=============================================================================================
class SNR(object):
	"""An data interface between the SNRPipelineHandler and the SNR pipeline.

	This is a class that defines the approximate start time and end time for which
	the SNR should be collected.
	"""
	def __init__(self, start, end, instrument, banks, bank_number=0, method="LLOID"):
		if start >= end:
			raise ValueError("Start time must be less than end time.")
		self.method = method
		self.bank_number = bank_number
		self.sngl_inspiral_table = banks[bank_number].sngl_inspiral_table
		self.s = start
		self.e = end
		self.epoch = None
		self.deltaT = None
		self.data = []

		self._start = None
		self._end = None
		self._duration = None
		self._instrument = instrument

	@property
	def start(self):
		"""float: The approximate start time of all SNRs timeseries based on the buffer timestamp.

		Note:
			This start time is not precise because the 'end_time' of each template was not added here.
			Please refer to the epoch of each timeseries object for precise start time.
		"""
		try:
			self.finish()
		except:
			pass

		return round(self._start)

	@property
	def end(self):
		"""float: The approximate end time of all SNRs timeseries calculated from buffer timestamp and data.

		Note:
			This end time is not precise because the 'end_time' of each template was not added here.
			Please refer to the epoch of each timeseries object for precise end time.
		"""
		try:
			self.finish()
		except:
			pass

		return round(self._end)

	@property
	def duration(self):
		"""float: The approximate duration of all SNRs timeseries."""
		try:
			self.finish()
		except:
			pass

		return self.end - self.start

	@property
	def instrument(self):
		"""str: The instrument of the SNR."""
		return str(self._instrument)

	def __getitem__(self, index):
		""":obj:`LAL series`: Allow access SNRs timeseries object by index."""
		try:
			self.finish()
		except:
			pass

		return self.data[index]

	def __len__(self):
		"""int: Return the number of SNRs timeseries."""
		return len(self.data)

	def finish(self, COMPLEX = False):
		"""Settling down the collected SNRs and parse them to LAL series.

		Note:
			This method can only be called once.

		Args:
			COMPLEX (bool): True if required complex SNR, False otherwise.

		"""
		assert self.epoch is not None, "No SNRs are obtained."
		tmp_snrs = numpy.concatenate(numpy.array(self.data), axis = 0)
		gps_start = self.epoch.gpsSeconds + self.epoch.gpsNanoSeconds * 10.**-9
		gps = gps_start + numpy.arange(len(tmp_snrs)) * self.deltaT

		if self.s - gps[0] < 0 or self.e - gps[-1] > 0:
			raise ValueError("Invalid choice of start time or end time. The data spans from %f to %f." % (gps[0], gps[-1]))
		else:
			s = abs(gps - self.s).argmin()
			e = abs(gps - self.e).argmin()

		# update data and epoch
		self._start = self.epoch = gps[s]
		self._end = gps[e]
		tmp_snrs = tmp_snrs[s:e].T if COMPLEX else numpy.abs(tmp_snrs[s:e].T)

		# parse data to tseries object
		self.data = [self._make_series(array, self.epoch + row.end) for array, row in zip(tmp_snrs, self.sngl_inspiral_table)]

		# .finish() again is forbidden
		def finish():
			raise NotImplementedError("Cannot .finish() because SNR has been .finish()ed.")
		self.finish = finish

	def _make_series(self, array, epoch):
		"""For internal use only."""
		para = {"name" : self.instrument,
			"epoch" : epoch,
			"deltaT" : self.deltaT,
			"f0": 0,
			"sampleUnits" : lal.DimensionlessUnit,
			"length" : len(array)}
		if array.dtype == numpy.float32:
			tseries = lal.CreateREAL4TimeSeries(**para)
		elif array.dtype == numpy.float64:
			tseries = lal.CreateREAL8TimeSeries(**para)
		elif array.dtype == numpy.complex64:
			tseries = lal.CreateCOMPLEX8TimeSeries(**para)
		elif array.dtype == numpy.complex128:
			tseries = lal.CreateCOMPLEX16TimeSeries(**para)
		else:
			raise ValueError("unsupported type : %s " % array.dtype)

		tseries.data.data = array
		return tseries

class SNRHandlerMixin(object):
	def __init__(self, *arg, **kwargs):
		super(SNRHandlerMixin, self).__init__(*arg, **kwargs)

	def appsink_new_snr_buffer(self, elem):
		"""Callback function for SNR appsink."""
		with self.lock:
			# Note: be sure to set property="%s" % instrument, for appsink element
			instrument = elem.name

			sample = elem.emit("pull-sample")
			if sample is None:
				return Gst.FlowReturn.OK

			success, rate = sample.get_caps().get_structure(0).get_int("rate")
			assert success == True

			if self.snr_document.snrdict[instrument].deltaT is None:
				self.snr_document.snrdict[instrument].deltaT = 1. / rate
			else:
				# sampling rate should not be changing
				assert self.snr_document.snrdict[instrument].deltaT == 1. / rate, "Data has different sampling rate."

			buf = sample.get_buffer()
			if buf.mini_object.flags & Gst.BufferFlags.GAP or buf.n_memory() == 0:
				return Gst.FlowReturn.OK

			cur_time_stamp = LIGOTimeGPS(0, sample.get_buffer().pts)

			if self.snr_document.snrdict[instrument].s >= cur_time_stamp and self.snr_document.snrdict[instrument].e > cur_time_stamp:
				# record the first timestamp closet to start time
				self.snr_document.snrdict[instrument].epoch = cur_time_stamp
				self.snr_document.snrdict[instrument].data = [pipeio.array_from_audio_sample(sample)]
			elif self.snr_document.snrdict[instrument].s <= cur_time_stamp < self.snr_document.snrdict[instrument].e:
				self.snr_document.snrdict[instrument].data.append(pipeio.array_from_audio_sample(sample))
			else:
				Gst.FlowReturn.OK

			return Gst.FlowReturn.OK

	def write_snrs(self, outdir, row_number=None, COMPLEX=False):
		"""Writing SNRs timeseries to LIGO_LW xml files."""
		for snrs in self.snr_document.snrdict.values():
			# make sure to call .finish()
			snrs.finish(COMPLEX)
		self.snr_document.write_output_url(outdir, row_number=row_number)


class SimpleSNRHandler(SNRHandlerMixin, simplehandler.Handler):
	"""Simple SNR pipeline handler.

	This is the SNR pipeline handler derived from simplehandler. It
	only implements the controls for collecting SNR timeseries.

	"""
	def __init__(self, pipeline, mainloop, snr_document, verbose=False):
		super(SimpleSNRHandler, self).__init__(mainloop, pipeline)
		self.lock = threading.Lock()
		self.snr_document = snr_document
		self.verbose = verbose

class Handler(SNRHandlerMixin, lloidhandler.Handler):
	"""Simplified version of lloidhandler.Handler.

	This is the SNR pipeline handler derived from lloidhandler. In
	addition to the control for collecting SNR timeseries, it
	implements controls for trigger generator.

	"""
	def __init__(self, snr_document, verbose=False):
		self.lock = threading.Lock()
		self.snr_document = snr_document
		self.verbose = verbose

	# Explictly delay the class initialization.
	def init(self, mainloop, pipeline, coincs_document, rankingstat, horizon_distance_func, gracedbwrapper, zerolag_rankingstatpdf_url=None, rankingstatpdf_url=None, ranking_stat_output_url=None, ranking_stat_input_url=None, likelihood_snapshot_interval=None, sngls_snr_threshold=None, FAR_trialsfactor=1.0, verbose=False):
		super(Handler, self).__init__(
			mainloop,
			pipeline,
			coincs_document,
			rankingstat,
			horizon_distance_func,
			gracedbwrapper = gracedbwrapper,
			zerolag_rankingstatpdf_url = zerolag_rankingstatpdf_url,
			rankingstatpdf_url = rankingstatpdf_url,
			ranking_stat_output_url = ranking_stat_output_url,
			ranking_stat_input_url = ranking_stat_input_url,
			likelihood_snapshot_interval = likelihood_snapshot_interval,
			sngls_snr_threshold = sngls_snr_threshold,
			FAR_trialsfactor = FAR_trialsfactor,
			kafka_server = None,
			cluster = True,
			tag = "0000",
			verbose = verbose
		)

#=============================================================================================
#
#					Template Utilities
#
#=============================================================================================
def write_simplified_sngl_inspiral_table(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, instrument, approximant, filename=None):
	"""Writing a simplified sngl_inspiral_table containing only one template.

	Args:
		m1 (float): mass1.
		m2 (float): mass2.
		s1x (float): spin 1 x-axis.
		s1y (float): spin 1 y-axis.
		s1z (float): spin 1 z-axis.
		s2x (float): spin 2 x-axis.
		s2y (float): spin 2 y-axis.
		s2z (float): spin 2 z-axis.
		instrument (str): The instrument for the template.
		approximant (str): The approximant used to simulate the waveform.
		filename (str, default=None): The output filename.

	Return:
		The file object representing the xmldoc.

	"""
	# Check if it is valid approximant
	templates.gstlal_valid_approximant(approximant)

	xmldoc = ligolw.Document()
	root = xmldoc.appendChild(ligolw.LIGO_LW())

	table = lsctables.New(lsctables.SnglInspiralTable)
	rows = table.RowType()

	# set all slots to impossible/dummy value
	for t, c in zip(table.columntypes, table.columnnames):
		if t == u"real_4" or t == u"real_8":
			rows.__setattr__(c, 0)
		elif t == u"int_4s" or t == u"int_8s":
			rows.__setattr__(c, 0)
		elif t == u"lstring":
			rows.__setattr__(c, "")
		else:
			rows.__setattr__(c, None)

	rows.mass1 = m1
	rows.mass2 = m2
	rows.mtotal = m1 + m2
	rows.mchirp = (m1 * m2)**0.6 / (m1 + m2)**0.2
	rows.spin1x = s1x
	rows.spin1y = s1y
	rows.spin1z = s1z
	rows.spin2x = s2x
	rows.spin2y = s2y
	rows.spin2z = s2z
	rows.ifo = instrument

	table.append(rows)
	root.appendChild(table)

	#FIXME: do something better than this
	root.appendChild(ligolw_param.Param.from_pyvalue("approximant", approximant))

	if filename is not None:
		ligolw_utils.write_filename(xmldoc, filename, gz = filename.endswith("gz"))

	return xmldoc

#FIXME: perhaps put it into cbc_template_fir.py? As an extra way to build template bank (not just svd bank).
def generate_templates(table, approximant, psd, f_low, time_slice, autocorrelation_length = None, f_high = None, verbose = False):
	"""Generate whiten templates from sngl_inspiral_table."""
	# Create workspace for making template bank
	workspace = cbc_template_fir.templates_workspace(table, approximant, psd, f_low, time_slice, autocorrelation_length = autocorrelation_length, fhigh = f_high)

	# Make sure it is just a big slice from start to end
	assert len(time_slice) == 1, "Should contain only one slice."

	if workspace.autocorrelation_length is not None:
		if not (workspace.autocorrelation_length % 2):
			raise ValueError("autocorrelation_length must be odd (got %d)" % autocorrelation_length)
		autocorrelation_bank = numpy.zeros((len(table), autocorrelation_length), dtype = "cdouble")
		autocorrelation_mask = cbc_template_fir.compute_autocorrelation_mask(autocorrelation_bank)
	else:
		autocorrelation_bank = None
		autocorrelation_mask = None

	templates = []
	sigmasq = []
	for i, row in enumerate(table):
		if verbose:
			sys.write.stderr("generating template %d/%d:  m1 = %g, m2 = %g, s1x = %g, s1y = %g, s1z = %g, s2x = %g, s2y = %g, s2z = %g\n" % (i + 1, len(table), row.mass1, row.mass2, row.spin1x, row.spin1y, row.spin1z, row.spin2x, row.spin2y, row.spin2z))

		template, autocorrelation, sigma = workspace.make_whitened_template(row)
		templates.append(template)
		sigmasq.append(sigma)

		if autocorrelation is not None:
			autocorrelation_bank[i, ::-1] = numpy.concatenate((autocorrelation[-(autocorrelation_length // 2):], autocorrelation[:(autocorrelation_length // 2  + 1)]))

	return numpy.array(templates), autocorrelation_bank, autocorrelation_mask, sigmasq, workspace.psd

class Bank(object):
	"""
	FIXME: This is a class used to mimic the behavior of the svd bank object.
	"""
	def __init__(self, bank_xmldoc, psd, rate, f_low, f_high = None, autocorrelation_length = None, verbose = False):
		self.bank_id = None
		self.sample_rate = rate
		self.template_bank_filename = None
		self.processed_psd = None
		self.horizon_factors = None
		self.horizon_distance_func = lambda psd, snr: [100, None]
		self.sngl_inspiral_table = lsctables.SnglInspiralTable.get_table(bank_xmldoc)

		# Until the correct way to set the time_slice for multiple templates is known, generating more than one
		# template is forbidden.  #FIXME: it looks like allowing this, the SNRs is unaffected
		# assert len(self.sngl_inspiral_table) == 1

		# FIXME: still correct if we have more than one templates?
		template = min(self.sngl_inspiral_table, key = lambda row: row.mchirp)
		self.template_duration = lalsim.SimInspiralChirpTimeBound(f_low, template.mass1 * lal.MSUN_SI, template.mass2 * lal.MSUN_SI, 0., 0.)
		self.time_slice = numpy.array([(rate, 0, self.template_duration)], dtype = [("rate", "int"),("begin", "float"), ("end", "float")])

		self.templates, self.autocorrelation_bank, self.autocorrelation_mask, self.sigmasq, self.processed_psd = generate_templates(
			self.sngl_inspiral_table,
			ligolw_param.get_pyvalue(bank_xmldoc, "approximant"),
			psd,
			f_low,
			self.time_slice,
			f_high = f_high,
			autocorrelation_length = autocorrelation_length,
			verbose = verbose
		)

	def get_rates(self):
		return set([self.sample_rate])

	@classmethod
	def from_url(cls, url, verbose = False):
		xmldoc = ligolw_utils.load_url(url, contenthandler = ContentHandler, verbose = verbose)

		banks = []

		for root in (elem for elem in xmldoc.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute("Name") and elem.Name == "gstlal_template_bank"):
			bank = cls.__new__(cls)

			bank.bank_id = ligolw_param.get_pyvalue(root, "bank_id")
			bank.sample_rate = ligolw_param.get_pyvalue(root, "sample_rate")
			bank.processed_psd = None
			bank.sngl_inspiral_table = lsctables.SnglInspiralTable.get_table(root)
			bank.template_bank_filename = ligolw_param.get_pyvalue(root, "template_bank_filename")
			bank.sigmasq = ligolw_array.get_array(root, "sigmasq").array
			bank.templates = ligolw_array.get_array(root, "templates").array
			bank.autocorrelation_bank = ligolw_array.get_array(root, "autocorrelation_bank").array
			bank.autocorrelation_mask = ligolw_array.get_array(root, "autocorrelation_mask").array
			bank.horizon_factors = dict((row.template_id, sigmasq**.5) for row, sigmasq in zip(bank.sngl_inspiral_table, bank.sigmasq))

			banks.append(bank)

		min_template_id, horizon_distance_func = svd_bank.horizon_distance_func(banks)
		horizon_norm, = (bank.horizon_factors[row.template_id] for row in bank.sngl_inspiral_table for bank in banks if row.template_id == min_template_id)
		for bank in banks:
			bank.horizon_distance_func = horizon_distance_func
			bank.horizon_factors = dict((tid, f / horizon_norm) for (tid, f) in bank.horizon_factors.items())

		return banks

def build_bank(bank_url, psd_file, sample_rate, f_low, f_high = None, autocorrelation_length = None, verbose = False):
	"""Return an instance of a Bank class."""
	bank_xmldoc = ligolw_utils.load_url(bank_url, contenthandler = ContentHandler, verbose = verbose)
	psd = lal.series.read_psd_xmldoc(ligolw_utils.load_url(psd_file, contenthandler = lal.series.PSDContentHandler))

	assert numpy.log2(sample_rate).is_integer(), "sample_rate can only be power of two."

	bank = Bank(bank_xmldoc, psd[lsctables.SnglInspiralTable.get_table(bank_xmldoc)[0].ifo], sample_rate, f_low, f_high, autocorrelation_length = autocorrelation_length, verbose = verbose)
	bank.template_bank_filename = bank_url
	#FIXME: dummy bank_id
	bank.bank_id = 0

	return bank

def write_bank(filename, banks, verbose = False):
	"""Write template bank to LIGO_LW xml file."""
	xmldoc = ligolw.Document()
	head = xmldoc.appendChild(ligolw.LIGO_LW())
	head.Name = u"gstlal_template_bank"

	for bank in banks:
		cloned_table = bank.sngl_inspiral_table.copy()
		cloned_table.extend(bank.sngl_inspiral_table)
		head.appendChild(cloned_table)

		head.appendChild(ligolw_param.Param.from_pyvalue('template_bank_filename', bank.template_bank_filename))
		head.appendChild(ligolw_param.Param.from_pyvalue('sample_rate', bank.sample_rate))
		head.appendChild(ligolw_param.Param.from_pyvalue('bank_id', bank.bank_id))
		head.appendChild(ligolw_array.Array.build('templates', bank.templates))
		head.appendChild(ligolw_array.Array.build('autocorrelation_bank', bank.autocorrelation_bank))
		head.appendChild(ligolw_array.Array.build('autocorrelation_mask', bank.autocorrelation_mask))
		head.appendChild(ligolw_array.Array.build('sigmasq', numpy.array(bank.sigmasq)))

	ligolw_utils.write_filename(xmldoc, filename, gz = filename.endswith('.gz'), verbose = verbose)

def parse_bank_files(bank_urls, verbose = False):
	"""Parse a dictionary of bank urls key by instrument into a dictionary of
	bank instance key by instrument."""
	banks_dict = {}
	for instrument, url in bank_urls.items():
		banks_dict[instrument] = (Bank.from_url(url))

	return banks_dict

#=============================================================================================
#
#					Output Utilities
#
#=============================================================================================
def read_xmldoc(xmldoc, root_name = u"gstlal_inspiral_snr"):
	if root_name is not None:
		root, = (elem for elem in xmldoc.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") if elem.Name == root_name)

	instrument = ligolw_param.get_pyvalue(root, "instrument")

	snrdict = defaultdict(list)
	autocorrelation_dict = defaultdict(list)
	for elem in (elem for elem in root.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") and elem.Name == "SNR_and_autocorrelation"):
		# get the time series
		snr_elem, = (elem for elem in elem.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") and elem.Name != "SNR_and_autocorrelation")
		if snr_elem.Name == u"REAL4TimeSeries":
			tseries = lal.series.parse_REAL4TimeSeries(snr_elem)
			snrdict[instrument].append(tseries)
		elif snr_elem.Name == u"REAL8TimeSeries":
			tseries = lal.series.parse_REAL8TimeSeries(snr_elem)
			snrdict[instrument].append(tseries)
		elif snr_elem.Name == u"COMPLEX8TimeSeries":
			tseries = lal.series.parse_COMPLEX8TimeSeries(snr_elem)
			snrdict[instrument].append(tseries)
		elif snr_elem.Name == u"COMPLEX16TimeSeries":
			tseries = lal.series.parse_COMPLEX16TimeSeries(snr_elem)
			snrdict[instrument].append(tseries)

		autocorrelation_dict[instrument].append(ligolw_array.get_array(elem, "autocorrelation_bank").array)

	assert snrdict is not None, "xmldoc contains no LAL Series or LAL Series is unsupported"

	return snrdict, autocorrelation_dict

# wrapper for writing snr series to URL
def write_url(xmldoc, filename, verbose = False):
	ligolw_utils.write_filename(xmldoc, filename, gz = filename.endswith(".gz"), verbose = verbose)

# wrapper for reading snr series from URL
def read_url(filename, contenthandler = SNRContentHandler, verbose = False):
	return ligolw_utils.load_url(filename, verbose = verbose, contenthandler = contenthandler)

#=============================================================================================
#
#					Gracedb Events Utilities
#
#=============================================================================================
def scan_svd_banks_for_row(coinc_xmldoc, banks_dict):
	"""Scan the sub bank id and row number of an event template from coinc.xml file.

	Args:
		coinc_xmldoc (:obj: `xmldoc`): The coinc.xml file from gracedb.
		banks_dict (dict): A dictionary of SVD banks key by instrument containing the event template.

	Returns:
		bank_number (int)
		row_number (int)

	"""
	eventid_trigger_dict = dict((row.ifo, row) for row in lsctables.SnglInspiralTable.get_table(coinc_xmldoc))

	assert len(set([row.template_id for row in eventid_trigger_dict.values()])) == 1, "Templates should have the same template_id."

	bank_number = None
	row_number = None
	for i, bank in enumerate(banks_dict.values()[0]):
		for j, row in enumerate(bank.sngl_inspiral_table):
			if row.template_id == eventid_trigger_dict.values()[0].template_id:
				bank_number = i
				row_number = j
				break
		if bank_number is not None:
			break
	assert bank_number is not None, "Cannot find the template listed in the coinc.xml."
	return bank_number, row_number

def svd_banks_from_event(gid, outdir=".", save=False, verbose=False):
	"""Find location of the SVD banks from gracedb event id.

	Args:
		gid (str): The gracedb event id.
		outdir (str, default="."): The output directory.
		save (bool, default=False): True if want to save the file, False otherwise.
		verbose (bool, default=False): Be verbose.

	Returns:
		banks_dict (dict of :obj: `Bank`): The SVD banks for an event key by instrument.
		bank_number (int)
		row_number (int)

	"""
	gracedb_client = gracedb.GraceDb()
	coinc_xmldoc = lvalert_helper.get_coinc_xmldoc(gracedb_client, gid)

	try:
		path = [row.value for row in lsctables.ProcessParamsTable.get_table(coinc_xmldoc) if row.param == "--gracedb-service-url"]
		bank_urls = inspiral.parse_svdbank_string([row.value for row in lsctables.ProcessParamsTable.get_table(coinc_xmldoc) if row.param == "--svd-bank"].pop())
		if path is not None:
			path = path.pop()
			for ifo, bank_url in bank_urls.items():
				bank_urls[ifo] = os.path.join(path, bank_url)
		banks_dict = inspiral.parse_bank_files(bank_urls, verbose = verbose)
	except IOError:
		sys.stderr.write("Files Not Found! Make sure you are on the LIGO-Caltech Computing Cluster or check if files exist.\nAbortting...\n")
		sys.exit()

	if save:
		try:
			for bank_url in bank_urls.values():
				outname = os.path.join(outdir, os.path.basename(bank_url))
				if verbose:
					sys.stderr.write("saving SVD bank files to %s  ...\n" % outname)
				shutil.copyfile(bank_url, outname)
		# FIXME: in python > 2.7, OSError will be raised if destination is not writable.
		except IOError as e:
			raise e

	# Just get one of the template bank from any instrument,
	# the templates should all have the same template_id because they are exact-matched.
	bank_number, row_number = scan_svd_banks_for_row(coinc_xmldoc, banks_dict)

	return banks_dict, bank_number, row_number

def framecache_from_event(gid, observatories, frame_types, time_span = 500, outdir = ".", filename = "frame.cache", verbose = False):
	"""Get the frame cache for an event given the gracedb event id.

	Args:
		gid (str): The gracedb event id.
		observatories (list): See gwdatafind.
		frame_type (list): See gwdatafind.
		time_span (float): The time span before and after the trigger time.
		outdir (str, default="."): The output directory.
		filename (str, default="frame.cache"): The output filename.
		verbose (bool): Be verbose.

	Returns:
		Dictionary of instruments, trigger_times, gps_start_time,
		gps_end_time, channels_name.

	"""
	assert time_span >= 500., "Please use time_span larger or equal to 500."

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

def psd_from_event(gid, outdir = ".", save = False, filename = "psd.xml.gz", verbose = False):
	"""Get the psd.xml.gz given a gracedb event id.

	Args:
		gid (str): The gracedb event id.
		outdir (str, default="."): The output directory.
		filename (str, default="psd.xml.gz"): The output filename.
		save (bool, default=False): True if want to save the file, False otherwise.
		verbose (bool, default=False): Be verbose.

	Returns:
		A dictionary of complex LAL Series key by instrument.

	"""
	gracedb_client = gracedb.GraceDb()
	psd_fileobj = lvalert_helper.get_filename(gracedb_client, gid, filename)
	xmldoc = ligolw_utils.load_fileobj(psd_fileobj, contenthandler = lal.series.PSDContentHandler)
	if save:
		if verbose:
			sys.stderr.write("saving psd file to %s ...\n" % os.path.join(outdir, filename))
		ligolw_utils.write_filename(xmldoc, filename, gz = filename.endswith("gz"))
	return lal.series.read_psd_xmldoc(xmldoc)

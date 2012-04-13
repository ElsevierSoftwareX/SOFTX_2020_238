#
# Copyright (C) 2009-2011  Kipp Cannon, Chad Hanna, Drew Keppel
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


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


import itertools
import numpy
import os
from scipy import random
from scipy import stats
import StringIO
import subprocess
try:
	import sqlite3
except ImportError:
        # pre 2.5.x
	from pysqlite2 import dbapi2 as sqlite3
import sys
import threading
import time
from collections import deque
import resource

try:
	from ligo import gracedb
except ImportError:
	print >>sys.stderr, "warning: gracedb import failed, gracedb uploads disabled"

from glue import iterutils
from glue import segments
from glue.ligolw import ligolw
from glue.ligolw import lsctables
from glue.ligolw import utils
from glue.ligolw.utils import ligolw_add
from glue.ligolw.utils import process as ligolw_process
from pylal.datatypes import LIGOTimeGPS
from pylal.date import XLALUTCToGPS
from pylal.xlal.datatypes.snglinspiraltable import from_buffer as sngl_inspirals_from_buffer
from pylal import ligolw_burca_tailor
from pylal import ligolw_tisi
from pylal import rate
from gstlal import bottle
from gstlal import streamthinca
from gstlal import svd_bank
from gstlal import far

lsctables.LIGOTimeGPS = LIGOTimeGPS


#
# =============================================================================
#
#                                     Misc
#
# =============================================================================
#


def channel_dict_from_channel_list(channel_list):
	"""
	given a list of channels like this ["H1=LSC-STRAIN",
	H2="SOMETHING-ELSE"] produce a dictionary keyed by ifo of channel
	names.  The default values are LSC-STRAIN for all detectors
	"""

	channel_dict = {"H1" : "LSC-STRAIN", "H2" : "LSC-STRAIN", "L1" : "LSC-STRAIN", "V1" : "LSC-STRAIN", "G1" : "LSC-STRAIN", "T1" : "LSC-STRAIN"}

	for channel in channel_list:
		ifo = channel.split("=")[0]
		chan = "".join(channel.split("=")[1:])
		channel_dict[ifo] = chan

	return channel_dict


def pipeline_channel_list_from_channel_dict(channel_dict):
	"""
	produce a string of channel name arguments suitable for a pipeline.py
	program that doesn't technically allow multiple options. For example
	--channel-name=H1=LSC-STRAIN --channel-name=H2=LSC-STRAIN
	"""

	outstr = ""
	for i, ifo in enumerate(channel_dict):
		if i == 0:
			outstr += "%s=%s " % (ifo, channel_dict[ifo])
		else:
			outstr += "--channel-name=%s=%s " % (ifo, channel_dict[ifo])

	return outstr


def parse_banks(bank_string):
	"""
	parses strings of form 
	
	H1:bank1.xml,H2:bank2.xml,L1:bank3.xml,H2:bank4.xml,... 
	
	into a dictionary of lists of bank files.
	"""
	out = {}
	if bank_string is None:
		return out
	for b in bank_string.split(','):
		ifo, bank = b.split(':')
		out.setdefault(ifo, []).append(bank)
	return out

def parse_bank_files(svd_banks, verbose, snr_threshold = None):
	"""
	given a dictionary of lists of svd template bank file names parse them
	into a dictionary of bank classes
	"""

	banks = {}

	for instrument, files in svd_banks.items():
		for n, filename in enumerate(files):
			# FIXME over ride the file name stored in the bank file with
			# this file name this bank I/O code needs to be fixed
			bank = svd_bank.read_bank(filename, verbose = verbose)
			bank.template_bank_filename = filename
			bank.logname = "%sbank%d" % (instrument,n)
			bank.number = n
			banks.setdefault(instrument,[]).append(bank)
			if snr_threshold is not None:
				bank.snr_threshold = snr_threshold

	return banks


def connect_appsink_dump_dot(pipeline, appsinks, basename, verbose = False):
	
	"""
	add a signal handler to write a pipeline graph upon receipt of the
	first trigger buffer.  the caps in the pipeline graph are not fully
	negotiated until data comes out the end, so this version of the graph
	shows the final formats on all links
	"""

	class AppsinkDumpDot(object):
		# data shared by all instances
		# number of times execute method has been invoked, and a mutex
		n_lock = threading.Lock()
		n = 0

		def __init__(self, pipeline, write_after, basename, verbose = False):
			self.pipeline = pipeline
			self.handler_id = None
			self.write_after = write_after
			self.filestem = "%s.%s" % (basename, "TRIGGERS")
			self.verbose = verbose

		def execute(self, elem):
			self.n_lock.acquire()
			type(self).n += 1
			if self.n >= self.write_after:
				pipeparts.write_dump_dot(self.pipeline, self.filestem, verbose = self.verbose)
			self.n_lock.release()
			elem.disconnect(self.handler_id)

	for sink in appsinks:
		appsink_dump_dot = AppsinkDumpDot(pipeline, len(appsinks), basename = basename, verbose = verbose)
		appsink_dump_dot.handler_id = sink.connect_after("new-buffer", appsink_dump_dot.execute)


#
# add metadata to an xml document in the style of lalapps_inspiral
#


def add_cbc_metadata(xmldoc, process, seg_in, seg_out):
	"""
	A convenience function to add metadata to a cbc output document
	"""
	
	#
	# add entry to search_summary table
	#

	try:
		tbl = lsctables.table.get_table(xmldoc, lsctables.SearchSummaryTable.tableName)
	except ValueError:
		tbl = xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.SearchSummaryTable))
	search_summary = tbl.RowType()
	search_summary.process_id = process.process_id
	search_summary.shared_object = None # FIXME
	search_summary.lalwrapper_cvs_tag = None # FIXME
	search_summary.lal_cvs_tag = None # FIXME
	search_summary.comment = process.comment
	search_summary.set_ifos(process.get_ifos())
	search_summary.set_in(seg_in)
	search_summary.set_out(seg_out)
	search_summary.nevents = None # FIXME
	search_summary.nnodes = 1
	tbl.append(search_summary)

	#
	# add entry to filter table
	#

	try:
		tbl = lsctables.table.get_table(xmldoc, lsctables.FilterTable.tableName)
	except ValueError:
		tbl = xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.FilterTable))
	tbl.sync_next_id()
	row = tbl.RowType()
	row.process_id = process.process_id
	row.program = process.program
	row.start_time = int(seg_in[0])
	row.filter_name = None # FIXME
	row.filter_id = tbl.get_next_id()
	row.param_set = None # FIXME
	row.comment = process.comment
	tbl.append(row)

	#
	# add entries to search_summvars table
	#

	try:
		tbl = lsctables.table.get_table(xmldoc, lsctables.SearchSummVarsTable.tableName)
	except ValueError:
		tbl = xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.SearchSummVarsTable))
	tbl.sync_next_id()
	# FIXME

	#
	# add entries to summ_value table
	#

	try:
		tbl = lsctables.table.get_table(xmldoc, lsctables.SummValueTable.tableName)
	except ValueError:
		tbl = xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.SummValueTable))
	tbl.sync_next_id()
	# FIXME

	#
	# done
	#

	return search_summary


#
# =============================================================================
#
#                           Parameter Distributions
#
# =============================================================================
#

#
# Functions to synthesize injections
#

def snr_distribution(size, startsnr):
	"""
	This produces a power law distribution in snr of size size starting at startsnr
	"""
	return startsnr * random.power(3, size)**-1 # 3 here actually means 2 :) according to scipy docs

def noncentrality(snrs, prefactor):
	"""
	This produces a set of noncentrality parameters that scale with snr^2 according to the prefactor
	"""
	return prefactor * random.rand(len(snrs)) * snrs**2 # FIXME power depends on dimensionality of the bank and the expectation for the mismatch for real signals
	#return prefactor * random.power(1, len(snrs)) * snrs**2 # FIXME power depends on dimensionality of the bank and the expectation for the mismatch for real signals

def chisq_distribution(df, non_centralities, size):
	"""
	This produces a set of noncentral chisq values of size size, with degrees of freedom given by df
	"""
	out = numpy.empty((len(non_centralities) * size,))
	for i, nc in enumerate(non_centralities):
		out[i*size:(i+1)*size] = random.noncentral_chisquare(df, nc, size)
	return out


#
# Book-keeping class
#


class DistributionsStats(object):
	"""
	A class used to populate a CoincParamsDistribution instance using
	event parameter data.
	"""

	binnings = {
		"H1_snr_chi": rate.NDBins((rate.LinearPlusOverflowBins(4., 26., 200), rate.LogarithmicPlusOverflowBins(.001, 0.5, 200))),
		"H2_snr_chi": rate.NDBins((rate.LinearPlusOverflowBins(4., 26., 200), rate.LogarithmicPlusOverflowBins(.001, 0.5, 200))),
		"L1_snr_chi": rate.NDBins((rate.LinearPlusOverflowBins(4., 26., 200), rate.LogarithmicPlusOverflowBins(.001, 0.5, 200))),
		"V1_snr_chi": rate.NDBins((rate.LinearPlusOverflowBins(4., 26., 200), rate.LogarithmicPlusOverflowBins(.001, 0.5, 200)))
	}

	# FIXME the characteristic width (which is relevant for smoothing)
	# should be roughly 1.0 in SNR (from Gaussian noise expectations).  So
	# it is tied to how many bins there are per SNR range.  With 200 bins
	# between 4 and 26 each bin is .11 wide in SNR. So a width of 9 bins
	# corresponds to .99 which is close to 1.0
	filters = {
		"H1_snr_chi": rate.gaussian_window2d(9, 9, sigma = 10),
		"H2_snr_chi": rate.gaussian_window2d(9, 9, sigma = 10),
		"L1_snr_chi": rate.gaussian_window2d(9, 9, sigma = 10),
		"V1_snr_chi": rate.gaussian_window2d(9, 9, sigma = 10)
	}

	def __init__(self):
		self.lock = threading.Lock()
		self.raw_distributions = ligolw_burca_tailor.CoincParamsDistributions(**self.binnings)
		self.smoothed_distributions = ligolw_burca_tailor.CoincParamsDistributions(**self.binnings)

	@staticmethod
	def likelihood_params_func(events, offsetvector):
		instruments = set(event.ifo for event in events)
		if "H1" in instruments:
			instruments.discard("H2")
		return dict(("%s_snr_chi" % event.ifo, (event.snr, event.chisq / event.snr**2)) for event in events if event.ifo in instruments)

	def add_single(self, event):
		self.raw_distributions.add_background(self.likelihood_params_func((event,), None))

	def add_background_prior(self, n = 1., transition = 10.):
		for param, binarr in self.raw_distributions.background_rates.items():
			# Custom handle the first and last over flow bins
			snrs = binarr.bins[0].centres()
			snrs[0] = snrs[1] * .9
			snrs[-1] = snrs[-2] * 1.1
			chi2_over_snr2s = binarr.bins[1].centres()
			chi2_over_snr2s[0] = chi2_over_snr2s[1] * .9
			chi2_over_snr2s[-1] = chi2_over_snr2s[-2] * 1.1
			for snr in snrs:
				p = numpy.exp(-snr**2 / 2. + snrs[0]**2 / 2. + numpy.log(n))
				p += (transition / snr)**6 * numpy.exp( -transition**2 / 2. + snrs[0]**2 / 2. + numpy.log(n)) # Softer fall off above some transition SNR for numerical reasons
				for chi2_over_snr2 in chi2_over_snr2s:
					binarr[snr, chi2_over_snr2] += p
			# normalize to the requested count
			binarr.array /= binarr.array.sum()
			binarr.array *= n

	def add_foreground_prior(self, n = 1., prefactors_range = (0.02, 0.5), df = 40, verbose = False):
		# FIXME:  for maintainability, this should be modified to
		# use the .add_injection() method of the .raw_distributions
		# attribute, but that will slow this down
		pfs = numpy.linspace(prefactors_range[0], prefactors_range[1], 10)
		for param, binarr in self.raw_distributions.injection_rates.items():
			if verbose:
				print >> sys.stderr, "synthesizing injections for %s" % param
			# Custom handle the first and last over flow bins
			snrs = binarr.bins[0].centres()
			snrs[0] = snrs[1] * .9
			snrs[-1] = snrs[-2] * 1.1
			chi2_over_snr2s = binarr.bins[1].centres()
			chi2_over_snr2s[0] = chi2_over_snr2s[1] * .9
			chi2_over_snr2s[-1] = chi2_over_snr2s[-2] * 1.1
			for i, snr in enumerate(snrs):
				for j, chi2_over_snr2 in enumerate(chi2_over_snr2s):
					chisq = chi2_over_snr2 * snr**2 * df # We record the reduced chi2
					dist = 0
					for pf in pfs:
						nc = pf * snr**2
						v = stats.ncx2.pdf(chisq, df, nc)
						if numpy.isfinite(v):
							dist += v
					dist *= (snr / snrs[0])**-2
					if numpy.isfinite(dist):
						binarr[snr, chi2_over_snr2] += dist
			# normalize to the requested count
			binarr.array /= binarr.array.sum()
			binarr.array *= n

	def finish(self, verbose = False):
		self.smoothed_distributions = self.raw_distributions.copy(self.raw_distributions)
		#self.smoothed_distributions.finish(filters = self.filters, verbose = verbose)
		# FIXME:  should be the line above, we'll temporarily do
		# the following.  the difference is that the above produces
		# PDFs while what follows produces probabilities in each
		# bin
		if verbose:
			print >>sys.stderr, "smoothing parameter distributions ...",
		for name, binnedarray in itertools.chain(self.smoothed_distributions.background_rates.items(), self.smoothed_distributions.injection_rates.items()):
			if verbose:
				print >>sys.stderr, "%s," % name,
			rate.filter_array(binnedarray.array, self.filters[name])
			binnedarray.array /= numpy.sum(binnedarray.array)
		if verbose:
			print >>sys.stderr, "done"

	@classmethod
	def from_filenames(cls, filenames, verbose = False):
		self = cls()
		self.raw_distributions, seglists = ligolw_burca_tailor.load_likelihood_data(filenames, u"gstlal_inspiral_likelihood", verbose = verbose)
		# FIXME:  produce error if binnings don't match this class's binnings attribute?
		binnings = dict((param, self.raw_distributions.zero_lag_rates[param].bins) for param in self.raw_distributions.zero_lag_rates)
		self.smoothed_distributions = ligolw_burca_tailor.CoincParamsDistributions(**binnings)
		return self, seglists

	def to_xml(self, seglists):
		return ligolw_burca_tailor.gen_likelihood_control(self.raw_distributions, seglists, u"gstlal_inspiral_likelihood")

	def to_filename(self, filename, seglists, verbose = False):
		# FIXME:  there might be times when we want to trap signals
		utils.write_filename(self.to_xml(seglists), filename, verbose = verbose, gz = (filename or "stdout").endswith(".gz"), trap_signals = None)


#
# =============================================================================
#
#                               Output Document
#
# =============================================================================
#


class Data(object):
	def __init__(self, filename, process_params, instruments, seg, out_seg, coincidence_threshold, distribution_stats, injection_filename = None, time_slide_file = None, comment = None, tmp_path = None, assign_likelihoods = False, likelihood_snapshot_interval = None, likelihood_retention_factor = 1.0, trials_factor = 1, thinca_interval = 50.0, gracedb_far_threshold = None, likelihood_file = None, gracedb_group = "Test", gracedb_type = "LowMass", verbose = False):
		#
		# initialize
		#
		
		# setup bottle routes
		bottle.route("/latency_histogram.txt")(self.write_latency_histogram)
		bottle.route("/latency_history.txt")(self.write_latency_history)
		bottle.route("/snr_history.txt")(self.write_snr_history)
		bottle.route("/ram_history.txt")(self.write_ram_history)
		bottle.route("/likelihood.xml")(self.write_likelihood_file)

		self.lock = threading.Lock()
		self.filename = filename
		self.instruments = instruments
		self.verbose = verbose
		self.distribution_stats = distribution_stats
		# True to enable likelihood assignment
		self.assign_likelihoods = assign_likelihoods
		# Set to None to disable period snapshots, otherwise set to seconds
		self.likelihood_snapshot_interval = likelihood_snapshot_interval
		# Set to 1.0 to disable background data decay
		self.likelihood_retention_factor = likelihood_retention_factor
		# FIXME:  should this live in the DistributionsStats object?
		self.likelihood_snapshot_timestamp = None
		# gracedb far threshold
		self.gracedb_far_threshold = gracedb_far_threshold
		self.gracedb_group = gracedb_group
		self.gracedb_type = gracedb_type

		# All possible instrument combinations
		# frozenset(ifos) for n in range(2, len(instruments)+1) for ifos in choices(instruments, n
		self.ifo_combos = [frozenset(ifos) for n in range(2, len(instruments)+1) for ifos in iterutils.choices(list(self.instruments), n)]

		# setup the first trials table instance (empty dict)
		self.trials_table = far.TrialsTable()
		self.trials_factor = trials_factor
		self.far = None
		self.likelihood_file = likelihood_file

		#
		# build the XML document
		#

		self.xmldoc = ligolw.Document()
		self.xmldoc.appendChild(ligolw.LIGO_LW())
		self.process = ligolw_process.register_to_xmldoc(self.xmldoc, "gstlal_inspiral", process_params, comment = comment, ifos = instruments)
		self.search_summary = add_cbc_metadata(self.xmldoc, self.process, seg, out_seg)
		# FIXME:  argh, ugly
		self.xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.SnglInspiralTable, columns = ("process_id", "ifo", "search", "channel", "end_time", "end_time_ns", "end_time_gmst", "impulse_time", "impulse_time_ns", "template_duration", "event_duration", "amplitude", "eff_distance", "coa_phase", "mass1", "mass2", "mchirp", "mtotal", "eta", "kappa", "chi", "tau0", "tau2", "tau3", "tau4", "tau5", "ttotal", "psi0", "psi3", "alpha", "alpha1", "alpha2", "alpha3", "alpha4", "alpha5", "alpha6", "beta", "f_final", "snr", "chisq", "chisq_dof", "bank_chisq", "bank_chisq_dof", "cont_chisq", "cont_chisq_dof", "sigmasq", "rsqveto_duration", "Gamma0", "Gamma1", "Gamma2", "Gamma3", "Gamma4", "Gamma5", "Gamma6", "Gamma7", "Gamma8", "Gamma9", "event_id")))
		self.xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.CoincDefTable))
		self.xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.CoincTable))
		self.xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.CoincMapTable))
		self.xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.TimeSlideTable))
		self.xmldoc.childNodes[-1].appendChild(lsctables.New(lsctables.CoincInspiralTable))

		#
		# optionally insert injection list document
		#

		if injection_filename is not None:
			ligolw_add.ligolw_add(self.xmldoc, [injection_filename], verbose = verbose)

		#
		# optionally insert a time slide table document.  if we
		# don't have one, add an all-zero offset vector.  remove
		# duplicate offset vectors when done
		#

		time_slide_table = lsctables.table.get_table(self.xmldoc, lsctables.TimeSlideTable.tableName)
		if time_slide_file is not None:
			ligolw_add.ligolw_add(self.xmldoc, [time_slide_file], verbose = verbose)
		else:
			for row in ligolw_tisi.RowsFromOffsetDict(dict.fromkeys(instruments, 0.0), time_slide_table.get_next_id(), self.process):
				time_slide_table.append(row)
		time_slide_mapping = ligolw_tisi.time_slides_vacuum(time_slide_table.as_dict())
		iterutils.inplace_filter(lambda row: row.time_slide_id not in time_slide_mapping, time_slide_table)
		for tbl in self.xmldoc.getElementsByTagName(ligolw.Table.tagName):
			tbl.applyKeyMapping(time_slide_mapping)

		#
		# if the output is an sqlite database, build the sqlite
		# database and convert the in-ram XML document to an
		# interface to the database file
		#

		if filename is not None and filename.endswith('.sqlite'):
			from glue.ligolw.utils import ligolw_sqlite
			from glue.ligolw import dbtables
			self.working_filename = dbtables.get_connection_filename(filename, tmp_path = tmp_path, replace_file = True, verbose = verbose)
			self.connection = sqlite3.connect(self.working_filename, check_same_thread=False)
			ligolw_sqlite.insert_from_xmldoc(self.connection, self.xmldoc, preserve_ids = True, verbose = verbose)
			self.xmldoc.removeChild(self.xmldoc.childNodes[-1]).unlink()
			self.xmldoc.appendChild(dbtables.get_xml(self.connection))
		else:
			self.connection = self.working_filename = None

		#
		# retrieve references to the table objects, now that we
		# know if they are database-backed or XML objects
		#

		self.process_table = lsctables.table.get_table(self.xmldoc, lsctables.ProcessTable.tableName)
		self.process_params_table = lsctables.table.get_table(self.xmldoc, lsctables.ProcessParamsTable.tableName)
		self.sngl_inspiral_table = lsctables.table.get_table(self.xmldoc, lsctables.SnglInspiralTable.tableName)
		self.coinc_definer_table = lsctables.table.get_table(self.xmldoc, lsctables.CoincDefTable.tableName)
		self.coinc_event_table = lsctables.table.get_table(self.xmldoc, lsctables.CoincTable.tableName)
		self.coinc_event_map_table = lsctables.table.get_table(self.xmldoc, lsctables.CoincMapTable.tableName)
		self.time_slide_table = lsctables.table.get_table(self.xmldoc, lsctables.TimeSlideTable.tableName)
		self.coinc_inspiral_table = lsctables.table.get_table(self.xmldoc, lsctables.CoincInspiralTable.tableName)

		# FIXME:  remove when lsctables.py has an ID generator attached to sngl_inspiral table
		self.sngl_inspiral_table.set_next_id(lsctables.SnglInspiralID(0))

		#
		# attach a StreamThinca instance to ourselves
		#

		self.stream_thinca = streamthinca.StreamThinca(
			self.xmldoc,
			self.process.process_id,
			coincidence_threshold = coincidence_threshold,
			thinca_interval = thinca_interval	# seconds
		)

		#
		# Fun output stuff
		#
		
		self.latency_histogram = rate.BinnedArray(rate.NDBins((rate.LinearPlusOverflowBins(5, 205, 22),)))
		self.latency_history = deque(maxlen=1000)
		self.snr_history = deque(maxlen=1000)
		self.ram_history = deque(maxlen = 1000)

	def appsink_new_buffer(self, elem):
		self.lock.acquire()
		try:
			# retrieve triggers from appsink element
			buffer = elem.emit("pull-buffer")
			timestamp = LIGOTimeGPS(0, buffer.timestamp)
			events = tuple(event for event in sngl_inspirals_from_buffer(buffer) if LIGOTimeGPS(event.end_time, event.end_time_ns) in self.search_summary.get_out())

			# set metadata on triggers
			for event in events:
				event.process_id = self.process.process_id
				event.event_id = self.sngl_inspiral_table.get_next_id()

			# update likelihood snapshot if needed
			if self.assign_likelihoods and (self.likelihood_snapshot_timestamp is None or (self.likelihood_snapshot_interval is not None and timestamp - self.likelihood_snapshot_timestamp >= self.likelihood_snapshot_interval)):
				#First time through, pick up a time stamp, finish the distributions, set the function
				if self.likelihood_snapshot_timestamp is None:
					self.likelihood_snapshot_timestamp = timestamp
					self.distribution_stats.finish(verbose = self.verbose)
					self.stream_thinca.set_likelihood_data(self.distribution_stats.smoothed_distributions, self.distribution_stats.likelihood_params_func)
				# generate smoothed snapshot of raw counts
				self.distribution_stats.finish(verbose = self.verbose)
				self.likelihood_snapshot_timestamp = timestamp
				# update stream thinca's likelihood data
				self.stream_thinca.set_likelihood_data(self.distribution_stats.smoothed_distributions, self.distribution_stats.likelihood_params_func)
				# decay the raw background counts to affect
				# a moving history
				# FIXME:  this will do bad things if the
				# instruments stop produce events;  the
				# decay should be tied to live time not
				# wall clock time
				for binnedarray in self.distribution_stats.raw_distributions.background_rates.values():
					binnedarray.array *= self.likelihood_retention_factor

				# create a FAR class 
				# livetime is set to None because it gets updated when coincidences are recorded
				# trials factor through from the command line
				self.far = far.FAR(None, self.trials_factor, self.distribution_stats)
				# FIXME don't hard code
				remap = {frozenset(["H1", "H2", "L1"]) : frozenset(["H1", "L1"]), frozenset(["H1", "H2", "V1"]) : frozenset(["H1", "V1"]), frozenset(["H1", "H2", "L1", "V1"]) : frozenset(["H1", "L1", "V1"])}

				# generate the background likelihood distributions
				for ifo_set in self.ifo_combos:
					self.far.updateFAPmap(ifo_set, remap, verbose = self.verbose)

				# hook up a reference to the Data class instance level trials_table
				self.far.trials_table = self.trials_table

				# write the new distribution stats to disk
				self.distribution_stats.lock.acquire()
				self.distribution_stats.to_filename(self.likelihood_file, segments.segmentlistdict.fromkeys(self.instruments, segments.segmentlist([self.search_summary.get_out()])), verbose = False)
				self.distribution_stats.lock.release()

			# run stream thinca
			noncoinc_sngls = self.stream_thinca.add_events(events, timestamp, FAP = self.far)

			# update the parameter distribution data.  only
			# update from sngls that weren't used in coincs
			for event in noncoinc_sngls:
				self.distribution_stats.add_single(event)

			# update output document
			if self.connection is not None:
				self.connection.commit()

			# do GraceDB alerts
			if self.gracedb_far_threshold is not None:
				self.do_gracedb_alerts()
				self.update_eye_candy()
		finally:
			self.lock.release()


	def write_likelihood_file(self):
		# write the new distribution stats to disk
		self.distribution_stats.lock.acquire()
		output = StringIO.StringIO()
		utils.write_fileobj(self.distribution_stats.to_xml(segments.segmentlistdict.fromkeys(self.instruments, segments.segmentlist([self.search_summary.get_out()]))), output, trap_signals = None)
		outstr = output.getvalue()
		output.close()
		self.distribution_stats.lock.release()
		return outstr

	def flush(self):
		# run StreamThinca's .flush().  returns the last remaining
		# non-coincident sngls.  add them to the distribution
		for event in self.stream_thinca.flush(FAP = self.far):
			self.distribution_stats.add_single(event)
		if self.connection is not None:
			self.connection.commit()

		# do GraceDB alerts
		if self.gracedb_far_threshold is not None:
			self.do_gracedb_alerts()


	def do_gracedb_alerts(self):
		try:
			gracedb
		except NameError:
			# gracedb import failed, disable event uploads
			return
		if self.stream_thinca.last_coincs:
			# FIXME:  this should maybe not be retrieved this
			# way.  and the .column_index() method is probably
			# useless
			coinc_inspiral_index = self.stream_thinca.last_coincs.coinc_inspiral_index
			for coinc_event_id, false_alarm_rate in sorted(self.stream_thinca.last_coincs.column_index(lsctables.CoincInspiralTable.tableName, "combined_far").items(), key = lambda (a, b): b):
				#
				# do we keep this event?
				#

				if false_alarm_rate > self.gracedb_far_threshold:
					continue

				#
				# fake a filename for end-user convenience
				#

				instruments = coinc_inspiral_index[coinc_event_id].get_ifos()
				observatories = "".join(sorted(set(instrument[0] for instrument in instruments)))
				instruments = "".join(sorted(instruments))
				description = "%s_%s_%s_%s" % (instruments, ("%.4g" % coinc_inspiral_index[coinc_event_id].mass).replace(".", "_").replace("-", "_"), self.gracedb_group, self.gracedb_type)
				end_time = int(coinc_inspiral_index[coinc_event_id].get_end())
				filename = "%s-%s-%d-%d.xml" % (observatories, description, end_time, 0)

				#
				# construct message and send to gracedb.
				# we go through the intermediate step of
				# first writing the document into a string
				# buffer incase there is some safety in
				# doing so in the event of a malformed
				# document;  instead of writing directly
				# into gracedb's input pipe and crashing
				# part way through.
				#

				if self.verbose:
					print >>sys.stderr, "sending %s to gracedb ..." % filename
				message = StringIO.StringIO()
				utils.write_fileobj(self.stream_thinca.last_coincs[coinc_event_id], message, gz = False, trap_signals = None)
				# FIXME: make this optional from command line?
				if True:
					resp = gracedb.Client().create(self.gracedb_group, self.gracedb_type, filename, message.getvalue())
					if "error" in resp:
						print >>sys.stderr, "gracedb upload of %s failed: %s" % (filename, resp["error"])
					elif self.verbose:
						if "warning" in resp:
							print >>sys.stderr, "gracedb issued warning: %s" % resp["warning"]
						print >>sys.stderr, "event assigned grace ID %s" % resp["output"]
				else:
					proc = subprocess.Popen(("/bin/cp", "/dev/stdin", filename), stdin = subprocess.PIPE)
					proc.stdin.write(message.getvalue())
					proc.stdin.flush()
					proc.stdin.close()
				message.close()

				# FIXME hack to keep from sending too many alerts.
				# Only send the best one in this set.  May not make
				# sense depending on what is in last_coincs.  FIX
				# PROPERLY.  This is probably mostly okay because we
				# should be doing coincidences every 10s which is a
				# reasonable time to cluster over
				break


	def update_eye_candy(self):
		if self.stream_thinca.last_coincs:
			self.ram_history.append((time.time(), (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) / 1048576.)) # GB
			latency_val = None
			snr_val = (0,0)
			coinc_inspiral_index = self.stream_thinca.last_coincs.coinc_inspiral_index
			for coinc_event_id, latency in self.stream_thinca.last_coincs.column_index(lsctables.CoincInspiralTable.tableName, "minimum_duration").items():
				self.latency_histogram[latency,] += 1
				if latency_val is None:
					t = float(coinc_inspiral_index[coinc_event_id].get_end())
					latency_val = (t, latency)
				snr = coinc_inspiral_index[coinc_event_id].snr
				if snr >= snr_val[0]:
					t = float(coinc_inspiral_index[coinc_event_id].get_end())
					snr_val = (t, snr)
			if latency_val is not None:
				self.latency_history.append(latency_val)
			if snr_val != (0,0):
				self.snr_history.append(snr_val)


	def write_latency_histogram(self):
		for latency, number in zip(self.latency_histogram.centres()[0][1:-1], self.latency_histogram.array[1:-1]):
			yield "%e %e\n" % (latency, number)


	def write_latency_history(self):
		# first one in the list is sacrificed for a time stamp
		for time, latency in self.latency_history:
			yield "%f %e\n" % (time, latency)


	def write_snr_history(self):
		# first one in the list is sacrificed for a time stamp
		for time, snr in self.snr_history:
			yield "%f %e\n" % (time, snr)


	def write_ram_history(self):
		# first one in the list is sacrificed for a time stamp
		for time, ram in self.ram_history:
			yield "%f %e\n" % (time, ram)


	def write_output_file(self, likelihood_file = None, verbose = False):
		if self.connection is not None:
			from glue.ligolw import dbtables
			self.connection.cursor().execute('UPDATE search_summary SET nevents = (SELECT count(*) FROM sngl_inspiral)')
			self.connection.cursor().execute('UPDATE process SET end_time = ?', (XLALUTCToGPS(time.gmtime()).seconds,))
			self.connection.commit()
			dbtables.build_indexes(self.connection, verbose = verbose)
			dbtables.put_connection_filename(self.filename, self.working_filename, verbose = verbose)
		else:
			self.sngl_inspiral_table.sort(lambda a, b: cmp(a.end_time, b.end_time) or cmp(a.end_time_ns, b.end_time_ns) or cmp(a.ifo, b.ifo))
			self.search_summary.nevents = len(self.sngl_inspiral_table)
			ligolw_process.set_process_end_time(self.process)
			utils.write_filename(self.xmldoc, self.filename, gz = (self.filename or "stdout").endswith(".gz"), verbose = verbose)

		# write out the snr / chisq histograms
		if likelihood_file is None:
			fname = os.path.split(self.filename)
			fname = os.path.join(fname[0], '%s_snr_chi.xml.gz' % ('.'.join(fname[1].split('.')[:-1]),))
		else:
			fname = likelihood_file
		self.distribution_stats.to_filename(fname, segments.segmentlistdict.fromkeys(self.instruments, segments.segmentlist([self.search_summary.get_out()])), verbose = verbose)

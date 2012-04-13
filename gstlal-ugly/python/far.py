#!/usr/bin/env python
#
# Copyright (C) 2011  Kipp Cannon, Chad Hanna, Drew Keppel
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

import sys
import numpy
from scipy import interpolate, random
from scipy.stats import poisson
from glue import iterutils
from glue.ligolw import ligolw
from glue.ligolw import param as ligolw_param
from glue.ligolw import lsctables
from glue.ligolw import utils
from glue.ligolw.utils import process as ligolw_process
from glue.ligolw.utils import segments as ligolw_segments
from glue.segmentsUtils import vote
from pylal import ligolw_burca2
from pylal import inject
from pylal import llwapp
from pylal import rate
from gstlal.svd_bank import read_bank
try:
	import sqlite3
except ImportError:
	# pre 2.5.x
	from pysqlite2 import dbapi2 as sqlite3

sqlite3.enable_callback_tracebacks(True)

#
# Function to compute the fap in a given file
#

def set_fap(options, Far, f):
	from glue.ligolw import dbtables

	# set up working file names
	working_filename = dbtables.get_connection_filename(f, tmp_path = options.tmp_space, verbose = options.verbose)
	connection = sqlite3.connect(working_filename)

	# define fap function
	connection.create_function("fap", 3, Far.fap_from_rank)

	# FIXME abusing FAR column
	connection.cursor().execute("UPDATE coinc_inspiral SET false_alarm_rate = (SELECT fap(coinc_event.likelihood, coinc_inspiral.ifos, coinc_event.time_slide_id) FROM coinc_event WHERE coinc_event.coinc_event_id == coinc_inspiral.coinc_event_id)")
	connection.commit()

	# all finished
	connection.commit()
	connection.close()
	dbtables.put_connection_filename(f, working_filename, verbose = options.verbose)

#
# Trials table
#

class TrialsTable(dict):
	def from_db(self, connection):
		for ifos, tsid, count in connection.cursor().execute('SELECT ifos, coinc_event.time_slide_id AS tsid, count(*) / nevents FROM sngl_inspiral JOIN coinc_event_map ON coinc_event_map.event_id == sngl_inspiral.event_id JOIN coinc_inspiral ON coinc_inspiral.coinc_event_id == coinc_event_map.coinc_event_id JOIN coinc_event ON coinc_event.coinc_event_id == coinc_event_map.coinc_event_id  WHERE coinc_event_map.table_name = "sngl_inspiral" GROUP BY tsid, ifos;'):
			try:
				self[ifos, tsid] += count
			except KeyError:
				self[ifos, tsid] = count
		connection.commit()

	def increment(self, n):
		for k in self:
			self[k] += n

	@classmethod
	def from_xml(cls, xml, name):
		xml, = [elem for elem in xml.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") and elem.getAttribute(u"Name") == u"%s:gstlal_inspiral_trialstable" % name]
		self = cls()
		for param in xml.getElementsByTagName(ligolw.Param.tagName):
			self[param.getAttribute(u"Name")] = param.pcdata
		return self

	def to_xml(self, name):
		xml = ligolw.LIGO_LW({u"Name": u"%s:gstlal_inspiral_trialstable" % name})
		for key, value in self.items():
			xml.appendChild(ligolw_param.from_pyvalue(key, value))
		return xml

#
# Function to compute the far in a given file
#

def set_far(options, Far, f):
	from glue.ligolw import dbtables

	working_filename = dbtables.get_connection_filename(f, tmp_path = options.tmp_space, verbose = options.verbose)
	connection = sqlite3.connect(working_filename)

	connection.create_function("far", 1, Far.compute_far)
	ids = [id for id, in connection.cursor().execute("SELECT DISTINCT(time_slide_id) FROM time_slide")]
	for id in ids:
		print >>sys.stderr, "computing rates for ", id
		# FIXME abusing FAR column
		connection.cursor().execute('DROP TABLE IF EXISTS ranktable')
		connection.commit()
		# FIXME any indicies on ranktable??  FIXME adding the +
		# 1e-20 / snr to the false alarm rate forces the
		# ranking to be in snr order for degenerate ranks.
		# This is okay for non degenerate things since we don't
		# have 1e-20 dynamic range, still this should be done
		# smarter.  Note it doesn't change the false alarm rate
		# in the database. It just changes how the CDF is
		# produced here which will modify the order of events
		# that get assigned a combined far
		connection.cursor().execute('CREATE TEMPORARY TABLE ranktable AS SELECT * FROM coinc_inspiral JOIN coinc_event ON coinc_event.coinc_event_id == coinc_inspiral.coinc_event_id WHERE coinc_event.time_slide_id == ? ORDER BY false_alarm_rate+1e-20 / snr', (id,))
		connection.commit()
		# FIXME someday we might want to consider the rate of multiple events.
		# having this sorted list will make that easier.  Then you can pass in the rowid like this
		#
		# far(ranktable.false_alarm_rate, ranktable.rowid)
		#
		# in order to have the number of events below a given FAP
		if connection.cursor().execute("SELECT name FROM sqlite_master WHERE name='sim_inspiral'").fetchall():
			connection.cursor().execute('UPDATE coinc_inspiral SET combined_far = (SELECT far(ranktable.false_alarm_rate) FROM ranktable WHERE ranktable.coinc_event_id == coinc_inspiral.coinc_event_id) WHERE coinc_inspiral.coinc_event_id IN (SELECT coinc_event_id FROM ranktable)')
		# For everything else we get a cumulative number
		else:
			connection.cursor().execute('UPDATE coinc_inspiral SET combined_far = (SELECT far(ranktable.false_alarm_rate) FROM ranktable WHERE ranktable.coinc_event_id == coinc_inspiral.coinc_event_id) WHERE coinc_inspiral.coinc_event_id IN (SELECT coinc_event_id FROM ranktable)')

	connection.commit()
	connection.close()
	dbtables.put_connection_filename(f, working_filename, verbose = options.verbose)

#
# Class to handle the computation of FAPs/FARs
#

class FAR(object):
	def __init__(self, livetime, trials_factor, distribution_stats = None, trials_table = None):
		self.distribution_stats = distribution_stats
		self.trials_table = trials_table
		if self.distribution_stats is not None:
			# FIXME:  this results in the
			# .smoothed_distributions object containing
			# *probabilities* not probability densities. this
			# might be changed in the future.
			self.distribution_stats.finish()
			self.likelihood_ratio = ligolw_burca2.LikelihoodRatio(self.distribution_stats.smoothed_distributions)
		else:
			self.likelihood_ratio = None
		self.livetime = livetime
		self.trials_factor = trials_factor
		self.reset()

	def reset(self):
		self.ccdf_interpolator = {}
		self.minrank = {}
		self.maxrank = {}
		self.likelihood_pdfs = {}

	def updateFAPmap(self, ifo_set, remap = {}, verbose = False):
		if self.distribution_stats is None:
			raise InputError, "must provide background bins file"

		# we might choose to statically map certain likelihood
		# distributions to others.  This is useful for ignoring H2 when
		# H1 is present. By default we don't
		
		remap_set = remap.setdefault(ifo_set, ifo_set)
		
		#
		# the target FAP resolution is 1 part in 10^6.  So depending on
		# how many instruments we have we have to take the nth root of
		# that number to set the scale in each detector. This is purely
		# for memory/CPU requirements
		#

		targetlen = int(1e6**(1. / len(remap_set)))

		# reduce typing
		background = self.distribution_stats.smoothed_distributions.background_rates
		injections = self.distribution_stats.smoothed_distributions.injection_rates


		for param in background:
			# FIXME only works if there is a 1-1 relationship between params and instruments
			instrument = param.split("_")[0]
			if instrument not in remap_set:
				continue

			# don't repeat the calculation if we have already done
			# it for the requested instrument and resolution
			if (instrument, targetlen) in self.likelihood_pdfs:
				if verbose:
					print >>sys.stderr, "already computed likelihood for ", instrument, " at resolution ", targetlen, " continuing..."
				continue
			else:
				if verbose:
					print >>sys.stderr, "updating likelihood background for ", instrument, " in ", ifo_set
	
			likelihoods = injections[param].array / background[param].array
			# ignore infs and nans because background is never
			# found in those bins.  the boolean array indexing
			# flattens the array
			likelihoods = likelihoods[numpy.isfinite(likelihoods)]
			minlikelihood = likelihoods[likelihoods != 0].min()
			maxlikelihood = likelihoods.max()

			# construct PDF
			# FIXME:  because the background array contains
			# probabilities and not probability densities, the
			# likelihood_pdfs contain probabilities and not
			# densities, as well, when this is done
			self.likelihood_pdfs[(instrument, targetlen)] = rate.BinnedArray(rate.NDBins((rate.LogarithmicPlusOverflowBins(minlikelihood, maxlikelihood, targetlen),)))
			for coords in iterutils.MultiIter(*background[param].bins.centres()):
				likelihood = self.likelihood_ratio({param: coords})
				if numpy.isfinite(likelihood):
					self.likelihood_pdfs[(instrument, targetlen)][likelihood,] += background[param][coords]

		# switch to str representation because we will be using these in DB queries
		ifostr = lsctables.ifos_from_instrument_set(ifo_set)

		# only recompute if necessary
		if ifostr not in self.ccdf_interpolator:
			if verbose:
				print >>sys.stderr, "computing joint likelihood background for ", ifostr
			ranks, weights = self.possible_ranks_array(self.likelihood_pdfs, remap_set, targetlen)
			# complementary cumulative distribution function
			ccdf = weights[::-1].cumsum()[::-1]
			ccdf /= ccdf[0]
			self.ccdf_interpolator[ifostr] = interpolate.interp1d(ranks, ccdf)
			# record min and max ranks so we know which end of the ccdf to use when we're out of bounds
			self.minrank[ifostr] = (min(ranks), ccdf[0])
			self.maxrank[ifostr] = (max(ranks), ccdf[-1])
		else:
			if verbose:
				print>>sys.stderr, "already computed faps for ", ifostr, " continuing"

	def fap_from_rank(self, rank, ifostr, tsid):
		# FIXME:  doesn't check that rank is a scalar
		if rank >= self.maxrank[ifostr][0]:
			return self.maxrank[ifostr][1]
		if rank <= self.minrank[ifostr][0]:
			return self.minrank[ifostr][1]
		fap = float(self.ccdf_interpolator[ifostr](rank))
		trials_factor = int(self.trials_table.setdefault((ifostr, tsid),1) * self.trials_factor) or 1
		return 1.0 - (1.0 - fap)**trials_factor

	def possible_ranks_array(self, likelihood_pdfs, ifo_set, targetlen):
		# start with an identity array to seed the outerproduct chain
		ranks = numpy.array([1.0])
		vals = numpy.array([1.0])
		# FIXME:  probably only works because the pdfs aren't pdfs but probabilities
		for ifo in ifo_set:
			likelihood_pdf = likelihood_pdfs[(ifo,targetlen)]
			# FIXME lower instead of centres() to avoid inf in the last bin
			ranks = numpy.outer(ranks, likelihood_pdf.bins.lower()[0])
			vals = numpy.outer(vals, likelihood_pdf.array)
			ranks = ranks.reshape((ranks.shape[0] * ranks.shape[1],))
			# FIXME nans arise from inf * 0.  Do we want these to be 0?
			#ranks[numpy.isnan(ranks)] = 0.0
			vals = vals.reshape((vals.shape[0] * vals.shape[1],))
		vals = vals[ranks.argsort()]
		ranks.sort()
		return ranks, vals

	def compute_far(self, fap):
		if fap == 0.:
			return 0.
		livetime = float(abs(self.livetime))
		return 0. - numpy.log(1. - fap) / livetime

def get_live_time(segments, verbose = True):
	livetime = float(abs(vote((segs for instrument, segs in segments.items() if instrument != "H2"), 2)))
	if verbose:
		print >> sys.stderr, "Livetime: ", livetime
	return livetime

def get_live_time_segs_from_search_summary_table(connection, program_name = "gstlal_inspiral"):
	from glue.ligolw import dbtables
	xmldoc = dbtables.get_xml(connection)
	farsegs = llwapp.segmentlistdict_fromsearchsummary(xmldoc, program_name).coalesce()
	xmldoc.unlink()
	return farsegs

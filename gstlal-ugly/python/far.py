#!/usr/bin/python

# FIXME proper copyright and GPL notice
# Copyright 2011 Kipp Cannon, Chad Hanna

import sys
import numpy
from scipy import interpolate, random
from scipy.stats import poisson
from glue import iterutils
from glue.ligolw import lsctables
from glue.ligolw import utils
from glue.ligolw.utils import process as ligolw_process
from glue.ligolw.utils import segments as ligolw_segments
from glue.segmentsUtils import vote
from pylal import ligolw_burca2
from pylal import inject
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

	# compute the faps
	print >>sys.stderr, "computing faps ..."
	for ifos, in connection.cursor().execute('SELECT DISTINCT(ifos) FROM coinc_inspiral').fetchall():

		print >>sys.stderr, "computing fap maps for ", ifos
		ifoset = frozenset(lsctables.instrument_set_from_ifos(ifos))
		#ifoset.discard("H2")
		#ifos = lsctables.ifos_from_instrument_set(ifoset)
		# FIXME make this remap an option, don't hardcode
		# remap means that you actually want to build a different
		# likelihood distribution.  Here is lets us avoid the H1/H2
		# correlations by ignoring H2.  THis only works if H2 was also
		# ignored in the ranking
		Far.updateFAPmap(ifoset, remap = {frozenset(["H1", "H2", "L1"]) : frozenset(["H1", "L1"]), frozenset(["H1", "H2", "V1"]) : frozenset(["H1", "V1"]), frozenset(["H1", "H2", "L1", "V1"]) : frozenset(["H1", "L1", "V1"])})

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
	@classmethod
	def from_db(cls, connection):
		self = cls(((ifos, tsid), count) for ifos, tsid, count in connection.cursor().execute('SELECT ifos, coinc_event.time_slide_id AS tsid, count(*) / nevents FROM sngl_inspiral JOIN coinc_event_map ON coinc_event_map.event_id == sngl_inspiral.event_id JOIN coinc_inspiral ON coinc_inspiral.coinc_event_id == coinc_event_map.coinc_event_id JOIN coinc_event ON coinc_event.coinc_event_id == coinc_event_map.coinc_event_id  WHERE coinc_event_map.table_name = "sngl_inspiral" GROUP BY tsid, ifos;'))
		connection.commit()
		return self

	def increment(self, n):
		for k in self:
			self[k] += n

#
# Function to compute the far in a given file
#

def set_far(options, Far, f):
	from glue.ligolw import dbtables

	working_filename = dbtables.get_connection_filename(f, tmp_path = options.tmp_space, verbose = options.verbose)
	connection = sqlite3.connect(working_filename)

	connection.create_function("far", 2, Far.compute_far)
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
		# For injections every event is treated as "the loudest"
		if connection.cursor().execute("SELECT name FROM sqlite_master WHERE name='sim_inspiral'").fetchall():
			connection.cursor().execute('UPDATE coinc_inspiral SET combined_far = (SELECT far(ranktable.false_alarm_rate, 1) FROM ranktable WHERE ranktable.coinc_event_id == coinc_inspiral.coinc_event_id) WHERE coinc_inspiral.coinc_event_id IN (SELECT coinc_event_id FROM ranktable)')
		# For everything else we get a cumulative number
		else:
			connection.cursor().execute('UPDATE coinc_inspiral SET combined_far = (SELECT far(ranktable.false_alarm_rate, ranktable.rowid) FROM ranktable WHERE ranktable.coinc_event_id == coinc_inspiral.coinc_event_id) WHERE coinc_inspiral.coinc_event_id IN (SELECT coinc_event_id FROM ranktable)')

	connection.commit()
	connection.close()
	dbtables.put_connection_filename(f, working_filename, verbose = options.verbose)

#
# Class to handle the computation of FAPs/FARs
#

class FAR(object):
	def __init__(self, livetime, trials_factor, distribution_stats = None):
		self.distribution_stats = distribution_stats
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
		# the target FAP resolution is 1 part in 10^7.  So depending on
		# how many instruments we have we have to take the nth root of
		# that number to set the scale in each detector. This is purely
		# for memory/CPU requirements
		#

		targetlen = int(1e7**(1. / len(remap_set)))

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

	def compute_far(self, fap, n = 1):
		if fap == 0.:
			return 0.
		livetime = float(abs(self.livetime))
		# the n = 1 case can be done exactly.  That is good since it is
		# the most important.
		if n == 1:
			return 0. - numpy.log(1. - fap) / livetime
		if n > 1 and n <= 100:
			nvec = numpy.logspace(-12, numpy.log10(n + 10. * n**.5), 100)
		else:
			nvec = numpy.logspace(numpy.log10(n - 10. * n**.5), numpy.log10(n + 10. * n**.5), 100)
		FAPS = 1. - poisson.cdf(n,nvec)
		#FIXME is this right since nvec is log spaced?
		interp = interpolate.interp1d(FAPS, nvec / livetime)
		if fap < FAPS[1]:
			return 0.
		if fap > FAPS[-1]:# This means that the FAP has gone off the edge.  We will bump it down because we don't really care about this being right.
			fap = FAPS[-1]
		return float(interp(fap))


def get_live_time(segments, verbose = True):
	livetime = float(abs(vote((segs for instrument, segs in segments.items() if instrument != "H2"), 2)))
	if verbose:
		print >> sys.stderr, "Livetime: ", livetime
	return livetime

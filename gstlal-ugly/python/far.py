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
# Utility functions
#

def linearize_array(arr):
	return arr.reshape((1,arr.shape[0] * arr.shape[1]))

#
# Function to compute the fap in a given file
#

def set_fap(options, Far, f):
	from glue.ligolw import dbtables

	# set up working file names
	working_filename = dbtables.get_connection_filename(f, tmp_path = options.tmp_space, verbose = options.verbose)
	connection = sqlite3.connect(working_filename)

	# define double and triple fap functions
	connection.create_function("fap2", 11, Far.compute_fap2)
	connection.create_function("fap3", 14, Far.compute_fap3)

	# compute the faps
	print >>sys.stderr, "computing faps ..."
	for ifos, in connection.cursor().execute('SELECT DISTINCT(ifos) FROM coinc_inspiral').fetchall():

		print >>sys.stderr, "computing faps for ", ifos
		ifoset = lsctables.instrument_set_from_ifos(ifos)
		ifoset.discard("H2")
		Far.updateFAPmap(ifoset)

		# FIXME abusing FAR column
		connection.cursor().execute(fap_query(ifoset, ifos))
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
		connection.cursor().execute('CREATE INDEX template_index ON sngl_inspiral(mass1,mass2,chi)')
		self = cls(((ifos, tsid, mass1, mass2, chi), count) for ifos, tsid, mass1, mass2, chi, count in connection.cursor().execute('SELECT ifos, coinc_event.time_slide_id, mass1, mass2, chi, count(*) / nevents FROM sngl_inspiral JOIN coinc_event_map ON coinc_event_map.event_id == sngl_inspiral.event_id JOIN coinc_inspiral ON coinc_inspiral.coinc_event_id == coinc_event_map.coinc_event_id JOIN coinc_event ON coinc_event.coinc_event_id == coinc_event_map.coinc_event_id  WHERE coinc_event_map.table_name = "sngl_inspiral" GROUP BY mass1, mass2, chi, ifos;'))
		connection.cursor().execute('DROP INDEX template_index')
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

	def updateFAPmap(self, instruments):
		if self.distribution_stats is None:
			raise InputError, "must provide background bins file"

		#
		# the target FAP resolution is 1 part in 10^7.  So depending on
		# how many instruments we have we have to take the nth root of
		# that number to set the scale in each detector. This is purely
		# for memory/CPU requirements
		#

		targetlen = int(1e7**(1. / len(instruments)))

		# reduce typing
		background = self.distribution_stats.smoothed_distributions.background_rates
		injections = self.distribution_stats.smoothed_distributions.injection_rates

		likelihood_pdfs = {}
		for param in background:
			instrument = param.split("_")[0]
			if instrument not in instruments:
				continue

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
			likelihood_pdfs[param] = rate.BinnedArray(rate.NDBins((rate.LogarithmicPlusOverflowBins(minlikelihood, maxlikelihood, targetlen),)))
			for coords in iterutils.MultiIter(*background[param].bins.centres()):
				likelihood = self.likelihood_ratio({param: coords})
				if numpy.isfinite(likelihood):
					likelihood_pdfs[param][likelihood,] += background[param][coords]
		# make sure we didn't skip any instruments' data
		assert len(likelihood_pdfs) == len(instruments)

		ranks, weights = self.possible_ranks_array(likelihood_pdfs)
		# complementary cumulative distribution function
		self.ccdf = weights[::-1].cumsum()[::-1]
		self.ccdf /= self.ccdf[0]
		self.ccdf_interpolator = interpolate.interp1d(ranks, self.ccdf)
		# record min and max ranks so we know which end of the ccdf to use when we're out of bounds
		self.minrank = min(ranks)
		self.maxrank = max(ranks)

	def fap_from_rank(self, rank):
		# FIXME:  doesn't check that rank is a scalar
		try:
			return self.ccdf_interpolator(rank)[0]
		except ValueError:
			# out of bounds, return ccdf edge
			if rank >= self.maxrank:
				return self.ccdf[-1]
			if rank < self.minrank:
				return self.ccdf[0]
			# shouldn't get here
			raise

	def possible_ranks_array(self, likelihood_pdfs):
		# start with an identity array to seed the outerproduct chain
		ranks = numpy.array([1.0])
		vals = numpy.array([1.0])
		# FIXME:  probably only works because the pdfs aren't pdfs but probabilities
		for likelihood_pdf in likelihood_pdfs.values():
			ranks = numpy.outer(ranks, likelihood_pdf.centres()[0])
			vals = numpy.outer(vals, likelihood_pdf.array)
			ranks = ranks.reshape((ranks.shape[0] * ranks.shape[1],))
			vals = vals.reshape((vals.shape[0] * vals.shape[1],))
		vals = vals[ranks.argsort()]
		ranks.sort()
		return ranks, vals

	# Method only works if likelihood ratio data is available
	def compute_rank(self, snr_chisq_dict):
		if self.distribution_stats is None:
			raise InputError, "must provide background bins file"
		snr_chisq_dict = dict((ifo + "_snr_chi", (snr, chisq**.5 / snr)) for ifo, (snr, chisq) in snr_chisq_dict.items())
		return self.likelihood_ratio(snr_chisq_dict)

	def compute_fap2(self, ifos, tsid, mass1, mass2, chi, ifo1, snr1, chisq1, ifo2, snr2, chisq2):
		trials_factor = self.trials_table.setdefault((ifos, tsid, mass1, mass2, chi),1) + self.trials_factor
		input = {ifo1:(snr1,chisq1), ifo2:(snr2,chisq2)}
		rank = self.compute_rank(input)
		fap = self.fap_from_rank(rank)
		fap = 1.0 - (1.0 - fap)**trials_factor
		return float(fap)

	def compute_fap3(self, ifos, tsid, mass1, mass2, chi, ifo1, snr1, chisq1, ifo2, snr2, chisq2, ifo3, snr3, chisq3):
		trials_factor = self.trials_table.setdefault((ifos, tsid, mass1, mass2, chi),1) + self.trials_factor
		input = {ifo1:(snr1,chisq1), ifo2:(snr2,chisq2), ifo3:(snr3,chisq3)}
		rank = self.compute_rank(input)
		fap = self.fap_from_rank(rank)
		fap = 1.0 - (1.0 - fap)**trials_factor
		return float(fap)

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
		return interp(fap)[0]


def get_live_time(segments, verbose = True):
	livetime = float(abs(vote((segs for instrument, segs in segments.items() if instrument != "H2"), 2)))
	if verbose:
		print >> sys.stderr, "Livetime: ", livetime
	return livetime

def two_fap_query(fap_ifos, ifostr):
	# NOTE Assumes exact mass1,mass2,chi coincidence
	fap_ifos = tuple(fap_ifos)
	query = '''UPDATE coinc_inspiral
	SET false_alarm_rate = (SELECT fap2(coinc_inspiral.ifos, coinc_event.time_slide_id, snglA.mass1, snglA.mass2, snglA.chi, snglA.ifo, snglA.snr, snglA.chisq, snglB.ifo, snglB.snr, snglB.chisq)
				FROM coinc_event_map AS mapA
				JOIN coinc_event_map AS mapB ON mapB.coinc_event_id == coinc_inspiral.coinc_event_id
				JOIN sngl_inspiral AS snglA ON snglA.event_id == mapA.event_id
				JOIN sngl_inspiral AS snglB ON snglB.event_id == mapB.event_id
				JOIN coinc_event ON coinc_event.coinc_event_id == mapA.coinc_event_id
				WHERE mapA.table_name == "sngl_inspiral"
				AND mapB.table_name == "sngl_inspiral"
				AND snglA.ifo == "%s"
				AND snglB.ifo == "%s"
				AND mapA.coinc_event_id == coinc_inspiral.coinc_event_id)
	WHERE ifos == "%s"''' % (fap_ifos[0], fap_ifos[1], ifostr)
	return query

def three_fap_query(fap_ifos, ifostr):
	fap_ifos = tuple(fap_ifos)
	query = '''UPDATE coinc_inspiral
	SET false_alarm_rate = (SELECT fap3(coinc_inspiral.ifos, coinc_event.time_slide_id, snglA.mass1, snglA.mass2, snglA.chi, snglA.ifo, snglA.snr, snglA.chisq, snglB.ifo, snglB.snr, snglB.chisq, snglC.ifo, snglC.snr, snglC.chisq)
				FROM coinc_event_map AS mapA
				JOIN coinc_event_map AS mapB ON mapB.coinc_event_id == coinc_inspiral.coinc_event_id
				JOIN coinc_event_map AS mapC ON mapC.coinc_event_id == coinc_inspiral.coinc_event_id
				JOIN sngl_inspiral AS snglA ON snglA.event_id == mapA.event_id
				JOIN sngl_inspiral AS snglB ON snglB.event_id == mapB.event_id
				JOIN sngl_inspiral AS snglC ON snglC.event_id == mapC.event_id
				JOIN coinc_event ON coinc_event.coinc_event_id == mapA.coinc_event_id
				WHERE mapA.table_name == "sngl_inspiral"
				AND mapB.table_name == "sngl_inspiral"
				AND mapC.table_name == "sngl_inspiral"
				AND snglA.ifo == "%s"
				AND snglB.ifo == "%s"
				AND snglC.ifo == "%s"
				AND mapA.coinc_event_id == coinc_inspiral.coinc_event_id)
	WHERE ifos == "%s"''' % (fap_ifos[0], fap_ifos[1], fap_ifos[2], ifostr)
	return query

def fap_query(fap_ifos, ifostr):
	return {2: two_fap_query, 3: three_fap_query}[len(fap_ifos)](fap_ifos, ifostr)


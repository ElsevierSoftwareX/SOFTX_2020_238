#!/usr/bin/python

# FIXME proper copyright and GPL notice
# Copyright 2011 Kipp Cannon, Chad Hanna

import sys
import numpy
from scipy import interpolate, random
from scipy.stats import poisson
from glue.ligolw import lsctables
from glue.ligolw import utils
from glue.ligolw.utils import process as ligolw_process
from glue.ligolw.utils import segments as ligolw_segments
from glue.segmentsUtils import vote
from pylal import inject
from pylal import rate
from gstlal import ligolw_output as gstlal_likelihood
from gstlal.svd_bank import read_bank
try:
	import sqlite3
except ImportError:
	# pre 2.5.x
	from pysqlite2 import dbapi2 as sqlite3

sqlite3.enable_callback_tracebacks(True)

#
# Functions to synthesize injections
#

def snr_distribution(size, startsnr):
	return startsnr * random.power(3, size)**-1 # 3 here actually means 2 :) according to scipy docs

def noncentrality(snrs, prefactor):
	return prefactor * random.rand(len(snrs)) * snrs**2 # FIXME power depends on dimensionality of the bank and the expectation for the mismatch for real signals
	#return prefactor * random.power(1, len(snrs)) * snrs**2 # FIXME power depends on dimensionality of the bank and the expectation for the mismatch for real signals

def chisq_distribution(df, non_centralities, size):
	out = numpy.empty((len(non_centralities) * size,))
	for i, nc in enumerate(non_centralities):
		out[i*size:(i+1)*size] = random.noncentral_chisquare(df, nc, size)
	return out

def populate_injections(bindict, prefactor = .3, df = 24, size = 1000000, verbose = True):
	for i, k in enumerate(bindict):
		binarr = bindict[k]
		if verbose:
			print >> sys.stderr, "synthesizing injections for ", k
		minsnr = binarr.bins[0].upper().min()
		random.seed(i) # FIXME changes as appropriate
		snrs = snr_distribution(size, minsnr)
		ncs = noncentrality(snrs, prefactor)
		chisqs = chisq_distribution(df, ncs, 1) / df
		for snr, chisq in zip(snrs, chisqs):
			binarr[snr, chisq**.5 / snr] += 1

#
# Utility functions
#

def smooth_bins(bA):
	#FIXME what is this window supposed to be??
	wn = rate.gaussian_window2d(15, 15, sigma = 8)
	rate.filter_array(bA.array.T, wn)
	sum = bA.array.sum()
	if sum != 0:
		bA.array /= sum # not the same as the to_pdf() method

def linearize_array(arr):
	return arr.reshape((1,arr.shape[0] * arr.shape[1]))

def get_nonzero(arr):
	# zeros, nans and infs mean that either the numerator or denominator of the
	# likelihood was zero.  We do not include those
	zb = arr != 0
	nb = numpy.isnan(arr)
	ib = numpy.isinf(arr)
	return arr[zb - nb - ib]

def count_to_rank(val, offset = 100):
	return numpy.log(val) + offset #FIXME It should be at least the absolute value of the minimum of the log of all bins

#
# Function to compute the fap in a given file
#

def set_fap(options, Far, f, rankoffset):
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
		Far.updateFAPmap(ifoset, rankoffset)

		# FIXME abusing FAR column
		connection.cursor().execute(fap_query(ifoset, ifos))
		connection.commit()

	# all finished
	connection.commit()
	connection.close()
	dbtables.put_connection_filename(f, working_filename, verbose = options.verbose)

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
	def __init__(self, livetime, trials_factor, counts = None, injections = None):
		self.injections = injections
		self.counts = counts
		self.livetime = livetime
		self.trials_factor = trials_factor
		self.trials_table = {}

	def updateFAPmap(self, instruments, rank_offset):
		if self.counts is None:
			raise InputError, "must provide background bins file"
		self.rank_offset = rank_offset

		#
		# the target FAP resolution is 1 part in 10^7.  So depending on
		# how many instruments we have we have to take the nth root of
		# that number to set the scale in each detector. This is purely
		# for memory/CPU requirements
		#

		targetlen = int(1e7**(1. / len(instruments)))
		nonzerorank = {}

		for ifo in instruments:
			# FIXME don't repeat calc by checking if it has been done??
			nonzerorank[ifo] = count_to_rank(get_nonzero(linearize_array(self.counts[ifo+"_snr_chi"].array)), offset = self.rank_offset)

		nonzerorank = self.rankBins(nonzerorank, targetlen)
		self.ranks, weights = self.possible_ranks_array(nonzerorank)
		fap, self.fap_from_rank = self.CDFinterp(self.ranks, weights)

	def set_trials_table(self, connection):

		connection.cursor().execute('CREATE INDEX template_index ON sngl_inspiral(mass1,mass2,chi)')

		for ifos, tsid, mass1, mass2, chi, count in connection.cursor().execute('SELECT ifos, coinc_event.time_slide_id, mass1, mass2, chi, count(*) / nevents FROM sngl_inspiral JOIN coinc_event_map ON coinc_event_map.event_id == sngl_inspiral.event_id JOIN coinc_inspiral ON coinc_inspiral.coinc_event_id == coinc_event_map.coinc_event_id JOIN coinc_event ON coinc_event.coinc_event_id == coinc_event_map.coinc_event_id  WHERE coinc_event_map.table_name = "sngl_inspiral" GROUP BY mass1, mass2, chi, ifos;'):
			self.trials_table[(ifos, tsid, mass1, mass2, chi)] = count
		connection.cursor().execute('DROP INDEX template_index')
		connection.commit()

	def increment_trials_table(self, n):
		for k in self.trials_table:
			self.trials_table[k] += n

	def rankBins(self, vec, size):
		out = {}
		minrank = min([v.min() for v in vec.values()])
		maxrank = max([v.max() for v in vec.values()])
		for ifo, ranks in vec.items():
			out[ifo] = rate.BinnedArray(rate.NDBins((rate.LinearBins(minrank, maxrank, size),)))
			for r in ranks:
				out[ifo][r,]+=1
		return out

	def possible_ranks_array(self, bAdict):
		# start with an identity array to seed the outerproduct chain
		ranks = numpy.array([1])
		vals = numpy.array([1])
		for ifo, bA in bAdict.items():
			ranks = numpy.outer(ranks, bA.centres()[0])
			vals = numpy.outer(vals, bA.array)
			ranks = ranks.reshape((ranks.shape[0] * ranks.shape[1],))
			vals = vals.reshape((vals.shape[0] * vals.shape[1],))
		vals = vals[ranks.argsort()]
		ranks.sort()
		return ranks, vals

	def CDFinterp(self, ranks, weights = None):
		if weights is None:
			FAP = (numpy.arange(len(ranks)) + 1.) / len(vec)
		else:
			FAP = weights.cumsum()
			FAP /= FAP[-1]
		# Rather than putting 0 for FAPS we cannot estimate, set it to the mimimum non zero value which is more meaningful
		return FAP, interpolate.interp1d(ranks, FAP, fill_value = (FAP[FAP !=0.0]).min(), bounds_error = False)

	# Method only works if counts is not None:
	def compute_rank(self, snr_chisq_dict):
		if self.counts is None:
			raise InputError, "must provide background bins file"
		rank = 1
		for ifo, (snr,chisq) in snr_chisq_dict.items():
			val = count_to_rank(self.counts[ifo+"_snr_chi"][snr, chisq**.5 / snr], offset = self.rank_offset)
			rank *= val
		if rank > self.ranks[-2]:
			rank = self.ranks[-2]
		return rank

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

	def FAR_from_FAP(self, fap, n = 1):
		# the n = 1 case can be done exactly.  That is good since it is
		# the most important.
		if n == 1:
			return 0. - numpy.log(1. - fap) / self.livetime
		if n > 1 and n <= 100:
			nvec = numpy.logspace(-12, numpy.log10(n + 10. * n**.5), 100)
		else:
			nvec = numpy.logspace(numpy.log10(n - 10. * n**.5), numpy.log10(n + 10. * n**.5), 100)
		FAPS = 1. - poisson.cdf(n,nvec)
		#FIXME is this right since nvec is log spaced?
		interp = interpolate.interp1d(FAPS, nvec / self.livetime)
		if fap < FAPS[1]:
			return 0.
		if fap > FAPS[-1]:# This means that the FAP has gone off the edge.  We will bump it down because we don't really care about this being right.
			fap = FAPS[-1]
		return interp(fap)[0]

	def compute_far(self, fap, n):
		if fap == 0.0:
			far = 0.
		else:
			far = self.FAR_from_FAP(fap, n)
		return far


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


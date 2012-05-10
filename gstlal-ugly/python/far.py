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


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


import itertools
import numpy
from scipy import interpolate
from scipy import stats
import sys
try:
	import sqlite3
except ImportError:
	# pre 2.5.x
	from pysqlite2 import dbapi2 as sqlite3
sqlite3.enable_callback_tracebacks(True)


from glue import iterutils
from glue.ligolw import ligolw
from glue.ligolw import ilwd
from glue.ligolw import param as ligolw_param
from glue.ligolw import lsctables
from glue.ligolw import utils
from glue import segments
from glue.segmentsUtils import vote
from pylal import ligolw_burca_tailor
from pylal import ligolw_burca2
from pylal import llwapp
from pylal import rate


#
# =============================================================================
#
#                             Trials Table Object
#
# =============================================================================
#


#
# Trials table
#


class TrialsTable(dict):
	"""
	A class to store the trials table from a coincident inspiral search
	with the intention of computing the false alarm probabiliy of an event after N
	trials.  This is a subclass of dict.  The trials table is keyed by the
	detectors that partcipated in the coincidence and the time slide id.
	"""
	class TrialsTableTable(lsctables.table.Table):
		tableName = "gstlal_trials:table"
		validcolumns = {
			"ifos": "lstring",
			"time_slide_id": "ilwd:char",
			"count": "int_8s"
		}
		class RowType(object):
			__slots__ = ("ifos", "time_slide_id", "count")

			def get_ifos(self):
				return lsctables.instrument_set_from_ifos(self.ifos)

			def set_ifos(self, ifos):
				self.ifos = lsctables.ifos_from_instrument_set(ifos)

			@property
			def key(self):
				return frozenset(self.get_ifos()), self.time_slide_id

			@classmethod
			def from_item(cls, ((ifos, time_slide_id), value)):
				self = cls()
				self.set_ifos(ifos)
				self.time_slide_id = time_slide_id
				self.count = value
				return self

	def __add__(self, other):
		out = type(self)()
		for k in self:
			out[k] = self[k]
		for k in other:
			try:
				out[k] += other[k]
			except KeyError:
				out[k] = other[k]
		return out

	def sum_over_time_slides(self):
		out = {}
		for (ifos, time_slide_id), count in self.items():
			try:
				out[ifos] += count
			except:
				out[ifos] = count
		return out

	def from_db(self, connection):
		"""
		Increment the trials table from values stored in the database
		found in "connection"
		"""		
		for ifos, tsid, count in connection.cursor().execute('SELECT ifos, coinc_event.time_slide_id AS tsid, count(*) / nevents FROM sngl_inspiral JOIN coinc_event_map ON coinc_event_map.event_id == sngl_inspiral.event_id JOIN coinc_inspiral ON coinc_inspiral.coinc_event_id == coinc_event_map.coinc_event_id JOIN coinc_event ON coinc_event.coinc_event_id == coinc_event_map.coinc_event_id  WHERE coinc_event_map.table_name = "sngl_inspiral" GROUP BY tsid, ifos;'):
			ifos = frozenset(lsctables.instrument_set_from_ifos(ifos))
			tsid = ilwd.get_ilwdchar(tsid)
			try:
				self[ifos, tsid] += count
			except KeyError:
				self[ifos, tsid] = count
		connection.commit()

	def increment(self, n):
		"""
		Increment all keys by n
		"""
		for k in self:
			self[k] += n

	@classmethod
	def from_xml(cls, xml):
		"""
		A class method to create a new instance of a TrialsTable from
		an xml representation of it.
		"""
		self = cls()
		for row in lsctables.table.get_table(xml, self.TrialsTableTable.tableName):
			self[row.key] = row.count
		return self

	def to_xml(self):
		"""
		A method to write this instance of a trials table to an xml
		representation.
		"""
		xml = lsctables.New(self.TrialsTableTable)
		for item in self.items():
			xml.append(xml.RowType.from_item(item))
		return xml


lsctables.TableByName[lsctables.table.StripTableName(TrialsTable.TrialsTableTable.tableName)] = TrialsTable.TrialsTableTable


#
# =============================================================================
#
#                 Parameter Distributions Book-Keeping Object
#
# =============================================================================
#


#
# Paramter Distributions
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
		self.raw_distributions = ligolw_burca_tailor.CoincParamsDistributions(**self.binnings)
		self.smoothed_distributions = ligolw_burca_tailor.CoincParamsDistributions(**self.binnings)
		self.likelihood_pdfs = {}

	def __add__(self, other):
		out = type(self)()
		out.raw_distributions += self.raw_distributions
		out.raw_distributions += other.raw_distributions
		#FIXME do we also add the smoothed distributions??
		return out

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

	def compute_single_instrument_background(self, verbose = False):
		# initialize a likelihood ratio evaluator
		likelihood_ratio_evaluator = ligolw_burca2.LikelihoodRatio(self.smoothed_distributions)

		# reduce typing
		background = self.smoothed_distributions.background_rates
		injections = self.smoothed_distributions.injection_rates

		self.likelihood_pdfs.clear()
		for param in background:
			# FIXME only works if there is a 1-1 relationship between params and instruments
			instrument = param.split("_")[0]

			if verbose:
				print >>sys.stderr, "updating likelihood background for %s" % instrument

			likelihoods = injections[param].array / background[param].array
			# ignore infs and nans because background is never
			# found in those bins.  the boolean array indexing
			# flattens the array
			finite_likelihoods = numpy.isfinite(likelihoods)
			likelihoods = likelihoods[finite_likelihoods]
			background_likelihoods = background[param].array[finite_likelihoods]
			sortindex = likelihoods.argsort()
			likelihoods = likelihoods[sortindex]
			background_likelihoods = background_likelihoods[sortindex]

			s = background_likelihoods.cumsum() / background_likelihoods.sum()
			# restrict range to the 1-1e10 confidence interval to make the best use of our bin resolution
			minlikelihood = likelihoods[s.searchsorted(1e-10)]
			maxlikelihood = likelihoods[s.searchsorted(1 - 1e-10)]
			if minlikelihood == 0:
				minlikelihood = likelihoods[likelihoods != 0].min()
			# construct PDF
			# FIXME:  because the background array contains
			# probabilities and not probability densities, the
			# likelihood_pdfs contain probabilities and not
			# densities, as well, when this is done
			self.likelihood_pdfs[instrument] = rate.BinnedArray(rate.NDBins((rate.LogarithmicPlusOverflowBins(minlikelihood, maxlikelihood, self.target_length),)))
			for coords in iterutils.MultiIter(*background[param].bins.centres()):
				likelihood = likelihood_ratio_evaluator({param: coords})
				if numpy.isfinite(likelihood):
					self.likelihood_pdfs[instrument][likelihood,] += background[param][coords]

	@classmethod
	def from_xml(cls, xml, name):
		self = cls()
		self.raw_distributions, process_id = ligolw_burca_tailor.CoincParamsDistributions.from_xml(xml, name)
		# FIXME:  produce error if binnings don't match this class's binnings attribute?
		binnings = dict((param, self.raw_distributions.zero_lag_rates[param].bins) for param in self.raw_distributions.zero_lag_rates)
		self.smoothed_distributions = ligolw_burca_tailor.CoincParamsDistributions(**binnings)
		return self, process_id

	@classmethod
	def from_filenames(cls, filenames, verbose = False):
		self = cls()
		self.raw_distributions, seglists = ligolw_burca_tailor.load_likelihood_data(filenames, u"gstlal_inspiral_likelihood", verbose = verbose)
		# FIXME:  produce error if binnings don't match this class's binnings attribute?
		binnings = dict((param, self.raw_distributions.zero_lag_rates[param].bins) for param in self.raw_distributions.zero_lag_rates)
		self.smoothed_distributions = ligolw_burca_tailor.CoincParamsDistributions(**binnings)
		return self, seglists

	def to_xml(self, process, name):
		return self.raw_distributions.to_xml(process, name)


#
# =============================================================================
#
#                       False Alarm Book-Keeping Object
#
# =============================================================================
#


def possible_ranks_array(likelihood_pdfs, ifo_set, targetlen):
	# find the minimum value for the binning that we care about.
	# this is product of all the peak values of likelihood,
	# times the smallest maximum likelihood value, divided by the
	# product of all maximum likelihood values
	Lp = []
	Lj = []
	# loop over all ifos
	for ifo in ifo_set:
		likelihood_pdf = likelihood_pdfs[ifo]
		# FIXME lower instead of centres() to avoid inf in the last bin
		ranks = likelihood_pdf.bins.lower()[0]
		vals = likelihood_pdf.array
		# sort likelihood values from lowest probability to highest
		ranks = ranks[vals.argsort()]
		# save peak likelihood
		Lp.append(ranks[-1])
		# save maximum likelihood value
		Lj.append(max(ranks))
	Lp = numpy.array(Lp)
	Lj = numpy.array(Lj)
	# create product of all maximum likelihood values
	L = numpy.exp(sum(numpy.log(Lj)))
	# compute minimum bin value we care about
	Lmin = numpy.exp(sum(numpy.log(Lp))) * min(Lj) / L
	# divide by a million for safety
	Lmin *= 1e-6

	# start with an identity array to seed the outerproduct chain
	ranks = numpy.array([1.0])
	vals = numpy.array([1.0])
	# FIXME:  probably only works because the pdfs aren't pdfs but probabilities
	for ifo in ifo_set:
		likelihood_pdf = likelihood_pdfs[ifo]
		# FIXME lower instead of centres() to avoid inf in the last bin
		ranks = numpy.outer(ranks, likelihood_pdf.bins.lower()[0])
		vals = numpy.outer(vals, likelihood_pdf.array)
		ranks = ranks.reshape((ranks.shape[0] * ranks.shape[1],))
		# FIXME nans arise from inf * 0.  Do we want these to be 0?
		#ranks[numpy.isnan(ranks)] = 0.0
		vals = vals.reshape((vals.shape[0] * vals.shape[1],))
		# rebin the outer-product
		minlikelihood = max(min(ranks[ranks != 0]), Lmin)
		maxlikelihood = max(ranks)
		new_likelihood_pdf = rate.BinnedArray(rate.NDBins((rate.LogarithmicPlusOverflowBins(minlikelihood, maxlikelihood, targetlen),)))
		for rank,val in zip(ranks,vals):
			new_likelihood_pdf[rank,] += val
		ranks = new_likelihood_pdf.bins.lower()[0]
		vals = new_likelihood_pdf.array

	# FIXME the size of these is targetlen which has to be small
	# since it is squared in the outer product.  We can afford to
	# store a 1e6 array.  Maybe we should try to make the resulting
	# pdf bigger somehow?
	return new_likelihood_pdf


#
# Class to handle the computation of FAPs/FARs
#


class LocalRankingData(object):
	def __init__(self, livetime_seg, distribution_stats, trials_table = None, target_length = 1000):
		self.distribution_stats = distribution_stats
		if trials_table is None:
			self.trials_table = TrialsTable()
		else:
			self.trials_table = trials_table
		self.livetime_seg = livetime_seg

		#
		# the target FAP resolution is 1000 bins by default. This is purely
		# for memory/CPU requirements
		#

		self.target_length = target_length

	def __iadd__(self, other):
		self.distribution_stats += other.distribution_stats
		self.trials_table += other.trials_table
		if self.livetime_seg[0] is None and other.livetime_seg[0] is not None:
			minstart = other.livetime_seg[0]
		elif self.livetime_seg[0] is not None and other.livetime_seg[0] is None:
			minstart = self.livetime_seg[0]
		# correctly handles case where both or neither are None
		else:
			minstart = min(self.livetime_seg[0], other.livetime_seg[0])
		# None is always less than everything else, so this is okay
		maxend = max(self.livetime_seg[1], other.livetime_seg[1])
		self.livetime_seg = segments.segment(minstart, maxend)

		return self

	@classmethod
	def from_xml(cls, xml, name = u"gstlal_inspiral_likelihood"):
		llw_elem, = [elem for elem in xml.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") and elem.getAttribute(u"Name") == u"%s:gstlal_inspiral_FAR" % name]
		distribution_stats, process_id = DistributionsStats.from_xml(llw_elem, name)
		# the code that writes these things has put the
		# livetime_seg into the out segment in the search_summary
		# table.  uninitialized segments get recorded as
		# [None,None), which breaks the .get_out() method of the
		# row class, so we need to reconstruct the segment
		# ourselves to trap that error without worrying that we're
		# masking other bugs with the try/except
		search_summary_row, = (row for row in lsctables.table.get_table(xml, lsctables.SearchSummaryTable.tableName) if row.process_id == process_id)
		try:
			# FIXME:  the latest glue can handle (None,None) itself
			livetime_seg = search_summary_row.get_out()
		except TypeError:
			livetime_seg = segments.segment(None, None)
		self = cls(livetime_seg, distribution_stats = distribution_stats, trials_table = TrialsTable.from_xml(llw_elem))
		
		# pull out the joint likelihood arrays if they are present
		for ba_elem in [elem for elem in xml.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") and "_joint_likelihood" in elem.getAttribute(u"Name")]:
			ifo_set = frozenset(lsctables.instrument_set_from_ifos(ba_elem.getAttribute(u"Name").split("_")[0]))
			self.joint_likelihood_pdfs[ifo_set] = rate.binned_array_from_xml(ba_elem, ba_elem.getAttribute(u"Name").split(":")[0])
		
		return self, process_id

	@classmethod
	def from_filenames(cls, filenames, name = u"gstlal_inspiral_likelihood", verbose = False):
		self, process_id = LocalRankingData.from_xml(utils.load_filename(filenames[0], verbose = verbose), name = name)
		for f in filenames[1:]:
			s, p = LocalRankingData.from_xml(utils.load_filename(f, verbose = verbose), name = name)
			self += s
		return self
		
	def to_xml(self, process, name = u"gstlal_inspiral_likelihood"):
		xml = ligolw.LIGO_LW({u"Name": u"%s:gstlal_inspiral_FAR" % name})
		xml.appendChild(self.trials_table.to_xml())
		xml.appendChild(self.distribution_stats.to_xml(process, name))
		for key in self.joint_likelihood_pdfs:
			ifostr = lsctables.ifos_from_instrument_set(key).replace(",","")
			xml.appendChild(rate.binned_array_to_xml(self.joint_likelihood_pdfs[key], "%s_joint_likelihood" % (ifostr,)))
		return xml

	def smooth_distribution_stats(self, verbose = False):
		if self.distribution_stats is not None:
			# FIXME:  this results in the
			# .smoothed_distributions object containing
			# *probabilities* not probability densities. this
			# might be changed in the future.
			self.distribution_stats.finish(verbose = verbose)

	def compute_joint_instrument_background(self, remap, instruments = None, verbose = False):
		# first get the single detector distributions
		self.distribution_stats.compute_single_instrument_background(verbose = verbose)

		self.joint_likelihood_pdfs.clear()

		# calculate all of the possible ifo combinations with at least
		# 2 detectors in order to get the joint likelihood pdfs
		likelihood_pdfs = self.distribution_stats.likelihood_pdfs
		if instruments is None:
			instruments = likelihood_pdfs.keys()
		for n in range(2, len(instruments) + 1):
			for ifos in iterutils.choices(instruments, n):
				ifo_set = frozenset(ifos)
				remap_set = remap.setdefault(ifo_set, ifo_set)

				if verbose:
					print >>sys.stderr, "computing joint likelihood background for %s remapped to %s" % (lsctables.ifos_from_instrument_set(ifo_set), lsctables.ifos_from_instrument_set(remap_set))

				# only recompute if necessary, some choices
				# may not be if a certain remap set is
				# provided
				if remap_set not in self.joint_likelihood_pdfs:
					self.joint_likelihood_pdfs[remap_set] = possible_ranks_array(likelihood_pdfs, remap_set, self.target_length)

				self.joint_likelihood_pdfs[ifo_set] = self.joint_likelihood_pdfs[remap_set]


class RankingData(object):
	def __init__(self, local_ranking_data):
		self.far_interval = 3600.0	# seconds

		# ensure the trials tables' keys match the likelihood
		# histograms' keys
		assert set(local_ranking_data.joint_likelihood_pdfs) == set(local_ranking_data.trials_table.sum_over_time_slides())

		# copy likelihood ratio PDFs
		self.joint_likelihood_pdfs = dict((key, copy.deepcopy(value)) for key, value in local_ranking_data.joint_likelihood_pdfs.items())

		# copy trials table counts
		self.trials_table = TrialsTable()
		self.trials_table += local_ranking_data.trials_table

		# copy livetime segment
		self.livetime_seg = local_ranking_data.livetime_seg

		self.ccdf_interpolator = {}
		self.minrank = {}
		self.maxrank = {}


	@classmethod
	def from_xml(cls, xml, name = u"gstlal_inspiral"):
		llw_elem, = [elem for elem in xml.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") and elem.getAttribute(u"Name") == u"%s:gstlal_inspiral_ranking_data" % name]

		class fake_local_ranking_data(object):
			pass
		fake_local_ranking_data = fake_local_ranking_data()
		fake_local_ranking_data.trials_table = TrialsTable.from_xml(llw_elem)
		fake_local_ranking_data.joint_likelihood_pdfs = {}
		for key in fake_local_ranking_data.trials_table:
			ifostr = lsctables.ifos_from_instrument_set(key).replace(",","")
			fake_local_ranking_data.joint_likelihood_pdfs[key] = rate.binned_array_from_xml(llw_elem, ifostr)

		# the code that writes these things has put the
		# livetime_seg into the out segment in the search_summary
		# table.  uninitialized segments get recorded as
		# [None,None), which breaks the .get_out() method of the
		# row class, so we need to reconstruct the segment
		# ourselves to trap that error without worrying that we're
		# masking other bugs with the try/except
		process_id = ligolw_param.get_pyvalue(llw_elem, u"process_id")
		search_summary_row, = (row for row in lsctables.table.get_table(xml, lsctables.SearchSummaryTable.tableName) if row.process_id == process_id)
		fake_local_ranking_data.livetime_seg = search_summary_row.get_out()

		self = cls(fake_local_ranking_data)

		return self, process_id


	def to_xml(self, process, search_summary, name = u"gstlal_inspiral"):
		xml = ligolw.LIGO_LW({u"Name": u"%s:gstlal_inspiral_ranking_data" % name})
		xml.appendChild(ligolw_param.new_param(u"process_id", u"ilwd:char", process.process_id))
		xml.appendChild(self.trials_table.to_xml())
		for key in self.joint_likelihood_pdfs:
			ifostr = lsctables.ifos_from_instrument_set(key).replace(",","")
			xml.appendChild(rate.binned_array_to_xml(self.joint_likelihood_pdfs[key], ifostr))
		assert search_summary.process_id == process.process_id
		search_summary.set_out(self.livetime_seg)
		return xml


	def __iadd__(self, other):
		our_trials = self.trials_table.sum_over_time_slides()
		other_trials = other.trials_table.sum_over_time_slides()

		our_keys = set(self.joint_likelihood_pdfs)
		other_keys  = set(other.joint_likelihood_pdfs)

		# PDFs that only we have are unmodified
		pass

		# PDFs that only the new data has get copied verbatim
		for k in other_keys - our_keys:
			self.joint_likelihood_pdfs[k] = copy.deepcopy(other.joint_likelihood_pdfs[k])

		# PDFs that we have and are in the new data get replaced
		# with the weighted sum, re-binned
		for k in our_keys & other_keys:
			minself, maxself, nself = self.joint_likelihood_pdfs[k].bins[0].min, self.joint_likelihood_pdfs[k].bins[0].max, self.joint_likelihood_pdfs[k].bins[0].n
			minother, maxother, nother = other.joint_likelihood_pdfs[k].bins[0].min, other.joint_likelihood_pdfs[k].bins[0].max, other.joint_likelihood_pdfs[k].bins[0].n
			new_joint_likelihood_pdf =  rate.BinnedArray(rate.NDBins((rate.LogarithmicPlusOverflowBins(min(minself, minother), max(maxself, maxother), max(nself, nother)),)))

			for x in self.joint_likelihood_pdfs[k].centres()[0]:
				new_joint_likelihood_pdf[x,] += self.joint_likelihood_pdfs[k][x,] * float(our_trials[k]) / (our_trials[k] + other_trials[k])
			for x in other.joint_likelihood_pdfs[k].centres()[0]:
				new_joint_likelihood_pdf[x,] += local_ranking_data.joint_likelihood_pdfs[k][x,] * float(other_trials[k]) / (our_trials[k] + other_trials[k])

			self.joint_likelihood_pdf[k] = new_joint_likelihood_pdf

		# combined trials counts
		self.trials_table += other.trials_table

		# merge livetime segments.  let the code crash if they're disjoint
		self.livetime_seg |= other.livetime_seg

	def compute_joint_cdfs(self):
		self.minrank.clear()
		self.maxrank.clear()
		self.ccdf_interpolator.clear()
		for key, lpdf in self.joint_likelihood_pdfs.items():
			ranks = lpdf.bins.lower()[0]
			weights = lpdf.array
			# complementary cumulative distribution function
			ccdf = weights[::-1].cumsum()[::-1]
			ccdf /= ccdf[0]
			self.ccdf_interpolator[key] = interpolate.interp1d(ranks, ccdf)
			# record min and max ranks so we know which end of the ccdf to use when we're out of bounds
			self.minrank[key] = (min(ranks), ccdf[0])
			self.maxrank[key] = (max(ranks), ccdf[-1])

	def fap_from_rank(self, rank, ifos, tsid):
		ifos = frozenset(ifos)
		# FIXME:  doesn't check that rank is a scalar
		if rank >= self.maxrank[ifos][0]:
			return self.maxrank[ifos][1]
		if rank <= self.minrank[ifos][0]:
			return self.minrank[ifos][1]
		fap = float(self.ccdf_interpolator[ifos](rank))
		try:
			trials = max(int(self.trials_table[ifos]), 1)
		except KeyError:
			trials = 1
		# normalize to the far interval
		if self.far_interval is not None:
			trials *= self.far_interval / float(abs(self.livetime_seg))
		return 1.0 - (1.0 - fap)**trials

	def compute_far(self, fap):
		if fap == 0.:
			return 0.
		if self.far_interval is not None:
			# use far interval for livetime
			livetime = self.far_interval
		else:
			livetime = float(abs(self.livetime_seg))
		return 0. - numpy.log(1. - fap) / livetime


#
# =============================================================================
#
#                                    Other
#
# =============================================================================
#


#
# Function to compute the fap in a given file
#


def set_fap(Far, f, tmp_path = None, verbose = False):
	"""
	Function to set the false alarm probability for a single database
	containing the usual inspiral tables.

	Far = LocalRankingData class instance
	f = filename of the databse (e.g.something.sqlite) 
	tmp_path = the local disk path to copy the database to in
		order to avoid sqlite commands over nfs 
	verbose = be verbose
	"""
	# FIXME this code should be moved into a method of the LocalRankingData class once other cleaning is done
	from glue.ligolw import dbtables

	# set up working file names
	working_filename = dbtables.get_connection_filename(f, tmp_path = tmp_path, verbose = verbose)
	connection = sqlite3.connect(working_filename)

	# define fap function
	connection.create_function("fap", 3, lambda rank, ifostr, tsid: Far.fap_from_rank(rank, lsctables.instrument_set_from_ifos(ifostr), ilwd.get_ilwdchar(tsid)))

	# FIXME abusing false_alarm_rate column, move for a false_alarm_probability column??
	connection.cursor().execute("UPDATE coinc_inspiral SET false_alarm_rate = (SELECT fap(coinc_event.likelihood, coinc_inspiral.ifos, coinc_event.time_slide_id) FROM coinc_event WHERE coinc_event.coinc_event_id == coinc_inspiral.coinc_event_id)")
	connection.commit()

	# all finished
	connection.commit()
	connection.close()
	dbtables.put_connection_filename(f, working_filename, verbose = verbose)


#
# Function to compute the far in a given file
#


def set_far(Far, f, tmp_path = None, verbose = False):
	from glue.ligolw import dbtables

	working_filename = dbtables.get_connection_filename(f, tmp_path = tmp_path, verbose = verbose)
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
	dbtables.put_connection_filename(f, working_filename, verbose = verbose)


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

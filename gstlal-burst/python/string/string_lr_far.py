####################
# Modules for calculating and storing likelihood ratio density and FAPFARs 
# for the cosmic string search.
#


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


import numpy
import random

from lal import rate
from lalburst import snglcoinc

from ligo.lw import ligolw
from ligo.lw import param as ligolw_param
from ligo.lw import utils as ligolw_utils 


from gstlal import string_extrinsics


#
# =============================================================================
#
#                              Likelihood ratio densities
#
# =============================================================================
#


#
# Numerator & denominator base class
#


class LnLRDensity(snglcoinc.LnLRDensity):
	# SNR, chi^2 binning definition
	snr2_chi2_binning = rate.NDBins((rate.ATanLogarithmicBins(10, 1e7, 801), rate.ATanLogarithmicBins(.1, 1e4, 801)))
	
	def __init__(self, instruments, snr_threshold, min_instruments=2):
		# check input
		if min_instruments < 2:
			raise ValueError("min_instruments=%d must be >=2" % min_instruments)
		if min_instruments > len(instruments):
			raise ValueError("not enough instruments (%s) to satisfy min_instruments=%d" % (", ".join(sorted(instruments)), min_instruments))
		if snr_threshold <= 0.:
			raise ValueError("SNR threshold = %g  must be >0" % snr_threshold)

		self.min_instruments = min_instruments
		self.snr_threshold = snr_threshold
		self.densities = {}
		for instrument in instruments:
			self.densities["%s_snr2_chi2" % instrument] = rate.BinnedLnPDF(self.snr2_chi2_binning)

	def __call__(self, **params):
		try:
			interps = self.interps
		except AttributeError:
			self.mkinterps()
			interps = self.interps
		return sum(interps[param](*value) for param, value in params.items()) if params else NegInf

	def __iadd__(self, other):
		if type(self) != type(other) or set(self.densities) != set(other.densities):
			raise TypeError("cannot add %s and %s" % (type(self), type(other)))
		for key, lnpdf in self.densities.items():
			lnpdf += other.densities[key]
		try:
			del self.interps
		except AttributeError:
			pass
		return self

	def increment(self, weight = 1.0, **params):
		instruments = params["snrs"].items()
		for instrument in instrumets:
			self.densities["%s_snr2_chi2" % instrument].count[params["snrs"][instrument], params["chi2s_over_snr2s"][instrument]] += weight

	def copy(self):
		new = type(self)([])
		for key, lnpdf in self.densities.items():
			new.densities[key] = lnpdf.copy()
		return new

	def mkinterps(self):
		self.interps = dict((key, lnpdf.mkinterp()) for key, lnpdf in self.densities.items())

	def finish(self):
		for key, lnpdf in self.densities.items():
			if key.endswith("_snr2_chi2"):
				rate.filter_array(lnpdf.array, rate.gaussian_window(11, 11, sigma = 20))
			else:
				# shouldn't get here
				raise Exception
			lnpdf.normalize()
		self.mkinterps()

	def to_xml(self, name):
		xml = super(LnLRDensity, self).to_xml(name)
		instruments = set(key.split("_", 1)[0] for key in self.densities if key.endswith("_snr2_chi2"))
		xml.appendChild(ligolw_param.Param.from_pyvalue(u"instruments", lsctables.instrumentsproperty.set(instruments)))
		xml.appendChild(ligolw_param.Param.from_pyvalue(u"min_instruments", self.min_instruments))
		xml.appendChild(ligolw_param.Param.from_pyvalue(u"snr_threshold", self.snr_threshold))
		for key, lnpdf in self.densities.items():
			xml.appendChild(lnpdf.to_xml(key))
		return xml

	@classmethod
	def from_xml(cls, xml, name):
		xml = cls.get_xml_root(xml, name)
		self = cls(
			instruments = lsctables.instrumentsproperty.get(ligolw_param.get_pyvalue(xml, u"instruments")),
			min_instruments = ligolw_param.get_pyvalue(xml, u"min_instruments"),
			snr_threshold = ligolw_param.get_pyvalue(xml, u"snr_threshold")
			)
		for key in self.densities:
			self.densities[key] = rate.BinnedLnPDF.from_xml(xml, key)
		return self


#
# Likelihood ratio density (numerator)
#


class LnSignalDensity(LnLRDensity):
	def add_signal_model(self, prefactors_range = (0.001, 0.30), inv_snr_pow = 4.):
		# normalize to 10 *mi*llion signals. This count makes the
		# density estimation code choose a suitable kernel size
		string_extrinsics.NumeratorSNRCHIPDF.add_signal_model(self.densities["snr2_chi2"], 10000000., prefactors_range, inv_snr_pow = inv_snr_pow, snr_min = self.snr_threshold)
		self.densities["snr2_chi2"].normalize()

	def to_xml(self, name):
		xml = super(LnSignalDensity, self).to_xml(name)
		return xml

	@classmethod
	def from_xml(cls, xml, name):
		xml = cls.get_xml_root(xml, name)
		self = super(LnSignalDensity, cls).from_xml(xml, name)
		return self


#
# Likelihood ratio density (denominator)
#


class LnNoiseDensity(LnLRDensity):
	def __call__(self, snrs, chi2s_over_snr2s):
		lnP = 0.0

		# Evaluate P(snrs, chi2s | noise)
		# It is assumed that SNR and chi^2 values are
		# independent between detectors, and furthermore
		# independent from their sensitivities.
		interps = self.interps
		return lnP + sum(interps["%s_snr2_chi2" % instrument](snrs[instrument], chi2_over_snr2) for instrument, chi2_over_snr2 in chi2s_over_snr2s.items())

	def candidate_count_model(self):
		pass

	def random_params(self):
		"""
		Generator that yields an endless sequence of randomly
		generated candidate parameters.  NOTE: the parameters will
		be within the domain of the repsective binnings but are not
		drawn from the PDF stored in those binnings --- this is not
		an MCMC style sampler.  Each value in the sequence is a
		three-element tuple.  The first two elements of each tuple
		provide the *args and **kwargs values for calls to this PDF
		or the numerator PDF or the ranking statistic object.  The
		final is the natural logarithm (up to an arbitrary
		constant) of the PDF from which the parameters have been
		drawn evaluated at the point described by the *args and
		**kwargs.

		See also:

		random_sim_params()

		The sequence is suitable for input to the .ln_lr_samples()
		log likelihood ratio generator.
		"""
		snr_slope = 0.8 / len(self.instruments)**3

		snr2chi2gens = dict((instrument, iter(self.densities["%s_snr2_chi2" % instrument].bins.randcoord(ns = (snr_slope, 1.), domain = (slice(self.snr_threshold, None), slice(None, None)))).next) for instrument in self.instruments)
		def nCk(n, k):
			return math.factorial(n) // math.factorial(k) // math.factorial(n - k)
		while 1:
			instruments = tuple(instrument for instrument, rate in rates.items() if rate > 0)
			assert len(instruments) < self.min_instruments, "number of instruments smaller than min_instruments"
			seq = sum((snr2chi2gens[instrument]() for instrument in instruments), ())
			# set params
			for instrument, value in zip(instruments, seq[0::2]):
				params["%s_snr2_chi2" % instrument] = (value[0], value[1])
			yield (), params, seq[1::2]

	def to_xml(self, name):
		xml = super(LnNoiseDensity, self).to_xml(name)
		xml.appendChild(self.triggerrates.to_xml(u"triggerrates"))
		return xml

	@classmethod
	def from_xml(cls, xml, name):
		xml = cls.get_xml_root(xml, name)
		self = super(LnNoiseDensity, cls).from_xml(xml, name)
		self.triggerrates = trigger_rate.triggerrates.from_xml(xml, u"triggerrates")
		self.triggerrates.coalesce()    # just in case
		return self


#
# =============================================================================
#
#                             Ranking Statistic PDF
#
# =============================================================================
#


#
# Class to compute ranking statistic PDFs for background-like and
# signal-like populations
#

class RankingStatPDF(object):
	ligo_lw_name_suffix = u"gstlal_string_rankingstatpdf"

	@staticmethod
	def density_estimate(lnpdf, name, kernel = rate.gaussian_window(4.)):
		"""
		For internal use only.
		"""
		assert not numpy.isnan(lnpdf.array).any(), "%s log likelihood ratio PDF contain NaNs" % name
		rate.filter_array(lnpdf.array, kernel)

	@staticmethod
	def binned_log_likelihood_ratio_rates_from_samples_wrapper(signal_lr_lnpdf, noise_lr_lnpdf, samples, nsamples):
		"""
		For internal use only.
		"""
		try:
			# want the forked processes to use different random
			# number sequences, so we re-seed Python and
			# numpy's random number generators here in the
			# wrapper in the hope that that takes care of it
			random.seed()
			numpy.random.seed()
			binned_log_likelihood_ratio_rates_from_samples(signal_lr_lnpdf, noise_lr_lnpdf, samples, nsamples)
			return signal_lr_lnpdf.array, noise_lr_lnpdf.array
		except:
			raise

	def __init__(self, rankingstat, signal_noise_pdfs = None, nsamples = 2**24, verbose = False):
		#
		# bailout used by .from_xml() class method to get an
		# uninitialized instance
		#

		if rankingstat is None:
			return

		#
		# initialize binnings
		#

		self.noise_lr_lnpdf = rate.BinnedLnPDF(rate.NDBins((rate.ATanBins(0., 110., 6000),)))
		self.signal_lr_lnpdf = rate.BinnedLnPDF(rate.NDBins((rate.ATanBins(0., 110., 6000),)))
		self.zero_lag_lnpdf = rate.BinnedLnPDF(rate.NDBins((rate.ATanBins(0., 110., 6000),)))

		#
		# bailout used by codes that want all-zeros histograms
		#

		if not nsamples:
			return

		#
		# run importance-weighted random sampling to populate binnings.
		#

		if signal_noise_pdfs is None:
			signal_noise_pdfs = rankingstat

		self.signal_lr_lnpdf.array, self.noise_lr_lnpdf.array = self.binned_log_likelihood_ratio_rates_from_samples_wrapper(
			self.signal_lr_lnpdf, 
			self.noise_lr_lnpdf,
			rankingstat.ln_lr_samples(rankingstat.denominator.random_params(), signal_noise_pdfs), 
			nsamples = nsamples)

		if verbose:
			print >> sys.stderr, "done computing ranking statistic PDFs"

		#
		# apply density estimation kernels to counts
		#

		self.density_estimate(self.noise_lr_lnpdf, "noise model")
		self.density_estimate(self.signal_lr_lnpdf, "signal model")

		#
		# set the total sample count in the noise and signal
		# ranking statistic histogram equal to the total expected
		# count of the respective events from the experiment. This
		# information is required so that when adding ranking
		# statistic PDFs in our .__iadd__() method they are
		# combined with the correct relative weights, so that
		# .__iadd__() has the effect of marginalizing the
		# distribution over the experients being combined.
		#

		self.noise_lr_lnpdf.array /= self.noise_lr_lnpdf.array.sum()
		self.noise_lr_lnpdf.normalize()
		self.signal_lr_lnpdf.array /= self.signal_lr_lnpdf_array.sum()
		self.signal_lr_lnpdf.normalize()


	def copy(self):
		new = self.__class__(None)
		new.noise_lr_lnpdf = self.noise_lr_lnpdf.copy()
		new.signal_lr_lnpdf = self.signal_lr_lnpdf.copy()
		new.zero_lag_lr_lnpdf = self.zero_lag_lr_lnpdf.copy()
		return new

	def collect_zero_lag_rates(self, connection, coinc_def_id):
		for ln_likelihood_ratio in connection.cursor().execute("""
SELECT
	likelihood,
FROM
	coinc_event
WHERE
	coinc_def_id == ?
	AND NOT EXISTS (
		SELECT
			*
		FROM
			time_slide
		WHERE
			time_slide.time_slide_id == coinc_event.time_slide_id
			AND time_slide.offset != 0
	)
""", (coinc_def_id,)):
			assert ln_likelihood_ratio is not None, "null likelihood ratio encountered.  probably coincs have not been ranked"
			self.zero_lag_lr_lnpdf.count[ln_likelihood_ratio,] += 1.
		self.zero_lag_lr_lnpdf.normalize()

	
	def density_estimate_zero_lag_rates(self):
		# apply density estimation preserving total count, then
		# normalize PDF
		count_before = self.zero_lag_lr_lnpdf.array.sum()
		# FIXME: should .normalize() be able to handle NaN?
		if count_before:
			self.density_estimate(self.zero_lag_lr_lnpdf, "zero lag")
			self.zero_lag_lr_lnpdf.array *= count_before / self.zero_lag_lr_lnpdf.array.sum()
		self.zero_lag_lr_lnpdf.normalize()
	

	def __iadd__(self, other):
		self.noise_lr_lnpdf += other.noise_lr_lnpdf
		self.noise_lr_lnpdf.normalize()
		self.signal_lr_lnpdf += other.signal_lr_lnpdf
		self.signal_lr_lnpdf.normalize()
		self.zero_lag_lr_lnpdf += other.zero_lag_lr_lnpdf
		self.zero_lag_lr_lnpdf.normalize()
		return self

	@classmethod
	def get_xml_root(cls, xml, name):
		"""
		Sub-classes can use this in their overrides of the
		.from_xml() method to find the root element of the XML
		serialization.
		"""
		name = u"%s:%s" % (name, cls.ligo_lw_name_suffix)
		xml = [elem for elem in xml.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") and elem.Name == name]
		if len(xml) != 1:
			raise ValueError("XML tree must contain exactly one %s element named %s" % (ligolw.LIGO_LW.tagName, name))
		return xml[0]


	@classmethod
	def from_xml(cls, xml, name):
		# find the root of the XML tree containing the
		# serialization of this object
		xml = xls.get_xml_root(xml, name)
		# create a mostly uninitialized instance
		self = cls(None)
		# popuate from XML
		self.noise_lr_lnpdf = rate.BinnedLnPDF.from_xml(xml, u"noise_lr_lnpdf")
		self.signal_lr_lnpdf = rate.BinnedLnPDF.from_xml(xml, u"signal_lr_lnpdf")
		self.zero_lag_lr_lnpdf = rate.BinnedLnPDF.from_xml(xml, u"zero_lag_lr_lnpdf")
		return self
	
	def to_xml(self, name):
		# do not allow ourselves to be written to disk without our
		# PDF's internal normalization metadata being up to date
		self.noise_lr_lnpdf.normalize()
		self.signal_lr_lnpdf.normalize()
		self.zero_lag_lr_lnpdf.normalize()

		xml = ligolw.LIGO_LW({u"Name": u"%s:%s" % (name, self.ligo_lw_name_suffix)})
		xml.appendChild(self.noise_lr_lnpdf.to_xml(u"noise_lr_lnpdf"))
		xml.appendChild(self.signal_lr_lnpdf.to_xml(u"signal_lr_lnpdf"))
		xml.appendChild(self.zero_lag_lr_lnpdf.to_xml(u"zero_lag_lr_lnpdf"))
		return xml


#
# =============================================================================
#
#                       False alarm rates/probabilities 
#
# =============================================================================
#


#
# Class to compute false-alarm probabilities and false-alarm rates from
# ranking statistic PDFs
#

class FAPFAR(object):
	def __init__(self, rankingstatpdf, livetime):
		# input checks
		if not rankingstatpdf.zero_lag_lr_lnpdf.array.any():
			raise ValueError("RankingStatPDF's zero-lag counts are all zero")

		self.livetime = livetime 
		
		# set the rate normalization LR threshold to the mode of
		# the zero-lag LR distribution.
		zl = rankingstat.zero_lag_lr_lnpdf.copy()
		rate_normalization_lr_threshold, = zl.argmax()

		# record trials factor, with safety checks
		counts = rankingstatpdf.zero_lag_lr_lnpdf.count()
		assert not numpy.isnan(counts.array).any(), "zero lag log likelihood ratio counts contain NaNs"
		assert (counts.array >= 0.).all(), "zero lag log likelihood ratio rates contain negative values"
		self.count_above_threshold = counts[rate_normalization_lr_threshold:,].sum()

		# get noise model ranking stat values and event counts from
		# bins
		threshold_index = rankingstatpdf.noise_lr_lnpdf.bins[0][rate_normalization_lr_threshold]
		ranks = rankingstatpdf.noise_lr_lnpdf.bins[0].lower()[threshold_index:]
		counts = rankingstatpdf.noise_lr_lnpdf.array[threshold_index:]
		assert not numpy.isnan(counts).any(), "background log likelihood ratio rates contain NaNs"
		assert (counts >= 0.).all(), "background log likelihood ratio rates contain negative values"

		# complementary cumulative distribution function
		ccdf = counts[::-1].cumsum()[::-1]
		ccdf /= ccdf[0]

		# ccdf is P(ranking statistic > threshold | a candidate).
		# we need P(rankins statistic > threshold), i.e. need to
		# correct for the possibility that no candidate is present.
		# specifically, the ccdf needs to =1-1/e at the candidate
		# identification threshold, and cdf=1/e at the candidate
		# threshold, in order for FAR(threshold) * livetime to
		# equal the actual observed number of candidates.
		ccdf = gstlalstats.poisson_p_not_0(ccdf)

		# safety checks
		assert not numpy.isnan(ranks).any(), "log likelihood ratio co-ordinates contain NaNs"
		assert not numpy.isinf(ranks).any(), "log likelihood ratio co-ordinates are not all finite"
		assert not numpy.isnan(ccdf).any(), "log likelihood ratio CCDF contains NaNs"
		assert ((0. <= ccdf) & (ccdf <= 1.)).all(), "log likelihood ratio CCDF failed to be normalized"

		# build interpolator.
		self.ccdf_interpolator = interpolate.interp1d(ranks, ccdf)

		# record min and max ranks so we know which end of the ccdf
		# to use when we're out of bounds
		self.minrank = ranks[0]
		self.maxrank = ranks[-1]

	@gstlalstats.assert_probability
	def ccdf_from_rank(self, rank):
		return self.ccdf_interpolator(numpy.clip(rank, self.minrank, self.maxrank))

	def fap_from_rank(self, rank):
		# implements equation (8) from Phys. Rev. D 88, 024025.
		# arXiv: 1209.0718
		return gstlalstats.fap_after_trials(self.ccdf_from_rank(rank), self.count_above_threshold)
	
	def far_from_rank(self, rank):
		# implements equation (B4) of Phys. Rev. D 88, 024025.
		# arXiv: 1209.0718. the return value is divided by T to
		# convert events/experiment to events/second. "tdp" =
		# true-dismissal probability = 1 - single-event FAP, the
		# integral in equation (B4)
		log_tdp = numpy.log1p(-self.ccdf_from_rank(rank))
		return self.count_above_threshold * -log_tdp / self.livetime
	
	# NOTE do we need rank_from_fap and rank_from_far?

	def assign_fapfars(self, connection):
		# assign false-alarm probabilities and false-alarm rates
		# FIXME we should fix whether we use coinc_burst or multi_burst,
		# whichever is less work.
		def as_float(f):
			def g(x):
				return float(f(x))
			return g
		connection.create_function("fap_from_rankingstat", 1, as_float(self.fap_from_rank))
		connection.create_function("far_from_rankingstat", 1, as_float(self.far_from_rank))
		connection.cursor().execute("""
UPDATE
	coinc_burst
SET
	false_alarm_probability = (
		SELECT
			fap_from_rankingstat(coinc_event.likelihood)
		FROM
			coinc_event
		WHERE
			coinc_event.coinc_event_id == coinc_burst.coinc_event_id
	),
	false_alarm_rate = (
		SELECT
			fap_from_rankingstat(coinc_event.likelihood)
		FROM
			coinc_event
		WHERE
			coinc_event.coinc_event_id == coinc_burst.coinc_event_id
	)
""")


#
# =============================================================================
#
#                                     I/O
#
# =============================================================================
#


def gen_likelihood_control_doc(xmldoc, rankingstat, rankingstatpdf):
	name = u"gstlal_string_likelihood"
	node = xmldoc.childNodes[-1]
	assert node.tagName == ligolw.LIGO_LW.tagName

	if rankingstat is not None:
		node.appendChild(rankingstat.to_xml(name))

	if rankingstatpdf is not None:
		node.appendChild(rankingstatpdf.to_xml(name))

	return xmldoc


def parse_likelihood_control_doc(xmldoc):
	name = u"gstlal_string_likelihood"
	try:
		rankingstat = RankingStat.from_xml(xmldoc, name)
	except ValueError:
		rankingstat = None
	try:
		rankingstatpdf = RankingStatPDF.from_xml(xmldoc, name)
	except ValueError:
		rankingstatpdf = None
	if rankingstat is None and rankingstatpdf is None:
		raise ValueError("document does not contain likelihood ratio data")
	return rankingstat, rankingstatpdf


def marginalize_pdf_urls(urls, which, ignore_missing_files = False, verbose = False):
	"""
	Implements marginalization of PDFs in ranking statistic data files.
	The marginalization is over the degree of freedom represented by
	the file collection.  One or both of the candidate parameter PDFs
	and ranking statistic PDFs can be processed, with errors thrown if
	one or more files is missing the required component.
	"""
	name = u"gstlal_string_likelihood"
	data = None
	for n, url in enumerate(urls, start = 1):
		#
		# load input document
		#

		if verbose:
			print >>sys.stderr, "%d/%d:" % (n, len(urls)),
		try:
			xmldoc = ligolw_utils.load_url(url, verbose = verbose, contenthandler = RankingStat.LIGOLWContentHandler)
		except IOError:
			# IOError is raised when an on-disk file is
			# missing.  urllib2.URLError is raised when a URL
			# cannot be loaded, but this is subclassed from
			# IOError so IOError will catch those, too.
			if not ignore_missing_files:
				raise
			if verbose:
				print >>sys.stderr, "Could not load \"%s\" ... skipping as requested" % url
			continue

		#
		# extract PDF objects compute weighted sum of ranking data
		# PDFs
		#

		if which == "RankingStat":
			if data is None:
				data = RankingStat.from_xml(xmldoc, name)
			else:
				data += RankingStat.from_xml(xmldoc, name)
		elif which == "RankingStatPDF":
			if data is None:
				data = RankingStatPDF.from_xml(xmldoc, name)
			else:
				data += RankingStatPDF.from_xml(xmldoc, name)
		else:
			raise ValueError("invalid which (%s)" % which)
		xmldoc.unlink()

	return data

####################
# Modules for calculating and storing likelihood ratio density
# for the cosmic string search.
#


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


from __future__ import print_function
try:
	from fpconst import NegInf
except ImportError:
	# not all machines have fpconst installed
	NegInf = float("-inf")

import itertools
import math
import numpy
import random
import sys
import warnings

from ligo.lw import ligolw
from ligo.lw import lsctables
from ligo.lw import param as ligolw_param
from ligo.lw import utils as ligolw_utils 
from ligo import segments

from lal import rate
from lalburst import snglcoinc

from gstlal.stats import trigger_rate
from . import string_extrinsics

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
	# SNR^2, chi^2/snr^2 binning definition
	snr2_chi2_binning = rate.NDBins((rate.ATanLogarithmicBins(10, 1e3, 801), rate.ATanLogarithmicBins(1e-3, 1.0, 801)))
	snr2_duration_binning = rate.NDBins((rate.ATanLogarithmicBins(10, 1e3, 801), rate.ATanLogarithmicBins(1e-4, 1e1, 801)))

	def __init__(self, instruments, delta_t, snr_threshold, num_templates, min_instruments = 2):
		# check input
		if min_instruments < 2:
			raise ValueError("min_instruments=%d must be >=2" % min_instruments)
		if min_instruments > len(instruments):
			raise ValueError("not enough instruments (%s) to satisfy min_instruments=%d" % (", ".join(sorted(instruments)), min_instruments))
		assert delta_t > 0 and snr_threshold > 0

		self.instruments = frozenset(instruments)
		self.delta_t = delta_t
		self.snr_threshold = snr_threshold
		self.num_templates = num_templates
		self.min_instruments = min_instruments
		self.densities = {}
		for instrument in self.instruments:
			self.densities["%s_snr2_chi2" % instrument] = rate.BinnedLnPDF(self.snr2_chi2_binning)
			self.densities["%s_snr2_duration" % instrument] = rate.BinnedLnPDF(self.snr2_duration_binning)

	def __call__(self):
		try:
			interps = self.interps
		except AttributeError:
			self.mkinterps()
			interps = self.interps

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

	def increment(self, event):
		self.densities["%s_snr2_chi2" % event.ifo].count[event.snr**2., event.chisq / event.chisq_dof / event.snr**2.] += 1.0
		self.densities["%s_snr2_duration" % event.ifo].count[event.snr**2., event.duration] += 1.0

	def copy(self):
		new = type(self)(self.instruments, min_instruments = self.min_instruments)
		for key, lnpdf in self.densities.items():
			new.densities[key] = lnpdf.copy()
		return new

	def mkinterps(self):
		self.interps = dict((key, lnpdf.mkinterp()) for key, lnpdf in self.densities.items())

	def finish(self):
		snrsq_kernel_width_at_64 = 16.,
		chisq_kernel_width = 0.02,
		sigma = 10.
		for key, lnpdf in self.densities.items():
			if key.endswith("_snr2_chi2"):
				numsamples = max(lnpdf.array.sum() / 10. + 1., 1e3)
				snrsq_bins, chisq_bins = lnpdf.bins
				snrsq_per_bin_at_64 = (snrsq_bins.upper() - snrsq_bins.lower())[snrsq_bins[64.]]
				chisq_per_bin_at_0_02 = (chisq_bins.upper() - chisq_bins.lower())[chisq_bins[0.02]]

				# apply Silverman's rule so that the width scales
				# with numsamples**(-1./6.) for a 2D PDF
				snrsq_kernel_bins = snrsq_kernel_width_at_64 / snrsq_per_bin_at_64 / numsamples**(1./6.)
				chisq_kernel_bins = chisq_kernel_width / chisq_per_bin_at_0_02 / numsamples**(1./6.)
				
				# check the size of the kernel. We don't ever let
				# it get smaller than the 2.5 times the bin size
				if snrsq_kernel_bins < 2.5:
					snrsq_kernel_bins = 2.5
					warnings.warn("Replacing snrsq kernel bins with 2.5")
				if chisq_kernel_bins < 2.5:
					chisq_kernel_bins = 2.5
					warnings.warn("Replacing chisq kernel bins with 2.5")

				# convolve bin count with density estimation kernel
				rate.filter_array(lnpdf.array, rate.gaussian_window(snrsq_kernel_bins, chisq_kernel_bins, sigma = sigma))

				# zero everything below the SNR cutoff. need to do the slicing
				# ourselves to avoid zero-ing the at-threshold bin
				lnpdf.array[:lnpdf.bins[0][self.snr_threshold],:] = 0.
			elif key.endswith("_snr2_duration"):
				# FIXME the duration filter kernel is left as a guess
				rate.filter_array(lnpdf.array, rate.gaussian_window(snrsq_kernel_bins, 11, sigma = sigma))
				# zero everything below the SNR cutoff. need to do the slicing
				# ourselves to avoid zero-ing the at-threshold bin
				lnpdf.array[:lnpdf.bins[0][self.snr_threshold],:] = 0.
			else:
				# shouldn't get here
				raise Exception
			lnpdf.normalize()
		self.mkinterps()

		#
		# never allow PDFs that have had the density estimation
		# transform applied to be written to disk:  on-disk files
		# must only ever provide raw counts.  also don't allow
		# density estimation to be applied twice
		#

		def to_xml(*args, **kwargs):
			raise NotImplementedError("writing .finish()'ed LnLRDensity object to disk is forbidden")
		self.to_xml = to_xml
		def finish(*args, **kwargs):
			raise NotImplementedError(".finish()ing a .finish()ed LnLRDensity object is forbidden")
		self.finish = finish

	def to_xml(self, name):
		xml = super(LnLRDensity, self).to_xml(name)
		xml.appendChild(ligolw_param.Param.from_pyvalue(u"instruments", lsctables.instrumentsproperty.set(self.instruments)))
		xml.appendChild(ligolw_param.Param.from_pyvalue(u"delta_t", self.delta_t))
		xml.appendChild(ligolw_param.Param.from_pyvalue(u"snr_threshold", self.snr_threshold))
		xml.appendChild(ligolw_param.Param.from_pyvalue(u"num_templates", self.num_templates))
		xml.appendChild(ligolw_param.Param.from_pyvalue(u"min_instruments", self.min_instruments))
		for key, lnpdf in self.densities.items():
			xml.appendChild(lnpdf.to_xml(key))
		return xml

	@classmethod
	def from_xml(cls, xml, name):
		xml = cls.get_xml_root(xml, name)
		self = cls(
			instruments = lsctables.instrumentsproperty.get(ligolw_param.get_pyvalue(xml, u"instruments")),
			delta_t = ligolw_param.get_pyvalue(xml, u"delta_t"),
			snr_threshold = ligolw_param.get_pyvalue(xml, u"snr_threshold"),
			num_templates = ligolw_param.get_pyvalue(xml, u"num_templates"),
			min_instruments = ligolw_param.get_pyvalue(xml, u"min_instruments")
			)
		for key in self.densities:
			self.densities[key] = rate.BinnedLnPDF.from_xml(xml, key)
		return self


#
# Likelihood ratio density (numerator)
#


class LnSignalDensity(LnLRDensity):
	snr_binning = rate.ATanLogarithmicBins(3.6, 1e3, 150)
	def __init__(self, *args, **kwargs):
		super(LnSignalDensity, self).__init__(*args, **kwargs)
		# initialize SNRPDF
		self.SNRPDF = {}
		for n in range(self.min_instruments, len(self.instruments) + 1):
			for ifo_combos in itertools.combinations(sorted(self.instruments), n):
				self.SNRPDF[ifo_combos] = rate.BinnedArray(rate.NDBins([self.snr_binning] * len(ifo_combos))) 

	def __call__(self, segments, snr2s, chi2s_over_snr2s, durations):
		super(LnSignalDensity, self).__call__()
		# pre-compute only when we first compute likelihood
		try:
			P_instruments_given_signal = self.P_instruments_given_signal
		except:
			self.mksignalpdfs()
		instruments = sorted(snr2s.keys())
		# evaluate P(instruments | horizon distances)
		lnP = math.log(self.P_instruments_given_signal[frozenset(instruments)])
		# evaluate SNR probability
		snrs = [math.sqrt(snr2s[ifo]) for ifo in instruments]
		# FIXME this log(0) issue should be dealt with in the PDF generaator
		try:
			lnP += math.log(self.InterpSNRPDF[tuple(instruments)](*snrs))
		except ValueError:
			return NegInf
		# evaluate P(snr^2, \chi^2 | snr^2) = P(snr^2, \chi^2) / P(snr)
		# and P(snr^2, duration | snr^2) = P(snr^2, duration) / P(snr)
		interps = self.interps
		lnPSNR = sum(self.marginalized_interpolated["%s_snr2_chi2" % instrument](snr2) for instrument, snr2 in snr2s.items())
		lnP += sum(interps["%s_snr2_duration" % instrument](snr2s[instrument], duration) for instrument, duration in durations.items())
		return lnP - 2.0 * lnPSNR + sum(interps["%s_snr2_chi2" % instrument](snr2s[instrument], chi2_over_snr2) for instrument, chi2_over_snr2 in chi2s_over_snr2s.items())

	def __iadd__(self, other):
		super(LnSignalDensity, self).__iadd__(other)
		if set(self.SNRPDF) != set(other.SNRPDF):
			raise TypeError("cannot add %s and %s" % (type(self), type(other)))
		for key, pdf in self.SNRPDF.items():
			pdf += other.SNRPDF[key]
		try:
			del self.interps
		except AttributeError:
			pass
		return self

	def copy(self):
		new = super(LnSignalDensity, self).copy()
		for key, pdf in self.SNRPDF.items():
			new.SNRPDF[key] = pdf.copy()
		return new

	def mksignalpdfs(self):
		# precompute P_instruments for all detector sets.
		#
		print("computing signal PDFs...", file=sys.stderr)
		# FIXME use harded-coded typical horizon distances for now.
		# we can and should use time-evolving horizon distance when
		# we combine results from multiple observing runs, or we want
		# to incorporate changes in sensitivity during a single run.
		# In any case, only the ratio between detectors is important,
		# so we scale it to "familiar" numbers, but the ratios are
		# calibrated by sensitivity obtained as a byproduct in the
		# template normalization in the trigger generator
		# (see horizon_distance in gstlal_cs_triggergen for detail)
		TYPICAL_HORIZON_DISTANCES = {'H1':195., 'L1':250., 'V1':94.} 
		# there are cases where one of the three is off and still form
		# coincs, but assume all three are on. For 3 ifos we're >~80%
		# of the time right, since at least 2 needs to be on to form coincs
		# and it's easier to form 2-det coincs when all 3 are on than
		# when only 2 are on.
		# in any case, this should be simultaneously dealt with when
		# we use time-evolving horizon distances
		self.P_instruments_given_signal = string_extrinsics.P_instruments_given_signal(horizon_distances = TYPICAL_HORIZON_DISTANCES)
		# interpolate pre-loaded SNR PDF
		self.InterpSNRPDF = dict.fromkeys(self.SNRPDF.keys())
		for n in range(self.min_instruments, len(self.instruments) + 1):
			for ifo_combos in itertools.combinations(sorted(self.instruments), n):
				self.InterpSNRPDF[ifo_combos] = rate.InterpBinnedArray(self.SNRPDF[ifo_combos])

		# we also need SNR PDF for each ifo to calculate P(snr, chi^2|snr) and P(snr, duration|snr).
		# marginalize over chi^2 to get SNR PDF
		self.marginalized = dict((key, lnpdf.marginalize(1)) for key, lnpdf in self.densities.items())
		# then interpolate
		self.marginalized_interpolated = dict((key, lnpdf.mkinterp()) for key, lnpdf in self.marginalized.items())
		#done
		print("done computing signal PDFs.", file=sys.stderr)

	def increment(self, events, weight):
		for event in events:
			self.densities["%s_snr2_chi2" % event.ifo].count[event.snr**2., event.chisq / event.chisq_dof / event.snr**2.] += weight
			self.densities["%s_snr2_duration" % event.ifo].count[event.snr**2., event.duration] += weight

	def to_xml(self, name):
		xml = super(LnSignalDensity, self).to_xml(name)
		for key, pdf in self.SNRPDF.items():
			xml.appendChild(pdf.to_xml('_'.join(key)+'_SNR'))
		return xml

	@classmethod
	def from_xml(cls, xml, name):
		xml = cls.get_xml_root(xml, name)
		self = super(LnSignalDensity, cls).from_xml(xml, name)
		for key in self.SNRPDF:
			self.SNRPDF[key] = rate.BinnedArray.from_xml(xml, '_'.join(key)+'_SNR')
		return self


#
# Likelihood ratio density (denominator)
#


class LnNoiseDensity(LnLRDensity):
	def __init__(self, *args, **kwargs):
		super(LnNoiseDensity, self).__init__(*args, **kwargs)
		# record of trigger counts vs time for all instruments in
		# the network
		self.triggerrates = trigger_rate.triggerrates((instrument, trigger_rate.ratebinlist()) for instrument in self.instruments)
		# initialize a CoincRates object
		self.coinc_rates = snglcoinc.CoincRates(
			instruments = self.instruments,
			delta_t = self.delta_t,
			min_instruments = self.min_instruments
		)

	def __call__(self, segments, snr2s, chi2s_over_snr2s, durations):
		super(LnNoiseDensity, self).__call__()
		interps = self.interps

		assert frozenset(segments) == self.instruments
		if len(snr2s) < self.min_instruments:
			return NegInf

		# FIXME:  the +/-3600 s window thing is a temporary hack to
		# work around the problem of vetoes creating short segments
		# that have no triggers in them but that can have
		# injections recovered in them.  the +/- 3600 s window is
		# just a guess as to what might be sufficient to work
		# around it.  you might might to make this bigger.
		triggers_per_second_per_template = {}
		for instrument, seg in segments.items():
			triggers_per_second_per_template[instrument] = (self.triggerrates[instrument] & trigger_rate.ratebinlist([trigger_rate.ratebin(seg[1] - 3600., seg[1] + 3600., count = 0)])).density / self.num_templates
		# sanity check rates
		assert all(triggers_per_second_per_template[instrument] for instrument in snr2s), "impossible candidate in %s at %s when rates were %s triggers/s/template" % (", ".join(sorted(snr2s)), ", ".join("%s s in %s" % (str(seg[1]), instrument) for instrument, seg in sorted(segments.items())), str(triggers_per_second_per_template))

		# P(t | noise) = (candidates per unit time @ t) / total
		# candidates.  by not normalizing by the total candidates
		# the return value can only ever be proportional to the
		# probability density, but we avoid the problem of the
		# ranking statistic definition changing on-the-fly while
		# running online, allowing candidates collected later to
		# have their ranking statistics compared meaningfully to
		# the values assigned to candidates collected earlier, when
		# the total number of candidates was smaller.
		lnP = math.log(sum(self.coinc_rates.strict_coinc_rates(**triggers_per_second_per_template).values()) * self.num_templates)
		# evaluate P(ifos | t, noise)
		lnP += self.coinc_rates.lnP_instruments(**triggers_per_second_per_template)[frozenset(snr2s)]

		# FIXME now multiplying P(snr, duration | noise),
		# should be P(snr, duration | snr, noise)
		lnP += sum(interps["%s_snr2_duration" % instrument](snr2s[instrument], duration) for instrument, duration in durations.items()) 

		return lnP + sum(interps["%s_snr2_chi2" % instrument](snr2s[instrument], chi2_over_snr2) for instrument, chi2_over_snr2 in chi2s_over_snr2s.items())

	def __iadd__(self, other):
		super(LnNoiseDensity, self).__iadd__(other)
		self.triggerrates += other.triggerrates
		return self

	def copy(self):
		new = super(LnNoiseDensity, self).copy()
		new.triggerrates = self.triggerrates.copy()
		# NOTE:  lnzerolagdensity in the copy is reset to None by
		# this operation.  it is left as an exercise to the calling
		# code to re-connect it to the appropriate object if
		# desired.
		return new

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
		**kwargs

		See also:

		random_sim_params()

		The sequence is suitable for input to the .ln_lr_samples()
		log likelihood ratio generator.
		"""
		# random sampler slope
		# in inspiral search CDF(\rho) \propto \rho^(0.8/#inst^3)
		# thus the PDF is P(\rho) \propto \rho^(-1+0.8/#inst^3)
		# correspond to P(\rho^2) \propto (\rho^2)^(-0.5+0.4/#inst^3)
		# so CDF(\rho^2) should be \propto (\rho^2)^(0.5+0.4/#inst^3)
		snr2_slope = 0.5 + 0.4/len(self.instruments)**3 
		# some limits to avoid overflow errors 
		snr2_max = 1e20
		chi2_over_snr2_min = 1e-20
		chi2_over_snr2_max = 1e20
		durations_min = 1./8196.
		durations_max = 1e4

		snr2chi2gens = dict((instrument, iter(self.densities["%s_snr2_chi2" % instrument].bins.randcoord(ns = (snr2_slope, 1.), domain = (slice(self.snr_threshold**2, snr2_max), slice(chi2_over_snr2_min, chi2_over_snr2_max)))).next) for instrument in self.instruments)
		durationgens = dict((instrument, iter(self.densities["%s_snr2_duration" % instrument].bins[1].randcoord(n = 1., domain = slice(durations_min, durations_max))).next) for instrument in self.instruments) 

		t_and_rate_gen = iter(self.triggerrates.random_uniform()).next

		random_randint = random.randint
		random_sample = random.sample
		segment = segments.segment

		def nCk(n, k):
			return math.factorial(n) // math.factorial(k) // math.factorial(n - k)
		while 1:
			# choose a t (not needed for params, but used to
			# choose detector combo with the correct
			# distribution).
			t, rates, lnP_t = t_and_rate_gen()
			# choose a set of instruments from among those that
			# were generating triggers at t.
			instruments = tuple(instrument for instrument, rate in rates.items() if rate > 0)
			if len(instruments) < self.min_instruments:
				# FIXME doing this biases lnP_t to lower values,
				# but the error is merely an overall normalization
				# error that won't hurt. t_and_rate_gen() can be
				# fixed to exclude from the sampling times that not 
				# enough detectors were generating triggers.
				continue
			k = random_randint(self.min_instruments, len(instruments))
			lnP_instruments = -math.log((len(instruments) - self.min_instruments + 1) * nCk(len(instruments), k))
			instruments = frozenset(random_sample(instruments, k))
			# ((snr, chisq2/snr2), ln P, (snr, chisq2/snr2), ln P, ...)
			seq = sum((snr2chi2gens[instrument]() for instrument in instruments), ())
			# (dur, lnP, dur, lnP, ...)
			seqdur = sum((durationgens[instrument]() for instrument in instruments), ())
			# set kwargs
			kwargs = dict(
				# segments is only used to estimate the local trigger rate at
				# that time so the duration of event doesn't really matter
				segments = dict.fromkeys(self.instruments, segment(t, t+1.0)),
				snr2s = dict((instrument, value[0]) for instrument, value in zip(instruments, seq[0::2])),
				chi2s_over_snr2s = dict((instrument, value[1]) for instrument, value in zip(instruments, seq[0::2])),
				durations = dict(zip(instruments, seqdur[0::2]))
			)
			yield (), kwargs, sum(seq[1::2], lnP_t + lnP_instruments) + sum(seqdur[1::2])

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

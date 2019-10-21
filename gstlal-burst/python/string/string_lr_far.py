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


import itertools
import math
import numpy
import random
import sys

from lal import rate
from . import snglcoinc

from ligo.lw import ligolw
from ligo.lw import lsctables
from ligo.lw import param as ligolw_param
from ligo.lw import utils as ligolw_utils 

# FIXME don't import gstlal modules in lalsuite
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
	# SNR, chi^2 binning definition
	snr2_chi2_binning = rate.NDBins((rate.ATanLogarithmicBins(10, 1e7, 801), rate.ATanLogarithmicBins(.1, 1e4, 801)))

	def __init__(self, instruments, delta_t, snr_threshold, min_instruments = 2):
		# check input
		if min_instruments < 2:
			raise ValueError("min_instruments=%d must be >=2" % min_instruments)
		if min_instruments > len(instruments):
			raise ValueError("not enough instruments (%s) to satisfy min_instruments=%d" % (", ".join(sorted(instruments)), min_instruments))
		assert delta_t > 0 and snr_threshold > 0

		self.instruments = frozenset(instruments)
		self.delta_t = delta_t
		self.snr_threshold = snr_threshold
		self.min_instruments = min_instruments
		self.densities = {}
		for instrument in self.instruments:
			self.densities["%s_snr2_chi2" % instrument] = rate.BinnedLnPDF(self.snr2_chi2_binning)

	def __call__(self):
		try:
			interps = self.interps
		except AttributeError:
			self.mkinterps()
			interps = self.interps
		#return sum(interps[param](*value) for param, value in params.items())

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
		#self.densities["%s_snr2_chi2" % event.ifo].count[event.snr, event.chisq / event.chisq_dof / event.snr**2.] += 1.0
		self.densities["%s_snr2_chi2" % event.ifo].count[event.snr**2, event.chisq / event.chisq_dof] += 1.0

	def copy(self):
		new = type(self)(self.instruments, min_instruments = self.min_instruments)
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
		xml.appendChild(ligolw_param.Param.from_pyvalue(u"instruments", lsctables.instrumentsproperty.set(self.instruments)))
		xml.appendChild(ligolw_param.Param.from_pyvalue(u"delta_t", self.delta_t))
		xml.appendChild(ligolw_param.Param.from_pyvalue(u"snr_threshold", self.snr_threshold))
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
			min_instruments = ligolw_param.get_pyvalue(xml, u"min_instruments")
			)
		for key in self.densities:
			self.densities[key] = rate.BinnedLnPDF.from_xml(xml, key)
		return self


#
# Likelihood ratio density (numerator)
#


class LnSignalDensity(LnLRDensity):
	def __init__(self, *args, **kwargs):
		super(LnSignalDensity, self).__init__(*args, **kwargs)

	def __call__(self, snr2s, chi2s):
		super(LnSignalDensity, self).__call__()
		interps = self.interps
		return sum(interps["%s_snr2_chi2" % instrument](snr2s[instrument], chi2) for instrument, chi2 in chi2s.items())
		#return sum(interps["%s_snr2_chi2" % instrument](snrs[instrument], chi2_over_snr2) for instrument, chi2_over_snr2 in chi2s_over_snr2s.items())

	def add_signal_model(self, prefactors_range = (0.001, 0.30), inv_snr_pow = 4.):
		# normalize to 10 *mi*llion signals. This count makes the
		# density estimation code choose a suitable kernel size
		for instrument in self.instruments:
			string_extrinsics.NumeratorSNRCHIPDF.add_signal_model(self.densities["%s_snr2_chi2" % instrument], 10000000., prefactors_range, inv_snr_pow = inv_snr_pow, snr_min = self.snr_threshold)
			self.densities["%s_snr2_chi2" % instrument].normalize()

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

	def __call__(self, snr2s, chi2s):
		# FIXME evaluate P(t|noise), P(ifos|t,noise) using the 
		# triggerrate record (cf inspiral_lr)
		super(LnNoiseDensity, self).__call__()
		interps = self.interps
		return sum(interps["%s_snr2_chi2" % instrument](snr2s[instrument], chi2) for instrument, chi2 in chi2s.items())

		#return sum(interps["%s_snr2_chi2" % instrument](snrs[instrument], chi2_over_snr2) for instrument, chi2_over_snr2 in chi2s_over_snr2s.items())

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
		**kwargs.

		See also:

		random_sim_params()

		The sequence is suitable for input to the .ln_lr_samples()
		log likelihood ratio generator.
		"""
		snr_slope = 0.8 / len(self.instruments)**3

		snr2chi2gens = dict((instrument, iter(self.densities["%s_snr2_chi2" % instrument].bins.randcoord(ns = (snr_slope, 1.), domain = (slice(self.snr_threshold, None), slice(None, None)))).next) for instrument in self.instruments)
		t_and_rate_gen = iter(self.triggerrates.random_uniform()).next
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
			k = random.randint(self.min_instruments, len(instruments))
			lnP_instruments = -math.log((len(instruments) - self.min_instruments + 1) * nCk(len(instruments), k))
			instruments = frozenset(random.sample(self.instruments, k))
			# ((snr, chisq2/snr2), ln P, (snr, chisq2/snr2), ln P, ...)
			seq = sum((snr2chi2gens[instrument]() for instrument in instruments), ())
			# set kwargs
			kwargs = dict(
				snr2s = dict((instrument, value[0]) for instrument, value in zip(instruments, seq[0::2])),
				chi2s = dict((instrument, value[1]) for instrument, value in zip(instruments, seq[0::2]))
				#chi2s_over_snr2s = dict((instrument, value[1]) for instrument, value in zip(instruments, seq[0::2]))
			)
			yield (), kwargs, sum(seq[1::2], lnP_t + lnP_instruments)

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

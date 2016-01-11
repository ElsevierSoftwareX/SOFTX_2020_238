# Copyright (C) 2011--2014  Kipp Cannon, Chad Hanna, Drew Keppel
# Copyright (C) 2013  Jacob Peoples
# Copyright (C) 2015  Heather Fong
# Copyright (C) 2015--2016  Kipp Cannon
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

## 
# @file
#
# A file that contains the rate estimation code.
#
# Review Status: Reviewed with actions
#
# | Names                                          | Hash                                        | Date       |
# | -------------------------------------------    | ------------------------------------------- | ---------- |
# |          Sathya, Duncan Me, Jolien, Kipp, Chad | 2fb185eda0edb9d49d79b8185f7b35457cafa06b    | 2015-05-14 |
#
# #### Actions
# - Increase nbins to at least 10,000
# - Check max acceptance

## @package rate_estimation

#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


try:
	from fpconst import NaN, NegInf, PosInf
except ImportError:
	# fpconst is not part of the standard library and might not be
	# available
	NaN = float("nan")
	NegInf = float("-inf")
	PosInf = float("+inf")
import math
import numpy
from scipy import optimize
from scipy import special
import sys


from glue.text_progress_bar import ProgressBar
from pylal import rate


from gstlal import emcee
from gstlal._rate_estimation import LogPosterior


#
# =============================================================================
#
#                                 Helper Code
#
# =============================================================================
#


def run_mcmc(n_walkers, n_dim, n_samples_per_walker, lnprobfunc, pos0 = None, args = (), n_burn = 100, progressbar = None):
	"""
	A generator function that yields samples distributed according to a
	user-supplied probability density function that need not be
	normalized.  lnprobfunc computes and returns the natural logarithm
	of the probability density, up to a constant offset.  n_dim sets
	the number of dimensions of the parameter space over which the PDF
	is defined and args gives any additional arguments to be passed to
	lnprobfunc, whose signature must be

		ln(P) = lnprobfunc(X, *args)

	where X is a numpy array of length n_dim.

	The generator yields a total of n_samples_per_walker arrays each of
	which is n_walkers by n_dim in size.  Each row is a sample drawn
	from the n_dim-dimensional parameter space.

	pos0 is an n_walkers by n_dim array giving the initial positions of
	the walkers (this parameter is currently not optional).  n_burn
	iterations of the MCMC sampler will be executed and discarded to
	allow the system to stabilize before samples are yielded to the
	calling code.  A chain can be continued by passing the last return
	value as pos0 and setting n_burn = 0.

	NOTE:  the samples yielded by this generator are not drawn from the
	PDF independently of one another.  The correlation length is not
	known at this time.
	"""
	#
	# construct a sampler
	#

	sampler = emcee.EnsembleSampler(n_walkers, n_dim, lnprobfunc, args = args)

	#
	# set walkers at initial positions
	#

	# FIXME:  implement
	assert pos0 is not None, "auto-selection of initial positions not implemented"

	#
	# burn-in:  run for a while to get better initial positions.
	# run_mcmc() doesn't like being asked for 0 iterations, so need to
	# check for that
	#

	if n_burn:
		pos0, ignored, ignored = sampler.run_mcmc(pos0, n_burn, storechain = False)
	if progressbar is not None:
		progressbar.increment(delta = n_burn)
	if n_burn and sampler.acceptance_fraction.min() < 0.4:
		print >>sys.stderr, "\nwarning:  low burn-in acceptance fraction (min = %g)" % sampler.acceptance_fraction.min()

	#
	# reset and yield positions distributed according to the supplied
	# PDF
	#

	sampler.reset()
	for coordslist, ignored, ignored in sampler.sample(pos0, iterations = n_samples_per_walker, storechain = False):
		yield coordslist
		if progressbar is not None:
			progressbar.increment()
	if n_samples_per_walker and sampler.acceptance_fraction.min() < 0.5:
		print >>sys.stderr, "\nwarning:  low sampler acceptance fraction (min %g)" % sampler.acceptance_fraction.min()


def moment_from_pdf(binnedarray, moment, c = 0.):
	if len(binnedarray.bins) != 1:
		raise ValueError("BinnedArray PDF must have 1 dimension")
	x = binnedarray.bins.centres()[0]
	dx = binnedarray.bins.upper()[0] - binnedarray.bins.lower()[0]
	return ((x - c)**moment * binnedarray.array * dx).sum()


def mean_from_pdf(binnedarray):
	return moment_from_pdf(binnedarray, 1.)


def variance_from_pdf(binnedarray):
	return moment_from_pdf(binnedarray, 2., mean_from_pdf(binnedarray))


def confidence_interval_from_binnedarray(binned_array, confidence = 0.95):
	"""
	Constructs a confidence interval based on a BinnedArray object
	containing a normalized 1-D PDF.  Returns the tuple (mode, lower bound,
	upper bound).
	"""
	# check for funny input
	if numpy.isnan(binned_array.array).any():
		raise ValueError("NaNs encountered in rate PDF")
	if numpy.isinf(binned_array.array).any():
		raise ValueError("infinities encountered in rate PDF")
	if (binned_array.array < 0.).any():
		raise ValueError("negative values encountered in rate PDF")
	if not 0.0 <= confidence <= 1.0:
		raise ValueError("confidence must be in [0, 1]")

	mode_index = numpy.argmax(binned_array.array)

	centres, = binned_array.centres()
	upper = binned_array.bins[0].upper()
	lower = binned_array.bins[0].lower()
	bin_widths = upper - lower
	if (bin_widths <= 0.).any():
		raise ValueError("rate PDF bin sizes must be positive")
	pdf = binned_array.array
	P = pdf * bin_widths
	# fix NaNs in infinite-sized bins
	P[numpy.isinf(bin_widths)] = 0.
	assert (pdf >= 0.).all()
	assert (P >= 0.).all()
	if abs(P.sum() - 1.0) > 1e-13:
		raise ValueError("rate PDF is not normalized")

	li = ri = mode_index
	P_sum = P[mode_index]
	while P_sum < confidence:
		if li <= 0 and ri >= len(P) - 1:
			raise ValueError("failed to achieve requested confidence")

		if li > 0:
			pdf_li = pdf[li - 1]
			P_li = P[li - 1]
		else:
			pdf_li = 0.
			P_li = 0.
		if ri < len(P) - 1:
			pdf_ri = pdf[ri + 1]
			P_ri = P[ri + 1]
		else:
			pdf_ri = 0.
			P_ri = 0.

		if pdf_li > pdf_ri:
			li -= 1
			P_sum += P_li
		elif pdf_ri > pdf_li:
			ri += 1
			P_sum += P_ri
		else:
			P_sum += P_li + P_ri
			li = max(li - 1, 0)
			ri = min(ri + 1, len(P) - 1)

	return centres[mode_index], lower[li], upper[ri]


#
# =============================================================================
#
#                            Event Rate Posteriors
#
# =============================================================================
#


def maximum_likelihood_rates(ln_f_over_b):
	# the upper bound is chosen to include N + \sqrt{N}
	return optimize.fmin((lambda x: -RatesLnPDF(x, ln_f_over_b)), (1.0, len(ln_f_over_b) + len(ln_f_over_b)**.5), disp = True)


def binned_rates_from_samples(samples):
	"""
	Construct and return a BinnedArray containing a histogram of a
	sequence of samples.  If limits is None (default) then the limits
	of the binning will be determined automatically from the sequence,
	otherwise limits is expected to be a tuple or other two-element
	sequence providing the (low, high) limits, and in that case the
	sequence can be a generator.
	"""
	lo, hi = math.floor(samples.min()), math.ceil(samples.max())
	nbins = int(math.sqrt(len(samples)) / 40.)
	binnedarray = rate.BinnedArray(rate.NDBins((rate.LogarithmicPlusOverflowBins(lo if lo !=0. else samples.min(), hi, nbins),)))
	for sample in samples:
		binnedarray[sample,] += 1.
	rate.filter_array(binnedarray.array, rate.gaussian_window(5), use_fft = False)
	return binnedarray


def calculate_rate_posteriors(ranking_data, ln_likelihood_ratios, progressbar = None, chain_file = None, nsample = 400000):
	"""
	FIXME:  document this
	"""
	#
	# check for bad input
	#

	if any(math.isnan(ln_lr) for ln_lr in ln_likelihood_ratios):
		raise ValueError("NaN log likelihood ratio encountered")
	if nsample < 0:
		raise ValueError("nsample < 0: %d" % nsample)

	#
	# for each sample of the ranking statistic, evaluate the ratio of
	# the signal ranking statistic PDF to background ranking statistic
	# PDF.
	# FIXME:  use an InterpBinnedArray for this
	#

	if ranking_data is not None:
		f = ranking_data.signal_likelihood_pdfs[None]
		b = ranking_data.background_likelihood_pdfs[None]
		ln_f_over_b = numpy.log(numpy.array([f[ln_lr,] / b[ln_lr,] for ln_lr in ln_likelihood_ratios]))
		# safety check
		if numpy.isnan(ln_f_over_b).any():
			raise ValueError("NaN encountered in ranking statistic log PDF ratios")
		if numpy.isinf(ln_f_over_b).any():
			raise ValueError("infinity encountered in ranking statistic log PDF ratios")
	elif nsample > 0:
		raise ValueError("must supply ranking data to run MCMC sampler")
	else:
		# no-op path
		ln_f_over_b = numpy.array([])

	#
	# initialize MCMC chain.  try loading a chain from a chain file if
	# provided, otherwise seed the walkers for a burn-in period
	#

	ndim = 2
	nwalkers = 10 * 2 * ndim	# must be even and >= 2 * ndim
	nburn = 1000

	pos0 = numpy.zeros((nwalkers, ndim), dtype = "double")

	if progressbar is not None:
		progressbar.max = nsample + nburn
		progressbar.show()

	i = 0
	if chain_file is not None and "chain" in chain_file:
		chain = chain_file["chain"].values()
		length = sum(sample.shape[0] for sample in chain)
		samples = numpy.empty((max(nsample, length), nwalkers, ndim), dtype = "double")
		if progressbar is not None:
			progressbar.max = samples.shape[0]
		# load chain from HDF file
		for sample in chain:
			samples[i:i+sample.shape[0],:,:] = sample
			i += sample.shape[0]
			if progressbar is not None:
				progressbar.update(i)
		if i:
			# skip burn-in, restart chain from last position
			nburn = 0
			pos0 = samples[i - 1,:,:]
	elif nsample <= 0:
		raise ValueError("no chain file to load or invalid chain file, and no new samples requested")
	if not i:
		# no chain file provided, or file does not contain sample
		# chain or sample chain is empty.  still need burn-in.
		# seed signal rate walkers from exponential distribution,
		# background rate walkers from poisson distribution
		samples = numpy.empty((nsample, nwalkers, ndim), dtype = "double")
		pos0[:,0] = numpy.random.exponential(scale = 1., size = (nwalkers,))
		pos0[:,1] = numpy.random.poisson(lam = len(ln_likelihood_ratios), size = (nwalkers,))
		i = 0

	#
	# run MCMC sampler to generate (foreground rate, background rate)
	# samples.  to improve the measurement of the tails of the PDF
	# using the MCMC sampler, we draw from a power of the PDF
	# and then correct the histogram of the samples (see below).
	#

	log_posterior = LogPosterior(ln_f_over_b)

	exponent = 2.25

	for j, coordslist in enumerate(run_mcmc(nwalkers, ndim, max(0, nsample - i), (lambda x: log_posterior(x) / exponent), n_burn = nburn, pos0 = pos0, progressbar = progressbar), i):
		# coordslist is nwalkers x ndim
		samples[j,:,:] = coordslist
		# dump samples to the chain file every 2048 steps
		if j + 1 >= i + 2048 and chain_file is not None:
			chain_file["chain/%08d" % i] = samples[i:j+1,:,:]
			chain_file.flush()
			i = j + 1
	# dump any remaining samples to chain file
	if chain_file is not None and i < samples.shape[0]:
		chain_file["chain/%08d" % i] = samples[i:,:,:]
		chain_file.flush()

	#
	# safety check
	#

	if samples.min() < 0:
		raise ValueError("MCMC sampler yielded negative rate(s)")

	#
	# compute marginalized PDFs for the foreground and background
	# rates.  for each PDF, the samples from the MCMC are histogrammed
	# and convolved with a density estimation kernel.  the samples have
	# been drawn from the square root of the correct PDF, so the counts
	# in the bins must be corrected before converting to a normalized
	# PDF.  how to correct count:
	#
	# correct count = (correct PDF) * (bin size)
	#               = (measured PDF)^exponent * (bin size)
	#               = (measured count / (bin size))^exponent * (bin size)
	#               = (measured count)^exponent / (bin size)^(exponent - 1)
	#
	# this assumes small bin sizes.
	#

	Rf_pdf = binned_rates_from_samples(samples[:,:,0].flatten())
	Rf_pdf.array = Rf_pdf.array**exponent / Rf_pdf.bins.volumes()**(exponent - 1.)
	Rf_pdf.to_pdf()

	Rb_pdf = binned_rates_from_samples(samples[:,:,1].flatten())
	Rb_pdf.array = Rb_pdf.array**exponent / Rb_pdf.bins.volumes()**(exponent - 1.)
	Rb_pdf.to_pdf()

	#
	# done
	#

	return Rf_pdf, Rb_pdf


#
# =============================================================================
#
#                            Foreground Posteriors
#
# =============================================================================
#


#
# code to compute P(signal) for each candidate.
#
# starting with equation (18) in Farr et al., and using (22) to marginalize
# over Rf and Rb we obtain
#
# P(\{f_i\}) \propto
#	[ \prod_{\{i | f_i = 1\}} \hat{f}(x_i) ] [ \prod_{\{i | f_i = 0\}}
#	\hat{b}(x_i) ] (2 m - 1)!! (2 n - 1)!!
#
# where m is the number of terms for which f_i = 1 and n = N - m is the
# number of terms for which f_i = 0.  rearranging factors of \hat{b} this
# can be written as
#
#	\propto [ \prod_{\{i | f_i = 1\}} \hat{f}(x_i) / \hat{b}(x_i) ] (2
#	m - 1)!! (2 n - 1)!!
#


def ln_double_fac_table(N):
	"""
	Compute a look-up table giving

	ln (2m - 1)!! (2n - 1)!!

	for 0 <= m <= N, n + m = N.
	"""
	#
	# (2m - 1)!! = (2m)! / (2^m m!)
	#
	# so
	#
	# ln (2m - 1)!! (2n - 1)!! =
	#	ln (2m)! + ln (2n)! - ln m! - ln n! - N ln 2
	#
	# and ln x! = ln \Gamma(x + 1)
	#
	# for which we can use the gammaln() function from scipy.  the
	# final N ln 2 term can be dropped because we only need the answer
	# up to an arbitrary constant but it's cheap to calculate so we
	# don't bother with that simplification.
	#
	# the array is invariant under reversal so a factor of 2 savings
	# can be realized by computing only the first half, and using it to
	# populate the second half without additional arithmetic cost, but
	# at the time of writing the array takes less than a second to
	# compute up to N = 10^6 so we don't bother getting fancy.
	#

	lnfac = lambda x: special.gammaln(x + 1.)

	m = numpy.arange(N + 1)
	n = N - m

	return lnfac(2. * m) + lnfac(2. * n) - lnfac(m) - lnfac(n) - N * math.log(2.)


def calculate_psignal_posteriors(ranking_data, ln_likelihood_ratios, progressbar = None, chain_file = None, nsample = 400000):
	"""
	FIXME:  document this
	"""
	#
	# check for bad input
	#

	if any(math.isnan(ln_lr) for ln_lr in ln_likelihood_ratios):
		raise ValueError("NaN log likelihood ratio encountered")
	if nsample < 0:
		raise ValueError("nsample < 0: %d" % nsample)

	#
	# for each sample of the ranking statistic, evaluate the ratio of
	# the signal ranking statistic PDF to background ranking statistic
	# PDF.
	# FIXME:  use an InterpBinnedArray for this
	#

	if ranking_data is not None:
		f = ranking_data.signal_likelihood_pdfs[None]
		b = ranking_data.background_likelihood_pdfs[None]
		ln_f_over_b = numpy.log(numpy.array([f[ln_lr,] / b[ln_lr,] for ln_lr in ln_likelihood_ratios]))
		# safety checks
		if numpy.isnan(ln_f_over_b).any():
			raise ValueError("NaN encountered in ranking statistic PDF ratios")
		if numpy.isinf(ln_f_over_b).any():
			raise ValueError("infinity encountered in ranking statistic PDF ratios")
	elif nsample > 0:
		raise ValueError("must supply ranking data to run MCMC sampler")
	else:
		# no-op path
		ln_f_over_b = numpy.array([])

	#
	# initialize MCMC sampler.  seed the walkers randomly
	#

	ndim = len(ln_likelihood_ratios)
	nwalkers = 50 * 2 * ndim	# must be even and >= 2 * ndim
	nburn = 1000

	pos0 = numpy.random.random((nwalkers, ndim)).round()

	#
	# run MCMC sampler to generate {f_i} vector samples
	#

	if progressbar is not None:
		progressbar.max = nsample + nburn
		progressbar.show()

	def log_posterior(f, ln_f_over_b = ln_f_over_b, ln_double_fac_table = ln_double_fac_table(len(ln_f_over_b))):
		"""
		Compute the natural logarithm of

		[ \prod_{\{i | f_i = 1\}} \hat{f}(x_i) / \hat{b}(x_i) ] (2 m - 1)!! (2 n - 1)!!
		"""
		assert len(f) == len(ln_f_over_b)

		# require 0 <= f_i <= 1
		if (f < 0.).any() or (f > 1.).any():
			return NegInf

		terms = numpy.compress(f.round(), ln_f_over_b)

		return terms.sum() + ln_double_fac_table[len(terms)]

	counts = numpy.zeros((nwalkers, ndim), dtype = "double")
	for coordslist in run_mcmc(nwalkers, ndim, nsample, log_posterior, n_burn = nburn, pos0 = pos0, progressbar = progressbar):
		counts += coordslist.round()

	#
	# compute P(signal) for each candidate from the mean of the boolean
	# samples
	#

	p_signal = (counts / nsample).mean(0)

	#
	# done
	#

	return p_signal

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
from lal import rate


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
	lnprobfunc, whose signature must be::

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


def moment_from_lnpdf(ln_pdf, moment, c = 0.):
	if len(ln_pdf.bins) != 1:
		raise ValueError("BinnedLnPDF must have 1 dimension")
	x, = ln_pdf.centres()
	return ((x - c)**moment * (ln_pdf.array / ln_pdf.array.sum())).sum()


def mean_from_lnpdf(ln_pdf):
	return moment_from_lnpdf(ln_pdf, 1.)


def variance_from_lnpdf(ln_pdf):
	return moment_from_pdf(ln_pdf, 2., mean_from_lnpdf(ln_pdf))


def median_from_lnpdf(ln_pdf):
	#
	# get bin probabilities from bin counts
	#

	assert (ln_pdf.array >= 0.).all(), "PDF contains negative counts"
	cdf = ln_pdf.array.cumsum()
	cdf /= cdf[-1]

	#
	# wrap (CDF - 0.5) in an InterpBinnedArray and use bisect to solve
	# for 0 crossing.  the interpolator gets confused where the x
	# co-ordinates are infinite and it breaks the bisect function for
	# some reason, so to work around the issue we start the upper and
	# lower boundaries in a little bit from the edges of the domain.
	# FIXME:  this is stupid, find a root finder that gives the correct
	# answer regardless.
	#

	func = rate.InterpBinnedArray(rate.BinnedArray(ln_pdf.bins, cdf - 0.5))
	median = optimize.bisect(
		func,
		ln_pdf.bins[0].lower()[2], ln_pdf.bins[0].upper()[-3],
		xtol = 1e-14, disp = False
	)

	#
	# check result (detects when the root finder has failed for some
	# reason)
	#

	assert abs(func(median)) < 1e-13, "failed to find median (got %g)" % median

	#
	# done
	#

	return median


def confidence_interval_from_lnpdf(ln_pdf, confidence = 0.95):
	"""
	Constructs a confidence interval based on a BinnedArray object
	containing a normalized 1-D PDF.  Returns the tuple (mode, lower bound,
	upper bound).
	"""
	# check for funny input
	if numpy.isnan(ln_pdf.array).any():
		raise ValueError("NaNs encountered in PDF")
	if numpy.isinf(ln_pdf.array).any():
		raise ValueError("infinities encountered in PDF")
	if (ln_pdf.array < 0.).any():
		raise ValueError("negative values encountered in PDF")
	if not 0.0 <= confidence <= 1.0:
		raise ValueError("confidence must be in [0, 1]")

	centres, = ln_pdf.centres()
	upper = ln_pdf.bins[0].upper()
	lower = ln_pdf.bins[0].lower()

	mode_index = ln_pdf.bins[ln_pdf.argmax()]

	# get bin probabilities from bin counts
	P = ln_pdf.array / ln_pdf.array.sum()
	assert (0. <= P).all()
	assert (P <= 1.).all()
	if abs(P.sum() - 1.0) > 1e-13:
		raise ValueError("PDF is not normalized (integral = %g)" % P.sum())

	# don't need ln_pdf anymore.  only need values at bin centres
	ln_pdf = ln_pdf.at_centres()

	li = ri = mode_index
	P_sum = P[mode_index]
	while P_sum < confidence:
		if li <= 0 and ri >= len(P) - 1:
			raise ValueError("failed to achieve requested confidence")

		if li > 0:
			ln_pdf_li = ln_pdf[li - 1]
			P_li = P[li - 1]
		else:
			ln_pdf_li = NegInf
			P_li = 0.
		if ri < len(P) - 1:
			ln_pdf_ri = ln_pdf[ri + 1]
			P_ri = P[ri + 1]
		else:
			ln_pdf_ri = NegInf
			P_ri = 0.

		if ln_pdf_li > ln_pdf_ri:
			li -= 1
			P_sum += P_li
		elif ln_pdf_ri > ln_pdf_li:
			ri += 1
			P_sum += P_ri
		else:
			P_sum += P_li + P_ri
			li = max(li - 1, 0)
			ri = min(ri + 1, len(P) - 1)

	return centres[mode_index], lower[li], upper[ri]


def f_over_b(rankingstatpdf, ln_likelihood_ratios):
	"""
	For each sample of the ranking statistic, evaluate the ratio of the
	signal ranking statistic PDF to background ranking statistic PDF.
	"""
	#
	# check for bad input
	#

	if any(math.isnan(ln_lr) for ln_lr in ln_likelihood_ratios):
		raise ValueError("NaN log likelihood ratio encountered")

	#
	# for each sample of the ranking statistic, evaluate the ratio of
	# the signal ranking statistic PDF to background ranking statistic
	# PDF.
	#

	f = rankingstatpdf.signal_lr_lnpdf.mkinterp()
	b = rankingstatpdf.noise_lr_lnpdf.mkinterp()
	f_over_b = numpy.array([math.exp(f(ln_lr) - b(ln_lr)) for ln_lr in ln_likelihood_ratios])
	# safety checks
	if numpy.isnan(f_over_b).any():
		raise ValueError("NaN encountered in ranking statistic PDF ratios")
	if numpy.isinf(f_over_b).any():
		raise ValueError("infinity encountered in ranking statistic PDF ratios")

	#
	# done
	#

	return f_over_b


#
# =============================================================================
#
#                            Event Rate Posteriors
#
# =============================================================================
#


def maximum_likelihood_rates(rankingstatpdf, ln_likelihood_ratios):
	# initialize posterior PDF for rates
	log_posterior = LogPosterior(numpy.log(f_over_b(rankingstatpdf, ln_likelihood_ratios)))

	# going to use a minimizer to find the max, so need to flip
	# function upside down
	f = lambda x: -log_posterior(x)

	# initial guesses are Rf=1, Rb=(# candidates)
	return tuple(optimize.fmin(f, (1.0, len(ln_likelihood_ratios)), disp = False))


def rate_posterior_from_samples(samples):
	"""
	Construct and return a BinnedArray containing a histogram of a
	sequence of samples.  If limits is None (default) then the limits
	of the binning will be determined automatically from the sequence,
	otherwise limits is expected to be a tuple or other two-element
	sequence providing the (low, high) limits, and in that case the
	sequence can be a generator.
	"""
	nbins = int(math.sqrt(len(samples)) / 40.)
	assert nbins >= 1, "too few samples to construct histogram"
	lo = samples.min() * (1. - nbins / len(samples))
	hi = samples.max() * (1. + nbins / len(samples))
	ln_pdf = rate.BinnedLnPDF(rate.NDBins((rate.LogarithmicBins(lo, hi, nbins),)))
	count = ln_pdf.count	# avoid attribute look-up in loop
	for sample in samples:
		count[sample,] += 1.
	rate.filter_array(ln_pdf.array, rate.gaussian_window(5), use_fft = False)
	ln_pdf.normalize()
	return ln_pdf


def calculate_rate_posteriors(rankingstatpdf, ln_likelihood_ratios, progressbar = None, chain_file = None, nsample = 400000):
	"""
	FIXME:  document this
	"""
	#
	# check for bad input
	#

	if nsample < 0:
		raise ValueError("nsample < 0: %d" % nsample)

	#
	# for each sample of the ranking statistic, evaluate the ratio of
	# the signal ranking statistic PDF to background ranking statistic
	# PDF.
	#

	if rankingstatpdf is not None:
		ln_f_over_b = numpy.log(f_over_b(rankingstatpdf, ln_likelihood_ratios))
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
	# been drawn from some power of the correct PDF, so the counts in
	# the bins must be corrected (BinnedLnPDF's internal array always
	# stores raw counts).  how to correct count:
	#
	# correct count = (correct PDF) * (bin size)
	#               = (measured PDF)^exponent * (bin size)
	#               = (measured count / (bin size))^exponent * (bin size)
	#               = (measured count)^exponent / (bin size)^(exponent - 1)
	#
	# this assumes small bin sizes.
	#

	Rf_ln_pdf = rate_posterior_from_samples(samples[:,:,0].flatten())
	Rf_ln_pdf.array = Rf_ln_pdf.array**exponent / Rf_ln_pdf.bins.volumes()**(exponent - 1.)
	Rf_ln_pdf.normalize()

	Rb_ln_pdf = rate_posterior_from_samples(samples[:,:,1].flatten())
	Rb_ln_pdf.array = Rb_ln_pdf.array**exponent / Rb_ln_pdf.bins.volumes()**(exponent - 1.)
	Rb_ln_pdf.normalize()

	#
	# done
	#

	return Rf_ln_pdf, Rb_ln_pdf


def calculate_alphabetsoup_rate_posteriors(rankingstatpdf, ln_likelihood_ratios, progressbar = None, nsample = 400000):
	"""
	FIXME:  document this
	"""
	#
	# check for bad input
	#

	if nsample < 0:
		raise ValueError("nsample < 0: %d" % nsample)

	#
	# for each sample of the ranking statistic, evaluate the ratio of
	# the signal ranking statistic PDF to background ranking statistic
	# PDF.
	#

	ln_f_over_b = numpy.log(f_over_b(rankingstatpdf, ln_likelihood_ratios))

	#
	# initialize MCMC chain.  try loading a chain from a chain file if
	# provided, otherwise seed the walkers for a burn-in period
	#

	ndim = 3
	nwalkers = 10 * 2 * ndim	# must be even and >= 2 * ndim
	nburn = 1000

	pos0 = numpy.zeros((nwalkers, ndim), dtype = "double")

	if progressbar is not None:
		progressbar.max = nsample + nburn
		progressbar.show()

	#
	# seed signal rate walkers from exponential distribution,
	# background rate walkers from poisson distribution
	#

	samples = numpy.empty((nsample, nwalkers, ndim), dtype = "double")
	pos0[:,0] = numpy.random.exponential(scale = 1., size = (nwalkers,))
	pos0[:,1] = numpy.random.exponential(scale = 1., size = (nwalkers,))
	pos0[:,2] = numpy.random.poisson(lam = len(ln_likelihood_ratios), size = (nwalkers,))

	#
	# run MCMC sampler to generate (foreground rate 1, foreground rate
	# 2, background rate) samples.
	#

	ln_f_over_b.sort()
	def log_posterior(Rf1f2b, x1 = math.exp(ln_f_over_b[-1]), x2 = math.exp(ln_f_over_b[-2]), std_log_posterior = LogPosterior(ln_f_over_b[:-2])):
		Rf1, Rf2, Rb = Rf1f2b
		if Rf1 < 0. or Rf2 < 0. or Rb < 0.:
			return NegInf
		return math.log(Rf1 / Rb * x1 + 1.) + math.log(Rf2 / Rb * x2 + 1.) + 2. * math.log(Rb) + std_log_posterior((Rf1 + Rf2, Rb))

	for j, coordslist in enumerate(run_mcmc(nwalkers, ndim, nsample, log_posterior, n_burn = nburn, pos0 = pos0, progressbar = progressbar)):
		# coordslist is nwalkers x ndim
		samples[j,:,:] = coordslist

	#
	# safety check
	#

	if samples.min() < 0:
		raise ValueError("MCMC sampler yielded negative rate(s)")

	#
	# compute marginalized PDFs for the foreground and background
	# rates.  for each PDF, the samples from the MCMC are histogrammed
	# and convolved with a density estimation kernel.
	#

	Rf1_pdf = rate_posterior_from_samples(samples[:,:,0].flatten())

	Rf2_pdf = rate_posterior_from_samples(samples[:,:,1].flatten())

	Rf12_pdf = rate_posterior_from_samples((samples[:,:,0] + samples[:,:,1]).flatten())

	Rb_pdf = rate_posterior_from_samples(samples[:,:,2].flatten())

	#
	# done
	#

	return Rf1_pdf, Rf2_pdf, Rf12_pdf, Rb_pdf


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


def calculate_psignal_posteriors(rankingstatpdf, ln_likelihood_ratios, progressbar = None, chain_file = None, nsample = 400000):
	"""
	FIXME:  document this
	"""
	#
	# check for bad input
	#

	if nsample < 0:
		raise ValueError("nsample < 0: %d" % nsample)

	#
	# for each sample of the ranking statistic, evaluate the ratio of
	# the signal ranking statistic PDF to background ranking statistic
	# PDF.
	#

	if rankingstatpdf is not None:
		ln_f_over_b = numpy.log(f_over_b(rankingstatpdf, ln_likelihood_ratios))
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


def calculate_psignal_posteriors_from_rate_samples(rankingstatpdf, ln_likelihood_ratios, nsample = 100000, progressbar = None):
	"""
	FIXME:  document this
	"""
	#
	# for each sample of the ranking statistic, evaluate the ratio of
	# the signal ranking statistic PDF to background ranking statistic
	# PDF.
	#

	f_on_b = f_over_b(rankingstatpdf, ln_likelihood_ratios)

	#
	# run MCMC sampler to generate (foreground rate, background rate)
	# samples and compute
	#
	# p(g_j = 1 | d) = 1/N \sum_{k = 1}^{N} Rf_k * f(x_j) / [Rf_k*f(x_j) + Rb_k*b(x_j)]
	#
	#

	ndim = 2
	nwalkers = 10 * 2 * ndim	# must be even and >= 2 * ndim
	nburn = 1000

	pos0 = numpy.zeros((nwalkers, ndim), dtype = "double")
	pos0[:,0] = numpy.random.exponential(scale = 1., size = (nwalkers,))
	pos0[:,1] = numpy.random.poisson(lam = len(ln_likelihood_ratios), size = (nwalkers,))

	if progressbar is not None:
		progressbar.max = nsample + nburn
		progressbar.show()

	p_signal = numpy.zeros_like(f_on_b)
	for coordslist in run_mcmc(nwalkers, ndim, nsample, LogPosterior(numpy.log(f_on_b)), n_burn = nburn, pos0 = pos0, progressbar = progressbar):
		# coordslist is nwalkers x 2 array providing an array of
		# (Rf, Rb) sample pairs
		for Rf, Rb in coordslist:
			x = Rf * f_on_b
			p_signal += x / (x + Rb)
	p_signal /= nwalkers * nsample

	#
	# done
	#

	return p_signal

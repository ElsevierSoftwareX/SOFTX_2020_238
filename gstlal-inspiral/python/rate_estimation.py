# Copyright (C) 2011--2014  Kipp Cannon, Chad Hanna, Drew Keppel
# Copyright (C) 2013  Jacob Peoples
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
import sys


from glue.text_progress_bar import ProgressBar
from pylal import rate


from gstlal import emcee
from gstlal._rate_estimation import *


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
	if sampler.acceptance_fraction.min() < 0.5:
		print >>sys.stderr, "\nwarning:  low sampler acceptance fraction (min %g)" % sampler.acceptance_fraction.min()


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
	nbins = len(samples) // 729
	binnedarray = rate.BinnedArray(rate.NDBins((rate.LinearBins(lo, hi, nbins),)))
	for sample in samples:
		binnedarray[sample,] += 1.
	rate.filter_array(binnedarray.array, rate.gaussian_window(5))
	numpy.clip(binnedarray.array, 0.0, PosInf, binnedarray.array)
	return binnedarray


def calculate_rate_posteriors(ranking_data, ln_likelihood_ratios, restrict_to_instruments = None, progressbar = None, chain_file = None, nsample = None):
	"""
	FIXME:  document this
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

	b = ranking_data.background_likelihood_pdfs[restrict_to_instruments]
	f = ranking_data.signal_likelihood_pdfs[restrict_to_instruments]
	ln_f_over_b = numpy.array([math.log(f[ln_lr,] / b[ln_lr,]) for ln_lr in ln_likelihood_ratios])

	# remove NaNs.  these occur because the ranking statistic PDFs have
	# been zeroed at the cut-off and some events get pulled out of the
	# database with ranking statistic values in that bin
	#
	# FIXME:  re-investigate the decision to zero the bin at threshold.
	# the original motivation for doing it might not be there any
	# longer
	ln_f_over_b = ln_f_over_b[~numpy.isnan(ln_f_over_b)]
	# safety check
	if numpy.isinf(ln_f_over_b).any():
		raise ValueError("infinity encountered in ranking statistic PDF ratios")

	#
	# initializer MCMC chain.  try loading a chain from a chain file if
	# provided, otherwise seed the walkers for a burn-in period
	#

	ndim = 2
	nwalkers = 10 * 2 * ndim	# must be even and >= 2 * ndim
	if nsample is None:
		nsample = 400000
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
	# samples.
	#

	for j, coordslist in enumerate(run_mcmc(nwalkers, ndim, max(0, nsample - i), posterior(ln_f_over_b), n_burn = nburn, pos0 = pos0, progressbar = progressbar), i):
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
	#               = (measured PDF)^2 * (bin size)
	#               = (measured count / (bin size))^2 * (bin size)
	#               = (measured count)^2 / (bin size)
	#
	# this assumes small bin sizes.
	#

	Rf_pdf = binned_rates_from_samples(samples[:,:,0].flatten())
	Rf_pdf.array = Rf_pdf.array**2. / Rf_pdf.bins.volumes()
	Rf_pdf.to_pdf()

	Rb_pdf = binned_rates_from_samples(samples[:,:,1].flatten())
	Rb_pdf.array = Rb_pdf.array**2. / Rb_pdf.bins.volumes()
	Rb_pdf.to_pdf()

	#
	# done
	#

	return Rf_pdf, Rb_pdf


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
	assert not numpy.isinf(bin_widths).any(), "infinite bin sizes not supported"
	P = binned_array.array * bin_widths
	if abs(P.sum() - 1.0) > 1e-13:
		raise ValueError("rate PDF is not normalized")

	li = ri = mode_index
	P_sum = P[mode_index]
	while P_sum < confidence:
		if li <= 0 and ri >= len(P) - 1:
			raise ValueError("failed to achieve requested confidence")
		P_li = P[li - 1] if li > 0 else 0.
		P_ri = P[ri + 1] if ri < len(P) - 1 else 0.
		assert P_li >= 0. and P_ri >= 0.
		if P_li > P_ri:
			P_sum += P_li
			li -= 1
		elif P_ri > P_li:
			P_sum += P_ri
			ri += 1
		else:
			P_sum += P_li + P_ri
			li = max(li - 1, 0)
			ri = min(ri + 1, len(P) - 1)
	return centres[mode_index], lower[li], upper[ri]

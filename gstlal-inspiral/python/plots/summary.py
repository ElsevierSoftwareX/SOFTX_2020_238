# Copyright (C) 2009-2013  Kipp Cannon, Chad Hanna, Drew Keppel
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

### A program to produce a variety of plots from a gstlal inspiral analysis, e.g. IFAR plots, missed found, etc.

#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


import bisect
import itertools
import math
import matplotlib
import matplotlib.figure
matplotlib.rcParams.update({
	"font.size": 12.0,
	"axes.titlesize": 12.0,
	"axes.labelsize": 12.0,
	"xtick.labelsize": 12.0,
	"ytick.labelsize": 12.0,
	"legend.fontsize": 10.0,
	"figure.dpi": 300,
	"savefig.dpi": 300,
	"text.usetex": True,
	"path.simplify": True,
	"font.family": "serif"
})
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy
import os
import scipy
import sys

import lal

from ligo import segments
from ligo.lw import lsctables
from gstlal.plots.util import golden_ratio


#
# =============================================================================
#
#                                  Utilities
#
# =============================================================================
#


def create_plot(x_label = None, y_label = None, width = 165.0, aspect = golden_ratio):
	#
	# width is in mm, default aspect ratio is the golden ratio
	#
	fig = matplotlib.figure.Figure()
	FigureCanvas(fig)
	if aspect is None:
		aspect = golden_ratio
	fig.set_size_inches(width / 25.4, width / 25.4 / aspect)
	axes = fig.add_axes((0.1, 0.12, .875, .80))
	axes.grid(True)
	if x_label is not None:
		axes.set_xlabel(x_label)
	if y_label is not None:
		axes.set_ylabel(y_label)
	return fig, axes


def marker_and_size(n):
	if n > 2000:
		return "ko", (10000.0 / n)**0.7
	else:
		return "k.", None


def sigma_region(mean, nsigma):
	return numpy.concatenate((mean - nsigma * numpy.sqrt(mean), (mean + nsigma * numpy.sqrt(mean))[::-1]))


#
# =============================================================================
#
#                            Plotting Routines
#
# =============================================================================
#


def plot_injection_param_dist(x, y, x_label, y_label, waveform, aspect=None):
	fig, axes = create_plot(x_label, y_label)
	waveform_name = waveform.replace("_", "\_")
	axes.set_title(fr"$\textrm{{Injection Parameter Distribution ({waveform_name} Injections)}}$")
	mrkr, markersize = marker_and_size(len(x))
	if markersize is not None:
		axes.plot(x, y, mrkr, markersize=markersize)
	else:
		axes.plot(x, y, mrkr)
	minx, maxx = axes.get_xlim()
	miny, maxy = axes.get_ylim()
	if aspect == 1:
		axes.set_xlim((min(minx, miny), max(maxx, maxy)))
		axes.set_ylim((min(minx, miny), max(maxx, maxy)))

	return fig


def plot_missed_found(found, missed, x_label, y_label, title, legend=True):
	fig, axes = create_plot(x_label, y_label)
	legend = []
	for ifos, (x_found, y_found) in found.items():
		legend.append("Found in %s" % ", ".join(sorted(ifos)))
		axes.semilogy(x_found, y_found, ".")
	if missed:
		legend.append("Missed")
		axes.semilogy(*missed, "k.")
	if legend:
		axes.legend(legend)
	axes.set_title(title)

	return fig


def plot_param_accuracy_histogram(data, label, title):
	fig, axes = create_plot(label, "Number")
	start = scipy.stats.mstats.mquantiles(data, 0.01)
	end = scipy.stats.mstats.mquantiles(data, 0.99)
	axes.hist(data, numpy.linspace(start, end, 100))
	axes.set_title(title)

	return fig


def plot_param_accuracy_scatter(x, y, z, x_label, y_label, z_label, title):
	fig, axes = create_plot(x_label, y_label)
	cb = axes.scatter(x, y, c=z, norm=matplotlib.colors.LogNorm(), vmin=1e-13, vmax=1e-3, linewidth=0.2, alpha=0.8)
	axes.set_title(title)
	fig.colorbar(cb, ax=axes).set_label(z_label)

	return fig


def plot_param_accuracy(x, y, x_label, y_label, title, loglog=False):
	fig, axes = create_plot(x_label, y_label)
	if loglog:
		axes.loglog(x, y, "kx")
	else:
		axes.plot(x, y, "kx")
	axes.set_title(title)

	return fig


def plot_snr_chi2_background(ifo, min_snr, injections, background, zerolag=None):
	fig, axes = create_plot(r"$\rho$", r"$\chi^{2}$")
	axes.loglog(injections[ifo].snr, injections[ifo].chi2, 'r.', label = "Inj")
	axes.loglog(background[ifo].snr, background[ifo].chi2, "kx", label = "Background")
	if zerolag:
		axes.loglog(zerolag[ifo].snr, zerolag[ifo].chi2, "bx", label = "Zero-lag")
		axes.set_title(fr"$\chi^{2}$ vs.\ $\rho$ in {ifo}")
	else:
		axes.set_title(fr"$\chi^{2}$ vs.\ $\rho$ in {ifo} (Closed box)")
	axes.legend(loc = "upper left")
	axes.set_xlim((min_snr, None))

	return fig


def plot_param_background_multiifo(ifo1, ifo2, min_snr, name, symbol, injections, background, zerolag=None):
	fig, axes = create_plot(fr"{symbol} in {ifo1}", fr"{symbol} in {ifo2}", aspect = 1.0)
	axes.grid(True, which = "both")
	axes.loglog([x for x, y in injections], [y for x, y in injections], "rx", label = "Injections")
	axes.loglog([x for x, y in background], [y for x, y in background], "kx", label = "Background")
	if zerolag:
		axes.loglog([x for x, y in zerolag], [y for x, y in zerolag], "bx", label = "Zero-lag")
		axes.set_title(fr"{name} in {ifo1} vs.\ {ifo2}")
	else:
		axes.set_title(fr"{name} in {ifo1} vs.\ {ifo2} (Closed Box)")
	axes.legend(loc = "lower right")
	axes.set_xlim((min_snr, None))
	axes.set_ylim((min_snr, None))

	return fig


def plot_mass_chi2_snr2_background(inj_mchirp, inj_chisq, inj_snr, bg_mchirp, bg_chisq, bg_snr, ifo):
	fig, axes = create_plot(r"$M_\mathrm{chirp}$", r"$\chi^{2}/\rho^2$")
	coll = axes.scatter(inj_mchirp, inj_chisq / inj_snr**2, c = inj_snr, label = "Injections", vmax = 20, linewidth = 0.2, alpha = 0.8)
	fig.colorbar(coll, ax = axes).set_label("SNR in %s" % ifo)
	axes.scatter(bg_mchirp, bg_chisq / bg_snr**2, edgecolor = 'k', c = bg_snr, marker = 'x', label = "Background")
	axes.legend(loc = "upper left")
	axes.semilogy()

	return fig


def plot_background_param_dist(x, y, z, x_label, y_label, z_label, title):
	fig, axes = create_plot(x_label, y_label)
	coll = axes.scatter(x, y, c = z, linewidth = 0.2, alpha = 0.8)
	fig.colorbar(coll, ax = axes).set_label(z_label)
	axes.set_title(title)

	return fig


def plot_candidate_lnL_vs_snr(min_snr, bg_snr, bg_ln_likelihood_ratio, zl_snr=None, zl_ln_likelihood_ratio=None):
	# assume at least two instruments are required
	fig, axes = create_plot(r"$\sqrt{\sum_{\mathrm{instruments}} \mathrm{SNR}^{2}}$", r"$\ln \mathcal{L}$")
	axes.grid(True, which = "both")
	axes.plot(bg_snr, bg_ln_likelihood_ratio, "kx", label = "Background")
	if zl_snr and zl_ln_likelihood_ratio:
		axes.plot(zl_snr, zl_ln_likelihood_ratio, "bx", label = "Zero-lag")
		axes.set_title(r"Candidate's $\ln \mathcal{L}$ vs.\ SNR")
	else:
		axes.set_title(r"Candidate's $\ln \mathcal{L}$ vs.\ SNR (Closed Box)")
	axes.set_xlim(((2. * min_snr**2.)**.5, None))
	axes.legend(loc = "upper left")

	return fig


def plot_rate_vs_ifar(zerolag_stats, fapfar, is_open_box=False, stair_steps=False):
	fig, axes = create_plot(r"Inverse False-Alarm Rate (s)", r"Number of Events")
	expected_count_y = numpy.logspace(-7, numpy.log10(len(zerolag_stats)), 100)
	expected_count_x = fapfar.livetime / expected_count_y
	axes.loglog()

	# determine plot limits
	xlim = (fapfar.livetime / fapfar.count_above_threshold, fapfar.livetime / 1e-4)
	xlim = max(zerolag_stats.min(), xlim[0]), max(2.**math.ceil(math.log(zerolag_stats.max(), 2.)), xlim[1])
	ylim = 0.001, (10.**math.ceil(math.log10(expected_count_y[::-1][bisect.bisect_right(expected_count_x[::-1], xlim[0])])) if xlim[0] is not None else None)

	# background. stair_steps option makes the background
	# stair-step-style like the observed counts
	if stair_steps:
		expected_count_x = expected_count_x.repeat(2)[1:]
		expected_count_y = expected_count_y.repeat(2)[:-1]
	line1, = axes.plot(expected_count_x, expected_count_y, 'k--', linewidth = 1)

	# error bands
	expected_count_x = numpy.concatenate((expected_count_x, expected_count_x[::-1]))
	line2, = axes.fill(expected_count_x, sigma_region(expected_count_y, 3.0).clip(*ylim), alpha = 0.25, facecolor = [0.75, 0.75, 0.75])
	line3, = axes.fill(expected_count_x, sigma_region(expected_count_y, 2.0).clip(*ylim), alpha = 0.25, facecolor = [0.5, 0.5, 0.5])
	line4, = axes.fill(expected_count_x, sigma_region(expected_count_y, 1.0).clip(*ylim), alpha = 0.25, facecolor = [0.25, 0.25, 0.25])

	# zero-lag
	N = numpy.arange(1., len(zerolag_stats) + 1., dtype = "double")
	line5, = axes.plot(zerolag_stats.repeat(2)[1:], N.repeat(2)[:-1], 'k', linewidth = 2)

	# legend
	labels = [r"Expected, $\langle N \rangle$", r"$\pm\sqrt{\langle N \rangle}$", r"$\pm 2\sqrt{\langle N \rangle}$", r"$\pm 3\sqrt{\langle N \rangle}$"]
	if is_open_box:
		axes.legend((line5, line1, line4, line3, line2), tuple(["Observed"] + labels), loc = "upper right")
	else:
		axes.legend((line5, line1, line4, line3, line2), tuple([r"Observed (time shifted)"] + labels), loc = "upper right")

	# adjust bounds of plot
	axes.set_xlim(xlim)
	axes.set_ylim(ylim)

	# set title
	if is_open_box:
		axes.set_title(r"Event Count vs.\ Inverse False-Alarm Rate Threshold")
	else:
		axes.set_title(r"Event Count vs.\ Inverse False-Alarm Rate Threshold (Closed Box)")

	return fig


def plot_rate_vs_lnL(zerolag_stats, fapfar, is_open_box):
	fig, axes = create_plot(r"$\ln \mathcal{L}$ Threshold", r"Number of Events $\geq \ln \mathcal{L}$")
	axes.semilogy()

	# plot limits and expected counts
	def expected_count(lr):
		return fapfar.far_from_rank(lr) * fapfar.livetime

	xlim = max(zerolag_stats.min(), fapfar.minrank), max(2. * math.ceil(zerolag_stats.max() / 2.), 30.)
	ylim = 0.001, 10.**math.ceil(math.log10(expected_count(xlim[0])))

	# expected count curve
	expected_count_x = numpy.linspace(xlim[0], xlim[1], 10000)
	expected_count_y = list(map(expected_count, expected_count_x))
	line1, = axes.plot(expected_count_x, expected_count_y, 'k--', linewidth = 1)

	# error bands
	expected_count_x = numpy.concatenate((expected_count_x, expected_count_x[::-1]))
	line2, = axes.fill(expected_count_x, sigma_region(expected_count_y, 3.0).clip(*ylim), alpha = 0.25, facecolor = [0.75, 0.75, 0.75])
	line3, = axes.fill(expected_count_x, sigma_region(expected_count_y, 2.0).clip(*ylim), alpha = 0.25, facecolor = [0.5, 0.5, 0.5])
	line4, = axes.fill(expected_count_x, sigma_region(expected_count_y, 1.0).clip(*ylim), alpha = 0.25, facecolor = [0.25, 0.25, 0.25])

	# zero-lag
	if zerolag_stats is not None:
		N = numpy.arange(1., len(zerolag_stats) + 1., dtype = "double")
		line5, = axes.plot(zerolag_stats.repeat(2)[1:], N.repeat(2)[:-1], 'k', linewidth = 2)

	# legend
	labels = [r"Noise Model, $\langle N \rangle$", r"$\pm\sqrt{\langle N \rangle}$", r"$\pm 2\sqrt{\langle N \rangle}$", r"$\pm 3\sqrt{\langle N \rangle}$"]
	if is_open_box:
		axes.legend((line5, line1, line4, line3, line2), tuple(["Observed"] + labels), loc = "upper right")
	else:
		axes.legend((line5, line1, line4, line3, line2), tuple([r"Observed (time shifted)"] + labels), loc = "upper right")

	# adjust bounds of plot
	axes.set_xlim(xlim)
	axes.set_ylim(ylim)

	# set title
	if is_open_box:
		axes.set_title(r"Event Count vs.\ Ranking Statistic Threshold")
	else:
		axes.set_title(r"Event Count vs.\ Ranking Statistic Threshold (Closed Box)")

	return fig


def plot_rate_vs_background_lnL(zerolag_stats, background_stats, fapfar):
	fig, axes = create_plot(r"$\ln \mathcal{L}$ Threshold", r"Number of Events $\geq \ln \mathcal{L}$")
	axes.semilogy()

	# plot limits and expected counts
	def expected_count(lr):
		return fapfar.far_from_rank(lr) * fapfar.livetime

	xlim = max(zerolag_stats.min(), fapfar.minrank), 2. * math.ceil(min(max(zerolag_stats.max(), background_stats.max()), 200.) / 2.)
	ylim = 10.**math.floor(math.log10(expected_count(zerolag_stats.max()))), 10.**math.ceil(math.log10(expected_count(xlim[0])))

	axes.set_position((0.10, 0.12, .88, .78))
	axes.set_title(r"Event Count vs.\ Ranking Statistic Threshold", position = (0.5, 1.05))

	# expected count curve
	expected_count_x = numpy.linspace(xlim[0], xlim[1], 10000)
	expected_count_y = list(map(expected_count, expected_count_x))
	line1, = axes.plot(expected_count_x, expected_count_y, 'k--', linewidth = 1)

	# time slide
	N = numpy.arange(1., len(background_stats) + 1., dtype = "double")
	line5, = axes.plot(background_stats.repeat(2)[1:], N.repeat(2)[:-1], 'k', linewidth = 2)

	# zero-lag
	N = numpy.arange(1., len(zerolag_stats) + 1., dtype = "double")
	line6, = axes.plot(zerolag_stats.repeat(2)[1:], N.repeat(2)[:-1], 'r', linewidth = 1)

	# legend
	axes.legend((line1, line5, line6), (r"Noise Model", r"Time-shifted Trial", r"Observed"), loc = "lower left", handlelength = 5)

	# adjust bounds of plot
	axes.set_xlim(xlim)
	axes.set_ylim(ylim)

	# build false-alarm probability labels
	ax = axes.twiny()
	ax.set_xlim(xlim)
	ax.set_xlabel(r"$P(\geq 1 \mathrm{candidate} | \mathrm{noise})$", horizontalalignment = "left", position = (0.05, 1.02), size = "small")

	if False:
		label_rank_alpha = [
			(None, xlim[0], None),	# left-edge of plot
			(r"$1 \sigma$", fapfar.rank_from_fap(math.erfc(1. / math.sqrt(2.))), 0.1),
			(r"$2 \sigma$", fapfar.rank_from_fap(math.erfc(2. / math.sqrt(2.))), 0.2),
			(r"$3 \sigma$", fapfar.rank_from_fap(math.erfc(3. / math.sqrt(2.))), 0.3),
			(r"$4 \sigma$", fapfar.rank_from_fap(math.erfc(4. / math.sqrt(2.))), 0.4),
			(r"$5 \sigma$", fapfar.rank_from_fap(math.erfc(5. / math.sqrt(2.))), 0.5)
		]
	else:
		label_rank_alpha = [
			(None, xlim[0], None),	# left-edge of plot
			(r"$10^{-1}$", fapfar.rank_from_fap(0.1), 0.1),
			(r"$10^{-3}$", fapfar.rank_from_fap(0.001), 0.2),
			(r"$10^{-6}$", fapfar.rank_from_fap(0.000001), 0.3)
		]

	for (_, lo, _), (_, hi, alpha) in zip(label_rank_alpha[:-1], label_rank_alpha[1:]):
		if hi < lo:	# can happen if xlim[0] > 1 sigma
			continue
		ax.axvspan(lo, hi, color = "k", alpha = alpha)

	ax.set_xticks([rank for (label, rank, _) in label_rank_alpha[1:]])
	ax.set_xticklabels([label for (label, rank, _) in label_rank_alpha[1:]])

	return fig

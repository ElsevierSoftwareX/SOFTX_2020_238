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

## @file

## @package plotfar

#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#

import warnings
import math
import matplotlib
from matplotlib import figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
matplotlib.rcParams.update({
	"font.size": 10.0,
	"axes.titlesize": 10.0,
	"axes.labelsize": 10.0,
	"xtick.labelsize": 8.0,
	"ytick.labelsize": 8.0,
	"legend.fontsize": 8.0,
	"figure.dpi": 300,
	"savefig.dpi": 300,
	"text.usetex": True
})
import numpy
from gstlal import plotutil
from gstlal import far

def init_plot(figsize):
	fig = figure.Figure()
	FigureCanvas(fig)
	fig.set_size_inches(figsize)
	axes = fig.gca()

	return fig, axes


def plot_snr_chi_pdf(rankingstat, instrument, which, snr_max, bankchisq = False, event_snr = None, event_chisq = None, sngls = None):
	if bankchisq:
		base = "snr_bankchi"
	else:
		base = "snr_chi"

	# also checks that which is an allowed value
	tag = {
		"background_pdf": "Noise",
		"injection_pdf": "Signal",
		"zero_lag_pdf": "Candidates",
		"LR": "LR"
	}[which]

	# sngls is a sequence of {instrument: (snr, chisq)} dictionaries,
	# obtain the co-ordinates for a sngls scatter plot for this
	# instrument from that.  need to handle case in which there are no
	# singles for this instrument
	if sngls is not None:
		sngls = numpy.array([sngl[instrument] for sngl in sngls if instrument in sngl])
		if not len(sngls):
			sngls = None

	if which == "background_pdf":
		# a ln PDF object
		binnedarray = rankingstat.denominator.densities["%s_%s" % (instrument, base)]
	elif which == "injection_pdf":
		# a ln PDF object.  numerator has only one, same for all
		# instruments
		binnedarray = rankingstat.numerator.densities["%s" % base]
	elif which == "zero_lag_pdf":
		# a ln PDF object
		binnedarray = rankingstat.zerolag.densities["%s_%s" % (instrument, base)]
	elif which == "LR":
		num = rankingstat.numerator.densities["%s" % base]
		den = rankingstat.denominator.densities["%s_%s" % (instrument, base)]
		assert num.bins == den.bins
		binnedarray = num.count.copy()
		with numpy.errstate(invalid = "ignore"):
			binnedarray.array[:,:] = num.at_centres() - den.at_centres()
		binnedarray.array[num.count.array == 0] = float("-inf")
	else:
		raise ValueError("invalid which (%s)" % which)

	# the last bin can have a centre at infinity, and its value is
	# always 0 anyway so there's no point in trying to include it
	x = binnedarray.bins[0].centres()[:-1]
	y = binnedarray.bins[1].centres()[:-1]
	z = binnedarray.at_centres()[:-1,:-1]
	if numpy.isnan(z).any():
		if numpy.isnan(z).all():
			warnings.warn("%s %s is all NaN, skipping" % (instrument, which))
			return None
		warnings.warn("%s %s contains NaNs" % (instrument, which))
		z = numpy.ma.masked_where(numpy.isnan(z), z)

	# the range of the plots
	xlo, xhi = rankingstat.snr_min, snr_max
	ylo, yhi = .0001, 1.

	x = x[binnedarray.bins[xlo:xhi, ylo:yhi][0]]
	y = y[binnedarray.bins[xlo:xhi, ylo:yhi][1]]
	z = z[binnedarray.bins[xlo:xhi, ylo:yhi]]

	fig, axes = init_plot((8., 8. / plotutil.golden_ratio))
	if which == "LR":
		norm = matplotlib.colors.Normalize(vmin = -80., vmax = +200.)
		levels = numpy.linspace(-80., +200, 141)
	elif which == "background_pdf":
		norm = matplotlib.colors.Normalize(vmin = -30., vmax = z.max())
		levels = 50
	elif which == "injection_pdf":
		norm = matplotlib.colors.Normalize(vmin = -60., vmax = z.max())
		levels = 50
	elif which == "zero_lag_pdf":
		norm = matplotlib.colors.Normalize(vmin = -30., vmax = z.max())
		levels = 50
	else:
		raise ValueError("invalid which (%s)" % which)

	mesh = axes.pcolormesh(x, y, z.T, norm = norm, cmap = "afmhot", shading = "gouraud")
	if which == "LR":
		cs = axes.contour(x, y, z.T, levels, norm = norm, colors = "k", linewidths = .5, alpha = .3)
		axes.clabel(cs, [-20., -10., 0., +10., +20.], fmt = "%g", fontsize = 8)
	else:
		axes.contour(x, y, z.T, levels, norm = norm, colors = "k", linestyles = "-", linewidths = .5, alpha = .3)
	if event_snr is not None and event_chisq is not None:
		axes.plot(event_snr, event_chisq / event_snr / event_snr, "ko", mfc = "None", mec = "g", ms = 14, mew=4)
	if sngls is not None:
		axes.plot(sngls[:,0], sngls[:,1] / sngls[:,0]**2., "b.", alpha = .2)
	axes.loglog()
	axes.grid(which = "both")
	#axes.set_xlim((xlo, xhi))
	#axes.set_ylim((ylo, yhi))
	fig.colorbar(mesh, ax = axes)
	axes.set_xlabel(r"$\mathrm{SNR}$")
	if bankchisq:
		label = "_{\mathrm{bank}}"
	else:
		label = ""
	if tag.lower() in ("signal",):
		axes.set_title(r"$\ln P(\chi%s^{2} / \mathrm{SNR}^{2} | \mathrm{SNR}, \mathrm{%s})$ in %s" % (label, tag.lower(), instrument))
	elif tag.lower() in ("noise", "candidates"):
		axes.set_title(r"$\ln P(\mathrm{SNR}, \chi%s^{2} / \mathrm{SNR}^{2} | \mathrm{%s})$ in %s" % (label, tag.lower(), instrument))
	elif tag.lower() in ("lr",):
		axes.set_title(r"$\ln P(\chi%s^{2} / \mathrm{SNR}^{2} | \mathrm{SNR}, \mathrm{signal} ) / P(\mathrm{SNR}, \chi%s^{2} / \mathrm{SNR}^{2} | \mathrm{noise})$ in %s" % (label, label, instrument))
	else:
		raise ValueError(tag)
	try:
		fig.tight_layout(pad = .8)
	except RuntimeError:
		if bankchisq:
			label = "bank"
		else:
			label = ""
		if tag.lower() in ("signal",):
			axes.set_title("ln P(chi%s^2 / SNR^2 | SNR, %s) in %s" % (label, tag.lower(), instrument))
		elif tag.lower() in ("noise", "candidates"):
			axes.set_title("ln P(SNR, chi%s^2 / SNR^2 | %s) in %s" % (label, tag.lower(), instrument))
		elif tag.lower() in ("lr",):
			axes.set_title("ln P(chi%s^2 / SNR^2 | SNR, signal) / P(SNR, chi%s^2 / SNR^2 | noise) in %s" % (label, label, instrument))
		fig.tight_layout(pad = .8)
	return fig


def plot_rates(rankingstat):
	fig = figure.Figure()
	FigureCanvas(fig)
	fig.set_size_inches((6., 6.))
	axes0 = fig.add_subplot(2, 2, 1)
	axes1 = fig.add_subplot(2, 2, 2)
	axes2 = fig.add_subplot(2, 2, 3)
	axes3 = fig.add_subplot(2, 2, 4)

	# singles counts
	labels = []
	sizes = []
	colours = []
	for instrument, count in sorted(rankingstat.denominator.triggerrates.counts.items()):
		labels.append("%s\n(%d)" % (instrument, count))
		sizes.append(count)
		colours.append(plotutil.colour_from_instruments((instrument,)))
	axes0.pie(sizes, labels = labels, colors = colours, autopct = "%.3g%%", pctdistance = 0.4, labeldistance = 0.8)
	axes0.set_title("Trigger Counts")

	# singles rates
	labels = []
	sizes = []
	colours = []
	for instrument, rate in sorted(rankingstat.denominator.triggerrates.densities.items()):
		labels.append("%s\n(%g Hz)" % (instrument, rate))
		sizes.append(rate)
		colours.append(plotutil.colour_from_instruments((instrument,)))
	axes1.pie(sizes, labels = labels, colors = colours, autopct = "%.3g%%", pctdistance = 0.4, labeldistance = 0.8)
	axes1.set_title(r"Mean Trigger Rates (per live-time)")

	# live time
	labels = []
	sizes = []
	colours = []
	seglists = rankingstat.denominator.triggerrates.segmentlistdict()
	for instruments in sorted(rankingstat.denominator.coinc_rates.all_instrument_combos, key = lambda instruments: sorted(instruments)):
		livetime = float(abs(seglists.intersection(instruments) - seglists.union(frozenset(seglists) - instruments)))
		labels.append("%s\n(%g s)" % (", ".join(sorted(instruments)), livetime))
		sizes.append(livetime)
		colours.append(plotutil.colour_from_instruments(instruments))
	axes2.pie(sizes, labels = labels, colors = colours, autopct = "%.3g%%", pctdistance = 0.4, labeldistance = 0.8)
	axes2.set_title("Live-time")

	# projected background counts
	labels = []
	sizes = []
	colours = []
	for instruments, count in sorted(rankingstat.denominator.candidate_count_model().items(), key = lambda (instruments, count): sorted(instruments)):
		labels.append("%s\n(%d)" % (", ".join(sorted(instruments)), count))
		sizes.append(count)
		colours.append(plotutil.colour_from_instruments(instruments))
	axes3.pie(sizes, labels = labels, colors = colours, autopct = "%.3g%%", pctdistance = 0.4, labeldistance = 0.8)
	axes3.set_title("Expected Background Candidate Counts\n(before $\ln \mathcal{L}$ cut)")

	fig.tight_layout(pad = .8)
	return fig


def plot_snr_joint_pdf(snrpdf, instruments, horizon_distances, min_instruments, max_snr, sngls = None):
	if len(instruments) < 1:
		raise ValueError("len(instruments) must be >= 1")

	# retrieve the PDF in binned array form (not the interpolator)
	binnedarray = snrpdf.get_snr_joint_pdf_binnedarray(instruments, horizon_distances, min_instruments)

	# the range of the axes
	xlo, xhi = far.RankingStat.snr_min, max_snr
	mask = binnedarray.bins[(slice(xlo, xhi),) * len(instruments)]

	# axes are in alphabetical order
	instruments = sorted(instruments)

	# sngls is a sequence of {instrument: (snr, chisq)} dictionaries,
	# digest into co-ordinate tuples for a sngls scatter plot
	if sngls is not None:
		# NOTE:  the PDFs are computed subject to the constraint
		# that the candidate is observed in precisely that set of
		# instruments, so we need to restrict ourselves, here, to
		# coincs that involve the combination of instruments in
		# question otherwise we'll be overlaying a scatter plot
		# that we don't believe to have been drawn from the PDF
		# we're showing.
		sngls = numpy.array([tuple(sngl[instrument][0] for instrument in instruments) for sngl in sngls if sorted(sngl) == instruments])

	x = [binning.centres() for binning in binnedarray.bins]
	z = binnedarray.array
	if numpy.isnan(z).any():
		warnings.warn("%s SNR PDF for %s contains NaNs" % (", ".join(instruments), ", ".join("%s=%g" % instdist for instdist in sorted(horizon_distances.items()))))
		z = numpy.ma.masked_where(numpy.isnan(z), z)

	x = [coords[m] for coords, m in zip(x, mask)]
	z = z[mask]

	# one last check for craziness to make error messages more
	# meaningful
	assert not numpy.isnan(z).any()
	assert not (z < 0.).any()

	# plot the natural logarithm of the PDF
	with numpy.errstate(divide = "ignore"):
		z = numpy.log(z)

	if len(instruments) == 1:
		# 1D case
		fig, axes = init_plot((5., 5. / plotutil.golden_ratio))

		# FIXME:  hack to allow all-0 PDFs to be plotted.  remove
		# when we have a version of matplotlib that doesn't crash,
		# whatever version of matplotlib that is
		if numpy.isinf(z).all():
			z[:] = -60.
			z[0] = -55.

		axes.semilogx(x[0], z, color = "k")
		ylo, yhi = -40., max(0., z.max())
		if sngls is not None and len(sngls) == 1:
			axes.axvline(sngls[0, 0])
		axes.set_xlim((xlo, xhi))
		axes.set_ylim((ylo, yhi))
		axes.grid(which = "both", linestyle = "-", linewidth = 0.2)
		axes.set_xlabel(r"$\mathrm{SNR}_{\mathrm{%s}}$" % instruments[0])
		axes.set_ylabel(r"$\ln P(\mathrm{SNR}_{\mathrm{%s}})$" % instruments[0])

	elif len(instruments) == 2:
		# 2D case
		fig, axes = init_plot((5., 4.))

		# FIXME:  hack to allow all-0 PDFs to be plotted.  remove
		# when we have a version of matplotlib that doesn't crash,
		# whatever version of matplotlib that is
		if numpy.isinf(z).all():
			z[:,:] = -60.
			z[0,0] = -55.

		norm = matplotlib.colors.Normalize(vmin = -40., vmax = max(0., z.max()))

		mesh = axes.pcolormesh(x[0], x[1], z.T, norm = norm, cmap = "afmhot", shading = "gouraud")
		axes.contour(x[0], x[1], z.T, 50, norm = norm, colors = "k", linestyles = "-", linewidths = .5, alpha = .3)

		if sngls is not None and len(sngls) == 1:
			axes.plot(sngls[0, 0], sngls[0, 1], "ko", mfc = "None", mec = "g", ms = 14, mew=4)
		elif sngls is not None:
			axes.plot(sngls[:,0], sngls[:,1], "b.", alpha = .2)

		axes.loglog()
		axes.grid(which = "both", linestyle = "-", linewidth = 0.2)
		fig.colorbar(mesh, ax = axes)
		# co-ordinates are in alphabetical order
		axes.set_xlabel(r"$\mathrm{SNR}_{\mathrm{%s}}$" % instruments[0])
		axes.set_ylabel(r"$\mathrm{SNR}_{\mathrm{%s}}$" % instruments[1])

	else:
		# FIXME:  figure out how to plot 3+D PDFs
		return None

	axes.set_title(r"$\ln P(%s | \{%s\}, \mathrm{signal})$" % (", ".join("\mathrm{SNR}_{\mathrm{%s}}" % instrument for instrument in instruments), ", ".join("{D_{\mathrm{H}}}_{\mathrm{%s}}=%.3g" % item for item in sorted(horizon_distances.items()))))
	fig.tight_layout(pad = .8)
	return fig


def plot_likelihood_ratio_pdf(rankingstatpdf, (xlo, xhi), title, which = "noise"):
	fig, axes = init_plot((8., 8. / plotutil.golden_ratio))

	if rankingstatpdf.zero_lag_lr_lnpdf.array.any():
		extincted = rankingstatpdf.new_with_extinction()
	else:
		extincted = None

	if which == "noise":
		lnpdf = rankingstatpdf.noise_lr_lnpdf
		extinctedlnpdf = extincted.noise_lr_lnpdf if extincted is not None else None
		zerolag_lnpdf = rankingstatpdf.zero_lag_lr_lnpdf
	elif which == "signal":
		lnpdf = rankingstatpdf.signal_lr_lnpdf
		extinctedlnpdf = extincted.signal_lr_lnpdf if extincted is not None else None
		zerolag_lnpdf = None
	else:
		raise ValueError("invalid which (%s)" % which)

	axes.semilogy(lnpdf.bins[0].centres(), numpy.exp(lnpdf.at_centres()), color = "r", label = "%s model without extinction" % title)
	if extincted is not None:
		axes.semilogy(extinctedlnpdf.bins[0].centres(), numpy.exp(extinctedlnpdf.at_centres()), color = "k", label = "%s model with extinction" % title)
	if zerolag_lnpdf is not None:
		axes.semilogy(zerolag_lnpdf.bins[0].centres(), numpy.exp(zerolag_lnpdf.at_centres()), color = "k", linestyle = "--", label = "Observed candidate density")

	axes.grid(which = "major", linestyle = "-", linewidth = 0.2)
	axes.set_title(r"Ranking Statistic Distribution Density Model for %s" % title)
	axes.set_xlabel(r"$\ln \mathcal{L}$")
	axes.set_ylabel(r"$P(\ln \mathcal{L} | \mathrm{%s})$" % which)
	yhi = math.exp(lnpdf[xlo:xhi,].max())
	ylo = math.exp(lnpdf[xlo:xhi,].min())
	if zerolag_lnpdf is not None:
		yhi = max(yhi, math.exp(zerolag_lnpdf[xlo:xhi,].max()))
	ylo = max(yhi * 1e-40, ylo)
	axes.set_ylim((10**math.floor(math.log10(ylo) - .5), 10**math.ceil(math.log10(yhi) + .5)))
	axes.set_xlim((xlo, xhi))
	axes.legend(loc = "lower left", handlelength = 3)
	fig.tight_layout(pad = .8)
	return fig


def plot_likelihood_ratio_ccdf(fapfar, (xlo, xhi), observed_ln_likelihood_ratios = None, is_open_box = False, ln_likelihood_ratio_markers = None):
	assert xlo < xhi

	fig, axes = init_plot((8., 8. / plotutil.golden_ratio))

	x = numpy.linspace(xlo, xhi, int(math.ceil(xhi - xlo)) * 8)
	axes.semilogy(x, fapfar.fap_from_rank(x), color = "k")

	ylo = fapfar.fap_from_rank(xhi)
	ylo = 10**math.floor(math.log10(ylo))
	yhi = 10.

	if observed_ln_likelihood_ratios is not None:
		observed_ln_likelihood_ratios = numpy.array(observed_ln_likelihood_ratios)
		x = observed_ln_likelihood_ratios[:,0]
		y = observed_ln_likelihood_ratios[:,1]
		axes.semilogy(x, y, color = "k", linestyle = "", marker = "+", label = r"Candidates" if is_open_box else r"Candidates (time shifted)")
		axes.legend(loc = "upper right")

	if ln_likelihood_ratio_markers is not None:
		for ln_likelihood_ratio in ln_likelihood_ratio_markers:
			axes.axvline(ln_likelihood_ratio)

	axes.set_xlim((xlo, xhi))
	axes.set_ylim((ylo, yhi))
	axes.grid(which = "major", linestyle = "-", linewidth = 0.2)
	axes.set_title(r"False Alarm Probability vs.\ Log Likelihood Ratio")
	axes.set_xlabel(r"$\ln \mathcal{L}$")
	axes.set_ylabel(r"$P(\mathrm{one\ or\ more\ candidates} \geq \ln \mathcal{L} | \mathrm{noise})$")
	fig.tight_layout(pad = .8)
	return fig


def plot_horizon_distance_vs_time(rankingstat, (tlo, thi), masses = (1.4, 1.4), tref = None):
	fig, axes = init_plot((8., 8. / plotutil.golden_ratio))
	axes.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1800.))
	axes.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5.))
	axes.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(50.))

	yhi = 1.
	for instrument, history in rankingstat.numerator.horizon_history.items():
		x = numpy.array([t for t in history.keys() if tlo <= t < thi])
		y = list(map(history.__getitem__, x))
		if tref is not None:
			x -= float(tref)
		axes.plot(x, y, color = plotutil.colour_from_instruments([instrument]), label = "%s" % instrument)
		try:
			yhi = max(max(y), yhi)
		except ValueError:
			pass
	if tref is not None:
		axes.set_xlabel("Time From GPS %.2f (s)" % float(tref))
	else:
		axes.set_xlim((math.floor(tlo), math.ceil(thi)))
		axes.set_xlabel("GPS Time (s)")
	axes.set_ylim((0., math.ceil(yhi / 10.) * 10.))
	axes.set_ylabel("Horizon Distance (Mpc)")
	axes.set_title(r"Horizon Distance for $%.3g\,\mathrm{M}_{\odot}$--$%.3g\,\mathrm{M}_{\odot}$ vs.\ Time" % masses)
	axes.grid(which = "major", linestyle = "-", linewidth = 0.2)
	axes.legend(loc = "lower left")
	fig.tight_layout(pad = .8)
	return fig

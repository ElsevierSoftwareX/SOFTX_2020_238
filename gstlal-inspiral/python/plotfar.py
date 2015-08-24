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
import numpy
from gstlal import plotutil
from gstlal import far

def init_plot(figsize):
	fig = figure.Figure()
	FigureCanvas(fig)
	fig.set_size_inches(figsize)
	axes = fig.gca()

	return fig, axes


def plot_snr_chi_pdf(coinc_param_distributions, instrument, binnedarray_string, snr_max, dynamic_range_factor = 1e-10, event_snr = None, event_chisq = None):
	key = "%s_snr_chi" % instrument
	if binnedarray_string == "LR":
		binnedarray = getattr(coinc_param_distributions, "injection_pdf")[key]
	else:
		binnedarray = getattr(coinc_param_distributions, binnedarray_string)[key]
	tag = {"background_pdf":"Noise", "injection_pdf":"Signal", "zero_lag_pdf":"Candidates", "LR":"LR"}[binnedarray_string]

	fig, axes = init_plot((8., 8. / plotutil.golden_ratio))
	# the last bin can have a centre at infinity, and its value is
	# always 0 anyway so there's no point in trying to include it
	x = binnedarray.bins[0].centres()[:-1]
	y = binnedarray.bins[1].centres()[:-1]
	z = binnedarray.array[:-1,:-1]
	# FIXME make the LR pdf returned by a method of this class instead?
	if binnedarray_string == "LR":
		denom_binnedarray = coinc_param_distributions.background_pdf[key]
		assert (denom_binnedarray.bins[0].centres()[:-1] == x).all()
		assert (denom_binnedarray.bins[1].centres()[:-1] == y).all()
		z /= denom_binnedarray.array[:-1,:-1]
	if numpy.isnan(z).any():
		warnings.warn("%s PDF contains NaNs" % instrument)
		z = numpy.ma.masked_where(numpy.isnan(z), z)
	if not z.any():
		warnings.warn("%s PDF is 0, skipping" % instrument)
		return None

	# the range of the plots
	xlo, xhi = far.ThincaCoincParamsDistributions.snr_min, snr_max
	ylo, yhi = .0001, 1.

	x = x[binnedarray.bins[xlo:xhi, ylo:yhi][0]]
	y = y[binnedarray.bins[xlo:xhi, ylo:yhi][1]]
	z = z[binnedarray.bins[xlo:xhi, ylo:yhi]]

	# matplotlib's colour bar seems to rely on being able to store the
	# ratio of the lowest and highest value in a double so the range
	# cannot be more than about 300 orders of magnitude.  experiments
	# show it starts to go wrong before that but 250 orders of
	# magnitude seems to be OK
	numpy.clip(z, z.max() * dynamic_range_factor, float("+inf"), out = z)

	mesh = axes.pcolormesh(x, y, z.T, norm = matplotlib.colors.LogNorm(), cmap = "afmhot", shading = "gouraud")
	axes.contour(x, y, z.T, norm = matplotlib.colors.LogNorm(), colors = "k", linewidths = .5)
	if event_snr is not None and event_chisq is not None:
		axes.plot(event_snr, event_chisq / event_snr / event_snr, 'ko', mfc = 'None', mec = 'g', ms = 14, mew=4)
	axes.loglog()
	axes.grid(which = "both")
	#axes.set_xlim((xlo, xhi))
	#axes.set_ylim((ylo, yhi))
	fig.colorbar(mesh, ax = axes)
	axes.set_xlabel(r"$\mathrm{SNR}$")
	axes.set_ylabel(r"$\chi^{2} / \mathrm{SNR}^{2}$")
	if tag.lower() in ("signal",):
		axes.set_title(r"%s %s $P(\chi^{2} / \mathrm{SNR}^{2} | \mathrm{SNR})$" % (instrument, tag))
	elif tag.lower() in ("noise", "candidates"):
		axes.set_title(r"%s %s $P(\mathrm{SNR}, \chi^{2} / \mathrm{SNR}^{2})$" % (instrument, tag))
	elif tag.lower() in ("lr",):
		axes.set_title(r"%s $P(\chi^{2} / \mathrm{SNR}^{2} | \mathrm{SNR}, \mathrm{signal} ) / P(\mathrm{SNR}, \chi^{2} / \mathrm{SNR}^{2} | \mathrm{noise} )$" % instrument)
	else:
		raise ValueError(tag)
	return fig


def plot_rates(coinc_param_distributions, ranking_data = None):
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
	for instrument, category in sorted(coinc_param_distributions.instrument_categories.items()):
		count = coinc_param_distributions.background_rates["instruments"][category,]
		if not count:
			continue
		labels.append("%s\n(%d)" % (instrument, count))
		sizes.append(count)
		colours.append(plotutil.colour_from_instruments((instrument,)))
	axes0.pie(sizes, labels = labels, colors = colours, autopct = "%.3g%%", pctdistance = 0.4, labeldistance = 0.8)
	axes0.set_title("Observed Background Event Counts")

	# projected background counts
	labels = []
	sizes = []
	colours = []
	for instruments in sorted(sorted(instruments) for instruments in coinc_param_distributions.count_above_threshold if instruments is not None):
		count = coinc_param_distributions.background_rates["instruments"][coinc_param_distributions.instrument_categories.category(instruments),]
		if len(instruments) < 2 or not count:
			continue
		labels.append("%s\n(%d)" % (", ".join(instruments), count))
		sizes.append(count)
		colours.append(plotutil.colour_from_instruments(instruments))
	axes1.pie(sizes, labels = labels, colors = colours, autopct = "%.3g%%", pctdistance = 0.4, labeldistance = 0.8)
	axes1.set_title("Projected Background Coincidence Counts")

	# recovered signal distribution
	# FIXME ranking data is not even used, why is this check here?
	if ranking_data is not None:
		labels = []
		sizes = []
		colours = []
		for instruments, fraction in sorted(coinc_param_distributions.Pinstrument_signal.items(), key = lambda (instruments, fraction): sorted(instruments)):
			if len(instruments) < 2 or not fraction:
				continue
			labels.append(", ".join(sorted(instruments)))
			sizes.append(fraction)
			colours.append(plotutil.colour_from_instruments(instruments))
		axes2.pie(sizes, labels = labels, colors = colours, autopct = "%.3g%%", pctdistance = 0.4, labeldistance = 0.8)
		axes2.set_title(r"Projected Recovered Signal Distribution")

	# observed counts
	labels = []
	sizes = []
	colours = []
	for instruments, count in sorted((sorted(instruments), count) for instruments, count in coinc_param_distributions.count_above_threshold.items() if instruments is not None):
		if len(instruments) < 2 or not count:
			continue
		labels.append("%s\n(%d)" % (", ".join(instruments), count))
		sizes.append(count)
		colours.append(plotutil.colour_from_instruments(instruments))
	axes3.pie(sizes, labels = labels, colors = colours, autopct = "%.3g%%", pctdistance = 0.4, labeldistance = 0.8)
	axes3.set_title("Observed Coincidence Counts")
#FIXME: remove when we have a new enough matplotlib on all the reference platforms
	try:
		fig.tight_layout(pad = .8)
		return fig
	except AttributeError:
		return fig

def plot_snr_joint_pdf(coinc_param_distributions, instruments, horizon_distances, max_snr):
	if len(instruments) > 2:
		# FIXME:  figure out how to plot 3D PDFs
		return None
	ignored, binnedarray, ignored = coinc_param_distributions.snr_joint_pdf_cache[(instruments, horizon_distances)]
	instruments = sorted(instruments)
	horizon_distances = dict(horizon_distances)
	fig, axes = init_plot((5, 4))
	x = binnedarray.bins[0].centres()
	y = binnedarray.bins[1].centres()
	z = binnedarray.array
	if numpy.isnan(z).any():
		warnings.warn("%s SNR PDF for %s contains NaNs" % (", ".join(instruments), ", ".join("%s=%g" % instdist for instdist in sorted(horizon_distances.items()))))
		z = numpy.ma.masked_where(numpy.isnan(z), z)

	# the range of the plots
	xlo, xhi = far.ThincaCoincParamsDistributions.snr_min, max_snr

	x = x[binnedarray.bins[xlo:xhi, xlo:xhi][0]]
	y = y[binnedarray.bins[xlo:xhi, xlo:xhi][1]]
	z = z[binnedarray.bins[xlo:xhi, xlo:xhi]]

	# don't try to plot blank PDFs (it upsets older matplotlibs)
	if z.max() == 0.:
		return None

	# these plots only require about 20 orders of magnitude of dynamic
	# range
	numpy.clip(z, z.max() * 1e-20, float("+inf"), out = z)

	# one last check for craziness to make error messages more
	# meaningful
	assert not numpy.isnan(z).any()
	assert not (z <= 0.).any()

	mesh = axes.pcolormesh(x, y, z.T, norm = matplotlib.colors.LogNorm(), cmap = "afmhot", shading = "gouraud")
	axes.contour(x, y, z.T, norm = matplotlib.colors.LogNorm(), colors = "k", linewidths = .5)
	axes.loglog()
	axes.grid(which = "both", linestyle = "-", linewidth = 0.2)
	#axes.set_xlim((xlo, xhi))
	#axes.set_ylim((xlo, xhi))
	fig.colorbar(mesh, ax = axes)
	# co-ordinates are in alphabetical order
	axes.set_xlabel(r"$\mathrm{SNR}_{\mathrm{%s}}$" % instruments[0])
	axes.set_ylabel(r"$\mathrm{SNR}_{\mathrm{%s}}$" % instruments[1])
	axes.set_title(r"$P(%s)$" % ", ".join("\mathrm{SNR}_{\mathrm{%s}}" % instrument for instrument in instruments))

	return fig
	

def plot_likelihood_ratio_pdf(ranking_data, instruments, (xlo, xhi), tag, binnedarray_string = "background_likelihood_pdfs"):
	pdf = getattr(ranking_data, binnedarray_string)[instruments]
	if binnedarray_string == "background_likelihood_pdfs":
		zerolag_pdf = ranking_data.zero_lag_likelihood_pdfs[instruments]
	else:
		zerolag_pdf = None

	fig, axes = init_plot((8., 8. / plotutil.golden_ratio))
	axes.semilogy(pdf.bins[0].centres(), pdf.array, color = "k")
	if zerolag_pdf is not None:
		axes.semilogy(zerolag_pdf.bins[0].centres(), zerolag_pdf.array, color = "k", linestyle = "--")
	axes.grid(which = "major", linestyle = "-", linewidth = 0.2)
	if instruments is None:
		axes.set_title(r"%s Log Likelihood Ratio PDF" % tag)
	else:
		axes.set_title(r"%s %s Log Likelihood Ratio PDF" % (", ".join(sorted(instruments)), tag))
	axes.set_xlabel(r"$\ln \mathcal{L}$")
	axes.set_ylabel(r"$P(\ln \mathcal{L} | \mathrm{%s})$" % tag.lower())
	yhi = pdf[xlo:xhi,].max()
	ylo = pdf[xlo:xhi,].min()
	if zerolag_pdf is not None:
		yhi = max(yhi, zerolag_pdf[xlo:xhi,].max())
		ylo = min(ylo, zerolag_pdf[xlo:xhi,].min())
	ylo = max(yhi * 1e-40, ylo)
	axes.set_ylim((10**math.floor(math.log10(ylo) - .5), 10**math.ceil(math.log10(yhi) + .5)))
	axes.set_xlim((xlo, xhi))
#FIXME: remove when we have a new enough matplotlib on all the reference platforms
	try:
		fig.tight_layout(pad = .8)
		return fig
	except AttributeError:
		return fig

def plot_likelihood_ratio_ccdf(fapfar, (xlo, xhi), tag, zerolag_ln_likelihood_ratios = None, event_likelihood = None):
	ccdf = fapfar.ccdf_interpolator

	fig, axes = init_plot((8., 8. / plotutil.golden_ratio))
	x = numpy.linspace(xlo, xhi, 10000)
	y = numpy.array([far.fap_after_trials(ccdf(likelihood), fapfar.zero_lag_total_count) for likelihood in x])
	axes.semilogy(x, y, color = "k")
	ylo, yhi = 1e-20, 10.
	if zerolag_ln_likelihood_ratios is not None:
		x,y = numpy.array([l[0] for l in zerolag_ln_likelihood_ratios]), numpy.array([l[1] for l in zerolag_ln_likelihood_ratios])
		axes.semilogy(x, y, color = "k", linewidth = 6, alpha = 0.3)
	if event_likelihood is not None:
		axes.axvline(event_likelihood, ylo, yhi)
	axes.set_ylim(ylo, yhi)
	axes.set_xlim((xlo, xhi))
	axes.grid(which = "major", linestyle = "-", linewidth = 0.2)
	axes.set_title(r"%s Log Likelihood Ratio CCDF" % tag)
	axes.set_xlabel(r"$\ln \mathcal{L}$")
	axes.set_ylabel(r"$P(\geq \ln \mathcal{L} | \mathrm{%s})$" % tag.lower())
#FIXME: remove when we have a new enough matplotlib on all the reference platforms
	try:	
		fig.tight_layout(pad = .8)
		return fig
	except AttributeError:
		return fig

def plot_horizon_distance_vs_time(coinc_param_distributions, (tlo,thi), tbins, colours = {"H1": "r", "H2": "b", "L1": "g", "V1": "m"}):
	horizon_history = coinc_param_distributions.horizon_history

	fig, axes = init_plot((8., 8. / plotutil.golden_ratio))
	t = numpy.linspace(tlo, thi, tbins)
	yhi = 0
	for ifo in horizon_history.keys():
		y = numpy.array([horizon_history[ifo][seg] for seg in t])
		axes.plot(t, y, color = colours[ifo], label = '%s' % ifo)
		yhi = max(y.max()+5., yhi)
	axes.set_ylim((0,yhi))
	axes.set_xlim((round(tlo), round(thi)))
	axes.set_ylabel('D_H (Mpc)')
	axes.set_xlabel('GPS Time (s)')
	axes.set_title('Horizon Distance')
	axes.legend(loc = "lower left")
	#FIXME: remove when we have a new enough matplotlib on all the reference platforms
	try:	
		fig.tight_layout(pad = .8)
		return fig
	except AttributeError:
		return fig


# Copyright (C) 2013  Kipp Cannon
# Copyright (C) 2015  Chad Hanna
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

##
# @file
#
# A file that contains the psd plotting module code
#
# ### Review Status
#
##
# @package python.plotpsd
#
# psd plotting module
#


import logging
import math
import matplotlib
from matplotlib import figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import ticker
import numpy
from ligo.lw import lsctables
from gstlal import reference_psd
from gstlal.plots import util as plotutil


def summarize_coinc_xmldoc(coinc_xmldoc):
	coinc_event, = lsctables.CoincTable.get_table(coinc_xmldoc)
	coinc_inspiral, = lsctables.CoincInspiralTable.get_table(coinc_xmldoc)
	offset_vector = lsctables.TimeSlideTable.get_table(coinc_xmldoc).as_dict()[coinc_event.time_slide_id] if coinc_event.time_slide_id is not None else None
	# FIXME:  MBTA uploads are missing process table
	#process, = lsctables.ProcessTable.get_table(coinc_xmldoc)
	sngl_inspirals = dict((row.ifo, row) for row in lsctables.SnglInspiralTable.get_table(coinc_xmldoc))

	mass1 = sngl_inspirals.values()[0].mass1
	mass2 = sngl_inspirals.values()[0].mass2
	if mass1 < mass2:
		mass1, mass2 = mass2, mass1
	end_time = coinc_inspiral.end
	on_instruments = coinc_inspiral.ifos
	logging.info("%g Msun -- %g Msun event in %s at %.2f GPS" % (mass1, mass2, ", ".join(sorted(sngl_inspirals)), float(end_time)))

	return sngl_inspirals, mass1, mass2, end_time, on_instruments


def axes_plot_cummulative_snr(axes, psds, coinc_xmldoc):
	sngl_inspirals, mass1, mass2, end_time, on_instruments = summarize_coinc_xmldoc(coinc_xmldoc)

	axes.grid(which = "both", linestyle = "-", linewidth = 0.2)
	axes.minorticks_on()

	for instrument, sngl_inspiral in sngl_inspirals.items():
		logging.info("found %s event with SNR %g" % (instrument, sngl_inspirals[instrument].snr))

		if instrument not in psds:
			logging.info("no PSD for %s" % instrument)
			continue
		psd = psds[instrument]
		if psd is None:
			logging.info("no PSD for %s" % instrument)
			continue
		psd_data = psd.data.data
		f = psd.f0 + numpy.arange(len(psd_data)) * psd.deltaF
		logging.info("found PSD for %s spanning [%g Hz, %g Hz]" % (instrument, f[0], f[-1]))

		# FIXME: horizon distance stopped at 0.9 max frequency due
		# to low pass filter messing up the end of the PSD.  if we
		# could figure out the frequency bounds and delta F we
		# could move this out of the loop for some speed
		horizon_distance = reference_psd.HorizonDistance(10., 0.9 * f[-1], psd.deltaF, mass1, mass2)

		# generate inspiral spectrum and clip PSD to its frequency
		# range
		inspiral_spectrum_x, inspiral_spectrum_y = horizon_distance(psd, sngl_inspiral.snr)[1]
		lo = int(round((inspiral_spectrum_x[0] - psd.f0) / psd.deltaF))
		hi = int(round((inspiral_spectrum_x[-1] - psd.f0) / psd.deltaF)) + 1
		f = f[lo:hi]
		psd_data = psd_data[lo:hi]

		# plot
		snr2 = (inspiral_spectrum_y / psd_data).cumsum() * psd.deltaF
		axes.semilogx(f, snr2**.5, color = plotutil.colour_from_instruments([instrument]), alpha = 0.8, linestyle = "-", label = "%s SNR = %.3g" % (instrument, sngl_inspiral.snr))

	axes.set_ylim([0., axes.get_ylim()[1]])

	axes.set_title(r"Cumulative SNRs for $%.3g\,\mathrm{M}_{\odot}$--$%.3g\,\mathrm{M}_{\odot}$ Merger Candidate at %.2f GPS" % (mass1, mass2, float(end_time)))
	axes.set_xlabel(r"Frequency (Hz)")
	axes.set_ylabel(r"Cumulative SNR")
	axes.legend(loc = "upper left")


def latex_horizon_distance(Mpc):
	if Mpc >= 256.:
		# :-O
		return "%s Gpc" % plotutil.latexnumber("%.4g" % (Mpc * 1e-3))
	elif Mpc >= 0.25:
		# :-)
		return "%s Mpc" % plotutil.latexnumber("%.4g" % Mpc)
	elif Mpc >= 2**-12:
		# :-(
		return "%s kpc" % plotutil.latexnumber("%.4g" % (Mpc * 1e3))
	else:
		# X-P
		return "%s pc" % plotutil.latexnumber("%.4g" % (Mpc * 1e6))


def axes_plot_psds(axes, psds, coinc_xmldoc = None):
	"""
	Places a PSD plot into a matplotlib Axes object.

	@param axes An Axes object into which the plot will be placed.

	@param psds A dictionary of PSDs as REAL8FrequencySeries keyed by
	instrument

	@param coinc_xmldoc An XML document containing a single event with all
	of the metadata as would be uploaded to gracedb.  This is optional.
	"""

	if coinc_xmldoc is not None:
		sngl_inspirals, mass1, mass2, end_time, on_instruments = summarize_coinc_xmldoc(coinc_xmldoc)
	else:
		# Use the cannonical BNS binary for horizon distance if an
		# event wasn't given
		sngl_inspirals = {}
		mass1, mass2, end_time = 1.4, 1.4, None
		on_instruments = set(psds)

	axes.grid(which = "both", linestyle = "-", linewidth = 0.2)
	axes.minorticks_on()

	min_psds, max_psds = [], []
	min_fs, max_fs = [], []
	for instrument, psd in sorted(psds.items()):
		if psd is None:
			continue
		psd_data = psd.data.data
		f = psd.f0 + numpy.arange(len(psd_data)) * psd.deltaF
		logging.info("found PSD for %s spanning [%g Hz, %g Hz]" % (instrument, f[0], f[-1]))
		min_fs.append(f[0])
		max_fs.append(f[-1])
		# FIXME: horizon distance stopped at 0.9 max frequency due
		# to low pass filter messing up the end of the PSD.  if we
		# could figure out the frequency bounds and delta F we
		# could move this out of the loop for some speed
		horizon_distance = reference_psd.HorizonDistance(10., 0.9 * f[-1], psd.deltaF, mass1, mass2)
		if instrument in on_instruments:
			alpha = 0.8
			linestyle = "-"
			label = "%s (%s Horizon)" % (instrument, latex_horizon_distance(horizon_distance(psd, 8.)[0]))
		else:
			alpha = 0.6
			linestyle = ":"
			label = "%s (Off, Last Seen With %s Horizon)" % (instrument, latex_horizon_distance(horizon_distance(psd, 8.)[0]))
		axes.loglog(f, psd_data, color = plotutil.colour_from_instruments([instrument]), alpha = alpha, linestyle = linestyle, label = label)
		if instrument in sngl_inspirals:
			logging.info("found %s event with SNR %g" % (instrument, sngl_inspirals[instrument].snr))
			inspiral_spectrum = horizon_distance(psd, sngl_inspirals[instrument].snr)[1]
			axes.loglog(inspiral_spectrum[0], inspiral_spectrum[1], color = plotutil.colour_from_instruments([instrument]), dashes = (5, 2), alpha = 0.8, label = "SNR = %.3g" % sngl_inspirals[instrument].snr)
		# record the minimum from within the rage 10 Hz -- 900 Hz
		min_psds.append(psd_data[int((10.0 - psd.f0) / psd.deltaF) : int((900 - psd.f0) / psd.deltaF)].min())
		# record the maximum from within the rage 1 Hz -- 900 Hz
		max_psds.append(psd_data[int((1.0 - psd.f0) / psd.deltaF) : int((900 - psd.f0) / psd.deltaF)].max())

	if min_fs:
		axes.set_xlim((6.0, max(max_fs)))
	else:
		axes.set_xlim((6.0, 3000.0))
	if min_psds:
		axes.set_ylim((10.**math.floor(math.log10(min(min_psds) / 3.)), 10.**math.ceil(math.log10(max(max_psds)))))

	# FIXME:  I don't understand how these work
	axes.yaxis.set_major_locator(ticker.LogLocator(10., subs = (1.0,)))
	axes.yaxis.set_minor_locator(ticker.LogLocator(10., subs = (0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)))

	title = r"Strain Noise Spectral Density for $%.3g\,\mathrm{M}_{\odot}$--$%.3g\,\mathrm{M}_{\odot}$ Merger Candidate" % (mass1, mass2)
	if end_time is not None:
		title += r" at %.2f GPS" % float(end_time)
	axes.set_title(title)
	axes.set_xlabel(r"Frequency (Hz)")
	axes.set_ylabel(r"Spectral Density ($\mathrm{strain}^2 / \mathrm{Hz}$)")
	axes.legend(loc = "upper right")


def plot_psds(psds, coinc_xmldoc = None, plot_width = 640):
	"""
	Produces a matplotlib figure of PSDs.

	@param psds A dictionary of PSDs as REAL8FrequencySeries keyed by
	instrument

	@param coinc_xmldoc An XML document containing a single event with all
	of the metadata as would be uploaded to gracedb.  This is optional.

	@param plot_width How wide to make the figure object in pixels
	(ignored if axes is provided).
	"""
	fig = figure.Figure()
	FigureCanvas(fig)
	fig.set_size_inches(plot_width / float(fig.get_dpi()), int(round(plot_width / plotutil.golden_ratio)) / float(fig.get_dpi()))
	axes_plot_psds(fig.gca(), psds, coinc_xmldoc = coinc_xmldoc)
	fig.tight_layout(pad = .8)
	return fig


def plot_cumulative_snrs(psds, coinc_xmldoc, plot_width = 640):
	"""
	Produces a matplotlib figure of cumulative SNRs.

	@param psds A dictionary of PSDs as REAL8FrequencySeries keyed by
	instrument

	@param coinc_xmldoc An XML document containing a single event with all
	of the metadata as would be uploaded to gracedb.

	@param plot_width How wide to make the figure object in pixels
	(ignored if axes is provided).
	"""
	fig = figure.Figure()
	FigureCanvas(fig)
	fig.set_size_inches(plot_width / float(fig.get_dpi()), int(round(plot_width / plotutil.golden_ratio)) / float(fig.get_dpi()))
	axes_plot_cummulative_snr(fig.gca(), psds, coinc_xmldoc)
	fig.tight_layout(pad = .8)
	return fig

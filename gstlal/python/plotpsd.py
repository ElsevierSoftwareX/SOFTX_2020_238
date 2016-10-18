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
import numpy
from gstlal import plotutil
from glue.ligolw import lsctables
from gstlal.reference_psd import horizon_distance


def plot_psds(psds, coinc_xmldoc = None, plot_width = 640):
	"""!
	Produces a matplotlib figure of PSDs. 

	@param psds A dictionary of PSDs as REAL8FrequencySeries keyed by
	instrument

	@param coinc_xmldoc An XML document containing a single event with all
	of the metadata as would be uploaded to gracedb.  This is optional.

	@param plot_width How wide to make the plot in pixels
	"""

	if coinc_xmldoc is not None:
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
	else:
		# Use the cannonical BNS binary for horizon distance if an event wasn't given
		sngl_inspirals = {}
		mass1, mass2, end_time = 1.4, 1.4, None
		on_instruments = set(psds)

	fig = figure.Figure()
	FigureCanvas(fig)
	fig.set_size_inches(plot_width / float(fig.get_dpi()), int(round(plot_width / plotutil.golden_ratio)) / float(fig.get_dpi()))
	axes = fig.gca()
	axes.grid(which = "both", linestyle = "-", linewidth = 0.2)

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
		# FIXME: Horizon distance stopped at 0.9 max frequency due
		# to low pass filter messing up the end of the PSD
		if instrument in on_instruments:
			alpha = 0.8
			linestyle = "-"
			label = "%s (%.4g Mpc Horizon)" % (instrument, horizon_distance(psd, mass1, mass2, 8, 10, f_max = 0.9 * max(f)))
		else:
			alpha = 0.6
			linestyle = ":"
			label = "%s (off, last seen at %.4g Mpc Horizon)" % (instrument, horizon_distance(psd, mass1, mass2, 8, 10, f_max = 0.9 * max(f)))
		axes.loglog(f, psd_data, color = plotutil.colour_from_instruments([instrument]), alpha = alpha, linestyle = linestyle, label = label)
		if instrument in sngl_inspirals:
			logging.info("found %s event with SNR %g" % (instrument, sngl_inspirals[instrument].snr))
			inspiral_spectrum = [None, None]
			horizon_distance(psd, mass1, mass2, sngl_inspirals[instrument].snr, 10, inspiral_spectrum = inspiral_spectrum)
			axes.loglog(inspiral_spectrum[0], inspiral_spectrum[1], color = plotutil.colour_from_instruments([instrument]), dashes = (5, 2), alpha = 0.8, label = "SNR = %.3g" % sngl_inspirals[instrument].snr)
		# record the minimum from within the rage 10 Hz -- 1 kHz
		min_psds.append(psd_data[int((10.0 - psd.f0) / psd.deltaF) : int((1000 - psd.f0) / psd.deltaF)].min())
		# record the maximum from within the rage 1 Hz -- 1 kHz
		max_psds.append(psd_data[int((1.0 - psd.f0) / psd.deltaF) : int((1000 - psd.f0) / psd.deltaF)].max())

	if min_fs:
		axes.set_xlim((6.0, max(max_fs)))
	else:
		axes.set_xlim((6.0, 3000.0))
	if min_psds:
		axes.set_ylim((10**math.floor(math.log10(min(min_psds))), 10**math.ceil(math.log10(max(max_psds)))))
	title = r"Strain Noise Spectral Density for $%.3g\,\mathrm{M}_{\odot}$--$%.3g\,\mathrm{M}_{\odot}$ Merger Candidate" % (mass1, mass2)
	if end_time is not None:
		title += r" at %.2f GPS" % float(end_time)
	axes.set_title(title)
	axes.set_xlabel(r"Frequency (Hz)")
	axes.set_ylabel(r"Spectral Density ($\mathrm{strain}^2 / \mathrm{Hz}$)")
	axes.legend(loc = "upper right")
	fig.tight_layout(pad = .8)

	return fig

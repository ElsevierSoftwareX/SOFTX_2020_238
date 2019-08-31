"""
Plotting SNR from LIGO light-weight XML file
"""
import sys
import numpy

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot

from gstlal import plotutil
from gstlal import svd_bank_snr

def axes_add_snr(axes, snrdict, center = None, span = None):
	"""

	Add snr time series to the plot

	Args:
		axes (object): matplotlib axes
		snrdict (dict): dictionary of snrs
		center (float): the center gpstime of the plot
		span (float): seconds to span around center

	"""
	col = 0
	for instrument, SNRs in snrdict.items():
		if len(SNRs) > 1:
			raise ValueError("Too many snr time series in snrdict.")

		data, center_gps_time, relative_gps_time = locate_center(SNRs[0], center, span)
		if numpy.iscomplexobj(data):
			axes[col].plot(relative_gps_time, data.real, color = plotutil.colour_from_instruments([instrument]), linestyle = "-", label = "%s_Real" % instrument, linewidth = 1)
			axes[col].plot(relative_gps_time, data.imag, color = plotutil.colour_from_instruments([instrument]), linestyle = "--", label = "%s_Imaginary" % instrument, linewidth = 1)
		else:
			axes[col].plot(relative_gps_time, data, color = plotutil.colour_from_instruments([instrument]), label = "%s" %instrument, linewidth = 1)
		axes[col].set_xlabel("GPS Time Since %f" % center_gps_time)
		axes[col].set_ylabel("SNR")
		axes[col].legend(loc = "upper left")
		axes[col].tick_params(labelbottom = True)
		col += 1

def plot_snr(SNR_dict, width = 8, center = None, span = None, verbose = False):
	"""

	Plot snr time series from snrdicts

	Args:
		SNR_dict: A dictionary containing (instrument, LAL series) pairs
		width (int): width of the output figure in inch
		center (float): the center gpstime of the plot
		span (float): seconds to span around center
		verbose (bool): be verbose

	Return:

		fig (object): matplotlib figure
	"""

	nrows = len(SNR_dict.keys())
	ncols = 1

	fig, axes = pyplot.subplots(nrows = nrows, ncols = ncols, sharex = True)
	if nrows == 1:
		axes = [axes]
	axes_add_snr(axes, SNR_dict, center = center, span = span)
	fig.set_size_inches(width, int(round(width/plotutil.golden_ratio)))
	fig.tight_layout(pad = 0.8)

	return fig

def locate_center(snr_series, center, span):
	"""

	locate the snr_series at nearest gpstime with certain span

	Args:
		snr_series (lal.series): A (snr) lal.series
		center (float): gps time
		span (float): seconds to span around center

	Return:
		data (numpy.array <type 'float'>): snr_series.data.data [center-span, center+span]
		gps_time (float): nearest gps_time at center
		relative_gps_time (numpy.array <type 'float'>): gpstime relative to gps_time

	"""

	start = None
	mid = 0
	end = None
	if center and span:
		start = find_nearest(snr_series, center - span)[1]
		mid = find_nearest(snr_series, center)[1]
		end = find_nearest(snr_series, center + span)[1]

	gps_time = (snr_series.epoch.gpsSeconds + snr_series.epoch.gpsNanoSeconds * 1.e-9 + (numpy.arange(snr_series.data.length) * snr_series.deltaT))
	relative_gps_time = (gps_time - gps_time[mid])[start:end]
	data = snr_series.data.data[start:end]

	return data, gps_time[mid], relative_gps_time

def find_nearest(snr_series, time):
	"""

	Find the nearest gps time available from the time series

	Args:
		snr_series (lal.series): A (snr) lal-series
		time (float): requested gpstime

	Return:
		(snr[gpstime_index], gpstime_index)

	"""

	gps_start = snr_series.epoch.gpsSeconds + snr_series.epoch.gpsNanoSeconds * 10.**-9
	gps = gps_start + numpy.arange(snr_series.data.length) * snr_series.deltaT
	if time - gps[0] >= 0 and time - gps[-1] <= 0:
		index = abs(gps - time).argmin()
	else:
		raise ValueError("Invalid choice of center time %f." % time)
	return (snr_series.data.data[index], index)

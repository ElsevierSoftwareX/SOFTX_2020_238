"""
Plotting SNR from LIGO light-weight XML file
"""
import sys
import numpy

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
        "font.size": 10.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 10.0,
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "legend.fontsize": 10.0,
        "figure.dpi": 100,
        "savefig.dpi": 100,
        "text.usetex": True
})
from matplotlib import pyplot

from gstlal import plotutil
from gstlal import svd_bank_snr

def plot_snr(SNR_dict, width=8, center=None, span=None, verbose=False):
	"""Plot snr time series from SNR_dicts.

	Args:
		SNR_dict: A dictionary containing (instrument, LAL series) pairs.
		width (int): The width of the output figure in inch.
		center (float): The center gpstime of the plot.
		span (float): Seconds to span around center.
		verbose (bool, default=False): Be verbose.

	Return:
		fig (object): Matplotlib figure.
	"""

	nrows = len(SNR_dict.keys())
	ncols = 1

	fig, axes = pyplot.subplots(nrows = nrows, ncols = ncols, sharex = True)
	if nrows == 1:
		axes = [axes]
	_axes_add_snr(axes, SNR_dict, center = center, span = span)
	fig.set_size_inches(width, int(round(width/plotutil.golden_ratio)))
	fig.tight_layout(pad = 0.8)

	return fig

def plot_snr_with_ac(SNR_dict, autocorrelation_dict, width=8, ref_trigger_time=None, verbose=False):
	"""Plot real part of the snr time series together with template autocorrelation.

	Args:
		SNR_dict (dict): A dictionary containing (instrument, LAL series) pairs.
		autocorrelation_dict (dict): A dictionary containing (instrument, numpy.array) pairs.
		width (int): The width of the output figure in inch.
		ref_trigger_time (float, default=None): The reference trigger time which it used to find the actual trigger time based on the time series.
		verbose (bool, default=False): Be verbose.

	Return:
		fig (object): Matplotlib figure.
	"""

	all_instruments = set(SNR_dict) & set(autocorrelation_dict)
	nrows = len(all_instruments)
	ncols = 1

	fig, axes = pyplot.subplots(nrows = nrows, ncols = ncols, sharex = False)
	if nrows == 1:
		axes = [axes]
	_axes_add_snr_with_ac(axes, SNR_dict, autocorrelation_dict, ref_trigger_time = ref_trigger_time)
	fig.set_size_inches(width, int(round(width/plotutil.golden_ratio)))
	fig.tight_layout(pad = 0.8)

	return fig

def _axes_add_snr(axes, SNR_dict, center=None, span=None):
	"""Add snr time series to the plot

	Args:
		axes (object): Matplotlib axes.
		SNR_dict (dict): A dictionary of only one snr.
		center (float): The center gpstime of the plot.
		span (float): Seconds to span around center.
	"""
	col = 0
	for instrument, SNRs in SNR_dict.items():
		if len(SNRs) > 1:
			raise ValueError("Too many SNR time series in SNR_dict.")

		data, center_gps_time, relative_gps_time = _trim_by_time(SNRs[0], center = center, span = span)
		if numpy.iscomplexobj(data):
			axes[col].plot(relative_gps_time, data.real, color = plotutil.colour_from_instruments([instrument]), linestyle = "-", label = r"$\mathrm{Real}: \ \rho(t)$", linewidth = 1)
			axes[col].plot(relative_gps_time, data.imag, color = plotutil.colour_from_instruments([instrument]), linestyle = "--", label = r"$\mathrm{Imaginary}: \ \rho(t)$", linewidth = 1)
		else:
			axes[col].plot(relative_gps_time, data, color = plotutil.colour_from_instruments([instrument]), label = r"$\|\rho(t)\|$" , linewidth = 1)
		axes[col].set_xlabel(r"$\mathrm{GPS \ Time \ Since \ %f}$" % center_gps_time)
		axes[col].set_ylabel(r"$\mathrm{%s} \ \rho(t)$" % instrument)
		axes[col].legend(loc = "upper left")
		axes[col].grid(True)
		axes[col].tick_params(labelbottom = True)
		col += 1

def _axes_add_snr_with_ac(axes, SNR_dict, autocorrelation_dict, ref_trigger_time):
	"""Add snr time series and template autocorrelation to the plot

	Args:
		axes (object): Matplotlib axes.
		SNR_dict (dict): A dictionary of only one snr.
		autocorrelation_dict (dict): A dictionary containing (instrument, numpy.array) pairs.
		ref_trigger_time (float, default=None): The reference trigger time which it used to find the actual trigger time based on the time series.
	"""
	assert len(SNR_dict) == len(autocorrelation_dict), "Number of instruments of SNRs and template autocorrelations does not match."

	all_instruments = set(SNR_dict) & set(autocorrelation_dict)
	col = 0

	for instrument in all_instruments:
		if len(SNR_dict[instrument]) > 1 or len(autocorrelation_dict[instrument]) > 1:
			raise ValueError("Too many SNR time series or template autocorrelation in the dictionary.")

		complex_snr, trigger_time = _find_trigger_time(SNR_dict[instrument][0], ref_trigger_time = ref_trigger_time)
		SNR, center_gps_time, relative_gps_time = _trim_by_samples(SNR_dict[instrument][0], center = trigger_time, samples = len(autocorrelation_dict[instrument][0]))
		phase = numpy.angle(complex_snr)

		# pick only the real part of SNR and autocorrelation, then scale the autocorrelation
		real_SNR = (SNR * numpy.exp(-1.j * phase)).real
		scaled_autocorrelation = autocorrelation_dict[instrument][0].real * max(real_SNR)

		# now do ploting
		axes[col].plot(relative_gps_time, real_SNR, color = plotutil.colour_from_instruments([instrument]), linestyle = "-", label = r"$\mathrm{Measured}:\rho(t)$" , linewidth = 1)
		axes[col].plot(relative_gps_time, scaled_autocorrelation, color = "black", linestyle = "--", label = r"$\mathrm{Scaled \ Autocorrelation}$" , linewidth = 1)

		axes[col].set_xlabel(r"$\mathrm{GPS \ Time \ Since \ %f}$" % center_gps_time)
		axes[col].set_ylabel(r"$\mathrm{%s} \ \rho(t)$" % instrument)
		axes[col].legend(loc = "upper left")
		axes[col].grid(True)
		axes[col].tick_params(labelbottom = True)
		col += 1


def _trim_by_time(snr_series, center, span):
	"""Trim the snr_series to the nearest center with certain span, the original time series is remained intact.

	Args:
		snr_series (lal.series): A SNR lal.series.
		center (float): The center gpstime of the trimmed data.
		span (float): Seconds to span around center.

	Return:
		data (numpy.array <type 'float'>): The snr_series.data.data[center-span, center+span].
		gps_time (float): The nearest gps_time at center.
		relative_gps_time (numpy.array <type 'float'>): The gpstime relative to gps_time.
	"""
	start = _find_nearest(snr_series, center - span)[1]
	mid = _find_nearest(snr_series, center)[1]
	end = _find_nearest(snr_series, center + span)[1]

	gps_time = (snr_series.epoch.gpsSeconds + snr_series.epoch.gpsNanoSeconds * 1.e-9 + (numpy.arange(snr_series.data.length) * snr_series.deltaT))

	return snr_series.data.data[start:end+1], gps_time[mid], (gps_time - gps_time[mid])[start:end+1]

def _trim_by_samples(snr_series, center, samples=351):
	"""Trim the snr_series to the nearest center with given samples, the original time series is remained intact.

	Args:
		snr_series (lal.series): A SNR LAL series.
		center (float): The center gpstime of the trimmed data.
		samples (int, default=351): The samples of the output data.

	Return:
		data (numpy.array <type 'float'>): The snr_series.data.data[center-span, center+span].
		gps_time (float): The nearest gps_time at center.
		relative_gps_time (numpy.array <type 'float'>): The gpstime relative to gps_time.
	"""
	assert samples % 2 == 1, "An odd number of samples is expected."

	left_right_samples = (samples - 1) // 2

	gps_time = (snr_series.epoch.gpsSeconds + snr_series.epoch.gpsNanoSeconds * 1.e-9 + (numpy.arange(snr_series.data.length) * snr_series.deltaT))

	mid = _find_nearest(snr_series, center)[1]
	start = mid - left_right_samples if mid - left_right_samples > 0 else 0
	end = mid + left_right_samples if mid + left_right_samples < len(gps_time) - 1 else len(gps_time) - 1

	return snr_series.data.data[start:end+1], gps_time[mid], (gps_time - gps_time[mid])[start:end+1]

def _find_trigger_time(snr_series, ref_trigger_time=None):
	"""Find the nearest gps time available from the time series

	Args:
		snr_series (lal.series): A SNR LAL series.
		time (float): Requested gpstime.

	Return:
		(SNR, index): A tuple of SNR and index nearest to time.
	"""
	# FIXME: perhaps as an options?
	span = 1.
	search_left_right_samples = int(round(span / snr_series.deltaT))

	gps_time = (snr_series.epoch.gpsSeconds + snr_series.epoch.gpsNanoSeconds * 1.e-9 + (numpy.arange(snr_series.data.length) * snr_series.deltaT))
	mid = _find_nearest(snr_series, ref_trigger_time)[1] if ref_trigger_time is not None else len(gps_time) // 2
	end = len(gps_time) - 1 if mid + search_left_right_samples > len(gps_time) - 1 else mid + search_left_right_samples
	start = 0 if mid - search_left_right_samples < 0 else mid - search_left_right_samples

	index = numpy.argmax(numpy.abs(snr_series.data.data[start:end+1]))

	return (snr_series.data.data[start:end+1][index], gps_time[start:end+1][index])

def _find_nearest(snr_series, time):
	"""Find the nearest gps time available from the time series

	Args:
		snr_series (lal.series): A SNR LAL series.
		time (float): Requested gpstime.

	Return:
		(SNR, index): A tuple of SNR and index nearest to time.
	"""
	gps_start = snr_series.epoch.gpsSeconds + snr_series.epoch.gpsNanoSeconds * 10.**-9
	gps = gps_start + numpy.arange(snr_series.data.length) * snr_series.deltaT
	if time - gps[0] >= 0 and time - gps[-1] <= 0:
		index = abs(gps - time).argmin()
	else:
		raise ValueError("Invalid choice of center time %f." % time)
	return (snr_series.data.data[index], index)

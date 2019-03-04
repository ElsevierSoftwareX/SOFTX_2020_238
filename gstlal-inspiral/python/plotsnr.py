"""
Plotting SNR from LIGO light-weight XML file
"""
import sys
import numpy

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot

from gstlal import svd_bank_snr

def plot_snr(filename, output = None, center = None, span = None, verbose = False):
	xmldoc = svd_bank_snr.read_url(filename, contenthandler = svd_bank_snr.SNRContentHandler, verbose = verbose)
	SNR_dict = svd_bank_snr.read_xmldoc(xmldoc)
	start = None
	mid = 0
	end = None

	if verbose:
		sys.stderr.write("Ploting SNR ...\n")

	for instrument, SNR in SNR_dict.items():
		if verbose and center and span:
			sys.stderr.write("Locating SNR at GPSTime: %f spanning %f s\n" % (center, span))
		if center and span:
			start = find_nearest(SNR, center - span)[1]
			mid = find_nearest(SNR, center)[1]
			end = find_nearest(SNR, center + span)[1]

		gps_time = (SNR.epoch.gpsSeconds + SNR.epoch.gpsNanoSeconds * 10.**-9 + (numpy.arange(SNR.data.length) * SNR.deltaT))
		relative_gps_time = (gps_time - gps_time[mid])[start:end]
		data = SNR.data.data[start:end]

		fig, ax = pyplot.subplots(nrows = 1, ncols = 1, figsize = [15,6])
		ax.plot(relative_gps_time, data)
		ax.set_xlabel("GPS Time Since %f" % gps_time[mid])
		ax.set_ylabel("SNR")

		pyplot.tight_layout()
		if output is None:
			output = "SNR_%s_since_%d.svg" %(SNR.name, gps_time[mid])
			if verbose:
				sys.stderr.write("%s --> Done\n" % output)
			fig.savefig(output)
		else:
			if verbose:
				sys.stderr.write("%s --> Done\n" % output)
			fig.savefig(output)
		pyplot.close()

def find_nearest(snr_series, time):
	gps_start = snr_series.epoch.gpsSeconds + snr_series.epoch.gpsNanoSeconds * 10.**-9
	gps = gps_start + numpy.arange(snr_series.data.length) * snr_series.deltaT
	if time - gps[0] > 0 and time - gps[-1] <0:
		index = abs(gps - time).argmin()
	else:
		raise ValueError("Invalid choice of center time %f." % time)
	return (snr_series.data.data[index], index)

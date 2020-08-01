##############################################################################
# Script to plot the rate vs threshold curve and efficiency curve, like how
# lalapps_string_final has done.


from __future__ import print_function


import math
import matplotlib
from matplotlib import figure
from matplotlib import patches
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
import numpy
from scipy import interpolate
from scipy import optimize
import sqlite3
import sys

from optparse import OptionParser

import lal
from lal import rate
from lal.utils import CacheEntry

from lalburst import SimBurstUtils
from lalburst import SnglBurstUtils
from lalburst import stringutils

from ligo import segments
from ligo.lw import dbtables
from ligo.lw.utils import process as ligolw_process



##############################################################################
# TO DO
#
# 1. efficiency: enable overriding rate_vs_thresh to enable comparisons
# 2. enable open-box results
##############################################################################


#
# =============================================================================
#
#                                 Command Line
#
# =============================================================================
#


def parse_command_line():
	parser = OptionParser()
	parser.add_option("--cal-uncertainty", metavar = "fraction", type = "float", help = "Set the fractional uncertainty in amplitude due to calibration uncertainty (eg. 0.08).  This option is required, use 0 to disable calibration uncertainty.")
	parser.add_option("--detection-threshold", metavar = "Hz", type = "float", help = "Override the false alarm rate threshold.  Only injection files will be processed, and the efficiency curve measured.")
	parser.add_option("--rankingstatpdf-file", metavar = "filename", help = "Set the name of the xml file containing the marginalized likelihood.")
	parser.add_option("-c", "--input-cache", metavar = "filename", help = "Process the files named in this LAL cache. See lalapps_path2cache for information on how to produce a LAL cache file. The input (& output) of the FAPFAR jobs should be OK.")
	parser.add_option("--likelihood-cache", metavar = "filename", help = "Also load the likelihood ratio data files listsed in this LAL cache. This is used to obtain the segments that were actually analyzed.")
	parser.add_option("--tmp-space", metavar = "dir", help = "Set the name of the tmp space if working with sqlite.")
	parser.add_option("--vetoes-name", metavar = "name", help = "Set the name of the segment lists to use as vetoes (default = do not apply vetoes).")
	parser.add_option("--verbose", "-v", action = "store_true", help = "Be verbose.")

	options, filenames = parser.parse_args()

	if options.cal_uncertainty is None:
		raise ValueError("must set --cal-uncertainty (use 0 to ignore calibration uncertainty)")

	if options.input_cache:
		filenames += [CacheEntry(line).path for line in open(options.input_cache)]
	if not filenames:
		raise ValueError("no candidate databases specified")

	options.likelihood_filenames = []
	if options.likelihood_cache is not None:
		options.likelihood_filenames += [CacheEntry(line).path for line in open(options.likelihood_cache)]
	if not options.likelihood_filenames:
		raise ValueError("no ranking statistic likelihood data files specified")

	return options, filenames


golden_ratio = 0.5 * (1.0 + math.sqrt(5.0))

def create_plot(x_label = None, y_label = None, width = 165.0, aspect = golden_ratio):
	fig = figure.Figure()
	fig.set_size_inches(width / 25.4, width / 25.4 / aspect)
	axes = fig.add_axes((0.1, 0.12, .875, .80))
	axes.grid(True)
	if x_label is not None:
		axes.set_xlabel(x_label)
	if y_label is not None:
		axes.set_ylabel(y_label)
	return fig, axes


#
# =============================================================================
#
#                           Rate vs. Threshold Plots
#
# =============================================================================
#


def sigma_region(mean, nsigma):
	return numpy.concatenate((mean - nsigma * numpy.sqrt(mean), (mean + nsigma * numpy.sqrt(mean))[::-1]))

def create_rate_vs_lnL_plot(axes, zerolag_stats, fapfar):
	axes.semilogy()

	#
	# plot limits and expected counts
	#

	def expected_count(lr):
		return fapfar.far_from_rank(lr) * fapfar.livetime

	xlim = max(zerolag_stats.min(), fapfar.minrank), max(2. * math.ceil(zerolag_stats.max() / 2.), 30.)
	#xlim = -10, max(2. * math.ceil(zerolag_stats.max() / 2.), 30.)
	ylim = 5e-7, 10.**math.ceil(math.log10(expected_count(xlim[0])))
	#xlim = -0.1, max(2. * math.ceil(zerolag_stats.max() / 2.), 10.)
	#ylim = 1e-3, 1e5

	#
	# expected count curve
	#

	expected_count_x = numpy.linspace(xlim[0], xlim[1], 10000)
	expected_count_y = map(expected_count, expected_count_x)
	line1, = axes.plot(expected_count_x, expected_count_y, 'k--', linewidth = 1)

	#
	# error bands
	#

	expected_count_x = numpy.concatenate((expected_count_x, expected_count_x[::-1]))
	line2, = axes.fill(expected_count_x, sigma_region(expected_count_y, 3.0).clip(*ylim), alpha = 0.25, facecolor = [0.75, 0.75, 0.75])
	line3, = axes.fill(expected_count_x, sigma_region(expected_count_y, 2.0).clip(*ylim), alpha = 0.25, facecolor = [0.5, 0.5, 0.5])
	line4, = axes.fill(expected_count_x, sigma_region(expected_count_y, 1.0).clip(*ylim), alpha = 0.25, facecolor = [0.25, 0.25, 0.25])

	#
	# zero-lag
	#

	if zerolag_stats is not None:
		N = numpy.arange(1., len(zerolag_stats) + 1., dtype = "double")
		line5, = axes.plot(zerolag_stats.repeat(2)[1:], N.repeat(2)[:-1], 'k', linewidth = 2)

	#
	# legend
	#

	axes.legend((line5, line1, line4, line3, line2), (r"Observed (time shifted)", r"Noise Model, $\langle N \rangle$", r"$\pm\sqrt{\langle N \rangle}$", r"$\pm 2\sqrt{\langle N \rangle}$", r"$\pm 3\sqrt{\langle N \rangle}$"), loc = "upper right")

	#
	# adjust bounds of plot
	#

	axes.set_xlim(xlim)
	axes.set_ylim(ylim)


class RateVsThreshold(object):
	def __init__(self, fapfar):
		self.fapfar = fapfar
		self.background_ln_likelihood_ratio = []
		self.background_far = []
		self.background_fap = []
		self.zerolag_ln_likelihood_ratio = []
		self.zerolag_far = []
		self.zerolag_fap = []

	def add_contents(self, contents):
		for ln_likelihood_ratio, far, fap, is_background in connection.cursor().execute("""
SELECT
	coinc_event.likelihood,
	coinc_burst.false_alarm_rate,
	coinc_burst.false_alarm_probability,
	EXISTS (
		SELECT
			*
		FROM
			time_slide
		WHERE
			time_slide.time_slide_id == coinc_event.time_slide_id
			AND time_slide.offset != 0
	)
FROM
	coinc_burst
	JOIN coinc_event ON (
		coinc_event.coinc_event_id == coinc_burst.coinc_event_id
	)
WHERE
	coinc_event.likelihood >= ?
		""", (self.fapfar.minrank,)):
			if far is None:
				continue
			if is_background:
				self.background_ln_likelihood_ratio.append(ln_likelihood_ratio)
				self.background_far.append(far)
				self.background_fap.append(fap)
			else:
				self.zerolag_ln_likelihood_ratio.append(ln_likelihood_ratio)
				self.zerolag_far.append(far)
				self.zerolag_fap.append(fap)

	def finish(self):
		fig, axes = create_plot(r"$\ln \mathcal{L}$ Threshold", r"Number of Events $\geq \ln \mathcal{L}$")
		# FIXME currently works for only closed box, when enabling
		# open-box results we should check gstlal_inspiral_plot_summary
		for (ln_likelihood_ratio, fars) in [(self.background_ln_likelihood_ratio, self.background_far)]:
			zerolag_stats = numpy.array(sorted(ln_likelihood_ratio, reverse = True))
			detection_threshold = zerolag_stats[0]
			create_rate_vs_lnL_plot(axes, zerolag_stats, self.fapfar)
		axes.set_title(r"Event Count vs.\ Ranking Statistic Threshold (Closed Box)")
		return fig, detection_threshold


#
# =============================================================================
#
#                            Efficiency curve
#
# =============================================================================
#


def slope(x, y):
	"""
	From the x and y arrays, compute the slope at the x co-ordinates
	using a first-order finite difference approximation.
	"""
	slope = numpy.zeros((len(x),), dtype = "double")
	slope[0] = (y[1] - y[0]) / (x[1] - x[0])
	for i in range(1, len(x) - 1):
		slope[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])
	slope[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
	return slope


def upper_err(y, yerr, deltax):
	z = y + yerr
	deltax = int(deltax)
	upper = numpy.zeros((len(yerr),), dtype = "double")
	for i in range(len(yerr)):
		upper[i] = max(z[max(i - deltax, 0) : min(i + deltax, len(z))])
	return upper - y


def lower_err(y, yerr, deltax):
	z = y - yerr
	deltax = int(deltax)
	lower = numpy.zeros((len(yerr),), dtype = "double")
	for i in range(len(yerr)):
		lower[i] = min(z[max(i - deltax, 0) : min(i + deltax, len(z))])
	return y - lower


def write_efficiency(fileobj, bins, eff, yerr):
	print("# A	e	D[e]", file=fileobj)
	for A, e, De in zip(bins.centres()[0], eff, yerr):
		print("%.16g	%.16g	%.16g" % (A, e, De), file=fileobj)


def render_data_from_bins(dump_file, axes, efficiency_num, efficiency_den, cal_uncertainty, filter_width, colour = "k", erroralpha = 0.3, linestyle = "-"):
	# extract array of x co-ordinates, and the factor by which x
	# increases from one sample to the next.
	(x,) = efficiency_den.centres()
	x_factor_per_sample = efficiency_den.bins[0].delta

	# compute the efficiency, the slope (units = efficiency per
	# sample), the y uncertainty (units = efficiency) due to binomial
	# counting fluctuations, and the x uncertainty (units = samples)
	# due to the width of the smoothing filter.
	eff = efficiency_num.array / efficiency_den.array
	dydx = slope(numpy.arange(len(x), dtype = "double"), eff)
	yerr = numpy.sqrt(eff * (1. - eff) / efficiency_den.array)
	xerr = numpy.array([filter_width / 2.] * len(yerr))

	# compute the net y err (units = efficiency) by (i) multiplying the
	# x err by the slope, (ii) dividing the calibration uncertainty
	# (units = percent) by the fractional change in x per sample and
	# multiplying by the slope, (iii) adding the two in quadradure with
	# the y err.
	net_yerr = numpy.sqrt((xerr * dydx)**2. + yerr**2. + (cal_uncertainty / x_factor_per_sample * dydx)**2.)

	# compute net xerr (units = percent) by dividing yerr by slope and
	# then multiplying by the fractional change in x per sample.
	net_xerr = net_yerr / dydx * x_factor_per_sample

	# write the efficiency data to file
	write_efficiency(dump_file, efficiency_den.bins, eff, net_yerr)

	# plot the efficiency curve and uncertainty region
	patch = patches.Polygon(zip(numpy.concatenate((x, x[::-1])), numpy.concatenate((eff + upper_err(eff, yerr, filter_width / 2.), (eff - lower_err(eff, yerr, filter_width / 2.))[::-1]))), edgecolor = colour, facecolor = colour, alpha = erroralpha)
	axes.add_patch(patch)
	line, = axes.plot(x, eff, colour + linestyle)

	# compute 50% point and its uncertainty
	A50 = optimize.bisect(interpolate.interp1d(x, eff - 0.5), x[0], x[-1], xtol = 1e-40)
	A50_err = interpolate.interp1d(x, net_xerr)(A50)

	# print some analysis FIXME:  this calculation needs attention
	num_injections = efficiency_den.array.sum()
	num_samples = len(efficiency_den.array)
	print("Bins were %g samples wide, ideal would have been %g" % (filter_width, (num_samples / num_injections / interpolate.interp1d(x, dydx)(A50)**2.0)**(1.0/3.0)), file=sys.stderr)
	print("Average number of injections in each bin = %g" % efficiency_den.array.mean(), file=sys.stderr)

	return line, A50, A50_err

def create_efficiency_plot(axes, all_injections, found_injections, detection_threshold, cal_uncertainty):
	filter_width = 16.7
	# formats
	axes.semilogx()
	axes.set_position([0.10, 0.150, 0.86, 0.77])

	# set desired yticks
	axes.set_yticks((0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
	axes.set_yticklabels((r"\(0\)", r"\(0.1\)", r"\(0.2\)", r"\(0.3\)", r"\(0.4\)", r"\(0.5\)", r"\(0.6\)", r"\(0.7\)", r"\(0.8\)", r"\(0.9\)", r"\(1.0\)"))
	axes.xaxis.grid(True, which = "major,minor")
	axes.yaxis.grid(True, which = "major,minor")

	# put made and found injections in the denominators and
	# numerators of the efficiency bins
	bins = rate.NDBins((rate.LogarithmicBins(min(sim.amplitude for sim in all_injections), max(sim.amplitude for sim in all_injections), 400),))
	efficiency_num = rate.BinnedArray(bins)
	efficiency_den = rate.BinnedArray(bins)
	for sim in found_injections:
		efficiency_num[sim.amplitude,] += 1
	for sim in all_injections:
		efficiency_den[sim.amplitude,] += 1

	# generate and plot trend curves.  adjust window function
	# normalization so that denominator array correctly
	# represents the number of injections contributing to each
	# bin:  make w(0) = 1.0.  note that this factor has no
	# effect on the efficiency because it is common to the
	# numerator and denominator arrays.  we do this for the
	# purpose of computing the Poisson error bars, which
	# requires us to know the counts for the bins
	windowfunc = rate.gaussian_window(filter_width)
	windowfunc /= windowfunc[len(windowfunc) / 2 + 1]
	rate.filter_array(efficiency_num.array, windowfunc)
	rate.filter_array(efficiency_den.array, windowfunc)

	# regularize:  adjust unused bins so that the efficiency is
	# 0, not NaN
	assert (efficiency_num.array <= efficiency_den.array).all()
	efficiency_den.array[(efficiency_num.array == 0) & (efficiency_den.array == 0)] = 1

	line1, A50, A50_err = render_data_from_bins(file("string_efficiency.dat", "w"), axes, efficiency_num, efficiency_den, cal_uncertainty, filter_width, colour = "k", linestyle = "-", erroralpha = 0.2)

	# add a legend to the axes
	axes.legend((line1,), (r"\noindent Injections recovered with $\log \Lambda > %.2f$" % detection_threshold,), loc = "lower right")

	# adjust limits
	axes.set_xlim([3e-22, 3e-19])
	axes.set_ylim([0.0, 1.0])


class Efficiency(object):
	def __init__(self, detection_threshold, cal_uncertainty):
		self.detection_threshold = detection_threshold
		self.cal_uncertainty = cal_uncertainty
		self.all = []
		self.found = []
	
	def add_contents(self, contents, live_seglists):
		# FIXME add quietest found and loudest missed injections

		#
		# update segment information
		#
	
		# for each injection, retrieve the highest likelihood ratio
		# of the burst coincs that were found to match the
		# injection, or null if no burst coincs matched the
		# injection
		offsetvectors = contents.time_slide_table.as_dict()
		stringutils.create_recovered_ln_likelihood_ratio_table(contents.connection, contents.bb_definer_id)
		for values in contents.connection.cursor().execute("""
SELECT
	sim_burst.*,
	recovered_ln_likelihood_ratio.ln_likelihood_ratio
FROM
	sim_burst
	LEFT JOIN recovered_ln_likelihood_ratio ON (
		recovered_ln_likelihood_ratio.simulation_id == sim_burst.simulation_id
	)
		"""):
			sim = contents.sim_burst_table.row_from_cols(values[:-1])
			ln_likelihood_ratio = values[-1]
			found = ln_likelihood_ratio is not None
			# were at least 2 instruments on when the injection
			# was made?
			instruments = offsetvectors[sim.time_slide_id].keys()
			if len([instrument for instrument in instruments if sim.time_at_instrument(instrument, offsetvectors[sim.time_slide_id]) in live_seglists[instrument]]) >= 2:
				# yes
				self.all.append(sim)
				if found and ln_likelihood_ratio > self.detection_threshold:
					self.found.append(sim)
					# 1/amplitude needs to be first so
					# that it acts as the sort key
					# FIXME used to have filenames recorded,
					# now it doesn't work. try make it work
					record = (1.0 / sim.amplitude, sim, offsetvectors[sim.time_slide_id], ln_likelihood_ratio)
			elif found:
				# no, but it was found anyway
				print("odd, injection %s was found but not injected ..." % sim.simulation_id, file=sys.stderr)

	def finish(self):
		fig, axes = create_plot(r"Injection Amplitude (\(\mathrm{s}^{-\frac{1}{3}}\))", "Detection Efficiency")
		create_efficiency_plot(axes, self.all, self.found, self.detection_threshold, self.cal_uncertainty)
		axes.set_title(r"Detection Efficiency vs.\ Amplitude")
		return fig


#
# =============================================================================
#
#                                  Main
#
# =============================================================================
#


def group_files(filenames, verbose = False):
        # Figure out which files are injection runs and which aren't
	# The files that contain sim_burst table are assumed to be
	# injection files.
        injection_filenames = []
        noninjection_filenames = []
        for n, filename in enumerate(filenames):
                if verbose:
                        print("%d/%d: %s" % (n + 1, len(filenames), filename), file=sys.stderr)
                connection = sqlite3.connect(filename)
                if "sim_burst" in dbtables.get_table_names(connection):
                        if verbose:
                                print("\t--> injections", file=sys.stderr)
                        injection_filenames.append(filename)
                else:
                        if verbose:
                                print("\t--> non-injections", file=sys.stderr)
                        noninjection_filenames.append(filename)
                connection.close()
        return injection_filenames, noninjection_filenames


#
# Parse command line
#

options, filenames = parse_command_line()

injection_filenames, noninjection_filenames = group_files(filenames, verbose = options.verbose)


#
# rate_vs_thresh "expected from noise model"
#

if options.rankingstatpdf_file is not None:
	rankingstatpdf = stringutils.marginalize_rankingstatpdf([options.rankingstatpdf_file], verbose = options.verbose)
else:
	raise ValueError("no rankingstat PDF file")

fapfar = stringutils.FAPFAR(rankingstatpdf)
print("livetime of the search: %f" % fapfar.livetime, file = sys.stderr)

rate_vs_thresh = RateVsThreshold(fapfar)


#
# process through non-inj files to construct the rate_vs_thresh plot
#

if options.detection_threshold is None:
	if options.verbose:
		print("Collecting background and zero-lag statistics ...", file=sys.stderr)

	# FIXME this and the inj version below should be combined into a
	# single function (see lalapps_string_final)
	for n, filename in enumerate(noninjection_filenames):
		if options.verbose:
			print("%d/%d: %s" % (n + 1, len(filenames), filename), file=sys.stderr)
		working_filename = dbtables.get_connection_filename(filename, tmp_path = options.tmp_space, verbose = options.verbose)
		connection = sqlite3.connect(str(working_filename))
		contents = SnglBurstUtils.CoincDatabase(connection, live_time_program = "StringSearch", search = "StringCusp", veto_segments_name = options.vetoes_name)
		if options.verbose:
			SnglBurstUtils.summarize_coinc_database(contents)

		rate_vs_thresh.add_contents(contents)
		connection.close()
		dbtables.discard_connection_filename(filename, working_filename, verbose = options.verbose)

	fig_rate_vs_thresh, detection_threshold = rate_vs_thresh.finish()
	print("Simulated \\log \\Lambda for highest-ranked zero-lag survivor: %.9g" % detection_threshold, file=sys.stderr)

	try:
		fig_rate_vs_thresh.tight_layout(rect=(0.03, 0.02, 0.98, 0.98))
	except ValueError:
		pass
	fig_rate_vs_thresh.savefig("string_rate.png")

else:
	detection_threshold = options.detection_threshold
	print("Likelihood ratio for highest-ranked zero-lag survivor from command line: %.9g" % detection_threshold, file=sys.stderr)

#
# process through inj files to construct the efficiency plot,
#

# initialize
efficiency = Efficiency(detection_threshold, options.cal_uncertainty)

for n, filename in enumerate(injection_filenames): 
	if options.verbose:
		print("%d/%d: %s" % (n + 1, len(filenames), filename), file=sys.stderr)
	working_filename = dbtables.get_connection_filename(filename, tmp_path = options.tmp_space, verbose = options.verbose)
	connection = sqlite3.connect(str(working_filename))
	contents = SnglBurstUtils.CoincDatabase(connection, live_time_program = "StringSearch", search = "StringCusp", veto_segments_name = options.vetoes_name)
	if options.verbose:
		SnglBurstUtils.summarize_coinc_database(contents)

	# the rankingstat object files that one need for getting the livesegs for
	# each injection set (injection0, injection1, ...)is the ones that contains 
	# the same start time and tag in the filenname.
	# FIXME clearly is a problem when having injection10 and beyond
	tag, = ligolw_process.get_process_params(contents.xmldoc, "StringSearch", "--user-tag")
	start_time, = ligolw_process.get_process_params(contents.xmldoc, "StringSearch", "--gps-start-time")
	these_likelihood_files = [f for f in options.likelihood_filenames if (str(tag) in f and str(start_time) in f)]
	assert len(these_likelihood_files) == 1
	rankingstat = stringutils.marginalize_rankingstat(these_likelihood_files, verbose = options.verbose)
	live_seglists = rankingstat.denominator.triggerrates.segmentlistdict() 
	del rankingstat

	efficiency.add_contents(contents, live_seglists)
	connection.close()
	dbtables.discard_connection_filename(filename, working_filename, verbose = options.verbose)

fig_efficiency = efficiency.finish() 


#
# Done. Save figures into files
#

try:
	fig_efficiency.tight_layout(rect=(0.03, 0.02, 0.98, 0.98))
except ValueError:
	pass

fig_efficiency.savefig("string_efficiency.png")

#!/usr/bin/python


import bisect
from optparse import OptionParser
import math
import matplotlib
matplotlib.rcParams.update({
	"font.size": 8.0,
	"axes.titlesize": 10.0,
	"axes.labelsize": 10.0,
	"xtick.labelsize": 8.0,
	"ytick.labelsize": 8.0,
	"legend.fontsize": 8.0,
	"figure.dpi": 600,
	"savefig.dpi": 600,
	"text.usetex": True,
	"path.simplify": True
})
from matplotlib import figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
try:
	import sqlite3
except ImportError:
	# pre 2.5.x
	from pysqlite2 import dbapi2 as sqlite3
import sys

from glue.ligolw import lsctables
from glue.ligolw import utils
from pylal import rate


def parse_command_line():
	parser = OptionParser()
	parser.add_option("--background-db", metavar = "filename")
	parser.add_option("--injections-db", metavar = "filename")
	options, filenames = parser.parse_args()

	return options

options = parse_command_line()


print >>sys.stderr, "openning databases ..."
backgrounddb = sqlite3.connect(options.background_db)
injectionsdb = sqlite3.connect(options.injections_db)

background_cursor = backgrounddb.cursor()
injections_cursor = injectionsdb.cursor()


if False:
	print >>sys.stderr, "measuring parameter space boundaries ..."
	def f(cursor_a, cursor_b, col):
		return cursor_a.execute("SELECT %s FROM sngl_inspiral" % col).fetchone()[0], cursor_b.execute("SELECT %s FROM sngl_inspiral" % col).fetchone()[0]

	minchisq = min(f(background_cursor, injections_cursor, "MIN(chisq)"))
	maxchisq = max(f(background_cursor, injections_cursor, "MAX(chisq)"))
	minsnr = min(f(background_cursor, injections_cursor, "MIN(snr)"))
	maxsnr = max(f(background_cursor, injections_cursor, "MAX(snr)"))

	del f


	likelihood = rate.BinnedRatios(rate.NDBins((rate.LogarithmicBins(minsnr, maxsnr, 750), rate.LogarithmicBins(minchisq, maxchisq, 750))))


	print >>sys.stderr, "collecting events ..."
	for coords in injections_cursor.execute("SELECT snr, chisq FROM sngl_inspiral WHERE event_id IN (SELECT event_id FROM coinc_event_map)"):
		likelihood.incnumerator(coords)
	for coords in background_cursor.execute("SELECT snr, chisq FROM sngl_inspiral"):
		likelihood.incdenominator(coords)


	print >>sys.stderr, "smoothing event densities ..."
	rate.filter_binned_ratios(likelihood, rate.gaussian_window2d(likelihood.numerator.bins.shape[0]/30, likelihood.numerator.bins.shape[1]/30))


	xmldoc = lsctables.ligolw.Document()
	xmldoc.appendChild(lsctables.ligolw.LIGO_LW())
	xmldoc.childNodes[-1].appendChild(rate.binned_ratios_to_xml(likelihood, "LLOID_roc_likelihood"))
	utils.write_filename(xmldoc, "roc_likelihood.xml.gz", gz = True, verbose = True)
else:
	likelihood = rate.binned_ratios_from_xml(utils.load_filename("roc_likelihood.xml.gz", gz = True, verbose = True), "LLOID_roc_likelihood")


def calc_likelihood(snr, chisq):
	return likelihood[(snr, chisq)]

backgrounddb.create_function("likelihood", 2, calc_likelihood)
injectionsdb.create_function("likelihood", 2, calc_likelihood)


print >>sys.stderr, "computing likelihood ratios ..."
background_likelihoods = [vals[0] for vals in background_cursor.execute("SELECT likelihood(snr, chisq) FROM sngl_inspiral")]
injections_likelihoods = [vals[0] for vals in injections_cursor.execute("""
SELECT
	(SELECT
		MAX(likelihood(sngl_inspiral.snr, sngl_inspiral.chisq))
	FROM
		sngl_inspiral
		JOIN coinc_event_map AS mapa ON (
			mapa.table_name == 'sngl_inspiral'
			AND mapa.event_id == sngl_inspiral.event_id
		)
		JOIN coinc_event_map AS mapb ON (
			mapb.coinc_event_id == mapa.coinc_event_id
		)
	WHERE
		mapb.table_name == 'sim_inspiral'
		AND mapb.event_id == sim_inspiral.simulation_id
	)
FROM
	sim_inspiral
""")]

background_likelihoods.sort()
injections_likelihoods.sort()


print "duplicate backgrounds: ", (True in [a == b for a, b in zip(background_likelihoods[:-1], background_likelihoods[1:])])
print "duplicate injections: ", (True in [a == b for a, b in zip(injections_likelihoods[:-1], injections_likelihoods[1:])])
print "0 in background: ", (0 in background_likelihoods)
print "inf in injections: ", (float("inf") in injections_likelihoods)


def remove_infs(background_likelihoods, injections_likelihoods):
	#
	# find the largest non-inf likelihood
	#

	max = min(background_likelihoods)
	for l in background_likelihoods + injections_likelihoods:
		if l > max and l != float("inf"):
			max = l

	#
	# assign something a bit larger than that to all the events that
	# came out as "inf"
	#

	max += 100
	for i in range(len(background_likelihoods)):
		if background_likelihoods[i] == float("inf"):
			background_likelihoods[i] = max
	for i in range(len(injections_likelihoods)):
		if injections_likelihoods[i] == float("inf"):
			injections_likelihoods[i] = max

remove_infs(background_likelihoods, injections_likelihoods)


def tap(likelihood, injections_likelihoods = sorted(filter(lambda x: x >= min(background_likelihoods), injections_likelihoods))):
	return 1.0 - float(bisect.bisect_left(injections_likelihoods, likelihood)) / len(injections_likelihoods)

def fap(likelihood, background_likelihoods = sorted(background_likelihoods)):
	return 1.0 - float(bisect.bisect_left(background_likelihoods, likelihood)) / len(background_likelihoods)

fig = figure.Figure()
FigureCanvas(fig)
fig.set_size_inches(165.0 / 25.4, 165.0 / 25.4 / ((1 + math.sqrt(5)) / 2))
axes = fig.gca()
axes.grid(True)
axes.set_xlabel("Fraction of noise surviving")
axes.set_ylabel(r"\begin{center}Fraction of injections surviving\\(of those that survive when all noise survives)\end{center}")

print >>sys.stderr, "plotting ..."
likelihoods = background_likelihoods + injections_likelihoods
likelihoods.sort()
x = map(fap, likelihoods)
y = map(tap, likelihoods)
axes.loglog(x, y)
axes.set_xlim(min(x), max(x))
axes.set_ylim(min(y), max(y))

print >>sys.stderr, "writing roc.png ..."
fig.savefig("roc.png")

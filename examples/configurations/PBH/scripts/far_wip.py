#!/usr/bin/python

import sys
import numpy
from scipy import interpolate
from glue.ligolw import lsctables
from glue.ligolw import utils
from glue.ligolw.utils import process as ligolw_process
from glue.ligolw.utils import segments as ligolw_segments
from pylal import inject
from pylal import rate
from optparse import OptionParser
from gstlal.svd_bank import read_bank

def smooth_bins(bA, stride = 5):
	wn = rate.gaussian_window2d(3*stride,3*stride, sigma=2*stride)
	rate.filter_array(bA.array.T,wn)

def decimate_array(arr, stride=5):
	return arr[::stride, ::stride]

def linearize_array(arr):
	return arr.reshape((1,arr.shape[0] * arr.shape[1]))

def get_nonzero(arr):
	return arr[arr != 0]

def possible_ranks_array(A, B, Alt, Blt, delta_t):
	out = numpy.outer(A, B) * 2. * delta_t / Alt / Blt
	out = out.reshape((out.shape[0] * out.shape[1],))
	out.sort()
	return out

def FAP_from_ranks(ranks):
	"""
	ranks should be sorted
	"""
	FAP = (numpy.arange(len(ranks))+1.) / len(ranks)
	return interpolate.interp1d(ranks, FAP, fill_value=0, bounds_error=False)
	
def FAR_from_FAP(faps, t):
	return 0. - numpy.log(1.-faps) / t

def parse_banks(bank_string):
	out = {}
	for b in bank_string.split(','):
		ifo, bank = b.split(':')
		out.setdefault(ifo, []).append(bank)
	return out

def get_trials_factor(triggerdoc, ifo):
	#FIXME don't hard code some of this stuff !!
	svd_bank_string, = ligolw_process.get_process_params(triggerdoc, "gstlal_inspiral", "--svd-bank")
	banks = parse_banks(svd_bank_string)
	print >> sys.stderr, "reading SVD bank ..."
	bank = read_bank(banks[ifo][0])
	for frag in bank.bank_fragments:
		if frag.rate == 256:
			return len(frag.singular_values)

def parse_command_line():
	parser = OptionParser(
		description = __doc__
	)
	parser.add_option("--background-bins-file", metavar = "filename", help = "Set the name of the xml file containing the snr / chisq background distributions")
	parser.add_option("--segments-file", metavar = "filename", help = "Set the name of the xml file containing analysis segments with name 'RESULTS'")
	parser.add_option("--vetoes-file", metavar = "filename", help = "Set the name of the xml file containing the veto segments with name 'vetoes'")
	parser.add_option("--stride", metavar = "int", type="int", default=5, help = "set the stride to decimate the bins, default 5")
	options, filenames = parser.parse_args()
	return options, filenames


# Parse command line
options, filenames = parse_command_line()


# load background data
snrchidoc = utils.load_filename(options.background_bins_file, verbose = True)
segments = ligolw_segments.segmenttable_get_by_name(utils.load_filename(options.segments_file, verbose = True), "RESULT").coalesce()
vetoes = ligolw_segments.segmenttable_get_by_name(utils.load_filename(options.vetoes_file, verbose = True), "vetoes").coalesce()


# retrieve rank data
print >>sys.stderr, "extracting rank data ..."
counts = {}
for arrayname in [child.getAttribute("Name") for child in snrchidoc.childNodes[-1].childNodes]:
	instrument = arrayname.split(":")[0]
	counts[instrument] = rate.binned_array_from_xml(snrchidoc, instrument)
	smooth_bins(counts[instrument], stride = options.stride)


# compute FAP mapping preliminaries
# FIXME only works for H1 / L1, dont' know how to handle correlated H2, not a problem for *this* search?
print >>sys.stderr, "computing FAP map preliminaries..."
H1nonzero = get_nonzero(linearize_array(decimate_array(counts["H1"].array, stride = options.stride)))
L1nonzero = get_nonzero(linearize_array(decimate_array(counts["L1"].array, stride = options.stride)))


# iterate over files to rank
for f in filenames:

	# get the trigger document
	triggerdoc = utils.load_filename(f, verbose = True)

	print >>sys.stderr, "computing live time ..."
	# intersect segments with search_summary segments and remove vetoes
	search_summary_segs = lsctables.table.get_table(triggerdoc, lsctables.SearchSummaryTable.tableName).get_in_segmentlistdict(lsctables.table.get_table(triggerdoc, lsctables.ProcessTable.tableName).get_ids_by_program("gstlal_inspiral")).coalesce()
	livetime = dict((instrument, float(abs(seglist))) for instrument, seglist in ((segments & search_summary_segs) - vetoes).items())


	print >>sys.stderr, "indexing tables ..."
	# index for rapid sngl_inspiral and coinc_inspiral row look-up
	index = dict((row.event_id, row) for row in lsctables.table.get_table(triggerdoc, lsctables.SnglInspiralTable.tableName))
	index.update(dict((row.coinc_event_id, row) for row in lsctables.table.get_table(triggerdoc, lsctables.CoincInspiralTable.tableName)))
	# index for rapid coinc tuple look-up
	coinc_index = {}
	for row in lsctables.table.get_table(triggerdoc, lsctables.CoincMapTable.tableName):
		coinc_index.setdefault(row.coinc_event_id, set()).add(index[row.event_id])


	# compute rates
	print >>sys.stderr, "computing rates ..."
	coincidence_threshold, = ligolw_process.get_process_params(triggerdoc, "gstlal_inspiral", "--coincidence-threshold")
	delta_t = coincidence_threshold + inject.light_travel_time("H1", "L1")
	ranks = possible_ranks_array(H1nonzero, L1nonzero, livetime["H1"], livetime["L1"], delta_t)
	faps = FAP_from_ranks(ranks)
	coinc_inspiral_table = lsctables.table.get_table(triggerdoc, lsctables.CoincInspiralTable.tableName)
	N = len(coinc_inspiral_table)
	# get trials factor
	trials_factor = max([get_trials_factor(triggerdoc, "H1"), get_trials_factor(triggerdoc, "L1")])
	print >>sys.stderr, "found trials factor to be %d ..." % (trials_factor,)
	for n, coinc_inspiral in enumerate(coinc_inspiral_table):
		if not n % 531:
			print >>sys.stderr, "\t%d / %d\r" % (n, N),
		instruments = coinc_inspiral.get_ifos()
		# if H1 and L1 don't participate, skip
		if not set(("H1", "L1")).issubset(instruments):
			coinc_inspiral.false_alarm_rate = None
			continue
		# retrieve \rho and \chi^{2} values
		rho_chi = dict((event.ifo, (event.snr, event.chisq**.5 / event.snr)) for event in coinc_index[coinc_inspiral.coinc_event_id])
		# assign rate
		coinc_inspiral.false_alarm_rate = (counts["H1"][rho_chi["H1"]] / livetime["H1"]) * (counts["L1"][rho_chi["L1"]] / livetime["L1"]) * 2. * delta_t
		# FIXME use intersection of H1 and L1 for livetime hack to
		# avoid going off the edge in interp doesn't really matter,
		# this is buried in the noise lower bound (the more important
		# bound :) handled by routine and results in 0.  It shouldn't
		# be possible to have a zero FAR for a non injection by
		# definition since all events are included in ranking.
		# However, perhaps the smoothing + stride could be a problem??
		# We have to make the smoothing bigger than the stride
		if coinc_inspiral.false_alarm_rate > ranks[-2]:
			coinc_inspiral.false_alarm_rate = ranks[-2]
		coinc_inspiral.combined_far = trials_factor * FAR_from_FAP(faps(coinc_inspiral.false_alarm_rate), livetime["H1"])


	# write output
	utils.write_filename(triggerdoc, f, gz = True, verbose = True)

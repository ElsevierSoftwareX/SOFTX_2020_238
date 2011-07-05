import sys
from glue.ligolw import lsctables
from glue.ligolw import utils
from glue.ligolw.utils import process as ligolw_process
from glue.ligolw.utils import segments as ligolw_segments
from pylal import inject
from pylal import rate


triggerdoc = utils.load_filename("H1H2L1-0009-871147452-871247452-triggers.xml.gz", verbose = True)
snrchidoc = utils.load_filename("snr_chi_H1H2L1-0009-871147452-871247452-triggers.xml.xml.gz", verbose = True)
segments = ligolw_segments.segmenttable_get_by_name(utils.load_filename("segments.xml", verbose = True), "RESULT").coalesce()
vetoes = ligolw_segments.segmenttable_get_by_name(utils.load_filename("vetoes.xml", verbose = True), "vetoes").coalesce()


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


print >>sys.stderr, "extracting rate data ..."
# retrieve rate data
counts = {}
for arrayname in [child.getAttribute("Name") for child in snrchidoc.childNodes[-1].childNodes]:
	instrument = arrayname.split(":")[0]
	counts[instrument] = rate.binned_array_from_xml(snrchidoc, instrument)


# compute rates
print >>sys.stderr, "computing rates ..."
coincidence_threshold, = ligolw_process.get_process_params(triggerdoc, "gstlal_inspiral", "--coincidence-threshold")
coinc_inspiral_table = lsctables.table.get_table(triggerdoc, lsctables.CoincInspiralTable.tableName)
N = len(coinc_inspiral_table)
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
	delta_t = coincidence_threshold + inject.light_travel_time("H1", "L1")
	coinc_inspiral.false_alarm_rate = (counts["H1"][rho_chi["H1"]] / livetime["H1"]) * (counts["L1"][rho_chi["L1"]] / livetime["L1"]) * 2 * delta_t
print >>sys.stderr, "\t%d / %d" % (n, N)


# write output
utils.write_filename(triggerdoc, "H1H2L1-0009-871147452-871247452-triggers.xml.gz", gz = True, verbose = True)

#!/usr/bin/env python



from optparse import OptionParser, Option
from glue import gpstime
from glue.pipeline import CondorDAGJob, CondorDAGNode, CondorDAG
import os
import tempfile
import re


# Some convenient classes

class EnvCondorJob(CondorDAGJob):
	def __init__(self, cmdline, outputname=None, subfilename=None):
		"""Create a job that runs an executable that is on the user's PATH.
		
		cmdline is a string containing the command to execute, and may contain
		references to environment variables and macros in the style of Condor
		submit files.
		
		By default, the name of the submit file will be taken from the name of
		the executable, which is derived from the part of the cmdline that precedes
		the first space."""
		CondorDAGJob.__init__(self, 'vanilla', '/usr/bin/env')
		cmdline = cmdline.replace('\n', ' ')
		if subfilename is None:
			subfilename = cmdline.split(' ', 1)[0]
		if outputname is None:
			outputname = subfilename
		self.add_arg(cmdline)
		self.add_condor_cmd("getenv", "true")
		self.set_stderr_file('%s.err' % outputname)
		self.set_stdout_file('%s.out' % outputname)
		self.set_sub_file('%s.sub' % subfilename)


def makeNode(dag, job, name=None, macros=None, parents=None, children=None):
	node = CondorDAGNode(job)
	# FIXME why does CondorDAGNode strip out underscores from argument names?
	node._CondorDAGNode__bad_macro_chars = re.compile(r'')
	if name is None:
		node.set_name(job.get_sub_file().rsplit(".", 1)[0])
	else:
		node.set_name(name)
	if macros is not None:
		for key, val in macros.iteritems():
			node.add_macro(key, val)
	if parents is not None:
		for parent in parents:
			node.add_parent(parent)
	if children is not None:
		for child in children:
			child.add_parent(node)
	dag.add_node(node)
	return node



# Parse command line

date_parts = ("year", "month", "day", "hour")
options, args = OptionParser(
	option_list = [Option("--"+s, type="int") for s in date_parts]
).parse_args()

for s in date_parts:
	if getattr(options, s) is None:
		raise RuntimeError("Argument --%s required but not provided" % s)
if options.hour not in (0, 8, 16):
	raise RuntimeError("Argument --hour expected to be one of (0, 8, 16), but %d was provided" % options.hour)

gps_start_time = gpstime.GpsSecondsFromPyUTC(gpstime.mkUTC(
	options.year, options.month, options.day, options.hour, 0, 0))
gps_end_time = gps_start_time + int(8.5 * 3600)

tmpdir = os.getenv("TMPDIR")
if tmpdir is None:
	raise RuntimeError("Environment variable TMPDIR must be set to point to a local directory")



# Make job subdir

dirname = "%04d-%02d-%02dT%02d" % (options.year, options.month, options.day, options.hour)
if not os.path.exists(dirname):
	os.mkdir(dirname)
os.chdir(dirname)



# Parameters

tmpltbank_start_time = 958740096
tmpltbank_end_time = 958742144
tmpltbank_duration = 2048
comment = "GSTLAL_8HOURLY"

ifodict = {
	"H1": {"reference_psd_start_time": 958739939, "reference_psd_end_time": 958743539},
	"L1": {"reference_psd_start_time": 958744974, "reference_psd_end_time": 958747374},
}

logfile = tempfile.mkstemp(prefix = "8hourly.%s" % dirname, suffix = '.log', dir = tmpdir)[1]



# Construct submit files

ligo_data_find_sub = EnvCondorJob(r"""ligo_data_find
	-o H -t H1_DMT_C00_L2 -u file -l -s 958739936 -e 958743552""")

lalapps_tmpltbank_sub = EnvCondorJob(r"""lalapps_tmpltbank
	--verbose
	--user-tag $(macro_comment)
	--gps-start-time $(macro_tmpltbank_start_time)
	--gps-end-time $(macro_tmpltbank_end_time)
	--grid-spacing Hexagonal
	--dynamic-range-exponent 69.0
	--enable-high-pass 30.0 --high-pass-order 8
	--strain-high-pass-order 8
	--minimum-mass 1.2 --maximum-mass 1.6
	--approximant TaylorF2 --order twoPN
	--standard-candle
	--calibrated-data real_8
	--candle-mass1 1.4 --candle-mass2 1.4
	--channel-name H1:DMT-STRAIN --frame-cache ligo_data_find.out
	--space Tau0Tau3 --number-of-segments 15
	--minimal-match 0.98 --candle-snr 8 --debug-level 33 --high-pass-attenuation 0.1
	--min-high-freq-cutoff SchwarzISCO --max-high-freq-cutoff SchwarzISCO
	--segment-length 524288
	--low-frequency-cutoff 40.0 --pad-data 8 --num-freq-cutoffs 1
	--sample-rate 2048 --high-frequency-cutoff 921.6 --resample-filter ldas
	--strain-high-pass-atten 0.1 --strain-high-pass-freq 30
	--min-total-mass 2.4 --max-total-mass 3.2
	--write-compress --spectrum-type median""")

gstlal_prune_duplicate_mass_pairs_sub = EnvCondorJob(r"""gstlal_prune_duplicate_mass_pairs
	H1-TMPLTBANK_$(macro_comment)-$(macro_tmpltbank_start_time)-$(macro_tmpltbank_duration).xml.gz tmpltbank.xml.gz""")

gstlal_inspiral_sub = EnvCondorJob(r"""gstlal_inspiral
	--verbose
	--online-data
	--comment $(macro_comment)
	--instrument $(macro_instrument)
	--reference-psd reference_psd.$(macro_instrument).xml.gz
	--gps-start-time $(macro_gps_start_time)
	--gps-end-time $(macro_gps_end_time)
	--template-bank tmpltbank.xml.gz
	--output gstlal_inspiral.$(macro_instrument).sqlite""",
	"gstlal_inspiral.$(macro_instrument)")

gstlal_reference_psd_sub = EnvCondorJob(r"""gstlal_reference_psd
	--verbose
	--online-data
	--instrument $(macro_instrument)
	--write-psd reference_psd.$(macro_instrument).xml.gz
	--gps-start-time $(macro_gps_start_time)
	--gps-end-time $(macro_gps_end_time)""",
	"reference_psd.$(macro_instrument)")

ligolw_segment_query_sub = EnvCondorJob(r"""ligolw_segment_query
	--query-segments
	--database
	--gps-start-time $(macro_gps_start_time)
	--gps-end-time $(macro_gps_end_time)
	--include-segments $(macro_instrument):DMT-SCIENCE:1
	--output-file science_segments.$(macro_instrument).xml""",
	"ligolw_segment_query.$(macro_instrument)")
ligolw_segment_query_sub.set_universe("local") # FIXME: Find out how to query segdb from cluster node.

ligolw_sqlite_sub = EnvCondorJob(r"""ligolw_sqlite
	--database gstlal_inspiral.$(macro_instrument).sqlite
	science_segments.$(macro_instrument).xml
	/archive/home/jveitch/public_html/S6inj/HL-INJECTIONS_S6_ALL.xml""",
	"ligolw_sqlite.$(macro_instrument)")

gstlal_8hourly_plots_sub = EnvCondorJob("gstlal_8hourly_plots --glob *.sqlite")
gstlal_8hourly_plots_sub.set_universe("local")

gstlal_plotlatency_sub = EnvCondorJob("""gstlal_plotlatency
	--disable-legend
	gstlal_inspiral.$(macro_instrument).out
	gstlal_inspiral.$(macro_instrument).out.png""",
	"gstlal_plotlatency.$(macro_instrument)")

gstlal_inspiral_page_sub = EnvCondorJob("gstlal_inspiral_page")



# Construct DAG nodes

dag = CondorDAG(logfile)
dag.set_dag_file("8hourly")

ligo_data_find_node = makeNode(dag, ligo_data_find_sub)

lalapps_tmpltbank_node = makeNode(dag,
	lalapps_tmpltbank_sub,
	macros = {
		"macro_comment": comment,
		"macro_tmpltbank_start_time": tmpltbank_start_time,
		"macro_tmpltbank_end_time": tmpltbank_end_time,
		"macro_tmpltbank_duration": tmpltbank_duration,
	},
	parents = (ligo_data_find_node,))

gstlal_prune_duplicate_mass_pairs_node = makeNode(dag,
	gstlal_prune_duplicate_mass_pairs_sub,
	macros = {
		"macro_comment": comment,
		"macro_tmpltbank_start_time": tmpltbank_start_time,
		"macro_tmpltbank_duration": tmpltbank_duration,
	},
	parents = (lalapps_tmpltbank_node,))

gstlal_8hourly_plots_node = makeNode(dag, gstlal_8hourly_plots_sub)

gstlal_inspiral_page_node = makeNode(dag, gstlal_inspiral_page_sub, parents=(gstlal_8hourly_plots_node,))

for ifo, props in ifodict.iteritems():
	gstlal_reference_psd_node = makeNode(dag, gstlal_reference_psd_sub,
		name = "gstlal_reference_psd.%s" % ifo,
		macros = {
			"macro_instrument": ifo,
			"macro_gps_start_time": props["reference_psd_start_time"],
			"macro_gps_end_time": props["reference_psd_end_time"],
		})

	gstlal_inspiral_node = makeNode(dag, gstlal_inspiral_sub,
		name = "gstlal_inspiral.%s" % ifo,
		macros = {
			"macro_comment": comment,
			"macro_instrument": ifo,
			"macro_gps_start_time": gps_start_time,
			"macro_gps_end_time": gps_end_time,
		},
		parents = (gstlal_reference_psd_node, gstlal_prune_duplicate_mass_pairs_node),
		children = (gstlal_8hourly_plots_node,))

	ligolw_segment_query_node = makeNode(dag, ligolw_segment_query_sub,
		name = "ligolw_segment_query.%s" % ifo,
		macros = {
			"macro_instrument": ifo,
			"macro_gps_start_time": gps_start_time,
			"macro_gps_end_time": gps_end_time,
		},
		parents = (gstlal_inspiral_node,)
	)

	ligolw_sqlite_node = makeNode(dag, ligolw_sqlite_sub,
		name = "ligolw_sqlite.%s" % ifo,
		macros = {
			"macro_instrument": ifo
		},
		parents = (gstlal_inspiral_node, ligolw_segment_query_node,),
		children = (gstlal_8hourly_plots_node,),
	)

	gstlal_plotlatency_node = makeNode(dag, gstlal_plotlatency_sub,
		name = "gstlal_plotlatency.%s" % ifo,
		macros = {"macro_instrument": ifo},
		parents = (gstlal_inspiral_node,),
		children = (gstlal_inspiral_page_node,))



# Write DAG and submit files.

dag.write_sub_files()
dag.write_dag()

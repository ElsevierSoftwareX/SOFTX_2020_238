#!/usr/bin/env python



from optparse import OptionParser, Option
from glue import gpstime
import os



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

tmpdir = os.getenv("TMPDIR")
if tmpdir is None:
	raise RuntimeError("Environment variable TMPDIR must be set to point to a local directory")


# Make job subdir

dirname = "%04d-%02d-%02dT%02d" % (options.year, options.month, options.day, options.hour)
os.mkdir(dirname)
os.chdir(dirname)




# Build submit files and DAG

submit_file_tail = r"""
log = %(tmpdir)s/8hourly.%(logname)s.log
universe = vanilla
executable = /usr/bin/env
notification = never
getenv = True
queue 1
""" % {"tmpdir": tmpdir, "logname": dirname}


print >>open("ligo_data_find.sub", "w"), r"""
arguments = ligo_data_find \
	-o H -t H1_DMT_C00_L2 -u file -l \
	-s $(macro_tmpltbank_start_time) -e $(macro_tmpltbank_end_time)

output = ligo_data_find.out
error = ligo_data_find.err
""" + submit_file_tail



print >>open("lalapps_tmpltbank.sub", "w"), r"""
arguments = lalapps_tmpltbank \
	--verbose \
	--user-tag $(macro_comment) \
	--gps-start-time $(macro_tmpltbank_start_time) \
	--gps-end-time $(macro_tmpltbank_end_time) \
	--grid-spacing Hexagonal \
	--dynamic-range-exponent 69.0 \
	--enable-high-pass 30.0 --high-pass-order 8 \
	--strain-high-pass-order 8 \
	--minimum-mass 1.2 --maximum-mass 1.6 \
	--approximant FindChirpSP --order twoPN \
	--standard-candle \
	--calibrated-data real_8 \
	--candle-mass1 1.4 --candle-mass2 1.4 \
	--channel-name H1:DMT-STRAIN --frame-cache ligo_data_find.out \
	--space Tau0Tau3 --number-of-segments 15 \
	--minimal-match 0.98 --candle-snr 8 --debug-level 33 --high-pass-attenuation 0.1 \
	--min-high-freq-cutoff SchwarzISCO --max-high-freq-cutoff SchwarzISCO \
	--segment-length 524288 \
	--low-frequency-cutoff 40.0 --pad-data 8 --num-freq-cutoffs 1 \
	--sample-rate 2048 --high-frequency-cutoff 921.6 --resample-filter ldas \
	--strain-high-pass-atten 0.1 --strain-high-pass-freq 30 \
	--min-total-mass 2.4 --max-total-mass 3.2 \
	--write-compress --spectrum-type median

output = lalapps_tmpltbank.out
error = lalapps_tmpltbank.err
""" + submit_file_tail



print >>open("prune_duplicate_mass_pairs.sub", "w"), r"""
arguments = ../prune_duplicate_mass_pairs.py \
	H1-TMPLTBANK_$(macro_comment)-$(macro_tmpltbank_start_time)-$(macro_tmpltbank_duration).xml.gz tmpltbank.xml.gz

output = prune_duplicate_mass_pairs.out
error = prune_duplicate_mass_pairs.err
""" + submit_file_tail



print >>open("gstlal_inspiral.sub", "w"), r"""
arguments = gstlal_inspiral \
	--verbose \
	--online-data \
	--comment $(macro_comment) \
	--instrument $(macro_instrument) \
	--reference-psd reference_psd.xml.gz \
	--gps-start-time $(macro_gps_start_time) \
	--gps-end-time $(macro_gps_end_time) \
	--template-bank tmpltbank.xml.gz \
	--output gstlal_inspiral.$(macro_instrument).sqlite

output = gstlal_inspiral.$(macro_instrument).out
error = gstlal_inspiral.$(macro_instrument).err
""" + submit_file_tail



print >>open("gstlal_reference_psd.sub", "w"), r"""
arguments = gstlal_reference_psd \
	--verbose \
	--online-data \
	--instrument $(macro_instrument) \
	--write-psd reference_psd.$(macro_instrument).xml.gz \
	--gps-start-time $(macro_gps_start_time) \
	--gps-end-time $(macro_gps_end_time)

output = reference_psd.$(macro_instrument).out
error = reference_psd.$(macro_instrument).err
""" + submit_file_tail



print >>open("8hourly.dag", "w"), (

	"""
	JOB ligo_data_find ligo_data_find.sub
	VARS ligo_data_find macro_tmpltbank_start_time="%(tmpltbank_start_time)d" macro_tmpltbank_end_time="%(tmpltbank_end_time)d"

	JOB lalapps_tmpltbank lalapps_tmpltbank.sub
	VARS lalapps_tmpltbank macro_comment="%(comment)s" macro_tmpltbank_start_time="%(tmpltbank_start_time)d" macro_tmpltbank_end_time="%(tmpltbank_end_time)d"
	PARENT ligo_data_find CHILD lalapps_tmpltbank

	JOB prune_duplicate_mass_pairs prune_duplicate_mass_pairs.sub
	VARS prune_duplicate_mass_pairs macro_comment="%(comment)s" macro_tmpltbank_start_time="%(tmpltbank_start_time)d" macro_tmpltbank_duration="%(tmpltbank_duration)d"
	PARENT lalapps_tmpltbank CHILD prune_duplicate_mass_pairs
	""".replace("\t","")

	+ "".join(
		"""

		JOB gstlal_reference_psd.%(i)s gstlal_reference_psd.sub
		VARS gstlal_reference_psd.%(i)s macro_instrument="%(i)s" macro_gps_start_time="%(s)d" macro_gps_end_time="%(e)d"

		JOB gstlal_inspiral.%(i)s gstlal_inspiral.sub
		VARS gstlal_inspiral.%(i)s macro_instrument="%(i)s" macro_comment="%%(comment)s" macro_gps_start_time="%%(gps_start_time)d" macro_gps_end_time="%%(gps_end_time)d"
		PARENT gstlal_reference_psd.%(i)s prune_duplicate_mass_pairs CHILD gstlal_inspiral.%(i)s
		""".replace("\t","") % args for args in (
			{"i":"H1","s":958739939,"e":958743539},
			{"i":"L1","s":958744974,"e":958747374}
		)
	)

) % {
	"comment": "GSTLAL_8HOURLY",
	"tmpltbank_start_time": 958740096,
	"tmpltbank_end_time": 958742144,
	"tmpltbank_duration": 2048,
	"gps_start_time": gps_start_time,
	"gps_end_time": gps_start_time + 3600
}

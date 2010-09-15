#!/usr/bin/env python
try:
	import sqlite3
except ImportError:
	# pre 2.5.x
	from pysqlite2 import dbapi2 as sqlite3

import datetime
import optparse
import os
import pytz
import sys
import time
import stat
from tempfile import mkstemp

import matplotlib
matplotlib.use('agg')
import pylab
import numpy
import math
from pylal import date
from pylal.xlal.datatypes.ligotimegps import LIGOTimeGPS
from glue.ligolw import utils, lsctables

from gstlal.ligolw_output import effective_snr
from gstlal.gstlal_svd_bank import read_bank
from gstlal.gstlal_reference_psd import read_psd



# Parse command line options

parser = optparse.OptionParser(usage="%prog --www-path /path --input suffix.sqlite ifo1 ifo2 ...")
parser.add_option("--input", "-i", help="suffix of input file (should end in .sqlite)")
parser.add_option("--www-path", "-p", help="path in which to base webpage")
parser.add_option("--skip-slow-plots", "-s", action="store_true", default=False, help="skip plotting the plots that take a long time")
opts, args = parser.parse_args()

if len(args) == 0 or opts.www_path is None:
    parser.print_usage()
    sys.exit()



# Open databases

# FIXME: use network.py's algorithm for filename determination; it is not well
# described
input_path, input_name = os.path.split(opts.input)
coincdb = sqlite3.connect(os.path.join(input_path, "".join(args) + "-" + input_name))
trigdbs = tuple( (ifo, sqlite3.connect(os.path.join(input_path, ifo + "-" + input_name))) for ifo in args)
alldbs = tuple(x[1] for x in trigdbs) + (coincdb,)

# Attach some functions to all databases
for db in alldbs:
	db.create_function("eff_snr", 2, effective_snr)
	db.create_function("sqrt", 1, math.sqrt)
	db.create_function("square", 1, lambda x: x * x)



# Convenience routines

def array_from_cursor(cursor):
	"""Get a Numpy array with named columns from an sqlite query cursor."""
	return numpy.array(cursor.fetchall(), dtype=[(desc[0], float) for desc in cursor.description])

def savefig(fname):
	"""Wraps pylab.savefig, but replaces the destination file atomically and destroys the plotting state."""
	fid, path = mkstemp(suffix = fname, dir = '.')
	pylab.savefig(path)
	os.chmod(path, stat.S_IRGRP | stat.S_IRUSR | stat.S_IROTH)
	os.rename(path, fname)
	os.close(fid)
	pylab.clf()

def to_table(fname, headings, rows):
	print >>open(fname, 'w'), '<table><tr>%s</tr>%s</table>' % (
		''.join('<th>%s</th>' % heading for heading in headings),
		''.join('<tr>%s</tr>' % ''.join('<td>%s</td>' % column for column in row) for row in rows)
	)

# time conversion
gps_start_time = int(coincdb.execute("SELECT value FROM process_params WHERE param='--gps-start-time' AND program='gstlal_inspiral'").fetchone()[0])
gps_end_time = int(coincdb.execute("SELECT value FROM process_params WHERE param='--gps-end-time' AND program='gstlal_inspiral'").fetchone()[0])
tz_dict = {"UTC": pytz.timezone("UTC"), "H": pytz.timezone("US/Pacific"), "L": pytz.timezone("US/Central"), "V": pytz.timezone("Europe/Rome")}
tz_fmt = "%Y-%m-%d %T"
dt_row_headers = ("GPS", "UTC", "Hanford", "Livingston", "Virgo", "Seconds since analysis start", "Seconds until analysis end")
def dt_to_rows(dt):
	"""
	Generate rows suitable for to_table based on a given datetime object.
	"""
	assert dt.tzinfo == tz_dict["UTC"]
	as_gps_int = date.XLALUTCToGPS(dt.timetuple()).seconds
	return ((str(as_gps_int), dt.strftime(tz_fmt),
	        dt.astimezone(tz_dict["H"]).strftime(tz_fmt),
	        dt.astimezone(tz_dict["L"]).strftime(tz_fmt),
	        dt.astimezone(tz_dict["V"]).strftime(tz_fmt),
	        str(as_gps_int - gps_start_time),
	        str(gps_end_time - as_gps_int)),)

wait = 5.0

ifostyle = {"H1": {"color": "red", "label": "H1"}, "L1": {"color": "green", "label": "L1"}, "V1": {"color": "purple", "label": "V1"}}
hist_ifostyle = {"H1": {"facecolor": "red", "label": "H1"}, "L1": {"facecolor": "green", "label": "L1"}, "V1": {"facecolor": "purple", "label": "V1"}}


# Save directory that we were in
old_path = os.getcwd()

# Change to input path to read template bank
if input_path != '':
	os.chdir(input_path)

# Read reference PSDs
# FIXME: figure out a way not to have to hard-code this.
psds = [(ifo, read_psd('reference_psd.%s.xml.gz' % ifo)) for ifo in args]

# Read orthogonal template banks
bankdict = dict((ifo, read_bank('bank.%s.pickle' % ifo)) for ifo in args)

# Read template bank parameters
template_bank_filename = coincdb.execute("SELECT value FROM process_params WHERE param = '--template-bank' LIMIT 1").fetchall()[0][0]
xmldoc = utils.load_filename(template_bank_filename, gz = template_bank_filename.endswith(".gz"))
table = lsctables.table.get_table(xmldoc, lsctables.SnglInspiralTable.tableName)

os.chdir(old_path)
os.chdir(opts.www_path)

pylab.plot(table.get_column('mass1'), table.get_column('mass2'), '.k')
pylab.xlabel('$m_1$ (solar masses)')
pylab.ylabel('$m_2$ (solar masses)')
pylab.title('Template placement by componenent mass')
savefig('tmpltbank_m1_m2.png')

pylab.plot(table.get_column('tau0'), table.get_column('tau3'), '.k')
pylab.xlabel(r'$\tau_0$ (solar masses)')
pylab.ylabel(r'$\tau_3$ (solar masses)')
pylab.title('Template placement by tau0, tau3')
savefig('tmpltbank_tau0_tau3.png')

pylab.plot(table.get_column('mchirp'), table.get_column('mtotal'), '.k')
pylab.xlabel('Chirp mass $\mathcal{M}_\mathrm{chirp}$ (solar masses)')
pylab.ylabel('Total mass $M$ (solar masses)')
pylab.title('Template placement by chirp mass and total mass')
savefig('tmpltbank_mchirp_mtotal.png')

for ifo, bank in bankdict.iteritems():
	ntemplates = 0
	for bf in bank.bank_fragments:
		next_ntemplates = ntemplates + bf.orthogonal_template_bank.shape[0]
		pylab.imshow(
			pylab.log10(abs(bf.orthogonal_template_bank[::-1,:])),
			extent = (bf.end, bf.start, ntemplates, next_ntemplates),
			hold=True, aspect='auto'
		)
		pylab.text(bf.end + bank.filter_length / 30, ntemplates + 0.5 * bf.orthogonal_template_bank.shape[0], '%d Hz' % bf.rate, size='x-small')
		ntemplates = next_ntemplates

	pylab.xlim(0, 1.15*bank.filter_length)
	pylab.ylim(0, 1.05*ntemplates)
	pylab.colorbar().set_label('$\mathrm{log}_{10} |u_{i}(t)|$')
	pylab.xlabel(r"Time $t$ until coalescence (seconds)")
	pylab.ylabel(r"Basis index $i$")
	pylab.title(r"%s orthonormal basis templates $u_{i}(t)$" % ifo)
	savefig('%s_orthobank.png' % ifo)
del bank

for ifo, psd in psds:
	pylab.loglog(
		pylab.arange(len(psd.data))*psd.deltaF + psd.f0,
		pylab.sqrt(psd.data),
		**ifostyle[ifo]
	)
	pylab.xlim(10, 2048);
	pylab.ylim(1e-23, 1e-18);
	pylab.xlabel("Frequency (Hz)")
	pylab.ylabel(r"Amplitude spectral density ($1/\sqrt{\mathrm{Hz}}$)")
	pylab.title(r"%s $h(t)$ spectrum used for singular value decomposition" % ifo)
	savefig('%s_asd.png' % ifo)
for ifo, psd in psds:
	pylab.loglog(
		pylab.arange(len(psd.data))*psd.deltaF + psd.f0,
		pylab.sqrt(psd.data),
		**ifostyle[ifo]
	)
	pylab.xlim(10, 2048);
	pylab.ylim(1e-23, 1e-18);
pylab.xlabel("Frequency (Hz)")
pylab.ylabel(r"Amplitude spectral density ($1/\sqrt{\mathrm{Hz}}$)")
pylab.title(r"$h(t)$ spectrum used for singular value decomposition")
pylab.legend()
savefig('asd.png')
del psd

# Free some memory
# FIXME: scope these things so that they get released automatically.
del xmldoc, table, bankdict, psds

# Write process params stuff options
to_table('processes.html', ('program', 'command-line arguments'),
    ((program, " ".join(" ".join(tup) for tup in coincdb.execute("SELECT param, value FROM process_params WHERE program=?", program))) for program in coincdb.execute("SELECT program FROM process ORDER BY start_time")))

while True:
	start = time.time()
	#
	# Table saying what time various things have happened
	#
	now_dt = datetime.datetime.now(tz_dict["UTC"])
	to_table("page_time.html", dt_row_headers, dt_to_rows(now_dt))

	last_trig_gps = 0
	for ifo, db in trigdbs:
		last_trig_gps = max(last_trig_gps, db.execute("SELECT end_time FROM sngl_inspiral ORDER BY end_time DESC LIMIT 1;").fetchone()[0])
	last_trig_dt = datetime.datetime(*date.XLALGPSToUTC(LIGOTimeGPS(last_trig_gps))[:6] + (0, tz_dict["UTC"]))
	to_table("trig_time.html", dt_row_headers, dt_to_rows(last_trig_dt))

	last_coinc_gps = coincdb.execute("SELECT end_time FROM coinc_inspiral ORDER BY end_time DESC LIMIT 1;").fetchone()
	if last_coinc_gps is not None:
		last_coinc_dt = datetime.datetime(*date.XLALGPSToUTC(LIGOTimeGPS(last_coinc_gps[0]))[:6] + (0, tz_dict["UTC"]))
		to_table("coinc_time.html", dt_row_headers, dt_to_rows(last_coinc_dt))

	# Make single detector plots.
	for ifo, db in trigdbs:
		params = array_from_cursor(db.execute("""
				SELECT snr,chisq,eff_snr(snr,chisq) as eff_snr,end_time+end_time_ns*1e-9 as end_time
				FROM sngl_inspiral ORDER BY event_id DESC LIMIT 10000
			"""))

		# Make per-detector plots
		pylab.figure(1)
	
		pylab.hist(params['snr'], 25, cumulative=-1, log=True, **hist_ifostyle[ifo])
		pylab.xlabel(r"$\rho$")
		pylab.ylabel("Count")
		pylab.title(r"$\rho$ histogram for %s" % ifo)
		savefig("%s_hist_snr.png" % ifo)

		pylab.hist(params['eff_snr'], 25, cumulative=-1, log=True, **hist_ifostyle[ifo])
		pylab.xlabel(r"$\rho_\mathrm{eff}$")
		pylab.ylabel("Count")
		pylab.title(r"$\rho_\mathrm{eff}$ histogram for %s" % ifo)
		savefig("%s_hist_eff_snr.png" % ifo)

		pylab.loglog(params['snr'], params['chisq'], '.', **ifostyle[ifo])
		pylab.xlabel(r"$\rho$")
		pylab.ylabel(r"$\chi^2$")
		pylab.title(r"$\chi^2$ vs. $\rho$ for %s" % ifo)
		savefig('%s_chisq_snr.png' % ifo)

		pylab.loglog(params['eff_snr'], params['chisq'], '.', **ifostyle[ifo])
		pylab.xlabel(r"$\rho_\mathrm{eff}$")
		pylab.ylabel(r"$\chi^2$")
		pylab.title(r"$\chi^2$ vs. $\rho_\mathrm{eff}$ for %s" % ifo)
		savefig('%s_chisq_eff_snr.png' % ifo)

		pylab.semilogy(params['end_time'], params['snr'], '.', **ifostyle[ifo])
		pylab.xlabel("End time")
		pylab.ylabel(r"$\rho$")
		pylab.title(r"$\rho$ vs. end time for %s" % ifo)
		savefig('%s_snr_end_time.png' % ifo)

		pylab.semilogy(params['end_time'], params['eff_snr'], '.', **ifostyle[ifo])
		pylab.xlabel("End time")
		pylab.ylabel(r"$\rho_\mathrm{eff}$")
		pylab.title(r"$\rho_\mathrm{eff}$ vs. end time for %s" % ifo)
		savefig('%s_eff_snr_end_time.png' % ifo)

		# Make overlaid versions
		pylab.figure(2)
		pylab.loglog(params['snr'], params['chisq'], '.', **ifostyle[ifo])
		pylab.figure(3)
		pylab.loglog(params['eff_snr'], params['chisq'], '.', **ifostyle[ifo])
		pylab.figure(4)
		pylab.semilogy(params['end_time'], params['snr'], '.', **ifostyle[ifo])
		pylab.figure(5)
		pylab.semilogy(params['end_time'], params['eff_snr'], '.', **ifostyle[ifo])		

	# Save overlaid versions
	pylab.figure(2)
	pylab.legend()
	pylab.xlabel(r"$\rho$")
	pylab.ylabel(r"$\chi^2$")
	pylab.title(r"$\chi^2$ vs. $\rho$")
	savefig("overlaid_chisq_snr.png")

	pylab.figure(3)
	pylab.legend()
	pylab.xlabel(r"$\rho_\mathrm{eff}$")
	pylab.ylabel(r"$\chi^2$")
	pylab.title(r"$\chi^2$ vs. $\rho_\mathrm{eff}$")
	savefig("overlaid_chisq_eff_snr.png")

	pylab.figure(4)
	pylab.legend()
	pylab.xlabel("End time")
	pylab.ylabel(r"$\rho$")
	pylab.title(r"$\rho$ vs. end time")
	savefig("overlaid_snr_end_time.png")

	pylab.figure(5)
	pylab.legend()
	pylab.xlabel("End time")
	pylab.ylabel(r"$\rho_\mathrm{eff}$")
	pylab.title(r"$\rho_\mathrm{eff}$ vs. end time")
	savefig("overlaid_eff_snr_end_time.png")

	# Make multiple detector plots.
	params = array_from_cursor(coincdb.execute("""
			SELECT
			avg(end_time + 1e-9 * end_time_ns) as mean_end_time,
			sqrt(sum(square(snr))) as combined_snr,
			sqrt(sum(square(eff_snr(snr, chisq)))) as combined_eff_snr
			FROM sngl_inspiral INNER JOIN coinc_event_map USING (event_id) GROUP BY coinc_event_id
		"""))

	pylab.semilogy(params['mean_end_time'], params['combined_snr'], '.k')
	pylab.xlabel('Mean end time')
	pylab.ylabel(r"Combined SNR, $\sqrt{\sum\rho^2}$")
	pylab.title('Combined SNR versus end time')
	savefig('combined_snr_end_time.png')

	pylab.semilogy(params['mean_end_time'], params['combined_eff_snr'], '.k')
	pylab.xlabel('Mean end time')
	pylab.ylabel('Combined effective SNR')
	pylab.title('Combined effective SNR versus end time')
	savefig('combined_eff_snr_end_time.png')

	to_table("trigcount.html", ("count", "ifo"),
		sorted([(db.execute("SELECT count(*) FROM sngl_inspiral").fetchall()[0][0], ifo) for ifo, db in trigdbs], reverse=True))

	to_table("coinccount.html", ("count", "ifos"),
		coincdb.execute("SELECT count(*) as n, ifos FROM coinc_inspiral GROUP BY ifos ORDER BY n DESC").fetchall())

	to_table("loudest.html", ("end_time", "combined_snr", "combined_eff_snr", "ifos", "mchirp", "mtotal"),
		coincdb.execute("""
			SELECT
			coinc_inspiral.end_time + 1e-9 * coinc_inspiral.end_time_ns,
			sqrt(sum(square(sngl_inspiral.snr))) as combined_snr,
			sqrt(sum(square(eff_snr(sngl_inspiral.snr, chisq)))) as combined_eff_snr,
			ifos,
			coinc_inspiral.mchirp,
			mtotal
			FROM sngl_inspiral INNER JOIN coinc_event_map USING (event_id) INNER JOIN coinc_inspiral USING (coinc_event_id) GROUP BY coinc_event_id ORDER BY ifo ASC, combined_eff_snr DESC LIMIT 10
		""").fetchall())

	stop = time.time()

	if (stop - start) < wait: time.sleep(wait - (stop-start))
	print time.time()-start

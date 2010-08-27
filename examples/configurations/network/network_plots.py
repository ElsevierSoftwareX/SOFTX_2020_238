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
import shutil
import sys
import time
import stat
from tempfile import mkstemp

import matplotlib
matplotlib.use('agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pylab
axis_pos = (0.1, 0.1, 0.8, 0.8)
import numpy
import math
from pylal import date

from gstlal.ligolw_output import effective_snr



# Parse command line options

parser = optparse.OptionParser(usage="%prog --www-path /path --input suffix.sqlite ifo1 ifo2 ...")
parser.add_option("--input", "-i", help="suffix of input file (should end in .sqlite)")
parser.add_option("--www-path", "-p", help="path in which to base webpage")
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

os.chdir(opts.www_path)

def to_table(fname, headings, rows):
	print >>open(fname, 'w'), '<!DOCTYPE html><html><body><table><tr>%s</tr>%s</table></body></html>' % (
		''.join('<th>%s</th>' % heading for heading in headings),
		''.join('<tr>%s</tr>' % ''.join('<td>%s</td>' % column for column in row) for row in rows)
	)

cnt = 0
wait = 5.0
#FIXME use glue
lastid = 0
ids = []
snrs = {}
times = {}
chisqs = {}
lines = []
labels = []
effsnrs = {}
Aeffsnrs = {}
Beffsnrs = {}

tz_dict = {"UTC": pytz.timezone("UTC"), "H": pytz.timezone("US/Pacific"), "L": pytz.timezone("US/Central"), "V": pytz.timezone("Europe/Rome")}
fmt = "%Y-%m-%d %T"
ifostyle = {"H1": {"color": "red", "label": "H1"}, "L1": {"color": "green", "label": "L1"}, "V1": {"color": "purple", "label": "V1"}}

while True:
	start = time.time()
	#
	# Table saying what time various things have happened
	#
	now_dt = datetime.datetime.now(tz_dict["UTC"])
	to_table("page_time.html", ("GPS", "UTC", "Hanford", "Livingston", "Virgo"),
		((date.XLALUTCToGPS(now_dt.timetuple()).seconds, now_dt.strftime(fmt),
		now_dt.astimezone(tz_dict["H"]).strftime(fmt),
		now_dt.astimezone(tz_dict["L"]).strftime(fmt),
		now_dt.astimezone(tz_dict["V"]).strftime(fmt)),))

	# Make single detector plots.
	for ifo, db in trigdbs:
		params = array_from_cursor(db.execute("""
				SELECT snr,chisq,eff_snr(snr,chisq) as eff_snr,end_time+end_time_ns*1e-9 as end_time
				FROM sngl_inspiral ORDER BY event_id DESC LIMIT 10000
			"""))

		# Make per-detector plots
		pylab.figure(1)

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

		# Make overlayed versions
		pylab.figure(2)
		pylab.loglog(params['snr'], params['chisq'], '.', **ifostyle[ifo])
		pylab.figure(3)
		pylab.loglog(params['eff_snr'], params['chisq'], '.', **ifostyle[ifo])
		pylab.figure(4)
		pylab.semilogy(params['end_time'], params['snr'], '.', **ifostyle[ifo])
		pylab.figure(5)
		pylab.semilogy(params['end_time'], params['eff_snr'], '.', **ifostyle[ifo])		

	# Save overlayed versions
	pylab.figure(2)
	pylab.legend()
	pylab.xlabel(r"$\rho$")
	pylab.ylabel(r"$\chi^2$")
	pylab.title(r"$\chi^2$ vs. $\rho$")
	savefig("overlayed_chisq_snr.png")

	pylab.figure(3)
	pylab.legend()
	pylab.xlabel(r"$\rho_\mathrm{eff}$")
	pylab.ylabel(r"$\chi^2$")
	pylab.title(r"$\chi^2$ vs. $\rho_\mathrm{eff}$")
	savefig("overlayed_chisq_eff_snr.png")

	pylab.figure(4)
	pylab.legend()
	pylab.xlabel("End time")
	pylab.ylabel(r"$\rho$")
	pylab.title(r"$\rho$ vs. end time")
	savefig("overlayed_snr_end_time.png")

	pylab.figure(5)
	pylab.legend()
	pylab.xlabel("End time")
	pylab.ylabel(r"$\rho_\mathrm{eff}$")
	pylab.title(r"$\rho_\mathrm{eff}$ vs. end time")
	savefig("overlayed_eff_snr_end_time.png")

	#
	# Table of loudest events
	# only do the loudest query every 5 waits
	#

	if (cnt % 6) == 0: to_table('loudest.html' , ["end_time", "end_time_ns", "snr", "ifos", "mchirp", "mass"], coincdb.cursor().execute('SELECT end_time, end_time_ns, snr, ifos, mchirp, mass FROM coinc_inspiral ORDER BY snr DESC LIMIT 10').fetchall())

	# FIXME don't hardcode ifos, don't do the join this way
	# FIXME cant rely on time, somehow has ids too
	query = 'SELECT coinc_inspiral.rowid, snglA.ifo, snglB.ifo, snglA.end_time+snglA.end_time_ns/1e9, snglB.end_time+snglB.end_time_ns/1e9, snglA.snr, snglA.chisq, snglB.snr, snglB.chisq FROM coinc_inspiral JOIN coinc_event_map AS mapA on mapA.coinc_event_id == coinc_inspiral.coinc_event_id JOIN coinc_event_map as mapB on mapB.coinc_event_id == mapA.coinc_event_id JOIN sngl_inspiral AS snglA ON snglA.event_id == mapA.event_id JOIN sngl_inspiral AS snglB ON snglB.event_id == mapB.event_id WHERE mapA.table_name == "sngl_inspiral:table" AND coinc_inspiral.rowid > ?;'
	
	for id, h1ifo, l1ifo, h1time, l1time, h1snr, h1chisq, l1snr, l1chisq in coincdb.cursor().execute(query,(lastid,)):
		ifo = h1ifo+","+l1ifo
		Aeffsnrs.setdefault(ifo,[]).append(effective_snr(h1snr, h1chisq))
		Beffsnrs.setdefault(ifo,[]).append(effective_snr(l1snr, l1chisq))
		snrs.setdefault(h1ifo,[]).append(h1snr)
		snrs.setdefault(l1ifo,[]).append(l1snr)
		times.setdefault(h1ifo,[]).append(h1time)
		times.setdefault(l1ifo,[]).append(l1time)
		chisqs.setdefault(h1ifo,[]).append(h1chisq)
		chisqs.setdefault(l1ifo,[]).append(l1chisq)
		effsnrs.setdefault(h1ifo,[]).append(effective_snr(h1snr,h1chisq))
		effsnrs.setdefault(l1ifo,[]).append(effective_snr(l1snr,l1chisq))
		ids.append(id)

	lastid = max(ids)
	#
	# snr vs time
	#

	fig = Figure()
	ax = fig.add_axes(axis_pos)
	for ifo in times.iterkeys():
		ax.semilogy(times[ifo], snrs[ifo],'.', label=ifo)
	ax.set_xlabel('Time')
	ax.set_ylabel('SNR')
	canvas = FigureCanvas(fig)
	canvas.print_figure('tmpsnr_vs_time.png')
	del fig, ax, canvas
	shutil.move('tmpsnr_vs_time.png', 'snr_vs_time.png')

	#
	# effective snr vs time
	#

	fig = Figure()
	ax = fig.add_axes(axis_pos)
	for ifo in times.iterkeys():
		ax.semilogy(times[ifo], effsnrs[ifo],'.', label=ifo)
	ax.set_ylabel('Effective SNR')
	ax.set_xlabel('Time')
	canvas = FigureCanvas(fig)
	canvas.print_figure('tmpeffsnr_vs_time.png')
	del fig, ax, canvas
	shutil.move('tmpeffsnr_vs_time.png', 'effsnr_vs_time.png')

	for ifo in times.iterkeys():
		#
		# SNR histogram
		#

		fig = Figure()
		ax = fig.add_axes(axis_pos)
		ax.hist(snrs[ifo], 25)
		ax.set_xlabel(ifo + ' SNR')
		ax.set_ylabel('Count')
		canvas = FigureCanvas(fig)
		canvas.print_figure(ifo+'tmpsnr_hist.png')
		del fig, ax, canvas
		shutil.move(ifo+'tmpsnr_hist.png', ifo+'snr_hist.png')

		#
		# Effective SNR histogram
		#

		fig = Figure()
		ax = fig.add_axes(axis_pos)
		ax.hist(effsnrs[ifo], 25)
		ax.set_xlabel(ifo + ' effective SNR')
		ax.set_ylabel('Count')
		canvas = FigureCanvas(fig)
		canvas.print_figure(ifo+'tmpeffsnr_hist.png')
		del fig, ax, canvas
		shutil.move(ifo+'tmpeffsnr_hist.png', ifo+'effsnr_hist.png')

	for ifo in Aeffsnrs.iterkeys():
		#
		# Effective snr scatter plot
		#

		fig = Figure()
		ax = fig.add_axes(axis_pos)
		ax.loglog(Aeffsnrs[ifo], Beffsnrs[ifo], '.', label=ifo)
		ax.set_xlabel('Effective SNR %s' % (ifo.split(',')[0],))
		ax.set_ylabel('Effective SNR %s' % (ifo.split(',')[1],))
		canvas = FigureCanvas(fig)
		canvas.print_figure(ifo+'effsnr_vs_effsnr.png')
		del fig, ax, canvas
		shutil.move(ifo+'effsnr_vs_effsnr.png', ifo+'effsnr_vs_effsnr.png')

	#
	# Chisq vs snr
	#
	fig = Figure()
	ax = fig.add_axes(axis_pos)
	for ifo in times.iterkeys():
		ax.loglog(snrs[ifo], chisqs[ifo], '.', label=ifo)
	ax.set_ylabel('Chi-squared')
	ax.set_xlabel('SNR')
	canvas = FigureCanvas(fig)
	canvas.print_figure('tmpchisq_vs_snr.png')
	del fig, ax, canvas
	shutil.move('tmpchisq_vs_snr.png','chisq_vs_snr.png')

	stop = time.time()

	cnt += 1
	if (stop - start) < wait: time.sleep(wait - (stop-start))
	print time.time()-start

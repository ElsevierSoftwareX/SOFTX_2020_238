#!/usr/bin/python
import sys
try:
        import sqlite3
except ImportError:
        # pre 2.5.x
        from pysqlite2 import dbapi2 as sqlite3

from glue.ligolw import ligolw
from glue.ligolw import lsctables
from glue.ligolw import dbtables
from glue.ligolw import utils
from glue.ligolw import table
from glue import segmentsUtils

from pylal import db_thinca_rings
from pylal import llwapp

from gstlal.ligolw_output import effective_snr

import matplotlib
matplotlib.use('Agg')
import pylab
import sys
import shutil
import time
import numpy

path = '/archive/home/channa/public_html/gstlal_inspiral_online/'

connection = sqlite3.connect(sys.argv[1])
dbtables.DBTable_set_connection(connection)

f = pylab.figure()

def row(f,tup):
	f.write('<tr>')
	for c in tup:
		f.write('<td>%s</td>' % (str(c),))
	f.write('<tr>\n')

def to_table(fname, names, tuplist):
	f = open(fname,"w")
	#f.write('<head><link rel="stylesheet" type="text/css" href="assets/report.css" /></head>')
	f.write('<table border=1>\n')
	row(f,names)
	for tup in tuplist:
		row(f, tup)
	f.write('</table>')
	f.close()

cnt = 0
wait = 5
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

while True:
	f.clf()

	#
	# Table of loudest events
	# only do the loudest query every 5 waits
	#

	if (cnt % 6) == 0: to_table(path+'test.html', ["end_time", "end_time_ns", "snr", "ifos", "mchirp", "mass"], connection.cursor().execute('SELECT end_time, end_time_ns, snr, ifos, mchirp, mass FROM coinc_inspiral ORDER BY snr DESC LIMIT 10').fetchall())

	# FIXME don't hardcode ifos, don't do the join this way
	# FIXME cant rely on time, somehow has ids too
	start = time.time()
	query = 'SELECT coinc_inspiral.rowid, snglA.ifo, snglB.ifo, snglA.end_time+snglA.end_time_ns/1e9, snglB.end_time+snglB.end_time_ns/1e9, snglA.snr, snglA.chisq, snglB.snr, snglB.chisq FROM coinc_inspiral JOIN coinc_event_map AS mapA on mapA.coinc_event_id == coinc_inspiral.coinc_event_id JOIN coinc_event_map as mapB on mapB.coinc_event_id == mapA.coinc_event_id JOIN sngl_inspiral AS snglA ON snglA.event_id == mapA.event_id JOIN sngl_inspiral AS snglB ON snglB.event_id == mapB.event_id WHERE mapA.table_name == "sngl_inspiral:table" AND coinc_inspiral.rowid > ?;'
	
	for id, h1ifo, l1ifo, h1time, l1time, h1snr, h1chisq, l1snr, l1chisq in connection.cursor().execute(query,(lastid,)):
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

	stop = time.time()
	print stop-start
	lastid = max(ids)
	print lastid
	#
	# snr vs time
	#

	pylab.subplot(111)
	for ifo in times.keys():
		lines.append(pylab.semilogy(times[ifo], snrs[ifo],'.', label=ifo))
		labels.append(ifo)
	pylab.xlabel('Time')
	pylab.ylabel('SNR')
	pylab.figlegend(lines,labels,"upper right")
	pylab.savefig(path+'tmpsnr_vs_time.png')
	shutil.move(path+'tmpsnr_vs_time.png',path+'snr_vs_time.png')
	f.clf()

	#
	# effective snr vs time
	#

	pylab.subplot(111)
	for ifo in times.keys():
		lines.append(pylab.semilogy(times[ifo], effsnrs[ifo],'.', label=ifo))
		labels.append(ifo)
	pylab.ylabel('Effective SNR')
	pylab.xlabel('Time')
	pylab.figlegend(lines,labels,"upper right")
	pylab.savefig(path+'tmpeffsnr_vs_time.png')
	shutil.move(path+'tmpeffsnr_vs_time.png',path+'effsnr_vs_time.png')

	f.clf()
	
	#
	# SNR histogram
	#

	for ifo in times.keys():
		pylab.subplot(111)
		pylab.hist(snrs[ifo],25)
		pylab.xlabel(ifo + ' SNR')
		pylab.ylabel('Count')
		pylab.savefig(path+ifo+'tmpsnr_hist.png')
		shutil.move(path+ifo+'tmpsnr_hist.png',path+ifo+'snr_hist.png')
		f.clf()

	for ifo in times.keys():
		pylab.subplot(111)
		pylab.hist(effsnrs[ifo],25)
		pylab.xlabel(ifo + ' effective SNR')
		pylab.ylabel('Count')
		pylab.savefig(path+ifo+'tmpeffsnr_hist.png')
		shutil.move(path+ifo+'tmpeffsnr_hist.png',path+ifo+'effsnr_hist.png')
		f.clf()

	#
	# Chisq vs snr
	#
	pylab.subplot(111)
	for ifo in times.keys():
		pylab.loglog(snrs[ifo], chisqs[ifo],'.', label=ifo)
	pylab.ylabel('Chi-squared')
	pylab.xlabel('SNR')
	pylab.figlegend(lines,labels,"upper right")
	pylab.savefig(path+'tmpchisq_vs_snr.png')
	shutil.move(path+'tmpchisq_vs_snr.png',path+'chisq_vs_snr.png')
	f.clf()

	#
	# Effective snr scatter plot
	#

	pylab.subplot(111)
	for ifo in Aeffsnrs.keys():
		pylab.loglog(Aeffsnrs[ifo], Beffsnrs[ifo],'.', label=ifo)
		pylab.xlabel('Effective SNR %s' % (ifo.split(',')[0],))
		pylab.ylabel('Effective SNR %s' % (ifo.split(',')[1],))
		pylab.figlegend(lines,labels,"upper right")
		pylab.savefig(path+ifo+'effsnr_vs_effsnr.png')
		shutil.move(path+ifo+'effsnr_vs_effsnr.png',path+ifo+'effsnr_vs_effsnr.png')
		f.clf()

	cnt += 1
	time.sleep(wait)


connection.close()


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
wait = 10
while True:
	f.clf()
	snrs = {}
	times = {}
	chisqs = {}
	lines = []
	labels = []
	effsnrs = {}

	# only do the loudest query every 5 waits
	if (cnt % 6) == 0: to_table(path+'test.html', ["end_time", "end_time_ns", "snr", "ifos", "mchirp", "mass"], connection.cursor().execute('SELECT end_time, end_time_ns, snr, ifos, mchirp, mass FROM coinc_inspiral ORDER BY snr DESC LIMIT 10').fetchall())

	for snr, chisq, t, ifo in connection.cursor().execute('SELECT snr, chisq, end_time+end_time_ns*1e-9, ifo FROM sngl_inspiral'):
		snrs.setdefault(ifo,[]).append(snr)
		times.setdefault(ifo,[]).append(t)
		chisqs.setdefault(ifo,[]).append(chisq)
		effsnrs.setdefault(ifo,[]).append(effective_snr(snr,chisq))

	#
	# snr vs time
	#

	pylab.subplot(111)
	csnrs = {}
	ctimes = {}
	for snr, t, ifo in connection.cursor().execute('SELECT snr, end_time+end_time_ns*1e-9, ifos FROM coinc_inspiral'):
		csnrs.setdefault(ifo,[]).append(snr)
		ctimes.setdefault(ifo,[]).append(t)
	for ifo in ctimes.keys():
		lines.append(pylab.semilogy(ctimes[ifo], csnrs[ifo],'.', label=ifo))
		labels.append(ifo)
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

	cnt += 1
	time.sleep(wait)


connection.close()


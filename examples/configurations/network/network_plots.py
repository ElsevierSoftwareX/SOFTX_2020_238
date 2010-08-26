#!/usr/bin/python
try:
	import sqlite3
except ImportError:
	# pre 2.5.x
	from pysqlite2 import dbapi2 as sqlite3

import os
import sys
import shutil
import time
import numpy
import optparse

import matplotlib
matplotlib.use('Agg')
import pylab
from glue.ligolw import dbtables
from glue import segmentsUtils

from gstlal.ligolw_output import effective_snr


parser = optparse.OptionParser(usage="%prog --www-path /path --input suffix.sqlite ifo1 ifo2 ...")
parser.add_option("--input", "-i", help="suffix of input file (should end in .sqlite)")
parser.add_option("--www-path", "-p", help="path in which to base webpage")
opts, args = parser.parse_args()

if len(args) == 0 or opts.www_path is None:
    parser.print_usage()
    sys.exit()

# FIXME: use network.py's algorithm for filename determination; it is not well
# described
input_path=os.path.split(opts.input)[0]
input_name=os.path.split(opts.input)[1]
input_filename = os.path.join(input_path, "".join(args) + "-" + input_name)

connection = sqlite3.connect(input_filename)
dbtables.DBTable_set_connection(connection)

os.chdir(opts.www_path)

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

while True:
	f.clf()

	#
	# Table of loudest events
	# only do the loudest query every 5 waits
	#

	if (cnt % 6) == 0: to_table('test.html' , ["end_time", "end_time_ns", "snr", "ifos", "mchirp", "mass"], connection.cursor().execute('SELECT end_time, end_time_ns, snr, ifos, mchirp, mass FROM coinc_inspiral ORDER BY snr DESC LIMIT 10').fetchall())

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

	lastid = max(ids)
	#
	# snr vs time
	#

	pylab.subplot(111)
	for ifo in times.keys():
		lines.append(pylab.semilogy(numpy.array(times[ifo]), numpy.array(snrs[ifo]),'.', label=ifo))
	pylab.xlabel('Time')
	pylab.ylabel('SNR')
	pylab.savefig('tmpsnr_vs_time.png')
	shutil.move('tmpsnr_vs_time.png', 'snr_vs_time.png')
	f.clf()

	#
	# effective snr vs time
	#

	pylab.subplot(111)
	for ifo in times.keys():
		lines.append(pylab.semilogy(numpy.array(times[ifo]), numpy.array(effsnrs[ifo]),'.', label=ifo))
	pylab.ylabel('Effective SNR')
	pylab.xlabel('Time')
	pylab.savefig('tmpeffsnr_vs_time.png')
	shutil.move('tmpeffsnr_vs_time.png','effsnr_vs_time.png')

	f.clf()
	
	#
	# SNR histogram
	#

	for ifo in times.keys():
		pylab.subplot(111)
		pylab.hist(numpy.array(snrs[ifo]),25)
		pylab.xlabel(ifo + ' SNR')
		pylab.ylabel('Count')
		pylab.savefig(ifo+'tmpsnr_hist.png')
		shutil.move(ifo+'tmpsnr_hist.png',ifo+'snr_hist.png')
		f.clf()

	for ifo in times.keys():
		pylab.subplot(111)
		pylab.hist(numpy.array(effsnrs[ifo]),25)
		pylab.xlabel(ifo + ' effective SNR')
		pylab.ylabel('Count')
		pylab.savefig(ifo+'tmpeffsnr_hist.png')
		shutil.move(ifo+'tmpeffsnr_hist.png',ifo+'effsnr_hist.png')
		f.clf()

	#
	# Chisq vs snr
	#
	pylab.subplot(111)
	for ifo in times.keys():
		pylab.loglog(numpy.array(snrs[ifo]), numpy.array(chisqs[ifo]),'.', label=ifo)
	pylab.ylabel('Chi-squared')
	pylab.xlabel('SNR')
	pylab.savefig('tmpchisq_vs_snr.png')
	shutil.move('tmpchisq_vs_snr.png','chisq_vs_snr.png')
	f.clf()

	#
	# Effective snr scatter plot
	#

	pylab.subplot(111)
	for ifo in Aeffsnrs.keys():
		pylab.loglog(numpy.array(Aeffsnrs[ifo]), numpy.array(Beffsnrs[ifo]),'.', label=ifo)
		pylab.xlabel('Effective SNR %s' % (ifo.split(',')[0],))
		pylab.ylabel('Effective SNR %s' % (ifo.split(',')[1],))
		pylab.savefig(ifo+'effsnr_vs_effsnr.png')
		shutil.move(ifo+'effsnr_vs_effsnr.png',ifo+'effsnr_vs_effsnr.png')
		f.clf()
	stop = time.time()

	cnt += 1
	if (stop - start) < wait: time.sleep(wait - (stop-start))
	print time.time()-start


connection.close()


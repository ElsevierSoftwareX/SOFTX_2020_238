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
import time
import numpy

path = '/archive/home/channa/public_html/gstlal_inspiral_online/'

connection = sqlite3.connect(sys.argv[1])
dbtables.DBTable_set_connection(connection)

f = pylab.figure()

while True:
	f.clf()
	snrs = {}
	times = {}
	chisqs = {}
	lines = []
	labels = []
	effsnrs = {}
	for snr, chisq, t, ifo in connection.cursor().execute('SELECT snr, chisq, end_time+end_time_ns*1e-9, ifo FROM sngl_inspiral'):
		snrs.setdefault(ifo,[]).append(snr)
		times.setdefault(ifo,[]).append(t)
		chisqs.setdefault(ifo,[]).append(chisq)
		effsnrs.setdefault(ifo,[]).append(effective_snr(snr,chisq))
		
	pylab.subplot(211)
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

	pylab.subplot(212)
	for ifo in times.keys():
		lines.append(pylab.semilogy(times[ifo], effsnrs[ifo],'.', label=ifo))
		labels.append(ifo)
		#pylab.loglog(snrs[ifo], chisqs[ifo],'.', label=ifo)
	pylab.ylabel('Effective SNR')
	pylab.xlabel('Time')
	pylab.figlegend(lines,labels,"upper right")
	pylab.savefig(path+'online.png')
	f.clf()

	pylab.subplot(121)
	for ifo in times.keys():
		pylab.loglog(snrs[ifo], chisqs[ifo],'.', label=ifo)
	pylab.ylabel('Effective SNR')
	pylab.xlabel('Time')
	#pylab.figlegend(lines,labels,"upper right")
	pylab.savefig(path+'snr_chisq.png')
		
	time.sleep(5)

	
connection.close()


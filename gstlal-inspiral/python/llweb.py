#!/usr/bin/env python
#
# Copyright (C) 2011  Chad Hanna
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

## @file

## @package llweb
#
# ### Review Status
#
# | Names                                              | Hash                                        | Date       | Diff to Head of Master      |
# | ---------------------------------------------------| ------------------------------------------- | ---------- | --------------------------- |
# | Florent, Sathya, Sarah Caudill, Jolien, Kipp, Chad | bd05ad3ba0617073f48d9be97b8784d8ab18ddb8    | 2015-09-11 | <a href="@gstlal_inspiral_cgit_diff/python/llweb.py?id=bd05ad3ba0617073f48d9be97b8784d8ab18ddb8">llweb.py</a> |
#
# #### Actions
#
# - This code has very little commenting and documenting in general.
# - On line 123, the tabs formatting does not work for small windows. They end up overlapping and being unreadable.
# - On line 171, the function now() can be updated or deprecated. Refer to comment above in gstlalcbcsummary.
# - There are FIXMEs for dealing with the 16Hz sample rate for LIGO vs the 1Hz sample rate for Virgo.
# - There is a FIXME for hardcoded IFOs H1 and L1.
#
# #### Complete Actions
#
# - On line 237, the function status() is defined to populate the status in the summary webpage header. You should probably have a second look at this function. In particular, it has a problem in that it will never catch the "MORE THAN 5 MIN BEHIND" case because the "MORE THAN 3 MIN BEHIND" case will catch it first.

import sys
import cgi
import cgitb
cgitb.enable()
import os
import bisect
import math
import json
import re

# This will run on a web server
os.environ["MPLCONFIGDIR"] = "/tmp"
import matplotlib
matplotlib.rcParams.update({
        "text.usetex": True,
})
matplotlib.use('Agg')
import numpy
import matplotlib.pyplot as plt
import time
from io import StringIO
import base64
import urllib.parse as urlparse

import lal
from ligo.lw import ligolw
from ligo.lw import utils as ligolw_utils
from gstlal import far
from gstlal import plotpsd
from gstlal import plotfar
from gstlal import plotsegments


def ceil10(x):
	return 10**math.ceil(math.log10(x))

def floor10(x):
	return 10**math.floor(math.log10(x))

css = """
<style type="text/css">

a {
	font: 12pt arial, sans-serif;
}

.big {
	font: 18pt arial, sans-serif;
}

hr {
	border: 0;
	height: 0;
	box-shadow: 0 0 4px 1px black;
}

em.red {
	font: 18pt arial, sans-serif;
	color: red
}

em.green {
	font: 18pt arial, sans-serif;
	color: green
}

table {
	font: 14pt arial, sans-serif;
	width: 90%;
}

th {
	text-align: left;
	border: 2pt outset;
}

td {
	padding: 3px;
}

.topbox {
	font: 14pt arial, sans-serif;
	border-radius: 25px;
	border: 2px solid gray;
	background: white;
	padding: 20px;
	width: 97%;
	box-shadow: 10px 10px 5px #888888;
}

.tabs {
  position: relative;
  min-height: 10in; /* This part sucks */
  clear: both;
  margin: 25px 0;
}

.tab {
  float: left;
}

.tab label {
  background: #eee;
  padding: 10px;
  border: 1px solid #ccc;
  margin-left: -1px;
  position: relative;
  left: 1px;
}

.tab [type=radio] {
  display: none;
}

.content {
  position: absolute;
  top: 28px;
  left: 0;
  background: white;
  right: 0;
  bottom: 0;
  padding: 20px;
  border: 1px solid #ccc;
}

[type=radio]:checked ~ label {
  background: white;
  border-bottom: 1px solid white;
  z-index: 2;
}

[type=radio]:checked ~ label ~ .content {
  z-index: 1;
}

</style>
"""

def now():
	#FIXME use lal when available
	return time.time() - 315964785

class GstlalWebSummary(object):
	def __init__(self, form):
		self.form = form
		self.directory = form["dir"][0]
		self.ifos = form["ifos"][0].split(",")
		self.registry = {}
		for id in [ '%04d' % (job,) for job in range(int(form["id"][0].split(",")[0]), 1+int(form["id"][0].split(",")[1]))]:
			url = '%s/%s%s' % (self.directory, id, "_registry.txt")
			try:
				with open(url,"r") as tmp:
					self.registry[id] = urlparse.urlparse(tmp.readline()).netloc
			except IOError:
				self.registry[id] = None

		# Class to automatically load data if it is not present
		class Data(dict):
			def __init__(s, *args, **kwargs):
				dict.__init__(s, *args, **kwargs)
			def __getitem__(s, k, web = self):
				if k not in s:
					web.load_data(k)
				return s.get(k)

		self.found = Data()
		self.missed = Data()
		self.now = now()

	def load_data(self, datatype):
		self.found[datatype] = {}; self.missed[datatype] = {}
		for id in sorted(self.registry):
			fname = "%s/%s/%s" % (self.directory, id, datatype)
			if datatype == "psds":
				try:
					self.found[datatype][id] = lal.series.read_psd_xmldoc(ligolw_utils.load_url("%s.xml" % fname, contenthandler = lal.series.PSDContentHandler))
				except KeyError:
					self.missed[datatype][id] = {}
			elif datatype == "likelihood":
				try:
					self.found[datatype][id] = far.parse_likelihood_control_doc(ligolw_utils.load_filename("%s.xml" % fname, contenthandler = far.RankingStat.LIGOLWContentHandler))
				except KeyError:
					self.missed[datatype][id] = {}
			elif datatype == "cumulative_segments":
				try:
					self.found[datatype][id] = plotsegments.parse_segments_xml("%s.xml" % fname)
				except KeyError:
					self.missed[datatype][id] = {}
			elif datatype == "marginalized_likelihood":
				fname = "%s/%s" % (self.directory, datatype)
				try:
					self.found[datatype] = far.parse_likelihood_control_doc(ligolw_utils.load_filename("%s.xml.gz" % fname, contenthandler = far.RankingStat.LIGOLWContentHandler))
				except KeyError:
					self.missed[datatype] = {}
			else:
				try:
					self.found[datatype][id] = numpy.loadtxt("%s.txt" % fname)
					if len(self.found[datatype][id].shape) == 1:
						self.found[datatype][id] = numpy.array([self.found[datatype][id],])
				except IOError:
					self.missed[datatype][id] = numpy.array([])
				except ValueError:
					self.missed[datatype][id] = numpy.array([])

	def status(self):
		valid_latency = self.valid_latency()
		if self.oldest_data() > 1800:
			return 2, "<em class=red>SOME DATA OLDER THAN %d seconds</em>" % self.oldest_data()
		if not valid_latency:
			return 1, "<em class=red>NO COINCIDENT EVENTS FOUND!</em>"
		if self.missed["latency_history"]:
			return 3, "<em class=red>%s NODES ARE NOT REPORTING!</em>" % len(self.missed["latency_history"])
		lat = [l for l in valid_latency if l > 300]
		if lat:
			return 2, "<em class=red>%s NODES ARE MORE THAN 5 MIN BEHIND!</em>" % len(lat)
		return 0, "<em class=green>OK</em>"

	def nagios(self):
		print >>sys.stdout, 'Cache-Control: no-cache, must-revalidate'
		print >>sys.stdout, 'Expires: Mon, 26 Jul 1997 05:00:00 GMT'
		print >>sys.stdout, 'Content-type: text/json\r\n'
		num, txt = self.status()
		print >>sys.stdout, json.dumps({"nagios_shib_scraper_ver": 0.1, "status_intervals":[{"num_status": num, "txt_status": re.sub('<[^<]+?>', '', txt)}]}, sort_keys=True, indent=4, separators=(',', ': '))

	def valid_latency(self):
		out = []
		for l in self.found["latency_history"].values():
			if l.shape != (1,0):
				out.append(l[-1,1])
			else:
				out.append(float("inf"))
		return out

	def valid_time_since_last(self):
		out = []
		for l in self.found["latency_history"].values():
			if l.shape != (1,0):
				out.append(now() - l[-1,0])
			else:
				out.append(float("inf"))
			return out

	def latency(self):
		out = self.valid_latency()
		return "%.2f &pm; %.2f" % (numpy.mean(out), numpy.std(out))

	def time_since_last(self):
		out = self.valid_time_since_last()
		return "%.2f &pm; %.2f" % (numpy.mean(out), numpy.std(out))

	def average_up_time(self):
		out = {}
		for ifo in self.ifos:
			# FIXME a hack to deal with 16 Hz sample rate for LIGO
			# statevector and 1 Hz for Virgo
			# FIXME this should go in gstlal proper.
			if ifo != "V1":
				fac = 16.
			else:
				fac = 1.
			out[ifo] = [sum(l[0,1:4])/fac for l in self.found["%s/state_vector_on_off_gap" % ifo].values()]
		return "<br>".join(["%s=%.0f s" % (ifo, numpy.mean(v)) for ifo,v in out.items()])

	def oldest_data(self):
		out = 0.
		for ifo in self.ifos:
			# FIXME a hack to deal with 16 Hz sample rate for LIGO
			# statevector and 1 Hz for Virgo
			# FIXME this should go in gstlal proper.
			if ifo != "V1":
				fac = 16.
			else:
				fac = 1.
			x = [self.now - l[0,0] for l in self.found["%s/state_vector_on_off_gap" % ifo].values()]
			out = max(out, max(x))
		return out

	def setup_plot(self):
		fig = plt.figure(figsize=(15, 4.0),)
		fig.patch.set_alpha(0.0)
		h = fig.add_subplot(111, axisbg = 'k')
		plt.subplots_adjust(top = 0.93, left = .062, right = 0.98, bottom = 0.45)
		return fig, h

	def finish_plot(self, ylim):
		plt.grid(color=(0.1,0.4,0.5), linewidth=2)
		ticks = ["%s : %s " % (id, reg) for (id, reg) in sorted(self.registry.items())]
		plt.xticks(numpy.arange(len(ticks))+.3, ticks, rotation=90, fontsize = 10)
		plt.xlim([0, len(ticks)])
		plt.ylim(ylim)
		tickpoints = numpy.linspace(ylim[0], ylim[1], 8)
		ticks = ["%.1e" % (10.**t,) for t in tickpoints]
		plt.yticks(tickpoints, ticks)
		f = StringIO.StringIO()
		plt.savefig(f, format="png")
		out = '<img src="data:image/png;base64,' + base64.b64encode(f.getvalue()) + '"></img>'
		f.close()
		return out

	def to_png(self, fig = None):
		f = StringIO.StringIO()
		if fig:
			fig.savefig(f, format="png")
		else:
			plt.savefig(f, format="png")
		out = '<img src="data:image/png;base64,' + base64.b64encode(f.getvalue()) + '"></img>'
		f.close()
		return out

	def plot(self, datatype, ifo = None):
		if "marginalized_likelihood" not in datatype:
			fig, h = self.setup_plot()
		found = self.found[datatype]
		missed = self.missed[datatype]
		if datatype == "latency_history":
			return self.plot_latency(fig, h, found, missed)
		if datatype == "snr_history":
			return self.plot_snr(fig, h, found, missed)
		if "state_vector" in datatype:
			return self.plot_livetime(fig, h, found, missed, ifo)
		if "bank" in datatype:
			return self.plot_single_col(fig, h, found, missed, col = 2, title = "Chirp Mass")
		if "ram_history" in datatype:
			return self.plot_ram(fig, h, found, missed)
		if "marginalized_likelihood" in datatype:
			return self.plot_likelihood_ccdf(found, missed)

	def plot_latency(self, fig, h, found, missed):
		found_x = range(len(found))
		found_y = numpy.log10(numpy.array([found[k][-1,1] for k in sorted(found)]))
		time_y = numpy.log10(now() - numpy.array([found[k][-1,0] for k in sorted(found)]))
		try:
			max_y = max(time_y.max(), found_y.max())
		except ValueError:
			max_y = 1
		missed_x = range(len(missed))
		missed_y = numpy.ones(len(missed_x)) * max_y

		h.bar(missed_x, missed_y, color='r', alpha=0.9, linewidth=2)
		h.bar(found_x, found_y, color='w', alpha=0.9, linewidth=2)
		h.bar(found_x, time_y, color='w', alpha=0.7, linewidth=2)
		plt.title("Time (s) since last event (gray) and latency (white)")
		return self.finish_plot([0, max_y])

	def plot_snr(self, fig, h, found, missed):
		found_x = range(len(found))
		maxsnr_y = numpy.log10(numpy.array([found[k][:,1].max() for k in sorted(found)]))
		mediansnr_y = numpy.log10(numpy.array([numpy.median(found[k][:,1]) for k in sorted(found)]))

		try:
			max_y = max(maxsnr_y)
		except ValueError:
			max_y = 1
		missed_x = range(len(missed))
		missed_y = numpy.ones(len(missed_x)) * max_y

		h.bar(missed_x, missed_y, color='r', alpha=0.9, linewidth=2)
		h.bar(found_x, mediansnr_y, color='w', alpha=0.9, linewidth=2)
		h.bar(found_x, maxsnr_y, color='w', alpha=0.7, linewidth=2)
		plt.title("SNR of last 1000 events: max (gray) and median (white)")
		return self.finish_plot([numpy.log10(5.5), max_y])


	def plot_livetime(self, fig, h, found, missed, ifo):
		found_x = range(len(found))
		# Handle log of 0 by setting it to max of (actual value, 1)
		on_y = numpy.log10(numpy.array([max(found[k][0][1],1) for k in sorted(found)]))
		off_y = numpy.log10(numpy.array([max(found[k][0][2],1) for k in sorted(found)]))
		gap_y = numpy.log10(numpy.array([max(found[k][0][3],1) for k in sorted(found)]))
		# FIXME Hack to adjust for high sample rate L1 and H1 state vector
		if ifo != "V1":
			on_y -= numpy.log10(16)
			off_y -= numpy.log10(16)
			gap_y -= numpy.log10(16)

		if len(found_x) > 0:
			max_y = max(on_y.max(), off_y.max(), gap_y.max())
			min_y = min(on_y.min(), off_y.min(), gap_y.min())
		else:
			max_y = 1
			min_y = 0

		missed_x = range(len(missed))
		missed_y = numpy.ones(len(missed_x)) * max_y

		h.bar(missed_x, missed_y, color='r', alpha=0.9, linewidth=2)
		h.bar(found_x, off_y, color='w', alpha=0.7, linewidth=2)
		h.bar(found_x, gap_y, color='b', alpha=0.5, linewidth=2)
		h.bar(found_x, on_y, color='w', alpha=0.5, linewidth=2)
		plt.title('%s Up time (gray) Down time (white) Dropped time (blue)' % (ifo,))
		return self.finish_plot([min_y*.9, max_y])

	def plot_single_col(self, fig, h, found, missed, col = 2, title = ''):
		found_x = range(len(found))
		found_y = numpy.log10(numpy.array([found[k][0][col] for k in sorted(found)]))

		try:
			max_y, min_y = max(found_y), min(found_y)
		except ValueError:
			max_y, min_y = (1,0)
		missed_x = range(len(missed))
		missed_y = numpy.ones(len(missed_x)) * max_y

		h.bar(missed_x, missed_y, color='r', alpha=0.9, linewidth=2)
		h.bar(found_x, found_y, color='w', alpha=0.9, linewidth=2)
		plt.title(title)
		return self.finish_plot([0.9 * min_y, max_y])

	def plot_ram(self, fig, h, found, missed):

		found_x = range(len(found))
		found_y = numpy.log10(numpy.array([found[k][0,1] for k in sorted(found)]))

		try:
			max_y, min_y = max(found_y), min(found_y)
		except ValueError:
			max_y, min_y = (1,0)
		missed_x = range(len(missed))
		missed_y = numpy.ones(len(missed_x)) * max_y

		h.bar(missed_x, missed_y, color='r', alpha=0.9, linewidth=2)
		h.bar(found_x, found_y, color='w', alpha=0.9, linewidth=2)
		plt.title("max RAM usage (GB)")
		return self.finish_plot([0.9 * min_y, max_y])

	def plot_likelihood_ccdf(self, found, missed):
		likelihood, ranking_data, nu = found
		fapfar = far.FAPFAR(ranking_data)
		fig = plotfar.plot_likelihood_ratio_ccdf(fapfar, (0., 40.))
		f = StringIO.StringIO()
		fig.savefig(f, format="png")
		out = '<img src="data:image/png;base64,' + base64.b64encode(f.getvalue()) + '"></img>'
		f.close()
		return out

	#
	# Single Node plots
	#

	def livetime_pie(self):
		out = ""
		for id in self.registry:
			for ifo in self.ifos:
				fig = plt.figure(figsize=(5,3),)
				fig.patch.set_alpha(0.0)
				h = fig.add_subplot(111, axisbg = 'k', aspect='equal')
				plt.subplots_adjust(bottom = 0, left = .25, top = 1, right = .75)
				plt.grid(color="w")

				discontdata = self.found["%s/strain_add_drop" % ifo][id]
				livetimedata = self.found["%s/state_vector_on_off_gap" % ifo][id]
				dt = livetimedata[0,2]
				lt = livetimedata[0,1]
				discont = discontdata[0,1]
				# FIXME Hack to adjust for high sample rate L1 and H1 state vector
				if "V1" not in ifo:
					dt /= 16
					lt /= 16
					discont /= 16
				data = [dt, lt, discont]
				explode = [0.0, 0, 0.15]
				labels = ["OFF : %g (s)" % dt, "ON : %g (s)" % lt, "MIA : %g (s)" % discont]

				h.pie(data, shadow=True, explode = explode, labels = labels, autopct='%1.1f%%', colors = ('0.5', '1.0', (0.7, 0.7, 1.)))

				plt.title(ifo)
				out += self.to_png()
		return out

	def psdplot(self, fmin = 10., fmax = 2048.):
		out = ""
		for id in self.registry:
			psds = self.found["psds"][id]
			fig = plotpsd.plot_psds(psds)
			out += self.to_png(fig = fig)
		return out

	def snrchiplot(self, binnedarray_string):
		out = ""
		for id in self.registry:
			likelihood, nu, nu = self.found["likelihood"][id]
			for ifo in self.ifos:
				fig = plotfar.plot_snr_chi_pdf(likelihood, ifo, binnedarray_string, 400)
				out += self.to_png(fig = fig)
		return out

	def jointsnrplot(self):
		out = ""
		for id in self.registry:
			likelihood, nu, nu = self.found["likelihood"][id]
			# FIXME dont hardcode IFOs and min_instruments
			instruments, min_instruments = (u"H1",u"L1"), 2
			fig = plotfar.plot_snr_joint_pdf(likelihood, instruments, likelihood.horizon_history.getdict(lal.GPSTimeNow()), min_instruments, 200.)
			out += self.to_png(fig = fig)
		return out

	def rateplot(self):
		out = ""
		for id in self.registry:
			likelihood, ranking_data, nu = self.found["likelihood"][id]
			fig = plotfar.plot_rates(likelihood)
			out += self.to_png(fig = fig)
		return out


	def plothistory(self, dataurl, xlabel = "", ylabel = "", title = ""):
		out = ""
		for id in self.registry:
			try:
				data = self.found[dataurl][id]
			except KeyError:
				out += "<em>Data not found</em>"
				continue
			fig = plt.figure(figsize=(5,3.5),)
			fig.patch.set_alpha(0.0)
			h = fig.add_subplot(111, axisbg = 'k')
			plt.subplots_adjust(bottom = 0.2, left = .16)
			plt.grid(color=(0.1,0.4,0.5), linewidth=2)

			h.semilogy(data[:,0] - data[-1,0], data[:,1], 'w', alpha=0.75, linewidth=2)
			plt.ylim([min(data[:,1]), max(data[:,1])])
			locs = [min(data[:,1]), numpy.median(data[:,1]), max(data[:,1])]
			labels = ['%.2g' % lab for lab in locs]
			plt.yticks(locs, labels)
			plt.xlabel(xlabel)
			plt.ylabel(ylabel)
			plt.title(title)
			out += self.to_png()
		return out

	def plotccdf(self):
		out = ""
		for id in self.registry:
			likelihood, ranking_data, nu = self.found["likelihood"][id]
			fapfar = far.FAPFAR(ranking_data)
			fig = plotfar.plot_likelihood_ratio_ccdf(fapfar, (0., 40.))
			out += self.to_png(fig = fig)
		return out

	def plotcumulativesegments(self):
		out = ""
		for id in self.registry:
			likelihood, ranking_data, nu = self.found["likelihood"][id]
			seglistdicts = self.found["cumulative_segments"][id]
			fig, h = plotsegments.plot_segments_history(seglistdicts)
			out += self.to_png(fig = fig)
		return out

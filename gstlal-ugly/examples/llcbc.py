#!/usr/bin/python
import sys
import cgi
import cgitb
import os
os.environ["MPLCONFIGDIR"] = "/tmp"
import matplotlib
matplotlib.use('Agg')
import numpy
# FIXME don't import this when it is "real"
from numpy import random
import matplotlib.pyplot as plt
cgitb.enable()
form = cgi.FieldStorage()

def plot(dataurl, plottype, xlabel = "", ylabel = "", title = ""):
	fig = plt.figure(figsize=(5,3.5),)
	fig.patch.set_alpha(0.0)
	h = fig.add_subplot(111, axisbg = 'k')
	plt.subplots_adjust(bottom = 0.2, left = .16)
	plt.grid(color="w")

	if plottype == "plot":
		data = numpy.loadtxt(dataurl)
		h.plot(data[:,0], data[:,1], 'w', alpha=0.75, linewidth=2)
	if plottype == "loglog":
		data = numpy.loadtxt(dataurl)
		h.loglog(data[:,0], data[:,1], 'w', alpha=0.75, linewidth=2)
	if plottype == "hist":
		data = numpy.loadtxt(dataurl)
		h.fill_between(data[:,0], data[:,1],  alpha=0.75, linewidth=2, facecolor="w", color="w")
	if plottype == "pie":
		# do real thing here
		plt.subplots_adjust(bottom = 0.13, top = 1.0)
		h.pie(10 * random.rand(4), shadow=True, labels = ("a","b","c","d"))

	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.savefig(sys.stdout, format="svg")


if "dir" not in form:
	raise ValueError("must specify dir")
if "id" not in form:
	raise ValueError("must specify id")

baseurl = '%s/%s_' % (form.getvalue("dir"), form.getvalue("id"))

print('Cache-Control: no-cache, must-revalidate', file=sys.stdout)
print('Expires: Mon, 26 Jul 1997 05:00:00 GMT', file=sys.stdout)
print('Content-type: text/html\r\n', file=sys.stdout)

print("""
<html>
<head>
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Expires" content="-1">
<meta http-equiv="CACHE-CONTROL" content="NO-CACHE">
</head>
<body bgcolor=#E0E0E0>
<table face="Arial">
<tr>
<td><div id='canvasa'>
""")

plot(baseurl+'latency_history.txt', "plot", ylabel = "Latency (s)", xlabel = "Time since last trigger (s)")

print("""
</div></td>
<td><div id='canvasb'>
""")

plot(baseurl+'latency_histogram.txt', "hist", ylabel = "Count", xlabel = "Latency (s)")


print("""
</div></td>
<tr>
<td><div id='canvasc'>
""")

plot(baseurl+'snr_history.txt', "plot", ylabel = "SNR", xlabel = "Time since last trigger (s)")


print("""
</div></td>
<td><div id='canvasd'></div></td>
</table>
</body>
""")



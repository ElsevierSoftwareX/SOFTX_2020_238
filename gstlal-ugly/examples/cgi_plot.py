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

# Must be the first line
print('Content-type: image/svg+xml\r\n', file=sys.stdout)

if "type" not in form:
	raise ValueError("must specify type in url, eg. type=plot or type=loglog or type=pie or type=hist")

plottype = form.getvalue("type")

fig = plt.figure(figsize=(5,3.5),)
fig.patch.set_alpha(0.0)
h = fig.add_subplot(111, axisbg = 'k')
plt.subplots_adjust(bottom = 0.2)
plt.grid(color="w")

if plottype == "plot":
	# do real thing here
	h.plot(random.randn(100), 'w', alpha=0.75, linewidth=2)
if plottype == "loglog":
	# do real thing here
	h.loglog(random.randn(100)**2, 'w', alpha=0.75, linewidth=2)
if plottype == "hist":
	# do real thing here
	h.fill_between(numpy.linspace(0,100, 11), random.rand(11), alpha=0.75, linewidth=2, facecolor="w", color="w")
if plottype == "pie":
	# do real thing here
	plt.subplots_adjust(bottom = 0.13, top = 1.0)
	h.pie(10 * random.rand(4), shadow=True, labels = ("a","b","c","d"))
	
# handle labels
if "xlabel" in form:
	plt.xlabel(form.getvalue("xlabel"))
if "ylabel" in form:
	plt.ylabel(form.getvalue("ylabel"))
if "title" in form:
	plt.title(form.getvalue("title"))
# write out the svg
plt.savefig(sys.stdout, format="svg")

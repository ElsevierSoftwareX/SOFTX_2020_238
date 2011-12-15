#!/usr/bin/python
import sys
import cgi
import cgitb
import os
os.environ["MPLCONFIGDIR"] = "/tmp"
import matplotlib
matplotlib.use('Agg')
from numpy import random
import matplotlib.pyplot as plt
cgitb.enable()
form = cgi.FieldStorage()

print >>sys.stdout, 'Content-type: image/svg+xml\r\n'
fig = plt.figure(figsize=(4,3), )
h = fig.add_subplot(111, axisbg = 'k')
h.plot(random.randn(100), 'w', alpha=0.75, linewidth=2)
plt.grid(color="w")
plt.savefig(sys.stdout, format="svg")

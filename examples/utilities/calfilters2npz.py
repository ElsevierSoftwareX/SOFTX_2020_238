#!/usr/bin/env python
#
# JBC. December 2010.

"""
Create a numpy npz file with the information of the filters.

The output file contains the calibration filters in the format that
gstlal_compute_strain expects.

Information about the filter files can be found in the DAC wiki at
https://wiki.ligo.org/foswiki/bin/view/DAC/S6Calibration#Time_domain
"""

# NOTE
#
# The new version of lalapps_ComputeStrainDriver reads filter files with
# a checksum. To convert any of the old filter files to the new version
# including the checksum, run:
#   echo "SHA-1 checksum: $(tail -n +1 $FILTERS_FILE | sha1sum)" > $NEW_FILTERS_FILE
#   cat $FILTERS_FILE >> $NEW_FILTERS_FILE
#
# Check with
#   tail -n +2 $NEW_FILTERS_FILE | sha1sum


import sys
import numpy

if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = 'S6H1Filters_942436815.txt'

def ffile_generator():
    for line in open(filename):
        first = line.split()[0]
        try:
            yield float(first)
        except:
            yield first

g = ffile_generator()

header = g.next()  # header
if header.startswith('SHA-1'):
    header = g.next()  # new format, one more line to go

cal_line_freq = g.next()
olg_resp = g.next() + 1j * g.next()
awhitening_resp = g.next() + 1j * g.next()
servo_resp = g.next() + 1j * g.next()

g.next()  # sensing
g.next()  # upsampling factor = 1
inv_sens_delay = int(g.next())
n = int(g.next())
inv_sensing = numpy.zeros(n)
for i in xrange(n):
    inv_sensing[i] = g.next()

g.next()  # servo
n = int(g.next())
servo = numpy.zeros(n)
for i in xrange(n):
    servo[i] = g.next()

g.next()  # actuation
n = int(g.next())
actuation = numpy.zeros(n)
for i in xrange(n):
    actuation[i] = g.next()

g.next()  # awhitening
n = int(g.next())
awhitening = numpy.zeros(n)
for i in xrange(n):
    awhitening[i] = g.next()

# Write to npz
numpy.savez('filters.npz',
            cal_line_freq=cal_line_freq,
            olg_resp=olg_resp,
            awhitening_resp=awhitening_resp,
            servo_resp=servo_resp,
            inv_sens_delay=inv_sens_delay,
            inv_sensing=inv_sensing,
            servo=servo,
            actuation=actuation,
            awhitening=awhitening)

#!/usr/bin/env python3
# Copyright (C) 2016  Aaron Viets
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

#
# =============================================================================
#
#				   Preamble
#
# =============================================================================
#


import random
import sys
import os
import numpy
import time
from math import pi
import resource
import datetime
import time
import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 22
matplotlib.rcParams['legend.fontsize'] = 18
matplotlib.rcParams['mathtext.default'] = 'regular'
matplotlib.use('Agg')
import glob
import matplotlib.pyplot as plt

from optparse import OptionParser, Option

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
GObject.threads_init()
Gst.init(None)

import lal
from lal import LIGOTimeGPS

from gstlal import pipeparts
from gstlal import calibration_parts
from gstlal import simplehandler
from gstlal import datasource

from ligo import segments
from gstlal import test_common


#
# =============================================================================
#
#				  Pipelines
#
# =============================================================================
#

def line_separation_test_01(pipeline, name, line_sep = 0.5):
	#
	# This test measures and plots the amount of contamination a calibration line adds to the result of demodulating at a nearby frequency as a function of frequency separation.
	#

	rate = 1000		# Hz
	buffer_length = 1.0	# seconds
	test_duration = 10.0	# seconds

	#
	# build pipeline
	#

	noise = test_common.test_src(pipeline, rate = 16384, test_duration = 1000, wave = 5, src_suffix = '0')
	noise = pipeparts.mkaudioamplify(pipeline, noise, 2.5)
	noise = pipeparts.mktee(pipeline, noise)

	signal = test_common.test_src(pipeline, rate = 16384, test_duration = 1000, wave = 0, freq = 16.3 + line_sep, src_suffix = '1')
	noisy_signal = calibration_parts.mkadder(pipeline, calibration_parts.list_srcs(pipeline, signal, noise))

	demod_noise = calibration_parts.demodulate(pipeline, noise, 16.3, True, 16, 20, 0) 
	demod_noise = pipeparts.mkgeneric(pipeline, demod_noise, "cabs")
	rms_noise = calibration_parts.compute_rms(pipeline, demod_noise, 16, 900, filter_latency = 0, rate_out = 1)
	pipeparts.mknxydumpsink(pipeline, rms_noise, "rms_noise.txt")

	demod_signal = calibration_parts.demodulate(pipeline, noisy_signal, 16.3, True, 16, 20, 0)
	demod_signal = pipeparts.mkgeneric(pipeline, demod_signal, "cabs")
	rms_signal = calibration_parts.compute_rms(pipeline, demod_signal, 16, 900, filter_latency = 0, rate_out = 1)
	pipeparts.mknxydumpsink(pipeline, rms_signal, "rms_signal.txt")

	#
	# done
	#
	
	return pipeline
	
#
# =============================================================================
#
#				     Main
#
# =============================================================================
#

line_seps = []
noise_rmss = []
signal_rmss = []
for i in range(0, 1000):
	line_sep = random.random()
	line_seps.append(line_sep)
	test_common.build_and_run(line_separation_test_01, "line_separation_test_01", line_sep = line_sep)
	data = numpy.loadtxt("rms_noise.txt")
	noise_rmss.append(data[-1][1])
	data = numpy.loadtxt("rms_signal.txt")
	signal_rmss.append(data[-1][1])
plt.figure(figsize = (10, 10))
plt.plot(line_seps, noise_rmss, 'r.', markersize = 5, label = 'noise')
plt.plot(line_seps, signal_rmss, 'g.', markersize = 5, label = 'noise with line')
leg = plt.legend(fancybox = True)
leg.get_frame().set_alpha(0.5)
plt.gca().set_xscale('linear')
plt.gca().set_yscale('log')
plt.title('Cross-contamination of Lines')
plt.ylabel('RMS of demodulated line')
plt.xlabel('Line Separation [Hz]')
plt.xlim(0.0, 1.0)
plt.ylim(0.00001, 1.0)
plt.grid(True)
plt.savefig('line_sep_test.png')
plt.savefig('line_sep_test.pdf')



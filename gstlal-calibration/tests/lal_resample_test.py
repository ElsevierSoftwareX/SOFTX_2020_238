#!/usr/bin/env python
# Copyright (C) 2017  Aaron Viets
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


import numpy
import sys

import matplotlib
from matplotlib import rc
rc('text', usetex = True)
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 32
matplotlib.rcParams['legend.fontsize'] = 20
matplotlib.rcParams['mathtext.default'] = 'regular'
matplotlib.use('Agg')
import glob
import matplotlib.pyplot as plt

from gstlal import pipeparts
from gstlal import calibration_parts
import test_common
from gi.repository import Gst


#
# =============================================================================
#
#				  Pipelines
#
# =============================================================================
#

def lal_resample_01(pipeline, name):

	#
	# This test passes an impulse through the resampler
	#

	rate_in = 128		# Hz
	rate_out = 1024		# Hz
	buffer_length = 0.77	# seconds
	test_duration = 30.0	# seconds
	quality = 4

	#
	# build pipeline
	#

	head = test_common.test_src(pipeline, buffer_length = buffer_length, wave = 0, freq = 0.25, rate = rate_in, test_duration = test_duration, width = 64)
	head = calibration_parts.mkinsertgap(pipeline, head, bad_data_intervals = [0.999999999, 1.00000001], block_duration = int(0.5 * Gst.SECOND), insert_gap = False, replace_value = 0.0)
	head = pipeparts.mktee(pipeline, head)
	pipeparts.mknxydumpsink(pipeline, head, "%s_in.txt" % name)
	head = pipeparts.mkgeneric(pipeline, head, "splitcounter")
	head = calibration_parts.mkresample(pipeline, head, quality, False, "audio/x-raw,format=F64LE,rate=%d" % rate_out)
	head = pipeparts.mkgeneric(pipeline, head, "splitcounter")
	pipeparts.mknxydumpsink(pipeline, head, "%s_out.txt" % name)

	#
	# done
	#
	
	return pipeline


def lal_resample_02(pipeline, name):

	#
	# This test passes a sinusoid through the resampler
	#

	rate_in = 1024		# Hz
	rate_out = 16384	# Hz
	buffer_length = 0.31	# seconds
	test_duration = 30.0	# seconds
	quality = 4

	#
	# build pipeline
	#

	head = test_common.test_src(pipeline, buffer_length = buffer_length, wave = 0, freq = 1, rate = rate_in, test_duration = test_duration, width = 64)
	#head = calibration_parts.mkinsertgap(pipeline, head, bad_data_intervals = [-2, 2])
	head = pipeparts.mktee(pipeline, head)
	pipeparts.mknxydumpsink(pipeline, head, "%s_in.txt" % name)
	head = pipeparts.mkgeneric(pipeline, head, "splitcounter")
	head = calibration_parts.mkresample(pipeline, head, quality, False, "audio/x-raw,format=F64LE,rate=%d" % rate_out)
	head = pipeparts.mkgeneric(pipeline, head, "splitcounter")
	pipeparts.mknxydumpsink(pipeline, head, "%s_out.txt" % name)

	#
	# done
	#

	return pipeline

def lal_resample_03(pipeline, name):

	#
	# This test passes ones through the resampler
	#

	rate_in = 128		# Hz
	rate_out = 16384	# Hz
	buffer_length = 0.25	# seconds
	test_duration = 10	# seconds
	quality = 4

	#
	# build pipeline
	#

	head = test_common.test_src(pipeline, buffer_length = buffer_length, wave = 0, freq = 0.0, rate = rate_in, test_duration = test_duration, width = 64)
	#head = calibration_parts.mkinsertgap(pipeline, head, bad_data_intervals = [-2, 2])
	head = pipeparts.mkgeneric(pipeline, head, "lal_add_constant", value = 1)
	head = pipeparts.mktee(pipeline, head)
	pipeparts.mknxydumpsink(pipeline, head, "%s_in.txt" % name)
	head = pipeparts.mkgeneric(pipeline, head, "splitcounter")
	head = calibration_parts.mkresample(pipeline, head, quality, False, "audio/x-raw,format=F64LE,rate=%d" % rate_out)
	head = pipeparts.mkgeneric(pipeline, head, "splitcounter")
	pipeparts.mknxydumpsink(pipeline, head, "%s_out.txt" % name)

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


test_common.build_and_run(lal_resample_01, "lal_resample_01")
test_common.build_and_run(lal_resample_02, "lal_resample_02")
test_common.build_and_run(lal_resample_03, "lal_resample_03")

indata = numpy.transpose(numpy.loadtxt("lal_resample_01_in.txt"))
intime = indata[0]
indata = indata[1]
outdata = numpy.transpose(numpy.loadtxt("lal_resample_01_out.txt"))
outtime = outdata[0]
outdata = outdata[1]

plt.figure(figsize = (15, 10))
plt.plot(intime, indata, 'blue', linewidth = 0.75, label = 'input')
plt.title("Upsampling impulses")
plt.plot(outtime, outdata, 'limegreen', linewidth = 0.75, label = 'output')
plt.grid(True, which = "both", linestyle = ':', linewidth = 0.3, color = 'black')
leg = plt.legend(fancybox = True)
leg.get_frame().set_alpha(0.8)
plt.savefig("lal_resample_01.png")

plt.figure(figsize = (15, 10))
plt.plot(intime, indata, 'blue', linewidth = 0.75, label = 'input')
plt.xlim(20.5, 21.5)
plt.title("Upsampling one impulse")
plt.plot(outtime, outdata, 'limegreen', linewidth = 0.75, label = 'output')
plt.grid(True, which = "both", linestyle = ':', linewidth = 0.3, color = 'black')
leg = plt.legend(fancybox = True)
leg.get_frame().set_alpha(0.8)
plt.savefig("lal_resample_01_zoom.png")

indata = numpy.transpose(numpy.loadtxt("lal_resample_02_in.txt"))
intime = indata[0]
indata = indata[1]
outdata = numpy.transpose(numpy.loadtxt("lal_resample_02_out.txt"))
outtime = outdata[0]
outdata = outdata[1]

plt.figure(figsize = (15, 10))
plt.plot(intime, indata, 'blue', linewidth = 0.75, label = 'input')
plt.title("Upsampling a sinusoid")
plt.plot(outtime, outdata, 'limegreen', linewidth = 0.75, label = 'output')
plt.grid(True, which = "both", linestyle = ':', linewidth = 0.3, color = 'black')
leg = plt.legend(fancybox = True)
leg.get_frame().set_alpha(0.8)
plt.savefig("lal_resample_02.png")

indata = numpy.transpose(numpy.loadtxt("lal_resample_03_in.txt"))
intime = indata[0]
indata = indata[1]
outdata = numpy.transpose(numpy.loadtxt("lal_resample_03_out.txt"))
outtime = outdata[0]
outdata = outdata[1]

plt.figure(figsize = (15, 10))
plt.plot(intime, indata, 'blue', linewidth = 0.75, label = 'input')
plt.ylim(0.9999999999999, 1.0000000000001)
plt.title("Upsampling ones")
plt.plot(outtime, outdata, 'limegreen', linewidth = 0.75, label = 'output')
plt.grid(True, which = "both", linestyle = ':', linewidth = 0.3, color = 'black')
leg = plt.legend(fancybox = True)
leg.get_frame().set_alpha(0.8)
plt.savefig("lal_resample_03.png")



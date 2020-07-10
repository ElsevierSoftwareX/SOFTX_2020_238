#!/usr/bin/env python3
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


import matplotlib
matplotlib.use('Agg')
import numpy as np
import sys
from matplotlib import rc
rc('text', usetex = True)
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 32
matplotlib.rcParams['legend.fontsize'] = 20
matplotlib.rcParams['mathtext.default'] = 'regular'
import glob
import matplotlib.pyplot as plt

from gstlal import pipeparts
from gstlal import calibration_parts
from gstlal import FIRtools as fir
from gstlal import ticks_and_grid
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
	quality = 5

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
	quality = 5

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
	quality = 5

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


def lal_resample_04(pipeline, name):

	#
	# This test passes noise through the resampler, downsampling and then upsampling
	#

	rate_in = 16384		# Hz
	rate_out = 2048		# Hz
	buffer_length = 1.0	# seconds
	test_duration = 10	# seconds
	quality = 5

	#
	# build pipeline
	#

	head = test_common.test_src(pipeline, buffer_length = buffer_length, wave = 5, freq = 0.0, rate = rate_in, test_duration = test_duration, width = 64)
	#head = calibration_parts.mkinsertgap(pipeline, head, bad_data_intervals = [-2, 2])
	head = pipeparts.mktee(pipeline, head)
	pipeparts.mknxydumpsink(pipeline, head, "%s_in.txt" % name)
	head = pipeparts.mkgeneric(pipeline, head, "splitcounter")
	head = calibration_parts.mkresample(pipeline, head, quality, False, "audio/x-raw,format=F64LE,rate=%d" % rate_out)
	head = pipeparts.mkgeneric(pipeline, head, "splitcounter")
	head = pipeparts.mktee(pipeline, head)
	pipeparts.mknxydumpsink(pipeline, head, "%s_down.txt" % name)
	head = calibration_parts.mkresample(pipeline, head, quality, False, "audio/x-raw,format=F64LE,rate=%d" % rate_in)
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


#test_common.build_and_run(lal_resample_01, "lal_resample_01")
#test_common.build_and_run(lal_resample_02, "lal_resample_02")
#test_common.build_and_run(lal_resample_03, "lal_resample_03")
#test_common.build_and_run(lal_resample_04, "lal_resample_04")

indata = np.transpose(np.loadtxt("lal_resample_01_in.txt"))
intime = indata[0]
indata = indata[1]
outdata = np.transpose(np.loadtxt("lal_resample_01_out.txt"))
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

indata = np.transpose(np.loadtxt("lal_resample_02_in.txt"))
intime = indata[0]
indata = indata[1]
outdata = np.transpose(np.loadtxt("lal_resample_02_out.txt"))
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

indata = np.transpose(np.loadtxt("lal_resample_03_in.txt"))
intime = indata[0]
indata = indata[1]
outdata = np.transpose(np.loadtxt("lal_resample_03_out.txt"))
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


indata = np.transpose(np.loadtxt("lal_resample_04_in.txt"))
intime = indata[0]
indata = indata[1]
downdata = np.transpose(np.loadtxt("lal_resample_04_down.txt"))
downtime = downdata[0]
downdata = downdata[1]
outdata = np.transpose(np.loadtxt("lal_resample_04_out.txt"))
outtime = outdata[0]
outdata = outdata[1]

# Find indices at 1 second and 9 seconds, and "chop" each data set to the same length
for i in range(len(intime)):
	if intime[i] == 1.0:
		start_idx = i
	if intime[i] == 9.0:
		end_idx = i
intime = intime[start_idx:end_idx]
indata = indata[start_idx:end_idx]

for i in range(len(downtime)):
        if downtime[i] == 1.0:
                start_idx = i
        if downtime[i] == 9.0:
                end_idx = i
downtime = downtime[start_idx:end_idx]
downdata = downdata[start_idx:end_idx]

for i in range(len(outtime)):
        if outtime[i] == 1.0:
                start_idx = i
        if outtime[i] == 9.0:
                end_idx = i
outtime = outtime[start_idx:end_idx]
outdata = outdata[start_idx:end_idx]

down_freq = np.arange(0.0, 1024.5, 1.0)
out_freq = np.arange(0.0, 8192.5, 1.0)

in_fft = np.zeros(8193, dtype = np.complex256)
down_fft = np.zeros(1025, dtype = np.complex256)
out_fft = np.zeros(8193, dtype = np.complex256)

for i in range(15):
	in_chunk = fir.DolphChebyshev(16384, 12) * indata[i * 8192 : i * 8192 + 16384]
	down_chunk = fir.DolphChebyshev(2048, 12) * downdata[i * 1024 : i * 1024 + 2048]
	out_chunk = fir.DolphChebyshev(16384, 12) * outdata[i * 8192 : i * 8192 + 16384]

	in_fft += fir.rfft(in_chunk)
	down_fft += fir.rfft(down_chunk)
	out_fft += fir.rfft(out_chunk)

down_fft *= 8

down_tf = down_fft / in_fft[:len(down_fft)]
out_tf = out_fft / in_fft

down_tf_mag = abs(down_tf)
out_tf_mag = abs(out_tf)

down_tf_phase = 180.0 / np.pi * np.angle(down_tf)
out_tf_phase = 180.0 / np.pi * np.angle(out_tf)

plt.figure(figsize = (12, 12))
plt.subplot(211)
plt.plot(down_freq, down_tf_mag, linewidth = 0.75)
plt.title("Downsampling 16384 - 2048 Hz TF")
plt.ylabel("Magnitude")
ticks_and_grid.ticks_and_grid(plt.gca(), yscale = 'log', xscale = 'log', ymin = 1e-10)

plt.subplot(212)
plt.plot(down_freq, down_tf_phase, linewidth = 0.75)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (deg)")
ticks_and_grid.ticks_and_grid(plt.gca(), yscale = 'linear', xscale = 'log')

plt.savefig("lal_resample_down_tf.png")

plt.figure(figsize = (12, 12))
plt.subplot(211)
plt.plot(out_freq, out_tf_mag, linewidth = 0.75)
plt.title("Resampling 16384 - 2048 - 16384 Hz TF")
plt.ylabel("Magnitude")
ticks_and_grid.ticks_and_grid(plt.gca(), yscale = 'log', xscale = 'log')

plt.subplot(212)
plt.plot(out_freq, out_tf_phase, linewidth = 0.75)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (deg)")
ticks_and_grid.ticks_and_grid(plt.gca(), yscale = 'linear', xscale = 'log')

plt.savefig("lal_resample_down_up_tf.png")


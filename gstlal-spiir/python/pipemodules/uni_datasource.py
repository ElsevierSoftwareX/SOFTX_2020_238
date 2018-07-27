
# Copyright (C) 2014  Kipp Cannon, Chad Hanna, Drew Keppel, Qi Chu
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
#                                   Preamble
#
# =============================================================================
#


import sys
import optparse
import math
import numpy
from scipy import fftpack

# The following snippet is taken from http://gstreamer.freedesktop.org/wiki/FAQ#Mypygstprogramismysteriouslycoredumping.2Chowtofixthis.3F
import pygtk
pygtk.require("2.0")
import gobject
gobject.threads_init()
import pygst
pygst.require('0.10')
import gst

from gstlal import bottle
from gstlal import pipeparts
from gstlal import reference_psd
from gstlal import datasource

import lal

## 
# @file
#
# A file that contains the unirate_datasource module code
#

##
# @package python.unirate_datasource
#
# unirate_datasource module

## #### produced whitened h(t) at (possibly) multiple sample rates
# ##### Gstreamer graph describing this function
#
# @dot
# digraph mkbasicsrc {
#	rankdir = LR;
#	compound=true;
#	node [shape=record fontsize=10 fontname="Verdana"];
#	edge [fontsize=8 fontname="Verdana"];
#
#	capsfilter1 [URL="\ref pipeparts.mkcapsfilter()"];
#	audioresample [URL="\ref pipeparts.mkresample()"];
#	capsfilter2 [URL="\ref pipeparts.mkcapsfilter()"];

#	reblock [URL="\ref pipeparts.mkreblock()"];
#	whiten [URL="\ref pipeparts.mkwhiten()"];
#	audioconvert [URL="\ref pipeparts.mkaudioconvert()"];
#	capsfilter3 [URL="\ref pipeparts.mkcapsfilter()"];
#	"segmentsrcgate()" [URL="\ref datasource.mksegmentsrcgate()", label="segmentsrcgate() \n [iff veto segment list provided]", style=filled, color=lightgrey];
#	tee [URL="\ref pipeparts.mktee()"];
#	audioamplifyr1 [URL="\ref pipeparts.mkaudioamplify()"];
#	capsfilterr1 [URL="\ref pipeparts.mkcapsfilter()"];

#	htgater1 [URL="\ref datasource.mkhtgate()", label="htgate() \n [iff ht gate specified]", style=filled, color=lightgrey];
#	tee1 [URL="\ref pipeparts.mktee()"];
#	audioamplifyr2 [URL="\ref pipeparts.mkaudioamplify()"];
#	capsfilterr2 [URL="\ref pipeparts.mkcapsfilter()"];

#	htgater2 [URL="\ref datasource.mkhtgate()", label="htgate() \n [iff ht gate specified]", style=filled, color=lightgrey];
#	tee2 [URL="\ref pipeparts.mktee()"];
#	audioamplify_rn [URL="\ref pipeparts.mkaudioamplify()"];
#	capsfilter_rn [URL="\ref pipeparts.mkcapsfilter()"];

#	htgate_rn [URL="\ref datasource.mkhtgate()", style=filled, color=lightgrey, label="htgate() \n [iff ht gate specified]"];
#	tee [URL="\ref pipeparts.mktee()"];
#
#	// nodes
#
#	"?" -> capsfilter1 -> audioresample;
#	audioresample -> capsfilter2;
#	capsfilter2 -> reblock;
#
#	reblock -> whiten;
#	whiten -> audioconvert;
#	audioconvert -> capsfilter3;
#	capsfilter3 -> "segmentsrcgate()";
#	"segmentsrcgate()" -> tee;
#
#	tee -> audioamplifyr1 [label="Rate 1"];
#	audioamplifyr1 -> capsfilterr1;
#	capsfilterr1 -> htgater1;
#	
#	htgater1 -> tee1 -> "? 1";
#
#	tee -> audioamplifyr2 [label="Rate 2"];
#	audioamplifyr2 -> capsfilterr2;
#	capsfilterr2 -> htgater2;
#
#	htgater2 -> tee2 -> "? 2";
#
#	tee ->  audioamplify_rn [label="Rate N"];
#	audioamplify_rn -> capsfilter_rn;
#	capsfilter_rn -> htgate_rn;
#	
#	htgate_rn -> tee_n -> "? 3";
#
# }
# @enddot
def mkwhitened_src(pipeline, src, max_rate, instrument, psd = None,
		psd_fft_length = 8, ht_gate_threshold = float("inf"),
		veto_segments = None, seekevent = None, nxydump_segment = None, nxydump_directory = '.',
		track_psd = False, block_duration = 1 * gst.SECOND, zero_pad =
		0, width = 64, fir_whitener = 0, statevector = None, dqvector =
		None):
	"""!
	Build pipeline stage to whiten and downsample h(t).

	- pipeline: the gstreamer pipeline to add this to
	- src: the gstreamer element that will be providing data to this 
	- max_rate: the rate that is been set to
	- instrument: the instrument to process
	- psd: a psd frequency series
	- psd_fft_length: length of fft used for whitening
	- ht_gate_threshold: gate h(t) if it crosses this value
	- veto_segments: segments to mark as gaps after whitening
	- track_psd: decide whether to dynamically track the spectrum or use the fixed spectrum provided
	- width: type convert to either 32 or 64 bit float
	"""

	#
	# down-sample to highest of target sample rates.  we include a caps
	# filter upstream of the resampler to ensure that this is, infact,
	# *down*-sampling.  if the source time series has a lower sample
	# rate than the highest target sample rate the resampler will
	# become an upsampler, and the result will likely interact poorly
	# with the whitener as it tries to ampify the non-existant
	# high-frequency components, possibly adding significant numerical
	# noise to its output.  if you see errors about being unable to
	# negotiate a format from this stage in the pipeline, it is because
	# you are asking for output sample rates that are higher than the
	# sample rate of your data source.
	#

	quality = 9
	head = pipeparts.mkcapsfilter(pipeline, src, "audio/x-raw-float, rate=[%d,MAX]" % max_rate)
	head = pipeparts.mkcapsfilter(pipeline, pipeparts.mkresample(pipeline, head, quality = quality), "audio/x-raw-float, rate=%d" % max_rate)
	head = pipeparts.mknofakedisconts(pipeline, head)	# FIXME:  remove when resampler is patched
	head = pipeparts.mkchecktimestamps(pipeline, head, "%s_timestamps_%d_hoft" % (instrument, max_rate))

	#
	# add a reblock element.  the whitener's gap support isn't 100% yet
	# and giving it smaller input buffers works around the remaining
	# weaknesses (namely that when it sees a gap buffer large enough to
	# drain its internal history, it doesn't know enough to produce a
	# short non-gap buffer to drain its history followed by a gap
	# buffer, it just produces one huge non-gap buffer that's mostly
	# zeros).
	#

	head = pipeparts.mkreblock(pipeline, head, block_duration = block_duration)
	head = pipeparts.mktee(pipeline, head)

	if nxydump_segment is not None:
                pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, head), "%s/before_whitened_data_%s_%d.dump" % (nxydump_directory, instrument, nxydump_segment[0]), segment = nxydump_segment)

	#
	# construct whitener.
	#

	if fir_whitener:
		# For now just hard code these acceptable inputs until we have
		# it working well in all situations or appropriate assertions
		psd = None
		head = pipeparts.mktee(pipeline, head)
		psd_fft_length = 16
		zero_pad = 0
		# because we are not asking the whitener to reassemble an
		# output time series (that we care about) we drop the
		# zero-padding in this code path.  the psd_fft_length is
		# reduced to account for the loss of the zero padding to
		# keep the Hann window the same as implied by the
		# user-supplied parameters.
		whiten = pipeparts.mkwhiten(pipeline, pipeparts.mkqueue(pipeline, head, max_size_time = 2 * psd_fft_length * gst.SECOND), fft_length = psd_fft_length - 2 * zero_pad, zero_pad = 0, average_samples = 64, median_samples = 7, expand_gaps = True, name = "lal_whiten_%s" % instrument)
		pipeparts.mkfakesink(pipeline, whiten)

		# high pass filter
		kernel = reference_psd.one_second_highpass_kernel(max_rate, cutoff = 12)
		assert len(kernel) % 2 == 1, "high-pass filter length is not odd"
		head = pipeparts.mkfirbank(pipeline, pipeparts.mkqueue(pipeline, head, max_size_buffers = 1), fir_matrix = numpy.array(kernel, ndmin = 2), block_stride = max_rate, time_domain = False, latency = (len(kernel) - 1) // 2)

		# FIXME at some point build an initial kernel from a reference psd
		psd_fir_kernel = reference_psd.PSDFirKernel()
		fir_matrix = numpy.zeros((1, 1 + max_rate * psd_fft_length), dtype=numpy.float64)
		head = pipeparts.mkfirbank(pipeline, head, fir_matrix = fir_matrix, block_stride = max_rate, time_domain = False, latency = 0)

		def set_fir_psd(whiten, pspec, firbank, psd_fir_kernel):
			psd_data = numpy.array(whiten.get_property("mean-psd"))
			psd = lal.CreateREAL8FrequencySeries(
				name = "psd",
				epoch = lal.LIGOTimeGPS(0),
				f0 = 0.0,
				deltaF = whiten.get_property("delta-f"),
				sampleUnits = lal.Unit(whiten.get_property("psd-units")),
				length = len(psd_data)
			)
			psd.data.data = psd_data
			kernel, latency, sample_rate = psd_fir_kernel.psd_to_linear_phase_whitening_fir_kernel(psd)
			
			#kernel = psd_fir_kernel.min_phase(kernel)
			#kernel = psd_fir_kernel.homomorphic(kernel, sample_rate)
			kernel, theta = psd_fir_kernel.linear_phase_fir_kernel_to_minimum_phase_whitening_fir_kernel(kernel, sample_rate)
			firbank.set_property("fir-matrix", numpy.array(kernel, ndmin = 2))
		whiten.connect_after("notify::mean-psd", set_fir_psd, head, psd_fir_kernel)

		# Gate after gaps
		# FIXME the -max(rates) extra padding is for the high pass
		# filter: NOTE it also needs to be big enough for the
		# downsampling filter, but that is typically smaller than the
		# HP filter (192 samples at Qual 9)
		if statevector:
			head = pipeparts.mkgate(pipeline, head, control = statevector, default_state = False, threshold = 1, hold_length = -max_rate, attack_length = -max_rate * (psd_fft_length + 1))
		if dqvector:
			head = pipeparts.mkgate(pipeline, head, control = dqvector, default_state = False, threshold = 1, hold_length = -max_rate, attack_length = -max_rate * (psd_fft_length + 1))
		# Drop initial data to let the PSD settle
		head = pipeparts.mkdrop(pipeline, head, drop_samples = 16 * psd_fft_length * max_rate)
		head = pipeparts.mkchecktimestamps(pipeline, head, "%s_timestamps_fir" % instrument)
		#head = pipeparts.mknxydumpsinktee(pipeline, head, filename = "after_mkfirbank.txt")
	else:
		if statevector:
			pipeparts.mkfakesink(pipeline, statevector)
		if dqvector:
			pipeparts.mkfakesink(pipeline, dqvector)
	
		head = whiten = pipeparts.mkwhiten(pipeline, head, fft_length = psd_fft_length, zero_pad = zero_pad, average_samples = 64, median_samples = 7, expand_gaps = True, name = "lal_whiten_%s" % instrument)
		# make the buffers going downstream smaller, this can
		# really help with RAM
		# add a reblock to reduce htgate latency when fft whitening 
		head = pipeparts.mkreblock(pipeline, head, block_duration = block_duration)

	# export PSD in ascii text format
	# FIXME:  also make them available in XML format as a single document
	@bottle.route("/%s/psd.txt" % instrument)
	def get_psd_txt(elem = whiten):
		delta_f = elem.get_property("delta-f")
		yield "# frequency\tspectral density\n"
		for i, value in enumerate(elem.get_property("mean-psd")):
			yield "%.16g %.16g\n" % (i * delta_f, value)

	if psd is None:
		# use running average PSD
		whiten.set_property("psd-mode", 0)
	else:
		if track_psd:
			# use running psd
			whiten.set_property("psd-mode", 0)
		else:
			# use fixed PSD
			whiten.set_property("psd-mode", 1)

		#
		# install signal handler to retrieve \Delta f and
		# f_{Nyquist} whenever they are known and/or change,
		# resample the user-supplied PSD, and install it into the
		# whitener.
		#

		def psd_units_or_resolution_changed(elem, pspec, psd):
			# make sure units are set, compute scale factor
			units = lal.Unit(elem.get_property("psd-units"))
			if units == lal.DimensionlessUnit:
				return
			scale = float(psd.sampleUnits / units)
			# get frequency resolution and number of bins
			delta_f = elem.get_property("delta-f")
			n = int(round(elem.get_property("f-nyquist") / delta_f) + 1)
			# interpolate, rescale, and install PSD
			psd = reference_psd.interpolate_psd(psd, delta_f)
			elem.set_property("mean-psd", psd.data.data[:n] * scale)
		whiten.connect_after("notify::f-nyquist", psd_units_or_resolution_changed, psd)
		whiten.connect_after("notify::delta-f", psd_units_or_resolution_changed, psd)
		whiten.connect_after("notify::psd-units", psd_units_or_resolution_changed, psd)



	#
	# convert to desired precision
	#

	head = pipeparts.mkaudioconvert(pipeline, head)
	head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw-float,\
			width=%d, rate=%d, channels=1" % (width, max_rate))
	head = pipeparts.mkchecktimestamps(pipeline, head, "%s_timestamps_%d_whitehoft" % (instrument, max_rate))


	#
	# optionally add vetoes
	#

	if veto_segments is not None:
		head = datasource.mksegmentsrcgate(pipeline, head, veto_segments, seekevent=seekevent, invert_output=True)

	# h(t) gate plugin (mkhtgate) was first not used in this file. It caused that 
	# the gpu pipeline could not find htgate , e.g. htgate for H1:
	# "could not find H1_ht_gate for H1 'whitehtsegments'"
	# where it should be:
        # "found H1_ht_gate for H1 'whitehtsegments'"

	#
	# optional gate on whitened h(t) amplitude.  attack and hold are
	# made to be 1/4 second or 1 sample, whichever is larger
	#

	# FIXME:  this could be omitted if ht_gate_threshold is None, but
	# we need to collect whitened h(t) segments, however something
	# could be done to collect those if these gates aren't here.
	# ht_gate_window = max(max_rate // 4, 1)	# samples
	# NOTE: ht_gate_window set to 0 to reduce latency from 4s to 0s. previous
	# setting = 0.25s. For each data block at 4s, it has to wait extra 0.25s
	# to finish processing causing 4s latency.

	ht_gate_window = max(max_rate // 4, 1)
	head = datasource.mkhtgate(pipeline, head, threshold = ht_gate_threshold if ht_gate_threshold is not None else float("+inf"), hold_length = ht_gate_window, attack_length = ht_gate_window, name = "%s_ht_gate" % instrument)
	# emit signals so that a user can latch on to them
	head.set_property("emit-signals", True)

	# uni data source differs from multi rate data source starts here
	# reblock incoming data into 1 second-long data segments

	# multi_downsample + spiirup
	# e.g. if downsample_depth = 8:
	#      1 channel, rate = 4096-> multi channel, rate = (32, 64, 128, 256, 512, 1024, 2048, 4096)
	# multi_downsample embeds amplitude correction (audioamplify)


	head = pipeparts.mktee(pipeline, head)
	if nxydump_segment is not None:
		pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, head), "%s/whitened_data_%s_%d.dump" % (nxydump_directory, instrument, nxydump_segment[0]), segment = nxydump_segment)
	
	return head



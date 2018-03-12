# Copyright (C) 2009--2013  Kipp Cannon, Chad Hanna, Drew Keppel
# Copyright (C) 2017 	    Patrick Godwin
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
import os
import optparse
import math
import numpy

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst, GstAudio
GObject.threads_init()
Gst.init(None)

import lal

from gstlal import bottle
from gstlal import pipeparts
from gstlal import reference_psd
from gstlal import datasource

# FIXME: Find a better way than using global variables.
PSD_FFT_LENGTH = 32
PSD_DROP_TIME = 16 * PSD_FFT_LENGTH

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
#	reblock -> whiten;
#	whiten -> audioconvert;
#	audioconvert -> capsfilter3;
#	capsfilter3 -> "segmentsrcgate()";
#	"segmentsrcgate()" -> tee;
#
#	tee -> audioamplifyr1 [label="Rate 1"];
#	audioamplifyr1 -> capsfilterr1;
#	capsfilterr1 -> htgater1;
#	htgater1 -> tee1 -> "? 1";
#
#	tee -> audioamplifyr2 [label="Rate 2"];
#	audioamplifyr2 -> capsfilterr2;
#	capsfilterr2 -> htgater2;
#	htgater2 -> tee2 -> "? 2";
#
#	tee ->  audioamplify_rn [label="Rate N"];
#	audioamplify_rn -> capsfilter_rn;
#	capsfilter_rn -> htgate_rn;
#	htgate_rn -> tee_n -> "? 3";
#
# }
# @enddot
def mkwhitened_multirate_src(pipeline, src, rates, native_rate, instrument, psd = None, psd_fft_length = PSD_FFT_LENGTH, veto_segments = None, track_psd = False, block_duration = int(1 * Gst.SECOND), zero_pad = 0, width = 64, channel_name = "hoft"):
	"""!
	Build pipeline stage to whiten and downsample auxiliary channels.

	- pipeline: the gstreamer pipeline to add this to
	- src: the gstreamer element that will be providing data to this 
	- rates: a list of the requested sample rates, e.g., [512,1024].
	- native_rate: Native sampling rate of channel
	- instrument: the instrument to process
	- psd: a psd frequency series
	- psd_fft_length: length of fft used for whitening
	- veto_segments: segments to mark as gaps after whitening
	- track_psd: decide whether to dynamically track the spectrum or use the fixed spectrum provided
	- width: type convert to either 32 or 64 bit float
	- channel_name: channel to whiten and downsample
	"""

	#head = pipeparts.mkchecktimestamps(pipeline, src, "%s_%s_%d_timestamps" % (instrument, channel_name,  native_rate))

	max_rate = max(rates)
	if native_rate > max_rate:
		head = pipeparts.mkaudiocheblimit(pipeline, src, cutoff = 0.9 * (max_rate/2), type = 2, ripple = 0.1)
		head = pipeparts.mkaudioundersample(pipeline, head)
		head = pipeparts.mkcapsfilter(pipeline, head, caps = "audio/x-raw, rate=%d" % max_rate)
	else:
		head = src

	# high pass filter
	# FIXME: don't hardcode native rate cutoff for high-pass filtering
	#if native_rate >= 128:
	#	kernel = reference_psd.one_second_highpass_kernel(max_rate, cutoff = 12)
	#	block_stride = block_duration * max_rate // Gst.SECOND
	#	assert len(kernel) % 2 == 1, "high-pass filter length is not odd"
	#	head = pipeparts.mkfirbank(pipeline, head, fir_matrix = numpy.array(kernel, ndmin = 2), block_stride = block_stride, time_domain = False, latency = (len(kernel) - 1) // 2)

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

	#
	# construct whitener.
	#

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
	whiten = pipeparts.mkwhiten(pipeline, pipeparts.mkqueue(pipeline, head, max_size_time = 2 * psd_fft_length * Gst.SECOND), fft_length = psd_fft_length - 2 * zero_pad, zero_pad = 0, average_samples = 64, median_samples = 7, expand_gaps = True, name = "%s_%s_lalwhiten" % (instrument, channel_name))
	pipeparts.mkfakesink(pipeline, whiten)

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
		kernel, theta = psd_fir_kernel.linear_phase_fir_kernel_to_minimum_phase_whitening_fir_kernel(kernel, sample_rate)
		# subtract DC offset from signal
		kernel -= numpy.mean(kernel)
		firbank.set_property("fir-matrix", numpy.array(kernel, ndmin = 2))
	whiten.connect_after("notify::mean-psd", set_fir_psd, head, psd_fir_kernel)

	# Drop initial data to let the PSD settle
	head = pipeparts.mkdrop(pipeline, head, drop_samples = PSD_DROP_TIME * max_rate)
	#head = pipeparts.mkchecktimestamps(pipeline, head, "%s_%s_timestampsfir" % (instrument, channel_name))
	
	#head = pipeparts.mknxydumpsinktee(pipeline, head, filename = "after_mkfirbank.txt")

	if psd is None:
		# use running average PSD
		whiten.set_property("psd-mode", 0)
	else:
		if track_psd:
			# use running average PSD
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
	if width == 64:
		head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw, rate=%d, format=%s" % (max_rate, GstAudio.AudioFormat.to_string(GstAudio.AudioFormat.F64)))
	else:
		head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw, rate=%d, format=%s" % (max_rate, GstAudio.AudioFormat.to_string(GstAudio.AudioFormat.F32)))
	#head = pipeparts.mkchecktimestamps(pipeline, head, "%s_%s_%d_timestampwhite" % (instrument, channel_name, max_rate))

	#
	# optionally add vetoes
	#

	if veto_segments is not None:
		head = datasource.mksegmentsrcgate(pipeline, head, veto_segments, invert_output=True)

	#
	# tee for highest sample rate stream
	#

	#head = {max_rate: pipeparts.mktee(pipeline, head)}
	tee = pipeparts.mktee(pipeline, head)
	head = {rate: None for rate in rates}
	head[max_rate] = pipeparts.mkqueue(pipeline, tee, max_size_buffers = 0, max_size_bytes = 0, max_size_time = Gst.SECOND * 8)
	#
	# down-sample whitened time series to remaining target sample rates
	# while applying an amplitude correction to adjust for low-pass
	# filter roll-off.  we also scale by \sqrt{original rate / new
	# rate}.  this is done to preserve the square magnitude of the time
	# series --- the inner product of the time series with itself.
	# really what we want is for
	#
	#	\int v_{1}(t) v_{2}(t) \diff t
	#		\approx \sum v_{1}(t) v_{2}(t) \Delta t
	#
	# to be preserved across different sample rates, i.e. for different
	# \Delta t.  what we do is rescale the time series and ignore
	# \Delta t, so we put 1/2 factor of the ratio of the \Delta t's
	# into the h(t) time series here, and, later, another 1/2 factor
	# into the template when it gets downsampled.
	#
	# by design, the output of the whitener is a unit-variance time
	# series.  however, downsampling it reduces the variance due to the
	# removal of some frequency components.  we require the input to
	# the orthogonal filter banks to be unit variance, therefore a
	# correction factor is applied via an audio amplify element to
	# adjust for the reduction in variance due to the downsampler.
	#

	for rate in sorted(set(rates))[:-1]:
		head[rate] = pipeparts.mkqueue(pipeline, tee, max_size_buffers = 0, max_size_bytes = 0, max_size_time = Gst.SECOND * 8)
		# low pass filter + audio undersampler to downsample
		# NOTE: as long as this fudge factor (0.9) for the high frequency cutoff is
		#       higher than the cutoff for the FIR bank basis, this should be fine
		head[rate] = pipeparts.mkaudiocheblimit(pipeline, head[rate], cutoff = 0.9 * (rate/2), type = 2, ripple = 0.1)
		head[rate] = pipeparts.mkaudioundersample(pipeline, head[rate])
		head[rate] = pipeparts.mkcapsfilter(pipeline, head[rate], caps = "audio/x-raw, rate=%d" % rate)

		#head[rate] = pipeparts.mkchecktimestamps(pipeline, head[rate], "%s_%s_%d_timestampswhite" % (instrument, channel_name, rate))

	#
	# done.  return value is a dictionary of tee elements indexed by
	# sample rate
	#

	return head
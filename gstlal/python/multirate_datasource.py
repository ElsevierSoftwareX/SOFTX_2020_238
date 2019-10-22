# Copyright (C) 2009--2013  Kipp Cannon, Chad Hanna, Drew Keppel
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
from ligo import segments

from gstlal import bottle
from gstlal import pipeparts
from gstlal import reference_psd
from gstlal import datasource

__doc__ = """

**Review Status**

+------------------------------------------------+------------------------------------------+------------+
| Names                                          | Hash                                     | Date       |
+================================================+==========================================+============+
| Florent, Sathya, Duncan Me, Jolien, Kipp, Chad | 8a6ea41398be79c00bdc27456ddeb1b590b0f68e | 2014-06-18 |
+------------------------------------------------+------------------------------------------+------------+
"""

# a macro to switch between a conventional whitener and a fir whitener below
try:
	if int(os.environ["GSTLAL_FIR_WHITEN"]):
		FIR_WHITENER = True
	else:
		FIR_WHITENER = False
except KeyError as e:
	print >> sys.stderr, "You must set the environment variable GSTLAL_FIR_WHITEN to either 0 or 1.  1 enables causal whitening. 0 is the traditional acausal whitening filter"
	raise


def mkwhitened_multirate_src(pipeline, src, rates, instrument, psd = None, native_rate = 16384, psd_fft_length = 32, ht_gate_threshold = float("+inf"), veto_segments = None, nxydump_segment = None, track_psd = False, block_duration = 1 * Gst.SECOND, zero_pad = None, width = 64, unit_normalize = True, statevector = None, dqvector = None, fir_whiten_reference_psd = None):
	"""
	Build pipeline stage to whiten and downsample h(t).

	* pipeline: the gstreamer pipeline to add this to
	* src: the gstreamer element that will be providing data to this
	* rates: a list of the requested sample rates, e.g., [512,1024].
	* instrument: the instrument to process
	* psd: a psd frequency series
	* native_rate: the sampling rate of the source
	* psd_fft_length: length of fft used for whitening
	* ht_gate_threshold: gate h(t) if it crosses this value
	* veto_segments: segments to mark as gaps after whitening
	* track_psd: decide whether to dynamically track the spectrum or use the fixed spectrum provided
	* width: type convert to either 32 or 64 bit float
	* fir_whiten_reference_psd: when using FIR whitener, use this PSD to define desired desired phase response

	**Gstreamer graph describing this function**

	.. graphviz::

	   digraph mkbasicsrc {
		rankdir = LR;
		compound=true;
		node [shape=record fontsize=10 fontname="Verdana"];
		edge [fontsize=8 fontname="Verdana"];

		capsfilter1 ;
		audioresample ;
		capsfilter2 ;
		reblock ;
		whiten ;
		audioconvert ;
		capsfilter3 ;
		"segmentsrcgate()" [label="segmentsrcgate() \\n [iff veto segment list provided]", style=filled, color=lightgrey];
		tee ;
		audioamplifyr1 ;
		capsfilterr1 ;
		htgater1 [label="htgate() \\n [iff ht gate specified]", style=filled, color=lightgrey];
		tee1 ;
		audioamplifyr2 ;
		capsfilterr2 ;
		htgater2 [label="htgate() \\n [iff ht gate specified]", style=filled, color=lightgrey];
		tee2 ;
		audioamplify_rn ;
		capsfilter_rn ;
		htgate_rn [style=filled, color=lightgrey, label="htgate() \\n [iff ht gate specified]"];
		tee ;

		// nodes

		"\<src\>" -> capsfilter1 -> audioresample;
		audioresample -> capsfilter2;
		capsfilter2 -> reblock;
		reblock -> whiten;
		whiten -> audioconvert;
		audioconvert -> capsfilter3;
		capsfilter3 -> "segmentsrcgate()";
		"segmentsrcgate()" -> tee;

		tee -> audioamplifyr1 [label="Rate 1"];
		audioamplifyr1 -> capsfilterr1;
		capsfilterr1 -> htgater1;
		htgater1 -> tee1 -> "\<return\> 1";

		tee -> audioamplifyr2 [label="Rate 2"];
		audioamplifyr2 -> capsfilterr2;
		capsfilterr2 -> htgater2;
		htgater2 -> tee2 -> "\<return\> 2";

		tee ->  audioamplify_rn [label="Rate N"];
		audioamplify_rn -> capsfilter_rn;
		capsfilter_rn -> htgate_rn;
		htgate_rn -> tee_n -> "\<return\> 3";
	   }

	"""
	#
	# input sanity checks
	#

	if psd is None and not track_psd:
		raise ValueError("must enable track_psd when psd is None")
	if int(psd_fft_length) != psd_fft_length:
		raise ValueError("psd_fft_length must be an integer")
	psd_fft_length = int(psd_fft_length)

	#
	# set default whitener zero-padding if needed
	#

	if zero_pad is None:
		if FIR_WHITENER:
			# in this configuration we are not asking the
			# whitener to reassemble an output time series
			# (that we care about) so we disable zero-padding
			# to get the most information from the whitener's
			# FFT blocks.
			zero_pad = 0
		elif psd_fft_length % 4:
			raise ValueError("default whitener zero-padding requires psd_fft_length to be multiple of 4")
		else:
			zero_pad = psd_fft_length // 4

	#
	# optionally apply vetoes pre-whitening by marking samples as gaps
	#

	# optional gate on whitened h(t) amplitude.  attack and hold are
	# made to be 1/2 second or 1 sample, whichever is larger
	# FIXME:  this could be omitted if ht_gate_threshold is None, but
	# we need to collect whitened h(t) segments, however something
	# could be done to collect those if these gates aren't here.
	ht_gate_window = max(native_rate // 2, 1) # samples

	src = mkcleandata(pipeline, src, psd, block_duration, instrument, native_rate, ht_gate_threshold, ht_gate_window, veto_segments = veto_segments)

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

	head = pipeparts.mkcapsfilter(pipeline, src, "audio/x-raw, rate=[%d,MAX]" % max(rates))
	head = pipeparts.mkinterpolator(pipeline, head)
	head = pipeparts.mkaudioconvert(pipeline, head)
	head = pipeparts.mkchecktimestamps(pipeline, head, "%s_timestamps_%d_hoft" % (instrument, max(rates)))

	#
	# construct whitener.
	#

	if FIR_WHITENER:
		head = pipeparts.mktee(pipeline, head)
		whiten = pipeparts.mkwhiten(pipeline, head, fft_length = psd_fft_length, zero_pad = zero_pad, average_samples = 64, median_samples = 7, expand_gaps = True, name = "lal_whiten_%s" % instrument)
		pipeparts.mkfakesink(pipeline, whiten)

		# high pass filter
		kernel = reference_psd.one_second_highpass_kernel(max(rates), cutoff = 12)
		block_stride = block_duration * max(rates) // Gst.SECOND
		assert len(kernel) % 2 == 1, "high-pass filter length is not odd"
		head = pipeparts.mkfirbank(pipeline, head, fir_matrix = numpy.array(kernel, ndmin = 2), block_stride = block_stride, time_domain = False, latency = (len(kernel) - 1) // 2)

		# FIR filter for whitening kernel
		head = pipeparts.mktdwhiten(pipeline, head, kernel = numpy.zeros(1 + max(rates) * psd_fft_length, dtype=numpy.float64), latency = 0)

		# compute whitening kernel from PSD
		def set_fir_psd(whiten, pspec, firelem, psd_fir_kernel):
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
			kernel, phase = psd_fir_kernel.linear_phase_fir_kernel_to_minimum_phase_whitening_fir_kernel(kernel, sample_rate)
			firelem.set_property("kernel", kernel)
		firkernel = reference_psd.PSDFirKernel()
		if fir_whiten_reference_psd is not None:
			assert fir_whiten_reference_psd.f0 == 0.
			# interpolate the reference phase PSD if its
			# resolution doesn't match what we'll eventually
			# require it to be.
			if psd_fft_length != round(1. / fir_whiten_reference_psd.deltaF):
				fir_whiten_reference_psd = reference_psd.interpolate_psd(fir_whiten_reference_psd, 1. / psd_fft_length)
			# confirm that the reference phase PSD's Nyquist is
			# sufficiently high, then reduce it to the required
			# Nyquist if needed.
			assert (psd_fft_length * max(rates)) // 2 + 1 <= len(fir_whiten_reference_psd.data.data), "fir_whiten_reference_psd Nyquist too low"
			if (psd_fft_length * max(rates)) // 2 + 1 < len(fir_whiten_reference_psd.data.data):
				fir_whiten_reference_psd = lal.CutREAL8FrequencySeries(fir_whiten_reference_psd, 0, (psd_fft_length * max(rates)) // 2 + 1)
			# set the reference phase PSD
			firkernel.set_phase(fir_whiten_reference_psd)
		whiten.connect_after("notify::mean-psd", set_fir_psd, head, firkernel)

		# Gate after gaps.  the queue sizes on the control inputs
		# need only be large enough to hold the state vector
		# streams until they are required.  the streams will be
		# consumed immediately when needed, so there is no risk
		# that these queues add to the latency, so make them
		# generously large.
		# FIXME the -max(rates) extra padding is for the high pass
		# filter: NOTE it also needs to be big enough for the
		# downsampling filter, but that is typically smaller than the
		# HP filter (192 samples at Qual 9)
		# FIXME: this first queue should not be needed.  what is
		# going on!?
		if statevector is not None or dqvector is not None:
			head = pipeparts.mkqueue(pipeline, head, max_size_buffers = 0, max_size_bytes = 0, max_size_time = Gst.SECOND * (psd_fft_length + 2))
		if statevector is not None:
			head = pipeparts.mkgate(pipeline, head, control = pipeparts.mkqueue(pipeline, statevector, max_size_buffers = 0, max_size_bytes = 0, max_size_time = 0), default_state = False, threshold = 1, hold_length = -max(rates), attack_length = -max(rates) * (psd_fft_length + 1))
		if dqvector is not None:
			head = pipeparts.mkgate(pipeline, head, control = pipeparts.mkqueue(pipeline, dqvector, max_size_buffers = 0, max_size_bytes = 0, max_size_time = 0), default_state = False, threshold = 1, hold_length = -max(rates), attack_length = -max(rates) * (psd_fft_length + 1))
		head = pipeparts.mkchecktimestamps(pipeline, head, "%s_timestamps_fir" % instrument)
		#head = pipeparts.mknxydumpsinktee(pipeline, head, filename = "after_mkfirbank.txt")
	else:
		# FIXME:  we should require fir_whiten_reference_psd to be
		# None in this code path for safety, but that's hard to do
		# since the calling code would need to know what
		# environment variable is being used to select the mode,
		# and we don't want to be duplicating that code all over
		# the place

		#
		# add a reblock element.  the whitener's gap support isn't
		# 100% yet and giving it smaller input buffers works around
		# the remaining weaknesses (namely that when it sees a gap
		# buffer large enough to drain its internal history, it
		# doesn't know enough to produce a short non-gap buffer to
		# drain its history followed by a gap buffer, it just
		# produces one huge non-gap buffer that's mostly zeros).
		# this is not required in the FIR-whitener case because
		# there we don't use the whitener's output time series for
		# anything.
		#

		head = pipeparts.mkreblock(pipeline, head, block_duration = block_duration)

		head = whiten = pipeparts.mkwhiten(pipeline, head, fft_length = psd_fft_length, zero_pad = zero_pad, average_samples = 64, median_samples = 7, expand_gaps = True, name = "lal_whiten_%s" % instrument)
		# make the buffers going downstream smaller, this can
		# really help with RAM
		head = pipeparts.mkreblock(pipeline, head, block_duration = block_duration)
	#
	# enable/disable PSD tracking
	#

	whiten.set_property("psd-mode", 0 if track_psd else 1)

	#
	# install signal handler to retrieve \Delta f and f_{Nyquist}
	# whenever they are known and/or change, resample the user-supplied
	# PSD, and install it into the whitener.
	#

	if psd is not None:
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
		head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw, rate=%d, format=%s" % (max(rates), GstAudio.AudioFormat.to_string(GstAudio.AudioFormat.F64)))
	elif width == 32:
		head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw, rate=%d, format=%s" % (max(rates), GstAudio.AudioFormat.to_string(GstAudio.AudioFormat.F32)))
	else:
		raise ValueError("invalid width: %d" % width)
	head = pipeparts.mkchecktimestamps(pipeline, head, "%s_timestamps_%d_whitehoft" % (instrument, max(rates)))

	#
	# tee for highest sample rate stream
	#

	head = {max(rates): pipeparts.mktee(pipeline, head)}

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
		# downsample. make sure each output stream is unit
		# normalized, otherwise the audio resampler removes power
		# according to the rate difference and filter rolloff
		if unit_normalize:
			# NOTE the interpolator is about as good as the
			# audioresampler at quality 10, hence the 10.
			head[rate] = pipeparts.mkaudioamplify(pipeline, head[max(rates)], 1. / math.sqrt(pipeparts.audioresample_variance_gain(10, max(rates), rate)))
		else:
			head[rate] = head[max(rates)]
		head[rate] = pipeparts.mkcapsfilter(pipeline, pipeparts.mkinterpolator(pipeline, head[rate]), caps = "audio/x-raw, rate=%d" % rate)
		head[rate] = pipeparts.mkchecktimestamps(pipeline, head[rate], "%s_timestamps_%d_whitehoft" % (instrument, rate))

		head[rate] = pipeparts.mktee(pipeline, head[rate])

	#
	# done.  return value is a dictionary of tee elements indexed by
	# sample rate
	#

	return head


def mkcleandata(pipeline, src, psd, block_duration, instrument, max_rate, ht_gate_threshold, ht_gate_window, veto_segments = None):
	block_stride = block_duration * max_rate // Gst.SECOND

	# tee off
	raw = pipeparts.mktee(pipeline, src)

	# set up whitening kernel
	psd_fir = reference_psd.PSDFirKernel()
	kernel, latency, _ = psd_fir.psd_to_linear_phase_whitening_fir_kernel(psd)

	# whiten data
	white = pipeparts.mkfirbank(pipeline, pipeparts.mkqueue(pipeline, raw, max_size_buffers = 0, max_size_bytes = 0, max_size_time = 0), fir_matrix = numpy.array(kernel, ndmin = 2), block_stride = block_stride, time_domain = False, latency = latency)

	# apply vetoes
	if veto_segments is not None:
		veto_segments = segments.segmentlist(veto_segments).protract(0.25).coalesce()
		white = datasource.mksegmentsrcgate(pipeline, white, veto_segments, invert_output = True)

	# h(t) gate
	clean = datasource.mkhtgate(pipeline, pipeparts.mkqueue(pipeline, raw, max_size_buffers = 0, max_size_bytes = 0, max_size_time = 0), control = pipeparts.mkqueue(pipeline, white, max_size_buffers = 0, max_size_bytes = 0, max_size_time = 0), threshold = ht_gate_threshold, hold_length = ht_gate_window, attack_length = ht_gate_window, name = "%s_ht_gate" % instrument)

	# emit signals so that a user can latch on to them
	clean.set_property("emit-signals", True)

	return clean

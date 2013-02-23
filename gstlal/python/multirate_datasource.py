# Copyright (C) 2009--2011  Kipp Cannon, Chad Hanna, Drew Keppel
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
from glue.ligolw.utils import segments as ligolw_segments
from glue.ligolw import utils
from glue.ligolw import ligolw
from glue import segments
from pylal.datatypes import LIGOTimeGPS


def mkwhitened_multirate_src(pipeline, src, rates, instrument, psd = None, psd_fft_length = 8, ht_gate_threshold = None, veto_segments = None, seekevent = None, nxydump_segment = None, track_psd = False, block_duration = None, zero_pad = 0, width = 64):
	"""Build pipeline stage to whiten and downsample h(t)."""

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
	head = pipeparts.mkcapsfilter(pipeline, src, "audio/x-raw-float, rate=[%d,MAX]" % max(rates))
	head = pipeparts.mkcapsfilter(pipeline, pipeparts.mkresample(pipeline, head, quality = quality), "audio/x-raw-float, rate=%d" % max(rates))
	head = pipeparts.mknofakedisconts(pipeline, head)	# FIXME:  remove when resampler is patched
	head = pipeparts.mkchecktimestamps(pipeline, head, "%s_timestamps_%d_hoft" % (instrument, max(rates)))

	#
	# add a reblock element.  to reduce disk I/O gstlal_inspiral asks
	# framesrc to provide enormous buffers, and it helps reduce the RAM
	# pressure of the pipeline by slicing them up.  also, the
	# whitener's gap support isn't 100% yet and giving it smaller input
	# buffers works around the remaining weaknesses (namely that when
	# it sees a gap buffer large enough to drain its internal history,
	# it doesn't know enough to produce a short non-gap buffer to drain
	# its history followed by a gap buffer, it just produces one huge
	# non-gap buffer that's mostly zeros).
	#

	head = pipeparts.mkreblock(pipeline, head, block_duration = block_duration)

	#
	# construct whitener.
	#

	head = whiten = pipeparts.mkwhiten(pipeline, head, fft_length = psd_fft_length, zero_pad = zero_pad, average_samples = 64, median_samples = 7, expand_gaps = True, name = "lal_whiten_%s" % instrument)
	head = pipeparts.mkaudioconvert(pipeline, head)
	head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw-float, width=%d, rate=%d, channels=1" % (width, max(rates)))

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
		# use running psd
		if track_psd:
			whiten.set_property("psd-mode", 0)
		# use fixed PSD
		else:
			whiten.set_property("psd-mode", 1)

		#
		# install signal handler to retrieve \Delta f and
		# f_{Nyquist} whenever they are known and/or change,
		# resample the user-supplied PSD, and install it into the
		# whitener.
		#

		def psd_resolution_changed(elem, pspec, psd):
			# get frequency resolution and number of bins
			delta_f = elem.get_property("delta-f")
			n = int(round(elem.get_property("f-nyquist") / delta_f) + 1)
			# interpolate and install PSD
			psd = reference_psd.interpolate_psd(psd, delta_f)
			elem.set_property("mean-psd", psd.data[:n])

		whiten.connect_after("notify::f-nyquist", psd_resolution_changed, psd)
		whiten.connect_after("notify::delta-f", psd_resolution_changed, psd)
	head = pipeparts.mkchecktimestamps(pipeline, head, "%s_timestamps_%d_whitehoft" % (instrument, max(rates)))

	#
	# optionally add vetoes
	#

	if veto_segments is not None:
		head = datasource.mksegmentsrcgate(pipeline, head, veto_segments, threshold=0.1, seekevent=seekevent, invert_output=True)

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

	# FIXME this for loop was reworked to allow the h(t) gate to go after
	# audioresamplers.  There is apparently a cornercase in the
	# audioresample element that is causing a problem
	for rate in sorted(set(rates)):
		if rate < max(rates): # downsample
			head[rate] = pipeparts.mkaudioamplify(pipeline, head[max(rates)], 1/math.sqrt(pipeparts.audioresample_variance_gain(quality, max(rates), rate)))
			head[rate] = pipeparts.mkcapsfilter(pipeline, pipeparts.mkresample(pipeline, head[rate], quality = quality), caps = "audio/x-raw-float, rate=%d" % rate)
			head[rate] = pipeparts.mknofakedisconts(pipeline, head[rate])	# FIXME:  remove when resampler is patched
			head[rate] = pipeparts.mkchecktimestamps(pipeline, head[rate], "%s_timestamps_%d_whitehoft" % (instrument, rate))

		#
		# optional gate on whitened h(t) amplitude
		#

		if ht_gate_threshold is not None:
			# all h(t) gates are controlled by the same max rate control input.
			head[rate] = datasource.mkhtgate(pipeline, head[rate], control = pipeparts.mkqueue(pipeline, head[max(rates)], max_size_time = 0, max_size_bytes = 0, max_size_buffers = 0), threshold = ht_gate_threshold, hold_length = -rate // 4, attack_length = -rate // 4, name = "%s_%d_ht_gate" % (instrument, rate))

			# emit signals so that a user can latch on to them
			head[rate].set_property("emit-signals", True)
	
		head[rate] = pipeparts.mktee(pipeline, head[rate])
	#
	# done.  return value is a dictionary of tee elements indexed by
	# sample rate
	#

	return head


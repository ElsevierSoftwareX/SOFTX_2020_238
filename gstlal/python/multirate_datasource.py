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

## 
# @file
#
# A file that contains the multirate_datasource module code
#
# ###Review Status
#
# | Names                                          | Hash                                     | Date       | Diff to Head of Master      |
# | ---------------------------------------------- | ---------------------------------------- | ---------- | --------------------------- |
# | Florent, Sathya, Duncan Me, Jolien, Kipp, Chad | 8a6ea41398be79c00bdc27456ddeb1b590b0f68e | 2014-06-18 | <a href="@gstlal_cgit_diff/python/multirate_datasource.py?id=HEAD&id2=8a6ea41398be79c00bdc27456ddeb1b590b0f68e">multirate_datasource.py</a> |

# #### Actions
#
# - Is the h(t) gate really necessary? It shouldn't really be used unless
# there is something really wrong with the data. Wishlist: Tee off from 
# the control panel and record on/off (This is already done).
#
# - Task for the review team: Check what data was analysed and how much
# data was "lost" due to application of internal data quality.
#
# - There seems to be a bug in resampler (even) in the latest version of gstreamer; 
# (produces one few sample). We need to better understand the consequence of this bug.

##
# @package python.multirate_datasource
#
# multirate_datasource module

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
#	nofakedisconts [URL="\ref pipeparts.mknofakedisconts()"];
#	reblock [URL="\ref pipeparts.mkreblock()"];
#	whiten [URL="\ref pipeparts.mkwhiten()"];
#	audioconvert [URL="\ref pipeparts.mkaudioconvert()"];
#	capsfilter3 [URL="\ref pipeparts.mkcapsfilter()"];
#	"segmentsrcgate()" [URL="\ref datasource.mksegmentsrcgate()", label="segmentsrcgate() \n [iff veto segment list provided]", style=filled, color=lightgrey];
#	tee [URL="\ref pipeparts.mktee()"];
#	audioamplifyr1 [URL="\ref pipeparts.mkaudioamplify()"];
#	capsfilterr1 [URL="\ref pipeparts.mkcapsfilter()"];
#	nofakediscontsr1 [URL="\ref pipeparts.mknofakedisconts()"];
#	htgater1 [URL="\ref datasource.mkhtgate()", label="htgate() \n [iff ht gate specified]", style=filled, color=lightgrey];
#	tee1 [URL="\ref pipeparts.mktee()"];
#	audioamplifyr2 [URL="\ref pipeparts.mkaudioamplify()"];
#	capsfilterr2 [URL="\ref pipeparts.mkcapsfilter()"];
#	nofakediscontsr2 [URL="\ref pipeparts.mknofakedisconts()"];
#	htgater2 [URL="\ref datasource.mkhtgate()", label="htgate() \n [iff ht gate specified]", style=filled, color=lightgrey];
#	tee2 [URL="\ref pipeparts.mktee()"];
#	audioamplify_rn [URL="\ref pipeparts.mkaudioamplify()"];
#	capsfilter_rn [URL="\ref pipeparts.mkcapsfilter()"];
#	nofakedisconts_rn [URL="\ref pipeparts.mknofakedisconts()"];
#	htgate_rn [URL="\ref datasource.mkhtgate()", style=filled, color=lightgrey, label="htgate() \n [iff ht gate specified]"];
#	tee [URL="\ref pipeparts.mktee()"];
#
#	// nodes
#
#	"?" -> capsfilter1 -> audioresample;
#	audioresample -> capsfilter2;
#	capsfilter2 -> nofakedisconts;
#	nofakedisconts -> reblock;
#	reblock -> whiten;
#	whiten -> audioconvert;
#	audioconvert -> capsfilter3;
#	capsfilter3 -> "segmentsrcgate()";
#	"segmentsrcgate()" -> tee;
#
#	tee -> audioamplifyr1 [label="Rate 1"];
#	audioamplifyr1 -> capsfilterr1;
#	capsfilterr1 -> nofakediscontsr1;
#	nofakediscontsr1 -> htgater1;
#	htgater1 -> tee1 -> "? 1";
#
#	tee -> audioamplifyr2 [label="Rate 2"];
#	audioamplifyr2 -> capsfilterr2;
#	capsfilterr2 -> nofakediscontsr2;
#	nofakediscontsr2 -> htgater2;
#	htgater2 -> tee2 -> "? 2";
#
#	tee ->  audioamplify_rn [label="Rate N"];
#	audioamplify_rn -> capsfilter_rn;
#	capsfilter_rn -> nofakedisconts_rn;
#	nofakedisconts_rn -> htgate_rn;
#	htgate_rn -> tee_n -> "? 3";
#
# }
# @enddot
def mkwhitened_multirate_src(pipeline, src, rates, instrument, psd = None, psd_fft_length = 8, ht_gate_threshold = float("inf"), veto_segments = None, seekevent = None, nxydump_segment = None, track_psd = False, block_duration = 1 * gst.SECOND, zero_pad = 0, width = 64, unit_normalize = True):
	"""!
	Build pipeline stage to whiten and downsample h(t).

	- pipeline: the gstreamer pipeline to add this to
	- src: the gstreamer element that will be providing data to this 
	- rates: a list of the requested sample rates, e.g., [512,1024].
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
	head = pipeparts.mkcapsfilter(pipeline, src, "audio/x-raw-float, rate=[%d,MAX]" % max(rates))
	head = pipeparts.mkcapsfilter(pipeline, pipeparts.mkresample(pipeline, head, quality = quality), "audio/x-raw-float, rate=%d" % max(rates))
	head = pipeparts.mknofakedisconts(pipeline, head)	# FIXME:  remove when resampler is patched
	head = pipeparts.mkchecktimestamps(pipeline, head, "%s_timestamps_%d_hoft" % (instrument, max(rates)))

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

	head = whiten = pipeparts.mkwhiten(pipeline, head, fft_length = psd_fft_length, zero_pad = zero_pad, average_samples = 64, median_samples = 7, expand_gaps = True, name = "lal_whiten_%s" % instrument)
	head = pipeparts.mkaudioconvert(pipeline, head)
	head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw-float, width=%d, rate=%d, channels=1" % (width, max(rates)))

	# make the buffers going downstream smaller, this can really help with
	# RAM
	head = pipeparts.mkreblock(pipeline, head, block_duration = block_duration)

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
		head = datasource.mksegmentsrcgate(pipeline, head, veto_segments, seekevent=seekevent, invert_output=True)

	#
	# optional gate on whitened h(t) amplitude.  attack and hold are
	# made to be 1/4 second or 1 sample, whichever is larger
	#

	# FIXME:  this could be omitted if ht_gate_threshold is None, but
	# we need to collect whitened h(t) segments, however something
	# could be done to collect those if these gates aren't here.
	ht_gate_window = max(max(rates) // 4, 1)	# samples
	head = datasource.mkhtgate(pipeline, head, threshold = ht_gate_threshold if ht_gate_threshold is not None else float("+inf"), hold_length = ht_gate_window, attack_length = ht_gate_window, name = "%s_ht_gate" % instrument)
	# emit signals so that a user can latch on to them
	head.set_property("emit-signals", True)

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

	quality = 9
	for rate in sorted(set(rates))[:-1]:
		# downsample. make sure each output stream is unit
		# normalized, otherwise the audio resampler removes power
		# according to the rate difference and filter rolloff
		if unit_normalize:
			head[rate] = pipeparts.mkaudioamplify(pipeline, head[max(rates)], 1. / math.sqrt(pipeparts.audioresample_variance_gain(quality, max(rates), rate)))
		else:
			head[rate] = head[max(rates)]
		head[rate] = pipeparts.mkcapsfilter(pipeline, pipeparts.mkresample(pipeline, head[rate], quality = quality), caps = "audio/x-raw-float, rate=%d" % rate)
		head[rate] = pipeparts.mknofakedisconts(pipeline, head[rate])	# FIXME:  remove when resampler is patched
		head[rate] = pipeparts.mkchecktimestamps(pipeline, head[rate], "%s_timestamps_%d_whitehoft" % (instrument, rate))

		head[rate] = pipeparts.mktee(pipeline, head[rate])

	#
	# done.  return value is a dictionary of tee elements indexed by
	# sample rate
	#

	return head


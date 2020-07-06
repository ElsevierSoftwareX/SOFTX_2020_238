# Copyright (C) 2009--2013  Kipp Cannon, Chad Hanna, Drew Keppel
# Copyright (C) 2017        Sydney J. Chamberlin, Patrick Godwin, Chad Hanna
# Copyright (C) 2018--2020  Patrick Godwin
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


import math

import numpy

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstAudio", "1.0")
from gi.repository import GObject
from gi.repository import Gst, GstAudio
GObject.threads_init()
Gst.init(None)

import lal

from gstlal import reference_psd
from gstlal import pipeparts
from gstlal import datasource
from gstlal.snax import utils


# framexmit ports in use on the LDG
# Look-up table to map instrument name to framexmit multicast address and
# port
#
# used in mkbasicmultisrc()
#
# FIXME:  this is only here temporarily while we test this approach to data
# aquisition.  obviously we can't hard-code this stuff
#
framexmit_ports = {
	"CIT": {
		"H1": ("224.3.2.1", 7096),
		"L1": ("224.3.2.2", 7097),
		"V1": ("224.3.2.3", 7098),
	}
}

PSD_FFT_LENGTH = 32
NATIVE_RATE_CUTOFF = 128


def set_fir_psd(whiten, pspec, firelem, psd_fir_kernel):
	"""
	compute whitening kernel from PSD
	"""
	psd_data = numpy.array(whiten.get_property("mean-psd"))
	psd = lal.CreateREAL8FrequencySeries(
		name="psd",
		epoch=lal.LIGOTimeGPS(0),
		f0=0.0,
		deltaF=whiten.get_property("delta-f"),
		sampleUnits=lal.Unit(whiten.get_property("psd-units")),
		length=len(psd_data),
	)
	psd.data.data = psd_data
	kernel, latency, sample_rate = psd_fir_kernel.psd_to_linear_phase_whitening_fir_kernel(psd)
	kernel, theta = psd_fir_kernel.linear_phase_fir_kernel_to_minimum_phase_whitening_fir_kernel(
		kernel, sample_rate
	)
	kernel -= numpy.mean(kernel)  # subtract DC offset from signal
	firelem.set_property("fir-matrix", numpy.array(kernel, ndmin=2))


def psd_units_or_resolution_changed(elem, pspec, psd):
	"""
	install signal handler to retrieve \Delta f and f_{Nyquist}
	whenever they are known and/or change, resample the user-supplied
	PSD, and install it into the whitener.
	"""
	# make sure units are set, compute scale factor
	units = lal.Unit(elem.get_property("psd-units"))
	if units == lal.DimensionlessUnit:
		return

	# get frequency resolution and number of bins
	delta_f = elem.get_property("delta-f")
	n = int(round(elem.get_property("f-nyquist") / delta_f) + 1)

	# interpolate, rescale, and install PSD
	psd = reference_psd.interpolate_psd(psd, delta_f)
	scale = float(psd.sampleUnits / units)
	elem.set_property("mean-psd", psd.data.data[:n] * scale)


def mktimequeue(pipeline, src, max_time=1):
	"""
	mkqueue with simplified time arguments
	"""
	max_size_time = Gst.SECOND * max_time
	return pipeparts.mkqueue(
		pipeline,
		src,
		max_size_buffers=0,
		max_size_bytes=0,
		max_size_time=max_size_time,
	)


def mkmultisrc(pipeline, data_source_info, channels, verbose=False):
	"""
	All the things for reading real or simulated channel data in one place.

	Consult the append_options() function and the DataSourceInfo class

	This src in general supports only one instrument although
	DataSourceInfo contains dictionaries of multi-instrument things.  By
	specifying the channels when calling this function you will only process
	the channels specified. A code wishing to have multiple multisrcs
	will need to call this multiple times with different sets of channels specified.
	"""

	ifo = data_source_info.instrument
	frame_segs = data_source_info.frame_segments[ifo]

	if data_source_info.data_source in ("white", "silence", "white_live"):
		head = {}
		for channel in channels:
			rate = data_source_info.channel_dict[channel]["fsamp"]
			t_offset = int(data_source_info.seg[0]) * Gst.SECOND

			if data_source_info.data_source == "white":
				head[channel] = pipeparts.mkfakesrc(
					pipeline,
					instrument=ifo,
					channel_name=channel,
					volume=1.0,
					rate=rate,
					timestamp_offset=t_offset,
				)
			elif data_source_info.data_source == "silence":
				head[channel] = pipeparts.mkfakesrc(
					pipeline,
					instrument=ifo,
					channel_name=channel,
					rate=rate,
					timestamp_offset=t_offset,
				)
			elif data_source_info.data_source == "white_live":
				head[channel] = pipeparts.mkfakesrc(
					pipeline,
					instrument=ifo,
					channel_name=channel,
					volume=1.0,
					is_live=True,
					rate=rate,
					timestamp_offset=t_offset,
				)

	elif data_source_info.data_source == "frames":
		src = pipeparts.mklalcachesrc(
			pipeline,
			location=data_source_info.frame_cache,
			cache_src_regex=ifo[0],
			cache_dsc_regex=ifo,
			blocksize=1048576,
		)
		demux = pipeparts.mkframecppchanneldemux(
			pipeline,
			src,
			do_file_checksum=False,
			skip_bad_files=True,
			channel_list=channels,
		)

		# allow frame reading and decoding to occur in a different thread
		head = {}
		for channel in channels:
			src = pipeparts.mkreblock(pipeline, None, block_duration=(4 * Gst.SECOND))
			pipeparts.src_deferred_link(demux, channel, src.get_static_pad("sink"))
			src = mktimequeue(pipeline, src, max_time=8)

			if frame_segs:
				# FIXME:  make segmentsrc generate segment samples at the channel sample rate?
				# FIXME:  make gate leaky when I'm certain that will work.
				src = pipeparts.mkgate(
					pipeline,
					src,
					threshold=1,
					control=pipeparts.mksegmentsrc(pipeline, frame_segs),
					name="%s_frame_segments_gate" % channel,
				)
				pipeparts.framecpp_channeldemux_check_segments.set_probe(
					src.get_static_pad("src"),
					frame_segs,
				)

			# fill in holes, skip duplicate data
			head[channel] = pipeparts.mkaudiorate(pipeline, src, skip_to_first=True, silent=False)

	elif data_source_info.data_source in ("framexmit", "lvshm", "white_live"):
		if data_source_info.data_source == "lvshm":
			# FIXME make wait_time adjustable through web interface
			#       or command line or both
			src = pipeparts.mklvshmsrc(
				pipeline,
				shm_name=data_source_info.shm_part_dict[ifo],
				assumed_duration=data_source_info.shm_assumed_duration,
				blocksize=data_source_info.shm_block_size,
				wait_time=120,
			)
		elif data_source_info.data_source == "framexmit":
			src = pipeparts.mkframexmitsrc(
				pipeline,
				multicast_iface=data_source_info.framexmit_iface,
				multicast_group=data_source_info.framexmit_addr[ifo][0],
				port=data_source_info.framexmit_addr[ifo][1],
				wait_time=120,
			)
		else:
			# impossible code path
			raise ValueError(data_source_info.data_source)

		demux = pipeparts.mkframecppchanneldemux(
			pipeline,
			src,
			do_file_checksum=False,
			skip_bad_files=True,
			channel_list=channels,
		)

		# channels
		head = {}
		for channel in channels:
			head[channel] = mktimequeue(pipeline, None, max_time=60)
			pipeparts.src_deferred_link(demux, channel, head[channel].get_static_pad("sink"))
			if data_source_info.latency_output:
				head[channel] = pipeparts.mklatency(
					pipeline,
					head[channel],
					name="stage1_afterFrameXmit_%s" % channel,
				)

			# fill in holes, skip duplicate data
			head[channel] = pipeparts.mkaudiorate(
				pipeline,
				head[channel],
				skip_to_first=True,
				silent=False,
			)

			# 10 minutes of buffering
			head[channel] = mktimequeue(pipeline, head[channel], max_time=600)

	else:
		raise ValueError("invalid data_source: %s" % data_source_info.data_source)

	for channel in head:
		head[channel] = pipeparts.mkaudioconvert(pipeline, head[channel])
		if verbose:
			head[channel] = pipeparts.mkprogressreport(
				pipeline,
				head[channel],
				"datasource_progress_%s" % channel,
			)

	return head


def mkcondition(
	pipeline,
	src,
	rates,
	native_rate,
	instrument,
	psd=None,
	psd_fft_length=PSD_FFT_LENGTH,
	veto_segments=None,
	nxydump_segment=None,
	track_psd=True,
	block_duration=0.25 * Gst.SECOND,
	width=64,
	channel_name="hoft",
):
	"""
	Build pipeline stage to whiten and downsample auxiliary channels.

	- pipeline: the gstreamer pipeline to add this to
	- src: the gstreamer element that will be providing data to this
	- rates: a list of the requested sample rates, e.g., [512,1024].
	- native_rate: Native sampling rate of channel
	- instrument: the instrument to process
	- psd: a psd frequency series
	- psd_fft_length: length of fft used for whitening
	- veto_segments: segments to mark as gaps after whitening
	- nxydump_segment: segment to dump to disk after whitening
	- track_psd: decide whether to dynamically track the spectrum or use the fixed spectrum provided
	- width: type convert to either 32 or 64 bit float
	- channel_name: channel to whiten and downsample

	"""

	# sanity checks
	if psd is None and not track_psd:
		raise ValueError("must enable track_psd when psd is None")
	if int(psd_fft_length) != psd_fft_length:
		raise ValueError("psd_fft_length must be an integer")
	psd_fft_length = int(psd_fft_length)

	# down-sample to highest of target sample rates.
	max_rate = max(rates)
	head = pipeparts.mkcapsfilter(pipeline, src, "audio/x-raw, rate=[%d,MAX]" % max_rate)
	head = pipeparts.mkinterpolator(pipeline, head)
	head = pipeparts.mkaudioconvert(pipeline, head)

	# construct whitener
	zero_pad = psd_fft_length // 4
	head = pipeparts.mktee(pipeline, head)
	whiten = pipeparts.mkwhiten(
		pipeline,
		head,
		fft_length=psd_fft_length,
		zero_pad=zero_pad,
		average_samples=64,
		median_samples=7,
		expand_gaps=True,
		name="%s_%s_lalwhiten" % (instrument, channel_name),
	)
	pipeparts.mkfakesink(pipeline, whiten)

	# high pass filter
	block_stride = int(block_duration * max_rate // Gst.SECOND)
	if native_rate >= NATIVE_RATE_CUTOFF:
		kernel = reference_psd.one_second_highpass_kernel(max_rate, cutoff=12)
		assert len(kernel) % 2 == 1, "high-pass filter length is not odd"
		head = pipeparts.mkfirbank(
			pipeline,
			head,
			fir_matrix=numpy.array(kernel, ndmin=2),
			block_stride=block_stride,
			time_domain=False,
			latency=(len(kernel) - 1) // 2,
		)

	# FIR filter for whitening kernel
	head = pipeparts.mkfirbank(
		pipeline,
		head,
		fir_matrix=numpy.zeros((1, 1 + max_rate * psd_fft_length), dtype=numpy.float64),
		block_stride=block_stride,
		time_domain=False,
		latency=0,
	)
	whiten.connect_after("notify::mean-psd", set_fir_psd, head, reference_psd.PSDFirKernel())

	# extra queue to deal with gaps produced by segmentsrc
	head = mktimequeue(pipeline, head, max_time=(psd_fft_length + 2))

	# Drop initial data to let the PSD settle
	head = pipeparts.mkdrop(pipeline, head, drop_samples=16 * psd_fft_length * max_rate)

	# enable/disable PSD tracking
	whiten.set_property("psd-mode", 0 if track_psd else 1)

	# install signal handler to retrieve \Delta f and f_{Nyquist}
	# whenever they are known and/or change, resample the user-supplied
	# PSD, and install it into the whitener.
	if psd is not None:
		whiten.connect_after("notify::f-nyquist", psd_units_or_resolution_changed, psd)
		whiten.connect_after("notify::delta-f", psd_units_or_resolution_changed, psd)
		whiten.connect_after("notify::psd-units", psd_units_or_resolution_changed, psd)

	# convert to desired precision
	head = pipeparts.mkaudioconvert(pipeline, head)
	if width == 64:
		format_str = GstAudio.AudioFormat.to_string(GstAudio.AudioFormat.F64)
	elif width == 32:
		format_str = GstAudio.AudioFormat.to_string(GstAudio.AudioFormat.F32)
	else:
		raise ValueError("invalid width: %d" % width)

	head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw, rate=%d, format=%s" % (max_rate, format_str))

	# optionally add vetoes
	if veto_segments is not None:
		head = datasource.mksegmentsrcgate(pipeline, head, veto_segments, invert_output=True)

	return head


def mkmultiband(pipeline, src, rates, native_rate, instrument, min_rate=128, quality=10):
	"""
	Build pipeline stage to split stream into
	multiple frequency bands by powers of 2.
	return value is a dictionary of elements indexed by sample rate
	"""
	head = {}
	tee = pipeparts.mktee(pipeline, src)
	for rate in rates:
		head[rate] = mktimequeue(pipeline, tee, max_time=8)

	# down-sample whitened time series to remaining target sample rates
	for rate in sorted(set(rates))[:-1]:
		gain = 1.0 / math.sqrt(pipeparts.audioresample_variance_gain(quality, max(rates), rate))
		head[rate] = pipeparts.mkaudioamplify(pipeline, head[rate], gain)
		head[rate] = pipeparts.mkcapsfilter(
			pipeline,
			pipeparts.mkinterpolator(pipeline, head[rate]),
			caps="audio/x-raw, rate=%d" % max(min_rate, rate),
		)

	return head


def mkextract(
	pipeline,
	src,
	instrument,
	channel,
	rate,
	waveforms,
	snr_threshold=5.5,
	sample_rate=1,
	nxydump_segment=None,
	feature_mode="timeseries",
	latency_output=False,
):
	"""
	Extract features from whitened timeseries.
	"""
	# determine whether to do time-domain or frequency-domain convolution
	n_samples = waveforms[channel].sample_pts(rate)
	time_domain = (n_samples * rate) < (5 * n_samples * numpy.log2(rate))

	# create fir bank from waveforms
	fir_matrix = numpy.array(list(waveforms[channel].generate_templates(rate, sampling_rate=rate)))
	head = mktimequeue(pipeline, src, max_time=30)
	head = pipeparts.mkfirbank(
		pipeline,
		head,
		fir_matrix=fir_matrix,
		time_domain=time_domain,
		block_stride=int(rate),
		latency=waveforms[channel].latency(rate)
	)

	# add queues, change stream format, add tags
	if latency_output:
		head = pipeparts.mklatency(
			pipeline,
			head,
			name=utils.latency_name('afterFIRbank', 4, channel, rate)
		)

	head = pipeparts.mkqueue(pipeline, head, max_size_buffers=1, max_size_bytes=0, max_size_time=0)
	head = pipeparts.mktogglecomplex(pipeline, head)
	head = pipeparts.mkcapsfilter(pipeline, head, caps="audio/x-raw, format=Z64LE, rate=%i" % rate)
	head = pipeparts.mktaginject(
		pipeline,
		head,
		"instrument=%s,channel-name=%s" % (instrument, channel)
	)

	# dump segments to disk if specified
	tee = pipeparts.mktee(pipeline, head)
	if nxydump_segment:
		nxydump_name = "snrtimeseries_%s_%s.txt" % (channel, repr(rate))
		pipeparts.mknxydumpsink(
			pipeline,
			pipeparts.mkqueue(pipeline, tee),
			nxydump_name,
			segment=nxydump_segment
		)

	# extract features from time series
	if feature_mode == 'timeseries':
		head = pipeparts.mktrigger(pipeline, tee, int(rate // sample_rate), max_snr=True)
	elif feature_mode == 'etg':
		head = pipeparts.mktrigger(pipeline, tee, rate, snr_thresh=snr_threshold)

	return head

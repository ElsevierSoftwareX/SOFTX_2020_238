# Copyright (C) 2010  Kipp Cannon, Chad Hanna
# Copyright (C) 2009  Kipp Cannon, Chad Hanna
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


from gstlal import pipeparts
from gstlal import cbc_template_fir


#
# =============================================================================
#
#                              Pipeline Elements
#
# =============================================================================
#


#
# sum-of-squares aggregator
#


def mkcontrolsnksrc(pipeline, rate, verbose = False, suffix = None):
	snk = gst.element_factory_make("lal_adder")
	snk.set_property("sync", True)
	pipeline.add(snk)
	src = pipeparts.mkcapsfilter(pipeline, snk, "audio/x-raw-float, rate=%d" % rate)
	if verbose:
		src = pipeparts.mkprogressreport(pipeline, src, "progress_sumsquares%s" % (suffix and "_%s" % suffix or ""))
	src = pipeparts.mktee(pipeline, src)
	return snk, src


#
# data source
#

def mkLLOIDbasicsrc(pipeline, seekevent, instrument, detector, fake_data = False, online_data = False, injection_filename = None, verbose = False):
	#
	# data source and progress report
	#
	print detector.block_size
	if fake_data:
		head = pipeparts.mkfakeLIGOsrc(pipeline, instrument = instrument, blocksize = detector.block_size, location = detector.frame_cache, channel_name = detector.channel)
	elif online_data:
		head = pipeparts.mkonlinehoftsrc(pipeline, instrument = instrument)
	else:
		head = pipeparts.mkframesrc(pipeline, instrument = instrument, blocksize = detector.block_size, location = detector.frame_cache, channel_name = detector.channel)

	if head.set_state(gst.STATE_READY) != gst.STATE_CHANGE_SUCCESS:
		raise RuntimeError, "Element %s did not want to enter ready state" % head.get_name()
	if not head.send_event(seekevent):
		raise RuntimeError, "Element %s did not handle seek event" % head.get_name()
	head = pipeparts.mkaudioconvert(pipeline, head)
	if verbose:
		head = pipeparts.mkprogressreport(pipeline, head, "progress_src_%s" % instrument)

	#
	# optional injections
	#

	if injection_filename is not None:
		head = pipeparts.mkinjections(pipeline, head, injection_filename)

	return head


def mkLLOIDsrc(pipeline, seekevent, instrument, detector, rates, psd = None, psd_fft_length = 8, fake_data = False, online_data = False, injection_filename = None, verbose = False, nxydump_segment = None):
	head = mkLLOIDbasicsrc(pipeline, seekevent, instrument, detector, fake_data=fake_data, online_data=online_data, injection_filename = injection_filename, verbose=verbose)

	#
	# down-sample to highest of target sample rates.  note:  there is
	# no check that this is, infact, *down*-sampling.  if the source
	# time series has a lower sample rate this will up-sample the data.
	# up-sampling will probably interact poorly with the whitener as it
	# will likely add (possibly significant) numerical noise when it
	# amplifies the non-existant high-frequency components
	#

	source_rate = max(rates)
	head = pipeparts.mkcapsfilter(pipeline, pipeparts.mkresample(pipeline, pipeparts.mkqueue(pipeline, head), quality = 9), "audio/x-raw-float, rate=%d" % source_rate)

	#
	# whiten
	#

	if psd is not None:
		#
		# use fixed PSD
		#

		head = pipeparts.mkwhiten(pipeline, head, fft_length = psd_fft_length, psd_mode = 1)

		#
		# install signal handler to retrieve \Delta f when it is
		# known, resample the user-supplied PSD, and install it
		# into the whitener.
		#

		def f_nyquist_changed(elem, pspec, psd):
			# get frequency resolution and number of bins
			delta_f = elem.get_property("delta-f")
			n = int(round(elem.get_property("f-nyquist") / delta_f) + 1)
			# interpolate and install PSD
			psd = cbc_template_fir.interpolate_psd(psd, delta_f)
			elem.set_property("mean-psd", psd.data[:n])

		head.connect_after("notify::f-nyquist", f_nyquist_changed, psd)
	else:
		#
		# use running average PSD
		#

		head = pipeparts.mkwhiten(pipeline, head, fft_length = psd_fft_length)
	head = pipeparts.mknofakedisconts(pipeline, head)	# FIXME:  remove after basetransform behaviour fixed

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
	# By design, the input to the orthogonal filter
	# banks is pre-whitened, so it is unit variance over short periods of time.
	# However, resampling it reduces the variance by a small, sample rate
	# dependent factor.  The audioamplify element applies a correction factor
	# that restores the input's unit variance.
	#

	quality = 9
	head = {source_rate: pipeparts.mktee(pipeline, head)}
	for rate in sorted(rates, reverse = True)[1:]:	# all but the highest rate
		head[rate] = pipeparts.mktee(pipeline, pipeparts.mkcapsfilter(pipeline, pipeparts.mkresample(pipeline, pipeparts.mkaudioamplify(pipeline, head[source_rate], 1/math.sqrt(pipeparts.audioresample_variance_gain(quality, source_rate, rate))), quality = quality), "audio/x-raw-float, rate=%d" % rate))

	#
	# done.  return value is a dictionary of tee elements indexed by
	# sample rate
	#

	#for rate, elem in head.items():
	#	pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, elem), "src_%d.dump" % rate, segment = nxydump_segment)
	return head


#
# one instrument, one template bank
#


def mkLLOIDbranch(pipeline, src, bank, bank_fragment, (control_snk, control_src), gate_attack_length, gate_hold_length):
	logname = "%s_%d_%d" % (bank.logname, bank_fragment.start, bank_fragment.end)

	#
	# FIR filter bank
	#
	# FIXME:  why the -1?  without it the pieces don't match but I
	# don't understand where this offset comes from.  it might really
	# need to be here, or it might be a symptom of a bug elsewhere.
	# figure this out.

	src = pipeparts.mktee(pipeline, pipeparts.mkreblock(pipeline, pipeparts.mkfirbank(pipeline, src, latency = -int(round(bank_fragment.start * bank_fragment.rate)) - 1, fir_matrix = bank_fragment.orthogonal_template_bank)))
	#pipeparts.mkvideosink(pipeline, pipeparts.mkcapsfilter(pipeline, pipeparts.mkhistogram(pipeline, src), "video/x-raw-rgb, width=640, height=480, framerate=1/4"))
	#pipeparts.mkogmvideosink(pipeline, pipeparts.mkcapsfilter(pipeline, pipeparts.mkchannelgram(pipeline, pipeparts.mkqueue(pipeline, src), plot_width = .125), "video/x-raw-rgb, width=640, height=480, framerate=64/1"), "orthosnr_channelgram_%s.ogv" % logname, verbose = True)

	#
	# compute weighted sum-of-squares, feed to sum-of-squares
	# aggregator
	#

	pipeparts.mkchecktimestamps(pipeline, pipeparts.mkresample(pipeline, pipeparts.mkqueue(pipeline, pipeparts.mksumsquares(pipeline, src, weights = bank_fragment.sum_of_squares_weights)), quality = 9), name = "timestamps_%s_after_sumsquare_resampler" % logname).link(control_snk)

	#
	# use sum-of-squares aggregate as gate control for orthogonal SNRs
	#

	src = pipeparts.mkgate(pipeline, pipeparts.mkqueue(pipeline, src), control = pipeparts.mkqueue(pipeline, control_src), threshold = bank.gate_threshold, attack_length = gate_attack_length, hold_length = gate_hold_length)
	src = pipeparts.mkchecktimestamps(pipeline, src, name = "timestamps_%s_after_gate" % logname)

	#
	# buffer orthogonal SNRs
	#
	# FIXME:  teach the collectpads object not to wait for buffers on
	# pads whose segments have not yet been reached by the input on the
	# other pads.  then this large queue buffer will not be required
	# because streaming can begin through the downstream adders without
	# waiting for input from all upstream elements.

	src = pipeparts.mkqueue(pipeline, src, max_size_buffers = 0, max_size_bytes = 0, max_size_time = gst.SECOND )

	#
	# reconstruct physical SNRs
	#

	src = pipeparts.mkmatrixmixer(pipeline, src, matrix = bank_fragment.mix_matrix)
	src = pipeparts.mkresample(pipeline, src, quality = 9)
	src = pipeparts.mknofakedisconts(pipeline, src)	# FIXME:  remove after basetransform behaviour fixed
	src = pipeparts.mkchecktimestamps(pipeline, src, name = "timestamps_%s_after_snr_resampler" % logname)

	#
	# done
	#
	# FIXME:  find a way to use less memory without this hack

	del bank_fragment.orthogonal_template_bank
	del bank_fragment.sum_of_squares_weights
	del bank_fragment.mix_matrix
	return src


def mkLLOIDsingle(pipeline, hoftdict, instrument, detector, bank, control_snksrc, verbose = False, nxydump_segment = None):
	logname = "%s%s" % (instrument, (bank.logname and "_%s" % bank.logname or ""))

	#
	# parameters
	#

	output_rate = max(bank.get_rates())
	autocorrelation_length = bank.autocorrelation_bank.shape[1]
	autocorrelation_latency = -(autocorrelation_length - 1) / 2

	#
	# snr aggregator
	#

	snr = gst.element_factory_make("lal_adder")
	snr.set_property("sync", True)
	pipeline.add(snr)

	#
	# loop over template bank slices
	#

	for bank_fragment in bank.bank_fragments:
		branch_snr = mkLLOIDbranch(
			pipeline,
			# FIXME:  the size isn't ideal:  the correct value
			# depends on how much data is accumulated in the
			# firbank element, and the value here is only
			# approximate and not tied to the fir bank
			# parameters so might not work if those change
			pipeparts.mkqueue(
				pipeline,
				pipeparts.mkdelay(
					pipeline,
					hoftdict[bank_fragment.rate],
					delay = int(round( (bank.filter_length - bank_fragment.end)*bank_fragment.rate )) ),
				max_size_bytes = 0,
				max_size_buffers = 0,
				max_size_time = 4 * int(math.ceil(bank.filter_length)) * gst.SECOND),
			bank,
			bank_fragment,
			control_snksrc,
			int(math.ceil(-autocorrelation_latency * (float(bank_fragment.rate) / output_rate))),
			int(math.ceil(-autocorrelation_latency * (float(bank_fragment.rate) / output_rate)))
		)
		#branch_snr = pipeparts.mktee(pipeline, branch_snr)
		#pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, branch_snr), "snr_%s_%02d.dump" % (logname, bank_fragment.start), segment = nxydump_segment)
		branch_snr.link(snr)

	#
	# snr
	#

	snr = pipeparts.mktee(pipeline, pipeparts.mktogglecomplex(pipeline, pipeparts.mkcapsfilter(pipeline, snr, "audio/x-raw-float, rate=%d" % output_rate)))
	#pipeparts.mknxydumpsink(pipeline, pipeparts.mktogglecomplex(pipeline, pipeparts.mkqueue(pipeline, snr)), "snr_%s.dump" % logname, segment = nxydump_segment)
	#pipeparts.mkogmvideosink(pipeline, pipeparts.mkcapsfilter(pipeline, pipeparts.mkchannelgram(pipeline, pipeparts.mkqueue(pipeline, snr), plot_width = .125), "video/x-raw-rgb, width=640, height=480, framerate=64/1"), "snr_channelgram_%s.ogv" % logname, audiosrc = pipeparts.mkaudioamplify(pipeline, pipeparts.mkqueue(pipeline, hoftdict[output_rate], max_size_time = 2 * int(math.ceil(bank.filter_length)) * gst.SECOND), 0.125), verbose = True)

	#
	# \chi^{2}
	#

	chisq = pipeparts.mkautochisq(pipeline, pipeparts.mkqueue(pipeline, snr), autocorrelation_matrix = bank.autocorrelation_bank, latency = autocorrelation_latency, snr_thresh=bank.snr_threshold)
	#chisq = pipeparts.mktee(pipeline, chisq)
	#pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, chisq), "chisq_%s.dump" % logname, segment = nxydump_segment)
	# FIXME:  find a way to use less memory without this hack
	del bank.autocorrelation_bank

	#
	# trigger generator and progress report
	#

	head = pipeparts.mktriggergen(pipeline, pipeparts.mkqueue(pipeline, snr), chisq, bank.template_bank_filename, bank.snr_threshold, bank.sigmasq)
	if verbose:
		head = pipeparts.mkprogressreport(pipeline, head, "progress_xml_%s" % logname)

	#
	# done
	#

	return head


#
# many instruments, many template banks
#


def mkLLOIDmulti(pipeline, seekevent, detectors, banks, psd, psd_fft_length = 8, fake_data = False, online_data = False, injection_filename = None, verbose = False, nxydump_segment = None):
	#
	# xml stream aggregator
	#

	# Input selector breaks seeks.  For a single detector and single template bank,
	# we don't need an input selector.  This is an ugly kludge to make seeks work
	# in this special (and very high priority) case.
	needs_input_selector = (len(detectors) * len(banks) > 1)
	if needs_input_selector:
		nto1 = gst.element_factory_make("input-selector")
		nto1.set_property("select-all", True)
		pipeline.add(nto1)

	#
	# loop over instruments and template banks
	#

	for instrument in detectors:
		rates = set(rate for bank in banks for rate in bank.get_rates())
		hoftdict = mkLLOIDsrc(pipeline, seekevent, instrument, detectors[instrument], rates, psd = psd, psd_fft_length = psd_fft_length, fake_data = fake_data, online_data = online_data, injection_filename = injection_filename, verbose = verbose, nxydump_segment = nxydump_segment)
		for bank in banks:
			control_snksrc = mkcontrolsnksrc(pipeline, max(bank.get_rates()), verbose = verbose, suffix = "%s%s" % (instrument, (bank.logname and "_%s" % bank.logname or "")))
			#pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, control_snksrc[1]), "control_%s.dump" % bank.logname, segment = nxydump_segment)
			head = mkLLOIDsingle(
				pipeline,
				hoftdict,
				instrument,
				detectors[instrument],
				bank,
				control_snksrc,
				verbose = verbose,
				nxydump_segment = nxydump_segment
			)
			if needs_input_selector:
				pipeparts.mkqueue(pipeline, head).link(nto1)

	#
	# done
	#

	if needs_input_selector:
		return nto1
	else:
		return pipeparts.mkqueue(pipeline, head)

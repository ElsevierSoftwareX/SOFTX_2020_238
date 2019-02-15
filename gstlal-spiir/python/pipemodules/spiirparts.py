
# Copyright (C) 2009--2013  Kipp Cannon, Chad Hanna, Drew Keppel
# Copyright (C) 2013, 2014  Qi Chu
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
#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


# The following snippet is taken from http://gstreamer.freedesktop.org/wiki/FAQ#Mypygstprogramismysteriouslycoredumping.2Chowtofixthis.3F
import pygtk
pygtk.require("2.0")
import gobject
gobject.threads_init()
import pygst
pygst.require('0.10')
import gst

import math
import sys
import numpy as np
import warnings
import StringIO
from gstlal.pipeio import repack_complex_array_to_real 

from gstlal import datasource
from gstlal import multirate_datasource
from gstlal import pipeio
from gstlal import pipeparts
from gstlal import pipemodules
from gstlal import simulation
from gstlal.pipemodules import snglrate_datasource
from gstlal.spiirbank import spiir_utils


def mkSPIIRmulti(pipeline, detectors, banks, psd, psd_fft_length = 8, ht_gate_threshold = None, veto_segments = None, verbose = False, nxydump_segment = None, nxydump_directory = '.', chisq_type = 'autochisq', track_psd = False, block_duration = gst.SECOND, blind_injections = None, peak_thresh = 4, gpu_acc = False):
	#
	# check for recognized value of chisq_type
	#

	if chisq_type not in ['autochisq' or 'autochisq_spearman']:
		raise ValueError("chisq_type must be either 'autochisq' or 'autochisq_spearman', given %s" % chisq_type)

	#
	# extract segments from the injection file for selected reconstruction
	#

	if detectors.injection_filename is not None:
		inj_seg_list = simulation.sim_inspiral_to_segment_list(detectors.injection_filename)
	else:
		inj_seg_list = None
		#
		# Check to see if we are specifying blind injections now that we know
		# we don't want real injections. Setting this
		# detectors.injection_filename will ensure that injections are added
		# but won't only reconstruct injection segments.
		#
		detectors.injection_filename = blind_injections

	#
	# construct dictionaries of whitened, conditioned, down-sampled
	# h(t) streams
	#

	hoftdicts = {}
	for instrument in detectors.channel_dict:
		src, statevector, dqvector = datasource.mkbasicsrc(pipeline, detectors, instrument, verbose=verbose)
		rates = set(rate for bank in banks[instrument] for rate in bank.get_rates()) # FIXME what happens if the rates are not the same?
		if veto_segments is not None and instrument in veto_segments.keys():
			hoftdicts[instrument] = multirate_datasource.mkwhitened_multirate_src(pipeline, src, rates, instrument, psd = psd[instrument], psd_fft_length = psd_fft_length, ht_gate_threshold = ht_gate_threshold, veto_segments = veto_segments[instrument], seekevent = detectors.seekevent, nxydump_segment = nxydump_segment, track_psd = track_psd, zero_pad = 0, width = 32)
		else:
			hoftdicts[instrument] = multirate_datasource.mkwhitened_multirate_src(pipeline, src, rates, instrument, psd = psd[instrument], psd_fft_length = psd_fft_length, ht_gate_threshold = ht_gate_threshold, veto_segments = None, seekevent = detectors.seekevent, nxydump_segment = nxydump_segment, track_psd = track_psd, zero_pad = 0, width = 32)

	#
	# construct trigger generators
	#
	# format of banklist : {'H1': <H1Bank0>, <H1Bank1>..;
	#			'L1': <L1Bank0>, <L1Bank1>..;..}
	# format of bank: <H1bank0>


	triggersrcs = dict((instrument, set()) for instrument in hoftdicts)

	for instrument, bank in [(instrument, bank) for instrument, banklist in banks.items() for bank in banklist]:
		suffix = "%s%s" % (instrument, (bank.logname and "_%s" % bank.logname or ""))
		snr = mkSPIIRhoftToSnrSlices(
			pipeline,
			hoftdicts[instrument],
			bank,
			instrument,
			verbose = verbose,
			nxydump_segment = nxydump_segment,
			quality = 1,
			gpu_acc = gpu_acc
		)
		snr = pipeparts.mkchecktimestamps(pipeline, snr, "timestamps_%s_snr" % suffix)

		snr = pipeparts.mktogglecomplex(pipeline, snr)
		snr = pipeparts.mktee(pipeline, snr)
		# FIXME you get a different trigger generator depending on the chisq calculation :/
		if chisq_type == 'autochisq':
			# FIXME don't hardcode
			# peak finding window (n) in samples is one second at max rate, ie max(rates)
			# FIXME: bank.snr_thresh is removed, use peak_thresh instead
			head = pipeparts.mkitac(pipeline, snr, max(rates), bank.template_bank_filename, autocorrelation_matrix = bank.autocorrelation_bank, mask_matrix = bank.autocorrelation_mask, snr_thresh = peak_thresh, sigmasq = bank.sigmasq)
			if verbose:
				head = pipeparts.mkprogressreport(pipeline, head, "progress_xml_%s" % suffix)
			triggersrcs[instrument].add(head)
		elif chisq_type == 'autochisq_spearman':
			# FIXME don't hardcode
			# peak finding window (n) in samples is one second at max rate, ie max(rates)
			# FIXME: bank.snr_thresh is removed, use peak_thresh instead
			head = mkitac_spearman(pipeline, snr, max(rates), bank.template_bank_filename, autocorrelation_matrix = bank.autocorrelation_bank, mask_matrix = bank.autocorrelation_mask, snr_thresh = peak_thresh, sigmasq = bank.sigmasq)
			if verbose:
				head = pipeparts.mkprogressreport(pipeline, head, "progress_xml_%s" % suffix)
			triggersrcs[instrument].add(head)

		# FIXME:  find a way to use less memory without this hack
		del bank.autocorrelation_bank
		if nxydump_segment is not None:
			pipeparts.mknxydumpsink(pipeline, pipeparts.mktogglecomplex(pipeline, pipeparts.mkqueue(pipeline, snr)), "%s/snr_gpu_%d_%s.dump" % (nxydump_directory, nxydump_segment[0], suffix), segment = nxydump_segment)
		#pipeparts.mkogmvideosink(pipeline, pipeparts.mkcapsfilter(pipeline, pipeparts.mkchannelgram(pipeline, pipeparts.mkqueue(pipeline, snr), plot_width = .125), "video/x-raw-rgb, width=640, height=480, framerate=64/1"), "snr_channelgram_%s.ogv" % suffix, audiosrc = pipeparts.mkaudioamplify(pipeline, pipeparts.mkqueue(pipeline, hoftdict[max(bank.get_rates())], max_size_time = 2 * int(math.ceil(bank.filter_length)) * gst.SECOND), 0.125), verbose = True)

	#
	# done
	#

	assert any(triggersrcs.values())
	return triggersrcs


def mkSPIIRhoftToSnrSlices(pipeline, src, bank, instrument, verbose = None, nxydump_segment = None, quality = 4, sample_rates = None, max_rate = None, gpu_acc = False):
	if sample_rates is None:
		sample_rates = sorted(bank.get_rates())
	else:
		sample_rates = sorted(sample_rates)
	#FIXME don't upsample everything to a common rate
	if max_rate is None:
		max_rate = max(sample_rates)
	prehead = None

	for sr in sample_rates:
		head = pipeparts.mkqueue(pipeline, src[sr], max_size_time=gst.SECOND * 10, max_size_buffers=0, max_size_bytes=0)
		head = pipeparts.mkreblock(pipeline, head)
		if gpu_acc:
			head = pipemodules.mkcudaiirbank(pipeline, head, a1 = bank.A[sr], b0 = bank.B[sr], delay = bank.D[sr], name = "gstlaliirbank_%d_%s_%s" % (sr, instrument, bank.logname))
		else :
			head = pipeparts.mkiirbank(pipeline, head, a1 = bank.A[sr], b0 = bank.B[sr], delay = bank.D[sr], name = "gstlaliirbank_%d_%s_%s" % (sr, instrument, bank.logname))

		head = pipeparts.mkqueue(pipeline, head, max_size_time=gst.SECOND * 10, max_size_buffers=0, max_size_bytes=0)
		if prehead is not None:
			adder = gst.element_factory_make("lal_adder")
			adder.set_property("sync", True)
			pipeline.add(adder)
			head.link(adder)
			prehead.link(adder)
			head = adder
		#	head = pipeparts.mkadder(pipeline, (head, prehead))
		# FIXME:  this should get a nofakedisconts after it until the resampler is patched
		head = pipeparts.mkresample(pipeline, head, quality = 1)
		if sr == max_rate:
			head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw-float, rate=%d" % max_rate)
		else:
			head = pipeparts.mkcapsfilter(pipeline, head, "audio/x-raw-float, rate=%d" % (2 * sr))
		prehead = head

	return head

def mkBuildBossSPIIR(pipeline, detectors, banks, psd, psd_fft_length = 8, ht_gate_threshold = None, veto_segments = None, verbose = False, nxydump_segment = None, chisq_type = 'autochisq', track_psd = False, block_duration = gst.SECOND, blind_injections = None, peak_thresh = 4):
	#
	# check for recognized value of chisq_type
	#

	if chisq_type not in ['autochisq', 'autochisq_spearman']:
		raise ValueError("chisq_type must be either 'autochisq' or 'autochisq_spearman', given %s" % chisq_type)

	#
	# extract segments from the injection file for selected reconstruction
	#

	if detectors.injection_filename is not None:
		inj_seg_list = simulation.sim_inspiral_to_segment_list(detectors.injection_filename)
	else:
		inj_seg_list = None
		#
		# Check to see if we are specifying blind injections now that we know
		# we don't want real injections. Setting this
		# detectors.injection_filename will ensure that injections are added
		# but won't only reconstruct injection segments.
		#
		detectors.injection_filename = blind_injections

	#
	# construct dictionaries of whitened, conditioned, down-sampled
	# h(t) streams
	#

	hoftdicts = {}
	max_instru_rates = {} 
	sngl_max_rate = 0
	for instrument in detectors.channel_dict:
		for bank_name in banks[instrument]:
			sngl_max_rate = max(spiir_utils.get_maxrate_from_xml(bank_name), sngl_max_rate)
		max_instru_rates[instrument] = sngl_max_rate
		src, statevector, dqvector = datasource.mkbasicsrc(pipeline, detectors, instrument, verbose=verbose)
		if verbose:
		  print "%s: max rate of all banks %d Hz" % (instrument, sngl_max_rate)
		if veto_segments is not None and instrument in veto_segments.keys():
			# fir_whitener set to 0, use FFT whitener
			hoftdicts[instrument] = \
			snglrate_datasource.mkwhitened_src(pipeline, src,
					sngl_max_rate, instrument, psd =
					psd[instrument], psd_fft_length =
					psd_fft_length, ht_gate_threshold =
					ht_gate_threshold, veto_segments =
					veto_segments[instrument], seekevent = detectors.seekevent, nxydump_segment =
					nxydump_segment, track_psd = track_psd,
					zero_pad = 0, width = 32, fir_whitener =
					0, statevector = statevector,
					dqvector = dqvector)
		else:
			hoftdicts[instrument] = \
			snglrate_datasource.mkwhitened_src(pipeline, src,
					sngl_max_rate, instrument, psd =
					psd[instrument], psd_fft_length =
					psd_fft_length, ht_gate_threshold =
					ht_gate_threshold, veto_segments = None, seekevent = detectors.seekevent, 
					nxydump_segment = nxydump_segment,
					track_psd = track_psd, zero_pad = 0,
					width = 32, fir_whitener =
					0, statevector =
					statevector, dqvector = dqvector)

	#
	# construct trigger generators
	#

	triggersrcs = dict((instrument, set()) for instrument in hoftdicts)
	# format of banklist : {'H1': <H1Bank0>, <H1Bank1>..;
	#			'L1': <L1Bank0>, <L1Bank1>..;..}
	# format of bank: <H1bank0>

#	pdb.set_trace()
	for instrument, bank_name in [(instrument, bank_name) for instrument, banklist in banks.items() for bank_name in banklist]:
		bankid = spiir_utils.get_bankid_from_bankname(bank_name)
		suffix = "%s_%d" % (instrument, bankid)
		head = pipeparts.mkqueue(pipeline, hoftdicts[instrument], max_size_time=gst.SECOND * 10, max_size_buffers=0, max_size_bytes=0)
		max_bank_rate = spiir_utils.get_maxrate_from_xml(bank_name)
		if max_bank_rate < max_instru_rates[instrument]:
			head = pipeparts.mkcapsfilter(pipeline, pipeparts.mkresample(pipeline, head, quality = 9), "audio/x-raw-float, rate=%d" % max_bank_rate)
		snr = pipeparts.mkreblock(pipeline, head)

#		bank_struct = build_bank_struct(bank, max_rates[instrument])
		snr = pipemodules.mkcudamultiratespiir(pipeline, snr, bank_name, gap_handle = 0, stream_id = bankid) # treat gap as zeros

		snr = pipeparts.mktogglecomplex(pipeline, snr)
		snr = pipeparts.mktee(pipeline, snr)
		# FIXME you get a different trigger generator depending on the chisq calculation :/
		bank_struct = spiir_utils.Bank()
		bank_struct.read_from_xml(bank_name)
		if chisq_type == 'autochisq':
			# FIXME don't hardcode
			# peak finding window (n) in samples is one second at max rate, ie max(rates)
			head = pipeparts.mkqueue(pipeline, snr, max_size_buffers=1)
			head = pipeparts.mkitac(pipeline, head, max_bank_rate, bank_struct.template_bank_filename, autocorrelation_matrix = bank_struct.autocorrelation_bank, mask_matrix = bank_struct.autocorrelation_mask, snr_thresh = peak_thresh, sigmasq = bank_struct.sigmasq)
			if verbose:
				head = pipeparts.mkprogressreport(pipeline, head, "progress_xml_%s" % suffix)
			triggersrcs[instrument].add(head)
		elif chisq_type == 'autochisq_spearman':
			# FIXME don't hardcode
			# peak finding window (n) in samples is one second at max rate, ie max(rates)
			head = pipeparts.mkqueue(pipeline, snr, max_size_buffers=1)
			head = pipemodules.mkitac_spearman(pipeline, head, max_bank_rate, bank_struct.template_bank_filename, autocorrelation_matrix = bank_struct.autocorrelation_bank, mask_matrix = bank_struct.autocorrelation_mask, snr_thresh = peak_thresh, sigmasq = bank_struct.sigmasq)
			if verbose:
				head = pipeparts.mkprogressreport(pipeline, head, "progress_xml_spearman_%s" % suffix)
			triggersrcs[instrument].add(head)
		# FIXME:  find a way to use less memory without this hack
		del bank_struct.autocorrelation_bank
		if nxydump_segment is not None:
			pipeparts.mknxydumpsink(pipeline, pipeparts.mktogglecomplex(pipeline, pipeparts.mkqueue(pipeline, snr)), "snr_gpu_%d_%s.dump" % (nxydump_segment[0], suffix), segment = nxydump_segment)
		#pipeparts.mkogmvideosink(pipeline, pipeparts.mkcapsfilter(pipeline, pipeparts.mkchannelgram(pipeline, pipeparts.mkqueue(pipeline, snr), plot_width = .125), "video/x-raw-rgb, width=640, height=480, framerate=64/1"), "snr_channelgram_%s.ogv" % suffix, audiosrc = pipeparts.mkaudioamplify(pipeline, pipeparts.mkqueue(pipeline, hoftdict[max(bank.get_rates())], max_size_time = 2 * int(math.ceil(bank.filter_length)) * gst.SECOND), 0.125), verbose = True)

	#
	# done
	#

	assert any(triggersrcs.values())
	return triggersrcs

def mkPostcohSPIIR(pipeline, detectors, banks, psd, psd_fft_length = 8, ht_gate_threshold = None, veto_segments = None, verbose = False, nxydump_segment = None, chisq_type = 'autochisq', track_psd = False, block_duration = gst.SECOND, blind_injections = None, cuda_postcoh_snglsnr_thresh = 4, cuda_postcoh_detrsp_fname = None, cuda_postcoh_hist_trials = 1, output_prefix = None, cuda_postcoh_output_skymap = 0):
#	pdb.set_trace()
	#
	# check for recognized value of chisq_type
	#

	if chisq_type not in ['autochisq']:
		raise ValueError("chisq_type must be either 'autochisq', given %s" % chisq_type)

	#
	# extract segments from the injection file for selected reconstruction
	#

	if detectors.injection_filename is not None:
		inj_seg_list = simulation.sim_inspiral_to_segment_list(detectors.injection_filename)
	else:
		inj_seg_list = None
		#
		# Check to see if we are specifying blind injections now that we know
		# we don't want real injections. Setting this
		# detectors.injection_filename will ensure that injections are added
		# but won't only reconstruct injection segments.
		#
		detectors.injection_filename = blind_injections

	#
	# construct dictionaries of whitened, conditioned, down-sampled
	# h(t) streams

	#
	#

	hoftdicts = {}
	max_instru_rates = {} 
	sngl_max_rate = 0
	for instrument in detectors.channel_dict:
		for instrument_from_bank, bank_list in [(instrument_from_bank, bank_list) for bank_dict in banks for instrument_from_bank, bank_list in bank_dict.items()]:
			if instrument_from_bank == instrument:
				sngl_max_rate = max(spiir_utils.get_maxrate_from_xml(bank_list[0]), sngl_max_rate)
		max_instru_rates[instrument] = sngl_max_rate
		src, statevector, dqvector = datasource.mkbasicsrc(pipeline, detectors, instrument, verbose=verbose)
		if veto_segments is not None:		
			hoftdicts[instrument] = snglrate_datasource.mkwhitened_src(pipeline, src, sngl_max_rate, instrument, psd = psd[instrument], psd_fft_length = psd_fft_length, ht_gate_threshold = ht_gate_threshold, veto_segments = veto_segments[instrument], seekevent = detectors.seekevent, nxydump_segment = nxydump_segment, track_psd = track_psd, zero_pad = 0, width = 32)
		else:
			hoftdicts[instrument] = snglrate_datasource.mkwhitened_src(pipeline, src, sngl_max_rate, instrument, psd = psd[instrument], psd_fft_length = psd_fft_length, ht_gate_threshold = ht_gate_threshold, veto_segments = None, seekevent = detectors.seekevent, nxydump_segment = nxydump_segment, track_psd = track_psd, zero_pad = 0, width = 32)

	#
	# construct trigger generators
	#
	triggersrcs = []

	# format of banks :	[{'H1': <H1Bank0>; 'L1': <L1Bank0>..;}
	#			 {'H1': <H1Bank1>; 'L1': <L1Bank1>..;}
	#			 ...]
	# format of bank_dict: {'H1': <H1Bank1>; 'L1': <L1Bank1>..;}
	autocorrelation_fname_list = []
	for bank_dict in banks:
		autocorrelation_fname = ""
		for instrument, bank_list in bank_dict.items():
			autocorrelation_fname += str(instrument)
			autocorrelation_fname += ":"
			autocorrelation_fname += str(bank_list[0])
			autocorrelation_fname += "," 
			if len(bank_list) != 1:
				raise ValueError("%s instrument: number of banks is not equal to 1, can not do coherent analysis" % instrument)
		autocorrelation_fname = autocorrelation_fname.rstrip(',')
		autocorrelation_fname_list.append(autocorrelation_fname)

	for instrument in banks[0].keys():
		hoftdicts[instrument] = pipeparts.mktee(pipeline, hoftdicts[instrument])

	for i_dict, bank_dict in enumerate(banks):
		postcoh = None
		head = None

		for instrument, bank_list in bank_dict.items():
			bankname = bank_list[0]
			bankid = spiir_utils.get_bankid_from_bankname(bankname)
			max_bank_rate = spiir_utils.get_maxrate_from_xml(bankname)
			head = pipeparts.mkqueue(pipeline, hoftdicts[instrument], max_size_time=gst.SECOND * 10, max_size_buffers=0, max_size_bytes=0)
			if max_bank_rate < max_instru_rates[instrument]:
				head = pipeparts.mkcapsfilter(pipeline, pipeparts.mkresample(pipeline, head, quality = 9), "audio/x-raw-float, rate=%d" % max_bank_rate)
			suffix = "%s_%d" % (instrument, bankid)
	
			head = pipeparts.mkreblock(pipeline, head)
			snr = pipeparts.mkcudamultiratespiir(pipeline, head, bankname, gap_handle = 0, stream_id = bankid) # treat gap as zeros
			if verbose:
				snr = pipeparts.mkprogressreport(pipeline, snr, "progress_done_gpu_filtering_%s" % suffix)

			if postcoh is None:
				postcoh = pipemodules.mkcudapostcoh(pipeline, snr, instrument, cuda_postcoh_detrsp_fname, autocorrelation_fname_list[i_dict], bank_list[0], hist_trials = cuda_postcoh_hist_trials, snglsnr_thresh = cuda_postcoh_snglsnr_thresh, output_skymap = cuda_postcoh_output_skymap, stream_id = bankid)
			else:
				snr.link_pads(None, postcoh, instrument)

		# FIXME: hard-coded to do compression
		if verbose:
			postcoh = pipeparts.mkprogressreport(pipeline, postcoh, "progress_xml_dump_bank_stream%d" % i_dict)
		head = mkpostcohfilesink(pipeline, postcoh, location = output_prefix, compression = 1, snapshot_interval = 0)
		triggersrcs.append(head)
	return triggersrcs

def parse_shift_string(shift_string):
	"""
	parses strings of form 

	det1:shift1, det2:shift2
	
	into a dictionary of lists of shifts.
	"""
	out = {}
	if shift_string is None:
		return out
	for det in shift_string.split(','):
		ifo, shift_val = det.split(':')
		if ifo in out:
			raise ValueError("Only one shift per instrument should be given")
		out[ifo] = shift_val
	return out


def mkPostcohSPIIROnline(pipeline, detectors, banks, psd,
		control_time_shift_string = None, psd_fft_length = 8,
		fir_whitener = 0,
		ht_gate_threshold = None, veto_segments = None, verbose = False,
		nxydump_segment = None, nxydump_directory = '.', chisq_type = 'autochisq', track_psd =
		False, block_duration = gst.SECOND, blind_injections = None,
		cuda_postcoh_snglsnr_thresh = 4.0, cuda_postcoh_cohsnr_thresh =
		5.0, cuda_postcoh_detrsp_fname = None, cuda_postcoh_hist_trials
		= 1, cuda_postcoh_output_skymap = 0, cuda_postcoh_detrsp_refresh_interval = 0, cohfar_file_path = None,
		cohfar_accumbackground_output_prefix = None,
		cohfar_accumbackground_output_name = None,
		cohfar_accumbackground_snapshot_interval = 0,
		cohfar_accumbackground_ifo_sense = 'H1:50,L1:100',
		cohfar_assignfar_refresh_interval = 86400,
		cohfar_assignfar_silent_time = 2147483647,
		cohfar_assignfar_input_fname = None):
	#
	# check for recognized value of chisq_type
	#

	if chisq_type not in ['autochisq']:
		raise ValueError("chisq_type must be either 'autochisq', given %s" % chisq_type)

	#
	# extract segments from the injection file for selected reconstruction
	#

	if detectors.injection_filename is not None:
		inj_seg_list = simulation.sim_inspiral_to_segment_list(detectors.injection_filename)
	else:
		inj_seg_list = None
		#
		# Check to see if we are specifying blind injections now that we know
		# we don't want real injections. Setting this
		# detectors.injection_filename will ensure that injections are added
		# but won't only reconstruct injection segments.
		#
		detectors.injection_filename = blind_injections

	#
	# construct dictionaries of whitened, conditioned, down-sampled
	# h(t) streams

	#
	#

	hoftdicts = {}
	max_instru_rates = {} 
	sngl_max_rate = 0
	for instrument in detectors.channel_dict:
		for instrument_from_bank, bank_list in [(instrument_from_bank, bank_list) for bank_dict in banks for instrument_from_bank, bank_list in bank_dict.items()]:
			if instrument_from_bank == instrument:
				sngl_max_rate = max(spiir_utils.get_maxrate_from_xml(bank_list[0]), sngl_max_rate)
		max_instru_rates[instrument] = sngl_max_rate
		src, statevector, dqvector = datasource.mkbasicsrc(pipeline, detectors, instrument, verbose=verbose)
		if verbose:
		  print "%s: max rate of all banks %d Hz" % (instrument, sngl_max_rate)
		if veto_segments is not None and instrument in veto_segments.keys():
			hoftdicts[instrument] = \
			snglrate_datasource.mkwhitened_src(pipeline, src,
					sngl_max_rate, instrument, psd =
					psd[instrument], psd_fft_length =
					psd_fft_length, ht_gate_threshold =
					ht_gate_threshold, veto_segments =
					veto_segments[instrument], seekevent = detectors.seekevent, nxydump_segment =
					nxydump_segment, nxydump_directory = nxydump_directory, track_psd = track_psd,
					zero_pad = 0, width = 32, fir_whitener =
					fir_whitener, statevector = statevector,
					dqvector = dqvector)
		else:
			hoftdicts[instrument] = \
			snglrate_datasource.mkwhitened_src(pipeline, src,
					sngl_max_rate, instrument, psd =
					psd[instrument], psd_fft_length =
					psd_fft_length, ht_gate_threshold =
					ht_gate_threshold, veto_segments = None, seekevent = detectors.seekevent, 
					nxydump_segment = nxydump_segment, nxydump_directory = nxydump_directory,
					track_psd = track_psd, zero_pad = 0,
					width = 32, fir_whitener =
					fir_whitener, statevector =
					statevector, dqvector = dqvector)

	#
	# construct trigger generators
	#
	triggersrcs = []
	ifos = ""
	for instrument in banks[0].keys():
		ifos += str(instrument)

	# format of banks :	[{'H1': <H1Bank0>; 'L1': <L1Bank0>..;}
	#			 {'H1': <H1Bank1>; 'L1': <L1Bank1>..;}
	#			 ...]
	# format of bank_dict: {'H1': <H1Bank1>; 'L1': <L1Bank1>..;}

	# assemble autocorrelation_fname for postcoh chisq calculation
	autocorrelation_fname_list = []
	for bank_dict in banks:
		autocorrelation_fname = ""
		for instrument, bank_list in bank_dict.items():
			autocorrelation_fname += str(instrument)
			autocorrelation_fname += ":"
			autocorrelation_fname += str(bank_list[0])
			autocorrelation_fname += "," 
			if len(bank_list) != 1:
				raise ValueError("%s instrument: number of banks is not equal to 1, can not do coherent analysis" % instrument)
		autocorrelation_fname = autocorrelation_fname.rstrip(',')
		autocorrelation_fname_list.append(autocorrelation_fname)


	shift_dict = parse_shift_string(control_time_shift_string)

	for instrument in banks[0].keys():
		hoftdicts[instrument] = pipeparts.mktee(pipeline, hoftdicts[instrument])

	for i_dict, bank_dict in enumerate(banks):
		postcoh = None
		head = None

		for instrument, bank_list in bank_dict.items():
			bankname = bank_list[0]
			bankid = spiir_utils.get_bankid_from_bankname(bankname)
			max_bank_rate = spiir_utils.get_maxrate_from_xml(bankname)
			head = pipeparts.mkqueue(pipeline, hoftdicts[instrument], max_size_time=gst.SECOND * 10, max_size_buffers=0, max_size_bytes=0)
			if max_bank_rate < max_instru_rates[instrument]:
				head = pipeparts.mkcapsfilter(pipeline, pipeparts.mkresample(pipeline, head, quality = 9), "audio/x-raw-float, rate=%d" % max_bank_rate)
			suffix = "%s_%d" % (instrument,  bankid)
			if instrument in shift_dict.keys():
				head = mktimeshift(pipeline, head, float(shift_dict[instrument]))
				if verbose:
					head = pipeparts.mkprogressreport(pipeline, head, "after_timeshift_%s" % suffix)

			head = pipeparts.mkreblock(pipeline, head)
			snr = pipemodules.mkcudamultiratespiir(pipeline, head, bank_list[0], gap_handle = 0, stream_id = bankid) # treat gap as zeros
			if verbose:
				snr = pipeparts.mkprogressreport(pipeline, snr, "progress_done_gpu_filtering_%s" % suffix)

			snr = pipeparts.mktee(pipeline, snr)

			if nxydump_segment is not None:
				pipeparts.mknxydumpsink(pipeline, pipeparts.mkqueue(pipeline, snr), "%s/snr_gpu_%d_%s.dump" % (nxydump_directory, nxydump_segment[0], suffix), segment = nxydump_segment)

			snr = pipeparts.mkqueue(pipeline, snr, max_size_time=gst.SECOND * 10, max_size_buffers=10, max_size_bytes=100000000)

			if postcoh is None:
				# make a queue for postcoh, otherwise it will be in the same thread with the first bank	
				postcoh = pipemodules.mkcudapostcoh(pipeline, snr, instrument, cuda_postcoh_detrsp_fname, autocorrelation_fname_list[i_dict], bank_list[0], hist_trials = cuda_postcoh_hist_trials, snglsnr_thresh = cuda_postcoh_snglsnr_thresh, cohsnr_thresh = cuda_postcoh_cohsnr_thresh, output_skymap = cuda_postcoh_output_skymap, detrsp_refresh_interval = cuda_postcoh_detrsp_refresh_interval, stream_id = bankid)
			else:
				snr.link_pads(None, postcoh, instrument)

		# FIXME: hard-coded to do compression
		if verbose:
			postcoh = pipeparts.mkprogressreport(pipeline, postcoh, "progress_xml_dump_bank_stream%d" % i_dict)
		print cohfar_accumbackground_ifo_sense
		if cohfar_accumbackground_output_prefix is None:
			postcoh = pipemodules.mkcohfar_accumbackground(pipeline, postcoh, ifo_sense = cohfar_accumbackground_ifo_sense, hist_trials = cuda_postcoh_hist_trials, output_prefix = None, output_name = cohfar_accumbackground_output_name[i_dict], history_fname = cohfar_assignfar_input_fname.split(',')[0], snapshot_interval = cohfar_accumbackground_snapshot_interval)
		else:
			postcoh = pipemodules.mkcohfar_accumbackground(pipeline, postcoh, ifo_sense = cohfar_accumbackground_ifo_sense, hist_trials = cuda_postcoh_hist_trials, output_prefix = cohfar_accumbackground_output_prefix[i_dict], output_name = None, history_fname = cohfar_assignfar_input_fname.split(',')[0], snapshot_interval = cohfar_accumbackground_snapshot_interval)
		postcoh = pipemodules.mkcohfar_assignfar(pipeline, postcoh, ifo_sense = cohfar_accumbackground_ifo_sense, assignfar_refresh_interval = cohfar_assignfar_refresh_interval, silent_time = cohfar_assignfar_silent_time, input_fname = cohfar_assignfar_input_fname)
		#head = mkpostcohfilesink(pipeline, postcoh, location = output_prefix[i_dict], compression = 1, snapshot_interval = snapshot_interval)
		triggersrcs.append(postcoh)
	return triggersrcs

def mkPostcohSPIIROffline(pipeline, detectors, banks, psd, control_time_shift_string = None, psd_fft_length = 8, ht_gate_threshold = None, veto_segments = None, verbose = False, nxydump_segment = None, chisq_type = 'autochisq', track_psd = False, block_duration = gst.SECOND, blind_injections = None, cuda_postcoh_snglsnr_thresh = 4.0, cuda_postcoh_cohsnr_thresh = 5.0, cuda_postcoh_detrsp_fname = None, cuda_postcoh_hist_trials = 1, cuda_postcoh_output_skymap = 0, cuda_postcohfilesink_output_prefix = None, cuda_postcohfilesink_snapshot_interval = 14400):
#	pdb.set_trace()
	#
	# check for recognized value of chisq_type
	#

	if chisq_type not in ['autochisq']:
		raise valueerror("chisq_type must be either 'autochisq', given %s" % chisq_type)

	#
	# extract segments from the injection file for selected reconstruction
	#

	if detectors.injection_filename is not None:
		inj_seg_list = simulation.sim_inspiral_to_segment_list(detectors.injection_filename)
	else:
		inj_seg_list = None
		#
		# check to see if we are specifying blind injections now that we know
		# we don't want real injections. setting this
		# detectors.injection_filename will ensure that injections are added
		# but won't only reconstruct injection segments.
		#
		detectors.injection_filename = blind_injections

	#
	# construct dictionaries of whitened, conditioned, down-sampled
	# h(t) streams

	#
	#

	hoftdicts = {}
	max_instru_rates = {} 
	sngl_max_rate = 0
	for instrument in detectors.channel_dict:
		for instrument_from_bank, bank_list in [(instrument_from_bank, bank_list) for bank_dict in banks for instrument_from_bank, bank_list in bank_dict.items()]:
			if instrument_from_bank == instrument:
				sngl_max_rate = max(spiir_utils.get_maxrate_from_xml(bank_list[0]), sngl_max_rate)
		max_instru_rates[instrument] = sngl_max_rate
		src, statevector, dqvector  = datasource.mkbasicsrc(pipeline, detectors, instrument, verbose=verbose)
		if veto_segments is not None:		
			hoftdicts[instrument] = \
			snglrate_datasource.mkwhitened_src(pipeline, src,
					sngl_max_rate, instrument, psd =
					psd[instrument], psd_fft_length =
					psd_fft_length, ht_gate_threshold =
					ht_gate_threshold, veto_segments =
					veto_segments[instrument], seekevent = detectors.seekevent, nxydump_segment =
					nxydump_segment, track_psd = track_psd,
					zero_pad = 0, width = 32, fir_whitener = 0, statevector = statevector, dqvector = dqvector)
		else:
			hoftdicts[instrument] = \
			snglrate_datasource.mkwhitened_src(pipeline, src,
					sngl_max_rate, instrument, psd =
					psd[instrument], psd_fft_length =
					psd_fft_length, ht_gate_threshold =
					ht_gate_threshold, veto_segments = None, seekevent = detectors.seekevent, 
					nxydump_segment = nxydump_segment,
					track_psd = track_psd, zero_pad = 0,
					width = 32, fir_whitener = 0, statevector = statevector, dqvector = dqvector)

	#
	# construct trigger generators
	#
	triggersrcs = []
	ifos = ""
	for instrument in banks[0].keys():
		ifos += str(instrument)

	# format of banks :	[{'h1': <h1bank0>; 'l1': <l1bank0>..;}
	#			 {'h1': <h1bank1>; 'l1': <l1bank1>..;}
	#			 ...]
	# format of bank_dict: {'h1': <h1bank1>; 'l1': <l1bank1>..;}

	autocorrelation_fname_list = []
	for bank_dict in banks:
		autocorrelation_fname = ""
		for instrument, bank_list in bank_dict.items():
			autocorrelation_fname += str(instrument)
			autocorrelation_fname += ":"
			autocorrelation_fname += str(bank_list[0])
			autocorrelation_fname += "," 
			if len(bank_list) != 1:
				raise valueerror("%s instrument: number of banks is not equal to other banks, can not do coherent analysis" % instrument)
		autocorrelation_fname = autocorrelation_fname.rstrip(',')
		autocorrelation_fname_list.append(autocorrelation_fname)

	for instrument in banks[0].keys():
		hoftdicts[instrument] = pipeparts.mktee(pipeline, hoftdicts[instrument])


	shift_dict = parse_shift_string(control_time_shift_string)


	for i_dict, bank_dict in enumerate(banks):
		postcoh = None
		head = None

		for instrument, bank_list in bank_dict.items():
			bankname = bank_list[0]
			bankid = spiir_utils.get_bankid_from_bankname(bankname)
			max_bank_rate = spiir_utils.get_maxrate_from_xml(bankname)
			head = pipeparts.mkqueue(pipeline, hoftdicts[instrument], max_size_time=gst.SECOND * 10, max_size_buffers=0, max_size_bytes=0)
			if max_bank_rate < max_instru_rates[instrument]:
				head = pipeparts.mkcapsfilter(pipeline, pipeparts.mkresample(pipeline, head, quality = 9), "audio/x-raw-float, rate=%d" % max_bank_rate)
			suffix = "%s_%d" % (instrument,  bankid)

			if instrument in shift_dict.keys():
				head = mktimeshift(pipeline, head, float(shift_dict[instrument]))


			head = pipeparts.mkreblock(pipeline, head)
			snr = pipemodules.mkcudamultiratespiir(pipeline, head, bank_list[0], gap_handle = 0, stream_id = bankid) # treat gap as zeros
			if verbose:
				snr = pipeparts.mkprogressreport(pipeline, snr, "progress_done_gpu_filtering_%s" % suffix)

			if postcoh is None:
				# make a queue for postcoh, otherwise it will be in the same thread with the first bank	
				snr = pipeparts.mkqueue(pipeline, snr, max_size_time=gst.SECOND * 10, max_size_buffers=0, max_size_bytes=0)
				postcoh = pipemodules.mkcudapostcoh(pipeline, snr, instrument, cuda_postcoh_detrsp_fname, autocorrelation_fname_list[i_dict], bank_list[0], hist_trials = cuda_postcoh_hist_trials, snglsnr_thresh = cuda_postcoh_snglsnr_thresh, cohsnr_thresh = cuda_postcoh_cohsnr_thresh, output_skymap = cuda_postcoh_output_skymap, stream_id = bankid)
			else:
				snr.link_pads(None, postcoh, instrument)

		# FIXME: hard-coded to do compression
		if verbose:
			postcoh = pipeparts.mkprogressreport(pipeline, postcoh, "progress_xml_dump_bank_stream%d" % i_dict)

		head = pipemodules.mkpostcohfilesink(pipeline, postcoh, location = cuda_postcohfilesink_output_prefix[i_dict], compression = 1, snapshot_interval = cuda_postcohfilesink_snapshot_interval)
		triggersrcs.append(head)
	return triggersrcs

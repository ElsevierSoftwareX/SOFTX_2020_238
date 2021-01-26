# Copyright (C) 2020  Patrick Godwin (patrick.godwin@ligo.org)
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


import os

from gstlal import plugins
from gstlal.config import Argument, Option
from gstlal.datafind import DataType, DataCache
from gstlal.dags.layers import Layer, Node
from gstlal.dags import util as dagutil


def reference_psd_layer(config, dag):
	requirements = {"request_cpu": 2, "request_memory": 2000, **config.condor}
	layer = Layer("gstlal_reference_psd", requirements=requirements)

	psd_cache = DataCache.generate(DataType.REFERENCE_PSD, config.ifo_combo, config.time_bins)

	for span, psds in psd_cache.groupby("time").items():
		start, end = span
		layer += Node(
			key = span,
			arguments = [
				Option("gps-start-time", int(start)),
				Option("gps-end-time", int(end)),
				Option("data-source", "frames"),
				Option("channel-name", format_ifo_args(config.ifos, config.source.channel_name)),
				Option("frame-type", format_ifo_args(config.ifos, config.source.frame_type)),
				Option("frame-segments-name", config.source.frame_segments_name),
				Option("data-find-server", config.source.data_find_server),
				Option("psd-fft-length", config.psd.fft_length),
			],
			inputs = Option("frame-segments-file", config.source.frame_segments_file),
			outputs = Option("write-psd", psds.files)
		)

	dag["reference_psd"] = layer
	return psd_cache


def median_psd_layer(config, dag, ref_psd_cache):
	requirements = {"request_cpu": 2, "request_memory": 2000, **config.condor}
	layer = Layer("gstlal_median_of_psds", parents="reference_psd", requirements=requirements)

	median_psd_cache = DataCache.generate(DataType.REFERENCE_PSD, config.ifo_combo, config.span)

	layer += Node(
		inputs = Argument("psds", ref_psd_cache.files),
		outputs = Option("output-name", median_psd_cache.files)
	)

	dag["median_psd"] = layer
	return median_psd_cache


def svd_bank_layer(config, dag, median_psd_cache):
	requirements = {"request_cpu": 1, "request_memory": 4000, **config.condor}
	layer = Layer("gstlal_inspiral_svd_bank", parents="median_psd", requirements=requirements)

	svd_cache = DataCache.generate(DataType.SVD_BANK, config.ifos, config.span, svd_bins=config.svd.bins)
	split_banks = DataCache.find(DataType.SPLIT_BANK, root=config.rootdir).groupby("bin")

	for (ifo, svd_bin), svd_banks in svd_cache.groupby("ifo", "bin").items():
		layer += Node(
			key = (ifo, svd_bin),
			arguments = [
				Option("instrument-override", ifo),
				Option("flow", config.svd.f_low),
				Option("sample-rate", config.svd.sample_rate),
				Option("samples-min", config.svd.samples_min),
				Option("samples-max-64", config.svd.samples_max_64),
				Option("samples-max-256", config.svd.samples_max_256),
				Option("samples-max", config.svd.samples_max),
				Option("svd-tolerance", config.svd.tolerance),
				Option("autocorrelation-length", config.svd.autocorrelation_length),
			],
			inputs = [
				Option("reference-psd", median_psd_cache.files),
				Argument("split-banks", split_banks[svd_bin].files),
			],
			outputs = Option("write-svd", svd_banks.files)
		)

	dag["svd_bank"] = layer
	return svd_cache


def filter_layer(config, dag, ref_psd_cache, svd_bank_cache):
	requirements = {"request_cpu": 2, "request_memory": 4000, **config.condor}
	layer = Layer("gstlal_inspiral", parents=("reference_psd", "svd_bank"), requirements=requirements)

	trigger_cache = DataCache.generate(DataType.TRIGGERS, config.ifo_combo, config.time_bins, svd_bins=config.svd.bins)
	dist_stat_cache = DataCache.generate(DataType.DIST_STATS, config.ifo_combo, config.time_bins, svd_bins=config.svd.bins)

	common_opts = [
		Option("track-psd"),
		Option("data-source", "frames"),
		Option("control-peak-time", 0),
		Option("psd-fft-length", config.psd.fft_length),
		Option("channel-name", format_ifo_args(config.ifos, config.source.channel_name)),
		Option("frame-type", format_ifo_args(config.ifos, config.source.frame_type)),
		Option("data-find-server", config.source.data_find_server),
		Option("frame-segments-name", config.source.frame_segments_name),
		Option("tmp-space", dagutil.condor_scratch_space()),
		Option("coincidence-threshold", config.filter.coincidence_threshold),
		Option("fir-stride", config.filter.fir_stride),
		Option("min-instruments", config.filter.min_instruments),
	]

	# disable service discovery if using singularity
	if config.condor.singularity_image:
		common_opts.append(Option("disable-service-discovery"))

	ref_psds = ref_psd_cache.groupby("time")
	svd_banks = svd_bank_cache.groupby("bin")
	dist_stats = dist_stat_cache.groupby("time", "bin")
	for (span, svd_bin), triggers in trigger_cache.groupby("time", "bin").items():
		start, end = span

		filter_opts = [
			Option("ht-gate-threshold", calc_gate_threshold(config, svd_bin)),
			Option("gps-start-time", int(start)),
			Option("gps-end-time", int(end)),
		]
		filter_opts.extend(common_opts)

		layer += Node(
			key = (span, svd_bin),
			parent_keys = {
				"reference_psd": [span],
				"svd_bank": [(ifo, svd_bin) for ifo in config.ifos],
			},
			arguments = filter_opts,
			inputs = [
				Option("frame-segments-file", config.source.frame_segments_file),
				Option("veto-segments-file", config.filter.veto_segments_file),
				Option("reference-psd", ref_psds[span].files),
				Option("time-slide-file", config.filter.time_slide_file),
				Option("svd-bank", svd_banks[svd_bin].files),
			],
			outputs = [
				Option("output", triggers.files),
				Option("ranking-stat-output", dist_stats[(span, svd_bin)].files),
			],
		)

	dag["filter"] = layer
	return trigger_cache, dist_stat_cache


def aggregate_layer(config, dag, time_bins):
	layer = Layer("gstlal_inspiral_aggregate", requirements=config.condor)

	return layer


def calc_gate_threshold(config, svd_bin, aggregate="max"):
	if ":" in config.filter.ht_gate_threshold:
		bank_mchirp = config.svd.stats[svd_bin][f"{aggregate}_mchirp"]
		min_mchirp, min_threshold, max_mchirp, max_threshold = [
			float(y) for x in config.filter.ht_gate_threshold.split("-") for y in x.split(":")
		]
		gate_mchirp_ratio = (max_threshold - min_threshold) / (max_mchirp - min_mchirp)
		return gate_mchirp_ratio * (bank_mchirp - min_mchirp) + min_threshold
	else: # uniform threshold
		return config.filter.ht_gate_threshold


def format_ifo_args(ifos, args):
	if isinstance(ifos, str):
		ifos = [ifos]
	return [f"{ifo}={args[ifo]}" for ifo in ifos]


@plugins.register
def layers():
	return {
		"reference_psd": reference_psd_layer,
		"median_psd": median_psd_layer,
		"svd_bank": svd_bank_layer,
		"filter": filter_layer,
		"aggregate": aggregate_layer,
	}

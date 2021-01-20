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


import glob
import os

from gstlal import plugins
from gstlal.config import Argument, Option
from gstlal.dags.layers import Layer, Node
from gstlal.dags import util as dagutils


def reference_psd_layer(config, dag, time_bins):
	layer = Layer("gstlal_reference_psd", requirements=config.condor, base_layer=True)

	for span in time_bins:
		start, end = span
		psd_path = data_path("psd", start)
		psd_file = dagutils.T050017_filename(config.ifos, "REFERENCE_PSD", span, '.xml.gz')

		layer += Node(
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
			inputs = [
				Option("frame-segments-file", config.source.frame_segments_file)
			],
			outputs = [
				Option("write-psd", os.path.join(psd_path, psd_file))
			],
		)

	return layer


def median_psd_layer(config, dag):
	layer = Layer("gstlal_median_of_psds", requirements=config.condor)

	median_path = data_path("median_psd", config.start)
	median_file = dagutils.T050017_filename(config.ifos, "REFERENCE_PSD", config.span, '.xml.gz')

	layer += Node(
		inputs = [Argument("psds", dag["reference_psd"].outputs["write-psd"])],
		outputs = [Option("output-name", os.path.join(median_path, median_file))]
	)

	return layer


def svd_bank_layer(config, dag, svd_bins):
	layer = Layer("gstlal_inspiral_svd_bank", requirements=config.condor)

	for svd_bin in svd_bins:
		for ifo in config.ifos:
			svd_path = data_path("svd_bank", config.start)
			svd_file = dagutils.T050017_filename(ifo, f"{svd_bin}_SVD", config.span, '.xml.gz')
			split_banks = glob.glob(os.path.join(config.rootdir, "split_bank", svd_bin, "*.xml.gz"))

			layer += Node(
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
					Option("reference-psd", dag["median_psd"].outputs["output-name"]),
					Argument("split-banks", split_banks),
				],
				outputs = [Option("write-svd", os.path.join(svd_path, svd_file))],
			)

	return layer


def filter_layer(config, dag, time_bins, svd_bins):
	layer = Layer("gstlal_inspiral", requirements=config.condor)

	common_opts = [
		Option("track-psd"),
		Option("data-source", "frames"),
		Option("control-peak-time", 0),
		Option("psd-fft-length", config.psd.fft_length),
		Option("channel-name", format_ifo_args(config.ifos, config.source.channel_name)),
		Option("frame-type", format_ifo_args(config.ifos, config.source.frame_type)),
		Option("data-find-server", config.source.data_find_server),
		Option("frame-segments-name", config.source.frame_segments_name),
		Option("tmp-space", dagutils.condor_scratch_space()),
		Option("coincidence-threshold", config.filter.coincidence_threshold),
		Option("fir-stride", config.filter.fir_stride),
		Option("min-instruments", config.filter.min_instruments),
	]

	# disable service discovery if using singularity
	if config.condor.singularity_image:
		common_opts.append(Option("disable-service-discovery"))

	for time_idx, span in enumerate(time_bins):
		start, end = span
		for svd_idx, svd_bin in enumerate(svd_bins):
			filter_opts = [
				Option("ht-gate-threshold", calc_gate_threshold(config, svd_bin)),
				Option("gps-start-time", int(start)),
				Option("gps-end-time", int(end)),
			]
			filter_opts.extend(common_opts)

			# filenames
			trigger_path = data_path("triggers", start)
			dist_stat_path = data_path("dist_stats", start)
			trigger_file = dagutils.T050017_filename(config.ifos, f"{svd_bin}_LLOID", span, '.xml.gz')
			dist_stat_file = dagutils.T050017_filename(config.ifos, f"{svd_bin}_DIST_STATS", span, '.xml.gz')

			# select relevant svd banks from previous layer
			num_ifos = len(config.ifos)
			start_svd_idx = num_ifos * svd_idx
			svd_banks = dag["svd_bank"].outputs["write-svd"][start_svd_idx:(start_svd_idx+num_ifos)]
			svd_bank_files = ",".join([f"{ifo}:{bank}" for ifo, bank in zip(config.ifos, svd_banks)])

			layer += Node(
				arguments = filter_opts,
				inputs = [
					Option("frame-segments-file", config.source.frame_segments_file),
					Option("veto-segments-file", config.filter.veto_segments_file),
					Option("reference-psd", dag["reference_psd"].outputs["write-psd"][time_idx]),
					Option("time-slide-file", config.filter.time_slide_file),
					Option("svd-bank", svd_bank_files),
				],
				outputs = [
					Option("output", trigger_file),
					Option("ranking-stat-output", dist_stat_file),
				],
			)

	return layer


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

def data_path(data_name, start, create=True):
	path = os.path.join(data_name, dagutils.gps_directory(start))
	os.makedirs(path, exist_ok=True)
	return path


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

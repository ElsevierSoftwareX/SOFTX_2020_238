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
	requirements = {"request_cpus": 2, "request_memory": 2000, **config.condor.submit}
	layer = Layer("gstlal_reference_psd", requirements=requirements)

	psd_cache = DataCache.generate(DataType.REFERENCE_PSD, config.ifo_combo, config.time_bins)
	psd_cache.layer = "reference_psd"

	for span, psds in psd_cache.groupby("time").items():
		start, end = span
		layer += Node(
			key = span,
			arguments = [
				Option("gps-start-time", int(start)),
				Option("gps-end-time", int(end)),
				Option("data-source", "frames"),
				Option("channel-name", dagutil.format_ifo_args(config.ifos, config.source.channel_name)),
				Option("frame-type", dagutil.format_ifo_args(config.ifos, config.source.frame_type)),
				Option("frame-segments-name", config.source.frame_segments_name),
				Option("data-find-server", config.source.data_find_server),
				Option("psd-fft-length", config.psd.fft_length),
			],
			inputs = Option("frame-segments-file", config.source.frame_segments_file),
			outputs = Option("write-psd", psds.files)
		)

	dag["reference_psd"] = layer
	return psd_cache


def median_psd_layer(config, dag, psd_cache):
	requirements = {"request_cpus": 2, "request_memory": 2000, **config.condor.submit}
	layer = Layer("gstlal_median_of_psds", parents=psd_cache.layer, requirements=requirements)

	median_psd_cache = DataCache.generate(DataType.REFERENCE_PSD, config.ifo_combo, config.span)
	median_psd_cache.layer = "median_psd"

	layer += Node(
		inputs = Argument("psds", psd_cache.files),
		outputs = Option("output-name", median_psd_cache.files)
	)

	dag["median_psd"] = layer
	return median_psd_cache


def smoothen_psd_layer(config, dag, psd_cache):
	requirements = {"request_cpus": 1, "request_memory": 2000, **config.condor.submit}
	layer = Layer("gstlal_psd_polyfit", parents=psd_cache.layer, requirements=requirements)

	smooth_psd_cache = DataCache.generate(DataType.SMOOTH_PSD, config.ifo_combo, config.time_bins)
	smooth_psd_cache.layer = "smoothen_psd"

	smooth_psds = smooth_psd_cache.groupby("time")
	for span, psds in psd_cache.groupby("time").items():
		layer += Node(
			key = span,
			parent_keys = {psd_cache.layer: span},
			arguments = Option("low-fit-freq", 10),
			inputs = Argument("psds", psd_cache.files),
			outputs = Option("output-name", smooth_psds[span].files)
		)

	dag["smoothen_psd"] = layer
	return smooth_psd_cache


@plugins.register
def layers():
	return {
		"reference_psd": reference_psd_layer,
		"median_psd": median_psd_layer,
		"smoothen_psd": smoothen_psd_layer,
	}

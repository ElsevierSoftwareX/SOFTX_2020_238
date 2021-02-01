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
	requirements = {"request_cpus": 2, "request_memory": 2000, **config.condor}
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
	requirements = {"request_cpus": 2, "request_memory": 2000, **config.condor}
	layer = Layer("gstlal_median_of_psds", parents="reference_psd", requirements=requirements)

	median_psd_cache = DataCache.generate(DataType.REFERENCE_PSD, config.ifo_combo, config.span)

	layer += Node(
		inputs = Argument("psds", ref_psd_cache.files),
		outputs = Option("output-name", median_psd_cache.files)
	)

	dag["median_psd"] = layer
	return median_psd_cache


def svd_bank_layer(config, dag, median_psd_cache):
	requirements = {"request_cpus": 1, "request_memory": 4000, **config.condor}
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
	requirements = {"request_cpus": 2, "request_memory": 4000, **config.condor}
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


def aggregate_layer(config, dag, trigger_cache, dist_stat_cache):
	# cluster triggers by SNR
	trg_layer = Layer(
		"lalapps_run_sqlite",
		name="cluster_triggers_by_snr",
		parents="filter",
		requirements={"request_cpus": 1, "request_memory": 2000, **config.condor}
	)

	# FIXME: find better way of discovering SQL file
	share_path = os.path.split(dagutil.which("gstlal_inspiral"))[0].replace("bin", "share/gstlal")
	snr_cluster_sql_file = os.path.join(share_path, "snr_simplify_and_cluster.sql")

	for svd_bin, triggers in trigger_cache.groupby("bin").items():
		trg_layer += Node(
			key = svd_bin,
			parent_keys = {"filter": [(span, svd_bin) for span in config.time_bins]},
			arguments = [
				Option("sql-file", snr_cluster_sql_file),
				Option("tmp-space", dagutil.condor_scratch_space()),
			],
			inputs = Argument("triggers", triggers.files),
		)

	# marginalize dist stats across time
	dist_layer = Layer(
		"gstlal_inspiral_marginalize_likelihood",
		name="marginalize_dist_stats_across_time_filter",
		parents="filter",
		requirements={"request_cpus": 1, "request_memory": 2000, **config.condor}
	)

	agg_dist_stat_cache = DataCache.generate(DataType.DIST_STATS, config.ifo_combo, config.span, svd_bins=config.svd.bins)

	dist_stats = dist_stat_cache.groupby("bin")
	for svd_bin, agg_dist_stats in agg_dist_stat_cache.groupby("bin").items():
		dist_layer += Node(
			key = svd_bin,
			parent_keys = {"filter": [(span, svd_bin) for span in config.time_bins]},
			arguments = Option("marginalize", "ranking-stat"),
			inputs = Argument("dist-stats", dist_stats[svd_bin].files),
			outputs = Option("output", agg_dist_stats.files)
		)

	dag["aggregate_triggers"] = trg_layer
	dag["aggregate_dist_stats"] = dist_layer

	return trigger_cache, agg_dist_stat_cache


def prior_layer(config, dag, median_psd_cache, dist_stat_cache):
	if "aggregate_dist_stats" in dag:
		parents = ("median_psd", "aggregate_dist_stats")
	else:
		parents = None

	layer = Layer(
		"gstlal_inspiral_create_prior_diststats",
		parents=parents,
		requirements={"request_cpus": 2, "request_memory": 4000, **config.condor}
	)

	prior_cache = DataCache.generate(DataType.DIST_STATS, config.ifo_combo, config.span, svd_bins=config.svd.bins)

	for svd_bin, prior in prior_cache.groupby("bin").items():
		prior_inputs = [
			Option("svd-file", config.svd.manifest),
			Option("mass-model-file", config.prior.mass_model),
			Option("psd-xml", median_psd_cache.files)
		]
		if config.prior.idq_timeseries:
			prior_inputs["idq-file"] = config.prior.idq_timeseries

		layer += Node(
			key = svd_bin,
			parent_keys = {"aggregate_dist_stats": [svd_bin]},
			arguments = [
				Option("df", "bandwidth"),
				Option("svd-bin", svd_bin),
				Option("background-prior", 1),
				Option("instrument", config.ifos),
				Option("min-instruments", config.filter.min_instruments),
				Option("coincidence-threshold", config.filter.coincidence_threshold),
			],
			inputs = prior_inputs,
			outputs = Option("write-likelihood", prior.files),
		)

	dag["prior"] = layer
	return prior_cache


def marginalize_layer(config, dag, prior_cache, dist_stat_cache):
	if "aggregate_dist_stats" in dag:
		parents = ("prior", "aggregate_dist_stats")
	else:
		parents = "prior"

	layer = Layer(
		"gstlal_inspiral_marginalize_likelihood",
		name="marginalize_dist_stats_across_time_rank",
		parents=parents,
		requirements={"request_cpus": 1, "request_memory": 2000, **config.condor}
	)

	marg_dist_stat_cache = DataCache.generate(DataType.DIST_STATS, config.ifo_combo, config.span, svd_bins=config.svd.bins)

	prior = prior_cache.groupby("bin")
	dist_stats = dist_stat_cache.groupby("bin")
	for svd_bin, marg_dist_stats in marg_dist_stat_cache.groupby("bin").items():
		parent_keys = {"prior": [svd_bin]}
		if "aggregate_dist_stats" in dag:
			parent_keys["aggregate_dist_stats"] = [svd_bin]

		layer += Node(
			key = svd_bin,
			parent_keys = parent_keys,
			arguments = Option("marginalize", "ranking-stat"),
			inputs = Argument("dist-stats", dist_stats[svd_bin].files + prior[svd_bin].files),
			outputs = Option("output", marg_dist_stats.files)
		)

	dag["marginalize"] = layer
	return marg_dist_stat_cache


def calc_pdf_layer(config, dag, dist_stat_cache):
	layer = Layer(
		"gstlal_inspiral_calc_rank_pdfs",
		parents="marginalize",
		requirements={"request_cpus": 1, "request_memory": 2000, **config.condor}
	)

	pdf_cache = DataCache.generate(DataType.DIST_STAT_PDFS, config.ifo_combo, config.span, svd_bins=config.svd.bins)

	dist_stats = dist_stat_cache.groupby("bin")
	for svd_bin, pdfs in pdf_cache.groupby("bin").items():
		layer += Node(
			key = svd_bin,
			parent_keys = {"marginalize": [svd_bin]},
			arguments = Option("ranking-stat-samples", config.rank.ranking_stat_samples),
			inputs = Argument("dist-stats", dist_stats[svd_bin].files),
			outputs = Option("output", pdfs.files)
		)

	dag["calc_pdf"] = layer
	return pdf_cache


def marginalize_pdf_layer(config, dag, pdf_cache):
	layer = Layer(
		"gstlal_inspiral_marginalize_likelihood",
		name="gstlal_inspiral_marginalize_pdfs",
		parents="calc_pdf",
		requirements={"request_cpus": 1, "request_memory": 2000, **config.condor}
	)

	marg_pdf_cache = DataCache.generate(DataType.DIST_STAT_PDFS, config.ifo_combo, config.span)

	layer += Node(
		parent_keys = {"calc_pdf": [svd_bin for svd_bin in config.svd.bins]},
		arguments = Option("marginalize", "ranking-stat-pdf"),
		inputs = Argument("dist-stat-pdfs", pdf_cache.files),
		outputs = Option("output", marg_pdf_cache.files)
	)

	dag["marginalize_pdf"] = layer
	return marg_pdf_cache


def calc_likelihood_layer(config, dag, trigger_cache, dist_stat_cache):
	if "aggregate_dist_stats" in dag:
		parents = ("marginalize", "aggregate_triggers", "aggregate_dist_stats")
	else:
		parents = "marginalize"

	layer = Layer(
		"gstlal_inspiral_calc_likelihood",
		parents=parents,
		requirements={"request_cpus": 1, "request_memory": 2000, **config.condor}
	)

	dist_stats = dist_stat_cache.groupby("bin")
	for svd_bin, triggers in trigger_cache.groupby("bin").items():
		layer += Node(
			parent_keys = {
				"marginalize": [svd_bin],
				"aggregate_triggers": [svd_bin],
				"aggregate_dist_stats": [svd_bin],
			},
			arguments = [
				Option("force"),
				Option("tmp-space", dagutil.condor_scratch_space()),
			],
			inputs = [
				Option("likelihood-url", dist_stats[svd_bin].files),
				Argument("triggers", triggers.files),
			],
			outputs = Argument("calc-triggers", triggers.files, include=False),
		)

	dag["calc_likelihood"] = layer
	return trigger_cache


def cluster_layer(config, dag, trigger_cache):
	# cluster triggers by likelihood
	layer = Layer(
		"lalapps_run_sqlite",
		name="cluster_triggers_by_likelihood",
		parents="calc_likelihood",
		requirements={"request_cpus": 1, "request_memory": 2000, **config.condor}
	)

	# FIXME: find better way of discovering SQL file
	share_path = os.path.split(dagutil.which("gstlal_inspiral"))[0].replace("bin", "share/gstlal")
	cluster_sql_file = os.path.join(share_path, "simplify_and_cluster.sql")

	for span, triggers in trigger_cache.groupby("time").items():
		layer += Node(
			key = span,
			arguments = [
				Option("sql-file", cluster_sql_file),
				Option("tmp-space", dagutil.condor_scratch_space()),
			],
			inputs = Argument("triggers", triggers.files),
		)

	dag["cluster"] = layer
	return trigger_cache


def compute_far_layer(config, dag, trigger_cache, pdf_cache):
	layer = Layer(
		"gstlal_compute_far_from_snr_chisq_histograms",
		name="compute_far",
		parents=("cluster", "marginalize_pdf"),
		requirements={"request_cpus": 1, "request_memory": 2000, **config.condor}
	)

	for span, triggers in trigger_cache.groupby("time").items():
		layer += Node(
			key = span,
			parent_keys = {"cluster": [span]},
			arguments = [
				Option("tmp-space", dagutil.condor_scratch_space()),
			],
			inputs = [
				Option("non-injection-db", triggers.files),
				Option("background-bins-file", pdf_cache.files),
			],
		)

	dag["compute_far"] = layer
	return trigger_cache


def calc_gate_threshold(config, svd_bin, aggregate="max"):
	"""
	Given a configuration, svd bin and aggregate, this calculates
	the h(t) gate threshold used for a given svd bin.
	"""
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
	"""
	Given a set of instruments and arguments keyed by instruments, this
	creates a list of strings in the form {ifo}={arg}. This is suitable
	for command line options like --channel-name which expects this
	particular format.
	"""
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
		"prior": prior_layer,
		"calc_pdf": calc_pdf_layer,
		"marginalize": marginalize_layer,
		"marginalize_pdf": marginalize_pdf_layer,
		"calc_likelihood": calc_likelihood_layer,
		"cluster": cluster_layer,
		"compute_far": compute_far_layer,
	}

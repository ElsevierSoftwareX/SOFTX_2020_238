# Copyright (C) 2013--2014  Kipp Cannon, Chad Hanna
# Copyright (C) 2019        Patrick Godwin
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

##
# @file
#
# A file that contains the inspiral_pipe module code; used to construct condor dags
#

##
# @package inspiral_pipe
#
# A module that contains the inspiral_pipe module code; used to construct condor dags
#
# ### Review Status
#
# | Names                                          | Hash                                        | Date       | Diff to Head of Master      |
# | -------------------------------------------    | ------------------------------------------- | ---------- | --------------------------- |
# | Florent, Sathya, Duncan Me, Jolien, Kipp, Chad | 8a6ea41398be79c00bdc27456ddeb1b590b0f68e    | 2014-06-18 | <a href="@gstlal_inspiral_cgit_diff/python/inspiral_pipe.py?id=HEAD&id2=8a6ea41398be79c00bdc27456ddeb1b590b0f68e">inspiral_pipe.py</a> |
#
# #### Actions
#
# - In inspiral_pipe.py Fix the InsiralJob.___init___: fix the arguments
# - On line 201, fix the comment or explain what the comment is meant to be

#
# imports
#

from collections import defaultdict
import copy
import doctest
import functools
import itertools
import os
import socket
import stat

import lal.series
from lal.utils import CacheEntry

from ligo import segments
from ligo.lw import lsctables, ligolw
from ligo.lw import utils as ligolw_utils

from gstlal import dagparts
from gstlal import datasource
from gstlal import inspiral
from gstlal import svd_bank


#
# LIGOLW initialization
#


class LIGOLWContentHandler(ligolw.LIGOLWContentHandler):
	pass
lsctables.use_in(LIGOLWContentHandler)


#
# DAG layers
#


def online_inspiral_layer(dag, jobs, options):
	job_tags = []
	inj_job_tags = []

	if options.ht_gate_threshold_linear is not None:
		template_mchirp_dict = get_svd_bank_params_online(options.bank_cache.values()[0])
	else: # saves cost of reading in svd banks
		template_mchirp_dict = None

	bank_groups = list(build_bank_groups(options.bank_cache, [1], options.max_jobs - 1))
	if len(options.likelihood_files) != len(bank_groups):
		raise ValueError("Likelihood files must correspond 1:1 with bank files")

	for num_insp_nodes, (svd_banks, likefile, zerolikefile) in enumerate(zip(bank_groups, options.likelihood_files, options.zerolag_likelihood_files)):
		svd_bank_string = ",".join([":".join([k, v[0]]) for k,v in svd_banks.items()])
		job_tags.append("%04d" % num_insp_nodes)

		# Calculate the appropriate ht-gate-threshold value
		threshold_values = get_threshold_values(template_mchirp_dict, [job_tags[-1]], [svd_bank_string], options)

		# Data source dag options
		if (options.data_source == "framexmit"):
			datasource_opts = {
				"framexmit-addr": datasource.framexmit_list_from_framexmit_dict(options.framexmit_dict),
				"framexmit-iface": options.framexmit_iface
			}
		else:
			datasource_opts = {
				"shared-memory-partition": datasource.pipeline_channel_list_from_channel_dict(options.shm_part_dict, opt = "shared-memory-partition"),
				"shared-memory-block-size": options.shared_memory_block_size,
				"shared-memory-assumed-duration": options.shared_memory_assumed_duration
			}

		common_opts = {
			"psd-fft-length": options.psd_fft_length,
			"reference-psd": options.reference_psd,
			"ht-gate-threshold": threshold_values,
			"channel-name": datasource.pipeline_channel_list_from_channel_dict(options.channel_dict),
			"state-channel-name": datasource.pipeline_channel_list_from_channel_dict(options.state_channel_dict, opt = "state-channel-name"),
			"dq-channel-name": datasource.pipeline_channel_list_from_channel_dict(options.dq_channel_dict, opt = "dq-channel-name"),
			"state-vector-on-bits": options.state_vector_on_bits,
			"state-vector-off-bits": options.state_vector_off_bits,
			"dq-vector-on-bits": options.dq_vector_on_bits,
			"dq-vector-off-bits": options.dq_vector_off_bits,
			"svd-bank": svd_bank_string,
			"tmp-space": dagparts.condor_scratch_space(),
			"track-psd": "",
			"control-peak-time": options.control_peak_time,
			"coincidence-threshold": options.coincidence_threshold,
			"fir-stride": options.fir_stride,
			"data-source": options.data_source,
			"gracedb-far-threshold": options.gracedb_far_threshold,
			"delay-uploads": options.delay_uploads,
			"gracedb-group": options.gracedb_group,
			"gracedb-pipeline": options.gracedb_pipeline,
			"gracedb-search": options.gracedb_search,
			"gracedb-label": options.gracedb_label,
			"gracedb-service-url": options.gracedb_service_url,
			"job-tag": job_tags[-1],
			"likelihood-snapshot-interval": options.likelihood_snapshot_interval,
			"far-trials-factor": options.far_trials_factor,
			"min-instruments": options.min_instruments,
			"time-slide-file": options.time_slide_file,
			"output-kafka-server": options.output_kafka_server
		}
		common_opts.update(datasource_opts)

		# If providing an activation counts file provide it.
		if options.activation_counts_file:
			common_opts.update({"activation-counts-file": options.activation_counts_file})

		# add ranking stat compression options, if requested
		if options.compress_ranking_stat:
			compress_opts = {
				"compress-ranking-stat": "",
				"compress-ranking-stat-threshold": options.compress_ranking_stat_threshold
			}
			common_opts.update(compress_opts)

		inspNode = dagparts.DAGNode(jobs['gstlalInspiral'], dag, [],
			opts = common_opts,
			input_files = {
				"ranking-stat-input": [likefile],
				"ranking-stat-pdf": options.marginalized_likelihood_file
			},
			output_files = {
				"output": "/dev/null",
				"ranking-stat-output": likefile,
				"zerolag-rankingstat-pdf": zerolikefile
			}
		)

		if str("%04d" %num_insp_nodes) in options.inj_channel_dict:
			# FIXME The node number for injection jobs currently follows the same
			# numbering system as non-injection jobs, except instead of starting at
			# 0000 the numbering starts at 1000. There is probably a better way to
			# do this in the future, this system was just the simplest to start
			# with
			inj_job_tags.append("%04d" % (num_insp_nodes + 1000))

			injection_opts = {
				"channel-name": datasource.pipeline_channel_list_from_channel_dict_with_node_range(options.inj_channel_dict, node = job_tags[-1]),
				"state-channel-name": datasource.pipeline_channel_list_from_channel_dict(options.inj_state_channel_dict, opt = "state-channel-name"),
				"dq-channel-name": datasource.pipeline_channel_list_from_channel_dict(options.inj_dq_channel_dict, opt = "dq-channel-name"),
				"state-vector-on-bits": options.inj_state_vector_on_bits,
				"state-vector-off-bits": options.inj_state_vector_off_bits,
				"dq-vector-on-bits": options.inj_dq_vector_on_bits,
				"dq-vector-off-bits": options.inj_dq_vector_off_bits,
				"gracedb-far-threshold": options.inj_gracedb_far_threshold,
				"gracedb-group": options.inj_gracedb_group,
				"gracedb-pipeline": options.inj_gracedb_pipeline,
				"gracedb-search": options.inj_gracedb_search,
				"gracedb-service-url": options.inj_gracedb_service_url,
				"job-tag": inj_job_tags[-1],
				"likelihood-snapshot-interval": options.likelihood_snapshot_interval,
				"far-trials-factor": options.far_trials_factor,
				"min-instruments": options.min_instruments,
				"time-slide-file": options.time_slide_file
			}

			common_opts.update(injection_opts)
			inspInjNode = dagparts.DAGNode(jobs['gstlalInspiralInj'], dag, [],
				opts = common_opts,
				input_files = {
					"ranking-stat-input": [likefile],
					"ranking-stat-pdf": options.marginalized_likelihood_file
				},
				output_files = {
					"output": "/dev/null"
				}
			)

	return job_tags, inj_job_tags


def event_upload_layer(dag, jobs, options, job_tags):
	job_options = {
		"kafka-server": options.output_kafka_server,
		"gracedb-group": options.gracedb_group,
		"gracedb-pipeline": options.gracedb_pipeline,
		"gracedb-search": options.gracedb_search,
		"gracedb-service-url": options.gracedb_service_url,
		"far-threshold": options.event_aggregator_far_threshold,
		"far-trials-factor": options.event_aggregator_far_trials_factor,
		"upload-cadence-type": options.event_aggregator_upload_cadence_type,
		"upload-cadence-factor": options.event_aggregator_upload_cadence_factor,
		"num-jobs": len(job_tags),
		"input-topic": "events",
		"rootdir": "event_uploader",
		"verbose": "",
	}
	return dagparts.DAGNode(jobs['eventUploader'], dag, [], opts = job_options)


def event_plotter_layer(dag, jobs, options):
	job_options = {
		"kafka-server": options.output_kafka_server,
		"gracedb-group": options.gracedb_group,
		"gracedb-pipeline": options.gracedb_pipeline,
		"gracedb-search": options.gracedb_search,
		"gracedb-service-url": options.gracedb_service_url,
		"verbose": "",
	}
	return dagparts.DAGNode(jobs['eventPlotter'], dag, [], opts = job_options)


def aggregator_layer(dag, jobs, options, job_tags):
	# set up common settings for aggregation jobs
	agg_options = {
		"dump-period": 0,
		"base-dir": "aggregator",
		"job-tag": os.getcwd(),
		"num-jobs": len(job_tags),
		"num-threads": 2,
		"job-start": 0,
		"kafka-server": options.output_kafka_server,
		"data-backend": options.agg_data_backend,
	}

	if options.agg_data_backend == 'influx':
		agg_options.update({
			"influx-database-name": options.influx_database_name,
			"influx-hostname": options.influx_hostname,
			"influx-port": options.influx_port,
		})
		if options.enable_auth:
			agg_options.update({"enable-auth": ""})
		if options.enable_https:
			agg_options.update({"enable-https": ""})

	# define routes used for aggregation jobs
	snr_routes = ["%s_snr_history" % ifo for ifo in options.channel_dict]
	network_routes = ["likelihood_history", "snr_history", "latency_history"]
	state_routes = ["%s_strain_dropped" % ifo for ifo in options.channel_dict]
	usage_routes = ["ram_history"]

	agg_routes = list(itertools.chain(snr_routes, network_routes, usage_routes, state_routes))

	gates = ["%ssegments" % gate for gate in ("statevector", "dqvector", "whiteht")]
	seg_routes = ["%s_%s" % (ifo, gate) for ifo in options.channel_dict for gate in gates]

	# analysis-based aggregation jobs
	# FIXME don't hard code the 1000
	max_agg_jobs = 1000
	agg_job_bounds = range(0, len(job_tags), max_agg_jobs) + [max_agg_jobs]
	agg_routes = list(dagparts.groups(agg_routes, max(max_agg_jobs // (4 * len(job_tags)), 1))) + ["far_history"]
	for routes in agg_routes:
		these_options = dict(agg_options)
		these_options["route"] = routes
		if routes == "far_history":
			these_options["data-type"] = "min"
		else:
			these_options["data-type"] = "max"

		for ii, (aggstart, aggend) in enumerate(zip(agg_job_bounds[:-1], agg_job_bounds[1:])):
			these_options["job-start"] = aggstart
			these_options["num-jobs"] = aggend - aggstart
			if ii == 0: ### elect first aggregator per route as leader
				these_options["across-jobs"] = ""
				aggNode = dagparts.DAGNode(jobs['aggLeader'], dag, [], opts = these_options)
			else:
				aggNode = dagparts.DAGNode(jobs['agg'], dag, [], opts = these_options)

	# segment-based jobs
	seg_routes = list(dagparts.groups(seg_routes, max(max_agg_jobs // (4 * len(job_tags)), 1)))
	for routes in seg_routes:
		these_options = dict(agg_options)
		these_options["route"] = routes
		these_options["data-type"] = "min"

		for ii, (aggstart, aggend) in enumerate(zip(agg_job_bounds[:-1], agg_job_bounds[1:])):
			these_options["job-start"] = aggstart
			these_options["num-jobs"] = aggend - aggstart
			if ii == 0: ### elect first aggregator per route as leader
				these_options["across-jobs"] = ""
				aggNode = dagparts.DAGNode(jobs['aggLeader'], dag, [], opts = these_options)
			else:
				aggNode = dagparts.DAGNode(jobs['agg'], dag, [], opts = these_options)

	# Trigger counting
	trigcount_options = {
		"output-period": 300,
		"num-jobs": len(job_tags),
		"num-threads": 2,
		"job-start": 0,
		"kafka-server": options.output_kafka_server,
		"gracedb-search": options.gracedb_search,
		"gracedb-pipeline": options.gracedb_pipeline,
	}
	dagparts.DAGNode(jobs['trigcount'], dag, [], opts = trigcount_options)

	# Trigger aggregation
	trigagg_options = {
		"dump-period": 0,
		"base-dir": "aggregator",
		"job-tag": os.getcwd(),
		"num-jobs": len(job_tags),
		"num-threads": 2,
		"job-start": 0,
		"kafka-server": options.output_kafka_server,
		"data-backend": options.agg_data_backend,
	}
	if options.agg_data_backend == 'influx':
		trigagg_options.update({
			"influx-database-name": options.influx_database_name,
			"influx-hostname": options.influx_hostname,
			"influx-port": options.influx_port,
		})
		if options.enable_auth:
			trigagg_options.update({"enable-auth": ""})
		if options.enable_https:
			trigagg_options.update({"enable-https": ""})

	return dagparts.DAGNode(jobs['trigagg'], dag, [], opts = trigagg_options)


def dq_monitor_layer(dag, jobs, options):
	outpath = 'aggregator'
	ll_dq_jobs = []

	for ifo in options.channel_dict:
		# Data source dag options
		if (options.data_source == "framexmit"):
			datasource_opts = {
				"framexmit-addr": datasource.framexmit_list_from_framexmit_dict({ifo: options.framexmit_dict[ifo]}),
				"framexmit-iface": options.framexmit_iface
			}
		else:
			datasource_opts = {
				"shared-memory-partition": datasource.pipeline_channel_list_from_channel_dict({ifo: options.shm_part_dict[ifo]}),
				"shared-memory-block-size": options.shared_memory_block_size,
				"shared-memory-assumed-duration": options.shared_memory_assumed_duration
			}

		common_opts = {
			"psd-fft-length": options.psd_fft_length,
			"channel-name": datasource.pipeline_channel_list_from_channel_dict({ifo: options.channel_dict[ifo]}),
			"state-channel-name": datasource.pipeline_channel_list_from_channel_dict({ifo: options.state_channel_dict[ifo]}, opt = "state-channel-name"),
			"dq-channel-name": datasource.pipeline_channel_list_from_channel_dict({ifo: options.dq_channel_dict[ifo]}, opt = "dq-channel-name"),
			"state-vector-on-bits": options.state_vector_on_bits,
			"state-vector-off-bits": options.state_vector_off_bits,
			"dq-vector-on-bits": options.dq_vector_on_bits,
			"dq-vector-off-bits": options.dq_vector_off_bits,
			"data-source": options.data_source,
			"out-path": outpath,
			"data-backend": options.agg_data_backend,
		}
		common_opts.update(datasource_opts)

		if options.agg_data_backend == 'influx':
			common_opts.update({
				"influx-database-name": options.influx_database_name,
				"influx-hostname": options.influx_hostname,
				"influx-port": options.influx_port,
			})
			if options.enable_auth:
				common_opts.update({"enable-auth": ""})
			if options.enable_https:
				common_opts.update({"enable-https": ""})

		ll_dq_jobs.append(dagparts.DAGNode(jobs['dq'], dag, [], opts = common_opts))

	return ll_dq_jobs


def ref_psd_layer(dag, jobs, parent_nodes, segsdict, channel_dict, options):
	psd_nodes = {}
	for ifos in segsdict:
		this_channel_dict = dict((k, channel_dict[k]) for k in ifos if k in channel_dict)
		for seg in segsdict[ifos]:
			psd_path = subdir_path([jobs['refPSD'].output_path, str(int(seg[0]))[:5]])
			psd_nodes[(ifos, seg)] = dagparts.DAGNode(
				jobs['refPSD'],
				dag,
				parent_nodes = parent_nodes,
				opts = {
					"gps-start-time":int(seg[0]),
					"gps-end-time":int(seg[1]),
					"data-source":"frames",
					"channel-name":datasource.pipeline_channel_list_from_channel_dict(this_channel_dict, ifos = ifos),
					"psd-fft-length":options.psd_fft_length,
					"frame-segments-name": options.frame_segments_name
				},
				input_files = {
					"frame-cache":options.frame_cache,
					"frame-segments-file":options.frame_segments_file
				},
				output_files = {
					"write-psd":dagparts.T050017_filename(ifos, "REFERENCE_PSD", seg, '.xml.gz', path = psd_path)
				},
			)

	# Make the reference PSD cache
	# FIXME Use machinery in inspiral_pipe.py to create reference_psd.cache
	with open('reference_psd.cache', "w") as output_cache_file:
		for node in psd_nodes.values():
			output_cache_file.write("%s\n" % CacheEntry.from_T050017("file://localhost%s" % os.path.abspath(node.output_files["write-psd"])))

	return psd_nodes


def median_psd_layer(dag, jobs, parent_nodes, options, boundary_seg, instruments):
	gpsmod5 = str(int(boundary_seg[0]))[:5]
	median_psd_path = subdir_path([jobs['medianPSD'].output_path, gpsmod5])

	# FIXME Use machinery in inspiral_pipe.py to create reference_psd.cache
	median_psd_nodes = []
	for chunk, nodes in enumerate(dagparts.groups(parent_nodes.values(), 50)):
		median_psd_node = dagparts.DAGNode(jobs['medianPSD'], dag,
			parent_nodes = parent_nodes.values(),
			input_files = {"": [node.output_files["write-psd"] for node in nodes]},
			output_files = {"output-name": dagparts.T050017_filename(instruments, "REFERENCE_PSD_CHUNK_%04d" % chunk, boundary_seg, '.xml.gz', path = median_psd_path)}
		)
		median_psd_nodes.append(median_psd_node)

	median_psd_node = dagparts.DAGNode(jobs['medianPSD'], dag,
		parent_nodes = median_psd_nodes,
		input_files = {"": [node.output_files["output-name"] for node in median_psd_nodes]},
		output_files = {"output-name": dagparts.T050017_filename(instruments, "REFERENCE_PSD", boundary_seg, '.xml.gz', path = subdir_path([jobs['medianPSD'].output_path, gpsmod5]))}
	)
	return median_psd_node


def svd_layer(dag, jobs, parent_nodes, psd, bank_cache, options, seg, output_dir, template_mchirp_dict):
	svd_nodes = {}
	new_template_mchirp_dict = {}
	svd_dtdphi_map = {}
	autocorrelation_dict = {}
	for autocorrelation in options.autocorrelation_length:
		min_chirp_mass, max_chirp_mass, autocorrelation_length  = autocorrelation.split(':')
		min_chirp_mass, max_chirp_mass, autocorrelation_length = float(min_chirp_mass), float(max_chirp_mass), int(autocorrelation_length)
		autocorrelation_dict[(min_chirp_mass,max_chirp_mass)] = autocorrelation_length

	for ifo, list_of_svd_caches in bank_cache.items():
		bin_offset = 0
		for j, svd_caches in enumerate(list_of_svd_caches):
			svd_caches = map(CacheEntry, open(svd_caches))
			for i, individual_svd_cache in enumerate(ce.path for ce in svd_caches):
				# First sort out the clipleft, clipright options
				clipleft = []
				clipright = []
				ids = []
				mchirp_interval = (float("inf"), 0)
				individual_svd_cache = map(CacheEntry, open(individual_svd_cache))
				for n, f in enumerate(ce.path for ce in individual_svd_cache):
					# handle template bank clipping
					clipleft.append(options.overlap[j] / 2)
					clipright.append(options.overlap[j] / 2)
					ids.append("%d_%d" % (i+bin_offset, n))
					if f in template_mchirp_dict:
						mchirp_interval = (min(mchirp_interval[0], template_mchirp_dict[f][0]), max(mchirp_interval[1], template_mchirp_dict[f][1]))
					svd_dtdphi_map["%04d" % (i+bin_offset)] = options.dtdphi_file[j]

				svd_bank_name = dagparts.T050017_filename(ifo, '%04d_SVD' % (i+bin_offset,), seg, '.xml.gz', path = jobs['svd'].output_path)
				if '%04d' % (i+bin_offset,) not in new_template_mchirp_dict and mchirp_interval != (float("inf"), 0):
					new_template_mchirp_dict['%04d' % (i+bin_offset,)] = mchirp_interval

				for key, value in autocorrelation_dict.iteritems():
					if key[0] <= new_template_mchirp_dict['%04d' % (i+bin_offset,)][1] < key[1]:
						options.autocorrelation_length = value

				svdnode = dagparts.DAGNode(
					jobs['svd'],
					dag,
					parent_nodes = parent_nodes,
					opts = {
						"svd-tolerance":options.tolerance,
						"flow":options.flow[j],
						"sample-rate":options.sample_rate,
						"clipleft":clipleft,
						"clipright":clipright,
						"samples-min":options.samples_min[j],
						"samples-max-256":options.samples_max_256,
						"samples-max-64":options.samples_max_64,
						"samples-max":options.samples_max,
						"autocorrelation-length":options.autocorrelation_length,
						"bank-id":ids,
						"identity-transform":options.identity_transform,
						"ortho-gate-fap":0.5
					},
					input_files = {"reference-psd":psd},
					input_cache_files = {"template-bank-cache":[ce.path for ce in individual_svd_cache]},
					input_cache_file_name = os.path.basename(svd_bank_name).replace(".xml.gz", ".cache"),
					output_files = {"write-svd":svd_bank_name},
				)

				# impose a priority to help with depth first submission
				svdnode.set_priority(99)
				svd_nodes.setdefault(ifo, []).append(svdnode)
			bin_offset += i+1

	# Plot template/svd bank jobs
	primary_ifo = bank_cache.keys()[0]
	dagparts.DAGNode(
		jobs['plotBanks'],
		dag,
		parent_nodes = sum(svd_nodes.values(),[]),
		opts = {"plot-template-bank":"", "output-dir": output_dir},
		input_files = {"template-bank-file":options.template_bank},
	)

	return svd_nodes, new_template_mchirp_dict, svd_dtdphi_map


def inspiral_layer(dag, jobs, psd_nodes, svd_nodes, segsdict, options, channel_dict, template_mchirp_dict):
	inspiral_nodes = {}
	for ifos in segsdict:
		# FIXME: handles more than 3 ifos with same cpu/memory requests
		inspiral_name = 'gstlalInspiral%dIFO' % min(len(ifos), 3)
		inspiral_inj_name = 'gstlalInspiralInj%dIFO' % min(len(ifos), 3)

		# setup dictionaries to hold the inspiral nodes
		inspiral_nodes[(ifos, None)] = {}
		ignore = {}
		injection_files = []
		for injections in options.injections:
			min_chirp_mass, max_chirp_mass, injections = injections.split(':')
			injection_files.append(injections)
			min_chirp_mass, max_chirp_mass = float(min_chirp_mass), float(max_chirp_mass)
			inspiral_nodes[(ifos, sim_tag_from_inj_file(injections))] = {}
			ignore[injections] = []
			for bgbin_index, bounds in sorted(template_mchirp_dict.items(), key = lambda (k,v): int(k)):
				if max_chirp_mass <= bounds[0]:
					ignore[injections].append(int(bgbin_index))
					# NOTE putting a break here assumes that the min chirp mass
					# in a subbank increases with bin number, i.e. XXXX+1 has a
					# greater minimum chirpmass than XXXX, for all XXXX. Note
					# that the reverse is not true, bin XXXX+1 may have a lower
					# max chirpmass than bin XXXX.
				elif min_chirp_mass > bounds[1]:
					ignore[injections].append(int(bgbin_index))

		# FIXME choose better splitting?
		numchunks = 50

		# only use a channel dict with the relevant channels
		this_channel_dict = dict((k, channel_dict[k]) for k in ifos if k in channel_dict)

		# get the svd bank strings
		svd_bank_strings_full = create_svd_bank_strings(svd_nodes, instruments = this_channel_dict.keys())

		# get a mapping between chunk counter and bgbin for setting priorities
		bgbin_chunk_map = {}

		for seg in segsdict[ifos]:
			if injection_files:
				output_seg_inj_path = subdir_path([jobs[inspiral_inj_name].output_path, str(int(seg[0]))[:5]])

			if jobs[inspiral_name] is None:
				# injection-only run
				inspiral_nodes[(ifos, None)].setdefault(seg, [None])

			else:
				output_seg_path = subdir_path([jobs[inspiral_name].output_path, str(int(seg[0]))[:5]])
				for chunk_counter, svd_bank_strings in enumerate(dagparts.groups(svd_bank_strings_full, numchunks)):
					bgbin_indices = ['%04d' % (i + numchunks * chunk_counter,) for i,s in enumerate(svd_bank_strings)]
					# setup output names
					output_paths = [subdir_path([output_seg_path, bgbin_indices[i]]) for i, s in enumerate(svd_bank_strings)]
					output_names = [dagparts.T050017_filename(ifos, '%s_LLOID' % idx, seg, '.xml.gz', path = path) for idx, path in zip(bgbin_indices, output_paths)]
					dist_stat_names = [dagparts.T050017_filename(ifos, '%s_DIST_STATS' % idx, seg, '.xml.gz', path = path) for idx, path in zip(bgbin_indices, output_paths)]

					for bgbin in bgbin_indices:
						bgbin_chunk_map.setdefault(bgbin, chunk_counter)

					# Calculate the appropriate ht-gate-threshold values according to the scale given
					threshold_values = get_threshold_values(template_mchirp_dict, bgbin_indices, svd_bank_strings, options)

					# non injection node
					noninjnode = dagparts.DAGNode(jobs[inspiral_name], dag,
						parent_nodes = sum((svd_node_list[numchunks*chunk_counter:numchunks*(chunk_counter+1)] for svd_node_list in svd_nodes.values()),[]),
						opts = {
							"psd-fft-length":options.psd_fft_length,
							"ht-gate-threshold":threshold_values,
							"frame-segments-name":options.frame_segments_name,
							"gps-start-time":int(seg[0]),
							"gps-end-time":int(seg[1]),
							"channel-name":datasource.pipeline_channel_list_from_channel_dict(this_channel_dict),
							"tmp-space":dagparts.condor_scratch_space(),
							"track-psd":"",
							"control-peak-time":options.control_peak_time,
							"coincidence-threshold":options.coincidence_threshold,
							"singles-threshold":options.singles_threshold,
							"fir-stride":options.fir_stride,
							"data-source":"frames",
							"local-frame-caching":"",
							"min-instruments":options.min_instruments,
							"reference-likelihood-file":options.reference_likelihood_file
						},
						input_files = {
							"time-slide-file":options.time_slide_file,
							"frame-cache":options.frame_cache,
							"frame-segments-file":options.frame_segments_file,
							"reference-psd":psd_nodes[(ifos, seg)].output_files["write-psd"],
							"blind-injections":options.blind_injections,
							"veto-segments-file":options.vetoes,
						},
						input_cache_files = {"svd-bank-cache":svd_bank_cache_maker(svd_bank_strings)},
						output_cache_files = {
							"output-cache":output_names,
							"ranking-stat-output-cache":dist_stat_names
						}
					)

					# Set a post script to check for file integrity
					if options.gzip_test:
						noninjnode.set_post_script("gzip_test.sh")
						noninjnode.add_post_script_arg(" ".join(output_names + dist_stat_names))

					# impose a priority to help with depth first submission
					noninjnode.set_priority(chunk_counter+15)

					inspiral_nodes[(ifos, None)].setdefault(seg, []).append(noninjnode)

			# process injections
			for injections in injection_files:
				# setup output names
				sim_name = sim_tag_from_inj_file(injections)

				bgbin_svd_bank_strings = [bgbin_svdbank for i, bgbin_svdbank in enumerate(zip(sorted(template_mchirp_dict.keys()), svd_bank_strings_full)) if i not in ignore[injections]]

				for chunk_counter, bgbin_list in enumerate(dagparts.groups(bgbin_svd_bank_strings, numchunks)):
					bgbin_indices, svd_bank_strings = zip(*bgbin_list)
					output_paths = [subdir_path([output_seg_inj_path, bgbin_index]) for bgbin_index in bgbin_indices]
					output_names = [dagparts.T050017_filename(ifos, '%s_LLOID_%s' % (idx, sim_name), seg, '.xml.gz', path = path) for idx, path in zip(bgbin_indices, output_paths)]
					svd_names = [s for i, s in enumerate(svd_bank_cache_maker(svd_bank_strings, injection = True))]
					try:
						reference_psd = psd_nodes[(ifos, seg)].output_files["write-psd"]
						parents = [svd_node_list[int(bgbin_index)] for svd_node_list in svd_nodes.values() for bgbin_index in bgbin_indices]
					except AttributeError: ### injection-only run
						reference_psd = psd_nodes[(ifos, seg)]
						parents = []

					svd_files = [CacheEntry.from_T050017(filename) for filename in svd_names]
					input_cache_name = dagparts.group_T050017_filename_from_T050017_files(svd_files, '.cache').replace('SVD', 'SVD_%s' % sim_name)

					# Calculate the appropriate ht-gate-threshold values according to the scale given
					threshold_values = get_threshold_values(template_mchirp_dict, bgbin_indices, svd_bank_strings, options)

					# setup injection node
					# FIXME: handles more than 3 ifos with same cpu/memory requests
					injnode = dagparts.DAGNode(jobs[inspiral_inj_name], dag,
						parent_nodes = parents,
						opts = {
							"psd-fft-length":options.psd_fft_length,
							"ht-gate-threshold":threshold_values,
							"frame-segments-name":options.frame_segments_name,
							"gps-start-time":int(seg[0]),
							"gps-end-time":int(seg[1]),
							"channel-name":datasource.pipeline_channel_list_from_channel_dict(this_channel_dict),
							"tmp-space":dagparts.condor_scratch_space(),
							"track-psd":"",
							"control-peak-time":options.control_peak_time,
							"coincidence-threshold":options.coincidence_threshold,
							"singles-threshold":options.singles_threshold,
							"fir-stride":options.fir_stride,
							"data-source":"frames",
							"local-frame-caching":"",
							"min-instruments":options.min_instruments,
							"reference-likelihood-file":options.reference_likelihood_file
						},
						input_files = {
							"time-slide-file":options.inj_time_slide_file,
							"frame-cache":options.frame_cache,
							"frame-segments-file":options.frame_segments_file,
							"reference-psd":reference_psd,
							"veto-segments-file":options.vetoes,
							"injections": injections
						},
						input_cache_files = {"svd-bank-cache":svd_names},
						input_cache_file_name = input_cache_name,
						output_cache_files = {"output-cache":output_names}
					)
					# Set a post script to check for file integrity
					if options.gzip_test:
						injnode.set_post_script("gzip_test.sh")
						injnode.add_post_script_arg(" ".join(output_names))

					# impose a priority to help with depth first submission
					if bgbin_chunk_map:
						injnode.set_priority(bgbin_chunk_map[bgbin_indices[-1]]+1)
					else:
						injnode.set_priority(chunk_counter+1)

					inspiral_nodes[(ifos, sim_name)].setdefault(seg, []).append(injnode)

	# Replace mchirplo:mchirphi:inj.xml with inj.xml
	options.injections = [inj.split(':')[-1] for inj in options.injections]

	# NOTE: Adapt the output of the gstlal_inspiral jobs to be suitable for the remainder of this analysis
	lloid_output, lloid_diststats = adapt_gstlal_inspiral_output(inspiral_nodes, options, segsdict)

	return inspiral_nodes, lloid_output, lloid_diststats


def expected_snr_layer(dag, jobs, ref_psd_parent_nodes, options, num_split_inj_snr_jobs):
	ligolw_add_nodes = []
	for inj in options.injections:
		inj_snr_nodes = []

		inj_splitter_node = dagparts.DAGNode(jobs['injSplitter'], dag, parent_nodes=[],
			opts = {
				"output-path":jobs['injSplitter'].output_path,
				"usertag": sim_tag_from_inj_file(inj.split(":")[-1]),
				"nsplit": num_split_inj_snr_jobs
			},
			input_files = {"": inj.split(":")[-1]}
		)
		inj_splitter_node.set_priority(98)

		# FIXME Use machinery in inspiral_pipe.py to create reference_psd.cache
		injection_files = ["%s/%s_INJ_SPLIT_%04d.xml" % (jobs['injSplitter'].output_path, sim_tag_from_inj_file(inj.split(":")[-1]), i) for i in range(num_split_inj_snr_jobs)]
		for injection_file in injection_files:
			injSNRnode = dagparts.DAGNode(jobs['gstlalInjSnr'], dag, parent_nodes=ref_psd_parent_nodes + [inj_splitter_node],
				# FIXME somehow choose the actual flow based on mass?
				# max(flow) is chosen for performance not
				# correctness hopefully though it will be good
				# enough
				opts = {"flow":max(options.flow),"fmax":options.fmax},
				input_files = {
					"injection-file": injection_file,
					"reference-psd-cache": "reference_psd.cache"
				}
			)
			injSNRnode.set_priority(98)
			inj_snr_nodes.append(injSNRnode)

		addnode = dagparts.DAGNode(jobs['ligolwAdd'], dag, parent_nodes=inj_snr_nodes,
			input_files = {"": ' '.join(injection_files)},
			output_files = {"output": os.path.basename(inj.split(":")[-1])}
		)

		ligolw_add_nodes.append(dagparts.DAGNode(jobs['lalappsRunSqlite'], dag, parent_nodes = [addnode],
			opts = {"sql-file":options.injection_proc_sql_file, "tmp-space":dagparts.condor_scratch_space()},
			input_files = {"":addnode.output_files["output"]}
			)
		)
	return ligolw_add_nodes


def summary_plot_layer(dag, jobs, farnode, options, injdbs, noninjdb, output_dir):
	plotnodes = []

	### common plot options
	common_plot_opts = {
		"segments-name": options.frame_segments_name,
		"tmp-space": dagparts.condor_scratch_space(),
		"output-dir": output_dir,
		"likelihood-file":"post_marginalized_likelihood.xml.gz",
		"shrink-data-segments": 32.0,
		"extend-veto-segments": 8.,
	}
	sensitivity_opts = {
		"output-dir":output_dir,
		"tmp-space":dagparts.condor_scratch_space(),
		"veto-segments-name":"vetoes",
		"bin-by-source-type":"",
		"dist-bins":200,
		"data-segments-name":"datasegments"
	}

	### plot summary
	opts = {"user-tag": "ALL_LLOID_COMBINED", "remove-precession": ""}
	opts.update(common_plot_opts)
	plotnodes.append(dagparts.DAGNode(jobs['plotSummary'], dag, parent_nodes=[farnode],
		opts = opts,
		input_files = {"": [noninjdb] + injdbs}
	))

	### isolated precession plot summary
	opts = {"user-tag": "PRECESSION_LLOID_COMBINED", "isolate-precession": "", "plot-group": 1}
	opts.update(common_plot_opts)
	plotnodes.append(dagparts.DAGNode(jobs['plotSummaryIsolatePrecession'], dag, parent_nodes=[farnode],
		opts = opts,
		input_files = {"":[noninjdb] + injdbs}
	))

	for injdb in injdbs:
		### individual injection plot summary
		opts = {"user-tag": injdb.replace(".sqlite","").split("-")[1], "remove-precession": "", "plot-group": 1}
		opts.update(common_plot_opts)
		plotnodes.append(dagparts.DAGNode(jobs['plotSnglInjSummary'], dag, parent_nodes=[farnode],
			opts = opts,
			input_files = {"":[noninjdb] + [injdb]}
		))

		### isolated precession injection plot summary
		opts = {"user-tag": injdb.replace(".sqlite","").split("-")[1].replace("ALL_LLOID","PRECESSION_LLOID"), "isolate-precession": "", "plot-group": 1}
		opts.update(common_plot_opts)
		plotnodes.append(dagparts.DAGNode(jobs['plotSnglInjSummaryIsolatePrecession'], dag, parent_nodes=[farnode],
			opts = opts,
			input_files = {"":[noninjdb] + [injdb]}
		))

	### sensitivity plots
	opts = {"user-tag": "ALL_LLOID_COMBINED"}
	opts.update(sensitivity_opts)
	plotnodes.append(dagparts.DAGNode(jobs['plotSensitivity'], dag, parent_nodes=[farnode],
		opts = opts,
		input_files = {"zero-lag-database": noninjdb, "": injdbs}
	))

	for injdb in injdbs:
		opts = {"user-tag": injdb.replace(".sqlite","").split("-")[1]}
		opts.update(sensitivity_opts)
		plotnodes.append(dagparts.DAGNode(jobs['plotSensitivity'], dag, parent_nodes=[farnode],
			opts = opts,
			input_files = {"zero-lag-database": noninjdb, "": injdb}
		))

	### background plots
	plotnodes.append(dagparts.DAGNode(jobs['plotBackground'], dag, parent_nodes = [farnode],
		opts = {"user-tag":"ALL_LLOID_COMBINED", "output-dir":output_dir},
		input_files = {"":"post_marginalized_likelihood.xml.gz", "database":noninjdb}
	))

	return plotnodes


def clean_merger_products_layer(dag, jobs, plotnodes, dbs_to_delete, margfiles_to_delete):
	"""clean intermediate merger products
	"""
	for db in dbs_to_delete:
		dagparts.DAGNode(jobs['rm'], dag, parent_nodes = plotnodes,
			input_files = {"": db}
		)

	for margfile in margfiles_to_delete:
		dagparts.DAGNode(jobs['rm'], dag, parent_nodes = plotnodes,
			input_files = {"": margfile}
		)
	return None


def inj_psd_layer(segsdict, options):
	psd_nodes = {}
	psd_cache_files = {}
	for ce in map(CacheEntry, open(options.psd_cache)):
		psd_cache_files.setdefault(frozenset(lsctables.instrumentsproperty.get(ce.observatory)), []).append((ce.segment, ce.path))
	for ifos in segsdict:
		reference_psd_files = sorted(psd_cache_files[ifos], key = lambda (s, p): s)
		ref_psd_file_num = 0
		for seg in segsdict[ifos]:
			while int(reference_psd_files[ref_psd_file_num][0][0]) < int(seg[0]):
				ref_psd_file_num += 1
			psd_nodes[(ifos, seg)] = reference_psd_files[ref_psd_file_num][1]
	ref_psd_parent_nodes = []
	return psd_nodes, ref_psd_parent_nodes


def mass_model_layer(dag, jobs, parent_nodes, instruments, options, seg, psd):
	"""mass model node
	"""
	if options.mass_model_file is None:
		# choose, arbitrarily, the lowest instrument in alphabetical order
		model_file_name = dagparts.T050017_filename(instruments, 'ALL_MASS_MODEL', seg, '.h5', path = jobs['model'].output_path)
		model_node = dagparts.DAGNode(jobs['model'], dag,
			input_files = {"template-bank": options.template_bank, "reference-psd": psd},
			opts = {"model":options.mass_model},
			output_files = {"output": model_file_name},
			parent_nodes = parent_nodes
		)
		return [model_node], model_file_name
	else:
		return [], options.mass_model_file


def merge_cluster_layer(dag, jobs, parent_nodes, db, db_cache, sqlfile, input_files=None):
	"""merge and cluster from sqlite database
	"""
	if input_files:
		input_ = {"": input_files}
	else:
		input_ = {}

	# Merge database into chunks
	sqlitenode = dagparts.DAGNode(jobs['toSqlite'], dag, parent_nodes = parent_nodes,
		opts = {"replace":"", "tmp-space":dagparts.condor_scratch_space()},
		input_files = input_,
		input_cache_files = {"input-cache": db_cache},
		output_files = {"database":db},
		input_cache_file_name = os.path.basename(db).replace('.sqlite','.cache')
	)

	# cluster database
	return dagparts.DAGNode(jobs['lalappsRunSqlite'], dag, parent_nodes = [sqlitenode],
		opts = {"sql-file": sqlfile, "tmp-space": dagparts.condor_scratch_space()},
		input_files = {"": db}
	)


def marginalize_layer(dag, jobs, svd_nodes, lloid_output, lloid_diststats, options, boundary_seg, instrument_set, model_node, model_file, ref_psd, svd_dtdphi_map, idq_file = None):
	instruments = "".join(sorted(instrument_set))
	margnodes = {}

	# NOTE! we rely on there being identical templates in each instrument,
	# so we just take one of the values of the svd_nodes which are a dictionary
	# FIXME, the svd nodes list has to be the same as the sorted keys of
	# lloid_output.  svd nodes should be made into a dictionary much
	# earlier in the code to prevent a mishap
	if svd_nodes:
		one_ifo_svd_nodes = dict(("%04d" % n, node) for n, node in enumerate( svd_nodes.values()[0]))

	# Here n counts the bins
	# FIXME - this is broken for injection dags right now because of marg nodes
	# first non-injections, which will get skipped if this is an injections-only run
	for bin_key in sorted(lloid_output[None].keys()):
		outputs = lloid_output[None][bin_key]
		diststats = lloid_diststats[bin_key]
		inputs = [o[0] for o in outputs]
		parents = dagparts.flatten([o[1] for o in outputs])
		rankfile = functools.partial(get_rank_file, instruments, boundary_seg, bin_key)

		if svd_nodes:
			parent_nodes = [one_ifo_svd_nodes[bin_key]] + model_node
			svd_file = one_ifo_svd_nodes[bin_key].output_files["write-svd"]
		else:
			parent_nodes = model_node
			svd_path = os.path.join(options.analysis_path, jobs['svd'].output_path)
			svd_file = dagparts.T050017_filename(instrument_set[0], '%s_SVD' % bin_key, boundary_seg, '.xml.gz', path = svd_path)

		# FIXME we keep this here in case we someday want to have a
		# mass bin dependent prior, but it really doesn't matter for
		# the time being.
		prior_input_files = {
			"svd-file": svd_file,
			"mass-model-file": model_file,
			"dtdphi-file": svd_dtdphi_map[bin_key],
			"psd-xml": ref_psd
		}
		if idq_file is not None:
			prior_input_files["idq-file"] = idq_file
		priornode = dagparts.DAGNode(jobs['createPriorDistStats'], dag,
			parent_nodes = parent_nodes,
			opts = {
				"instrument": instrument_set,
				"background-prior": 1,
				"min-instruments": options.min_instruments,
				"coincidence-threshold":options.coincidence_threshold,
				"df": "bandwidth"
			},
			input_files = prior_input_files,
			output_files = {"write-likelihood": rankfile('CREATE_PRIOR_DIST_STATS', job=jobs['createPriorDistStats'])}
		)
		# Create a file that has the priors *and* all of the diststats
		# for a given bin marginalized over time. This is all that will
		# be needed to compute the likelihood
		diststats_per_bin_node = dagparts.DAGNode(jobs['marginalize'], dag,
			parent_nodes = [priornode] + parents,
			opts = {"marginalize": "ranking-stat"},
			input_cache_files = {"likelihood-cache": diststats + [priornode.output_files["write-likelihood"]]},
			output_files = {"output": rankfile('MARG_DIST_STATS', job=jobs['marginalize'])},
			input_cache_file_name = rankfile('MARG_DIST_STATS')
		)

		margnodes[bin_key] = diststats_per_bin_node

	return margnodes


def calc_rank_pdf_layer(dag, jobs, marg_nodes, options, boundary_seg, instrument_set, with_zero_lag = False):
	rankpdf_nodes = []
	rankpdf_zerolag_nodes = []
	instruments = "".join(sorted(instrument_set))

	# Here n counts the bins
	for bin_key in sorted(marg_nodes.keys()):
		rankfile = functools.partial(get_rank_file, instruments, boundary_seg, bin_key)

		calcranknode = dagparts.DAGNode(jobs['calcRankPDFs'], dag,
			parent_nodes = [marg_nodes[bin_key]],
			opts = {"ranking-stat-samples":options.ranking_stat_samples},
			input_files = {"": marg_nodes[bin_key].output_files["output"]},
			output_files = {"output": rankfile('CALC_RANK_PDFS', job=jobs['calcRankPDFs'])},
		)
		rankpdf_nodes.append(calcranknode)

		if with_zero_lag:
			calcrankzerolagnode = dagparts.DAGNode(jobs['calcRankPDFsWithZerolag'], dag,
				parent_nodes = [marg_nodes[bin_key]],
				opts = {"add-zerolag-to-background": "", "ranking-stat-samples": options.ranking_stat_samples},
				input_files = {"": marg_nodes[bin_key].output_files["output"]},
				output_files = {"output": rankfile('CALC_RANK_PDFS_WZL', job=jobs['calcRankPDFsWithZerolag'])},
			)
			rankpdf_zerolag_nodes.append(calcrankzerolagnode)

	return rankpdf_nodes, rankpdf_zerolag_nodes


def likelihood_layer(dag, jobs, marg_nodes, lloid_output, lloid_diststats, options, boundary_seg, instrument_set):
	likelihood_nodes = {}
	instruments = "".join(sorted(instrument_set))

	# non-injection jobs
	for bin_key in sorted(lloid_output[None].keys()):
		outputs = lloid_output[None][bin_key]
		diststats = lloid_diststats[bin_key]
		inputs = [o[0] for o in outputs]

		# (input files for next job, dist stat files, parents for next job)
		if bin_key in marg_nodes:
			likelihood_url = marg_nodes[bin_key].output_files["output"]
			parents = [marg_nodes[bin_key]]
		else:
			likelihood_url = lloid_diststats[bin_key][0]
			parents = []

		likelihood_nodes[None, bin_key] = (inputs, likelihood_url, parents)

	# injection jobs
	for inj in options.injections:
		lloid_nodes = lloid_output[sim_tag_from_inj_file(inj)]
		for bin_key in sorted(lloid_nodes.keys()):
			outputs = lloid_nodes[bin_key]
			diststats = lloid_diststats[bin_key]
			if outputs is not None:
				inputs = [o[0] for o in outputs]
				parents = dagparts.flatten([o[1] for o in outputs])

				if bin_key in marg_nodes:
					parents.append(marg_nodes[bin_key])
					likelihood_url = marg_nodes[bin_key].output_files["output"]
				else:
					likelihood_url = lloid_diststats[bin_key][0]

				likelihood_nodes[sim_tag_from_inj_file(inj), bin_key] = (inputs, likelihood_url, parents)

	return likelihood_nodes


def sql_cluster_and_merge_layer(dag, jobs, likelihood_nodes, ligolw_add_nodes, options, boundary_seg, instruments, with_zero_lag = False):
	num_chunks = 100
	innodes = {}

	# after assigning the likelihoods cluster and merge by sub bank and whether or not it was an injection run
	for (sim_tag, bin_key), (inputs, likelihood_url, parents) in sorted(likelihood_nodes.items()):
		db = inputs_to_db(jobs, inputs, job_type = 'toSqliteNoCache')
		xml = inputs_to_db(jobs, inputs, job_type = 'ligolwAdd').replace(".sqlite", ".xml.gz")
		snr_cluster_sql_file = options.snr_cluster_sql_file if sim_tag is None else options.injection_snr_cluster_sql_file
		cluster_sql_file = options.cluster_sql_file if sim_tag is None else options.injection_sql_file
		likelihood_job = jobs['calcLikelihood'] if sim_tag is None else jobs['calcLikelihoodInj']

		# If we have only have 1 input file per bin, assume file is already clustered
		# FIXME This means dags that run over a single segment O(1000)
		# seconds will not have snr chisq clustering applied before the
		# likelihood-ratio assignment, which also means those dags will
		# not be able to be reranked. This is probably not a big deal,
		# because a dag that small can quickly be rerun
		if len(inputs) > 1:
			# cluster sub banks
			cluster_node = dagparts.DAGNode(jobs['lalappsRunSqlite'], dag, parent_nodes = parents,
				opts = {"sql-file": snr_cluster_sql_file, "tmp-space":dagparts.condor_scratch_space()},
				input_files = {"":inputs}
				)

			# merge sub banks
			merge_node = dagparts.DAGNode(jobs['ligolwAdd'], dag, parent_nodes = [cluster_node],
				input_files = {"":inputs},
				output_files = {"output":xml}
				)

			# cluster and simplify sub banks
			cluster_node = [dagparts.DAGNode(jobs['lalappsRunSqlite'], dag, parent_nodes = [merge_node],
				opts = {"sql-file": snr_cluster_sql_file, "tmp-space":dagparts.condor_scratch_space()},
				input_files = {"":xml}
				)]

		else:
			cluster_node = []

		# assign likelihoods
		likelihood_node = dagparts.DAGNode(likelihood_job, dag,
			parent_nodes = cluster_node,
			opts = {"tmp-space": dagparts.condor_scratch_space(), "force": ""},
			input_files = {"likelihood-url":likelihood_url, "": xml}
			)

		sqlitenode = dagparts.DAGNode(jobs['toSqliteNoCache'], dag, parent_nodes = [likelihood_node],
			opts = {"replace":"", "tmp-space":dagparts.condor_scratch_space()},
			input_files = {"":xml},
			output_files = {"database":db},
		)
		sqlitenode = dagparts.DAGNode(jobs['lalappsRunSqlite'], dag, parent_nodes = [sqlitenode],
			opts = {"sql-file": cluster_sql_file, "tmp-space":dagparts.condor_scratch_space()},
			input_files = {"":db}
		)

		innodes.setdefault(sim_tag_from_inj_file(sim_tag) if sim_tag else None, []).append(sqlitenode)

	# make sure outnodes has a None key, even if its value is an empty list
	# FIXME injection dag is broken
	innodes.setdefault(None, [])

	if options.vetoes is None:
		vetoes = []
	else:
		vetoes = [options.vetoes]

	chunk_nodes = []
	dbs_to_delete = []
	# Process the chirp mass bins in chunks to paralellize the merging process
	for chunk, nodes in enumerate(dagparts.groups(innodes[None], num_chunks)):
		try:
			dbs = [node.input_files[""] for node in nodes]
			parents = nodes

		except AttributeError:
			# analysis started at merger step but seeded by lloid files which
			# have already been merged into one file per background
			# bin, thus the analysis will begin at this point
			dbs = nodes
			parents = []

		dbfiles = [CacheEntry.from_T050017("file://localhost%s" % os.path.abspath(filename)) for filename in dbs]
		noninjdb = dagparts.group_T050017_filename_from_T050017_files(dbfiles, '.sqlite', path = jobs['toSqlite'].output_path)

		# Merge and cluster the final non injection database
		noninjsqlitenode = merge_cluster_layer(dag, jobs, parents, noninjdb, dbs, options.cluster_sql_file)
		chunk_nodes.append(noninjsqlitenode)
		dbs_to_delete.append(noninjdb)

	# Merge the final non injection database
	outnodes = []
	injdbs = []
	if options.non_injection_db: #### injection-only run
		noninjdb = options.non_injection_db
	else:
		final_nodes = []
		for chunk, nodes in enumerate(dagparts.groups(innodes[None], num_chunks)):
			noninjdb = dagparts.T050017_filename(instruments, 'PART_LLOID_CHUNK_%04d' % chunk, boundary_seg, '.sqlite')

			# cluster the final non injection database
			noninjsqlitenode = merge_cluster_layer(dag, jobs, nodes, noninjdb, [node.input_files[""] for node in nodes], options.cluster_sql_file)
			final_nodes.append(noninjsqlitenode)

		input_files = (vetoes + [options.frame_segments_file])
		input_cache_files = [node.input_files[""] for node in final_nodes]
		noninjdb = dagparts.T050017_filename(instruments, 'ALL_LLOID', boundary_seg, '.sqlite')
		noninjsqlitenode = merge_cluster_layer(dag, jobs, final_nodes, noninjdb, input_cache_files, options.cluster_sql_file, input_files=input_files)

		if with_zero_lag:
			cpnode = dagparts.DAGNode(jobs['cp'], dag, parent_nodes = [noninjsqlitenode],
				input_files = {"":"%s %s" % (noninjdb, noninjdb.replace('ALL_LLOID', 'ALL_LLOID_WZL'))}
			)
			outnodes.append(cpnode)
		else:
			outnodes.append(noninjsqlitenode)

	if options.injections:
		iterable_injections = options.injections
	else:
		iterable_injections = options.injections_for_merger

	for injections in iterable_injections:
		# extract only the nodes that were used for injections
		chunk_nodes = []

		for chunk, injnodes in enumerate(dagparts.groups(innodes[sim_tag_from_inj_file(injections)], num_chunks)):
			try:
				dbs = [injnode.input_files[""] for injnode in injnodes]
				parents = injnodes
			except AttributeError:
				dbs = injnodes
				parents = []

			# Setup the final output names, etc.
			dbfiles = [CacheEntry.from_T050017("file://localhost%s" % os.path.abspath(filename)) for filename in dbs]
			injdb = dagparts.group_T050017_filename_from_T050017_files(dbfiles, '.sqlite', path = jobs['toSqlite'].output_path)

			# merge and cluster
			clusternode = merge_cluster_layer(dag, jobs, parents, injdb, dbs, options.cluster_sql_file)
			chunk_nodes.append(clusternode)
			dbs_to_delete.append(injdb)


		final_nodes = []
		for chunk, injnodes in enumerate(dagparts.groups(innodes[sim_tag_from_inj_file(injections)], num_chunks)):
			# Setup the final output names, etc.
			injdb = dagparts.T050017_filename(instruments, 'PART_LLOID_%s_CHUNK_%04d' % (sim_tag_from_inj_file(injections), chunk), boundary_seg, '.sqlite')

			# merge and cluster
			clusternode = merge_cluster_layer(dag, jobs, injnodes, injdb, [node.input_files[""] for node in injnodes], options.cluster_sql_file)
			final_nodes.append(clusternode)

		# Setup the final output names, etc.
		injdb = dagparts.T050017_filename(instruments, 'ALL_LLOID_%s' % sim_tag_from_inj_file(injections), boundary_seg, '.sqlite')
		injdbs.append(injdb)
		injxml = injdb.replace('.sqlite','.xml.gz')

		xml_input = injxml

		# merge and cluster
		parent_nodes = final_nodes + ligolw_add_nodes
		input_files = (vetoes + [options.frame_segments_file, injections])
		input_cache_files = [node.input_files[""] for node in final_nodes]
		clusternode = merge_cluster_layer(dag, jobs, parent_nodes, injdb, input_cache_files, options.cluster_sql_file, input_files=input_files)

		clusternode = dagparts.DAGNode(jobs['toXML'], dag, parent_nodes = [clusternode],
			opts = {"tmp-space":dagparts.condor_scratch_space()},
			output_files = {"extract":injxml},
			input_files = {"database":injdb}
		)

		inspinjnode = dagparts.DAGNode(jobs['ligolwInspinjFind'], dag, parent_nodes = [clusternode],
			opts = {"time-window":0.9},
			input_files = {"":injxml}
		)

		sqlitenode = dagparts.DAGNode(jobs['toSqliteNoCache'], dag, parent_nodes = [inspinjnode],
			opts = {"replace":"", "tmp-space":dagparts.condor_scratch_space()},
			output_files = {"database":injdb},
			input_files = {"":xml_input}
		)

		if with_zero_lag:
			cpnode = dagparts.DAGNode(jobs['cp'], dag, parent_nodes = [sqlitenode],
				input_files = {"":"%s %s" % (injdb, injdb.replace('ALL_LLOID', 'ALL_LLOID_WZL'))}
			)
			outnodes.append(cpnode)
		else:
			outnodes.append(sqlitenode)

	return injdbs, noninjdb, outnodes, dbs_to_delete


def final_marginalize_layer(dag, jobs, rankpdf_nodes, rankpdf_zerolag_nodes, options, with_zero_lag = False):
	ranknodes = [rankpdf_nodes, rankpdf_zerolag_nodes]
	margjobs = [jobs['marginalize'], jobs['marginalizeWithZerolag']]
	margfiles = [options.marginalized_likelihood_file, options.marginalized_likelihood_file]
	if with_zero_lag:
		filesuffixs = ['', '_with_zerolag']
	else:
		filesuffixs = ['']
	margnum = 16
	all_margcache = []
	all_margnodes = []
	final_margnodes = []
	for nodes, job, margfile, filesuffix in zip(ranknodes, margjobs, margfiles, filesuffixs):
		try:
			margin = [node.output_files["output"] for node in nodes]
			parents = nodes
		except AttributeError: ### analysis started at merger step
			margin = nodes
			parents = []

		margnodes = []
		margcache = []

		# split up the marginalization into groups of 10
		# FIXME: is it actually groups of 10 or groups of 16?
		for margchunk in dagparts.groups(margin, margnum):
			if nodes:
				marg_ce = [CacheEntry.from_T050017("file://localhost%s" % os.path.abspath(filename)) for filename in margchunk]
				margcache.append(dagparts.group_T050017_filename_from_T050017_files(marg_ce, '.xml.gz', path = job.output_path))
				margnodes.append(dagparts.DAGNode(job, dag, parent_nodes = parents,
					opts = {"marginalize": "ranking-stat-pdf"},
					output_files = {"output": margcache[-1]},
					input_cache_files = {"likelihood-cache": margchunk},
					input_cache_file_name = os.path.basename(margcache[-1]).replace('.xml.gz','.cache')
				))

		all_margcache.append(margcache)
		all_margnodes.append(margnodes)

	if not options.marginalized_likelihood_file: ### not an injection-only run
		for nodes, job, margnodes, margcache, margfile, filesuffix in zip(ranknodes, margjobs, all_margnodes, all_margcache, margfiles, filesuffixs):
			final_margnodes.append(dagparts.DAGNode(job, dag, parent_nodes = margnodes,
				opts = {"marginalize": "ranking-stat-pdf"},
				output_files = {"output": "marginalized_likelihood%s.xml.gz"%filesuffix},
				input_cache_files = {"likelihood-cache": margcache},
				input_cache_file_name = "marginalized_likelihood%s.cache"%filesuffix
			))

	return final_margnodes, dagparts.flatten(all_margcache)


def compute_far_layer(dag, jobs, margnodes, injdbs, noninjdb, final_sqlite_nodes, options, with_zero_lag = False):
	"""compute FAPs and FARs
	"""
	margfiles = [options.marginalized_likelihood_file, options.marginalized_likelihood_file]
	if with_zero_lag:
		filesuffixs = ['', '_with_zerolag']
	else:
		filesuffixs = ['']
	if options.marginalized_likelihood_file:
		assert not margnodes, "no marg nodes should be produced in an injection-only DAG"
		margnodes = [None, None]

	for margnode, margfile, filesuffix in zip(margnodes, margfiles, filesuffixs):
		if options.marginalized_likelihood_file:
			parents = final_sqlite_nodes
			marginalized_likelihood_file = margfile
		else:
			parents = [margnode] + final_sqlite_nodes
			marginalized_likelihood_file = margnode.output_files["output"]

		farnode = dagparts.DAGNode(jobs['ComputeFarFromSnrChisqHistograms'], dag, parent_nodes = parents,
			opts = {"tmp-space":dagparts.condor_scratch_space()},
			input_files = {
				"background-bins-file": marginalized_likelihood_file,
				"injection-db": [injdb.replace('ALL_LLOID', 'ALL_LLOID_WZL') for injdb in injdbs] if 'zerolag' in filesuffix else injdbs,
				"non-injection-db": noninjdb.replace('ALL_LLOID', 'ALL_LLOID_WZL') if 'zerolag' in filesuffix else noninjdb
			}
		)

		if 'zerolag' not in filesuffix:
			outnode = farnode

	return outnode


def horizon_dist_layer(dag, jobs, psd_nodes, options, boundary_seg, output_dir, instruments):
	"""calculate horizon distance
	"""
	dagparts.DAGNode(jobs['horizon'], dag,
		parent_nodes = psd_nodes.values(),
		input_files = {"":[node.output_files["write-psd"] for node in psd_nodes.values()]},
		output_files = {"":dagparts.T050017_filename(instruments, "HORIZON", boundary_seg, '.png', path = output_dir)}
	)


def summary_page_layer(dag, jobs, plotnodes, options, boundary_seg, injdbs, output_dir):
	"""create a summary page
	"""
	output_user_tags = ["ALL_LLOID_COMBINED", "PRECESSION_LLOID_COMBINED"]
	output_user_tags.extend([injdb.replace(".sqlite","").split("-")[1] for injdb in injdbs])
	output_user_tags.extend([injdb.replace(".sqlite","").split("-")[1].replace("ALL_LLOID", "PRECESSION_LLOID") for injdb in injdbs])

	dagparts.DAGNode(jobs['summaryPage'], dag, parent_nodes = plotnodes,
		opts = {
			"title":"gstlal-%d-%d-closed-box" % (int(boundary_seg[0]), int(boundary_seg[1])),
			"webserver-dir":options.web_dir,
			"glob-path":output_dir,
			"output-user-tag":output_user_tags
		}
	)


#
# environment utilities
#


def webserver_url():
	"""!
	The stupid pet tricks to find webserver on the LDG.
	"""
	host = socket.getfqdn()
	#FIXME add more hosts as you need them
	if "cit" in host or "ligo.caltech.edu" in host:
		return "https://ldas-jobs.ligo.caltech.edu"
	if ".phys.uwm.edu" in host or ".cgca.uwm.edu" in host or ".nemo.uwm.edu" in host:
		return "https://ldas-jobs.cgca.uwm.edu"
	# FIXME:  this next system does not have a web server, but not
	# having a web server is treated as a fatal error so we have to
	# make something up if we want to make progress
	if ".icrr.u-tokyo.ac.jp" in host:
		return "https://ldas-jobs.icrr.u-tokyo.ac.jp"

	raise NotImplementedError("I don't know where the webserver is for this environment")


#
# DAG utilities
#


def load_analysis_output(options):
	# load triggers
	bgbin_lloid_map = defaultdict(dict)
	for ce in map(CacheEntry, open(options.lloid_cache)):
		try:
			bgbin_idx, _, inj = ce.description.split('_', 2)
		except:
			bgbin_idx, _ = ce.description.split('_', 1)
			inj = None
		finally:
			bgbin_lloid_map[sim_tag_from_inj_file(inj)].setdefault(bgbin_idx, []).append((ce.path, []))

	# load dist stats
	lloid_diststats = {}
	for ce in map(CacheEntry, open(options.dist_stats_cache)):
		if 'DIST_STATS' in ce.description and not 'CREATE_PRIOR' in ce.description:
			lloid_diststats.setdefault(ce.description.split("_")[0], []).append(ce.path)

	# load svd dtdphi map
	svd_dtdphi_map, instrument_set = load_svd_dtdphi_map(options)

	# modify injections option, as is done in 'adapt_inspiral_output'
	# FIXME: don't do this, find a cleaner way of handling this generally
	options.injections = [inj.split(':')[-1] for inj in options.injections]

	return bgbin_lloid_map, lloid_diststats, svd_dtdphi_map, instrument_set


def load_svd_dtdphi_map(options):
	svd_dtdphi_map = {}
	bank_cache = load_bank_cache(options)
	instrument_set = bank_cache.keys()
	for ifo, list_of_svd_caches in bank_cache.items():
		bin_offset = 0
		for j, svd_caches in enumerate(list_of_svd_caches):
			for i, individual_svd_cache in enumerate(ce.path for ce in map(CacheEntry, open(svd_caches))):
				svd_dtdphi_map["%04d" % (i+bin_offset)] = options.dtdphi_file[j]
			bin_offset += i+1

	return svd_dtdphi_map, instrument_set


def get_threshold_values(template_mchirp_dict, bgbin_indices, svd_bank_strings, options):
	"""Calculate the appropriate ht-gate-threshold values according to the scale given
	"""
	if options.ht_gate_threshold_linear is not None:
		# A scale is given
		mchirp_min, ht_gate_threshold_min, mchirp_max, ht_gate_threshold_max = [float(y) for x in options.ht_gate_threshold_linear.split("-") for y in x.split(":")]
		# use max mchirp in a given svd bank to decide gate threshold
		bank_mchirps = [template_mchirp_dict[bgbin_index][1] for bgbin_index in bgbin_indices]
		gate_mchirp_ratio = (ht_gate_threshold_max - ht_gate_threshold_min)/(mchirp_max - mchirp_min)
		return [gate_mchirp_ratio*(bank_mchirp - mchirp_min) + ht_gate_threshold_min for bank_mchirp in bank_mchirps]
	elif options.ht_gate_threshold is not None:
		return [options.ht_gate_threshold]*len(svd_bank_strings) # Use the ht-gate-threshold value given
	else:
		return None


def inputs_to_db(jobs, inputs, job_type = 'toSqlite'):
	dbfiles = [CacheEntry.from_T050017("file://localhost%s" % os.path.abspath(filename)) for filename in inputs]
	db = dagparts.group_T050017_filename_from_T050017_files(dbfiles, '.sqlite')
	return os.path.join(subdir_path([jobs[job_type].output_path, CacheEntry.from_T050017(db).description[:4]]), db)


def cache_to_db(cache, jobs):
	hi_index = cache[-1].description.split('_')[0]
	db = os.path.join(jobs['toSqlite'].output_path, os.path.basename(cache[-1].path))
	db.replace(hi_index, '%04d' % ((int(hi_index) + 1) / options.num_files_per_background_bin - 1,))
	return db


def get_rank_file(instruments, boundary_seg, n, basename, job=None):
	if job:
		return dagparts.T050017_filename(instruments, '_'.join([n, basename]), boundary_seg, '.xml.gz', path = job.output_path)
	else:
		return dagparts.T050017_filename(instruments, '_'.join([n, basename]), boundary_seg, '.cache')


#
# Utility functions
#


def group(inlist, parts):
	"""!
	group a list roughly according to the distribution in parts, e.g.

	>>> A = range(12)
	>>> B = [2,3]
	>>> for g in group(A,B):
	...     print g
	...
	[0, 1]
	[2, 3]
	[4, 5]
	[6, 7, 8]
	[9, 10, 11]
	"""
	mult_factor = len(inlist) // sum(parts) + 1
	l = copy.deepcopy(inlist)
	for i, p in enumerate(parts):
		for j in range(mult_factor):
			if not l:
				break
			yield l[:p]
			del l[:p]


def parse_cache_str(instr):
	"""!
	A way to decode a command line option that specifies different bank
	caches for different detectors, e.g.,

	>>> bankcache = parse_cache_str("H1=H1_split_bank.cache,L1=L1_split_bank.cache,V1=V1_split_bank.cache")
	>>> bankcache
	{'V1': 'V1_split_bank.cache', 'H1': 'H1_split_bank.cache', 'L1': 'L1_split_bank.cache'}
	"""

	dictcache = {}
	if instr is None: return dictcache
	for c in instr.split(','):
		ifo = c.split("=")[0]
		cache = c.replace(ifo+"=","")
		dictcache[ifo] = cache
	return dictcache


def build_bank_groups(cachedict, numbanks = [2], maxjobs = None):
	"""!
	given a dictionary of bank cache files keyed by ifo from .e.g.,
	parse_cache_str(), group the banks into suitable size chunks for a single svd
	bank file according to numbanks.  Note, numbanks can be should be a list and uses
	the algorithm in the group() function
	"""
	outstrs = []
	ifos = sorted(cachedict.keys())
	files = zip(*[[CacheEntry(f).path for f in open(cachedict[ifo],'r').readlines()] for ifo in ifos])
	for n, bank_group in enumerate(group(files, numbanks)):
		if maxjobs is not None and n > maxjobs:
			break
		c = dict(zip(ifos, zip(*bank_group)))
		outstrs.append(c)

	return outstrs


def get_svd_bank_params_online(svd_bank_cache):
	template_mchirp_dict = {}
	for ii, ce in enumerate([CacheEntry(f) for f in open(svd_bank_cache)]):
		min_mchirp, max_mchirp = float("inf"), 0
		xmldoc = ligolw_utils.load_url(ce.path, contenthandler = svd_bank.DefaultContentHandler)
		for root in (elem for elem in xmldoc.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") and elem.Name == "gstlal_svd_bank_Bank"):
			snglinspiraltable = lsctables.SnglInspiralTable.get_table(root)
			mchirp_column = snglinspiraltable.getColumnByName("mchirp")
			min_mchirp, max_mchirp = min(min_mchirp, min(mchirp_column)), max(max_mchirp, max(mchirp_column))
		template_mchirp_dict["%04d" % ii] = (min_mchirp, max_mchirp)
		xmldoc.unlink()
	return template_mchirp_dict


def get_svd_bank_params(svd_bank_cache, online = False):
	if not online:
		bgbin_file_map = {}
		max_time = 0
	template_mchirp_dict = {}
	for ce in sorted([CacheEntry(f) for f in open(svd_bank_cache)], cmp = lambda x,y: cmp(int(x.description.split("_")[0]), int(y.description.split("_")[0]))):
		if not online:
			bgbin_file_map.setdefault(ce.observatory, []).append(ce.path)
		if not template_mchirp_dict.setdefault(ce.description.split("_")[0], []):
			min_mchirp, max_mchirp = float("inf"), 0
			xmldoc = ligolw_utils.load_url(ce.path, contenthandler = svd_bank.DefaultContentHandler)
			for root in (elem for elem in xmldoc.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") and elem.Name == "gstlal_svd_bank_Bank"):
				snglinspiraltable = lsctables.SnglInspiralTable.get_table(root)
				mchirp_column = snglinspiraltable.getColumnByName("mchirp")
				min_mchirp, max_mchirp = min(min_mchirp, min(mchirp_column)), max(max_mchirp, max(mchirp_column))
				if not online:
					max_time = max(max_time, max(snglinspiraltable.getColumnByName("template_duration")))
			template_mchirp_dict[ce.description.split("_")[0]] = (min_mchirp, max_mchirp)
			xmldoc.unlink()
	if not online:
		return template_mchirp_dict, bgbin_file_map, max_time
	else:
		return template_mchirp_dict


def sim_tag_from_inj_file(injections):
	if injections is None:
		return None
	return os.path.basename(injections).replace('.xml', '').replace('.gz', '').replace('-','_')


def load_bank_cache(options):
	bank_cache = {}
	for bank_cache_str in options.bank_cache:
		for c in bank_cache_str.split(','):
			ifo = c.split("=")[0]
			cache = c.replace(ifo+"=","")
			bank_cache.setdefault(ifo, []).append(cache)

	return bank_cache


def get_bank_params(options, verbose = False):
	bank_cache = load_bank_cache(options)

	max_time = 0
	template_mchirp_dict = {}
	for n, cache in enumerate(bank_cache.values()[0]):
		for ce in map(CacheEntry, open(cache)):
			for ce in map(CacheEntry, open(ce.path)):
				xmldoc = ligolw_utils.load_filename(ce.path, verbose = verbose, contenthandler = LIGOLWContentHandler)
				snglinspiraltable = lsctables.SnglInspiralTable.get_table(xmldoc)
				max_time = max(max_time, max(snglinspiraltable.getColumnByName('template_duration')))
				idx = options.overlap[n]/2
				template_mchirp_dict[ce.path] = [min(snglinspiraltable.getColumnByName('mchirp')[idx:-idx]), max(snglinspiraltable.getColumnByName('mchirp')[idx:-idx])]
				xmldoc.unlink()

	return template_mchirp_dict, bank_cache, max_time


def subdir_path(dirlist):
	output_path = '/'.join(dirlist)
	try:
		os.mkdir(output_path)
	except:
		pass
	return output_path


def analysis_segments(analyzable_instruments_set, allsegs, boundary_seg, max_template_length, min_instruments = 2):
	"""get a dictionary of all the disjoint 2+ detector combination segments
	"""
	segsdict = segments.segmentlistdict()
	# 512 seconds for the whitener to settle + the maximum template_length FIXME don't hard code
	start_pad = 512 + max_template_length
	# Chosen so that the overlap is only a ~5% hit in run time for long segments...
	segment_length = int(5 * start_pad)
	for n in range(min_instruments, 1 + len(analyzable_instruments_set)):
		for ifo_combos in itertools.combinations(list(analyzable_instruments_set), n):
			segsdict[frozenset(ifo_combos)] = allsegs.intersection(ifo_combos) - allsegs.union(analyzable_instruments_set - set(ifo_combos))
			segsdict[frozenset(ifo_combos)] &= segments.segmentlist([boundary_seg])
			segsdict[frozenset(ifo_combos)] = segsdict[frozenset(ifo_combos)].protract(start_pad)
			segsdict[frozenset(ifo_combos)] = dagparts.breakupsegs(segsdict[frozenset(ifo_combos)], segment_length, start_pad)
			if not segsdict[frozenset(ifo_combos)]:
				del segsdict[frozenset(ifo_combos)]
	return segsdict


def create_svd_bank_strings(svd_nodes, instruments = None):
	# FIXME assume that the number of svd nodes is the same per ifo, a good assumption though
	outstrings = []
	for i in range(len(svd_nodes.values()[0])):
		svd_bank_string = ""
		for ifo in svd_nodes:
			if instruments is not None and ifo not in instruments:
				continue
			try:
				svd_bank_string += "%s:%s," % (ifo, svd_nodes[ifo][i].output_files["write-svd"])
			except AttributeError:
				svd_bank_string += "%s:%s," % (ifo, svd_nodes[ifo][i])
		svd_bank_string = svd_bank_string.strip(",")
		outstrings.append(svd_bank_string)
	return outstrings


def svd_bank_cache_maker(svd_bank_strings, injection = False):
	if injection:
		dir_name = "gstlal_inspiral_inj"
	else:
		dir_name = "gstlal_inspiral"
	svd_cache_entries = []
	parsed_svd_bank_strings = [inspiral.parse_svdbank_string(single_svd_bank_string) for single_svd_bank_string in svd_bank_strings]
	for svd_bank_parsed_dict in parsed_svd_bank_strings:
		for filename in svd_bank_parsed_dict.itervalues():
			svd_cache_entries.append(CacheEntry.from_T050017(filename))

	return [svd_cache_entry.url for svd_cache_entry in svd_cache_entries]


def adapt_gstlal_inspiral_output(inspiral_nodes, options, segsdict):
	# first get the previous output in a usable form
	lloid_output = {}
	for inj in options.injections + [None]:
		lloid_output[sim_tag_from_inj_file(inj)] = {}
	lloid_diststats = {}
	if options.dist_stats_cache:
		for ce in map(CacheEntry, open(options.dist_stats_cache)):
			lloid_diststats[ce.description.split("_")[0]] = [ce.path]
	for ifos in segsdict:
		for seg in segsdict[ifos]:
			# iterate over the mass space chunks for each segment
			for node in inspiral_nodes[(ifos, None)][seg]:
				if node is None:
					break
				len_out_files = len(node.output_files["output-cache"])
				for f in node.output_files["output-cache"]:
					# Store the output files and the node for use as a parent dependency
					lloid_output[None].setdefault(CacheEntry.from_T050017(f).description.split("_")[0], []).append((f, [node]))
				for f in node.output_files["ranking-stat-output-cache"]:
					lloid_diststats.setdefault(CacheEntry.from_T050017(f).description.split("_")[0] ,[]).append(f)
			for inj in options.injections:
				for injnode in inspiral_nodes[(ifos, sim_tag_from_inj_file(inj))][seg]:
					if injnode is None:
						continue
					for f in injnode.output_files["output-cache"]:
						# Store the output files and the node and injnode for use as a parent dependencies
						bgbin_index = CacheEntry.from_T050017(f).description.split("_")[0]
						try:
							lloid_output[sim_tag_from_inj_file(inj)].setdefault(bgbin_index, []).append((f, lloid_output[None][bgbin_index][-1][1]+[injnode]))
						except KeyError:
							lloid_output[sim_tag_from_inj_file(inj)].setdefault(bgbin_index, []).append((f, [injnode]))

	return lloid_output, lloid_diststats


def set_up_scripts(options):
	# Make an xml integrity checker
	if options.gzip_test:
		with open("gzip_test.sh", "w") as f:
			f.write("#!/bin/bash\nsleep 60\ngzip --test $@")
		os.chmod("gzip_test.sh", stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH | stat.S_IWUSR)

	# A pre script to backup data before feeding to lossy programs
	# (e.g. clustering routines)
	with open("store_raw.sh", "w") as f:
		f.write("""#!/bin/bash
		for f in $@;do mkdir -p $(dirname $f)/raw;cp $f $(dirname $f)/raw/$(basename $f);done""")
	os.chmod("store_raw.sh", stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH | stat.S_IWUSR)


def load_reference_psd(options):
	ref_psd = lal.series.read_psd_xmldoc(ligolw_utils.load_filename(options.reference_psd, verbose = options.verbose, contenthandler = lal.series.PSDContentHandler))

	# FIXME Use machinery in inspiral_pipe.py to create reference_psd.cache
	with open('reference_psd.cache', "w") as output_cache_file:
		output_cache_file.write("%s\n" % CacheEntry.from_T050017("file://localhost%s" % os.path.abspath(options.reference_psd)))

	return ref_psd


if __name__ == "__main__":
	import doctest
	doctest.testmod()

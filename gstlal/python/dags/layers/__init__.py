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


from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
import glob
import itertools
import os
from typing import List, Tuple, Union

import htcondor

from lal.utils import CacheEntry
from ligo.segments import segment, segmentlist

from gstlal.config import Argument, Option
from gstlal.dags import util as dagutils


@dataclass
class Layer:
	executable: str
	name: str = ""
	universe: str = "vanilla"
	log_dir: str = "logs"
	retries: int = 3
	transfer_files: bool = False
	parents: Union[None, 'Layer', Iterable] = None
	requirements: dict = field(default_factory=dict)
	inputs: dict = field(default_factory=dict)
	outputs: dict = field(default_factory=dict)
	nodes: list = field(default_factory=list)
	_index_map: dict = field(default_factory=dict)

	def __post_init__(self):
		if not self.name:
			self.name = os.path.basename(self.executable)

	def config(self):
		# check that nodes are valid
		self.validate()

		# add base submit opts + requirements
		submit_options = {
			"universe": self.universe,
			"executable": dagutils.which(self.executable),
			"arguments": self._arguments(),
			**self.requirements,
		}

		# file submit opts
		if self.transfer_files:
			inputs = self._inputs()
			outputs = self._outputs()
			if inputs or outputs:
				submit_options["should_transfer_files"] = "YES"
				submit_options["when_to_transfer_output"] = "ON_SUCCESS"
				submit_options["success_exit_code"] = 0
			if inputs:
				submit_options["transfer_inputs"] = inputs
			if outputs:
				submit_options["transfer_outputs"] = outputs

		# log submit opts
		submit_options["output"] = f"{self.log_dir}/$(nodename)-$(cluster)-$(process).out"
		submit_options["error"] = f"{self.log_dir}/$(nodename)-$(cluster)-$(process).err"

		# extra boilerplate submit opts
		submit_options["notification"] = "never"

		return {
			"name": self.name,
			"submit_description": htcondor.Submit(submit_options),
			"vars": self._vars(),
			"retries": self.retries,
		}

	def append(self, node):
		for input_ in node.inputs:
			self.inputs.setdefault(input_.name, []).append(input_.argument)
		for output in node.outputs:
			self.outputs.setdefault(output.name, []).append(output.argument)
		if node.key:
			self._index_map[node.key] = len(self.nodes)
		self.nodes.append(node)

	def extend(self, nodes):
		for node in nodes:
			self.append(node)

	def __iadd__(self, nodes):
		if isinstance(nodes, Iterable):
			self.extend(nodes)
		else:
			self.append(nodes)
		return self

	def to_index(self, key):
		if key not in self._index_map:
			raise KeyError
		return self._index_map[key]

	def validate(self):
		assert self.nodes, "at least one node must be connected to this layer"

		# check arg names across nodes are equal
		args = [arg.name for arg in self.nodes[0].arguments]
		for node in self.nodes[:-1]:
			assert args == [arg.name for arg in node.arguments]

		# check input/output names across nodes are equal
		inputs = [arg.name for arg in self.nodes[0].inputs]
		for node in self.nodes[:-1]:
			assert inputs == [arg.name for arg in node.inputs]
		outputs = [arg.name for arg in self.nodes[0].outputs]
		for node in self.nodes[:-1]:
			assert outputs == [arg.name for arg in node.outputs]

	def _arguments(self):
		args = [f"$({arg.condor_name})" for arg in self.nodes[0].arguments]
		io_args = []
		io_opts = []
		for arg in itertools.chain(self.nodes[0].inputs, self.nodes[0].outputs):
			if isinstance(arg, Argument):
				io_args.append(f"$({arg.condor_name})")
			else:
				io_opts.append(f"$({arg.condor_name})")
		return " ".join(itertools.chain(args, io_opts, io_args))

	def _inputs(self):
		return ",".join([f"$(input_{arg.condor_name})" for arg in self.nodes[0].inputs])

	def _outputs(self):
		return ",".join([f"$(output_{arg.condor_name})" for arg in self.nodes[0].outputs])

	def _vars(self):
		allvars = []
		for i, node in enumerate(self.nodes):
			nodevars = {arg.condor_name: arg.vars() for arg in node.arguments}
			nodevars["nodename"] = f"{self.name}_{i:04X}"
			if node.inputs:
				nodevars.update({f"{arg.condor_name}": arg.vars() for arg in node.inputs})
				nodevars.update({f"input_{arg.condor_name}": arg.files() for arg in node.inputs})
			if node.outputs:
				nodevars.update({f"{arg.condor_name}": arg.vars() for arg in node.outputs})
				nodevars.update({f"output_{arg.condor_name}": arg.files() for arg in node.outputs})
			allvars.append(nodevars)

		return allvars


@dataclass
class Node:
	key: Union[None, Iterable] = None
	parent_keys: dict = field(default_factory=dict)
	parent_indices: dict = field(default_factory=dict)
	arguments: list = field(default_factory=list)
	inputs: list = field(default_factory=list)
	outputs: list = field(default_factory=list)

	def __post_init__(self):
		if self.parent_keys and self.parent_indices:
			raise ValueError("cannot set both 'parent_keys' and 'parent_indices'")
		if isinstance(self.arguments, Argument) or isinstance(self.arguments, Option):
			self.arguments = [self.arguments]
		if isinstance(self.inputs, Argument) or isinstance(self.inputs, Option):
			self.inputs = [self.inputs]
		if isinstance(self.outputs, Argument) or isinstance(self.outputs, Option):
			self.outputs = [self.outputs]


class DataType(Enum):
	REFERENCE_PSD = 1
	SPLIT_BANK = 2
	SVD_BANK = 3
	TRIGGERS = 4
	DIST_STATS = 5
	DIST_STAT_PDFS = 6

	def __str__(self):
		return self.name.upper()


@dataclass
class DataCache:
	name: "DataType"
	cache: list = field(default_factory=list)

	@property
	def files(self):
		return [entry.path for entry in self.cache]

	def groupby(self, *group):
		# determine groupby operation
		keyfunc = self._groupby_keyfunc(group)

		# return groups of DataCaches keyed by group
		grouped = defaultdict(list)
		for entry in self.cache:
			grouped[keyfunc(entry)].append(entry)
		return {key: DataCache(self.name, cache) for key, cache in grouped.items()}

	@staticmethod
	def _groupby_keyfunc(groups):
		if isinstance(groups, str):
			groups = [groups]

		def keyfunc(key):
			keys = []
			for group in groups:
				if group in set(("ifo", "instrument", "observatory")):
					keys.append(key.observatory)
				elif group in set(("time", "segment", "time_bin")):
					keys.append(key.segment)
				elif group in set(("bin", "svd_bin")):
					keys.append(key.description.split("_")[0])
				else:
					raise ValueError(f"{group} not a valid groupby operation")
			if len(keys) > 1:
				return tuple(keys)
			else:
				return keys[0]

		return keyfunc

	@classmethod
	def generate(cls, name, ifos, time_bins, svd_bins=None, root=None, create_dirs=True):
		if isinstance(ifos, str):
			ifos = [ifos]
		if isinstance(time_bins, segment):
			time_bins = segmentlist([time_bins])
		if svd_bins and isinstance(svd_bins, str):
			svd_bins = [svd_bins]

		cache = []
		for ifo in ifos:
			for span in time_bins:
				path = data_path(str(name).lower(), span[0], create=create_dirs)
				if svd_bins:
					for svd_bin in svd_bins:
						desc = f"{svd_bin}_GSTLAL_{str(name)}"
						filename = dagutils.T050017_filename(ifo, desc, span, ".xml.gz")
						cache.append(os.path.join(path, filename))
				else:
					desc = f"GSTLAL_{str(name)}"
					filename = dagutils.T050017_filename(ifo, desc, span, ".xml.gz")
					cache.append(os.path.join(path, filename))

		if root:
			cache = [os.path.join(root, entry) for entry in cache]
		return cls(name, [CacheEntry.from_T050017(entry) for entry in cache])

	@classmethod
	def find(cls, name, start=None, end=None, root=None, segments=None, svd_bins=None):
		cache = []
		if svd_bins:
			svd_bins = set([svd_bins]) if isinstance(svd_bins, str) else set(svd_bins)
			for svd_bin in svd_bins:
				pattern = f"*-{svd_bin}_*GSTLAL_{str(name)}*-*-*.xml.gz"
				glob_path = os.path.join(str(name).lower(), "*", pattern)
				if root:
					glob_path = os.path.join(root, glob_path)
				cache.extend(glob.glob(glob_path))
		else:
			pattern = f"*-*GSTLAL_{str(name)}*-*-*.xml.gz"
			glob_path = os.path.join(str(name).lower(), "*", pattern)
			if root:
				glob_path = os.path.join(root, glob_path)
			cache.extend(glob.glob(glob_path))

		return cls(name, [CacheEntry.from_T050017(entry) for entry in cache])


def data_path(data_name, start, create=True):
	path = os.path.join(data_name, dagutils.gps_directory(start))
	os.makedirs(path, exist_ok=True)
	return path

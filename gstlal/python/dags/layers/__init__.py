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


from collections.abc import Iterable
from dataclasses import dataclass, field
import itertools
import os
from typing import List, Tuple, Union

import htcondor

from gstlal.config import Argument
from gstlal.dags.util import which


@dataclass
class Layer:
	executable: str
	name: str = ""
	universe: str = "vanilla"
	log_dir: str = "logs"
	retries: int = 3
	base_layer: bool = False
	requirements: dict = field(default_factory=dict)
	inputs: dict = field(default_factory=dict)
	outputs: dict = field(default_factory=dict)
	nodes: list = field(default_factory=list)

	def __post_init__(self):
		if not self.name:
			self.name = os.path.basename(self.executable)

	def config(self):
		# check that nodes are valid
		self.validate()

		# add base submit opts + requirements
		submit_options = {
			"universe": self.universe,
			"executable": which(self.executable),
			"arguments": self._arguments(),
			**self.requirements,
		}

		# file submit opts
		inputs = self._inputs()
		outputs = self._outputs()
		if inputs or outputs:
			submit_options["should_transfer_files"] = "YES"
			submit_options["when_to_transfer_output"] = "ON_SUCCESS"
			submit_options["success_exit_code"] = 0
			submit_options["preserve_relative_paths"] = True
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
		args = [f"$({arg.name})" for arg in self.nodes[0].arguments]
		io_args = []
		io_opts = []
		for arg in itertools.chain(self.nodes[0].inputs, self.nodes[0].outputs):
			if isinstance(arg, Argument):
				io_args.append(f"$({arg.name})")
			else:
				io_opts.append(f"$({arg.name})")
		return " ".join(itertools.chain(args, io_opts, io_args))

	def _inputs(self):
		return ",".join([f"$(input_{arg.name})" for arg in self.nodes[0].inputs])

	def _outputs(self):
		return ",".join([f"$(output_{arg.name})" for arg in self.nodes[0].outputs])

	def _vars(self):
		allvars = []
		for i, node in enumerate(self.nodes):
			nodevars = {arg.name: arg.vars() for arg in node.arguments}
			nodevars["nodename"] = f"{self.name}_{i:04X}"
			if node.inputs:
				nodevars.update({f"{arg.name}": arg.vars() for arg in node.inputs})
				nodevars.update({f"input_{arg.name}": arg.files() for arg in node.inputs})
			if node.outputs:
				nodevars.update({f"{arg.name}": arg.vars() for arg in node.outputs})
				nodevars.update({f"output_{arg.name}": arg.files() for arg in node.outputs})
			allvars.append(nodevars)

		return allvars


@dataclass
class Node:
	arguments: list = field(default_factory=list)
	inputs: list = field(default_factory=list)
	outputs: list = field(default_factory=list)

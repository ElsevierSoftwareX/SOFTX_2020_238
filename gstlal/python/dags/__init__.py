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
import re
from typing import Tuple

import htcondor
from htcondor import dags
from htcondor.dags.node import BaseNode
import pluggy

from gstlal import plugins


class DAG(dags.DAG):
	_has_layers = False

	def __init__(self, config, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.config = config
		self._node_layers = {}
		self._layers = {}

		# register layers to DAG if needed
		if not self._has_layers:
			for layer_name, layer in self._get_registered_layers().items():
				self.register_layer(layer_name)(layer)

	def __setitem__(self, key, layer):
		if key in self._layers:
			return KeyError(f"{key} layer already added to DAG")
		self._layers[key] = layer

		# register layer and parent-child relationships
		if not layer.parents:
			self._node_layers[key] = self.layer(**layer.config())
		else:
			parents = layer.parents
			if isinstance(parents, str):
				parents = [parents]
			edge = self._get_edge_type(layer, parents[0])
			self._node_layers[key] = self._node_layers[parents[0]].child_layer(
				**layer.config(),
				edge=edge
			)
			for parent in parents[1:]:
				edge = self._get_edge_type(layer, parent)
				self._node_layers[key].add_parents(
					self._node_layers[parent],
					edge=edge
				)

	def __getitem__(self, key):
		return self._layers[key]

	def __contains__(self, key):
		return key in self._layers

	def create_log_dir(self, log_dir="logs"):
		os.makedirs(log_dir, exist_ok=True)

	def write_dag(self, filename, path=None, **kwargs):
		write_dag(self, dag_file_name=filename, dag_dir=path, **kwargs)

	def write_script(self, filename, path=None, formatter=None):
		if path:
			filename = os.path.join(path, filename)
		if not formatter:
			formatter = HexFormatter()

		# write script
		with open(filename, "w") as f:
			# traverse DAG in breadth-first order
			for layer in self.walk(dags.WalkOrder("BREADTH")):
				# grab relevant submit args, format $(arg) to {arg}
				executable = layer.submit_description['executable']
				args = layer.submit_description['arguments']
				args = re.sub(r"\$\(((\w+?))\)", r"{\1}", args)

				# evaluate vars for each node in layer, write to disk
				for idx, node_vars in enumerate(layer.vars):
					node_name = formatter.generate(layer.name, idx)
					print(f"# Job {node_name}", file=f)
					print(executable + " " + args.format(**node_vars) + "\n", file=f)

	@classmethod
	def register_layer(cls, layer_name):
		"""Register a layer to the DAG, making it callable.
		"""
		def register(func):
			def wrapped(self, *args, **kwargs):
				return func(self.config, self, *args, **kwargs)
			setattr(cls, layer_name, wrapped)
		return register

	def _get_edge_type(self, child, parent_name):
		parent = self._layers[parent_name]

		# check for any explicit parent relationship
		keys_defined = any(parent_name in node.parent_keys for node in child.nodes)
		indices_defined = any(parent_name in node.parent_indices for node in child.nodes)

		# if no explicit relationship, assume many-to-many connected
		if not keys_defined and not indices_defined:
			return dags.ManyToMany()

		# else get corresponding node edges
		indices = []
		for child_idx, child_node in enumerate(child.nodes):
			if keys_defined:
				for parent_key in child_node.parent_keys[parent_name]:
					indices.append((parent.to_index(parent_key), child_idx))
			else:
				for parent_idx in child_node.parent_indices[parent_name]:
					indices.append((parent_idx, child_idx))

		return EdgeConnector(indices)

	@classmethod
	def _get_registered_layers(cls):
		"""Get all registered DAG layers.
		"""
		# set up plugin manager
		manager = pluggy.PluginManager("gstlal")
		manager.add_hookspecs(plugins)

		# load layers
		from gstlal.dags.layers import psd
		manager.register(psd)

		# add all registered plugins to registry
		registered = {}
		for plugin_name in manager.hook.layers():
			for name, layer in plugin_name.items():
				registered[name] = layer

		return registered


class HexFormatter(dags.SimpleFormatter):
	"""A hex-based node formatter that produces names like LayerName_000C.

	"""
	def __init__(self, offset: int = 0):
		self.separator = "."
		self.index_format = "{:04X}"
		self.offset = offset

	def parse(self, node_name: str) -> Tuple[str, int]:
		layer, index = node_name.split(self.separator)
		index = int(index, 16)
		return layer, index - self.offset


class EdgeConnector(dags.BaseEdge):
	"""This edge connects individual nodes in layers given an explicit mapping.

	"""
	def __init__(self, indices):
		self.indices = indices

	def get_edges(self, parent, child, join_factory):
		for parent_idx, child_idx in self.indices:
			yield (parent_idx,), (child_idx,)


def write_dag(dag, dag_dir=None, formatter=None, **kwargs):
	if not formatter:
		formatter = HexFormatter()
	if not dag_dir:
		dag_dir = os.getcwd()
	return htcondor.dags.write_dag(dag, dag_dir, node_name_formatter=formatter, **kwargs)

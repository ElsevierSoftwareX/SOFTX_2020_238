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
from typing import Tuple

import htcondor
from htcondor import dags
from htcondor.dags.node import BaseNode
import pluggy

from gstlal import plugins


class DAG(dags.DAG):
	def __init__(self, config, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.config = config
		self._current_layer = self
		self._layers = {}

	def __setitem__(self, key, value):
		self._layers[key] = value

	def __getitem__(self, key):
		return self._layers[key]

	def write(self, filename, path=None, **kwargs):
		write_dag(self, dag_file_name=filename, dag_dir=path, **kwargs)

	@classmethod
	def register_layer(cls, layer_name):
		"""Register a layer to the DAG, making it callable.
		"""
		def register(func):
			def wrapped(self, *args, **kwargs):
				layer = func(self.config, self, *args, **kwargs)
				self[layer_name] = layer
				if layer.base_layer and not isinstance(layer, BaseNode):
					self._current_layer = self.layer(**layer.config())
				else:
					self._current_layer = self._current_layer.child_layer(**layer.config())
				return self
			setattr(cls, layer_name, wrapped)
		return register


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


def create_log_dir(log_dir="logs"):
	os.makedirs(log_dir, exist_ok=True)


def write_dag(dag, dag_dir=None, node_name_formatter=None, **kwargs):
	if not node_name_formatter:
		node_name_formatter = HexFormatter()
	if not dag_dir:
		dag_dir = os.getcwd()
	return htcondor.dags.write_dag(dag, dag_dir, node_name_formatter=node_name_formatter, **kwargs)


def _get_registered_layers():
	"""Get all registered DAG layers.
	"""
	# set up plugin manager
	manager = pluggy.PluginManager("gstlal")
	manager.add_hookspecs(plugins)
	
	# add all registered plugins to registry
	registered = {}
	for plugin_name in manager.hook.layers():
		for name, layer in plugin_name.items():
			registered[name] = layer
	
	return registered


# register layers to DAG
for layer_name, layer in _get_registered_layers().items():
	DAG.register_layer(layer_name)(layer)

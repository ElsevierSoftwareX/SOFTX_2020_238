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


import pluggy

from gstlal import plugins
from gstlal.dags import DAG as BaseDAG


class DAG(BaseDAG):
	pass


def _get_registered_layers():
	"""Get all registered DAG layers.
	"""
	# set up plugin manager
	manager = pluggy.PluginManager("gstlal")
	manager.add_hookspecs(plugins)
	
	# load layers
	from gstlal.dags.layers import inspiral
	manager.register(inspiral)

	# add all registered plugins to registry
	registered = {}
	for plugin_name in manager.hook.layers():
		for name, layer in plugin_name.items():
			registered[name] = layer
	
	return registered


# register layers to DAG
for layer_name, layer in _get_registered_layers().items():
	DAG.register_layer(layer_name)(layer)

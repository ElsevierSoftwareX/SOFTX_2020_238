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


import json

from gstlal.config import Config as BaseConfig
from gstlal.config import dotdict, replace_keys


class Config(BaseConfig):
	"""
	Hold configuration used for inspiral-specific analyzes.
	"""
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

		# section-specific options
		self.psd = dotdict(replace_keys(kwargs["psd"]))
		self.svd = dotdict(replace_keys(kwargs["svd"]))
		self.filter = dotdict(replace_keys(kwargs["filter"]))

	def load_svd_manifest(self, manifest_file):
		with open(manifest_file, "r") as f:
			self.svd.stats = json.load(f)
		self.svd.bins = self.svd.stats.keys()

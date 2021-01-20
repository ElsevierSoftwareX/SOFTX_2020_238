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
import getpass
import os
from typing import List, Tuple, Union

import yaml

from lal import LIGOTimeGPS
from ligo.lw import utils as ligolw_utils
from ligo.lw.utils import segments as ligolw_segments
from ligo.segments import segment, segmentlist, segmentlistdict

from gstlal.dags import profiles


class Config:
	"""
	Hold configuration used for analyzes.
	"""
	def __init__(self, **kwargs):
		# general options
		self.rootdir = os.getcwd()
		if isinstance(kwargs["ifos"], list):
			self.ifos = kwargs["ifos"]
		else:
			self.ifos = parse_ifo_string(kwargs["ifos"])

		# time options
		self.start = LIGOTimeGPS(kwargs["start"])
		if "stop" in kwargs:
			self.stop = LIGOTimeGPS(kwargs["stop"])
			self.duration = self.stop - self.start
		else:
			self.duration = kwargs["duration"]
			self.stop = self.start + self.duration
		self.span = segment(self.start, self.stop)

		if "segments" in kwargs:
			xmldoc = ligolw_utils.load_filename(
				kwargs["segments"],
				contenthandler=ligolw_segments.LIGOLWContentHandler
			)
			self.segments = ligolw_segments.segmenttable_get_by_name(xmldoc, "segments").coalesce()
		else:
			self.segments = segmentlistdict((ifo, segmentlist([self.span])) for ifo in self.ifos)

		# section-specific options
		self.source = dotdict(replace_keys(kwargs["source"]))

		# condor options
		condor_config = kwargs["condor"]
		if "accounting-group-user" in condor_config:
			accounting_group_user = condor_config["accounting-group-user"]
		else:
			accounting_group_user = getpass.getuser()

		condor_opts = {
			"want_graceful_removal": "True",
			"kill_sig": "15",
			"accounting_group": condor_config["accounting-group"],
			"accounting_group_user": accounting_group_user,
		}
		requirements = []

		# load site profile
		profile = profiles.load_profile(condor_config["profile"])
		assert profile["scheduler"] == "condor", "only the condor scheduler is available"

		# add profile-specific options
		if "directives" in profile:
			condor_opts.update(profile["directives"])
		if "requirements" in profile:
			requirements.extend(profile["requirements"])

		# singularity-specific options
		if "singularity-image" in condor_config:
			singularity_image = condor_config["singularity-image"]
			requirements.append("(HAS_SINGULARITY=?=True)")
			condor_opts['+SingularityImage'] = f'"{singularity_image}"'
			condor_opts['x509userproxy'] = 'x509_proxy'
			condor_opts['use_x509userproxy'] = True
			condor_opts['transfer_executable'] = False
			condor_opts['getenv'] = False

		# condor requirements
		condor_opts['requirements'] = " && ".join(requirements)

		# set up condor config
		self.condor = dotdict(replace_keys(condor_opts))

	@classmethod
	def load(cls, path):
		"""
		Load configuration from a file given a file path.
		"""
		with open(path, "r") as f:
			return cls(**yaml.safe_load(f))


@dataclass
class Argument:
	name: str
	argument: Union[str, List]

	def __post_init__(self):
		self.condor_name = self.name.replace("-", "_")

	def vars(self):
		if isinstance(self.argument, Iterable) and not isinstance(self.argument, str):
			return " ".join(self.argument)
		else:
			return f"{self.argument}"

	def files(self):
		if isinstance(self.argument, Iterable) and not isinstance(self.argument, str):
			return ",".join(self.argument)
		else:
			return f"{self.argument}"


@dataclass
class Option:
	name: str
	argument: Union[None, str, List] = None

	def __post_init__(self):
		self.condor_name = self.name.replace("-", "_")

	def vars(self):
		if not self.argument:
			return f"--{self.name}"
		elif isinstance(self.argument, Iterable) and not isinstance(self.argument, str):
			return " ".join([f"--{self.name} {arg}" for arg in self.argument])
		else:
			return f"--{self.name} {self.argument}"

	def files(self):
		if not self.argument:
			return
		elif isinstance(self.argument, Iterable) and not isinstance(self.argument, str):
			return ",".join([f"{arg}" for arg in self.argument])
		else:
			return f"{self.argument}"


class dotdict(dict):
	"""
	A dictionary supporting dot notation.

	Implementation from https://gist.github.com/miku/dc6d06ed894bc23dfd5a364b7def5ed8.

	"""
	__getattr__ = dict.get
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		for k, v in self.items():
			if isinstance(v, dict):
				self[k] = dotdict(v)


def parse_ifo_string(ifos):
	"""
	Given a string of IFO pairs (e.g. H1L1), return a list of IFOs.
	"""
	return [ifos[2*n:2*n+2] for n in range(len(ifos) // 2)]


def replace_keys(dict_):
	return {k.replace("-", "_"): v for k, v in dict_.items()}

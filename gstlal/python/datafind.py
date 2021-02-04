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
from dataclasses import dataclass, field
from enum import Enum
import glob
import math
import os

import gwdatafind
from lal.utils import CacheEntry
from ligo.segments import segment, segmentlist


DEFAULT_DATAFIND_SERVER = os.getenv('LIGO_DATAFIND_SERVER', 'ldr.ldas.cit:80')


class DataType(Enum):
	REFERENCE_PSD = (0, "xml.gz")
	MEDIAN_PSD = (1, "xml.gz")
	TRIGGERS = (2, "xml.gz")
	DIST_STATS = (3, "xml.gz")
	PRIOR_DIST_STATS = (4, "xml.gz")
	MARG_DIST_STATS = (5, "xml.gz")
	DIST_STAT_PDFS = (6, "xml.gz")
	TEMPLATE_BANK = (7, "xml.gz")
	SPLIT_BANK = (8, "xml.gz")
	SVD_BANK = (9, "xml.gz")

	def __init__(self, value, extension):
		self.extension = extension

	def __str__(self):
		return self.name.upper()

	def description(self, svd_bin=None):
		description = "GSTLAL"
		if svd_bin:
			description = f"{svd_bin}_{description}"
		return f"{description}_{str(self.name)}"

	def filename(self, ifos, span=None, svd_bin=None):
		if not span:
			span = segment(0, 0)
		return T050017_filename(ifos, self.description(svd_bin), span, self.extension)

	def file_pattern(self, svd_bin=None):
		if svd_bin:
			return f"*-{svd_bin}_{self.description()}-*-*{self.extension}"
		else:
			return f"*-*{self.description()}-*-*{self.extension}"


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
				path = cls._data_path(str(name).lower(), span[0], create=create_dirs)
				if svd_bins:
					for svd_bin in svd_bins:
						filename = name.filename(ifo, span, svd_bin)
						cache.append(os.path.join(path, filename))
				else:
					filename = name.filename(ifo, span)
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
				glob_path = os.path.join(str(name).lower(), "*", name.file_pattern(svd_bin))
				if root:
					glob_path = os.path.join(root, glob_path)
				cache.extend(glob.glob(glob_path))
		else:
			glob_path = os.path.join(str(name).lower(), "*", name.file_pattern())
			if root:
				glob_path = os.path.join(root, glob_path)
			cache.extend(glob.glob(glob_path))

		return cls(name, [CacheEntry.from_T050017(entry) for entry in cache])

	@staticmethod
	def _data_path(data_name, start, create=True):
		path = os.path.join(data_name, gps_directory(start))
		os.makedirs(path, exist_ok=True)
		return path


def load_frame_cache(start, end, frame_types, host=None):
	"""
	Given a span and a set of frame types, loads a frame cache.
	"""
	if not host:
		host = DEFAULT_DATAFIND_SERVER
	cache = []
	conn = gwdatafind.connect(host=host)
	for ifo, frame_type in frame_types.items():
		urls = conn.find_urls(ifo[0], frame_type, start, end)
		cache.extend([CacheEntry.from_T050017(url) for url in urls])

	return cache


def gps_directory(gpstime):
	"""
	Given a gps time, returns the directory name where files corresponding
	to this time will be written to, e.g. 1234567890 -> '12345'.
	"""
	return str(int(gpstime))[:5]


def T050017_filename(instruments, description, seg, extension, path=None):
	"""
	A function to generate a T050017 filename.
	"""
	if not isinstance(instruments, str):
		instruments = "".join(sorted(instruments))
	start, end = seg
	start = int(math.floor(start))
	try:
		duration = int(math.ceil(end)) - start
	# FIXME this is not a good way of handling this...
	except OverflowError:
		duration = 2000000000
	extension = extension.strip('.')
	if path is not None:
		return '%s/%s-%s-%d-%d.%s' % (path, instruments, description, start, duration, extension)
	else:
		return '%s-%s-%d-%d.%s' % (instruments, description, start, duration, extension)

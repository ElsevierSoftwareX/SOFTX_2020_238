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

import gwdatafind
from lal.utils import CacheEntry


DEFAULT_DATAFIND_SERVER = os.getenv('LIGO_DATAFIND_SERVER', 'ldr.ldas.cit:80')


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

# Copyright (C) 2013--2014  Kipp Cannon, Chad Hanna
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

import socket, copy, doctest
from ligo import segments
from glue.ligolw import lsctables, ligolw
from glue.ligolw import utils as ligolw_utils
from gstlal import svd_bank
from lal.utils import CacheEntry


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
	for ce in [CacheEntry(f) for f in open(svd_bank_cache)]:
		if not template_mchirp_dict.setdefault("%04d" % int(ce.description.split("_")[3]), []):
			min_mchirp, max_mchirp = float("inf"), 0
			xmldoc = ligolw_utils.load_url(ce.path, contenthandler = svd_bank.DefaultContentHandler)
			for root in (elem for elem in xmldoc.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") and elem.Name == "gstlal_svd_bank_Bank"):
				snglinspiraltable = lsctables.SnglInspiralTable.get_table(root)
				mchirp_column = snglinspiraltable.getColumnByName("mchirp")
				min_mchirp, max_mchirp = min(min_mchirp, min(mchirp_column)), max(max_mchirp, max(mchirp_column))
			template_mchirp_dict["%04d" % int(ce.description.split("_")[3])] = (min_mchirp, max_mchirp)
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


if __name__ == "__main__":
	import doctest
	doctest.testmod()

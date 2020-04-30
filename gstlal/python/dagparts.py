# Copyright (C) 2010  Kipp Cannon (kipp.cannon@ligo.org)
# Copyright (C) 2010 Chad Hanna (chad.hanna@ligo.org)
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 51
# Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

## @file

## @package dagparts

#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


"""
DAG construction tools.
"""


import doctest
import itertools
import math
import os
import sys
import socket
import subprocess
import tempfile

from ligo import segments

from lal.utils import CacheEntry

from gstlal import pipeline

__author__ = "Kipp Cannon <kipp.cannon@ligo.org>, Chad Hanna <chad.hanna@ligo.org>"
__date__ = "$Date$" #FIXME
__version__ = "$Revision$" #FIXME


#
# =============================================================================
#
# Environment utilities
#
# =============================================================================
#


def which(prog):
	which = subprocess.Popen(['which',prog], stdout=subprocess.PIPE)
	out = which.stdout.read().strip()
	if not out:
		raise ValueError("could not find %s in your path, have you built the proper software and sourced the proper environment scripts?" % prog)
	return out


def condor_scratch_space():
	"""!
	A way to standardize the condor scratch space even if it changes
	>>> condor_scratch_space()
	'_CONDOR_SCRATCH_DIR'
	"""
	return "_CONDOR_SCRATCH_DIR"


def log_path():
	"""!
	The stupid pet tricks to find log space on the LDG.
	Defaults to checking TMPDIR first.
	"""
	host = socket.getfqdn()
	try:
		return os.environ['TMPDIR']
	except KeyError:
		print("\n\n!!!! $TMPDIR NOT SET !!!!\n\n\tPLEASE email your admin to tell them to set $TMPDIR to be the place where a users temporary files should be\n")
		#FIXME add more hosts as you need them
		if 'cit' in host or 'caltech.edu' in host:
			tmp = '/usr1/' + os.environ['USER']
			print(f"falling back to {tmp}")
			return tmp
		if 'phys.uwm.edu' in host:
			tmp = '/localscratch/' + os.environ['USER']
			print(f"falling back to {tmp}")
			return tmp
		if 'aei.uni-hannover.de' in host:
			tmp = '/local/user/' + os.environ['USER']
			print(f"falling back to {tmp}")
			return tmp
		if 'phy.syr.edu' in host:
			tmp = '/usr1/' + os.environ['USER']
			print(f"falling back to {tmp}")
			return tmp

		raise KeyError("$TMPDIR is not set and I don't recognize this environment")


#
# =============================================================================
#
# Condor DAG utilities
#
# =============================================================================
#


class DAG(pipeline.CondorDAG):
	"""!
	A thin subclass of pipeline.CondorDAG.

	Extra features include an add_node() method and a cache writing method.
	Also includes some standard setup, e.g., log file paths etc.
	"""
	def __init__(self, name, logpath = log_path()):
		self.basename = name.replace(".dag","")
		tempfile.tempdir = logpath
		tempfile.template = self.basename + '.dag.log.'
		logfile = tempfile.mktemp()
		fh = open( logfile, "w" )
		fh.close()
		pipeline.CondorDAG.__init__(self,logfile)
		self.set_dag_file(self.basename)
		self.jobsDict = {}
		self.output_cache = []

	def add_node(self, node, retry = 3):
		node.set_retry(retry)
		node.add_macro("macronodename", node.get_name())
		pipeline.CondorDAG.add_node(self, node)

	def write_cache(self):
		out = self.basename + ".cache"
		f = open(out,"w")
		for c in self.output_cache:
			f.write(str(c)+"\n")
		f.close()


class DAGJob(pipeline.CondorDAGJob):
	"""!
	A job class that subclasses pipeline.CondorDAGJob and adds some extra
	boiler plate items for gstlal jobs which tends to do the "right" thing
	when given just an executable name.
	"""
	def __init__(self, executable, tag_base = None, universe = "vanilla", condor_commands = {}):
		self.__executable = which(executable)
		self.__universe = universe
		if tag_base:
			self.tag_base = tag_base
		else:
			self.tag_base = os.path.split(self.__executable)[1]
		self.__prog__ = self.tag_base
		pipeline.CondorDAGJob.__init__(self, self.__universe, self.__executable)
		self.add_condor_cmd('getenv','True')
		self.add_condor_cmd('environment',"GST_REGISTRY_UPDATE=no;")
		self.set_sub_file(self.tag_base+'.sub')
		self.set_stdout_file('logs/$(macronodename)-$(cluster)-$(process).out')
		self.set_stderr_file('logs/$(macronodename)-$(cluster)-$(process).err')
		self.number = 1
		# make an output directory for files
		self.output_path = self.tag_base
		try:
			os.mkdir(self.output_path)
		except:
			pass
		for cmd, val in condor_commands.items():
			self.add_condor_cmd(cmd, val)


class DAGNode(pipeline.CondorDAGNode):
	"""!
	A node class that subclasses pipeline.CondorDAGNode that automates
	adding the node to the dag, makes sensible names and allows a list of parent
	nodes to be provided.

	It tends to do the "right" thing when given a job, a dag, parent nodes, dictionary
	options relevant to the job, a dictionary of options related to input files and a
	dictionary of options related to output files.

	NOTE and important and subtle behavior - You can specify an option with
	an empty argument by setting it to "".  However options set to None are simply
	ignored.
	"""
	def __init__(self, job, dag, parent_nodes, opts = {}, input_files = {}, output_files = {}, input_cache_files = {}, output_cache_files = {}, input_cache_file_name = None):
		pipeline.CondorDAGNode.__init__(self, job)
		for p in parent_nodes:
			self.add_parent(p)
		self.set_name("%s_%04X" % (job.tag_base, job.number))
		job.number += 1
		dag.add_node(self)

		self.input_files = input_files.copy()
		self.input_files.update(input_cache_files)
		self.output_files = output_files.copy()
		self.output_files.update(output_cache_files)

		self.cache_inputs = {}
		self.cache_outputs = {}

		for opt, val in opts.items() + output_files.items() + input_files.items():
			if val is None:
				continue # not the same as val = '' which is allowed
			if not hasattr(val, "__iter__"): # catches list like things but not strings
				if opt == "":
					self.add_var_arg(val)
				else:
					self.add_var_opt(opt, val)
			# Must be an iterable
			else:
				if opt == "":
					[self.add_var_arg(a) for a in val]
				else:
					self.add_var_opt(opt, pipeline_dot_py_append_opts_hack(opt, val))

		# Create cache files for long command line arguments and store them in the job's subdirectory. NOTE the svd-bank string
		# is handled by gstlal_inspiral_pipe directly

		cache_dir = os.path.join(job.tag_base, 'cache')

		for opt, val in input_cache_files.items():
			if not os.path.isdir(cache_dir):
				os.mkdir(cache_dir)
			cache_entries = [CacheEntry.from_T050017("file://localhost%s" % os.path.abspath(filename)) for filename in val]
			if input_cache_file_name is None:
				cache_file_name = group_T050017_filename_from_T050017_files(cache_entries, '.cache', path = cache_dir)
			else:
				cache_file_name = os.path.join(cache_dir, input_cache_file_name)
			open(cache_file_name, "w").write("\n".join(map(str, cache_entries)))
			self.add_var_opt(opt, cache_file_name)
			# Keep track of the cache files being created
			self.cache_inputs.setdefault(opt, []).append(cache_file_name)

		for opt, val in output_cache_files.items():
			if not os.path.isdir(cache_dir):
				os.mkdir(cache_dir)
			cache_entries = [CacheEntry.from_T050017("file://localhost%s" % os.path.abspath(filename)) for filename in val]
			cache_file_name = group_T050017_filename_from_T050017_files(cache_entries, '.cache', path = cache_dir)
			open(cache_file_name, "w").write("\n".join(map(str, cache_entries)))
			self.add_var_opt(opt, cache_file_name)
			# Keep track of the cache files being created
			self.cache_outputs.setdefault(opt, []).append(cache_file_name)


def condor_command_dict_from_opts(opts, defaultdict = None):
	"""!
	A function to turn a list of options into a dictionary of condor commands, e.g.,

	>>> condor_command_dict_from_opts(["+Online_CBC_SVD=True", "TARGET.Online_CBC_SVD =?= True"])
	{'TARGET.Online_CBC_SVD ': '?= True', '+Online_CBC_SVD': 'True'}
	>>> condor_command_dict_from_opts(["+Online_CBC_SVD=True", "TARGET.Online_CBC_SVD =?= True"], {"somecommand":"somevalue"})
	{'somecommand': 'somevalue', 'TARGET.Online_CBC_SVD ': '?= True', '+Online_CBC_SVD': 'True'}
	>>> condor_command_dict_from_opts(["+Online_CBC_SVD=True", "TARGET.Online_CBC_SVD =?= True"], {"+Online_CBC_SVD":"False"})
	{'TARGET.Online_CBC_SVD ': '?= True', '+Online_CBC_SVD': 'True'}
	"""

	if defaultdict is None:
		defaultdict = {}
	for o in opts:
		osplit = o.split("=")
		k = osplit[0]
		v = "=".join(osplit[1:])
		defaultdict.update([(k, v)])
	return defaultdict


def pipeline_dot_py_append_opts_hack(opt, vals):
	"""!
	A way to work around the dictionary nature of pipeline.py which can
	only record options once.

	>>> pipeline_dot_py_append_opts_hack("my-favorite-option", [1,2,3])
	'1 --my-favorite-option 2 --my-favorite-option 3'
	"""
	out = str(vals[0])
	for v in vals[1:]:
		out += " --%s %s" % (opt, str(v))
	return out


#
# =============================================================================
#
# Segment utilities
#
# =============================================================================
#


def breakupseg(seg, maxextent, overlap):
	if maxextent <= 0:
		raise ValueError("maxextent must be positive, not %s" % repr(maxextent))

	# Simple case of only one segment
	if abs(seg) < maxextent:
		return segments.segmentlist([seg])

	# adjust maxextent so that segments are divided roughly equally
	maxextent = max(int(abs(seg) / (int(abs(seg)) // int(maxextent) + 1)), overlap)
	maxextent = int(math.ceil(abs(seg) / math.ceil(abs(seg) / maxextent)))
	end = seg[1]

	seglist = segments.segmentlist()


	while abs(seg):
		if (seg[0] + maxextent + overlap) < end:
			seglist.append(segments.segment(seg[0], seg[0] + maxextent + overlap))
			seg = segments.segment(seglist[-1][1] - overlap, seg[1])
		else:
			seglist.append(segments.segment(seg[0], end))
			break

	return seglist


def breakupsegs(seglist, maxextent, overlap):
	newseglist = segments.segmentlist()
	for bigseg in seglist:
		newseglist.extend(breakupseg(bigseg, maxextent, overlap))
	return newseglist


def breakupseglists(seglists, maxextent, overlap):
	for instrument, seglist in seglists.iteritems():
		newseglist = segments.segmentlist()
		for bigseg in seglist:
			newseglist.extend(breakupseg(bigseg, maxextent, overlap))
		seglists[instrument] = newseglist


#
# =============================================================================
#
# File utilities
#
# =============================================================================
#


def cache_to_instruments(cache):
	"""!
	Given a cache, returns back a string containing all the IFOs that are
	contained in each of its cache entries, sorted by IFO name.
	"""
	observatories = set()
	for cache_entry in cache:
		observatories.update(groups(cache_entry.observatory, 2))
	return ''.join(sorted(list(observatories)))


def T050017_filename(instruments, description, seg, extension, path = None):
	"""!
	A function to generate a T050017 filename.
	"""
	if not isinstance(instruments, basestring):
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


def group_T050017_filename_from_T050017_files(cache_entries, extension, path = None):
	"""!
	A function to return the name of a file created from multiple files following
	the T050017 convention. In addition to the T050017 requirements, this assumes
	that numbers relevant to organization schemes will be the first entry in the
	description, e.g. 0_DIST_STATS, and that all files in a given cache file are
	from the same group of ifos and either contain data from the same segment or
	from the same background bin.  Note, that each file doesn't have to be from
	the same IFO, for example the template bank cache could contain template bank
	files from H1 and template bank files from L1.
	"""
	# Check that every file has same observatory.
	observatories = cache_to_instruments(cache_entries)
	split_description = cache_entries[0].description.split('_')
	min_bin = [x for x in split_description[:2] if x.isdigit()]
	max_bin = [x for x in cache_entries[-1].description.split('_')[:2] if x.isdigit()]
	seg = segments.segmentlist(cache_entry.segment for cache_entry in cache_entries).extent()
	if min_bin:
		min_bin = min_bin[0]
	if max_bin:
		max_bin = max_bin[-1]
	if min_bin and (min_bin == max_bin or not max_bin):
		# All files from same bin, thus segments may be different.
		# Note that this assumes that if the last file in the cache
		# does not start with a number that every file in the cache is
		# from the same bin, an example of this is the cache file
		# generated for gstlal_inspiral_calc_likelihood, which contains
		# all of the DIST_STATS files from a given background bin and
		# then CREATE_PRIOR_DIST_STATS files which are not generated
		# for specific bins
		return T050017_filename(observatories, cache_entries[0].description, seg, extension, path = path)
	elif min_bin and max_bin and min_bin != max_bin:
		if split_description[1].isdigit():
			description_base = split_description[2:]
		else:
			description_base = split_description[1:]
		# Files from different bins, thus segments must be same
		return T050017_filename(observatories, '_'.join([min_bin, max_bin] + description_base), seg, extension, path = path)
	else:
		print("ERROR: first and last file of cache file do not match known pattern, cannot name group file under T050017 convention. \nFile 1: %s\nFile 2: %s" % (cache_entries[0].path, cache_entries[-1].path), file=sys.stderr)
		raise ValueError


#
# =============================================================================
#
# Misc utilities
#
# =============================================================================
#


def groups(l, n):
	"""!
	Given a list, returns back sublists with a maximum size n.
	"""
	for i in range(0, len(l), n):
		yield l[i:i+n]


def flatten(lst):
	"""!
	Flatten a list by one level of nesting.
	"""
	return list(itertools.chain.from_iterable(lst))


if __name__ == "__main__":
	import doctest
	doctest.testmod()

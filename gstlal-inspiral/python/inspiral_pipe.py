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

import sys, os
import subprocess, socket, tempfile, copy, doctest
from glue import pipeline, lal
from glue.ligolw import utils, lsctables, array


#
# environment utilities
#


def which(prog):
	"""!
	Like the which program to find the path to an executable

	>>> which("ls")
	'/bin/ls'

	"""
	which = subprocess.Popen(['which',prog], stdout=subprocess.PIPE)
	out = which.stdout.read().strip()
	if not out:
		print >>sys.stderr, "ERROR: could not find %s in your path, have you built the proper software and source the proper env. scripts?" % (prog,prog)
		raise ValueError
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
		print "\n\n!!!! $TMPDIR NOT SET !!!!\n\n\tPLEASE email your admin to tell them to set $TMPDIR to be the place where a users temporary files should be\n"
		#FIXME add more hosts as you need them
		if 'cit' in host or 'caltech.edu' in host:
			tmp = '/usr1/' + os.environ['USER']
			print "falling back to ", tmp
			return tmp
		if 'phys.uwm.edu' in host:
			tmp = '/localscratch/' + os.environ['USER']
			print "falling back to ", tmp
			return tmp
		if 'aei.uni-hannover.de' in host:
			tmp = '/local/user/' + os.environ['USER']
			print "falling back to ", tmp
			return tmp
		if 'phy.syr.edu' in host:
			tmp = '/usr1/' + os.environ['USER']
			print "falling back to ", tmp
			return tmp

		raise KeyError("$TMPDIR is not set and I don't recognize this environment")


#
# DAG class
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

	def add_node(self, node):
		node.set_retry(3)
		node.add_macro("macronodename", node.get_name())
		pipeline.CondorDAG.add_node(self, node)

	def write_cache(self):
		out = self.basename + ".cache"
		f = open(out,"w")
		for c in self.output_cache:
			f.write(str(c)+"\n")
		f.close()


class InspiralJob(pipeline.CondorDAGJob):
	"""!
	A job class that subclasses pipeline.CondorDAGJob and adds some extra
	boiler plate items for gstlal inspiral jobs
	"""
	def __init__(self, executable, tag_base, universe = "vanilla"):
		self.__prog__ = tag_base
		self.__executable = executable
		self.__universe = universe
		pipeline.CondorDAGJob.__init__(self, self.__universe, self.__executable)
		self.add_condor_cmd('getenv','True')
		self.add_condor_cmd('environment',"GST_REGISTRY_UPDATE=no;")
		self.tag_base = tag_base
		self.set_sub_file(tag_base+'.sub')
		self.set_stdout_file('logs/$(macronodename)-$(cluster)-$(process).out')
		self.set_stderr_file('logs/$(macronodename)-$(cluster)-$(process).err')
		self.number = 1
		# make an output directory for files
		self.output_path = tag_base
		try:
			os.mkdir(self.output_path)
		except:
			pass


class InspiralNode(pipeline.CondorDAGNode):
	"""!
	A node class that subclasses pipeline.CondorDAGNode that automates
	adding the node to the dag, makes sensible names and allows a list of parent
	nodes to be provided.
	"""
	def __init__(self, job, dag, p_node=[]):
		pipeline.CondorDAGNode.__init__(self, job)
		for p in p_node:
			self.add_parent(p)
		self.set_name("%s_%04X" % (job.tag_base, job.number))
		job.number += 1
		dag.add_node(self)


class generic_job(InspiralJob):
	"""!
	A generic job class which tends to do the "right" thing when given just
	an executable name but otherwise is a subclass of InspiralJob and thus
	pipeline.CondorDAGJob
	"""
	def __init__(self, program, tag_base = None, condor_commands = {}, **kwargs):
		executable = which(program)
		InspiralJob.__init__(self, executable, tag_base or os.path.split(executable)[1], **kwargs)
		for cmd,val in condor_commands.items():
			self.add_condor_cmd(cmd, val)


class generic_node(InspiralNode):
	"""!
	A generic node class which tends to do the "right" thing when given a
	job, a dag, parent nodes, a dictionary options relevant to the job, a
	dictionary of options related to input files and a dictionary of options
	related to output files.  Otherwise it is a subclass of InspiralNode and thus
	pipeline.CondorDAGNode

	NOTE and important and subtle behavior - You can specify an option with
	an empty argument by setting it to "".  However options set to None are simply
	ignored.
	"""
	def __init__(self, job, dag, parent_nodes, opts = {}, input_files = {}, output_files = {}, input_cache_files = {}, output_cache_files = {}):
		InspiralNode.__init__(self, job, dag, parent_nodes)

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

		for opt, val in input_cache_files.items():
			cache_entries = [lal.CacheEntry.from_T050017(url) for url in val]
			cache_file_name = "{0}/{1}_{2}.cache".format(job.tag_base, opt.replace("-cache", "").replace("-", "_"), job.number-1)
			with open(cache_file_name, "w") as cache_file:
				lal.Cache(cache_entries).tofile(cache_file)
			self.add_var_opt(opt, cache_file_name)
			# Keep track of the cache files being created
			self.cache_inputs.setdefault(opt, []).append(cache_file_name)

		for opt, val in output_cache_files.items():
			cache_entries = [lal.CacheEntry.from_T050017(url) for url in val]
			cache_file_name = "{0}/{1}_{2}.cache".format(job.tag_base, opt.replace("-cache", "").replace("-", "_"), job.number-1)
			with open(cache_file_name, "w") as cache_file:
				lal.Cache(cache_entries).tofile(cache_file)
			self.add_var_opt(opt, cache_file_name)
			# Keep track of the cache files being created
			self.cache_outputs.setdefault(opt, []).append(cache_file_name)

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
	files = zip(*[[lal.CacheEntry(f).path for f in open(cachedict[ifo],'r').readlines()] for ifo in ifos])
	for n, bank_group in enumerate(group(files, numbanks)):
		if maxjobs is not None and n > maxjobs:
			break
		c = dict(zip(ifos, zip(*bank_group)))
		outstrs.append(c)

	return outstrs


def T050017_filename(instruments, description, start, end, extension, path = None):
	"""!
	A function to generate a T050017 filename.
	"""
	if type(instruments) != type(str()):
		instruments = "".join(sorted(instruments))
	duration = end - start
	extension = extension.strip('.')
	if path is not None:
		return '%s/%s-%s-%d-%d.%s' % (path, instruments, description, start, duration, extension)
	else:
		return '%s-%s-%d-%d.%s' % (instruments, description, start, duration, extension)


if __name__ == "__main__":
	import doctest
	doctest.testmod()


def condor_command_dict_from_opts(opts, defaultdict = {}):
	"""!
	A function to turn a list of options into a dictionary of condor commands, e.g.,

	>>> condor_command_dict_from_opts(["+Online_CBC_SVD=True", "TARGET.Online_CBC_SVD =?= True"])
	{'TARGET.Online_CBC_SVD ': '?= True', '+Online_CBC_SVD': 'True'}
	>>> condor_command_dict_from_opts(["+Online_CBC_SVD=True", "TARGET.Online_CBC_SVD =?= True"], {"somecommand":"somevalue"})
	{'somecommand': 'somevalue', 'TARGET.Online_CBC_SVD ': '?= True', '+Online_CBC_SVD': 'True'}
	>>> condor_command_dict_from_opts(["+Online_CBC_SVD=True", "TARGET.Online_CBC_SVD =?= True"], {"+Online_CBC_SVD":"False"})
	{'TARGET.Online_CBC_SVD ': '?= True', '+Online_CBC_SVD': 'True'}
	"""

	for o in opts:
		osplit = o.split("=")
		k = osplit[0]
		v = "=".join(osplit[1:])
		defaultdict.update([(k, v)])
	return defaultdict

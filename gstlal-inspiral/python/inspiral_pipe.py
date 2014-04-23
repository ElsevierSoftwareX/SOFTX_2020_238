import sys, os
import subprocess, socket, tempfile, copy
from glue import pipeline, lal
from glue.ligolw import utils, lsctables, array


#
# environment utilities
#


def which(prog):
	"""!
	like the which program to find the path to an executable
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

	Extra features include and add_node method and a cache writing method.
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
		self.node_id = 0
		self.output_cache = []

	def add_node(self, node):
		node.set_retry(0)
		self.node_id += 1
		node.add_macro("macroid", self.node_id)
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
	def __init__(self, executable, tag_base):
		self.__prog__ = tag_base
		self.__executable = executable
		self.__universe = 'vanilla'
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
		InspiralJob.__init__(self, executable, tag_base or os.path.split(executable)[1])
		for cmd,val in condor_commands.items():
			self.add_condor_cmd(cmd, val)


class generic_node(InspiralNode):
	"""!
	A generic node class which tends to do the "right" thing when given a
	job, a dag, parent nodes, a dictionary options relevant to the job, a
	dictionary of options related to input files and a dictionary of options
	related to output files.  Otherwise it is a subclass of InspiralNode and thus
	pipeline.CondorDAGNode
	"""
	def __init__(self, job, dag, parent_nodes, opts = {}, input_files = {}, output_files = {}):
		InspiralNode.__init__(self, job, dag, parent_nodes)

		self.input_files = input_files
		self.output_files = output_files

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


def pipeline_dot_py_append_opts_hack(opt, vals):
	"""!
	A way to work around the dictionary nature of pipeline.py which can
	only record options once.

	>>> inspiral_pipe.pipeline_dot_py_append_opts_hack("my-favorite-option", [1,2,3])
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

	>>> bankcache = inspiral_pipe.parse_cache_str("H1=H1_split_bank.cache,L1=L1_split_bank.cache,V1=V1_split_bank.cache")
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
		if maxjobs is not None and n > maxJobs:
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

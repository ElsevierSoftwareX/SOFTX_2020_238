import sys, os
import subprocess, socket, tempfile
from glue import pipeline, lal
from glue.ligolw import utils, lsctables, array

###############################################################################
# environment utilities
###############################################################################

def which(prog):
	which = subprocess.Popen(['which',prog], stdout=subprocess.PIPE)
	out = which.stdout.read().strip()
	if not out:
		print >>sys.stderr, "ERROR: could not find %s in your path, have you built the proper software and source the proper env. scripts?" % (prog,prog)
		raise ValueError
	return out

def condor_scratch_space():
	return "_CONDOR_SCRATCH_DIR"

def log_path():
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


###############################################################################
# DAG class
###############################################################################

class DAG(pipeline.CondorDAG):

	def __init__(self, name, logpath = log_path()):
		self.basename = name
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


#
# Generic job classes
#


class InspiralJob(pipeline.CondorDAGJob):
	"""
	A generic job class for gstlal inspiral stuff
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
		self.set_stdout_file('logs/'+tag_base+'-$(macroid)-$(macronodename)-$(cluster)-$(process).out')
		self.set_stderr_file('logs/'+tag_base+'-$(macroid)-$(macronodename)-$(cluster)-$(process).err')
		self.number = 1
		# make an output directory for files
		self.output_path = tag_base
		try:
			os.mkdir(self.output_path)
		except:
			pass

class InspiralNode(pipeline.CondorDAGNode):
	"""
	A generic node class for gstlal inspiral stuff
	"""
	def __init__(self, job, dag, p_node=[]):
		pipeline.CondorDAGNode.__init__(self, job)
		for p in p_node:
			self.add_parent(p)
		self.set_name("%s_%04X" % (job.tag_base, job.number))
		job.number += 1
		dag.add_node(self)



###############################################################################
# Utility functions
###############################################################################

def num_bank_files(cachedict):
	ifo = cachedict.keys()[0]
	f = open(cachedict[ifo],'r')
	cnt = 0
	for l in f:
		cnt+=1
	f.close()
	return cnt

def parse_cache_str(instr):
	dictcache = {}
	if instr is None: return dictcache
	for c in instr.split(','):
		ifo = c.split("=")[0]
		cache = c.replace(ifo+"=","")
		dictcache[ifo] = cache
	return dictcache

def build_bank_groups(cachedict, numbanks = [2], maxjobs = None):
	numfiles = num_bank_files(cachedict)
	filedict = {}
	cnt = 0
	job = 0
	for ifo in cachedict:
		filedict[ifo] = open(cachedict[ifo],'r')

	loop = True
	outstrs = []
	while cnt < numfiles:
		job += 1
		if maxjobs is not None and job > maxjobs:
			break
		position = int(float(cnt) / numfiles * len(numbanks))
		c = {}
		for i in range(numbanks[position]):
			for ifo, f in filedict.items():
				if cnt < numfiles:
					c.setdefault(ifo, []).append(lal.CacheEntry(f.readline()).path)
				else:
					break
			cnt += 1
		outstrs.append(c)
	return outstrs


def build_bank_string(cachedict, numbanks = [2], maxjobs = None):
	numfiles = num_bank_files(cachedict)
	filedict = {}
	cnt = 0
	job = 0
	for ifo in cachedict:
		filedict[ifo] = open(cachedict[ifo],'r')

	loop = True
	outstrs = []
	outcounts = []
	while cnt < numfiles:
		job += 1
		if maxjobs is not None and job > maxjobs:
			break
		position = int(float(cnt) / numfiles * len(numbanks))
		c = ''
		for i in range(numbanks[position]):
			for ifo, f in filedict.items():
				if cnt < numfiles:
					c += '%s:%s,' % (ifo, lal.CacheEntry(f.readline()).path)
				else:
					break
			cnt += 1
		c = c.strip(',')
		outcounts.append(numbanks[position])
		outstrs.append(c)
	total_banks = sum(outcounts)
	return [(s, total_banks / outcounts[i]) for i, s in enumerate(outstrs)]

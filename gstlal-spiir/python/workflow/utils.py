
import math
import subprocess
from glue import pipeline
# copied from inspiral_pipe.T050017_filename
def T050017_filename(instruments, description, seg, extension, path = None):
	"""!
	A function to generate a T050017 filename.
	"""
	if not isinstance(instruments, basestring):
		instruments = "".join(sorted(instruments))
	start, end = seg
	start = int(math.floor(start))
	duration = int(math.ceil(end)) - start
	extension = extension.strip('.')
	if path is not None:
		return '%s/%s-%s-%d-%d.%s' % (path, instruments, description, start, duration, extension)
	else:
		return '%s-%s-%d-%d.%s' % (instruments, description, start, duration, extension)


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

def get_latest_online_output(outdir, keyword):
	ls_proc = subprocess.Popen(["ls", outdir], stdout = subprocess.PIPE, stderr = subprocess.PIPE)
	ls_out = ""
	try:
		ls_out = subprocess.check_output(["grep", keyword], stdin = ls_proc.stdout)
	except:
		print "no %s file yet" % keyword
		return
	ls_proc.wait()
	ls_fnames = ls_out.split("\n")
	return "%s/%s" % (outdir, max(ls_fnames))

def get_start_end_time(fname):
	start_time = fname.split('_')[-2]
	duration = fname.split('_')[-1].split('.')[0]
	end_time = int(start_time) + int(duration)
	return (int(start_time), end_time)


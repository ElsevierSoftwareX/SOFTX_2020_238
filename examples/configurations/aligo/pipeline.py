"""
Simple Condor DAG generation facility.
"""
__copyright__ = "Copyright 2010, Leo Singer"
__author__    = "Leo Singer <leo.singer@ligo.org>"
__all__       = ["EnvCondorJob", "makeNode", "makeDAG"]


from glue.pipeline import CondorDAGJob, CondorDAGNode, CondorDAG
from tempfile import mkstemp
import os
import os.path
import re


class EnvCondorJob(CondorDAGJob):
	def __init__(self, cmdline, outputname=None, subfilename=None):
		"""Create a job that runs an executable that is on the user's PATH.
		
		cmdline is a string containing the command to execute, and may contain
		references to environment variables and macros in the style of Condor
		submit files.
		
		By default, the name of the submit file will be taken from the name of
		the executable, which is derived from the part of the cmdline that precedes
		the first space."""
		CondorDAGJob.__init__(self, 'vanilla', '/usr/bin/env')
		cmdline = cmdline.replace('\n', ' ')
		if subfilename is None:
			subfilename = cmdline.strip().split(' ', 1)[0]
		if outputname is None:
			outputname = subfilename
		self.add_arg(cmdline)
		self.add_condor_cmd("getenv", "true")
		self.set_stderr_file('%s.err' % outputname)
		self.set_stdout_file('%s.out' % outputname)
		self.set_sub_file('%s.sub' % subfilename)


def makeNode(dag, job, name=None, parents=None, children=None, **kwargs):
	node = CondorDAGNode(job)
	# FIXME why does CondorDAGNode strip out underscores from argument names?
	node._CondorDAGNode__bad_macro_chars = re.compile(r'')
	if name is None:
		try:
			numberOfNodes = job.numberOfNodes
		except AttributeError:
			numberOfNodes = 0
		node.set_name("%s_%d" % (job.get_sub_file().rsplit(".", 1)[0], numberOfNodes))
		job.numberOfNodes = numberOfNodes + 1
	else:
		node.set_name(name)
	for key, val in kwargs.iteritems():
		node.add_macro(key, val)
	if parents is not None:
		for parent in parents:
			node.add_parent(parent)
	if children is not None:
		for child in children:
			child.add_parent(node)
	dag.add_node(node)
	return node


def makeDAG(name):
	logfile = mkstemp(prefix=os.path.basename(name), suffix='.log', dir=os.getenv('TMPDIR'))[1]
	dag = CondorDAG(logfile)
	dag.set_dag_file(name)
	return dag

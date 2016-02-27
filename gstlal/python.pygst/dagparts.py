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


import os
import sys
import socket
import subprocess
import tempfile
import math

from glue import segments
from glue import pipeline


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
		print >>sys.stderr, "ERROR: could not find %s in your path, have you built the proper software and sourced the proper environment scripts?" % (prog,prog)
		raise ValueError 
	return out


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


#
# =============================================================================
#
# Condor DAG utilities
#
# =============================================================================
#


class CondorDAG(pipeline.CondorDAG):

	def __init__(self, name, logpath = log_path()):
		self.basename = name
		fh, logfile = tempfile.mkstemp(dir = log_path(), prefix = self.basename + '.dag.log.')
		os.close(fh)
		pipeline.CondorDAG.__init__(self,logfile)
		self.set_dag_file(self.basename)
		self.jobsDict = {}
		self.node_id = 0
		self.output_cache = []

	def add_node(self, node, retry = 0):
		node.set_retry(retry)
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


class CondorDAGJob(pipeline.CondorDAGJob):
	"""
	A generic job class for gstlal stuff
	"""
	def __init__(self, executable, tag_base):
		self.__prog__ = tag_base
		self.__executable = executable
		self.__universe = 'vanilla'
		pipeline.CondorDAGJob.__init__(self, self.__universe, self.__executable)
		self.add_condor_cmd('getenv','True')
		self.tag_base = tag_base
		self.set_sub_file(tag_base+'.sub')
		self.set_stdout_file('logs/'+tag_base+'-$(macroid)-$(macronodename)-$(cluster)-$(process).out')
		self.set_stderr_file('logs/'+tag_base+'-$(macroid)-$(macronodename)-$(cluster)-$(process).err')
		self.number = 1


class CondorDAGNode(pipeline.CondorDAGNode):
	"""
	A generic node class for gstlal stuff
	"""
	def __init__(self, job, dag, p_node=[]):
		pipeline.CondorDAGNode.__init__(self, job)
		for p in p_node:
			self.add_parent(p)
		dag.add_node(self)


#
# =============================================================================
#
# Segment utilities
#
# =============================================================================
#


def breakupseg(seg, maxextent, overlap):
	if maxextent <= 0:
		raise ValueError, "maxextent must be positive, not %s" % repr(maxextent)

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

# Copyright (C) 2010  Kipp Cannon (kipp.cannon@ligo.org)
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


import math
import os
import sys
import time


from glue import iterutils
from glue import segments
from glue import segmentsUtils
from glue import pipeline
from glue.lal import CacheEntry
from pylal.datatypes import LIGOTimeGPS
from pylal import ligolw_tisi
from pylal import llwapp
from lalapps import power


__author__ = "Kipp Cannon <kipp.cannon@ligo.org>"
__date__ = "$Date$" #FIXME
__version__ = "$Revision$" #FIXME


#
# =============================================================================
#
#                                   Helpers
#
# =============================================================================
#


def get_files_per_sicluster(config_parser):
	return config_parser.getint("pipeline", "files_per_sicluster")


def get_files_per_thinca(config_parser):
	return config_parser.getint("pipeline", "files_per_thinca")


def get_files_per_inspinjfind(config_parser):
	return config_parser.getint("pipeline", "files_per_inspinjfind")


#
# =============================================================================
#
#                            DAG Node and Job Class
#
# =============================================================================
#


class InspInjJob(pipeline.CondorDAGJob):
	"""
	A lalapps_inspinj job used by the gstlal pipeline. The static
	options are read from the [lalapps_inspinj] section in the ini
	file.  The stdout and stderr from the job are directed to the logs
	directory.  The job runs in the universe specified in the ini file.
	The path to the executable is determined from the ini file.
	"""
	def __init__(self, config_parser):
		"""
		config_parser = ConfigParser object
		"""
		pipeline.CondorDAGJob.__init__(self, power.get_universe(config_parser), power.get_executable(config_parser, "lalapps_inspinj"))

		self.add_ini_opts(config_parser, "lalapps_inspinj")

		self.set_stdout_file(os.path.join(power.get_out_dir(config_parser), "lalapps_inspinj-$(macrochannelname)-$(macrogpsstarttime)-$(macrogpsendtime)-$(cluster)-$(process).out"))
		self.set_stderr_file(os.path.join(power.get_out_dir(config_parser), "lalapps_inspinj-$(macrochannelname)-$(macrogpsstarttime)-$(macrogpsendtime)-$(cluster)-$(process).err"))
		self.set_sub_file("lalapps_inspinj.sub")


class InspInjNode(pipeline.CondorDAGNode, pipeline.AnalysisNode):
	def __init__(self, job):
		pipeline.CondorDAGNode.__init__(self, job)
		pipeline.AnalysisNode.__init__(self)
		self.__usertag = None
		self.output_cache = []

	def set_user_tag(self, tag):
		self.__usertag = tag
		self.add_var_opt("user-tag", self.__usertag)

	def get_user_tag(self):
		if self.output_cache:
			raise AttributeError, "cannot change attributes after computing output cache"
		return self.__usertag

	def set_start(self, start):
		if self.output_cache:
			raise AttributeError, "cannot change attributes after computing output cache"
		self.add_var_opt("gps-start-time", start)

	def set_end(self, end):
		if self.output_cache:
			raise AttributeError, "cannot change attributes after computing output cache"
		self.add_var_opt("gps-end-time", end)

	def get_start(self):
		return self.get_opts().get("macrogpsstarttime", None)

	def get_end(self):
		return self.get_opts().get("macrogpsendtime", None)

	def get_ifo(self):
		return "H1H2L1V1"

	def get_output_cache(self):
		"""
		Returns a LAL cache of the output file name.  Calling this
		method also induces the output name to get set, so it must
		be at least once.
		"""
		if not self.output_cache:
			self.output_cache = [CacheEntry(self.get_ifo(), self.__usertag, segments.segment(LIGOTimeGPS(self.get_start()), LIGOTimeGPS(self.get_end())), "file://localhost" + os.path.abspath(self.get_output()))]
		return self.output_cache

	def get_output_files(self):
		raise NotImplementedError

	def get_output(self):
		if self._AnalysisNode__output is None:
			if None in (self.get_start(), self.get_end(), self.get_ifo(), self.__usertag):
				raise ValueError, "start time, end time, ifo, or user tag has not been set"
			seg = segments.segment(LIGOTimeGPS(self.get_start()), LIGOTimeGPS(self.get_end()))
			self.set_output("%s-INSPINJ_%s-%d-%d.xml.gz" % ("HLV", self.__usertag, int(self.get_start()), int(self.get_end()) - int(self.get_start())))
		return self._AnalysisNode__output


class GstLalInspiralJob(pipeline.CondorDAGJob, pipeline.AnalysisJob):
	"""
	A gstlal_inspiral job used by the gstlal inspiral pipeline. The
	static options are read from the [gstlal_inspiral] and
	[gstlal_inspiral_<inst>] sections in the ini file. The stdout and
	stderr from the job are directed to the logs directory.  The job
	runs in the universe specified in the ini file. The path to the
	executable is determined from the ini file.
	"""
	def __init__(self, config_parser):
		"""
		config_parser = ConfigParser object
		"""
		pipeline.CondorDAGJob.__init__(self, power.get_universe(config_parser), power.get_executable(config_parser, "gstlal_inspiral"))
		pipeline.AnalysisJob.__init__(self, config_parser)
		self.add_ini_opts(config_parser, "gstlal_inspiral")
		self.set_stdout_file(os.path.join(power.get_out_dir(config_parser), "gstlal_inspiral-$(cluster)-$(process).out"))
		self.set_stderr_file(os.path.join(power.get_out_dir(config_parser), "gstlal_inspiral-$(cluster)-$(process).err"))
		self.add_condor_cmd("getenv", "True")
		self.set_sub_file("gstlal_inspiral.sub")


class GstLalInspiralNode(pipeline.AnalysisNode):
	def __init__(self, job):
		pipeline.CondorDAGNode.__init__(self, job)
		pipeline.AnalysisNode.__init__(self)
		self.__usertag = None
		self.output_cache = []

	def set_ifo(self, instrument):
		"""
		Load additional options from the per-instrument section in
		the config file.
		"""
		if self.output_cache:
			raise AttributeError, "cannot change attributes after computing output cache"
		pipeline.AnalysisNode.set_ifo(self, instrument)
		self.add_var_opt("instrument", instrument)
		for optvalue in self.job()._AnalysisJob__cp.items("gstlal_inspiral_%s" % instrument):
			self.add_var_arg("--%s %s" % optvalue)

	def set_user_tag(self, tag):
		if self.output_cache:
			raise AttributeError, "cannot change attributes after computing output cache"
		self.__usertag = tag
		self.add_var_opt("comment", self.__usertag)

	def get_user_tag(self):
		return self.__usertag

	def get_output_cache(self):
		"""
		Returns a LAL cache of the output file name.  Calling this
		method also induces the output name to get set, so it must
		be at least once.
		"""
		if not self.output_cache:
			self.output_cache = [CacheEntry(self.get_ifo(), self.__usertag, segments.segment(LIGOTimeGPS(self.get_start()), LIGOTimeGPS(self.get_end())), "file://localhost" + os.path.abspath(self.get_output()))]
		return self.output_cache

	def get_output_files(self):
		raise NotImplementedError

	def get_output(self):
		if self._AnalysisNode__output is None:
			if None in (self.get_start(), self.get_end(), self.get_ifo(), self.__usertag):
				raise ValueError, "start time, end time, ifo, or user tag has not been set"
			seg = segments.segment(LIGOTimeGPS(self.get_start()), LIGOTimeGPS(self.get_end()))
			self.set_output("triggers/%s-INSPIRAL_%s-%d-%d.xml.gz" % (self.get_ifo(), self.__usertag, int(self.get_start()), int(self.get_end()) - int(self.get_start())))
		return self._AnalysisNode__output

	def set_injection_file(self, file):
		"""
		Set the name of the XML file from which to read a list of
		software injections.
		"""
		self.add_var_opt("injections", file)
		self.add_input_file(file)


class SiclusterJob(pipeline.CondorDAGJob):
	def __init__(self, config_parser):
		pipeline.CondorDAGJob.__init__(self, "vanilla", power.get_executable(config_parser, "ligolw_sicluster"))
		self.set_sub_file("ligolw_sicluster.sub")
		self.set_stdout_file(os.path.join(power.get_out_dir(config_parser), "ligolw_sicluster-$(cluster)-$(process).out"))
		self.set_stderr_file(os.path.join(power.get_out_dir(config_parser), "ligolw_sicluster-$(cluster)-$(process).err"))
		self.add_condor_cmd("getenv", "True")
		self.add_condor_cmd("Requirements", "Memory > 1100")
		self.add_ini_opts(config_parser, "ligolw_sicluster")


class SiclusterNode(pipeline.CondorDAGNode):
	def __init__(self, *args):
		pipeline.CondorDAGNode.__init__(self, *args)
		self.input_cache = []
		self.output_cache = self.input_cache

	def set_name(self, *args):
		pipeline.CondorDAGNode.set_name(self, *args)
		self.cache_name = os.path.join(self._CondorDAGNode__job.cache_dir, "%s.cache" % self.get_name())
		self.add_var_opt("input-cache", self.cache_name)

	def add_input_cache(self, cache):
		self.input_cache.extend(cache)

	def add_file_arg(self, filename):
		raise NotImplementedError

	def write_input_files(self, *args):
		f = file(self.cache_name, "w")
		for c in self.input_cache:
			print >>f, str(c)
		pipeline.CondorDAGNode.write_input_files(self, *args)

	def get_input_cache(self):
		return self.input_cache

	def get_output_cache(self):
		return self.output_cache

	def get_output_files(self):
		raise NotImplementedError

	def get_output(self):
		raise NotImplementedError


class InspinjfindJob(pipeline.CondorDAGJob):
	def __init__(self, config_parser):
		pipeline.CondorDAGJob.__init__(self, "vanilla", power.get_executable(config_parser, "ligolw_inspinjfind"))
		self.set_sub_file("ligolw_inspinjfind.sub")
		self.set_stdout_file(os.path.join(power.get_out_dir(config_parser), "ligolw_inspinjfind-$(cluster)-$(process).out"))
		self.set_stderr_file(os.path.join(power.get_out_dir(config_parser), "ligolw_inspinjfind-$(cluster)-$(process).err"))
		self.add_condor_cmd("getenv", "True")
		self.add_ini_opts(config_parser, "ligolw_inspinjfind")


class InspinjfindNode(pipeline.CondorDAGNode):
	def __init__(self, *args):
		pipeline.CondorDAGNode.__init__(self, *args)
		self.input_cache = []
		self.output_cache = self.input_cache

	def add_input_cache(self, cache):
		self.input_cache.extend(cache)
		for c in cache:
			filename = c.path()
			pipeline.CondorDAGNode.add_file_arg(self, filename)
			self.add_output_file(filename)

	def add_file_arg(self, filename):
		raise NotImplementedError

	def get_input_cache(self):
		return self.input_cache

	def get_output_cache(self):
		return self.output_cache

	def get_output_files(self):
		raise NotImplementedError

	def get_output(self):
		raise NotImplementedError


class ThincaJob(pipeline.CondorDAGJob):
	def __init__(self, config_parser):
		pipeline.CondorDAGJob.__init__(self, "vanilla", power.get_executable(config_parser, "ligolw_thinca"))
		self.set_sub_file("ligolw_thinca.sub")
		self.set_stdout_file(os.path.join(power.get_out_dir(config_parser), "ligolw_thinca-$(cluster)-$(process).out"))
		self.set_stderr_file(os.path.join(power.get_out_dir(config_parser), "ligolw_thinca-$(cluster)-$(process).err"))
		self.add_condor_cmd("getenv", "True")
		self.add_condor_cmd("Requirements", "Memory >= $(macrominram)")
		self.add_ini_opts(config_parser, "ligolw_thinca")


class ThincaNode(pipeline.CondorDAGNode):
	def __init__(self, *args):
		pipeline.CondorDAGNode.__init__(self, *args)
		self.input_cache = []
		self.output_cache = self.input_cache

	def add_input_cache(self, cache):
		self.input_cache.extend(cache)
		for c in cache:
			filename = c.path()
			pipeline.CondorDAGNode.add_file_arg(self, filename)
			self.add_output_file(filename)
		longest_duration = max([abs(cache_entry.segment) for cache_entry in self.input_cache])
		if longest_duration > 25000:
			# ask for >= 1300 MB
			self.add_macro("macrominram", 1300)
		elif longest_duration > 10000:
			# ask for >= 800 MB
			self.add_macro("macrominram", 800)
		else:
			# run on any node
			self.add_macro("macrominram", 0)

	def add_file_arg(self, filename):
		raise NotImplementedError

	def get_input_cache(self):
		return self.input_cache

	def get_output_cache(self):
		return self.output_cache

	def get_output_files(self):
		raise NotImplementedError

	def get_output(self):
		raise NotImplementedError


class RunSqliteJob(pipeline.CondorDAGJob):
	"""
        A lalapps_run_sqlite job used by the gstlal pipeline. The static
        options are read from the [lalapps_run_sqlite] section in the ini
        file.  The stdout and stderr from the job are directed to the logs
        directory.  The job runs in the universe specified in the ini file.
        The path to the executable is determined from the ini file.
	"""
        def __init__(self, config_parser):
                """
                config_parser = ConfigParser object
                """
                pipeline.CondorDAGJob.__init__(self, power.get_universe(config_parser), power.get_executable(config_parser, "lalapps_run_sqlite"))
                self.add_ini_opts(config_parser, "lalapps_run_sqlite")
                self.set_stdout_file(os.path.join(power.get_out_dir(config_parser), "lalapps_run_sqlite-$(cluster)-$(process).out"))
                self.set_stderr_file(os.path.join(power.get_out_dir(config_parser), "lalapps_run_sqlite-$(cluster)-$(process).err"))
		self.add_condor_cmd("getenv", "True")
                self.set_sub_file("lalapps_run_sqlite.sub")


class RunSqliteNode(pipeline.CondorDAGNode):
        def __init__(self, *args):
                pipeline.CondorDAGNode.__init__(self, *args)
		self.input_cache = []
		self.output_cache = self.input_cache

	def add_input_cache(self, cache):
		self.input_cache.extend(cache)

	def get_output_cache(self):
		return self.output_cache	


class SqliteToXMLJob(pipeline.CondorDAGJob):
        """
        A ligolw_sqlite job used by the gstlal pipeline. The static
        options are read from the [ligolw_sqlite] section in the ini
        file.  The stdout and stderr from the job are directed to the logs
        directory.  The job runs in the universe specified in the ini file.
        The path to the executable is determined from the ini file.
        """
        def __init__(self, config_parser):
                """
                config_parser = ConfigParser object
                """
                pipeline.CondorDAGJob.__init__(self, power.get_universe(config_parser), power.get_executable(config_parser, "ligolw_sqlite"))
                self.add_ini_opts(config_parser, "sqlitetoxml")
                self.set_stdout_file(os.path.join(power.get_out_dir(config_parser), "ligolw_sqlite-$(cluster)-$(process).out"))
                self.set_stderr_file(os.path.join(power.get_out_dir(config_parser), "ligolw_sqlite-$(cluster)-$(process).err"))
                self.add_condor_cmd("getenv", "True")
                self.set_sub_file("ligolw_sqlitetoxml.sub")


class SqliteToXMLNode(pipeline.CondorDAGNode):
        def __init__(self, *args):
                pipeline.CondorDAGNode.__init__(self, *args)
                self.input_cache = []
                self.output_cache = []

        def add_input_cache(self, cache):
                if self.output_cache:
                        raise AttributeError, "cannot change attributes after computing output cache"
                self.input_cache.extend(cache)

        def add_file_arg(self, filename):
                raise NotImplementedError

        def set_output(self, filename):
                if self.output_cache:
                        raise AttributeError, "cannot change attributes after computing output cache"
                self.add_macro("macrodatabase", filename)
		self.add_macro("macroextract", filename.replace(".sqlite", ".xml.gz"))
	
        def get_input_cache(self):
                return self.input_cache

        def get_output_cache(self):
                if not self.output_cache:
                        self.output_cache = [power.make_cache_entry(self.input_cache, None, self.get_opts()["macroextract"])]
                return self.output_cache

        def get_output_files(self):
                raise NotImplementedError

        def get_output(self):
                raise NotImplementedError


#
# =============================================================================
#
#                                DAG Job Types
#
# =============================================================================
#


#
# This is *SUCH* a hack I don't know where to begin.  Please, shoot me.
#


inspinjjob = None
gstlalinspiraljob = None
inspinjfindjob = None
siclusterjob = None
thincajob = None


def init_job_types(config_parser, job_types = ("inspinj", "gstlalinspiral", "inspinjfind", "sicluster", "thinca", "runsqlite", "sqlitetoxml")):
	"""
	Construct definitions of the submit files.
	"""
	global inspinjjob, gstlalinspiraljob, inspinjfindjob, siclusterjob, thincajob, runsqlitejob, sqlitetoxmljob

	# lalapps_binj
	if "inspinj" in job_types:
		inspinjjob = InspInjJob(config_parser)

	# gstlal_inspiral
	if "gstlalinspiral" in job_types:
		gstlalinspiraljob = GstLalInspiralJob(config_parser)

	# ligolw_inspbinjfind
	if "inspinjfind" in job_types:
		inspinjfindjob = InspinjfindJob(config_parser)
		inspinjfindjob.files_per_inspinjfind = get_files_per_inspinjfind(config_parser)
		if inspinjfindjob.files_per_inspinjfind < 1:
			raise ValueError, "files_per_inspinjfind < 1"

	# ligolw_sicluster
	if "sicluster" in job_types:
		siclusterjob = SiclusterJob(config_parser)
		siclusterjob.files_per_sicluster = get_files_per_sicluster(config_parser)
		if siclusterjob.files_per_sicluster < 1:
			raise ValueError, "files_per_sicluster < 1"
		siclusterjob.cache_dir = power.get_cache_dir(config_parser)

	# ligolw_thinca
	if "thinca" in job_types:
		thincajob = ThincaJob(config_parser)
		thincajob.files_per_thinca = get_files_per_thinca(config_parser)
		if thincajob.files_per_thinca < 1:
			raise ValueError, "files_per_thinca < 1"
	
	# lalapps_run_sqlite
	if "runsqlite" in job_types:
		runsqlitejob = RunSqliteJob(config_parser)

	# ligolw_sqlite
	if "sqlitetoxml" in job_types:
		sqlitetoxmljob = SqliteToXMLJob(config_parser)


#
# =============================================================================
#
#                                 Segmentation
#
# =============================================================================
#


def remove_too_short_segments(seglistdict, min_segment_length):
	"""
	Remove segments from seglistdict that are too short to analyze.

	CAUTION:  this function modifies seglistdict in place.
	"""
	for seglist in seglistdict.values():
		iterutils.inplace_filter(lambda seg: abs(seg) >= min_segment_length, seglist)


def compute_segment_lists(seglists, offset_vectors, min_segment_length, verbose = False):
	seglists = seglists.copy()

	# cull too-short single-instrument segments from the input
	# segmentlist dictionary;  this can significantly increase the
	# speed of the llwapp.get_coincident_segmentlistdict() function
	# when the input segmentlists have had many data quality holes
	# poked out of them
	remove_too_short_segments(seglists, min_segment_length)

	# extract the segments that are coincident under the time slides
	seglists = llwapp.get_coincident_segmentlistdict(seglists, ligolw_tisi.time_slide_component_vectors(offset_vectors, 2))

	# again remove too-short segments
	remove_too_short_segments(seglists, min_segment_length)

	# done
	return seglists


#
# =============================================================================
#
#                            Single Node Fragments
#
# =============================================================================
#


def make_gstlalinspiral_fragment(dag, parents, instrument, seg, tag, framecache, injargs = {}):
	node = GstLalInspiralNode(gstlalinspiraljob)
	node.set_name("gstlal_inspiral_%s_%s_%d_%d" % (tag, instrument, int(seg[0]), int(abs(seg))))
	map(node.add_parent, parents)
	# FIXME:  GstLalInspiralNode should not be subclassed from
	# AnalysisNode, because that class is too hard-coded.  For example,
	# there is no way to switch to analysing gaussian noise except to
	# comment out this line in the code.
	node.set_cache(framecache)
	node.set_ifo(instrument)
	node.set_start(seg[0])
	node.set_end(seg[1])
	node.set_user_tag(tag)
	for arg, value in injargs.iteritems():
		# this is a hack, but I can't be bothered
		node.add_var_arg("--%s %s" % (arg, value))
	dag.add_node(node)
	return set([node])


def make_inspinj_fragment(dag, seg, tag, offset):
	# FIXME:  this function is still broken

	# one injection every time-step / pi seconds
	period = float(inspinjjob.get_opts()["time-step"]) / math.pi

	# adjust start time to be commensurate with injection period
	start = seg[0] - seg[0] % period + period * offset

	node = InspInjNode(inspinjjob)
	node.set_start(start)
	node.set_end(seg[1])
	node.set_name("lalapps_inspinj_%d" % int(start))
	node.set_user_tag(tag)
	node.add_macro("macroseed", int(time.time() + start))
	dag.add_node(node)
	return set([node])


def make_inspinjfind_fragment(dag, parents, tag, verbose = False):
	input_cache = power.collect_output_caches(parents)
	nodes = set()
	while input_cache:
		node = InspinjfindNode(inspinjfindjob)
		node.add_input_cache([cache_entry for (cache_entry, parent) in input_cache[:inspinjfindjob.files_per_inspinjfind]])
		for cache_entry, parent in input_cache[:inspinjfindjob.files_per_inspinjfind]:
			node.add_parent(parent)
		del input_cache[:inspinjfindjob.files_per_inspinjfind]
		seg = power.cache_span(node.get_input_cache())
		node.set_name("ligolw_inspinjfind_%s_%d_%d" % (tag, int(seg[0]), int(abs(seg))))
		node.add_macro("macrocomment", tag)
		dag.add_node(node)
		nodes.add(node)
	return nodes


def make_sicluster_fragment(dag, parents, tag, verbose = False):
	input_cache = power.collect_output_caches(parents)
	nodes = set()
	while input_cache:
		node = SiclusterNode(siclusterjob)
		node.add_input_cache([cache_entry for (cache_entry, parent) in input_cache[:siclusterjob.files_per_sicluster]])
		for cache_entry, parent in input_cache[:siclusterjob.files_per_sicluster]:
			node.add_parent(parent)
		del input_cache[:siclusterjob.files_per_sicluster]
		seg = power.cache_span(node.get_input_cache())
		node.set_name("ligolw_sicluster_%s_%d_%d" % (tag, int(seg[0]), int(abs(seg))))
		node.add_macro("macrocomment", tag)
		node.set_retry(3)
		dag.add_node(node)
		nodes.add(node)
	return nodes


def make_thinca_fragment(dag, parents, tag, verbose = False):
	input_cache = power.collect_output_caches(parents)
	nodes = set()
	while input_cache:
		node = ThincaNode(thincajob)
		node.add_input_cache([cache_entry for (cache_entry, parent) in input_cache[:thincajob.files_per_thinca]])
		for cache_entry, parent in input_cache[:thincajob.files_per_thinca]:
			node.add_parent(parent)
		del input_cache[:thincajob.files_per_thinca]
		seg = power.cache_span(node.get_input_cache())
		node.set_name("ligolw_thinca_%s_%d_%d" % (tag, int(seg[0]), int(abs(seg))))
		node.add_macro("macrocomment", tag)
		dag.add_node(node)
		nodes.add(node)
	return nodes


def make_runsqlite_fragment(dag, parents, tag, verbose = False):
	input_cache = power.collect_output_caches(parents)
        nodes = set()
        for cache_entry, parent in input_cache:
                node = RunSqliteNode(runsqlitejob)
		node.add_input_cache([cache_entry])
                node.add_parent(parent)
                node.set_name("lalapps_run_sqlite_%s_%d" % (tag, len(nodes)))
		[node.add_file_arg(f.path()) for f in parent.get_output_cache()]
                dag.add_node(node)
                nodes.add(node)
        return nodes


def make_sqlitetoxml_fragment(dag, parents, tag, verbose = False):
        input_cache = power.collect_output_caches(parents)
        nodes = set()
        for cache_entry, parent in input_cache:
                node = SqliteToXMLNode(sqlitetoxmljob)
                node.add_input_cache([cache_entry])
                node.add_parent(parent)
                node.set_name("ligolw_sqlitetoxml_%s_%d" % (tag, len(nodes)))
                node.set_output(cache_entry.path())
                dag.add_node(node)
                nodes.add(node)
        return nodes


def make_thinca_fragment_maxextent(dag, parents, tag, verbose = False):
	input_cache = power.collect_output_caches(parents)
	nodes = set()
	for i, (cache,parent) in enumerate(input_cache):
		node = ThincaNode(thincajob)
		seg = cache.segment
		if i > 0 and not seg.disjoint(input_cache[i-1][0].segment):
			lo = input_cache[i-1][0].segment[1]
		else:
			lo = segments.NegInfinity
		if i < len(input_cache) - 1 and not seg.disjoint(input_cache[i+1][0].segment):
			hi = input_cache[i+1][0].segment[0]
		else:
			hi = segments.PosInfinity
		node.add_var_opt("coinc-end-time-segment",segmentsUtils.to_range_strings(segments.segmentlist([segments.segment(lo, hi)]))[0])
		node.add_input_cache([cache])
		node.add_parent(parent)
		seg = power.cache_span(node.get_input_cache())
		node.set_name("ligolw_thinca_%s_%d_%d" % (tag, int(seg[0]), int(abs(seg))))
		node.add_macro("macrocomment", tag)
		dag.add_node(node)
		nodes.add(node)
	return nodes

#
# =============================================================================
#
#       DAG Fragment Combining Multiple lalapps_inspinj With ligolw_add
#
# =============================================================================
#


def make_multiinspinj_fragment(dag, seg, tag):
	# FIXME: this function still hasn't been ported
	flow = float(powerjob.get_opts()["low-freq-cutoff"])
	fhigh = flow + float(powerjob.get_opts()["bandwidth"])

	nodes = make_inspinj_fragment(dag, seg, tag, 0.0, flow, fhigh)
	return make_lladd_fragment(dag, nodes, tag)


#
# =============================================================================
#
#       Analyze All Segments in a segmentlistdict Using gstlal_inspiral
#
# =============================================================================
#


#
# one segment
#


def make_gstlalinspiral_segment_fragment(dag, datafindnodes, instrument, segment, tag, inspinjnodes = set(), verbose = False):
	"""
	Construct a DAG fragment for an entire segment.
	"""
	# only one frame cache file can be provided as input, and only one
	# injection description file can be provided as input
	# the unpacking indirectly tests that the file count is correct
	[framecache] = [node.get_output() for node in datafindnodes]
	if inspinjnodes:
		[simfile] = [cache_entry.path() for node in inspinjnodes for cache_entry in node.get_output_cache()]
		injargs = {"injections": simfile}
	else:
		injargs = {}
	if verbose:
		print >>sys.stderr, "Segment: " + str(segment)
	return make_gstlalinspiral_fragment(dag, datafindnodes | inspinjnodes, instrument, segment, tag, framecache, injargs = injargs)


#
# all segments
#


def make_single_instrument_stage(dag, datafinds, seglistdict, tag, inspinjnodes = set(), verbose = False):
	nodes = []
	for instrument, seglist in seglistdict.iteritems():
		for seg in seglist:
			if verbose:
				print >>sys.stderr, "generating %s fragment %s" % (instrument, str(seg))

			# find the datafind job this job is going to need
			dfnodes = set([node for node in datafinds if (node.get_ifo() == instrument) and (seg in segments.segment(node.get_start(), node.get_end()))])
			if len(dfnodes) != 1:
				raise ValueError, "error, not exactly 1 datafind is suitable for trigger generator job at %s in %s" % (str(seg), instrument)

			# trigger generator jobs
			nodes += make_gstlalinspiral_segment_fragment(dag, dfnodes, instrument, seg, tag, inspinjnodes = inspinjnodes, verbose = verbose)

	# done
	return nodes


def breakupseg(seg, maxextent, overlap):
	if maxextent <= 0:
		raise ValueError, "maxextent must be positive, not %s" % repr(maxextent)

	seglist = segments.segmentlist()

	while abs(seg) > maxextent:
		seglist.append(segments.segment(seg[0], seg[0] + maxextent))
		seg = segments.segment(seglist[-1][1] - overlap, seg[1])

	seglist.append(seg)

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

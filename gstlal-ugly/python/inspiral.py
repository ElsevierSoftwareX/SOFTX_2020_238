#
# Copyright (C) 2009-2011  Kipp Cannon, Chad Hanna, Drew Keppel
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

from gstlal import svd_bank

def channel_dict_from_channel_list(channel_list):
	"""
	given a list of channels like this ["H1=LSC-STRAIN",
	H2="SOMETHING-ELSE"] produce a dictionary keyed by ifo of channel
	names.  The default values are LSC-STRAIN for all detectors
	"""

	channel_dict = {"H1" : "LSC-STRAIN", "H2" : "LSC-STRAIN", "L1" : "LSC-STRAIN", "V1" : "LSC-STRAIN", "G1" : "LSC-STRAIN", "T1" : "LSC-STRAIN"}

	for channel in channel_list:
		ifo = channel.split("=")[0]
		chan = "".join(channel.split("=")[1:])
		channel_dict[ifo] = chan

	return channel_dict


def pipeline_channel_list_from_channel_dict(channel_dict):
	"""
	produce a string of channel name arguments suitable for a pipeline.py program that doesn't technically allow multiple options. For example --channel-name=H1=LSC-STRAIN --channel-name=H2=LSC-STRAIN
	"""

	outstr = ""
	for i, ifo in enumerate(channel_dict):
		if i == 0:
			outstr += "%s=%s " % (ifo, channel_dict[ifo])
		else:
			outstr += "--channel-name=%s=%s " % (ifo, channel_dict[ifo])

	return outstr


def parse_banks(bank_string):
	"""
	parses strings of form H1:bank1.xml,H2:bank2.xml,L1:bank3.xml,H2:bank4.xml,...
	"""
	out = {}
	if bank_string is None:
		return out
	for b in bank_string.split(','):
		ifo, bank = b.split(':')
		out.setdefault(ifo, []).append(bank)
	return out

def parse_bank_files(svd_banks, verbose):
	"""
	given a dictionary of lists svd template bank file names parse them
	into a dictionary of bakn classes
	"""

	banks = {}

	for instrument, files in svd_banks.items():
		for n, filename in enumerate(files):
			# FIXME over ride the file name stored in the bank file with
			# this file name this bank I/O code needs to be fixed
			bank = svd_bank.read_bank(filename, verbose = verbose)
			bank.template_bank_filename = filename
			bank.logname = "%sbank%d" % (instrument,n)
			bank.number = n
			banks.setdefault(instrument,[]).append(bank)

	return banks


def connect_appsink_dump_dot(pipeline, appsinks, basename, verbose = False):
	
	"""
	add a signal handler to write a pipeline graph upon receipt of the
	first trigger buffer.  the caps in the pipeline graph are not fully
	negotiated until data comes out the end, so this version of the graph
	shows the final formats on all links
	"""

	class AppsinkDumpDot(object):
		# data shared by all instances
		# number of times execute method has been invoked, and a mutex
		n_lock = threading.Lock()
		n = 0

		def __init__(self, pipeline, write_after, basename, verbose = False):
			self.pipeline = pipeline
			self.handler_id = None
			self.write_after = write_after
			self.filestem = "%s.%s" % (basename, "TRIGGERS")
			self.verbose = verbose

		def execute(self, elem):
			self.n_lock.acquire()
			type(self).n += 1
			if self.n >= self.write_after:
				pipeparts.write_dump_dot(self.pipeline, self.filestem, verbose = self.verbose)
			self.n_lock.release()
			elem.disconnect(self.handler_id)

	for sink in appsinks:
		appsink_dump_dot = AppsinkDumpDot(pipeline, len(appsinks), basename = basename, verbose = verbose)
		appsink_dump_dot.handler_id = sink.connect_after("new-buffer", appsink_dump_dot.execute)



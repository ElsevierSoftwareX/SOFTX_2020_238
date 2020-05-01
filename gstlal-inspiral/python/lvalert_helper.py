#!/usr/bin/env python
#
# Copyright (C) 2013  Kipp Cannon
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

## @file gstlal_inspiral_lvalert_psd_plotter
# A program to listen to lvalerts, download the psd from gstlal gracedb events, plot it, and upload the results
#
# ### Command line interface
#
#	+ `--no-upload`: Write plots to disk.
#	+ `--skip-404`: Skip events that give 404 (file not found) errors (default is to abort).
#	+ `--verbose`: Be verbose.
#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


import http.client
import logging
import os.path
import time
import io

from glue.ligolw import ligolw
from glue.ligolw import array as ligolw_array
from glue.ligolw import param as ligolw_param
from glue.ligolw import lsctables
from glue.ligolw import utils as ligolw_utils


class LIGOLWContentHandler(ligolw.LIGOLWContentHandler):
	pass
ligolw_array.use_in(LIGOLWContentHandler)
ligolw_param.use_in(LIGOLWContentHandler)
lsctables.use_in(LIGOLWContentHandler)


def get_filename(gracedb_client, graceid, filename, retries = 3, retry_delay = 10.0, ignore_404 = False):
	for i in range(retries):
		logging.info("retrieving \"%s\" for %s" % (filename, graceid))
		response = gracedb_client.files(graceid, filename)
		if response.status == http.client.OK:
			return response
		if response.status == http.client.NOT_FOUND and ignore_404:
			logging.warning("retrieving \"%s\" for %s: (%d) %s.  skipping ..." % (filename, graceid, response.status, response.reason))
			return None
		logging.warning("retrieving \"%s\" for %s: (%d) %s.  pausing ..." % (filename, graceid, response.status, response.reason))
		time.sleep(retry_delay)
	raise Exception("retrieving \"%s\" for %s: (%d) %s" % (filename, graceid, response.status, response.reason))


def get_coinc_xmldoc(gracedb_client, graceid, filename = "coinc.xml"):
	return ligolw_utils.load_fileobj(get_filename(gracedb_client, graceid, filename = filename), contenthandler = LIGOLWContentHandler)[0]


def upload_fig(fig, gracedb_client, graceid, filename, log_message, tagname = "psd"):
	plotfile = io.StringIO()
	fig.savefig(plotfile, format = os.path.splitext(filename)[-1][1:])
	logging.info("uploading \"%s\" for %s" % (filename, graceid))
	response = gracedb_client.writeLog(graceid, log_message, filename = filename, filecontents = plotfile.getvalue(), tagname = tagname)
	if response.status != http.client.CREATED:
		raise Exception("upload of \"%s\" for %s failed: %s" % (filename, graceid, response["error"]))


def upload_file(gracedb_client, graceid, filename, log_message = "A file", tagname = None):
	logging.info("uploading \"%s\" for %s" % (filename, graceid))
	response = gracedb_client.writeLog(graceid, log_message, filename = filename, filecontents = io.FileIO(filename).readall(), tagname = tagname)
	if response.status != http.client.CREATED:
		raise Exception("upload of \"%s\" for %s failed: %s" % (filename, graceid, response["error"]))


def upload_xmldoc(gracedb_client, graceid, filename, xmldoc, log_message = "A file", tagname = None):
	logging.info("uploading \"%s\" for %s" % (filename, graceid))
	output = io.StringIO()
	ligolw_utils.write_fileobj(xmldoc, output, gz = filename.endswith(".gz"))
	response = gracedb_client.writeLog(graceid, log_message, filename = filename, filecontents = output.getvalue(), tagname = tagname)
	output.close()
	if response.status != http.client.CREATED:
		raise Exception("upload of \"%s\" for %s failed: %s" % (filename, graceid, response["error"]))

#
# Copyright (C) 2011--2013 Kipp Cannon, Chad Hanna, Drew Keppel
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

### Compute FAR and FAP distributions from the likelihood CCDFs.

#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


from optparse import OptionParser
import sqlite3
import sys


from ligo.lw import dbtables
from ligo.lw import lsctables
from ligo.lw import ligolw
from ligo.lw import utils as ligolw_utils
from ligo.lw.utils import process as ligolw_process


from lal.utils import CacheEntry
from lalburst import burca
from lalburst import stringutils


#
# =============================================================================
#
#                                 Command Line
#
# =============================================================================
#


def parse_command_line():
	parser = OptionParser()
	parser.add_option("--rankingstatpdf-file", metavar = "filename", help = "Set the name of the xml file containing the marginalized likelihood.")
	parser.add_option("-c", "--input-cache", metavar = "filename", help = "Also process the files named in this LAL cache.  See lalapps_path2cache for information on how to produce a LAL cache file.")
	parser.add_option("--non-injection-db", metavar = "filename", default = [], action = "append", help = "Provide the name of a database from a non-injection run.  Can be given multiple times.")
	parser.add_option("--tmp-space", metavar = "dir", help = "Set the name of the tmp space if working with sqlite.")
	parser.add_option("--verbose", "-v", action = "store_true", help = "Be verbose.")
	options, filenames = parser.parse_args()

	process_params = options.__dict__.copy()

	if options.input_cache:
		filenames += [CacheEntry(line).path for line in open(options.input_cache)]
	if not filenames:
		raise ValueError("no candidate databases specified")

	return options, process_params, filenames


#
# =============================================================================
#
#                                     Main
#
# =============================================================================
#


#
# Parse command line
#


options, process_params, filenames = parse_command_line()


#
# Construct likelihood distribution data from input files
#


rankingstatpdf = stringutils.marginalize_rankingstatpdf([options.rankingstatpdf_file], verbose = options.verbose)


#
# Histogram zero-lag likelihood ratios
#


if options.verbose:
	print >>sys.stderr, "beginning count of above-threshold events"

rankingstatpdf.candidates_lr_lnpdf.array[:] = 0.

for n, filename in enumerate(options.non_injection_db, start = 1):
	if options.verbose:
		print >>sys.stderr, "%d/%d: %s" % (n, len(options.non_injection_db), filename)
	working_filename = dbtables.get_connection_filename(filename, tmp_path = None, verbose = options.verbose)
	connection = sqlite3.connect(working_filename)
	
	#
	# update counts
	#

	xmldoc = dbtables.get_xml(connection)
	coinc_def_id = lsctables.CoincDefTable.get_table(xmldoc).get_coinc_def_id(burca.StringCuspBBCoincDef.search, burca.StringCuspBBCoincDef.search_coinc_type, create_new = False) 
	xmldoc.unlink()
	rankingstatpdf.collect_zero_lag_rates(connection, coinc_def_id)

	#
	# done
	#

	connection.close()
	dbtables.discard_connection_filename(filename, working_filename, verbose = options.verbose)


#
# Apply density estimation to zero-lag rates
#


rankingstatpdf.density_estimate_zero_lag_rates()


#
# Initialize the FAP & FAR assignment machine
#


fapfar = stringutils.FAPFAR(rankingstatpdf)


#
# Iterate over databases
#


if options.verbose:
	print >>sys.stderr, "assigning FAPs and FARs"

for n, filename in enumerate(filenames, start = 1):
	#
	# get working copy of database
	#

	if options.verbose:
		print >>sys.stderr, "%d/%d: %s" % (n, len(filename), filename)

	working_filename = dbtables.get_connection_filename(filename, tmp_path = options.tmp_space, verbose = options.verbose)
	connection = sqlite3.connect(working_filename)

	#
	# record our passage
	#

	xmldoc = dbtables.get_xml(connection)
	process = ligolw_process.register_to_xmldoc(xmldoc, u"lalapps_string_compute_fapfar", process_params)

	#
	# assign FAPs and FARs
	#

	fapfar.assign_fapfars(connection)

	#
	# done, restore file to original location
	#

	ligolw_process.set_process_end_time(process)
	connection.cursor().execute("UPDATE process SET end_time = ? WHERE process_id == ?", (process.end_time, process.process_id))

	connection.commit()
	connection.close()
	dbtables.put_connection_filename(filename, working_filename, verbose = options.verbose)

if options.verbose:
	print >>sys.stderr, "FAP and FAR assignment complete"


#
# write parameter and ranking stat distribution file now with
# zero-lag counts populated
#
# FIXME:  do not write this output file, rely on stand-alone tool to
# collect zero-lag counts before running this program, and make that
# information available to other tools that way
#


xmldoc = ligolw.Document()
xmldoc.appendChild(ligolw.LIGO_LW())
xmldoc.childNodes[-1].appendChild(rankingstatpdf.to_xml())
# FIXME dont hard code
outname = "post_STRING_RANKINGSTATPDF.xml.gz"
ligolw_utils.write_filename(xmldoc, outname, gz = outname.endswith(".gz"), verbose = options.verbose)

if options.verbose:
	print >>sys.stderr, "done"

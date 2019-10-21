#
# Copyright (C) 2019 Daichi Tsuna
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

# a module to construct ranking statistic PDFs from the calc_likelihood outputs

#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


from optparse import OptionParser

from ligo.lw import ligolw
from ligo.lw import utils as ligolw_utils
from ligo.lw.utils import process as ligolw_process

from lal.utils import CacheEntry
from lalburst import git_version
from lalburst import stringutils
from lalburst import string_lr_far


__author__ = "Daichi Tsuna <daichi.tsuna@ligo.org>"
__version__ = "git id %s" % git_version.id
__date__ = git_version.date


#
# =============================================================================
#
#                                 Command Line
#
# =============================================================================
#


def parse_command_line():
	parser = OptionParser(
		version = "Name: %%prog\n%s" % git_version.verbose_msg
	)
	parser.add_option("-c", "--input-cache", metavar = "filename", help = "Also process the files named in this LAL cache.  See lalapps_path2cache for information on how to produce a LAL cache file.")
	parser.add_option("-n", "--ranking-stat-samples", metavar = "N", default = 2**24, type = "int", help = "Construct ranking statistic histograms by drawing this many samples from the ranking statistic generator (default = 2^24).")
	parser.add_option("-o", "--output", metavar = "filename", help = "Write merged likelihood ratio histograms to this LIGO Light-Weight XML file.")
	parser.add_option("-v", "--verbose", action = "store_true", help = "Be verbose.")
	options, filenames = parser.parse_args()

	paramdict = options.__dict__.copy()

	if options.input_cache is not None:
		filenames = [CacheEntry(line).path for line in open(options.input_cache)]
	if not filenames:
		raise ValueError("no ranking statistic likelihood data files specified")

	if options.output is None:
		raise ValueError("must set --output")
		
	return options, filenames, paramdict


#
# =============================================================================
#
#                                     Main
#
# =============================================================================
#


#
# command line
#


options, filenames, paramdict = parse_command_line()


#
# load rankingstat data
#


rankingstat = stringutils.marginalize_rankingstat(filenames, verbose = options.verbose) 


#
# generate rankingstatpdfs
#


rankingstatpdf = stringutils.RankingStatPDF(rankingstat, nsamples = options.ranking_stat_samples, verbose = options.verbose)


#
# write to output
#


xmldoc = ligolw.Document()
xmldoc.appendChild(ligolw.LIGO_LW())
xmldoc.childNodes[-1].appendChild(rankingstatpdf.to_xml())
process = ligolw_process.register_to_xmldoc(xmldoc, program = u"lalapps_string_meas_likelihood", paramdict = paramdict, version = __version__, cvs_repository = "lscsoft", cvs_entry_time = __date__, comment = u"")
ligolw_process.set_process_end_time(process)
ligolw_utils.write_filename(xmldoc, options.output, gz = (options.output or "stdout").endswith(".gz"), verbose = options.verbose)

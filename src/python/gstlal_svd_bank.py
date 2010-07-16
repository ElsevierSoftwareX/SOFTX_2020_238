# Copyright (C) 2010  Kipp Cannon, Chad Hanna, Leo Singer
# Copyright (C) 2009  Kipp Cannon, Chad Hanna
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


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


__all__ = ("Bank", "build_bank", "read_bank", "write_bank")

import numpy
import sys
import cPickle

try:
	all
except NameError:
	# Python < 2.5 compatibility
	from glue.iterutils import all
from glue.ligolw import lsctables
from glue.ligolw import utils
from glue.ligolw.utils import process as ligolw_process


from gstlal import cbc_template_fir
from gstlal import misc as gstlalmisc
from gstlal import templates
from gstlal_reference_psd import read_psd


#
# =============================================================================
#
#                                  Utilities
#
# =============================================================================
#

#
# Read Approximant
#

def read_approximant(xmldoc):
	approximant=ligolw_process.get_process_params(xmldoc, "tmpltbank", "--approximant")
	approximant=approximant[0]

	supported_approximants=[u'FindChirpSP', u'IMRPhenomB']
	if approximant not in supported_approximants:
		raise ValueError, "unsupported approximant %s"% approximant
	return approximant

#
#check final frequency is populated and return the max final frequency
#

def check_ffinal_and_find_max_ffinal(xmldoc):
	sngl_inspiral_table=lsctables.table.get_table(xmldoc, lsctables.SnglInspiralTable.tableName)
	f_final=sngl_inspiral_table.getColumnByName("f_final")
	if not all(f_final):
		raise ValueError, "f_final column not populated"
	return max(f_final)

#
# sum-of-squares false alarm probability
#


def sum_of_squares_threshold_from_fap(fap, coefficients):
	return gstlalmisc.max_stat_thresh(coefficients, fap)
	#return gstlalmisc.cdf_weighted_chisq_Pinv(coefficients, numpy.zeros(coefficients.shape, dtype = "double"), numpy.ones(coefficients.shape, dtype = "int"), 0.0, 1.0 - fap, -1, fap / 16.0)


#
# =============================================================================
#
#                              Pipeline Metadata
#
# =============================================================================
#


class BankFragment(object):
	def __init__(self, rate, start, end):
		self.rate = rate
		self.start = start
		self.end = end

	def set_template_bank(self, template_bank, tolerance, snr_thresh, verbose = False):
		if verbose:
			print >>sys.stderr, "\t%d templates of %d samples" % template_bank.shape

		self.orthogonal_template_bank, self.singular_values, self.mix_matrix, self.chifacs = cbc_template_fir.decompose_templates(template_bank, tolerance)

		self.sum_of_squares_weights = numpy.sqrt(self.chifacs.mean() * gstlalmisc.ss_coeffs(self.singular_values,snr_thresh))

		if verbose:
			print >>sys.stderr, "\tidentified %d components" % self.orthogonal_template_bank.shape[0]
			print >>sys.stderr, "\tsum-of-squares expectation value is %g" % self.chifacs.mean()


class Bank(object):

	def __init__(self, bank_xmldoc, psd, time_slices, gate_fap, snr_threshold, tolerance, flow = 40.0, autocorrelation_length = None, logname = None, verbose = False):
		# FIXME: remove template_bank_filename when no longer needed
		# by trigger generator element
		self.template_bank_filename = None
		self.filter_length = max(time_slices['end'])
		self.snr_threshold = snr_threshold
		self.logname = logname

		# Generate downsampled templates
		template_bank, self.autocorrelation_bank, self.sigmasq = cbc_template_fir.generate_templates(
			lsctables.table.get_table( bank_xmldoc,lsctables.SnglInspiralTable.tableName ),
			read_approximant(bank_xmldoc),
			psd,
			flow,
			time_slices,
			autocorrelation_length = autocorrelation_length,
			verbose = verbose)

		# Assign template banks to fragments
		self.bank_fragments = [BankFragment(rate,begin,end) for rate,begin,end in time_slices]
		for i, bank_fragment in enumerate(self.bank_fragments):
			if verbose:
				print >>sys.stderr, "constructing template decomposition %d of %d:  %g s ... %g s" % (i + 1, len(self.bank_fragments), -bank_fragment.end, -bank_fragment.start)
			bank_fragment.set_template_bank(template_bank[i], tolerance, self.snr_threshold, verbose = verbose)

		self.gate_threshold = sum_of_squares_threshold_from_fap(gate_fap, numpy.array([weight**2 for bank_fragment in self.bank_fragments for weight in bank_fragment.sum_of_squares_weights], dtype = "double"))
		if verbose:
			print >>sys.stderr, "sum-of-squares threshold for false-alarm probability of %.16g:  %.16g" % (gate_fap, self.gate_threshold)

	def get_rates(self):
		return set(bank_fragment.rate for bank_fragment in self.bank_fragments)

	# FIXME: remove set_template_bank_filename when no longer needed
	# by trigger generator element
	def set_template_bank_filename(self,name):
		self.template_bank_filename = name



def build_bank(template_bank_filename, psd, flow, ortho_gate_fap, snr_threshold, svd_tolerance, verbose):
	# Open template bank file
	bank_xmldoc = utils.load_filename(
		template_bank_filename,
		gz = template_bank_filename.endswith(".gz"),
		verbose = verbose)

	# Get sngl inspiral table
	bank_sngl_table = lsctables.table.get_table( bank_xmldoc,lsctables.SnglInspiralTable.tableName )

	# Choose how to break up templates in time
	time_freq_bounds = templates.time_slices(
		zip(bank_sngl_table.get_column('mass1'),bank_sngl_table.get_column('mass2')),
		fhigh=check_ffinal_and_find_max_ffinal(bank_xmldoc),
		flow = flow,
		verbose=verbose)

	# Generate templates, perform SVD, get orthogonal basis
	# and store as Bank object
	bank = Bank(
		bank_xmldoc,
		psd,
		time_freq_bounds,
		gate_fap = ortho_gate_fap,
		snr_threshold = snr_threshold,
		tolerance = svd_tolerance,
		flow = flow,
		autocorrelation_length = 201,	# samples
		verbose = verbose
	)

	# FIXME: remove this when no longer needed
	# by trigger generator element.
	bank.set_template_bank_filename(template_bank_filename)
	return bank


def write_bank(filename, bank):
	f = open(filename, "wb")
	try:
		cPickle.dump(bank, f, -1)
	finally:
		f.close()


def read_bank(filename):
	f = open(filename, "rb")
	try:
		bank = cPickle.load(f)
	finally:
		f.close()
	return bank

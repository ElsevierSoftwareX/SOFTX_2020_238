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

try:
	all
except NameError:
	# Python < 2.5 compatibility
	from glue.iterutils import all
from glue.ligolw import ligolw, lsctables, array, param, utils, types
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

	supported_approximants=[u'FindChirpSP', u'TaylorF2', u'IMRPhenomB']
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

	def set_template_bank(self, template_bank, tolerance, snr_thresh, identity = False, verbose = False):
		if verbose:
			print >>sys.stderr, "\t%d templates of %d samples" % template_bank.shape

		self.orthogonal_template_bank, self.singular_values, self.mix_matrix, self.chifacs = cbc_template_fir.decompose_templates(template_bank, tolerance, identity = identity)

		self.sum_of_squares_weights = numpy.sqrt(self.chifacs.mean() * gstlalmisc.ss_coeffs(self.singular_values,snr_thresh))

		if verbose:
			print >>sys.stderr, "\tidentified %d components" % self.orthogonal_template_bank.shape[0]
			print >>sys.stderr, "\tsum-of-squares expectation value is %g" % self.chifacs.mean()


class Bank(object):

	def __init__(self, bank_xmldoc, psd, time_slices, gate_fap, snr_threshold, tolerance, flow = 40.0, autocorrelation_length = None, logname = None, identity = False, verbose = False):
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
			bank_fragment.set_template_bank(template_bank[i], tolerance, self.snr_threshold, identity = identity, verbose = verbose)

		self.gate_threshold = sum_of_squares_threshold_from_fap(gate_fap, numpy.array([weight**2 for bank_fragment in self.bank_fragments for weight in bank_fragment.sum_of_squares_weights], dtype = "double"))
		if verbose:
			print >>sys.stderr, "sum-of-squares threshold for false-alarm probability of %.16g:  %.16g" % (gate_fap, self.gate_threshold)

	def get_rates(self):
		return set(bank_fragment.rate for bank_fragment in self.bank_fragments)

	# FIXME: remove set_template_bank_filename when no longer needed
	# by trigger generator element
	def set_template_bank_filename(self,name):
		self.template_bank_filename = name



def build_bank(template_bank_filename, psd, flow, ortho_gate_fap, snr_threshold, svd_tolerance, padding = 1.1, identity = False, verbose = False):
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
		padding = padding,
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
		identity = identity,
		verbose = verbose
	)

	# FIXME: remove this when no longer needed
	# by trigger generator element.
	bank.set_template_bank_filename(template_bank_filename)
	return bank


def write_bank(filename, bank):
	"""Write an SVD bank to a LIGO_LW xml file."""

	# Create new document
	xmldoc = ligolw.Document()
	root = ligolw.LIGO_LW()

	# Add root-level scalar params
	root.appendChild(param.new_param('filter_length', types.FromPyType[float], bank.filter_length))
	root.appendChild(param.new_param('gate_threshold', types.FromPyType[float], bank.gate_threshold))
	root.appendChild(param.new_param('logname', types.FromPyType[str], bank.logname))
	root.appendChild(param.new_param('snr_threshold', types.FromPyType[float], bank.snr_threshold))
	root.appendChild(param.new_param('template_bank_filename', types.FromPyType[str], bank.template_bank_filename))

	# Add root-level arrays
	root.appendChild(array.from_array('autocorrelation_bank_real', bank.autocorrelation_bank.real))
	root.appendChild(array.from_array('autocorrelation_bank_imag', bank.autocorrelation_bank.imag))
	root.appendChild(array.from_array('sigmasq', numpy.array(bank.sigmasq)))

	# Write bank fragments
	for i, frag in enumerate(bank.bank_fragments):
		# Start new container
		el = ligolw.LIGO_LW()

		# Add scalar params
		el.appendChild(param.new_param('start', types.FromPyType[float], frag.start))
		el.appendChild(param.new_param('end', types.FromPyType[float], frag.end))
		el.appendChild(param.new_param('rate', types.FromPyType[int], frag.rate))

		# Add arrays
		el.appendChild(array.from_array('chifacs', frag.chifacs))
		el.appendChild(array.from_array('mix_matrix', frag.mix_matrix))
		el.appendChild(array.from_array('orthogonal_template_bank', frag.orthogonal_template_bank))
		el.appendChild(array.from_array('singular_values', frag.singular_values))
		el.appendChild(array.from_array('sum_of_squares_weights', frag.sum_of_squares_weights))

		# Add bank fragment container to root container
		root.appendChild(el)

	# Add root container to document
	xmldoc.appendChild(root)

	# Write to file
	utils.write_filename(xmldoc, filename, gz=filename.endswith('.gz'))


def read_bank(filename):
	"""Read an SVD bank from a LIGO_LW xml file."""

	# Load document
	xmldoc = utils.load_filename(filename, gz=filename.endswith('.gz'))
	root = xmldoc.childNodes[0]

	# Create new SVD bank object
	bank = Bank.__new__(Bank)

	# Read root-level scalar parameters
	bank.filter_length = param.get_pyvalue(root, 'filter_length')
	bank.gate_threshold = param.get_pyvalue(root, 'gate_threshold')
	bank.logname = param.get_pyvalue(root, 'logname')
	bank.snr_threshold = param.get_pyvalue(root, 'snr_threshold')
	bank.template_bank_filename = param.get_pyvalue(root, 'template_bank_filename')

	# Read root-level arrays
	autocorrelation_bank_real = array.get_array(root, 'autocorrelation_bank_real').array
	autocorrelation_bank_imag = array.get_array(root, 'autocorrelation_bank_imag').array
	bank.autocorrelation_bank = autocorrelation_bank_real + (0+1j) * autocorrelation_bank_imag
	bank.sigmasq = array.get_array(root, 'sigmasq').array

	bank_fragments = []

	# Read bank fragments
	for el in (node for node in root.childNodes if node.tagName == 'LIGO_LW'):
		frag = BankFragment.__new__(BankFragment)

		# Read scalar params
		frag.start = param.get_pyvalue(el, 'start')
		frag.end = param.get_pyvalue(el, 'end')
		frag.rate = param.get_pyvalue(el, 'rate')

		# Read arrays
		frag.chifacs = array.get_array(el, 'chifacs').array
		frag.mix_matrix = array.get_array(el, 'mix_matrix').array
		frag.orthogonal_template_bank = array.get_array(el, 'orthogonal_template_bank').array
		frag.singular_values = array.get_array(el, 'singular_values').array
		frag.sum_of_squares_weights = array.get_array(el, 'sum_of_squares_weights').array

		bank_fragments.append(frag)

	bank.bank_fragments = bank_fragments

	return bank

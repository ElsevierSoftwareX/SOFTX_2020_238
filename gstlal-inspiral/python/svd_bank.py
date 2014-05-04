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

## @file
#
# ### Review Status
#
# | Names                                          | Hash                                        | Date       |
# | -------------------------------------------    | ------------------------------------------- | ---------- |
# | Florent, Sathya, Duncan Me, Jolien, Kipp, Chad | 7536db9d496be9a014559f4e273e1e856047bf71    | 2014-04-30 |
#
# #### Actions
# - Consider a study of how to supply the svd / time slice boundaries
# 

## @package svd_bank


#
# =============================================================================
#
#				   Preamble
#
# =============================================================================
#


import numpy
import sys

from glue.ligolw import ligolw
from glue.ligolw import lsctables
from glue.ligolw import array as ligolw_array
from glue.ligolw import param as ligolw_param
from glue.ligolw import utils as ligolw_utils
from glue.ligolw import types as ligolw_types
from glue.ligolw.utils import process as ligolw_process
from pylal import series

Attributes = ligolw.sax.xmlreader.AttributesImpl

from gstlal import cbc_template_fir
from gstlal import misc as gstlalmisc
from gstlal import templates


# FIXME:  require calling code to provide the content handler
class DefaultContentHandler(ligolw.LIGOLWContentHandler):
	pass
ligolw_array.use_in(DefaultContentHandler)
ligolw_param.use_in(DefaultContentHandler)
lsctables.use_in(DefaultContentHandler)


#
# =============================================================================
#
#				  Utilities
#
# =============================================================================
#

#
# Read approximant
#

def read_approximant(xmldoc, programs = ("tmpltbank", "lalapps_cbc_sbank")):
	process_ids = set()
	for program in programs:
		process_ids |= lsctables.table.get_table(xmldoc, lsctables.ProcessTable.tableName).get_ids_by_program(program)
	if not process_ids:
		raise ValueError("document must contain process entries for at least one of the programs %s" % ", ".join(programs))
	approximant = set(row.pyvalue for row in lsctables.table.get_table(xmldoc, lsctables.ProcessParamsTable.tableName) if (row.process_id in process_ids) and (row.param == "--approximant"))
	if not approximant:
		raise ValueError("document must contain an 'approximant' process_params entry for one or more of the programs %s" % ", ".join("'%s'" for program in programs))
	if len(approximant) > 1:
		raise ValueError("document must contain only one approximant")
	approximant = approximant.pop()
	templates.gstlal_valid_approximant(approximant)
	return approximant

#
# check final frequency is populated and return the max final frequency
#

def check_ffinal_and_find_max_ffinal(xmldoc):
	sngl_inspiral_table=lsctables.table.get_table(xmldoc, lsctables.SnglInspiralTable.tableName)
	f_final=sngl_inspiral_table.getColumnByName("f_final")
	if not all(f_final):
		raise ValueError("f_final column not populated")
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
#			      Pipeline Metadata
#
# =============================================================================
#


class BankFragment(object):
	def __init__(self, rate, start, end):
		self.rate = rate
		self.start = start
		self.end = end

	def set_template_bank(self, template_bank, tolerance, snr_thresh, identity_transform = False, verbose = False):
		if verbose:
			print >>sys.stderr, "\t%d templates of %d samples" % template_bank.shape

		self.orthogonal_template_bank, self.singular_values, self.mix_matrix, self.chifacs = cbc_template_fir.decompose_templates(template_bank, tolerance, identity = identity_transform)

		if self.singular_values is not None:
			self.sum_of_squares_weights = numpy.sqrt(self.chifacs.mean() * gstlalmisc.ss_coeffs(self.singular_values,snr_thresh))
		else:
			self.sum_of_squares_weights = None
		if verbose:
			print >>sys.stderr, "\tidentified %d components" % self.orthogonal_template_bank.shape[0]
			print >>sys.stderr, "\tsum-of-squares expectation value is %g" % self.chifacs.mean()


class Bank(object):
	def __init__(self, bank_xmldoc, psd, time_slices, gate_fap, snr_threshold, tolerance, flow = 40.0, autocorrelation_length = None, logname = None, identity_transform = False, verbose = False, bank_id = None):
		# FIXME: remove template_bank_filename when no longer needed
		# by trigger generator element
		self.template_bank_filename = None
		self.filter_length = max(time_slices['end'])
		self.snr_threshold = snr_threshold
		self.logname = logname
		self.bank_id = bank_id

		# Generate downsampled templates
		template_bank, self.autocorrelation_bank, self.autocorrelation_mask, self.sigmasq, processed_psd = cbc_template_fir.generate_templates(
			lsctables.table.get_table( bank_xmldoc,lsctables.SnglInspiralTable.tableName ),
			read_approximant(bank_xmldoc),
			psd,
			flow,
			time_slices,
			autocorrelation_length = autocorrelation_length,
			verbose = verbose)
		
		# Include signal inspiral table
		self.sngl_inspiral_table = lsctables.table.get_table(bank_xmldoc, lsctables.SnglInspiralTable.tableName)
		# Include the processed psd
		self.processed_psd = processed_psd

		# Assign template banks to fragments
		self.bank_fragments = [BankFragment(rate,begin,end) for rate,begin,end in time_slices]
		for i, bank_fragment in enumerate(self.bank_fragments):
			if verbose:
				print >>sys.stderr, "constructing template decomposition %d of %d:  %g s ... %g s" % (i + 1, len(self.bank_fragments), -bank_fragment.end, -bank_fragment.start)
			bank_fragment.set_template_bank(template_bank[i], tolerance, self.snr_threshold, identity_transform = identity_transform, verbose = verbose)

		if bank_fragment.sum_of_squares_weights is not None:
			self.gate_threshold = sum_of_squares_threshold_from_fap(gate_fap, numpy.array([weight**2 for bank_fragment in self.bank_fragments for weight in bank_fragment.sum_of_squares_weights], dtype = "double"))
		else:
			self.gate_threshold = 0
		if verbose:
			print >>sys.stderr, "sum-of-squares threshold for false-alarm probability of %.16g:  %.16g" % (gate_fap, self.gate_threshold)

	def get_rates(self):
		return set(bank_fragment.rate for bank_fragment in self.bank_fragments)

	# FIXME: remove set_template_bank_filename when no longer needed
	# by trigger generator element
	def set_template_bank_filename(self,name):
		self.template_bank_filename = name



def build_bank(template_bank_filename, psd, flow, ortho_gate_fap, snr_threshold, svd_tolerance, padding = 1.5, identity_transform = False, verbose = False, autocorrelation_length = 201, samples_min = 1024, samples_max_256 = 1024, samples_max_64 = 2048, samples_max = 4096, bank_id = None, contenthandler = DefaultContentHandler):
	"""!
	Return an instance of a Bank class.

	@param template_bank_filename The template bank filename containing a subbank of templates to decompose in a single inpsiral table.
	@param psd A class instance of a psd.
	@param flow The lower frequency cutoff.
	@param ortho_gate_fap The FAP threshold for the sum of squares threshold, see http://arxiv.org/abs/1101.0584
	@param snr_threshold The SNR threshold for the search
	@param svd_tolerance The target SNR loss of the SVD, see http://arxiv.org/abs/1005.0012
	@param padding The padding from Nyquist for any template time slice, e.g., if a time slice has a Nyquist of 256 Hz and the padding is set to 2, only allow the template frequency to extend to 128 Hz.
	@param identity_transform Don't do the SVD, just do time slices and keep the raw waveforms
	@param verbose Be verbose
	@param autocorrelation_length The number of autocorrelation samples to use in the chisquared test.  Must be odd
	@param samples_min The minimum number of samples to use in any time slice
	@param samples_max_256 The maximum number of samples to have in any time slice greater than or equal to 256 Hz
	@param samples_max_64 The maximum number of samples to have in any time slice greater than or equal to 64 Hz
	@param samples_max The maximum number of samples in any time slice below 64 Hz
	@param bank_id The id of the bank in question
	@param contenthandler The ligolw content handler for file I/O
	"""

	# Open template bank file
	bank_xmldoc = ligolw_utils.load_filename(template_bank_filename, contenthandler = contenthandler, verbose = verbose)

	# Get sngl inspiral table
	bank_sngl_table = lsctables.table.get_table( bank_xmldoc,lsctables.SnglInspiralTable.tableName )

	# Choose how to break up templates in time
	time_freq_bounds = templates.time_slices(
		bank_sngl_table,
		fhigh=check_ffinal_and_find_max_ffinal(bank_xmldoc),
		flow = flow,
		padding = padding,
		samples_min = samples_min,
		samples_max_256 = samples_max_256,
		samples_max_64 = samples_max_64,
		samples_max = samples_max,
		verbose=verbose)

	# Generate templates, perform SVD, get orthogonal basis
	# and store as Bank object
	bank = Bank(
		bank_xmldoc,
		psd[bank_sngl_table[0].ifo],
		time_freq_bounds,
		gate_fap = ortho_gate_fap,
		snr_threshold = snr_threshold,
		tolerance = svd_tolerance,
		flow = flow,
		autocorrelation_length = autocorrelation_length,	# samples
		identity_transform = identity_transform,
		verbose = verbose,
		bank_id = bank_id
	)

	# FIXME: remove this when no longer needed
	# by trigger generator element.
	bank.set_template_bank_filename(template_bank_filename)
	return bank


def write_bank(filename, banks, cliplefts = None, cliprights = None, contenthandler = DefaultContentHandler, verbose = False):
	"""Write SVD banks to a LIGO_LW xml file."""

	# Create new document
	xmldoc = ligolw.Document()
	lw = ligolw.LIGO_LW()

	for bank, clipleft, clipright in zip(banks, cliplefts, cliprights):
		# set up root for this sub bank
		root = ligolw.LIGO_LW(Attributes({u"Name": u"gstlal_svd_bank_Bank"}))
		lw.appendChild(root)

		# FIXME FIXME FIXME move this clipping stuff to the Bank class
		# Open template bank file
		bank_xmldoc = ligolw_utils.load_filename(bank.template_bank_filename, contenthandler = contenthandler, verbose = verbose)

		# Get sngl inspiral table
		sngl_inspiral_table = lsctables.table.get_table(bank_xmldoc, lsctables.SnglInspiralTable.tableName)

		# set the right clipping index
		clipright = len(sngl_inspiral_table) - clipright
		
		# Apply clipping option to sngl inspiral table
		sngl_inspiral_table = sngl_inspiral_table[clipleft:clipright]

		# put the bank table into the output document
		new_sngl_table = lsctables.New(lsctables.SnglInspiralTable)
		for row in sngl_inspiral_table:
			new_sngl_table.append(row)

		# put the possibly clipped table into the file
		root.appendChild(new_sngl_table)

		# Add root-level scalar params
		root.appendChild(ligolw_param.new_param('filter_length', ligolw_types.FromPyType[float], bank.filter_length))
		root.appendChild(ligolw_param.new_param('gate_threshold', ligolw_types.FromPyType[float], bank.gate_threshold))
		root.appendChild(ligolw_param.new_param('logname', ligolw_types.FromPyType[str], bank.logname))
		root.appendChild(ligolw_param.new_param('snr_threshold', ligolw_types.FromPyType[float], bank.snr_threshold))
		root.appendChild(ligolw_param.new_param('template_bank_filename', ligolw_types.FromPyType[str], bank.template_bank_filename))
		root.appendChild(ligolw_param.new_param('bank_id', ligolw_types.FromPyType[str], bank.bank_id))

		# apply clipping to autocorrelations and sigmasq
		bank.autocorrelation_bank = bank.autocorrelation_bank[clipleft:clipright,:]
		bank.sigmasq = bank.sigmasq[clipleft:clipright]

		# Add root-level arrays
		# FIXME:  ligolw format now supports complex-valued data
		root.appendChild(ligolw_array.from_array('autocorrelation_bank_real', bank.autocorrelation_bank.real))
		root.appendChild(ligolw_array.from_array('autocorrelation_bank_imag', bank.autocorrelation_bank.imag))
		root.appendChild(ligolw_array.from_array('autocorrelation_mask', bank.autocorrelation_mask))
		root.appendChild(ligolw_array.from_array('sigmasq', numpy.array(bank.sigmasq)))

		# Write bank fragments
		for i, frag in enumerate(bank.bank_fragments):
			# Start new container
			el = ligolw.LIGO_LW()

			# Apply clipping option
			if frag.mix_matrix is not None:
				frag.mix_matrix = frag.mix_matrix[:,clipleft*2:clipright*2]
			frag.chifacs = frag.chifacs[clipleft*2:clipright*2]

			# Add scalar params
			el.appendChild(ligolw_param.new_param('start', ligolw_types.FromPyType[float], frag.start))
			el.appendChild(ligolw_param.new_param('end', ligolw_types.FromPyType[float], frag.end))
			el.appendChild(ligolw_param.new_param('rate', ligolw_types.FromPyType[int], frag.rate))

			# Add arrays
			el.appendChild(ligolw_array.from_array('chifacs', frag.chifacs))
			if frag.mix_matrix is not None:
				el.appendChild(ligolw_array.from_array('mix_matrix', frag.mix_matrix))
			el.appendChild(ligolw_array.from_array('orthogonal_template_bank', frag.orthogonal_template_bank))
			if frag.singular_values is not None:
				el.appendChild(ligolw_array.from_array('singular_values', frag.singular_values))
			if frag.sum_of_squares_weights is not None:
				el.appendChild(ligolw_array.from_array('sum_of_squares_weights', frag.sum_of_squares_weights))

			# Add bank fragment container to root container
			root.appendChild(el)

	# put a copy of the processed PSD file in
	# FIXME in principle this could be different for each bank included in
	# this file, but we only put one here
	series.make_psd_xmldoc({bank.sngl_inspiral_table[0].ifo: bank.processed_psd}, lw)

	# add top level LIGO_LW to document
	xmldoc.appendChild(lw)

	# Write to file
	ligolw_utils.write_filename(xmldoc, filename, gz = filename.endswith('.gz'), verbose = verbose)


def read_banks(filename, contenthandler = DefaultContentHandler, verbose = False):
	"""Read SVD banks from a LIGO_LW xml file."""

	# Load document
	xmldoc = ligolw_utils.load_filename(filename, contenthandler = contenthandler, verbose = verbose)

	banks = []

	for root in (elem for elem in xmldoc.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") and elem.Name == "gstlal_svd_bank_Bank"):
	
		# Create new SVD bank object
		bank = Bank.__new__(Bank)

		# Read sngl inspiral table
		bank.sngl_inspiral_table = lsctables.table.get_table(root, lsctables.SnglInspiralTable.tableName)

		# Read root-level scalar parameters
		bank.filter_length = ligolw_param.get_pyvalue(root, 'filter_length')
		bank.gate_threshold = ligolw_param.get_pyvalue(root, 'gate_threshold')
		bank.logname = ligolw_param.get_pyvalue(root, 'logname')
		bank.snr_threshold = ligolw_param.get_pyvalue(root, 'snr_threshold')
		bank.template_bank_filename = ligolw_param.get_pyvalue(root, 'template_bank_filename')
		bank.bank_id = ligolw_param.get_pyvalue(root, 'bank_id')

		# Read root-level arrays
		bank.autocorrelation_bank = ligolw_array.get_array(root, 'autocorrelation_bank_real').array + 1j * ligolw_array.get_array(root, 'autocorrelation_bank_imag').array
		bank.autocorrelation_mask = ligolw_array.get_array(root, 'autocorrelation_mask').array
		bank.sigmasq = ligolw_array.get_array(root, 'sigmasq').array

		# Read bank fragments
		bank.bank_fragments = []
		for el in (node for node in root.childNodes if node.tagName == ligolw.LIGO_LW.tagName):
			frag = BankFragment.__new__(BankFragment)

			# Read scalar params
			frag.start = ligolw_param.get_pyvalue(el, 'start')
			frag.end = ligolw_param.get_pyvalue(el, 'end')
			frag.rate = ligolw_param.get_pyvalue(el, 'rate')

			# Read arrays
			frag.chifacs = ligolw_array.get_array(el, 'chifacs').array
			try:
				frag.mix_matrix = ligolw_array.get_array(el, 'mix_matrix').array
			except ValueError:
				frag.mix_matrix = None
			frag.orthogonal_template_bank = ligolw_array.get_array(el, 'orthogonal_template_bank').array
			try:
				frag.singular_values = ligolw_array.get_array(el, 'singular_values').array
			except ValueError:
				frag.singular_values = None
			try:
				frag.sum_of_squares_weights = ligolw_array.get_array(el, 'sum_of_squares_weights').array
			except ValueError:
				frag.sum_of_squares_weights = None
			bank.bank_fragments.append(frag)

		banks.append(bank)
	return banks

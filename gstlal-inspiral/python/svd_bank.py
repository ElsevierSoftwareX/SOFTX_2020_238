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
# The module to implement SVD decomposition of CBC waveforms
#
# ### Review Status
#
# | Names                                          | Hash                                        | Date       | Diff to Head of Master      |
# | -------------------------------------------    | ------------------------------------------- | ---------- | --------------------------- |
# | Florent, Sathya, Duncan Me, Jolien, Kipp, Chad | 7536db9d496be9a014559f4e273e1e856047bf71    | 2014-04-30 | <a href="@gstlal_inspiral_cgit_diff/python/svd_bank.py?id=HEAD&id2=7536db9d496be9a014559f4e273e1e856047bf71">svd_bank.py</a> |
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
import os
import sys
import warnings

import lal

from ligo.lw import ligolw
from ligo.lw import lsctables
from ligo.lw import array as ligolw_array
from ligo.lw import param as ligolw_param
from ligo.lw import utils as ligolw_utils
from ligo.lw import types as ligolw_types
from ligo.lw.utils import process as ligolw_process

Attributes = ligolw.sax.xmlreader.AttributesImpl

from gstlal import cbc_template_fir
from gstlal import misc as gstlalmisc
from gstlal import templates
from gstlal import reference_psd


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

def read_approximant(xmldoc, programs = ("gstlal_bank_splitter",)):
	process_ids = set()
	for program in programs:
		process_ids |= lsctables.ProcessTable.get_table(xmldoc).get_ids_by_program(program)
	if not process_ids:
		raise ValueError("document must contain process entries from %s" % ", ".join(programs))
	approximant = set(row.pyvalue for row in lsctables.ProcessParamsTable.get_table(xmldoc) if (row.process_id in process_ids) and (row.param == "--approximant"))
	if not approximant:
		raise ValueError("document must contain an 'approximant' process_params entry from %s" % ", ".join("'%s'" for program in programs))
	if len(approximant) > 1:
		raise ValueError("document must contain only one approximant")
	approximant = approximant.pop()
	templates.gstlal_valid_approximant(approximant)
	return approximant

#
# check final frequency is populated and return the max final frequency
#

def check_ffinal_and_find_max_ffinal(xmldoc):
	f_final = lsctables.SnglInspiralTable.get_table(xmldoc).getColumnByName("f_final")
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
	def __init__(self, bank_xmldoc, psd, time_slices, gate_fap, snr_threshold, tolerance, flow = 40.0, autocorrelation_length = None, logname = None, identity_transform = False, verbose = False, bank_id = None, fhigh = None):
		# FIXME: remove template_bank_filename when no longer needed
		# by trigger generator element
		self.template_bank_filename = None
		self.filter_length = time_slices['end'].max()
		self.snr_threshold = snr_threshold
		if logname is not None and not logname:
			raise ValueError("logname cannot be empty if it is set")
		self.logname = logname
		self.bank_id = bank_id

		# Generate downsampled templates
		template_bank, self.autocorrelation_bank, self.autocorrelation_mask, self.sigmasq, bank_workspace = cbc_template_fir.generate_templates(
			lsctables.SnglInspiralTable.get_table(bank_xmldoc),
			read_approximant(bank_xmldoc),
			psd,
			flow,
			time_slices,
			autocorrelation_length = autocorrelation_length,
			fhigh = fhigh,
			verbose = verbose)

		# Include signal inspiral table
		sngl_inspiral_table = lsctables.SnglInspiralTable.get_table(bank_xmldoc)
		self.sngl_inspiral_table = sngl_inspiral_table.copy()
		self.sngl_inspiral_table.extend(sngl_inspiral_table)
		# Include the processed psd
		self.processed_psd = bank_workspace.psd
		# Include some parameters passed from the bank workspace
		self.newdeltaF = 1. / bank_workspace.working_duration
		self.working_f_low = bank_workspace.working_f_low
		self.f_low = bank_workspace.f_low
		self.sample_rate_max = bank_workspace.sample_rate_max

		# Assign template banks to fragments
		self.bank_fragments = [BankFragment(rate,begin,end) for rate,begin,end in time_slices]
		for i, bank_fragment in enumerate(self.bank_fragments):
			if verbose:
				print >>sys.stderr, "constructing template decomposition %d of %d:  %g s ... %g s" % (i + 1, len(self.bank_fragments), -bank_fragment.end, -bank_fragment.start)
			bank_fragment.set_template_bank(template_bank[i], tolerance, self.snr_threshold, identity_transform = identity_transform, verbose = verbose)

		if bank_fragment.sum_of_squares_weights is not None:
			self.gate_threshold = sum_of_squares_threshold_from_fap(gate_fap, numpy.array([weight**2 for bank_fragment in self.bank_fragments for weight in bank_fragment.sum_of_squares_weights], dtype = "double"))
		else:
			self.gate_threshold = 0.
		if verbose:
			print >>sys.stderr, "sum-of-squares threshold for false-alarm probability of %.16g:  %.16g" % (gate_fap, self.gate_threshold)

	def get_rates(self):
		return set(bank_fragment.rate for bank_fragment in self.bank_fragments)

	# FIXME: remove set_template_bank_filename when no longer needed
	# by trigger generator element
	def set_template_bank_filename(self,name):
		self.template_bank_filename = name



def build_bank(template_bank_url, psd, flow, ortho_gate_fap, snr_threshold, svd_tolerance, padding = 1.5, identity_transform = False, verbose = False, autocorrelation_length = 201, samples_min = 1024, samples_max_256 = 1024, samples_max_64 = 2048, samples_max = 4096, bank_id = None, contenthandler = None, sample_rate = None, instrument_override = None):
	"""!
	Return an instance of a Bank class.

	@param template_bank_url The template bank filename or url containing a subbank of templates to decompose in a single inpsiral table.
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
	bank_xmldoc = ligolw_utils.load_url(template_bank_url, contenthandler = contenthandler, verbose = verbose)

	# Get sngl inspiral table
	bank_sngl_table = lsctables.SnglInspiralTable.get_table(bank_xmldoc)

	# override instrument if needed (this is useful if a generic instrument independent bank file is provided
	if instrument_override is not None:
		for row in bank_sngl_table:
			row.ifo = instrument_override

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
		sample_rate = sample_rate,
		verbose=verbose)

	if sample_rate is not None:
		fhigh=check_ffinal_and_find_max_ffinal(bank_xmldoc)
	else:
		fhigh=None
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
		bank_id = bank_id,
		fhigh = fhigh
	)

	# FIXME: remove this when no longer needed
	# by trigger generator element.
	bank.set_template_bank_filename(ligolw_utils.local_path_from_url(template_bank_url))
	return bank


def write_bank(filename, banks, psd_input, cliplefts = None, cliprights = None, verbose = False):
	"""Write SVD banks to a LIGO_LW xml file."""

	# Create new document
	xmldoc = ligolw.Document()
	lw = xmldoc.appendChild(ligolw.LIGO_LW())

	for bank, clipleft, clipright in zip(banks, cliplefts, cliprights):
		# set up root for this sub bank
		root = lw.appendChild(ligolw.LIGO_LW(Attributes({u"Name": u"gstlal_svd_bank_Bank"})))

		# FIXME FIXME FIXME move this clipping stuff to the Bank class
		# set the right clipping index
		clipright = len(bank.sngl_inspiral_table) - clipright

		# Apply clipping option to sngl inspiral table
		# put the bank table into the output document
		new_sngl_table = bank.sngl_inspiral_table.copy()
		for row in bank.sngl_inspiral_table[clipleft:clipright]:
			# FIXME need a proper id column
			row.Gamma1 = int(bank.bank_id.split("_")[0])
			new_sngl_table.append(row)

		# put the possibly clipped table into the file
		root.appendChild(new_sngl_table)

		# Add root-level scalar params
		root.appendChild(ligolw_param.Param.from_pyvalue('filter_length', bank.filter_length))
		root.appendChild(ligolw_param.Param.from_pyvalue('gate_threshold', bank.gate_threshold))
		root.appendChild(ligolw_param.Param.from_pyvalue('logname', bank.logname or ""))
		root.appendChild(ligolw_param.Param.from_pyvalue('snr_threshold', bank.snr_threshold))
		root.appendChild(ligolw_param.Param.from_pyvalue('template_bank_filename', bank.template_bank_filename))
		root.appendChild(ligolw_param.Param.from_pyvalue('bank_id', bank.bank_id))
		root.appendChild(ligolw_param.Param.from_pyvalue('new_deltaf', bank.newdeltaF))
		root.appendChild(ligolw_param.Param.from_pyvalue('working_f_low', bank.working_f_low))
		root.appendChild(ligolw_param.Param.from_pyvalue('f_low', bank.f_low))
		root.appendChild(ligolw_param.Param.from_pyvalue('sample_rate_max', bank.sample_rate_max))
		root.appendChild(ligolw_param.Param.from_pyvalue('gstlal_fir_whiten', os.environ['GSTLAL_FIR_WHITEN']))

		# apply clipping to autocorrelations and sigmasq
		bank.autocorrelation_bank = bank.autocorrelation_bank[clipleft:clipright,:]
		bank.autocorrelation_mask = bank.autocorrelation_mask[clipleft:clipright,:]
		bank.sigmasq = bank.sigmasq[clipleft:clipright]

		# Add root-level arrays
		# FIXME:  ligolw format now supports complex-valued data
		root.appendChild(ligolw_array.Array.build('autocorrelation_bank_real', bank.autocorrelation_bank.real))
		root.appendChild(ligolw_array.Array.build('autocorrelation_bank_imag', bank.autocorrelation_bank.imag))
		root.appendChild(ligolw_array.Array.build('autocorrelation_mask', bank.autocorrelation_mask))
		root.appendChild(ligolw_array.Array.build('sigmasq', numpy.array(bank.sigmasq)))

		# Write bank fragments
		for i, frag in enumerate(bank.bank_fragments):
			# Start new bank fragment container
			el = root.appendChild(ligolw.LIGO_LW())

			# Apply clipping option
			if frag.mix_matrix is not None:
				frag.mix_matrix = frag.mix_matrix[:,clipleft*2:clipright*2]
			frag.chifacs = frag.chifacs[clipleft*2:clipright*2]

			# Add scalar params
			el.appendChild(ligolw_param.Param.from_pyvalue('rate', frag.rate))
			el.appendChild(ligolw_param.Param.from_pyvalue('start', frag.start))
			el.appendChild(ligolw_param.Param.from_pyvalue('end', frag.end))

			# Add arrays
			el.appendChild(ligolw_array.Array.build('chifacs', frag.chifacs))
			if frag.mix_matrix is not None:
				el.appendChild(ligolw_array.Array.build('mix_matrix', frag.mix_matrix))
			el.appendChild(ligolw_array.Array.build('orthogonal_template_bank', frag.orthogonal_template_bank))
			if frag.singular_values is not None:
				el.appendChild(ligolw_array.Array.build('singular_values', frag.singular_values))
			if frag.sum_of_squares_weights is not None:
				el.appendChild(ligolw_array.Array.build('sum_of_squares_weights', frag.sum_of_squares_weights))

	# put a copy of the processed PSD file in
	# FIXME in principle this could be different for each bank included in
	# this file, but we only put one here
	psd = psd_input[bank.sngl_inspiral_table[0].ifo]
	lal.series.make_psd_xmldoc({bank.sngl_inspiral_table[0].ifo: psd}, lw)

	# Write to file
	ligolw_utils.write_filename(xmldoc, filename, gz = filename.endswith('.gz'), verbose = verbose)


def read_banks(filename, contenthandler, verbose = False):
	"""Read SVD banks from a LIGO_LW xml file."""

	# Load document
	xmldoc = ligolw_utils.load_url(filename, contenthandler = contenthandler, verbose = verbose)

	banks = []

	# FIXME in principle this could be different for each bank included in
	# this file, but we only put one in the file for now
	# FIXME, right now there is only one instrument so we just pull out the
	# only psd there is
	try:
		raw_psd = lal.series.read_psd_xmldoc(xmldoc).values()[0]
	except ValueError:
		# the bank file does not contain psd ligolw element.
		raw_psd = None

	for root in (elem for elem in xmldoc.getElementsByTagName(ligolw.LIGO_LW.tagName) if elem.hasAttribute(u"Name") and elem.Name == "gstlal_svd_bank_Bank"):

		# Create new SVD bank object
		bank = Bank.__new__(Bank)

		# Read sngl inspiral table
		bank.sngl_inspiral_table = lsctables.SnglInspiralTable.get_table(root)
		bank.sngl_inspiral_table.parentNode.removeChild(bank.sngl_inspiral_table)

		# Read root-level scalar parameters
		bank.filter_length = ligolw_param.get_pyvalue(root, 'filter_length')
		bank.gate_threshold = ligolw_param.get_pyvalue(root, 'gate_threshold')
		bank.logname = ligolw_param.get_pyvalue(root, 'logname') or None
		bank.snr_threshold = ligolw_param.get_pyvalue(root, 'snr_threshold')
		bank.template_bank_filename = ligolw_param.get_pyvalue(root, 'template_bank_filename')
		bank.bank_id = ligolw_param.get_pyvalue(root, 'bank_id')

		try:
			bank.newdeltaF = ligolw_param.get_pyvalue(root, 'new_deltaf')
			bank.working_f_low = ligolw_param.get_pyvalue(root, 'working_f_low')
			bank.f_low = ligolw_param.get_pyvalue(root, 'f_low')
			bank.sample_rate_max = ligolw_param.get_pyvalue(root, 'sample_rate_max')
		except ValueError:
			pass

		# Read root-level arrays
		bank.autocorrelation_bank = ligolw_array.get_array(root, 'autocorrelation_bank_real').array + 1j * ligolw_array.get_array(root, 'autocorrelation_bank_imag').array
		bank.autocorrelation_mask = ligolw_array.get_array(root, 'autocorrelation_mask').array
		bank.sigmasq = ligolw_array.get_array(root, 'sigmasq').array

		# prepare the horizon distance factors
		bank.horizon_factors = dict((row.template_id, sigmasq**.5) for row, sigmasq in zip(bank.sngl_inspiral_table, bank.sigmasq))

		if raw_psd is not None:
			# reproduce the whitening psd and attach a reference to the psd
			bank.processed_psd = cbc_template_fir.condition_psd(raw_psd, bank.newdeltaF, minfs = (bank.working_f_low, bank.f_low), maxfs = (bank.sample_rate_max / 2.0 * 0.90, bank.sample_rate_max / 2.0))

		# Read bank fragments
		bank.bank_fragments = []
		for el in (node for node in root.childNodes if node.tagName == ligolw.LIGO_LW.tagName):
			frag = BankFragment(
				rate = ligolw_param.get_pyvalue(el, 'rate'),
				start = ligolw_param.get_pyvalue(el, 'start'),
				end = ligolw_param.get_pyvalue(el, 'end')
			)

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
	template_id, func = horizon_distance_func(banks)
	horizon_norm = None
	for bank in banks:
		if template_id in bank.horizon_factors:
			assert horizon_norm is None
			horizon_norm = bank.horizon_factors[template_id]
	for bank in banks:
		bank.horizon_distance_func = func
		bank.horizon_factors = dict((tid, f / horizon_norm) for (tid, f) in bank.horizon_factors.items())
	xmldoc.unlink()
	return banks


def svdbank_templates_mapping(filenames, contenthandler, verbose = False):
	"""
	From a list of the names of files containing SVD bank objects,
	construct a dictionary mapping filename to list of sngl_inspiral
	templates in that file.  Typically this mapping is inverted through
	the use of some sort of "template identity" function to map each
	template to the filename that contains that template.

	Example:

	Assuming the (mass1, mass2) tuple is known to uniquely identify the
	templates

	>>> def template_id(row):
	...	return row.mass1, row.mass2
	...
	>>> mapping = svdbank_templates_mapping([], DefaultContentHandler)
	>>> template_to_filename = dict((template_id(tempate), filename) for filename, templates in mapping.items() for template in templates)
	"""
	mapping = {}
	for n, filename in enumerate(filenames, start = 1):
		if verbose:
			print >>sys.stderr, "%d/%d:" % (n, len(filenames)),
		mapping[filename] = sum((bank.sngl_inspiral_table for bank in read_banks(filename, contenthandler, verbose = verbose)), [])
	return mapping

def preferred_horizon_distance_template(banks):
	template_id, m1, m2, s1z, s2z = min((row.template_id, row.mass1, row.mass2, row.spin1z, row.spin2z) for bank in banks for row in bank.sngl_inspiral_table)
	return template_id, m1, m2, s1z, s2z

def horizon_distance_func(banks):
	"""
	Takes a dictionary of objects returned by read_banks keyed by instrument
	"""
	# span is [15 Hz, 0.85 * Nyquist frequency]
	# find the Nyquist frequency for the PSD to be used for each
	# instrument.  require them to all match
	nyquists = set((max(bank.get_rates())/2. for bank in banks))
	if len(nyquists) != 1:
		warnings.warn("all banks should have the same Nyquist frequency to define a consistent horizon distance function (got %s)" % ", ".join("%g" % rate for rate in sorted(nyquists)))
	# assume default 4 s PSD.  this is not required to be correct, but
	# for best accuracy it should not be larger than the true value and
	# for best performance it should not be smaller than the true
	# value.
	deltaF = 1. / 4.
	# use the minimum template id as the cannonical horizon function
	template_id, m1, m2, s1z, s2z = preferred_horizon_distance_template(banks)

	return template_id, reference_psd.HorizonDistance(15.0, 0.85 * max(nyquists), deltaF, m1, m2, spin1 = (0., 0., s1z), spin2 = (0., 0., s2z))

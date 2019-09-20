# Copyright (C) 2017,2018  Heather Fong
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


import h5py
import math
import numpy
import os
try:
	from scipy.interpolate import PPoly
except ImportError:
	# argh, scipy too old
	# FIXME:  delete this when we can rely on LDG sites having a
	# new-enough scipy
	from lal.rate import IrregularBins
	class PPoly(object):
		def __init__(self, c, x):
			self.intervals = IrregularBins(x)
			self.coeffs = c
			self.x0 = x

		def __call__(self, x):
			i = self.intervals[x]
			return numpy.poly1d(self.coeffs[:,i].squeeze())(x - self.x0[i]),


from gstlal import stats as gstlalstats


# FIXME:  caution, this information might get organized differently later.
# for now we just need to figure out where the gstlal-inspiral directory in
# share/ is.  don't write anything that assumes that this module will
# continue to define any of these symbols
from gstlal import paths as gstlal_config_paths


__all__ = [
	"UniformInTemplatePopulationModel",
	"SourcePopulationModel"
]


#
# =============================================================================
#
#                               Population Model
#
# =============================================================================
#


class UniformInTemplatePopulationModel(object):
	# FIXME:  this needs to learn the template chirp masses so it can
	# take out the factor of mchirp**(5./3.) factor on which the
	# horizon distance depends (which appears in a volume factor in the
	# numerator of the likelihood ratio)
	def __init__(self, template_ids):
		"""
		Assumes uniform in template population model, no
                astrophysical prior.
		"""
		self.lnP = 0.


	@gstlalstats.assert_ln_probability
	def lnP_template_signal(self, template_id, snr):
		assert snr >= 0.
		return self.lnP


class SourcePopulationModel(object):
	#
	# NOTE: This is no longer the default file used in the population
	# model.  Various tools including the dag have been modified to allow
	# the user to specify the mass model file at run time.  If you want to
	# use this file, you should point the dag script to wherever it is on
	# your filesystem.
	#
	# NOTE: future code will have this comment and these next two lines deleted.
	#
	#POPULATION_MODELS_PATH = os.path.join(gstlal_config_paths["pkgdatadir"], "population_models")
	#DEFAULT_FILENAME = os.path.join(POPULATION_MODELS_PATH, "O2/lnP_template_signal_BBH_logm_reweighted_mchirp.hdf5")


	def __init__(self, template_ids, filename = None):
		"""
		Sets the polynomial coefficients, given the template ID and
		SNR for a source population model, from which lnP is then
		computed using PPoly.
		"""
		if filename is not None:
			with h5py.File(filename, 'r') as model:
				coefficients = model['coefficients'].value
				snr_bp = model['SNR'].value
                                try:
                                        model_ids = model['template_id'].value
                                except KeyError:
                                        # FIXME: assume sequential order if model['event_id'] doesn't exist
                                        #model_ids = numpy.arange(numpy.shape(model['coefficients'].value)[-1])
                                        model_ids = model['event_id'].value
			# PPoly can construct an array of polynomials by just
			# feeding it the coefficients array all in one go, but then
			# it insists on evaluating all of them at once.  we don't
			# want to suffer that cost, so we have to make an array of
			# PPoly objects ourselves, and index into it to evaluate
			# just one.  since we have to do this anyway, we use a
			# dictionary to also solve the problem of mapping
			# template_id to a specific polynomial
                        template_indices = {}
                        for template_id in template_ids:
                                # maps template ID to the right coefficient array, since template IDs
                                # in the bank may not be in sequential order
                                try:
                                        template_indices[template_id] = numpy.where(model_ids==template_id)[0][0]
                                except IndexError:
                                        raise IndexError("template ID %d is not in this model" % template_id)
			self.polys = dict((template_id, PPoly(coefficients[:,:,[template_indices[template_id]]], snr_bp)) for template_id in template_ids)
			self.max_snr = snr_bp.max()
		else:
			self.polys = None
			self.max_snr = None

	@gstlalstats.assert_ln_probability
	def lnP_template_signal(self, template_id, snr):
		assert snr >= 0.
		try:
			lnP_vs_snr = self.polys[template_id]
		except KeyError:
			raise KeyError("template ID %d is not in this model" % template_id)
		# PPoly's .__call__() returns an array, so we need the
		# final [0] to flatten it
		return lnP_vs_snr(min(snr, self.max_snr))[0]

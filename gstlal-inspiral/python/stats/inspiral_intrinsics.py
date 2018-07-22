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
	DEFAULT_FILENAME = os.path.join(gstlal_config_paths["pkgdatadir"], "lnP_template_signal_BNS_gaussian_lowspin_Ozel.hdf5")


	def __init__(self, template_ids, filename = None):
		"""
		Sets the polynomial coefficients, given the template ID 
                and SNR for a source population model, from which
                lnP is then computed using PPoly.
		"""
		with h5py.File(filename if filename is not None else self.DEFAULT_FILENAME, 'r') as model:
			coefficients = model['coefficients'].value
			snr_bp = model['SNR'].value
		# PPoly can construct an array of polynomials by just
		# feeding it the coefficients array all in one go, but then
		# it insists on evaluating all of them at once.  we don't
		# want to suffer that cost, so we have to make an array of
		# PPoly objects ourselves, and index into it to evaluate
		# just one.  since we have to do this anyway, we use a
		# dictionary to also solve the problem of mapping
		# template_id to a specific polynomial
		self.polys = dict((template_id, PPoly(coefficients[:,:,[template_id]], snr_bp)) for template_id in template_ids)
		self.max_snr = snr_bp.max()


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
